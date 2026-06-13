"""Settlement extraction from the hand-labeled collapse-print photographs.

The labeled photographs encode the visible layer bands with alternating
red/yellow fills.  This extractor reads those labels column-by-column, assigns
the bands to layer ids, and measures each layer's top-boundary drop from the
frame in which it is covered by the next layer to a later frame.  The final
per-layer medians provide the experimental points used in the cylinder
CG/DG/DGB comparison figure.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from matplotlib.colors import rgb_to_hsv
from PIL import Image


CASE_DIR = Path(__file__).resolve().parent
LABELED_DIR = CASE_DIR / "print_layers" / "labeled"
OUTPUT_DATA_DIR = CASE_DIR / "output" / "data"

N_LAYERS = 11
FRAME_PERIOD_S = 13.61
PITCH_CONVENTION_PX = 49.0
PX_OF_H0_FALLBACK = 28.441466845341736
PX_OF_H0_BAND_FALLBACK = (27.04394583256784, 29.99129485566932)


@dataclass(frozen=True)
class LabelParams:
    """Column parser thresholds for the hand-colored label bands."""

    min_run_px: int = 4
    merge_gap_px: int = 4
    seam_max_px: int = 6
    dark_guard_px: int = 4
    roi_y_min: int = 450
    roi_y_max: int = 1070
    sat_min: float = 0.6
    yellow_h: tuple[float, float] = (0.09, 0.18)
    yellow_v_min: float = 0.55
    red_h_max: float = 0.04
    red_h_min_hi: float = 0.93
    red_v_min: float = 0.30
    dark_v_max: float = 0.28


def _layer_color(layer_id: int) -> str:
    return "R" if layer_id % 2 == 1 else "Y"


def _color_masks(path: Path, params: LabelParams):
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    hsv = rgb_to_hsv(rgb)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    yellow = (
        (s >= params.sat_min)
        & (v >= params.yellow_v_min)
        & (h >= params.yellow_h[0])
        & (h <= params.yellow_h[1])
    )
    red = (
        (s >= params.sat_min)
        & (v >= params.red_v_min)
        & ((h <= params.red_h_max) | (h >= params.red_h_min_hi))
    )
    dark = v <= params.dark_v_max
    return red, yellow, dark


def _runs_from_mask(mask_col: np.ndarray, params: LabelParams):
    ys = np.flatnonzero(mask_col[params.roi_y_min : params.roi_y_max])
    if ys.size == 0:
        return []
    ys = ys + params.roi_y_min
    runs = []
    start = int(ys[0])
    prev = int(ys[0])
    for y_raw in ys[1:]:
        y = int(y_raw)
        if y - prev <= params.merge_gap_px + 1:
            prev = y
            continue
        if prev - start + 1 >= params.min_run_px:
            runs.append((start, prev))
        start = prev = y
    if prev - start + 1 >= params.min_run_px:
        runs.append((start, prev))
    return runs


def _assign_runs_to_layers(runs, frame_layer: int, params: LabelParams):
    """Assign detected color bands to physical layer ids.

    Runs are sorted from bottom to top.  We try every consecutive layer-id
    assignment whose colors match the red/yellow parity.  Top-anchored
    assignments are preferred when lower layers have been squashed out of the
    image; otherwise the unique color-consistent assignment is used.
    """
    if not runs:
        return {}, "no_stack"

    runs = sorted(runs, key=lambda item: 0.5 * (item[1] + item[2]), reverse=True)
    n_runs = len(runs)
    if n_runs > frame_layer:
        return {}, "too_many_runs"

    candidates = []
    for start_layer in range(1, frame_layer - n_runs + 2):
        ok = all(
            color == _layer_color(start_layer + i)
            for i, (color, _y0, _y1) in enumerate(runs)
        )
        if ok:
            candidates.append(start_layer)
    if not candidates:
        return {}, "bad_parity"

    top_start = frame_layer - n_runs + 1
    if top_start in candidates:
        start_layer = top_start
    elif 1 in candidates:
        start_layer = 1
    elif len(candidates) == 1:
        start_layer = candidates[0]
    else:
        return {}, "ambiguous"

    for lower, upper in zip(runs[:-1], runs[1:]):
        gap = int(lower[1]) - int(upper[2]) - 1
        if gap > params.seam_max_px:
            return {}, "seam_too_wide"

    return {
        start_layer + i: {
            "color": color,
            "top": int(y0),
            "bottom": int(y1),
        }
        for i, (color, y0, y1) in enumerate(runs)
    }, "ok"


def _frame_boundary_rows(path: Path, frame_layer: int, params: LabelParams):
    red, yellow, dark = _color_masks(path, params)
    height, width = red.shape
    rows = np.full((width, N_LAYERS), np.nan, dtype=np.float32)
    reasons = np.full((width, N_LAYERS), "no_stack", dtype=object)

    for x in range(width):
        runs = [("R", y0, y1) for y0, y1 in _runs_from_mask(red[:, x], params)]
        runs += [("Y", y0, y1) for y0, y1 in _runs_from_mask(yellow[:, x], params)]
        assigned, reason = _assign_runs_to_layers(runs, frame_layer, params)
        if reason != "ok":
            reasons[x, :frame_layer] = reason
            continue

        for layer_id, band in assigned.items():
            y_top = int(band["top"])
            y0 = max(0, y_top - params.dark_guard_px)
            y1 = min(height, y_top + params.dark_guard_px + 1)
            if np.any(dark[y0:y1, x]):
                reasons[x, layer_id - 1] = "dark_guard"
                continue
            rows[x, layer_id - 1] = y_top
            reasons[x, layer_id - 1] = "ok"
        for layer_id in range(1, frame_layer + 1):
            if layer_id not in assigned and reasons[x, layer_id - 1] == "no_stack":
                reasons[x, layer_id - 1] = "band_missing"

    return rows, reasons


def _frame_band_edges(path: Path, frame_layer: int, params: LabelParams):
    """Return top and bottom row coordinates for assigned painted bands."""
    red, yellow, dark = _color_masks(path, params)
    height, width = red.shape
    top = np.full((width, N_LAYERS), np.nan, dtype=np.float32)
    bottom = np.full((width, N_LAYERS), np.nan, dtype=np.float32)
    reasons = np.full((width, N_LAYERS), "no_stack", dtype=object)

    for x in range(width):
        runs = [("R", y0, y1) for y0, y1 in _runs_from_mask(red[:, x], params)]
        runs += [("Y", y0, y1) for y0, y1 in _runs_from_mask(yellow[:, x], params)]
        assigned, reason = _assign_runs_to_layers(runs, frame_layer, params)
        if reason != "ok":
            reasons[x, :frame_layer] = reason
            continue

        for layer_id, band in assigned.items():
            y_top = int(band["top"])
            y_bottom = int(band["bottom"])
            y0 = max(0, y_top - params.dark_guard_px)
            y1 = min(height, y_top + params.dark_guard_px + 1)
            if np.any(dark[y0:y1, x]):
                reasons[x, layer_id - 1] = "dark_guard"
                continue
            top[x, layer_id - 1] = y_top
            bottom[x, layer_id - 1] = y_bottom
            reasons[x, layer_id - 1] = "ok"
        for layer_id in range(1, frame_layer + 1):
            if layer_id not in assigned and reasons[x, layer_id - 1] == "no_stack":
                reasons[x, layer_id - 1] = "band_missing"

    return top, bottom, reasons


def _load_px_of_h0(output_dir: Path):
    checks_path = output_dir / "experiment_checks.json"
    if checks_path.exists():
        checks = json.loads(checks_path.read_text())
        frozen = checks.get("scale", {}).get("px_of_9mm_frozen", {})
        if frozen:
            central = float(frozen.get("central", PX_OF_H0_FALLBACK))
            lo = float(frozen.get("lo", PX_OF_H0_BAND_FALLBACK[0]))
            hi = float(frozen.get("hi", PX_OF_H0_BAND_FALLBACK[1]))
            return central, (lo, hi), frozen.get("basis", "experiment_checks.json")
    return (
        PX_OF_H0_FALLBACK,
        PX_OF_H0_BAND_FALLBACK,
        "fallback frozen wave-geometry scale",
    )


def build_experiment_dataset(
    labeled_dir: Path = LABELED_DIR,
    output_dir: Path = OUTPUT_DATA_DIR,
    params: LabelParams = LabelParams(),
):
    """Extract and write labeled-photo settlement data.

    Returns a dictionary with arrays and metadata.  Files written:
    ``experiment_uz.csv``, ``experiment_uz.json``, and ``experiment_uz.npz``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = [labeled_dir / f"layer_{i:02d}.jpg" for i in range(1, N_LAYERS + 1)]
    missing = [p for p in image_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing labeled photos: {missing}")

    frame_rows = []
    frame_bottom_rows = []
    frame_reasons = []
    for frame_layer, path in enumerate(image_paths, start=1):
        rows, bottom_rows, reasons = _frame_band_edges(path, frame_layer, params)
        frame_rows.append(rows)
        frame_bottom_rows.append(bottom_rows)
        frame_reasons.append(reasons)

    width = frame_rows[0].shape[0]
    boundary_row_px = np.stack(frame_rows, axis=0)
    material_edge_row_px = np.stack(frame_bottom_rows, axis=0)
    reason_text = np.stack(frame_reasons, axis=0)

    px_of_h0, px_of_h0_band, scale_basis = _load_px_of_h0(output_dir)
    scale_factor = PITCH_CONVENTION_PX / px_of_h0
    scale_factor_band = np.asarray(
        [PITCH_CONVENTION_PX / px_of_h0_band[1], PITCH_CONVENTION_PX / px_of_h0_band[0]],
        dtype=float,
    )

    uz_pitch = np.full((N_LAYERS, N_LAYERS), np.nan, dtype=float)
    uz_q25_pitch = np.full_like(uz_pitch, np.nan)
    uz_q75_pitch = np.full_like(uz_pitch, np.nan)
    uz_col_pitch = np.full((N_LAYERS, width, N_LAYERS), np.nan, dtype=np.float32)
    uz_material_pitch = np.full_like(uz_pitch, np.nan)
    uz_material_q25_pitch = np.full_like(uz_pitch, np.nan)
    uz_material_q75_pitch = np.full_like(uz_pitch, np.nan)
    uz_material_col_pitch = np.full_like(uz_col_pitch, np.nan)
    n_valid_final = np.zeros(N_LAYERS, dtype=int)
    n_valid_material_final = np.zeros(N_LAYERS, dtype=int)
    feasible = np.zeros(N_LAYERS, dtype=bool)
    material_feasible = np.zeros(N_LAYERS, dtype=bool)
    final_frame = N_LAYERS

    for layer_idx in range(N_LAYERS):
        coverage_frame = layer_idx + 2
        if coverage_frame > final_frame:
            continue
        coverage_rows = boundary_row_px[coverage_frame - 1, :, layer_idx]
        for frame in range(coverage_frame, final_frame + 1):
            current_rows = boundary_row_px[frame - 1, :, layer_idx]
            valid = np.isfinite(coverage_rows) & np.isfinite(current_rows)
            if not np.any(valid):
                continue
            delta_px = current_rows[valid] - coverage_rows[valid]
            vals = 100.0 * delta_px / PITCH_CONVENTION_PX
            uz_col_pitch[frame - 1, valid, layer_idx] = vals.astype(np.float32)
            uz_pitch[frame - 1, layer_idx] = float(np.median(vals))
            uz_q25_pitch[frame - 1, layer_idx] = float(np.percentile(vals, 25))
            uz_q75_pitch[frame - 1, layer_idx] = float(np.percentile(vals, 75))

        if layer_idx + 1 <= final_frame - 2:
            feasible[layer_idx] = True
            final_valid = np.isfinite(uz_col_pitch[final_frame - 1, :, layer_idx])
            n_valid_final[layer_idx] = int(np.sum(final_valid))

        # DG-consistent visible-material metric for the plot: track the lower
        # edge of the same painted bead from its deposition frame to the final
        # frame.  The top seam is frequently covered by the next bead; the lower
        # painted edge remains the more stable visible material marker.
        ref_frame = layer_idx + 1
        if ref_frame <= final_frame:
            ref_rows = material_edge_row_px[ref_frame - 1, :, layer_idx]
            for frame in range(ref_frame, final_frame + 1):
                current_rows = material_edge_row_px[frame - 1, :, layer_idx]
                valid = np.isfinite(ref_rows) & np.isfinite(current_rows)
                if not np.any(valid):
                    continue
                delta_px = current_rows[valid] - ref_rows[valid]
                vals = 100.0 * delta_px / PITCH_CONVENTION_PX
                uz_material_col_pitch[frame - 1, valid, layer_idx] = vals.astype(
                    np.float32
                )
                uz_material_pitch[frame - 1, layer_idx] = float(np.median(vals))
                uz_material_q25_pitch[frame - 1, layer_idx] = float(
                    np.percentile(vals, 25)
                )
                uz_material_q75_pitch[frame - 1, layer_idx] = float(
                    np.percentile(vals, 75)
                )

        # Layers 1-3 suffer startup/occlusion artifacts in the front photos;
        # layer 11 has no later final frame.  Layers 4-10 are the reliable
        # material-edge validation window.
        if 3 <= layer_idx <= 9:
            material_feasible[layer_idx] = True
            final_valid = np.isfinite(
                uz_material_col_pitch[final_frame - 1, :, layer_idx]
            )
            n_valid_material_final[layer_idx] = int(np.sum(final_valid))

    uz_h0 = uz_pitch * scale_factor
    uz_q25_h0 = uz_q25_pitch * scale_factor
    uz_q75_h0 = uz_q75_pitch * scale_factor
    uz_material_h0 = uz_material_pitch * scale_factor
    uz_material_q25_h0 = uz_material_q25_pitch * scale_factor
    uz_material_q75_h0 = uz_material_q75_pitch * scale_factor

    csv_path = output_dir / "experiment_uz.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "layer",
                "feasible",
                "coverage_frame",
                "final_frame",
                "n_valid_cols",
                "uz_final_med_pct_h0",
                "uz_final_q25_pct_h0",
                "uz_final_q75_pct_h0",
                "uz_final_med_pct_pitch49",
                "uz_final_q25_pct_pitch49",
                "uz_final_q75_pct_pitch49",
                "uz_final_med_px",
                "uz_material_edge_pct_pitch49",
                "uz_material_edge_q25_pct_pitch49",
                "uz_material_edge_q75_pct_pitch49",
                "material_edge_feasible",
                "material_edge_n_valid_cols",
                "pitch_px",
                "px_of_9mm",
                "n_excluded_cols",
            ]
        )
        final_idx = final_frame - 1
        for layer_idx in range(N_LAYERS):
            is_feasible = bool(feasible[layer_idx])
            coverage_frame = layer_idx + 2 if layer_idx + 2 <= final_frame else ""
            n_valid = int(n_valid_final[layer_idx]) if is_feasible else 0
            med_pitch = uz_pitch[final_idx, layer_idx]
            q25_pitch = uz_q25_pitch[final_idx, layer_idx]
            q75_pitch = uz_q75_pitch[final_idx, layer_idx]
            med_material = uz_material_pitch[final_idx, layer_idx]
            q25_material = uz_material_q25_pitch[final_idx, layer_idx]
            q75_material = uz_material_q75_pitch[final_idx, layer_idx]
            med_h0 = uz_h0[final_idx, layer_idx]
            q25_h0 = uz_q25_h0[final_idx, layer_idx]
            q75_h0 = uz_q75_h0[final_idx, layer_idx]
            med_px = med_pitch * PITCH_CONVENTION_PX / 100.0
            writer.writerow(
                [
                    layer_idx + 1,
                    int(is_feasible),
                    coverage_frame if is_feasible else "",
                    final_frame if is_feasible else "",
                    n_valid,
                    f"{med_h0:.2f}" if np.isfinite(med_h0) and is_feasible else "",
                    f"{q25_h0:.2f}" if np.isfinite(q25_h0) and is_feasible else "",
                    f"{q75_h0:.2f}" if np.isfinite(q75_h0) and is_feasible else "",
                    f"{med_pitch:.2f}" if np.isfinite(med_pitch) and is_feasible else "",
                    f"{q25_pitch:.2f}" if np.isfinite(q25_pitch) and is_feasible else "",
                    f"{q75_pitch:.2f}" if np.isfinite(q75_pitch) and is_feasible else "",
                    f"{med_px:.2f}" if np.isfinite(med_px) and is_feasible else "",
                    f"{med_material:.2f}" if np.isfinite(med_material) else "",
                    f"{q25_material:.2f}" if np.isfinite(q25_material) else "",
                    f"{q75_material:.2f}" if np.isfinite(q75_material) else "",
                    int(bool(material_feasible[layer_idx])),
                    int(n_valid_material_final[layer_idx]),
                    f"{PITCH_CONVENTION_PX:.2f}",
                    f"{px_of_h0:.2f}",
                    width - n_valid,
                ]
            )

    reason_codes = {name: i for i, name in enumerate(sorted(set(reason_text.reshape(-1))))}
    reason_arr = np.vectorize(reason_codes.get)(reason_text).astype(np.uint8)
    npz_path = output_dir / "experiment_uz.npz"
    np.savez(
        npz_path,
        uz_layer_pct=uz_h0,
        uz_q25_pct=uz_q25_h0,
        uz_q75_pct=uz_q75_h0,
        uz_col_pct=uz_col_pitch * scale_factor,
        uz_layer_pct_pitch49=uz_pitch,
        uz_q25_pct_pitch49=uz_q25_pitch,
        uz_q75_pct_pitch49=uz_q75_pitch,
        uz_material_edge_pct=uz_material_h0,
        uz_material_edge_q25_pct=uz_material_q25_h0,
        uz_material_edge_q75_pct=uz_material_q75_h0,
        uz_material_edge_col_pct=uz_material_col_pitch * scale_factor,
        uz_material_edge_pct_pitch49=uz_material_pitch,
        uz_material_edge_q25_pct_pitch49=uz_material_q25_pitch,
        uz_material_edge_q75_pct_pitch49=uz_material_q75_pitch,
        scale_factor=np.asarray(scale_factor),
        scale_factor_band=scale_factor_band,
        px_of_9mm=np.asarray(px_of_h0),
        px_of_9mm_band=np.asarray(px_of_h0_band),
        boundary_row_px=boundary_row_px,
        material_edge_row_px=material_edge_row_px,
        boundary_reason=reason_arr,
        valid_col=np.isfinite(uz_col_pitch[final_frame - 1]),
        material_edge_valid_col=np.isfinite(uz_material_col_pitch[final_frame - 1]),
        n_valid=n_valid_final,
        material_edge_n_valid=n_valid_material_final,
        feasible=feasible,
        material_edge_feasible=material_feasible,
        pitch_px=np.asarray(PITCH_CONVENTION_PX),
        snapshot_t_s=np.arange(1, N_LAYERS + 1, dtype=float) * FRAME_PERIOD_S,
        snapshot_layer=np.arange(1, N_LAYERS + 1, dtype=int),
        gate_passed=np.asarray(
            abs(float(uz_pitch[final_frame - 1, 3]) - 86.0) <= 1.0
        ),
        boundary_def=np.asarray("band_top"),
        pitch_mode=np.asarray("global_pitch49"),
    )

    per_layer = {}
    final_idx = final_frame - 1
    for layer_idx in range(N_LAYERS):
        counts = Counter(reason_text[final_idx, :, layer_idx])
        per_layer[f"L{layer_idx + 1}"] = {
            "feasible": bool(feasible[layer_idx]),
            "n_valid_cols": int(n_valid_final[layer_idx]),
            "uz_final_med_pct_pitch49": (
                float(uz_pitch[final_idx, layer_idx])
                if np.isfinite(uz_pitch[final_idx, layer_idx])
                else None
            ),
            "uz_final_q25_pct_pitch49": (
                float(uz_q25_pitch[final_idx, layer_idx])
                if np.isfinite(uz_q25_pitch[final_idx, layer_idx])
                else None
            ),
            "uz_final_q75_pct_pitch49": (
                float(uz_q75_pitch[final_idx, layer_idx])
                if np.isfinite(uz_q75_pitch[final_idx, layer_idx])
                else None
            ),
            "uz_final_med_pct_h0": (
                float(uz_h0[final_idx, layer_idx])
                if np.isfinite(uz_h0[final_idx, layer_idx])
                else None
            ),
            "uz_material_edge_pct_pitch49": (
                float(uz_material_pitch[final_idx, layer_idx])
                if np.isfinite(uz_material_pitch[final_idx, layer_idx])
                else None
            ),
            "material_edge_feasible": bool(material_feasible[layer_idx]),
            "material_edge_n_valid_cols": int(n_valid_material_final[layer_idx]),
            "exclusions": {k: int(v) for k, v in counts.items() if k != "ok"},
        }

    json_path = output_dir / "experiment_uz.json"
    payload = {
        "case": "collapse_print_hand_label_settlement",
        "method": (
            "Per image column, the hand-labeled top boundary of each printed "
            "layer is read at integer-pixel resolution.  Layer j's settlement "
            "is the boundary drop from its coverage frame (j+1) to each later "
            "frame, aggregated as the median and interquartile range over "
            "valid columns."
        ),
        "boundary_def": "band_top",
        "primary_plot_metric": "material_edge_bottom_from_deposition",
        "primary_plot_normalization": "global_pitch49",
        "pitch_px": PITCH_CONVENTION_PX,
        "frame_period_s": FRAME_PERIOD_S,
        "renormalization": {
            "note": (
                "The pitch49 fields preserve the image-layer-pitch convention "
                "used for the layer-4 86% anchor.  The *_pct_h0 fields rescale "
                "the same pixel drops by the frozen px(9 mm) image scale."
            ),
            "px_of_9mm": px_of_h0,
            "px_of_9mm_band": list(px_of_h0_band),
            "scale_basis": scale_basis,
            "factor": scale_factor,
            "factor_band": scale_factor_band.tolist(),
        },
        "params": params.__dict__,
        "gate": {
            "target_pct_pitch49": 86.0,
            "tol_pct": 1.0,
            "L4_final_pct_pitch49": float(uz_pitch[final_idx, 3]),
            "passed": bool(abs(float(uz_pitch[final_idx, 3]) - 86.0) <= 1.0),
        },
        "infeasible_layers": {
            "L1-L3": (
                "not used in the material-edge plot because startup bands are "
                "partly occluded or compressed below the stable-label threshold"
            ),
            "L11": "never covered by a subsequent layer",
        },
        "reason_codes": reason_codes,
        "per_layer": per_layer,
    }
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n")

    return {
        "csv": csv_path,
        "json": json_path,
        "npz": npz_path,
        "uz_pitch49": uz_pitch,
        "uz_h0": uz_h0,
        "feasible": feasible,
        "n_valid": n_valid_final,
    }


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Extract settlement from hand-labeled collapse-print photos."
    )
    parser.add_argument(
        "--labeled-dir",
        type=Path,
        default=LABELED_DIR,
        help="Directory containing layer_XX.jpg labeled photos.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DATA_DIR,
        help="Directory for experiment_uz CSV/JSON/NPZ outputs.",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None):
    args = _build_arg_parser().parse_args(list(argv) if argv is not None else None)
    result = build_experiment_dataset(args.labeled_dir, args.output_dir)
    print(f"wrote {result['csv']}")
    print(f"wrote {result['json']}")
    print(f"wrote {result['npz']}")


if __name__ == "__main__":
    main()
