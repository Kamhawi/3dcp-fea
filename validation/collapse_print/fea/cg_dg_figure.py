# Author: Abdallah Kamhawi <Kamhawi@umich.edu>

"""CG-vs-DG response comparison for the collapse-print cylinder (B&W + red).

Two panels of non-computational comparisons (the cost study lives in
``cost_figure.py``): (a) final per-layer settlement of CG, bonded DG,
cohesive DG, and the hand-labeled photo measurements, (b) the DG-only
inter-layer damage diagnostic.

Usage (after the canonical CG, DG, and bonded-DG runs):
    python -m validation.collapse_print.fea.cg_dg_figure \
        [dg_run] [cg_run] [dgb_run]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[3]))

try:
    from verification.plotting import new_figure, save_figure
except ImportError:
    def new_figure(height, width=8.5):
        return plt.figure(figsize=(width, height))

    def save_figure(fig, *paths):
        for path in paths:
            fig.savefig(path, dpi=300, bbox_inches="tight")

CASE_DIR = Path(__file__).resolve().parents[1]
FIG_DIR = CASE_DIR / "output" / "figures"
DATA_DIR = CASE_DIR / "output" / "data"

COLOR_CG = "#000000"
COLOR_DG = "#CC2222"
COLOR_DGB = "#777777"
COLOR_FAINT = "#BBBBBB"
LABEL_SIZE = 10
FONT_SIZE = 8
N_LAYERS = 11


def _latest_run(element: str) -> Path:
    pattern = "run_DGB_*" if element == "DGB" else f"run_{element}_*"
    runs = sorted(
        (CASE_DIR / "fea" / "output").glob(pattern),
        key=lambda p: p.stat().st_mtime,
    )
    if not runs:
        raise FileNotFoundError(f"no {element} run found under fea/output/")
    return runs[-1]


def _final_uz(run_dir: Path, metric: str) -> np.ndarray:
    rows = list(csv.DictReader(open(run_dir / "fea_uz.csv")))
    out = np.full(N_LAYERS, np.nan)
    prefix = f"uz_{metric}_pct_L"
    for L in range(N_LAYERS):
        col = [r[f"{prefix}{L+1}"] for r in rows]
        col = [float(x) for x in col if x != ""]
        if col:
            out[L] = col[-1]
    return out


def _damage_trace(run_dir: Path):
    steps = json.load(open(run_dir / "results.json"))["steps"]
    t = np.array([s["time_s"] for s in steps])
    dmg = np.array([s.get("max_interface_damage", 0.0) for s in steps])
    return t, dmg


def _experiment_final(normalization: str, photo_metric: str):
    npz_path = DATA_DIR / "experiment_uz.npz"
    if not npz_path.exists():
        from validation.collapse_print.experiment_settlement import (
            build_experiment_dataset,
        )

        build_experiment_dataset()

    data = np.load(npz_path, allow_pickle=True)
    if photo_metric == "material_edge":
        feasible_key = "material_edge_feasible"
        keys = (
            ("uz_material_edge_pct", "uz_material_edge_q25_pct", "uz_material_edge_q75_pct")
            if normalization == "h0"
            else (
                "uz_material_edge_pct_pitch49",
                "uz_material_edge_q25_pct_pitch49",
                "uz_material_edge_q75_pct_pitch49",
            )
        )
    else:
        feasible_key = "feasible"
        keys = (
            ("uz_layer_pct", "uz_q25_pct", "uz_q75_pct")
            if normalization == "h0"
            else ("uz_layer_pct_pitch49", "uz_q25_pct_pitch49", "uz_q75_pct_pitch49")
        )

    med = np.asarray(data[keys[0]][-1], dtype=float)
    q25 = np.asarray(data[keys[1]][-1], dtype=float)
    q75 = np.asarray(data[keys[2]][-1], dtype=float)
    feasible = np.asarray(data[feasible_key], dtype=bool)
    finite = np.isfinite(med)
    primary = feasible & finite
    if photo_metric == "material_edge" and "material_edge_row_px" in data.files:
        startup_med, startup_q25, startup_q75 = _startup_material_edge(data, normalization)
        startup = np.isfinite(startup_med) & ~primary
        med[startup] = startup_med[startup]
        q25[startup] = startup_q25[startup]
        q75[startup] = startup_q75[startup]
        finite = np.isfinite(med)
    return med, q25, q75, primary, finite


def _startup_material_edge(data, normalization: str):
    """Conservative L1-L3 display metric from coverage to final.

    The main material-edge dataset is referenced to deposition, which is stable
    for the mid/late layers.  The startup layers are partly occluded while the
    first few beads are still being painted, so their gray display markers use
    the post-coverage window instead.
    """
    rows = np.asarray(data["material_edge_row_px"], dtype=float)
    n_layers = rows.shape[2]
    med = np.full(n_layers, np.nan)
    q25 = np.full(n_layers, np.nan)
    q75 = np.full(n_layers, np.nan)
    final = rows[-1]
    pitch_px = float(np.asarray(data["pitch_px"]))
    factor = float(np.asarray(data["scale_factor"])) if normalization == "h0" else 1.0
    for layer_idx in range(min(3, n_layers)):
        coverage_frame = layer_idx + 2
        ref = rows[coverage_frame - 1, :, layer_idx]
        cur = final[:, layer_idx]
        ok = np.isfinite(ref) & np.isfinite(cur)
        if not np.any(ok):
            continue
        vals = 100.0 * (cur[ok] - ref[ok]) / pitch_px * factor
        med[layer_idx] = float(np.median(vals))
        q25[layer_idx] = float(np.percentile(vals, 25))
        q75[layer_idx] = float(np.percentile(vals, 75))
    return med, q25, q75


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Create the collapse-print CG/DG/DGB comparison figure."
    )
    parser.add_argument("dg_run", nargs="?", type=Path, default=None)
    parser.add_argument("cg_run", nargs="?", type=Path, default=None)
    parser.add_argument("dgb_run", nargs="?", type=Path, default=None)
    parser.add_argument(
        "--metric",
        choices=["front", "full"],
        default="full",
        help=(
            "Simulation settlement metric. The default full-ring value is the "
            "original panel-a metric; front arc emulates the camera-facing view."
        ),
    )
    parser.add_argument(
        "--experiment-normalization",
        choices=["pitch49", "h0"],
        default="pitch49",
        help=(
            "Photo-label normalization. pitch49 preserves the image layer-pitch "
            "convention used for the layer-4 86% anchor; h0 applies the frozen "
            "px(9 mm) rescale."
        ),
    )
    parser.add_argument(
        "--photo-metric",
        choices=["material_edge", "top_seam"],
        default="material_edge",
        help=(
            "Experimental label metric. material_edge tracks the lower visible "
            "edge of each painted bead from deposition to final; top_seam "
            "reproduces the old coverage-to-final seam measurement."
        ),
    )
    return parser


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)
    dg_dir = args.dg_run or _latest_run("DG")
    cg_dir = args.cg_run or _latest_run("CG")
    dgb_dir = args.dgb_run or _latest_run("DGB")

    uz_dg = _final_uz(dg_dir, args.metric)
    uz_cg = _final_uz(cg_dir, args.metric)
    uz_dgb = _final_uz(dgb_dir, args.metric)
    exp_med, exp_q25, exp_q75, exp_ok, exp_finite = _experiment_final(
        args.experiment_normalization,
        args.photo_metric,
    )
    t_dmg, dmg = _damage_trace(dg_dir)

    fig = new_figure(2.6, width=8.5)
    plt.rcParams.update({k: FONT_SIZE for k in (
        "font.size", "axes.titlesize", "axes.labelsize", "xtick.labelsize",
        "ytick.labelsize", "legend.fontsize")})

    ax_a = fig.add_axes([0.065, 0.165, 0.40, 0.71])
    ax_b = fig.add_axes([0.580, 0.165, 0.40, 0.71])
    fig.text(0.012, 0.95, "a", fontsize=LABEL_SIZE, fontweight="bold",
             ha="left", va="center")
    fig.text(0.525, 0.95, "b", fontsize=LABEL_SIZE, fontweight="bold",
             ha="left", va="center")

    # (a) per-layer final settlement.  The photo-label series uses the visible
    # material edge, not the occlusion-prone top seam.
    layers = np.arange(1, N_LAYERS + 1)
    fin = np.isfinite(uz_cg) & np.isfinite(uz_dg) & np.isfinite(uz_dgb)
    ax_a.plot(layers[fin], uz_cg[fin], "o-", color=COLOR_CG, lw=1.1, ms=3.6,
              label="CG")
    ax_a.plot(layers[fin], uz_dgb[fin], "^-.", color=COLOR_DGB, lw=1.1, ms=3.5,
              label="bonded DG")
    ax_a.plot(layers[fin], uz_dg[fin], "s--", color=COLOR_DG, lw=1.1, ms=3.4,
              label="DG")
    exp_layers = layers[exp_ok]
    exp_y = exp_med[exp_ok]
    yerr = np.vstack((exp_y - exp_q25[exp_ok], exp_q75[exp_ok] - exp_y))
    low_conf = exp_finite & ~exp_ok
    if np.any(low_conf):
        low_y = exp_med[low_conf]
        low_err = np.vstack(
            (low_y - exp_q25[low_conf], exp_q75[low_conf] - low_y)
        )
        ax_a.errorbar(
            layers[low_conf],
            low_y,
            yerr=low_err,
            fmt="D",
            color=COLOR_CG,
            mfc="white",
            mec=COLOR_CG,
            ecolor=COLOR_CG,
            elinewidth=0.8,
            capsize=2.0,
            ms=3.7,
            label="_nolegend_",
            zorder=4,
        )
    ax_a.errorbar(
        exp_layers,
        exp_y,
        yerr=yerr,
        fmt="D",
        color=COLOR_CG,
        mfc="white",
        mec=COLOR_CG,
        ecolor=COLOR_CG,
        elinewidth=0.8,
        capsize=2.0,
        ms=3.7,
        label="photo labels",
        zorder=5,
    )
    ax_a.set_xlabel("Layer")
    ax_a.set_ylabel(r"Final settlement $u_z$ [% of $h_0$]")
    ax_a.set_xticks(layers)
    ax_a.set_ylim(0, None)
    ax_a.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=2,
                frameon=False, handlelength=1.6, borderaxespad=0.0,
                columnspacing=1.0)

    # (b) DG-only diagnostic: max inter-layer damage history
    ax_b.plot(t_dmg, dmg, "-", color=COLOR_DG, lw=1.3)
    ax_b.axhline(1.0, color=COLOR_FAINT, lw=0.8, ls="--")
    ax_b.text(t_dmg[-1], 1.0, "full damage", ha="right", va="bottom",
              fontsize=FONT_SIZE - 1, color="#777777")
    ax_b.annotate(f"{dmg.max():.2f}", (t_dmg[-1], dmg[-1]),
                  textcoords="offset points", xytext=(-2, 5),
                  ha="right", fontsize=FONT_SIZE - 1, color=COLOR_DG)
    ax_b.set_ylim(0, 1.1)
    ax_b.set_xlabel("Time [s]")
    ax_b.set_ylabel("Max interface damage")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths = [FIG_DIR / f"cg_dg_comparison.{e}" for e in ("png", "pdf")]
    save_figure(fig, *paths)
    plt.close(fig)
    for p in paths:
        print(f"wrote {p}")
    pk_dg = int(np.nanargmax(uz_dg)) + 1
    pk_cg = int(np.nanargmax(uz_cg)) + 1
    pk_dgb = int(np.nanargmax(uz_dgb)) + 1
    cohesive_excess = uz_dg - uz_dgb
    residual_discretization = uz_dgb - uz_cg
    print(
        f"[PAPER] metric={args.metric}, photo={args.photo_metric}, "
        f"experiment={args.experiment_normalization} | "
        f"DG peak L{pk_dg} {np.nanmax(uz_dg):.1f}% | "
        f"bonded DG peak L{pk_dgb} {np.nanmax(uz_dgb):.1f}% | "
        f"CG peak L{pk_cg} {np.nanmax(uz_cg):.1f}% | "
        f"photo L4 {exp_med[3]:.1f}% | max damage {dmg.max():.3f}"
    )
    print(
        f"[PAPER] gap decomposition ({args.metric}): "
        f"max(DG-bondedDG)={np.nanmax(cohesive_excess):.1f}% of pitch; "
        f"max|bondedDG-CG|={np.nanmax(np.abs(residual_discretization)):.1f}%"
    )


if __name__ == "__main__":
    main()
