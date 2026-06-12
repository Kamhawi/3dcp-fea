# Author: Abdallah Kamhawi <Kamhawi@umich.edu>

"""Qualitative simulation-progress figure: photos vs CG and DG fields.

Three rows x six frames: the print photographs (top), the CG simulation
(middle), and the DG simulation (bottom), each rendered as the deformed front
view colored by the displacement magnitude |u| [mm] at the same six instants
on a shared Cool-to-Warm scale.

Usage (after the canonical CG and DG runs):
    python -m validation.collapse_print.fea.simulation_progress \
        [dg_run_dir] [cg_run_dir]
"""

from __future__ import annotations

import sys
from pathlib import Path

import adios2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from PIL import Image

from validation.collapse_print.fea.setup import build_validation_state

FRAMES = (4, 6, 8, 9, 10, 11)        # photo k = print state after layer k
LAYER_PERIOD_S = 13.61               # measured layer cycle time
CASE_DIR = Path(__file__).resolve().parents[1]
PHOTO_DIR = CASE_DIR / "print_layers" / "original"
FIG_DIR = CASE_DIR / "output" / "figures"
# photo crop around the print (pixels): x-range, then y bottom/top (image rows)
CROP = dict(cx0=562, cx1=1361, cy0=1067, cy1=485)
CMAP = plt.cm.coolwarm
FONT_SIZE = 8


def _latest_run(element: str) -> Path:
    runs = sorted(
        (CASE_DIR / "fea" / "output").glob(f"run_{element}_*"),
        key=lambda p: p.stat().st_mtime,
    )
    if not runs:
        raise FileNotFoundError(f"no {element} run found under fea/output/")
    return runs[-1]


def _read_displacements(run_dir: Path, target_times):
    """u (n_blocks, 3) at the stored step nearest each target time."""
    import csv

    with open(run_dir / "step_metrics.csv") as f:
        times = np.array([float(r["time_s"]) for r in csv.DictReader(f)])
    steps = [int(np.argmin(np.abs(times - t))) for t in target_times]

    out = {}
    with adios2.FileReader(str(run_dir / "disp.bp")) as f:
        for s in sorted(set(steps)):
            u = f.read("displacement", step_selection=[s, 1])
            out[s] = np.asarray(u, float).reshape(-1, 3)
    return steps, [times[s] for s in steps], out


def _assert_vtx_order(run_dir: Path, element: str):
    """The VTX dof order must match the function-space dof order."""
    ck = CASE_DIR / "fea" / f"checkpoints_{element}" / "checkpoint_latest.npz"
    if not ck.exists():
        return
    u_ck = np.load(ck)["u"].reshape(-1, 3)
    with adios2.FileReader(str(run_dir / "disp.bp")) as f:
        n_steps = int(f.num_steps())
        u_last = np.asarray(
            f.read("displacement", step_selection=[n_steps - 1, 1]), float
        ).reshape(-1, 3)
    err = np.max(np.abs(u_ck - u_last[: len(u_ck)]))
    assert err < 1e-10, f"{element} VTX/dof order mismatch (max diff {err:.2e})"


def _outer_quads(state, u_blocks, t_val, warp=1.0):
    """Deformed outer-wall quads + |u| color values for active cells."""
    msh = state.msh
    nuo = msh.topology.index_map(msh.topology.dim).size_local
    dof_xyz = state.V.tabulate_dof_coordinates()
    bs = state.V.dofmap.bs

    quads, uz_vals, depth = [], [], []
    for c in range(nuo):
        if state.birth_times_dolfinx[c] > t_val:
            continue
        blocks = (state.cell_to_dofs[c][0::3] // bs).astype(int)   # 8 dof blocks
        X = dof_xyz[blocks]
        U = u_blocks[blocks]
        r = np.hypot(X[:, 0], X[:, 1])
        outer = np.argsort(r)[-4:]                                  # outer wall
        Xo, Uo = X[outer], U[outer]
        th = np.arctan2(Xo[:, 1], Xo[:, 0])
        if th.max() - th.min() > np.pi:                             # seam wrap
            th = np.where(th < 0, th + 2 * np.pi, th)
        z = Xo[:, 2]
        lo = np.argsort(z)[:2]; hi = np.argsort(z)[2:]
        lo = lo[np.argsort(th[lo])]; hi = hi[np.argsort(th[hi])[::-1]]
        order = np.concatenate([lo, hi])
        P = Xo[order] + warp * Uo[order]
        quads.append(P[:, [0, 2]])                                  # (x, z) view
        uz_vals.append(float(np.linalg.norm(Uo, axis=1).mean()))    # |u|
        depth.append(float((Xo[:, 1] + Uo[:, 1]).mean()))
    return np.array(quads), np.array(uz_vals), np.array(depth)


def _draw_field(ax, state, u_blocks, t_val, vmax_mm, zmax):
    quads, mag, depth = _outer_quads(state, u_blocks, t_val)
    order = np.argsort(-depth)                       # painter: far first
    colors = CMAP(np.clip(mag / vmax_mm, 0, 1))
    ax.add_collection(PolyCollection(
        quads[order], facecolors=colors[order], edgecolors="none"))
    ax.set_xlim(-130, 130)
    ax.set_ylim(-0.30 * zmax, 1.10 * zmax)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


def main():
    dg_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else _latest_run("DG")
    cg_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else _latest_run("CG")
    target_times = [k * LAYER_PERIOD_S for k in FRAMES]

    states, steps, fea_times, u_by = {}, {}, {}, {}
    for el, run in (("DG", dg_dir), ("CG", cg_dir)):
        _assert_vtx_order(run, el)
        states[el] = build_validation_state(element=el)
        steps[el], fea_times[el], u_by[el] = _read_displacements(run, target_times)

    zmax = float(states["DG"].cylinder_mesh.sheet_z.max())

    # shared color scale [mm] from the DG final frame (the larger of the two)
    _, mag_fin, _ = _outer_quads(
        states["DG"], u_by["DG"][steps["DG"][-1]], fea_times["DG"][-1]
    )
    vmax_mm = float(np.ceil(mag_fin.max()))

    cx0, cx1, cy0, cy1 = CROP["cx0"], CROP["cx1"], CROP["cy0"], CROP["cy1"]
    aspect = (cx1 - cx0) / (cy0 - cy1)

    n_rows = 3
    left, hgap, panel_w = 0.030, 0.010, 0.150
    panel_h_in = 8.5 * panel_w / aspect
    vgap_in, bot_in, top_in = 0.10, 0.05, 0.05
    fig_h = n_rows * panel_h_in + (n_rows - 1) * vgap_in + bot_in + top_in
    fig = plt.figure(figsize=(8.5, fig_h))
    plt.rcParams.update({k: FONT_SIZE for k in (
        "font.size", "axes.titlesize", "axes.labelsize",
        "xtick.labelsize", "ytick.labelsize")})
    panel_h = panel_h_in / fig_h
    vgap, bot = vgap_in / fig_h, bot_in / fig_h
    row_y = {0: bot + 2 * (panel_h + vgap),        # photos
             1: bot + 1 * (panel_h + vgap),        # CG
             2: bot}                               # DG

    for i, k in enumerate(FRAMES):
        x0 = left + i * (panel_w + hgap)
        # row 0: photograph
        ax_p = fig.add_axes([x0, row_y[0], panel_w, panel_h])
        org = np.asarray(Image.open(PHOTO_DIR / f"layer_{k:02d}.jpg"))
        ax_p.imshow(org[..., :3])
        ax_p.set_xlim(cx0, cx1); ax_p.set_ylim(cy0, cy1)
        ax_p.set_xticks([]); ax_p.set_yticks([]); ax_p.grid(False)
        for s in ax_p.spines.values():
            s.set_visible(False)
        ax_p.text(0.03, 0.96, rf"$t$={k*LAYER_PERIOD_S:.0f} s",
                  transform=ax_p.transAxes, ha="left", va="top",
                  fontsize=FONT_SIZE, fontstyle="italic")

        # rows 1-2: CG then DG fields at the same instant
        for r, el in ((1, "CG"), (2, "DG")):
            ax_f = fig.add_axes([x0, row_y[r], panel_w, panel_h])
            _draw_field(ax_f, states[el], u_by[el][steps[el][i]],
                        fea_times[el][i], vmax_mm, zmax)

    for r, lab in ((0, "print"), (1, "CG"), (2, "DG")):
        fig.text(0.012, row_y[r] + panel_h / 2, lab, rotation=90,
                 ha="center", va="center", fontsize=FONT_SIZE)

    cax = fig.add_axes([0.973, bot, 0.008, 2 * panel_h + vgap])
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0, vmax_mm))
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(r"displacement magnitude $|\mathbf{u}|$  [mm]",
                 fontsize=FONT_SIZE)
    cb.ax.tick_params(labelsize=FONT_SIZE - 1)
    cb.outline.set_linewidth(0.6)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths = [FIG_DIR / f"simulation_progress.{e}" for e in ("png", "pdf")]
    for p in paths:
        fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    for p in paths:
        print(f"wrote {p}")
    print(f"color scale: 0 to {vmax_mm:.0f} mm (|u|)")


if __name__ == "__main__":
    main()
