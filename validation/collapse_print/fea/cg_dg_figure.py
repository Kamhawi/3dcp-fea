# Author: Abdallah Kamhawi <Kamhawi@umich.edu>

"""CG-vs-DG response comparison for the collapse-print cylinder (B&W + red).

Two panels of non-computational comparisons (the cost study lives in
``cost_figure.py``): (a) final per-layer settlement of the two
discretizations, (b) the DG-only inter-layer damage diagnostic.

Usage (after the canonical CG and DG runs):
    python -m validation.collapse_print.fea.cg_dg_figure [dg_run] [cg_run]
"""

from __future__ import annotations

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

COLOR_CG = "#000000"
COLOR_DG = "#CC2222"
COLOR_FAINT = "#BBBBBB"
LABEL_SIZE = 10
FONT_SIZE = 8
N_LAYERS = 11


def _latest_run(element: str) -> Path:
    runs = sorted(
        (CASE_DIR / "fea" / "output").glob(f"run_{element}_*"),
        key=lambda p: p.stat().st_mtime,
    )
    if not runs:
        raise FileNotFoundError(f"no {element} run found under fea/output/")
    return runs[-1]


def _final_uz(run_dir: Path) -> np.ndarray:
    rows = list(csv.DictReader(open(run_dir / "fea_uz.csv")))
    out = np.full(N_LAYERS, np.nan)
    for L in range(N_LAYERS):
        col = [r[f"uz_full_pct_L{L+1}"] for r in rows]
        col = [float(x) for x in col if x != ""]
        if col:
            out[L] = col[-1]
    return out


def _damage_trace(run_dir: Path):
    steps = json.load(open(run_dir / "results.json"))["steps"]
    t = np.array([s["time_s"] for s in steps])
    dmg = np.array([s.get("max_interface_damage", 0.0) for s in steps])
    return t, dmg


def main():
    dg_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else _latest_run("DG")
    cg_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else _latest_run("CG")
    uz_dg, uz_cg = _final_uz(dg_dir), _final_uz(cg_dir)
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

    # (a) per-layer final settlement, CG vs DG (full ring, since coverage)
    layers = np.arange(1, N_LAYERS + 1)
    fin = np.isfinite(uz_cg) & np.isfinite(uz_dg)
    ax_a.plot(layers[fin], uz_cg[fin], "o-", color=COLOR_CG, lw=1.1, ms=3.6,
              label="CG")
    ax_a.plot(layers[fin], uz_dg[fin], "s--", color=COLOR_DG, lw=1.1, ms=3.4,
              label="DG")
    ax_a.set_xlabel("Layer")
    ax_a.set_ylabel(r"Final settlement $u_z$ [% of $h_0$]")
    ax_a.set_xticks(layers[fin])
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
    pk_dg, pk_cg = int(np.nanargmax(uz_dg)) + 1, int(np.nanargmax(uz_cg)) + 1
    print(f"[PAPER] DG peak L{pk_dg} {np.nanmax(uz_dg):.1f}% | "
          f"CG peak L{pk_cg} {np.nanmax(uz_cg):.1f}% | max damage {dmg.max():.3f}")


if __name__ == "__main__":
    main()
