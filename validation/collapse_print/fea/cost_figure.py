# Author: Abdallah Kamhawi <Kamhawi@umich.edu>

"""Computational-cost figure for the cylinder CG, bonded-DG, and DG runs.

Mirrors the barrel-vault cost figure (three letter-width panels) with the two
discretizations and the bonded-DG control overlaid: (a) wall-clock per step
and cumulative, (b) active displacement DOFs and active cells, (c) Newton
iterations per step. The runs use the serial direct solver, so each Newton
iteration costs exactly one linear solve. Prints a [PAPER] block with the
timing decomposition.

Usage (after the canonical CG, DG, and bonded-DG runs):
    python -m validation.collapse_print.fea.cost_figure [dg_run] [cg_run] [dgb_run]
"""

from __future__ import annotations

import argparse
import csv
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
COLOR_DGB = "#777777"
COLOR_REF = "#777777"
LABEL_SIZE = 10
FONT_SIZE = 8


def _latest_run(element: str) -> Path:
    pattern = "run_DGB_*" if element == "DGB" else f"run_{element}_*"
    runs = sorted(
        (CASE_DIR / "fea" / "output").glob(pattern),
        key=lambda p: p.stat().st_mtime,
    )
    if not runs:
        raise FileNotFoundError(f"no {element} run found under fea/output/")
    return runs[-1]


def _metrics(run_dir: Path) -> dict:
    rows = list(csv.DictReader(open(run_dir / "step_metrics.csv")))
    get = lambda k, cast=float: np.array([cast(r[k]) for r in rows])
    return {
        "step": np.arange(1, len(rows) + 1),
        "step_wall": get("time_step_total_s"),
        "cumul_wall": get("cumul_wall_s"),
        "active_dofs": get("active_dofs", lambda v: int(v) if v else 0),
        "active_cells": get("active_cells", int),
        "newton": get("newton_iters", int),
        "t_newton": get("time_newton_s"),
        "t_perzyna": get("time_perzyna_s"),
        "t_proj": get("time_proj_s"),
        "t_io": get("time_io_s"),
    }


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Create the collapse-print computational-cost figure."
    )
    parser.add_argument("dg_run", nargs="?", type=Path, default=None)
    parser.add_argument("cg_run", nargs="?", type=Path, default=None)
    parser.add_argument("dgb_run", nargs="?", type=Path, default=None)
    return parser


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)
    dg_dir = args.dg_run or _latest_run("DG")
    cg_dir = args.cg_run or _latest_run("CG")
    dgb_dir = args.dgb_run or _latest_run("DGB")
    dg, cg, dgb = _metrics(dg_dir), _metrics(cg_dir), _metrics(dgb_dir)

    fig = new_figure(2.4, width=8.5)
    plt.rcParams.update({k: FONT_SIZE for k in (
        "font.size", "axes.titlesize", "axes.labelsize", "xtick.labelsize",
        "ytick.labelsize", "legend.fontsize")})

    ax_a = fig.add_axes([0.060, 0.19, 0.205, 0.70])
    ax_b = fig.add_axes([0.395, 0.19, 0.205, 0.70])
    ax_c = fig.add_axes([0.730, 0.19, 0.205, 0.70])
    for x, lab in ((0.012, "a"), (0.347, "b"), (0.682, "c")):
        fig.text(x, 0.945, lab, fontsize=LABEL_SIZE, fontweight="bold",
                 ha="left", va="center")
    x1 = dg["step"][-1]

    # (a) wall-clock per step (log) + cumulative (right axis)
    ax_a.plot(cg["step"], cg["step_wall"], color=COLOR_CG, lw=0.9, label="CG")
    ax_a.plot(dgb["step"], dgb["step_wall"], color=COLOR_DGB, lw=0.9,
              label="bonded DG")
    ax_a.plot(dg["step"], dg["step_wall"], color=COLOR_DG, lw=0.9, label="DG")
    ax_a.set_yscale("log")
    ax_a.set_xlabel("Time step")
    ax_a.set_ylabel("Wall-clock / step [s]")
    ax_a.set_xlim(1, x1)
    ax_a.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=3,
                frameon=False, handlelength=1.4, borderaxespad=0.0,
                columnspacing=0.8)
    ax_aR = ax_a.twinx()
    ax_aR.plot(cg["step"], cg["cumul_wall"], "--", color=COLOR_CG, lw=1.1)
    ax_aR.plot(dgb["step"], dgb["cumul_wall"], "--", color=COLOR_DGB, lw=1.1)
    ax_aR.plot(dg["step"], dg["cumul_wall"], "--", color=COLOR_DG, lw=1.1)
    ax_aR.set_ylabel("Cumulative [s]")
    ax_aR.set_ylim(0, None)
    ax_aR.grid(False)

    # (b) active DOFs (left) + active cells (right, identical schedules)
    ax_b.plot(cg["step"], cg["active_dofs"] / 1e3, color=COLOR_CG, lw=1.4)
    ax_b.plot(dgb["step"], dgb["active_dofs"] / 1e3, color=COLOR_DGB, lw=1.0)
    ax_b.plot(dg["step"], dg["active_dofs"] / 1e3, color=COLOR_DG, lw=1.4)
    for m in (cg, dg):
        ax_b.axhline(m["active_dofs"][-1] / 1e3, color=COLOR_REF, lw=0.6,
                     linestyle="--")
    ax_b.text(5, dg["active_dofs"][-1] / 1e3 + 0.6,
              f"{int(dg['active_dofs'][-1]):,} DG/DGB",
              ha="left", va="bottom", fontsize=FONT_SIZE - 1, color=COLOR_DG)
    ax_b.text(5, cg["active_dofs"][-1] / 1e3 + 0.6,
              f"{int(cg['active_dofs'][-1]):,} CG",
              ha="left", va="bottom", fontsize=FONT_SIZE - 1, color=COLOR_CG)
    ax_b.set_xlabel("Time step")
    ax_b.set_ylabel(r"Active DOFs [$\times10^3$]")
    ax_b.set_xlim(1, x1)
    ax_b.set_ylim(0, dg["active_dofs"][-1] / 1e3 * 1.16)
    ax_bR = ax_b.twinx()
    ax_bR.plot(dg["step"], dg["active_cells"], color=COLOR_REF, lw=1.1,
               linestyle="-.")
    ax_bR.set_ylabel("Active cells", color=COLOR_REF)
    ax_bR.tick_params(axis="y", colors=COLOR_REF)
    ax_bR.set_ylim(0, dg["active_cells"][-1] * 1.05)
    ax_bR.grid(False)

    # (c) Newton iterations per step (direct solver: one linear solve each)
    ax_c.plot(cg["step"], cg["newton"], color=COLOR_CG, lw=1.0,
              drawstyle="steps-mid", label="CG")
    ax_c.plot(dgb["step"], dgb["newton"], color=COLOR_DGB, lw=1.0,
              drawstyle="steps-mid", label="bonded DG")
    ax_c.plot(dg["step"], dg["newton"], color=COLOR_DG, lw=1.0,
              drawstyle="steps-mid", label="DG")
    ax_c.set_xlabel("Time step")
    ax_c.set_ylabel("Newton iterations")
    ax_c.set_xlim(1, x1)
    ax_c.set_ylim(0, max(cg["newton"].max(), dgb["newton"].max(), dg["newton"].max()) + 2)
    ax_c.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=3,
                frameon=False, handlelength=1.4, borderaxespad=0.0,
                columnspacing=0.8)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths = [FIG_DIR / f"cost_cylinder.{e}" for e in ("png", "pdf")]
    save_figure(fig, *paths)
    plt.close(fig)
    for p in paths:
        print(f"wrote {p}")

    # [PAPER] numbers
    for tag, m in (("DG", dg), ("bonded DG", dgb), ("CG", cg)):
        total = m["cumul_wall"][-1]
        tn, tp = m["t_newton"].sum(), m["t_perzyna"].sum()
        tj, ti = m["t_proj"].sum(), m["t_io"].sum()
        other = total - tn - tp - tj - ti
        print(f"[PAPER] {tag}: total {total:.1f}s | Newton {tn:.1f}s "
              f"({100*tn/total:.0f}%) | Perzyna {tp:.1f}s | proj {tj:.1f}s | "
              f"I/O {ti:.1f}s | overhead {other:.1f}s | "
              f"Newton iters {int(m['newton'].sum())}")
    print(f"[PAPER] DG/CG wall ratio: "
          f"{dg['cumul_wall'][-1]/cg['cumul_wall'][-1]:.1f}x")
    print(f"[PAPER] bondedDG/CG wall ratio: "
          f"{dgb['cumul_wall'][-1]/cg['cumul_wall'][-1]:.1f}x | "
          f"DG/bondedDG wall ratio: {dg['cumul_wall'][-1]/dgb['cumul_wall'][-1]:.1f}x")


if __name__ == "__main__":
    main()
