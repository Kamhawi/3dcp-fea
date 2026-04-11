"""Milestone comparison figure for the hollow-cylinder validation case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Mapping, Optional


def _load_results(results_or_path):
    if isinstance(results_or_path, Mapping):
        return results_or_path
    path = Path(results_or_path)
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_comparison_figure(results_or_path, png_path=None, pdf_path=None):
    """Write the six-panel layer comparison figure."""
    import matplotlib.pyplot as plt

    results = _load_results(results_or_path)
    milestones = results.get("milestones", {})
    milestone_layers = results.get("validation", {}).get(
        "milestone_layers",
        [5, 10, 15, 20, 25, 30],
    )

    if not milestones:
        raise ValueError("results.json does not contain milestone profile data.")

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.0), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, layer in zip(axes, milestone_layers):
        milestone = milestones.get(str(layer))
        if milestone is None:
            ax.set_visible(False)
            continue

        z_vals = milestone["z_profile_mm"]
        x_nom = milestone["x_nominal_profile_mm"]
        x_def = milestone["x_deformed_profile_mm"]

        ax.plot(x_nom, z_vals, color="#a0a0a0", linestyle="--", linewidth=1.5)
        ax.plot(x_def, z_vals, color="#d35400", linewidth=2.5)
        ax.fill_betweenx(z_vals, x_nom, x_def, color="#f5cba7", alpha=0.45)
        ax.set_title(f"Layer {layer}", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.20)
        ax.set_xlim(min(x_nom) - 5.0, max(x_def) + 8.0)
        ax.set_ylim(0.0, max(z_vals) + 10.0)

    for ax in axes[3:]:
        ax.set_xlabel("Visible Side x [mm]")
    for ax in axes[::3]:
        ax.set_ylabel("Height z [mm]")

    fig.suptitle("Cylinder Print Validation Milestones", fontsize=14, fontweight="bold")
    fig.tight_layout()

    if png_path is not None:
        png_path = Path(png_path)
        png_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_path, dpi=220, bbox_inches="tight")
    if pdf_path is not None:
        pdf_path = Path(pdf_path)
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(pdf_path, bbox_inches="tight")

    plt.close(fig)


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Render the six-panel cylinder validation comparison figure."
    )
    parser.add_argument("results", type=Path, help="Path to results.json")
    parser.add_argument(
        "--png",
        type=Path,
        default=None,
        help="Optional PNG output path (default: alongside results.json).",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=None,
        help="Optional PDF output path (default: alongside results.json).",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None):
    parser = _build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    png_path = args.png
    pdf_path = args.pdf
    if png_path is None:
        png_path = args.results.with_name("comparison_layers_5_10_15_20_25_30.png")
    if pdf_path is None:
        pdf_path = args.results.with_name("comparison_layers_5_10_15_20_25_30.pdf")

    write_comparison_figure(args.results, png_path=png_path, pdf_path=pdf_path)


if __name__ == "__main__":
    main()

