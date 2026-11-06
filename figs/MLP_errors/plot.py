"""Plot energy, force and charge errors for 23 molecules.

Usage: python figs/MLP_errors/plot.py --output mlp_errors_output
Requires NumPy and Matplotlib.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np

DATA = Path(__file__).resolve().parent / "data.json"
PANELS = [
    ("energy_mae_kcal_mol", "#b95a58", "(a) Energy", r"kcal mol$^{-1}$", .2, "%.1f", .4),
    ("force_error_kcal_mol_A", "#6686ad", "(b) Force", r"kcal mol$^{-1}$ $\mathrm{\AA}^{-1}$", .4, "%.1f", .8),
    ("charge_mae_e", "#86b5a1", "(c) Charge", r"$e$", .004, "%.3f", .008),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("mlp_errors_output"))
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    data = json.loads(DATA.read_text(encoding="utf-8"))
    numbers = np.arange(1, 24)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 9.5,
        "xtick.labelsize": 10, "ytick.labelsize": 9, "axes.linewidth": .7,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(150 / 25.4, 100 / 25.4))
    for ax, (key, color, title, unit, step, formatter, upper) in zip(axes, PANELS):
        values = np.asarray(data[key])
        if (values.shape != (23,) or not np.isfinite(values).all()
                or np.any(values < 0) or np.any(values >= upper)):
            raise ValueError(f"{key}: expected 23 finite values within the plotting range")
        ax.bar(numbers, values, width=.42, color=color, edgecolor="black", linewidth=.55, zorder=3)
        ax.set_ylim(0, upper)
        ax.set_yticks(np.arange(0, upper + step / 2, step))
        ax.yaxis.set_major_formatter(FormatStrFormatter(formatter))
        ax.set_xlim(.35, 23.65)
        ax.set_xticks(numbers, [str(n) for n in numbers])
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold", pad=4)
        ax.text(1, 1.03, unit, transform=ax.transAxes, ha="right", va="bottom", fontsize=9)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#e5e5e5", linewidth=.55)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="both", length=3, width=.7)
    axes[-1].set_xlabel("Molecule", fontsize=11, labelpad=7)
    fig.subplots_adjust(left=.09, right=.985, bottom=.16, top=.925, hspace=.25)
    for extension in ["pdf", "svg", "png"]:
        fig.savefig(output / f"mlp_errors.{extension}", dpi=600)
    plt.close(fig)
    print(f"Saved MLP errors to {output.resolve()}")


if __name__ == "__main__":
    main()
