"""Plot 2-propoxyethanol and hydrazine conformer populations without molecular artwork.

Data contain the population percentage in each angle bin.
Usage: python figs/torsion_populations/plot.py --output torsion_populations_output
Requires NumPy and Matplotlib.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

DATA = Path(__file__).resolve().parent / "data"
MODELS = [("MM xCM5", "#86b5a1"), ("ML/MM xCM5", "#e29957"),
          ("ML/MM xCM5′", "#b95a58")]
PANELS = [("4687447", "2-Propoxyethanol", r"OH torsion $|\phi_{\mathrm{OH}}|$ (°)", 10, 29),
          ("7261305", "Hydrazine", r"NH$_2$ relative orientation $|\tau|$ (°)", 5, 22)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("torsion_populations_output"))
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8,
        "axes.labelpad": 4, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .65, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "xtick.major.width": .65, "ytick.major.width": .65,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })
    fig = plt.figure(figsize=(7.4, 3.43))
    for i, (mid, title, xlabel, width, ymax) in enumerate(PANELS):
        data = json.loads((DATA / f"{mid}.json").read_text(encoding="utf-8"))
        edges = np.arange(0, 181, width)
        ax = fig.add_axes([(.63 + 3.7 * i) / 7.4, .49 / 3.43, 2.89 / 7.4, 2.30 / 3.43])
        for label, color in MODELS:
            population = np.asarray(data[label])
            if (population.shape != (len(edges) - 1,) or not np.isfinite(population).all()
                    or np.any(population < 0) or not np.isclose(population.sum(), 100)):
                raise ValueError(f"{mid}, {label}: invalid population percentages")
            ax.stairs(population, edges, fill=True, color=color, alpha=.07, lw=0, zorder=1)
            ax.stairs(population, edges, color=color, lw=1.2, baseline=None, zorder=2)
        ax.set(xlim=(0, 180), ylim=(0, ymax), xticks=np.arange(0, 181, 30),
               yticks=[0, 5, 10, 15, 20], xlabel=xlabel,
               ylabel=f"Population per {width:g}° bin (%)")
        ax.set_title(title, fontsize=9, pad=9)
        fig.text((.16 + 3.7 * i) / 7.4, 2.82 / 3.43, "ab"[i],
                 fontsize=11, weight="bold", va="bottom")
    fig.legend([Line2D([0], [0], color=color, lw=1.6) for _, color in MODELS],
               [label for label, _ in MODELS], loc="center", bbox_to_anchor=(.5, 3.24 / 3.43),
               ncol=3, frameon=False, fontsize=7, handlelength=2.7, columnspacing=2.8)
    for extension in ["pdf", "png", "svg"]:
        fig.savefig(output / f"torsion_populations.{extension}", dpi=300)
    plt.close(fig)
    print(f"Saved torsion populations to {output.resolve()}")


if __name__ == "__main__":
    main()
