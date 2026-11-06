"""Plot DFT/model energy parity with NumPy and Matplotlib.

Each molecule has 400 geometries: 100 each from MM/ML gas/aqueous trajectories.
Input model errors already have one global mean energy offset removed per model.
The displayed R² measures agreement with y=x, not squared Pearson correlation.
Usage: python figs/energy_parity/plot.py --output energy_parity_output
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA = Path(__file__).resolve().parent / "data"
MOLECULES = [("3259411", "Glycerol"), ("4687447", "2-Propoxyethanol")]
MODELS = [("mm_centered_error", "#86b5a1", "MM"),
          ("ml_centered_error", "#516d7e", "ML")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("energy_parity_output"))
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .65, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.65))
    fig.subplots_adjust(left=.085, right=.985, bottom=.19, top=.88, wspace=.32)
    results = {}
    for ax, (mid, title), letter in zip(axes, MOLECULES, "ab"):
        with np.load(DATA / f"{mid}.npz", allow_pickle=False) as data:
            x = data["dft_energy_relative_kcal_mol"]
            predictions = {label: x + data[key] for key, _, label in MODELS}
        if any(v.shape != (400,) or not np.isfinite(v).all()
               for v in [x, *predictions.values()]):
            raise ValueError(f"{mid}: expected 400 finite energies per series")
        if any(np.any(np.abs(v) >= 20) for v in [x, *predictions.values()]):
            raise ValueError(f"{mid}: fixed plot limits would hide points")
        ax.plot([-20, 20], [-20, 20], color="#888888", lw=.8,
                ls=(0, (4, 3)), zorder=0)
        results[mid] = {}
        for _, color, label in MODELS:
            y = predictions[label]
            error = y - x
            results[mid][label] = {
                "n": len(x), "pearson_r": float(np.corrcoef(x, y)[0, 1]),
                "parity_R2": float(1 - np.sum(error**2) / np.sum((x - x.mean())**2)),
                "MAE": float(np.abs(error).mean()),
                "RMSE": float(np.sqrt(np.mean(error**2))),
            }
            ax.scatter(x, y, s=11, c=color, alpha=.80, edgecolors="white",
                       linewidths=.2, label=label, zorder=2)
        ax.set(xlim=(-20, 20), ylim=(-20, 20), aspect="equal",
               xticks=[-20, -10, 0, 10, 20], yticks=[-20, -10, 0, 10, 20],
               xlabel="DFT relative energy (kcal mol$^{-1}$)",
               ylabel="Model relative energy (kcal mol$^{-1}$)")
        for label, y_pos in [("MM", .135), ("ML", .020)]:
            values = results[mid][label]
            text = (f"{label}:  $R^2$ = {values['parity_R2']:.3f}\n"
                    f"RMSE = {values['RMSE']:.3f} kcal mol$^{{-1}}$")
            ax.text(.97, y_pos, text, transform=ax.transAxes, ha="right",
                    va="bottom", fontsize=6.5, linespacing=1.3,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=.9, pad=1.5))
        ax.set_title(title, fontsize=10, pad=9)
        ax.text(-.20, 1.08, letter, transform=ax.transAxes, fontsize=12, weight="bold")
        legend = ax.legend(loc="upper left", fontsize=8, frameon=True, fancybox=False,
                           handlelength=1, handletextpad=.4, borderpad=.35, labelspacing=.3)
        legend.get_frame().set(edgecolor="#666666", linewidth=.65, linestyle=(0, (3, 2)))
    for extension in ["pdf", "png", "svg"]:
        fig.savefig(output / f"energy_parity.{extension}", dpi=300)
    plt.close(fig)
    (output / "statistics.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"Saved energy parity and statistics to {output.resolve()}")


if __name__ == "__main__":
    main()
