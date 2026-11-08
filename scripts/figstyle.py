"""Shared matplotlib style for the project's publication figures."""

import matplotlib.pyplot as plt

# Okabe-Ito colorblind-safe palette, subset with sufficient contrast on white.
OKABE_ON_WHITE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#000000"]


def apply_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "lines.linewidth": 1.8,
            "lines.markersize": 4.5,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "savefig.dpi": 300,
            "figure.dpi": 300,
            "svg.fonttype": "path",  # embed text as paths so fonts never render as boxes
        }
    )
