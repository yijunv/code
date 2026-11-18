"""Data-scaling figures for the Bayesian renal-CT triage project.

Raw data (all repo-relative):
  - results/bayes/{data10pct,data20pct,data30pct,datafull}/metrics.json
  - results/vanilla/{vanilla10pct,vanilla20pct,vanillafull}/metrics.json
  - results/stats_validation_results.json (AUROC 95% bootstrap CI)
Outputs:
  - assets/figures/data_scaling_dice.png / .svg
  - assets/figures/data_scaling_uncertainty.png / .svg

Design notes:
  - Single seed (42), one run per group: no run-to-run error bars are shown.
  - AUROC uncertainty is the 95% bootstrap CI (1000 resamples, seed 42).
  - Selective accuracy is a point estimate only; no CI is computed.
  - Log x-axis because training cases span 14 to 147 (about one order of magnitude).
"""

import json
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import figstyle

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))

BAYES_BASE = os.path.join(REPO_DIR, "results", "bayes")
VANILLA_BASE = os.path.join(REPO_DIR, "results", "vanilla")
STATS_PATH = os.path.join(REPO_DIR, "results", "stats_validation_results.json")
OUT_DIR = os.path.join(REPO_DIR, "assets", "figures")

BAYES_NAMES = ["data10pct", "data20pct", "data30pct", "datafull"]
VANILLA_NAMES = ["vanilla10pct", "vanilla20pct", "vanillafull"]

C_BLUE = "#0072B2"
C_ORANGE = "#D55E00"
C_GREEN = "#009E73"
C_GRAY = "#666666"


def load_metrics(base, names):
    rows = []
    for name in names:
        path = os.path.join(base, name, "metrics.json")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        rows.append((data["n_train_cases"], data["metrics"]))
    return rows


def load_auroc_ci():
    if not os.path.exists(STATS_PATH):
        return {}
    with open(STATS_PATH, encoding="utf-8") as f:
        stats = json.load(f)
    out = {}
    for name, entry in stats.items():
        a = entry.get("auroc_entropy_wrong") or {}
        if a.get("point") is not None and a.get("ci95_low") is not None:
            out[name] = (a["point"], a["ci95_low"], a["ci95_high"])
    return out


def style_logx(ax, cases):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(mticker.FixedLocator(cases))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: str(int(v))))
    ax.grid(True, which="major", alpha=0.25)


def annotate(ax, xs, ys, dx=0, dy=0, offsets=None, anchor_ys=None, pad_pt=6):
    """Place value labels near points with a clear gap from the data line."""
    fig = ax.figure
    fig.canvas.draw()
    scale = fig.dpi / 72.0
    bbox = ax.get_window_extent()
    pad_x = pad_pt * scale
    pad_y = pad_pt * scale
    inv = ax.transData.inverted()
    if offsets is None:
        offsets = [(dx, dy)] * len(xs)
    if anchor_ys is None:
        anchor_ys = ys
    for x, y, ay, (odx, ody) in zip(xs, ys, anchor_ys, offsets):
        px, py = ax.transData.transform((x, ay))
        tx = min(max(px + odx * scale, bbox.x0 + pad_x), bbox.x1 - pad_x)
        ty = min(max(py + ody * scale, bbox.y0 + pad_y), bbox.y1 - pad_y)
        axx, axy = inv.transform((tx, ty))
        ax.text(
            axx,
            axy,
            f"{y:.3f}",
            fontsize=6.5,
            color="#222222",
            ha="center",
            va="center",
        )


def figure_dice():
    bayes = load_metrics(BAYES_BASE, BAYES_NAMES)
    vanilla = load_metrics(VANILLA_BASE, VANILLA_NAMES)
    fig, ax = plt.subplots(figsize=(6.4, 3.8), layout="constrained")

    bx = [n for n, _ in bayes]
    bd = [m["dice"] for _, m in bayes]
    vx = [n for n, _ in vanilla]
    vd = [m["dice"] for _, m in vanilla]

    ax.plot(bx, bd, marker="o", ls="-", color=C_BLUE, label="Bayesian U-Net")
    ax.plot(vx, vd, marker="s", ls="--", color=C_ORANGE, label="Vanilla U-Net")

    style_logx(ax, sorted(set(bx + vx)))
    ax.set_xlabel("Training cases (n)")
    ax.set_ylabel("Test Dice (tumor)")
    ax.set_ylim(0.0, 1.0)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    annotate(ax, bx, bd, offsets=[(0, 12), (0, 12), (0, 12), (0, -14)])
    annotate(ax, vx, vd, offsets=[(0, -14), (0, -14), (0, 14)])
    ax.legend(loc="lower right")

    fig.savefig(os.path.join(OUT_DIR, "data_scaling_dice.png"), dpi=300)
    fig.savefig(os.path.join(OUT_DIR, "data_scaling_dice.svg"))
    plt.close(fig)


def figure_uncertainty():
    bayes = load_metrics(BAYES_BASE, BAYES_NAMES)
    ci = load_auroc_ci()
    xs = [n for n, _ in bayes]
    auroc = [m["auroc_entropy_wrong"] for _, m in bayes]
    sel70 = [m["selective_accuracy_70"] for _, m in bayes]
    sel50 = [m["selective_accuracy_50"] for _, m in bayes]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(6.4, 5.6), layout="constrained", sharex=True
    )

    pts = []
    lows = []
    highs = []
    for name, value in zip(BAYES_NAMES, auroc):
        point, lo, hi = ci.get(name, (value, None, None))
        pts.append(point)
        lows.append(lo)
        highs.append(hi)

    if all(v is not None for v in lows) and all(v is not None for v in highs):
        yerr = [
            [p - lo for p, lo in zip(pts, lows)],
            [hi - p for p, hi in zip(pts, highs)],
        ]
    else:
        yerr = None

    ax1.errorbar(
        xs,
        pts,
        yerr=yerr,
        marker="o",
        ls="-",
        color=C_BLUE,
        capsize=3,
        capthick=0.8,
        elinewidth=0.8,
        label="AUROC",
    )
    ax1.axhline(0.5, color=C_GRAY, lw=0.8, ls=":")
    ax1.text(
        0.02,
        0.92,
        "chance (AUROC) = 0.5",
        transform=ax1.transAxes,
        fontsize=6.5,
        color=C_GRAY,
        va="top",
    )
    style_logx(ax1, xs)
    ax1.set_ylabel("AUROC\n(entropy vs. wrong)")
    ax1.set_ylim(0.435, 0.70)
    ax1.yaxis.set_major_locator(mticker.MultipleLocator(0.05))
    label_anchors = [h if h is not None else p for p, h in zip(pts, highs)]
    annotate(ax1, xs, pts, anchor_ys=label_anchors, dy=8)
    ax1.legend(loc="lower right", fontsize=7)

    ax2.plot(
        xs,
        sel70,
        marker="s",
        markersize=5.5,
        markeredgecolor="white",
        markeredgewidth=0.6,
        ls="--",
        color=C_ORANGE,
        label="70% coverage",
    )
    ax2.plot(
        xs,
        sel50,
        marker="^",
        markersize=5.0,
        markeredgecolor="white",
        markeredgewidth=0.6,
        ls=":",
        color=C_GREEN,
        label="50% coverage",
    )
    style_logx(ax2, xs)
    ax2.set_xlabel("Training cases (n)")
    ax2.set_ylabel("Selective accuracy")
    ax2.set_ylim(0.36, 0.66)
    ax2.yaxis.set_major_locator(mticker.MultipleLocator(0.05))
    annotate(ax2, xs, sel70, dy=-14)
    annotate(ax2, xs, sel50, dy=14)
    ax2.legend(loc="upper left", fontsize=7)

    fig.savefig(os.path.join(OUT_DIR, "data_scaling_uncertainty.png"), dpi=300)
    fig.savefig(os.path.join(OUT_DIR, "data_scaling_uncertainty.svg"))
    plt.close(fig)


def main():
    figstyle.apply_style()
    os.makedirs(OUT_DIR, exist_ok=True)
    figure_dice()
    figure_uncertainty()
    print("saved ->", OUT_DIR)


if __name__ == "__main__":
    main()
