import csv
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))

os.environ.setdefault("MPLCONFIGDIR", os.path.join(REPO_DIR, ".mplcache"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker

import figstyle

figstyle.apply_style()

BASE = os.path.join(REPO_DIR, "results", "bayes")
OUT = os.path.join(REPO_DIR, "assets", "figures")
os.makedirs(OUT, exist_ok=True)

GROUPS = [("data10pct", "10%"), ("data20pct", "20%"), ("data30pct", "30%"), ("datafull", "full")]


def load(name):
    ents, dices = [], []
    with open(os.path.join(BASE, name, "tumor_entropy_scores.csv"), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ents.append(float(row["tumor_mean_entropy"]))
            dices.append(float(row["dice"]))
    ent = np.asarray(ents)
    dice = np.asarray(dices)
    order = np.argsort(ent, kind="mergesort")
    d_sorted = dice[order]
    correct = d_sorted >= 0.5
    n = len(d_sorted)
    coverage = np.arange(1, n + 1) / n
    selective = np.cumsum(correct) / np.arange(1, n + 1)
    return coverage, selective, d_sorted


def read_dice(name):
    with open(os.path.join(BASE, name, "metrics.json"), encoding="utf-8") as f:
        return float(json.load(f)["metrics"]["dice"])


# 数据表：Risk-Coverage 曲线底层数据
rows = []
for name, label in GROUPS:
    coverage, selective, _ = load(name)
    rows.append((label, coverage, selective))
with open(os.path.join(REPO_DIR, "results", "risk_coverage_data.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["group", "coverage", "selective_accuracy"])
    for label, coverage, selective in rows:
        for c, s in zip(coverage, selective):
            w.writerow([label, round(float(c), 6), round(float(s), 6)])

# 图 1：Risk-Coverage 曲线（完整主图 + 低覆盖/高准确率两个局部放大图）
# 注意：三条曲线仍使用完整 871 个原始切片点；放大图只改坐标范围，不裁剪/平滑/插值数据。
fig = plt.figure(figsize=(13, 9), layout="constrained")
gs = fig.add_gridspec(2, 2, width_ratios=[1.4, 1.0], height_ratios=[1.0, 1.0])
ax_main = fig.add_subplot(gs[:, 0])
ax_low = fig.add_subplot(gs[0, 1])
ax_high = fig.add_subplot(gs[1, 1])


def draw_curve(ax, xlim, ylim, title):
    for i, (label, coverage, selective) in enumerate(rows):
        ax.plot(coverage, selective, color=figstyle.OKABE_ON_WHITE[i], linewidth=1.3, label=label)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(True, which="major", alpha=0.25)
    ax.set_title(title, fontsize=11)
    ax.tick_params(labelsize=9, pad=5)


draw_curve(ax_main, (0.0, 1.0), (0.0, 0.70), "Risk-coverage (full range)")
ax_main.xaxis.set_major_locator(mticker.MultipleLocator(0.10))
ax_main.yaxis.set_major_locator(mticker.MultipleLocator(0.05))
ax_main.set_xlabel("Coverage (fraction of tumor-positive slices accepted)")
ax_main.set_ylabel("Selective accuracy (Dice ≥ 0.5)")
ax_main.legend(title="Training data", loc="upper left")

draw_curve(ax_low, (0.0, 0.15), (0.0, 0.68), "Low-coverage zoom (0.00-0.15)")
ax_low.xaxis.set_major_locator(mticker.MultipleLocator(0.025))
ax_low.yaxis.set_major_locator(mticker.MultipleLocator(0.025))
ax_low.set_xlabel("Coverage")
ax_low.set_ylabel("Selective accuracy")

draw_curve(ax_high, (0.10, 1.00), (0.46, 0.67), "High-accuracy zoom (0.10-1.00)")
ax_high.xaxis.set_major_locator(mticker.MultipleLocator(0.10))
ax_high.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
ax_high.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
ax_high.yaxis.set_major_locator(mticker.MultipleLocator(0.02))
ax_high.set_xlabel("Coverage")
ax_high.set_ylabel("Selective accuracy")

fig.savefig(os.path.join(OUT, "risk_coverage.png"), dpi=300)
fig.savefig(os.path.join(OUT, "risk_coverage.svg"))
plt.close(fig)

# 图 2：三组测试 Dice 对比
dice_values = [read_dice(name) for name, _ in GROUPS]
labels = [label for _, label in GROUPS]
fig, ax = plt.subplots(figsize=(89 / 25.4, 60 / 25.4), layout="constrained")
bars = ax.bar(labels, dice_values, color=figstyle.OKABE_ON_WHITE[:4], width=0.5)
ax.set_ylabel("Test tumor Dice")
ax.set_ylim(0, 0.9)
ax.set_xlim(-0.75, 3.75)
ax.grid(True, axis="y", alpha=0.3)
for bar, val in zip(bars, dice_values):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.012, f"{val:.3f}", ha="center", va="bottom")
ax.set_title("Test tumor Dice across training sizes", loc="center")
fig.savefig(os.path.join(OUT, "dice_comparison.png"), dpi=300)
fig.savefig(os.path.join(OUT, "dice_comparison.svg"))
plt.close(fig)

print("figures written to", OUT)
