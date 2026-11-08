import json
import os

os.environ.setdefault("MPLCONFIGDIR", r"D:\codex工作空间\.mplcache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import figstyle

figstyle.apply_style()

RESULTS = r"D:\codex工作空间\stats_validation_results.json"
OUT = r"D:\codex工作空间\figures"

STYLES = ["-", "--", "-."]
MARKERS = ["o", "s", "^"]
LABELS = {"data10pct": "10%", "data20pct": "20%", "datafull": "full"}

with open(RESULTS, encoding="utf-8") as f:
    data = json.load(f)

fig, ax = plt.subplots(figsize=(89 / 25.4, 70 / 25.4), layout="constrained")
for i, (name, label) in enumerate(LABELS.items()):
    bins = data[name]["reliability_bins"]
    ax.plot(
        bins["entropy_centers"],
        bins["error_rate"],
        color=figstyle.OKABE_ON_WHITE[i],
        linestyle=STYLES[i],
        marker=MARKERS[i],
        label=label,
    )

ax.set_xlabel("Mean tumor entropy (bin)")
ax.set_ylabel("Error rate (fraction with Dice < 0.5)")
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)
ax.legend(title="Training data")
ax.set_title("Reliability of entropy as an uncertainty signal")
fig.savefig(os.path.join(OUT, "reliability.png"), dpi=300)
fig.savefig(os.path.join(OUT, "reliability.svg"))
plt.close(fig)

print("saved ->", os.path.join(OUT, "reliability.png"))
