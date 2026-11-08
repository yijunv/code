import glob
import json
import os

os.environ.setdefault("MPLCONFIGDIR", r"D:\codex工作空间\.mplcache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import figstyle
import run_experiments_new as runexp

figstyle.apply_style()

DATA_ROOT = r"D:\大创代码\kits19\datasets"
MANIFEST = r"D:\大创代码\output_ft\experiments\datafull\manifest.json"
SAMPLES = r"D:\大创代码\output_ft\experiments\datafull\samples"
OUT = r"D:\codex工作空间\figures"


def tumor_mean_entropy(f):
    d = np.load(f, allow_pickle=False)
    lab = d["label"]
    ent = d["entropy"]
    tm = lab == 2
    if tm.sum() == 0:
        return None
    return float(ent[tm].mean())


files = sorted(glob.glob(os.path.join(SAMPLES, "sample_*.npz")))
scored = []
for i, f in enumerate(files):
    me = tumor_mean_entropy(f)
    if me is not None:
        scored.append((me, i, f))
scored.sort(key=lambda x: x[0])
low, high = scored[0], scored[-1]

with open(MANIFEST, encoding="utf-8") as f:
    test_cases = json.load(f)["test_cases"]
ds = runexp.KiTS19SliceDataset([os.path.join(DATA_ROOT, c) for c in test_cases], transform=None)


def panel(img, lab, pred, ent):
    fig, axes = plt.subplots(1, 3, figsize=(178 / 25.4, 62 / 25.4), layout="constrained")
    axes[0].imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("CT slice")

    axes[1].imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].contour(lab == 2, levels=[0.5], colors="#0072B2", linewidths=1.5)
    axes[1].contour(pred == 2, levels=[0.5], colors="#D55E00", linewidths=1.5, linestyles="dashed")
    axes[1].set_title("Tumor overlay")
    axes[1].legend(
        handles=[
            Line2D([0], [0], color="#0072B2", linewidth=1.5, label="Ground truth tumor"),
            Line2D([0], [0], color="#D55E00", linewidth=1.5, linestyle="dashed", label="Predicted tumor"),
        ],
        loc="upper right",
    )

    im = axes[2].imshow(ent, cmap="viridis", interpolation="nearest")
    axes[2].set_title("Pixel entropy")
    fig.colorbar(im, ax=axes[2], label="Entropy (nats)", fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    return fig


def render(item, tag):
    me, idx, f = item
    img, _ = ds[idx]
    img = img.squeeze(0).numpy()
    d = np.load(f, allow_pickle=False)
    fig = panel(img, d["label"], d["pred"], d["entropy"])
    fig.suptitle(f"{tag}: mean tumor entropy = {me:.3f}")
    fig.savefig(os.path.join(OUT, f"overlay_{tag.lower()}.png"), dpi=300)
    fig.savefig(os.path.join(OUT, f"overlay_{tag.lower()}.svg"))
    plt.close(fig)
    print(f"{tag}: sample idx {idx}, mean tumor entropy {me:.3f}")


render(low, "low_uncertainty")
render(high, "high_uncertainty")
print("done")
