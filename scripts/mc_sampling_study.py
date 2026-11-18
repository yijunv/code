import json
import os

import numpy as np
import torch

import run_experiments as runexp

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MC_LIST = [5, 10, 20, 50]
MAX_MC = max(MC_LIST)
EPS = 1e-8

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
DATA_ROOT = os.environ.get("KITS19_DATA_ROOT", "")
MODEL_PATH = os.environ.get("KITS19_MODEL_PATH", "")
MANIFEST = os.environ.get("KITS19_MANIFEST", "")
OUT = os.path.join(REPO_DIR, "results", "mc_sampling_results.json")


def mc_aggregate(probs_k):
    """probs_k: (k, C, H, W) softmax 概率；返回平均概率、argmax 预测、逐像素熵。"""
    mean_probs = probs_k.mean(dim=0)
    mean_pred = torch.argmax(mean_probs, dim=0)
    entropy = -torch.sum(mean_probs * torch.log(mean_probs + EPS), dim=0)
    return mean_probs, mean_pred, entropy


def rank_auc(score, label):
    score = np.asarray(score, float)
    label = np.asarray(label, bool)
    pos = label.sum()
    neg = (~label).sum()
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(score, kind="mergesort")
    s = score[order]
    ranks = np.empty(len(score))
    i = 0
    n = len(score)
    while i < n:
        j = i
        while j < n and s[j] == s[i]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2 + 1
        i = j
    return float((ranks[label].sum() - pos * (pos + 1) / 2) / (pos * neg))


def summarize(ent, dice):
    ent = np.asarray(ent)
    dice = np.asarray(dice)
    n = len(dice)
    correct = dice >= 0.5
    order = np.argsort(ent, kind="mergesort")
    c = correct[order]
    cum = np.cumsum(c).astype(float)

    def sel_at(cov):
        k = max(1, min(n, int(round(cov * n))))
        return float(cum[k - 1] / k)

    return {
        "n_tumor_positive_slices": n,
        "selective_accuracy_all": round(float(correct.mean()), 4),
        "selective_accuracy_70": round(sel_at(0.7), 4),
        "selective_accuracy_50": round(sel_at(0.5), 4),
        "auroc_entropy_wrong": None if rank_auc(ent, dice < 0.5) is None else round(rank_auc(ent, dice < 0.5), 4),
        "aurc_risk_coverage": round(float(np.mean(1 - cum / np.arange(1, n + 1))), 4),
    }


def main():
    if not (DATA_ROOT and MODEL_PATH and MANIFEST):
        raise SystemExit("需要设置环境变量 KITS19_DATA_ROOT / KITS19_MODEL_PATH / KITS19_MANIFEST（本地数据与权重路径）")
    with open(MANIFEST, encoding="utf-8") as f:
        test_cases = json.load(f)["test_cases"]
    paths = [os.path.join(DATA_ROOT, c) for c in test_cases]
    ds = runexp.KiTS19SliceDataset(paths, transform=None)

    model = runexp.UNet(dropout_rate=0.1).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.train()
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
            m.eval()

    per_ent = {k: [] for k in MC_LIST}
    per_dice = {k: [] for k in MC_LIST}

    with torch.no_grad():
        for i in range(len(ds)):
            img, mask = ds[i]
            img = img.unsqueeze(0).expand(MAX_MC, -1, -1, -1).to(DEVICE)  # (MAX_MC,1,H,W)
            mask = mask.unsqueeze(0).to(DEVICE)  # (1,H,W)
            logits = model(img)  # (MAX_MC, C, H, W)
            probs = torch.softmax(logits, dim=1)

            target_tumor = (mask == 2).squeeze(0)
            tsum = int(target_tumor.sum())
            for k in MC_LIST:
                _, mean_pred, entropy = mc_aggregate(probs[:k])
                pred_tumor = mean_pred == 2
                inter = int((pred_tumor & target_tumor).sum())
                psum = int(pred_tumor.sum())
                dice = 2 * inter / (psum + tsum + 1e-8)
                if tsum > 0:
                    per_ent[k].append(float(entropy[target_tumor].mean()))
                    per_dice[k].append(dice)

            if (i + 1) % 100 == 0:
                print(f"  切片 {i + 1}/{len(ds)}", flush=True)

    results = {str(k): summarize(per_ent[k], per_dice[k]) for k in MC_LIST}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
