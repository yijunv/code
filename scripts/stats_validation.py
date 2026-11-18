import csv
import json
import os

import numpy as np
from scipy import stats

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "bayes")
NAMES = ["data10pct", "data20pct", "data30pct", "datafull"]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "stats_validation_results.json")


def load(name):
    ent, dice = [], []
    with open(os.path.join(BASE, name, "tumor_entropy_scores.csv"), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ent.append(float(row["tumor_mean_entropy"]))
            dice.append(float(row["dice"]))
    return np.asarray(ent), np.asarray(dice)


def error_rate_by_entropy_bin(ent, dice, n_bins=10):
    ent = np.asarray(ent)
    dice = np.asarray(dice)
    order = np.argsort(ent, kind="mergesort")
    e = ent[order]
    d = dice[order]
    n = len(e)
    centers, rates, counts = [], [], []
    for b in range(n_bins):
        lo = b * n // n_bins
        hi = (b + 1) * n // n_bins
        if hi <= lo:
            continue
        centers.append(float(e[lo:hi].mean()))
        rates.append(float((d[lo:hi] < 0.5).mean()))
        counts.append(hi - lo)
    return centers, rates, counts


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


def bootstrap_auroc(ent, dice, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(dice)
    point = rank_auc(ent, dice < 0.5)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        a = rank_auc(ent[idx], dice[idx] < 0.5)
        if a is not None:
            boots.append(a)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"point": round(point, 4), "ci95_low": round(float(lo), 4), "ci95_high": round(float(hi), 4)}


def main():
    report = {}
    for name in NAMES:
        ent, dice = load(name)
        r, p = stats.pearsonr(ent, dice)
        rho, ps = stats.spearmanr(ent, dice)
        centers, rates, counts = error_rate_by_entropy_bin(ent, dice, n_bins=10)
        report[name] = {
            "n": int(len(dice)),
            "pearson_r": round(float(r), 4),
            "pearson_p": float(p),
            "spearman_rho": round(float(rho), 4),
            "spearman_p": float(ps),
            "auroc_entropy_wrong": bootstrap_auroc(ent, dice),
            "reliability_bins": {
                "entropy_centers": [round(c, 4) for c in centers],
                "error_rate": [round(x, 4) for x in rates],
                "counts": counts,
            },
        }

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
