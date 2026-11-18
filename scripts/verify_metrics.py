import csv
import json
import os

import numpy as np

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "bayes")
NAMES = ["data10pct", "data20pct", "data30pct", "datafull"]


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


for name in NAMES:
    ent, dice = [], []
    with open(os.path.join(BASE, name, "tumor_entropy_scores.csv"), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ent.append(float(row["tumor_mean_entropy"]))
            dice.append(float(row["dice"]))
    ent = np.array(ent)
    dice = np.array(dice)
    n = len(dice)
    correct = dice >= 0.5
    order = np.argsort(ent, kind="mergesort")
    d = dice[order]
    c = correct[order]
    cum = np.cumsum(c).astype(float)
    sel = cum / np.arange(1, n + 1)

    def sel_at(cov):
        k = max(1, min(n, int(round(cov * n))))
        return float(cum[k - 1] / k)

    sel_all = float(correct.mean())
    sel70 = sel_at(0.7)
    sel50 = sel_at(0.5)
    auroc = rank_auc(ent, dice < 0.5)
    aurc = float(np.mean(1 - sel))

    with open(os.path.join(BASE, name, "metrics.json"), encoding="utf-8") as f:
        m = json.load(f)["metrics"]

    print(name)
    print(f"  n_slices          ours={n} saved={m['n_tumor_positive_slices']}")
    print(f"  sel_all           ours={sel_all:.4f} saved={m['selective_accuracy_all']}")
    print(f"  sel70             ours={sel70:.4f} saved={m['selective_accuracy_70']}")
    print(f"  sel50             ours={sel50:.4f} saved={m['selective_accuracy_50']}")
    print(f"  auroc_wrong       ours={auroc:.4f} saved={m['auroc_entropy_wrong']}")
    print(f"  aurc              ours={aurc:.4f} saved={m['aurc_risk_coverage']}")
