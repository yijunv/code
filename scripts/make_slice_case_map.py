# -*- coding: utf-8 -*-
"""从本地 KiTS19 数据重建 sample_id -> (case_id, slice_index) 对照表。

口径与 scripts/run_experiments.py 一致：
  - 全量病例 = 有 imaging + segmentation 的 case_XXXXX（排序）
  - split_cases(seed=42, train/val/test = 70/15/15)
  - 测试集按 split_cases 返回顺序，逐病例、逐非空轴向切片（升序）编号
  - sample_id = 1 起，sample_%04d

用法（需要本机数据集）：
  python scripts/make_slice_case_map.py --data-root <kits19/datasets>
"""
import argparse
import csv
import os
import random

import nibabel as nib
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))


def _find_volume(case_dir, name):
    for ext in (".nii", ".nii.gz"):
        p = os.path.join(case_dir, name + ext)
        if os.path.isfile(p):
            return p
    return None


def scan_valid_cases(data_root):
    names = []
    for entry in sorted(os.listdir(data_root)):
        case_dir = os.path.join(data_root, entry)
        if not (entry.startswith("case_") and os.path.isdir(case_dir)):
            continue
        if _find_volume(case_dir, "imaging") and _find_volume(case_dir, "segmentation"):
            names.append(entry)
    return names


def split_cases(case_dirs, train_ratio, val_ratio, seed):
    rng = random.Random(seed)
    cases = list(case_dirs)
    rng.shuffle(cases)
    n = len(cases)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return cases[:n_train], cases[n_train:n_train + n_val], cases[n_train + n_val:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default=os.path.join(REPO_DIR, "results", "slice_case_map.csv"))
    args = ap.parse_args()

    all_cases = scan_valid_cases(args.data_root)
    train, val, test = split_cases(all_cases, 0.70, 0.15, 42)
    print(f"all={len(all_cases)} train={len(train)} val={len(val)} test={len(test)}")

    rows = []
    idx = 0
    n_tumor = 0
    for case in test:
        seg = np.asarray(nib.load(os.path.join(args.data_root, case, "segmentation.nii.gz")).dataobj, dtype=np.uint8)
        nonempty = np.where(seg.sum(axis=(1, 2)) > 0)[0]
        for s in nonempty:
            idx += 1
            has_tumor = int(bool((seg[s] == 2).any()))
            n_tumor += has_tumor
            rows.append({"sample_id": "sample_%04d" % idx, "case_id": case, "slice_index": int(s), "has_tumor": has_tumor})

    print(f"total_nonempty_slices={idx} tumor_positive_slices={n_tumor}")
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "case_id", "slice_index", "has_tumor"])
        w.writeheader()
        w.writerows(rows)
    print("saved ->", args.out)


if __name__ == "__main__":
    main()
