# -*- coding: utf-8 -*-
"""小样本降采样工具（M5 · 人员2 负责）

在"按病例划分 train/val/test"的基础上，对训练集按比例做降采样，生成 10%/20%
等小样本实验子集，并把每个实验的病例清单固化为 manifest.json（接口契约 C：
seed / 数据量 / 病例清单 / 配置，供人员4 做统计检验与 Risk-Coverage 曲线）。

口径（与 总模型代码进行中.ipynb 模块3/模块4 完全一致）：
- 按病例划分，禁止同一病例的切片跨集合（避免数据泄漏）；
- seed 固定并写入 manifest，保证可复现；
- 测试集永远冻结：test 始终取全量划分的测试病例，不做任何降采样；
- 兼容 .nii 与 .nii.gz（沿用 notebook 已修复的口径）。

命令行用法（本地 4 病例快速验证）：
    python downsample.py --data-root E:/大创/kits19_small --out-dir E:/大创/experiments \
        --fractions 0.10,0.20 --seed 42 --train-ratio 0.5 --val-ratio 0.25 --max-cases 4

Kaggle 全量跑：
    python downsample.py --data-root /kaggle/input/datasets/user123454321/kits19-1 \
        --out-dir /kaggle/working/experiments --fractions 0.10,0.20 --seed 42

也可 import 后调用 build_experiments()。
"""
import argparse
import json
import os
import random


def _find_volume(case_dir, name):
    """在病例目录中查找 imaging/segmentation 文件，兼容 .nii 和 .nii.gz。"""
    for ext in (".nii", ".nii.gz"):
        path = os.path.join(case_dir, name + ext)
        if os.path.isfile(path):
            return path
    return None


def scan_valid_cases(data_root):
    """返回 data_root 下同时具有 imaging 与 segmentation 的病例名列表（排序后）。"""
    names = []
    for entry in sorted(os.listdir(data_root)):
        case_dir = os.path.join(data_root, entry)
        if not (entry.startswith("case_") and os.path.isdir(case_dir)):
            continue
        if _find_volume(case_dir, "imaging") and _find_volume(case_dir, "segmentation"):
            names.append(entry)
    return names


def split_cases(case_dirs, train_ratio, val_ratio, seed):
    """按病例划分 train/val/test，返回 (train, val, test)。与 notebook 模块4 口径一致。"""
    rng = random.Random(seed)
    cases = list(case_dirs)
    rng.shuffle(cases)
    n = len(cases)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_cases = cases[:n_train]
    val_cases = cases[n_train:n_train + n_val]
    test_cases = cases[n_train + n_val:]
    return train_cases, val_cases, test_cases


def downsample_train_cases(train_cases, fraction, seed):
    """在训练集内按病例做降采样，返回降采样后的训练病例名列表。

    - fraction 为 (0, 1] 的浮点数，例如 0.10 / 0.20；
    - 用 seed 打乱后取前 n 个（确定性，可复现）；
    - n = int(len * fraction)，不足 1 个时取 1（保证小数据也能出实验）。
    """
    if not (0.0 < fraction <= 1.0):
        raise ValueError("fraction 必须在 (0, 1] 之间，当前为 %r" % fraction)
    rng = random.Random(seed)
    shuffled = list(train_cases)
    rng.shuffle(shuffled)
    n = int(len(shuffled) * fraction)
    if n == 0 and shuffled:
        n = 1
    return shuffled[:n]


def write_manifest(experiment_dir, *, seed, fraction, train_cases, val_cases, test_cases,
                   n_total_cases, data_root, img_size=256):
    """按接口契约 C 写出 manifest.json，返回文件路径。"""
    os.makedirs(experiment_dir, exist_ok=True)
    manifest = {
        "experiment": os.path.basename(experiment_dir),
        "data_root": data_root,
        "img_size": img_size,
        "seed": seed,
        "data_fraction": fraction,
        "n_total_cases": n_total_cases,
        "n_train_cases": len(train_cases),
        "n_val_cases": len(val_cases),
        "n_test_cases": len(test_cases),
        "train_cases": sorted(train_cases),
        "val_cases": sorted(val_cases),
        "test_cases": sorted(test_cases),
        "notes": "测试集冻结：test 取全量划分测试病例，不做降采样；train 为降采样后的子集。",
    }
    path = os.path.join(experiment_dir, "manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return path


def build_experiments(data_root, out_dir, fractions=(0.10, 0.20), seed=42,
                      train_ratio=0.70, val_ratio=0.15, max_cases=None, img_size=256):
    """扫描病例 -> 全量按病例划分 -> 对 train 逐个 fraction 降采样 -> 写 manifest。

    返回生成的 manifest 文件路径列表。
    """
    all_names = scan_valid_cases(data_root)
    if max_cases is not None and len(all_names) > max_cases:
        all_names = all_names[:max_cases]
    train_cases, val_cases, test_cases = split_cases(all_names, train_ratio, val_ratio, seed)
    if not train_cases:
        raise RuntimeError("训练集为空，无法降采样。请检查 data_root 与划分比例。")
    if not val_cases:
        print("[警告] 验证集为空（病例数太少或 val_ratio 太小），后续训练将无验证曲线。")
    if not test_cases:
        print("[警告] 测试集为空（病例数太少或划分比例不合适），评估模块12 将跳过。")

    manifests = []
    for fraction in fractions:
        sub_train = downsample_train_cases(train_cases, fraction, seed)
        exp_dir = os.path.join(out_dir, "data{:.0f}pct".format(fraction * 100))
        manifests.append(write_manifest(
            exp_dir,
            seed=seed,
            fraction=fraction,
            train_cases=sub_train,
            val_cases=val_cases,
            test_cases=test_cases,
            n_total_cases=len(all_names),
            data_root=data_root,
            img_size=img_size,
        ))
    return manifests


def main(argv=None):
    parser = argparse.ArgumentParser(description="KiTS19 小样本降采样工具（M5）")
    parser.add_argument("--data-root", required=True, help="病例根目录（含 case_* 子目录）")
    parser.add_argument("--out-dir", required=True, help="实验输出目录")
    parser.add_argument("--fractions", default="0.10,0.20", help="逗号分隔的降采样比例")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--max-cases", type=int, default=None, help="只取前 N 个病例（本地调试用）")
    parser.add_argument("--img-size", type=int, default=256)
    args = parser.parse_args(argv)

    fractions = [float(x) for x in args.fractions.split(",")]
    manifests = build_experiments(
        args.data_root, args.out_dir,
        fractions=fractions,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        max_cases=args.max_cases,
        img_size=args.img_size,
    )
    print("\n==== 降采样实验清单（契约 C manifest）====")
    for path in manifests:
        with open(path, encoding="utf-8") as f:
            m = json.load(f)
        print("已生成:", path)
        print("  训练 {} 例 | 验证 {} 例 | 测试 {} 例（冻结）| 数据比例 {:.0f}%".format(
            m["n_train_cases"], m["n_val_cases"], m["n_test_cases"], m["data_fraction"] * 100))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())