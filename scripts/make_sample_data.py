# -*- coding: utf-8 -*-
"""从本地 KiTS19 数据生成一份脱敏样例（sample_data/），供 Web 团队联调推理。

用法（需要在有数据集和权重的机器上跑）：
  python scripts/make_sample_data.py --data-root <kits19/datasets> --weights <full/best_unet.pth>

产物（全部放在 --outdir）：
  slice_hounsfield.npy    原始轴向切片（HU 值，未缩放）
  slice_preprocessed.npy  预处理后的模型输入（256x256, [0,1]）
  slice.png               灰度预览
  mean_pred.npy           分割 0/1/2
  pred_binary.npy         肿瘤二值掩码
  mask.png                肿瘤掩码预览
  entropy.npy             逐像素熵
  entropy.png             熵热图预览
  sample_result.json      与 /api/infer 返回结构一致的样例结果
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_DIR, "src"))

import nibabel as nib  # noqa: E402
import torch  # noqa: E402

from preprocess import preprocess_slice  # noqa: E402
from infer import UNet, mc_predict  # noqa: E402

COVERAGE_THRESHOLDS = {0.70: 0.5804, 0.50: 0.3673}


def load_volume(path, dtype):
    return np.asarray(nib.load(path).dataobj, dtype=dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--case", default="case_00000")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--outdir", default=os.path.join(REPO_DIR, "sample_data"))
    ap.add_argument("--coverage", type=float, default=0.70, choices=[0.70, 0.50])
    ap.add_argument("--mc", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    threshold = COVERAGE_THRESHOLDS[args.coverage]
    case_dir = os.path.join(args.data_root, args.case)
    img_vol = load_volume(os.path.join(case_dir, "imaging.nii.gz"), np.float32)
    seg_vol = load_volume(os.path.join(case_dir, "segmentation.nii.gz"), np.uint8)

    tumor_counts = (seg_vol == 2).sum(axis=(1, 2))
    idx = int(np.argmax(tumor_counts))
    if tumor_counts[idx] == 0:
        raise SystemExit(f"{args.case} 没有肿瘤标注，换个病例再试")

    raw = img_vol[idx]
    pre = preprocess_slice(raw)

    model = UNet(dropout_rate=0.1)
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    img_t = torch.as_tensor(pre).unsqueeze(0)
    _, mean_pred, _, entropy, pred_binary = mc_predict(model, img_t, args.mc, device)

    tumor = mean_pred == 2
    if int(tumor.sum()) == 0:
        decision, tumor_ent = "no_tumor", None
    else:
        tumor_ent = float(entropy[tumor].mean())
        decision = "refer" if tumor_ent >= threshold else "accept"

    os.makedirs(args.outdir, exist_ok=True)
    np.save(os.path.join(args.outdir, "slice_hounsfield.npy"), raw)
    np.save(os.path.join(args.outdir, "slice_preprocessed.npy"), pre)
    np.save(os.path.join(args.outdir, "mean_pred.npy"), mean_pred.astype(np.uint8))
    np.save(os.path.join(args.outdir, "pred_binary.npy"), pred_binary.astype(np.uint8))
    np.save(os.path.join(args.outdir, "entropy.npy"), entropy.astype(np.float32))

    try:
        from PIL import Image
        Image.fromarray((pre * 255).astype(np.uint8)).save(os.path.join(args.outdir, "slice.png"))
        Image.fromarray((pred_binary * 255).astype(np.uint8)).save(os.path.join(args.outdir, "mask.png"))
        Image.fromarray((entropy / (entropy.max() + 1e-8) * 255).astype(np.uint8)).save(os.path.join(args.outdir, "entropy.png"))
    except ImportError:
        pass

    result = {
        "case": args.case,
        "slice_index_in_volume": idx,
        "tumor_mean_entropy": tumor_ent,
        "threshold": threshold,
        "coverage": args.coverage,
        "decision": decision,
    }
    with open(os.path.join(args.outdir, "sample_result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
