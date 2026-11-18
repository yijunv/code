# -*- coding: utf-8 -*-
"""单张 CT 切片 / 整卷 NIfTI 的在线推理入口（研究原型）。

加载 full 组权重 -> 预处理 -> MC-Dropout(N=10) -> 分割/熵 -> 阈值分诊。
用法示例：
  python src/infer.py --weights best_unet.pth --image slice.npy
  python src/infer.py --weights best_unet.pth --image case.nii.gz --coverage 0.70

说明：PNG 输入丢失了 HU 刻度，只用于界面联调；真实推理请用 NIfTI 或保存 HU 值的 .npy。
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn

from preprocess import preprocess_slice, load_volume_slices

NUM_CLASSES = 3
DEFAULT_MC = 10
# full 组肿瘤平均熵的 70%/50% 分位数（见 docs/WEB_HANDOFF.md 第四节）
COVERAGE_THRESHOLDS = {0.70: 0.5804, 0.50: 0.3673}


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_rate=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=3, base_channels=32, dropout_rate=0.1):
        super().__init__()
        c1, c2, c3, c4, c5 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8, base_channels * 16
        self.enc1 = DoubleConv(in_channels, c1, dropout_rate)
        self.enc2 = DoubleConv(c1, c2, dropout_rate)
        self.enc3 = DoubleConv(c2, c3, dropout_rate)
        self.enc4 = DoubleConv(c3, c4, dropout_rate)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(c4, c5, dropout_rate)
        self.up4 = nn.ConvTranspose2d(c5, c4, 2, 2)
        self.dec4 = DoubleConv(c5, c4, dropout_rate)
        self.up3 = nn.ConvTranspose2d(c4, c3, 2, 2)
        self.dec3 = DoubleConv(c4, c3, dropout_rate)
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, 2)
        self.dec2 = DoubleConv(c3, c2, dropout_rate)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, 2)
        self.dec1 = DoubleConv(c2, c1, dropout_rate)
        self.out_conv = nn.Conv2d(c1, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.out_conv(d1)


@torch.no_grad()
def mc_predict(model, img, n_samples, device):
    """img: (1,H,W) 已归一化 [0,1]；返回五组 numpy 数组，口径与训练脚本一致。"""
    model.train()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.eval()
    img = img.unsqueeze(0).to(device)
    stack = [model(img).unsqueeze(0) for _ in range(n_samples)]
    probs = torch.softmax(torch.cat(stack, dim=0), dim=2)
    mean_probs = probs.mean(0).squeeze(0)
    mean_pred = torch.argmax(mean_probs, dim=0)
    pred_binary = (mean_pred == 2).float().cpu().numpy()
    variance = probs.var(0).squeeze(0).mean(0)
    entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8), dim=0)
    return mean_probs.cpu().numpy(), mean_pred.cpu().numpy(), variance.cpu().numpy(), entropy.cpu().numpy(), pred_binary


def decide(tumor_mean_entropy, threshold):
    return "refer" if tumor_mean_entropy >= threshold else "accept"


def infer_slice(model, img, n_samples, device, threshold):
    _, mean_pred, _, entropy, pred_binary = mc_predict(model, img, n_samples, device)
    tumor = mean_pred == 2
    if int(tumor.sum()) == 0:
        return {"decision": "no_tumor", "tumor_mean_entropy": None, "threshold": threshold,
                "mean_pred": mean_pred, "entropy": entropy, "pred_binary": pred_binary}
    tumor_ent = float(entropy[tumor].mean())
    return {"decision": decide(tumor_ent, threshold), "tumor_mean_entropy": tumor_ent,
            "threshold": threshold, "mean_pred": mean_pred, "entropy": entropy, "pred_binary": pred_binary}


def save_arrays(outdir, prefix, mean_pred, entropy, pred_binary):
    os.makedirs(outdir, exist_ok=True)
    np.save(os.path.join(outdir, f"{prefix}_mean_pred.npy"), mean_pred.astype(np.uint8))
    np.save(os.path.join(outdir, f"{prefix}_entropy.npy"), entropy.astype(np.float32))
    np.save(os.path.join(outdir, f"{prefix}_pred_binary.npy"), pred_binary.astype(np.uint8))
    try:
        from PIL import Image
        Image.fromarray((entropy / (entropy.max() + 1e-8) * 255).astype(np.uint8)).save(os.path.join(outdir, f"{prefix}_entropy.png"))
        Image.fromarray((pred_binary * 255).astype(np.uint8)).save(os.path.join(outdir, f"{prefix}_mask.png"))
    except ImportError:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="best_unet.pth 权重路径")
    ap.add_argument("--image", required=True, help="单张切片 .npy/.png 或整卷 .nii/.nii.gz")
    ap.add_argument("--coverage", type=float, default=0.70, choices=[0.70, 0.50])
    ap.add_argument("--mc", type=int, default=DEFAULT_MC)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--outdir", default="infer_out")
    ap.add_argument("--slice-axis", type=int, default=0, help="NIfTI 切片轴，默认与训练一致")
    args = ap.parse_args()

    threshold = COVERAGE_THRESHOLDS[args.coverage]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = UNet(dropout_rate=0.1)
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.to(device)

    if args.image.lower().endswith((".nii", ".nii.gz")):
        vols = load_volume_slices(args.image, slice_axis=args.slice_axis)
    elif args.image.lower().endswith(".npy"):
        arr = np.load(args.image).astype(np.float32)
        vols = arr[None, ...] if arr.ndim == 2 else arr
    else:
        from PIL import Image
        arr = np.asarray(Image.open(args.image).convert("L"), dtype=np.float32)
        vols = arr[None, ...]

    results = []
    for i, sl in enumerate(vols):
        img = preprocess_slice(sl)
        img_t = torch.as_tensor(img, dtype=torch.float32).unsqueeze(0)
        r = infer_slice(model, img_t, args.mc, device, threshold)
        save_arrays(args.outdir, f"slice_{i}", r["mean_pred"], r["entropy"], r["pred_binary"])
        results.append({
            "slice_index": i,
            "tumor_mean_entropy": r["tumor_mean_entropy"],
            "threshold": threshold,
            "coverage": args.coverage,
            "decision": r["decision"],
        })

    print(json.dumps({"status": "ok", "slices": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
