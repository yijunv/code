# -*- coding: utf-8 -*-
"""
KiTS19 贝叶斯 UNet 小样本实验 · 一键复现脚本
=====================================================================
从 yijunv/code 的主 notebook（总模型代码进行中-修复版.ipynb）抽取并整理，
口径与原仓库逐行一致：seed=42、按病例 70/15/15 划分、HU[-200,300]、256x256、
2D U-Net(base=32)、CE+Dice、Adam(lr=1e-4)、ReduceLROnPlateau、batch=8、epoch=50、
MC Dropout N=10。额外补上被删掉的普通 UNet 对照组（DROPOUT_RATE=0）。

用法
---------------------------------------------------------------------
1) Kaggle 全量复现三组 + 普通 UNet 对照（默认）：
    python run_experiments.py

2) 只跑贝叶斯三组：
    python run_experiments.py --variants bayes

3) 5 折交叉验证（仅 full 数据，每折 30 epoch，MC=20）：
    python run_experiments.py --cv

4) 本地 4 例冒烟（先确认逻辑能跑通）：
    python run_experiments.py --data-root <数据目录> --max-cases 4 --epochs 1 --batch-size 2

5) 中断后从断点继续训练（断点会自动保存，重跑时加 --resume 即可）：
    python run_experiments.py --data-root <数据目录> --variants bayes --fractions 1.00 --resume

产物（每个实验目录内）：
    manifest.json            病例划分（契约 C，可复现）
    best_unet.pth            最优模型权重
    checkpoint.pt            断点（模型+优化器+调度器+AMP 状态+RNG，--resume 续训用）
    evaluation_metrics.csv   测试集指标
    metrics.json             实验信息 + 最优验证指标 + 测试集指标
    samples/sample_XXXX.npz  贝叶斯 MC 输出（mean/pred/pred_binary/variance/entropy/label）
    tumor_entropy_scores.csv 逐切片肿瘤熵 + Dice/IoU（画 Risk-Coverage 曲线用）
"""
import argparse
import csv
import glob
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

try:
    import nibabel as nib
except ImportError:
    nib = None
    print("[警告] 未安装 nibabel，请先 pip install nibabel>=5.0")

# ---------------- 固定随机种子 ----------------
SEED = 42
HU_MIN, HU_MAX = -200, 300
IMG_SIZE = 256
NUM_CLASSES = 3
TRAIN_RATIO, VAL_RATIO = 0.70, 0.15


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------- 病例扫描 / 划分 / 降采样（与 downsample.py 一致） ----------------
def _find_volume(case_dir, name):
    for ext in (".nii", ".nii.gz"):
        p = os.path.join(case_dir, name + ext)
        if os.path.isfile(p):
            return p
    return None


def _load_volume(path, dtype):
    img = nib.load(path)
    try:
        return np.asarray(img.dataobj, dtype=dtype)
    except Exception:
        return img.get_fdata().astype(dtype)


def scan_valid_cases(data_root, max_cases=None):
    names = []
    for entry in sorted(os.listdir(data_root)):
        case_dir = os.path.join(data_root, entry)
        if not (entry.startswith("case_") and os.path.isdir(case_dir)):
            continue
        if _find_volume(case_dir, "imaging") and _find_volume(case_dir, "segmentation"):
            names.append(entry)
    if max_cases is not None and len(names) > max_cases:
        names = names[:max_cases]
    return names


def split_cases(case_dirs, train_ratio, val_ratio, seed):
    rng = random.Random(seed)
    cases = list(case_dirs)
    rng.shuffle(cases)
    n = len(cases)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return cases[:n_train], cases[n_train:n_train + n_val], cases[n_train + n_val:]


def downsample_train(train_cases, fraction, seed):
    rng = random.Random(seed)
    shuffled = list(train_cases)
    rng.shuffle(shuffled)
    n = int(len(shuffled) * fraction)
    if n == 0 and shuffled:
        n = 1
    return shuffled[:n]


# ---------------- 数据集 ----------------
class KiTS19SliceDataset(Dataset):
    """提取 2D 轴向切片，只保留含肾脏/肿瘤标注的非空切片，并常驻内存。"""

    def __init__(self, case_dirs, img_size=IMG_SIZE, hu_min=HU_MIN, hu_max=HU_MAX, transform=None):
        self.img_size = img_size
        self.transform = transform
        self.images, self.masks = [], []
        self._preload(case_dirs, hu_min, hu_max)
        self.tumor_flags = np.array([bool((m == 2).any()) for m in self.masks], dtype=bool)

    def _preload(self, case_dirs, hu_min, hu_max):
        total = len(case_dirs)
        for i, case_dir in enumerate(case_dirs, 1):
            print(f"  [加载 {i}/{total}] {os.path.basename(case_dir)}", flush=True)
            img_vol = _load_volume(_find_volume(case_dir, "imaging"), np.float32)
            seg_vol = _load_volume(_find_volume(case_dir, "segmentation"), np.uint8)
            nonempty = np.where(seg_vol.sum(axis=(1, 2)) > 0)[0]
            for idx in nonempty:
                img = img_vol[idx]
                mask = seg_vol[idx]
                img = np.clip(img, hu_min, hu_max)
                img = (img - hu_min) / (hu_max - hu_min)
                img_t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
                mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
                img_t = F.interpolate(img_t, size=(self.img_size, self.img_size),
                                      mode="bilinear", align_corners=False)
                mask_t = F.interpolate(mask_t, size=(self.img_size, self.img_size), mode="nearest")
                self.images.append(img_t.squeeze(0).squeeze(0).numpy())
                self.masks.append(mask_t.squeeze(0).squeeze(0).numpy().astype(np.uint8))
        print(f"  切片预加载完成: {len(self.images)} 个有效切片", flush=True)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = torch.from_numpy(self.images[i]).unsqueeze(0)
        mask = torch.from_numpy(self.masks[i]).long()
        if self.transform is not None:
            img, mask = self.transform(img, mask)
        return img, mask


class TrainAugmentation:
    def __call__(self, img, mask):
        if random.random() < 0.5:
            img, mask = torch.flip(img, dims=[-1]), torch.flip(mask, dims=[-1])
        if random.random() < 0.5:
            img, mask = torch.flip(img, dims=[-2]), torch.flip(mask, dims=[-2])
        if random.random() < 0.5:
            k = random.choice([1, 2, 3])
            img, mask = torch.rot90(img, k, dims=[-2, -1]), torch.rot90(mask, k, dims=[-2, -1])
        return img, mask


# ---------------- 模型 ----------------
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


# ---------------- 损失 / 优化器 ----------------
# 类别权重（背景/肾脏/肿瘤）：肿瘤又小又少，放大权重逼模型重视肿瘤；肿瘤权重可在 3~20 间调。
CLASS_WEIGHTS = [0.5, 1.0, 8.0]


class SoftDiceLoss(nn.Module):
    def __init__(self, num_classes, class_weights=None, eps=1e-6):
        super().__init__()
        self.num_classes, self.eps = num_classes, eps
        self.class_weights = None if class_weights is None else torch.tensor(class_weights)

    def forward(self, logits, target):
        probs = torch.softmax(logits, dim=1)
        oh = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        inter = torch.sum(probs * oh, (0, 2, 3))
        union = torch.sum(probs + oh, (0, 2, 3))
        dice = (2 * inter + self.eps) / (union + self.eps)
        if self.class_weights is not None:
            w = self.class_weights.to(logits.device)
            dice = (dice * w).sum() / w.sum()
        else:
            dice = dice.mean()
        return 1.0 - dice


class FocalLoss(nn.Module):
    """Focal loss：把训练注意力放到“错得厉害/难分”的像素上，gamma=0 时退化为交叉熵。"""

    def __init__(self, num_classes, gamma=2.0, weight=None, eps=1e-8):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.eps = eps
        if weight is not None:
            self.register_buffer("weight", torch.tensor(weight, dtype=torch.float32))
        else:
            self.weight = None

    def forward(self, logits, target):
        weight = None if self.weight is None else self.weight.to(logits.device)
        ce = F.cross_entropy(logits, target, weight=weight, reduction="none")
        pt = torch.exp(-ce)
        return (((1.0 - pt) ** self.gamma) * ce).mean()


class CombinedLoss(nn.Module):
    def __init__(self, num_classes, focal_gamma=2.0):
        super().__init__()
        self.focal = FocalLoss(num_classes, gamma=focal_gamma, weight=CLASS_WEIGHTS)
        self.dice = SoftDiceLoss(num_classes, class_weights=CLASS_WEIGHTS)

    def forward(self, logits, target):
        return self.focal(logits, target) + self.dice(logits, target)


def build_optimizer(model, lr=1e-4, focal_gamma=2.0):
    criterion = CombinedLoss(NUM_CLASSES, focal_gamma=focal_gamma)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    return criterion, optimizer, scheduler


# ---------------- 断点续训 ----------------
CHECKPOINT_FILENAME = "checkpoint.pt"
_RESUME_REQUIRED_META = (
    "experiment", "seed", "epochs", "dropout_rate", "learning_rate",
    "train_cases", "val_cases", "test_cases",
)


def checkpoint_path(out_dir):
    return os.path.join(out_dir, CHECKPOINT_FILENAME)


def save_checkpoint(path, state):
    """原子写入断点：先写临时文件再替换，避免中断时留下半截文件。"""
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path):
    """读取断点；不存在或损坏时返回 None，由调用方降级为从头训练。"""
    if not os.path.isfile(path):
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001 - 断点损坏不应中断训练
        print(f"  [警告] 断点文件损坏，无法续训，将重新训练：{exc}", flush=True)
        return None


def checkpoint_compatible(state, meta):
    """断点与当前实验配置/数据划分是否一致，防止用错实验的断点续训。"""
    saved = state.get("meta") if isinstance(state, dict) else None
    if not isinstance(saved, dict):
        return False
    return all(saved.get(key) == meta.get(key) for key in _RESUME_REQUIRED_META)


def capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state):
    if not state:
        return
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    if state.get("torch") is not None:
        torch.set_rng_state(state["torch"])
    if state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def build_checkpoint(epoch, best_val_dice, history, model, optimizer, scheduler, scaler, meta):
    return {
        "epoch": int(epoch),
        "best_val_dice": float(best_val_dice),
        "history": history,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "rng_state": capture_rng_state(),
        "meta": meta,
    }


# ---------------- 训练 / 验证 ----------------
def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    running = 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            loss = criterion(model(imgs), masks)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        running += loss.item() * imgs.size(0)
    return running / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running, inter_sum, psum_sum, tsum_sum = 0.0, 0.0, 0.0, 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        running += criterion(logits, masks).item() * imgs.size(0)
        preds = torch.argmax(logits, dim=1)
        pt, tt = preds == 2, masks == 2
        inter_sum += (pt & tt).sum().item()
        psum_sum += pt.sum().item()
        tsum_sum += tt.sum().item()
    denom = psum_sum + tsum_sum
    dice = (2 * inter_sum + 1e-6) / (denom + 1e-6) if denom > 0 else 1.0
    iou = (inter_sum + 1e-6) / (denom - inter_sum + 1e-6) if denom > 0 else 1.0
    return {"loss": running / len(loader.dataset), "dice": dice, "iou": iou}


def compute_tumor_dice_iou(pred, target, eps=1e-6):
    pt = pred == 2
    tt = target == 2
    inter = (pt & tt).sum().item()
    psum, tsum = pt.sum().item(), tt.sum().item()
    if psum == 0 and tsum == 0:
        return 1.0, 1.0
    dice = (2 * inter + eps) / (psum + tsum + eps)
    iou = (inter + eps) / (psum + tsum - inter + eps)
    return float(dice), float(iou)


def train_experiment(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs, out_dir,
                     meta=None, resume_state=None, checkpoint_every=1):
    history = {"train_loss": [], "val_dice": []}
    start_epoch = 1
    best_val_dice = -1.0
    best_ckpt = os.path.join(out_dir, "best_unet.pth")
    ckpt_path = checkpoint_path(out_dir)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    if resume_state is not None:
        model.load_state_dict(resume_state["model_state"])
        optimizer.load_state_dict(resume_state["optimizer_state"])
        scheduler.load_state_dict(resume_state["scheduler_state"])
        scaler.load_state_dict(resume_state["scaler_state"])
        start_epoch = int(resume_state["epoch"]) + 1
        best_val_dice = float(resume_state["best_val_dice"])
        if resume_state.get("history"):
            history = resume_state["history"]
        restore_rng_state(resume_state.get("rng_state"))
        print(f"  [续训] 从 epoch {start_epoch} 继续，best_val_dice={best_val_dice:.4f}", flush=True)

    meta = meta or {}
    for epoch in range(start_epoch, epochs + 1):
        tl = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        history["train_loss"].append(tl)
        if len(val_loader) > 0:
            vm = evaluate(model, val_loader, criterion, device)
            scheduler.step(vm["loss"])
            history["val_dice"].append(vm["dice"])
            print(f"  Epoch {epoch:02d}/{epochs} | train_loss={tl:.4f} | val_dice={vm['dice']:.4f}")
            if vm["dice"] > best_val_dice:
                best_val_dice = vm["dice"]
                torch.save(model.state_dict(), best_ckpt)
        else:
            print(f"  Epoch {epoch:02d}/{epochs} | train_loss={tl:.4f} | (无验证集)")
        if checkpoint_every > 0 and (epoch % checkpoint_every == 0 or epoch == epochs):
            save_checkpoint(ckpt_path, build_checkpoint(
                epoch, best_val_dice, history, model, optimizer, scheduler, scaler, meta))
    if best_val_dice < 0.0:
        torch.save(model.state_dict(), best_ckpt)
    return history, best_val_dice


# ---------------- MC Dropout 推理 ----------------
@torch.no_grad()
def mc_predict(model, img, n_samples, device):
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


# ---------------- 指标（新口径，与 tickets/01 锁定一致） ----------------
def eval_metrics(pred_all, target_all, entropy_all):
    """pred_all/target_all: (N,H,W) 3类 argmax；entropy_all: (N,H,W) 逐像素熵。

    主指标为全局肿瘤 Dice；拒绝/覆盖/选择性准确率只统计 GT 含肿瘤的切片；
    选择性准确率 = 被接受肿瘤阳性切片中 Dice>=0.5 的占比。
    """
    pred_tumor = (pred_all == 2).astype(np.float32)
    target_tumor = (target_all == 2).astype(np.float32)
    inter = float((pred_tumor * target_tumor).sum())
    psum = float(pred_tumor.sum())
    tsum = float(target_tumor.sum())
    global_dice = (2 * inter + 1e-6) / (psum + tsum + 1e-6) if psum + tsum > 0 else 1.0
    global_iou = (inter + 1e-6) / (psum + tsum - inter + 1e-6) if psum + tsum > 0 else 1.0
    pix_acc = float(np.mean(pred_all == target_all))

    per_ent, per_dice, ids = [], [], []
    for b in range(len(pred_all)):
        tm = target_tumor[b] == 1
        if tm.sum() == 0:
            continue
        per_ent.append(float(entropy_all[b][tm].mean()))
        d_inter = float((pred_tumor[b] * target_tumor[b]).sum())
        d_psum = float(pred_tumor[b].sum())
        d_tsum = float(target_tumor[b].sum())
        per_dice.append(2 * d_inter / (d_psum + d_tsum + 1e-8))
        ids.append(b)
    per_ent, per_dice = np.array(per_ent), np.array(per_dice)
    n_tp = int(len(per_dice))

    def rank_auc(score, label):
        """rank-sum AUC：score 能否把 label==1 排到前面（ties 用平均秩）。"""
        if n_tp < 2:
            return None
        pos = int(label.sum())
        neg = n_tp - pos
        if pos == 0 or neg == 0:
            return None
        order = np.argsort(score, kind="mergesort")
        sorted_score = score[order]
        ranks = np.empty(n_tp, dtype=float)
        i = 0
        while i < n_tp:
            j = i
            while j < n_tp and sorted_score[j] == sorted_score[i]:
                j += 1
            ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
            i = j
        return float((ranks[label].sum() - pos * (pos + 1) / 2.0) / (pos * neg))

    if n_tp > 0:
        correct = per_dice >= 0.5
        sel_all = float(correct.mean())
        order = np.argsort(per_ent, kind="mergesort")
        correct_sorted = correct[order]
        cum_correct = np.cumsum(correct_sorted).astype(float)
        sel_cov = {}
        for cov in (0.7, 0.5):
            k = max(1, min(n_tp, int(round(cov * n_tp))))
            sel_cov[cov] = float(cum_correct[k - 1] / k)
        auroc = rank_auc(per_ent, per_dice < 0.5)
        aurc = float(np.mean(1.0 - cum_correct / np.arange(1, n_tp + 1)))
    else:
        sel_all = sel_cov = {0.7: None, 0.5: None}
        auroc = aurc = None

    return {
        "dice": round(global_dice, 4),
        "iou": round(global_iou, 4),
        "pixel_acc": round(pix_acc, 4),
        "n_tumor_positive_slices": n_tp,
        "tumor_mean_entropy": round(float(per_ent.mean()), 4) if n_tp else None,
        "selective_accuracy_all": round(sel_all, 4) if sel_all is not None else None,
        "selective_accuracy_70": round(sel_cov[0.7], 4) if sel_cov[0.7] is not None else None,
        "selective_accuracy_50": round(sel_cov[0.5], 4) if sel_cov[0.5] is not None else None,
        "auroc_entropy_wrong": None if auroc is None else round(auroc, 4),
        "aurc_risk_coverage": None if aurc is None else round(aurc, 4),
    }, per_ent, per_dice, ids


# ---------------- 单实验流水线 ----------------
def run_one_experiment(args, cfg, exp_name, train_cases, val_cases, test_cases, dropout_rate):
    exp_dir = os.path.join(args.out_dir, "experiments", exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "samples"), exist_ok=True)
    print(f"\n[{exp_name}] 开始加载数据（训练 {len(train_cases)} + 验证 {len(val_cases)} + 测试 {len(test_cases)} 例）...", flush=True)

    manifest = {
        "experiment": exp_name, "data_root": args.data_root, "img_size": IMG_SIZE, "seed": SEED,
        "train_ratio": TRAIN_RATIO, "val_ratio": VAL_RATIO, "test_ratio": round(1 - TRAIN_RATIO - VAL_RATIO, 4),
        "n_train_cases": len(train_cases), "n_val_cases": len(val_cases), "n_test_cases": len(test_cases),
        "train_cases": sorted(train_cases), "val_cases": sorted(val_cases), "test_cases": sorted(test_cases),
        "dropout_rate": dropout_rate, "epochs": args.epochs, "batch_size": args.batch_size,
        "learning_rate": args.lr, "mc_samples": args.mc_samples,
        "focal_gamma": args.focal_gamma, "tumor_oversample": args.tumor_oversample,
        "pretrained": args.pretrained,
    }
    with open(os.path.join(exp_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    def paths(names):
        return [os.path.join(args.data_root, n) for n in names]

    train_ds = KiTS19SliceDataset(paths(train_cases), transform=TrainAugmentation())
    val_ds = KiTS19SliceDataset(paths(val_cases), transform=None)
    test_ds = KiTS19SliceDataset(paths(test_cases), transform=None)
    if args.tumor_oversample > 1.0:
        weights = torch.from_numpy(
            np.where(train_ds.tumor_flags, float(args.tumor_oversample), 1.0).astype(np.float64))
        sampler = WeightedRandomSampler(weights, num_samples=len(train_ds), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                                  num_workers=cfg["num_workers"], drop_last=False)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=cfg["num_workers"], drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=cfg["num_workers"])
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=cfg["num_workers"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}\n[{exp_name}] dropout={dropout_rate} | 训练 {len(train_cases)} 例 | 验证 {len(val_cases)} 例 | 测试 {len(test_cases)} 例 | device={device}\n{'='*60}")
    model = UNet(dropout_rate=dropout_rate).to(device)
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            model.load_state_dict(torch.load(args.pretrained, map_location="cpu"))
            print(f"  已加载预训练权重: {args.pretrained}", flush=True)
        else:
            print(f"  警告: 预训练权重不存在，将从随机初始化开始: {args.pretrained}", flush=True)
    criterion, optimizer, scheduler = build_optimizer(model, lr=args.lr, focal_gamma=args.focal_gamma)

    checkpoint_meta = {
        "experiment": exp_name,
        "seed": SEED,
        "epochs": args.epochs,
        "dropout_rate": dropout_rate,
        "learning_rate": args.lr,
        "train_cases": sorted(train_cases),
        "val_cases": sorted(val_cases),
        "test_cases": sorted(test_cases),
    }
    resume_state = None
    if args.resume:
        loaded = load_checkpoint(checkpoint_path(exp_dir))
        if loaded is None:
            print("  [提示] 未找到可用断点，将从 epoch 1 开始训练", flush=True)
        elif not checkpoint_compatible(loaded, checkpoint_meta):
            print("  [警告] 断点与当前配置/数据划分不一致，忽略断点，从 epoch 1 开始训练", flush=True)
        else:
            resume_state = loaded
            print(f"  [续训] 检测到断点（上次完成 epoch {loaded.get('epoch')}），继续训练", flush=True)

    t0 = time.time()
    _, best_val = train_experiment(model, train_loader, val_loader, criterion, optimizer, scheduler, device,
                                   args.epochs, exp_dir, meta=checkpoint_meta, resume_state=resume_state,
                                   checkpoint_every=args.checkpoint_every)
    print(f"  训练完成，用时 {time.time()-t0:.0f}s，最优验证 Dice={best_val:.4f}")

    model.load_state_dict(torch.load(os.path.join(exp_dir, "best_unet.pth"), map_location=device))
    model.to(device)

    # 推理 + 评估
    if dropout_rate > 0:
        entropies, labels, preds_all = [], [], []
        for idx in range(len(test_ds)):
            img, mask = test_ds[idx]
            mp, pred, var, ent, pb = mc_predict(model, img, args.mc_samples, device)
            np.savez(os.path.join(exp_dir, "samples", "sample_%04d.npz" % (idx + 1)),
                     mean=mp, pred=pred, pred_binary=pb, variance=var, entropy=ent, label=mask.numpy())
            entropies.append(ent)
            labels.append(mask.numpy())
            preds_all.append(pred)
        entropy_all = np.stack(entropies)
        pred_all = np.stack(preds_all)
        target_all = np.stack(labels)
        metrics, per_ent, per_dice, ids = eval_metrics(pred_all, target_all, entropy_all)
    else:
        model.eval()
        pred_all, target_all = [], []
        with torch.no_grad():
            for imgs, masks in test_loader:
                pred_all.append(torch.argmax(model(imgs.to(device)), dim=1).cpu().numpy())
                target_all.append(masks.numpy())
        pred_all = np.concatenate(pred_all)
        target_all = np.concatenate(target_all)
        entropy_all = np.zeros_like(target_all, dtype=np.float32)
        metrics, per_ent, per_dice, ids = eval_metrics(pred_all, target_all, entropy_all)
        metrics["tumor_mean_entropy"] = None
        metrics["selective_accuracy_all"] = None
        metrics["selective_accuracy_70"] = None
        metrics["selective_accuracy_50"] = None
        metrics["auroc_entropy_wrong"] = None
        metrics["aurc_risk_coverage"] = None

    # 逐切片肿瘤熵分数表（Risk-Coverage 用；只保留 GT 含肿瘤的切片）
    rows = []
    for j, idx in enumerate(ids):
        rows.append({"sample_id": "sample_%04d" % (idx + 1), "tumor_mean_entropy": round(float(per_ent[j]), 4),
                     "dice": round(float(per_dice[j]), 4)})
    with open(os.path.join(exp_dir, "tumor_entropy_scores.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "tumor_mean_entropy", "dice"])
        w.writeheader(); w.writerows(rows)

    with open(os.path.join(exp_dir, "evaluation_metrics.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[k for k in metrics if metrics[k] is not None])
        w.writeheader(); w.writerow({k: v for k, v in metrics.items() if v is not None})

    out_meta = {"experiment": exp_name, "best_val_dice": round(best_val, 4), "metrics": metrics,
                "n_train_cases": len(train_cases), "n_test_slices": len(test_ds), "dropout_rate": dropout_rate}
    with open(os.path.join(exp_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(out_meta, f, ensure_ascii=False, indent=2)
    print(f"  Dice={metrics['dice']} IoU={metrics['iou']} pixel_acc={metrics['pixel_acc']} | 产物 -> {exp_dir}")
    return out_meta


# ---------------- 5 折交叉验证（仅 full，bayes） ----------------
def run_cv(args, all_cases):
    from sklearn.model_selection import KFold
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_dice, fold_times = [], []
    print(f"\n{'='*60}\n5-Fold CV (full data, {args.cv_epochs} epochs, MC={args.cv_mc})\n{'='*60}")
    for fold, (tr_idx, va_idx) in enumerate(kf.split(all_cases), 1):
        t0 = time.time()
        tr = [all_cases[i] for i in tr_idx]
        va = [all_cases[i] for i in va_idx]
        train_ds = KiTS19SliceDataset([os.path.join(args.data_root, c) for c in tr], transform=TrainAugmentation())
        val_ds = KiTS19SliceDataset([os.path.join(args.data_root, c) for c in va], transform=None)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        model = UNet(dropout_rate=0.1).to(device)
        criterion, optimizer, scheduler = build_optimizer(model, lr=args.lr, focal_gamma=args.focal_gamma)
        scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
        best, ckpt = 0.0, os.path.join(args.out_dir, f"best_unet_fold{fold}.pth")
        for epoch in range(1, args.cv_epochs + 1):
            train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
            vm = evaluate(model, val_loader, criterion, device)
            scheduler.step(vm["loss"])
            if vm["dice"] > best:
                best = vm["dice"]; torch.save(model.state_dict(), ckpt)
            if epoch % 5 == 0:
                print(f"  Fold {fold} Epoch {epoch:02d} val_dice={vm['dice']:.4f}")
        # MC 评估
        model.load_state_dict(torch.load(ckpt, map_location=device)); model.to(device)
        pred_all, target_all, ent_all = [], [], []
        for imgs, masks in val_loader:
            entropies = []
            model.train()
            for _ in range(args.cv_mc):
                with torch.no_grad():
                    lp = model(imgs.to(device))
                entropies.append(torch.softmax(lp, dim=1))
            model.eval()
            mp = torch.mean(torch.stack(entropies), dim=0)
            pred_all.append(torch.argmax(mp, dim=1).cpu().numpy())
            target_all.append(masks.numpy())
            ent_all.append((-torch.sum(mp * torch.log(mp + 1e-8), dim=1)).cpu().numpy())
        pred_all = np.concatenate(pred_all); target_all = np.concatenate(target_all); ent_all = np.concatenate(ent_all)
        pt, tt = (pred_all == 2).astype(np.float32), (target_all == 2).astype(np.float32)
        inter = (pt * tt).sum(); psum, tsum = pt.sum(), tt.sum()
        fd = 2 * inter / (psum + tsum + 1e-6)
        fold_dice.append(float(fd)); fold_times.append(time.time() - t0)
        print(f"  Fold {fold} 肿瘤 Dice={fd:.4f} 用时 {fold_times[-1]/60:.1f} min")
    print(f"\n5-Fold CV 汇总: 各折 {[f'{x:.4f}' for x in fold_dice]} | 平均 {np.mean(fold_dice):.4f} ± {np.std(fold_dice):.4f}")
    pd = {"folds": [round(x, 4) for x in fold_dice], "mean": round(float(np.mean(fold_dice)), 4),
          "std": round(float(np.std(fold_dice)), 4), "total_hours": round(float(sum(fold_times)) / 3600, 2)}
    with open(os.path.join(args.out_dir, "5fold_cv_results.json"), "w", encoding="utf-8") as f:
        json.dump(pd, f, ensure_ascii=False, indent=2)
    return pd


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=None, help="KiTS19 病例根目录（含 case_* 子目录）")
    ap.add_argument("--out-dir", default="/kaggle/working" if os.path.isdir("/kaggle/working") else "output")
    ap.add_argument("--variants", default="bayes,vanilla", help="bayes / vanilla / bayes,vanilla")
    ap.add_argument("--fractions", default="0.10,0.20,1.00")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--mc-samples", type=int, default=10)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--focal-gamma", type=float, default=2.0, help="focal loss 的 gamma，0 表示退化为交叉熵")
    ap.add_argument("--tumor-oversample", type=float, default=1.0, help="肿瘤切片过采样权重，>1 时启用（如 3.0）")
    ap.add_argument("--pretrained", default=None, help="预训练权重 .pth，用于迁移学习初始化（如 datafull/best_unet.pth）")
    ap.add_argument("--resume", action="store_true", help="若实验目录内存在断点 checkpoint.pt，则从断点继续训练")
    ap.add_argument("--checkpoint-every", type=int, default=1, help="每隔多少个 epoch 保存一次断点（0 表示不保存）")
    ap.add_argument("--max-cases", type=int, default=None)
    ap.add_argument("--cv", action="store_true")
    ap.add_argument("--cv-epochs", type=int, default=30)
    ap.add_argument("--cv-mc", type=int, default=20)
    args = ap.parse_args()
    args.out_dir = os.path.abspath(args.out_dir)

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"device={device} | seed={SEED} | epochs={args.epochs} | batch={args.batch_size} | lr={args.lr}")

    # 自动探测数据目录
    if args.data_root is None:
        cand = "/kaggle/input/datasets/user123454321/kits19-1"
        if not os.path.isdir(cand):
            cand = None
            for d in glob.glob("/kaggle/input/*"):
                for sub in [d] + [os.path.join(d, e) for e in os.listdir(d) if os.path.isdir(os.path.join(d, e))]:
                    if os.path.isdir(sub) and any(e.startswith("case_") for e in os.listdir(sub)):
                        cand = sub; break
                if cand: break
        if cand is None:
            sys.exit("找不到 KiTS19 数据目录，请用 --data-root 指定")
        args.data_root = cand
    print(f"数据目录: {args.data_root}")

    all_cases = scan_valid_cases(args.data_root, args.max_cases)
    print(f"有效病例数: {len(all_cases)}")
    train_cases, val_cases, test_cases = split_cases(all_cases, TRAIN_RATIO, VAL_RATIO, SEED)
    print(f"全量划分: 训练 {len(train_cases)} | 验证 {len(val_cases)} | 测试 {len(test_cases)}（冻结）")

    if args.cv:
        run_cv(args, all_cases)
        return

    variants = [v.strip() for v in args.variants.split(",")]
    fractions = [float(x) for x in args.fractions.split(",")]
    num_workers = args.num_workers
    cfg = {"num_workers": num_workers}
    summary = []
    for variant in variants:
        rate = 0.1 if variant == "bayes" else 0.0
        for frac in fractions:
            if frac >= 1.0:
                name = "datafull" if variant == "bayes" else "vanillafull"
                sub = train_cases
            else:
                sub = downsample_train(train_cases, frac, SEED)
                name = f"data{int(frac*100)}pct" if variant == "bayes" else f"vanilla{int(frac*100)}pct"
            meta = run_one_experiment(args, cfg, name, sub, val_cases, test_cases, rate)
            summary.append(meta)

    print("\n" + "=" * 70)
    print("实验汇总")
    print("=" * 70)
    for s in summary:
        m = s["metrics"]
        print(f"{s['experiment']:16s} | 训练{s['n_train_cases']:3d}例 | best_val_dice={s['best_val_dice']:.4f} | Dice={m['dice']} IoU={m['iou']} pix_acc={m['pixel_acc']} | sel70={m.get('selective_accuracy_70')} auroc={m.get('auroc_entropy_wrong')}")
    with open(os.path.join(args.out_dir, "experiments", "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("汇总已保存: " + os.path.join(args.out_dir, "experiments", "summary.json"))


if __name__ == "__main__":
    main()
