# -*- coding: utf-8 -*-
"""在线推理用的预处理，口径与 scripts/run_experiments.py 的 KiTS19SliceDataset 一致。

口径：HU 裁剪 [-200, 300] -> 归一化 [0, 1] -> 双线性缩放 256x256（align_corners=False）。
"""
from __future__ import annotations

import numpy as np

HU_MIN, HU_MAX = -200, 300
IMG_SIZE = 256


def _resize(img: np.ndarray, size: int) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    t = torch.as_tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t.squeeze(0).squeeze(0).numpy()


def preprocess_slice(arr: np.ndarray) -> np.ndarray:
    """把一张 2D CT 切片（HU 值）变成模型输入。

    返回 shape=(256,256) 的 float32 数组，值域 [0,1]。
    """
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"期望 2D 切片，实际 shape={arr.shape}")
    arr = np.clip(arr, HU_MIN, HU_MAX)
    arr = (arr - HU_MIN) / (HU_MAX - HU_MIN)
    return _resize(arr, IMG_SIZE).astype(np.float32)


def load_volume_slices(path: str, slice_axis: int = 0) -> np.ndarray:
    """读 NIfTI 体积，沿给定轴切成 2D 切片（HU 值），返回 (N,H,W) float32。

    slice_axis 默认 0，与训练脚本 KiTS19SliceDataset 的轴向一致（img_vol[idx]）。
    web 端接入前请先用一张已知样例核对轴向，避免切错方向。
    """
    import nibabel as nib

    vol = np.asarray(nib.load(path).dataobj, dtype=np.float32)
    return np.moveaxis(vol, slice_axis, 0)
