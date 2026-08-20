import numpy as np
import torch
from scipy.stats import pearsonr, ttest_rel
from sklearn.metrics import brier_score_loss
import warnings
warnings.filterwarnings("ignore")
# 项目统一固定随机种子（申报书实验协议要求seed=42）
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
# 1. 基础分割指标 Dice / IoU / 像素准确率（CT肿瘤分割用）
def dice_coefficient(pred_mask: np.ndarray, target_mask: np.ndarray, eps: float = 1e-6) -> float:
    intersection = np.sum(pred_mask * target_mask, axis=(1, 2))
    union = np.sum(pred_mask, axis=(1, 2)) + np.sum(target_mask, axis=(1, 2))
    dice = (2 * intersection + eps) / (union + eps)
    return float(np.mean(dice))

def iou_score(pred_mask: np.ndarray, target_mask: np.ndarray, eps: float = 1e-6) -> float:
    intersection = np.sum(pred_mask * target_mask, axis=(1, 2))
    total = np.sum(pred_mask, axis=(1, 2)) + np.sum(target_mask, axis=(1, 2))
    union = total - intersection
    iou = (intersection + eps) / (union + eps)
    return float(np.mean(iou))

def pixel_accuracy(pred_mask: np.ndarray, target_mask: np.ndarray) -> float:
    correct = np.sum(pred_mask == target_mask)
    total = pred_mask.size
    return float(correct / total)

# 2. 不确定性指标（申报书核心：熵、拒绝率、覆盖率）
def tumor_mean_entropy(entropy_map: np.ndarray, tumor_mask: np.ndarray) -> float:
    batch_mean = []
    for b in range(entropy_map.shape[0]):
        tumor_pixels = entropy_map[b][tumor_mask[b] == 1]
        batch_mean.append(np.mean(tumor_pixels) if len(tumor_pixels) > 0 else 0.0)
    return float(np.mean(batch_mean))

def rejection_coverage_acc(entropy_map: np.ndarray, tumor_mask: np.ndarray, pred_mask: np.ndarray, target_mask: np.ndarray, threshold: float):
    N = entropy_map.shape[0]
    rej_count = 0
    select_dice_list = []
    for b in range(N):
        tumor_ent = np.mean(entropy_map[b][tumor_mask[b]==1]) if np.sum(tumor_mask[b])>0 else 0
        if tumor_ent > threshold:
            rej_count += 1
        else:
            d = dice_coefficient(pred_mask[b:b+1], target_mask[b:b+1])
            select_dice_list.append(d)
    rej_rate = rej_count / N
    coverage = 1 - rej_rate
    select_acc = np.mean(select_dice_list) if len(select_dice_list) > 0 else 0.0
    return rej_rate, coverage, select_acc

# 3. 校准指标 ECE（论文评估指标）
def expected_calibration_error(uncertainty_vals: np.ndarray, seg_acc: np.ndarray, bins=10):
    bin_edges = np.linspace(0, 1, bins+1)
    ece = 0.0
    for i in range(bins):
        mask = (uncertainty_vals >= bin_edges[i]) & (uncertainty_vals < bin_edges[i+1])
        if np.sum(mask) == 0:
            continue
        bin_size = np.sum(mask)
        mean_conf = np.mean(1 - uncertainty_vals[mask])
        mean_acc = np.mean(seg_acc[mask])
        ece += (bin_size / len(uncertainty_vals)) * np.abs(mean_conf - mean_acc)
    return float(ece)

# 4. 统计学检验（申报要求：Pearson、配对t检验）
def pearson_correlation(x: np.ndarray, y: np.ndarray):
    corr, p = pearsonr(x, y)
    return float(corr), float(p)

def paired_ttest(group1: np.ndarray, group2: np.ndarray):
    stat, p = ttest_rel(group1, group2)
    return float(stat), float(p)

# 统一一键计算所有指标（团队所有人统一调用入口）
def calculate_all_metrics(pred: np.ndarray, target: np.ndarray, entropy: np.ndarray, tumor_mask: np.ndarray, reject_threshold: float = 0.5):
    dice = dice_coefficient(pred, target)
    iou = iou_score(pred, target)
    pix_acc = pixel_accuracy(pred, target)
    tumor_ent = tumor_mean_entropy(entropy, tumor_mask)
    rej_rate, cover, sel_acc = rejection_coverage_acc(entropy, tumor_mask, pred, target, reject_threshold)
    res = {
        "dice": round(dice, 4),
        "iou": round(iou, 4),
        "pixel_acc": round(pix_acc, 4),
        "tumor_mean_entropy": round(tumor_ent, 4),
        "rejection_rate": round(rej_rate, 4),
        "coverage": round(cover, 4),
        "selective_accuracy": round(sel_acc, 4)
    }
    return res

# 读取模型输出npz文件工具（后期对接真实数据用）
def load_npz_sample(npz_path: str, mask_npy_path: str):
    """从 npz 读取样本预测/熵图/标签（肿瘤二值口径，兼容新旧两版 npz）"""
    data = np.load(npz_path)
    entropy_map = data["entropy"]
    if "pred_binary" in data.files:
        pred_mask = data["pred_binary"].astype(np.float32)   # 新版契约A：肿瘤二值
    else:
        pred_mask = data["pred"].astype(np.float32)          # 旧版：pred 本身即 0/1 肿瘤二值
    target_mask = np.load(mask_npy_path)
    return pred_mask[None, ...], target_mask[None, ...], entropy_map[None, ...]