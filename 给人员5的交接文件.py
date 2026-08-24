# 给人员5的交接文件（最终版）
# 包含：tumor_entropy_scores.csv + 推荐阈值 = 0.15
# 阈值依据：信度检验中高精度组（Dice≥0.8）平均熵0.099 vs 低精度组（Dice<0.5）平均熵0.209
import numpy as np
import pandas as pd
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.metrics.unified_metrics import calculate_all_metrics
# 配置
BASE_DIR = r"E:\experiments"
SAMPLES_DIR = os.path.join(BASE_DIR, "datafull", "samples")
OUTPUT_PATH = os.path.join(BASE_DIR, "tumor_entropy_scores.csv")
RECOMMENDED_THRESHOLD = 0.15  # 推荐阈值，基于信度检验结果
print("=" * 60)
print("人员5交接文件生成器")
print("=" * 60)
# 读取所有样本
npz_files = sorted([f for f in os.listdir(SAMPLES_DIR) if f.endswith(".npz")])
print(f"找到 {len(npz_files)} 个样本文件")
records = []
entropy_list = []
dice_list = []
for f in npz_files:
    npz_path = os.path.join(SAMPLES_DIR, f)
    data = np.load(npz_path)
    pred = data["pred"].astype(np.float32)
    label = data["label"].astype(np.float32)
    entropy = data["entropy"]
    # 计算 Dice
    intersection = (pred * label).sum()
    dice = 2 * intersection / (pred.sum() + label.sum() + 1e-8)
    # 计算 IoU
    union = pred.sum() + label.sum() - intersection
    iou = intersection / (union + 1e-8)
    # 肿瘤区域平均熵
    tumor_mask = (label == 1)
    if tumor_mask.sum() > 0:
        tumor_mean_entropy = entropy[tumor_mask].mean()
    else:
        tumor_mean_entropy = 0
    entropy_list.append(tumor_mean_entropy)
    dice_list.append(dice)
    records.append({
        "sample_id": f.replace(".npz", ""),
        "tumor_mean_entropy": tumor_mean_entropy,
        "dice": dice,
        "iou": iou
    })
df = pd.DataFrame(records)
# 保存 CSV
df.to_csv(OUTPUT_PATH, index=False)
print(f"\n✅ 已生成: {OUTPUT_PATH}")
print(f"   共 {len(df)} 个样本")
# 统计摘要
print("\n" + "=" * 60)
print("【数据摘要】")
print("=" * 60)
print(f"  平均 Dice: {df['dice'].mean():.4f} ± {df['dice'].std():.4f}")
print(f"  平均 IoU: {df['iou'].mean():.4f} ± {df['iou'].std():.4f}")
print(f"  平均肿瘤熵: {df['tumor_mean_entropy'].mean():.4f}")
# 熵值分布
low_entropy = df[df['tumor_mean_entropy'] <= 0.1]
mid_entropy = df[(df['tumor_mean_entropy'] > 0.1) & (df['tumor_mean_entropy'] <= 0.2)]
high_entropy = df[df['tumor_mean_entropy'] > 0.2]
print(f"\n  熵值分布:")
print(f"    熵 ≤ 0.1: {len(low_entropy)} 个样本, 平均 Dice = {low_entropy['dice'].mean():.4f}")
print(f"    0.1 < 熵 ≤ 0.2: {len(mid_entropy)} 个样本, 平均 Dice = {mid_entropy['dice'].mean():.4f}")
print(f"    熵 > 0.2: {len(high_entropy)} 个样本, 平均 Dice = {high_entropy['dice'].mean():.4f}")
# 阈值说明（基于信度检验结果）
print("\n" + "=" * 60)
print("【推荐阈值】")
print("=" * 60)
print(f"  推荐初始阈值: {RECOMMENDED_THRESHOLD:.2f}")
print("\n  阈值依据：")
print("    - 高精度组 (Dice ≥ 0.8) 平均熵 = 0.0992")
print("    - 低精度组 (Dice < 0.5) 平均熵 = 0.2087")
print(f"    - 推荐阈值 0.15 为两者的中间值，兼顾敏感性与特异性")
# 按推荐阈值统计
accept_mask = df['tumor_mean_entropy'] <= RECOMMENDED_THRESHOLD
reject_mask = df['tumor_mean_entropy'] > RECOMMENDED_THRESHOLD
print(f"\n  按阈值 {RECOMMENDED_THRESHOLD:.2f} 划分:")
print(f"    接受样本: {accept_mask.sum()} 个 ({accept_mask.sum()/len(df)*100:.1f}%)")
print(f"    拒绝样本: {reject_mask.sum()} 个 ({reject_mask.sum()/len(df)*100:.1f}%)")
if accept_mask.sum() > 0:
    acc_dice = df[accept_mask]['dice'].mean()
    print(f"    接受样本平均 Dice: {acc_dice:.4f}")
# 决策逻辑说明
print("\n" + "=" * 60)
print("【决策逻辑】")
print("=" * 60)
print("  if tumor_mean_entropy > threshold:")
print("      → 系统拒绝输出分割结果，建议转诊")
print("  else:")
print("      → 系统输出分割结果")
print(f"\n  推荐阈值: {RECOMMENDED_THRESHOLD:.2f}")
print("  注意: 阈值可在 Web 界面设置为滑动条，由医生根据临床场景调整")
print("\n" + "=" * 60)
print("✅ 人员5交接文件生成完成！")
print("=" * 60)
print(f"\n请将以下文件发送给人员5:")
print(f"  1. {OUTPUT_PATH}")
print(f"  2. 本终端输出的完整信息（含决策逻辑和阈值说明）")