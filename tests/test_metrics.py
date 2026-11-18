import numpy as np
import csv
from src.metrics.unified_metrics import calculate_all_metrics, pearson_correlation

if __name__ == "__main__":
    # 模拟CT图像数据，用来自测代码，不用真实数据集
    batch_size = 8
    H, W = 128, 128
    pred_mask = np.random.randint(0, 2, size=(batch_size, H, W))
    target_mask = np.random.randint(0, 2, size=(batch_size, H, W))
    tumor_mask = target_mask.copy()
    entropy_map = np.random.rand(batch_size, H, W)

    # 一键计算全套指标
    metrics_result = calculate_all_metrics(
        pred=pred_mask,
        target=target_mask,
        entropy=entropy_map,
        tumor_mask=tumor_mask,
        reject_threshold=0.45
    )
    print("==== 项目统一指标计算结果 ====")
    for k, v in metrics_result.items():
        print(f"{k:22s}: {v}")

    # 测试论文统计学相关性检验
    ent_list = np.array([metrics_result["tumor_mean_entropy"]]*batch_size) + np.random.randn(batch_size)*0.01
    dice_list = np.array([metrics_result["dice"]]*batch_size) + np.random.randn(batch_size)*0.02
    corr, p = pearson_correlation(ent_list, dice_list)
    print(f"\n相关性检验：相关系数={corr:.4f}，p值={p:.4f}")

    # 自动导出标准表格，后续实验统一格式
    with open("实验指标汇总.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics_result.keys()))
        writer.writeheader()
        writer.writerow(metrics_result)
    print("\n已生成标准指标表格：实验指标汇总.csv")
    print(" 统一指标库自测完成，无报错即可交付团队使用！")