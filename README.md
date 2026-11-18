# 肾癌 CT 智能分诊 · 贝叶斯 U-Net 研究原型

基于贝叶斯深度学习（MC-Dropout）的肾癌 CT 分割与不确定性分诊：模型给出肿瘤分割和逐像素熵，当肿瘤区域平均熵超过阈值时建议转诊/人工复核。

研究原型，仅供科研演示，不构成临床诊断结论。

## 目录结构

- `src/`：在线推理入口。`preprocess.py`（HU 裁剪、归一化、缩放）、`infer.py`（模型加载 + MC 推理 + 阈值分诊）。
- `scripts/`：训练/评估/画图脚本。`run_experiments.py` 是训练主脚本，`verify_metrics.py` 是四组指标自检，`make_sample_data.py` 生成脱敏样例。
- `results/`：四组 bayes（10%/20%/30%/full）+ 三组 vanilla 的指标、逐切片熵+Dice、统计检验结果。
- `assets/figures/`：Risk-Coverage、Dice 对比、可靠性、覆盖图、数据标度图（png + svg）。
- `sample_data/`：一张脱敏样例切片及其预期输出，用于对答案。
- `tests/`：回归测试。
- `docs/`：`WEB_HANDOFF.md`（web 端交接）、`PAPER_HANDOFF.md`（论文端交接）。

## 快速开始

1. `pip install -r requirements.txt`（torch 按自己机器装 CPU/CUDA 对应版本）。
2. 自检指标：`python scripts/verify_metrics.py`，应打印四组且全部一致。
3. 看结果：打开 `results/` 和 `assets/figures/`。
4. 跑推理（拿到权重后）：`python src/infer.py --weights best_unet.pth --image sample_data/slice_preprocessed.npy`。

## 结果速览

| 训练比例 | 训练病例数 | 肿瘤 Dice | 熵 AUROC |
|---|---|---|---|
| 10% | 14 | 0.6749 | 0.4878 |
| 20% | 29 | 0.6877 | 0.5492 |
| 30% | 44 | 0.7121 | 0.6013 |
| full | 147 | 0.748 | 0.5562 |

更多数字看 `results/stats_validation_results.json` 和各组 `metrics.json`。

## 权重

四组 `best_unet.pth` 不进仓库，由项目方私下发给需要的人；web 端真实推理用 full 这组。

## 安全边界

- 不放 KiTS19 原始影像、不放真实患者数据，只放脱敏样例和结果。
- 不放凭证、手机号、微信 ID 等个人信息。
- 页面与文档需标注「研究原型，非临床诊断」。
