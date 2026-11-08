# GitHub 文件导出清单（GITHUB_EXPORT_MANIFEST）

> 本清单用于下一轮实际推送前的核对。本轮只列出，不执行任何 git 操作。大小以当前实测为准（约值）。敏感等级：公开 / 内部 / 禁止。

## 可直接提交（公开，非敏感）

| 文件或目录 | 当前绝对路径 | 建议仓库相对路径 | 类型 | 大小 | 用途 | 是否必需 | 推送方式 | 数据敏感等级 |
|---|---|---|---|---|---|---|---|---|
| 训练/评估主脚本 | `D:\codex工作空间\run_experiments_new.py` | `scripts/run_experiments.py` | 代码 | 38 KB | 复现训练与评估 | 是 | Git | 公开 |
| 指标自检 | `D:\codex工作空间\verify_metrics.py` | `scripts/verify_metrics.py` | 代码 | 2 KB | 复算指标 | 是 | Git | 公开 |
| 统计检验 | `D:\codex工作空间\stats_validation.py` | `scripts/stats_validation.py` | 代码 | 3 KB | 熵效度检验 | 是 | Git | 公开 |
| MC 采样研究 | `D:\codex工作空间\mc_sampling_study.py` | `scripts/mc_sampling_study.py` | 代码 | 4 KB | 阈值/采样搜索 | 是 | Git | 公开 |
| 画图脚本 | `D:\codex工作空间\plot_results.py`、`plot_reliability.py`、`plot_overlays.py`、`figstyle.py` | `scripts/` | 代码 | 4.5/1.3/2.8/0.9 KB | 重新出图 | 是 | Git | 公开 |
| 回归测试 | `D:\codex工作空间\test_eval_metrics.py`、`test_training.py`、`test_mc_study.py`、`test_stats_validation.py` | `tests/` | 代码 | <1 KB 各 | 单元测试 | 是 | Git | 公开 |
| 三组 bayes 指标 | `D:\大创代码\output_ft\experiments\{data10pct,data20pct,datafull}\metrics.json` | `results/bayes/{data10pct,data20pct,datafull}/metrics.json` | JSON | <1 KB | 主指标 | 是 | Git | 公开 |
| 三组 bayes 逐切片 | 同上 `tumor_entropy_scores.csv` | `results/bayes/.../tumor_entropy_scores.csv` | CSV | ~22 KB/组 | 画曲线 | 是 | Git | 公开 |
| 三组 bayes 指标单行 | 同上 `evaluation_metrics.csv` | `results/bayes/.../evaluation_metrics.csv` | CSV | <1 KB | 附录 | 是 | Git | 公开 |
| 三组 vanilla 指标 | `D:\大创代码\output_ft_vanilla\experiments\{vanilla10pct,vanilla20pct,vanillafull}\metrics.json` | `results/vanilla/.../metrics.json` | JSON | <1 KB | 对照 | 是 | Git | 公开 |
| 统计结果 | `D:\codex工作空间\stats_validation_results.json` | `results/stats_validation_results.json` | JSON | 3 KB | 统计证据 | 是 | Git | 公开 |
| 阈值/采样结果 | `D:\codex工作空间\mc_sampling_results.json` | `results/mc_sampling_results.json` | JSON | 1 KB | 阈值来源 | 是 | Git | 公开 |
| 曲线数据 | `D:\codex工作空间\figures\risk_coverage_data.csv` | `results/risk_coverage_data.csv` | CSV | 60 KB | 曲线底层数据 | 是 | Git | 公开 |
| Risk-Coverage 图 | `D:\codex工作空间\figures\risk_coverage.png/.svg` | `assets/figures/risk_coverage.png/.svg` | 图 | 0.7 MB | 论文配图 | 是 | Git | 公开 |
| Dice 对比图 | `D:\codex工作空间\figures\dice_comparison.png/.svg` | `assets/figures/dice_comparison.png/.svg` | 图 | 0.04 MB | 论文配图 | 是 | Git | 公开 |
| 可靠性图 | `D:\codex工作空间\figures\reliability.png/.svg` | `assets/figures/reliability.png/.svg` | 图 | 0.1 MB | 论文配图 | 是 | Git | 公开 |
| 覆盖图/热图 | `D:\codex工作空间\figures\overlay_low_uncertainty.png/.svg`、`overlay_high_uncertainty.png/.svg` | `assets/figures/` | 图 | 0.3 MB 各 | 示例图 | 是 | Git | 公开 |
| Web 交接文档 | `D:\codex工作空间\docs\WEB_HANDOFF.md` | `docs/WEB_HANDOFF.md` | 文档 | ~30 KB | Web 交接 | 是 | Git | 公开 |
| GitHub 清单 | `D:\codex工作空间\docs\GITHUB_EXPORT_MANIFEST.md` | `docs/GITHUB_EXPORT_MANIFEST.md` | 文档 | 本文件 | 推送核对 | 是 | Git | 公开 |
| 论文交接文档 | `D:\codex工作空间\论文同学交接文档.md` | `docs/PAPER_HANDOFF.md` | 文档 | 31 KB | 论文交接 | 是（可选） | Git | 公开 |

## 需用 Git LFS 或 GitHub Release（大文件，公开待授权）

| 文件或目录 | 当前绝对路径 | 建议仓库相对路径 | 类型 | 大小 | 用途 | 是否必需 | 推送方式 | 数据敏感等级 |
|---|---|---|---|---|---|---|---|---|
| full 模型权重 | `D:\大创代码\output_ft\experiments\datafull\best_unet.pth` | `models/datafull/best_unet.pth` | 权重 | 29.7 MB | Web 推理/复现 | 是（完整系统） | Git LFS 或 Release | 待授权 |
| 10% 模型权重 | `D:\大创代码\output_ft\experiments\data10pct\best_unet.pth` | `models/data10pct/best_unet.pth` | 权重 | 29.7 MB | 复现 | 否 | Git LFS 或 Release | 待授权 |
| 20% 模型权重 | `D:\大创代码\output_ft\experiments\data20pct\best_unet.pth` | `models/data20pct/best_unet.pth` | 权重 | 29.7 MB | 复现 | 否 | Git LFS 或 Release | 待授权 |

> 三组权重合计约 89 MB。GitHub 建议单个文件超过 50 MB 用 LFS；此处每个 29.7 MB 可直接提交，但为不膨胀仓库历史，建议放 Release 或 LFS，并先获得导师对“公开模型权重”的授权。

## 禁止推送

| 文件或目录 | 当前绝对路径 | 原因 | 数据敏感等级 |
|---|---|---|---|
| KiTS19 原始数据 | `D:\大创代码\kits19\datasets`（26.9 GB，510 文件） | 原始影像、体积巨大，不可再分发 | 禁止 |
| 申报书 | 本机微信聊天文件临时目录（路径含个人微信 ID，公开版本已脱敏） | 含个人手机号，未获公开授权 | 禁止 |
| 论文初稿 | `D:/微信聊天文件/.../基于贝叶斯深度学习的肾癌 CT 影像智能分诊研究(3)(1).docx` | 未发表稿件，未获公开授权 | 禁止 |
| 实验原始产物 | `D:\大创代码\output_ft`、`D:\大创代码\output_ft_vanilla`、`D:\大创代码\output` | 含 samples、checkpoint、绝对路径，且体积大 | 禁止（仅抽取上表公开项） |
| venv / 缓存 / 临时 | `D:\大创代码\kits19`、`__pycache__`、`.mplcache`、`发送给论文手` 临时打包目录 | 环境与临时文件 | 禁止 |
| 任何凭证 | 无（当前未发现） | 如出现 API Key/令牌/密码，一律禁止 | 禁止 |

## 推送前安全检查（下一轮执行）

1. `rg -i "password|token|secret|api[_-]?key|手机号|phone|D:\\大创代码" <repo>` 应为空（绝对路径与个人信息零命中）。
2. 确认 `models/` 权重已走 Release/LFS 且获得公开授权。
3. 确认无真实影像、无 docx、无 venv/缓存进入暂存区。
4. 运行 `python -m unittest discover -s tests`（4 个测试通过）与 `scripts/verify_metrics.py`。

## 待补充的仓库信息

- 目标仓库地址（owner/repo，公开或私有）。
- 是否公开模型权重（三组 best_unet.pth）。
- 开源 LICENSE 选择（如 MIT / Apache-2.0，由导师决定）。
- requirements.txt 需要从 venv 冻结生成（本轮未生成，避免误锁版本）。
