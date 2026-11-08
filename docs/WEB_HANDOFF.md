# Web 交互系统交接文档（WEB_HANDOFF）

> 读者：未参与本研究、具备 Web 开发能力的同学。本文档给出可直接执行的入口、接口契约与 GitHub 清单。科研背景与结果详见 `../论文同学交接文档.md`，本文不再重复长篇科研内容。
> 本轮只编写交接材料，不执行任何 git 操作。

## 一、给 Web 同学的一页速览

项目目标：做一个“肾癌 CT 智能分诊”研究原型——上传/选择一张肾癌 CT，贝叶斯 U-Net（MC-Dropout）给出肿瘤分割、逐像素不确定性（熵），当肿瘤区域平均熵超过阈值时输出“建议转诊”，否则输出分割结果。

已完成的研究工作（可直接复用）：

- 模型：2D U-Net（3 类：背景/肾脏/肿瘤），代码 `run_experiments_new.py` 中 `UNet`（约 192 行）、`mc_predict`（约 457 行）。
- 权重：三组训练好的 `best_unet.pth`，各约 29.7 MB，推荐使用 full 组 `D:\大创代码\output_ft\experiments\datafull\best_unet.pth`。
- 预处理：HU 裁剪 [-200,300] 后归一化 [0,1]，缩放到 256×256（在 `KiTS19SliceDataset` 中实现）。
- 推理：MC-Dropout 采样 N=10 次，输出平均概率、argmax 分割、逐像素熵、逐像素方差、肿瘤二值掩码。
- 结果：三组实验指标（`output_ft\experiments\*\metrics.json`）、逐切片熵+Dice（`tumor_entropy_scores.csv`）、Risk-Coverage 曲线与覆盖图（`figures\`）。

推荐的最小可行系统（MVP）：先用预计算样例与结果图做页面展示（上传样例选择、原始图、分割掩码、熵热图、分诊结论、指标展示），不接真实模型推理；验证 UI 后，再接入真实推理。

系统目前不能提供的功能：真实模型在线推理接口（现有代码是离线批处理脚本，没有 HTTP 服务）、用户登录与权限、真实医疗影像的合规处理、并发与异步任务。以上均为 Web 团队需新写或确认的内容。

当前最大的集成风险：现有代码是“训练 + 整测试集离线评估”的一体化脚本，预处理逻辑内嵌在数据集类里，没有独立的“单张影像在线推理”入口；Web 团队若直接改 `run_experiments_new.py` 主流程会走错入口。正确做法是把 `UNet`、`mc_predict` 和预处理参数复用，另写一个薄薄的推理服务。

建议的接手顺序：① 跑通“最小成功路径”（第八部分）；② 做 MVP（预计算结果展示）；③ 抽出预处理函数并写推理服务；④ 接真实推理与异步任务；⑤ 再做权限、日志、删除机制等上线项。

## 二、现有系统与代码地图

主要代码目录与入口（单文件脚本 `D:\codex工作空间\run_experiments_new.py`，权威副本 `D:\大创代码\run_experiments.py`，两者内容一致）：

| 模块（函数/类） | 位置 | 负责什么 | 不负责什么 |
|---|---|---|---|
| `main()` | 743 行 | argparse 入口，编排训练/评估/交叉验证 | 不提供在线推理 |
| `run_one_experiment()` | 554 行 | 单次实验：加载数据→训练→推理→评估→写文件 | 离线批处理编排 |
| `KiTS19SliceDataset` | 123 行 | 读 NIfTI、逐切片、预处理、常驻内存 | 只针对带标注的整卷 NIfTI，不处理单张上传图 |
| `UNet` / `DoubleConv` | 192 / 178 行 | 分割模型，3 类输出 | 不做不确定性聚合 |
| `mc_predict()` | 457 行 | MC-Dropout 推理：返回 mean_probs/mean_pred/variance/entropy/pred_binary | 不做阈值拒绝 |
| `eval_metrics()` | 474 行 | 新口径指标（Dice/IoU/选择性准确率/AUROC/AURC） | 离线评估，需要真值标注 |
| `FocalLoss`/`SoftDiceLoss`/`CombinedLoss`/`build_optimizer` | 250/230/270/280 行 | 训练损失与优化器 | 推理不需要 |
| `train_experiment()`/`train_one_epoch()` | 412/362 行 | 训练循环与断点 | 推理不需要 |

数据流（离线）：NIfTI 体积 → `_load_volume`（nibabel）→ 逐轴向切片 → 只保留标注非空切片 → HU clip [-200,300] → 归一化 [0,1] → resize 256×256 → `DataLoader` → `UNet` 前向 → 损失/推理 → `eval_metrics` → `metrics.json` 与 `tumor_entropy_scores.csv`、`samples/*.npz`。

推理函数契约（Web 要复用的 seam）：

```python
def mc_predict(model, img, n_samples, device):
    # img: (1,1,H,W) 张量，已归一化 [0,1]，H=W=256
    # 返回五个 numpy 数组：
    #   mean_probs (3,H,W), mean_pred (H,W) 0/1/2,
    #   variance (H,W), entropy (H,W), pred_binary (H,W) 0/1
```

注意：`mc_predict` 内部先 `model.train()` 保持 Dropout 激活、再把 BatchNorm 切到 eval；Web 端必须沿用这个顺序，否则 MC-Dropout 会退化成单次前向（熵恒为 0）。

配置：无独立配置文件，参数都在 `main()` 的 argparse 默认值和 `run_one_experiment` 里硬编码/传递（HU_MIN/HU_MAX、IMG_SIZE=256、NUM_CLASSES=3、CLASS_WEIGHTS、TRAIN_RATIO 等模块级常量）。Web 团队应把预处理参数抽成常量/配置，不要散落修改。

## 三、推荐的系统架构与集成边界

现有代码是离线批处理，与在线推理的差异：它一次遍历整个测试集、依赖带标注 NIfTI、把预处理内嵌在 Dataset 里、无并发/超时/错误恢复。在线服务需要“加载一次模型 + 单张输入 → 单次输出”的形态。

建议边界：

- Web 前端：新写（页面、上传/样例选择、展示）。
- 后端 API：新写（接收请求、调用推理服务、返回 JSON/图）。
- 模型推理服务：部分复用（`UNet` + `mc_predict`），新写一个“推理 adapter”负责加载权重、预处理单张输入、调用 `mc_predict`、套阈值拒绝。
- 预处理/后处理：把 `KiTS19SliceDataset` 里的预处理抽成独立函数 `preprocess_slice()`（建议接口，尚未实现），推理服务调用它。
- 文件与结果存储：新写（上传临时目录、结果文件、清理策略）。
- 用户权限：新写（登录、权限）。
- 日志与监控：新写。

两个实施层级：

1. MVP（推荐先做）：后端只读预计算结果——三组 `tumor_entropy_scores.csv`、`figures\` 里的图、`metrics.json`，前端展示“样例 CT + 分割图 + 熵热图 + 分诊结论 + 指标”。成本低、当天可跑通，适合先对齐 UI 和数据口径。
2. 完整系统：接入真实模型。后端加载 `datafull\best_unet.pth`，对上传/选择的 NIfTI 或切片做预处理 + `mc_predict` + 阈值拒绝，返回结果并持久化。成本高（需处理大文件上传、GPU/CPU 推理耗时、异步任务、超时、错误输入）。

风险：完整系统若直接暴露真实医疗影像上传，会引入数据合规、删除与审计问题；MVP 用预计算结果则完全规避。

## 四、模型输入输出契约

输入（模型层）：

- 单张切片：`float32` 张量 shape `(1,1,256,256)`，单通道，像素已从 HU 裁剪 [-200,300] 并归一化到 [0,1]。
- 若用户上传整卷 NIfTI：需先按轴向切片、对每张切片做同样预处理；当前无单卷在线入口（待 Web 团队新写）。
- 支持批量：`mc_predict` 的单次调用可把 `n_samples` 次前向做成一个 batch（现有代码用 `img.expand(N, ...)` 批量前向），但“多患者并发”目前不支持。

模型文件与路径：

- full 权重：`D:\大创代码\output_ft\experiments\datafull\best_unet.pth`（29.7 MB，`torch.load` 加载到 `UNet(dropout_rate=0.1)`）。
- 另两组：`data10pct`、`data20pct` 同目录同名文件（小样本研究用，Web 不必接入）。

输出（`mc_predict` 返回，全部 numpy）：

- `mean_pred`：`(256,256)` 整数 0/1/2（0 背景、1 肾脏、2 肿瘤）。
- `pred_binary`：`(256,256)` 0/1 肿瘤掩码。
- `entropy`：`(256,256)` 浮点，逐像素信息熵（nats）。
- `variance`：`(256,256)` 浮点，逐像素方差（3 类平均）。
- `mean_probs`：`(3,256,256)` 平均 softmax 概率。

阈值与筛选（来自 `mc_sampling_results.json`，口径见 `../论文同学交接文档.md` 4.5）：

- 切片级不确定度 = 肿瘤区域（`pred_binary==1` 或 `mean_pred==2`）内 `entropy` 的平均值。
- 推荐阈值：70% 覆盖率对应熵阈值 0.5804；50% 覆盖率对应 0.3673。
- 规则：肿瘤平均熵 ≥ 阈值 → “建议转诊/人工复核”；< 阈值 → “可采纳 AI 分割结果”。

建议接口草案（尚未实现，仅作示例）：

```http
POST /api/infer
Content-Type: multipart/form-data   # field: file (nifti)
```

```json
{
  "status": "ok",
  "case_id": "sample_0001",
  "slices": [
    {
      "slice_index": 23,
      "tumor_mean_entropy": 0.031,
      "threshold": 0.5804,
      "coverage": 0.70,
      "decision": "accept",
      "tumor_dice_hint": null,
      "mask_url": "/files/sample_0001_23_mask.png",
      "entropy_url": "/files/sample_0001_23_entropy.png"
    }
  ]
}
```

错误处理：空文件/非 NIfTI/尺寸异常返回 4xx 与明确信息；推理超时返回 5xx 并不留临时文件；无肿瘤预测时 `decision` 不应输出“正常”，应标注“未检测到肿瘤区域”。

资源：GPU 上单张 256×256 切片 10 次 MC 前向约几十毫秒；CPU 上明显更慢，MVP 建议预计算。

## 五、Web 页面与用户流程建议

可设计页面/流程：

- 首页/样例选择：从预计算样例里选一个（现有 `samples/*.npz` 已删，但 `figures\` 和 `tumor_entropy_scores.csv` 可支撑展示）。
- 上传页（完整系统）：上传 NIfTI/切片，后端推理。
- 结果页：并排展示原始 CT、分割掩码、熵热图，显示肿瘤平均熵、阈值、覆盖率、分诊结论。
- 阈值/覆盖率设置（研究演示）：滑块调整覆盖率，映射到熵阈值，实时更新“接受/转诊”。
- 指标页：展示三组 Dice 表、Risk-Coverage 曲线（`figures\risk_coverage.png`）、可靠性图（`figures\reliability.png`）。
- 下载：掩码图、熵图、结果 JSON。

区分：

- 现有代码已支持：生成分割掩码、熵图、指标表、Risk-Coverage 曲线（通过 `plot_*.py` 脚本离线生成）。
- 需新增后端接口：上传推理、阈值滑块、结果下载、异步任务。
- 仅研究演示：阈值调整、覆盖率权衡展示。
- 暂不建议提供给终端用户：真实医疗影像上传、自动临床诊断结论。

页面必须显著标注“研究原型，仅供科研演示，不构成临床诊断结论”。

## 六、本地运行与端到端验证

环境：Python 3.12 venv `D:\大创代码\kits19`（torch 2.11.0+cu128、numpy、scipy、scikit-learn、matplotlib、nibabel）。pip 保持 24.3.1，勿升级。

数据位置：`D:\大创代码\kits19\datasets`（前 210 例有标注，26.9 GB）。

最小成功路径（不训练，只验证推理与结果一致）：

1. 在 VSCode 打开 `D:\codex工作空间`，解释器选 `D:\大创代码\kits19\Scripts\python.exe`。
2. 跑指标自检：`D:\大创代码\kits19\Scripts\python.exe verify_metrics.py`，应打印三组选择性准确率/AUROC/AURC 与 `metrics.json` 一致。
3. 跑画图：`plot_results.py`、`plot_reliability.py`、`plot_overlays.py`，应在 `figures\` 生成 `risk_coverage.png`、`reliability.png`、`overlay_low_uncertainty.png`、`overlay_high_uncertainty.png`。
4. 核对：`stats_validation_results.json` 中 full 的 AUROC=0.5562、10% 的 AUROC=0.4878。

预期输出与判定：上一步生成的图与 `figures\` 现有文件一致；`verify_metrics.py` 打印数值与 `output_ft\experiments\*\metrics.json` 四舍五入一致。

常见错误：终端 `python` 不是 venv 版（用完整路径）；`plot_overlays.py` 会重新加载测试集约 1 分钟（正常）；DataLoader 多进程报错（训练命令加 `--num-workers 0`）。

缺失项：`samples/*.npz` 已被删除以省空间（可用 `best_unet.pth` 重新推理生成）；无在线推理 API；无 requirements.txt（需按 `pip` 冻结的依赖补一份）。

## 七、可复用成果与论文结果映射

下列成果可直接用于 Web 页面或 README，编号沿用 `../论文同学交接文档.md` 的资产编号：

| 编号 | 内容 | 原始路径 | 建议仓库相对路径 | 格式/大小 | 脱敏 | 可否公开 | 建议页面 | 备注 |
|---|---|---|---|---|---|---|---|---|
| R1 | 三组 bayes 指标 | `output_ft\experiments\{data10pct,data20pct,datafull}\metrics.json` | `results/metrics.json` | JSON <1 KB | 是 | 可 | 指标页 | 主数字 |
| R2 | 三组逐切片熵+Dice | 同上 `tumor_entropy_scores.csv` | `results/tumor_entropy_scores.csv` | CSV ~22 KB/组 | 是 | 可 | 结果分析/附录 | 画曲线用 |
| R3 | Risk-Coverage 图 | `figures\risk_coverage.png/.svg` | `assets/figures/risk_coverage.png` | PNG 0.7 MB | 是 | 可 | 指标页 | 主图+放大 |
| R4 | Dice 对比图 | `figures\dice_comparison.png/.svg` | `assets/figures/dice_comparison.png` | PNG ~0.04 MB | 是 | 可 | 指标页 | — |
| R5 | 可靠性图 | `figures\reliability.png/.svg` | `assets/figures/reliability.png` | PNG ~0.1 MB | 是 | 可 | 指标页 | — |
| R6 | 低/高熵覆盖图 | `figures\overlay_*.png/.svg` | `assets/figures/overlay_*.png` | PNG ~0.3 MB | 是 | 可 | 结果页示例 | 蓝=真值肿瘤、橙=预测 |
| R7 | 统计检验结果 | `stats_validation_results.json` | `results/stats_validation_results.json` | JSON ~3 KB | 是 | 可 | 指标页/附录 | — |
| R8 | MC 采样与阈值 | `mc_sampling_results.json` | `results/mc_sampling_results.json` | JSON ~1 KB | 是 | 可 | 阈值设置页 | 阈值来源 |

风险：这些结果基于 KiTS19 单一数据集、seed=42；不要写成“产品已验证”。10% 组熵不可靠，页面若展示三组对比，需同时注明。

## 八、GitHub 仓库规划与推送清单

完整清单见同目录 `GITHUB_EXPORT_MANIFEST.md`。核心结论：

- 直接提交：代码、测试、画图/分析脚本、`docs\`、`results\` 的 json/csv、`assets\figures\` 的图、README、LICENSE、requirements。
- 用 Git LFS 或 GitHub Release：三组 `best_unet.pth`（各 29.7 MB，合计约 89 MB），建议放 Release 或 LFS，不直接进 Git 历史。
- 禁止上传：KiTS19 原始数据（26.9 GB）、任何真实医疗影像、申报书与论文初稿（含个人电话，见第十部分）、venv、缓存、临时文件、含绝对路径的本地配置。
- 绝对禁止：API Key、密码、令牌、Cookie、云存储地址等凭证。

仓库中只能引用相对路径；README 的示例命令不能写 `D:\大创代码\...` 这种本机绝对路径。

## 九、建议的 GitHub 仓库结构

```text
repo/
  README.md                 # 快速开始：环境、样例、推理、指标
  LICENSE                   # 待选（见第十部分）
  requirements.txt          # 依赖版本（需从 venv 冻结生成，待补）
  src/                      # 推理服务 + 预处理函数（Web 团队新写/重构）
    infer.py
    preprocess.py
  scripts/                  # 现有脚本原样保留
    run_experiments.py
    verify_metrics.py
    stats_validation.py
    mc_sampling_study.py
    plot_results.py
    plot_reliability.py
    plot_overlays.py
    figstyle.py
  tests/                    # 现有 4 个回归测试
  configs/                  # 阈值/覆盖率等（从硬编码抽出，待补）
  sample_data/              # 一个脱敏样例切片/图（待准备）
  results/                  # metrics.json、tumor_entropy_scores.csv、统计 json
  assets/figures/           # 5 张图
  docs/
    WEB_HANDOFF.md
    GITHUB_EXPORT_MANIFEST.md
  deployment/               # 部署说明（后续）
```

不要放进仓库：`kits19` 数据集、`output_ft`/`output` 原始实验目录、venv、`__pycache__`、`.mplcache`、任何含个人信息的 docx。

README 快速开始（建议内容）：① `pip install -r requirements.txt`；② 下载模型权重到 `models/`；③ 运行 `scripts/verify_metrics.py` 与 `scripts/plot_results.py`；④ 打开 `assets/figures/` 查看结果；⑤ 启动 `src/infer.py`（Web 团队实现后）。

## 十、安全、隐私、许可与研究边界

- 是否含患者数据：KiTS19 是公开数据集，但本机另有 90 例无标注影像；所有真实影像均不得上传 GitHub。
- 数据授权：KiTS19 为公开科研数据集，但“可公开”不等同于“可再分发全部原始影像”，建议只上传脱敏样例与结果，不上传原始 NIfTI。
- 模型与结果是否可开源：代码与结果可开源；模型权重需与导师确认开源许可后再公开。
- 第三方许可：依赖库、字体、图片需核对各自许可。
- 登录/权限/审计：研究原型建议至少加访问日志与删除机制；是否登录由导师决定。
- 是否允许上传真实医疗数据：默认不允许；如确需，必须先完成伦理/隐私评估，并加删除与审计。
- 避免误写：所有页面标注“研究原型，不构成临床诊断结论”；指标不写成“临床验证结果”。
- 谁决定仓库公开/私有：指导教师/项目负责人。

待确认的阻塞事项：伦理审批文件、数据使用授权、模型开源许可、仓库公开或私有的决定——均未提供，需导师确认后才能决定哪些文件可公开。

## 十一、测试与验收标准

| 验收项 | 验证方法 | 责任人 |
|---|---|---|
| README 可安装 | 干净环境 `pip install -r requirements.txt` 成功 | Web 同学 |
| 样例跑通推理 | 用 `scripts/verify_metrics.py` 复现 full 组 AUROC=0.5562 | Web 同学 |
| 页面正确展示 | 页面显示 `figures\risk_coverage.png` 与三组 Dice 表 | Web 同学 |
| API 处理异常 | 上传空文件/非 NIfTI 返回 4xx，超时返回 5xx 且不残留临时文件 | Web 同学 |
| 数字一致 | 页面展示的 Dice 与 `results/metrics.json` 一致 | Web 同学 |
| 克隆不依赖本机路径 | 克隆后按 README 能跑，无 `D:\` 绝对路径 | Web 同学 |
| 无敏感信息 | 仓库不含凭证、真实影像、个人电话 | 交接负责人 |
| MVP 可独立演示 | 不接真实推理也能用预计算结果演示完整流程 | Web 同学 |

## 十二、已知限制与待解决问题

P0（开始前必须确认）：

- 仓库公开/私有、模型权重与结果的公开授权（导师决定）。
- 是否允许上传真实医疗影像（默认否）。
- requirements.txt 缺失，需从 venv 冻结生成。

P1（MVP 前应解决）：

- 预处理逻辑从 `KiTS19SliceDataset` 抽出为独立函数。
- 确定 MVP 用预计算样例还是接入真实推理。
- 页面显著标注“研究原型，非临床诊断”。

P2（完整系统上线前解决）：

- 推理性能与并发（CPU 下 10 次 MC 前向较慢）、异步任务、超时与错误恢复。
- 登录/权限、上传文件的保存时间与删除机制、审计日志。
- 浏览器兼容、前后端技术栈选型、模型版本管理。

## 十三、下一轮 Git 推送执行计划（本轮不执行）

- 目标仓库地址：待填（下一轮由用户提供）。
- 准备分支：`main`（或 `docs/web-handoff`）。
- 准备复制/提交：`docs/WEB_HANDOFF.md`、`docs/GITHUB_EXPORT_MANIFEST.md`、`scripts\`、`tests\`、`results\`、`assets\figures\`、README、requirements.txt、LICENSE（若确定）。
- 排除：`kits19` 数据集、`output_ft`/`output`、venv、`__pycache__`、`.mplcache`、申报书/初稿 docx、任何真实影像。
- 推送前安全检查：`rg -i "password|token|secret|api[_-]?key|手机号|phone"` 无命中；确认无 `D:\大创代码` 绝对路径。
- 推送前运行：`python -m unittest discover -s tests`（现有 4 个测试）与 `scripts/verify_metrics.py`。
- 建议 commit 结构：docs（交接文档）→ scripts+tests（代码）→ results+assets（结果与图）。
- 推送后检查：README 渲染、docs 链接、Release/LFS 文件可下载、无敏感文件。
- 合并策略：若远程已有内容，先 `git fetch` 再 rebase/merge，保留双方 README。
- 停止条件：发现大文件未走 LFS/Release、发现任何凭证或个人信息、发现真实影像、远程已有冲突且无法安全合并——立即停止并上报。

> 本计划不含任何访问令牌、账号密码或认证信息。下一轮提供仓库地址后，再按 `GITHUB_EXPORT_MANIFEST.md` 执行实际推送。
