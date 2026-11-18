# Web 交互系统交接文档（WEB_HANDOFF）

> 读者：未参与本研究、具备 Web 开发能力的同学。本文档给出可直接执行的入口、接口契约与仓库清单。科研背景与结果见同目录 `PAPER_HANDOFF.md`，这里不重复长篇科研内容。

## 一、给 Web 同学的一页速览

项目目标：做一个「肾癌 CT 智能分诊」研究原型——上传/选择一张肾癌 CT，贝叶斯 U-Net（MC-Dropout）给出肿瘤分割、逐像素不确定性（熵），当肿瘤区域平均熵超过阈值时输出「建议转诊」，否则输出分割结果。

已完成的研究工作（可直接复用）：

- 模型：2D U-Net（3 类：背景/肾脏/肿瘤），在 `scripts/run_experiments.py` 里，类名 `UNet`，推理函数 `mc_predict`。
- 权重：四组训练好的 `best_unet.pth`，各约 31 MB，是纯 `state_dict`；web 端只接 full 这组。权重文件由项目方私下发送（网盘/群文件），拿到后放到本地 `models/` 目录，不随仓库分发。
- 预处理：HU 裁剪 [-200,300] 后归一化 [0,1]，缩放到 256×256（逻辑在 `KiTS19SliceDataset` 里，web 端要把它抽成独立函数）。
- 推理：MC-Dropout 采样 N=10，输出平均概率、argmax 分割、逐像素熵、逐像素方差、肿瘤二值掩码。
- 结果：四组实验指标在 `results/bayes/{data10pct,data20pct,data30pct,datafull}/metrics.json`，逐切片熵+Dice 在同目录 `tumor_entropy_scores.csv`，Risk-Coverage 与覆盖图在 `assets/figures/`。

推荐的最小可行系统（MVP）：先用预计算结果和现有图做页面展示（样例选择、原始图、分割掩码、熵热图、分诊结论、指标展示），不接真实模型推理；验证 UI 后再接真实推理。

系统目前没有的：真实在线推理接口（现有代码是离线批处理，没有 HTTP 服务）、登录权限、真实影像合规处理、并发异步。这些要 Web 团队新写或确认。

最大集成风险：现有代码是「训练 + 整测试集离线评估」的一体化脚本，预处理内嵌在数据集类里，没有独立的「单张影像在线推理」入口。正确做法：复用 `UNet`、`mc_predict` 和预处理参数，另写一个薄的推理服务，别去改 `run_experiments.py` 主流程。

建议接手顺序：① 跑通第八部分的最小验证；② 做 MVP（预计算结果展示）；③ 抽出预处理函数、写推理服务；④ 接真实推理与异步任务；⑤ 权限、日志、删除机制等上线项。

## 二、现有系统与代码地图

入口在 `scripts/run_experiments.py`（仓库内唯一权威副本）。

| 模块（函数/类） | 位置 | 负责什么 | 不负责什么 |
|---|---|---|---|
| `KiTS19SliceDataset` | 约 123 行 | 读 NIfTI、逐切片、预处理、常驻内存 | 只处理带标注整卷，不处理单张上传图 |
| `UNet` / `DoubleConv` | 约 192 / 179 行 | 分割模型，3 类输出 | 不做不确定性聚合 |
| `mc_predict()` | 约 457 行 | MC-Dropout 推理，返回均值/分割/方差/熵/肿瘤掩码 | 不做阈值拒绝 |
| `main()` | 约 743 行 | argparse 入口，编排训练/评估/交叉验证 | 不提供在线推理 |

数据流（离线）：NIfTI 体积 → 逐轴向切片 → 只保留标注非空切片 → HU clip [-200,300] → 归一化 [0,1] → resize 256×256 → `UNet` 前向 → `mc_predict` → `metrics.json`、`tumor_entropy_scores.csv`。

推理函数契约（Web 要复用的 seam）：

```python
def mc_predict(model, img, n_samples, device):
    # img: (1,1,H,W) 张量，已归一化 [0,1]，H=W=256
    # 返回五个 numpy 数组：
    #   mean_probs (3,H,W), mean_pred (H,W) 0/1/2,
    #   variance (H,W), entropy (H,W), pred_binary (H,W) 0/1
```

注意：`mc_predict` 内部先 `model.train()` 保持 Dropout 激活、再把 BatchNorm 切到 eval；Web 端必须沿用这个顺序，否则 MC-Dropout 退化成单次前向（熵恒为 0）。

配置：参数是 `run_experiments.py` 里的模块级常量（`HU_MIN/HU_MAX=-200/300`、`IMG_SIZE=256`、`NUM_CLASSES=3`）。Web 团队应把预处理参数抽成常量/配置，不要散落修改。

## 三、推荐的系统架构与集成边界

- Web 前端：新写。
- 后端 API：新写（接收请求、调用推理服务、返回 JSON/图）。
- 模型推理服务：复用 `UNet` + `mc_predict`，新写一个推理 adapter（加载权重、预处理单张输入、调用 `mc_predict`、套阈值拒绝）。
- 预处理/后处理：把 `KiTS19SliceDataset` 里的预处理抽成独立函数 `preprocess_slice()`（建议接口，尚未实现）。
- 文件与结果存储、用户权限、日志监控：新写。

两个实施层级：

1. MVP（推荐先做）：后端只读预计算结果——`results/bayes/*/tumor_entropy_scores.csv`、`assets/figures/` 里的图、`results/bayes/*/metrics.json`，前端展示「样例 CT + 分割图 + 熵热图 + 分诊结论 + 指标」。成本低，适合先对齐 UI 和口径。
2. 完整系统：接入真实模型。后端加载 full 的 `best_unet.pth`，对上传的 NIfTI 或切片做预处理 + `mc_predict` + 阈值拒绝，返回结果并持久化。

风险：完整系统直接暴露真实医疗影像上传，会引入合规、删除、审计问题；MVP 用预计算结果则完全规避。

## 四、模型输入输出契约

输入（模型层）：

- 单张切片：float32 张量 shape `(1,1,256,256)`，单通道，已从 HU 裁剪 [-200,300] 并归一化到 [0,1]。
- 整卷 NIfTI：需先按轴向切片、逐张做同样预处理；当前没有单卷在线入口，要 Web 团队新写。
- 支持批量：`mc_predict` 可把 `n_samples` 次前向做成一个 batch；多患者并发目前不支持。

模型文件与加载：

- full 权重：`best_unet.pth`，约 31 MB，纯 `state_dict`，由项目方私下发送，不随仓库分发。
- 加载方式：`UNet(dropout_rate=0.1)`，然后 `model.load_state_dict(torch.load("best_unet.pth", map_location="cpu"))`。
- 其余三组（10%、20%、30%）是研究对照，web 不必接入。

输出（`mc_predict` 返回，全部 numpy）：

- `mean_pred`：`(256,256)` 整数 0/1/2（0 背景、1 肾脏、2 肿瘤）。
- `pred_binary`：`(256,256)` 0/1 肿瘤掩码。
- `entropy`：`(256,256)` 浮点，逐像素信息熵（nats）。
- `variance`：`(256,256)` 浮点，逐像素方差。
- `mean_probs`：`(3,256,256)` 平均 softmax 概率。

切片级不确定度 = 肿瘤区域（`pred_binary==1` 或 `mean_pred==2`）内 `entropy` 的平均值。

拒绝口径：把切片按肿瘤平均熵从小到大排序，固定接受比例（coverage），阈值取对应分位数的熵。

- 推荐阈值：70% 覆盖率对应熵阈值 0.5804；50% 覆盖率对应 0.3673（这两个数就是 full 组 `results/bayes/datafull/tumor_entropy_scores.csv` 里肿瘤平均熵的 70%/50% 分位数）。
- 规则：肿瘤平均熵 ≥ 阈值 → 「建议转诊/人工复核」；< 阈值 → 「可采纳 AI 分割结果」。
- MC 采样次数选 10，依据见 `results/mc_sampling_results.json`；口径见 `PAPER_HANDOFF.md`。

建议接口草案（尚未实现，仅示例）：

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

错误处理：空文件/非 NIfTI/尺寸异常返回 4xx 与明确信息；推理超时返回 5xx 且不留临时文件；无肿瘤预测时 `decision` 不输出「正常」，标注「未检测到肿瘤区域」。

资源：GPU 上单张 256×256 切片 10 次 MC 前向约几十毫秒；CPU 明显更慢，MVP 建议预计算。

## 五、Web 页面与用户流程建议

页面/流程：

- 首页/样例选择：从预计算样例里选一个展示。
- 上传页（完整系统）：上传 NIfTI/切片，后端推理。
- 结果页：并排展示原始 CT、分割掩码、熵热图，显示肿瘤平均熵、阈值、覆盖率、分诊结论。
- 阈值/覆盖率设置（研究演示）：滑块调覆盖率，映射到熵阈值，实时更新「接受/转诊」。
- 指标页：展示四组 Dice 表、Risk-Coverage 曲线（`assets/figures/risk_coverage.png`）、可靠性图（`assets/figures/reliability.png`）、两张数据标度图（`data_scaling_dice`、`data_scaling_uncertainty`）。
- 下载：掩码图、熵图、结果 JSON。

区分：

- 已有：分割掩码、熵图、指标表、Risk-Coverage 曲线（由 `scripts/plot_*.py` 离线生成）。
- 需新增：上传推理接口、阈值滑块、结果下载、异步任务。
- 仅研究演示：阈值调整、覆盖率权衡展示。
- 不建议提供给终端用户：真实医疗影像上传、自动临床诊断结论。

页面必须显著标注「研究原型，仅供科研演示，不构成临床诊断结论」。

## 六、仓库里有什么 / 还缺什么

仓库（main 分支）当前包含：

- `scripts/`：`run_experiments.py`（模型+训练+推理主脚本）、`stats_validation.py`、`mc_sampling_study.py`、`plot_results.py`、`plot_reliability.py`、`plot_overlays.py`、`figstyle.py`、`verify_metrics.py`。
- `results/bayes/`：`data10pct`、`data20pct`、`data30pct`、`datafull` 四组，每组 `metrics.json`、`evaluation_metrics.csv`、`tumor_entropy_scores.csv`。
- `results/vanilla/`：`vanilla10pct`、`vanilla20pct`、`vanillafull` 三组 `metrics.json`。
- `results/`：`stats_validation_results.json`、`mc_sampling_results.json`、`risk_coverage_data.csv`。
- `assets/figures/`：`dice_comparison`、`reliability`、`risk_coverage`、`overlay_low_uncertainty`、`overlay_high_uncertainty`、`data_scaling_dice`、`data_scaling_uncertainty`（png + svg）。
- `tests/`、`requirements.txt`、`docs/`。

还不具备、需要补的：

- 权重不在仓库，由项目方私下发送给 Web 同学（拿到后放到 `models/` 目录）。
- 已有 `src/preprocess.py`、`src/infer.py`：在线推理入口（预处理 + 模型加载 + MC 推理 + 阈值分诊），Web 团队可直接参考或在其上包一层 HTTP 服务。
- 已有 `sample_data/` 脱敏样例（见第七部分）。
- `verify_metrics.py` 已改成读仓库 `results/` 并含四组，克隆后可直接自检。
- `requirements.txt` 是最小依赖集；torch 请按各自机器装 CPU/CUDA 对应版本。

## 七、sample_data 样例包要求（给 Web 团队联调用）

为了验证预处理和推理对不对，仓库需要一份脱敏样例，建议 `sample_data/` 放这些：

- 一张 256×256 的切片 numpy（.npy 或 .npz，含预处理后的灰度图）。
- 这张切片的预期输出：`mean_pred` 掩码、肿瘤掩码、熵图（各一张 .npy 或 .png）。
- 一个样例结果 JSON（跟 `/api/infer` 返回结构一致），方便前端开发对字段。
- 说明文件：来源（KiTS19 哪个 case/slice，已脱敏）、用什么权重、阈值多少、期望的 `decision`。

要求：不含原始 NIfTI，不包含任何患者信息；只放一张即可，目的是让 Web 同学不用整套数据集就能端到端调通。

仓库已提供这份样例，直接看 `sample_data/README.md`。

## 八、最小验证路径（不需要权重和数据）

Web 同学克隆仓库后，先做这几步确认环境能跑：

1. `pip install -r requirements.txt`。
2. 打开 `results/bayes/datafull/metrics.json`，确认 `dice=0.748`、`auroc_entropy_wrong=0.5562`。
3. 打开 `results/stats_validation_results.json`，确认 `data30pct.auroc_entropy_wrong.point=0.6013`、`data10pct.auroc_entropy_wrong.point=0.4878`。
4. 打开 `assets/figures/`，确认能看到 `risk_coverage.png`、`reliability.png`、`data_scaling_dice.png`、`data_scaling_uncertainty.png`。

（`verify_metrics.py` 只读仓库 `results/`，不需要权重和数据；`mc_sampling_study.py`、`plot_overlays.py` 这类复现脚本才需要本地数据集和权重。）

## 九、建议的仓库结构（目标态）

```text
repo/
  README.md
  LICENSE
  requirements.txt
  src/
    infer.py            # 推理服务入口（已有，可参考）
    preprocess.py       # 预处理函数（已有，可参考）
  scripts/              # 现有脚本
    run_experiments.py
    verify_metrics.py
    stats_validation.py
    mc_sampling_study.py
    plot_results.py
    plot_reliability.py
    plot_overlays.py
    figstyle.py
  tests/
  configs/              # 阈值/覆盖率配置（从硬编码抽出，待补）
  sample_data/          # 脱敏样例（已有，见第七部分）
  results/              # 四组 bayes + 三组 vanilla + 统计 json
  assets/figures/
  docs/
    WEB_HANDOFF.md
    PAPER_HANDOFF.md
  deployment/           # 部署说明（后续）
```

不要放进仓库：原始数据集、`output_ft`/`output` 实验目录、venv、缓存、含个人信息的文档、任何真实影像。

## 十、安全、隐私、许可与研究边界

- 患者数据：KiTS19 是公开数据集，但所有真实影像都不上传仓库，只放脱敏样例和结果。
- 数据授权：KiTS19 公开≠可再分发原始影像；只上传脱敏样例与结果。
- 模型与结果开源：代码、结果、模型权重均已确认可公开。
- 登录/权限/审计：研究原型至少加访问日志与删除机制；是否登录由导师定。
- 是否允许上传真实医疗数据：默认不允许；确需则先做伦理/隐私评估并加删除与审计。
- 所有页面标注「研究原型，不构成临床诊断结论」；指标不写成「临床验证结果」。
- 仓库公开/私有：已确认公开。

待确认的阻塞事项：伦理审批文件、数据使用授权（KiTS19 公开数据集的使用条款）。公开问题已敲定，不再阻塞。

## 十一、测试与验收标准

| 验收项 | 验证方法 | 责任人 |
|---|---|---|
| README 可安装 | 干净环境 `pip install -r requirements.txt` 成功 | Web 同学 |
| 数字一致 | 页面展示的四组 Dice 与 `results/bayes/*/metrics.json` 一致 | Web 同学 |
| 页面正确展示 | 显示 `assets/figures/risk_coverage.png` 与四组 Dice 表 | Web 同学 |
| API 处理异常 | 上传空文件/非 NIfTI 返回 4xx，超时返回 5xx 且不残留临时文件 | Web 同学 |
| 克隆不依赖本机路径 | 克隆后按 README 能跑，无任何绝对路径 | Web 同学 |
| 无敏感信息 | 仓库不含凭证、真实影像、个人信息 | 交接负责人 |
| MVP 可独立演示 | 不接真实推理也能用预计算结果演示完整流程 | Web 同学 |
| 真实推理可跑 | 用项目方私发的 full 权重 + 样例数据，单张切片出分割/熵/分诊结论 | Web 同学 |

## 十二、已知限制与待解决问题

P0（开始前必须确认）：

- 公开授权：已确认，代码/结果/权重都可公开。
- 是否允许上传真实医疗影像（默认否）。

P1（MVP 前应解决）：

- 预处理逻辑从 `KiTS19SliceDataset` 抽出为独立函数。
- 确定 MVP 用预计算样例还是接真实推理。
- 页面显著标注「研究原型，非临床诊断」。

P2（完整系统上线前解决）：

- 推理性能与并发、异步任务、超时与错误恢复。
- 登录/权限、上传文件保存与删除、审计日志。
- 浏览器兼容、技术栈选型、模型版本管理。
