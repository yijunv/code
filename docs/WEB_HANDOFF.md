# Web 交互系统交接文档（WEB_HANDOFF）

> 这份文档写给你们——负责 web 端的同学。里面是我们这边做好的东西、接口约定和仓库清单，你们照着就能开工。科研背景和结果在 `PAPER_HANDOFF.md`，这里不重复长篇内容。

## 一、给你们的一页速览

我们要做的是一个「肾癌 CT 智能分诊」研究原型：上传/选择一张肾癌 CT，贝叶斯 U-Net（MC-Dropout）输出肿瘤分割和逐像素不确定性（熵），肿瘤区域平均熵超过阈值就提示「建议转诊」，否则给出分割结果。

我们这边已经做好的，你们直接复用：

- 模型：2D U-Net（3 类：背景/肾脏/肿瘤），在 `scripts/run_experiments.py` 里，类名 `UNet`，推理函数 `mc_predict`。
- 权重：四组训练好的 `best_unet.pth`，各约 31 MB，纯 `state_dict`；你们只需要接 full 这组。权重我们会私下发给你们（网盘/群文件），拿到后放到本地 `models/` 目录。
- 预处理：HU 裁剪 [-200,300] → 归一化 [0,1] → 缩放 256×256，已经抽好在 `src/preprocess.py`。
- 推理：MC-Dropout 采样 N=10，输出平均概率、argmax 分割、逐像素熵、逐像素方差、肿瘤二值掩码，已经写在 `src/infer.py`。
- 结果：四组实验指标在 `results/bayes/{data10pct,data20pct,data30pct,datafull}/metrics.json`，逐切片熵+Dice 在同目录 `tumor_entropy_scores.csv`，图在 `assets/figures/`。

建议先做 MVP：用现成的预计算结果和图片把页面搭起来（样例选择、原始图、分割掩码、熵热图、分诊结论、指标展示），先不接真实推理；UI 验证通过后再接真实推理。

现在还没有的：真实在线推理接口（现有代码是离线批处理，没有 HTTP 服务）、登录权限、真实影像合规处理、并发异步。这些要你们写。

最大的坑：现有代码是「训练 + 整测试集离线评估」的一体化脚本。你们别去改 `run_experiments.py` 的主流程，直接复用 `UNet`、`mc_predict` 和预处理参数，在 `src/infer.py` 基础上包一层 HTTP 服务就行。

接手顺序建议：① 跑通第八部分的验证；② 做 MVP（预计算结果展示）；③ 写推理服务（`src/` 里已经有内核）；④ 接真实推理与异步任务；⑤ 权限、日志、删除机制。

## 二、现有代码地图

入口在 `scripts/run_experiments.py`（仓库唯一权威副本）。

| 模块（函数/类） | 位置 | 负责什么 | 不负责什么 |
|---|---|---|---|
| `KiTS19SliceDataset` | 约 123 行 | 读 NIfTI、逐切片、预处理 | 只处理带标注整卷，不处理单张上传图 |
| `UNet` / `DoubleConv` | 约 192 / 179 行 | 分割模型，3 类输出 | 不做不确定性聚合 |
| `mc_predict()` | 约 457 行 | MC-Dropout 推理，返回均值/分割/方差/熵/肿瘤掩码 | 不做阈值拒绝 |
| `main()` | 约 743 行 | 训练/评估入口 | 不提供在线推理 |

离线数据流：NIfTI → 逐轴向切片 → 只保留标注非空切片 → HU clip [-200,300] → 归一化 [0,1] → resize 256×256 → `UNet` 前向 → `mc_predict` → `metrics.json`、`tumor_entropy_scores.csv`。

`mc_predict` 契约（你们要复用的部分）：

```python
def mc_predict(model, img, n_samples, device):
    # img: (1,1,H,W) 张量，已归一化 [0,1]，H=W=256
    # 返回五个 numpy 数组：
    #   mean_probs (3,H,W), mean_pred (H,W) 0/1/2,
    #   variance (H,W), entropy (H,W), pred_binary (H,W) 0/1
```

注意：`mc_predict` 内部先 `model.train()` 保持 Dropout 激活、再把 BatchNorm 切到 eval。你们必须沿用这个顺序，否则 MC-Dropout 退化成单次前向，熵会恒为 0。

预处理参数是 `run_experiments.py` 里的模块级常量：`HU_MIN/HU_MAX=-200/300`、`IMG_SIZE=256`、`NUM_CLASSES=3`。你们把这些抽成配置，别散落改。

## 三、系统架构与你们要写的部分

- 前端：你们写（页面、上传/样例选择、展示）。
- 后端 API：你们写（接收请求、调推理、返回 JSON/图）。
- 模型推理服务：复用 `src/infer.py`（加载权重、预处理、`mc_predict`、阈值分诊），你们在它外面包 HTTP。
- 预处理/后处理：`src/preprocess.py` 已经抽好，直接用。
- 文件存储、用户权限、日志监控：你们写。

两层实现：

1. MVP（先做这个）：后端只读现成结果——`results/bayes/*/tumor_entropy_scores.csv`、`assets/figures/` 里的图、`results/bayes/*/metrics.json`，前端展示「样例 CT + 分割图 + 熵热图 + 分诊结论 + 指标」。成本低，先对齐 UI 和口径。
2. 完整系统：接真实模型。后端加载 full 的 `best_unet.pth`，对上传的 NIfTI 或切片做预处理 + `mc_predict` + 阈值拒绝，返回结果。

提醒：完整系统一旦暴露真实医疗影像上传，会牵扯合规、删除、审计；MVP 用预计算结果就没这些事。

## 四、模型输入输出契约

输入（模型层）：

- 单张切片：float32 张量 shape `(1,1,256,256)`，单通道，已从 HU 裁剪 [-200,300] 并归一化到 [0,1]。
- 整卷 NIfTI：先按轴向切片、逐张做同样预处理（`src/preprocess.py` 里有 `load_volume_slices`）。
- 批量：`mc_predict` 可把 `n_samples` 次前向做成一个 batch；多患者并发目前不支持。

模型文件与加载：

- full 权重：`best_unet.pth`，约 31 MB，纯 `state_dict`，我们会发给你们。
- 加载：`UNet(dropout_rate=0.1)`，然后 `model.load_state_dict(torch.load("best_unet.pth", map_location="cpu"))`。
- 其余三组（10%、20%、30%）是研究对照，你们不用接。

输出（`mc_predict` 返回，numpy）：

- `mean_pred`：`(256,256)` 0/1/2（背景/肾脏/肿瘤）。
- `pred_binary`：`(256,256)` 0/1 肿瘤掩码。
- `entropy`：`(256,256)` 逐像素信息熵（nats）。
- `variance`：`(256,256)` 逐像素方差。
- `mean_probs`：`(3,256,256)` 平均 softmax 概率。

分诊规则：

- 切片级不确定度 = 肿瘤区域（`pred_binary==1` 或 `mean_pred==2`）内 `entropy` 的平均值。
- 把切片按肿瘤平均熵从小到大排序，固定接受比例（coverage），阈值取对应分位数的熵。
- 推荐阈值：70% 覆盖率 → 0.5804；50% 覆盖率 → 0.3673（就是 full 组 `results/bayes/datafull/tumor_entropy_scores.csv` 里肿瘤平均熵的 70%/50% 分位数）。
- 判定：肿瘤平均熵 ≥ 阈值 → 「建议转诊/人工复核」；< 阈值 → 「可采纳 AI 分割结果」。
- MC 采样次数用 10，依据见 `results/mc_sampling_results.json`。

接口草案（还没实现，给你们参考）：

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

错误处理：空文件/非 NIfTI/尺寸异常返回 4xx；推理超时返回 5xx 且不留临时文件；没检测到肿瘤时 `decision` 不要写「正常」，写「未检测到肿瘤区域」。

资源：GPU 上单张 256×256 切片 10 次 MC 前向约几十毫秒；CPU 慢很多，所以 MVP 先预计算。

## 五、页面与用户流程建议

- 首页/样例选择：从预计算样例里选一个展示。
- 上传页（完整系统）：上传 NIfTI/切片，后端推理。
- 结果页：并排展示原始 CT、分割掩码、熵热图，显示肿瘤平均熵、阈值、覆盖率、分诊结论。
- 阈值/覆盖率设置：滑块调覆盖率，映射到熵阈值，实时更新「接受/转诊」。
- 指标页：四组 Dice 表、Risk-Coverage（`assets/figures/risk_coverage.png`）、可靠性图（`assets/figures/reliability.png`）、两张数据标度图（`data_scaling_dice`、`data_scaling_uncertainty`）。
- 下载：掩码图、熵图、结果 JSON。

现成的和要你们写的：

- 现成：分割掩码、熵图、指标表、Risk-Coverage 曲线（`scripts/plot_*.py` 离线生成）。
- 要你们写：上传推理接口、阈值滑块、结果下载、异步任务。
- 只做研究演示：阈值调整、覆盖率权衡展示。
- 不建议开放给终端用户：真实医疗影像上传、自动临床诊断结论。

页面必须显著标注「研究原型，仅供科研演示，不构成临床诊断结论」。

## 六、仓库里有什么、你们还要做什么

仓库里已经有：

- `src/preprocess.py`、`src/infer.py`：预处理和在线推理入口，你们直接在它外面包 HTTP 就行。
- `sample_data/`：一张脱敏样例切片 + 预期输出，给你们对答案（见第七部分）。
- `scripts/`：训练/评估/画图脚本，`verify_metrics.py` 是四组指标自检。
- `results/`：四组 bayes（10%/20%/30%/full）+ 三组 vanilla 指标和逐切片熵+Dice。
- `assets/figures/`：Risk-Coverage、Dice 对比、可靠性、覆盖图、两张数据标度图。
- `tests/`、`requirements.txt`、`docs/`。

你们要做的：

- 权重：我们会私下发给你们，拿到放 `models/`。
- 环境：`pip install -r requirements.txt`，torch 按你们机器装 CPU/CUDA 对应版本。
- 写后端 HTTP + 前端页面（`src/infer.py` 是命令行入口，不是服务）。

## 七、sample_data：给你们对答案的样例

我们已经在 `sample_data/` 放了一份脱敏样例，你们不用装整套数据集就能验证自己的预处理和推理对不对。文件说明见 `sample_data/README.md`。

用法：① 拿 `slice_hounsfield.npy` 跑你们自己的预处理，结果应该和 `slice_preprocessed.npy` 一致；② 把 `slice_preprocessed.npy` 喂给模型，对比 `mean_pred`、`pred_binary`、`entropy`；③ 按 `sample_result.json` 里的阈值算肿瘤平均熵，结论应该和里面的 `decision` 一致。

来源：KiTS19 公开数据集 `case_00000` 第 311 个轴向切片，full 权重、MC=10，已脱敏。

## 八、你们克隆后先做这几步

1. `pip install -r requirements.txt`。
2. 打开 `results/bayes/datafull/metrics.json`，确认 `dice=0.748`、`auroc_entropy_wrong=0.5562`。
3. 打开 `results/stats_validation_results.json`，确认 `data30pct.auroc_entropy_wrong.point=0.6013`、`data10pct.auroc_entropy_wrong.point=0.4878`。
4. 打开 `assets/figures/`，确认能看到 `risk_coverage.png`、`reliability.png`、`data_scaling_dice.png`、`data_scaling_uncertainty.png`。

（`verify_metrics.py` 只读仓库 `results/`，不需要权重和数据；`mc_sampling_study.py`、`plot_overlays.py` 这类复现脚本才需要本地数据集和权重，你们用不到。）

## 九、仓库结构（目标态）

```text
repo/
  README.md
  LICENSE
  requirements.txt
  src/
    infer.py
    preprocess.py
  scripts/
    run_experiments.py
    verify_metrics.py
    stats_validation.py
    mc_sampling_study.py
    make_sample_data.py
    plot_results.py
    plot_reliability.py
    plot_overlays.py
    plot_data_scaling.py
    figstyle.py
  tests/
  configs/              # 阈值/覆盖率配置（待抽）
  sample_data/
  results/
  assets/figures/
  docs/
    WEB_HANDOFF.md
    PAPER_HANDOFF.md
  deployment/           # 部署说明（后续）
```

不要放进仓库：原始数据集、实验输出目录、venv、缓存、含个人信息的文档、任何真实影像。

## 十、安全、隐私、许可与研究边界

- 患者数据：KiTS19 是公开数据集，但真实影像不上传仓库，只放脱敏样例和结果。
- 数据授权：KiTS19 公开≠可再分发原始影像。
- 开源：代码、结果、权重我们都确认可以公开。
- 登录/权限/审计：研究原型至少加访问日志和删除机制，要不要登录听导师的。
- 上传真实医疗数据：默认不允许；真要接，先做伦理/隐私评估并加删除与审计。
- 所有页面标注「研究原型，不构成临床诊断结论」，指标别写成「临床验证结果」。

待确认：伦理审批文件、数据使用授权（KiTS19 使用条款）。公开问题已经敲定。

## 十一、验收标准

| 验收项 | 验证方法 |
|---|---|
| 能装上 | 干净环境 `pip install -r requirements.txt` 成功 |
| 数字一致 | 页面展示的四组 Dice 与 `results/bayes/*/metrics.json` 一致 |
| 页面正确 | 显示 `assets/figures/risk_coverage.png` 和四组 Dice 表 |
| 异常处理 | 空文件/非 NIfTI 返回 4xx，超时返回 5xx 且不残留临时文件 |
| 不依赖本机路径 | 克隆后按 README 能跑，无任何绝对路径 |
| 无敏感信息 | 仓库不含凭证、真实影像、个人信息 |
| MVP 能演示 | 不接真实推理也能用预计算结果演示完整流程 |
| 真实推理能跑 | 用我们发的 full 权重 + sample_data，单张切片出分割/熵/分诊结论 |

这些主要靠你们自测；「无敏感信息」这条我们这边把关。

## 十二、已知限制与待解决

开始前要确认的：

- 上传真实医疗影像：默认不允许。
- 公开授权：已确认，代码/结果/权重都可公开。

MVP 前要解决的：

- 把预处理参数抽成配置（`src/preprocess.py` 已经抽了函数，参数可以再抽成 config）。
- 定好 MVP 是先纯预计算还是直接接真实推理。
- 页面显著标注「研究原型，非临床诊断」。

完整系统上线前要解决的：

- 推理性能与并发、异步任务、超时与错误恢复。
- 登录/权限、上传文件保存与删除、审计日志。
- 浏览器兼容、技术栈选型、模型版本管理。
