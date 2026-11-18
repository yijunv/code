# sample_data 样例说明

这是一份脱敏样例，用来验证 Web 端自己写的预处理和推理对不对，不需要整套数据集。

## 文件

- `slice_hounsfield.npy`：一张轴向切片的原始 HU 值（未缩放）。
- `slice_preprocessed.npy`：预处理后的模型输入（256x256，值域 [0,1]）。口径是 HU 裁剪 [-200,300] -> 归一化 [0,1] -> 双线性缩放到 256。
- `slice.png`：切片灰度预览。
- `mean_pred.npy`：模型分割结果，0 背景 / 1 肾脏 / 2 肿瘤（256x256）。
- `pred_binary.npy`：肿瘤二值掩码 0/1。
- `mask.png`：肿瘤掩码预览。
- `entropy.npy`：逐像素熵（256x256）。
- `entropy.png`：熵图预览。
- `sample_result.json`：这张切片的分诊结论，字段和 `/api/infer` 返回一致。

## 怎么用

1. 拿 `slice_hounsfield.npy` 跑你自己写的预处理，结果应该和 `slice_preprocessed.npy` 基本一致（最多差一点浮点误差）。
2. 把 `slice_preprocessed.npy` 喂给模型推理，对比 `mean_pred`、`pred_binary`、`entropy`，应该和这里的 .npy 一致。
3. 按 `sample_result.json` 里的阈值算肿瘤平均熵，结论应该和里面的 `decision` 一致。

## 来源

KiTS19 公开数据集 `case_00000` 的第 311 个轴向切片（已脱敏，不含患者信息），用 full 权重、MC=10 推理。
