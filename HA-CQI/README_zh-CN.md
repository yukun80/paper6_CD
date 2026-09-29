# HA-CQI

HA-CQI 使用灾前和灾后合成孔径雷达（SAR）影像进行二值洪水变化分割。
各命令示例独立使用，从仓库根目录开始执行。包含 `cd HA-CQI` 的示例在模型目录运行。
检查点路径中的 `<resolved_run_name>` 和 `<run>` 应替换为训练日志中的实际名称。

## 模型设计

- **分层 CNN-DINO 语义编码器**：共享 EfficientNet-B2 与 CNN-FPN，融合冻结 DINOv3 第 `[5,8,11]` 层语义特征，输出五级特征金字塔。
- **协调对齐（HA）**：对浅层双时相特征进行共享风格校准，并支持可变形软对齐。
- **变化查询交互（CQI）**：在五级特征上通过可学习变化查询建模双时相差异。
- **Omni-Scale State-Space Change Decoder（OSCD）**：聚合区域上下文，逐级恢复局部边界。
- **多尺度辅助头**：在五级特征上提供辅助监督。

模型入口为 `model/architectures/ha_cqi.py`，训练与推理逻辑位于 `model/engine.py`。

## 数据集

```text
datasets/<DATASET_NAME>/
├── train/{A,B,label}
└── val/{A,B,label}
```

默认数据集为 `datasets/S1GFloods_CD_DINO_BG_75_25_`，训练集和验证集分别位于
`train/{A,B,label}` 和 `val/{A,B,label}`。A、B 和 label 须完整同名配对，实际训练成员以运行时扫描结果为准。
`STATS_MODE=auto` 自动维护归一化统计，每次运行的 `data_snapshot.json` 记录实际样本与统计信息。
广西数据包含参与训练的样本，其整景结果应按相应数据使用条件解释。

## 数据制备

`scripts/prepare_fused_sar_cd_dataset.py` 对新制备的 VarFloods 切片采用灾前和灾后整景联合
P2/P98 拉伸，同一场景的切片复用拉伸参数。S1GFloods 影像 PNG 原样复制。

独立推理场景使用 `prepare_tiles.py`，默认切片大小为 256×256，步长为 128，最小共同有效比例为 0.01，
采用整景联合 P2/P98 拉伸。广西示例：

```bash
python HA-CQI/scripts/prepare_tiles.py \
  --pre-image datasets/LT1_Guangxi/LT_Guangxi_pre.tif \
  --post-image datasets/LT1_Guangxi/LT_Guangxi_post.tif \
  --output-dir datasets/LT1_Guangxi_CD_infer \
  --scene-id LT1_Guangxi
```

输出包括 `test/A`、`test/B`、`test/valid_mask`、`tile_manifest.csv` 和 `prepare_report.json`。
`--dry-run` 仅统计。`--overwrite` 重建影像及有效区切片并更新清单和报告，保留已有标签。
源影像或切片参数改变后，应重新生成标签。输出目录须与源影像目录分开。

已有影像切片时，使用同网格标签生成标签切片：

```bash
python HA-CQI/scripts/prepare_sar_scene_label_infer.py \
  --src-root datasets/LT1_Guangxi \
  --label-image LT_Guangxi_Label.tif \
  --tiles-root datasets/LT1_Guangxi_CD_infer \
  --strict
```

标签写入 `test/label`、`test/label_tif` 并补充清单。已有非空标签目录时需显式传入
`--overwrite`，该参数仅替换标签目录。

## 环境与预训练权重

在已有的 `hacqi` conda 环境中安装并检查依赖：

```bash
cd HA-CQI
conda activate hacqi
python -m pip install -r requirements-oscd.txt
python -c "import torch, timm, mmcv, selective_scan_cuda; print(torch.__version__)"
python -m pip check
```

依赖配置对应 Python 3.11、Torch 2.4、CUDA 12 和 CXX11 ABI false，使用 `mamba_ssm 2.2.4`
预编译 wheel，并固定 `einops 0.8.1`、`ninja 1.13.0` 和 `transformers 4.44.2`。
GPU 训练需要可用的 selective-scan CUDA kernel，CPU 路径用于测试。

主线使用 `efficientnet_b2`，以下预训练权重需在本地准备：

```text
HA-CQI/dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth
HA-CQI/pretrained/efficientnet_b2_ra-bcdf34b7.pth
```

## 训练

默认启动：

```bash
cd HA-CQI
conda activate hacqi
bash trainval_s1gfloods.sh
```

常用参数覆盖示例：

```bash
cd HA-CQI
DATASET_NAME=S1GFloods_CD_DINO_BG_75_25_ \
DATA_ROOT=../datasets \
STATS_MODE=auto \
RUN_NAME=S1GFloods-HA-CQI-B2-OSCD-DINO5-8-11-CLEAN-s1 \
BATCH_SIZE=12 \
NUM_WORKERS=8 \
LR=1e-4 \
AMP_DTYPE=bf16 \
EVAL_FG_THRESHOLD=0.40 \
bash trainval_s1gfloods.sh
```

训练在验证集的 `0.05–0.95` 阈值网格上以 `0.01` 为步长，按
`Flood IoU → Precision → 较高 threshold` 选择 `best_primary`。
`EVAL_FG_THRESHOLD=0.40` 仅用于验证诊断。每次训练从 CNN/DINO 预训练权重开始，当前入口不支持续训。
测试与推理使用声明 `efficientnet_b2 + imagenet + oscd_v1` 的 checkpoint v2。

训练输出位于 `HA-CQI/checkpoints/<resolved_run_name>/`，包括：

```text
<run>_efficientnet_b2_best_primary.pth
<run>_efficientnet_b2_last.pth
<run>_efficientnet_b2_epoch10.pth ...
selection.json
metrics.jsonl
options.json
data_snapshot.json
vis/
```

关闭 HA 软对齐的消融命令：

```bash
cd HA-CQI
SOFT_ALIGNMENT=0 RUN_NAME=S1GFloods-HA-CQI-noalign bash trainval_s1gfloods.sh
```

## 测试与单对影像推理

测试结果通过 `--save_test` 保存至运行目录的 `pred/`：

```bash
cd HA-CQI
python test.py \
  --checkpoint checkpoints/<resolved_run_name>/<run>_efficientnet_b2_best_primary.pth \
  --gpu_ids 0 \
  --save_test
```

对一组灾前和灾后影像执行推理：

```bash
cd HA-CQI
python run.py \
  --checkpoint checkpoints/<resolved_run_name>/<run>_efficientnet_b2_best_primary.pth \
  --img_A /path/to/pre_image.tif \
  --img_B /path/to/post_image.tif \
  --output outputs/run_pred.png \
  --gpu_ids 0
```

## SAR 整景瓦片推理

瓦片目录须包含 `tile_manifest.csv`，且清单引用的 A/B 影像与 `valid_mask` 均可读取。
河南 GF3 场景示例：

```bash
python HA-CQI/scripts/infer_sar_scene_tiles.py \
  --tiles-root datasets/GF3_Henan_CD_infer \
  --checkpoint HA-CQI/checkpoints/<b2_run>/<b2_run>_efficientnet_b2_best_primary.pth \
  --gpu_ids 0 \
  --batch_size 8 \
  --output-dir HA-CQI/outputs/gf3_henan_corrected
```

`<b2_run>` 替换为实际运行名。其他场景使用同一入口，替换输入和输出目录即可。
归一化统计默认从检查点元数据读取。推理阈值来自显式 `--threshold` 或
`meta.selection.threshold`，两者均缺失时程序报错。

默认保存切片与整景结果，整景文件位于 `<output-dir>/mosaic/`：

```text
change_binary_raw.tif/png  # 阈值化后的原始二值结果
change_binary.tif/png      # 一致性伪斑过滤后的二值结果
```

`<output-dir>/infer_report.json` 记录检查点、阈值、输入输出路径和模型配置。
定量评估使用二值 TIFF 的有效掩码，PNG 中 NoData 与前景均显示白色。
`--disable_blob_filter` 关闭伪斑过滤，`--skip-tiles` 跳过切片级 PNG/TIF 保存。

## 整景评估

```bash
python HA-CQI/scripts/evaluate_sar_scene.py \
  --prediction-dir HA-CQI/outputs/gf3_zhuozhou \
  --ground-truth datasets/GF3_Zhuozhou/GF3_Zhuozhou_label.tif
```

当前推理报告对应二值图评价，排除预测和标签 NoData（包括标签未知值 `3`），
输出 TP/FP/TN/FN、IoU、F1、P/R、准确率及有效和忽略像素数量。
若需评价原始二值图，可显式指定：

```bash
python HA-CQI/scripts/evaluate_sar_scene.py \
  --binary HA-CQI/outputs/<run>/mosaic/change_binary_raw.tif \
  --ground-truth datasets/GF3_Zhuozhou/GF3_Zhuozhou_label.tif
```

## 代码校验

修改核心代码后，在 `HA-CQI` 目录执行：

```bash
python -m py_compile \
  option.py \
  model/architectures/ha_cqi.py \
  model/backbones/builder.py \
  model/modules/*.py \
  model/decode_heads/*.py \
  model/engine.py \
  model/checkpointing.py \
  data/transform.py \
  utils/flood_evaluation.py \
  trainval.py \
  test.py \
  run.py \
  scripts/diagnose_cross_domain_features.py

python -m unittest tests/test_pipeline_contracts.py

bash -n trainval_s1gfloods.sh trainval.sh
```
