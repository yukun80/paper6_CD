# HA-CQI

HA-CQI 是本目录当前使用的合成孔径雷达（SAR）洪水变化检测算法，执行二值洪水变化分割。

## 模型设计

当前模型名称为 **HA-CQI**，由以下部分组成：

- **模块 I：分层 CNN-DINO 语义编码器**：共享的灾前/灾后编码器融合
  EfficientNet-B0 或 MobileNetV2 的 CNN-FPN 特征金字塔与 DINOv3 语义特征，输出
  分辨率对齐的 `P1-P5` 多尺度特征。
- **模块 II：协调对齐（Harmonized Alignment, HA）**：先在浅层特征上执行成对共享的
  风格校准，缓解 SAR 辐射差异；随后可选地在 `P1/P2/P3` 上执行可变形软对齐。
- **模块 III：变化查询交互（Change Query Interaction, CQI）**：可学习的变化查询通过
  双向注意力与多尺度特征差异交互，为解码器提供显式的变化感知上下文。
- **Mask2Former 风格分割头**：轻量级查询掩膜解码器基于高分辨率掩膜特征和经 CQI 增强的
  上下文预测二值洪水变化 logits。
- **尺度感知辅助头**：在 `P1-P5` 上提供辅助预测，使小型内涝斑块和大范围淹没区都能获得
  有效监督。

主要代码路径如下：

```text
model/architectures/ha_cqi.py
model/engine.py
model/modules/harmonized_alignment.py
model/modules/change_query_interaction.py
model/decode_heads/mask2former_change_head.py
trainval.py
test.py
run.py
trainval_s1gfloods.sh
```

## 数据集

训练数据加载器要求以下标准二值变化检测目录结构：

```text
datasets/<DATASET_NAME>/
├── train/{A,B,label}
├── val/{A,B,label}
├── train_tif/{A,B,label}
├── val_tif/{A,B,label}
├── manifest_train.csv
├── manifest_val.csv
└── channel_stats_s1gfloods_train.json
```

默认训练数据集为 `datasets/S1GFloods_CD_DINO_BG_75_25`。该版本保留全部 581 个有效
全背景 tile，并沿用确定性的全局随机 `75/25` 划分；旧数据集不会被覆盖。

若尚未构建融合后的 S1GFloods 数据集，请从仓库根目录运行 HA-CQI 的数据预处理工具，并计算
训练集通道统计量：

```bash
python HA-CQI/scripts/prepare_fused_sar_cd_dataset.py \
  --s1gfloods-root datasets/S1GFloods \
  --varfloods-root datasets/VarFloods \
  --out-root datasets/S1GFloods_CD_DINO_BG_75_25 \
  --tile-size 256 \
  --stride 128 \
  --train-ratio 0.75 \
  --seed 42 \
  --dry-run

python HA-CQI/scripts/prepare_fused_sar_cd_dataset.py \
  --s1gfloods-root datasets/S1GFloods \
  --varfloods-root datasets/VarFloods \
  --out-root datasets/S1GFloods_CD_DINO_BG_75_25 \
  --tile-size 256 \
  --stride 128 \
  --train-ratio 0.75 \
  --seed 42

python HA-CQI/scripts/compute_s1gfloods_cd_stats.py \
  --data-root datasets/S1GFloods_CD_DINO_BG_75_25 \
  --split train \
  --output datasets/S1GFloods_CD_DINO_BG_75_25/channel_stats_s1gfloods_train.json
```

预期计数为 train/val `5064/1688`、全背景 `437/144`。构建报告会记录数据指纹、来源/区域/
背景分布、2,044 对跨 split 重叠窗口，以及旧数据集四个控制文件的构建前后 SHA256。
默认不允许覆盖已存在的新目录。

## 预训练权重

默认本地权重路径如下：

```text
HA-CQI/dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth
HA-CQI/pretrained/efficientnet_b0_ra-3dd342df.pth
```

`efficientnet_b0` 需要本地 PyTorch `.pth` 或 `.pt` 权重文件；训练入口不会自动下载权重。

## 环境迁移

在新机器获得 `dl311_dino.tar.gz` 后，执行：

```bash
conda activate base
mkdir -p "$CONDA_PREFIX/envs/dl311_dino"
tar -xzf dl311_dino.tar.gz -C "$CONDA_PREFIX/envs/dl311_dino"
"$CONDA_PREFIX/envs/dl311_dino/bin/conda-unpack"
conda activate "$CONDA_PREFIX/envs/dl311_dino"
```

## 训练

从 `HA-CQI` 目录运行：

```bash
cd HA-CQI
bash trainval_s1gfloods.sh
```

常用环境变量覆盖示例：

```bash
cd HA-CQI
DATASET_NAME=S1GFloods_CD_DINO_BG_75_25 \
DATA_ROOT=../datasets \
RUN_NAME=S1GFloods-HA-CQI-vits16 \
BATCH_SIZE=6 \
EVAL_FG_THRESHOLD=0.40 \
bash trainval_s1gfloods.sh
```

默认训练脚本使用以下关键配置：

- `DATASET_NAME=S1GFloods_CD_DINO_BG_75_25`
- `DATA_ROOT=../datasets`
- `STATS_FILE=../datasets/S1GFloods_CD_DINO_BG_75_25/channel_stats_s1gfloods_train.json`
- `DINO_ARCH=dinov3_vits16`
- `DINO_WEIGHT=dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth`
- `BACKBONE=efficientnet_b0`
- `BACKBONE_WEIGHT=pretrained/efficientnet_b0_ra-3dd342df.pth`
- `NUM_CHANGE_QUERIES=16`
- `MASK_QUERIES=32`
- `MASK_DECODER_LAYERS=3`
- `EVAL_FG_THRESHOLD=0.40`（仅用于 validation 细粒度诊断）
- `SEED=1`
- `FOCAL_BG_WEIGHT=0.25`、`FOCAL_FG_WEIGHT=0.75`
- `DINO_INPUT_NORM=shared`

新训练在 validation 的 `0.05–0.95`（步长 `0.01`）阈值网格上按
`Flood IoU → Precision → 较高 threshold` 联合选择，只生成：

```text
<run>_<backbone>_best_primary.pth
<run>_<backbone>_last.pth
<run>_<backbone>_epoch10.pth ...
selection.json
metrics.jsonl
options.json
```

完整续训使用 `RESUME=/path/to/*_last.pth`；仅初始化网络使用
`python trainval.py ... --init_checkpoint /path/to/checkpoint.pth`。resume 会校验数据指纹和
训练/loss/模型配置。`--resume`、`--init_checkpoint`、测试和推理入口均只接受 checkpoint v2。
`S1GFloods-HA-CQI-vits16-20260427/` 及其结果保留为磁盘归档，当前代码不支持加载。

训练结果写入：

```text
HA-CQI/checkpoints/<resolved_run_name>/
HA-CQI/checkpoints/<resolved_run_name>/vis/
```

关闭 HA 软对齐以进行消融实验：

```bash
cd HA-CQI
SOFT_ALIGNMENT=0 RUN_NAME=S1GFloods-HA-CQI-noalign bash trainval_s1gfloods.sh
```

## 测试

使用训练日志输出的实际运行目录名：

```bash
cd HA-CQI
python test.py \
  --checkpoint checkpoints/<resolved_run_name>/<run>_efficientnet_b0_best_primary.pth \
  --gpu_ids 0 \
  --save_test
```

检查点元数据保存了重建 HA-CQI 架构所需的选项。启用 `--save_test` 时，测试预测结果保存到：

```text
HA-CQI/checkpoints/<resolved_run_name>/pred/
```

## 单对影像推理

`run.py` 可直接对一组灾前/灾后影像执行推理：

```bash
cd HA-CQI
python run.py \
  --checkpoint checkpoints/<resolved_run_name>/<run>_efficientnet_b0_best_primary.pth \
  --img_A /path/to/pre_image.tif \
  --img_B /path/to/post_image.tif \
  --output outputs/run_pred.png \
  --gpu_ids 0
```

## GF3 整景瓦片推理

整景推理使用与 HA-CQI 相同的模型重建逻辑，将重叠瓦片的预测拼接回整景变化图。瓦片目录必须
包含 `tile_manifest.csv`，且清单中引用的灾前 A 图像、灾后 B 图像和 `valid_mask` 文件均可读取。

河南 GF3 场景示例：

```bash
python HA-CQI/scripts/infer_gf3_henan_tiles.py \
  --tiles-root datasets/GF3_Henan_CD_infer \
  --checkpoint HA-CQI/checkpoints/S1GFloods-HA-CQI-corrected-baseline-s1-20260822/S1GFloods-HA-CQI-corrected-baseline-s1-20260822_efficientnet_b0_best_primary.pth \
  --gpu_ids 0 \
  --batch_size 8 \
  --output-dir HA-CQI/outputs/gf3_henan_corrected
```

stats 默认从 checkpoint v2 metadata 解析。推理阈值只允许两种来源：显式
`--threshold`，或 checkpoint 的 `meta.selection.threshold`；两者都缺失时直接失败。

涿州 GF3 场景使用同一入口，仅替换瓦片根目录和输出目录：

```bash
python HA-CQI/scripts/infer_gf3_henan_tiles.py \
  --tiles-root datasets/GF3_Zhuozhou_CD_infer \
  --checkpoint HA-CQI/checkpoints/S1GFloods-HA-CQI-corrected-baseline-s1-20260822/S1GFloods-HA-CQI-corrected-baseline-s1-20260822_efficientnet_b0_best_primary.pth \
  --gpu_ids 0 \
  --batch_size 8 \
  --output-dir HA-CQI/outputs/gf3_zhuozhou_corrected
```

广西 LT-1 场景测试：

```bash
python HA-CQI/scripts/infer_gf3_henan_tiles.py \
  --tiles-root datasets/LT1_Guangxi_CD_infer \
  --checkpoint HA-CQI/checkpoints/S1GFloods-HA-CQI-corrected-baseline-s1-20260822/S1GFloods-HA-CQI-corrected-baseline-s1-20260822_efficientnet_b0_best_primary.pth \
  --gpu_ids 0 \
  --batch_size 8 \
  --skip-tiles \
  --output-dir HA-CQI/outputs/lt1_guangxi_corrected
```

默认会保存切片级结果和整景拼接结果。整景输出位于 `<output-dir>/mosaic/`，包括：

```text
change_prob.tif            # 拼接后的前景概率图
change_binary_raw.tif/png  # 阈值化后的原始二值结果
change_binary.tif/png      # 一致性伪斑过滤后的二值结果
```

`<output-dir>/infer_report.json` 记录检查点、阈值、瓦片数、输出路径、拼接源影像和模型配置。
如需只保留原始拼接结果，可传入 `--disable_blob_filter`；如需跳过切片级 PNG/TIF 保存，可传入
`--skip-tiles`。

## 整景评估

```bash
python HA-CQI/scripts/evaluate_sar_scene.py \
  --prediction-dir HA-CQI/outputs/gf3_zhuozhou \
  --ground-truth datasets/GF3_Zhuozhou/GF3_Zhuozhou_label.tif
```

评估器排除 metadata nodata 和历史标签值 `3`，并同时报告 raw/filtered 的 IoU、F1、P/R、
PR-AUC、Brier、ECE、背景 tile FP、FP 像素比例、最大/P95 FP 连通域及 tiny/small/large
coverage recall。河南只用于外部校准/诊断，涿州作为锁定测试，不参与训练 epoch 选择。

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
  trainval.py \
  test.py \
  run.py

python -m unittest tests/test_pipeline_contracts.py

bash -n trainval_s1gfloods.sh trainval.sh
```
