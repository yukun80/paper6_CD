# 训练

[返回目录](README.md) · [项目首页](../../README.md)

## 默认启动

```bash
cd HarmoSSM
conda activate hacqi
bash trainval_s1gfloods.sh
```

## 明确指定训练参数

```bash
cd HarmoSSM
conda activate hacqi

DATASET_NAME=S1GFloods_CD_DINO_BG_75_25_ \
DATA_ROOT=../datasets \
STATS_MODE=auto \
RUN_NAME=S1GFloods-HarmoSSM-B2-OSCD-DINO5-8-11-CLEAN-s1 \
DINO_ARCH=dinov3_vits16 \
DINO_WEIGHT=dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
DINO_FUSION_LAYERS="5 8 11" \
BACKBONE_WEIGHT=pretrained/efficientnet_b2_ra-bcdf34b7.pth \
BATCH_SIZE=12 \
NUM_WORKERS=8 \
LR=1e-4 \
HEAD_LR_MULT=2.0 \
AMP=1 \
AMP_DTYPE=bf16 \
SEED=1 \
bash trainval_s1gfloods.sh
```

## 选择最佳模型

训练在验证集的 `0.05–0.95` 阈值网格上以 `0.01` 为步长，按
`Flood IoU → Precision → 较高 threshold` 选择 `best_primary`。
`EVAL_FG_THRESHOLD=0.40` 仅用于验证诊断。每次训练从 CNN/DINO 预训练权重开始，当前入口不支持续训。
测试与推理使用声明 `efficientnet_b2 + imagenet + oscd_v1` 的 checkpoint v2。

`EVAL_FG_THRESHOLD`、`THRESHOLD_MIN/MAX/STEP` 可通过环境变量覆盖。

## 关闭软对齐

```bash
cd HarmoSSM
SOFT_ALIGNMENT=0 RUN_NAME=S1GFloods-HarmoSSM-noalign bash trainval_s1gfloods.sh
```

## 修改边界

- 不重建数据集，不用 manifest/fingerprint 阻止训练；
- 不改 HA、CQI、OSCD、输入尺寸或 SAR 色调映射；
- 不启用新的背景抑制 loss；
- 只有清洗后 baseline 仍仅在跨域场景出现 FP，才单独测试
  `RADIOMETRIC_JITTER_MODE=independent`；
- FP、tiny recall 和跨域泛化收益需要完整训练验证，轻量工程测试不代表算法指标提升。

相关内容：[数据准备](data.md) · [训练记录](checkpoints.md) · [短步验证](validation.md)
