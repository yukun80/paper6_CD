# 当前主线训练命令

当前主线默认包含：
- `soft alignment` on（作用于 `p1/p2/p3`）
- `multilevel_v2` DINO 协同
- `hybrid refiner`
- tiny-heavy 训练配置

如果要做 `soft alignment` 消融，只使用单开关：
- `SOFT_ALIGNMENT=1`：默认主线
- `SOFT_ALIGNMENT=0`：关闭 soft alignment，此时模型会自动切到轻量双时相协同适配器，而不是裸 direct diff

## 1. 默认 tiny-heavy 训练
适合当前“小尺度内涝点优先”的主线设置。默认会使用：
- `BEST_METRIC=tiny_combo`
- `EVAL_FG_THRESHOLD=0.40`
- 更晚的 `topo warmup`
- 更弱的 `topo / consistency` 约束

```bash
RUN_NAME=S1GFloods-hybrid-mv2-tiny-b4 \
BACKBONE_WEIGHT=pretrained/efficientnet_b0_ra-3dd342df.pth \
BATCH_SIZE=4 \
bash ChangeDINO-main/trainval_s1gfloods.sh
```

## 2. soft alignment 消融
只改一个总开关，其他训练配置保持一致。

```bash
RUN_NAME=S1GFloods-hybrid-mv2-noalign-b4 \
BACKBONE_WEIGHT=pretrained/efficientnet_b0_ra-3dd342df.pth \
BATCH_SIZE=4 \
SOFT_ALIGNMENT=0 \
bash ChangeDINO-main/trainval_s1gfloods.sh
```

## 3. 整体 IoU 优先训练
如果目标是整体分割精度，而不是 tiny flood 召回，可切回 `iou_1` 作为主选择指标。

```bash
RUN_NAME=S1GFloods-hybrid-mv2-iou-b6 \
BACKBONE_WEIGHT=pretrained/efficientnet_b0_ra-3dd342df.pth \
BATCH_SIZE=6 \
BEST_METRIC=iou_1 \
EVAL_FG_THRESHOLD=0.50 \
TOPO_WARMUP_EPOCHS=10 \
TOPO_LOSS_WEIGHT=0.5 \
BRANCH_CONSISTENCY_WEIGHT=0.05 \
CONSISTENCY_WARMUP_EPOCHS=15 \
bash ChangeDINO-main/trainval_s1gfloods.sh
```

## 4. 低显存训练
结构保持不变，只降低 batch size。

```bash
RUN_NAME=S1GFloods-hybrid-mv2-tiny-b4 \
BACKBONE_WEIGHT=pretrained/efficientnet_b0_ra-3dd342df.pth \
BATCH_SIZE=4 \
bash ChangeDINO-main/trainval_s1gfloods.sh
```

## checkpoint 说明
- 默认 `best.pth` 由 `BEST_METRIC` 决定。
- 训练过程还会额外保存：
  - `best_iou`
  - `best_tiny_recall`
  - `best_tiny_combo`
- 汇总信息保存在对应 checkpoint 目录下的 `best_metrics.json`。
- checkpoint 的 `model_config` 会记录 `disable_soft_alignment`，推理阶段会按训练时的 alignment 开关自动重建模型。
