# 当前主线训练命令

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

## 2. 整体 IoU 优先训练
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

## 3. 低显存训练
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
