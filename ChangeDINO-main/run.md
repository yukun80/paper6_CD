# ── 默认训练（第二阶段主线：hybrid + micro_gate + multilevel_v2，batch=6）──
RUN_NAME=S1GFloods-hybrid-mv2-b6 \
BACKBONE_WEIGHT=pretrained/efficientnet_b0_ra-3dd342df.pth \
BATCH_SIZE=6 \
bash ChangeDINO-main/trainval_s1gfloods.sh

# ── 大 batch 训练 ──
RUN_NAME=S1GFloods-hybrid-mv2-b10 \
BACKBONE_WEIGHT=pretrained/efficientnet_b0_ra-3dd342df.pth \
BATCH_SIZE=10 \
bash ChangeDINO-main/trainval_s1gfloods.sh

# ── 调整拓扑图粒度 ──
RUN_NAME=S1GFloods-hybrid-mv2-g8k8-b6 \
BACKBONE_WEIGHT=pretrained/efficientnet_b0_ra-3dd342df.pth \
BATCH_SIZE=6 \
TOPO_GRID=8 TOPO_K=8 \
bash ChangeDINO-main/trainval_s1gfloods.sh

# ── 收紧一致性 warmup，观察早期精度是否回升 ──
RUN_NAME=S1GFloods-hybrid-mv2-warm5-b6 \
BACKBONE_WEIGHT=pretrained/efficientnet_b0_ra-3dd342df.pth \
BATCH_SIZE=6 \
CONSISTENCY_WARMUP_EPOCHS=5 \
bash ChangeDINO-main/trainval_s1gfloods.sh

# ── 退回最小 tiny prior 复杂度：保持主线结构，只关掉 micro gate 观察对比 ──
RUN_NAME=S1GFloods-hybrid-mv2-nomicro-b6 \
BACKBONE_WEIGHT=pretrained/efficientnet_b0_ra-3dd342df.pth \
BATCH_SIZE=6 \
MICRO_GATE=0 \
bash ChangeDINO-main/trainval_s1gfloods.sh

# ── 退回旧版单层 DINO 协同（兼容对比）──
RUN_NAME=S1GFloods-hybrid-legacy-b6 \
BACKBONE_WEIGHT=pretrained/efficientnet_b0_ra-3dd342df.pth \
BATCH_SIZE=6 \
DINO_COLLAB_MODE=legacy \
BRANCH_CONSISTENCY_WEIGHT=0.0 \
bash ChangeDINO-main/trainval_s1gfloods.sh

# ── 自定义分层对比度池化核大小（P2/P3/P4/P5）──
RUN_NAME=S1GFloods-hybrid-mv2-p3555-b6 \
BACKBONE_WEIGHT=pretrained/efficientnet_b0_ra-3dd342df.pth \
BATCH_SIZE=6 \
bash ChangeDINO-main/trainval_s1gfloods.sh \
  --contrast_pool_sizes 3 5 5 5

# ── 更低显存备选：保持结构不变，仅进一步降 batch ──
RUN_NAME=S1GFloods-hybrid-mv2-b4 \
BACKBONE_WEIGHT=pretrained/efficientnet_b0_ra-3dd342df.pth \
BATCH_SIZE=4 \
bash ChangeDINO-main/trainval_s1gfloods.sh

# ── 串行对比实验：legacy vs multilevel_v2 ──
RUN_NAME=S1GFloods-hybrid-legacy-b6 \
BACKBONE_WEIGHT=pretrained/efficientnet_b0_ra-3dd342df.pth \
BATCH_SIZE=6 \
DINO_COLLAB_MODE=legacy \
BRANCH_CONSISTENCY_WEIGHT=0.0 \
bash ChangeDINO-main/trainval_s1gfloods.sh && \
RUN_NAME=S1GFloods-hybrid-mv2-b6 \
BACKBONE_WEIGHT=pretrained/efficientnet_b0_ra-3dd342df.pth \
BATCH_SIZE=6 \
bash ChangeDINO-main/trainval_s1gfloods.sh

# 训练结束后对比：默认按 iou_1 选 best checkpoint
grep -hE "iou_1|F1_1|recall_1|precision_1|tiny_recall_1|small_recall_1|large_recall_1" \
  ChangeDINO-main/checkpoints/S1GFloods-hybrid-legacy-b6-*/record.txt \
  ChangeDINO-main/checkpoints/S1GFloods-hybrid-mv2-b6-*/record.txt \
  ChangeDINO-main/checkpoints/S1GFloods-hybrid-mv2-warm5-b6-*/record.txt \
  | sort -t',' -k3 -rn | head -5
