# ── 默认训练（ContrastAwareDiff + FloodTopoRouter 双向精修版，batch=8）──
RUN_NAME=S1GFloods-topo-b8 \
BATCH_SIZE=8 \
bash ChangeDINO-main/trainval_s1gfloods.sh

# ── 大 batch 训练 ──
RUN_NAME=S1GFloods-topo-b12 \
BATCH_SIZE=12 \
bash ChangeDINO-main/trainval_s1gfloods.sh

# ── 自定义拓扑参数 ──
RUN_NAME=S1GFloods-topo-g8k8 \
BATCH_SIZE=8 \
TOPO_GRID=8 TOPO_K=8 \
bash ChangeDINO-main/trainval_s1gfloods.sh

# ── 自定义对比度池化核大小 ──
RUN_NAME=S1GFloods-topo-pool7 \
BATCH_SIZE=8 \
bash ChangeDINO-main/trainval_s1gfloods.sh \
  --contrast_pool_size 7

# ── 串行对比实验 ──
RUN_NAME=S1GFloods-topo-b8  BATCH_SIZE=8  bash ChangeDINO-main/trainval_s1gfloods.sh && \
RUN_NAME=S1GFloods-topo-b16 BATCH_SIZE=16 bash ChangeDINO-main/trainval_s1gfloods.sh

# 训练结束后对比
grep -h "iou_1" \
  ChangeDINO-main/checkpoints/S1GFloods-topo-b8-*/record.txt \
  ChangeDINO-main/checkpoints/S1GFloods-topo-b16-*/record.txt \
  | sort -t',' -k3 -rn | head -5
