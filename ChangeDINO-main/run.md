# ── 实验 A：高精度模式（grid=32, batch=4）──
RUN_NAME=S1GFloods-grid32-b4 \
BATCH_SIZE=4 TOPO_GRID=32 TOPO_K=16 \
bash ChangeDINO-main/trainval_s1gfloods.sh

# ── 实验 B：均衡模式（grid=16, batch=10）──
RUN_NAME=S1GFloods-grid16-b10 \
BATCH_SIZE=10 TOPO_GRID=16 TOPO_K=12 \
bash ChangeDINO-main/trainval_s1gfloods.sh

RUN_NAME=S1GFloods-grid32-b4  BATCH_SIZE=4  TOPO_GRID=32 TOPO_K=16 bash ChangeDINO-main/trainval_s1gfloods.sh && \
RUN_NAME=S1GFloods-grid16-b10 BATCH_SIZE=10 TOPO_GRID=16 TOPO_K=12 bash ChangeDINO-main/trainval_s1gfloods.sh

训练结束后对比
grep -h "iou_1" \
  ChangeDINO-main/checkpoints/S1GFloods-grid32-b4-*/record.txt \
  ChangeDINO-main/checkpoints/S1GFloods-grid16-b10-*/record.txt \
  | sort -t',' -k3 -rn | head -5