# S1GFloods 训练入口

conda activate hacqi
cd /home/yukun80/codes/paper6_waterlogging

# 预检
CUDA_VISIBLE_DEVICES=0 \
bash baselines/ChangeDINO_raw/trainval_s1gfloods.sh check-env

# 正式训练
CUDA_VISIBLE_DEVICES=0 \
bash baselines/ChangeDINO_raw/trainval_s1gfloods.sh full-train