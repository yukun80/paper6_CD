conda activate opencd
cd /home/yukun80/codes/paper6_waterlogging
export CUDA_VISIBLE_DEVICES=0

# 环境、数据与权重预检
bash baselines/open-cd/scripts/s1gfloods/run_all_s1gfloods.sh check-env

# 短训练验证（本轮已通过，需要时可再次执行）
bash baselines/open-cd/scripts/s1gfloods/run_all_s1gfloods.sh smoke-train

# 正式训练全部10个模型，创建新批次
bash baselines/open-cd/scripts/s1gfloods/run_all_s1gfloods.sh full-train