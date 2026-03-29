CUDA_LAUNCH_BLOCKING=1 ./baselines/open-cd/scripts/urban_sar_floods/run_fcsiam_conc_3c.sh full-train --best-fscore

# Open-CD on S1GFloods_CD_DINO (PNG train/val layout)
cd baselines/open-cd
python tools/train.py configs/changer/changer_ex_r18_256x256_40k_s1gfloods.py
python tools/train.py configs/bit/bit_r18_256x256_40k_s1gfloods.py
python tools/train.py configs/ifn/ifn_256x256_40k_s1gfloods.py
python tools/train.py configs/fcsn/fc_siam_diff_256x256_40k_s1gfloods.py
python tools/train.py configs/changestar/changestar_farseg_1x96_256x256_40k_s1gfloods.py
python tools/train.py configs/lightcdnet/lightcdnet_s_256x256_40k_s1gfloods.py

# Serial batch train for all S1GFloods configs
bash scripts/s1gfloods/run_all_s1gfloods.sh check-env
bash scripts/s1gfloods/run_all_s1gfloods.sh smoke-train --gpus 1
bash scripts/s1gfloods/run_all_s1gfloods.sh full-train --gpus 1

# Start Open-CD batch training automatically after a running ChangeDINO job exits
cd /home/yukun/codes/paper6_waterlogging
bash scripts/monitor/run_opencd_after_pid.sh \
  --pid <trainval_s1gfloods_pid> \
  --poll-seconds 30 \
  --opencd-mode full-train \
  --opencd-gpus 1

cd baselines/CMCDNet
python tools/train.py my_scripts/urban_sar_floods/cmcd_urban_sar_floods_3c_r50_effb2_30e.py

PYTHONPATH=panopticon python panopticon/urban_floods/train.py --config-file panopticon/configs/urban_floods_seg.yaml


<!-- 模型预测推理 -->
cd /home/yukun/codes/paper6_waterlogging/baselines/open-cd
bash scripts/s1gfloods/run_all_gf3_henan_infer.sh \
  --batch-dir work_dirs/s1gfloods-batch-20260326-004236 \
  --device cuda:0 \
  --batch-size 1 \
  --threshold 0.5

# 输出位置
# - 每个模型的切片预测: work_dirs/.../<model_tag>/infer_gf3_henan_png/
# - 每个模型的整景拼接: work_dirs/.../<model_tag>/infer_gf3_henan_full/
# - 批量汇总表: work_dirs/.../infer_gf3_henan_summary.tsv
