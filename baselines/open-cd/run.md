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

<!-- 模型训练 -->
bash baselines/open-cd/scripts/s1gfloods/run_all_s1gfloods_extra.sh full-train \
  --data-root datasets/S1GFloods_CD_DINO \
  --batch-root baselines/open-cd/work_dirs \
  --gpus 1 \
  --save-best mIoU

<!-- 模型预测推理 -->
cd /home/yukun/codes/paper6_waterlogging/baselines/open-cd
bash scripts/s1gfloods/run_all_gf3_henan_infer.sh \
  --batch-dir work_dirs/s1gfloods-batch-20260326-004236 \
  --data-root ../../datasets/GF3_Henan_CD_infer \
  --device cuda:0 \
  --batch-size 1 \
  --threshold 0.5

cd /home/yukun/codes/paper6_waterlogging/baselines/open-cd
bash scripts/s1gfloods/run_all_gf3_henan_infer.sh \
  --batch-dir work_dirs/s1gfloods-batch-20260326-004236 \
  --data-root ../../datasets/GF3_Zhuozhou_CD_infer \
  --device cuda:0 \
  --batch-size 1 \
  --threshold 0.5

<!-- 新模型推理命令 -->
cd /home/yukun/codes/paper6_waterlogging/baselines/open-cd
bash scripts/s1gfloods/run_all_gf3_henan_infer.sh \
  --batch-dir work_dirs/s1gfloods-batch-20260331-165142 \
  --data-root ../../datasets/GF3_Henan_CD_infer \
  --device cuda:0 \
  --batch-size 1 \
  --threshold 0.5

# 单模型调试推理
python tools/infer_gf3_henan.py \
  work_dirs/s1gfloods-batch-20260326-004236/bit_r18_256x256_40k_s1gfloods/bit_r18_256x256_40k_s1gfloods.py \
  work_dirs/s1gfloods-batch-20260326-004236/bit_r18_256x256_40k_s1gfloods/best_mIoU_iter_40000.pth \
  --data-root ../../datasets/GF3_Henan_CD_infer \
  --work-dir work_dirs/s1gfloods-batch-20260326-004236/bit_r18_256x256_40k_s1gfloods \
  --device cuda:0 \
  --batch-size 4 \
  --threshold 0.5

# 输出位置
# - 批量脚本会根据 --data-root 自动生成数据集后缀，例如 GF3_Zhuozhou_CD_infer -> gf3_zhuozhou_cd_infer
# - 每个模型的切片预测: work_dirs/.../<model_tag>/infer_gf3_henan_png_<suffix>/
# - 每个模型的整景拼接: work_dirs/.../<model_tag>/infer_gf3_henan_full_<suffix>/
# - 批量日志目录: work_dirs/.../infer_logs_<suffix>/
# - 批量汇总表: work_dirs/.../infer_gf3_henan_summary_<suffix>.tsv
# - 整景报告: work_dirs/.../<model_tag>/infer_gf3_henan_full_<suffix>/infer_report.json

# 说明
# - GF3_Henan_CD_infer 来自 ChangeDINO-main/scripts/prepare_gf3_henan_infer.py，
#   当前切片参数为 tile_size=256、stride=128，存在 50% overlap。
# - Open-CD 推理脚本会读取 tile_manifest.csv 与 prepare_report.json，
#   先做切片预测，再使用 valid_mask + Hanning 权重窗拼接整景结果。
# - 批量脚本不会再复用旧的固定输出目录，适合在同一批 work_dir 下保存多个数据集的推理结果。
# - 最终推荐使用 infer_gf3_henan_full_<suffix>/ 下的 change_prob.tif、
#   change_binary.tif、change_binary.png 作为正式输出；
#   infer_gf3_henan_png_<suffix>/ 主要用于切片级排查。
# - 运行中出现的 registry / visualizer warning 当前不影响推理与拼接结果落盘。
