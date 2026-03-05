CUDA_LAUNCH_BLOCKING=1 ./baselines/open-cd/scripts/urban_sar_floods/run_fcsiam_conc_3c.sh full-train --best-fscore

cd baselines/CMCDNet
python tools/train.py my_scripts/urban_sar_floods/cmcd_urban_sar_floods_3c_r50_effb2_30e.py