# ChangeDINO [[paper]](https://arxiv.org/abs/2511.16322)
*[Ching-Heng Cheng](https://scholar.google.com/citations?user=2UmoEfcAAAAJ&hl=zh-TW), [Chih-Chung Hsu](https://cchsu.info/wordpress/)*

*Advanced Computer Vision LAB, National Cheng Kung University and National Yang Ming Chiao Tung University.*

This is a PyTorch implementation for "[ChangeDINO: DINOv3-Driven Building Change Detection in Optical Remote Sensing Imagery](https://arxiv.org/abs/2511.16322)." This document summarizes the environment requirements, dataset layout, and common commands you need to run the project.

Arch.
<center><img src="./demo/ChangeDINO.png" width=1080 alt="ChangeDINO"></center>

# Quick Guide

## Setup
Recommended: Python 3.10, PyTorch 2.4.0 (CUDA 11.8)
```bash
cd ChangeDINO
conda create -n changedino python=3.10
conda activate changedino
# recommendation (torch <= 2.4.0 and cuda <= 12.1 for mmcv installation)
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
# mmcv need to fit torch and cuda version
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.4/index.html
pip install -r requirements.txt
```

## Dataset
Place all datasets under `--dataroot` (ex. `/path/to/CD-Dataset`). 

Expect the data structure:
```
/path/to/CD-Dataset/
└── WHU-CD/                      # matches --dataset
    ├── train/
    │   ├── A/                   # T1 images (pre-change)
    │   ├── B/                   # T2 images (post-change)
    │   └── label/               # binary masks (0 or 255 / 0 or 1)
    ├── val/
    │   ├── A/
    │   ├── B/
    │   └── label/
    └── test/
        ├── A/
        ├── B/
        └── label/
```

### S1GFloods
For this repository, raw `datasets/S1GFloods` is a flat layout:
```text
datasets/S1GFloods/
├── A/
├── B/
└── Label/
```

Build the fused ChangeDINO layout from raw `S1GFloods` plus `VarFloods/*/PRO` first:
```bash
python ChangeDINO-main/scripts/prepare_fused_sar_cd_dataset.py \
  --s1gfloods-root datasets/S1GFloods \
  --varfloods-root datasets/VarFloods \
  --out-root datasets/S1GFloods_CD_DINO \
  --tile-size 256 \
  --stride 128 \
  --train-ratio 0.9 \
  --seed 42 \
  --overwrite
```

This rebuilds `datasets/S1GFloods_CD_DINO` into a train/val-only dataset:
```text
datasets/S1GFloods_CD_DINO/
├── train/{A,B,label}
├── val/{A,B,label}
├── train_tif/{A,B,label}
├── val_tif/{A,B,label}
├── manifest_all.csv
├── manifest_train.csv
├── manifest_val.csv
└── split_report.json
```

Then compute train-split normalization stats:
```bash
python ChangeDINO-main/scripts/compute_s1gfloods_cd_stats.py \
  --data-root datasets/S1GFloods_CD_DINO \
  --split train \
  --output datasets/S1GFloods_CD_DINO/channel_stats_s1gfloods_train.json
```

## Pre-trained Weights (Google Drive)
For the DINOv3 pre-trained weight, please [download here](https://drive.google.com/file/d/1r6g0D6zV-1e8gJHij1edsE_uzvZ72L3u/view?usp=drive_link) and place it under `dinov3/weights/`.

For the default CNN backbone, place the local ConvNeXtV2 nano weight at:
```text
ChangeDINO-main/pretrained/convnextv2_nano_22k_224_ema.pt
```
The current default training/inference configuration uses this file via `--backbone_weight pretrained/convnextv2_nano_22k_224_ema.pt`.

For the full ChangeDINO's pre-trained weights, which can be obtained from the following links:

+ [LEVIR-CD](https://drive.google.com/file/d/1slYOZBmChzP7N7776ODGL4PB807xPr9d/view?usp=sharing)
+ [WHU-CD](https://drive.google.com/file/d/1vVaALwCoYrnDyoCXhH989sKkSRD_rRT2/view?usp=sharing)
+ [SYSU-CD](https://drive.google.com/file/d/12rD8gHNvkIfE8Wr6LdYoaWDNGT7zfSG7/view?usp=sharing)
+ [S2Looking](https://drive.google.com/file/d/1HzNkAdS8zPks5KLAr0yLYRrzqWlbC45w/view?usp=sharing)

## Train / Validate
```bash
cd dinov3/ChangeDINO
python trainval.py \
  --name WHU-ChangeDINO \
  --dataset WHU-CD \
  --dataroot /path/to/CD-Dataset \
  --gpu_ids 0 \
  --batch_size 16 \
  --num_epochs 100 \
  --lr 5e-4
```
Important flags live in `option.py` (datasets, GPUs, checkpoints, backbone/FPN choices, learning rate, etc.). Training runs are saved under `checkpoints/<name>-YYYYMMDD` and, if needed, `checkpoints/<name>-YYYYMMDD-<index>`; the best checkpoint is `<resolved_name>_<backbone>_best.pth`.

### Train / Validate on S1GFloods
```bash
cd ChangeDINO-main
BACKBONE_WEIGHT=pretrained/convnextv2_nano_22k_224_ema.pt \
bash trainval_s1gfloods.sh
```

Switch to a different local DINOv3 checkpoint by overriding env vars:
```bash
cd ChangeDINO-main
DINO_ARCH=dinov3_vits16 \
DINO_WEIGHT=dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
BACKBONE_WEIGHT=pretrained/convnextv2_nano_22k_224_ema.pt \
RUN_NAME=S1GFloods-ChangeDINO-vits16 \
bash trainval_s1gfloods.sh
```

Equivalent explicit command:
```bash
python trainval.py \
  --name S1GFloods-ChangeDINO-vitl16 \
  --dataset S1GFloods_CD_DINO \
  --dataroot ../datasets \
  --dataset_mode sar \
  --stats_file ../datasets/S1GFloods_CD_DINO/channel_stats_s1gfloods_train.json \
  --backbone convnextv2_nano \
  --backbone_weight pretrained/convnextv2_nano_22k_224_ema.pt \
  --dino_arch dinov3_vitl16 \
  --dino_weight dinov3/weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth \
  --gpu_ids 0 \
  --batch_size 8 \
  --num_epochs 100 \
  --lr 1e-4
```

Notes for S1GFloods:
- SAR branch now replaces plain `|F_pre - F_post|` with directional difference features plus deformable cross-attention alignment on high-resolution pyramid levels (`p2/p3`).
- New SAR alignment/difference hyper-parameters are exposed in `option.py`, including `--align_window`, `--align_points`, `--align_heads`, `--align_on_levels`, `--align_offset_groups`, and `--directional_diff_expand`.
- Input PNGs can be true grayscale or RGB-converted grayscale; loader normalizes both to 3-channel RGB tensors.
- The fused builder writes training PNGs under `train/` and `val/`, and preserves VarFloods tiles as GeoTIFF sidecars under `train_tif/` and `val_tif/`.
- SAR mode disables saturation jitter and uses milder brightness/contrast perturbation.
- You still need the DINOv3 checkpoint under `ChangeDINO-main/dinov3/weights/`.
- Current default CNN backbone is `convnextv2_nano`, and the default local weight path is `pretrained/convnextv2_nano_22k_224_ema.pt`.
- Only `convnextv2_nano` and `mobilenetv2` are supported now; `resnet18d` has been removed.
- `--dino_arch` now supports `dinov3_vits16`, `dinov3_vitb16`, and `dinov3_vitl16`; if omitted, it is inferred from `--dino_weight`.
- `--extract_ids` defaults follow the chosen DINO architecture automatically, so `vits16` no longer reuses the old ViT-L layer ids.
- If you rename the prepared dataset directory, keep `--dataset` and `--stats_file` consistent with that exact folder name.
- The fused dataset no longer creates a `test/` split; `trainval.py` still works unchanged because it only consumes `train` and `val`.
- Training run directories are now auto-resolved to `checkpoints/<name>-YYYYMMDD` and, if needed, `checkpoints/<name>-YYYYMMDD-<index>` to avoid overwriting old experiments.
- The final checkpoint filename also uses that resolved run name, so test/inference commands must point to the actual generated directory name instead of the unsuffixed base `RUN_NAME`.

### Auto Handoff to Open-CD Batch Training
If you want `baselines/open-cd` to start automatically after the current `trainval_s1gfloods.sh` process exits, launch the monitor script from the repository root in the same conda environment:

```bash
cd /path/to/paper6_waterlogging
bash scripts/monitor/run_opencd_after_pid.sh \
  --pid <trainval_s1gfloods_pid> \
  --poll-seconds 30 \
  --opencd-mode full-train \
  --opencd-gpus 1
```

Useful notes:
- Find the current training PID with `pgrep -af trainval_s1gfloods.sh` or `ps -ef | grep trainval_s1gfloods.sh | grep -v grep`.
- The monitor script only watches the specified PID and does not modify the running ChangeDINO job.
- Once that PID exits, it triggers `baselines/open-cd/scripts/s1gfloods/run_all_s1gfloods.sh`.
- Start the monitor from the same environment used for Open-CD training, otherwise the follow-up job may fail because `mmengine/mmseg/opencd` are unavailable.
- Logs are written under `logs/handoffs/` by default.

### GF3 Henan Whole-Scene Inference
Use the S1GFloods-trained checkpoint to run tiled inference on the GF3 Henan pre/post pair.

1. Prepare PNG/TIF tiles with explicit valid masks:
```bash
python ChangeDINO-main/scripts/prepare_gf3_henan_infer.py \
  --src-root datasets/GF3_Henan \
  --pre-image Pre_Zhengzhou_descending_clip.tif \
  --post-image Post_Zhengzhou_descending_clip.tif \
  --out-root datasets/GF3_Henan_CD_infer \
  --tile-size 256 \
  --stride 128 \
  --overwrite
```

2. Run tiled inference and stitch back to whole-scene outputs:
```bash
python ChangeDINO-main/scripts/infer_gf3_henan_tiles.py \
  --tiles-root datasets/GF3_Henan_CD_infer \
  --checkpoint ChangeDINO-main/checkpoints/<resolved_run_name>/<resolved_run_name>_convnextv2_nano_best.pth \
  --stats_file <path_to_s1gfloods_stats.json> \
  --gpu_ids 0 \
  --batch_size 8 \
  --output-dir ChangeDINO-main/outputs/gf3_henan
```

Notes for GF3 Henan:
- Tiles are saved as both `PNG` and `tif`: `PNG` is the actual model input to stay closer to S1GFloods training data, while `tif` preserves original float32 values and georeferencing.
- `valid_mask` is generated for every tile and is used during stitching so `nodata` pixels do not contribute to predictions.
- Final outputs include `change_prob.tif`, `change_binary.tif`, and `change_binary.png`.
- New checkpoints written by `trainval.py` carry `model_config` metadata, so the tiled inference script can auto-restore the trained DINO architecture and layer ids.
- For old checkpoints without metadata, pass matching `--dino_arch` and `--dino_weight` explicitly during inference.

### S1 Henan Whole-Scene Inference
Use the same S1GFloods-trained checkpoint to run tiled inference on the Sentinel-1 Henan pre/post VH pair.

1. Prepare PNG/TIF tiles with explicit valid masks:
```bash
python ChangeDINO-main/scripts/prepare_s1_henan_infer.py \
  --src-root datasets/S1_Henan \
  --pre-image Zhengzhou_S1GRD_ASCENDING_VH_pre.tif \
  --post-image Zhengzhou_S1GRD_ASCENDING_VH_Post.tif \
  --out-root datasets/S1_Henan_CD_infer \
  --tile-size 256 \
  --stride 128 \
  --overwrite
```

2. Run tiled inference and stitch back to whole-scene outputs:
```bash
python ChangeDINO-main/scripts/infer_s1_henan_tiles.py \
  --tiles-root datasets/S1_Henan_CD_infer \
  --checkpoint ChangeDINO-main/checkpoints/<resolved_run_name>/<resolved_run_name>_convnextv2_nano_best.pth \
  --stats_file datasets/S1GFloods_CD_DINO/channel_stats_s1gfloods_train.json \
  --gpu_ids 0 \
  --batch_size 8 \
  --output-dir ChangeDINO-main/outputs/s1_henan
```

Notes for S1 Henan:
- Current defaults expect single-band Sentinel-1 VH tif inputs.
- Pre/Post tif must share the same shape, CRS, transform, and nodata definition.
- Final outputs include `change_prob.tif`, `change_binary.tif`, and `change_binary.png`.

## Test
```bash
python test.py \
  --name WHU-ChangeDINO \
  --dataset WHU-CD \
  --dataroot /path/to/CD-Dataset \
  --gpu_ids 0 \
  --save_test
```
This loads the best checkpoint, runs on the `test` split, prints metrics, and saves predictions (if `--save_test`) under `checkpoints/<name>/pred/`. When the model was trained by `trainval.py`, use the actual resolved training run name such as `<base_name>-YYYYMMDD` or `<base_name>-YYYYMMDD-<index>`.

S1GFloods example:
```bash
python test.py \
  --name <resolved_run_name> \
  --dataset S1GFloods_CD_DINO \
  --dataroot ../datasets \
  --dataset_mode sar \
  --stats_file ../datasets/S1GFloods_CD_DINO/channel_stats_s1gfloods_train.json \
  --gpu_ids 0 \
  --save_test
```

Adjust `--gpu_ids`, `--num_workers`, and other options as needed, and use `trainval.sh` for ready-made command examples.

## Comparison
<center><img src="./demo/levir_whu_table.png" width=480 alt="levir_whu_table"></center>
<center><img src="./demo/levir_whu_plot.png" width=720 alt="levir_whu_plot"></center>
<center><img src="./demo/adapt_dino_feats.png" width=480 alt="adapt_dino_feats"></center>

## Citation 

 If you use this code for your research, please cite our papers.  

```
@misc{cheng2025changedinodinov3drivenbuildingchange,
      title={ChangeDINO: DINOv3-Driven Building Change Detection in Optical Remote Sensing Imagery}, 
      author={Ching-Heng Cheng and Chih-Chung Hsu},
      year={2025},
      eprint={2511.16322},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.16322}, 
}
```
