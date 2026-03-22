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

Convert it to ChangeDINO layout first:
```bash
python ChangeDINO-main/scripts/prepare_s1gfloods_cd.py \
  --src-root datasets/S1GFloods \
  --out-root datasets/S1GFloods_CD_DINO \
  --seed 42 \
  --overwrite
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
Important flags live in `option.py` (datasets, GPUs, checkpoints, backbone/FPN choices, learning rate, etc.). Results are saved to `checkpoints/<name>`; the best checkpoint is `<name>_<backbone>_best.pth`.

### Train / Validate on S1GFloods
```bash
cd ChangeDINO-main
bash trainval_s1gfloods.sh
```

Equivalent explicit command:
```bash
python trainval.py \
  --name S1GFloods-ChangeDINO \
  --dataset S1GFloods_CD_DINO \
  --dataroot ../datasets \
  --dataset_mode sar \
  --stats_file ../datasets/S1GFloods_CD_DINO/channel_stats_s1gfloods_train.json \
  --gpu_ids 0 \
  --batch_size 8 \
  --num_epochs 100 \
  --lr 1e-4
```

Notes for S1GFloods:
- Input PNGs can be true grayscale or RGB-converted grayscale; loader normalizes both to 3-channel RGB tensors.
- SAR mode disables saturation jitter and uses milder brightness/contrast perturbation.
- You still need the DINOv3 checkpoint under `ChangeDINO-main/dinov3/weights/`.
- If you rename the prepared dataset directory, keep `--dataset` and `--stats_file` consistent with that exact folder name.

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
  --checkpoint ChangeDINO-main/checkpoints/S1GFloods-ChangeDINO/S1GFloods-ChangeDINO_mobilenetv2_best.pth \
  --stats_file <path_to_s1gfloods_stats.json> \
  --gpu_ids 0 \
  --batch_size 8 \
  --output-dir ChangeDINO-main/outputs/gf3_henan
```

Notes for GF3 Henan:
- Tiles are saved as both `PNG` and `tif`: `PNG` is the actual model input to stay closer to S1GFloods training data, while `tif` preserves original float32 values and georeferencing.
- `valid_mask` is generated for every tile and is used during stitching so `nodata` pixels do not contribute to predictions.
- Final outputs include `change_prob.tif`, `change_binary.tif`, and `change_binary.png`.

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
  --checkpoint ChangeDINO-main/checkpoints/S1GFloods-ChangeDINO/S1GFloods-ChangeDINO_mobilenetv2_best.pth \
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
This loads the best checkpoint, runs on the `test` split, prints metrics, and saves predictions (if `--save_test`) under `checkpoints/<name>/pred/`.

S1GFloods example:
```bash
python test.py \
  --name S1GFloods-ChangeDINO \
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
