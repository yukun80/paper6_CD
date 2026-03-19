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
  --out-root datasets/S1GFloods_CD \
  --seed 42 \
  --overwrite
```

Then compute train-split normalization stats:
```bash
python ChangeDINO-main/scripts/compute_s1gfloods_cd_stats.py \
  --data-root datasets/S1GFloods_CD \
  --split train \
  --output datasets/S1GFloods_CD/channel_stats_s1gfloods_train.json
```

### VarFloods
For this repository, raw `datasets/VarFloods` is organized by region:
```text
datasets/VarFloods/
├── Bolivia/
│   └── PRO/
│       ├── A/*.tif
│       ├── B/*.tif
│       └── label/*.tif
├── France/
├── Honduras/
├── Libya/
└── VietNam/
```

Prepare ChangeDINO-ready tiles from `PRO` only:
```bash
python ChangeDINO-main/scripts/prepare_varfloods_cd.py \
  --src-root datasets/VarFloods \
  --out-root datasets/VarFloods_CD \
  --tile-size 256 \
  --stride 256 \
  --seed 42 \
  --overwrite \
  --strict
```

Then compute train-split normalization stats:
```bash
python ChangeDINO-main/scripts/compute_varfloods_cd_stats.py \
  --data-root datasets/VarFloods_CD \
  --split train \
  --output datasets/VarFloods_CD/channel_stats_varfloods_train.json
```

Notes for VarFloods:
- The prepared dataset keeps image tiles as single-band `float32` GeoTIFFs.
- ChangeDINO now reads SAR `tif` tiles directly via `rasterio`, applies a stable percentile stretch, and expands them to 3-channel tensors internally.
- Labels are converted to binary `0/1` tif tiles during preparation.

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
  --dataset S1GFloods_CD \
  --dataroot ../datasets \
  --dataset_mode sar \
  --stats_file ../datasets/S1GFloods_CD/channel_stats_s1gfloods_train.json \
  --gpu_ids 0 \
  --batch_size 16 \
  --num_epochs 100 \
  --lr 1e-4
```

Notes for S1GFloods:
- Input PNGs can be true grayscale or RGB-converted grayscale; loader normalizes both to 3-channel RGB tensors.
- SAR mode disables saturation jitter and uses milder brightness/contrast perturbation.
- You still need the DINOv3 checkpoint under `ChangeDINO-main/dinov3/weights/`.
- If you rename the prepared dataset directory (for example `S1GFloods_CD_DINO`), keep `--dataset` and `--stats_file` consistent with that exact folder name.

### Train / Validate on VarFloods
```bash
cd ChangeDINO-main
bash trainval_varfloods.sh
```

Equivalent explicit command:
```bash
python trainval.py \
  --name VarFloods-ChangeDINO \
  --dataset VarFloods_CD \
  --dataroot ../datasets \
  --dataset_mode sar \
  --stats_file ../datasets/VarFloods_CD/channel_stats_varfloods_train.json \
  --gpu_ids 0 \
  --batch_size 8 \
  --input_size 256 \
  --num_epochs 100 \
  --lr 1e-4
```

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
  --dataset S1GFloods_CD \
  --dataroot ../datasets \
  --dataset_mode sar \
  --stats_file ../datasets/S1GFloods_CD/channel_stats_s1gfloods_train.json \
  --gpu_ids 0 \
  --save_test
```

VarFloods example:
```bash
python test.py \
  --name VarFloods-ChangeDINO \
  --dataset VarFloods_CD \
  --dataroot ../datasets \
  --dataset_mode sar \
  --stats_file ../datasets/VarFloods_CD/channel_stats_varfloods_train.json \
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
