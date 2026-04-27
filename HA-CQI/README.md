# HA-CQI

HA-CQI is the current SAR flood change detection algorithm in this directory. The
implementation is organized around the paper narrative in
`model_design/paper_narrative.md` and targets binary flood-change segmentation on
S1GFloods-style pre/post SAR image pairs.

## Model Design

The active model name is **HA-CQI**.

Main modules:

- **Module I: Hierarchical CNN-DINO Semantic Encoder**  
  A shared pre/post encoder combines an EfficientNet-B0 or MobileNetV2 CNN-FPN
  pyramid with DINOv3 semantic features. The encoder outputs aligned-resolution
  pyramid features `P1-P5`.
- **Module II: Harmonized Alignment (HA)**  
  Pair-shared style calibration reduces SAR radiometric mismatch on shallow
  features, then optional deformable soft alignment aligns pre/post features on
  `P1/P2/P3`.
- **Module III: Change Query Interaction (CQI)**  
  Learnable change queries interact with multi-scale feature differences through
  two-way attention, giving the decoder explicit change-centric context.
- **Mask2Former-style Segmentation Head**  
  A lightweight query mask decoder predicts binary flood-change logits from
  high-resolution mask features and CQI-enhanced context.
- **Size-aware Auxiliary Heads**  
  Auxiliary predictions on `P1-P5` keep supervision visible to both small
  waterlogging objects and large inundation regions.

The active code path is:

```text
model/architectures/ha_cqi.py
model/engine.py
model/modules/harmonized_alignment.py
model/modules/change_query_interaction.py
model/decode_heads/mask2former_change_head.py
trainval.py
test.py
run.py
trainval_s1gfloods.sh
```

## Dataset

The training loader expects the standard binary change-detection layout:

```text
datasets/<DATASET_NAME>/
├── train/{A,B,label}
├── val/{A,B,label}
├── train_tif/{A,B,label}
├── val_tif/{A,B,label}
├── manifest_train.csv
├── manifest_val.csv
└── channel_stats_s1gfloods_train.json
```

The default training dataset is `datasets/S1GFloods_CD_DINO`. The optional
`datasets/S1GFloods_CD_DINO_` directory is treated as an experiment-specific
variant and must be selected explicitly with `DATASET_NAME=S1GFloods_CD_DINO_`.

If the fused S1GFloods dataset has not been prepared yet, run the repository
SAR dataset preprocessing utilities from the repository root, then compute the
train-split channel statistics:

```bash
python <prepare_fused_sar_cd_dataset.py> \
  --s1gfloods-root datasets/S1GFloods \
  --varfloods-root datasets/VarFloods \
  --out-root datasets/S1GFloods_CD_DINO \
  --tile-size 256 \
  --stride 128 \
  --train-ratio 0.9 \
  --seed 42 \
  --overwrite

python <compute_s1gfloods_cd_stats.py> \
  --data-root datasets/S1GFloods_CD_DINO \
  --split train \
  --output datasets/S1GFloods_CD_DINO/channel_stats_s1gfloods_train.json
```

## Weights

Default local weights:

```text
HA-CQI/dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth
HA-CQI/pretrained/efficientnet_b0_ra-3dd342df.pth
```

`efficientnet_b0` requires a local PyTorch `.pth/.pt` file. The training entry
does not implicitly download weights.

## Train

Run from the HA-CQI directory:

```bash
cd HA-CQI
bash trainval_s1gfloods.sh
```

Useful environment overrides:

```bash
cd HA-CQI
DATASET_NAME=S1GFloods_CD_DINO \
DATA_ROOT=../datasets \
RUN_NAME=S1GFloods-HA-CQI-vits16 \
BATCH_SIZE=6 \
BEST_METRIC=tiny_safe_combo \
EVAL_FG_THRESHOLD=0.40 \
bash trainval_s1gfloods.sh
```

The default script uses:

- `DATASET_NAME=S1GFloods_CD_DINO`
- `DATA_ROOT=../datasets`
- `STATS_FILE=../datasets/S1GFloods_CD_DINO/channel_stats_s1gfloods_train.json`
- `DINO_ARCH=dinov3_vits16`
- `DINO_WEIGHT=dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth`
- `BACKBONE=efficientnet_b0`
- `BACKBONE_WEIGHT=pretrained/efficientnet_b0_ra-3dd342df.pth`
- `NUM_CHANGE_QUERIES=16`
- `MASK_QUERIES=32`
- `MASK_DECODER_LAYERS=3`
- `BEST_METRIC=tiny_safe_combo`
- `EVAL_FG_THRESHOLD=0.40`

Training outputs are written under:

```text
HA-CQI/checkpoints/<resolved_run_name>/
HA-CQI/checkpoints/<resolved_run_name>/vis/
```

Disable HA soft alignment for ablation:

```bash
cd HA-CQI
SOFT_ALIGNMENT=0 RUN_NAME=S1GFloods-HA-CQI-noalign bash trainval_s1gfloods.sh
```

## Test

Use the resolved run directory name printed by training:

```bash
cd HA-CQI
python test.py \
  --name <resolved_run_name> \
  --dataset S1GFloods_CD_DINO \
  --dataroot ../datasets \
  --dataset_mode sar \
  --stats_file ../datasets/S1GFloods_CD_DINO/channel_stats_s1gfloods_train.json \
  --gpu_ids 0 \
  --save_test
```

The checkpoint metadata stores the HA-CQI architecture options needed to rebuild
the model at inference time.
Saved test predictions are written to
`HA-CQI/checkpoints/<resolved_run_name>/pred/` when `--save_test` is enabled.

## Pair Inference

`run.py` performs direct inference on one pre/post image pair:

```bash
cd HA-CQI
python run.py \
  --name <resolved_run_name> \
  --dataset S1GFloods_CD_DINO \
  --dataroot ../datasets \
  --dataset_mode sar \
  --stats_file ../datasets/S1GFloods_CD_DINO/channel_stats_s1gfloods_train.json \
  --img_A /path/to/pre_image.tif \
  --img_B /path/to/post_image.tif \
  --output outputs/run_pred.png \
  --gpu_ids 0
```

Whole-scene tiled inference should call the same HA-CQI model entrypoint when
stitching tiles back to a scene-level mask.

## Validation Commands

After code changes, run:

```bash
python -m py_compile \
  option.py \
  model/architectures/ha_cqi.py \
  model/backbones/builder.py \
  model/modules/*.py \
  model/decode_heads/*.py \
  model/engine.py \
  trainval.py \
  test.py \
  run.py

bash -n trainval_s1gfloods.sh trainval.sh
```
