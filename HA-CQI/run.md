# HA-CQI B2 + OSCD 强召回动态数据运行说明

## 1. 当前 baseline

主线固定为：

- 当前目录数据集 `S1GFloods_CD_DINO_BG_75_25_`；
- EfficientNet-B2 五级 CNN-FPN；
- frozen DINOv3 ViT-S/16 LVD，ImageNet normalization，融合层 `[5,8,11]`；
- HA + 五级 CQI + OSCD；
- P1–P5 auxiliary；
- Focal `0.25/0.75`、foreground Tversky `0.70→0.55`、support/coarse `0.03/0.02`；
- batch 12、8 workers、bf16、AdamW base/head LR `1e-4/2e-4`、80 epoch cosine。

此前用于 0825 的 P1/P2 query-free、浅层监督减法、dual-class overlap 与 boundary loss 已删除。
时相独立辐射增强仅保留为显式消融，默认仍为 shared。

## 2. 正式训练

```bash
cd HA-CQI
conda activate hacqi

DATASET_NAME=S1GFloods_CD_DINO_BG_75_25_ \
DATA_ROOT=../datasets \
STATS_MODE=auto \
RUN_NAME=S1GFloods-HA-CQI-B2-OSCD-DINO5-8-11-CLEAN-s1 \
DINO_ARCH=dinov3_vits16 \
DINO_WEIGHT=dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
DINO_FUSION_LAYERS="5 8 11" \
BACKBONE_WEIGHT=pretrained/efficientnet_b2_ra-bcdf34b7.pth \
BATCH_SIZE=12 \
NUM_WORKERS=8 \
LR=1e-4 \
HEAD_LR_MULT=2.0 \
AMP=1 \
AMP_DTYPE=bf16 \
SEED=1 \
bash trainval_s1gfloods.sh
```

`STATS_MODE=auto` 会扫描当前 train/val 的 A/B/label 同名集合。完整删除三元组后，下次进程启动
自动减样本；只缺任一文件会失败。train A/B 的文件名、大小或 mtime 变化时，mean/std 缓存自动
失效。历史 manifest、split report 与 fingerprint 不决定运行成员。

若需要复现指定统计文件：

```bash
STATS_MODE=file \
STATS_FILE=../datasets/S1GFloods_CD_DINO_BG_75_25_/channel_stats_s1gfloods_train.json \
bash trainval_s1gfloods.sh
```

文件模式只验证 train split、三个有限 mean/std 与正数 std。

## 3. 短步冒烟

```bash
cd HA-CQI
conda activate hacqi
python trainval.py \
  --name smoke-hacqi-b2-dino5 \
  --dataset S1GFloods_CD_DINO_BG_75_25_ \
  --dataroot ../datasets \
  --stats_mode auto \
  --dino_fusion_layers 5 8 11 \
  --gpu_ids 0 \
  --batch_size 1 \
  --num_workers 0 \
  --num_epochs 1 \
  --max_train_steps 2 \
  --max_val_steps 2 \
  --seed 1 \
  --amp \
  --amp_dtype bf16
```

正式训练不得保留 `--max_train_steps/--max_val_steps`。

## 4. Checkpoint 与训练记录

每个 run 记录：

```text
*_efficientnet_b2_best_primary.pth
*_efficientnet_b2_last.pth
*_efficientnet_b2_epoch10.pth ...
selection.json
metrics.jsonl
options.json
data_snapshot.json
```

每次训练从 CNN/DINO 预训练权重开始，epoch 从 1、global step 从 0 开始。
不支持续训或整模型初始化；`--resume` 和 `--init_checkpoint` 已删除，传入将报未知参数错误。
脚本不再读取 `RESUME` 环境变量，旧命令设置该变量也会启动全新训练。
新 checkpoint 仅保存 `network/meta`，保留 v2 模型配置、阈值选择、epoch/global_step 和审计信息；
不保存任何训练恢复状态。best_primary、last、每 10 epoch 的 periodic 名称与保存时机不变。
新旧 v2 checkpoint 均可用于测试与推理，旧格式仍不支持。
早期 B2-OSCD v2 的 `[2,5,8,11] + raw[1:]` 仅按可证明等价关系解释为 `[5,8,11]`；
显式 `[2,8,11]` 的 0825 checkpoint 仍按原层路由复现。

## 5. 测试与推理

```bash
cd HA-CQI
python test.py \
  --checkpoint checkpoints/<run>/<run>_efficientnet_b2_best_primary.pth \
  --gpu_ids 0 \
  --save_test
```

```bash
cd HA-CQI
python run.py \
  --checkpoint checkpoints/<run>/<run>_efficientnet_b2_best_primary.pth \
  --img_A /path/to/pre.tif \
  --img_B /path/to/post.tif \
  --output outputs/pair.png \
  --gpu_ids 0
```

推理阈值优先级为显式 `--threshold`，其次 checkpoint v2 `meta.selection.threshold`；两者均缺失
时失败。推理归一化直接使用 checkpoint 的实际 mean/std，不重新扫描训练目录。

## 6. 当前停止边界

- 不重建数据集，不用 manifest/fingerprint 阻止训练；
- 不改 HA、CQI、OSCD、输入尺寸或 SAR 色调映射；
- 不启用新的背景抑制 loss；
- 只有清洗后 baseline 仍仅在跨域场景出现 FP，才单独测试
  `RADIOMETRIC_JITTER_MODE=independent`；
- FP、tiny recall 和跨域泛化收益需要完整训练验证，轻量工程测试不代表算法指标提升。
