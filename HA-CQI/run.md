# HA-CQI A 路线运行说明

## 1. 默认 corrected-data baseline

```bash
cd HA-CQI
bash trainval_s1gfloods.sh
```

默认保持 HA、CQI、decoder、ViT-S、EfficientNet-B0、256 输入和 batch 6 不变，使用：

- `S1GFloods_CD_DINO_BG_75_25`；
- 自然频率采样；
- focal 背景/前景权重 `0.25/0.75`；
- Tversky beta `0.70→0.55`；
- support/coarse `0.03/0.02`；
- aux `1.0→0.5`；
- DINO normalization `shared`。

validation 在 `0.05–0.95/0.01` 上联合选阈值，primary 固定为 Flood IoU。IoU 并列时依次
选择更高 precision 和更高 threshold。tiny/small/component 指标只报告，不保存独立 best。

## 2. 可复现与短步诊断

```bash
cd HA-CQI
python trainval.py \
  --name smoke-hacqi \
  --dataset S1GFloods_CD_DINO_BG_75_25 \
  --dataroot ../datasets \
  --stats_file ../datasets/S1GFloods_CD_DINO_BG_75_25/channel_stats_s1gfloods_train.json \
  --gpu_ids 0 \
  --batch_size 6 \
  --num_epochs 1 \
  --max_train_steps 2 \
  --max_val_steps 2 \
  --seed 1 \
  --amp
```

`--max_train_steps/--max_val_steps` 只用于冒烟。完整训练不得保留这两个限制。

## 3. Checkpoint v2

新训练目录只产生：

```text
*_best_primary.pth
*_last.pth
*_epoch10.pth ...
selection.json
metrics.jsonl
options.json
```

完整续训：

```bash
cd HA-CQI
RESUME=checkpoints/<run>/<run>_efficientnet_b0_last.pth bash trainval_s1gfloods.sh
```

仅加载网络参数：

```bash
cd HA-CQI
python trainval.py <其余训练参数> --init_checkpoint /path/to/checkpoint.pth
```

resume 恢复 optimizer、scheduler、GradScaler、epoch、global step、随机状态和 DataLoader
generator，并对模型、loss、训练配置及 dataset fingerprint 做 fail-closed 校验。所有加载入口
均只接受 checkpoint v2；20260427 权重仅作为磁盘归档，当前代码不再支持。

## 4. 第一轮单变量消融顺序

只有当前变量的胜者进入下一轮，其他配置保持不变：

1. `DINO_INPUT_NORM=imagenet`；
2. `SUPPORT_CONSISTENCY_WEIGHT=0`；
3. `FOCAL_BG_WEIGHT=0.5 FOCAL_FG_WEIGHT=0.5`；
4. `TVERSKY_BETA_START=0.5 TVERSKY_BETA_END=0.5`；
5. `AUX_LOSS_WEIGHT_END=0.25`。

完整 corrected-data baseline（在本轮验收后由用户手动启动）：

```bash
cd HA-CQI
RUN_NAME=S1GFloods-HA-CQI-corrected-baseline-s1 \
SEED=42 \
BATCH_SIZE=10 \
DINO_INPUT_NORM=shared \
FOCAL_BG_WEIGHT=0.25 \
FOCAL_FG_WEIGHT=0.75 \
SUPPORT_CONSISTENCY_WEIGHT=0.03 \
TVERSKY_BETA_START=0.70 \
TVERSKY_BETA_END=0.55 \
AUX_LOSS_WEIGHT_END=0.50 \
AMP=1 \
AMP_DTYPE=fp16 \
bash trainval_s1gfloods.sh
```

五个单变量候选命令如下。每轮运行前，应把未测试项改回当前晋级配置；不要机械地把五条命令
一次性并行运行。

```bash
# 1. 仅 DINO normalization：与 baseline shared 对照
DINO_INPUT_NORM=imagenet RUN_NAME=S1GFloods-HA-CQI-ab01-dino-imagenet bash trainval_s1gfloods.sh

# 2. 仅 support consistency：DINO_INPUT_NORM 必须填写上一轮胜者
DINO_INPUT_NORM=shared SUPPORT_CONSISTENCY_WEIGHT=0 \
RUN_NAME=S1GFloods-HA-CQI-ab02-support0 bash trainval_s1gfloods.sh

# 3. 仅 focal class weights
DINO_INPUT_NORM=shared FOCAL_BG_WEIGHT=0.5 FOCAL_FG_WEIGHT=0.5 \
RUN_NAME=S1GFloods-HA-CQI-ab03-focal-balanced bash trainval_s1gfloods.sh

# 4. 仅 Tversky
DINO_INPUT_NORM=shared TVERSKY_BETA_START=0.5 TVERSKY_BETA_END=0.5 \
RUN_NAME=S1GFloods-HA-CQI-ab04-tversky-balanced bash trainval_s1gfloods.sh

# 5. 仅 aux end weight
DINO_INPUT_NORM=shared AUX_LOSS_WEIGHT_END=0.25 \
RUN_NAME=S1GFloods-HA-CQI-ab05-aux025 bash trainval_s1gfloods.sh
```

若某一前序候选晋级，后续命令必须显式携带该胜者的变量。例如 DINO `imagenet` 晋级后，
后四条均将 `DINO_INPUT_NORM=shared` 改为 `imagenet`；其他 loss 胜者同理。

## 5. 显式测试与推理

```bash
cd HA-CQI
python test.py \
  --checkpoint checkpoints/<run>/<run>_efficientnet_b0_best_primary.pth \
  --gpu_ids 0 \
  --save_test
```

```bash
cd HA-CQI
python run.py \
  --checkpoint checkpoints/<run>/<run>_efficientnet_b0_best_primary.pth \
  --img_A /path/to/pre.tif \
  --img_B /path/to/post.tif \
  --output outputs/pair.png \
  --gpu_ids 0
```

阈值优先级统一为：显式 `--threshold` → checkpoint v2 `meta.selection.threshold`。若两者
都不存在则直接失败，不再使用固定阈值。checkpoint v2 的 stats 路径会自动解析；显式 CLI
始终优先。训练参数 `--eval_fg_threshold` 仅用于 validation 细粒度诊断。

## 6. 整景评估

```bash
python HA-CQI/scripts/evaluate_sar_scene.py \
  --prediction-dir HA-CQI/outputs/gf3_zhuozhou \
  --ground-truth datasets/GF3_Zhuozhou/GF3_Zhuozhou_label.tif
```

河南仅用于外部校准/诊断，涿州为锁定测试；两者均不得参与训练 checkpoint 的 epoch 选择。

## 7. 当前实验边界

- 不改 backbone、HA、CQI、decoder、激活函数、输入尺寸和后处理；
- 不启用 70/30 sampler、hard-negative mining 或 SAR nuisance augmentation；
- baseline 保持 batch 6/fp16；batch 12/bf16 后续单变量验证；
- 当前 75/25 随机划分保留 2,044 对跨 split 重叠窗口，因此不是严格跨区域验证。
