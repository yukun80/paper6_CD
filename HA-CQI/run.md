# HA-CQI B2 + OSCD 运行说明

## 1. 默认 baseline

```bash
cd HA-CQI
conda activate hacqi
bash trainval_s1gfloods.sh
```

默认配置固定为：

- 数据集 `S1GFloods_CD_DINO_BG_75_25`，自然频率采样；
- EfficientNet-B2 五级 CNN-FPN；
- 冻结 DINOv3 ViT-S/16 LVD，DINO 输入固定为 ImageNet normalization；
- HA + CQI + Omni-Scale State-Space Change Decoder（OSCD）；
- focal `0.25/0.75`、Tversky beta `0.70→0.55`、support/coarse `0.03/0.02`、aux `1.0→0.5`；
- batch `12`、8 workers、bf16、AdamW base LR `1e-4`、head LR `2e-4`、cosine 80 epochs。

`BACKBONE`、`DINO_INPUT_NORM` 及 MobileNetV2 接口已经移除。`BACKBONE_WEIGHT` 仍可覆盖，
但只能提供与 EfficientNet-B2 完整兼容的本地 PyTorch 权重。

首次运行需在 `hacqi` 环境安装固定 CUDA 扩展：

```bash
python -m pip install -r requirements-oscd.txt
python -m pip check
```

CUDA selective-scan 不可用或 ABI 不匹配时直接失败；CPU recurrence 仅用于结构测试。

validation 在 `0.05–0.95/0.01` 上联合选阈值，primary 固定为 Flood IoU；IoU 并列时依次
选择更高 precision 和更高 threshold。`0.40` 只用于 validation 细粒度诊断。

## 2. 可复现短步诊断

```bash
cd HA-CQI
conda activate hacqi
python trainval.py \
  --name smoke-hacqi-b2 \
  --dataset S1GFloods_CD_DINO_BG_75_25 \
  --dataroot ../datasets \
  --stats_file ../datasets/S1GFloods_CD_DINO_BG_75_25/channel_stats_s1gfloods_train.json \
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

`--max_train_steps/--max_val_steps` 只用于冒烟，正式训练不得保留。CUDA 可用但不支持 bf16
时会直接失败；需要旧 GPU 时必须显式改用 `--amp_dtype fp16`。

## 3. Checkpoint v2

新训练目录产生：

```text
*_efficientnet_b2_best_primary.pth
*_efficientnet_b2_last.pth
*_efficientnet_b2_epoch10.pth ...
selection.json
metrics.jsonl
options.json
```

完整续训：

```bash
cd HA-CQI
RESUME=checkpoints/<b2_run>/<b2_run>_efficientnet_b2_last.pth \
bash trainval_s1gfloods.sh
```

仅加载网络参数：

```bash
cd HA-CQI
python trainval.py <其余训练参数> \
  --init_checkpoint checkpoints/<b2_run>/<b2_run>_efficientnet_b2_best_primary.pth
```

resume 恢复 optimizer、scheduler、GradScaler、epoch、global step、随机状态和 DataLoader
generator，并对模型、loss、训练配置及 dataset fingerprint 做 fail-closed 校验。B0、`shared`
DINO normalization、非 `oscd_v1` decoder、v1/raw checkpoint 都会在 state-dict 加载前被拒绝。

## 4. B2 baseline 后的单变量消融

固定 B2、ImageNet DINO normalization、batch 12/bf16，其余候选依次测试；只有胜者进入下一轮：

```bash
# 1. 关闭 support consistency
SUPPORT_CONSISTENCY_WEIGHT=0 \
RUN_NAME=S1GFloods-HA-CQI-B2-ab01-support0 bash trainval_s1gfloods.sh

# 2. 平衡 focal class weights
FOCAL_BG_WEIGHT=0.5 FOCAL_FG_WEIGHT=0.5 \
RUN_NAME=S1GFloods-HA-CQI-B2-ab02-focal-balanced bash trainval_s1gfloods.sh

# 3. 平衡 Tversky
TVERSKY_BETA_START=0.5 TVERSKY_BETA_END=0.5 \
RUN_NAME=S1GFloods-HA-CQI-B2-ab03-tversky-balanced bash trainval_s1gfloods.sh

# 4. 降低后期 auxiliary 权重
AUX_LOSS_WEIGHT_END=0.25 \
RUN_NAME=S1GFloods-HA-CQI-B2-ab04-aux025 bash trainval_s1gfloods.sh
```

不要并行执行以上命令，也不要一次改变多个变量。每轮需显式携带前序胜者配置。

## 5. 测试与推理

```bash
cd HA-CQI
python test.py \
  --checkpoint checkpoints/<b2_run>/<b2_run>_efficientnet_b2_best_primary.pth \
  --gpu_ids 0 \
  --save_test
```

```bash
cd HA-CQI
python run.py \
  --checkpoint checkpoints/<b2_run>/<b2_run>_efficientnet_b2_best_primary.pth \
  --img_A /path/to/pre.tif \
  --img_B /path/to/post.tif \
  --output outputs/pair.png \
  --gpu_ids 0
```

推理阈值优先级为：显式 `--threshold` → checkpoint v2 `meta.selection.threshold`；两者均
缺失时失败。checkpoint 的 stats 路径会自动解析，显式 CLI 始终优先。

## 6. 当前边界

- 不改变 HA、CQI、DCNv2、辅助头、输入尺寸、loss 数值或整景后处理；
- 不实现标准 Mask2Former，不引入伪彩色、pair-shared radiometric stretch 或更大 DINO；
- OSCD 对 FP、tiny flood 和跨区域一致性的收益需要完整训练或消融实验验证；
- 当前随机 75/25 split 仍有 2,044 对跨 split 重叠窗口，不是严格跨区域泛化验证；
- 4090 的 batch 12/bf16 必须以实际 peak allocated/reserved memory 为准，不能由 CPU 冒烟替代。
