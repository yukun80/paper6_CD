# HA-CQI 运行说明

## 1. 默认 SAR 洪水变化检测训练

```bash
cd HA-CQI
bash trainval_s1gfloods.sh
```

默认入口会启用 HA-CQI 主线结构：

- 共享 `CNN-FPN + DINOv3` 层次语义编码器
- HA 浅层风格协调与 `P1/P2/P3` 软对齐
- CQI 多尺度变化查询交互
- 轻量 Mask2Former-style 二分类分割头
- `P1-P5` 辅助监督，用于兼顾小目标和大区域

## 2. 常用覆盖参数

```bash
cd HA-CQI
DATASET_NAME=S1GFloods_CD_DINO_ \
DATA_ROOT=../datasets \
RUN_NAME=S1GFloods-HA-CQI-vits16 \
BATCH_SIZE=8 \
BEST_METRIC=tiny_safe_combo \
EVAL_FG_THRESHOLD=0.40 \
bash trainval_s1gfloods.sh
```

默认数据集为 `datasets/S1GFloods_CD_DINO`。`datasets/S1GFloods_CD_DINO_`
只作为可选实验数据目录使用，需要通过 `DATASET_NAME=S1GFloods_CD_DINO_`
显式指定。

当前脚本支持通过环境变量覆盖：

- `DATASET_NAME` 
- `DATA_ROOT`
- `STATS_FILE`
- `DINO_ARCH`
- `DINO_WEIGHT`
- `BACKBONE`
- `BACKBONE_WEIGHT`
- `RUN_NAME`
- `BATCH_SIZE`
- `NUM_CHANGE_QUERIES`
- `CQI_HEADS`
- `MASK_DIM`
- `MASK_QUERIES`
- `MASK_DECODER_LAYERS`
- `MASK_HEADS`
- `BEST_METRIC`
- `EVAL_FG_THRESHOLD`

## 3. HA 软对齐消融

```bash
cd HA-CQI
SOFT_ALIGNMENT=0 \
RUN_NAME=S1GFloods-HA-CQI-noalign \
bash trainval_s1gfloods.sh
```

`SOFT_ALIGNMENT=0` 只关闭 HA 中的 deformable soft alignment，保留 pair-shared
style calibration、CQI、mask decoder 和辅助监督。

## 4. 小目标与大区域训练策略

默认使用：

- `BEST_METRIC=tiny_safe_combo`
- `EVAL_FG_THRESHOLD=0.40`
- final mask loss + `P1-P5` auxiliary loss

如需更偏整体区域 IoU，可覆盖：

```bash
cd HA-CQI
BEST_METRIC=iou_1 \
EVAL_FG_THRESHOLD=0.50 \
RUN_NAME=S1GFloods-HA-CQI-iou \
bash trainval_s1gfloods.sh
```

## 5. checkpoint 说明

训练目录自动解析为 `checkpoints/<resolved_run_name>`。最佳权重文件为：

```text
checkpoints/<resolved_run_name>/<resolved_run_name>_efficientnet_b0_best.pth
```

checkpoint 的 `meta.model_config` 会记录 HA-CQI 结构参数，测试与推理阶段可据此恢复模型。
验证可视化写入 `checkpoints/<resolved_run_name>/vis/`；测试预测在启用
`--save_test` 后写入 `checkpoints/<resolved_run_name>/pred/`。

## 6. CUDA/cuDNN 稳定性

默认训练会关闭 `cudnn.benchmark`，并对 HA-CQI 中已知容易触发 cuDNN
`FIND was unable to find an engine` 的小卷积局部绕开 cuDNN；模型和张量仍在
CUDA 上运行。

如果当前 PyTorch/cuDNN/CUDA 组合仍在其他卷积上报同类 `FIND`，可用全局兜底：

```bash
cd HA-CQI
HA_CQI_DISABLE_CUDNN=1 \
DATASET_NAME=S1GFloods_CD_DINO_ \
DATA_ROOT=../datasets \
RUN_NAME=S1GFloods-HA-CQI-vits16 \
BATCH_SIZE=8 \
BEST_METRIC=tiny_safe_combo \
EVAL_FG_THRESHOLD=0.40 \
bash trainval_s1gfloods.sh
```

`HA_CQI_DISABLE_CUDNN=1` 只禁用 cuDNN 后端，不会把训练切到 CPU。
