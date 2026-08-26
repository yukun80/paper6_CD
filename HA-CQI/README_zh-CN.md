# HA-CQI

HA-CQI 是本目录当前使用的合成孔径雷达（SAR）洪水变化检测算法，执行二值洪水变化分割。

## 模型设计

当前模型名称为 **HA-CQI**，由以下部分组成：

- **模块 I：分层 CNN-DINO 语义编码器**：共享的灾前/灾后编码器融合
  EfficientNet-B2 的五级 CNN-FPN 特征金字塔与冻结 DINOv3 语义特征，输出分辨率对齐的
  `P1-P5` 多尺度特征。CNN 使用数据集统计；DINO 分支先反归一化到 `[0,1]`，再固定使用
  LVD 的 ImageNet mean/std。默认只抽取真正参与 P3–P5 融合的 `[5,8,11]`，adapter 不再
  通过切片静默丢弃浅层特征。
- **模块 II：协调对齐（Harmonized Alignment, HA）**：先在浅层特征上执行成对共享的
  风格校准，缓解 SAR 辐射差异；随后可选地在 `P1/P2/P3` 上执行可变形软对齐。
- **模块 III：变化查询交互（Change Query Interaction, CQI）**：P1–P5 均由可学习变化查询
  通过双向注意力与多尺度双时相特征交互，为解码器提供强召回变化原语。
- **Omni-Scale State-Space Change Decoder（OSCD）**：以 `P3-P5` 变化原语执行多尺度区域
  聚合、Pixel Unshuffle 对齐和一次四方向 SS2D，再用 `P2/P1` 逐级恢复局部边界。它保持
  二类 dense prediction，不创建第二套 decoder queries 或集合式监督。
- **多尺度辅助头**：固定在 `P1-P5` 上提供辅助预测，使小型内涝斑块和大范围淹没区都能获得
  有效监督。

主要代码路径如下：

```text
model/architectures/ha_cqi.py
model/engine.py
model/modules/harmonized_alignment.py
model/modules/change_query_interaction.py
model/decode_heads/omni_scale_state_space_change_decoder.py
model/decode_heads/state_space_scan.py
trainval.py
test.py
run.py
trainval_s1gfloods.sh
```

## 数据集

训练数据加载器只以当前目录中的完整同名三元组为成员依据：

```text
datasets/<DATASET_NAME>/
├── train/{A,B,label}
└── val/{A,B,label}
```

默认数据集为 `datasets/S1GFloods_CD_DINO_BG_75_25_`。2026-08-26 清洗后的启动快照为
train/val `4737/1593`，但该数字不是代码硬约束。完整删除同名 A/B/label 三元组后，下次启动会
自动使用剩余样本；若只缺少其中任一文件，加载器会明确失败。训练期间不支持热删除。

`manifest_*.csv`、`split_report.json` 和历史 dataset fingerprint 仅作构建审计，不决定训练成员，
也不阻止 resume。默认 `STATS_MODE=auto` 按当前 train A/B 的文件名、大小和 mtime 生成运行快照，
自动命中或重算 mean/std 缓存。每个 run 会保存包含实际文件名、样本/前景统计和归一化参数的
`data_snapshot.json`。若需复现实验，可显式使用 `STATS_MODE=file` 与 `STATS_FILE=...`；文件模式
只校验 train split、三个有限 mean/std 和正数 std，不绑定 manifest。

## 预训练权重

默认本地权重路径如下：

```text
HA-CQI/dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth
HA-CQI/pretrained/efficientnet_b2_ra-bcdf34b7.pth
```

主线只支持 `efficientnet_b2`，需要本地 PyTorch `.pth` 或 `.pt` 权重文件且不会在线下载。
加载时所有 feature keys 必须匹配，只允许忽略 `conv_head/bn2/classifier` 的 8 个分类头键。
旧 B0 权重和 B0 checkpoint 保留为归档，但 B2-only 主线会明确拒绝。

## 运行环境

当前主线使用 `hacqi` conda 环境：

```bash
conda activate hacqi
python -m pip install -r requirements-oscd.txt
python -c "import torch, timm, mmcv, selective_scan_cuda; print(torch.__version__)"
```

`requirements-oscd.txt` 固定为 Python 3.11、Torch 2.4、CUDA 12、CXX11 ABI false 对应的
`mamba_ssm 2.2.4` 预编译 wheel，并固定 `einops 0.8.1`、`ninja 1.13.0` 和
`transformers 4.44.2` 以满足 wheel 元数据；后两者不进入 OSCD 前向图。CUDA 缺少或无法
加载 selective-scan kernel 时 OSCD 会直接失败；CPU 只运行仓库内 FP32 reference recurrence，
用于测试而非训练。安装后应执行 `python -m pip check`。

## 训练

从 `HA-CQI` 目录运行：

```bash
cd HA-CQI
conda activate hacqi
bash trainval_s1gfloods.sh
```

常用环境变量覆盖示例：

```bash
cd HA-CQI
DATASET_NAME=S1GFloods_CD_DINO_BG_75_25_ \
DATA_ROOT=../datasets \
STATS_MODE=auto \
RUN_NAME=S1GFloods-HA-CQI-B2-OSCD-DINO5-8-11-CLEAN-s1 \
BATCH_SIZE=12 \
NUM_WORKERS=8 \
LR=1e-4 \
AMP_DTYPE=bf16 \
EVAL_FG_THRESHOLD=0.40 \
bash trainval_s1gfloods.sh
```

默认训练脚本使用以下关键配置：

- `DATASET_NAME=S1GFloods_CD_DINO_BG_75_25_`
- `DATA_ROOT=../datasets`
- `STATS_MODE=auto`（`STATS_FILE` 仅在 `STATS_MODE=file` 时使用）
- `DINO_ARCH=dinov3_vits16`
- `DINO_WEIGHT=dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth`
- `BACKBONE_WEIGHT=pretrained/efficientnet_b2_ra-bcdf34b7.pth`
- `NUM_CHANGE_QUERIES=16`
- `DINO_FUSION_LAYERS="5 8 11"`
- `RADIOMETRIC_JITTER_MODE=shared`
- P1–P5 CQI 与 P1–P5 auxiliary 固定启用
- 原 Focal + foreground Tversky + support/coarse loss 固定启用
- decoder 固定为 `oscd_v1`：128 通道、四方向 SS2D、state dimension 1
- `EVAL_FG_THRESHOLD=0.40`（仅用于 validation 细粒度诊断）
- `SEED=1`
- `FOCAL_BG_WEIGHT=0.25`、`FOCAL_FG_WEIGHT=0.75`
- `BATCH_SIZE=12`、`NUM_WORKERS=8`
- `LR=1e-4`、head LR multiplier `2.0`
- `AMP=1`、`AMP_DTYPE=bf16`（bf16 不使用 GradScaler）
- DINO 输入归一化固定为 `imagenet`，不再提供 `shared` 开关

MMCV 2.1 的 DCNv2 CUDA kernel 不实现 BF16，因此仅 DCNv2 在内部回退 FP32 并将输出恢复为
外层 dtype；其余模型仍使用 bf16 autocast。这是算子兼容处理，不改变模型参数或结构。

新训练在 validation 的 `0.05–0.95`（步长 `0.01`）阈值网格上按
`Flood IoU → Precision → 较高 threshold` 联合选择，只生成：

```text
<run>_efficientnet_b2_best_primary.pth
<run>_efficientnet_b2_last.pth
<run>_efficientnet_b2_epoch10.pth ...
selection.json
metrics.jsonl
options.json
```

每个 run 还会保存 `data_snapshot.json`。完整续训使用 `RESUME=/path/to/*_last.pth`；仅初始化网络使用
`python trainval.py ... --init_checkpoint /path/to/checkpoint.pth`。resume 会校验
训练/loss/模型配置；数据成员或自动统计变化只给出醒目 warning，并在当前目录快照上继续。`--resume`、`--init_checkpoint`、测试和推理入口均只接受声明
`efficientnet_b2 + imagenet + oscd_v1` 的 checkpoint v2。2026-08-23/24 的早期 B2-OSCD
v2 checkpoint 可被精确映射为其实际有效层 `[5,8,11]` 以复现实验；其他缺失/模糊路由不会猜测。
`S1GFloods-HA-CQI-vits16-20260427/` 及其结果保留为磁盘归档，当前代码不支持加载。

此前试验性的浅层 query-free、浅层监督减法、dual-class overlap 和 boundary loss 已从主线删除，
避免在清洗数据的强召回 baseline 中继续叠加背景抑制。时相独立辐射增强仍保留为后续单变量实验，
但默认关闭：`RADIOMETRIC_JITTER_MODE=shared`。只有清洗后模型仍仅在跨域场景出现 FP 时，才测试
`RADIOMETRIC_JITTER_MODE=independent`。

训练结果写入：

```text
HA-CQI/checkpoints/<resolved_run_name>/
HA-CQI/checkpoints/<resolved_run_name>/vis/
```

关闭 HA 软对齐以进行消融实验：

```bash
cd HA-CQI
SOFT_ALIGNMENT=0 RUN_NAME=S1GFloods-HA-CQI-noalign bash trainval_s1gfloods.sh
```

## 测试

使用训练日志输出的实际运行目录名：

```bash
cd HA-CQI
python test.py \
  --checkpoint checkpoints/<resolved_run_name>/<run>_efficientnet_b2_best_primary.pth \
  --gpu_ids 0 \
  --save_test
```

检查点元数据保存了重建 HA-CQI 架构所需的选项。启用 `--save_test` 时，测试预测结果保存到：

```text
HA-CQI/checkpoints/<resolved_run_name>/pred/
```

## 单对影像推理

`run.py` 可直接对一组灾前/灾后影像执行推理：

```bash
cd HA-CQI
python run.py \
  --checkpoint checkpoints/<resolved_run_name>/<run>_efficientnet_b2_best_primary.pth \
  --img_A /path/to/pre_image.tif \
  --img_B /path/to/post_image.tif \
  --output outputs/run_pred.png \
  --gpu_ids 0
```

## GF3 整景瓦片推理

整景推理使用与 HA-CQI 相同的模型重建逻辑，将重叠瓦片的预测拼接回整景变化图。瓦片目录必须
包含 `tile_manifest.csv`，且清单中引用的灾前 A 图像、灾后 B 图像和 `valid_mask` 文件均可读取。

河南 GF3 场景示例：

```bash
python HA-CQI/scripts/infer_gf3_henan_tiles.py \
  --tiles-root datasets/GF3_Henan_CD_infer \
  --checkpoint HA-CQI/checkpoints/<b2_run>/<b2_run>_efficientnet_b2_best_primary.pth \
  --gpu_ids 0 \
  --batch_size 8 \
  --output-dir HA-CQI/outputs/gf3_henan_corrected
```

stats 默认从 checkpoint v2 metadata 解析。推理阈值只允许两种来源：显式
`--threshold`，或 checkpoint 的 `meta.selection.threshold`；两者都缺失时直接失败。

涿州 GF3 场景使用同一入口，仅替换瓦片根目录和输出目录：

```bash
python HA-CQI/scripts/infer_gf3_henan_tiles.py \
  --tiles-root datasets/GF3_Zhuozhou_CD_infer \
  --checkpoint HA-CQI/checkpoints/<b2_run>/<b2_run>_efficientnet_b2_best_primary.pth \
  --gpu_ids 0 \
  --batch_size 8 \
  --output-dir HA-CQI/outputs/gf3_zhuozhou_corrected
```

广西 LT-1 场景测试：

```bash
python HA-CQI/scripts/infer_gf3_henan_tiles.py \
  --tiles-root datasets/LT1_Guangxi_CD_infer \
  --checkpoint HA-CQI/checkpoints/<b2_run>/<b2_run>_efficientnet_b2_best_primary.pth \
  --gpu_ids 0 \
  --batch_size 8 \
  --skip-tiles \
  --output-dir HA-CQI/outputs/lt1_guangxi_corrected
```

默认会保存切片级结果和整景拼接结果。整景输出位于 `<output-dir>/mosaic/`，包括：

```text
change_prob.tif            # 拼接后的前景概率图
change_binary_raw.tif/png  # 阈值化后的原始二值结果
change_binary.tif/png      # 一致性伪斑过滤后的二值结果
```

`<output-dir>/infer_report.json` 记录检查点、阈值、瓦片数、输出路径、拼接源影像和模型配置。
如需只保留原始拼接结果，可传入 `--disable_blob_filter`；如需跳过切片级 PNG/TIF 保存，可传入
`--skip-tiles`。

## 整景评估

```bash
python HA-CQI/scripts/evaluate_sar_scene.py \
  --prediction-dir HA-CQI/outputs/gf3_zhuozhou \
  --ground-truth datasets/GF3_Zhuozhou/GF3_Zhuozhou_label.tif
```

评估器排除 metadata nodata 和历史标签值 `3`，并同时报告 raw/filtered 的 IoU、F1、P/R、
PR-AUC、Brier、ECE、背景 tile FP、FP 像素比例、最大/P95 FP 连通域及 tiny/small/large
coverage recall，并新增 2/4 像素容差 Boundary F1。河南只用于外部校准/诊断，涿州作为
锁定测试，不参与训练 epoch 选择。

统一跨域特征诊断（新 checkpoint 可额外输出 P1–P5/CQI 指标）：

```bash
python HA-CQI/scripts/diagnose_cross_domain_features.py \
  --tiles-root datasets/GF3_Henan_CD_infer \
  --probability HA-CQI/outputs/<henan_run>/mosaic/change_prob.tif \
  --ground-truth datasets/GF3_Henan/GF3_Zhengzhou_label.tif \
  --checkpoint HA-CQI/checkpoints/<run>/<run>_efficientnet_b2_best_primary.pth \
  --threshold <checkpoint_selected_threshold> \
  --role calibration \
  --output /tmp/<run>_henan_feature_diagnostic.json
```

Zhuozhou 必须使用 `--role locked_test`；LT1 使用 `--role qualitative` 且不能据此选模型。

## 代码校验

修改核心代码后，在 `HA-CQI` 目录执行：

```bash
python -m py_compile \
  option.py \
  model/architectures/ha_cqi.py \
  model/backbones/builder.py \
  model/modules/*.py \
  model/decode_heads/*.py \
  model/engine.py \
  model/checkpointing.py \
  data/transform.py \
  utils/flood_evaluation.py \
  trainval.py \
  test.py \
  run.py \
  scripts/diagnose_cross_domain_features.py

python -m unittest tests/test_pipeline_contracts.py

bash -n trainval_s1gfloods.sh trainval.sh
```
