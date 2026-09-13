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
也不作为训练准入条件。默认 `STATS_MODE=auto` 按当前 train A/B 的文件名、大小和 mtime 生成运行快照，
自动命中或重算 mean/std 缓存。每个 run 会保存包含实际文件名、样本/前景统计和归一化参数的
`data_snapshot.json`。若需复现实验，可显式使用 `STATS_MODE=file` 与 `STATS_FILE=...`；文件模式
只校验 train split、三个有限 mean/std 和正数 std，不绑定 manifest。

### 新制备 VarFloods 切片的整景拉伸

`scripts/prepare_fused_sar_cd_dataset.py` 使用独立的 `data/scene_stretch.py`，
与交付包的 `scene_pair_histogram_v1` 对齐，不依赖交付目录运行。
每个原始灾前/灾后场景对在共同有效 SAR 像素上联合拟合一次，再供所有 train/val 切片复用。
默认按非重叠 1024×1024 块扫描，使用 65,536 个直方图箱估计 P2/P98；
`--stretch-low/high` 调整整景联合百分位。映射不使用 GT，也不进行对数转换或直方图匹配。

`split_report.json` 的 `scene_stretches` 记录源图路径、百分位、算法版本、冻结上下限、
有效计数、直方图误差界、截断比例和退化处理。`--dry-run` 同样扫描与报告统计，但不写文件。
常数整景输出零；百分位退化时采用联合 min/max；拟合时无共同有效像素则报错。
没有入选切片的源场景不参与拟合，继续沿用原切片筛选规则。

S1GFloods 影像 PNG 原样复制；标签、原始 TIFF 导出及划分规则沿用既有逻辑。
此改动只影响未来新制备的 VarFloods PNG，不更新现有数据，也不改变直接读取 TIFF 的训练/推理入口。
新映射会改变输入分布，旧 checkpoint 精度未验证；当前划分规则也不构成空间独立性证明。

### 独立 test/推理场景制备入口

`python HA-CQI/scripts/prepare_tiles.py` 只使用命令行参数，不读取 YAML，
也不依赖交付目录；所有路径相对当前工作目录解析。影像制备复用上述整景拉伸模块，
输出 `test/A`、`test/B`、`test/valid_mask`、`tile_manifest.csv` 和 `prepare_report.json`。
默认 256×256、stride 128、最小共同有效比例 0.01、整景联合 P2/P98。
固定上下限可用 `--stretch-mode value --value-min ... --value-max ...`。
`--dry-run` 仅统计。`--overwrite` 只重建 `test/A`、`test/B`、`test/valid_mask`，
并更新影像 manifest 和制备报告；保留 `test/label`、`test/label_tif`、标签报告及其他文件。
同一源场景和相同窗口的已有标签引用会保留；源数据或切片参数改变后，应重新生成标签。
不覆盖源图所在目录。

从仓库根目录新建广西影像切片：

```bash
python HA-CQI/scripts/prepare_tiles.py \
  --pre-image datasets/LT1_Guangxi/LT_Guangxi_pre.tif \
  --post-image datasets/LT1_Guangxi/LT_Guangxi_post.tif \
  --output-dir datasets/LT1_Guangxi_CD_infer \
  --scene-id LT1_Guangxi
```

已有影像切片时直接运行标签制备，无需重切 A/B：

```bash
python HA-CQI/scripts/prepare_sar_scene_label_infer.py \
  --src-root datasets/LT1_Guangxi \
  --label-image LT_Guangxi_Label.tif \
  --tiles-root datasets/LT1_Guangxi_CD_infer \
  --strict
```

标签程序兼容旧报告的 CRS/仿射字符串及新报告的 WKT/EPSG、六参数仿射数组，
保留同网格校验，拒绝冲突 CRS 和错位标签，不改写输入报告。
标签写入 `test/label`、`test/label_tif` 并补充 manifest；标签与推理均按清单读取影像或有效区路径，
标签程序仅接受清单中 `test/` 下的路径；旧 `tiles/` 布局会在写出前报错，
需使用 `HA-CQI/scripts/prepare_tiles.py` 重建影像切片，不自动搬移或改写路径。
已有非空标签目录需显式 `--overwrite`，该参数在标签程序中仅替换标签目录。

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

CUDA_VISIBLE_DEVICES=0 \
RUN_NAME=HA-CQI-OSCD \
DATASET_NAME=S1GFloods_CD_DINO_BG_75_25_ \
STATS_MODE=auto \
SEED=3407 \
BATCH_SIZE=12 \
NUM_WORKERS=8 \
LR=1e-4 \
AMP=1 \
AMP_DTYPE=bf16 \
bash HA-CQI/trainval_s1gfloods.sh
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

每个 run 还会保存 `data_snapshot.json`。每次训练从 CNN/DINO 预训练权重开始，
epoch 从 1、global step 从 0 开始；不支持续训或用 HA-CQI checkpoint 初始化训练。
新 checkpoint 仅包含 `network` 和 `meta`，保留模型、阈值选择、epoch/global_step、训练与数据审计信息，
不保存优化器、调度器、scaler、随机数或数据加载器恢复状态。best_primary、last 和每 10 epoch 的
periodic 保存名称与时机不变。测试和推理入口兼容包含训练状态的旧 v2 checkpoint，且仍只接受声明
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

## SAR 整景瓦片推理

整景推理使用与 HA-CQI 相同的模型重建逻辑，将重叠瓦片的预测拼接回整景变化图。瓦片目录必须
包含 `tile_manifest.csv`，且清单中引用的灾前 A 图像、灾后 B 图像和 `valid_mask` 文件均可读取。

河南 GF3 场景示例：

```bash
python HA-CQI/scripts/infer_sar_scene_tiles.py \
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
python HA-CQI/scripts/infer_sar_scene_tiles.py \
  --tiles-root datasets/GF3_Zhuozhou_CD_infer \
  --checkpoint HA-CQI/checkpoints/<b2_run>/<b2_run>_efficientnet_b2_best_primary.pth \
  --gpu_ids 0 \
  --batch_size 8 \
  --output-dir HA-CQI/outputs/gf3_zhuozhou_corrected
```

广西 LT-1 场景测试：

```bash
python HA-CQI/scripts/infer_sar_scene_tiles.py \
  --tiles-root datasets/LT1_Guangxi_CD_infer \
  --checkpoint HA-CQI/checkpoints/<b2_run>/<b2_run>_efficientnet_b2_best_primary.pth \
  --gpu_ids 0 \
  --batch_size 8 \
  --skip-tiles \
  --output-dir HA-CQI/outputs/lt1_guangxi_corrected
```

其他场景同样使用 `infer_sar_scene_tiles.py`，显式替换输入与输出目录：

| 场景 | `--tiles-root` | `--output-dir` |
| --- | --- | --- |
| Brazos River | `datasets/USA_Brazos_River_CD_infer` | `HA-CQI/outputs/USA_Brazos_River_20260826_1` |
| San Jacinto | `datasets/USA_San_Jacinto_CD_infer` | `HA-CQI/outputs/USA_San_Jacinto_20260826_1` |

默认会保存切片级结果和整景拼接结果。整景输出位于 `<output-dir>/mosaic/`，包括：

```text
change_binary_raw.tif/png  # 阈值化后的原始二值结果
change_binary.tif/png      # 一致性伪斑过滤后的二值结果
```

`<output-dir>/infer_report.json` 记录检查点、阈值、瓦片数、输出路径、拼接源影像和模型配置。
主工程验证/推理在网络输出后使用 FP64 softmax、阈值比较及整景累积；网络、损失和混合精度
配置不变。这是数值精度实验，不代表模型性能提升。新运行不再写出 `change_prob.tif`，
报告记录 `probability_saved=false`、`probability_dtype=float64`；已有概率 TIFF 不删除。
PNG 中 NoData 与前景均显示白色，有效背景为深灰色，定量评估使用二值 TIFF 的有效掩码。
如需只保留原始拼接结果，可传入 `--disable_blob_filter`；如需跳过切片级 PNG/TIF 保存，可传入
`--skip-tiles`。

## 整景评估

```bash
python HA-CQI/scripts/evaluate_sar_scene.py \
  --prediction-dir HA-CQI/outputs/gf3_zhuozhou \
  --ground-truth datasets/GF3_Zhuozhou/GF3_Zhuozhou_label.tif
```

新报告默认触发仅二值图评估，忽略同目录遗留的旧概率 TIFF。该模式排除预测和标签 NoData，
包括标签未知值 `3`，直接报告 TP/FP/TN/FN、IoU、F1、P/R、准确率及有效/忽略像素数量。
不扫描阈值或输出最优阈值；报告中的生成阈值仅作来源记录。可显式指定原始二值图：

```bash
python HA-CQI/scripts/evaluate_sar_scene.py \
  --binary HA-CQI/outputs/<run>/mosaic/change_binary_raw.tif \
  --ground-truth datasets/GF3_Zhuozhou/GF3_Zhuozhou_label.tif
```

历史概率图仍可通过 `--probability` 或旧版 `--prediction-dir` 评估并扫描阈值。
为保持兼容，历史概率模式保留原先“覆盖区内未知 GT 归背景”的口径，与新二值模式不同；
报告记录 mode 和 label_policy，不能混用不同口径直接宣称精度提升。
评估报告不再计算 PR-AUC、Brier、ECE，也不再输出 `calibration` 指标块；
诊断的 `calibration` 数据角色与验证 IoU 阈值搜索保留。

以下跨域特征诊断依赖已有历史概率 TIFF，不适用于新生成的仅二值输出：

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

2026-09-12 补充 LT 样本：从原推理数据复制 6 组 train、2 组 val 的 A/B/label，来源保留；
现场成员为 train 4747、val 1597，后续仍以实时配对扫描为准，`stats_mode=auto` 自动刷新统计。
本次检查新增 LT 样本没有训练/验证窗口面积重叠，但不代表整个数据集空间独立。
广西已参与训练，其完整场景不能称为完全未见的外部测试。

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
