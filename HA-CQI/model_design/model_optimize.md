# HA-CQI 0824→0825 保召回优化实施基线

> 状态（2026-08-26）：低质量变化标注已由用户从
> `S1GFloods_CD_DINO_BG_75_25_` 的 train/val 目录中清理。当前主线恢复 0824 的强召回
> 表征，同时保留已证实合理的 B2、DINO 输入契约、显式层路由、HA、五级 CQI 与 OSCD。
> 所有性能收益仍需要完整训练或消融实验验证。

## 1. 当前证据与决策

- 清洗前 FP 形态与大量错误变化标注相似，数据监督噪声是比“背景 tile 不足”更强的根因。
- 0825 的 `[2,8,11]` 与背景抑制型减法降低成片 FP，但 Henan/Zhuozhou tiny/small recall
  同时下降，不能作为新的默认模型。
- 0824 实际路径为 DINO 抽取 `[2,5,8,11]` 后 adapter 丢首层，即有效
  `[5,8,11]`。当前实现直接请求并融合 `[5,8,11]`，不再隐式丢层。
- `[2,8,11]` 的 0825 checkpoint 仍按其显式元数据复现；这不改变新训练默认值。
- HA、五级 CQI、OSCD 不是 0824→0825 recall 下降的单独变量，保持不变。

## 2. 锁定模型路径

```text
Input
  → EfficientNet-B2 + frozen DINOv3 [5,8,11]
  → HA
  → P1–P5 CQI
  → OSCD
  → P1–P5 Auxiliary + Main Prediction
  → Focal + foreground Tversky + support/coarse consistency
```

固定恢复：

- Focal class weights `0.25/0.75`；
- Tversky beta `0.70→0.55`；
- auxiliary `1.0→0.5`；
- support/coarse `0.03/0.02`；
- P1–P5 CQI 与 P1–P5 auxiliary。

从主线删除：

- P1/P2 `local_structural`；
- `shallow_aux_supervision` 开关；
- dual-class overlap；
- Boundary loss。

Boundary F1、FP component、DINO 任意层和 P1–P5 特征诊断继续保留，因为它们不改变数值路径。
时相独立 brightness/contrast jitter 保留但默认关闭（shared），只作后续单变量实验。

## 3. 动态数据契约

训练成员只来自：

```text
train/A  train/B  train/label
val/A    val/B    val/label
```

每个 split 的三组文件名集合必须完全一致且非空。不读取 manifest 决定成员，不使用静默交集，
不校验 dataset fingerprint。完整删除一个三元组后，下次启动生效；训练过程中不支持热删除。

2026-08-26 当前只读审计：

| 项目 | Train | Val | 总计 |
|---|---:|---:|---:|
| 实际样本 | 4,737 | 1,593 | 6,330 |
| 背景 tile | 437 | 144 | 581 |

这些计数记录在运行快照中，不是代码硬编码验收门槛。

## 4. 自动统计与运行快照

训练默认 `stats_mode=auto`：

1. 扫描当前 train A/B；
2. 以相对文件名、大小和 mtime 生成非约束性的 train image snapshot ID；
3. 命中或重算 mean/std 缓存；
4. 缓存仅用于加速，不参与训练准入。

当前数据实测：

```text
train pairs = 4737
num_images  = 9474
mean = [0.5191673478, 0.5191673478, 0.5191673478]
std  = [0.2783174480, 0.2783174480, 0.2783174480]
```

每个 run 保存 `data_snapshot.json`，包括 train/val 实际文件名、样本数、背景/前景 tile、
前景像素比例、runtime snapshot ID、mean/std、stats 来源和启动时间。

`stats_mode=file` 用于历史复现，只校验 JSON 可读、split=train、三个 mean/std 有限且 std>0；
不校验 manifest SHA 或 fingerprint。

## 5. Checkpoint 与训练记录

- 每次训练从 CNN/DINO 预训练权重开始，不支持续训或整模型训练初始化；
- 新 checkpoint v2 仅保存 `network/meta`，不包含训练恢复状态；
- metadata 保留模型、loss、训练配置、epoch/global_step 与阈值选择，`dino_fusion_layers` 明确写入；
- data config 记录 runtime snapshot ID、实际计数与 stats 来源；
- 每次运行独立建立 best selection；best_primary、last、periodic 的名称与保存时机不变；
- 测试与推理兼容旧 v2，继续校验模型与推理阈值契约。

metrics 与 selection 使用运行时计数，不使用历史 manifest count。

## 6. 下一次完整训练

```bash
cd HA-CQI
conda activate hacqi

DATASET_NAME=S1GFloods_CD_DINO_BG_75_25_ \
DATA_ROOT=../datasets \
STATS_MODE=auto \
RUN_NAME=S1GFloods-HA-CQI-B2-OSCD-DINO5-8-11-CLEAN-s1 \
DINO_FUSION_LAYERS="5 8 11" \
BATCH_SIZE=12 \
NUM_WORKERS=8 \
LR=1e-4 \
HEAD_LR_MULT=2.0 \
AMP=1 \
AMP_DTYPE=bf16 \
SEED=1 \
bash trainval_s1gfloods.sh
```

该实验相对 0824 的核心科学变量是清除低质量监督；模型强召回路径恢复一致。不要同时启用
independent radiometric jitter 或修改 loss。

## 7. 评价与后续决策

优先比较验证集最佳 IoU 和匹配工作点下的指标：

- source validation best IoU；
- Henan/Zhuozhou tiny/small/large R10、R25；
- 与 0824 相同 Recall 下的 FP 像素比例；
- FP component 数量、P95、max；
- selected threshold 的 Precision/Recall；
- Boundary F1 仅作诊断；
- LT1 无 GT，只报告概率面积、组件与视觉结果。

解释规则：

- Recall 恢复且 FP 下降：错误标签是主要来源，停止模型改造；
- Recall 恢复但 FP 仍高：下一步只测试整景分时相百分位/辐射映射；
- Recall 仍低：先审计是否误删有效变化样本，不继续修改 decoder/loss；
- 只有跨域场景仍有 FP：再单变量测试 independent radiometric jitter。

## 8. 暂不实施

- 新数据集重建或严格 manifest/fingerprint 契约；
- P1/P2 query-free 或 DINO layer-2 gate；
- dual overlap、Boundary loss、新背景抑制 loss；
- OSCD/HA/CQI 重构；
- 默认 SAR 色调映射、伪彩色、更大 DINO 或更高输入分辨率；
- 完整训练由用户在工程验收后显式启动。
