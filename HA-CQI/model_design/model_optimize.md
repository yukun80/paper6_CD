# HA-CQI 证据审计与后续优化路线

> 状态（2026-08-23）：A 路线的数据、训练、checkpoint、评估与 loss 工程已落地；本轮继续
> 清理兼容层。模型结构、backbone、HA、CQI、decoder 与 256 输入保持不变。
>
> 证据标记：**已证实**表示代码、数据报告或结果直接支持；**高概率**表示机制与现象一致；
> **待消融**表示当前证据不足以判断因果贡献。

## 1. 当前锁定契约

- 默认数据集：`datasets/S1GFloods_CD_DINO_BG_75_25`。
- 固定划分：合并 S1GFloods 与 VarFloods 后全局 `shuffle(seed=42)`，前 75% train、后 25% val。
- 样本数：总计 6,752，train/val 为 5,064/1,688；全背景为 437/144。
- 所有有效全背景 tile 均保留；自然频率采样，不启用 70/30 sampler。
- 该划分仍有 2,044 对跨 split 重叠 VarFloods 窗口。validation 只能作为同分布选择集，
  不能解释为严格跨区域泛化证据。
- primary 固定为 validation Flood IoU，并在 `0.05–0.95/0.01` 网格联合选择阈值；IoU
  并列时依次选择更高 precision 和更高 threshold。
- checkpoint 仅支持 v2。推理阈值只来自显式 `--threshold` 或
  `meta.selection.threshold`，缺失即失败。
- 20260427 权重与旧整景结果仅作磁盘归档，当前代码不再加载。

## 2. 当前架构理解

```text
单波段 SAR → 3 通道与数据集归一化
  → 共享双时相编码器
      ├─ EfficientNet-B0 → FPN → P1–P5
      └─ 冻结 DINOv3 ViT-S/16 → adapter → P3–P5 语义特征
  → Harmonized Alignment
      ├─ P1/P2 pair-shared style calibration
      └─ P1–P3 可选 deformable soft alignment
  → 五尺度 CQI：pre/post/diff/abs-diff 与变化查询双向交互
  → query-gated dense change decoder + P1–P5 auxiliary heads
  → 2 类 logits → softmax → selection threshold
  → overlap 拼接 → 可选整景 blob filter
  → Focal + Tversky + auxiliary + support/coarse consistency
```

只读审计曾得到总参数约 32.93M、可训练参数约 11.33M、冻结 DINO 约 21.60M。因此没有
直接证据表明容量不足；扩大 backbone 不能修复数据泄漏、背景先验或跨域校准问题。

### 设计叙事与实现的有效差异

| 设计项 | 当前判断 | 后续处理 |
|---|---|---|
| `Encoder → HA → CQI → Decoder` | 主线一致 | 保持 |
| PSC“累计源域原型” | 实际为可训练参数，没有累计过程 | 修正文稿或单独实现并消融 |
| Semantic Calibration 残差 | 实际是拼接后卷积、BN、激活与 CBAM，没有显式 identity residual | 修正文稿 |
| “Standard Mask2Former” | 当前没有 masked cross-attention、no-object、Hungarian matching 或 per-query set loss | 应称 query-gated dense decoder |
| size-aware supervision | 当前是固定多尺度权重与一致性项，不按 GT 组件大小分配监督 | 应称 multi-scale auxiliary supervision |

## 3. 当前瓶颈与证据等级

| 优先级 | 问题 | 证据与机制 | 状态 |
|---|---|---|---|
| P0 | validation 存在空间泄漏 | 2,044 对重叠窗口跨 split；会高估同分布泛化 | **已证实，用户接受本轮不改** |
| P0 | validation 与跨传感器整景目标仍不等价 | 源域 tile 指标不能表达 GF-3/Sentinel-1 的 scene-level FP 和辐射偏移 | **已证实** |
| P1 | SAR 辐射/传感器域偏移 | 独立 tile 百分位拉伸、时相散射差异与 pair-shared calibration 假设可能冲突 | **高概率** |
| P1 | loss 方向仍偏召回 | Focal 前景权重 0.75；Tversky beta 0.70→0.55；浅层 support 可传播错误前景 | **机制已证实，贡献待消融** |
| P1 | DINO 输入契约 | baseline 的 `shared` 延续 SAR 归一化；`imagenet` 分支已提供但尚需单变量训练判断 | **输入差异已证实，收益待消融** |
| P1 | 高分辨率噪声路径较重 | P1/P2 同时经过 FPN、HA、CQI、辅助监督和 decoder，可能放大 speckle/错位 | **待消融** |
| P2 | 大模型收益不确定 | 当前模型并非明显欠容量，且自然图像/光学预训练与 SAR 存在域差异 | **暂无“越大越好”证据** |

保留背景后，用户观察到大片 FP 已改善，这支持“训练背景先验不足”是原问题的重要贡献因子。
剩余大片 FP 更可能来自跨域辐射变化、浅层纹理响应、loss 偏召回与概率校准不稳定；具体占比仍需
单变量实验，不能仅凭视觉结果归因。

### 历史 checkpoint 结论（非操作性记录）

旧训练曾同时按 Flood IoU、tiny any-hit recall 和若干 tiny-heavy 组合指标保存多个 `best_*`。
any-hit 指标容易奖励过预测，而且 patch 指标不能约束整景连通 FP。该历史结论已促成现在的
`best_primary + selection threshold` 契约。旧文件不再用于当前训练、初始化、测试或推理。

## 4. 五项初始猜想的当前判断

1. **DINOv3 / EfficientNet 规模偏小：部分成立。** 可以作为后续对照，但当前没有容量瓶颈
   的直接证据。应先完成 DINO normalization 和 DINO-off 对照。
2. **模块重叠或特征冲突：部分成立。** FPN/HA 均改变浅层采样，P1/P2 又承受多重监督；
   是否有害只能通过结构减法验证。
3. **loss 无法抑制 FP：成立，但贡献量待消融。** 所有方向性设置都更重视召回，但不能据此
   把剩余大片 FP 全部归因于 loss。
4. **激活、batch、LR、optimizer、scheduler 不合理：部分成立。** AdamW、分组 LR、cosine
   和现有激活没有异常证据；小 batch 对 BN 稳定性值得单变量测试。
5. **整体架构不适配 SAR：部分成立。** 双时相共享编码、多尺度差异和显式变化交互方向合理；
   更明确的问题是 SAR 辐射契约与跨域验证，而不是缺少更多模块。

## 5. A 路线后的单变量实验顺序

每轮只改一项，只有胜者进入下一轮：

| 顺序 | 唯一变量 | 对照 | 假设成立信号 | 停止条件 |
|---:|---|---|---|---|
| 0 | corrected-data baseline | 当前旧 loss 数值行为 | 建立可复现基线 | 不作为消融结论 |
| 1 | `DINO_INPUT_NORM=imagenet` | `shared` | 外部 IoU/P 提升、FP 下降 | 源域和外部均无收益 |
| 2 | `SUPPORT_CONSISTENCY_WEIGHT=0` | `0.03` | FP 下降且 tiny coverage 可接受 | IoU 或 tiny recall 明显下降 |
| 3 | Focal `0.5/0.5` | `0.25/0.75` | precision/FP 明显改善 | recall 与 IoU 损失过大 |
| 4 | Tversky `0.5/0.5` | `0.70→0.55` | P/R 更平衡 | 无稳定改善 |
| 5 | aux end `0.25` | `0.50` | 浅层噪声 FP 下降 | tiny coverage 损失过大 |

建议晋级门槛：macro IoU 提升至少 1 pp，或 FP 面积下降至少 20% 且 IoU 损失不超过
0.5 pp、tiny coverage 损失不超过 2 pp。正式结论至少需要 2/3 seeds 同方向。

## 6. 后续结构消融与升级门槛

在数据、阈值与 loss 固定后，优先做结构减法：

1. query branch 关闭；
2. deformable alignment 全关；
3. alignment 仅保留 P2/P3；
4. DINO 分支关闭；
5. P1/P2 auxiliary 降权。

解释规则：

- query-off 不降反升：删除 query 分支，不重写 Mask2Former。
- alignment-off 提升跨域 precision：当前 alignment 可能在对齐散射伪变化。
- DINO-off 不降：先简化或调整 DINO，不升级到更大模型。
- P1/P2 降权降低 FP 且 tiny coverage 可接受：支持浅层噪声放大假设。

只有上述阶段不足时，才按单变量顺序测试 ViT-L SAT、EfficientNet-B2、native 384/512 crop
或真正的 query mask decoder。可借鉴 BAN 的“冻结基础模型 + 双时相适配器”、Changer 的简化
交互，以及 ChangeMamba 的长程时空关系思想；这些工作只提供设计候选，不构成立即叠加模块的
证据。

## 7. RTX 4090 资源建议

| 配置 | 物理 batch 起点 | 精度 | accumulation | checkpointing |
|---|---:|---|---:|---|
| ViT-S + B0，256 | 先复现 6，再测 12 | fp16 baseline；bf16 单变量 | 通常不需要 | 不需要 |
| ViT-S + B2/B3，256 | 8–12 | bf16 | 通常不需要 | 不需要 |
| 冻结 ViT-L SAT + B0，256 | 4 起测 | bf16 | 累积到 effective 12 | 通常不需要 |
| 解冻 ViT-L | 暂不推荐 | bf16 | 需要 | 需要 |

4090 的优先用途应是稳定 batch、三随机种子、外部场景评估和可解释消融，而非直接扩大模型。
现有 256 PNG 放大不会增加真实信息；提高分辨率必须从原始 SAR 重新生成 native crop。

## 8. 当前验收与下一阶段判断标准

- 主指标：Flood IoU/F1、Precision/Recall；
- FP 指标：背景 tile FP rate、FP 像素比例、最大与 P95 FP 连通域；
- 小目标：tiny/small/large 组件 coverage≥10% 与 ≥25% recall；
- 校准：PR-AUC、Brier、ECE；
- 外部域：河南只用于校准/诊断，涿州作为锁定测试，不参与 epoch 选择；
- 当前 validation 的 2,044 对重叠窗口必须随结果披露。

如果 Level 1–2 已使跨域 macro IoU 提升至少 2 pp、FP 面积下降至少 30%，且 tiny coverage
损失在 2 pp 内，则停止大规模架构重构。
