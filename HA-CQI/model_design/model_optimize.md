# HA-CQI 证据审计与 B2 + OSCD 后续路线

> 状态（2026-08-23）：A 路线的数据、训练、checkpoint、评估与 loss 已落地；模型主线已
> 断代切换为 EfficientNet-B2-only，修正冻结 DINOv3 输入契约，并以 MMSCoPE-inspired
> OSCD 替换无集合监督的第二套 decoder queries。本文档描述当前代码，不把尚未完成的
> B2 + OSCD 正式训练结果写成结论。
>
> 证据标记：**已证实**表示代码、数据报告或结果直接支持；**高概率**表示机制与现象一致；
> **待消融**表示当前证据不足以判断因果贡献。

## 1. 当前锁定契约

- 默认数据集为 `datasets/S1GFloods_CD_DINO_BG_75_25`；样本总计 6,752，train/val 为
  5,064/1,688，全背景为 437/144。
- 划分仍是合并 S1GFloods 与 VarFloods 后全局 `shuffle(seed=42)` 的 75/25 随机划分，存在
  2,044 对跨 split 重叠窗口，因此 validation 不是严格跨区域泛化证据。
- backbone 固定为 EfficientNet-B2，五级通道为 `[16,24,48,120,352]`，stride 为
  `[2,4,8,16,32]`；B0/MobileNetV2 及四级 FPN 路径已移除。
- 冻结 DINOv3 ViT-S/16 LVD 只抽取 `[2,5,8,11]`；输入固定执行“数据集反归一化到
  `[0,1]` → ImageNet mean/std”，不再提供 `shared` 开关。
- decoder 固定为 `oscd_v1`：P3-P5 负责多尺度区域/全局上下文，P2/P1 负责逐级细节重建；
  不含 decoder queries、masked cross-attention、Hungarian matching、no-object 或集合式 loss。
- primary 固定为 validation Flood IoU，在 `0.05–0.95/0.01` 网格联合选阈值；IoU 并列时
  依次选择更高 precision 和更高 threshold。
- checkpoint 仅支持 v2；模型配置必须声明 `efficientnet_b2 + imagenet + oscd_v1`。推理
  阈值只来自显式 `--threshold` 或 `meta.selection.threshold`。
- 旧 B0/20260427 权重和输出只作磁盘归档，当前代码不再加载、初始化或 resume。

## 2. 当前架构与唯一职责

```text
单波段 SAR → 三通道重复灰度 → 数据集统计归一化
  → 共享双时相编码器
      ├─ EfficientNet-B2 → 五级 FPN → P1–P5 局部/多尺度特征
      └─ 反归一化 → ImageNet normalization → frozen DINOv3
         → 4 个指定中间层 → adapter → P3–P5 语义锚
  → HA
      ├─ P1/P2 pair-shared statistics + learnable canonical affine
      └─ P1–P3 可选 deformable soft alignment
  → CQI：pre/post/diff/abs-diff 与 change queries 双向交互，输出 D1–D5
  → OSCD
      ├─ D3–D5 projection → region aggregation → Pixel Unshuffle → four-way SS2D
      └─ D2/D1 progressive detail reconstruction
  → binary dense head + P1–P5 auxiliary heads
  → 2 类 logits → softmax → checkpoint selection threshold
  → Focal + Tversky + auxiliary + support/coarse consistency
```

职责边界：FPN 负责多尺度局部表达；DINO 只提供 P3–P5 语义锚；HA 负责可比性；CQI 负责
用唯一一套 change queries 构造变化原语；OSCD 不再查询双时相关系，只做 multi-scale
context aggregation 与 mask reconstruction。由此消除了原 decoder 与 CQI 的 query 职责重叠。

## 3. 本轮已修正的确定性问题

| 问题 | 代码证据 | 当前处理 |
|---|---|---|
| B0/四级/MobileNet 分支造成结构歧义 | builder、FPN、推理入口含多分支 | 固定 B2 五级路径并删除 MobileNetV2 |
| DINO 输入直接沿用数据集 normalization | CNN 与 LVD 预训练分布契约不同 | 反归一化后固定 ImageNet mean/std |
| DINO 内部强制 fp16 | 覆盖外层 bf16/fp32 上下文 | 删除内部 autocast，继承外层精度 |
| 先物化 12 层再筛 4 层 | 8 个中间输出无消费者 | 直接请求 `[2,5,8,11]` |
| decoder queries 缺少集合式监督且与 CQI 职责重叠 | class/mask query 分支直接调制 dense logits | 删除第二套 queries，改用 OSCD dense reconstruction |
| PSC 被描述为累计源域原型 | 参数只通过梯度学习，无累计状态 | 改称 learnable canonical affine |
| decoder 上下文/细节贡献不可观察 | 旧日志只记录 query/scale weights | 记录 scan、context、P2/P1 detail RMS 及比例 |

旧 decoder 删除审计：源码 SHA256 为
`991c5bb671bb0c928d6b7bb973b2932b72eec149fb62c892fd0e0a6706de5845`，参数量
926,673，接口输出为 `[B,2,H,W]`。该记录只用于断代追溯，不保留运行时兼容分支。

实现依据为 [SegMAN CVPR 2025 论文](https://openaccess.thecvf.com/content/CVPR2025/html/Fu_SegMAN_Omni-scale_Context_Modeling_with_State_Space_Models_and_Local_CVPR_2025_paper.html)、
[官方 decoder](https://github.com/yunxiangfu2001/SegMAN/blob/main/segmentation/mmseg/models/decode_heads/segman_decoder.py)
与 [Mamba 官方 selective-scan 实现](https://github.com/state-spaces/mamba)。OSCD 保留
region aggregation、Pixel Unshuffle 和一次 SS2D，但将输入改为 CQI 变化原语，并新增 P2/P1
细节重建；因此只称 MMSCoPE-inspired，不把未训练的性能写成创新结论。

## 4. 剩余瓶颈与证据等级

| 优先级 | 问题 | 证据与机制 | 状态 |
|---|---|---|---|
| P0 | validation 空间泄漏 | 2,044 对重叠窗口跨 split | **已证实，用户接受本轮不改** |
| P0 | 源域 patch 指标不等于跨传感器整景目标 | 无法覆盖 GF-3/Sentinel-1 scene-level FP 与辐射偏移 | **已证实** |
| P1 | SAR 辐射/传感器域偏移 | 独立 tile 百分位拉伸可能改变跨 tile 与跨时相关系 | **高概率，待消融** |
| P1 | loss 仍偏召回 | Focal 前景 0.75、Tversky beta 0.70→0.55、support 路径都保护前景 | **机制已证实，贡献待消融** |
| P1 | 高分辨率路径仍可能放大噪声 | P1/P2 经 FPN、HA、CQI、aux；OSCD 只在重建阶段使用它们 | **待消融** |
| P1 | B2 与新 DINO 输入的净收益未知 | 同时建立了新的模型/输入 baseline，尚无完整训练结果 | **需要重新训练** |
| P1 | OSCD 是否传播错误上下文 | SS2D 可增强空间一致性，也可能扩散伪变化 | **需要完整训练/消融** |

保留背景后大片 FP 已改善，支持“背景先验不足”是旧问题的重要贡献因子。剩余 FP 不能仅凭
视觉结果归因于 backbone、loss 或对齐模块，必须联合观察背景 tile FP、最大/P95 FP 连通域、
校准指标和外部整景结果。

## 5. 新 baseline 与训练契约

新 baseline 固定为：

```text
EfficientNet-B2 + frozen DINOv3 ViT-S/16 LVD
+ ImageNet DINO normalization + HA + CQI
+ OSCD v1 + A 路线 loss
+ batch 12 + bf16 + AdamW 1e-4/2e-4 + cosine 80 epochs
```

batch 从实际 corrected B0 baseline 的 10 增至 12，仅增加 20%；同时 B2 增加参数，因此 base
LR 保持 `1e-4`、head LR 保持 `2e-4`，避免同时改变模型、输入契约和优化强度。bf16 不使用
GradScaler；8 workers、pin memory 开启、persistent workers 关闭；不使用 gradient accumulation。
OSCD 独立实测参数为 4,230,274（4.230M）；全模型为 39,989,994（39.990M），其中可训练
18,388,842（18.389M），冻结 DINOv3 为 21,601,152（21.601M）。RTX 4090 上合成 batch 12
的 bf16 forward-loss-backward-AdamW step 在一轮预热后实测峰值为 13.165 GiB allocated、
14.039 GiB reserved，单步 0.898 s，logits 与梯度有限。该结果只证明运行契约和资源上限，
不代表模型性能提升。

MMCV 2.1 的 DCNv2 CUDA kernel 不支持 BF16，因此 DCNv2 内部使用局部 FP32 precision island，
输出恢复外层 dtype；其余网络仍继承 bf16 autocast。RTX 4090 D 上合成 batch 12 的一次完整
forward-loss-backward 实测峰值为 13.574 GiB allocated、14.410 GiB reserved，logits 与梯度均
有限；因此 batch 12 在当前 24 GB 设备上已通过单步显存验收，但完整训练吞吐和长期稳定性仍需
正式运行观察。

## 6. B2 baseline 后的单变量顺序

| 顺序 | 唯一变量 | 对照 | 假设成立信号 | 停止条件 |
|---:|---|---|---|---|
| 0 | 完整 B2 + OSCD baseline | 归档旧 decoder/B0 仅作历史参照 | 建立新契约基线 | 不作为模块因果结论 |
| 1 | support consistency `0` | `0.03` | FP 降且 tiny coverage 可接受 | IoU/tiny recall 明显下降 |
| 2 | Focal `0.5/0.5` | `0.25/0.75` | precision 与 FP 改善 | recall/IoU 损失过大 |
| 3 | Tversky `0.5/0.5` | `0.70→0.55` | P/R 更平衡 | 无稳定改善 |
| 4 | aux end `0.25` | `0.50` | 浅层噪声 FP 下降 | tiny coverage 损失过大 |
| 5 | deformable alignment off / P2-P3 only | P1-P3 | 跨域 precision 改善 | 源域与外部均下降 |
| 6 | DINO branch off | 当前冻结 DINO | 不降或泛化更稳 | 中大区域 IoU 明显下降 |

晋级门槛：macro IoU 提升至少 1 pp，或 FP 面积下降至少 20% 且 IoU 损失不超过 0.5 pp、
tiny coverage 损失不超过 2 pp；正式结论至少需 2/3 seeds 同方向。

## 7. 暂不实施

- 标准 Mask2Former 重构；
- 额外 decoder queries、Hungarian/no-object/set supervision；
- 删除 CQI、HA、DCNv2 或 P1/P2 auxiliary；
- pseudo-color SAR、pair-shared radiometric stretch；
- 更大 DINO、B3、输入分辨率提升或新 loss；
- 70/30 sampler、hard-negative mining 或 SAR nuisance augmentation。

这些改动只有在 B2 baseline 和上述结构减法不足时才进入下一阶段。现有 256 PNG 放大不会
增加真实信息；提高分辨率必须从原始 SAR 重新生成 native crop。

## 8. 正式训练观察项

- 主指标：Flood IoU/F1、Precision/Recall；
- FP：背景 tile FP rate、FP 像素比例、最大/P95 FP 连通域；
- 小目标：tiny/small/large coverage≥10% 与 ≥25% recall；
- 校准：PR-AUC、Brier、ECE；
- decoder 诊断：scan/context/P2 detail/P1 detail RMS 与 context/detail ratio；
- 外部域：河南只作校准/诊断，涿州为锁定测试，不参与 epoch 选择；
- 资源：4090 peak allocated/reserved、step time、是否出现 OOM/NaN/Inf。

只有完整训练与外部评估完成后，才能判断 B2、固定 DINO 输入契约和 OSCD 的实际收益。
