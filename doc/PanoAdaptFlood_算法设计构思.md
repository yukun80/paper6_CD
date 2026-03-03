# PanoAdaptFlood：基于视觉基础模型的 SAR+相干性+文本条件洪水变化检测

更新时间：2026-03-03  
适用仓库：`/home/yukun/codes/paper6_waterlogging`

## 1. 研究目标

构建一个面向 UrbanSARFloods Benchmark 的多模态洪水变化检测框架，融合：
- `Panopticon` 的可变通道 SAR 建模能力；
- `AdaptOVCD` 的文本语义控制思想；
- 视觉基础模型（ViT/DINO 系列）的强表征能力。

目标任务：
- 输入：灾前/灾中双时相 SAR 强度与相干性（多通道）。
- 输出：像素级语义变化结果，至少包含：
  - `0`: background / no-change / no-flood
  - `1`: open-area flood change
  - `2`: urban flood change
- 支持文本指令控制推理目标：
  - 仅城市洪水变化
  - 仅开放区洪水变化
  - 同时输出两类变化

本项目优先级：**性能优先 + 创新合理性优先**，不强制零样本。

## 2. 可行性与复杂性评估

### 2.1 可行性结论

结论：**高可行（工程复杂度中高）**。

依据：
1. 数据层面匹配。
- UrbanSARFloods 提供了双时相、SAR 强度与相干性、城市/开放区域相关标注，天然适合变化检测。
2. 模型层面可复用。
- Panopticon 已有 `PanopticonPE + ChnAttn + ChnEmb`（`panopticon/dinov2/models/panopticon.py`）。
- AdaptOVCD 已有文本语义与阈值后处理模块（`AdaptOVCD-main/Module/*`、`core/identification/*`）。
3. 工程层面可落地。
- 已有数据转换脚本 `datasets/script/convert_urban_sar_floods_to_cd.py`，可统一为标准 CD 输入输出。

### 2.2 复杂性来源

主要复杂点：
1. 文本模块是否带来真实增益。
- 若文本仅用于后处理，创新和性能提升都有限。
- 需要把文本嵌入变成训练期可反传条件。
2. 城市洪水类（FU）长尾困难。
- 类间不平衡与边界破碎导致召回率和稳定性较差。
3. 双时相+多通道融合复杂。
- 需要兼顾时相差异、通道缺失鲁棒性、跨事件泛化。

### 2.3 风险判定

风险可控，不建议降级为纯单时相洪水分割。  
建议采用“主线监督模型 + 可插拔后处理模块 + 完整消融验证”的路线。

## 3. 任务定义与数据约定

### 3.1 任务定义

主任务定义为三类语义变化分割：
- `y=0`: 背景/无变化/非洪水
- `y=1`: 开放区洪水变化
- `y=2`: 城市洪水变化

可扩展任务：
- 二阶段：先二值变化，再 open/urban 分类；
- 多任务：联合预测 `change mask + urban/open region mask`。

### 3.2 数据输入约定（基于现有转换脚本）

来自 `datasets/script/convert_urban_sar_floods_to_cd.py`：
- 原始 SAR 8 通道映射：
  - `1 pre_intensity_vv`
  - `2 pre_intensity_vh`
  - `3 post_intensity_vv`
  - `4 post_intensity_vh`
  - `5 pre_coherence_vv`
  - `6 pre_coherence_vh`
  - `7 post_coherence_vv`
  - `8 post_coherence_vh`
- 转换后：
  - `A(T1)` = `[1,2,5,6]`
  - `B(T2)` = `[3,4,7,8]`
- 标签值：`[0,1,2]`

当前本地划分文件：
- `datasets/urban_sar_floods/Train_dataset.txt`
- `datasets/urban_sar_floods/Valid_dataset.txt`

## 4. 总体算法：FloodPan-TextCD（推荐主线）

### 4.1 架构总览

模型由六部分组成：
1. `Pano-SAR Encoder`：Panopticon 风格可变通道编码器。  
2. `Bi-temporal Fusion`：双时相变化特征融合模块。  
3. `Text-Conditioned Decoder`：文本条件引导解码器。  
4. `Region Auxiliary Head`：城市/开放区辅助监督头。  
5. `Adaptive Refinement (optional)`：ACT/ACF 后处理增益。  
6. `Uncertainty & Confidence`：置信校准与误检抑制。

### 4.2 模块细节

#### 4.2.1 Pano-SAR Encoder（借鉴 Panopticon）

设计：
- 输入采用 `x_dict` 形式，兼容 `PanopticonPE`：
  - `imgs`: `[B, C, H, W]`
  - `chn_ids`: `[B, C]`（或可扩展为 `[B, C, 2]`）
  - `spec_masks`: 可选通道掩码
- 主干使用 ViT（建议 `ViT-B/16` 或 `ViT-L/14`）。
- 对 T1/T2 采用 Siamese 共享权重编码。

关键增强：
- 通道 dropout / channel masking（迁移 `ListChnMask` 思路）；
- 通道补齐和缺失鲁棒（迁移 `ChnPad` 思路）；
- SAR 专用扰动增强（speckle/辐射扰动）。

#### 4.2.2 Bi-temporal Fusion（变化建模核心）

输入：`F1, F2`（T1/T2 多尺度 token）。  
融合分支：
- `Diff`: `|F2-F1|`
- `Concat`: `[F1,F2]`
- `Cross`: `CrossAttn(F1,F2)` 与 `CrossAttn(F2,F1)`

门控融合：
- `Fchg = Gate([Fdiff, Fcat, Fxattn])`

输出：变化语义特征图 `Fchg`，供后续解码。

#### 4.2.3 Text-Conditioned Decoder（借鉴 AdaptOVCD 文本设计）

文本类别原型：
- `t_bg`: background
- `t_open`: open-area flood change
- `t_urban`: urban flood change

机制：
- 文本编码器（CLIP/DGTRS-CLIP）输出文本向量；
- 文本向量作为 query，与 `Fchg` 做 cross-attention；
- 生成类别感知 mask logits。

关键约束：
- 文本分支参与训练反传，不仅做后处理过滤。

#### 4.2.4 Region Auxiliary Head（解决城市洪水难例）

增加区域辅助头：
- 预测 `urban/open/background` 区域图；
- 与变化头联合训练，减少 open/urban 混淆。

#### 4.2.5 Adaptive Refinement（可选）

复用 AdaptOVCD 思想作为增益模块：
- ACT：`Module/adaptive_change_thresholding.py`
- ACF：`Module/confidence_filtering.py`

用途：
- 作为推理后处理增强，抑制小斑块误检与低置信噪声。

## 5. 训练策略（监督优先）

### 5.1 训练范式

采用“全监督主线 + 少样本辅助实验”双路线：

1. 主线（全监督）
- 使用 UrbanSARFloods train/val 全量训练，追求最优性能。

2. 扩展（少样本）
- 在固定 backbone 上做小样本微调实验，评估样本效率。
- 该部分作为附加实验，不影响主线收敛。

### 5.2 分阶段训练

建议三阶段：
1. 阶段A：初始化
- 使用 Panopticon/SAR 相关权重初始化编码器。
2. 阶段B：无文本强基线
- 先训练 `Pano-SAR Encoder + Fusion + SegHead`，获得高性能基线。
3. 阶段C：文本联合训练
- 引入 `Text-Conditioned Decoder` 与区域辅助头联合优化。

### 5.3 损失函数

总损失：

`L = L_ce + λ1*L_dice + λ2*L_focal + λ3*L_txt + λ4*L_boundary + λ5*L_region`

说明：
- `L_ce`: 三类交叉熵（带类权重）
- `L_dice`: 缓解类不平衡
- `L_focal`: 强化难例学习
- `L_txt`: 区域特征与文本原型对齐损失
- `L_boundary`: 城市洪水边界强化
- `L_region`: 区域辅助头监督

### 5.4 不平衡与难例策略

- 类别重加权（优先提升 `urban flood` 召回）；
- 分层采样（事件级 + 类别级）；
- OHEM / hard patch mining；
- 早期以 recall 为主，后期再平衡 precision。

## 6. 文本指令机制

### 6.1 Prompt 设计

推荐模板：
- urban: `flooded urban blocks, flooded streets near buildings`
- open: `flooded open fields, rural inundation in open area`
- background: `dry ground, unchanged non-flooded area`

### 6.2 训练期增强

- 同义 prompt 随机替换；
- 正负 prompt 对比学习；
- prompt dropout（防止对单模板过拟合）。

### 6.3 推理期指令

支持三种模式：
1. 仅 `urban flood change`
2. 仅 `open flood change`
3. `urban + open` 同时输出

## 7. 实验设计

### 7.1 对比基线

1. UNet/ChangeFormer Siamese（CNN 基线）
2. ViT Siamese（无 PanopticonPE、无文本）
3. ViT + PanopticonPE（无文本）
4. ViT + Text Decoder（无 PanopticonPE）
5. 完整模型 FloodPan-TextCD

### 7.2 指标

主指标：
- `F1_open`, `F1_urban`
- `IoU_open`, `IoU_urban`
- `mF1`, `mIoU`

补充指标：
- Precision/Recall（重点关注 `urban recall`）
- 文本一致性准确率（不同指令下输出匹配程度）

### 7.3 必做消融

1. 去掉 PanopticonPE
2. 去掉文本分支
3. 文本仅后处理 vs 文本端到端训练
4. 仅强度 vs 强度+相干性
5. 有无 ACT/ACF
6. 有无区域辅助头

### 7.4 泛化评估

- 随机划分评估（标准）
- 跨事件评估（更严格，建议作为主结论支撑）

## 8. 工程落地方案

### 8.1 复用路径

Panopticon 复用：
- `panopticon/dinov2/models/panopticon.py`
- `panopticon/dinov2/data/augmentations.py`

AdaptOVCD 复用：
- `AdaptOVCD-main/Module/adaptive_change_thresholding.py`
- `AdaptOVCD-main/Module/confidence_filtering.py`
- `AdaptOVCD-main/changeformer/core/identification/*`

数据工具：
- `datasets/script/convert_urban_sar_floods_to_cd.py`

### 8.2 新增工程目录建议

- `models/panoadaptflood/`
- `configs/panoadaptflood/`
- `train_panoadaptflood.py`
- `eval_panoadaptflood.py`

### 8.3 MVP（最小可行版本）

MVP-v1：
1. 双时相输入（A/B）+ PanopticonPE 编码
2. 简单 `|F2-F1|` 融合
3. 三类分割头
4. `CE + Dice + Focal`

MVP-v2：
1. 加文本辅助损失（先不做复杂 cross-attn）
2. 加区域辅助头

Full：
1. 文本 cross-attn 解码
2. ACT/ACF 推理增强
3. 完整消融与跨事件评估

## 9. 里程碑

1. `M1 (2周)`：数据转换、训练管线、无文本 ViT 基线
2. `M2 (2周)`：接入 PanopticonPE，完成主干监督训练
3. `M3 (2周)`：接入文本条件解码与对齐损失
4. `M4 (1-2周)`：ACT/ACF、跨事件泛化与消融收敛

## 10. 预期贡献

1. 提出一个面向 UrbanSARFloods 的监督式多模态语义变化检测框架；
2. 系统融合 Panopticon 的 SAR 可变通道建模与 AdaptOVCD 的文本语义控制；
3. 在“开放区洪水变化 + 城市洪水变化”双任务上给出可复现、可扩展的视觉基础模型基线。

---

## 附：当前版本边界说明

- 本方案不依赖零样本设定，默认追求监督性能上限。  
- 若文本模块增益不足，可降级为辅助监督分支，但需在消融中明确报告。  
- `datasets/urban_sar_floods_CD/` 当前为空目录，需先执行转换脚本再开始训练。
