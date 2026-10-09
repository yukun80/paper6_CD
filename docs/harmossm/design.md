# 模型设计

[返回目录](README.md) · [项目首页](../../README.md)

HarmoSSM 面向双时相 SAR 城市洪水范围制图；下述章节说明当前实现的组成。

SAR 城市洪水范围制图的困难并不只来自双时相之间是否存在变化，更来自变化关系是否建立在可比较的双时相特征之上。对跨传感器或跨场景 SAR 影像而言，轻微但稳定的灰度色调和散射统计偏移会优先污染浅层局部响应；在此基础上，残余几何错位和高层语义不稳定会继续削弱双时相特征的可比性；即使特征已经具备较好的比较基础，城市道路积水、建筑阴影、永久水体邻域和 speckle 噪声仍会产生与洪水相似的伪变化响应。围绕这条问题链，HarmoSSM 将方法主线组织为 **Harmonized Alignment (HA)** 和 **Change Query Interaction (CQI)** 两个核心模块，对应 `feature harmonization and alignment -> query-based bitemporal change primitive construction` 的因果顺序；随后采用受 SegMAN MMSCoPE 启发的 **Omni-Scale State-Space Change Decoder (OSCD)** 聚合多尺度空间上下文并重建最终洪水范围图。

给定灾前图像 `I_pre` 与灾后图像 `I_post`，单波段 SAR 首先以重复灰度形成三通道表示。共享 EfficientNet-B2 编码器使用当前训练目录自动统计的归一化输入，冻结 DINOv3 分支则先将该输入反归一化到 `[0,1]`，再应用 LVD 预训练对应的 ImageNet mean/std。当前强召回 baseline 显式抽取 DINO `[5,8,11]` 三层并全部用于 `P3-P5` 语义融合，不再抽取四层后在 adapter 内静默丢弃一层。编码器提取双时相金字塔特征 `\{P_t^l\}_{l=1}^{5}`，其中 `t \in \{\text{pre}, \text{post}\}`。`P1-P2` 保留高分辨率局部纹理和边界细节，`P3-P5` 承担中高层语义表达。HA 输出经风格协调和局部对齐的可比较双时相特征；CQI 再通过变化查询与双时相 pair tokens 的交互，在五个尺度上生成结构化变化原语 `\{D^l\}_{l=1}^{5}`；OSCD 以 `D3-D5` 建模多尺度空间上下文，再通过 `D2-D1` 恢复局部细节，输出二值洪水范围预测，并为下游水深估计提供 `SAR-derived change-defined flood support`。

## 章节

- [协调对齐 HA](design-ha.md)
- [变化查询交互 CQI](design-cqi.md)
- [多尺度解码 OSCD](design-oscd.md)

## 当前训练基线

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

## 实现入口

- [模型组装](../../HarmoSSM/model/architectures/harmossm.py)
- [训练与推理引擎](../../HarmoSSM/model/engine.py)
- [模块目录](../../HarmoSSM/model/modules/)
- [解码器目录](../../HarmoSSM/model/decode_heads/)

相关内容：[训练](training.md) · [历史资料](history.md)
