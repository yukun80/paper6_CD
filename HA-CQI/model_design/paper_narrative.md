# HA-CQI：面向 SAR 城市洪水范围制图的协调对齐与变化查询交互方法

*HA-CQI: Harmonized Alignment and Change Query Interaction for SAR Urban Flood Extent Mapping*

SAR 城市洪水范围制图的困难并不只来自双时相之间是否存在变化，更来自变化关系是否建立在可比较的双时相特征之上。对跨传感器或跨场景 SAR 影像而言，轻微但稳定的灰度色调和散射统计偏移会优先污染浅层局部响应；在此基础上，残余几何错位和高层语义不稳定会继续削弱双时相特征的可比性；即使特征已经具备较好的比较基础，城市道路积水、建筑阴影、永久水体邻域和 speckle 噪声仍会产生与洪水相似的伪变化响应。围绕这条问题链，HA-CQI 将方法主线组织为 **Harmonized Alignment (HA)** 和 **Change Query Interaction (CQI)** 两个核心模块，对应 `feature harmonization and alignment -> query-based bitemporal change primitive construction` 的因果顺序；随后采用 **Standard Mask2Former Head for Mask Prediction** 将多尺度变化原语转化为最终洪水范围图。

给定灾前图像 `I_pre` 与灾后图像 `I_post`，共享编码器提取双时相金字塔特征 `\{P_t^l\}_{l=1}^{5}`，其中 `t \in \{\text{pre}, \text{post}\}`。`P1-P2` 保留高分辨率局部纹理和边界细节，`P3-P5` 承担更稳定的中高层语义表达。HA 首先输出一组已经完成风格协调、语义校准和局部对齐的可比较双时相特征；CQI 再通过变化查询与双时相 pair tokens 的交互，在五个尺度上生成结构化变化原语 `\{D^l\}_{l=1}^{5}`；标准 Mask2Former head 以这些变化原语为多尺度输入，输出二值洪水范围预测，并为后续 CFDepth 提供 `SAR-derived change-defined flood support`。

## Harmonized Alignment (HA)

HA 的目标不是直接判别变化，而是在双时相交互之前先让双时相特征具备更稳定的比较基础。对于 SAR 洪水变化检测，浅层特征最容易受到成像风格、局部散射和细微错位的共同干扰；若这些不确定性直接进入变化交互模块，即使后续解码结构再复杂，也只能在被污染的变化原语上做补救。因此，HA 将前端的特征准备统一组织为一个协调模块，并在内部保留三个命名子模块：**Pair-shared Style Calibration (PSC)**、**Semantic Calibration** 和 **Deformable Alignment**。这三个子模块分别处理浅层风格统计偏移、中高层语义稳定性以及双时相局部几何对应关系，从而使变化查询交互模块面对的是“已协调的双时相特征”，而不是原始且混杂的多源响应。

**Pair-shared Style Calibration (PSC)** 作用于浅层 `P1-P2`，负责在不破坏双时相内部一致性的前提下压缩传感器风格偏移。由于同一对 `pre/post` 图像通常共享相同传感器风格，真正需要被抑制的是整对输入相对于训练源域的统计偏移，而不是双时相内部的差异。因此，对任意浅层尺度 `l \in \{1,2\}`，我们先估计一对输入共享的通道统计

\[
\left[\mu_{\mathrm{pair}}^l,\sigma_{\mathrm{pair}}^l\right]
=
\mathrm{Stat}\!\left(\left[P_{\mathrm{pre}}^l,P_{\mathrm{post}}^l\right]\right),
\]

并将当前浅层特征映射到由源域原型统计定义的稳定空间

\[
\tilde P_t^l
=
\frac{P_t^l-\mu_{\mathrm{pair}}^l}{\sigma_{\mathrm{pair}}^l+\epsilon}
\odot \sigma_{\mathrm{src}}^l
\oplus \mu_{\mathrm{src}}^l,
\quad
t \in \{\mathrm{pre},\mathrm{post}\}.
\]

这里的 `(\mu_{\mathrm{src}}^l,\sigma_{\mathrm{src}}^l)` 表示训练阶段累积得到的浅层原型统计。为了避免重标定削弱 SAR 特有的局部散射纹理，HA 在此之后再接入一个轻量残差恢复块 `R^l(\cdot)`，得到浅层协调特征

\[
\hat P_t^l = \tilde P_t^l + R^l(\tilde P_t^l).
\]

**Semantic Calibration** 作用于中高层 `P3-P5`，负责用冻结的 DINOv3 提供稳定语义锚，而不让视觉基础模型直接主导解码。设 `V_t^l` 为 DINOv3 在对应尺度上的语义表示，则校准后的中高层特征写为

\[
\bar P_t^l = P_t^l + \phi_l(V_t^l),
\quad l \in \{3,4,5\},
\]

其中 `\phi_l(\cdot)` 是尺度适配后的语义映射函数。该子模块的作用不是在全网重复注入 ViT 语义，而是先稳定中高层场景表达，为后续变化构造提供更可靠的区域级上下文。

**Deformable Alignment** 则负责处理双时相之间的局部几何残差。为此，HA 在 `P1-P3` 上对灾前特征执行可变形局部对齐，利用灾后特征作为 query，自适应搜索灾前特征中的最匹配采样位置：

\[
A_{\mathrm{pre}}^l(p)
=
\sum_{k=1}^{K}
w_k^l(p)\,
\hat P_{\mathrm{pre}}^l\!\left(p+\Delta p_k^l(p)\right),
\quad l \in \{1,2\},
\]

并在 `l=3` 时对 `\bar P_{\mathrm{pre}}^3` 采用同类可变形聚合。这样，HA 内部的 `PSC`、`Semantic Calibration` 和 `Deformable Alignment` 共同输出一组经过风格协调、语义校准和局部对齐的可比较特征；这些特征与灾后分支共同构成 CQI 模块的输入。HA 的作用因此非常明确：它输出的是可比较双时相特征，而不是最终的变化判别结果。

## Change Query Interaction (CQI)

CQI 是 HA-CQI 中负责变化关系建模的核心双时相交互模块。经过 HA 后，灾前与灾后特征已经具备更好的可比性，但 SAR 城市洪水变化的判别仍然不是简单的逐点响应匹配问题：淹没区域可能表现为后向散射减弱，也可能体现为道路、建筑阴影和水面邻域关系的重组；同时，speckle 噪声、建筑散射波动、阴影扰动和残余错位会产生与洪水相似的伪变化。基于这一观察，CQI 将 SAR 洪水变化定义为“可比较双时相特征中与稳定背景关系不一致的响应”。它不预设暗化、亮化或局部对比变化的固定形式，而是引入少量可学习的 change queries，从双时相 pair tokens 中主动聚合与真实变化相关的跨时相不一致关系。

给定 HA 输出的第 `l` 层可比较特征 `X_{\mathrm{pre}}^l` 与 `X_{\mathrm{post}}^l`，CQI 首先构造保留方向性和差异基底的双时相 pair tokens：

\[
Z^l
=
\mathrm{MLP}_z^l\!\left(
\left[
X_{\mathrm{pre}}^l,
X_{\mathrm{post}}^l,
X_{\mathrm{post}}^l-X_{\mathrm{pre}}^l,
\left|X_{\mathrm{post}}^l-X_{\mathrm{pre}}^l\right|
\right]\right).
\]

这里 `Z^l` 不是简单差分图，而是一个面向交互的双时相描述符：前两项保留灾前和灾后状态本身，第三项保留变化方向，第四项提供稳定的幅值差异基底。这样，模型仍能学习 SAR 洪水中的变暗、亮化和局部对比变化，但这些现象不再被写死为独立结构分支。

在 pair tokens 之上，CQI 为每个尺度维护一组紧凑的可学习变化查询 `Q^l`。这些查询不对应预设类别，而是作为变化模式的可学习原型，从 `Z^l` 中读取跨时相不一致关系：

\[
\hat Q^l
=
Q^l
+
\mathrm{Attn}_{q\leftarrow z}^l(Q^l,Z^l).
\]

随后，更新后的变化查询再把变化感知上下文回写到 dense pair tokens：

\[
\tilde Z^l
=
Z^l
+
\mathrm{Attn}_{z\leftarrow q}^l(Z^l,\hat Q^l).
\]

这一 two-way interaction 的作用是让少量查询先概括当前尺度中的变化模式，再用这些模式反向调制所有空间位置。与直接在 `X_{\mathrm{pre}}^l` 和 `X_{\mathrm{post}}^l` 之间做无约束 cross-attention 不同，CQI 的注意力发生在 pair tokens 与 change queries 之间，因此不会把双时相特征过早混合成难以解释的表示。对于高分辨率浅层，`Attn` 可采用窗口化或稀疏路由形式以减少 speckle 和无关背景干扰；对于中高层，查询可聚合更大范围的区域一致性，用于识别大面积淹没和稳定背景之间的关系差异。

最终，CQI 将变化感知后的 pair tokens 映射为多尺度变化原语：

\[
D^l
=
\Phi_l\!\left(\tilde Z^l\right)
=
\Phi_l\!\left(
Z^l+\mathrm{Attn}_{z\leftarrow q}^l
\left(Z^l,Q^l+\mathrm{Attn}_{q\leftarrow z}^l(Q^l,Z^l)\right)
\right).
\]

`D^l` 的语义是“由变化查询解释后的双时相不一致响应”，而不是某一种固定差分形式。这样，CQI 把可比较特征转化为对 SAR 洪水变化敏感、同时对伪变化更稳健的结构化变化原语。HA 解决的是双时相特征是否可比，CQI 解决的是可比特征中哪些跨时相关系指向真实洪水变化；这些结构化变化原语随后交给标准 Mask2Former head 完成最终 mask prediction。

## Standard Mask2Former Head for Mask Prediction

在 HA 与 CQI 已经完成可比较特征构造和洪水相关变化关系建模之后，最终的洪水范围预测交由标准 Mask2Former head 完成。该预测头的输入为 CQI 输出的多尺度变化原语 `\{D^l\}_{l=1}^{5}`，其职责不是重新定义新的变化建模机制，而是将这些已经具备变化语义的多尺度特征组织为集合式 mask prediction 问题。

具体地，pixel decoder 首先对 `\{D^l\}_{l=1}^{5}` 进行多尺度融合，生成具有较高空间分辨率的 dense mask feature；同时，transformer decoder 维护一组 learnable mask queries，并通过与多尺度变化特征的交互逐层更新这些 queries。随后，class branch 预测每个 query 的类别置信度，mask branch 将更新后的 mask queries 与 dense mask feature 结合，生成候选洪水 mask。最终预测通过标准 mask classification 方式得到二值 flood extent mask，并将其作为后续 CFDepth 的 `SAR-derived change-defined flood support`。

因此，HA-CQI 的整体叙事闭环应保持克制而清晰：HA 负责把双时相特征协调到可比较状态，CQI 负责从可比较特征中生成面向洪水变化的结构化变化原语，Standard Mask2Former Head for Mask Prediction 负责把这些变化原语转换为最终洪水范围图。Mask2Former head 在本文中是标准密集预测头，而不是新的方法创新点；本文的创新仍集中在 HA 的可比较特征构造和 CQI 的变化查询交互。
