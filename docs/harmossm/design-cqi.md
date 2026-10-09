# 变化查询交互 CQI

[返回目录](README.md) · [项目首页](../../README.md)

CQI 是 HarmoSSM 中负责变化关系建模的核心双时相交互模块。经过 HA 后，灾前与灾后特征已经具备更好的可比性，但 SAR 城市洪水变化的判别仍然不是简单的逐点响应匹配问题：淹没区域可能表现为后向散射减弱，也可能体现为道路、建筑阴影和水面邻域关系的重组；同时，speckle 噪声、建筑散射波动、阴影扰动和残余错位会产生与洪水相似的伪变化。基于这一观察，CQI 将 SAR 洪水变化定义为“可比较双时相特征中与稳定背景关系不一致的响应”。它不预设暗化、亮化或局部对比变化的固定形式，而是引入少量可学习的 change queries，从双时相 pair tokens 中主动聚合与真实变化相关的跨时相不一致关系。

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

这一 two-way interaction 的作用是让少量查询先概括当前尺度中的变化模式，再用这些模式反向调制所有空间位置。与直接在 `X_{\mathrm{pre}}^l` 和 `X_{\mathrm{post}}^l` 之间做无约束 cross-attention 不同，CQI 的注意力发生在 dense pair tokens 与 change queries 之间，因此不会把双时相特征过早混合成难以解释的表示。当前强召回 baseline 对五个尺度均使用 query-token 全局交互；此前试验性的 P1/P2 query-free 路径因小目标召回下降风险且数据噪声证据更强，已从主线删除。

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

`D^l` 的语义是“由变化查询解释后的双时相不一致响应”，而不是某一种固定差分形式。这样，CQI 把可比较特征转化为对 SAR 洪水变化敏感、同时对伪变化更稳健的结构化变化原语。HA 解决的是双时相特征是否可比，CQI 解决的是可比特征中哪些跨时相关系指向真实洪水变化；这些结构化变化原语随后交给不含额外 queries 的 OSCD 完成最终 mask reconstruction。

相关内容：[设计概要](design.md) · [OSCD](design-oscd.md)
