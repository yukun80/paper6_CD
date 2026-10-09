# 协调对齐 HA

[返回目录](README.md) · [项目首页](../../README.md)

HA 的目标不是直接判别变化，而是在双时相交互之前先让双时相特征具备更稳定的比较基础。对于 SAR 洪水变化检测，浅层特征最容易受到成像风格、局部散射和细微错位的共同干扰；若这些不确定性直接进入变化交互模块，即使后续解码结构再复杂，也只能在被污染的变化原语上做补救。因此，HA 将前端的特征准备统一组织为一个协调模块，并在内部保留三个命名子模块：**Pair-shared Style Calibration (PSC)**、**Semantic Calibration** 和 **Deformable Alignment**。这三个子模块分别处理浅层风格统计偏移、中高层语义稳定性以及双时相局部几何对应关系，从而使变化查询交互模块面对的是“已协调的双时相特征”，而不是原始且混杂的多源响应。

**Pair-shared Style Calibration (PSC)** 作用于浅层 `P1-P2`，负责在不破坏双时相内部一致性的前提下压缩传感器风格偏移。由于同一对 `pre/post` 图像通常共享相同传感器风格，真正需要被抑制的是整对输入相对于训练源域的统计偏移，而不是双时相内部的差异。因此，对任意浅层尺度 `l \in \{1,2\}`，我们先估计一对输入共享的通道统计

\[
\left[\mu_{\mathrm{pair}}^l,\sigma_{\mathrm{pair}}^l\right]
=
\mathrm{Stat}\!\left(\left[P_{\mathrm{pre}}^l,P_{\mathrm{post}}^l\right]\right),
\]

并将当前浅层特征映射到由可学习 canonical affine 定义的稳定空间

\[
\tilde P_t^l
=
\frac{P_t^l-\mu_{\mathrm{pair}}^l}{\sigma_{\mathrm{pair}}^l+\epsilon}
\odot \sigma_{\mathrm{can}}^l
\oplus \mu_{\mathrm{can}}^l,
\quad
t \in \{\mathrm{pre},\mathrm{post}\}.
\]

这里的 `(\mu_{\mathrm{can}}^l,\sigma_{\mathrm{can}}^l)` 是通过梯度学习的参数，并不是运行时累计的源域原型统计，其中 `\sigma_{\mathrm{can}}^l` 由 softplus 保证为正。为了避免重标定削弱 SAR 特有的局部散射纹理，HA 在此之后再接入一个轻量残差恢复块 `R^l(\cdot)`，得到浅层协调特征

\[
\hat P_t^l = \tilde P_t^l + R^l(\tilde P_t^l).
\]

**Semantic Calibration** 作用于中高层 `P3-P5`，负责用冻结的 DINOv3 提供稳定语义锚，而不让视觉基础模型直接主导解码。设 `V_t^l` 为 DINOv3 在对应尺度上的语义表示，则校准后的中高层特征写为

\[
\bar P_t^l =
\mathrm{CBAM}_l\!\left(
\mathrm{Conv}_l\!\left([P_t^l,\phi_l(V_t^l)]\right)
\right),
\quad l \in \{3,4,5\},
\]

其中 `\phi_l(\cdot)` 是尺度适配后的语义映射函数，方括号表示通道拼接。实际实现采用 concat-conv-normalization-activation-CBAM，不包含显式 identity residual addition。该子模块的作用不是在全网重复注入 ViT 语义，而是先稳定中高层场景表达，为后续变化构造提供更可靠的区域级上下文。

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

相关内容：[设计概要](design.md) · [CQI](design-cqi.md)
