# 多尺度解码 OSCD

[返回目录](README.md) · [项目首页](../../README.md)

在 HA 与 CQI 已经完成可比较特征构造和洪水相关变化关系建模之后，OSCD 只承担多尺度上下文聚合与 dense mask reconstruction。该设计借鉴 SegMAN MMSCoPE 的多尺度区域聚合、Pixel Unshuffle 对齐与单次二维 state-space scan，但输入改为 CQI 输出的五级双时相变化原语，并针对小型积水保留独立的 P2/P1 细节重建路径。因此 OSCD 不是对 SegMAN decoder 的逐项复制。

首先，将 `D3-D5` 投影到等宽通道并对齐到 `D3` 分辨率，得到上下文基底 `F`。OSCD 对 `F` 分别采用 identity、`3\times3/stride\ 2` 和 `5\times5/stride\ 4` 区域聚合，形成三种有效感受野；前两路再分别使用倍率 4 和 2 的 Pixel Unshuffle 无损对齐到 `D3` 的四分之一空间尺度。通道压缩后，三路特征拼接为 omni-scale tensor，并仅执行一次横向、纵向及其反向的四方向 SS2D。该过程在 256 输入下发生于 `8\times8` 网格，用线性序列传播补充 CQI 所不承担的 dense 长距离空间依赖。

扫描结果上采样后与 `F` 的短路径、全局池化上下文以及 `D3-D5` stage features 共同重建 `R3`；随后依次与投影后的 `D2`、`D1` 融合，得到 `R2` 和 `R1`，最后通过二类卷积头输出 `[B,2,H,W]`。浅层 `D1-D2` 不进入全局扫描，从结构上限制 speckle 和轻微错位被长距离传播的机会，同时保留局部边界信息。非 32 倍数输入只在 `D3` 右侧和底部 pad 到 4 的倍数，并在上下文重建后裁回原尺寸。

OSCD 不使用 learnable mask queries、mask classification、masked cross-attention、Hungarian matching、no-object 类或 set loss。当前职责闭环为：HA 负责双时相可比性，CQI 在五个尺度分别维护 change queries，OSCD 不额外引入 queries 生成结构化变化原语，OSCD 负责 omni-scale context aggregation 与 detail reconstruction。SS2D 对大片 FP、tiny flood 和跨区域空间一致性的净收益仍需完整训练或消融实验验证，本文不以结构验收替代性能证据。

相关内容：[设计概要](design.md) · [CQI](design-cqi.md)
