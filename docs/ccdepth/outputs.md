# 输出说明

[返回目录](README.md) · [项目首页](../../README.md)

成功运行仅保留下列 8 个 TIFF；允许 GIS 后续自行生成 `.ovr`、`.aux.xml`。

- `CCDepth_WSE.tif`：成功域水面高程。
- `CCDepth_depth_solved.tif`：max(S-Z,0)，保留成功域有效零值。
- `CCDepth.tif`：FP64 的 S-Z>0.01 m 旧发布域，再编码为 Float32；不对舍入后的数值重新阈值化。
- `CCDepth_depth_signed.tif`：signed depth。
- `CCDepth_WSE_gradient_solved.tif`：水面梯度，m/m。
- `CCDepth_gradient_directions.tif`：bit0=x、bit1=y，域外255；单方向有效时保留可观测幅值。
- `CCDepth_status.tif`：0=支持不足，3=次目标接受，4=主目标回退，5=主目标失败，域外-1。
- `CCDepth_QA.tif`：1=掩膜无效，2=DEM无效，4=支持不足，8=失败，16=成功，32=主目标回退，64=非负且≤0.01，128=负signed depth，256=单方向梯度，512=hard下界活跃；1024=弱边界放宽准入且成功审计；背景0。

浮点 NoData=-9999；QA保留65535。每幅 TIFF 的 `CCDEPTH` 元数据命名空间内嵌实际配置、运行身份、源码/输入/预处理哈希、像元哈希和终态/失败原因汇总。`audit` 校验全部产品的像元哈希、网格、编码、有效域、状态、QA与计数；不生成外部报告。终端显示简短进度和结果。每分量详细审计仅在工作状态库中保存，成功清理后不再单独提供逐分量表。

1024 位的组合读取和覆盖率口径见[高覆盖模式](coverage.md)。

相关内容：[输入要求](inputs.md) · [恢复](recovery.md) · [历史输出](history.md)
