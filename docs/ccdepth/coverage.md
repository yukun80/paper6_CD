# 弱边界高覆盖模式

[返回目录](README.md) · [项目首页](../../README.md)

`runtime.coverage_mode` 默认 `baseline_v1`，保留原准入条件。显式选择
`weak_boundary_v1` 时，原条件不通过但至少有一个 `beta > 0` 的分量也进入
原求解器。正权重需要有效湿侧、干侧样本、有效坡度和未倒置的水位区间。
这取消了新增分量的三个高权重锚点及空间跨度要求，不改变边界权重、
初始化公式、目标函数、硬约束、更新顺序、迭代预算或收敛审计。

新增估计依赖较弱的水位证据，覆盖率提高不代表精度提高。尤其单点锚定
可能使一个分量主要依赖空间连续性推断，不能视作独立水位观测。
无正权重证据仍为状态 0；求解失败仍为状态 5；不填零、不扩张掩膜。
原本可求解的分量走相同数值路径。

`CCDepth_QA.tif` 新增位 `1024`：通过放宽边界准入且成功审计的像元。
用 `(QA & 1024) != 0` 提取，不能用 `QA == 1024`；它与成功位 16 及
其他 QA 位组合。状态 3/4 仍描述优化终态，不表示边界证据强弱。
TIFF 内嵌配置和 `solve.weak_boundary_pixels`；最终仍只保留 8 个 TIFF。
覆盖率使用 `CCDepth_depth_solved.tif`，包含有效零深度；旧规则的
`CCDepth.tif` 只保留大于 0.01 m 的水深，覆盖率会更低。

先按[运行准备](running.md)配置真实输入。以下命令独立从项目根目录进入 `CCDepth/`：

```bash
cd CCDepth
python -m ccdepth_local run --config configs/zhengzhou_label_highcoverage_v1.json
python -m ccdepth_local run --config configs/zhengzhou_seg_highcoverage_v1.json
```

两份配置使用独立的 `runs/CCDepth/Zhengzhou/*_highcoverage_v1` 目录。中断可在
相同命令后加 `--resume`；不能用旧源码的检查点恢复新运行。历史 TIFF
不修改。沿用已授权的行列直配模式，DEM 与掩膜原始 affine 存在偏移，
没有重采样；高覆盖模式不能消除该空间误差。

相关内容：[输入要求](inputs.md) · [输出说明](outputs.md)
