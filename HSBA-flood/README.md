# HSBA-flood

基于 HSBA-flood 论文思路复现的无监督 SAR 洪水检测工程。算法主线固定为：

1. 读取灾前参考影像 `XR` 与灾后洪水期影像 `XF`
2. 严格构造 `valid_mask`，排除 NoData / NaN / Inf
3. 计算变化图 `XC = XR - XF`
4. 分别在 `XF` 与 `XC` 上运行 HSBA，得到 `BM_F` 与 `BM_C`
5. 取交集 `BM = BM_F & BM_C`
6. 在 `BM` 内做双高斯拟合
7. 从 `XF` 的水体分布估计 `sigma_seed`
8. 联合搜索 `sigma_RG` 与 `delta_sigma_CD`
9. 执行 `seed + region growing + change detection`
10. 在 `XR` 上复用水体逻辑提取永久水体并剔除

## 为什么不能把 NoData 填成极低值

本工程最重要的约束是：NoData 绝不参与统计。  
若把 NoData 填成 `0`、`-9999` 或极小 dB 值，再参与直方图、Otsu、双高斯拟合或区域生长，会把无效区误当作“暗水体”，直接污染：

- BM 的双峰判断
- `Gw/Gnw` 与 `Gc/Gnc` 拟合
- `sigma_seed`
- `sigma_RG`
- `delta_sigma_CD`

因此代码统一采用 `valid_mask` 机制，所有算法步骤都只在有效像元内执行。

## 核心变量含义

- `XR`: 灾前正常期参考 SAR 影像
- `XF`: 灾后洪水期 SAR 影像
- `XC = XR - XF`: 变化图，洪水区域通常更偏正
- `BM_F`: `XF` 上通过 HSBA 选中的 bimodal 区域
- `BM_C`: `XC` 上通过 HSBA 选中的 bimodal 区域
- `BM`: 二者交集，用作全局类分布拟合区
- `sigma_seed`: 高置信暗水体种子阈值
- `sigma_RG`: 区域生长允许扩展到的散射上界
- `delta_sigma_CD`: 变化检测最小下降量阈值

## 永久水体剔除逻辑

先在 `XF` 上通过 `sigma_seed + sigma_RG + delta_sigma_CD` 得到洪水候选，再在 `XR` 上复用 `sigma_seed + sigma_RG` 提取参考期低散射水面样区域，最后：

`final_flood = cd_mask & (~permanent_water_mask)`

## 工程结构

```text
HSBA-flood/
├── configs/
├── outputs/
├── src/
│   ├── io_utils.py
│   ├── nodata_utils.py
│   ├── preprocess.py
│   ├── hist_utils.py
│   ├── gaussian_fit.py
│   ├── quadtree.py
│   ├── hsba.py
│   ├── region_growing.py
│   ├── threshold_search.py
│   ├── flood_mapping.py
│   ├── visualization.py
│   ├── evaluate.py
│   └── main.py
└── README.md
```

## 运行方法

在 `HSBA-flood/` 目录下运行：

```bash
python -m src.main --config configs/gf3_henan.yaml
python -m src.main --config configs/s1_henan.yaml
python -m src.main --config configs/gf3_henan.yaml --config configs/s1_henan.yaml
```

## 输出内容

每个数据集输出：

- GeoTIFF: `valid_mask`, `XF_valid`, `XR_valid`, `XC_valid`, `BM_F`, `BM_C`, `BM_intersection`, `seed_mask`, `rg_mask`, `cd_mask`, `permanent_water_mask`, `final_flood_map`
- PNG: 预览图、BM 图、两张直方图拟合图、RMSE 热图、最终叠加图
- 文本/参数: `run.log`, `config_used.yaml`, `fitted_params.json`, `summary_metrics.json`

## 工程化补充假设

- 输入影像已是 dB 域，不再额外做 log 转换。
- 论文里提到的 Gamma 5x5 去斑默认视为已在上游完成，本工程不再重复滤波。
- Sentinel-1 默认使用 `pre_clip + Post_clip` 配置，以保证逐像元变化检测的空间一致性。
- 区域生长采用 8 邻域形态学重建，实现上显式禁止跨 NoData。
- 当 BM 像元过少时，只做一次温和回退：放宽 `min_valid_ratio_per_tile` 与减小 `min_tile_size`。

## 大幅面低内存优化

- 主流程不再常驻 `XR_valid / XF_valid / XC_valid` 三张整图副本，只保留 `XR / XF / valid_mask`。
- `XC` 在 HSBA、阈值搜索和导出阶段都按需计算，不再预先构造整幅差值图。
- 区域生长改为 `scipy.ndimage.binary_propagation`，避免 Python `deque` flood fill 在大图上放大内存和耗时。
- `XF_valid.tif / XR_valid.tif / XC_valid.tif` 仍可导出，但改为分块写出。
- 预览 PNG 默认按 `preview_max_pixels` 自动降采样，避免仅为了画图再复制大数组。

可选配置：

- `save_selected_tile_details`: 是否把每个入选 tile 的详细拟合结果写入摘要 JSON，大图建议关闭。
- `export_full_valid_rasters`: 是否导出三张 valid 浮点中间结果。
- `preview_max_pixels`: 预览图的最大像素数，超过后自动降采样。
- `preview_downsample_stride`: 手动指定预览步长，未设置时自动估算。

## 常见失败原因与排查

1. `BM 交集像元过少`
   检查输入两景是否真正共网格，或适当放宽 `min_valid_ratio_per_tile`

2. `XF[BM] / XC[BM] 双高斯拟合失败`
   检查影像是否几乎单峰、NoData 是否正确屏蔽、`hist_bins` 是否过大

3. `搜索空间为空`
   检查 `sigma_seed` 是否过高，或 `delta_cd_*` / `sigma_rg_step` 是否配置异常

4. 结果出现大片假水体
   优先检查 NoData 是否被当作普通像元参与了任何统计过程
