========================
一、输入数据
========================

共有两组数据，分别独立运行整套算法，并输出各自结果：

1. GF-3 数据对
   - 灾前参考图像（XR）:
     datasets/GF3_Henan/Pre_Zhengzhou_descending_clip.tif
   - 灾后洪水期图像（XF）:
     datasets/GF3_Henan/Post_Zhengzhou_descending_clip.tif

2. Sentinel-1 数据对
   - 灾前参考图像（XR）:
     datasets/S1_Henan/Zhengzhou_S1GRD_ASCENDING_VH_pre_clip.tif
   - 灾后洪水期图像（XF）:
     datasets/S1_Henan/Zhengzhou_S1GRD_ASCENDING_VH_Post.tif

注意：
- 所有影像都是 float32，且数值已经是 dB，不需要再做 log 转换。
- 两组数据中都可能包含 NoData 像素，必须读取原始 nodata 元数据；如果元数据缺失，也要检测 NaN、Inf，必要时支持用户在配置里手动指定 nodata。
- 不能用 0、-9999、最小值替换 NoData 后再参与拟合；正确做法是全流程使用 valid mask 排除 NoData。
- 灾前图像 XR 表示正常期参考，灾后图像 XF 表示洪水期。
- 变化图定义为：
  XC = XR - XF
  因为洪水发生后，水面通常导致后向散射降低，所以洪水区域在 XC 中通常偏大（正值更明显）。

========================
二、目标输出
========================

对每一组数据（GF3_Henan、S1_Henan）分别输出：

1. 中间结果 GeoTIFF
   - valid_mask.tif
   - XF_valid.tif（仅用于可视化，可保留 nodata）
   - XR_valid.tif
   - XC_valid.tif
   - BM_F.tif
   - BM_C.tif
   - BM_intersection.tif
   - seed_mask.tif
   - rg_mask.tif
   - cd_mask.tif
   - permanent_water_mask.tif
   - final_flood_map.tif

2. 中间结果图像 PNG
   - xf_preview.png
   - xr_preview.png
   - xc_preview.png
   - bm_f_preview.png
   - bm_c_preview.png
   - bm_intersection_preview.png
   - hist_fit_xf_bm.png
   - hist_fit_xc_bm.png
   - threshold_search_rmse_heatmap.png
   - final_flood_overlay.png

3. 日志与参数文件
   - run.log
   - fitted_params.json
   - config_used.yaml
   - summary_metrics.json（即便没有真值，也要记录各种统计量）

4. 如果存在真值掩膜接口，则预留评估模块
   - confusion_matrix
   - OA, Precision, Recall, F1, Kappa

========================
三、算法主线
========================

必须按如下思路实现，不允许简化成普通阈值分割：

Step 1. 读取并预处理 XR、XF
Step 2. 构造有效像元掩膜 valid_mask
Step 3. 计算变化图 XC = XR - XF
Step 4. 在 XF 上运行 HSBA，得到 BM_F
Step 5. 在 XC 上运行 HSBA，得到 BM_C
Step 6. 取交集 BM = BM_F & BM_C
Step 7. 在 BM 内分别对 XF 和 XC 做双高斯拟合
Step 8. 从 XF 的水体分布估计 seed threshold
Step 9. 联合搜索 region growing threshold 和 change detection threshold
Step 10. 在整幅图像上做 seed + region growing + change detection
Step 11. 在 XR 上提取永久水体/类水体低散射区域
Step 12. final_flood = flood_raw - permanent_water
Step 13. 导出结果与中间图件

========================
四、NoData 处理要求（必须严格落实）
========================

这是本任务的关键约束之一，必须专门实现：

1. 读取 rasterio profile 中的 nodata 值。
2. 同时检查：
   - np.isnan
   - np.isinf
   - 像元是否等于 nodata 元数据值
3. 构造：
   valid_mask = valid_XR & valid_XF
   只有灾前灾后都有效的位置才允许参与后续运算。
4. 对 XC 的计算必须写成：
   XC[valid_mask] = XR[valid_mask] - XF[valid_mask]
   XC[~valid_mask] = np.nan 或保持 mask
5. 所有直方图统计、Otsu 初始化、双高斯拟合、区域生长候选区、变化检测、永久水体剔除，都只能在 valid_mask 内执行。
6. 导出 GeoTIFF 时：
   - 最终洪水图可以写成 uint8（0/1），但 NoData 区域要写明确 nodata，例如 255
   - 中间浮点影像保持 float32，并在无效区域写 nodata/NaN
7. 可视化时也要遮罩无效像元，而不是显示成很暗的水体区域。
8. 代码中严禁任何“将 NoData 填充为极小值后参与阈值”的做法。

========================
五、HSBA 的具体实现要求
========================

HSBA 不是最终分类器，而是一个“自动寻找适合双峰建模局部区域”的四叉树分层方法。实现时必须忠实反映这一点。

--------------------------------
5.1 四叉树分层
--------------------------------

对输入图像 X（可为 XF 或 XC）：

1. level 0 为整幅图像。
2. 每个 tile 可分裂为四个子块：
   - top-left
   - top-right
   - bottom-left
   - bottom-right
3. 一直递归到最小 tile 尺寸 min_tile_size。
4. 推荐默认：
   - min_tile_size = 128 像元
   - 或保证 tile 中有效像元数 >= 4096
   二者可以组合使用。
5. 每个 tile 必须记录：
   - level
   - row_start, row_end
   - col_start, col_end
   - valid_pixel_count

--------------------------------
5.2 tile 跳过规则
--------------------------------

对任意 tile，如果出现以下情况，直接跳过，不做双高斯拟合：

1. 有效像元过少（例如 < 1024 或低于用户配置阈值）
2. 有效像元占 tile 面积比例太低（例如 < 0.5）
3. 所有有效像元几乎是常数，标准差过低
4. tile 被上层已经选中的 bimodal tile 完全覆盖

注意：
- 必须按照“从大到小层级扫描”实现。
- 如果某个较大 tile 已经通过三条件并被选中，那么它的所有子 tile 不再继续检查。
- 不要写成遍历完所有层再做冲突消解。

--------------------------------
5.3 tile 直方图拟合
--------------------------------

对每个候选 tile：

1. 取 tile 内 valid_mask 为 True 的像元。
2. 计算直方图：
   - 默认 bins = 128 或 256
   - 范围使用该 tile 有效像元的最小值和最大值
3. 用 Otsu 阈值作为双高斯初值分割：
   - 左侧类初值均值 mu1_0
   - 右侧类初值均值 mu2_0
   - 对应标准差 sd1_0, sd2_0
   - 峰值幅值 A1_0, A2_0
4. 拟合双高斯：
   h_f(y) = A1 * exp(-(y-mu1)^2 / (2*sd1^2)) + A2 * exp(-(y-mu2)^2 / (2*sd2^2))
5. 拟合建议：
   - 使用 scipy.optimize.curve_fit 或 least_squares
   - 设置合理 bounds，约束 sd > 0
   - 若拟合失败，返回 fail
6. 拟合完成后强制排序：
   - 保证 mu1 < mu2
   - 若不满足则交换两组参数

--------------------------------
5.4 三个通过条件
--------------------------------

对拟合后的双高斯，计算：

(1) Ashman’s D
AD = sqrt(2) * abs(mu1 - mu2) / sqrt(sd1^2 + sd2^2)
通过条件：AD > 2

(2) Bhattacharyya Coefficient
BC = sum_k sqrt(h_norm[k] * hf_norm[k])
其中：
- h_norm 是归一化后的真实直方图
- hf_norm 是归一化后的拟合直方图
通过条件：BC > 0.99

(3) Surface Ratio
area1 = A1 * sd1 * sqrt(2*pi)
area2 = A2 * sd2 * sqrt(2*pi)
SR = min(area1, area2) / max(area1, area2)
通过条件：SR > 0.1

只有同时满足 AD > 2、BC > 0.99、SR > 0.1 时，tile 才是 bimodal tile。

--------------------------------
5.5 生成 BM
--------------------------------

实现 HSBA(X) 后输出二值掩膜 BM：
- BM=1 表示该像元属于被选中的 bimodal tile
- BM=0 表示未被选中
- 无效像元位置可保留 0，但导出时要遵循 valid_mask

分别运行：
- BM_F = HSBA(XF)
- BM_C = HSBA(XC)

然后：
- BM = BM_F & BM_C & valid_mask

注意：如果 BM 中有效像元太少，程序不能崩溃，需要日志提示并尝试回退策略，例如：
1. 放宽 min_valid_ratio
2. 放宽 min_tile_size
3. 仍然不够则终止并给出清晰错误信息

========================
六、在 BM 内拟合全局类分布
========================

在 BM=True 的所有像元上重新做双高斯拟合：

1. 对 XF[BM] 拟合：
   - 一类是低散射水体类 Gw
   - 一类是较高散射非水体类 Gnw
   要求自动识别较小均值的那一类为水体类。

2. 对 XC[BM] 拟合：
   - 一类是变化较小的未变化类 Gnc
   - 一类是变化较大的变化类 Gc
   要求自动识别较大均值的那一类为变化类。

输出参数：
- A_w, mu_w, sd_w
- A_nw, mu_nw, sd_nw
- A_c, mu_c, sd_c
- A_nc, mu_nc, sd_nc

必须保存：
- BM 区域 XF 直方图 + 双高斯曲线图
- BM 区域 XC 直方图 + 双高斯曲线图

========================
七、阈值估计与联合搜索
========================

需要估计三个关键阈值：

1. sigma_seed
2. sigma_RG
3. delta_sigma_CD

--------------------------------
7.1 seed threshold
--------------------------------

目标：
- sigma_seed 表示“非常确定的水体种子阈值”

原则：
- 水体类 Gw 的均值较低（更暗）
- sigma_seed 应位于 Gw 主峰区域与经验直方图开始分离的位置附近

实现要求：
1. 在 BM 区域内建立归一化经验直方图 h_bm
2. 建立归一化的 Gw 曲线
3. 从低 backscatter 向高 backscatter 扫描
4. 找到第一个满足：
   abs(h_bm - Gw_norm) > tau
   的 bin 位置
5. tau 默认设为 0.01 或配置参数
6. 该 bin 中心值作为 sigma_seed

同时输出以下备用信息：
- 双高斯交点 threshold_intersection
- mu_w, mu_nw
这样便于调试与 sanity check

--------------------------------
7.2 region growing threshold 与 change threshold
--------------------------------

必须通过联合网格搜索确定 sigma_RG 和 delta_sigma_CD，而不是手工拍脑袋。

要求：
- sigma_RG > sigma_seed
- delta_sigma_CD > 0 通常更合理，但允许从 0 开始搜索

推荐搜索范围：

1. sigma_RG
   - 下界：sigma_seed
   - 上界：min(mu_nw, mu_w + 4*sd_w) 或 BM 区域有效高分位
   - 步长：0.1 dB 或 0.2 dB

2. delta_sigma_CD
   - 下界：0
   - 上界：8 dB
   - 步长：0.1 dB 或 0.2 dB

每组候选参数执行：

a. 用 XF <= sigma_seed 构建种子 seed_mask
b. 在 XF 上区域生长，允许扩展到 XF <= sigma_RG 的连通区域，得到 rg_mask
c. 在 rg_mask 内施加变化约束 XC >= delta_sigma_CD，得到 cd_mask
d. 在 BM 区域内统计 cd_mask 对应像元的经验分布
e. 计算该经验分布与理论水体分布 Gw 的 RMSE
f. 保存到搜索热图矩阵

选择 RMSE 最小的一组参数作为最终：
- sigma_RG_best
- delta_sigma_CD_best

必须输出：
- threshold_search_rmse_heatmap.png
- fitted_params.json 中记录搜索空间与最优值

========================
八、区域生长实现要求
========================

必须实现真正的区域生长，而不是简单把所有 XF <= sigma_RG 的像元全部选出来。

推荐做法之一：

方式 A：基于形态学重建
- marker = seed_mask
- mask = (XF <= sigma_RG) & valid_mask
- reconstruction(marker, mask)

方式 B：BFS/DFS flood fill
- 起点是 seed_mask 中所有 True 像元
- 邻域为 8 邻域
- 只有满足 XF <= sigma_RG 且 valid_mask=True 的相邻像元才可以被扩展进入 rg_mask

要求：
1. 默认使用 8 邻域
2. 严禁跨越 NoData 区域进行连通
3. rg_mask 必须完全由 seed 可达
4. 输出 seed_mask 和 rg_mask 供检查

========================
九、变化检测约束
========================

变化检测不是独立分类器，而是对区域生长结果的进一步筛选。

公式：
- cd_mask = rg_mask & (XC >= delta_sigma_CD_best) & valid_mask

解释：
- 只有在灾后更暗、且相对于灾前出现足够大散射下降的区域，才保留为洪水候选。

注意：
- 如果某些地物在两时相都很暗，但变化不显著，不应判为新洪水。
- 这一约束能抑制永久水体、阴影、道路等低散射背景的误检。

========================
十、永久水体/类水体剔除
========================

必须在 XR 上复用同样的水体判别逻辑，以剔除参考期已存在的永久水体或类水体低散射区域。

实现要求：
1. 使用 XR 而不是 XF
2. 参考以下规则生成 permanent_water_mask：
   - 先用 XR <= sigma_seed 作为 seed
   - 在 XR <= sigma_RG_best 范围内做区域生长
3. 得到 permanent_water_mask
4. 最终洪水图：
   final_flood = cd_mask & (~permanent_water_mask)

注意：
- 永久水体 mask 同样必须受 valid_mask 约束
- 如果 permanent_water_mask 与 final_flood 存在重叠，必须从最终结果中删除
- 输出 permanent_water_mask.tif

========================
十一、后处理要求
========================

在 final_flood 生成后，可做轻量后处理，但不能过度平滑：

1. 去除面积很小的孤立连通域
   - 例如小于 9 像元或小于用户配置阈值
2. 不能明显改变水体边界的主体结构

========================
十二、工程结构要求
========================

请实现为一个规范的 Python 项目，而不是只有一个超长脚本。推荐结构：

HSBA-flood/
├── configs/
│   ├── gf3_henan.yaml
│   └── s1_henan.yaml
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
├── outputs/
│   ├── GF3_Henan/
│   └── S1_Henan/
└── README.md

========================
十三、配置文件要求
========================

配置项至少包括：

dataset_name:
pre_path:
post_path:
output_dir:

nodata_override: null
hist_bins: 256
min_tile_size: 128
min_valid_pixels_per_tile: 4096
min_valid_ratio_per_tile: 0.5
ashman_d_threshold: 2.0
bc_threshold: 0.99
sr_threshold: 0.1
seed_tau: 0.01

sigma_rg_step: 0.1
delta_cd_min: 0.0
delta_cd_max: 8.0
delta_cd_step: 0.1

rg_connectivity: 8
min_component_pixels: 9

save_intermediate_tifs: true
save_intermediate_pngs: true
verbose_logging: true

========================
十四、鲁棒性要求
========================

必须考虑以下异常情况：

1. 双高斯拟合失败
   - 对单个 tile：直接跳过该 tile
   - 对 BM 全局拟合：允许切换为更稳健的初始化方式，仍失败则终止

2. 搜索空间全为空
   - 输出日志说明原因
   - 不允许静默失败

3. 全流程日志要清晰记录：
   - 原始 nodata 值
   - valid 像元比例
   - 每层 tile 数量
   - 每层通过 tile 数量
   - BM 占比
   - 全局双高斯参数
   - sigma_seed, sigma_RG_best, delta_sigma_CD_best
   - 最终洪水像元数

========================
十五、可视化要求
========================

输出的图要有科研复核价值：

1. XF / XR / XC 预览图
   - NoData 区域透明或灰色斜线，不可显示成正常低值
2. BM_F / BM_C / BM 预览图
3. XF[BM] 直方图 + 双高斯
   - 标出 mu_w, mu_nw, sigma_seed, intersection
4. XC[BM] 直方图 + 双高斯
   - 标出 mu_nc, mu_c, delta_sigma_CD_best
5. RMSE 热图
   - x 轴 delta_sigma_CD
   - y 轴 sigma_RG
   - 标出最优点
6. 最终洪水图叠加在 XF 或伪彩底图上

========================
十六、命令行接口要求
========================

main.py 必须支持：

1. 运行 GF3 数据：
   python -m src.main --config configs/gf3_henan.yaml

2. 运行 Sentinel-1 数据：
   python -m src.main --config configs/s1_henan.yaml

3. 支持 batch 模式：
   python -m src.main --config configs/gf3_henan.yaml --config configs/s1_henan.yaml

========================
十七、README 要求
========================

README 中必须说明：

1. HSBA-flood 的核心思想
2. 为什么不能把 NoData 填成极低值
3. XF / XR / XC 的定义
4. BM_F、BM_C、BM 的含义
5. sigma_seed、sigma_RG、delta_sigma_CD 的含义
6. 永久水体剔除逻辑
7. 代码结构说明
8. 运行方法
9. 常见失败原因与排查思路

========================
十八、必须遵守的实现原则
========================

1. 不要偷换成深度学习
2. 不要偷换成单次 Otsu 阈值
3. 不要忽略 NoData
4. 不要把 NoData 当作低散射水体
5. 不要把 region growing 简化成 XF <= sigma_RG 的全局阈值
6. 不要跳过 BM_F 与 BM_C 的交集步骤
7. 不要省略永久水体剔除
8. 所有关键参数都要可记录、可复现实验

========================
十九、最终交付内容
========================

请直接生成：

1. 完整项目代码
2. 两份配置文件
   - gf3_henan.yaml
   - s1_henan.yaml
3. README.md
4. 可以直接运行的 main.py
5. 所有模块的必要注释
6. 对关键算法步骤添加日志输出
7. 若某些论文细节未明示，请采用合理工程实现，并在 README 中明确标注“工程化补充假设”

现在开始实现整个项目，保证代码完整、结构清晰、可直接运行，并优先确保 NoData 处理正确、HSBA 逻辑正确、阈值搜索逻辑正确。
```
