> 实施修订（2026-09-21）：已采用用户确认的原EPSG:4326网格方案；本文件原米制投影主线、ArcPy及GEE兼容门禁以 IMPLEMENTATION_DECISIONS.md 和 README.md 为准。实际阶段状态以 reports/Gxx/report.json 的源码哈希和实测结果为准。

# CFDepth_local 逐模块构建与验收方案

版本：实施规格 v1.0  
日期：2026-09-21  
适用仓库：`yukun80/paper6_CD`  
算法基线提交：`a88f1dd7897fc479e85ee2f6191506017a920457`  
算法来源：`CFDepth_geemap/cfdepth` 的 v3.2.1 数值规则。

本文是待执行的构建方案，不是代码完成报告。所有模块和验收门禁初始均为 PENDING；文中的命令、接口、阈值是开发合同，不表示仓库已经实现或本次已经运行。

## 1. 任务边界

输入只要求两个已经对齐的单波段栅格：`flood_mask.tif` 和 `dem.tif`。按本任务约定，输入新增洪水范围正确，有效像元中 1 为待估深区，0 为边界统计可使用的背景。不增加永久水体识别、掩膜语义分类、范围补全或变化检测模块。

NoData、文件外部、无效 DEM 仍需单独处理；这是数值有效域与栅格读写问题，不是重新讨论输入洪水范围是否正确。无效位置不参与湿干样本统计。

保留新版的软水位区间、内部硬地形下界、边缘软下界、八邻域连续性、四色六候选更新、主/次目标状态机、目标归一化和最终审计。不恢复旧版最近锚点分配、10 km 传播上限或固定 50 次平滑。[R1–R3]

首版不调整权重、锚点门槛、下界、预算或收敛参数；不增加学习模型、流量、河网、HAND 或外部水位。不以 FwDET 的 CostAllocation、一次线性求解后截断或低通滤波替代新版优化器。

主线是：Rasterio 读写 + NumPy/SciPy 参考计算 → 全流程正确 → Numba 加速。ArcPy 作为同一数组核心的可选 GIS 入口，不能复制成第二套算法。生产路径不依赖 GEE、geemap、网络、账户或云端资产。

## 2. 顺序、交付及门禁

| 阶段 | 模块 | 本阶段应回答的问题 | 门禁 |
|---|---|---|---|
| M00 | 工程、配置、参考快照与测试数据 | 后续究竟复现哪一个算法？ | G00 |
| M01 | 输入读写与网格合同 | 两个栅格是否被无损、正确地解释？ | G01 |
| M02 | 计算域、八连通、距离和坡度 | 域、拓扑与空间单位是否正确？ | G02 |
| M03 | 边界区间、权重与初始化 | 边界约束是否按源代码构建？ | G03 |
| M04 | 目标函数和单像元更新 | 每一次数值更新是否真正在最小化目标？ | G04 |
| M05 | 四色求解与主/次目标状态机 | 是否正确收敛、接受、重试和回退？ | G05 |
| M06 | 深度、梯度、QA 与结果审计 | 输出是否保留正确的值、有效域和失败原因？ | G06 |
| M07 | 命令行、检查点、恢复和 ArcPy 接口 | 是否能从 TIFF 稳定得到可复现产品？ | G07 |
| M08 | 编译加速、回归与真实场景验收 | 更快之后是否还是同一个算法？ | G08 |

固定执行循环：实现当前模块 → 当前单元测试 → 截至当前全部回归 → 写阶段报告 → 通过后进入下一模块。

任何必测项目 FAILED 或 SKIPPED 都不构成本阶段通过。不得通过删测试、改变期望数组、放宽容差、降低锚点门槛或改变掩膜来绕过失败。必要的合同修订单独记录依据、版本和影响，并重新验证受影响的上游模块。

阶段报告放在 `reports/Gxx/report.json` 和 `reports/Gxx/summary.md`，记录：实现源码哈希、基线提交、配置哈希、测试数据哈希、命令、环境版本、测试数量、通过/失败/跳过项目、最大数值差、残留问题及状态。上游代码变化后，旧报告不继续代表新版本通过。

可选的 GEE 对照和 ArcPy 对照分别报告。缺少 GEE 快照不阻止已经通过独立测试的本地米制模式，但不得标记“已与 GEE 全链等价”；选择 ArcPy 作为交付后端时，ArcPy 真机测试就不再是可选项。

## 3. 目录与数据接口

以下是待创建的目录，不覆盖现有 `CFDepth` 或 `CFDepth_geemap`。

```text
CFDepth_local/
  pyproject.toml
  README.md
  configs/
    projected_metric.json
    compat_v321.json
  reference/
    upstream/                 # 冻结源码，只供引用/测试
    defaults_v321.json
    manifest.json             # 原始来源、提交、文件哈希
  cfdepth_local/
    __init__.py
    __main__.py
    config.py
    contracts.py
    io_rasterio.py
    io_arcpy.py
    grid.py
    domains.py
    topology.py
    terrain.py
    boundary.py
    initialization.py
    energy.py
    coordinate.py
    solver.py
    state.py
    diagnostics.py
    products.py
    checkpoint.py
    pipeline.py
    cli.py
    kernels_numba.py           # M08 才启用
  tests/
    fixtures/
    reference_energy.py       # 独立写法，不调用被测能量函数
    test_m00_*.py
    ...
    test_m08_*.py
  reports/
  runs/
```

为避免后续模块反复改接口，M00 定义以下容器：

| 容器 | 最少字段 |
|---|---|
| `GridSpec` | CRS、6 参数 affine、height、width、横纵分辨率、单位、profile、全局行列偏移 |
| `RasterInputs` | DEM FP64、洪水布尔数组、mask_valid、dem_valid、GridSpec、输入哈希 |
| `DomainData` | observed、support、dry、hard、soft、boundary、component_id、分量表 |
| `TopologyData` | 有效像元编号、邻接、边权、degree、global row/col、color |
| `BoundaryData` | lower、upper、mid、beta0、beta、样本数、诊断、eligible、initial_S |
| `ComponentState` | status、attempt、sweeps、stable、prev_primary、prev_total、base_primary、base_mid |
| `SolverResult` | S、baseS、分量状态、收敛记录、最终审计、求解有效域 |

浮点内部计算统一 float64。布尔域不得与浮点数值数组共用 NoData 哨兵。分量编号使用整数，背景编号为 0；分量 status 的 0 表示支持不足，不能与背景编号混淆。

配置分成 `algorithm` 和 `runtime`。前者完整固定 v3.2.1 数值规则，后者才包含输入路径、后端、输出路径、日志和检查点设置。原 `tileScale/maxPixels` 等云端配置保存在参考快照，不伪装成本地数值参数。

## M00：工程、基线和测试数据

### 实现

读取固定提交中的 `numerics.py`、`defaults.json`、`solver.py`、`components.py`、`grid.py` 和相关测试；保存原始字节和哈希。原源码作为只读参考，不将当前 main 分支变化自动吸收进来。已有数值核查 ZIP 可用于补充参考，但不能替代对新模块的实际测试。

创建独立运行环境。已有 ArcGIS Pro 用户在克隆环境中测试 ArcPy；纯本地核心使用独立环境，不升级现有检测训练环境。主依赖为 NumPy、SciPy、Rasterio、pytest；Numba 在 M08 引入或在此前仅做安装检查。依赖版本经安装、导入和测试后锁定，不把网上“最新版本”当作兼容性证据。

生产包导入时不导入 ee、geemap 或 arcpy。ArcPy 仅在所选适配器调用时导入。reference 中的上游函数不得成为生产运行对 GEE 的隐式依赖。

### 必须完整冻结的数值默认值

```json
{
  "pairRadius": 2,
  "peerRadius": 3,
  "slopeScaleDeg": 5,
  "sigmaFloor": 0.25,
  "dispersionScaleMeters": 1,
  "madScale": 1.4826,
  "minSamplesPerSide": 3,
  "highWeight": 0.5,
  "minAnchors": 3,
  "minSpanPixels": 2,
  "minDepth": 0.01,
  "lambdaC": 1,
  "lambdaB": 10,
  "lambdaT": 1,
  "muRatios": [0.001, 0.0001, 0.00001, 0.000001],
  "sweepsPerStage": 10,
  "maxSweeps": 2000,
  "residualTolerance": 0.001,
  "objectiveTolerance": 0.000001,
  "objectiveFloor": 1e-8,
  "budgetRelative": 0.0001,
  "budgetAbsolute": 1e-8,
  "hardTolerance": 1e-9,
  "monotonicTolerance": 1e-10
}
```

源配置还含诊断、NoData 和云端执行字段；原始 JSON 整体归档，产品 NoData 默认 -9999。默认值以固定源文件为准。[R3]

### 先建立而非临时拼凑的测试夹具

| 夹具 | 用途 |
|---|---|
| F01 平底盆地 | 内部 99 m、背景 100 m、米制 30 m 网格、足够背景；最终应为常水面 99.5 m/水深 0.5 m 的算法构造测试 |
| F02 人工平面 | DEM 坡度和 WSE 梯度解析检查；不把平面 DEM 自动当成全算法的水深真值 |
| F03 两个远隔分量 | 独立分量及无跨分量传播 |
| F04 对角连接、窄桥和孤立像元 | 八连通和四色正确性 |
| F05 NoData 孔洞、边缘截断 | 有效域和边界缺样本 |
| F06 合法负高程、零高程 | 防止把海拔符号当作无效或海洋判断 |
| F07 高 MAD、反向区间、零样本、无锚点 | 权重与支持不足路径 |
| F08 人工固定图和边界条件 | 独立能量、坐标极小值、状态机测试 |
| F09 非方形唯一编码栅格 | TIFF 往返、行列方向、原点与有效域 |
| F10 长条与较大连通分量 | 性能、内存与未收敛行为 |

F01 是明确构造出的期望数值解，不是新增实测水深精度。其他夹具的期望值采用手算、小型独立枚举或冻结的源结果，不能用待测生产函数自己生成答案。

### G00

基线文件哈希吻合；完整数值默认值一致；包可安装/导入；无 EE 生产依赖；未知配置字段和非法参数明确拒绝；夹具生成可复现。此阶段不写全流程 run 实现。

## M01：栅格读写与输入合同

### 实现文件和接口

`io_rasterio.py`、`grid.py`、`contracts.py`：

```python
read_inputs(mask_path, dem_path, config) -> RasterInputs
validate_alignment(mask_meta, dem_meta, config) -> GridSpec
write_raster(path, values, valid_mask, grid, dtype, nodata) -> None
```

主线先完成 Rasterio 读写，ArcPy 在 M07 接入同一个接口。

### 规则

检查单波段、有效掩膜内严格 0/1、DEM 有效 mask 与有限性、行列数、CRS 等价、完整 affine、像元大小、原点和单位。不把有效的 0/负高程删除；不只依赖 `isfinite` 判断数值有限的 NoData 哨兵。兼容模式还需保留上游 finite 判定的数值范围约定。

同网格输入直接进入计算，不再次最大值聚合、插值、裁剪或重投影。旋转/剪切或不支持单位明确拒绝，由独立预处理处理。已有正确坐标系不得清除。

主线 `projected_metric_v1` 使用北向、米制投影网格，垂直单位明确为 m。`compat_v321` 只作为原经纬度实现的对照配置：保留原距离定义，不用同一个公式混算度与米。首版不自动做垂直基准转换。

内部数组约定首行为栅格北侧、行号向南递增；坐标解释由 affine 决定。保留全局窗口偏移，以便后面分量裁切仍保持原四色奇偶。

### 验证

使用 11×17 的 F09，以 `1000*row + col` 编码，设置不对称孔洞与合法零值。写出、读回并比较每个像元中心坐标和数值；再测试窗口读取和拼回。测试半像元偏移、横纵分辨率不同、CRS 不同、0/1/3 未处理编码、全 NoData、负高程、零高程与 metadata-only NoData。

### G01

整数/布尔数组及 mask 完全一致；浮点往返等于预期 dtype 转换后的数组；CRS/shape/affine 保持合同。未对齐输入必须失败且不创建求解产品。完整输入哈希和网格摘要保存。

## M02：域、八连通、拓扑和地形量

### 实现

`domains.py`、`topology.py`、`terrain.py`：

```python
build_domains(inputs, params) -> DomainData
label_components(support) -> labels, component_table
build_topology(domain, grid, params) -> TopologyData
compute_slope(dem, dem_valid, grid, profile) -> slope_deg, slope_valid
```

已经对齐输入下，令 M 为洪水布尔数组、O 为掩膜有效域、V 为有效 DEM：

```text
observed = O
support  = O & M & V
dry      = O & ~M & V
hard     = erosion_3x3(support & O) & support
soft     = support & ~hard
boundary = support & ~erosion_3x3(support)
```

腐蚀使用 3×3 全 1 结构、一次迭代；文件外域按 False，不反射/周期填充，不把未知位置加入 dry。空 support 与当前上游一样明确报错，不生成伪成功空产品。

`scipy.ndimage.label` 显式提供 3×3 全 1 结构；默认二维 label 是四连通，不能使用默认值。[R4] 不增加最小面积清理、孔洞填补或矢量化简化。

同分量八邻居才连边，不包含中心像元。设 h_ref 为米制网格南北像元长度：

```text
w_pq = lambdaC * h_ref^2 / d_pq^2
d_pq = sqrt((dc*dx)^2 + (dr*dy)^2)  # 米制北向网格
```

正方形、lambdaC=1 时，正邻接权重 1、对角权重 0.5。非方形时分别按 dx/dy 计算；不要再次乘一次 lambdaC。每条无向边在全局能量中只计一次。

四色定义 `global_col % 2 + 2*(global_row % 2)`，颜色顺序固定 0、1、2、3。裁切分量包围盒后不重置全局奇偶。

### 坡度接口必须独立验证

投影主线定义 `central4_metric_v1`：在中心及上下左右 DEM 均有效时计算横纵中心差分，并取 `atan(hypot(gx,gy))` 转度；缺少必要邻居时 slope_valid=False，不填 0，也不偷偷切换成单边差分。该算子是明确的本地空间实现定义。

GEE 官方只明确 `ee.Terrain.slope` 用四邻居、输出度、边缘可缺值；这些说明不足以证明任意自写差分或 ArcPy 默认 Slope 与其完全数值等价。[R5] compat 模式需用小样本坡度快照校准；不能只凭都叫 slope 就宣称等价。

### 验证

F04 的对角像元必须同分量；仅相邻而不连通的块保持独立；F05 孔洞不参与邻接和 dry。检查每个 support 像元恰有一个编号、分量像元数之和等于 support 总数。

用枚举验证所有边的同分量、对称性、距离、权重和颜色不相等；正方形 30 m 测正邻 30 m、对角 30*sqrt(2) m；另测非方形网格。

F02 使用已知局部坐标平面 Z=ax+by+c，期望 slope=atan(sqrt(a²+b²))；测试平地、非方形像元及缺值边界。将输入同时平移一个常数高程，坡度不变。

### G02

域和拓扑必须逐像元/逐边一致；距离与人工小图坡度通过预定双精度容差（建议相对 1e-12、坡度绝对 1e-8 度）；正负全局偏移的四色规则正确。兼容坡度的未验证状态必须独立记录。

## M03：边界区间、权重和初始化

### 实现

`boundary.py`、`initialization.py`：

```python
collect_boundary_samples(...) -> SampleStats
build_boundary_intervals(...) -> BoundaryData
assess_component_support(...) -> component_support_table
initialize_surface(...) -> initial_S
```

只在 boundary 像元遍历小窗口。湿侧样本必须属于中心所在分量；干侧样本必须属于 dry，且与该分量有八邻接关系。保持源代码的方形窗口，不改为法线抽样，不混入其他分量湿样本。[R1–R2]

对每个中心样本集合独立计算：

```text
median = median(z)
MAD = median(abs(z - median))
sigma_w = max(0.25, 1.4826*MAD_w)
sigma_d = max(0.25, 1.4826*MAD_d)
L = median_w - sigma_w
U = median_d + sigma_d
mid = (L+U)/2
```

偶数样本中位数处理需与参考一致。不能用均值替代中位数、标准差替代 MAD，或对相邻像元的各自残差再做一次中值滤波来冒充本中心 MAD。

初始权重 beta0 按 `boundary_terms` 完整迁移：样本数、坡度、离散度、湿干高程顺序因子。随后固定 beta0>=highWeight 的候选，用同分量 peerRadius 邻域中、排除中心的候选中点计算 peer 统计，再生成 beta。peer 数不足 3 不启用该惩罚。不根据正在更新的 beta 原地反复改变 peer 集合。

任何无样本、无有效坡度或不可用统计均为无效约束；L>U 时 beta=0，不排序上下界强行修复。非活动项必须显式跳过或用安全占位，不能因 `0*NaN` 污染目标；保存诊断原因。

高权重点定义 beta>=0.5；eligible 仍按至少 3 点且 `max(xmax-xmin,ymax-ymin)>=2`。不额外引入周向覆盖门槛。[R1,R3]

对每个 eligible 分量：

```text
s0 = sum(beta*mid)/sum(beta)
initial_S[p] = max(s0, Z[p]+minDepth), p in hard
initial_S[p] = s0,                  p in soft
```

使用全部有效 beta 而不是只用高权重点计算 s0。支持不足分量置 status=0，不进入求解，也不降低门槛重试。

### 验证

将少量 7×7/11×11 夹具的样本坐标、样本数、中位数、MAD、L/U/mid 手工列出，逐项核对。另用独立直接枚举验证随机小图，避免让测试调用生产样本收集函数再验证自己。

构造 beta0/最终 beta 分歧，确保 peer 选择是两遍固定流程。测试邻近不同分量、反向区间、无干样本、无湿样本、坡度缺失和过少锚点。相隔超过局部依赖范围的分量之间，改变 B 的 DEM 不应影响 A；不能把共用背景样本的近邻场景错误地要求为完全独立。

F01 应得到 L=98.75、U=100.25、mid=99.5 m；通过支持判定的内部初始水面为 99.5 m。这一检查不要求不同边界位置 beta 完全相同。

### G03

样本坐标/计数/eligible 完全一致；固定小图中位数、MAD、区间及初值绝对差建议≤1e-10 m；权重绝对差≤1e-12。参考是固定输入及坡度下的统计，不把未校准的 GEE 坡度差异混入边界公式测试。

阶段快照包含域、拓扑、边界约束和 initial_S，后续可直接用它测试求解器。分量局部读入时使用至少 pairRadius+peerRadius+1 的原始上下文，并用整图与带上下文局部结果比较验证。

## M04：独立能量与六候选数值内核

### 实现

`energy.py`、`coordinate.py`、`diagnostics.py`：

```python
energy_terms(S, model, mu) -> raw_terms, normalized_terms
coordinate_minimum(v) -> new_value
fixed_point_residual(S, model, mu) -> residual_max
hard_violation(S, model) -> violation_max
```

只实现纯 Python/NumPy 参考版本，不进行 JIT 或并行。

定义 T=Z+minDepth、E 为每条无向边只计一次的集合：

```text
J0 = sum_E w_pq*(S_p-S_q)^2
   + lambdaB*sum_B beta*dist(S,[L,U])^2
   + lambdaT*sum_soft max(T-S,0)^2
Jm = sum_B beta*(S-mid)^2
Jmu = J0 + mu*Jm
Q = sum_E w_pq + lambdaB*sum_B beta + lambdaT*count(soft)
E0 = J0/Q
Em = Jm/Q
Etotal = (J0+mu*Jm)/Q
```

Q 与 mu 无关。若使用双向邻接存储，连续性项和对应归一化边权部分都乘 1/2；单像元 degree 则为该点全部邻边之和。hard 区域是约束 S>=T，不是额外软罚项。[R1–R2]

单像元更新中：

```text
degree = sum_q w_pq
neighborSum = sum_q w_pq*S_q
boundary = lambdaB*beta
soft = lambdaT*is_soft
midWeight = mu*beta
```

六候选顺序、可行区间、最小分母保护和严格小于的平局选择规则保留原代码。不能把 soft 与 hard 重复计入，也不能把 midWeight 加到全域。

### 验证

先用 2–4 节点手算图检查 J0/Jm/Q，防止二倍因子错误。独立能量函数使用边列表直接求和，不复用生产 `energy_terms`。

至少 1,000 个随机单像元问题：分别覆盖三段区间、地形上下、hard、beta=0、零宽区间及候选端点。对照冻结的原始标量函数及 SciPy 独立有界最小化。参考搜索范围必须覆盖可行极小值，不将任意过窄范围视为真值。

小型完整图使用独立目标和解析梯度交给另一求解器。主目标非唯一时比较能量、约束和固定点残差，不要求任意参考优化器返回相同 S；需要逐像元比较时采用构造的唯一解问题。

### G04

标量目标差建议≤1e-8*(1+abs(E_ref))；与冻结标量函数的普通有限测试差应达到约 1e-10 量级并按输入幅度使用组合容差。hard 约束不得违规。全图能量/归一化通过手算和独立计算；非活动缺值不产生 NaN；不能仅靠与另一份相同代码一致而判定正确。

## M05：完整四色求解与主/次目标状态机

### 实现

`solver.py`、`state.py`：

```python
sweep_four_colors(S, model, mu, active) -> S
solve_component(prepared, params) -> SolverResult
advance_component(state, diagnostics, params) -> state, actions
final_audit(result, prepared, params) -> audit
```

一轮 sweep 依次更新 0、1、2、3 色。某色更新读取前面各色已经更新的值，同色像元没有邻接，可暂按顺序处理。固定点残差通过“当前 S 下重新计算一次各坐标最小值”获得，不用上一轮最大改变量代替。

每 10 次 sweep 检查一次，首次阶段前初始化 prev_primary/prev_total。生产规则完整沿用：有限性、hard 违规≤1e-9 m、总目标单调容差1e-10、主/总目标相对变化≤1e-6、固定点残差≤0.001 m、连续至少2次检查稳定。相对目标变化分母保留 objectiveFloor=1e-8。

### 状态合同

| status | 含义与动作 |
|---|---|
| 0 | 支持不足，不求解 |
| 1 | 求解 J0，mu=0 |
| 2 | 求解 J0+mu*Jm |
| 3 | 次目标收敛且主目标增量在预算内，接受 |
| 4 | 所有次目标尝试未通过，恢复已收敛主目标解并接受 |
| 5 | 主目标失败，不发布该分量深度 |

主目标通过后保存独立的 baseS 副本及 E0_base、Em_base，不能使 baseS 与当前 S 共享可变内存。依次尝试 mu=lambdaB*[1e-3,1e-4,1e-5,1e-6]。每次尝试最多2000 sweep，不是所有尝试共享2000总次数。[R1,R3]

次目标接受条件是已经收敛且：

```text
E0_candidate - E0_base <= max(1e-4*E0_base, 1e-8)
```

次目标中允许 J0 在预算内上升，不能强制 J0 每次扫描都下降；同一次尝试需检查的是固定 mu 下 Jmu 的单调性。尝试失败时恢复 baseS，重置 sweeps/stable，用新 mu 和 baseS 重新建立 prev_total，不能比较不同 mu 的总目标。

全部次目标失败后 status=4、S=baseS，最终审计 mu=0。status=3 最终审计使用被接受的 mu。status=4 不是 DEM+0.01 m 的无锚点兜底。

### 验证

固定 M03 快照，对比原标量内核驱动的参考四色扫描与本地扫描，保存第1轮、第10轮和阶段结束水面。测试奇数偏移的分量包围盒，防止重新从(0,0)配色。

独立状态机测试必须强制经过：主目标成功、主目标耗尽/非有限失败、第一次次目标成功、预算拒绝后重试、未收敛后重试、四次拒绝恢复、终止状态不再更新。强制分支可用测试专用小预算或构造诊断，不改变生产参数。

固定 prepared 后，改变分量 B 的 S 不影响 A 的邻居和能量；检查所有成功分量的有限性、约束、残差和预算。F01 最终水面99.5 m、水深0.5 m。

### G05

所有状态路径及动作通过；同初值同顺序的10轮小图水面最大差≤1e-6 m，归一化能量差建议≤1e-8*(1+abs(E_ref))；固定点/约束满足生产阈值。F01 深度误差≤1e-7 m。阈值临界案例单列，不因浮点差异删像元或更换状态标签。

已有 GEE 小图快照时可做额外 compat 验证。没有快照时继续独立本地开发，报告 GEE_EQUIVALENCE=PENDING；不能把原标量对照叫作真实服务端验证。

## M06：深度、梯度、质量标记与产品审计

### 实现

`products.py` 和 `diagnostics.py`：

```python
build_products(result, inputs, prepared, params) -> ProductBundle
audit_products(bundle, prepared, result) -> AuditReport
```

成功域只由 eligible、status=3/4、有限性和最终审计决定，不由正深度阈值决定：

```text
solved_valid = eligible & (status in {3,4}) & finite(S) & component_audit_passed
signed_depth = S-Z
solved_depth = max(S-Z, 0)                  # solved_valid 全域
legacy_valid = solved_valid & (S-Z>0.01)   # 旧产品域
```

不得用后处理修改 S 来强行消除 soft 区负深度；保留 signed_depth 诊断。支持不足/主目标失败仍是 NoData，不补成0；已求解零深度必须保留为有效0。

### 产品合同

| 文件 | 规则 |
|---|---|
| `CFDepth_WSE.tif` | 完整 solved 域 S |
| `CFDepth_depth_solved.tif` | 完整 solved 域 max(S-Z,0) |
| `CFDepth.tif` | 保持旧的 >0.01 m FP64 发布规则 |
| `CFDepth_depth_signed.tif` | 可选，完整 solved 域原始 S-Z |
| `CFDepth_WSE_gradient_solved.tif` | 基于完整 solved 域，m/m |
| `CFDepth_WSE_gradient_legacy.tif` | compat 可选，基于旧正水深域 |
| `CFDepth_gradient_directions.tif` | bit0=x方向有效，bit1=y方向有效 |
| `CFDepth_status.tif` | support 内0–5；域外NoData=-1，不以0作NoData |
| `CFDepth_QA.tif` | 以下独立位定义 |
| `components.csv` | 分量状态、原因、数量、目标、残差、mu、sweeps、耗时 |
| `run.json` | 完整可追踪配置、输入、网格、版本、审计及文件哈希 |

建议首版 QA UInt16 固定位：1=mask无效；2=DEM无效；4=支持不足；8=主目标失败；16=成功求解；32=仅接受主目标；64=非负且≤minDepth；128=signed_depth<0；256=仅一个梯度方向可用；512=hard约束活跃。合法背景值为0；成功与失败位不得同时置位。像元在输入矩形内的无效信息以QA记录，不强制掩膜掉；NoData保留值65535。

梯度沿x/y使用相应距离：双侧有效时中心差分，只有一侧时单边差分，均无时该方向无效。完整梯度域与旧产品域分别计算，不混用掩膜。只有一个方向有效时是可观测幅值，方向位必须保留。[R1–R2]

最终写Float32之前，先在FP64中固定legacy_valid；读回不重复阈值化。NoData=-9999仅用于磁盘编码，不能进入统计或邻居求和。

### 验证与 G06

人工设定 status 和 signed_depth={-0.2,0,0.005,0.01,0.02}，核对成功域、两个深度产品、QA及旧发布掩膜。构造FP64略高于0.01但Float32舍入到阈值的案例，掩膜仍保持FP64决定的结果。

用人工WSE平面S=ax+by+c检验解析梯度sqrt(a²+b²)，另测单边和孔洞；不要以完整CFDepth是否重建某个人工斜面作为梯度函数测试。常水平面梯度为0。

验收要求每个support像元有明确状态，成功/失败/不足计数加和一致；S、depth和梯度都不能泄漏到不应有效的区域。未完成status1/2不能伪装成最终产品。

## M07：运行入口、恢复、写出与 ArcPy 适配

### 实现

`pipeline.py`、`cli.py`、`checkpoint.py`、`io_arcpy.py`。按M01→M02→M03→M04/M05→M06串联，不在pipeline内重写边界或求解公式。

提供check、prepare、solve、run、audit五种入口。prepare保存可重复使用的固定中间量；solve仅接收经过哈希核对的prepared。运行目录包含resolved_config、input_manifest、grid、prepared、checkpoint、components、产品、审计和日志。

恢复只在完整检查阶段保存：S、baseS、所有分量状态、基线/本地算法版本、配置哈希、输入哈希、prepared哈希、全局索引。采用临时文件+校验+原子发布；中断恢复不从部分写出的状态猜测。输入、网格或数值配置不同必须拒绝恢复。

禁止覆盖不属于当前run的输出，禁止自动删除源数据。不需要每10轮写GeoTIFF；可按时间间隔和阶段转换保存检查点，最终才发布产品。

ArcPy只实现与M01相同的读写/运行包装。RasterToNumPyArray与NumPyArrayToRaster需显式维护lower-left、横纵像元尺寸、空间参考和NoData，进行真实往返验证。[R6–R7] 不把ArcPy默认Slope、FocalStatistics、CostAllocation引入求解核心。

### 验证

从F01/F09的TIFF跑完整流程、读回全部产品并审计。分别在prepared完成、主目标中途、中点尝试中途、恢复baseS之后、产品写出之前中断，然后恢复。与未中断运行比较S、mask、status、attempt、sweeps、目标和输出。

篡改输入、参数或prepared哈希，必须拒绝恢复。写出异常不得留下完成标志。对ArcPy进行非方形栅格、非方形像元、孔洞、负高程和合法零值的真机读写；相同数组核心返回相同数值结果。

### G07

完整TIFF流水线可复现；相同环境/后端下恢复结果与连续运行应一致；若存在浮点差异，使用预定数值标准并调查，不宣称逐比特一致。输出网格和mask完全一致。

ArcPy环境未实际执行的测试明确SKIPPED；Rasterio交付可通过主线验收，但不能写“ArcPy已验证”。

## M08：加速与最终回归

### 实现顺序

先对已经通过G07的串行程序分段计时，记录读入、标签、坡度、边界、主目标、次目标、统计和写出，记录峰值RSS。

依次优化：边界循环→六候选/四色循环→同色像元并行→分量间并行。每一项优化后重跑全部回归，保留Python参考后端和固定比较数据。Numba首版fastmath=False；总目标统计优先保留确定性顺序，避免并行归约改变阈值附近状态。Numba支持编译循环与并行，但性能应由实际profile判断。[R8]

一个分量只存一组状态标量，不铺成8个全图Float64波段。初版优先分量包围盒数组；必要时再引入压缩索引或外存，均另加等价测试。禁止为生产建立N×N稠密矩阵。

一个大连通分量不能分成互不通信的块独立求解后拼接。初版超过内存预算可明确报错并列出分量规模；支持更大场景时再实现外存或带边界交换和全局审计的区域分解。局部halo只保证边界统计上下文，不截断全局优化依赖。

### 验证

所有夹具以纯Python、Numba串行、Numba并行逐阶段比较；额外用不同线程数重复运行。第一次JIT时间与预热后时间分开报告；不在实施前承诺加速倍数。

真实数据先选完整小场景或选定AOI、再中场景、最后整景。对照时本地与参考必须是同一AOI/同一prepared，不能拿独立裁片解与整景解的同位置值要求一致。

真实场景先验收工程正确性：无静默域变化、全部分量可解释、成功域审计通过、失败明确标识、耗时内存可记录。RMSE是否提升属于后续独立精度实验，不能用参考水深调参数来获得“迁移通过”。

### G08

各加速后端通过相同数值、域和状态机测试；峰值内存和分阶段性能有真实记录；命令行从TIFF到产品可用；中断恢复成功；不要求100%正深度覆盖。尚未实现的超大分量处理/ArcPy/GEE兼容特性按实际状态报告。

## 4. 命令行与逐阶段执行合同

以下命令是待开发接口。只有相应模块实现后才执行，不得把本文当作现有命令说明书。

```bash
# M00 创建包后，从仓库根目录安装
python -m pip install -e ./CFDepth_local

# 每阶段先当前测试，再累计回归；具体测试文件随阶段建立
python -m pytest CFDepth_local/tests/test_m00_*.py -q
python -m pytest CFDepth_local/tests/test_m01_*.py -q
# ... 依次直到 test_m08_*.py
python -m pytest CFDepth_local/tests -q

# M07 完成后
python -m cfdepth_local check --config CFDepth_local/configs/projected_metric.json
python -m cfdepth_local prepare --config CFDepth_local/configs/projected_metric.json
python -m cfdepth_local solve --run-dir CFDepth_local/runs/example_001
python -m cfdepth_local run --config CFDepth_local/configs/projected_metric.json
python -m cfdepth_local run --config CFDepth_local/configs/projected_metric.json --resume
python -m cfdepth_local audit --run-dir CFDepth_local/runs/example_001
```

PowerShell等环境不依赖shell通配符时，可显式列出测试文件，或使用预先注册的m00/m01等pytest marker；测试文件名和命令须在各阶段报告中记录。

示意配置：

```json
{
  "run_id": "example_001",
  "inputs": {
    "mask": "data/flood_mask.tif",
    "dem": "data/dem.tif",
    "require_aligned": true,
    "vertical_unit": "m"
  },
  "spatial_profile": "projected_metric_v1",
  "slope_method": "central4_metric_v1",
  "algorithm_profile": "v3.2.1_frozen",
  "io_backend": "rasterio",
  "solver_backend": "python_reference",
  "precision": "float64",
  "output_dir": "CFDepth_local/runs/example_001",
  "write_full_solved_depth": true,
  "write_legacy_positive_depth": true,
  "write_qa": true
}
```

v3.2.1_frozen解析为完整冻结配置，不能只载入示例列出的几项。M08通过后才允许将solver_backend改为numba_cd。输入范围的正确性是本任务前提，不增加mask_semantics开关或额外水体数据需求。

## 5. 交给编码助手的每轮任务约束

```text
只执行当前阶段 Mxx。先读取本方案、冻结源码和上游通过报告，核对依赖。
限定修改 CFDepth_local 内当前模块、直接相关测试及必要的合同文件。
不改旧 CFDepth/CFDepth_geemap，不切换算法，不调生产数值参数。
实现后先运行当前模块测试，再运行截至当前阶段的全部回归。
对缺陷给出最小复现；不得以删测试、改期望、放宽容差或改变有效域规避。
报告变更文件、接口、测试命令、通过/失败/跳过项、最大差值和残留问题。
仅当前门禁通过且报告已保存后继续下一阶段；否则停止在当前阶段修复。
不得把文档、mock、原仓库记录或未执行命令标为本地实测通过。
```

最终交付应包括生产包、冻结数值配置、所有阶段报告、完整测试夹具、命令行、产品合同、实测环境锁定和性能记录。构建成功意味着正确执行所定义算法；真实水深精度仍由后续独立实验评价。

## 6. 来源与证据边界

[R1] 上一轮《CFDepth_新版审查与本地重构技术方案.md》，尤其第2、6、9、10节：冻结的目标、状态机、本地输入/输出合同与验收建议。

[R2] 固定提交中的CFDepth_geemap/cfdepth/numerics.py、solver.py、components.py、grid.py：算法来源。M00必须实际取完整源码而非只依赖本文公式。

[R3] 固定提交中的CFDepth_geemap/cfdepth/defaults.json：数值默认值来源。

[R4] SciPy ndimage.label官方文档：默认四连通与显式八连通结构。

[R5] Earth Engine ee.Terrain.slope官方文档：度单位、四邻居和边缘缺值；不提供足以替代小图对照的完整跨平台等价保证。

[R6–R7] ArcGIS Pro RasterToNumPyArray/NumPyArrayToRaster官方文档：数组转换、原点、横纵分辨率、NoData。

[R8] Numba Performance Tips官方文档：编译循环、fastmath、并行及profile原则。

```text
https://github.com/yukun80/paper6_CD/tree/a88f1dd7897fc479e85ee2f6191506017a920457/CFDepth_geemap
https://github.com/yukun80/paper6_CD/blob/a88f1dd7897fc479e85ee2f6191506017a920457/CFDepth_geemap/cfdepth/numerics.py
https://github.com/yukun80/paper6_CD/blob/a88f1dd7897fc479e85ee2f6191506017a920457/CFDepth_geemap/cfdepth/defaults.json
https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.label.html
https://developers.google.com/earth-engine/apidocs/ee-terrain-slope
https://doc.esri.com/en/arcgis-pro/latest/arcpy/functions/rastertonumpyarray-function.html
https://doc.esri.com/en/arcgis-pro/latest/arcpy/functions/numpyarraytoraster-function.html
https://numba.readthedocs.io/en/stable/user/performance-tips.html
```

本文新增的模块划分、接口名、文件名、QA位、测试夹具、命令与工程容差都是实施设计，不是对上游现有功能的声称。当前任务只完成方案编制，M00–M08仍待实际实现与验收。
