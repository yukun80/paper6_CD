# CFDepth v3.2.1：GEE 分阶段水面求解

入口 `CFDepth_0919.txt` 可整份复制到 Earth Engine Code Editor，顶部仍只需在
`ee.Image(image4)` 中更换导入变量。生产只使用现有洪水范围和FABDEM。
本次优化保留软边界区间、分量隔离、四色下降、中点预算、正式10次扫描/阶段、
2000次上限、收敛容差、NoData和两个最终产品，不启动云端任务或覆盖历史结果。

## 小数掩膜修复（v3.2.1）

v3.2的quick组真实日志为4 PASS、1 FAIL、0 SKIP：字段a分组统计为
13.184313725490195，原始参考为11.28627450980392。已确认分组路径额外执行
`updateMask(ids.gt(0))`，而参考没有；本地适配器误用min模拟该调用，未暴露小数权重丢失。
本次代码修复不等于已经在真实GEE解释并消除了全部差值。

`groupedInput`将原始逐波段掩膜乘以分量准入值，再显式更新掩膜；不恢复原无效像元。
准入值由满掩膜常量底图生成，合法正整数编号为1，缺失、非正数、非整数及≥2^53为0。
不单靠`unmask`假定原cid的小数掩膜已经变成满掩膜。cid仅作为分组键，不替换数据权重。
保留sum/min/max的顺序修复、权重语义及原1e-9字段对照容差。

quick组中的mixed_reducers现在同时检查：

- 对齐4×2网格的两个分量、0/0.4/1掩膜、无效数据与被排除的背景；逐波段对照前后掩膜。
- 原来的非对齐区域和原始参考输入；不调用生产helper生成参考，不手工推算边界覆盖权重。
- 一次取回所有字段、独立参考和有限样本，打印后汇总全部失败；legacy_a只作历史诊断。
- 0.4掩膜用独立GEE常量掩膜作为表示对照，避免混淆服务端掩膜表示与JS字面量。
  不放宽分组与独立参考间的数值容差，也不将背景或null当作有效编号。

其他updateMask调用已按用途审查：分量ID、broadcast、邻接边、边界采样及最终资格用于
二值成员筛选；边界相容性权重保存在weight波段，不用mask代替。本轮只修复已确认的
分组权重路径，不全局重写prepare、energy或gradient的掩膜。
本地适配器现支持单/多波段更新、原零掩膜保持无效、新小数掩膜及掩膜波段数错误；
它不模拟真实clip、重投影、边界量化或整个GEE服务端。
接口依据：[updateMask](https://developers.google.com/earth-engine/apidocs/ee-image-updatemask)。

## 边界极值参考修复（仍为v3.2.1）

后续quick日志只有`non_aligned peak: 37 vs 27`失败，其余字段与掩膜检查未报告差异。
第4列值为37，和区域[0.2,0.2,3.2,2.2]相交，但中心3.5在外；独立max参考按像元中心
排除了它，混合归约实际包含了它。本轮只修正探针参考范围，生产入口、版本、运行名均不变。

sum继续在原区域统计；min/max参考在包围原区域的网格对齐窗口内读取原有非零掩膜像元，
不重新放开无效区域、不复用生产分组helper生成参考。另采集第4列和外侧第5列各一个像元，
检查37仍有效且掩膜未改变、外侧仍无效。所有字段和诊断先打印再断言；不放宽原容差。
本地回归证明：即使把分组值与参考同时写成27，独立边界像元断言仍拒绝通过。
依据：[GEE区域像元入选规则](https://developers.google.com/earth-engine/guides/reducers_reduce_region)。

测试目录仅保留一个可直接复制的GEE入口`tests/gee_acceptance_probe.js`，生成器也只输出该文件。
两个旧独立探针及其运行脚本已删除；其中独有的米制投影、418像元小场景BFS检查已迁入
geometry和topology组。共享场景函数、本地测试、数组适配器、SciPy和模板对照仍被使用，保留。
大场景继续通过主入口`fixture:'large'`执行分阶段验收。本轮没有启动云端任务。

## 本轮修改与性能依据

- 分组归约将所有加权sum放在min/max之前，同步重排输入波段；保留字段名和原权重语义。
  `diagnostics`原来的sum/max/sum/min组合违反GEE接口限制，不能靠全部改为unweighted规避。
  依据：[官方归约说明](https://developers.google.com/earth-engine/guides/reducers_intro)。
- 单像元最小化的六个区间、分母及固定分子项只为当前mu构建一次；改变mu时重新构建。
  保留候选顺序、加法顺序及严格较优比较，不引入最低水位偏好或近似解。
- 静态平移按影像对象、偏移和网格上下文缓存；同时复用坐标、四色、纬度和距离表达式。
  动态水面不进入静态缓存，必要的reproject仍保留，不增加资产缓存波段。
- components的独立客户端查询从原8次合并为3次；默认跳过旧加权计数和普通含null去重两项
  历史排错统计。成员、有效编号、编号集合与数值有效性仍全部检查。
  `diagnosticDetails:true`可开启详细统计。此开关不进入数值参数签名。
- 资产元数据按本次运行缓存；诊断、准备验收和单轮扫描中可并行取回的值用Dictionary合并请求。
  合并客户端请求不代表服务器只做一次归约，更不能直接推断等比例加速。
- 保留components阶段的一次矢量化，其余阶段只用栅格编号。当前没有逐阶段实测能证明
  矢量化是主要瓶颈；[connectedComponents](https://developers.google.com/earth-engine/apidocs/ee-image-connectedcomponents)
  会屏蔽超过maxSize的对象，不能直接替代大分量处理。

本地固定mu的40次最小化（10轮四色扫描）运算构造调用：**11,440 → 6,941，减少39.33%**。
复现：`node CFDepth/tests/benchmark_templates.js`。这是数学内核表达式构造计数，
不包含所有影像运算、序列化或服务器优化，不是GEE运行时间或水深精度的测量结果。
生产打印的`Stage setup elapsed ms`包含客户端及同步查询，**不含导出任务完成时间**。
请另记录Tasks中components、prepare及相同步数iterate的开始/完成时间与成功状态。
只有实测表明components占主要耗时，再讨论分块栅格连通与跨块合并。

## 分组GEE验收

将 `tests/gee_acceptance_probe.js` 整份复制到空白Code Editor。在文件最上方设置：

```javascript
var PROBE_GROUP = 'quick';
```

| 组 | 检查 | 预计选中项数 |
|---|---|---:|
| quick（默认） | 三种网格整数索引、混合归约、5编号与空值 | 5 |
| geometry | 三种网格物理中心、投影与二值标签（含历史米制网格） | 9 |
| topology | 孔洞、对角连接、独立分量、相对编号、418像元小场景 | 7 |
| prepare | 小场景实际components和生产prepare、边界与初始化 | 4 |
| solver | 显式小图、四色、非平衡单轮扫描、诊断和输出资格 | 5 |
| gradient | 两种纬度解析斜面、单方向、孤立像元与NoData | 2 |
| assets | 已保存small阶段资产的合同、来源、状态及可用时最终产品 | 1 |

每次只执行选中组及其列出的前置项。未选项为NOT_RUN，不算失败；真实依赖失败为SKIP。
每项记录`elapsed_ms`（包含客户端、网络与服务端等待）。仅打印`PASS GROUP <组名>`，
不宣称全流程通过。资源超限仍是失败，不自动缩小分辨率或放松判定。

依次运行quick、geometry、topology、prepare、solver、gradient。
solver显式构造两个小分量及软区间，**不调用prepare、不展开多轮iterate**；
production prepare由prepare组独立覆盖。这避免原先准备加四轮迭代的长未落盘计算链。
坐标取floor后转Int64；物理中心用`ee.ErrorMargin(0, 'meters')`，不改生产几何容差。

然后在主入口以新runId（例如`cfdepth_v321_small`）、`fixture:'small'`、实际assetRoot
及其余默认数值参数运行components→prepare→iterate1→iterate2及后续，每个任务COMPLETED
后再进入下一阶段。不要修改正式sweepsPerStage来匹配旧探针。

在独立探针选择assets并设置顶部：

```javascript
var PROBE_GROUP = 'assets';
var PROBE_ASSETS = {
  assetRoot: 'projects/你的项目/assets',
  runId: 'cfdepth_v321_small',
  step: 2 // 实际已经完成的状态资产序号，至少2；结束后改为最终序号重跑
};
```

assets只读取components、prepared、state_00001和所选状态，不创建任务、不重新展开迭代。
核对保存/恢复波段、网格、区域、来源、版本、参数、快照令牌及状态值。
若还有status1/2，明确说明最终产品尚未验收；继续通过主入口求解。
求解结束后重跑assets会检查415个深度像元、0.5m水深及常水面零梯度，
随后主入口final仍独立执行残差/约束/预算门槛并生成两个导出任务。
assets组可读合同通过不等于此前每次任务均经过验证；任务历史也须留存。

## 验收记录（2026-09-19）

| 证据 | 状态 |
|---|---|
| 用户提供的v3.1真实GEE日志 | 19 PASS、2 FAIL（混合归约）、1 SKIP；此前坐标/分量/prepare/实际扫描已通过 |
| v3.2 quick真实GEE日志 | 4 PASS、1 FAIL（分组a权重差异）、0 SKIP |
| v3.2.1边界修复前quick真实GEE日志 | 4 PASS、1 FAIL（peak 37 vs 27）、0 SKIP |
| 本轮本地86组测试 | 通过；配置16、投影7、分量15、数值13、共享路径9、几何/调度5、优化8、混合探针断言5、阶段网格8 |
| v3.2.1语法、生成文件同步、SciPy对照 | 通过；独立目标差1.78e-15，最大水面差8.44e-9m |
| v3.2.1分组真实GEE验收 | 用户报告非assets组均通过；本轮更新后的assets组仍待实际保存/恢复验收 |
| v3.2.1小/大场景资产保存、恢复、最终产品 | 用户small components已保存，实际DEM网格正确但数组属性缺失；本轮兼容恢复待云端验证，大场景及最终产品未验收 |
| 同输入逐阶段性能与美国生产场景 | 待实测；仍要求259026支持像元、5个有效编号及集合一致 |

新增测试复现旧混合归约异常，检查部分掩膜/非对齐边界下字段映射与加权语义；
2000组约束及邻近浮点值与固定v3.1内核逐值一致；另保留1500组独立导数求根、SciPy小图、
中点接受/减权/回退、静态缓存隔离、mu变更重建、状态与NoData回归。
这些都是功能和数值证据，不代表遥感水深精度改善。未修改历史图、论文、深度产品、
`CFDepth_inference`、`CFDepth_WSE_260914.txt`或`CFDepth_0826 copy.txt`。

## 运行步骤

先在 GEE 上方 Imports 导入洪水影像，再修改文件顶部的 `getFloodInput()` 选择项。
按当前 Imports，`image` 为广西、`image2` 为郑州、`image3` 为涿州、`image4` 为美国；
这些变量名以你自己的 Imports 为准。默认选择 `image4`，不会自动猜测或切换地区。
例如选择郑州时，只修改函数内的一处导入变量名：

```javascript
function getFloodInput() {
  return ee.Image(image2);
}
```

**洪水输入是导入的影像对象，不是文件夹。** `CONFIG.assetRoot` 是另一项配置，
专门保存分阶段计算的临时状态；不能填 `image4`、`ee.String` 或列表。
默认 `assetRoot: ''` 自动使用导入影像的父目录，不必新建文件夹或填写项目占位符。
例如 `projects/yukun80/assets/USA_Brazos_River_flood_mask` 将使用
`projects/yukun80/assets`，其下按 `<runId>_components`、`<runId>_prepared`、
`<runId>_state_00001` 等名称保存临时状态。支持项目资产和旧式 `users/...` 路径。

也可显式填写已有目录，例如 `assetRoot: 'projects/yukun80/assets/CFDepth_runs'`；
该目录需自行预先创建。计算影像没有资产 ID 或运行合成场景时，必须显式填写目录；
`inputRevision` 不能代替保存位置。不同实验/地区请使用不同 `runId`，避免名称冲突。
恢复同一v3.2.1运行时保留原目录及 `runId`；自动模式只有在父目录与原目录一致时才读取同一批状态。
**不能恢复v3.0/v3.1/v3.2阶段资产**：此次修复分组统计权重，必须以新runId从components重建。
默认运行名为 `cfdepth_v321_run01`；旧资产保留，不迁移、不删除。
脚本核对输入和普通文件夹的元数据，兼容 Code Editor 返回的 `Image/Folder`
以及 REST 格式的 `IMAGE/FOLDER`；其他类型及缺失类型仍拒绝，并显示路径、实际类型和预期类型。
对于 `projects/<项目>/assets` 项目根目录，使用 `ee.data.listAssets(root, {pageSize: 1})`
检查可访问性，不要求它返回普通文件夹元数据；空目录正常通过。
脚本打印本阶段路径及下一步操作。
**目录可读不代表可写**：权限、配额和导出错误仍由 GEE 显示，不自动改用其他位置。

配置输入与运行标识：

```javascript
assetRoot: '',               // 自动保存到导入影像父目录；无需填写占位路径
runId: 'zhengzhou_soft_v321_01', // 每次实验使用新名称
stage: 'components',
step: 1,
fixture: '',                 // 真实输入；使用顶部 getFloodInput() 选择的导入影像
inputRevision: '',           // 若 getFloodInput() 为无资产 ID 的计算表达式，填写明确修订号
exportName: 'CFDepth_Zhengzhou_v321',
exportFolder: 'FloodDepth'    // 最终两个产品的 Google Drive 文件夹，不是临时资产目录
```

若误将影像放入 `assetRoot`，入口会在任何字符串方法调用前给出明确提示；
不要按 GEE 通用提示改成 `ee.List(...).indexOf(...)`，那并不能修正配置含义。
真实运行缺少所选导入影像时会提前停止；`fixture: 'small'/'large'` 无需导入影像。
这里只修复入口配置，不代表云端任务已经通过。

`getFloodInput()` 的有效像元必须为 0/1。NoData 必须通过掩膜表示，不能把255当作1。
脚本在 FABDEM 原网格执行 max 聚合以保留原淹没支持；只有原输入完全可观测且非淹没的
背景才允许作为干侧样本。输入有效性缺口不视为干岸。
当前生产网格限定为无旋转、北向上的 EPSG:4326 FABDEM，其他网格明确报错。

| 阶段 | 配置 | 本阶段新建的临时资产 |
|---|---|---|
| 分量准备 | `stage:'components'` | `<runId>_components` |
| 区间、固定权重、初始化 | `stage:'prepare'` | `<runId>_prepared` |
| 第一阶段求解 | `stage:'iterate', step:1` | `<runId>_state_00001` |
| 后续阶段 | `stage:'iterate', step:2,3,...` | 对应的新状态资产 |
| 正式输出 | `stage:'final', step:N` | 仅创建两个 Drive 导出任务 |

每次运行后，在 Tasks 中手动启动导出，等到 **COMPLETED** 再执行下一阶段。
`prepared` 已包含第0阶段状态，不另建 `state_00000`。
不要同时运行依赖尚未完成的资产的阶段，也不要跳步或重命名状态资产。
相同目标资产已经存在时脚本停止，不覆盖或自动删除。

控制台每阶段显示分量状态数量：

| status | 含义 | 主深度输出 |
|---:|---|---|
| 0 | 边界支持不足 | NoData |
| 1 | 主目标仍在求解 | 禁止启动 final |
| 2 | 中点次级目标仍在求解 | 禁止启动 final |
| 3 | 次级目标收敛且通过主目标预算 | 允许 |
| 4 | 所有次级尝试失败，保留收敛主解 | 允许 |
| 5 | 主目标未收敛或数值/约束失败 | NoData |

控制台不再出现状态1或2、且最新状态资产任务已完成后，设置 `stage:'final'`、
`step` 为该资产编号。final 会重新检查残差、硬约束和主目标预算，
然后创建主水深及梯度两个任务。若状态尚在求解，不会输出“部分完成”的主产品。

保存与恢复共用 `stageBandNames` / `packStage`，类型不同，必需波段也不同：

| kind | 波段 |
|---|---|
| components | dem、cid、support、dry、hard |
| prepared | components波段，加lower、upper、mid、weight、eligible及全部state波段 |
| state | S、baseS、status、attempt、sweeps、stable、prev_primary、prev_total、base_primary、base_mid |

state保持精简，DEM和边界从components/prepared快照读取；不再要求state含dem。
旧版本、未知类型、错误阶段、网格或快照令牌不一致均停止。

阶段资产保存 double 水面、分量ID与状态。分量ID在生成时为int64，保存为double之前
验证小于 `2^53`，避免 float32 舍入；资产固定原仿射网格和 `sample` 金字塔。
恢复检查版本、参数、运行名、阶段序号、实际波段网格、输入资产修订及准备资产令牌。

**阶段网格属性兼容修复（保留v3.2.1及现有运行名）**：新资产用字符串
`grid_transform_json: JSON.stringify(T)` 保存仿射变换；导出的 `crsTransform` 仍为六元素数组。
生产入口与assets探针使用同一恢复函数，优先解析JSON，同时兼容旧`grid_transform`数组；
两者均存在时必须完全一致。属性存在但格式错误、非法或与实际波段网格冲突，一律停止。
仅当两个属性都缺失且版本为v3.2.1时，才从实际dem波段（state为S波段）恢复内存变换。
仍核对必需波段、全部波段网格、声明CRS、来源、签名和快照；不修改云端属性或令牌。
small额外要求 `EPSG:4326`、`[1/3600,0,110,0,-1/3600,30]`；prepared/state必须与components网格及区域一致。

用户已保存的 `cfdepth_v321_small_components` 无需删除或重建。复制更新后的主入口，保持：

```javascript
assetRoot: 'projects/yukun80/assets',
runId: 'cfdepth_v321_small',
fixture: 'small',
stage: 'prepare',
step: 1,
```

运行后在Tasks启动prepare任务，完成后再依次执行iterate step=1、step=2，各步均等待COMPLETED。
然后复制更新后的统一探针执行assets组，填写同一目录、运行名及已完成的状态序号。
本地新增测试模拟数组属性丢失，覆盖新字符串、旧数组、缺失兼容、损坏拒绝及跨阶段校验；
这些测试不能代替上述真实资产恢复验收。本轮未启动任何云端任务。
初次准备与恢复使用同一个 `getFloodInput()` 的来源令牌；影像对象不进入配置参数签名。
计算表达式没有自动内容哈希；更改这类 `getFloodInput()` 必须更新 `inputRevision` 并使用新runId。
FABDEM 在 components 阶段固化，此后读快照，不混用后来的 DEM 集合版本。

临时资产不是最终候选深度或 QA 产品；不会自动清理。确认产品验收后可自行清理
对应运行前缀。组件矢量化和分组栅格统计仍可能受 GEE 资源限制：如果云端报错，
保留最后成功阶段，不能把任务失败解释为已收敛或静默降低分辨率。

## 算法与参数

### 边界相容性及区间

配对半径2像元；湿侧只采样同分量像元。干侧须为可观测背景并邻接同一分量。
每个中心窗口单独求中位数及 `median(abs(sample - window_median))`，
不使用“各邻居减去各自中位数”的替代MAD。

`sigma=max(0.25,1.4826*MAD)`，`lower=wet_median-sigma_wet`，
`upper=dry_median+sigma_dry`，`mid=(lower+upper)/2`。
上下界倒置时权重为0，不交换端点。区间是地形容差，不是有覆盖保证的统计置信区间。

初步权重为以下四项乘积：

1. `min(1, min(n_wet,n_dry)/3)`；
2. `1/(1+(slope_deg/5)^2)`；
3. `1/(1+(sigma_wet^2+sigma_dry^2)/(1 m)^2)`；
4. `1/(1+max(wet_median-dry_median,0)^2/(sigma_wet^2+sigma_dry^2))`。

半径3窗口中只参考同分量、初步权重≥0.5的其他候选，排除自身。
参考少于3点时第五项为1；否则以参考中点的median/MAD构造
`excess=max(abs(mid-peer_median)-3*max(0.25,1.4826*peer_MAD),0)`，
第五项为 `1/(1+(excess/(sigma_wet+sigma_dry))^2)`。
最终权重限制在[0,1]并在求解前冻结。这是地形相容性分数，不是语义可信概率。

每分量至少3个最终权重≥0.5的候选，且行/列跨度至少2像元才可求解。
保留8连通定义，原网格矢量化后以分量内最小唯一像元key回栅格；不使用
`connectedComponents(maxSize)` 隐式丢弃大型洪泛区。后续统计为分组栅格统计，
不反复对复杂多边形执行 `reduceRegions`。

回写使用显式属性 `component_id`，绘制前检查存在性、正整数、精确表示范围和唯一性。
绘制底图及结果均锁定DEM网格；保留未按support掩膜的绘制结果用于诊断，最终编号
仍仅由原support限定。检查用满掩膜0/1指示图及 `sum().unweighted()` 统计像元中心，
不再用默认加权sum的浮点数相等代替集合一致性；不改变目标函数或面积统计的归约方式。

`Component pixel diagnostics`默认只打印简短像元/分量统计；异常或`diagnosticDetails:true`时显示完整字段：

| 字段 | 含义与验收 |
|---|---|
| expected / restored | 原支持与最终编号覆盖的整数像元数 |
| missing / added | 逐像元丢失/新增，均必须为0；数量相等仍可能失败 |
| raw_spill | 未按support掩膜的绘制结果在support以外的像元数，仅用于定位几何回写，不据此扩展support |
| invalid_ids | 支持范围内非正整数或超出精确范围的编号，必须为0 |
| vector_count / distinct_ids | 矢量分量数与有效栅格非空唯一编号数，必须一致；后者使用countDistinctNonNull |
| distinct_ids_including_null | 普通countDistinct的观测结果，可包含null，仅作诊断，不固定减1 |
| missing_ids / unknown_ids | 缺失/未知编号与栅格像元数，均须为空；缺失编号在栅格中的像元数为0 |
| raster_id_pixels | 有效编号的非加权像元直方图，逐编号核对矢量集合 |
| raw_component_histogram | 服务端原始直方图，保留可能存在的字符串键`"null"` |
| histogram_null_count | `"null"`桶计数；未返回该桶时为0，仅作诊断，不作为分量或有效像元 |
| histogram_distinct_ids / histogram_pixels | 直方图编号数须等于非空去重数，像元总数须等于restored |
| unknown_id_pixels | 未在矢量集合中声明的有效编号像元数，必须为0 |
| min_id / max_id | 绘制属性的编号范围 |
| weighted_expected / weighted_restored | 旧加权统计方式的结果，仅供诊断，不作为验收门槛 |

若只有旧加权统计或普通去重计数不一致，而像元、非空计数及具体编号集合检查通过，
仅在详细模式打印提示并继续。正式计数只包含支持范围内的有效正整数，背景和null不作为分量。
不能仅因为5个矢量分量得到普通计数6就减1，仍须核对非空计数和具体编号集合。
直方图也可能返回`"null"`桶；只将这个精确键单独记录，不笼统忽略其他非数字键。
所有桶计数仍须为有限正整数且小于2^53；有效编号直方图的像元总数必须等于restored。
仅`{"null":12}`或空直方图不能代表5个有效编号，会因有效编号/像元检查失败而停止。
本地直方图模拟现在同时覆盖包含和不包含null桶的返回形式，不再假定它必然忽略空值。

统一探针quick组的6×2小场景首先独立采样全部12个网格位置，检查5个支持像元、
指定编号各出现一次，以及背景编号的掩膜。合成输入显式锁定原网格；采样使用
`dropNulls:false`保留空值，不用直方图验证自身。失败时打印行列、支持、编号及各自掩膜，
再根据实际数据区分输入/网格问题与归约问题。null桶计数不硬编码为7或12。
topology组继续使用独立BFS验证分量归属；该组预期7项PASS，不创建导出任务。

对当前美国日志，只有保持259026个支持像元、得到5个非空编号且集合完全一致才可通过；
这些观测值没有硬编码到通用算法中。如果非空计数仍是6，则继续按未知编号诊断定位。
真实缺失、增加或编号异常仍停止且不创建本阶段资产；Layers中添加默认隐藏的丢失、
新增、非法编号、未知编号图层，并打印最多10个异常像元的DEM行列和回写前后编号。
采样只限制打印记录数，不降采样栅格；异常范围很大时诊断查询也可能受GEE资源限制。
不做邻近填补、形态学扩张或分量合并。美国场景的实际原因须依据这些新诊断确认，
不能仅凭本地通过认定是权重差异或已经解决。

原版使用累积代价传播及固定50轮平滑，没有这些新增的分量编号与阶段恢复接口。
新版此前的JavaScript资产类型、WKT参数和归约空值适配存在实现缺陷；本地测试又未覆盖
真实服务端语义。旧版能运行不能证明新版检查正确，也不能证明旧版具备新版的收敛保证。
此次修复不回退算法、不删除检查；完整云端流程仍须单独验收。
接口依据：[countDistinctNonNull](https://developers.google.com/earth-engine/apidocs/ee-reducer-countdistinctnonnull)。

### 主目标与次级目标

主目标包含8邻接水面连续性、边界区间距离平方及边缘软地形下界三项。
`lambdaC=1, lambdaB=10, lambdaT=1`。图边权为
`lambdaC*(北向原像元长度/实际两像元距离)^2`，对称边仅计一次。
腐蚀1像元后的可观测内部施加 `S>=DEM+0.01` 硬下界；边缘以下界违反平方作软惩罚。
这种几何内部只是一种高支持度代理，不能证明其不存在检测误差。

分量内加权中点均值初始化，内部投影到下界。无依据分量不参加求解，
不会利用 `DEM+0.01` 回退制造主结果。
四色Gauss–Seidel坐标下降逐像元精确枚举分段二次函数的有效区间并取最小值；
边界中点永远不被强制恢复。

- 每阶段10次完整扫描，每次主/次级求解最多2000次；接受或失败的分量冻结。
- 最大坐标最优更新残差≤0.001m，并连续两阶段主/总目标相对变化≤1e-6。
- 同时检查有限值、硬下界及目标非增；数值单调容差为 `1e-10*max(1,previous_total)`。
- 用该分量固定的“图边权+区间权+软地形权”总和归一化目标。
- 中点项仅作用于边界，`mu/lambdaB` 依次为1e-3、1e-4、1e-5、1e-6。
- 主目标增量预算为 `max(1e-4*base_primary,1e-8 m²)`。
- 拒绝的次级尝试从保存的主解重新开始；全部失败则恢复主解。主解未收敛则NoData。
- 已接受的次级解在final阶段继续用其μ检验总目标残差，同时独立核对不含中点项的主目标预算。

这些是数值验收标准，不是毫米级水深准确度；参数尚未以真实参考水深验证。

### 主水深与梯度

主产品还要求深度严格大于0.01m。背景、输入无效、边界支持不足、主求解失败及非正深度
统一输出 `-9999`，GeoTIFF声明相同NoData。不单独输出候选深度或QA文件。
控制台分别显示支持面积、边界支持不足面积、求解失败面积和正水深面积。

梯度沿用260914的四邻域和中心/单侧差分。EPSG:4326网格采用球面局部距离：
`dx=6378137*cos(latitude)*abs(delta_lon_rad)`，`dy=6378137*abs(delta_lat_rad)`；
相邻点采用中间纬度，中心差分分母为两侧长度之和。不是统一nominalScale，也不调用
`Terrain.slope`计算WSE梯度。terrain slope仅保留用于边界权重，单位仍是度。

只在主深度有效范围内计算。两个轴均有效时平方和开方；一个轴有效时为该轴导数绝对值；
没有方向时NoData。masked方向不当作观测零值，控制台分别统计一个/两个方向的像元数。
单方向幅值不等于完整二维梯度；梯度较小只能表明当前诊断下更平滑，不能证明水深更准。
保存m/m，现有图11绘图乘1000转换成‰，不是百分比或角度转换。

## 验收方法

### 投影错误修复与接口检查记录

`Projection.transform()` 返回 WKT 字符串，不是 `crsTransform` 接受的数值列表。
入口现从 `raw.projection().getInfo()` 读取 CRS 和六元素仿射变换，按原始输入网格
联合统计 `raw_max`（是否有非0/1值）与 `raw_count`（有效像元数），不设置替代 scale，
不将输入强行改成经纬度投影。空有效域、非法编码、异常统计结果分别报错。
输入支持投影坐标和旋转；FABDEM计算网格仍限定为北向上、无旋转的EPSG:4326。

| 接口环节 | 本轮检查与处理 | 证据边界 |
|---|---|---|
| 输入投影与归约 | 六个有限数值、非零行列式，原网格/原掩膜统计 | 本地调用参数测试通过；真实归约待GEE小脚本 |
| DEM、重投影、区域归约、矢量化 | DEM读入时校验；所有后续调用沿用已校验的P/CRS/T，无WKT参数 | 静态检查；跨投影聚合、拓扑和资源需求待云端 |
| Image/Number与状态转换 | 栅格使用Image运算；分量属性显式转Number；条件通过对应适配器处理 | 共享标量数值测试通过；真实服务端掩膜语义待完整阶段 |
| 分组归约 | 加权sum波段稳定排序在前，min/max在后，cid最后；同步校验字段与输入数量 | 静态及本地检查；服务端字段与空分组待合成阶段 |
| 阶段恢复 | 校验元数据网格、必需波段及各波段实际网格相等；保留原签名和来源核对 | 本地异常/兼容测试通过；未写读真实资产 |
| 资产及GeoTIFF导出 | 导出前再次校验DEM网格；保持原NoData和两个最终输出 | 静态检查；未启动任务 |

分组统计的空结果仍采用已有空列表处理，不把空组伪造为有效分量。未在本地模拟
完整GEE掩膜、聚合器或矢量化服务；这些仍须使用下述小/大场景验证。
本轮未修改数值目标、松弛/收敛规则、边界权重或降分辨率策略。

接口依据：[Projection.transform](https://developers.google.com/earth-engine/apidocs/ee-projection-transform)、
[reduceRegion](https://developers.google.com/earth-engine/apidocs/ee-image-reduceregion)、
[Reducer.combine](https://developers.google.com/earth-engine/apidocs/ee-reducer-combine)、
[Reducer.group](https://developers.google.com/earth-engine/apidocs/ee-reducer-group)。

本地从仓库根目录运行：

```bash
node --check < CFDepth/CFDepth_0919.txt
node --check CFDepth/tests/test_config.js
node --check CFDepth/tests/test_projection.js
node --check CFDepth/tests/test_components.js
node --check CFDepth/tests/gee_acceptance_probe.js
python3 CFDepth/tests/sync_gee_probes.py --check
node CFDepth/tests/test_config.js
node CFDepth/tests/test_projection.js
node CFDepth/tests/test_components.js
node CFDepth/tests/test_pipeline.js
node CFDepth/tests/test_probe_geometry.js
node CFDepth/tests/test_numerics.js
node CFDepth/tests/test_optimization.js
node CFDepth/tests/test_mixed_probe.js
node CFDepth/tests/test_stage_grid.js
node CFDepth/tests/benchmark_templates.js
/home/yukun80/miniconda3/envs/hacqi/bin/python CFDepth/tests/check_scipy_reference.py
```

测试直接读取入口中的单像元精确最小化、边界权重、梯度和状态转换函数。
独立二分导数根验证1500组约束；小网格显式边列表与SciPy独立优化复核全局目标和解。
本地测试不模拟 GEE 的真实掩膜、reduceToVectors 或资产服务，不能替代下列验收。

统一入口的geometry组覆盖经纬度、历史米制仿射参数、旋转输入、无效掩膜、非法标签和全掩膜。
topology组覆盖孔洞、对角连通、独立斑块和418像元小场景，使用独立八连通BFS核对。
共享计算函数由生成器从生产入口同步；生成检查也验证该副本一致。
每组通过只说明该组通过，仍须完成其他组及实际资产流程。

GEE小合成场景：显式填写已有可写目录 `assetRoot`，设置新runId、`fixture:'small'`，
按上述完整阶段顺序运行；合成场景没有输入资产，不能自动推导保存目录。
场景包括孔洞、一像元间隔的独立斑块、对角连接的两像元斑块及孤立点。
components自动要求418个支持像元、4个分量；final要求415个主水深像元，水深0.5m，
孤立点和两像元分量为NoData。预计主/次级各两个稳定阶段，但以实际状态为准。

GEE大合成场景：同样显式填写 `assetRoot`，另用新runId、`fixture:'large'`；原网格344×344连通支持，
预期118336个有效水深像元、1个分量、水深0.5m。用于检查大分量不被尺寸门槛截断。
常水面梯度应为0（有效），不能把0梯度当NoData。

上述两例通过后，才对真实小区域执行；随后选一个真实大分量检查资源开销和收敛。
不使用降分辨率、放松容差或降低支持门槛掩盖失败。输入/参数变化必须新建runId。
实际场景的覆盖率、误差和GEE运行耗时需独立记录，本次不重绘历史论文图。

### 探针维护

`tests/gee_acceptance_probe.js` 是唯一可直接复制的生成入口，包含主入口的完整共享函数，
不包含生产运行入口。修改主入口或 `tests/probes/` 下独立测试场景后运行：

```bash
python3 CFDepth/tests/sync_gee_probes.py
python3 CFDepth/tests/sync_gee_probes.py --check
```

`--check` 只核对不改写；修改生成文件不会替代修改源场景。统一探针不依赖GEE模块发布。
