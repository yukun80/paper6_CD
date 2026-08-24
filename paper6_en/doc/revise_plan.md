**结论先行：论文核心研究值得继续发表，但当前版本不宜原样重投 Journal of Hydrology。** 本次拒稿不是单一实验缺失，而是三个问题叠加：论文身份仍偏计算机视觉；CFDepth 的证据等级不足以支持“水深估算方法”的强定位；部分结果和实验流程存在明显的可复现性风险。最优修改路径不是新建完整城市水动力模型或重新开展灾后实测，而是：

1. 先完成 Table 2 和原始预测结果审计；
2. 重构为“水文问题主导、AI 为观测手段”的论文；
3. 补充 HA-CQI 的机制性消融与稳健性实验；
4. 利用公开水动力基准完成一次受控水深验证；
5. 通过洪水掩膜扰动实验量化 extent-to-depth 误差传播。

# 一、论文当前状态分析

## 1. 当前论文核心贡献

当前稿件构建了一个由“范围识别—水深近似”组成的两阶段框架：

第一阶段 HA-CQI 从双时相 SAR 影像中识别新增淹没区域。HA 通过浅层统计协调、局部可变形对齐和 DINOv3 语义校准增强灾前—灾后特征可比性；CQI 将灾前状态、灾后状态、带方向差值和绝对差值组织为 pair tokens，再通过 learnable change queries 建模洪水相关变化；最终由 Mask2Former 预测洪水范围。

第二阶段 CFDepth 将 HA-CQI 输出的新增淹没支持面与 DEM 结合，在连通分量内部通过初始地形约束水面、外边界高程平滑和最近可达边界传播生成一阶水深近似。

当前结果显示，HA-CQI 在郑州和涿州案例上的 IoU 分别为 73.38% 和 86.23%；CFDepth 相比 FwDET 将有效正水深覆盖率从 74.65% 和 65.92% 提高到 97.83% 和 96.38%，并降低了重建水面高程场的局部梯度。稿件还使用 8 个现场照片参考点进行了水深区间一致性检查。

## 2. 当前创新定位

当前稿件在贡献列表中将创新概括为“统一 extent-to-depth 框架、HA-CQI、CFDepth 和两个真实事件验证”。这一表述在遥感算法论文中成立，但不足以应对 Journal of Hydrology 的审稿标准。

目前最有潜力、也最应强化的创新不是“组合了多少网络模块”，而是以下科学问题：

> **变化检测得到的是新增淹没支持面，而不是完整洪水期水面。永久水体和部分低高程连接区域被排除后，传统基于湿—干边界推断水位的方法会面临边界条件失配。**

这一问题比“设计了 HA、CQI、Mask2Former 和 CFDepth”更适合水文学期刊。建议把创新重新组织为三个层次：

1. **问题层创新**：明确提出 change-defined flood support 与 complete flood-period inundation support 之间的边界条件差异。
2. **观测层创新**：通过 HA-CQI 解决城市 SAR 中辐射统计偏移、残余错位和复杂散射引起的伪变化。
3. **推断层创新**：研究在不完整新增淹没支持面下，能够恢复何种程度的、时间特定的、DEM 约束一阶水深信息，以及其误差边界。

这样可以把论文从“多个成熟模块的组合应用”改写为“针对一种尚未被充分讨论的洪水观测边界条件设计的方法体系”。

## 3. 主要优势

第一，研究问题具有明确的应急需求。城市洪水期间通常只能快速获得遥感范围，而无法同步获取空间连续水深，这使 extent-to-depth 具有实际价值。

第二，论文不是单纯公开数据集测试，而是包含郑州 2021 和涿州 2023 两个 GF3 城市洪涝案例，且两个场景在建成区密度、河流邻近性和淹没形态方面存在差异。

第三，当前稿件已经意识到水深结果不能等价于水动力真值。Discussion 中将照片比较限定为区间一致性，将 valid depth ratio 和 WSE-gradient ECDF 定义为覆盖完整性和内部一致性诊断，而不是直接水深精度。这一方向是正确的，但摘要、章节命名、贡献表述和结论仍未完全贯彻这一证据等级。

第四，CFDepth 与 FwDET 使用相同洪水支持面和 DEM 输入，这种受控比较排除了上游范围差异，是一个合理的算法对比基础。

## 4. 可能导致再次拒稿的问题

### 4.1 Table 2 是当前最严重的结果可信度风险

审稿人已经指出 Table 2 存在异常规则性。对当前表格进一步核查可以发现：

- 涿州场景中，全部 9 种方法的 Precision 均恰好等于 F1 加 4.00 个百分点；
- 郑州场景中，9 种方法里有 7 种方法的 Precision 与 F1 恰好相差正或负 4.00 个百分点；
- F1、IoU、Precision 和 Recall 之间在数学上仍能相互对应，因此问题不像普通四舍五入错误，更像是多个指标由固定关系反推、复制或批量生成。

这不代表可以直接认定数据错误，但已经构成必须从像元级 TP、FP、FN、TN 重新核算的高风险信号。只修改表格数字而不保留原始预测、混淆矩阵和统一评估脚本是不够的。

在完成这一审计前，不应继续依据当前 Table 2 设计标题、摘要或投稿策略。

### 4.2 CFDepth 的有效水深覆盖率具有较强的“设计内生性”

CFDepth 在公式中通过

\[
S_{\mathrm{nb}}(p)=\max \{\tilde z_{\mathrm{out}}(q^*(p)),Z(p)+d_{\min}\}
\]

以及后续的

\[
\max\{S_0(p),Z(p)+d_{\min}\}
\]

显式保证被保留为 flooded 的像元获得正水深。因此，valid depth ratio 接近 100% 在一定程度上是算法约束的直接结果，而不是外部证据证明水深更准确。

该指标可以保留，但应改名或解释为：

- depth availability；
- positive-depth completion ratio；
- unresolved-pixel reduction。

它不能继续作为 CFDepth 主要性能证据。

### 4.3 当前水深验证只证明了“完整”和“平滑”，没有证明“数值正确”

现有水深证据包括：

- 有效正水深覆盖率；
- WSE-gradient ECDF；
- 8 个照片参考点的视觉区间一致性。

这些证据分别反映覆盖完整性、局部水面平滑性和视觉合理性，但不能给出 MAE、RMSE、bias 或 NSE 等数值精度。审稿人对此判断是合理的。

同时，较平滑的水面并不必然更正确。道路、堤坝、桥涵、地下排水、泵站和局部地形障碍都可能形成真实的水位不连续。当前稿件需要明确区分：

| 证据 | 能证明什么 | 不能证明什么 |
|---|---|---|
| Valid depth ratio | 是否返回了正水深 | 水深是否正确 |
| WSE-gradient ECDF | 水面局部变化是否较平缓 | 平缓水面是否符合真实水动力 |
| 现场照片 | 深浅等级是否大致合理 | 像元级绝对水深精度 |
| 水动力基准 | 数值误差和系统偏差 | 真实城市事件中的全部水动力过程 |

### 4.4 缺少解释“为什么有效”的实验

当前稿件把性能提升归因于 HA、CQI 和 Mask2Former 的共同作用，但没有消融实验，也没有针对各模块对应问题的稳健性分析。

审稿人真正需要的不是更多特征图，而是：

- PSC 是否真的缓解跨时相统计偏移；
- deformable alignment 是否真的提高残余错位条件下的稳健性；
- DINOv3 是否在单通道 SAR 上提供了有效语义，而不是增加参数；
- CQI 是否主要改善小斑块、狭窄积水和复杂边界；
- 这些模块是否具有统计显著的独立贡献。

这也是编辑认为论文只“报告更高指标，但没有解释提升原因”的直接来源。

### 4.5 训练、标签和预处理信息不足

当前稿件只说明训练数据来自 VarFloods 和 S1GFloods，未完整报告：

- 每个数据集包含多少事件、影像对和切片；
- 训练、验证、测试如何划分；
- 是否保证 event-disjoint；
- 同一事件的重叠切片是否跨集合；
- 标签如何统一；
- SAR 是否经过辐射定标、对数转换、截断和归一化；
- 数据增强方式；
- GF3 是否参与微调或阈值选择；
- GF3 洪水标签由何种数据和标注流程生成；
- 永久水体是否从标签中剔除；
- 模糊区域如何处理。

这些不是附加信息，而是当前所有精度结论成立的前提。

### 4.6 DINOv3 与 SAR 的跨域适配没有说明

当前稿件没有清楚回答：

- 单通道 SAR 是否复制为三通道；
- 输入采用何种数值范围和归一化；
- DINOv3 的预训练数据域；
- 使用了哪个模型规模、哪些层；
- 特征如何缩放到 P3–P5；
- 是否完全冻结；
- 自然图像语义是否可能抑制 SAR 特有散射细节。

这会直接削弱 semantic calibration 的可信度。

### 4.7 SAR—DEM 的有效空间尺度没有明确

稿件只写明将 SAR 支持面与 FABDEM 对齐到公共网格，但没有说明：

- 最终水深产品的有效空间分辨率；
- DEM 上采样或 SAR 下采样方式；
- 高程插值方法；
- 垂向基准是否一致；
- 深度梯度是在何种网格上计算。

即使输出图显示为较细像元，水深的有效分辨率仍受 DEM 控制，不能由 SAR 像元尺寸决定。该问题会影响照片点比较和 WSE-gradient 的解释。

### 4.8 双时相间隔带来非洪水变化混杂

当前 Table 1 中，郑州和涿州的灾前—灾后影像分别跨越约一年。采用相近季节可以降低部分物候差异，但仍可能包含城市建设、道路变化、农业变化和河道变化。

需要明确：

- 为什么选择这一灾前时相；
- 是否存在更临近事件的影像；
- 如何排除年度非洪水变化；
- 模型对这类变化的误检率；
- 是否按建筑、道路、农田、永久水体分层统计 FP。

### 4.9 论文视觉和结构仍然强调网络，而不是水文问题

当前正文 Fig. 1–4、Eq. (1)–(10) 集中展示 PSC、deformable alignment、pair token、query attention 和 Mask2Former。对遥感算法读者是正常的，但对 Journal of Hydrology 读者会形成“先有网络，再找洪水应用”的阅读印象。

编辑关于“Sections 3 and 4 contain excessive algorithmic details”的判断与当前版式是一致的。

## 5. 当前最适合的投稿定位

论文存在两种可行定位，必须选择其一，不能继续处于中间状态。

### 定位 A：Journal of Hydrology / NHESS 路线

中心问题应改为：

> 不完整、变化定义的洪水支持面如何改变 DEM 约束水深推断的边界条件、误差传播和适用范围？

在这一定位下，CFDepth 必须增加受控水动力验证和掩膜误差敏感性分析；HA-CQI 是获取洪水观测支持面的前端，而不是全文中心。

### 定位 B：JAG / IEEE JSTARS 路线

中心问题应改为：

> 如何在复杂城市 SAR 场景中获得跨传感器、跨事件的稳定新增淹没范围，并进一步形成面向应急判读的 DEM 约束深度代理？

在这一定位下，HA-CQI 是主要创新，CFDepth 被明确降级为 secondary operational extension 或 first-order depth proxy。水动力基准仍有价值，但不再是论文能否成立的唯一前提。

## 6. 当前缺少且不能自行假设的信息

以下信息会直接影响最终投稿决策：

1. Table 2 是否能由保存的原始预测和混淆矩阵独立复现；
2. VarFloods、S1GFloods 的样本量和事件划分；
3. GF3 数据是否参与微调、模型选择或阈值选择；
4. GF3 洪水标签的制作来源、标注人员和永久水体处理；
5. DINOv3 的具体版本及 SAR 输入转换方式；
6. CFDepth 中 \(d_{\min}\)、\(r_b\)、\(D_{\max}\) 的实际取值和选择依据；
7. FABDEM 与 SAR 的重采样和垂向基准处理；
8. 现场照片的来源、拍摄时间、坐标误差和使用权限；
9. SAR 精确成像时间及其与降雨峰值、洪峰或退水阶段的关系。

# 二、审稿意见逐条分析与响应策略

以下分类对应编辑和两位审稿人的全部实质性意见。

| 审稿意见 | 问题判断 | 是否接受修改 | 修改策略 | 工作量 |
|---|---|---|---|---|
| AE：论文偏计算机视觉，水文学贡献不足 | 完全合理，是拒稿主因 | 必须修改 | 将主线改为 change-defined support 的边界条件失配、时间特定水深和误差传播；HA-CQI 定位为观测模块 | 中 |
| AE：方法创新没有清楚阐明 | 合理；当前呈现容易被理解为模块组合 | 必须修改 | 不再以模块数量定义创新；建立“问题创新—观测创新—推断创新”三级贡献 | 小 |
| AE：算法细节过多，水文读者难以理解 | 合理 | 必须修改 | 标准 backbone、FPN、Mask2Former 公式和完整结构图移至补充材料；正文保留 HA、CQI 和 CFDepth 的关键原理 | 小 |
| AE：没有解释为什么优于基线 | 完全合理 | 必须修改 | 增加模块消融、错位扰动、辐射扰动、小斑块/边界分层分析和 land-cover-stratified error analysis | 中 |
| R1.1：Related Work 与 Introduction 重复 | 合理 | 必须修改 | 删除独立 Related Work；将文献综述嵌入引言中的三个 gap | 小 |
| R1.2：Section 3 描述大量成熟方法 | 合理 | 部分接受 | 标准模块仅引用；保留针对 SAR 问题的改动和必要公式；详细实现放 Supplement | 小 |
| R1.3：Section 4 混有方法内容 | 合理 | 必须修改 | 单独设置 Data and reference construction 与 Experimental design；将预处理、标签、指标、参数从 Results 中移出 | 小 |
| R1.4：计算比较多，但针对科学问题的比较不足 | 完全合理 | 必须修改 | 增加机制性消融、洪水掩膜扰动和 change-defined support 受控基准，不再只比较最终 IoU | 中 |
| R1.5：Discussion 太短 | 完全合理 | 必须修改 | 扩展为机制解释、证据等级、误差传播、适用域、失效场景和应急意义五个小节 | 小—中 |
| R2：范围误差会传播到水深 | 完全合理 | 必须修改 | 对掩膜进行腐蚀、膨胀、随机 FP/FN 和概率阈值扰动，绘制输入 IoU—水深误差曲线 | 中 |
| R2：水面平滑、边界水位、DEM 精度等强假设 | 合理 | 必须修改 | 在方法前增加 Assumptions and applicability 表；在 Discussion 中逐项说明成立条件与失效场景 | 小 |
| R2：形态连通不等于水力连通 | 完全合理 | 必须修改 | 将 connected component 明确称为 raster/geodesic connectivity，不再称为 hydraulic connectivity；讨论桥涵、排水和障碍物 | 小 |
| R2：水深是 SAR 成像时刻而非峰值 | 完全合理 | 必须修改 | 表格中补充精确成像时间及相对事件阶段；全文统一使用 time-specific depth approximation | 小 |
| R2：CFDepth 缺乏强独立验证 | 完全合理 | 采用替代实现 | 不新建城市水动力模型；使用公开水动力深度图和 DTM 做受控基准；真实城市事件保留为应用验证 | 中；现场验证为大且不建议 |
| R2：照片不能替代实测水深 | 合理 | 必须修改 | 章节改名为 Photographic plausibility assessment；提供来源、时间、坐标、对象高度假设和不确定范围 | 小—中 |
| R2：valid ratio 不等于精度 | 完全合理 | 必须修改 | 改称 depth availability；降为辅助指标；增加 MAE、RMSE、bias、NSE 等外部基准指标 | 小—中 |
| R2：WSE 平滑不等于精度 | 完全合理 | 必须修改 | 将 ECDF 定位为 internal-regularity diagnostic；在公共基准中计算相对真实 WSE 的梯度误差 | 小—中 |
| R2：训练和评估流程不充分 | 完全合理 | 必须修改 | 补充样本量、event-disjoint split、切片重叠、标签统一、归一化、增强、阈值和随机种子 | 小，前提是记录完整 |
| R2：GF3 标签来源不清 | 完全合理 | 必须修改 | 新增 Reference flood map construction 小节；报告多源依据、标注流程、永久水体和不确定区域 | 小—中 |
| R2：Table 2 数值异常规则 | 严重且合理 | 必须修改，优先级最高 | 从原始 TP/FP/FN/TN 全量重算；统一脚本；输出原始计数和评估掩膜；检查所有 baseline checkpoint | 小—中 |
| R2：DINOv3 用于 SAR 的方式不清楚 | 完全合理 | 必须修改 | 说明通道转换、数值归一化、模型版本、预训练域、提取层、冻结方式和尺度映射 | 小 |
| R2：照片区间应有不确定性 | 合理 | 必须修改 | 至少由两名标注者独立估计；报告对象尺寸区间、透视误差和空间邻域 | 小 |
| Minor：水深精度表述过强 | 合理 | 必须修改 | 将 accuracy/validation/estimate 改为 approximation、consistency、availability，除非增加基准后有相应证据 | 小 |
| Minor：limitations 不充分 | 合理 | 必须修改 | 补充 DEM、时相、排水、泵站、孤立积水、FP、连接误差和尺度限制 | 小 |
| Minor：重复参考文献 | 明确错误 | 必须修改 | 合并 Chen et al. 2022a/2022b 和 Daudt et al. 2018a/2018b，检查全表 DOI 和出版状态 | 小 |

# 三、综合修改方案

## 1. 标题修改

### 当前问题

当前标题同时放置 HA-CQI、CFDepth、Urban Flood Extent Mapping、Change-Defined Depth Estimation 和 Bi-Temporal SAR Imagery，明显是模块驱动型标题。对 Journal of Hydrology 编辑而言，这会强化“计算机视觉论文”的第一印象。

### 推荐标题

在尚无数值水深基准时，首选：

> **Urban Flood Extent and DEM-Constrained Depth Approximation from Bi-Temporal SAR**

强调 change-defined support 时：

> **From Change-Defined Flood Extent to DEM-Constrained Depth in Urban SAR Mapping**

只有在补充水动力基准且结果支持后，才建议使用：

> **Urban Flood Extent and DEM-Constrained Depth Estimation from Bi-Temporal SAR**

### 修改理由

- 删除算法缩写，优先呈现科学问题；
- 使用 approximation 限定当前证据等级；
- 避免把“first-order”理解成低质量结果，可在摘要和方法中定义；
- 保留 extent-to-depth 的整体特色。

**是否新增实验：** 否。  
**工作量：** 小。

## 2. 摘要修改

### 当前问题

当前摘要对 HA-CQI 和 CFDepth 的流程介绍较多，但对以下问题交代不足：

- 为什么 change-defined support 与普通洪水范围不同；
- 水深结果是成像时刻的近似，而非峰值水深；
- valid depth ratio 和照片区间不是数值精度；
- 缺少受控水深基准。

### 建议结构

第一句：应急需求与时间限制。

第二句：提出真正的双重 gap：

- 城市 SAR 双时相特征不可比；
- 新增淹没支持面缺失永久水体和低高程连接，使传统边界水位法失稳。

第三句：用一句话概括 HA-CQI，不罗列所有子模块。

第四句：定义 CFDepth 是 extent-conditioned、DEM-constrained、time-specific first-order approximation。

第五句：报告经重新核验后的范围结果。

第六句：

- 若完成公共水动力基准，报告 MAE、RMSE、bias 和 mask sensitivity；
- 若未完成，不再把 valid ratio 作为主要精度结论，只写成 coverage diagnostic。

### 应删除或降级的内容

- “field-photo reference validation”；
- 将 valid depth ratio 的大幅提升直接解释为更准确；
- 将较低 WSE gradient 解释为更真实；
- 将输出用于 peak flood damage assessment。

**是否新增实验：** 摘要本身不需要，但最终版本取决于公共基准。  
**工作量：** 小。

## 3. 引言修改

### 3.1 研究 gap 的重新构建

建议将 Introduction 和 Related Work 合并，并按以下逻辑重写。

#### Gap 1：城市 SAR 洪水不是普通二类分割问题

建成区洪水可能同时表现为：

- 开阔水体后向散射降低；
- 建筑—水面双程散射增强；
- 阴影和叠掩变化；
- 道路和局部积水引起的邻域结构重组。

因此，双时相特征需要在差分或交互前完成统计、几何和语义上的可比化。

#### Gap 2：变化检测范围不等于完整洪水期范围

现有 FwDET、FLEXTH 和 FlDepth 等方法通常利用洪水边界和 DEM 推断水面，但输入多为相对完整的洪水期范围，或显式引入永久水体、no-data mask 和中心线等信息。

双时相变化检测输出的是“新增淹没区域”：

- 通常不包含灾前永久水体；
- 可能不包含河道和低高程连接区域；
- 可能包含碎片化或狭窄支持面；
- 可能受到 FP/FN 影响。

因此，其外边界不一定是可靠的湿—干水位边界。当前论文真正应解决的是这种 **boundary-condition mismatch**，而不是泛泛宣称现有 DEM 方法存在不连续。

FwDET v2.0 本身通过水动力模型验证，并将其输出限定为 first-order、synoptic depth；FLEXTH 进一步采用 ICESat-2、水动力模拟和合成 no-data mask 评估；FlDepth 也采用水动力模拟与 ICESat-2 ATL13 验证。它们为本论文提供的启示是：现场数据稀缺并不意味着可以省略受控基准，而是应借助独立模拟和卫星测高构建不同层次的证据。  

#### Gap 3：范围误差如何传递到水深尚未量化

当前 extent-to-depth 流程的关键不确定性链条是：

\[
\text{SAR observation}
\rightarrow \hat M
\rightarrow \text{component topology}
\rightarrow \text{boundary elevation}
\rightarrow \hat S
\rightarrow \hat d
\]

当前稿件只评估了第一项和最后输出，没有量化中间传播过程。

### 3.2 建议提出三个研究问题

- **RQ1：** 统计协调、局部对齐和 change-query interaction 是否分别提高城市 SAR 洪水变化识别的稳健性？
- **RQ2：** 完整洪水支持面被转换为变化定义支持面后，传统 DEM 水深方法的误差如何变化？
- **RQ3：** CFDepth 在何种 extent error、DEM 质量和连接条件下能够提供可用的一阶水深近似？

### 3.3 建议重写贡献点

1. 提出 change-defined flood support 的边界条件失配问题，并区分其与完整洪水期范围。
2. 提出 HA-CQI，从统计、局部几何和变化关系三个层面提高复杂城市 SAR 双时相特征可比性。
3. 提出 CFDepth，在连通分量内部对不完整新增淹没支持面执行 DEM 约束水面补全。
4. 通过外部城市事件、受控水动力基准和掩膜扰动分析分别评价范围识别、深度精度和误差传播。

**是否新增实验：** 引言本身不需要，但第四项必须由新增实验支持。  
**工作量：** 中，主要是结构重写和文献比较。

## 4. 方法部分修改

### 4.1 建议的新论文结构

1. Introduction  
2. Study areas, data, and reference construction  
3. Methodology  
   3.1 Problem formulation and assumptions  
   3.2 HA-CQI  
   3.3 CFDepth  
   3.4 Uncertainty and quality indicators  
4. Experimental design  
5. Results  
6. Discussion  
7. Conclusions  

### 4.2 不建议继续扩展网络结构

当前算法本身不需要再增加新的注意力、损失函数或视觉基础模型。继续加模块只会强化“architecture stacking”的评价。

应压缩：

- EfficientNet-B0 和 FPN 的常规说明；
- 标准 Mask2Former decoder 的完整公式；
- 通用 Focal/Tversky loss 说明；
- 与创新无关的结构细节。

这些内容可以放入 Supplementary Methods。

### 4.3 HA-CQI 正文应保留的内容

正文只围绕三个明确问题展开：

| 城市 SAR 问题 | 对应设计 | 需要验证的假设 |
|---|---|---|
| 双时相浅层散射统计不同 | Pair-shared style calibration | 对辐射缩放和偏移更稳健 |
| 建筑/道路边缘存在残余错位 | Deformable alignment | 对 1–3 像元错位更稳健 |
| 洪水可能表现为变暗、变亮或结构重组 | Pair tokens + change queries | 对小斑块和复杂边界召回更高 |
| 通用语义与 SAR 存在域差异 | Frozen DINOv3 calibration | DINOv3 具有增益且不破坏边界 |

### 4.4 必须补充 DINOv3 实现说明

至少包括：

- 模型名称和参数规模；
- 预训练来源；
- 单通道 SAR 到模型输入的转换；
- 归一化数值；
- 选取的 transformer block；
- 是否共享预/后时相分支；
- 是否冻结全部参数；
- 特征如何与 FPN 尺度对齐；
- DINOv3 分支的参数量和推理开销。

### 4.5 CFDepth 应增加“假设与输出定义”

建议在方法部分直接加入表格：

| 假设 | 适用条件 | 失效风险 |
|---|---|---|
| 外边界高程近似局部水面 | 低坡、缓变或积水型洪水 | 边界错位导致系统偏差 |
| 连通分量内部可局部传播 | 支持面拓扑与真实积水结构近似一致 | 窄错误连接会合并不同积水单元 |
| DEM 可表达有效地形 | 地形误差小于目标水深尺度 | 建筑、道路、桥涵和排水结构缺失 |
| 水面局部变化相对平缓 | 缓流或静水型场景 | 快速洪流和强水力坡降不适用 |
| SAR 范围对应特定时刻 | SAR 与事件阶段明确 | 不能解释为峰值或累计最大水深 |

### 4.6 区分形态连通与水力连通

当前算法只保证在二值支持面内进行 geodesic/reachable propagation。它不能证明两个像元在真实城市排水系统中水力连通。

正文应明确使用：

- raster connectivity；
- support-constrained reachability；
- component-restricted propagation。

避免使用 hydraulic connectivity，除非引入道路、涵洞、河道或水动力信息。

### 4.7 将 \(d_{\min}=0.01\) m 定义为数值正则项

1 cm 水深通常低于公共 DEM 的实际垂向不确定性，因此不宜称为“physically interpretable positive depth”。更准确的表达是：

> a numerical positivity regularization used to prevent non-positive depth within retained flooded pixels.

同时应报告：

- 被 \(d_{\min}\) 强制修正的像元比例；
- 使用初始水面、边界补全和最小水深正则的像元比例。

### 4.8 增加输出质量标识

不必重新设计网络，可以为 CFDepth 结果同步输出一个 reliability layer，至少包含：

- HA-CQI 洪水概率；
- DEM 坡度或局部起伏；
- 外边界高程离散度；
- 到有效边界的传播距离；
- 连通分量大小；
- 是否由 fallback surface 补全。

这比只输出一个确定性水深图更符合应急产品要求。

**是否新增实验：** 参数敏感性和质量图需要中等工作；其余主要为文字和代码统计。  
**工作量：** 小—中。

## 5. 实验与验证部分修改

这是决定能否重投 Journal of Hydrology 的核心。

### 5.1 Priority 0：重新审计全部范围精度

必须从每个方法的二值预测重新生成：

- TP、FP、FN、TN；
- IoU；
- Precision；
- Recall；
- F1；
- MCC 或 balanced accuracy；
- boundary F1。

具体要求：

1. 所有方法使用同一评估掩膜和同一像元范围；
2. 记录 threshold、NoData 和永久水体处理；
3. 输出原始混淆矩阵到 Supplement；
4. 使用同一个独立评估脚本；
5. 检查是否混淆百分比与小数；
6. 检查结果是否来自真实 checkpoint，而非手动汇总表；
7. 将评估脚本随代码发布。

在该步骤完成前，不应保留当前 Table 2 中任何数值结论。

### 5.2 补全训练和测试协议

新增一张数据表：

| 数据源 | 事件数 | 影像对数 | 切片数 | 空间分辨率 | 极化 | 用途 | 划分原则 |
|---|---:|---:|---:|---:|---|---|---|

必须说明：

- event-disjoint split；
- 同一事件切片不跨训练/验证；
- 切片窗口和重叠率；
- 标签统一规则；
- SAR 定标、log scaling、clipping、normalization；
- augmentation；
- 随机种子；
- GF3 是否零样本迁移；
- 模型选择只依据哪一验证集。

### 5.3 补充 GF3 参考标签说明

建议设置独立小节：

> Reference flood map construction and uncertainty

应包括：

- 标签数据来源；
- 是否结合光学、SAR、官方产品、新闻照片和土地覆盖；
- 标注者数量；
- 永久水体处理方式；
- 边界不确定区是否排除；
- 标签最终分辨率；
- 是否进行独立复核；
- 对难以判断的建筑阴影、树冠和河岸如何处理。

若标签主要由当前 post-event SAR 解释得到，需要主动讨论 reference dependence，避免把同一 SAR 视觉特征既作为输入又作为绝对真值。

### 5.4 HA-CQI 消融实验

最低可接受配置：

| Variant | PSC | Deformable alignment | DINO calibration | CQI |
|---|---:|---:|---:|---:|
| Baseline |  |  |  |  |
| +PSC | ✓ |  |  |  |
| +Alignment | ✓ | ✓ |  |  |
| +Semantic calibration | ✓ | ✓ | ✓ |  |
| Full HA-CQI | ✓ | ✓ | ✓ | ✓ |

建议至少报告：

- IoU；
- Recall；
- boundary F1；
- 小连通斑块 recall；
- 参数量和推理时间放 Supplement。

若训练成本受限，可以把 DINOv3 和 CQI 分别作为两个关键单因素消融，但不能完全没有消融。

### 5.5 针对模块原理的稳健性实验

这些实验比增加更多通用基线更有价值。

#### 残余错位实验

人为对灾前影像施加：

- ±1 像元；
- ±2 像元；
- ±3 像元。

比较 baseline、无 alignment 和完整 HA-CQI 的 IoU 退化曲线。

#### 辐射统计扰动实验

对灾前或灾后 SAR 施加合理的增益、偏置或噪声扰动，评价 PSC 是否减少性能退化。

#### 目标形态分层

按洪水连通分量面积或形状划分：

- 小斑块；
- 狭窄线状区域；
- 中型斑块；
- 大型连片淹没。

报告 component recall 和 boundary F1，验证 CQI 是否真正改善论文声称的碎片化和狭窄区域。

#### 地物分层误差

基于现有土地覆盖或 OSM 数据，分别统计：

- built-up；
- roads；
- permanent water vicinity；
- cropland；
- low-backscatter non-water。

这可直接回应“暗道路被误判后仍会生成水深”的审稿意见。

### 5.6 最推荐的低成本水深验证：公开水动力基准

不建议从头建设郑州或涿州城市二维水动力模型。更高投入产出比的方案是复用公开水动力基准。

Betterle and Salamon 的 FLEXTH 研究使用了公开的西班牙 Tera–Órbigo–Esla 河流汇流区百年一遇水深图和 2 m LiDAR DTM，同时还使用了 Brazos River 水动力模拟、ICESat-2 和合成 no-data mask。FwDET v2.0 也采用同一 DEM 和模型范围作为输入，以隔离水深算法本身误差；FlDepth 则采用 MIKE 21FM 和 ICESat-2 ATL13 双重验证。  

建议采用以下三级实验。

#### D1：完整支持面受控基准

1. 获取公开参考水深和对应 DTM；
2. 将参考深度 \(d_{\mathrm{ref}}>0\) 转换为标准洪水范围；
3. 使用相同范围和相同 DTM 分别运行：
   - FwDET v2.0；
   - FLEXTH；
   - CFDepth；
4. 只比较深度算法，不引入 SAR 范围误差。

指标：

- MAE；
- RMSE；
- bias；
- NSE；
- Spearman correlation；
- 深度等级一致率；
- 水面高程 MAE；
- valid coverage。

应分别报告：

- common-valid pixels 上的精度；
- 全支持面上的覆盖和惩罚误差。

这样可以避免“某方法只在较少像元上计算，所以误差较低”或“强制补全提高覆盖但降低精度”的混淆。

#### D2：模拟 change-defined support

这是最能证明 CFDepth 独特价值的实验。

从完整水动力洪水范围中依次构建：

1. 去除永久水体后的新增淹没支持面；
2. 去除部分低高程连接区域；
3. 边界腐蚀 1–3 个像元；
4. 随机移除 1%、5%、10% 的 flood pixels；
5. 注入 1%、5%、10% 的 false positives；
6. 创建碎片化和窄连接支持面。

在每种扰动下比较 CFDepth、FwDET 和 FLEXTH 相对于原始参考水深的误差。

最终输出应是：

\[
\text{Mask IoU}
\rightarrow
\text{Depth MAE/RMSE/Bias}
\]

的敏感性曲线。

这一实验能够同时回答：

- CFDepth 是否真正适用于 change-defined support；
- FP 与 FN 哪一种对水深影响更大；
- 何种掩膜质量以下不应输出水深；
- valid coverage 的提高是否以数值误差为代价。

#### D3：真实城市事件应用

郑州和涿州继续保留，但其角色改为：

- 外部跨传感器/跨事件洪水范围评估；
- 城市真实场景下的空间合理性展示；
- 照片辅助的区间 plausibility check；
- failure-case analysis。

不要再用它们承担绝对水深精度验证。

### 5.7 ICESat-2 的使用策略

ICESat-2 不应被列为必须完成项。只有同时满足以下条件才值得加入：

- 轨迹穿过有效淹没区域；
- 与 SAR 获取时间足够接近；
- 有可识别的水面光子；
- 干期与湿期垂向基准能够匹配；
- 城市建筑和植被噪声可控。

对于持续时间短、空间碎片化的城市积水，强行使用无时空匹配的 ICESat-2 可能引入比原问题更大的不确定性。当前最稳妥的路线仍是公开水动力基准。

### 5.8 DEM 和参数敏感性

建议至少改变：

- DEM 数据源：FABDEM 与另一种公开 DEM；
- \(d_{\min}\)；
- boundary smoothing radius \(r_b\)；
- maximum propagation distance \(D_{\max}\)；
- 输出网格尺度。

报告：

- 水深中位数变化；
- 深度等级变化比例；
- WSE-gradient 变化；
- fallback completion ratio。

这一实验量化的是输入和参数不确定性，不应称为精度验证。

### 5.9 现场照片修改

将“Field-photo reference validation”改为：

> **Photographic interval-consistency assessment**

新增表格：

| Point | Photo source | Capture time | Location uncertainty | Reference object | Assumed dimension | Estimated interval | Annotator agreement |
|---|---|---|---|---|---|---|---|

若照片无法可靠确定时间和位置，应删除具体点位水深数值，只作为灾情背景图使用。

## 6. 讨论与限制部分修改

建议 Discussion 至少分为五节。

### 6.1 Why HA-CQI improves urban flood mapping

不要重复 Table 2，而要建立模块—现象对应关系：

- PSC 减少跨时相统计偏移；
- alignment 减少建筑和道路边缘伪变化；
- DINOv3 提供区域级上下文，但自然图像先验仍存在域差异；
- CQI 同时表达变暗、变亮和邻域重组；
- small-component 和 boundary 分析支持这些解释。

### 6.2 What CFDepth estimates—and what it does not

明确：

- 输出是 SAR 成像时刻的 DEM-constrained first-order approximation；
- 不是最大洪水深度；
- 不是水动力模拟；
- 不是地下排水系统水位；
- 不包括被 change detection 排除的永久水体；
- 精度受支持面和 DEM 共同限制。

### 6.3 Error propagation and uncertainty

按以下链条讨论：

1. SAR false positive 会在干燥位置生成虚假正水深；
2. false negative 会改变连通分量和有效边界；
3. 狭窄错误连接会把两个独立积水区合并；
4. DEM 偏差会直接传递到水面和深度；
5. 边界高程误差会沿传播路径影响内部像元；
6. minimum-depth enforcement 会提高 coverage，但可能引入正偏差。

### 6.4 Applicability envelope

较适合：

- 缓流、静水或积水型洪水；
- 地势相对平缓；
- DEM 能表达主要地形；
- 洪水范围相对可靠；
- 连通分量与真实积水单元大致一致。

不适合或低可信：

- 快速山洪和强水力坡降；
- 地下排水、泵站和涵洞显著控制水位；
- 桥梁、堤坝和高架道路密集；
- 独立街区积水被窄连接误合并；
- DEM 垂向误差与目标水深同量级；
- SAR false positives 较高；
- SAR 获取时间与洪峰错开且事件阶段未知。

### 6.5 Operational implications

论文可以合理声称：

- 快速提供新增淹没范围；
- 为现场排查提供相对深浅和覆盖信息；
- 支持优先级排序和灾情概览；
- 输出应与可靠性图层共同使用。

不应声称：

- 可替代二维水动力模型；
- 可直接用于工程设计；
- 可代表事件最大水深；
- 已证明像元级数值精度。

# 四、最低成本修改路线（Priority Plan）

以下时间估计以代码、原始预测和数据能够直接复现为前提。

| 优先级 | 必须完成的工作 | 预计工作量 | 对录用概率的影响 |
|---|---|---|---|
| **Priority 0：结果可信度门槛** | 重算 Table 2；保存 TP/FP/FN/TN；核查所有 baseline；补全标签来源、数据划分和 DINOv3 输入 | 小—中，约 2–5 个工作日 | 决定论文是否具备继续投稿基础 |
| **Priority 1：论文身份重构** | 标题和摘要重写；合并 Related Work；压缩标准网络；增加假设、时间特定性、有效分辨率和 limitations | 小—中，约 1 周 | 显著降低编辑初筛风险 |
| **Priority 1：HA-CQI 机制证据** | 核心消融；1–3 像元错位测试；小斑块/边界分层；地物类型 FP/FN | 中，约 1–2 周 | 直接回应编辑和审稿人关于“为什么更好” |
| **Priority 2：CFDepth 受控验证** | 公开水动力基准；CFDepth/FwDET/FLEXTH 比较；完整支持面与 change-defined support 两套实验 | 中，约 2–4 周 | 决定是否有资格重投 Journal of Hydrology |
| **Priority 2：误差传播** | 掩膜腐蚀、膨胀、随机 FP/FN、阈值扰动；参数和 DEM 敏感性 | 中 | 将论文从结果展示提升到方法可靠性研究 |
| **Priority 3：增强性工作** | 第三个城市、ICESat-2、额外高分辨率 DTM、多时相序列 | 中—大 | 有增益，但不是当前最小发表集合 |
| **不建议投入** | 为历史事件重新开展现场测量；从零建设完整城市水动力模型；新获取 UAV/LiDAR；增加更多网络模块 | 大 | 投入高，不能保证解决核心审稿问题 |

## 最小必要修改集合

无论转投哪个一区期刊，至少应完成：

1. Table 2 全量重算；
2. 训练/验证/测试和 GF3 标签流程公开；
3. DINOv3 SAR 输入说明；
4. 核心消融实验；
5. 标题、摘要、引言和 Discussion 重构；
6. time-specific depth 与 peak depth 明确区分；
7. valid ratio 和 WSE-gradient 降级为诊断指标；
8. 照片验证改为不确定区间一致性；
9. 删除重复参考文献；
10. 明确 SAR—DEM 的有效输出尺度。

重投 Journal of Hydrology 时，再增加：

11. 一个公开水动力基准；
12. 一个 extent error propagation 实验。

# 五、投稿策略分析

## 方案 A：重投 Journal of Hydrology

### 1. 当前决定性质

审稿信是明确的“不推荐发表”，而不是 Major Revision。Reviewer #1 虽然鼓励作者彻底修改后再次投稿，但这不等于编辑正式发出的 revise-and-resubmit invitation。Elsevier 对 Editorial Manager 状态的定义也明确区分 Reject 与 Revise；因此，再次投稿应按新稿件处理，而不是上传普通 revision。 

### 2. 成功概率判断

以下为基于稿件状态的策略估计，不是期刊官方录用率：

| 修改程度 | 再投 JoH 的策略性成功概率 |
|---|---:|
| 当前版本原样或只润色文字 | 低于 10% |
| 完成结构重写、补全细节，但没有水深基准 | 约 10%–20% |
| 完成指标审计、HA-CQI 消融、公开水动力基准和掩膜敏感性 | 约 25%–40% |

JoH 官方页面目前显示 Impact Factor 7.3、CiteScore 11.8，并覆盖广泛水文学子领域；这意味着仅有较高遥感精度不足以形成期刊匹配，必须让边界条件、WSE、误差传播和适用域成为论文主体。

### 3. 必须补强的问题

重投 JoH 的最低门槛是：

- Table 2 完全可复现；
- 论文不再以网络模块组织全文；
- 有针对 HA、CQI 的机制性证据；
- CFDepth 至少在一个水动力基准上有 MAE/RMSE/bias；
- 有 extent error propagation；
- 区分 coverage、smoothness 和 accuracy；
- 明确时间特定性和水力假设；
- 真实城市案例只承担外部应用验证。

### 4. 是否值得投入

**值得，但有条件。**

若公开水动力基准能够直接获取，且代码能够在 2–4 周内完成 CFDepth、FwDET、FLEXTH 的统一比较，则重投 JoH 仍有合理性。

若无法完成公共基准，或 CFDepth 只提高覆盖率但没有改善数值误差，则不应继续以 JoH 为首选。此时应将 CFDepth 降为应急辅助模块，转向应用遥感期刊。

### 5. 重投方式

建议在新投稿 cover letter 中主动说明：

- 前稿件编号；
- 这是实质性重构后的新版本；
- 原始结果已重新审计；
- 新增了哪些数据协议、消融、基准和敏感性分析；
- 论文科学问题从算法框架调整为 change-defined support 的水文边界条件问题。

不建议对当前决定提出 appeal，因为编辑和审稿人的核心质疑具有实质性依据。

## 方案 B：转投其他一区期刊

当前公开信息显示：JAG 官方页面列出 Impact Factor 8.2、CiteScore 14.6；IEEE JSTARS 官方页面列出 Impact Factor 6.3，并明确强调 applied Earth observation；JHRS 官方页面列出 Impact Factor 5.6、CiteScore 7.1；WRM 官方页面列出 2025 Impact Factor 5.7、首轮决定中位数 23 天，并且截至 2026 年 8 月已发表到 500 余篇文章编号。NHESS 的公开 SJR 2025 信息为 Q1。

需要说明：公开 SJR Q1、JCR Q1 和中科院一区不是同一个分区体系，且年度结果会变化。正式投稿前应以所在机构可访问的 2026 JCR 和中科院分区数据库核验。

| 推荐顺序 | 期刊 | 匹配原因 | 必要修改 | 难度判断 | 发文规模 |
|---:|---|---|---|---|---|
| 1 | **International Journal of Applied Earth Observation and Geoinformation** | 最适合“遥感观测—环境应用—灾害产品”的整体框架；接受 EO 数据和应用方法 | 完成结果审计、标签说明、消融；CFDepth 可作为 first-order depth product，公共基准强烈推荐但不是绝对前提 | 中偏高 | 高；2026 年按连续月度卷出版，6 月已至 Volume 150。 |
| 2 | **IEEE JSTARS** | 与 SAR、变化检测、应用型 Earth observation 和信息产品高度匹配；对算法细节容忍度高于水文学期刊 | Table 2 审计、DINOv3 说明、消融、跨传感器协议；将 HA-CQI 设为主要贡献 | 中 | 高；官方定位为 applied Earth observation，当前 IF 6.3，公开 SJR 2025 Q1。 |
| 3 | **Natural Hazards and Earth System Sciences** | 城市洪水、遥感监测、灾害应急和不确定性高度匹配；FLEXTH 已在该刊发表，说明主题适配 | 必须比较 FLEXTH，并增加公共水动力基准、误差传播和完整 limitations | 中偏高；公开评审会放大数据与验证问题 | 中高；2026 年持续出版大量洪水和 AI 灾害研究，公开 SJR 2025 Q1。 |
| 4 | **Journal of Hydrology: Regional Studies** | 两个中国城市洪涝案例具有区域水文与应急应用价值，期刊亦关注 AI in regional hydrology | 强化事件背景、成像时刻、区域差异和水动力基准；减少纯算法内容 | 中 | 中高；2026 年已连续出版 Volume 63–67，并有区域水文 AI 专题。 |
| 5 | **Water Resources Management** | 偏重应用水文、模拟、预测和管理，发文量大；适合快速应急深度产品 | 必须明确区别于已发表于该刊的 FlDepth；需要公共基准和应急决策意义 | 中 | 很高；官方页面显示截至 2026 年 8 月文章编号已超过 500。 |

### 具体转投建议

- **不增加公共水动力基准：** IEEE JSTARS 优先，JAG 次之。论文应以 HA-CQI 为主，CFDepth 为 secondary application。
- **增加一个公共基准和掩膜敏感性：** JAG 优先；同时具备重投 JoH、NHESS 或 JHRS 的条件。
- **CFDepth 在基准中仅改善 coverage、未改善 error：** 不再把 CFDepth 作为与 HA-CQI 并列的主要创新；转 JSTARS。
- **CFDepth 在 change-defined support 扰动下明显优于 FwDET/FLEXTH：** 这是重投 JoH 最有价值的新证据。
- **CFDepth 在基准中没有稳定优势：** 应重新定位为工程实现或从标题中移除，不宜继续靠文字强化创新。

RSE、TGRS、ISPRS JPRS 和 HESS 不符合当前“尽量降低新增工作量”的目标，因为它们大概率会要求更广泛的跨区域验证、更强的方法独创性或更严格的物理验证。

# 六、最终建议

## 1. 当前论文是否值得继续修改

**值得。**

论文的问题不是研究方向失效，而是当前证据链没有与期刊定位对齐。城市 SAR 新增淹没范围与 DEM 水深推断之间的边界条件差异是一个有价值的问题，两个真实 GF3 洪涝案例也具有发表基础。

但继续修改的首要前提是：**Table 2 必须能够由原始预测完全复现。** 若不能复现，必须暂停投稿并重跑所有方法；任何文字重构都不能替代结果可信度。

## 2. 最推荐投稿路线

采用两道决策门槛：

### Gate 1：结果审计

- Table 2 可以复现：继续。
- Table 2 无法复现：全面重算，不进入投稿阶段。

### Gate 2：CFDepth 公共基准

- CFDepth 在完整支持面和模拟 change-defined support 下均能保持更低误差或更强抗扰动性：完成 Priority 0–2 后重投 Journal of Hydrology。
- CFDepth 主要提高 coverage，但数值误差优势有限：以 HA-CQI 为主，转投 JAG 或 IEEE JSTARS。
- CFDepth 对掩膜误差高度敏感且不优于 FLEXTH/FwDET：弱化或移除其独立方法创新定位。

## 3. 最小必要修改集合

对任何候选一区期刊，最低限度应完成：

- 重新核算并审计 Table 2；
- 完整描述数据、预处理、划分和标签；
- 增加 HA-CQI 核心消融；
- 增加错位或小斑块机制分析；
- 删除独立 Related Work 并压缩标准网络；
- 标题去除模块堆叠；
- 将水深限定为 time-specific first-order approximation；
- 将 valid ratio、WSE gradient 和照片证据降到正确的证据等级；
- 扩展误差传播和适用范围讨论；
- 修正重复参考文献。

重投 Journal of Hydrology 时，必须再增加：

- 一个公开水动力深度基准；
- 一个洪水掩膜误差传播实验。

## 4. 不建议投入的高成本修改

不建议：

- 为郑州和涿州历史事件重新组织现场水深测量；
- 从零搭建完整城市排水—地表二维耦合模型；
- 专门采集 UAV/LiDAR；
- 在没有时空匹配时强行使用 ICESat-2；
- 再增加新的 Transformer、foundation model 或 attention 模块；
- 用更强措辞掩盖水深证据不足。

**最终推荐：先用一个公共水动力基准和一个掩膜扰动实验检验 CFDepth 是否确有独立科学价值。结果成立，则重投 Journal of Hydrology；结果只支持覆盖补全，则转投 JAG 或 IEEE JSTARS。该路线在控制新增工作量的同时，能够最大程度提升论文可信度和发表概率。**