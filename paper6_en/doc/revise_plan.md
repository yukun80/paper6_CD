# 面向 Remote Sensing of Environment 的洪水制图论文重构实施方案

> 目标期刊：**Remote Sensing of Environment（RSE）**。
> 更新日期：2026-09-07。本轮将第二章压缩为两个小节、四个主题段落，GF3与地形资料表前置，训练评价及GF3图移入4.1；最新记录见7.25。引言及B1—B8保留。
> 仅修改两稿原第四段及方案记录；其他正文、图件和BibTeX不变。TeX同目录串行编译，不新增实验。
> 四张新版模型图仍仅交付网页端image2提示词；正文已临时显示图2—5及图11的旧图，五处不再使用空白占位框。图注注明旧图状态，模型差异及未核验统计不作为已验证结论。不运行训练、推理、新增评价或栅格重算，不修改算法、数据、权重和实验产物，不提交或推送。

核心问题：**复杂背景下，大范围淹没与弱小、狭窄、破碎洪水并存时，如何利用双时相 SAR 识别新增淹没，并利用所得范围与 DEM 估算水深。**

洪水制图由新增受灾范围识别和范围内水深估算两个相衔接的阶段组成；HA-CQI承担主要检测方法贡献，CFDepth承担给定范围下的水深估算阶段。跨传感器属于实验条件，不预设为标题或全文创新主轴。“弱小”在本方案中分别指可能的低对比观测和小空间对象，二者不等同；“狭窄”“破碎”“大范围”用于组织已有案例，不暗含已经完成对应类别的全景定量评价。

**已确认不安排额外验证**：不新增尺度或地类分层统计、ROI 统计、消融、辐射或位移扰动、水深支持区扰动、重复训练、测速或新研究区。允许读取已有代码、记录、指标和案例，核对其来源及适用范围。无法由既有材料回答的问题登记为证据缺口，并限制相关论断；不得将其重新列入强制补实验日程。

RSE 官方定位强调生物物理与定量遥感研究，也包括机器学习、变化检测和水文应用。因此，本稿以景观、SAR 观测和制图信息之间的联系解释方法价值，不将改投期刊理解为增加未经验证的物理机制。[Elsevier 官方期刊介绍](https://shop.elsevier.com/journals/remote-sensing-of-environment/0034-4257)（本会话核验日期：2026-09-05）。具体作者指南页面本次返回 403，格式要求见 B8，不据第三方模板推定。

文件中相对仓库的代码、数据路径用于核查，不应原样进入正式论文。EN 指 [当前英文稿](../elsarticle-template-harv_2.tex)，CN 指 [当前中文稿](../Manuscript_revised_cn.tex)；第1节及标明“实施前”的行号是历史审计位置；实际重构后按节名、语义label和第7节实施记录定位。第1节保留修改得失审计，不将其中的实施前状态视为当前状态。原稿指 [原始英文 PDF](../elsarticle-template-harv_2_raw.pdf)，页码为 PDF 页码。分析还使用了 [审稿意见](review_comment.md)、[最初修改方案](<revise_plan copy.md>)、本目录 DOCX 和现有图件。

## 1. 修改得失审计：保留、恢复与重写

### 1.1 七项问题的版本对照

| 问题 | 原稿依据 | 实施前稿及旧方案依据 | 保留／恢复／重写决定 | 关闭标准 |
|---|---|---|---|---|
| ① 标题定位 | 原稿第1页将 HA-CQI、CFDepth 与 Urban Flood Extent 并列；研究对象直观，但城市限定过窄 | EN 79、CN 99 以 across Sensors 为主；旧方案 §0.4、§6 将该标题冻结 | 保留双时相 SAR、新增淹没及深度扩展；恢复任务导向；撤销 urban-only、cross-sensor 统领和旧标题冻结 | 标题对象、方法与适用范围明确，CFDepth 为下游；不得暗示全球或纯城市通用性 |
| ② 章节顺序 | 原稿 §2 为 Related Work，§3 为方法，数据混在 §4 | EN 138／158／446／484、CN 对应位置：引言→方法→数据→实验；旧方案要求数据前移但未落实 | 保留取消独立 Related Work 的改进；第二章数据、第三章方法；重写章间过渡及图号映射 | 两稿章节同序；图按首次正文引用编号；无路径覆盖或丢失引用 |
| ③ 方法压缩 | 原稿第9–20页包含较多成熟技术说明及旧预测头 | 当前方法七个二级节；旧方案 §5、§10 出现八节／九节互相冲突 | 保留当前有效实现；按可比观测、变化证据、空间重建三个科学功能组织检测；CFDepth 单列第四节 | Overview 无独立二级标题；必要输入、公式、损失和复现信息有明确去处 |
| ④ 科学背景 | 原稿第2页行28–43有气候、城市化、暴露及制图需求，但因果与范围需重新核验 | EN 140 直接从范围与深度定义进入 SAR；146 偏重基础模型及 state-space 技术，150–156 再次跨传感器定位 | 恢复风险—暴露—观测需求入口；保留双时相目标清晰的改进；重写气候证据和研究问题链 | 降水、洪水和暴露不互相替代；摘要不列骨干／层号；引言不由结果倒推领域缺口 |
| ⑤ 复杂场景缺口 | 原稿第3页行72–84讨论散射与弱小狭窄破碎区域，第21–22页行434–457分别描述两景 | EN 144 合并讨论，550 简述案例；已有文献确实覆盖复杂场景 | 恢复两景差异和观测困难的解释；保留 ROI 证据不能代表全景专项统计的限制；重写排他性文献缺口 | 每个困难对应文献、现有影像或案例位置；不把肉眼印象写成面积占比、类别精度或模块因果 |
| ⑥ 方法及图件真实性 | 原稿仍有 source prototypes、HA 内语义校准及 Mask2Former | EN 已改为可学习 canonical、融合先于 HA、OSCD，但图1–4仍有旧机制或箭头错误 | 保留当前文字对代码的纠正；四张模型图实质重绘；不以本轮表达重组为算法修改理由 | 图、公式、代码同序；不加入物理散射反演、对象匹配或未实现拓扑机制 |
| ⑦ 自然严谨写作 | 原稿 Introduction 与 Related Work 重复；审稿人 R1.1、R1.2 指出综述与成熟技术冗余 | EN 602／604 复述526的数字；630 将工作稿 TBD 流程写入结论；旧方案含大量相互冲突的完成状态 | 保留主题化综述和证据分级；讨论转向场景含义和条件；删除正文中的工作流说明、机械转折及反复自我辩护 | 每段一个中心信息；解释有来源；中英文含义与结论强度一致；TBD 管理留在方案 |

### 1.2 对历史方案与审稿意见的处理

替换前 `revise_plan.md` 的 SHA256 为 `1815685e57e4376c5b9de29f81cf31d5afbd75fc02d97ba1bfaba31d19d0e88c`。本节记录替换理由，以下旧要求均不再是执行指令：

- JAG／IJAEOG 目标、TGRS 目标及 IEEE 模板迁移；当前唯一目标为 RSE。
- 以 Sentinel-1→GF3 迁移定义全文 research gap，以及未经排除重叠就称郑州为独立测试。
- 标题冻结、六段引言冻结、八／九节方法、旧图号不动或仅靠 caption 修复旧模型图。
- 将 `74.23%/86.71%` 的聚合表一致性等同于原始评估证据闭合。
- 允许模拟数值加上 TBD 后进入摘要、结果解释或结论；新增消融、分层统计及扰动实验的任务安排。
- 将覆盖和平滑视为水深准确性，或将 Brazos 条件水深评价当成 HA-CQI 端到端评价。

最初修改方案中对表格规律性、照片证据、范围误差传播、地形尺度和连接假设的担忧仍有效；其水文学主导路线、额外实验清单及旧模型细节不沿用。AE 关于研究问题不清、算法细节遮蔽贡献的意见转化为本方案结构和写作标准；“解释为什么有效”只能由已有文献、代码设计与现有结果分别支撑，不编造消融结论。

证据等级统一为：**F＝当前可核查事实；D＝方法设计目的；I＝受约束的解释；H＝待验证假设；B＝来源或版本未闭合的阻断。** 整体指标仅评价完整方法，不能自动升级为单个模块的作用证据。

## 2. 新论文骨架与迁移映射

### 2.1 固定章节结构

| 章／节 | 主要任务 | 从当前稿迁入／压缩的内容 | 章间承接 |
|---|---|---|---|
| 1 Introduction | 风险和观测需求、复杂景观问题、已有研究、具体设计切入及贡献 | 保留综述融入引言；压缩单列技术谱系 | 从研究问题转入可观察这些问题的资料与场景 |
| 2 Study Areas and Data | 2.1 Study Areas and Flood Settings；2.2 Remote Sensing Observations and Terrain Data | 当前数据章、实验中散落的数据与标签说明；研究区图示格局、GF3观测和FABDEM用途；训练、标签及Brazos评价资料集中于4.1 | 说明数据提供的观测及限制，再进入方法处理 |
| 3 Methodology | 章首总述＋下述四个二级节 | 当前输入、编码、HA、CQI、OSCD、训练目标及CFDepth；成熟技术只保留必要说明 | 明确范围与水深输出不同，分别进入评价 |
| 4 Experiments and Results | 4.1 Data Configuration, Implementation, and Evaluation Protocol；4.2 Overall flood-change mapping results；4.3 Mapping performance in complex scenes；4.4 Conditional depth-estimation results | 合并实现／指标；总体比较后接两景案例，最后条件水深；不保留无实测支撑的独立消融结果节 | 从观察转入解释、应用含义和局限 |
| 5 Discussion | 5.1 Landscape-dependent mapping performance；5.2 Conditional value of depth estimates；5.3 Applicability and limitations | 合并重复数字和相关限制；删除 cross-sensor 主标题及未实测扰动解释 | 回答哪些结论可以归纳、适用到哪里 |
| 6 Conclusion | 研究问题回应、可核验发现及适用范围 | 删除模块清单、未闭合数字、工作稿状态和新增实验承诺 | 收束全文，不引入新结果 |

数据章节报告资料来源和用途，具体划分归入4.1，不把当前文件夹成员直接当成历史训练成员。实验章记录实际 checkpoint、阈值选择及有效评价域；若记录未闭合，则保留 B1–B3，不能用默认配置代替事实。

### 2.2 四个方法小节及内容归属

| 编号 | 中文／英文标题 | 合并内容 | 必须保留的复现要点 |
|---|---|---|---|
| 3.1 | 双时相 SAR 可比表征构建／Comparable Representation of Bi-Temporal SAR Observations | 输入、共享编码、语义融合、成对共享校准、局部软对齐 | 新增淹没标签及输入张量；SAR 映射与三通道复制；CNN 与 DINO 不同归一化路径；冻结方式；成对统计和可学习仿射；对齐方向 |
| 3.2 | 多尺度新增淹没证据交互／Multiscale Interaction of Inundation-Change Evidence | 四类配对输入、CQI、五级稠密变化表征 | pre／post／有向差／绝对差的次序；投影及两次 attention；每尺度分别交互，不宣称跨尺度 query 通信 |
| 3.3 | 结合区域上下文与局部细节的淹没重建／Inundation Reconstruction from Regional Context and Local Detail | OSCD、最终分类、多尺度辅助监督及完整损失 | D3–D5 上下文、D2／D1细节；二类输出；Focal、Tversky、auxiliary、support及coarse一致性项；损失权重与调度 |
| 3.4 | 范围与地形约束的一阶水深估算／First-Order Depth Estimation Constrained by Inundation Extent and Terrain | CFDepth边界、锚点、初始化、连续性细化与水深 | 支持区、DEM和永久水体资料的实现契约；锚点与下界约束；无可靠锚点时处理；NoData、时间含义和假设 |

Overview 只写章首总述段，连接 `双时相SAR → HA-CQI → 新增淹没范围 → 范围＋DEM → CFDepth`。标准骨干、FPN、SS2D与通用损失引用原文并简述采用方式；关键设计和必要公式留正文，细配置集中在紧凑实现表，训练超参数归4.1。

语义 section label 尽量保留。合并后的 `sec:encoder`、`sec:ha` 等旧定位若仍被引用，应指向相应新节或删除冗余引用并同步两稿；不能留下失效标签，也不为匹配数字机械重命名。

### 2.3 当前图件、TXT与引用映射

下表替换此前GF3图在第二章的编号安排。JPG和同名TXT采用唯一临时名两阶段迁移；图片字节内容不变，稳定label保留。方法图仍为旧版临时示意图，重绘要求继续有效。

| 当前号／内容 | 本轮迁移 | 稳定label |
|---|---|---|
| 1 研究区与照片 | figure1不变 | fig:dataset |
| 2 总体流程 | figure3→figure2 | fig:framework |
| 3 可比表征 | figure4→figure3 | fig:encoder_ha |
| 4 变化交互 | figure5→figure4 | fig:cqi |
| 5 范围重建 | figure6→figure5 | fig:oscd |
| 6 GF3影像与参考 | figure2→figure6，移至4.1 | fig:gf3_eval_data |
| 7—12 结果与诊断 | 文件名与编号不变 | 原label全部保留 |

表中迁移同时应用于.jpg和.txt。两稿includegraphics同步，四份image2提示词现为figure2—5.txt，真实资料图6保留常规排版说明。图6保持78%正文宽度及纵横比，紧接实验章观测说明；第二章不提前引用图6，以保持首次引用顺序。全仓TeX检索仅两稿直接引用这些图片；技术方案中的历史摘录不作为活动图片路径改写。


## 3. 科学叙事、贡献边界与标题摘要

### 3.1 核心问题链与文献缺口

采用以下单一叙事：

**洪水影响与制图需求 → 新增淹没的状态判别 → 跨尺度证据组织 → 区域与边缘重建 → 有效边界与条件估深 → 完整工作概述 → 三条贡献。**

首段采用总—分—总：以洪水频发与广泛影响引入研究意义，接受灾地表、人口暴露及范围／水深信息需求，最后强调及时准确获取二者对于洪水制图与灾情评估的价值。当前用L3支持灾害影响和暴露，用L13支持SAR观测；不再展开局部频次或范围增长判断，因此撤出首段L24／L25／L26引用，条目与第5节证据边界保留。首段不使用“部分地区”“一些地区”“河流”及“重要资料基础”；不将去除这些措辞理解为宣称所有洪水类型统一增长或本文方法已经具有普适性能。

I2—I5改为总分展开、段末提出未解决的信息协调要求，替换各段独立总分总及编号式问题开场。I2从灾前参考与新增淹没定义进入，评述状态、差异及邻域信息的联合使用，段末落到可比较的成对表征和变化含义；I3从淹没区域大小和形态的差异进入，按多层／多尺度提取、信息选择、地表辅助约束分类评述七篇论文，段末提出连片淹没与小片、狭长区域识别的协调；I4承接洪水变化范围重建，均衡比较专用CNN／Transformer变化检测与基础模型变化检测适配路线，段末提出区域变化判断与局部定位的协调；I5承接范围与边界到估深，比较固定范围与范围增强路线，段末提出有效高程约束和条件水深需求。

不以微弱变化检测、物理散射校正或干扰消除为I2主张；不从PNG格式推断观测没有物理来源。四段不提前介绍HA／CQI／OSCD／CFDepth，不声称上一问题已独立解决，也不将一类困难写成下一类困难的原因。文献按路线与信息关系组织，不以方法名称顺序或逐篇指标串联。Giustarini与DAM-Net提供灾前参考及语义变化路线，AWCA-Net在I2简述关系、I3说明尺度处理，I4不再重复引用它支撑另一缺口；SMAGNet与SWOT路线撤出本轮引言以集中主题，原BibTeX条目及其余章节引用保留。

FwDET、FLEXTH、RS-FloodXDepth均提供已有边界处理或范围—水深路线，CFDepth只能表述为给定范围和DEM条件下的水深估算阶段，不声称首创范围连接、边界筛选或恢复漏检范围。

### 3.2 贡献表达及其证据边界

引言I6以完整段落概述研究目标、HA-CQI整体思路及CFDepth第二阶段估深关系；I7用三条贡献依次说明成对表征与判别、多尺度交互与空间重建、条件水深估算。前两条属于HA-CQI主要方法，第三条为水深估算阶段。C1—C3继续用于内部证据追踪，不把模块组合或流程连接本身列为独创。

| 层次 | 可以陈述的贡献 | 依据 | 必须避免的扩张 |
|---|---|---|---|
| C1 主要方法：可比表征与变化证据组织 | HA-CQI将成对校准和局部对应处理置于变化交互之前，用灾前后状态及差异共同组织新增淹没证据 | 当前HA/CQI代码；与L7、L8具体路径的关系 | 宣称消除了SAR散射差异、恢复了阴影／叠掩区域，或仅凭完整模型分数证明HA作用 |
| C2 主方法内部：范围重建 | 结合深层区域上下文、浅层细节与分尺度监督，构建统一的像素级新增淹没预测 | OSCD与训练目标实现；已有图7／8的可见空间表现 | 将采用SS2D、FPN或基础模型本身当创新；声称已量化提高弱对比、宽度、破碎度专项性能 |
| C3 第二阶段：条件水深 | 在给定范围和DEM条件下使用边界锚点及连续性约束产生一阶水深场 | CFDepth v2实现；既有Brazos条件比较，受B4限制 | 把上下游串联、边界筛选或连通域处理称为首创；泛化为城市实测水深或端到端精度 |

全文的“学到什么”来自已追溯结果和现有案例：制图遗漏／误报怎样分布、两景的表现有何不同、条件水深的覆盖和误差意味着什么。设计目的与解释分别使用D、I；没有模块实验的因果解释标H并从正式结论中排除。若某贡献没有足够结果支撑，减少结论强度，不用凑足三条的模板制造发现。

### 3.3 已确定标题与引言用语

2026-09-06用户确认以下标题，替换本节原五类候选及此前以Complex Landscapes和DEM-Constrained修饰的工作标题；不再将旧候选作为待选择的执行方案。

- **英文：HA-CQI for Bi-Temporal SAR Flood Change Detection with Change-Defined Depth Estimation**
- **中文：HA-CQI 双时相 SAR 洪水变化检测及新增淹没范围内的水深估算**

两稿的`\title`及`pdftitle`同步。标题前置HA-CQI及变化检测，水深作为水深估算阶段；Change-defined指由变化检测确定的新增淹没范围，不表示水深随时间的变化，也不表示HA-CQI直接回归水深。标题省略DEM约束，正文仍保留CFDepth的地形输入、给定范围和一阶估深假设。依据为用户确认文本、`0626SCI题目润提示词.docx`的贡献主次与准确表达约束，以及`CFDepth/CFDepth_0826.txt`开头对输入和floodSupport的定义。

引言沿用“复杂地表背景”（complex surface conditions）。以下是7.6标题用语调整时的参考句，现仅保留术语依据，不再冻结为首段末句：

> Identifying newly inundated land from SAR and deriving complementary depth information from terrain can support flood impact assessment under complex surface conditions.
>
> 利用 SAR 识别新增淹没，并结合地形获取补充水深信息，可为复杂地表背景下的洪水影响评估提供支持。

本轮保持标题和complex surface conditions用语不变；原先冻结I2—I8散射主线、八段及首段末句位置的要求失效，由4.2八个正文单元结构取代。英文原1200—1400词目标为此前起草预算；本次拆段及衔接调整后为1400词，包含三条编号贡献，不为凑词数增写其他段落。标题不增加性能、跨传感器泛化或范围与水深同时准确的承诺。

### 3.4 摘要：参照首稿结构的七个逻辑单元（已重写）

本轮以首稿PDF第1—2页摘要为结构参考，恢复“信息需求—科学困难—方法回应—评价结果—应用意义”顺序，与新版Introduction对应。原摘要的城市限定、模拟IoU、未经核验的GF3覆盖比例及照片区间一致性不恢复；当前稿末尾集中的审核／来源限制清单改为对应结果句的必要限定。英文为一个连贯段落、230词（计入数字，连字符词计一词），中文同步；230—250词为用户确认写作目标，不作为已核验的RSE硬性规定。

| 单元 | 目的与中心信息 | 实际证据及对应引言 | DOCX约束与审查 |
|---|---|---|---|
| A1 需求 | 快速洪灾评估需要淹没范围与水深 | 首稿摘要开头；新版I1 | W-I／W-C；解除城市限定，不宣称已验证决策收益 |
| A2 检测困难 | 复杂地表背景影响时相特征比较；大范围与小面积或弱对比、狭窄、破碎形态共存 | I2—I4；现有SAR与参考面板 | W-I／W-G；不把弱对比、小面积与破碎形态混为一类 |
| A3 下游边界 | 新增淹没排除原有水体，并非所有边界都是干湿岸线高程约束 | I5；CFDepth v2输入及边界定义 | W-G／W-C；不宣称恢复漏检范围或反演水深变化 |
| A4 主要方法 | HA-CQI先协调成对表征，再联合状态与差异，经区域上下文和局部细节重建范围 | I6；现有实现 | W-C；动作顺序真实，不列骨干、层号或查询数；无组件效果归因 |
| A5 水深估算阶段 | 给定范围、有效边界与DEM支持一阶条件估深 | I5、I6；CFDepth v2 | W-C；HA-CQI承担主要检测贡献，CFDepth承担第二阶段估深；保持地形与支持区条件 |
| A6 评价与结果 | 郑州／涿州参考图描述形态；Brazos既有产物比较报告覆盖比例与MAE | 图7—8参考列；图12及表4；`demo/script/depth_validation_table.csv`和`quantitative_evaluation.py` | W-R；场景观察不变成检测性能；产物比较不变成当前实现验证 |
| A7 意义 | 新增淹没分布和给定范围内部的条件水深提供互补信息 | I1、I6及实际输出定义 | W-C；不保证范围、水深同时准确，不加入实测应用效益 |

摘要不放TBD；所有未闭合事项继续记入B1—B8。已核对数值为：正水深参考域内CFDepth标记产物覆盖95.81%、FwDET覆盖41.91%；共同有效像元MAE分别为1.016 m、1.673 m。数值只对应现有Brazos产物；相关性低且NSE均为负的限制置于同一结果句中。正文及原始CSV的其他指标保留，本轮未重算栅格或派生提升百分比。

实读`前言写作提示词-新.docx`、`创新与贡献-英文版.docx`和`3结果与讨论写作提示词_最后.docx`的动机、贡献分层与证据约束；用户确认的段落顺序和范围优先，不套用DOCX中的额外分析、文献检索或实验扩展任务。

## 4. DOCX约束与逐段实施规划

### 4.1 实读提示词映射

本会话已用DOCX的 `word/document.xml` 提取并读取写作约束；以下P指文档段落位置，若空段计数造成偏差，以关键词定位。每节和每段在起草前回看对应约束；提示词示例不能充当文献或实验事实。

| 代码 | DOCX及实读内容 | 适用章节 | 执行约束 |
|---|---|---|---|
| W-T | [0626SCI题目润提示词](0626SCI题目润提示词.docx)：最小贡献单元、最高贡献层次、五种策略（P1–60、72–125） | 标题及摘要对象定位 | 贡献载体优先；方法、对象和范围明确；不堆指标或案例名；不将词数偏好当期刊规定 |
| W-I | [前言写作提示词-新](前言写作提示词-新.docx)：背景／挑战／分类研究／贡献对应（P40–165及后部约束） | 引言、摘要动机、数据章过渡 | 先核对贡献与问题关系；核心文献须研究同类问题；不能以本文结果倒写领域空白 |
| W-L | [前言写作-文献综述写作、引用提示词](前言写作-文献综述写作、引用提示词.docx)，全文27个body段落 | 引言文献段及讨论对照 | 按问题分类；说明研究关系；不逐篇报结果；引用与论点直接对应，禁止虚构 |
| W-G | [research gap总结](<research gap总结.docx>)：已做／尚缺／启发／提出／评价／贡献 | 引言末段、摘要和结论呼应 | 缺口与贡献相配；“反推”仅作检索思路，必须由文献核验；不预报性能，不凑不存在的不足 |
| W-C | [创新与贡献-英文版](创新与贡献-英文版.docx)：做了什么／提出什么／学到什么／为何重要 | 方法、贡献、摘要、结论 | 区分技术采用、设计和发现；贡献依证据；不能为三条模板制造创新；英文自然凝练 |
| W-R | [3结果与讨论写作提示词_最后](3结果与讨论写作提示词_最后.docx)：目的、设置、观察、解释与含义（P1–53及后部约束） | 数据、实验、结果、讨论 | 每段一个论点；实测与图像观察分开；无依据的解释标假设；不以缺材料为由造数 |
| 排除 | [Highlights](Highlights.docx)，实际内容是MT-InSAR／DR-GAT火后滑坡 | 不用于本论文内容 | 不移植科学内容；若未来需要RSE highlights，基于本稿已验证贡献另写 |

用户的证据与范围要求优先：不用机械让步转折，不逐段重复限制声明，不照搬DOCX中的词句模板。本轮先确定中文科学问题与表述，再同步英文；两稿按同一逻辑单元保持含义、引用及结论强度一致，不逐词硬译。

### 4.2 Introduction：背景、四个问题、工作概述与三条贡献（八个正文单元）

当前正文为八个单元：I1、I2、I3、I4a、I4b、I5、I6、I7。I4a／I4b分别对应新第四／第五段，保留原I5水深、I6工作概述、I7贡献ID，其正文位置顺延为第六／第七／第八部分。两段共同服务变化范围重建问题，不增加独立贡献。此前已删除的引言末章节说明不恢复。

#### 4.2.1 修改得失与模板借鉴

首稿的背景、变化辨识、多形态洪水与范围—水深联系保留；原Related Work有效内容融入问题段，不恢复独立章节。前轮稿与当时磁盘计划基本一致，本轮属于对计划的科学叙事修订，不记为未按旧方案执行。撤去独立散射综述和I6—8反复总结，恢复问题主题句与明确的方法回应。

实读[Graph-Enhanced模板](<reference template/Graph-Enhanced.md>)的背景、极化分类综述、集中三项瓶颈和三条贡献。借鉴其“总述—论证—瓶颈与贡献对应”，不宣称模板原来就每段一个问题，不移植极化机理或标签稀缺课题。

#### 4.2.2 逐单元执行与审查

| 单元／目的 | 证据位置 | 中心信息 | 承接关系与编号回应 | DOCX | 强度／审查 |
|---|---|---|---|---|---|
| I1 背景与意义 | L3、L13、L23（SAR散射原理）、L9（范围—地形估深）；L24—26仅保留证据记录 | 洪水频发与广泛影响—受灾地表及暴露—范围与水深互补—观测条件—SAR主动成像、散射方向与暗色响应—范围信息与水深的区别—精细制图及评估价值 | 以及时准确获取范围和水深的需求承接I2辨识问题 | W-I、W-L | 不以地域限定削弱研究对象，也不宣称各类洪水统一增长；及时准确是应用要求而非已验证性能 |
| I2 新增淹没判别 | L4、L7、L8、L26；HA／CQI代码 | 灾前参考→状态与差异联合路线→成对比较和淹没含义需求 | 段末需求引出I3大小与形态差异；对应I7① | W-I、W-L、W-G | 总分展开，问题留段末；不声称微弱变化专项性能或显式状态分类 |
| I3 多尺度与洪水形态 | L27—L29多层／多尺度提取；L30／L8信息选择；L15／L31地表辅助约束 | 目的：解释大小与形态差异；中心信息：多尺度提取、特征选择与地表条件互补，段末归纳大小区域的兼顾要求 | 承接I2判别需求，引出I4范围内部与边缘重建；对应I7② | W-L：超过5篇分类、同类连续引用、无精度罗列；W-G：缺口具体且非排他 | 七篇中五篇为2022—2025年；承认已有多尺度方法，不将注意力或区域专门化等同于形态适配，不作专项效果结论 |
| I4a 专用变化检测（正文第四段） | L41—L44及L32 | 保留原开头及五篇评述；SemT-Former后引出大规模预训练表征的使用 | 承接I3尺度形态，段末引出I4b；共同对应I7② | W-L：方法关系，不造技术继承 | 不暗示专用网络均未使用预训练，不增加缺口 |
| I4b 基础模型适配（正文第五段） | L45—L49 | 适配与特征整合→五篇基础模型变化检测→SAR洪水适用性及区域—局部协调需求 | 接I4a，现有结尾承接I5水深段；共同对应I7② | W-L、W-G：不称基础模型天然优越 | 十篇引用及评述总量不变，任务差异不等于创新 |
| I5 条件水深估算 | L9—11、L20；CFDepth | 承接范围与边界→边界赋值及范围增强→有效约束筛选 | 段末提出给定范围内的条件估深；对应I7③ | W-L、W-G | 不宣称首次边界处理或CFDepth恢复漏检范围 |
| I6 研究工作概述 | 当前HA-CQI与CFDepth方法正文、指定AWCA-Net引言结构 | 完整交代两阶段制图目标→第一阶段检测→范围输入→第二阶段条件水深；不列性能数字 | 接I5估深需求，引出I7贡献 | W-C：总体概括，区分流程与创新 | 只说明真实设计和用途，不引入独立模块效果 |
| I7 三条贡献及承接 | I2—I5问题、现有实现与证据边界 | ①成对表征与变化判别；②多尺度交互和区域—细节重建；③CFDepth条件估深 | 前两条为主要检测方法，第三条为水深估算阶段；不恢复已删除的章节承接句 | W-C：三条均不加粗；紧凑贡献句，说明提出什么及针对什么问题 | 不称首次、最高精度或已验证专项作用；不新增引用 |

本轮起草前重读W-I P40—165、W-L全文、W-G P1—150及贡献规则、W-C P350—430；用户确认的I6完整概述与I7三条贡献取代此前四条回应；中文先行后同步英文。I2—I5以研究路线总述开篇、段中展开、段末留问题；当前以7.13取代前轮总分总要求。I3的大小与形态差异和I4的内部—边缘组织分别审查，保留论证完整度而不反复解释模块。

### 4.3 Study Areas and Data：两个小节、四段（当前实施）

| 段号／目的 | 证据位置 | 中心信息 | 承接关系 | DOCX | 强度 |
|---|---|---|---|---|---|
| D1 位置与郑州场景（2.1） | 图1位置、(a)、P1—P4 | 两地事件与位置；郑州地表格局及局部受灾环境 | 转入涿州 | W-I、W-R | 图示观察，不推断SAR细节恢复能力 |
| D2 涿州与场景联系（2.1） | 图1(b)、P5—P8、Lan2025 | 地表交错、局部受灾环境及区域背景；正向概括两种场景 | 进入观测资料 | W-R | 照片用于示例，图注保留一次区间未核验说明 |
| D3 GF3观测（2.2） | 表1原有参数 | 日期、模式、极化及采样间距，灾前后观测用途 | 衔接地形输入 | W-R | 原参数不改，元数据缺口保留 |
| D4 FABDEM（2.2） | Hawker2022、既有约30 m记录 | 产品性质、地形及边界高程作用，范围与地形共同支持估深 | 进入方法 | W-R、W-C | 不增加未知版本或垂直基准 |

本四段规划替换此前七段。训练资料、标签、划分、预处理和Brazos评价统一位于4.1；第二章只保留场景及观测、地形资料。`sec:depth_data`仍为2.2，方法末尾涉及评价产物来源的引用改指4.1。资料表1、方法配置表2、总体比较表3、水深比较表4。

### 4.4 Methodology：总述及四节逐段安排

| 段号／目的 | 证据位置 | 中心信息 | 承接关系 | DOCX | 强度 |
|---|---|---|---|---|---|
| M0 总述 | HA-CQI组装代码；CFDepth接口；新图2 | 两时相输入、范围输出和条件水深的上下游关系 | 接3.1观测表征；不独立设Overview二级节 | W-C、W-I | F／D |
| M11 定义输入与任务 | EN162；数据读入和归一化代码 | 单波段SAR复制三通道不产生额外极化信息；输出新增淹没概率／mask | 为M12共享表征建立符号 | W-C | F；标签语义受4.1／B6约束 |
| M12 共享编码与语义辅助 | semantic_encoder、dino_adapter；新图3 | CNN空间信息与冻结DINO语义辅助，融合先于HA | 接M13成对校准 | W-C | F；光学预训练先验是采用技术，不是SAR散射机制 |
| M13 成对共享校准 | harmonized_alignment；EN218起 | P1／P2配对统计映射到可学习canonical仿射空间；保留必要公式 | 接M14局部对应处理 | W-C | F／D；不是训练集累计prototype |
| M14 局部对齐与输出 | deformable_alignment；实际运行配置 | 灾后query、灾前key/value及灾前残差校正；说明启用层级 | 接3.2变化证据交互 | W-C | F／D；不声称恢复不可观测区 |
| M21 配对证据定义 | change_query_interaction；新图4 | 灾前、灾后、有向差和绝对差共同形成输入 | 接M22双向交互 | W-C | F／D |
| M22 交互运算 | CQI两次attention及投影实现 | query读取tokens、tokens读取query；保留必要公式 | 接M23五尺度输出 | W-C | F；query无预定义物理类别 |
| M23 多尺度变化输出 | ha_cqi、CQI；D1–D5接口 | 各尺度分别产生稠密变化特征，没有跨尺度query通信 | 进入3.3空间重建 | W-C | F／D |
| M31 区域上下文 | OSCD深层路径；新图5 | D3–D5区域聚合和四方向SS2D | 接M32浅层恢复 | W-C | F／D；注明已有技术来源 |
| M32 局部细节及输出 | OSCD浅层路径与二类head | D2／D1恢复局部信息并形成洪水／背景输出 | 接M33训练约束 | W-C | F／D；不宣称显式宽度或拓扑保持 |
| M33 完整训练目标 | engine.py239–334及实际options | 主损失、辅助监督、support及coarse一致性；保留调度与依赖 | 说明检测训练完成，再转下游3.4 | W-C | F／D；不遗失非标准损失，也不宣称专项效果 |
| M41 支持区与边界 | CFDepth_0826.txt；B4；新图2水深插图 | 输入范围、DEM、永久水体辅助条件与边界筛选 | 接M42锚点和初始化 | W-C | F／D；区分GEE与Python实现 |
| M42 锚点与初始水面 | GEE v2及Python对应代码 | 湿／干侧高程线索形成锚点，范围内传播只作初始化 | 接M43连续性细化 | W-C | F；锚点不是实测水位 |
| M43 连续性与深度输出 | 固定锚点／DEM下界；d=max(S−Z,0) | 连续水面场细化后减DEM；记录无锚点回退和NoData | 接M44适用条件 | W-C | F／D；不是每分量单一常水位 |
| M44 假设与时间含义 | 实现及review R2；L9–L12 | 图像连通不等于水力连通；成像时刻与峰值区分；DEM和范围误差可传递 | 引出第四章检测及条件深度分别评价 | W-C、W-R | F／I；没有扰动实测则不写传播量或方向结论 |

### 4.5 Experiments and Results：七段

| 段号／目的 | 证据位置 | 中心信息 | 承接关系 | DOCX | 强度 |
|---|---|---|---|---|---|
| R1 实验协议（4.1） | options、selection、infer_report；B1–B3 | 实际训练配置、比较方法、checkpoint、阈值来源、整景有效域与重叠处理 | 接R2评价量定义 | W-R、W-C | F／B；不照搬参考论文划分或硬件设置 |
| R2 评价口径（4.1） | 既有评估代码；EN494–520 | 检测P／R／F1／IoU；OA辅助；深度覆盖与共同有效域误差分开；WSE图来源未闭合而停用 | 为R3总体比较定口径 | W-R | F；不新增指标计算，不混用Pearson r²与NSE |
| R3 总体检测比较（4.2） | 主表及对应原始报告；B1／B2 | 只报告来源闭合的完整模型表现，结合P／R解释误报漏检平衡 | 接R4／R5空间位置解释 | W-R | F／B；主数字未闭合则该定量段停止回填 |
| R4 郑州现有案例（4.3） | 新图7的灾前后、标签及TP／FP／FN面板 | 描述所示狭窄、细碎或低对比位置的遗漏、误报及边缘差异 | 与R5混合景观对照 | W-R | 限定图像观察；不转为全景小目标召回或HA机制 |
| R5 涿州现有案例（4.3） | 新图8及对应资料 | 描述大连通区域、农田／河流邻近区及破碎边缘的可见差异 | 从范围观察转到R6下游水深 | W-R | 限定图像观察；不由两景差异隔离地类或洪水成因效应 |
| R6 GF3条件深度（4.4） | 新图9、10、11；B4／B5；照片B6 | 说明范围内深度分布、可用性和连续性诊断的含义，按此顺序引用图 | 接R7参考深度比较 | W-R | 无可靠来源仅保留受限展示；删除Panel A模拟差值与推论 |
| R7 Brazos条件比较（4.4） | 新图12、现有CSV／脚本和Panel B；B4 | 按既有脚本的正参考域／共同有效域分别报告覆盖和误差，保留低相关、负NSE及产品版本限制 | 接第五章解释和适用范围 | W-R、W-C | 有记录的产物评价；不是HA-CQI端到端或城市实测验证 |

当前没有可核验的模块消融结果，正文已不保留“责任消融”独立结果节，不展示模拟行；其缺口在B5登记。本轮不以新增实验填补。4.4保留现有可追溯Brazos面板；模拟扰动Panel C已从共享水深表及正文移除，未扰动行不重复作为另一项实验。

### 4.6 Discussion与Conclusion：按最终段落同步

| 段号／目的 | 证据位置 | 中心信息 | 承接关系 | DOCX | 强度 |
|---|---|---|---|---|---|
| S1 两景制图需求（5.1） | R4--R5、D1--D2 | 郑州局部与涿州区域、边界需求并存 | 接S2设计路径 | W-R | 限定观察，不作出现频率统计 |
| S2 设计与已有方法（5.1） | HA/CQI/OSCD；AWCA-Net与RAFNet | 信息组织不同，不据此证明组件收益 | 接S3下游方法意义 | W-C、W-R | D／I |
| S3 条件水深方法（5.2） | M41--M44；FwDET/FLEXTH/RS-FloodXDepth | 干湿界面与地形约束可形成条件信息 | 接S4评价含义 | W-L、W-R | F／D，不称水动力求解 |
| S4 Brazos结果解释（5.2） | R7及既有CSV | 覆盖、误差、相关与NSE不同；负NSE明确限制 | 接S5可观测性 | W-R | 产物级F；B4版本归因阻断 |
| S5 观测、标签与外推（5.3） | Wagner/FloodPlanet/Portalés；B3/B6 | SAR物理限制、标签尺度、来源及重合限制 | 接S6地形与范围 | W-L、W-R | F／B，不承诺泛化 |
| S6 范围与地形限制（5.3） | CFDepth实现；Travert 2026；B4/B6 | 亚像元地形、范围误差、锚点和水力联系限制 | 接结论 | W-R、W-C | 文献与设计依赖；无本稿扰动趋势 |
| C01 回答问题 | I6、M0--M44 | HA-CQI新增淹没识别与CFDepth条件水深的关系 | 接C02发现与边界 | W-C、W-G | F／D |
| C02 发现与范围 | R4--R7、S1--S6 | 选定空间案例与既有水深产物的有限发现 | 全文结束 | W-C、W-R | 不含未闭合检测指标、组件因果和新增实验承诺 |

I/D/M/R/S/C的逻辑写作单元按各表定位；本轮I由八段调整为六个逻辑单元，原“42”不再作为固定总数。方法公式前后说明允许分段，不将规划单元数误当最终TeX空行段落数。两稿以相同逻辑单元和等价公式组织。数据与结果中的TBD是工作稿的具体来源占位，结论不写工作流。

## 5. 文献证据表与参考实验的借鉴边界

### 5.1 已核验来源与适用范围

核验级别A：本会话已读出版正文或本地全文中的相关方法／结论；B：已核验出版页、摘要或原文片段，足以支持表内窄论点，不足以证明“全文完全没有某机制”。不以期刊声望替代论点对应。新增正式引用已写入`reference.bib`并同步两稿。正式版本与作者预印本的读取范围分开登记；全文未取得时只引用摘要直接支持的窄论点。作者全表以已核验BibTeX条目为准。

| ID／核验 | 准确来源、年份与DOI | 可支撑论点 | 适用边界／禁止外推 | 拟引用位置 |
|---|---|---|---|---|
| L1／A | Madakumbura, G. D., Thackeray, C. W., Norris, J., Goldenson, N., Hall, A. (2021). **Anthropogenic influence on extreme precipitation over global land areas seen in multiple observational datasets**. Nature Communications 12,3944. [10.1038/s41467-021-24262-x](https://www.nature.com/articles/s41467-021-24262-x) | 多观测资料中年最大日降水包含人为影响信号 | 对象为Rx1day，不能直接推导全球各类洪水频次、面积或两次个例归因 | 历史背景依据，本轮未用于引言 |
| L2／A | Blöschl, G., Hall, J., Viglione, A., et al. (2019). **Changing climate both increases and decreases European river floods**. Nature 573,108–111. [10.1038/s41586-019-1495-6](https://www.nature.com/articles/s41586-019-1495-6) | 欧洲河流洪水变化方向具有区域差异 | 欧洲河流流量研究，不代表所有洪水类型、地区及淹没面积趋势 | 历史审计保留；已撤出引言 |
| L3／A | Tellman, B., Sullivan, J. A., Kuhn, C., et al. (2021). **Satellite imaging reveals increased proportion of population exposed to floods**. Nature 596,80–86. [10.1038/s41586-021-03695-w](https://doi.org/10.1038/s41586-021-03695-w) | 卫星淹没观测对人口暴露分析的价值；事件样本为2000–2018年913次大洪水 | 人口变化分析时期为2000–2015；不将累计范围当年度面积趋势，不将人口增长当危险度或资产价值增长 | I1 |
| L4／B | Giustarini, L., Hostache, R., Matgen, P., Schumann, G. J.-P., Bates, P. D., Mason, D. C. (2013). **A Change Detection Approach to Flood Mapping in Urban Areas Using TerraSAR-X**. IEEE TGRS 51(4),2417–2430. [10.1109/TGRS.2012.2210901](https://doi.org/10.1109/TGRS.2012.2210901) | 城市洪水变化检测已有研究，参考影像有助于区分阴影、光滑地表和永久水体 | 高分辨率TerraSAR-X与具体场景条件；出版年2013，不能以DOI年份替代 | I2；原方法与讨论引用保留 |
| L5／B | Mason, D. C., Giustarini, L., Garcia-Pintado, J., Cloke, H. L. (2014). **Detection of flooded urban areas in high resolution Synthetic Aperture Radar images using double scattering**. IJAEOG 28,150–159. [10.1016/j.jag.2013.12.002](https://doi.org/10.1016/j.jag.2013.12.002) | 淹水建筑附近可能出现增强散射；观测几何、阴影和叠掩影响城市可见性 | 结合高分辨率LiDAR与散射模型；不能写城市洪水必然变亮，不能说HA-CQI显式反演双程散射 | 历史散射背景依据，不用于本轮引言 |
| L6／B | Tsyganskaya, V., Martinis, S., Marzahn, P., Ludwig, R. (2018). **Detection of Temporary Flooded Vegetation Using Sentinel-1 Time Series Data**. Remote Sensing 10(8),1286. [10.3390/rs10081286](https://www.mdpi.com/2072-4292/10/8/1286) | 临时开阔水与被淹植被响应不同，多时相与辅助地类资料有作用 | 不能认定本稿涿州农田均为被淹植被；时间序列和极化条件不等同单通道复制输入 | 历史散射背景依据，不用于本轮引言 |
| L7／B | Saleh, T., Weng, X., Holail, S., Hao, C., Xia, G.-S. (2024). **DAM-Net: Flood detection from SAR imagery using differential attention metric-based vision transformers**. ISPRS JPRS 212,440–453. [10.1016/j.isprsjprs.2024.05.018](https://doi.org/10.1016/j.isprsjprs.2024.05.018) | SAR洪水已有专门差分注意和多尺度表示路线 | 不能把双时相、attention或语义token存在本身列为独创；不能以该文整体分数证明本文模块优越 | I2 |
| L8／A | Du, M., Shao, Z., Xiao, X., Zhang, J., Zhu, D., Wang, J., Balz, T., Li, D. (2026). **High-precision flood change detection with lightweight SAR transformer network and context-aware attention for enriched-diverse and complex flooding scenarios**. ISPRS JPRS 231,507–531. [10.1016/j.isprsjprs.2025.11.011](https://doi.org/10.1016/j.isprsjprs.2025.11.011) | AWCA-Net已明确处理细微、异质变化及复杂洪水形态；全文支持具体信息路径比较 | 不宣称前人忽略复杂场景；其多尺度权重不能随意改称逐像素选核；不照搬实验结论 | I2—I4、第5.3节 |
| L9／B | Cian, F., Marconcini, M., Ceccato, P., Giupponi, C. (2018). **Flood depth estimation by means of high-resolution SAR images and lidar data**. NHESS 18,3063–3084. [10.5194/nhess-18-3063-2018](https://nhess.copernicus.org/articles/18/3063/2018/) | SAR范围结合高程估深已有直接研究 | LiDAR条件不能支撑本稿其他DEM的城市水深准确性；范围至水深联接并非首创 | I5、S4 |
| L10／A | Cohen, S., Raney, A., Munasinghe, D., et al. (2019). **The Floodwater Depth Estimation Tool (FwDET v2.0) for improved remote sensing analysis of coastal flooding**. NHESS 19,2053–2065. [10.5194/nhess-19-2053-2019](https://nhess.copernicus.org/articles/19/2053/2019/) | 边界水位分配；v2.0已讨论海岸、永久水体和传播约束 | 不说FwDET从未处理永久水体；ArcGIS与QGIS实现不能混称，须核对本稿比较版本 | I5、M41、S4 |
| L11／A | Betterle, A., Salamon, P. (2024). **Water depth estimate and flood extent enhancement for satellite-based inundation maps**. NHESS 24,2817–2836. [10.5194/nhess-24-2817-2024](https://nhess.copernicus.org/articles/24/2817/2024/) | FLEXTH处理no-data／永久水体相关缺口；有效边界、坡度、连通区域及回退有先例 | 不把这些单项写成CFDepth独有创新；论文的Brazos掩膜敏感性不等于本文已做实验 | I5、M41—M44、S4—S5 |
| L12／B | Yang, Z., Yuan, W., Ran, W., Liu, C., Adriano, B., Shibasaki, R., Koshimura, S. (2026). **Flood Depth Mapping from SAR Imagery Using CS-Mamba with DEM Sensitivity Analysis**. ISPRS Annals XI-3-2026,493–500. [10.5194/isprs-annals-XI-3-2026-493-2026](https://isprs-annals.copernicus.org/articles/XI-3-2026/493/2026/) | SAR分割连接FLEXTH并考察DEM，说明已有上下游应用路线 | Annals会议论文集不是ISPRS JPRS；不能代替本稿水深参考证据；不声称本轮已完整读其PDF | 未进入本版正文；保留为邻近路线 |

骨干、DINOv3、FPN、可变形注意、SS2D及Focal／Tversky等技术文献在方法中保留准确来源，已结合当前BibTeX与原始方法核对，不为压缩方法而删除必要归属。只有出版来源已核验的条目才能写成正式文献；重复DOI、作者年后缀和引用key已同步两稿处理。

### 5.2 本轮近期文献检索、读取范围与正式引用

检索窗为**2024-01-01至2026-09-05**，重点2025--2026及截至该日已上线的Article in Press。RSE、Journal of Hydrology、ISPRS Journal of Photogrammetry and Remote Sensing和Journal of Remote Sensing（Science Partner Journal）分别检索；同时纳入直接影响范围至水深论点的WRR和NHESS。关键词组合覆盖SAR flood、urban/vegetated/heterogeneous landscapes、bi-temporal change、benchmark/labels/generalization、flood depth/DEM/incomplete boundaries。只将出版原文、出版摘要、作者稿和DOI元数据作为依据；推荐文章、预印本和正式版分开。未获得可证明的精确上线日时明确留TBD，不用DOI年份、Crossref登记日期或卷期月代替。该日期缺口不改变已核验正式来源的窄论点引用资格。

L1--L12保留经典背景与已有复杂场景证据；L2仅作历史审计，不再分配引言位置。L8 AWCA-Net已正式发表于2026年，是I2—I3的重要对照，不能被新文献清单替换。L12仅作邻近路线登记，不以ISPRS Annals冒称JPRS，本次不在正文引用。

| ID／BibTeX key | 作者、准确题名、期刊和年份 | DOI／原始入口 | 上线日期 | 读取范围／原文位置 | 支持论点 | 适用边界 | 正文位置 |
|---|---|---|---|---|---|---|---|
| L13／`wagner2026gfm` | Wolfgang Wagner; Bernhard Bauer-Marschallinger; Florian Roth; Tobias Raiger-Stachl; Christoph Reimer; Niall McCormick; Patrick Matgen; Marco Chini; Yu Li; Sandro Martinis; Marc Wieland; Franziska Kraft; Davide Festa; Muhammed Hassaan; Mark Edwin Tupas; Jie Zhao; Michaela Seewald; Michael Riffler; Luca Molini; Richard Kidd; Christian Briese; Peter Salamon. **The fully-automatic Sentinel-1 Global Flood Monitoring service: Scientific challenges and future directions**. Remote Sensing of Environment, 2026 | [10.1016/j.rse.2025.115108](https://doi.org/10.1016/j.rse.2025.115108) | 2025-11-04 | A：出版PDF摘要、业务体系及观测限制；PDF首页在线日期 | 三算法及辅助信息的业务体系；城市、植被、暗目标与时机限制 | 不把网络设计等同克服不可观测性 | I1、S5 |
| L14／`portales2025flood` | Enrique Portalés-Julià; Gonzalo Mateo-García; Luis Gómez-Chova. **Understanding flood detection models across Sentinel-1 and Sentinel-2 modalities and benchmark datasets**. Remote Sensing of Environment, 2025 | [10.1016/j.rse.2025.114882](https://doi.org/10.1016/j.rse.2025.114882) | [TBD: 精确上线日；已核验2025年10月卷期] | B：出版摘要的跨模态／跨数据集评价结论 | 数据集内部高分与跨数据集泛化不同 | 不能将该文结果当作本稿跨传感器已验证 | 仅S5；已撤出引言的评价限制段 |
| L15／`li2026rafnet` | Song Li; Kai Liu; Qiao Wang. **Near-real-time SAR flood detection using feature fusion and region-adaptive deep learning method**. Journal of Hydrology, 2026 | [10.1016/j.jhydrol.2026.135523](https://doi.org/10.1016/j.jhydrol.2026.135523) | [TBD: 精确上线日；已核验2026年7月卷期] | B：出版摘要的区域特征／专家分支和三类任务 | 城市与开阔区自适应处理已有直接研究 | 三类制图与本稿二值新增淹没不同；不横比精度、不以摘要推断缺失机制 | I3、S2 |
| L16／`lan2025haihe` | Ling Lan; Xiekang Wang. **Large-scale flood mapping using Sentinel-1 and Sentinel-2 imagery: Spatio-temporal analysis of the 23·7 Haihe basin-wide extreme flood**. Journal of Hydrology, 2025 | [10.1016/j.jhydrol.2025.132777](https://doi.org/10.1016/j.jhydrol.2025.132777) | [TBD: 精确上线日；已核验2025年6月卷期] | B：出版摘要的S1/S2海河流域制图及影响地类 | 海河极端洪水涉及农田与建成区的区域背景 | 流域统计不能替代本稿涿州AOI地类或面积统计 | 仅D1；研究区背景已移出引言 |
| L17／`lee2026smagnet` | Hyunho Lee; Wenwen Li. **A Spatially Masked Adaptive Gated Network for multimodal post-flood water extent mapping using SAR and incomplete multispectral data**. ISPRS Journal of Photogrammetry and Remote Sensing, 2026 | [10.1016/j.isprsjprs.2025.12.023](https://doi.org/10.1016/j.isprsjprs.2025.12.023) | [TBD: 正式版精确上线日；卷期2026年2月，预印本日期不可替代] | B：出版元数据与作者预印本摘要（arXiv:2601.00123） | SAR与不完整多光谱互补 | 灾后水体任务不等于新增淹没；未用预印本替代正式版出版信息 | I3 |
| L18／`zhang2025floodplanet` | Zhijie Zhang; Jonathan Giezendanner; Rohit Mukherjee; Beth Tellman; Alexander Melancon; Matt Purri; Iksha Gurung; Upmanu Lall; Kobus Barnard; Andrew Molthan. **Assessing Inundation Semantic Segmentation Models Trained on High- versus Low-Resolution Labels using FloodPlanet, a Manually Labeled Multi-Sourced High-Resolution Flood Dataset**. Journal of Remote Sensing, 2025 | [10.34133/remotesensing.0575](https://doi.org/10.34133/remotesensing.0575) | 2025-05-15 | B：出版摘要及标签尺度／细窄对象片段 | 高低分辨率标签和跨区域评价影响模型解释 | 不能借其细窄标签或他法结果证明HA-CQI能力 | S5；旧I4已撤出 |
| L19／`li2026swotdepth` | Zixi Li; Jiayin Xiao; Fuqiang Tian; Fuxin Chai. **Flood depth mapping with SWOT-derived training data: evaluating the potential of open-source datasets**. Journal of Hydrology, 2026 | [10.1016/j.jhydrol.2026.136000](https://doi.org/10.1016/j.jhydrol.2026.136000) | [TBD: 精确上线日；已核验2026年9月卷期，截止检索日已上线] | B：出版摘要的SWOT衍生训练参考与开源资料 | SWOT水深参考支持学习式估深的另一条路线 | 训练与观测条件不同于范围--DEM；不移植其误差 | I5 |
| L20／`tian2026floodxdepth` | D. Tian; H. Liu; L. Wang; S. Cohen; T. Mandal. **RS‐FloodXDepth: Enhancing Remote Sensing‐Derived Flood Extent and Estimating Flood Depth Using a Hydrologically Guided Region‐Growing Method and High‐Resolution DEMs**. Water Resources Research, 2026 | [10.1029/2025wr042384](https://doi.org/10.1029/2025wr042384) | 2026-06-12 | A：Wiley出版正文方法、摘要与在线信息 | 水文引导区域生长可扩展不完整范围并估深 | 撤销首次范围--水深联接／首次边界缺口等论点 | I5、S3 |
| L21／`hawker2022fabdem` | Laurence Hawker; Peter Uhe; Luntadila Paulo; Jeison Sosa; James Savage; Christopher Sampson; Jeffrey Neal. **A 30 m global map of elevation with forests and buildings removed**. Environmental Research Letters, 2022 | [10.1088/1748-9326/ac4d4f](https://doi.org/10.1088/1748-9326/ac4d4f) | 2022-02-03 | B：出版摘要及产品说明 | FABDEM的约30 m间距与林木、建筑偏差处理 | 不保证城市亚像元地形或本稿水深准确性 | D5 |
| L22／`travert2026uncertainty` | Jean-Paul Travert; Cédric Goeury; Sébastien Boyaval; Vito Bacchi; Fabrice Zaoui. **Evaluating the effects of preprocessing, method selection, and hyperparameter tuning on SAR-based flood mapping and water depth estimation**. NHESS 26, 2387--2413, 2026 | [10.5194/nhess-26-2387-2026](https://nhess.copernicus.org/articles/26/2387/2026/) | 2026-05-27 | A：出版HTML摘要、§2资料、§5.3.2及§6.1 | 输入洪水范围与处理参数影响下游水深解释 | 加龙河两个事件、模拟与水痕参考；不移植敏感性大小，不据此安排本稿新实验 | S6 |

补充元数据：L1在线2021-07-06，L2在线2019-08-28，L3在线2021-08-04，L6在线2018-08-15，均与DOI记录核对；完整作者已写入正式BibTeX。L13出版PDF由[作者机构DLR存档](https://elib.dlr.de/218412/1/1-s2.0-S0034425725005127-main.pdf)读取；L17作者稿仅支撑摘要范围的输入／任务描述。L8沿用已读本地出版PDF方法pp.510--514，DOI中的2025不是引用年份。

**保留但不进入正文的线索**：[Flood capture: A new method relying on ground-based remote sensing and surveying](https://doi.org/10.34133/remotesensing.1076)，JRS页面记录2026-09-01上线；[TBD: 尚未取得最终PDF，核对作者顺序、地面遥感方法和适用范围；出版页与机构索引作者排序有冲突，关闭前不写入正式BibTeX或机制评述]。ISPRS 2026光学--SAR异构变化及JRS在轨灾害处理属于邻近任务，未为堆砌近期引用加入主线。

当前八个正文单元引用分配：I1用L3／L13／L23／L9；I2用L4／L7／L8／L26；I3用L27／L28／L29、L30／L8、L15／L31（三组七篇）；I4a用L41—L44及L32，I4b用L45—L49（每段五篇）；I5用L9—L11／L20；I6／I7不新增引用。第四段撤出单时相分割、轮廓及标签路线，旧BibTeX条目保留；仅补正四条现有文献元数据。

本轮新增到引言的既有条目核验：

| ID／key | 准确来源及读取范围 | 支持论点／边界 | 位置 |
|---|---|---|---|
| L23／`amitrano2024sarFloodReview` | Amitrano, D.; Di Martino, G.; Di Simone, A.; Imperatore, P. (2024). **Flood Detection with SAR: A Review of Techniques and Datasets**. Remote Sensing 16(4),656. [10.3390/rs16040656](https://www.mdpi.com/2072-4292/16/4/656)；[TBD: 精确上线日；已核验2024年卷期]；出版HTML §3预处理、处理技术及复杂场景 | 阈值、统计、景观信息及SAR干扰来源的综述入口；不借综述推断本文模块效果 | I2、I4 |
| 既有技术引用／`simeoni2025dinov3` | Siméoni, O., et al. (2025). **DINOv3**. [arXiv:2508.10104](https://arxiv.org/abs/2508.10104)；[作者模型卡](https://github.com/facebookresearch/dinov3/blob/main/MODEL_CARD.md)的全局／稠密视觉表征用途 | 仅支撑预训练图像上下文来源；明确为预印本，不外推为SAR洪水有效性 | 已撤出引言，方法原有引用保留 |

本轮对Nature背景原文、出版社可读取摘要及本地AWCA-Net出版PDF既有审计、NHESS公开正文进行针对性复核。部分Elsevier全文入口403时使用出版摘要及已读原文，不扩大机制评述；SMAGNet仅沿用作者稿摘要可支持的输入与任务描述。既有精确上线日期TBD继续保留。新文献已经涉及复杂场景及不完整范围，缺口表述因此定位为本文具体问题及证据组织选择，不宣称领域首次。

### 5.3 指定AWCA-Net论文的实验组织借鉴

已读取 [用户指定本地PDF](<../参考文献/ISPRS26_changedetection_High-precision flood change detection with lig.pdf>) 的相关方法、实验与讨论；以下为印刷页码，PDF页码＝印刷页码−506。

| 原文位置 | 原文做法 | 本稿采用方式 | 不移植的内容 |
|---|---|---|---|
| pp.515–517，§4–5 | 先交代数据、方法、实施与P／R／F1／IoU | D3–D4、R1–R2集中交代既有实验身份和口径 | 随机划分、超参数、硬件和预处理条件不能照搬 |
| pp.517–522，§6.1、表3–6／图8–11 | 总体量化后以TP／TN／FP／FN局部图解释差异 | R3总体比较后接R4／R5已有案例 | 不新增五次重复、均值±标准差或显著性检验 |
| pp.522–524，§6.2、图12–13 | 按地区、成因、淹没比例及模块分组 | 只借鉴结果回应场景困难的组织原则 | 不新增任何分组统计、成因分析或分组消融 |
| pp.525–529，§6.3、图14–18 | 整景与局部结合 | 使用现有两景图说明位置与可见表现 | 不新增GRD／预处理对照，不将数据集整景展示自动称独立泛化 |
| §6.4，表8 p.529 | 组件与损失消融 | 明确本稿无可核验模块证据，限制机制结论 | 不新增HA／CQI／OSCD／DINO实验，不保留模拟消融论证 |
| §6.5–6.6，pp.527–530 | 效率及局限讨论 | 借鉴基于证据说明适用条件 | 不新增测速或轻量化卖点；不照抄各模块均有效的归因 |

该文实验规模不是本轮新增工作依据。具体方法比较与现有案例足以组织本稿问题，但不能替代尚缺的性能或机制证据。已有研究的消融结论也需按实际表格理解，不一概写每个模块对所有数据集都有增益。

### 5.4 六单元引言新增背景证据（2026-09-06）

| ID／key | 准确来源、日期及DOI | 实际读取范围／原文位置 | 支撑句与适用边界 | 引用位置 |
|---|---|---|---|---|
| L24／`feng2024globalRivers` | Dongmei Feng; Colin J. Gleason. **More flow upstream and less flow downstream: The changing form and function of global rivers**. Science 386(6727),1305—1311 (2024). 在线2024-12-12、卷期2024-12-13。[10.1126/science.adl5728](https://doi.org/10.1126/science.adl5728) | B：PubMed原始摘要与书目信息记录[PMID39666815](https://pubmed.ncbi.nlm.nih.gov/39666815/)；作者实验室出版列表交叉确认题名／作者／DOI。摘要首句给1984—2018年，第四句给最小河流百年洪水频次方向。Science出版页403，大学课件第45页仅含转载图，不能当全文 | 只写该全球分析中所研究最小河流百年洪水频率增加，此前首段曾以“部分地区”概括存在该趋势的空间范围；7.12起该引用撤出首段，仅作为历史局部极端洪水频次证据保留，不推断城市内涝、山洪或沿海洪水均有同样趋势；河流规模与洪水类型边界在此保留；不写具体增幅，不外推全部河流、全部洪水类型或因果机制。[TBD: 最终出版全文方法、百年洪水阈值定义及不确定性尚未核对；关闭前维持摘要级窄表述] | 历史I1证据记录；当前首段不引用 |
| L25／`mazzoleni2022floodExtent` | Maurizio Mazzoleni; Francesco Dottori; Hannah L. Cloke; Giuliano Di Baldassarre. **Deciphering human influence on annual maximum flood extent at the global level**. Communications Earth & Environment 3,262 (2022). 在线2022-11-01。[10.1038/s43247-022-00598-0](https://www.nature.com/articles/s43247-022-00598-0) | A：出版HTML摘要、Results“Global trends of annual maximum flood extent”、Fig.1及Results首段指标定义；106个大型流域、1985—2018年 | 季节性年最大淹没范围总体增加但有区域差异；不能等同灾害损失面积、各事件峰值或全球所有洪水统一增加 | 历史I1证据记录；当前首段不引用 |
| L26／`misra2025globalFloods` | Amit Misra; Kevin White; Simone Fobi Nsutezo; William Straka III; Juan Lavista. **Mapping global floods with 10 years of satellite radar data**. Nature Communications 16,5762 (2025). 在线2025-07-01。[10.1038/s41467-025-60973-1](https://www.nature.com/articles/s41467-025-60973-1) | A：出版HTML摘要、Results的全球地图与“Temporal analysis of flooding trends”（Fig.6／Table2）、Author information；观测窗2014-10至2024-09 | 潜在淹没范围增加；保留时间记录与气候归因限制。城市／地形等排除区限制适用范围；相对既有数据库新增检出面积不能当作实际洪水增长 | I1 |

本轮进一步核对AWCA-Net与RAFNet出版摘要、FloodPlanet出版摘要及可读取正文；相应机制评述保持已有范围。没有由新背景文献增加任何训练或验证任务。Science两轮全文路径核查未取得正文，停止其定量增幅和机制结论分支，采用可核验原始摘要的限定方向；该限制不阻断其余引言重写。


### 5.5 第三段多尺度洪水识别文献证据（2026-09-07）

本轮按W-L《前言写作-文献综述写作、引用提示词.docx》全文约束起草：超过五篇按问题与方法归类，精简而保留差异，连续引用同类研究，不逐篇列结果。用户批准的总分展开与段末问题优先；不构造时间继承关系。补检窗口为2022—2025年，同时保留正式出版于2026年的L8和L15。七篇引用中五篇发表于2022—2025年，新增三条BibTeX；其余四条复用。下表与L8／L15共同组成I3证据。

| ID／引用key | 作者、准确题名、正式出版信息与DOI | 上线日期与读取范围／原文位置 | 支持内容与段内位置 | 适用边界与审查 |
|---|---|---|---|---|
| L27／`wu2022msdeeplab` | Han Wu; Huina Song; Jianhua Huang; Hua Zhong; Ronghui Zhan; Xuyang Teng; Zhaoyang Qiu; Meilin He; Jiayi Cao. **Flood Detection in Dual-Polarization SAR Images Based on Multi-Scale Deeplab Model**. Remote Sensing 14(20),5181 (2022). [10.3390/rs14205181](https://www.mdpi.com/2072-4292/14/20/5181) | 2022-10-17；出版页检索索引的摘要、方法与Fig.8多层融合说明；本轮直接访问429，未宣称重新取得全文 | 多层水体特征与上下文融合；I3第一组 | 各时相水体提取后比较，不冒称端到端双时相检测；不引用精度，也不据其小水体结果证明本文作用 |
| L28／`zhao2023siamdwenet` | Bofei Zhao; Haigang Sui; Junyi Liu. **Siam-DWENet: Flood inundation detection for SAR imagery using a cross-task transfer siamese network**. IJAEOG 116,103132 (2023). [10.1016/j.jag.2022.103132](https://www.sciencedirect.com/science/article/pii/S156984322200320X) | 正式卷期2023-02；[TBD: 精确首次上线日尚未核验，不影响正式年引用]；出版索引元数据、Highlights与Abstract；直访403 | 孪生双时相网络的注意力和多尺度金字塔；I3第一组 | DOI年份2022不替代正式年2023；摘要不足以判定各尺度的独立效果，未作该推断 |
| L29／`tahermanesh2025siscnet` | Sahand Tahermanesh; Ali Mohammadzadeh; Amin Mohsenifar; Armin Moghimi. **SISCNet: A novel Siamese inception-based network with spatial and channel attention for flood detection in Sentinel-1 imagery**. Remote Sensing Applications: Society and Environment 38,101571 (2025). [10.1016/j.rsase.2025.101571](https://www.sciencedirect.com/science/article/pii/S2352938525001247) | 正式卷期2025-04；[TBD: 精确首次上线日尚未核验]；出版索引摘要、Introduction末段和贡献列表 | 成对共享分支并行提取不同邻域信息；I3第一组 | 不把不同滤波器存在等同于已证明各种洪水形态性能；不采用该文关于前人普遍缺失机制的概括 |
| L30／`yadav2022attentive` | Ritu Yadav; Andrea Nascetti; Yifang Ban. **Deep attentive fusion network for flood detection on uni-temporal Sentinel-1 data**. Frontiers in Remote Sensing 3,1060144 (2022). [10.3389/frsen.2022.1060144](https://www.frontiersin.org/journals/remote-sensing/articles/10.3389/frsen.2022.1060144/full) | 2022-12-14；出版全文§4.1／Fig.3、§4.2／Fig.4及正式PDF首页 | Attentive U-Net对空间与通道信息加权，与L8自适应窗口并列说明信息选择；I3第二组 | 正文明示单时相；仅引用Attentive U-Net分支，不将另一个融合DEM／永久水体的分支混为同一输入；不把注意力等同于尺度自适应 |
| L31／`zhu2025globalfloodsar` | Yuting Zhu; Kei Yoshimura; Yingying Liu; Haohuan Fu. **Global flood extent monitoring using SAR satellite and hydrological data: A multi-scale and multi-source approach**. Journal of Hydrology 663,134074 (2025). [10.1016/j.jhydrol.2025.134074](https://www.sciencedirect.com/science/article/pii/S002216942501412X) | [TBD: 精确首次上线日尚未从出版原文闭合]；出版索引作者、Highlights、Abstract；直访403 | 地表覆盖、历史水体及水文资料补充概率制图，与L15区域差异化对照；I3第三组 | 不将多源概率框架等同于神经特征尺度选择；输入条件不同，不横比指标；作者以出版页为准，作者实验室列表的末位作者差异不沿用 |

L8本轮复核出版摘要及Highlights：自适应窗口与多尺度信息处理的窄论点成立，不在I3重复I2的差异引导路线。L15沿用既有出版摘要的区域特征与专家分支证据，不由地表类别推断目标大小适配。段末大小区域兼顾是本稿的研究需求，不能由七篇引用或现有典型案例推导“所有方法仍未解决”或本文专项提升。LiST-Net、DMCF-Net等未闭合候选不用于凑数；本轮采用已取得出版方法文字的Yadav等作为信息选择补充。

### 5.6 第四段旧技术路线证据（历史记录，当前I4由5.7替代）

本节保留既有核查依据，不再作为第四段引用或写作要求；L32仍在当前I4使用，DINOv3原始论文留在方法章节。

起草前实际重读W-L《前言写作-文献综述写作、引用提示词.docx》，按“超过五篇分类、关系连接、精简且不逐篇列结果”执行。用户确认只改第四段：HA可比性仍由I2承接，CQI多尺度仍由I3承接；I4不包揽三个模块。以下九条加既有L18 FloodPlanet共十篇。除DINOv3原始技术报告外，九篇为期刊研究；不能统称十篇期刊论文。文献用于比较技术路线与条件，不以未采用HA／CQI／DINO构造创新。

| ID／key | 作者、题名、正式来源及DOI | 读取范围与日期 | 评述职责及边界 |
|---|---|---|---|
| L32／`saleh2024high` | Tamer Saleh; Shimaa Holail; Xiongwu Xiao; Gui-Song Xia. **High-precision flood detection and mapping via multi-temporal SAR change analysis with semantic token-based transformer**. IJAEOG 131,103991 (2024). [10.1016/j.jag.2024.103991](https://doi.org/10.1016/j.jag.2024.103991) | 出版页索引Abstract／Highlights；2024-07卷期，[TBD: 精确首次上线日] | 时相与语义上下文；不采用摘要关于CNN或其他方法的笼统不足，不引用性能数字 |
| L33／`jamali2024wvresunet` | Jamali, Ali and Roy, Swalpa Kumar and Hashemi Beni, Leila and Pradhan, Biswajeet and Li, Jonathan and Ghamisi, Pedram. **Residual wave vision U-Net for flood mapping using dual polarization Sentinel-1 SAR imagery**. International Journal of Applied Earth Observation and Geoinformation 127,103662 (2024). [10.1016/j.jag.2024.103662](https://doi.org/10.1016/j.jag.2024.103662) | 出版索引Abstract／Highlights及前轮作者机构原版方法；2024-03卷期，原版PDF2024-01-20上线 | 空间混合与残差编码—解码；单时相双极化SAR，不冒称双时相；不采用原文CNN只能判断有无不能定位的过度概括 |
| L34／`simeoni2025dinov3` | Oriane Siméoni et al.（完整作者与顺序见既有BibTeX）. **DINOv3**. arXiv:2508.10104 (2025). [10.48550/arXiv.2508.10104](https://doi.org/10.48550/arXiv.2508.10104) | 作者原稿arXiv页面及Abstract；2025-08-13提交，按当前可核验arXiv版本引用 | 自监督预训练提供密集表征的技术依据，不是洪水专项证据；不声称自然图像语义自动解决SAR迁移 |
| L35／`sergi2025reservoirsam` | Sergi, G. and Bocchino, F. and Ravanelli, R. and Crespi, M.. **Monitoring water reservoirs extent with Segment Anything Model applied to Sentinel imagery**. European Journal of Remote Sensing 58,2527248 (2025). [10.1080/22797254.2025.2527248](https://doi.org/10.1080/22797254.2025.2527248) | 原刊PDF首页及全文索引Abstract；2025-07-10上线 | 原版SAM+种子提示用于Sentinel-1／2水库水体；提示位置／输入条件敏感性不能直接套用DINO；非新增洪水 |
| L36／`he2024weakflood` | He, Yongjun and Wang, Jinfei and Zhang, Ying and Liao, Chunhua. **An efficient urban flood mapping framework towards disaster response driven by weakly supervised semantic segmentation with decoupled training samples**. ISPRS Journal of Photogrammetry and Remote Sensing 207,338--358 (2024). [10.1016/j.isprsjprs.2023.12.009](https://doi.org/10.1016/j.isprsjprs.2023.12.009) | 出版索引Abstract／Introduction末段；2024-01卷期，[TBD: 精确首次上线日] | SAM辅助航空洪水弱标签，非直接SAR编码；城市为该代表研究条件，不限定本文对象 |
| L37／`shokati2026sam` | Shokati, Hadi and Seufferheld, Kay D. and Fiener, Peter and Scholten, Thomas. **Rapid flood mapping from aerial imagery using fine-tuned SAM and ResNet-backboned U-Net**. Hydrology and Earth System Sciences 30,743--756 (2026). [10.5194/hess-30-743-2026](https://doi.org/10.5194/hess-30-743-2026) | 出版HTML Abstract、Introduction及方法；2026-02-09上线 | 微调SAM与U-Net用于航空洪水图像；仅比较使用方式，不将光学结果外推双时相SAR |
| L38／`xu2022waterlevelset` | Xu, Chuan and Zhang, Shanshan and Zhao, Bofei and Liu, Chang and Sui, Haigang and Yang, Wei and Mei, Liye. **SAR image water extraction using the attention U-net and multi-scale level set method: flood monitoring in South China in 2020 as a test case**. Geo-spatial Information Science 25,155--168 (2022). [10.1080/10095020.2021.1978275](https://doi.org/10.1080/10095020.2021.1978275) | 出版页索引方法及期刊目录；2021-10-29上线，正式卷期2022 | 初始水体范围供水平集细化；水体提取和后续洪水监测，不是端到端双时相CD；作者Wei Yang按原刊核对，不沿用前轮候选误写 |
| L39／`soudagar2025activecontour` | Soudagar, Rasheeda and Chowdhury, Arnab Roy and Bhardwaj, Alok. **Enhanced large-scale flood mapping using data-efficient unsupervised framework based on morphological active contour model and single synthetic aperture radar image**. Journal of Environmental Management 380,124836 (2025). [10.1016/j.jenvman.2025.124836](https://doi.org/10.1016/j.jenvman.2025.124836) | 出版索引Abstract／Highlights／CRediT；2025-04卷期，[TBD: 精确首次上线日未从当前可读出版记录闭合] | 单时相SAR活动轮廓与形态相关处理，承认已有回应；不能宣称本文采用活动轮廓或对其有已验证优势 |
| L40／`garg2023distillation` | Garg, Shubhika and Feinstein, Ben and Timnat, Shahar and Batchu, Vishal and Dror, Gideon and Gerzi Rosenthal, Adi and Gulshan, Varun. **Cross-modal distillation for flood extent mapping**. Environmental Data Science 2,e37 (2023). [10.1017/eds.2023.34](https://doi.org/10.1017/eds.2023.34) | 出版全文§3.1.1 Edge-weighted loss及§3.2 Improving weak labels；2023-11-07上线 | 跨模态蒸馏、内外边缘加权与标签条件；不宣称本模型实现这些机制 |

L18本轮复核出版摘要、研究问题及作者资料：正式上线2025-05-15；仅用高低分辨率标签比较支持监督精细度影响学习／评价，不借用其精度证明本稿能力。DINOv3作者全名单复用已核验条目，不虚构正式期刊版本。部分出版直链受访问限制，索引可读内容不登记为整篇全文；未闭合的精确上线日不替代正式卷期年份。

与实现对应：`semantic_encoder.py`先融合DINO中高层表征，`ha_cqi.py`依次执行共享编码、HA、CQI及解码；本段只为中高层语义与局部细节的后续利用提出研究需求。不增加SAM、显式边界监督、活动轮廓或拓扑模块。本轮撤销前一版以区域生长／业务化服务／光学遮挡补足十篇的第四段清单；对应历史文献保留，不再列为当前I4要求。

### 5.7 当前第四段：变化检测技术路线证据（2026-09-07）

起草前重读W-L DOCX，沿用首稿Introduction／Related Work的路线比较思路，不恢复独立Related Work。两类各五篇；先说明特征比较、语义回传和变化解码，再说明基础模型的双时相整合与任务适配。引言不提前列出本文模块；一般地物变化与SAR洪水任务明确区分，既有方法未在SAR洪水验证不等于本稿创新已经成立。

| ID／key | 作者、准确题名、正式年份与DOI | 原始来源及实际读取范围 | 本段支持内容和适用边界 |
|---|---|---|---|
| L41／`daudt2018fully` | Daudt, Rodrigo Caye and Le Saux, Bertrand and Boulch, Alexandre. **Fully Convolutional Siamese Networks for Change Detection**. 2018 25th IEEE International Conference on Image Processing (ICIP) (2018). DOI 10.1109/ICIP.2018.8451652 | [原始来源](https://rcdaudt.github.io/files/2018icip-fully-convolutional.pdf)；作者稿方法：FC-Siam-conc／diff；2018正式会议版 | 成对特征拼接或差异供密集解码；一般RGB／多光谱变化，非SAR洪水 |
| L42／`chen2021remote` | Chen, Hao and Qi, Zipeng and Shi, Zhenwei. **Remote Sensing Image Change Detection With Transformers**. IEEE Transactions on Geoscience and Remote Sensing (2022). DOI 10.1109/TGRS.2021.3095166 | [原始来源](https://levir.buaa.edu.cn/static/pdfs/2022_hao_chen_remote.pdf)；作者稿Transformer编码／解码；正式卷期2022，非DOI中的2021 | 双时相语义上下文反馈像元特征；不据此证明本稿查询机制 |
| L43／`bandara2022changeformer` | Bandara, Wele Gedara Chaminda and Patel, Vishal M.. **A Transformer-Based Siamese Network for Change Detection**. IGARSS 2022--2022 IEEE International Geoscience and Remote Sensing Symposium (2022). DOI 10.1109/IGARSS46834.2022.9883686 | [原始来源](https://arxiv.org/abs/2201.01293)；作者稿方法及会议版本；2022正式会议版 | 层次编码及差异解码；本段不重复多尺度综述 |
| L44／`han2023change` | Han, Chengxi and Wu, Chen and Guo, Haonan and Hu, Meiqi and Li, Jiepan and Chen, Hongruixuan. **Change guiding network: Incorporating change prior to guide change detection in remote sensing imagery**. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing (2023). DOI 10.1109/JSTARS.2023.3310208 | [原始来源](https://github.com/ChengxiHAN/CGNet-CD)；作者仓库及作者稿方法；正式发表2023，arXiv上传2024 | 深层变化先验引导特征融合；承认已有内部覆盖和边缘研究 |
| L32／`saleh2024high` | Saleh, Tamer and Holail, Shimaa and Xiao, Xiongwu and Xia, Gui-Song. **High-precision flood detection and mapping via multi-temporal SAR change analysis with semantic token-based transformer**. International Journal of Applied Earth Observation and Geoinformation (2024). DOI 10.1016/j.jag.2024.103991 | [原始来源](https://doi.org/10.1016/j.jag.2024.103991)；出版页索引Abstract／Highlights；2024卷期 | SAR洪水时相关系与语义变化表征；不以摘要推断其他方法没有某机制 |
| L45／`chen2023ttp` | Chen, Keyan and Liu, Chengyang and Li, Wenyuan and Liu, Zili and Chen, Hao and Zhang, Haotian and Zou, Zhengxia and Shi, Zhenwei. **Time Travelling Pixels: Bitemporal Features Integration with Foundation Model for Remote Sensing Image Change Detection**. IGARSS 2024--2024 IEEE International Geoscience and Remote Sensing Symposium (2024). DOI 10.1109/IGARSS53475.2024.10640593 | [原始来源](https://arxiv.org/abs/2312.16202)；作者稿§2.2—2.4；2023预印本、2024正式会议版 | 基础模型内双时相特征交互；不写成物理时间运动或洪水专项验证 |
| L46／`li2024new` | Li, Kaiyu and Cao, Xiangyong and Meng, Deyu. **A new learning paradigm for foundation model-based remote-sensing change detection**. IEEE Transactions on Geoscience and Remote Sensing (2024). DOI 10.1109/TGRS.2024.3365825 | [原始来源](https://github.com/likyoo/BAN)；作者稿及仓库架构说明；2024正式版 | 冻结通用表征经桥接注入双时相适配分支；桥接适配不等于HA成对校准 |
| L47／`ding2024adapting` | Ding, Lei and Zhu, Kun and Peng, Daifeng and Tang, Hao and Yang, Kuiwu and Bruzzone, Lorenzo. **Adapting segment anything model for change detection in VHR remote sensing images**. IEEE Transactions on Geoscience and Remote Sensing (2024). DOI 10.1109/TGRS.2024.3368168 | [原始来源](https://iris.unitn.it/retrieve/7ae0365e-4ddd-4000-a4bc-a65b4acebed1/TGRS3368168.pdf)；作者接受稿方法，11页；2024正式版 | FastSAM表征适配到变化检测；不是分别提示分割后直接相减 |
| L48／`wei2025ass` | Wei, Chenlong and Wu, Xiaofeng and Wang, Bin. **ASS-CD: Adapting Segment Anything Model and Swin-Transformer for Change Detection in Remote Sensing Images**. Remote Sensing (2025). DOI 10.3390/rs17030369 | [原始来源](https://www.mdpi.com/2072-4292/17/3/369)；出版索引§3方法；上线2025-01-22；本轮直链访问受限 | FastSAM与Swin特征通过适配器交互；模型间注意力不误称时相间交互 |
| L49／`dong2026peftcd` | Dong, Sijun and Hu, Yuxuan and Wang, Libo and Chen, Geng and Meng, Xiaoliang. **PeftCD: Leveraging vision foundation models with parameter-efficient fine-tuning for remote sensing change detection**. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing (2026). DOI 10.1109/JSTARS.2026.3679260 | [原始来源](https://github.com/dyzy41/PeftCD)；作者仓库Abstract／架构说明／正式引文；2026 | SAM2或DINOv3孪生编码与参数高效微调，非CLIP；[TBD: 最终卷页和精确上线日，未补造；不影响目前按DOI引用] |

除明确记录的日期外，其余文献的首次上线日不作为论据：[TBD: 如投稿元数据需要精确上线日，再查出版社记录；当前引用采用已核验正式年份，不以预印本年份替代]。稳定BibTeX key不随年份更名。PeftCD DOI由作者正式引文确认；不将其误写为CLIP路线。十篇均服务I4的变化范围重建比较，不把基础模型、语义token、全局—局部结合本身列为独创，也不声称已有专项实验验证本稿边缘效果。

## 6. 方法审计、问题回应与制图方案

### 6.1 当前实现及公式保留清单

算法事实以当前有效代码、实际运行配置和实现参考为准，不能以旧图或方便的叙述替代。以下是表达重组，不改变算法、训练数学、数据或权重。

| 对应内容 | 实现定位（仓库根相对路径） | 已核查事实与必要保留内容 |
|---|---|---|
| 主链与辅助输出 | `HA-CQI/model/architectures/ha_cqi.py:49–73` | 共享编码→HA→CQI→OSCD；辅助头接D1–D5；输入／输出定义和共享编码关系保留 |
| 编码与语义辅助 | `HA-CQI/model/modules/semantic_encoder.py:73–118`；`dino_adapter.py:43–47,71–90` | EfficientNet-B2 CNN-FPN；冻结DINOv3 ViT-S/16；层[5,8,11]融合P3–P5；固定LVD ImageNet归一化；无逐幅DINO特征z-score；配置放实现表 |
| 成对共享校准 | `HA-CQI/model/modules/harmonized_alignment.py:8–38,125–140` | P1／P2使用两时相及空间统计，映射到可学习canonical仿射，再残差恢复；保留均值／方差、可学习参数和恢复项公式 |
| 局部软对齐 | `HA-CQI/model/modules/deformable_alignment.py:101–171`；HA实际配置 | 灾后query、灾前key/value，pre／post／delta用于offset，灾前特征残差校正；P1–P3按启用配置处理；不画成两支路对称warp |
| CQI | `HA-CQI/model/modules/change_query_interaction.py:32–69` | 四类pair primitives经真实投影，query←tokens及tokens←query；默认16 queries但不作为科学创新；五尺度独立交互；两次attention及输出恢复公式保留 |
| OSCD | `HA-CQI/model/decode_heads/omni_scale_state_space_change_decoder.py:147–189,205–238` | D3–D5区域上下文和四方向SS2D，D2／D1恢复细节，二类密集head；保留区域—细节重建关系而非全算子推导 |
| 总损失 | `HA-CQI/model/engine.py:239–334`；相关运行`options.json` | 主Focal／Tversky、多尺度辅助、support-preservation、coarse-consistency及调度不得遗漏；P1／P2辅助输出还用于一致性目标，不等于仅两个aux loss |
| GEE v2水深 | `CFDepth/CFDepth_0826.txt`，尤其边界与永久水体处理段596–629 | 当前稿对应边界筛选→湿干侧锚点→水面初始化→连续性细化；保留筛选条件、锚点关系、固定锚点／DEM下界和最终深度公式 |
| 本地Python水深 | `CFDepth_inference/cfdepth.py:434–480,517,567–630`；`raster_io.py` | 最近锚点传播只作初始化；无锚点回退DEM+0.01m；固定锚点与下界下细化；输出`d=max(S−Z,0)`；网格／NoData规则清楚；不把其当GEE完整等价实现 |

需交代的运行快照示例为 B3 中的B2-OSCD运行：batch 12、80 epochs、bf16、AdamW base LR 1e-4、head multiplier 2、weight decay 5e-4、cosine；Focal类别权重0.25／0.75，辅助总权重1→0.5，Tversky beta 0.70→0.55，support 0.03、coarse 0.02，warmup 5、ramp 10。**这些只描述该记录，不等于主表所有方法的已核验协议。** 4.1已明确这些设置仅属于关联记录；不将其写成主表已核验协议。

`SOFT_ALIGNMENT=0`只关闭软对齐，成对校准仍运行，不能称为w/o HA。当前没有足以支持“无DINO／无CQI／无SS2D／去浅层监督”表格的完整运行证据。此事实用于限制描述，不构成新增代码或实验任务。

### 6.2 CFDepth实现参考冲突的处理

- 仓库说明中的 `HA-CQI-CFDepth/CFDepth/CFDepth_GEE.txt` 当前不存在；不修改AGENTS，也不假装该路径可用。
- `CFDepth/CFDepth_GEE.txt` 是旧base-depth加补全路线；当前稿方法组织以存在的 `CFDepth/CFDepth_0826.txt` v2为主要核查参考，旧文件仅作历史对照。
- GEE v2包括永久水体相关边界筛选；本地Python没有该完整旁路。输入mask已经去除永久水体，并不等于Python能够识别所有flood–permanent-water界面。论文不能合并两种实现的功能后声称均已具备。
- GEE当前输入赋值为 `image2`，头注存在 `image` 旧说明；后续复现说明以实际代码赋值为准。GEE与Python湿干侧邻域中位数均为局部窗口处理，不能写成严格按水力单元独立取样。
- 最终WSE是受支持区与高程约束的连续场，不是每个组件一个常数；栅格连接不代表真实水力连接。0.01m下界是数值处理，不是测得浅水深，也不是返回正深度即准确。
- 当前Brazos和GF3水深产物由哪个版本、支持区及参数生成仍见B4。方法表达可采用当前v2，但结果归因在记录补齐前停止。

### 6.3 场景困难—设计—既有证据—验证边界

“所需证据”只说明何种材料才能支持某类论断，**不是本轮或本计划新增实验清单**；不存在材料时保留限制。

| 场景困难 | 已实现回应 | 性质 | 现有证据可用范围 | 缺证时的处理 |
|---|---|---|---|---|
| 背景统计波动及残余位移可能形成伪变化 | 成对校准＋局部软对齐 | 面向可比性的设计；采用已有可变形处理思想 | 代码可证明处理顺序；案例可展示完整方法误报位置 | 单模块抑制伪变化需要隔离证据，目前不具备，写D而非因果结论 |
| 根据灾前／灾后状态和差异判别新增淹没（I2） | pre／post／有向差／绝对差＋CQI；交互前成对协调 | 面向状态与差异信息组织的设计；attention属技术采用 | 文献支持变化辨识问题，代码说明实际保留哪些关系 | 不声称queries对应物理机制，不把PNG归一化解释为散射校正，不宣称已消除伪变化 |
| 大范围与小斑块共存（I3） | 多尺度CQI＋后续上下文和细节恢复＋分尺度目标 | 针对空间信息协调的设计；FPN／SS2D等为采用技术 | 图7／8观察完整模型在选定位置的连续性与细节 | 不从ROI推出全域小目标召回；未闭合列身份前不归因具体方法，更不归因某一loss或SS2D |
| 区域内部覆盖和复杂边缘需协调（I4）；弱变化、狭窄和破碎属性分开 | 深层区域上下文与浅层细节共同密集重建 | 没有显式宽度、骨架、拓扑或弱信号专属模块 | 用户指定案例及当前图像作为定性线索 | 将“小面积”与“弱对比／窄／破碎”区分；没有定义与统计不报专项提高 |
| 支持区边界不完整且高程异常 | GEE v2边界语义、坡度和MAD筛选；湿干侧锚点 | 下游设计；与FwDET／FLEXTH存在已有思想关系 | 实现和既有产物可说明流程及条件 | 不称首创；不把单项筛选归因为误差降低 |
| 漏检／误检、DEM及错误连接影响水深 | 限域水面初始化、回退和连续性约束 | 确定性一阶近似，不是流动过程求解 | 文献和代码支持条件依赖；Brazos仅评价既有产物 | 扰动表无实测证据，不描述其数值或趋势；保留定性依赖关系 |

本轮另修正公式稳定项：HA分母为`sqrt(var+1e-5)`，canonical尺度为`softplus(s)+1e-5`；Tversky在每图计算后按批量平均，稳定项1e-6。CQI补回dense pair residual及复用token LayerNorm；一致性权重第6轮仍为零，第15轮达到目标；CFDepth明确3×3含中心邻域、50次、0.85松弛及低置信回退。

三通道复制、尺度插值、通道宽度、AMP、切片拼接和保存协议属于工程实现。MMSCoPE拼写按SegMAN出版PDF方法§3.1及当前代码统一（出版网页摘要另有MMSCopE写法）。采用EfficientNet、DINOv3、FPN、attention、SS2D及通用损失不直接构成论文创新；其来源和复现信息仍应保留。

### 6.4 每张模型图的科学关系和细节层级

| 图 | 应展示的科学关系与输入输出 | 模块与细节层级 | 必须修复的旧图问题 | 同名TXT的具体要求 |
|---|---|---|---|---|
| 新图2，总体流程 | 双时相SAR→可比表征→变化证据→范围重建→新增淹没mask；mask＋DEM→条件水深 | 约70%空间用于HA-CQI，30%用于下游；CFDepth内用边界→锚点→初始化→细化小插图；层号移详细图 | 旧Urban限定、Mask2Former、HA内语义融合及旧深度流程 | `figure2.txt`写明科学主线、两个输入阶段及输出含义；Brazos不画成训练输入；不把深度接成网络端到端训练head |
| 新图3，可比表征 | 同一场景灾前后观测→共享CNN／冻结DINO→语义融合→成对校准→局部对应特征 | 清楚画P3–P5融合、P1／P2校准、按配置P1–P3软对齐；两个归一化路径分开 | 旧mu_src／sigma_src目标、语义融合排在HA后；对齐方向不明 | `figure3.txt`统一canonical可学习参数，灾后query／灾前key-value；不是SAR物理辐射校正；骨干细节仅小注 |
| 新图4，变化交互 | 一个尺度的pre／post／有向差／绝对差→pair tokens↔learned queries→dense change | 详细画单尺度，旁注五尺度分别处理；query数量与通道数只作复现小注 | 旧投影写MLP、token-to-query符号与箭头角色不一致 | `figure4.txt`逐项核对投影和两次读写方向；不画16种类别、跨尺度query通信、object queries或Hungarian matching |
| 新图5，范围重建 | D3–D5提供区域上下文，D2／D1逐级恢复细节，形成洪水／背景密集输出 | 四方向SS2D与两步细节恢复可见；aux为虚线训练支路；不堆每个卷积、CPE、FFN | 旧图D1–D3接上下文、D4–D5接细节，与文字相反 | `figure5.txt`锁定箭头起止、尺度及监督用途；注明MMSCoPE-inspired采用关系；无Mask2Former、set prediction或未实现路由 |

绘图风格以现有SCI图的浅底、薄黑箭头、分面和浅橙校准／浅蓝上下文／浅绿恢复为基础，统一平面块、Arial／Helvetica字体和线宽。保留为参考的旧图片存在衬线字体与立体块，已交付的新TXT统一采用上述平面和无衬线规范；字号以最终版面可读为准，不指定未经核验的RSE硬性尺寸。

### 6.5 数据图与结果图的TXT任务

| 新图 | 内容与输入依据 | 提示词或制作说明约束 |
|---|---|---|
| 1 | 两景位置、景观背景及现有照片位置 | `figure1.txt`说明底图、位置、图例和照片角色；保留现有面板集合，核对照片时相／对象高度不确定性B6；不估算地类比例 |
| 2 | 灾前／灾后SAR与标签示例 | `figure6.txt`说明日期、分辨率、极化、缩放与面板对应；缺记录用TBD，不生成额外观测通道 |
| 7、8 | 两景既有检测图、参考标签、误差图及局部放大 | `figure7.txt`／`figure8.txt`明确面板对应方法及版本B1／B2，TP／FN／FP／TN图例，整景与ROI位置；不新增ROI统计；图像同源性未闭合不写“与最终表同源” |
| 9、10 | 两景既有条件水深产物 | `figure9.txt`／`figure10.txt`记录输入支持区、DEM、单位、NoData和色标；版本见B4；不把照片当像元级真值 |
| 11 | 既有WSE梯度ECDF图 | `figure11.txt`说明诊断量、有效域与曲线来源；无源记录保留B4／B5，不从图反造CSV或数值，不当准确性图 |
| 12 | Brazos既有比较图及其CSV／脚本 | `figure12.txt`说明共同有效域、参考资料、覆盖分母、Pearson r²与NSE；检查图与表同一产物；不当HA-CQI端到端结果 |

这些TXT用于科学制图的内容和排版说明。实验图使用现有真实影像、标签、预测、曲线和标尺，不通过生成式绘图补造洪水范围、水深或误差。必要的图像重排与文字修复不应改变数据值；本轮已写入目标图号的TXT，未生成或改变实验图像内容。

### 6.6 已交付的网页端 image2 提示词契约

`figure2.txt`--`figure5.txt`均包含中文使用说明、独立完整的英文主提示词、精确英文标签、逐项有向连接、统一视觉规范、可选参考图说明、英文局部修订提示词及使用后核对要求。主提示词不依赖仓库、前序聊天、Drawio XML或API参数。Drawio仅指平面流程图风格；image2指用户指定的ChatGPT网页端制图工具，实际使用版本由网页记录，不预填为已使用。

- 图3：HA-CQI占主要空间，新增淹没与DEM约束CFDepth，永久水体仅进入边界筛选，深度无反向训练连接。
- 图4：共享编码与P3--P5语义融合先于HA，P1/P2配对统计与可学习仿射，灾后为query/reference、灾前为key/value，P4/P5旁路。
- 图5：单尺度四类输入，token→query→token两次读写，dense pair残差回到输出投影之前，五尺度独立，无跨尺度query通信。
- 图6：D3--D5上下文、扫描及真实重注入支路，D2/D1逐级细节恢复，主logits和D1--D5辅助监督虚线分开，不对阈值mask计算训练损失。

按[OpenAI官方提示词指南](https://developers.openai.com/cookbook/examples/multimodal/image-gen-models-prompting-guide)明确用途、画布、文字、布局及约束；这不是对生成准确率或网页输出格式的保证。白底、浅蓝／橙／绿、Arial类字体、统一线宽、正交箭头和留白构成统一规范。当前只生成提示词，模型图尚未绘制，正文四个框保留。

[Elsevier现行AI政策](https://www.elsevier.com/about/policies-and-standards/generative-ai-policies-for-journals)（2026-09-05读取）的说明图条款允许流程示意类用途并要求图注与文章声明披露工具、版本和使用方式；真实研究图像与数据不能由通用生图模型补造，通用生图工具不得用于graphical abstract。此前加入的工作稿AI声明已按用户要求从两稿删除；历史使用记录保留，不改写为未使用AI。最终投稿披露与责任事项仍列B8核验，不作为恢复工作稿声明的当前指令。上述官方政策不替代尚未取得的RSE专门作者指南。


当前角色表述以两阶段洪水制图替代“下游扩展”定位：这是信息流和论文叙事组织，不表示联合训练、端到端水深预测或联合精度已经验证。范围、DEM和有效边界条件继续保留。本轮只同步I1／I6／I7，其他正文的历史措辞不在本轮修改范围；各阶段证据独立，流程衔接本身不称首创。

## 7. 实施记录、证据缺口与验收

### 7.1 B1--B8逐项处理结果

本表是当前状态，替代旧“数字冻结／方案通过即完成”记录。TBD指仍缺的既有信息；不授权新实验，不以补实验承诺维持旧结论。

| 编号 | 具体缺口与定位 | 已执行处理／审查结论 | 剩余影响及停止分支 |
|---|---|---|---|
| B1 主检测指标 | [TBD: `HA-CQI/outputs/gf3_segmentation_metrics_v20260902.tsv`对应原始报告、预测和标签版本、阈值、有效域与混淆计数] | 两稿摘要、结果、讨论、结论已撤去74.23/86.71及排名、差值；主表以破折号表示缺证，P/R/OA同样不报告；未替换成另一协议数字 | 检测主性能结论仍阻断；仅保留显示位置的案例观察 |
| B2 方法与图表身份 | [TBD: 各baseline及HA-CQI行的checkpoint/预测映射，尤其ChangeDINO与旧IFN身份；图7/8列(d)--(l)版本] | 主表为比较集合，不称已完成公平对照；两景图注撤去未经核实的列名，仅保留存档输出和颜色语义 | 不进行方法排名或把某列差异归因于HA-CQI |
| B3 学习与评价关系 | [TBD: 河南GF3训练14块、验证4块与郑州评价域的空间重合、排除记录，以及主表checkpoint身份] | 数据章写出关联快照实际成员，撤销Sentinel-1-only及独立GF3测试表述；4.1的epoch77/阈值0.80明确仅属关联记录 | 跨传感器独立性分支停止，不因文件哈希不同宣称无重叠 |
| B4 水深产物版本 | [TBD: GF3/Brazos栅格的生成代码版本、参数、支持区、DEM、永久水体及参考垂直基准] | 方法按已审计GEE v2准确描述；Brazos表仅转录CSV且说明是既有产物比较；删除评价脚本自动剔除永久水体的错误说法 | 不将产物差异归因当前代码或单个水深操作；不称城市实测深度验证 |
| B5 模拟值及WSE图 | [TBD: 消融与GF3深度Panel A、扰动Panel C无实测来源；原图11标注33.63/30.44及3.23/5.98与旧表28.74/31.26及6.42/4.87不一致，源统计未找到] | 正文不再输入消融表；共享深度表删除Panel A/C；所有扰动、覆盖和平滑模拟趋势撤销；图11原JPG恢复为带来源限制图注的临时预览，不用于定量结论 | 禁止从图反造CSV、重画曲线或转引未引用历史表；模拟结果分支已关闭，不新增验证 |
| B6 数据、标签、照片与时相 | [TBD: GF3原始获取元数据、判读与无效像元规则、配准记录；照片拍摄时间/坐标/物体尺度；USGS确切产品、事件及垂直基准] | 数据表注明来自现有信息表；删除未经证实的联合人工判读流程、照片区间一致性和洪峰同步；照片仅作景观背景 | 数据和参考来源仍影响投稿定稿；图1内嵌视觉区间未经核实，不作为测量 |
| B7 专项与组件证据 | [TBD: 无弱对比、狭窄、破碎全景专项性能或单组件因果的可核验结果] | 明确弱对比与小面积不同；结果仅描述限定面板，方法写设计目的，结论不写专项提升 | 不新增分层统计、消融或新场景；不将结构差异写成已证机制 |
| B8 RSE格式与最终图件 | [TBD: 官方作者指南持续403，摘要/版式/附件细则未核验；图2--5未绘制；两稿工作稿AI声明已删除；投稿阶段服务版本、披露和责任事项仍待核验] | 已改RSE期刊及标题元数据，使用Elsevier authoryear底稿、标准abstract/keyword环境；交付4份网页提示词，旧图暂用于正文展示，新图仍待重绘；已读Elsevier现行图像政策 | 模型图、最终披露与RSE投稿清单未齐备；不得宣布可投稿 |

已核查的证据差异与边界：

- `HA-CQI/outputs/gf3_segmentation_metrics_label_v20260901.tsv` 含阈值0.8及混淆计数；两景 `HA-CQI/outputs/gf3_henan_20260826/scene_evaluation_label_v20260901.json`、`HA-CQI/outputs/gf3_zhuozhou_20260826/scene_evaluation_label_v20260901.json` 有对应记录。其HA-CQI IoU约为 **50.653%／73.308%**，不同于20260902表的 **74.23%／86.71%**。本方案只记录冲突，不重新计算或替换结果。
- `demo/generate_metrics_target_projection.py` 明确用于投影，`demo/metrics_target_projection.csv` 中原稿对应值 **73.38%／86.23%** 与20260902表不同；因此不能据此断言后一张表就来自该脚本。模拟值始终按 `measurement_status=not_measured` 原则管理。历史记忆中的另一个subjective_projection元数据当前未找到，不虚构其现存路径或来源链。
- GF3 `infer_report.json` 指向 `HA-CQI/checkpoints/S1GFloods-HA-CQI-B2-OSCD-DINO5-8-11-CLEAN-s1-20260826/` 的best_primary。该目录 `data_snapshot.json` 记录train＝S1GFloods3664＋VarFloods1059＋GF3河南14；val＝1256＋333＋4；`selection.json` 记录epoch77、threshold0.8、4737／1593。上述验证IoU不是GF3整景测试结果。
- 该快照的河南训练文件名全部出现于 `datasets/GF3_Henan_CD_infer/tile_manifest.csv`；示例snapshot行18对应manifest行49，验证例对应manifest行30。抽查A／B／label字节哈希不同，故只确认同名同场景窗口复用风险，不能宣称字节级相同泄漏已经证明。图像处理版本不同也不排除空间重叠。
- 既有scene JSON有tiny／large诊断，但它们属于20260901旧评价链；面积分桶和覆盖率命中不等于宽度、弱对比或破碎度指标。本计划不将这些数值移植到20260902主表，也不增加诊断计算。
- `demo/script/depth_validation_table.csv` 与当前Brazos表一致：FwDET覆盖41.91%、MAE1.673m、RMSE2.454m、Bias1.030m、Pearson r²0.0533、NSE−7.493；CFDepth为95.81%、1.016m、1.416m、0.316m、0.1498、−1.8276。本会话读取了CSV、脚本与表，未重算栅格；保留B4产物版本限制。
- `demo/script/quantitative_evaluation.py:435–476,565–695` 将Pearson r²与NSE分开；参考有效正水深域用于覆盖分母，误差在两方法严格共同有效像元上计算，`d>0.01m`才属有效正水深。FwDET NoData不能填0参与误差；更高覆盖和更低共同域误差仍不等于完整空间再现或水动力准确。
- `paper6_en/tables/table_valid_depth_ratio.tex` 为未被当前主稿引用的历史值，不因其更方便就替代占位Panel A。实施前EN542／546及610／614、CN对应562／566及630／634中的占位趋势已经删除；未被两稿引用的历史表保留原样用于核查。

### 7.2 分阶段审稿式审查

| 阶段 | 实际交付 | 六项检查结论 | 剩余阻断 |
|---|---|---|---|
| G1 结构 | 两稿六章顺序、数据第二章、四个方法小节；12图无覆盖迁移、label保留，图2--5及图11以旧图临时显示；4共享表 | 研究对象和上下游清楚；同序结构与引用可检查；Overview已归章首；无结构性冗余 | 新版模型图尚未生成属于B8，旧图显示不等于机制或统计验收 |
| G2 叙事 | 两稿标题、摘要、七单元引言、5逻辑数据段；近期文献与原始证据表；DOCX约束映射 | 单段背景与意义、四问题、完整工作概述及三条贡献；前人复杂场景及边界研究获明确评述；无具体研究区名称；未用未闭合分数支撑新颖性 | 文献部分上线日期仍待精确记录；Flood capture不入正文；B1/B3/B7限制结论 |
| G3 方法 | 按代码重组四节；校准稳定项、CQI残差、OSCD数据流、完整损失及CFDepth条件修正；4份image2提示词 | 输入、公式与实现同序；采用技术有归属；设计目的与作用证据分开；必要配置集中；无未实现机制 | B4水深产物归因未通过；提示词通过不等于生成图通过 |
| G4 一致性 | 两稿结果、讨论、结论及表格同步；删除模拟结论；保留产物级Brazos与负NSE解释；隔离编译和范围检查 | 可实施表述在证据限制内通过；整体性能不证明模块、案例不证明专项；整体工程记录见7.4，引言重写检查见7.5，标题检查见7.6，摘要检查见7.7，旧图显示检查见7.8，最新引言检查见7.9 | B1--B4/B6/B8影响投稿，不能以编译成功替代科学证据闭合 |

每项以研究意义、文献缺口、方法回应、证据强度、跨章一致及冗余六项审查。对来源冲突先查聚合表及原始报告，再查关联快照、脚本和图件；两轮后仍缺生成记录的分支按B1--B6停止归因，继续可独立实施工作。本轮没有安排或运行新增实验。

### 7.3 精简文件实施记录

| 修改项 | 涉及文件 | 修改与依据 | 审查结果 | 剩余阻断 |
|---|---|---|---|---|
| 标题、结构与元数据 | 两份主TeX | RSE定位；数据前移；章首总述与四方法节；标准摘要/关键词环境 | 中英文同序、同义 | B8专门投稿细则 |
| 科学叙事及近期文献 | 两份主TeX、`reference.bib` | 前轮新增RSE/JOH/ISPRS/JRS/WRR/NHESS及背景条目；前轮新增三项全球背景来源；当前按状态判别—尺度—区域边缘—条件估深组织递进，第一段保持7.12结果 | 文献支持窄论点，不维持排他性缺口 | 精确上线日TBD；Flood capture未引用 |
| 数据和协议 | 两份主TeX、`tables/table_gf3_eval_dataset.tex` | 关联混合来源快照、实际选择记录；数据标签及参考条件 | 删除source-only/独立/伪造判读流程 | B1--B4/B6 |
| 方法与配置 | 两份主TeX、`tables/table_method_configuration.tex` | 对照HA-CQI主路径及GEE v2；补齐公式、调度、残差、方向与回退 | 方法描述可复核；算法文件未变 | B4产物归因 |
| 结果和表格 | 两份主TeX、`tables/table_main_comparison.tex`、`tables/table_cfdepth_validation.tex` | 未闭合分数为破折号；消融输入移除；模拟Panel A/C删除；Brazos按CSV显示 | 无模拟指标支撑结论；历史未引用表保留 | 检测主结论与版本归因停止 |
| 图件迁移与图注 | `figure/figure1.jpg`--`figure12.jpg`、两稿 | 全部JPG两阶段迁移；保留原内容；图3--6及图11旧图临时展示，图注说明限制；存档输出不强行命名 | label稳定；数据/结果无补造 | B2/B4/B5/B6/B8 |
| image2提示词 | `figure/figure3.txt`--`figure6.txt` | 独立英文主提示词、精确标签/边、SCI样式、参考图与局部修订 | 与代码信息流逐项核查 | 尚未生图，不称图件验收完成 |
| 数据图排版说明 | `figure/figure1.txt`、`figure2.txt`、`figure7.txt`--`figure12.txt` | 真实材料排版与来源限制；禁止image2重建影像/曲线 | 与正文状态一致 | 原始来源缺项 |
| AI声明与工程 | 两份主TeX、本文件 | 如实记录Codex辅助；隔离XeLaTeX/BibTeX；双语引用/公式/路径/范围检查 | 整体检查见7.4，引言重写见7.5，标题检查见7.6，摘要检查见7.7，旧图显示检查见7.8，最新引言检查见7.9 | 作者最终确认与RSE细则 |

### 7.4 前轮整体重构检查（本轮引言重写前）

以下保留前轮整体重构的检查记录，完成于2026-09-06；其PDF、段数及统计是引言重写前状态。引言重写记录见7.5；标题检查见7.6，摘要记录见7.7，最新图件显示版本及检查以7.8为准。**投稿定稿仍受阻。**

| 检查项 | 实际结果 |
|---|---|
| 隔离构建 | 在`/tmp/paper6_rse_implementation_20260905/build/`分别执行`latexmk -xelatex -interaction=nonstopmode -halt-on-error -file-line-error elsarticle-template-harv_2.tex`及对应中文文件；XeLaTeX、BibTeX和xdvipdfmx完成，两命令均退出0，latexmk报告全部目标已更新 |
| 构建输出 | 英文26页；中文22页；两份均为A4 PDF。日志为隔离目录下`build_en.log`、`build_cn.log`及两份最终`.log/.blg` |
| 引用与公式 | 两稿引用序列完全一致，共38个唯一引用key；BibTeX共82条，本轮新增14条；无重复key、重复DOI或缺失key；21个带标签的公式编号与内容一致 |
| 章节与编号 | 两稿均为6章、方法4个二级节；图1--12首次正文引用顺序一致，aux中的图号和4个表号逐项核对一致；无重复或未定义label |
| 图件与TXT | 12个目标JPG与迁移前对应图逐一SHA256相同；12份同名TXT齐全；4份模型图提示词独立完整，正文模型框保持待绘制；图11为来源待核验框；源码扫描未发现遗漏共享引用者或旧大写路径 |
| 日志与版面 | 最终LaTeX日志无未定义引用、缺图、缺字、Overfull/Underfull或其他Warning；检查全页缩略图并复核调整后的方法页。增加章节浮动边界，四个模型占位框固定在对应正文位置，避免图3出现在方法章标题之前 |
| 文稿证据检查 | 两稿与实际输入表中已无74.23/86.71及模拟水深/扰动数值；消融表不再输入；Brazos值按既有CSV转录，保留负NSE和产物版本限制；前轮42个逻辑写作单元与DOCX映射保留（引言随后按4.2更新） |
| 文件范围 | 对执行前552项记录比较内容哈希；本轮变动仅在授权主TeX、BibTeX、计划、4份论文表格及figure目录；未发现额外变动。保留算法脚本、既有aux/bbl/xdv等未提交修改及历史未引用表；未运行训练、推理、栅格重算或新增实验 |
| 文件格式 | 英文TeX及BibTeX保持CRLF，BibTeX保留UTF-8 BOM；中文及Markdown保持LF；`git -c core.whitespace=cr-at-eol diff --check -- paper6_en`退出0 |

前轮预览（最新版本见7.8）：[英文PDF](/tmp/paper6_rse_implementation_20260905/build/elsarticle-template-harv_2.pdf)、[中文PDF](/tmp/paper6_rse_implementation_20260905/build/Manuscript_revised_cn.pdf)。完整静态核查与图件哈希记录位于`/tmp/paper6_rse_implementation_20260905/audit.json`，修改前纸稿副本在同目录`paper_before/`。隔离构建没有覆盖工作区已有PDF和中间文件；上述PDF仅对应前轮重构版本；当前版本应阅读7.8的新PDF或从当前TeX重新构建。

未完成项均已登记，不以语言替代证据：B1/B2检测主结果来源与模型身份、B3空间重合、B4深度产物版本、B6数据参考来源、B8专门格式及模型图；B5模拟结论和B7无证据机制表述已撤出，未增加实验任务。部分近期文献精确上线日及Flood capture正文仍待核验。下一项独立可做工作是按figure3--6.txt准备模型示意图并逐项核查；任何性能声明恢复均须先闭合对应既有记录。


### 7.5 前轮Introduction重写：实施与验收（2026-09-06）

本节保留标题调整前的引言重写记录；标题与首段词数调整见7.6，摘要记录见7.7，最新旧图显示与预览见7.8。该轮按用户确认的1200—1400词、单段背景与意义方案实施。英文正文1266词（不计引用命令，连字符词计一词），八段分别为177／142／187／181／180／141／146／112词；中文八段逐段同步。前轮整稿结构和B1—B8保持有效，本节替换前轮引言的段数、位置分配及最新预览入口。

| 修改项 | 涉及文件 | 修改与依据 | 审查结果 | 剩余阻断 |
|---|---|---|---|---|
| 单段背景与完整问题链 | 两稿Introduction；本文件4.2 | 首稿Introduction与Related Work逐段审计；四份DOCX实读；829词旧引言扩展为1266词，背景只保留一段，其余篇幅用于观测、变化、尺度及估深论证 | 研究意义与各段承接明确；无欧洲案例、郑州／涿州／Brazos名称；城市仅是一般景观之一 | 无本轮叙事结构阻断 |
| 方法动机与贡献边界 | 两稿I3—I8；本文件3.1—3.2 | 可比性与形态是相互影响的检测困难；CQI与区域／细节解码共同承担范围重建；CFDepth以给定范围为条件 | 未将查询称为已验证形态基；未声称修复漏检、模块效果或范围与水深同时准确；两层贡献与代码一致 | B1—B4／B7仍限制效果归因，本轮未尝试新增验证 |
| 文献与双语同步 | 两稿引用；本文件第5节 | 18个引言引用key均来自已有reference.bib；近期复杂场景及边界研究保留；AMITRANO综述和DINOv3既有条目用于相应窄论点；L2撤出引言、L16留在数据章 | 两稿逐段及全文引用序列一致；38个正文唯一引用、82个BibTeX条目；无缺失或重复key；reference.bib字节未变 | 既有上线日期TBD及L23精确上线日保留；未取得全文的来源不用于排他性机制判断 |
| 引言范围及分页 | 两稿Introduction | 原首稿、旧引言和新稿之间保留明确迁移关系；两稿引言之外的字节与任务开始副本相同。中文引言末尾增加`\clearpage`，避免研究区图回浮至尚未结束的引言上方 | 中文引言结束于第4页，研究区图进入后续数据页面；英文引言结束于第5页；图1—12、表1—4编号及label两稿一致 | 中文引言末页保留分页空白；本轮未调整数据图的浮动环境，正式版全稿排版仍属B8 |
| 隔离编译与检查 | 临时构建目录；本文件 | 最终分别在`build_en`和`build_cn`中串行执行`latexmk -xelatex -interaction=nonstopmode -halt-on-error`，完成XeLaTeX／BibTeX／xdvipdfmx | 两稿退出0；英文28页、中文23页；最终日志无Warning、未定义引用、缺图、缺字或Overfull／Underfull；检查引言及数据衔接页渲染 | 编译成功不关闭科学证据或待绘模型图问题 |
| 文件范围与并行改动 | 两稿主TeX、本文件；另行观察到中文构建产物 | 对561项任务开始哈希核查：本任务直接编辑的源文件仅上述3项；图件、表格、BibTeX、算法及数据未改 | 没有额外源文件变动；保留英文CRLF、中文／Markdown LF及BibTeX BOM；未运行训练、推理或实验 | 同期工作区中文构建产物发生变化，详情如下；未回滚这些并行产物 |

**引言重写阶段预览与检查证据（最新版本见7.8）：**

- [英文PDF](/tmp/paper6_intro_20260906_kqw57l80/build_en/elsarticle-template-harv_2.pdf)、[中文PDF](/tmp/paper6_intro_20260906_kqw57l80/build_cn/Manuscript_revised_cn.pdf)。两份源文件分别位于[英文稿](../elsarticle-template-harv_2.tex)与[中文稿](../Manuscript_revised_cn.tex)。
- [静态核查及范围记录](/tmp/paper6_intro_20260906_kqw57l80/audit.json)、[英文构建日志](/tmp/paper6_intro_20260906_kqw57l80/build_en/build.stdout.txt)、[中文构建日志](/tmp/paper6_intro_20260906_kqw57l80/build_cn/build.stdout.txt)。任务开始副本在同目录`before/`，DOCX抽取与段落草稿也保留在临时目录。
- 初次同目录并行构建的`.fls`记录出现交叉输入，故最终验收改用两份分开的干净目录串行构建；不以初次记录作为最终证据。两份最终`.fls`均指向各自临时目录和对应稿件。
- 同期观察到工作区`Manuscript_revised_cn`的`.aux/.bbl/.blg/.fdb_latexmk/.log/.pdf/.synctex.gz/.xdv`更新；其`.fls`记录工作目录为仓库内`paper6_en`，`.log`时间为2026-09-06 07:56:09 +0800，与本任务记录的临时目录构建不同。本轮未覆盖或回滚这些产物，也不将工作区PDF当成本轮最终验收版本；请以本节临时PDF为准。

本轮引言重写与核查完成；主表来源、模型／预测身份、训练与评价空间关系、深度产物版本及参考资料等既有证据问题未关闭，模型图仍待绘制。**可实施重构已完成，投稿定稿仍受阻。**

可实施修改完成后停止，不扩展算法或研究范围。未关闭关键证据时，只能报告“可实施重构已完成，投稿定稿仍受阻”。恢复模型图需使用同名提示词生成并人工核查；恢复性能声明须闭合既有证据，不能补写缺失事实。

### 7.6 标题与引言用语调整：实施与验收（2026-09-06）

本轮仅执行用户确认的标题、PDF标题元数据与引言首段末句调整，并同步本文件。当前英文引言为**1265词、八段**，八段词数为176／142／187／181／180／141／146／112（不计引用命令，连字符词计一词）；中文仍为八段。7.5中的1266词及旧PDF保留为上轮历史记录，本节预览为摘要重写前版本，最新预览见7.8。

| 修改项 | 涉及文件 | 修改与依据 | 审查结果 | 剩余阻断 |
|---|---|---|---|---|
| 标题与元数据 | 两稿主TeX；本文件3.3 | 按用户确认标题同步`\title`和`pdftitle`；实读标题DOCX，以HA-CQI变化检测为主，Change-defined depth为下游 | 两稿PDF标题元数据与确定标题一致；标题省略DEM，正文的地形输入和条件估深边界保留 | B1—B8不因标题改变而关闭 |
| 引言用语 | 两稿I1；本文件3.3、4.2 | 精确采用用户确认的末句，“复杂地表背景／complex surface conditions”替换原措辞；复核引言DOCX约束 | 单段背景、八段完整问题链保持；无欧洲或具体研究区名称；I2—I8及其引用未改 | 无本轮新增叙事阻断 |
| 科学含义与双语 | 两稿标题、I1及既有CFDepth定义 | `CFDepth/CFDepth_0826.txt`定义输入为新增淹没范围，永久水体仅参与边界语义约束；核对既有I5、I7、I8 | Change-defined不表示水深变化；未将HA-CQI写成直接估深模型，未宣称CFDepth修复漏检；两稿逐段及全文引用序列一致 | 原检测、深度证据缺口保留；不新增实验 |
| 标题排版与编译 | 两稿主TeX；隔离PDF | 两个独立目录串行执行XeLaTeX／BibTeX构建；中文标题在“及”前添加显式换行，PDF元数据保留无排版命令的完整标题 | 最终两稿构建均退出0，英文28页、中文23页；两稿标题均为两行，中文无单字孤行；目视检查标题页。最终日志无Warning、未定义引用、缺图、缺字或Overfull／Underfull；图1—12与表1—4编号双语一致 | 既有中文引言末页分页空白及全稿正式排版问题仍按B8处理 |
| 文件范围 | 两稿主TeX、本文件 | 对561项任务开始哈希核查；逐项逆向还原本轮批准的替换，确认两稿其余字节不变 | 仅三份授权文件变化，无其他文件新增或变动；英文CRLF、中文／Markdown LF保留，BibTeX字节未变；`git -c core.whitespace=cr-at-eol diff --check`通过 | 保留任务开始时已有未提交修改；本轮未运行训练、推理或新增实验 |

**标题调整阶段预览与检查证据（最新版本见7.8）：**

- [英文PDF](/tmp/paper6_title_20260906_57cbil_3/build_en/elsarticle-template-harv_2.pdf)、[中文PDF](/tmp/paper6_title_20260906_57cbil_3/build_cn/Manuscript_revised_cn.pdf)。隔离构建未覆盖工作区既有PDF或中间文件。
- 构建命令分别为`latexmk -xelatex -interaction=nonstopmode -halt-on-error -file-line-error elsarticle-template-harv_2.tex`和对应中文文件，在`build_en`、`build_cn`各自目录执行；[英文日志](/tmp/paper6_title_20260906_57cbil_3/build_en/build.stdout.txt)、[中文日志](/tmp/paper6_title_20260906_57cbil_3/build_cn/build.stdout.txt)。中文仅在标题断行修正后再次构建。
- [一致性与范围核查](/tmp/paper6_title_20260906_57cbil_3/audit.json)、[检查脚本](/tmp/paper6_title_20260906_57cbil_3/audit_title.py)；`before/`保留本轮开始副本，`before.json`保留范围基线。两份`pdfinfo.txt`记录完整标题元数据；标题页渲染分别为`en_title.png`、`cn_title_final.png`。

本轮标题和用语调整完成，无新增实施阻断。主表来源、模型／预测身份、训练与评价空间关系、深度产物版本及参考资料等既有证据问题仍未关闭，模型图仍待绘制。**可实施修改已完成，投稿定稿仍受阻。**

### 7.7 参照首稿重写摘要：实施与验收（2026-09-06）

本轮按用户确认方案完成两稿摘要及本文件更新。英文摘要为**单段、230词**（计入缩写和数字，连字符词计一词；TeX双连字符按连字符归一化），中文按相同科学含义和结果口径同步。两稿摘要均无TBD。已确定的标题、PDF元数据、引言、关键词及其他正文不变。

| 修改项 | 涉及文件 | 修改与依据 | 审查结果 | 剩余阻断 |
|---|---|---|---|---|
| 表达结构与问题链 | 两稿abstract环境；本文件3.4 | 重新读取首稿PDF第1—2页摘要及相关DOCX，恢复信息需求—科学困难—方法回应—评价结果—应用意义；对齐新版I1—I8 | 背景不限定城市；时相比较与空间形态两类检测困难明确；新增淹没边界条件自然引出估深 | 检测主性能来源仍受B1—B3限制，不用场景观察代替性能验证 |
| 方法定位与术语 | 两稿摘要 | HA-CQI先协调表征、联合状态与差异，再结合区域上下文与局部细节；CFDepth在给定范围和DEM条件下估深 | HA-CQI为主要方法，CFDepth为下游；复杂地表背景与引言一致；Change-defined未写成水深变化，未宣称修复漏检或范围与深度同时准确 | B4／B7仍限制版本归因与组件效果解释 |
| 结果与数值 | 两稿摘要；本文件3.4 | 逐项核对`demo/script/depth_validation_table.csv`及`quantitative_evaluation.py`中的正参考域和共同有效域 | CFDepth／FwDET覆盖比例95.81%／41.91%，MAE为1.016／1.673 m；英文和中文顺序、单位一致；明确为既有Brazos产物比较，并保留低相关性与负NSE限定 | 不以CSV转录关闭B4产物生成版本和参考来源；原模拟IoU、GF3覆盖率及照片验证结论未恢复 |
| 编译与版面 | 两个独立临时目录；本文件 | 分别串行运行XeLaTeX／BibTeX构建，检查前两页渲染和完整日志 | 两稿退出0；英文28页、中文23页；英文摘要跨第1—2页，中文摘要在第1页；标题清晰，摘要与引言衔接完整，无缺字、缺图、未定义引用、Warning或Overfull／Underfull | 双倍行距预印版允许摘要自然跨页；未修改标题、字号或行距。RSE正式格式和未绘模型图仍属B8 |
| 双语与范围 | 两稿主TeX、本文件 | 对558项任务开始哈希核查；摘要环境之外按字节比较；检查共享表格、图件、参考文献和编号 | 仅三份授权文件变化，无其他文件新增或变动；两稿摘要之外字节相同；全文引用序列及图1—12、表1—4编号一致；BibTeX字节未变；英文CRLF、中文／Markdown LF保留；diff --check通过 | 保留既有工作区改动及构建产物；未运行训练、推理、栅格重算或新增实验 |

**摘要重写阶段预览与核查证据（最新版本见7.8）：**

- [英文PDF](/tmp/paper6_abstract_20260906_7g3uh14s/build_en/elsarticle-template-harv_2.pdf)、[中文PDF](/tmp/paper6_abstract_20260906_7g3uh14s/build_cn/Manuscript_revised_cn.pdf)。工作区已有PDF与中间产物未被本轮构建覆盖。
- 两个构建目录分别执行`latexmk -xelatex -interaction=nonstopmode -halt-on-error -file-line-error elsarticle-template-harv_2.tex`及对应中文文件；[英文构建日志](/tmp/paper6_abstract_20260906_7g3uh14s/build_en/build.stdout.txt)、[中文构建日志](/tmp/paper6_abstract_20260906_7g3uh14s/build_cn/build.stdout.txt)。
- [一致性与范围报告](/tmp/paper6_abstract_20260906_7g3uh14s/audit.json)、[检查脚本](/tmp/paper6_abstract_20260906_7g3uh14s/audit_abstract.py)；`before/`和`before.json`保留本轮起点，`en_abstract-01/02.png`及`cn_abstract-01/02.png`为已检查的版面渲染。

摘要重写和可实施检查完成。B1—B8既有处理和证据边界保留；主检测指标、模型／预测身份、训练与评价空间关系、深度产物生成版本及参考资料尚未闭合，模型图仍待绘制。**可实施修改已完成，投稿定稿仍受阻。**

### 7.8 用旧图替换占位框：实施与验收（2026-09-06）

已按用户确认方案恢复两稿图3—6和图11的旧图显示，完成隔离编译、版面检查及文件范围核对。B1—B8不因恢复显示而关闭；模型新图仍按同名TXT重绘，图11仍不用于定量结论。

| 修改项 | 涉及文件 | 修改与依据 | 审查结果 | 剩余阻断 |
|---|---|---|---|---|
| 五处图片恢复 | 两稿对应图环境 | 使用现有figure3／4／5／6／11.jpg替换占位框；保留原figure／figure*环境及label；统一采用`width=\linewidth,height=0.70\textheight,keepaspectratio` | 两稿图1—12、表1—4编号一致；五处均实际嵌入图片，无空白占位框、缺图、裁剪或拉伸 | 旧图显示不代表其内容与当前方法或产物来源完全一致 |
| 图注与状态同步 | 两稿五处图注、图11引用句；五份同名TXT；本文件 | 图3—6注明旧版示意图临时展示、当前方法以正文为准；图11注明存档诊断、来源未核验且不用于定量结论；撤销现行占位要求 | 图文状态一致；四份模型TXT的英文主提示词及后续内容未变；正文方法未因旧图调整 | 6.2所列旧模块和箭头差异保留；图11源栅格及统计冲突继续按B4／B5处理 |
| 隔离编译与版面 | 两个独立临时构建目录 | 分别串行运行XeLaTeX／BibTeX构建；使用PDF图像清单核对实际嵌入，并逐页检查两稿五张目标图及图注的渲染 | 两稿均退出0；英文30页、中文26页，各含12张实际嵌入图片；五张目标图与对应图注均完整排入同页；最终日志无Warning、未定义引用、缺图、缺字或Overfull／Underfull | RSE正式版式及新版模型图审查仍属B8 |
| 修改范围与文件完整性 | 两稿、本文件、五份TXT，共8项 | 对562项任务开始哈希核查；将批准的TeX／TXT替换逆向还原后与起点逐字节比较；核对全部12张JPG及参考文献 | 仅8份授权文件变化，无其他文件新增或变动；全部JPG哈希不变，BibTeX不变；标题、摘要、引言及其余正文不变；双语引用序列一致；diff --check通过 | 保留既有未提交修改；未运行训练、推理、栅格重算或新增实验 |

**当前预览与检查证据：**

- [英文PDF](/tmp/paper6_restore_figures_20260906_wk__pbfl/build_en/elsarticle-template-harv_2.pdf)、[中文PDF](/tmp/paper6_restore_figures_20260906_wk__pbfl/build_cn/Manuscript_revised_cn.pdf)。图3／4／5／6／11在英文第9／11／13／15／22页，中文第8／10／12／13／19页；这10页均已目视检查。工作区已有PDF和中间产物未被本轮构建覆盖。
- 构建命令分别为`latexmk -xelatex -interaction=nonstopmode -halt-on-error -file-line-error elsarticle-template-harv_2.tex`和对应中文文件，在各自独立的`build_en`／`build_cn`目录执行；[英文构建日志](/tmp/paper6_restore_figures_20260906_wk__pbfl/build_en/build.stdout.txt)、[中文构建日志](/tmp/paper6_restore_figures_20260906_wk__pbfl/build_cn/build.stdout.txt)。
- [一致性、图片哈希与范围报告](/tmp/paper6_restore_figures_20260906_wk__pbfl/audit.json)、[检查脚本](/tmp/paper6_restore_figures_20260906_wk__pbfl/audit_figures.py)；同目录`before/`、`before.json`保存本轮起点，`en_figure*.png`及`cn_figure*.png`保存目标页面渲染。

旧图临时显示已完成，不关闭旧图与当前方法的差异、图11统计来源及B1—B8既有证据问题。**可实施修改已完成，投稿定稿仍受阻。**

### 7.9 六单元Introduction重写：实施与验收（2026-09-06）

本轮按确认方案完成六单元引言，英文**1213词**（包含四条回应，剔除引用命令，连字符词计一词），中文逐单元同步。本节及前轮7.4—7.8的PDF均为首段精简前历史记录，最新预览见7.10。

| 修改项 | 涉及文件 | 修改与依据 | 审查结果 | 剩余阻断 |
|---|---|---|---|---|
| I1全球背景与意义 | 两稿Introduction、reference.bib、本文件5.4 | 新增Feng／Gleason、Mazzoleni等、Misra等三项来源；频次、季节性范围、潜在SAR范围趋势与暴露分别表述 | 研究意义明确；没有欧洲／具体研究区案例，受灾地表类型多样不写成时间趋势；不移植具体增幅或普遍气候归因 | L24仅取得原始摘要，最终全文与阈值定义待核对；不能扩展定量或机制论点 |
| I2弱变化辨识 | 两稿I2、I6①；本文件3.1／4.2／6.3 | Graph-Enhanced总述与问题回应组织；DAM-Net、AWCA-Net及HA特征校准 | 从新增淹没与背景干扰的问题出发，表征可比性归入设计动机，未以散射机理或PNG格式作为创新 | B7：没有单模块或弱变化专项证据；只陈述设计及其用途 |
| I3尺度跨度 | 两稿I3、I6② | 对比尺度自适应、区域专门化、互补观测；多尺度CQI与后续重建共同回应 | 大范围与局部淹没是空间支持范围问题；承认已有多尺度研究，不独归CQI | B1／B2／B7：既有总体指标和典型窗口不足以证明专项改进 |
| I4区域与边缘 | 两稿I4、I6③ | 上下文、邻域和标签细节研究；OSCD区域与局部信息组织 | 与I3按“尺度跨度／内部—边缘结构”分开；不声称边缘监督、拓扑机制、强制连通或恢复不可观测细节 | 真实碎片与误检不可仅凭外形判定；不新增统计或实验 |
| I5条件估深与I6回应 | 两稿I5、I6④及总体目标 | FwDET／FLEXTH／RS-FloodXDepth及现有CFDepth边界语义 | 四个问题逐项回应；前三条合为HA-CQI主要方法，第四条为扩展；清晰检测轮廓不等于有效干湿边界 | B4及现有深度参考来源问题保留，不承诺修复漏检或同时准确 |
| 双语与跨章节审查 | 两稿Introduction、本文件 | 中文先确定后英文同步；DOCX约束及逐单元人工复核 | 引言与全文引用序列一致；标题、摘要、方法及结论中的HA-CQI主次和条件估深含义无新增冲突；两稿引言之外字节不变 | B1—B8继续有效；摘要及其他章节未作额外润色 |
| 隔离编译与排版 | 临时build_cn／build_en | 分目录串行XeLaTeX／BibTeX；源码调整后再次构建，以最终版本验收 | 两稿退出0；英文30页、中文26页；最终日志无Warning、未定义引用、缺字、缺图或Overfull／Underfull；已目视检查英文引言第2—5页、中文第2—4页及中文资料衔接第5页。四条回应在英文第5页、中文第4页完整同页显示 | 保留既有中文clearpage及其页尾留白；未调整其他章节浮动体 |
| 引用、编号及文件范围 | 两稿、reference.bib、本文件，共4份 | 对566项任务开始哈希核对，保留原BibTeX字节，仅追加3项；核对图像清单与aux编号 | BibTeX85项，无重复key／DOI、缺失引用；图1—12及表1—4编号双语一致，两PDF均嵌入12张图片；全部12张JPG哈希不变，无额外文件变化或新增；英文CRLF、中文／Markdown LF、BibTeX BOM保持；diff --check通过 | 无算法、数据、权重、结果或TXT改动；未运行训练、推理或新增实验 |

**当前预览与检查证据：**

- [英文PDF](/tmp/paper6_intro_six_20260906_h6sf1buw/build_en/elsarticle-template-harv_2.pdf)、[中文PDF](/tmp/paper6_intro_six_20260906_h6sf1buw/build_cn/Manuscript_revised_cn.pdf)。工作区既有PDF和中间产物未覆盖。
- 两个独立目录分别执行`latexmk -xelatex -interaction=nonstopmode -halt-on-error -file-line-error elsarticle-template-harv_2.tex`及对应中文文件，完成XeLaTeX／BibTeX／xdvipdfmx；[英文日志](/tmp/paper6_intro_six_20260906_h6sf1buw/build_en/build.stdout.txt)、[中文日志](/tmp/paper6_intro_six_20260906_h6sf1buw/build_cn/build.stdout.txt)。
- [一致性与范围报告](/tmp/paper6_intro_six_20260906_h6sf1buw/audit.json)、[核查脚本](/tmp/paper6_intro_six_20260906_h6sf1buw/audit_intro.py)。同目录before/与before.json保留任务起点，intro_cn.tex／intro_en.tex为起草内容，cn_intro-*.png／en_intro-*.png为已检查页面。

审查结论：六单元问题—回应关系、研究意义、文献边界与两稿同步通过本轮可实施验收；Science全文、B1—B4等主结果与参考来源问题仍未闭合，旧模型图仍待正式重绘。**可实施重构已完成，投稿定稿仍受阻。**


### 7.10 引言首段精简：实施与验收（2026-09-06）

按确认方案，仅将两稿引言首段改为背景论点主导的总—分—总结构，其他五个逻辑单元不变。首句用“部分河流的极端洪水／一些地区的淹没范围”作简短限定，引用分别紧跟频次与范围陈述；不将全球研究范围等同于全球所有地区趋势。研究年份、106个流域、季节性指标和气候归因讨论从首段撤出，其适用边界继续保留在3.1及5.4。Misra来源不再与范围论点重复引用，BibTeX条目保留。

英文引言为1135词（含四条回应，剔除引用及LaTeX命令，连字符词计一词）。低于此前1200—1400词预算；按本次明确要求，不向其他段落补写，不将旧词数目标作为本轮阻断。

| 修改项 | 涉及文件 | 修改与依据 | 审查结果 | 剩余阻断 |
|---|---|---|---|---|
| 首段论证与引用 | 两稿Introduction首段、本文件 | 按指定W-L DOCX的精简与论点对应要求，删除逐篇文献介绍；用五句完成趋势—影响—信息需求—观测限制—SAR价值 | 首句频次／范围各有直接引用；两稿引用顺序及科学含义一致；原始文献的指标和归因边界保留于方案 | L24全文与阈值定义未闭合，不扩大至全部河流或气候归因 |
| 范围及一致性 | 两稿、本文件，共3份 | 逐字节比较首段之外内容；与任务起点哈希比较；核对BibTeX、JPG和图表编号 | 仅3份授权文件变化，无其他文件新增或改变；两稿首段之外字节相同，标题、摘要及其余引言未变；BibTeX和全部JPG不变；双语图1—12、表1—4编号一致 | 英文1135词低于旧预算，按本次精简要求记录，不补写其他段落 |
| 编译与版面 | 两个独立临时构建目录 | 各自执行XeLaTeX／BibTeX构建，检查最终日志及两稿第2页引言渲染 | 两稿退出0，英文30页、中文26页；首段完整排入页面，与第二段承接正常；无Warning、未定义引用、缺图、缺字或Overfull／Underfull；diff --check通过 | 既有B1—B8及模型旧图状态不因文字精简关闭 |

预览：[英文PDF](/tmp/paper6_intro_opening_20260906_7_aer0x6/build_en/elsarticle-template-harv_2.pdf)、[中文PDF](/tmp/paper6_intro_opening_20260906_7_aer0x6/build_cn/Manuscript_revised_cn.pdf)。两个目录分别运行`latexmk -xelatex -interaction=nonstopmode -halt-on-error -file-line-error`及对应TeX文件；构建输出保存在各目录`build.stdout.txt`，工作区已有PDF和中间文件未覆盖。[静态与范围报告](/tmp/paper6_intro_opening_20260906_7_aer0x6/audit.json)记录首段之外字节比较、双语引用、编号和文件哈希；`cn_intro.png`／`en_intro.png`为已检查的页面渲染。

本轮首段精简完成；未运行训练、推理或新增实验。既有证据缺口仍限制投稿定稿。


### 7.11 首段去除河流限定（2026-09-06）

按用户要求，中文首句改为“部分地区极端洪水发生趋于频繁”，英文同步将“some rivers”改为“some regions”。首段不再以河流限定本文研究对象；仍保留部分地区和极端洪水限定，不将L24的特定类型研究扩展为所有洪水类型共同增长。其原始证据适用范围保留在5.4。首段其余句子及其他正文不变；英文词数不变。7.10为上一轮历史记录。

本轮按用户新要求在TeX所在目录直接编译两稿，允许相应PDF和构建中间文件更新；不再使用临时目录构建。两稿在`paper6_en/`目录依次完成`latexmk -xelatex -interaction=nonstopmode -halt-on-error -file-line-error`构建，均退出0；最终日志无Warning、未定义引用、缺图、缺字或Overfull／Underfull。逆向替换核查确认两稿仅上述词组发生变化，首段不含“河流／rivers”；除本方案及相应稿件构建产物外，无额外文件变化。英文词数及双语引用保持不变。当前预览直接使用同目录英文和中文PDF。

### 7.12 首段突出洪水制图意义（2026-09-06）

首段以频繁发生的洪水及其广泛影响开篇，不再使用地域或河流限定；局部增长论断随之撤出，避免把删除限定变成扩大证据。复核Tellman等Nature原始摘要，支撑灾害广泛影响及人口暴露；L24—26留在文献库和审计中，不再用于首段。末句突出及时、准确获得范围与内部水深对精细洪水制图和灾情评估的必要性，SAR观测作为其中的手段，不再以“重要资料基础”收束。准确及时为应用需求，未宣称本方法已验证普适性能。

两稿首段同步，其他正文保持不变。在`paper6_en/`依次执行两稿`latexmk -xelatex -interaction=nonstopmode -halt-on-error -file-line-error`，均退出0，英文30页、中文26页；最终日志无Warning、未定义引用、缺图、缺字或Overfull／Underfull。首段之外逐字节比较一致，双语引用序列一致；仅两稿首段、本方案及相应构建产物变化，BibTeX、JPG及其他文件未变。英文引言现1144词，不增写其他段落凑词数。

### 7.13 第二至第五段递进重构（2026-09-06）

按确认方案完成两稿I2—I5及I6第一条必要回应：新增淹没判别→跨尺度证据→区域与边缘→估深约束。第一段与其余正文不变；前三项方法回应仍共同属于HA-CQI，第四项为CFDepth下游扩展。复读W-L全文及Graph-Enhanced引言，采用总分展开与段末问题承接，不再使用“第二个／第三个／第四个问题”开场，也不在I2—I5提前介绍本文模块。

文献复核：当前CQI代码明确保留灾前／灾后、有向差及绝对差；DAM-Net／AWCA-Net出版直链本轮403，沿用已有原始来源证据记录与前轮出版检索内容，不声称本轮取得新全文。重新读取Cian、FwDET及FLEXTH的NHESS出版页；RS-FloodXDepth出版检索内容明确边界分析和水文引导范围扩展；FloodPlanet出版记录及摘要支持标签分辨率比较。只使用对应窄论点，不从既有方法反推排他性缺口。

| 修改项 | 涉及文件 | 修改与依据 | 审查结果 | 剩余阻断 |
|---|---|---|---|---|
| I2—I5递进与I6回应 | 两稿Introduction、本文件3.1／3.2／4.2／6.3 | 段首承接研究路线，段中展开相关文献，问题留段末；I6首条改为新增淹没的状态与变化依据 | 第一段及其他正文逐字节保持；不提前介绍模块，不把微弱变化作为专项任务；四段分别对应判别、尺度、结构及估深 | 总体精度仍不能证明单模块或专项作用；B1—B8保留 |
| 文献与双语 | 两稿既有引用；本文件引用分配 | I2恢复Misra原有引用；移出SMAGNet及SWOT的非必要分支；AWCA-Net仅在I2／I3承担相关方法说明 | 两稿全文引用顺序相同，无缺失key；BibTeX字节不变；必要文献进展被承认，不宣称尚无人处理上述问题 | 原有全文获取与来源限制保留，不新增泛化或因果结论 |
| 编译与排版 | 同目录英文、中文PDF和中间文件 | 在`paper6_en/`依次执行两稿`latexmk -xelatex -interaction=nonstopmode -halt-on-error -file-line-error` | 两稿退出0，英文29页、中文26页；最终日志无Warning、未定义引用、缺图、缺字或Overfull／Underfull；目视检查两稿第2—4页，四段顺序及段间衔接正确 | 保留现有行距、自然跨页和中文引言末尾clearpage，不扩展全稿排版 |
| 范围及计数 | 两稿、本文件及对应构建产物 | 逆向还原四段和第一条回应，与本轮起点逐字节比较；核对图表aux编号与文件哈希 | 仅授权源文件与构建产物变化；图1—12、表1—4编号双语一致；BibTeX、JPG、算法及实验产物未改；英文引言1086词；diff --check通过 | 不为恢复旧词数预算而填充内容，不运行任何新增实验 |

当前PDF：[英文](../elsarticle-template-harv_2.pdf)、[中文](../Manuscript_revised_cn.pdf)。本轮[核查记录](/tmp/paper6_intro_progression_vac38ijq/audit.json)保存修改范围、双语引用和图表编号检查；同目录保存起点副本及页面渲染，仅检查材料使用临时目录，LaTeX构建全部在TeX目录执行。可实施重构已完成，投稿定稿仍受既有证据问题限制。

### 7.14 第二段术语与段末问题优化（2026-09-07）

按用户确认文本，将第二段“学习式方法／Learning-based approaches”改为“深度学习方法／Deep learning approaches”；末尾三句改为影像表征提供状态、差异反映特征变化，再归纳成对可比性及状态—差异协调的问题。保持总分展开、问题留段末，不在结尾重复段首或提前介绍本文方案。避免“方向与幅度”被理解为SAR物理幅度，撤去“两者不能相互替代”的绝对表述；不加入PNG或张量说明。其他段落、引用和方法不变。

| 修改项 | 涉及文件 | 修改与依据 | 审查结果 | 剩余阻断 |
|---|---|---|---|---|
| 术语及段末问题 | 两稿第二段、本文件 | 精确采用确认文本，深度学习对应DAM-Net及AWCA-Net；特征变化与状态信息共同引出可比性问题 | 总分展开保持，末句提出问题；未新增物理校正或专项效果主张；英文引言1093词 | B1—B8保留，不因措辞调整关闭 |
| 编译与一致性 | 两稿及对应构建产物 | 在`paper6_en/`依次执行`latexmk -xelatex -interaction=nonstopmode -halt-on-error -file-line-error`及对应稿名 | 两稿退出0，英文29页、中文26页；日志无Warning、未定义引用、缺图、缺字或Overfull／Underfull；引用顺序及图表编号双语一致 | 不扩展修改其他段落或排版设置 |
| 文件范围 | 两稿、本文件及构建产物 | 逆向还原批准替换后逐字节比较，与起点哈希核对 | 其他正文及引用完全不变；仅授权源文件和对应构建产物变化；BibTeX、JPG及实验产物未改；diff --check通过 | 既有未提交修改保留 |

当前PDF使用TeX同目录版本；[范围与日志核查记录](/tmp/paper6_p2_terms_l52vnrbh/audit.json)为本轮检查材料，LaTeX构建未在临时目录执行。

### 7.15 第二段删除重复解释（2026-09-07）

按确认方案删除两稿第二段原第二句，并将末尾三句替换为指定单句：由背景外观差异影响特征比较，直接落到成对可比性与结合灾前／灾后状态判别新增淹没的需求。保留总分展开、段末归纳问题，不重复抽象定义；“提高可比性”仅为研究需求，不是已验证性能。“深度学习方法”、代表研究及全部引用保持不变。7.14末尾三句写法为历史版本，当前以本节单句替换为准。

| 修改项 | 涉及文件 | 修改与依据 | 审查结果 | 剩余阻断 |
|---|---|---|---|---|
| 删除与合并 | 两稿第二段、本文件 | 精确采用批准文本，删除重复第二句，最后三句合为一句 | 总述—参考研究—深度学习进展—应用区别需求—判别问题顺序保留；其他正文及全部引用不变 | B1—B8不变；可比性提高不作已验证效果 |
| 编译与一致性 | 两稿PDF及中间文件 | 在`paper6_en/`依次执行两稿`latexmk -xelatex -interaction=nonstopmode -halt-on-error -file-line-error` | 两稿退出0，英文28页、中文26页；日志无Warning、未定义引用、缺图、缺字或Overfull／Underfull；双语引用及图表编号一致；英文引言1043词 | 不为旧词数目标增写其他段落 |
| 范围核查 | 两稿、本文件及构建产物 | 将本轮指定替换应用到起点副本，与当前源文件逐字节比较；核对文件哈希 | 仅授权文本及对应构建产物变化；BibTeX、JPG及其他文件未变；diff --check通过 | 保留既有未提交修改，未新增实验 |

PDF为TeX同目录当前版本；[本轮核查记录](/tmp/paper6_p2_concise_6y2flg94/audit.json)保留修改范围及构建日志检查结果。

### 7.16 第三段七篇文献分类重写（2026-09-07）

仅重写两稿I3、增加必要参考文献及更新本文件当前规划。按批准文本保留开头两句和末尾两句；中间按多层／多尺度提取、特征选择、地表辅助约束分类，引用七篇不同论文（2022年两篇、2023年一篇、2025年两篇、2026年两篇）。复用Siam-DWENet与Zhu等已有条目，新增MS-Deeplab、SISCNet及Yadav等三个条目。I2状态判别和I4内部覆盖／边缘定位均保持原文，不引入模块作用结论或新增实验。

| 修改项 | 涉及文件 | 修改与依据 | 审查结果 | 剩余阻断 |
|---|---|---|---|---|
| I3叙事与文献 | 两稿第三段、reference.bib、本文件 | 按W-L分类，来源详见5.5；删除空间支持范围、同一幅影像、面积与变化幅度对照 | 七篇／五篇2022—2025年；段末提出大小淹没协调需求；单时相、双时相与多源概率路线明确区分 | 部分来源仅取得出版索引摘要／方法片段，精确上线日TBD已定位；不据此断言缺失机制 |
| 双语与修改范围 | 两稿、本文件、三个新增BibTeX条目 | 同步方法关系、任务区别及结论强度 | 将新第三段逆向还原后与起点逐字节比较，两稿其他正文完全不变；两稿七篇引用顺序一致，既有BibTeX内容字节不变，仅追加三条；JPG哈希不变 | B1—B8保留；本轮文献扩充不关闭专项或模块级证据缺口 |
| 编译与排版 | TeX同目录PDF及构建文件 | 按用户要求在paper6_en依次编译 | 两稿latexmk退出0；英文30页、中文26页；最终日志无Warning、未定义引用、缺字及Overfull／Underfull，图1—12与表1—4编号双语一致；英文引言1078词 | 不调整其他段落以填补旧词数预算 |

本轮仅四份授权源文件及对应构建产物变化，保留原有未提交修改。`git -c core.whitespace=cr-at-eol diff --check`通过；目视检查英文第3页与中文第2—3页的第三段及衔接，未见明显溢出或缺字。PDF为TeX同目录版本：[英文](../elsarticle-template-harv_2.pdf)、[中文](../Manuscript_revised_cn.pdf)。[核查记录](/tmp/paper6_p3_literature_gbgelbpp/audit.json)含起点范围还原、引用年份、图表编号和文件哈希结果；临时目录仅存核查材料，不作LaTeX构建目录。本轮段落修改完成不代表B1—B8关闭，投稿定稿仍受既有证据问题限制。

### 7.17 第四段技术路线比较重写（2026-09-07）

| 修改项 | 涉及文件 | 修改与依据 | 审查结果 | 剩余阻断 |
|---|---|---|---|---|
| I4路线与问题 | 两稿第四段、本文件I4／5.6 | 首句使用洪水，删除面积相近；按专用网络→基础模型不同用途→重建与监督→标签组织十篇引用，段末采用批准文本 | 总分展开；不把模块缺席等同缺口；不提前介绍HA-CQI模块，不扩展I2／I3及贡献 | B1—B8不变；总体指标不能证明语义融合或细节专项作用 |
| 文献与双语 | reference.bib、两稿 | 新增七条，补全SemT-Former DOI；DINOv3保持技术报告身份；任务及输入条件分别说明 | 第四段十篇不同来源、双语引用顺序及全文引用序列一致，无重复或缺失key；BibTeX仅新增七条并补全一个DOI | 部分来源精确上线日／全文可读范围已在5.6登记，未用摘要推断缺失机制 |
| 编译及范围 | 四份源文件及同目录构建产物 | paper6_en依次编译，按本轮起点逐字节比较 | 两稿latexmk退出0，英文31页、中文27页；最终日志无Warning、未定义引用、缺字或Overfull／Underfull；图1—12、表1—4编号双语一致；逆向还原P4后其余正文逐字节不变，JPG哈希不变 | 不新增实验，不扩大其他段落修改 |

本轮英文引言1155词（沿用剔除引用与LaTeX命令的计数口径），不向其他段落补写。目视检查中文第3页及英文第3—4页：段落与第五段衔接正常，无明显溢出；保留自然跨页。`git -c core.whitespace=cr-at-eol diff --check`通过。仅四份授权源文件和对应构建产物变化，既有未提交修改保留。[核查记录](/tmp/paper6_p4_routes_3rtvjq9p/audit.json)保留范围、引用及编号审查；临时目录只存核查材料，LaTeX在TeX同目录执行。当前PDF：[英文](../elsarticle-template-harv_2.pdf)、[中文](../Manuscript_revised_cn.pdf)。第四段重写已完成，B1—B8仍限制投稿定稿。

### 7.18 第四段变化检测路线重构（2026-09-07）

| 修改项 | 涉及文件 | 修改与依据 | 审查结果 | 剩余阻断 |
|---|---|---|---|---|
| I4问题与路线 | 中英文TeX第四段 | 洪水变化检测开篇；专用网络五篇与基础模型适配五篇均衡评述；按5.7来源组织 | 总分展开、段末归纳区域判断与局部定位问题；第二、三、五段及贡献保持 | B1—B8保留；不新增专项效果结论 |
| 引文元数据 | reference.bib | 补正CGNet、SAM-CD、ASS-CD及PeftCD DOI；PeftCD作者Libo与题名大小写；无条目增删 | 正式年份与预印本年份分开，稳定key保留；双语引用一致 | PeftCD最终卷页／精确上线日按5.7登记，不虚构 |
| 工程与范围 | 两稿构建产物、本方案 | 在paper6_en依次执行latexmk -xelatex；源文件范围与日志核查 | 两稿编译成功；无未定义引用、缺字或Overfull／Underfull；图表编号及全文引用序列一致；非第四段正文逐字节未变 | B1—B8仍限制投稿定稿，不运行实验 |

本轮英文引言1160词（沿用剔除引用与LaTeX命令、连字符词计一词的口径）。中文PDF 27页、英文PDF 31页；目视检查中文第3页和英文第3—4页，第四段与前后段自然衔接，无明显排版溢出。四份授权源文件之外仅对应LaTeX构建产物变化，所有既有JPG哈希不变，其他正文和既有无关修改保留。BibTeX可由修改前文件仅应用四条元数据修订逐字节重建；无条目增删。`git -c core.whitespace=cr-at-eol diff --check`通过。[本轮范围和构建核查记录](/tmp/paper6_p4_cd_kmhs83jy/audit.json)。当前PDF：[英文](../elsarticle-template-harv_2.pdf)、[中文](../Manuscript_revised_cn.pdf)。本轮第四段修改完成，不代表B1—B8证据问题已关闭。

### 7.19 引言结尾：完整概述与三条贡献（2026-09-07）

| 修改项 | 涉及文件 | 修改与依据 | 审查结果 | 剩余阻断 |
|---|---|---|---|---|
| I6工作概述 | 两稿引言结尾 | 采用批准中文段落并同步英文，说明目标、检测思路及下游估深；借鉴AWCA-Net引言的概述—贡献组织 | 不列性能数字；流程衔接不称创新；HA-CQI不直接回归水深 | B1—B8不变 |
| I7三条贡献 | 两稿及本方案当前I6／I7规划 | 成对判别、多尺度及空间重建、条件水深三条；删除“具体回应如下”和旧列表后解释 | 前两条为HA-CQI设计贡献，第三条为CFDepth下游扩展；无单模块效果承诺 | 第五段六至七篇文献扩充仍独立待实施，本轮保持原文 |
| 工程检查 | 两稿构建产物 | 在paper6_en依次构建；逐字节核对授权结尾之外正文 | 两稿编译通过；最终日志无Warning、未定义引用、缺字或Overfull／Underfull；结尾外正文逐字节保持，引用及图表编号一致 | 不运行实验 |

本轮英文Introduction为1273词（剔除引用和LaTeX命令，连字符词计一词）。保留历史记录中的旧词数与四条回应描述，其不再作为当前执行要求。

本轮仅两稿TeX结尾、本方案及对应构建产物变化，reference.bib、图件和其他正文不变。中文第4页、英文第5—6页进行目视检查。英文第二条标题用different替代diverse消除轻微行宽溢出；英文引言末加入与中文一致的\clearpage，防止后续研究区图浮入贡献列表。此为引言结尾局部分页控制，未修改图环境或后续章节。核查材料：[范围、引用及日志审查](/tmp/p6_intro_end_w0qd703w/audit.json)。B1—B8继续保留，本轮完成不代表投稿定稿已就绪。

### 7.20 删除工作稿AI声明与中文版TBD编辑备注（2026-09-07）

按用户确认删除两稿AI声明标题及整段正文。中文版主文件9处TBD中1处随声明删除，其余8处编辑备注清理；3个实际输入表格各清理1处中文备注。英文其他TBD及表格英文分支保留。未引用的历史消融表、图像文字和实验产物不改。历史声明与使用记录仅为历史，不构成当前恢复声明的要求，也不表示未使用AI。

删除备注按修改前行号追踪如下；其中核查事项继续待办，不因移出正文而关闭。

| 原文件及行号 | 原备注全文 | 证据归属与处理 |
|---|---|---|
| `Manuscript_revised_cn.tex:195` | [TBD: 核实照片坐标、拍摄时间及区间估计来源。] | B6；正文保留照片区间未核验且不作测量 |
| `Manuscript_revised_cn.tex:214` | [TBD: 查明河南GF3窗口与郑州评价域的空间重合关系，并闭合该训练记录与预测产物的对应。] | B2／B3；正文保留混合来源和独立性限制 |
| `Manuscript_revised_cn.tex:216` | [TBD: 明确各场景产物对应的预处理配置、标签来源、有效像元掩膜及配准记录。] | B6；正文保留标签与处理来源不完整 |
| `Manuscript_revised_cn.tex:220` | [TBD: 明确USGS产品及事件、垂直基准、支撑区来源、DEM版本，以及生成比较栅格的CFDepth/FwDET版本。] | B4／B6；深度产物、参考与来源问题仍未闭合，保留既有条件限制 |
| `Manuscript_revised_cn.tex:478` | [TBD: 将每个比较条目与检查点、预测版本、阈值及评价报告逐一对应。] | B1／B2；不补造指标，不进行排名 |
| `Manuscript_revised_cn.tex:510` | [TBD: 在将差异归因于具体方法前，核实（d）--（l）列的方法与检查点对应。] | B2；图注改为各输出列与具体方法及检查点的对应尚未核实 |
| `Manuscript_revised_cn.tex:532` | [TBD: 明确生成版本、准确支撑区及NoData定义；照片区间不作为验证。] | B4／B6；深度产物、参考与来源问题仍未闭合，保留既有条件限制 |
| `Manuscript_revised_cn.tex:560` | [TBD: 闭合参考产品身份、垂直基准及生成版本来源。] | B4／B6；深度产物、参考与来源问题仍未闭合，保留既有条件限制 |
| `Manuscript_revised_cn.tex:597` | [TBD: 作者核验最终内容、补充准确服务与模型版本，并在投稿前确认最终责任声明。] | B8；声明随整节删除，保留历史与最终投稿核验事项 |
| `tables/table_gf3_eval_dataset.tex:15` | [TBD: 对照原始采集元数据核实日期、极化、成像模式及像元间距；该信息表不证明评价独立性。] | B3／B6；自然表述保留未核验及不证明独立性 |
| `tables/table_main_comparison.tex:24` | [TBD: 闭合各行的方法--检查点--预测--阈值--有效域对应。] | B1／B2；不补造指标，不进行排名 |
| `tables/table_cfdepth_validation.tex:16` | [TBD: 将评价栅格与生成实现、参考产品及垂直基准对应。] | B4／B6；深度产物、参考与来源问题仍未闭合，保留既有条件限制 |

| 修改项 | 涉及文件 | 修改与依据 | 审查结果 | 剩余阻断 |
|---|---|---|---|---|
| 声明删除 | 两稿主TeX | 用户确认删除标题和整段正文 | 不影响致谢及参考文献 | B8最终投稿事项仍待核验 |
| 中文编辑标记 | 中文主TeX、三个共用表格中文备注 | 删除编辑指令，保留必要科学限制；记录如上 | 中文PDF检索TBD为0；主文件及实际表格中文分支均清理；英文分支逐字节保持 | B1—B8均不视为关闭 |
| 工程与范围 | 两稿构建、本方案 | paper6_en同目录串行构建、PDF检索、引用编号及源文件审查 | 两稿编译成功，日志无Warning、缺字、未定义引用或Overfull／Underfull；引用与图表编号检查通过 | 无新增实验 |

本轮最终中文PDF为26页、英文31页；英文保留11处TBD，中文为0处。两稿均无AI使用声明；英文声明之外正文逐字节不变，表格仅中文备注变动，数值及英文分支保持。检查中文第5页数据表、第16页比较图与第20页致谢—参考文献衔接，无空包装或明显排版溢出。仅6份授权源文件及对应构建产物变化，BibTeX、图片和历史消融表哈希不变。`git -c core.whitespace=cr-at-eol diff --check`通过。[完整核查记录](/tmp/p6_cleanup_dmok44qi/audit.json)。

### 7.21 首段原理与两阶段制图贡献（2026-09-07）

| 修改项 | 涉及文件 | 修改与依据 | 审查结果 | 剩余阻断 |
|---|---|---|---|---|
| I1原理句 | 两稿首段 | 在SAR观测优势后补一句灾前参考、对应位置比较与新增洪水判别；复用cian2018flood | Cian原刊摘要明确先用SAR变化检测获范围、再结合地形估深；不移植其指数机制或性能 | 不声称解决物理幅度校正 |
| I6／I7 | 两稿及当前规划 | 两阶段方法目标、范围从第一阶段传入第二阶段；三条贡献均取消加粗，第三条改为范围内水深估算 | HA-CQI与CFDepth共同服务洪水制图，不等于联合训练或已验证整体精度；前两条内容除加粗外不变 | B1—B8保留；条件估深不修复漏检 |
| 工程与范围 | 两稿构建、本方案 | 在paper6_en依次编译，逐字节验证授权句之外正文 | 两稿编译通过；最终日志无Warning、缺字、未定义引用或Overfull／Underfull；授权句段外正文逐字节不变 | 无新增实验 |

当前英文Introduction为1340词（剔除引用和LaTeX命令，连字符词计一词）。[Cian原始来源](https://nhess.copernicus.org/articles/18/3063/2018/)摘要为本轮原理概括依据；已有BibTeX不改。历史实施记录中的下游扩展描述不再是当前I6／I7执行要求。

本轮中文PDF 26页、英文31页。中文第2／4页和英文第5／6页已目视检查：新增原理句位置正确，贡献无加粗，列表正常跨页，章节承接无图件插入。中文TBD检索为0；两稿引用序列一致，较修改前仅首段新增一次cian2018flood；图表编号一致。仅两稿、本方案及对应构建产物变化，reference.bib、图表源文件及图件哈希不变。`git -c core.whitespace=cr-at-eol diff --check`通过。[范围、引用、编号及日志检查](/tmp/p6_twostage_p08uke7_/audit.json)。B1—B8未关闭，不新增联合精度或模块效果结论。

### 7.22 首段SAR成像原理与范围—水深衔接（2026-09-07）

| 修改项 | 涉及文件 | 修改与依据 | 审查结果 | 剩余阻断 |
|---|---|---|---|---|
| SAR物理解释 | 两稿I1 | 用主动微波成像、表面趋于光滑时的反射方向与返回信号减弱替换任务描述；引用既有Amitrano2024条目 | 仅两句；保留when／can条件，不宣称所有洪水变暗，不将观测解释作为HA-CQI物理校正功能 | 无新增机制效果结论 |
| 逻辑与标点 | 两稿I1、本方案I1 | 中文分号改句号；英文首段分号亦改句号；补明暗不直接对应水深及地形估深联系，Cian引用移至此处 | 由SAR范围线索递进到水深需求，再归纳两阶段制图价值 | B1—B8不变 |
| 编译及范围 | 两稿构建产物 | paper6_en同目录依次编译；引用、编号及授权范围核查 | 两稿编译通过；日志无Warning、缺字、未定义引用或Overfull／Underfull；首段外正文逐字节不变 | 无新增实验 |

原理依据：[Amitrano等（2024），Flood Detection with SAR: A Review of Techniques and Datasets](https://www.mdpi.com/2072-4292/16/4/656)，表面散射与水面粗糙度相关段落。估深衔接依据：[Cian等（2018）](https://nhess.copernicus.org/articles/18/3063/2018/)，SAR变化范围与地形估深路线。两篇现有BibTeX条目均不修改。7.21中的任务描述为历史，当前由本节物理说明替代。

当前英文Introduction为1395词（剔除引用与LaTeX命令，连字符词计一词）。

本轮最终中文PDF 26页、英文31页；两稿第2页目视检查通过，首段原理及逻辑衔接清楚，无明显溢出。首段引用顺序均为Tellman、Wagner、Amitrano、Cian；全文引用序列与图表编号一致。中文版PDF的TBD为0。仅两稿首段、本方案及对应构建产物变化，BibTeX、图件及其他源文件哈希不变。`git -c core.whitespace=cr-at-eol diff --check`通过。[本轮范围、引用与日志审查](/tmp/p6_physics_dkjt1dg9/audit.json)。B1—B8继续保留，不以本轮语言修改宣布证据闭合。

### 7.23 第四段拆分为专用变化检测与基础模型适配（2026-09-07）

| 修改项 | 涉及文件 | 修改与依据 | 审查结果 | 剩余阻断 |
|---|---|---|---|---|
| I4a／I4b拆段 | 两稿原第四段 | SemT-Former后增加批准过渡句并换段；新段用适配与特征整合开篇，余下评述及结尾保留 | 两段各五篇、引用顺序不变，不制造技术继承或新增创新点 | B1—B8不变 |
| 位置映射 | 本方案4.2及引用分配 | 原I4拆I4a／I4b，对应正文4／5；原I5／I6／I7对应正文6／7／8，稳定ID保留 | 水深段后续称“水深段（原第五段，现第六段）”；六至七篇文献扩充仍独立待实施 | 不在本轮修改水深段 |
| 工程检查 | 两稿构建产物 | paper6_en依次编译；原第四段外正文逐字节核对 | 两稿编译通过；日志无Warning、未定义引用、缺字或Overfull／Underfull；原第四段外正文逐字节保持 | 无新增实验 |

英文Introduction现为1400词（剔除引用与LaTeX命令，连字符词计一词）；不向其他段落增删内容以凑旧词数预算。历史记录中的旧段号按本节映射理解。

本轮两稿各段引用数为5＋5，全文引用顺序不变，图表编号一致。中文PDF 26页、英文31页，中文第3页及英文第4页目视确认拆段与水深段衔接正常。中文PDF未出现TBD。仅两稿原第四段、本方案和对应构建产物变化，BibTeX、图件及其他源文件哈希不变。`git -c core.whitespace=cr-at-eol diff --check`通过。[本轮范围、引用及日志检查](/tmp/p6_split_6enihta7/audit.json)。B1—B8保留。

### 7.24 第二章研究区与数据重构（2026-09-07）

本轮按用户批准的第二章方案实施，替换旧三小节、五段数据规划。参考本地 `RSE2024_Flood inundation monitoring using multi-source.pdf` 第二章的位置—场景—资料组织方式；采集参数后移按本文既定分工执行。沿用已读W-I／W-R／W-C中中心信息、证据对应及用途说明的约束，不复制参考论文的气候数字、土地覆盖来源或结果。

| 修改项 | 涉及文件 | 修改与依据 | 审查结果 | 剩余阻断 |
|---|---|---|---|---|
| 研究区四段 | 两稿第二章2.1及图1图注 | 围绕图1位置、地表格局、P1—P8展开，说明两景联系 | 图示观察与事件背景分开，不推定面积比例和成因 | 土地覆盖产品、年份与照片来源未闭合，见下 |
| 资料三段 | 两稿2.2及图2图注 | SAR影像、数据集标签、地形与深度参考按用途组织；保留sec:depth_data | 图2仍在第二章；Brazos不评价检测，照片不验证水深 | B3、B4、B6保留 |
| 参数迁移 | 两稿4.1 | GF3表、样本数与划分、三通道和归一化说明、DEM间距迁入；训练配置和指标公式未改 | 参数集中于4.1，表格文件和数值未改 | 河南GF3空间重合及预测对应未闭合 |
| 工程与范围 | 两稿及本方案 | 引言、方法实质内容、结果、图像和参考文献不改 | 两稿在paper6_en/内依次latexmk -xelatex编译通过；第二章4＋3段，图1/2与表1—4编号一致；无未定义引用、重复标签、缺字或Overfull/Underfull警告；中文版PDF中TBD为零 | B1—B8不因结构重构关闭 |

来源追查读取图1及同名TXT、原稿相关叙述和已有图件资料，未找到可闭合以下事项的记录：

- [TBD: 图1(a)(b)土地覆盖底图的产品、年份及原始栅格或制图脚本未找到；仅按图示格局描述，不计算地类比例或指定产品。]
- [TBD: 图1红色边界与GF3实际评价域的空间对应未闭合；区域背景图不作为评价域边界。]
- [TBD: P1—P8的来源、坐标、拍摄时相及水深区间依据未闭合，归B6；照片仅作场景展示。]

英文原图注的照片核验备注由正文明确限制和本节B6记录承接，其他英文数据核验备注迁入4.1。中文不增加TBD。未运行任何新增实验。

本轮最终检查：两稿章节与4.1以外的正文与任务开始快照逐字节等效（保留各自原换行格式）；图像、TXT、表格源文件及reference.bib哈希均未变。变更限两稿、方案和编译产物。图1/2改用单栏figure的H定位，解决页顶浮动超前于章节／首次引用的问题；已查看中文版第5页及英文版第7、9页渲染，图像、图注无裁切。英文图1/2位于第7/9页，中文位于第5/7页。稳定label顺序为图1—12、表1方法配置／表2GF3／表3总体比较／表4深度比较。既有无关改动保留，未修改算法、数据或实验结果。可实施的第二章重构完成，来源和评价证据未闭合，投稿定稿仍受阻。

### 7.25 第二章四段精简与GF3展示后移（2026-09-07）

本轮替换7.24及其后的图2第二章布局要求，历史记录仅作修改追踪。当前第二章采用正向场景与资料叙述，不把来源核查任务反复写入正文。具体来源缺口仍归B1—B8；未核验的照片水深区间仅在图1图注说明一次。

| 修改项 | 涉及文件 | 修改与依据 | 审查结果 | 剩余阻断 |
|---|---|---|---|---|
| 四段正文 | 两稿第二章 | 两节各两段，位置与郑州、涿州及联系、GF3观测、FABDEM用途 | 删除推断道路建筑间细节可恢复的表述，无训练资料混入 | 图1底图年份、产品及照片来源仍待核验（B6） |
| 资料表 | table_gf3_eval_dataset.tex及两稿2.2 | 原GF3参数不变，增FABDEM约30 m及用途引用 | 资料表前置；未知版本、基准未补造 | GF3原始元数据及实际DEM版本仍未闭合 |
| 实验资料 | 两稿4.1 | GF3图与掩膜说明后移，合并标签、训练及Brazos信息 | 既有训练策略、指标与结果保留；来源引用指向4.1 | B1—B4、B6保留 |
| 图件与TXT | figure2—6.jpg/txt、两稿路径 | 循环安全迁移；新图6为GF3，方法图2—5 | 两稿图1—12顺序一致，JPG逐一哈希匹配，TXT编号与路径同步 | 旧模型图机制差异和重绘任务不因改号关闭 |

未新增实验，不修改算法、数据、权重或实验结果。旧四份模型图提示词保持内容，只同步编号和用途状态。当前图件表和D1—D4为执行依据，7.24及更早段落编号仅为历史。

本轮验收：两稿第二章均为2＋2主题段落，无训练资料段；资料表1包含原GF3两行及FABDEM分区，方法配置表2，结果表3/4。两稿已在paper6_en/内依次执行latexmk -xelatex构建，最终无未定义引用、缺字及Overfull/Underfull警告，中文版PDF无TBD。图6保持78%正文宽度；实验章新页起排，图文同页（中文15页、英文18页），已查看渲染确认无裁切。英文资料表位于第8页，中文第6页。稳定label及图表计数检查通过。源文件差异与起始快照核对，修改限第二章、4.1资料、必要来源交叉引用、图片路径与分页；引言、方法实质、指标公式和结果正文未改。全部JPG按新旧映射哈希一致，参考文献库及其余表格未改。保留已有无关修改，未运行实验。可实施修改完成，B1—B8仍影响投稿定稿。
