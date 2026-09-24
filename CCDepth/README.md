# CCDepth_local

离线复现固定提交 `a88f1dd7897fc479e85ee2f6191506017a920457` 的 CFDepth 数值规则。生产不导入 ee、geemap 或 arcpy，不访问网络。

## 运行

在已配置的 Figure11 Conda 环境中：

```powershell
Set-Location 'E:\Documents\paper_library\6th_paper_city_flood\figure\figure11\CCDepth_local'
python -m ccdepth_local check --config configs/pakistan_relaxed_050m_lean.json
python -m ccdepth_local run --config configs/pakistan_relaxed_050m_lean.json
# 中断后恢复同一运行，或重试成功产品的工作区清理：
python -m ccdepth_local run --config configs/pakistan_relaxed_050m_lean.json --resume
# 成功清理后仍可独立审计，不需要原始输入及预处理数组：
python -m ccdepth_local audit --run-dir runs/CCDepth/pakistan_relaxed_050m_lean_v1
```

也可直接使用已配置的 Conda 环境中的 `python`，或运行新入口 `ccdepth-local`。支持分步 `prepare --config ...` 和 `solve --run-dir ...`。新配置输出至独立的 `runs/CCDepth/...`，不会覆盖旧结果。配置中的相对路径以配置文件目录为基准。

Guangxi、Zhengzhou 和 Zhuozhou 的四组评估使用 `configs/guangxi_label_relaxed_lean.json`、`configs/zhengzhou_label_relaxed_lean.json`、`configs/zhengzhou_seg_relaxed_lean.json` 和 `configs/zhuozhou_label_relaxed_lean.json`。四组配置的 `alignment_mode` 为 `rowcol_direct_v1`：DEM 与掩膜存在平移 affine 偏差时按相同行列读取，不重采样，输出网格采用掩膜网格；偏移量和警告写入 TIFF 的 `CCDEPTH` 元数据。按固定顺序运行：

```powershell
.\scripts\run_four_depth_evaluations.ps1
# 中断后从同一配置恢复：
.\scripts\run_four_depth_evaluations.ps1 -Resume
```

注意：当前 `datasets` 中不存在 `guangxi_label_relaxed_lean.json` 和
`zhuozhou_label_relaxed_lean.json` 所引用的两个历史掩膜文件。批处理会在
对应 `check` 阶段停止；改名时未将其静默替换为名称相近但来源未核实的栅格。
当前可用的 Guangxi 标签配置为 `guangxi_label_updated_lean.json`。

原默认参数及原配置保留。新配置沿用残差门槛 0.5 m、目标相对变化门槛 0.001。连续两次稳定检查、2000 轮预算、硬约束和次目标预算不变。本次改造不自动执行整景；整景耗时与像元一致性须在真实运行后核实。

## 空间与数值合同

输入为北向 EPSG:4326 单波段同网格 TIFF。掩膜有效值严格为 0/1，本项目 NoData 为 3；DEM 按米解释，应用有效掩膜并排除非有限及 abs(z)>=1e11，保留合法负高程与零值。四角对齐容差为 1e-5 像元，使用掩膜 affine 输出，不重投影或重采样。

球面近似半径 6378137 m；纬向距离使用像元对中纬度余弦。连续性权重使用原版参考长度归一化；坡度采用 `central4_geographic_v1`。内部 float64、Numba fastmath=False，能量按确定性顺序归约。全局四色顺序、六候选更新、初始化和权重未改变。

## 最终输出

成功运行仅保留下列 8 个 TIFF；允许 GIS 后续自行生成 `.ovr`、`.aux.xml`。

- `CCDepth_WSE.tif`：成功域水面高程。
- `CCDepth_depth_solved.tif`：max(S-Z,0)，保留成功域有效零值。
- `CCDepth.tif`：FP64 的 S-Z>0.01 m 旧发布域，再编码为 Float32；不对舍入后的数值重新阈值化。
- `CCDepth_depth_signed.tif`：signed depth。
- `CCDepth_WSE_gradient_solved.tif`：水面梯度，m/m。
- `CCDepth_gradient_directions.tif`：bit0=x、bit1=y，域外255；单方向有效时保留可观测幅值。
- `CCDepth_status.tif`：0=支持不足，3=次目标接受，4=主目标回退，5=主目标失败，域外-1。
- `CCDepth_QA.tif`：1=掩膜无效，2=DEM无效，4=支持不足，8=失败，16=成功，32=主目标回退，64=非负且≤0.01，128=负signed depth，256=单方向梯度，512=hard下界活跃；背景0。

浮点 NoData=-9999；QA保留65535。每幅 TIFF 的 `CCDEPTH` 元数据命名空间内嵌实际配置、运行身份、源码/输入/预处理哈希、像元哈希和终态/失败原因汇总。`audit` 校验全部产品的像元哈希、网格、编码、有效域、状态、QA与计数；不生成外部报告。终端显示简短进度和结果。每分量详细审计仅在工作状态库中保存，成功清理后不再单独提供逐分量表。

## 临时存储与恢复

运行期间只有一个 `.work`：预处理紧凑数组、8个紧凑产品数组、SQLite状态库、必要磁盘映射及一个活动分量的检查点。不再生成逐分量结果JSON、JSONL、CSV或最终JSON报告。边界诊断仍参与原计算及测试，但 beta0、湿干样本数、诊断码不再保存为整景数组。

拓扑超预算才建立磁盘暂存目录，使用结束后关闭并清理。分量仍完整耦合；默认预算为可用内存60%、上限20 GiB，不切断邻接关系。生产求解只保留当前检查记录，测试可显式 `collect_history=True`。

短分量不写检查点。活动分量超过300秒，在下一完整检查阶段保存；已开始保存的分量也在阶段转换时保存，最多保留两代。终态不写新检查点。紧凑产品先刷新，再批量提交SQLite分量记录；存在活动检查点时立即提交完成分量，随后删除其检查点。中断后验证已提交产品摘要，未提交分量重算或从检查点继续。

8幅产品先在工作区完整生成并审计，再逐个原子发布。所有完成信息一致且最终审计通过后，关闭数据库和磁盘映射并删除 `.work`。发布中断或审计失败保留工作状态。清理失败会明确报错；即使状态库已删除，`--resume` 也可利用 TIFF 完成信息验证后重试清理，不重新求解。

恢复严格核对输入、配置、源码及预处理内容。新格式不能恢复旧源码的未完成检查点。预处理尚未完成时没有可恢复的求解状态，应改用新的输出目录重新开始。不要让两个进程同时写入同一个运行目录。恢复保证针对进程中断，不承诺断电或磁盘损坏恢复。

## 精简包说明

本目录仅保留可运行源码、配置、运行脚本、核心说明和已发布 TIFF。开发测试、冻结参考源码、阶段报告及环境清单已从精简副本中移除；这里不再提供 pytest 回归或历史构建报告。运行时使用已配置的外部 Conda 环境，环境本体不存放在本目录内。

Pakistan 历史运行仍是原样保留的 `CFDepth*.tif`。其中 7 个 TIFF 与原清单文件哈希一致；`CFDepth.tif` 文件哈希不同，但像元值均与 signed depth 一致，差异集中在 288,114 个 Float32 值 `0.009999999776482582`，与先按 Float64 的 `>0.01 m` 旧规则筛选、再编码 Float32 的输出规则相符。其余既有运行产品也未重命名或改写。

历史运行的 TIFF 保留 `CFDepth*.tif` 文件名及 `CFDEPTH` 元数据。审计器兼容旧格式；新运行只发布 `CCDepth*.tif` 和 `CCDEPTH` 元数据。

`CCDepth_local_逐模块构建与验收方案.md` 是历史实施规格，其中的测试命令和阶段报告路径属于原开发工作区，不能在本精简副本中执行或查阅。文中 `CFDepth_geemap` 与固定提交仍指上游算法来源。
