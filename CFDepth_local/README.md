# CFDepth_local

离线复现固定提交 `a88f1dd7897fc479e85ee2f6191506017a920457` 中 CFDepth_geemap v3.2.1 的数值规则。生产不导入 ee、geemap 或 arcpy，不访问网络。

## 已确认空间合同

输入为北向 EPSG:4326 单波段同网格 TIFF。掩膜有效值为 0/1，本项目掩膜 NoData 为 3；DEM 高程按米解释，先应用文件有效掩膜，再排除非有限及 abs(z)>=1e11。保留合法负高程和零值。对齐容差为四角最大 1e-5 像元；使用掩膜 affine 输出，不重采样输入。

球面近似半径 R=6378137 m，经向距离为 R*dlat，纬向距离为 R*dlon*cos(像元对中纬度)。连续性权重为 lambdaC*h_ref²/d²。坡度为中心及四邻均有效时的实际米制距离中心差分 `central4_geographic_v1`。这是本地空间算子定义；本交付不要求 GEE 服务端逐像元对照。

## Windows 运行

在本目录的 PowerShell 中运行，复用 `../cenv`：

```powershell
& ../cenv/python.exe -m cfdepth_local check --config configs/pakistan_geographic.json
& ../cenv/python.exe -m cfdepth_local run --config configs/pakistan_geographic.json
# 仅在该运行已有完整 prepared 时恢复：
& ../cenv/python.exe -m cfdepth_local run --config configs/pakistan_geographic.json --resume
& ../cenv/python.exe -m cfdepth_local audit --run-dir runs/pakistan_geographic_v1
```

也可分开调用 `prepare --config ...` 和 `solve --run-dir ...`。路径相对于配置文件解析。复用已有 run 必须显式恢复；输入、源码、配置及 prepared 内容改变时拒绝恢复。环境变量只在当前 Python 进程设置，不修改系统环境。

## 数值与状态

所有内部浮点计算使用 float64；原始默认参数保持不变，仅允许通过配置覆盖 `residualTolerance` 和 `objectiveTolerance`（有限正标量），其余生产参数仍冻结。四色由整图行列奇偶确定。主目标和每次次目标分别最多 2000 sweep，检查间隔 10，连续两次稳定并满足固定点残差才算收敛。Numba fastmath=False，总目标串行确定性归约。

新增 `configs/pakistan_relaxed_050m.json` 使用最大残差 0.5 m、目标函数相对变化门槛 0.001；求解和最终分量审计使用解析后保存的实际配置。硬约束及主目标预算不变。运行命令：

```powershell
& '..\cenv\python.exe' -m cfdepth_local run --config 'configs\pakistan_relaxed_050m.json'
# 仅恢复这个新运行：
& '..\cenv\python.exe' -m cfdepth_local run --config 'configs\pakistan_relaxed_050m.json' --resume
```

新输出目录为 `runs/pakistan_relaxed_050m_obj001_v1`。新参数必须从新运行开始，不能恢复旧参数检查点。修改前源码、测试、配置、环境记录及原验收报告保存于 `snapshots/baseline_182908664974`，文件校验清单为 `snapshot_manifest.json`。原整景产品与 G00–G08 报告保留；这些历史报告不能作为新参数整景验收。此次修改不自动执行新整景，也不预判覆盖率。

状态 0=支持不足，1/2=求解中，3=接受次目标，4=恢复并接受主目标，5=主目标失败。最终产品只能包含 0/3/4/5。状态 5 的原因与诊断保存在 components.csv、components.jsonl 和 component_results；不会降低锚点门槛或增加生产预算以获得覆盖。

## 输出

- `CFDepth_WSE.tif`、`CFDepth_depth_solved.tif`：成功求解域水面与 max(S-Z,0)，保留有效零值。
- `CFDepth.tif`：FP64 中 S-Z>0.01 的旧发布域，随后编码 Float32；不重新阈值化舍入结果。
- `CFDepth_depth_signed.tif`：原始 signed depth。
- `CFDepth_WSE_gradient_solved.tif`：完整成功域梯度，m/m；方向文件 bit0=x、bit1=y。单方向有效时保留可观测幅值。
- `CFDepth_status.tif`：support 内终态，域外 -1。
- `CFDepth_QA.tif`：1=mask无效，2=DEM无效，4=支持不足，8=主目标失败，16=成功，32=主目标回退，64=非负且≤0.01，128=负signed depth，256=单方向梯度，512=hard下界活跃；背景0。
- 浮点磁盘 NoData=-9999；QA保留65535，方向域外255。run.json 仅在完整发布审计通过后产生。

## 内存、恢复和验收

prepared 保存磁盘映射标签、行优先分量索引和紧凑统计；边界按带6像元上下文的窗口计算。最大分量保持完整全局耦合，逐分量求解；拓扑分配超预算转磁盘映射。默认预算为启动可用内存60%、上限20GiB。

检查点保存 S/baseS、状态、完整检查历史及哈希身份。阶段转换和每300秒后的完整检查点保存，保留两代。产品先写临时 TIFF 再原子替换；每100个组件刷新对应紧凑数组并持久化组件日志。进程中断时未提交组件重算或从已校验检查点恢复。检查点采用文件关闭、校验及原子替换，未对每个小文件强制物理磁盘刷新；保证进程中断恢复，不承诺操作系统崩溃或断电后的存储持久性。

`python scripts/gate.py N` 运行当前阶段及累计回归，报告含源码、测试、配置哈希。旧哈希报告不能代表修改后源码。G08还要求真实整景验收测试通过。所有分量有明确终态且成功域审计通过才叫整景完成，不要求所有洪水像元都有深度，不以水深精度改善作为迁移验收标准。

原始数据及旧 CFDepth/CFDepth_geemap 不作修改。环境安装前后快照和锁定文件位于 environment。完整参考源码位于 reference/upstream，仅供测试和溯源。
