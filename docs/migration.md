# 算法文档整理清单

[返回项目首页](../README.md) · [文档维护约定](maintenance.md)

整理日期：2026-10-09。范围仅为 HA-CQI、CCDepth 的 9 份 Markdown 说明，以及根入口和规则。
论文、出图、数据、程序及既有结果不属于本次迁移范围；已移除的两个交付包不纳入。

## 原文去向

下表旧位置以项目根目录为基准；已删除的旧文件只列名称，不提供失效链接。

| 原位置 | 当前去向或处理 |
| --- | --- |
| `HA-CQI/README_zh-CN.md` | 改为简短 [README](../HarmoSSM/README.md)；正文拆入环境、数据、训练、预测与验证章节 |
| `HA-CQI/run.md` | 合并到[数据](harmossm/data.md)、[训练](harmossm/training.md)、[训练记录](harmossm/checkpoints.md)、[验证](harmossm/validation.md)；删除旧文件 |
| `HA-CQI/model_design/paper_narrative.md` | 拆为[设计概要](harmossm/design.md)、HA、CQI、OSCD 四篇；删除旧文件 |
| `HA-CQI/dinov3/weights/readme.md` | 下载入口并入[环境与权重](harmossm/environment.md)；删除旧文件 |
| `HA-CQI/model_design/model_optimize.md` | 历史优化方案，原位原文保留；从[历史资料](harmossm/history.md)访问 |
| `CCDepth/README.md` | 保留短入口；正文拆入运行、输入、输出、恢复及历史说明 |
| `CCDepth/HIGH_COVERAGE.md` | 迁至[高覆盖模式](ccdepth/coverage.md)；删除旧文件 |
| `CCDepth/CCDepth_local_逐模块构建与验收方案.md` | 历史开发规格，原位原文保留；从[历史资料](ccdepth/history.md)访问 |
| `CCDepth/IMPLEMENTATION_DECISIONS.md` | 历史修订记录，原位原文保留；从[历史资料](ccdepth/history.md)访问 |

根 README 新增统一导航；根 AGENTS.md 保留规则与链接，详细算法操作归入 docs。
历史文件是“当前说明集中到 docs”的明确例外，不生成第二份历史正文。

## 已确认的过期内容修正

- 删除依赖不存在的 `requirements-oscd.txt` 的安装命令，将旧版本组合明确标为历史记录。
- 删除不存在的 `tests/test_pipeline_contracts.py` 的运行命令，改列当前存在的评价测试。
- CCDepth 示例改用项目相对位置，不再要求旧电脑上的 Windows 绝对路径。
- 当前 12 份 CCDepth 配置引用的 mask、DEM 路径均不存在；不再称更新版广西配置可直接使用。
- CQI 实现为五个尺度分别维护查询，修正旧设计末段“全网唯一一套”的歧义。
- 根规则不再把已移除的集成包当作当前入口，并补上现有本地 CCDepth 实现。

## 验证及证据边界

- 9 份原说明中，6 份当前说明完成拆分或合并，3 份历史文件逐字保持不变。
- 根 docs 新增 22 份短文档（包括目录、维护约定和本清单）；根及算法目录另保留 3 个 README。
- 根入口、工作规则和 docs 共 26 份文件均可从首页到达；全部本地链接及 18 段 Bash 示例语法检查通过，所有新整理文件均不超过 150 行。
- 8 个帮助入口实际运行成功：训练、测试、单对预测、切片、标签准备、整景预测、整景评价及 CCDepth。
- 对照实际帮助输出检查 15 段命令示例，未发现不支持的参数；说明中明确引用的 17 个程序及配置路径均存在。
- 使用本机 `hacqi` 环境运行现有整景评价测试，4 项通过；关键依赖导入成功，Torch 为 2.4.0+cu121，`pip check` 未发现依赖冲突。
- 对照整理前记录，两个算法目录中除授权变动的说明外，47,364 个文件的大小和修改时间未变；其中已记录内容摘要的程序、配置和历史原文均一致。
- Git 变动仅涉及授权范围内的 Markdown 文档；格式检查通过。检查使用临时文件，没有向项目新增检查工具或自动任务。

本次没有启动完整训练或整景计算，不构成模型精度、水深结果或 GPU 性能复现。
CCDepth 配置引用的真实输入缺失；从零重建 HA-CQI 环境的完整安装清单未提供；
外部权重下载链接及其文件对应关系未核实。这些限制已写入对应使用说明。
历史记录中的数值及完成结论不视作本次独立复现。

后续算法名称与路径已统一为 HarmoSSM，参见[更名说明](harmossm/rename.md)。本页原位置及原验证记录保留当时名称。
