# CFDepth：Python/geemap 调度，GEE 求解，本地 GeoTIFF

本工程移植 `CFDepth/CFDepth_0919.txt` 的 v3.2.1 数值规则，原文件不修改。
Python 只负责构建云端计算、调度任务、记录状态和下载；不依赖 Code Editor 或 Notebook 页面。
**仍需要 Earth Engine、网络与计算项目权限。** 输入和计算阶段保存在 GEE，结果不经过 Google Drive。
Python 实现标识为 `cfdepth-python-ee-v3.2.1-1`，仅管理 `cfdepth_py321_` 开头的独立运行。
不会接管 JavaScript 运行，也不使用历史本地 CFDepth 算法替代云端求解。

## 环境和认证

已在 `hacqi` 中安装 earthengine-api 1.7.43、geemap 0.37.2、geedim 2.0.0。
安装前后已安装包版本保持一致，详情见 `environment/before.json`、`after.json`、`check.json`。
安装约束和新增依赖锁定表分别为 `environment/constraints.txt`、`requirements.lock`。

从仓库根目录执行（下面命令也可用 hacqi 的绝对 Python 路径）：

```bash
conda activate hacqi
cd /home/yukun80/codes/paper6_waterlogging
python CFDepth_geemap/run.py check-env
# 仅首次需要；当前用户已完成认证。不要把认证码、令牌写进配置或发送给他人。
python CFDepth_geemap/run.py auth
```

凭据使用 Earth Engine 标准用户配置目录；本工程不会复制或记录其内容。
认证与计算项目不同：配置中 `project` 为 `yukun80`，`check` 会验证初始化、临时目录读取、FABDEM和输入。
能读取资产不代表有写权限；真实 small 任务检验写入、读取与删除。

## small 验收及续接

```bash
python CFDepth_geemap/run.py check --config CFDepth_geemap/configs/small.json
python CFDepth_geemap/run.py run --config CFDepth_geemap/configs/small.json
# 已存在本地记录时必须显式使用 --resume。
python CFDepth_geemap/run.py run --config CFDepth_geemap/configs/small.json --resume
```

需要测试中断恢复时，在首次 run 上添加 `--stop-after-step 2`；保存并核验第二步后退出。
之后运行同一配置加 `--resume`，从实际第三步继续。Ctrl+C 也只停止本地调度，已提交的云端任务继续执行。
小场景验收要求418个支持像元、4个连通分量；最终415个水深有效像元，水深0.5 m（误差≤1e-7 m），有效梯度0（误差≤1e-10 m/m）。

## 郑州命令（本轮不自动运行）

配置 `configs/zhengzhou.json` 已指定：

- `flood_asset`: `projects/yukun80/assets/GF3_ZhengzhouC_label`
- `asset_root`: `projects/yukun80/assets`
- `run_id`: `cfdepth_py321_zhengzhou_01`
- `fixture`: 空字符串

先完成 small 验收，再依次执行：

```bash
python CFDepth_geemap/run.py check --config CFDepth_geemap/configs/zhengzhou.json
python CFDepth_geemap/run.py run --config CFDepth_geemap/configs/zhengzhou.json
# 中断后：
python CFDepth_geemap/run.py run --config CFDepth_geemap/configs/zhengzhou.json --resume
```

输入必须为已掩膜无效区的单波段0/1影像。若NoData=3尚未掩膜，预检会拒绝，不静默当作背景。
首版不上传本地TIFF；DEM使用 `projects/sat-io/open-datasets/FABDEM`，保持原EPSG:4326北向网格。

## 目录及配置

相对路径均按仓库根目录解析。修改输出路径、输入或数值参数后应使用新的运行名，不复用已有记录。

| 内容 | 位置 |
|---|---|
| 洪水和FABDEM | 现有EE资产 |
| components / prepared / 最新state | `<asset_root>/<run_id>_...` |
| 配置、journal.json、run.log、process.lock | `CFDepth_geemap/runs/<run_id>/` |
| CFDepth.tif、CFDepth_WSE_gradient.tif、report.json、owner.json | `CFDepth_geemap/outputs/<run_id>/` |

`parameters` 可覆盖 `cfdepth/defaults.json` 中数值参数；缺省保持与当前JS版本一致。
`poll_seconds` 默认30；`download_threads` 默认2；`cleanup_previous_states` 默认true。
关闭清理会保留之后的全部state，但不会恢复已删除资产。

每阶段10次完整四色扫描，每次主目标或中点尝试最多2000次扫描。
保持原六候选最小化、分量边界支持判断、主目标预算与收敛标准；FP64计算、Float32最终产品。
未收敛或支持不足分量不输出水深。正水深覆盖是产品可用性，不是水深准确率。
梯度单位为m/m，绘图转‰乘1000；一个方向有效时仅代表该方向可观测幅值。

## 安全恢复和滚动清理

一个运行只能有一个本地调度进程。不要在另一台机器同时运行相同run_id。
新state任务COMPLETED且合同核验通过后，将最新step、令牌和待删清单原子写入本地journal并回读，然后逐项删除本运行旧state。
components、prepared和最新state保留；不会创建云端automation文件夹/集合。

任务提交使用持久化request ID，保存服务端返回的operation ID。两者不一定相同。
响应丢失先查询任务，不盲目再次提交。任务FAILED/CANCELLED停止；网络错误有限退避30/60/120秒。
未知提交结果或同名任务歧义也停止，保留日志以便核查。

必须备份 `runs/<run_id>/journal.json`。本版缺失journal而云端已有该运行资产/任务时一律拒绝推测恢复，
无论历史state是否完整。恢复原journal或使用新run_id；不自动删除冲突资产。
最新state丢失、输入来源变化、参数/网格/波段或快照令牌冲突均停止。
关闭清理不会撤回已经写入的待删除授权；会先完成已有清理，再停止安排新清理。

下载由geemap/geedim按固定CRS、仿射网格和尺寸分块直传本地，nearest、Float32、NoData=-9999。
先写本运行临时文件，核对网格、像元数量、数值、NoData与掩膜，持久化收据后发布。
两个产品成功才写完成标志；下载失败后 `--resume` 使用最新云端state，不重新求解。
已完成运行续接会复核文件哈希；不会覆盖不属于本运行的文件。

## 本地测试

```bash
PYTHONPATH=CFDepth_geemap python -m unittest discover -s CFDepth_geemap/tests -v
python -m compileall -q CFDepth_geemap/cfdepth CFDepth_geemap/run.py
```

数值测试直接调用原JS标量内核并与SciPy独立最小化对照；接口测试使用官方Python客户端的静态算法定义，
验证真实表达式构建和参数签名，**并不模拟服务器求值**。
任务模拟覆盖锁、暂停/续接、提交和删除丢响应、失败、缺失记录/最新state、清理权限及下载失败。
实际执行证据见 `VALIDATION.md`，云端排队速度和真实场景水深性能不能由small测试推断。

官方参考：[认证](https://developers.google.com/earth-engine/guides/auth)、
[Python客户端与服务端](https://developers.google.com/earth-engine/guides/client_server)、
[geemap直接下载](https://geemap.org/common/#geemap.common.download_ee_image)。

真实接口小图与9块下载验收（需要认证，不创建资产任务）：

```bash
PYTHONPATH=CFDepth_geemap python CFDepth_geemap/tests/cloud_probe.py
PYTHONPATH=CFDepth_geemap python CFDepth_geemap/tests/cloud_download_probe.py
```

`configs/small_keep_all.json` 是已执行的保留全部state对照配置。测试产物保留供核查，默认small/郑州配置仍滚动清理。
本机通过环境代理访问GEE；geedim 2.0下载会话使用限定作用域适配继承环境代理，SSL验证保留。
`download_threads` 映射为geedim 2.0的 `max_requests/max_cpus`，不使用其已经失效的`num_threads`。
水深有效范围由云端FP64规则决定；本地Float32值可能恰好舍入到阈值，文件核验不会再次使用不同精度重新阈值化。
