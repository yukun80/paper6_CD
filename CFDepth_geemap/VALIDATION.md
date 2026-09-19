# 验收记录（2026-09-20）

本记录区分客户端实现、真实GEE接口、small产品和真实场景性能。
原 `CFDepth/CFDepth_0919.txt` SHA256 仍为
`20ddf390e86bfe1498e59701386741a12cce651129d7c35b16c66cc7a57756bc`，本次未改动。

## 已通过：环境与本地测试

- hacqi安装 earthengine-api 1.7.43、geemap 0.37.2、geedim 2.0.0及新增依赖。
- 原有已安装包无版本变化；Torch 2.4.0+cu121、Torchvision 0.19.0+cu121、NumPy 1.26.4、SciPy 1.13.1、Rasterio 1.3.11可导入；pip check通过。
- 22项Python测试通过：300个六候选更新与原JS/SciPy对照、50个边界权重对照、200个状态迁移对照，梯度/空值、真实客户端图构造、阶段合同、持久化/锁/失败恢复、下载元数据及代理作用域检查。
- 原JS的13项数值测试及独立整图SciPy对照通过：目标差约1.78e-15，最大水面差约8.44e-9 m。
- Python编译检查通过。服务器求值没有由本地模拟替代。

证据：`environment/check.json`、`tests.log`、`js-scipy.log`、`before.json`、`after.json`。

## 已通过：真实GEE接口与数值小图

用户已完成认证，`yukun80`初始化与FABDEM访问通过。
`tests/cloud_probe.py` 不创建资产的真实求值结果：

| 检查 | 结果 |
|---|---:|
| 非平衡扫描最大水面更新 | 2.2866106890 m |
| 主目标（扫描前→后） | 276.0852550 → 10.2959399 |
| 硬下界违规 | 0 |
| 改变另一分量后本分量的差异 | 0 |
| 解析梯度最大误差 | 1.94e-13 m/m |
| 0.4小数掩膜加权和（生产/独立参考） | 10.8 / 10.8 |

证据：`environment/cloud-probe.json`。

## 已通过：完整small、续接、删除和本地下载

运行 `cfdepth_py321_small_01`：

- components：418个支持像元、4个分量；prepared以及4轮state任务均成功。
- 第2轮完成后主动退出；从实际第2轮续接，没有重新创建components或prepared。
- 每次最新state核验和本地授权落盘之后删除旧state，最终仅保留components、prepared、state_00004三个云端资产。
- 2个分量支持不足，共3个像元；其余415个像元接受解。无运行中、求解失败或最终审核失败分量。
- 深度/梯度直接下载本地，未创建Drive任务。两个GeoTIFF为36×36、Float32、EPSG:4326、原网格、NoData=-9999。
- 深度有效像元415个，全部0.5 m；梯度有效像元415个，最大约1.84e-15 m/m，满足原≤1e-10的零梯度容差。
- 下载曾中断，修复连接配置后从最终state恢复下载，没有重新求解。

结果在 `outputs/cfdepth_py321_small_01/`；调度和日志在 `runs/cfdepth_py321_small_01/`。

另运行 `cfdepth_py321_small_keepall_01` 保留全部state作为对照：6个阶段任务成功，保留6项资产。
两次最终水深、掩膜、网格完全一致；梯度最大差约7.02e-16 m/m，符合原容差，但不宣称逐比特相同。
最终state的status/attempt/sweeps/stable完全一致，水面最大差约4.26e-14 m。
证据：`environment/cloud-retention.json`、`retention-products.json`以及两个运行的journal和产品report。

真实9块下载探针 `tests/cloud_download_probe.py`：36×36网格切为9块，每像元唯一值并包含NoData孔洞；全部1296像元与独立数组逐像元相同。
证据：`environment/cloud-download-probe.json`。

## 真实接口发现并已修复的问题

1. EE把不存在和无权限合并到同一错误：创建前以父目录完整列表再次核对，不把权限错误直接当作缺失。
2. 提交request ID不同于operation ID：持久化服务端返回ID，丢响应时按唯一描述查询任务，禁止盲目重交。
3. Float64状态直方图键为`1.0`等：严格按数值解析0–5，并核对状态像元总数与支持范围。
4. geedim 2.0的aiohttp会话默认不继承环境代理：下载期间使用限定作用域的会话适配，`trust_env=True`，退出恢复原接口；不修改第三方安装文件，不记录代理凭据。
5. geedim 2.0的`num_threads`已无效：使用`max_requests/max_cpus`落实配置的并发限制。

## 已预检但未运行

郑州输入 `projects/yukun80/assets/GF3_ZhengzhouC_label` 单波段0/1检查通过，12,919,326个有效输入像元；FABDEM原网格读取通过。
只做只读预检，没有提交郑州components、迭代或导出任务。

大场景资源开销、完整郑州水深产品及水深准确度仍待实际运行和独立验证。
small通过证明本次所覆盖路径可运行，不代表所有真实场景均已验收，不代表精度提升。
