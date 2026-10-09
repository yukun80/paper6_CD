# 运行准备与场景配置

[返回目录](README.md) · [项目首页](../../README.md)

## 环境

[项目配置](../../CCDepth/pyproject.toml) 声明 Python >=3.11，依赖 numpy、scipy、rasterio、numba。
当前精简副本不携带环境快照；旧文档中的 Figure11 环境和 Windows 绝对路径不作为本机前提。
先在已有环境检查入口：

```bash
cd CCDepth
python -m ccdepth_local --help
```

## 运行步骤

在具备所需依赖的 Python 环境中，从项目根目录进入 `CCDepth/`。
以下是使用步骤示例；先准备真实输入并修改自己的配置副本，再执行检查及计算：

```bash
cd CCDepth
python -m ccdepth_local check --config configs/pakistan_relaxed_050m_lean.json
python -m ccdepth_local run --config configs/pakistan_relaxed_050m_lean.json
# 中断后恢复同一运行，或重试成功产品的工作区清理：
python -m ccdepth_local run --config configs/pakistan_relaxed_050m_lean.json --resume
# 成功清理后仍可独立审计，不需要原始输入及预处理数组：
python -m ccdepth_local audit --run-dir runs/CCDepth/pakistan_relaxed_050m_lean_v1
```

也可直接使用已配置的 Conda 环境中的 `python`，或在安装本地包后使用 `ccdepth-local`。支持分步 `prepare --config ...` 和 `solve --run-dir ...`。新配置输出至独立的 `runs/CCDepth/...`，不会覆盖旧结果。配置中的相对路径以配置文件目录为基准。

Guangxi、Zhengzhou 和 Zhuozhou 的四组评估使用 `configs/guangxi_label_relaxed_lean.json`、`configs/zhengzhou_label_relaxed_lean.json`、`configs/zhengzhou_seg_relaxed_lean.json` 和 `configs/zhuozhou_label_relaxed_lean.json`。四组配置使用 `rowcol_direct_v1`，具体含义见[输入与网格约定](inputs.md)。按固定顺序运行：

```powershell
.\scripts\run_four_depth_evaluations.ps1
# 中断后从同一配置恢复：
.\scripts\run_four_depth_evaluations.ps1 -Resume
```

**输入现状（2026-10-09）**：逐份解析现有 12 份配置后，所引用的 mask 和 DEM 路径均不存在，
包括 `guangxi_label_updated_lean.json`。因此不能把这些示例当作本机开箱即用命令。
配置中的 `../../datasets` 相对于配置所在目录解析；请按真实数据位置制作自己的配置，
并设置独立输出目录。不要仅凭名称相似替换输入。
上面的四场景脚本为 PowerShell 脚本，需在该环境内从 `CCDepth/` 运行。

原默认参数及原配置保留。新配置沿用残差门槛 0.5 m、目标相对变化门槛 0.001。连续两次稳定检查、2000 轮预算、硬约束和次目标预算不变。整景耗时与像元一致性须在真实运行后核实，不能由本次文档核对推断。

相关内容：[输入要求](inputs.md) · [恢复](recovery.md) · [高覆盖模式](coverage.md)
