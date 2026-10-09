# Repository Guidelines

## 工作范围与入口

本仓库同时存放算法、论文和其他写作材料。各类材料保持独立，未经要求不进行文体转换或跨目录整理。

- `HarmoSSM/`：当前 SAR 洪水变化检测主线；[使用目录](docs/harmossm/README.md)。
- `CCDepth/`：当前本地水深估计实现；[使用目录](docs/ccdepth/README.md)。
- `datasets/`：源数据及处理后的数据；当前训练集见[数据说明](docs/harmossm/data.md)。
- `paper6_en/`：论文正文、图表和参考文献；不随算法文档重构搬移。
- `demo/`：出图与比较材料；不作为当前算法主线。
- 专利、技术方案及历史基线遵守下方边界；相应目录不在当前检出中时不得假定它们存在。
- `HA-CQI-CFDepth/` 及已移除的交付副本不作为当前操作入口。

## 文档同步要求

- 讨论方案时正常讨论，不要求每轮对话同步文档。
- **方案确认并执行功能修改时，必须在同一次交付中更新相关文档及目录。**
- 修改不影响文档时，在交付说明中简短说明原因。
- 当前算法正文统一放根 `docs/`，根及算法目录 README 仅保留概要和入口。
- 遵循[文档维护约定](docs/maintenance.md)：按主题逐层展开，一份内容只维护一处。
- 既有历史方案及修订记录原位原文保留，从历史入口访问；不当作当前用法。
- 不增加文档检查工具、定期任务或提交限制。

## 算法约束

- 主线为 HarmoSSM，不是 ChangeDINO。架构和说明以[模型设计](docs/harmossm/design.md)及现有程序为依据。
- 保留共享 CNN-DINO 编码器、EfficientNet-B2、冻结 DINOv3 `[5,8,11]`、ImageNet normalization、HA、五级 CQI、OSCD 及多尺度辅助监督。
- 未经明确要求，不引入历史 ChangeDINO Risk-Aware、HybridRefiner、topo-router、micro-gate 等结构。
- `best_primary` 按验证 Flood IoU 联合选择阈值；`0.40` 仅为诊断阈值。
- 仅接受 checkpoint v2；旧记录中的 HA-CQI 身份可兼容读取，新记录使用 HarmoSSM，详见[更名说明](docs/harmossm/rename.md)。20260427 等旧格式档案不受支持。
- 训练成员来自实际 train/val 下 A/B/label 的严格同名集合；`stats_mode=auto` 随训练图像成员变化更新统计。
- 每次训练从预训练权重开始；不支持续训或整模型初始化，详见[训练记录](docs/harmossm/checkpoints.md)。
- CCDepth 接收 0/1 淹没支持区与 DEM；遵守[输入约定](docs/ccdepth/inputs.md)、[输出说明](docs/ccdepth/outputs.md)和[恢复约定](docs/ccdepth/recovery.md)。
- 区分当前实现与历史 CFDepth 来源；变化检测只提供上游范围，不自动扩大水深专利保护对象。
- 不将旧 ChangeDINO 默认值、UrbanSARFloods 12 通道、分层 floodness/flood_type、pos_mIoU 优先、PPO/prompt/SAM 尝试或已移除路径当作当前规范。

## 操作与验证

- HarmoSSM：[环境与权重](docs/harmossm/environment.md)、[训练](docs/harmossm/training.md)、[预测与评价](docs/harmossm/inference.md)、[运行验证](docs/harmossm/validation.md)。
- CCDepth：[运行准备](docs/ccdepth/running.md)、[高覆盖模式](docs/ccdepth/coverage.md)。
- 修改 Python 后运行相应 `py_compile`；修改脚本后检查语法，并使用代表性输入验证实际行为。
- 数据处理脚本支持 `--dry-run` 时先预览，再进行小样本实际验证。
- 整景预测改动需检查报告、二值拼接结果、有效掩码及 NoData，不能只以程序启动成功判定完成。
- 文档变更需核对链接、执行目录和说明与现状的一致性；不得编造未执行的训练或精度验证。
- 论文修改后在 `paper6_en/` 编译 `elsarticle-template-harv_2.tex`，核对图表、引用编号；现有工具可用时使用 `latexmk -xelatex`。
- 专利或方案修改后检查材料类型、来源、术语及保护范围。

## 数据、交付与安全

- 不提交数据集、模型权重、大型栅格、缓存及大量生成结果；不将密钥或令牌写入文件。
- 训练数据根目录和统计文件必须一致；场景推理输入须有清单、A/B 图像和有效掩码。
- 本机路径优先使用环境变量或运行参数。清理生成结果前核对目标，保留用户材料，除非已明确授权删除。
- 保留用户已有改动，禁止借文档整理改写算法、数据、结果或历史原文。
- 提交信息简短明确，可用 `scope: change summary`；PR 说明目的、范围、复现命令、验证结果及数据和环境前提。

## Coding Style & Naming Conventions

- Python uses 4-space indentation, PEP8-compatible layout, and type hints for new utilities.
- Add concise Chinese comments/docstrings for key classes, key methods, and non-trivial logic blocks.
- Keep comments focused on intent and behavior; avoid restating obvious assignments.
- Prefer runtime arguments such as `--data-root`, `--tiles-root`, `--checkpoint`, `--stats_file`, and `--output-dir` over hard-coded absolute paths.
- Keep script names descriptive and task-specific.
- Use structured readers for CSV, JSON, raster, and LaTeX/BibTeX content where practical.
- Preserve the checkpoint v2 contract: require `meta.format_version == 2`, `network`, `meta.model_config`, and an explicit or selected inference threshold.

## Writing Artifact Rules

### `paper6_en/`

- Write in academic manuscript style, grounded in the current figures, tables, code, and experiment outputs.
- Use `paper6_en/elsarticle-template-harv_2.tex`, `reference.bib`, `figure/`, and `tables/` as the authoritative paper artifacts.
- Do not import patent claim wording into the paper.

### `patent/`

- Target artifact is a **Chinese invention patent application text**, not a technical disclosure.
- The CFDepth depth-estimation method in `paper6_en/elsarticle-template-harv_2.tex` is the core method source.
- The historical implementation path `HA-CQI-CFDepth/CFDepth/CFDepth_GEE.txt` is absent in this checkout. Do not treat it as a live source; verify method details against the authorized manuscript and available implementation without expanding patent scope.
- Existing documents under `patent/` are format references.
- `patent/参考文献/` is prior-art/background only, not the invention source.
- Do not write repository paths, paper titles, code names, or experiment-only details into the formal patent body.
- Do not accidentally protect the HarmoSSM change-detection model when the user asks for the CFDepth water-depth invention.

### `technical proposal/`

- Treat this as a project technical-solution document, not a manuscript or patent application.
- Emphasize system architecture, implementation routes, datasets, validation, deployment assumptions, and deliverables.
- Keep claims and performance statements tied to available evidence; do not invent project results.
- Quote the path in shell commands: `'technical proposal'`.

## Reasoning and Communication

Spend time on thinking; you do not need to use the commentary channel to report progress to me.
DO NOT send optional commentary.
