# 开发日志

## 2026-03-07

### 已完成

1. 新建独立工程 `Region-Aware_Hierarchical_PanFlood`。
2. 自动扫描并确认 UrbanSARFloods 真实数据组织（8ch/12ch、split、标签编码）。
3. 复用 Panopticon 编码器并实现 `PanopticonBackboneWrapper`。
4. 实现层次化模型：Floodness + Router + Open/Urban Experts + Final Fusion。
5. 实现轻量 `StateMemoryAttention`（借鉴 SAM4D memory 思想）。
6. 实现 `AutoPromptRefiner`（自动伪提示细化）。
7. 实现分层损失、采样策略、训练/评估/推理/可视化脚本。
8. 完成阶段化配置（stage1~stage4）。

### 关键设计原因

- Urban 类别极长尾，采用 urban 过采样 + urban loss 增权。
- 将 open/urban 拆为双专家，避免三分类单头在机制差异上的折中。
- 将 memory 设计成“状态维度检索”而非重型时序 transformer，便于调参与调试。
- prompt refinement 使用自动伪提示，避免依赖人工点击交互。

### 尚未完成

- 在线 hard patch 挖掘（目前是静态接口）
- 更强边界解码器和不确定性约束

### 下一步

1. 跑通 stage1 完整训练，确认稳定收敛。
2. 做 stage2/3/4 消融，检查 memory 与 prompt 的增益。
3. 针对 urban 类别再调 sampler 与 focal 参数。

## 2026-03-07（补充：CUDA 强制与缺口收敛）

### 已补齐

1. 训练/评估/推理/可视化统一接入 `runtime.require_cuda`，默认强制 CUDA（`base.yaml`）。
2. 新增 `tools/check_cuda_runtime.py`，可直接诊断 torch-CUDA 可用性与 AMP/matmul。
3. 新增 `tools/build_hard_scores.py`，可生成 `sampler.hard_score_file` 所需 JSON。
4. `builder` 修复 hard score 路径解析，并支持 `scores`/`by_gt_name` 两种格式。
5. 数据集支持 `auto_label_mapping` 后，采样统计和类别直方图改为与训练映射一致。
6. `train.py` 新增实验产物：
   - `device_report.json`
   - `label_mapping.json`
   - `module_param_summary.json`
7. Prompt refinement 细化为 open/urban 双分支行为，不再单路共享。
8. 入口脚本改为“先确定 device，再按需禁用 xformers，再导入 builder”，避免 CPU fallback 时 xformers 算子报错。

### 当前环境诊断结果（conda: panopticon）

- `torch==2.0.0+cu117`，`xformers==0.0.18`
- `nvidia-smi` 可见 GPU，但 `torch.cuda.is_available()` 报错 `Error 304`
- 在 `require_cuda=true` 下，`train/eval` 会按预期 fail-fast（防止误用 CPU 训练）

### 仍待完成

1. 修复当前环境 CUDA runtime 后，执行“至少 1 epoch 的 CUDA 前向+反向”验收。
2. 在线 hard mining（动态更新难例分数）尚未实现。
