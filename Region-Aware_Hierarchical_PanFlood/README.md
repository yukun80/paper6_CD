# Region-Aware Hierarchical PanFlood

面向 UrbanSARFloods 的区域感知、层次化、双专家洪涝分割工程（独立封装，不改动 `panopticon/` 与 `SAM4D-main/`）。

## 1. 数据与格式实测结论

通过真实读取确认（非假设）：

- `datasets/urban_sar_floods_256`
  - `SAR/*_SAR.tif`: `float32`, `(8,256,256)`
  - `GT/*_GT.tif`: 常见标签 `{0,1,2}`，但局部 split 可能只含子集（如 `{0,1}`）
- `datasets/urban_sar_floods_ch12_256`
  - `SAR/*_SAR.tif`: `float32`, `(12,256,256)`
  - `GT/*_GT.tif`: 主标签三类
  - `GT_flood_type/*`: `{0,1,255(ignore)}`
- split 文件条目是相对路径风格，工程内部采用“按 GT 文件名索引”解析，避免路径歧义。

## 2. 模型结构

1. `PanopticonBackboneWrapper`
- 实际 import `panopticon/dinov2/models/vision_transformer.py`
- 支持 `8ch / 8ch_plus_engineered / 12ch`
- 集成 channel-role id：
  - measurement type (`coherence/intensity/delta`)
  - temporal role (`pre/co/post/pre_minus_co/post_minus_pre`)
  - polarization (`VV/VH`)
  - source role (`raw/engineered`) + `source_role_embedding`

2. 分层头
- `Floodness Head`: flood/non-flood 粗分割
- `Router Head`: open-like / urban-like / ambiguous 软路由
- `Open Expert`: 大感受野细化
- `Urban Expert`: 高分辨细节细化
- `Final Fusion`: router soft gate 融合为三类

3. SAM4D 思想迁移（非机械照搬）
- `StateMemoryAttention`：将多状态 SAR 证据组织为 memory banks 并做轻量检索
- `AutoPromptRefiner`：自动伪提示细化（无人工点击）
  - open 分支偏边界收缩与误检抑制
  - urban 分支偏小斑块恢复与弱响应增强

## 3. 损失与采样

- 分层损失：
  - `L_floodness = BCE + Dice`
  - `L_router = CE`
  - `L_open = BCE + Dice`
  - `L_urban = BCE(pos_weight) + Dice`
  - `L_final = Focal/CE + Dice`
  - `L_boundary` + `L_consistency`
- 采样：
  - Urban oversampling（按样本 urban/open 像素比加权）
  - Hard patch sampling（`sampler.hard_score_file`）
  - `tools/build_hard_scores.py` 可生成可直接使用的 hard score JSON

## 4. CUDA 运行策略（默认强制）

- `config/base.yaml` 默认 `runtime.require_cuda: true`
- `train/eval/infer/visualize` 均支持 `--device {auto,cuda,cpu}`
- 当 `require_cuda=true` 且 CUDA 不可用时，脚本会 fail-fast 退出
- 若需临时 CPU 调试，请在配置中将 `runtime.require_cuda=false`（入口会自动禁用 xformers）
- 训练会自动记录：
  - `device_report.json`
  - `label_mapping.json`
  - `module_param_summary.json`

先诊断环境：

```bash
conda run -n panopticon python Region-Aware_Hierarchical_PanFlood/tools/check_cuda_runtime.py --device cuda
```

## 5. 关键脚本

1. 数据统计
```bash
python Region-Aware_Hierarchical_PanFlood/tools/inspect_dataset.py \
  --config-file Region-Aware_Hierarchical_PanFlood/config/base.yaml \
  --split train
```

2. 生成 hard scores
```bash
python Region-Aware_Hierarchical_PanFlood/tools/build_hard_scores.py \
  --config-file Region-Aware_Hierarchical_PanFlood/config/stage2_router_dual_expert.yaml \
  --split train \
  --output Region-Aware_Hierarchical_PanFlood/outputs/hard_scores_train.json
```

3. 训练
```bash
python Region-Aware_Hierarchical_PanFlood/tools/train.py \
  --config-file Region-Aware_Hierarchical_PanFlood/config/stage4_full_prompt_refine.yaml \
  --device cuda
```

4. 验证
```bash
python Region-Aware_Hierarchical_PanFlood/tools/eval.py \
  --config-file Region-Aware_Hierarchical_PanFlood/config/stage4_full_prompt_refine.yaml \
  --ckpt Region-Aware_Hierarchical_PanFlood/outputs/<run_name>/best.pth \
  --split val \
  --device cuda
```

5. 推理
```bash
python Region-Aware_Hierarchical_PanFlood/tools/infer.py \
  --config-file Region-Aware_Hierarchical_PanFlood/config/stage4_full_prompt_refine.yaml \
  --ckpt Region-Aware_Hierarchical_PanFlood/outputs/<run_name>/best.pth \
  --sar-dir datasets/urban_sar_floods_256/03_FU/SAR \
  --out-dir Region-Aware_Hierarchical_PanFlood/outputs/infer_vis \
  --device cuda
```

6. 可视化
```bash
python Region-Aware_Hierarchical_PanFlood/tools/visualize_predictions.py \
  --config-file Region-Aware_Hierarchical_PanFlood/config/stage4_full_prompt_refine.yaml \
  --ckpt Region-Aware_Hierarchical_PanFlood/outputs/<run_name>/best.pth \
  --split val \
  --out-dir Region-Aware_Hierarchical_PanFlood/outputs/vis \
  --device cuda
```

## 6. 阶段进展

- 阶段1：完成（最小可运行）
- 阶段2：完成（router + 双专家）
- 阶段3：完成（state memory）
- 阶段4：完成（auto prompt refinement）

## 7. Panopticon / SAM4D 复用与替代说明

1. Panopticon 可直接复用部分
- `PanopticonPE + ViT backbone`、权重加载逻辑（teacher/model 前缀兼容）

2. SAM4D 不直接复用部分
- camera/LiDAR 双流结构与权重无法直接迁移到 SAR/InSAR 多状态输入
- 替代为：
  - `StateMemoryAttention`（状态证据检索）
  - `AutoPromptRefiner`（自动伪提示细化）

## 8. 当前不足与未完成项

1. 当前 `panopticon` conda 环境中，`torch.cuda.is_available()` 可能返回 `False`（即使 `nvidia-smi` 可见 GPU）；代码已强制 CUDA fail-fast，但环境本身仍需修复后才能完成“CUDA 1 epoch 实跑”。
2. hard sample 目前是离线分数驱动，尚未实现在线 OHEM/动态难例池。
3. `StateMemoryAttention` 仍是轻量版，可继续扩展到更强的时序记忆写入/读取机制。
