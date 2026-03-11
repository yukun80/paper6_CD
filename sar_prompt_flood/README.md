# SAR Prompt Flood

面向“有标注参考集训练 + 无标注目标集零样本迁移”的 SAR 洪水变化分割实验分支。

## 数据角色

参考训练集：

- `datasets/urban_sar_floods_test/urban_sar_floods_test_tiles_512_band5_band7`
- 目录结构：`pre/`, `post/`, `GT/`

目标推理集：

- `datasets/GF3_Henan/GF3_Henan_tiles_512`
- 目录结构：`pre/`, `post/`

## 当前流程

1. 在有标注参考集上训练参考引导 proposal
2. 在有标注参考集上训练 prompt policy
3. 在无标注 GF3 目标集上做单参考检索、prompt 优化和 tile 级预测
4. 推理阶段不微调 SAM

## 配置

默认配置：

- `sar_prompt_flood/config/urban_sar_reference.json`

其中：

- `reference_data.*` 控制参考训练集
- `target_data.*` 控制 GF3 目标集

## 训练与推理

先下载 SAM 权重：

```bash
python datasets/script/download_sam_vit_b.py
```

训练参考引导 proposal：

```bash
python -m sar_prompt_flood.train_reference_prompt \
  --config-file sar_prompt_flood/config/urban_sar_reference.json
```

训练 prompt policy：

```bash
python -m sar_prompt_flood.train_prompt_policy \
  --config-file sar_prompt_flood/config/urban_sar_reference.json
```

在 GF3 目标集上推理：

```bash
python -m sar_prompt_flood.run_reference_prompt_pipeline \
  --config-file sar_prompt_flood/config/urban_sar_reference.json
```

## 输出

默认输出目录：

- `sar_prompt_flood/work_dir/reference_supervised`
- `sar_prompt_flood/outputs/reference_supervised_gf3`

主要产物：

- `best_reference_prompt.pth`
- `best_prompt_policy.pth`
- `summary.json`
- `tile_metrics.json`
- `predictions/*.png`
- `predictions/*_mask.tif`
- `visuals/*.png`

## 说明

- 首次训练若参考集 `splits/` 不存在，会自动生成 `train.txt / val.txt / ref_bank.txt`
- GF3 目标集无标签，`summary.json` 默认输出无监督统计，不输出 IoU/Dice
- 当前版本默认每个 GF3 tile 只检索一个参考样本，不做多参考融合和全图回拼
