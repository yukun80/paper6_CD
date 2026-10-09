# 运行验证

[返回目录](README.md) · [项目首页](../../README.md)

## 修改后基础检查

修改核心代码后，在 `HarmoSSM` 目录执行：

```bash
cd HarmoSSM
python -m py_compile \
  option.py \
  model/architectures/harmossm.py \
  model/backbones/builder.py \
  model/modules/*.py \
  model/decode_heads/*.py \
  model/engine.py \
  model/checkpointing.py \
  data/transform.py \
  utils/flood_evaluation.py \
  trainval.py \
  test.py \
  run.py \
  scripts/diagnose_cross_domain_features.py \
  scripts/infer_sar_scene_tiles.py

bash -n trainval_s1gfloods.sh trainval.sh
```

原说明中的 `tests/test_pipeline_contracts.py` 已不存在，不再列为可运行检查。
现有整景评价测试可从项目根目录运行：

```bash
python -m unittest discover -s HarmoSSM/scripts -p test_evaluate_scene_comparison.py
```

## 小样本训练验证

下列步骤会实际训练并写入结果，仅在需要验证算法改动时执行；本次文档整理未执行。

```bash
cd HarmoSSM
conda activate hacqi
python trainval.py \
  --name smoke-harmossm-b2-dino5 \
  --dataset S1GFloods_CD_DINO_BG_75_25_ \
  --dataroot ../datasets \
  --stats_mode auto \
  --dino_fusion_layers 5 8 11 \
  --gpu_ids 0 \
  --batch_size 1 \
  --num_workers 0 \
  --num_epochs 1 \
  --max_train_steps 2 \
  --max_val_steps 2 \
  --seed 1 \
  --amp \
  --amp_dtype bf16
```

正式训练不得保留 `--max_train_steps/--max_val_steps`。

## 结果边界

基础检查与短步训练不能证明跨场景精度或性能提升。切片推理改动还应检查
`infer_report.json`、整景二值输出、有效掩码和 NoData。当前整景预测内部计算概率，
但不保存概率栅格（报告中 `probability_saved=false`）。
本次文档核对情况见[整理清单](../migration.md)。

相关内容：[环境准备](environment.md) · [训练](training.md)

更名兼容测试（从项目根目录开始）：

```bash
cd HarmoSSM
python -m unittest discover -s tests -p test_checkpoint_rename.py
```
