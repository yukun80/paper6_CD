# 预测与评价

[返回目录](README.md) · [项目首页](../../README.md)

## 测试与单对影像

测试结果通过 `--save_test` 保存至运行目录的 `pred/`：

```bash
cd HarmoSSM
python test.py \
  --checkpoint checkpoints/<resolved_run_name>/<run>_efficientnet_b2_best_primary.pth \
  --gpu_ids 0 \
  --save_test
```

对一组灾前和灾后影像执行推理：

```bash
cd HarmoSSM
python run.py \
  --checkpoint checkpoints/<resolved_run_name>/<run>_efficientnet_b2_best_primary.pth \
  --img_A /path/to/pre_image.tif \
  --img_B /path/to/post_image.tif \
  --output outputs/run_pred.png \
  --gpu_ids 0
```

## 整景预测

瓦片目录须包含 `tile_manifest.csv`，且清单引用的 A/B 影像与 `valid_mask` 均可读取。
河南 GF3 场景示例：

```bash
python HarmoSSM/scripts/infer_sar_scene_tiles.py \
  --tiles-root datasets/GF3_Henan_CD_infer \
  --checkpoint HarmoSSM/checkpoints/<b2_run>/<b2_run>_efficientnet_b2_best_primary.pth \
  --gpu_ids 0 \
  --batch_size 8 \
  --output-dir HarmoSSM/outputs/gf3_henan_corrected
```

`<b2_run>` 替换为实际运行名。其他场景使用同一入口，替换输入和输出目录即可。
归一化统计默认从检查点元数据读取。推理阈值来自显式 `--threshold` 或
`meta.selection.threshold`，两者均缺失时程序报错。

默认保存切片与整景结果，整景文件位于 `<output-dir>/mosaic/`：

```text
change_binary_raw.tif/png  # 阈值化后的原始二值结果
change_binary.tif/png      # 一致性伪斑过滤后的二值结果
```

`<output-dir>/infer_report.json` 记录检查点、阈值、输入输出路径和模型配置。
当前不保存整景概率栅格，报告中 `probability_saved=false`。
定量评估使用二值 TIFF 的有效掩码，PNG 中 NoData 与前景均显示白色。
`--disable_blob_filter` 关闭伪斑过滤，`--skip-tiles` 跳过切片级 PNG/TIF 保存。

## 整景评价

```bash
python HarmoSSM/scripts/evaluate_sar_scene.py \
  --prediction-dir HarmoSSM/outputs/gf3_zhuozhou \
  --ground-truth datasets/GF3_Zhuozhou/GF3_Zhuozhou_label.tif
```

当前推理报告对应二值图评价，排除预测和标签 NoData（包括标签未知值 `3`），
输出 TP/FP/TN/FN、IoU、F1、P/R、准确率及有效和忽略像素数量。
若需评价原始二值图，可显式指定：

```bash
python HarmoSSM/scripts/evaluate_sar_scene.py \
  --binary HarmoSSM/outputs/<run>/mosaic/change_binary_raw.tif \
  --ground-truth datasets/GF3_Zhuozhou/GF3_Zhuozhou_label.tif
```

相关内容：[数据准备](data.md) · [训练记录](checkpoints.md)
