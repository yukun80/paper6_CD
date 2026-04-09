# GF3 河南整景 SAR 洪水变化检测精度对比评估

## 实验条件

| 项目 | 说明 |
|------|------|
| 训练数据集 | S1GFloods (S1GFloods_CD_DINO) |
| 推理数据集 | GF3_Henan_CD_infer (郑州 GF-3 SAR 整景切片, 256×256, stride=128) |
| 切片总数 | 5901 |
| 二值化阈值 | 0.5 |
| 源影像 | Pre_Zhengzhou_ascending_s1_radmatch.tif + Post_Zhengzhou_descending_clip.tif |

## 精度对比表

> **说明**：下表中精度值为模拟占位数据，待替换为真实评估结果。
> 指标含义：OA = Overall Accuracy, mIoU = mean IoU, mF1 = mean F1,
> IoU\_c / F1\_c / P\_c / R\_c = 变化类(change)的 IoU / F1 / Precision / Recall。

| 方法 | Backbone | OA | mIoU | mF1 | IoU\_c | F1\_c | P\_c | R\_c | Kappa |
|------|----------|----|------|-----|--------|-------|------|------|-------|
| FC-Siam-Diff | FC-Siam | 90.12 | 78.35 | 86.42 | 68.21 | 81.10 | 83.47 | 78.85 | 72.53 |
| IFN | IFN | 90.87 | 79.64 | 87.15 | 70.38 | 82.61 | 85.12 | 80.24 | 74.18 |
| BIT | ResNet-18 | 91.53 | 80.72 | 87.96 | 72.15 | 83.87 | 84.93 | 82.83 | 75.62 |
| Changer-Ex | IA-ResNet-18 | 92.08 | 81.95 | 88.73 | 74.42 | 85.36 | 87.21 | 83.58 | 77.14 |
| ChangeStar-FarSeg | ResNet-18 + FarSegFPN | 92.64 | 82.87 | 89.41 | 76.18 | 86.52 | 88.35 | 84.76 | 78.53 |
| LightCDNet-S | LightCDNet-Small | 93.15 | 83.76 | 90.08 | 77.83 | 87.49 | 86.72 | 88.28 | 79.87 |
| **ChangeDINO (Ours)** | **DINOv3-ViT-S/16 + MobileNetV2** | **94.28** | **85.91** | **91.67** | **82.46** | **90.37** | **89.15** | **91.62** | **83.25** |
| **ChangeDINO (Ours)** | **DINOv3-ViT-S/16 + EfficientNet-B0** | **95.37** | **88.24** | **93.15** | **86.73** | **92.88** | **91.56** | **94.23** | **86.41** |

## 模型与权重对应关系

| 方法 | 框架 | 配置/权重标识 |
|------|------|---------------|
| FC-Siam-Diff | Open-CD | `fc_siam_diff_256x256_40k_s1gfloods` |
| IFN | Open-CD | `ifn_256x256_40k_s1gfloods` |
| BIT | Open-CD | `bit_r18_256x256_40k_s1gfloods` |
| Changer-Ex | Open-CD | `changer_ex_r18_256x256_40k_s1gfloods` |
| ChangeStar-FarSeg | Open-CD | `changestar_farseg_1x96_256x256_40k_s1gfloods` |
| LightCDNet-S | Open-CD | `lightcdnet_s_256x256_40k_s1gfloods` |
| ChangeDINO (MobileNetV2) | ChangeDINO | `S1GFloods-ChangeDINO-vits16` / `*_mobilenetv2_best.pth` |
| ChangeDINO (EfficientNet-B0) | ChangeDINO | `S1GFloods-ChangeDINO-vits16-20260327` / `*_efficientnet_b0_best.pth` |

## 推理输出路径

| 方法 | 切片预测 | 整景拼接 |
|------|----------|----------|
| Open-CD 各模型 | `baselines/open-cd/work_dirs/s1gfloods-batch-20260326-004236/<model_tag>/infer_gf3_henan_png/` | `baselines/open-cd/work_dirs/s1gfloods-batch-20260326-004236/<model_tag>/infer_gf3_henan_full/` |
| ChangeDINO (MobileNetV2) | `ChangeDINO-main/outputs/gf3_henan_vits16/tile_png/` | `ChangeDINO-main/outputs/gf3_henan_vits16/mosaic/` |
| ChangeDINO (EfficientNet-B0) | `ChangeDINO-main/outputs/gf3_henan_vits16_20260327/tile_png/` | `ChangeDINO-main/outputs/gf3_henan_vits16_20260327/mosaic/` |
