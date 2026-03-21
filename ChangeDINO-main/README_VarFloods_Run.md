# VarFloods 运行说明

本文档说明如何使用当前仓库中的 `ChangeDINO-main` 在 `VarFloods` 数据集上完成数据准备、训练和测试。

## 1. 前置条件

- 工作目录根路径：
  - `/home/yukun/codes/paper6_waterlogging`
- 已安装 `ChangeDINO-main/requirements.txt` 中依赖
- 额外安装 `rasterio`
- 已准备好 DINOv3 权重文件：
  - `ChangeDINO-main/dinov3/weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth`

如果环境中还没有 `rasterio`，可先安装：

```bash
pip install rasterio
```

## 2. 构造 VarFloods_CD 数据集

从仓库根目录运行：

```bash
cd /home/yukun/codes/paper6_waterlogging

python ChangeDINO-main/scripts/prepare_varfloods_cd.py \
  --src-root datasets/VarFloods \
  --out-root datasets/VarFloods_CD \
  --tile-size 256 \
  --stride 256 \
  --seed 42 \
  --overwrite \
  --strict
```

说明：

- 仅使用 `datasets/VarFloods/<Region>/PRO`
- 切片大小为 `256x256`
- 划分比例为 `train/val/test = 8:1:1`
- 输出标签为二值 `tif`
  - 洪水：`1`
  - 背景：`0`
  - 标签元数据不设置 `nodata`

输出目录：

```text
datasets/VarFloods_CD/
├── train/
│   ├── A/
│   ├── B/
│   └── label/
├── val/
│   ├── A/
│   ├── B/
│   └── label/
├── test/
│   ├── A/
│   ├── B/
│   └── label/
├── split_manifest.csv
├── split_report.json
└── channel_stats_varfloods_train.json
```

## 3. 计算训练集归一化统计

从仓库根目录运行：

```bash
cd /home/yukun/codes/paper6_waterlogging

python ChangeDINO-main/scripts/compute_varfloods_cd_stats.py \
  --data-root datasets/VarFloods_CD \
  --split train \
  --output datasets/VarFloods_CD/channel_stats_varfloods_train.json
```

该脚本会统计 `train/A` 和 `train/B` 中单波段 `tif` 的归一化参数，并写入：

- `datasets/VarFloods_CD/channel_stats_varfloods_train.json`

## 4. 开始训练

进入 `ChangeDINO-main` 后运行：

```bash
cd /home/yukun/codes/paper6_waterlogging/ChangeDINO-main
bash trainval_varfloods.sh
```

对应的完整训练命令为：

```bash
cd /home/yukun/codes/paper6_waterlogging/ChangeDINO-main

python trainval.py \
  --name VarFloods-ChangeDINO \
  --dataset VarFloods_CD \
  --dataroot ../datasets \
  --dataset_mode sar \
  --stats_file ../datasets/VarFloods_CD/channel_stats_varfloods_train.json \
  --gpu_ids 0 \
  --batch_size 8 \
  --input_size 256 \
  --num_epochs 100 \
  --lr 1e-4
```

训练输出默认保存在：

- `ChangeDINO-main/checkpoints/VarFloods-ChangeDINO/`

## 5. 测试与保存预测结果

训练完成后，在 `ChangeDINO-main` 下运行：

```bash
python test.py \
  --name VarFloods-ChangeDINO \
  --dataset VarFloods_CD \
  --dataroot ../datasets \
  --dataset_mode sar \
  --stats_file ../datasets/VarFloods_CD/channel_stats_varfloods_train.json \
  --gpu_ids 0 \
  --save_test
```

说明：

- 默认加载 `checkpoints/VarFloods-ChangeDINO/` 下最佳 checkpoint
- 测试集为 `datasets/VarFloods_CD/test`
- 若加上 `--save_test`，预测结果会保存在：
  - `ChangeDINO-main/checkpoints/VarFloods-ChangeDINO/pred/`

## 6. 可选：先做 1 epoch 冒烟验证

如果想先快速确认训练流程可用，可运行：

```bash
cd /home/yukun/codes/paper6_waterlogging/ChangeDINO-main

python trainval.py \
  --name VarFloods-ChangeDINO-smoke \
  --dataset VarFloods_CD \
  --dataroot ../datasets \
  --dataset_mode sar \
  --stats_file ../datasets/VarFloods_CD/channel_stats_varfloods_train.json \
  --gpu_ids 0 \
  --batch_size 2 \
  --num_workers 0 \
  --input_size 256 \
  --num_epochs 1 \
  --lr 1e-4
```

## 7. 当前实现要点

- `VarFloods` 图像输入保持为单波段 `float32 tif`
- `ChangeDINO` 训练时直接读取 `tif`
- 读取后内部执行 SAR 稳健拉伸，并扩展为 3 通道输入张量
- `S1GFloods` 的原有 `png` 训练链路保持兼容
