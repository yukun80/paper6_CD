# 数据准备

[返回目录](README.md) · [项目首页](../../README.md)

## 数据目录

```text
datasets/<DATASET_NAME>/
├── train/{A,B,label}
└── val/{A,B,label}
```

默认数据集为 `datasets/S1GFloods_CD_DINO_BG_75_25_`，训练集和验证集分别位于
`train/{A,B,label}` 和 `val/{A,B,label}`。A、B 和 label 须完整同名配对，实际训练成员以运行时扫描结果为准。
`STATS_MODE=auto` 自动维护归一化统计，每次运行的 `data_snapshot.json` 记录实际样本与统计信息。
广西数据包含参与训练的样本，其整景结果应按相应数据使用条件解释。

## 成员与归一化统计

`STATS_MODE=auto` 会扫描当前 train/val 的 A/B/label 同名集合。完整删除三元组后，下次进程启动
自动减样本；只缺任一文件会失败。train A/B 的文件名、大小或 mtime 变化时，mean/std 缓存自动
失效。历史 manifest、split report 与 fingerprint 不决定运行成员。

若需要复现指定统计文件：

```bash
cd HarmoSSM
STATS_MODE=file \
STATS_FILE=../datasets/S1GFloods_CD_DINO_BG_75_25_/channel_stats_s1gfloods_train.json \
bash trainval_s1gfloods.sh
```

文件模式只验证 train split、三个有限 mean/std 与正数 std。

## 场景切片与标签

`scripts/prepare_fused_sar_cd_dataset.py` 对新制备的 VarFloods 切片采用灾前和灾后整景联合
P2/P98 拉伸，同一场景的切片复用拉伸参数。S1GFloods 影像 PNG 原样复制。

独立推理场景使用 `prepare_tiles.py`，默认切片大小为 256×256，步长为 128，最小共同有效比例为 0.01，
采用整景联合 P2/P98 拉伸。广西示例：

```bash
python HarmoSSM/scripts/prepare_tiles.py \
  --pre-image datasets/LT1_Guangxi/LT_Guangxi_pre.tif \
  --post-image datasets/LT1_Guangxi/LT_Guangxi_post.tif \
  --output-dir datasets/LT1_Guangxi_CD_infer \
  --scene-id LT1_Guangxi
```

输出包括 `test/A`、`test/B`、`test/valid_mask`、`tile_manifest.csv` 和 `prepare_report.json`。
`--dry-run` 仅统计。`--overwrite` 重建影像及有效区切片并更新清单和报告，保留已有标签。
源影像或切片参数改变后，应重新生成标签。输出目录须与源影像目录分开。

已有影像切片时，使用同网格标签生成标签切片：

```bash
python HarmoSSM/scripts/prepare_sar_scene_label_infer.py \
  --src-root datasets/LT1_Guangxi \
  --label-image LT_Guangxi_Label.tif \
  --tiles-root datasets/LT1_Guangxi_CD_infer \
  --strict
```

标签写入 `test/label`、`test/label_tif` 并补充清单。已有非空标签目录时需显式传入
`--overwrite`，该参数仅替换标签目录。

相关内容：[训练](training.md) · [预测与评价](inference.md)
