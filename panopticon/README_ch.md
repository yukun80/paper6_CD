# Panopticon：面向地球观测的任意传感器基础模型

本仓库提供了论文《Panopticon: Advancing Any-Sensor Foundation Models for Earth Observation》的官方代码实现。

Panopticon 的代码架构深度构建于 [DINOv2 官方代码库](https://github.com/facebookresearch/dinov2) 之上。我们的核心创新主要体现在 `dinov2/models/panopticon.py` 中的**切片嵌入 (Patch Embedding)** 机制，以及 `dinov2/data/augmentations.py` 中的**数据增强 (Augmentations)** 策略。


*Panopticon 架构概览。橙色框标记了其相对于标准 DINOv2 的创新与改进之处。*

---

## 最新动态

* 我们发布了重构后的评估代码库 [Geobreeze](https://github.com/geobreeze/geobreeze)！
* Panopticon 已作为官方模型集成至 [TorchGeo 0.7](https://torchgeo.readthedocs.io/en/stable/api/models.html#panopticon) 中，这使得在该模型上调用现有的遥感数据集变得极其简便。

---

## 如何使用 Panopticon

只需安装 `torch`，即可参考以下示例代码调用 Panopticon：

```python
import torch

# 加载模型
model = torch.hub.load('Panopticon-FM/panopticon', 'panopticon_vitb14')

# 生成示例输入数据
x_dict = dict(
  imgs = torch.randn(2, 3, 224, 224),  # 形状为 (B, C, H, W)
  chn_ids = torch.tensor([[664, 559, 493]]).repeat(2,1)  # 形状为 (B, C)，此处为 RGB 波长（单位：纳米）
)

# 获取图像级特征（适用于分类、回归等任务）
normed_cls_token = model(x_dict)
assert tuple(normed_cls_token.shape) == (2, 768)

# 获取切片级特征（适用于语义分割任务）
blk_indices = [3, 5, 7, 11]
blocks = model.get_intermediate_layers(x_dict, n=blk_indices, return_class_token=True)
assert len(blocks) == 4
cls_tokens = [blk[1] for blk in blocks]
patch_tokens = [blk[0] for blk in blocks]
assert tuple(cls_tokens[0].shape) == (2, 768)
assert tuple(patch_tokens[0].shape) == (2, (224/14)**2, 768)

```

**使用建议：**
为了获得最佳性能，请遵循 Panopticon 的预训练设置：使用 224x224 分辨率的图像，并进行标准正态归一化。代码会自动下载教师网络权重。此外，您可以从 [Hugging Face 仓库](https://huggingface.co/lewaldm/panopticon) 获取完整权重以及论文中提到的基于 RGB 预训练的 DINO Head 权重。

**SAR 传感器支持：**
当输入合成孔径雷达（SAR）通道时，请从 -1 到 -12 的 12 个类别中选择对应的 `chn_id`。详细映射关系请参考 [Sentinel-1 元数据说明](https://github.com/geobreeze/geobreeze/blob/main/geobreeze/datasets/metadata/sensors/sentinel1.yaml)（对应 `gaussian.mu` 参数）。

---

## 环境安装

请使用 Conda 配置实验环境：

```bash
conda create -n panopticon python=3.9 --yes
conda activate panopticon
pip install -r requirements.txt

```

在代码根目录下创建一个 `.env` 文件，并配置以下环境变量：

```bash
GEO_BENCH_DIR=/geobench/数据集/路径
RDIR=/资源/文件夹/路径
CDIR=/.../panopticon/dinov2/configs
ODIR=/输出/文件夹/路径

```

数据集将存储在 `$RDIR/datasets/` 目录下。

---

## 训练数据集

### FMoW

我们使用了 `fmow` 与 `fmow-sentinel` 的组合数据集。

1. 从 [官方仓库](https://github.com/fMoW/dataset) 下载 `fmow` 数据集。您只需要 `fmow-full` 版本，无需 `fmow-rgb`。解压至您的 `dataset` 文件夹。注意：该数据集体量庞大（约 3.5TB），解压需要较长时间。
2. 从 [此链接](https://purl.stanford.edu/vg497cb6002) 下载 `fmow-sentinel` 数据集（约 80GB）。
3. 在 `fmow/` 目录下创建 `metadata_v2` 文件夹，并从 [此 Google Drive](https://drive.google.com/drive/folders/1nsTN-5v6jHusYm_NnKrVLN-TYo7RmUz8?usp=drive_link) 下载元数据文件。

### MMEarth

1. 从 [MMEarth 官方仓库](https://github.com/vishalned/MMEarth-data) 下载数据集。建议至少下载 MMEarth 完整版（约 600GB）。
2. 下载 [元数据 parquet 文件](https://drive.google.com/drive/folders/1LfTBRxJNzgDFIrW1yD9gPbV2Rl-ZV3-d?usp=drive_link) 并将其放入 `mmearth/data_1M_v001` 文件夹中。

### SatlasPretrain

请遵循 [Satlas-Pretrain 官方指南](https://github.com/allenai/satlas/blob/main/SatlasPretrain.md) 下载以下数据集：

* Sentinel-2
* Sentinel-1
* Landsat 8/9
注：我们未采用 NAIP 数据集进行训练，因为它仅覆盖美国本土的 RGB-NIR 波段，在光谱和地理多样性上稍显不足。但其地面采样距离 (GSD) 达 1 米，您可根据自身需求考虑使用。

### SpectralEarth

请在 [DLR 官网](https://geoservice.dlr.de/web/datasets/enmap_spectralearth) 注册并申请下载该数据集。

### 元数据文件 (Metadata Files)

为了满足本研究中“传感器对 (Paired-sensor)”输入的要求，您需要按地理足迹（Footprint）索引的元数据文件。请从 [Hugging Face](https://huggingface.co/lewaldm/panopticon) 下载，并将各自的 `metadata_v2` 文件夹放置在 `fmow`、`fmow-sentinel` 和 `satlas` 对应的目录下（`mmearth` 和 `spectral_earth` 已内置此类索引）。

---

## 评估数据集

本仓库目前主要用于模型训练，严谨的基准测试对比功能将在近期通过独立仓库发布。目前您可以使用以下数据集进行初步评估：

### GeoBench

请遵循 [GeoBench 官方下载指南](https://github.com/ServiceNow/geo-bench)。

---

## 训练流程

加载环境变量后，在根目录下执行以下命令：

**单 GPU 训练：**

```bash
PYTHONPATH=. python dinov2/train/train.py \
  --config-file=dinov2/configs/quick_check.yaml \
  --output_dir=.

```

**多 GPU 训练：**

```bash
PYTHONPATH=. torchrun --nproc_per_node=2 dinov2/train/train.py \
  --config-file=dinov2/configs/quick_check.yaml \
  --output_dir=.

```

关于默认参数的详细说明，请参阅 `dinov2/configs/pretrain_default.yaml`。

---

## 评估流程

本节主要针对 Panopticon 本身的评估（对应论文中的**消融实验 (Ablation Study)**）。与其他模型的横向对比代码请移步 [Geobreeze](https://github.com/geobreeze/geobreeze)。

您可以直接在预训练配置文件中指定评估配置。若需进行独立评估，可运行 `dinov2/eval/eval.py/main`。

**评估示例：**

```bash
PYTHONPATH=. python dinov2/eval/eval.py main \
  --model-obj=$ODIR/my_run_dir \
  --config-obj=dinov2/configs/eval/5min;dinov2/configs/eval/1h/geobench/so2sat.yaml \

```

---

## 针对 DINOv2 的改进说明

为了便于大家进行精确对比，本仓库的首次提交记录即为原始的 DINOv2 代码，第二次提交则引入了 Panopticon 的改进。主要变化如下：

* **新增模块**：在 `dinov2/models/panopticon.py` 中增加了全新的切片嵌入机制，并在 `dinov2/data/augmentations.py` 中引入了新的增强算法。
* **数据适配**：增加了对多种遥感数据集的支持，并将核心数据对象修改为包含图像与通道信息的字典（Dictionary）。
* **评估引擎**：在保持核心逻辑不变的前提下，增加了更易配置的封装器 (Wrappers)，并实现了基于 `torch.distributed` 环境的 GPU 任务分配。
* **灵活性**：增强了权重加载（Checkpoint Loading）的灵活性。

---

  1. 任意通道输入的 Patch Embedding + 通道交叉注意力
     panopticon.py:10
  2. 光谱增强（通道子采样、通道补齐、单传感器/多传感器视图策略）
     augmentations.py:57
  3. 两阶段训练配置与超参（Stage2 为多数据集+PanopticonAugmentation）
     stage2.yaml:70
  4. 数据侧“footprint/sensor as augmentation”输入组织（FMoW/Satlas/MMEarth/SpectralEarth）
     fmow.py:314
     satlas.py:132
     mmearth.py:387
     spectral_earth.py:280

• - /home/yukun/codes/paper6_waterlogging/panopticon/dinov2/models/panopticon.py
  - /home/yukun/codes/paper6_waterlogging/panopticon/dinov2/data/augmentations.py
  - /home/yukun/codes/paper6_waterlogging/panopticon/dinov2/configs/stage2.yaml
  - /home/yukun/codes/paper6_waterlogging/panopticon/dinov2/configs/stage1.yaml
  - /home/yukun/codes/paper6_waterlogging/panopticon/dinov2/models/vision_transformer.py
  - /home/yukun/codes/paper6_waterlogging/panopticon/dinov2/data/loaders.py
  - /home/yukun/codes/paper6_waterlogging/panopticon/dinov2/data/collate.py
  - /home/yukun/codes/paper6_waterlogging/panopticon/dinov2/data/datasets/fmow.py
  - /home/yukun/codes/paper6_waterlogging/panopticon/dinov2/data/datasets/satlas.py
  - /home/yukun/codes/paper6_waterlogging/panopticon/dinov2/data/datasets/mmearth.py
  - /home/yukun/codes/paper6_waterlogging/panopticon/dinov2/data/datasets/spectral_earth.py
