# 环境与预训练权重

[返回目录](README.md) · [项目首页](../../README.md)

## 环境现状

当前目录没有 `requirements-oscd.txt`，不再提供依赖该文件的安装命令。
已有 `hacqi` 环境可先进行导入检查；这不是从零重建环境的完整安装说明：

```bash
conda activate hacqi
python -c "import torch, timm, mmcv, selective_scan_cuda; print(torch.__version__)"
python -m pip check
```

原说明记录的环境组合为 Python 3.11、Torch 2.4、CUDA 12、CXX11 ABI false，
以及 mamba_ssm 2.2.4、einops 0.8.1、ninja 1.13.0、transformers 4.44.2。
这些是旧环境记录，不是当前提供的安装清单；重新搭建环境时需核实实际依赖和可用安装包。
GPU 训练需要 selective-scan CUDA kernel；帮助命令成功不等于 GPU 训练验证通过。

## 权重

主线使用 `efficientnet_b2`。默认权重位于：

```text
HarmoSSM/dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth
HarmoSSM/pretrained/efficientnet_b2_ra-bcdf34b7.pth
```

2026-10-09 整理时已确认两份文件在本机存在，未重新验证其内容或下载来源。
训练脚本允许用 `DINO_WEIGHT`、`BACKBONE_WEIGHT` 指定实际位置；脚本内相对路径以 `HarmoSSM/` 为基准。

原权重说明提供的 [DINOv3 下载入口](https://drive.google.com/file/d/1r6g0D6zV-1e8gJHij1edsE_uzvZ72L3u/view?usp=drive_link)
在此保留用于追溯；本次未核实外部链接可用性及其文件与本地权重是否一致。

相关内容：[数据准备](data.md) · [训练](training.md) · [运行验证](validation.md)
