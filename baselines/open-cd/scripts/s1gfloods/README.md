# Open-CD 9 模型批量重训

## 三地区推理输出（2026-09-12 更新）

在 `opencd` 环境从仓库根目录执行：

```bash
for tiles in GF3_Henan_CD_infer GF3_Zhuozhou_CD_infer LT1_Guangxi_CD_infer
do
  CUDA_VISIBLE_DEVICES=0 bash baselines/open-cd/scripts/s1gfloods/run_all_gf3_henan_infer.sh \
    --batch-dir baselines/open-cd/work_dirs/s1gfloods-batch-20260911-234448 \
    --data-root "datasets/$tiles" \
    --output-root baselines/open-cd/outputs \
    --batch-size 1 --device cuda:0 || break
done
```

默认输出为 `outputs/<model_tag>/<Zhengzhou|Zhuozhou|Guangxi>/`，包含 `tile_png/`、`mosaic/` 和批量入口保存的 `infer.log`。`mosaic/` 内保存概率GeoTIFF、二值GeoTIFF/PNG以及 `infer_report.json`。汇总位于 `outputs/_summaries/<训练批次名>/<地区>.tsv`。旧结果不迁移；重复运行覆盖同模型同地区的同名文件。

底层预处理器只允许单切片推理，默认 `--batch-size 1`；旧命令指定更大值时仅作为外层分组大小，逐片前向并校验返回数量，保持切片顺序。单模型入口 `tools/infer_gf3_henan.py` 同样支持 `--output-root`，`--out-dir` 和 `--mosaic-dir` 可分别覆盖默认路径。

少量验证可指定独立 `/tmp` 输出目录并使用 `--limit 5 --skip-mosaic`；此模式只保存5个切片，不能视为完整整景结果。正式推理不指定这两个参数。

切片 PNG 保存模型原生 `pred_sem_seg`：IFN 使用框架单通道默认 `>0.3`，CGNet、ChangeStar、STANet 使用训练配置的 `>0.5`；其他7模型使用双通道 argmax，平局归背景。整景先融合前景概率，再按对应阈值严格 `>` 二值化；双通道的整景判别边界为0.5。重叠融合结果与单张切片不保证逐像素一致。

两个推理入口均已移除 `--threshold`，旧命令携带该参数会报错，请删除该参数。这里恢复的是训练验证规则，11个checkpoint没有保存经验证集搜索的最优阈值。程序从主头配置解析规则并在模型构建后核对，辅助头和ChangeStar内部头不参与最终规则选择。日志的 `[DECISION]`、整景报告的 `decision_rule` 以及批量TSV记录阈值、来源、切片和整景规则；`--skip-mosaic` 仍在日志及汇总记录规则。

推理仅在内存配置中补齐单通道缺省阈值、关闭未使用的可视化后端，并直接使用默认 `pseudo_collate`；不改写训练配置、权重或环境。保留输出通道建议、损失设置提示和第三方接口弃用 warning，没有全局屏蔽警告。

统一入口 `run_all_s1gfloods.sh` 默认依次训练：FC-Siam-Diff、IFN、BIT、LightCDNet、ChangeFormer、CGNet、SNUNet、HANet、TTP ViT-SAM-B。**不包含 Changer、STANet、ChangeStar**。`run_all_s1gfloods_extra.sh` 复用同一实现，默认只运行后五个模型。

默认数据为 `datasets/S1GFloods_CD_DINO_BG_75_25`（末尾无下划线）。现场目录决定成员，不读取旧 CSV 决定训练集。默认 seed=42，单 GPU 串行；保留模型结构、损失、优化器类型和参数衰减，使用下述 SAR 优化方案。默认普通模型 batch=16、TTP=4，训练20000迭代，每2000迭代验证及定期保存，不加载旧训练 checkpoint，不续训。

## Batch 翻倍配置（2026-09-14，待手动验收）

`--batch-scale` 支持1或2，默认2。相对于原普通模型8、TTP 2的配置，batch翻倍，总迭代40000→20000，warmup 1000→500，PolyLR从500衰减至20000，验证及定期保存间隔4000→2000。学习率不自动放大，验证/测试batch不变，短训练保持新batch但仍只训练两次迭代。

`--batch-scale 1` 恢复原batch、40000迭代及原调度；两种设置累计抽样量相同，但优化更新次数不同，不保证精度等价或完整训练一定提速。生效配置、终端输出和 `batch_plan.json` 的 `training_settings` 记录实际batch与调度。

本次只修改代码和相关测试，不自动执行测试、短训练、吞吐量基准或正式训练。旧batch的验收不能代表新配置通过。手动验收需检查9个模型的显存、有限损失、梯度和权重保存，并比较原/新batch各25次训练迭代的吞吐量（前5次预热，后20次CUDA同步计时）。显存不足或吞吐量下降应如实记录，不自动降低batch。验收通过后再更新 `baselines/open-cd/run.md`；该文件目前仍是旧版命令说明。

## 当前环境结论与准备边界

2026-09-11 检查：`hacqi` 的 Torch 2.4.0/cu121、MMCV 2.2.0、MMEngine 0.10.7 可以导入，但缺少 mmseg/mmdet/mmpretrain；MMCV 2.2.0 超出本仓库 `<2.2.0` 上限。4090 D 的基础 CUDA 算子可用，但未验证11模型的训练峰值。数据检查为 train 4729 对、val 1593 对，RGB 256×256、标签0/255，配对一致；这些数量仅为检查时快照。

以下是**供用户依次执行**的准备命令，不会由训练脚本自动执行。独立环境组合由官方版本约束与源码接口推导，尚未在当前机器安装验证；每步成功后再执行下一步。不要在 `hacqi` 中安装或降级这些包。

## 1. 创建专用环境与安装依赖

从仓库根目录开始：

```bash
cd /home/yukun80/codes/paper6_waterlogging
source /home/yukun80/miniconda3/etc/profile.d/conda.sh
conda create -n opencd python=3.10 pip -y
conda activate opencd
```

```bash
python -m pip install 'numpy==1.26.4'
python -m pip install torch==2.1.0 torchvision==0.16.0 \
  --index-url https://download.pytorch.org/whl/cu121
```

```bash
python -m pip install --only-binary=mmcv mmcv==2.1.0 \
  -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
python -m pip install -r baselines/open-cd/scripts/s1gfloods/environment-requirements.txt
python -m pip check
```

固定 Transformers 4.30.2 是为了保留 MMPretrain 旧导入链使用的 `apply_chunking_to_forward` 等接口；不要随意升级。训练入口会将本地 Open-CD 加入 `PYTHONPATH`，该工作副本没有安装包入口，不需要 `pip install -e .`。

依据：[PyTorch 2.1.0 官方安装组合](https://pytorch.org/get-started/previous-versions/#v210)、[MMCV 官方 wheel](https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html)、[MMPretrain 1.2.0 的导入接口](https://github.com/open-mmlab/mmpretrain/blob/v1.2.0/mmpretrain/models/multimodal/blip/language_model.py)。requirements 固定核心及直接依赖，不是全部传递依赖的 lockfile。

## 2. 显式准备本地预训练权重

VGG16 为 IFN 代码的内部 torchvision 初始化依赖；BIT 使用 ResNet18-v1c。下面命令联网下载并验证官方文件名中的 SHA256 前缀。TTP 使用已存在的 `baselines/open-cd/pretrained/vit-base-p16_sam-pre_3rdparty_sa1b-1024px_20230411-2320f9cc.pth`，不下载替代权重。

```bash
python - <<'PY'
from pathlib import Path
import hashlib
import torch
cache = Path('baselines/open-cd/pretrained/torch/hub/checkpoints').resolve()
cache.mkdir(parents=True, exist_ok=True)
urls = [
    'https://download.pytorch.org/models/vgg16-397923af.pth',
    'https://download.openmmlab.com/pretrain/third_party/resnet18_v1c-b5776b93.pth',
]
for url in urls:
    target = cache / url.rsplit('/', 1)[-1]
    prefix = target.stem.rsplit('-', 1)[-1]
    if not target.exists():
        temporary = target.with_suffix('.download')
        torch.hub.download_url_to_file(url, str(temporary), hash_prefix=prefix)
        temporary.rename(target)
    digest = hashlib.sha256()
    with target.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    if not digest.hexdigest().startswith(prefix):
        raise RuntimeError(f'Checksum mismatch: {target}')
    print(target, digest.hexdigest())
PY
```

训练入口统一设置 `TORCH_HOME=baselines/open-cd/pretrained/torch`，并将 BIT/TTP 的初始化路径改为本地文件。预检验证存在性及 SHA256 前缀；IFN 使用同一 Torch 缓存。缺权重时停止，不在队列中下载。

## 3. 检查、预览和短训练验证

只读数据/资产检查与环境探测（含小型 CUDA NMS，不训练模型）：

```bash
bash baselines/open-cd/scripts/s1gfloods/run_all_s1gfloods.sh check-env
```

命令预览仅需 NumPy、Pillow、MMEngine：打印全部生效配置、命令和缺失资产；不创建批次、不构建模型、不验证运行依赖。预览退出成功不代表环境可训练。

```bash
bash baselines/open-cd/scripts/s1gfloods/run_all_s1gfloods.sh full-train --dry-run
```

环境检查成功后，用户显式执行短训练验证：每模型2次训练迭代、前2个验证样本，**保持正式训练 batch size**。输出为 `work_dirs/s1gfloods-smoke-时间戳/`，不会被原推理脚本选为最新正式训练批次。

```bash
bash baselines/open-cd/scripts/s1gfloods/run_all_s1gfloods.sh smoke-train
```

仅全部9模型短训练成功，才可判定该环境和入口已通过基础运行验证；这不证明完整40k训练一定成功。安装验收后可显式保存环境快照：

```bash
python -m pip freeze > baselines/open-cd/pretrained/opencd-environment-freeze.txt
```

## 4. 正式训练及结果

```bash
CUDA_VISIBLE_DEVICES=0 bash baselines/open-cd/scripts/s1gfloods/run_all_s1gfloods.sh full-train
```

可选参数：`--data-root PATH`、`--batch-root PATH`、`--torch-home PATH`、`--seed 42`；路径相对调用目录解析。模型子集使用短名、空格分隔，始终按固定清单顺序执行：

```bash
bash baselines/open-cd/scripts/s1gfloods/run_all_s1gfloods.sh full-train \
  --models bit changeformer ttp --seed 42
```

可用短名：`fc_siam_diff ifn bit lightcdnet changeformer cgnet snunet hanet ttp`。`--gpus`仅接受1；多卡训练不属于此次协议。每个模型固定seed但不强制确定性算子，不承诺跨环境逐位一致。

每次批次运行重新校验全量 train/val，并只从 train/A、train/B PNG联合计算 RGB 总体均值/标准差：float64累积、0–255数值尺度，重复为六通道。val不参与统计；不再对PNG做拉伸。结果保存在新批次，数据集及旧统计不改写。成员指纹基于相对路径、大小和mtime，属于变更检测摘要，不是逐文件内容哈希。训练期间应保持数据不变。

正式输出：`work_dirs/s1gfloods-batch-时间戳/`，有同名目录时使用后缀，新建失败则停止。每个模型目录下有 `<model_tag>.py` 生效配置、best checkpoint及MMEngine日志；批次根目录有：

- `summary.tsv`、`succeeded_models.txt`、`failed_models.txt`：逐模型状态、退出码、best路径、耗时。
- `logs/`：终端与文件同步记录的训练输出。
- `normalization.json`、`data_members.json`：本轮统计、配对列表、成员摘要。
- `environment.json`、`batch_plan.json`：依赖版本、CUDA信息、资产摘要、种子与实际命令。

全局预检失败时不启动任何模型。单模型失败继续后续模型，最后只要有失败就返回非零；进程成功但缺少best checkpoint也视为失败。Ctrl-C/SIGTERM终止当前训练进程组并停止队列，返回130，不自动重试或续训。旧两个历史批次不被修改。

此数据没有独立test，配置中的test仍指向val；本入口只训练与验证，不将其报告成独立测试结果。

## 脚本回归验证

```bash
bash -n baselines/open-cd/scripts/s1gfloods/run_all_s1gfloods.sh \
  baselines/open-cd/scripts/s1gfloods/run_all_s1gfloods_extra.sh
python -m py_compile baselines/open-cd/scripts/s1gfloods/batch_train.py \
  baselines/open-cd/scripts/s1gfloods/test_batch_train.py
python -m unittest discover -s baselines/open-cd/scripts/s1gfloods -p test_batch_train.py -v
```

这些测试使用临时合成PNG和轻量子进程，验证统计、配置与编排行为，不构建/训练实际模型。

## SAR 训练优化（2026-09-13）

- 专用 `SharedSARRadiometric`：A/B 共用亮度偏移（±10）和对比度（0.8–1.2），各以0.5概率启用，先亮度后对比度并裁剪至0–255。不使用色相/饱和度增强，保留同步旋转和翻转。
- 删除256→256无效裁剪。训练标签按前景比例0、(0,10%]、(10%,100%]分组；非空组概率为原比例的一半加均衡比例的一半，组内均匀有放回。验证集不重采样，批次 `sampling.json` 保存实际分组及概率。
- FC-Siam-Diff、IFN、BIT、SNUNet、HANet学习率为5e-4，LightCDNet为1e-3；ChangeFormer为6e-5（head×10）、CGNet为5e-4、TTP为4e-4，后三项保持原值。
- 同时保存 `best_FloodIoU` 与 `best_mIoU`。默认 `--save-best FloodIoU` 决定主权重，可显式选择mIoU。FloodIoU使用change类未舍入IoU百分数；前景并集为零返回NaN，不生成伪造最优分数。若主指标始终不可用、没有best权重，该模型不能报告成功。
- 新批次推理按summary.tsv中的主权重解析，旧批次保持历史行为。主权重记录缺失或失效时明确报错，不偷偷回退。
- 以上配置是待验证实验方案，不代表已经证明性能提升。基础两迭代验证不代表40k训练或跨地区精度验收。

从仓库根目录依次执行（前一步成功后再执行下一步）：

```bash
conda activate opencd
export CUDA_VISIBLE_DEVICES=0
bash baselines/open-cd/scripts/s1gfloods/run_all_s1gfloods.sh check-env
bash baselines/open-cd/scripts/s1gfloods/run_all_s1gfloods.sh smoke-train
bash baselines/open-cd/scripts/s1gfloods/run_all_s1gfloods.sh full-train
```

`full-train --dry-run` 只打印生效配置和命令；`--models bit ifn` 可选择子集。训练STANet、ChangeStar参数会被拒绝，但历史模型推理仍可使用。
