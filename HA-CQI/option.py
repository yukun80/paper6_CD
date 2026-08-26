import argparse
import json
import math
from pathlib import Path
from typing import List

import torch

from data.runtime_snapshot import resolve_auto_channel_stats
from model.backbones import DEFAULT_BACKBONE_NAME, DEFAULT_BACKBONE_WEIGHT
from model.checkpointing import checkpoint_model_config, load_checkpoint_payload
from model.modules.dino_meta import (
    DINO_ARCH_CHOICES,
    resolve_dino_arch,
    resolve_dino_fusion_layers,
)


OPTICAL_MEAN = [0.430, 0.411, 0.296]
OPTICAL_STD = [0.213, 0.156, 0.143]
PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent
DEFAULT_DATA_ROOT = "../datasets"
DEFAULT_DATASET = "S1GFloods_CD_DINO_BG_75_25_"
DEFAULT_STATS_FILE = "../datasets/S1GFloods_CD_DINO_BG_75_25_/channel_stats_s1gfloods_train.json"
DEFAULT_DINO_WEIGHT = "dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"


def _resolve_repo_relative_path(path_str: str) -> str:
    """将仓库内相对路径解析到 HA-CQI 目录，避免受 cwd 影响。"""
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return str(path)

    for base_dir in (PROJECT_DIR, REPO_ROOT):
        candidate = (base_dir / path).resolve()
        if candidate.exists():
            return str(candidate)
    return path_str


def _resolve_existing_project_path(path_str: str) -> str:
    """优先按 HA-CQI 目录解析输入路径，再兼容仓库根目录相对路径。"""
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return str(path)

    for base_dir in (PROJECT_DIR, REPO_ROOT):
        candidate = (base_dir / path).resolve()
        if candidate.exists():
            return str(candidate)
    return str((PROJECT_DIR / path).resolve())


def _resolve_project_output_path(path_str: str) -> str:
    """输出目录相对 HA-CQI 解析，保证默认 checkpoint 写入 HA-CQI/checkpoints。"""
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return str(path)
    return str((PROJECT_DIR / path).resolve())


def _parse_float_list(values: List[str] | None, field_name: str) -> List[float] | None:
    if values is None:
        return None
    if len(values) != 3:
        raise ValueError(f"{field_name} expects exactly 3 floats, got {len(values)}")
    return [float(v) for v in values]


def _validate_backbone_weight_path(path_str: str) -> None:
    suffixes = {s.lower() for s in Path(path_str).suffixes}
    if not suffixes:
        raise ValueError(
            "--backbone_weight must point to a local PyTorch weight file (.pth/.pt), got path without suffix."
        )
    if suffixes & {".tar", ".gz", ".zip", ".ckpt", ".index"}:
        raise ValueError(
            "--backbone_weight must be a local PyTorch .pth/.pt file. TensorFlow/TPU checkpoints must be converted first."
        )
    if not (".pth" in suffixes or ".pt" in suffixes):
        raise ValueError(
            f"--backbone_weight must be a local PyTorch .pth/.pt file, got: {path_str}"
        )


def _load_stats_from_json(stats_path: Path) -> tuple[List[float], List[float]]:
    payload = json.loads(stats_path.read_text(encoding="utf-8"))
    stats = payload.get("recommended_config_fields", payload)
    return _load_stats_values(stats, stats_path)


def resolve_norm_stats(opt) -> tuple[List[float], List[float]]:
    """根据数据集和显式参数解析输入归一化参数。"""
    mean = _parse_float_list(opt.mean, "mean")
    std = _parse_float_list(opt.std, "std")
    if mean is not None or std is not None:
        if mean is None or std is None:
            raise ValueError("mean/std must be provided together")
        if any(not math.isfinite(value) for value in mean + std):
            raise ValueError("mean/std must contain finite values")
        if any(value <= 0.0 for value in std):
            raise ValueError("std must contain positive values")
        opt.stats_source = "explicit"
        opt.runtime_data_snapshot = None
        return mean, std

    stats_mode = str(getattr(opt, "stats_mode", "file")).lower()
    if stats_mode == "auto":
        dataset_root = Path(opt.dataroot) / str(opt.dataset)
        cache_dir = Path(opt.checkpoint_dir) / ".runtime_stats_cache"
        stats, snapshot, cache_path, cache_hit = resolve_auto_channel_stats(
            dataset_root,
            cache_dir,
        )
        opt.runtime_data_snapshot = snapshot
        opt.stats_file = str(cache_path)
        opt.stats_source = "auto_cache" if cache_hit else "auto_computed"
        print(
            f"[INFO] runtime data snapshot={snapshot['runtime_snapshot_id'][:12]} "
            f"| train={snapshot['splits']['train']['samples']} "
            f"| val={snapshot['splits']['val']['samples']} "
            f"| stats={opt.stats_source}"
        )
        recommended = stats.get("recommended_config_fields", stats)
        return _load_stats_values(recommended, cache_path)

    if opt.stats_file:
        opt.stats_source = "file"
        opt.runtime_data_snapshot = None
        return _load_stats_from_json(Path(opt.stats_file))

    dataset_name = str(opt.dataset)
    if dataset_name.startswith("S1GFloods"):
        candidate_dirs = [Path(opt.dataroot) / dataset_name, Path(opt.dataroot) / "S1GFloods_CD"]
        for dataset_dir in candidate_dirs:
            default_stats = dataset_dir / "channel_stats_s1gfloods_train.json"
            if default_stats.exists():
                opt.stats_source = "discovered_file"
                opt.runtime_data_snapshot = None
                return _load_stats_from_json(default_stats)
        opt.stats_source = "sar_fallback"
        opt.runtime_data_snapshot = None
        return [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

    opt.stats_source = "optical_default"
    opt.runtime_data_snapshot = None
    return OPTICAL_MEAN.copy(), OPTICAL_STD.copy()


def _load_stats_values(
    stats: dict,
    source: Path,
) -> tuple[List[float], List[float]]:
    """校验已解析的统计字段，供文件与自动缓存共享数值契约。"""
    mean = stats.get("mean")
    std = stats.get("std")
    if mean is None or std is None:
        raise ValueError(f"Stats file missing mean/std: {source}")
    mean_values = _parse_float_list([str(v) for v in mean], "mean")
    std_values = _parse_float_list([str(v) for v in std], "std")
    if any(not math.isfinite(value) for value in mean_values + std_values):
        raise ValueError(f"Stats mean/std must be finite: {source}")
    if any(value <= 0.0 for value in std_values):
        raise ValueError(f"Stats std must be positive: {source}")
    return mean_values, std_values


def validate_stats_provenance(opt) -> None:
    """统计量必须来自训练 split；不绑定可变的数据集成员清单。"""
    if (
        str(getattr(opt, "stats_mode", "file")) != "file"
        or str(getattr(opt, "stats_source", "file")) != "file"
        or not opt.stats_file
    ):
        return
    stats_path = Path(opt.stats_file)
    if not stats_path.is_file():
        raise FileNotFoundError(f"Stats file not found: {stats_path}")
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    stats_split = stats.get("split")
    if stats_split != "train":
        raise ValueError(f"HA-CQI normalization stats must use train split, got: {stats_split}")


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def init(self):
        # B2-only 与 DINOv3-LVD 输入契约仍写入 Namespace/checkpoint，但不再暴露可变 CLI。
        self.parser.set_defaults(
            backbone=DEFAULT_BACKBONE_NAME,
            dino_input_norm="imagenet",
        )
        self.parser.add_argument(
            "--gpu_ids", type=str, default="0", help="gpu ids: e.g. 0. use -1 for CPU"
        )
        self.parser.add_argument("--name", type=str, default="S1GFloods-HA-CQI")
        self.parser.add_argument(
            "--dataroot", type=str, default=DEFAULT_DATA_ROOT
        )
        self.parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
        self.parser.add_argument(
            "--dataset_mode",
            type=str,
            default="auto",
            choices=["auto", "default", "sar"],
            help="数据增强与归一化策略。auto 会按数据集名自动选择。",
        )
        self.parser.add_argument(
            "--checkpoint_dir",
            type=str,
            default="./checkpoints",
            help="models are saved here",
        )
        self.parser.add_argument(
            "--checkpoint",
            type=str,
            default="",
            help="测试/推理使用的显式 checkpoint 路径；训练保存目录仍由 checkpoint_dir 控制。",
        )
        self.parser.add_argument(
            "--threshold",
            type=float,
            default=None,
            help="测试/推理显式阈值；为空时从 checkpoint metadata 自动解析。",
        )

        self.parser.add_argument(
            "--save_test", action="store_true"
        )
        self.parser.add_argument(
            "--vis_path", type=str, default="vis", help="results are saved here"
        )

        self.parser.add_argument("--phase", type=str, default="train")
        self.parser.add_argument(
            "--backbone_weight",
            type=str,
            default=DEFAULT_BACKBONE_WEIGHT,
            help="EfficientNet-B2 本地 PyTorch 预训练权重；不接受其他架构或 TPU/TensorFlow checkpoint。",
        )
        self.parser.add_argument(
            "--dino_arch",
            type=str,
            default="auto",
            choices=DINO_ARCH_CHOICES,
            help="DINOv3 架构名；auto 会按权重文件名推断。",
        )
        self.parser.add_argument(
            "--dino_weight",
            type=str,
            default=DEFAULT_DINO_WEIGHT,
            help="DINOv3 预训练权重路径（相对当前工作目录或绝对路径）。",
        )
        self.parser.add_argument("--fpn_channels", type=int, default=128)
        self.parser.add_argument("--deform_groups", type=int, default=4)
        self.parser.add_argument("--gamma_mode", type=str, default="SE")
        self.parser.add_argument("--beta_mode", type=str, default="contextgatedconv")
        self.parser.add_argument(
            "--disable_soft_alignment",
            action="store_true",
            help="关闭 deformable soft-alignment；关闭后 P1/P2/P3 将改用轻量双时相协同适配器。",
        )
        self.parser.add_argument(
            "--align_window",
            type=int,
            default=5,
            help="Deformable cross-attention 的局部对齐窗口大小，需为奇数。",
        )
        self.parser.add_argument(
            "--align_points",
            type=int,
            default=9,
            help="每个 query 在局部邻域内采样的 deformable points 数量。",
        )
        self.parser.add_argument(
            "--align_heads",
            type=int,
            default=4,
            help="deformable cross-attention 的 head 数。",
        )
        self.parser.add_argument(
            "--align_on_levels",
            nargs="+",
            type=int,
            default=[1, 2, 3],
            help="在哪些金字塔层上启用 deformable soft-alignment，默认在 p1/p2/p3。",
        )
        self.parser.add_argument(
            "--align_qkv_bias",
            action="store_true",
            help="是否为 deformable cross-attention 的 q/k/v 投影启用 bias。",
        )
        self.parser.add_argument(
            "--align_offset_groups",
            type=int,
            default=4,
            help="偏移预测卷积的 group 数，用于控制对齐模块开销。",
        )
        self.parser.add_argument(
            "--num_change_queries",
            type=int,
            default=16,
            help="CQI 每个尺度的 learnable change query 数量。",
        )
        self.parser.add_argument(
            "--cqi_heads",
            type=int,
            default=4,
            help="CQI two-way attention 的注意力头数。",
        )
        self.parser.add_argument(
            "--dino_fusion_layers",
            nargs='+',
            type=int,
            default=None,
            help="真正进入 P3/P4/P5 融合的三个 DINO 层；默认按架构选择早/中/深层。",
        )
        self.parser.add_argument(
            "--focal_class_weights",
            nargs=2,
            type=float,
            default=[0.25, 0.75],
            metavar=("BACKGROUND", "FOREGROUND"),
            help="Focal loss 的背景/前景权重，解析后归一化为和 1。",
        )
        self.parser.add_argument("--gamma", type=float, default=2.0, help="gamma for Focal loss")
        self.parser.add_argument("--batch_size", type=int, default=12)
        self.parser.add_argument("--num_epochs", type=int, default=80)
        self.parser.add_argument("--input_size", type=int, default=256, help="训练/推理默认输入尺寸")
        self.parser.add_argument("--num_workers", type=int, default=8, help="#threads for loading data")
        self.parser.add_argument(
            "--radiometric_jitter_mode",
            type=str,
            default="shared",
            choices=["shared", "independent"],
            help="SAR 亮度/对比增强模式；Exp-3 对随机单一时相独立扰动。",
        )
        self.parser.add_argument("--lr", type=float, default=1e-4)
        self.parser.add_argument("--weight_decay", type=float, default=5e-4)
        self.parser.add_argument(
            "--head_lr_mult",
            type=float,
            default=2.0,
            help="随机初始化的 HA/CQI/head 模块相对基础学习率的倍率。",
        )
        self.parser.add_argument(
            "--aux_loss_weight",
            type=float,
            default=1.0,
            help="辅助监督总权重；内部会对各尺度 aux 权重归一化。",
        )
        self.parser.add_argument(
            "--aux_loss_weight_end",
            type=float,
            default=0.5,
            help="辅助监督在中后期退火后的目标总权重。",
        )
        self.parser.add_argument(
            "--aux_decay_start_epoch",
            type=int,
            default=5,
            help="从该 epoch 开始线性衰减辅助监督权重。",
        )
        self.parser.add_argument(
            "--tversky_beta_start",
            type=float,
            default=0.70,
            help="Tversky FN 权重起始值，用于训练早期保护召回。",
        )
        self.parser.add_argument(
            "--tversky_beta_end",
            type=float,
            default=0.55,
            help="Tversky FN 权重退火终值，用于中后期平衡 precision/IoU。",
        )
        self.parser.add_argument(
            "--loss_anneal_epochs",
            type=int,
            default=20,
            help="Tversky 与 aux 权重退火的线性周期。",
        )
        self.parser.add_argument(
            "--support_consistency_weight",
            type=float,
            default=0.03,
            help="P1/P2 高分辨率支持区域的 final mask 保留损失最大权重。",
        )
        self.parser.add_argument(
            "--coarse_consistency_weight",
            type=float,
            default=0.02,
            help="缺乏 P1/P2 局部支持时的粗尺度前景抑制损失最大权重。",
        )
        self.parser.add_argument(
            "--consistency_warmup_epochs",
            type=int,
            default=5,
            help="前若干 epoch 不启用 HA-CQI 自一致性约束。",
        )
        self.parser.add_argument(
            "--consistency_ramp_epochs",
            type=int,
            default=10,
            help="自一致性约束从 0 线性升到目标权重的周期。",
        )
        self.parser.add_argument(
            "--amp",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="启用 CUDA 自动混合精度；B2 baseline 默认启用，可用 --no-amp 关闭。",
        )
        self.parser.add_argument(
            "--amp_dtype",
            type=str,
            default="bf16",
            choices=["fp16", "bf16"],
            help="AMP autocast 精度类型；B2 baseline 默认 bf16。",
        )
        self.parser.add_argument(
            "--grad_scaler_init_scale",
            type=float,
            default=4096.0,
            help="fp16 GradScaler 初始 scale；较低默认值避免复杂 HA-CQI 首步连续溢出。",
        )
        self.parser.add_argument("--seed", type=int, default=1, help="模型、训练与 DataLoader 随机种子")
        self.parser.add_argument(
            "--deterministic",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="启用可复现的 cuDNN/DataLoader 配置。",
        )
        self.parser.add_argument(
            "--max_train_steps",
            type=int,
            default=-1,
            help="每个 epoch 最多训练 step；-1 表示完整 epoch，仅用于冒烟诊断。",
        )
        self.parser.add_argument(
            "--max_val_steps",
            type=int,
            default=-1,
            help="每个 epoch 最多验证 step；-1 表示完整验证，仅用于冒烟诊断。",
        )
        self.parser.add_argument(
            "--resume",
            type=str,
            default="",
            help="checkpoint v2 路径；恢复完整训练状态。",
        )
        self.parser.add_argument(
            "--init_checkpoint",
            type=str,
            default="",
            help="仅加载显式 checkpoint 的网络参数。",
        )
        self.parser.add_argument(
            "--stats_mode",
            type=str,
            default="auto",
            choices=["auto", "file"],
            help="auto 按当前 train 目录成员自动缓存 mean/std；file 显式使用 --stats_file。",
        )
        self.parser.add_argument(
            "--stats_file",
            type=str,
            default=DEFAULT_STATS_FILE,
            help="stats_mode=file 时使用的 train split JSON 统计文件。",
        )
        self.parser.add_argument(
            "--mean",
            nargs="+",
            default=None,
            help="3 通道归一化均值，优先级高于 --stats_file。",
        )
        self.parser.add_argument(
            "--std",
            nargs="+",
            default=None,
            help="3 通道归一化方差，优先级高于 --stats_file。",
        )
        self.parser.add_argument(
            "--tiny_area_thresh",
            type=int,
            default=100,
            help="验证时 tiny flood 连通域面积阈值。",
        )
        self.parser.add_argument(
            "--small_area_thresh",
            type=int,
            default=400,
            help="验证时 small flood 连通域面积阈值。",
        )
        self.parser.add_argument(
            "--eval_fg_threshold",
            type=float,
            default=0.40,
            help="仅用于 validation 细粒度诊断，不参与 checkpoint 选择或推理阈值回退。",
        )
        self.parser.add_argument(
            "--threshold_min",
            type=float,
            default=0.05,
            help="联合选择的最小验证阈值。",
        )
        self.parser.add_argument(
            "--threshold_max",
            type=float,
            default=0.95,
            help="联合选择的最大验证阈值。",
        )
        self.parser.add_argument(
            "--threshold_step",
            type=float,
            default=0.01,
            help="联合选择的验证阈值步长。",
        )
    def parse(self):
        self.init()
        return self.prepare(self.parser.parse_args())

    def prepare(self, opt):
        """校验并解析已由 HA-CQI 参数解析器产生的 Namespace。"""
        self.opt = opt

        if self.opt.dataset_mode == "auto":
            if str(self.opt.dataset).startswith("S1GFloods"):
                self.opt.dataset_mode = "sar"
            else:
                self.opt.dataset_mode = "default"

        str_ids = self.opt.gpu_ids.split(",")
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0 and torch.cuda.is_available():
            torch.cuda.set_device(self.opt.gpu_ids[0])

        self.opt.dataroot = _resolve_existing_project_path(self.opt.dataroot)
        if self.opt.stats_file:
            self.opt.stats_file = _resolve_existing_project_path(self.opt.stats_file)
        self.opt.checkpoint_dir = _resolve_project_output_path(self.opt.checkpoint_dir)
        if self.opt.checkpoint:
            self.opt.checkpoint = _resolve_existing_project_path(self.opt.checkpoint)
        if self.opt.resume:
            self.opt.resume = _resolve_existing_project_path(self.opt.resume)
        if self.opt.init_checkpoint:
            self.opt.init_checkpoint = _resolve_existing_project_path(self.opt.init_checkpoint)

        self.opt.dino_weight = _resolve_repo_relative_path(self.opt.dino_weight)
        if self.opt.backbone_weight:
            _validate_backbone_weight_path(self.opt.backbone_weight)
            self.opt.backbone_weight = _resolve_repo_relative_path(self.opt.backbone_weight)
        self.opt.dino_arch = resolve_dino_arch(self.opt.dino_arch, self.opt.dino_weight)
        self.opt.dino_fusion_layers = resolve_dino_fusion_layers(
            self.opt.dino_arch,
            self.opt.dino_fusion_layers,
        )
        if self.opt.disable_soft_alignment:
            self.opt.align_on_levels = []
        else:
            self.opt.align_on_levels = sorted({int(v) for v in self.opt.align_on_levels})
            invalid_align_levels = [v for v in self.opt.align_on_levels if v not in {1, 2, 3}]
            if invalid_align_levels:
                raise ValueError(
                    f"--align_on_levels only supports P1/P2/P3, got {invalid_align_levels}"
                )
        if self.opt.num_change_queries < 1:
            raise ValueError("--num_change_queries must be a positive integer")
        if self.opt.cqi_heads < 1:
            raise ValueError("--cqi_heads must be a positive integer")
        if self.opt.fpn_channels != 128:
            raise ValueError("HA-CQI OSCD v1 requires --fpn_channels 128")
        if self.opt.head_lr_mult <= 0.0:
            raise ValueError("--head_lr_mult must be positive")
        focal_weights = [float(value) for value in self.opt.focal_class_weights]
        if any(value < 0.0 for value in focal_weights) or sum(focal_weights) <= 0.0:
            raise ValueError("--focal_class_weights must be non-negative with a positive sum")
        focal_weight_sum = sum(focal_weights)
        self.opt.focal_class_weights = [value / focal_weight_sum for value in focal_weights]
        if self.opt.aux_loss_weight < 0.0:
            raise ValueError("--aux_loss_weight must be non-negative")
        if self.opt.aux_loss_weight_end < 0.0:
            raise ValueError("--aux_loss_weight_end must be non-negative")
        if self.opt.aux_decay_start_epoch < 1:
            raise ValueError("--aux_decay_start_epoch must be >= 1")
        if not 0.0 <= self.opt.tversky_beta_start <= 1.0:
            raise ValueError("--tversky_beta_start must be within [0, 1]")
        if not 0.0 <= self.opt.tversky_beta_end <= 1.0:
            raise ValueError("--tversky_beta_end must be within [0, 1]")
        if self.opt.loss_anneal_epochs < 1:
            raise ValueError("--loss_anneal_epochs must be >= 1")
        if self.opt.support_consistency_weight < 0.0:
            raise ValueError("--support_consistency_weight must be non-negative")
        if self.opt.coarse_consistency_weight < 0.0:
            raise ValueError("--coarse_consistency_weight must be non-negative")
        if self.opt.consistency_warmup_epochs < 0:
            raise ValueError("--consistency_warmup_epochs must be >= 0")
        if self.opt.consistency_ramp_epochs < 1:
            raise ValueError("--consistency_ramp_epochs must be >= 1")
        if self.opt.grad_scaler_init_scale <= 0.0:
            raise ValueError("--grad_scaler_init_scale must be positive")
        if not 0.0 <= self.opt.eval_fg_threshold <= 1.0:
            raise ValueError("--eval_fg_threshold must be within [0, 1]")
        if not 0.0 <= self.opt.threshold_min <= self.opt.threshold_max <= 1.0:
            raise ValueError("threshold range must satisfy 0 <= min <= max <= 1")
        if self.opt.threshold_step <= 0.0:
            raise ValueError("--threshold_step must be positive")
        if self.opt.max_train_steps == 0 or self.opt.max_train_steps < -1:
            raise ValueError("--max_train_steps must be -1 or positive")
        if self.opt.max_val_steps == 0 or self.opt.max_val_steps < -1:
            raise ValueError("--max_val_steps must be -1 or positive")
        if self.opt.resume and self.opt.init_checkpoint:
            raise ValueError("--resume and --init_checkpoint are mutually exclusive")
        if self.opt.resume and not Path(self.opt.resume).is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {self.opt.resume}")
        if self.opt.init_checkpoint and not Path(self.opt.init_checkpoint).is_file():
            raise FileNotFoundError(f"Initialization checkpoint not found: {self.opt.init_checkpoint}")
        if self.opt.checkpoint and not Path(self.opt.checkpoint).is_file():
            raise FileNotFoundError(f"Checkpoint not found: {self.opt.checkpoint}")
        if self.opt.threshold is not None and not 0.0 <= self.opt.threshold <= 1.0:
            raise ValueError("--threshold must be within [0, 1]")
        if self.opt.small_area_thresh < self.opt.tiny_area_thresh:
            raise ValueError("--small_area_thresh must be >= --tiny_area_thresh")
        if self.opt.batch_size < 1:
            raise ValueError("--batch_size must be positive")
        if self.opt.num_workers < 0:
            raise ValueError("--num_workers must be non-negative")
        if self.opt.lr <= 0.0:
            raise ValueError("--lr must be positive")

        # 在模型构建前给旧 backbone/input/decoder checkpoint 明确的契约错误。
        for checkpoint_path in (self.opt.resume, self.opt.init_checkpoint):
            if checkpoint_path:
                checkpoint_model_config(
                    load_checkpoint_payload(checkpoint_path, map_location="cpu")
                )
        self.opt.mean, self.opt.std = resolve_norm_stats(self.opt)
        validate_stats_provenance(self.opt)

        args = vars(self.opt)

        print("------------ Options -------------")
        for k, v in sorted(args.items()):
            if k == "runtime_data_snapshot" and isinstance(v, dict):
                splits = v.get("splits", {})
                v = (
                    f"id={str(v.get('runtime_snapshot_id', ''))[:12]} "
                    f"train={splits.get('train', {}).get('samples')} "
                    f"val={splits.get('val', {}).get('samples')} "
                    "(full file list saved to data_snapshot.json)"
                )
            print("%s: %s" % (str(k), str(v)))
        print("-------------- End ----------------")

        return self.opt
