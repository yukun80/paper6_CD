import argparse
import json
from pathlib import Path
from typing import List

import torch

from model.modules.dino_meta import DINO_ARCH_CHOICES, resolve_dino_arch, resolve_extract_ids


OPTICAL_MEAN = [0.430, 0.411, 0.296]
OPTICAL_STD = [0.213, 0.156, 0.143]
PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent
DEFAULT_DATA_ROOT = "../datasets"
DEFAULT_DATASET = "S1GFloods_CD_DINO"
DEFAULT_STATS_FILE = "../datasets/S1GFloods_CD_DINO/channel_stats_s1gfloods_train.json"
DEFAULT_DINO_WEIGHT = "dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
DEFAULT_BACKBONE_WEIGHT = "pretrained/efficientnet_b0_ra-3dd342df.pth"


def _resolve_repo_relative_path(path_str: str) -> str:
    """将仓库内相对路径解析到 HA-CQI 目录，避免受 cwd 影响。"""
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return str(path)

    candidate = (PROJECT_DIR / path).resolve()
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
    mean = stats.get("mean")
    std = stats.get("std")
    if mean is None or std is None:
        raise ValueError(f"Stats file missing mean/std: {stats_path}")
    mean = _parse_float_list([str(v) for v in mean], "mean")
    std = _parse_float_list([str(v) for v in std], "std")
    return mean, std


def resolve_norm_stats(opt) -> tuple[List[float], List[float]]:
    """根据数据集和显式参数解析输入归一化参数。"""
    mean = _parse_float_list(opt.mean, "mean")
    std = _parse_float_list(opt.std, "std")
    if mean is not None or std is not None:
        if mean is None or std is None:
            raise ValueError("mean/std must be provided together")
        return mean, std

    if opt.stats_file:
        return _load_stats_from_json(Path(opt.stats_file))

    dataset_name = str(opt.dataset)
    if dataset_name.startswith("S1GFloods"):
        candidate_dirs = [Path(opt.dataroot) / dataset_name, Path(opt.dataroot) / "S1GFloods_CD"]
        for dataset_dir in candidate_dirs:
            default_stats = dataset_dir / "channel_stats_s1gfloods_train.json"
            if default_stats.exists():
                return _load_stats_from_json(default_stats)
        return [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

    return OPTICAL_MEAN.copy(), OPTICAL_STD.copy()


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def init(self):
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
            "--save_test", action="store_true"
        )
        self.parser.add_argument(
            "--result_dir", type=str, default="./results", help="results are saved here"
        )
        self.parser.add_argument(
            "--vis_path", type=str, default="vis", help="results are saved here"
        )
        self.parser.add_argument("--load_pretrain", action='store_true')

        self.parser.add_argument("--phase", type=str, default="train")
        self.parser.add_argument(
            "--backbone",
            type=str,
            default="efficientnet_b0",
            help="CNN backbone，支持 efficientnet_b0、mobilenetv2。",
        )
        self.parser.add_argument(
            "--backbone_weight",
            type=str,
            default=DEFAULT_BACKBONE_WEIGHT,
            help="CNN backbone 本地 PyTorch 预训练权重路径；不接受 TPU/TensorFlow 原始 checkpoint。",
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
        self.parser.add_argument("--fpn", type=str, default="fpn")
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
            "--mask_dim",
            type=int,
            default=128,
            help="轻量 Mask2Former-style head 的 mask feature 维度。",
        )
        self.parser.add_argument(
            "--mask_queries",
            type=int,
            default=32,
            help="Mask2Former-style head 的 learnable mask query 数量。",
        )
        self.parser.add_argument(
            "--mask_decoder_layers",
            type=int,
            default=3,
            help="Mask2Former-style transformer decoder 层数。",
        )
        self.parser.add_argument(
            "--mask_heads",
            type=int,
            default=4,
            help="Mask2Former-style transformer decoder 注意力头数。",
        )
        self.parser.add_argument('--n_layers', nargs='+', type=int, default=[1, 1, 1, 1])
        self.parser.add_argument(
            '--extract_ids',
            nargs='+',
            type=int,
            default=None,
            help="从 DINO 主干抽取的层号；默认按 --dino_arch 自动选择。",
        )
        self.parser.add_argument("--alpha", type=float, default=0.25)
        self.parser.add_argument("--gamma", type=int, default=4, help="gamma for Focal loss")

        self.parser.add_argument("--batch_size", type=int, default=16)
        self.parser.add_argument("--num_epochs", type=int, default=100)
        self.parser.add_argument("--input_size", type=int, default=256, help="训练/推理默认输入尺寸")
        self.parser.add_argument("--num_workers", type=int, default=4, help="#threads for loading data")
        self.parser.add_argument("--lr", type=float, default=5e-4)
        self.parser.add_argument("--weight_decay", type=float, default=5e-4)
        self.parser.add_argument(
            "--amp",
            action="store_true",
            help="启用 CUDA 自动混合精度训练与验证，降低显存占用。",
        )
        self.parser.add_argument(
            "--amp_dtype",
            type=str,
            default="fp16",
            choices=["fp16", "bf16"],
            help="AMP autocast 精度类型；默认 fp16，数值不稳定时可切换 bf16。",
        )
        self.parser.add_argument(
            "--split_seed", type=int, default=42, help="S1GFloods 划分脚本与实验配置的默认随机种子"
        )
        self.parser.add_argument(
            "--stats_file",
            type=str,
            default=DEFAULT_STATS_FILE,
            help="JSON 统计文件路径；若为空则按数据集使用默认值或自动发现。",
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
            default=0.5,
            help="验证/测试时前景概率阈值，替代硬编码 argmax 以提升 tiny flood 召回调节能力。",
        )
        self.parser.add_argument(
            "--best_metric",
            type=str,
            default="tiny_safe_combo",
            choices=["iou_1", "tiny_recall", "tiny_combo", "tiny_safe_combo"],
            help="标准 best 权重保存依据；tiny-heavy 场景建议使用 tiny_safe_combo。",
        )
    def parse(self):
        self.init()
        self.opt = self.parser.parse_args()

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

        self.opt.dino_weight = _resolve_repo_relative_path(self.opt.dino_weight)
        if self.opt.backbone_weight:
            _validate_backbone_weight_path(self.opt.backbone_weight)
            self.opt.backbone_weight = _resolve_repo_relative_path(self.opt.backbone_weight)
        self.opt.dino_arch = resolve_dino_arch(self.opt.dino_arch, self.opt.dino_weight)
        self.opt.extract_ids = resolve_extract_ids(self.opt.dino_arch, self.opt.extract_ids)
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
        if self.opt.mask_dim < 1:
            raise ValueError("--mask_dim must be a positive integer")
        if self.opt.mask_queries < 1:
            raise ValueError("--mask_queries must be a positive integer")
        if self.opt.mask_decoder_layers < 1:
            raise ValueError("--mask_decoder_layers must be a positive integer")
        if self.opt.mask_heads < 1:
            raise ValueError("--mask_heads must be a positive integer")
        if not 0.0 <= self.opt.eval_fg_threshold <= 1.0:
            raise ValueError("--eval_fg_threshold must be within [0, 1]")
        if self.opt.small_area_thresh < self.opt.tiny_area_thresh:
            raise ValueError("--small_area_thresh must be >= --tiny_area_thresh")
        self.opt.mean, self.opt.std = resolve_norm_stats(self.opt)

        args = vars(self.opt)

        print("------------ Options -------------")
        for k, v in sorted(args.items()):
            print("%s: %s" % (str(k), str(v)))
        print("-------------- End ----------------")

        return self.opt
