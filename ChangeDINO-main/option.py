import argparse
import json
from pathlib import Path
from typing import List

import torch

from model.blocks.dinov3_meta import DINO_ARCH_CHOICES, resolve_dino_arch, resolve_extract_ids


OPTICAL_MEAN = [0.430, 0.411, 0.296]
OPTICAL_STD = [0.213, 0.156, 0.143]
DEFAULT_BACKBONE_WEIGHT = "pretrained/efficientnet_b0_ra-3dd342df.pth"


def _resolve_repo_relative_path(path_str: str) -> str:
    """将仓库内相对路径解析到 ChangeDINO-main 目录，避免受 cwd 影响。"""
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return str(path)

    project_root = Path(__file__).resolve().parent
    candidate = (project_root / path).resolve()
    if candidate.exists():
        return str(candidate)
    return path_str

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


def normalize_contrast_pool_sizes(
    contrast_pool_sizes: List[int] | None,
    contrast_pool_size: int | None,
) -> List[int]:
    """兼容旧单值参数与新分层参数，统一返回 P2/P3/P4/P5 四层配置。"""
    if contrast_pool_sizes is not None:
        sizes = [int(v) for v in contrast_pool_sizes]
    elif contrast_pool_size is not None:
        sizes = [int(contrast_pool_size)]
    else:
        sizes = [5, 5, 5, 5]

    if len(sizes) == 1:
        sizes = sizes * 4
    elif len(sizes) != 4:
        raise ValueError(
            f"contrast_pool_sizes expects 1 or 4 ints, got {len(sizes)} values: {sizes}"
        )

    for size in sizes:
        if size < 1 or size % 2 == 0:
            raise ValueError(f"contrast pool size must be positive odd integer, got {size}")
    return sizes


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
        self.parser.add_argument("--name", type=str, default="WHU")
        self.parser.add_argument(
            "--dataroot", type=str, default="/ssddd/chingheng/CD-Dataset"
        )
        self.parser.add_argument("--dataset", type=str, default="WHU-CD")
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
        self.parser.add_argument(
            "--contrast_pool_size",
            type=int,
            default=None,
            help="兼容旧版本的单值池化核大小；若未指定 --contrast_pool_sizes，则复制到四层。",
        )
        self.parser.add_argument(
            "--contrast_pool_sizes",
            nargs="+",
            type=int,
            default=None,
            help="ContrastAwareDiff 在 P2/P3/P4/P5 的局部池化核大小，可传 1 个或 4 个奇数。",
        )
        self.parser.add_argument(
            "--refiner",
            type=str,
            default="topo",
            choices=["topo", "hybrid"],
            help="末端精修器：topo 仅做拓扑连通修复，hybrid 结合 p1 像素级 tiny flood 精修。",
        )
        self.parser.add_argument(
            "--topo_grid_size",
            type=int,
            default=16,
            help="FloodTopoRouter 节点网格大小 G，产生 G×G 个图节点。",
        )
        self.parser.add_argument(
            "--topo_hidden_dim",
            type=int,
            default=128,
            help="TopoEdgeTransformer 隐藏维度。",
        )
        self.parser.add_argument(
            "--topo_neighbor_k",
            type=int,
            default=12,
            help="每个网格节点的 KNN 邻居数。",
        )
        self.parser.add_argument(
            "--topo_neighbor_mode",
            type=str,
            default="mixed",
            choices=["knn", "mixed"],
            help="FloodTopoRouter 的邻接模式：knn 为纯局部近邻，mixed 额外加入轴向长边。",
        )
        self.parser.add_argument(
            "--topo_long_offsets",
            nargs="+",
            type=int,
            default=[2, 4],
            help="mixed 邻接下的轴向长边偏移量，单位为网格单元。",
        )
        self.parser.add_argument(
            "--topo_n_hops",
            type=int,
            default=3,
            help="图消息传递跳数，控制洪水证据传播范围。",
        )
        self.parser.add_argument(
            "--topo_loss_weight",
            type=float,
            default=0.5,
            help="拓扑连通性损失权重。",
        )
        self.parser.add_argument(
            "--topo_warmup_epochs",
            type=int,
            default=10,
            help="前若干个 epoch 不启用拓扑损失，避免早期训练过度偏向大区域结构。",
        )
        self.parser.add_argument(
            "--topo_min_node_occ",
            type=float,
            default=0.25,
            help="拓扑损失中参与监督的最小节点占据率，低于该值的 tiny 节点不计入负样本。",
        )

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
            default="dinov3/weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
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
            "--p2_window_size",
            type=int,
            default=8,
            help="P2 层 OCDA 窗口大小，默认与稳定三模块版保持一致。",
        )
        self.parser.add_argument(
            "--micro_gate",
            action="store_true",
            help="启用 DQ 风格动态微小目标门控，仅调节 P2/P3 差分与拓扑修正强度。",
        )
        self.parser.add_argument(
            "--dino_collab_mode",
            type=str,
            default="none",
            choices=["none", "legacy", "multilevel_v2"],
            help="DINO 与 CNN 的协同模式；none 表示仅编码器 PFF 融合（方案A，推荐），legacy/multilevel_v2 为旧版解码器注入（已弃用）。",
        )
        self.parser.add_argument(
            "--branch_consistency_weight",
            type=float,
            default=0.05,
            help="p2->p1 单向语义一致性损失权重，仅约束 non-tiny 区域。",
        )
        self.parser.add_argument(
            "--consistency_warmup_epochs",
            type=int,
            default=15,
            help="前若干个 epoch 不启用 p2->p1 一致性损失，避免 early-stage 错误 tiny prior 误约束。",
        )
        self.parser.add_argument(
            "--coarse_fp_consistency_weight",
            type=float,
            default=0.03,
            help="粗尺度前景需受 p2 局部变化支持的约束权重，用于抑制整块误报。",
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
            "--split_seed", type=int, default=42, help="S1GFloods 划分脚本与实验配置的默认随机种子"
        )
        self.parser.add_argument(
            "--stats_file",
            type=str,
            default="",
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
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        self.opt.dino_weight = _resolve_repo_relative_path(self.opt.dino_weight)
        if self.opt.backbone_weight:
            _validate_backbone_weight_path(self.opt.backbone_weight)
            self.opt.backbone_weight = _resolve_repo_relative_path(self.opt.backbone_weight)
        self.opt.dino_arch = resolve_dino_arch(self.opt.dino_arch, self.opt.dino_weight)
        self.opt.extract_ids = resolve_extract_ids(self.opt.dino_arch, self.opt.extract_ids)
        self.opt.contrast_pool_sizes = normalize_contrast_pool_sizes(
            self.opt.contrast_pool_sizes, self.opt.contrast_pool_size
        )
        if self.opt.p2_window_size < 1:
            raise ValueError("--p2_window_size must be a positive integer")
        if self.opt.branch_consistency_weight < 0:
            raise ValueError("--branch_consistency_weight must be >= 0")
        if self.opt.coarse_fp_consistency_weight < 0:
            raise ValueError("--coarse_fp_consistency_weight must be >= 0")
        if self.opt.consistency_warmup_epochs < 0:
            raise ValueError("--consistency_warmup_epochs must be >= 0")
        if self.opt.topo_warmup_epochs < 0:
            raise ValueError("--topo_warmup_epochs must be >= 0")
        if self.opt.disable_soft_alignment:
            self.opt.align_on_levels = []
        else:
            self.opt.align_on_levels = sorted({int(v) for v in self.opt.align_on_levels})
            invalid_align_levels = [v for v in self.opt.align_on_levels if v not in {1, 2, 3}]
            if invalid_align_levels:
                raise ValueError(
                    f"--align_on_levels only supports P1/P2/P3, got {invalid_align_levels}"
                )
        if not 0.0 <= self.opt.eval_fg_threshold <= 1.0:
            raise ValueError("--eval_fg_threshold must be within [0, 1]")
        if not getattr(self.opt, "topo_long_offsets", None):
            self.opt.topo_long_offsets = [2, 4]
        self.opt.topo_long_offsets = [int(v) for v in self.opt.topo_long_offsets if int(v) > 0]
        if self.opt.topo_neighbor_mode == "mixed" and not self.opt.topo_long_offsets:
            raise ValueError("--topo_long_offsets must contain at least one positive integer when mode=mixed")
        if self.opt.small_area_thresh < self.opt.tiny_area_thresh:
            raise ValueError("--small_area_thresh must be >= --tiny_area_thresh")
        self.opt.mean, self.opt.std = resolve_norm_stats(self.opt)

        args = vars(self.opt)

        print("------------ Options -------------")
        for k, v in sorted(args.items()):
            print("%s: %s" % (str(k), str(v)))
        print("-------------- End ----------------")

        return self.opt
