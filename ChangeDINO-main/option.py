import argparse
import json
from pathlib import Path
from typing import List

import torch

from model.blocks.dinov3_meta import DINO_ARCH_CHOICES, resolve_dino_arch, resolve_extract_ids


OPTICAL_MEAN = [0.430, 0.411, 0.296]
OPTICAL_STD = [0.213, 0.156, 0.143]
DEFAULT_BACKBONE_WEIGHT = "pretrained/convnextv2_nano_22k_224_ema.pt"


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
            "--topo_n_hops",
            type=int,
            default=2,
            help="图消息传递跳数，控制洪水证据传播范围。",
        )
        self.parser.add_argument(
            "--topo_loss_weight",
            type=float,
            default=0.5,
            help="拓扑连通性损失权重。",
        )

        self.parser.add_argument("--phase", type=str, default="train")
        self.parser.add_argument("--backbone", type=str, default="convnextv2_nano")
        self.parser.add_argument(
            "--backbone_weight",
            type=str,
            default=DEFAULT_BACKBONE_WEIGHT,
            help="CNN backbone 预训练权重路径；默认使用仓库内 ConvNeXtV2 nano 权重。",
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
            default=[2, 3],
            help="在哪些金字塔层上启用 deformable soft-alignment，默认 p2/p3。",
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
            "--directional_diff_expand",
            type=float,
            default=4.0,
            help="定向差分混合器的通道扩张比例。",
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
            self.opt.backbone_weight = _resolve_repo_relative_path(self.opt.backbone_weight)
        self.opt.dino_arch = resolve_dino_arch(self.opt.dino_arch, self.opt.dino_weight)
        self.opt.extract_ids = resolve_extract_ids(self.opt.dino_arch, self.opt.extract_ids)
        self.opt.mean, self.opt.std = resolve_norm_stats(self.opt)

        args = vars(self.opt)

        print("------------ Options -------------")
        for k, v in sorted(args.items()):
            print("%s: %s" % (str(k), str(v)))
        print("-------------- End ----------------")

        return self.opt
