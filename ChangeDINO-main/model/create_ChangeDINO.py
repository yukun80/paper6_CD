from .ChangeDINO import ChangeModel
import torch
from torch import nn
import os
import torch.optim as optim
from datetime import datetime
from .loss.focal import FocalLoss
from .loss.dice import DICELoss


def get_model(backbone_name="efficientnet_b0", fpn_channels=128, n_layers=[1, 1, 1, 1], **kwargs):
    model = ChangeModel(backbone_name, fpn_channels, n_layers=n_layers, **kwargs)
    # print(model)
    return model


def resolve_unique_run_name(checkpoint_dir: str, base_name: str) -> str:
    """为训练实验名追加日期后缀，并在重名时递增序号，避免覆盖旧记录。"""
    date_suffix = datetime.now().strftime("%Y%m%d")
    candidate_name = f"{base_name}-{date_suffix}"
    candidate_dir = os.path.join(checkpoint_dir, candidate_name)
    if not os.path.exists(candidate_dir):
        return candidate_name

    index = 1
    while True:
        candidate_name = f"{base_name}-{date_suffix}-{index}"
        candidate_dir = os.path.join(checkpoint_dir, candidate_name)
        if not os.path.exists(candidate_dir):
            return candidate_name
        index += 1


class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.device = torch.device(
            "cuda:%s" % opt.gpu_ids[0] if torch.cuda.is_available() else "cpu"
        )
        self.opt = opt
        self.base_lr = opt.lr

        # 仅训练阶段自动创建新实验目录，测试/推理保持用户传入的目录名不变。
        resolved_name = opt.name
        if getattr(opt, "phase", "train") == "train":
            resolved_name = resolve_unique_run_name(opt.checkpoint_dir, opt.name)
        self.opt.name = resolved_name
        self.save_dir = os.path.join(opt.checkpoint_dir, resolved_name)
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"save_dir resolved to: {self.save_dir}")

        self.model = get_model(
            backbone_name=opt.backbone,
            fpn_name=opt.fpn,
            backbone_weight=opt.backbone_weight,
            fpn_channels=opt.fpn_channels,
            deform_groups=opt.deform_groups,
            gamma_mode=opt.gamma_mode,
            beta_mode=opt.beta_mode,
            n_layers=opt.n_layers,
            align_window=opt.align_window,
            align_points=opt.align_points,
            align_heads=opt.align_heads,
            align_on_levels=opt.align_on_levels,
            align_qkv_bias=opt.align_qkv_bias,
            align_offset_groups=opt.align_offset_groups,
            contrast_pool_sizes=getattr(opt, "contrast_pool_sizes", [5, 5, 5, 5]),
            p2_window_size=getattr(opt, "p2_window_size", 8),
            refiner=getattr(opt, "refiner", "topo"),
            micro_gate=getattr(opt, "micro_gate", False),
            dino_collab_mode=getattr(opt, "dino_collab_mode", "multilevel_v2"),
            branch_consistency_weight=float(getattr(opt, "branch_consistency_weight", 0.05)),
            consistency_warmup_epochs=int(getattr(opt, "consistency_warmup_epochs", 15)),
            topo_grid_size=getattr(opt, "topo_grid_size", 16),
            topo_hidden_dim=getattr(opt, "topo_hidden_dim", 128),
            topo_neighbor_k=getattr(opt, "topo_neighbor_k", 12),
            topo_neighbor_mode=getattr(opt, "topo_neighbor_mode", "mixed"),
            topo_long_offsets=getattr(opt, "topo_long_offsets", [2, 4]),
            topo_n_hops=getattr(opt, "topo_n_hops", 3),
            topo_min_node_occ=getattr(opt, "topo_min_node_occ", 0.25),
            dino_arch=opt.dino_arch,
            extract_ids=opt.extract_ids,
            dino_weight=opt.dino_weight,
            device=self.device,
        )
        self.topo_loss_weight = getattr(opt, "topo_loss_weight", 0.5)
        self.branch_consistency_weight = float(getattr(opt, "branch_consistency_weight", 0.05))
        self.aux_head_weights = [1.0, 1.0, 0.5, 0.25, 0.25]
        self.focal = FocalLoss(alpha=opt.alpha, gamma=opt.gamma)
        self.dice = DICELoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay
        )
        self.schedular = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, opt.num_epochs, eta_min=1e-7
        )
        if opt.load_pretrain:
            self.load_ckpt(self.model, self.optimizer, opt.name, opt.backbone)
        self.model.to(self.device)

        print("---------- Networks initialized -------------")

    def forward(self, x1, x2, label, epoch: int | None = None):
        final_pred, preds, topo_loss, consistency_loss = self.model(
            x1, x2, gt_mask=label, current_epoch=epoch
        )
        label = label.long()
        focal = 0.5 * self.focal(final_pred, label)
        dice = self.dice(final_pred, label)
        for weight, pred in zip(self.aux_head_weights, preds):
            focal += 0.5 * weight * self.focal(pred, label)
            dice += weight * self.dice(pred, label)
        return final_pred, focal, dice, topo_loss, consistency_loss

    @torch.inference_mode()
    def inference(self, x1, x2):
        pred = self.model._forward(x1, x2)
        return pred

    def load_ckpt(self, network, optimizer, name, backbone):
        save_filename = "%s_%s_best.pth" % (name, backbone)
        save_path = os.path.join(self.save_dir, save_filename)
        if not os.path.isfile(save_path):
            print("%s not exists yet!" % save_path)
            raise ("%s must exist!" % save_filename)
        else:
            checkpoint = torch.load(
                save_path, map_location=self.device, weights_only=True
            )
            state_dict = checkpoint["network"]
            current_state = network.state_dict()
            filtered_state = {}
            skipped = []
            for key, value in state_dict.items():
                if key not in current_state:
                    continue
                if current_state[key].shape != value.shape:
                    skipped.append(
                        (key, tuple(value.shape), tuple(current_state[key].shape))
                    )
                    continue
                filtered_state[key] = value
            network.load_state_dict(filtered_state, strict=False)
            if skipped:
                print(
                    "skip incompatible pretrain keys:",
                    [f"{key}:{src}->{dst}" for key, src, dst in skipped[:5]],
                )
            print("load pre-trained")

    def _build_checkpoint_meta(self):
        """保存推理重建模型所需的最小配置，避免不同 DINO 尺寸下靠默认值猜结构。"""
        return {
            "model_config": {
                "backbone": self.opt.backbone,
                "backbone_weight": self.opt.backbone_weight,
                "fpn_channels": int(self.opt.fpn_channels),
                "deform_groups": int(self.opt.deform_groups),
                "gamma_mode": self.opt.gamma_mode,
                "beta_mode": self.opt.beta_mode,
                "n_layers": [int(v) for v in self.opt.n_layers],
                "align_window": int(self.opt.align_window),
                "align_points": int(self.opt.align_points),
                "align_heads": int(self.opt.align_heads),
                "align_on_levels": [int(v) for v in self.opt.align_on_levels],
                "align_qkv_bias": bool(self.opt.align_qkv_bias),
                "align_offset_groups": int(self.opt.align_offset_groups),
                "contrast_pool_sizes": [
                    int(v) for v in getattr(self.opt, "contrast_pool_sizes", [5, 5, 5, 5])
                ],
                "p2_window_size": int(getattr(self.opt, "p2_window_size", 8)),
                "refiner": getattr(self.opt, "refiner", "topo"),
                "micro_gate": bool(getattr(self.opt, "micro_gate", False)),
                "dino_collab_mode": getattr(self.opt, "dino_collab_mode", "multilevel_v2"),
                "branch_consistency_weight": float(
                    getattr(self.opt, "branch_consistency_weight", 0.05)
                ),
                "consistency_warmup_epochs": int(
                    getattr(self.opt, "consistency_warmup_epochs", 15)
                ),
                "topo_grid_size": int(getattr(self.opt, "topo_grid_size", 16)),
                "topo_hidden_dim": int(getattr(self.opt, "topo_hidden_dim", 128)),
                "topo_neighbor_k": int(getattr(self.opt, "topo_neighbor_k", 12)),
                "topo_neighbor_mode": getattr(self.opt, "topo_neighbor_mode", "mixed"),
                "topo_long_offsets": [
                    int(v) for v in getattr(self.opt, "topo_long_offsets", [2, 4])
                ],
                "topo_n_hops": int(getattr(self.opt, "topo_n_hops", 3)),
                "topo_min_node_occ": float(getattr(self.opt, "topo_min_node_occ", 0.25)),
                "dino_arch": self.opt.dino_arch,
                "dino_weight": self.opt.dino_weight,
                "extract_ids": [int(v) for v in self.opt.extract_ids],
            }
        }

    def save_ckpt(self, network, optimizer, model_name, backbone):
        save_filename = "%s_%s_best.pth" % (model_name, backbone)
        save_path = os.path.join(self.save_dir, save_filename)
        if os.path.exists(save_path):
            os.remove(save_path)
        torch.save(
            {
                "network": network.cpu().state_dict(),
                "optimizer": optimizer.state_dict(),
                "meta": self._build_checkpoint_meta(),
            },
            save_path,
        )
        if torch.cuda.is_available():
            network.cuda()

    def save(self, model_name, backbone):
        self.save_ckpt(self.model, self.optimizer, model_name, backbone)

    def save_epoch_ckpt(self, network, optimizer, model_name, backbone, epoch):
        """每 N epoch 定期保存一次快照，文件名含 epoch 编号，与 best 权重互不覆盖。"""
        save_filename = f"{model_name}_{backbone}_epoch{epoch}.pth"
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(
            {
                "network": network.cpu().state_dict(),
                "optimizer": optimizer.state_dict(),
                "meta": self._build_checkpoint_meta(),
                "epoch": epoch,
            },
            save_path,
        )
        if torch.cuda.is_available():
            network.cuda()

    def save_periodic(self, model_name, backbone, epoch):
        self.save_epoch_ckpt(self.model, self.optimizer, model_name, backbone, epoch)

    def name(self):
        return self.opt.name


def create_model(opt):
    model = Model(opt)
    print("model [%s] was created" % model.name())

    return model.to(model.device)
