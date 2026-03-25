from .ChangeDINO import ChangeModel
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import os
import torch.optim as optim
from datetime import datetime
from .loss.focal import FocalLoss
from .loss.dice import DICELoss


def get_model(backbone_name="convnextv2_nano", fpn_channels=128, n_layers=[1, 1, 1, 1], **kwargs):
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
            directional_diff_expand=opt.directional_diff_expand,
            dino_arch=opt.dino_arch,
            extract_ids=opt.extract_ids,
            dino_weight=opt.dino_weight,
            device=self.device,
        )
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

    def forward(self, x1, x2, label):
        final_pred, preds = self.model(x1, x2)
        label = label.long()
        focal = self.focal(final_pred, label)
        dice = self.dice(final_pred, label)
        for i in range(len(preds)):
            focal += self.focal(preds[i], label)
            dice += 0.5 * self.dice(preds[i], label)

        return final_pred, focal, dice

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
            network.load_state_dict(checkpoint["network"], strict=False)
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
                "directional_diff_expand": float(self.opt.directional_diff_expand),
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

    def name(self):
        return self.opt.name


def create_model(opt):
    model = Model(opt)
    print("model [%s] was created" % model.name())

    return model.to(model.device)
