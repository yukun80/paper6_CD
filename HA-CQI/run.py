import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from data.tif_io import is_tiff_path, read_sar_tif
from model.engine import build_hacqi_engine
from model.checkpointing import (
    apply_checkpoint_model_config,
    checkpoint_data_config,
    checkpoint_model_config,
    extract_network_state,
    load_checkpoint_payload,
    resolve_inference_threshold,
)
from option import (
    Options,
    _resolve_existing_project_path,
    _resolve_project_output_path,
    _resolve_repo_relative_path,
    _validate_backbone_weight_path,
    resolve_norm_stats,
    validate_stats_provenance,
)
from model.modules.dino_meta import resolve_dino_arch, resolve_extract_ids


def build_parser() -> ArgumentParser:
    """
    Reuse the training options so checkpoints/backbone configs stay in sync,
    and extend with paths for single-pair inference.
    """
    opt_builder = Options()
    opt_builder.init()
    parser = opt_builder.parser
    parser.set_defaults(name="HA-CQI")
    parser.add_argument("--img_A", required=True, help="Path to time-A image.")
    parser.add_argument("--img_B", required=True, help="Path to time-B image.")
    parser.add_argument(
        "--output",
        type=str,
        default="./outputs/run_pred.png",
        help="Where to save the binary prediction mask (0/255).",
    )
    return parser


def parse_and_prepare() -> tuple[object, dict]:
    parser = build_parser()
    opt = parser.parse_args()
    if not opt.checkpoint:
        parser.error("--checkpoint is required for pair inference")
    explicit_arguments = {
        token[2:].split("=", 1)[0].replace("-", "_")
        for token in sys.argv[1:]
        if token.startswith("--")
    }
    opt.checkpoint = _resolve_existing_project_path(opt.checkpoint)
    payload = load_checkpoint_payload(opt.checkpoint, map_location="cpu")
    checkpoint_config = checkpoint_model_config(payload)
    opt = apply_checkpoint_model_config(opt, checkpoint_config, explicit_arguments)
    data_config = checkpoint_data_config(payload) or {}
    if "stats_file" not in explicit_arguments and data_config.get("stats_file"):
        opt.stats_file = data_config["stats_file"]
    if "dataset" not in explicit_arguments and data_config.get("dataset"):
        opt.dataset = data_config["dataset"]
    if "dataroot" not in explicit_arguments and data_config.get("dataroot"):
        opt.dataroot = data_config["dataroot"]
    opt.threshold, opt.threshold_source = resolve_inference_threshold(
        payload,
        explicit_threshold=opt.threshold,
    )

    if opt.dataset_mode == "auto":
        if str(opt.dataset).startswith("S1GFloods"):
            opt.dataset_mode = "sar"
        else:
            opt.dataset_mode = "default"

    str_ids = opt.gpu_ids.split(",")
    opt.gpu_ids = []
    for str_id in str_ids:
        gid = int(str_id)
        if gid >= 0:
            opt.gpu_ids.append(gid)
    if torch.cuda.is_available() and opt.gpu_ids:
        torch.cuda.set_device(opt.gpu_ids[0])

    opt.phase = "test"
    opt.batch_size = 1
    opt.num_workers = 0
    opt.dataroot = _resolve_existing_project_path(opt.dataroot)
    if opt.stats_file:
        opt.stats_file = _resolve_existing_project_path(opt.stats_file)
    opt.checkpoint_dir = _resolve_project_output_path(opt.checkpoint_dir)
    opt.output = _resolve_project_output_path(opt.output)
    opt.dino_weight = _resolve_repo_relative_path(opt.dino_weight)
    if opt.backbone_weight:
        _validate_backbone_weight_path(opt.backbone_weight)
        opt.backbone_weight = _resolve_repo_relative_path(opt.backbone_weight)
    opt.dino_arch = resolve_dino_arch(opt.dino_arch, opt.dino_weight)
    opt.extract_ids = resolve_extract_ids(opt.dino_arch, opt.extract_ids)
    opt.mean, opt.std = resolve_norm_stats(opt)
    validate_stats_provenance(opt)

    print("------------ Options -------------")
    for k, v in sorted(vars(opt).items()):
        print(f"{k}: {v}")
    print("-------------- End ----------------")

    return opt, payload


def load_image(path, to_tensor, normalize):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} does not exist.")
    if is_tiff_path(path):
        tensor = normalize(read_sar_tif(path)).unsqueeze(0)
        return None, tensor
    img = Image.open(path).convert("RGB")
    tensor = normalize(to_tensor(img)).unsqueeze(0)
    return img, tensor


def main():
    opt, payload = parse_and_prepare()

    os.makedirs(os.path.dirname(opt.output) or ".", exist_ok=True)
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(tuple(opt.mean), tuple(opt.std))

    _, img_A = load_image(opt.img_A, to_tensor, normalize)
    _, img_B = load_image(opt.img_B, to_tensor, normalize)

    model = build_hacqi_engine(opt)
    model.model.load_state_dict(extract_network_state(payload), strict=True)
    model.eval()
    img_A = img_A.to(model.device)
    img_B = img_B.to(model.device)

    with torch.no_grad():
        pred = model.inference(img_A, img_B)
        pred_prob = torch.softmax(pred, dim=1)[:, 1]
        pred = (pred_prob >= float(opt.threshold)).long()
        pred_img = Image.fromarray(
            (pred[0].cpu().detach().numpy() * 255).astype("uint8")
        )
        pred_img.save(opt.output)
        print(
            f"Saved prediction to {opt.output} "
            f"(threshold={opt.threshold:.2f}, source={opt.threshold_source})"
        )


if __name__ == "__main__":
    main()
