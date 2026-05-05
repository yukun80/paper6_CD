import os
from argparse import ArgumentParser

import torch
from PIL import Image
from torchvision import transforms

from data.tif_io import is_tiff_path, read_sar_tif
from model.create_ChangeDINO import create_model
from option import Options, resolve_norm_stats
from model.blocks.dinov3_meta import resolve_dino_arch, resolve_extract_ids


def build_parser() -> ArgumentParser:
    """
    Reuse the training options so checkpoints/backbone configs stay in sync,
    and extend with paths for single-pair inference.
    """
    opt_builder = Options()
    opt_builder.init()
    parser = opt_builder.parser
    parser.set_defaults(name="WHU-ChangeDINO")
    parser.add_argument("--img_A", required=True, help="Path to time-A image.")
    parser.add_argument("--img_B", required=True, help="Path to time-B image.")
    parser.add_argument(
        "--output",
        type=str,
        default="./outputs/run_pred.png",
        help="Where to save the binary prediction mask (0/255).",
    )
    return parser


def parse_and_prepare() -> object:
    parser = build_parser()
    opt = parser.parse_args()

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
    if not opt.gpu_ids:
        raise ValueError("gpu_ids must include at least one GPU id (e.g., 0).")

    if torch.cuda.is_available():
        torch.cuda.set_device(opt.gpu_ids[0])
    else:
        raise EnvironmentError("CUDA is not available but gpu_ids were provided.")

    opt.phase = "test"
    opt.load_pretrain = True
    opt.batch_size = 1
    opt.num_workers = 0
    opt.dino_arch = resolve_dino_arch(opt.dino_arch, opt.dino_weight)
    opt.extract_ids = resolve_extract_ids(opt.dino_arch, opt.extract_ids)
    opt.mean, opt.std = resolve_norm_stats(opt)

    print("------------ Options -------------")
    for k, v in sorted(vars(opt).items()):
        print(f"{k}: {v}")
    print("-------------- End ----------------")

    return opt


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
    opt = parse_and_prepare()

    os.makedirs(os.path.dirname(opt.output) or ".", exist_ok=True)
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(tuple(opt.mean), tuple(opt.std))

    _, img_A = load_image(opt.img_A, to_tensor, normalize)
    _, img_B = load_image(opt.img_B, to_tensor, normalize)

    img_A = img_A.cuda(non_blocking=True)
    img_B = img_B.cuda(non_blocking=True)

    model = create_model(opt)
    model.eval()

    with torch.no_grad():
        pred = model.inference(img_A, img_B)
        pred = torch.argmax(pred, dim=1)
        pred_img = Image.fromarray(
            (pred[0].cpu().detach().numpy() * 255).astype("uint8")
        )
        pred_img.save(opt.output)
        print(f"Saved prediction to {opt.output}")


if __name__ == "__main__":
    main()
