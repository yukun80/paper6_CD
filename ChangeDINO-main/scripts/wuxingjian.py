import numpy as np
import rasterio
from scipy.io import savemat
import os


def normalize_sar_to_uint8(image_array):
    """
    对单通道 SAR 影像进行 2%-98% 线性拉伸，并转换为 0-255 的 uint8 格式
    """
    img_clean = np.nan_to_num(image_array)
    p2, p98 = np.percentile(img_clean, (2, 98))
    img_clipped = np.clip(img_clean, p2, p98)

    # 避免分母为0
    denominator = p98 - p2 if p98 != p2 else 1e-8
    img_normalized = (img_clipped - p2) / denominator * 255.0

    return img_normalized.astype(np.uint8)


def read_tif_with_rasterio(tif_path):
    """
    使用 rasterio 读取单通道 tif，并返回二维 numpy 数组
    """
    with rasterio.open(tif_path) as src:
        # read(1) 读取第一个波段 (1-based index)
        image = src.read(1)
    return image


def create_dino_inference_mat(pre_tif, post_tif, output_mat):
    print(f"正在使用 rasterio 读取影像...")
    # 1. 读取 Sentinel-1 单通道 TIF 数据
    pre_img = read_tif_with_rasterio(pre_tif)
    post_img = read_tif_with_rasterio(post_tif)

    # 去除可能多余的维度
    pre_img = np.squeeze(pre_img)
    post_img = np.squeeze(post_img)

    if pre_img.shape != post_img.shape:
        raise ValueError(f"影像维度不匹配！Pre: {pre_img.shape}, Post: {post_img.shape}")

    H, W = pre_img.shape
    print(f"影像原始维度为: {H} x {W}")

    print("正在进行灰度映射与 8-bit 转换...")
    pre_uint8 = normalize_sar_to_uint8(pre_img)
    post_uint8 = normalize_sar_to_uint8(post_img)

    print("正在构建三通道输入特征...")
    image_t1 = np.stack([pre_uint8, pre_uint8, pre_uint8], axis=-1)
    image_t2 = np.stack([post_uint8, post_uint8, post_uint8], axis=-1)

    Ref_gt = np.ones((H, W), dtype=np.uint8)

    mat_dict = {"Ref_gt": Ref_gt, "image_t1": image_t1, "image_t2": image_t2}

    savemat(output_mat, mat_dict)
    print(f"\n转换成功！文件已保存至: {output_mat}")
    print(f"输出结构校验:")
    print(f"  - 'Ref_gt'   -> 维度: {Ref_gt.shape}, 类型: {Ref_gt.dtype}")
    print(f"  - 'image_t1' -> 维度: {image_t1.shape}, 类型: {image_t1.dtype}")
    print(f"  - 'image_t2' -> 维度: {image_t2.shape}, 类型: {image_t2.dtype}")


if __name__ == "__main__":
    pre_path = "/home/yukun/codes/paper6_waterlogging/datasets/S1_Henan/Zhengzhou_S1GRD_ASCENDING_VH_pre_clip.tif"
    post_path = "/home/yukun/codes/paper6_waterlogging/datasets/S1_Henan/Zhengzhou_S1GRD_ASCENDING_VH_Post_clip.tif"
    out_path = "/home/yukun/codes/paper6_waterlogging/datasets/S1_Henan/dataset#16.mat"

    create_dino_inference_mat(pre_path, post_path, out_path)
