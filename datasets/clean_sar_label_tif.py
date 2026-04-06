#!/usr/bin/env python3
"""清理 SAR 洪水变化标签中的零散小前景，并保持 GeoTIFF 元数据不变。"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import rasterio
from scipy import ndimage

"""
默认直接处理仓库中的两个 GF3 标签：
python datasets/clean_sar_label_tif.py --dry-run

显式指定输入：
python datasets/clean_sar_label_tif.py \
  --input datasets/GF3_Henan/GF3_Zhengzhou_label.tif \
          datasets/GF3_Zhuozhou/GF3_Zhuozhou_label.tif \
  --min-area 16 \
  --morph-op close \
  --kernel-size 3
"""


DEFAULT_INPUTS = (
    Path("datasets/GF3_Zhuozhou/GF3_Zhuozhou_label.tif"),
    Path("datasets/GF3_Henan/GF3_Zhengzhou_label.tif"),
)


@dataclass(frozen=True)
class CleanStats:
    """记录单个标签的清洗前后统计，便于复核参数效果。"""

    input_path: str
    output_path: str
    report_path: str
    shape: tuple[int, int]
    dtype: str
    nodata: float | int | None
    foreground_value: int
    background_value: int
    connectivity: int
    min_area: int
    morph_op: str
    kernel_size: int
    valid_pixels: int
    foreground_pixels_before: int
    foreground_pixels_after_area_filter: int
    foreground_pixels_after_morph: int
    removed_components: int
    removed_pixels: int
    component_count_before: int
    component_count_after_area_filter: int
    component_count_after_morph: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Clean small noisy objects in SAR label GeoTIFF files")
    parser.add_argument(
        "--input",
        type=Path,
        nargs="*",
        default=None,
        help="待处理的标签 tif 路径；未提供时默认处理两个 GF3 标签。",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="可选：从目录中按 glob 搜索待处理 tif。",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="*_label.tif",
        help="与 --input-dir 配合使用的匹配模式。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="输出目录；未提供时默认与输入文件同目录。",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_clean",
        help="输出文件名后缀，例如 xxx_clean.tif。",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=16,
        help="删除小于该像素面积的前景连通域。",
    )
    parser.add_argument(
        "--connectivity",
        type=int,
        choices=[4, 8],
        default=8,
        help="连通域连接方式：4 邻域或 8 邻域。",
    )
    parser.add_argument(
        "--morph-op",
        type=str,
        choices=["none", "close", "open-close"],
        default="close",
        help="面积过滤后的轻量形态学处理方式。",
    )
    parser.add_argument(
        "--kernel-size",
        type=int,
        default=3,
        help="形态学结构元尺寸，建议使用奇数。",
    )
    parser.add_argument(
        "--foreground-value",
        type=int,
        default=1,
        help="前景标签值，默认 1。",
    )
    parser.add_argument(
        "--background-value",
        type=int,
        default=0,
        help="背景标签值，默认 0。",
    )
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已存在输出文件。")
    parser.add_argument("--dry-run", action="store_true", help="仅打印统计信息，不写文件。")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.min_area < 1:
        raise ValueError("--min-area must be >= 1")
    if args.kernel_size < 1:
        raise ValueError("--kernel-size must be >= 1")
    if args.morph_op != "none" and args.kernel_size % 2 == 0:
        raise ValueError("--kernel-size must be odd when morphology is enabled")
    return args


def resolve_input_paths(args: argparse.Namespace) -> list[Path]:
    """聚合显式输入、目录搜索和默认 GF3 标签，避免命令入口太死板。"""
    paths: list[Path] = []
    if args.input:
        paths.extend(args.input)
    if args.input_dir is not None:
        if not args.input_dir.is_dir():
            raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
        paths.extend(sorted(args.input_dir.glob(args.glob)))
    if not paths:
        paths = [path for path in DEFAULT_INPUTS if path.is_file()]

    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        if not path.is_file():
            raise FileNotFoundError(f"Input file not found: {path}")
        seen.add(resolved)
        deduped.append(path)
    if not deduped:
        raise ValueError("No input label tif files were found.")
    return deduped


def build_valid_mask(arr: np.ndarray, nodata: float | int | None) -> np.ndarray:
    """只在有效像素内做标签清洗，避免 nodata 区域被误并入前景。"""
    valid = np.isfinite(arr)
    if nodata is not None:
        valid &= arr != nodata
    return valid


def build_connectivity_structure(connectivity: int) -> np.ndarray:
    if connectivity == 4:
        return ndimage.generate_binary_structure(rank=2, connectivity=1)
    return np.ones((3, 3), dtype=bool)


def count_components(mask: np.ndarray, structure: np.ndarray) -> int:
    _, count = ndimage.label(mask, structure=structure)
    return int(count)


def remove_small_components(
    mask: np.ndarray,
    structure: np.ndarray,
    min_area: int,
) -> tuple[np.ndarray, int, int, int, int]:
    """通过连通域面积过滤，优先去掉真正零散的小噪声块。"""
    labeled, component_count = ndimage.label(mask, structure=structure)
    if component_count == 0:
        return mask.copy(), 0, 0, 0, 0

    component_sizes = np.bincount(labeled.ravel())[1:]
    keep_ids = np.flatnonzero(component_sizes >= min_area) + 1
    keep_mask = np.zeros(component_count + 1, dtype=bool)
    keep_mask[keep_ids] = True
    cleaned = keep_mask[labeled]

    removed = component_sizes < min_area
    removed_components = int(removed.sum())
    removed_pixels = int(component_sizes[removed].sum())
    kept_components = int((~removed).sum())
    return cleaned, int(component_count), kept_components, removed_components, removed_pixels


def apply_morphology(mask: np.ndarray, valid_mask: np.ndarray, op: str, kernel_size: int) -> np.ndarray:
    """仅做轻量修补；默认闭运算用于填补极小裂缝，不主动强切边界。"""
    if op == "none":
        return mask.copy()

    kernel = np.ones((kernel_size, kernel_size), dtype=bool)
    working = mask & valid_mask
    if op == "close":
        refined = ndimage.binary_closing(working, structure=kernel)
    else:
        refined = ndimage.binary_opening(working, structure=kernel)
        refined = ndimage.binary_closing(refined, structure=kernel)
    return refined & valid_mask


def build_output_paths(input_path: Path, output_dir: Path | None, suffix: str) -> tuple[Path, Path]:
    output_root = input_path.parent if output_dir is None else output_dir
    output_path = output_root / f"{input_path.stem}{suffix}{input_path.suffix}"
    report_path = output_root / f"{input_path.stem}{suffix}_report.json"
    return output_path, report_path


def ensure_writable(paths: Iterable[Path], overwrite: bool, dry_run: bool) -> None:
    if dry_run:
        return
    for path in paths:
        if path.exists() and not overwrite:
            raise FileExistsError(f"Output exists, use --overwrite to replace it: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)


def clean_label_array(
    arr: np.ndarray,
    nodata: float | int | None,
    foreground_value: int,
    background_value: int,
    min_area: int,
    connectivity: int,
    morph_op: str,
    kernel_size: int,
) -> tuple[np.ndarray, CleanStats]:
    """核心清洗逻辑：先面积过滤，再按需做轻量形态学修补。"""
    valid_mask = build_valid_mask(arr, nodata)
    foreground_mask = valid_mask & (arr == foreground_value)
    structure = build_connectivity_structure(connectivity)

    area_filtered, before_components, after_area_components, removed_components, removed_pixels = (
        remove_small_components(foreground_mask, structure, min_area)
    )
    morphed = apply_morphology(area_filtered, valid_mask, morph_op, kernel_size)

    out_arr = arr.copy()
    out_arr[valid_mask] = background_value
    out_arr[morphed] = foreground_value

    stats = CleanStats(
        input_path="",
        output_path="",
        report_path="",
        shape=(int(arr.shape[0]), int(arr.shape[1])),
        dtype=str(arr.dtype),
        nodata=nodata,
        foreground_value=int(foreground_value),
        background_value=int(background_value),
        connectivity=int(connectivity),
        min_area=int(min_area),
        morph_op=morph_op,
        kernel_size=int(kernel_size),
        valid_pixels=int(valid_mask.sum()),
        foreground_pixels_before=int(foreground_mask.sum()),
        foreground_pixels_after_area_filter=int(area_filtered.sum()),
        foreground_pixels_after_morph=int(morphed.sum()),
        removed_components=removed_components,
        removed_pixels=removed_pixels,
        component_count_before=before_components,
        component_count_after_area_filter=after_area_components,
        component_count_after_morph=count_components(morphed, structure),
    )
    return out_arr, stats


def write_tif(output_path: Path, arr: np.ndarray, profile: dict[str, object]) -> None:
    updated = profile.copy()
    updated.update(
        {
            "driver": "GTiff",
            "count": 1,
            "dtype": str(arr.dtype),
            "compress": updated.get("compress", "LZW") or "LZW",
        }
    )
    with rasterio.open(output_path, "w", **updated) as ds:
        ds.write(arr, 1)


def write_report(report_path: Path, stats: CleanStats) -> None:
    report = asdict(stats)
    report["area_filter_delta_pixels"] = (
        stats.foreground_pixels_after_area_filter - stats.foreground_pixels_before
    )
    report["morph_delta_pixels"] = (
        stats.foreground_pixels_after_morph - stats.foreground_pixels_after_area_filter
    )
    report["net_delta_pixels"] = (
        stats.foreground_pixels_after_morph - stats.foreground_pixels_before
    )
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def process_file(input_path: Path, args: argparse.Namespace) -> CleanStats:
    output_path, report_path = build_output_paths(input_path, args.output_dir, args.suffix)
    ensure_writable((output_path, report_path), overwrite=args.overwrite, dry_run=args.dry_run)

    with rasterio.open(input_path) as src:
        if src.count != 1:
            raise ValueError(f"Only single-band label tif is supported: {input_path}")
        arr = src.read(1)
        cleaned, stats = clean_label_array(
            arr=arr,
            nodata=src.nodata,
            foreground_value=args.foreground_value,
            background_value=args.background_value,
            min_area=args.min_area,
            connectivity=args.connectivity,
            morph_op=args.morph_op,
            kernel_size=args.kernel_size,
        )
        stats = CleanStats(
            **{
                **asdict(stats),
                "input_path": str(input_path),
                "output_path": str(output_path),
                "report_path": str(report_path),
            }
        )
        if not args.dry_run:
            write_tif(output_path, cleaned, src.profile)
            write_report(report_path, stats)
    return stats


def print_summary(stats_list: Sequence[CleanStats], dry_run: bool) -> None:
    mode = "DRY-RUN" if dry_run else "DONE"
    for stats in stats_list:
        area_delta = stats.foreground_pixels_after_area_filter - stats.foreground_pixels_before
        morph_delta = stats.foreground_pixels_after_morph - stats.foreground_pixels_after_area_filter
        net_delta = stats.foreground_pixels_after_morph - stats.foreground_pixels_before
        print(f"[{mode}] input : {stats.input_path}")
        print(f"[{mode}] output: {stats.output_path}")
        print(f"[{mode}] report: {stats.report_path}")
        print(
            f"[{mode}] area filter fg: {stats.foreground_pixels_before} -> "
            f"{stats.foreground_pixels_after_area_filter} (delta {area_delta})"
        )
        print(
            f"[{mode}] morph refine fg: {stats.foreground_pixels_after_area_filter} -> "
            f"{stats.foreground_pixels_after_morph} (delta {morph_delta})"
        )
        print(
            f"[{mode}] components: {stats.component_count_before} -> "
            f"{stats.component_count_after_morph}, removed_components={stats.removed_components}, "
            f"removed_pixels={stats.removed_pixels}"
        )
        print(
            f"[{mode}] params: min_area={stats.min_area}, connectivity={stats.connectivity}, "
            f"morph_op={stats.morph_op}, kernel_size={stats.kernel_size}, nodata={stats.nodata}"
        )
        print(f"[{mode}] net foreground delta: {net_delta}")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    input_paths = resolve_input_paths(args)
    stats_list = [process_file(path, args) for path in input_paths]
    print_summary(stats_list, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
