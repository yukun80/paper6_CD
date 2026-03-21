#!/usr/bin/env python3
"""将 ETCI-2021 时序语义分割切片整理为 ChangeDINO 可直接读取的变化检测目录。"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

"""
python ChangeDINO-main/scripts/prepare_etci2021_cd.py \
  --src-root datasets/ETCI-2021 \
  --out-root datasets/ETCI2021_CD_DINO \
  --seed 42 \
  --overwrite
"""

SCENE_PATTERN = re.compile(r"^(?P<region>.+)_(?P<timestamp>\d{8}t\d{6})$")
VH_PATTERN = re.compile(r"^(?P<prefix>.+_\d{8}t\d{6})_(?P<tile>x-\d+_y-\d+)_vh\.png$")
LABEL_PATTERN = re.compile(r"^(?P<prefix>.+_\d{8}t\d{6})_(?P<tile>x-\d+_y-\d+)\.png$")
MAX_WHITE_RATIO = 0.3
MAX_WHITE_RATIO_DIFF = 0.10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Prepare ETCI-2021 in ChangeDINO CD layout")
    parser.add_argument("--src-root", type=Path, default=Path("datasets/ETCI-2021"))
    parser.add_argument("--out-root", type=Path, default=Path("datasets/ETCI2021_CD_DINO"))
    parser.add_argument(
        "--pairing-policy",
        choices=["adjacent", "flood-priority", "hybrid"],
        default="hybrid",
        help="hybrid=相邻对 + 洪水显著增加优先对。",
    )
    parser.add_argument(
        "--val-regions",
        type=str,
        default="",
        help="逗号分隔的验证区域名。默认自动选择训练集中时相数最少的区域作为 val。",
    )
    parser.add_argument(
        "--train-regions",
        type=str,
        default="",
        help="逗号分隔的训练区域名。为空时使用 train 下除 val 之外的全部区域。",
    )
    parser.add_argument(
        "--test-subsets",
        type=str,
        default="",
        help="已废弃；当前版本忽略 test_internal，只使用 train 中的数据构造 train/val。",
    )
    parser.add_argument(
        "--link-mode",
        choices=["copy", "symlink", "hardlink"],
        default="copy",
        help="A/B 图像写入方式；label 始终重新生成。",
    )
    parser.add_argument(
        "--min-change-ratio",
        type=float,
        default=0.001,
        help="低于该阈值的弱变化样本会按 drop-zero-change-mode 强过滤。",
    )
    parser.add_argument(
        "--drop-zero-change-mode",
        choices=["drop", "drop80"],
        default="drop80",
        help="对于 0 < change_ratio < min-change-ratio 的样本，drop=直接丢弃，drop80=丢弃 80%。",
    )
    parser.add_argument(
        "--flood-increase-threshold",
        type=float,
        default=0.01,
        help="用于从多时相中挑选“前少后多”优先配对的占比差阈值。",
    )
    parser.add_argument(
        "--max-white-ratio",
        type=float,
        default=MAX_WHITE_RATIO,
        help="任一 VH 图像纯白像素比例超过该阈值时丢弃样本对。",
    )
    parser.add_argument(
        "--max-white-ratio-diff",
        type=float,
        default=MAX_WHITE_RATIO_DIFF,
        help="仅当 T1/T2 的纯白像素比例差值超过该阈值时才因不一致而丢弃。",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def parse_region_list(value: str) -> list[str]:
    if not value.strip():
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def ensure_args(args: argparse.Namespace) -> None:
    if not args.src_root.is_dir():
        raise FileNotFoundError(f"Source root not found: {args.src_root}")
    if not 0 <= args.min_change_ratio < 1:
        raise ValueError("--min-change-ratio must be within [0, 1)")
    if not 0 <= args.flood_increase_threshold < 1:
        raise ValueError("--flood-increase-threshold must be within [0, 1)")
    if not 0 <= args.max_white_ratio <= 1:
        raise ValueError("--max-white-ratio must be within [0, 1]")
    if not 0 <= args.max_white_ratio_diff <= 1:
        raise ValueError("--max-white-ratio-diff must be within [0, 1]")


def parse_scene_dir(scene_dir: Path) -> tuple[str, str]:
    match = SCENE_PATTERN.match(scene_dir.name)
    if not match:
        raise ValueError(f"Unexpected scene directory name: {scene_dir.name}")
    return match.group("region"), match.group("timestamp")


def scan_vh_map(folder: Path, scene_name: str, strict: bool) -> dict[str, Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing directory: {folder}")
    mapping: dict[str, Path] = {}
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() != ".png":
            continue
        match = VH_PATTERN.match(path.name)
        if not match:
            if strict:
                raise ValueError(f"Unexpected VH file name: {path}")
            continue
        if strict and match.group("prefix") != scene_name:
            raise ValueError(f"VH file prefix mismatch: {path.name} vs scene {scene_name}")
        mapping[match.group("tile")] = path
    if not mapping:
        raise ValueError(f"No VH tiles found under: {folder}")
    return mapping


def scan_label_map(folder: Path, scene_name: str, strict: bool, required: bool) -> dict[str, Path]:
    if not folder.is_dir():
        if required:
            raise FileNotFoundError(f"Missing directory: {folder}")
        return {}
    mapping: dict[str, Path] = {}
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() != ".png":
            continue
        match = LABEL_PATTERN.match(path.name)
        if not match:
            if strict:
                raise ValueError(f"Unexpected label file name: {path}")
            continue
        if strict and match.group("prefix") != scene_name:
            raise ValueError(f"Label file prefix mismatch: {path.name} vs scene {scene_name}")
        mapping[match.group("tile")] = path
    if required and not mapping:
        raise ValueError(f"No label tiles found under: {folder}")
    return mapping


def build_observations(split_dir: Path, subset_name: str, strict: bool) -> list[dict[str, object]]:
    observations: list[dict[str, object]] = []
    if not split_dir.is_dir():
        return observations
    for scene_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        region, timestamp = parse_scene_dir(scene_dir)
        tiles_dir = scene_dir / "tiles"
        scene_name = scene_dir.name
        vh_map = scan_vh_map(tiles_dir / "vh", scene_name, strict)
        water_map = scan_label_map(tiles_dir / "water_body_label", scene_name, strict, required=True)
        flood_map = scan_label_map(tiles_dir / "flood_label", scene_name, strict, required=False)

        water_keys = sorted(water_map)
        vh_keys = sorted(vh_map)
        if vh_keys != water_keys:
            raise ValueError(f"VH/water tile mismatch under: {scene_dir}")
        if flood_map and sorted(flood_map) != water_keys:
            raise ValueError(f"Water/flood tile mismatch under: {scene_dir}")

        for tile_id in water_keys:
            observations.append(
                {
                    "subset": subset_name,
                    "region": region,
                    "timestamp": timestamp,
                    "tile_id": tile_id,
                    "vh_path": vh_map[tile_id],
                    "water_path": water_map[tile_id],
                    "flood_path": flood_map.get(tile_id),
                }
            )
    return observations


def resolve_split_regions(args: argparse.Namespace, train_observations: list[dict[str, object]]) -> tuple[list[str], list[str]]:
    region_counts = Counter(str(item["region"]) for item in train_observations)
    if not region_counts:
        raise ValueError("No training observations found under source root")

    explicit_val = parse_region_list(args.val_regions)
    explicit_train = parse_region_list(args.train_regions)
    all_regions = sorted(region_counts)

    if explicit_val:
        val_regions = explicit_val
    else:
        val_regions = [min(all_regions, key=lambda region: (region_counts[region], region))]

    if explicit_train:
        train_regions = explicit_train
    else:
        train_regions = [region for region in all_regions if region not in set(val_regions)]

    unknown = sorted((set(train_regions) | set(val_regions)) - set(all_regions))
    if unknown:
        raise ValueError(f"Unknown train/val regions: {unknown}")
    if set(train_regions) & set(val_regions):
        raise ValueError("train-regions and val-regions must be disjoint")
    if not train_regions or not val_regions:
        raise ValueError("Resolved train/val regions must both be non-empty")
    return sorted(train_regions), sorted(val_regions)


def group_observations(observations: list[dict[str, object]]) -> dict[tuple[str, str, str], list[dict[str, object]]]:
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for item in observations:
        key = (str(item["subset"]), str(item["region"]), str(item["tile_id"]))
        groups[key].append(item)
    for items in groups.values():
        items.sort(key=lambda item: str(item["timestamp"]))
    return groups


def load_binary_mask(path: Path) -> np.ndarray:
    """读取 RGB/灰度 PNG 掩码，并统一转为 bool 二值图。"""
    arr = np.array(Image.open(path).convert("L"), dtype=np.uint8)
    return arr > 0


def load_union_mask(observation: dict[str, object]) -> np.ndarray:
    """按 ETCI 口径构建单时相水体并集掩码：water_body OR flood。"""
    water_mask = load_binary_mask(Path(observation["water_path"]))
    flood_path = observation["flood_path"]
    if flood_path is None:
        return water_mask
    flood_mask = load_binary_mask(Path(flood_path))
    return water_mask | flood_mask


def compute_union_ratio(observation: dict[str, object]) -> float:
    ratio = observation.get("union_ratio")
    if ratio is not None:
        return float(ratio)
    union_mask = load_union_mask(observation)
    ratio = float(union_mask.mean())
    observation["union_ratio"] = ratio
    return ratio


def compute_white_ratio(image_path: Path) -> float:
    """统计 VH PNG 中纯白像素占比；RGB 必须三个通道都为 255。"""
    arr = np.array(Image.open(image_path), dtype=np.uint8)
    if arr.ndim == 2:
        white_mask = arr == 255
    elif arr.ndim == 3:
        white_mask = np.all(arr == 255, axis=-1)
    else:
        raise ValueError(f"Unsupported image ndim for white-ratio check: {image_path} -> {arr.ndim}")
    return float(white_mask.mean())


def get_white_ratio(observation: dict[str, object]) -> float:
    ratio = observation.get("white_ratio")
    if ratio is not None:
        return float(ratio)
    ratio = compute_white_ratio(Path(observation["vh_path"]))
    observation["white_ratio"] = ratio
    return ratio


def build_priority_pair(observations: list[dict[str, object]], threshold: float) -> tuple[int, int] | None:
    best_pair: tuple[int, int] | None = None
    best_increase = float("-inf")
    for idx_late in range(1, len(observations)):
        late_ratio = compute_union_ratio(observations[idx_late])
        for idx_early in range(idx_late):
            early_ratio = compute_union_ratio(observations[idx_early])
            increase = late_ratio - early_ratio
            if increase < threshold:
                continue
            if increase > best_increase:
                best_increase = increase
                best_pair = (idx_early, idx_late)
    return best_pair


def build_pair_specs(observations: list[dict[str, object]], policy: str, threshold: float) -> list[tuple[int, int, str]]:
    if len(observations) < 2:
        return []

    pairs: list[tuple[int, int, str]] = []
    seen: set[tuple[int, int]] = set()

    if policy in {"adjacent", "hybrid"}:
        for idx in range(len(observations) - 1):
            spec = (idx, idx + 1)
            if spec in seen:
                continue
            seen.add(spec)
            pairs.append((idx, idx + 1, "adjacent"))

    if policy in {"flood-priority", "hybrid"}:
        priority_pair = build_priority_pair(observations, threshold)
        if priority_pair is not None and priority_pair not in seen:
            seen.add(priority_pair)
            pairs.append((priority_pair[0], priority_pair[1], "flood-priority"))

    return pairs


def prepare_dirs(out_root: Path, dry_run: bool) -> None:
    for split in ("train", "val"):
        for subdir in ("A", "B", "label"):
            if dry_run:
                continue
            (out_root / split / subdir).mkdir(parents=True, exist_ok=True)


def clean_output_root(out_root: Path, overwrite: bool, dry_run: bool) -> None:
    if not out_root.exists() or not overwrite or dry_run:
        return
    shutil.rmtree(out_root)


def write_one(src: Path, dst: Path, link_mode: str, overwrite: bool, dry_run: bool) -> None:
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        if not dry_run:
            dst.unlink()
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if link_mode == "copy":
        shutil.copy2(src, dst)
    elif link_mode == "symlink":
        dst.symlink_to(src.resolve())
    elif link_mode == "hardlink":
        dst.hardlink_to(src)
    else:
        raise ValueError(f"Unsupported link mode: {link_mode}")


def save_label_png(mask: np.ndarray, dst: Path, overwrite: bool, dry_run: bool) -> None:
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        if not dry_run:
            dst.unlink()
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(dst)


def resolve_split(region: str, train_regions: set[str], val_regions: set[str]) -> str:
    if region in val_regions:
        return "val"
    if region in train_regions:
        return "train"
    raise ValueError(f"Cannot resolve split for region={region}")


def build_sample_name(region: str, tile_id: str, ts1: str, ts2: str) -> str:
    return f"{region}_{tile_id}__t1-{ts1}__t2-{ts2}.png"


def maybe_drop_low_change(change_ratio: float, mode: str, rng: random.Random) -> str | None:
    if change_ratio == 0.0:
        return "zero_change"
    if mode == "drop":
        return "low_change_drop"
    if mode == "drop80" and rng.random() < 0.8:
        return "low_change_drop80"
    return None


def write_pair_sample(
    out_root: Path,
    split: str,
    obs_t1: dict[str, object],
    obs_t2: dict[str, object],
    pair_kind: str,
    args: argparse.Namespace,
    rng: random.Random,
) -> tuple[bool, dict[str, object]]:
    mask_t1 = load_union_mask(obs_t1)
    mask_t2 = load_union_mask(obs_t2)
    change_mask = (~mask_t1) & mask_t2
    change_ratio = float(change_mask.mean())
    white_ratio_t1 = get_white_ratio(obs_t1)
    white_ratio_t2 = get_white_ratio(obs_t2)
    white_ratio_diff = abs(white_ratio_t1 - white_ratio_t2)

    record = {
        "subset": str(obs_t1["subset"]),
        "split": split,
        "region": str(obs_t1["region"]),
        "tile_id": str(obs_t1["tile_id"]),
        "timestamp_t1": str(obs_t1["timestamp"]),
        "timestamp_t2": str(obs_t2["timestamp"]),
        "pair_kind": pair_kind,
        "t1_union_ratio": round(compute_union_ratio(obs_t1), 8),
        "t2_union_ratio": round(compute_union_ratio(obs_t2), 8),
        "white_ratio_t1": round(white_ratio_t1, 8),
        "white_ratio_t2": round(white_ratio_t2, 8),
        "white_ratio_diff": round(white_ratio_diff, 8),
        "change_ratio": round(change_ratio, 8),
    }

    if white_ratio_diff > args.max_white_ratio_diff:
        record["drop_reason"] = "white_ratio_mismatch"
        return False, record
    if white_ratio_t1 > args.max_white_ratio or white_ratio_t2 > args.max_white_ratio:
        record["drop_reason"] = "white_ratio_too_high"
        return False, record
    if change_ratio < args.min_change_ratio:
        drop_reason = maybe_drop_low_change(change_ratio, args.drop_zero_change_mode, rng)
        if drop_reason is not None:
            record["drop_reason"] = drop_reason
            return False, record

    sample_name = build_sample_name(
        region=str(obs_t1["region"]),
        tile_id=str(obs_t1["tile_id"]),
        ts1=str(obs_t1["timestamp"]),
        ts2=str(obs_t2["timestamp"]),
    )
    write_one(Path(obs_t1["vh_path"]), out_root / split / "A" / sample_name, args.link_mode, args.overwrite, args.dry_run)
    write_one(Path(obs_t2["vh_path"]), out_root / split / "B" / sample_name, args.link_mode, args.overwrite, args.dry_run)
    save_label_png(change_mask, out_root / split / "label" / sample_name, args.overwrite, args.dry_run)

    record["sample_name"] = sample_name
    record["label_positive_pixels"] = int(change_mask.sum())
    return True, record


def write_manifest(out_root: Path, manifest_rows: list[dict[str, object]], dry_run: bool) -> None:
    if dry_run or not manifest_rows:
        return
    fieldnames = [
        "sample_name",
        "subset",
        "split",
        "region",
        "tile_id",
        "timestamp_t1",
        "timestamp_t2",
        "pair_kind",
        "t1_union_ratio",
        "t2_union_ratio",
        "white_ratio_t1",
        "white_ratio_t2",
        "white_ratio_diff",
        "change_ratio",
        "label_positive_pixels",
    ]
    with (out_root / "manifest.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)


def write_report(
    out_root: Path,
    args: argparse.Namespace,
    train_regions: list[str],
    val_regions: list[str],
    observations: list[dict[str, object]],
    candidate_counter: Counter,
    kept_counter: Counter,
    dropped_counter: Counter,
    manifest_rows: list[dict[str, object]],
    drop_examples: list[dict[str, object]],
    dry_run: bool,
) -> None:
    region_scene_counts: dict[str, int] = Counter()
    region_tile_group_counts: dict[str, int] = Counter()
    unique_scenes: set[tuple[str, str]] = set()
    unique_tile_groups: set[tuple[str, str]] = set()
    for item in observations:
        if str(item["subset"]) == "train":
            region = str(item["region"])
            timestamp = str(item["timestamp"])
            tile_id = str(item["tile_id"])
            if (region, timestamp) not in unique_scenes:
                unique_scenes.add((region, timestamp))
                region_scene_counts[region] += 1
            if (region, tile_id) not in unique_tile_groups:
                unique_tile_groups.add((region, tile_id))
                region_tile_group_counts[region] += 1

    payload = {
        "src_root": str(args.src_root),
        "out_root": str(args.out_root),
        "seed": args.seed,
        "pairing_policy": args.pairing_policy,
        "train_regions": train_regions,
        "val_regions": val_regions,
        "link_mode": args.link_mode,
        "min_change_ratio": args.min_change_ratio,
        "drop_zero_change_mode": args.drop_zero_change_mode,
        "flood_increase_threshold": args.flood_increase_threshold,
        "max_white_ratio": args.max_white_ratio,
        "max_white_ratio_diff": args.max_white_ratio_diff,
        "counts": {
            "candidate_pairs": dict(candidate_counter),
            "kept_pairs": dict(kept_counter),
            "dropped_pairs": dict(dropped_counter),
            "manifest_rows": len(manifest_rows),
        },
        "train_region_scene_counts": dict(region_scene_counts),
        "train_region_tile_group_counts": dict(region_tile_group_counts),
        "examples": {
            "kept": [row["sample_name"] for row in manifest_rows[:10]],
            "dropped": drop_examples[:10],
        },
    }
    if dry_run:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return
    (out_root / "split_report.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_args(args)

    train_observations = build_observations(args.src_root / "train", "train", args.strict)
    if args.test_subsets.strip():
        print("[WARN] --test-subsets is deprecated and ignored; only src-root/train is used.")
    observations = train_observations
    if not train_observations:
        raise ValueError(f"No observations found under {args.src_root}")

    train_regions, val_regions = resolve_split_regions(args, train_observations)
    train_region_set = set(train_regions)
    val_region_set = set(val_regions)

    clean_output_root(args.out_root, args.overwrite, args.dry_run)
    prepare_dirs(args.out_root, args.dry_run)

    groups = group_observations(observations)
    rng = random.Random(args.seed)
    candidate_counter: Counter = Counter()
    kept_counter: Counter = Counter()
    dropped_counter: Counter = Counter()
    manifest_rows: list[dict[str, object]] = []
    drop_examples: list[dict[str, object]] = []

    for (_, region, _tile_id), items in sorted(groups.items()):
        pair_specs = build_pair_specs(items, args.pairing_policy, args.flood_increase_threshold)
        if not pair_specs:
            continue
        for idx_t1, idx_t2, pair_kind in pair_specs:
            obs_t1 = items[idx_t1]
            obs_t2 = items[idx_t2]
            split = resolve_split(region=str(region), train_regions=train_region_set, val_regions=val_region_set)
            candidate_counter[split] += 1
            kept, record = write_pair_sample(args.out_root, split, obs_t1, obs_t2, pair_kind, args, rng)
            if kept:
                manifest_rows.append(record)
                kept_counter[split] += 1
            else:
                reason = str(record["drop_reason"])
                dropped_counter[reason] += 1
                if len(drop_examples) < 20:
                    drop_examples.append(record)

    write_manifest(args.out_root, manifest_rows, args.dry_run)
    write_report(
        out_root=args.out_root,
        args=args,
        train_regions=train_regions,
        val_regions=val_regions,
        observations=observations,
        candidate_counter=candidate_counter,
        kept_counter=kept_counter,
        dropped_counter=dropped_counter,
        manifest_rows=manifest_rows,
        drop_examples=drop_examples,
        dry_run=args.dry_run,
    )

    print(
        "[DONE] kept=",
        len(manifest_rows),
        "candidate=",
        sum(candidate_counter.values()),
        "train=",
        kept_counter["train"],
        "val=",
        kept_counter["val"],
    )


if __name__ == "__main__":
    main()
