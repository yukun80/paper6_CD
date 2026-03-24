# GEMINI.md - SAR Flood Change Detection Project

## 1. Project Overview
- **Objective:** Develop a SAR-based binary flood change detection algorithm and complete a research paper.
- **Core Algorithm:** `ChangeDINO`, a dual-branch architecture (CNN + frozen DINOv3 ViT-L/16 backbone) specifically adapted for SAR binary change detection.
- **Key Dataset:** `S1GFloods` (SAR gray-scale PNG dataset for binary flood change detection).
- **Secondary Assets:** `GF3_Henan`, `S1_Henan` (Used for large-scale scene inference and validation).

## 2. Technical Specifications & Norms
### 2.1 Core Workflow: ChangeDINO
- **Input:** Bi-temporal SAR images (T1, T2) and binary change masks.
- **Data Format:** Prepared via `ChangeDINO-main/scripts/prepare_s1gfloods_cd.py` into a standardized `A/B/label` structure.
- **Normalization:** Uses dataset-specific channel statistics stored in `channel_stats_s1gfloods_train.json`.
- **Primary Metric:** Binary Change Detection IoU and F1-score (as reported by `ChangeDINO` train/val scripts).

### 2.2 Abandoned/Deprecated Specifications (DO NOT USE)
- **12-Channel SAR Norms:** All multi-channel (12ch) polarimetric/temporal ordering specifications are deprecated.
- **Hierarchical Labels:** `floodness`/`flood_type` and tripartite classifications are replaced by binary change labels.
- **Metrics:** `pos_mIoU` (for two flood classes) is deprecated; focus solely on binary change detection metrics.
- **Frameworks:** `Open-CD`, `Panopticon`, and `exp_template` are kept for historical reference only and are NOT part of the current default workflow.

## 3. Current Progress
- **Algorithm Adaptation:** `ChangeDINO` core is fully adapted for SAR binary change detection, including grayscale-to-RGB tensor conversion and SAR-specific augmentations.
- **Data Engineering:** `S1GFloods` is processed and ready for training.
- **Inference Pipeline:** Tiled inference and stitching scripts for `S1_Henan` and `GF3_Henan` are implemented.

## 4. Active Workflow Commands
- **Prepare Data:** `python ChangeDINO-main/scripts/prepare_s1gfloods_cd.py --src-root datasets/S1GFloods --out-root datasets/S1GFloods_CD_DINO --seed 42`
- **Train:** `bash ChangeDINO-main/trainval_s1gfloods.sh`
- **Inference:** `python ChangeDINO-main/scripts/infer_s1_henan_tiles.py ...`

## 5. Immediate Next Steps
1. **Environment:** Ensure `kornia` is installed.
2. **Execution:** Execute full `ChangeDINO` training on `S1GFloods` and monitor loss/metrics.
3. **Paper Writing:** Initiate drafting in `paper_writing/` focusing on the DINOv3-driven SAR change detection approach.
