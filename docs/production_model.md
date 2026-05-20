# Production model — merged_yolov8n

The merged-dataset YOLOv8n run is the production model going forward. It supersedes the baseline (`baseline_yolov8n`), which is kept as a reference point only.

## How it was trained

- **Architecture:** YOLOv8n (3.0M params, 8.1 GFLOPs)
- **Dataset:** Merged Pakistani plate dataset v1 — Burhan Khan + ubaidp1049, perceptual-hash deduplicated, stratified 80/15/5 split. 1,003 train / 187 valid / 65 test images.
- **Hyperparameters:** identical to baseline — `imgsz=960`, `batch=16`, `epochs=100`, `patience=20`, `seed=42`, `optimizer=auto`.
- **Hardware:** Kaggle T4 GPU, training time ~37 minutes.
- **Best epoch:** auto-selected by Ultralytics' fitness function.

## Validation metrics at best.pt epoch

| Metric | Baseline | **Merged** | Δ |
|---|---:|---:|---:|
| Precision | 0.894 | **0.936** | +0.043 |
| Recall | 0.837 | **0.945** | **+0.107** |
| mAP@0.5 | 0.906 | **0.974** | **+0.068** |
| mAP@0.5:0.95 | 0.674 | **0.747** | **+0.073** |

Recall improvement is the headline. The baseline missed ~16% of plates; the merged model misses ~5.5%.

## Held-out test metrics

| Metric | Value |
|---|---:|
| mAP@0.5 | 0.9926 |
| mAP@0.5:0.95 | 0.7677 |
| Precision | 0.9848 |
| Recall | 0.9526 |

## Inference speed

- T4 GPU: 8.9 ms / image (≈112 FPS)
- CPU (estimated, imgsz=960): 3-5 FPS — adequate for gate-camera deployment.

## Weight retrieval

Weights are gitignored. Re-obtain via:
- Re-train: `notebooks/04_kaggle_merged_training.ipynb` on Kaggle, or
- Download the bundled artifact zip from GitHub Releases (added when v0.1 ships).

## Known limitations

- Validation/test sets were re-split during the merge, so direct test-set comparison to baseline is approximate.
- Only 5 ubaidp1049 images landed in test, so test metrics primarily measure performance on Burhan Khan's distribution.
- Real gate-camera footage (night, rain, motion blur, dirty plates) was not in either training dataset. Field testing on self-recorded footage is required before deployment.