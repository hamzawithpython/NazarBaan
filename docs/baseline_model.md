# Baseline model artifacts

Trained on Kaggle T4 GPU. The actual `.pt` weight file (`models/baseline_yolov8n/train/weights/best.pt`, 6.3 MB) is gitignored to keep the repo light.

## How to obtain the weights

- Re-train via `notebooks/02_kaggle_baseline_training.ipynb` on Kaggle, or
- Download the bundled artifact zip from the project's GitHub Releases (added when v0.1 ships).

## What the bundle contains

- `train/weights/best.pt` — best-epoch checkpoint (loaded for inference)
- `train/weights/last.pt` — final-epoch checkpoint
- `train/results.csv` — per-epoch metrics
- `train/args.yaml` — exact training configuration used
- `test/` — held-out test-set evaluation outputs

## Reported metrics (test split, 69 images, 74 plate instances)

- mAP@0.5: **0.9915**
- mAP@0.5:0.95: **0.7824**
- Precision: **0.9625**
- Recall: **0.9459**
