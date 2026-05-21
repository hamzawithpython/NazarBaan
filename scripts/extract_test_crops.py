"""
Extract plate crops from the held-out test set for OCR labeling and benchmarking.

For each test image: run the production detector, crop every detection at
conf >= 0.5 (more permissive than the deployment threshold so we get edge
cases too), and save the crops with deterministic filenames.

Also writes a CSV scaffold (data/processed/test_crops/_ground_truth.csv)
with one row per crop and an empty `text` column. I fill that in manually.

Run: python scripts/extract_test_crops.py
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = PROJECT_ROOT / "models" / "merged_yolov8n" / "train" / "weights" / "best.pt"
TEST_IMGS = PROJECT_ROOT / "data" / "processed" / "merged_v1" / "test" / "images"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "test_crops"
CSV_PATH = OUT_DIR / "_ground_truth.csv"

# Use a lower threshold than deployment so I capture edge cases too
EXTRACT_CONF = 0.50
IMG_SIZE = 960


def main() -> None:
    if OUT_DIR.exists():
        print(f"Removing previous crops at {OUT_DIR}")
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(WEIGHTS))
    test_images = sorted(TEST_IMGS.glob("*.jpg"))
    print(f"Cropping plates from {len(test_images)} test images at conf >= {EXTRACT_CONF}...\n")

    rows: list[dict] = []
    crop_index = 0

    for img_path in tqdm(test_images):
        result = model(img_path, imgsz=IMG_SIZE, conf=EXTRACT_CONF, verbose=False)[0]
        img = Image.open(img_path)

        for i in range(len(result.boxes)):
            x1, y1, x2, y2 = map(int, result.boxes.xyxy[i].cpu().numpy().tolist())
            conf = float(result.boxes.conf[i])
            crop = img.crop((x1, y1, x2, y2))

            # Skip absurdly tiny crops — they're rare false positives or
            # downscaled artifacts; OCR can't read them anyway
            if crop.size[0] < 30 or crop.size[1] < 15:
                continue

            crop_name = f"{crop_index:03d}__{img_path.stem[:40]}.jpg"
            crop.save(OUT_DIR / crop_name)

            rows.append({
                "crop_id": f"{crop_index:03d}",
                "source_image": img_path.name,
                "source_dataset": img_path.stem.split("__")[0],
                "detect_conf": round(conf, 4),
                "crop_w": crop.size[0],
                "crop_h": crop.size[1],
                "text": "",  # fill in manually
            })
            crop_index += 1

    # Write CSV with BOM-free utf-8 (we keep that habit from the .env lesson)
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCropped {crop_index} plates into {OUT_DIR}")
    print(f"Ground-truth CSV scaffold: {CSV_PATH}")
    print(f"\nNext step: open the CSV and fill the 'text' column for the first 30 rows.")


if __name__ == "__main__":
    main()