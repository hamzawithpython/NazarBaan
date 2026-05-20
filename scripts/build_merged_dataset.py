"""
Build the merged training dataset for NazarBaan Phase 4.

Combines:
  - data/raw/Pk-Number-Plates-1/ (Burhan Khan, 1208 images, pre-split)
  - data/raw/ubaidp1049_raw/     (ubaidp1049, 76 images, flat)

Steps:
  1. Pool all (image, label) pairs into one list with a 'source' tag.
  2. Perceptual-hash every image; drop near-duplicates (Hamming distance <= 5).
  3. Stratify by source, then split 80/15/5 train/valid/test.
  4. Write the merged dataset to data/processed/merged_v1/ with data.yaml.

Run: python scripts/build_merged_dataset.py
"""

from __future__ import annotations

import random
import shutil
from collections import defaultdict
from pathlib import Path

import imagehash
import yaml
from PIL import Image
from tqdm import tqdm

# Reproducible split — same seed gives same train/valid/test every run
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BURHAN_ROOT = PROJECT_ROOT / "data" / "raw" / "Pk-Number-Plates-1"
UBAID_IMGS = PROJECT_ROOT / "data" / "raw" / "ubaidp1049_raw" / "images" / "images"
UBAID_LBLS = PROJECT_ROOT / "data" / "raw" / "ubaidp1049_raw" / "labels" / "labels"
OUT_ROOT = PROJECT_ROOT / "data" / "processed" / "merged_v1"

# Hamming-distance threshold: <= this many bit flips between hashes = "same image"
DUP_THRESHOLD = 5
SPLITS = {"train": 0.80, "valid": 0.15, "test": 0.05}


def collect_pairs() -> list[dict]:
    """Walk both datasets and return one record per (image, label) pair."""
    records: list[dict] = []

    # Burhan Khan — already pre-split, but I'm re-pooling everything to re-split fresh
    for split in ["train", "valid", "test"]:
        img_dir = BURHAN_ROOT / split / "images"
        lbl_dir = BURHAN_ROOT / split / "labels"
        for img in img_dir.glob("*.jpg"):
            label = lbl_dir / f"{img.stem}.txt"
            if label.exists():
                records.append({"img": img, "lbl": label, "source": "burhan_khan"})

    # ubaidp1049 — flat layout
    for img in UBAID_IMGS.glob("*.jpg"):
        label = UBAID_LBLS / f"{img.stem}.txt"
        if label.exists():
            records.append({"img": img, "lbl": label, "source": "ubaidp1049"})

    return records


def dedupe(records: list[dict]) -> list[dict]:
    """Drop perceptual-hash duplicates. Keeps the first occurrence."""
    print(f"\nHashing {len(records)} images for deduplication...")
    seen_hashes: list[imagehash.ImageHash] = []
    kept: list[dict] = []
    dropped_by_source: dict[str, int] = defaultdict(int)

    for rec in tqdm(records):
        try:
            h = imagehash.phash(Image.open(rec["img"]))
        except Exception as e:
            print(f"  Skipping unreadable image {rec['img'].name}: {e}")
            continue

        is_dup = any((h - prev) <= DUP_THRESHOLD for prev in seen_hashes)
        if is_dup:
            dropped_by_source[rec["source"]] += 1
        else:
            seen_hashes.append(h)
            kept.append(rec)

    print(f"\nDeduplication summary:")
    print(f"  kept:    {len(kept)}")
    print(f"  dropped: {sum(dropped_by_source.values())}  ({dict(dropped_by_source)})")
    return kept


def stratified_split(records: list[dict]) -> dict[str, list[dict]]:
    """Group by source, shuffle within group, then assign to splits.
    Ensures both burhan_khan and ubaidp1049 are represented in every split."""
    by_source: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_source[r["source"]].append(r)

    splits: dict[str, list[dict]] = {k: [] for k in SPLITS}
    for source, recs in by_source.items():
        random.shuffle(recs)
        n = len(recs)
        n_train = int(n * SPLITS["train"])
        n_valid = int(n * SPLITS["valid"])
        # Test gets the remainder so we never lose images to rounding
        splits["train"].extend(recs[:n_train])
        splits["valid"].extend(recs[n_train:n_train + n_valid])
        splits["test"].extend(recs[n_train + n_valid:])
        print(f"  {source:>14}: train={n_train}  valid={n_valid}  test={n - n_train - n_valid}")

    # Reshuffle each split so source order is random (not all burhan first then ubaid)
    for split in splits.values():
        random.shuffle(split)
    return splits


def write_split(split_name: str, records: list[dict]) -> None:
    """Copy files into data/processed/merged_v1/<split>/images and /labels."""
    img_out = OUT_ROOT / split_name / "images"
    lbl_out = OUT_ROOT / split_name / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    for rec in records:
        # Prefix with source so two datasets can have same filename without collision
        new_name = f"{rec['source']}__{rec['img'].stem}"
        shutil.copy2(rec["img"], img_out / f"{new_name}{rec['img'].suffix}")
        shutil.copy2(rec["lbl"], lbl_out / f"{new_name}.txt")


def write_data_yaml() -> None:
    """Write the YOLO data.yaml. Uses relative paths so the dataset is portable
    between local machine and Kaggle."""
    cfg = {
        "path": ".",            # interpreted relative to wherever data.yaml lives
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 1,
        "names": ["Number-Plate"],
    }
    with open(OUT_ROOT / "data.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"\nWrote {OUT_ROOT / 'data.yaml'}")


def main() -> None:
    if OUT_ROOT.exists():
        print(f"Removing previous merged dataset at {OUT_ROOT}")
        shutil.rmtree(OUT_ROOT)

    print("Collecting (image, label) pairs from both source datasets...")
    records = collect_pairs()
    print(f"  total pairs found: {len(records)}")

    kept = dedupe(records)

    print(f"\nStratified split (seed={RANDOM_SEED}):")
    splits = stratified_split(kept)

    print(f"\nCopying files into {OUT_ROOT}...")
    for split_name, recs in splits.items():
        write_split(split_name, recs)
        print(f"  {split_name}: {len(recs)} images")

    write_data_yaml()
    print("\nDone.")


if __name__ == "__main__":
    main()