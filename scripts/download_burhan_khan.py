"""
Download the Burhan Khan "Pk Number Plates" dataset from Roboflow Universe.

Anchor dataset for NazarBaan baseline training: ~1,678 Pakistani plate
images in YOLOv8 format. Output lands in data/raw/burhan_khan/.

Run: python scripts/download_burhan_khan.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from roboflow import Roboflow

# Project root is one level up from scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# Roboflow project coordinates — these are public and safe to hardcode
WORKSPACE = "burhan-khan"
PROJECT = "pk-number-plates"
VERSION = 1  # If a newer version exists, we'll bump this after inspection
FORMAT = "yolov8"


def main() -> None:
    # Load API key from .env; the key itself never lives in source
    load_dotenv(PROJECT_ROOT / ".env", encoding="utf-8-sig")
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        sys.exit("ROBOFLOW_API_KEY missing from .env — aborting.")

    # Roboflow SDK dumps into the current working dir, so we chdir first
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(RAW_DIR)

    print(f"Downloading {WORKSPACE}/{PROJECT} v{VERSION} into {RAW_DIR}")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(WORKSPACE).project(PROJECT)
    dataset = project.version(VERSION).download(FORMAT)

    print(f"\nDone. Dataset location: {dataset.location}")


if __name__ == "__main__":
    main()
