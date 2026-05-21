"""
Build a synthetic gate-camera video by stitching held-out test images together.

Each test image is held for ~1 second (30 frames at 30 fps), then a short
black-frame transition simulates the gap between vehicles. The output is
intentionally crude — its job is to exercise the end-to-end pipeline, not to
fool any tracker.

Output: data/processed/synthetic_gate_video.mp4
Run: python scripts/synthesize_test_video.py
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_IMGS = PROJECT_ROOT / "data" / "processed" / "merged_v1" / "test" / "images"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "synthetic_gate_video.mp4"

# Video parameters
FPS = 30
FRAMES_PER_IMAGE = 30      # ~1 second hold per "vehicle"
TRANSITION_FRAMES = 5      # short black gap between vehicles
FRAME_W, FRAME_H = 1280, 720  # standard 720p gate-camera resolution
RANDOM_SEED = 42


def fit_image_to_frame(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Letterbox an arbitrarily-sized image into a fixed-resolution frame
    on a black background, preserving aspect ratio."""
    src_h, src_w = img.shape[:2]
    scale = min(target_w / src_w, target_h / src_h)
    new_w, new_h = int(src_w * scale), int(src_h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y_off = (target_h - new_h) // 2
    x_off = (target_w - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def main() -> None:
    random.seed(RANDOM_SEED)
    images = sorted(TEST_IMGS.glob("*.jpg"))
    random.shuffle(images)
    print(f"Stitching {len(images)} test images into {OUT_PATH.name}...")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUT_PATH), fourcc, FPS, (FRAME_W, FRAME_H))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter at {OUT_PATH}")

    transition = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)

    for img_path in tqdm(images):
        # PIL load + RGB->BGR for OpenCV
        pil = Image.open(img_path).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        frame = fit_image_to_frame(img_bgr, FRAME_W, FRAME_H)

        for _ in range(FRAMES_PER_IMAGE):
            writer.write(frame)
        for _ in range(TRANSITION_FRAMES):
            writer.write(transition)

    writer.release()

    duration_sec = len(images) * (FRAMES_PER_IMAGE + TRANSITION_FRAMES) / FPS
    print(f"\nWrote {OUT_PATH}")
    print(f"  Resolution: {FRAME_W}x{FRAME_H}  @ {FPS} fps")
    print(f"  Duration:   ~{duration_sec:.1f} seconds")
    print(f"  File size:  {OUT_PATH.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()