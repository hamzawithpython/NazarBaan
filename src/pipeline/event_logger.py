"""
GateEventLogger — turns a stream of TrackedPlate observations into
"vehicle entered the gate at X with plate Y" events, deduplicated per track.

Core idea:
  * A "trigger zone" is a polygon (or simple rectangle) in frame coordinates.
  * A track is logged exactly once, on its FIRST frame whose plate center
    falls inside the trigger zone. Subsequent frames of the same track in
    the zone don't re-log.
  * On that first triggering frame, OCR is run on the cropped plate.
  * Tracks that never enter the zone are never logged.

This is the deduplication layer that makes the pipeline a product instead
of a stream of duplicate detections.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from src.pipeline.tracker import TrackedPlate


@dataclass
class GateEvent:
    """One logged vehicle entry — one row in the gate log."""
    timestamp_iso: str    # ISO 8601, second precision (real wall-clock for live runs)
    frame_idx: int        # frame number in the video / stream
    track_id: int         # YOLO+ByteTrack ID; useful for debugging
    plate_text: str       # OCR output, post-processed; "" if OCR failed
    plate_confidence: float
    detect_confidence: float
    bbox: tuple[int, int, int, int]


class GateEventLogger:
    """Stateful: tracks which IDs have already been logged, holds the trigger
    zone, runs OCR exactly once per qualifying track.

    Usage:
        ocr = PlateReader()
        logger = GateEventLogger(ocr=ocr, trigger_zone=(0, 360, 1280, 720))
        for frame_idx, (frame_img, tracks) in enumerate(zip(frames, track_stream)):
            new_events = logger.process_frame(frame_idx, frame_img, tracks)
            for event in new_events:
                print(event)
        logger.save_csv("events.csv")
    """

    def __init__(
        self,
        ocr,                               # any object with .read(img) -> (text, conf)
        trigger_zone: tuple[int, int, int, int],   # x1, y1, x2, y2 of the gate trigger
    ) -> None:
        self._ocr = ocr
        self._trigger = trigger_zone
        self._logged_track_ids: set[int] = set()
        self._events: list[GateEvent] = []

    @property
    def events(self) -> list[GateEvent]:
        return list(self._events)

    @property
    def trigger_zone(self) -> tuple[int, int, int, int]:
        return self._trigger

    def _in_trigger(self, bbox: tuple[int, int, int, int]) -> bool:
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        tx1, ty1, tx2, ty2 = self._trigger
        return tx1 <= cx <= tx2 and ty1 <= cy <= ty2

    def process_frame(
        self,
        frame_idx: int,
        frame_img: np.ndarray,    # BGR (OpenCV) is fine; we'll convert
        tracks: list[TrackedPlate],
    ) -> list[GateEvent]:
        """Process one frame. Return any NEW events triggered this frame
        (typically zero or one)."""
        new_events: list[GateEvent] = []

        for plate in tracks:
            # Skip untracked detections (-1) — can't dedupe without an ID
            if plate.track_id < 0:
                continue
            # Already logged this car's entry — don't log again
            if plate.track_id in self._logged_track_ids:
                continue
            # Not yet in the trigger zone — wait
            if not self._in_trigger(plate.bbox):
                continue

            # First-time entry of this track into the trigger zone — log it
            crop = self._crop(frame_img, plate.bbox)
            plate_text, plate_conf = self._ocr.read(crop)

            event = GateEvent(
                timestamp_iso=datetime.now().isoformat(timespec="seconds"),
                frame_idx=frame_idx,
                track_id=plate.track_id,
                plate_text=plate_text,
                plate_confidence=round(plate_conf, 4),
                detect_confidence=round(plate.conf, 4),
                bbox=plate.bbox,
            )
            self._events.append(event)
            new_events.append(event)
            self._logged_track_ids.add(plate.track_id)

        return new_events

    @staticmethod
    def _crop(frame_bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> Image.Image:
        """Crop a plate from a BGR frame and return as a PIL RGB image
        suitable for PlateReader.read()."""
        x1, y1, x2, y2 = bbox
        # Guard against out-of-bounds (rare, but happens with edge detections)
        h, w = frame_bgr.shape[:2]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))
        crop_bgr = frame_bgr[y1:y2, x1:x2]
        crop_rgb = crop_bgr[:, :, ::-1]   # BGR -> RGB
        return Image.fromarray(crop_rgb)

    def save_csv(self, path: str | Path) -> None:
        """Dump all events to CSV. Bbox is serialized as a single string."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        rows = []
        for e in self._events:
            row = asdict(e)
            row["bbox"] = f"{e.bbox[0]},{e.bbox[1]},{e.bbox[2]},{e.bbox[3]}"
            rows.append(row)

        # utf-8 (no BOM) — Excel and pandas both read it fine
        with open(path, "w", newline="", encoding="utf-8") as f:
            if not rows:
                f.write("timestamp_iso,frame_idx,track_id,plate_text,plate_confidence,detect_confidence,bbox\n")
                return
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)