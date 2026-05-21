"""
PlateTracker — wraps YOLO + Ultralytics built-in tracker (ByteTrack) to
emit (track_id, bbox, confidence) tuples per detection per frame.

Single responsibility: detect plates and assign persistent IDs across frames.
No OCR, no logging, no state — that's the event_logger's job.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from ultralytics import YOLO


@dataclass(frozen=True)
class TrackedPlate:
    """Single tracked plate detection in one frame."""
    track_id: int          # persistent ID across frames (-1 if untracked)
    bbox: tuple[int, int, int, int]   # x1, y1, x2, y2 in pixels
    conf: float            # detection confidence


class PlateTracker:
    """Wraps a YOLO model with ByteTrack for frame-by-frame plate tracking.

    Usage:
        tracker = PlateTracker("models/.../best.pt")
        for tracks in tracker.track_stream(video_path):
            for plate in tracks:
                print(plate.track_id, plate.bbox)
    """

    def __init__(
        self,
        weights_path: str | Path,
        conf_threshold: float = 0.70,
        imgsz: int = 960,
    ) -> None:
        self._model = YOLO(str(weights_path))
        self._conf = conf_threshold
        self._imgsz = imgsz

    def track_stream(self, video_path: str | Path) -> Iterable[list[TrackedPlate]]:
        """Yield a list of TrackedPlate objects for each frame in the video.

        Uses Ultralytics' built-in ByteTrack via model.track(persist=True).
        Frames with no detections yield an empty list (keeps frame indexing aligned)."""
        # stream=True yields one Results object per frame instead of buffering all
        results = self._model.track(
            source=str(video_path),
            conf=self._conf,
            imgsz=self._imgsz,
            tracker="bytetrack.yaml",   # ships with ultralytics; no extra files
            persist=True,
            stream=True,
            verbose=False,
        )

        for frame_result in results:
            boxes = frame_result.boxes
            if boxes is None or len(boxes) == 0:
                yield []
                continue

            # track_id may be None for the first-frame matches before track init
            ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [-1] * len(boxes)
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()

            yield [
                TrackedPlate(
                    track_id=int(ids[i]),
                    bbox=tuple(map(int, xyxy[i].tolist())),
                    conf=float(confs[i]),
                )
                for i in range(len(boxes))
            ]