"""
Streamlit-friendly wrapper around the Phase 7 end-to-end pipeline.

Lets the app process an uploaded video, persist events to the DB, and report
progress incrementally so the UI stays responsive.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from src.ocr import PlateReader
from src.pipeline import GateEventLogger, PlateTracker
from src.app.storage import GateStore


@dataclass
class PipelineRunResult:
    annotated_video_path: Path
    total_frames: int
    events_logged: int
    elapsed_seconds: float
    fps: float


def annotate_frame(frame, tracks, trigger_zone, banner_event=None):
    """Draw trigger zone, detections, and optional event banner. Same logic
    as scripts/run_gate_pipeline.py, extracted here so the app reuses it."""
    overlay = frame.copy()
    tx1, ty1, tx2, ty2 = trigger_zone
    cv2.rectangle(overlay, (tx1, ty1), (tx2, ty2), (0, 200, 255), -1)
    frame = cv2.addWeighted(overlay, 0.15, frame, 0.85, 0)
    cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), (0, 200, 255), 2)
    cv2.putText(frame, "TRIGGER ZONE", (tx1 + 10, ty1 + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    for plate in tracks:
        x1, y1, x2, y2 = plate.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"ID:{plate.track_id} {plate.conf:.2f}"
        cv2.putText(frame, label, (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    if banner_event is not None:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 50), (0, 215, 255), -1)
        text = f"LOGGED  ID {banner_event['track_id']}  PLATE: {banner_event['plate_text'] or '<unread>'}"
        cv2.putText(frame, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

    return frame


def run_pipeline_on_video(
    video_path: Path,
    weights_path: Path,
    annotated_out_path: Path,
    store: GateStore,
    trigger_zone: tuple[int, int, int, int],
    conf_threshold: float = 0.70,
    imgsz: int = 960,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    source_label: str = "video",
) -> PipelineRunResult:
    """Run the full pipeline on one video. Persist events to the store as they
    fire. Write an annotated video alongside.

    progress_cb(current_frame, total_frames) is called once per frame so the
    Streamlit UI can update a progress bar.
    """
    tracker = PlateTracker(weights_path, conf_threshold=conf_threshold, imgsz=imgsz)
    ocr = PlateReader(gpu=False)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Resolve trigger_zone: fractions [0,1] -> absolute pixels.
    # Must happen BEFORE constructing GateEventLogger or its internal copy
    # stays fractional and _in_trigger() never matches.
    if max(trigger_zone) <= 1.0:
        tx1, ty1, tx2, ty2 = trigger_zone
        trigger_zone = (int(tx1 * width), int(ty1 * height),
                        int(tx2 * width), int(ty2 * height))
        print(f"Trigger zone resolved to absolute pixels: {trigger_zone}")

    logger_obj = GateEventLogger(ocr=ocr, trigger_zone=trigger_zone)

    # Resolve trigger_zone: if all 4 values are in [0, 1], treat as fractions
    # of frame size. Otherwise treat as absolute pixel coordinates. This lets
    # the same default work across any video resolution and orientation.
    if max(trigger_zone) <= 1.0:
        tx1, ty1, tx2, ty2 = trigger_zone
        trigger_zone = (int(tx1 * width), int(ty1 * height),
                        int(tx2 * width), int(ty2 * height))
        print(f"Trigger zone resolved to absolute pixels: {trigger_zone}")

    annotated_out_path.parent.mkdir(parents=True, exist_ok=True)
    # Try H.264 (browser-safe). If OpenCV's build lacks H.264 support, fall back
    # to mp4v — file is valid but most browsers won't preview it inline.
    h264_path = annotated_out_path.with_suffix(".h264.mp4")
    fourcc_h264 = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(h264_path), fourcc_h264, fps, (width, height))
    if writer.isOpened():
        annotated_out_path = h264_path
    else:
        # Fallback to mp4v
        writer = cv2.VideoWriter(str(annotated_out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                  fps, (width, height))

    track_stream = tracker.track_stream(video_path)
    start = time.perf_counter()
    frame_idx = 0
    last_event_dict = None
    last_event_frame = -1
    BANNER_FRAMES = int(fps)  # show banner for ~1 second

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        try:
            tracks = next(track_stream)
        except StopIteration:
            tracks = []

        new_events = logger_obj.process_frame(frame_idx, frame, tracks)
        for ev in new_events:
            # Persist event first so we get the event_id
            event_id = store.insert_event(
                timestamp_iso=ev.timestamp_iso,
                frame_idx=ev.frame_idx,
                track_id=ev.track_id,
                ocr_plate_text=ev.plate_text,
                ocr_confidence=ev.plate_confidence,
                detect_confidence=ev.detect_confidence,
                bbox_str=",".join(str(x) for x in ev.bbox),
                source=source_label,
            )
            # Save the cropped plate image keyed by event_id so the UI can render it
            crop_path = annotated_out_path.parent / "crops" / f"event_{event_id}.jpg"
            crop_path.parent.mkdir(parents=True, exist_ok=True)
            x1, y1, x2, y2 = ev.bbox
            crop_bgr = frame[max(0, y1):y2, max(0, x1):x2]
            if crop_bgr.size > 0:
                cv2.imwrite(str(crop_path), crop_bgr)

            last_event_dict = {"track_id": ev.track_id, "plate_text": ev.plate_text}
            last_event_frame = frame_idx

        banner = (
            last_event_dict
            if last_event_dict and frame_idx - last_event_frame < BANNER_FRAMES
            else None
        )
        annotated = annotate_frame(frame, tracks, trigger_zone, banner_event=banner)
        writer.write(annotated)

        frame_idx += 1
        if progress_cb is not None:
            progress_cb(frame_idx, total_frames)

    cap.release()
    writer.release()
    elapsed = time.perf_counter() - start

    return PipelineRunResult(
        annotated_video_path=annotated_out_path,
        total_frames=frame_idx,
        events_logged=len(logger_obj.events),
        elapsed_seconds=elapsed,
        fps=frame_idx / elapsed if elapsed > 0 else 0.0,
    )