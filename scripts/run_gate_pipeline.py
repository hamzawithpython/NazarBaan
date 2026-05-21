"""
Run the NazarBaan end-to-end gate pipeline on a video.

Inputs:
  - YOLO weights (production model from Phase 4)
  - A video file (synthetic for now; real gate footage later)

Outputs:
  - data/processed/gate_events.csv               structured event log
  - data/processed/gate_pipeline_annotated.mp4   video with bboxes, IDs,
                                                  trigger zone, and event flashes

Run: python scripts/run_gate_pipeline.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
from tqdm import tqdm

# Make src importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ocr import PlateReader
from src.pipeline import PlateTracker, GateEventLogger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = PROJECT_ROOT / "models" / "merged_yolov8n" / "train" / "weights" / "best.pt"
VIDEO_IN = PROJECT_ROOT / "data" / "processed" / "synthetic_gate_video.mp4"
VIDEO_OUT = PROJECT_ROOT / "data" / "processed" / "gate_pipeline_annotated.mp4"
CSV_OUT = PROJECT_ROOT / "data" / "processed" / "gate_events.csv"

# Trigger zone — center band of the 1280x720 frame.
# Why this shape: in real gate footage the boom barrier sits roughly center-screen
# at slight depression. For the synthetic video the letterboxed test images also
# land in the center band, so this catches the plate when it's "at the gate."
TRIGGER_ZONE = (320, 200, 960, 540)   # x1, y1, x2, y2


def annotate_frame(frame, tracks, trigger_zone, new_event=None):
    """Draw bounding boxes, track IDs, the trigger zone, and any new event banner."""
    # Trigger zone — semi-transparent overlay
    overlay = frame.copy()
    tx1, ty1, tx2, ty2 = trigger_zone
    cv2.rectangle(overlay, (tx1, ty1), (tx2, ty2), (0, 200, 255), -1)
    frame = cv2.addWeighted(overlay, 0.15, frame, 0.85, 0)
    cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), (0, 200, 255), 2)
    cv2.putText(frame, "TRIGGER ZONE", (tx1 + 10, ty1 + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    # Detections — green box, white ID label
    for plate in tracks:
        x1, y1, x2, y2 = plate.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"ID:{plate.track_id} {plate.conf:.2f}"
        cv2.putText(frame, label, (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    # New-event banner — yellow stripe at top with the plate text
    if new_event is not None:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 50), (0, 215, 255), -1)
        text = f"LOGGED  ID {new_event.track_id}  PLATE: {new_event.plate_text or '<unread>'}"
        cv2.putText(frame, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 0, 0), 2)

    return frame


def main() -> None:
    print(f"Weights: {WEIGHTS}")
    print(f"Video in:  {VIDEO_IN}")
    print(f"Video out: {VIDEO_OUT}")
    print(f"Trigger zone (x1,y1,x2,y2): {TRIGGER_ZONE}\n")

    # Build the three components
    tracker = PlateTracker(WEIGHTS, conf_threshold=0.70, imgsz=960)
    ocr = PlateReader(gpu=False)
    logger = GateEventLogger(ocr=ocr, trigger_zone=TRIGGER_ZONE)

    # Open the source video once with OpenCV — we need its frames for annotation.
    # Ultralytics' tracker also reads it (separate read pass), but disk caching
    # makes the duplicate read cheap.
    cap = cv2.VideoCapture(str(VIDEO_IN))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {total_frames} frames @ {fps} fps, {width}x{height}\n")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(VIDEO_OUT), fourcc, fps, (width, height))

    track_stream = tracker.track_stream(VIDEO_IN)
    start_t = time.perf_counter()

    pbar = tqdm(total=total_frames, desc="Processing")
    frame_idx = 0
    last_event = None
    last_event_frame = -1
    EVENT_BANNER_FRAMES = 30   # show "LOGGED" banner for ~1 second after each event

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        try:
            tracks = next(track_stream)
        except StopIteration:
            tracks = []

        new_events = logger.process_frame(frame_idx, frame, tracks)
        if new_events:
            last_event = new_events[-1]
            last_event_frame = frame_idx

        # Show banner for ~1 second after the event
        banner = last_event if (last_event and frame_idx - last_event_frame < EVENT_BANNER_FRAMES) else None
        annotated = annotate_frame(frame, tracks, TRIGGER_ZONE, new_event=banner)
        writer.write(annotated)

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    writer.release()
    elapsed = time.perf_counter() - start_t

    logger.save_csv(CSV_OUT)

    print(f"\nProcessed {frame_idx} frames in {elapsed:.1f}s  ({frame_idx/elapsed:.2f} FPS)")
    print(f"Events logged: {len(logger.events)}")
    print(f"\nAnnotated video: {VIDEO_OUT}")
    print(f"Event log CSV:   {CSV_OUT}")

    # First 5 events to console for a quick sanity look
    if logger.events:
        print(f"\nFirst {min(5, len(logger.events))} events:")
        for e in logger.events[:5]:
            print(f"  frame={e.frame_idx:>5}  track={e.track_id:>3}  "
                  f"plate={e.plate_text!r:<14}  conf={e.plate_confidence:.2f}")


if __name__ == "__main__":
    main()