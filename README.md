# NazarBaan — Pakistani ANPR for Gated Communities

> *Nazarbaan (نظربان) — Urdu for "watchman, one who keeps watch."*

An automated number-plate recognition system designed for Pakistani housing societies. Replaces pen-and-register gate logs with a real-time computer vision pipeline that detects, reads, and logs every vehicle entering and exiting — with a driver-facing display, entry/exit pairing, and blacklist matching.

---

## The Problem

Pakistani gated communities log vehicles manually: a guard with a register, a pen, and a queue of impatient drivers. The result:

- **Long queues** at peak hours.
- **Illegible records** — handwritten plates are unsearchable.
- **No real-time check** against stolen-vehicle or blacklist databases.
- **No entry/exit pairing** — once a vehicle is logged in, there's no audit trail confirming it left.
- **Vulnerable to social engineering** — *"Bhai I'm visiting Block C"* and the gate opens.

When a vehicle is later stolen from the society, investigators have nothing.

## The Solution

NazarBaan is an end-to-end ANPR pipeline:

1. **Detect** — fine-tuned YOLOv8 locates the license plate in the camera feed.
2. **Read** — OCR extracts the alphanumeric plate text.
3. **Log** — entry/exit events are timestamped in a local database.
4. **Match** — plates are checked against the society's registered and blacklisted vehicle lists.
5. **Display** — a gate-side screen shows the driver their detected plate for visual confirmation.

---

## Tech Stack

- **Detection:** Ultralytics YOLOv8 (fine-tuned on Pakistani plate datasets)
- **OCR:** PaddleOCR / EasyOCR (TBD after benchmarking)
- **Pipeline:** OpenCV, NumPy
- **App:** Streamlit + SQLite
- **Training:** Kaggle GPU (P100/T4)
- **Tracking:** Ultralytics built-in tracker (ByteTrack / BoT-SORT)

---

## Project Status

🚧 **In active development.** See \docs/\ for design notes and \eports/\ for evaluation results.

| Phase | Status |
|-------|--------|
| 1. Project setup | ✅ |
| 2. Dataset acquisition & inspection | ⏳ |
| 3. Baseline YOLOv8 training | ⏳ |
| 4. Dataset merging & retraining | ⏳ |
| 5. OCR integration | ⏳ |
| 6. Pipeline + tracking | ⏳ |
| 7. Streamlit gate app | ⏳ |
| 8. Demo & deployment | ⏳ |

---

## Repository Structure

\\\
nazarbaan/
├── data/              # Raw, processed, and external datasets (gitignored)
├── notebooks/         # Exploratory analysis and training notebooks
├── src/
│   ├── detection/     # YOLO training and inference
│   ├── ocr/           # Plate text recognition
│   ├── pipeline/      # End-to-end inference pipeline
│   └── app/           # Streamlit gate application
├── models/            # Trained weights (gitignored)
├── configs/           # YOLO data.yaml and hyperparameter configs
├── scripts/           # One-off utilities (data conversion, etc.)
├── reports/           # Metrics, figures, final writeup
├── tests/             # Unit tests
└── docs/              # Design notes
\\\

---

## Author

Built by **Hamza Asif** as the Computer Vision Module Project (Cohort 16).
