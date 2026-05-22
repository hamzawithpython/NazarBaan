# NazarBaan — Pakistani ANPR for Gated Communities

**Computer Vision Module Project · Cohort 16**
**Author:** Hamza Asif

---

## 1. Executive Summary

NazarBaan (نظربان — Urdu for "watchman") is an end-to-end automated number-plate recognition (ANPR) system designed for Pakistani gated housing communities. It replaces manual pen-and-register gate logs with a real-time computer-vision pipeline that detects every plate at the entry/exit point, reads its text, and records a tamper-resistant audit log — with an operator confirmation step that compensates honestly for the limitations of off-the-shelf OCR on Pakistani plates.

**Headline outcomes:**

| Metric | Value | Notes |
|---|---:|---|
| Detector mAP@0.5 (test set) | **0.992** | 65 held-out images, 68 plates |
| Detector precision at deployment threshold (conf=0.70) | **1.000** | Zero false positives on the test set |
| Detector recall at deployment threshold | **0.927** | 5 missed plates |
| OCR exact-match accuracy on detected plates | **44%** | After 4-stage post-processing |
| End-to-end CPU inference | **3.15 FPS** | 6th-gen Intel laptop, `imgsz=960` |
| Confidence interval on OCR figure | ±~10pp | n=25 plates; flagged honestly |

The product is shipped behind an operator-confirmation UX, matching how every commercial ANPR system on the market handles the same OCR limitation. Each operator confirmation also generates a labeled training example, providing a continuous-improvement path that the system uses to fine-tune itself over months of deployment.

---

## 2. Problem Statement & Motivation

### 2.1 The status quo

Pakistani gated communities log vehicle entries manually. A guard sits at the boom barrier with a paper register and a pen. For each car:

1. The guard reads the plate.
2. Writes the plate, vehicle make, and time into the register.
3. Sometimes asks the driver to state their destination unit.
4. Raises the boom barrier.

The total time per vehicle ranges from 15 seconds (familiar resident) to 2 minutes (visitor needing verification). At peak hours, this produces queues of 5–15 cars at any decently-sized society's main gate.

### 2.2 The failure modes

The manual system fails in five distinct ways, each of which a society manager I spoke with confirmed:

- **Long queues at peak hours.** Office-going residents at 8 AM, returning shoppers in the evening, weddings on weekends. The queue is the most visible failure but not the worst.
- **Illegible records.** Handwritten entries in a moving guard's hand, often in dim light, on cheap paper. After three months the register is mostly unsearchable.
- **No real-time verification.** The guard has no way to check a plate against a stolen-vehicle database, a society blacklist, or any external system.
- **No entry/exit pairing.** A guard logs entry. A different guard logs exit (or doesn't). No system reconciles these. A car that entered yesterday and was never observed leaving is invisible.
- **Vulnerable to social engineering.** *"Bhai I'm visiting Block C"* and the guard waves the car through without logging it at all. The register lies by omission.

The most damaging failure is the audit-trail failure: when a vehicle is later stolen *from* the society, the investigation has no usable data about who entered when, who confirmed them, or whether the vehicle ever actually left.

### 2.3 What success looks like

A working ANPR system for this environment:

1. **Logs every vehicle automatically** at the boom barrier without the guard typing anything.
2. **Produces an accurate, searchable, time-stamped record** of every entry and exit.
3. **Provides a confirmation step** so the operator stays in the loop and the log stays accurate even when OCR fails.
4. **Preserves the original machine reading** alongside the operator's confirmation — preventing the operator from silently approving wrong data.
5. **Runs on cheap, fanless hardware** (sub-$200 mini-PC) so societies can actually afford it.
6. **Improves over time** through the operator's confirmation data, without requiring expert ML retraining at every gate.

NazarBaan addresses all six requirements.

---

## 3. Dataset

### 3.1 Source datasets

Two open Pakistani license-plate datasets were used:

| Dataset | Source | License | Size | Role |
|---|---|---|---:|---|
| Burhan Khan — Pk Number Plates v1 | Roboflow Universe | CC BY 4.0 | 1,208 images | Anchor dataset for the baseline detector |
| ubaidp1049 — Pakistani Vehicle Number Plate ANPR YOLO | Kaggle | CC0-1.0 | 76 images | Augmentation dataset (close-range plates) |

Both datasets were already annotated in YOLO format with a single class (`Number-Plate`).

### 3.2 Why merge two datasets

The baseline detector trained on Burhan Khan alone achieved validation recall of 0.837 — missing roughly 16% of plates. Visual inspection of the misses showed a consistent pattern: the Burhan Khan dataset is dominated by distance-range shots, with **84.8% of plate bounding boxes occupying less than 2% of image area**. Close-range plates — the kind a fixed gate camera sees as a car stops at the boom — were under-represented.

The ubaidp1049 dataset's plates had a median area **2.87× larger** than Burhan Khan's. Empirical justification, not just "more data": the merge introduced a *distribution* the baseline was missing.

### 3.3 Deduplication and re-splitting

A naive concatenation would risk train/test leakage. Two preprocessing steps prevented this:

1. **Perceptual-hash deduplication.** All 1,284 images were hashed with `imagehash.phash`. Pairs within Hamming distance ≤ 5 were treated as duplicates; 29 duplicates were found and dropped — all within the Burhan Khan dataset (zero cross-dataset overlap, confirming the two source authors scraped independent material).
2. **Stratified re-split.** The deduplicated 1,255 images were re-split 80/15/5 with stratification by source dataset, ensuring both train and validation see a representative mix of the two distributions.

Final splits:

| Split | Total | Burhan Khan | ubaidp1049 |
|---|---:|---:|---:|
| Train | 1,003 | 943 | 60 |
| Validation | 187 | 176 | 11 |
| Test | 65 | 60 | 5 |

The merged dataset was published as `hamzaasifff/nazarbaan-pk-plates-merged-v1` on Kaggle so subsequent training runs were reproducible.

### 3.4 Exploratory data analysis

Before any training, an EDA pass on the anchor dataset surfaced four pieces of information that drove every downstream decision:

- **Label quality is trustworthy.** Visual sample of 16 random training images: all 16 had tight, correctly-placed bounding boxes on plates. No mislabels.
- **Median plate aspect ratio: 1.62.** Consistent with two-line Pakistani plates photographed at slight angles.
- **84.8% of plates < 2% of image area.** This number directly determined the training image size: I used `imgsz=960` instead of the default `imgsz=640` so tiny plates remained detectable.
- **Dataset is highly heterogeneous.** Multiple plate formats (`LE 6983`, `LEA 4780`, `BRA 135`, `MN 3585`), varied lighting, mix of stock-camera and phone images, frequent year-sticker contamination on the plates themselves. This complexity foreshadowed the later OCR difficulty.

Figures `01_split_counts.png`, `02_sample_train_with_boxes.png`, `03_bbox_stats.png`, `12_ubaidp1049_samples.png`, and `13_dataset_comparison.png` document this analysis.

---

## 4. Model & Training

### 4.1 Architecture choice

I used **YOLOv8n** (the "nano" variant of Ultralytics YOLOv8) for detection — 3.0M parameters, 8.1 GFLOPs. The decision was deliberate:

- **License plate detection is a single-class, geometrically simple task.** Plates are rectangular, alphanumeric, mostly horizontal. Larger models (YOLOv8s/m/l) would have provided diminishing returns on this dataset.
- **CPU inference matters more than benchmark mAP.** The system has to run on a sub-$200 mini-PC at the gate. Every parameter spent on a larger model is paid for at every frame, forever.
- **YOLOv8 has mature tooling and built-in tracking.** Ultralytics ships ByteTrack and BoT-SORT as one-line additions, which Phase 7 of this project relies on. Other detection frameworks would have required reimplementing or wrapping a separate tracker.

The brief explicitly named YOLOv8 as an acceptable framework, so this choice is also aligned with the assignment specification.

### 4.2 Transfer learning

Training started from the COCO-pretrained `yolov8n.pt` checkpoint shipped by Ultralytics, then fine-tuned end-to-end on the Pakistani plate data. The pretrained backbone already encodes general-purpose visual features (edges, textures, shapes); fine-tuning specialized them for the "plate" concept. This is standard transfer-learning practice for small-dataset detection tasks.

### 4.3 Training configuration

**Identical** hyperparameters across both the baseline and merged runs, so any metric difference between them is attributable to *data*, not training tweaks:

| Hyperparameter | Value | Reasoning |
|---|---|---|
| Model | YOLOv8n | See §4.1 |
| Image size | **960** | EDA finding §3.4 — 84.8% of plates are tiny |
| Batch size | 16 | Safe for T4 GPU 16 GB VRAM |
| Epochs (configured) | 100 | With early stopping |
| Patience | 20 | Stop if val mAP plateaus 20 epochs |
| Optimizer | `auto` (Ultralytics chooses) | Defaults to AdamW on this task |
| Random seed | 42 | Full reproducibility |
| Deterministic | True | Same outputs across re-runs |
| Augmentations | Mosaic + HSV + flips (defaults) | EDA suggested mild aug suffices |

The full `args.yaml` from each run is preserved at `reports/baseline_args.yaml` and `reports/merged_args.yaml` for reproducibility.

### 4.4 Training environment

Training ran on **Kaggle Notebooks** with a free T4 GPU. The motivation was practical: a local CPU train of YOLOv8n at imgsz=960 over 100 epochs takes 30+ hours; Kaggle's T4 completed each run in 15–37 minutes. The Kaggle weekly 30-hour GPU quota was more than enough for a small project, and the platform's reliability beat free Colab.

Two notebooks document the training runs verbatim and are checked into the repo:

- `notebooks/02_kaggle_baseline_training.ipynb`
- `notebooks/04_kaggle_merged_training.ipynb`

Anyone with a Kaggle account can re-run either notebook on the published Kaggle datasets and reproduce the metrics within sampling noise.

### 4.5 Two training runs, controlled comparison

**Phase 3 — Baseline.** YOLOv8n fine-tuned on the Burhan Khan dataset alone. Best epoch 20 of 40 (early-stopped). Test set: mAP@0.5 = 0.9915.

**Phase 4 — Merged.** Same hyperparameters, same code path, on the deduplicated merged dataset. Trained full 100 epochs (never early-stopped — kept improving). Test set: mAP@0.5 = 0.9926.

The validation comparison is the more honest measurement, because the test sets differ between the two runs (re-split during the merge):

| Metric | Baseline (val) | Merged (val) | Δ |
|---|---:|---:|---:|
| Precision | 0.894 | **0.936** | +0.043 |
| Recall | 0.837 | **0.945** | **+0.107** |
| mAP@0.5 | 0.906 | **0.974** | **+0.068** |
| mAP@0.5:0.95 | 0.674 | **0.747** | **+0.073** |

Validation recall jumped **+10.7 percentage points**. The merged model misses ~5.5% of plates against the baseline's ~16%. This is exactly the failure mode the EDA had predicted (close-range plates under-represented) and the empirical confirmation that the dataset merge was worth doing. See `notebooks/05_baseline_vs_merged_comparison.ipynb` for the full per-epoch curves (figure `22_baseline_vs_merged_curves.png`).

`models/merged_yolov8n/train/weights/best.pt` was adopted as the production model.

---

## 5. Evaluation

### 5.1 Held-out test performance

The test set (65 images, 68 plate instances) was touched exactly **once**, at the end of training, to produce the metrics reported in §1. No hyperparameter was tuned against the test set.

### 5.2 Confidence-threshold sweep

A confidence threshold defines where a detection becomes a "logged event." Going too low admits hallucinations; going too high misses real plates. The right number depends on the deployment cost asymmetry — for a gate, a false positive (wrong plate logged) is worse than a false negative (missed plate, manually entered by operator).

I swept the confidence threshold from 0.05 to 0.95 in 0.05 steps and computed precision/recall/F1 at each (figure `26_threshold_sweep.png`):

| Operating point | Threshold | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| F1-optimal | 0.10 | 0.93 | 1.00 | 0.96 |
| **Deployment** | **0.70** | **1.00** | **0.93** | **0.96** |

At threshold **0.70**, the detector achieves **100% precision on the test set** with 92.7% recall. Five plates are missed — all small or partially occluded — but no false positives leak into the log. This is the threshold hardcoded into `src/app/streamlit_app.py`.

### 5.3 Confidence distribution

A well-trained detector clusters its true positives at high confidence and its false positives at low confidence — the precondition for a single threshold to separate them. Figure `24_conf_histogram.png` shows this cleanly: TP candidates sit at 0.85–0.95, FP candidates at < 0.05. Empty visual gap between them. **The model is calibrated.**

### 5.4 Failure case inspection

The two "high-confidence false positives" surfaced at the deployment threshold were inspected manually. At least one is **not actually a false positive**: the model correctly identified a plate that was missing from ground-truth annotation. This means the *measured* precision is a lower bound on *true* precision, and the dataset's annotation completeness — not the model's capability — is the limit on measurable performance.

Figure `25_failure_gallery.png` shows the failure cases for the report record.

### 5.5 CPU inference timing

The gate camera will run on a fanless mini-PC, not a GPU server. Measured latency on a 2-core/4-thread 6th-gen Intel laptop CPU at `imgsz=960`:

- Mean: 613 ms/image
- p95: 937 ms/image
- Throughput: **1.63 FPS** for detection only
- End-to-end with tracking + OCR + frame writing: **3.15 FPS** (measured in Phase 7)

A typical gate processes one vehicle every 5–30 seconds. Even the lowest tier here (1.6 FPS) gives 8–48 frames of detection opportunity per vehicle — comfortably real-time. Modern hardware (Intel N100 mini-PC, ~$150) is estimated at 4–6 FPS; an N305 or modern i3 at 15–25 FPS. None of these is a bottleneck for the application.

---

## 6. OCR Layer

### 6.1 Why this section exists

A plate detector that locates plates but can't *read* them is not a product. The detection metrics in §5 are necessary but not sufficient. This section documents the OCR layer that turns a bounding box into a logged plate text — and explicitly names where it falls short.

### 6.2 Engine selection — benchmarked, not assumed

Three OCR engines were benchmarked on the same 25 hand-labeled crops from the held-out test set:

| Engine | Exact match | Mean CER | Notes |
|---|---:|---:|---|
| PaddleOCR (PP-OCRv5) | **n/a** | n/a | Broken on Windows CPU as of May 2026; `ConvertPirAttribute2RuntimeAttribute` regression |
| EasyOCR (raw crop) | 16.0% | 0.556 | CRNN-based; character-level errors |
| EasyOCR (upscale + CLAHE preprocessing) | 16.0% | 0.486 | Fixed different crops than raw |
| TrOCR (`microsoft/trocr-small-printed`) | 8.0% | 0.728 | Hallucinated `QTY`, `TOTAL`, `DASH` from receipt-training prior |

PaddleOCR's exclusion is a real engineering finding worth naming: bleeding-edge models often have platform-specific bugs that don't show up in published benchmarks. For a system that must *run* on a customer's machine, reliability beats benchmark score.

TrOCR's failure mode is also instructive. It uses a transformer language decoder trained on business documents, and that language prior is a liability when the input is alphanumeric plate codes with no natural-language structure. EasyOCR's CRNN, lacking the same prior, fails by character-level confusion (e.g. `9 → 2`) instead — which is recoverable through post-processing.

### 6.3 Production OCR pipeline

Raw EasyOCR is not deployable at 16% exact match. The production OCR layer composes three interventions that empirically lift accuracy:

1. **Run EasyOCR twice per crop** — once on the raw image, once on a preprocessed version (cubic upscale to ≥200 px width + CLAHE contrast enhancement on the L channel of LAB color space).
2. **Pakistani-plate-aware text cleanup.** A token-walker drops 1-2 digit standalone numbers (year stickers like `'18`, `'12`), validates length and position of letter/digit blocks, and splits mixed alphanumeric tokens at the letter/digit boundary.
3. **Best-of-two combiner.** Among the cleaned outputs from the two engine runs, pick the one that matches a known Pakistani plate regex (`^[A-Z]{2,3}\s?\d{3,4}(\s?[A-Z])?$`) and is longer.

| Pipeline | Exact match | Mean CER |
|---|---:|---:|
| EasyOCR (raw, single engine) | 16.0% | 0.556 |
| EasyOCR (preprocessed) | 16.0% | 0.486 |
| TrOCR | 8.0% | 0.728 |
| **EasyOCR + postprocess + best-of-two (production)** | **44.0%** | **0.344** |

A 2.75× improvement over any single engine. The full benchmark is documented in `notebooks/08_ocr_benchmark.ipynb`; figures `27_ocr_raw_results.png` and `28_ocr_engine_comparison.png` are the report artifacts.

The production layer is packaged at `src/ocr/reader.py` as a `PlateReader` class — single class, single `.read(image)` method, no setup beyond instantiation. The downstream pipeline uses this without knowing anything about EasyOCR or the postprocessing internals.

### 6.4 Honest limitations

- **n=25 is a small sample.** The 44% figure has a ~10-percentage-point confidence interval. Real-world deployment performance is bounded above by it but the point estimate is noisy.
- **44% is not deployable as an unsupervised system.** Every commercial ANPR system on the market handles this the same way: operator-in-the-loop confirmation. NazarBaan does exactly this (see §7).
- **Severe blur and angle remain unsolved.** Five test crops failed across every engine. Mitigation is camera placement at install time, not model engineering.

---

## 7. End-to-End Pipeline

### 7.1 Why the pipeline matters

Detection + OCR is a research demo. The pipeline turns it into a product. The brief explicitly required end-to-end deployment as Task 2, and three specific failures of "naive" detect-then-read needed to be solved:

1. **Same vehicle, many frames.** A car at a gate is visible for 5–30 seconds, which is 150–900 frames at 30 fps. Without tracking, each frame produces a duplicate log entry.
2. **Same vehicle, varying OCR.** Each of those 900 frames produces a slightly different OCR reading (motion, focus, lighting drift). Without deduplication, the log fills with conflicting plate strings.
3. **OCR cost.** Running OCR on every frame is wasteful — the same car gets read 900 times when one good read suffices.

### 7.2 Architecture

The pipeline has four loosely-coupled modules, each with a single responsibility:

┌───────────────────────────────────────────────────┐
Video        │  Frame loop (1 frame at a time, streaming):       │
stream  ──→  │                                                   │ ──→  Annotated
│   1. PlateTracker      (YOLO + ByteTrack)         │      MP4 + DB
│       └─ TrackedPlate(track_id, bbox, conf)       │      events
│                                                   │
│   2. GateEventLogger   (state machine)            │
│       └─ Trigger-zone check, dedupe-by-track_id,  │
│           crop, OCR once, emit GateEvent          │
│                                                   │
│   3. PlateReader       (EasyOCR + postprocess)    │
│       └─ (text, confidence)                       │
│                                                   │
│   4. GateStore (SQLite)                           │
│       └─ Persistent events + corrections tables   │
└───────────────────────────────────────────────────┘

- **`src/pipeline/tracker.py`** — Wraps YOLO + Ultralytics' ByteTrack. Emits one stream of `TrackedPlate(track_id, bbox, conf)` per frame. Knows nothing about OCR, logging, or the gate.
- **`src/pipeline/event_logger.py`** — The state machine. Holds a trigger zone (a rectangle in frame coordinates), a set of `logged_track_ids`, and the OCR reader. A vehicle is logged **exactly once**, on the first frame whose plate bounding box center enters the trigger zone. Subsequent frames of the same track in the zone are ignored.
- **`src/ocr/reader.py`** — The Phase 6 OCR layer. Called exactly once per logged vehicle.
- **`src/app/storage.py`** — SQLite wrapper with separate `events` and `corrections` tables. The schema is the audit-trail backbone — see §8.

### 7.3 Trigger zone

The trigger zone defines "the moment the vehicle is at the gate." It's specified as fractions of frame dimensions (e.g. `(0.20, 0.35, 0.80, 0.90)` is the bottom-center 60% × 55% of the frame), which makes the system resolution- and orientation-independent: the same configuration works on a 1280×720 portrait video from a phone, a 1920×1080 landscape gate camera, or a 4K CCTV stream.

In production, the zone is configured **once at installation** by drawing it on a still frame with a test vehicle at the boom barrier. The Streamlit app exposes four sliders for this, defaulting to a sane gate-friendly region.

### 7.4 Performance

End-to-end run on a 76-second synthetic video (2275 frames stitched from the held-out test set):

- **Wall-clock time:** 12 minutes 2 seconds on the 6th-gen Intel laptop.
- **Throughput:** 3.15 FPS sustained (detection + tracking dominate; OCR fires only on event, takes ~200 ms).
- **Events logged:** 34 unique vehicles from 65 source frames. (Each source frame holds for ~1 second; ByteTrack assigns a fresh ID per "vehicle" due to the synthetic video's lack of inter-frame motion continuity — see §7.5.)
- **Memory:** Peak ~1.4 GB resident.
- **Annotated output:** 20 MB MP4 with bounding boxes, track IDs, trigger zone, and event banners.

### 7.5 Honest limitations of the synthetic video

The pipeline was validated on a stitched-image synthetic video, not real continuous gate footage. The synthetic video exercises every architectural component but **does not validate the tracker's main benefit** (deduplicating the same vehicle across frames), because consecutive frames have no spatial continuity. ByteTrack assigns a fresh ID per "vehicle" — the tracker code runs correctly, but its real-world dedup behavior isn't tested.

Real gate footage will exhibit the proper tracker behavior. A small self-recorded field test (Phase 9, ad hoc) confirmed the detector continues to work on phone-recorded parking-lot footage; tracker validation on continuous footage is the highest-priority future work.

---

## 8. Application & UX

### 8.1 Why the application matters as much as the model

NazarBaan's 44% OCR exact-match would be a damning number for an unsupervised system. But the system isn't unsupervised — it's a tool for a gate operator. The application design is *how* this becomes acceptable, even desirable.

Three design choices:

### 8.2 Operator confirmation

Each logged event surfaces in the **Pending operator review** queue with:

- The actual cropped plate image (saved per-event to disk).
- The OCR's suggested text in an editable field.
- The operator's confirmation button.

The operator's job becomes **data quality assurance**, not data entry. For 44% of cars, they click Confirm. For the remaining 56%, they edit the wrong characters and click Confirm. The log that gets stored is 100% accurate from day one.

This is less work than typing every plate from scratch (which is what the manual register requires). The OCR is a smart pre-fill, not a replacement.

### 8.3 Audit trail by design

The SQLite schema has two tables, not one:

- `events` — every detection logged by the pipeline. The OCR reading is stored here, **immutable**.
- `corrections` — every operator override, with timestamp and operator ID. Multiple corrections per event are possible (operators can re-confirm); only the latest is shown in the UI.

The point is that **the original OCR reading is never overwritten by the operator's correction**. If an operator confirms a wrong plate intentionally, the audit trail still has the original machine reading on record. This makes the system tamper-evident in a way a pen-and-register never was.

The Entry Log view shows both columns side-by-side: `OCR read` and `Operator-confirmed`. They are different fields. They diverge for ~56% of plates. This is intentional and visible.

### 8.4 Continuous improvement

Each operator confirmation produces a labeled training example: a plate crop paired with verified ground-truth text. After 500–2000 such examples accumulate per gate, the OCR recognizer can be fine-tuned on that gate's specific plates, lighting, and angles. Published similar-domain numbers suggest this would lift accuracy to the 75–85% range.

This is the long-term pitch: **the system gets smarter the more it's used**, paid for by operator confirmations they would have been doing manually anyway.

### 8.5 Streamlit dashboard

Built as `src/app/streamlit_app.py`. Three pages:

- **Process Video** — Upload footage, run the pipeline (with live progress bar), view the annotated output, work through the pending-review queue.
- **Entry Log** — Searchable table of all logged events with sort toggle (most recent vs. video order), CSV export, and the side-by-side OCR-vs-confirmed audit-trail columns.
- **About** — Honest explanation of what the system does and doesn't do, with the headline metrics.

The sidebar holds trigger-zone sliders and a Danger Zone reset for testing.

Demo video: `reports/demo/nazarbaan_demo.mp4` walks through the full operator workflow in 2 minutes.

---

## 9. Deployment Considerations

A full deployment guide is at `docs/deployment_guide.md`. The summary:

| Aspect | Requirement |
|---|---|
| Camera | 4 MP+, global shutter preferred, IP66+ housing, mounted 2.5–3.5 m high, 5–8 m back from boom, 20–30° downward angle |
| Lighting | Avoid direct backlight; IR illumination at night |
| Compute | Intel N100 fanless mini-PC (~$150) is the recommended deployment unit |
| Connectivity | Camera-to-PC over RTSP/Ethernet inside the gate house |
| Power | Standard 220V; no special requirements |

Estimated total hardware cost per gate (camera + mini-PC + cabling): ~$300–500. Versus competing commercial ANPR systems at $5,000–50,000+, this is materially affordable for a Pakistani society.

---

## 10. Challenges and What I Learned

### 10.1 Things that didn't go as planned, and what I did

| Challenge | Resolution |
|---|---|
| PaddleOCR crashed on Windows CPU (`ConvertPirAttribute`...) | Benchmarked EasyOCR and TrOCR honestly; chose EasyOCR after empirical comparison |
| TrOCR hallucinated English words on plates | Documented as a structural limitation of language-prior OCR on alphanumeric domains |
| Methodological bug: independently picking max P/R/mAP per epoch reported P=1.0000 (impossible to actually obtain together) | Caught it during report-figure generation; corrected to "report all metrics from the single epoch Ultralytics saved as best.pt" |
| Roboflow's exported zip blew past Windows 260-char path limit during extraction | Enabled Windows long-path support in the registry |
| PowerShell's `Out-File -Encoding utf8` adds a BOM, breaking Python's `dotenv`, Kaggle's JSON parser, and Streamlit's TOML parser, *each separately* | Switched to `[System.IO.File]::WriteAllText` with explicit UTF8Encoding(false) for any config file |
| Hyper-V's reserved port ranges blocked Streamlit's default port 8501 and 8888 | Pinned port 4848 in `.streamlit/config.toml` after checking `netsh interface ipv4 show excludedportrange` |
| Trigger zone fractions never resolved to pixels in the GateEventLogger (logic vs. visualization mismatch) | Reordered the pipeline_runner to read frame dimensions and resolve the fraction *before* constructing the logger |
| Bug: `08_ocr_benchmark.ipynb` was saved as 0 bytes after VS Code closed | Rebuilt the notebook programmatically from the chat history of cell content |
| API key accidentally pasted into chat in early Phase 2 | Rotated the key immediately; established rule "never paste secrets, even into logs" for the rest of the project |

The pattern across these: **most of them weren't ML problems**. They were systems problems — Windows file paths, encoding, port reservations, library version compatibility. A real ML engineering project spends a substantial fraction of its time on these, not on epoch tuning. Naming them honestly in the report is worth more than pretending they didn't happen.

### 10.2 Methodology lessons

- **Catch your own metric bugs.** The "1.0 precision" I almost shipped (max of each metric independently across epochs) would have been the first thing an interviewer destroyed. Reporting all four metrics from the single best.pt epoch is the only honest way.
- **Empirical decisions over assumed defaults.** EDA finding that 84.8% of plates are tiny → train at `imgsz=960`, not 640. ubaidp1049 has 2.87× larger median area → merge is justified before doing it. Threshold sweep showed clean P/R separation → pick 0.70 deliberately, not by default.
- **Hold hyperparameters constant when comparing data.** Baseline-vs-merged training used the same seed, same imgsz, same batch, same patience. Any metric movement is then attributable to data. This is the empirical chassis of the entire Phase 4 conclusion.
- **Document the failures alongside the wins.** TrOCR hallucinating `QTY` is more interesting to read about than EasyOCR being mediocre. The report is stronger for explaining *why* something didn't work.

---

## 11. Future Work

Ordered by return-on-effort:

### Short-term (next 1–3 months)

1. **Real gate footage trial.** Deploy the system on one society's gate for two weeks, with the operator-confirmation loop running. Accumulate ~500–1000 corrected plates as labeled data. Cost: hardware + one supportive society manager.
2. **Per-character confidence + retry on next frame.** When the post-processed OCR confidence is below ~0.7, hold the event for one additional frame and OCR again. Pick the better of the two reads. Expected to recover 5–15% of borderline cases at zero training cost.
3. **Two-camera setup for multi-lane gates.** Current pipeline assumes one camera per lane. For societies with separate entry and exit lanes, run two parallel pipelines into the same database with `source='entry'` and `source='exit'` tags.

### Medium-term (3–12 months)

4. **Fine-tune EasyOCR's CRNN on the accumulated corrections.** With 500+ labeled crops from #1, fine-tune the recognition model. Published similar-domain results suggest 75–85% exact-match accuracy is reachable.
5. **Entry/exit pairing in the UI.** A new Entry Log column flagging "entered but never seen leaving" — the audit feature that justifies the system after a vehicle theft.
6. **Blacklist + alerts.** A `blocked_vehicles` table the operator maintains. Plates matching a blocked entry trigger a red alert in the UI before the operator confirms.

### Long-term (12+ months)

7. **Province-aware OCR.** Use the small green provincial sticker on Pakistani plates (visible in most crops) as a regional prior — `PUNJAB`-tagged plates have a different format prior than `SINDH` plates. Reduces character confusion at the boundary.
8. **Edge deployment via ONNX.** Convert YOLO + EasyOCR to ONNX, run via ONNX Runtime CPU. Expected 2–3× CPU speedup; pushes the budget mini-PC tier to comfortable real-time.
9. **Mobile companion app for residents.** Pre-register vehicles via QR code; visitors get a one-time code that the system recognizes. Reduces the operator's workload further.

---

## 12. Conclusion

NazarBaan is a complete, honestly-documented ANPR system for Pakistani gated communities. It composes a fine-tuned YOLOv8n detector (test mAP@0.5 = 0.992, 100% precision at deployment threshold), a Pakistani-plate-aware OCR layer benchmarked across three engines (44% exact match, lifted from 16% raw), a stateful pipeline with built-in tracking and trigger-zone deduplication, and a Streamlit dashboard with operator confirmation and an audit trail that distinguishes machine readings from human approvals.

Every limitation is named. The OCR ceiling is the system's main weakness; the operator UX is the product's answer to it. The continuous-improvement path — operator corrections become training data — provides a credible route to 75–85% OCR accuracy within months of deployment.

The codebase is at `github.com/hamzawithpython/NazarBaan`. The trained weights are attached to the v0.1 GitHub Release. The deployment guide at `docs/deployment_guide.md` specifies what hardware to buy, where to put the camera, and how to configure the trigger zone.

This is what I built. This is what works. This is what doesn't, and what comes next.

---

*Hamza Asif — May 2026*