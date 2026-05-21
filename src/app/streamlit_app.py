"""
NazarBaan — gate-camera dashboard.

A Streamlit app that wraps the Phase 7 pipeline with an operator UI:
  1. Process Video — upload footage, run the pipeline, see events flagged
     in a confirmation queue.
  2. Entry Log — searchable history; click any event to view crop + correct.
  3. About — honest system explanation with metrics from the report.

Run from the project root:
    streamlit run src/app/streamlit_app.py
"""

from __future__ import annotations

import sys
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd
from PIL import Image
import streamlit as st

# Make `from src...` work when streamlit launches this file directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.app.pipeline_runner import run_pipeline_on_video
from src.app.storage import GateStore, StoredEvent


# ────────────────────────────────────────────────────────────────────
# Paths & config
# ────────────────────────────────────────────────────────────────────
WEIGHTS = PROJECT_ROOT / "models" / "merged_yolov8n" / "train" / "weights" / "best.pt"
# Proportional defaults: center 60% width, middle band that catches plates
# in either landscape (mounted gate cam) or portrait (handheld phone) framing.
DEFAULT_TRIGGER_ZONE = (0.20, 0.35, 0.80, 0.90)  # fractions of frame width/height
ANNOTATED_DIR = PROJECT_ROOT / "data" / "processed" / "app_runs"


# ────────────────────────────────────────────────────────────────────
# Page config + global look
# ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NazarBaan — ANPR Gate System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 2rem; }
      .stMetric { background: #f4f6f9; padding: 0.6rem; border-radius: 6px; }
      .small-mono { font-family: monospace; font-size: 0.85rem; color: #444; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ────────────────────────────────────────────────────────────────────
# Cached resources
# ────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_store() -> GateStore:
    """Single shared GateStore across reruns."""
    return GateStore()


# ────────────────────────────────────────────────────────────────────
# Sidebar — identity + nav
# ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚗 NazarBaan")
    st.markdown("*Automated number-plate recognition for Pakistani gated communities.*")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["Process Video", "Entry Log", "About"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### Workflow")
    st.markdown(
        "1. **Process** a gate-camera video.  \n"
        "2. **Confirm** OCR readings as the operator.  \n"
        "3. **Search & audit** the entry log.  \n"
    )
    st.markdown("---")
    st.caption(f"Model: `merged_yolov8n` &nbsp; · &nbsp; trigger conf 0.70")

    st.markdown("---")
    with st.expander("🎯 Trigger zone", expanded=False):
        st.caption(
            "Defines where in the frame a vehicle's plate must appear for the system to log it. "
            "Values are percentages of frame size, so this works at any resolution or orientation. "
            "For a fixed gate camera, set this once to match the boom barrier position."
        )
        tz_left = st.slider("Left %", 0, 100, 20, key="tz_left")
        tz_right = st.slider("Right %", 0, 100, 80, key="tz_right")
        tz_top = st.slider("Top %", 0, 100, 35, key="tz_top")
        tz_bottom = st.slider("Bottom %", 0, 100, 90, key="tz_bottom")
        st.session_state["trigger_zone"] = (tz_left / 100, tz_top / 100,
                                            tz_right / 100, tz_bottom / 100)

    st.markdown("---")
    with st.expander("⚠️ Danger zone", expanded=False):
        st.caption(
            "Clears every event, every correction, and every generated annotated video. "
            "Useful for re-testing from a clean slate. **This cannot be undone.**"
        )
        confirm = st.checkbox("I understand, wipe everything")
        if st.button("🗑️ Reset all data", disabled=not confirm, type="secondary"):
            counts = get_store().wipe_all()
            # Also nuke the saved annotated videos so app_runs/ doesn't bloat
            import shutil
            runs_dir = PROJECT_ROOT / "data" / "processed" / "app_runs"
            if runs_dir.exists():
                shutil.rmtree(runs_dir, ignore_errors=True)
            st.success(
                f"Cleared {counts['events_deleted']} events, "
                f"{counts['corrections_deleted']} corrections, and the app_runs folder."
            )
            st.rerun()


store = get_store()


# ────────────────────────────────────────────────────────────────────
# PAGE: PROCESS VIDEO
# ────────────────────────────────────────────────────────────────────
def page_process_video():
    st.title("Process Video")
    st.markdown(
        "Upload gate-camera footage. The detector locates plates, the tracker assigns "
        "persistent IDs per vehicle, and OCR reads each plate **exactly once** as it "
        "enters the trigger zone."
    )

    # ---- Stats strip
    s = store.stats()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total events", s["total_events"])
    c2.metric("Confirmed", s["confirmed"])
    c3.metric("Pending review", s["unconfirmed"])
    c4.metric("Today", s["today"])

    st.markdown("---")

    # ---- Upload + run
    uploaded = st.file_uploader(
        "Gate-camera video (MP4)",
        type=["mp4", "mov", "avi"],
        help="Recommended: 720p+, 15-25 fps, plate clearly visible at the boom.",
    )

    if uploaded is not None:
        st.info(f"Loaded: **{uploaded.name}** ({uploaded.size / 1e6:.1f} MB)")

        if st.button("▶ Run pipeline", type="primary"):
            ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            in_path = ANNOTATED_DIR / f"input_{stamp}.mp4"
            out_path = ANNOTATED_DIR / f"annotated_{stamp}.mp4"
            in_path.write_bytes(uploaded.getvalue())

            progress = st.progress(0.0, text="Initialising pipeline…")
            status = st.empty()

            def on_progress(cur: int, total: int) -> None:
                if total > 0:
                    progress.progress(min(cur / total, 1.0),
                                      text=f"Processing frame {cur} / {total}")

            try:
                result = run_pipeline_on_video(
                    video_path=in_path,
                    weights_path=WEIGHTS,
                    annotated_out_path=out_path,
                    store=store,
                    trigger_zone=st.session_state.get("trigger_zone", DEFAULT_TRIGGER_ZONE),
                    conf_threshold=0.70,
                    imgsz=960,
                    progress_cb=on_progress,
                    source_label="video",
                )
            except Exception as ex:
                progress.empty()
                st.error(f"Pipeline failed: {ex}")
                return

            progress.progress(1.0, text="Done.")
            status.success(
                f"Processed **{result.total_frames}** frames in "
                f"**{result.elapsed_seconds:.1f}s** ({result.fps:.2f} FPS) — "
                f"**{result.events_logged}** events logged."
            )

            # Show annotated video inline
            st.video(str(result.annotated_video_path))
            st.session_state["last_run_at"] = datetime.now().isoformat(timespec="seconds")

    # ---- Confirmation queue
    st.markdown("---")
    st.subheader("Pending operator review")
    st.caption(
        "Each event below is a vehicle the system logged but the operator hasn't yet "
        "confirmed. The OCR reading is shown as a suggestion — edit it if wrong, then "
        "click Confirm. Original OCR readings are preserved in the audit log."
    )

    pending = store.list_events(limit=1000, only_unconfirmed=True)
    if not pending:
        st.info("No pending events. Upload a video to populate the queue.")
        return

    for ev in pending:
        _render_event_card(ev)


def _render_event_card(ev: StoredEvent) -> None:
    """One pending-event row: plate crop image + OCR suggestion + Confirm button."""
    crop_path = (PROJECT_ROOT / "data" / "processed" / "app_runs"
                 / "crops" / f"event_{ev.event_id}.jpg")

    with st.container(border=True):
        cols = st.columns([1.5, 1, 2, 2])

        # Crop image (or placeholder text if missing)
        with cols[0]:
            if crop_path.exists():
                st.image(str(crop_path), use_container_width=True)
            else:
                st.caption("_no crop saved_")

        # Metadata
        with cols[1]:
            st.markdown(f"**#{ev.event_id}**")
            st.caption(f"Frame {ev.frame_idx}")
            st.caption(f"Track {ev.track_id}")

        # Confidences
        with cols[2]:
            st.markdown(f"OCR conf: **{ev.ocr_confidence:.2f}**")
            st.markdown(f"Detect conf: **{ev.detect_confidence:.2f}**")
            st.caption(ev.timestamp_iso)

        # Confirm interaction
        with cols[3]:
            text_key = f"text_input_{ev.event_id}"
            corrected = st.text_input(
                "Plate text",
                value=ev.ocr_plate_text,
                key=text_key,
            )
            confirm_key = f"confirm_btn_{ev.event_id}"
            if st.button("✓ Confirm", key=confirm_key, type="primary"):
                if not corrected.strip():
                    st.warning("Plate text can't be empty.")
                else:
                    store.confirm_event(ev.event_id, corrected.strip())
                    st.rerun()

# ────────────────────────────────────────────────────────────────────
# PAGE: ENTRY LOG
# ────────────────────────────────────────────────────────────────────
def page_entry_log():
    st.title("Entry Log")
    st.markdown(
        "Every vehicle the system has logged. Each row shows the original OCR reading and "
        "the operator-confirmed text (if any). The two columns are kept separate as an "
        "audit trail."
    )

    s = store.stats()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total events", s["total_events"])
    c2.metric("Confirmed", s["confirmed"])
    c3.metric("Pending review", s["unconfirmed"])
    c4.metric("Today", s["today"])

    st.markdown("---")

    cf1, cf2, cf3 = st.columns([2, 1, 1])
    search = cf1.text_input("Search by plate text", placeholder="e.g. LEA")
    only_unconf = cf2.checkbox("Only unconfirmed", value=False)
    sort_mode = cf3.selectbox(
        "Order by",
        ["Most recent first", "Video order (frame ↑)"],
        label_visibility="collapsed",
    )

    events = store.list_events(limit=500, search=search or None, only_unconfirmed=only_unconf)
    if not events:
        st.info("No events match.")
        return

    df = pd.DataFrame(
        [
            {
                "Event": e.event_id,
                "Frame": e.frame_idx,
                "When": e.timestamp_iso,
                "Track": e.track_id,
                "OCR read": e.ocr_plate_text or "—",
                "Operator-confirmed": e.confirmed_plate_text or "—",
                "OCR conf": f"{e.ocr_confidence:.2f}",
                "Detect conf": f"{e.detect_confidence:.2f}",
                "Confirmed at": e.confirmed_at or "",
                "By": e.confirmed_by or "",
            }
            for e in events
        ]
    )
    if sort_mode == "Video order (frame ↑)":
        df = df.sort_values("Frame", ascending=True).reset_index(drop=True)
    st.dataframe(df, hide_index=True, use_container_width=True)

    st.download_button(
        "Download as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"nazarbaan_entry_log_{datetime.now():%Y%m%d_%H%M}.csv",
        mime="text/csv",
    )


# ────────────────────────────────────────────────────────────────────
# PAGE: ABOUT
# ────────────────────────────────────────────────────────────────────
def page_about():
    st.title("About NazarBaan")
    st.markdown(
        """
        **NazarBaan** (نظربان — Urdu for *watchman*) is an automated number-plate
        recognition system designed for Pakistani gated communities.

        It replaces pen-and-register gate logs with a real-time computer-vision pipeline
        that **detects, reads, and logs every vehicle** entering or exiting, with a
        driver-facing display, entry/exit pairing, and a tamper-resistant audit trail.
        """
    )

    st.markdown("### Why this matters")
    st.markdown(
        "Pakistani gated communities log vehicles manually: a guard with a register, a pen, "
        "and a queue of impatient drivers. The result: long queues, illegible records, no "
        "real-time check against stolen-vehicle databases, no entry/exit pairing, and no "
        "audit trail when a vehicle is later stolen. NazarBaan fixes this."
    )

    st.markdown("---")
    st.markdown("### What the model achieves")

    c1, c2, c3 = st.columns(3)
    c1.metric("Detector mAP@0.5 (test set)", "0.992")
    c2.metric("Detector precision @ conf 0.70", "1.000")
    c3.metric("End-to-end exact-match OCR", "44 %")

    st.caption(
        "Detector trained on a merged Pakistani plate dataset (Burhan Khan + ubaidp1049, "
        "1,255 deduplicated images). Held-out test set: 65 images, 68 plates. OCR layer "
        "combines EasyOCR (raw + preprocessed) with Pakistani-plate-aware postprocessing "
        "(year-sticker stripping, format validation, best-of-two combiner)."
    )

    st.markdown("---")
    st.markdown("### How the operator loop works")
    st.markdown(
        "44 % exact-match OCR is **not** deployable as an unsupervised system. NazarBaan "
        "embraces this honestly. Every commercial ANPR system on the market does the same "
        "thing: the operator sees the OCR suggestion next to the live plate image and "
        "**confirms or edits before the boom barrier opens**. Each correction is logged "
        "and becomes training data — after ~500 corrections per gate, a fine-tuned OCR "
        "model dedicated to that gate's plates, lighting, and angle can lift accuracy to "
        "the 75–85 % range typical of published similar-domain results."
    )

    st.markdown("---")
    st.markdown("### Built by")
    st.markdown(
        "**Hamza Asif** — Computer Vision portfolio project (Cohort 16).  \n"
        "Source: [github.com/hamzawithpython/NazarBaan](https://github.com/hamzawithpython/NazarBaan)"
    )


# ────────────────────────────────────────────────────────────────────
# Router
# ────────────────────────────────────────────────────────────────────
if page == "Process Video":
    page_process_video()
elif page == "Entry Log":
    page_entry_log()
else:
    page_about()