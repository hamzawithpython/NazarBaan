"""
Persistent storage for the NazarBaan gate app.

SQLite-backed event log with two tables:
  - events: every vehicle logged by the pipeline (one row per entry)
  - corrections: every operator override of an OCR-suggested plate text

The corrections table is *additive* — original OCR readings are never
overwritten. This preserves the audit trail (what the system said vs what
actually happened) and provides the training-data feedback loop documented
in the deployment guide.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional


DEFAULT_DB_PATH = Path("data/processed/nazarbaan.db")


@dataclass
class StoredEvent:
    """One row from the events table, joined with the latest correction (if any)."""
    event_id: int
    timestamp_iso: str
    frame_idx: int
    track_id: int
    ocr_plate_text: str        # what the OCR layer originally read
    ocr_confidence: float
    detect_confidence: float
    bbox_str: str              # 'x1,y1,x2,y2'
    source: str                # 'video' or 'live'
    confirmed_plate_text: Optional[str]  # latest operator correction, if any
    confirmed_at: Optional[str]
    confirmed_by: Optional[str]


SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    event_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp_iso   TEXT NOT NULL,
    frame_idx       INTEGER,
    track_id        INTEGER,
    ocr_plate_text  TEXT,
    ocr_confidence  REAL,
    detect_confidence REAL,
    bbox_str        TEXT,
    source          TEXT
);

CREATE TABLE IF NOT EXISTS corrections (
    correction_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id        INTEGER NOT NULL,
    confirmed_plate_text TEXT NOT NULL,
    confirmed_at    TEXT NOT NULL,
    confirmed_by    TEXT,
    FOREIGN KEY(event_id) REFERENCES events(event_id)
);

CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp_iso DESC);
CREATE INDEX IF NOT EXISTS idx_corrections_event ON corrections(event_id);
"""


class GateStore:
    """Thin wrapper around the SQLite DB. Safe to instantiate multiple times;
    each call opens its own short-lived connection inside _conn()."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as cx:
            cx.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        cx = sqlite3.connect(self._db_path)
        cx.row_factory = sqlite3.Row
        try:
            yield cx
            cx.commit()
        finally:
            cx.close()

    # ---------- Event writes ----------

    def insert_event(
        self,
        timestamp_iso: str,
        frame_idx: int,
        track_id: int,
        ocr_plate_text: str,
        ocr_confidence: float,
        detect_confidence: float,
        bbox_str: str,
        source: str = "video",
    ) -> int:
        """Insert a new event row, return its event_id."""
        with self._conn() as cx:
            cur = cx.execute(
                """INSERT INTO events
                   (timestamp_iso, frame_idx, track_id, ocr_plate_text,
                    ocr_confidence, detect_confidence, bbox_str, source)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (timestamp_iso, frame_idx, track_id, ocr_plate_text,
                 ocr_confidence, detect_confidence, bbox_str, source),
            )
            return int(cur.lastrowid)

    def confirm_event(
        self,
        event_id: int,
        confirmed_text: str,
        confirmed_by: Optional[str] = "operator",
    ) -> None:
        """Record an operator's confirmed/corrected plate text for this event."""
        with self._conn() as cx:
            cx.execute(
                """INSERT INTO corrections
                   (event_id, confirmed_plate_text, confirmed_at, confirmed_by)
                   VALUES (?, ?, ?, ?)""",
                (event_id, confirmed_text,
                 datetime.now().isoformat(timespec="seconds"), confirmed_by),
            )

    def wipe_all(self) -> dict:
        """Delete every event and correction. Returns the row counts cleared."""
        with self._conn() as cx:
            n_events = cx.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            n_corrections = cx.execute("SELECT COUNT(*) FROM corrections").fetchone()[0]
            cx.execute("DELETE FROM corrections")
            cx.execute("DELETE FROM events")
            # Reset auto-increment counters so the next event starts at id=1 again
            cx.execute("DELETE FROM sqlite_sequence WHERE name IN ('events', 'corrections')")
        return {"events_deleted": int(n_events), "corrections_deleted": int(n_corrections)}

    # ---------- Event reads ----------

    def list_events(
        self,
        limit: int = 200,
        search: Optional[str] = None,
        only_unconfirmed: bool = False,
    ) -> list[StoredEvent]:
        """Return recent events joined with their latest correction (if any).
        search: substring match against either OCR text or confirmed text."""
        query = """
            SELECT e.*,
                   c.confirmed_plate_text,
                   c.confirmed_at,
                   c.confirmed_by
            FROM events e
            LEFT JOIN (
                SELECT event_id,
                       confirmed_plate_text,
                       confirmed_at,
                       confirmed_by,
                       ROW_NUMBER() OVER (PARTITION BY event_id ORDER BY correction_id DESC) AS rn
                FROM corrections
            ) c ON c.event_id = e.event_id AND c.rn = 1
            WHERE 1=1
        """
        params: list = []
        if search:
            query += " AND (e.ocr_plate_text LIKE ? OR c.confirmed_plate_text LIKE ?)"
            params += [f"%{search}%", f"%{search}%"]
        if only_unconfirmed:
            query += " AND c.confirmed_plate_text IS NULL"
        query += " ORDER BY e.timestamp_iso DESC, e.event_id DESC LIMIT ?"
        params.append(limit)

        with self._conn() as cx:
            rows = cx.execute(query, params).fetchall()

        return [
            StoredEvent(
                event_id=r["event_id"],
                timestamp_iso=r["timestamp_iso"],
                frame_idx=r["frame_idx"],
                track_id=r["track_id"],
                ocr_plate_text=r["ocr_plate_text"] or "",
                ocr_confidence=r["ocr_confidence"] or 0.0,
                detect_confidence=r["detect_confidence"] or 0.0,
                bbox_str=r["bbox_str"] or "",
                source=r["source"] or "video",
                confirmed_plate_text=r["confirmed_plate_text"],
                confirmed_at=r["confirmed_at"],
                confirmed_by=r["confirmed_by"],
            )
            for r in rows
        ]

    def stats(self) -> dict:
        """Quick dashboard numbers."""
        with self._conn() as cx:
            total = cx.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            confirmed = cx.execute(
                "SELECT COUNT(DISTINCT event_id) FROM corrections"
            ).fetchone()[0]
            today = cx.execute(
                "SELECT COUNT(*) FROM events WHERE date(timestamp_iso) = date('now')"
            ).fetchone()[0]
        return {
            "total_events": int(total),
            "confirmed": int(confirmed),
            "today": int(today),
            "unconfirmed": int(total - confirmed),
        }