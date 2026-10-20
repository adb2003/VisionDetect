"""Detection history: metadata only, in SQLite.

What is kept is the record that an analysis happened and what it found -- never
the image. The distinction matters and is visible in the UI: a history row says
"14 objects across 5 classes at 10:42", and the image it came from is gone
unless the result is still in the live cache.

SQLite is opened per call rather than held open. These are single-row writes a
few times a minute; a connection per operation costs nothing measurable and
avoids sharing a handle across the thread pool FastAPI runs sync routes on.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS detection_history (
    analysis_id          TEXT PRIMARY KEY,
    filename             TEXT    NOT NULL,
    timestamp            TEXT    NOT NULL,
    total_objects        INTEGER NOT NULL,
    unique_classes       INTEGER NOT NULL,
    processing_time_ms   REAL    NOT NULL,
    model                TEXT    NOT NULL,
    width                INTEGER NOT NULL,
    height               INTEGER NOT NULL,
    confidence_threshold REAL    NOT NULL,
    iou_threshold        REAL    NOT NULL,
    top_classes          TEXT    NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_history_timestamp
    ON detection_history (timestamp DESC);
"""


class HistoryStore:
    """Append-and-read access to the analysis log."""

    def __init__(self, path: Path, *, enabled: bool = True) -> None:
        self.path = path
        self.enabled = enabled
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as connection:
                connection.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        return connection

    def record(
        self,
        *,
        analysis_id: str,
        filename: str,
        total_objects: int,
        unique_classes: int,
        processing_time_ms: float,
        model: str,
        width: int,
        height: int,
        confidence_threshold: float,
        iou_threshold: float,
        top_classes: str,
    ) -> None:
        """Insert a run, or update it in place when settings are re-applied."""
        if not self.enabled:
            return
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO detection_history (
                    analysis_id, filename, timestamp, total_objects,
                    unique_classes, processing_time_ms, model, width, height,
                    confidence_threshold, iou_threshold, top_classes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(analysis_id) DO UPDATE SET
                    timestamp            = excluded.timestamp,
                    total_objects        = excluded.total_objects,
                    unique_classes       = excluded.unique_classes,
                    processing_time_ms   = excluded.processing_time_ms,
                    model                = excluded.model,
                    confidence_threshold = excluded.confidence_threshold,
                    iou_threshold        = excluded.iou_threshold,
                    top_classes          = excluded.top_classes
                """,
                (
                    analysis_id,
                    filename,
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    total_objects,
                    unique_classes,
                    processing_time_ms,
                    model,
                    width,
                    height,
                    confidence_threshold,
                    iou_threshold,
                    top_classes,
                ),
            )

    def list(self, *, limit: int = 100, offset: int = 0) -> tuple[list[dict[str, Any]], int]:
        if not self.enabled:
            return [], 0
        with self._connect() as connection:
            total = connection.execute(
                "SELECT COUNT(*) AS n FROM detection_history"
            ).fetchone()["n"]
            rows = connection.execute(
                """
                SELECT * FROM detection_history
                ORDER BY timestamp DESC, rowid DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
        return [dict(row) for row in rows], int(total)

    def delete(self, analysis_id: str) -> bool:
        if not self.enabled:
            return False
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM detection_history WHERE analysis_id = ?", (analysis_id,)
            )
            return cursor.rowcount > 0

    def clear(self) -> int:
        if not self.enabled:
            return 0
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM detection_history")
            return cursor.rowcount
