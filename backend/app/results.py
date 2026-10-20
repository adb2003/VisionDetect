"""In-memory store for completed detection runs.

Results are held in the process, not on disk. That is a deliberate privacy
position rather than an expedient one: an uploaded site photograph is the
user's, and the default behaviour should be that closing the server is enough to
be rid of it. Nothing is written unless `persist_uploads` is explicitly set.

The original pixels are kept alongside the result for the TTL so that changing
the confidence threshold re-runs the model on the server instead of asking the
browser to upload the same photograph again.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.detector import RawDetection


@dataclass
class StoredResult:
    """One completed run: its payload, its detections, and its source image."""

    result_id: str
    payload: dict[str, Any]
    """The `DetectionResponse` as a plain dict, minus the inline data URL."""

    detections: list[RawDetection]
    image: np.ndarray
    """Original BGR pixels, retained for re-runs and for re-rendering exports."""

    filename: str
    created_at: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.created_at = time.time()


class ResultStore:
    """Bounded, TTL-expiring cache of recent results."""

    def __init__(self, *, ttl_seconds: int, max_entries: int) -> None:
        self._ttl = ttl_seconds
        self._max = max_entries
        self._items: OrderedDict[str, StoredResult] = OrderedDict()
        self._lock = threading.Lock()

    def put(self, result: StoredResult) -> None:
        with self._lock:
            self._items[result.result_id] = result
            self._items.move_to_end(result.result_id)
            self._expire_locked()
            while len(self._items) > self._max:
                self._items.popitem(last=False)

    def get(self, result_id: str) -> StoredResult | None:
        with self._lock:
            self._expire_locked()
            result = self._items.get(result_id)
            if result is not None:
                self._items.move_to_end(result_id)
            return result

    def has(self, result_id: str) -> bool:
        return self.get(result_id) is not None

    def ids(self) -> set[str]:
        with self._lock:
            self._expire_locked()
            return set(self._items)

    def discard(self, result_id: str) -> None:
        with self._lock:
            self._items.pop(result_id, None)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def _expire_locked(self) -> None:
        if self._ttl <= 0:
            return
        cutoff = time.time() - self._ttl
        stale = [key for key, item in self._items.items() if item.created_at < cutoff]
        for key in stale:
            del self._items[key]
