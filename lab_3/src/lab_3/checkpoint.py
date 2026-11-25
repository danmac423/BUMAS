"""Periodic saving of partial results, so that an interrupted run can be resumed."""

import pickle
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def atomic_pickle(path: Path, obj: object) -> None:
    """Pickle to a temporary file and rename it, so ``path`` is never left half-written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(obj, f)
    tmp.replace(path)


class Checkpoint:
    """Results per record id, saved to disk at most every ``interval`` seconds."""

    def __init__(self, path: Path, meta: Mapping[str, str], interval: float = 30.0) -> None:
        """Load earlier results if they were produced with the same settings.

        Args:
            path: Checkpoint file.
            meta: Settings that affect the results; a saved file with different ones is ignored.
            interval: Minimum number of seconds between periodic saves.
        """
        self.path, self.meta, self.interval = path, dict(meta), interval
        self.results: dict[str, Any] = {}
        if path.exists():
            with path.open("rb") as f:
                saved = pickle.load(f)
            if saved["meta"] == self.meta:
                self.results = saved["results"]
        self._saved_at = time.monotonic()

    def update(self, new: Mapping[str, Any]) -> None:
        """Add results and save if the interval has passed."""
        self.results.update(new)
        if time.monotonic() - self._saved_at >= self.interval:
            self.save()

    def save(self) -> None:
        """Write all results now."""
        atomic_pickle(self.path, {"meta": self.meta, "results": self.results})
        self._saved_at = time.monotonic()
