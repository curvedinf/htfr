"""Lightweight JSONL profiler for HTFR pipelines."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional


class PipelineProfiler:
    """Collects timestamped profiling entries and writes them to JSONL."""

    def __init__(self, output_path: Optional[str] = None) -> None:
        self._path = Path(output_path) if output_path else None
        self._entries: list[Dict[str, Any]] = []

    @property
    def enabled(self) -> bool:
        return self._path is not None

    def record(self, name: str, duration: float, metadata: Optional[Dict[str, Any]] = None) -> None:
        if not self.enabled:
            return
        self._entries.append(
            {
                "timestamp": time.time(),
                "name": name,
                "duration": float(duration),
                "metadata": metadata or {},
            }
        )

    def flush(self) -> None:
        if not self.enabled or not self._entries:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            for entry in self._entries:
                handle.write(json.dumps(entry) + "\n")
        self._entries.clear()
