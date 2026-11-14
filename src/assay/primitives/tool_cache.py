# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Best-effort result cache for in-process tool dispatch.

A shared, in-process result cache that a ``ChatToolsBackend`` may consult so
identical tool invocations (same name plus normalized arguments) can return a
cached result string instead of dispatching again. The structural protocol
does not require an adapter to honor the cache.

Shared-by-reference semantics: a single ``ToolCallCache`` instance is
passed to every workflow operation that wants cross-operation reuse. Callers
choose that sharing scope explicitly.

Concurrency: hit/miss counters are locked, but lookup followed by insertion is
not an atomic ``get_or_compute`` operation. Concurrent identical misses may
therefore dispatch the same tool more than once, and concurrent writes use
last-writer-wins semantics. Share this cache for reuse, not for exactly-once
execution.

Keying uses ``(tool_name, json.dumps(args, sort_keys=True, default=str))``.
This is cache-key normalization, not Assay record canonicalization. Callers
should supply JSON-shaped arguments; the ``default=str`` fallback is lossy and
can intentionally collapse a custom object with its string representation.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field


def canonical_args(arguments: dict) -> str:
    """Stable, sorted JSON spelling used only for tool-cache keys."""
    return json.dumps(arguments, sort_keys=True, default=str, ensure_ascii=False)


@dataclass
class ToolCallCache:
    """Best-effort ``(name, args)`` result cache with hit/miss statistics.

    The cache does not promise atomic get-or-compute or exactly-once tool
    execution under concurrency.
    """

    _store: dict[tuple[str, str], str] = field(default_factory=dict)
    _hits: int = 0
    _misses: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def get(self, name: str, arguments: dict) -> str | None:
        """Return cached result string for ``(name, arguments)`` or None."""
        key = (name, canonical_args(arguments))
        value = self._store.get(key)
        with self._lock:
            if value is None:
                self._misses += 1
            else:
                self._hits += 1
        return value

    def put(self, name: str, arguments: dict, result: str) -> None:
        """Insert or overwrite — last writer wins. (Idempotent for an
        identical (name, args, result); a re-put with a different result
        for the same key replaces the prior entry.)"""
        key = (name, canonical_args(arguments))
        self._store[key] = result

    def clear(self) -> None:
        """Drop all entries + reset counters."""
        self._store.clear()
        with self._lock:
            self._hits = 0
            self._misses = 0

    @property
    def hits(self) -> int:
        with self._lock:
            return self._hits

    @property
    def misses(self) -> int:
        with self._lock:
            return self._misses

    @property
    def size(self) -> int:
        return len(self._store)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"hits": self._hits, "misses": self._misses, "size": len(self._store)}


__all__ = ["ToolCallCache", "canonical_args"]
