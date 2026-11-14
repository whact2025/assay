# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Transactional run-folder records using POSIX ``fcntl`` locking."""

from __future__ import annotations

import dataclasses
import fcntl
import json
import math
import os
import secrets
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class RunAlreadyFinalizedError(RuntimeError):
    """A caller attempted to replace a terminal run result."""


_RUN_WRITE_LOCK = threading.RLock()


@contextmanager
def _run_write_guard(run_dir: Path):
    """Serialize one run's mutable prefix and immutable finalization."""

    lock_path = run_dir / ".run-write.lock"
    with _RUN_WRITE_LOCK, lock_path.open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


@dataclass(frozen=True)
class RunResult:
    """Immutable terminal status written to a run's ``result.json``."""

    status: str
    duration_s: float = 0.0
    cost_usd: float = 0.0
    error_kind: str | None = None
    error_message: str | None = None


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _create_once(path: Path, payload: bytes) -> bool:
    """Atomically create ``path`` without replacing an existing value."""

    tmp = path.parent / f".tmp_{path.name}_{os.getpid()}_{secrets.token_hex(4)}"
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError:
            return False
        # Persist the directory entry where supported.
        try:
            dir_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
        return True
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _atomic_replace(path: Path, payload: bytes) -> None:
    """Durably replace a private run-folder document."""

    tmp = path.parent / f".tmp_{path.name}_{os.getpid()}_{secrets.token_hex(4)}"
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        try:
            dir_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


@dataclass(frozen=True)
class RunHandle:
    run_dir: Path
    recipe: str
    started_at: str
    _opened_monotonic: float = field(
        default_factory=time.monotonic, repr=False, compare=False
    )

    @property
    def finalized(self) -> bool:
        return (self.run_dir / "result.json").is_file()

    @property
    def elapsed_s(self) -> float:
        """Current monotonic wall time since this run handle opened."""

        return max(0.0, time.monotonic() - self._opened_monotonic)

    def write_event(self, payload: dict) -> None:
        with _run_write_guard(self.run_dir):
            if self.finalized:
                raise RunAlreadyFinalizedError(
                    f"run {self.run_dir.name!r} is finalized; "
                    "cannot append events"
                )
            line = _json_bytes(payload) + b"\n"
            fd = os.open(
                self.run_dir / "events.jsonl",
                os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                0o600,
            )
            try:
                view = memoryview(line)
                while view:
                    written = os.write(fd, view)
                    if written <= 0:
                        raise OSError("events.jsonl append made no progress")
                    view = view[written:]
                os.fsync(fd)
            finally:
                os.close(fd)

    def write_summary(self, payload: Any) -> None:
        """Atomically publish ``summary.json`` before finalization.

        A killed process can leave no summary or the previous complete
        checkpoint, never a half-written JSON document. Run folders are
        private and the file is mode 0600. Finalized runs are immutable.
        """

        with _run_write_guard(self.run_dir):
            if self.finalized:
                raise RunAlreadyFinalizedError(
                    f"run {self.run_dir.name!r} is finalized; "
                    "cannot replace summary"
                )
            data = json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                default=str,
            ).encode("utf-8")
            _atomic_replace(self.run_dir / "summary.json", data)

    def finalize(self, result: RunResult) -> None:
        """Commit one immutable terminal result.

        Repeating the same finalization is idempotent. A conflicting second
        result raises rather than silently rewriting run history.
        """

        with _run_write_guard(self.run_dir):
            path = self.run_dir / "result.json"
            payload = _json_bytes(dataclasses.asdict(result))
            if _create_once(path, payload):
                return
            try:
                existing = path.read_bytes()
            except FileNotFoundError:
                # A racing writer's link should already be visible, but retry
                # once if an unusual filesystem delayed visibility.
                if _create_once(path, payload):
                    return
                existing = path.read_bytes()
            if existing == payload:
                return
            raise RunAlreadyFinalizedError(
                f"run {self.run_dir.name!r} already has a different "
                "terminal result"
            )

    def finalize_elapsed(self, result: RunResult) -> RunResult:
        """Measure elapsed time and commit one immutable terminal result.

        ``result`` must retain the default zero duration so callers cannot
        accidentally mix an independently measured value with this handle's
        monotonic clock. The first call snapshots that clock exactly once.
        Repeating the same non-duration terminal fields returns the already
        recorded result; a conflicting repeat fails closed.
        """

        if result.duration_s != 0.0:
            raise ValueError(
                "finalize_elapsed requires RunResult.duration_s to retain "
                "its default value"
            )

        with _run_write_guard(self.run_dir):
            path = self.run_dir / "result.json"
            try:
                existing = path.read_bytes()
            except FileNotFoundError:
                elapsed = time.monotonic() - self._opened_monotonic
                if not math.isfinite(elapsed) or elapsed < 0.0:
                    raise ValueError(
                        "measured elapsed duration must be finite and "
                        "non-negative"
                    )
                measured = dataclasses.replace(result, duration_s=elapsed)
                payload = _json_bytes(dataclasses.asdict(measured))
                if _create_once(path, payload):
                    return measured
                existing = path.read_bytes()

            recorded: RunResult | None = None
            try:
                recorded = RunResult(**json.loads(existing))
                recorded_duration = recorded.duration_s
                duration_is_valid = (
                    isinstance(recorded_duration, (int, float))
                    and not isinstance(recorded_duration, bool)
                    and math.isfinite(recorded_duration)
                    and recorded_duration >= 0.0
                )
            except (
                OverflowError,
                TypeError,
                UnicodeDecodeError,
                ValueError,
                json.JSONDecodeError,
            ):
                duration_is_valid = False

            if (
                recorded is not None
                and duration_is_valid
                and dataclasses.replace(recorded, duration_s=0.0) == result
            ):
                return recorded
            raise RunAlreadyFinalizedError(
                f"run {self.run_dir.name!r} already has a different "
                "terminal result"
            )

    def __enter__(self) -> "RunHandle":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if not self.finalized:
            duration = self.elapsed_s
            if exc is None:
                result = RunResult(status="completed", duration_s=duration)
            else:
                result = RunResult(
                    status="failed",
                    duration_s=duration,
                    error_kind=getattr(exc_type, "__name__", "Exception"),
                    error_message=str(exc)[:2000],
                )
            # Finalization errors should not hide the workflow exception.
            try:
                self.finalize(result)
            except Exception:
                if exc is None:
                    raise
        return False
