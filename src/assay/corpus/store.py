# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Corpus — on-disk store for Spec versions, artifacts, and run handles.

``Corpus`` is a regular mutable resource owner, not a frozen record. The
constructor pre-creates the six on-disk subdirectories
(``versions/``, ``artifacts/``, ``transcripts/``, ``diffs/``,
``traces/``, ``runs/``) and defensively hydrates the kind→dataclass
registry so builtin constraint subtypes resolve regardless of import order.

Spec manifests live under ``versions/v<N>_<short>.json`` (idempotent,
hash-keyed bytes). Raw blobs live under
``artifacts/<kind>/<hex>.bin`` (content-addressed). Each
``open_run()`` stamps a fresh ``runs/<ts>_<recipe>_<tag>/`` directory, or
returns the same deterministic folder when given an idempotency key.
Measurements are indexed separately in ``measurements.sqlite`` and still use
the same canonical content-hash identity as other typed records.

The Spec and run locks use ``fcntl.flock``. The current persistence
implementation therefore requires a POSIX runtime.
"""
from __future__ import annotations

import dataclasses
import datetime
import fcntl
import hashlib
import json
import os
import re
import secrets
import sqlite3
import stat
import tempfile
import threading
import weakref
from contextlib import AbstractContextManager, ContextDecorator, contextmanager
from pathlib import Path
from types import TracebackType
from typing import BinaryIO, Callable, Iterable, Iterator, Literal

from assay.bootstrap import populate_registry
from assay.canonical import hash_record, parse, serialize
from assay.types import (
    ActivationConstraint,
    ContentHash,
    Implementation,
    LifecycleEvent,
    Measurement,
    Spec,
    Verdict,
)

from .codec import (
    decode_dataclass,
    validate_content_hash,
    validate_dataclass_instance,
)
from .runs import RunHandle, _create_once


_SUBDIRS = ("versions", "artifacts", "transcripts", "diffs", "traces", "runs")
_VERSION_FILE_RE = re.compile(
    r"^v(0|[1-9][0-9]*)_[0-9a-f]{12}\.json$"
)
_VERSION_FILE_LIKE_RE = re.compile(r"^v[0-9]+(?:_|\.json(?:\.|$))")
_RUN_RECIPE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SPEC_WRITE_LOCK = threading.RLock()
_SPEC_FORK_GUARD = threading.RLock()
_SPEC_AUTHORITY_GUARDS = threading.local()
_SPEC_AUTHORITY_OWNERS = weakref.WeakSet()
_SPEC_MANIFEST_MAX_BYTES = 64 * 1024 * 1024
_SPEC_MAX_VERSION = (1 << 63) - 1
_BOUNDED_ARTIFACT_MAX_ENTRIES = 1_000_000
_BOUNDED_ARTIFACT_MAX_DEPTH = 64

_MEASUREMENTS_DB_NAME = "measurements.sqlite"
_MEASUREMENTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS measurements (
    ref            TEXT PRIMARY KEY,
    bench_id       TEXT NOT NULL,
    commit_sha     TEXT NOT NULL,
    config_hash    TEXT NOT NULL,
    raw_output_ref TEXT,
    timestamp      TEXT,
    machine        TEXT,
    cwd            TEXT,
    toolchain_json TEXT,
    backend        TEXT,
    model          TEXT,
    recipe         TEXT,
    recipe_commit  TEXT,
    gpu_model      TEXT,
    run_id         TEXT,
    values_json    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_meas_bench_id    ON measurements(bench_id);
CREATE INDEX IF NOT EXISTS idx_meas_commit      ON measurements(commit_sha);
CREATE INDEX IF NOT EXISTS idx_meas_config_hash ON measurements(config_hash);
CREATE INDEX IF NOT EXISTS idx_meas_timestamp   ON measurements(timestamp);
"""


def _prepare_spec_process_fork() -> None:
    """Exclude a fork by another thread from every live Spec lock region."""

    _SPEC_FORK_GUARD.acquire()


def _resume_spec_process_after_parent_fork() -> None:
    """Drop the temporary fork depth while preserving any enclosing owner."""

    _SPEC_FORK_GUARD.release()


def _reset_spec_process_state_after_fork() -> None:
    """Close inherited leases and install fresh child-process thread locks."""

    global _SPEC_FORK_GUARD, _SPEC_WRITE_LOCK
    try:
        for owner in tuple(_SPEC_AUTHORITY_OWNERS):
            if owner._owner_pid is None:
                # Construction publishes no process-owned resource. Preserve
                # a never-entered one-shot owner for first use in the child.
                continue
            try:
                owner._abandon_inherited_process_state()
            except BaseException:
                # The child must never unlock the parent's shared open-file
                # description. A failed close remains bounded to this child
                # and the process will close the descriptor on exit.
                pass
    finally:
        _SPEC_WRITE_LOCK = threading.RLock()
        _SPEC_FORK_GUARD = threading.RLock()
        _SPEC_AUTHORITY_GUARDS.pid = os.getpid()
        _SPEC_AUTHORITY_GUARDS.paths = frozenset()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(
        before=_prepare_spec_process_fork,
        after_in_parent=_resume_spec_process_after_parent_fork,
        after_in_child=_reset_spec_process_state_after_fork,
    )


class ConcurrentSpecError(RuntimeError):
    """A Spec write cannot extend the current unambiguous lineage.

    This covers stale versions or parents, a competing same-version write, and
    divergent manifests already present on disk. Callers should reload the
    latest Spec and use the locked transaction API rather than overwrite
    lineage history.
    """


class SpecAuthorityAcquisitionLease:
    """Source-owned proof that a retained Spec authority is live.

    Callers that must survive an asynchronous exception between ``__enter__``
    and their own cleanup registration retain the accompanying authority's
    ``retry_cleanup`` callback before entry.  The callback is inert while the
    lease is pending and becomes live before the exact Spec snapshot returns.
    """

    def __init__(self) -> None:
        self._phase: Literal[
            "pending",
            "active",
            "settling",
            "released",
            "abandoned",
        ] = "pending"

    @property
    def entry_pending(self) -> bool:
        return self._phase == "pending"

    @property
    def cleanup_required(self) -> bool:
        return self._phase in {"active", "settling"}

    @property
    def cleanup_complete(self) -> bool:
        return self._phase in {"released", "abandoned"}

    def _publish(self) -> None:
        if self._phase != "pending":
            raise RuntimeError("Spec authority acquisition lease is not reusable")
        self._phase = "active"

    def _begin_settlement(self) -> None:
        if self._phase == "active":
            self._phase = "settling"

    def _mark_released(self) -> None:
        if self._phase not in {"active", "settling"}:
            raise RuntimeError("Spec authority lease has no live release")
        self._phase = "released"

    def _abandon(self) -> None:
        if self._phase == "pending":
            self._phase = "abandoned"

    def _abandon_after_fork(self) -> None:
        """Retire a child copy without claiming to release parent authority."""

        if self._phase != "released":
            self._phase = "abandoned"


class SpecAuthorityCleanupError(RuntimeError):
    """Exact retry authority for an incompletely released Spec guard."""

    def __init__(
        self,
        owner: "RetainedSpecAuthority",
        *,
        primary_error: BaseException,
        cleanup_error: BaseException,
    ) -> None:
        self.owner = owner
        self.primary_error = primary_error
        self.cleanup_error = cleanup_error
        super().__init__(
            "Spec authority cleanup remains pending; inspect primary_error "
            "and cleanup_error for the retained failure details"
        )

    def retry_cleanup(self) -> None:
        """Retry the exact retained guard, flock, stream, and thread lock."""

        self.owner.retry_cleanup()


def _rlock_depth(lock: threading.RLock) -> tuple[int, bool]:
    """Return this thread's RLock depth and whether that depth is exact."""

    recursion_count = getattr(lock, "_recursion_count", None)
    if callable(recursion_count):
        return int(recursion_count()), True
    is_owned = getattr(lock, "_is_owned", None)
    if callable(is_owned):
        return (1 if is_owned() else 0), False
    raise RuntimeError("Spec thread-lock ownership is not observable")


def _spec_write_lock_depth() -> tuple[int, bool]:
    return _rlock_depth(_SPEC_WRITE_LOCK)


def _spec_fork_guard_depth() -> tuple[int, bool]:
    return _rlock_depth(_SPEC_FORK_GUARD)


def _add_exception_note(error: BaseException, note: str) -> None:
    """Attach diagnostic context when the running Python supports notes."""

    add_note = getattr(error, "add_note", None)
    if callable(add_note):
        add_note(note)


class RetainedSpecAuthority(
    AbstractContextManager[Spec | None],
    ContextDecorator,
):
    """A Spec snapshot whose exact cleanup can be retained before entry.

    The owner acquires the same process/thread locks as Corpus writers and
    publishes its lease only after the thread guard and immutable latest Spec
    are installed.  Cleanup is idempotent and advances in dependency order;
    an interrupted release retains this object for explicit retry.
    """

    def __init__(self, corpus: "Corpus") -> None:
        self._corpus = corpus
        self.acquisition_lease = SpecAuthorityAcquisitionLease()
        self._entry_started = False
        self._owner_pid: int | None = None
        self._owner_thread_id: int | None = None
        self._lock_key: Path | None = None
        self._previous_authorities: frozenset[Path] | None = None
        self._guard_restore_pending = False
        self._fork_guard_depth_before: int | None = None
        self._fork_guard_depth_exact = False
        self._fork_guard_release_pending = False
        self._thread_lock_depth_before: int | None = None
        self._thread_lock_depth_exact = False
        self._thread_lock_release_pending = False
        self._lock_stream: BinaryIO | None = None
        self._flock_release_pending = False
        self._retained_primary_error: BaseException | None = None
        with _SPEC_FORK_GUARD:
            _SPEC_AUTHORITY_OWNERS.add(self)

    @property
    def cleanup_complete(self) -> bool:
        return bool(
            self.acquisition_lease.cleanup_complete
            and not self._guard_restore_pending
            and not self._flock_release_pending
            and self._lock_stream is None
            and not self._thread_lock_release_pending
            and not self._fork_guard_release_pending
        )

    def _recreate_cm(self) -> "RetainedSpecAuthority":
        """Return a fresh one-shot owner when used as a context decorator."""

        return type(self)(self._corpus)

    def _assert_cleanup_thread(self) -> None:
        if self._owner_pid is None or self._owner_thread_id is None:
            return
        if os.getpid() != self._owner_pid:
            raise RuntimeError(
                "Spec authority cleanup belongs to another process generation"
            )
        if threading.get_ident() != self._owner_thread_id:
            raise RuntimeError("Spec authority cleanup belongs to another thread")

    def _thread_lock_has_added_depth(self) -> bool:
        baseline = self._thread_lock_depth_before
        if baseline is None:
            return False
        current, exact = _spec_write_lock_depth()
        if self._thread_lock_depth_exact and exact:
            return current > baseline
        if baseline:
            raise RuntimeError(
                "cannot prove nested Spec thread-lock cleanup on this runtime"
            )
        return current > 0

    def _fork_guard_has_added_depth(self) -> bool:
        baseline = self._fork_guard_depth_before
        if baseline is None:
            return False
        current, exact = _spec_fork_guard_depth()
        if self._fork_guard_depth_exact and exact:
            return current > baseline
        if baseline:
            raise RuntimeError(
                "cannot prove nested Spec fork-guard cleanup on this runtime"
            )
        return current > 0

    def _release_thread_lock(self) -> None:
        if not self._thread_lock_release_pending:
            return
        if not self._thread_lock_has_added_depth():
            self._thread_lock_release_pending = False
            return
        _SPEC_WRITE_LOCK.release()
        if self._thread_lock_has_added_depth():
            raise RuntimeError("Spec authority thread-lock depth did not settle")
        self._thread_lock_release_pending = False

    def _release_fork_guard(self) -> None:
        if not self._fork_guard_release_pending:
            return
        if not self._fork_guard_has_added_depth():
            self._fork_guard_release_pending = False
            return
        _SPEC_FORK_GUARD.release()
        if self._fork_guard_has_added_depth():
            raise RuntimeError("Spec authority fork-guard depth did not settle")
        self._fork_guard_release_pending = False

    def _abandon_inherited_process_state(self) -> None:
        """Close only this child's inherited descriptor and forget old locks."""

        stream = self._lock_stream
        if stream is not None:
            try:
                stream.close()
            except BaseException:
                if stream.closed:
                    self._lock_stream = None
                raise
            self._lock_stream = None
        # Never issue LOCK_UN for the inherited open description: that would
        # revoke the still-live parent's flock.  The child received a fresh
        # module RLock in the at-fork hook, so its copy of the parent's depth
        # must likewise be abandoned rather than released through that lock.
        self._flock_release_pending = False
        self._thread_lock_release_pending = False
        self._fork_guard_release_pending = False
        self._guard_restore_pending = False
        self._previous_authorities = None
        self.acquisition_lease._abandon_after_fork()

    def _cleanup(self) -> None:
        if self.cleanup_complete:
            return
        if self._owner_pid is not None and os.getpid() != self._owner_pid:
            self._abandon_inherited_process_state()
            return
        self._assert_cleanup_thread()
        self.acquisition_lease._begin_settlement()

        if self._flock_release_pending:
            stream = self._lock_stream
            if stream is None:
                raise RuntimeError("Spec authority flock lacks its open stream")
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            self._flock_release_pending = False

        if self._lock_stream is not None:
            stream = self._lock_stream
            try:
                stream.close()
            except BaseException:
                if stream.closed:
                    self._lock_stream = None
                raise
            self._lock_stream = None

        self._release_thread_lock()
        # Keep the same-thread fail-fast guard published until every lower
        # level owner has settled.  Otherwise a failed unlock would permit a
        # reentrant writer to block forever on a second open-description flock.
        if self._guard_restore_pending:
            previous = self._previous_authorities
            if previous is None:
                raise RuntimeError("Spec authority guard lacks prior state")
            _SPEC_AUTHORITY_GUARDS.paths = previous
            self._guard_restore_pending = False
        self._release_fork_guard()
        if (
            self._guard_restore_pending
            or self._flock_release_pending
            or self._lock_stream is not None
            or self._thread_lock_release_pending
            or self._fork_guard_release_pending
        ):
            raise RuntimeError("Spec authority cleanup retained unresolved state")
        if self.acquisition_lease.entry_pending:
            self.acquisition_lease._abandon()
        else:
            self.acquisition_lease._mark_released()

    def _cleanup_error(
        self,
        *,
        primary_error: BaseException,
        cleanup_error: BaseException,
    ) -> SpecAuthorityCleanupError:
        if self._retained_primary_error is None:
            self._retained_primary_error = primary_error
        return SpecAuthorityCleanupError(
            self,
            primary_error=self._retained_primary_error,
            cleanup_error=cleanup_error,
        )

    def __enter__(self) -> Spec | None:
        if self._entry_started or not self.acquisition_lease.entry_pending:
            raise RuntimeError("retained Spec authority is not reusable")
        self._entry_started = True
        self._owner_pid = os.getpid()
        self._owner_thread_id = threading.get_ident()
        try:
            (
                self._fork_guard_depth_before,
                self._fork_guard_depth_exact,
            ) = _spec_fork_guard_depth()
            if self._fork_guard_depth_before and not self._fork_guard_depth_exact:
                raise RuntimeError(
                    "retained Spec authority cannot prove reentrant fork-guard "
                    "depth on this runtime"
                )
            self._fork_guard_release_pending = True
            acquired = _SPEC_FORK_GUARD.acquire()
            if not acquired:
                raise RuntimeError("failed to acquire the Spec fork guard")

            versions_dir = self._corpus._versions_dir()
            lock_path = versions_dir / ".spec-write.lock"
            self._lock_key = lock_path.resolve()
            active_authorities = _active_spec_authority_paths()
            if self._lock_key in active_authorities:
                raise RuntimeError(
                    "Spec writes are forbidden inside locked_spec_authority()"
                )

            (
                self._thread_lock_depth_before,
                self._thread_lock_depth_exact,
            ) = _spec_write_lock_depth()
            if (
                self._thread_lock_depth_before
                and not self._thread_lock_depth_exact
            ):
                raise RuntimeError(
                    "retained Spec authority cannot prove reentrant thread-lock "
                    "depth on this runtime"
                )
            self._thread_lock_release_pending = True
            write_lock_acquired = _SPEC_WRITE_LOCK.acquire()
            if not write_lock_acquired:
                raise RuntimeError("failed to acquire the Spec thread lock")

            self._lock_stream = lock_path.open("a+b")
            # LOCK_UN is idempotent for this unique open description. Mark it
            # pending before LOCK_EX so an exception after kernel acquisition
            # but before Python return still has an exact release action.
            self._flock_release_pending = True
            fcntl.flock(self._lock_stream.fileno(), fcntl.LOCK_EX)

            self._previous_authorities = active_authorities
            self._guard_restore_pending = True
            _SPEC_AUTHORITY_GUARDS.paths = active_authorities | {self._lock_key}
            spec = self._corpus.latest_spec()
            self.acquisition_lease._publish()
            return spec
        except BaseException as entry_error:
            try:
                self._cleanup()
            except BaseException as cleanup_error:
                if self.cleanup_complete:
                    _add_exception_note(
                        entry_error,
                        "Spec authority cleanup was interrupted only after "
                        "every retained owner released",
                    )
                else:
                    raise self._cleanup_error(
                        primary_error=entry_error,
                        cleanup_error=cleanup_error,
                    ) from entry_error
            raise

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        error: BaseException | None,
        _traceback: TracebackType | None,
    ) -> bool:
        try:
            self._cleanup()
        except BaseException as cleanup_error:
            if self.cleanup_complete:
                if error is not None:
                    _add_exception_note(
                        error,
                        "Spec authority cleanup was interrupted only after "
                        "every retained owner released",
                    )
                    return False
                raise
            primary_error = error if error is not None else cleanup_error
            raise self._cleanup_error(
                primary_error=primary_error,
                cleanup_error=cleanup_error,
            ) from cleanup_error
        return False

    def retry_cleanup(self) -> None:
        """Retry this exact authority release without entering it again."""

        try:
            self._cleanup()
        except BaseException as cleanup_error:
            if self.cleanup_complete:
                raise
            primary_error = (
                self._retained_primary_error
                if self._retained_primary_error is not None
                else cleanup_error
            )
            raise self._cleanup_error(
                primary_error=primary_error,
                cleanup_error=cleanup_error,
            ) from cleanup_error


def _active_spec_authority_paths() -> frozenset[Path]:
    """Return this thread's guards without inheriting them across ``fork``."""

    pid = os.getpid()
    if getattr(_SPEC_AUTHORITY_GUARDS, "pid", None) != pid:
        _SPEC_AUTHORITY_GUARDS.pid = pid
        _SPEC_AUTHORITY_GUARDS.paths = frozenset()
    return getattr(_SPEC_AUTHORITY_GUARDS, "paths", frozenset())


@dataclasses.dataclass(frozen=True)
class CommitRefs:
    """Record references published by one Corpus transaction."""

    implementation_refs: tuple[ContentHash, ...] = ()
    verdict_refs: tuple[ContentHash, ...] = ()
    measurement_refs: tuple[ContentHash, ...] = ()
    lifecycle_event_refs: tuple[ContentHash, ...] = ()


@dataclasses.dataclass(frozen=True)
class CommitResult:
    """The Spec manifest and refs made visible by ``Corpus.commit``."""

    spec: Spec
    refs: CommitRefs


_RECORD_KIND_TYPES: dict[str, type] = {
    "implementations": Implementation,
    "verdicts": Verdict,
    "measurements": Measurement,
    "lifecycle_events": LifecycleEvent,
}
_RECORD_TYPE_KINDS: dict[type, str] = {
    record_type: kind for kind, record_type in _RECORD_KIND_TYPES.items()
}


class Corpus:
    """On-disk corpus rooted at ``spec_dir``.

    Side-effecting by design: persistence methods read or write the underlying
    directory tree and, for Measurements, the SQLite index. Use the context
    manager or ``close()`` to release the lazily opened database connection.
    """

    def __init__(self, spec_dir: Path) -> None:
        self.spec_dir = Path(spec_dir)
        self.spec_dir.mkdir(parents=True, exist_ok=True)
        for sub in _SUBDIRS:
            (self.spec_dir / sub).mkdir(parents=True, exist_ok=True)
        versions = self.spec_dir / "versions"
        versions_stat = versions.lstat()
        if not stat.S_ISDIR(versions_stat.st_mode):
            raise ConcurrentSpecError(
                "Corpus versions authority must be a real directory, not "
                f"an alias or special entry: {versions}"
            )
        self._versions_directory_identity = (
            versions_stat.st_dev,
            versions_stat.st_ino,
        )
        populate_registry()
        self._measurements_conn: sqlite3.Connection | None = None
        self._measurements_conn_lock = threading.RLock()

    # ------------------------------------------------------------------
    # Measurements (SQLite-backed)
    # ------------------------------------------------------------------

    def _measurements_db(self) -> sqlite3.Connection:
        with self._measurements_conn_lock:
            if self._measurements_conn is None:
                db_path = self.spec_dir / _MEASUREMENTS_DB_NAME
                # Corpus commits are serialized by the Spec lock but may be
                # initiated by different worker threads.  SQLite itself is
                # built in serialized mode; disabling Python's creator-thread
                # guard lets this connection follow those serialized calls.
                conn = sqlite3.connect(str(db_path), check_same_thread=False)
                # WAL lets readers share the DB with a concurrent measurement
                # writer.
                # busy_timeout backstops a transient-lock collision instead
                # of returning SQLITE_BUSY immediately.
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA busy_timeout=5000")
                conn.executescript(_MEASUREMENTS_SCHEMA)
                conn.commit()
                self._measurements_conn = conn
            return self._measurements_conn

    def close(self) -> None:
        """Release the SQLite connection. Idempotent."""
        with self._measurements_conn_lock:
            if self._measurements_conn is not None:
                self._measurements_conn.close()
                self._measurements_conn = None

    def __enter__(self) -> "Corpus":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _put_measurement(self, m: Measurement) -> ContentHash:
        if not isinstance(m, Measurement):
            raise TypeError(
                f"measurement must be Measurement, got {type(m).__name__}"
            )
        validate_dataclass_instance(m)
        ref = ContentHash(hash_record(m))
        ex = m.execution
        row = (
            ref,
            m.bench_id,
            m.commit,
            m.config_hash,
            m.raw_output_ref,
            ex.timestamp,
            ex.machine,
            ex.cwd,
            json.dumps(ex.toolchain or {}, sort_keys=True),
            ex.backend,
            ex.model,
            ex.recipe,
            ex.recipe_commit,
            ex.gpu_model,
            ex.run_id,
            json.dumps(m.values, sort_keys=True),
        )
        with self._measurements_conn_lock:
            conn = self._measurements_db()
            conn.execute(
                "INSERT OR REPLACE INTO measurements VALUES "
                "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                row,
            )
            conn.commit()
        return ref

    def put_measurements(
        self, measurements: Iterable[Measurement]
    ) -> list[ContentHash]:
        """Bulk-insert Measurements into the SQLite store.

        Returns ContentHash refs in input order. A single transaction avoids
        per-row autocommit overhead for large ingestions.
        """
        rows: list[tuple] = []
        refs: list[ContentHash] = []
        for m in measurements:
            if not isinstance(m, Measurement):
                raise TypeError(
                    f"measurement must be Measurement, got {type(m).__name__}"
                )
            validate_dataclass_instance(m)
            ref = ContentHash(hash_record(m))
            refs.append(ref)
            ex = m.execution
            rows.append((
                ref,
                m.bench_id,
                m.commit,
                m.config_hash,
                m.raw_output_ref,
                ex.timestamp,
                ex.machine,
                ex.cwd,
                json.dumps(ex.toolchain or {}, sort_keys=True),
                ex.backend,
                ex.model,
                ex.recipe,
                ex.recipe_commit,
                ex.gpu_model,
                ex.run_id,
                json.dumps(m.values, sort_keys=True),
            ))
        with self._measurements_conn_lock:
            conn = self._measurements_db()
            conn.executemany(
                "INSERT OR REPLACE INTO measurements VALUES "
                "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                rows,
            )
            conn.commit()
        return refs

    def _measurement_from_row(self, row: tuple) -> Measurement:
        (
            ref, bench_id, commit_sha, config_hash, raw_output_ref,
            timestamp, machine, cwd, toolchain_json, backend, model,
            recipe, recipe_commit, gpu_model, run_id, values_json,
        ) = row
        measurement = decode_dataclass(
            {
                "bench_id": bench_id,
                "commit": commit_sha,
                "config_hash": config_hash,
                "values": json.loads(values_json),
                "raw_output_ref": raw_output_ref,
                "execution": {
                    "timestamp": timestamp or "",
                    "machine": machine or "",
                    "cwd": cwd or "",
                    "toolchain": (
                        json.loads(toolchain_json) if toolchain_json else {}
                    ),
                    "backend": backend,
                    "model": model,
                    "recipe": recipe,
                    "recipe_commit": recipe_commit,
                    "gpu_model": gpu_model,
                    "run_id": run_id,
                },
            },
            Measurement,
        )
        expected_ref = ContentHash(hash_record(measurement))
        if ref != expected_ref:
            raise ValueError(
                f"corrupt measurement row {ref!r}: canonical content hashes "
                f"to {expected_ref!r}"
            )
        return measurement

    def iter_measurements(
        self,
        *,
        bench_id: str | None = None,
        commit: str | None = None,
        config_hash: str | None = None,
        refs: Iterable[ContentHash] | None = None,
    ) -> Iterator[Measurement]:
        """Stream Measurements matching the given filters.

        ``refs`` (when provided) restricts to spec-version-scoped rows
        and preserves the input order so callers iterating
        ``spec.measurement_refs`` see results in spec order."""
        clauses: list[str] = []
        params: list[object] = []
        if bench_id is not None:
            clauses.append("bench_id = ?")
            params.append(bench_id)
        if commit is not None:
            clauses.append("commit_sha = ?")
            params.append(commit)
        if config_hash is not None:
            clauses.append("config_hash = ?")
            params.append(config_hash)
        ref_list: list[ContentHash] | None = None
        if refs is not None:
            ref_list = list(refs)
            if not ref_list:
                return
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        conn = self._measurements_db()
        if ref_list is None:
            cur = conn.execute(f"SELECT * FROM measurements{where}", params)
            for row in cur:
                yield self._measurement_from_row(row)
            return
        # SQLite caps host parameters at SQLITE_MAX_VARIABLE_NUMBER (~32K).
        # Chunk ref filter into windows that stay under it.
        by_ref: dict[str, tuple] = {}
        chunk = 500
        for i in range(0, len(ref_list), chunk):
            window = ref_list[i:i + chunk]
            placeholders = ",".join("?" for _ in window)
            chunk_clauses = clauses + [f"ref IN ({placeholders})"]
            chunk_where = " WHERE " + " AND ".join(chunk_clauses)
            chunk_params = list(params) + list(window)
            cur = conn.execute(
                f"SELECT * FROM measurements{chunk_where}", chunk_params,
            )
            for row in cur:
                by_ref[row[0]] = row
        for r in ref_list:
            row = by_ref.get(r)
            if row is None:
                continue
            yield self._measurement_from_row(row)

    def bench_id_summary(
        self, *, refs: Iterable[ContentHash] | None = None,
    ) -> list[tuple[str, int, str | None]]:
        """Return ``(bench_id, sample_count, last_timestamp)`` per benchmark,
        sorted **most-recently-active first** (then by count).

        Recency-first matters for temporal corpora: ranking purely by sample
        count buries newly-added benchmarks (few samples but current) behind
        long-lived high-volume ones. A cheap aggregate — no per-row JSON
        parse. With ``refs`` it scopes to a spec version's measurements;
        without, it groups the whole table.
        """
        conn = self._measurements_db()
        if refs is None:
            cur = conn.execute(
                "SELECT bench_id, COUNT(*), MAX(timestamp) FROM measurements "
                "GROUP BY bench_id"
            )
            rows = [(bid, n, ts) for bid, n, ts in cur]
        else:
            ref_list = list(refs)
            if not ref_list:
                return []
            agg: dict[str, tuple[int, str | None]] = {}
            chunk = 500
            for i in range(0, len(ref_list), chunk):
                window = ref_list[i:i + chunk]
                placeholders = ",".join("?" for _ in window)
                cur = conn.execute(
                    f"SELECT bench_id, COUNT(*), MAX(timestamp) FROM measurements "
                    f"WHERE ref IN ({placeholders}) GROUP BY bench_id",
                    list(window),
                )
                for bid, n, ts in cur:
                    pn, pts = agg.get(bid, (0, None))
                    nts = pts if ts is None else (ts if pts is None else max(pts, ts))
                    agg[bid] = (pn + int(n or 0), nts)
            rows = [(bid, n, ts) for bid, (n, ts) in agg.items()]
        # Most recent first (None timestamps last), then highest count.
        rows.sort(key=lambda r: (r[2] or "", r[1]), reverse=True)
        return rows

    def distinct_machines(
        self, *, refs: Iterable[ContentHash] | None = None,
    ) -> list[tuple[str, int]]:
        """Return ``(machine, sample_count)`` per distinct ``execution.machine``
        string, most-frequent first. A cheap aggregate (no JSON parse) used to
        surface the platform/GPU a spec's measurements ran on. With ``refs`` it
        scopes to a spec version; without, it groups the whole table.
        """
        conn = self._measurements_db()
        if refs is None:
            cur = conn.execute(
                "SELECT machine, COUNT(*) FROM measurements "
                "WHERE machine IS NOT NULL AND machine <> '' GROUP BY machine"
            )
            agg: dict[str, int] = {m: int(n or 0) for m, n in cur}
        else:
            ref_list = list(refs)
            if not ref_list:
                return []
            agg = {}
            chunk = 500
            for i in range(0, len(ref_list), chunk):
                window = ref_list[i:i + chunk]
                placeholders = ",".join("?" for _ in window)
                cur = conn.execute(
                    f"SELECT machine, COUNT(*) FROM measurements "
                    f"WHERE ref IN ({placeholders}) AND machine IS NOT NULL "
                    f"AND machine <> '' GROUP BY machine",
                    list(window),
                )
                for m, n in cur:
                    agg[m] = agg.get(m, 0) + int(n or 0)
        rows = list(agg.items())
        rows.sort(key=lambda r: r[1], reverse=True)
        return rows

    def bench_id_version_series(
        self,
        metric_keys: Iterable[str],
        *,
        bench_ids: Iterable[str] | None = None,
        refs: Iterable[ContentHash] | None = None,
        machines: Iterable[str] | None = None,
    ) -> dict[str, list[dict]]:
        """Per ``(bench_id, version)``, the within-version mean of each metric,
        ordered by time — the valid temporal series for a regression baseline
        (pooling a bench_id across versions would wash out the version-to-
        version regressions the baseline exists to track).

        Version = ``commit_sha``. Returns::

            {bench_id: [{"version": str, "timestamp": str | None,
                         "count": int, "means": {metric: float | None},
                         "counts": {metric: int}}, ...]}

        ``count`` is total rows for the (bench_id, version); ``counts[metric]``
        is how many of those rows carried that metric (non-null) — use it to
        size disjoint cohorts (e.g. successful vs crashed cells).

        each list sorted by timestamp (then version). Means are computed in
        SQL over ``values_json`` (no per-row Python parse). ``bench_ids``
        (when given) restricts the scan — pass the top-N for a sparkline
        page or a single id for a drill-in chart. ``machines`` (when given)
        restricts to those ``execution.machine`` strings — used to scope a
        series to one GPU/platform so different-hardware runs aren't pooled
        into one within-version mean. ``metric_keys`` restricted to
        ``[A-Za-z0-9_.]+``.
        """
        import re as _re

        keys = [k for k in dict.fromkeys(metric_keys)
                if _re.fullmatch(r"[A-Za-z0-9_.]+", k)]
        if not keys:
            return {}
        bset = list(dict.fromkeys(bench_ids)) if bench_ids is not None else None
        if bset is not None and not bset:
            return {}
        mset = list(dict.fromkeys(machines)) if machines is not None else None
        if mset is not None and not mset:
            return {}

        exprs: list[str] = ["bench_id", "commit_sha", "MIN(timestamp)", "COUNT(*)"]
        for k in keys:
            je = f"json_extract(values_json,'$.{k}')"
            exprs += [f"SUM({je})", f"COUNT({je})"]
        select = ", ".join(exprs)

        # Accumulate sum/count (additive across ref-chunks) per (bench, version).
        acc: dict[tuple[str, str], dict] = {}

        def _merge(rows) -> None:
            for row in rows:
                bid, commit, ts, cnt = row[0], row[1], row[2], int(row[3] or 0)
                e = acc.setdefault((bid, commit), {
                    "ts": ts, "count": 0,
                    "sums": {k: 0.0 for k in keys}, "cnts": {k: 0 for k in keys},
                })
                if ts is not None and (e["ts"] is None or ts < e["ts"]):
                    e["ts"] = ts
                e["count"] += cnt
                i = 4
                for k in keys:
                    e["sums"][k] += float(row[i] or 0.0)
                    e["cnts"][k] += int(row[i + 1] or 0)
                    i += 2

        conn = self._measurements_db()
        base_clauses: list[str] = []
        base_params: list[object] = []
        if bset is not None:
            base_clauses.append("bench_id IN (" + ",".join("?" for _ in bset) + ")")
            base_params += bset
        if mset is not None:
            base_clauses.append("machine IN (" + ",".join("?" for _ in mset) + ")")
            base_params += mset

        if refs is None:
            where = (" WHERE " + " AND ".join(base_clauses)) if base_clauses else ""
            _merge(conn.execute(
                f"SELECT {select} FROM measurements{where} "
                f"GROUP BY bench_id, commit_sha", base_params,
            ))
        else:
            ref_list = list(refs)
            if not ref_list:
                return {}
            chunk = 500
            for i in range(0, len(ref_list), chunk):
                window = ref_list[i:i + chunk]
                clauses = base_clauses + [
                    "ref IN (" + ",".join("?" for _ in window) + ")"
                ]
                _merge(conn.execute(
                    f"SELECT {select} FROM measurements "
                    f"WHERE {' AND '.join(clauses)} GROUP BY bench_id, commit_sha",
                    base_params + list(window),
                ))

        out: dict[str, list[dict]] = {}
        for (bid, commit), e in acc.items():
            means = {
                k: (e["sums"][k] / e["cnts"][k] if e["cnts"][k] else None)
                for k in keys
            }
            out.setdefault(bid, []).append({
                "version": commit, "timestamp": e["ts"],
                "count": e["count"], "means": means,
                # Per-metric non-null counts: `count` is total rows for the
                # (bench_id, version); `counts[k]` is how many had metric `k`.
                # Lets callers separate cohorts that set disjoint keys (e.g.
                # successful cells carry `mean_fps`, crashed cells `cell_failed`).
                "counts": dict(e["cnts"]),
            })
        for bid in out:
            out[bid].sort(key=lambda p: (p["timestamp"] or "", p["version"]))
        return out

    def bench_machine_version_sums(
        self,
        metric_keys: Iterable[str],
        *,
        bench_ids: Iterable[str] | None = None,
        refs: Iterable[ContentHash] | None = None,
    ) -> dict[tuple[str, str], dict[str, dict]]:
        """Raw per-``(bench_id, machine, version)`` metric sums + counts, so a
        caller can fold ``machine`` into a coarser group (e.g. GPU model) and
        compute correct within-(version, group) means without pooling across
        hardware. One table scan. Returns::

            {(bench_id, machine): {version: {"ts": str|None, "count": int,
                "sums": {metric: float}, "cnts": {metric: int}}}}

        ``metric_keys`` restricted to ``[A-Za-z0-9_.]+``.
        """
        import re as _re

        keys = [k for k in dict.fromkeys(metric_keys)
                if _re.fullmatch(r"[A-Za-z0-9_.]+", k)]
        if not keys:
            return {}
        bset = list(dict.fromkeys(bench_ids)) if bench_ids is not None else None
        if bset is not None and not bset:
            return {}

        exprs = ["bench_id", "machine", "commit_sha", "MIN(timestamp)", "COUNT(*)"]
        for k in keys:
            je = f"json_extract(values_json,'$.{k}')"
            exprs += [f"SUM({je})", f"COUNT({je})"]
        select = ", ".join(exprs)

        acc: dict[tuple[str, str], dict[str, dict]] = {}

        def _merge(rows) -> None:
            for row in rows:
                bid, machine, commit, ts, cnt = (
                    row[0], row[1] or "", row[2], row[3], int(row[4] or 0))
                per = acc.setdefault((bid, machine), {})
                e = per.setdefault(commit, {
                    "ts": ts, "count": 0,
                    "sums": {k: 0.0 for k in keys}, "cnts": {k: 0 for k in keys}})
                if ts is not None and (e["ts"] is None or ts < e["ts"]):
                    e["ts"] = ts
                e["count"] += cnt
                i = 5
                for k in keys:
                    e["sums"][k] += float(row[i] or 0.0)
                    e["cnts"][k] += int(row[i + 1] or 0)
                    i += 2

        conn = self._measurements_db()
        base_clauses: list[str] = []
        base_params: list[object] = []
        if bset is not None:
            base_clauses.append("bench_id IN (" + ",".join("?" for _ in bset) + ")")
            base_params += bset
        group = " GROUP BY bench_id, machine, commit_sha"
        if refs is None:
            where = (" WHERE " + " AND ".join(base_clauses)) if base_clauses else ""
            _merge(conn.execute(f"SELECT {select} FROM measurements{where}{group}",
                                base_params))
        else:
            ref_list = list(refs)
            if not ref_list:
                return {}
            chunk = 500
            for i in range(0, len(ref_list), chunk):
                window = ref_list[i:i + chunk]
                clauses = base_clauses + [
                    "ref IN (" + ",".join("?" for _ in window) + ")"]
                _merge(conn.execute(
                    f"SELECT {select} FROM measurements "
                    f"WHERE {' AND '.join(clauses)}{group}",
                    base_params + list(window)))
        return acc

    def get_measurement(self, ref: ContentHash) -> Measurement | None:
        validate_content_hash(ref, path="measurement key")
        with self._measurements_conn_lock:
            conn = self._measurements_db()
            cur = conn.execute(
                "SELECT * FROM measurements WHERE ref = ?", (ref,)
            )
            row = cur.fetchone()
        if row is None:
            return None
        return self._measurement_from_row(row)

    def list_measurement_refs(self) -> Iterator[ContentHash]:
        conn = self._measurements_db()
        for (ref,) in conn.execute("SELECT ref FROM measurements"):
            yield ContentHash(ref)

    # ------------------------------------------------------------------
    # Spec manifests
    # ------------------------------------------------------------------

    def _versions_dir(self) -> Path:
        """Return the constructor-bound Spec directory or fail closed."""

        versions = self.spec_dir / "versions"
        try:
            current = versions.lstat()
        except OSError as exc:
            raise ConcurrentSpecError(
                f"Corpus versions authority is unavailable: {versions} ({exc})"
            ) from exc
        identity = (current.st_dev, current.st_ino)
        if (
            not stat.S_ISDIR(current.st_mode)
            or identity != self._versions_directory_identity
        ):
            raise ConcurrentSpecError(
                "Corpus versions authority changed after construction: "
                f"{versions}"
            )
        return versions

    def put_spec(self, spec: Spec) -> ContentHash:
        """Publish one fully validated Spec manifest.

        The cross-process lock covers lineage validation, reference
        validation, and the manifest rename.  Direct callers must persist all
        referenced records first.  New workflow code should prefer
        :meth:`commit`, which performs both operations under this lock.
        """

        with self._spec_write_guard():
            return self._put_spec_locked(spec, latest=self.latest_spec())

    @contextmanager
    def _spec_write_guard(self):
        versions_dir = self._versions_dir()
        lock_path = versions_dir / ".spec-write.lock"
        lock_key = lock_path.resolve()
        active_authorities = _active_spec_authority_paths()
        if lock_key in active_authorities:
            raise RuntimeError(
                "Spec writes are forbidden inside locked_spec_authority()"
            )
        # ``flock`` serializes processes; the RLock also serializes threads
        # because POSIX record locks are process-scoped.
        owner_pid = os.getpid()
        with (
            _SPEC_FORK_GUARD,
            _SPEC_WRITE_LOCK,
            lock_path.open("a+b") as lock_stream,
        ):
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                # A forked child shares this open-file description. Unlocking
                # it there would revoke the still-live parent's authority;
                # closing the child's duplicate on context unwind is enough.
                if os.getpid() == owner_pid:
                    fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)

    def retained_spec_authority(self) -> RetainedSpecAuthority:
        """Return a Spec authority whose cleanup can be retained before entry.

        Callers composing this guard with another authority should register
        ``result.retry_cleanup`` before invoking ``result.__enter__``.  This
        closes the asynchronous-exception boundary between the callee's
        acquired return and the caller publishing its own cleanup callback.
        """

        return RetainedSpecAuthority(self)

    @contextmanager
    def locked_spec_authority(self) -> Iterator[Spec | None]:
        """Yield the exact latest Spec while excluding concurrent writers.

        This is the public guard for a caller whose non-Corpus side effect must
        be authorized by one immutable Spec snapshot.  The exclusive lock is
        held for the complete context, so callers must keep the region bounded
        and must not invoke ``put_spec`` or ``commit`` from inside it; such a
        same-thread write raises ``RuntimeError`` instead of deadlocking on a
        second ``flock`` acquisition.

        For composition across an asynchronous interruption boundary, use
        :meth:`retained_spec_authority` and pre-register ``retry_cleanup``
        before entry. This ordinary convenience context preserves cross-Corpus
        nesting on every supported Python runtime.
        """

        with self._spec_write_guard():
            lock_key = (self._versions_dir() / ".spec-write.lock").resolve()
            active_authorities = _active_spec_authority_paths()
            _SPEC_AUTHORITY_GUARDS.paths = active_authorities | {lock_key}
            try:
                yield self.latest_spec()
            finally:
                _SPEC_AUTHORITY_GUARDS.paths = active_authorities

    def _put_spec_locked(
        self, spec: Spec, *, latest: Spec | None
    ) -> ContentHash:
        validate_dataclass_instance(spec)
        for constraint in spec.constraints:
            if (
                isinstance(constraint, ActivationConstraint)
                and constraint.status == "parked"
            ):
                raise ValueError(
                    "cannot publish a parked Spec manifest: "
                    f"{constraint.reason}"
                )
        versions_dir = self._versions_dir()
        if spec.version > 0:
            try:
                predecessor, persisted_parent = self._load_spec(
                    spec.version - 1
                )
            except FileNotFoundError:
                pass
            else:
                hydrated_parent = ContentHash(hash_record(predecessor))
                hydrated_root = predecessor.root or hydrated_parent
                if (
                    hydrated_parent != persisted_parent
                    and spec.parent == hydrated_parent
                    and spec.root == hydrated_root
                ):
                    spec = dataclasses.replace(
                        spec,
                        parent=persisted_parent,
                        root=predecessor.root or persisted_parent,
                    )
        try:
            existing, existing_digest = self._load_spec(spec.version)
        except FileNotFoundError:
            existing = None
        if existing is not None:
            if existing == spec:
                return existing_digest
            raise ConcurrentSpecError(
                f"spec version {spec.version} was already committed by a "
                "concurrent writer with different canonical content; "
                "refusing to overwrite history"
            )

        if latest is None:
            if spec.version != 0 or spec.parent is not None or spec.root is not None:
                raise ValueError(
                    "the first Spec manifest must be version 0 with "
                    "parent=None and root=None"
                )
        else:
            expected_version = latest.version + 1
            if spec.version != expected_version:
                raise ConcurrentSpecError(
                    f"cannot commit Spec v{spec.version}; latest is "
                    f"v{latest.version}, so the only valid next version is "
                    f"v{expected_version}"
                )
            persisted_latest, expected_parent = self._load_spec(latest.version)
            if persisted_latest != latest:
                raise ConcurrentSpecError(
                    "latest Spec changed before lineage publication"
                )
            if spec.parent != expected_parent:
                raise ConcurrentSpecError(
                    f"Spec v{spec.version} parent {spec.parent!r} does not "
                    f"match latest v{latest.version} persisted hash "
                    f"{expected_parent!r}; "
                    "rebase before committing"
                )
            expected_root = latest.root or expected_parent
            if spec.root != expected_root:
                raise ValueError(
                    f"Spec v{spec.version} root {spec.root!r} does not match "
                    f"lineage root {expected_root!r}"
                )

        canonical = serialize(spec)
        short = hashlib.sha256(canonical).hexdigest()[:12]
        manifest_path = versions_dir / f"v{spec.version}_{short}.json"
        manifest_bytes = serialize(parse(canonical))
        if len(manifest_bytes) > _SPEC_MANIFEST_MAX_BYTES:
            raise ValueError(
                "Spec canonical manifest exceeds the supported "
                f"{_SPEC_MANIFEST_MAX_BYTES}-byte bound"
            )
        digest = ContentHash(
            "sha256:" + hashlib.sha256(manifest_bytes).hexdigest()
        )
        self._validate_spec_references(spec, previous=latest)
        _atomic_write(manifest_path, manifest_bytes)
        return digest

    def _validate_spec_references(
        self, spec: Spec, *, previous: Spec | None
    ) -> None:
        """Reject malformed, missing, wrong-kind, or rewritten references."""

        for field_name in ("parent", "root"):
            value = getattr(spec, field_name)
            if value is not None:
                validate_content_hash(value, path=f"Spec.{field_name}")
        for index, ref in enumerate(spec.depends_on):
            # Cross-Spec dependencies may live in another Corpus.  Their
            # address is validated here; existence is resolved by the owner.
            validate_content_hash(ref, path=f"Spec.depends_on[{index}]")

        bindings: tuple[tuple[str, str, type], ...] = (
            ("implementation_refs", "implementations", Implementation),
            ("verdict_refs", "verdicts", Verdict),
            ("measurement_refs", "measurements", Measurement),
            ("lifecycle_event_refs", "lifecycle_events", LifecycleEvent),
        )
        for field_name, kind, record_type in bindings:
            refs = getattr(spec, field_name)
            prior_refs = getattr(previous, field_name) if previous else ()
            if refs[: len(prior_refs)] != prior_refs:
                raise ValueError(
                    f"Spec.{field_name} is append-only; v{spec.version} "
                    "removed, reordered, or replaced an existing reference"
                )
            for index, ref in enumerate(refs):
                validate_content_hash(ref, path=f"Spec.{field_name}[{index}]")
            self._require_records(
                refs,
                kind=kind,
                record_type=record_type,
                newly_added=refs[len(prior_refs) :],
            )

    def _require_records(
        self,
        refs: tuple[ContentHash, ...],
        *,
        kind: str,
        record_type: type,
        newly_added: tuple[ContentHash, ...],
    ) -> None:
        if kind == "measurements":
            missing = self._missing_measurement_refs(refs)
            if missing:
                raise ValueError(
                    "Spec.measurement_refs contains missing records: "
                    f"{sorted(missing)}"
                )
            # New rows also pass through the typed decoder so a malformed
            # nested ContentHash cannot become reachable from a Spec.
            for ref in dict.fromkeys(newly_added):
                self.get_record(ref, as_type=record_type)
            return

        new_set = set(newly_added)
        for ref in dict.fromkeys(refs):
            blob = self._read_kind_artifact(ref, kind=kind)
            if ref in new_set:
                decode_dataclass(parse(blob), record_type)

    def _missing_measurement_refs(
        self, refs: tuple[ContentHash, ...]
    ) -> set[ContentHash]:
        wanted = set(refs)
        if not wanted:
            return set()
        found: set[ContentHash] = set()
        values = tuple(wanted)
        with self._measurements_conn_lock:
            conn = self._measurements_db()
            for offset in range(0, len(values), 500):
                window = values[offset : offset + 500]
                placeholders = ",".join("?" for _ in window)
                for (ref,) in conn.execute(
                    f"SELECT ref FROM measurements WHERE ref IN ({placeholders})",
                    list(window),
                ):
                    found.add(ContentHash(ref))
        return wanted - found

    def commit(
        self,
        *,
        implementations: Iterable[Implementation] = (),
        verdicts: Iterable[Verdict] = (),
        measurements: Iterable[Measurement] = (),
        lifecycle_events: Iterable[LifecycleEvent] = (),
        lifecycle_event_factory: Callable[[int], LifecycleEvent] | None = None,
        record_refs: CommitRefs | None = None,
        transform: Callable[[Spec], Spec] | None = None,
    ) -> CommitResult:
        """Persist records and publish one rebased Spec manifest.

        The latest Spec is read only after acquiring the same lock used by
        ``put_spec``.  All records are content-addressed before the candidate
        manifest is built, and that manifest is the sole visibility point.
        A process death can leave unreachable records, but never a Spec that
        references an absent record.  Repeated records are idempotent and are
        not appended twice.

        ``lifecycle_event_factory`` is evaluated under the Spec write lock and
        receives the exact version this transaction will publish.  This lets a
        structural authority change, its evidence records, and its terminal
        lifecycle marker share one visibility point. ``record_refs`` attaches
        records that were already persisted by an upstream primitive or bulk
        loader. ``transform`` receives the already-versioned candidate,
        allowing a caller to replace inline structural data with
        ``dataclasses.replace`` without creating a second Spec version. It
        must preserve lineage and all staged refs.
        """

        staged = (
            tuple(implementations),
            tuple(verdicts),
            tuple(measurements),
            tuple(lifecycle_events),
        )
        with self._spec_write_guard():
            latest = self.latest_spec()
            if latest is None:
                raise RuntimeError("cannot commit records: corpus has no Spec")
            lifecycle_records = staged[3]
            if lifecycle_event_factory is not None:
                event = lifecycle_event_factory(latest.version + 1)
                if not isinstance(event, LifecycleEvent):
                    raise TypeError(
                        "lifecycle_event_factory must return LifecycleEvent, "
                        f"got {type(event).__name__}"
                    )
                lifecycle_records += (event,)
            return self._commit_records_locked(
                latest,
                implementations=staged[0],
                verdicts=staged[1],
                measurements=staged[2],
                lifecycle_events=lifecycle_records,
                record_refs=record_refs,
                transform=transform,
            )

    def commit_lifecycle_event(
        self, factory: Callable[[int], LifecycleEvent]
    ) -> tuple[CommitResult, LifecycleEvent]:
        """Create and commit a lifecycle event against the race-correct version."""

        with self._spec_write_guard():
            latest = self.latest_spec()
            if latest is None:
                raise RuntimeError(
                    "cannot commit lifecycle event: corpus has no Spec"
                )
            event = factory(latest.version + 1)
            if not isinstance(event, LifecycleEvent):
                raise TypeError(
                    "lifecycle factory must return LifecycleEvent, got "
                    f"{type(event).__name__}"
                )
            result = self._commit_records_locked(
                latest, lifecycle_events=(event,)
            )
            return result, event

    def _commit_records_locked(
        self,
        latest: Spec,
        *,
        implementations: tuple[Implementation, ...] = (),
        verdicts: tuple[Verdict, ...] = (),
        measurements: tuple[Measurement, ...] = (),
        lifecycle_events: tuple[LifecycleEvent, ...] = (),
        record_refs: CommitRefs | None = None,
        transform: Callable[[Spec], Spec] | None = None,
    ) -> CommitResult:
        supplied = record_refs or CommitRefs()
        if not isinstance(supplied, CommitRefs):
            raise TypeError(
                f"record_refs must be CommitRefs, got {type(supplied).__name__}"
            )
        validate_dataclass_instance(supplied)
        refs = CommitRefs(
            implementation_refs=supplied.implementation_refs + tuple(
                self.put_record(record, kind="implementations")
                for record in implementations
            ),
            verdict_refs=supplied.verdict_refs + tuple(
                self.put_record(record, kind="verdicts") for record in verdicts
            ),
            measurement_refs=supplied.measurement_refs + (
                tuple(self.put_measurements(measurements))
                if measurements
                else ()
            ),
            lifecycle_event_refs=supplied.lifecycle_event_refs + tuple(
                self.put_record(record, kind="lifecycle_events")
                for record in lifecycle_events
            ),
        )

        missing = CommitRefs(
            implementation_refs=_missing_refs(
                refs.implementation_refs, latest.implementation_refs
            ),
            verdict_refs=_missing_refs(refs.verdict_refs, latest.verdict_refs),
            measurement_refs=_missing_refs(
                refs.measurement_refs, latest.measurement_refs
            ),
            lifecycle_event_refs=_missing_refs(
                refs.lifecycle_event_refs, latest.lifecycle_event_refs
            ),
        )
        has_new_refs = any(dataclasses.astuple(missing))
        if not has_new_refs and transform is None:
            return CommitResult(spec=latest, refs=refs)

        candidate = latest.with_appended_refs(
            implementation_refs=missing.implementation_refs,
            verdict_refs=missing.verdict_refs,
            measurement_refs=missing.measurement_refs,
            lifecycle_event_refs=missing.lifecycle_event_refs,
        )
        persisted_latest, persisted_parent = self._load_spec(latest.version)
        if persisted_latest != latest:
            raise ConcurrentSpecError(
                "latest Spec changed before transaction publication"
            )
        candidate = dataclasses.replace(
            candidate,
            parent=persisted_parent,
            root=latest.root or persisted_parent,
        )
        if transform is not None:
            candidate = transform(candidate)
            if not isinstance(candidate, Spec):
                raise TypeError(
                    "commit transform must return Spec, got "
                    f"{type(candidate).__name__}"
                )
        _assert_commit_refs_present(candidate, refs)
        self._put_spec_locked(candidate, latest=latest)
        return CommitResult(spec=candidate, refs=refs)

    def _load_spec(self, version: int) -> tuple[Spec, ContentHash]:
        versions_dir = self._versions_dir()
        candidates = list(versions_dir.glob(f"v{version}_*.json"))
        if not candidates:
            raise FileNotFoundError(
                f"no spec manifest for version {version} under {versions_dir}"
            )
        # Identical content collapses to one hash-derived filename, so more
        # than one manifest for a version is a divergent lineage. Refuse to
        # choose one by mtime and silently hide the other.
        if len(candidates) > 1:
            raise ConcurrentSpecError(
                f"spec version {version} has {len(candidates)} divergent "
                f"manifests ({sorted(p.name for p in candidates)}) — concurrent "
                f"writers committed conflicting specs; one writer's refs would "
                f"be lost. Resolve by choosing the authoritative manifest or "
                f"merging their refs into a fresh version."
            )
        path = candidates[0]
        try:
            before = path.lstat()
        except OSError as exc:
            raise ConcurrentSpecError(
                f"spec manifest cannot be inspected: {path} ({exc})"
            ) from exc
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or before.st_nlink != 1
            or before.st_size > _SPEC_MANIFEST_MAX_BYTES
        ):
            raise ConcurrentSpecError(
                "spec manifest must be one bounded, singly linked regular "
                f"file: {path}"
            )
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        try:
            descriptor = os.open(path, flags)
        except OSError as exc:
            raise ConcurrentSpecError(
                f"spec manifest cannot be opened safely: {path} ({exc})"
            ) from exc
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        try:
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_nlink != 1
                or opened.st_size > _SPEC_MANIFEST_MAX_BYTES
                or any(
                    getattr(before, field) != getattr(opened, field)
                    for field in stable_fields
                )
            ):
                raise ConcurrentSpecError(
                    f"spec manifest changed while opening: {path}"
                )
            chunks: list[bytes] = []
            observed = 0
            while True:
                chunk = os.read(descriptor, 64 * 1024)
                if not chunk:
                    break
                observed += len(chunk)
                if observed > _SPEC_MANIFEST_MAX_BYTES:
                    raise ConcurrentSpecError(
                        f"spec manifest exceeded its size bound: {path}"
                    )
                chunks.append(chunk)
            finished = os.fstat(descriptor)
        except OSError as exc:
            raise ConcurrentSpecError(
                f"spec manifest cannot be read safely: {path} ({exc})"
            ) from exc
        finally:
            os.close(descriptor)
        try:
            current = path.lstat()
        except OSError as exc:
            raise ConcurrentSpecError(
                f"spec manifest disappeared during inspection: {path} ({exc})"
            ) from exc
        if any(
            getattr(opened, field) != getattr(finished, field)
            or getattr(opened, field) != getattr(current, field)
            for field in stable_fields
        ):
            raise ConcurrentSpecError(
                f"spec manifest changed during inspection: {path}"
            )
        current_candidates = list(versions_dir.glob(f"v{version}_*.json"))
        if len(current_candidates) != 1 or current_candidates[0] != path:
            raise ConcurrentSpecError(
                f"spec version {version} changed manifest topology during "
                "inspection"
            )
        blob = b"".join(chunks)
        d = parse(blob)
        spec = self._spec_from_dict(d)
        canonical = serialize(parse(blob))
        expected_name = (
            f"v{version}_{hashlib.sha256(canonical).hexdigest()[:12]}.json"
        )
        if (
            spec.version != version
            or path.name != expected_name
            or blob != canonical
        ):
            raise ConcurrentSpecError(
                "spec manifest content does not match its exact canonical "
                f"version/hash filename: {path.name!r}"
            )
        digest = ContentHash("sha256:" + hashlib.sha256(blob).hexdigest())
        return spec, digest

    def get_spec(self, version: int) -> Spec:
        spec, _digest = self._load_spec(version)
        return spec

    def get_spec_with_digest(self, version: int) -> tuple[Spec, ContentHash]:
        """Return one safely loaded Spec and its persisted raw SHA-256.

        The digest authenticates the exact canonical bytes stored in the
        manifest, not a possibly schema-enriched reserialization of the
        hydrated dataclass. Consumers that bind lineage or external side
        effects should use this atomic pair rather than ``hash_record(spec)``.
        """

        return self._load_spec(version)

    def latest_spec(self) -> Spec | None:
        versions_dir = self._versions_dir()
        best: int | None = None
        versions: set[int] = set()
        for p in versions_dir.iterdir():
            m = _VERSION_FILE_RE.match(p.name)
            if m is None:
                if _VERSION_FILE_LIKE_RE.match(p.name) is not None:
                    raise ConcurrentSpecError(
                        "malformed Spec-like manifest filename cannot be "
                        f"ignored: {p.name!r}"
                    )
                continue
            version_text = m.group(1)
            if len(version_text) > 19:
                raise ConcurrentSpecError(
                    f"Spec manifest version is outside the supported bound: "
                    f"{p.name!r}"
                )
            n = int(version_text)
            if n > _SPEC_MAX_VERSION:
                raise ConcurrentSpecError(
                    f"Spec manifest version is outside the supported bound: "
                    f"{p.name!r}"
                )
            versions.add(n)
            if best is None or n > best:
                best = n
        if best is None:
            return None
        if min(versions) != 0 or best != len(versions) - 1:
            raise ConcurrentSpecError(
                "Spec lineage is not contiguous from v0 through "
                f"v{best}: found {sorted(versions)}"
            )

        previous: Spec | None = None
        previous_digest: ContentHash | None = None
        for version in range(best + 1):
            current, current_digest = self._load_spec(version)
            if previous is None:
                if (
                    current.version != 0
                    or current.parent is not None
                    or current.root is not None
                ):
                    raise ConcurrentSpecError(
                        "Spec lineage root must be v0 with parent=None and "
                        "root=None"
                    )
            else:
                assert previous_digest is not None
                if current.parent != previous_digest:
                    raise ConcurrentSpecError(
                        f"Spec v{version} does not extend the exact canonical "
                        f"v{version - 1} lineage"
                    )
                expected_root = previous.root or current.parent
                if current.root != expected_root:
                    raise ConcurrentSpecError(
                        f"Spec v{version} does not extend the exact canonical "
                        f"v{version - 1} lineage"
                    )
                for field_name in (
                    "implementation_refs",
                    "verdict_refs",
                    "measurement_refs",
                    "lifecycle_event_refs",
                ):
                    prior_refs = getattr(previous, field_name)
                    current_refs = getattr(current, field_name)
                    if current_refs[: len(prior_refs)] != prior_refs:
                        raise ConcurrentSpecError(
                            f"Spec v{version} rewrites append-only "
                            f"{field_name} history"
                        )
            previous = current
            previous_digest = current_digest
        assert previous is not None
        return previous

    # ------------------------------------------------------------------
    # Artifacts
    # ------------------------------------------------------------------

    def put_artifact(
        self,
        blob: bytes,
        *,
        kind: str,
        name_hint: str | None = None,
    ) -> ContentHash:
        """Store bytes by SHA-256 in a trusted artifact namespace.

        ``kind`` is joined directly below ``artifacts/`` and is not validated;
        callers must supply one safe path component, never untrusted input or
        traversal syntax. ``name_hint`` is informational and does not affect
        storage identity or filename.
        """
        # Content-addressed storage: filename is always
        # ``<hex>.bin``, regardless of ``name_hint``. The hint is
        # informational only — using it as a filename suffix breaks
        # idempotence when callers pass differing hints for identical
        # bytes (would create duplicates). Fixed-extension write is
        # the only safe approach for a content-addressed store.
        hex_ = hashlib.sha256(blob).hexdigest()
        parent = self.spec_dir / "artifacts" / kind
        parent.mkdir(parents=True, exist_ok=True)
        path = parent / f"{hex_}.bin"
        if path.exists() and path.read_bytes() == blob:
            return ContentHash(f"sha256:{hex_}")
        _atomic_write(path, blob)
        return ContentHash(f"sha256:{hex_}")

    def put_record(self, record, *, kind: str) -> ContentHash:
        """Validate and persist a typed lifecycle record.

        Measurements route through the SQLite measurements table at
        ``<spec_dir>/measurements.sqlite``; everything else lands at
        ``artifacts/<kind>/<hex>.bin``. The returned ``ContentHash``
        equals ``hash_record(record)`` either way, so spec-side
        composition (``spec.with_verdict(v)`` / ``with_measurement(m)``)
        keeps matching. ``kind`` and the concrete record class must agree;
        arbitrary dataclasses are not accepted into a lifecycle namespace.
        """
        expected_type = _RECORD_KIND_TYPES.get(kind)
        if expected_type is None:
            raise ValueError(
                f"unknown record kind {kind!r}; expected one of "
                f"{sorted(_RECORD_KIND_TYPES)}"
            )
        if not isinstance(record, expected_type):
            raise TypeError(
                f"kind={kind!r} requires {expected_type.__name__}; got "
                f"{type(record).__name__}"
            )
        validate_dataclass_instance(record)
        if kind == "measurements":
            return self._put_measurement(record)
        return self.put_artifact(serialize(record), kind=kind)

    def get_artifact(self, key: ContentHash) -> bytes:
        # Defensive lookup: stem-match the requested hex AND verify
        # the returned bytes' sha256 matches the key. Tampered /
        # corrupted / planted files are treated as misses.
        validated = validate_content_hash(key, path="artifact key")
        hex_ = validated.split(":", 1)[1]
        artifacts_dir = self.spec_dir / "artifacts"
        if not artifacts_dir.is_dir():
            raise FileNotFoundError(f"no artifact for key {key!r}")
        for p in artifacts_dir.rglob("*"):
            if not (p.is_file() and p.stem == hex_):
                continue
            # Skip stale in-flight tempfiles (_atomic_write leftovers).
            if p.name.startswith(".tmp_"):
                continue
            data = p.read_bytes()
            if hashlib.sha256(data).hexdigest() != hex_:
                # Tampered / planted — keep looking for a clean copy.
                continue
            return data
        raise FileNotFoundError(f"no artifact for key {key!r}")

    def get_artifact_bounded(
        self,
        key: ContentHash,
        *,
        max_bytes: int,
    ) -> bytes:
        """Return one authenticated artifact without an unbounded read.

        Candidate files are discovered and opened relative to a held
        ``artifacts/`` directory descriptor. Symlinked directories and
        non-regular or multiply linked candidates are never followed. The
        byte ceiling is enforced from stable metadata and again while the
        SHA-256 digest is streamed, before any complete candidate is retained
        in memory.

        This preserves ``get_artifact``'s content-addressed lookup across
        artifact kinds while giving control-plane consumers a safe boundary
        for small authority records. A matching-stem candidate above the
        ceiling raises ``ValueError`` when no valid bounded copy exists;
        missing or corrupt bounded candidates raise ``FileNotFoundError``.
        """

        if (
            isinstance(max_bytes, bool)
            or not isinstance(max_bytes, int)
            or max_bytes < 0
        ):
            raise ValueError("artifact max_bytes must be a non-negative integer")
        validated = validate_content_hash(key, path="artifact key")
        hex_ = validated.split(":", 1)[1]
        artifacts_dir = self.spec_dir / "artifacts"
        try:
            root_metadata = artifacts_dir.lstat()
        except OSError as exc:
            raise FileNotFoundError(f"no artifact for key {key!r}") from exc
        if (
            not stat.S_ISDIR(root_metadata.st_mode)
            or stat.S_ISLNK(root_metadata.st_mode)
        ):
            raise FileNotFoundError(f"no artifact for key {key!r}")

        directory_flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        file_flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        try:
            root_fd = os.open(artifacts_dir, directory_flags)
        except OSError as exc:
            raise FileNotFoundError(f"no artifact for key {key!r}") from exc
        try:
            opened_root = os.fstat(root_fd)
        except OSError as exc:
            os.close(root_fd)
            raise FileNotFoundError(f"no artifact for key {key!r}") from exc
        if (
            not stat.S_ISDIR(opened_root.st_mode)
            or (opened_root.st_dev, opened_root.st_ino)
            != (root_metadata.st_dev, root_metadata.st_ino)
        ):
            os.close(root_fd)
            raise FileNotFoundError(f"no artifact for key {key!r}")

        stable_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        entries_seen = 0
        oversized = False

        def inspect_candidate(parent_fd: int, name: str) -> bytes | None:
            nonlocal oversized
            try:
                before = os.stat(
                    name,
                    dir_fd=parent_fd,
                    follow_symlinks=False,
                )
            except OSError:
                return None
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
            ):
                return None
            if before.st_size > max_bytes:
                oversized = True
                return None
            descriptor: int | None = None
            try:
                descriptor = os.open(
                    name,
                    file_flags,
                    dir_fd=parent_fd,
                )
                opened = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or opened.st_nlink != 1
                    or any(
                        getattr(before, field) != getattr(opened, field)
                        for field in stable_fields
                    )
                ):
                    return None
                digest = hashlib.sha256()
                chunks: list[bytes] = []
                observed = 0
                while True:
                    chunk = os.read(
                        descriptor,
                        min(1024 * 1024, max_bytes + 1),
                    )
                    if not chunk:
                        break
                    observed += len(chunk)
                    if observed > max_bytes:
                        oversized = True
                        return None
                    digest.update(chunk)
                    chunks.append(chunk)
                finished = os.fstat(descriptor)
                try:
                    current = os.stat(
                        name,
                        dir_fd=parent_fd,
                        follow_symlinks=False,
                    )
                except OSError:
                    return None
                if any(
                    getattr(opened, field) != getattr(finished, field)
                    or getattr(opened, field) != getattr(current, field)
                    for field in stable_fields
                ):
                    return None
                if digest.hexdigest() != hex_:
                    return None
                return b"".join(chunks)
            except OSError:
                return None
            finally:
                if descriptor is not None:
                    os.close(descriptor)

        def walk(directory_fd: int, depth: int) -> bytes | None:
            nonlocal entries_seen
            if depth > _BOUNDED_ARTIFACT_MAX_DEPTH:
                raise ValueError(
                    "artifact tree exceeds its traversal depth ceiling"
                )
            try:
                with os.scandir(directory_fd) as iterator:
                    for entry in iterator:
                        entries_seen += 1
                        if entries_seen > _BOUNDED_ARTIFACT_MAX_ENTRIES:
                            raise ValueError(
                                "artifact tree exceeds its entry ceiling"
                            )
                        name = entry.name
                        try:
                            metadata = os.stat(
                                name,
                                dir_fd=directory_fd,
                                follow_symlinks=False,
                            )
                        except OSError:
                            continue
                        if (
                            stat.S_ISREG(metadata.st_mode)
                            and Path(name).stem == hex_
                        ):
                            result = inspect_candidate(directory_fd, name)
                            if result is not None:
                                return result
                            continue
                        if (
                            not stat.S_ISDIR(metadata.st_mode)
                            or stat.S_ISLNK(metadata.st_mode)
                        ):
                            continue
                        child_fd: int | None = None
                        try:
                            child_fd = os.open(
                                name,
                                directory_flags,
                                dir_fd=directory_fd,
                            )
                            opened = os.fstat(child_fd)
                            if (
                                not stat.S_ISDIR(opened.st_mode)
                                or (opened.st_dev, opened.st_ino)
                                != (metadata.st_dev, metadata.st_ino)
                            ):
                                continue
                            result = walk(child_fd, depth + 1)
                            if result is not None:
                                return result
                        except OSError:
                            continue
                        finally:
                            if child_fd is not None:
                                os.close(child_fd)
            except OSError as exc:
                raise FileNotFoundError(
                    f"no artifact for key {key!r}"
                ) from exc
            return None

        try:
            result = walk(root_fd, 0)
        finally:
            os.close(root_fd)
        if result is not None:
            return result
        if oversized:
            raise ValueError(
                f"artifact for key {key!r} exceeds {max_bytes} bytes"
            )
        raise FileNotFoundError(f"no artifact for key {key!r}")

    def _read_kind_artifact(self, key: ContentHash, *, kind: str) -> bytes:
        """Read one content-addressed record from its declared namespace."""

        validated = validate_content_hash(key, path=f"{kind} record key")
        hex_ = validated.split(":", 1)[1]
        path = self.spec_dir / "artifacts" / kind / f"{hex_}.bin"
        if not path.is_file():
            raise ValueError(
                f"missing {kind} record {validated!r} at {path}"
            )
        blob = path.read_bytes()
        if hashlib.sha256(blob).hexdigest() != hex_:
            raise ValueError(
                f"corrupt {kind} record {validated!r}: content hash mismatch"
            )
        return blob

    def get_record(self, key: ContentHash, *, as_type: type) -> object:
        validate_content_hash(key, path="record key")
        kind = _RECORD_TYPE_KINDS.get(as_type)
        if kind is None:
            raise TypeError(
                f"unsupported record type {as_type!r}; expected one of "
                f"{[item.__name__ for item in _RECORD_TYPE_KINDS]}"
            )
        if as_type is Measurement:
            m = self.get_measurement(key)
            if m is None:
                raise FileNotFoundError(f"no measurement for key {key!r}")
            return m
        blob = self._read_kind_artifact(key, kind=kind)
        return decode_dataclass(parse(blob), as_type)

    def list_artifacts(self, *, kind: str) -> Iterable[ContentHash]:
        """Yield hashes from a trusted single-component artifact namespace.

        As with ``put_artifact``, ``kind`` is not path-sanitized and must not
        come from untrusted input.
        """
        if kind == "measurements":
            yield from self.list_measurement_refs()
            return
        kind_dir = self.spec_dir / "artifacts" / kind
        if not kind_dir.exists():
            return
        emitted: set[str] = set()
        for p in kind_dir.iterdir():
            if not p.is_file():
                continue
            # Skip in-flight tempfiles (_atomic_write leftovers / racing writers).
            if p.name.startswith(".tmp_"):
                continue
            # Skip anything whose stem isn't a 64-char lowercase-hex sha256.
            stem = p.stem
            if len(stem) != 64 or any(c not in "0123456789abcdef" for c in stem):
                continue
            if stem in emitted:
                continue
            emitted.add(stem)
            yield ContentHash(f"sha256:{stem}")

    # ------------------------------------------------------------------
    # Runs
    # ------------------------------------------------------------------

    def open_run(
        self, recipe: str, *, idempotency_key: str | None = None
    ) -> RunHandle:
        if not isinstance(recipe, str) or not _RUN_RECIPE_RE.fullmatch(recipe):
            raise ValueError(
                "recipe must match [A-Za-z0-9][A-Za-z0-9_.-]{0,127}; "
                f"got {recipe!r}"
            )
        ts = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ"
        )
        # Use a 32-bit random tag and retry the atomic mkdir on the unlikely
        # collision before surfacing allocation failure.
        runs_dir = self.spec_dir / "runs"
        if idempotency_key is not None:
            if not isinstance(idempotency_key, str) or not idempotency_key:
                raise ValueError("idempotency_key must be a non-empty string")
            digest = hashlib.sha256(
                f"{recipe}\0{idempotency_key}".encode("utf-8")
            ).hexdigest()[:24]
            run_dir = runs_dir / f"idempotent_{recipe}_{digest}"
            run_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
            # ``mode`` is ignored when an idempotent directory already
            # exists. Reassert the privacy invariant on every open.
            run_dir.chmod(0o700)
            metadata_path = run_dir / "run.json"
            metadata = {
                "recipe": recipe,
                "started_at": ts,
                "idempotency_digest": digest,
            }
            _create_once(
                metadata_path,
                json.dumps(
                    metadata, sort_keys=True, separators=(",", ":")
                ).encode("utf-8"),
            )
            try:
                stored = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"open_run: invalid idempotency metadata at {metadata_path}"
                ) from exc
            if stored.get("recipe") != recipe or stored.get(
                "idempotency_digest"
            ) != digest:
                raise RuntimeError(
                    f"open_run: idempotency collision at {metadata_path}"
                )
            return RunHandle(
                run_dir=run_dir,
                recipe=recipe,
                started_at=str(stored.get("started_at") or ts),
            )
        for _ in range(8):
            tag = secrets.token_hex(4)
            run_dir = runs_dir / f"{ts}_{recipe}_{tag}"
            try:
                run_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
            except FileExistsError:
                continue
            run_dir.chmod(0o700)
            return RunHandle(run_dir=run_dir, recipe=recipe, started_at=ts)
        raise RuntimeError(
            f"open_run: could not allocate a unique run dir under {runs_dir} "
            f"after 8 attempts (extreme same-second contention?)"
        )

    # ------------------------------------------------------------------
    # Internal reconstruction helpers
    # ------------------------------------------------------------------

    def _spec_from_dict(self, d: dict) -> Spec:
        return decode_dataclass(d, Spec)


def _atomic_write(path: Path, data: bytes) -> None:
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(parent), prefix=".tmp_", suffix=path.suffix or ".tmp"
    )
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
        try:
            dir_fd = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            # Some filesystems do not support fsync on directory handles.
            pass
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _missing_refs(
    requested: tuple[ContentHash, ...], existing: tuple[ContentHash, ...]
) -> tuple[ContentHash, ...]:
    seen = set(existing)
    missing: list[ContentHash] = []
    for ref in requested:
        if ref in seen:
            continue
        seen.add(ref)
        missing.append(ref)
    return tuple(missing)


def _assert_commit_refs_present(spec: Spec, refs: CommitRefs) -> None:
    bindings = (
        ("implementation_refs", refs.implementation_refs),
        ("verdict_refs", refs.verdict_refs),
        ("measurement_refs", refs.measurement_refs),
        ("lifecycle_event_refs", refs.lifecycle_event_refs),
    )
    for field_name, staged in bindings:
        published = getattr(spec, field_name)
        absent = tuple(ref for ref in staged if ref not in published)
        if absent:
            raise ValueError(
                f"commit transform dropped staged {field_name}: {absent}"
            )
