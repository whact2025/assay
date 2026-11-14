# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import dataclasses
import fcntl
import hashlib
import os
import select
import signal
import stat
import sys
import threading
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path

import pytest

from assay import (
    BenchConfig,
    ContentHash,
    ExecutionContext,
    Implementation,
    ImplementationDirective,
    LifecycleEvent,
    Measurement,
    MutationRegion,
    PlanPolicy,
    Slice,
    Spec,
    Tier,
    TierDAG,
)
from assay.canonical import hash_record, parse, serialize
from assay.corpus import (
    CommitRefs,
    Corpus,
    RecordDecodeError,
    RunHandle,
    SpecAuthorityCleanupError,
)
from assay.corpus.codec import decode_dataclass


MINIMAL_YAML = """
schema_version: 1
project_context:
  project: assay
  repo_url: https://github.com/whact2025/assay.git
  base_commit: "0000000"
  branch: main
  submodule_path: null
  runners: {}
intent: |
  Stand up the foundation phase.
constraints:
  - kind: budget
    usd_limit: 200
done_criteria:
  - kind: parity
    reference_commit: abc
    test_name: t
"""


def _read_fork_report(descriptor: int, pid: int) -> bytes:
    """Read one child result without letting a broken fork hook hang CI."""

    readable, _, _ = select.select((descriptor,), (), (), 3.0)
    if not readable:
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)
        pytest.fail("forked Spec-authority probe did not report within 3s")
    return os.read(descriptor, 1)


def _execution(run_id: str = "test") -> ExecutionContext:
    return ExecutionContext(
        timestamp="2026-06-28T00:00:00Z",
        machine="test-host",
        cwd="/workspace",
        toolchain={},
        backend=None,
        model=None,
        recipe="test",
        recipe_commit="deadbeef",
        gpu_model=None,
        run_id=run_id,
    )


def _event(kind: str) -> LifecycleEvent:
    return LifecycleEvent(
        kind=kind,
        reason=f"{kind} in test",
        affected_versions=(1,),
        execution=_execution(kind),
    )


def _implementation(corpus: Corpus, slice_id: str = "slice-a") -> Implementation:
    diff_ref = corpus.put_artifact(
        f"diff for {slice_id}".encode(), kind="diffs"
    )
    transcript_ref = corpus.put_artifact(
        f"transcript for {slice_id}".encode(), kind="transcripts"
    )
    stderr_ref = corpus.put_artifact(b"warning", kind="stderr")
    return Implementation(
        slice_id=slice_id,
        base_commit="base",
        result_commit="result",
        diff_ref=diff_ref,
        transcript_ref=transcript_ref,
        files_modified=("src/a.py", "tests/test_a.py"),
        classification="success",
        execution=_execution(slice_id),
        scope_escapes=("unexpected.txt",),
        artifacts={"stderr": stderr_ref},
        cost_usd=1.25,
        input_tokens=123,
        output_tokens=45,
        cache_creation_tokens=23,
        cache_read_tokens=67,
        stop_reason="max_turns",
    )


def test_constructor_creates_layout(tmp_path: Path) -> None:
    c = Corpus(tmp_path / "store")
    for sub in ("versions", "artifacts", "transcripts", "diffs", "traces", "runs"):
        assert (c.spec_dir / sub).is_dir(), sub


def test_put_spec_returns_content_hash(tmp_path: Path) -> None:
    c = Corpus(tmp_path / "store")
    s = Spec.from_yaml(MINIMAL_YAML)
    h = c.put_spec(s)
    assert h.startswith("sha256:")
    assert len(h) == len("sha256:") + 64


def test_put_spec_rejects_parked_activation_constraint(tmp_path: Path) -> None:
    c = Corpus(tmp_path / "store")
    parked_yaml = MINIMAL_YAML.replace(
        "constraints:\n  - kind: budget\n    usd_limit: 200",
        "constraints:\n"
        "  - kind: activation\n"
        "    status: parked\n"
        "    reason: predecessor has not landed",
    )
    spec = Spec.from_yaml(parked_yaml)

    with pytest.raises(ValueError, match="cannot publish a parked Spec"):
        c.put_spec(spec)

    assert c.latest_spec() is None


def test_latest_spec_round_trip_hash_equal(tmp_path: Path) -> None:
    c = Corpus(tmp_path / "store")
    s = Spec.from_yaml(MINIMAL_YAML)
    c.put_spec(s)
    loaded = c.latest_spec()
    assert loaded is not None
    assert hash_record(loaded) == hash_record(s)


def test_spec_codec_rejects_unsupported_schema_version() -> None:
    payload = parse(serialize(Spec.from_yaml(MINIMAL_YAML)))
    payload["schema_version"] = 2

    with pytest.raises(
        RecordDecodeError, match="unsupported Spec schema_version 2"
    ):
        decode_dataclass(payload, Spec)


def test_spec_codec_round_trips_complete_nested_tier_dag(tmp_path: Path) -> None:
    corpus = Corpus(tmp_path / "store")
    base = Spec.from_yaml(MINIMAL_YAML)
    corpus.put_spec(base)
    directive = ImplementationDirective(
        slice_id="slice-a",
        intent="Implement the typed boundary.",
        files_in_scope=("src/a.py",),
        creates_paths=("tests/test_a.py",),
        verify_mode="tests",
        blast_radius={"max_files": 2},
        mutation_regions=(
            MutationRegion(
                file="src/a.py", start_marker="START", end_marker="END"
            ),
        ),
        requirement_refs=(
            "The typed boundary must preserve input ordering.",
            "Invalid input must fail without partial output.",
        ),
    )
    planned = base.with_tier_dag(
        TierDAG(
            tiers=(
                Tier(
                    tier_id="t0",
                    slices=(Slice(id="slice-a", directive=directive),),
                    depends_on=(),
                    smoke_bench_config=BenchConfig(
                        bench_id="smoke",
                        runner="tests.runner",
                        parameters={"samples": 1},
                        env={"MODE": "test"},
                    ),
                ),
            )
        )
    )

    corpus.put_spec(planned)
    restored = corpus.latest_spec()

    assert restored == planned
    assert isinstance(restored.tier_dag, TierDAG)
    assert isinstance(restored.tier_dag.tiers, tuple)
    restored_directive = restored.tier_dag.tiers[0].slices[0].directive
    assert restored_directive.requirement_refs == directive.requirement_refs
    restored_tier = restored.tier_dag.tiers[0]
    assert isinstance(restored_tier.smoke_bench_config, BenchConfig)
    assert isinstance(
        restored_tier.slices[0].directive.mutation_regions[0], MutationRegion
    )

    legacy_payload = parse(serialize(planned))
    legacy_directive = legacy_payload["tier_dag"]["tiers"][0]["slices"][0][
        "directive"
    ]
    del legacy_directive["requirement_refs"]
    restored_legacy = decode_dataclass(legacy_payload, Spec)
    assert (
        restored_legacy.tier_dag.tiers[0].slices[0].directive.requirement_refs
        == ()
    )


def test_latest_spec_none_on_empty(tmp_path: Path) -> None:
    c = Corpus(tmp_path / "store")
    assert c.latest_spec() is None


def test_locked_spec_authority_yields_exact_snapshot_and_blocks_writer(
    tmp_path: Path,
) -> None:
    corpus = Corpus(tmp_path / "store")
    initial = Spec.from_yaml(MINIMAL_YAML)
    corpus.put_spec(initial)
    writer_started = threading.Event()

    def publish_lifecycle():
        writer_started.set()
        return corpus.commit(lifecycle_events=(_event("annotated"),))

    with ThreadPoolExecutor(max_workers=1) as executor:
        with corpus.locked_spec_authority() as locked:
            assert locked == initial
            future = executor.submit(publish_lifecycle)
            assert writer_started.wait(timeout=1.0)
            with pytest.raises(TimeoutError):
                future.result(timeout=0.05)
            assert corpus.latest_spec() == locked

        committed = future.result(timeout=1.0)

    assert committed.spec.version == initial.version + 1
    assert corpus.latest_spec() == committed.spec


def test_locked_spec_authority_rejects_same_thread_writer(tmp_path: Path) -> None:
    corpus = Corpus(tmp_path / "store")
    initial = Spec.from_yaml(MINIMAL_YAML)
    corpus.put_spec(initial)

    with corpus.locked_spec_authority() as locked:
        with pytest.raises(RuntimeError, match="locked_spec_authority"):
            corpus.commit(lifecycle_events=(_event("forbidden"),))
        assert corpus.latest_spec() == locked

    committed = corpus.commit(lifecycle_events=(_event("allowed"),))
    assert committed.spec.version == initial.version + 1


def test_retained_spec_authority_lease_closes_exact_enter_return_interrupt(
    tmp_path: Path,
) -> None:
    """Pre-owned cleanup is live before the source return can be interrupted."""

    from assay.corpus import store

    corpus = Corpus(tmp_path / "store")
    initial = Spec.from_yaml(MINIMAL_YAML)
    corpus.put_spec(initial)
    authority = corpus.retained_spec_authority()
    enter_code = type(authority).__enter__.__code__
    interrupt = KeyboardInterrupt("Spec authority __enter__ return")
    injected = False

    def interrupt_return(frame, event, _arg):
        nonlocal injected
        if (
            not injected
            and frame.f_code is enter_code
            and frame.f_locals.get("self") is authority
            and event == "return"
        ):
            injected = True
            sys.settrace(None)
            raise interrupt
        return interrupt_return

    sys.settrace(interrupt_return)
    try:
        with pytest.raises(KeyboardInterrupt) as captured:
            with ExitStack() as cleanup:
                cleanup.callback(authority.retry_cleanup)
                authority.__enter__()
    finally:
        sys.settrace(None)

    lock_key = (corpus.spec_dir / "versions" / ".spec-write.lock").resolve()
    assert captured.value is interrupt
    assert injected
    assert authority.cleanup_complete
    assert lock_key not in store._active_spec_authority_paths()
    with corpus.locked_spec_authority() as locked:
        assert locked == initial


def test_retained_spec_authority_caller_preownership_closes_return_gap(
    tmp_path: Path,
) -> None:
    """A pre-registered callback owns authority after the callee returned."""

    from assay.corpus import store

    corpus = Corpus(tmp_path / "store")
    initial = Spec.from_yaml(MINIMAL_YAML)
    corpus.put_spec(initial)
    authority = corpus.retained_spec_authority()
    interrupt = KeyboardInterrupt("caller return after Spec authority entry")
    injected = False

    def acquire_authority():
        return authority.__enter__()

    acquire_code = acquire_authority.__code__

    def interrupt_return(frame, event, _arg):
        nonlocal injected
        if not injected and frame.f_code is acquire_code and event == "return":
            injected = True
            sys.settrace(None)
            raise interrupt
        return interrupt_return

    sys.settrace(interrupt_return)
    try:
        with pytest.raises(KeyboardInterrupt) as captured:
            with ExitStack() as cleanup:
                cleanup.callback(authority.retry_cleanup)
                acquire_authority()
    finally:
        sys.settrace(None)

    lock_key = (corpus.spec_dir / "versions" / ".spec-write.lock").resolve()
    assert captured.value is interrupt
    assert injected
    assert authority.cleanup_complete
    assert lock_key not in store._active_spec_authority_paths()
    with corpus.locked_spec_authority() as locked:
        assert locked == initial


def test_retained_spec_authority_releases_post_flock_acquire_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LOCK_EX success cannot outrun the source-owned cleanup action."""

    from assay.corpus import store

    corpus = Corpus(tmp_path / "store")
    initial = Spec.from_yaml(MINIMAL_YAML)
    corpus.put_spec(initial)
    authority = corpus.retained_spec_authority()
    interrupt = KeyboardInterrupt("after kernel flock acquisition")
    original_flock = store.fcntl.flock
    interrupted = False

    def interrupt_after_flock(descriptor, operation):
        nonlocal interrupted
        result = original_flock(descriptor, operation)
        if operation == store.fcntl.LOCK_EX and not interrupted:
            interrupted = True
            raise interrupt
        return result

    monkeypatch.setattr(store.fcntl, "flock", interrupt_after_flock)

    with pytest.raises(KeyboardInterrupt) as captured:
        authority.__enter__()

    assert captured.value is interrupt
    assert interrupted
    assert authority.cleanup_complete
    with corpus.locked_spec_authority() as locked:
        assert locked == initial


def test_retained_spec_authority_releases_post_thread_lock_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An acquire call that raises after success retains one exact depth."""

    from assay.corpus import store

    corpus = Corpus(tmp_path / "store")
    initial = Spec.from_yaml(MINIMAL_YAML)
    corpus.put_spec(initial)
    interrupt = KeyboardInterrupt("after thread-lock acquisition")

    class InterruptingRLock:
        def __init__(self) -> None:
            self.lock = threading.RLock()
            self.interrupted = False
            self.depth = 0

        def acquire(self, *args, **kwargs):
            acquired = self.lock.acquire(*args, **kwargs)
            if acquired:
                self.depth += 1
            if not self.interrupted:
                self.interrupted = True
                raise interrupt
            return acquired

        def release(self) -> None:
            self.lock.release()
            self.depth -= 1

        def _recursion_count(self) -> int:
            return self.depth

        def _is_owned(self) -> bool:
            return self.lock._is_owned()

        def __enter__(self):
            self.acquire()
            return self

        def __exit__(self, *_args):
            self.release()
            return False

    observed_lock = InterruptingRLock()
    monkeypatch.setattr(store, "_SPEC_WRITE_LOCK", observed_lock)
    authority = corpus.retained_spec_authority()

    with pytest.raises(KeyboardInterrupt) as captured:
        authority.__enter__()

    assert captured.value is interrupt
    assert observed_lock.interrupted
    assert observed_lock.depth == 0
    assert authority.cleanup_complete
    with corpus.locked_spec_authority() as locked:
        assert locked == initial


def test_retained_spec_authority_cleanup_error_retains_exact_retry_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An interrupted unlock blocks writers until the exact owner retries."""

    from assay.corpus import store

    corpus = Corpus(tmp_path / "store")
    initial = Spec.from_yaml(MINIMAL_YAML)
    corpus.put_spec(initial)
    authority = corpus.retained_spec_authority()
    assert authority.__enter__() == initial
    interrupt = KeyboardInterrupt("flock release interrupted")
    original_flock = store.fcntl.flock
    release_interrupted = False

    def interrupt_first_release(descriptor, operation):
        nonlocal release_interrupted
        if operation == store.fcntl.LOCK_UN and not release_interrupted:
            release_interrupted = True
            raise interrupt
        return original_flock(descriptor, operation)

    monkeypatch.setattr(store.fcntl, "flock", interrupt_first_release)

    with pytest.raises(SpecAuthorityCleanupError) as captured:
        authority.__exit__(None, None, None)

    assert captured.value.owner is authority
    assert captured.value.cleanup_error is interrupt
    assert not authority.cleanup_complete
    lock_key = (corpus.spec_dir / "versions" / ".spec-write.lock").resolve()
    assert lock_key in store._active_spec_authority_paths()
    with pytest.raises(RuntimeError, match="locked_spec_authority"):
        corpus.put_spec(initial)
    writer_started = threading.Event()

    def publish_again():
        writer_started.set()
        return corpus.put_spec(initial)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(publish_again)
        assert writer_started.wait(timeout=1.0)
        with pytest.raises(TimeoutError):
            future.result(timeout=0.05)
        captured.value.retry_cleanup()
        assert future.result(timeout=1.0)

    assert authority.cleanup_complete
    assert lock_key not in store._active_spec_authority_paths()


def test_retained_spec_authority_does_not_invoke_hostile_cleanup_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cleanup publishes the exact owner without truth-testing or formatting."""

    from assay.corpus import store

    corpus = Corpus(tmp_path / "store")
    initial = Spec.from_yaml(MINIMAL_YAML)
    corpus.put_spec(initial)
    authority = corpus.retained_spec_authority()
    assert authority.__enter__() == initial

    class HostilePrimaryError(RuntimeError):
        def __bool__(self) -> bool:
            raise KeyboardInterrupt("primary exception truth test")

        def __str__(self) -> str:
            raise KeyboardInterrupt("primary exception formatting")

    class HostileCleanupError(RuntimeError):
        def __str__(self) -> str:
            raise KeyboardInterrupt("cleanup exception formatting")

    primary = HostilePrimaryError("body failed")
    release_error = HostileCleanupError("flock release failed")
    original_flock = store.fcntl.flock
    release_failed = False

    def fail_first_release(descriptor, operation):
        nonlocal release_failed
        if operation == store.fcntl.LOCK_UN and not release_failed:
            release_failed = True
            raise release_error
        return original_flock(descriptor, operation)

    monkeypatch.setattr(store.fcntl, "flock", fail_first_release)

    with pytest.raises(SpecAuthorityCleanupError) as captured:
        authority.__exit__(type(primary), primary, None)

    assert captured.value.primary_error is primary
    assert captured.value.cleanup_error is release_error
    assert captured.value.owner is authority
    captured.value.retry_cleanup()
    assert authority.cleanup_complete


def test_locked_spec_authority_remains_a_reusable_context_decorator(
    tmp_path: Path,
) -> None:
    """The retained implementation preserves ContextDecorator compatibility."""

    corpus = Corpus(tmp_path / "store")
    initial = Spec.from_yaml(MINIMAL_YAML)
    corpus.put_spec(initial)

    @corpus.locked_spec_authority()
    def observe_latest() -> Spec | None:
        return corpus.latest_spec()

    assert observe_latest() == initial
    assert observe_latest() == initial


def test_retained_spec_authority_replays_cleanup_return_interrupt_after_release(
    tmp_path: Path,
) -> None:
    """A settled cleanup return replays SIGINT without a stale retry owner."""

    corpus = Corpus(tmp_path / "store")
    initial = Spec.from_yaml(MINIMAL_YAML)
    corpus.put_spec(initial)
    authority = corpus.retained_spec_authority()
    assert authority.__enter__() == initial
    cleanup_code = authority._cleanup.__code__
    interrupt = KeyboardInterrupt("after exact Spec cleanup")
    injected = False

    def interrupt_return(frame, event, _arg):
        nonlocal injected
        if (
            not injected
            and frame.f_code is cleanup_code
            and frame.f_locals.get("self") is authority
            and event == "return"
        ):
            injected = True
            sys.settrace(None)
            raise interrupt
        return interrupt_return

    sys.settrace(interrupt_return)
    try:
        with pytest.raises(KeyboardInterrupt) as captured:
            authority.__exit__(None, None, None)
    finally:
        sys.settrace(None)

    assert captured.value is interrupt
    assert injected
    assert authority.cleanup_complete
    with corpus.locked_spec_authority() as locked:
        assert locked == initial


def test_retained_spec_authority_entry_failure_releases_all_owners(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Snapshot hydration failure cannot retain the guard or writer locks."""

    from assay.corpus import store

    corpus = Corpus(tmp_path / "store")
    initial = Spec.from_yaml(MINIMAL_YAML)
    corpus.put_spec(initial)
    failure = RuntimeError("latest Spec hydration failed")
    monkeypatch.setattr(
        corpus,
        "latest_spec",
        lambda: (_ for _ in ()).throw(failure),
    )
    authority = corpus.retained_spec_authority()

    with pytest.raises(RuntimeError) as captured:
        authority.__enter__()

    lock_key = (corpus.spec_dir / "versions" / ".spec-write.lock").resolve()
    assert captured.value is failure
    assert authority.cleanup_complete
    assert lock_key not in store._active_spec_authority_paths()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_retained_spec_authority_fork_abandons_only_child_copy(
    tmp_path: Path,
) -> None:
    """Child cleanup preserves the parent flock and receives fresh locks."""

    corpus = Corpus(tmp_path / "store")
    initial = Spec.from_yaml(MINIMAL_YAML)
    corpus.put_spec(initial)
    authority = corpus.retained_spec_authority()
    assert authority.__enter__() == initial
    release_read, release_write = os.pipe()
    report_read, report_write = os.pipe()
    pid = os.fork()
    if pid == 0:  # pragma: no cover - assertions are reported through a pipe
        os.close(release_write)
        os.close(report_read)
        status = b"E"
        try:
            authority.retry_cleanup()
            if not authority.cleanup_complete:
                raise RuntimeError("inherited authority was not abandoned")
            if os.read(release_read, 1) != b"R":
                raise RuntimeError("parent release signal was not received")
            finished = threading.Event()
            observed: list[Spec | None] = []

            def enter_from_fresh_child_thread() -> None:
                with corpus.locked_spec_authority() as locked:
                    observed.append(locked)
                finished.set()

            worker = threading.Thread(
                target=enter_from_fresh_child_thread,
                daemon=True,
            )
            worker.start()
            worker.join(timeout=1.0)
            status = b"P" if finished.is_set() and observed == [initial] else b"B"
        except BaseException:
            status = b"E"
        finally:
            os.write(report_write, status)
            os._exit(0)

    os.close(release_read)
    os.close(report_write)
    lock_path = corpus.spec_dir / "versions" / ".spec-write.lock"
    with lock_path.open("a+b") as probe:
        with pytest.raises(BlockingIOError):
            fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    authority.retry_cleanup()
    os.write(release_write, b"R")
    os.close(release_write)
    assert _read_fork_report(report_read, pid) == b"P"
    os.close(report_read)
    waited, child_status = os.waitpid(pid, 0)
    assert waited == pid
    assert os.waitstatus_to_exitcode(child_status) == 0


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_spec_write_guard_child_unwind_cannot_unlock_parent(
    tmp_path: Path,
) -> None:
    """Inherited generator cleanup closes its fd without shared LOCK_UN."""

    corpus = Corpus(tmp_path / "store")
    guard = corpus._spec_write_guard()
    guard.__enter__()
    report_read, report_write = os.pipe()
    pid = os.fork()
    if pid == 0:  # pragma: no cover - assertions are reported through a pipe
        os.close(report_read)
        status = b"P"
        try:
            guard.__exit__(None, None, None)
        except BaseException:
            status = b"E"
        os.write(report_write, status)
        os._exit(0)

    os.close(report_write)
    assert _read_fork_report(report_read, pid) == b"P"
    os.close(report_read)
    waited, child_status = os.waitpid(pid, 0)
    assert waited == pid
    assert os.waitstatus_to_exitcode(child_status) == 0
    lock_path = corpus.spec_dir / "versions" / ".spec-write.lock"
    with lock_path.open("a+b") as probe:
        with pytest.raises(BlockingIOError):
            fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    guard.__exit__(None, None, None)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_spec_authority_fork_waits_for_other_thread_owner(
    tmp_path: Path,
) -> None:
    """A cross-thread fork cannot inherit another thread's live lock state."""

    corpus = Corpus(tmp_path / "store")
    initial = Spec.from_yaml(MINIMAL_YAML)
    corpus.put_spec(initial)
    entered = threading.Event()
    release = threading.Event()

    def hold_authority() -> None:
        with corpus.locked_spec_authority():
            entered.set()
            assert release.wait(timeout=2.0)

    owner = threading.Thread(target=hold_authority)
    owner.start()
    assert entered.wait(timeout=1.0)
    releaser = threading.Timer(0.1, release.set)
    releaser.start()
    report_read, report_write = os.pipe()
    pid = os.fork()
    if pid == 0:  # pragma: no cover - assertions are reported through a pipe
        os.close(report_read)
        status = b"E"
        try:
            with corpus.locked_spec_authority() as locked:
                status = b"P" if locked == initial else b"B"
        except BaseException:
            status = b"E"
        os.write(report_write, status)
        os._exit(0)

    os.close(report_write)
    assert release.is_set()
    owner.join(timeout=1.0)
    releaser.join(timeout=1.0)
    assert not owner.is_alive()
    assert _read_fork_report(report_read, pid) == b"P"
    os.close(report_read)
    waited, child_status = os.waitpid(pid, 0)
    assert waited == pid
    assert os.waitstatus_to_exitcode(child_status) == 0


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_pending_retained_authority_remains_enterable_after_fork(
    tmp_path: Path,
) -> None:
    """Process ownership begins at entry, not at one-shot owner construction."""

    corpus = Corpus(tmp_path / "store")
    initial = Spec.from_yaml(MINIMAL_YAML)
    corpus.put_spec(initial)
    authority = corpus.retained_spec_authority()
    report_read, report_write = os.pipe()
    pid = os.fork()
    if pid == 0:  # pragma: no cover - assertions are reported through a pipe
        os.close(report_read)
        status = b"E"
        try:
            with authority as locked:
                status = b"P" if locked == initial else b"B"
        except BaseException:
            status = b"E"
        os.write(report_write, status)
        os._exit(0)

    os.close(report_write)
    assert _read_fork_report(report_read, pid) == b"P"
    os.close(report_read)
    waited, child_status = os.waitpid(pid, 0)
    assert waited == pid
    assert os.waitstatus_to_exitcode(child_status) == 0
    with authority as locked:
        assert locked == initial


def test_locked_spec_authority_supports_cross_corpus_nesting(
    tmp_path: Path,
) -> None:
    """The ordinary guard needs no private exact RLock-depth probe."""

    first = Corpus(tmp_path / "first")
    second = Corpus(tmp_path / "second")
    first_spec = Spec.from_yaml(MINIMAL_YAML)
    second_spec = Spec.from_yaml(MINIMAL_YAML)
    first.put_spec(first_spec)
    second.put_spec(second_spec)

    with first.locked_spec_authority() as locked_first:
        with second.locked_spec_authority() as locked_second:
            assert locked_first == first_spec
            assert locked_second == second_spec


def test_put_artifact_is_idempotent(tmp_path: Path) -> None:
    c = Corpus(tmp_path / "store")
    blob = b"hello world"
    h1 = c.put_artifact(blob, kind="diffs", name_hint="patch.bin")
    h2 = c.put_artifact(blob, kind="diffs", name_hint="patch.bin")
    assert h1 == h2
    files = list((c.spec_dir / "artifacts" / "diffs").iterdir())
    assert len(files) == 1


def test_get_artifact_bounded_enforces_ceiling_before_sparse_read(
    tmp_path: Path,
) -> None:
    corpus = Corpus(tmp_path / "store")
    ref = corpus.put_artifact(b"bounded", kind="diffs")

    assert corpus.get_artifact_bounded(ref, max_bytes=7) == b"bounded"
    with pytest.raises(ValueError, match="exceeds 6 bytes"):
        corpus.get_artifact_bounded(ref, max_bytes=6)


def test_get_artifact_bounded_skips_oversized_same_stem_decoy(
    tmp_path: Path,
) -> None:
    corpus = Corpus(tmp_path / "store")
    payload = b"content-verified"
    ref = corpus.put_artifact(payload, kind="diffs")
    digest = str(ref).split(":", 1)[1]
    rogue = corpus.spec_dir / "artifacts" / "aaa_rogue"
    rogue.mkdir()
    decoy = rogue / f"{digest}.bin"
    with decoy.open("wb") as stream:
        stream.truncate(1024 * 1024 * 1024)

    assert corpus.get_artifact_bounded(
        ref,
        max_bytes=len(payload),
    ) == payload


@pytest.mark.parametrize("invalid", [-1, True, 1.5, "12"])
def test_get_artifact_bounded_rejects_invalid_ceiling(
    tmp_path: Path,
    invalid: object,
) -> None:
    corpus = Corpus(tmp_path / "store")
    ref = corpus.put_artifact(b"payload", kind="diffs")

    with pytest.raises(ValueError, match="non-negative integer"):
        corpus.get_artifact_bounded(  # type: ignore[arg-type]
            ref,
            max_bytes=invalid,
        )


def test_open_run_materializes_folder(tmp_path: Path) -> None:
    c = Corpus(tmp_path / "store")
    handle = c.open_run("plan_v1")
    assert isinstance(handle, RunHandle)
    assert handle.recipe == "plan_v1"
    assert handle.run_dir.is_dir()
    assert handle.run_dir.parent == c.spec_dir / "runs"
    assert stat.S_IMODE(handle.run_dir.stat().st_mode) == 0o700


def test_open_run_with_idempotency_key_reuses_folder(tmp_path: Path) -> None:
    corpus = Corpus(tmp_path / "store")
    first = corpus.open_run("plan", idempotency_key="request-123")
    second = corpus.open_run("plan", idempotency_key="request-123")
    other = corpus.open_run("plan", idempotency_key="request-456")
    assert first.run_dir == second.run_dir
    assert first.started_at == second.started_at
    assert other.run_dir != first.run_dir
    assert (first.run_dir / "run.json").is_file()
    assert stat.S_IMODE(first.run_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE((first.run_dir / "run.json").stat().st_mode) == 0o600


@pytest.mark.parametrize("recipe", ["../escape", "has space", "", "/abs"])
def test_open_run_rejects_unsafe_recipe(recipe: str, tmp_path: Path) -> None:
    corpus = Corpus(tmp_path / "store")
    with pytest.raises(ValueError, match="recipe"):
        corpus.open_run(recipe)


def test_open_run_rejects_empty_idempotency_key(tmp_path: Path) -> None:
    corpus = Corpus(tmp_path / "store")
    with pytest.raises(ValueError, match="idempotency_key"):
        corpus.open_run("plan", idempotency_key="")


def test_put_spec_rejects_divergent_same_version_write(tmp_path: Path) -> None:
    """ACOR-2 (write-side): a second writer committing a DIFFERENT spec at
    the same version (same parent, divergent content) is rejected rather
    than silently leaving two manifests where one writer's refs vanish."""
    from assay.corpus import ConcurrentSpecError

    c = Corpus(tmp_path / "store")
    s = Spec.from_yaml(MINIMAL_YAML)
    c.put_spec(s)  # version committed by writer A
    s2 = dataclasses.replace(s, intent="a divergent intent at the same version")
    assert s2.version == s.version  # same version, different content
    with pytest.raises(ConcurrentSpecError, match="concurrent writer"):
        c.put_spec(s2)
    # Idempotent re-write of the SAME content still works.
    assert c.put_spec(s).startswith("sha256:")


def test_put_spec_requires_contiguous_parent_lineage(tmp_path: Path) -> None:
    from assay.corpus import ConcurrentSpecError

    corpus = Corpus(tmp_path / "store")
    base = Spec.from_yaml(MINIMAL_YAML)
    corpus.put_spec(base)
    first_ref = corpus.put_record(_event("first"), kind="lifecycle_events")
    second_ref = corpus.put_record(_event("second"), kind="lifecycle_events")
    first = base.with_appended_refs(
        lifecycle_event_refs=(first_ref,),
    )
    skipped = first.with_appended_refs(
        lifecycle_event_refs=(second_ref,),
    )

    with pytest.raises(ConcurrentSpecError, match="only valid next version"):
        corpus.put_spec(skipped)

    wrong_parent = dataclasses.replace(
        first, parent=ContentHash("sha256:" + "f" * 64),
    )
    with pytest.raises(ConcurrentSpecError, match="does not match latest"):
        corpus.put_spec(wrong_parent)

    corpus.put_spec(first)
    corpus.put_spec(skipped)
    assert corpus.latest_spec() == skipped


def test_put_spec_rejects_nonzero_first_manifest(tmp_path: Path) -> None:
    corpus = Corpus(tmp_path / "store")
    base = Spec.from_yaml(MINIMAL_YAML)
    nonzero = base.with_appended_refs(
        verdict_refs=(ContentHash("sha256:" + "1" * 64),),
    )

    with pytest.raises(ValueError, match="first Spec manifest"):
        corpus.put_spec(nonzero)


def test_get_spec_raises_on_divergent_manifests(tmp_path: Path) -> None:
    """ACOR-2 (read backstop): if two divergent manifests for one version
    ever coexist (the exactly-simultaneous TOCTOU residue), get_spec /
    latest_spec refuse to silently mtime-pick one and drop the other."""
    from assay.corpus import ConcurrentSpecError

    c = Corpus(tmp_path / "store")
    s = Spec.from_yaml(MINIMAL_YAML)
    c.put_spec(s)
    v = s.version
    # A second divergent manifest slipped in for the same version.
    (c.spec_dir / "versions" / f"v{v}_deadbeefcafe.json").write_bytes(b"{}")
    with pytest.raises(ConcurrentSpecError, match="divergent"):
        c.get_spec(v)
    with pytest.raises(ConcurrentSpecError):
        c.latest_spec()


def test_latest_spec_rejects_malformed_newer_manifest_filename(
    tmp_path: Path,
) -> None:
    from assay.corpus import ConcurrentSpecError

    corpus = Corpus(tmp_path / "store")
    base = Spec.from_yaml(MINIMAL_YAML)
    corpus.put_spec(base)
    event_ref = corpus.put_record(
        _event("advance lineage"), kind="lifecycle_events"
    )
    advanced = base.with_appended_refs(lifecycle_event_refs=(event_ref,))
    corpus.put_spec(advanced)
    manifest, = tuple(
        (corpus.spec_dir / "versions").glob(f"v{advanced.version}_*.json")
    )
    manifest.rename(manifest.with_name(f"v{advanced.version}_NOT-HEX.json"))

    with pytest.raises(
        ConcurrentSpecError,
        match="malformed Spec-like manifest filename",
    ):
        corpus.latest_spec()


def test_latest_spec_rejects_noncanonical_version_spelling(
    tmp_path: Path,
) -> None:
    from assay.corpus import ConcurrentSpecError

    corpus = Corpus(tmp_path / "store")
    corpus.put_spec(Spec.from_yaml(MINIMAL_YAML))
    (corpus.spec_dir / "versions" / "v00_deadbeefcafe.json").write_bytes(
        b"{}"
    )

    with pytest.raises(
        ConcurrentSpecError,
        match="malformed Spec-like manifest filename",
    ):
        corpus.latest_spec()


def test_latest_spec_rejects_unbounded_version_without_allocating_range(
    tmp_path: Path,
) -> None:
    from assay.corpus import ConcurrentSpecError

    corpus = Corpus(tmp_path / "store")
    corpus.put_spec(Spec.from_yaml(MINIMAL_YAML))
    huge_version = "9" * 30
    (
        corpus.spec_dir
        / "versions"
        / f"v{huge_version}_deadbeefcafe.json"
    ).write_bytes(b"{}")

    with pytest.raises(ConcurrentSpecError, match="supported bound"):
        corpus.latest_spec()


def test_latest_spec_validates_ambiguous_historical_versions(
    tmp_path: Path,
) -> None:
    from assay.corpus import ConcurrentSpecError

    corpus = Corpus(tmp_path / "store")
    base = Spec.from_yaml(MINIMAL_YAML)
    corpus.put_spec(base)
    event_ref = corpus.put_record(_event("advance"), kind="lifecycle_events")
    advanced = base.with_appended_refs(lifecycle_event_refs=(event_ref,))
    corpus.put_spec(advanced)
    (corpus.spec_dir / "versions" / "v0_deadbeefcafe.json").write_bytes(b"{}")

    with pytest.raises(ConcurrentSpecError, match="divergent"):
        corpus.latest_spec()
    with pytest.raises(ConcurrentSpecError):
        corpus.put_spec(advanced.with_appended_refs())


def test_latest_spec_rejects_a_missing_historical_version(
    tmp_path: Path,
) -> None:
    from assay.corpus import ConcurrentSpecError

    corpus = Corpus(tmp_path / "store")
    base = Spec.from_yaml(MINIMAL_YAML)
    corpus.put_spec(base)
    event_ref = corpus.put_record(_event("advance"), kind="lifecycle_events")
    corpus.put_spec(base.with_appended_refs(lifecycle_event_refs=(event_ref,)))
    root_manifest, = tuple(
        (corpus.spec_dir / "versions").glob("v0_*.json")
    )
    root_manifest.rename(root_manifest.with_suffix(".saved"))

    with pytest.raises(ConcurrentSpecError, match="malformed Spec-like"):
        corpus.latest_spec()


def test_spec_reader_accepts_legacy_omitted_optional_fields(
    tmp_path: Path,
) -> None:
    corpus = Corpus(tmp_path / "store")
    spec = dataclasses.replace(
        Spec.from_yaml(MINIMAL_YAML),
        plan_policy=PlanPolicy(),
    )
    payload = parse(serialize(spec))
    del payload["plan_policy"]["max_verification_only_empty_scope_slices"]
    legacy_blob = serialize(payload)
    short = hashlib.sha256(legacy_blob).hexdigest()[:12]
    manifest = corpus.spec_dir / "versions" / f"v0_{short}.json"
    manifest.write_bytes(legacy_blob)

    loaded = corpus.get_spec(0)
    paired_spec, paired_digest = corpus.get_spec_with_digest(0)
    assert paired_spec == loaded
    assert paired_digest == ContentHash(
        "sha256:" + hashlib.sha256(legacy_blob).hexdigest()
    )
    assert loaded.plan_policy is not None
    assert loaded.plan_policy.max_verification_only_empty_scope_slices is None
    assert corpus.latest_spec() == loaded
    assert corpus.put_spec(loaded) == ContentHash(
        "sha256:" + hashlib.sha256(legacy_blob).hexdigest()
    )
    helper_child = loaded.with_appended_refs()
    corpus.put_spec(helper_child)
    legacy_digest = ContentHash(
        "sha256:" + hashlib.sha256(legacy_blob).hexdigest()
    )
    direct_child = dataclasses.replace(
        helper_child,
        parent=legacy_digest,
        root=legacy_digest,
    )
    assert corpus.latest_spec() == direct_child

    committed = corpus.commit(lifecycle_events=(_event("legacy child"),))
    direct_digest = ContentHash(
        "sha256:" + hashlib.sha256(serialize(direct_child)).hexdigest()
    )
    assert committed.spec.parent == direct_digest
    assert committed.spec.root == direct_child.root
    assert corpus.latest_spec() == committed.spec


def test_latest_spec_rejects_persisted_hydrated_only_legacy_parent(
    tmp_path: Path,
) -> None:
    from assay.corpus import ConcurrentSpecError

    corpus = Corpus(tmp_path / "store")
    spec = dataclasses.replace(
        Spec.from_yaml(MINIMAL_YAML),
        plan_policy=PlanPolicy(),
    )
    payload = parse(serialize(spec))
    del payload["plan_policy"]["max_verification_only_empty_scope_slices"]
    legacy_blob = serialize(payload)
    legacy_short = hashlib.sha256(legacy_blob).hexdigest()[:12]
    (
        corpus.spec_dir / "versions" / f"v0_{legacy_short}.json"
    ).write_bytes(legacy_blob)
    loaded = corpus.get_spec(0)
    dangling_child = loaded.with_appended_refs()
    dangling_blob = serialize(dangling_child)
    dangling_short = hashlib.sha256(dangling_blob).hexdigest()[:12]
    (
        corpus.spec_dir / "versions" / f"v1_{dangling_short}.json"
    ).write_bytes(dangling_blob)

    with pytest.raises(
        ConcurrentSpecError,
        match="does not extend the exact canonical",
    ):
        corpus.latest_spec()


def test_spec_operations_reject_replaced_versions_directory(
    tmp_path: Path,
) -> None:
    from assay.corpus import ConcurrentSpecError

    corpus = Corpus(tmp_path / "store")
    base = Spec.from_yaml(MINIMAL_YAML)
    corpus.put_spec(base)
    versions = corpus.spec_dir / "versions"
    displaced = tmp_path / "displaced-versions"
    versions.rename(displaced)
    versions.symlink_to(displaced, target_is_directory=True)

    with pytest.raises(ConcurrentSpecError, match="authority changed"):
        corpus.get_spec(0)
    with pytest.raises(ConcurrentSpecError, match="authority changed"):
        corpus.latest_spec()
    with pytest.raises(ConcurrentSpecError, match="authority changed"):
        corpus.put_spec(base)


def test_get_spec_rejects_in_place_canonical_manifest_rewrite(
    tmp_path: Path,
) -> None:
    from assay.corpus import ConcurrentSpecError

    corpus = Corpus(tmp_path / "store")
    spec = Spec.from_yaml(MINIMAL_YAML)
    corpus.put_spec(spec)
    manifest, = tuple(
        (corpus.spec_dir / "versions").glob(f"v{spec.version}_*.json")
    )
    tampered = dataclasses.replace(spec, intent="rewritten in place")
    manifest.write_bytes(serialize(tampered))

    with pytest.raises(
        ConcurrentSpecError,
        match="exact canonical version/hash filename",
    ):
        corpus.get_spec(spec.version)
    with pytest.raises(ConcurrentSpecError):
        corpus.latest_spec()
    with pytest.raises(ConcurrentSpecError):
        corpus.put_spec(tampered)


def test_get_spec_rejects_renamed_canonical_manifest(tmp_path: Path) -> None:
    from assay.corpus import ConcurrentSpecError

    corpus = Corpus(tmp_path / "store")
    spec = Spec.from_yaml(MINIMAL_YAML)
    corpus.put_spec(spec)
    manifest, = tuple(
        (corpus.spec_dir / "versions").glob(f"v{spec.version}_*.json")
    )
    renamed = manifest.with_name(f"v{spec.version}_deadbeefcafe.json")
    manifest.rename(renamed)

    with pytest.raises(
        ConcurrentSpecError,
        match="exact canonical version/hash filename",
    ):
        corpus.get_spec(spec.version)


@pytest.mark.parametrize("entry_kind", ("symlink", "hardlink", "fifo"))
def test_get_spec_rejects_aliased_or_special_manifest(
    tmp_path: Path,
    entry_kind: str,
) -> None:
    from assay.corpus import ConcurrentSpecError

    corpus = Corpus(tmp_path / "store")
    spec = Spec.from_yaml(MINIMAL_YAML)
    corpus.put_spec(spec)
    manifest, = tuple(
        (corpus.spec_dir / "versions").glob(f"v{spec.version}_*.json")
    )
    if entry_kind == "symlink":
        target = tmp_path / "manifest-target.json"
        manifest.rename(target)
        manifest.symlink_to(target)
    elif entry_kind == "hardlink":
        os.link(manifest, tmp_path / "manifest-alias.json")
    else:
        manifest.unlink()
        os.mkfifo(manifest)

    with pytest.raises(
        ConcurrentSpecError,
        match="bounded, singly linked regular file",
    ):
        corpus.get_spec(spec.version)
    with pytest.raises(ConcurrentSpecError):
        corpus.put_spec(spec)


def test_put_spec_enforces_the_manifest_read_bound(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from assay.corpus import store

    corpus = Corpus(tmp_path / "store")
    monkeypatch.setattr(store, "_SPEC_MANIFEST_MAX_BYTES", 256)
    oversized = dataclasses.replace(
        Spec.from_yaml(MINIMAL_YAML),
        intent="x" * 1024,
    )

    with pytest.raises(ValueError, match="exceeds the supported"):
        corpus.put_spec(oversized)
    assert tuple((corpus.spec_dir / "versions").glob("v*.json")) == ()


def test_get_spec_rejects_divergent_sibling_created_during_read(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from assay.corpus import ConcurrentSpecError, store

    corpus = Corpus(tmp_path / "store")
    spec = Spec.from_yaml(MINIMAL_YAML)
    corpus.put_spec(spec)
    sibling = (
        corpus.spec_dir
        / "versions"
        / f"v{spec.version}_deadbeefcafe.json"
    )
    real_read = store.os.read
    injected = False

    def inject_sibling(descriptor: int, size: int) -> bytes:
        nonlocal injected
        chunk = real_read(descriptor, size)
        if not injected:
            injected = True
            sibling.write_bytes(b"{}")
        return chunk

    monkeypatch.setattr(store.os, "read", inject_sibling)
    with pytest.raises(
        ConcurrentSpecError,
        match="changed manifest topology during inspection",
    ):
        corpus.get_spec(spec.version)
    assert injected


def test_record_codec_round_trips_nested_dataclasses_and_tuples(
    tmp_path: Path,
) -> None:
    corpus = Corpus(tmp_path / "store")
    implementation = _implementation(corpus)

    ref = corpus.put_record(implementation, kind="implementations")
    restored = corpus.get_record(ref, as_type=Implementation)

    assert restored == implementation
    assert isinstance(restored.execution, ExecutionContext)
    assert isinstance(restored.files_modified, tuple)
    assert isinstance(restored.scope_escapes, tuple)
    assert restored.artifacts == implementation.artifacts
    assert restored.input_tokens == 123
    assert restored.output_tokens == 45
    assert restored.cache_creation_tokens == 23
    assert restored.cache_read_tokens == 67
    assert restored.stop_reason == "max_turns"


def test_implementation_telemetry_is_optional_for_legacy_records(
    tmp_path: Path,
) -> None:
    corpus = Corpus(tmp_path / "store")
    payload = parse(serialize(_implementation(corpus)))
    payload.pop("input_tokens")
    payload.pop("output_tokens")
    payload.pop("cache_creation_tokens")
    payload.pop("cache_read_tokens")
    payload.pop("stop_reason")

    restored = decode_dataclass(payload, Implementation)

    assert restored.input_tokens == 0
    assert restored.output_tokens == 0
    assert restored.cache_creation_tokens is None
    assert restored.cache_read_tokens is None
    assert restored.stop_reason is None


def test_measurement_sqlite_round_trip_uses_the_same_typed_codec(
    tmp_path: Path,
) -> None:
    corpus = Corpus(tmp_path / "store")
    raw_ref = corpus.put_artifact(b"benchmark output", kind="raw_output")
    measurement = Measurement(
        bench_id="bench-a",
        commit="cafebabe",
        config_hash="config-a",
        values={"fps": 60.0, "frame_ms": 16.7},
        raw_output_ref=raw_ref,
        execution=_execution("measurement"),
    )

    ref = corpus.put_record(measurement, kind="measurements")
    restored = corpus.get_record(ref, as_type=Measurement)

    assert restored == measurement
    assert isinstance(restored.execution, ExecutionContext)
    assert isinstance(restored.execution.toolchain, dict)


def test_spec_commit_rejects_tampered_measurement_row(tmp_path: Path) -> None:
    corpus = Corpus(tmp_path / "store")
    base = Spec.from_yaml(MINIMAL_YAML)
    corpus.put_spec(base)
    raw_ref = corpus.put_artifact(b"benchmark output", kind="raw_output")
    measurement = Measurement(
        bench_id="bench-a",
        commit="cafebabe",
        config_hash="config-a",
        values={"fps": 60.0},
        raw_output_ref=raw_ref,
        execution=_execution("measurement"),
    )
    ref = corpus.put_record(measurement, kind="measurements")
    conn = corpus._measurements_db()
    conn.execute(
        "UPDATE measurements SET values_json = ? WHERE ref = ?",
        ('{"fps":1.0}', ref),
    )
    conn.commit()

    candidate = base.with_appended_refs(measurement_refs=(ref,))
    with pytest.raises(ValueError, match="corrupt measurement row"):
        corpus.put_spec(candidate)
    assert corpus.latest_spec() == base


def test_record_codec_is_strict_about_unknown_fields(tmp_path: Path) -> None:
    corpus = Corpus(tmp_path / "store")
    payload = parse(serialize(_implementation(corpus)))
    payload["unknown_field"] = "strict codecs must reject undeclared data"

    with pytest.raises(RecordDecodeError, match="unknown fields"):
        decode_dataclass(payload, Implementation)


def test_put_record_rejects_malformed_nested_content_hash(tmp_path: Path) -> None:
    corpus = Corpus(tmp_path / "store")
    malformed = dataclasses.replace(
        _implementation(corpus), diff_ref=ContentHash("sha256:not-a-digest")
    )

    with pytest.raises(RecordDecodeError, match="64 lowercase hex"):
        corpus.put_record(malformed, kind="implementations")


def test_put_record_rejects_wrong_record_kind(tmp_path: Path) -> None:
    corpus = Corpus(tmp_path / "store")

    with pytest.raises(TypeError, match="requires LifecycleEvent"):
        corpus.put_record(_implementation(corpus), kind="lifecycle_events")


def test_put_spec_rejects_dangling_and_wrong_kind_references(
    tmp_path: Path,
) -> None:
    corpus = Corpus(tmp_path / "store")
    base = Spec.from_yaml(MINIMAL_YAML)
    corpus.put_spec(base)

    dangling = base.with_appended_refs(
        verdict_refs=(ContentHash("sha256:" + "a" * 64),)
    )
    with pytest.raises(ValueError, match="missing verdicts record"):
        corpus.put_spec(dangling)

    event_ref = corpus.put_record(_event("wrong-kind"), kind="lifecycle_events")
    wrong_kind = base.with_appended_refs(verdict_refs=(event_ref,))
    with pytest.raises(ValueError, match="missing verdicts record"):
        corpus.put_spec(wrong_kind)

    corrupt_ref = corpus.put_record(_event("corrupt"), kind="lifecycle_events")
    corrupt_hex = corrupt_ref.split(":", 1)[1]
    (
        corpus.spec_dir
        / "artifacts"
        / "lifecycle_events"
        / f"{corrupt_hex}.bin"
    ).write_bytes(b"tampered")
    corrupt = base.with_appended_refs(lifecycle_event_refs=(corrupt_ref,))
    with pytest.raises(ValueError, match="content hash mismatch"):
        corpus.put_spec(corrupt)

    assert corpus.latest_spec() == base


def test_put_spec_enforces_append_only_record_collections(tmp_path: Path) -> None:
    corpus = Corpus(tmp_path / "store")
    base = Spec.from_yaml(MINIMAL_YAML)
    corpus.put_spec(base)
    event_ref = corpus.put_record(_event("one"), kind="lifecycle_events")
    first = base.with_appended_refs(lifecycle_event_refs=(event_ref,))
    corpus.put_spec(first)
    rewritten = dataclasses.replace(
        first.with_appended_refs(), lifecycle_event_refs=()
    )

    with pytest.raises(ValueError, match="append-only"):
        corpus.put_spec(rewritten)

    assert corpus.latest_spec() == first


def test_commit_publishes_multiple_record_types_in_one_spec_version(
    tmp_path: Path,
) -> None:
    corpus = Corpus(tmp_path / "store")
    base = Spec.from_yaml(MINIMAL_YAML)
    corpus.put_spec(base)
    implementation = _implementation(corpus)
    event = _event("landed")

    result = corpus.commit(
        implementations=(implementation,), lifecycle_events=(event,)
    )

    assert result.spec.version == 1
    assert result.spec.implementation_refs == result.refs.implementation_refs
    assert result.spec.lifecycle_event_refs == result.refs.lifecycle_event_refs
    assert len(tuple((corpus.spec_dir / "versions").glob("v*.json"))) == 2
    assert (
        corpus.get_record(
            result.refs.implementation_refs[0], as_type=Implementation
        )
        == implementation
    )

    repeated = corpus.commit(
        implementations=(implementation,), lifecycle_events=(event,)
    )
    assert repeated.spec == result.spec
    assert len(tuple((corpus.spec_dir / "versions").glob("v*.json"))) == 2


def test_commit_attaches_records_persisted_by_upstream_primitive(
    tmp_path: Path,
) -> None:
    corpus = Corpus(tmp_path / "store")
    corpus.put_spec(Spec.from_yaml(MINIMAL_YAML))
    event_ref = corpus.put_record(_event("pre-staged"), kind="lifecycle_events")

    result = corpus.commit(
        record_refs=CommitRefs(lifecycle_event_refs=(event_ref,))
    )

    assert result.spec.lifecycle_event_refs == (event_ref,)
    assert result.refs.lifecycle_event_refs == (event_ref,)


def test_commit_transform_is_same_version_and_cannot_drop_staged_refs(
    tmp_path: Path,
) -> None:
    corpus = Corpus(tmp_path / "store")
    base = Spec.from_yaml(MINIMAL_YAML)
    corpus.put_spec(base)

    committed = corpus.commit(
        lifecycle_events=(_event("transformed"),),
        transform=lambda candidate: dataclasses.replace(
            candidate, execution=_execution("transaction")
        ),
    )
    assert committed.spec.version == 1
    assert committed.spec.execution == _execution("transaction")

    staged = _event("must-remain")
    with pytest.raises(ValueError, match="dropped staged"):
        corpus.commit(
            lifecycle_events=(staged,),
            transform=lambda candidate: dataclasses.replace(
                candidate,
                lifecycle_event_refs=committed.spec.lifecycle_event_refs,
            ),
        )
    assert corpus.latest_spec() == committed.spec

    # The failed transaction may leave only a content-addressed orphan.  A
    # retry reuses it and makes it visible through one valid manifest.
    recovered = corpus.commit(lifecycle_events=(staged,))
    assert recovered.spec.version == committed.spec.version + 1
    assert recovered.refs.lifecycle_event_refs[0] in (
        recovered.spec.lifecycle_event_refs
    )


def test_commit_factory_stamps_lifecycle_with_structural_change_atomically(
    tmp_path: Path,
) -> None:
    corpus = Corpus(tmp_path / "store")
    base = Spec.from_yaml(MINIMAL_YAML)
    corpus.put_spec(base)
    implementation = _implementation(corpus)

    result = corpus.commit(
        implementations=(implementation,),
        lifecycle_event_factory=lambda target_version: LifecycleEvent(
            kind="landed",
            reason="evidence and authority share one visibility point",
            affected_versions=(target_version,),
            execution=_execution("atomic"),
        ),
        transform=lambda candidate: dataclasses.replace(
            candidate, intent="approved replacement"
        ),
    )

    assert result.spec.version == 1
    assert result.spec.intent == "approved replacement"
    assert len(result.refs.implementation_refs) == 1
    assert len(result.refs.lifecycle_event_refs) == 1
    event = corpus.get_record(
        result.refs.lifecycle_event_refs[0], as_type=LifecycleEvent
    )
    assert event.affected_versions == (result.spec.version,)
    assert len(tuple((corpus.spec_dir / "versions").glob("v*.json"))) == 2


def test_commit_factory_rejects_wrong_record_type_without_publishing(
    tmp_path: Path,
) -> None:
    corpus = Corpus(tmp_path / "store")
    base = Spec.from_yaml(MINIMAL_YAML)
    corpus.put_spec(base)

    with pytest.raises(TypeError, match="must return LifecycleEvent"):
        corpus.commit(  # type: ignore[arg-type,return-value]
            lifecycle_event_factory=lambda _version: object(),
        )

    assert corpus.latest_spec() == base


def test_concurrent_lifecycle_commits_rebase_and_stamp_correct_versions(
    tmp_path: Path,
) -> None:
    corpus = Corpus(tmp_path / "store")
    corpus.put_spec(Spec.from_yaml(MINIMAL_YAML))

    def publish(label: str):
        return corpus.commit_lifecycle_event(
            lambda target_version: LifecycleEvent(
                kind=label,
                reason="concurrent test",
                affected_versions=(target_version,),
                execution=_execution(label),
            )
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(publish, "one")
        second = pool.submit(publish, "two")
        outcomes = (first.result(), second.result())

    latest = corpus.latest_spec()
    assert latest is not None
    assert latest.version == 2
    assert len(latest.lifecycle_event_refs) == 2
    stamped = sorted(
        event.affected_versions[0] for _result, event in outcomes
    )
    assert stamped == [1, 2]


def test_concurrent_measurement_commits_share_connection_safely(
    tmp_path: Path,
) -> None:
    corpus = Corpus(tmp_path / "store")
    corpus.put_spec(Spec.from_yaml(MINIMAL_YAML))
    # Initialize the cached connection on this thread before worker use.
    assert tuple(corpus.list_measurement_refs()) == ()
    raw_refs = (
        corpus.put_artifact(b"one", kind="raw_output"),
        corpus.put_artifact(b"two", kind="raw_output"),
    )

    def publish(index: int):
        measurement = Measurement(
            bench_id=f"bench-{index}",
            commit=f"commit-{index}",
            config_hash=f"config-{index}",
            values={"fps": float(index + 1)},
            raw_output_ref=raw_refs[index],
            execution=_execution(f"measurement-{index}"),
        )
        return corpus.commit(measurements=(measurement,))

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = tuple(pool.map(publish, (0, 1)))

    latest = corpus.latest_spec()
    assert latest is not None
    assert latest.version == 2
    assert len(latest.measurement_refs) == 2
    assert {result.refs.measurement_refs[0] for result in outcomes} == set(
        latest.measurement_refs
    )
