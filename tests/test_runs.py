# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Tests for assay.corpus.runs (RunHandle / RunResult).

AQA-1: this module was dark (no dedicated coverage). It backs every
workflow's run folder (events.jsonl + result.json per arch §9), so pin
its append + finalize behavior.
"""

from __future__ import annotations

import json
import stat
import threading
from pathlib import Path

import pytest

from assay.corpus import RunAlreadyFinalizedError, RunHandle, RunResult
from assay.corpus import runs as runs_module


def test_run_result_defaults():
    r = RunResult(status="passed")
    assert r.status == "passed"
    assert r.duration_s == 0.0
    assert r.cost_usd == 0.0


def test_run_handle_exposes_monotonic_elapsed_time(
    tmp_path: Path,
    monkeypatch,
):
    handle = RunHandle(
        run_dir=tmp_path,
        recipe="swebench",
        started_at="20260610T000000Z",
        _opened_monotonic=10.0,
    )
    monkeypatch.setattr(runs_module.time, "monotonic", lambda: 12.5)

    assert handle.elapsed_s == 2.5


def test_write_event_appends_jsonl(tmp_path: Path):
    h = RunHandle(run_dir=tmp_path, recipe="plan", started_at="20260610T000000Z")
    h.write_event({"phase": 1, "ok": True})
    h.write_event({"phase": 2, "ok": False})
    lines = (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"phase": 1, "ok": True}
    assert json.loads(lines[1]) == {"phase": 2, "ok": False}
    assert stat.S_IMODE((tmp_path / "events.jsonl").stat().st_mode) == 0o600


def test_write_event_retries_partial_os_write(tmp_path: Path, monkeypatch):
    real_write = runs_module.os.write

    def partial_write(fd, payload):
        view = memoryview(payload)
        return real_write(fd, view[: max(1, len(view) // 2)])

    monkeypatch.setattr(runs_module.os, "write", partial_write)
    handle = RunHandle(
        run_dir=tmp_path, recipe="plan", started_at="20260610T000000Z",
    )
    payload = {"message": "x" * 1000}
    handle.write_event(payload)
    assert json.loads((tmp_path / "events.jsonl").read_text()) == payload


def test_write_summary_atomically_publishes_valid_json(tmp_path: Path):
    handle = RunHandle(
        run_dir=tmp_path, recipe="plan", started_at="20260610T000000Z"
    )
    payload = {"status": "running", "progress": {"done": 2, "total": 5}}

    handle.write_summary(payload)

    summary_path = tmp_path / "summary.json"
    assert json.loads(summary_path.read_text(encoding="utf-8")) == payload
    assert not tuple(tmp_path.glob(".tmp_summary.json_*"))


def test_write_summary_replaces_previous_complete_document(tmp_path: Path):
    handle = RunHandle(
        run_dir=tmp_path, recipe="plan", started_at="20260610T000000Z"
    )
    handle.write_summary({"checkpoint": 1, "stale": True})

    handle.write_summary({"checkpoint": 2})

    assert json.loads(
        (tmp_path / "summary.json").read_text(encoding="utf-8")
    ) == {"checkpoint": 2}


def test_write_summary_uses_private_file_mode(tmp_path: Path):
    handle = RunHandle(
        run_dir=tmp_path, recipe="plan", started_at="20260610T000000Z"
    )

    handle.write_summary({"status": "running"})

    mode = stat.S_IMODE((tmp_path / "summary.json").stat().st_mode)
    assert mode == 0o600


def test_summary_cannot_be_replaced_after_finalize(tmp_path: Path):
    handle = RunHandle(
        run_dir=tmp_path, recipe="plan", started_at="20260610T000000Z"
    )
    handle.write_summary({"status": "running"})
    handle.finalize(RunResult(status="passed"))

    with pytest.raises(RunAlreadyFinalizedError, match="cannot replace summary"):
        handle.write_summary({"status": "late"})

    assert json.loads(
        (tmp_path / "summary.json").read_text(encoding="utf-8")
    ) == {"status": "running"}


def test_finalize_writes_result_json(tmp_path: Path):
    h = RunHandle(run_dir=tmp_path, recipe="tier", started_at="20260610T000000Z")
    h.finalize(RunResult(status="failed", duration_s=1.5, cost_usd=0.25))
    blob = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert blob == {
        "status": "failed",
        "duration_s": 1.5,
        "cost_usd": 0.25,
        "error_kind": None,
        "error_message": None,
    }
    assert stat.S_IMODE((tmp_path / "result.json").stat().st_mode) == 0o600


def test_finalize_is_idempotent_but_immutable(tmp_path: Path):
    handle = RunHandle(
        run_dir=tmp_path, recipe="tier", started_at="20260610T000000Z"
    )
    result = RunResult(status="passed", cost_usd=0.25)
    handle.finalize(result)
    handle.finalize(result)
    with pytest.raises(RunAlreadyFinalizedError, match="different"):
        handle.finalize(RunResult(status="failed"))


def test_finalize_elapsed_records_one_monotonic_snapshot(
    tmp_path: Path,
    monkeypatch,
):
    handle = RunHandle(
        run_dir=tmp_path,
        recipe="tier",
        started_at="20260610T000000Z",
        _opened_monotonic=10.0,
    )
    snapshots = iter([12.5])
    monkeypatch.setattr(runs_module.time, "monotonic", lambda: next(snapshots))

    measured = handle.finalize_elapsed(
        RunResult(status="passed", cost_usd=0.25)
    )

    assert measured == RunResult(
        status="passed", duration_s=2.5, cost_usd=0.25
    )
    assert json.loads((tmp_path / "result.json").read_text())[
        "duration_s"
    ] == 2.5


def test_finalize_elapsed_rejects_explicit_duration(tmp_path: Path):
    handle = RunHandle(
        run_dir=tmp_path, recipe="tier", started_at="20260610T000000Z"
    )

    with pytest.raises(ValueError, match="retain its default value"):
        handle.finalize_elapsed(RunResult(status="passed", duration_s=1.0))

    assert not (tmp_path / "result.json").exists()


@pytest.mark.parametrize("snapshot", [9.5, float("nan"), float("inf")])
def test_finalize_elapsed_rejects_invalid_monotonic_measurement(
    tmp_path: Path,
    monkeypatch,
    snapshot: float,
):
    handle = RunHandle(
        run_dir=tmp_path,
        recipe="tier",
        started_at="20260610T000000Z",
        _opened_monotonic=10.0,
    )
    monkeypatch.setattr(runs_module.time, "monotonic", lambda: snapshot)

    with pytest.raises(ValueError, match="finite and non-negative"):
        handle.finalize_elapsed(RunResult(status="passed"))

    assert not (tmp_path / "result.json").exists()


def test_finalize_elapsed_same_terminal_is_idempotent(
    tmp_path: Path,
    monkeypatch,
):
    handle = RunHandle(
        run_dir=tmp_path,
        recipe="tier",
        started_at="20260610T000000Z",
        _opened_monotonic=10.0,
    )
    snapshots = iter([12.5])
    monkeypatch.setattr(runs_module.time, "monotonic", lambda: next(snapshots))
    terminal = RunResult(status="passed", cost_usd=0.25)

    first = handle.finalize_elapsed(terminal)
    repeated = handle.finalize_elapsed(terminal)

    assert repeated == first
    assert repeated.duration_s == 2.5


def test_finalize_elapsed_conflicting_terminal_fails_closed(
    tmp_path: Path,
    monkeypatch,
):
    handle = RunHandle(
        run_dir=tmp_path,
        recipe="tier",
        started_at="20260610T000000Z",
        _opened_monotonic=10.0,
    )
    monkeypatch.setattr(runs_module.time, "monotonic", lambda: 12.5)
    handle.finalize_elapsed(RunResult(status="passed", cost_usd=0.25))

    with pytest.raises(RunAlreadyFinalizedError, match="different"):
        handle.finalize_elapsed(RunResult(status="failed", cost_usd=0.25))

    assert json.loads((tmp_path / "result.json").read_text())["status"] == (
        "passed"
    )


def test_finalize_elapsed_fails_closed_for_unrepresentable_recorded_duration(
    tmp_path: Path,
):
    handle = RunHandle(
        run_dir=tmp_path, recipe="tier", started_at="20260610T000000Z"
    )
    huge_duration = 10**400
    handle.finalize(RunResult(status="passed", duration_s=huge_duration))

    with pytest.raises(RunAlreadyFinalizedError, match="different"):
        handle.finalize_elapsed(RunResult(status="passed"))

    assert json.loads((tmp_path / "result.json").read_text())[
        "duration_s"
    ] == huge_duration


def test_events_cannot_be_appended_after_finalize(tmp_path: Path):
    handle = RunHandle(
        run_dir=tmp_path, recipe="tier", started_at="20260610T000000Z"
    )
    handle.finalize(RunResult(status="passed"))
    with pytest.raises(RunAlreadyFinalizedError, match="finalized"):
        handle.write_event({"late": True})


def test_finalize_waits_for_inflight_event_publication(
    tmp_path: Path, monkeypatch,
):
    handle = RunHandle(
        run_dir=tmp_path, recipe="tier", started_at="20260610T000000Z"
    )
    write_entered = threading.Event()
    release_write = threading.Event()
    real_write = runs_module.os.write

    def blocking_write(fd, payload):
        write_entered.set()
        assert release_write.wait(timeout=2)
        return real_write(fd, payload)

    monkeypatch.setattr(runs_module.os, "write", blocking_write)
    writer = threading.Thread(target=lambda: handle.write_event({"phase": 1}))
    finalizer = threading.Thread(
        target=lambda: handle.finalize(RunResult(status="passed"))
    )
    writer.start()
    assert write_entered.wait(timeout=2)
    finalizer.start()
    finalizer.join(timeout=0.05)
    assert finalizer.is_alive(), "finalize bypassed the in-flight event lock"

    release_write.set()
    writer.join(timeout=2)
    finalizer.join(timeout=2)
    assert not writer.is_alive()
    assert not finalizer.is_alive()
    assert json.loads((tmp_path / "events.jsonl").read_text()) == {"phase": 1}
    assert json.loads((tmp_path / "result.json").read_text())["status"] == "passed"


def test_context_manager_finalizes_success(tmp_path: Path):
    handle = RunHandle(
        run_dir=tmp_path, recipe="tier", started_at="20260610T000000Z"
    )
    with handle:
        handle.write_event({"phase": "work"})
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["status"] == "completed"
    assert result["error_kind"] is None


def test_context_manager_finalizes_exception_without_suppressing(tmp_path: Path):
    handle = RunHandle(
        run_dir=tmp_path, recipe="tier", started_at="20260610T000000Z"
    )
    with pytest.raises(RuntimeError, match="boom"):
        with handle:
            raise RuntimeError("boom")
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["status"] == "failed"
    assert result["error_kind"] == "RuntimeError"
    assert result["error_message"] == "boom"
