# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for ``assay.primitives.bench``.

Uses a synthetic Python runner (written to disk and invoked via
``-m`` against a tmp sys.path prefix) so we exercise the real
subprocess path without needing a benchmark project.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from assay import BenchConfig
from assay.corpus.store import Corpus
from assay.primitives.bench import (
    BenchRunnerError,
    _coerce_values,
    _config_hash,
    _parse_kpi_payload,
    bench,
)


# ----------------------------------------------------------------------
# Helpers — write a one-off runner module that reads stdin JSON and
# emits a predictable KPI payload.
# ----------------------------------------------------------------------


def _write_runner(tmp_path: Path, name: str, body: str) -> str:
    """Write ``tmp_path/<name>.py`` and return the dotted module name.

    Prepends ``tmp_path`` to sys.path of the spawned subprocess via
    ``PYTHONPATH`` so ``python -m <name>`` resolves it.
    """
    (tmp_path / f"{name}.py").write_text(textwrap.dedent(body))
    return name


def _bench_env_with_pythonpath(tmp_path: Path) -> dict[str, str]:
    return {"PYTHONPATH": str(tmp_path) + os.pathsep + os.environ.get("PYTHONPATH", "")}


# ----------------------------------------------------------------------
# _parse_kpi_payload
# ----------------------------------------------------------------------


def test_parse_full_stdout_is_one_object():
    obj = _parse_kpi_payload(b'{"values": {"fps": 60.0}}')
    assert obj == {"values": {"fps": 60.0}}


def test_parse_falls_back_to_last_line():
    obj = _parse_kpi_payload(b"running\nbench\n{\"values\":{\"fps\":30}}\n")
    assert obj == {"values": {"fps": 30}}


def test_parse_rejects_empty():
    with pytest.raises(BenchRunnerError, match="empty stdout"):
        _parse_kpi_payload(b"")


def test_parse_rejects_garbage():
    with pytest.raises(BenchRunnerError, match="could not parse"):
        _parse_kpi_payload(b"this is not json\n")


def test_parse_rejects_non_object():
    with pytest.raises(BenchRunnerError, match="must be a JSON object"):
        _parse_kpi_payload(b"[1, 2, 3]")


# ----------------------------------------------------------------------
# _coerce_values
# ----------------------------------------------------------------------


def test_coerce_values_accepts_numeric():
    out = _coerce_values({"a": 1, "b": 2.5, "c": "3"})
    assert out == {"a": 1.0, "b": 2.5, "c": 3.0}


def test_coerce_values_rejects_non_numeric():
    with pytest.raises(BenchRunnerError, match="not numeric"):
        _coerce_values({"a": "not a number"})


def test_coerce_values_rejects_non_object():
    with pytest.raises(BenchRunnerError, match="must be an object"):
        _coerce_values([1, 2, 3])


# ----------------------------------------------------------------------
# _config_hash determinism
# ----------------------------------------------------------------------


def test_config_hash_is_deterministic():
    c1 = BenchConfig(bench_id="x", runner="r", parameters={"a": 1}, env={"E": "v"})
    c2 = BenchConfig(bench_id="x", runner="r", parameters={"a": 1}, env={"E": "v"})
    assert _config_hash(c1) == _config_hash(c2)


def test_config_hash_changes_with_env_override():
    c1 = BenchConfig(bench_id="x", runner="r", env={"CUDA_VISIBLE_DEVICES": "0"})
    c2 = BenchConfig(bench_id="x", runner="r", env={"CUDA_VISIBLE_DEVICES": "1"})
    assert _config_hash(c1) != _config_hash(c2)


def test_config_hash_ignores_working_dir():
    c1 = BenchConfig(bench_id="x", runner="r", working_dir="/a")
    c2 = BenchConfig(bench_id="x", runner="r", working_dir="/b")
    assert _config_hash(c1) == _config_hash(c2)


# ----------------------------------------------------------------------
# bench() end-to-end via a synthetic runner subprocess
# ----------------------------------------------------------------------


def test_bench_runs_and_returns_measurement(tmp_path: Path):
    name = _write_runner(
        tmp_path,
        "_passing_runner",
        """
        import json, sys
        cfg = json.loads(sys.stdin.read())
        out = {"values": {"fps": 42.0}, "raw_output": "ok"}
        print(json.dumps(out))
        """,
    )
    corpus = Corpus(tmp_path / "_corpus")
    cfg = BenchConfig(
        bench_id="b1",
        runner=name,
        parameters={},
        env=_bench_env_with_pythonpath(tmp_path),
        commit="cafecafe",
    )
    m = bench(cfg, corpus=corpus, log_dir=tmp_path / "_logs")
    assert m.bench_id == "b1"
    assert m.commit == "cafecafe"
    assert m.values == {"fps": 42.0}
    assert m.raw_output_ref.startswith("sha256:")
    assert (tmp_path / "_logs" / "bench_b1.log").exists()
    assert len(tuple((tmp_path / "_logs").glob("bench_b1_*.log"))) == 1
    # Raw output retrievable.
    assert corpus.get_artifact(m.raw_output_ref) == b"ok"


def test_bench_uses_injected_process_boundary(tmp_path: Path) -> None:
    name = "_injected_runner"
    corpus = Corpus(tmp_path / "_corpus")
    cfg = BenchConfig(
        bench_id="injected",
        runner=name,
        parameters={"samples": 3},
        env={"BENCH_DEVICE": "gpu0"},
        working_dir=str(tmp_path),
        timeout_seconds=19,
        commit="cafecafe",
    )
    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    def supervised(command, **kwargs):
        calls.append((tuple(command), kwargs))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=b'{"values":{"fps":7.0}}',
            stderr=b"",
        )

    measurement = bench(
        cfg,
        corpus=corpus,
        process_runner=supervised,
    )

    assert measurement.values == {"fps": 7.0}
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command == (sys.executable, "-m", name)
    assert kwargs["input"] == (
        b'{"bench_id": "injected", "parameters": {"samples": 3}, '
        b'"env": {"BENCH_DEVICE": "gpu0"}}'
    )
    assert kwargs["cwd"] == str(tmp_path)
    assert kwargs["env"]["BENCH_DEVICE"] == "gpu0"
    assert kwargs["capture_output"] is True
    assert kwargs["timeout"] == 19
    assert kwargs["check"] is False


def test_falsey_injected_bench_runner_is_not_bypassed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FalseyRunner:
        def __init__(self) -> None:
            self.called = False

        def __bool__(self) -> bool:
            return False

        def __call__(self, command, **_kwargs):
            self.called = True
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=b'{"values":{"fps":2.0}}',
                stderr=b"",
            )

    def forbidden_stdlib(*_args, **_kwargs):
        raise AssertionError("stdlib runner must not be selected")

    monkeypatch.setattr(subprocess, "run", forbidden_stdlib)
    runner = FalseyRunner()
    measurement = bench(
        BenchConfig(bench_id="falsey", runner="runner"),
        corpus=Corpus(tmp_path / "_corpus"),
        process_runner=runner,
    )

    assert runner.called is True
    assert measurement.values == {"fps": 2.0}


def test_bench_logs_are_unique_and_bench_id_cannot_escape(tmp_path: Path):
    name = _write_runner(
        tmp_path,
        "_repeat_runner",
        """
        import json
        print(json.dumps({"values": {"fps": 1.0}}))
        """,
    )
    corpus = Corpus(tmp_path / "_corpus")
    cfg = BenchConfig(
        bench_id="../unsafe/id",
        runner=name,
        parameters={},
        env=_bench_env_with_pythonpath(tmp_path),
        commit="cafecafe",
    )
    log_dir = tmp_path / "_logs"

    bench(cfg, corpus=corpus, log_dir=log_dir)
    bench(cfg, corpus=corpus, log_dir=log_dir)

    files = tuple(log_dir.iterdir())
    assert len(files) == 3  # two immutable attempts + one latest view
    assert all(path.parent == log_dir for path in files)
    assert not (tmp_path / "unsafe").exists()


def test_bench_raises_on_nonzero_exit(tmp_path: Path):
    name = _write_runner(
        tmp_path,
        "_failing_runner",
        """
        import sys
        print("disaster", file=sys.stderr)
        sys.exit(7)
        """,
    )
    corpus = Corpus(tmp_path / "_corpus")
    cfg = BenchConfig(
        bench_id="b2",
        runner=name,
        env=_bench_env_with_pythonpath(tmp_path),
    )
    with pytest.raises(BenchRunnerError, match=r"exited 7"):
        bench(cfg, corpus=corpus)


def test_bench_raises_on_unparseable_output(tmp_path: Path):
    name = _write_runner(
        tmp_path,
        "_garbage_runner",
        """
        import sys
        print("not json")
        """,
    )
    corpus = Corpus(tmp_path / "_corpus")
    cfg = BenchConfig(
        bench_id="b3",
        runner=name,
        env=_bench_env_with_pythonpath(tmp_path),
    )
    with pytest.raises(BenchRunnerError, match="could not parse"):
        bench(cfg, corpus=corpus)


def test_bench_raises_on_empty_values(tmp_path: Path):
    name = _write_runner(
        tmp_path,
        "_empty_values_runner",
        """
        import json
        print(json.dumps({"values": {}}))
        """,
    )
    corpus = Corpus(tmp_path / "_corpus")
    cfg = BenchConfig(
        bench_id="b4",
        runner=name,
        env=_bench_env_with_pythonpath(tmp_path),
    )
    with pytest.raises(BenchRunnerError, match="empty / missing"):
        bench(cfg, corpus=corpus)


def test_bench_rejects_empty_runner_path(tmp_path: Path):
    corpus = Corpus(tmp_path / "_corpus")
    cfg = BenchConfig(bench_id="b5", runner="")
    with pytest.raises(BenchRunnerError, match="non-empty module path"):
        bench(cfg, corpus=corpus)


def test_bench_respects_timeout(tmp_path: Path):
    name = _write_runner(
        tmp_path,
        "_slow_runner",
        """
        import time
        time.sleep(5)
        """,
    )
    corpus = Corpus(tmp_path / "_corpus")
    cfg = BenchConfig(
        bench_id="b6",
        runner=name,
        env=_bench_env_with_pythonpath(tmp_path),
        timeout_seconds=0.5,
    )
    with pytest.raises(BenchRunnerError, match="timed out"):
        bench(cfg, corpus=corpus)


def test_bench_uses_raw_output_path_when_provided(tmp_path: Path):
    """If the runner writes a separate raw_output file, bench should pick it up."""
    raw_dump = tmp_path / "raw_dump.txt"
    raw_dump.write_text("hello-from-runner")
    name = _write_runner(
        tmp_path,
        "_path_runner",
        f"""
        import json
        print(json.dumps({{"values": {{"fps": 1.0}}, "raw_output_path": "{raw_dump.as_posix()}"}}))
        """,
    )
    corpus = Corpus(tmp_path / "_corpus")
    cfg = BenchConfig(
        bench_id="b7",
        runner=name,
        env=_bench_env_with_pythonpath(tmp_path),
    )
    m = bench(cfg, corpus=corpus)
    assert corpus.get_artifact(m.raw_output_ref) == b"hello-from-runner"


def test_bench_rejects_nonfinite_kpi(tmp_path: Path):
    """ABE-2: a NaN/Inf KPI is a failed run, not a stored poison value."""
    name = _write_runner(
        tmp_path, "_nan_runner",
        """
        import json
        print(json.dumps({"values": {"fps": float("nan")}}))
        """,
    )
    corpus = Corpus(tmp_path / "_corpus")
    cfg = BenchConfig(bench_id="bnan", runner=name,
                      env=_bench_env_with_pythonpath(tmp_path))
    with pytest.raises(BenchRunnerError, match="finite"):
        bench(cfg, corpus=corpus)


def test_bench_raises_on_missing_declared_raw_output_path(tmp_path: Path):
    """ABE-1: a runner that declares raw_output_path but leaves no file
    there must fail, not store a silent zero-byte trace."""
    name = _write_runner(
        tmp_path, "_missing_path_runner",
        """
        import json
        print(json.dumps({"values": {"fps": 1.0}, "raw_output_path": "/no/such/file.txt"}))
        """,
    )
    corpus = Corpus(tmp_path / "_corpus")
    cfg = BenchConfig(bench_id="bmiss", runner=name,
                      env=_bench_env_with_pythonpath(tmp_path))
    with pytest.raises(BenchRunnerError, match="no file exists"):
        bench(cfg, corpus=corpus)
