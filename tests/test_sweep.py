# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for ``assay.primitives.sweep``.

Exercises the multi-config fan-out + ordering guarantee using the
same synthetic-runner-on-disk trick as ``test_bench``.
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest

from assay import BenchConfig
from assay.corpus.store import Corpus
from assay.primitives.bench import BenchRunnerError
from assay.primitives.sweep import sweep


def _write_runner(tmp_path: Path, name: str, body: str) -> str:
    (tmp_path / f"{name}.py").write_text(textwrap.dedent(body))
    return name


def _env_with_pythonpath(tmp_path: Path) -> dict[str, str]:
    return {"PYTHONPATH": str(tmp_path) + os.pathsep + os.environ.get("PYTHONPATH", "")}


# ----------------------------------------------------------------------
# Happy path: yields results in input order
# ----------------------------------------------------------------------


def test_sweep_returns_in_input_order(tmp_path: Path):
    name = _write_runner(
        tmp_path,
        "_echo_runner",
        """
        import json, sys
        cfg = json.loads(sys.stdin.read())
        params = cfg.get("parameters", {})
        print(json.dumps({"values": {"x": float(params.get("x", 0))}}))
        """,
    )
    corpus = Corpus(tmp_path / "_corpus")
    env = _env_with_pythonpath(tmp_path)
    configs = [
        BenchConfig(bench_id=f"b{i}", runner=name, parameters={"x": i}, env=env)
        for i in range(5)
    ]
    results = list(sweep(configs, parallelism=3, corpus=corpus))
    assert [r.values["x"] for r in results] == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert [r.bench_id for r in results] == [f"b{i}" for i in range(5)]


def test_sweep_empty_returns_empty(tmp_path: Path):
    corpus = Corpus(tmp_path / "_corpus")
    assert list(sweep([], parallelism=4, corpus=corpus)) == []


def test_sweep_rejects_zero_parallelism(tmp_path: Path):
    corpus = Corpus(tmp_path / "_corpus")
    with pytest.raises(ValueError, match=">= 1"):
        list(sweep([], parallelism=0, corpus=corpus))


def test_sweep_propagates_runner_errors(tmp_path: Path):
    name = _write_runner(
        tmp_path,
        "_break_runner",
        """
        import sys
        sys.exit(1)
        """,
    )
    corpus = Corpus(tmp_path / "_corpus")
    env = _env_with_pythonpath(tmp_path)
    configs = [BenchConfig(bench_id="b", runner=name, env=env)]
    with pytest.raises(BenchRunnerError):
        list(sweep(configs, parallelism=1, corpus=corpus))


def test_sweep_materializes_generator_inputs(tmp_path: Path):
    """Configs passed as a generator should still produce results in input order."""
    name = _write_runner(
        tmp_path,
        "_id_runner",
        """
        import json, sys
        cfg = json.loads(sys.stdin.read())
        print(json.dumps({"values": {"id": float(cfg.get("parameters", {}).get("id", -1))}}))
        """,
    )
    corpus = Corpus(tmp_path / "_corpus")
    env = _env_with_pythonpath(tmp_path)

    def gen():
        for i in range(3):
            yield BenchConfig(
                bench_id=f"g{i}",
                runner=name,
                parameters={"id": i},
                env=env,
            )

    results = list(sweep(gen(), parallelism=2, corpus=corpus))
    assert [r.values["id"] for r in results] == [0.0, 1.0, 2.0]
