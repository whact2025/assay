# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Single-shot benchmark runner primitive.

Subprocess-invokes the runner identified by ``BenchConfig.runner`` (a dotted
Python module path), passing a runner-visible JSON envelope containing
``bench_id``, ``parameters``, and ``env`` to stdin. Stdout may be one complete
JSON KPI object or progress text whose final non-empty line is that object. On
success the primitive returns a typed ``Measurement`` with parsed values, a
configuration hash, and execution context.

Contract is described in detail in ``assay/types/bench.py``.

The primitive owns:
- subprocess launch (``python -m <runner>``)
- stdin / stdout / stderr handling
- timeout enforcement
- JSON payload validation
- persisting ``raw_output`` to the corpus
- Measurement assembly (commit, config_hash, execution context)

It does NOT own:
- GPU enumeration / scheduling — the caller constructs resource-bound configs
- regression envelopes / comparisons — consumers can compose review with a
  perf-envelope rubric
- breach testing or gates — recipe-layer concerns

Raises ``BenchRunnerError`` (a ``ValueError`` subclass) on timeout, nonzero
exit, malformed or empty KPI data, a KPI that cannot be coerced to a finite
float, or a declared raw-output file that does not exist. The caller may
compose retries; ``bench`` itself performs one subprocess attempt.
"""

from __future__ import annotations

import datetime as _datetime
import json
import math
import os
import platform
import re
import secrets
import socket
import subprocess
import sys
import time
from pathlib import Path
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ..canonical import hash_record
from ..corpus.runs import _atomic_replace
from ..types.artifacts import Measurement
from ..types.bench import BenchConfig
from ..types.execution import ExecutionContext
from ..types.identity import ContentHash

if TYPE_CHECKING:
    from ..corpus.store import Corpus


class BenchRunnerError(ValueError):
    """The benchmark subprocess failed or violated its output contract."""


def _log_stem(bench_id: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", bench_id).strip("._-") or "bench"
    if stem != bench_id:
        import hashlib

        stem += "_" + hashlib.sha256(bench_id.encode()).hexdigest()[:8]
    return stem[:128]


def _config_hash(config: BenchConfig) -> str:
    """Stable content hash of the runner-visible portion of the config.

    Excludes ``working_dir`` (environmental, not semantic) and
    ``timeout_seconds`` (runner-irrelevant). Includes ``env`` because
    overrides like CUDA_VISIBLE_DEVICES legitimately change the
    Measurement's identity.
    """
    payload = hash_record(
        {
            "bench_id": config.bench_id,
            "runner": config.runner,
            "parameters": config.parameters,
            "env": config.env,
        }
    )
    # hash_record returns ``sha256:<hex>``; strip the scheme prefix to
    # produce a bare hex that satisfies Measurement.config_hash callers
    # using either form.
    return payload.split(":", 1)[1] if ":" in payload else payload


def _snapshot_execution(*, recipe: str | None = None) -> ExecutionContext:
    return ExecutionContext(
        timestamp=_datetime.datetime.now(_datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        machine=socket.gethostname(),
        cwd=os.getcwd(),
        toolchain={
            "assay": _read_assay_version(),
            "python": platform.python_version(),
        },
        backend=None,
        model=None,
        recipe=recipe,
        recipe_commit=None,
        gpu_model=None,
        run_id=None,
    )


def _read_assay_version() -> str:
    try:
        from importlib.metadata import version
        return version("assay")
    except Exception:  # noqa: BLE001
        return "0.0.0"


def _parse_kpi_payload(stdout: bytes) -> dict:
    """Extract the runner's KPI JSON object from stdout bytes.

    Prefer the last ``\\n``-terminated JSON object in the stream so a
    runner can emit progress text before the final payload. If the
    entire stream is a single JSON object, that's accepted too.
    """
    text = stdout.decode("utf-8", errors="replace").strip()
    if not text:
        raise BenchRunnerError("runner produced empty stdout")
    # Strategy 1: entire stdout is one JSON object.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    # Strategy 2: last line is a JSON object.
    last_line = text.splitlines()[-1].strip()
    try:
        obj = json.loads(last_line)
    except json.JSONDecodeError as exc:
        raise BenchRunnerError(
            f"could not parse a JSON KPI payload from runner stdout: "
            f"{exc}; stdout tail={text[-200:]!r}"
        ) from None
    if not isinstance(obj, dict):
        raise BenchRunnerError(
            f"runner KPI payload must be a JSON object, got {type(obj).__name__}"
        )
    return obj


def _coerce_values(raw: object) -> dict[str, float]:
    if not isinstance(raw, dict):
        raise BenchRunnerError(
            f"runner KPI payload `values` must be an object, got "
            f"{type(raw).__name__}"
        )
    coerced: dict[str, float] = {}
    for k, v in raw.items():
        try:
            fv = float(v)
        except (TypeError, ValueError) as exc:
            raise BenchRunnerError(
                f"runner KPI {k!r}={v!r} is not numeric: {exc}"
            ) from None
        # A runner that crashed or divided by zero mid-compute can emit NaN or
        # Infinity, and JSON round-trips
        # them silently — storing one poisons every downstream
        # scaling/regression aggregate. A non-finite KPI is a failed run,
        # not a measurement.
        if not math.isfinite(fv):
            raise BenchRunnerError(
                f"runner KPI {k!r}={v!r} is not finite (NaN/Inf) — treating "
                f"as a failed run rather than storing a poison value"
            )
        coerced[str(k)] = fv
    return coerced


def bench(
    config: BenchConfig,
    *,
    corpus: "Corpus",
    log_dir: Path | None = None,
    process_runner: Callable[..., subprocess.CompletedProcess[Any]] | None = None,
) -> Measurement:
    """Execute one benchmark configuration and persist its raw evidence.

    ``log_dir`` optionally receives one immutable per-attempt log and a
    latest-log snapshot. Raw output is independently content-addressed in the
    Corpus and referenced by the returned Measurement. ``process_runner`` is
    an optional ``subprocess.run``-compatible execution boundary; standalone
    Assay uses the standard library default, while a journal-owning workflow
    can inject positive descendant settlement.
    """
    if not config.runner:
        raise BenchRunnerError(
            f"BenchConfig.runner must be a non-empty module path; "
            f"got {config.runner!r}"
        )

    cmd = [sys.executable, "-m", config.runner]
    cwd = config.working_dir or os.getcwd()
    proc_env = {**os.environ, **(config.env or {})}

    config_json = json.dumps(
        {
            "bench_id": config.bench_id,
            "parameters": config.parameters,
            "env": config.env,
        }
    ).encode("utf-8")

    log_path: Path | None = None
    latest_log_path: Path | None = None
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        stem = _log_stem(config.bench_id)
        log_path = log_dir / (
            f"bench_{stem}_{time.time_ns()}_{secrets.token_hex(4)}.log"
        )
        latest_log_path = log_dir / f"bench_{stem}.log"

    def _persist_log(data: bytes) -> None:
        if log_path is None or latest_log_path is None:
            return
        _atomic_replace(log_path, data)
        _atomic_replace(latest_log_path, data)

    try:
        run_process = (
            process_runner if process_runner is not None else subprocess.run
        )
        completed = run_process(
            cmd,
            input=config_json,
            cwd=cwd,
            env=proc_env,
            capture_output=True,
            timeout=config.timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        _persist_log(
            f"# command\n{cmd}\n# cwd\n{cwd}\n# returncode\nTIMEOUT\n"
            f"# stdout\n".encode()
            + (exc.stdout or b"")
            + b"\n# stderr\n"
            + (exc.stderr or b"")
        )
        raise BenchRunnerError(
            f"runner {config.runner!r} timed out after "
            f"{config.timeout_seconds}s"
        ) from exc

    stderr_text = completed.stderr.decode("utf-8", errors="replace")
    stdout_text = completed.stdout.decode("utf-8", errors="replace")
    _persist_log(
        (
            f"# command\n{cmd}\n# cwd\n{cwd}\n# returncode\n"
            f"{completed.returncode}\n# stdout\n{stdout_text}\n# stderr\n"
            f"{stderr_text}\n"
        ).encode("utf-8")
    )

    if completed.returncode != 0:
        raise BenchRunnerError(
            f"runner {config.runner!r} exited {completed.returncode}; "
            f"stderr tail={stderr_text[-400:]!r}"
        )

    payload = _parse_kpi_payload(completed.stdout)
    values = _coerce_values(payload.get("values"))
    if not values:
        raise BenchRunnerError(
            "runner KPI payload has empty / missing `values` map"
        )

    # Persist any raw output the runner emitted.
    raw_blob: bytes
    if "raw_output" in payload and isinstance(payload["raw_output"], str):
        raw_blob = payload["raw_output"].encode("utf-8")
    elif "raw_output_path" in payload and isinstance(
        payload["raw_output_path"], str
    ):
        rp = Path(payload["raw_output_path"])
        candidate = rp if rp.is_absolute() else (Path(cwd) / rp)
        # A runner that declares raw_output_path but leaves no file there had
        # a problem; refuse to silently store a zero-byte
        # trace (indistinguishable from a legitimately-empty capture).
        if not candidate.is_file():
            raise BenchRunnerError(
                f"runner declared raw_output_path "
                f"{payload['raw_output_path']!r} but no file exists at "
                f"{candidate} — refusing to store an empty trace that would "
                f"mask an incomplete run"
            )
        raw_blob = candidate.read_bytes()
    else:
        # Fall back to recording the full stdout payload so the Measurement
        # always has a content-addressable trail.
        raw_blob = completed.stdout

    raw_ref = corpus.put_artifact(
        raw_blob,
        kind="traces",
        name_hint=f"bench_{config.bench_id}.raw",
    )

    return Measurement(
        bench_id=config.bench_id,
        commit=config.commit or "unknown",
        config_hash=_config_hash(config),
        values=values,
        raw_output_ref=ContentHash(raw_ref),
        execution=_snapshot_execution(),
    )


__all__ = ["bench", "BenchRunnerError"]
