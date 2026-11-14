# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for ``assay.primitives.parallel``."""

from __future__ import annotations

import threading
import time

import pytest

from assay import Invocation, InvocationResult, Message, Usage
from assay.primitives.parallel import parallel


class _RecordingBackend:
    """Backend stub that records call order + can simulate latency."""

    name = "stub"
    model = "stub-model"

    def __init__(self, latency_s: float = 0.0):
        self._latency_s = latency_s
        self.calls: list[Invocation] = []
        self._lock = threading.Lock()
        self.max_concurrent = 0
        self._inflight = 0

    def invoke(self, invocation: Invocation) -> InvocationResult:
        with self._lock:
            self._inflight += 1
            self.max_concurrent = max(self.max_concurrent, self._inflight)
        if self._latency_s:
            time.sleep(self._latency_s)
        with self._lock:
            self._inflight -= 1
            self.calls.append(invocation)
        return InvocationResult(
            final_message=f"echo:{invocation.user}",
            transcript=(Message(role="assistant", content=invocation.user),),
            usage=Usage(),
            stop_reason="end_turn",
        )


def _inv(user: str) -> Invocation:
    return Invocation(
        system="sys",
        user=user,
        tools=None,
        max_turns=1,
        budget_usd=None,
    )


# ----------------------------------------------------------------------
# Ordering / shape
# ----------------------------------------------------------------------


def test_parallel_preserves_input_order():
    backend = _RecordingBackend()
    invocations = [_inv(f"q{i}") for i in range(5)]
    results = parallel(invocations, backend, max_workers=3)
    assert [r.final_message for r in results] == [f"echo:q{i}" for i in range(5)]


def test_parallel_empty_list_returns_empty_list():
    backend = _RecordingBackend()
    assert parallel([], backend) == []


def test_parallel_rejects_non_list():
    backend = _RecordingBackend()
    with pytest.raises(TypeError, match="must be a list"):
        parallel(iter([_inv("a")]), backend)  # type: ignore[arg-type]


def test_parallel_rejects_zero_workers():
    backend = _RecordingBackend()
    with pytest.raises(ValueError, match=">= 1"):
        parallel([_inv("a")], backend, max_workers=0)


# ----------------------------------------------------------------------
# Concurrency cap (max_workers)
# ----------------------------------------------------------------------


def test_parallel_respects_max_workers():
    # 8 invocations with 50ms latency each; max_workers=2 caps in-flight.
    backend = _RecordingBackend(latency_s=0.05)
    invocations = [_inv(f"q{i}") for i in range(8)]
    parallel(invocations, backend, max_workers=2)
    assert backend.max_concurrent <= 2
    # And we DO get parallelism: max_concurrent should hit the cap, not 1.
    assert backend.max_concurrent == 2


def test_parallel_single_worker_serializes():
    backend = _RecordingBackend(latency_s=0.02)
    invocations = [_inv(f"q{i}") for i in range(4)]
    parallel(invocations, backend, max_workers=1)
    assert backend.max_concurrent == 1


# ----------------------------------------------------------------------
# Error propagation
# ----------------------------------------------------------------------


def test_parallel_propagates_backend_exceptions():
    class BoomBackend:
        name = "boom"
        model = "boom-model"

        def invoke(self, invocation):
            raise RuntimeError("backend failed")

    with pytest.raises(RuntimeError, match="backend failed"):
        parallel([_inv("a")], BoomBackend(), max_workers=2)
