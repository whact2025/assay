# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Pure, bounded fan-out over backend invocations.

Bounded ``ThreadPoolExecutor.map`` over a list of ``Invocation`` values
against a single ``Backend``. Returns results in **input order** regardless
of completion order.

The primitive does NOT:
- decide which invocations to run (recipe layer owns the call set)
- interpret results (recipe layer reads ``error``, ``final_message``, etc.)
- define retry behavior (the concrete backend adapter owns that policy)
- write a Spec version or persist anything

Because ``Backend.invoke`` is synchronous, a bounded thread executor is the
direct abstraction; this primitive has no asynchronous caller contract.

Exception semantics follow ``Executor.map``: results and exceptions are
observed in input order, while already-running synchronous backend calls own
their normal policy timeout/cancellation behavior. Python cannot safely kill
an arbitrary running thread.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

from ..types.invocation import Invocation, InvocationResult

if TYPE_CHECKING:
    from ..backend.protocol import Backend


def parallel(
    invocations: list[Invocation],
    backend: "Backend",
    *,
    max_workers: int = 4,
    log_dir: Path | None = None,
) -> list[InvocationResult]:
    """Fan out a concrete list of invocations and preserve input order.

    ``log_dir`` is accepted but not consumed; backend-specific logging must be
    configured on the backend instance.
    Exceptions are observed in input order with ``Executor.map`` semantics.
    """
    if not isinstance(invocations, list):
        # Strict list input so callers don't accidentally pass a
        # generator, whose one-shot consumption can make downstream result
        # correlation ambiguous.
        raise TypeError(
            f"invocations must be a list, got {type(invocations).__name__}"
        )
    if max_workers < 1:
        raise ValueError(f"max_workers must be >= 1, got {max_workers}")
    if not invocations:
        return []
    with ThreadPoolExecutor(
        max_workers=max_workers,
        thread_name_prefix="assay-parallel",
    ) as executor:
        return list(executor.map(backend.invoke, invocations))


__all__ = ["parallel"]
