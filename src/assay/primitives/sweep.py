# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Multi-point, parameterized benchmark execution.

Infrastructure-agnostic fan-out over :func:`assay.primitives.bench` calls.
``parallelism`` is a worker-pool size — just an integer. **``sweep``
knows nothing about GPUs.**

Callers own resource discovery, health filtering, and the construction of each
``BenchConfig.env``. A runner consumes only its own configuration; neither it
nor ``sweep`` receives a worker index or an implicit GPU assignment.

The returned iterator preserves the **input order** of ``configs`` regardless
of worker completion order.

Failures follow ``Executor.map`` semantics and are observed in input order
while the call eagerly materializes the result list. Work already running in
other threads is not cancelled, and no iterator is returned after a failure.
Callers that need per-point isolation must compose it outside ``sweep``.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Iterator

from ..types.artifacts import Measurement
from ..types.bench import BenchConfig
from .bench import bench as _bench

if TYPE_CHECKING:
    from ..corpus.store import Corpus


def sweep(
    configs: Iterable[BenchConfig],
    *,
    parallelism: int,
    corpus: "Corpus",
    log_dir: Path | None = None,
) -> Iterator[Measurement]:
    """Run each ``BenchConfig`` in ``configs`` and yield ``Measurement``s.

    The function eagerly materializes ``configs`` and all results. Despite the
    iterator return type, no Measurement is yielded until every worker has
    completed successfully. Result order matches input order.
    """
    if parallelism < 1:
        raise ValueError(f"parallelism must be >= 1, got {parallelism}")
    materialized = list(configs)
    if not materialized:
        return iter(())
    run_one = partial(_bench, corpus=corpus, log_dir=log_dir)
    with ThreadPoolExecutor(
        max_workers=parallelism,
        thread_name_prefix="assay-sweep",
    ) as executor:
        results = list(executor.map(run_one, materialized))
    return iter(results)


__all__ = ["sweep"]
