# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Public re-export for the corpus subpackage.

``Corpus`` is the on-disk store; ``RunHandle`` and ``RunResult`` are
the per-invocation primitives. Importing this module is sufficient for
``from assay.corpus import Corpus, RunHandle``.
"""

from __future__ import annotations

from .codec import RecordDecodeError
from .runs import RunAlreadyFinalizedError, RunHandle, RunResult
from .store import (
    CommitRefs,
    CommitResult,
    ConcurrentSpecError,
    Corpus,
    RetainedSpecAuthority,
    SpecAuthorityAcquisitionLease,
    SpecAuthorityCleanupError,
)

__all__ = [
    "Corpus",
    "CommitRefs",
    "CommitResult",
    "ConcurrentSpecError",
    "RetainedSpecAuthority",
    "SpecAuthorityAcquisitionLease",
    "SpecAuthorityCleanupError",
    "RecordDecodeError",
    "RunHandle",
    "RunResult",
    "RunAlreadyFinalizedError",
]
