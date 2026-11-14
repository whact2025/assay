# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Fail-closed admission for backend execution evidence.

The backend protocol stays source-compatible: third-party adapters may still
return the conservative, incomplete :class:`ExecutionCompleteness` default.
Consumers choose whether such a successful result is diagnostic-only or must
be converted into a typed failure before it can enter a promotion path.
Semantic consumers may retain a terminal, trace-complete result with explicit
token/cost debt, but that debt must remain visible at later control boundaries.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

from .invocation import IncompleteEvidence, InvocationResult


ResultEvidenceAdmission = Literal["enforce", "semantic", "diagnostic"]
RESULT_EVIDENCE_ADMISSION_MODES: frozenset[str] = frozenset(
    {"enforce", "semantic", "diagnostic"}
)

_COMPLETENESS_FIELDS = (
    "token_usage_complete",
    "cost_complete",
    "terminal_complete",
    "trace_complete",
)


def missing_execution_evidence(result: InvocationResult) -> tuple[str, ...]:
    """Return the stable names of incomplete evidence fields."""

    return tuple(
        field_name
        for field_name in _COMPLETENESS_FIELDS
        if not getattr(result.completeness, field_name)
    )


def admit_result_evidence(
    result: InvocationResult,
    *,
    mode: ResultEvidenceAdmission,
) -> InvocationResult:
    """Admit a result according to its intended downstream authority.

    Existing backend failures are preserved: incomplete evidence on an error
    response is useful diagnostic metadata, not a reason to mask the original
    failure. Complete successes are returned unchanged.

    ``semantic`` admits an otherwise successful result only when terminal and
    trace evidence are complete. Missing token or cost evidence remains on the
    returned ``InvocationResult`` as explicit accounting debt. This lets a
    consumer parse and persist content from a successful resumed attempt
    without pretending that every physical attempt was metered. The consumer
    must carry that debt across its own paid-work and promotion boundaries.

    ``enforce`` additionally requires complete token and cost evidence.
    ``diagnostic`` leaves all nominal successes untouched.
    """

    if mode not in RESULT_EVIDENCE_ADMISSION_MODES:
        raise ValueError(
            f"unknown result evidence admission mode {mode!r}; expected one "
            f"of {sorted(RESULT_EVIDENCE_ADMISSION_MODES)}"
        )
    missing = missing_execution_evidence(result)
    if result.error is not None or not missing or mode == "diagnostic":
        return result
    if mode == "semantic" and not any(
        field_name in {"terminal_complete", "trace_complete"}
        for field_name in missing
    ):
        return result
    return replace(
        result,
        error=IncompleteEvidence(
            missing=missing,
            message=(
                "backend result is diagnostic-only because required "
                "execution evidence is incomplete: "
                + ", ".join(missing)
            ),
        ),
    )


__all__ = [
    "RESULT_EVIDENCE_ADMISSION_MODES",
    "ResultEvidenceAdmission",
    "admit_result_evidence",
    "missing_execution_evidence",
]
