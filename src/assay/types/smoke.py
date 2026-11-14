# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""``SmokeResult`` — outcome of the ``smoke_check`` cheap-gate primitive.

The smoke check is a pre-build syntax/parse gate that runs before
expensive build/verify subprocesses. It is deliberately small in scope:
language-aware subprocess (``cargo check`` for Rust, ``compileall`` plus
optional ``ruff --select F`` for Python, skip-with-pass for recognized
``.cpp``/``.cc``/``.cxx``/``.h``/``.hpp``/``.cu`` extensions), timeout-bounded,
returns a typed result the workflow can route on.

The motivation is to reject a syntactically broken diff before a much more
expensive build or test phase. The workflow, not this result type, decides how
to route a failure.

Leaf module: stdlib only.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SmokeResult:
    """Outcome of one ``smoke_check`` invocation.

    Fields:
      passed: True when the selected lightweight command succeeds. False on a
        Rust ``cargo check`` failure, a Python syntax/ruff-F failure, a missing
        required tool, or required-command timeout. This is not dependency-
        import resolution; the recognized C++/CUDA suffix set is explicitly a
        skip-with-pass lane, and optional Ruff absence/timeout is a soft pass.
      language: The language family that drove the check
        (``"rust"`` / ``"python"`` / ``"cpp"`` / ``None`` when no
        recognised files were in scope).
      log_excerpt: Tail-truncated stdout/stderr of the smoke subprocess
        (up to ``SMOKE_LOG_MAX_BYTES``). Empty after an executed command
        succeeds; the C++/CUDA skip-with-pass lane carries a routing message.
      duration_ms: Wall-clock duration of the smoke check, useful for
        operator visibility into when the gate itself becomes the
        bottleneck.
    """

    passed: bool
    language: str | None
    log_excerpt: str
    duration_ms: int


__all__ = ["SmokeResult"]
