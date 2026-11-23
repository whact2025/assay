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
to route a failure; ``infrastructure_failure`` tells it whether the failed
check judged the code at all.

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
        An infrastructure failure starts with a ``<language>: <reason>`` line.
      duration_ms: Wall-clock duration of the smoke check, useful for
        operator visibility into when the gate itself becomes the
        bottleneck.
      infrastructure_failure: True when a required tool is missing, a
        required command timed out, or a tool ended without a verdict on
        the source. The result is still a failure (``passed`` is False;
        the combination with ``passed=True`` is rejected), but it is not
        evidence that the checked source is broken. False for every pass and
        for a failure the tool attributed to source content. The flag does
        not mean "the environment is at fault" or "a retry would succeed": a
        tool configuration in the checked tree that the tool rejects, or a
        Cargo failure that names no source location, is flagged as well.
        Rejected Rust runner parameters are a configuration failure and are
        not flagged.
    """

    passed: bool
    language: str | None
    log_excerpt: str
    duration_ms: int
    infrastructure_failure: bool = False

    def __post_init__(self) -> None:
        if self.infrastructure_failure and self.passed:
            raise ValueError(
                "SmokeResult.infrastructure_failure requires passed=False"
            )


__all__ = ["SmokeResult"]
