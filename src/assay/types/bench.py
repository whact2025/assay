# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Configuration for one benchmark subprocess invocation.

The dataclass is intentionally opaque-ish: ``parameters`` is a free-form
dict the runner understands, ``env`` carries per-invocation environment
overrides (e.g. ``CUDA_VISIBLE_DEVICES`` from a sweep adapter), and
``runner`` is a dotted Python module path identifying the entry point.

The runner subprocess contract:

- ``assay.primitives.bench`` subprocess-invokes ``python -m <runner>``.
- It passes ``bench_id``, ``parameters``, and ``env`` as a JSON object on
  stdin. Process-only fields such as ``runner``, ``working_dir``, timeout, and
  commit provenance are not included.
- The runner emits a JSON object of the following shape as either all of
  stdout or its final line (earlier progress lines are permitted)::

      {
        "values":           {"<kpi>": <float-coercible JSON value>, ...},
        "raw_output":        "<optional inline text>",   // small
        "raw_output_path":   "<optional path>",          // large
      }

  ``values`` must be nonempty. Each value is passed to ``float(value)`` and the
  result must be finite; numeric strings and booleans are therefore accepted
  by the implemented contract. Either raw output field may be absent, in which
  case the complete stdout stream is retained. Inline ``raw_output`` takes
  precedence when both are present; a declared ``raw_output_path`` must name
  an existing file.

- The runner exits 0 on success. A nonzero exit or contract violation raises
  ``assay.primitives.bench.BenchRunnerError`` and produces no Measurement.

Leaf module: stdlib only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .frozen import freeze_dict


def _empty_dict() -> dict:
    return {}


def _empty_str_dict() -> dict[str, str]:
    return {}


@dataclass(frozen=True)
class BenchConfig:
    bench_id: str
    runner: str
    parameters: dict = field(default_factory=_empty_dict)
    env: dict[str, str] = field(default_factory=_empty_str_dict)
    timeout_seconds: float | None = None
    working_dir: str | None = None
    commit: str | None = None  # base commit for Measurement provenance

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", freeze_dict(self.parameters))
        object.__setattr__(self, "env", freeze_dict(self.env))


__all__ = ["BenchConfig"]
