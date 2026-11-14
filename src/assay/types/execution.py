# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Execution-context primitives for assay.

Defines:
    ExecutionContext: frozen dataclass capturing the runtime
        environment of a single assay invocation (timestamp,
        machine, cwd, toolchain mapping, and optional backend /
        model / recipe / gpu / run-id descriptors).

Every field is required and has no default. Callers may construct the dataclass
with positional or keyword arguments as usual.

Leaf module: stdlib only.
"""

from __future__ import annotations

from dataclasses import dataclass

from .frozen import freeze_dict


@dataclass(frozen=True)
class ExecutionContext:
    timestamp: str
    machine: str
    cwd: str
    toolchain: dict[str, str]
    backend: str | None
    model: str | None
    recipe: str | None
    recipe_commit: str | None
    gpu_model: str | None
    run_id: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "toolchain", freeze_dict(self.toolchain))
