# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Constraint and criterion primitives for assay.

Constraints describe what a Spec authors as boundary conditions on the
work; criteria describe what the work must achieve. Both follow the
same discriminated-union shape: a ``kind: str`` field discriminates the
concrete dataclass type at YAML-load time via the registries in
``assay.canonical.registry``.

The Protocols below are marker types only — they encode the contract
``kind: str`` so that recipes and validators can introspect generically.
The generic builtin subtypes live in this module. Consumer packages may add
domain-specific dataclasses to the subtype registries before parsing.

Leaf module: stdlib only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class Constraint(Protocol):
    kind: str


@runtime_checkable
class Criterion(Protocol):
    kind: str


# `kind` is declared last so non-default fields never follow defaulted
# ones (Python 3.10+ dataclass rule).
@dataclass(frozen=True)
class BudgetConstraint:
    usd_limit: float | None = None
    time_limit_hours: float | None = None
    kind: str = "budget"


@dataclass(frozen=True)
class BlastRadiusConstraint:
    no_edits_outside: tuple[str, ...] = ()
    kind: str = "blast_radius"


@dataclass(frozen=True)
class ActivationConstraint:
    status: str
    reason: str
    kind: str = "activation"

    def __post_init__(self) -> None:
        if self.status not in {"active", "parked"}:
            raise ValueError(
                "activation status must be 'active' or 'parked', "
                f"got {self.status!r}"
            )


@dataclass(frozen=True)
class AssertionCriterion:
    test_name: str
    must_pass: bool
    kind: str = "assertion"


@dataclass(frozen=True)
class ParityCriterion:
    reference_commit: str
    test_name: str
    kind: str = "parity"
