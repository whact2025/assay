# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Bootstrap the kind→dataclass registry.

The Constraint and Criterion registries in ``assay.canonical.registry`` start
empty. ``populate_registry()`` inserts Assay's builtin subtypes. Calling it is
idempotent (it uses ``setdefault``); importing this module also calls it once.
``Spec.from_yaml`` and ``Corpus`` hydrate the builtins at their boundaries, so
normal callers do not need an explicit bootstrap import.

Consumer packages may call ``populate_registry`` and then register additional
kinds directly in ``CONSTRAINT_TYPES`` or ``CRITERION_TYPES``.
"""

from __future__ import annotations

from .canonical.registry import CONSTRAINT_TYPES, CRITERION_TYPES
from .types.constraints import (
    ActivationConstraint,
    AssertionCriterion,
    BlastRadiusConstraint,
    BudgetConstraint,
    ParityCriterion,
)


def populate_registry() -> None:
    """Register every builtin kind without replacing consumer overrides."""
    CONSTRAINT_TYPES.setdefault("budget", BudgetConstraint)
    CONSTRAINT_TYPES.setdefault("blast_radius", BlastRadiusConstraint)
    CONSTRAINT_TYPES.setdefault("activation", ActivationConstraint)
    CRITERION_TYPES.setdefault("assertion", AssertionCriterion)
    CRITERION_TYPES.setdefault("parity", ParityCriterion)


# Module-import-time hydration. Python's import lock guarantees this
# runs at most once per process; setdefault makes it idempotent under
# any reload or monkey-patch scenarios.
populate_registry()
