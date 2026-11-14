# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Discriminator registry for constraint and criterion subtypes.

This module is a leaf in the canonical subpackage: it imports no
``assay.types`` modules to avoid circular imports. :mod:`assay.bootstrap`
installs builtin kinds idempotently; normal package import currently reaches
that bootstrap, and Spec/Corpus boundaries also call it defensively. Consumers
can register additional dataclass types before parsing their discriminators.
"""

CONSTRAINT_TYPES: dict[str, type] = {}
CRITERION_TYPES: dict[str, type] = {}
