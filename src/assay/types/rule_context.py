# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""``RuleContext`` — opt-in environment for context-aware validation rules.

Most validation rules are pure functions of the ``Spec`` dataclass: they
inspect intent, DAG structure, constraints, and criteria without I/O. Such
rules satisfy the canonical determinism contract: same inputs and rule
implementation produce the same Verdict, allowing the Corpus to deduplicate
it by content hash.

A small minority of useful rules need to ground their checks against
the filesystem — e.g., "every path in ``files_in_scope`` must exist"
or "every ``creates_paths`` entry's parent directory must exist on
disk." Forcing those checks into pure rules is impossible; threading
``workspace_root`` through every rule's signature would impose the
dependency on rules that do not need it.

``RuleContext`` is a small dataclass carrying optional ambient state, passed
only to rules that opt in via signature shape. The dispatcher
inside ``assay.primitives.validate`` introspects each resolved rule's
arity:

- ``rule(spec)``           — pure (runs always)
- ``rule(spec, ctx)``      — context-aware (requires the caller to pass
                              ``ctx``)

If a selected ruleset contains a context-aware rule and the caller omits
``ctx``, validation raises ``RuleContextRequiredError`` before any rule runs.
There is intentionally no skip behavior: a Verdict cannot claim complete
coverage after silently omitting a selected rule.

Verdicts produced by context-aware rules reflect the **working tree
at validation time** on the machine where validate ran. Two machines
with divergent checkouts (different submodule pointers, WIP edits,
stale caches) can produce different verdicts for the same Spec —
that disagreement is the diagnostic, not a bug: the spec's
``project_context.base_commit`` declares the intended tree, and a
divergent local checkout failing validation can be a meaningful signal.
``hash_record`` does not incorporate ``ctx`` itself; only the findings emitted
by the rule enter the Verdict's canonical payload.

Leaf module: stdlib only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuleContext:
    """Ambient environment for context-aware validation rules.

    All fields default to ``None`` so a caller can construct a minimal context.
    Each rule defines whether a missing field is a finding, an empty result, or
    another explicit outcome; the dispatcher only requires that the context
    object itself is present for a two-argument rule.

    Fields:
        workspace_root: Absolute path of the workspace root. Rules joining
            ``workspace_root`` and ``submodule_path`` can resolve paths
            relative to the Spec's declared project lane.
        submodule_path: Spec's declared submodule path
            (mirror of ``spec.project_context.submodule_path``,
            duplicated here so context-aware rules don't have to
            re-read the Spec to compute paths).
    """

    workspace_root: Path | None = None
    submodule_path: str | None = None
