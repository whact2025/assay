# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Tier DAG primitives for assay.

``TierDAG`` carries planner output as immutable data. Workflow consumers walk
it via ``topo_order()``; Assay's planning primitive also inspects it for
structural safety and size-policy checks before accepting a candidate.

Leaf module: stdlib only.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import re
from typing import TYPE_CHECKING

from .frozen import freeze_dict

if TYPE_CHECKING:
    # Type-only forward ref for Tier.smoke_bench_config; importing it
    # under TYPE_CHECKING keeps dag.py a runtime stdlib-only leaf module.
    from .bench import BenchConfig


@dataclass(frozen=True)
class MutationRegion:
    """Source-level mutation scope marker for one file.

    When an ImplementationDirective declares a region for a file, the
    implementer prompt constrains edits to lines between the named start/end
    markers. Two enforcement layers are expected:

    1. ``assay.primitives.implement`` renders the marker constraint into the
       model prompt.
    2. The consuming workflow must compare resulting diff hunks with the
       declared regions and reject out-of-range writes. Assay records the
       regions but does not perform that post-diff validation itself.

    Marker values are literal substrings and do not include a language comment
    prefix. Interpretation of comment syntax belongs to the consumer's
    deterministic validator.
    """

    file: str
    start_marker: str = "CRUCIBLE-MUTATE-START"
    end_marker: str = "CRUCIBLE-MUTATE-END"


@dataclass(frozen=True)
class ReadOnlySourceReference:
    """Identity and inclusive line locator for existing source evidence.

    This record does not grant mutation authority or attest that source bytes
    were inspected. Consumers must verify the whole-file digest and capture
    the selected lines before presenting them as grounded review evidence.
    """

    reference_id: str
    path: str
    sha256: str
    start_line: int
    end_line: int

    def __post_init__(self) -> None:
        if not isinstance(self.reference_id, str) or re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", self.reference_id
        ) is None:
            raise ValueError("reference_id must be a stable identifier of 1-64 characters")
        if (
            not isinstance(self.path, str)
            or not self.path
            or len(self.path) > 4096
            or any(part in {"", ".", ".."} for part in self.path.split("/"))
            or any(
                ord(char) < 32 or ord(char) == 127 or 0xD800 <= ord(char) <= 0xDFFF
                for char in self.path
            )
            or any(char in self.path for char in "\\*?[]")
        ):
            raise ValueError("path must be a normalized relative POSIX file path")
        if not isinstance(self.sha256, str) or re.fullmatch(
            r"sha256:[0-9a-f]{64}", self.sha256
        ) is None:
            raise ValueError("sha256 must contain 'sha256:' and 64 lowercase hex digits")
        if (
            type(self.start_line) is not int
            or type(self.end_line) is not int
            or self.start_line < 1
            or self.end_line < self.start_line
        ):
            raise ValueError("source lines must be positive inclusive integers in order")


@dataclass(frozen=True)
class ImplementationDirective:
    slice_id: str
    intent: str
    files_in_scope: tuple[str, ...]
    creates_paths: tuple[str, ...]
    verify_mode: str
    blast_radius: dict
    # Source-level mutation regions. When non-empty, the implementer
    # is constrained to edit only lines between the named start/end
    # marker comments in the listed files. Files not listed here have
    # no source-level constraint (only the coarse ``files_in_scope`` ∪
    # ``creates_paths`` bound). Default empty tuple — most directives
    # don't carry regions; per-file opt-in is the contract.
    mutation_regions: tuple[MutationRegion, ...] = ()
    # Exact, verbatim clauses copied from the authoritative task root.
    # These are requirements, not planner-generated implementation notes:
    # consumers must preserve their wording and give them precedence over
    # ``intent`` when the two conflict. Empty means no quoted root clauses.
    requirement_refs: tuple[str, ...] = field(
        default=(),
        metadata={"persist_optional": True, "canonical_omit_empty": True},
    )
    # Existing source/test evidence only; never part of the mutation scope.
    read_only_references: tuple[ReadOnlySourceReference, ...] = field(
        default=(),
        metadata={"persist_optional": True, "canonical_omit_empty": True},
    )

    def __post_init__(self) -> None:
        if type(self.read_only_references) is not tuple or any(
            type(ref) is not ReadOnlySourceReference for ref in self.read_only_references
        ):
            raise TypeError("read_only_references must be a tuple of ReadOnlySourceReference")
        ids = [ref.reference_id for ref in self.read_only_references]
        if len(set(ids)) != len(ids):
            raise ValueError("read_only_references must have distinct reference IDs")
        object.__setattr__(self, "blast_radius", freeze_dict(self.blast_radius))


@dataclass(frozen=True)
class Slice:
    id: str
    directive: ImplementationDirective


@dataclass(frozen=True)
class Tier:
    tier_id: str
    slices: tuple[Slice, ...]
    depends_on: tuple[str, ...]
    # Optional benchmark configuration interpreted by the consuming workflow.
    # Assay carries this data but does not choose when to run it, select a
    # baseline, or define a regression threshold. None means no tier benchmark.
    smoke_bench_config: "BenchConfig | None" = None


@dataclass(frozen=True)
class TierDAG:
    tiers: tuple[Tier, ...]

    def topo_order(self) -> list[Tier]:
        """Kahn's algorithm over ``tier_id`` -> ``depends_on``.

        Raises ``ValueError('cycle detected')`` if the graph is not a
        DAG. Returns tiers in dependency-resolved order.
        """
        by_id: dict[str, Tier] = {t.tier_id: t for t in self.tiers}
        in_deg: dict[str, int] = {t.tier_id: len(t.depends_on) for t in self.tiers}
        # adjacency: dep -> [dependents]
        adj: dict[str, list[str]] = {tid: [] for tid in by_id}
        for t in self.tiers:
            for dep in t.depends_on:
                if dep not in by_id:
                    raise ValueError(f"unknown dependency {dep!r} from {t.tier_id!r}")
                adj[dep].append(t.tier_id)

        queue: deque[str] = deque(
            tid for tid, deg in in_deg.items() if deg == 0
        )
        ordered: list[Tier] = []
        while queue:
            tid = queue.popleft()
            ordered.append(by_id[tid])
            for dependent in adj[tid]:
                in_deg[dependent] -= 1
                if in_deg[dependent] == 0:
                    queue.append(dependent)

        if len(ordered) != len(self.tiers):
            raise ValueError("cycle detected")
        return ordered

    def all_slices(self) -> list[Slice]:
        out: list[Slice] = []
        for t in self.topo_order():
            out.extend(t.slices)
        return out

    def parallel_conflicts(
        self,
    ) -> list[tuple[str, str, str, tuple[str, ...]]]:
        """Return same-tier slice pairs whose write scopes overlap.

        Slices within one tier run in parallel, so their
        ``files_in_scope ∪ creates_paths`` MUST be pairwise disjoint —
        two slices editing the same file race at implement time and
        smear each other's edits. Returns
        ``(tier_id, slice_a, slice_b, overlapping_paths)`` per
        conflicting pair; empty when the DAG is parallel-safe.

        The planner is *prompted* to keep tiers disjoint; this makes
        the invariant a structural, deterministic check the recipe
        layer can enforce, rather than advisory prose that drifts.
        """
        conflicts: list[tuple[str, str, str, tuple[str, ...]]] = []
        for t in self.tiers:
            scopes: list[tuple[str, set[str]]] = [
                (s.id, set(s.directive.files_in_scope) | set(s.directive.creates_paths))
                for s in t.slices
            ]
            for i in range(len(scopes)):
                for j in range(i + 1, len(scopes)):
                    overlap = scopes[i][1] & scopes[j][1]
                    if overlap:
                        conflicts.append(
                            (t.tier_id, scopes[i][0], scopes[j][0], tuple(sorted(overlap)))
                        )
        return conflicts
