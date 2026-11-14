# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Spec — the unified work-lifecycle entity.

A Spec is an immutable, content-addressed record carrying the full
lifecycle of one unit of work from authored intent (v0) through
generated plan / implementations / verdicts / measurements.

The ``with_*`` helpers produce new versions; assay code never mutates
a Spec. Large collections (implementations / verdicts / etc.) are
stored as ``tuple[ContentHash, ...]`` references; the Corpus resolves them to
typed records on demand.

YAML parsing:
    ``Spec.from_yaml(text)`` parses a strict schema-version-1 YAML document
    into a v0 Spec. Constraints and criteria use the discriminator registries
    from ``assay.canonical.registry``. The parser hydrates builtin kinds before
    decoding.
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import yaml

from ..canonical import hash_record
from ..canonical.registry import CONSTRAINT_TYPES, CRITERION_TYPES
from .artifacts import Implementation, LifecycleEvent, Measurement
from .constraints import Constraint, Criterion
from .dag import TierDAG
from .execution import ExecutionContext
from .findings import Verdict
from .identity import ContentHash, ProjectContext
from .policy import PlanPolicy


_UNSET = object()
_SPEC_SCHEMA_VERSION = 1
_SPEC_YAML_FIELDS = frozenset({
    "schema_version",
    "project_context",
    "intent",
    "constraints",
    "done_criteria",
    "depends_on",
    "plan_policy",
})
_PROJECT_CONTEXT_YAML_FIELDS = frozenset({
    "project",
    "repo_url",
    "base_commit",
    "branch",
    "submodule_path",
    "runners",
    "runner_parameters",
})
_CONTENT_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def _reject_unknown_fields(
    raw: dict, allowed: frozenset[str], label: str,
) -> None:
    unknown = sorted(
        (key for key in raw if key not in allowed),
        key=repr,
    )
    if unknown:
        raise ValueError(f"{label} has unknown fields: {unknown!r}")


def _validate_spec_schema_version(value: object) -> None:
    if type(value) is not int or value != _SPEC_SCHEMA_VERSION:
        raise ValueError(
            "unsupported Spec schema_version "
            f"{value!r}; expected {_SPEC_SCHEMA_VERSION}"
        )


def _parse_depends_on(raw: dict) -> tuple[ContentHash, ...]:
    if "depends_on" not in raw:
        return ()

    value = raw["depends_on"]
    if not isinstance(value, list):
        raise ValueError(
            "depends_on must be a list of canonical sha256 ContentHash strings, "
            f"got {type(value).__name__}"
        )

    dependencies: list[ContentHash] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or _CONTENT_HASH_RE.fullmatch(item) is None:
            raise ValueError(
                f"depends_on[{index}] must be 'sha256:' followed by 64 "
                f"lowercase hex digits, got {item!r}"
            )
        dependencies.append(ContentHash(item))
    return tuple(dependencies)


class UnknownConstraintKind(ValueError):
    """Raised when a Spec YAML carries a constraint/criterion ``kind``
    not registered in CONSTRAINT_TYPES/CRITERION_TYPES.

    The kind-to-dataclass registries are the extension point for consumer
    constraint and criterion subtypes.
    """


@dataclass(frozen=True)
class Spec:
    version: int
    parent: ContentHash | None
    root: ContentHash | None
    schema_version: int

    project_context: ProjectContext
    execution: ExecutionContext | None

    intent: str
    constraints: tuple[Constraint, ...]
    done_criteria: tuple[Criterion, ...]
    plan_policy: PlanPolicy | None

    depends_on: tuple[ContentHash, ...] = ()
    tier_dag: TierDAG | None = None
    implementation_refs: tuple[ContentHash, ...] = ()
    verdict_refs: tuple[ContentHash, ...] = ()
    measurement_refs: tuple[ContentHash, ...] = ()
    lifecycle_event_refs: tuple[ContentHash, ...] = ()

    def __post_init__(self) -> None:
        _validate_spec_schema_version(self.schema_version)

    # --- mutation helpers (pure; return a new Spec) -----------------

    def _next_version(self, **changes: Any) -> "Spec":
        return replace(
            self,
            version=self.version + 1,
            parent=ContentHash(hash_record(self)),
            root=self.root if self.root is not None else ContentHash(hash_record(self)),
            **changes,
        )

    def with_tier_dag(self, dag: TierDAG) -> "Spec":
        return self._next_version(tier_dag=dag)

    def with_implementation(self, impl: Implementation) -> "Spec":
        ref = ContentHash(hash_record(impl))
        return self._next_version(implementation_refs=self.implementation_refs + (ref,))

    def with_verdict(self, v: Verdict) -> "Spec":
        ref = ContentHash(hash_record(v))
        return self._next_version(verdict_refs=self.verdict_refs + (ref,))

    def with_measurement(self, m: Measurement) -> "Spec":
        ref = ContentHash(hash_record(m))
        return self._next_version(measurement_refs=self.measurement_refs + (ref,))

    def with_measurements(
        self, refs: "Iterable[ContentHash]",
    ) -> "Spec":
        """Bulk variant of ``with_measurement``: append every ref in
        ``refs`` and bump the version once. This avoids repeatedly
        re-canonicalizing an increasingly large Spec during bulk ingestion.

        ``refs`` are accepted directly rather than ``Measurement``s so
        callers can route the underlying records through
        ``Corpus.put_measurements`` (one transaction) without
        rehashing here."""
        new_refs = tuple(refs)
        if not new_refs:
            return self
        return self._next_version(
            measurement_refs=self.measurement_refs + new_refs,
        )

    def with_appended_refs(
        self,
        *,
        implementation_refs: "Iterable[ContentHash]" = (),
        verdict_refs: "Iterable[ContentHash]" = (),
        measurement_refs: "Iterable[ContentHash]" = (),
        lifecycle_event_refs: "Iterable[ContentHash]" = (),
        tier_dag: TierDAG | None | object = _UNSET,
    ) -> "Spec":
        """Append a transaction's refs and bump the lineage exactly once.

        Multi-role workflows often produce several content-addressed records
        before publishing their next Spec manifest. Chaining ``with_verdict``
        or ``with_implementation`` for each record creates intermediate parent
        hashes; if only the final Spec is persisted, those parents are
        dangling. This bulk commit primitive preserves every record while
        creating one contiguous parent edge.
        """

        changes: dict[str, Any] = {
            "implementation_refs": self.implementation_refs
            + tuple(implementation_refs),
            "verdict_refs": self.verdict_refs + tuple(verdict_refs),
            "measurement_refs": self.measurement_refs
            + tuple(measurement_refs),
            "lifecycle_event_refs": self.lifecycle_event_refs
            + tuple(lifecycle_event_refs),
        }
        if tier_dag is not _UNSET:
            changes["tier_dag"] = tier_dag
        return self._next_version(**changes)

    def with_lifecycle_event(self, ev: LifecycleEvent) -> "Spec":
        ref = ContentHash(hash_record(ev))
        return self._next_version(lifecycle_event_refs=self.lifecycle_event_refs + (ref,))

    # --- YAML construction ------------------------------------------

    @classmethod
    def from_yaml(cls, text: str) -> "Spec":
        # Hydrate the kind registry before parsing so unknown kinds
        # raise UnknownConstraintKind rather than silently constructing
        # an opaque dict.
        from .. import bootstrap as _b  # local import avoids circular
        _b.populate_registry()

        raw = yaml.safe_load(text)
        if not isinstance(raw, dict):
            raise ValueError(f"top-level YAML must be a mapping, got {type(raw).__name__}")
        _validate_spec_schema_version(raw.get("schema_version"))
        _reject_unknown_fields(raw, _SPEC_YAML_FIELDS, "Spec YAML")

        pc_raw = raw["project_context"]
        if not isinstance(pc_raw, dict):
            raise ValueError(
                "project_context must be a mapping, got "
                f"{type(pc_raw).__name__}"
            )
        _reject_unknown_fields(
            pc_raw,
            _PROJECT_CONTEXT_YAML_FIELDS,
            "project_context",
        )
        runner_parameters = pc_raw.get("runner_parameters", {})
        if not isinstance(runner_parameters, dict):
            raise ValueError(
                "project_context.runner_parameters must be a mapping, got "
                f"{type(runner_parameters).__name__}"
            )
        pc = ProjectContext(
            project=pc_raw["project"],
            repo_url=pc_raw["repo_url"],
            base_commit=pc_raw["base_commit"],
            branch=pc_raw.get("branch"),
            submodule_path=pc_raw.get("submodule_path"),
            runners=dict(pc_raw.get("runners") or {}),
            runner_parameters=dict(runner_parameters),
        )

        constraints = tuple(
            _construct_kinded(c, CONSTRAINT_TYPES, "constraint")
            for c in raw.get("constraints") or ()
        )
        criteria = tuple(
            _construct_kinded(c, CRITERION_TYPES, "criterion")
            for c in raw.get("done_criteria") or ()
        )
        depends_on = _parse_depends_on(raw)

        plan_policy = None
        pp = raw.get("plan_policy")
        if pp is not None and not isinstance(pp, dict):
            raise ValueError(
                "plan_policy must be a mapping, got "
                f"{type(pp).__name__}"
            )
        if pp:
            _reject_unknown_fields(
                pp,
                frozenset(field.name for field in dataclasses.fields(PlanPolicy)),
                "plan_policy",
            )
            # Optional policy fields inherit PlanPolicy's canonical defaults.
            kwargs: dict = {
                "max_critique_rounds": pp["max_critique_rounds"],
                "judge_required": pp["judge_required"],
                "breaker_required": pp["breaker_required"],
                "persistence_veto_threshold": pp["persistence_veto_threshold"],
            }
            for opt_key in (
                "fresh_judge_reassessment_min_prior_rounds",
                "judge_veto_weighted_threshold",
                "cumulative_inflation_ratio",
                "cumulative_inflation_floor",
                "max_total_slices",
                "max_total_tiers",
                "inflation_compaction_retries",
                "max_verification_only_empty_scope_slices",
                "planner_guidance",
                "max_critic_failures",
                "max_judge_failures",
            ):
                if opt_key in pp:
                    kwargs[opt_key] = pp[opt_key]
            severity_weights = pp.get("judge_veto_severity_weights")
            if severity_weights is not None:
                if isinstance(severity_weights, dict):
                    severity_weights = tuple(severity_weights.items())
                kwargs["judge_veto_severity_weights"] = severity_weights
            plan_policy = PlanPolicy(**kwargs)

        return cls(
            version=0,
            parent=None,
            root=None,
            schema_version=raw["schema_version"],
            project_context=pc,
            execution=None,
            intent=raw["intent"],
            constraints=constraints,
            done_criteria=criteria,
            plan_policy=plan_policy,
            depends_on=depends_on,
        )

    @classmethod
    def load_v0(cls, yaml_path: Path) -> "Spec":
        return cls.from_yaml(Path(yaml_path).read_text(encoding="utf-8"))


def _construct_kinded(raw: dict, registry: dict[str, type], label: str):
    if not isinstance(raw, dict):
        raise ValueError(f"{label} entry must be a mapping, got {type(raw).__name__}")
    payload = dict(raw)  # copy; we'll pop kind
    kind = payload.pop("kind", None)
    if kind is None:
        raise UnknownConstraintKind(
            f"{label} missing 'kind' discriminator; register a concrete "
            "kind before parsing"
        )
    cls = registry.get(kind)
    if cls is None:
        raise UnknownConstraintKind(
            f"unknown {label} kind {kind!r}; register it before parsing"
        )
    field_names = frozenset(
        field.name for field in dataclasses.fields(cls) if field.name != "kind"
    )
    _reject_unknown_fields(payload, field_names, f"{label} kind {kind!r}")
    return cls(kind=kind, **payload)
