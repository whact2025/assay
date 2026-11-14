# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Spec-referenced artifact records.

Each record lives behind one of a Spec's content-addressed ``*_refs`` tuples.
Consumers resolve references with ``Corpus.get_record`` to hydrate typed
instances on demand.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .backend_execution import ExecutionCompleteness
from .execution import ExecutionContext
from .frozen import freeze_dict
from .identity import ContentHash
from .invocation import Usage, UsagePricingEvidence
from .cost_bounds import UsageCostBounds


@dataclass(frozen=True)
class Implementation:
    slice_id: str
    base_commit: str
    result_commit: str | None
    diff_ref: ContentHash
    transcript_ref: ContentHash
    files_modified: tuple[str, ...]
    classification: str
    execution: ExecutionContext
    # Newly dirty paths observed outside ``files_in_scope ∪ creates_paths``.
    # Paths already dirty before invocation are excluded to avoid attributing
    # ambient/sibling work to this call; consequently, edits to a pre-dirty
    # out-of-scope path are not detectable here. Empty means no new escape was
    # observed, not proof that the complete workspace was pristine.
    scope_escapes: tuple[str, ...]
    # Execution-side artifact refs. ``assay.primitives.implement`` adds a
    # ``capture_error`` entry when trustworthy Git capture fails. A wrapping
    # workflow may add stable keys for build/verify stderr, warnings, tool
    # output, or log excerpts after persisting those bytes with
    # ``corpus.put_artifact``. Default empty keeps clean standalone use terse.
    artifacts: dict[str, ContentHash] = field(default_factory=dict)
    # Total cost (USD) of the backend invocation that produced this
    # Implementation, as reported by the backend's Usage record. 0.0
    # for backends that don't surface cost (deterministic stubs in
    # tests, vendor SDKs without a cost field). Populated by
    # ``assay.primitives.implement`` from ``InvocationResult.usage.cost_usd``;
    # downstream workflows can account for spend without retaining the
    # transient Usage object.
    cost_usd: float = 0.0
    # Token telemetry from the backend invocation. Canonical zero values are
    # omitted and the strict decoder supplies the declared optional defaults.
    input_tokens: int = field(
        default=0,
        metadata={"persist_optional": True, "canonical_omit_empty": True},
    )
    output_tokens: int = field(
        default=0,
        metadata={"persist_optional": True, "canonical_omit_empty": True},
    )
    # Preserve non-natural backend termination independently of the coarse
    # classification. A tool loop may hit max_turns without returning a
    # BackendError; recording the stop reason keeps a turn-exhausted no-op
    # distinguishable from an intentional empty result.
    stop_reason: str | None = field(
        default=None,
        metadata={"persist_optional": True, "canonical_omit_empty": True},
    )
    # Aggregate provider evidence across every physical attempt contributing
    # to this implementation. ``None`` is reserved for deterministic or
    # legacy producer paths with no InvocationResult; the implement primitive
    # always copies the backend result's explicit completeness value.
    execution_completeness: ExecutionCompleteness | None = field(
        default=None,
        metadata={
            "persist_optional": True,
            "canonical_omit_empty": True,
        },
    )
    # Durable, prompt-free per-response pricing inputs copied from the
    # InvocationResult. ``None`` preserves deterministic and legacy records.
    pricing_evidence: UsagePricingEvidence | None = field(
        default=None,
        metadata={
            "persist_optional": True,
            "canonical_omit_empty": True,
        },
    )
    # Keep additive fields after the established positional constructor
    # sequence. Explicit integers are the authoritative cache-token subsets
    # of the aggregate input count; incomplete pricing evidence is not a
    # substitute. ``None`` distinguishes a legacy record that predates these
    # fields from an exact zero reported by a new implementer invocation.
    cache_creation_tokens: int | None = field(
        default=None,
        metadata={"persist_optional": True, "canonical_omit_default": True},
    )
    cache_read_tokens: int | None = field(
        default=None,
        metadata={"persist_optional": True, "canonical_omit_default": True},
    )
    cost_bounds: UsageCostBounds | None = field(
        default=None,
        metadata={"persist_optional": True, "canonical_omit_empty": True},
    )

    @property
    def accounting_debt(self) -> tuple[str, ...]:
        """Names of incomplete aggregate accounting fields, if any."""

        completeness = self.execution_completeness
        if completeness is None:
            return ()
        token_usage_complete = completeness.token_usage_complete
        if (
            token_usage_complete
            and self.cache_creation_tokens is None
            and self.cache_read_tokens is None
            and not (
                self.pricing_evidence is not None
                and self.pricing_evidence.complete
            )
        ):
            # Legacy Implementations can assert complete inclusive token
            # totals without carrying the cache subsets added later. Only a
            # complete legacy ledger can recover those exact subsets.
            token_usage_complete = False
        debt: list[str] = []
        if not token_usage_complete:
            debt.append("token_usage_complete")
        if not completeness.cost_complete:
            debt.append("cost_complete")
        return tuple(debt)

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifacts", freeze_dict(self.artifacts))
        if self.cost_bounds is not None and type(self.cost_bounds) is not UsageCostBounds:
            raise TypeError("Implementation.cost_bounds must be UsageCostBounds or None")
        if self.cost_bounds is not None and any(
            segment.cache_read_tokens is None for segment in self.cost_bounds.segments
        ) and self.execution_completeness is not None and (
            self.execution_completeness.token_usage_complete or self.execution_completeness.cost_complete
        ):
            raise ValueError("unknown cache reads cannot have complete actual accounting")
        if (
            self.execution_completeness is not None
            and type(self.execution_completeness) is not ExecutionCompleteness
        ):
            raise TypeError(
                "Implementation.execution_completeness must be an "
                "ExecutionCompleteness or None"
            )
        if self.pricing_evidence is not None and (
            type(self.pricing_evidence) is not UsagePricingEvidence
        ):
            raise TypeError(
                "Implementation.pricing_evidence must be a "
                "UsagePricingEvidence or None"
            )
        cache_values = (
            self.cache_creation_tokens,
            self.cache_read_tokens,
        )
        if (cache_values[0] is None) != (cache_values[1] is None):
            raise ValueError(
                "Implementation cache-token aggregates must either both be "
                "present or both be absent"
            )
        if cache_values[0] is not None:
            for field_name, value in zip(
                ("cache_creation_tokens", "cache_read_tokens"),
                cache_values,
            ):
                if type(value) is not int or value < 0:
                    raise ValueError(
                        f"Implementation.{field_name} must be an exact "
                        "non-negative integer when present"
                    )

        completeness = self.execution_completeness
        if completeness is not None and completeness.token_usage_complete:
            for field_name in ("input_tokens", "output_tokens"):
                value = getattr(self, field_name)
                if type(value) is not int or value < 0:
                    raise ValueError(
                        f"Implementation.{field_name} must be an exact "
                        "non-negative integer when token usage is complete"
                    )
            if cache_values[0] is not None and (
                cache_values[0] + cache_values[1] > self.input_tokens
            ):
                raise ValueError(
                    "Implementation cache-token subsets must not exceed "
                    "input_tokens when token usage is complete"
                )
        if self.pricing_evidence is not None and self.pricing_evidence.complete:
            if completeness is None or not (
                completeness.token_usage_complete
                and completeness.cost_complete
            ):
                raise ValueError(
                    "complete Implementation.pricing_evidence requires "
                    "complete token usage and cost evidence"
                )
            # Legacy Implementations have no top-level cache fields. Complete
            # pricing evidence was their sole exact cache authority, so use
            # those counters only for reconciliation without changing the
            # stored record. New records must reconcile their explicit pair.
            aggregate = self.pricing_evidence.usage
            cache_creation_tokens = (
                aggregate.cache_creation_tokens
                if cache_values[0] is None
                else cache_values[0]
            )
            cache_read_tokens = (
                aggregate.cache_read_tokens
                if cache_values[1] is None
                else cache_values[1]
            )
            if not self.pricing_evidence.reconciles(
                Usage(
                    input_tokens=self.input_tokens,
                    output_tokens=self.output_tokens,
                    cost_usd=self.cost_usd,
                    cache_creation_tokens=cache_creation_tokens,
                    cache_read_tokens=cache_read_tokens,
                )
            ):
                raise ValueError(
                    "complete Implementation.pricing_evidence must reconcile "
                    "aggregate usage"
                )


@dataclass(frozen=True)
class Measurement:
    bench_id: str
    commit: str
    config_hash: str
    values: dict[str, float]
    raw_output_ref: ContentHash
    execution: ExecutionContext

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", freeze_dict(self.values))


@dataclass(frozen=True)
class LifecycleEvent:
    kind: str
    reason: str
    affected_versions: tuple[int, ...]
    execution: ExecutionContext
