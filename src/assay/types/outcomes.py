# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Typed terminal outcomes for multi-stage Assay primitives."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from .backend_execution import ExecutionCompleteness
from .dag import TierDAG
from .identity import ContentHash
from .invocation import Usage, UsagePricingEvidence
from .cost_bounds import UsageCostBounds
from .spec import Spec


PlanDecision = Literal["accepted", "rejected", "inconclusive"]


@dataclass(frozen=True)
class PlanOutcome:
    """Terminal result of generated or fixed-candidate plan adjudication.

    Only an accepted outcome attaches the candidate DAG to ``spec``;
    non-acceptance preserves any prior executable DAG.  A rejected or
    inconclusive candidate remains available for diagnostics via
    ``candidate_dag``/``candidate_ref`` but cannot be mistaken for approval by
    downstream recipes. ``rounds_used`` counts adjudication rounds; it is not
    a Planner-call count because fixed-candidate review deliberately makes no
    Planner call.
    """

    spec: Spec
    candidate_dag: TierDAG | None
    candidate_ref: ContentHash | None
    decision: PlanDecision
    status: str
    rounds_used: int
    final_critic_decision: str | None
    final_judge_decision: str | None
    reason: str
    cost_usd: float = 0.0
    candidate_validation_repairs_used: int = 0
    # Aggregate model usage across Planner, Critic, and Judge calls. These
    # additive fields mirror Verdict's persistence shape: legacy PlanOutcome
    # records retain their canonical bytes, while ``usage`` below exposes the
    # complete typed projection to new consumers.
    input_tokens: int = field(
        default=0,
        metadata={"persist_optional": True, "canonical_omit_empty": True},
    )
    output_tokens: int = field(
        default=0,
        metadata={"persist_optional": True, "canonical_omit_empty": True},
    )
    cache_creation_tokens: int = field(
        default=0,
        metadata={"persist_optional": True, "canonical_omit_empty": True},
    )
    cache_read_tokens: int = field(
        default=0,
        metadata={"persist_optional": True, "canonical_omit_empty": True},
    )
    # Logical-call completeness is the conjunction of every model-backed
    # result contributing to the outcome. ``None`` preserves legacy and
    # deterministic outcomes that carry no model invocation evidence.
    execution_completeness: ExecutionCompleteness | None = field(
        default=None,
        metadata={"persist_optional": True, "canonical_omit_empty": True},
    )
    # Ordered physical-response pricing inputs, concatenated in protocol call
    # order (Planner, Critic, optional Judge, then the next semantic round).
    pricing_evidence: UsagePricingEvidence | None = field(
        default=None,
        metadata={"persist_optional": True, "canonical_omit_empty": True},
    )
    cost_bounds: UsageCostBounds | None = field(
        default=None,
        metadata={"persist_optional": True, "canonical_omit_empty": True},
    )

    @property
    def accepted(self) -> bool:
        return self.decision == "accepted"

    @property
    def usage(self) -> Usage:
        """Return aggregate inclusive usage for the complete plan protocol."""

        return Usage(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cost_usd=self.cost_usd,
            cache_creation_tokens=self.cache_creation_tokens,
            cache_read_tokens=self.cache_read_tokens,
            cost_bounds=self.cost_bounds,
        )

    @property
    def accounting_debt(self) -> tuple[str, ...]:
        """Names of incomplete aggregate accounting fields, if any."""

        completeness = self.execution_completeness
        if completeness is None:
            return ()
        return tuple(
            field_name
            for field_name in ("token_usage_complete", "cost_complete")
            if not getattr(completeness, field_name)
        )

    def __post_init__(self) -> None:
        if self.cost_bounds is not None and type(self.cost_bounds) is not UsageCostBounds:
            raise TypeError("PlanOutcome.cost_bounds must be UsageCostBounds or None")
        if self.cost_bounds is not None and any(
            segment.cache_read_tokens is None for segment in self.cost_bounds.segments
        ) and self.execution_completeness is not None and (
            self.execution_completeness.token_usage_complete or self.execution_completeness.cost_complete
        ):
            raise ValueError("unknown cache reads cannot have complete actual accounting")
        completeness = self.execution_completeness
        if (
            completeness is not None
            and type(completeness) is not ExecutionCompleteness
        ):
            raise TypeError(
                "PlanOutcome.execution_completeness must be an "
                "ExecutionCompleteness or None"
            )
        pricing_evidence = self.pricing_evidence
        if pricing_evidence is not None and (
            type(pricing_evidence) is not UsagePricingEvidence
        ):
            raise TypeError(
                "PlanOutcome.pricing_evidence must be a UsagePricingEvidence "
                "or None"
            )
        if completeness is not None and completeness.token_usage_complete:
            for field_name in (
                "input_tokens",
                "output_tokens",
                "cache_creation_tokens",
                "cache_read_tokens",
            ):
                value = getattr(self, field_name)
                if type(value) is not int or value < 0:
                    raise ValueError(
                        f"PlanOutcome.{field_name} must be an exact "
                        "non-negative integer when token usage is complete"
                    )
            if (
                self.cache_creation_tokens + self.cache_read_tokens
                > self.input_tokens
            ):
                raise ValueError(
                    "PlanOutcome cache-token subsets must not exceed "
                    "input_tokens when token usage is complete"
                )
        if completeness is not None and completeness.cost_complete:
            if (
                type(self.cost_usd) not in (int, float)
                or not math.isfinite(self.cost_usd)
                or self.cost_usd < 0
            ):
                raise ValueError(
                    "PlanOutcome.cost_usd must be a finite non-negative "
                    "number when cost is complete"
                )
        if pricing_evidence is not None and pricing_evidence.complete:
            if completeness is None or not (
                completeness.token_usage_complete and completeness.cost_complete
            ):
                raise ValueError(
                    "complete PlanOutcome.pricing_evidence requires complete "
                    "token usage and cost evidence"
                )
            if not pricing_evidence.reconciles(self.usage):
                raise ValueError(
                    "complete PlanOutcome.pricing_evidence must reconcile "
                    "aggregate usage"
                )


__all__ = ["PlanDecision", "PlanOutcome"]
