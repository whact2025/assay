# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Policy primitives for assay.

Defines:
    PlanPolicy: frozen dataclass carrying planning-loop limits, Judge routing,
        persistence/inflation gates, and consumer policy metadata.
    BackendPolicy: frozen dataclass capturing backend-invocation policy
        (retry budget, timeout, cancellation / max-turn handling).

Leaf module: stdlib only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class PlanPolicy:
    # Default Critic⇄Judge refinement depth (initial + N critiques).
    max_critique_rounds: int = 4
    judge_required: bool = True
    # Reserved consumer policy flag. The Assay planning primitive currently
    # does not execute or route a breaker role from this value.
    breaker_required: bool = False
    # Before Critic/Judge consensus, a finding surviving this many rounds
    # vetoes the plan early. Passing consensus retains below-floor advisories;
    # the weighted Judge backstop remains authoritative before acceptance.
    persistence_veto_threshold: int = 3

    # Weighted-severity Judge veto. After critic + judge both decide
    # ``pass``, sum
    # ``severity_weights[finding.severity]`` over active, unsuppressed judge
    # findings; addressed / dismissed findings contribute zero. If the total
    # is >= ``weighted_threshold``, treat as veto. The
    # weight table is a tuple-of-tuples to keep the dataclass
    # hashable. ``threshold=10.0`` means a single Critical finding
    # (weight 10) trips the veto; 20 Lows (sum 6) do not.
    judge_veto_weighted_threshold: float = 10.0
    judge_veto_severity_weights: tuple[tuple[str, float], ...] = (
        ("Critical", 10.0),
        ("High", 3.0),
        ("Medium", 1.0),
        ("Low", 0.3),
    )

    # Cumulative-inflation guard. Anchor is
    # the slice count of the FIRST DAG produced in this debate loop.
    # On revision rounds, if the current DAG's slice count exceeds
    # ``max(floor, anchor * ratio)``, the plan is treated as having
    # bloated — veto. ``ratio=inf`` (operator override) disables.
    cumulative_inflation_ratio: float = 2.5
    cumulative_inflation_floor: int = 8

    # Optional consumer bounds. Generic Assay plans remain unbounded; workflows
    # with a known safe working set can opt into hard totals.
    # An over-limit revision receives a deterministic compaction request before
    # it can become a terminal veto when ``inflation_compaction_retries`` is
    # positive.  This keeps the guard fail-closed without making a single
    # over-eager revision terminal.
    max_total_slices: int | None = field(
        default=None,
        metadata={"persist_optional": True, "canonical_omit_empty": True},
    )
    max_total_tiers: int | None = field(
        default=None,
        metadata={"persist_optional": True, "canonical_omit_empty": True},
    )
    inflation_compaction_retries: int = field(
        default=0,
        metadata={"persist_optional": True, "canonical_omit_empty": True},
    )

    # Optional empty-scope hygiene.  ``None`` preserves the generic behavior.
    # Consumers may allow a small number of explicit verification-only slices;
    # other empty-scope slices are treated as planner filler and sent back for
    # revision rather than silently entering the executable DAG.
    max_verification_only_empty_scope_slices: int | None = field(
        default=None,
        # ``0`` is a meaningful opt-in (permit no empty-scope slices), so this
        # field cannot use canonical_omit_empty: that would erase 0 and hydrate
        # it as the behaviorally distinct default ``None``.
        metadata={"persist_optional": True},
    )

    # Trusted workflow-specific decomposition guidance appended to the generic
    # planner contract. It is policy rather than task content, allowing a
    # caller to refine decomposition without changing the default contract.
    planner_guidance: str | None = field(
        default=None,
        metadata={"persist_optional": True, "canonical_omit_empty": True},
    )

    # Per-role circuit breakers. An ``inconclusive`` Critic or Judge verdict
    # means that role failed to adjudicate the candidate. The planning primitive stops
    # after this many consecutive failures; any conclusive role decision
    # (pass / revise / reject) resets that role's counter.
    max_critic_failures: int = 3
    max_judge_failures: int = 3

    # Optional convergence softening for consumers that run an independent
    # Judge on every candidate.  Keep this additive field last so legacy
    # positional PlanPolicy construction retains its existing ABI.  After
    # this many prior conclusively judged candidates (``pass`` or ``revise``),
    # active Judge feedback has been sent to the Planner, and the revised
    # candidate earns a Critic pass, one Judge reviews that current candidate
    # without its own prior frontier in the prompt. Inconclusive role failures
    # never unlock this behavior. The complete history remains durable, the
    # current Critic frontier remains visible, and acceptance still requires
    # this fresh Judge decision to pass. This escape hatch is one-shot per
    # debate; a later Judge again receives normal prior closure. ``None``
    # disables it.
    fresh_judge_reassessment_min_prior_rounds: int | None = field(
        default=None,
        metadata={"persist_optional": True, "canonical_omit_empty": True},
    )

    def __post_init__(self) -> None:
        reassessment_rounds = self.fresh_judge_reassessment_min_prior_rounds
        if reassessment_rounds is not None and (
            isinstance(reassessment_rounds, bool)
            or not isinstance(reassessment_rounds, int)
            or reassessment_rounds < 1
        ):
            raise ValueError(
                "PlanPolicy.fresh_judge_reassessment_min_prior_rounds must be "
                "None or an integer >= 1"
            )
        for name in ("max_critic_failures", "max_judge_failures"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"PlanPolicy.{name} must be an integer >= 1")
        for name in ("max_total_slices", "max_total_tiers"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ValueError(
                    f"PlanPolicy.{name} must be None or an integer >= 1"
                )
        retries = self.inflation_compaction_retries
        if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
            raise ValueError(
                "PlanPolicy.inflation_compaction_retries must be an integer >= 0"
            )
        empty_limit = self.max_verification_only_empty_scope_slices
        if empty_limit is not None and (
            isinstance(empty_limit, bool)
            or not isinstance(empty_limit, int)
            or empty_limit < 0
        ):
            raise ValueError(
                "PlanPolicy.max_verification_only_empty_scope_slices must be "
                "None or an integer >= 0"
            )
        if self.planner_guidance is not None and (
            not isinstance(self.planner_guidance, str)
            or not self.planner_guidance.strip()
        ):
            raise ValueError(
                "PlanPolicy.planner_guidance must be None or a non-empty string"
            )


@dataclass(frozen=True)
class BackendPolicy:
    retry_budget: int
    timeout_seconds: float | None
    on_cancellation: Literal["raise", "convert_to_backend_error"]
    on_max_turns: Literal["raise", "continue_with_warning"]

    def __post_init__(self) -> None:
        if (
            isinstance(self.retry_budget, bool)
            or not isinstance(self.retry_budget, int)
            or self.retry_budget < 1
        ):
            raise ValueError(
                "BackendPolicy.retry_budget must be an integer >= 1"
            )
        timeout = self.timeout_seconds
        if timeout is not None and (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError(
                "BackendPolicy.timeout_seconds must be None or a finite "
                "number > 0"
            )
        if self.on_cancellation not in {
            "raise",
            "convert_to_backend_error",
        }:
            raise ValueError(
                "BackendPolicy.on_cancellation must be 'raise' or "
                "'convert_to_backend_error'"
            )
        if self.on_max_turns not in {"raise", "continue_with_warning"}:
            raise ValueError(
                "BackendPolicy.on_max_turns must be 'raise' or "
                "'continue_with_warning'"
            )
