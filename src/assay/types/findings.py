# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Findings, evidence, and verdicts for assay.

Both deterministic validation (``assay.primitives.validate``) and adversarial
review (``assay.primitives.review``) return a ``Verdict`` carrying zero or
more ``Finding`` instances. The ``Finding.source`` field discriminates
deterministic findings from model-produced review findings.

A ``ValidationRuleset`` names a collection of rule_ids that
``assay.primitives.validate`` runs against a Spec.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .backend_execution import ExecutionCompleteness
from .execution import ExecutionContext
from .frozen import freeze_dict
from .identity import ContentHash
from .invocation import Usage, UsagePricingEvidence
from .cost_bounds import UsageCostBounds


# Verdict.kind taxonomy.
#
# ``ADVISORY_KINDS`` describe bounded loop-step decisions: a non-pass result
# requests revision or escalation, while a later iteration may still converge.
# ``GATE_KINDS`` describe terminal gates: every decision other than ``pass``
# blocks approval. The union must equal ``ALL_VERDICT_KINDS`` and match the
# ``Verdict.kind`` ``Literal`` below; tests enforce parity.
#
# Downstream consumers import these constants rather than redeclaring the
# role taxonomy.
ADVISORY_KINDS: frozenset[str] = frozenset({"critic", "judge", "inflation"})
GATE_KINDS: frozenset[str] = frozenset({"validate", "breach_judge", "review"})
ALL_VERDICT_KINDS: frozenset[str] = ADVISORY_KINDS | GATE_KINDS

ReviewDecision = Literal["pass", "revise", "reject", "inconclusive"]
FindingDisposition = Literal[
    "confirmed", "suspected", "dismissed", "addressed", "unfixable"
]
CriterionOutcome = Literal[
    "satisfied", "violated", "inconclusive", "not_applicable"
]
REVIEW_DECISIONS: frozenset[str] = frozenset(
    {"pass", "revise", "reject", "inconclusive"}
)
FINDING_DISPOSITIONS: frozenset[str] = frozenset(
    {"confirmed", "suspected", "dismissed", "addressed", "unfixable"}
)


@dataclass(frozen=True)
class Evidence:
    kind: str
    location: str
    excerpt: str
    # Stable identity of the material the excerpt came from. Model-produced
    # review_v2 evidence must set this explicitly (``artifact``,
    # ``context:<key>``, ``prior:<finding-id>``, or ``tool:<call-id>``).
    # The default keeps deterministic rules concise; review_v2 never relies
    # on it because the response validator requires the JSON field.
    source_id: str = "artifact"


@dataclass(frozen=True)
class Proposal:
    """A generated remediation, kept separate from source evidence."""

    action: str
    replacement: str | None = None


@dataclass(frozen=True)
class AttackSpec:
    """Declarative adversarial experiment; intentionally not executable code."""

    setup: str
    stimulus: str
    oracle: str


@dataclass(frozen=True)
class ReviewCoverage:
    """What a reviewer actually observed.

    Character counts are populated by ``assay.primitives.review`` rather than
    trusted from model output. They make prompt elision explicit in the
    persisted verdict and prevent an incomplete review from silently passing.
    """

    complete: bool
    sources_reviewed: tuple[str, ...]
    omissions: tuple[str, ...]
    artifact_chars_total: int = 0
    artifact_chars_reviewed: int = 0


@dataclass(frozen=True)
class CriterionResult:
    """A complete, evidence-backed result for one rubric criterion."""

    criterion_id: str
    outcome: CriterionOutcome
    reasoning: str
    evidence: tuple[Evidence, ...]
    headline: bool = False


@dataclass(frozen=True)
class Finding:
    id: str
    severity: Literal["Critical", "High", "Medium", "Low"]
    category: str
    evidence: tuple[Evidence, ...]
    suggested_action: str | None
    source: Literal["deterministic", "llm"]
    persistence_count: int = 1
    suppressed: bool = False
    # review_v2 fields. Defaults are for deterministic rules created inside
    # Assay; model responses are validated strictly and must provide every
    # field explicitly.
    criterion_id: str = "deterministic"
    confidence: float = 1.0
    disposition: FindingDisposition = "confirmed"
    targets: tuple[str, ...] = ()
    proposal: Proposal | None = None
    attack_spec: AttackSpec | None = None


@dataclass(frozen=True)
class Verdict:
    rubric: str
    decision: ReviewDecision
    findings: tuple[Finding, ...]
    artifact_ref: ContentHash
    artifact_commit: str | None
    reasoning: str
    coverage: ReviewCoverage
    backend_name: str | None
    transcript_ref: ContentHash | None
    cost_usd: float
    # Optional execution provenance for consumer-created Verdicts. Assay's
    # validate and review primitives currently set this to None; run-time
    # context is recorded separately so equivalent verdict content can retain
    # stable identity.
    execution: ExecutionContext | None
    # Exact model usage for this review operation. Canonical zero values are
    # omitted and the strict decoder supplies the declared optional defaults.
    input_tokens: int = field(
        default=0,
        metadata={"persist_optional": True, "canonical_omit_empty": True},
    )
    output_tokens: int = field(
        default=0,
        metadata={"persist_optional": True, "canonical_omit_empty": True},
    )
    # Output-side artifacts captured during the review turn — distinct
    # from ``artifact_ref`` (the rubric's INPUT). Keys are stable kinds
    # ("stderr", "build_warnings", "tool_output", "log_excerpt"); values
    # are corpus refs persisted via ``corpus.put_artifact``. Empty by
    # default so callers that don't populate it stay terse.
    # Convention: callers never mutate the dict in place — construct a
    # new Verdict via ``dataclasses.replace`` when appending artifacts.
    artifacts: dict[str, ContentHash] = field(default_factory=dict)
    # Semantic discriminator for the verdict's role in a larger workflow.
    # It lets consumers distinguish decisive gates from advisory rounds
    # without collapsing the four-way ``decision`` into a boolean:
    #
    # - ``"validate"`` — deterministic validation gate.
    #   Any non-pass decision blocks.
    # - ``"critic"`` — adversarial critic round inside the planning primitive or
    #   a workflow review step. ``revise`` requests another bounded
    #   iteration; ``inconclusive`` requires retry/escalation policy.
    # - ``"judge"`` — judge decision inside planning or review.
    #   The outer protocol decides whether another iteration is possible.
    # - ``"inflation"`` — deterministic critic_inflation check inside
    #   planning. Always advisory; it flags critic discipline rather
    #   than directly approving or rejecting the candidate.
    # - ``"breach_judge"`` — breach-workflow judge.
    #   A decisive accept/reject gate.
    # - ``"review"`` — generic review-rubric verdict not
    #   assigned a more specific role. This is the canonical default.
    kind: Literal[
        "validate", "critic", "judge", "inflation", "breach_judge", "review"
    ] = "review"
    # Provenance for reproducible model review. These are populated by
    # the review primitive and remain optional for deterministic validators.
    rubric_ref: ContentHash | None = None
    rendered_prompt_ref: ContentHash | None = None
    model_name: str | None = None
    model_effort: str | None = None
    capability_profile: str | None = None
    criterion_results: tuple[CriterionResult, ...] = ()
    # Aggregate provider evidence for every physical attempt that contributed
    # to this model-backed verdict. ``None`` means that no backend invocation
    # produced the verdict (for example deterministic validation), not that a
    # model call was complete. Review always copies the InvocationResult value.
    #
    # A successful resumed response can therefore remain semantically useful
    # while false token/cost fields explicitly preserve accounting debt from
    # an earlier incomplete physical attempt.
    execution_completeness: ExecutionCompleteness | None = field(
        default=None,
        metadata={
            "persist_optional": True,
            "canonical_omit_empty": True,
        },
    )
    # Ordered physical-response pricing inputs for every model call that
    # contributed to this Verdict. Review concatenates segments in call order.
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
    # fields from an exact zero reported by a new model-backed review. This is
    # necessary because both states used to have the same omitted wire shape.
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
    def passed(self) -> bool:
        """Readability projection of ``decision``; models never control it."""

        return self.decision == "pass"

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
            # A legacy Verdict can assert complete inclusive input/output
            # counts without carrying the cache subsets added later. Only a
            # complete legacy ledger can recover those exact subsets. Expose
            # the otherwise hidden debt to direct consumers just as planning
            # does when it aggregates the Verdict.
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
            raise TypeError("Verdict.cost_bounds must be UsageCostBounds or None")
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
                "Verdict.execution_completeness must be an "
                "ExecutionCompleteness or None"
            )
        if self.pricing_evidence is not None and (
            type(self.pricing_evidence) is not UsagePricingEvidence
        ):
            raise TypeError(
                "Verdict.pricing_evidence must be a UsagePricingEvidence "
                "or None"
            )
        cache_values = (
            self.cache_creation_tokens,
            self.cache_read_tokens,
        )
        if (cache_values[0] is None) != (cache_values[1] is None):
            raise ValueError(
                "Verdict cache-token aggregates must either both be present "
                "or both be absent"
            )
        if cache_values[0] is not None:
            for field_name, value in zip(
                ("cache_creation_tokens", "cache_read_tokens"),
                cache_values,
            ):
                if type(value) is not int or value < 0:
                    raise ValueError(
                        f"Verdict.{field_name} must be an exact non-negative "
                        "integer when present"
                    )

        completeness = self.execution_completeness
        if completeness is not None and completeness.token_usage_complete:
            for field_name in ("input_tokens", "output_tokens"):
                value = getattr(self, field_name)
                if type(value) is not int or value < 0:
                    raise ValueError(
                        f"Verdict.{field_name} must be an exact non-negative "
                        "integer when token usage is complete"
                    )
            if cache_values[0] is not None and (
                cache_values[0] + cache_values[1] > self.input_tokens
            ):
                raise ValueError(
                    "Verdict cache-token subsets must not exceed "
                    "input_tokens when token usage is complete"
                )
        if self.pricing_evidence is not None and self.pricing_evidence.complete:
            if completeness is None or not (
                completeness.token_usage_complete
                and completeness.cost_complete
            ):
                raise ValueError(
                    "complete Verdict.pricing_evidence requires complete "
                    "token usage and cost evidence"
                )
            # Legacy Verdicts have no top-level cache fields. Complete pricing
            # evidence was their sole exact cache authority, so use those two
            # counters only for reconciliation without mutating or rehashing
            # the stored record. New records carry an explicit pair (including
            # exact zeros), which must agree with complete evidence.
            evidence_usage = self.pricing_evidence.usage
            cache_creation_tokens = (
                evidence_usage.cache_creation_tokens
                if cache_values[0] is None
                else cache_values[0]
            )
            cache_read_tokens = (
                evidence_usage.cache_read_tokens
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
                    "complete Verdict.pricing_evidence must reconcile "
                    "aggregate usage"
                )
        # ``Literal[...]`` is a typing hint, not a runtime check. A typo
        # in a caller (``kind="critc"``) would silently slip through and
        # consumers could misclassify the verdict role. Enforce membership
        # in the canonical set so configuration mistakes surface fast.
        if self.kind not in ALL_VERDICT_KINDS:
            raise ValueError(
                f"Verdict.kind={self.kind!r} is not one of "
                f"{sorted(ALL_VERDICT_KINDS)}; see "
                f"assay/types/findings.py for the canonical taxonomy."
            )
        if self.decision not in REVIEW_DECISIONS:
            raise ValueError(
                f"Verdict.decision={self.decision!r} is not one of "
                f"{sorted(REVIEW_DECISIONS)}"
            )


@dataclass(frozen=True)
class ValidationRuleset:
    name: str
    rule_ids: tuple[str, ...]
