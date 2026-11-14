# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Re-exports of every public type for ergonomic ``from assay.types import X``."""

from __future__ import annotations

from .artifacts import Implementation, LifecycleEvent, Measurement
from .bench import BenchConfig
from .cost_bounds import (
    UsageCostBoundSegment, UsageCostBounds, combine_cost_bounds, cost_bounds_cover,
)
from .backend_execution import (
    BACKEND_EXECUTION_TRACE_VERSION,
    BackendExecutionEvent,
    BackendExecutionTrace,
    ExecutionCompleteness,
)
from .constraints import (
    ActivationConstraint,
    AssertionCriterion,
    BlastRadiusConstraint,
    BudgetConstraint,
    Constraint,
    Criterion,
    ParityCriterion,
)
from .invocation import (
    BackendError,
    BudgetExhausted,
    ContentRejected,
    ExternalCancellation,
    IncompleteEvidence,
    Invocation,
    InvocationResult,
    MaxTurnsExceeded,
    Message,
    StaleConversation,
    TransportError,
    Usage,
    UsagePricingEvidence,
    UsagePricingSegment,
)
from .conversation import (
    CONVERSATION_HANDLE_SCHEMA_VERSION,
    ConversationHandle,
    ConversationIdentity,
    ConversationMode,
    ConversationRecoveryDisposition,
    ConversationRecoveryResolution,
    ConversationResumeAction,
    ConversationResumeDecision,
    ConversationResumeScope,
    ConversationState,
    ConversationTransportAction,
    ConversationTransportOutcome,
    ConversationTransportRequest,
    PendingConversationOperation,
    PendingConversationOperationState,
    advance_conversation_handle,
    decide_conversation_resume,
    mark_conversation_state,
    new_conversation_handle,
)
from .dag import (
    ImplementationDirective, MutationRegion, ReadOnlySourceReference, Slice, Tier, TierDAG,
)
from .execution import ExecutionContext
from .evidence import (
    RESULT_EVIDENCE_ADMISSION_MODES,
    ResultEvidenceAdmission,
    admit_result_evidence,
    missing_execution_evidence,
)
from .findings import (
    ADVISORY_KINDS,
    ALL_VERDICT_KINDS,
    FINDING_DISPOSITIONS,
    REVIEW_DECISIONS,
    AttackSpec,
    CriterionOutcome,
    CriterionResult,
    Evidence,
    Finding,
    FindingDisposition,
    GATE_KINDS,
    Proposal,
    ReviewCoverage,
    ReviewDecision,
    ValidationRuleset,
    Verdict,
)
from .identity import ContentHash, ProjectContext
from .policy import BackendPolicy, PlanPolicy
from .rubric import Rubric
from .rule_context import RuleContext
from .smoke import SmokeResult
from .spec import Spec, UnknownConstraintKind
from .outcomes import PlanDecision, PlanOutcome
from .tools import Tool, ToolSet

__all__ = [
    # identity
    "ContentHash",
    "ProjectContext",
    # execution
    "ExecutionContext",
    # conversation continuity
    "CONVERSATION_HANDLE_SCHEMA_VERSION",
    "ConversationHandle",
    "ConversationIdentity",
    "ConversationMode",
    "ConversationRecoveryDisposition",
    "ConversationRecoveryResolution",
    "ConversationResumeAction",
    "ConversationResumeDecision",
    "ConversationResumeScope",
    "ConversationState",
    "ConversationTransportAction",
    "ConversationTransportOutcome",
    "ConversationTransportRequest",
    "PendingConversationOperation",
    "PendingConversationOperationState",
    "advance_conversation_handle",
    "decide_conversation_resume",
    "mark_conversation_state",
    "new_conversation_handle",
    # constraints / criteria
    "Constraint",
    "ActivationConstraint",
    "BudgetConstraint",
    "BlastRadiusConstraint",
    "Criterion",
    "AssertionCriterion",
    "ParityCriterion",
    # policy
    "PlanPolicy",
    "BackendPolicy",
    "BACKEND_EXECUTION_TRACE_VERSION",
    "BackendExecutionEvent",
    "BackendExecutionTrace",
    "ExecutionCompleteness",
    "RESULT_EVIDENCE_ADMISSION_MODES",
    "ResultEvidenceAdmission",
    "admit_result_evidence",
    "missing_execution_evidence",
    # rubric
    "Rubric",
    # smoke
    "SmokeResult",
    # findings / verdicts
    "Evidence",
    "CriterionResult",
    "CriterionOutcome",
    "Proposal",
    "AttackSpec",
    "Finding",
    "Verdict",
    "ReviewCoverage",
    "ReviewDecision",
    "FindingDisposition",
    "REVIEW_DECISIONS",
    "FINDING_DISPOSITIONS",
    "ValidationRuleset",
    "RuleContext",
    "ADVISORY_KINDS",
    "GATE_KINDS",
    "ALL_VERDICT_KINDS",
    # tools
    "Tool",
    "ToolSet",
    # invocation
    "Invocation",
    "InvocationResult",
    "Usage",
    "UsageCostBoundSegment",
    "UsageCostBounds",
    "combine_cost_bounds",
    "cost_bounds_cover",
    "UsagePricingEvidence",
    "UsagePricingSegment",
    "Message",
    "BackendError",
    "TransportError",
    "BudgetExhausted",
    "MaxTurnsExceeded",
    "ContentRejected",
    "ExternalCancellation",
    "IncompleteEvidence",
    "StaleConversation",
    # dag
    "ImplementationDirective",
    "MutationRegion",
    "ReadOnlySourceReference",
    "Slice",
    "Tier",
    "TierDAG",
    # artifacts
    "Implementation",
    "Measurement",
    "LifecycleEvent",
    # bench
    "BenchConfig",
    # spec
    "Spec",
    "UnknownConstraintKind",
    "PlanDecision",
    "PlanOutcome",
]
