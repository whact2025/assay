# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Diagnostic bounds survive primitive aggregation without becoming exact."""

import json
from dataclasses import replace

import pytest

from assay import (
    BackendExecutionEvent, BackendExecutionTrace, ExecutionCompleteness,
    InvocationResult, PlanPolicy, Usage, UsageCostBounds, UsageCostBoundSegment,
    combine_cost_bounds, parse, serialize,
)
from assay.corpus.codec import decode_dataclass
from assay.corpus.store import Corpus
from assay.primitives.plan import plan
from assay.primitives.review import review

from .test_plan import ScriptedBackend, _DAG_OK, _make_spec, _rubric, _VERDICT_PASS
from .test_review_chunk_protocol import _ObservationBackend
from .test_review import _make_rubric


POLICY = "test-cost-interval-v1"


def bound_usage():
    bound = UsageCostBounds((UsageCostBoundSegment(126, 74, 0, None, 0.0011478, 0.001488, POLICY),), True)
    return Usage(126, 74, bound.lower_bound_usd, cost_bounds=bound)


def bounded(result):
    return replace(result, usage=bound_usage(), pricing_evidence=None,
        completeness=ExecutionCompleteness(terminal_complete=True, trace_complete=True),
        execution_trace=BackendExecutionTrace(
            backend="test", requested_model="test",
            events=(BackendExecutionEvent(0, "terminal", name="end_turn"),),
        ))


class BoundedScriptedBackend(ScriptedBackend):
    def invoke(self, invocation):
        return bounded(super().invoke(invocation))


class BoundedChunkBackend(_ObservationBackend):
    def invoke(self, invocation):
        return bounded(super().invoke(invocation))


@pytest.mark.parametrize("budget,decision,calls", [(0.01, "accepted", 3), (0.0025, "inconclusive", 2)])
def test_plan_budgets_upper_bound_and_persists_unknown_actual_cost(tmp_path, budget, decision, calls):
    planner = BoundedScriptedBackend([json.dumps(_DAG_OK)])
    critic = BoundedScriptedBackend([_VERDICT_PASS])
    judge = BoundedScriptedBackend([_VERDICT_PASS])
    outcome = plan(
        _make_spec(), planner_backend=planner, critic_backend=critic, judge_backend=judge,
        corpus=Corpus(tmp_path), critic_rubric=_rubric("critic"), judge_rubric=_rubric("judge"),
        policy=PlanPolicy(max_critique_rounds=1, judge_required=True), budget_usd=budget,
        diagnostic_cost_policy=POLICY,
    )
    assert outcome.decision == decision
    assert len(planner.invocations) + len(critic.invocations) + len(judge.invocations) == calls
    assert critic.invocations[0].budget_usd == pytest.approx(budget - 0.001488)
    assert outcome.cost_usd == pytest.approx(calls * 0.0011478)
    assert outcome.cost_bounds.upper_bound_usd == pytest.approx(calls * 0.001488)
    assert outcome.cost_bounds.covers(outcome.usage)
    assert not outcome.execution_completeness.cost_complete
    assert outcome.accounting_debt == ("token_usage_complete", "cost_complete")
    assert decode_dataclass(parse(serialize(outcome)), type(outcome)) == outcome


@pytest.mark.parametrize("policy,budget,expected_calls", [(None, 1.0, 1), (POLICY, 1.0, 4), (POLICY, 0.0012, 1)])
def test_chunk_review_requires_opt_in_and_uses_conservative_remainder(tmp_path, policy, budget, expected_calls):
    backend = BoundedChunkBackend("Low")
    verdict = review(
        b"A" * 200001, _make_rubric(), backend=backend, corpus=Corpus(tmp_path),
        chunk_chars=100000, budget_usd=budget, diagnostic_cost_policy=policy,
    )
    assert len(backend.invocations) == expected_calls
    assert verdict.cost_bounds.coverage_complete
    assert verdict.cost_bounds.upper_bound_usd == pytest.approx(expected_calls * 0.001488)
    assert not verdict.execution_completeness.cost_complete
    if expected_calls > 1:
        assert backend.invocations[1].budget_usd == pytest.approx(budget - 0.001488)
    else:
        assert verdict.decision == "inconclusive"


def test_mixed_exact_and_bounded_history_requires_complete_exact_evidence():
    partial = InvocationResult("", (), bound_usage(), "end_turn")
    exact = InvocationResult("", (), Usage(5, 6, 0.1), "end_turn",
        completeness=ExecutionCompleteness(True, True, True, True),
        execution_trace=BackendExecutionTrace(
            backend="test", requested_model="test", events=(BackendExecutionEvent(0, "terminal"),),
        ))
    bound = combine_cost_bounds([partial, exact])
    assert bound.coverage_complete
    assert bound.segments[1].policy == "observed-exact-cost-v1"
    assert bound.upper_bound_usd == pytest.approx(0.101488)
    assert not combine_cost_bounds([partial, replace(exact, completeness=ExecutionCompleteness())]).coverage_complete


@pytest.mark.parametrize("field", ["cost_complete", "token_usage_complete"])
def test_unknown_cache_cannot_claim_complete_accounting(field):
    with pytest.raises(ValueError, match="unknown cache reads"):
        InvocationResult("", (), bound_usage(), "end_turn", completeness=ExecutionCompleteness(**{field: True}))


@pytest.mark.parametrize("mode", ["no-opt-in", "wrong-policy", "uncovered-planner", "uncovered-critic"])
def test_plan_never_crosses_uncovered_diagnostic_accounting(tmp_path, mode):
    planner_type = ScriptedBackend if mode == "uncovered-planner" else BoundedScriptedBackend
    critic_type = ScriptedBackend if mode == "uncovered-critic" else BoundedScriptedBackend
    planner = planner_type([json.dumps(_DAG_OK)])
    critic = critic_type([_VERDICT_PASS])
    judge = BoundedScriptedBackend([_VERDICT_PASS])
    selected_policy = None if mode == "no-opt-in" else "wrong-policy" if mode == "wrong-policy" else POLICY
    outcome = plan(
        _make_spec(), planner_backend=planner, critic_backend=critic, judge_backend=judge,
        corpus=Corpus(tmp_path), critic_rubric=_rubric("critic"), judge_rubric=_rubric("judge"),
        policy=PlanPolicy(max_critique_rounds=1, judge_required=True), budget_usd=0.01,
        diagnostic_cost_policy=selected_policy,
    )
    assert outcome.status == "accounting_incomplete"
    assert outcome.decision == "inconclusive"
    assert len(planner.invocations) == 1
    assert len(critic.invocations) == (1 if mode == "uncovered-critic" else 0)
    assert judge.invocations == []
