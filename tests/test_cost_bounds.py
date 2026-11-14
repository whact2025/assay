# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Conservative budget evidence never asserts exact observed billing."""

from dataclasses import replace

import pytest

from assay import (
    BackendExecutionEvent, BackendExecutionTrace,
    ExecutionCompleteness, InvocationResult, Usage, UsageCostBoundSegment,
    UsageCostBounds, combine_cost_bounds, cost_bounds_cover, parse, serialize,
)


def segment(**changes):
    return replace(UsageCostBoundSegment(126, 74, 0, None, 0.0011478, 0.001488, "test-policy"), **changes)


def usage():
    bounds = UsageCostBounds((segment(),), True)
    return Usage(126, 74, bounds.lower_bound_usd, cost_bounds=bounds)


def test_bound_preserves_unknown_cache_and_separate_budget_debit():
    value = usage()
    assert value.cost_bounds.covers(value)
    assert value.cost_usd == 0.0011478
    assert value.budget_cost_usd == 0.001488
    assert value.cost_bounds.segments[0].cache_read_tokens is None
    assert parse(serialize(value))["cost_bounds"]["segments"][0]["cache_read_tokens"] is None
    assert UsageCostBounds.from_record(value.cost_bounds.to_record()) == value.cost_bounds


def test_legacy_canonical_usage_unchanged():
    assert serialize(Usage()) == b'{"cost_usd":0.0,"input_tokens":0,"output_tokens":0}'


@pytest.mark.parametrize("field,value", [
    ("input_tokens", True), ("output_tokens", 1.0), ("cache_creation_tokens", -1),
    ("cache_read_tokens", "0"), ("cache_read_tokens", True),
    ("cache_read_tokens", 127), ("cache_creation_tokens", 127),
    ("lower_bound_usd", float("nan")), ("upper_bound_usd", float("inf")),
    ("upper_bound_usd", -1), ("upper_bound_usd", True),
    ("lower_bound_usd", 1.0), ("policy", ""),
])
def test_invalid_segments_rejected(field, value):
    with pytest.raises((TypeError, ValueError)):
        segment(**{field: value})


@pytest.mark.parametrize("changes", [
    {"input_tokens": 125}, {"output_tokens": 73}, {"cost_usd": 0.0},
    {"cache_read_tokens": 1}, {"cache_creation_tokens": 1},
])
def test_bounds_must_cover_every_aggregate_dimension(changes):
    value = usage()
    assert not value.cost_bounds.covers(replace(value, **changes))


def test_mixed_or_incomplete_history_does_not_gain_coverage():
    value = usage()
    assert combine_cost_bounds([value, value]).upper_bound_usd == pytest.approx(0.002976)
    assert not combine_cost_bounds([value, Usage()]).coverage_complete
    incomplete = replace(value, cost_bounds=replace(value.cost_bounds, coverage_complete=False))
    assert not combine_cost_bounds([incomplete]).coverage_complete
    assert incomplete.budget_cost_usd == incomplete.cost_usd


@pytest.mark.parametrize("policy,terminal,trace,accepted", [
    (None, True, True, False), ("other", True, True, False),
    ("test-policy", False, True, False), ("test-policy", True, False, False),
    ("test-policy", True, True, True),
])
def test_primitive_requires_explicit_policy_and_lifecycle(policy, terminal, trace, accepted):
    result = InvocationResult("answer", (), usage(), "end_turn", completeness=ExecutionCompleteness(
        terminal_complete=terminal, trace_complete=trace,
    ), execution_trace=BackendExecutionTrace(
        backend="test", requested_model="test",
        events=(BackendExecutionEvent(0, "terminal", name="end_turn"),),
    ))
    assert cost_bounds_cover(result, policy=policy) is accepted
    assert not result.completeness.complete
    assert not result.completeness.cost_complete


@pytest.mark.parametrize("mutation", ["extra", "missing", "tuple", "segment_extra", "flag"])
def test_wire_record_is_closed(mutation):
    record = usage().cost_bounds.to_record()
    if mutation == "extra":
        record["extra"] = True
    elif mutation == "missing":
        del record["coverage_complete"]
    elif mutation == "tuple":
        record["segments"] = tuple(record["segments"])
    elif mutation == "segment_extra":
        record["segments"][0]["extra"] = True
    else:
        record["coverage_complete"] = 1
    with pytest.raises((TypeError, ValueError)):
        UsageCostBounds.from_record(record)
