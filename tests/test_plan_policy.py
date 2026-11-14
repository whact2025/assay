# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Tests for ``PlanPolicy`` defaults and bounded-plan veto behavior.

Covers:
- Canonical defaults use max_critique_rounds=4 and
  persistence_veto_threshold=3.
- ``judge_veto_severity_weights`` carries the canonical weight table.
- ``_compute_weighted_severity`` adds weights correctly, handles
  unknown severities, empty findings.
- Weighted-severity veto fires when sum >= threshold even with a
  `pass` decision; it does not fire when sum < threshold.
- Cumulative-inflation veto fires when slice count exceeds
  ``max(floor, anchor*ratio)``; doesn't fire on the anchor round.
- The inflation veto is disabled when ratio=float('inf').
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from assay import Finding, PlanPolicy
from assay.canonical import parse, serialize
from assay.corpus.codec import decode_dataclass


def test_default_policy_values():
    """PlanPolicy exposes one canonical set of bounded-loop defaults."""
    p = PlanPolicy()
    assert p.max_critique_rounds == 4
    assert p.judge_required is True
    assert p.fresh_judge_reassessment_min_prior_rounds is None
    assert p.breaker_required is False
    assert p.persistence_veto_threshold == 3
    assert p.judge_veto_weighted_threshold == 10.0
    assert dict(p.judge_veto_severity_weights) == {
        "Critical": 10.0, "High": 3.0, "Medium": 1.0, "Low": 0.3,
    }
    assert p.cumulative_inflation_ratio == 2.5
    assert p.cumulative_inflation_floor == 8
    assert p.max_total_slices is None
    assert p.max_total_tiers is None
    assert p.inflation_compaction_retries == 0
    assert p.max_verification_only_empty_scope_slices is None
    assert p.planner_guidance is None
    assert p.max_critic_failures == 3
    assert p.max_judge_failures == 3


def test_new_policy_field_preserves_legacy_positional_core_abi():
    policy = PlanPolicy(2, True, False, 7)

    assert policy.max_critique_rounds == 2
    assert policy.judge_required is True
    assert policy.breaker_required is False
    assert policy.persistence_veto_threshold == 7
    assert policy.fresh_judge_reassessment_min_prior_rounds is None


def test_pre_bounds_policy_record_hydrates_with_new_defaults():
    """Records written before bounded-plan fields remain readable."""

    legacy_payload = {
        "max_critique_rounds": 4,
        "judge_required": True,
        "breaker_required": False,
        "persistence_veto_threshold": 3,
        "judge_veto_weighted_threshold": 10.0,
        "judge_veto_severity_weights": [
            ["Critical", 10.0],
            ["High", 3.0],
            ["Medium", 1.0],
            ["Low", 0.3],
        ],
        "cumulative_inflation_ratio": 2.5,
        "cumulative_inflation_floor": 8,
        "max_critic_failures": 3,
        "max_judge_failures": 3,
    }

    restored = decode_dataclass(legacy_payload, PlanPolicy)

    assert restored.max_total_slices is None
    assert restored.max_total_tiers is None
    assert restored.inflation_compaction_retries == 0
    assert restored.max_verification_only_empty_scope_slices is None
    assert restored.planner_guidance is None
    assert restored.fresh_judge_reassessment_min_prior_rounds is None


def test_zero_empty_scope_allowance_survives_canonical_round_trip():
    policy = PlanPolicy(max_verification_only_empty_scope_slices=0)

    restored = decode_dataclass(parse(serialize(policy)), PlanPolicy)

    assert restored.max_verification_only_empty_scope_slices == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_critic_failures", 0),
        ("max_critic_failures", -1),
        ("max_critic_failures", True),
        ("max_judge_failures", 0),
        ("max_judge_failures", 1.5),
        ("max_total_slices", 0),
        ("max_total_tiers", True),
        ("inflation_compaction_retries", -1),
        ("inflation_compaction_retries", True),
        ("max_verification_only_empty_scope_slices", -1),
        ("fresh_judge_reassessment_min_prior_rounds", 0),
        ("fresh_judge_reassessment_min_prior_rounds", True),
    ],
)
def test_role_failure_limits_must_be_positive_integers(field, value):
    with pytest.raises(ValueError, match=field):
        PlanPolicy(**{field: value})


def test_compute_weighted_severity_default_weights():
    from assay.primitives.plan import _compute_weighted_severity
    weights = (("Critical", 10.0), ("High", 3.0), ("Medium", 1.0), ("Low", 0.3))

    # Empty
    assert _compute_weighted_severity((), weights) == 0.0

    # Single Critical → trips default threshold of 10.0
    f = Finding(
        id="x", severity="Critical", category="other",
        evidence=(), suggested_action=None, source="judge",
    )
    assert _compute_weighted_severity((f,), weights) == 10.0

    # 20 Lows = 6.0 (under default threshold)
    lows = tuple(
        Finding(id=f"l{i}", severity="Low", category="other",
                evidence=(), suggested_action=None, source="judge")
        for i in range(20)
    )
    assert _compute_weighted_severity(lows, weights) == pytest.approx(6.0)

    # Unknown severity → weight 1.0
    f_unknown = Finding(
        id="u", severity="Unknown", category="other",
        evidence=(), suggested_action=None, source="judge",
    )
    assert _compute_weighted_severity((f_unknown,), weights) == 1.0

    # Mix: 1 Critical + 5 High + 10 Medium = 10 + 15 + 10 = 35
    mixed = (
        Finding(id="c", severity="Critical", category="other",
                evidence=(), suggested_action=None, source="judge"),
        *(Finding(id=f"h{i}", severity="High", category="other",
                  evidence=(), suggested_action=None, source="judge")
          for i in range(5)),
        *(Finding(id=f"m{i}", severity="Medium", category="other",
                  evidence=(), suggested_action=None, source="judge")
          for i in range(10)),
    )
    assert _compute_weighted_severity(mixed, weights) == pytest.approx(35.0)


def test_slice_count_helper():
    from assay.primitives.plan import _slice_count
    dag = MagicMock()
    tier1 = MagicMock()
    tier1.slices = (MagicMock(), MagicMock(), MagicMock())  # 3
    tier2 = MagicMock()
    tier2.slices = (MagicMock(), MagicMock())  # 2
    dag.tiers = (tier1, tier2)
    assert _slice_count(dag) == 5


def test_policy_yaml_round_trip_with_new_fields(tmp_path):
    """spec.yaml can declare the new optional plan_policy fields."""
    import textwrap
    from assay import Spec
    yaml_text = textwrap.dedent("""
    schema_version: 1
    project_context:
      project: x
      repo_url: ssh://example/x.git
      base_commit: abc
      branch: null
      submodule_path: null
      runners: {}
    intent: test
    constraints: []
    done_criteria: []
    plan_policy:
      max_critique_rounds: 2
      judge_required: true
      fresh_judge_reassessment_min_prior_rounds: 3
      breaker_required: false
      persistence_veto_threshold: 5
      judge_veto_weighted_threshold: 15.0
      cumulative_inflation_ratio: 3.0
      cumulative_inflation_floor: 12
      max_total_slices: 16
      max_total_tiers: 12
      inflation_compaction_retries: 1
      max_verification_only_empty_scope_slices: 1
      planner_guidance: Keep independently verifiable increments.
      max_critic_failures: 2
      max_judge_failures: 4
    """)
    spec = Spec.from_yaml(yaml_text)
    assert spec.plan_policy is not None
    assert spec.plan_policy.max_critique_rounds == 2
    assert spec.plan_policy.fresh_judge_reassessment_min_prior_rounds == 3
    assert spec.plan_policy.persistence_veto_threshold == 5
    assert spec.plan_policy.judge_veto_weighted_threshold == 15.0
    assert spec.plan_policy.cumulative_inflation_ratio == 3.0
    assert spec.plan_policy.cumulative_inflation_floor == 12
    assert spec.plan_policy.max_total_slices == 16
    assert spec.plan_policy.max_total_tiers == 12
    assert spec.plan_policy.inflation_compaction_retries == 1
    assert spec.plan_policy.max_verification_only_empty_scope_slices == 1
    assert spec.plan_policy.planner_guidance == (
        "Keep independently verifiable increments."
    )
    assert spec.plan_policy.max_critic_failures == 2
    assert spec.plan_policy.max_judge_failures == 4


def test_policy_yaml_uses_defaults_for_omitted_optional_fields(tmp_path):
    """Omitted optional policy fields inherit canonical defaults."""
    import textwrap
    from assay import Spec
    yaml_text = textwrap.dedent("""
    schema_version: 1
    project_context:
      project: x
      repo_url: ssh://example/x.git
      base_commit: abc
      branch: null
      submodule_path: null
      runners: {}
    intent: test
    constraints: []
    done_criteria: []
    plan_policy:
      max_critique_rounds: 2
      judge_required: true
      breaker_required: false
      persistence_veto_threshold: 5
    """)
    spec = Spec.from_yaml(yaml_text)
    assert spec.plan_policy is not None
    # New fields inherit defaults.
    assert spec.plan_policy.judge_veto_weighted_threshold == 10.0
    assert spec.plan_policy.cumulative_inflation_ratio == 2.5


def test_policy_yaml_dict_severity_weights():
    """The YAML representation prefers a dict form for readability;
    the parser converts dict → tuple-of-tuples."""
    import textwrap
    from assay import Spec
    yaml_text = textwrap.dedent("""
    schema_version: 1
    project_context:
      project: x
      repo_url: ssh://example/x.git
      base_commit: abc
      branch: null
      submodule_path: null
      runners: {}
    intent: test
    constraints: []
    done_criteria: []
    plan_policy:
      max_critique_rounds: 2
      judge_required: true
      breaker_required: false
      persistence_veto_threshold: 5
      judge_veto_severity_weights:
        Critical: 20.0
        High: 5.0
    """)
    spec = Spec.from_yaml(yaml_text)
    weights = dict(spec.plan_policy.judge_veto_severity_weights)
    assert weights["Critical"] == 20.0
    assert weights["High"] == 5.0
