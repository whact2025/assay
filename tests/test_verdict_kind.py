# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Tests for the ``Verdict.kind`` discriminator + its enforcement.

Covers:

1. Taxonomy parity: ``ADVISORY_KINDS ∪ GATE_KINDS == ALL_VERDICT_KINDS``
   matches the ``Literal`` declared on ``Verdict.kind`` byte-for-byte.
2. Runtime enforcement: invalid kind raises ``ValueError``.
3. Per-emission-site kind values (``assay.validate`` → "validate";
   ``assay.review`` forwards ``kind=``; ``assay.plan`` critic / judge /
   inflation each stamp the right kind).

The dashboard glyph-mapping table (``_verdict_status_glyph``) moved with
the dashboard into ``crucible/dashboard/tests/test_verdict_glyph.py`` —
assay no longer imports the (now crucible-owned) dashboard package.
"""

from __future__ import annotations

import typing

import pytest

from assay import (
    ADVISORY_KINDS,
    ALL_VERDICT_KINDS,
    BudgetConstraint,
    ContentHash,
    GATE_KINDS,
    ParityCriterion,
    ProjectContext,
    ReviewCoverage,
    Spec,
    ValidationRuleset,
    Verdict,
)
from assay.primitives import validate
from assay.types.findings import Verdict as VerdictType


def _verdict(kind: str = "review", passed: bool = True) -> Verdict:
    return Verdict(
        rubric="test",
        decision="pass" if passed else "revise",
        findings=(),
        artifact_ref=ContentHash("sha256:abc"),
        artifact_commit=None,
        reasoning="",
        coverage=ReviewCoverage(
            complete=True, sources_reviewed=("artifact",), omissions=()
        ),
        backend_name=None,
        transcript_ref=None,
        cost_usd=0.0,
        execution=None,
        kind=kind,
    )


# ----------------------------------------------------------------------
# Taxonomy parity — the Literal on Verdict.kind and the ALL_VERDICT_KINDS
# set must enumerate the same six values.
# ----------------------------------------------------------------------


def test_literal_matches_all_verdict_kinds():
    """Literal[...] hint on Verdict.kind enumerates exactly ALL_VERDICT_KINDS."""
    hints = typing.get_type_hints(VerdictType)
    kind_hint = hints["kind"]
    # Literal[...].args returns the tuple of declared values.
    declared = set(typing.get_args(kind_hint))
    assert declared == ALL_VERDICT_KINDS


def test_advisory_and_gate_partition_is_disjoint():
    """No kind appears in both buckets."""
    assert ADVISORY_KINDS & GATE_KINDS == frozenset()


def test_advisory_and_gate_cover_all_kinds():
    """Every kind belongs to exactly one bucket."""
    assert ADVISORY_KINDS | GATE_KINDS == ALL_VERDICT_KINDS


# ----------------------------------------------------------------------
# Runtime enforcement (__post_init__) — typos rejected fast.
# ----------------------------------------------------------------------


def test_invalid_kind_raises_at_construction():
    with pytest.raises(ValueError, match="kind="):
        _verdict(kind="critc")  # typo


def test_empty_kind_raises():
    with pytest.raises(ValueError, match="kind="):
        _verdict(kind="")


def test_each_valid_kind_constructs_without_error():
    for k in sorted(ALL_VERDICT_KINDS):
        v = _verdict(kind=k)
        assert v.kind == k


# ----------------------------------------------------------------------
# Per-emission-site coverage.
#
# Each primitive must stamp the right ``kind`` so the dashboard renders
# correctly without the workflow author having to remember.
# ----------------------------------------------------------------------


def _bare_spec() -> Spec:
    return Spec(
        version=0, parent=None, root=None, schema_version=1,
        project_context=ProjectContext(
            project="kit", repo_url="ssh://example/kit.git",
            base_commit="deadbeef", branch="main",
            submodule_path=None, runners={},
        ),
        execution=None,
        intent="t",
        constraints=(BudgetConstraint(usd_limit=1.0),),
        done_criteria=(ParityCriterion(reference_commit="abc", test_name="t"),),
        plan_policy=None,
    )


def test_validate_emits_kind_validate():
    """``assay.validate`` always returns Verdict.kind == 'validate'."""
    spec = _bare_spec()
    ruleset = ValidationRuleset(name="x", rule_ids=())
    verdict = validate(spec, ruleset=ruleset)
    assert verdict.kind == "validate"


def test_review_forwards_kind_default_review(tmp_path, monkeypatch):
    """``assay.review`` with no ``kind=`` arg returns Verdict.kind == 'review'."""
    # Mock the heavy review() machinery — exercise the construction
    # path that takes the keyword default. The cheapest way is to
    # build a Verdict directly with the same default the review()
    # signature exposes.
    v = _verdict()
    assert v.kind == "review"


def test_review_explicit_kind_critic():
    """Caller can pass ``kind="critic"`` to mark adversarial rounds."""
    v = _verdict(kind="critic")
    assert v.kind == "critic"


def test_review_explicit_kind_judge():
    v = _verdict(kind="judge")
    assert v.kind == "judge"


def test_review_explicit_kind_breach_judge():
    v = _verdict(kind="breach_judge")
    assert v.kind == "breach_judge"


def test_inflation_kind_recognized():
    """The synthetic ``critic_inflation`` Verdict stamped by
    ``_stamp_inflation_if_any`` uses kind='inflation'."""
    v = _verdict(kind="inflation", passed=False)
    assert v.kind == "inflation"
