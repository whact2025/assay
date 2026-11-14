# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Negative boundaries for exact requirement-quote correction hints.

These tests use only deterministic validation and scripted backends. A useful
hint must never accept an invalid citation, infer wording, enlarge the bounded
repair allowance, or authorize changes to the rest of the candidate.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path

import pytest

from assay import AssertionCriterion, BlastRadiusConstraint, PlanPolicy, Verdict
from assay.corpus.store import Corpus
from assay.primitives.plan import (
    _parse_tier_dag,
    _requirement_ref_source_findings,
)
from tests.test_plan import (
    _DAG_OK,
    _VERDICT_PASS,
    _make_spec,
    _plan_with_scripts,
)


_CANONICAL = "Persist `cache` safely."
_OMITTED = "Persist cache safely."


def _candidate(requirement_ref: str) -> dict:
    payload = copy.deepcopy(_DAG_OK)
    payload["tiers"][0]["slices"][0]["directive"]["requirement_refs"] = [
        requirement_ref,
    ]
    return payload


def _finding(spec, requirement_ref: str):
    dag = _parse_tier_dag(_candidate(requirement_ref))
    assert dag is not None
    findings = _requirement_ref_source_findings(
        dag, root_spec=spec, round_index=0,
    )
    assert len(findings) == 1
    finding = findings[0]
    assert finding.severity == "High"
    assert finding.id == "non_root_requirement_ref_round_0_0_0_0"
    assert finding.evidence[0].kind == "non_root_requirement_ref"
    return finding


def _assert_no_hint(spec, requirement_ref: str) -> None:
    finding = _finding(spec, requirement_ref)
    assert "canonical_replacement" not in finding.evidence[0].excerpt
    assert "exact_citable_text" not in (finding.suggested_action or "")


@pytest.mark.parametrize(
    "intent",
    [
        f"{_CANONICAL} {_CANONICAL}",
        "Persist `cache` safely. Persist cache `safely`.",
    ],
)
def test_quote_hint_rejects_multiple_source_spans_in_same_intent(intent: str):
    _assert_no_hint(_make_spec(intent), _OMITTED)


@pytest.mark.parametrize("other_source", ["constraint", "criterion"])
def test_quote_hint_rejects_matching_spans_in_different_root_sources(
    other_source: str,
):
    spec = _make_spec(_CANONICAL)
    if other_source == "constraint":
        spec = dataclasses.replace(
            spec,
            constraints=(BlastRadiusConstraint(no_edits_outside=(_CANONICAL,)),),
        )
    else:
        spec = dataclasses.replace(
            spec,
            done_criteria=(AssertionCriterion(test_name=_CANONICAL, must_pass=True),),
        )
    _assert_no_hint(spec, _OMITTED)


@pytest.mark.parametrize(
    ("intent", "requirement_ref"),
    [
        (_CANONICAL, "persist cache safely."),
        (_CANONICAL, "Persist cache safely!"),
        (_CANONICAL, "Store cache safely."),
        (_CANONICAL, "Persist `cache safely."),
        (_CANONICAL, "Persist cache` safely."),
        (_CANONICAL, "`Persist cache safely.`"),
        (_CANONICAL, "Persist ``cache`` safely."),
        (_OMITTED, _CANONICAL),
        ("Persist `cache safely.", _OMITTED),
        ("Persist cache` safely.", _OMITTED),
        ("Persist ``cache`` safely.", _OMITTED),
        ("Persist ```cache``` safely.", _OMITTED),
        (r"Persist \`cache\` safely.", _OMITTED),
        ("Keep ``escaped`` and `cache` intact.", "Keep ``escaped`` and cache intact."),
        ("`foobar` and `baz` and `qux`", "bar and baz and qu"),
        ("`foobar` and `baz`", "bar and baz"),
        ("`foobar` and `baz`", "foobar and ba"),
        (r"\`outside Persist `cache` safely.", _OMITTED),
        ("``outside`` Persist `cache` safely.", _OMITTED),
    ],
)
def test_quote_hint_does_not_infer_or_reinterpret_delimiters(
    intent: str, requirement_ref: str,
):
    _assert_no_hint(_make_spec(intent), requirement_ref)


@pytest.mark.parametrize(
    "requirement_ref",
    [_CANONICAL, "Persist\n\t`cache` safely.", "`cache`", "cache"],
)
def test_quote_hint_does_not_change_existing_verbatim_acceptance(
    requirement_ref: str,
):
    dag = _parse_tier_dag(_candidate(requirement_ref))
    assert dag is not None
    original = dag.tiers[0].slices[0].directive.requirement_refs
    assert _requirement_ref_source_findings(
        dag, root_spec=_make_spec(_CANONICAL), round_index=0,
    ) == ()
    assert dag.tiers[0].slices[0].directive.requirement_refs == original


def _run_correction(tmp_path: Path, responses: list[dict], *, allowance: int = 1):
    return _plan_with_scripts(
        _make_spec(_CANONICAL),
        Corpus(tmp_path),
        planner=[json.dumps(response) for response in responses],
        critic=[_VERDICT_PASS],
        judge=[_VERDICT_PASS],
        policy=PlanPolicy(
            max_critique_rounds=0,
            judge_required=True,
            breaker_required=False,
            persistence_veto_threshold=2,
        ),
        candidate_validator=lambda _candidate: (),
        max_candidate_validation_repairs=allowance,
    )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("intent", "unrelated semantic rewrite"),
        ("files_in_scope", ["src/unrelated.py"]),
        ("creates_paths", ["src/extra.py"]),
        ("blast_radius", {"max_files": 2, "languages": []}),
    ],
)
def test_quote_correction_cannot_change_unrelated_directive_fields(
    tmp_path: Path, field: str, replacement,
):
    repaired = _candidate(_CANONICAL)
    repaired["tiers"][0]["slices"][0]["directive"][field] = replacement
    outcome, planner, critic, judge = _run_correction(
        tmp_path, [_candidate(_OMITTED), repaired],
    )

    assert outcome.status == "planner_contract_failure"
    assert outcome.decision == "rejected"
    assert outcome.spec.tier_dag is None
    assert outcome.candidate_validation_repairs_used == 1
    assert len(planner.invocations) == 2
    assert not critic.invocations
    assert judge is not None and not judge.invocations
    corpus = Corpus(tmp_path)
    findings = [
        finding
        for ref in outcome.spec.verdict_refs
        for finding in corpus.get_record(ref, as_type=Verdict).findings
    ]
    assert any(f.id == "candidate_repair_unrelated_delta" for f in findings)


def test_quote_hint_does_not_autoaccept_or_add_an_extra_correction(tmp_path: Path):
    outcome, planner, critic, judge = _run_correction(
        tmp_path,
        [_candidate(_OMITTED), _candidate(_OMITTED), _candidate(_CANONICAL)],
    )

    assert outcome.status == "planner_contract_failure"
    assert outcome.decision == "rejected"
    assert outcome.spec.tier_dag is None
    assert outcome.candidate_validation_repairs_used == 1
    assert len(planner.invocations) == 2
    assert not critic.invocations
    assert judge is not None and not judge.invocations


def test_quote_hint_does_not_grant_a_disabled_correction(tmp_path: Path):
    outcome, planner, critic, judge = _run_correction(
        tmp_path,
        [_candidate(_OMITTED), _candidate(_CANONICAL)],
        allowance=0,
    )

    assert not outcome.accepted
    assert outcome.spec.tier_dag is None
    assert outcome.candidate_validation_repairs_used == 0
    assert len(planner.invocations) == 1
    assert not critic.invocations
    assert judge is not None and not judge.invocations
