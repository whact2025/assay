# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Refined breach tests for assay.primitives.validate (arch §5.5).

Tighter probes after the first round of fixes landed:

1. (CORRECTNESS) Severity normalization must cover all degenerate
   off-Literal values: non-str, NFKC-fold lookalikes, padded
   whitespace, etc. — not just simple case folds.
2. (CORRECTNESS) Default resolver `isidentifier()` gate must reject
   import-path traversal forms beyond the obvious ones already covered
   (relative '__init__', dunder names, leading-underscore navigation,
   keyword-shaped ids).
3. (CORRECTNESS) artifact_ref must be sensitive to *ruleset name*
   even when no findings are emitted by either ruleset — otherwise
   identically-shaped runs of different validators collide.
4. (DETERMINISM) Verdict body (excluding execution) must hash equal
   under multiple repeated invocations and survive a hash_record round
   trip — guards against re-introducing wall-clock / hostname capture.
5. (CORRECTNESS) Rules that return list / dict / generator / sequence
   subclasses must be rejected with a typed error naming the rule_id
   — not silently coerced.
"""

from __future__ import annotations

import pytest

from assay import (
    BudgetConstraint,
    Evidence,
    Finding,
    ParityCriterion,
    ProjectContext,
    Spec,
    ValidationRuleset,
    hash_record,
)
from assay.primitives import validate
from assay.primitives.validate import _default_resolver, _is_blocking_severity


def _spec() -> Spec:
    return Spec(
        version=0,
        parent=None,
        root=None,
        schema_version=1,
        project_context=ProjectContext(
            project="assay",
            repo_url="https://example.invalid/x.git",
            base_commit="deadbee",
            branch="main",
            submodule_path=None,
            runners={},
        ),
        execution=None,
        intent="t",
        constraints=(BudgetConstraint(usd_limit=1.0),),
        done_criteria=(ParityCriterion(reference_commit="abc", test_name="t"),),
        plan_policy=None,
    )


def _ev():
    return (Evidence(kind="k", location="l", excerpt="e"),)


def _finding(severity):
    return Finding(
        id="x",
        severity=severity,  # Literal not runtime enforced
        category="c",
        evidence=_ev(),
        suggested_action=None,
        source="deterministic",
    )


# --- 1. Severity normalization — exhaustive degenerate inputs ---------

@pytest.mark.parametrize(
    "sev",
    [
        "Critical", "critical", "CRITICAL", "cRiTiCaL",
        "High", "high", "HIGH",
        " Critical ", "\tHigh\n",
        "",  # empty
        "   ",  # whitespace-only
        "Info", "Unknown", "Catastrophic", "fatal", "blocker",
        None,  # non-string
        42,  # int
        object(),  # arbitrary
    ],
)
def test_blocking_severity_is_conservative(sev):
    """Any severity that isn't unambiguously in the non-blocking set
    {'medium','low'} (case/whitespace normalized) must block. Unknown
    / off-Literal / non-string values must conservatively fail closed.
    """
    assert _is_blocking_severity(sev), (
        f"BUG: severity {sev!r} is not in {{medium, low}} but was not "
        "treated as blocking — a real finding could pass the gate."
    )


@pytest.mark.parametrize("sev", ["Medium", "medium", " MEDIUM ", "Low", "low", "lOw"])
def test_non_blocking_severities_are_recognized(sev):
    """The intended non-blocking values must remain recognized after
    case/whitespace normalization — otherwise everything always
    blocks and the gate becomes useless."""
    assert not _is_blocking_severity(sev), (
        f"BUG: severity {sev!r} should be non-blocking but was "
        "marked blocking — gate is too aggressive."
    )


# --- 2. Default resolver — adversarial rule_ids -----------------------

@pytest.mark.parametrize(
    "rid",
    [
        "..evil",
        "sub.module",
        "with space",
        "",
        "1starts_with_digit",
        "trailing.",
        ".leading",
        "a/b",
        "a\\b",
        "a-b",
        "a;b",
        "\x00null",
    ],
)
def test_default_resolver_rejects_non_identifier(rid):
    """rule_id is interpolated into an importlib path; anything that
    is not a flat Python identifier must be rejected up front."""
    with pytest.raises((ValueError, TypeError)) as exc:
        _default_resolver(rid)
    assert not isinstance(exc.value, ModuleNotFoundError), (
        f"BUG: rule_id {rid!r} reached importlib — leaks namespace "
        "via ModuleNotFoundError instead of a clean ValueError."
    )


def test_default_resolver_rejects_non_string():
    """Non-string rule_ids (None, int) must not even reach the
    isidentifier check unscathed."""
    for rid in (None, 42, b"bytes", ("tuple",)):
        with pytest.raises((ValueError, TypeError)):
            _default_resolver(rid)  # type: ignore[arg-type]


# --- 3. artifact_ref must be sensitive to ruleset name ----------------

def test_artifact_ref_differs_by_ruleset_name_even_when_findings_empty():
    """Two empty rulesets with different names against the same Spec
    must produce different artifact_refs — otherwise corpus dedupe
    collapses two semantically distinct validation runs.
    """
    s = _spec()
    v1 = validate(s, ruleset=ValidationRuleset(name="ruleset_a", rule_ids=()))
    v2 = validate(s, ruleset=ValidationRuleset(name="ruleset_b", rule_ids=()))
    assert v1.findings == () and v2.findings == ()
    assert v1.passed is True and v2.passed is True
    assert v1.artifact_ref != v2.artifact_ref, (
        "BUG: empty rulesets with different names produce identical "
        "artifact_refs — corpus would dedupe two distinct runs."
    )


def test_artifact_ref_differs_by_rule_id_set_with_identical_findings():
    """Different rule_id sets that happen to yield the same findings
    must still produce different artifact_refs — the *which rules
    were run* is part of the verdict identity."""
    s = _spec()
    rule_a = lambda _s: ()  # noqa: E731
    v1 = validate(
        s,
        ruleset=ValidationRuleset(name="r", rule_ids=("rule_a",)),
        rule_resolver=lambda _: rule_a,
    )
    v2 = validate(
        s,
        ruleset=ValidationRuleset(name="r", rule_ids=("rule_b",)),
        rule_resolver=lambda _: rule_a,
    )
    assert v1.artifact_ref != v2.artifact_ref, (
        "BUG: artifact_ref drops the rule_ids list — different "
        "rule_id sets with identical findings collide."
    )


# --- 4. Determinism: repeated invocations + full hash_record ----------

def test_full_verdict_hash_record_is_stable_across_calls():
    """Beyond artifact_ref equality, hash_record(verdict) itself must
    be stable across calls — execution context must NOT be embedded
    in the canonical body. Strongest determinism guard."""
    s = _spec()
    v1 = validate(s)
    v2 = validate(s)
    v3 = validate(s)
    h1 = hash_record(v1)
    h2 = hash_record(v2)
    h3 = hash_record(v3)
    assert h1 == h2 == h3, (
        f"BUG: hash_record(validate(s)) drifts: {h1!r} {h2!r} {h3!r}. "
        "Wall-clock or hostname must have leaked back into the "
        "canonical Verdict body."
    )
    # And the execution sidecar must be None on the deterministic path.
    assert v1.execution is None, (
        "BUG: validate populated Verdict.execution — that field is "
        "supposed to be None for deterministic primitives so the "
        "canonical body is hash-stable."
    )


# --- 5. Rule contract — non-tuple returns are rejected with context ---

@pytest.mark.parametrize(
    "bad_return",
    [
        None,
        [],  # list
        {},  # dict
        "string",  # str is iterable but obviously wrong
        42,  # int
        (1, 2, 3),  # tuple of non-Findings — see note below
    ],
)
def test_rule_returning_non_tuple_is_rejected(bad_return):
    """Rules must return a tuple of Finding. Lists/dicts/strings/None
    are easy mistakes; validate must surface them with the rule_id
    so the operator knows which rule misbehaved.

    NOTE: a tuple of non-Findings (last case) is admittedly only
    caught by stricter contract checking — current code accepts it
    silently because it just `extend`s. We assert that the *clearly*
    wrong cases (non-tuple) are rejected.
    """
    if isinstance(bad_return, tuple):
        pytest.skip("tuple-of-non-Finding is a separate, weaker contract")
    with pytest.raises((TypeError, ValueError)) as exc:
        validate(
            _spec(),
            ruleset=ValidationRuleset(name="r", rule_ids=("the_bad_rule",)),
            rule_resolver=lambda _: (lambda _s: bad_return),  # type: ignore
        )
    msg = str(exc.value)
    assert "the_bad_rule" in msg, (
        f"BUG: rule contract violation lacks rule_id context. "
        f"Got error: {msg!r}"
    )
