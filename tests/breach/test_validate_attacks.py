# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Breach tests for assay.primitives.validate (arch §5.5).

Each test asserts what *should* be true. A failing test = a bug found.

Attack vectors:
  1. (CORRECTNESS) Severity Literal is not runtime-enforced; lowercase
     'critical' / 'high' bypass the passed gate silently. validate's
     gate uses an exact case-sensitive string match.
  2. (DETERMINISM) Same spec → different Verdict each call because
     execution.timestamp is now() and machine is gethostname(). Breaks
     content-addressed dedupe at the Verdict level.
  3. (CORRECTNESS) Empty/whitespace severity also bypasses the gate.
  4. (CORRECTNESS) A rule returning None crashes the call site
     instead of raising a typed contract error.
  5. (ROLLBACK_SAFETY) Default resolver f-strings rule_id straight
     into an importable module path with no identifier validation —
     traversal / arbitrary-module concerns.
  6. (PERFORMANCE_CLIFF / CORRECTNESS) Duplicate rule_ids fire the
     same rule N times, fanning out duplicate findings.
  7. (CORRECTNESS) artifact_ref is computed from spec only; two
     verdicts with opposite verdicts collide.
"""

from __future__ import annotations

import time
import pytest

from assay import (
    BudgetConstraint,
    Evidence,
    Finding,
    ParityCriterion,
    ProjectContext,
    Spec,
    ValidationRuleset,
)
from assay.primitives import validate
from assay.primitives.validate import _default_resolver


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


def _ev() -> tuple[Evidence, ...]:
    return (Evidence(kind="k", location="l", excerpt="e"),)


def _finding(severity: str) -> Finding:
    return Finding(
        id="x",
        severity=severity,  # Literal not runtime-enforced
        category="c",
        evidence=_ev(),
        suggested_action=None,
        source="deterministic",
    )


# --- 1. Severity gate is case-sensitive (CORRECTNESS bug) ------------
def test_lowercase_critical_must_fail_gate():
    """Correctness invariant: if a deterministic rule emits a finding
    whose severity is semantically Critical, the verdict must NOT
    report passed=True. validate currently uses a case-sensitive
    `in ('Critical','High')` check, so lowercase typos silently pass.

    This test EXPECTS passed=False and FAILS against current code,
    surfacing the silent-failure bug.
    """
    v = validate(
        _spec(),
        ruleset=ValidationRuleset(name="r", rule_ids=("any",)),
        rule_resolver=lambda _: (lambda _s: (_finding("critical"),)),
    )
    assert v.findings, "rule emitted a finding"
    assert v.passed is False, (
        "BUG: a 'critical' finding (case mismatch) silently passed "
        "the gate because the check is exact-string. Either the "
        "severity should be normalized/validated, or the Finding "
        "constructor should reject off-Literal severities."
    )


def test_uppercase_high_must_fail_gate():
    """Same root cause: 'HIGH' is rejected by the gate, treated as
    benign. validate must normalize or Finding must validate."""
    v = validate(
        _spec(),
        ruleset=ValidationRuleset(name="r", rule_ids=("any",)),
        rule_resolver=lambda _: (lambda _s: (_finding("HIGH"),)),
    )
    assert v.passed is False, "BUG: 'HIGH' (uppercase) bypasses gate"


def test_empty_severity_must_be_rejected_or_fail_gate():
    """An empty-string severity should either be rejected at Finding
    construction or treated as failing the gate. Currently it's a
    silent pass."""
    v = validate(
        _spec(),
        ruleset=ValidationRuleset(name="r", rule_ids=("any",)),
        rule_resolver=lambda _: (lambda _s: (_finding(""),)),
    )
    assert v.passed is False, "BUG: empty-string severity is silently treated as passing"


# --- 2. Determinism (CORRECTNESS bug) --------------------------------
def test_validate_is_deterministic_for_same_spec():
    """Hashing the resulting Verdict for two calls on the same Spec
    should be stable, so the corpus can dedupe. Currently the Verdict
    embeds execution.timestamp = now() and machine = gethostname(),
    so identical work produces non-equal verdicts.
    """
    from assay import hash_record

    s = _spec()
    v1 = validate(s)
    time.sleep(0.002)
    v2 = validate(s)
    # The content-address of the work product should not depend on
    # wall-clock time — but it does, via execution context.
    assert hash_record(v1) == hash_record(v2), (
        "BUG: validate's Verdict is non-deterministic because "
        "_snapshot_execution_context() captures datetime.now() and "
        "socket.gethostname(). Same Spec produces different Verdict "
        "content-hashes."
    )


# --- 3. Rule contract — None return (CORRECTNESS / robustness) -------
def test_rule_returning_none_raises_clean_typed_error():
    """The rule contract is `Callable[[Spec], tuple[Finding, ...]]`.
    A misbehaving rule returning None should raise a clean
    TypeError/ValueError that identifies the offending rule_id, not
    a raw `extend` TypeError from deep in validate.
    """
    with pytest.raises((TypeError, ValueError)) as exc:
        validate(
            _spec(),
            ruleset=ValidationRuleset(name="r", rule_ids=("bad_rule",)),
            rule_resolver=lambda _: (lambda _s: None),  # type: ignore
        )
    # Helpful contract-error should name the rule that misbehaved.
    assert "bad_rule" in str(exc.value), (
        "BUG: rule contract violation surfaces as raw "
        "`extend(None)` TypeError with no rule_id context — operator "
        "can't tell which rule produced bad output."
    )


# --- 4. Default resolver — no rule_id validation (ROLLBACK_SAFETY) ---
def test_default_resolver_validates_rule_id_is_identifier():
    """rule_id is f-stringed straight into an import path. Anything
    that's not a flat Python identifier (dotted paths, dunders,
    relative components) should be rejected up front, rather than
    leaking an ImportError that exposes the crucible.rules namespace.
    """
    bad_ids = ["..evil", "sub.module", "with space", "", "1starts_with_digit"]
    for rid in bad_ids:
        with pytest.raises((ValueError, TypeError)) as exc:
            _default_resolver(rid)
        assert not isinstance(exc.value, ModuleNotFoundError), (
            f"BUG: rule_id {rid!r} reached importlib without "
            "validation — leaks internal namespace via "
            "ModuleNotFoundError instead of a clean ValueError."
        )


# --- 5. Duplicate rule_ids (PERFORMANCE_CLIFF + CORRECTNESS) ---------
def test_duplicate_rule_ids_are_deduped():
    """A ruleset with duplicate rule_ids is almost certainly a
    config typo. Running the same rule N times wastes work and
    inflates findings, which then propagate downstream
    (persistence_count gets skewed, etc.).
    """
    calls: list[int] = []
    f = _finding("Low")

    def rule(_s):
        calls.append(1)
        return (f,)

    v = validate(
        _spec(),
        ruleset=ValidationRuleset(name="r", rule_ids=("same", "same", "same")),
        rule_resolver=lambda _: rule,
    )
    assert len(calls) == 1, (
        f"BUG: rule fired {len(calls)} times due to duplicate "
        "rule_ids — no dedupe at the validate boundary."
    )
    assert len(v.findings) == 1, (
        "BUG: duplicate rule_ids produced duplicate findings"
    )


# --- 6. Verdict identity (CORRECTNESS) -------------------------------
def test_artifact_ref_distinguishes_different_verdicts():
    """artifact_ref is keyed only on the Spec, not on the
    ruleset/findings. Two verdicts with opposite results (one passed,
    one failed) on the same Spec produce the same artifact_ref. A
    corpus keying verdicts by artifact_ref will lose one of them.
    """
    s = _spec()
    v_empty = validate(s)  # passes
    crit = _finding("Critical")
    v_crit = validate(
        s,
        ruleset=ValidationRuleset(name="r", rule_ids=("c",)),
        rule_resolver=lambda _: (lambda _s: (crit,)),
    )
    assert v_empty.passed is True
    assert v_crit.passed is False
    assert v_empty.artifact_ref != v_crit.artifact_ref, (
        "BUG: opposite verdicts on the same Spec share artifact_ref. "
        "Either Verdict should be content-addressed against "
        "(spec, ruleset, findings), or callers must key on a "
        "different field."
    )


# --- 7. (Removed) Stub regression marker -----------------------------
# Originally asserted that the §5.N primitives raised NotImplementedError
# without dereferencing args. All seven primitives (plan / review /
# implement / bench / parallel / sweep / validate) are now real; the
# stub invariant no longer applies. Deleted together with the last stub
# per the greenfield discipline (no compat shims, no orphan invariants).
