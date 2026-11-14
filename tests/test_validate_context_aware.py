# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Tests for context-aware rule dispatch in ``assay.primitives.validate``.

Two arities are supported:
- ``rule(spec)``       — pure (default; runs always)
- ``rule(spec, ctx)``  — context-aware (requires ``ctx`` and receives the
                         supplied ``RuleContext``)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from assay import (
    BudgetConstraint,
    Finding,
    ParityCriterion,
    ProjectContext,
    RuleContext,
    Spec,
    ValidationRuleset,
)
from assay.primitives import RuleContextRequiredError, validate


def _bare_spec() -> Spec:
    return Spec(
        version=0,
        parent=None,
        root=None,
        schema_version=1,
        project_context=ProjectContext(
            project="kit",
            repo_url="ssh://example/kit.git",
            base_commit="deadbeef",
            branch="main",
            submodule_path="kit",
            runners={},
        ),
        execution=None,
        intent="placeholder",
        constraints=(BudgetConstraint(usd_limit=10.0),),
        done_criteria=(ParityCriterion(reference_commit="abc", test_name="t"),),
        plan_policy=None,
    )


# ----------------------------------------------------------------------
# Pure (arity-1) rule behavior — unchanged
# ----------------------------------------------------------------------


def test_validate_verdict_kind_is_validate():
    """validate() must emit Verdict.kind == 'validate' so the dashboard
    can render its pass/fail as a decisive gate (vs critic/judge loop
    verdicts which are advisory)."""
    spec = _bare_spec()
    ruleset = ValidationRuleset(name="x", rule_ids=())
    verdict = validate(spec, ruleset=ruleset)
    assert verdict.kind == "validate"


def _pure_rule_always_clean(spec):
    return ()


def _pure_rule_one_finding(spec):
    return (
        Finding(
            id="pure_finding",
            category="other",
            severity="Low",
            source="deterministic",
            evidence=(),
            suggested_action="ignore",
        ),
    )


def test_pure_rule_runs_without_ctx():
    spec = _bare_spec()
    ruleset = ValidationRuleset(name="pure", rule_ids=("pure_rule_always_clean",))
    resolver = lambda rid: {"pure_rule_always_clean": _pure_rule_always_clean}[rid]  # noqa: E731
    verdict = validate(spec, ruleset=ruleset, rule_resolver=resolver)
    assert verdict.passed
    assert verdict.findings == ()


def test_pure_rule_runs_with_ctx_ignored():
    """Passing ctx to a pure rule is harmless — rule still gets only spec."""
    spec = _bare_spec()
    ruleset = ValidationRuleset(name="pure", rule_ids=("pure_rule_one_finding",))
    resolver = lambda rid: {"pure_rule_one_finding": _pure_rule_one_finding}[rid]  # noqa: E731
    ctx = RuleContext(workspace_root=Path("/tmp"))
    verdict = validate(spec, ruleset=ruleset, rule_resolver=resolver, ctx=ctx)
    assert len(verdict.findings) == 1
    assert verdict.findings[0].id == "pure_finding"


# ----------------------------------------------------------------------
# Context-aware (arity-2) rule behavior
# ----------------------------------------------------------------------


def _ctx_rule_uses_workspace(spec, ctx):
    """Emits a finding referencing the workspace_root it received."""
    return (
        Finding(
            id="ctx_rule_finding",
            category="other",
            severity="Low",
            source="deterministic",
            evidence=(),
            suggested_action=f"workspace_root={ctx.workspace_root}",
        ),
    )


def test_ctx_rule_requires_ctx():
    """Omitting context is a typed caller error, never an implicit pass."""
    spec = _bare_spec()
    ruleset = ValidationRuleset(name="ctx", rule_ids=("ctx_rule_uses_workspace",))
    resolver = lambda rid: {"ctx_rule_uses_workspace": _ctx_rule_uses_workspace}[rid]  # noqa: E731
    with pytest.raises(
        RuleContextRequiredError,
        match=r"ruleset 'ctx' requires ctx.*'ctx_rule_uses_workspace'",
    ):
        validate(spec, ruleset=ruleset, rule_resolver=resolver)


def test_ctx_rule_runs_when_ctx_passed():
    """Context-aware rules receive the passed RuleContext."""
    spec = _bare_spec()
    ruleset = ValidationRuleset(name="ctx", rule_ids=("ctx_rule_uses_workspace",))
    resolver = lambda rid: {"ctx_rule_uses_workspace": _ctx_rule_uses_workspace}[rid]  # noqa: E731
    ctx = RuleContext(workspace_root=Path("/my/workspace"))
    verdict = validate(spec, ruleset=ruleset, rule_resolver=resolver, ctx=ctx)
    assert len(verdict.findings) == 1
    # The rule renders ctx.workspace_root, a Path — compare against its
    # OS-native rendering so the test is portable (it previously hardcoded
    # the POSIX form and failed on Windows; the rule/validate logic is fine).
    assert verdict.findings[0].suggested_action == f"workspace_root={Path('/my/workspace')}"


def test_mixed_arity_ruleset():
    """A ruleset can mix pure and context-aware rules; both fire when ctx is set."""
    spec = _bare_spec()
    ruleset = ValidationRuleset(
        name="mixed",
        rule_ids=("pure_rule_one_finding", "ctx_rule_uses_workspace"),
    )
    resolver = lambda rid: {  # noqa: E731
        "pure_rule_one_finding": _pure_rule_one_finding,
        "ctx_rule_uses_workspace": _ctx_rule_uses_workspace,
    }[rid]
    ctx = RuleContext(workspace_root=Path("/tmp"))
    verdict = validate(spec, ruleset=ruleset, rule_resolver=resolver, ctx=ctx)
    finding_ids = {f.id for f in verdict.findings}
    assert finding_ids == {"pure_finding", "ctx_rule_finding"}


def test_missing_ctx_preflight_runs_no_rules():
    """Ruleset wiring is checked atomically before even pure rules execute."""
    spec = _bare_spec()
    calls: list[str] = []

    def pure_rule(spec):
        calls.append("pure")
        return ()

    def context_rule(spec, ctx):
        calls.append("context")
        return ()

    ruleset = ValidationRuleset(
        name="mixed",
        rule_ids=("pure_rule", "context_rule"),
    )
    resolver = lambda rid: {  # noqa: E731
        "pure_rule": pure_rule,
        "context_rule": context_rule,
    }[rid]
    with pytest.raises(RuleContextRequiredError, match="context_rule"):
        validate(spec, ruleset=ruleset, rule_resolver=resolver)
    assert calls == []


# ----------------------------------------------------------------------
# Misshaped rule signatures — surface a clear error
# ----------------------------------------------------------------------


def _bad_rule_zero_args():
    return ()


def _bad_rule_three_args(spec, ctx, extra):
    return ()


def test_rule_arity_zero_raises():
    spec = _bare_spec()
    ruleset = ValidationRuleset(name="bad", rule_ids=("bad_rule_zero_args",))
    resolver = lambda rid: {"bad_rule_zero_args": _bad_rule_zero_args}[rid]  # noqa: E731
    with pytest.raises(ValueError, match="arity"):
        validate(spec, ruleset=ruleset, rule_resolver=resolver)


def test_rule_arity_three_raises():
    spec = _bare_spec()
    ruleset = ValidationRuleset(name="bad", rule_ids=("bad_rule_three_args",))
    resolver = lambda rid: {"bad_rule_three_args": _bad_rule_three_args}[rid]  # noqa: E731
    with pytest.raises(ValueError, match="arity"):
        validate(spec, ruleset=ruleset, rule_resolver=resolver)


def _pure_rule_raises(spec):
    raise RuntimeError("rule blew up")


def test_throwing_rule_fails_closed_not_crash():
    """AREV-5: a rule that raises during execution must produce a Critical
    'rule_error' finding (fail closed), not crash validate() and take the
    pipeline down."""
    spec = _bare_spec()
    ruleset = ValidationRuleset(name="boom", rule_ids=("pure_rule_raises",))
    resolver = lambda rid: {"pure_rule_raises": _pure_rule_raises}[rid]  # noqa: E731
    verdict = validate(spec, ruleset=ruleset, rule_resolver=resolver)
    assert not verdict.passed
    assert len(verdict.findings) == 1
    f = verdict.findings[0]
    assert f.id == "rule_error_pure_rule_raises"
    assert f.severity == "Critical"
    assert f.category == "rule_error"
    assert "RuntimeError" in f.evidence[0].excerpt
