# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Deterministic, rule-driven validation primitive.

The ``validate`` primitive runs a named ``ValidationRuleset`` against a
``Spec``: each rule id is resolved to a callable that returns a tuple
of ``Finding``; the union becomes the ``Verdict.findings`` field. The
verdict is content-addressed against ``(spec, ruleset, findings)`` via
``hash_record`` so that opposite verdicts on the same Spec do not
collide.

The ``rule_resolver`` argument is the generic integration point for consumer-
owned rules. When it is omitted, the built-in resolver imports
``crucible.rules.<rule_id>`` lazily. That default rejects rule ids that are not
flat Python identifiers so an attacker-controlled id cannot traverse the
import namespace. Consumers outside Crucible should always pass a resolver.

The Verdict body itself is deterministic for a given (spec, ruleset,
rule outputs): no wall-clock timestamp or hostname is captured here.
Execution context (when/where the validation ran) belongs in a
sidecar run-record, not in the canonical Verdict body.

No corpus side effect — ``validate`` does not persist the verdict; the
caller decides whether to attach it via ``Spec.with_verdict``.
"""

from __future__ import annotations

import importlib
import inspect
from typing import Callable

from ..canonical import hash_record
from ..types.findings import (
    Evidence,
    Finding,
    ReviewCoverage,
    ValidationRuleset,
    Verdict,
)
from ..types.identity import ContentHash
from ..types.rule_context import RuleContext
from ..types.spec import Spec


# Severity values that, when observed in a finding, cause a reject decision.
# Anything outside the known non-blocking set
# ({"medium", "low"}) is treated as blocking so that off-Literal values
# (empty strings, typos, case variants like 'critical'/'HIGH') cannot
# silently bypass the gate. ``Finding.severity`` is a typing.Literal
# and is therefore not runtime-enforced by the frozen dataclass.
_NON_BLOCKING_SEVERITIES = frozenset({"medium", "low"})


class RuleContextRequiredError(ValueError):
    """Raised when a selected context-aware rule has no ``RuleContext``.

    Missing ambient state is a caller wiring error, not a rule result.  A
    validator cannot issue a complete Verdict when it did not execute every
    selected rule, so this exception is raised before any rule is invoked.
    """


def _is_blocking_severity(severity: str) -> bool:
    """Return True if ``severity`` requires a reject decision.

    Normalizes case/whitespace; unknown or malformed values are
    conservatively treated as blocking.
    """
    if not isinstance(severity, str):
        return True
    return severity.strip().lower() not in _NON_BLOCKING_SEVERITIES


def _default_resolver(rule_id: str):
    if not isinstance(rule_id, str) or not rule_id.isidentifier():
        raise ValueError(
            f"rule_id {rule_id!r} is not a flat Python identifier; "
            "the default resolver requires `rule_id.isidentifier()` "
            "to avoid leaking the internal namespace via importlib."
        )
    module = importlib.import_module(f"crucible.rules.{rule_id}")
    return getattr(module, rule_id)


def _rule_arity(rule: Callable) -> int:
    """Return the number of *positional* parameters the rule declares.

    Used by the dispatcher to choose between the pure
    ``rule(spec) -> tuple[Finding, ...]`` shape (arity 1) and the
    context-aware ``rule(spec, ctx) -> tuple[Finding, ...]`` shape
    (arity 2). Counts ``POSITIONAL_OR_KEYWORD`` and ``POSITIONAL_ONLY``
    parameters; ``*args`` / ``**kwargs`` / keyword-only parameters
    are not counted (rules MUST take spec + optional ctx as plain
    positionals — keyword-only ctx would be ambiguous w.r.t. the
    introspection rule).

    A rule that declares arity 0 or 3+ is a config error — the
    caller will surface a ValueError at dispatch time.
    """
    sig = inspect.signature(rule)
    return sum(
        1
        for p in sig.parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    )


def _rule_error_finding(rule_id: str, exc: BaseException) -> Finding:
    """A validation rule that raises during execution fails the gate as
    a Critical finding rather than crashing the whole ``validate()`` call.
    The gate is supposed to be the safety net; a buggy rule
    must not take the pipeline down with it."""
    return Finding(
        id=f"rule_error_{rule_id}",
        severity="Critical",
        category="rule_error",
        evidence=(
            Evidence(
                kind="exception",
                location=rule_id,
                excerpt=f"{type(exc).__name__}: {exc}"[:500],
            ),
        ),
        suggested_action=(
            f"rule {rule_id!r} raised during execution; fix the rule "
            "or the input it choked on"
        ),
        source="deterministic",
    )


def validate(
    spec: Spec,
    *,
    ruleset: ValidationRuleset | None = None,
    rule_resolver: Callable[[str], Callable[..., tuple[Finding, ...]]] | None = None,
    ctx: RuleContext | None = None,
) -> Verdict:
    """Run ``ruleset`` against ``spec`` and return a typed Verdict.

    When ``ruleset`` is None, an empty ruleset named ``'empty'`` is used (no
    findings, ``decision='pass'``). When ``rule_resolver`` is None, the
    built-in Crucible import path described in the module docstring is used.

    Duplicate ids in ``ruleset.rule_ids`` are deduped (order-preserving)
    so a misconfigured ruleset cannot silently fan out work.

    Rule signature shapes (introspected per rule):

    - ``rule(spec) -> tuple[Finding, ...]``       — pure rule, runs
      with the Spec only. Verdict is deterministic for the same
      ``(spec, ruleset)`` across machines.
    - ``rule(spec, ctx) -> tuple[Finding, ...]``  — context-aware
      rule. Receives the ``RuleContext`` passed via ``ctx=`` kwarg.
      Verdicts reflect the local environment at validation time
      (e.g., working-tree state for filesystem-grounded rules).
      Two machines may produce different verdicts on the same spec
      if their local environments diverge — that disagreement IS
      the diagnostic signal (the spec's declared base_commit /
      submodule_path describes the intended environment).

    Selecting any context-aware rule without supplying ``ctx`` raises
    :class:`RuleContextRequiredError` before any rule executes. Missing
    ambient state is a caller configuration error: silently omitting a
    selected check could otherwise produce a false ``pass`` Verdict.
    """
    effective = ValidationRuleset(name="empty", rule_ids=()) if ruleset is None else ruleset
    resolver = rule_resolver if rule_resolver is not None else _default_resolver

    # Order-preserving dedupe — duplicate rule_ids are almost certainly
    # a config typo; running the same rule N times distorts
    # persistence_count and wastes work.
    deduped_rule_ids: tuple[str, ...] = tuple(dict.fromkeys(effective.rule_ids))

    # Resolve and validate the complete ruleset before executing any rule.
    # This gives configuration failures atomic semantics: a missing context
    # or malformed signature cannot occur after earlier rules have already
    # performed work.
    resolved_rules: list[tuple[str, Callable, int]] = []
    for rid in deduped_rule_ids:
        rule = resolver(rid)
        arity = _rule_arity(rule)
        if arity not in (1, 2):
            raise ValueError(
                f"rule {rid!r} has unsupported arity {arity}; "
                "must be rule(spec) or rule(spec, ctx)."
            )
        resolved_rules.append((rid, rule, arity))

    context_rule_ids = tuple(
        rid for rid, _rule, arity in resolved_rules if arity == 2
    )
    if ctx is None and context_rule_ids:
        joined = ", ".join(repr(rid) for rid in context_rule_ids)
        raise RuleContextRequiredError(
            f"ruleset {effective.name!r} requires ctx for context-aware "
            f"rule(s): {joined}"
        )

    all_findings: list[Finding] = []
    for rid, rule, arity in resolved_rules:
        # Only the rule's *execution* is guarded: a rule that raises must
        # fail the gate (Critical finding), not crash validate() and take
        # the whole pipeline down. Arity / return-type are assay's
        # own wiring contract — those stay hard raises (and are pinned by
        # tests), surfaced at dev time, not turned into findings.
        if arity == 1:
            try:
                result = rule(spec)
            except Exception as exc:  # noqa: BLE001 — throwing rule fails closed
                all_findings.append(_rule_error_finding(rid, exc))
                continue
        elif arity == 2:
            # The complete ruleset was preflighted above, so ctx cannot be
            # None on this branch. The assertion protects the type narrowing
            # and documents that rule execution never has skip semantics.
            assert ctx is not None
            try:
                result = rule(spec, ctx)
            except Exception as exc:  # noqa: BLE001 — throwing rule fails closed
                all_findings.append(_rule_error_finding(rid, exc))
                continue
        if result is None or not isinstance(result, tuple):
            raise TypeError(
                f"rule {rid!r} returned {type(result).__name__}; "
                "rules must return tuple[Finding, ...]."
            )
        all_findings.extend(result)

    findings_tuple = tuple(all_findings)
    decision = (
        "reject"
        if any(_is_blocking_severity(f.severity) for f in findings_tuple)
        else "pass"
    )

    # artifact_ref must incorporate ruleset identity and findings so
    # that opposite verdicts on the same Spec do not collide. We hash
    # the deterministic verdict body (excluding execution context),
    # which is exactly the content a corpus would key on.
    artifact_payload = {
        "spec": spec,
        "ruleset": effective,
        "findings": findings_tuple,
        "decision": decision,
    }

    return Verdict(
        rubric=effective.name,
        decision=decision,
        findings=findings_tuple,
        backend_name=None,
        transcript_ref=None,
        cost_usd=0.0,
        artifact_ref=ContentHash(hash_record(artifact_payload)),
        artifact_commit=spec.project_context.base_commit,
        reasoning="",
        coverage=ReviewCoverage(
            complete=True,
            sources_reviewed=("spec",),
            omissions=(),
        ),
        # Execution context (wall-clock timestamp, hostname, cwd) is
        # intentionally omitted from the canonical Verdict body so
        # that ``hash_record(verdict)`` is deterministic for identical
        # inputs. Run-context belongs in a sidecar record, not here.
        execution=None,
        kind="validate",
    )


__all__ = ["RuleContextRequiredError", "validate"]
