# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Tests for the type substrate: frozen-dataclass invariants and
``with_*`` semantics on Spec.

Every Spec / Verdict / Finding / Implementation / ... is frozen;
attempting to mutate a field raises ``dataclasses.FrozenInstanceError``.
The ``with_*`` helpers must NOT mutate the receiver — they return a
new instance with version + 1.
"""

from __future__ import annotations

import dataclasses

import pytest

from assay.canonical import hash_record, parse, serialize

from assay import (
    ActivationConstraint,
    AssertionCriterion,
    BenchConfig,
    BlastRadiusConstraint,
    BudgetConstraint,
    ContentHash,
    Evidence,
    ExecutionCompleteness,
    ExecutionContext,
    Finding,
    Implementation,
    LifecycleEvent,
    Measurement,
    ParityCriterion,
    PlanOutcome,
    PlanPolicy,
    ProjectContext,
    ReadOnlySourceReference,
    ReviewCoverage,
    Spec,
    Tier,
    TierDAG,
    Usage,
    UsagePricingEvidence,
    UsagePricingSegment,
    Verdict,
)
from assay.corpus.codec import decode_dataclass


def test_frozen_record_mappings_are_deeply_immutable():
    execution = _exec_ctx()
    project = _project_ctx()
    config = BenchConfig(
        bench_id="b",
        runner="runner",
        parameters={"nested": {"items": [1, 2]}},
    )
    with pytest.raises(TypeError, match="immutable"):
        execution.toolchain["assay"] = "mutated"
    with pytest.raises(TypeError, match="immutable"):
        project.runners["verify"] = "evil.runner"
    configured_project = ProjectContext(
        project="assay",
        repo_url="https://github.com/whact2025/assay.git",
        base_commit="0000000",
        branch="main",
        submodule_path=None,
        runners={},
        runner_parameters={"per_language": {"rust": {"features": ["gpu"]}}},
    )
    with pytest.raises(TypeError, match="immutable"):
        configured_project.runner_parameters["per_language"]["rust"][
            "features"
        ] = ()
    with pytest.raises(TypeError, match="immutable"):
        config.parameters["nested"]["extra"] = True
    assert config.parameters["nested"]["items"] == (1, 2)


def _exec_ctx():
    return ExecutionContext(
        timestamp="2026-05-16T15:00:00+00:00",
        machine="perflab1",
        cwd="/tmp",
        toolchain={"assay": "0.0.0"},
        backend=None,
        model=None,
        recipe=None,
        recipe_commit=None,
        gpu_model=None,
        run_id=None,
    )


def _project_ctx():
    return ProjectContext(
        project="assay",
        repo_url="https://github.com/whact2025/assay.git",
        base_commit="0000000",
        branch="main",
        submodule_path=None,
        runners={},
    )


def _v0_spec() -> Spec:
    return Spec(
        version=0,
        parent=None,
        root=None,
        schema_version=1,
        project_context=_project_ctx(),
        execution=None,
        intent="test",
        constraints=(BudgetConstraint(usd_limit=10.0),),
        done_criteria=(ParityCriterion(reference_commit="abc", test_name="t"),),
        plan_policy=None,
    )


# --- Frozen dataclass invariants --------------------------------------


def test_spec_is_frozen():
    s = _v0_spec()
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.version = 1  # type: ignore[misc]


def test_verdict_is_frozen():
    v = Verdict(
        rubric="r",
        decision="pass",
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
        execution=_exec_ctx(),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        v.passed = False  # type: ignore[misc]


def test_plan_outcome_pricing_fields_preserve_legacy_shape_and_decode():
    legacy = PlanOutcome(
        spec=_v0_spec(),
        candidate_dag=None,
        candidate_ref=None,
        decision="inconclusive",
        status="no_candidate",
        rounds_used=1,
        final_critic_decision=None,
        final_judge_decision=None,
        reason="legacy result",
        cost_usd=0.25,
    )

    legacy_payload = parse(serialize(legacy))
    assert "input_tokens" not in legacy_payload
    assert "output_tokens" not in legacy_payload
    assert "cache_creation_tokens" not in legacy_payload
    assert "cache_read_tokens" not in legacy_payload
    assert "execution_completeness" not in legacy_payload
    assert "pricing_evidence" not in legacy_payload
    assert hash_record(legacy) == hash_record(legacy_payload)
    restored_legacy = decode_dataclass(legacy_payload, PlanOutcome)
    assert restored_legacy == legacy
    assert restored_legacy.usage == Usage(cost_usd=0.25)

    segment = UsagePricingSegment(
        input_tokens=13,
        output_tokens=5,
        cache_creation_tokens=2,
        cache_read_tokens=3,
        cost_usd=0.25,
    )
    metered = dataclasses.replace(
        legacy,
        input_tokens=13,
        output_tokens=5,
        cache_creation_tokens=2,
        cache_read_tokens=3,
        execution_completeness=ExecutionCompleteness(
            token_usage_complete=True,
            cost_complete=True,
            terminal_complete=True,
            trace_complete=False,
        ),
        pricing_evidence=UsagePricingEvidence(
            segments=(segment,),
            complete=True,
        ),
    )
    metered_payload = parse(serialize(metered))
    assert metered_payload["input_tokens"] == 13
    assert metered_payload["cache_read_tokens"] == 3
    assert decode_dataclass(metered_payload, PlanOutcome) == metered

    with pytest.raises(ValueError, match="must reconcile aggregate usage"):
        dataclasses.replace(metered, output_tokens=6)


def test_verdict_usage_preserves_legacy_canonical_shape_and_decode():
    field_names = [item.name for item in dataclasses.fields(Verdict)]
    assert field_names.index("cache_creation_tokens") > field_names.index(
        "pricing_evidence"
    )
    assert field_names.index("cache_read_tokens") == (
        field_names.index("cache_creation_tokens") + 1
    )

    verdict = Verdict(
        rubric="r",
        decision="pass",
        findings=(),
        artifact_ref=ContentHash("sha256:" + "a" * 64),
        artifact_commit=None,
        reasoning="",
        coverage=ReviewCoverage(
            complete=True, sources_reviewed=("artifact",), omissions=()
        ),
        backend_name=None,
        transcript_ref=None,
        cost_usd=0.0,
        execution=_exec_ctx(),
    )

    legacy_payload = parse(serialize(verdict))
    assert "input_tokens" not in legacy_payload
    assert "output_tokens" not in legacy_payload
    assert "cache_creation_tokens" not in legacy_payload
    assert "cache_read_tokens" not in legacy_payload
    assert hash_record(verdict) == hash_record(legacy_payload)
    restored_legacy = decode_dataclass(legacy_payload, Verdict)
    assert restored_legacy.input_tokens == 0
    assert restored_legacy.output_tokens == 0
    assert restored_legacy.cache_creation_tokens is None
    assert restored_legacy.cache_read_tokens is None
    assert restored_legacy == verdict

    exact_zero = dataclasses.replace(
        verdict,
        cache_creation_tokens=0,
        cache_read_tokens=0,
    )
    exact_zero_payload = parse(serialize(exact_zero))
    assert exact_zero_payload["cache_creation_tokens"] == 0
    assert exact_zero_payload["cache_read_tokens"] == 0
    assert decode_dataclass(exact_zero_payload, Verdict) == exact_zero
    assert hash_record(exact_zero) != hash_record(verdict)

    metered = dataclasses.replace(
        verdict,
        input_tokens=123,
        output_tokens=45,
        cache_creation_tokens=50,
        cache_read_tokens=20,
    )
    metered_payload = parse(serialize(metered))
    assert metered_payload["input_tokens"] == 123
    assert metered_payload["output_tokens"] == 45
    assert metered_payload["cache_creation_tokens"] == 50
    assert metered_payload["cache_read_tokens"] == 20
    assert decode_dataclass(metered_payload, Verdict) == metered
    assert hash_record(metered) != hash_record(verdict)

    with pytest.raises(ValueError, match="cache-token subsets must not exceed"):
        dataclasses.replace(
            metered,
            input_tokens=10,
            execution_completeness=ExecutionCompleteness(
                token_usage_complete=True,
                cost_complete=False,
            ),
        )

    with pytest.raises(ValueError, match="both be present or both be absent"):
        dataclasses.replace(verdict, cache_creation_tokens=0)


def test_verdict_legacy_complete_cache_evidence_preserves_hash_and_positionals():
    segment = UsagePricingSegment(
        input_tokens=200_811,
        output_tokens=20_074,
        cache_creation_tokens=0,
        cache_read_tokens=169_472,
        cost_usd=0.844,
    )
    completeness = ExecutionCompleteness(
        token_usage_complete=True,
        cost_complete=True,
        terminal_complete=True,
        trace_complete=True,
    )
    # This is the complete positional constructor shape that existed before
    # the additive cache aggregates. It must remain valid without changing
    # the legacy canonical bytes or content hash.
    legacy = Verdict(
        "r",
        "pass",
        (),
        ContentHash("sha256:" + "a" * 64),
        None,
        "legacy complete evidence",
        ReviewCoverage(
            complete=True,
            sources_reviewed=("artifact",),
            omissions=(),
        ),
        "claude",
        ContentHash("sha256:" + "b" * 64),
        0.844,
        None,
        200_811,
        20_074,
        {},
        "critic",
        None,
        None,
        "claude-opus",
        "high",
        "source-tools",
        (),
        completeness,
        UsagePricingEvidence(segments=(segment,), complete=True),
    )

    payload = parse(serialize(legacy))
    assert "cache_creation_tokens" not in payload
    assert "cache_read_tokens" not in payload
    restored = decode_dataclass(payload, Verdict)
    assert restored == legacy
    assert hash_record(restored) == hash_record(payload)
    assert restored.cache_creation_tokens is None
    assert restored.cache_read_tokens is None
    assert restored.accounting_debt == ()


def test_verdict_legacy_incomplete_cache_evidence_exposes_token_debt():
    incomplete_evidence = UsagePricingEvidence(
        segments=(
            UsagePricingSegment(
                input_tokens=17,
                output_tokens=1,
                cache_creation_tokens=15,
                cache_read_tokens=0,
                cost_usd=0.10,
            ),
            UsagePricingSegment(
                input_tokens=17,
                output_tokens=1,
                cache_creation_tokens=15,
                cache_read_tokens=0,
                cost_usd=0.10,
            ),
        ),
        complete=False,
    )
    legacy = Verdict(
        rubric="r",
        decision="pass",
        findings=(),
        artifact_ref=ContentHash("sha256:" + "a" * 64),
        artifact_commit=None,
        reasoning="legacy incomplete evidence",
        coverage=ReviewCoverage(
            complete=True,
            sources_reviewed=("artifact",),
            omissions=(),
        ),
        backend_name="claude",
        transcript_ref=ContentHash("sha256:" + "b" * 64),
        cost_usd=0.21,
        execution=None,
        input_tokens=17,
        output_tokens=7,
        execution_completeness=ExecutionCompleteness(
            token_usage_complete=True,
            cost_complete=True,
            terminal_complete=True,
            trace_complete=True,
        ),
        pricing_evidence=incomplete_evidence,
    )

    payload = parse(serialize(legacy))
    assert "cache_creation_tokens" not in payload
    assert "cache_read_tokens" not in payload
    restored = decode_dataclass(payload, Verdict)
    assert restored == legacy
    assert hash_record(restored) == hash_record(payload)
    assert restored.accounting_debt == ("token_usage_complete",)

    exact_zero = dataclasses.replace(
        legacy,
        cache_creation_tokens=0,
        cache_read_tokens=0,
    )
    assert exact_zero.accounting_debt == ()


def test_verdict_persists_aggregate_accounting_debt():
    verdict = Verdict(
        rubric="r",
        decision="revise",
        findings=(),
        artifact_ref=ContentHash("sha256:" + "a" * 64),
        artifact_commit=None,
        reasoning="valid semantic result",
        coverage=ReviewCoverage(
            complete=True,
            sources_reviewed=("artifact",),
            omissions=(),
        ),
        backend_name="claude",
        transcript_ref=ContentHash("sha256:" + "b" * 64),
        cost_usd=0.4,
        execution=None,
        execution_completeness=ExecutionCompleteness(
            token_usage_complete=False,
            cost_complete=False,
            terminal_complete=True,
            trace_complete=True,
        ),
    )

    payload = parse(serialize(verdict))
    assert payload["execution_completeness"] == {
        "token_usage_complete": False,
        "cost_complete": False,
        "terminal_complete": True,
        "trace_complete": True,
    }
    assert verdict.accounting_debt == (
        "token_usage_complete",
        "cost_complete",
    )
    assert decode_dataclass(payload, Verdict) == verdict


def test_verdict_artifacts_defaults_to_empty_dict():
    """Verdict.artifacts is an output-side carrier for execution-side
    artifacts, defaulting to an empty dict — callsites that don't
    populate it stay terse."""
    v = Verdict(
        rubric="r", decision="pass", findings=(),
        artifact_ref=ContentHash("sha256:abc"),
        artifact_commit=None, reasoning="",
        coverage=ReviewCoverage(
            complete=True, sources_reviewed=("artifact",), omissions=()
        ),
        backend_name=None, transcript_ref=None,
        cost_usd=0.0, execution=_exec_ctx(),
    )
    assert v.artifacts == {}


def test_verdict_artifacts_carries_content_hashes():
    """The artifacts field maps stable kind strings to corpus refs."""
    v = Verdict(
        rubric="r", decision="revise", findings=(),
        artifact_ref=ContentHash("sha256:abc"),
        artifact_commit=None, reasoning="",
        coverage=ReviewCoverage(
            complete=True, sources_reviewed=("artifact",), omissions=()
        ),
        backend_name=None, transcript_ref=None,
        cost_usd=0.0, execution=_exec_ctx(),
        artifacts={
            "stderr": ContentHash("sha256:1111"),
            "build_warnings": ContentHash("sha256:2222"),
        },
    )
    assert v.artifacts["stderr"] == "sha256:1111"
    assert v.artifacts["build_warnings"] == "sha256:2222"


def test_finding_is_frozen():
    f = Finding(
        id="x",
        severity="Low",
        category="c",
        evidence=(Evidence(kind="k", location="l", excerpt="e"),),
        suggested_action=None,
        source="deterministic",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        f.severity = "Critical"  # type: ignore[misc]


def test_project_context_is_frozen():
    pc = _project_ctx()
    with pytest.raises(dataclasses.FrozenInstanceError):
        pc.project = "other"  # type: ignore[misc]


def test_constraints_are_frozen():
    c = BudgetConstraint(usd_limit=5.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.usd_limit = 10.0  # type: ignore[misc]
    r = BlastRadiusConstraint(no_edits_outside=("a", "b"))
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.no_edits_outside = ()  # type: ignore[misc]


def test_dag_types_are_frozen():
    d = TierDAG(tiers=())
    with pytest.raises(dataclasses.FrozenInstanceError):
        d.tiers = (Tier(tier_id="t0", slices=(), depends_on=()),)  # type: ignore[misc]


def test_mutation_region_defaults_and_freeze():
    """MutationRegion carries (file, start_marker, end_marker) with
    the canonical CRUCIBLE-MUTATE-* markers as defaults."""
    from assay import MutationRegion
    r = MutationRegion(file="src/bvh.rs")
    assert r.file == "src/bvh.rs"
    assert r.start_marker == "CRUCIBLE-MUTATE-START"
    assert r.end_marker == "CRUCIBLE-MUTATE-END"
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.file = "other.rs"  # type: ignore[misc]


def test_implementation_directive_mutation_regions_default_empty():
    """ImplementationDirective.mutation_regions defaults to () —
    region-level scoping is per-file opt-in."""
    from assay import ImplementationDirective
    d = ImplementationDirective(
        slice_id="s", intent="t", files_in_scope=(), creates_paths=(),
        verify_mode="build_only", blast_radius={},
    )
    assert d.mutation_regions == ()
    assert d.requirement_refs == ()
    assert "requirement_refs" not in parse(serialize(d))
    assert d.read_only_references == ()
    assert "read_only_references" not in parse(serialize(d))


def _source_reference(**changes):
    return ReadOnlySourceReference(**{
        "reference_id": "existing-test.1",
        "path": "tests/test_existing.py",
        "sha256": "sha256:" + "a" * 64,
        "start_line": 3,
        "end_line": 8,
        **changes,
    })


@pytest.mark.parametrize("changes", [
    {"reference_id": ""}, {"reference_id": "x" * 65},
    {"reference_id": "white space"}, {"reference_id": "../test"},
    {"reference_id": 1}, {"path": ""}, {"path": "/absolute.py"},
    {"path": "../escape.py"}, {"path": "src/../escape.py"},
    {"path": "src/./file.py"}, {"path": "./file.py"},
    {"path": "src//file.py"}, {"path": "src/"},
    {"path": "src\\file.py"}, {"path": "src/*.py"},
    {"path": "src/a?.py"}, {"path": "src/[ab].py"},
    {"path": "src/file\n.py"}, {"path": "src/\ud800.py"},
    {"path": "a" * 4097}, {"path": None},
    {"sha256": "a" * 64}, {"sha256": "sha256:" + "A" * 64},
    {"sha256": "sha256:" + "a" * 63}, {"sha256": None},
    {"start_line": True}, {"end_line": True}, {"start_line": 0},
    {"start_line": 1.0}, {"end_line": "8"}, {"end_line": 2},
])
def test_read_only_reference_rejects_ambiguous_identity_or_locator(changes):
    with pytest.raises(ValueError):
        _source_reference(**changes)


def test_read_only_reference_is_frozen_typed_and_closed():
    from assay import ImplementationDirective
    reference = _source_reference()
    with pytest.raises(dataclasses.FrozenInstanceError):
        reference.path = "different.py"
    directive = ImplementationDirective(
        slice_id="s", intent="t", files_in_scope=("output.md",),
        creates_paths=(), verify_mode="tests", blast_radius={},
        read_only_references=(reference,),
    )
    restored = decode_dataclass(parse(serialize(directive)), ImplementationDirective)
    assert restored == directive
    assert type(restored.read_only_references[0]) is ReadOnlySourceReference
    payload = parse(serialize(reference))
    payload["writable"] = True
    with pytest.raises(ValueError, match="unknown"):
        decode_dataclass(payload, ReadOnlySourceReference)
    for invalid in ([reference], (dict(payload),)):
        with pytest.raises(TypeError, match="tuple"):
            dataclasses.replace(directive, read_only_references=invalid)
    with pytest.raises(ValueError, match="distinct"):
        dataclasses.replace(directive, read_only_references=(reference, reference))


def test_implementation_directive_carries_authoritative_requirement_refs():
    from assay import ImplementationDirective

    clauses = (
        "The returned values retain their declared order.",
        "Unknown values are rejected.",
    )
    directive = ImplementationDirective(
        slice_id="s",
        intent="generated implementation notes",
        files_in_scope=(),
        creates_paths=(),
        verify_mode="build_only",
        blast_radius={},
        requirement_refs=clauses,
    )

    assert directive.requirement_refs == clauses


def test_implementation_directive_carries_mutation_regions():
    from assay import ImplementationDirective, MutationRegion
    regions = (
        MutationRegion(file="src/bvh.rs"),
        MutationRegion(
            file="src/lib.rs",
            start_marker="OPTIM-START",
            end_marker="OPTIM-END",
        ),
    )
    d = ImplementationDirective(
        slice_id="s", intent="t",
        files_in_scope=("src/bvh.rs", "src/lib.rs"),
        creates_paths=(), verify_mode="build_only", blast_radius={},
        mutation_regions=regions,
    )
    assert len(d.mutation_regions) == 2
    assert d.mutation_regions[0].file == "src/bvh.rs"
    assert d.mutation_regions[1].start_marker == "OPTIM-START"


def test_tier_smoke_bench_config_defaults_to_none():
    """Tier.smoke_bench_config defaults to None — most tiers don't
    declare a headline smoke bench."""
    t = Tier(tier_id="t0", slices=(), depends_on=())
    assert t.smoke_bench_config is None


def test_tier_carries_smoke_bench_config():
    """When the planner sets a smoke_bench_config, it rides on the Tier
    record. The consumer (workflows/tier.py Phase 2.5) reads it to
    decide whether to gate breach on a single-sample regression."""
    from assay import BenchConfig
    cfg = BenchConfig(bench_id="headline_bvh", runner="fabric_render.bench")
    t = Tier(
        tier_id="t0", slices=(), depends_on=(),
        smoke_bench_config=cfg,
    )
    assert t.smoke_bench_config is not None
    assert t.smoke_bench_config.bench_id == "headline_bvh"
    assert t.smoke_bench_config.runner == "fabric_render.bench"


# --- with_* semantics --------------------------------------------------


def test_with_tier_dag_bumps_version_and_preserves_receiver():
    s0 = _v0_spec()
    dag = TierDAG(tiers=())
    s1 = s0.with_tier_dag(dag)
    assert s0.version == 0  # receiver unchanged
    assert s0.tier_dag is None
    assert s1.version == 1
    assert s1.tier_dag is dag
    assert s1.parent is not None
    assert s1.parent.startswith("sha256:")
    assert s1.root is not None  # v0 hash captured


def test_with_appended_refs_bumps_lineage_once():
    spec = _v0_spec()
    verdict_refs = (
        ContentHash("sha256:" + "1" * 64),
        ContentHash("sha256:" + "2" * 64),
    )

    committed = spec.with_appended_refs(verdict_refs=verdict_refs)

    assert committed.version == spec.version + 1
    assert committed.parent == ContentHash(hash_record(spec))
    assert committed.verdict_refs == verdict_refs


def test_with_implementation_appends_ref_and_does_not_inline():
    s0 = _v0_spec()
    impl = Implementation(
        slice_id="slice0",
        base_commit="aaa",
        result_commit="bbb",
        diff_ref=ContentHash("sha256:diff"),
        transcript_ref=ContentHash("sha256:trans"),
        files_modified=("a.py",),
        classification="success",
        execution=_exec_ctx(),
        scope_escapes=(),
    )
    s1 = s0.with_implementation(impl)
    assert len(s0.implementation_refs) == 0  # receiver unchanged
    assert len(s1.implementation_refs) == 1
    assert s1.implementation_refs[0].startswith("sha256:")
    assert s1.version == 1


def test_implementation_artifacts_defaults_to_empty_dict():
    """Implementation.artifacts mirrors Verdict.artifacts — a
    workflow-populated carrier for build stderr, verify output, and
    tool-call payloads captured around the implement step."""
    impl = Implementation(
        slice_id="slice0", base_commit="aaa", result_commit=None,
        diff_ref=ContentHash("sha256:diff"),
        transcript_ref=ContentHash("sha256:trans"),
        files_modified=(), classification="success",
        execution=_exec_ctx(),
        scope_escapes=(),
    )
    assert impl.artifacts == {}
    assert impl.scope_escapes == ()
    assert impl.input_tokens == 0
    assert impl.output_tokens == 0
    assert impl.cache_creation_tokens is None
    assert impl.cache_read_tokens is None
    assert impl.stop_reason is None


def test_implementation_usage_preserves_legacy_shape_and_exact_zero():
    field_names = [item.name for item in dataclasses.fields(Implementation)]
    assert field_names.index("cache_creation_tokens") > field_names.index(
        "pricing_evidence"
    )
    assert field_names.index("cache_read_tokens") == (
        field_names.index("cache_creation_tokens") + 1
    )

    legacy = Implementation(
        slice_id="slice0",
        base_commit="aaa",
        result_commit=None,
        diff_ref=ContentHash("sha256:" + "d" * 64),
        transcript_ref=ContentHash("sha256:" + "e" * 64),
        files_modified=("a.py",),
        classification="success",
        execution=_exec_ctx(),
        scope_escapes=(),
        cost_usd=0.25,
        input_tokens=123,
        output_tokens=45,
        execution_completeness=ExecutionCompleteness(
            token_usage_complete=True,
            cost_complete=True,
            terminal_complete=True,
            trace_complete=True,
        ),
    )

    legacy_payload = parse(serialize(legacy))
    assert "cache_creation_tokens" not in legacy_payload
    assert "cache_read_tokens" not in legacy_payload
    assert hash_record(legacy) == hash_record(legacy_payload)
    restored_legacy = decode_dataclass(legacy_payload, Implementation)
    assert restored_legacy == legacy
    assert restored_legacy.cache_creation_tokens is None
    assert restored_legacy.cache_read_tokens is None
    assert restored_legacy.accounting_debt == ("token_usage_complete",)

    exact_zero = dataclasses.replace(
        legacy,
        cache_creation_tokens=0,
        cache_read_tokens=0,
    )
    exact_zero_payload = parse(serialize(exact_zero))
    assert exact_zero_payload["cache_creation_tokens"] == 0
    assert exact_zero_payload["cache_read_tokens"] == 0
    assert decode_dataclass(exact_zero_payload, Implementation) == exact_zero
    assert hash_record(exact_zero) != hash_record(legacy)
    assert exact_zero.accounting_debt == ()

    metered = dataclasses.replace(
        legacy,
        cache_creation_tokens=50,
        cache_read_tokens=20,
    )
    metered_payload = parse(serialize(metered))
    assert metered_payload["cache_creation_tokens"] == 50
    assert metered_payload["cache_read_tokens"] == 20
    assert decode_dataclass(metered_payload, Implementation) == metered

    with pytest.raises(ValueError, match="cache-token subsets must not exceed"):
        dataclasses.replace(metered, input_tokens=10)

    with pytest.raises(ValueError, match="both be present or both be absent"):
        dataclasses.replace(legacy, cache_creation_tokens=0)


def test_implementation_complete_pricing_evidence_reconciles_legacy_caches():
    evidence = UsagePricingEvidence(
        segments=(UsagePricingSegment(123, 45, 50, 20, 0.25),),
        complete=True,
    )
    completeness = ExecutionCompleteness(
        token_usage_complete=True,
        cost_complete=True,
        terminal_complete=True,
        trace_complete=True,
    )
    # This is the complete positional constructor shape that existed before
    # the additive cache aggregates. It must remain valid without changing
    # the legacy canonical bytes or content hash.
    legacy = Implementation(
        "slice0",
        "aaa",
        None,
        ContentHash("sha256:" + "d" * 64),
        ContentHash("sha256:" + "e" * 64),
        ("a.py",),
        "success",
        _exec_ctx(),
        (),
        {},
        0.25,
        123,
        45,
        None,
        completeness,
        evidence,
    )

    payload = parse(serialize(legacy))
    assert "cache_creation_tokens" not in payload
    assert "cache_read_tokens" not in payload
    restored = decode_dataclass(payload, Implementation)
    assert restored == legacy
    assert hash_record(restored) == hash_record(payload)
    assert restored.accounting_debt == ()

    exact = dataclasses.replace(
        legacy,
        cache_creation_tokens=50,
        cache_read_tokens=20,
    )
    assert decode_dataclass(
        parse(serialize(exact)), Implementation
    ) == exact

    with pytest.raises(ValueError, match="must reconcile aggregate usage"):
        dataclasses.replace(
            legacy,
            cache_creation_tokens=49,
            cache_read_tokens=20,
        )


def test_implementation_persists_aggregate_accounting_debt():
    implementation = Implementation(
        slice_id="slice0",
        base_commit="aaa",
        result_commit=None,
        diff_ref=ContentHash("sha256:" + "d" * 64),
        transcript_ref=ContentHash("sha256:" + "e" * 64),
        files_modified=("a.py",),
        classification="hard_fail_transport",
        execution=_exec_ctx(),
        scope_escapes=(),
        cost_usd=0.2,
        execution_completeness=ExecutionCompleteness(
            token_usage_complete=False,
            cost_complete=False,
            terminal_complete=True,
            trace_complete=True,
        ),
    )

    payload = parse(serialize(implementation))
    assert payload["execution_completeness"] == {
        "token_usage_complete": False,
        "cost_complete": False,
        "terminal_complete": True,
        "trace_complete": True,
    }
    assert implementation.accounting_debt == (
        "token_usage_complete",
        "cost_complete",
    )
    assert decode_dataclass(payload, Implementation) == implementation


def test_implementation_artifacts_threads_via_replace():
    """The frozen-dataclass + dict-default pattern: callers use
    dataclasses.replace to append rather than mutating in place."""
    impl = Implementation(
        slice_id="slice0", base_commit="aaa", result_commit=None,
        diff_ref=ContentHash("sha256:diff"),
        transcript_ref=ContentHash("sha256:trans"),
        files_modified=(), classification="hard_fail_empty_diff",
        execution=_exec_ctx(),
        scope_escapes=(),
    )
    impl2 = dataclasses.replace(
        impl,
        artifacts={**impl.artifacts, "stderr": ContentHash("sha256:err")},
    )
    assert impl.artifacts == {}            # receiver unchanged
    assert impl2.artifacts == {"stderr": "sha256:err"}


def test_with_verdict_appends_ref():
    s0 = _v0_spec()
    v = Verdict(
        rubric="judge",
        decision="pass",
        findings=(),
        artifact_ref=ContentHash("sha256:art"),
        artifact_commit=None,
        reasoning="",
        coverage=ReviewCoverage(
            complete=True, sources_reviewed=("artifact",), omissions=()
        ),
        backend_name=None,
        transcript_ref=None,
        cost_usd=0.0,
        execution=_exec_ctx(),
    )
    s1 = s0.with_verdict(v)
    assert len(s1.verdict_refs) == 1
    assert s1.version == 1


def test_with_measurement_appends_ref():
    s0 = _v0_spec()
    m = Measurement(
        bench_id="b0",
        commit="xyz",
        config_hash="cfg",
        values={"p50_ms": 4.2},
        raw_output_ref=ContentHash("sha256:raw"),
        execution=_exec_ctx(),
    )
    s1 = s0.with_measurement(m)
    assert len(s1.measurement_refs) == 1


def test_with_lifecycle_event_appends_ref():
    s0 = _v0_spec()
    ev = LifecycleEvent(
        kind="landed",
        reason="passed gate",
        affected_versions=(0,),
        execution=_exec_ctx(),
    )
    s1 = s0.with_lifecycle_event(ev)
    assert len(s1.lifecycle_event_refs) == 1
    assert s1.version == 1


def test_version_chain_is_monotonic():
    s = _v0_spec()
    for _ in range(5):
        s = s.with_lifecycle_event(
            LifecycleEvent(
                kind="annotated",
                reason=f"v{s.version}",
                affected_versions=(s.version,),
                execution=_exec_ctx(),
            )
        )
    assert s.version == 5
    assert len(s.lifecycle_event_refs) == 5


def test_root_is_v0_hash_throughout_chain():
    s0 = _v0_spec()
    s1 = s0.with_tier_dag(TierDAG(tiers=()))
    s2 = s1.with_lifecycle_event(
        LifecycleEvent(
            kind="annotated",
            reason="r",
            affected_versions=(1,),
            execution=_exec_ctx(),
        )
    )
    assert s1.root == s2.root
    assert s2.root is not None


def test_plan_policy_frozen():
    p = PlanPolicy(
        max_critique_rounds=2,
        judge_required=True,
        breaker_required=False,
        persistence_veto_threshold=3,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.max_critique_rounds = 5  # type: ignore[misc]


# --- YAML loading ------------------------------------------------------


def test_spec_from_yaml_minimal(tmp_path):
    yaml_text = """
schema_version: 1
project_context:
  project: assay
  repo_url: https://github.com/whact2025/assay.git
  base_commit: "0000000"
  branch: main
  submodule_path: null
  runners: {}
intent: |
  Stand up the foundation phase.
constraints:
  - kind: budget
    usd_limit: 200
done_criteria:
  - kind: parity
    reference_commit: abc
    test_name: t
"""
    s = Spec.from_yaml(yaml_text)
    assert s.version == 0
    assert s.parent is None
    assert s.intent.startswith("Stand up")
    assert len(s.constraints) == 1
    assert s.constraints[0].kind == "budget"
    assert len(s.done_criteria) == 1
    assert s.done_criteria[0].kind == "parity"
    assert s.project_context.runner_parameters == {}


def test_project_runner_parameters_are_generic_bound_spec_data():
    yaml_text = """
schema_version: 1
project_context:
  project: fabric-usd
  repo_url: ssh://example.invalid/fabric-usd.git
  base_commit: "0000000"
  branch: main
  submodule_path: fabric-usd
  runners:
    build: crucible.common.runners.polyglot
    verify: crucible.common.runners.polyglot
  runner_parameters:
    per_language:
      rust:
        crate: fabric-render-ovrtx
        cargo_features: [ovrtx]
        cargo_test_args: ["--", "--test-threads=1"]
intent: Exercise a configured project runner.
constraints: []
done_criteria: []
"""
    spec = Spec.from_yaml(yaml_text)

    rust = spec.project_context.runner_parameters["per_language"]["rust"]
    assert rust["crate"] == "fabric-render-ovrtx"
    assert rust["cargo_features"] == ("ovrtx",)
    assert rust["cargo_test_args"] == ("--", "--test-threads=1")
    payload = parse(serialize(spec))
    assert payload["project_context"]["runner_parameters"] == {
        "per_language": {
            "rust": {
                "cargo_features": ["ovrtx"],
                "cargo_test_args": ["--", "--test-threads=1"],
                "crate": "fabric-render-ovrtx",
            },
        },
    }
    assert decode_dataclass(payload, Spec) == spec


def test_empty_runner_parameters_preserve_legacy_canonical_shape_and_decode():
    spec = _v0_spec()
    payload = parse(serialize(spec))

    assert "runner_parameters" not in payload["project_context"]
    restored = decode_dataclass(payload, Spec)
    assert restored == spec
    assert restored.project_context.runner_parameters == {}


def test_runner_parameters_reject_non_json_values():
    with pytest.raises(TypeError, match="JSON-compatible"):
        ProjectContext(
            project="assay",
            repo_url="https://example.invalid/assay.git",
            base_commit="0000000",
            branch="main",
            submodule_path=None,
            runners={},
            runner_parameters={"bad": {1, 2}},
        )


@pytest.mark.parametrize("invalid", [None, False, [], 0, ""])
def test_project_context_requires_runner_parameter_mapping(invalid):
    with pytest.raises(TypeError, match="must be a mapping"):
        ProjectContext(
            project="assay",
            repo_url="https://example.invalid/assay.git",
            base_commit="0000000",
            branch="main",
            submodule_path=None,
            runners={},
            runner_parameters=invalid,
        )


@pytest.mark.parametrize("rendered", ["null", "false", "[]", "0", "''"])
def test_yaml_requires_runner_parameter_mapping(rendered):
    yaml_text = f"""
schema_version: 1
project_context:
  project: assay
  repo_url: https://example.invalid/assay.git
  base_commit: "0000000"
  branch: main
  submodule_path: null
  runners: {{}}
  runner_parameters: {rendered}
intent: reject malformed runner parameters
constraints: []
done_criteria: []
"""
    with pytest.raises(ValueError, match="must be a mapping"):
        Spec.from_yaml(yaml_text)


def test_runner_parameters_reject_non_finite_numbers():
    with pytest.raises(ValueError, match="non-finite"):
        ProjectContext(
            project="assay",
            repo_url="https://example.invalid/assay.git",
            base_commit="0000000",
            branch="main",
            submodule_path=None,
            runners={},
            runner_parameters={"timeout": float("nan")},
        )


def test_assertion_criterion_is_registered_exported_and_roundtrips():
    yaml_text = """
schema_version: 1
project_context:
  project: assay
  repo_url: https://github.com/whact2025/assay.git
  base_commit: '0000000'
  branch: main
  submodule_path: null
  runners: {}
intent: Test an absolute postcondition.
constraints: []
done_criteria:
  - kind: assertion
    test_name: cargo_test_absolute_postcondition
    must_pass: true
"""
    spec = Spec.from_yaml(yaml_text)

    criterion = spec.done_criteria[0]
    assert isinstance(criterion, AssertionCriterion)
    assert criterion.test_name == "cargo_test_absolute_postcondition"
    assert criterion.must_pass is True

    restored = decode_dataclass(parse(serialize(spec)), Spec)
    assert restored == spec
    assert isinstance(restored.done_criteria[0], AssertionCriterion)


def test_activation_constraint_is_registered_exported_and_roundtrips():
    yaml_text = """
schema_version: 1
project_context:
  project: assay
  repo_url: https://github.com/whact2025/assay.git
  base_commit: '0000000'
  branch: main
  submodule_path: null
  runners: {}
intent: Park a future work unit.
constraints:
  - kind: activation
    status: parked
    reason: predecessor has not landed
done_criteria: []
"""
    spec = Spec.from_yaml(yaml_text)

    constraint = spec.constraints[0]
    assert isinstance(constraint, ActivationConstraint)
    assert constraint.status == "parked"
    assert constraint.reason == "predecessor has not landed"

    restored = decode_dataclass(parse(serialize(spec)), Spec)
    assert restored == spec
    assert isinstance(restored.constraints[0], ActivationConstraint)


def test_activation_constraint_rejects_unknown_status():
    with pytest.raises(ValueError, match="active.*parked"):
        ActivationConstraint(status="waiting", reason="not a valid state")


def test_spec_load_v0_preserves_canonical_depends_on(tmp_path):
    first = "sha256:" + "1" * 64
    second = "sha256:" + "a" * 64
    yaml_path = tmp_path / "spec.yaml"
    yaml_path.write_text(
        _strict_spec_yaml(
            top_level_extra=(
                "depends_on:\n"
                f'  - "{first}"\n'
                f'  - "{second}"\n'
            )
        ),
        encoding="utf-8",
    )

    spec = Spec.load_v0(yaml_path)

    assert spec.depends_on == (ContentHash(first), ContentHash(second))


@pytest.mark.parametrize(
    "depends_on_yaml,error",
    [
        ("depends_on: null\n", "depends_on must be a list"),
        (
            'depends_on: "sha256:' + "1" * 64 + '"\n',
            "depends_on must be a list",
        ),
        ("depends_on:\n  dependency: value\n", "depends_on must be a list"),
        ("depends_on:\n  - sha256:abc\n", "64 lowercase hex digits"),
        (
            'depends_on:\n  - "sha256:' + "A" * 64 + '"\n',
            "64 lowercase hex digits",
        ),
        ("depends_on:\n  - 42\n", "64 lowercase hex digits"),
    ],
)
def test_spec_from_yaml_rejects_malformed_depends_on(depends_on_yaml, error):
    with pytest.raises(ValueError, match=error):
        Spec.from_yaml(_strict_spec_yaml(top_level_extra=depends_on_yaml))


def test_spec_from_yaml_unknown_kind_raises():
    from assay import UnknownConstraintKind

    yaml_text = """
schema_version: 1
project_context:
  project: assay
  repo_url: https://github.com/whact2025/assay.git
  base_commit: 0000000
  branch: null
  submodule_path: null
  runners: {}
intent: t
constraints:
  - kind: not_a_real_kind
done_criteria: []
"""
    with pytest.raises(UnknownConstraintKind):
        Spec.from_yaml(yaml_text)


@pytest.mark.parametrize("schema_version", [0, 2, -1, "1", True, None])
def test_spec_from_yaml_rejects_unsupported_schema_version(schema_version):
    yaml_text = _strict_spec_yaml().replace(
        "schema_version: 1", f"schema_version: {schema_version!r}"
    )
    with pytest.raises(ValueError, match="unsupported Spec schema_version"):
        Spec.from_yaml(yaml_text)


def test_unsupported_schema_is_rejected_before_schema_specific_decoding():
    with pytest.raises(ValueError, match="unsupported Spec schema_version 2"):
        Spec.from_yaml("schema_version: 2\nlegacy_shape: true\n")


def test_direct_spec_construction_rejects_unsupported_schema_version():
    with pytest.raises(ValueError, match="unsupported Spec schema_version 2"):
        dataclasses.replace(_v0_spec(), schema_version=2)


def _strict_spec_yaml(
    *,
    top_level_extra: str = "",
    project_context_extra: str = "",
    constraint_extra: str = "",
    criterion_extra: str = "",
    plan_policy_extra: str = "",
) -> str:
    return f"""
schema_version: 1
project_context:
  project: assay
  repo_url: https://github.com/whact2025/assay.git
  base_commit: 0000000
  branch: null
  submodule_path: null
  runners: {{}}
{project_context_extra}intent: t
constraints:
  - kind: budget
    usd_limit: 10
{constraint_extra}done_criteria:
  - kind: parity
    reference_commit: abc
    test_name: t
{criterion_extra}plan_policy:
  max_critique_rounds: 2
  judge_required: true
  breaker_required: false
  persistence_veto_threshold: 3
{plan_policy_extra}{top_level_extra}"""


@pytest.mark.parametrize(
    "yaml_text,label",
    [
        (_strict_spec_yaml(top_level_extra="unexpected: true\n"), "Spec YAML"),
        (
            _strict_spec_yaml(project_context_extra="  unexpected: true\n"),
            "project_context",
        ),
        (
            _strict_spec_yaml(constraint_extra="    unexpected: true\n"),
            "constraint kind 'budget'",
        ),
        (
            _strict_spec_yaml(criterion_extra="    unexpected: true\n"),
            "criterion kind 'parity'",
        ),
        (
            _strict_spec_yaml(plan_policy_extra="  unexpected: true\n"),
            "plan_policy",
        ),
    ],
)
def test_spec_from_yaml_rejects_unknown_fields(yaml_text, label):
    with pytest.raises(ValueError, match=f"{label} has unknown fields"):
        Spec.from_yaml(yaml_text)


def _conflict_slice(sid, files, creates=()):
    from assay import ImplementationDirective, Slice
    return Slice(id=sid, directive=ImplementationDirective(
        slice_id=sid, intent="t", files_in_scope=tuple(files),
        creates_paths=tuple(creates), verify_mode="build_only", blast_radius={},
    ))


def test_tierdag_parallel_conflicts_detects_same_tier_overlap():
    """APLAN-2: same-tier slices sharing a write path are flagged; the
    files_in_scope ∪ creates_paths union is what's compared."""
    from assay import Tier, TierDAG
    dag = TierDAG(tiers=(
        Tier(tier_id="t0", depends_on=(), slices=(
            _conflict_slice("s0", ["a.py", "shared.py"]),
            _conflict_slice("s1", ["shared.py", "b.py"]),
        )),
        Tier(tier_id="t1", depends_on=("t0",), slices=(
            _conflict_slice("s2", ["c.py"]),
            _conflict_slice("s3", [], creates=["d.py"]),
        )),
    ))
    conflicts = dag.parallel_conflicts()
    assert len(conflicts) == 1
    tier_id, a, b, overlap = conflicts[0]
    assert tier_id == "t0"
    assert {a, b} == {"s0", "s1"}
    assert overlap == ("shared.py",)


def test_tierdag_parallel_conflicts_counts_creates_vs_scope_overlap():
    """A slice that CREATES a path another slice edits is also a conflict."""
    from assay import Tier, TierDAG
    dag = TierDAG(tiers=(
        Tier(tier_id="t0", depends_on=(), slices=(
            _conflict_slice("s0", ["gen.py"]),
            _conflict_slice("s1", [], creates=["gen.py"]),
        )),
    ))
    assert len(dag.parallel_conflicts()) == 1


def test_tierdag_parallel_conflicts_empty_when_disjoint():
    from assay import Tier, TierDAG
    dag = TierDAG(tiers=(
        Tier(tier_id="t0", depends_on=(), slices=(
            _conflict_slice("s0", ["a.py"]),
            _conflict_slice("s1", ["b.py"]),
        )),
    ))
    assert dag.parallel_conflicts() == []
