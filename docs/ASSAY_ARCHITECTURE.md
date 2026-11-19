<!--
SPDX-FileCopyrightText: 2026 Whact Inc.
SPDX-License-Identifier: MIT
-->

# Assay architecture

This document describes the architecture implemented by the current Assay
source tree. Source and contract tests are authoritative when this reference
and code disagree.

See the [README](../README.md) for a guided introduction and the
[public API reference](PUBLIC_API.md) for concise import and call-boundary
semantics.

Assay is a library of typed primitives for evidence-backed, Spec-driven
engineering. It provides:

- immutable records for intent, plans, evidence, invocations, and outcomes;
- deterministic JSON bytes and SHA-256 content identities;
- a local Corpus for Spec lineages, typed records, artifacts, measurements,
  and run journals;
- minimal backend and tool protocols;
- eight callable primitives: `plan`, `review`, `validate`, `implement`,
  `bench`, `sweep`, `parallel`, and `smoke_check`.

Assay is not a workflow engine. A caller chooses the backend, sequences gates,
owns retries and rollback, publishes returned records, and decides what runs
next.

---

## 1. System boundary

### 1.1 Package responsibilities

| Package area | Implemented responsibility |
|---|---|
| `assay.types` | Public records, policies, outcomes, protocols, and the `Spec` lineage value. |
| `assay.canonical` | Deterministic JSON conversion, parsing, hashing, and kind registries. |
| `assay.corpus` | Filesystem/SQLite persistence, strict record hydration, transactions, and run journals. |
| `assay.backend` | Runtime-checkable `Backend` and `ChatToolsBackend` protocols. |
| `assay.primitives` | The eight callable operations plus Git capture, response repair policy, and tool-call cache. |
| `assay.bootstrap` | Idempotent registration of built-in constraint and criterion types. |
| `assay._safety` | Explicit byte-oriented secret and terminal-control redaction. |

The top-level `assay` package re-exports selected types, canonical helpers,
backend protocols, the registries, `redact_secrets`, and the tool cache. The
eight operations are imported from `assay.primitives`; they are not exported
from `assay`. Importing the tool-cache submodule initializes the
`assay.primitives` package under normal Python package semantics, so this is an
API namespace distinction rather than a promise that primitive modules remain
unloaded.

### 1.2 External ownership

Assay's public design separates mechanics from product policy. An integrating
package owns:

- end-to-end workflows and their state transitions;
- concrete backend adapters and model selection;
- review rubric content;
- deterministic validation-rule implementations;
- build, test, verification, and benchmark runners;
- workspace creation, merge, rollback, and process containment;
- human interfaces, dashboards, and notifications.

Assay has no static import of an integrating workflow package. Three current
defaults nevertheless retain integration-specific vocabulary:

- without a custom `rule_resolver`, `validate` dynamically imports
  `crucible.rules.<rule_id>`;
- `MutationRegion` defaults to `CRUCIBLE-MUTATE-START` and
  `CRUCIBLE-MUTATE-END` markers;
- the built-in planner and implementer prompts contain Crucible-oriented
  wording and a concrete project-layout example.

A product-neutral validator supplies `rule_resolver`; callers may override
mutation markers and `implement(system_prompt=...)`. `plan` currently exposes
no system-prompt override. These are implemented defaults, not architectural
requirements for extensions.

### 1.3 Architectural invariants

1. A `Spec` is the durable lifecycle root; large outputs are referenced by
   content hash rather than embedded.
2. Approval is never inferred from truthiness. Review has four decisions and
   planning has three terminal decisions.
3. Model output is untrusted. Structured output, source coverage, and evidence
   are checked by the harness before a verdict can pass.
4. Corpus publication is explicit. A primitive may persist supporting
   artifacts, but callers publish returned typed records and Spec manifests
   unless the primitive contract says otherwise.
5. Concurrency is bounded and opt-in. Ordered fan-out does not imply workflow
   scheduling, workspace isolation, cancellation, or partial-success policy.

---

## 2. Data model and immutability

### 2.1 What “immutable” means

Public record classes are frozen dataclasses unless stated otherwise. Frozen
dataclasses prevent attribute reassignment. Mapping-bearing records also call
`freeze_dict` for the fields listed below; this recursively copies mappings to
`FrozenDict`, lists/tuples to tuples, and sets to frozensets:

- `ProjectContext.runners` and `ProjectContext.runner_parameters`;
- `ExecutionContext.toolchain`;
- `ImplementationDirective.blast_radius`;
- `Implementation.artifacts`;
- `Measurement.values`;
- `BenchConfig.parameters` and `BenchConfig.env`;
- `Tool.input_schema`;
- `Verdict.artifacts`.

This is not universal runtime coercion. Tuple annotations are not automatically
validated by ordinary dataclass construction, `Invocation.response_schema` is
not deep-frozen, and extension objects satisfying `Constraint` or `Criterion`
can carry their own mutability. The Corpus codec is the strict type boundary
for persisted records.

`Corpus` and `ToolCallCache` are intentionally mutable. `RunHandle` is a frozen
handle whose methods mutate files under its run directory.

### 2.2 Identity and execution context

`ContentHash` is `typing.NewType("ContentHash", str)`. At runtime it is a
string, not an `isinstance`-checkable class. Canonical Corpus/YAML boundaries
validate the exact form:

```text
sha256:<64 lowercase hexadecimal characters>
```

Constructing `ContentHash(value)` alone performs no validation.

| Type | Fields and meaning |
|---|---|
| `ProjectContext` | `project`, `repo_url`, `base_commit`, optional `branch`, optional `submodule_path`, runner-role mapping `runners`, and JSON-shaped `runner_parameters`. Runner parameters reject non-string map keys, unsupported objects, and non-finite floats. |
| `ExecutionContext` | `timestamp`, `machine`, `cwd`, `toolchain`, optional `backend`, `model`, `recipe`, `recipe_commit`, `gpu_model`, and `run_id`. |

`ExecutionContext` is provenance data, not a validated clock or platform type.
The dataclass accepts arbitrary strings. Assay producers normally generate UTC
ISO strings, but there is no type-level timestamp-format, timezone, hostname,
path, commit, or tool-version invariant.

### 2.3 Constraints, criteria, and registries

`Constraint` and `Criterion` are runtime-checkable marker protocols with one
required attribute, `kind: str`.

Built-in constraints:

| Kind | Type | Fields |
|---|---|---|
| `budget` | `BudgetConstraint` | `usd_limit`, `time_limit_hours` |
| `blast_radius` | `BlastRadiusConstraint` | `no_edits_outside` |
| `activation` | `ActivationConstraint` | `status`, `reason`; status must be `active` or `parked` |

Built-in criteria:

| Kind | Type | Fields |
|---|---|---|
| `assertion` | `AssertionCriterion` | `test_name`, `must_pass` |
| `parity` | `ParityCriterion` | `reference_commit`, `test_name` |

`assay.bootstrap.populate_registry()` inserts these classes with `setdefault`
into `CONSTRAINT_TYPES` and `CRITERION_TYPES`. Importing `assay.bootstrap`
calls it once; `Spec.from_yaml` and `Corpus` also ensure built-ins are
registered. An external package may register another frozen dataclass under a
new `kind`. Registry mutation is process-global, so registration belongs at
controlled application bootstrap, before concurrent decoding starts.

### 2.4 The Spec

```python
@dataclass(frozen=True)
class Spec:
    version: int
    parent: ContentHash | None
    root: ContentHash | None
    schema_version: int
    project_context: ProjectContext
    execution: ExecutionContext | None
    intent: str
    constraints: tuple[Constraint, ...]
    done_criteria: tuple[Criterion, ...]
    plan_policy: PlanPolicy | None
    depends_on: tuple[ContentHash, ...] = ()
    tier_dag: TierDAG | None = None
    implementation_refs: tuple[ContentHash, ...] = ()
    verdict_refs: tuple[ContentHash, ...] = ()
    measurement_refs: tuple[ContentHash, ...] = ()
    lifecycle_event_refs: tuple[ContentHash, ...] = ()
```

The only accepted `schema_version` is integer `1`; booleans and other numeric
representations are rejected. This exact-match check applies to direct
construction, YAML ingestion, and Corpus hydration.

A v0 Spec has `version=0`, `parent=None`, `root=None`, and normally
`execution=None`. On the first transition, `parent` and `root` both become the
canonical hash of v0. Later transitions set `parent` to the immediate prior
Spec hash and retain the same root.

Pure helpers create new values:

- `with_tier_dag(dag)` replaces inline plan data and advances one version;
- `with_implementation`, `with_verdict`, `with_measurement`, and
  `with_lifecycle_event` hash and append one record reference;
- `with_measurements(refs)` appends many measurement refs with one version
  advance and is a no-op for an empty iterable;
- `with_appended_refs(...)` appends multiple record classes and can replace
  `tier_dag` with one version advance.

These helpers do not persist their record arguments. A caller must store a
record in the matching Corpus namespace before publishing a Spec that refers
to it.

The Corpus enforces contiguous lineage and append-only lifecycle reference
tuples. It does not compare every other field with the prior version. Normal
helpers preserve authored fields, but the persistence layer does not itself
forbid a correctly parented caller from replacing `intent`, constraints,
project context, dependencies, execution context, plan policy, or inline DAG.
Any stronger authority rule belongs in the publishing workflow.

There is no `phase`, `status`, `next_action`, or mutable state-machine field.
Lifecycle meaning comes from populated data and referenced `LifecycleEvent`
records. An `ActivationConstraint(status="parked")` is special: it can exist as
a value, but `Corpus.put_spec` rejects publication while it remains parked.

### 2.5 Plan and directive types

| Type | Fields / behavior |
|---|---|
| `MutationRegion` | `file`, `start_marker`, `end_marker`; default markers are `CRUCIBLE-MUTATE-START` and `CRUCIBLE-MUTATE-END`. |
| `ImplementationDirective` | `slice_id`, generated `intent`, `files_in_scope`, `creates_paths`, `verify_mode`, `blast_radius`, optional `mutation_regions`, exact root `requirement_refs`, and optional `read_only_references` that never grant write authority. |
| `ReadOnlySourceReference` | Stable `reference_id`, normalized relative POSIX `path`, whole-file `sha256`, and inclusive positive integer `start_line`/`end_line`; a locator, not an attestation of source verification. |
| `Slice` | `id`, `directive`. |
| `Tier` | `tier_id`, sibling `slices`, `depends_on`, optional `smoke_bench_config`. |
| `TierDAG` | Ordered tuple of tiers plus `topo_order`, `all_slices`, and `parallel_conflicts`. |

`TierDAG.topo_order()` implements Kahn ordering and raises for unknown
dependencies or cycles. Duplicate tier IDs also raise, but only through the
generic `ValueError("cycle detected")` path rather than a dedicated duplicate-ID
diagnostic. The plan primitive supplies additional parsing and policy gates.

`parallel_conflicts()` reports exact path-string overlaps between
`files_in_scope ∪ creates_paths` for sibling slices in one tier. It does not
resolve symlinks, directory containment, glob overlap, case folding, or
semantic conflicts.

`MutationRegion` is data and prompt material in Assay. The `implement`
primitive does not inspect diff hunks against those markers; deterministic
region enforcement is an external workflow responsibility.

### 2.6 Evidence, findings, and verdicts

| Type | Contract |
|---|---|
| `Evidence` | `kind`, `location`, exact `excerpt`, and stable `source_id` (default `artifact` for deterministic callers). |
| `Proposal` | Generated remediation: `action`, optional `replacement`. It is not evidence. |
| `AttackSpec` | Declarative `setup`, `stimulus`, and `oracle`. It is not executable code or evidence. |
| `ReviewCoverage` | `complete`, `sources_reviewed`, `omissions`, and harness-computed artifact character totals. |
| `CriterionResult` | One criterion ID, outcome, reasoning, evidence, and optional headline flag. |
| `Finding` | Stable ID, severity, category, evidence, action, source, persistence/suppression metadata, criterion ID, confidence, disposition, targets, proposal, and attack spec. |
| `Verdict` | Four-way decision, findings, input identity, coverage, backend/transcript/cost/usage data, output artifact refs, role kind, rubric/prompt/model provenance, and criterion results. |

Review decisions are exactly `pass`, `revise`, `reject`, and `inconclusive`.
`Verdict.passed` is only `decision == "pass"`; it must not replace the full
decision at a workflow boundary.

Finding dispositions are `confirmed`, `suspected`, `dismissed`, `addressed`,
and `unfixable`. Criterion outcomes are `satisfied`, `violated`,
`inconclusive`, and `not_applicable`.

The public typing aliases for those closed strings are `ReviewDecision`,
`FindingDisposition`, and `CriterionOutcome`; their runtime membership sets
are `REVIEW_DECISIONS` and `FINDING_DISPOSITIONS`. Planning similarly exports
`PlanDecision` for `accepted | rejected | inconclusive`.

`Finding` uses `Literal` annotations but does not validate those strings at
ordinary construction. `Verdict` runtime-validates its own `decision` and
`kind`, not every nested Finding field. The strict review parser validates
model-authored findings; deterministic producers are responsible for emitting
the declared shape (and `validate` conservatively blocks unknown severities).

Verdict kinds are runtime-validated:

- advisory: `critic`, `judge`, `inflation`;
- gate: `validate`, `breach_judge`, `review`.

The exported sets are `ADVISORY_KINDS`, `GATE_KINDS`, and their union
`ALL_VERDICT_KINDS`. This taxonomy classifies roles; it does not execute a
workflow gate. Consumers still interpret each decision in context.

### 2.7 Artifact and outcome records

| Type | Fields / identity |
|---|---|
| `Implementation` | Slice/base/result commits, diff/transcript refs, modified files, classification, execution, scope escapes, attached artifact refs, cost, tokens, optional ordered pricing evidence, and non-normal stop reason. |
| `Measurement` | `bench_id`, `commit`, semantic `config_hash`, numeric values, raw-output ref, and execution. |
| `LifecycleEvent` | Free-form `kind`, `reason`, affected Spec versions, and execution. |
| `SmokeResult` | `passed`, selected language, bounded log excerpt, duration in milliseconds, and `infrastructure_failure` for a failed check that produced no verdict on the code. |
| `PlanOutcome` | Returned Spec, diagnostic candidate and ref, terminal plan decision/status, round decisions, reason, aggregate inclusive usage, execution completeness, and ordered pricing evidence. |

Plan decisions are `accepted`, `rejected`, and `inconclusive`.
`PlanOutcome.accepted` is the corresponding convenience projection.

`LifecycleEvent.kind`, `Implementation.classification`, and
`PlanOutcome.status` are strings rather than closed enums. The producing
primitive/workflow defines their vocabulary.

### 2.8 Invocation, tools, and policies

The remaining frozen records are:

- `Tool(name, description, input_schema)`;
- `Invocation(system, user, tools, max_turns, budget_usd, policy,
  response_schema)`;
- `Usage(input_tokens, output_tokens, cost_usd, cache_creation_tokens,
  cache_read_tokens)`, where cache counters are non-overlapping subsets of
  total inclusive input;
- `UsagePricingSegment(input_tokens, output_tokens, cache_creation_tokens,
  cache_read_tokens, cost_usd)` and
  `UsagePricingEvidence(segments, complete)` for ordered, prompt-free physical
  response pricing inputs;
- `Message(role, content)`;
- `InvocationResult(final_message, transcript, usage, stop_reason, error,
  completeness, execution_trace, conversation, pricing_evidence)`;
- `ExecutionCompleteness`, `BackendExecutionTrace`, and
  `BackendExecutionEvent` for redacted terminal evidence. The optional
  `BackendExecutionTrace.tool_authority_digest` binds a run to the canonical,
  secret-free ToolSet authority advertised by its backend;
- the five backend-error records described in §7;
- `BenchConfig(bench_id, runner, parameters, env, timeout_seconds,
  working_dir, commit)`;
- `ValidationRuleset(name, rule_ids)` and
  `RuleContext(workspace_root, submodule_path)`;
- `Rubric`, `PlanPolicy`, and `BackendPolicy` described in §6;
- `ReviewResponseRepairPolicy` described in §8.2;
- Corpus transaction/result types described in §5.

`Invocation.system`, `Invocation.user`, and `Message.content` are excluded from
`repr` to reduce incidental prompt leakage. That is not storage redaction;
transcripts still contain the message content.

The additive cache fields and optional pricing evidence are omitted from
canonical JSON at their defaults, so pre-feature records retain their bytes
and hashes. A complete `UsagePricingEvidence` must reconcile every aggregate
token counter and cost and requires complete token/cost execution evidence.
Incomplete evidence may retain known segments in relative order without
claiming that they account for the whole logical invocation. `Implementation`
and `Verdict` carry that ledger durably. `Verdict` additionally carries the
authoritative aggregate input, output, cache-creation, and cache-read counters
used by planning. Exact zero cache counters are distinct from absent counters
on legacy records. A legacy record may recover cache totals from a complete
ledger; absent counters with an incomplete ledger preserve token-accounting
debt. Consumers must not sum an incomplete ledger to reconstruct aggregates:
a backend may expose an authoritative terminal total alongside cumulative,
repeated, or hidden intermediate response snapshots.
Multi-call review concatenates segments in provider-call order and marks the
ledger complete only when every contributor is complete and the segments
reconcile with the authoritative aggregate.

### 2.9 Provenance by producer

Provenance is explicit and differs by operation:

| Producer | Provenance written by Assay |
|---|---|
| `validate` | `artifact_commit` from `Spec.project_context.base_commit`; deterministic artifact hash; no backend, transcript, usage, or execution context. |
| `review` | Input/rubric/rendered-prompt/transcript refs, backend and model metadata, capability/effort when exposed by the adapter, tokens and cost; `execution=None`. |
| `implement` | Base Git commit, diff/transcript refs, backend/model, process timestamp/host/cwd/toolchain, tokens/cost, and stop reason; `result_commit=None`. |
| `bench` | Bench/config/commit/raw-output identities and process timestamp/host/cwd/toolchain; backend/model are null. |
| `plan` | Per-call transcripts and review verdicts plus aggregate reported cost; terminal lifecycle execution uses recipe `plan`, an empty cwd, host/time, and Python version. |
| `RunHandle` | Run directory, recipe, start timestamp, elapsed monotonic duration, events, summary, and terminal result; it is a journal separate from typed record execution fields. |

Because `ExecutionContext` is part of an `Implementation` or `Measurement`,
those record hashes normally differ across executions even when semantic work
or KPI values match. The deterministic `validate` Verdict deliberately omits
execution context so identical inputs and rule outputs hash identically.

---

## 3. Canonical bytes and content identity

### 3.1 Exact serializer behavior

`assay.canonical.serialize` implements Assay's deterministic Python JSON
profile. The precise pipeline is:

1. Accept `None`, booleans, integers, finite floats, strings, dataclass
   instances, lists/tuples, and dicts with string keys.
2. Expand dataclasses by declared field order. A field carrying
   `canonical_omit_empty` metadata is omitted when its value is falsey.
3. Normalize string values and dict keys to Unicode NFC.
4. Format floats with `.17g`, parse the result back to `float`, and reject NaN
   and infinity.
5. Convert tuples and lists to JSON arrays.
6. Call Python `json.dumps` with `ensure_ascii=False`, `sort_keys=True`,
   `separators=(",", ":")`, and `allow_nan=False`.
7. Encode the result as UTF-8 with no trailing newline.

This is the implemented identity contract. It must not be described as full
RFC 8785/JCS compliance: number rendering and key ordering are Python's JSON
behavior, not ECMAScript's normative JCS algorithms.

Unsupported objects and non-string mapping keys raise. Callers must also avoid
distinct keys that normalize to the same NFC spelling; the serializer builds a
normalized dict and does not provide a separate collision diagnostic.

`parse(blob)` accepts `bytes` or `bytearray`, decodes UTF-8, and delegates to
`json.loads`. It is a JSON parser, not a canonical-input validator. Therefore
`parse(noncanonical_json)` may succeed; canonical identity always comes from
`serialize`/`hash_record`.

`hash_record(value)` is:

```text
"sha256:" + sha256(serialize(value)).hexdigest()
```

It returns a plain string so canonical code does not depend on the
`ContentHash` type module.

### 3.2 Persist-optional fields

Selected optional record fields carry two independent metadata flags:

- `persist_optional`: the strict decoder may supply the dataclass default when
  the field is absent;
- `canonical_omit_empty`: the serializer omits a falsey default from identity.

All other dataclass fields are required during hydration. Unknown fields are
always rejected. This is a narrow schema declaration on individual fields,
not a general migration or best-effort compatibility layer.

### 3.3 Strict Corpus hydration

Canonical JSON loses tuple, dataclass, `Path`, protocol-union, and `NewType`
information. `assay.corpus.codec.decode_dataclass` reconstructs the declared
graph recursively and checks:

- exact dataclass field sets, subject only to `persist_optional` defaults;
- tuple/list/mapping/union/literal shapes;
- strict booleans and integers;
- numeric values for float annotations, excluding booleans;
- full `ContentHash` syntax;
- `Constraint`, `Criterion`, and `BackendError` discriminators;
- constructor-level validation after hydration.

`RecordDecodeError` reports invalid persisted data. There is no automatic
schema migration, field dropping, or unknown-kind fallback.

---

## 4. Spec authoring

### 4.1 YAML surface

`Spec.from_yaml` accepts only these top-level keys:

```yaml
schema_version: 1
project_context:
  project: example
  repo_url: https://example.invalid/repo.git
  base_commit: 0123456789abcdef
  branch: main
  submodule_path: null
  runners:
    benchmark: example.runners.benchmark
  runner_parameters:
    languages: [python]
intent: |
  Implement the stated behavior.
constraints:
  - kind: budget
    usd_limit: 25.0
    time_limit_hours: 2.0
done_criteria:
  - kind: assertion
    test_name: test_behavior
    must_pass: true
depends_on:
  - sha256:0000000000000000000000000000000000000000000000000000000000000000
plan_policy:
  max_critique_rounds: 2
  judge_required: true
  breaker_required: false
  persistence_veto_threshold: 3
```

Allowed `project_context` keys are `project`, `repo_url`, `base_commit`,
`branch`, `submodule_path`, `runners`, and `runner_parameters`. Unknown fields
at either level are rejected.

Constraint and criterion entries must be mappings with a registered `kind`.
Their remaining keys must exactly match init fields of the registered
dataclass. Missing/unknown kinds raise `UnknownConstraintKind`.

`depends_on`, when present, must be a list of canonical hash strings. These are
cross-Spec addresses; the local Corpus validates their syntax but does not
require the target to exist locally.

A non-empty `plan_policy` mapping must supply the four core fields shown above.
Optional policy keys take dataclass defaults when absent. An absent or empty
mapping results in `Spec.plan_policy is None`.

YAML creates v0 only: no execution context, DAG, lifecycle records, or output
refs can be authored through this loader. `Spec.load_v0(path)` reads UTF-8 and
delegates to `from_yaml`.

### 4.2 Paths are not identity

Assay does not implement a workspace-wide Spec discovery service. A caller
chooses one directory and constructs `Corpus(spec_dir)`. A conventional
`spec.yaml` may live there, but `Corpus` neither requires nor reads it
automatically. Moving an unpublished authored directory does not alter
canonical Spec bytes; moving runtime storage changes where its local artifacts
are found.

---

## 5. Corpus and run persistence

### 5.1 Physical layout

The Corpus uses the following layout:

```text
<spec-dir>/
  versions/
    .spec-write.lock
    v0_<12-hex>.json
    v1_<12-hex>.json
  artifacts/
    <caller-kind>/
      <64-hex>.bin
  transcripts/
  diffs/
  traces/
  runs/
    <UTC>_<recipe>_<8-hex>/
      .run-write.lock
      events.jsonl
      summary.json
      result.json
    idempotent_<recipe>_<24-hex>/
      run.json
      ...
  measurements.sqlite
  measurements.sqlite-wal
  measurements.sqlite-shm
```

The constructor creates only the six top-level directories `versions`,
`artifacts`, `transcripts`, `diffs`, `traces`, and `runs`. Manifests, namespace
directories, lock files, run contents, and SQLite files appear lazily when the
corresponding APIs are used. WAL/SHM presence depends on connection state.

For a controlled, single-component `kind`, generic `put_artifact` writes go
under `artifacts/<kind>/`, including kinds such as `transcripts`, `diffs`, and
`traces`. The top-level `transcripts/`,
`diffs/`, and `traces/` directories are created by the current layout but are
not destinations used by `put_artifact`. Callers must use the API rather than
infer an artifact path from those reserved directories.

### 5.2 Raw artifacts and typed records

`put_artifact(blob, kind, name_hint)` hashes raw bytes and atomically writes
`artifacts/<kind>/<hex>.bin`. `name_hint` does not affect path or identity.
Equal bytes in the same namespace are idempotent. The returned ref addresses
the bytes, not metadata about their logical name.

`kind` is a trusted namespace input, not an untrusted filename. The current
method does not reject separators or parent components before joining it under
`artifacts/`; callers must pass one controlled path component. Supplying
model- or user-authored path text as `kind` can escape the intended namespace
and is outside the API's safety contract.

`get_artifact(ref)` searches artifact namespaces for a matching stem, ignores
temporary files, re-hashes candidate bytes, and returns only an exact match.
Corrupt or planted candidates are skipped. Because the lookup is
namespace-agnostic, use `get_record` when record kind matters.

`get_artifact_bounded(ref, max_bytes=...)` provides the bounded equivalent for
small authority and control-plane records. It walks from a held artifact-root
descriptor without following aliases, accepts only stable single-link regular
files, enforces fixed traversal depth and entry ceilings, and checks the byte
limit from metadata and again while streaming the SHA-256. It never retains a
complete over-limit candidate in memory. A same-stem over-limit decoy is
skipped when another namespace contains a valid bounded copy.

`list_artifacts(kind=...)` yields de-duplicated refs for filenames with a
canonical 64-lowercase-hex stem in that namespace; it ignores temporary and
malformed names but does not re-hash each listed file. For kind
`measurements`, it delegates to the SQLite ref index.

Typed lifecycle records have exactly four plural namespaces:

| Kind | Required concrete type | Storage |
|---|---|---|
| `implementations` | `Implementation` | canonical bytes under `artifacts/implementations/` |
| `verdicts` | `Verdict` | canonical bytes under `artifacts/verdicts/` |
| `measurements` | `Measurement` | row in `measurements.sqlite` |
| `lifecycle_events` | `LifecycleEvent` | canonical bytes under `artifacts/lifecycle_events/` |

`put_record` validates the live dataclass by serializing and strictly decoding
it before storage. Kind/type mismatches and malformed nested hashes fail.
`get_record(ref, as_type=...)` chooses the namespace from the exact requested
class and performs strict hydration. Arbitrary dataclasses are not accepted as
lifecycle records.

Raw support material—prompts, diffs, logs, rubric snapshots, reviewed inputs,
and similar bytes—uses `put_artifact` and is referenced from a typed record
where appropriate.

### 5.3 Spec manifests and lineage checks

`put_spec(spec)` validates the complete dataclass graph, activation state,
lineage, and references before an atomic manifest rename.

For the first manifest it requires v0 with null parent/root. Each subsequent
manifest must be exactly `latest.version + 1`, its parent must hash to the
latest Spec, and its root must match the lineage root. Lifecycle ref tuples
must retain the complete prior prefix. Every referenced record must exist in
the correct namespace and match its address. Cross-Spec `depends_on` entries
receive syntax validation only.

The manifest name includes the canonical decimal version and exactly the first
12 hex characters of its persisted canonical content hash. Rewriting identical
bytes is idempotent. A different manifest at an already occupied version raises
`ConcurrentSpecError`; readers also raise if divergent same-version files are
present rather than choosing by mtime. Reads use a bounded no-follow descriptor,
require one singly linked regular file, prove stable metadata, canonical raw
JSON, and the exact content-derived filename, then hydrate it. Hashing the raw
canonical shape before hydration preserves compatibility with older manifests
that legitimately omit a subsequently added optional field.
`get_spec_with_digest()` exposes the safely loaded Spec and that exact raw
SHA-256 as one pair for downstream lineage/CAS authority; consumers must not
substitute `hash_record(hydrated_spec)` for persisted identity.

`latest_spec()` returns `None` when no manifest exists. Otherwise it validates
the complete contiguous v0-through-tip history, authenticated parent/root
edges, and every append-only reference prefix before returning the tip. New
transaction commits use the persisted predecessor digest. For legacy optional
field compatibility, `put_spec` recognizes the exact deterministic parent/root
produced by a public `Spec.with_*` helper and normalizes it to the persisted
predecessor digest before serializing the new manifest. A caller using that
legacy direct path must reload the published version before deriving another
child; `Corpus.commit` returns the normalized published Spec.
Historical child manifests that already name only a rehydrated, non-persisted
predecessor representation are rejected as dangling provenance rather than
silently treated as exact lineage.
Malformed manifest-like filenames, missing or divergent historical versions,
and a replaced constructor-bound `versions/` directory fail closed.

### 5.4 Transactional publication

The workflow-facing API is:

```text
Corpus.commit(
    *,
    implementations=(),
    verdicts=(),
    measurements=(),
    lifecycle_events=(),
    lifecycle_event_factory=None,
    record_refs=None,
    transform=None,
) -> CommitResult
```

`CommitRefs` groups the four ref tuples. `CommitResult` returns the published
Spec and the transaction's refs.

`commit` holds the Spec write lock while it:

1. reads the latest Spec;
2. stores new typed records and checks supplied refs;
3. de-duplicates refs already reachable from the latest Spec;
4. advances the lineage once;
5. optionally applies `transform` to that already-versioned candidate;
6. checks that the transform retained every staged ref;
7. atomically publishes the manifest.

The transform may replace inline structure but must return a `Spec` preserving
valid lineage and staged refs. If there are no new refs and no transform,
`commit` returns the current Spec without creating a version.

`lifecycle_event_factory(next_version)` runs inside the lock so its
`affected_versions` can use the actual publication version.
`commit_lifecycle_event(factory)` is the focused form and returns both the
commit result and event.

A crash can leave an unreachable content-addressed record or SQLite row, but
the manifest is the visibility point and is never published with a missing
record.

### 5.5 Measurement index

Measurements use a SQLite table keyed by canonical record hash. Writes validate
the typed record; reads reconstruct it and verify that the row hashes back to
its stored ref.

Available operations include:

- `put_measurements` for one bulk transaction;
- `get_measurement` and `list_measurement_refs`;
- `iter_measurements`, filtered by bench ID, commit, config hash, and/or an
  ordered ref list;
- `bench_id_summary` and `distinct_machines`;
- `bench_id_version_series` for within-version metric means;
- `bench_machine_version_sums` for per-benchmark/machine/version sums and
  counts.

Ref-scoped query methods chunk large SQL `IN` sets and preserve/refold results
according to their documented return contract. Metric keys accepted by the two
series APIs are filtered to `[A-Za-z0-9_.]+` before interpolation into JSON SQL
paths; if none remain, those APIs return an empty result.

`Corpus` is a context manager; `close()` idempotently closes its SQLite
connection.

### 5.6 Run journals

`open_run(recipe, idempotency_key=None)` validates recipe names against
`[A-Za-z0-9][A-Za-z0-9_.-]{0,127}` and creates a mode-0700 directory. A normal
run uses a UTC timestamp and random 32-bit tag with bounded collision retries.
An idempotency key deterministically selects a 24-hex digest directory and
validates its create-once `run.json` metadata.

`RunHandle` provides:

- `write_event(dict)`: locked, fsynced JSONL append;
- `write_summary(value)`: locked atomic replacement before finalization;
- `finalize(RunResult)`: create-once immutable `result.json`; an identical
  repeat is idempotent and a conflicting repeat raises
  `RunAlreadyFinalizedError`;
- `finalize_elapsed(RunResult)`: require the default-zero duration, snapshot
  finite non-negative monotonic elapsed time once, and return the recorded
  terminal result; retries compare the non-duration terminal fields;
- `finalized` and monotonic `elapsed_s` properties;
- context-manager finalization as `completed` or `failed` without suppressing
  the original exception.

After finalization, events and summaries are immutable. Event, summary, result,
and lock writes are serialized with an in-process lock plus `flock` and use
private file modes where the API creates data files.

Assay does not automatically create `command.txt`, environment captures,
stdout/stderr files, or build/test logs. A workflow may add those files, but
they are not part of the `RunHandle` contract.

The current Corpus/run implementation imports `fcntl` unconditionally and uses
`flock`; it therefore requires a POSIX platform. The root package's current
import graph reaches Corpus code through `assay.primitives.bench`, so ordinary
`import assay` has the same platform requirement. The package does not provide
a Windows lock fallback.

### 5.7 Locking and atomicity

Spec writes use a module-level reentrant thread lock plus a per-Corpus
`versions/.spec-write.lock` advisory `flock`. The in-process lock is global to
the module, so separate Corpus instances in one process serialize Spec writes.
The file lock serializes cooperating processes for a given path.
`Corpus.locked_spec_authority()` exposes that lock for a bounded external
side effect which must remain authorized by one exact latest Spec. The guard
is thread- and fork-aware: a same-thread nested Spec write raises instead of
deadlocking, while a different thread or process blocks normally. It does not
authorize a Corpus write inside the context; callers release the guard and use
the ordinary commit/CAS path to publish resulting evidence.
`Corpus.retained_spec_authority()` is the composition-safe form: its owner
publishes a source acquisition lease before returning the immutable snapshot,
and callers pre-register `retry_cleanup` before invoking `__enter__`. This
closes the asynchronous callee-to-caller return gap that ordinary `with`
bytecode cannot own. Partial or interrupted release raises
`SpecAuthorityCleanupError` with the exact retained owner for explicit retry.
The retained path fails closed before acquisition if the runtime cannot prove
current-thread ownership of its native reentrant lock. Reentrant retained
acquisition also requires an exact recursion-count probe; the ordinary guard
remains the compatible cross-Corpus nesting surface when that probe is absent.
A fork from another thread waits for the bounded Spec-lock region to settle.
The child then receives fresh process-local thread locks and an empty
same-thread guard registry. If it unwinds a same-thread authority inherited
from its parent, it closes only the child's duplicate descriptor and abandons
that copied lease without issuing `LOCK_UN` against the parent's live open
description.

Atomic files are written to same-directory temporary files, flushed and
`fsync`ed, then installed with `os.replace`; directory fsync is attempted where
supported. Measurement connection creation, writes, and single-record lookup
use the connection lock; iterator and aggregate reads use the shared SQLite
connection without that Python lock. SQLite WAL and a five-second busy timeout
provide the database-level concurrency behavior. These mechanisms protect
Corpus data; they do not lock a source-code workspace or coordinate external
workflows.

---

## 6. Policy types

### 6.1 PlanPolicy

`PlanPolicy` defaults are:

| Field | Default | Consumed by `plan` |
|---|---:|---|
| `max_critique_rounds` | `4` | Initial attempt plus this many ordinary revision attempts; at least one total attempt is made. |
| `judge_required` | `True` | Requires distinct judge backend and rubric. |
| `fresh_judge_reassessment_min_prior_rounds` | `None` | Opt-in positive conclusive prior-Judge-round threshold (`pass` or `revise`; inconclusive failures do not count). Once reached, after active Judge feedback is delivered to the Planner and the revised candidate receives a Critic pass, omit the Judge's own prior frontier from one Judge prompt and require a fresh current-state decision. This escape hatch is one-shot per debate; durable history and the current Critic frontier remain intact. |
| `breaker_required` | `False` | Stored policy data; the Assay `plan` primitive does not read it. |
| `persistence_veto_threshold` | `3` | Before Critic/Judge consensus, rejects a defect lineage that remains active for this many rounds; values below one disable this veto. Pass consensus may retain below-floor advisories, while the weighted Judge backstop still runs before acceptance. |
| `judge_veto_weighted_threshold` | `10.0` | Veto threshold after critic/judge pass. |
| `judge_veto_severity_weights` | Critical 10, High 3, Medium 1, Low 0.3 | Weights active, unsuppressed judge findings. |
| `cumulative_inflation_ratio` | `2.5` | Revision slice cap relative to the first parsed candidate; infinity disables the ratio component. |
| `cumulative_inflation_floor` | `8` | Minimum ratio-derived cap. |
| `max_total_slices` | `None` | Optional absolute slice cap. |
| `max_total_tiers` | `None` | Optional absolute tier cap. |
| `inflation_compaction_retries` | `0` | Conditional repair attempts after an over-limit candidate. |
| `max_verification_only_empty_scope_slices` | `None` | Optional hygiene limit for empty-write-scope verification slices. |
| `planner_guidance` | `None` | Trusted caller-supplied decomposition policy appended to the planner prompt. |
| `max_critic_failures` | `3` | Consecutive inconclusive critic limit. |
| `max_judge_failures` | `3` | Consecutive inconclusive judge limit. |

Construction validates positive role-failure limits, positive optional total
caps, nonnegative compaction/empty-scope limits, and nonblank guidance. Other
numeric relationships are consumed as supplied.

### 6.2 Rubric

`Rubric` carries `name`, system prompt, user template, response format,
severity floor, allowed decisions, complete-artifact requirement, and required
criterion IDs.

Construction enforces:

- response format exactly `review_v2`;
- a user template containing `{artifact}`;
- normalized severity in `Critical`, `High`, `Medium`, or `Low`;
- non-empty, de-duplicated allowed decisions from the four-way taxonomy;
- Boolean `requires_complete_artifact`;
- unique, non-empty required criterion strings.

`Rubric.from_yaml_text` requires `name`, `system_prompt`, and `user_template`.
`Rubric.from_yaml(path)` reads UTF-8. Rubric content remains caller-owned data.

### 6.3 BackendPolicy and budgets

`BackendPolicy` contains `retry_budget`, `timeout_seconds`, `on_cancellation`,
and `on_max_turns`. Assay's `Backend` protocol and primitives pass this value
through an `Invocation`; they do not execute retry/timeout behavior themselves.
A concrete adapter owns policy interpretation.

`Invocation.budget_usd` is likewise an adapter-facing per-invocation limit.
`review`'s chunk protocol and `plan` additionally meter reported
`Usage.cost_usd` across their own multiple calls and fail closed when aggregate
reported cost exhausts or exceeds the primitive limit. A backend that reports
zero cost cannot be independently priced by Assay.

`UsagePricingEvidence` is accounting evidence, not an enforcement mechanism.
It allows consumers to replay mixed-tier pricing after a call, but does not
turn a provider that lacks mid-call cancellation into a hard pre-call USD cap.

---

## 7. Backend and tool protocols

### 7.1 Backend

```python
@runtime_checkable
class Backend(Protocol):
    name: str
    model: str
    def invoke(self, invocation: Invocation) -> InvocationResult: ...
```

Assay performs structural runtime checks only. There is no backend registry,
environment lookup, default model, asynchronous API, or streaming surface.
Additive conversation records provide identity/operation-bound continuity
requests, outcomes, and cumulative handles; consumers still own durable
storage, provider lifecycle, and recovery UX.

The minimal protocol remains stable. `FeatureAwareBackend` is an additive
runtime-checkable protocol exposing a `BackendFeatureSet` for one concrete
route. Its closed values describe observed structured-output, tool-loop,
continuity, event, limit, deadline, retry, cache, cancellation, project-
instruction, and delegation behavior. The all-default profile is intentionally
conservative.

An invocation can carry vendor-neutral `Tool` descriptors and an optional JSON
Schema for the final response. The schema advertises a constraint to capable
adapters; Assay's parsers remain authoritative because an adapter may ignore
it.

### 7.2 Error values and exceptions

`InvocationResult.error` accepts a `BackendError` protocol value:

| Record | `kind` | Default retry flag | Additional fields |
|---|---|---:|---|
| `TransportError` | `transport` | `True` | optional HTTP status |
| `BudgetExhausted` | `budget_exhausted` | `False` | spent and limit USD |
| `MaxTurnsExceeded` | `max_turns` | `False` | used and limit turns |
| `ContentRejected` | `content_rejected` | `False` | message |
| `ExternalCancellation` | `cancellation` | `True` | cause |
| `IncompleteEvidence` | `incomplete_evidence` | `False` | missing evidence fields |
| `StaleConversation` | `stale_conversation` | `False` | provider rejection message |

Authentication has no separate record; adapters conventionally use a
non-retriable transport error, for example status 401.

Operational adapter failures should be returned in `InvocationResult.error`
when possible, but Python exceptions are not prohibited by the structural
protocol. Primitive behavior is therefore explicit: `review` and `plan`
convert returned error values to typed fail-closed outcomes; `implement`
classifies them; `parallel` propagates raised exceptions; no primitive wraps
all possible adapter exceptions.

### 7.3 Explicit tool chat

`ChatToolsBackend` extends `Backend` with:

```text
chat_with_tools(
    invocation,
    toolset,
    *,
    cache: ToolCallCache | None = None,
    max_rounds: int = 25,
) -> InvocationResult
```

This is a structural method contract, not a loop implementation. A concrete
adapter may drive a direct request/dispatch loop or delegate to an autonomous
SDK/CLI loop. It owns vendor conversion, dispatch transport, transcript
capture, and error mapping. The Protocol cannot prove enforcement, so an
adapter must either honor a supplied `cache` or `max_rounds` control according
to its declared feature profile or reject it before transport. Silently
ignoring a supplied control is non-conforming. Assay ships no generic
implementation.

`ToolSet` requires:

- `to_anthropic_tools()`;
- `to_openai_tools()`;
- `dispatch(name, arguments) -> str`;
- `mcp_config() -> Any | None`.

The two rendering methods deliberately put vendor-shape ownership on the
concrete tool set rather than in Assay. Validation and authorization inside
`dispatch` are also implementation responsibilities.

### 7.4 ToolCallCache

`ToolCallCache` is an in-memory `(tool name, canonical arguments) -> result
string` map. `canonical_args` is exactly `json.dumps(arguments,
sort_keys=True, default=str, ensure_ascii=False)`. `default=str` intentionally
collapses values such as `Path("x")` and `"x"` to the same spelling.

`get` increments hit/miss counters, `put` overwrites the key, `clear` resets
entries and counters, and `stats` reports hits, misses, and size. Counter
read-modify-write operations are lock-protected. Cache lifetime, sharing,
authorization, invalidation, and suitability of caching a side-effecting tool
are caller decisions; nothing is persisted.

---

## 8. Evidence and review protocol

### 8.1 Source registry and strict `review_v2`

`review` accepts an artifact as bytes, a `Path`, or a `sha256:` string resolved
through the Corpus. A plain text string is not an artifact value unless it is
a hash; callers encode text to bytes.

`prompt_artifact_text`, when supplied, replaces the artifact text presented to
the reviewer while the original raw artifact remains the persisted input.
Registering a distinct `evidence_source_texts["artifact"]` makes that retained
text the evidence authority and treats the prompt envelope as a separate,
non-evidentiary harness source. These advanced parameters let a workflow show
an envelope without allowing it to redefine the retained evidence.

The retained source registry always contains `artifact`. Context entries are
registered as `context:<key>`. `evidence_source_texts` can add exact retained
sources such as `prior:<id>` or `tool:<id>`, and
`evidence_source_provider` can add sources after backend tool calls finish.
Conflicting text under one source ID is rejected. `required_source_ids` must
exist before invocation and must appear in terminal coverage.

`single_prompt_required_source_ids` is an explicit, single-prompt-only coverage
relaxation for a caller that independently owns and normalizes coverage. When
provided, it must be a non-empty, unique tuple containing `artifact` and must be
a subset of `required_source_ids`. It changes only the minimum source inventory
accepted by the normal single-prompt parser. Source registration and both the
raw-size and rendered-envelope chunk protocols continue to mark the complete
`required_source_ids` inventory as terminal-required; a dynamic switch to
map/reduce therefore cannot silently weaken the terminal source ledger.

The backend must return exactly one JSON object with schema version
`assay.review.v2`; one whole-response plain/`json` Markdown fence is tolerated,
but surrounding commentary, multiple objects, and trailing tokens are not.
Root and nested objects use exact field sets; duplicate JSON keys, duplicate
finding IDs, duplicate targets, duplicate coverage entries, and unknown fields
fail schema validation.

Every finding requires at least one evidence object. Evidence is accepted only
when:

- its source is registered and declared reviewed;
- its excerpt is a verbatim substring of the retained source;
- any declared `chars:start-end` span selects that exact excerpt;
- its kind is not generated material such as a patch, proposal, test code, or
  revision.

Generated remediation belongs in `Proposal`; an adversarial experiment belongs
in `AttackSpec`.

A rubric with `required_criteria` requires exactly one `CriterionResult` per
declared ID, exactly one headline, non-empty grounded evidence for every
result, and decision/outcome consistency.

Decision validation is fail-closed:

- `pass` cannot coexist with an active finding at or above the severity floor;
- `revise`/`reject` require a blocking active finding;
- `reject` requires a confirmed or unfixable blocker;
- a complete-artifact rubric cannot pass incomplete coverage;
- a model cannot overrule harness-known source elision or missing sources.

Malformed output, returned backend errors, invalid evidence, impossible
coverage, and prompt-envelope failures produce an `inconclusive` Verdict with
a deterministic Critical diagnostic. A narrowly recoverable malformed
`revise` response may preserve independently valid findings for actionability,
but the verdict remains inconclusive.

### 8.2 Deterministic response repair

`ReviewResponseRepairPolicy` is an opt-in, closed set of source-bound repairs:

- repair a wrong character span only when the exact excerpt occurs uniquely;
- optionally remove one extra literal `\"` layer from a non-exact excerpt only
  when its declared retained source is one complete strict JSON document and
  the transformed excerpt occurs there exactly once; no other escape is
  interpreted, and machine locations, ambiguity, and required-anchor mode fail
  closed; map/reduce review disables this single-source repair;
- optionally restore an exact source span when a nonblank quote with unchanged
  outer boundaries differs only by whitespace-run normalization and its
  normalized form occurs exactly once; ambiguity and non-whitespace changes
  remain invalid; only valid-JSON retained sources also normalize JSON string
  escapes for newline, carriage return, and tab, never escaped literal
  backslashes; direct and dynamically selected chunk review disable this
  single-prompt repair;
- materialize harness-generated source anchors;
- optionally replace an unknown anchor with an exact span only when its
  declared retained source contains the supplied nonblank excerpt exactly
  once; this requires source-anchor materialization, never fuzzy-matches the
  digest, and never relaxes required-anchor catalog semantics;
- materialize strict JSON paths to exact scalar/container source spans; scalar
  binding lexically discovers raw tokens that decode to the same scalar value,
  performs at most 64
  full-document candidate reparses, and fails closed above that bound;
- optionally require source anchors;
- derive a redundant decision from a complete criterion table;
- separately allow the narrow `revise -> pass` repair only when complete
  coverage, nonblocking findings, and fully satisfied criteria prove it;
- select anchor size from 1 through 512 characters.

There is no arbitrary callback. Applied repairs persist an audit report and raw
response refs in `Verdict.artifacts`. Unresolved enabled machine references
fail normal schema validation. When optional anchor material would overflow the
single-review envelope, the policy degrades without anchors and records that
degradation; required anchors fail before backend invocation instead.

### 8.3 Large-source map/reduce

If any retained source exceeds 200,000 characters, their combined size does, or
the complete rendered single-review prompt exceeds 200,000 characters, `review`
switches to a bounded map/reduce protocol:

1. split every source into exact, non-overlapping chunks of at most 160,000
   raw characters, adaptively shrinking a chunk until its complete serialized
   mapper invocation (system prompt, rubric, source inventory, metadata, and
   JSON-escaped source text) is at most 200,000 characters;
2. fail before transport if that exact partition would require more than 256
   mapper chunks, otherwise invoke and validate one mapper response per chunk
   serially;
3. record source/chunk hashes, coordinates, response hashes, and completion;
4. hierarchically reduce reports when one terminal envelope would exceed
   200,000 characters;
5. run one terminal reducer under `review_v2`;
6. verify target closure, evidence reachability, prior adjudication, complete
   coverage, and consistency with mapped blocking observations.

Mapper count, mapper/intermediate/terminal response sizes, finding counts,
summary sizes, reduction depth, and aggregate reported cost are bounded. The
256-mapper ceiling admits about 40.9 million raw characters at the nominal
160,000-character chunk size; adaptive envelope shrinkage can lower that
capacity. Partitioning reserves a mapper slot with each exact range and fails
before any backend call if a 257th slot is needed, so near-irreducible prompts
cannot create unbounded one-character calls or retained reports. Any failed
map/reduce step returns an inconclusive verdict and a harness-authored source
manifest. If mandatory rubric/protocol overhead plus even one source character
cannot fit the mapper envelope, the harness returns an inconclusive verdict
before backend transport. The mapper is serial; this protocol is not parallel
fan-out.

### 8.4 Review provenance and persistence

`review` persists supporting artifacts before returning:

- the raw reviewed artifact;
- rubric/effective-policy snapshot;
- exact rendered prompt;
- transcript;
- chunk manifests/prompts and repair reports when applicable.

The returned Verdict records `artifact_ref`, `rubric_ref`,
`rendered_prompt_ref`, backend/model/capability metadata, token/cost telemetry,
and those support refs. `review` currently sets `Verdict.execution=None` and
does not persist the Verdict into the `verdicts` namespace. The caller stores
or commits the returned record.

The `log_dir` parameter is accepted for API consistency but is not used by
`review`; adapter logging is configured on the backend.

---

## 9. Callable primitives

### 9.1 `plan`

```text
plan(
    spec,
    *,
    planner_backend=None,
    candidate_dag=None,
    critic_backend,
    judge_backend,
    corpus,
    critic_rubric,
    judge_rubric=None,
    policy=None,
    log_dir=None,
    max_turns=5,
    budget_usd=None,
    allowed_tools=None,
    prompt_toolset=None,
    project_root=None,
    candidate_transform=None,
    candidate_validator=None,
    candidate_context_provider=None,
    max_candidate_validation_repairs=0,
    review_fn=review,
) -> PlanOutcome
```

Exactly one of `planner_backend` and `candidate_dag` is required. In generated
mode, the planner and critic must be different object instances. A required
judge must have both backend and rubric, and its backend object must be
distinct from every enabled role. Distinct objects are the implemented
independence check; Assay does not compare provider, model, credentials, or
process identity.

`candidate_dag` selects a one-shot fixed-candidate review. It performs no
Planner call, repair, or compaction; disallows Planner tool authority; and
requires one Critic and one Judge even if the supplied policy disables Judge.
Only two passes install the DAG. Every other result remains diagnostic and a
revision must arrive in a new invocation.

Each planner invocation advertises a strict TierDAG JSON Schema. Parsed scope
paths are normalized to the project execution root: a leading
`<submodule_path or project>/` segment is removed unless the caller's
`project_root`, an existing absolute directory, holds a directory of that
name. The segment is then kept on every path, existing or new, so one
candidate is never split between kept and stripped spellings; a path that only
repeats the name is not rewritten and is left to the consumer's candidate
validator. Without `project_root` the segment is always removed; the process
working directory is never consulted. A parsed candidate can be
passed through one caller-supplied `candidate_transform`; an exception or
non-DAG return ends inconclusively before review.

After candidate admission, an optional `candidate_context_provider` captures
consumer-verified source context once for the Critic/Judge pair. Its immutable
string mapping supplements, but cannot overwrite, reserved root/artifact/history
context. Callback errors fail before review while preserving prior Planner
usage. Consumers own verified excerpt capture, limits, retention, rubric
rendering and final revalidation. Reference records remain separate from
mutation scopes; empty references preserve historical canonical and review
artifact bytes.

Before model review, deterministic gates check:

- absolute/relative size bounds and optional bounded compaction;
- sibling write-scope conflicts;
- empty tiers/slices only when
  `max_verification_only_empty_scope_slices` is configured, including that
  configured verification-only allowance;
- exact `requirement_refs` drawn from root intent, constraints, or criteria.

Requirement linkage is case-sensitive and contiguous after whitespace
normalization; it proves source linkage, not semantic entailment. The
authoritative strings are `Spec.intent` plus Python's `str(constraint)` and
`str(criterion)` dataclass representations—not original YAML source slices.
The Critic/Judge remains responsible for entailment. `plan` does not call
`TierDAG.topo_order`, so unknown tier dependencies and cycles are not a
deterministic planning gate. A consumer must call `topo_order()` (and handle its
errors) before executing an accepted candidate.

An invalid requirement reference may receive an exact-source repair hint;
the hint does not make it valid. Besides exact legacy display wrappers, the
planner recognizes omission of complete single-backtick pairs when the
backtick-free wording identifies exactly one span across all root sources.
It supplies the canonical quotation and end-exclusive offsets in the
whitespace-normalized root catalog. Ambiguous spans, changed wording/case or
punctuation, added or partial delimiters, and unsupported escaped/repeated
delimiters receive no such hint. Quotes must start outside a code span; an
unsupported delimiter in the preceding source also suppresses this optional
hint conservatively. Only `requirement_refs` may change during the
existing bounded source-linkage correction. The corrected candidate must
pass the unchanged exact-source gate and normal Critic/Judge review; no root
text is rewritten and no generated feedback becomes requirement authority.

The fallback parser turns non-JSON, missing structures, and parseable JSON of
the wrong shape into a failed attempt that is fed back to the Planner. A slice,
directive, or `blast_radius` must be an object and a list field an array; an
array member that is itself an object or array, text that cannot be encoded as
UTF-8, and nesting or integer literals beyond the interpreter's limits are
rejected the same way. Identifiers, `intent`, `verify_mode`, and scalar array
members are coerced with `str()`, and nesting depth has no explicit bound of
its own. A workflow must contain backend exceptions and other unexpected
failures.

The candidate then receives critic review and, when required, independent judge
review. Plan tracks full role history, active finding lineages, consecutive
inconclusive role failures, critic finding inflation, aggregate inclusive
usage, persistent defects, and weighted judge severity. Revision attempts
receive the prior canonical DAG and active feedback. Both reviewers receive a
canonical `assay.plan-review-context.v1` envelope inside the existing
`spec_constraints` context. It contains the root constraint strings plus only
the explicitly named hard maximum for total slices, total tiers, and
verification-only empty-scope slices. Free-form planner guidance is
intentionally excluded from reviewer authority. This makes a decomposition
demand beyond the typed plan envelope visibly unsupported while leaving
semantic entailment to the root Spec. The versioned object is a documented
value-shape migration within the existing `spec_constraints` context key.
A consumer may opt into one fresh Judge reassessment after a configured
positive number of prior conclusively judged
candidates, once the Planner has consumed active Judge feedback and the revised
candidate receives a Critic pass. That reassessment omits the Judge's own prior
frontier—and therefore its historical target-closure obligations—from this one
prompt. It retains the current Critic frontier and complete durable audit
history. The candidate still requires the fresh independent Judge's root-to-
candidate audit to pass plus every ordinary current-verdict backstop; the
historical closure omission is the explicit one-shot softening, not an inferred
approval. The one-shot is consumed when that Judge prompt is invoked, including
when the result is inconclusive. If that Judge revises or is inconclusive, later
rounds return to the ordinary Planner and Judge target-closure flow.

`PlanOutcome.usage` projects the durable aggregate token, cache-token, and cost
fields across every model-backed Planner, Critic, and Judge result. Its
`execution_completeness` is the field-wise conjunction of those logical calls.
`pricing_evidence` concatenates their physical response segments in protocol
order: Planner, Critic, optional Judge, then the next semantic round. The
combined evidence is marked complete only when every contributing call carries
complete evidence and the concatenated segments reconcile the aggregate
`Usage`. Missing legacy evidence can therefore never be promoted to complete
or interpreted as an exact zero. The additive
token/cache/completeness/evidence fields are canonically omitted at their
legacy defaults; `cost_usd` remains the compatible scalar projection.

Only `decision="accepted"` attaches `candidate_dag` to `PlanOutcome.spec`.
Rejected and inconclusive outcomes retain a diagnostic candidate/ref but leave
the prior executable DAG unchanged. Every terminal outcome adds a lifecycle
event (`planned`, `plan_rejected`, or `plan_inconclusive`) and advances the
returned Spec exactly one version. Review verdict refs accumulated during the
debate are attached in that same returned version.

A retriable Planner backend error consumes the attempt it occurred in. If the
round budget ends on such an attempt, the outcome is `inconclusive` with status
`planner_infrastructure_failure` rather than `no_candidate` or
`rounds_exhausted`, and a reviewed candidate stays diagnostic. When a later
attempt received a Planner reply, the ordinary status stands and its reason
counts the lost attempts.

`plan` persists planner/reviewer transcripts, candidate bytes, Verdict records,
inflation records, and the terminal lifecycle event through the Corpus. It does
not call `put_spec` or `commit` for the final `PlanOutcome.spec`; the workflow
must publish that returned manifest. `review_fn` is an injectable review seam,
primarily useful for controlled composition and tests.

With the default `review_fn`, `log_dir` has no effect because `review` delegates
adapter logging to the backend. A custom review function may interpret it.

### 9.2 `review`

```text
review(
    artifact,
    rubric,
    *,
    backend,
    corpus,
    log_dir=None,
    artifact_commit=None,
    max_turns=5,
    budget_usd=None,
    context=None,
    kind="review",
    prompt_artifact_text=None,
    evidence_source_texts=None,
    evidence_source_provider=None,
    required_source_ids=("artifact",),
    response_repair_policy=None,
    chunk_chars=160000,
    chunk_response_retry_budget=0,
) -> Verdict
```

This is the strict evidence protocol in §8. It performs model calls and
persists support artifacts, but the caller publishes the returned Verdict.
For chunked inputs, `chunk_chars` is a per-review integer in `[1, 160000]` that
bounds each mapper's source range; it defaults to the historical 160,000-character
limit and the effective value is retained in the source manifest. The
review-wide `chunk_response_retry_budget` authorizes
bounded fresh responses for only the currently parse- or schema-rejected map.
Earlier map ledger entries are not replayed, and transport/accounting/budget
failures are never retried. A schema-rejected response is also inspected one
finding at a time through the complete chunk parser. Independently valid,
source-grounded findings are retained by the harness and merged into the next
accepted map; they are not copied into the repair prompt, and the repair cannot
erase or downgrade a retained identity. Normalized duplicate identities are
not salvaged. A non-identical reuse across attempts or an accumulated set over
the chunk finding bound terminates with an audited inconclusive result. The
source manifest contains a
harness-authored map attempt ledger binding each immutable request and
persisted response to its per-call budget/turn authorization, response schema,
validation status, retained-finding identities, accepted response lineage,
usage, and pricing-segment coordinates; it also records the aggregate budget
and the authorized and consumed retry counts.

### 9.3 `validate`

```text
validate(
    spec,
    *,
    ruleset=None,
    rule_resolver=None,
    ctx=None,
) -> Verdict
```

`ruleset=None` means an empty named ruleset and returns a deterministic pass.
Duplicate rule IDs are de-duplicated in declaration order. The complete rule
set is resolved and signature-checked before execution:

- `rule(spec) -> tuple[Finding, ...]` is pure;
- `rule(spec, ctx) -> tuple[Finding, ...]` is context-aware.

Selecting a two-argument rule without `ctx` raises
`RuleContextRequiredError` before any rule runs. Unsupported arity and non-tuple
returns are configuration errors and raise. An exception thrown while a valid
rule runs becomes a deterministic Critical `rule_error` finding, and remaining
rules continue.

High, Critical, and unknown/off-contract severities reject. Medium and Low are
nonblocking in this deterministic primitive. The Verdict has `kind="validate"`,
complete Spec coverage, no execution/transcript/backend, and an `artifact_ref`
hashing `(spec, ruleset, findings, decision)`. That ref is an identity value,
not automatically persisted artifact bytes. `validate` has no Corpus side
effect and the caller publishes the Verdict.

### 9.4 `implement`

```text
implement(
    spec,
    slice_id,
    directive,
    *,
    backend,
    corpus,
    log_dir=None,
    max_turns=50,
    budget_usd=None,
    system_prompt=None,
    allowed_tools=None,
    cwd=None,
    after_invoke=None,
    before_capture=None,
) -> Implementation
```

`implement` invokes one backend once for one directive. It does not build,
test, commit, merge, rollback, or publish a Spec.

Before invocation it records the scoped and unscoped Git state. After a backend
returns, it durably stores the transcript and optionally invokes
`after_invoke(result, transcript_ref)` exactly once. That callback is a trusted
workflow checkpoint seam. If transcript persistence or `after_invoke` fails,
Assay retains that original failure, still invokes `before_capture` exactly once
so the workflow can quiesce its provider, and then re-raises before any later Git
capture. A workflow that also performs deterministic finalization in
`before_capture` must self-gate that tail on its checkpoint state. If both
checkpointing and safety fail, the checkpoint failure remains outermost and the
safety failure is its cause. A backend that raises instead of returning a result
does not enter either callback.

After a successful checkpoint, Assay runs `before_capture` and captures the
cumulative `HEAD -> working tree` binary patch and modified paths for the
declared scope. Classification uses whether this invocation changed that scoped
state, so an inherited dirty patch is not laundered as new work. When the Corpus
is inside that worktree, one top-anchored literal exclusion for the entire
Corpus root is applied to every pre- and post-invocation diff/status command.
The exclusion remains authoritative for empty, `.` and explicitly overlapping
scopes, and requires no index or Corpus mutation, so concurrent artifact writes
cannot smear implementation evidence. A Corpus equal to the capture worktree is
rejected before backend invocation rather than silently excluding all source.

Git capture uses sanitized configuration, disables executable hooks/ext
protocols/fsmonitor/text conversion, rejects active clean filters, treats paths
as literal pathspecs, and uses an isolated temporary index/object store.
Untracked files are represented without mutating the live index. Capture
failure produces `hard_fail_capture` plus a persisted diagnostic; the primitive
does not return a successful empty patch.

Newly dirty files outside `files_in_scope ∪ creates_paths` become
`scope_escapes`. Files already dirty before invocation are treated as ambient,
so modifying an already-dirty out-of-scope file can be a conservative false
negative. Rollback remains a workflow obligation.

Classification precedence is:

1. a `before_capture` or post-invocation Git capture error ->
   `hard_fail_capture`;
2. a returned backend error -> `hard_fail_<kind>`, except for the narrow
   transport/max-turn salvage below;
3. with no blocking backend error, scope escape -> `hard_fail_scope_escape`;
4. no invocation-local changes with `verify_mode="skip"` ->
   `soft_fail_no_changes`;
5. no invocation-local changes otherwise -> `hard_fail_empty_diff`;
6. otherwise -> `success`.

Returned `transport` or `max_turns` errors with genuine in-scope changes and no
escape are salvaged for deterministic downstream verification rather than
discarded. The presence of an escape prevents that salvage, so the backend
error classification remains authoritative. A max-turn stop reason without an
error is also retained; a clean max-turn no-op is classified as a hard failure.

The returned record keeps the cumulative scoped patch/file list, transcript,
execution snapshot, usage, scope escapes, and classification. Diff and
transcript bytes are persisted, but the `Implementation` record is not. The
primitive always sets `result_commit=None`; a workflow that lands the work
stamps the resulting commit later. `log_dir` is currently unused.

`capture_workspace_delta(cwd=None, path_scope=None, max_diff_bytes=None)` exposes
the same Git capture without a backend call. A positive byte limit terminates
capture immediately after the first excess byte and raises `GitCaptureError`.

### 9.5 `bench`

```text
bench(config, *, corpus, log_dir=None, process_runner=None) -> Measurement
```

`bench` runs `[sys.executable, "-m", config.runner]` in
`config.working_dir` or the process cwd. It sends this JSON on stdin:

```json
{"bench_id":"...","parameters":{},"env":{}}
```

The child inherits the process environment with `config.env` overrides. It
must exit zero and emit either one JSON object or progress text whose final
line is the JSON object. `values` must be a non-empty object whose values can be
converted to finite floats.

Raw-output precedence is inline string `raw_output`, then existing
`raw_output_path` (relative paths resolve against the child cwd), then complete
stdout. Raw bytes are persisted under artifact kind `traces`.

The semantic `config_hash` includes bench ID, runner module, parameters, and
environment overrides. It excludes timeout, working directory, and commit.
`Measurement.commit` is `config.commit` or `"unknown"`.
The generated `ExecutionContext.cwd` is the Assay process cwd, not necessarily
`config.working_dir`; the latter is only the benchmark subprocess cwd.

Timeout, nonzero exit, malformed/empty/nonfinite KPI output, and a declared but
missing raw-output file raise `BenchRunnerError`; no Measurement is returned.
When `log_dir` is set, Assay atomically writes a unique invocation log and a
latest-by-bench convenience log with a sanitized filename. The raw artifact is
persisted, but the Measurement record is not automatically committed.
When `process_runner` is supplied, it must implement the subset of
`subprocess.run` used here; Assay delegates the launch to it so an owning
workflow can enforce descendant settlement. The default remains
`subprocess.run`.

### 9.6 `sweep`

```text
sweep(configs, *, parallelism, corpus, log_dir=None) -> Iterator[Measurement]
```

`sweep` validates positive parallelism, materializes the entire config iterable,
and runs `bench` through a bounded `ThreadPoolExecutor`. It waits for the mapped
result list before returning an iterator, and order matches input regardless of
completion order.

A point exception is observed in input order and propagates; there is no
partial-success envelope or retry. Already-started work follows normal executor
shutdown semantics. Hardware assignment and resource admission live in runner
configuration/workflow policy, not `sweep`.

### 9.7 `parallel`

```text
parallel(invocations, backend, *, max_workers=4, log_dir=None)
    -> list[InvocationResult]
```

The input must be a concrete list and `max_workers` must be positive. One
shared backend object is invoked concurrently by a bounded thread pool. Results
are returned in input order. Empty input returns immediately.

`parallel` does not choose calls, interpret results, clone the backend, retry,
persist, cancel in-flight threads, or catch backend exceptions. A raised
exception is observed according to `Executor.map` input order. The caller must
ensure the backend instance is thread-safe. `log_dir` is currently unused.

### 9.8 `smoke_check`

```text
smoke_check(
    directive,
    *,
    cwd,
    timeout=120,
    runner_parameters=None,
    process_runner=None,
) -> SmokeResult
```

Language selection examines declared edit/create paths in this precedence:

1. any `.rs`: run `cargo check --workspace`, or `cargo check -p <crate>` with
   optional comma-joined features projected from generic runner parameters and
   `per_language.rust`;
2. otherwise any `.py`: run the current interpreter's `compileall -q` in
   isolated mode without `site` (`-I -S`, with a private
   `-X pycache_prefix`), then `ruff check --select F --no-fix --no-cache`
   when Ruff is discoverable; both commands receive the paths after `--`;
3. otherwise `.cpp`, `.cc`, `.cxx`, `.h`, `.hpp`, or `.cu`: explicit
   pass-through with `language="cpp"`;
4. otherwise: pass-through with `language=None`.

Rust parameter shape errors, missing Cargo, subprocess failure, and timeout are
encoded as failed `SmokeResult`s. Python syntax/Ruff findings fail with a
tail-bounded 4096-byte excerpt; missing/unavailable Ruff after syntax success is
a soft pass. A missing Python path also follows `compileall`'s return code and
can therefore be a soft pass. `timeout` applies per spawned check, not to the
whole multi-command smoke operation. No files and unknown extensions—including
`.c`, `.cuh`, and `.hxx`—pass with `language=None`. The operation has no Corpus
or backend side effect.
When supplied, `process_runner` owns every subprocess launch through the same
`subprocess.run`-compatible subset; the default remains the standard library.

`SmokeResult.infrastructure_failure` separates a failed check that could not
run from a rejection of the code. It is true, with `passed=False`, when a
required tool is missing, a required command times out, or a tool exits without
its verdict status. `compileall` and Ruff report findings with exit 1; any
other nonzero exit is an infrastructure failure, and so is a `compileall`
exit 1 that names no rejected source (an interpreter that failed to start, or
only operating-system errors while reading a source or writing the private
cache). `cargo check` reports a build failure with exit 101; that exit is
attributed to the code only when the output contains an `error` diagnostic
that carries a `-->` source span, or a TOML parse error that Cargo reports as
a cause of its failure to parse the manifest or the lock file, because Cargo
uses the same status for registry, build-directory lock, permission, and
killed-compiler failures. The span may follow the further lines of a
multi-line message. rustc indents those lines; Cargo 1.85.0 prints them in
the first column for a manifest that is not TOML, so a first-column line
directly below the `error` header, or below another such line, continues the
message unless it is a diagnostic header itself or `Caused by:`. A span that
belongs to a warning, note, or help does not count, and a message line that
is shaped like a diagnostic header ends the diagnostic. The TOML parse error
counts only as a `Caused by:` entry of the Cargo error that names the
manifest or the lock file, and only in the wording
`TOML parse error at line N`: a Cargo configuration file that is not TOML,
which may lie outside the checked tree, the same words in a build script's
output, and a span-less parse error in other words are flagged. Neither rule
examines whether the named file lies inside the checked tree. Rust parameter
shape errors are configuration failures and are not flagged. The flag means
that the check gave no verdict; it does not establish that the environment is
at fault or that a retry would succeed, because a tool configuration in the
checked tree that Ruff or Cargo rejects and a Cargo failure that names no
source location are flagged as well. The isolated interpreter keeps the
checked tree and every `PYTHON*` variable from replacing the standard-library
`compileall`; running it without `site` keeps the `.pth` files and any
`sitecustomize` module of the interpreter's environment out as well, which
matters when an editable install puts a directory of the checked tree on that
path. The `--` separator keeps a declared path that looks like an option from
being read as one.

---

## 10. Failure semantics by boundary

| Boundary | Configuration/programming failure | Operational/content failure |
|---|---|---|
| Canonical/codec | Raises `TypeError`, `ValueError`, or `RecordDecodeError`. | Corrupt persisted data raises; there is no fallback decoding. |
| Corpus | Invalid lineage, missing/wrong-kind refs, parked Spec, or bad transform raises. | Concurrent version conflict raises `ConcurrentSpecError`; filesystem/SQLite errors propagate. |
| `plan` | Missing required judge, reused backend objects, or an invalid `project_root` raise; invalid transforms end inconclusively. | Returned backend errors, unparseable or wrong-shape planner replies, policy vetoes, budget exhaustion, and review failures become rejected/inconclusive `PlanOutcome`s. Unchecked backend exceptions can propagate; planner nesting depth has no explicit bound. |
| `review` | Invalid sources, context, template placeholders, or repair-policy wiring raises before/around invocation. | Returned backend errors and invalid model output become inconclusive Verdicts. Unchecked backend/provider exceptions can propagate. |
| `validate` | Resolver/import, arity, missing context, and return-shape errors raise. | A rule exception becomes a Critical finding; other rules continue. |
| `implement` | Initial Git HEAD/capture setup can raise `GitCaptureError`; unchecked backend exceptions propagate. | Returned backend errors and post-call capture/scope/no-change conditions become classification strings. |
| `bench` | Empty runner and invalid runner output raise `BenchRunnerError`. | Timeout/nonzero exit/missing output/nonfinite values raise `BenchRunnerError`. |
| `sweep` / `parallel` | Invalid collection/worker count raises. | Worker exceptions propagate in input observation order. |
| `smoke_check` | Normal malformed Rust runner parameters are encoded as failure. | Expected tool absence, nonzero exit, syntax errors, and timeouts are encoded in `SmokeResult`; `infrastructure_failure` marks the ones that are not a verdict on the code. |

Fail-closed does not mean “never raises.” Assay turns untrusted adjudication
failures into typed non-approval where that primitive defines such an outcome;
caller wiring errors, corrupt persistence, and unhandled implementation faults
remain exceptions.

---

## 11. Concurrency model

Assay has four distinct concurrency domains:

1. **Backend fan-out.** `parallel` shares one backend across worker threads.
   Thread safety belongs to the adapter.
2. **Benchmark fan-out.** `sweep` calls `bench` concurrently and shares one
   Corpus. Atomic artifacts and measurement locks protect persistence, while
   runners own external resources.
3. **Workspace mutation.** `implement` accepts explicit `cwd` and never changes
   process cwd. Git observation is isolated from the live index, but Assay does
   not serialize or isolate model writes. Concurrent implementations need
   separate workspaces or externally proven disjoint write scopes and a
   deliberate merge/rollback protocol.
4. **Corpus publication.** Workers may produce content-addressed records, but
   one `Corpus.commit` publishes a coherent next Spec. Spec and run locks
   protect their respective files; neither is a distributed workflow lock.

Additional constraints:

- `plan` is serial across planner, critic, and judge calls.
- large-source review maps chunks serially and reduces serially.
- `Executor.map` preserves output order, not completion order.
- Python cannot safely kill arbitrary already-running backend threads.
- registry mutation and workflow-global policy changes should finish before
  worker threads start.
- `ToolCallCache` is suitable for ordinary shared in-process use, but it does
  not provide a single-flight guarantee: two misses can dispatch the same tool
  concurrently before either result is stored.

Deterministic aggregation requires callers to sort or retain declared order
when collecting independently completed records, then publish them in one
transaction.

---

## 12. Workflow composition and publication

A typical external workflow is explicit:

```text
load or author v0
  -> validate deterministic preconditions
  -> plan with independently routed roles
  -> require PlanOutcome.decision == accepted
  -> publish the returned Spec
  -> execute tiers in topological order
       -> establish workspace isolation/resource admission
       -> bounded implement fan-out
       -> collect every started result
       -> smoke/build/test/review/benchmark gates
       -> rollback rejected work
       -> persist typed records and support artifacts
       -> Corpus.commit once in deterministic order
  -> terminal verification and lifecycle event
```

This is not hidden inside any Assay primitive. In particular:

- `TierDAG` is inert execution data. `plan` inspects its structure, but no
  Assay primitive executes its tiers or slices.
- `Implementation.classification="success"` is not build/test approval.
- `SmokeResult.passed` is only the cheap language gate.
- a returned backend result with no error is not automatically valid model
  output.
- records produced by `review`, `validate`, `implement`, and `bench` are not
  all published automatically.
- a Spec returned by `plan` is not a persisted manifest until the caller writes
  it.

At every review gate, preserve all four decisions: pass approves, revise asks
for a bounded iteration, reject is a decisive negative, and inconclusive is a
failure to adjudicate. At every plan gate, only accepted makes the candidate
executable.

---

## 13. Security and trust boundaries

### 13.1 Untrusted model/source text

Review frames source text as untrusted data, validates exact response shapes,
grounds evidence against retained text, and separates generated proposals from
observations. Native response schemas are defense in depth; parser checks are
the authority.

Plan performs deterministic gates after model parsing. Implement treats the
workspace as untrusted during Git capture and disables executable Git features
that could run while observing a patch.

### 13.2 Redaction

`redact_secrets(blob)` accepts bytes/bytearray, strips ANSI CSI and BEL,
redacts conservative token shapes and secret-like environment values captured
at module import, and replaces the current home prefix with `$HOME`. `None`
returns empty bytes; other input types raise `TypeError`.

Redaction is explicit. Corpus, review, implement, transcript serialization, and
run journals do not call it automatically. A workflow must redact untrusted
logs before persistence or prompt inclusion when its policy requires that.

### 13.3 What content addressing proves

A content hash proves byte equality under the implemented serializer or raw
artifact hash. It does not prove correctness, authorship, freshness, source
trust, safe execution, or approval. Artifact, typed-record, and measurement
reads verify their full content address. Spec reads prove canonical persisted
bytes against the exact 12-hex filename prefix and validate the complete
contiguous parent/root and append-only-reference lineage before returning the
latest manifest. The short filename prefix is collision detection rather than
a full content address; the Verdict and workflow policy still establish
meaning, and a filename alone is not an integrity proof.

---

## 14. Extension points

Supported extension seams are narrow and explicit:

- register immutable `Constraint`/`Criterion` dataclasses by unique `kind`;
- implement `Backend`, optionally `ChatToolsBackend`;
- implement `ToolSet` and choose cache scope;
- provide `ValidationRuleset` plus a resolver;
- author `Rubric` values under mandatory `review_v2`;
- implement benchmark runner modules obeying the stdin/stdout contract;
- supply `candidate_transform` for deterministic, pre-review DAG projection;
- compose primitives and persistence in an external workflow.

Schema changes are greenfield contract changes. The runtime accepts only Spec
schema 1 and review schema `assay.review.v2`; it exposes no migrator registry or
best-effort reinterpretation API. Individual `persist_optional` fields are the
only declared absent-field defaults in typed record hydration.

---

## 15. Non-goals

Assay does not provide:

- workflow discovery, scheduling, pause/resume, or next-action decisions;
- source-workspace provisioning, branch management, merges, or rollback;
- build/test runner execution other than benchmark modules and cheap smoke
  checks;
- model/provider selection or credentials;
- guaranteed adapter retries, cancellation, or process containment;
- dashboard, CLI, MCP server, notification, or human-approval UI;
- regression-statistics interpretation;
- a distributed database or artifact replication service;
- automatic redaction;
- automatic migration of old schemas.

---

## 16. Contract-test map

The test suite pins the architecture at its highest-risk boundaries:

- `test_types.py`, `test_plan_policy.py`, and `test_verdict_kind.py`: frozen
  records, deep-frozen mappings, schema/defaults, lineage helpers, and role
  taxonomy;
- `test_canonical.py`: deterministic key order, float round-trip, NFC,
  unsupported non-finite values, and hash spelling;
- `test_corpus.py`, `test_runs.py`, and breach corpus tests: strict codecs,
  lineage/ref integrity, transactions, concurrent writers, idempotency, and
  immutable finalization;
- `test_backend.py`, `test_toolset_protocol.py`, and `test_tool_cache.py`:
  protocol surfaces, errors, tool dispatch, and cache semantics;
- primitive-specific tests: response/evidence attacks, policy vetoes, Git
  capture safety, runner parsing, ordering, and smoke dispatch.

Tests use stub backends and local subprocesses. Assay's unit suite requires no
live model, network, or GPU.

---

## 17. Mental model

Keep these facts together:

1. **Spec is the lineage root.** Inline structure is small; output collections
   are content refs.
2. **Canonical means the exact Python JSON profile in §3.** It is deterministic
   for supported values, not an RFC 8785 claim.
3. **Corpus is a local transactional store.** Typed lifecycle records have
   strict namespaces; measurements are indexed in SQLite; run journals are a
   separate mutable-then-finalized surface.
4. **Backends are structural protocols.** Adapters own provider behavior and
   may still raise if they violate the preferred error-value convention.
5. **Evidence is retained-source-bound.** Model assertions cannot manufacture
   coverage or observations.
6. **Only accepted plans become executable; only pass verdicts approve.**
   Inconclusive is never success.
7. **Primitives do one bounded job.** Workflows own sequencing, publication,
   resources, rollback, and lifecycle policy.
8. **Concurrency is explicit.** Ordered results and atomic storage do not
   create workspace isolation or single-flight execution.
