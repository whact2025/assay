<!--
SPDX-FileCopyrightText: 2026 Whact Inc.
SPDX-License-Identifier: MIT
-->

# Assay

Assay is a Python 3.10+ library for evidence-backed engineering automation. It
provides immutable records, deterministic content hashes, a transactional
on-disk corpus, vendor-neutral backend protocols, and eight composable
primitives:

`plan`, `review`, `validate`, `implement`, `bench`, `sweep`, `parallel`, and
`smoke_check`.

Assay deliberately stops at the mechanism boundary. A workflow decides which
primitive runs next, how failures are retried or escalated, which rules and
rubrics express product policy, and whether a result is published. Model
adapters, product runners, rules, rubrics, and complete workflows are supplied
by consumers. Crucible is one such workflow package.

Three current defaults retain Crucible-specific vocabulary: validation's
fallback rule namespace, `MutationRegion` marker text, and the built-in
planner/implementer prompts. They are documented precisely under
[implementation-bound defaults](docs/PUBLIC_API.md#implementation-bound-defaults)
rather than presented as generic extension requirements.

```text
workflow / product policy
        |
        | composes
        v
Assay primitives -> typed records -> Corpus
        |
        v
vendor-neutral Backend protocol
```

Assay has no workflow scheduler and no model SDK dependency.

## Installation

Install the package from this checkout:

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -e .
```

For development, the repository declares its test tools in the `dev`
dependency group:

```bash
uv sync --group dev
uv run pytest -q
uv run --with ruff ruff check .
```

The only runtime dependency is PyYAML.

The current package requires a POSIX platform. `Corpus` and `RunHandle`
locking import `fcntl` and use `flock`, and the top-level package's import graph
loads that persistence package even when a caller initially imports only core
types. There is no Windows locking fallback.

## A first Spec lifecycle

Author `example/spec.yaml`:

```yaml
schema_version: 1
project_context:
  project: example
  repo_url: https://github.com/example/example.git
  base_commit: "0000000000000000000000000000000000000000"
  branch: main
  submodule_path: null
  runners: {}
  runner_parameters: {}
intent: |
  Add a deterministic greeting function and cover it with a unit test.
constraints:
  - kind: budget
    usd_limit: 5.0
    time_limit_hours: 1.0
done_criteria:
  - kind: assertion
    test_name: test_greeting
    must_pass: true
```

Load version 0, publish it, validate it, and atomically publish the resulting
verdict with version 1:

```python
from pathlib import Path

from assay import Spec
from assay.corpus import Corpus
from assay.primitives import validate

spec_dir = Path("example")

with Corpus(spec_dir) as corpus:
    authored = Spec.load_v0(spec_dir / "spec.yaml")
    authored_ref = corpus.put_spec(authored)

    verdict = validate(authored)
    committed = corpus.commit(verdicts=(verdict,))

    assert verdict.decision == "pass"
    assert committed.spec.version == 1
    assert committed.spec.verdict_refs == (committed.refs.verdict_refs[0],)

print(authored_ref)
```

Calling `validate()` without a ruleset intentionally runs an empty ruleset, so
the example passes. Production workflows normally supply a
`ValidationRuleset` and either a `rule_resolver` or the companion Crucible rule
package.

## Mental model

A `Spec` is the immutable lifecycle manifest for one unit of work. Version 0
contains authored intent, project identity, constraints, completion criteria,
dependencies, and optional planning policy. Later versions can carry an
accepted `TierDAG` and append content hashes for implementations, verdicts,
measurements, and lifecycle events.

Large records are not embedded in every manifest. They are stored once and
referenced by `sha256:<64 lowercase hex digits>` values:

```text
Spec v0
   |
   | parent/root lineage
   v
Spec v1 ---- implementation_refs[] ----> Implementation
        |--- verdict_refs[] ------------> Verdict
        |--- measurement_refs[] --------> Measurement
        `--- lifecycle_event_refs[] ----> LifecycleEvent
```

`Spec.with_*` methods return a new in-memory version; they never mutate their
receiver. `Corpus.commit()` is the normal publication path for newly produced
lifecycle records because it persists records, rebases under the Spec write
lock, and exposes one new manifest as the transaction's visibility point.

Run telemetry is a separate sidecar concern. `Corpus.open_run()` creates a
`RunHandle` for mutable events and summaries followed by one immutable terminal
`RunResult`. Run folders do not replace Spec lineage or content-addressed
records.

## Public API

Every public type in [`assay.types`](src/assay/types/__init__.py) is also
available from `assay`:

```python
from assay import (
    Backend,
    BenchConfig,
    ContentHash,
    Finding,
    Implementation,
    Invocation,
    InvocationResult,
    Measurement,
    PlanOutcome,
    ProjectContext,
    Rubric,
    Spec,
    TierDAG,
    Verdict,
)
```

The top-level package additionally exports:

- canonical helpers: `serialize`, `parse`, `hash_record`,
  `CONSTRAINT_TYPES`, and `CRITERION_TYPES`;
- backend protocols: `Backend`, `ChatToolsBackend`, and the additive
  `FeatureAwareBackend`, plus `BackendFeatureSet`;
- safety and tool helpers: `redact_secrets`, `ToolCallCache`, and
  `canonical_args`.

The public type families are:

- identity and execution: `ContentHash`, `ProjectContext`, and
  `ExecutionContext`;
- constraints and criteria: `Constraint`, `BudgetConstraint`,
  `BlastRadiusConstraint`, `ActivationConstraint`, `Criterion`,
  `AssertionCriterion`, and `ParityCriterion`;
- planning: `PlanPolicy`, `PlanDecision`, `PlanOutcome`,
  `ImplementationDirective`, `MutationRegion`, `Slice`, `Tier`, and `TierDAG`;
- review and validation: `Rubric`, `Evidence`, `CriterionResult`,
  `CriterionOutcome`, `Proposal`, `AttackSpec`, `Finding`, `Verdict`,
  `ReviewCoverage`, `ReviewDecision`, `FindingDisposition`,
  `ValidationRuleset`, `RuleContext`, `REVIEW_DECISIONS`,
  `FINDING_DISPOSITIONS`, `ADVISORY_KINDS`, `GATE_KINDS`, and
  `ALL_VERDICT_KINDS`;
- invocation and tools: `BackendPolicy`, `Tool`, `ToolSet`, `Invocation`,
  `InvocationResult`, `Usage`, `Message`, `BackendError`, and its seven concrete
  error records; identity-bound continuity uses `ConversationIdentity`,
  `ConversationHandle`, `ConversationTransportRequest`, and
  `ConversationTransportOutcome`; normalized execution evidence uses
  `ExecutionCompleteness`, `BackendExecutionTrace`, and
  `BackendExecutionEvent`;
- lifecycle records: `Implementation`, `Measurement`, `LifecycleEvent`,
  `BenchConfig`, and `SmokeResult`;
- Spec construction: `Spec` and `UnknownConstraintKind`.

Callable primitives are imported from `assay.primitives`, not eagerly from the
top-level package:

```python
from assay.primitives import (
    bench,
    capture_workspace_delta,
    implement,
    parallel,
    plan,
    review,
    smoke_check,
    sweep,
    validate,
)
```

Corpus types are imported from `assay.corpus`:

```python
from assay.corpus import (
    CommitRefs,
    CommitResult,
    ConcurrentSpecError,
    Corpus,
    RecordDecodeError,
    RunAlreadyFinalizedError,
    RunHandle,
    RunResult,
)
```

The primitives package re-exports selected controls such as
`RuleContextRequiredError`, `GitCaptureError`, and
`ReviewResponseRepairPolicy`. Other specialized errors are imported from the
module that owns them:

```python
from assay.primitives.bench import BenchRunnerError
from assay.primitives.review import ReviewSchemaError
```

## Spec contract

`Spec.from_yaml()` and `Spec.load_v0()` accept exactly `schema_version: 1`.
Unknown fields, unknown constraint or criterion kinds, malformed dependency
hashes, and any other schema version are rejected. Parsed Specs start with
`version=0`, `parent=None`, `root=None`, and `execution=None`.

The authored YAML fields are:

- `schema_version`
- `project_context`
- `intent`
- `constraints`
- `done_criteria`
- `depends_on` (optional content hashes)
- `plan_policy` (optional)

`project_context` accepts `project`, `repo_url`, `base_commit`, `branch`,
`submodule_path`, `runners`, and `runner_parameters`. Runner parameters must be
a deterministic JSON-shaped value: string keys with nested nulls, booleans,
integers, finite floats, strings, lists/tuples, and mappings. Mapping fields are
defensively frozen after construction.

Assay registers these built-in discriminated records:

| Registry | `kind` | Concrete type | Fields apart from `kind` |
|---|---|---|---|
| constraint | `budget` | `BudgetConstraint` | `usd_limit`, `time_limit_hours` |
| constraint | `blast_radius` | `BlastRadiusConstraint` | `no_edits_outside` |
| constraint | `activation` | `ActivationConstraint` | `status`, `reason` |
| criterion | `assertion` | `AssertionCriterion` | `test_name`, `must_pass` |
| criterion | `parity` | `ParityCriterion` | `reference_commit`, `test_name` |

An activation status is either `active` or `parked`. A parked Spec can be
parsed and inspected, but `Corpus.put_spec()` refuses to publish it.

Domain packages extend YAML construction by registering a frozen dataclass
before parsing or hydrating the Spec:

```python
from dataclasses import dataclass

from assay import CONSTRAINT_TYPES


@dataclass(frozen=True)
class LicenseConstraint:
    identifier: str
    kind: str = "license"


CONSTRAINT_TYPES["license"] = LicenseConstraint
```

The registered key and the dataclass's stable `kind` value form the wire
discriminator.

### Lineage and references

For a published lineage:

- the first manifest is version 0 with no parent or root;
- the next manifest must be exactly the latest version plus one;
- `parent` must hash the latest manifest;
- `root` remains the hash of version 0;
- lifecycle reference tuples are append-only;
- every local lifecycle reference must resolve to a record of the correct
  type and namespace before the manifest becomes visible;
- `depends_on` hashes are address-validated but may refer to Specs owned by a
  different corpus.

Concurrent writers cannot replace the same version. A conflicting writer gets
`ConcurrentSpecError` and must rebase its records onto `latest_spec()`.

## Canonical identity

`assay.canonical.serialize()` produces Assay's deterministic UTF-8 JSON form:

- dataclass instances become objects;
- tuples and lists become arrays;
- dictionary keys must be strings and are sorted;
- strings and keys are normalized to Unicode NFC;
- finite floats retain 17-significant-digit round-trip precision;
- non-finite floats and unsupported Python objects are rejected;
- output has no insignificant whitespace or trailing newline.

`hash_record(value)` returns the SHA-256 of those bytes as a canonical
`sha256:<hex>` string. `ContentHash` is a `typing.NewType` over `str`, so it is
for static distinction rather than `isinstance()` checks.

`parse()` returns ordinary JSON-shaped Python values. Typed record hydration is
performed by the strict corpus codec, which rejects missing required fields,
unknown fields, invalid nested hashes, and type mismatches.

## Corpus semantics

Constructing `Corpus(spec_dir)` creates `versions/`, `artifacts/`,
`transcripts/`, `diffs/`, `traces/`, and `runs/`. Content-addressed blobs are
written to `artifacts/<kind>/<sha256>.bin`. Measurement records are indexed in
`measurements.sqlite`; their raw output remains a content-addressed artifact.

The central write operations have distinct purposes:

| Operation | Contract |
|---|---|
| `put_artifact(blob, kind=...)` | Idempotently store arbitrary bytes by their SHA-256. `name_hint` is informational and never changes identity. |
| `get_artifact_bounded(ref, max_bytes=...)` | Stream and hash-verify an artifact through no-follow descriptors while enforcing byte, traversal-depth, and entry-count ceilings before retaining the complete payload. |
| `put_record(record, kind=...)` | Validate and store an `Implementation`, `Verdict`, `Measurement`, or `LifecycleEvent` in its required namespace. |
| `put_spec(spec)` | Publish an already assembled next manifest after validating lineage and every reachable local record. |
| `commit(...)` | Persist supplied lifecycle records and atomically publish one rebased next manifest. |
| `get_record(ref, as_type=...)` | Verify namespace and hash, then strictly hydrate the requested lifecycle type. |

`commit()` writes content-addressed records before publishing the manifest. An
interrupted transaction can therefore leave an unreachable record, but it
cannot publish a Spec that points at a missing record. Repeating identical
record input is idempotent and does not append duplicate refs.

`put_artifact(kind=...)` does not validate or sanitize the namespace string;
callers must supply a trusted, single path component. Do not pass user- or
model-controlled path text as an artifact kind.

### Run records

`Corpus.open_run(recipe)` allocates a private run directory. Supplying an
`idempotency_key` deterministically reopens the same recipe/key directory.
Within a run:

- `write_event()` appends one durable JSON line;
- `write_summary()` atomically replaces the latest complete summary;
- `finalize()` creates one immutable terminal result;
- `finalize_elapsed()` measures one finite non-negative monotonic duration and
  returns the immutable terminal result;
- repeating the same final result is idempotent;
- attempting to replace a different final result raises
  `RunAlreadyFinalizedError`;
- the context manager finalizes as `completed` or `failed` when needed.

Events and summaries cannot be written after finalization.

## Primitive contracts

| Primitive | Produces | Owns | Does not own |
|---|---|---|---|
| `validate` | `Verdict` | Deterministic rule dispatch and gate decision | Persistence, retries, workflow routing |
| `review` | `Verdict` | Backend invocation, strict response parsing, evidence grounding, review provenance | Domain rubric content, retry/escalation policy |
| `plan` | `PlanOutcome` | Bounded Planner/Critic/Judge protocol and candidate safety checks | Publishing the returned Spec, executing its DAG |
| `implement` | `Implementation` | One backend invocation, Git delta/transcript capture, coarse classification | Build, test, commit, rollback, region enforcement |
| `bench` | `Measurement` | One runner subprocess, timeout, KPI validation, raw-output artifact | Device scheduling, comparisons, regression policy |
| `sweep` | ordered `Measurement` iterator | Bounded multi-config `bench` fan-out | Partial-result policy, device assignment |
| `parallel` | ordered `InvocationResult` list | Bounded multi-invocation backend fan-out | Result interpretation, persistence, retries |
| `smoke_check` | `SmokeResult` | Cheap language-aware pre-build check | Full compilation and product verification |

### Validation

`validate(spec, ruleset=..., rule_resolver=..., ctx=...)` supports exactly two
rule signatures:

```python
from assay import Finding, RuleContext, Spec


def pure_rule(spec: Spec) -> tuple[Finding, ...]:
    return ()


def context_rule(
    spec: Spec, ctx: RuleContext,
) -> tuple[Finding, ...]:
    return ()
```

Rules are resolved and signature-checked before any rule executes. A selected
two-argument rule requires `RuleContext`; omission raises
`RuleContextRequiredError` before partial work can occur. A rule exception is
converted into a Critical deterministic finding. A malformed rule signature or
non-tuple return is a configuration error and raises.

Critical and High findings reject. Medium and Low findings are retained but do
not reject. Unknown severity values fail closed as blocking. The returned
Verdict has `kind="validate"`, complete Spec coverage, no backend provenance,
and no execution timestamp, which keeps equal deterministic results hash-equal.

If no resolver is supplied, rule id `foo` is loaded from
`crucible.rules.foo.foo`. This optional lookup convention requires Crucible to
be importable. Passing `rule_resolver` keeps validation independent of that
companion package.

### Implementation

`implement()` asks one backend to apply an `ImplementationDirective`, then
captures the Git working-tree delta, changed paths, transcript, usage, stop
reason, and execution context. It stores the diff and transcript as artifacts
and returns the typed record; the workflow publishes that record.

The directive's `files_in_scope + creates_paths` bounds capture and supports
file-level `scope_escapes` detection. Those paths are relative to `cwd` and are
read as written; no leading directory name is dropped. A `MutationRegion` is
included in the implementer prompt, but line-range enforcement belongs to the
consuming workflow. `classification` is an early transport/scope/delta signal,
not a build or test result.

The primitive never commits, builds, tests, or rolls back. `result_commit` is
therefore `None`. Concurrent workflows must pass an explicit per-workspace
`cwd`; relying on process cwd is suitable only when no other thread changes or
uses it. `capture_workspace_delta()` exposes the same bounded Git capture for
workflow-owned checkpoints.

### Bench, sweep, and parallel

`bench()` runs `sys.executable -m <BenchConfig.runner>`, sends this JSON on
stdin:

```json
{"bench_id":"example","parameters":{},"env":{}}
```

The runner exits zero and writes one JSON object to stdout:

```json
{
  "values": {"latency_ms": 12.5},
  "raw_output": "optional inline trace"
}
```

`values` must be a non-empty mapping whose values can be converted to finite
floats. A runner may use `raw_output_path` instead of `raw_output`; a declared
path must exist. Timeout, nonzero exit, malformed output, empty KPIs, and
NaN/Infinity raise `BenchRunnerError`. `working_dir` and `timeout_seconds` do
not affect the measurement's configuration hash; runner, bench id, parameters,
and environment do.
Journal-owning consumers can pass a `subprocess.run`-compatible
`process_runner` to place the benchmark under their own descendant-settlement
boundary; standalone calls use `subprocess.run`.

`sweep()` eagerly materializes its configs, runs `bench()` in a bounded thread
pool, and returns results in input order. Errors propagate; the primitive does
not define a partial-success policy. `parallel()` likewise preserves input
order and requires a concrete `list[Invocation]`. Its shared backend must be
safe for concurrent synchronous `invoke()` calls.

### Smoke checks

`smoke_check()` selects from paths in `files_in_scope + creates_paths`:

- Rust: `cargo check --workspace`, or `cargo check -p <crate>`, with optional
  features from `runner_parameters`;
- Python: `compileall` in an isolated interpreter without `site` (`-I -S`),
  followed by Ruff F-rules when Ruff is available;
- `.cpp`, `.cc`, `.cxx`, `.h`, `.hpp`, or `.cu`: an explicit pass-through
  because no generic cheap parser is configured;
- no recognized files: pass with `language=None`.

Other suffixes, including `.c`, `.cuh`, and `.hxx`, currently follow the
unrecognized-file path and therefore return `language=None`.

Subprocess failures and timeouts are represented in the returned `SmokeResult`.
`infrastructure_failure` is true when the failed check produced no verdict on
the code: a missing required tool, a timeout, or an exit without the tool's
verdict status.
The result is not persisted automatically.
An optional `process_runner` applies one workflow-owned execution boundary to
all smoke subprocesses; standalone calls retain the standard-library default.

## Evidence-backed review

`review()` accepts artifact bytes, a `Path`, or a content hash already in the
Corpus. It persists the reviewed bytes, rubric, rendered prompt, transcript,
and relevant failure/repair diagnostics. Its required response is the closed
`assay.review.v2` JSON object described by `Rubric`.

The model selects one decision:

- `pass`
- `revise`
- `reject`
- `inconclusive`

`Verdict.passed` is the derived expression `decision == "pass"`; it is not a
model field. Workflows should branch on the full decision because `revise`,
`reject`, and `inconclusive` have different meanings.

A review can pass only when all of these conditions hold:

- the decision is allowed by the rubric;
- there is no active finding at or above the rubric's severity floor;
- required source coverage is complete when the rubric requires it;
- every evidence excerpt is a verbatim substring of its registered source;
- every evidence `source_id` was reviewed and has retained source text;
- when `required_criteria` is non-empty, exactly one evidence-backed
  `CriterionResult` exists for every criterion and its outcome agrees with the
  decision.

`reject` requires a confirmed or unfixable blocking finding. `revise` requires
an active blocking finding. Generated fixes and attacks are represented as
`Proposal` and `AttackSpec`; they cannot be passed off as source evidence.

Backend errors, malformed JSON, duplicate keys, schema violations, ungrounded
evidence, contradictory decisions, or incomplete approval claims return an
`inconclusive` Verdict with a deterministic Critical diagnostic. Large review
inputs, and single-review inputs whose complete rendered prompt exceeds 200,000
characters, use the bounded chunk and reduction protocol so coverage is based
on the mapper completion ledger rather than a reducer's assertion. Mapper
chunks contain at most the per-review `chunk_chars` limit and shrink adaptively
when JSON escaping, source metadata, or rubric instructions would make the
exact serialized mapper prompt exceed 200,000 characters. `chunk_chars` must be
an integer from 1 through 160,000, defaults to 160,000 for compatibility, and
its effective value is recorded in the source manifest. If even one source
character plus mandatory protocol overhead cannot fit, review fails
deterministically before invoking the backend. A review may use at most 256
mapper chunks (about 40.9 million raw characters at the nominal chunk size).
If adaptive envelope fitting would require a 257th chunk, partitioning fails
before transport, bounding provider calls and retained mapper evidence even
when only one or a few source characters fit per invocation.
Callers may authorize a non-negative, review-wide
`chunk_response_retry_budget`. Only a parse- or schema-rejected current map is
retried; completed prefix maps are retained, while transport, accounting, and
budget failures remain terminal. When a schema-rejected map contains other
independently valid, source-grounded findings, the harness retains them across
the repair instead of trusting the replacement response to repeat them. The
rejected model text is not reflected into the repair prompt. A normalized
identity may recur across attempts only for an identical typed finding; a
conflicting reuse, duplicate identity in one rejected response, or retained set
larger than the bounded finding limit fails closed rather than selecting a
winner or growing the reduction envelope. The source
manifest records the authorization, consumption, immutable request/response
hashes, persisted response references, retained-finding identities, accepted
response lineage, per-call budget and turn authorization, response-schema
identity, validation result, usage, and pricing-segment coordinates for every
map attempt.

`evidence_source_texts`, `context`, and the post-invocation
`evidence_source_provider` form the retained source registry. Tool and prior
finding evidence is accepted only when the corresponding source text is
registered. `required_source_ids` defines the sources whose coverage is
mandatory. A harness that independently owns and normalizes single-prompt
coverage may set `single_prompt_required_source_ids` to a non-empty subset that
includes `artifact`. That relaxation applies only to the normal single-prompt
parser: if source size or the rendered envelope selects either chunk path, the
full `required_source_ids` set remains authoritative in mapper accounting and
the terminal reduction ledger.

`ReviewResponseRepairPolicy` is an opt-in set of closed, deterministic response
canonicalizations: exact excerpt spans, unique whitespace-normalized quotes,
one-level JSON quote-escaped excerpts, source anchors, JSON paths, and tightly
gated decision derivation. The JSON quote-escape mode applies only when the
declared retained source is one complete strict JSON document: it replaces
only literal `\"` pairs with `"`, then requires one exact match in that same
source. It does not interpret any other escape, and map/reduce review disables
this single-source repair. The
whitespace mode accepts only a nonblank quote with meaningful outer boundaries
whose whitespace-collapsed form occurs exactly once; it restores the retained
source's exact bytes and character span, while ambiguity or any non-whitespace
difference remains invalid. Only when the retained source itself is valid JSON,
the same closed normalization treats JSON string escapes for newline, carriage
return, and tab as whitespace but preserves escaped literal backslashes. An
independently opted-in mode may also replace an
unknown anchor with an exact character span only when the declared retained
source contains the supplied nonblank excerpt exactly once; it never fuzzy-
matches anchor digests. This mode also requires source-anchor materialization
and never relaxes a required-anchor catalog. There is no arbitrary repair
callback. Every applied
repair is persisted as provenance; if a repair cannot be proven from retained
sources, the review remains inconclusive.

### Executable review example

This backend satisfies the protocol and returns a minimal valid passing review:

```python
import json
from pathlib import Path

from assay import InvocationResult, Message, Rubric, Usage
from assay.corpus import Corpus
from assay.primitives import review


class StaticBackend:
    name = "static"
    model = "static-reviewer"

    def invoke(self, invocation):
        payload = {
            "schema_version": "assay.review.v2",
            "decision": "pass",
            "reasoning": "The complete artifact satisfies the rubric.",
            "coverage": {
                "complete": True,
                "sources_reviewed": ["artifact"],
                "omissions": [],
            },
            "findings": [],
        }
        text = json.dumps(payload)
        return InvocationResult(
            final_message=text,
            transcript=(Message(role="assistant", content=text),),
            usage=Usage(),
            stop_reason="end_turn",
        )


rubric = Rubric(
    name="statement",
    system_prompt="Review only the supplied evidence.",
    user_template="Review this artifact:\n{artifact}",
)

with Corpus(Path(".assay-review")) as corpus:
    verdict = review(
        b"The greeting function is deterministic.",
        rubric,
        backend=StaticBackend(),
        corpus=corpus,
    )

assert verdict.decision == "pass"
assert verdict.coverage.complete
```

## Planning and policy

`plan()` runs a bounded planner/critic/judge loop. Planner, critic, and supplied
judge objects must be distinct backend instances. When
`PlanPolicy.judge_required` is true, both `judge_backend` and `judge_rubric` are
required.

For an externally authored DAG, pass `candidate_dag=` and leave
`planner_backend=None`. This fixed-candidate mode makes zero Planner calls,
disallows Planner tools and candidate-repair attempts, and requires one
independent Critic and Judge review even when the stored policy normally makes
Judge optional. A revise, reject, or inconclusive decision never installs the
DAG; the caller must correct the source and start a new review invocation.

Each planner invocation advertises a strict `TierDAG` response schema. The
parsed candidate is checked for same-tier write-scope conflicts, authoritative
requirement references, configured tier/slice/empty-scope bounds, and the
validity of any optional `candidate_transform` result.

Generated scope paths are canonicalized against the explicit `project_root=`
directory, never the process working directory. A leading
`<submodule_path or project>/` segment is removed unless that existing absolute
directory holds a directory of the same name; it is then kept on every path,
existing or new. A `files_in_scope` entry that only repeats the name is left
for the consumer's candidate validator to report; a `creates_paths` entry
cannot be told apart from a new file below that directory and is kept as
written. Without `project_root` the segment is always removed. An invalid root
raises before any backend call.

Directives may carry typed `read_only_references` to existing source or tests:
stable `reference_id`, project-relative normalized POSIX `path`, whole-file
`sha256:<64 lowercase hex>` and inclusive `start_line`/`end_line`. These records
never add edit authority, blast radius or parallel write conflicts. Empty
references are omitted from persisted canonical data and review artifacts,
preserving historical hashes. Literal source paths and locators belong in the
structured references; directive prose can refer to their stable IDs.

`candidate_context_provider(candidate)` optionally supplies a `Mapping[str,
str]` of verified source evidence after candidate admission. Assay captures one
immutable snapshot per candidate round and shares it with Critic and Judge.
Artifact, root-requirement and prior-findings keys cannot be overwritten.
Callback exceptions or invalid mappings stop before reviewer transport with
`candidate_context_failed`, retaining the candidate and any Planner usage.
Consumers own source verification, excerpt limits, evidence retention, rubric
rendering of context, and source revalidation before acceptance. A reference
record by itself is not proof of verified source or fulfillment of a root goal.

`plan()` does not call `TierDAG.topo_order()`. Unknown tier dependencies and
cycles are therefore not a deterministic planning gate; a workflow must call
`topo_order()` and handle its errors before executing an accepted candidate.
The response schema helps capable backends produce valid identifiers and
directive shapes, but the parser remains tolerant enough that a workflow must
not treat schema advertisement alone as validation.

The parser treats non-JSON, incomplete objects, and parseable JSON of the wrong
shape as failed plan attempts that are fed back to the Planner: a slice, or a
non-empty directive or `blast_radius`, that is not an object; a non-empty list
field that is not an array (a bare string is never split into characters); an
object or array where a path, tier dependency, or requirement reference is
expected; text that cannot be encoded as UTF-8; and nesting or integer literals
beyond the interpreter's limits. Identifiers, `intent`, `verify_mode`, and
scalar array members are still coerced with `str()`, and nesting depth has no
explicit bound of its own.
Backend response schemas are defense in depth; callers must still contain
backend exceptions and other unexpected failures at the workflow boundary.

`PlanPolicy` controls critique depth, judge routing, persistent-finding veto,
weighted judge veto, cumulative plan-inflation limits, optional total tier and
slice caps, compaction retries, verification-only empty-scope allowance,
planner guidance, and consecutive critic/judge failure breakers.

The `breaker_required` field is not an execution switch: `plan()` has no breaker
backend or breaker round. Planner, critic, and judge are the complete role set
implemented by this primitive.

`PlanOutcome.decision` is `accepted`, `rejected`, or `inconclusive`. Only an
accepted outcome attaches the candidate DAG to `outcome.spec`. Other candidates
remain available through `candidate_dag` and `candidate_ref` for diagnosis but
cannot be mistaken for executable state. Debate verdicts and the terminal
lifecycle event are persisted; `outcome.spec` is staged but not published as a
manifest. The caller must publish it through a concurrency-aware workflow.
`outcome.usage`, `execution_completeness`, and `pricing_evidence` aggregate all
model-backed calls in protocol order. In fixed-candidate mode that ledger
contains only Critic and Judge work. Pricing evidence is complete only when
every call supplies complete physical segments that exactly reconcile the
inclusive aggregate usage.

A retriable Planner backend error costs the attempt it occurred in. When the
round budget ends on such an attempt the outcome is `inconclusive` with status
`planner_infrastructure_failure`, never `no_candidate` or `rounds_exhausted`.
When a later attempt still received a Planner reply the ordinary status
stands; only the reasons of `rounds_exhausted` and `no_candidate` count the
lost attempts.

## Backends, invocations, and tools

`Backend` is a runtime-checkable structural protocol:

```python
from __future__ import annotations

from assay import Invocation, InvocationResult


class Backend:
    name: str
    model: str

    def invoke(self, invocation: Invocation) -> InvocationResult: ...
```

`Invocation` carries system/user text, optional tool descriptors, a turn limit,
an optional USD budget, optional `BackendPolicy`, optional final-response JSON
Schema, and an optional identity/operation-bound continuity request.
`InvocationResult` carries the final message, complete transcript, token/cost
usage, stop reason, optional typed error, conservative execution completeness,
an optional redacted provider-neutral trace, an optional continuity outcome,
and optional ordered per-response pricing evidence. Cache creation/read token
counters are subsets of total input. Complete pricing evidence must reconcile
the aggregate usage and token/cost completeness; multi-call review preserves
the call order and fails closed on pricing completeness. Existing
adapters that omit the additive evidence remain source-compatible, but their
completeness defaults to false.

`FeatureAwareBackend` is an optional narrowing protocol. Its
`BackendFeatureSet` uses closed values for structured output, tool-loop
ownership, continuity, event detail, limits, deadlines, retries, cache,
cancellation, project instructions, and native delegation. Unknown adapters
resolve conservatively in consumers; the minimal `Backend` protocol is
unchanged.

Assay provides seven concrete backend-error records:

- `TransportError`
- `BudgetExhausted`
- `MaxTurnsExceeded`
- `ContentRejected`
- `ExternalCancellation`
- `IncompleteEvidence`
- `StaleConversation`

The live `BackendError` surface is a structural protocol, although strict
Corpus hydration recognizes the seven built-in discriminators above. Adapters
can conventionally represent authentication failure as a non-retriable
`TransportError` with HTTP status 401. `BackendPolicy` is invocation data for
an adapter to interpret; Assay does not run an independent retry engine around
`Backend.invoke()`.

`ChatToolsBackend` adds `chat_with_tools(invocation, toolset, cache=...,
max_rounds=...)`. A `ToolSet` publishes vendor-native descriptor lists,
dispatches validated calls, and may expose MCP bridge configuration.
`ToolCallCache` keys results by tool name plus canonicalized arguments and
reports hit/miss counts. The cache is in-process and explicit; callers choose
whether to share it across phases.

## Provenance and secret handling

Provenance is carried by the record that owns it:

- `Implementation`: base commit, diff/transcript refs, modified files, scope
  escapes, backend/model execution context, usage, cost, and stop reason;
- `Verdict` from review: input artifact, rubric, rendered prompt, transcript,
  backend/model/capability fields, tokens, cost, coverage, criterion results,
  and diagnostic artifact refs;
- deterministic validation `Verdict`: no wall-clock execution context, backend,
  or transcript;
- `Measurement`: commit, configuration hash, raw-output ref, KPI values, and
  execution context;
- `LifecycleEvent`: event kind, reason, affected versions, and execution
  context;
- `RunResult`: terminal status, elapsed time, cost, and optional error kind and
  message. Context-manager-generated exception messages are truncated, while
  direct construction does not impose a length bound.

Prompt-bearing `Message.content`, `Invocation.system`, and `Invocation.user` are
excluded from dataclass `repr`, but that is not storage or transport redaction.
Neither Corpus nor the primitives automatically sanitize supplied bytes.

Call `redact_secrets(blob)` before persisting captured logs or placing them in a
model prompt. It strips ANSI/BEL control sequences, redacts recognized token
shapes and secret-valued environment entries captured when Assay is imported,
and replaces the local home-directory prefix with `$HOME`. Redaction is
best-effort and must be applied by the workflow at every untrusted capture
boundary.

## Repository layout

```text
assay/
├── pyproject.toml
├── README.md
├── src/assay/
│   ├── __init__.py
│   ├── bootstrap.py
│   ├── canonical/
│   ├── types/
│   ├── corpus/
│   ├── primitives/
│   └── backend/
└── tests/
```

This README is the guided introduction. See the
[`public API reference`](docs/PUBLIC_API.md) for import and boundary semantics
and the [`architecture reference`](docs/ASSAY_ARCHITECTURE.md) for the complete
data, persistence, evidence, concurrency, and failure model. The package
modules and tests are the executable definition.

## License

Copyright (c) 2026 Whact Inc. Licensed under the [MIT License](LICENSE).
