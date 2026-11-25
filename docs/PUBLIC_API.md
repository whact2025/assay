<!--
SPDX-FileCopyrightText: 2026 Whact Inc.
SPDX-License-Identifier: MIT
-->

# Assay public API

This page is an import and behavior reference for the implemented Python API.
Start with the [README](../README.md) for a guided lifecycle and use the
[architecture reference](ASSAY_ARCHITECTURE.md) for data, persistence,
concurrency, and trust boundaries. This page does not redefine those formats.

## Import surfaces

The supported convenience surfaces are the names declared in each package's
`__all__`:

| Import | Purpose |
|---|---|
| `from assay import ...` | Core records, backend protocols, canonical encoding, redaction, and tool-cache helpers |
| `from assay.types import ...` | Public records, aliases, policies, and structural type protocols |
| `from assay.primitives import ...` | Workflow operations and selected public helper/error types |
| `from assay.corpus import ...` | Filesystem Corpus, transactional commit records, and run handles |
| `from assay.canonical import ...` | Deterministic serialization, hashing, and subtype registries |
| `from assay.backend import ...` | `Backend`, `ChatToolsBackend`, and additive `FeatureAwareBackend` protocols; closed `BackendFeatureSet` |

Callable primitives and `Corpus` are deliberately not re-exported from the
root `assay` module. For example:

```python
from assay import Spec, Rubric
from assay.corpus import Corpus
from assay.primitives import plan, review, validate
```

Lower-level modules contain additional implementation helpers. Unless a name
is in that module's `__all__`, treat it as internal. Some operation-specific
errors are exported only by their defining module:

```python
from assay.primitives.bench import BenchRunnerError
from assay.primitives.review import ReviewSchemaError
```

The current package requires a POSIX runtime: Corpus and run-folder locking use
`fcntl.flock`. Git-backed implementation capture also requires a Git executable
when those APIs are used.

## Records and identity

`Spec` is the immutable lifecycle root. Its `with_*` methods return a new
version and compute references, but they do not persist the referenced record.
For new workflow writes, prefer `Corpus.commit(...)`: it persists typed
records, rebases on the latest Spec while holding the cross-process Spec lock,
and publishes one manifest as the visibility point.

`ContentHash` is a `typing.NewType` over `str`; constructing it performs no
validation. Corpus read/write boundaries require the exact form
`sha256:<64 lowercase hexadecimal digits>`.

Frozen records recursively freeze mapping and collection fields in their
`__post_init__` methods where required. Python `Literal` annotations are not
runtime checks by themselves; the records that need closed runtime taxonomies,
such as `Verdict`, validate them explicitly.

`BackendExecutionTrace.tool_authority_digest`, when present, is the
`sha256:<64 lowercase hexadecimal digits>` identity of the canonical,
secret-free tool authority used for the invocation. It may be compared across
backend runs without exposing tool credentials or endpoint details.

## Corpus

Create a Corpus with a directory dedicated to one Spec lineage:

```python
from pathlib import Path
from assay.corpus import Corpus

with Corpus(Path(".assay/example")) as corpus:
    latest = corpus.latest_spec()
```

Important write contracts:

- Initialize an empty Corpus by publishing one v0 Spec with `put_spec`; a
  transaction commit requires an existing lineage root.
- `put_spec(spec)` publishes a fully formed next manifest and rejects stale
  parents, skipped versions, divergent same-version manifests, missing typed
  records, and rewritten append-only reference lists.
- `get_spec(version)` accepts exactly one bounded, singly linked regular file,
  proves that its raw JSON is canonical, and recomputes its exact version and
  12-hex content-derived filename. The reader authenticates the persisted raw
  shape before hydration so canonical manifests from older compatible codecs
  remain readable when a newer optional field gains a default.
- `get_spec_with_digest(version)` returns that safely hydrated Spec together
  with the full SHA-256 of its exact persisted raw canonical bytes. Use the
  atomic pair for lineage, CAS, or external authority; `hash_record(spec)` can
  describe a schema-enriched hydrated shape and is not a persistence identity.
- `latest_spec()` validates every contiguous manifest from v0 through the tip,
  including authenticated parent/root identity and append-only reference
  prefixes; it never hides a missing, divergent, malformed, or corrupt
  historical version. New commits bind the persisted predecessor digest. If a
  public `Spec.with_*` helper computes a predecessor hash from a canonical
  legacy manifest whose older raw shape omitted a subsequently added optional
  field, `put_spec` recognizes that exact deterministic shape and normalizes
  the parent/root to the persisted digest before serialization. Because the
  helper's in-memory value is then not byte-identical to the published child,
  reload the returned version before deriving another direct child; `commit`
  performs this normalization and returns the published Spec directly. A
  previously persisted child that names only a rehydrated, non-persisted
  predecessor representation remains a dangling edge and fails closed.
  A `Corpus` also binds the real `versions/` directory created or observed by
  its constructor and rejects later alias or inode replacement. The remaining
  filesystem concurrency contract is cooperative: Corpus writers use the
  shared `flock`; uncooperating same-UID directory mutation is outside it.
- `commit(...)` is the transaction-oriented path for appending
  Implementations, Verdicts, Measurements, LifecycleEvents, or already stored
  references. Repeated references are idempotent.
- `locked_spec_authority()` yields the exact latest Spec while holding the
  same cross-process lock used by `put_spec` and `commit`. Use it only for a
  bounded external side effect whose authorization must not change between
  decision and write. Do not call a Spec writer inside the context; a
  same-thread attempt raises `RuntimeError`, and other writers wait until the
  context exits. Publish any resulting Corpus evidence afterward with an
  optimistic parent/CAS transform. This ordinary generator context preserves
  cross-Corpus nesting on all supported Python versions, but `with` syntax
  cannot publish its cleanup across the caller's own `__enter__` return
  bytecode.
- `retained_spec_authority()` exposes that owner explicitly for composed
  control-plane code. Register `authority.retry_cleanup` in an outer cleanup
  stack *before* calling `authority.__enter__()`. Its source-owned acquisition
  lease stays inert before entry, becomes live before the immutable snapshot
  returns, and retains an interrupted guard/flock/thread-lock release for
  exact retry through `SpecAuthorityCleanupError.owner`. The retained guard
  fails closed before acquisition on a Python runtime that cannot report
  current-thread ownership of its native reentrant lock. Reentrant retained
  acquisition additionally requires an exact recursion-count probe; use the
  ordinary guard for cross-Corpus nesting where that probe is unavailable.
  A fork from another thread waits for the bounded authority region to end.
  After `fork`, child process state uses fresh in-process locks; unwinding an
  inherited same-thread authority abandons the child's copied lease without
  unlocking the still-live parent authority.
- Typed records other than Measurements are canonical bytes under
  `artifacts/<kind>/`. Measurements use `measurements.sqlite`; their identity
  is still `hash_record(measurement)`.
- `put_artifact(blob, kind=..., name_hint=...)` stores raw bytes by SHA-256.
  `name_hint` is informational and does not affect the filename or identity.
  `kind` is not sanitized: it must be a trusted, single safe path component;
  passing untrusted traversal syntax can escape the intended namespace.
- `get_artifact_bounded(ref, max_bytes=...)` is the control-plane read path
  for artifacts with a declared maximum size. It opens the Corpus artifact
  tree and candidates through no-follow descriptors, requires stable regular
  single-link files, streams and verifies the requested SHA-256, and enforces
  the byte ceiling both before and during the read. Traversal also has fixed
  depth and entry ceilings. A matching-stem candidate above `max_bytes`
  raises `ValueError` when no valid bounded copy exists; missing or corrupt
  bounded candidates raise `FileNotFoundError`.
- `open_run(recipe, idempotency_key=...)` creates a private run folder.
  `RunHandle.finalize` is idempotent only for the same terminal result; a
  conflicting second result raises `RunAlreadyFinalizedError`.
  `RunHandle.finalize_elapsed` fills the default-zero duration from one
  monotonic-clock snapshot and returns the recorded `RunResult`. A retry with
  the same non-duration terminal fields returns that recorded result; callers
  cannot supply their own duration through this method.

## Primitives

All primitives are synchronous.

### `validate`

Resolves a `ValidationRuleset`, preflights every rule signature, and returns a
typed Verdict. Pure one-argument rules can be deterministic for identical
inputs; context-aware rules may reflect the local environment. Two-argument
rules require a `RuleContext`; omitting it raises
`RuleContextRequiredError` before any rule executes. Exceptions raised by a
rule become a synthetic Critical finding, while malformed rule signatures and
return types remain caller errors.

Pass `rule_resolver=` for consumer-owned rules. If omitted, the current built-in
resolver imports `crucible.rules.<rule_id>`; this is an implementation-bound
default, not a generic plugin discovery mechanism.

### `plan`

Runs generated or fixed-candidate plan adjudication and returns `PlanOutcome`.
Only `decision == "accepted"` attaches an executable DAG to the returned Spec.
Rejected and inconclusive candidates remain diagnostic data. Planner, Critic,
and Judge backends must be independently routed when those roles are enabled.
The outcome durably aggregates inclusive `Usage`, field-wise
`ExecutionCompleteness`, and ordered `UsagePricingEvidence` across every
model-backed role call and semantic round. Combined pricing evidence is
complete only when all contributing calls provide complete segments that
reconcile the aggregate usage; absent legacy evidence remains incomplete.

Exactly one candidate source is required: pass `planner_backend=` to generate
and refine a DAG, or pass `candidate_dag=` with `planner_backend=None` to review
an already typed DAG. Fixed-candidate mode invokes no Planner, accepts no
Planner tools or repair attempts, and always routes one Critic and one Judge.
Any non-pass requires a new invocation and leaves `outcome.spec.tier_dag`
unchanged; it never installs the rejected candidate. A prior executable DAG,
if supplied by a programmatic caller, remains the base Spec's DAG.

The planner parser validates the optional `read_only_references` array as closed typed
`ReadOnlySourceReference` records: stable ID, normalized relative POSIX path,
whole-file SHA-256 and inclusive positive integer line range. References add
no mutation authority; omitted or empty arrays preserve historical bytes.

`candidate_context_provider: Callable[[TierDAG], Mapping[str, str]] | None`
provides consumer-verified context once per admitted candidate round. A copied
immutable snapshot is supplied to both reviews. Reserved artifact, root and
prior-findings keys reject; exceptions or malformed context yield
`candidate_context_failed`, preserving the candidate and prior paid usage
without invoking reviewers. Consumers must render the supplied keys in their
rubrics, retain source evidence, enforce capture bounds and revalidate source.
The callback does not replace deterministic candidate validation.

`project_root: str | os.PathLike[str] | None` is the existing absolute
directory that generated scope paths are relative to. A leading
`<submodule_path or project>/` segment is kept only when that directory holds
a directory of the same name, and then on every path: an existing file and a
new file below it stay together. A path that only repeats the name is not
rewritten in that case; `plan` checks no path for existence, so a
`candidate_validator` must report a `files_in_scope` entry that only repeats
the name. A `creates_paths` entry cannot be told apart from a new file below
that directory and is kept as written. `None` always strips the segment. The
process working directory is never consulted, an invalid root raises
`ValueError` or `TypeError` before any backend call, and fixed-candidate
review does not use it.

The planner invocation advertises a strict response schema, and the fallback
parser rejects a reply of the wrong shape instead of raising or guessing.
Non-JSON, a slice that is not an object, a non-empty directive or
`blast_radius` that is not an object, a non-empty list field that is not an
array, an object or array used as a path, dependency or requirement reference,
text that is not UTF-8 encodable, and nesting or integer literals beyond the
interpreter's limits all become failed attempts fed back to the Planner.
Identifiers, `intent`, `verify_mode` and scalar array members are coerced with
`str()`; nesting depth has no explicit bound of its own. Callers must contain
backend exceptions and other unexpected failures and call
`TierDAG.topo_order()` before execution because `plan` does not gate unknown
dependencies or cycles.
`PlanPolicy.breaker_required` is currently carried as consumer metadata; this
primitive does not route or execute a breaker role.

### `review`

Reviews bytes, a Path, or an existing `sha256:...` artifact-reference string
under the strict `assay.review.v2` contract. It validates source coverage, evidence
coordinates, criterion results, decision consistency, and severity-floor
behavior. Large retained sources and complete rendered single-review prompts
over 200,000 characters use a bounded map/reduce protocol. Each mapper receives
at most 160,000 raw source characters, adaptively reduced so its exact serialized
system-plus-user envelope never exceeds 200,000 characters. Irreducible rubric
or protocol overhead fails deterministically before transport. The protocol
admits at most 256 mapper chunks total; adaptive partitioning that would need a
257th chunk fails before the first backend call. This caps serial provider work
and retained mapper evidence while preserving exact, non-overlapping coverage
for inputs that fit within the limit. Malformed
approval cannot become a pass. Expected backend/result failures return an
inconclusive Verdict with synthetic evidence; invalid Python arguments and
unexpected backend exceptions may raise.

`required_source_ids` is the authoritative terminal coverage inventory. A
coverage-owning harness may also pass `single_prompt_required_source_ids` as a
non-empty, unique subset containing `artifact`; only the normal single-prompt
parser uses that narrower minimum. If review selects either map/reduce path,
including a dynamic switch caused by rendered-envelope growth, source manifests
and the terminal reducer still require the complete `required_source_ids` set.

`map_context_source_ids` is an opt-in tuple of retained, terminal-required
`context:*` source IDs. When map/reduce runs, each artifact mapper receives the
exact selected context text as `map_context_sources`; context and other source
maps do not. These companions are interpretation-only: mapper findings remain
restricted to the artifact chunk's own source ID and exact text, and every
selected context still has its separate map and terminal-coverage obligation.
The exact companion serialization participates in the mapper envelope limit
and every retry. The source manifest and artifact-map attempt ledger retain the
selected IDs, hashes, and character counts without duplicating their text.
Omitting the argument, or passing an empty tuple, preserves the existing
prompts and manifest shape. Single-prompt review validates the selection but
does not add a second copy of context already rendered by the rubric template.

`ReviewResponseRepairPolicy` enables only its declared, deterministic,
source-bound repairs. Its optional whitespace-normalized quote recovery
restores exact retained bytes only when the nonblank quote preserves meaningful
outer boundaries, differs solely in whitespace runs, and has exactly one
normalized match. Only when the retained source itself is valid JSON, newline,
carriage-return, and tab string escapes participate in that normalization while
escaped literal backslashes remain lexical content. Assay disables this
operation for both direct and dynamically selected map/reduce review. Its
optional JSON quote-escape recovery removes exactly one literal `\"` layer
only when the declared source is one complete strict JSON document and the
resulting exact excerpt occurs uniquely in that same source; all other escapes,
non-descriptive locations, and ambiguity remain invalid. Like whitespace
normalization, this operation is disabled when review uses map/reduce. Its
optional unknown-anchor recovery requires an exact, unique, nonblank excerpt in
the response's declared retained source and never fuzzy-matches the opaque
anchor digest. It is effective only alongside source-anchor materialization and
never relaxes a required-anchor catalog. Supplying no policy selects strict
parsing.

### `implement` and `capture_workspace_delta`

`implement` performs one backend invocation, captures the scoped cumulative
HEAD-to-working-tree Git delta, persists diff/transcript evidence, and returns
an Implementation. Before/after captures determine whether this invocation
changed the scope for classification, but the persisted patch may include
preexisting in-scope work. It does not run verification, commit changes, or
roll back scope escapes. Concurrent calls must pass an explicit, distinct
`cwd` for each workspace.

When `allowed_tools` is nonempty, `implement` and `plan` place one empty-schema
Tool descriptor per supplied name on `Invocation.tools`; `None` or an empty
tuple places `None`. The concrete backend decides how to interpret that field.

`capture_workspace_delta` exposes the same bounded, non-mutating Git capture
for work completed outside `implement`. Both paths can raise `GitCaptureError`
when Git cannot produce trustworthy evidence.

Scope paths are literal. `implement` reads the directive's `files_in_scope`
and `creates_paths`, and `capture_workspace_delta` reads `path_scope`, as
paths relative to `cwd` and passes them to Git as exact names. No leading
component is dropped, including one equal to the name of the `cwd` directory,
and the working tree is not consulted to interpret a path, so a path names the
same file before and after that file exists. A write to a path the scope does
not name is outside it, also when the `cwd` directory is named `proj`: with a
scope of `proj/src/a.py`, an invocation that newly dirties `src/a.py` returns
that path in `scope_escapes` and is classified `hard_fail_scope_escape`,
unless a capture failure, a backend error or a turn ceiling takes precedence.
A `src/a.py` that was already dirty before the invocation is ambient: a
further write to it is neither captured nor listed in `scope_escapes`, and
the result is classified as if that write had not happened; an invocation
that also changes a path the scope names can therefore still be a `success`.
A caller whose paths are relative to another directory rewrites them before
the call.

### `bench` and `sweep`

`bench` runs `python -m <BenchConfig.runner>` once. It sends `bench_id`,
`parameters`, and `env` on stdin and accepts a KPI JSON object as all of stdout
or its final line. Values must be nonempty and each must be coercible by
`float(value)` to a finite result. Timeout, nonzero exit, invalid output, and a
missing declared raw-output file raise `BenchRunnerError`.
`process_runner=` may supply a `subprocess.run`-compatible execution boundary;
when omitted, the standard library runner preserves standalone behavior.

`sweep` eagerly materializes both configurations and ordered results using a
thread pool. It does not assign GPUs or return partial results after a failure.

### `parallel` and `smoke_check`

`parallel` accepts a concrete list of Invocations, invokes one Backend from a
bounded thread pool, and returns results in input order. Backend exceptions use
`Executor.map` observation semantics.

`smoke_check` uses case-sensitive suffix matching and selects Rust before
Python before the exact suffix set `.cpp`, `.cc`, `.cxx`, `.h`, `.hpp`, and
`.cu`. Rust runs `cargo check`; Python runs `compileall` in an isolated
interpreter without `site` (`-I -S`) and, when available, Ruff's F rules; the
listed C++/CUDA suffixes form an explicit skip-with-pass lane. Other C-family
suffixes are unrecognized. Required-command failures are encoded in
`SmokeResult`; optional Ruff absence/timeout is a soft pass.
`SmokeResult.infrastructure_failure` is true for a failure that is not a
verdict on the code: a missing required tool, a timeout, or an exit without the
tool's verdict status. A `cargo check` exit 101 is attributed to the code only
when its output contains an `error` diagnostic with a `-->` source span, or a
TOML parse error that Cargo reports for the manifest or the lock file; the
same parse error in a Cargo configuration file is flagged. The flag reports a
missing verdict; it does not say that a retry would succeed.
`process_runner=` provides the same optional workflow-owned execution boundary
for every required and optional smoke subprocess.

## Backend protocols

`Backend` requires public `name` and `model` attributes plus
`invoke(Invocation) -> InvocationResult`. `ChatToolsBackend` additionally
requires `chat_with_tools(...)`. Both are runtime-checkable structural
protocols: `isinstance` verifies member presence, not method signatures or
behavior. Assay ships no concrete model adapter.

Backends report expected operational failures through
`InvocationResult.error`, whose value satisfies the `BackendError` protocol.
Unexpected programming errors may still raise. Retry, timeout, cancellation,
and max-turn policy are adapter responsibilities expressed through
`BackendPolicy` and Invocation fields.

`InvocationResult.completeness` defaults every evidence dimension to false,
and `execution_trace` defaults to `None`, preserving constructor compatibility
without treating absent token, cost, terminal, or trace evidence as complete.
`Usage` also carries default-omitted cache-creation and cache-read subsets of
total input. Optional `UsagePricingEvidence` preserves ordered physical-call
segments; a complete record must reconcile aggregate Usage and requires
complete token/cost evidence. `Implementation` and `Verdict` propagate this
prompt-free ledger durably; `Verdict` also persists the authoritative
aggregate cache counters, including exact zeros and aggregation across chunked
review. Absent cache counters distinguish legacy records: a complete ledger
can recover their exact values, while an incomplete ledger is diagnostic
evidence and leaves token-accounting debt. Such segments may be cumulative or
repeated even when the backend reports exact terminal totals.
Feature-aware consumers may narrow to `FeatureAwareBackend`; the minimal
`Backend` surface is unchanged.

## Canonical encoding

`serialize` implements Assay's deterministic JSON profile: NFC strings,
string-keyed objects, sorted keys, compact UTF-8 output, finite numbers, and
17-significant-digit float normalization. It is intentionally not advertised
as RFC 8785. `parse` decodes UTF-8 JSON but does not prove the input was already
canonical. `hash_record(value)` is SHA-256 over `serialize(value)` and returns
the prefixed string form.

Constraint and criterion registries are mutable extension points. Normal
package import currently hydrates builtin kinds through the package import
graph; `Spec.from_yaml`, `Corpus`, and `assay.bootstrap.populate_registry`
also idempotently ensure they are present. Consumers may register additional
dataclass types before parsing their discriminators and should not depend on an
initially empty registry.

## Redaction and logging

`redact_secrets` is explicit, best-effort redaction for `bytes` or `bytearray`;
`None` is treated as empty bytes. Corpus and the primitives do not apply it
implicitly. A workflow that sends captured output to a model or stores it for
postmortem must choose and apply its redaction policy at that boundary.

The `log_dir` parameters on `plan`, `implement`, `review`, and `parallel` are
currently accepted but do not themselves write logs. Configure backend-
specific logging on the backend instance. `bench` is different: its `log_dir`
writes subprocess logs.

## Implementation-bound defaults

Most public records and protocols are consumer-neutral, but three current
defaults remain product-bound behavior and are intentionally documented rather
than hidden:

- Omitting `validate(rule_resolver=...)` resolves
  `crucible.rules.<rule_id>`.
- `MutationRegion` defaults its marker text to `CRUCIBLE-MUTATE-START` and
  `CRUCIBLE-MUTATE-END`; callers can supply other literal markers.
- The built-in planner and implementer system prompts contain Crucible-oriented
  wording and project-layout guidance. `implement(system_prompt=...)` can
  replace its prompt; `plan` currently has no public system-prompt override.

These defaults are implementation facts, not extension requirements. Removing
them requires a behavior/API change rather than a documentation rewrite.
