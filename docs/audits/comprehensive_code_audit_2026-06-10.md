# assay — Comprehensive Code Audit (archived snapshot)

> **Archived evidence, not current status.** Sections 1–8 record the tree and
> conclusions observed on 2026-06-10. Their present-tense findings, line
> numbers, severity labels, and backlog order are intentionally preserved as
> audit history and must not be treated as an open-work list. Section 9 records
> the first remediation pass; section 10 reconciles later implementation
> changes. Use the [README](../../README.md),
> [public API reference](../PUBLIC_API.md),
> [architecture reference](../ASSAY_ARCHITECTURE.md), and current tests for
> the active contract.

**Date:** 2026-06-10
**Tree:** assay @ `a32b513` + the `75e34b1` review fixes landed this session (RUB-5/RUB-10).
**Method:** 7 parallel deep-audit agents (corpus/canonical · implement · plan/dag/spec ·
review/validate · backend/safety · bench/sweep/parallel · QA/arch-conformance), graded against
`docs/ASSAY_ARCHITECTURE.md`. **The highest-severity findings were re-verified by hand against
the source** — including refuting two agent over-claims. Same method/format as the crucible and
fabric-usd audits.
**Legend:** ✅ = re-verified by me against the code this session. ◐ = agent-reported, plausible,
confirm the exact lines before fixing. ❌-REFUTED = agent claim checked and found wrong.

> **Why this audit matters.** assay is the substrate beneath crucible and fabric-usd — the
> typed primitives, the content-addressed Corpus, the Backend protocol, the review/validate
> gates. Several findings from the crucible audit turn out to have their **root cause here**:
> the cost-undercount/lost-work class (ARC-2/CRU-04) is a Corpus concurrency property; the
> parallel-tier-conflict gap (RUB-3) is a missing structural check in `dag.py`; the
> fail-open-severity bug (RUB-5, fixed) had an un-fixed *other half* in the floor. Fixing
> things in assay fixes them for every consumer at once.

---

## 1. Executive summary

assay is a clean, well-typed, genuinely domain-free substrate (the §13 genericness boundary
holds — zero `import crucible` / isaac/fabric leakage in `src/assay`; arch-doc claims verify
6/6). Canonical serialization is **provably deterministic** for the Spec/Verdict/Implementation
chain (17-digit floats, NFC, sorted keys, no wall-clock/PID/random in the hashed payload) — the
content-addressing foundation is sound. The implement primitive correctly captures untracked
files (`git add --intent-to-add`) and detects scope-escapes post-hoc; the review AND-gate is
structurally sound; the importlib rule resolver's `isidentifier()` guard is traversal-safe; the
backend error-model returns (not raises) typed errors. Good bones.

The real issues cluster in four places, and three of them are the substrate roots of crucible
findings:

1. **Corpus concurrency (ACOR-2) — the crown-jewel risk.** Concurrent same-version `put_spec`
   is **last-writer-wins by mtime**, silently dropping the other writer's appended refs. It is
   safe *today* only because callers serialize the append (crucible's tier runner appends
   sequentially after the parallel implement phase) — an **unenforced invariant**, not a
   guarantee. This is the substrate root of the lost-work / cost-undercount class.
2. **Gate fail-open via the floor (AREV-1/2) — the other half of RUB-5.** I fixed
   finding-severity to fail closed this session, but `severity_floor` itself is taken from
   rubric YAML uncoerced; a typo'd/lowercase floor (`"critical"`) falls through
   `_SEVERITY_ORDER.get(floor, 1)` → **Medium**, silently weakening the gate.
3. **Backend-error → parse-failure conflation (APLAN-1).** The planner can't distinguish a
   backend transport/budget error from malformed JSON — the assay-layer cousin of the 529
   silent-noop class.
4. **Silent-bad-measurement + secret-leak surface (ABE-1/2, ABK-1/7).** Missing raw-output
   files → empty bytes silently; NaN/Inf KPIs stored as real; `_safety.redact_secrets` exists
   but is **never called by any primitive** (workflow-responsibility, silent-leak failure mode);
   `Invocation`/`Message` prompt fields lack `repr=False`.

No Critical bug is *live* in the current single-writer-append usage, but several are **latent
Critical** — they activate the moment a caller does what the Corpus claims to support
(concurrent writes) or fat-fingers a rubric floor. Most fixes are small and land in the
substrate once for all consumers.

**Verification corrections (intellectual honesty):** two agent-reported corpus findings were
**refuted** on inspection — ACOR-4 ("`get_artifact` returns wrong bytes across kinds") is false
(line 686 hash-verifies every read; same-hex ⇒ same-bytes by content-addressing), and ACOR-5
("duplicate-extension artifacts") is false (`put_artifact` enforces a fixed `.bin` name,
line 630-639). They're recorded in §6 so they aren't re-raised.

---

## 2. P0 — Corpus integrity & the gate (substrate roots of crucible findings)

### ACOR-2 ✅ Concurrent same-version `put_spec` silently drops appended refs (latent Critical)
- **Severity:** Critical (latent — gated by caller serialization today) · **Category:** Concurrency / data-integrity · **Arch:** §4.4 (concurrent writers)
- **Where:** [corpus/store.py:577-617](../../src/assay/corpus/store.py) — `put_spec` writes `versions/v{spec.version}_{short}.json` (filename = version + content-hash); `latest_spec` → `get_spec(N)` → `max(candidates, key=mtime)` (line 600).
- **Mechanics (✅ verified):** two workers both `latest_spec()` → same v3 → each `spec.with_implementation(impl_X)` → each `put_spec` writes a *distinct* `v4_<hashX>.json` (no filename collision, per §4.4's design). But the two v4 manifests carry **conflicting content** (each has only its own ref appended), and `latest_spec` returns whichever has the later mtime — the other worker's implementation/verdict ref is **silently lost from the lineage**. §4.4's hash-suffix scheme prevents a filesystem *error*; it does **not** merge concurrent appends. This is the substrate root of the ARC-2 cost-undercount and CRU-04 lost-work classes in the crucible audit.
- **Why not live today:** crucible's tier runner parallelizes the *implement* phase but appends to the Spec **sequentially** afterward (per the crucible architect audit). So the race isn't currently triggered — but nothing in assay *enforces* that, and `assay.parallel` + any future concurrent-append caller would trip it.
- **Action:** make same-version append safe at the substrate: a CAS/read-modify-write retry in `put_spec` (re-read latest, detect a competing same-version write, merge-or-retry), **or** a documented `Corpus.append_to_spec(base, change)` with optimistic-concurrency, **or** at minimum a per-spec lock file + a loud error when two distinct `v{N}_*` manifests exist for the same N (today they silently coexist and one wins). Whichever — the unenforced "callers must serialize" invariant must become enforced or explicit. · **Effort:** M–L

### AREV-1 / AREV-2 ✅ `severity_floor` not normalized → a typo'd floor silently weakens the gate (the other half of RUB-5)
- **Severity:** High · **Category:** Gate-integrity · **Arch:** §5.3
- **Where:** `types/rubric.py:51` (`severity_floor: str = "Medium"`, no validation) + `primitives/review.py` `_passes` (`floor_rank = _SEVERITY_ORDER.get(severity_floor, 1)` ✅ — default rank 1 = Medium).
- **Evidence:** this session's RUB-5 fix made *finding* severities fail-closed (unknown → Critical), but the **floor** is still taken from YAML uncoerced and defaulted to Medium on any miss. A rubric authored with `severity_floor: critical` (lowercase) or `High ` (trailing space) → `_SEVERITY_ORDER.get(...)` misses → floor silently becomes Medium → the gate passes High findings it should block. Fail-open, and invisible.
- **Action:** normalize + validate `severity_floor` at `Rubric` construction (`from_yaml_text`/`__post_init__`): `capitalize()` + membership check, raise `ValueError` on unknown (or coerce via the same `_normalize_severity`). Belt-and-suspenders: change `_passes`'s floor default to Critical rank. *This completes the RUB-5 fix — do it next.* · **Effort:** S

### AREV-5 ◐ A throwing validation rule crashes the gate instead of failing closed
- **Severity:** High · **Category:** Error-handling · **Where:** `primitives/validate.py` ◐ (~:137-160) — rule callables invoked in a loop with no try/except; an exception (TypeError, import error, rule bug) propagates and crashes `validate()` rather than producing a Critical finding.
- **Action:** wrap each rule call; on exception emit a synthetic Critical `rule_error` finding (id = rule_id, excerpt = exception) and continue — fail closed, consistent with the parse-failure path. · **Effort:** M

### ACOR-1 / ACOR-3 ◐ SQLite measurement persistence: `INSERT OR REPLACE` + `json.dumps` (not canonical) for values
- **Severity:** Medium · **Category:** Data-integrity / hash-stability · **Where:** `corpus/store.py:149,190` (`INSERT OR REPLACE INTO measurements`) and the `m.values` JSON path (◐ :145,186 `json.dumps` rather than canonical `serialize`).
- **Evidence (◐, needs `hash_record(Measurement)` coverage check):** (a) `INSERT OR REPLACE` is last-writer-wins on the ref key — if two semantically-distinct measurements ever map to one ref, one is clobbered; (b) persisting `values` via `json.dumps` while hashing via canonical `serialize` (17-digit) means a measurement round-tripped through SQLite can re-hash differently → idempotence break / dedup miss / cost re-ingest drift. Both hinge on exactly what `hash_record` covers (does it include `ExecutionContext.timestamp`?). Confirm, then: store `values` via canonical `serialize`, and reconsider `OR REPLACE` vs `ON CONFLICT DO NOTHING`. · **Effort:** S–M

### ACOR-6 ◐ `open_run` 16-bit tag, no retry → `FileExistsError` crash under burst concurrency
- **Severity:** Medium · **Where:** `corpus/store.py:758` — `secrets.token_hex(2)` (16 bits) + `mkdir(exist_ok=False)`, no retry. Birthday collision ~50% at ~256 same-second `open_run` calls → an uncaught `FileExistsError` crashes the worker. **Action:** `token_hex(4)` + a small retry loop. · **Effort:** S

---

## 3. P1 — Planner, implement, measurement correctness

### APLAN-1 ◐ Backend error indistinguishable from parse failure in the planner
- **Severity:** High · **Category:** Error-swallowing · **Where:** `primitives/plan.py` ◐ (~:341-351, :575) — `_run_planner` returns `(None, ...)` for *both* `result.error is not None` (transport/budget/cancel) and JSON-parse failure; the loop feeds a "your JSON was invalid" rebuke either way.
- **Evidence:** the assay-layer cousin of the crucible 529 class (CRU-01). A transport error becomes a fix-your-JSON retry; a budget-exhaustion becomes the same. The workflow can't react correctly (retry-with-backoff vs escalate vs abort) because the cause is erased.
- **Action:** return a discriminator (`backend_error` vs `parse_failure`) and branch — propagate backend errors as backend errors. · **Effort:** S

### APLAN-2 ◐ Parallel-tier file-conflict is not enforced structurally (root of crucible RUB-3)
- **Severity:** High · **Category:** Spec gap · **Where:** `types/dag.py` ◐ (TierDAG construction) — no check that sibling slices in one tier have disjoint `files_in_scope ∪ creates_paths`; the invariant lives only in the planner *prompt* and crucible's `plan_critic` rubric (which the crucible audit found had *drifted out* of the swebench/deepswe variants — RUB-3).
- **Evidence:** because it's advisory-only, two same-tier slices touching one file race at implement time. Enforcing it in `dag.py` (structurally) closes it for every consumer and makes the rubric check a backstop, not the sole defense.
- **Action:** add `TierDAG` validation: for each tier with ≥2 slices, reject non-disjoint write scopes with the overlap cited. · **Effort:** M

### AIMP-1 ◐ `_normalize_scope_for_cwd` partitions on `/` → breaks on Windows backslash paths
- **Severity:** Medium · **Category:** Windows-portability · **Where:** `primitives/implement.py` ◐ (~:252, `head, _, rest = p.partition("/")`). A planner-emitted `kit\source\foo.rs` won't normalize → diff sees no changes → misclassified `hard_fail_empty_diff` though the implementer wrote correctly. **Action:** normalize separators (`.replace("\\","/")` / `PurePath`) before partitioning. · **Effort:** S

### AIMP-2 ◐ `git add --intent-to-add` leaves the index dirty after diff capture
- **Severity:** Medium · **Where:** `primitives/implement.py` ◐ (~:280-288). The `-N` index entries persist after the diff is captured; a downstream `git reset`/`checkout`/`stash` (rollback, sibling slice) can interact unexpectedly. **Action:** `git reset HEAD -- <scope>` to clear intent-to-add after capture. · **Effort:** S

### ABE-1 ◐ `bench` raw-output-path: missing file → empty bytes, silently
- **Severity:** High · **Category:** Data-integrity · **Where:** `primitives/bench.py` ◐ (~:234, `if rp.is_absolute() or rp.exists(): ... read_bytes() if rp.is_file() else b""`). A runner that names a raw-output path that doesn't exist yields a zero-byte `raw_output_ref` indistinguishable from a legitimately-empty capture — a silently-incomplete measurement. **Action:** raise `BenchRunnerError` when a declared raw-output path is absent. · **Effort:** S

### ABE-2 ◐ NaN / Infinity KPI values propagate silently into stored measurements
- **Severity:** High · **Category:** Numeric integrity · **Where:** `primitives/bench.py` ◐ (~:149, `float(v)` accepts `NaN`/`inf`; JSON round-trips them). A runner that crashes mid-compute and emits `{"fps": NaN}` stores it as a real measurement → poisons scaling/regression downstream. **Action:** reject/flag non-finite KPI values at coercion. · **Effort:** S

### ABE-5 ◐ `sweep` is fail-fast (`asyncio.gather` no `return_exceptions`) — partial results dropped
- **Severity:** Medium (documented behavior) · **Where:** `primitives/sweep.py` ◐ (~:50-59). One config's failure cancels the rest; no partial-sweep persistence. It's documented, but for regression sweeps partial results are often the point. **Action:** offer a `continue_on_error` mode that marks failed cells visibly and keeps the rest. · **Effort:** M

---

## 4. P2 — Secret-leak surface & backend contract

### ABK-7 ◐ `_safety.redact_secrets()` exists, is exported & tested — but **no primitive calls it**
- **Severity:** High · **Category:** Security (silent-leak failure mode) · **Where:** `_safety.py` (defined + re-exported) vs `primitives/{implement,plan,review}.py` (◐ no call). The module's own docstring calls it "the single chokepoint workflows call before rendering a captured blob into the next prompt or persisting via `corpus.put_artifact`" — but it's purely a *workflow* responsibility, unenforced. A workflow that forgets leaks secrets into the corpus + prompts with no warning.
- **Action:** decide the boundary — either (a) document loudly + add a leak-shaped-content warning at `put_artifact`/prompt-render, or (b) wire redaction into the capture path by default with an opt-out. At minimum, an enforced chokepoint, not honor-system. · **Effort:** S (doc/warn) / M (enforce)

### ABK-1 ◐ `Invocation` / `Message` prompt fields lack `repr=False` → secrets in tracebacks/debug dumps
- **Severity:** Medium · **Where:** `types/invocation.py` ◐ (`system`/`user`/`content` plain str fields). A traceback or SdkDebug dump reprs the full prompt — including any embedded credential. **Action:** `repr=False` on prompt-bearing fields (pairs with ABK-7). · **Effort:** S

### ABK-2 ◐ Backend protocol under-specifies the adapter contract (ARC-9b surface)
- **Severity:** Medium · **Where:** `backend/protocol.py` ◐ — `Backend` requires only `name` + `invoke`; no `model` attr, no policy/budget/max_turns conformance, `BackendError.kind` is a bare `str` not a closed Literal. A broken crucible adapter passes `isinstance` and fails only at runtime. **Action:** tighten the protocol (model attr, `kind` Literal) + add the **assay↔crucible contract tests** (ARC-9b) pinning what adapters must satisfy. · **Effort:** M

### ABK-8 ✅ `tool_cache.put` docstring says "first writer wins"; code is last-writer-wins
- **Severity:** Low · **Where:** `primitives/tool_cache.py` ◐ (~:56 docstring vs :58 `dict[key]=value`). Doc bug only. **Action:** fix the docstring. · **Effort:** S

---

## 5. P3 — QA, packaging, hygiene

- **AQA-3 ◐ No CI/lint/type gates** — like crucible, assay has no `.github/workflows`, no ruff/mypy. The substrate that everything builds on has no automated gate; the two pre-existing reds below sat undetected. Add CI (`pytest` + ruff) + the ARC-9b contract tests.
- **AQA-2 ◐ Python-floor mismatch** — assay `requires-python >=3.10`, crucible `>=3.12`. Confirm assay actually stays 3.10-compatible (no `tomllib`/3.11+ syntax) or raise the floor.
- **Pre-existing reds (both reproduced on clean assay `main`):**
  - `test_bench.py::test_bench_uses_raw_output_path_when_provided` — ◐ **root-caused as a Windows TEST bug** (the test embeds a tmp `Path` into an f-string of runner source; on Windows the backslashes form `\U...` → `unicodeescape` SyntaxError in the spawned runner → exit 1). Fix the test (`.as_posix()` / `repr()` / `json.dumps`), not bench.py. Passes on POSIX CI.
  - `test_validate_context_aware.py::test_ctx_rule_runs_when_ctx_passed` — ◐ two agents read the `validate()` arity-dispatch path as **correct**; likely the same Windows/env class. Confirm on Linux before treating as a code bug.
- **AQA-1 ◐ `corpus/runs.py` is dark** (RunHandle/RunResult, 31 LOC, no dedicated test). **ACOR-9 ◐** `_atomic_write` does tmp+rename but no dir-fsync (durability on hard crash). Both Low.
- **Git/genericness:** clean — `src/assay.egg-info` untracked, no committed debris, zero domain leakage in code (§13 holds).

---

## 6. Refuted agent claims (recorded so they aren't re-raised)

- **ACOR-4 ❌-REFUTED** ("`get_artifact` returns wrong bytes across kinds"): [store.py:686](../../src/assay/corpus/store.py) hash-verifies every read (`sha256(data) != hex_ → skip`), and the hex *is* the content hash, so same-hex ⇒ same-bytes. Worst case is a cross-kind `rglob` (a **Low perf** note — O(tree) per lookup), never wrong bytes.
- **ACOR-5 ❌-REFUTED** ("duplicate-extension artifacts inflate `list_artifacts`"): [store.py:630-643](../../src/assay/corpus/store.py) `put_artifact` writes a **fixed `<hex>.bin`** regardless of `name_hint`, with a comment stating exactly that rationale. No duplicate-extension path exists.
- **ACOR-1 ❌-REFUTED** ("`INSERT OR REPLACE` clobbers distinct measurements"): the SQLite primary key is `ref = hash_record(m)`, and `hash_record(Measurement)` **includes `execution.timestamp`** (verified: two measurements identical but for the timestamp produce different refs). So two runs never share a ref — `OR REPLACE` only overwrites a byte-identical re-ingest (idempotent), never a distinct measurement.
- **ACOR-3 ❌-REFUTED** ("`json.dumps` values vs canonical hash → divergence on re-ingest"): the `values_json` column is storage only, not the hash source; the ref is canonical `hash_record` of the float values. Finite floats (guaranteed by ABE-2) round-trip losslessly through `json.dumps`/`loads`, so a read-back measurement re-hashes to the same ref. No divergence. (Residual sliver: `put_measurements` bulk-ingest from a non-bench source could carry a non-finite value `json.dumps` emits as invalid-JSON `NaN`; the bench path is guarded — a Low note, not a divergence.)

---

## 7. Cross-correlation: assay roots of crucible findings

| crucible finding | assay root | fix altitude |
|---|---|---|
| ARC-2 cost-undercount / CRU-04 lost-work | **ACOR-2** (concurrent put_spec last-writer-wins) | fix in Corpus → fixes both consumers |
| RUB-5 fail-open severity (finding side, fixed) | **AREV-1/2** (floor side, unfixed) | normalize the floor → completes RUB-5 |
| RUB-3 parallel-conflict rubric drift | **APLAN-2** (no structural dag.py check) | enforce in `dag.py` → rubric becomes backstop |
| CRU-01 529 silent-noop (backend, fixed) | **APLAN-1** (planner can't tell error from bad-JSON) | discriminate in `plan.py` |

The pattern: several crucible-layer patches are treating symptoms whose root is one layer down.
The highest-leverage assay fixes (ACOR-2, AREV-1/2, APLAN-1/2) each retire a *class* across all
consumers.

---

## 8. Prioritized backlog

| # | Item | Sev | Effort | Verify |
|---|---|---|---|---|
| 1 | **AREV-1/2** normalize+validate `severity_floor` (completes RUB-5) | High | S | unit: typo'd floor rejected / fails closed |
| 2 | **APLAN-1** discriminate backend-error vs parse-failure in planner | High | S | unit: backend error not fed as "fix JSON" |
| 3 | **ABE-1/2** bench: raise on missing raw-output; reject non-finite KPIs | High | S | unit: missing path raises; NaN rejected |
| 4 | **AREV-5** rule exception → Critical finding, not crash | High | M | unit: throwing rule fails closed |
| 5 | **ABK-7/1** redaction chokepoint + `repr=False` on prompt fields | High/Med | S–M | unit: secret not in repr; leak path documented/enforced |
| 6 | **APLAN-2** structural parallel-tier disjointness check in `dag.py` | High | M | unit: overlapping same-tier scopes rejected |
| 7 | **ACOR-2** same-version `put_spec` CAS / lock / loud-error | Critical (latent) | M–L | concurrency test: N writers, no ref lost |
| 8 | **ACOR-1/3/6** measurement canonical-values + OR-REPLACE review; open_run tag+retry | Med | S–M | confirm `hash_record` coverage first |
| 9 | **AIMP-1/2** Windows path-sep normalize; clear intent-to-add index | Med | S | — (Windows) |
| 10 | **AQA-3 / ARC-9b** assay CI (pytest+ruff) + adapter contract tests | Med | M | CI green; contract test fails a broken stub |
| 11 | Fix the 2 pre-existing reds (test_bench Windows f-string; confirm validate on Linux) | Low | S | both green on Linux + Windows |

**Recommended order:** 1–5 are small, high-value, and several *complete work already started this
session* (AREV-1/2 finishes RUB-5; APLAN-1 mirrors CRU-01) → then 6 (APLAN-2, retires RUB-3 at
the root) → 7 (ACOR-2, the crown-jewel concurrency hardening, needs a real concurrency test) →
8–11. Items 1–6 are CPU-testable here via `cd assay && uv run pytest`; item 7 wants a
concurrency/stress test ideally on perflab1.

---

## 9. Execution pass (2026-06-10) — backlog landed on `main`

8 commits over `a32b513` (the audit was `c98c962`); full assay suite **331 passed / 1 skipped**,
`ruff check src` clean. Both pre-existing reds fixed in passing.

| # / IDs | Commit | Status |
|---|---|---|
| 1, 4, 8 — AREV-1/2/5/8 (gate floor + fail-closed rules + {artifact}) | `0089667` | ✅ done + tested |
| 2, 6 — APLAN-1/2 (backend-error discrimination + structural parallel-conflict gate) | `bea2a4e` | ✅ done + tested |
| 3, 8 — ABE-1/2, ACOR-6 (NaN/missing-raw-output; open_run retry) + bench red | `a35cee9` | ✅ done + tested |
| 7 — ACOR-2 (concurrent put_spec divergence → loud) | `441ae2a` | ✅ detection done + tested; true CAS-append deferred |
| 5, ARC-9b — ABK-1/8 (repr-exclude prompts; cache docstring; contract test) | `bfa25ff` | ✅ done + tested |
| 9 — AIMP-1/2 (Windows path-sep; clear intent-to-add) | `4abc0f5` | ✅ done + tested |
| 10, 11, AQA-1 — ruff+CI gate; runs.py test; ACOR-6 breach closure | `d8934ca` | ✅ done + tested |
| 11 — validate red (Windows path render) | `0089667` | ✅ fixed |

**Deliberately deferred (with rationale), not skipped:**
- **ACOR-2 true atomic CAS-append** — the loud-detection landed (silent loss → diagnosable
  error); a real compare-and-append API that *prevents* the exactly-simultaneous TOCTOU residue
  wants perflab1 concurrency/stress testing.
- **ACOR-1/3 (measurement SQLite) — VERIFIED, REFUTED (now in §6), not deferred.** The
  `hash_record(Measurement)` coverage check (probe: two measurements identical but for
  `execution.timestamp` → **different refs**) shows the ref/primary-key includes the timestamp,
  so distinct measurements never collide and `INSERT OR REPLACE` only overwrites byte-identical
  re-ingests (idempotent). Values round-trip losslessly (finite floats, guaranteed by ABE-2),
  and the ref is canonical `hash_record` of the float values (not the JSON string), so a
  read-back measurement re-hashes to the same ref. No clobber, no divergence.
- **APLAN-1/2 loop-level scripted-backend tests — DONE** (`b1163b2`): backend-error-≠-parse-failure
  and conflict-rejected-never-adopted, via `plan()` with the ScriptedBackend.
- **ABK-7 (enforce the redaction chokepoint at the capture path)** — a behavior/policy change
  (what gets stored/prompted); ABK-1's `repr=False` closed the incidental-leak vector.
- **ABK-2 (tighten the Backend protocol to require `model`)** — would break conforming adapters
  that don't declare it; pinned the *current* surface via the ARC-9b contract test instead.
- **ACOR-9 (dir-fsync after rename)** — Low; dir-fsync is a no-op on some Windows builds.
- **AQA-2 (python floor 3.10 vs crucible 3.12)** — left as declared support; ruff/tests pass.

---

## 10. Current disposition (2026-07-07)

This reconciliation supersedes the unresolved/deferred wording above without
rewriting the original evidence:

- **ACOR-2 is closed at the supported transaction boundary.**
  `Corpus.commit(...)` acquires the same cross-process/thread lock as
  `put_spec`, reads the latest Spec under that lock, deduplicates and appends
  staged refs, validates the candidate, and publishes one manifest. Direct
  stale or divergent `put_spec` calls fail with `ConcurrentSpecError`.
- **ACOR-9 is closed.** Both run-document replacement and Corpus atomic writes
  fsync the file and attempt a parent-directory fsync after publication.
- **ABK-2 is closed for the declared protocol.** `Backend` now requires public
  `name`, `model`, and `invoke` members; contract tests pin runtime structural
  narrowing for `Backend` and `ChatToolsBackend`.
- **APLAN-1/2 loop coverage is present.** Scripted-backend tests cover backend
  error discrimination and rejection of a conflicting candidate before it can
  become executable state.
- **ABK-7 remains an explicit policy boundary, not an automatic Corpus
  behavior.** `redact_secrets` is exported and documented, while callers own
  whether captured bytes must be redacted before persistence or model use.
  This avoids silently changing forensic evidence but requires workflow-level
  discipline.
- **AQA-2 remains intentional.** The package metadata declares Python 3.10+
  support; the higher Python floor of any one consumer is not Assay's floor.

All other items marked done in section 9 remain represented in the current
implementation and regression tests. Current test counts belong in CI output,
not this dated audit.
