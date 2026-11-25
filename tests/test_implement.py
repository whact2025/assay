# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for ``assay.primitives.implement``.

Covers the offline portions:
- Prompt construction (system + user from a directive)
- Subprocess-helper edge cases (porcelain parsing, missing git repo)
- Classification matrix (success / soft_fail / hard_fail variants)
- End-to-end with a stub Backend + real Corpus in a temp git repo

No real LLM is invoked; the backend is a stub returning canned data.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import importlib
import json
import os
import shlex
import stat
import subprocess
import sys
import threading
import types
from pathlib import Path

import pytest


from assay import (
    BudgetExhausted,
    ExecutionCompleteness,
    ImplementationDirective,
    Invocation,
    InvocationResult,
    Message,
    ProjectContext,
    ReadOnlySourceReference,
    Spec,
    TransportError,
    Usage,
    UsagePricingEvidence,
    UsagePricingSegment,
)
from assay.corpus.store import Corpus
from assay.primitives.implement import (
    GitCaptureError,
    _IMPLEMENTER_SYSTEM_PROMPT,
    _build_implementer_prompt,
    _capture_pathspecs,
    _classify,
    _git_diff_bytes,
    _git_head_sha,
    _git_modified_files,
    _serialize_transcript,
    capture_workspace_delta,
    implement,
)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


def _make_spec(project: str = "demo") -> Spec:
    return Spec(
        version=0,
        parent=None,
        root=None,
        schema_version=1,
        project_context=ProjectContext(
            project=project,
            repo_url="https://example.invalid/demo.git",
            base_commit="0" * 40,
            branch=None,
            submodule_path=None,
            runners={},
        ),
        execution=None,
        intent="exercise the implementer",
        constraints=(),
        done_criteria=(),
        plan_policy=None,
        depends_on=(),
        tier_dag=None,
        implementation_refs=(),
        verdict_refs=(),
        measurement_refs=(),
        lifecycle_event_refs=(),
    )


def _make_directive(
    slice_id: str = "S0",
    *,
    files_in_scope: tuple[str, ...] = ("hello.py",),
    creates_paths: tuple[str, ...] = (),
    blast_radius: dict | None = None,
    verify_mode: str = "build_only",
    requirement_refs: tuple[str, ...] = (),
) -> ImplementationDirective:
    return ImplementationDirective(
        slice_id=slice_id,
        intent="add hello() returning 'hi'",
        files_in_scope=files_in_scope,
        creates_paths=creates_paths,
        verify_mode=verify_mode,
        blast_radius=blast_radius or {},
        requirement_refs=requirement_refs,
    )


class StubBackend:
    """Minimal Backend-conformant stub."""

    name = "stub"
    model = "stub-model-v1"

    def __init__(
        self,
        *,
        final: str = "DONE — stubbed.",
        transcript: tuple[Message, ...] | None = None,
        error: TransportError | None = None,
        side_effect: callable | None = None,
        stop_reason: str = "end_turn",
    ) -> None:
        self._final = final
        self._transcript = transcript or (
            Message(role="user", content="ping"),
            Message(role="assistant", content=final),
        )
        self._error = error
        self._side_effect = side_effect
        self._stop_reason = stop_reason
        self.invocations: list[Invocation] = []

    def invoke(self, invocation: Invocation) -> InvocationResult:
        self.invocations.append(invocation)
        if self._side_effect is not None:
            self._side_effect()
        return InvocationResult(
            final_message=self._final,
            transcript=self._transcript,
            usage=Usage(input_tokens=10, output_tokens=20, cost_usd=0.01),
            stop_reason=self._stop_reason,
            error=self._error,
        )


# ----------------------------------------------------------------------
# Prompt construction
# ----------------------------------------------------------------------


def test_system_prompt_is_stable_string():
    assert "implementation agent" in _IMPLEMENTER_SYSTEM_PROMPT.lower()
    assert "DONE" in _IMPLEMENTER_SYSTEM_PROMPT


def test_user_prompt_includes_directive_fields():
    prompt = _build_implementer_prompt(_make_spec(), "S1", _make_directive("S1"))
    assert "S1" in prompt
    assert "hello.py" in prompt
    assert "build_only" in prompt
    assert "demo" in prompt
    assert "hi" in prompt  # intent body


def test_user_prompt_places_authoritative_requirements_before_generated_intent():
    clause_one = "The emitted records must retain their original order."
    clause_two = "Duplicate records must be rejected."
    directive = _make_directive(
        requirement_refs=(clause_one, clause_two),
    )

    prompt = _build_implementer_prompt(_make_spec(), "S1", directive)

    assert "exact, verbatim requirements from the task root" in prompt
    assert "win over the generated implementation intent" in prompt
    assert prompt.index(clause_one) < prompt.index(clause_two)
    assert prompt.index(clause_two) < prompt.index(directive.intent)
    assert "## Generated implementation intent" in prompt


def test_user_prompt_omits_root_requirements_section_for_legacy_directive():
    prompt = _build_implementer_prompt(_make_spec(), "S1", _make_directive())

    assert "## Authoritative root requirements" not in prompt
    assert "## Intent" in prompt


def test_user_prompt_handles_empty_files_in_scope():
    d = _make_directive(files_in_scope=())
    prompt = _build_implementer_prompt(_make_spec(), "S2", d)
    assert "none declared" in prompt


def test_user_prompt_handles_creates_paths():
    d = _make_directive(creates_paths=("new/file.py",))
    prompt = _build_implementer_prompt(_make_spec(), "S3", d)
    assert "new/file.py" in prompt


def test_user_prompt_handles_blast_radius():
    d = _make_directive(blast_radius={"max_files": 3, "max_lines": 50})
    prompt = _build_implementer_prompt(_make_spec(), "S4", d)
    assert "max_files" in prompt
    assert "Blast-radius" in prompt


def test_user_prompt_omits_mutation_regions_when_empty():
    """Empty mutation_regions → no Mutation regions section in the
    user-facing prompt."""
    d = _make_directive()  # defaults: no mutation_regions
    prompt = _build_implementer_prompt(_make_spec(), "S5", d)
    assert "Mutation regions" not in prompt
    assert "CRUCIBLE-MUTATE" not in prompt


def test_user_prompt_renders_mutation_regions_when_populated():
    """When mutation_regions is populated, the prompt spells out the
    per-file marker constraint so the implementer LLM sees the
    scoping policy at authoring time."""
    from assay import MutationRegion
    d = _make_directive(
        files_in_scope=("src/bvh.rs", "src/lib.rs"),
    )
    # _make_directive helper doesn't accept mutation_regions, so build
    # an augmented copy via dataclasses.replace.
    import dataclasses as dc
    d = dc.replace(
        d,
        mutation_regions=(
            MutationRegion(file="src/bvh.rs"),
            MutationRegion(
                file="src/lib.rs",
                start_marker="OPTIM-START",
                end_marker="OPTIM-END",
            ),
        ),
    )
    prompt = _build_implementer_prompt(_make_spec(), "S6", d)
    assert "Mutation regions" in prompt
    assert "`src/bvh.rs`" in prompt
    assert "CRUCIBLE-MUTATE-START" in prompt
    assert "OPTIM-START" in prompt
    assert "rejected by the mutation_scope validator" in prompt


# ----------------------------------------------------------------------
# Subprocess helpers — non-git working directory
# ----------------------------------------------------------------------


def test_git_helpers_safe_outside_repo(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert _git_head_sha() == "unknown"
    assert _git_diff_bytes() == b""
    assert _git_modified_files() == ()


# ----------------------------------------------------------------------
# Subprocess helpers — real temp git repo
# ----------------------------------------------------------------------


def _init_repo(path: Path) -> str:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@assay"], cwd=path, check=True
    )
    subprocess.run(
        ["git", "config", "user.name", "assay test"], cwd=path, check=True
    )
    (path / "seed.txt").write_text("seed\n")
    subprocess.run(["git", "add", "seed.txt"], cwd=path, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "init"], cwd=path, check=True
    )
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return sha


def _record_git_capture_scratch(monkeypatch):
    """Observe real scratch paths without replacing their cleanup policy."""

    implement_module = importlib.import_module("assay.primitives.implement")
    original = implement_module._git_capture_scratch
    observed: list[dict[str, object]] = []

    @contextlib.contextmanager
    def recording_scratch(worktree_root):
        with original(worktree_root) as scratch:
            observed.append({
                "path": scratch,
                "parent": scratch.parent,
                "device": scratch.stat().st_dev,
                "worktree": worktree_root,
            })
            yield scratch

    monkeypatch.setattr(
        implement_module,
        "_git_capture_scratch",
        recording_scratch,
    )
    return implement_module, observed


def test_git_head_sha_in_real_repo(tmp_path: Path, monkeypatch):
    sha = _init_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert _git_head_sha() == sha


def test_git_modified_files_detects_new_file(tmp_path: Path, monkeypatch):
    _init_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "hello.py").write_text("def hi(): return 'hi'\n")
    files = _git_modified_files()
    assert "hello.py" in files


def test_git_modified_files_detects_modified_tracked(tmp_path: Path, monkeypatch):
    _init_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "seed.txt").write_text("seed\nmodified\n")
    files = _git_modified_files()
    assert "seed.txt" in files


def test_git_diff_bytes_nonempty_on_change(tmp_path: Path, monkeypatch):
    _init_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "seed.txt").write_text("seed\nchanged\n")
    diff = _git_diff_bytes()
    assert diff != b""
    assert b"changed" in diff or b"+changed" in diff


def test_git_diff_scratch_ignores_poisoned_tmpdir_and_cleans_success(
    tmp_path: Path,
    monkeypatch,
):
    workspace = tmp_path / "workspace"
    repo = workspace / "repo"
    repo.mkdir(parents=True)
    _init_repo(repo)
    (repo / "seed.txt").write_text("seed\nchanged\n")
    poisoned_tmpdir = workspace / "poisoned-tmpdir"
    poisoned_tmpdir.write_text("not a directory")
    implement_module, observed = _record_git_capture_scratch(monkeypatch)
    # Simulate both a poisoned TMPDIR environment and tempfile's cached global
    # default. An implicit TemporaryDirectory would fail with ENOTDIR here.
    monkeypatch.setenv("TMPDIR", str(poisoned_tmpdir))
    monkeypatch.setattr(
        implement_module.tempfile,
        "tempdir",
        str(poisoned_tmpdir),
    )

    diff = _git_diff_bytes(cwd=str(repo), path_scope=("seed.txt",))

    assert b"changed" in diff
    assert len(observed) == 1
    scratch = observed[0]["path"]
    assert scratch.parent == workspace.resolve()
    assert not scratch.is_relative_to(repo.resolve())
    assert observed[0]["device"] == repo.stat().st_dev
    assert not scratch.exists()
    assert poisoned_tmpdir.read_text() == "not a directory"


def test_git_diff_scratch_cleans_after_capture_failure(
    tmp_path: Path,
    monkeypatch,
):
    workspace = tmp_path / "workspace"
    repo = workspace / "repo"
    repo.mkdir(parents=True)
    _init_repo(repo)
    (repo / "seed.txt").write_text("seed\nchanged\n")
    _implement_module, observed = _record_git_capture_scratch(monkeypatch)
    real_run = subprocess.run

    def fail_diff(command, *args, **kwargs):
        if "diff" in command:
            return subprocess.CompletedProcess(
                command,
                1,
                stdout=b"",
                stderr=b"injected diff failure",
            )
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", fail_diff)

    with pytest.raises(GitCaptureError, match="injected diff failure"):
        _git_diff_bytes(cwd=str(repo), path_scope=("seed.txt",))

    assert len(observed) == 1
    scratch = observed[0]["path"]
    assert scratch.parent == workspace.resolve()
    assert not scratch.exists()


def test_git_diff_scratch_rejects_escape_from_workspace_parent(
    tmp_path: Path,
    monkeypatch,
):
    workspace = tmp_path / "workspace"
    repo = workspace / "repo"
    repo.mkdir(parents=True)
    _init_repo(repo)
    (repo / "seed.txt").write_text("seed\nchanged\n")
    implement_module = importlib.import_module("assay.primitives.implement")
    escaped = tmp_path / "escaped-scratch"

    @contextlib.contextmanager
    def escaping_temporary_directory(*, prefix, dir):
        assert prefix == ".assay-git-capture-"
        assert Path(dir) == workspace.resolve()
        escaped.mkdir()
        try:
            yield str(escaped)
        finally:
            escaped.rmdir()

    monkeypatch.setattr(
        implement_module.tempfile,
        "TemporaryDirectory",
        escaping_temporary_directory,
    )

    with pytest.raises(GitCaptureError, match="escaped its selected parent"):
        _git_diff_bytes(cwd=str(repo), path_scope=("seed.txt",))

    assert not escaped.exists()


def test_git_diff_scratch_rejects_cross_device_directory(
    tmp_path: Path,
    monkeypatch,
):
    workspace = tmp_path / "workspace"
    repo = workspace / "repo"
    repo.mkdir(parents=True)
    _init_repo(repo)
    implement_module = importlib.import_module("assay.primitives.implement")
    scratch = workspace / ".assay-git-capture-fake"
    real_stat = Path.stat

    @contextlib.contextmanager
    def cross_device_temporary_directory(*, prefix, dir):
        assert prefix == ".assay-git-capture-"
        assert Path(dir) == workspace.resolve()
        scratch.mkdir()
        try:
            yield str(scratch)
        finally:
            scratch.rmdir()

    def cross_device_stat(path, *args, **kwargs):
        observed = real_stat(path, *args, **kwargs)
        if path == scratch:
            return types.SimpleNamespace(st_dev=observed.st_dev + 1)
        return observed

    monkeypatch.setattr(
        implement_module.tempfile,
        "TemporaryDirectory",
        cross_device_temporary_directory,
    )
    monkeypatch.setattr(Path, "stat", cross_device_stat)

    with pytest.raises(GitCaptureError, match="crossed the worktree filesystem"):
        with implement_module._git_capture_scratch(repo):
            pytest.fail("cross-device scratch must not be yielded")

    assert not scratch.exists()


def test_git_diff_scratch_supports_read_only_git_admin(
    tmp_path: Path,
    monkeypatch,
):
    workspace = tmp_path / "workspace"
    repo = workspace / "repo"
    repo.mkdir(parents=True)
    _init_repo(repo)
    (repo / "seed.txt").write_text("seed\nchanged\n")
    git_admin = repo / ".git"
    original_mode = stat.S_IMODE(git_admin.stat().st_mode)
    git_admin.chmod(0o555)
    _implement_module, observed = _record_git_capture_scratch(monkeypatch)

    try:
        diff = _git_diff_bytes(cwd=str(repo), path_scope=("seed.txt",))
    finally:
        git_admin.chmod(original_mode)

    assert b"changed" in diff
    assert len(observed) == 1
    scratch = observed[0]["path"]
    assert scratch.parent == workspace.resolve()
    assert not scratch.exists()


def test_git_diff_scratch_never_trusts_symlinked_git_admin(
    tmp_path: Path,
    monkeypatch,
):
    workspace = tmp_path / "workspace"
    repo = workspace / "repo"
    repo.mkdir(parents=True)
    _init_repo(repo)
    git_admin = repo / ".git"
    relocated_admin = repo / "admin"
    git_admin.rename(relocated_admin)
    git_admin.symlink_to("admin", target_is_directory=True)
    implement_module, observed = _record_git_capture_scratch(monkeypatch)

    diff = _git_diff_bytes(cwd=str(repo))

    assert implement_module._literal_git_admin_directory(repo.resolve()) is None
    assert len(observed) == 1
    scratch = observed[0]["path"]
    assert scratch.parent == workspace.resolve()
    assert not scratch.exists()
    assert b".assay-git-capture-" not in diff


def test_linked_worktree_scratch_stays_outside_enclosing_repo(
    tmp_path: Path,
):
    source = tmp_path / "source"
    outer = tmp_path / "outer"
    source.mkdir()
    outer.mkdir()
    _init_repo(source)
    _init_repo(outer)
    linked = outer / "linked"
    subprocess.run(
        [
            "git", "worktree", "add", "-q", "-b", "scratch-audit", str(linked),
        ],
        cwd=source,
        check=True,
    )
    implement_module = importlib.import_module("assay.primitives.implement")

    with implement_module._git_capture_scratch(linked) as scratch:
        sentinel = b"linked scratch sentinel\n"
        (scratch / "index").write_bytes(sentinel)
        outer_diff = _git_diff_bytes(cwd=str(outer))
        assert scratch.parent == tmp_path.resolve()
        assert not scratch.is_relative_to(outer.resolve())
        assert b".assay-git-capture-" not in outer_diff
        assert sentinel not in outer_diff

    assert not scratch.exists()


def test_git_admin_compatibility_fallback_is_literal_and_cleanup_safe(
    tmp_path: Path,
    monkeypatch,
):
    outer = tmp_path / "outer"
    repo = outer / "repo"
    outer.mkdir()
    repo.mkdir()
    _init_repo(outer)
    _init_repo(repo)
    implement_module = importlib.import_module("assay.primitives.implement")
    real_temporary_directory = implement_module.tempfile.TemporaryDirectory
    attempted: list[Path] = []

    def selective_temporary_directory(*, prefix, dir):
        parent = Path(dir)
        attempted.append(parent)
        if parent == tmp_path.resolve():
            raise PermissionError("external workspace parent is read-only")
        return real_temporary_directory(prefix=prefix, dir=dir)

    monkeypatch.setattr(
        implement_module.tempfile,
        "TemporaryDirectory",
        selective_temporary_directory,
    )

    with implement_module._git_capture_scratch(repo) as scratch:
        assert scratch.parent == (repo / ".git").resolve()
        assert scratch.is_relative_to(repo / ".git")
        sentinel = b"fallback scratch sentinel\n"
        (scratch / "index").write_bytes(sentinel)
        monkeypatch.setattr(
            implement_module.tempfile,
            "TemporaryDirectory",
            real_temporary_directory,
        )
        outer_diff = _git_diff_bytes(cwd=str(outer))
        assert b".assay-git-capture-" not in outer_diff
        assert sentinel not in outer_diff

    assert attempted == [tmp_path.resolve(), (repo / ".git").resolve()]
    assert not scratch.exists()


def test_implement_pre_and_post_capture_use_workspace_scratch(
    tmp_path: Path,
    monkeypatch,
):
    workspace = tmp_path / "workspace"
    repo = workspace / "repo"
    repo.mkdir(parents=True)
    _init_repo(repo)
    _implement_module, observed = _record_git_capture_scratch(monkeypatch)

    result = implement(
        _make_spec(),
        "S0",
        _make_directive(files_in_scope=("seed.txt",)),
        backend=StubBackend(
            side_effect=lambda: (repo / "seed.txt").write_text(
                "seed\nchanged\n"
            ),
        ),
        corpus=Corpus(tmp_path / "corpus"),
        cwd=str(repo),
    )

    assert result.classification == "success"
    # The immediate pre-scope snapshot and post-invocation capture both route
    # through the same explicit workspace scratch policy.
    assert len(observed) == 2
    for record in observed:
        scratch = record["path"]
        assert scratch.parent == workspace.resolve()
        assert not scratch.is_relative_to(repo.resolve())
        assert record["device"] == repo.stat().st_dev
        assert not scratch.exists()


def test_git_diff_bytes_preserves_racy_clean_detection(
    tmp_path: Path,
    monkeypatch,
):
    _init_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    subprocess.run(
        ["git", "config", "core.trustctime", "false"],
        cwd=tmp_path,
        check=True,
    )
    tracked = tmp_path / "seed.txt"
    fixed_ns = 1_700_000_000_000_000_000
    tracked_stat = tracked.stat()
    os.utime(tracked, ns=(tracked_stat.st_atime_ns, fixed_ns))
    subprocess.run(
        ["git", "update-index", "--refresh"],
        cwd=tmp_path,
        check=True,
    )
    index = Path(
        subprocess.run(
            [
                "git", "rev-parse", "--path-format=absolute",
                "--git-path", "index",
            ],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    index_stat = index.stat()
    os.utime(index, ns=(index_stat.st_atime_ns, fixed_ns))

    # Preserve size and mtime, and disable ctime trust above, reproducing the
    # exact case where only Git's racy-clean index timestamp forces hashing.
    tracked.write_text("SEED\n", encoding="utf-8")
    rewritten_stat = tracked.stat()
    os.utime(tracked, ns=(rewritten_stat.st_atime_ns, fixed_ns))

    diff = _git_diff_bytes(path_scope=("seed.txt",))

    assert b"-seed" in diff
    assert b"+SEED" in diff


def test_git_diff_bytes_uses_isolated_index_and_object_store(
    tmp_path: Path,
    monkeypatch,
):
    """New-file capture must not transiently mutate repository metadata."""

    _init_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "seed.txt").write_text("seed\nstaged-user-change\n")
    subprocess.run(["git", "add", "seed.txt"], cwd=tmp_path, check=True)
    (tmp_path / "new_file.py").write_text("print('new')\n")

    index = Path(
        subprocess.run(
            [
                "git", "rev-parse", "--path-format=absolute",
                "--git-path", "index",
            ],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    objects = Path(
        subprocess.run(
            [
                "git", "rev-parse", "--path-format=absolute",
                "--git-path", "objects",
            ],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )

    def object_inventory() -> dict[str, tuple[int, int, int, int, int]]:
        inventory = {}
        for path in objects.rglob("*"):
            if path.is_dir():
                continue
            metadata = path.lstat()
            inventory[path.relative_to(objects).as_posix()] = (
                metadata.st_mode,
                metadata.st_nlink,
                metadata.st_size,
                metadata.st_mtime_ns,
                metadata.st_ctime_ns,
            )
        return inventory

    before_index = index.read_bytes()
    before_mode = index.stat().st_mode
    before_objects = object_inventory()

    diff = _git_diff_bytes(path_scope=("seed.txt", "new_file.py"))

    assert b"staged-user-change" in diff
    assert b"new_file.py" in diff
    assert index.read_bytes() == before_index
    assert index.stat().st_mode == before_mode
    assert object_inventory() == before_objects
    assert subprocess.run(
        ["git", "ls-files", "--stage", "--", "new_file.py"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout == ""

    # Public capture also asks for modified paths. Force a stale stat-cache
    # entry and prove the read-side status query cannot refresh the live index.
    metadata = (tmp_path / "seed.txt").stat()
    os.utime(
        tmp_path / "seed.txt",
        ns=(metadata.st_atime_ns, metadata.st_mtime_ns + 2_000_000_000),
    )
    before_public_index = index.read_bytes()
    public_diff, public_files, _ = capture_workspace_delta(
        cwd=str(tmp_path),
        path_scope=("seed.txt", "new_file.py"),
    )
    assert b"new_file.py" in public_diff
    assert set(public_files) == {"seed.txt", "new_file.py"}
    assert index.read_bytes() == before_public_index
    assert object_inventory() == before_objects


def test_git_diff_bytes_preserves_split_index(tmp_path: Path, monkeypatch):
    """The temporary index carries, but never rewrites, split-index state."""

    _init_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    subprocess.run(
        ["git", "update-index", "--split-index"],
        cwd=tmp_path,
        check=True,
    )
    (tmp_path / "new_file.py").write_text("print('new')\n")
    index = Path(
        subprocess.run(
            [
                "git", "rev-parse", "--path-format=absolute",
                "--git-path", "index",
            ],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    shared = Path(
        subprocess.run(
            ["git", "rev-parse", "--shared-index-path"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    before_index = index.read_bytes()
    before_shared = shared.read_bytes()

    diff = _git_diff_bytes(path_scope=("new_file.py",))

    assert b"new_file.py" in diff
    assert index.read_bytes() == before_index
    assert shared.read_bytes() == before_shared


def test_git_diff_bytes_treats_unusual_filenames_literally(
    tmp_path: Path,
    monkeypatch,
):
    """NUL parsing and literal pathspecs retain exact planner path names."""

    _init_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    newline_name = "new\nfile.txt"
    colon_name = ":(exclude)not-magic.txt"
    backslash_name = "literal\\backslash.txt"
    (tmp_path / newline_name).write_text("newline-name-marker\n")
    (tmp_path / colon_name).write_text("colon-name-marker\n")
    (tmp_path / backslash_name).write_text("backslash-name-marker\n")

    diff = _git_diff_bytes(
        path_scope=(newline_name, colon_name, backslash_name)
    )

    assert b"newline-name-marker" in diff
    assert b"colon-name-marker" in diff
    assert b"backslash-name-marker" in diff
    assert set(
        _git_modified_files(
            path_scope=(newline_name, colon_name, backslash_name)
        )
    ) == {
        newline_name,
        colon_name,
        backslash_name,
    }
    public_diff, public_files, _ = capture_workspace_delta(
        cwd=str(tmp_path),
        path_scope=(newline_name, colon_name, backslash_name),
    )
    assert b"backslash-name-marker" in public_diff
    assert set(public_files) == {newline_name, colon_name, backslash_name}


def test_git_capture_rejects_filter_without_running_clean_driver(
    tmp_path: Path,
    monkeypatch,
):
    """A model-written attribute cannot turn capture into code execution."""

    _init_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    sentinel = tmp_path / "clean-filter-ran"
    clean_driver = tmp_path / "sentinel_clean.py"
    clean_driver.write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "Path(sys.argv[1]).write_text('ran')\n"
        "sys.stdout.buffer.write(sys.stdin.buffer.read())\n"
    )
    subprocess.run(
        [
            "git",
            "config",
            "filter.sentinel.clean",
            shlex.join((sys.executable, str(clean_driver), str(sentinel))),
        ],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "filter.sentinel.required", "true"],
        cwd=tmp_path,
        check=True,
    )

    # Both mutations happen after the configured driver exists, matching a
    # model that activates a local/global/system driver through attributes.
    (tmp_path / ".gitattributes").write_text("*.txt filter=sentinel\n")
    (tmp_path / "seed.txt").write_text("seed\nchanged\n")

    with pytest.raises(GitCaptureError, match="repository-local filter/include"):
        _git_diff_bytes(path_scope=("seed.txt", ".gitattributes"))
    assert not sentinel.exists()

    with pytest.raises(GitCaptureError, match="repository-local filter/include"):
        _git_modified_files(path_scope=("seed.txt", ".gitattributes"))
    assert not sentinel.exists()


def test_git_capture_rejects_worktree_filter_config(
    tmp_path: Path,
    monkeypatch,
):
    _init_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    subprocess.run(
        ["git", "config", "extensions.worktreeConfig", "true"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "--worktree", "FiLtEr.Hostile.ClEaN", "false"],
        cwd=tmp_path,
        check=True,
    )
    (tmp_path / ".gitattributes").write_text("seed.txt filter=hostile\n")
    (tmp_path / "seed.txt").write_text("seed\nchanged\n")

    with pytest.raises(GitCaptureError, match="repository-local filter/include"):
        _git_modified_files(path_scope=("seed.txt", ".gitattributes"))


def test_git_capture_rejects_conditional_config_include(
    tmp_path: Path,
    monkeypatch,
):
    _init_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    included = tmp_path / "included.gitconfig"
    included.write_text("[filter \"hostile\"]\n\tclean = false\n")
    subprocess.run(
        [
            "git",
            "config",
            "--local",
            f"InClUdEiF.gitdir:{tmp_path.as_posix()}/.path",
            str(included),
        ],
        cwd=tmp_path,
        check=True,
    )

    with pytest.raises(GitCaptureError, match="repository-local filter/include"):
        _git_modified_files(path_scope=("seed.txt",))


def test_git_capture_allows_inert_filter_attribute_and_ignores_global_driver(
    tmp_path: Path,
    monkeypatch,
):
    """An attribute is safe when capture has no executable driver for it."""

    _init_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    sentinel = tmp_path / "global-clean-filter-ran"
    clean_driver = tmp_path / "global_clean.py"
    clean_driver.write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "Path(sys.argv[1]).write_text('ran')\n"
        "sys.stdout.buffer.write(sys.stdin.buffer.read())\n"
    )
    global_config = tmp_path / "attacker-global.gitconfig"
    subprocess.run(
        [
            "git",
            "config",
            "--file",
            str(global_config),
            "filter.opaque-store.clean",
            shlex.join((sys.executable, str(clean_driver), str(sentinel))),
        ],
        check=True,
    )
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(global_config))

    (tmp_path / ".gitattributes").write_text(
        "seed.txt filter=opaque-store\n"
    )
    (tmp_path / "seed.txt").write_text("seed\nchanged\n")

    diff = _git_diff_bytes(path_scope=("seed.txt", ".gitattributes"))
    assert b"changed" in diff
    assert set(
        _git_modified_files(path_scope=("seed.txt", ".gitattributes"))
    ) == {"seed.txt", ".gitattributes"}
    assert not sentinel.exists()


def test_git_capture_ignores_home_attributes(tmp_path: Path, monkeypatch):
    _init_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    hostile_home = tmp_path / "hostile-home"
    attributes = hostile_home / ".config" / "git" / "attributes"
    attributes.parent.mkdir(parents=True)
    attributes.write_text(
        "seed.txt working-tree-encoding=not-a-real-encoding\n"
    )
    monkeypatch.setenv("HOME", str(hostile_home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(hostile_home / ".config"))
    (tmp_path / "seed.txt").write_text("seed\nchanged\n")

    diff = _git_diff_bytes(path_scope=("seed.txt",))
    assert b"changed" in diff
    assert _git_modified_files(path_scope=("seed.txt",)) == ("seed.txt",)


def _sentinel_git_command(tmp_path: Path, sentinel: Path) -> str:
    driver = tmp_path / f"driver-{sentinel.name}.py"
    driver.write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "Path(sys.argv[1]).write_text('git executed configured command')\n"
    )
    return shlex.join((sys.executable, str(driver), str(sentinel)))


def test_git_capture_ignores_global_and_ambient_executable_config(
    tmp_path: Path,
    monkeypatch,
):
    """Host-level Git config cannot execute code during capture."""

    _init_repo(tmp_path)
    global_sentinel = tmp_path / "global-git-command-ran"
    ambient_sentinel = tmp_path / "ambient-diff-ran"
    global_command = _sentinel_git_command(tmp_path, global_sentinel)
    ambient_command = _sentinel_git_command(tmp_path, ambient_sentinel)
    global_config = tmp_path / "attacker-global.gitconfig"
    subprocess.run(
        ["git", "config", "--file", str(global_config), "core.fsmonitor",
         global_command],
        check=True,
    )
    subprocess.run(
        ["git", "config", "--file", str(global_config),
         "diff.sentinel.command", global_command],
        check=True,
    )
    subprocess.run(
        ["git", "config", "--file", str(global_config),
         "diff.sentinel.textconv", global_command],
        check=True,
    )
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(global_config))
    monkeypatch.setenv("GIT_EXTERNAL_DIFF", ambient_command)
    monkeypatch.setenv("GIT_ALTERNATE_OBJECT_DIRECTORIES", "/attacker/objects")
    (tmp_path / ".gitattributes").write_text("*.txt diff=sentinel\n")
    (tmp_path / "seed.txt").write_text("seed\nchanged\n")

    diff, files, _ = capture_workspace_delta(
        cwd=str(tmp_path),
        path_scope=("seed.txt", ".gitattributes"),
    )

    assert b"changed" in diff
    assert set(files) == {"seed.txt", ".gitattributes"}
    assert not global_sentinel.exists()
    assert not ambient_sentinel.exists()


def test_git_capture_disables_local_fsmonitor_diff_and_textconv(
    tmp_path: Path,
):
    """Repository-local executable facilities are neutralized per command."""

    _init_repo(tmp_path)
    sentinel = tmp_path / "local-git-command-ran"
    command = _sentinel_git_command(tmp_path, sentinel)
    for key in (
        "core.fsmonitor",
        "diff.sentinel.command",
        "diff.sentinel.textconv",
    ):
        subprocess.run(
            ["git", "config", key, command],
            cwd=tmp_path,
            check=True,
        )
    (tmp_path / ".gitattributes").write_text("*.txt diff=sentinel\n")
    (tmp_path / "seed.txt").write_text("seed\nchanged\n")

    diff, files, _ = capture_workspace_delta(
        cwd=str(tmp_path),
        path_scope=("seed.txt", ".gitattributes"),
    )

    assert b"changed" in diff
    assert set(files) == {"seed.txt", ".gitattributes"}
    assert not sentinel.exists()


def test_git_capture_streaming_bound_rejects_oversized_patch(tmp_path: Path):
    _init_repo(tmp_path)
    (tmp_path / "seed.txt").write_text("changed\n" * 4096)

    with pytest.raises(GitCaptureError, match="max_diff_bytes=256"):
        capture_workspace_delta(
            cwd=str(tmp_path),
            path_scope=("seed.txt",),
            max_diff_bytes=256,
        )


def test_git_capture_streaming_bound_preserves_exact_patch(tmp_path: Path):
    _init_repo(tmp_path)
    (tmp_path / "seed.txt").write_text("seed\nchanged\n")

    unbounded = capture_workspace_delta(
        cwd=str(tmp_path),
        path_scope=("seed.txt",),
    )
    bounded = capture_workspace_delta(
        cwd=str(tmp_path),
        path_scope=("seed.txt",),
        max_diff_bytes=len(unbounded[0]),
    )

    assert bounded == unbounded


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "10"])
def test_git_capture_rejects_invalid_streaming_bound(
    tmp_path: Path,
    value,
):
    _init_repo(tmp_path)

    with pytest.raises(ValueError, match="max_diff_bytes"):
        capture_workspace_delta(cwd=str(tmp_path), max_diff_bytes=value)


def test_git_capture_allows_text_and_eol_attributes(
    tmp_path: Path,
    monkeypatch,
):
    """Benign normalization attributes do not disable ordinary capture."""

    _init_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".gitattributes").write_text("*.txt text eol=lf\n")
    (tmp_path / "seed.txt").write_bytes(b"seed\r\nchanged\r\n")

    diff, files, _ = capture_workspace_delta(
        cwd=str(tmp_path),
        path_scope=("seed.txt", ".gitattributes"),
    )

    assert b"changed" in diff
    assert set(files) == {"seed.txt", ".gitattributes"}


@pytest.mark.parametrize(
    "failed_command",
    [
        "--is-inside-work-tree",
        "--show-toplevel",
        "ls-files",
        "check-attr",
        "add",
        "diff",
    ],
)
def test_git_diff_bytes_fails_closed_on_git_capture_error(
    tmp_path: Path,
    monkeypatch,
    failed_command: str,
):
    """A partial patch is never returned when isolated Git plumbing fails."""

    _init_repo(tmp_path)
    (tmp_path / "new_file.py").write_text("print('new')\n")
    real_run = subprocess.run

    def injected_failure(command, *args, **kwargs):
        if failed_command in command:
            return subprocess.CompletedProcess(
                command,
                1,
                stdout=b"",
                stderr=b"injected capture failure",
            )
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", injected_failure)

    with pytest.raises(GitCaptureError, match="failed with exit 1"):
        capture_workspace_delta(
            cwd=str(tmp_path),
            path_scope=("new_file.py",),
        )


def test_git_modified_files_expands_untracked_dir_to_files(tmp_path: Path, monkeypatch):
    """A brand-new directory of untracked files must report at FILE
    granularity, not collapse to a single ``dir/`` entry.

    porcelain-v1 defaults to collapsing an entirely-untracked directory to
    one ``dir/`` line; ``-uall`` expands it. The unscoped scope-escape
    capture relies on this so a greenfield new-crate slice (all files under
    a new dir, each in creates_paths) doesn't read as one out-of-scope
    write and trip ``hard_fail_scope_escape``.
    """
    _init_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "crates" / "newcrate" / "src").mkdir(parents=True)
    (tmp_path / "crates" / "newcrate" / "Cargo.toml").write_text("[package]\n")
    (tmp_path / "crates" / "newcrate" / "src" / "lib.rs").write_text("// new\n")
    files = _git_modified_files()
    assert "crates/newcrate/Cargo.toml" in files
    assert "crates/newcrate/src/lib.rs" in files
    # The collapsed directory form must NOT appear.
    assert "crates/newcrate/" not in files


# ----------------------------------------------------------------------
# Classification
# ----------------------------------------------------------------------


def test_classify_success():
    assert _classify(None, ("a.py",), "build_only") == "success"


def test_classify_soft_fail_only_for_skip_verify_mode():
    # Audit-only slices (verify_mode=="skip") legitimately produce no
    # diff — keep the soft classification.
    assert _classify(None, (), "skip") == "soft_fail_no_changes"


def test_classify_hard_fail_empty_diff_for_mutating_slices():
    # A mutating directive (verify_mode != "skip") that produced no
    # files_modified is a hard fail — escalates the prior silent
    # "soft_fail_no_changes" surface so misconfigured slices don't
    # mask as success-adjacent downstream.
    assert _classify(None, (), "build_only") == "hard_fail_empty_diff"
    assert _classify(None, (), "tests") == "hard_fail_empty_diff"


def test_classify_hard_fail_includes_error_kind():
    # error_kind takes precedence over verify_mode — a transport error
    # with NO files written is always a hard fail.
    assert _classify("transport", (), "build_only") == "hard_fail_transport"
    assert _classify("transport", (), "skip") == "hard_fail_transport"
    # Non-transport errors stay a hard fail even when files were written
    # (a cancellation mid-edit leaves a partial, untrusted state).
    assert _classify("cancellation", ("a.py",), "build_only") == "hard_fail_cancellation"


def test_classify_transport_with_files_remains_incomplete():
    # Provider errors can arrive after tool calls wrote a partial diff. The
    # files remain captured as evidence, but the invocation did not complete
    # and therefore cannot become a successful implementation.
    assert (
        _classify("transport", ("a.py",), "build_only")
        == "hard_fail_transport"
    )
    assert _classify("transport", ("a.py",), "tests") == "hard_fail_transport"
    assert _classify(
        "transport", ("a.py",), "build_only", scope_escapes=("b.py",),
    ) == "hard_fail_transport"


def test_classify_hard_fail_scope_escape_escalates_orphan_writes():
    """When the implementer wrote outside its declared scope, the
    classification escalates to ``hard_fail_scope_escape`` so the
    workflow rolls back the full modification set in the same run.
    Prior behavior left the orphan files on disk for the reaper to
    catch on the NEXT invocation — which didn't help the in-flight
    slice cascade that depended on a clean baseline."""
    # In-scope file modified + out-of-scope file modified → escape
    # classification regardless of whether the in-scope work itself
    # was sensible.
    assert _classify(
        None, ("a.py",), "build_only", scope_escapes=("b.py",),
    ) == "hard_fail_scope_escape"


def test_classify_scope_escape_takes_precedence_over_empty_diff():
    """If both empty-diff AND scope-escape conditions hold (the
    implementer wrote only out-of-scope files), the scope-escape
    classification wins — it's a more specific failure and signals
    the broader-rollback path explicitly."""
    assert _classify(
        None, (), "build_only", scope_escapes=("rogue.py",),
    ) == "hard_fail_scope_escape"


def test_classify_error_kind_still_takes_precedence_over_scope_escape():
    """A transport-class failure is upstream of scope enforcement —
    if the backend errored, we never had a clean attempt to assess
    scope escapes against."""
    assert _classify(
        "transport", ("a.py",), "build_only", scope_escapes=("b.py",),
    ) == "hard_fail_transport"


# ----------------------------------------------------------------------
# Transcript serialization
# ----------------------------------------------------------------------


def test_serialize_transcript_roundtrips_to_json():
    transcript = (
        Message(role="user", content="hello"),
        Message(role="assistant", content="hi"),
    )
    blob = _serialize_transcript(transcript)
    decoded = json.loads(blob.decode("utf-8"))
    assert decoded == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]


def test_serialize_transcript_handles_unicode():
    transcript = (Message(role="user", content="café ☕"),)
    blob = _serialize_transcript(transcript)
    assert "café" in blob.decode("utf-8")


# ----------------------------------------------------------------------
# End-to-end with a real Corpus + stub backend
# ----------------------------------------------------------------------


def test_implement_success_path(tmp_path: Path, monkeypatch):
    _init_repo(tmp_path)
    monkeypatch.chdir(tmp_path)

    corpus = Corpus(tmp_path / "_corpus")
    spec = _make_spec()
    directive = _make_directive("S0")

    # The backend "implements" by writing a file as a side effect.
    def write_file():
        (tmp_path / "hello.py").write_text("def hi(): return 'hi'\n")

    backend = StubBackend(side_effect=write_file)
    impl = implement(spec, "S0", directive, backend=backend, corpus=corpus)

    assert impl.slice_id == "S0"
    assert impl.base_commit != "unknown"
    assert impl.result_commit is None  # primitive never commits
    assert "hello.py" in impl.files_modified
    assert impl.classification == "success"
    assert impl.diff_ref.startswith("sha256:")
    assert impl.transcript_ref.startswith("sha256:")
    assert impl.execution.backend == "stub"
    assert impl.execution.model == "stub-model-v1"


def test_implement_checkpoints_returned_result_before_capture(
    tmp_path: Path,
    monkeypatch,
):
    _init_repo(tmp_path)
    monkeypatch.chdir(tmp_path)

    corpus = Corpus(tmp_path / "_corpus")
    transcript = (
        Message(role="user", content="checkpoint me"),
        Message(role="assistant", content="DONE"),
    )
    expected_result = InvocationResult(
        final_message="DONE",
        transcript=transcript,
        usage=Usage(
            input_tokens=31,
            output_tokens=7,
            cache_creation_tokens=3,
            cache_read_tokens=2,
            cost_usd=0.25,
        ),
        stop_reason="end_turn",
        completeness=ExecutionCompleteness(
            token_usage_complete=True,
            cost_complete=True,
            terminal_complete=True,
        ),
    )
    events: list[str] = []
    checkpoint_ref = None

    class ResultBackend:
        name = "result-backend"
        model = "result-model"

        def invoke(self, _invocation):
            events.append("backend")
            (tmp_path / "hello.py").write_text(
                "def hi(): return 'hi'\n",
                encoding="utf-8",
            )
            return expected_result

    def checkpoint(result, transcript_ref):
        nonlocal checkpoint_ref
        events.append("after_invoke")
        checkpoint_ref = transcript_ref
        assert result is expected_result
        assert result.usage.cache_creation_tokens == 3
        assert result.usage.cache_read_tokens == 2
        assert result.completeness.token_usage_complete is True
        assert result.completeness.cost_complete is True
        assert corpus.get_artifact(transcript_ref) == _serialize_transcript(
            transcript
        )

    def prepare_capture():
        events.append("before_capture")

    impl = implement(
        _make_spec(),
        "S0",
        _make_directive("S0"),
        backend=ResultBackend(),
        corpus=corpus,
        after_invoke=checkpoint,
        before_capture=prepare_capture,
    )

    assert events == ["backend", "after_invoke", "before_capture"]
    assert checkpoint_ref == impl.transcript_ref
    assert impl.classification == "success"
    # The early Corpus write is workflow evidence even when the Corpus resides
    # inside the captured checkout; it must not become an implementer escape.
    assert impl.scope_escapes == ()


def test_implement_checkpoint_failure_runs_safety_before_propagating(
    tmp_path: Path,
    monkeypatch,
):
    _init_repo(tmp_path)
    monkeypatch.chdir(tmp_path)

    corpus = Corpus(tmp_path / "_corpus")
    transcript = (
        Message(role="user", content="ping"),
        Message(role="assistant", content="DONE — stubbed."),
    )
    backend = StubBackend(transcript=transcript)
    callback_ref = None
    events: list[str] = []

    def checkpoint(result, transcript_ref):
        nonlocal callback_ref
        events.append("after_invoke")
        callback_ref = transcript_ref
        assert result.usage.input_tokens == 10
        raise RuntimeError("checkpoint storage unavailable")

    def prepare_capture():
        events.append("before_capture")

    with pytest.raises(RuntimeError, match="checkpoint storage unavailable"):
        implement(
            _make_spec(),
            "S0",
            _make_directive("S0"),
            backend=backend,
            corpus=corpus,
            after_invoke=checkpoint,
            before_capture=prepare_capture,
        )

    assert callback_ref is not None
    assert corpus.get_artifact(callback_ref) == _serialize_transcript(
        transcript
    )
    assert events == ["after_invoke", "before_capture"]


def test_implement_capture_failure_retains_earlier_checkpoint(
    tmp_path: Path,
    monkeypatch,
):
    _init_repo(tmp_path)
    monkeypatch.chdir(tmp_path)

    corpus = Corpus(tmp_path / "_corpus")
    transcript = (
        Message(role="user", content="ping"),
        Message(role="assistant", content="DONE — stubbed."),
    )
    backend = StubBackend(transcript=transcript)
    callback_ref = None

    def checkpoint(_result, transcript_ref):
        nonlocal callback_ref
        callback_ref = transcript_ref

    def fail_capture_preparation():
        assert callback_ref is not None
        raise OSError("capture preparation unavailable")

    impl = implement(
        _make_spec(),
        "S0",
        _make_directive("S0"),
        backend=backend,
        corpus=corpus,
        after_invoke=checkpoint,
        before_capture=fail_capture_preparation,
    )

    assert impl.classification == "hard_fail_capture"
    assert impl.transcript_ref == callback_ref
    assert corpus.get_artifact(impl.transcript_ref) == _serialize_transcript(
        transcript
    )
    assert b"capture preparation unavailable" in corpus.get_artifact(
        impl.artifacts["capture_error"]
    )


def test_implement_excludes_in_worktree_corpus_from_empty_scope(
    tmp_path: Path,
    monkeypatch,
):
    _init_repo(tmp_path)
    monkeypatch.chdir(tmp_path)

    corpus = Corpus(tmp_path / "_corpus")
    corpus.put_artifact(b"pre-existing evidence", kind="logs")
    impl = implement(
        _make_spec(),
        "S0",
        _make_directive(
            "S0",
            files_in_scope=(),
            verify_mode="skip",
        ),
        backend=StubBackend(),
        corpus=corpus,
    )

    assert impl.classification == "soft_fail_no_changes"
    assert impl.files_modified == ()
    assert impl.scope_escapes == ()
    assert corpus.get_artifact(impl.diff_ref) == b""


def test_implement_excludes_in_worktree_corpus_from_dot_scope(
    tmp_path: Path,
    monkeypatch,
):
    _init_repo(tmp_path)
    monkeypatch.chdir(tmp_path)

    corpus = Corpus(tmp_path / "_corpus")

    def write_source():
        (tmp_path / "hello.py").write_text(
            "def hi(): return 'hi'\n",
            encoding="utf-8",
        )

    impl = implement(
        _make_spec(),
        "S0",
        _make_directive("S0", files_in_scope=(".",)),
        backend=StubBackend(side_effect=write_source),
        corpus=corpus,
    )

    diff = corpus.get_artifact(impl.diff_ref)
    assert impl.classification == "success"
    assert impl.files_modified == ("hello.py",)
    assert impl.scope_escapes == ()
    assert b"hello.py" in diff
    assert b"_corpus" not in diff


def test_implement_excludes_corpus_even_when_scope_names_corpus_root(
    tmp_path: Path,
    monkeypatch,
):
    _init_repo(tmp_path)
    monkeypatch.chdir(tmp_path)

    corpus = Corpus(tmp_path / "_corpus")
    corpus.put_artifact(b"pre-existing evidence", kind="logs")
    impl = implement(
        _make_spec(),
        "S0",
        _make_directive(
            "S0",
            files_in_scope=("_corpus",),
            verify_mode="skip",
        ),
        backend=StubBackend(),
        corpus=corpus,
    )

    assert impl.classification == "soft_fail_no_changes"
    assert impl.files_modified == ()
    assert impl.scope_escapes == ()
    assert corpus.get_artifact(impl.diff_ref) == b""


def test_implement_excludes_concurrent_corpus_writes_and_transcripts(
    tmp_path: Path,
    monkeypatch,
):
    _init_repo(tmp_path)
    monkeypatch.chdir(tmp_path)

    corpus = Corpus(tmp_path / "_corpus")
    writer_started = threading.Event()
    stop_writer = threading.Event()
    writer_errors: list[BaseException] = []
    writer = None

    def write_corpus_until_stopped():
        sequence = 0
        try:
            while not stop_writer.is_set() and sequence < 200:
                corpus.put_artifact(
                    f"concurrent evidence {sequence}".encode(),
                    kind="logs",
                )
                sequence += 1
                writer_started.set()
        except BaseException as exc:  # noqa: BLE001 - asserted in owner thread
            writer_errors.append(exc)
            writer_started.set()

    def checkpoint(_result, _transcript_ref):
        nonlocal writer
        corpus.put_artifact(
            _serialize_transcript((
                Message(role="assistant", content="different transcript"),
            )),
            kind="transcripts",
        )
        writer = threading.Thread(target=write_corpus_until_stopped)
        writer.start()
        assert writer_started.wait(timeout=5)

    def write_source():
        (tmp_path / "hello.py").write_text(
            "def hi(): return 'hi'\n",
            encoding="utf-8",
        )

    try:
        impl = implement(
            _make_spec(),
            "S0",
            _make_directive("S0", files_in_scope=(".",)),
            backend=StubBackend(side_effect=write_source),
            corpus=corpus,
            after_invoke=checkpoint,
        )
    finally:
        stop_writer.set()
        if writer is not None:
            writer.join(timeout=5)

    assert writer is not None and not writer.is_alive()
    assert writer_errors == []
    diff = corpus.get_artifact(impl.diff_ref)
    assert impl.classification == "success"
    assert impl.files_modified == ("hello.py",)
    assert impl.scope_escapes == ()
    assert b"_corpus" not in diff


def test_implement_deleted_transcript_has_no_raw_resolution_failure(
    tmp_path: Path,
    monkeypatch,
):
    _init_repo(tmp_path)
    monkeypatch.chdir(tmp_path)

    corpus = Corpus(tmp_path / "_corpus")
    deleted_path = None

    def delete_transcript(_result, transcript_ref):
        nonlocal deleted_path
        deleted_path = (
            corpus.spec_dir
            / "artifacts"
            / "transcripts"
            / f"{str(transcript_ref).removeprefix('sha256:')}.bin"
        )
        deleted_path.unlink()

    def write_source():
        (tmp_path / "hello.py").write_text(
            "def hi(): return 'hi'\n",
            encoding="utf-8",
        )

    impl = implement(
        _make_spec(),
        "S0",
        _make_directive("S0"),
        backend=StubBackend(side_effect=write_source),
        corpus=corpus,
        after_invoke=delete_transcript,
    )

    assert deleted_path is not None and not deleted_path.exists()
    assert impl.classification == "success"
    assert impl.files_modified == ("hello.py",)
    assert impl.scope_escapes == ()


def test_implement_transcript_storage_failure_runs_safety_without_post_capture(
    tmp_path: Path,
    monkeypatch,
):
    _init_repo(tmp_path)
    monkeypatch.chdir(tmp_path)

    corpus = Corpus(tmp_path / "_corpus")
    implement_module = importlib.import_module("assay.primitives.implement")
    real_diff = implement_module._git_diff_bytes
    real_modified = implement_module._git_modified_files
    real_put_artifact = corpus.put_artifact
    backend_returned = False
    post_checkpoint_git_calls: list[str] = []
    events: list[str] = []

    def observe_diff(*args, **kwargs):
        if backend_returned:
            post_checkpoint_git_calls.append("diff")
        return real_diff(*args, **kwargs)

    def observe_modified(*args, **kwargs):
        if backend_returned:
            post_checkpoint_git_calls.append("status")
        return real_modified(*args, **kwargs)

    def fail_transcript(blob, *, kind, name_hint=None):
        if kind == "transcripts":
            raise OSError("transcript store unavailable")
        return real_put_artifact(blob, kind=kind, name_hint=name_hint)

    def mark_backend_return():
        nonlocal backend_returned
        backend_returned = True
        events.append("backend")

    monkeypatch.setattr(implement_module, "_git_diff_bytes", observe_diff)
    monkeypatch.setattr(implement_module, "_git_modified_files", observe_modified)
    monkeypatch.setattr(corpus, "put_artifact", fail_transcript)

    with pytest.raises(OSError, match="transcript store unavailable"):
        implement(
            _make_spec(),
            "S0",
            _make_directive("S0"),
            backend=StubBackend(side_effect=mark_backend_return),
            corpus=corpus,
            after_invoke=lambda *_args: events.append("after_invoke"),
            before_capture=lambda: events.append("before_capture"),
        )

    assert events == ["backend", "before_capture"]
    assert post_checkpoint_git_calls == []


def test_implement_checkpoint_error_remains_primary_over_safety_error(
    tmp_path: Path,
    monkeypatch,
):
    _init_repo(tmp_path)
    monkeypatch.chdir(tmp_path)

    corpus = Corpus(tmp_path / "_corpus")
    events: list[str] = []

    def fail_checkpoint(_result, _transcript_ref):
        events.append("after_invoke")
        raise ValueError("accounting checkpoint failed")

    def fail_safety():
        events.append("before_capture")
        raise OSError("provider quiescence failed")

    with pytest.raises(ValueError, match="accounting checkpoint failed") as caught:
        implement(
            _make_spec(),
            "S0",
            _make_directive("S0"),
            backend=StubBackend(),
            corpus=corpus,
            after_invoke=fail_checkpoint,
            before_capture=fail_safety,
        )

    assert events == ["after_invoke", "before_capture"]
    assert isinstance(caught.value.__cause__, OSError)
    assert str(caught.value.__cause__) == "provider quiescence failed"


def test_implement_returned_error_runs_checkpoint_and_safety_once(
    tmp_path: Path,
    monkeypatch,
):
    _init_repo(tmp_path)
    monkeypatch.chdir(tmp_path)

    corpus = Corpus(tmp_path / "_corpus")
    events: list[str] = []
    impl = implement(
        _make_spec(),
        "S0",
        _make_directive("S0"),
        backend=StubBackend(
            error=TransportError(http_status=503, message="returned terminal")
        ),
        corpus=corpus,
        after_invoke=lambda *_args: events.append("after_invoke"),
        before_capture=lambda: events.append("before_capture"),
    )

    assert events == ["after_invoke", "before_capture"]
    assert impl.classification == "hard_fail_transport"


def test_implement_raised_backend_error_runs_no_post_return_callbacks(
    tmp_path: Path,
    monkeypatch,
):
    _init_repo(tmp_path)
    monkeypatch.chdir(tmp_path)

    corpus = Corpus(tmp_path / "_corpus")
    events: list[str] = []

    class RaisedTerminalBackend:
        name = "raised-terminal"
        model = "raised-terminal-model"

        def invoke(self, _invocation):
            raise RuntimeError("raw backend terminal")

    with pytest.raises(RuntimeError, match="raw backend terminal"):
        implement(
            _make_spec(),
            "S0",
            _make_directive("S0"),
            backend=RaisedTerminalBackend(),
            corpus=corpus,
            after_invoke=lambda *_args: events.append("after_invoke"),
            before_capture=lambda: events.append("before_capture"),
        )

    assert events == []


@pytest.mark.parametrize(
    "failed_command",
    ["--is-inside-work-tree", "ls-files", "check-attr", "add", "diff"],
)
def test_implement_never_succeeds_with_an_incomplete_git_capture(
    tmp_path: Path,
    monkeypatch,
    failed_command: str,
):
    """A plumbing failure produces typed failure, never success+empty diff."""

    _init_repo(tmp_path)
    corpus = Corpus(tmp_path / "_corpus")
    backend_ran = False

    def write_file():
        nonlocal backend_ran
        (tmp_path / "hello.py").write_text("def hi(): return 'hi'\n")
        backend_ran = True

    real_run = subprocess.run

    def injected_failure(command, *args, **kwargs):
        # The implementation primitive also takes a guarded pre-invocation
        # dirty snapshot. Inject into the post-backend capture this test is
        # asserting about, rather than making setup itself fail closed.
        if backend_ran and failed_command in command:
            return subprocess.CompletedProcess(
                command,
                1,
                stdout=b"",
                stderr=b"injected capture failure",
            )
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", injected_failure)
    impl = implement(
        _make_spec(),
        "S0",
        _make_directive("S0"),
        backend=StubBackend(side_effect=write_file),
        corpus=corpus,
        cwd=str(tmp_path),
    )

    assert impl.classification == "hard_fail_capture"
    assert impl.files_modified == ()
    assert corpus.get_artifact(impl.diff_ref) == b""
    assert "capture_error" in impl.artifacts
    assert b"injected capture failure" in corpus.get_artifact(
        impl.artifacts["capture_error"]
    )


def test_implement_fails_capture_when_status_and_repo_probe_fail(
    tmp_path: Path,
    monkeypatch,
):
    """A failed status cannot masquerade as a clean skip-mode capture."""

    _init_repo(tmp_path)
    corpus = Corpus(tmp_path / "_corpus")
    backend_ran = False

    def write_file():
        nonlocal backend_ran
        (tmp_path / "hello.py").write_text("def hi(): return 'hi'\n")
        backend_ran = True

    real_run = subprocess.run
    failed_post_status = False

    def injected_failure(command, *args, **kwargs):
        nonlocal failed_post_status
        if backend_ran and "status" in command:
            failed_post_status = True
            return subprocess.CompletedProcess(
                command, 1, stdout=b"", stderr=b"status failed"
            )
        if "--is-inside-work-tree" in command and failed_post_status:
            return subprocess.CompletedProcess(
                command, 1, stdout=b"", stderr=b"probe failed"
            )
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", injected_failure)
    impl = implement(
        _make_spec(),
        "S0",
        _make_directive("S0", verify_mode="skip"),
        backend=StubBackend(side_effect=write_file),
        corpus=corpus,
        cwd=str(tmp_path),
    )

    assert impl.classification == "hard_fail_capture"
    assert impl.files_modified == ()
    assert b"probe failed" in corpus.get_artifact(
        impl.artifacts["capture_error"]
    )


def test_git_head_failure_in_repo_is_typed_not_unknown(
    tmp_path: Path,
    monkeypatch,
):
    """A real checkout never publishes an Implementation with unknown base."""

    _init_repo(tmp_path)
    real_run = subprocess.run

    def injected_failure(command, *args, **kwargs):
        if command[-2:] == ["rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(
                command, 1, stdout=b"", stderr=b"HEAD probe failed"
            )
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", injected_failure)

    with pytest.raises(GitCaptureError, match="HEAD probe failed"):
        implement(
            _make_spec(),
            "S0",
            _make_directive("S0"),
            backend=StubBackend(),
            corpus=Corpus(tmp_path / "_corpus"),
            cwd=str(tmp_path),
        )


def test_implement_hard_fail_empty_diff_for_mutating_directive(
    tmp_path: Path, monkeypatch,
):
    _init_repo(tmp_path)
    monkeypatch.chdir(tmp_path)

    corpus = Corpus(tmp_path / "_corpus")
    backend = StubBackend()  # no side effect, no file changes
    # _make_directive() defaults to verify_mode="build_only" — the
    # directive expected changes, the implementer produced none, so the
    # classification escalates to hard_fail_empty_diff (rather than
    # silently passing through as soft_fail_no_changes).
    impl = implement(
        _make_spec(), "S0", _make_directive(), backend=backend, corpus=corpus
    )

    assert impl.classification == "hard_fail_empty_diff"
    assert impl.files_modified == ()
    # Diff artifact is still persisted (empty bytes, valid ContentHash)
    # because postmortem investigation wants to see "nothing was written".
    assert impl.diff_ref.startswith("sha256:")


def test_implement_captures_cost_usd_from_invocation_usage(
    tmp_path: Path, monkeypatch,
):
    """The returned Implementation carries the cost_usd from the
    backend's Usage record — so workflows reading impl.cost_usd
    downstream (e.g. swebench's cumulative-budget gate) get the real
    spend instead of always-zero."""
    _init_repo(tmp_path)
    monkeypatch.chdir(tmp_path)

    corpus = Corpus(tmp_path / "_corpus")
    backend = StubBackend(
        side_effect=lambda: (tmp_path / "hello.py").write_text(
            "def hi(): return 'hi'\n"
        ),
    )
    impl = implement(
        _make_spec(), "S0", _make_directive(),
        backend=backend, corpus=corpus,
    )
    assert impl.classification == "success"
    # StubBackend hardcodes cost_usd=0.01 in its Usage record.
    assert impl.cost_usd == 0.01
    assert impl.input_tokens == 10
    assert impl.output_tokens == 20
    assert impl.cache_creation_tokens == 0
    assert impl.cache_read_tokens == 0


def test_implement_propagates_durable_pricing_evidence(
    tmp_path: Path, monkeypatch,
):
    _init_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    evidence = UsagePricingEvidence(
        segments=(UsagePricingSegment(10, 20, 2, 3, 0.01),),
        complete=True,
    )

    class _PricingBackend(StubBackend):
        def invoke(self, invocation):
            result = super().invoke(invocation)
            return InvocationResult(
                final_message=result.final_message,
                transcript=result.transcript,
                usage=Usage(
                    input_tokens=10,
                    output_tokens=20,
                    cost_usd=0.01,
                    cache_creation_tokens=2,
                    cache_read_tokens=3,
                ),
                stop_reason=result.stop_reason,
                error=result.error,
                completeness=ExecutionCompleteness(
                    token_usage_complete=True,
                    cost_complete=True,
                ),
                pricing_evidence=evidence,
            )

    corpus = Corpus(tmp_path / "_corpus")
    impl = implement(
        _make_spec(),
        "S0",
        _make_directive(),
        backend=_PricingBackend(),
        corpus=corpus,
    )

    assert impl.pricing_evidence == evidence
    assert impl.cache_creation_tokens == 2
    assert impl.cache_read_tokens == 3
    ref = corpus.put_record(impl, kind="implementations")
    from assay import Implementation

    assert corpus.get_record(ref, as_type=Implementation) == impl
    assert impl.stop_reason is None


def test_implement_preserves_usage_when_pricing_evidence_is_incomplete(
    tmp_path: Path, monkeypatch,
):
    """Incomplete segment evidence cannot replace exact aggregate usage."""

    _init_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    segment = UsagePricingSegment(39_335, 7, 14_267, 25_062, 0.10190475)
    evidence = UsagePricingEvidence(
        segments=(segment, segment, segment),
        complete=False,
    )

    class _IncompletePricingBackend(StubBackend):
        def invoke(self, invocation):
            result = super().invoke(invocation)
            return InvocationResult(
                final_message=result.final_message,
                transcript=result.transcript,
                usage=Usage(
                    input_tokens=39_335,
                    output_tokens=1_730,
                    cost_usd=0.14497975,
                    cache_creation_tokens=14_267,
                    cache_read_tokens=25_062,
                ),
                stop_reason="budget_exhausted",
                error=BudgetExhausted(
                    spent_usd=0.14497975,
                    limit_usd=0.08154425,
                ),
                completeness=ExecutionCompleteness(
                    token_usage_complete=True,
                    cost_complete=True,
                    terminal_complete=True,
                    trace_complete=False,
                ),
                pricing_evidence=evidence,
            )

    corpus = Corpus(tmp_path / "_corpus")
    impl = implement(
        _make_spec(),
        "S0",
        _make_directive(),
        backend=_IncompletePricingBackend(
            side_effect=lambda: (tmp_path / "hello.py").write_text(
                "def hi(): return 'hi'\n"
            ),
        ),
        corpus=corpus,
    )

    assert impl.input_tokens == 39_335
    assert impl.output_tokens == 1_730
    assert impl.cost_usd == pytest.approx(0.14497975)
    assert impl.cache_creation_tokens == 14_267
    assert impl.cache_read_tokens == 25_062
    assert impl.pricing_evidence == evidence
    assert impl.accounting_debt == ()
    assert impl.classification == "hard_fail_budget_exhausted"
    ref = corpus.put_record(impl, kind="implementations")
    from assay import Implementation

    assert corpus.get_record(ref, as_type=Implementation) == impl


def test_implement_preserves_max_turns_cause_for_empty_diff(
    tmp_path: Path, monkeypatch,
):
    _init_repo(tmp_path)
    monkeypatch.chdir(tmp_path)

    class _MaxTurnsBackend:
        name = "stub-max-turns"
        model = "stub-model-v1"

        def invoke(self, invocation):
            return InvocationResult(
                final_message="still investigating",
                transcript=(),
                usage=Usage(input_tokens=101, output_tokens=17, cost_usd=0.0),
                stop_reason="max_turns",
            )

    impl = implement(
        _make_spec(),
        "S0",
        _make_directive(),
        backend=_MaxTurnsBackend(),
        corpus=Corpus(tmp_path / "_corpus"),
    )

    assert impl.classification == "hard_fail_max_turns"
    assert impl.stop_reason == "max_turns"
    assert impl.input_tokens == 101
    assert impl.output_tokens == 17


def test_implement_marks_partial_diff_before_max_turns_incomplete(
    tmp_path: Path, monkeypatch,
):
    _init_repo(tmp_path)
    monkeypatch.chdir(tmp_path)

    class _PartialMaxTurnsBackend:
        name = "stub-max-turns"
        model = "stub-model-v1"

        def invoke(self, invocation):
            (tmp_path / "hello.py").write_text("def hi(): return 'hi'\n")
            return InvocationResult(
                final_message="",
                transcript=(),
                usage=Usage(input_tokens=13, output_tokens=5, cost_usd=0.0),
                stop_reason="max_turns",
            )

    impl = implement(
        _make_spec(),
        "S0",
        _make_directive(),
        backend=_PartialMaxTurnsBackend(),
        corpus=Corpus(tmp_path / "_corpus"),
    )

    assert impl.classification == "hard_fail_max_turns"
    assert impl.files_modified == ("hello.py",)
    assert impl.stop_reason == "max_turns"
    assert impl.execution_completeness == ExecutionCompleteness()
    assert impl.accounting_debt == (
        "token_usage_complete",
        "cost_complete",
    )


@pytest.mark.parametrize(
    ("stop_reason", "error", "expected_classification"),
    (
        pytest.param(
            "error",
            TransportError(http_status=None, message="request timed out"),
            "hard_fail_transport",
            id="transport",
        ),
        pytest.param(
            "max_turns",
            None,
            "hard_fail_max_turns",
            id="max-turns",
        ),
        pytest.param(
            "end_turn",
            None,
            "hard_fail_empty_diff",
            id="clean-end",
        ),
    ),
)
def test_implement_does_not_launder_inherited_scoped_diff_as_local_work(
    tmp_path: Path,
    monkeypatch,
    stop_reason: str,
    error: TransportError | None,
    expected_classification: str,
):
    """Cumulative dirt is retained as evidence, not fresh-write authority."""

    _init_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "seed.txt"
    target.write_text("seed\ninherited tier change\n", encoding="utf-8")
    inherited_diff = _git_diff_bytes(
        cwd=str(tmp_path),
        path_scope=("seed.txt",),
    )
    corpus = Corpus(tmp_path / "_corpus")

    impl = implement(
        _make_spec(),
        "S0",
        _make_directive(files_in_scope=("seed.txt",), verify_mode="tests"),
        backend=StubBackend(error=error, stop_reason=stop_reason),
        corpus=corpus,
        cwd=str(tmp_path),
    )

    assert impl.classification == expected_classification
    # Composition still receives the cumulative HEAD-to-workspace postimage.
    assert impl.files_modified == ("seed.txt",)
    assert corpus.get_artifact(impl.diff_ref) == inherited_diff


def test_implement_retains_partial_transport_diff_as_failure_evidence(
    tmp_path: Path,
    monkeypatch,
):
    """A real edit remains captured without laundering a failed invocation."""

    _init_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "seed.txt"
    target.write_text("seed\ninherited tier change\n", encoding="utf-8")
    inherited_diff = _git_diff_bytes(
        cwd=str(tmp_path),
        path_scope=("seed.txt",),
    )
    corpus = Corpus(tmp_path / "_corpus")

    impl = implement(
        _make_spec(),
        "S0",
        _make_directive(files_in_scope=("seed.txt",), verify_mode="tests"),
        backend=StubBackend(
            error=TransportError(http_status=None, message="request timed out"),
            stop_reason="error",
            side_effect=lambda: target.write_text(
                "seed\ninherited tier change\nlocal invocation change\n",
                encoding="utf-8",
            ),
        ),
        corpus=corpus,
        cwd=str(tmp_path),
    )

    cumulative_diff = corpus.get_artifact(impl.diff_ref)
    assert impl.classification == "hard_fail_transport"
    assert impl.files_modified == ("seed.txt",)
    assert impl.stop_reason == "error"
    assert impl.execution_completeness == ExecutionCompleteness()
    assert impl.accounting_debt == (
        "token_usage_complete",
        "cost_complete",
    )
    assert cumulative_diff != inherited_diff
    assert b"local invocation change" in cumulative_diff


def test_implement_honors_system_prompt_override(
    tmp_path: Path, monkeypatch,
):
    """When ``system_prompt=`` is set, ``assay.implement`` uses it
    INSTEAD of the default ``_IMPLEMENTER_SYSTEM_PROMPT`` — letting
    workflows whose agent contract differs (e.g. swebench, which
    needs the agent to iterate on tests rather than skip them) opt
    in to their own rules."""
    _init_repo(tmp_path)
    monkeypatch.chdir(tmp_path)

    corpus = Corpus(tmp_path / "_corpus")

    captured: list[str] = []

    class _CapturingBackend:
        name = "stub"
        model = "stub-model-v1"

        def __init__(self): pass

        def invoke(self, invocation):
            captured.append(invocation.system)
            (tmp_path / "hello.py").write_text("def hi(): return 'hi'\n")
            return InvocationResult(
                final_message="DONE",
                transcript=(
                    Message(role="user", content=invocation.user),
                    Message(role="assistant", content="DONE"),
                ),
                usage=Usage(input_tokens=1, output_tokens=1, cost_usd=0.0),
                stop_reason="end_turn",
            )

    custom_prompt = "You are a test-iterating swebench agent. Read tests first."
    impl = implement(
        _make_spec(), "S0", _make_directive(),
        backend=_CapturingBackend(), corpus=corpus,
        system_prompt=custom_prompt,
    )
    assert impl.classification == "success"
    assert captured == [custom_prompt]


def test_implement_default_system_prompt_when_not_overridden(
    tmp_path: Path, monkeypatch,
):
    """When ``system_prompt=`` is omitted, the default
    ``_IMPLEMENTER_SYSTEM_PROMPT`` is used — preserves the existing
    tier-workflow contract for the bulk of callers."""
    _init_repo(tmp_path)
    monkeypatch.chdir(tmp_path)

    corpus = Corpus(tmp_path / "_corpus")
    captured: list[str] = []

    class _CapturingBackend:
        name = "stub"
        model = "stub-model-v1"
        def __init__(self): pass
        def invoke(self, invocation):
            captured.append(invocation.system)
            (tmp_path / "hello.py").write_text("def hi(): return 'hi'\n")
            return InvocationResult(
                final_message="DONE",
                transcript=(
                    Message(role="user", content=invocation.user),
                    Message(role="assistant", content="DONE"),
                ),
                usage=Usage(input_tokens=1, output_tokens=1, cost_usd=0.0),
                stop_reason="end_turn",
            )

    implement(
        _make_spec(), "S0", _make_directive(),
        backend=_CapturingBackend(), corpus=corpus,
    )
    # The default prompt mentions the implementer protocol's
    # "DONE" token contract, which any future replacement would also
    # have to preserve.
    assert len(captured) == 1
    assert "DONE" in captured[0]


def test_implement_cost_usd_zero_on_backend_error(
    tmp_path: Path, monkeypatch,
):
    """Backend error paths still produce a typed Implementation; cost
    falls through to whatever Usage carried (which is the default 0.01
    in the stub, but real backends typically don't surface cost on
    transport errors). Test pins the contract that cost_usd is always
    a float, never absent."""
    _init_repo(tmp_path)
    monkeypatch.chdir(tmp_path)

    corpus = Corpus(tmp_path / "_corpus")
    backend = StubBackend(error=TransportError(http_status=500, message="boom"))
    impl = implement(
        _make_spec(), "S0", _make_directive(),
        backend=backend, corpus=corpus,
    )
    assert impl.classification == "hard_fail_transport"
    assert isinstance(impl.cost_usd, float)


def test_implement_soft_fail_no_changes_for_audit_only_directive(
    tmp_path: Path, monkeypatch,
):
    _init_repo(tmp_path)
    monkeypatch.chdir(tmp_path)

    corpus = Corpus(tmp_path / "_corpus")
    backend = StubBackend()  # no side effect, no file changes
    # Audit-only slices (verify_mode="skip") legitimately produce no
    # diff and keep the soft classification.
    impl = implement(
        _make_spec(), "S0",
        _make_directive(verify_mode="skip"),
        backend=backend, corpus=corpus,
    )

    assert impl.classification == "soft_fail_no_changes"
    assert impl.files_modified == ()
    assert impl.diff_ref.startswith("sha256:")


def test_implement_hard_fail_on_backend_error(tmp_path: Path, monkeypatch):
    _init_repo(tmp_path)
    monkeypatch.chdir(tmp_path)

    corpus = Corpus(tmp_path / "_corpus")
    backend = StubBackend(error=TransportError(http_status=500, message="boom"))
    impl = implement(
        _make_spec(), "S0", _make_directive(), backend=backend, corpus=corpus
    )

    assert impl.classification == "hard_fail_transport"
    # Transcript still persisted on hard fail.
    assert impl.transcript_ref.startswith("sha256:")


def test_implement_invokes_backend_with_built_prompt(
    tmp_path: Path, monkeypatch
):
    _init_repo(tmp_path)
    monkeypatch.chdir(tmp_path)

    corpus = Corpus(tmp_path / "_corpus")
    backend = StubBackend()
    implement(
        _make_spec(),
        "S0",
        _make_directive("S0", files_in_scope=("scoped.py",)),
        backend=backend,
        corpus=corpus,
    )

    assert len(backend.invocations) == 1
    inv = backend.invocations[0]
    assert "scoped.py" in inv.user
    assert "S0" in inv.user
    assert inv.tools is None


def test_implement_persists_diff_to_corpus(tmp_path: Path, monkeypatch):
    """The diff returned by the primitive must be retrievable via the corpus."""
    _init_repo(tmp_path)
    monkeypatch.chdir(tmp_path)

    corpus = Corpus(tmp_path / "_corpus")

    def make_change():
        # Write to a file inside the directive's scope; the
        # scope-bounded git-diff capture excludes anything outside
        # files_in_scope ∪ creates_paths.
        (tmp_path / "seed.txt").write_text("seed\nMODIFIED\n")

    backend = StubBackend(side_effect=make_change)
    impl = implement(
        _make_spec(), "S0",
        _make_directive(files_in_scope=("seed.txt",)),
        backend=backend, corpus=corpus,
    )

    diff_blob = corpus.get_artifact(impl.diff_ref)
    assert b"MODIFIED" in diff_blob


def test_implement_greenfield_new_crate_no_scope_escape(
    tmp_path: Path, monkeypatch
):
    """A greenfield slice that stands up a NEW crate (a brand-new dir of
    files, each declared in creates_paths) must classify ``success`` — not
    ``hard_fail_scope_escape``.

    Regression: porcelain-v1 collapsed the entirely-untracked dir to a
    single ``crates/foo/`` entry in the unscoped scope-escape capture,
    which never matched the file-level creates_paths, so the whole new dir
    read as one out-of-scope write. ``-uall`` keeps that capture at file
    granularity. Mirrors the fabric-render-ovrtx Phase-A crate spike.
    """
    _init_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    corpus = Corpus(tmp_path / "_corpus")

    creates = (
        "crates/foo/Cargo.toml",
        "crates/foo/src/lib.rs",
        "crates/foo/build.rs",
    )

    def scaffold_new_crate():
        for rel in creates:
            p = tmp_path / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("// scaffolded\n")
        # Also edit one tracked, in-scope file (the workspace manifest).
        (tmp_path / "seed.txt").write_text("seed\nregister-member\n")

    backend = StubBackend(side_effect=scaffold_new_crate)
    impl = implement(
        _make_spec(),
        "S0",
        _make_directive(files_in_scope=("seed.txt",), creates_paths=creates),
        backend=backend,
        corpus=corpus,
    )

    assert impl.scope_escapes == (), impl.scope_escapes
    assert impl.classification == "success", impl.classification


def test_capture_workspace_delta_public_helper(tmp_path: Path):
    """capture_workspace_delta (the hand-completion capture surface) returns
    the scoped diff, modified files, and base commit — including a brand-new
    file (intent-to-add visibility) — and matches implement()'s capture."""
    from assay.primitives.implement import capture_workspace_delta

    base = _init_repo(tmp_path)
    (tmp_path / "seed.txt").write_text("seed\nedited\n")
    (tmp_path / "new_file.py").write_text("print('hi')\n")

    diff, files, sha = capture_workspace_delta(
        cwd=str(tmp_path), path_scope=("seed.txt", "new_file.py")
    )
    assert sha == base
    assert set(files) == {"seed.txt", "new_file.py"}
    assert b"edited" in diff and b"new_file.py" in diff

    # Scoping excludes out-of-scope edits.
    (tmp_path / "other.txt").write_text("outside\n")
    diff2, files2, _ = capture_workspace_delta(
        cwd=str(tmp_path), path_scope=("seed.txt",)
    )
    assert files2 == ("seed.txt",)
    assert b"other.txt" not in diff2


def test_capture_workspace_delta_bounds_diff_and_files_by_one_shot_scope(
    tmp_path: Path,
):
    """A scope handed over as a one-shot iterable bounds the diff and the
    file list alike."""
    _init_repo(tmp_path)
    (tmp_path / "seed.txt").write_text("seed\nedited\n")
    (tmp_path / "other.txt").write_text("outside the scope\n")

    diff, files, _ = capture_workspace_delta(
        cwd=str(tmp_path), path_scope=iter(("seed.txt",))
    )

    assert files == ("seed.txt",)
    assert b"edited" in diff
    assert b"other.txt" not in diff


def test_execution_model_uses_required_protocol_attr(tmp_path: Path):
    """Execution provenance uses the Backend protocol's public model."""
    from assay.primitives.implement import _snapshot_execution

    class PublicModelBackend:
        name = "real"
        model = "claude-opus-4-8"

    b = PublicModelBackend()
    ex = _snapshot_execution(
        backend_name=b.name,
        model=b.model,
    )
    assert ex.model == "claude-opus-4-8"


# ----------------------------------------------------------------------
# Scope paths are literal — a leading component is never dropped
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "holds_same_named_directory",
    [False, True],
    ids=["no_same_named_directory", "same_named_directory"],
)
def test_capture_reads_scope_path_led_by_checkout_name_as_written(
    tmp_path: Path,
    holds_same_named_directory: bool,
):
    """A scope path names the same file before and after that file exists.

    ``proj/src/a.py`` in a checkout directory named ``proj`` is
    ``<checkout>/proj/src/a.py`` and never ``<checkout>/src/a.py``, whether
    or not the checkout already holds a ``proj`` directory."""
    repo = tmp_path / "proj"
    (repo / "src").mkdir(parents=True)
    _init_repo(repo)
    if holds_same_named_directory:
        (repo / "proj").mkdir()
    (repo / "src" / "a.py").write_text("root twin\n")
    (repo / "seed.txt").write_text("seed\nedited\n")
    scope = ("proj/src/a.py", "seed.txt")

    diff_before, files_before, _ = capture_workspace_delta(
        cwd=str(repo), path_scope=scope
    )

    (repo / "proj" / "src").mkdir(parents=True)
    (repo / "proj" / "src" / "a.py").write_text("as written\n")
    diff_after, files_after, _ = capture_workspace_delta(
        cwd=str(repo), path_scope=scope
    )

    assert files_before == ("seed.txt",)
    assert b"edited" in diff_before
    assert b"root twin" not in diff_before
    assert set(files_after) == {"proj/src/a.py", "seed.txt"}
    assert b"as written" in diff_after
    assert b"root twin" not in diff_after


def test_capture_keeps_scope_path_when_its_directory_is_removed(
    tmp_path: Path,
):
    """Deleting the last file of a directory is captured under the scope
    path that named the file while it existed."""
    repo = tmp_path / "proj"
    (repo / "proj" / "src").mkdir(parents=True)
    _init_repo(repo)
    (repo / "proj" / "src" / "a.py").write_text("a = 1\n")
    subprocess.run(["git", "add", "proj/src/a.py"], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "add a"], cwd=repo, check=True
    )
    scope = ("proj/src/a.py",)

    _, files_before, _ = capture_workspace_delta(
        cwd=str(repo), path_scope=scope
    )

    (repo / "proj" / "src" / "a.py").unlink()
    (repo / "proj" / "src").rmdir()
    diff_after, files_after, _ = capture_workspace_delta(
        cwd=str(repo), path_scope=scope
    )

    assert files_before == ()
    assert files_after == ("proj/src/a.py",)
    assert b"deleted file mode" in diff_after


@pytest.mark.parametrize(
    "holds_same_named_directory",
    [False, True],
    ids=["no_same_named_directory", "same_named_directory"],
)
def test_implement_reports_write_beside_scope_path_as_escape(
    tmp_path: Path,
    holds_same_named_directory: bool,
):
    """A write to ``src/a.py`` is outside a scope that names
    ``proj/src/a.py``, also when the checkout directory is named ``proj``."""
    repo = tmp_path / "proj"
    (repo / "src").mkdir(parents=True)
    _init_repo(repo)
    if holds_same_named_directory:
        (repo / "proj").mkdir()
    corpus = Corpus(repo / "_corpus")

    def write_root_twin():
        (repo / "src" / "a.py").write_text("root twin\n")

    backend = StubBackend(side_effect=write_root_twin)
    impl = implement(
        _make_spec(),
        "S0",
        _make_directive("S0", files_in_scope=("proj/src/a.py",)),
        backend=backend,
        corpus=corpus,
        cwd=str(repo),
    )

    assert "- `proj/src/a.py`" in backend.invocations[0].user
    assert impl.files_modified == ()
    assert corpus.get_artifact(impl.diff_ref) == b""
    assert impl.scope_escapes == ("src/a.py",)
    assert impl.classification == "hard_fail_scope_escape"


@pytest.mark.parametrize(
    "holds_same_named_directory",
    [False, True],
    ids=["no_same_named_directory", "same_named_directory"],
)
def test_implement_captures_write_to_scope_path_led_by_checkout_name(
    tmp_path: Path,
    holds_same_named_directory: bool,
):
    """A write to ``proj/src/a.py`` is the invocation's file; a ``src/a.py``
    that was dirty before the invocation is not."""
    repo = tmp_path / "proj"
    (repo / "src").mkdir(parents=True)
    _init_repo(repo)
    if holds_same_named_directory:
        (repo / "proj").mkdir()
    (repo / "src" / "a.py").write_text("root twin\n")
    corpus = Corpus(repo / "_corpus")

    def write_as_written():
        (repo / "proj" / "src").mkdir(parents=True)
        (repo / "proj" / "src" / "a.py").write_text("as written\n")

    impl = implement(
        _make_spec(),
        "S0",
        _make_directive("S0", files_in_scope=("proj/src/a.py",)),
        backend=StubBackend(side_effect=write_as_written),
        corpus=corpus,
        cwd=str(repo),
    )

    diff = corpus.get_artifact(impl.diff_ref)
    assert impl.files_modified == ("proj/src/a.py",)
    assert b"as written" in diff
    assert b"root twin" not in diff
    assert impl.scope_escapes == ()
    assert impl.classification == "success"


def test_implement_captures_deletion_that_removes_the_scope_path_directory(
    tmp_path: Path,
):
    """Deleting ``proj/src/a.py`` together with its directory is a change
    inside the scope that names it, not an escape."""
    repo = tmp_path / "proj"
    (repo / "proj" / "src").mkdir(parents=True)
    _init_repo(repo)
    (repo / "proj" / "src" / "a.py").write_text("a = 1\n")
    subprocess.run(["git", "add", "proj/src/a.py"], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "add a"], cwd=repo, check=True
    )
    corpus = Corpus(repo / "_corpus")

    def delete_with_directory():
        (repo / "proj" / "src" / "a.py").unlink()
        (repo / "proj" / "src").rmdir()

    impl = implement(
        _make_spec(),
        "S0",
        _make_directive("S0", files_in_scope=("proj/src/a.py",)),
        backend=StubBackend(side_effect=delete_with_directory),
        corpus=corpus,
        cwd=str(repo),
    )

    assert impl.files_modified == ("proj/src/a.py",)
    assert b"deleted file mode" in corpus.get_artifact(impl.diff_ref)
    assert impl.scope_escapes == ()
    assert impl.classification == "success"


def test_implement_leaves_ambient_file_beside_scope_path_uncaptured(
    tmp_path: Path,
):
    """A ``src/a.py`` that was dirty before the invocation is ambient to a
    scope that names ``proj/src/a.py``: a further write to it is neither
    captured nor reported as an escape, and with no other change the result
    is an empty diff."""
    repo = tmp_path / "proj"
    (repo / "src").mkdir(parents=True)
    _init_repo(repo)
    (repo / "src" / "a.py").write_text("root twin\n")
    corpus = Corpus(repo / "_corpus")

    def write_root_twin_again():
        (repo / "src" / "a.py").write_text("root twin\nwritten again\n")

    impl = implement(
        _make_spec(),
        "S0",
        _make_directive("S0", files_in_scope=("proj/src/a.py",)),
        backend=StubBackend(side_effect=write_root_twin_again),
        corpus=corpus,
        cwd=str(repo),
    )

    assert impl.files_modified == ()
    assert corpus.get_artifact(impl.diff_ref) == b""
    assert impl.scope_escapes == ()
    assert impl.classification == "hard_fail_empty_diff"


def test_implement_classifies_scope_change_without_the_ambient_write(
    tmp_path: Path,
):
    """An invocation that changes a path its scope names and also writes to
    the ambient ``src/a.py`` is classified as if that write had not
    happened: a success whose file list and diff do not hold it."""
    repo = tmp_path / "proj"
    (repo / "src").mkdir(parents=True)
    _init_repo(repo)
    (repo / "src" / "a.py").write_text("root twin\n")
    corpus = Corpus(repo / "_corpus")

    def write_both():
        (repo / "src" / "a.py").write_text("root twin\nwritten again\n")
        (repo / "seed.txt").write_text("seed\nedited\n")

    impl = implement(
        _make_spec(),
        "S0",
        _make_directive("S0", files_in_scope=("proj/src/a.py", "seed.txt")),
        backend=StubBackend(side_effect=write_both),
        corpus=corpus,
        cwd=str(repo),
    )

    diff = corpus.get_artifact(impl.diff_ref)
    assert impl.files_modified == ("seed.txt",)
    assert b"edited" in diff
    assert b"written again" not in diff
    assert impl.scope_escapes == ()
    assert impl.classification == "success"


# ----------------------------------------------------------------------
# _capture_pathspecs — what the scope keeps on its way to Git
# ----------------------------------------------------------------------
def test_capture_pathspecs_return_the_scope_as_given():
    scope = ("src/a.py", "src/b.py")

    assert _capture_pathspecs(None, ()) is None
    assert _capture_pathspecs((), ()) == ()
    assert _capture_pathspecs(scope, ()) == scope


def test_capture_pathspecs_wrap_every_include_when_exclusions_are_given():
    exclusion = ":(top,literal,exclude)_corpus"

    assert _capture_pathspecs(
        ("src/a.py", ":(exclude)b.py"), (exclusion,)
    ) == (
        ":(literal)src/a.py",
        ":(literal):(exclude)b.py",
        exclusion,
    )
    assert _capture_pathspecs(None, (exclusion,)) == (exclusion,)
    assert _capture_pathspecs((), (exclusion,)) == (exclusion,)


@pytest.mark.skipif(os.name == "nt", reason="backslash is a Windows separator")
def test_capture_pathspecs_keep_posix_backslash_as_a_name_byte():
    assert _capture_pathspecs(("src/lit\\eral.py",), ()) == (
        "src/lit\\eral.py",
    )


def test_capture_pathspecs_use_forward_slashes_on_windows(monkeypatch):
    scope = ("src\\lib\\a.py", "src\\b.py", "src/c.py")
    exclusion = ":(top,literal,exclude)_corpus"
    with monkeypatch.context() as windows:
        windows.setattr(os, "name", "nt")
        plain = _capture_pathspecs(scope, ())
        wrapped = _capture_pathspecs(scope, (exclusion,))

    assert plain == ("src/lib/a.py", "src/b.py", "src/c.py")
    assert wrapped == (
        ":(literal)src/lib/a.py",
        ":(literal)src/b.py",
        ":(literal)src/c.py",
        exclusion,
    )


def test_read_only_reference_is_context_and_never_allows_mutation(tmp_path):
    _init_repo(tmp_path)
    original = b"seed\n"
    reference = ReadOnlySourceReference(
        reference_id="existing-seed", path="seed.txt",
        sha256="sha256:" + hashlib.sha256(original).hexdigest(),
        start_line=1, end_line=1,
    )
    directive = dataclasses.replace(
        _make_directive(), read_only_references=(reference,),
    )
    prompt = _build_implementer_prompt(_make_spec(), "S0", directive)
    assert "## Read-only source references" in prompt
    assert "They grant no write permission" in prompt
    assert reference.sha256 in prompt
    assert "`seed.txt` lines 1-1" in prompt
    scope_section = prompt.split("## Files in scope", 1)[1].split(
        "## Read-only source references", 1,
    )[0]
    assert "seed.txt" not in scope_section

    def write_reference():
        (tmp_path / "seed.txt").write_text("illicit reference edit\n")
        (tmp_path / "hello.py").write_text("print('allowed')\n")

    result = implement(
        _make_spec(), "S0", directive,
        backend=StubBackend(side_effect=write_reference),
        corpus=Corpus(tmp_path / "_corpus"), cwd=str(tmp_path),
    )
    assert result.classification == "hard_fail_scope_escape"
    assert result.scope_escapes == ("seed.txt",)
    assert result.files_modified == ("hello.py",)
