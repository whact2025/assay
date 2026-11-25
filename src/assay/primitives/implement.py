# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Backend-driven implementation and bounded Git-delta capture.

Translates one slice's ``ImplementationDirective`` into a backend
invocation, captures the scoped HEAD-to-working-tree diff via Git, persists the
diff and agent transcript to the Corpus, and returns a typed Implementation.
The persisted patch/file list is cumulative for the working tree; internal
before/after captures separately determine whether this invocation changed it.

Scope paths are literal. Each ``files_in_scope``/``creates_paths`` entry, like
each ``path_scope`` entry of ``capture_workspace_delta``, names one path
relative to ``cwd``. It is never rewritten against the working tree, so the
captures before and after the invocation and the scope-escape check all read
it as the same path.

The primitive is deliberately minimal: it does NOT run build/verify
(the recipe layer owns that), does NOT commit to git (the recipe
decides whether to keep changes), and does NOT iterate beyond a single
backend invocation. The ``classification`` field is a coarse early
signal — recipes refine it after running build/verify.

Git calls use the explicit ``cwd`` argument when supplied and otherwise use
``os.getcwd()``. Concurrent callers must pass an explicit per-workspace path;
changing process cwd is not thread-safe.
"""

from __future__ import annotations

import datetime as _datetime
import json
import logging
import os
import platform
import queue
import re
import shutil
import socket
import stat
import subprocess
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterator

from ..types.artifacts import Implementation
from ..types.dag import ImplementationDirective
from ..types.execution import ExecutionContext
from ..types.identity import ContentHash
from ..types.invocation import Invocation, InvocationResult, Message
from ..types.tools import Tool, ToolSet
from .prompt_abi import (
    PromptABIProfile,
    PromptOutputContract,
    profile_for_backend,
    render_prompt_abi_v2,
)

logger = logging.getLogger(__name__)


class GitCaptureError(RuntimeError):
    """Git plumbing could not produce a complete, trustworthy delta."""


_GIT_SAFE_OPTIONS: tuple[str, ...] = (
    "-c", "core.hooksPath=/dev/null",
    "-c", f"core.attributesFile={os.devnull}",
    "-c", "core.fsmonitor=false",
    "-c", "core.untrackedCache=false",
    "-c", "protocol.file.allow=never",
    "-c", "protocol.ext.allow=never",
)

_GIT_CAPTURE_SCRATCH_PREFIX = ".assay-git-capture-"


def _enclosing_git_worktree(probe: Path) -> Path | None:
    """Resolve the worktree enclosing ``probe`` without ambient Git state."""

    environment = _sanitized_git_env(
        GIT_DISCOVERY_ACROSS_FILESYSTEM="1",
        GIT_OPTIONAL_LOCKS="0",
        GIT_LITERAL_PATHSPECS="1",
    )
    result = subprocess.run(
        _git_argv(
            "rev-parse", "--path-format=absolute", "--show-toplevel",
        ),
        check=False,
        capture_output=True,
        cwd=str(probe),
        env=environment,
    )
    if result.returncode != 0:
        if _has_git_admin_marker(probe):
            raise GitCaptureError(
                _git_failure("resolve enclosing Git worktree", result)
            )
        return None
    rendered = os.fsdecode(result.stdout).strip()
    if not rendered:
        raise GitCaptureError(
            "resolve enclosing Git worktree returned empty output"
        )
    root = Path(rendered).resolve(strict=True)
    resolved_probe = probe.resolve(strict=True)
    if not resolved_probe.is_relative_to(root):
        raise GitCaptureError(
            "Git reported a worktree that does not enclose its probe path"
        )
    return root


def _outermost_enclosing_worktree(worktree_root: Path) -> Path:
    """Walk repository nesting until no parent worktree remains."""

    outermost = worktree_root.expanduser().resolve(strict=True)
    while outermost.parent != outermost:
        enclosing = _enclosing_git_worktree(outermost.parent)
        if enclosing is None:
            break
        if enclosing == outermost or not outermost.is_relative_to(enclosing):
            raise GitCaptureError(
                "enclosing Git worktree discovery did not make outward progress"
            )
        outermost = enclosing
    return outermost


def _literal_git_admin_directory(worktree_root: Path) -> Path | None:
    """Return a literal, in-worktree ``.git`` directory, never a symlink."""

    marker = worktree_root / ".git"
    try:
        marker_stat = marker.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISDIR(marker_stat.st_mode):
        return None
    resolved = marker.resolve(strict=True)
    if resolved != marker:
        return None
    return resolved


@contextmanager
def _git_capture_scratch(worktree_root: Path) -> Iterator[Path]:
    """Yield isolated Git scratch on the worktree's filesystem.

    Capture can copy a large index and must not inherit the process-global
    temporary directory: benchmark hosts commonly keep ``/tmp`` on a small
    root volume while workspaces live on a large data volume. Repository
    nesting is resolved first, then scratch is created beyond the outermost
    worktree boundary. This prevents an unscoped or concurrent capture in a
    superproject from mistaking another capture's index for model output.

    A literal, non-symlink ``.git`` directory on the active worktree is a
    fallback only when the external parent is unavailable. Git excludes that
    namespace by construction; a symlink or gitfile is never accepted as
    exclusion proof.

    ``TemporaryDirectory`` supplies collision-resistant creation and recursive
    cleanup.  The explicit parent and post-creation checks fail closed if a
    substituted or future temporary-directory implementation escapes that
    parent or crosses devices.
    """

    try:
        workspace = worktree_root.expanduser().resolve(strict=True)
        if not workspace.is_dir():
            raise GitCaptureError(
                f"Git capture worktree is not a directory: {workspace}"
            )
        workspace_device = workspace.stat().st_dev
        outermost = _outermost_enclosing_worktree(workspace)
        candidates: list[tuple[str, Path]] = []
        external_parent = outermost.parent.resolve(strict=True)
        if (
            external_parent != outermost
            and external_parent.stat().st_dev == workspace_device
        ):
            candidates.append(("external-workspace", external_parent))
        fallback = _literal_git_admin_directory(workspace)
        if (
            fallback is not None
            and fallback.stat().st_dev == workspace_device
        ):
            candidates.append(("literal-git-admin", fallback))
    except GitCaptureError:
        raise
    except OSError as exc:
        raise GitCaptureError(f"Git capture scratch failed: {exc}") from exc

    creation_errors: list[str] = []
    for label, scratch_parent in candidates:
        try:
            temporary = tempfile.TemporaryDirectory(
                prefix=_GIT_CAPTURE_SCRATCH_PREFIX,
                dir=str(scratch_parent),
            )
        except OSError as exc:
            creation_errors.append(f"{label}: {exc}")
            continue
        with temporary as raw_scratch:
            try:
                scratch = Path(raw_scratch).resolve(strict=True)
                if scratch.parent != scratch_parent:
                    raise GitCaptureError(
                        "Git capture scratch escaped its selected parent"
                    )
                if scratch.stat().st_dev != workspace_device:
                    raise GitCaptureError(
                        "Git capture scratch crossed the worktree filesystem"
                    )
                if label == "external-workspace" and any(
                    scratch.is_relative_to(root)
                    for root in (workspace, outermost)
                ):
                    raise GitCaptureError(
                        "external Git capture scratch entered a worktree"
                    )
            except GitCaptureError:
                raise
            except OSError as exc:
                raise GitCaptureError(
                    f"Git capture scratch validation failed: {exc}"
                ) from exc
            yield scratch
            return

    details = "; ".join(creation_errors) or "no same-device safe parent"
    raise GitCaptureError(
        f"Git capture scratch has no writable safe location: {details}"
    )


def _git_argv(*args: str) -> list[str]:
    """Build capture-only Git argv with executable facilities disabled."""

    return ["git", *_GIT_SAFE_OPTIONS, *args]


def _sanitized_git_env(**overrides: str) -> dict[str, str]:
    """Return a scoped Git environment without ambient control variables."""

    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("GIT_")
    }
    env.update({
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_PAGER": "cat",
    })
    env.update(overrides)
    return env


def _bounded_subprocess_stdout(
    args: list[str],
    *,
    cwd: str,
    env: dict[str, str],
    max_bytes: int,
) -> bytes:
    """Run a capture-only command without buffering unbounded stdout.

    Two bounded producer threads drain stdout and stderr concurrently.  The
    small queue provides backpressure, stdout is retained only through
    ``max_bytes + 1``, and stderr is reduced to a diagnostic tail.  Git's
    external-command facilities are disabled by the caller's argv/env, so
    killing the direct process also closes the complete producer here.
    """

    events: queue.Queue[tuple[str, bytes | None]] = queue.Queue(maxsize=16)
    process = subprocess.Popen(
        args,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    assert process.stderr is not None

    def _drain(label: str, stream) -> None:
        try:
            while chunk := stream.read(64 * 1024):
                events.put((label, chunk))
        finally:
            stream.close()
            events.put((label, None))

    readers = (
        threading.Thread(
            target=_drain,
            args=("stdout", process.stdout),
            daemon=True,
        ),
        threading.Thread(
            target=_drain,
            args=("stderr", process.stderr),
            daemon=True,
        ),
    )
    for reader in readers:
        reader.start()

    output = bytearray()
    stderr_tail = bytearray()
    oversized = False
    active_readers = len(readers)
    try:
        while active_readers:
            label, chunk = events.get()
            if chunk is None:
                active_readers -= 1
                continue
            if label == "stdout":
                remaining = max_bytes + 1 - len(output)
                if remaining > 0:
                    output.extend(chunk[:remaining])
                if len(output) > max_bytes:
                    oversized = True
                    if process.poll() is None:
                        process.kill()
            else:
                stderr_tail.extend(chunk)
                if len(stderr_tail) > 64 * 1024:
                    del stderr_tail[:-64 * 1024]
        returncode = process.wait()
    except BaseException:
        if process.poll() is None:
            process.kill()
        process.wait()
        raise
    finally:
        for reader in readers:
            reader.join()

    if oversized:
        raise GitCaptureError(
            f"capture Git diff exceeded max_diff_bytes={max_bytes}"
        )
    if returncode != 0:
        result = subprocess.CompletedProcess(
            args,
            returncode,
            stdout=bytes(output),
            stderr=bytes(stderr_tail),
        )
        raise GitCaptureError(_git_failure("capture Git diff", result))
    return bytes(output)


def _git_failure(command: str, result: subprocess.CompletedProcess) -> str:
    stderr = result.stderr.decode("utf-8", errors="replace").strip()
    return f"{command} failed with exit {result.returncode}: {stderr}"


def _has_git_admin_marker(cwd: str | os.PathLike[str] | None) -> bool:
    """Whether ``cwd`` is lexically inside a checkout with a .git marker.

    This distinguishes the supported outside-a-repository empty result from a
    broken Git probe inside a real checkout. It does not replace Git's own
    repository resolution.
    """

    try:
        current = Path(cwd or os.getcwd()).resolve()
    except (OSError, RuntimeError):
        return False
    for candidate in (current, *current.parents):
        marker = candidate / ".git"
        if marker.is_dir():
            if (marker / "HEAD").exists():
                return True
            continue
        if marker.is_file():
            try:
                prefix, _, rendered = marker.read_text(
                    encoding="utf-8",
                    errors="replace",
                ).partition(":")
            except OSError:
                continue
            if prefix.strip() != "gitdir" or not rendered.strip():
                continue
            git_dir = Path(rendered.strip())
            if not git_dir.is_absolute():
                git_dir = marker.parent / git_dir
            if (git_dir / "HEAD").exists():
                return True
    return False


if TYPE_CHECKING:
    from ..backend.protocol import Backend
    from ..corpus.store import Corpus
    from ..types.spec import Spec


_IMPLEMENTER_ROLE_PROTOCOL = """You are an implementation agent for the assay / crucible toolchain.

Your job: make targeted, structured code changes per a written directive.

Rules:
- Use only the exact tools rendered in this prompt.
- Write or edit files to satisfy the directive's intent.
- Stay strictly inside the declared `files_in_scope`. Do not modify
  files outside that list.
- Create exactly the files listed in `creates_paths` (if any).
- Stop when the changes are complete. Do NOT iterate, polish, or
  refactor beyond what the directive specifies.
- Do NOT commit to git or mutate the project registry, lockfile, or unrelated
  configuration.
- Build or test only when an exact attached tool explicitly provides that
  operation. Workflow-owned validation remains authoritative.
"""


_IMPLEMENTER_OUTPUT_CONTRACT = PromptOutputContract(
    completion_report=(
        "When the implementation is complete, respond with the literal token "
        "`DONE` followed by a one-paragraph summary of what changed. If the "
        "directive cannot be completed, respond with `BLOCKED` followed by "
        "the concrete blocker; never use `DONE` for partial work."
    ),
    validation=(
        "Assay captures and validates the Git delta independently of this "
        "completion report."
    ),
)


def _implementer_system_prompt(
    backend: object,
    *,
    invocation_tools: tuple[Tool, ...] = (),
    prompt_toolset: ToolSet | None = None,
) -> str:
    """Render the implementation protocol for the routed tool surface."""

    profile = profile_for_backend(
        backend,
        invocation_tools=invocation_tools,
        toolset=prompt_toolset,
    )
    tool_names = {tool.name for tool in profile.tools}
    has_build_tool = any(
        name == "build_project" or name.endswith("__build_project")
        for name in tool_names
    )
    validation_clause = (
        "A bounded build tool is attached. Use at most one targeted "
        "self-check/repair loop when the directive calls for it; the workflow "
        "reruns the authoritative gate."
        if has_build_tool
        else (
            "No build/test tool is attached. The workflow owns authoritative "
            "testing; do not claim to have run it."
        )
    )
    return render_prompt_abi_v2(
        _IMPLEMENTER_ROLE_PROTOCOL,
        profile=profile,
        output=_IMPLEMENTER_OUTPUT_CONTRACT,
        security_clauses=(
            "The declared files and mutation regions are an authority ceiling.",
            validation_clause,
        ),
    )


# Compatibility fixture for callers that inspect the tool-free default.
_IMPLEMENTER_SYSTEM_PROMPT = render_prompt_abi_v2(
    _IMPLEMENTER_ROLE_PROTOCOL,
    profile=PromptABIProfile(overlay="generic"),
    output=_IMPLEMENTER_OUTPUT_CONTRACT,
    security_clauses=(
        "The declared files and mutation regions are an authority ceiling.",
        "No build/test tool is attached. The workflow owns authoritative "
        "testing; do not claim to have run it.",
    ),
)


def _file_size_hint(rel_path: str) -> str:
    """Annotate an existing path with provider-neutral byte-size context."""

    from pathlib import Path as _P
    try:
        sz = _P(rel_path).stat().st_size
    except OSError:
        return ""
    kb = sz / 1024
    return f"  *({kb:.0f} KB)*"


def _build_implementer_prompt(
    spec: "Spec",
    slice_id: str,
    directive: ImplementationDirective,
) -> str:
    """Render the user-facing prompt that describes one slice."""
    lines: list[str] = [
        f"# DIRECTIVE — slice `{slice_id}`",
        "",
        f"**Verify mode:** {directive.verify_mode}",
        "",
    ]
    if directive.requirement_refs:
        lines.extend([
            "## Authoritative root requirements",
            "",
            (
                "The clauses below are exact, verbatim requirements from the "
                "task root. They are authoritative and win over the generated "
                "implementation intent if the two conflict."
            ),
            "",
        ])
        for index, clause in enumerate(directive.requirement_refs, start=1):
            lines.extend([
                f"### Root requirement {index}",
                "",
                clause,
                "",
            ])
        lines.extend([
            "## Generated implementation intent",
            "",
        ])
    else:
        lines.extend([
            "## Intent",
            "",
        ])
    lines.extend([
        directive.intent.strip(),
        "",
        "## Files in scope",
        "",
    ])
    if directive.files_in_scope:
        # Pre-compute provider-neutral size hints. Tool-specific limits belong
        # to the attached catalogue rendered by prompt ABI v2.
        for p in directive.files_in_scope:
            lines.append(f"- `{p}`{_file_size_hint(p)}")
    else:
        lines.append("- (none declared — minimize edits)")
    lines.append("")
    lines.append("## Files to create")
    lines.append("")
    if directive.creates_paths:
        for p in directive.creates_paths:
            lines.append(f"- `{p}`")
    else:
        lines.append("- (none — directive is edit-only)")
    lines.append("")

    if directive.read_only_references:
        lines.extend([
            "## Read-only source references",
            "",
            "These references identify existing evidence, not mutation scope. "
            "They grant no write permission. Only Files in scope and Files to "
            "create above authorize edits. Root requirements remain authoritative. "
            "Consumers must verify the whole-file hashes before relying on these "
            "line excerpts; the reference records alone do not prove verification.",
            "",
        ])
        for ref in directive.read_only_references:
            lines.append(
                f"- `{ref.reference_id}`: `{ref.path}` lines "
                f"{ref.start_line}-{ref.end_line}, `{ref.sha256}`"
            )
        lines.append("")

    if directive.blast_radius:
        lines.append("## Blast-radius constraints")
        lines.append("")
        lines.append("```json")
        lines.append(
            json.dumps(directive.blast_radius, indent=2, sort_keys=True)
        )
        lines.append("```")
        lines.append("")
    if directive.mutation_regions:
        # Source-level mutation scope: spell out the per-file marker
        # constraint so the implementer LLM sees the scoping policy at
        # authoring time. Deterministic enforcement of the diff vs.
        # marker positions happens in the consumer's validator.
        lines.append("## Mutation regions")
        lines.append("")
        lines.append(
            "The following files have source-level mutation regions. "
            "Only edit lines BETWEEN the named start/end marker comments "
            "in these files. Edits outside marked regions will be "
            "rejected by the mutation_scope validator."
        )
        lines.append("")
        for r in directive.mutation_regions:
            lines.append(
                f"- `{r.file}` between `{r.start_marker}` and "
                f"`{r.end_marker}`"
            )
        lines.append("")
        lines.append(
            "Files NOT listed here are bounded only by `files_in_scope` "
            "/ `creates_paths` above — no per-line region constraint."
        )
        lines.append("")
    lines.append("## Project context")
    lines.append("")
    lines.append(f"- Project: `{spec.project_context.project}`")
    lines.append(f"- Repo URL: `{spec.project_context.repo_url}`")
    lines.append(f"- Base commit: `{spec.project_context.base_commit}`")
    if spec.project_context.submodule_path:
        lines.append(
            f"- Submodule path: `{spec.project_context.submodule_path}`"
        )
    lines.append("")
    lines.append("Begin. Respond with `DONE` + summary when complete.")
    return "\n".join(lines)


def _git_capture(args: list[str], *, cwd: str | None = None) -> str:
    try:
        r = subprocess.run(
            _git_argv(*args),
            check=False,
            capture_output=True,
            text=True,
            cwd=cwd,
            env=_sanitized_git_env(GIT_OPTIONAL_LOCKS="0"),
        )
        return r.stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def _capture_pathspecs(
    path_scope: tuple[str, ...] | None,
    exclude_pathspecs: tuple[str, ...],
) -> tuple[str, ...] | None:
    """Return exact include pathspecs plus trusted Git-native exclusions.

    Every scope path reaches Git as written. No leading component is dropped
    and the working tree is not consulted, so a path names the same file
    before and after that file exists. The one rewrite is the Windows
    separator.

    Ordinary callers retain ``GIT_LITERAL_PATHSPECS=1`` and their historical
    plain path strings.  ``implement`` supplies trusted exclusion magic for its
    Corpus root; in that mode every untrusted include is explicitly wrapped in
    ``:(literal)`` before Git pathspec magic is enabled.
    """

    if path_scope and os.name == "nt":
        # Planner-authored paths may use OS-native separators. Normalize
        # backslashes only on Windows; on POSIX a backslash is a valid literal
        # filename byte and rewriting it would silently scope a different
        # path. Git accepts forward slashes on Windows.
        path_scope = tuple(path.replace("\\", "/") for path in path_scope)
    if not exclude_pathspecs:
        return path_scope
    includes = tuple(
        f":(literal){path}"
        for path in (path_scope or ())
    )
    return (*includes, *exclude_pathspecs)


def _corpus_capture_exclusions(
    corpus: "Corpus",
    *,
    cwd: str | None,
) -> tuple[str, ...]:
    """Exclude an in-worktree Corpus without touching either filesystem tree.

    The pathspec is top-anchored and literal, so a broad/empty implementation
    scope cannot re-include the Corpus and unusual path bytes cannot become Git
    pathspec syntax.  Resolving with ``strict=False`` avoids a raw filesystem
    exception if a concurrent cleanup removes a Corpus descendant; the Corpus
    root itself was already created by ``Corpus``.
    """

    try:
        probe = Path(cwd or os.getcwd()).resolve(strict=False)
        worktree_root = _enclosing_git_worktree(probe)
        if worktree_root is None:
            return ()
        corpus_root = Path(corpus.spec_dir).resolve(strict=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise GitCaptureError(
            f"resolve Corpus capture exclusion failed: {exc}"
        ) from exc
    try:
        relative = corpus_root.relative_to(worktree_root)
    except ValueError:
        return ()
    if not relative.parts:
        raise GitCaptureError(
            "Corpus root cannot equal the implementation capture worktree"
        )
    return (f":(top,literal,exclude){relative.as_posix()}",)


def _guard_against_git_filters(
    *,
    cwd: str | None,
    path_scope: tuple[str, ...] | None,
    env: dict[str, str],
) -> tuple[str, ...]:
    """Fail closed before Git can execute a content filter during capture.

    ``.gitattributes`` is part of the model-writable working tree.  A model
    could therefore attach a configured clean filter to a path that capture's
    ``status``, ``diff``, or intent-to-add operation is about to inspect.
    Capture suppresses global/system configuration, rejects repository-local
    filter commands and config includes, and can then safely treat any active
    but unconfigured filter attribute as an inert passthrough.  This permits
    pre-materialized binary-store paths without executing their driver.

    Return the untracked names so the diff path can reuse the same safe
    enumeration for its temporary intent-to-add entries.
    """

    configured = subprocess.run(
        _git_argv(
            "config",
            "--no-includes",
            "--name-only",
            "--get-regexp",
            ".*",
        ),
        check=False,
        capture_output=True,
        cwd=cwd,
        env=env,
    )
    if configured.returncode not in {0, 1}:
        raise GitCaptureError(
            _git_failure(
                "inspect repository-local Git filter controls",
                configured,
            )
        )
    configured_names = tuple(
        line.decode("utf-8", errors="surrogateescape")
        for line in configured.stdout.splitlines()
        if line
    )
    executable_controls = tuple(
        name
        for name in configured_names
        if (
            name.lower().startswith(("include.", "includeif."))
            or re.fullmatch(
                r"filter\..*\.(clean|smudge|process|required)",
                name.lower(),
            )
        )
    )
    if executable_controls:
        raise GitCaptureError(
            "refusing Git capture because repository-local filter/include "
            f"controls are configured: {executable_controls[0][:500]}"
        )

    common = _git_argv("ls-files")
    tracked_cmd = [*common, "--cached", "-z"]
    untracked_cmd = [*common, "--others", "--exclude-standard", "-z"]
    if path_scope:
        tracked_cmd.extend(("--", *path_scope))
        untracked_cmd.extend(("--", *path_scope))

    tracked_result = subprocess.run(
        tracked_cmd,
        check=False,
        capture_output=True,
        cwd=cwd,
        env=env,
    )
    if tracked_result.returncode != 0:
        probe = subprocess.run(
            _git_argv("rev-parse", "--is-inside-work-tree"),
            check=False,
            capture_output=True,
            cwd=cwd,
            env=env,
        )
        if not (
            probe.returncode == 0 and probe.stdout.strip() == b"true"
        ) and not _has_git_admin_marker(cwd):
            # Preserve the supported outside-a-repository empty result. Real
            # checkouts (including environment-selected and
            # linked worktrees) must fail closed on a broken enumeration.
            return ()
        raise GitCaptureError(
            _git_failure(
                "enumerate tracked paths for Git filter guard",
                tracked_result,
            )
        )
    untracked_result = subprocess.run(
        untracked_cmd,
        check=False,
        capture_output=True,
        cwd=cwd,
        env=env,
    )
    if untracked_result.returncode != 0:
        raise GitCaptureError(
            _git_failure(
                "enumerate untracked paths for Git filter guard",
                untracked_result,
            )
        )

    untracked_records = tuple(
        record for record in untracked_result.stdout.split(b"\0") if record
    )
    relevant_records = tuple(
        dict.fromkeys(
            record
            for output in (tracked_result.stdout, untracked_result.stdout)
            for record in output.split(b"\0")
            if record
        )
    )
    if relevant_records:
        attributes = subprocess.run(
            _git_argv("check-attr", "-z", "filter", "--stdin"),
            check=False,
            capture_output=True,
            input=b"\0".join(relevant_records) + b"\0",
            cwd=cwd,
            env=env,
        )
        if attributes.returncode != 0:
            raise GitCaptureError(
                _git_failure("resolve Git filter attributes", attributes)
            )
        fields = attributes.stdout.split(b"\0")
        if fields and not fields[-1]:
            fields.pop()
        if len(fields) != 3 * len(relevant_records):
            raise GitCaptureError(
                "resolve Git filter attributes returned malformed output"
            )
    return tuple(os.fsdecode(record) for record in untracked_records)


def capture_workspace_delta(
    *,
    cwd: str | None = None,
    path_scope: tuple[str, ...] | None = None,
    max_diff_bytes: int | None = None,
) -> tuple[bytes, tuple[str, ...], str]:
    """Public capture of the current working-tree delta.

    Returns ``(diff_bytes, files_modified, base_commit)`` for the optionally
    scoped working tree. ``base_commit`` is ``"unknown"`` when HEAD cannot be
    resolved. This is the same Git capture the
    ``implement`` primitive performs after its backend invocation
    (including intent-to-add visibility for brand-new files in an isolated
    temporary index/object store, and Windows path-separator normalization).

    This lets manual or externally driven completion paths record the same
    byte-level delta as ``implement`` without duplicating Git plumbing.

    ``path_scope`` is read literally. Each entry is the path of one file or
    directory relative to ``cwd`` and reaches Git as an exact name. No leading
    component is dropped, including one equal to the name of the ``cwd``
    directory, and the working tree is not consulted to interpret an entry, so
    it names the same path before and after that path exists. A caller whose
    paths are relative to another directory rewrites them before the call.
    ``None`` or an empty tuple applies no scope.

    ``max_diff_bytes`` enables a streaming hard limit for callers handling
    untrusted worktrees. The producer is terminated as soon as it emits one
    byte beyond the limit; the oversized patch is never materialized.
    """
    if (
        max_diff_bytes is not None
        and (
            isinstance(max_diff_bytes, bool)
            or not isinstance(max_diff_bytes, int)
            or max_diff_bytes <= 0
        )
    ):
        raise ValueError("max_diff_bytes must be a positive integer or None")
    if path_scope is not None:
        # Several Git commands read the scope. A one-shot iterable would be
        # spent by the first and leave the rest unscoped.
        path_scope = tuple(path_scope)
    return (
        _git_diff_bytes(
            cwd=cwd,
            path_scope=path_scope,
            max_diff_bytes=max_diff_bytes,
        ),
        _git_modified_files(cwd=cwd, path_scope=path_scope),
        _git_head_sha(cwd=cwd),
    )


def _git_diff_bytes(
    cwd: str | None = None,
    path_scope: tuple[str, ...] | None = None,
    max_diff_bytes: int | None = None,
    *,
    exclude_pathspecs: tuple[str, ...] = (),
) -> bytes:
    # `git diff HEAD` only shows tracked changes, so a slice that creates
    # a brand-new file (matching its `creates_paths` declaration) would
    # otherwise produce an empty diff and be misclassified as no change even
    # though the implementer correctly wrote what the directive required.
    # `git add --intent-to-add` (-N) marks
    # untracked files as "to be added" without staging content, which
    # makes them visible to `git diff` as new-file diffs while keeping
    # the index otherwise clean.
    path_scope = _capture_pathspecs(path_scope, exclude_pathspecs)
    literal_pathspecs = "0" if exclude_pathspecs else "1"

    # Never borrow the checkout's live index for capture. An add-N/reset
    # sequence can rewrite raw index
    # extensions and briefly left intent-to-add entries behind if the process
    # died between the two commands.  A copied temporary index preserves the
    # caller's staged state while making the entire capture non-mutating and
    # crash-safe.  A temporary object directory contains the empty blob that
    # Git may materialize for intent-to-add; the repository object database is
    # read through an alternate and remains byte-for-byte untouched.
    repo_cwd = cwd or os.getcwd()
    base_env = _sanitized_git_env(
        GIT_OPTIONAL_LOCKS="0",
        GIT_LITERAL_PATHSPECS="1",
    )
    try:
        worktree_result = subprocess.run(
            _git_argv("rev-parse", "--is-inside-work-tree"),
            check=False,
            capture_output=True,
            cwd=repo_cwd,
            env=base_env,
        )
        if (
            worktree_result.returncode != 0
            or worktree_result.stdout.strip() != b"true"
        ):
            if _has_git_admin_marker(repo_cwd):
                raise GitCaptureError(
                    _git_failure("resolve Git worktree", worktree_result)
                )
            return b""
        root_result = subprocess.run(
            _git_argv(
                "rev-parse", "--path-format=absolute", "--show-toplevel",
            ),
            check=False,
            capture_output=True,
            cwd=repo_cwd,
            env=base_env,
        )
        if root_result.returncode != 0:
            raise GitCaptureError(
                _git_failure("resolve Git worktree root", root_result)
            )
        root_text = os.fsdecode(root_result.stdout).strip()
        if not root_text:
            raise GitCaptureError("resolve Git worktree root returned empty output")
        worktree_root = Path(root_text).resolve(strict=True)
        resolved_cwd = Path(repo_cwd).resolve(strict=True)
        if not resolved_cwd.is_relative_to(worktree_root):
            raise GitCaptureError(
                "resolved Git cwd is outside the reported worktree root"
            )
        index_result = subprocess.run(
            _git_argv(
                "rev-parse", "--path-format=absolute", "--git-path", "index",
            ),
            check=False,
            capture_output=True,
            cwd=repo_cwd,
            env=base_env,
        )
        objects_result = subprocess.run(
            _git_argv(
                "rev-parse", "--path-format=absolute", "--git-path", "objects",
            ),
            check=False,
            capture_output=True,
            cwd=repo_cwd,
            env=base_env,
        )
        if index_result.returncode != 0:
            raise GitCaptureError(_git_failure("resolve Git index", index_result))
        if objects_result.returncode != 0:
            raise GitCaptureError(
                _git_failure("resolve Git object directory", objects_result)
            )
        source_index = Path(os.fsdecode(index_result.stdout).strip())
        source_objects = Path(os.fsdecode(objects_result.stdout).strip())

        with _git_capture_scratch(worktree_root) as temp_root:
            temp_index = temp_root / "index"
            temp_objects = temp_root / "objects"
            temp_objects.mkdir()
            if source_index.is_file():
                source_index_mtime_ns = source_index.stat().st_mtime_ns
                shutil.copyfile(source_index, temp_index)
                # Keep Git's racy-clean boundary identical to the source
                # index. ``copyfile`` assigns a fresh mtime; a same-size
                # worktree rewrite can then look older than the temporary
                # index and be trusted without content comparison. Preserve
                # only mtime (not modes/xattrs) so isolated capture remains
                # metadata-minimal while retaining Git's safety check.
                temp_index_stat = temp_index.stat()
                os.utime(
                    temp_index,
                    ns=(
                        temp_index_stat.st_atime_ns,
                        source_index_mtime_ns,
                    ),
                )

                # A split index names its shared backing file relative to the
                # index location.  Copy that immutable backing alongside the
                # temporary index so capture remains safe without ever
                # opening the checkout's index for write.
                shared_result = subprocess.run(
                    _git_argv("rev-parse", "--shared-index-path"),
                    check=False,
                    capture_output=True,
                    cwd=repo_cwd,
                    env=base_env,
                )
                if shared_result.returncode != 0:
                    raise GitCaptureError(
                        _git_failure("resolve shared Git index", shared_result)
                    )
                shared_text = os.fsdecode(shared_result.stdout).strip()
                if shared_result.returncode == 0 and shared_text:
                    shared_index = Path(shared_text)
                    if not shared_index.is_file():
                        raise GitCaptureError(
                            f"shared Git index is missing: {shared_index}"
                        )
                    shutil.copyfile(
                        shared_index,
                        temp_root / shared_index.name,
                    )

            env = _sanitized_git_env(
                GIT_INDEX_FILE=str(temp_index),
                GIT_OBJECT_DIRECTORY=str(temp_objects),
                # Planner-authored paths are exact repository-relative names,
                # never Git pathspec expressions.  This also makes leading
                # ':' filenames safe. Trusted Corpus exclusions use explicit
                # magic while wrapping every planner path in ``:(literal)``.
                GIT_LITERAL_PATHSPECS=literal_pathspecs,
                GIT_OPTIONAL_LOCKS="0",
                GIT_ALTERNATE_OBJECT_DIRECTORIES=str(source_objects),
            )

            if not temp_index.exists():
                read_tree = subprocess.run(
                    _git_argv("read-tree", "HEAD"),
                    check=False,
                    capture_output=True,
                    cwd=repo_cwd,
                    env=env,
                )
                if read_tree.returncode != 0:
                    raise GitCaptureError(
                        _git_failure("initialize temporary Git index", read_tree)
                    )

            new_paths = _guard_against_git_filters(
                cwd=repo_cwd,
                path_scope=path_scope,
                env=env,
            )
            if new_paths:
                add_paths = (
                    tuple(f":(literal){path}" for path in new_paths)
                    if exclude_pathspecs
                    else new_paths
                )
                add_result = subprocess.run(
                    _git_argv(
                        "add", "--intent-to-add", "--", *add_paths,
                    ),
                    check=False,
                    capture_output=True,
                    cwd=repo_cwd,
                    env=env,
                )
                if add_result.returncode != 0:
                    raise GitCaptureError(
                        _git_failure("mark temporary intent-to-add", add_result)
                    )

            # Intent-to-add changed the temporary index's tracked-name set.
            # Re-resolve attributes immediately before diff rather than
            # assuming the earlier preflight remains representative.
            _guard_against_git_filters(
                cwd=repo_cwd,
                path_scope=path_scope,
                env=env,
            )

            # `--binary` emits literal, re-applicable binary patches and
            # implies `--full-index`.
            diff_cmd: list[str] = [
                *_git_argv("diff"),
                "--no-ext-diff", "--no-textconv", "--binary", "HEAD",
            ]
            if path_scope:
                diff_cmd.extend(("--", *path_scope))
            if max_diff_bytes is not None:
                return _bounded_subprocess_stdout(
                    diff_cmd,
                    cwd=repo_cwd,
                    env=env,
                    max_bytes=max_diff_bytes,
                )
            result = subprocess.run(
                diff_cmd,
                check=False,
                capture_output=True,
                cwd=repo_cwd,
                env=env,
            )
            if result.returncode != 0:
                raise GitCaptureError(_git_failure("capture Git diff", result))
            return result.stdout
    except GitCaptureError:
        raise
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise GitCaptureError(f"Git diff capture failed: {exc}") from exc


def _git_modified_files(
    cwd: str | None = None,
    path_scope: tuple[str, ...] | None = None,
    *,
    exclude_pathspecs: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Parse ``git status --porcelain`` into a list of changed paths.

    When ``path_scope`` is supplied, ``git status -- <paths>`` filters
    the report to the declared file set. This is the bracket that lets
    sibling slices in the same tier run concurrently without smearing
    each other's modifications into the captured Implementation record.
    """
    path_scope = _capture_pathspecs(path_scope, exclude_pathspecs)
    # ``--untracked-files=all`` (-uall) forces git to list every untracked
    # FILE individually. Without it, porcelain-v1 collapses an entirely-
    # untracked directory into a single ``dir/`` entry — so a greenfield
    # slice that stands up a new crate (all files under a brand-new dir,
    # each declared in ``creates_paths``) reports its modifications at
    # DIRECTORY granularity in the unscoped scope-escape capture (line ~611),
    # which never string-matches the file-level ``creates_paths``. The whole
    # new dir then reads as one out-of-scope write and the slice is wrongly
    # classified ``hard_fail_scope_escape``. -uall keeps this capture at the
    # same file granularity as the path-scoped capture and the declared scope.
    args = [
        *_git_argv("status"),
        "--porcelain=v1", "-z", "--untracked-files=all",
    ]
    if path_scope:
        args.extend(("--", *path_scope))
    env = _sanitized_git_env(
        # Status is a read-side observation. Suppress its optional index
        # refresh so capture cannot churn raw stat-cache extensions.
        GIT_OPTIONAL_LOCKS="0",
        GIT_LITERAL_PATHSPECS="0" if exclude_pathspecs else "1",
    )
    try:
        _guard_against_git_filters(
            cwd=cwd,
            path_scope=path_scope,
            env=env,
        )
        result = subprocess.run(
            args,
            check=False,
            capture_output=True,
            cwd=cwd,
            env=env,
        )
        if result.returncode != 0:
            probe = subprocess.run(
                _git_argv("rev-parse", "--is-inside-work-tree"),
                check=False,
                capture_output=True,
                cwd=cwd,
                env=env,
            )
            if probe.returncode == 0 and probe.stdout.strip() == b"true":
                raise GitCaptureError(_git_failure("capture Git status", result))
            if _has_git_admin_marker(cwd):
                raise GitCaptureError(
                    _git_failure("resolve Git worktree", probe)
                )
            return ()
        records = result.stdout.split(b"\0")
    except GitCaptureError:
        raise
    except (OSError, subprocess.SubprocessError) as exc:
        raise GitCaptureError(f"Git status capture failed: {exc}") from exc
    files: list[str] = []
    index = 0
    while index < len(records):
        record = records[index]
        index += 1
        if len(record) < 4:
            continue
        status = record[:2]
        path = os.fsdecode(record[3:])
        if path:
            files.append(path)
        # In porcelain -z mode a rename/copy reports the destination in this
        # record and the source as the next NUL record. Keep the destination,
        # matching the public contract, and skip the source.
        if b"R" in status or b"C" in status:
            index += 1
    return tuple(files)


def _git_head_sha(cwd: str | None = None) -> str:
    try:
        result = subprocess.run(
            _git_argv("rev-parse", "HEAD"),
            check=False,
            capture_output=True,
            cwd=cwd,
            env=_sanitized_git_env(GIT_OPTIONAL_LOCKS="0"),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        if _has_git_admin_marker(cwd):
            raise GitCaptureError(f"resolve Git HEAD failed: {exc}") from exc
        return "unknown"
    sha = result.stdout.decode("ascii", errors="replace").strip()
    if result.returncode == 0 and sha:
        return sha
    if _has_git_admin_marker(cwd):
        raise GitCaptureError(_git_failure("resolve Git HEAD", result))
    return "unknown"


def _read_assay_version() -> str:
    try:
        from importlib.metadata import version

        return version("assay")
    except Exception:  # noqa: BLE001
        return "0.0.0"


def _snapshot_execution(
    *, backend_name: str, model: str | None
) -> ExecutionContext:
    return ExecutionContext(
        timestamp=_datetime.datetime.now(
            _datetime.timezone.utc
        ).isoformat(timespec="seconds"),
        machine=socket.gethostname(),
        cwd=os.getcwd(),
        toolchain={
            "assay": _read_assay_version(),
            "python": platform.python_version(),
        },
        backend=backend_name,
        model=model,
        recipe=None,
        recipe_commit=None,
        gpu_model=None,
        run_id=None,
    )


def _serialize_transcript(transcript: tuple[Message, ...]) -> bytes:
    payload = [{"role": m.role, "content": m.content} for m in transcript]
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def _classify(
    error_kind: str | None,
    files_modified: tuple[str, ...],
    verify_mode: str,
    *,
    scope_escapes: tuple[str, ...] = (),
) -> str:
    """Map a slice's post-implement signals to a classification string.

    A backend error always becomes ``hard_fail_<kind>``. Partial writes remain
    captured as evidence, but an incomplete terminal call must never become a
    successful implementation. When no authoritative backend error remains, a
    file-level escape becomes
    ``hard_fail_scope_escape``. With no modified files, mutating directives are
    ``hard_fail_empty_diff`` while ``verify_mode='skip'`` is
    ``soft_fail_no_changes``. All other results are ``success``.

    Region-level mutation checks are consumer-owned and may apply a separate
    classification. Consumers must also roll back ``scope_escapes``; this
    function classifies evidence but does not modify the working tree.
    """
    if error_kind is not None:
        # Tool calls may write to disk before a provider timeout, transport
        # failure, or max-turn ceiling. Keep those files in the Implementation
        # record for audit/recovery, but do not promote an incomplete terminal
        # call to success merely because mutations were observed.
        return f"hard_fail_{error_kind}"
    if scope_escapes:
        # The implementer wrote outside its declared scope. Even if the
        # in-scope diff would have built cleanly, allowing the slice to
        # pass leaves the out-of-scope files on disk as orphans that
        # the path-scoped rollback can't see — they then bias the next
        # slice's verify baseline. Escalating lets the consumer clean both the
        # in-scope diff and escaped paths before another slice observes the
        # workspace.
        return "hard_fail_scope_escape"
    if not files_modified:
        # An empty diff is a hard fail when the directive expected
        # changes. Audit-only slices (verify_mode == "skip") legitimately
        # produce no diff, so they keep the soft classification. This
        # asymmetry prevents misconfigured mutating slices from
        # passing through as success-adjacent downstream.
        return ("soft_fail_no_changes"
                if verify_mode == "skip"
                else "hard_fail_empty_diff")
    return "success"


def implement(
    spec: "Spec",
    slice_id: str,
    directive: ImplementationDirective,
    *,
    backend: "Backend",
    corpus: "Corpus",
    log_dir: Path | None = None,
    max_turns: int = 50,
    budget_usd: float | None = None,
    system_prompt: str | None = None,
    allowed_tools: tuple[str, ...] | None = None,
    prompt_toolset: ToolSet | None = None,
    cwd: str | None = None,
    after_invoke: Callable[[InvocationResult, ContentHash], None] | None = None,
    before_capture: Callable[[], None] | None = None,
) -> Implementation:
    """Implementer primitive — see module docstring for full contract.

    Args:
      system_prompt: Optional override of the default implementation protocol.
        The default forbids running tests because verification is a separate
        phase. A workflow that expects test-driven iteration must provide its
        own prompt and preserve the required completion/no-commit protocol.
      allowed_tools: Optional names encoded as ``Tool`` descriptors with empty
        input schemas on ``Invocation.tools``. ``None`` or an empty tuple
        produces ``Invocation.tools=None``. The concrete backend decides how
        that request maps to its tool surface.
      prompt_toolset: Metadata-only ToolSet already attached to ``backend``.
        Prompt ABI v2 renders its exact descriptor names and descriptions; the
        argument does not attach, authorize, or dispatch tools.
      cwd: Working directory used for the internal ``git rev-parse`` /
        ``git diff`` / ``git status`` captures. When None (default),
        process cwd is used — fine for single-threaded workflows.
        Multi-threaded fan-out MUST pass each thread's per-workspace path here;
        otherwise concurrent calls share process cwd and can capture smeared
        diffs. The backend's own cwd is configured separately.
      log_dir: Accepted but unused; logging is configured on the backend and
        this primitive does not write to ``log_dir``.
      after_invoke: Optional workflow-owned checkpoint callback invoked exactly
        once after a backend returns an ``InvocationResult`` and its transcript
        is durably stored, but before ``before_capture``. It receives the exact
        result and transcript reference. If transcript persistence or this
        callback fails, ``before_capture`` still runs exactly once for provider
        safety, then the original error propagates before any later Git capture.
      before_capture: Optional workflow-owned safety callback invoked after
        ``after_invoke`` but before any post-invocation Git diff/status. Durable
        workspace journals use this to restore Git policy controls that an
        implementation may have mutated. A workflow combining safety and
        deterministic finalization in this callback must self-gate the latter
        on successful invocation checkpoint state.
    """
    base_commit = _git_head_sha(cwd=cwd)
    scope = tuple(directive.files_in_scope) + tuple(directive.creates_paths)
    corpus_exclusions = _corpus_capture_exclusions(corpus, cwd=cwd)

    # Baseline snapshots BEFORE the implementer runs.  ``pre_dirty`` supports
    # the repo-wide scope-escape check below.  The scoped diff/status pair is
    # separate evidence: post-invocation ``diff_bytes`` and ``files_modified``
    # are intentionally cumulative HEAD→workspace artifacts, so a path made
    # dirty by an earlier tier cannot by itself prove that this invocation
    # wrote anything.  Comparing the same capture on both sides preserves the
    # cumulative artifact while keeping transport/max-turn salvage causal.
    pre_dirty = set(_git_modified_files(
        cwd=cwd,
        path_scope=None,
        exclude_pathspecs=corpus_exclusions,
    ))
    pre_scope_diff_bytes = _git_diff_bytes(
        cwd=cwd,
        path_scope=scope or None,
        exclude_pathspecs=corpus_exclusions,
    )
    pre_scope_files_modified = _git_modified_files(
        cwd=cwd,
        path_scope=scope or None,
        exclude_pathspecs=corpus_exclusions,
    )

    tool_descriptors: tuple = (
        tuple(Tool(name=n, description="", input_schema={}) for n in allowed_tools)
        if allowed_tools else None
    )
    invocation = Invocation(
        system=(
            system_prompt
            if system_prompt is not None
            else _implementer_system_prompt(
                backend,
                invocation_tools=tool_descriptors or (),
                prompt_toolset=prompt_toolset,
            )
        ),
        user=_build_implementer_prompt(spec, slice_id, directive),
        tools=tool_descriptors,
        max_turns=max_turns,
        budget_usd=budget_usd,
    )

    result = backend.invoke(invocation)
    # Retain the first checkpoint failure while still running the workflow's
    # provider-safety callback exactly once. Crucible uses its callback-local
    # accounting state to quiesce but skip deterministic finalization when this
    # checkpoint did not complete. Assay itself performs no later Git capture
    # in that state.
    checkpoint_error: BaseException | None = None
    transcript_ref: ContentHash | None = None
    try:
        transcript_ref = corpus.put_artifact(
            _serialize_transcript(result.transcript),
            kind="transcripts",
            name_hint=f"implement_{slice_id}.json",
        )
        if after_invoke is not None:
            after_invoke(result, transcript_ref)
    except BaseException as exc:  # noqa: BLE001 - re-raised after safety
        checkpoint_error = exc

    pre_capture_error: GitCaptureError | None = None
    if before_capture is not None:
        try:
            before_capture()
        except BaseException as exc:  # noqa: BLE001 - preserve checkpoint
            if checkpoint_error is not None:
                if exc is checkpoint_error:
                    raise checkpoint_error
                raise checkpoint_error from exc
            if not isinstance(exc, Exception):
                raise
            pre_capture_error = GitCaptureError(
                f"pre-capture workspace safety check failed: {exc}"
            )
    if checkpoint_error is not None:
        raise checkpoint_error
    assert transcript_ref is not None

    # Capture the working-tree delta after invocation. Always persist —
    # partial diffs from a failed run are valuable for postmortem. When
    # the directive declares ``files_in_scope`` / ``creates_paths``,
    # those bound the diff capture so concurrent sibling slices don't
    # smear each other (the runtime contract that replaces the deleted
    # worktree isolation).
    capture_error = pre_capture_error
    try:
        if capture_error is not None:
            raise capture_error
        diff_bytes = _git_diff_bytes(
            cwd=cwd,
            path_scope=scope or None,
            exclude_pathspecs=corpus_exclusions,
        )
        files_modified = _git_modified_files(
            cwd=cwd,
            path_scope=scope or None,
            exclude_pathspecs=corpus_exclusions,
        )
    except GitCaptureError as exc:
        # Never let a separately successful status query turn an incomplete
        # or empty patch into a success-class Implementation. The transcript
        # and a typed capture diagnostic remain useful postmortem evidence,
        # while the wrapping workflow can roll the workspace back.
        capture_error = exc
        diff_bytes = b""
        files_modified = ()
        logger.error("slice %s Git capture failed: %s", slice_id, exc)

    # Scope-escape check: when the directive declared a scope, also
    # capture the unscoped working-tree state so we can flag files the
    # implementer wrote outside its declared lane. Ambient dirt (files
    # already modified before the implementer ran — uv.lock auto-touch,
    # editor swap files, sibling-slice writes from a parallel tier) is
    # subtracted via ``pre_dirty`` so false positives don't trip the
    # warning on every shared-cwd run.
    #
    # Edge case: a file dirty both before AND after the implementer's
    # run is treated as ambient (no escape flagged). That's a
    # conservative false-negative, preferable to reporting phantom escapes on
    # every busy workspace.
    if scope and capture_error is None:
        try:
            all_modified = set(_git_modified_files(
                cwd=cwd,
                path_scope=None,
                exclude_pathspecs=corpus_exclusions,
            ))
            in_scope = set(files_modified)
            scope_escapes = tuple(
                sorted(
                    path for path in all_modified
                    if path not in in_scope and path not in pre_dirty
                )
            )
        except GitCaptureError as exc:
            capture_error = exc
            diff_bytes = b""
            files_modified = ()
            scope_escapes = ()
            logger.error("slice %s Git capture failed: %s", slice_id, exc)
        else:
            if scope_escapes:
                logger.warning(
                    "slice %s wrote %d file(s) outside its declared scope: %s",
                    slice_id,
                    len(scope_escapes),
                    list(scope_escapes)[:5],
                )
    else:
        scope_escapes = ()

    # Keep the persisted patch/file list cumulative for shared-workspace tier
    # composition and retry-in-place.  Classification, however, must only
    # treat work as this invocation's partial result when the authorized
    # scoped state actually changed relative to its immediate baseline.
    invocation_changed_scope = (
        capture_error is None
        and (
            diff_bytes != pre_scope_diff_bytes
            or files_modified != pre_scope_files_modified
        )
    )
    classification_files_modified = (
        files_modified if invocation_changed_scope else ()
    )

    diff_ref = corpus.put_artifact(
        diff_bytes,
        kind="diffs",
        name_hint=f"{slice_id}.diff",
    )

    capture_artifacts = {}
    if capture_error is not None:
        capture_artifacts["capture_error"] = corpus.put_artifact(
            str(capture_error).encode("utf-8", errors="replace"),
            kind="logs",
            name_hint=f"{slice_id}_capture_error.log",
        )

    error_kind = result.error.kind if result.error is not None else None
    # Some adapters deliberately return a partial result at the turn ceiling
    # and therefore leave ``error`` unset. Preserve the causal signal for every
    # such result, including one with mutations: the captured diff is useful
    # evidence, but the incomplete call is not a successful implementation.
    if (
        error_kind is None
        and result.stop_reason in {"max_turns", "max-turns"}
    ):
        error_kind = "max_turns"
    classification = (
        "hard_fail_capture"
        if capture_error is not None
        else _classify(
            error_kind,
            classification_files_modified,
            directive.verify_mode,
            scope_escapes=scope_escapes,
        )
    )

    execution = _snapshot_execution(
        backend_name=backend.name,
        model=backend.model,
    )

    cost_usd = float(
        getattr(getattr(result, "usage", None), "cost_usd", 0.0) or 0.0
    )
    input_tokens = int(
        getattr(getattr(result, "usage", None), "input_tokens", 0) or 0
    )
    output_tokens = int(
        getattr(getattr(result, "usage", None), "output_tokens", 0) or 0
    )
    cache_creation_tokens = int(
        getattr(
            getattr(result, "usage", None),
            "cache_creation_tokens",
            0,
        )
        or 0
    )
    cache_read_tokens = int(
        getattr(
            getattr(result, "usage", None),
            "cache_read_tokens",
            0,
        )
        or 0
    )
    stop_reason = (
        result.stop_reason if result.stop_reason != "end_turn" else None
    )

    return Implementation(
        slice_id=slice_id,
        base_commit=base_commit,
        # We never commit here. The consumer is responsible for committing and
        # stamping a resulting SHA before publishing its final record.
        result_commit=None,
        diff_ref=diff_ref,
        transcript_ref=transcript_ref,
        files_modified=files_modified,
        classification=classification,
        execution=execution,
        scope_escapes=scope_escapes,
        artifacts=capture_artifacts,
        cost_usd=cost_usd,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_tokens=cache_creation_tokens,
        cache_read_tokens=cache_read_tokens,
        execution_completeness=result.completeness,
        pricing_evidence=result.pricing_evidence,
        cost_bounds=result.cost_bounds,
        stop_reason=stop_reason,
    )


__all__ = ["GitCaptureError", "capture_workspace_delta", "implement"]
