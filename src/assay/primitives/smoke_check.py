# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Pre-build cheap-gate primitive — language-aware syntax / parse check.

Runs before the workflow's expensive build/verify subprocess so a
slice with a syntactically broken diff fails fast (~seconds) instead
of burning a full ``cargo build`` (~minutes) or full ``cargo test
--workspace`` cycle.

Subprocess dispatch by file extension on
``directive.files_in_scope ∪ directive.creates_paths``. Suffix matching is
case-sensitive:

- Any ``.rs`` files → ``cargo check``. By default this checks the whole
  workspace. When the caller supplies the canonical Spec's generic
  ``runner_parameters`` envelope, the Rust smoke command uses only its
  projected crate and feature selection. A ``per_slice.<slice_id>`` overlay
  is selected before the language projection so one Spec can own multiple
  nested Cargo projects without rewriting its runner binding between tiers.
- Any ``.py`` files (and no Rust) → ``python -m compileall -q`` for
  syntax check, plus ``ruff --select F`` (pyflakes-style undefined-name
  + unused-import detection) when ``ruff`` is on PATH. ruff missing is
  not a failure — compileall covers the syntax floor.
- Any ``.cpp``, ``.cc``, ``.cxx``, ``.h``, ``.hpp``, or ``.cu`` files (and no
  Rust/Python) → skip-with-pass. No generic cheap parse step is configured; the
  consumer's build owns this lane. Other C-family suffixes are unrecognized.
  ``SmokeResult.language="cpp"`` so operators see the routing decision.
- No recognised files → pass with ``language=None``.

Required Rust and Python syntax commands are timeout-bounded (default 120 s).
Their timeout returns ``passed=False`` with a language-tagged excerpt. Ruff is
optional: absence or timeout after compileall succeeds is a soft pass.

Pure: no corpus interaction, no LLM, no env mutation. The workflow
that calls smoke_check owns persisting the result (e.g. via
``corpus.put_artifact(result.log_excerpt.encode(), kind="logs")`` and
attaching the ref onto the slice's ``Implementation.artifacts``).
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from ..types.dag import ImplementationDirective
from ..types.smoke import SmokeResult


SMOKE_LOG_MAX_BYTES = 4096


def _tail_bytes(blob: bytes, n: int) -> str:
    return blob[-n:].decode("utf-8", errors="replace")


def _has_ext(paths: tuple[str, ...], exts: tuple[str, ...]) -> bool:
    return any(p.endswith(exts) for p in paths)


def _ruff_available(
    process_runner: Callable[..., subprocess.CompletedProcess[Any]],
    *,
    suppress_os_errors: bool,
) -> bool:
    try:
        r = process_runner(
            ["ruff", "--version"],
            capture_output=True, text=True, timeout=5,
        )
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    except OSError:
        # The standalone stdlib boundary historically treated an optional
        # Ruff probe's OS-level launch failure like an absent executable.  A
        # workflow-supplied runner can use OSError subclasses for a failed
        # settlement or kernel-boundary attestation, however; never downgrade
        # those authority failures into a successful smoke result.
        if suppress_os_errors:
            return False
        raise


def smoke_check(
    directive: ImplementationDirective,
    *,
    cwd: str,
    timeout: int = 120,
    runner_parameters: Mapping[str, object] | None = None,
    process_runner: Callable[..., subprocess.CompletedProcess[Any]] | None = None,
) -> SmokeResult:
    """Cheap pre-build syntax/parse gate. Pure subprocess; no LLM.

    Args:
      directive: Slice directive whose ``files_in_scope`` +
        ``creates_paths`` drive the language-dispatch decision.
      cwd: Directory to run the smoke subprocess in. Must be the
        target repo's working tree.
      timeout: Per-subprocess timeout in seconds (default 120).
      runner_parameters: Optional generic, Spec-bound runner configuration.
        Rust projects the shared fields plus ``per_language.rust`` and uses
        only ``crate`` and ``cargo_features`` for the cheap check. Frozen
        mappings and tuple-valued frozen lists are accepted.
      process_runner: Optional ``subprocess.run``-compatible execution
        boundary supplied by the owning workflow. The default preserves
        Assay's standalone behavior; journal-owning workflows can inject a
        descendant-supervised implementation without duplicating smoke logic.

    Returns:
      ``SmokeResult`` — see ``assay.types.smoke`` for field semantics.

    Failures of the required command (missing tool, timeout, or nonzero exit)
    are encoded into ``SmokeResult.passed=False`` with a tail-truncated
    excerpt. Ruff is an optional second pass, so its absence or timeout is not
    a failure. Invalid Python argument types and other operating-system errors
    may still raise.
    """
    files = tuple(directive.files_in_scope) + tuple(directive.creates_paths)
    if not files:
        return SmokeResult(
            passed=True, language=None, log_excerpt="", duration_ms=0,
        )

    has_rust = _has_ext(files, (".rs",))
    has_python = _has_ext(files, (".py",))
    has_cpp = _has_ext(files, (".cpp", ".cc", ".cxx", ".h", ".hpp", ".cu"))

    started = time.monotonic()
    run_process = (
        process_runner if process_runner is not None else subprocess.run
    )

    if has_rust:
        return _smoke_rust(
            cwd,
            timeout,
            started,
            runner_parameters=runner_parameters,
            process_runner=run_process,
            slice_id=directive.slice_id,
        )
    if has_python:
        py_files = tuple(f for f in files if f.endswith(".py"))
        return _smoke_python(
            py_files,
            cwd,
            timeout,
            started,
            process_runner=run_process,
            suppress_optional_runner_os_errors=process_runner is None,
        )
    if has_cpp:
        # No cheap parse step that's worth wiring across kit/cmake/etc.
        # Pass-through; the consumer's build phase is the real gate.
        return SmokeResult(
            passed=True,
            language="cpp",
            log_excerpt="cpp: smoke check skipped (no cheap pre-build gate)",
            duration_ms=int((time.monotonic() - started) * 1000),
        )

    return SmokeResult(
        passed=True, language=None, log_excerpt="", duration_ms=0,
    )


def _merge_parameter_maps(
    base: Mapping[str, object],
    overlay: Mapping[str, object],
) -> dict[str, object]:
    """Recursively overlay canonical runner selections."""

    merged = dict(base)
    for key, value in overlay.items():
        prior = merged.get(key)
        if isinstance(prior, Mapping) and isinstance(value, Mapping):
            merged[key] = _merge_parameter_maps(prior, value)
        else:
            merged[key] = value
    return merged


def _select_slice_runner_parameters(
    runner_parameters: Mapping[str, object] | None,
    slice_id: str,
) -> Mapping[str, object] | None:
    """Select a confined canonical per-slice runner overlay."""

    if runner_parameters is None:
        return None
    if not isinstance(runner_parameters, Mapping):
        raise ValueError("runner_parameters must be a mapping")
    if not all(isinstance(key, str) for key in runner_parameters):
        raise ValueError("runner_parameters keys must be strings")

    selected = {
        key: value
        for key, value in runner_parameters.items()
        if key != "per_slice"
    }
    per_slice = runner_parameters.get("per_slice")
    if per_slice is None:
        return selected
    if not isinstance(per_slice, Mapping):
        raise ValueError("runner_parameters.per_slice must be a mapping")
    for configured_slice_id, overlay in per_slice.items():
        if not isinstance(configured_slice_id, str) or not isinstance(
            overlay, Mapping
        ):
            raise ValueError(
                "runner_parameters.per_slice requires string keys and "
                "mapping values"
            )
        if "per_slice" in overlay:
            raise ValueError(
                f"runner_parameters.per_slice.{configured_slice_id} may not "
                "nest another per_slice envelope"
            )
    overlay = per_slice.get(slice_id)
    if isinstance(overlay, Mapping):
        selected = _merge_parameter_maps(selected, overlay)
    return selected


def _project_rust_runner_parameters(
    runner_parameters: Mapping[str, object] | None,
) -> tuple[str | None, tuple[str, ...], str | None]:
    """Validate and project the generic envelope for Rust smoke selection."""

    if runner_parameters is None:
        projected: dict[str, object] = {}
    elif not isinstance(runner_parameters, Mapping):
        raise ValueError("runner_parameters must be a mapping")
    else:
        if not all(isinstance(key, str) for key in runner_parameters):
            raise ValueError("runner_parameters keys must be strings")
        projected = {
            key: value
            for key, value in runner_parameters.items()
            if key not in {"languages", "per_language"}
        }
        per_language = runner_parameters.get("per_language")
        if per_language is not None:
            if not isinstance(per_language, Mapping):
                raise ValueError("runner_parameters.per_language must be a mapping")
            for language, overlay in per_language.items():
                if not isinstance(language, str) or not isinstance(overlay, Mapping):
                    raise ValueError(
                        "runner_parameters.per_language requires string keys "
                        "and mapping values"
                    )
                if not all(isinstance(key, str) for key in overlay):
                    raise ValueError(
                        f"runner_parameters.per_language.{language} keys must "
                        "be strings"
                    )
                if "per_language" in overlay:
                    raise ValueError(
                        f"runner_parameters.per_language.{language} may not "
                        "nest another per_language envelope"
                    )
            rust_overlay = per_language.get("rust")
            if rust_overlay is not None:
                projected.update(rust_overlay)

    crate = projected.get("crate")
    if crate is not None and (not isinstance(crate, str) or not crate):
        raise ValueError("Rust runner parameter 'crate' must be a non-empty string")

    raw_features = projected.get("cargo_features")
    if raw_features is None:
        features: tuple[str, ...] = ()
    elif not isinstance(raw_features, (list, tuple)) or not all(
        isinstance(feature, str) and feature for feature in raw_features
    ):
        raise ValueError(
            "Rust runner parameter 'cargo_features' must be a string list"
        )
    else:
        features = tuple(raw_features)

    project_subdir = projected.get("project_subdir")
    if project_subdir is not None and (
        not isinstance(project_subdir, str) or not project_subdir.strip()
    ):
        raise ValueError(
            "Rust runner parameter 'project_subdir' must be a non-empty string"
        )
    return crate, features, project_subdir


def _rust_smoke_working_dir(cwd: str, project_subdir: str | None) -> str:
    """Resolve the Spec-selected Cargo project beneath the workflow root."""

    root = Path(cwd).resolve()
    if project_subdir is None:
        return str(root)
    subdir = Path(project_subdir)
    if subdir.is_absolute() or ".." in subdir.parts:
        raise ValueError(
            "Rust runner parameter 'project_subdir' must be a confined "
            "relative path"
        )
    candidate = (root / subdir).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            "Rust runner parameter 'project_subdir' escapes the workflow root"
        ) from exc
    if not candidate.is_dir():
        raise ValueError(
            "Rust runner parameter 'project_subdir' does not resolve to a "
            f"directory: {project_subdir!r}"
        )
    return str(candidate)


def _smoke_rust(
    cwd: str,
    timeout: int,
    started: float,
    *,
    runner_parameters: Mapping[str, object] | None,
    process_runner: Callable[..., subprocess.CompletedProcess[Any]],
    slice_id: str,
) -> SmokeResult:
    try:
        crate, features, project_subdir = _project_rust_runner_parameters(
            _select_slice_runner_parameters(runner_parameters, slice_id)
        )
        cargo_cwd = _rust_smoke_working_dir(cwd, project_subdir)
    except (TypeError, ValueError) as exc:
        return SmokeResult(
            passed=False,
            language="rust",
            log_excerpt=f"rust: invalid runner parameters: {exc}",
            duration_ms=int((time.monotonic() - started) * 1000),
        )

    cmd = ["cargo", "check"]
    if crate is None:
        cmd.append("--workspace")
    else:
        cmd.extend(("-p", crate))
    if features:
        cmd.extend(("--features", ",".join(features)))

    try:
        r = process_runner(
            cmd,
            cwd=cargo_cwd,
            capture_output=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        return SmokeResult(
            passed=False, language="rust",
            log_excerpt="rust: cargo not on PATH",
            duration_ms=int((time.monotonic() - started) * 1000),
        )
    except subprocess.TimeoutExpired:
        return SmokeResult(
            passed=False, language="rust",
            log_excerpt=f"rust: smoke timeout after {timeout}s",
            duration_ms=int((time.monotonic() - started) * 1000),
        )
    if r.returncode == 0:
        return SmokeResult(
            passed=True, language="rust", log_excerpt="",
            duration_ms=int((time.monotonic() - started) * 1000),
        )
    return SmokeResult(
        passed=False, language="rust",
        log_excerpt=_tail_bytes(r.stderr or b"", SMOKE_LOG_MAX_BYTES),
        duration_ms=int((time.monotonic() - started) * 1000),
    )


def _smoke_python(
    py_files: tuple[str, ...],
    cwd: str,
    timeout: int,
    started: float,
    *,
    process_runner: Callable[..., subprocess.CompletedProcess[Any]],
    suppress_optional_runner_os_errors: bool,
) -> SmokeResult:
    # First: compileall for the syntax-error floor.  It writes bytecode even
    # when PYTHONDONTWRITEBYTECODE is set, so isolate those writes in a private
    # temporary cache.  Overriding (without mutating) the inherited environment
    # also keeps a hostile or sentinel PYTHONPYCACHEPREFIX such as /dev/null
    # from turning a valid source file into an infrastructure failure.
    with tempfile.TemporaryDirectory(prefix="assay-smoke-pycache-") as pycache:
        environment = dict(os.environ)
        environment["PYTHONPYCACHEPREFIX"] = pycache
        try:
            r = process_runner(
                # Use the interpreter executing Assay.  ``python`` is not a
                # portable PATH entry (many production images expose only
                # ``python3``), and choosing a different interpreter would make
                # the smoke gate disagree with the running toolchain.
                [sys.executable, "-m", "compileall", "-q", *py_files],
                cwd=cwd,
                capture_output=True,
                timeout=timeout,
                env=environment,
            )
        except FileNotFoundError:
            return SmokeResult(
                passed=False, language="python",
                log_excerpt=f"python: interpreter unavailable: {sys.executable}",
                duration_ms=int((time.monotonic() - started) * 1000),
            )
        except subprocess.TimeoutExpired:
            return SmokeResult(
                passed=False, language="python",
                log_excerpt=f"python: compileall timeout after {timeout}s",
                duration_ms=int((time.monotonic() - started) * 1000),
            )
    if r.returncode != 0:
        # compileall writes the error to stdout for "Compiling …" lines
        # and to stderr for SyntaxError; concatenate so the operator
        # sees what failed without having to inspect both streams.
        tail = (r.stdout or b"") + b"\n" + (r.stderr or b"")
        return SmokeResult(
            passed=False, language="python",
            log_excerpt=_tail_bytes(tail, SMOKE_LOG_MAX_BYTES),
            duration_ms=int((time.monotonic() - started) * 1000),
        )

    # Second pass: ruff F-rules if available. Missing ruff is not a
    # failure — compileall covered the syntax floor; ruff is the
    # nice-to-have undefined-name gate.
    if not _ruff_available(
        process_runner,
        suppress_os_errors=suppress_optional_runner_os_errors,
    ):
        return SmokeResult(
            passed=True, language="python", log_excerpt="",
            duration_ms=int((time.monotonic() - started) * 1000),
        )
    try:
        r = process_runner(
            [
                "ruff",
                "check",
                "--select",
                "F",
                "--no-fix",
                "--no-cache",
                *py_files,
            ],
            cwd=cwd,
            capture_output=True,
            timeout=timeout,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        # Treat as soft-pass — ruff failures should not block when the
        # syntax floor already passed.
        return SmokeResult(
            passed=True, language="python", log_excerpt="",
            duration_ms=int((time.monotonic() - started) * 1000),
        )
    if r.returncode != 0:
        return SmokeResult(
            passed=False, language="python",
            log_excerpt=_tail_bytes(r.stdout or b"", SMOKE_LOG_MAX_BYTES),
            duration_ms=int((time.monotonic() - started) * 1000),
        )
    return SmokeResult(
        passed=True, language="python", log_excerpt="",
        duration_ms=int((time.monotonic() - started) * 1000),
    )


__all__ = ["smoke_check", "SmokeResult", "SMOKE_LOG_MAX_BYTES"]
