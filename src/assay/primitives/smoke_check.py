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
- Any ``.py`` files (and no Rust) → ``python -I -S -m compileall -q`` for
  syntax check, plus ``ruff --select F`` (pyflakes-style undefined-name
  + unused-import detection) when ``ruff`` is on PATH. ruff missing is
  not a failure — compileall covers the syntax floor. The interpreter runs
  isolated and without ``site``, so no module of the checked tree, no
  ``.pth`` file or ``sitecustomize`` module of the interpreter's environment,
  and no ``PYTHON*`` variable can replace or redirect the standard-library
  check, and both commands take the paths after ``--``, so a path that looks
  like an option stays a file.
- Any ``.cpp``, ``.cc``, ``.cxx``, ``.h``, ``.hpp``, or ``.cu`` files (and no
  Rust/Python) → skip-with-pass. No generic cheap parse step is configured; the
  consumer's build owns this lane. Other C-family suffixes are unrecognized.
  ``SmokeResult.language="cpp"`` so operators see the routing decision.
- No recognised files → pass with ``language=None``.

Required Rust and Python syntax commands are timeout-bounded (default 120 s).
Their timeout returns ``passed=False`` with a language-tagged excerpt. Ruff is
optional: absence or timeout after compileall succeeds is a soft pass.

A failed result separates a verdict on the code from a check that could not
run. ``SmokeResult.infrastructure_failure`` is True when a required tool is
missing, a required command times out, or a tool ends without one of its
verdict statuses: ``compileall`` and Ruff report findings with exit 1, and
``cargo check`` reports a build failure with exit 101. Exit 101 is also
Cargo's status for a registry, build-directory lock, permission, or
killed-compiler failure, so the Rust lane attributes it to the code only when
the output carries an ``error`` diagnostic with a ``-->`` source span, or a
TOML parse error that Cargo reports as a cause of its failure to parse the
manifest or the lock file. A TOML parse error in a Cargo configuration file,
which may lie outside the checked tree, or in a build script's output is not
such a report. ``compileall`` exit 1 is attributed to the code only when it
reports a source it rejected, not when every reported file failed with an
operating-system error. The flag says that the check gave no verdict; it does
not say that the environment is at fault or that a retry would succeed.

Pure: no corpus interaction, no LLM, no env mutation. The workflow
that calls smoke_check owns persisting the result (e.g. via
``corpus.put_artifact(result.log_excerpt.encode(), kind="logs")`` and
attaching the ref onto the slice's ``Implementation.artifacts``).
"""

from __future__ import annotations

import re
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

# ``compileall`` and Ruff exit 1 when they report findings. Any other nonzero
# status is the tool ending without a verdict.
_FINDINGS_EXIT = 1

# ``compileall -q`` opens every failed path with this marker. The next line is
# either the compiler's rejection of the source or, when the file could not be
# read or its bytecode could not be cached, an errno-bearing ``OSError``.
_COMPILEALL_ERROR_MARKER = b"*** Error compiling "
_COMPILEALL_OS_ERROR_RE = re.compile(rb"[A-Za-z_][A-Za-z0-9_]*: \[Errno \d+\] ")

# Cargo's exit status when the requested build failed. Signals, a toolchain
# proxy without a toolchain, and rejected command lines use other statuses.
_CARGO_BUILD_FAILURE_EXIT = 101

# Terminal styling that ``CARGO_TERM_COLOR=always`` wraps around diagnostics.
_ANSI_ESCAPE_RE = re.compile(rb"\x1b\[[0-9;]*[A-Za-z]")

# Evidence that the toolchain rejected source content: an ``error`` diagnostic
# that carries a ``--> <file>:<line>:<column>`` span (every rustc syntax,
# resolution, type, borrow, and denied-lint error, and Cargo's spanned
# manifest errors). A diagnostic is its header, which starts in the first
# column, the further lines of a multi-line message, and the indented or
# blank lines after them. rustc indents the further message lines. Cargo
# 1.85.0 prints them in the first column for a manifest that is not TOML, so
# a first-column line directly below an ``error`` header, or below another
# such line, continues the message unless it is a diagnostic header itself or
# ``Caused by:``. Every other line that starts in the first column, except
# the span itself, opens another diagnostic or a Cargo status block, so a
# span that belongs to a warning, note, or help does not count. A message
# line shaped like a diagnostic header ends the diagnostic as well; that
# failure is flagged.
_RUST_ERROR_HEADER_RE = re.compile(rb"error(?:\[E\d+\])?: ")
_RUST_LEVEL_HEADER_RE = re.compile(
    rb"(?:error|warning|note|help)(?:\[E\d+\])?: "
)
_RUST_SPAN_MARKER = b"--> "

# The other evidence is a span-less TOML parse error in one of the two project
# files whose TOML Cargo parses: the lock file, and the manifest when a Cargo
# release reports it as a cause instead of a spanned diagnostic. Cargo prints
# an error as a header and a chain of causes, each a first-column
# ``Caused by:`` line followed by the indented cause. The parse error counts
# only as such a cause, after the header or an earlier cause of the same chain
# named the manifest or the lock file, and only in the wording
# ``TOML parse error at line N``: a release that words it differently and
# gives no span is flagged. Cargo prints the same parse error for every
# configuration file it loads, including ones outside the checked tree, and
# relays a build script's stderr indented below a cause; neither is a
# rejection of the code. Whether the named file lies inside the checked tree
# is not examined, as for a span.
_CARGO_CAUSE_HEADER = b"Caused by:"
_CARGO_PROJECT_FILE_PARSE_ERROR = rb"failed to parse (?:manifest at|lock file at:) "
_CARGO_PROJECT_FILE_HEADER_RE = re.compile(
    rb"error: " + _CARGO_PROJECT_FILE_PARSE_ERROR
)
_CARGO_PROJECT_FILE_CAUSE_RE = re.compile(_CARGO_PROJECT_FILE_PARSE_ERROR)
_CARGO_TOML_PARSE_ERROR_RE = re.compile(rb"TOML parse error at line \d+")


def _tail_bytes(blob: bytes, n: int) -> str:
    return blob[-n:].decode("utf-8", errors="replace")


def _combined_output(stdout: bytes | None, stderr: bytes | None) -> bytes:
    out = stdout or b""
    err = stderr or b""
    if not out.strip():
        return err
    if not err.strip():
        return out
    return out + b"\n" + err


def _infrastructure_failure(
    language: str,
    started: float,
    reason: str,
    output: bytes = b"",
) -> SmokeResult:
    """Encode a required check that produced no verdict about the code.

    The ``<language>: <reason>`` line always survives truncation; the tool
    output is tail-bounded into the remaining excerpt budget. The excerpt is
    bounded in tool-output bytes: the tail is cut before it is decoded, so a
    multi-byte character split by the cut, or output that is not UTF-8, is
    decoded to U+FFFD and the encoded excerpt can then be longer than
    ``SMOKE_LOG_MAX_BYTES``.
    """

    headline = f"{language}: {reason}"
    budget = SMOKE_LOG_MAX_BYTES - len(headline.encode("utf-8")) - 1
    tail = _tail_bytes(output, budget).strip() if budget > 0 else ""
    return SmokeResult(
        passed=False,
        language=language,
        log_excerpt=f"{headline}\n{tail}" if tail else headline,
        duration_ms=int((time.monotonic() - started) * 1000),
        infrastructure_failure=True,
    )


def _count_compileall_failures(stdout: bytes) -> tuple[int, int]:
    """Return ``(rejected sources, operating-system errors)`` for one run."""

    rejected = 0
    os_errors = 0
    lines = stdout.splitlines()
    for index, line in enumerate(lines):
        if not line.startswith(_COMPILEALL_ERROR_MARKER):
            continue
        detail = lines[index + 1] if index + 1 < len(lines) else b""
        if _COMPILEALL_OS_ERROR_RE.match(detail):
            os_errors += 1
        else:
            rejected += 1
    return rejected, os_errors


def _cargo_rejected_source(output: bytes) -> bool:
    """Return whether Cargo's output attributes the failure to source."""

    text = _ANSI_ESCAPE_RE.sub(b"", output)
    in_error_diagnostic = False
    message_continues = False
    names_project_file = False
    cause_follows = False
    for line in text.splitlines():
        body = line.lstrip(b" \t")
        is_cause = cause_follows
        cause_follows = False
        is_below_message = message_continues
        message_continues = False
        if body.startswith(_RUST_SPAN_MARKER):
            if in_error_diagnostic:
                return True
        elif body and len(body) == len(line):
            if line.rstrip() == _CARGO_CAUSE_HEADER:
                # A cause ends the diagnostic it explains; the chain of
                # causes continues with the next line.
                in_error_diagnostic = False
                cause_follows = True
            elif (
                is_below_message
                and _RUST_LEVEL_HEADER_RE.match(line) is None
            ):
                # A further first-column line of the ``error`` message: the
                # diagnostic stays open, and so does what its header named.
                message_continues = True
            else:
                # Any other line that starts in the first column ends the
                # diagnostic before it and decides whether an ``error`` one
                # begins.
                in_error_diagnostic = (
                    _RUST_ERROR_HEADER_RE.match(line) is not None
                )
                message_continues = in_error_diagnostic
                names_project_file = (
                    _CARGO_PROJECT_FILE_HEADER_RE.match(line) is not None
                )
        elif is_cause:
            if _CARGO_PROJECT_FILE_CAUSE_RE.match(body):
                names_project_file = True
            elif names_project_file and _CARGO_TOML_PARSE_ERROR_RE.match(body):
                return True
    return False


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
    excerpt. ``SmokeResult.infrastructure_failure`` marks the failures that
    are not a verdict on the code: a missing tool, a timeout, or an exit
    without the tool's verdict status. Ruff is an optional second pass, so its
    absence or timeout is not a failure. Invalid Python argument types and
    other operating-system errors may still raise.
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
        return _infrastructure_failure("rust", started, "cargo not on PATH")
    except subprocess.TimeoutExpired:
        return _infrastructure_failure(
            "rust", started, f"smoke timeout after {timeout}s",
        )
    if r.returncode == 0:
        return SmokeResult(
            passed=True, language="rust", log_excerpt="",
            duration_ms=int((time.monotonic() - started) * 1000),
        )
    output = _combined_output(r.stdout, r.stderr)
    if r.returncode != _CARGO_BUILD_FAILURE_EXIT:
        return _infrastructure_failure(
            "rust",
            started,
            f"cargo check exited {r.returncode}, not its build-failure "
            f"status {_CARGO_BUILD_FAILURE_EXIT}",
            output,
        )
    if not _cargo_rejected_source(output):
        return _infrastructure_failure(
            "rust",
            started,
            f"cargo check exited {_CARGO_BUILD_FAILURE_EXIT} without a "
            "compiler diagnostic that points at source",
            output,
        )
    return SmokeResult(
        passed=False, language="rust",
        log_excerpt=_tail_bytes(output, SMOKE_LOG_MAX_BYTES),
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
    # temporary cache.
    with tempfile.TemporaryDirectory(prefix="assay-smoke-pycache-") as pycache:
        try:
            r = process_runner(
                # Use the interpreter executing Assay.  ``python`` is not a
                # portable PATH entry (many production images expose only
                # ``python3``), and choosing a different interpreter would make
                # the smoke gate disagree with the running toolchain.
                #
                # ``-I`` keeps the checked tree off ``sys.path``: with ``-m``
                # the working directory is otherwise searched first, so a
                # ``compileall.py`` or ``py_compile.py`` in the tree would run
                # in place of the standard library and decide the result.  It
                # also ignores every ``PYTHON*`` variable, so the private
                # cache is passed as an ``-X`` option and a hostile or sentinel
                # ambient PYTHONPYCACHEPREFIX such as /dev/null cannot turn a
                # valid source file into an infrastructure failure.  ``-P``
                # would not do: it requires Python 3.11.
                #
                # ``-I`` does not imply ``-S``.  ``site`` would still process
                # the ``.pth`` files of this interpreter's environment and
                # import ``sitecustomize``; an editable install of the checked
                # tree puts one of its directories on that path, where such a
                # module could replace the compiler.  ``compileall`` needs the
                # standard library only, so ``site`` stays out.
                #
                # ``--`` ends the options: ``compileall`` would otherwise read
                # a declared path such as ``-x.py`` as ``-x .py``, skip every
                # file, and exit 0.
                [
                    sys.executable,
                    "-I",
                    "-S",
                    "-X",
                    f"pycache_prefix={pycache}",
                    "-m",
                    "compileall",
                    "-q",
                    "--",
                    *py_files,
                ],
                cwd=cwd,
                capture_output=True,
                timeout=timeout,
            )
        except FileNotFoundError:
            return _infrastructure_failure(
                "python", started, f"interpreter unavailable: {sys.executable}",
            )
        except subprocess.TimeoutExpired:
            return _infrastructure_failure(
                "python", started, f"compileall timeout after {timeout}s",
            )
    if r.returncode not in (0, _FINDINGS_EXIT):
        return _infrastructure_failure(
            "python",
            started,
            f"compileall exited {r.returncode} without a syntax verdict",
            _combined_output(r.stdout, r.stderr),
        )
    if r.returncode == _FINDINGS_EXIT:
        rejected, os_errors = _count_compileall_failures(r.stdout or b"")
        if not rejected:
            return _infrastructure_failure(
                "python",
                started,
                (
                    f"compileall could not read or cache {os_errors} source "
                    "file(s)"
                    if os_errors
                    else "compileall exited 1 without a syntax verdict"
                ),
                _combined_output(r.stdout, r.stderr),
            )
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
                # As above: an option-shaped path is a file, not a Ruff flag.
                "--",
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
    if r.returncode == 0:
        return SmokeResult(
            passed=True, language="python", log_excerpt="",
            duration_ms=int((time.monotonic() - started) * 1000),
        )
    if r.returncode != _FINDINGS_EXIT:
        # Ruff exits 2 for an invalid configuration, a rejected command line,
        # or an internal error and writes the cause to stderr.
        return _infrastructure_failure(
            "python",
            started,
            f"ruff exited {r.returncode} without a lint verdict",
            _combined_output(r.stdout, r.stderr),
        )
    return SmokeResult(
        passed=False, language="python",
        log_excerpt=_tail_bytes(r.stdout or b"", SMOKE_LOG_MAX_BYTES),
        duration_ms=int((time.monotonic() - started) * 1000),
    )


__all__ = ["smoke_check", "SmokeResult", "SMOKE_LOG_MAX_BYTES"]
