# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Tests for ``assay.primitives.smoke_check``.

Covers the four language dispatch arms (Rust / Python / C++ / none),
the timeout path, the no-files pass-through, the isolation of the Python
syntax floor from the checked tree, and the split between a rejection of the
code and a check that could not run (``SmokeResult.infrastructure_failure``).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Mapping
from importlib import import_module
from pathlib import Path
from types import MappingProxyType

import pytest

from assay import ImplementationDirective, ProjectContext, SmokeResult
from assay.primitives import smoke_check


_smoke_module = import_module("assay.primitives.smoke_check")
SMOKE_LOG_MAX_BYTES = _smoke_module.SMOKE_LOG_MAX_BYTES


def _private_cache(command) -> Path:
    """The ``-X pycache_prefix=<dir>`` operand of an isolated compile command."""
    prefix = "pycache_prefix="
    operands = [
        argument[len(prefix):]
        for option, argument in zip(command, command[1:])
        if option == "-X" and argument.startswith(prefix)
    ]
    assert len(operands) == 1, command
    return Path(operands[0])


def _directive(
    files_in_scope: tuple[str, ...] = (),
    creates_paths: tuple[str, ...] = (),
    verify_mode: str = "build_only",
) -> ImplementationDirective:
    return ImplementationDirective(
        slice_id="smoke_t",
        intent="t",
        files_in_scope=files_in_scope,
        creates_paths=creates_paths,
        verify_mode=verify_mode,
        blast_radius={},
    )


def test_no_files_passes_with_no_language(tmp_path: Path):
    result = smoke_check(_directive(), cwd=str(tmp_path))
    assert result.passed is True
    assert result.language is None
    assert result.log_excerpt == ""
    assert result.duration_ms == 0


def test_python_syntax_clean_passes(tmp_path: Path):
    (tmp_path / "hello.py").write_text("def hello():\n    return 42\n")
    result = smoke_check(
        _directive(files_in_scope=("hello.py",)),
        cwd=str(tmp_path),
    )
    assert result.passed is True
    assert result.language == "python"
    assert result.log_excerpt == ""


def test_python_smoke_uses_injected_process_boundary(tmp_path: Path) -> None:
    (tmp_path / "hello.py").write_text("value = 42\n")
    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []
    observed_cache: Path | None = None

    def supervised(command, **kwargs):
        nonlocal observed_cache
        calls.append((tuple(command), kwargs))
        if "compileall" in command:
            observed_cache = _private_cache(command)
            assert observed_cache.is_dir()
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    result = smoke_check(
        _directive(files_in_scope=("hello.py",)),
        cwd=str(tmp_path),
        timeout=17,
        process_runner=supervised,
    )

    assert result.passed is True
    assert result.infrastructure_failure is False
    assert observed_cache is not None and not observed_cache.exists()
    assert [command for command, _kwargs in calls] == [
        (
            sys.executable,
            "-I",
            "-S",
            "-X",
            f"pycache_prefix={observed_cache}",
            "-m",
            "compileall",
            "-q",
            "--",
            "hello.py",
        ),
        ("ruff", "--version"),
        (
            "ruff",
            "check",
            "--select",
            "F",
            "--no-fix",
            "--no-cache",
            "--",
            "hello.py",
        ),
    ]
    # Isolation is carried by the command line; the environment is inherited
    # and ignored by the isolated interpreter.
    assert calls[0][1] == {
        "cwd": str(tmp_path),
        "capture_output": True,
        "timeout": 17,
    }
    assert calls[1][1] == {
        "capture_output": True,
        "text": True,
        "timeout": 5,
    }
    assert calls[2][1] == {
        "cwd": str(tmp_path),
        "capture_output": True,
        "timeout": 17,
    }


def test_injected_runner_oserror_from_ruff_probe_propagates(tmp_path: Path) -> None:
    (tmp_path / "hello.py").write_text("value = 42\n")

    class SettlementProofError(OSError):
        pass

    def supervised(command, **_kwargs):
        if "compileall" in command:
            return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")
        raise SettlementProofError("descendant settlement proof missing")

    with pytest.raises(SettlementProofError, match="settlement proof missing"):
        smoke_check(
            _directive(files_in_scope=("hello.py",)),
            cwd=str(tmp_path),
            process_runner=supervised,
        )


def test_falsey_injected_smoke_runner_is_not_bypassed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FalseyRunner:
        def __init__(self) -> None:
            self.calls: list[tuple[str, ...]] = []

        def __bool__(self) -> bool:
            return False

        def __call__(self, command, **_kwargs):
            self.calls.append(tuple(command))
            return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    def forbidden_stdlib(*_args, **_kwargs):
        raise AssertionError("stdlib runner must not be selected")

    monkeypatch.setattr(_smoke_module.subprocess, "run", forbidden_stdlib)
    runner = FalseyRunner()
    result = smoke_check(
        _directive(files_in_scope=("src/lib.rs",)),
        cwd=str(tmp_path),
        process_runner=runner,
    )

    assert result.passed is True
    assert runner.calls == [("cargo", "check", "--workspace")]


def test_python_syntax_uses_private_cache_with_hostile_ambient_prefix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "hello.py").write_text("def hello():\n    return 42\n")
    monkeypatch.setenv("PYTHONPYCACHEPREFIX", os.devnull)
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    before = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}
    cache_dirs: list[Path] = []
    original_run = subprocess.run

    def _run(command, **kwargs):
        cache = _private_cache(command)
        assert cache.is_dir()
        assert cache != Path(os.devnull)
        cache_dirs.append(cache)
        return original_run(command, **kwargs)

    monkeypatch.setattr(_smoke_module.subprocess, "run", _run)
    monkeypatch.setattr(
        _smoke_module,
        "_ruff_available",
        lambda _process_runner, **_kwargs: False,
    )

    result = smoke_check(
        _directive(files_in_scope=("hello.py",)),
        cwd=str(tmp_path),
    )

    assert result.passed is True
    assert result.language == "python"
    assert result.log_excerpt == ""
    assert os.environ["PYTHONPYCACHEPREFIX"] == os.devnull
    assert os.environ["PYTHONDONTWRITEBYTECODE"] == "1"
    assert cache_dirs and all(not cache.exists() for cache in cache_dirs)
    assert {path.relative_to(tmp_path) for path in tmp_path.rglob("*")} == before


def test_python_timeout_removes_private_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "hello.py").write_text("def hello():\n    return 42\n")
    monkeypatch.setenv("PYTHONPYCACHEPREFIX", os.devnull)
    cache_dirs: list[Path] = []

    def _timeout(command, **kwargs):
        cache = _private_cache(command)
        assert cache.is_dir()
        cache_dirs.append(cache)
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(_smoke_module.subprocess, "run", _timeout)

    result = smoke_check(
        _directive(files_in_scope=("hello.py",)),
        cwd=str(tmp_path),
        timeout=3,
    )

    assert result.passed is False
    assert result.language == "python"
    assert result.infrastructure_failure is True
    assert result.log_excerpt == "python: compileall timeout after 3s"
    assert os.environ["PYTHONPYCACHEPREFIX"] == os.devnull
    assert cache_dirs and all(not cache.exists() for cache in cache_dirs)


def test_python_syntax_error_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "broken.py").write_text("def broken(:\n    return\n")
    monkeypatch.setenv("PYTHONPYCACHEPREFIX", os.devnull)
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    result = smoke_check(
        _directive(files_in_scope=("broken.py",)),
        cwd=str(tmp_path),
    )
    assert result.passed is False
    assert result.language == "python"
    assert result.infrastructure_failure is False
    assert "SyntaxError" in result.log_excerpt or "syntax" in result.log_excerpt.lower()
    assert not any(path.name == "__pycache__" for path in tmp_path.rglob("*"))


def test_python_missing_file_is_soft_pass(tmp_path: Path):
    """``compileall -q`` exits 0 on missing files (prints a warning to
    stderr but doesn't fail). The smoke check inherits this — a missing
    file shouldn't gate the build, because:

    (a) for files_in_scope: assay.implement already ran and either
        succeeded (file exists) or hard_fail_empty_diff'd; a missing
        file here means something pre-implement is wrong and the
        workflow will catch it via the diff capture.
    (b) for creates_paths: the implementer was supposed to create it;
        if it didn't, the diff is empty and _classify already
        flagged it.

    So the smoke gate's job is to catch SYNTAX errors, not missing
    files."""
    result = smoke_check(
        _directive(files_in_scope=("never_existed.py",)),
        cwd=str(tmp_path),
    )
    assert result.passed is True
    assert result.language == "python"


@pytest.mark.skipif(
    shutil.which("cargo") is None,
    reason="cargo not installed in this env",
)
def test_rust_check_runs_when_cargo_available(tmp_path: Path):
    # Initialize a minimal cargo project so ``cargo check`` has
    # something to verify. ``cargo init`` creates a Cargo.toml + src/.
    subprocess.run(
        ["cargo", "init", "--name", "smoke_t", "--lib", str(tmp_path)],
        capture_output=True, check=True,
    )
    result = smoke_check(
        _directive(files_in_scope=("src/lib.rs",)),
        cwd=str(tmp_path),
        timeout=60,
    )
    assert result.language == "rust"
    # ``cargo init`` writes a minimal valid lib — should pass.
    assert result.passed is True


def test_rust_default_checks_workspace_without_all_features(
    monkeypatch,
    tmp_path: Path,
):
    commands: list[list[str]] = []

    def _run(command, **_kwargs):
        commands.append(list(command))
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(_smoke_module.subprocess, "run", _run)
    result = smoke_check(
        _directive(files_in_scope=("src/lib.rs",)),
        cwd=str(tmp_path),
    )

    assert result.passed is True
    assert commands == [["cargo", "check", "--workspace"]]


def test_rust_smoke_forwards_injected_runner_arguments(tmp_path: Path) -> None:
    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    def supervised(command, **kwargs):
        calls.append((tuple(command), kwargs))
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    result = smoke_check(
        _directive(files_in_scope=("src/lib.rs",)),
        cwd=str(tmp_path),
        timeout=23,
        runner_parameters={
            "per_language": {
                "rust": {"crate": "selected", "cargo_features": ["f1", "f2"]},
            },
        },
        process_runner=supervised,
    )

    assert result.passed is True
    assert calls == [
        (
            ("cargo", "check", "-p", "selected", "--features", "f1,f2"),
            {
                "cwd": str(tmp_path),
                "capture_output": True,
                "timeout": 23,
            },
        ),
    ]


def test_rust_uses_frozen_spec_crate_and_features(monkeypatch, tmp_path: Path):
    commands: list[list[str]] = []

    def _run(command, **_kwargs):
        commands.append(list(command))
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    context = ProjectContext(
        project="fabric-usd",
        repo_url="ssh://example.invalid/fabric-usd.git",
        base_commit="a" * 40,
        branch="main",
        submodule_path="fabric-usd",
        runners={"build": "runner", "verify": "runner"},
        runner_parameters={
            "per_language": {
                "rust": {
                    "crate": "fabric-render-ovrtx",
                    "cargo_features": ["ovrtx", "trace"],
                    # Verify-only arguments must not leak into cargo check.
                    "cargo_test_args": ["--", "--test-threads=1"],
                },
            },
        },
    )
    assert isinstance(context.runner_parameters, Mapping)
    assert context.runner_parameters["per_language"]["rust"][
        "cargo_features"
    ] == ("ovrtx", "trace")

    monkeypatch.setattr(_smoke_module.subprocess, "run", _run)
    result = smoke_check(
        _directive(files_in_scope=("src/lib.rs",)),
        cwd=str(tmp_path),
        runner_parameters=context.runner_parameters,
    )

    assert result.passed is True
    assert commands == [[
        "cargo",
        "check",
        "-p",
        "fabric-render-ovrtx",
        "--features",
        "ovrtx,trace",
    ]]


def test_rust_uses_confined_spec_project_subdir(monkeypatch, tmp_path: Path):
    project = tmp_path / "native" / "runtime-rust"
    project.mkdir(parents=True)
    calls: list[tuple[list[str], str]] = []

    def _run(command, **kwargs):
        calls.append((list(command), kwargs["cwd"]))
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(_smoke_module.subprocess, "run", _run)
    result = smoke_check(
        _directive(creates_paths=("native/runtime-rust/src/lib.rs",)),
        cwd=str(tmp_path),
        runner_parameters={
            "per_language": {
                "rust": {"project_subdir": "native/runtime-rust"},
            },
        },
    )

    assert result.passed is True
    assert calls == [(["cargo", "check", "--workspace"], str(project))]


def test_rust_selects_project_subdir_for_current_slice(monkeypatch, tmp_path: Path):
    spike = tmp_path / "native" / "spikes" / "rust"
    runtime = tmp_path / "native" / "runtime-rust"
    spike.mkdir(parents=True)
    runtime.mkdir(parents=True)
    calls: list[tuple[list[str], str]] = []

    def _run(command, **kwargs):
        calls.append((list(command), kwargs["cwd"]))
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(_smoke_module.subprocess, "run", _run)
    result = smoke_check(
        _directive(creates_paths=("native/runtime-rust/src/lib.rs",)),
        cwd=str(tmp_path),
        runner_parameters={
            "per_language": {
                "rust": {"project_subdir": "native/spikes/rust"},
            },
            "per_slice": {
                "smoke_t": {
                    "per_language": {
                        "rust": {"project_subdir": "native/runtime-rust"},
                    },
                },
            },
        },
    )

    assert result.passed is True
    assert calls == [(["cargo", "check", "--workspace"], str(runtime))]


def test_rust_rejects_spec_project_subdir_escape(tmp_path: Path):
    result = smoke_check(
        _directive(creates_paths=("native/runtime-rust/src/lib.rs",)),
        cwd=str(tmp_path),
        runner_parameters={
            "per_language": {"rust": {"project_subdir": "../outside"}},
        },
    )

    assert result.passed is False
    assert result.infrastructure_failure is False
    assert result.language == "rust"
    assert "confined relative path" in result.log_excerpt


def test_rust_accepts_mapping_and_tuple_runner_parameters(
    monkeypatch,
    tmp_path: Path,
):
    commands: list[list[str]] = []

    def _run(command, **_kwargs):
        commands.append(list(command))
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    runner_parameters = MappingProxyType({
        "cargo_features": ("shared",),
        "per_language": MappingProxyType({
            "rust": MappingProxyType({"crate": "selected-crate"}),
        }),
    })
    monkeypatch.setattr(_smoke_module.subprocess, "run", _run)

    result = smoke_check(
        _directive(files_in_scope=("src/lib.rs",)),
        cwd=str(tmp_path),
        runner_parameters=runner_parameters,
    )

    assert result.passed is True
    assert commands == [[
        "cargo",
        "check",
        "-p",
        "selected-crate",
        "--features",
        "shared",
    ]]


@pytest.mark.parametrize(
    "runner_parameters",
    [
        [],
        {1: "not-a-string-key"},
        {"per_language": []},
        {"per_language": {"rust": []}},
        {"per_language": {1: {}}},
        {"per_language": {"rust": {1: "not-a-string-key"}}},
        {"per_language": {"rust": {"per_language": {}}}},
        {"per_language": {"python": {"per_language": {}}}},
        {"per_slice": []},
        {"per_slice": {1: {}}},
        {"per_slice": {"smoke_t": []}},
        {"per_slice": {"smoke_t": {"per_slice": {}}}},
        {"per_language": {"rust": {"crate": ""}}},
        {"per_language": {"rust": {"crate": 42}}},
        {"per_language": {"rust": {"cargo_features": "ovrtx"}}},
        {"per_language": {"rust": {"cargo_features": ("ovrtx", "")}}},
    ],
)
def test_rust_malformed_runner_parameters_fail_before_cargo(
    monkeypatch,
    tmp_path: Path,
    runner_parameters: object,
):
    def _unexpected_run(*_args, **_kwargs):
        raise AssertionError("cargo must not start for malformed parameters")

    monkeypatch.setattr(_smoke_module.subprocess, "run", _unexpected_run)
    result = smoke_check(
        _directive(files_in_scope=("src/lib.rs",)),
        cwd=str(tmp_path),
        runner_parameters=runner_parameters,  # type: ignore[arg-type]
    )

    assert result.passed is False
    # A rejected Spec selection is a configuration failure, not a check that
    # the environment prevented from running.
    assert result.infrastructure_failure is False
    assert result.language == "rust"
    assert "invalid runner parameters" in result.log_excerpt


def test_rust_when_cargo_missing(monkeypatch, tmp_path: Path):
    """Tests the FileNotFoundError branch independent of cargo's presence
    by faking the binary lookup."""
    # Override PATH so the subprocess can't find cargo.
    monkeypatch.setenv("PATH", "/nonexistent")
    result = smoke_check(
        _directive(files_in_scope=("src/lib.rs",)),
        cwd=str(tmp_path),
        timeout=5,
    )
    assert result.passed is False
    assert result.infrastructure_failure is True
    assert result.language == "rust"
    assert "cargo" in result.log_excerpt.lower()


def test_cpp_files_skip_with_pass(tmp_path: Path):
    """C++ has no cheap pre-build gate that's worth wiring; the smoke
    primitive deliberately passes-through with the language tag so
    operators see the routing decision."""
    (tmp_path / "foo.cpp").write_text("int main(){return 0;}\n")
    result = smoke_check(
        _directive(files_in_scope=("foo.cpp",)),
        cwd=str(tmp_path),
    )
    assert result.passed is True
    assert result.language == "cpp"
    assert "skipped" in result.log_excerpt


def test_mixed_rust_and_python_picks_rust(tmp_path: Path):
    """Rust is the first dispatch arm — a slice with both Rust and Python
    in scope routes to cargo check."""
    monkeypatch_path = tmp_path
    (monkeypatch_path / "a.py").write_text("def a():pass\n")
    # No real Cargo.toml — cargo check will fail. We just need to
    # verify the routing went to Rust, which is observable from the
    # language tag.
    result = smoke_check(
        _directive(files_in_scope=("src/lib.rs", "a.py")),
        cwd=str(tmp_path),
        timeout=10,
    )
    assert result.language == "rust"


def test_unknown_extension_passes_with_no_language(tmp_path: Path):
    result = smoke_check(
        _directive(files_in_scope=("data.yaml", "README.md")),
        cwd=str(tmp_path),
    )
    assert result.passed is True
    assert result.language is None


def test_creates_paths_participate_in_dispatch(tmp_path: Path):
    """Dispatch decision uses creates_paths even though the files don't
    exist yet — language tag confirms the routing. See the
    test_python_missing_file_is_soft_pass docstring for why we don't
    fail-hard on missing files here."""
    result = smoke_check(
        _directive(creates_paths=("new_module.py",)),
        cwd=str(tmp_path),
    )
    assert result.language == "python"
    assert result.passed is True


def test_smoke_result_is_frozen():
    result = SmokeResult(
        passed=True, language="python", log_excerpt="", duration_ms=5,
    )
    import dataclasses
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.passed = False  # type: ignore[misc]


def test_duration_ms_nonnegative(tmp_path: Path):
    (tmp_path / "hello.py").write_text("x = 1\n")
    result = smoke_check(
        _directive(files_in_scope=("hello.py",)),
        cwd=str(tmp_path),
    )
    assert result.duration_ms >= 0


# ----------------------------------------------------------------------
# SmokeResult.infrastructure_failure
# ----------------------------------------------------------------------


def test_smoke_result_infrastructure_failure_defaults_false() -> None:
    result = SmokeResult(
        passed=False, language="python", log_excerpt="x", duration_ms=1,
    )
    assert result.infrastructure_failure is False


def test_smoke_result_rejects_passed_infrastructure_failure() -> None:
    with pytest.raises(ValueError, match="infrastructure_failure"):
        SmokeResult(
            passed=True,
            language="python",
            log_excerpt="",
            duration_ms=1,
            infrastructure_failure=True,
        )


def _scripted_runner(script: dict[str, object]):
    """A ``subprocess.run``-compatible runner keyed by the tool it launches.

    ``script`` maps ``"compileall"``, ``"ruff --version"``, ``"ruff check"``
    and ``"cargo"`` to a ``(returncode, stdout, stderr)`` tuple or to an
    exception to raise.
    """

    calls: list[tuple[str, ...]] = []

    def run(command, **_kwargs):
        calls.append(tuple(command))
        if "compileall" in command:
            key = "compileall"
        elif command[0] == "ruff":
            key = "ruff --version" if command[1] == "--version" else "ruff check"
        else:
            key = command[0]
        outcome = script[key]
        if isinstance(outcome, BaseException):
            raise outcome
        returncode, stdout, stderr = outcome
        return subprocess.CompletedProcess(command, returncode, stdout, stderr)

    run.calls = calls  # type: ignore[attr-defined]
    return run


# ----------------------------------------------------------------------
# Python: the syntax floor runs in an isolated interpreter
# ----------------------------------------------------------------------


@pytest.mark.parametrize("module", ["compileall", "py_compile"])
def test_python_syntax_floor_ignores_workspace_stdlib_shadow(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    module: str,
) -> None:
    """A work-tree module named like the standard library never runs.

    With ``-m`` the working directory is otherwise first on ``sys.path``; the
    shadow would execute in the host interpreter and, by exiting 0, pass a
    file that does not compile.
    """

    tree = tmp_path / "tree"
    tree.mkdir()
    executed = tmp_path / "shadow-executed"
    (tree / f"{module}.py").write_text(
        "import pathlib\n"
        f"pathlib.Path({str(executed)!r}).write_text('ran')\n"
        "raise SystemExit(0)\n"
    )
    (tree / "broken.py").write_text("def broken(:\n    return\n")
    monkeypatch.setattr(
        _smoke_module,
        "_ruff_available",
        lambda _process_runner, **_kwargs: False,
    )
    before = {path.relative_to(tree) for path in tree.rglob("*")}

    result = smoke_check(
        _directive(files_in_scope=("broken.py",)),
        cwd=str(tree),
    )

    assert not executed.exists()
    assert result.passed is False
    assert result.infrastructure_failure is False
    assert result.language == "python"
    assert "SyntaxError" in result.log_excerpt
    assert {path.relative_to(tree) for path in tree.rglob("*")} == before


def test_python_syntax_floor_ignores_ambient_pythonpath_shadow(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tree = tmp_path / "tree"
    tree.mkdir()
    shadow_dir = tmp_path / "ambient"
    shadow_dir.mkdir()
    executed = tmp_path / "shadow-executed"
    (shadow_dir / "compileall.py").write_text(
        "import pathlib\n"
        f"pathlib.Path({str(executed)!r}).write_text('ran')\n"
        "raise SystemExit(0)\n"
    )
    (tree / "broken.py").write_text("def broken(:\n    return\n")
    monkeypatch.setenv("PYTHONPATH", str(shadow_dir))
    monkeypatch.setattr(
        _smoke_module,
        "_ruff_available",
        lambda _process_runner, **_kwargs: False,
    )

    result = smoke_check(
        _directive(files_in_scope=("broken.py",)),
        cwd=str(tree),
    )

    assert not executed.exists()
    assert result.passed is False
    assert result.infrastructure_failure is False
    assert "SyntaxError" in result.log_excerpt


def test_python_syntax_floor_ignores_site_customization_in_checked_tree(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A ``sitecustomize`` module of the checked tree never runs.

    An editable install puts a directory of the tree on the path of the
    interpreter that runs Assay through a ``.pth`` file. ``-I`` alone still
    imports ``site``, which processes that file and then imports
    ``sitecustomize`` from the tree; there the module can replace the compiler
    and pass a file that does not compile.
    """

    environment = tmp_path / "environment"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(environment)],
        capture_output=True, check=True,
    )
    interpreter = environment / "bin" / "python"
    (site_packages,) = (environment / "lib").glob("python*/site-packages")
    tree = tmp_path / "tree"
    package_root = tree / "src"
    package_root.mkdir(parents=True)
    executed = tmp_path / "sitecustomize-executed"
    (site_packages / "__editable__.tree.pth").write_text(f"{package_root}\n")
    (package_root / "sitecustomize.py").write_text(
        "import pathlib\n"
        "import py_compile\n"
        f"pathlib.Path({str(executed)!r}).write_text('ran')\n"
        "py_compile.compile = lambda *args, **kwargs: None\n"
    )
    (package_root / "broken.py").write_text("def broken(:\n    return\n")
    # Control: with ``site`` enabled this interpreter imports the module.
    subprocess.run(
        [str(interpreter), "-I", "-c", "pass"],
        capture_output=True, check=True,
    )
    if not executed.exists():
        pytest.skip("the interpreter ships a sitecustomize of its own")
    executed.unlink()
    monkeypatch.setattr(sys, "executable", str(interpreter))
    monkeypatch.setattr(
        _smoke_module,
        "_ruff_available",
        lambda _process_runner, **_kwargs: False,
    )

    result = smoke_check(
        _directive(files_in_scope=("src/broken.py",)),
        cwd=str(tree),
    )

    assert not executed.exists()
    assert result.passed is False
    assert result.infrastructure_failure is False
    assert result.language == "python"
    assert "SyntaxError" in result.log_excerpt


def test_python_syntax_floor_compiles_option_shaped_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A declared path that looks like an option is still a file to check.

    Without the ``--`` terminator ``compileall`` reads ``-x.py`` as
    ``-x .py``, skips every path that matches ``.py`` and exits 0, so the
    broken file next to it passes the syntax floor.
    """

    (tmp_path / "-x.py").write_text("value = 1\n")
    (tmp_path / "broken.py").write_text("def broken(:\n    return\n")
    monkeypatch.setattr(
        _smoke_module,
        "_ruff_available",
        lambda _process_runner, **_kwargs: False,
    )

    result = smoke_check(
        _directive(files_in_scope=("-x.py", "broken.py")),
        cwd=str(tmp_path),
    )

    assert result.passed is False
    assert result.infrastructure_failure is False
    assert "SyntaxError" in result.log_excerpt


def test_python_commands_pass_paths_after_option_terminator(
    tmp_path: Path,
) -> None:
    runner = _scripted_runner({
        "compileall": (0, b"", b""),
        "ruff --version": (0, "ruff 0.0.0\n", ""),
        "ruff check": (0, b"", b""),
    })

    result = smoke_check(
        _directive(files_in_scope=("-x.py", "pkg/mod.py", "README.md")),
        cwd=str(tmp_path),
        process_runner=runner,
    )

    assert result.passed is True
    compile_command, _probe, ruff_command = runner.calls
    assert compile_command[-4:] == ("-q", "--", "-x.py", "pkg/mod.py")
    assert ruff_command[-4:] == ("--no-cache", "--", "-x.py", "pkg/mod.py")


# ----------------------------------------------------------------------
# Python: a check that could not run is not a verdict on the code
# ----------------------------------------------------------------------


def test_python_interpreter_unavailable_is_infrastructure_failure(
    tmp_path: Path,
) -> None:
    runner = _scripted_runner({"compileall": FileNotFoundError(sys.executable)})

    result = smoke_check(
        _directive(files_in_scope=("hello.py",)),
        cwd=str(tmp_path),
        process_runner=runner,
    )

    assert result.passed is False
    assert result.infrastructure_failure is True
    assert result.language == "python"
    assert result.log_excerpt == (
        f"python: interpreter unavailable: {sys.executable}"
    )
    assert len(runner.calls) == 1


@pytest.mark.parametrize(
    ("returncode", "stderr"),
    [
        (2, b"[runner settlement error: background descendants outlived exit]\n"),
        (-9, b""),
        (137, b"Killed\n"),
    ],
    ids=["supervisor_forced_exit", "signal", "killed_via_shell"],
)
def test_python_compileall_abnormal_exit_is_infrastructure_failure(
    tmp_path: Path,
    returncode: int,
    stderr: bytes,
) -> None:
    runner = _scripted_runner({"compileall": (returncode, b"", stderr)})

    result = smoke_check(
        _directive(files_in_scope=("hello.py",)),
        cwd=str(tmp_path),
        process_runner=runner,
    )

    assert result.passed is False
    assert result.infrastructure_failure is True
    assert result.log_excerpt.startswith(
        f"python: compileall exited {returncode} without a syntax verdict"
    )
    if stderr.strip():
        assert stderr.decode().strip() in result.log_excerpt
    # The optional Ruff pass never runs after an unavailable syntax floor.
    assert len(runner.calls) == 1


def test_python_exit_one_without_compileall_diagnostic_is_infrastructure_failure(
    tmp_path: Path,
) -> None:
    """Exit 1 is also the interpreter's own status for a startup failure."""

    runner = _scripted_runner({
        "compileall": (
            1,
            b"",
            b"Fatal Python error: Failed to import encodings module\n",
        ),
    })

    result = smoke_check(
        _directive(files_in_scope=("hello.py",)),
        cwd=str(tmp_path),
        process_runner=runner,
    )

    assert result.passed is False
    assert result.infrastructure_failure is True
    assert result.log_excerpt.startswith(
        "python: compileall exited 1 without a syntax verdict"
    )
    assert "Failed to import encodings module" in result.log_excerpt


_REJECTED_PYTHON_SOURCES = {
    "syntax_error": b"def broken(:\n    return\n",
    "indentation_error": b"def f():\nreturn 1\n",
    "tab_error": b"def f():\n\tx = 1\n        y = 2\n",
    "null_byte": b"x = 1\x00\n",
    "undecodable_source": b"# -*- coding: utf-8 -*-\nx = '\xff'\n",
}


@pytest.mark.parametrize("case", sorted(_REJECTED_PYTHON_SOURCES))
def test_python_rejected_source_is_code_failure(tmp_path: Path, case: str) -> None:
    """Every shape in which compileall reports a source it rejected."""

    (tmp_path / "rejected.py").write_bytes(_REJECTED_PYTHON_SOURCES[case])

    result = smoke_check(
        _directive(files_in_scope=("rejected.py",)),
        cwd=str(tmp_path),
    )

    assert result.passed is False
    assert result.infrastructure_failure is False
    assert result.language == "python"
    assert "rejected.py" in result.log_excerpt


_needs_unprivileged_posix = pytest.mark.skipif(
    not hasattr(os, "geteuid") or os.geteuid() == 0,
    reason="needs POSIX permission semantics as a non-root user",
)


@_needs_unprivileged_posix
def test_python_unreadable_source_is_infrastructure_failure(
    tmp_path: Path,
) -> None:
    source = tmp_path / "secret.py"
    source.write_text("value = 1\n")
    source.chmod(0)
    try:
        result = smoke_check(
            _directive(files_in_scope=("secret.py",)),
            cwd=str(tmp_path),
        )
    finally:
        source.chmod(0o644)

    assert result.passed is False
    assert result.infrastructure_failure is True
    assert result.log_excerpt.startswith(
        "python: compileall could not read or cache 1 source file(s)"
    )
    assert "PermissionError" in result.log_excerpt


@_needs_unprivileged_posix
def test_python_rejected_source_outranks_unreadable_source(tmp_path: Path) -> None:
    unreadable = tmp_path / "secret.py"
    unreadable.write_text("value = 1\n")
    unreadable.chmod(0)
    (tmp_path / "broken.py").write_text("def broken(:\n    return\n")
    try:
        result = smoke_check(
            _directive(files_in_scope=("secret.py", "broken.py")),
            cwd=str(tmp_path),
        )
    finally:
        unreadable.chmod(0o644)

    assert result.passed is False
    assert result.infrastructure_failure is False
    assert "SyntaxError" in result.log_excerpt


# What ``compileall -q`` prints when the operating system refuses a bytecode
# write into the cache: the first is real output for a cache prefix that is
# not a directory, the second is the same report for a full device.
_COMPILEALL_CACHE_NOT_A_DIRECTORY = (
    b"*** Error compiling 'a.py'...\n"
    b"NotADirectoryError: [Errno 20] Not a directory: '/dev/null/app'\n"
)
_COMPILEALL_CACHE_DEVICE_FULL = (
    b"*** Error compiling 'b.py'...\n"
    b"OSError: [Errno 28] No space left on device: "
    b"'/tmp/assay-smoke-pycache-k3/app/b.cpython-312.pyc.140283'\n"
)
_COMPILEALL_REJECTED_SOURCE = (
    b"*** Error compiling 'broken.py'...\n"
    b'  File "broken.py", line 1\n'
    b"    def broken(:\n"
    b"               ^\n"
    b"SyntaxError: invalid syntax\n"
    b"\n"
)


@pytest.mark.parametrize(
    ("stdout", "failed_files"),
    [
        (_COMPILEALL_CACHE_NOT_A_DIRECTORY, 1),
        (_COMPILEALL_CACHE_DEVICE_FULL, 1),
        (_COMPILEALL_CACHE_NOT_A_DIRECTORY + _COMPILEALL_CACHE_DEVICE_FULL, 2),
    ],
    ids=["cache_not_a_directory", "device_full", "two_files"],
)
def test_python_cache_write_failure_is_infrastructure_failure(
    tmp_path: Path,
    stdout: bytes,
    failed_files: int,
) -> None:
    """An operating-system error while caching bytecode judges no source."""

    runner = _scripted_runner({"compileall": (1, stdout, b"")})

    result = smoke_check(
        _directive(files_in_scope=("a.py", "b.py")),
        cwd=str(tmp_path),
        process_runner=runner,
    )

    assert result.passed is False
    assert result.infrastructure_failure is True
    assert result.language == "python"
    assert result.log_excerpt.startswith(
        f"python: compileall could not read or cache {failed_files} "
        "source file(s)\n"
    )
    assert stdout.decode().strip() in result.log_excerpt
    # The optional Ruff pass never runs after an unavailable syntax floor.
    assert len(runner.calls) == 1


@pytest.mark.parametrize(
    "stdout",
    [
        _COMPILEALL_CACHE_DEVICE_FULL + _COMPILEALL_REJECTED_SOURCE,
        _COMPILEALL_REJECTED_SOURCE + _COMPILEALL_CACHE_NOT_A_DIRECTORY,
    ],
    ids=["os_error_first", "rejection_first"],
)
def test_python_rejected_source_outranks_cache_write_failure(
    tmp_path: Path,
    stdout: bytes,
) -> None:
    """One source the compiler rejected is a verdict on the code."""

    runner = _scripted_runner({"compileall": (1, stdout, b"")})

    result = smoke_check(
        _directive(files_in_scope=("a.py", "b.py", "broken.py")),
        cwd=str(tmp_path),
        process_runner=runner,
    )

    assert result.passed is False
    assert result.infrastructure_failure is False
    assert not result.log_excerpt.startswith("python: ")
    assert "SyntaxError: invalid syntax" in result.log_excerpt
    assert len(runner.calls) == 1


def test_python_ruff_findings_are_code_failure(tmp_path: Path) -> None:
    runner = _scripted_runner({
        "compileall": (0, b"", b""),
        "ruff --version": (0, "ruff 0.0.0\n", ""),
        "ruff check": (1, b"a.py:1:7: F821 Undefined name `nope`\n", b""),
    })

    result = smoke_check(
        _directive(files_in_scope=("a.py",)),
        cwd=str(tmp_path),
        process_runner=runner,
    )

    assert result.passed is False
    assert result.infrastructure_failure is False
    assert "F821" in result.log_excerpt


@pytest.mark.parametrize("returncode", [2, -9])
def test_python_ruff_abnormal_exit_is_infrastructure_failure(
    tmp_path: Path,
    returncode: int,
) -> None:
    """Ruff exit 2 is an invalid configuration, command line, or internal
    error and carries no lint verdict; its cause is on stderr."""

    runner = _scripted_runner({
        "compileall": (0, b"", b""),
        "ruff --version": (0, "ruff 0.0.0\n", ""),
        "ruff check": (returncode, b"", b"ruff failed\n  Cause: bad config\n"),
    })

    result = smoke_check(
        _directive(files_in_scope=("a.py",)),
        cwd=str(tmp_path),
        process_runner=runner,
    )

    assert result.passed is False
    assert result.infrastructure_failure is True
    assert result.log_excerpt.startswith(
        f"python: ruff exited {returncode} without a lint verdict"
    )
    assert "bad config" in result.log_excerpt


_RUFF_PROBE_OK = (0, "ruff 0.0.0\n", "")


@pytest.mark.parametrize(
    ("ruff", "ruff_check_started"),
    [
        ({"ruff --version": FileNotFoundError("ruff")}, False),
        ({"ruff --version": (1, "", "ruff: cannot start\n")}, False),
        (
            {"ruff --version": subprocess.TimeoutExpired(cmd="ruff", timeout=5)},
            False,
        ),
        (
            {
                "ruff --version": _RUFF_PROBE_OK,
                "ruff check": subprocess.TimeoutExpired(cmd="ruff", timeout=9),
            },
            True,
        ),
        (
            {
                "ruff --version": _RUFF_PROBE_OK,
                "ruff check": FileNotFoundError("ruff"),
            },
            True,
        ),
    ],
    ids=[
        "probe_missing",
        "probe_nonzero",
        "probe_timeout",
        "check_timeout",
        "check_missing",
    ],
)
def test_python_optional_ruff_pass_that_cannot_run_is_soft_pass(
    tmp_path: Path,
    ruff: dict[str, object],
    ruff_check_started: bool,
) -> None:
    """Ruff is optional: only a required tool yields a flagged failure."""

    runner = _scripted_runner({"compileall": (0, b"", b""), **ruff})

    result = smoke_check(
        _directive(files_in_scope=("a.py",)),
        cwd=str(tmp_path),
        timeout=9,
        process_runner=runner,
    )

    assert result.passed is True
    assert result.infrastructure_failure is False
    assert result.language == "python"
    assert result.log_excerpt == ""
    assert [command[:2] for command in runner.calls[1:]] == [
        ("ruff", "--version"),
        *([("ruff", "check")] if ruff_check_started else []),
    ]


def test_infrastructure_reason_survives_log_truncation(tmp_path: Path) -> None:
    runner = _scripted_runner({
        "compileall": (3, b"", b"x" * (SMOKE_LOG_MAX_BYTES * 4)),
    })

    result = smoke_check(
        _directive(files_in_scope=("a.py",)),
        cwd=str(tmp_path),
        process_runner=runner,
    )

    assert result.infrastructure_failure is True
    assert result.log_excerpt.startswith(
        "python: compileall exited 3 without a syntax verdict\n"
    )
    assert len(result.log_excerpt.encode("utf-8")) <= SMOKE_LOG_MAX_BYTES


@pytest.mark.parametrize("scope", ["hello.py", "src/lib.rs"])
def test_injected_runner_oserror_from_required_command_propagates(
    tmp_path: Path,
    scope: str,
) -> None:
    """A workflow runner's authority failure is never folded into a result."""

    class SettlementProofError(OSError):
        pass

    def supervised(command, **_kwargs):
        raise SettlementProofError("descendant settlement proof missing")

    with pytest.raises(SettlementProofError, match="settlement proof missing"):
        smoke_check(
            _directive(files_in_scope=(scope,)),
            cwd=str(tmp_path),
            process_runner=supervised,
        )


# ----------------------------------------------------------------------
# Rust: exit 101 is the code's failure only with a source diagnostic
# ----------------------------------------------------------------------


_RUSTC_TYPE_ERROR = b"""\
    Checking demo v0.1.0 (/app)
error[E0425]: cannot find value `x` in this scope
 --> src/lib.rs:2:5
  |
2 |     x
  |     ^ not found in this scope

For more information about this error, try `rustc --explain E0425`.
error: could not compile `demo` (lib) due to 1 previous error
"""

_RUSTC_KILLED = b"""\
warning: unused variable: `y`
 --> src/lib.rs:2:9
  |
2 |     let y = 1;
  |         ^ help: if this is intentional, prefix it with an underscore: `_y`
  |

error: could not compile `demo` (lib)

Caused by:
  process didn't exit successfully: `rustc --crate-name demo --edition=2021 src/lib.rs` (signal: 9, SIGKILL: kill)
"""

# Output of a failing ``cargo check`` (exit 101) in which the toolchain
# rejected source content.
_CARGO_CODE_FAILURES = {
    "type_error": _RUSTC_TYPE_ERROR,
    "syntax_error_without_code": b"""\
error: expected one of `!` or `::`, found `}`
 --> src/lib.rs:3:1
  |
2 |     broken
  |           - expected one of `!` or `::`
3 | }
  | ^ unexpected token

error: could not compile `demo` (lib) due to 1 previous error
""",
    "denied_lint": b"""\
error: unused variable: `x`
 --> src/lib.rs:3:9
  |
3 |     let x = 1;
  |         ^ help: if this is intentional, prefix it with an underscore: `_x`
  |
note: the lint level is defined here
 --> src/lib.rs:1:9

error: could not compile `demo` (lib) due to 1 previous error
""",
    "forced_colour": (
        b"\x1b[0m\x1b[1m\x1b[38;5;9merror[E0425]\x1b[0m\x1b[0m\x1b[1m: cannot "
        b"find value `x` in this scope\x1b[0m\n"
        b"\x1b[0m \x1b[0m\x1b[0m\x1b[1m\x1b[38;5;12m--> \x1b[0m\x1b[0m"
        b"src/lib.rs:2:5\x1b[0m\n"
    ),
    "crlf_output": (
        b"error[E0308]: mismatched types\r\n --> src\\lib.rs:2:5\r\n  |\r\n"
    ),
    "manifest_error_with_span": b"""\
error: invalid type: integer `1`, expected a string
 --> Cargo.toml:2:8
  |
2 | name = 1
  |        ^
  |
""",
    # Reconstructed, not captured: a Cargo release that reports a manifest
    # that is not TOML without a span, as a cause of the error that names the
    # manifest.
    "manifest_toml_parse_error": b"""\
error: failed to parse manifest at `/app/Cargo.toml`

Caused by:
  could not parse input as TOML

Caused by:
  TOML parse error at line 2, column 8
    |
  2 | name =
    |        ^
  invalid string
""",
    # The same report for a workspace member: the manifest is named by a
    # cause, in the chain cargo 1.95.0 prints for a member it cannot load.
    "workspace_member_manifest_toml_parse_error": b"""\
error: failed to load manifest for workspace member `/app/member`
referenced by workspace at `/app/Cargo.toml`

Caused by:
  failed to parse manifest at `/app/member/Cargo.toml`

Caused by:
  could not parse input as TOML

Caused by:
  TOML parse error at line 2, column 8
    |
  2 | name =
    |        ^
  invalid string
""",
    # A real compile error stays one when another compiler process was
    # killed in the same run.
    "real_error_and_a_killed_compiler": _RUSTC_TYPE_ERROR + _RUSTC_KILLED,
    # The remaining cases are verbatim cargo 1.95.0 output (package path
    # shortened). rustc prints the further lines of a multi-line message
    # between the ``error`` header and the span, indented to the message
    # column, whatever their first character is.
    "two_line_message": (
        b"    Checking demo v0.1.0 (C:\\app)\n"
        b"error: first line of a custom message\n"
        b"       second line of the message\n"
        b" --> src\\lib.rs:1:1\n"
        b"  |\n"
        b'1 | compile_error!("first line of a custom message\\nsecond line of the message");\n'
        b"  | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n"
        b"\n"
        b"error: could not compile `demo` (lib) due to 1 previous error\n"
    ),
    "two_line_message_forced_colour": (
        b"\x1b[1m\x1b[92m    Checking\x1b[0m demo v0.1.0 (C:\\app)\n"
        b"\x1b[1m\x1b[91merror\x1b[0m\x1b[1m\x1b[97m: first line of a custom message\x1b[0m\n"
        b"       \x1b[1m\x1b[97msecond line of the message\x1b[0m\n"
        b" \x1b[1m\x1b[96m--> \x1b[0msrc\\lib.rs:1:1\n"
        b"  \x1b[1m\x1b[96m|\x1b[0m\n"
        b"\x1b[1m\x1b[96m1\x1b[0m \x1b[1m\x1b[96m|\x1b[0m compile_error!"
        b'("first line of a custom message\\nsecond line of the message");\n'
        b"  \x1b[1m\x1b[96m|\x1b[0m \x1b[1m\x1b[91m^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
        b"^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\x1b[0m\n"
        b"\n"
        b"\x1b[1m\x1b[91merror\x1b[0m: could not compile `demo` (lib) due to 1 previous error\n"
    ),
    "denied_lint_two_line_note": (
        b"    Checking demo v0.1.0 (C:\\app)\n"
        b"error: use of deprecated function `f`: use g\n"
        b"       instead\n"
        b" --> src\\lib.rs:5:14\n"
        b"  |\n"
        b"5 | pub fn h() { f() }\n"
        b"  |              ^\n"
        b"  |\n"
        b"note: the lint level is defined here\n"
        b" --> src\\lib.rs:1:9\n"
        b"  |\n"
        b"1 | #![deny(deprecated)]\n"
        b"  |         ^^^^^^^^^^\n"
        b"\n"
        b"error: could not compile `demo` (lib) due to 1 previous error\n"
    ),
    "bullet_list_message": (
        b"    Checking demo v0.1.0 (C:\\app)\n"
        b"error: invalid configuration:\n"
        b"       - missing field `a`\n"
        b"       - missing field `b`\n"
        b" --> src\\lib.rs:1:1\n"
        b"  |\n"
        b'1 | compile_error!("invalid configuration:\\n- missing field `a`\\n- missing field `b`");\n'
        b"  | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n"
        b"\n"
        b"error: could not compile `demo` (lib) due to 1 previous error\n"
    ),
    "blank_line_in_message": (
        b"    Checking demo v0.1.0 (C:\\app)\n"
        b"error: first paragraph\n"
        b"       \n"
        b"       second paragraph\n"
        b" --> src\\lib.rs:1:1\n"
        b"  |\n"
        b'1 | compile_error!("first paragraph\\n\\nsecond paragraph");\n'
        b"  | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n"
        b"\n"
        b"error: could not compile `demo` (lib) due to 1 previous error\n"
    ),
    "gutter_shaped_message": (
        b"    Checking demo v0.1.0 (C:\\app)\n"
        b"error: table:\n"
        b"       | a | b |\n"
        b"       = total\n"
        b"       : end\n"
        b" --> src\\lib.rs:1:1\n"
        b"  |\n"
        b'1 | compile_error!("table:\\n| a | b |\\n= total\\n: end");\n'
        b"  | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n"
        b"\n"
        b"error: could not compile `demo` (lib) due to 1 previous error\n"
    ),
    # The lock file is the other project file whose TOML Cargo parses; a
    # current Cargo still reports it without a span.
    "lock_file_toml_parse_error": (
        b"error: failed to parse lock file at: C:\\app\\Cargo.lock\n"
        b"\n"
        b"Caused by:\n"
        b"  TOML parse error at line 2, column 10\n"
        b"    |\n"
        b"  2 | [[package\n"
        b"    |          ^\n"
        b"  unclosed array table, expected `]]`\n"
    ),
    # The remaining cases are verbatim cargo 1.85.0 output (paths shortened)
    # for a manifest that is not TOML. That release prints the further lines
    # of the message in the first column, directly below the ``error`` header
    # and before the span.
    "manifest_two_line_message": (
        b"error: invalid string\n"
        b"expected `\"`, `'`\n"
        b" --> Cargo.toml:2:7\n"
        b"  |\n"
        b"2 | name =\n"
        b"  |       ^\n"
        b"  |\n"
    ),
    "manifest_two_line_message_forced_colour": (
        b"\x1b[1m\x1b[91merror\x1b[0m: \x1b[1m\x1b[97minvalid string\n"
        b"expected `\"`, `'`\x1b[0m\n"
        b" \x1b[1m\x1b[96m-->\x1b[0m Cargo.toml:2:7\n"
        b"\x1b[1m\x1b[96m  |\x1b[0m\n"
        b"\x1b[1m\x1b[96m2 |\x1b[0m name =\n"
        b"\x1b[1m\x1b[96m  |\x1b[0m\x1b[1m\x1b[91m       ^\x1b[0m\n"
        b"\x1b[1m\x1b[96m  |\x1b[0m\n"
    ),
    "manifest_two_line_message_unclosed_table": (
        b"error: invalid table header\n"
        b"expected `.`, `]`\n"
        b" --> Cargo.toml:1:9\n"
        b"  |\n"
        b"1 | [package\n"
        b"  |         ^\n"
        b"  |\n"
    ),
    # A duplicated key whose name holds two line breaks: every further line
    # of the message is in the first column.
    "manifest_three_line_message": (
        b"error: duplicate key `first\n"
        b"second\n"
        b"third` in table `package.metadata`\n"
        b" --> Cargo.toml:7:1\n"
        b"  |\n"
        b"7 | \"first\\nsecond\\nthird\" = 2\n"
        b"  | ^\n"
        b"  |\n"
    ),
    # An array left open at the end of the file: the span is in the first
    # column as well.
    "manifest_message_before_first_column_span": (
        b"error: invalid array\n"
        b"expected `]`\n"
        b"--> Cargo.toml:1:1\n"
        b" |\n"
        b" |\n"
    ),
    "workspace_member_manifest_two_line_message": (
        b"error: invalid string\n"
        b"expected `\"`, `'`\n"
        b" --> member\\Cargo.toml:2:7\n"
        b"  |\n"
        b"2 | name =\n"
        b"  |       ^\n"
        b"  |\n"
        b"error: failed to load manifest for workspace member `C:\\app\\member`\n"
        b"referenced by workspace at `C:\\app\\Cargo.toml`\n"
    ),
    # Constructed: a directory name with a line break splits the header that
    # names the lock file. The further line belongs to that header, so the
    # parse error is still a cause of the error that named the file.
    "lock_file_header_with_two_lines_toml_parse_error": (
        b"error: failed to parse lock file at: /app/odd\n"
        b"name/Cargo.lock\n"
        b"\n"
        b"Caused by:\n"
        b"  TOML parse error at line 2, column 10\n"
        b"    |\n"
        b"  2 | [[package\n"
        b"    |          ^\n"
        b"  unclosed array table, expected `]]`\n"
    ),
}

# Verbatim cargo 1.95.0 output (paths shortened). The first names the manifest
# and rejects a value in it without a TOML parse error; the second is a Cargo
# configuration file in the user's Cargo home that is not TOML.
_CARGO_MANIFEST_EDITION_ERROR = (
    b"error: failed to parse manifest at `C:\\app\\Cargo.toml`\n"
    b"\n"
    b"Caused by:\n"
    b"  failed to parse the `edition` key\n"
    b"\n"
    b"Caused by:\n"
    b"  supported edition values are `2015`, `2018`, `2021`, or `2024`, but `2077` is unknown\n"
)
_CARGO_USER_CONFIGURATION_NOT_TOML = (
    b"error: could not load Cargo configuration\n"
    b"\n"
    b"Caused by:\n"
    b"  could not parse TOML configuration in `C:\\Users\\dev\\.cargo\\config.toml`\n"
    b"\n"
    b"Caused by:\n"
    b"  TOML parse error at line 1, column 5\n"
    b"    |\n"
    b"  1 | [net\n"
    b"    |     ^\n"
    b"  unclosed table, expected `]`\n"
)

# Output of a failing ``cargo check`` (exit 101) without an ``error``
# diagnostic that points at source: Cargo, the registry, the filesystem, or
# the compiler process failed, or the compiler reported an error without a
# span.
_CARGO_ENVIRONMENT_FAILURES = {
    "registry_unreachable": b"""\
    Updating crates.io index
warning: spurious network error (3 tries remaining): [6] Couldn't resolve host name (Could not resolve host: index.crates.io)
error: failed to get `serde` as a dependency of package `demo v0.1.0 (/app)`

Caused by:
  failed to query replaced source registry `crates-io`

Caused by:
  download of config.json failed

Caused by:
  [6] Couldn't resolve host name (Could not resolve host: index.crates.io)
""",
    "build_dir_lock_permission": b"""\
error: failed to open: /app/target/debug/.cargo-lock

Caused by:
  Permission denied (os error 13)
""",
    "metadata_write_permission": b"""\
error: failed to write /app/target/debug/deps/libdemo-1a2b3c.rmeta: Permission denied (os error 13)

error: could not compile `demo` (lib) due to 1 previous error
""",
    # The only span in the output belongs to a warning.
    "compiler_killed": _RUSTC_KILLED,
    "no_manifest": (
        b"error: could not find `Cargo.toml` in `/app` or any parent directory\n"
    ),
    "build_script_failed": b"""\
error: failed to run custom build command for `openssl-sys v0.9.99`

Caused by:
  process didn't exit successfully: `/app/target/debug/build/openssl-sys-1/build-script-main` (exit status: 101)
  --- stderr
  Could not find directory of OpenSSL installation
""",
    "standard_library_missing": b"""\
error[E0463]: can't find crate for `std`
  |
  = note: the `wasm32-unknown-unknown` target may not be installed

error: could not compile `demo` (lib) due to 1 previous error
""",
    # Verbatim cargo 1.95.0 output (package path shortened): both errors
    # carry no span, and the span after them belongs to a warning, which
    # opens its own diagnostic.
    "span_less_errors_before_a_spanned_warning": (
        b"    Checking demo v0.1.0 (C:\\app)\n"
        b"error: `#[panic_handler]` function required, but not found\n"
        b"\n"
        b"error: unwinding panics are not supported without std\n"
        b"  |\n"
        b'  = help: using nightly cargo, use -Zbuild-std with panic="abort" to avoid unwinding\n'
        b'  = note: since the core library is usually precompiled with panic="unwind", '
        b'rebuilding your crate with panic="abort" may not be enough to fix the problem\n'
        b"\n"
        b"warning: unused variable: `y`\n"
        b" --> src\\main.rs:3:18\n"
        b"  |\n"
        b"3 | pub fn f() { let y = 1; }\n"
        b"  |                  ^ help: if this is intentional, prefix it with an underscore: `_y`\n"
        b"  |\n"
        b"  = note: `#[warn(unused_variables)]` (part of `#[warn(unused)]`) on by default\n"
        b"\n"
        b'warning: `demo` (bin "demo") generated 1 warning\n'
        b'error: could not compile `demo` (bin "demo") due to 2 previous errors; 1 warning emitted\n'
    ),
    # Verbatim cargo 1.95.0 output (paths shortened) with a TOML parse error
    # that is not Cargo's report on the manifest or the lock file. A Cargo
    # configuration file is tool configuration and may lie outside the
    # checked tree: above the crate, or in the user's Cargo home.
    "cargo_configuration_above_crate_toml_parse_error": (
        b"error: could not load Cargo configuration\n"
        b"\n"
        b"Caused by:\n"
        b"  could not parse TOML configuration in `C:\\work\\.cargo\\config.toml`\n"
        b"\n"
        b"Caused by:\n"
        b"  TOML parse error at line 1, column 7\n"
        b"    |\n"
        b"  1 | [build\n"
        b"    |       ^\n"
        b"  unclosed table, expected `]`\n"
    ),
    "user_cargo_configuration_toml_parse_error": (
        _CARGO_USER_CONFIGURATION_NOT_TOML
    ),
    # Naming the manifest is not enough: this cause is no TOML parse error.
    "manifest_edition_error": _CARGO_MANIFEST_EDITION_ERROR,
    # The two reports in one output: the parse error is a cause of the
    # configuration error, not of the error that named the manifest.
    "manifest_error_before_configuration_toml_parse_error": (
        _CARGO_MANIFEST_EDITION_ERROR + _CARGO_USER_CONFIGURATION_NOT_TOML
    ),
    # Cargo relays a build script's stderr indented, so neither the phrase
    # nor a whole manifest report printed by the script is a cause of Cargo's.
    "build_script_prints_toml_parse_error": (
        b"   Compiling demo v0.1.0 (C:\\app)\n"
        b"error: failed to run custom build command for `demo v0.1.0 (C:\\app)`\n"
        b"\n"
        b"Caused by:\n"
        b"  process didn't exit successfully: `C:\\app\\target\\debug\\build\\demo-097e7dd9bc4e1099\\build-script-build` (exit code: 1)\n"
        b"  --- stderr\n"
        b"  TOML parse error at line 1, column 1\n"
    ),
    "build_script_relays_manifest_parse_chain": (
        b"   Compiling demo v0.1.0 (C:\\app)\n"
        b"error: failed to run custom build command for `demo v0.1.0 (C:\\app)`\n"
        b"\n"
        b"Caused by:\n"
        b"  process didn't exit successfully: `C:\\app\\target\\debug\\build\\demo-097e7dd9bc4e1099\\build-script-build` (exit code: 1)\n"
        b"  --- stderr\n"
        b"  error: failed to load manifest for workspace member `/app/member`\n"
        b"  referenced by workspace at `/app/Cargo.toml`\n"
        b"\n"
        b"  Caused by:\n"
        b"    failed to parse manifest at `/app/member/Cargo.toml`\n"
        b"\n"
        b"  Caused by:\n"
        b"    TOML parse error at line 1, column 1\n"
    ),
    # Verbatim cargo 1.85.0 output (paths shortened): one workspace member
    # fails without a span while the build script of another still compiles.
    # The ``warning`` headers directly below the last ``error`` header open
    # diagnostics of their own; they do not continue its message, so the span
    # is the warning's.
    "span_less_error_directly_before_a_spanned_warning": (
        b"   Compiling b v0.1.0 (C:\\app\\b)\n"
        b"    Checking a v0.1.0 (C:\\app\\a)\n"
        b"error: `#[panic_handler]` function required, but not found\n"
        b"\n"
        b"error: unwinding panics are not supported without std\n"
        b"  |\n"
        b"  = help: using nightly cargo, use -Zbuild-std with panic=\"abort\" to avoid unwinding\n"
        b"  = note: since the core library is usually precompiled with panic=\"unwind\", rebuilding your crate with panic=\"abort\" may not be enough to fix the problem\n"
        b"\n"
        b"error: could not compile `a` (bin \"a\") due to 2 previous errors\n"
        b"warning: build failed, waiting for other jobs to finish...\n"
        b"warning: unused variable: `y`\n"
        b"   --> b\\build.rs:402:9\n"
        b"    |\n"
        b"402 |     let y = 1;\n"
        b"    |         ^ help: if this is intentional, prefix it with an underscore: `_y`\n"
        b"    |\n"
        b"    = note: `#[warn(unused_variables)]` on by default\n"
        b"\n"
        b"warning: `b` (build script) generated 1 warning\n"
    ),
    # Constructed: a ``note`` or ``help`` header directly below a span-less
    # ``error`` header opens a diagnostic of its own in the same way.
    "span_less_error_directly_before_a_spanned_note": (
        b"error: could not compile `demo` (lib) due to 1 previous error\n"
        b"note: the lint level is defined here\n"
        b" --> src/lib.rs:1:9\n"
    ),
    "span_less_error_directly_before_a_spanned_help": (
        b"error: could not compile `demo` (lib) due to 1 previous error\n"
        b"help: consider importing this function\n"
        b" --> src/lib.rs:1:1\n"
    ),
    # Constructed: so does an ``error`` header directly below another one.
    # The parse error is a cause of the configuration error, and the manifest
    # that the first header named is forgotten.
    "manifest_header_directly_before_configuration_toml_parse_error": (
        b"error: failed to parse manifest at `C:\\app\\Cargo.toml`\n"
        + _CARGO_USER_CONFIGURATION_NOT_TOML
    ),
    # Verbatim cargo 1.85.0 output (paths shortened): the span is a line of
    # the build script's stderr, which Cargo relays below a cause. A cause
    # ends the ``error`` diagnostic it explains.
    "build_script_relays_a_span": (
        b"   Compiling demo v0.1.0 (C:\\app)\n"
        b"error: failed to run custom build command for `demo v0.1.0 (C:\\app)`\n"
        b"\n"
        b"Caused by:\n"
        b"  process didn't exit successfully: `C:\\app\\target\\debug\\build\\demo-ef2dc771a6a2a70c\\build-script-build` (exit code: 1)\n"
        b"  --- stderr\n"
        b"  error: forged by the build script\n"
        b"   --> src/lib.rs:1:1\n"
    ),
    # Constructed from the report above, without the blank line Cargo prints
    # before ``Caused by:``: a cause line directly below the header is not a
    # further line of its message either.
    "cause_directly_below_header_relays_a_span": (
        b"error: failed to run custom build command for `demo v0.1.0 (C:\\app)`\n"
        b"Caused by:\n"
        b"  process didn't exit successfully: `C:\\app\\target\\debug\\build\\demo-ef2dc771a6a2a70c\\build-script-build` (exit code: 1)\n"
        b"  --- stderr\n"
        b"   --> src/lib.rs:1:1\n"
    ),
    # Constructed: only first-column lines directly below the header continue
    # its message. After a blank or an indented line, a first-column line
    # ends the diagnostic as before.
    "first_column_text_after_a_blank_line_before_a_span": (
        b"error: failed to write /app/target/debug/deps/libdemo-1a2b3c.rmeta: Permission denied (os error 13)\n"
        b"\n"
        b"text in the first column\n"
        b" --> src/lib.rs:1:1\n"
    ),
    "first_column_text_after_an_indented_line_before_a_span": (
        b"error[E0463]: can't find crate for `std`\n"
        b"  |\n"
        b"  = note: the `wasm32-unknown-unknown` target may not be installed\n"
        b"text in the first column\n"
        b" --> src/lib.rs:1:1\n"
    ),
    "no_output": b"",
}


@pytest.mark.parametrize("case", sorted(_CARGO_CODE_FAILURES))
def test_rust_source_diagnostic_is_code_failure(tmp_path: Path, case: str) -> None:
    runner = _scripted_runner({"cargo": (101, b"", _CARGO_CODE_FAILURES[case])})

    result = smoke_check(
        _directive(files_in_scope=("src/lib.rs",)),
        cwd=str(tmp_path),
        process_runner=runner,
    )

    assert result.passed is False
    assert result.infrastructure_failure is False
    assert result.language == "rust"
    # The excerpt is the toolchain's own output, not a gate reason line.
    assert not result.log_excerpt.startswith("rust: ")
    assert result.log_excerpt.strip()


def test_rust_source_diagnostic_on_stdout_is_code_failure(tmp_path: Path) -> None:
    runner = _scripted_runner({"cargo": (101, _RUSTC_TYPE_ERROR, b"")})

    result = smoke_check(
        _directive(files_in_scope=("src/lib.rs",)),
        cwd=str(tmp_path),
        process_runner=runner,
    )

    assert result.passed is False
    assert result.infrastructure_failure is False
    assert "E0425" in result.log_excerpt
    assert "src/lib.rs:2:5" in result.log_excerpt


@pytest.mark.parametrize("case", sorted(_CARGO_ENVIRONMENT_FAILURES))
def test_rust_exit_101_without_source_diagnostic_is_infrastructure_failure(
    tmp_path: Path,
    case: str,
) -> None:
    output = _CARGO_ENVIRONMENT_FAILURES[case]
    runner = _scripted_runner({"cargo": (101, b"", output)})

    result = smoke_check(
        _directive(files_in_scope=("src/lib.rs",)),
        cwd=str(tmp_path),
        process_runner=runner,
    )

    assert result.passed is False
    assert result.infrastructure_failure is True
    assert result.language == "rust"
    assert result.log_excerpt.startswith(
        "rust: cargo check exited 101 without a compiler diagnostic"
    )
    if output.strip():
        # Cargo's own words are kept for the operator.
        assert output.strip().splitlines()[-1].decode() in result.log_excerpt


@pytest.mark.parametrize(
    ("returncode", "stderr"),
    [
        (137, b"Killed\n"),
        (-9, b""),
        (130, b""),
        (1, b"error: toolchain 'stable-x86_64-unknown-linux-gnu' is not installed\n"),
        (2, b"error: unexpected argument '--no-such-flag' found\n"),
        # Killed after printing a real diagnostic: the run did not finish, so
        # there is no verdict to attribute.
        (137, _RUSTC_TYPE_ERROR + b"Killed\n"),
    ],
    ids=[
        "sigkill_via_shell",
        "sigkill",
        "interrupted",
        "toolchain_proxy_failed",
        "usage_error",
        "killed_mid_run",
    ],
)
def test_rust_exit_other_than_build_failure_is_infrastructure_failure(
    tmp_path: Path,
    returncode: int,
    stderr: bytes,
) -> None:
    runner = _scripted_runner({"cargo": (returncode, b"", stderr)})

    result = smoke_check(
        _directive(files_in_scope=("src/lib.rs",)),
        cwd=str(tmp_path),
        process_runner=runner,
    )

    assert result.passed is False
    assert result.infrastructure_failure is True
    assert result.log_excerpt.startswith(
        f"rust: cargo check exited {returncode}, not its build-failure status 101"
    )


def test_rust_timeout_is_infrastructure_failure(tmp_path: Path) -> None:
    runner = _scripted_runner({
        "cargo": subprocess.TimeoutExpired(cmd="cargo", timeout=5),
    })

    result = smoke_check(
        _directive(files_in_scope=("src/lib.rs",)),
        cwd=str(tmp_path),
        timeout=5,
        process_runner=runner,
    )

    assert result.passed is False
    assert result.infrastructure_failure is True
    assert result.language == "rust"
    assert result.log_excerpt == "rust: smoke timeout after 5s"


def test_rust_failure_classification_keeps_spec_bound_command(
    tmp_path: Path,
) -> None:
    runner = _scripted_runner({"cargo": (101, b"", _RUSTC_TYPE_ERROR)})

    result = smoke_check(
        _directive(files_in_scope=("src/lib.rs",)),
        cwd=str(tmp_path),
        runner_parameters={
            "per_language": {
                "rust": {"crate": "selected", "cargo_features": ["f1", "f2"]},
            },
        },
        process_runner=runner,
    )

    assert result.infrastructure_failure is False
    assert runner.calls == [
        ("cargo", "check", "-p", "selected", "--features", "f1,f2"),
    ]


@pytest.mark.skipif(
    shutil.which("cargo") is None,
    reason="cargo not installed in this env",
)
def test_real_cargo_separates_compile_error_from_missing_manifest(
    tmp_path: Path,
) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    subprocess.run(
        ["cargo", "init", "--name", "smoke_t", "--lib", str(project)],
        capture_output=True, check=True,
    )
    (project / "src" / "lib.rs").write_text("pub fn f() -> u32 { missing }\n")
    broken = smoke_check(
        _directive(files_in_scope=("src/lib.rs",)),
        cwd=str(project),
        timeout=120,
    )
    assert broken.passed is False
    assert broken.infrastructure_failure is False, broken.log_excerpt
    assert "lib.rs:1:" in broken.log_excerpt

    empty = tmp_path / "empty"
    empty.mkdir()
    no_manifest = smoke_check(
        _directive(files_in_scope=("src/lib.rs",)),
        cwd=str(empty),
        timeout=120,
    )
    assert no_manifest.passed is False
    assert no_manifest.infrastructure_failure is True, no_manifest.log_excerpt
