# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Tests for ``assay.primitives.smoke_check``.

Covers the four language dispatch arms (Rust / Python / C++ / none),
the timeout path, the no-files pass-through, and the contract that
the function never raises (failures become typed SmokeResult fields).
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
            observed_cache = Path(kwargs["env"]["PYTHONPYCACHEPREFIX"])
            assert observed_cache.is_dir()
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    result = smoke_check(
        _directive(files_in_scope=("hello.py",)),
        cwd=str(tmp_path),
        timeout=17,
        process_runner=supervised,
    )

    assert result.passed is True
    assert [command for command, _kwargs in calls] == [
        (sys.executable, "-m", "compileall", "-q", "hello.py"),
        ("ruff", "--version"),
        (
            "ruff",
            "check",
            "--select",
            "F",
            "--no-fix",
            "--no-cache",
            "hello.py",
        ),
    ]
    compile_kwargs = calls[0][1]
    assert compile_kwargs["cwd"] == str(tmp_path)
    assert compile_kwargs["capture_output"] is True
    assert compile_kwargs["timeout"] == 17
    assert isinstance(compile_kwargs["env"], dict)
    assert observed_cache is not None and not observed_cache.exists()
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
        cache = Path(kwargs["env"]["PYTHONPYCACHEPREFIX"])
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
        cache = Path(kwargs["env"]["PYTHONPYCACHEPREFIX"])
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
