# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Identity primitives for assay.

Defines:
    ContentHash: a ``typing.NewType`` alias over ``str`` carrying the
        canonical format ``'sha256:<lowercase-hex>'``. Because
        ``typing.NewType`` produces a callable rather than a runtime
        class, ``isinstance(x, ContentHash)`` is *not* supported.
        ``ContentHash(value)`` performs no runtime validation. Corpus
        boundaries require exactly ``sha256:`` plus 64 lowercase hexadecimal
        digits; advanced callers can use
        ``assay.corpus.codec.validate_content_hash`` for the same check.
    ProjectContext: frozen dataclass describing the project under
        analysis (repo URL, base commit, optional branch / submodule
        path, runner mapping).

Leaf module: stdlib only.
"""

from __future__ import annotations

import math
import typing
from collections.abc import Mapping
from dataclasses import dataclass, field

from .frozen import freeze_dict

ContentHash = typing.NewType("ContentHash", str)


def _validate_runner_parameters(value: object, *, path: str) -> None:
    """Require a deterministic JSON-shaped runner configuration."""

    if value is None or type(value) in {bool, int, str}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings, got {key!r}")
            _validate_runner_parameters(item, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_runner_parameters(item, path=f"{path}[{index}]")
        return
    raise TypeError(
        f"{path} must contain only JSON-compatible values, got "
        f"{type(value).__name__}"
    )


@dataclass(frozen=True)
class ProjectContext:
    project: str
    repo_url: str
    base_commit: str
    branch: str | None
    submodule_path: str | None
    runners: dict[str, str]
    runner_parameters: dict[str, object] = field(
        default_factory=dict,
        metadata={
            # Empty parameters are omitted canonically; the strict decoder may
            # supply only this declared default when the field is absent.
            "canonical_omit_empty": True,
            "persist_optional": True,
        },
    )

    def __post_init__(self) -> None:
        if not isinstance(self.runner_parameters, Mapping):
            raise TypeError("runner_parameters must be a mapping")
        _validate_runner_parameters(
            self.runner_parameters,
            path="runner_parameters",
        )
        object.__setattr__(self, "runners", freeze_dict(self.runners))
        object.__setattr__(
            self,
            "runner_parameters",
            freeze_dict(self.runner_parameters),
        )
