# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Executable checks for repository-local documentation contracts."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from urllib.parse import unquote


_ROOT = Path(__file__).resolve().parents[1]
_MARKDOWN_FILES = (_ROOT / "README.md", *sorted((_ROOT / "docs").rglob("*.md")))
_FENCE_RE = re.compile(
    r"^```(?P<language>[^\n`]*)\n(?P<body>.*?)^```\s*$",
    re.MULTILINE | re.DOTALL,
)
_LINK_RE = re.compile(r"(?<!!)\[[^\]]*\]\((?P<target>[^)]+)\)")


def _line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def test_python_fences_parse() -> None:
    """Every Python-labelled example must be valid standalone syntax."""

    for path in _MARKDOWN_FILES:
        text = path.read_text(encoding="utf-8")
        for match in _FENCE_RE.finditer(text):
            if match.group("language").strip().lower() not in {"py", "python"}:
                continue
            line = _line_number(text, match.start())
            ast.parse(match.group("body"), filename=f"{path}:{line}")


def test_local_markdown_links_resolve() -> None:
    """Inline links to repository files must name an existing target."""

    missing: list[str] = []
    for path in _MARKDOWN_FILES:
        text = path.read_text(encoding="utf-8")
        for match in _LINK_RE.finditer(text):
            target = match.group("target").strip()
            if target.startswith("<") and ">" in target:
                target = target[1 : target.index(">")]
            else:
                # Optional Markdown titles begin after a quoted whitespace
                # separator. Repository paths currently contain no spaces.
                target = target.split(' "', 1)[0].split(" '", 1)[0]
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            local = unquote(target.split("#", 1)[0])
            if not local:
                continue
            resolved = (path.parent / local).resolve()
            if not resolved.exists():
                line = _line_number(text, match.start())
                missing.append(f"{path.relative_to(_ROOT)}:{line}: {target}")

    assert not missing, "broken repository-local Markdown links:\n" + "\n".join(
        missing
    )
