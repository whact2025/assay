# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Redaction utilities for blobs that may leak into LLM prompts or logs.

The ``Verdict.artifacts`` / ``Implementation.artifacts`` fields thread
execution-side diagnostics (stderr, build warnings, tool-call output)
into subsequent prompts. Those blobs frequently carry secrets that
may need redaction before reaching a model endpoint or retained log:

- Known API-token shapes for several providers.
- Exact values of environment variables whose names look credential-bearing.
- Local filesystem paths under ``$HOME``.
- Common ANSI CSI sequences and BEL bytes.

``redact_secrets`` is an explicit policy tool for workflows rendering captured
bytes into prompts or logs. Corpus and the primitives do not call it
implicitly: callers decide whether to preserve exact forensic bytes or redact
them at a trust boundary.

Design:

- Pure function: bytes in → bytes out. No filesystem I/O, no logging
  side effects, no env access at call time (the deny-list is captured
  at import time via ``os.environ`` snapshot — see ``_ENV_VALUES``).
- Best-effort for bytes input: pattern replacement does not decode the blob,
  so arbitrary binary data is accepted. ``None`` becomes ``b""``; other
  non-bytes inputs raise ``TypeError``.
- Conservative rather than comprehensive: token patterns are prefix-specific,
  while environment values are selected by a credential-like name heuristic.
  This is not a general data-loss-prevention scanner.

The helper is re-exported from ``assay`` for
``from assay import redact_secrets``.
"""

from __future__ import annotations

import os
import re

__all__ = ["redact_secrets", "REDACTED"]


REDACTED = b"<REDACTED>"


# ANSI CSI/SGR sequences — strip them entirely. Keep BEL (\x07) and
# other terminal control bytes out of the LLM context.
_ANSI = re.compile(rb"\x1b\[[0-9;?]*[A-Za-z]")
_BEL = re.compile(rb"\x07")

# Token-shape regexes. Anchored at byte level so we don't need to
# decode/encode for the common path. Each is conservative — only the
# canonical published prefixes for these providers.
_TOKEN_PATTERNS: tuple[re.Pattern[bytes], ...] = (
    re.compile(rb"sk-ant-[A-Za-z0-9_\-]{20,}"),            # Anthropic
    re.compile(rb"sk-proj-[A-Za-z0-9_\-]{20,}"),           # OpenAI (project)
    re.compile(rb"sk-[A-Za-z0-9]{20,}"),                   # OpenAI classic key
    re.compile(rb"gh[opsu]_[A-Za-z0-9]{30,}"),             # GitHub PAT/OAuth
    re.compile(rb"xox[bopas]-[A-Za-z0-9\-]{10,}"),         # Slack
    re.compile(rb"AKIA[0-9A-Z]{16}"),                      # AWS access key
    re.compile(rb"AIza[0-9A-Za-z_\-]{35}"),                # Google API key
    re.compile(rb"glpat-[A-Za-z0-9_\-]{20,}"),             # GitLab PAT
)

# Snapshot the env at import time so the redaction set is stable across
# calls. We only strip values for env names that look like they hold
# secrets — listing every env var as a deny pattern would over-redact.
_SECRET_ENV_NAME_PATTERN = re.compile(
    r"(KEY|TOKEN|SECRET|PASSWORD|PASSWD|PWD|CREDENTIAL)",
    re.IGNORECASE,
)


def _snapshot_env_values() -> tuple[bytes, ...]:
    out: list[bytes] = []
    for name, value in os.environ.items():
        if not value or len(value) < 8:
            # Short values are noise — substring-match would over-redact
            # unrelated text. Real tokens are always ≥ 8 chars.
            continue
        if _SECRET_ENV_NAME_PATTERN.search(name):
            out.append(value.encode("utf-8", errors="replace"))
    # Sort longest first so that nested values redact correctly
    # (e.g. an OAuth token plus a refresh token sharing a prefix).
    out.sort(key=len, reverse=True)
    return tuple(out)


_ENV_VALUES: tuple[bytes, ...] = _snapshot_env_values()


def _home_path_pattern() -> re.Pattern[bytes] | None:
    home = os.environ.get("HOME") or os.path.expanduser("~")
    if not home or home in ("/", "~"):
        return None
    # Replace ``/home/operator/...`` with ``$HOME/...`` and bare ``/home/operator``.
    escaped = re.escape(home.encode("utf-8", errors="replace"))
    return re.compile(escaped + rb"(/[^\s'\"]*)?")


_HOME_PATTERN: re.Pattern[bytes] | None = _home_path_pattern()


def redact_secrets(blob: bytes | bytearray | None) -> bytes:
    """Strip common ANSI control bytes and redact configured secret patterns.

    Empty input passes through unchanged and ``None`` is treated as empty.
    Other non-bytes inputs raise ``TypeError``; callers should encode text
    before crossing this boundary.
    """
    if blob is None:
        return b""
    if not isinstance(blob, (bytes, bytearray)):
        raise TypeError(
            f"redact_secrets expected bytes, got {type(blob).__name__}"
        )
    if not blob:
        return bytes(blob)

    out = bytes(blob)
    out = _ANSI.sub(b"", out)
    out = _BEL.sub(b"", out)

    for pat in _TOKEN_PATTERNS:
        out = pat.sub(REDACTED, out)

    for env_value in _ENV_VALUES:
        if env_value and env_value in out:
            out = out.replace(env_value, REDACTED)

    if _HOME_PATTERN is not None:
        out = _HOME_PATTERN.sub(b"$HOME\\1", out)

    return out
