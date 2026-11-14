# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Tests for ``assay._safety.redact_secrets``.

Covers the four redaction passes (ANSI strip, token patterns, env-value
denylist, $HOME path replacement) and the boundary cases the module
docstring promises: empty input, non-bytes input, and binary blobs.
"""

from __future__ import annotations

import importlib

import pytest


def test_empty_input_passes_through():
    from assay._safety import redact_secrets

    assert redact_secrets(b"") == b""


def test_none_input_returns_empty():
    from assay._safety import redact_secrets

    assert redact_secrets(None) == b""


def test_non_bytes_raises_typeerror():
    from assay._safety import redact_secrets

    with pytest.raises(TypeError):
        redact_secrets("not bytes")  # type: ignore[arg-type]


def test_ansi_csi_sequences_stripped():
    from assay._safety import redact_secrets

    # ANSI red-bold-then-reset around the word "ERROR"
    raw = b"\x1b[31;1mERROR\x1b[0m: something broke"
    assert redact_secrets(raw) == b"ERROR: something broke"


def test_bel_byte_stripped():
    from assay._safety import redact_secrets

    raw = b"alert\x07: queue full"
    assert redact_secrets(raw) == b"alert: queue full"


def test_anthropic_token_redacted():
    from assay._safety import redact_secrets, REDACTED

    raw = b"Authorization: Bearer sk-ant-api03-AAAAAAAAAAAAAAAAAAAAAA\n"
    out = redact_secrets(raw)
    assert b"sk-ant-api03" not in out
    assert REDACTED in out


def test_openai_classic_token_redacted():
    from assay._safety import redact_secrets, REDACTED

    raw = b"X-API-Key: sk-1234567890abcdef1234567890abcdef\n"
    out = redact_secrets(raw)
    assert b"sk-1234567890" not in out
    assert REDACTED in out


def test_github_pat_redacted():
    from assay._safety import redact_secrets, REDACTED

    raw = b"GH_TOKEN=ghp_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n"
    out = redact_secrets(raw)
    assert b"ghp_AAAAAAA" not in out
    assert REDACTED in out


def test_aws_access_key_redacted():
    from assay._safety import redact_secrets, REDACTED

    raw = b"AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE\n"
    out = redact_secrets(raw)
    assert b"AKIAIOSFODNN7EXAMPLE" not in out
    assert REDACTED in out


def test_unrelated_strings_with_token_prefixes_pass_through():
    """A bare ``sk-`` without enough trailing chars must NOT match — the
    conservative-redaction promise from the module docstring."""
    from assay._safety import redact_secrets

    raw = b"the function is sk-related to crypto"
    out = redact_secrets(raw)
    # The phrase survives intact (the pattern requires ≥ 20 trailing
    # alphanumerics).
    assert b"sk-related" in out


def test_env_value_denylist_redacted(monkeypatch):
    # Set a secret-shaped env var and reload the module so its
    # import-time _ENV_VALUES snapshot picks it up.
    monkeypatch.setenv("MY_FAKE_API_KEY", "shhhhh_this_is_secret_xyz123")
    import assay._safety as _safety
    importlib.reload(_safety)

    raw = b"got result: shhhhh_this_is_secret_xyz123 from server\n"
    out = _safety.redact_secrets(raw)
    assert b"shhhhh_this_is_secret_xyz123" not in out
    assert _safety.REDACTED in out

    # Restore the un-snapshotted version for other tests.
    monkeypatch.delenv("MY_FAKE_API_KEY", raising=False)
    importlib.reload(_safety)


def test_non_secret_env_name_not_redacted(monkeypatch):
    """Only env names matching KEY/TOKEN/SECRET/etc. are in the deny set.
    A random env var with a distinctive value should pass through."""
    monkeypatch.setenv("MY_PROJECT_PATH", "unique_marker_zzz123")
    import assay._safety as _safety
    importlib.reload(_safety)

    raw = b"path: unique_marker_zzz123\n"
    out = _safety.redact_secrets(raw)
    assert b"unique_marker_zzz123" in out

    monkeypatch.delenv("MY_PROJECT_PATH", raising=False)
    importlib.reload(_safety)


def test_short_env_value_not_redacted(monkeypatch):
    """A secret-named env var with a ≤ 7-char value must NOT participate
    in substring redaction — the threshold guards against accidental
    removal of common words."""
    monkeypatch.setenv("MY_FAKE_KEY", "ab")  # length < 8
    import assay._safety as _safety
    importlib.reload(_safety)

    raw = b"ab is just two letters\n"
    out = _safety.redact_secrets(raw)
    assert b"ab is just two letters" in out

    monkeypatch.delenv("MY_FAKE_KEY", raising=False)
    importlib.reload(_safety)


def test_home_path_replaced(monkeypatch):
    monkeypatch.setenv("HOME", "/home/operator")
    import assay._safety as _safety
    importlib.reload(_safety)

    raw = b"file at /home/operator/src/foo.py line 12"
    out = _safety.redact_secrets(raw)
    assert b"/home/operator" not in out
    assert b"$HOME/src/foo.py" in out

    monkeypatch.delenv("HOME", raising=False)
    importlib.reload(_safety)


def test_binary_blob_does_not_crash():
    from assay._safety import redact_secrets

    raw = bytes(range(256)) * 4   # 1KB of every byte value
    out = redact_secrets(raw)
    # The byte 0x1b at byte 27 starts an ANSI sequence; the function
    # may strip portions of binary blobs incidentally. The contract is
    # "doesn't crash"; correctness is best-effort for binary.
    assert isinstance(out, bytes)


def test_bytearray_input_accepted():
    from assay._safety import redact_secrets

    raw = bytearray(b"ghp_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
    out = redact_secrets(raw)
    assert b"ghp_AAAAAAA" not in out
