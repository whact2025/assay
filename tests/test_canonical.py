# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Tests for ``assay.canonical``: serialize, parse, hash_record.

Coverage:
- Round-trip stability across 10 iterations (same hash every time).
- Float precision (17-digit round-trip).
- Unicode NFC normalization (combining-character equivalence).
- Dict key ordering is deterministic regardless of insertion order.
- Tuples flatten to lists; round-trip recovers the list form.
- Dataclasses recursively expand to dicts.
- Non-finite floats are rejected.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

import assay
import assay.canonical
from assay.canonical import hash_record, parse, serialize


def test_simple_roundtrip_stable_across_10_iterations():
    obj = {"foo": "bar", "n": 42, "pi": 3.141592653589793}
    h0 = hash_record(obj)
    for _ in range(10):
        assert hash_record(obj) == h0


def test_dict_key_ordering_is_canonical():
    a = {"alpha": 1, "beta": 2, "gamma": 3}
    b = {"gamma": 3, "alpha": 1, "beta": 2}
    assert serialize(a) == serialize(b)
    assert hash_record(a) == hash_record(b)


def test_float_17_digit_precision():
    # 0.1 + 0.2 != 0.3 in IEEE 754; serialization preserves the actual bits.
    v = 0.1 + 0.2
    serialized = serialize({"v": v})
    parsed = parse(serialized)
    assert parsed["v"] == v


def test_unicode_nfc_normalization():
    # 'é' as NFC (single codepoint U+00E9) vs NFD (e + combining acute U+0301).
    nfc = "café"
    nfd = "café"
    assert nfc != nfd  # raw form differs
    assert serialize({"name": nfc}) == serialize({"name": nfd})


def test_tuple_serializes_as_list():
    t = (1, 2, 3)
    s = serialize({"items": t})
    parsed = parse(s)
    assert parsed["items"] == [1, 2, 3]


def test_dataclass_recursive_expansion():
    @dataclass(frozen=True)
    class Inner:
        x: int
        y: str

    @dataclass(frozen=True)
    class Outer:
        name: str
        inner: Inner
        values: tuple[float, ...]

    obj = Outer(name="o", inner=Inner(x=1, y="two"), values=(1.5, 2.5))
    serialized = serialize(obj)
    parsed = parse(serialized)
    assert parsed == {"name": "o", "inner": {"x": 1, "y": "two"}, "values": [1.5, 2.5]}


def test_non_finite_floats_rejected():
    with pytest.raises(ValueError):
        serialize({"x": float("nan")})
    with pytest.raises(ValueError):
        serialize({"x": float("inf")})


def test_hash_record_format():
    h = hash_record({"x": 1})
    assert h.startswith("sha256:")
    assert len(h) == len("sha256:") + 64
    assert all(c in "0123456789abcdef" for c in h[len("sha256:"):])


def test_two_equivalent_dicts_have_same_hash():
    h1 = hash_record({"a": 1, "b": [1, 2]})
    h2 = hash_record({"b": [1, 2], "a": 1})
    assert h1 == h2


def test_different_floats_have_different_hashes():
    assert hash_record({"v": 1.0}) != hash_record({"v": 1.0000001})


def test_canonical_surface_has_no_schema_compatibility_api():
    for module in (assay, assay.canonical):
        assert not hasattr(module, "migrate")
        assert not hasattr(module, "register_migrator")
        assert not hasattr(module, "MigrationGapError")
