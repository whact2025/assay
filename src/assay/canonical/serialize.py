# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Deterministic serialization for Assay records.

This is an Assay-specific canonical JSON profile, not an RFC 8785
implementation. In particular, Assay normalizes strings to NFC and first
round-trips floats through ``.17g`` before Python's JSON encoder renders them.
The profile is:

- Floats are normalized through ``.17g`` (IEEE 754 round-trip) before the
  Python JSON encoder chooses their wire spelling.
- Strings normalize to Unicode NFC before serialization.
- Dict keys serialize in Python's lexicographic Unicode order.
- No insignificant whitespace.
- No trailing newline.
- UTF-8 byte output.
- ``hash_record`` returns ``'sha256:<lowercase-hex>'``.

Dataclass instances are walked field by field so
``metadata={"canonical_omit_empty": True}`` can omit empty values and
``metadata={"canonical_omit_default": True}`` can preserve a legacy shape
when an additive field retains its declared default. Nested containers are
expanded recursively and tuples become JSON arrays.

Leaf module: stdlib only.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import unicodedata
from typing import Any

from .registry import CONSTRAINT_TYPES, CRITERION_TYPES  # noqa: F401  (re-exported)


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert a value to a JSON-canonicalizable structure.

    - Dataclasses -> dict of fields
    - Tuples / lists -> list
    - Dicts -> dict (keys must be str; non-str keys are rejected)
    - Strings -> NFC-normalized
    - Floats -> 17-significant-digit string-formatted float (re-parsed)
    - None / bool / int -> pass through
    """
    if obj is None or isinstance(obj, bool):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            raise ValueError(f"non-finite float {obj!r} is not canonicalizable")
        # 17 significant digits guarantees IEEE 754 round-trip.
        # We round-trip through float() so that the in-memory value
        # matches what json.dumps would emit (avoiding double-format).
        return float(format(obj, ".17g"))
    if isinstance(obj, str):
        return unicodedata.normalize("NFC", obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        result: dict[str, Any] = {}
        for f in dataclasses.fields(obj):
            field_value = getattr(obj, f.name)
            if f.metadata.get("canonical_omit_empty") and not field_value:
                continue
            if f.metadata.get("canonical_omit_default"):
                if f.default is not dataclasses.MISSING:
                    default = f.default
                elif f.default_factory is not dataclasses.MISSING:
                    default = f.default_factory()
                else:
                    raise TypeError(
                        f"{type(obj).__name__}.{f.name} declares "
                        "canonical_omit_default without a default"
                    )
                if field_value == default:
                    continue
            result[f.name] = _to_jsonable(field_value)
        return result
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        result_d: dict[str, Any] = {}
        for k, v in obj.items():
            if not isinstance(k, str):
                raise TypeError(
                    f"canonical dict keys must be str, got {type(k).__name__}"
                )
            result_d[unicodedata.normalize("NFC", k)] = _to_jsonable(v)
        return result_d
    raise TypeError(f"cannot canonicalize {type(obj).__name__}: {obj!r}")


def serialize(obj: Any) -> bytes:
    """Serialize a supported value to Assay-canonical UTF-8 JSON bytes.

    The accepted graph is intentionally narrow: dataclass instances, string-
    keyed dicts, lists/tuples, finite numbers, booleans, strings, and ``None``.
    Unsupported values and non-string keys raise ``TypeError``; non-finite
    floats raise ``ValueError``.
    """
    jsonable = _to_jsonable(obj)
    return json.dumps(
        jsonable,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def parse(blob: bytes | bytearray) -> Any:
    """Decode UTF-8 JSON bytes into ordinary Python JSON values.

    ``parse`` is a decoder, not a canonical-form validator: valid noncanonical
    JSON (for example, indented JSON) is accepted. For supported input values,
    ``parse(serialize(x))`` equals Assay's canonicalized JSON-shaped form.
    """
    if not isinstance(blob, (bytes, bytearray)):
        raise TypeError(f"expected bytes, got {type(blob).__name__}")
    return json.loads(blob.decode("utf-8"))


def hash_record(obj: Any) -> str:
    """Compute the canonical ContentHash for ``obj``.

    Returns a string of the form ``'sha256:<lowercase-hex>'``. The
    return type is ``str`` (rather than ``ContentHash``) so this
    module can stay free of ``assay.types`` imports — callers that
    need the ``ContentHash`` NewType can simply cast.
    """
    return "sha256:" + hashlib.sha256(serialize(obj)).hexdigest()
