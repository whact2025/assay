# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Strict, schema-driven hydration for persisted Assay records.

Canonical serialization deliberately erases Python container and ``NewType``
information so it can produce plain JSON.  This module is the inverse at the
Corpus boundary: it reconstructs the declared dataclass graph recursively and
rejects data that does not match that graph.  Persisted records therefore do
not leak JSON ``dict``/``list`` values into fields declared as dataclasses or
tuples.

The decoder is intentionally strict. A stored document with unknown fields,
missing required fields, or an invalid content hash is corrupt rather than an
implicit request for best-effort translation. A missing field is accepted only
when its dataclass metadata declares ``persist_optional`` and a default.
"""

from __future__ import annotations

import dataclasses
import re
import sys
import types
import typing
from pathlib import Path
from typing import Any

from assay.canonical.registry import CONSTRAINT_TYPES, CRITERION_TYPES
from assay.types.identity import ContentHash


_CONTENT_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class RecordDecodeError(ValueError):
    """A canonical JSON document does not satisfy its declared schema."""


def validate_content_hash(value: object, *, path: str = "value") -> ContentHash:
    """Return ``value`` as ``ContentHash`` after strict format validation."""

    if not isinstance(value, str) or _CONTENT_HASH_RE.fullmatch(value) is None:
        raise RecordDecodeError(
            f"{path} must be 'sha256:' followed by 64 lowercase hex digits; "
            f"got {value!r}"
        )
    return ContentHash(value)


def decode_dataclass(data: object, as_type: type[Any]) -> Any:
    """Hydrate ``data`` as ``as_type`` using its complete type schema.

    ``as_type`` must be a concrete dataclass class.  Nested dataclasses,
    variadic/fixed tuples, mappings, optionals/unions, literals, paths, and
    ``typing.NewType`` aliases are reconstructed recursively.  Assay's
    protocol-typed discriminated unions (constraints, criteria, and backend
    errors) resolve through their ``kind`` field.
    """

    if not isinstance(as_type, type) or not dataclasses.is_dataclass(as_type):
        raise TypeError(f"as_type must be a dataclass class, got {as_type!r}")
    return _decode(data, as_type, path=as_type.__name__)


def validate_dataclass_instance(value: object) -> None:
    """Validate a live dataclass through its canonical persisted shape."""

    if isinstance(value, type) or not dataclasses.is_dataclass(value):
        raise TypeError(
            f"record must be a dataclass instance, got {type(value).__name__}"
        )
    # Import locally to keep this module's import graph one-way.
    from assay.canonical import parse, serialize

    decode_dataclass(parse(serialize(value)), type(value))


def _decode(value: object, annotation: object, *, path: str) -> Any:
    if annotation is Any or annotation is object:
        return value

    if annotation is None or annotation is type(None):
        if value is not None:
            raise RecordDecodeError(f"{path} must be null, got {value!r}")
        return None

    if annotation is ContentHash:
        return validate_content_hash(value, path=path)

    # ``typing.NewType`` values are callables with ``__supertype__`` rather
    # than runtime classes.  Decode their underlying value before casting.
    supertype = getattr(annotation, "__supertype__", None)
    if supertype is not None:
        decoded = _decode(value, supertype, path=path)
        return annotation(decoded)  # type: ignore[operator]

    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)

    if origin is typing.Annotated:
        return _decode(value, args[0], path=path)

    if origin is typing.Literal:
        if value not in args or type(value) not in {type(item) for item in args}:
            raise RecordDecodeError(
                f"{path} must be one of {args!r}, got {value!r}"
            )
        return value

    if origin in (typing.Union, types.UnionType):
        failures: list[str] = []
        for candidate in args:
            try:
                return _decode(value, candidate, path=path)
            except (RecordDecodeError, TypeError) as exc:
                failures.append(str(exc))
        raise RecordDecodeError(
            f"{path} does not match any member of {annotation!r}: "
            + "; ".join(failures)
        )

    if origin in (tuple, typing.Tuple):
        if not isinstance(value, list):
            raise RecordDecodeError(
                f"{path} must be a JSON array for tuple, got "
                f"{type(value).__name__}"
            )
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(
                _decode(item, args[0], path=f"{path}[{index}]")
                for index, item in enumerate(value)
            )
        if args and len(value) != len(args):
            raise RecordDecodeError(
                f"{path} must contain {len(args)} items, got {len(value)}"
            )
        if not args:
            return tuple(value)
        return tuple(
            _decode(item, item_type, path=f"{path}[{index}]")
            for index, (item, item_type) in enumerate(zip(value, args))
        )

    if origin in (list, typing.List):
        if not isinstance(value, list):
            raise RecordDecodeError(
                f"{path} must be a JSON array, got {type(value).__name__}"
            )
        item_type = args[0] if args else Any
        return [
            _decode(item, item_type, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]

    if origin in (set, frozenset, typing.Set, typing.FrozenSet):
        if not isinstance(value, list):
            raise RecordDecodeError(
                f"{path} must be a JSON array, got {type(value).__name__}"
            )
        item_type = args[0] if args else Any
        items = (
            _decode(item, item_type, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
        if origin in (frozenset, typing.FrozenSet):
            return frozenset(items)
        return set(items)

    if origin in (dict, typing.Dict) or annotation is dict:
        if not isinstance(value, dict):
            raise RecordDecodeError(
                f"{path} must be a JSON object, got {type(value).__name__}"
            )
        key_type, value_type = args if len(args) == 2 else (str, Any)
        return {
            _decode(key, key_type, path=f"{path}.<key>"): _decode(
                item, value_type, path=f"{path}[{key!r}]"
            )
            for key, item in value.items()
        }

    concrete = _discriminated_type(annotation, value, path=path)
    if concrete is not None:
        return _decode(value, concrete, path=path)

    if isinstance(annotation, type) and dataclasses.is_dataclass(annotation):
        return _decode_dataclass(value, annotation, path=path)

    if annotation is Path:
        if not isinstance(value, str):
            raise RecordDecodeError(f"{path} must be a path string")
        return Path(value)

    if annotation is bool:
        if type(value) is not bool:
            raise RecordDecodeError(f"{path} must be bool, got {value!r}")
        return value
    if annotation is int:
        if type(value) is not int:
            raise RecordDecodeError(f"{path} must be int, got {value!r}")
        return value
    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RecordDecodeError(f"{path} must be numeric, got {value!r}")
        # Python's numeric tower permits an int wherever a float is declared.
        # Preserve the canonical JSON number spelling so hydration does not
        # change the content hash of YAML-authored values such as ``200``.
        return value
    if annotation is str:
        if not isinstance(value, str):
            raise RecordDecodeError(f"{path} must be str, got {value!r}")
        return value

    raise TypeError(f"{path}: unsupported persisted annotation {annotation!r}")


def _decode_dataclass(value: object, cls: type[Any], *, path: str) -> Any:
    if not isinstance(value, dict):
        raise RecordDecodeError(
            f"{path} must be a JSON object for {cls.__name__}, got "
            f"{type(value).__name__}"
        )

    fields = tuple(field for field in dataclasses.fields(cls) if field.init)
    names = {field.name for field in fields}
    by_name = {field.name: field for field in fields}
    missing = sorted(
        name
        for name in names - value.keys()
        if not by_name[name].metadata.get("persist_optional")
    )
    unknown = sorted(value.keys() - names)
    if missing:
        raise RecordDecodeError(f"{path} is missing fields: {missing}")
    if unknown:
        raise RecordDecodeError(f"{path} has unknown fields: {unknown}")

    hints = _resolved_hints(cls)
    kwargs: dict[str, Any] = {}
    for field in fields:
        if field.name in value:
            kwargs[field.name] = _decode(
                value[field.name],
                hints.get(field.name, field.type),
                path=f"{path}.{field.name}",
            )
            continue
        if not field.metadata.get("persist_optional"):
            # The complete required-field check above has already reported
            # this case. Keep the branch defensive if that logic changes.
            raise RecordDecodeError(f"{path} is missing field: {field.name}")
        if field.default is not dataclasses.MISSING:
            kwargs[field.name] = field.default
        elif field.default_factory is not dataclasses.MISSING:
            kwargs[field.name] = field.default_factory()
        else:
            raise TypeError(
                f"{path}.{field.name} is persist_optional without a default"
            )
    try:
        return cls(**kwargs)
    except (TypeError, ValueError) as exc:
        raise RecordDecodeError(f"{path} failed validation: {exc}") from exc


def _resolved_hints(cls: type[Any]) -> dict[str, Any]:
    module = sys.modules.get(cls.__module__)
    globalns = dict(vars(module)) if module is not None else {}
    # Some leaf modules keep cross-type imports under TYPE_CHECKING.  Supply
    # Assay's public type namespace so those forward references still resolve
    # at the persistence boundary.
    import assay.types as assay_types

    globalns.update(vars(assay_types))
    try:
        return typing.get_type_hints(cls, globalns=globalns)
    except (NameError, TypeError) as exc:
        raise TypeError(
            f"cannot resolve persisted schema for {cls.__module__}.{cls.__name__}: "
            f"{exc}"
        ) from exc


def _discriminated_type(
    annotation: object, value: object, *, path: str
) -> type[Any] | None:
    """Resolve Assay's protocol-typed ``kind`` unions."""

    from assay.types import BackendError, Constraint, Criterion
    from assay.types.invocation import (
        BudgetExhausted,
        ContentRejected,
        ExternalCancellation,
        IncompleteEvidence,
        MaxTurnsExceeded,
        StaleConversation,
        TransportError,
    )

    registry: dict[str, type[Any]] | None = None
    if annotation is Constraint:
        registry = CONSTRAINT_TYPES
    elif annotation is Criterion:
        registry = CRITERION_TYPES
    elif annotation is BackendError:
        registry = {
            "transport": TransportError,
            "budget_exhausted": BudgetExhausted,
            "max_turns": MaxTurnsExceeded,
            "content_rejected": ContentRejected,
            "cancellation": ExternalCancellation,
            "incomplete_evidence": IncompleteEvidence,
            "stale_conversation": StaleConversation,
        }
    if registry is None:
        return None
    if not isinstance(value, dict):
        raise RecordDecodeError(
            f"{path} must be an object carrying a 'kind' discriminator"
        )
    kind = value.get("kind")
    if not isinstance(kind, str) or kind not in registry:
        raise RecordDecodeError(
            f"{path}.kind must be one of {sorted(registry)}, got {kind!r}"
        )
    return registry[kind]


__all__ = [
    "RecordDecodeError",
    "decode_dataclass",
    "validate_content_hash",
    "validate_dataclass_instance",
]
