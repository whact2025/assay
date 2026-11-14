# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Recursive immutable containers for JSON-shaped frozen-record fields."""

from __future__ import annotations

from typing import Any, Mapping


class FrozenDict(dict):
    """A ``dict``-compatible mapping that rejects every mutation method.

    Subclassing ``dict`` keeps JSON-schema and canonical-serialization code
    compatible while closing the shallow-frozen dataclass loophole.
    """

    @staticmethod
    def _immutable(*args: Any, **kwargs: Any) -> None:
        raise TypeError("FrozenDict is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable

    def copy(self) -> "FrozenDict":
        return self

    def __copy__(self) -> "FrozenDict":
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> "FrozenDict":
        return self


def freeze_dict(value: Mapping | None) -> FrozenDict:
    """Freeze nested mappings, lists/tuples, and sets in a copied mapping.

    Arbitrary leaf objects are retained by reference; callers should use this
    helper with value graphs whose leaves are already immutable.
    """
    if isinstance(value, FrozenDict):
        return value
    return FrozenDict({
        key: freeze_value(item) for key, item in (value or {}).items()
    })


def freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return freeze_dict(value)
    if isinstance(value, (list, tuple)):
        return tuple(freeze_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(freeze_value(item) for item in value)
    return value


__all__ = ["FrozenDict", "freeze_dict", "freeze_value"]
