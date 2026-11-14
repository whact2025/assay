# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Provider-neutral, explicitly uncertain cost evidence for diagnostics.

These records are not exact pricing evidence. A workflow must separately opt
in to their policy before using their upper bound for paid continuation.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class UsageCostBoundSegment:
    input_tokens: int
    output_tokens: int
    cache_creation_tokens: int
    cache_read_tokens: int | None
    lower_bound_usd: float
    upper_bound_usd: float
    policy: str

    def __post_init__(self) -> None:
        for name in ("input_tokens", "output_tokens", "cache_creation_tokens"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be an exact non-negative integer")
        if self.cache_read_tokens is not None and (
            type(self.cache_read_tokens) is not int or self.cache_read_tokens < 0
        ):
            raise ValueError("cache_read_tokens must be an exact integer or None")
        if self.cache_creation_tokens + (self.cache_read_tokens or 0) > self.input_tokens:
            raise ValueError("known cache subsets exceed input tokens")
        for name in ("lower_bound_usd", "upper_bound_usd"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.lower_bound_usd > self.upper_bound_usd:
            raise ValueError("cost lower bound exceeds upper bound")
        if not isinstance(self.policy, str) or not self.policy.strip():
            raise ValueError("cost-bound policy must be non-empty")


@dataclass(frozen=True)
class UsageCostBounds:
    """Ordered interval evidence, distinct from complete actual accounting.

    ``coverage_complete`` means every started provider response is represented;
    it never means that cache counts or the actual charge are known.
    """

    segments: tuple[UsageCostBoundSegment, ...]
    coverage_complete: bool

    def __post_init__(self) -> None:
        if type(self.segments) is not tuple or any(
            type(segment) is not UsageCostBoundSegment for segment in self.segments
        ):
            raise TypeError("cost-bound segments must be a typed tuple")
        if type(self.coverage_complete) is not bool:
            raise TypeError("coverage_complete must be bool")

    @property
    def lower_bound_usd(self) -> float:
        return math.fsum(segment.lower_bound_usd for segment in self.segments)

    @property
    def upper_bound_usd(self) -> float:
        return math.fsum(segment.upper_bound_usd for segment in self.segments)

    def covers(self, usage: Any) -> bool:
        """Reconcile full coverage to known usage and its cost lower bound."""

        if not self.coverage_complete:
            return False
        for name in ("input_tokens", "output_tokens", "cache_creation_tokens", "cache_read_tokens"):
            value = getattr(usage, name, None)
            if type(value) is not int or value < 0:
                return False
            known_total = sum(getattr(segment, name) or 0 for segment in self.segments)
            if value != known_total:
                return False
        cost = getattr(usage, "cost_usd", None)
        return (
            type(cost) in (int, float)
            and math.isfinite(cost)
            and math.isclose(cost, self.lower_bound_usd, rel_tol=1e-12, abs_tol=1e-12)
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "segments": [asdict(segment) for segment in self.segments],
            "coverage_complete": self.coverage_complete,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "UsageCostBounds":
        if type(record) is not dict or set(record) != {"segments", "coverage_complete"}:
            raise ValueError("invalid cost-bound record fields")
        segments = record["segments"]
        if type(segments) is not list:
            raise ValueError("cost-bound segments must be a list")
        fields = {
            "input_tokens", "output_tokens", "cache_creation_tokens", "cache_read_tokens",
            "lower_bound_usd", "upper_bound_usd", "policy",
        }
        if any(type(item) is not dict or set(item) != fields for item in segments):
            raise ValueError("invalid cost-bound segment fields")
        return cls(
            segments=tuple(UsageCostBoundSegment(**item) for item in segments),
            coverage_complete=record["coverage_complete"],
        )


def combine_cost_bounds(
    values: list[Any] | tuple[Any, ...],
    *,
    completeness: list[Any] | tuple[Any, ...] | None = None,
) -> UsageCostBounds | None:
    """Concatenate evidence without granting coverage to unbounded calls."""

    usages = [getattr(value, "usage", value) for value in values]
    available = [getattr(value, "cost_bounds", None) for value in values]
    if not any(bound is not None for bound in available):
        return None
    if completeness is not None and len(completeness) != len(values):
        raise ValueError("cost-bound completeness inventory must match calls")
    for index, (value, usage, bound) in enumerate(zip(values, usages, available, strict=True)):
        if bound is not None:
            continue
        evidence = (
            completeness[index] if completeness is not None
            else getattr(value, "completeness", getattr(value, "execution_completeness", None))
        )
        if getattr(evidence, "complete", False) is not True:
            continue
        try:
            exact_segment = UsageCostBoundSegment(
                input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
                cache_creation_tokens=usage.cache_creation_tokens,
                cache_read_tokens=usage.cache_read_tokens,
                lower_bound_usd=usage.cost_usd, upper_bound_usd=usage.cost_usd,
                policy="observed-exact-cost-v1",
            )
        except (AttributeError, TypeError, ValueError):
            continue
        if exact_segment.cache_read_tokens is not None:
            available[index] = UsageCostBounds((exact_segment,), True)
    segments = tuple(
        segment
        for bound in available
        if type(bound) is UsageCostBounds
        for segment in bound.segments
    )
    return UsageCostBounds(
        segments=segments,
        coverage_complete=all(
            type(bound) is UsageCostBounds and bound.covers(usage)
            for bound, usage in zip(available, usages, strict=True)
        ),
    )


def cost_bounds_cover(value: Any, *, policy: str | None) -> bool:
    """Generic primitive admission; callers must supply explicit policy."""

    if not isinstance(policy, str) or not policy:
        return False
    usage = getattr(value, "usage", value)
    bounds = getattr(value, "cost_bounds", None) or getattr(usage, "cost_bounds", None)
    completeness = getattr(value, "completeness", None)
    if completeness is None:
        completeness = getattr(value, "execution_completeness", None)
    return (
        type(bounds) is UsageCostBounds
        and bool(bounds.segments)
        and any(segment.policy == policy for segment in bounds.segments)
        and all(
            segment.policy == policy or (
                segment.policy == "observed-exact-cost-v1"
                and segment.cache_read_tokens is not None
                and segment.lower_bound_usd == segment.upper_bound_usd
            )
            for segment in bounds.segments
        )
        and bounds.covers(usage)
        and getattr(completeness, "terminal_complete", False) is True
        and getattr(completeness, "trace_complete", False) is True
    )
