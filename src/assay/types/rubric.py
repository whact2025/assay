# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Reviewer prompt material and the strict review-v2 response contract.

A Rubric is pure data: a system prompt (the reviewer's role), a user
template (with an ``{artifact}`` placeholder), and a few knobs that
control parsing and pass/fail logic.

Assay ships the schema and loader but no domain rubric data. Consumers supply
Rubric instances directly or load their own YAML files.

Response-format contract (``response_format='review_v2'``):

``criterion_results`` is present and required only when
``required_criteria`` is nonempty. For a generic rubric with no declared
criteria, that member is omitted and rejected as an extra field.

    {
      "schema_version": "assay.review.v2",
      "decision": "pass" | "revise" | "reject" | "inconclusive",
      "reasoning": "<one-paragraph rationale>",
      "coverage": {
        "complete": <bool>,
        "sources_reviewed": ["artifact", ...],
        "omissions": []
      },
      "findings": [
        {
          "id": "<short-stable-id>",
          "criterion_id": "<rubric-criterion-id>",
          "severity": "Critical" | "High" | "Medium" | "Low",
          "confidence": <0..1>,
          "disposition": "confirmed" | "suspected" | "dismissed" |
                         "addressed" | "unfixable",
          "category": "<freeform-string>",
          "targets": ["<prior-finding-id>", ...],
          "evidence": [
            {"source_id": "artifact", "kind": "...",
             "location": "...", "excerpt": "..."}
          ],
          "proposal": {"action": "...", "replacement": null} | null,
          "attack_spec": {"setup": "...", "stimulus": "...",
                          "oracle": "..."} | null
        }
      ],
      "criterion_results": [
        {
          "criterion_id": "<declared criterion>",
          "outcome": "satisfied" | "violated" | "inconclusive" |
                     "not_applicable",
          "reasoning": "<evidence-backed result>",
          "evidence": [{"source_id": "artifact", "kind": "...",
                        "location": "...", "excerpt": "..."}],
          "headline": true | false
        }
      ]
    }

The reviewer is instructed to emit exactly one such JSON object as its
final message; ``assay.primitives.review`` extracts and validates it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import yaml

from .findings import REVIEW_DECISIONS, ReviewDecision


_VALID_SEVERITIES = ("Critical", "High", "Medium", "Low")


@dataclass(frozen=True)
class Rubric:
    name: str
    system_prompt: str
    user_template: str  # must contain `{artifact}` placeholder
    response_format: str = "review_v2"
    # Lowest severity that counts toward pass/fail. Findings strictly
    # below the floor are recorded but don't fail the verdict.
    severity_floor: str = "Medium"
    # Restrict the global decision taxonomy for a role. A Critic may allow
    # only pass/revise/inconclusive while a terminal Judge may additionally
    # reject. ``pass`` is the only decision whose Verdict.passed property is
    # true; recipes must interpret all other outcomes explicitly.
    allowed_decisions: tuple[ReviewDecision, ...] = (
        "pass",
        "revise",
        "reject",
        "inconclusive",
    )
    # A complete view is required for approval by default. Reviewers can
    # still reject/revise on a partial artifact, but never silently pass it.
    requires_complete_artifact: bool = True
    # Terminal rubric roles can require one evidence-backed result for every
    # declared criterion. Generic/advisory rubrics leave this empty.
    required_criteria: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # Normalize and validate severity_floor at construction. review()'s
        # _passes compares finding severities against this
        # floor via _SEVERITY_ORDER.get(floor, <Medium>); a typo'd or
        # mis-cased floor ("critical", "High ") would miss the table and
        # silently degrade the gate to Medium — passing High findings it
        # should block. Coerce here so every construction path (YAML
        # load, direct, dataclasses.replace) carries a canonical floor.
        floor = (
            self.severity_floor.strip().capitalize()
            if isinstance(self.severity_floor, str)
            else ""
        )
        if floor not in _VALID_SEVERITIES:
            raise ValueError(
                f"rubric {self.name!r}: severity_floor must be one of "
                f"{_VALID_SEVERITIES}, got {self.severity_floor!r}"
            )
        object.__setattr__(self, "severity_floor", floor)
        if self.response_format != "review_v2":
            raise ValueError(
                f"rubric {self.name!r}: response_format must be 'review_v2', "
                f"got {self.response_format!r}"
            )
        if not self.allowed_decisions:
            raise ValueError(
                f"rubric {self.name!r}: allowed_decisions cannot be empty"
            )
        normalized_decisions: list[ReviewDecision] = []
        for decision in self.allowed_decisions:
            if not isinstance(decision, str) or decision not in REVIEW_DECISIONS:
                raise ValueError(
                    f"rubric {self.name!r}: unknown decision {decision!r}; "
                    f"expected one of {sorted(REVIEW_DECISIONS)}"
                )
            if decision not in normalized_decisions:
                normalized_decisions.append(cast(ReviewDecision, decision))
        object.__setattr__(self, "allowed_decisions", tuple(normalized_decisions))
        if not isinstance(self.requires_complete_artifact, bool):
            raise ValueError(
                f"rubric {self.name!r}: requires_complete_artifact must be bool"
            )
        if (
            not isinstance(self.required_criteria, tuple)
            or not all(
                isinstance(item, str) and item.strip()
                for item in self.required_criteria
            )
            or len(self.required_criteria) != len(set(self.required_criteria))
        ):
            raise ValueError(
                f"rubric {self.name!r}: required_criteria must be unique "
                "non-empty strings"
            )
        # Require the placeholder here rather than failing later at format().
        if "{artifact}" not in self.user_template:
            raise ValueError(
                f"rubric {self.name!r}: user_template must contain the "
                "'{artifact}' placeholder"
            )

    @classmethod
    def from_yaml_text(cls, text: str) -> "Rubric":
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ValueError("rubric YAML root must be a mapping")
        missing = [k for k in ("name", "system_prompt", "user_template") if k not in data]
        if missing:
            raise ValueError(f"rubric missing required keys: {missing}")
        allowed = data.get(
            "allowed_decisions",
            ("pass", "revise", "reject", "inconclusive"),
        )
        if not isinstance(allowed, (list, tuple)):
            raise ValueError("rubric allowed_decisions must be an array")
        return cls(
            name=str(data["name"]),
            system_prompt=str(data["system_prompt"]),
            user_template=str(data["user_template"]),
            response_format=str(data.get("response_format", "review_v2")),
            severity_floor=str(data.get("severity_floor", "Medium")),
            allowed_decisions=tuple(allowed),
            requires_complete_artifact=data.get(
                "requires_complete_artifact", True
            ),
            required_criteria=tuple(data.get("required_criteria", ())),
        )

    @classmethod
    def from_yaml(cls, path: Path) -> "Rubric":
        return cls.from_yaml_text(Path(path).read_text(encoding="utf-8"))


__all__ = ["Rubric"]
