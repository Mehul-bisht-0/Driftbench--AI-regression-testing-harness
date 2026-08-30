"""Scoring records: assertion outcomes, judge verdicts, per-replicate results,
and the run manifest that pins down exactly what was executed."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from .types import Trajectory, Usage


@dataclass
class AssertionResult:
    name: str
    passed: bool
    detail: str = ""
    critical: bool = True  # a failed non-critical assertion does not fail the task

    def to_dict(self) -> dict:
        return {"name": self.name, "passed": self.passed,
                "detail": self.detail, "critical": self.critical}

    @classmethod
    def from_dict(cls, d: dict) -> "AssertionResult":
        return cls(name=d["name"], passed=d["passed"],
                   detail=d.get("detail", ""), critical=d.get("critical", True))


@dataclass
class JudgeResult:
    """A rubric verdict. score is 1-5; ``label`` is the binarised verdict used for
    calibration against hand labels."""

    score: int = 0
    label: str = "unscored"  # good | bad | unscored
    reasoning: str = ""
    per_criterion: dict = field(default_factory=dict)
    judge_model: str = ""
    usage: Usage = field(default_factory=Usage)

    def to_dict(self) -> dict:
        return {"score": self.score, "label": self.label,
                "reasoning": self.reasoning, "per_criterion": self.per_criterion,
                "judge_model": self.judge_model, "usage": self.usage.to_dict()}

    @classmethod
    def from_dict(cls, d: dict) -> "JudgeResult":
        return cls(score=d.get("score", 0), label=d.get("label", "unscored"),
                   reasoning=d.get("reasoning", ""),
                   per_criterion=d.get("per_criterion", {}),
                   judge_model=d.get("judge_model", ""),
                   usage=Usage.from_dict(d.get("usage", {})))


@dataclass
class ReplicateResult:
    """One (task, variant, replicate) cell: the trajectory plus its scores."""

    trajectory: Trajectory
    assertions: list[AssertionResult] = field(default_factory=list)
    judge: Optional[JudgeResult] = None
    outcome: str = "fail"

    @property
    def task_id(self) -> str:
        return self.trajectory.task_id

    @property
    def variant_id(self) -> str:
        return self.trajectory.variant_id

    @property
    def replicate(self) -> int:
        return self.trajectory.replicate

    @property
    def passed(self) -> bool:
        return self.outcome == "pass"

    def failed_assertions(self) -> list[AssertionResult]:
        return [a for a in self.assertions if not a.passed]

    def to_dict(self) -> dict:
        return {
            "trajectory": self.trajectory.to_dict(),
            "assertions": [a.to_dict() for a in self.assertions],
            "judge": self.judge.to_dict() if self.judge else None,
            "outcome": self.outcome,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ReplicateResult":
        return cls(
            trajectory=Trajectory.from_dict(d["trajectory"]),
            assertions=[AssertionResult.from_dict(a) for a in d.get("assertions", [])],
            judge=JudgeResult.from_dict(d["judge"]) if d.get("judge") else None,
            outcome=d.get("outcome", "fail"),
        )


@dataclass
class RunManifest:
    """Everything needed to say what a run *was*. Two runs are comparable only if
    their suite_digest and replicate count match; the reporter warns otherwise."""

    run_id: str
    created_at: float = field(default_factory=time.time)
    variant_id: str = ""
    variant_digest: str = ""  # digest of system prompt + model + effort
    suite_digest: str = ""  # digest of task ids + their assertion definitions
    agent: str = ""
    model: str = ""
    effort: str = ""
    replicates: int = 1
    task_ids: list[str] = field(default_factory=list)
    master_seed: int = 0
    judge_model: str = ""
    notes: str = ""
    harness_version: str = "0.1.0"

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d: dict) -> "RunManifest":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})
