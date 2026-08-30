"""Core data model.

Everything the harness records is one of the dataclasses here. They are plain
stdlib dataclasses with explicit dict round-tripping, so a trajectory recorded
today still loads after the code moves on. Bump SCHEMA_VERSION on any change
that is not backwards compatible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from . import pricing

SCHEMA_VERSION = 1

# Step kinds. "final" is the terminating assistant answer; "error" is a harness
# or transport failure, not a tool returning is_error.
STEP_KINDS = ("tool_call", "tool_result", "assistant_text", "final", "error")

# Terminal outcome of one replicate.
OUTCOMES = ("pass", "fail", "error", "refusal", "timeout")


@dataclass
class Step:
    index: int
    kind: str
    tool: Optional[str] = None
    args: dict = field(default_factory=dict)
    result: Any = None
    is_error: bool = False
    text: str = ""
    latency_ms: int = 0
    canon: str = ""  # canonical token, filled in by canon.annotate()

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "kind": self.kind,
            "tool": self.tool,
            "args": self.args,
            "result": self.result,
            "is_error": self.is_error,
            "text": self.text,
            "latency_ms": self.latency_ms,
            "canon": self.canon,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Step":
        return cls(**{k: d.get(k, getattr(cls, k, None)) for k in (
            "index", "kind", "tool", "args", "result", "is_error", "text",
            "latency_ms", "canon")})


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    requests: int = 0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.cache_write_tokens += other.cache_write_tokens
        self.requests += other.requests

    def cost_usd(self, model: str, batch: bool = False) -> float:
        return pricing.cost_usd(
            model,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cache_read_tokens=self.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens,
            batch=batch,
        )

    def to_dict(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "requests": self.requests,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Usage":
        return cls(**{k: d.get(k, 0) for k in (
            "input_tokens", "output_tokens", "cache_read_tokens",
            "cache_write_tokens", "requests")})


@dataclass
class Trajectory:
    """One agent episode: everything observable about a single attempt."""

    task_id: str
    variant_id: str
    replicate: int
    seed: int
    steps: list[Step] = field(default_factory=list)
    final_text: str = ""
    stop_reason: str = "end_turn"
    refusal_category: Optional[str] = None
    usage: Usage = field(default_factory=Usage)
    wall_ms: int = 0
    error: Optional[str] = None
    agent: str = ""
    model: str = ""
    digest: str = ""  # canonical trajectory digest, filled in by canon.annotate()

    def tool_calls(self) -> list[Step]:
        return [s for s in self.steps if s.kind == "tool_call"]

    def tool_names(self) -> list[str]:
        return [s.tool for s in self.steps if s.kind == "tool_call" and s.tool]

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "variant_id": self.variant_id,
            "replicate": self.replicate,
            "seed": self.seed,
            "steps": [s.to_dict() for s in self.steps],
            "final_text": self.final_text,
            "stop_reason": self.stop_reason,
            "refusal_category": self.refusal_category,
            "usage": self.usage.to_dict(),
            "wall_ms": self.wall_ms,
            "error": self.error,
            "agent": self.agent,
            "model": self.model,
            "digest": self.digest,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Trajectory":
        return cls(
            task_id=d["task_id"],
            variant_id=d["variant_id"],
            replicate=d["replicate"],
            seed=d["seed"],
            steps=[Step.from_dict(s) for s in d.get("steps", [])],
            final_text=d.get("final_text", ""),
            stop_reason=d.get("stop_reason", "end_turn"),
            refusal_category=d.get("refusal_category"),
            usage=Usage.from_dict(d.get("usage", {})),
            wall_ms=d.get("wall_ms", 0),
            error=d.get("error"),
            agent=d.get("agent", ""),
            model=d.get("model", ""),
            digest=d.get("digest", ""),
        )
