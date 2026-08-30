"""Agent implementations. ``build`` is the only thing the runner imports.

The live agent is imported lazily so that the whole offline path - suite, runner,
scoring, diffing, reports - works in an environment with no ``anthropic`` package
and no API key. That is not politeness; an eval harness you cannot run on a
laptop without credentials does not get run.
"""

from __future__ import annotations

from .base import Agent, Recorder, new_trajectory, render_user_message
from .scripted import ScriptedAgent

AGENTS = ("scripted", "anthropic")


def build(variant, jitter: float = 0.0, **kw):
    kind = (variant.agent or "scripted").lower()
    if kind == "scripted":
        return ScriptedAgent(variant, jitter=jitter)
    if kind in ("anthropic", "live", "claude"):
        from .anthropic_agent import AnthropicAgent  # optional dependency
        return AnthropicAgent(variant, **kw)
    raise ValueError(f"unknown agent {variant.agent!r}; expected one of {AGENTS}")


__all__ = ["Agent", "Recorder", "ScriptedAgent", "build", "new_trajectory",
           "render_user_message", "AGENTS"]
