"""Agent interface and trajectory recording.

Both the offline fixture agent and the live Anthropic agent produce the same
``Trajectory``, because the scoring, diffing, and flakiness code must not know or
care which one ran. That symmetry is what lets the harness be tested with the
scripted agent and then pointed at a real model without changing a line of the
analysis path.

``Recorder`` owns the step-index bookkeeping and the tool-call budget so no agent
implementation has to get that right twice.
"""

from __future__ import annotations

import time
from typing import Any, Optional, Protocol, runtime_checkable

from .. import canon, tools
from ..types import Step, Trajectory
from ..world import World


class Recorder:
    """Builds one Trajectory, enforcing the tool-call budget as it goes."""

    def __init__(self, traj: Trajectory, world: World, max_calls: int,
                 allowed_tools: Optional[list[str]] = None) -> None:
        self.traj = traj
        self.world = world
        self.max_calls = max_calls
        self.allowed = set(allowed_tools) if allowed_tools else None
        self._t0 = time.perf_counter()

    # --- state ------------------------------------------------------------
    @property
    def calls_made(self) -> int:
        return len(self.traj.tool_calls())

    def budget_left(self) -> int:
        return self.max_calls - self.calls_made

    def _append(self, **kw) -> Step:
        step = Step(index=len(self.traj.steps), **kw)
        self.traj.steps.append(step)
        return step

    # --- recording --------------------------------------------------------
    def say(self, text: str) -> None:
        if text:
            self._append(kind="assistant_text", text=text)

    def call(self, tool: str, args: dict, latency_ms: int = 0) -> tuple[Any, bool]:
        """Record a tool call, execute it against the world, record the result.

        Budget exhaustion and disallowed tools are recorded as errored results
        rather than raised, because "the agent burned its budget" is a finding
        about the agent, and losing the partial trajectory would hide it.
        """
        if self.budget_left() <= 0:
            self._append(kind="error",
                         text=f"tool-call budget of {self.max_calls} exhausted "
                              f"before {tool}")
            self.traj.stop_reason = "max_steps"
            return f"Error: tool-call budget of {self.max_calls} exhausted", True

        self._append(kind="tool_call", tool=tool, args=dict(args),
                     latency_ms=latency_ms)
        if self.allowed is not None and tool not in self.allowed:
            result, is_error = (
                f"Error: tool {tool!r} is not available for this task", True)
        else:
            result, is_error = tools.call_tool(self.world, tool, args)
        self._append(kind="tool_result", tool=tool, result=result,
                     is_error=is_error, latency_ms=latency_ms)
        return result, is_error

    def harness_error(self, text: str) -> None:
        self._append(kind="error", text=text)
        self.traj.error = text

    def finish(self, text: str, stop_reason: str = "end_turn",
               refusal_category: Optional[str] = None) -> Trajectory:
        self._append(kind="final", text=text)
        self.traj.final_text = text
        if self.traj.stop_reason != "max_steps":
            self.traj.stop_reason = stop_reason
        self.traj.refusal_category = refusal_category
        self.traj.wall_ms = int((time.perf_counter() - self._t0) * 1000)
        canon.annotate(self.traj)
        return self.traj


@runtime_checkable
class Agent(Protocol):
    """What the runner needs from a system under test."""

    name: str
    model: str

    def run(self, task, world: World, seed: int, replicate: int) -> Trajectory:
        ...


def new_trajectory(task, variant, seed: int, replicate: int, agent_name: str,
                   model: str) -> Trajectory:
    return Trajectory(task_id=task.id, variant_id=variant.id, replicate=replicate,
                      seed=seed, agent=agent_name, model=model)


def render_user_message(task) -> str:
    """The user turn. ``given`` facts are inlined so that tasks which test
    "use what you were handed" can be scored on whether a tool was called."""
    if not task.given:
        return task.prompt
    facts = "\n".join(f"- {k}: {v}" for k, v in sorted(task.given.items()))
    return f"{task.prompt}\n\nContext you already have:\n{facts}"
