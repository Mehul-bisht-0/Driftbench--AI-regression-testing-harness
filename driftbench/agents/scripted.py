"""The offline fixture agent: a system under test you can run 800 times for free.

This is not a mock of an agent so much as a *specimen*. It reads the system prompt
the same way a real agent would be influenced by it - through
``policy.parse`` - and each task declares which operating rules its correct
behaviour depends on. Remove a rule from the prompt and the tasks that named it
fall back to their ``degraded`` plan; the rest are untouched. So the headline demo
is a real measurement of a real behaviour change, not a scripted light show.

It also supplies the two things the analysis code needs ground truth for:

* **outcome flakiness** - seeded tool faults plus bounded retries, so
  ``err-flaky-doc-index`` fails at a known ~16% rate.
* **latent flakiness** - with ``jitter`` above zero it sometimes takes a longer
  route to the same correct answer, which is the case that passes every run today
  and breaks the day the budget tightens.
"""

from __future__ import annotations

from typing import Any, Optional

from .. import policy
from ..seeding import SeededRandom
from ..types import Trajectory
from ..world import World
from .base import Recorder, new_trajectory


class ScriptedAgent:
    """Replays a task's declared plan, subject to the prompt's operating rules."""

    def __init__(self, variant, jitter: float = 0.0) -> None:
        self.variant = variant
        self.jitter = jitter
        self.name = "scripted"
        self.model = "scripted"
        self.policy = set(variant.policy())

    # --- plan selection ---------------------------------------------------
    def choose_plan(self, task) -> tuple[list, str, list[str]]:
        """Pick the good plan or the degraded one, and say why.

        Returns ``(plan, final, missing_flags)``. A task that names a rule the
        prompt does not contain degrades; a task that names nothing is immune to
        prompt edits, which is exactly what you want most of the suite to be.
        """
        script = task.script or {}
        needed = list(script.get("needs") or ())
        missing = [f for f in needed if f not in self.policy]
        if missing and script.get("degraded"):
            fallback = script["degraded"]
            return list(fallback.get("plan") or ()), fallback.get("final", ""), missing
        return list(script.get("plan") or ()), script.get("final", ""), missing

    # --- execution --------------------------------------------------------
    def run(self, task, world: World, seed: int, replicate: int) -> Trajectory:
        rng = SeededRandom(seed, "agent")
        traj = new_trajectory(task, self.variant, seed, replicate, self.name,
                              self.model)
        rec = Recorder(traj, world, max_calls=task.max_steps,
                       allowed_tools=task.allowed_tools)

        plan, final, missing = self.choose_plan(task)
        script = task.script or {}
        # A prompt that says "retry" is what licenses retrying at all: an agent
        # that retries a permanently dead tool forever is its own failure mode.
        retries_allowed = (script.get("max_retries", 0)
                           if script.get("retry_on_error")
                           and "bounded_retry" in self.policy else 0)
        broken: Optional[tuple[str, Any]] = None

        for i, (tool_name, args) in enumerate(plan):
            if self.jitter and rng.substream(f"jitter/{i}").chance(self.jitter):
                # Same answer, longer route: latent flakiness with a known cause.
                rec.call(tool_name, dict(args), latency_ms=self._latency(rng, i))
            result, is_error = rec.call(tool_name, dict(args),
                                        latency_ms=self._latency(rng, i))
            attempt = 0
            while is_error and attempt < retries_allowed:
                attempt += 1
                result, is_error = rec.call(tool_name, dict(args),
                                            latency_ms=self._latency(rng, i))
            if is_error:
                broken = (tool_name, result)
                break

        if broken is not None:
            tool_name, result = broken
            return rec.finish(script.get("fail_final") or self._fail_text(
                tool_name, result, retries_allowed))
        return rec.finish(final)

    def _latency(self, rng: SeededRandom, i: int) -> int:
        """Fake latency, drawn from a labelled substream so adding a draw
        elsewhere cannot shift it. Only ever used for reporting."""
        return rng.substream(f"latency/{i}").randint(40, 400)

    @staticmethod
    def _fail_text(tool_name: str, result: Any, retries: int) -> str:
        attempts = retries + 1
        return (f"I could not complete this. {tool_name} failed on all "
                f"{attempts} attempt(s) - it returned: {result}. I do not want to "
                f"guess at the answer, so I am reporting the failure instead.")

    def describe(self, task) -> str:
        """Why this task will behave the way it does, for --explain."""
        _, _, missing = self.choose_plan(task)
        if not missing:
            return "nominal plan"
        return f"degraded: prompt is missing {', '.join(missing)}"


def policy_summary(variant) -> str:
    present = sorted(variant.policy())
    absent = sorted(set(policy.ALL_FLAGS) - set(present))
    return f"{len(present)}/{len(policy.ALL_FLAGS)} rules present" + (
        f"; missing {', '.join(absent)}" if absent else "")
