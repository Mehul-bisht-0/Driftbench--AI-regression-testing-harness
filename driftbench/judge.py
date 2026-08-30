"""LLM judge: scores a trajectory against a rubric using an Anthropic model.

The judge reads the task prompt, the rubric criteria, and a formatted version
of the trajectory transcript, then returns a JSON object with per-criterion
scores (1-5) and reasoning. The caller bins the weighted aggregate into a
"good" / "bad" label.

This module lazily imports ``anthropic`` so the offline path (scripted agent,
all analysis) works without the package installed.
"""

from __future__ import annotations

import json
from typing import Optional

from .results import JudgeResult
from .taskspec import TaskSpec
from .types import Trajectory, Usage


def _format_trajectory(traj: Trajectory, max_steps: int = 60) -> str:
    """Render a trajectory into a readable transcript for the judge."""
    lines = []
    for step in traj.steps[:max_steps]:
        if step.kind == "tool_call":
            args_str = json.dumps(step.args, default=str)
            lines.append(f"[tool_call] {step.tool}({args_str})")
        elif step.kind == "tool_result":
            result_str = json.dumps(step.result, default=str)[:200]
            if step.is_error:
                lines.append(f"[tool_error] {step.tool}: {result_str}")
            else:
                lines.append(f"[tool_result] {step.tool}: {result_str}")
        elif step.kind == "assistant_text":
            lines.append(f"[assistant] {step.text[:200]}")
        elif step.kind == "final":
            lines.append(f"[final_answer] {step.text[:500]}")
        elif step.kind == "error":
            lines.append(f"[harness_error] {step.text[:200]}")
    if traj.error:
        lines.append(f"[harness_error] {traj.error}")
    if len(traj.steps) > max_steps:
        lines.append(f"... ({len(traj.steps) - max_steps} more steps omitted)")
    return "\n".join(lines)


def _build_judge_prompt(task: TaskSpec, traj: Trajectory) -> str:
    """Construct the scoring prompt for the judge."""
    criteria_text = ""
    if task.rubric:
        for c in task.rubric.criteria:
            criteria_text += (
                f"- {c.key} (weight {c.weight}): {c.question}\n"
            )
    else:
        criteria_text = (
            "- correctness (weight 2.0): Does the final answer state the "
            "correct facts, with no invented details?\n"
            "- grounding (weight 1.5): Is every factual claim traceable to "
            "something a tool actually returned?\n"
            "- efficiency (weight 0.5): Did the agent reach the answer without "
            "redundant or irrelevant tool calls?\n"
        )

    transcript = _format_trajectory(traj)

    guidance = task.rubric.guidance if task.rubric else ""

    return f"""You are scoring an AI agent's performance on a support-operations task.

TASK:
{task.prompt}

RUBRIC:
{criteria_text}

{f"SCORING GUIDANCE: {guidance}" if guidance else ""}

TRANSCRIPT:
{transcript}

FINAL ANSWER:
{traj.final_text[:1000]}

Score each criterion on a 1-5 scale where:
  1 = completely fails this criterion
  2 = partially fails
  3 = acceptable
  4 = good
  5 = excellent

Respond with ONLY a JSON object (no prose, no markdown fences):
{{
  "scores": {{"criterion_key": score, ...}},
  "reasoning": "one sentence explaining the overall assessment"
}}
"""


class Judge:
    """Scores trajectories against rubrics using an LLM.

    Requires ``anthropic`` package and ``ANTHROPIC_API_KEY`` env var.
    """

    def __init__(self, model: str = "claude-sonnet-5", batch: bool = False):
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "the 'anthropic' package is required for judge scoring. "
                "Install it with: pip install 'driftbench[live]'"
            ) from None
        self.model = model
        self.batch = batch
        self.client = anthropic.Anthropic()

    def score(self, task: TaskSpec, traj: Trajectory) -> JudgeResult:
        """Score one trajectory against the task's rubric."""
        prompt = _build_judge_prompt(task, traj)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:
            return JudgeResult(
                score=0, label="error",
                reasoning=f"judge API error: {exc}",
                judge_model=self.model,
            )

        text = response.content[0].text if response.content else ""

        # Parse JSON response
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown fences
            import re
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                parsed = json.loads(m.group())
            else:
                return JudgeResult(
                    score=0, label="error",
                    reasoning=f"judge returned unparseable output: {text[:200]}",
                    judge_model=self.model,
                )

        scores = parsed.get("scores", {})
        reasoning = parsed.get("reasoning", "")

        # Compute weighted average
        if task.rubric and task.rubric.criteria:
            total_weight = sum(c.weight for c in task.rubric.criteria)
            weighted_sum = 0.0
            per_criterion = {}
            for c in task.rubric.criteria:
                s = scores.get(c.key, 3)  # default to 3 if missing
                s = max(1, min(5, int(s)))
                per_criterion[c.key] = s
                weighted_sum += s * c.weight
            avg = weighted_sum / total_weight if total_weight > 0 else 3.0
        else:
            # Fallback: simple average of all scores
            vals = [max(1, min(5, int(v))) for v in scores.values()]
            avg = sum(vals) / len(vals) if vals else 3.0
            per_criterion = dict(zip(scores.keys(), vals))

        threshold = (task.rubric.pass_threshold if task.rubric else 4)
        label = "good" if avg >= threshold else "bad"

        # Track usage
        usage = Usage()
        if hasattr(response, "usage") and response.usage:
            usage.input_tokens = response.usage.input_tokens
            usage.output_tokens = response.usage.output_tokens

        return JudgeResult(
            score=round(avg),
            label=label,
            reasoning=reasoning,
            per_criterion=per_criterion,
            judge_model=self.model,
            usage=usage,
        )
