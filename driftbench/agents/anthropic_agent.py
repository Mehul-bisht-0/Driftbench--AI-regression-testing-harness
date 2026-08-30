"""Live Anthropic agent: runs a real agent loop against Claude with tool use.

This is the system under test when you want to measure how a real LLM behaves
on your task suite. It uses the Anthropic Messages API with tool use, running
each turn in a loop until the model stops or the budget is exhausted.

Requires ``anthropic`` package and ``ANTHROPIC_API_KEY`` env var. Lazy-imported
by ``agents/__init__.py`` so the offline path never needs it.
"""

from __future__ import annotations

import json
from typing import Optional

from .. import tools as tools_mod
from ..types import Trajectory
from ..world import World
from .base import Recorder, new_trajectory, render_user_message


class AnthropicAgent:
    """Agent that calls Claude via the Messages API with tool use."""

    def __init__(self, variant, **kw) -> None:
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "the 'anthropic' package is required for the live agent. "
                "Install it with: pip install 'driftbench[live]'"
            ) from None

        self.name = "anthropic"
        self.model = variant.model
        self.effort = variant.effort
        self.system = variant.system_prompt
        self.variant = variant
        self.client = anthropic.Anthropic()

        # Build tool definitions for the API
        allowed = variant.policy()
        self.tool_defs = tools_mod.anthropic_tool_defs()

    def run(self, task, world: World, seed: int, replicate: int) -> Trajectory:
        """Run one task in a tool-use loop against Claude.

        The model does not see the seed — it just sees the task prompt and
        tool results, exactly as a production agent would. The Recorder
        handles step indexing and budget enforcement.
        """
        traj = new_trajectory(task, self.variant, seed, replicate,
                              self.name, self.model)
        rec = Recorder(traj, world, max_calls=task.max_steps,
                       allowed_tools=task.allowed_tools)

        # Build the initial user message
        user_msg = render_user_message(task)
        messages = [{"role": "user", "content": user_msg}]

        for turn in range(task.max_steps):
            if rec.budget_left() <= 0:
                break

            try:
                response = self.client.messages.create(
                    model=self.model,
                    system=self.system,
                    messages=messages,
                    tools=self.tool_defs,
                    max_tokens=4096,
                )
            except Exception as exc:
                rec.harness_error(f"API error: {type(exc).__name__}: {exc}")
                return rec.finish(f"Error: {exc}", stop_reason="error")

            # Track usage
            if hasattr(response, "usage") and response.usage:
                traj.usage.input_tokens += response.usage.input_tokens
                traj.usage.output_tokens += response.usage.output_tokens

            # Process response blocks
            tool_uses = []
            text_parts = []

            for block in response.content:
                if block.type == "text":
                    text_parts.append(block.text)
                    rec.say(block.text)
                elif block.type == "tool_use":
                    tool_uses.append(block)

            if tool_uses:
                # Execute tool calls and build tool_result messages
                assistant_content = response.content
                messages.append({"role": "assistant",
                                 "content": assistant_content})

                tool_results = []
                for tu in tool_uses:
                    args = tu.input if isinstance(tu.input, dict) else {}
                    result, is_error = rec.call(tu.name, args)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "content": json.dumps(result, default=str)
                                   if not is_error else f"Error: {result}",
                        "is_error": is_error,
                    })

                messages.append({"role": "user", "content": tool_results})

            # Check stop reason
            if response.stop_reason == "end_turn":
                final_text = " ".join(text_parts) if text_parts else ""
                return rec.finish(final_text, stop_reason="end_turn")

            if response.stop_reason == "max_tokens":
                final_text = " ".join(text_parts) if text_parts else ""
                return rec.finish(final_text, stop_reason="max_steps")

            if response.stop_reason == "refusal":
                final_text = " ".join(text_parts) if text_parts else ""
                return rec.finish(final_text, stop_reason="refusal")

            # stop_reason == "tool_use" → continue loop

        # Budget exhausted without end_turn
        return rec.finish("", stop_reason="max_steps")
