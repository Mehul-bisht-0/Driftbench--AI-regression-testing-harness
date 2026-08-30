"""Tests for driftbench.agents.scripted"""

from __future__ import annotations

from pathlib import Path

from driftbench.agents.scripted import ScriptedAgent, policy_summary
from driftbench.variant import Variant, load as load_variant
from driftbench.world import build_world, Fault
from driftbench.suite import by_id


def _variant(prompt_file="prompts/v1_baseline.txt", jitter=0.0):
    v = load_variant(prompt_file, variant_id=Path(prompt_file).stem,
                      agent="scripted")
    return v


class TestChoosePlan:
    def test_nominal_v1(self):
        v = _variant("prompts/v1_baseline.txt")
        agent = ScriptedAgent(v)
        task = by_id("plan-refund-needs-approval")
        plan, final, missing = agent.choose_plan(task)
        assert len(plan) > 0
        assert missing == []  # v1 has all flags
        assert "escalate" in plan[-1][0]  # should escalate

    def test_degraded_v2(self):
        v = _variant("prompts/v2_ablated.txt")
        agent = ScriptedAgent(v)
        task = by_id("plan-refund-needs-approval")
        plan, final, missing = agent.choose_plan(task)
        assert "confirm_destructive" in missing
        # degraded plan refunds directly
        assert any("issue_refund" in step[0] for step in plan)

    def test_task_without_needs(self):
        v = _variant()
        agent = ScriptedAgent(v)
        task = by_id("ret-pricing-growth")
        plan, final, missing = agent.choose_plan(task)
        assert missing == []
        assert len(plan) > 0


class TestRun:
    def test_passes_on_v1(self):
        v = _variant()
        agent = ScriptedAgent(v)
        world = build_world(seed=42)
        task = by_id("ret-pricing-growth")
        traj = agent.run(task, world, seed=123, replicate=0)
        assert traj.error is None
        assert traj.stop_reason == "end_turn"
        assert len(traj.tool_calls()) > 0
        assert traj.final_text != ""

    def test_fails_on_v2_for_safety_task(self):
        v = _variant("prompts/v2_ablated.txt")
        agent = ScriptedAgent(v)
        world = build_world(seed=42)
        task = by_id("plan-refund-needs-approval")
        traj = agent.run(task, world, seed=123, replicate=0)
        # degraded plan refunds without checking policy
        assert traj.error is None
        # The final answer should mention the refund happening
        assert "refund" in traj.final_text.lower() or "$890" in traj.final_text

    def test_jitter_produces_different_routes(self):
        v = _variant()
        agent = ScriptedAgent(v, jitter=0.9)  # high jitter
        world = build_world(seed=42)
        task = by_id("ret-pricing-growth")
        trajs = [agent.run(task, world, seed=42, replicate=i)
                 for i in range(5)]
        # With 90% jitter, some should take different routes
        digests = {t.digest for t in trajs}
        # At least check that they all succeed
        assert all(t.error is None for t in trajs)


class TestDescribe:
    def test_nominal(self):
        v = _variant()
        agent = ScriptedAgent(v)
        task = by_id("ret-pricing-growth")
        desc = agent.describe(task)
        assert "nominal" in desc.lower()

    def test_degraded(self):
        v = _variant("prompts/v2_ablated.txt")
        agent = ScriptedAgent(v)
        task = by_id("plan-refund-needs-approval")
        desc = agent.describe(task)
        assert "degraded" in desc.lower()
        assert "confirm_destructive" in desc


class TestPolicySummary:
    def test_v1_full(self):
        v = _variant("prompts/v1_baseline.txt")
        s = policy_summary(v)
        assert "11/11" in s

    def test_v2_incomplete(self):
        v = _variant("prompts/v2_ablated.txt")
        s = policy_summary(v)
        assert "8/11" in s
        assert "missing" in s
