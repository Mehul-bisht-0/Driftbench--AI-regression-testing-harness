"""Tests for driftbench.runner"""

from __future__ import annotations

from driftbench.runner import decide_outcome, run_cell, RunResult, score
from driftbench.types import Trajectory, Step
from driftbench.variant import Variant, load as load_variant
from driftbench.results import ReplicateResult, AssertionResult, RunManifest
from driftbench.suite import by_id
from driftbench.world import build_world


def _variant():
    return load_variant("prompts/v1_baseline.txt", variant_id="v1_baseline",
                        agent="scripted")


class TestDecideOutcome:
    def test_pass(self):
        traj = Trajectory(task_id="t", variant_id="v", replicate=0, seed=1,
                          stop_reason="end_turn")
        assertions = [AssertionResult(name="a", passed=True, detail="ok")]
        assert decide_outcome(traj, assertions) == "pass"

    def test_fail_critical(self):
        traj = Trajectory(task_id="t", variant_id="v", replicate=0, seed=1,
                          stop_reason="end_turn")
        assertions = [AssertionResult(name="a", passed=False, detail="fail",
                                      critical=True)]
        assert decide_outcome(traj, assertions) == "fail"

    def test_fail_non_critical_still_passes(self):
        traj = Trajectory(task_id="t", variant_id="v", replicate=0, seed=1,
                          stop_reason="end_turn")
        assertions = [AssertionResult(name="a", passed=False, detail="warn",
                                      critical=False)]
        assert decide_outcome(traj, assertions) == "pass"

    def test_error(self):
        traj = Trajectory(task_id="t", variant_id="v", replicate=0, seed=1,
                          error="crash", stop_reason="error")
        assert decide_outcome(traj, []) == "error"

    def test_refusal(self):
        traj = Trajectory(task_id="t", variant_id="v", replicate=0, seed=1,
                          stop_reason="refusal")
        assert decide_outcome(traj, []) == "refusal"

    def test_timeout(self):
        traj = Trajectory(task_id="t", variant_id="v", replicate=0, seed=1,
                          stop_reason="max_steps")
        assert decide_outcome(traj, []) == "timeout"


class TestRunCell:
    def test_passes_on_v1(self):
        v = _variant()
        task = by_id("ret-pricing-growth")
        from driftbench.agents import build as build_agent
        agent = build_agent(v)
        result = run_cell(agent, task, v, master_seed=42, replicate=0)
        assert result.outcome == "pass"
        assert result.trajectory.error is None
        assert len(result.assertions) > 0

    def test_deterministic(self):
        from driftbench.agents import build as build_agent
        v = _variant()
        task = by_id("ret-pricing-growth")
        agent = build_agent(v)
        r1 = run_cell(agent, task, v, master_seed=42, replicate=0)
        r2 = run_cell(agent, task, v, master_seed=42, replicate=0)
        assert r1.outcome == r2.outcome
        assert r1.trajectory.digest == r2.trajectory.digest


class TestScore:
    def test_scores_trajectory(self):
        from driftbench.agents import build as build_agent
        from driftbench.world import build_world
        v = _variant()
        task = by_id("ret-pricing-growth")
        world = build_world(42)
        agent = build_agent(v)
        traj = agent.run(task, world, seed=123, replicate=0)
        assertions = score(task, traj, world)
        assert len(assertions) > 0
        assert all(a.passed for a in assertions if a.critical)


class TestRunResult:
    def test_serialization_round_trip(self):
        from driftbench.agents import build as build_agent
        v = _variant()
        task = by_id("ret-pricing-growth")
        agent = build_agent(v)
        from driftbench.runner import run_suite
        result = run_suite(v, tasks=[task], replicates=2, master_seed=42,
                           progress=None)
        # Round-trip
        d = result.to_dict()
        restored = RunResult.from_dict(d)
        assert len(restored.results) == len(result.results)
        assert restored.manifest.run_id == result.manifest.run_id
        for orig, load in zip(result.results, restored.results):
            assert orig.outcome == load.outcome
            assert len(orig.trajectory.steps) == len(load.trajectory.steps)

    def test_pass_counts(self):
        from driftbench.agents import build as build_agent
        v = _variant()
        task = by_id("ret-pricing-growth")
        agent = build_agent(v)
        from driftbench.runner import run_suite
        result = run_suite(v, tasks=[task], replicates=3, master_seed=42,
                           progress=None)
        counts = result.pass_counts()
        assert "ret-pricing-growth" in counts
        p, n = counts["ret-pricing-growth"]
        assert p == 3
        assert n == 3

    def test_outcome_counts(self):
        from driftbench.agents import build as build_agent
        v = _variant()
        task = by_id("ret-pricing-growth")
        agent = build_agent(v)
        from driftbench.runner import run_suite
        result = run_suite(v, tasks=[task], replicates=2, master_seed=42,
                           progress=None)
        oc = result.outcome_counts()
        assert oc.get("pass", 0) == 2
