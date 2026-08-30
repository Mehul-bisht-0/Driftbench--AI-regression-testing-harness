"""Tests for driftbench.compare"""

from __future__ import annotations

from driftbench import compare
from driftbench.compare import comparability, IncomparableRuns, _verdict, TaskComparison


class FakeManifest:
    def __init__(self, suite_digest="abc", replicates=5, master_seed=42,
                 task_ids=None, run_id="run1"):
        self.suite_digest = suite_digest
        self.replicates = replicates
        self.master_seed = master_seed
        self.task_ids = task_ids or []
        self.run_id = run_id
        self.variant_id = "v1"
        self.variant_digest = "def"


class FakeRun:
    def __init__(self, manifest, results=None):
        self.manifest = manifest
        self.results = results or []
        self.variant = None

    def by_task(self):
        return {}


class TestComparability:
    def test_compatible(self):
        base = FakeRun(FakeManifest(task_ids=["t1", "t2"]))
        cand = FakeRun(FakeManifest(task_ids=["t1", "t2"]))
        reasons = comparability(base, cand)
        assert reasons == []

    def test_different_suite(self):
        base = FakeRun(FakeManifest(suite_digest="aaa", task_ids=["t1"]))
        cand = FakeRun(FakeManifest(suite_digest="bbb", task_ids=["t1"]))
        reasons = comparability(base, cand)
        assert any("suite changed" in r for r in reasons)

    def test_different_replicates(self):
        base = FakeRun(FakeManifest(replicates=5))
        cand = FakeRun(FakeManifest(replicates=10))
        reasons = comparability(base, cand)
        assert any("replicate" in r for r in reasons)

    def test_different_seeds(self):
        base = FakeRun(FakeManifest(master_seed=42))
        cand = FakeRun(FakeManifest(master_seed=99))
        reasons = comparability(base, cand)
        assert any("seed" in r for r in reasons)

    def test_no_common_tasks(self):
        base = FakeRun(FakeManifest(task_ids=["t1"]))
        cand = FakeRun(FakeManifest(task_ids=["t2"]))
        reasons = comparability(base, cand)
        assert any("no tasks" in r for r in reasons)


class TestVerdict:
    def test_stable(self):
        t = TaskComparison(task_id="t", delta=0.0, significant=False,
                           traj=None, base_classes=1, cand_classes=1)
        assert _verdict(t, floor=0.1) == "stable"

    def test_regression(self):
        t = TaskComparison(task_id="t", delta=-0.5, significant=True)
        assert _verdict(t, floor=0.1) == "regression"

    def test_improvement(self):
        t = TaskComparison(task_id="t", delta=0.5, significant=True)
        assert _verdict(t, floor=0.1) == "improvement"

    def test_noise(self):
        t = TaskComparison(task_id="t", delta=-0.2, significant=False)
        assert _verdict(t, floor=0.1) == "noise"

    def test_behavior_drift_identical_scores(self):
        from driftbench.diff import TrajectoryDiff, Op
        # Same scores but different trajectory
        ops = [Op("match", 0, 0, "a", "a"),
               Op("sub", 1, 1, "b", "c")]
        d = TaskComparison(task_id="t", delta=0.0, significant=False,
                           traj=TrajectoryDiff(ops=ops, cost=2.0,
                                               a_len=2, b_len=2),
                           base_classes=2, cand_classes=2)
        assert _verdict(d, floor=0.1) == "behavior_drift"

    def test_behavior_drift_different_classes(self):
        t = TaskComparison(task_id="t", delta=0.0, significant=False,
                           traj=None, base_classes=3, cand_classes=2)
        assert _verdict(t, floor=0.1) == "behavior_drift"
