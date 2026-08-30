"""Tests for driftbench.diff"""

from __future__ import annotations

from driftbench import canon, diff
from driftbench.types import Step, Trajectory


def _traj(*tokens, stop="end_turn"):
    """Build a minimal trajectory from canonical tokens (for testing alignment)."""
    steps = []
    for i, tok in enumerate(tokens):
        if tok.startswith("call:"):
            tool = tok.split("(", 1)[0].split(":", 1)[1] if "(" in tok else tok.split(":", 1)[1]
            steps.append(Step(index=i, kind="tool_call", tool=tool, args={}))
        elif tok.startswith("result:"):
            steps.append(Step(index=i, kind="tool_result", tool="x"))
        elif tok == "final":
            steps.append(Step(index=i, kind="final", text="done"))
        else:
            steps.append(Step(index=i, kind="assistant_text", text=tok))
    return Trajectory(task_id="t", variant_id="v", replicate=0, seed=1,
                      steps=steps, stop_reason=stop)


class TestAlign:
    def test_identical(self):
        a = ["call:a", "result:a:ok", "final"]
        b = ["call:a", "result:a:ok", "final"]
        ops, cost = diff.align(a, b)
        assert cost == 0.0
        assert all(op.kind == "match" for op in ops)

    def test_one_substitution(self):
        a = ["call:a", "final"]
        b = ["call:b", "final"]
        ops, cost = diff.align(a, b)
        assert any(op.kind == "sub" for op in ops)
        assert cost > 0.0

    def test_one_insertion(self):
        a = ["call:a", "final"]
        b = ["call:a", "call:b", "final"]
        ops, cost = diff.align(a, b)
        assert any(op.kind == "ins" for op in ops)

    def test_one_deletion(self):
        a = ["call:a", "call:b", "final"]
        b = ["call:a", "final"]
        ops, cost = diff.align(a, b)
        assert any(op.kind == "del" for op in ops)

    def test_empty_a(self):
        ops, cost = diff.align([], ["final"])
        assert cost > 0
        assert len(ops) == 1
        assert ops[0].kind == "ins"

    def test_empty_b(self):
        ops, cost = diff.align(["final"], [])
        assert cost > 0
        assert ops[0].kind == "del"

    def test_both_empty(self):
        ops, cost = diff.align([], [])
        assert cost == 0.0
        assert ops == []


class TestTrajectoryDiff:
    def test_identical(self):
        a = ["call:a", "result:a:ok", "final"]
        d = diff.diff_tokens(a, a)
        assert d.identical is True
        assert d.divergence == 0.0
        assert d.similarity == 1.0

    def test_divergence_range(self):
        a = ["call:a", "final"]
        b = ["call:b", "call:c", "final"]
        d = diff.diff_tokens(a, b)
        assert 0.0 <= d.divergence <= 1.0

    def test_first_divergence_none_when_identical(self):
        d = diff.diff_tokens(["final"], ["final"])
        assert d.first_divergence() is None
        assert d.first_divergence_at() is None

    def test_first_divergence_found(self):
        a = ["call:a", "final"]
        b = ["call:b", "final"]
        d = diff.diff_tokens(a, b)
        op = d.first_divergence()
        assert op is not None
        assert op.kind == "sub"

    def test_render(self):
        d = diff.diff_tokens(["call:a", "final"], ["call:b", "final"])
        lines = d.render()
        assert len(lines) > 0
        assert "~" in lines[0]  # substitution marker

    def test_counts(self):
        a = ["call:a", "call:b", "final"]
        b = ["call:a", "call:c", "final"]
        d = diff.diff_tokens(a, b)
        c = d.counts()
        assert c["match"] >= 1
        assert c["sub"] >= 1


class TestToolDelta:
    def test_same_tools(self):
        t1 = _traj("call:a", "call:b", "final")
        t2 = _traj("call:a", "call:b", "final")
        td = diff.tool_delta(t1, t2)
        assert td.empty is True

    def test_added_tool(self):
        t1 = _traj("call:a", "final")
        t2 = _traj("call:a", "call:b", "final")
        td = diff.tool_delta(t1, t2)
        assert "b" in td.added
        assert "a" in td.kept

    def test_removed_tool(self):
        t1 = _traj("call:a", "call:b", "final")
        t2 = _traj("call:a", "final")
        td = diff.tool_delta(t1, t2)
        assert "b" in td.removed

    def test_summary_empty(self):
        td = diff.ToolDelta()
        assert "same tools" in td.summary()

    def test_summary_with_changes(self):
        td = diff.ToolDelta(added={"b": 2}, removed={"a": 1})
        s = td.summary()
        assert "+b" in s
        assert "-a" in s


class TestCluster:
    def test_all_identical(self):
        trajs = [_traj("call:a", "final") for _ in range(5)]
        classes = diff.cluster(trajs)
        assert len(classes) == 1
        assert classes[0].size == 5

    def test_two_behaviours(self):
        t1 = _traj("call:a", "final")
        t2 = _traj("call:b", "final")
        trajs = [t1, t1, t1, t2, t2]
        classes = diff.cluster(trajs)
        assert len(classes) == 2
        assert classes[0].size == 3  # modal
        assert classes[1].size == 2

    def test_empty(self):
        classes = diff.cluster([])
        assert classes == []


class TestTrajectoryClass:
    def test_label(self):
        cls = diff.TrajectoryClass(
            digest="abc",
            tokens=["call:search_docs", "call:read_doc", "final"],
            members=[0, 1, 2],
        )
        assert "search_docs" in cls.label()
        assert "read_doc" in cls.label()

    def test_plan(self):
        cls = diff.TrajectoryClass(
            digest="abc",
            tokens=["call:search_docs", "result:search_docs:ok", "call:read_doc", "final"],
            members=[0],
        )
        plan = cls.plan()
        assert plan == ["search_docs", "read_doc"]
