"""Tests for driftbench.canon"""

from __future__ import annotations

import json

from driftbench import canon
from driftbench.types import Step, Trajectory


class TestCanonToken:
    def test_tool_call_shape(self):
        step = Step(index=0, kind="tool_call", tool="search_docs",
                    args={"query": "refund policy"})
        tok = canon.canon_token(step, level="shape")
        assert tok == "call:search_docs"

    def test_tool_call_semantic(self):
        step = Step(index=0, kind="tool_call", tool="search_docs",
                    args={"query": "refund policy"})
        tok = canon.canon_token(step, level="semantic")
        assert tok.startswith("call:search_docs(")
        # query is word-bag normalised
        assert "policy" in tok or "refund" in tok

    def test_tool_call_strict(self):
        step = Step(index=0, kind="tool_call", tool="read_file",
                    args={"path": "src/main.py"})
        tok_strict = canon.canon_token(step, level="strict")
        tok_semantic = canon.canon_token(step, level="semantic")
        # strict preserves path casing, semantic lowercases
        assert "src/main.py" in tok_strict or "src/main.py" in tok_semantic

    def test_tool_result_ok(self):
        step = Step(index=1, kind="tool_result", tool="search_docs",
                    result={"results": []}, is_error=False)
        tok = canon.canon_token(step)
        assert tok == "result:search_docs:ok"

    def test_tool_result_error(self):
        step = Step(index=1, kind="tool_result", tool="http_get",
                    result="503 timeout", is_error=True)
        tok = canon.canon_token(step)
        assert tok == "result:http_get:err"

    def test_assistant_text_shape(self):
        step = Step(index=0, kind="assistant_text", text="hello world")
        tok = canon.canon_token(step, level="shape")
        assert tok == "say"

    def test_assistant_text_semantic(self):
        step = Step(index=0, kind="assistant_text", text="hello world")
        tok = canon.canon_token(step, level="semantic")
        assert tok.startswith("say:")

    def test_final(self):
        step = Step(index=0, kind="final", text="done")
        assert canon.canon_token(step) == "final"

    def test_error(self):
        step = Step(index=0, kind="error", text="budget exhausted")
        tok = canon.canon_token(step)
        assert tok.startswith("harness_error:")

    def test_invalid_level_raises(self):
        step = Step(index=0, kind="final")
        import pytest
        with pytest.raises(ValueError, match="unknown canon level"):
            canon.canon_token(step, level="invalid")


class TestCanonSequence:
    def test_basic(self):
        traj = Trajectory(task_id="t", variant_id="v", replicate=0, seed=1)
        traj.steps = [
            Step(index=0, kind="tool_call", tool="search_docs",
                 args={"query": "test"}),
            Step(index=1, kind="tool_result", tool="search_docs",
                 result="ok"),
            Step(index=2, kind="final", text="done"),
        ]
        seq = canon.canon_sequence(traj, level="shape")
        assert seq == ["call:search_docs", "result:search_docs:ok", "final"]


class TestDigest:
    def test_deterministic(self):
        traj = Trajectory(task_id="t", variant_id="v", replicate=0, seed=1)
        traj.steps = [Step(index=0, kind="final", text="ok")]
        d1 = canon.trajectory_digest(traj)
        d2 = canon.trajectory_digest(traj)
        assert d1 == d2
        assert len(d1) == 20  # blake2b 10-byte hex

    def test_different_stop_reasons_differ(self):
        t1 = Trajectory(task_id="t", variant_id="v", replicate=0, seed=1,
                        stop_reason="end_turn")
        t1.steps = [Step(index=0, kind="final", text="ok")]
        t2 = Trajectory(task_id="t", variant_id="v", replicate=0, seed=1,
                        stop_reason="refusal")
        t2.steps = [Step(index=0, kind="final", text="ok")]
        assert canon.trajectory_digest(t1) != canon.trajectory_digest(t2)


class TestNormText:
    def test_basic(self):
        assert canon.norm_text("Hello, World!") == "hello world"
        assert canon.norm_text("  lots   of   spaces  ") == "lots of spaces"

    def test_empty(self):
        assert canon.norm_text("") == ""
        assert canon.norm_text(None) == ""


class TestWordBag:
    def test_order_insensitive(self):
        a = canon.word_bag("python list sort")
        b = canon.word_bag("sort python list")
        assert a == b

    def test_case_insensitive(self):
        assert canon.word_bag("Hello World") == canon.word_bag("hello world")


class TestAnnotate:
    def test_fills_canon_and_digest(self):
        traj = Trajectory(task_id="t", variant_id="v", replicate=0, seed=1)
        traj.steps = [
            Step(index=0, kind="tool_call", tool="read_file",
                 args={"path": "a.txt"}),
            Step(index=1, kind="tool_result", tool="read_file",
                 result="content"),
            Step(index=2, kind="final", text="done"),
        ]
        canon.annotate(traj)
        assert traj.digest != ""
        assert traj.steps[0].canon.startswith("call:read_file")
        assert traj.steps[2].canon == "final"


class TestArgNormalizers:
    def test_search_docs_normalizer(self):
        normed = canon.normalize_args("search_docs",
                                       {"query": "Refund Policy Days",
                                        "limit": 5}, "semantic")
        # query should be word-bag normalised and sorted
        assert "refund" in normed["query"]
        # limit is removed by the normalizer
        assert "limit" not in normed

    def test_path_normalizer(self):
        normed = canon.normalize_args("read_file",
                                       {"path": "./src/Main.py"}, "semantic")
        assert normed["path"] == "src/main.py"

    def test_strict_level_no_normalizer(self):
        normed = canon.normalize_args("read_file",
                                       {"path": "./src/Main.py"}, "strict")
        assert normed["path"] == "./src/Main.py"
