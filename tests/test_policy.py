"""Tests for driftbench.policy"""

from __future__ import annotations

from pathlib import Path

import pytest

from driftbench import policy


V1_PROMPT = Path("prompts/v1_baseline.txt").read_text(encoding="utf-8")
V2_PROMPT = Path("prompts/v2_ablated.txt").read_text(encoding="utf-8")


class TestParse:
    def test_v1_all_flags(self):
        found = policy.parse(V1_PROMPT)
        for flag in policy.ALL_FLAGS:
            assert flag in found, f"v1 baseline missing flag: {flag}"

    def test_v2_missing_safety_line(self):
        found = policy.parse(V2_PROMPT)
        # These 3 flags come from the single removed line
        assert "confirm_destructive" not in found
        assert "no_unrequested_outward" not in found
        assert "refuse_bulk" not in found

    def test_v2_preserves_others(self):
        found = policy.parse(V2_PROMPT)
        # These should still be present
        assert "no_speculation" in found
        assert "prefer_context" in found
        assert "consult_policy" in found
        assert "protect_secrets" in found
        assert "terse_format" in found
        assert "ask_when_ambiguous" in found
        assert "no_redundant_calls" in found
        assert "bounded_retry" in found

    def test_empty_prompt(self):
        assert policy.parse("") == set()
        assert policy.parse(None) == set()


class TestMissing:
    def test_v1_no_missing(self):
        m = policy.missing(V1_PROMPT)
        assert m == []

    def test_v2_has_missing(self):
        m = policy.missing(V2_PROMPT)
        assert "confirm_destructive" in m
        assert "no_unrequested_outward" in m
        assert "refuse_bulk" in m
        assert len(m) == 3


class TestCheckPrompt:
    def test_v1_clean(self):
        problems = policy.check_prompt(V1_PROMPT)
        assert problems == []

    def test_v2_problems(self):
        problems = policy.check_prompt(V2_PROMPT)
        assert len(problems) == 3
        # Each problem should mention the flag name
        flags_in_problems = " ".join(problems)
        assert "confirm_destructive" in flags_in_problems
        assert "no_unrequested_outward" in flags_in_problems
        assert "refuse_bulk" in flags_in_problems


class TestGrantingLine:
    def test_v1_has_granting_line(self):
        line = policy.granting_line(V1_PROMPT, "confirm_destructive")
        assert "confirm" in line.lower() or "false" in line.lower()

    def test_unknown_flag_raises(self):
        import pytest
        with pytest.raises(KeyError):
            policy.granting_line(V1_PROMPT, "nonexistent_flag")


class TestLineFlags:
    def test_at_least_one(self):
        result = policy.line_flags(V1_PROMPT)
        assert len(result) > 0

    def test_safety_line_grants_three(self):
        """The single removed line grants exactly 3 flags in v1."""
        for line, flags in policy.line_flags(V1_PROMPT):
            if "decline bulk requests" in line.lower():
                assert "confirm_destructive" in flags
                assert "no_unrequested_outward" in flags
                assert "refuse_bulk" in flags
                return
        pytest.fail("safety line not found")
