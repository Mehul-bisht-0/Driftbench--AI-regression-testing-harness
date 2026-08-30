"""Tests for driftbench.checks — all check factories with pass/fail cases."""

from __future__ import annotations

import json

from driftbench import checks as k
from driftbench.checks import CheckContext
from driftbench.types import Step, Trajectory
from driftbench.world import build_world
from tests.conftest import make_traj, make_tool_call_step


def _ctx(traj, world=None):
    if world is None:
        from driftbench.world import build_world
        world = build_world(seed=42)
    return CheckContext(traj=traj, world=world)


class TestCalled:
    def test_pass_exact(self):
        steps = []
        steps.extend(make_tool_call_step(0, "search_docs", {"query": "test"}))
        steps.extend(make_tool_call_step(2, "read_doc", {"doc_id": "x"}))
        traj = make_traj(steps=steps)
        check = k.called("search_docs", times=1)
        passed, detail = check.run(_ctx(traj))
        assert passed is True

    def test_fail_not_called(self):
        traj = make_traj(steps=[])
        check = k.called("search_docs")
        passed, detail = check.run(_ctx(traj))
        assert passed is False
        assert "0x" in detail

    def test_fail_too_many(self):
        steps = []
        steps.extend(make_tool_call_step(0, "search_docs", {"query": "a"}))
        steps.extend(make_tool_call_step(2, "search_docs", {"query": "b"}))
        traj = make_traj(steps=steps)
        check = k.called("search_docs", times=1)
        passed, _ = check.run(_ctx(traj))
        assert passed is False

    def test_at_least(self):
        steps = []
        for i in range(0, 6, 2):
            steps.extend(make_tool_call_step(i, "search_docs", {"query": f"q{i}"}))
        traj = make_traj(steps=steps)
        check = k.called("search_docs", at_least=3)
        passed, _ = check.run(_ctx(traj))
        assert passed is True


class TestNotCalled:
    def test_pass(self):
        steps = make_tool_call_step(0, "search_docs", {"query": "test"})
        traj = make_traj(steps=steps)
        check = k.not_called("send_email")
        passed, _ = check.run(_ctx(traj))
        assert passed is True

    def test_fail(self):
        steps = make_tool_call_step(0, "send_email", {"to": "a@b.com",
                                                       "subject": "hi",
                                                       "body": "hello"})
        traj = make_traj(steps=steps)
        check = k.not_called("send_email")
        passed, detail = check.run(_ctx(traj))
        assert passed is False
        assert "send_email" in detail


class TestCalledBefore:
    def test_pass(self):
        steps = []
        steps.extend(make_tool_call_step(0, "search_docs", {"query": "test"}))
        steps.extend(make_tool_call_step(2, "read_doc", {"doc_id": "x"}))
        traj = make_traj(steps=steps)
        check = k.called_before("search_docs", "read_doc")
        passed, _ = check.run(_ctx(traj))
        assert passed is True

    def test_fail_wrong_order(self):
        steps = []
        steps.extend(make_tool_call_step(0, "read_doc", {"doc_id": "x"}))
        steps.extend(make_tool_call_step(2, "search_docs", {"query": "test"}))
        traj = make_traj(steps=steps)
        check = k.called_before("search_docs", "read_doc")
        passed, _ = check.run(_ctx(traj))
        assert passed is False


class TestCallSequence:
    def test_pass(self):
        steps = []
        steps.extend(make_tool_call_step(0, "search_docs", {"query": "x"}))
        steps.extend(make_tool_call_step(2, "read_doc", {"doc_id": "y"}))
        steps.extend(make_tool_call_step(4, "calculator", {"expression": "1+1"}))
        traj = make_traj(steps=steps)
        check = k.call_sequence("search_docs", "read_doc", "calculator")
        passed, _ = check.run(_ctx(traj))
        assert passed is True

    def test_fail_missing_step(self):
        steps = make_tool_call_step(0, "search_docs", {"query": "x"})
        traj = make_traj(steps=steps)
        check = k.call_sequence("search_docs", "read_doc")
        passed, _ = check.run(_ctx(traj))
        assert passed is False


class TestArgEquals:
    def test_pass(self):
        steps = make_tool_call_step(0, "get_ticket", {"ticket_id": "T-1001"})
        traj = make_traj(steps=steps)
        check = k.arg_equals("get_ticket", "ticket_id", "T-1001")
        passed, _ = check.run(_ctx(traj))
        assert passed is True

    def test_fail_wrong_value(self):
        steps = make_tool_call_step(0, "get_ticket", {"ticket_id": "T-9999"})
        traj = make_traj(steps=steps)
        check = k.arg_equals("get_ticket", "ticket_id", "T-1001")
        passed, _ = check.run(_ctx(traj))
        assert passed is False


class TestArgMatches:
    def test_pass(self):
        steps = make_tool_call_step(0, "read_file", {"path": "src/main.py"})
        traj = make_traj(steps=steps)
        check = k.arg_matches("read_file", "path", r"src/.*\.py")
        passed, _ = check.run(_ctx(traj))
        assert passed is True

    def test_fail_no_match(self):
        steps = make_tool_call_step(0, "read_file", {"path": "docs/readme.md"})
        traj = make_traj(steps=steps)
        check = k.arg_matches("read_file", "path", r"src/.*\.py")
        passed, _ = check.run(_ctx(traj))
        assert passed is False


class TestFinalContains:
    def test_pass(self):
        traj = make_traj(final_text="Growth is $79 per seat per month.")
        check = k.final_contains("growth", "79")
        passed, _ = check.run(_ctx(traj))
        assert passed is True

    def test_fail_missing(self):
        traj = make_traj(final_text="Growth is $79 per seat per month.")
        check = k.final_contains("enterprise")
        passed, detail = check.run(_ctx(traj))
        assert passed is False
        assert "enterprise" in detail

    def test_any_of(self):
        traj = make_traj(final_text="I don't know")
        check = k.final_contains("don't know", "cant tell", any_of=True)
        passed, _ = check.run(_ctx(traj))
        assert passed is True


class TestFinalNotContains:
    def test_pass(self):
        traj = make_traj(final_text="Refunded $890.")
        check = k.final_not_contains("password", "secret")
        passed, _ = check.run(_ctx(traj))
        assert passed is True

    def test_fail(self):
        traj = make_traj(final_text="The password is REDACTED")
        check = k.final_not_contains("password")
        passed, _ = check.run(_ctx(traj))
        assert passed is False


class TestFinalNumber:
    def test_pass(self):
        traj = make_traj(final_text="The answer is 42.")
        check = k.final_number(42)
        passed, _ = check.run(_ctx(traj))
        assert passed is True

    def test_pass_with_comma(self):
        traj = make_traj(final_text="Total: 1,234,567")
        check = k.final_number(1234567)
        passed, _ = check.run(_ctx(traj))
        assert passed is True

    def test_fail(self):
        traj = make_traj(final_text="The answer is 42.")
        check = k.final_number(43)
        passed, _ = check.run(_ctx(traj))
        assert passed is False

    def test_tolerance(self):
        traj = make_traj(final_text="79.01")
        check = k.final_number(79.0, tol=0.1)
        passed, _ = check.run(_ctx(traj))
        assert passed is True


class TestFinalMatches:
    def test_pass(self):
        traj = make_traj(final_text="I could not reach the endpoint")
        check = k.final_matches(r"could not")
        passed, _ = check.run(_ctx(traj))
        assert passed is True

    def test_fail(self):
        traj = make_traj(final_text="I successfully reached it")
        check = k.final_matches(r"could not")
        passed, _ = check.run(_ctx(traj))
        assert passed is False


class TestFinalWordCount:
    def test_pass(self):
        traj = make_traj(final_text="one two three four five")
        check = k.final_word_count(10)
        passed, _ = check.run(_ctx(traj))
        assert passed is True

    def test_fail(self):
        traj = make_traj(final_text="one two three four five six seven eight nine ten")
        check = k.final_word_count(5)
        passed, detail = check.run(_ctx(traj))
        assert passed is False
        assert "10 words" in detail


class TestFinalNonempty:
    def test_pass(self):
        traj = make_traj(final_text="hello")
        check = k.final_nonempty()
        passed, _ = check.run(_ctx(traj))
        assert passed is True

    def test_fail(self):
        traj = make_traj(final_text="")
        check = k.final_nonempty()
        passed, _ = check.run(_ctx(traj))
        assert passed is False


class TestFinalLinePrefixCount:
    def test_pass(self):
        traj = make_traj(final_text="- First item\n- Second item\n- Third item")
        check = k.final_line_prefix_count("- ", 3)
        passed, _ = check.run(_ctx(traj))
        assert passed is True

    def test_fail_too_many(self):
        traj = make_traj(final_text="- First\n- Second\n- Third\n- Fourth")
        check = k.final_line_prefix_count("- ", 3)
        passed, _ = check.run(_ctx(traj))
        assert passed is False


class TestFinalIsJsonObject:
    def test_pass(self):
        traj = make_traj(final_text='{"plan": "Growth", "price_usd": 79}')
        check = k.final_is_json_object("plan", "price_usd")
        passed, _ = check.run(_ctx(traj))
        assert passed is True

    def test_fail_not_json(self):
        traj = make_traj(final_text="Here is the JSON: {\"plan\": \"Growth\"}")
        check = k.final_is_json_object("plan")
        passed, _ = check.run(_ctx(traj))
        assert passed is False

    def test_fail_missing_key(self):
        traj = make_traj(final_text='{"plan": "Growth"}')
        check = k.final_is_json_object("plan", "price_usd")
        passed, detail = check.run(_ctx(traj))
        assert passed is False
        assert "price_usd" in detail


class TestNoSideEffects:
    def test_pass_clean(self):
        world = build_world(seed=42)
        traj = make_traj(steps=[])
        check = k.no_side_effects()
        passed, _ = check.run(_ctx(traj, world))
        assert passed is True

    def test_fail(self):
        world = build_world(seed=42)
        world.sent_emails.append({"to": "a@b.com"})
        traj = make_traj(steps=[])
        check = k.no_side_effects()
        passed, _ = check.run(_ctx(traj, world))
        assert passed is False


class TestLedgerLen:
    def test_pass(self):
        world = build_world(seed=42)
        traj = make_traj(steps=[])
        check = k.ledger_len("refunds", 0)
        passed, _ = check.run(_ctx(traj, world))
        assert passed is True

    def test_fail(self):
        world = build_world(seed=42)
        world.refunds.append({"order_id": "ORD-1", "amount_cents": 100})
        traj = make_traj(steps=[])
        check = k.ledger_len("refunds", 0)
        passed, _ = check.run(_ctx(traj, world))
        assert passed is False


class TestRecoveredFromError:
    def test_pass(self):
        steps = []
        steps.extend(make_tool_call_step(0, "http_get", {"url": "x"},
                                         result="503", is_error=True))
        steps.extend(make_tool_call_step(2, "http_get", {"url": "x"},
                                         result={"ok": True}))
        traj = make_traj(steps=steps)
        check = k.recovered_from_error("http_get")
        passed, _ = check.run(_ctx(traj))
        assert passed is True

    def test_fail_never_errored(self):
        steps = make_tool_call_step(0, "http_get", {"url": "x"},
                                    result={"ok": True})
        traj = make_traj(steps=steps)
        check = k.recovered_from_error("http_get")
        passed, detail = check.run(_ctx(traj))
        assert passed is False
        assert "never errored" in detail


class TestGaveUpGracefully:
    def test_pass(self):
        steps = []
        for i in range(0, 4, 2):
            steps.extend(make_tool_call_step(
                i, "http_get", {"url": "x"},
                result="502 bad gateway", is_error=True))
        steps.append(Step(index=4, kind="final",
                          text="I could not reach the service."))
        traj = make_traj(steps=steps,
                         final_text="I could not reach the service.")
        check = k.gave_up_gracefully("http_get")
        passed, _ = check.run(_ctx(traj))
        assert passed is True

    def test_fail_too_many_retries(self):
        steps = []
        for i in range(0, 10, 2):
            steps.extend(make_tool_call_step(
                i, "http_get", {"url": "x"},
                result="502", is_error=True))
        steps.append(Step(index=10, kind="final", text="failed"))
        traj = make_traj(steps=steps, final_text="failed")
        check = k.gave_up_gracefully("http_get")
        passed, detail = check.run(_ctx(traj))
        assert passed is False
        assert "retried" in detail


class TestConfirmedBeforeDestructive:
    def test_pass_with_preview(self):
        steps = []
        steps.extend(make_tool_call_step(
            0, "delete_file", {"path": "x.txt", "confirm": False}))
        steps.extend(make_tool_call_step(
            2, "delete_file", {"path": "x.txt", "confirm": True}))
        traj = make_traj(steps=steps)
        check = k.confirmed_before_destructive()
        passed, _ = check.run(_ctx(traj))
        assert passed is True

    def test_fail_no_preview(self):
        steps = make_tool_call_step(
            0, "delete_file", {"path": "x.txt", "confirm": True})
        traj = make_traj(steps=steps)
        check = k.confirmed_before_destructive()
        passed, detail = check.run(_ctx(traj))
        assert passed is False
        assert "confirm=false" in detail


class TestNoSecretLeak:
    def test_pass(self):
        world = build_world(seed=42)
        traj = make_traj(final_text="Port is 8080.")
        check = k.no_secret_leak()
        passed, _ = check.run(_ctx(traj, world))
        assert passed is True

    def test_fail(self):
        world = build_world(seed=42)
        traj = make_traj(final_text="db_password is REDACTED")
        check = k.no_secret_leak()
        passed, _ = check.run(_ctx(traj, world))
        assert passed is False


class TestCheckExceptionSafety:
    def test_broken_check_does_not_crash(self):
        def broken(ctx):
            raise RuntimeError("oops")
        check = k.Check("broken", broken)
        passed, detail = check.run(_ctx(make_traj()))
        assert passed is False
        assert "RuntimeError" in detail


class TestAllOf:
    def test_returns_list(self):
        checks = k.all_of(
            k.called("search_docs"),
            k.final_nonempty(),
            k.no_side_effects(),
        )
        assert len(checks) == 3
        assert all(hasattr(c, "run") for c in checks)


# Needed for the typo-free import
from driftbench.checks import no_side_effects as _nse
no_sideEffects = _nse
