"""Tests for driftbench.tools"""

from __future__ import annotations

from driftbench import tools
from driftbench.world import build_world, Fault


class TestRegistry:
    def test_has_tools(self):
        names = tools.names()
        assert len(names) >= 10
        assert "search_docs" in names
        assert "read_doc" in names
        assert "read_file" in names
        assert "write_file" in names
        assert "delete_file" in names
        assert "get_ticket" in names
        assert "issue_refund" in names
        assert "send_email" in names
        assert "calculator" in names
        assert "http_get" in names
        assert "escalate_to_human" in names

    def test_tool_spec(self):
        spec = tools.get("search_docs")
        assert spec is not None
        assert spec.name == "search_docs"
        assert "query" in spec.properties
        assert "query" in spec.required

    def test_destructive_names(self):
        d = tools.destructive_names()
        assert "write_file" in d
        assert "delete_file" in d
        assert "issue_refund" in d
        assert "send_email" in d

    def test_outward_names(self):
        o = tools.outward_names()
        assert "send_email" in o
        assert "issue_refund" in o


class TestInputSchema:
    def test_has_required_fields(self):
        spec = tools.get("search_docs")
        schema = spec.input_schema()
        assert schema["type"] == "object"
        assert "query" in schema["required"]
        assert "query" in schema["properties"]


class TestValidateArgs:
    def test_valid(self):
        spec = tools.get("search_docs")
        err = tools.validate_args(spec, {"query": "test"})
        assert err is None

    def test_missing_required(self):
        spec = tools.get("search_docs")
        err = tools.validate_args(spec, {})
        assert err is not None
        assert "query" in err

    def test_unknown_arg(self):
        spec = tools.get("search_docs")
        err = tools.validate_args(spec, {"query": "test", "extra": True})
        assert err is not None
        assert "extra" in err

    def test_wrong_type(self):
        spec = tools.get("search_docs")
        err = tools.validate_args(spec, {"query": 123})
        assert err is not None
        assert "string" in err

    def test_not_dict(self):
        spec = tools.get("search_docs")
        err = tools.validate_args(spec, "not a dict")
        assert err is not None
        assert "object" in err


class TestCallTool:
    def test_unknown_tool(self):
        world = build_world(seed=42)
        result, is_error = tools.call_tool(world, "nonexistent_tool", {})
        assert is_error is True
        assert "no such tool" in result

    def test_search_docs(self):
        world = build_world(seed=42)
        result, is_error = tools.call_tool(world, "search_docs",
                                           {"query": "pricing"})
        assert is_error is False
        assert "results" in result

    def test_read_doc(self):
        world = build_world(seed=42)
        result, is_error = tools.call_tool(world, "read_doc",
                                           {"doc_id": "doc-pricing"})
        assert is_error is False
        assert result["title"] == "Pricing tiers"

    def test_read_doc_not_found(self):
        world = build_world(seed=42)
        result, is_error = tools.call_tool(world, "read_doc",
                                           {"doc_id": "nonexistent"})
        assert is_error is True
        assert "no document" in result

    def test_list_files(self):
        world = build_world(seed=42)
        result, is_error = tools.call_tool(world, "list_files", {"path": ""})
        assert is_error is False
        assert "files" in result
        assert len(result["files"]) > 0

    def test_read_file(self):
        world = build_world(seed=42)
        result, is_error = tools.call_tool(world, "read_file",
                                           {"path": "README.md"})
        assert is_error is False
        assert "Example service" in result["content"]

    def test_write_file(self):
        world = build_world(seed=42)
        result, is_error = tools.call_tool(world, "write_file",
                                           {"path": "new.txt",
                                            "content": "hello"})
        assert is_error is False
        assert result["bytes"] == 5
        assert "new.txt" in world.written_files

    def test_delete_file_preview(self):
        world = build_world(seed=42)
        result, is_error = tools.call_tool(world, "delete_file",
                                           {"path": "README.md",
                                            "confirm": False})
        assert is_error is False
        assert result["deleted"] is False

    def test_delete_file_commit(self):
        world = build_world(seed=42)
        result, is_error = tools.call_tool(world, "delete_file",
                                           {"path": "README.md",
                                            "confirm": True})
        assert is_error is False
        assert result["deleted"] is True
        assert "README.md" in world.deleted_files

    def test_get_ticket(self):
        world = build_world(seed=42)
        result, is_error = tools.call_tool(world, "get_ticket",
                                           {"ticket_id": "T-1001"})
        assert is_error is False
        assert result["subject"] == "Checkout fails with 500"

    def test_get_ticket_not_found(self):
        world = build_world(seed=42)
        result, is_error = tools.call_tool(world, "get_ticket",
                                           {"ticket_id": "T-9999"})
        assert is_error is True

    def test_list_tickets(self):
        world = build_world(seed=42)
        result, is_error = tools.call_tool(world, "list_tickets",
                                           {"status": "open"})
        assert is_error is False
        assert len(result["tickets"]) == 2

    def test_update_ticket(self):
        world = build_world(seed=42)
        result, is_error = tools.call_tool(world, "update_ticket",
                                           {"ticket_id": "T-1001",
                                            "status": "closed",
                                            "comment": "Fixed"})
        assert is_error is False
        assert result["updated"] is True
        assert len(world.ticket_updates) == 1

    def test_get_order(self):
        world = build_world(seed=42)
        result, is_error = tools.call_tool(world, "get_order",
                                           {"order_id": "ORD-77"})
        assert is_error is False
        assert result["total_cents"] == 89000

    def test_issue_refund_preview(self):
        world = build_world(seed=42)
        result, is_error = tools.call_tool(world, "issue_refund",
                                           {"order_id": "ORD-12",
                                            "amount_cents": 100,
                                            "confirm": False})
        assert is_error is False
        assert result["refunded"] is False

    def test_issue_refund_commit(self):
        world = build_world(seed=42)
        result, is_error = tools.call_tool(world, "issue_refund",
                                           {"order_id": "ORD-12",
                                            "amount_cents": 100,
                                            "confirm": True})
        assert is_error is False
        assert result["refunded"] is True
        assert len(world.refunds) == 1

    def test_send_email(self):
        world = build_world(seed=42)
        result, is_error = tools.call_tool(world, "send_email",
                                           {"to": "a@b.com",
                                            "subject": "Hi",
                                            "body": "Hello"})
        assert is_error is False
        assert result["sent"] is True
        assert len(world.sent_emails) == 1

    def test_send_email_invalid_address(self):
        world = build_world(seed=42)
        result, is_error = tools.call_tool(world, "send_email",
                                           {"to": "not-an-email",
                                            "subject": "Hi",
                                            "body": "Hello"})
        assert is_error is True

    def test_calculator(self):
        world = build_world(seed=42)
        result, is_error = tools.call_tool(world, "calculator",
                                           {"expression": "(2900 + 890) * 0.2"})
        assert is_error is False
        assert abs(result["value"] - 758.0) < 0.001

    def test_calculator_division_by_zero(self):
        world = build_world(seed=42)
        result, is_error = tools.call_tool(world, "calculator",
                                           {"expression": "1/0"})
        assert is_error is True

    def test_http_get(self):
        world = build_world(seed=42)
        result, is_error = tools.call_tool(world, "http_get",
                                           {"url": "https://status.example.com/api"})
        assert is_error is False
        assert result["status"] == 200

    def test_http_get_not_found(self):
        world = build_world(seed=42)
        result, is_error = tools.call_tool(world, "http_get",
                                           {"url": "https://nonexistent.example.com"})
        assert is_error is True

    def test_escalate_to_human(self):
        world = build_world(seed=42)
        result, is_error = tools.call_tool(world, "escalate_to_human",
                                           {"reason": "need help"})
        assert is_error is False
        assert result["escalated"] is True
        assert len(world.escalations) == 1


class TestFaultInjection:
    def test_fail_first_n(self):
        world = build_world(seed=42, faults={
            "search_docs": Fault(fail_first_n=2, message="index rebuilding"),
        })
        # First two calls fail
        _, err1 = tools.call_tool(world, "search_docs", {"query": "a"})
        assert err1 is True
        _, err2 = tools.call_tool(world, "search_docs", {"query": "b"})
        assert err2 is True
        # Third call succeeds
        _, err3 = tools.call_tool(world, "search_docs", {"query": "c"})
        assert err3 is False

    def test_permanent_fault(self):
        world = build_world(seed=42, faults={
            "http_get": Fault(permanent=True, message="502 bad gateway"),
        })
        for _ in range(5):
            _, err = tools.call_tool(world, "http_get",
                                     {"url": "https://example.com"})
            assert err is True
