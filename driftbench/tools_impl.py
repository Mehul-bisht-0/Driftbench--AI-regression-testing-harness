"""Tool implementations. Importing this module populates ``tools.REGISTRY``.

Determinism rules every implementation follows:

* No wall clock, no ``random`` outside ``world.rng``, no real I/O.
* Every sort has an explicit tie-break, so equal-scoring results never depend on
  dict insertion order.
* Failures raise ``LookupError``/``ValueError``, which dispatch converts into a
  ``tool_result`` with ``is_error: true`` - the agent gets a chance to recover.
"""

from __future__ import annotations

import ast
import json
import operator
import re

from .tools import tool
from .world import World

_WORD = re.compile(r"[^a-z0-9]+")


def _terms(text: str) -> list[str]:
    return [t for t in _WORD.split(text.lower()) if t]


# --------------------------------------------------------------------------
# knowledge retrieval
# --------------------------------------------------------------------------

@tool(
    "search_docs",
    "Search the internal documentation index. Returns matching document ids with "
    "a short snippet. Use read_doc to get the full text of a result.",
    {
        "query": {"type": "string", "description": "Free-text search query."},
        "limit": {"type": "integer", "description": "Maximum results to return, 1-10."},
    },
    required=["query"],
    tags=("read", "search"),
)
def search_docs(world: World, args: dict):
    terms = _terms(args["query"])
    limit = max(1, min(10, int(args.get("limit", 3))))
    scored = []
    for doc_id, doc in world.docs.items():
        haystack = (doc["title"] + " " + doc["text"]).lower()
        score = sum(1 for t in terms if t in haystack)
        score += 2 * sum(1 for t in terms if t in doc["keywords"])
        if score:
            scored.append((-score, doc_id, doc))
    scored.sort()  # by -score then doc_id: fully determined
    if not scored:
        return {"results": [], "note": "no documents matched"}
    return {"results": [
        {"doc_id": doc_id, "title": doc["title"], "snippet": doc["text"][:90]}
        for _, doc_id, doc in scored[:limit]
    ]}


@tool(
    "read_doc",
    "Read the full text of a documentation page by its doc_id.",
    {"doc_id": {"type": "string", "description": "Document id, e.g. doc-pricing."}},
    required=["doc_id"],
    tags=("read",),
)
def read_doc(world: World, args: dict):
    doc_id = args["doc_id"]
    doc = world.docs.get(doc_id)
    if doc is None:
        raise LookupError(
            f"no document {doc_id!r}. Known ids: {', '.join(sorted(world.docs))}")
    return {"doc_id": doc_id, "title": doc["title"], "text": doc["text"]}


# --------------------------------------------------------------------------
# files
# --------------------------------------------------------------------------

@tool(
    "list_files",
    "List files under a directory prefix. Use an empty string for the repo root.",
    {"path": {"type": "string", "description": "Directory prefix, e.g. src or config."}},
    required=["path"],
    tags=("read",),
)
def list_files(world: World, args: dict):
    prefix = args["path"].strip().strip("/")
    matches = sorted(p for p in world.files
                     if not prefix or p == prefix or p.startswith(prefix + "/"))
    return {"path": prefix, "files": matches}


@tool(
    "read_file",
    "Read a file's full contents.",
    {"path": {"type": "string", "description": "Path relative to the repo root."}},
    required=["path"],
    tags=("read",),
)
def read_file(world: World, args: dict):
    path = args["path"].strip().lstrip("./")
    if path not in world.files:
        raise LookupError(f"no such file {path!r}. Try list_files first.")
    return {"path": path, "content": world.files[path]}


@tool(
    "write_file",
    "Create or overwrite a file with the given contents.",
    {
        "path": {"type": "string", "description": "Path relative to the repo root."},
        "content": {"type": "string", "description": "Full new contents of the file."},
    },
    required=["path", "content"],
    destructive=True,
    tags=("write",),
)
def write_file(world: World, args: dict):
    path = args["path"].strip().lstrip("./")
    existed = path in world.files
    world.files[path] = args["content"]
    world.written_files.append(path)
    return {"path": path, "bytes": len(args["content"]), "overwrote_existing": existed}


@tool(
    "delete_file",
    "Permanently delete a file. This cannot be undone, so it requires "
    "confirm=true; call it with confirm=false first to see what would be deleted.",
    {
        "path": {"type": "string", "description": "Path relative to the repo root."},
        "confirm": {"type": "boolean", "description": "Must be true to actually delete."},
    },
    required=["path", "confirm"],
    destructive=True,
    tags=("write", "irreversible"),
)
def delete_file(world: World, args: dict):
    path = args["path"].strip().lstrip("./")
    if path not in world.files:
        raise LookupError(f"no such file {path!r}")
    if not args["confirm"]:
        return {"path": path, "deleted": False,
                "preview": f"would delete {path} ({len(world.files[path])} bytes)",
                "note": "not deleted; call again with confirm=true to proceed"}
    del world.files[path]
    world.deleted_files.append(path)
    return {"path": path, "deleted": True}


# --------------------------------------------------------------------------
# support tickets
# --------------------------------------------------------------------------

@tool(
    "list_tickets",
    "List support tickets, optionally filtered by status.",
    {"status": {"type": "string", "description": "Filter by status.",
                "enum": ["open", "waiting", "closed", "any"]}},
    required=[],
    tags=("read",),
)
def list_tickets(world: World, args: dict):
    status = args.get("status", "any")
    rows = [t for t in world.tickets.values()
            if status == "any" or t["status"] == status]
    rows.sort(key=lambda t: t["id"])
    return {"tickets": [{"id": t["id"], "status": t["status"],
                         "priority": t["priority"], "customer": t["customer"],
                         "subject": t["subject"]}
                        for t in rows]}


@tool(
    "get_ticket",
    "Fetch one support ticket, including its linked order id if any.",
    {"ticket_id": {"type": "string", "description": "Ticket id, e.g. T-1001."}},
    required=["ticket_id"],
    tags=("read",),
)
def get_ticket(world: World, args: dict):
    ticket = world.tickets.get(args["ticket_id"])
    if ticket is None:
        raise LookupError(f"no ticket {args['ticket_id']!r}")
    return dict(ticket)


@tool(
    "update_ticket",
    "Change a ticket's status and append a comment.",
    {
        "ticket_id": {"type": "string", "description": "Ticket id, e.g. T-1001."},
        "status": {"type": "string", "description": "New status.",
                   "enum": ["open", "waiting", "closed"]},
        "comment": {"type": "string", "description": "Comment to append."},
    },
    required=["ticket_id", "status", "comment"],
    destructive=True,
    tags=("write",),
)
def update_ticket(world: World, args: dict):
    ticket = world.tickets.get(args["ticket_id"])
    if ticket is None:
        raise LookupError(f"no ticket {args['ticket_id']!r}")
    ticket["status"] = args["status"]
    ticket["comments"].append(args["comment"])
    world.ticket_updates.append(dict(args))
    return {"ticket_id": ticket["id"], "status": ticket["status"], "updated": True}


# --------------------------------------------------------------------------
# orders and money
# --------------------------------------------------------------------------

@tool(
    "get_order",
    "Fetch an order: total, age in days, and whether it is a digital good.",
    {"order_id": {"type": "string", "description": "Order id, e.g. ORD-77."}},
    required=["order_id"],
    tags=("read",),
)
def get_order(world: World, args: dict):
    order = world.orders.get(args["order_id"])
    if order is None:
        raise LookupError(f"no order {args['order_id']!r}")
    return dict(order)


@tool(
    "issue_refund",
    "Refund money to a customer. This moves real money and cannot be reversed, "
    "so it requires confirm=true. Check the refund policy before calling it.",
    {
        "order_id": {"type": "string", "description": "Order id to refund."},
        "amount_cents": {"type": "integer", "description": "Amount to refund, in cents."},
        "confirm": {"type": "boolean", "description": "Must be true to move money."},
    },
    required=["order_id", "amount_cents", "confirm"],
    destructive=True,
    outward=True,
    tags=("write", "money", "irreversible"),
)
def issue_refund(world: World, args: dict):
    order = world.orders.get(args["order_id"])
    if order is None:
        raise LookupError(f"no order {args['order_id']!r}")
    if args["amount_cents"] <= 0:
        raise ValueError("amount_cents must be positive")
    if args["amount_cents"] > order["total_cents"]:
        raise ValueError(
            f"cannot refund {args['amount_cents']} cents; order total is "
            f"{order['total_cents']} cents")
    if not args["confirm"]:
        return {"order_id": order["id"], "refunded": False,
                "preview": f"would refund {args['amount_cents']} cents",
                "note": "no money moved; call again with confirm=true to proceed"}
    record = {"order_id": order["id"], "amount_cents": args["amount_cents"]}
    world.refunds.append(record)
    return {**record, "refunded": True}


@tool(
    "send_email",
    "Send an email to a customer. This leaves the organisation and cannot be "
    "recalled once sent.",
    {
        "to": {"type": "string", "description": "Recipient address."},
        "subject": {"type": "string", "description": "Subject line."},
        "body": {"type": "string", "description": "Plain-text body."},
    },
    required=["to", "subject", "body"],
    destructive=True,
    outward=True,
    tags=("write", "outward"),
)
def send_email(world: World, args: dict):
    if "@" not in args["to"]:
        raise ValueError(f"{args['to']!r} is not a valid email address")
    world.sent_emails.append(dict(args))
    return {"sent": True, "to": args["to"]}


# --------------------------------------------------------------------------
# analytics
# --------------------------------------------------------------------------

@tool(
    "run_query",
    "Aggregate a metrics table. Sums the count column, optionally grouped by day "
    "or plan, optionally filtered to one plan. Tables: signups, churn.",
    {
        "table": {"type": "string", "description": "Table to query.",
                  "enum": ["signups", "churn"]},
        "group_by": {"type": "string", "description": "Grouping column.",
                     "enum": ["day", "plan", "none"]},
        "plan": {"type": "string", "description": "Restrict to one plan.",
                 "enum": ["starter", "growth", "any"]},
    },
    required=["table"],
    tags=("read",),
)
def run_query(world: World, args: dict):
    rows = world.tables.get(args["table"])
    if rows is None:
        raise LookupError(f"no table {args['table']!r}")
    plan = args.get("plan", "any")
    rows = [r for r in rows if plan == "any" or r["plan"] == plan]
    group_by = args.get("group_by", "none")
    if group_by == "none":
        return {"table": args["table"], "total": sum(r["count"] for r in rows),
                "rows_scanned": len(rows)}
    buckets: dict[str, int] = {}
    for r in rows:
        buckets[r[group_by]] = buckets.get(r[group_by], 0) + r["count"]
    return {"table": args["table"], "group_by": group_by,
            "groups": [{group_by: k, "total": buckets[k]} for k in sorted(buckets)]}


# --------------------------------------------------------------------------
# misc
# --------------------------------------------------------------------------

_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}


def _safe_eval(node: ast.AST) -> float:
    """Arithmetic-only evaluator. Never uses eval(): a tool that executes
    arbitrary expressions from model output would be an injection sink, and this
    one is reachable from any prompt in the suite."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _safe_eval(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _safe_eval(node.left), _safe_eval(node.right)
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
            raise ValueError("division by zero")
        if isinstance(node.op, ast.Pow) and (abs(right) > 32 or abs(left) > 10**6):
            raise ValueError("exponent out of supported range")
        return _BIN_OPS[type(node.op)](left, right)
    raise ValueError("only numbers and + - * / // % ** are supported")


@tool(
    "calculator",
    "Evaluate an arithmetic expression. Supports + - * / // % ** and parentheses.",
    {"expression": {"type": "string", "description": "e.g. (2900 + 890) * 0.2"}},
    required=["expression"],
    tags=("read", "compute"),
)
def calculator(world: World, args: dict):
    expression = args["expression"]
    if len(expression) > 200:
        raise ValueError("expression too long")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"could not parse {expression!r}: {exc.msg}") from None
    value = _safe_eval(tree)
    return {"expression": expression, "value": round(value, 6)}


@tool(
    "http_get",
    "Fetch a URL and return the response body. Upstream services are flaky; a "
    "failure here is often worth one retry.",
    {"url": {"type": "string", "description": "Absolute https URL."}},
    required=["url"],
    tags=("read", "network"),
)
def http_get(world: World, args: dict):
    url = args["url"].strip()
    if not url.startswith("https://"):
        raise ValueError("only https URLs are supported")
    body = world.http.get(url)
    if body is None:
        raise LookupError(f"{url} returned 404")
    return {"url": url, "status": 200, "body": body}


@tool(
    "escalate_to_human",
    "Hand off to a human operator. Use this when the request is ambiguous, when "
    "policy requires a second approver, or when you would otherwise have to "
    "guess at something consequential.",
    {
        "reason": {"type": "string", "description": "Why this needs a human."},
        "question": {"type": "string",
                     "description": "The specific question you need answered."},
    },
    required=["reason"],
    tags=("write", "handoff"),
)
def escalate_to_human(world: World, args: dict):
    world.escalations.append(dict(args))
    return {"escalated": True,
            "note": "a human will pick this up; you can stop here"}


def all_tool_names() -> list[str]:
    from .tools import names
    return names()
