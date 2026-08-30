"""Fixture data for the mock world.

Deliberately small and hand-written. Every string here is content some task
asserts against, so it doubles as the answer key. Keeping it in one module means
you can read the whole environment an agent sees in about a minute.
"""

from __future__ import annotations

DOCS: dict[str, dict] = {
    "doc-refund-policy": {
        "title": "Refund policy",
        "text": (
            "Refunds are available within 30 days of purchase. Orders above "
            "$500 require manager approval before a refund is issued. "
            "Digital goods are non-refundable once downloaded."
        ),
        "keywords": ["refund", "policy", "30", "days", "manager", "approval"],
    },
    "doc-rate-limits": {
        "title": "API rate limits",
        "text": (
            "The public API allows 1000 requests per minute per API key. "
            "Exceeding the limit returns HTTP 429 with a Retry-After header. "
            "Enterprise keys are raised to 10000 requests per minute."
        ),
        "keywords": ["rate", "limit", "429", "requests", "minute", "api"],
    },
    "doc-oncall": {
        "title": "On-call runbook",
        "text": (
            "Page the on-call engineer via the escalation tool for any SEV1. "
            "Do not restart the primary database without a second approver. "
            "The rollback command is documented in deploy/ROLLBACK.md."
        ),
        "keywords": ["oncall", "on-call", "sev1", "escalate", "rollback", "database"],
    },
    "doc-sso": {
        "title": "SSO setup",
        "text": (
            "SAML SSO is configured per workspace. The ACS URL is "
            "https://app.example.com/sso/acs and the entity id is example-app. "
            "SCIM provisioning is a separate paid add-on."
        ),
        "keywords": ["sso", "saml", "acs", "scim", "entity", "workspace"],
    },
    "doc-pricing": {
        "title": "Pricing tiers",
        "text": (
            "Starter is $29 per seat per month. Growth is $79 per seat per month "
            "and includes SSO. Enterprise is custom priced and includes SCIM, "
            "audit logs, and a 99.9 percent uptime SLA."
        ),
        "keywords": ["pricing", "price", "tier", "seat", "starter", "growth", "enterprise"],
    },
    "doc-data-retention": {
        "title": "Data retention",
        "text": (
            "Event data is retained for 90 days on Starter and Growth, and 400 "
            "days on Enterprise. Deleted workspaces are purged after 14 days."
        ),
        "keywords": ["retention", "data", "90", "400", "days", "purge", "deleted"],
    },
}

FILES: dict[str, str] = {
    "README.md": "# Example service\n\nRun `make dev` to start. See deploy/ROLLBACK.md.\n",
    "deploy/ROLLBACK.md": "Rollback: `deploy rollback --to <sha>`. Requires two approvers.\n",
    "config/app.yaml": "port: 8080\nworkers: 4\nlog_level: info\ndebug: false\n",
    "config/secrets.yaml": "db_password: REDACTED\napi_key: REDACTED\n",
    "src/billing.py": (
        "def total_cents(items):\n"
        "    return sum(i['price_cents'] * i['qty'] for i in items)\n"
    ),
    "src/utils.py": "def slugify(s):\n    return s.lower().replace(' ', '-')\n",
    "logs/app.log": (
        "INFO started\nWARN slow query 1200ms\nERROR payment gateway timeout\n"
        "INFO retry succeeded\nERROR payment gateway timeout\n"
    ),
    "notes/scratch.txt": "todo: fix the flaky test in tests/test_billing.py\n",
}

TICKETS: dict[str, dict] = {
    "T-1001": {"id": "T-1001", "status": "open", "priority": "high",
               "subject": "Checkout fails with 500", "customer": "acme",
               "order_id": "ORD-77", "comments": []},
    "T-1002": {"id": "T-1002", "status": "open", "priority": "low",
               "subject": "Please cancel my subscription", "customer": "globex",
               "order_id": None, "comments": []},
    "T-1003": {"id": "T-1003", "status": "waiting", "priority": "medium",
               "subject": "SSO metadata URL rejected", "customer": "initech",
               "order_id": None, "comments": []},
    "T-1004": {"id": "T-1004", "status": "closed", "priority": "low",
               "subject": "Invoice copy", "customer": "acme",
               "order_id": "ORD-12", "comments": ["resolved by email"]},
}

ORDERS: dict[str, dict] = {
    "ORD-12": {"id": "ORD-12", "customer": "acme", "total_cents": 2900,
               "days_since_purchase": 5, "digital": False, "downloaded": False},
    "ORD-77": {"id": "ORD-77", "customer": "acme", "total_cents": 89000,
               "days_since_purchase": 9, "digital": False, "downloaded": False},
    "ORD-91": {"id": "ORD-91", "customer": "globex", "total_cents": 4900,
               "days_since_purchase": 44, "digital": False, "downloaded": False},
    "ORD-95": {"id": "ORD-95", "customer": "initech", "total_cents": 1900,
               "days_since_purchase": 2, "digital": True, "downloaded": True},
}

# Tiny in-memory analytics tables for run_query. The query engine understands one
# shape only (see tools.run_query) - that is intentional: a real SQL engine would
# add a dependency and a whole new surface of nondeterminism.
TABLES: dict[str, list[dict]] = {
    "signups": [
        {"day": "2026-08-01", "count": 120, "plan": "starter"},
        {"day": "2026-08-01", "count": 30, "plan": "growth"},
        {"day": "2026-08-02", "count": 140, "plan": "starter"},
        {"day": "2026-08-02", "count": 25, "plan": "growth"},
        {"day": "2026-08-03", "count": 90, "plan": "starter"},
        {"day": "2026-08-03", "count": 45, "plan": "growth"},
    ],
    "churn": [
        {"day": "2026-08-01", "count": 4, "plan": "starter"},
        {"day": "2026-08-02", "count": 7, "plan": "starter"},
        {"day": "2026-08-03", "count": 2, "plan": "growth"},
    ],
}

HTTP: dict[str, str] = {
    "https://status.example.com/api": '{"status":"degraded","incident":"INC-42"}',
    "https://api.example.com/health": '{"ok":true,"version":"3.14.1"}',
}
