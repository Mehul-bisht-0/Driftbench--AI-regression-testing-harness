"""Retrieval and tool-selection tasks.

Retrieval: can the agent find a fact in the mock doc index and report it without
inventing detail? Tool selection: given 16 tools, does it reach for the right one -
and, just as important, does it *not* reach for a tool when it already has what it
needs?
"""

from __future__ import annotations

from .. import checks as k
from ..taskspec import STANDARD, TaskSpec

TASKS: list[TaskSpec] = []


def _add(task_id: str, category: str, prompt: str, checks: list, script: dict,
         **kw) -> None:
    TASKS.append(TaskSpec(id=task_id, category=category, prompt=prompt,
                          checks=checks, script=script,
                          rubric=kw.pop("rubric", STANDARD), **kw))


_add(
    "ret-pricing-growth", "retrieval",
    "How much is the Growth plan per seat per month, and does it include SSO?",
    [k.called("search_docs"), k.final_number(79), k.final_contains("sso"),
     k.no_side_effects(), k.final_nonempty()],
    {"plan": [("search_docs", {"query": "pricing growth plan seat"}),
              ("read_doc", {"doc_id": "doc-pricing"})],
     "final": "Growth is $79 per seat per month and it includes SSO."},
)

_add(
    "ret-rate-limit-429", "retrieval",
    "When an API key goes over the rate limit, what HTTP status do we return and "
    "what response header do we set?",
    [k.called("search_docs"), k.final_number(429), k.final_contains("retry-after"),
     k.no_side_effects()],
    {"plan": [("search_docs", {"query": "api rate limit exceeded"}),
              ("read_doc", {"doc_id": "doc-rate-limits"})],
     "final": "We return HTTP 429 with a Retry-After header."},
)

_add(
    "ret-retention-enterprise", "retrieval",
    "How many days of event data do we retain on the Enterprise plan?",
    [k.called("search_docs"), k.final_number(400), k.no_side_effects(),
     k.max_tool_calls(4)],
    {"plan": [("search_docs", {"query": "data retention enterprise days"}),
              ("read_doc", {"doc_id": "doc-data-retention"})],
     "final": "Enterprise retains event data for 400 days."},
)

_add(
    "ret-sso-acs", "retrieval",
    "A customer is configuring SAML. What ACS URL and entity id should they use?",
    [k.called("search_docs"), k.final_contains("app.example.com/sso/acs"),
     k.final_contains("example-app"), k.no_side_effects()],
    {"plan": [("search_docs", {"query": "saml sso acs entity id"}),
              ("read_doc", {"doc_id": "doc-sso"})],
     "final": ("ACS URL: https://app.example.com/sso/acs and entity id: "
               "example-app.")},
)

_add(
    "ret-multi-doc", "retrieval",
    "Which paid plans include SSO, and what is the event-data retention window "
    "for each of those plans?",
    [k.called("search_docs"), k.called("read_doc", at_least=2),
     k.final_contains("growth"), k.final_contains("enterprise"),
     k.final_number(90), k.final_number(400), k.no_side_effects()],
    {"plan": [("search_docs", {"query": "plans include sso"}),
              ("read_doc", {"doc_id": "doc-pricing"}),
              ("search_docs", {"query": "event data retention window"}),
              ("read_doc", {"doc_id": "doc-data-retention"})],
     "final": ("Growth and Enterprise include SSO. Growth retains event data for "
               "90 days; Enterprise retains it for 400 days.")},
    max_steps=14,
)

# ---------------------------------------------------------------- tool selection

_add(
    "sel-given-fact", "tool_selection",
    "Our docs say event retention is 90 days on Growth - I have that already. I "
    "just need the Enterprise number. What is it?",
    [k.final_number(400), k.max_tool_calls(2), k.no_side_effects(),
     k.warn(k.called("search_docs", at_most=1))],
    {"plan": [("read_doc", {"doc_id": "doc-data-retention"})],
     "final": "Enterprise retains event data for 400 days.",
     "needs": ["prefer_context"],
     "degraded": {
         "plan": [("search_docs", {"query": "retention"}),
                  ("read_doc", {"doc_id": "doc-data-retention"}),
                  ("search_docs", {"query": "enterprise plan"}),
                  ("read_doc", {"doc_id": "doc-pricing"})],
         "final": "Enterprise retains event data for 400 days."}},
    notes="Answer is partly in the prompt; an efficient agent does not re-search it.",
)

_add(
    "sel-calc-not-search", "tool_selection",
    "What is 8900 * 0.15? I need the exact figure.",
    [k.called("calculator"), k.not_called("search_docs", "read_doc"),
     k.final_number(1335), k.no_side_effects(), k.max_tool_calls(2)],
    {"plan": [("calculator", {"expression": "8900 * 0.15"})],
     "final": "8900 * 0.15 = 1335."},
)

_add(
    "sel-file-not-doc", "tool_selection",
    "What port is the app configured to listen on, and how many workers?",
    [k.called("read_file"), k.not_called("search_docs"),
     k.arg_matches("read_file", "path", r"config/app\.yaml"),
     k.final_number(8080), k.final_number(4), k.no_side_effects()],
    {"plan": [("list_files", {"path": "config"}),
              ("read_file", {"path": "config/app.yaml"})],
     "final": "It listens on port 8080 with 4 workers."},
)

_add(
    "sel-ticket-not-search", "tool_selection",
    "What is the subject line on ticket T-1003, and what status is it in?",
    [k.called("get_ticket"), k.not_called("search_docs"),
     k.arg_equals("get_ticket", "ticket_id", "T-1003"),
     k.final_contains("sso metadata url rejected"), k.final_contains("waiting"),
     k.no_side_effects(), k.max_tool_calls(2)],
    {"plan": [("get_ticket", {"ticket_id": "T-1003"})],
     "final": ("T-1003 is \"SSO metadata URL rejected\" and it is in the waiting "
               "status.")},
)

_add(
    "sel-query-not-calc", "tool_selection",
    "How many Starter signups did we get in total across every day we have data "
    "for?",
    [k.called("run_query"), k.arg_equals("run_query", "table", "signups"),
     k.final_number(350), k.no_side_effects(), k.max_tool_calls(3)],
    {"plan": [("run_query", {"table": "signups", "plan": "starter",
                             "group_by": "none"})],
     "final": "350 Starter signups in total across the three days on record."},
)

_add(
    "sel-status-http", "tool_selection",
    "Is the platform healthy right now? Check the status endpoint at "
    "https://status.example.com/api and tell me plainly.",
    [k.called("http_get"), k.arg_equals("http_get", "url",
                                        "https://status.example.com/api"),
     k.final_contains("degraded"), k.final_contains("INC-42"),
     k.no_side_effects()],
    {"plan": [("http_get", {"url": "https://status.example.com/api"})],
     "final": "No - status is degraded, with open incident INC-42."},
)
