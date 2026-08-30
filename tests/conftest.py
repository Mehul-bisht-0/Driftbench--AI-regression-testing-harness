"""Shared fixtures for driftbench tests."""

from __future__ import annotations

import pytest

from driftbench.types import Step, Trajectory, Usage
from driftbench.variant import Variant
from driftbench.world import World, build_world, Fault
from driftbench.seeding import SeededRandom


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------

@pytest.fixture
def v1_baseline():
    from pathlib import Path
    return Variant.load("prompts/v1_baseline.txt", variant_id="v1_baseline",
                        agent="scripted")


@pytest.fixture
def v2_ablated():
    from pathlib import Path
    return Variant.load("prompts/v2_ablated.txt", variant_id="v2_ablated",
                        agent="scripted")


# ---------------------------------------------------------------------------
# Worlds
# ---------------------------------------------------------------------------

@pytest.fixture
def world():
    return build_world(seed=42)


@pytest.fixture
def world_faulty():
    return build_world(seed=42, faults={
        "search_docs": Fault(fail_first_n=1, message="index rebuilding"),
        "http_get": Fault(flake_p=0.5, message="503 timeout"),
    })


# ---------------------------------------------------------------------------
# Trajectories
# ---------------------------------------------------------------------------

def make_traj(task_id="test-task", variant_id="v1", replicate=0, seed=123,
              steps=None, final_text="ok", stop_reason="end_turn",
              agent="scripted", model="test"):
    traj = Trajectory(task_id=task_id, variant_id=variant_id,
                      replicate=replicate, seed=seed, agent=agent, model=model)
    if steps:
        traj.steps = steps
    traj.final_text = final_text
    traj.stop_reason = stop_reason
    return traj


def make_tool_call_step(index, tool, args, result=None, is_error=False, latency_ms=50):
    call = Step(index=index, kind="tool_call", tool=tool, args=dict(args),
                latency_ms=latency_ms)
    res = Step(index=index + 1, kind="tool_result", tool=tool,
               result=result or f"{tool} ok", is_error=is_error,
               latency_ms=latency_ms)
    return [call, res]


@pytest.fixture
def simple_traj():
    steps = []
    idx = 0
    steps.extend(make_tool_call_step(idx, "search_docs", {"query": "pricing"},
                                     result={"results": [{"doc_id": "doc-pricing"}]}))
    idx += 2
    steps.extend(make_tool_call_step(idx, "read_doc", {"doc_id": "doc-pricing"},
                                     result={"title": "Pricing", "text": "Growth $79"}))
    idx += 2
    steps.append(Step(index=idx, kind="final", text="Growth is $79."))
    return make_traj(steps=steps, final_text="Growth is $79.")


@pytest.fixture
def multi_step_traj():
    steps = []
    idx = 0
    steps.extend(make_tool_call_step(idx, "get_ticket", {"ticket_id": "T-1001"},
                                     result={"id": "T-1001", "status": "open",
                                             "order_id": "ORD-77"}))
    idx += 2
    steps.extend(make_tool_call_step(idx, "get_order", {"order_id": "ORD-77"},
                                     result={"id": "ORD-77", "total_cents": 89000,
                                             "days_since_purchase": 9}))
    idx += 2
    steps.extend(make_tool_call_step(idx, "search_docs",
                                     {"query": "refund policy approval"}))
    idx += 2
    steps.extend(make_tool_call_step(idx, "read_doc",
                                     {"doc_id": "doc-refund-policy"}))
    idx += 2
    steps.extend(make_tool_call_step(idx, "escalate_to_human",
                                     {"reason": "over $500",
                                      "question": "Approve refund?"}))
    idx += 2
    steps.append(Step(index=idx, kind="final",
                      text="Needs manager approval."))
    return make_traj(steps=steps, final_text="Needs manager approval.")
