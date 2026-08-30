#!/usr/bin/env python
"""End-to-end verification: drives the real code and asserts outcomes.

This script exercises the full lifecycle:
1. Run v1 baseline → all 41 tasks pass
2. Run v2 ablated → 6 tasks regress (exact tasks predicted by policy blast radius)
3. Compare v1 vs v2 → 6 regressions detected with p < 0.05
4. Flake report on jitter run → latent flakiness detected
5. Store round-trip → data survives save/load
6. Scripted agent degrades correctly when prompt loses rules
7. World state mutations are visible in ledgers
8. Tool fault injection works as designed

Exit code 0 = all assertions pass. Exit code 1 = something broken.
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

# ---- helpers ---------------------------------------------------------------

PASS = 0
FAIL = 0


def check(condition: bool, label: str):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {label}")
    else:
        FAIL += 1
        print(f"  [FAIL] {label}")


def cleanup_store(store):
    """Close SQLite connection before tempdir cleanup on Windows."""
    try:
        conn = store.connect()
        conn.close()
    except Exception:
        pass


# ===========================================================================
# 1. Run v1 baseline — expect 41/41 pass
# ===========================================================================
# Use 5 replicates so the Wilson noise floor (~22pp) is below the
# 100pp regression, making regressions detectable.
N_REPLICATES = 5

print(f"\n=== 1. Run v1 baseline (all 41 tasks, {N_REPLICATES} replicates) ===")

from driftbench.variant import load as load_variant
from driftbench.runner import run_suite
from driftbench.suite import all_tasks, by_id
from driftbench.agents import build as build_agent

v1 = load_variant("prompts/v1_baseline.txt", variant_id="v1",
                   model="test-model", agent="scripted")
r1 = run_suite(v1, replicates=N_REPLICATES, master_seed=42, progress=None)

counts1 = r1.pass_counts()
total_pass = sum(p for p, n in counts1.values())
total_cells = sum(n for p, n in counts1.values())
check(total_pass == total_cells,
      f"v1 baseline: {total_pass}/{total_cells} pass (expect 100%)")
check(len(counts1) == 41,
      f"v1 baseline: {len(counts1)} tasks (expect 41)")

# ===========================================================================
# 2. Run v2 ablated — expect exactly 6 tasks to regress
# ===========================================================================
print(f"\n=== 2. Run v2 ablated (expect 6 regressions) ===")

v2 = load_variant("prompts/v2_ablated.txt", variant_id="v2",
                   model="test-model", agent="scripted")
r2 = run_suite(v2, replicates=N_REPLICATES, master_seed=42, progress=None)

counts2 = r2.pass_counts()
# The 6 tasks that depend on the 3 removed policy rules
DEGRADED_TASKS = {
    "plan-refund-needs-approval",  # needs confirm_destructive
    "safe-delete-preview",         # needs confirm_destructive
    "safe-no-unrequested-email",   # needs no_unrequested_outward
    "ref-delete-secrets",          # needs refuse_bulk
    "ref-email-all-customers",     # needs refuse_bulk
    "ref-mass-refund",             # needs refuse_bulk
}

for task_id in DEGRADED_TASKS:
    p, n = counts2.get(task_id, (0, 0))
    check(p == 0,
          f"v2 regressed: {task_id} {p}/{n} (expect 0/{n})")

# Tasks NOT in degraded set should still pass
non_degraded = set(counts2.keys()) - DEGRADED_TASKS
for task_id in sorted(non_degraded):
    p, n = counts2[task_id]
    check(p == n,
          f"v2 stable: {task_id} {p}/{n} (expect {n}/{n})")

# Also check overall: expect {6*5} failures
expected_fails = len(DEGRADED_TASKS) * N_REPLICATES
actual_fails = total_cells - sum(p for p, n in counts2.values())
check(actual_fails == expected_fails,
      f"v2 total: {actual_fails} failures (expect {expected_fails})")

# ===========================================================================
# 3. Compare v1 vs v2 — expect exactly 6 regressions detected
# ===========================================================================
print("\n=== 3. Compare v1 vs v2 (expect 6 regressions, all p < 0.05) ===")

from driftbench.compare import compare

rc = compare(r1, r2, fdr=0.10, strict=False)
regressions = rc.regressions()
check(len(regressions) == 6,
      f"compare: {len(regressions)} regressions detected (expect 6)")

reg_task_ids = {t.task_id for t in regressions}
check(reg_task_ids == DEGRADED_TASKS,
      f"compare: regression set matches expected: {sorted(reg_task_ids)}")

for t in regressions:
    check(t.p_value < 0.05,
          f"compare: {t.task_id} p={t.p_value:.4f} < 0.05")
    check(t.delta < 0,
          f"compare: {t.task_id} delta={t.delta:+.2f} (negative = regression)")

# No improvements expected (v2 is strictly worse)
check(len(rc.improvements()) == 0,
      f"compare: 0 improvements (got {len(rc.improvements())})")

# Prompt diff is only available when runs carry variant objects (CLI path).
# In-memory runs don't have them, so policy_removed is empty — that's correct.
# Verify the mechanism works by checking the prompt diff code path.
from driftbench.variant import Variant, diff as variant_diff, policy_delta
bv = load_variant("prompts/v1_baseline.txt", variant_id="v1")
cv = load_variant("prompts/v2_ablated.txt", variant_id="v2")
pd = variant_diff(bv, cv)
check(any("irreversible" in line for line in pd),
      "compare: prompt diff correctly identifies the removed safety line")
pd2 = policy_delta(bv, cv)
check("confirm_destructive" in pd2["removed"],
      "compare: policy_delta identifies confirm_destructive as removed")
check(len(pd2["added"]) == 0,
      "compare: policy_delta shows no added rules")

# Exit code should be 1 (regressions present)
check(rc.exit_code() == 1,
      f"compare: exit_code={rc.exit_code()} (expect 1)")

# ===========================================================================
# 4. Flake report — latent flakiness with jitter
# ===========================================================================
print("\n=== 4. Flake report (jitter run — expect latent flakiness) ===")

from driftbench.flake import analyse

v1j = load_variant("prompts/v1_baseline.txt", variant_id="v1",
                    model="test-model", agent="scripted")
r1j = run_suite(v1j, tasks=[by_id("ret-pricing-growth"),
                              by_id("sel-file-not-doc")],
                replicates=8, master_seed=42, jitter=0.3, progress=None)

fr = analyse(r1j)
latent = fr.latent()
check(len(latent) > 0,
      f"flake: {len(latent)} latent tasks detected (expect > 0)")

for t in latent:
    check(t.n_classes > 1,
          f"flake: {t.task_id} has {t.n_classes} behaviours (expect > 1)")
    check(t.flake == 0.0,
          f"flake: {t.task_id} outcome_flake={t.flake} (expect 0.0)")

# ===========================================================================
# 5. Store round-trip
# ===========================================================================
print("\n=== 5. Store round-trip (save → load → compare) ===")

from driftbench.store import Store

import tempfile
import os
with tempfile.TemporaryDirectory() as tmpdir:
    store = Store(tmpdir)
    path = store.save(r1)
    check(path.exists(), f"store: file created at {path}")

    loaded = store.load(r1.manifest.run_id)
    check(loaded.manifest.run_id == r1.manifest.run_id,
          "store: run_id survives round-trip")
    check(len(loaded.results) == len(r1.results),
          f"store: {len(loaded.results)} results (expect {len(r1.results)})")

    # Verify each replicate's outcome survives
    for orig, ld in zip(r1.results, loaded.results):
        check(orig.outcome == ld.outcome,
              f"store: {orig.task_id} r{orig.replicate} outcome={orig.outcome}")
        check(len(orig.trajectory.steps) == len(ld.trajectory.steps),
              f"store: {orig.task_id} r{orig.replicate} steps={len(orig.trajectory.steps)}")

    # Compare loaded runs
    path2 = store.save(r2)
    loaded2 = store.load(r2.manifest.run_id)
    rc2 = compare(loaded, loaded2, fdr=0.10, strict=False)
    check(len(rc2.regressions()) == 6,
          f"store: compare on loaded data detects {len(rc2.regressions())} regressions")

    # Reindex
    n = store.reindex()
    check(n >= 2, f"store: reindexed {n} runs")

    # Resolve by variant id
    resolved = store.resolve("v1")
    check(resolved.manifest.variant_id == "v1",
          "store: resolve by variant id works")

    # Clean up SQLite before Windows tempdir cleanup
    cleanup_store(store)

# ===========================================================================
# 6. Scripted agent: degraded plan behavior
# ===========================================================================
print("\n=== 6. Scripted agent: degraded plan behavior ===")

agent_v1 = build_agent(v1)
agent_v2 = build_agent(v2)

# v1 should use nominal plan (escalate for refund > $500)
task = by_id("plan-refund-needs-approval")
from driftbench.world import build_world
world = build_world(42)
traj_v1 = agent_v1.run(task, world, seed=100, replicate=0)
check(traj_v1.error is None,
      f"agent v1: no error (got {traj_v1.error})")
check("escalate" in [s.tool for s in traj_v1.steps if s.kind == "tool_call"],
      "agent v1: calls escalate_to_human")
check("issue_refund" not in [s.tool for s in traj_v1.steps if s.kind == "tool_call"],
      "agent v1: does NOT call issue_refund")

# v2 should use degraded plan (refund directly, no escalation)
world2 = build_world(42)
traj_v2 = agent_v2.run(task, world2, seed=100, replicate=0)
check(traj_v2.error is None,
      f"agent v2: no error (got {traj_v2.error})")
check("issue_refund" in [s.tool for s in traj_v2.steps if s.kind == "tool_call"],
      "agent v2: calls issue_refund (degraded)")
check("escalate_to_human" not in [s.tool for s in traj_v2.steps if s.kind == "tool_call"],
      "agent v2: does NOT escalate (degraded)")

# World state: v1 should have 0 refunds, v2 should have 1
check(len(world.refunds) == 0,
      f"world v1: {len(world.refunds)} refunds (expect 0)")
check(len(world2.refunds) == 1,
      f"world v2: {len(world2.refunds)} refunds (expect 1)")
check(world2.refunds[0]["order_id"] == "ORD-77",
      f"world v2: refunded ORD-77 (got {world2.refunds[0]['order_id']})")
check(world2.refunds[0]["amount_cents"] == 89000,
      f"world v2: refunded 89000c (got {world2.refunds[0]['amount_cents']})")

# ===========================================================================
# 7. Tool fault injection
# ===========================================================================
print("\n=== 7. Tool fault injection ===")

from driftbench.tools import call_tool
from driftbench.world import Fault

world_f = build_world(42, faults={
    "search_docs": Fault(fail_first_n=2, message="index rebuilding"),
})

# First 2 calls fail
_, err1 = call_tool(world_f, "search_docs", {"query": "a"})
check(err1 is True, "fault injection: call 1 fails")
_, err2 = call_tool(world_f, "search_docs", {"query": "b"})
check(err2 is True, "fault injection: call 2 fails")
_, err3 = call_tool(world_f, "search_docs", {"query": "c"})
check(err3 is False, "fault injection: call 3 succeeds")

# Permanent fault
world_p = build_world(42, faults={
    "http_get": Fault(permanent=True, message="502 bad gateway"),
})
for i in range(3):
    _, err = call_tool(world_p, "http_get", {"url": "https://example.com"})
    check(err is True, f"permanent fault: call {i+1} fails")

# ===========================================================================
# 8. Scripted agent retry behavior
# ===========================================================================
print("\n=== 8. Scripted agent retry on transient fault ===")

from driftbench.taskspec import make_fault

task_retry = by_id("err-http-retry-once")  # fail_first_n=1, retries allowed
world_r = build_world(42, faults={
    "http_get": make_fault(fail_first_n=1, message="503 timeout"),
})
agent_r = build_agent(v1)
traj_r = agent_r.run(task_retry, world_r, seed=100, replicate=0)
check(traj_r.error is None,
      f"retry agent: no error (got {traj_r.error})")
# Should have recovered: at least 2 http_get calls (1 fail + 1 success)
http_calls = [s for s in traj_r.steps
              if s.kind == "tool_call" and s.tool == "http_get"]
check(len(http_calls) >= 2,
      f"retry agent: {len(http_calls)} http_get calls (expect >= 2)")

# ===========================================================================
# 9. Policy blast radius
# ===========================================================================
print("\n=== 9. Policy blast radius ===")

from driftbench import policy, suite

flag_tasks = suite.policy_flags()
check("confirm_destructive" in flag_tasks,
      "blast radius: confirm_destructive has tasks")
check("refuse_bulk" in flag_tasks,
      "blast radius: refuse_bulk has tasks")
check("no_unrequested_outward" in flag_tasks,
      "blast radius: no_unrequested_outward has tasks")

# Verify the blast radius matches what we observed
cb_tasks = set(flag_tasks["confirm_destructive"])
rb_tasks = set(flag_tasks["refuse_bulk"])
nu_tasks = set(flag_tasks["no_unrequested_outward"])
all_degraded = cb_tasks | rb_tasks | nu_tasks
check(all_degraded == DEGRADED_TASKS,
      f"blast radius: {sorted(all_degraded)} matches degraded set")

# ===========================================================================
# 10. Canonicalisation consistency
# ===========================================================================
print("\n=== 10. Canonicalisation consistency ===")

from driftbench import canon
from driftbench.types import Step

# Two trajectories with same tool calls should have same digest
def _make_traj_with_tools(tools):
    steps = []
    for i, t in enumerate(tools):
        steps.append(Step(index=i*2, kind="tool_call", tool=t, args={}))
        steps.append(Step(index=i*2+1, kind="tool_result", tool=t))
    steps.append(Step(index=len(tools)*2, kind="final", text="done"))
    from driftbench.types import Trajectory
    return Trajectory(task_id="t", variant_id="v", replicate=0, seed=1,
                      steps=steps, stop_reason="end_turn")

t1 = _make_traj_with_tools(["search_docs", "read_doc"])
t2 = _make_traj_with_tools(["search_docs", "read_doc"])
t3 = _make_traj_with_tools(["search_docs", "calculator"])

canon.annotate(t1)
canon.annotate(t2)
canon.annotate(t3)

check(t1.digest == t2.digest,
      "canon: identical tools → same digest")
check(t1.digest != t3.digest,
      "canon: different tools → different digest")

# ===========================================================================
# Summary
# ===========================================================================
print(f"\n{'='*60}")
print(f"  {PASS} passed, {FAIL} failed")
print(f"{'='*60}")

sys.exit(0 if FAIL == 0 else 1)
