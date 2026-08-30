"""Execution: variant x task x replicate, with derived seeds and scored results.

The only interesting decision in here is what "pass" means.

**Assertions gate; the judge reports.** ``outcome`` is decided purely by the
critical programmatic checks, which are deterministic and reproducible. The rubric
judge runs alongside and is recorded per replicate, but by default it does not
change the outcome. The reason is circular otherwise: the judge is the instrument
being calibrated against hand labels, and an instrument that also decides the
answer cannot be evaluated against it. Pass ``strict_judge=True`` if you want the
judge to gate as well, and expect your regression counts to inherit its noise.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

from . import canon, seeding, suite as suite_mod
from .agents import build as build_agent
from .checks import CheckContext
from .results import AssertionResult, ReplicateResult, RunManifest
from .taskspec import TaskSpec
from .types import Trajectory
from .variant import Variant
from .world import build_world

Progress = Optional[Callable[[str], None]]


@dataclass
class RunResult:
    manifest: RunManifest
    variant: Variant
    results: list[ReplicateResult] = field(default_factory=list)

    def by_task(self) -> dict[str, list[ReplicateResult]]:
        out: dict[str, list[ReplicateResult]] = {}
        for r in self.results:
            out.setdefault(r.task_id, []).append(r)
        for rows in out.values():
            rows.sort(key=lambda r: r.replicate)
        return out

    def task_ids(self) -> list[str]:
        return list(self.by_task())

    def pass_counts(self) -> dict[str, tuple[int, int]]:
        """task -> (passes, total)."""
        return {t: (sum(1 for r in rows if r.passed), len(rows))
                for t, rows in self.by_task().items()}

    def total_cost_usd(self) -> float:
        return sum(r.trajectory.usage.cost_usd(r.trajectory.model or self.variant.model)
                   for r in self.results) + sum(
            r.judge.usage.cost_usd(r.judge.judge_model)
            for r in self.results if r.judge)

    def outcome_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.results:
            counts[r.outcome] = counts.get(r.outcome, 0) + 1
        return counts

    def to_dict(self) -> dict:
        return {"manifest": self.manifest.to_dict(),
                "variant": self.variant.to_dict(),
                "results": [r.to_dict() for r in self.results]}

    @classmethod
    def from_dict(cls, d: dict) -> "RunResult":
        return cls(manifest=RunManifest.from_dict(d["manifest"]),
                   variant=Variant.from_dict(d["variant"]),
                   results=[ReplicateResult.from_dict(r) for r in d.get("results", [])])


def score(task: TaskSpec, traj: Trajectory, world) -> list[AssertionResult]:
    ctx = CheckContext(traj=traj, world=world, task=task)
    out = []
    for check in task.checks:
        passed, detail = check.run(ctx)
        out.append(AssertionResult(name=check.name, passed=passed, detail=detail,
                                   critical=check.critical))
    return out


def decide_outcome(traj: Trajectory, assertions: list[AssertionResult],
                   judge=None, strict_judge: bool = False) -> str:
    if traj.error:
        return "error"
    if traj.stop_reason == "refusal":
        return "refusal"
    if any(not a.passed for a in assertions if a.critical):
        return "fail"
    if strict_judge and judge is not None and judge.label == "bad":
        return "fail"
    if traj.stop_reason == "max_steps":
        return "timeout"
    return "pass"

def run_cell(agent, task: TaskSpec, variant: Variant, master_seed: int,
             replicate: int, judge=None, strict_judge: bool = False,
             canon_level: str = canon.DEFAULT_LEVEL) -> ReplicateResult:
    """One (task, replicate) cell: fresh world, run, score.

    Two seeds, not one. The world is built from ``env_seed``, which does not
    include the variant, so replicate 7 of a fault-injected task faces the same
    fault schedule under every variant and a v1-vs-v2 report is a paired
    comparison. The agent gets ``derive_seed``, which does include the variant,
    because its own draws are part of the system under test. The recorded seed is
    the agent's; the environment's is recoverable from the manifest's master seed.

    A crash inside the agent becomes a recorded ``error`` outcome rather than an
    aborted run. Losing 39 good results because task 12 raised is how people stop
    trusting a harness.
    """
    seed = seeding.derive_seed(master_seed, variant.id, task.id, replicate)
    world = build_world(seeding.env_seed(master_seed, task.id, replicate),
                        faults=task.faults, given=task.given,
                        overlay=task.overlay)
    try:
        traj = agent.run(task, world, seed, replicate)
    except Exception as exc:  # noqa: BLE001 - agent bugs must not abort the run
        traj = Trajectory(task_id=task.id, variant_id=variant.id,
                          replicate=replicate, seed=seed,
                          agent=getattr(agent, "name", "?"),
                          model=getattr(agent, "model", ""),
                          error=f"{type(exc).__name__}: {exc}",
                          stop_reason="error")
        canon.annotate(traj, canon_level)
    else:
        canon.annotate(traj, canon_level)

    assertions = score(task, traj, world)
    verdict = None
    if judge is not None and traj.error is None:
        verdict = judge.score(task, traj)
    return ReplicateResult(trajectory=traj, assertions=assertions, judge=verdict,
                           outcome=decide_outcome(traj, assertions, verdict,
                                                  strict_judge))


def make_run_id(variant: Variant, suite_dig: str) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    short = canon.digest_strings(variant.digest(), suite_dig, str(time.time()))[:6]
    return f"{variant.id}-{stamp}-{short}"

def run_suite(variant: Variant, tasks: Optional[Iterable[TaskSpec]] = None,
              replicates: int = 5, master_seed: int = 20260830,
              judge=None, strict_judge: bool = False, jitter: float = 0.0,
              workers: int = 1, notes: str = "",
              canon_level: str = canon.DEFAULT_LEVEL,
              progress: Progress = None) -> RunResult:
    """Run every (task, replicate) cell for one variant.

    ``replicates`` is the whole point of the harness: one run of one task tells
    you nothing about whether a change is real. Five is the floor for seeing
    flakiness at all; twenty is where the Wilson intervals get tight enough to
    call a small regression.
    """
    task_list = list(tasks) if tasks is not None else suite_mod.all_tasks()
    if replicates < 1:
        raise ValueError("replicates must be at least 1")
    agent = build_agent(variant, jitter=jitter)
    suite_dig = suite_mod.digest(task_list)

    manifest = RunManifest(
        run_id=make_run_id(variant, suite_dig), variant_id=variant.id,
        variant_digest=variant.digest(), suite_digest=suite_dig,
        agent=getattr(agent, "name", variant.agent), model=variant.model,
        effort=variant.effort, replicates=replicates,
        task_ids=[t.id for t in task_list], master_seed=master_seed,
        judge_model=getattr(judge, "model", "") if judge else "",
        notes=notes,
    )

    cells = [(task, rep) for task in task_list for rep in range(replicates)]
    done = 0
    total = len(cells)

    def one(cell) -> ReplicateResult:
        task, rep = cell
        return run_cell(agent, task, variant, master_seed, rep, judge,
                        strict_judge, canon_level)

    results: list[ReplicateResult] = []
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for res in pool.map(one, cells):
                results.append(res)
                done += 1
                if progress:
                    progress(f"[{done}/{total}] {res.task_id} r{res.replicate} "
                             f"{res.outcome}")
    else:
        for cell in cells:
            res = one(cell)
            results.append(res)
            done += 1
            if progress:
                progress(f"[{done}/{total}] {res.task_id} r{res.replicate} "
                         f"{res.outcome}")

    order = {t.id: i for i, t in enumerate(task_list)}
    results.sort(key=lambda r: (order.get(r.task_id, 0), r.replicate))
    return RunResult(manifest=manifest, variant=variant, results=results)
