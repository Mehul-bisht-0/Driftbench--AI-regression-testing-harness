"""The suite: 40 tasks across nine categories.

Assembled from five modules rather than one file so that the categories stay
legible and a task's neighbours are the tasks it is meant to be compared with.

The ``digest()`` here is what makes cross-run comparison honest. It hashes every
task id *together with* that task's full signature - prompt, check definitions,
rubric, injected faults, step budget - so editing a check or reworing a prompt
produces a different digest, and ``driftbench compare`` refuses to diff runs from
two different suites instead of quietly reporting a regression that is really
just an edited test.
"""

from __future__ import annotations

import fnmatch
from typing import Iterable, Optional

from .. import seeding
from ..taskspec import TaskSpec
from . import discipline, planning, resilience, retrieval, safety

MODULES = (retrieval, planning, resilience, safety, discipline)

# Declared order is the report order: retrieval first because it is the easiest
# thing to get right, refusal boundaries last because they are the hardest.
CATEGORY_ORDER = (
    "retrieval", "tool_selection", "multi_step", "state_tracking",
    "error_recovery", "honesty", "format_compliance", "efficiency",
    "ambiguity", "safety_confirm", "refusal_boundary",
)


def _collect() -> list[TaskSpec]:
    tasks: list[TaskSpec] = []
    seen: dict[str, str] = {}
    for module in MODULES:
        for task in module.TASKS:
            if task.id in seen:
                raise ValueError(
                    f"duplicate task id {task.id!r} in {module.__name__} "
                    f"(already defined in {seen[task.id]})")
            seen[task.id] = module.__name__
            tasks.append(task)
    unknown = {t.category for t in tasks} - set(CATEGORY_ORDER)
    if unknown:
        raise ValueError(f"tasks use categories missing from CATEGORY_ORDER: "
                         f"{sorted(unknown)}")
    tasks.sort(key=lambda t: (CATEGORY_ORDER.index(t.category), t.id))
    return tasks


TASKS: list[TaskSpec] = _collect()
BY_ID: dict[str, TaskSpec] = {t.id: t for t in TASKS}

def all_tasks() -> list[TaskSpec]:
    return list(TASKS)


def by_id(task_id: str) -> TaskSpec:
    try:
        return BY_ID[task_id]
    except KeyError:
        raise KeyError(f"no task {task_id!r}; try `driftbench list`") from None


def categories() -> list[str]:
    present = {t.category for t in TASKS}
    return [c for c in CATEGORY_ORDER if c in present]


def by_category() -> dict[str, list[TaskSpec]]:
    out: dict[str, list[TaskSpec]] = {c: [] for c in categories()}
    for task in TASKS:
        out[task.category].append(task)
    return out


def select(pattern: Optional[str] = None) -> list[TaskSpec]:
    """Filter tasks by comma-separated globs against id or category.

    ``None`` or ``"all"`` means everything. ``"safe-*,refusal_boundary"`` means
    the four confirmation tasks plus every refusal task.
    """
    if not pattern or pattern.strip().lower() == "all":
        return all_tasks()
    globs = [p.strip() for p in pattern.split(",") if p.strip()]
    picked = [t for t in TASKS
              if any(fnmatch.fnmatch(t.id, g) or fnmatch.fnmatch(t.category, g)
                     for g in globs)]
    if not picked:
        raise ValueError(f"pattern {pattern!r} matched no tasks; "
                         f"known categories: {', '.join(categories())}")
    return picked


def tasks_needing(flag: str) -> list[TaskSpec]:
    """Tasks whose fixture behaviour degrades when a policy flag is absent.

    Only meaningful for the offline ScriptedAgent, and used by the demo to state
    up front how many tasks a given prompt line is load-bearing for.
    """
    return [t for t in TASKS if flag in (t.script.get("needs") or ())]


def policy_flags() -> dict[str, list[str]]:
    """flag -> task ids that depend on it. The demo's blast-radius table."""
    out: dict[str, list[str]] = {}
    for task in TASKS:
        for flag in task.script.get("needs") or ():
            out.setdefault(flag, []).append(task.id)
    return {k: sorted(v) for k, v in sorted(out.items())}


def digest(tasks: Optional[Iterable[TaskSpec]] = None) -> str:
    """Identity of the suite *as a test set*, including check definitions."""
    chosen = list(tasks) if tasks is not None else TASKS
    return seeding.suite_digest(
        [t.id for t in chosen], extra=[t.signature() for t in chosen])
