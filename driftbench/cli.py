"""CLI entry point for driftbench.

``driftbench <command> [options]``

Commands:
    run        Execute a variant against the task suite
    compare    Diff two recorded runs
    flake      Flakiness report for one run
    list       List recorded runs
    show       Show one run's details
    variants   Show prompt flags and blast radius
    reindex    Rebuild the SQLite index from JSONL archives
    suite      List tasks in the suite
    check      Validate a prompt file against expected phrases
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="driftbench",
        description="Regression and flakiness testing for LLM agents.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Shared --root flag for all subcommands that need the store
    _store_parent = argparse.ArgumentParser(add_help=False)
    _store_parent.add_argument("--root", default="runs",
                               help="Store directory (default: ./runs)")

    # --- run ---------------------------------------------------------------
    p_run = sub.add_parser("run", help="Execute a variant against the suite",
                           parents=[_store_parent])
    p_run.add_argument("--variant", "-v", required=True,
                       help="Path to system prompt .txt file")
    p_run.add_argument("--model", default="claude-opus-5",
                       help="Model id (default: claude-opus-5)")
    p_run.add_argument("--effort", default="medium",
                       choices=["low", "medium", "high", "max"],
                       help="Effort level (default: medium)")
    p_run.add_argument("--agent", default="scripted",
                       choices=["scripted", "anthropic", "live", "claude"],
                       help="Agent type (default: scripted)")
    p_run.add_argument("--replicates", "-n", type=int, default=5,
                       help="Replicates per task (default: 5)")
    p_run.add_argument("--seed", type=int, default=20260830,
                       help="Master seed (default: 20260830)")
    p_run.add_argument("--tasks", default=None,
                       help="Comma-separated task id/category globs (default: all)")
    p_run.add_argument("--workers", type=int, default=1,
                       help="Parallel workers (default: 1)")
    p_run.add_argument("--jitter", type=float, default=0.0,
                       help="Scripted agent jitter probability (default: 0.0)")
    p_run.add_argument("--strict-judge", action="store_true",
                       help="Judge also gates pass/fail")
    p_run.add_argument("--judge-model", default="claude-sonnet-5",
                       help="Model for the judge (default: claude-sonnet-5)")
    p_run.add_argument("--no-judge", action="store_true",
                       help="Skip judge scoring")
    p_run.add_argument("--canon-level", default="semantic",
                       choices=["shape", "semantic", "strict"],
                       help="Canonicalisation level (default: semantic)")
    p_run.add_argument("--notes", default="", help="Notes for the run")
    p_run.add_argument("--explain", action="store_true",
                       help="Show what each task will do under this variant")

    # --- compare -----------------------------------------------------------
    p_cmp = sub.add_parser("compare", help="Diff two recorded runs",
                           parents=[_store_parent])
    p_cmp.add_argument("base", help="Baseline run id or variant id")
    p_cmp.add_argument("candidate", help="Candidate run id or variant id")
    p_cmp.add_argument("--fdr", type=float, default=0.10,
                       help="Benjamini-Hochberg FDR threshold (default: 0.10)")
    p_cmp.add_argument("--level", default="semantic",
                       choices=["shape", "semantic", "strict"])
    p_cmp.add_argument("--lenient", action="store_true",
                       help="Downgrade comparability guards to warnings")

    # --- flake -------------------------------------------------------------
    p_flake = sub.add_parser("flake", help="Flakiness report for one run",
                            parents=[_store_parent])
    p_flake.add_argument("run", help="Run id or variant id")
    p_flake.add_argument("--level", default="semantic",
                         choices=["shape", "semantic", "strict"])
    p_flake.add_argument("--top", type=int, default=10,
                         help="Show top N unstable tasks (default: 10)")

    # --- list --------------------------------------------------------------
    p_list = sub.add_parser("list", help="List recorded runs",
                           parents=[_store_parent])
    p_list.add_argument("--variant", default=None, help="Filter by variant id")
    p_list.add_argument("--limit", type=int, default=20,
                        help="Max rows (default: 20)")

    # --- show --------------------------------------------------------------
    p_show = sub.add_parser("show", help="Show one run's details",
                           parents=[_store_parent])
    p_show.add_argument("run", help="Run id or variant id or path")
    p_show.add_argument("--task", default=None, help="Filter to one task id")

    # --- variants ----------------------------------------------------------
    p_var = sub.add_parser("variants", help="Show prompt flags and blast radius")
    p_var.add_argument("prompt", nargs="?", default=None,
                       help="Path to prompt file (default: show both prompts/)")
    p_var.add_argument("--check", default=None,
                       help="Check a prompt for missing phrases")

    # --- reindex -----------------------------------------------------------
    sub.add_parser("reindex", help="Rebuild SQLite index from JSONL archives",
                   parents=[_store_parent])

    # --- suite -------------------------------------------------------------
    p_suite = sub.add_parser("suite", help="List tasks in the suite")
    p_suite.add_argument("--category", default=None, help="Filter by category")

    # --- check -------------------------------------------------------------
    p_check = sub.add_parser("check", help="Validate a prompt file")
    p_check.add_argument("prompt", help="Path to prompt file")

    args = parser.parse_args(argv)

    # Lazy imports so --help is instant
    from .store import Store

    store_root = getattr(args, 'root', 'runs')
    store = Store(store_root)

    if args.command == "run":
        return _cmd_run(args, store)
    elif args.command == "compare":
        return _cmd_compare(args, store)
    elif args.command == "flake":
        return _cmd_flake(args, store)
    elif args.command == "list":
        return _cmd_list(args, store)
    elif args.command == "show":
        return _cmd_show(args, store)
    elif args.command == "variants":
        return _cmd_variants(args)
    elif args.command == "reindex":
        return _cmd_reindex(store)
    elif args.command == "suite":
        return _cmd_suite(args)
    elif args.command == "check":
        return _cmd_check(args)
    else:
        parser.print_help()
        return 1


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------

def _cmd_run(args, store) -> int:
    from . import report
    from .runner import run_suite
    from .variant import Variant, load

    variant = load(args.variant, variant_id=Path(args.variant).stem,
                   model=args.model, effort=args.effort, agent=args.agent)

    if args.explain:
        from . import suite as suite_mod
        tasks = suite_mod.select(args.tasks)
        from .agents.scripted import ScriptedAgent
        agent = ScriptedAgent(variant, jitter=args.jitter)
        for task in tasks:
            desc = agent.describe(task)
            print(f"  {task.id:<32} {desc}")
        return 0

    # Build judge
    judge = None
    if not args.no_judge:
        if args.agent in ("scripted",):
            # Scripted agent: no judge needed (deterministic)
            judge = None
        else:
            try:
                from .judge import Judge
                judge = Judge(model=args.judge_model)
            except ImportError:
                print("  warning: anthropic package not installed, "
                      "skipping judge", file=sys.stderr)
                judge = None

    from . import suite as suite_mod
    tasks = suite_mod.select(args.tasks)

    def progress(msg):
        print(f"  {msg}", file=sys.stderr)

    print(f"Running {variant.id} [{variant.model}/{variant.effort}] "
          f"against {len(tasks)} tasks x {args.replicates} replicates...",
          file=sys.stderr)

    t0 = time.monotonic()
    result = run_suite(
        variant, tasks=tasks, replicates=args.replicates,
        master_seed=args.seed, judge=judge,
        strict_judge=args.strict_judge, jitter=args.jitter,
        workers=args.workers, notes=args.notes,
        canon_level=args.canon_level, progress=progress,
    )
    elapsed = time.monotonic() - t0

    # Save
    path = store.save(result)
    print(f"\n  saved to {path}", file=sys.stderr)
    print(f"  elapsed {elapsed:.1f}s", file=sys.stderr)

    # Print summary
    print()
    print(report.render_run(result))
    return 0


def _cmd_compare(args, store) -> int:
    from . import report
    from .compare import compare, IncomparableRuns

    base = store.resolve(args.base)
    cand = store.resolve(args.candidate)

    try:
        rc = compare(base, cand, fdr=args.fdr, level=args.level,
                     strict=not args.lenient)
    except IncomparableRuns as exc:
        print(f"  error: {exc}", file=sys.stderr)
        return 1

    print(report.render_comparison(rc))
    return rc.exit_code()


def _cmd_flake(args, store) -> int:
    from . import report
    from .flake import analyse

    run = store.resolve(args.run)
    fr = analyse(run, level=args.level)
    print(report.render_flake(fr))
    return 0


def _cmd_list(args, store) -> int:
    from . import report
    rows = store.runs(variant_id=args.variant, limit=args.limit)
    print(report.render_run_list(rows))
    return 0


def _cmd_show(args, store) -> int:
    from . import report

    run = store.resolve(args.run)
    if args.task:
        by_task = run.by_task()
        if args.task not in by_task:
            print(f"  error: task {args.task!r} not in this run", file=sys.stderr)
            return 1
        rows = by_task[args.task]
        for r in rows:
            print(f"  replicate {r.replicate}  outcome={r.outcome}  "
                  f"steps={len(r.trajectory.steps)}  "
                  f"calls={len(r.trajectory.tool_calls())}")
            if r.trajectory.final_text:
                text = r.trajectory.final_text[:200]
                print(f"    final: {text}")
            if r.assertions:
                for a in r.assertions:
                    sym = "+" if a.passed else "x"
                    print(f"    [{sym}] {a.name}: {a.detail}")
            print()
    else:
        print(report.render_run(run))
    return 0


def _cmd_variants(args) -> int:
    from . import report, policy
    from .variant import Variant

    if args.check:
        text = Path(args.check).read_text(encoding="utf-8")
        variant = Variant(id=Path(args.check).stem, system_prompt=text)
        print(report.render_variant_info(variant))
        return 0

    # Show all prompt files in prompts/
    prompts_dir = Path("prompts")
    if not prompts_dir.exists():
        print("  no prompts/ directory found", file=sys.stderr)
        return 1

    for p in sorted(prompts_dir.glob("*.txt")):
        text = p.read_text(encoding="utf-8")
        variant = Variant(id=p.stem, system_prompt=text, source=str(p))
        print(report.render_variant_info(variant))
        print()

    return 0


def _cmd_reindex(store) -> int:
    n = store.reindex()
    print(f"  reindexed {n} run(s)")
    return 0


def _cmd_suite(args) -> int:
    from . import suite as suite_mod

    tasks = suite_mod.all_tasks()
    if args.category:
        tasks = [t for t in tasks if t.category == args.category]

    by_cat = {}
    for t in tasks:
        by_cat.setdefault(t.category, []).append(t)

    for cat in suite_mod.CATEGORY_ORDER:
        cat_tasks = by_cat.get(cat, [])
        if not cat_tasks:
            continue
        print(f"\n  {cat} ({len(cat_tasks)} tasks):")
        for t in cat_tasks:
            nchecks = len(t.checks)
            ncrit = len(t.critical_checks())
            needs = t.script.get("needs", [])
            needs_str = f"  needs [{', '.join(needs)}]" if needs else ""
            print(f"    {t.id:<32} {ncrit}/{nchecks} checks{needs_str}")

    print(f"\n  {len(tasks)} tasks total")
    return 0


def _cmd_check(args) -> int:
    from . import policy
    from .variant import Variant

    text = Path(args.prompt).read_text(encoding="utf-8")
    variant = Variant(id=Path(args.prompt).stem, system_prompt=text)

    present = sorted(variant.policy())
    absent = sorted(set(policy.ALL_FLAGS) - set(present))

    print(f"  {len(present)}/{len(policy.ALL_FLAGS)} rules present")
    if present:
        print(f"  present: {', '.join(present)}")
    if absent:
        print(f"  MISSING: {', '.join(absent)}")
        for flag in absent:
            print(f"    {flag}: expected phrase /{policy.DIRECTIVES[flag]}/")
        return 1
    else:
        print("  all expected phrases found")
        return 0


if __name__ == "__main__":
    sys.exit(main())
