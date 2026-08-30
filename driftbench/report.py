"""Terminal-friendly text renderers for comparison, flake, and run results.

Pure text, no HTML. ANSI colors when stdout is a TTY, plain text otherwise.
No external dependencies.
"""

from __future__ import annotations

import sys
from typing import Optional, Sequence

from . import diff
from .compare import RunComparison, TaskComparison
from .flake import FlakeReport, TaskFlake, BANDS, BAND_MEANING
from .runner import RunResult

# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------

def _use_color() -> bool:
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


_COLOR = _use_color()

_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RED = "\033[31m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_CYAN = "\033[36m"
_MAGENTA = "\033[35m"


def _c(code: str, text: str) -> str:
    if not _COLOR:
        return text
    return f"{code}{text}{_RESET}"


# ---------------------------------------------------------------------------
# Comparison report
# ---------------------------------------------------------------------------

def render_comparison(rc: RunComparison) -> str:
    """Full comparison report as a single string."""
    lines: list[str] = []
    lines.append(_c(_BOLD, rc.headline()))
    for w in rc.warnings:
        lines.append(_c(_YELLOW, f"  warning: {w}"))
    if rc.policy_removed:
        lines.append(_c(_YELLOW, "  prompt lost: " + ", ".join(rc.policy_removed)))
    if rc.policy_added:
        lines.append(_c(_GREEN, "  prompt gained: " + ", ".join(rc.policy_added)))
    if rc.prompt_diff:
        lines.append("")
        lines.append(_c(_DIM, "  prompt diff:"))
        for ln in rc.prompt_diff:
            lines.append(f"  {ln}")

    for group, title, color in (
        ("regression", "regressions", _RED),
        ("improvement", "improvements", _GREEN),
        ("behavior_drift", "behaviour drift", _YELLOW),
        ("noise", "moved but not significant", _DIM),
    ):
        rows = rc.of(group)
        if not rows:
            continue
        lines.append("")
        lines.append(_c(color, f"{title} ({len(rows)}):"))
        for t in rows:
            lines.append(_c(color, "  " + t.headline()))
            for ln in t.explain()[1:]:
                lines.append(f"  {ln}")

    if not rc.regressions() and not rc.improvements() and not rc.drift():
        lines.append("")
        lines.append(_c(_GREEN, "  all tasks stable"))

    # Stable count
    stable = len(rc.of("stable"))
    if stable:
        lines.append("")
        lines.append(_c(_DIM, f"  {stable} task(s) stable (omitted)"))

    lines.append("")
    lines.append(_c(_DIM, f"  noise floor +/-{rc.noise_floor * 100:.1f}pp, "
                     f"BH FDR {rc.fdr:.0%}, {rc.replicates} replicates"))
    return "\n".join(lines)


def render_comparison_brief(rc: RunComparison) -> str:
    """One-line summary of a comparison."""
    return rc.headline()


# ---------------------------------------------------------------------------
# Flake report
# ---------------------------------------------------------------------------

def render_flake(fr: FlakeReport) -> str:
    """Full flakiness report as a single string."""
    lines: list[str] = []
    lines.append(_c(_BOLD, fr.summary()))
    lines.append("")

    unstable = fr.unstable()
    if not unstable:
        lines.append(_c(_GREEN, "  all tasks stable"))
        return "\n".join(lines)

    band_colors = {
        "chaotic": _RED,
        "flaky": _YELLOW,
        "jittery": _CYAN,
        "stable": _GREEN,
    }

    for t in unstable:
        color = band_colors.get(t.band(), "")
        lines.append(_c(color, t.headline()))
        for ln in t.explain()[1:]:
            lines.append(f"  {ln}")
        lines.append("")

    latent = fr.latent()
    if latent:
        lines.append(_c(_YELLOW, f"  {len(latent)} latent flake(s) - green every run, "
                       "different route each time"))
        for t in latent:
            lines.append(f"    {t.task_id}: {t.n_classes} behaviours, "
                         f"{t.entropy:.2f} entropy")
        lines.append("")

    return "\n".join(lines)


def render_flake_brief(fr: FlakeReport) -> str:
    """One-line summary of a flake report."""
    return fr.summary()


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def render_run(result: RunResult) -> str:
    """Full run details: manifest header + per-task grid."""
    lines: list[str] = []
    m = result.manifest
    lines.append(_c(_BOLD, f"Run {m.run_id}"))
    lines.append(f"  variant:   {m.variant_id}")
    lines.append(f"  agent:     {m.agent}  model: {m.model}  effort: {m.effort}")
    lines.append(f"  tasks:     {len(m.task_ids)}")
    lines.append(f"  replicates:{m.replicates}  seed: {m.master_seed}")
    if m.notes:
        lines.append(f"  notes:     {m.notes}")
    lines.append(f"  suite:     {m.suite_digest[:12]}")
    lines.append(f"  variant:   {m.variant_digest[:12]}")
    lines.append("")

    counts = result.outcome_counts()
    total = sum(counts.values())
    passed = counts.get("pass", 0)
    lines.append(f"  outcome: {passed}/{total} passed "
                 f"({counts.get('pass', 0)} pass, "
                 f"{counts.get('fail', 0)} fail, "
                 f"{counts.get('error', 0)} error, "
                 f"{counts.get('refusal', 0)} refusal, "
                 f"{counts.get('timeout', 0)} timeout)")
    cost = result.total_cost_usd()
    if cost > 0:
        lines.append(f"  cost:    ${cost:.4f}")
    lines.append("")

    # Per-task grid
    by_task = result.by_task()
    lines.append(f"  {'task':<32} {'pass':>5}  {'fail':>5}  outcome")
    lines.append(f"  {'-'*32} {'-'*5}  {'-'*5}  {'-'*8}")
    for task_id in m.task_ids:
        rows = by_task.get(task_id, [])
        p = sum(1 for r in rows if r.passed)
        f = len(rows) - p
        outcomes = {}
        for r in rows:
            outcomes[r.outcome] = outcomes.get(r.outcome, 0) + 1
        outcome_str = ", ".join(f"{k}:{v}" for k, v in sorted(outcomes.items()))
        color = _GREEN if f == 0 else _RED
        lines.append(_c(color, f"  {task_id:<32} {p:>5}  {f:>5}  {outcome_str}"))

    return "\n".join(lines)


def render_run_brief(result: RunResult) -> str:
    """One-line summary of a run."""
    m = result.manifest
    counts = result.outcome_counts()
    total = sum(counts.values())
    passed = counts.get("pass", 0)
    return (f"{m.run_id}  {m.variant_id}  {len(m.task_ids)}t x{m.replicates}  "
            f"{passed}/{total} ({passed/total:.0%})" if total else m.run_id)


# ---------------------------------------------------------------------------
# Run listing (store rows)
# ---------------------------------------------------------------------------

def render_run_list(rows) -> str:
    """Render a list of RunRow objects."""
    if not rows:
        return "  (no runs recorded)"
    lines = [f"  {'run_id':<38} {'when':<16} {'variant':<10} "
             f"{'tasks':>5} {'result':>10}"]
    lines.append(f"  {'-'*38} {'-'*16} {'-'*10} {'-'*5} {'-'*10}")
    for r in rows:
        lines.append(f"  {r.line()}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Trajectory diff (side-by-side)
# ---------------------------------------------------------------------------

def render_trajectory_diff(d: diff.TrajectoryDiff, left: str = "baseline",
                           right: str = "candidate", width: int = 46) -> str:
    """Render a trajectory diff as text."""
    return "\n".join(diff.side_by_side(d, left, right, width))


# ---------------------------------------------------------------------------
# Behaviour classes
# ---------------------------------------------------------------------------

def render_classes(classes: list[diff.TrajectoryClass],
                   level: str = "semantic") -> str:
    """One line per behaviour cluster."""
    lines = diff.explain_classes(classes, level)
    return "\n".join("  " + ln for ln in lines) if lines else "  (no data)"


# ---------------------------------------------------------------------------
# Variant / policy info
# ---------------------------------------------------------------------------

def render_variant_info(variant) -> str:
    """Show a variant's policy flags and blast radius."""
    from . import policy, suite
    lines = []
    lines.append(_c(_BOLD, f"Variant: {variant.id}"))
    lines.append(f"  model:  {variant.model}")
    lines.append(f"  effort: {variant.effort}")
    lines.append(f"  agent:  {variant.agent}")
    lines.append(f"  source: {variant.source or '(inline)'}")
    lines.append(f"  digest: {variant.digest()[:12]}")
    lines.append("")

    present = sorted(variant.policy())
    absent = sorted(set(policy.ALL_FLAGS) - set(present))
    lines.append(f"  {len(present)}/{len(policy.ALL_FLAGS)} operating rules present")
    if absent:
        lines.append(_c(_YELLOW, f"  missing: {', '.join(absent)}"))
    lines.append("")

    flag_tasks = suite.policy_flags()
    if present:
        lines.append("  blast radius:")
        for flag in present:
            tasks = flag_tasks.get(flag, [])
            if tasks:
                lines.append(f"    {flag}: {', '.join(tasks)}")
        lines.append("")

    problems = policy.check_prompt(variant.system_prompt)
    if problems:
        lines.append(_c(_YELLOW, "  warnings:"))
        for p in problems:
            lines.append(f"    {p}")
    else:
        lines.append(_c(_GREEN, "  all expected phrases found"))

    return "\n".join(lines)
