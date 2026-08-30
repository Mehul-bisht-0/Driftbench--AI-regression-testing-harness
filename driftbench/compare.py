"""Comparison: what changed between two runs, and which changes are real.

The report this file produces is the product. Everything else exists to make its
cells trustworthy, and the reason it is a separate module from ``stats`` is that
turning numbers into a verdict involves three judgement calls that deserve to be
visible and arguable:

1. **A change has to clear the noise floor**, which is measured from the
   baseline's own replicate-to-replicate variance rather than picked. On 20
   replicates that is around 8 percentage points; asserting a 1-cell move on 5
   replicates is how eval dashboards earn their reputation.
2. **A change has to survive multiple-comparison correction.** 41 tasks is 41
   hypothesis tests, and at p<0.05 you expect two false alarms per run by
   construction. Benjamini-Hochberg at 10% keeps single-task regressions
   detectable while bounding how much of the red you are looking at is noise.
3. **A task whose pass rate did not move can still have changed.** Same score,
   different trajectory is ``BEHAVIOR_DRIFT``: not a failure, but the thing that
   turns into one later, and the only reason to have recorded trajectories.

Two runs are comparable only if they ran the same suite the same number of times.
``compare`` refuses otherwise instead of quietly producing a table where half the
cells mean something different from the other half.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from . import canon, diff, flake, stats
from .results import ReplicateResult

VERDICTS = ("regression", "improvement", "behavior_drift", "noise", "stable")

SYMBOL = {"regression": "FAIL", "improvement": "GAIN", "behavior_drift": "DRIFT",
          "noise": "noise", "stable": "ok"}


class IncomparableRuns(ValueError):
    """Raised when the two runs cannot be put in the same table."""


def comparability(base, cand) -> list[str]:
    """Blocking reasons, in the order a user should read them. Empty means go."""
    bad = []
    bm, cm = base.manifest, cand.manifest
    if bm.suite_digest != cm.suite_digest:
        bad.append(
            f"suite changed ({bm.suite_digest[:8]} -> {cm.suite_digest[:8]}): the "
            "task set or its assertions differ, so a green cell does not mean the "
            "same thing in both runs - re-run the baseline")
    if bm.replicates != cm.replicates:
        bad.append(
            f"replicate counts differ ({bm.replicates} vs {cm.replicates}): "
            "behaviour entropy is not comparable across sample sizes")
    if bm.master_seed != cm.master_seed:
        bad.append(
            f"master seeds differ ({bm.master_seed} vs {cm.master_seed}): the two "
            "runs faced different environments, so any delta mixes the change you "
            "made with the change in fixtures")
    shared = set(bm.task_ids) & set(cm.task_ids)
    if not shared:
        bad.append("the two runs have no tasks in common")
    return bad


def _failed_assertion_names(rows: Sequence[ReplicateResult]) -> set[str]:
    return {a.name for r in rows for a in r.assertions
            if not a.passed and a.critical}


@dataclass
class TaskComparison:
    task_id: str
    base_pass: int = 0
    base_n: int = 0
    cand_pass: int = 0
    cand_n: int = 0
    delta: float = 0.0
    p_value: float = 1.0
    significant: bool = False  # after Benjamini-Hochberg across the whole suite
    base_ci: Optional[stats.Interval] = None
    cand_ci: Optional[stats.Interval] = None
    verdict: str = "stable"
    traj: Optional[diff.TrajectoryDiff] = None  # modal base vs modal candidate
    base_classes: int = 1
    cand_classes: int = 1
    base_flake: float = 0.0
    cand_flake: float = 0.0
    novel_share: float = 0.0  # candidate replicates on a route the baseline never took
    new_failures: list[str] = field(default_factory=list)
    fixed_failures: list[str] = field(default_factory=list)
    example_replicate: Optional[int] = None
    detail: str = ""  # one failing assertion's own explanation, verbatim

    @property
    def base_rate(self) -> float:
        return self.base_pass / self.base_n if self.base_n else 0.0

    @property
    def cand_rate(self) -> float:
        return self.cand_pass / self.cand_n if self.cand_n else 0.0

    @property
    def changed(self) -> bool:
        return self.verdict in ("regression", "improvement", "behavior_drift")

    def symbol(self) -> str:
        return SYMBOL[self.verdict]

    def headline(self) -> str:
        return (f"{self.symbol():<6} {self.task_id:<28} "
                f"{self.base_pass}/{self.base_n} -> {self.cand_pass}/{self.cand_n}"
                f"  {self.delta * 100:+.0f}pp  p={self.p_value:.2g}")

    def explain(self) -> list[str]:
        lines = [self.headline()]
        if self.new_failures:
            lines.append("    now failing: " + ", ".join(self.new_failures))
        if self.fixed_failures:
            lines.append("    now passing: " + ", ".join(self.fixed_failures))
        if self.detail:
            lines.append(f"    {self.detail}")
        if self.traj is not None and not self.traj.identical:
            lines.append(f"    {self.traj.explain_first()} "
                         f"(divergence {self.traj.divergence:.2f})")
        if self.verdict == "behavior_drift":
            lines.append("    same score, different route - not a failure yet")
        if self.novel_share:
            lines.append(f"    {self.novel_share:.0%} of candidate runs took a "
                         "route the baseline never took")
        if self.verdict == "noise":
            lines.append("    inside the baseline's own run-to-run variance")
        return lines

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id, "verdict": self.verdict,
            "base_pass": self.base_pass, "base_n": self.base_n,
            "cand_pass": self.cand_pass, "cand_n": self.cand_n,
            "base_rate": self.base_rate, "cand_rate": self.cand_rate,
            "delta": self.delta, "p_value": self.p_value,
            "significant": self.significant,
            "base_ci": [self.base_ci.lo, self.base_ci.hi] if self.base_ci else None,
            "cand_ci": [self.cand_ci.lo, self.cand_ci.hi] if self.cand_ci else None,
            "base_classes": self.base_classes, "cand_classes": self.cand_classes,
            "base_flake": self.base_flake, "cand_flake": self.cand_flake,
            "novel_share": self.novel_share,
            "new_failures": list(self.new_failures),
            "fixed_failures": list(self.fixed_failures),
            "example_replicate": self.example_replicate,
            "detail": self.detail,
            "divergence": self.traj.divergence if self.traj else 0.0,
            "first_divergence": self.traj.explain_first() if self.traj else "",
        }

@dataclass
class RunComparison:
    base_run_id: str = ""
    cand_run_id: str = ""
    base_variant: str = ""
    cand_variant: str = ""
    base_label: str = ""
    cand_label: str = ""
    replicates: int = 0
    fdr: float = 0.10
    noise_floor: float = 0.0
    level: str = canon.DEFAULT_LEVEL
    tasks: list[TaskComparison] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    prompt_diff: list[str] = field(default_factory=list)
    policy_removed: list[str] = field(default_factory=list)
    policy_added: list[str] = field(default_factory=list)

    def of(self, verdict: str) -> list[TaskComparison]:
        rows = [t for t in self.tasks if t.verdict == verdict]
        return sorted(rows, key=lambda t: (t.delta, t.p_value, t.task_id))

    def regressions(self) -> list[TaskComparison]:
        return self.of("regression")

    def improvements(self) -> list[TaskComparison]:
        return sorted(self.of("improvement"), key=lambda t: (-t.delta, t.task_id))

    def drift(self) -> list[TaskComparison]:
        return sorted(self.of("behavior_drift"),
                      key=lambda t: (-(t.traj.divergence if t.traj else 0.0),
                                     t.task_id))

    def noise(self) -> list[TaskComparison]:
        return self.of("noise")

    def verdict_counts(self) -> dict[str, int]:
        out = {v: 0 for v in VERDICTS}
        for t in self.tasks:
            out[t.verdict] += 1
        return out

    def ok(self) -> bool:
        """The gate. Behaviour drift is reported but does not block."""
        return not self.regressions()

    def exit_code(self) -> int:
        return 0 if self.ok() else 1

    def headline(self) -> str:
        c = self.verdict_counts()
        n = len(self.tasks)
        bits = [f"{c['regression']} of {n} tasks regressed"]
        if c["improvement"]:
            bits.append(f"{c['improvement']} improved")
        if c["behavior_drift"]:
            bits.append(f"{c['behavior_drift']} drifted behaviour")
        if c["noise"]:
            bits.append(f"{c['noise']} moved within noise")
        return (f"{self.base_label or self.base_variant} -> "
                f"{self.cand_label or self.cand_variant}: " + ", ".join(bits)
                + f" ({self.replicates} replicates, noise floor "
                  f"+/-{self.noise_floor * 100:.1f}pp, BH FDR {self.fdr:.0%})")

    def explain(self) -> list[str]:
        lines = [self.headline()]
        for w in self.warnings:
            lines.append(f"  warning: {w}")
        if self.policy_removed:
            lines.append("  prompt lost: " + ", ".join(self.policy_removed))
        if self.policy_added:
            lines.append("  prompt gained: " + ", ".join(self.policy_added))
        for group, title in (("regression", "regressions"),
                             ("improvement", "improvements"),
                             ("behavior_drift", "behaviour drift"),
                             ("noise", "moved but not significant")):
            rows = self.of(group)
            if not rows:
                continue
            lines.append("")
            lines.append(f"{title} ({len(rows)}):")
            for t in rows:
                lines.extend("  " + ln for ln in t.explain())
        return lines

    def to_dict(self) -> dict:
        return {
            "base_run_id": self.base_run_id, "cand_run_id": self.cand_run_id,
            "base_variant": self.base_variant, "cand_variant": self.cand_variant,
            "replicates": self.replicates, "fdr": self.fdr,
            "noise_floor": self.noise_floor, "level": self.level,
            "counts": self.verdict_counts(), "ok": self.ok(),
            "warnings": list(self.warnings),
            "prompt_diff": list(self.prompt_diff),
            "policy_removed": list(self.policy_removed),
            "policy_added": list(self.policy_added),
            "tasks": [t.to_dict() for t in self.tasks],
        }


def _verdict(t: TaskComparison, floor: float) -> str:
    moved = abs(t.delta) > max(floor, 1e-9)
    if t.significant and moved:
        return "regression" if t.delta < 0 else "improvement"
    if abs(t.delta) > 1e-9:
        return "noise"
    if t.traj is not None and not t.traj.identical:
        return "behavior_drift"
    if t.base_classes != t.cand_classes:
        return "behavior_drift"
    return "stable"

def _one_task(tid: str, base_rows, cand_rows, bf, cf,
              level: str) -> TaskComparison:
    bp = sum(1 for r in base_rows if r.passed)
    cp = sum(1 for r in cand_rows if r.passed)
    d = stats.proportion_delta(bp, len(base_rows), cp, len(cand_rows))
    bmod, cmod = diff.modal(bf.classes), diff.modal(cf.classes)
    seen = {c.digest for c in bf.classes}
    novel = [c for c in cf.classes if c.digest not in seen]
    novel_share = sum(c.size for c in novel) / cf.n if cf.n else 0.0
    tdiff = None
    if bmod is not None and cmod is not None:
        tdiff = diff.diff_tokens(bmod.tokens, cmod.tokens, level)
        if tdiff.identical and novel:
            # The typical run is unchanged and a minority took a new route. Diff
            # against the biggest new one, or the row says "drift" and shows
            # nothing - which is how drift gets ignored.
            tdiff = diff.diff_tokens(bmod.tokens, novel[0].tokens, level)

    base_fail, cand_fail = (_failed_assertion_names(base_rows),
                            _failed_assertion_names(cand_rows))
    example = next((r for r in cand_rows if not r.passed), None)
    detail = ""
    if example is not None:
        crit = [a for a in example.failed_assertions() if a.critical]
        if crit:
            detail = f"{crit[0].name}: {crit[0].detail}"
        elif example.trajectory.error:
            detail = f"harness error: {example.trajectory.error}"

    return TaskComparison(
        task_id=tid, base_pass=bp, base_n=len(base_rows), cand_pass=cp,
        cand_n=len(cand_rows), delta=d["delta"], p_value=d["p_value"],
        base_ci=d["base_ci"], cand_ci=d["cand_ci"], traj=tdiff,
        base_classes=bf.n_classes, cand_classes=cf.n_classes,
        base_flake=bf.flake, cand_flake=cf.flake, novel_share=novel_share,
        new_failures=sorted(cand_fail - base_fail),
        fixed_failures=sorted(base_fail - cand_fail),
        example_replicate=example.replicate if example is not None else None,
        detail=detail)


def _prompt_context(base, cand) -> dict:
    """The one-line prompt diff and the operating-rule flags it changed."""
    from . import variant as variant_mod
    bv, cv = getattr(base, "variant", None), getattr(cand, "variant", None)
    if bv is None or cv is None:
        return {}
    delta = variant_mod.policy_delta(bv, cv)
    return {"prompt_diff": variant_mod.diff(bv, cv),
            "policy_removed": delta["removed"], "policy_added": delta["added"],
            "base_label": bv.label(), "cand_label": cv.label()}

def compare(base, cand, fdr: float = 0.10, level: str = canon.DEFAULT_LEVEL,
            strict: bool = True) -> RunComparison:
    """Diff two runs task by task.

    ``strict=False`` downgrades the comparability guards to warnings carried on
    the report. It exists for the one honest use - comparing a run against an
    older baseline while you look at *behaviour* rather than scores - and every
    number in the result should be treated as indicative when it is set.
    """
    reasons = comparability(base, cand)
    if reasons and strict:
        raise IncomparableRuns("; ".join(reasons))

    base_by, cand_by = base.by_task(), cand.by_task()
    shared = [t for t in base.manifest.task_ids if t in base_by and t in cand_by]
    bf = flake.analyse(base, level).by_id()
    cf = flake.analyse(cand, level).by_id()

    floor = flake.FlakeReport(
        replicates=base.manifest.replicates, level=level,
        tasks=[bf[t] for t in shared]).noise_floor()

    rows = [_one_task(t, base_by[t], cand_by[t], bf[t], cf[t], level)
            for t in shared]
    for row, significant in zip(rows, stats.benjamini_hochberg(
            [r.p_value for r in rows], fdr)):
        row.significant = significant
        row.verdict = _verdict(row, floor)

    report = RunComparison(
        base_run_id=base.manifest.run_id, cand_run_id=cand.manifest.run_id,
        base_variant=base.manifest.variant_id, cand_variant=cand.manifest.variant_id,
        replicates=base.manifest.replicates, fdr=fdr, noise_floor=floor,
        level=level, tasks=rows, warnings=list(reasons))
    for key, value in _prompt_context(base, cand).items():
        setattr(report, key, value)

    dropped = sorted(set(base.manifest.task_ids) ^ set(cand.manifest.task_ids))
    if dropped:
        report.warnings.append(
            f"{len(dropped)} task(s) present in only one run, excluded: "
            + ", ".join(dropped[:6]) + ("..." if len(dropped) > 6 else ""))
    return report
