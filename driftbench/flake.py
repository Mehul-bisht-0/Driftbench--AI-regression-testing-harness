"""Flakiness: variance across runs that were supposed to be identical.

Hard core #2. Every task in a run is executed N times against a bit-identical
environment (same seed, same fixtures, same faults), so any variation left is the
agent's own. Three numbers, because "flaky" hides three different problems:

* **outcome flake** - ``4p(1-p)`` over pass/fail. What a CI owner feels.
* **behaviour classes** - how many distinct canonical trajectories appeared, and
  the normalised entropy over their sizes. What a debugger needs.
* **spread** - mean pairwise divergence. Whether those classes differ by one
  argument or take unrelated paths.

The case this module exists for is **latent flakiness**: outcome flake 0.0, three
behaviour classes. The task is green every single time and the agent is doing
something different every single time. Nobody's dashboard shows this, and it is
the state a real regression arrives from - the day one of those paths stops
working, the task starts failing "for no reason" and the change that broke it
shipped weeks earlier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from . import canon, diff, stats
from .results import ReplicateResult

# Worst first. Ordering is used for sorting reports, so it is data, not a comment.
BANDS = ("chaotic", "flaky", "jittery", "stable")

BAND_MEANING = {
    "stable": "one behaviour, one outcome - safe to gate on",
    "jittery": "same outcome every run, different route - latent risk",
    "flaky": "the outcome itself moves between runs",
    "chaotic": "close to a coin flip, or a new behaviour almost every run",
}

# 4p(1-p) >= 0.75 means the pass rate sits in [0.25, 0.75].
CHAOTIC_FLAKE = 0.75

@dataclass
class TaskFlake:
    """One task's variance across its replicates."""

    task_id: str
    n: int = 0
    passes: int = 0
    outcomes: dict[str, int] = field(default_factory=dict)
    classes: list = field(default_factory=list)  # diff.TrajectoryClass
    entropy: float = 0.0
    spread: float = 0.0
    flake: float = 0.0
    ci: Optional[stats.Interval] = None
    calls: list[int] = field(default_factory=list)  # tool calls per replicate
    level: str = canon.DEFAULT_LEVEL

    @property
    def pass_rate(self) -> float:
        return self.passes / self.n if self.n else 0.0

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    @property
    def call_spread(self) -> int:
        return max(self.calls) - min(self.calls) if self.calls else 0

    @property
    def latent(self) -> bool:
        """Unanimous outcome, more than one behaviour. The interesting case."""
        return self.flake == 0.0 and self.n_classes > 1

    def band(self) -> str:
        if self.n <= 1:
            return "stable"  # one replicate cannot show variance; say so, not "unknown"
        if self.flake >= CHAOTIC_FLAKE or self.n_classes == self.n >= 5:
            return "chaotic"
        if self.flake > 0.0:
            return "flaky"
        if self.n_classes > 1:
            return "jittery"
        return "stable"

    def score(self) -> float:
        """0 stable, 1 unusable.

        Outcome variance dominates because it is what blocks a merge; behaviour
        variance is weighted lower but not zero because it is the leading
        indicator. The behaviour term takes whichever of entropy and spread is
        worse - many near-identical routes and two wildly different ones are both
        worth a look, and neither should be able to hide behind the other.
        """
        return min(1.0, 0.7 * self.flake + 0.3 * max(self.entropy, self.spread))

    def headline(self) -> str:
        bits = [f"{self.passes}/{self.n} pass"]
        if self.ci is not None:
            bits.append(f"CI {self.ci.pct()}")
        bits.append(f"{self.n_classes} behaviour"
                    f"{'s' if self.n_classes != 1 else ''}")
        if self.call_spread:
            bits.append(f"{min(self.calls)}-{max(self.calls)} calls")
        return f"{self.band().upper():<8} {self.task_id}  " + ", ".join(bits)

    def explain(self) -> list[str]:
        """The headline, then one line per distinct behaviour."""
        lines = [self.headline()]
        if self.latent:
            lines.append("    passes every run and takes a different route: "
                         "the failure is already there, just not triggered yet")
        lines.extend("    " + ln for ln in diff.explain_classes(self.classes,
                                                               self.level))
        off = [f"{o}x{c}" for o, c in sorted(self.outcomes.items()) if o != "pass"]
        if off:
            lines.append("    non-pass outcomes: " + ", ".join(off))
        return lines

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id, "n": self.n, "passes": self.passes,
            "pass_rate": self.pass_rate, "outcomes": dict(self.outcomes),
            "band": self.band(), "score": self.score(), "flake": self.flake,
            "entropy": self.entropy, "spread": self.spread,
            "n_classes": self.n_classes, "latent": self.latent,
            "calls": list(self.calls), "level": self.level,
            "ci": [self.ci.lo, self.ci.hi] if self.ci else None,
            "classes": [{"digest": c.digest, "size": c.size,
                         "members": list(c.members), "plan": c.plan(),
                         "traits": c.traits()} for c in self.classes],
        }


def analyse_task(task_id: str, rows: Sequence[ReplicateResult],
                 level: str = canon.DEFAULT_LEVEL) -> TaskFlake:
    trajs = [r.trajectory for r in rows]
    classes = diff.cluster(trajs, level)
    outcomes: dict[str, int] = {}
    for r in rows:
        outcomes[r.outcome] = outcomes.get(r.outcome, 0) + 1
    passes = sum(1 for r in rows if r.passed)
    n = len(rows)
    return TaskFlake(
        task_id=task_id, n=n, passes=passes, outcomes=outcomes, classes=classes,
        entropy=stats.normalized_entropy(diff.class_counts(classes)),
        spread=diff.mean_pairwise_divergence(trajs, level),
        flake=stats.outcome_flake(passes, n), ci=stats.wilson(passes, n),
        calls=[len(t.tool_calls()) for t in trajs], level=level)

@dataclass
class FlakeReport:
    run_id: str = ""
    variant_id: str = ""
    replicates: int = 0
    level: str = canon.DEFAULT_LEVEL
    tasks: list[TaskFlake] = field(default_factory=list)

    def by_id(self) -> dict[str, TaskFlake]:
        return {t.task_id: t for t in self.tasks}

    def by_band(self) -> dict[str, list[TaskFlake]]:
        out: dict[str, list[TaskFlake]] = {b: [] for b in BANDS}
        for t in self.tasks:
            out[t.band()].append(t)
        return out

    def unstable(self) -> list[TaskFlake]:
        """Everything that is not perfectly repeatable, worst first."""
        return sorted((t for t in self.tasks if t.band() != "stable"),
                      key=lambda t: (-t.score(), t.task_id))

    def latent(self) -> list[TaskFlake]:
        return [t for t in self.tasks if t.latent]

    def worst(self, n: int = 5) -> list[TaskFlake]:
        return self.unstable()[:n]

    def noise_floor(self, confidence: float = 0.95) -> float:
        """The pass-rate move a rerun of this same variant could produce alone.

        Median half-width of the per-task Wilson intervals. At 20 replicates and
        near-perfect rates that is about 0.08, so a task drifting 20/20 -> 18/20
        is not news; ``compare`` uses this to keep noise out of the regression
        list instead of hard-coding a threshold that has no relationship to how
        many replicates were actually run.
        """
        halves = [stats.wilson(t.passes, t.n, confidence).width / 2
                  for t in self.tasks if t.n]
        return stats.median(halves)

    def summary(self) -> str:
        bands = self.by_band()
        parts = [f"{len(bands[b])} {b}" for b in BANDS if bands[b]]
        line = f"{len(self.tasks)} tasks x {self.replicates} replicates"
        if parts:
            line += ": " + ", ".join(parts)
        latent = len(self.latent())
        if latent:
            line += f"; {latent} latent (green but non-deterministic)"
        return line + f"; noise floor +/-{self.noise_floor() * 100:.1f}pp"

    def to_dict(self) -> dict:
        return {"run_id": self.run_id, "variant_id": self.variant_id,
                "replicates": self.replicates, "level": self.level,
                "noise_floor": self.noise_floor(),
                "tasks": [t.to_dict() for t in self.tasks]}


def analyse(run, level: str = canon.DEFAULT_LEVEL) -> FlakeReport:
    """Per-task variance for a whole run, in the run's own task order."""
    grouped = run.by_task()
    return FlakeReport(
        run_id=run.manifest.run_id, variant_id=run.manifest.variant_id,
        replicates=run.manifest.replicates, level=level,
        tasks=[analyse_task(tid, rows, level) for tid, rows in grouped.items()])


def noise_floor(run, confidence: float = 0.95,
                level: str = canon.DEFAULT_LEVEL) -> float:
    return analyse(run, level).noise_floor(confidence)
