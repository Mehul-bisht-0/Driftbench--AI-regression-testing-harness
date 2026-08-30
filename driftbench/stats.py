"""Statistics, stdlib only.

Every number the reports quote is computed here, and every one of them exists to
answer a question that "6 of 41 went red" cannot:

* **Wilson interval** - 3/5 passes is not 60%, it is somewhere between 23% and
  88%. Printing 60% from five runs is the most common way an eval overstates what
  it knows.
* **Fisher's exact test** - the right test for two small binomial samples. A chi2
  approximation on n=20 with zero failures is not trustworthy, and pass counts are
  exactly the case Fisher's test was designed for.
* **Benjamini-Hochberg** - 41 tasks means 41 hypothesis tests. At p<0.05 you
  expect two false alarms per run by construction, and two red cells that are
  noise will burn a morning.
* **Cohen's kappa** - raw agreement between judge and hand labels is inflated by
  the base rate. If 90% of trajectories are good, a judge that says "good" every
  time scores 90% agreement and is worthless; kappa says so.
* **Normalised entropy** - how many distinct behaviours a task produces across
  identical runs, on a 0-1 scale that is comparable between tasks.

No scipy. The distributions here are small and discrete, so exact computation with
``math.lgamma`` is both faster and more accurate than an approximation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

# z for a two-sided normal interval, by confidence level.
Z = {0.80: 1.2815515655446004, 0.90: 1.6448536269514722,
     0.95: 1.959963984540054, 0.99: 2.5758293035489004}


@dataclass
class Interval:
    lo: float
    hi: float

    @property
    def width(self) -> float:
        return self.hi - self.lo

    def contains(self, x: float) -> bool:
        return self.lo <= x <= self.hi

    def pct(self, digits: int = 0) -> str:
        return f"{self.lo * 100:.{digits}f}-{self.hi * 100:.{digits}f}%"

def wilson(successes: int, n: int, confidence: float = 0.95) -> Interval:
    """Wilson score interval for a binomial proportion.

    Chosen over the normal approximation because it behaves at the edges: 20/20
    gives (0.84, 1.0) rather than the nonsense (1.0, 1.0), and 0/20 gives
    (0.0, 0.16) rather than certainty that the task can never pass.
    """
    if n <= 0:
        return Interval(0.0, 1.0)
    z = Z.get(round(confidence, 2), Z[0.95])
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return Interval(max(0.0, center - half), min(1.0, center + half))


def _log_choose(n: int, k: int) -> float:
    if k < 0 or k > n:
        return -math.inf
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1))


def hypergeom_pmf(k: int, a_total: int, b_total: int, drawn: int) -> float:
    """P(k successes) drawing ``drawn`` items from ``a_total`` + ``b_total``."""
    num = _log_choose(a_total, k) + _log_choose(b_total, drawn - k)
    den = _log_choose(a_total + b_total, drawn)
    if num == -math.inf:
        return 0.0
    return math.exp(num - den)


def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Two-tailed Fisher's exact p-value for the table [[a, b], [c, d]].

    Rows are groups (baseline, candidate); columns are outcomes (pass, fail).
    Summing every table at most as probable as the observed one is the textbook
    two-tailed definition, and with n<=100 per cell it costs nothing.
    """
    row1, row2 = a + b, c + d
    col1 = a + c
    total = row1 + row2
    if total == 0 or row1 == 0 or row2 == 0 or col1 == 0 or col1 == total:
        return 1.0
    observed = hypergeom_pmf(a, row1, row2, col1)
    tol = observed * (1 + 1e-9)
    lo = max(0, col1 - row2)
    hi = min(row1, col1)
    p = sum(prob for k in range(lo, hi + 1)
            if (prob := hypergeom_pmf(k, row1, row2, col1)) <= tol)
    return min(1.0, p)

def benjamini_hochberg(pvalues: Sequence[float], fdr: float = 0.10) -> list[bool]:
    """Which p-values survive a Benjamini-Hochberg FDR correction.

    Returns a list aligned with the input. Bonferroni would be the conservative
    choice, but on 41 tasks it hides real single-task regressions, and a
    regression report that misses regressions is worse than one with an occasional
    false alarm. FDR at 10% is the compromise: at most ~1 in 10 of the cells you
    are shown is expected to be noise.
    """
    n = len(pvalues)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: pvalues[i])
    keep = [False] * n
    cutoff = -1
    for rank, idx in enumerate(order, start=1):
        if pvalues[idx] <= fdr * rank / n:
            cutoff = rank
    for rank, idx in enumerate(order, start=1):
        if rank <= cutoff:
            keep[idx] = True
    return keep


def shannon_entropy(counts: Iterable[int]) -> float:
    """Entropy in bits of a distribution given as raw counts."""
    values = [c for c in counts if c > 0]
    total = sum(values)
    if total <= 0 or len(values) <= 1:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in values)


def normalized_entropy(counts: Iterable[int]) -> float:
    """Entropy on 0-1, normalised by the most this sample size could produce.

    The denominator is ``log2(total observations)``, not ``log2(classes seen)``:
    dividing by the classes you happened to observe would score "two behaviours,
    evenly split" and "five behaviours, evenly split" both at 1.0, throwing away
    the count that matters.

    The consequence is that the number is comparable *within* a run, where every
    task has the same replicate count, and not across runs with different
    replicate counts - which is why ``compare`` refuses to diff runs whose
    replicate counts differ rather than putting two of these side by side.
    """
    values = [c for c in counts if c > 0]
    total = sum(values)
    if total <= 1 or len(values) <= 1:
        return 0.0
    return shannon_entropy(values) / math.log2(total)


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def median(xs: Sequence[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2

@dataclass
class Agreement:
    """Judge-vs-human agreement. ``kappa`` is the number that decides whether the
    judge is usable; ``raw`` is the number people quote."""

    n: int = 0
    raw: float = 0.0
    kappa: float = 0.0
    ci: Optional[Interval] = None
    confusion: dict = None  # (human, judge) -> count
    per_class: dict = None  # class -> {precision, recall, f1, support}

    def verdict(self) -> str:
        """Landis-Koch bands, with the honest warning attached."""
        if self.n < 20:
            return "too few labels to say"
        k = self.kappa
        if k >= 0.81:
            return "almost perfect"
        if k >= 0.61:
            return "substantial - usable"
        if k >= 0.41:
            return "moderate - use as a warning signal only"
        if k >= 0.21:
            return "fair - do not gate on this judge"
        return "poor - the judge is not measuring what you labelled"


def cohens_kappa(pairs: Sequence[tuple[str, str]]) -> Agreement:
    """Agreement between two raters over the same items.

    ``pairs`` is ``(human_label, judge_label)``. Kappa corrects for the agreement
    you would get by chance from the marginal rates, which is why a judge that
    always says "good" on a 90%-good sample scores ~0 here and 0.90 on raw
    agreement.
    """
    n = len(pairs)
    if n == 0:
        return Agreement(confusion={}, per_class={})
    labels = sorted({x for pair in pairs for x in pair})
    confusion = {(h, j): 0 for h in labels for j in labels}
    for h, j in pairs:
        confusion[(h, j)] += 1
    agree = sum(confusion[(l, l)] for l in labels)
    raw = agree / n
    expected = sum((sum(confusion[(l, j)] for j in labels) / n)
                   * (sum(confusion[(h, l)] for h in labels) / n)
                   for l in labels)
    kappa = 0.0 if expected >= 1.0 else (raw - expected) / (1 - expected)

    per_class = {}
    for l in labels:
        tp = confusion[(l, l)]
        judged = sum(confusion[(h, l)] for h in labels)
        actual = sum(confusion[(l, j)] for j in labels)
        precision = tp / judged if judged else 0.0
        recall = tp / actual if actual else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall else 0.0)
        per_class[l] = {"precision": precision, "recall": recall, "f1": f1,
                        "support": actual}
    return Agreement(n=n, raw=raw, kappa=kappa, ci=wilson(agree, n),
                     confusion=confusion, per_class=per_class)

def bootstrap_ci(values: Sequence[float], statistic=mean, iterations: int = 2000,
                 confidence: float = 0.95, seed: int = 0) -> Interval:
    """Percentile bootstrap CI, with a seeded resampler.

    Seeded because a confidence interval that moves when you rerun the report is
    an invitation to rerun until it says what you want.
    """
    if not values:
        return Interval(0.0, 0.0)
    from .seeding import SeededRandom
    rng = SeededRandom(seed, "bootstrap")._rng
    n = len(values)
    stats = []
    for _ in range(iterations):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        stats.append(statistic(sample))
    stats.sort()
    alpha = (1 - confidence) / 2
    lo = stats[min(len(stats) - 1, int(alpha * len(stats)))]
    hi = stats[min(len(stats) - 1, int((1 - alpha) * len(stats)))]
    return Interval(lo, hi)


def outcome_flake(passes: int, n: int) -> float:
    """4p(1-p): 0 when a task always agrees with itself, 1 at a coin flip.

    Scaled so the number reads as "how much of this task's signal is noise".
    """
    if n <= 0:
        return 0.0
    p = passes / n
    return 4 * p * (1 - p)


def proportion_delta(base_pass: int, base_n: int, cand_pass: int,
                     cand_n: int) -> dict:
    """Everything needed to judge one task's change between two runs."""
    base_rate = base_pass / base_n if base_n else 0.0
    cand_rate = cand_pass / cand_n if cand_n else 0.0
    return {
        "base_rate": base_rate,
        "cand_rate": cand_rate,
        "delta": cand_rate - base_rate,
        "p_value": fisher_exact(base_pass, base_n - base_pass,
                                cand_pass, cand_n - cand_pass),
        "base_ci": wilson(base_pass, base_n),
        "cand_ci": wilson(cand_pass, cand_n),
    }
