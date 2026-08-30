"""Trajectory diffing: global alignment over canonical step tokens.

Hard core #1. Two runs of the same task are two sequences of steps of different
lengths, and the useful questions are all alignment questions: where did they
first diverge, is this "same plan with a different argument" or "a different plan",
how far apart are they on a scale you can threshold.

Needleman-Wunsch (global alignment, not local) because the whole episode matters -
a trajectory that gets the right answer after three extra calls is not "mostly
identical with a good local match", it is a worse trajectory, and a global score
should say so.

The cost model is the design content. Substituting one tool for another is the
expensive edit; keeping the tool and changing an argument is cheap; inserting or
deleting a step sits between them. So an agent that searched for "refund policy"
instead of "refund rules" scores as nearly identical, and one that called
``issue_refund`` where the baseline called ``escalate_to_human`` scores as far
away - which is the ordering a human reviewer would give.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from . import canon
from .types import Trajectory

# Edit costs. Tuned so "same tools, different args" ranks nearer than "one extra
# call", which in turn ranks nearer than "different tool".
COST_MATCH = 0.0
COST_SUB_ARGS = 1.0   # same tool, arguments differ
COST_GAP = 1.5        # a step present in one run and absent in the other
COST_SUB_TOOL = 2.0   # different tool at the same position
COST_SUB_KIND = 3.0   # different kind of step entirely (call vs final vs error)


def token_kind(token: str) -> str:
    return token.split(":", 1)[0] if ":" in token else token


def token_tool(token: str) -> Optional[str]:
    """The tool name inside a canonical token, if it has one."""
    kind = token_kind(token)
    if kind == "call":
        rest = token[5:]
        return rest.split("(", 1)[0]
    if kind == "result":
        parts = token.split(":")
        return parts[1] if len(parts) > 2 else None
    return None

def sub_cost(a: str, b: str) -> float:
    if a == b:
        return COST_MATCH
    ka, kb = token_kind(a), token_kind(b)
    if ka != kb:
        return COST_SUB_KIND
    ta, tb = token_tool(a), token_tool(b)
    if ta is not None and ta == tb:
        return COST_SUB_ARGS
    if ka in ("call", "result"):
        return COST_SUB_TOOL
    return COST_SUB_ARGS


@dataclass
class Op:
    kind: str  # match | sub | ins | del
    i: Optional[int]  # index in a
    j: Optional[int]  # index in b
    a: str = ""
    b: str = ""

    def symbol(self) -> str:
        return {"match": " ", "sub": "~", "ins": "+", "del": "-"}[self.kind]


def align(a: Sequence[str], b: Sequence[str]) -> tuple[list[Op], float]:
    """Needleman-Wunsch. Returns the edit script and its total cost.

    O(len(a) * len(b)) in time and memory. Trajectories here are tens of steps,
    so the quadratic table is a few kilobytes and exactness is worth more than
    the banded approximation that a million-token diff would need.
    """
    n, m = len(a), len(b)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + COST_GAP
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + COST_GAP
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = min(
                dp[i - 1][j - 1] + sub_cost(a[i - 1], b[j - 1]),
                dp[i - 1][j] + COST_GAP,
                dp[i][j - 1] + COST_GAP,
            )

    ops: list[Op] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            c = sub_cost(a[i - 1], b[j - 1])
            if abs(dp[i][j] - (dp[i - 1][j - 1] + c)) < 1e-9:
                ops.append(Op("match" if c == COST_MATCH else "sub", i - 1, j - 1,
                              a[i - 1], b[j - 1]))
                i, j = i - 1, j - 1
                continue
        if i > 0 and abs(dp[i][j] - (dp[i - 1][j] + COST_GAP)) < 1e-9:
            ops.append(Op("del", i - 1, None, a[i - 1], ""))
            i -= 1
            continue
        ops.append(Op("ins", None, j - 1, "", b[j - 1]))
        j -= 1
    ops.reverse()
    return ops, dp[n][m]

@dataclass
class TrajectoryDiff:
    ops: list[Op]
    cost: float
    a_len: int
    b_len: int
    level: str = canon.DEFAULT_LEVEL

    @property
    def identical(self) -> bool:
        return all(op.kind == "match" for op in self.ops)

    @property
    def divergence(self) -> float:
        """0.0 identical, 1.0 nothing in common. Normalised by the worst possible
        cost for these two lengths, so a 3-step and a 30-step trajectory are still
        scored on the same scale."""
        worst = COST_GAP * (self.a_len + self.b_len)
        return 0.0 if worst == 0 else min(1.0, self.cost / worst)

    @property
    def similarity(self) -> float:
        return 1.0 - self.divergence

    def first_divergence(self) -> Optional[Op]:
        for op in self.ops:
            if op.kind != "match":
                return op
        return None

    def first_divergence_at(self) -> Optional[int]:
        """Step index (in the baseline) where the two runs part company."""
        op = self.first_divergence()
        if op is None:
            return None
        if op.i is not None:
            return op.i
        return op.j

    def explain_first(self) -> str:
        op = self.first_divergence()
        if op is None:
            return "identical trajectories"
        at = self.first_divergence_at()
        if op.kind == "sub":
            return f"step {at}: {_short(op.a)} became {_short(op.b)}"
        if op.kind == "ins":
            return f"step {at}: extra {_short(op.b)}"
        return f"step {at}: missing {_short(op.a)}"

    def counts(self) -> dict[str, int]:
        out = {"match": 0, "sub": 0, "ins": 0, "del": 0}
        for op in self.ops:
            out[op.kind] += 1
        return out

    def render(self, width: int = 46) -> list[str]:
        lines = []
        for op in self.ops:
            left = _short(op.a, width) if op.a else ""
            right = _short(op.b, width) if op.b else ""
            lines.append(f" {op.symbol()} {left:<{width}} {right}")
        return lines

def _arg_values(blob: str) -> list[str]:
    try:
        parsed = json.loads(blob)
    except Exception:
        return [blob] if blob else []
    if not isinstance(parsed, dict):
        return [str(parsed)]
    return [str(v) for v in parsed.values() if v not in (None, "", {}, [])]


def _short(token: str, width: int = 46) -> str:
    """A canonical token squeezed into one readable cell.

    ``call:read_file({"path":"logs/app.log"})`` renders as
    ``read_file(logs/app.log)``: the tool plus the arguments that distinguish this
    call from another of the same tool, which is what someone scanning a diff
    column is actually reading for.
    """
    kind = token_kind(token)
    if kind == "call":
        tool, _, rest = token[5:].partition("(")
        blob = rest[:-1] if rest.endswith(")") else rest
        vals = _arg_values(blob)
        label = f"{tool}({', '.join(vals)})" if vals else tool
    elif kind == "result":
        parts = token.split(":")
        label = f"-> {parts[1]} {parts[2]}" if len(parts) > 2 else token
    else:
        label = token
    return label if len(label) <= width else label[:width - 2] + ".."


def diff_tokens(a: Sequence[str], b: Sequence[str],
                level: str = canon.DEFAULT_LEVEL) -> TrajectoryDiff:
    ops, cost = align(list(a), list(b))
    return TrajectoryDiff(ops=ops, cost=cost, a_len=len(a), b_len=len(b), level=level)


def diff_trajectories(a: Trajectory, b: Trajectory,
                      level: str = canon.DEFAULT_LEVEL) -> TrajectoryDiff:
    """Diff two episodes at the given canonical level.

    Tokens are recomputed from the steps rather than read off ``step.canon``: the
    stored tokens were written at whatever level their run used, and comparing a
    ``shape`` token against a ``semantic`` one is a silent category error that
    would show every pair as maximally different.
    """
    return diff_tokens(canon.canon_sequence(a, level),
                       canon.canon_sequence(b, level), level)

@dataclass
class ToolDelta:
    """Which tools each run used, ignoring order and count-matched pairs."""

    added: dict[str, int] = field(default_factory=dict)    # extra in b
    removed: dict[str, int] = field(default_factory=dict)  # present only in a
    kept: dict[str, int] = field(default_factory=dict)

    @property
    def empty(self) -> bool:
        return not self.added and not self.removed

    def summary(self) -> str:
        if self.empty:
            return "same tools, same counts"
        bits = []
        if self.added:
            bits.append("+" + ", ".join(f"{t}x{n}" if n > 1 else t
                                       for t, n in sorted(self.added.items())))
        if self.removed:
            bits.append("-" + ", ".join(f"{t}x{n}" if n > 1 else t
                                       for t, n in sorted(self.removed.items())))
        return "; ".join(bits)


def tool_delta(a: Trajectory, b: Trajectory) -> ToolDelta:
    """Order-free comparison of tool usage.

    Complements the alignment rather than duplicating it: a run that makes the
    same three calls in a different order has a large edit script and an empty
    tool delta, and the pair of facts says "same tools, reordered" - which is a
    different bug report from "called a different tool".
    """
    ca, cb = Counter(a.tool_names()), Counter(b.tool_names())
    return ToolDelta(
        added={t: n for t, n in sorted((cb - ca).items())},
        removed={t: n for t, n in sorted((ca - cb).items())},
        kept={t: min(ca[t], cb[t]) for t in sorted(set(ca) & set(cb))},
    )


def side_by_side(d: TrajectoryDiff, left: str = "baseline",
                 right: str = "candidate", width: int = 46) -> list[str]:
    """The diff as text, with a header and the legend a reader needs once."""
    head = [f"   {left:<{width}} {right}",
            f"   {'-' * width} {'-' * min(width, len(right) + 8)}"]
    body = d.render(width)
    tail = [f"   cost {d.cost:.1f}  divergence {d.divergence:.3f}  "
            f"({d.level}; ~ substitution, + extra, - missing)"]
    return head + body + tail

@dataclass
class TrajectoryClass:
    """One distinct behaviour, and every replicate that produced it."""

    digest: str
    tokens: list[str] = field(default_factory=list)
    members: list[int] = field(default_factory=list)  # replicate indices
    representative: Optional[Trajectory] = None

    @property
    def size(self) -> int:
        return len(self.members)

    def plan(self) -> list[str]:
        return [token_tool(t) or "?" for t in self.tokens if token_kind(t) == "call"]

    def label(self) -> str:
        return " -> ".join(self.plan()) or "(no tool calls)"

    def error_results(self) -> int:
        return sum(1 for t in self.tokens
                   if t.startswith("result:") and t.endswith(":err"))

    def traits(self) -> str:
        """What else marks this class out, beyond the plan.

        Two classes can share a first divergence and still be different runs -
        "retried once and recovered" and "retried once and gave up" both start by
        inserting the same extra call. The traits are what tells them apart in a
        one-line summary.
        """
        bits = []
        n = self.error_results()
        if n:
            bits.append(f"{n} tool error{'s' if n > 1 else ''}")
        stop = getattr(self.representative, "stop_reason", "") or ""
        if stop and stop != "end_turn":
            bits.append(f"stop={stop}")
        return ", ".join(bits)


def cluster(trajs: Iterable[Trajectory],
            level: str = canon.DEFAULT_LEVEL) -> list[TrajectoryClass]:
    """Group runs that behaved identically.

    Keyed on the canonical digest at ``level``, recomputed here rather than read
    from ``traj.digest`` for the same reason ``diff_trajectories`` recomputes
    tokens. Sorted by descending size then first replicate: index 0 is the modal
    behaviour, everything after it is drift, and the order does not shuffle
    between reports.
    """
    groups: dict[str, TrajectoryClass] = {}
    for traj in trajs:
        digest = canon.trajectory_digest(traj, level)
        cls = groups.get(digest)
        if cls is None:
            cls = groups[digest] = TrajectoryClass(
                digest=digest, tokens=canon.canon_sequence(traj, level),
                representative=traj)
        cls.members.append(traj.replicate)
    for cls in groups.values():
        cls.members.sort()
    return sorted(groups.values(), key=lambda c: (-c.size, c.members[0]))


def class_counts(classes: Sequence[TrajectoryClass]) -> list[int]:
    """Sizes only - the input ``stats.normalized_entropy`` wants."""
    return [c.size for c in classes]


def modal(classes: Sequence[TrajectoryClass]) -> Optional[TrajectoryClass]:
    return classes[0] if classes else None

def explain_classes(classes: Sequence[TrajectoryClass],
                    level: str = canon.DEFAULT_LEVEL) -> list[str]:
    """One line per behaviour: how often it happened, and for the minority
    classes, where it first departs from the modal one."""
    if not classes:
        return []
    base = classes[0]
    total = sum(c.size for c in classes)

    def tail(c: TrajectoryClass) -> str:
        t = c.traits()
        return f" [{t}]" if t else ""

    lines = [f"{base.size}/{total} modal: {base.label()}{tail(base)}"]
    for c in classes[1:]:
        why = diff_tokens(base.tokens, c.tokens, level).explain_first()
        lines.append(f"{c.size}/{total} {why}{tail(c)}")
    return lines


def divergence_matrix(trajs: Sequence[Trajectory],
                      level: str = canon.DEFAULT_LEVEL) -> list[list[float]]:
    """Symmetric pairwise divergence. O(n^2) alignments over n replicates."""
    seqs = [canon.canon_sequence(t, level) for t in trajs]
    n = len(seqs)
    out = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            out[i][j] = out[j][i] = diff_tokens(seqs[i], seqs[j], level).divergence
    return out


def mean_pairwise_divergence(trajs: Sequence[Trajectory],
                             level: str = canon.DEFAULT_LEVEL) -> float:
    """How far apart a set of identical-input runs is, on the 0-1 scale.

    Deliberately separate from entropy over classes: entropy counts how *many*
    behaviours appeared, this says how *different* they were. Twenty runs that
    differ by one search argument and twenty that take unrelated paths both score
    high entropy; only the second scores high here.
    """
    rows = list(trajs)
    if len(rows) < 2:
        return 0.0
    m = divergence_matrix(rows, level)
    pairs = [m[i][j] for i in range(len(m)) for j in range(i + 1, len(m))]
    return sum(pairs) / len(pairs)


def medoid(trajs: Sequence[Trajectory],
           level: str = canon.DEFAULT_LEVEL) -> Optional[Trajectory]:
    """The run least unlike all the others - the one to put in a report.

    An average trajectory does not exist, so the honest summary of twenty runs is
    one actual run, picked for being central rather than for being first.
    """
    rows = list(trajs)
    if not rows:
        return None
    totals = [sum(row) for row in divergence_matrix(rows, level)]
    return rows[min(range(len(rows)), key=lambda i: (totals[i], rows[i].replicate))]
