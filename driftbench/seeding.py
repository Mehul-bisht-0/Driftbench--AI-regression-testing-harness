"""Deterministic seeding.

Two honest caveats up front, because they shape the whole design:

1. **The model is not seedable.** ``temperature`` / ``top_p`` / ``top_k`` were
   removed from Claude Opus 5, Sonnet 5, Fable 5, and Opus 4.8/4.7 - sending them
   returns a 400. There is no sampling seed either. So you cannot make a live run
   bit-reproducible. That is not a gap in this harness; it is the reason the
   harness measures variance instead of pretending it away.

2. **The environment is fully seedable**, and that is what we control here. Mock
   tool latency, fake-service jitter, injected failures, and the scripted fixture
   agent all draw from streams derived here. Given the same seed, the environment
   an agent faces is byte-identical, so any difference between two replicates is
   attributable to the agent rather than to the harness.

Seeds are derived by hashing, not by ``hash()`` (which is salted per process) and
not by sequential counters (which shift when you reorder tasks). Adding a task to
the suite therefore does not perturb the seeds of existing tasks.

RNG use goes through named substreams for the same reason: if all draws came from
one sequential stream, inserting a single ``random()`` call anywhere would change
every value after it, and last week's recorded trajectories would stop replaying.
"""

from __future__ import annotations

import hashlib
import random
from typing import Iterable

MASK64 = (1 << 64) - 1


def seed_from(*parts: object) -> int:
    """Stable 64-bit seed from any tuple of values.

    Deterministic across processes, Python versions, and platforms - unlike the
    builtin ``hash()``, which is randomised per interpreter by PYTHONHASHSEED.
    """
    h = hashlib.blake2b(digest_size=8)
    for p in parts:
        h.update(repr(p).encode("utf-8"))
        h.update(b"\x1f")
    return int.from_bytes(h.digest(), "big") & MASK64


def derive_seed(master_seed: int, variant_id: str, task_id: str, replicate: int) -> int:
    """The seed for one (variant, task, replicate) cell - the *agent's* stream.

    Deliberately *excludes* the run id: two runs launched with the same master
    seed face the same environment, which is what makes cross-run comparison a
    comparison of agents rather than of dice.

    It deliberately *includes* the variant id, because anything the agent itself
    draws from (retry jitter, latency it chooses to spend) belongs to the agent
    under test and should not be correlated between two different agents. The
    environment is the opposite case - see ``env_seed``.
    """
    return seed_from(master_seed, variant_id, task_id, replicate)


def env_seed(master_seed: int, task_id: str, replicate: int) -> int:
    """The seed for the *world* one cell runs in - no variant id.

    This is what makes a v1-vs-v2 report a paired comparison. Replicate 7 of a
    task with a 40%-failure-rate tool gets the same fault schedule under both
    variants, so a task that moves between runs moved because of the change you
    made, not because the dice came up differently for the candidate. Include the
    variant here and every flaky task contributes noise to every comparison, with
    no way to tell that noise from a regression.
    """
    return seed_from(master_seed, task_id, replicate)


class SeededRandom:
    """A random source that is stable under code change.

    ``substream(label)`` returns an independent generator keyed by label, so a new
    draw added in one code path cannot perturb values drawn in another.
    """

    def __init__(self, seed: int, label: str = "root") -> None:
        self.seed = seed
        self.label = label
        self._rng = random.Random(seed_from(seed, label))
        self._children: dict[str, "SeededRandom"] = {}

    def substream(self, label: str) -> "SeededRandom":
        if label not in self._children:
            self._children[label] = SeededRandom(self.seed, f"{self.label}/{label}")
        return self._children[label]

    # --- draws -----------------------------------------------------------
    def random(self) -> float:
        return self._rng.random()

    def randint(self, a: int, b: int) -> int:
        return self._rng.randint(a, b)

    def choice(self, seq):
        return self._rng.choice(list(seq))

    def shuffled(self, seq) -> list:
        items = list(seq)
        self._rng.shuffle(items)
        return items

    def chance(self, p: float) -> bool:
        """True with probability p. p<=0 and p>=1 short-circuit without consuming
        the stream, so setting a jitter knob to 0 leaves other draws untouched."""
        if p <= 0.0:
            return False
        if p >= 1.0:
            return True
        return self._rng.random() < p

    def __repr__(self) -> str:
        return f"SeededRandom(seed={self.seed}, label={self.label!r})"


def suite_digest(task_ids: Iterable[str], extra: Iterable[str] = ()) -> str:
    """Identity of a suite. Runs with different suite digests are not comparable
    apples-to-apples, and the reporter says so rather than silently diffing."""
    h = hashlib.blake2b(digest_size=10)
    for t in sorted(task_ids):
        h.update(t.encode("utf-8"))
        h.update(b"\x1f")
    for e in sorted(extra):
        h.update(e.encode("utf-8"))
        h.update(b"\x1e")
    return h.hexdigest()
