"""The mock world: all state an agent can observe or mutate.

Every tool in this harness reads and writes a ``World``. Nothing touches the real
filesystem, network, or clock. Two consequences that matter:

* Runs are free, so N=20 replicates across 40 tasks is a coffee break, not a
  budget request.
* Given the same seed the environment is byte-identical, so any difference
  between two replicates is the agent's, not the harness's.

Side effects are appended to logs (``sent_emails``, ``refunds``, ``deleted_files``
...) rather than applied destructively, because assertions need to inspect what
the agent *tried* to do, including the things you hope it did not do.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Optional

from . import fixtures
from .seeding import SeededRandom


@dataclass
class Fault:
    """Injected failure for one tool.

    ``fail_first_n``   deterministically fail the first N calls, then succeed.
                       Used by the error-recovery tasks: the agent must retry.
    ``flake_p``        fail each call with this probability, drawn from the seeded
                       stream. Used to create *honest* flakiness that the
                       flakiness report has to detect.
    ``message``        the error text handed back to the agent.
    ``permanent``      never recover; the agent's correct move is to stop and say
                       so rather than retry forever.
    """

    fail_first_n: int = 0
    flake_p: float = 0.0
    message: str = "upstream unavailable"
    permanent: bool = False


@dataclass
class World:
    rng: SeededRandom
    files: dict[str, str] = field(default_factory=dict)
    docs: dict[str, dict] = field(default_factory=dict)
    tickets: dict[str, dict] = field(default_factory=dict)
    orders: dict[str, dict] = field(default_factory=dict)
    tables: dict[str, list[dict]] = field(default_factory=dict)
    http: dict[str, str] = field(default_factory=dict)

    # Side-effect ledgers. Assertions read these.
    sent_emails: list[dict] = field(default_factory=list)
    refunds: list[dict] = field(default_factory=list)
    deleted_files: list[str] = field(default_factory=list)
    written_files: list[str] = field(default_factory=list)
    ticket_updates: list[dict] = field(default_factory=list)
    escalations: list[dict] = field(default_factory=list)

    faults: dict[str, Fault] = field(default_factory=dict)
    call_counts: dict[str, int] = field(default_factory=dict)
    # Facts injected straight into the task prompt; tasks assert the agent used
    # these instead of calling a tool to rediscover them.
    given: dict[str, Any] = field(default_factory=dict)

    def bump(self, tool: str) -> int:
        """Record a call and return its 1-based index for this tool."""
        self.call_counts[tool] = self.call_counts.get(tool, 0) + 1
        return self.call_counts[tool]

    def check_fault(self, tool: str, nth: int) -> Optional[str]:
        """Return an error message if this call should fail, else None."""
        fault = self.faults.get(tool)
        if fault is None:
            return None
        if fault.permanent:
            return fault.message
        if nth <= fault.fail_first_n:
            return fault.message
        if fault.flake_p and self.rng.substream(f"fault/{tool}/{nth}").chance(fault.flake_p):
            return fault.message
        return None

    def total_calls(self) -> int:
        return sum(self.call_counts.values())


def build_world(seed: int, faults: Optional[dict[str, Fault]] = None,
                given: Optional[dict] = None,
                overlay: Optional[dict[str, Any]] = None) -> World:
    """Fresh world from the fixtures.

    Deep-copied so a task that mutates state cannot leak into the next replicate -
    the single most common source of "why does this only fail when run second".
    """
    world = World(
        rng=SeededRandom(seed, "world"),
        files=copy.deepcopy(fixtures.FILES),
        docs=copy.deepcopy(fixtures.DOCS),
        tickets=copy.deepcopy(fixtures.TICKETS),
        orders=copy.deepcopy(fixtures.ORDERS),
        tables=copy.deepcopy(fixtures.TABLES),
        http=copy.deepcopy(fixtures.HTTP),
        faults=dict(faults or {}),
        given=dict(given or {}),
    )
    for key, value in (overlay or {}).items():
        if not hasattr(world, key):
            raise KeyError(f"world overlay targets unknown field {key!r}")
        current = getattr(world, key)
        if isinstance(current, dict) and isinstance(value, dict):
            current.update(copy.deepcopy(value))
        else:
            setattr(world, key, copy.deepcopy(value))
    return world
