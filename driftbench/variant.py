"""A variant is the thing under test: prompt + model + effort + agent.

Everything a run holds constant except the tasks. The ``digest`` matters as much
as the id: two runs both labelled "v1" that used different prompt files are the
single most common way an eval lies to you, so comparison keys on the digest and
the reporter shows it next to the label.

``diff`` is deliberately line-oriented and plain. When six tasks turn red the
first question is always "what exactly changed in the prompt", and the answer
should be one screen, not a git archaeology exercise.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import canon, policy

DEFAULT_MODEL = "claude-opus-5"
DEFAULT_EFFORT = "medium"  # output_config.effort: low | medium | high | max


@dataclass
class Variant:
    id: str
    system_prompt: str
    model: str = DEFAULT_MODEL
    effort: str = DEFAULT_EFFORT
    agent: str = "scripted"
    source: str = ""  # where the prompt came from, for the report header
    notes: str = ""
    _policy: Optional[frozenset] = field(default=None, repr=False, compare=False)

    def digest(self) -> str:
        """Identity of the variant. Any of these four changing makes results from
        the old variant a different experiment, not a comparable baseline."""
        return canon.digest_strings(self.system_prompt, self.model, self.effort,
                                    self.agent)

    def policy(self) -> frozenset:
        if self._policy is None:
            self._policy = frozenset(policy.parse(self.system_prompt))
        return self._policy

    def has(self, flag: str) -> bool:
        return flag in self.policy()

    def label(self) -> str:
        return f"{self.id} [{self.model}/{self.effort}/{self.agent}]"

    def to_dict(self) -> dict:
        return {"id": self.id, "system_prompt": self.system_prompt,
                "model": self.model, "effort": self.effort, "agent": self.agent,
                "source": self.source, "notes": self.notes,
                "digest": self.digest(), "policy": sorted(self.policy())}

    @classmethod
    def from_dict(cls, d: dict) -> "Variant":
        return cls(id=d["id"], system_prompt=d.get("system_prompt", ""),
                   model=d.get("model", DEFAULT_MODEL),
                   effort=d.get("effort", DEFAULT_EFFORT),
                   agent=d.get("agent", "scripted"),
                   source=d.get("source", ""), notes=d.get("notes", ""))


def load(path: str | Path, variant_id: Optional[str] = None,
         model: str = DEFAULT_MODEL, effort: str = DEFAULT_EFFORT,
         agent: str = "scripted", notes: str = "") -> Variant:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"no prompt file at {p}")
    return Variant(id=variant_id or p.stem, system_prompt=p.read_text(encoding="utf-8"),
                   model=model, effort=effort, agent=agent, source=str(p),
                   notes=notes)


def diff(before: Variant, after: Variant, context: int = 0) -> list[str]:
    """Unified diff of two variants' prompts, plus config lines that changed."""
    out = list(difflib.unified_diff(
        before.system_prompt.splitlines(), after.system_prompt.splitlines(),
        fromfile=f"{before.id} ({before.source or 'inline'})",
        tofile=f"{after.id} ({after.source or 'inline'})",
        lineterm="", n=context))
    for field_name in ("model", "effort", "agent"):
        a, b = getattr(before, field_name), getattr(after, field_name)
        if a != b:
            out.append(f"! {field_name}: {a} -> {b}")
    return out


def policy_delta(before: Variant, after: Variant) -> dict[str, list[str]]:
    """Which operating rules the edit added and removed.

    This is the line the demo hangs on: "you removed one rule, it granted three
    directives, and six tasks depend on them."
    """
    a, b = before.policy(), after.policy()
    return {"removed": sorted(a - b), "added": sorted(b - a)}
