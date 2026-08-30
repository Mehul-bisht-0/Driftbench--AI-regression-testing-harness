"""Policy directives: the bridge between a system prompt and fixture behaviour.

The offline demo has to be *real* to be worth anything. A fake demo hard-codes
"task X fails in variant 2"; this one reads the system prompt, extracts which
operating rules are present, and lets the scripted agent behave accordingly. Take
a rule out of the prompt file and the tasks that depended on it genuinely change
their trajectory, exactly as they would with a live model - and, importantly, the
tasks that did not depend on it genuinely do not.

Detection is by distinctive phrase rather than by a hidden machine-readable tag,
so ``prompts/*.txt`` stays a plain system prompt you could paste into a live run.
The trade-off is that rewording a rule can silently drop a flag, which is why
``check_prompt`` exists and why ``tests/test_policy.py`` pins the baseline to the
full flag set.
"""

from __future__ import annotations

import re
from typing import Iterable

# flag -> the phrase in the prompt that grants it. One prompt line may grant
# several flags; that is the point. The demo deletes the single line that grants
# confirm_destructive + no_unrequested_outward + refuse_bulk, and six tasks turn.
DIRECTIVES: dict[str, str] = {
    "no_speculation": r"say you do not know",
    "prefer_context": r"use what the user has already given",
    "confirm_destructive": r"confirm\s*=\s*false",
    "no_unrequested_outward": r"irreversible or outward-facing action the user did not ask for",
    "refuse_bulk": r"decline bulk requests",
    "consult_policy": r"read the relevant policy document",
    "protect_secrets": r"never open or repeat credential material",
    "terse_format": r"follow the output format",
    "ask_when_ambiguous": r"if a request is ambiguous",
    "no_redundant_calls": r"do not repeat a tool call",
    "bounded_retry": r"retry a failing tool at most",
}

ALL_FLAGS = tuple(DIRECTIVES)
_COMPILED = {flag: re.compile(rx, re.I | re.S) for flag, rx in DIRECTIVES.items()}


def parse(system_prompt: str) -> set[str]:
    """Which operating rules this prompt actually contains."""
    text = re.sub(r"\s+", " ", system_prompt or "")
    return {flag for flag, rx in _COMPILED.items() if rx.search(text)}

def missing(system_prompt: str) -> list[str]:
    return sorted(set(ALL_FLAGS) - parse(system_prompt))


def granting_line(system_prompt: str, flag: str) -> str:
    """The prompt line that grants ``flag``, for report footnotes."""
    rx = _COMPILED.get(flag)
    if rx is None:
        raise KeyError(f"unknown policy flag {flag!r}")
    for line in (system_prompt or "").splitlines():
        if rx.search(re.sub(r"\s+", " ", line)):
            return line.strip()
    return ""


def line_flags(system_prompt: str) -> list[tuple[str, list[str]]]:
    """Per-line flag attribution: which rule is load-bearing for what."""
    out = []
    for line in (system_prompt or "").splitlines():
        flat = re.sub(r"\s+", " ", line)
        flags = sorted(f for f, rx in _COMPILED.items() if rx.search(flat))
        if flags:
            out.append((line.strip(), flags))
    return out


def check_prompt(system_prompt: str, expect: Iterable[str] = ALL_FLAGS) -> list[str]:
    """Complaints about a prompt file, for the CLI's ``variants`` command.

    A reworded rule that no longer matches its phrase would silently make tasks
    fail for the wrong reason, and "the eval broke because I edited the prompt
    prose" is precisely the confusion this harness exists to remove.
    """
    found = parse(system_prompt)
    problems = []
    for flag in expect:
        if flag not in found:
            problems.append(f"prompt does not grant {flag!r} "
                            f"(expected phrase /{DIRECTIVES[flag]}/)")
    return problems
