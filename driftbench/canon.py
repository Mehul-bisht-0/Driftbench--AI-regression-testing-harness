"""Canonicalisation: turning a trajectory into a comparable sequence of tokens.

This is the substrate everything else stands on. Diffing, "are these two runs
identical", and trajectory clustering all reduce to: project each step onto a
canonical string, then compare sequences of strings.

Three granularities, because the interesting questions differ:

``shape``      tool names only. "Did the agent take the same *plan*?"
``semantic``   tool + normalised args. Argument spelling that does not change
               meaning ("sort a python list" vs "python list sort") collapses to
               one token. This is the default for clustering and flakiness.
``strict``     tool + args with only key-ordering and float rounding applied.
               Any argument change at all shows up. Used for exact-replay checks.

Normalisation is *registered per tool*, not guessed globally, because only the
tool author knows which arguments are order-free or case-free.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Callable, Iterable

LEVELS = ("shape", "semantic", "strict")
DEFAULT_LEVEL = "semantic"

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]+")

# tool name -> callable(args) -> args, applied only at the "semantic" level.
ARG_NORMALIZERS: dict[str, Callable[[dict], dict]] = {}


def register_arg_normalizer(tool: str) -> Callable[[Callable], Callable]:
    def deco(fn: Callable[[dict], dict]) -> Callable[[dict], dict]:
        ARG_NORMALIZERS[tool] = fn
        return fn
    return deco


def collapse_ws(s: str) -> str:
    return _WS.sub(" ", s).strip()


def word_bag(s: str) -> str:
    """Order- and case-insensitive projection of a free-text argument.

    Used for search queries, where "python list sort" and "sort python list" are
    the same call and should not register as a trajectory difference.
    """
    words = sorted(set(_PUNCT.sub(" ", s.lower()).split()))
    return " ".join(words)


def norm_path(s: str) -> str:
    return collapse_ws(s.replace("\\", "/").lstrip("./")).lower()


def _norm_value(v: Any, deep: bool) -> Any:
    """Structural normalisation applied at every level: stable key order, floats
    rounded so 0.30000000000000004 == 0.3, whitespace collapsed in strings."""
    if isinstance(v, dict):
        return {k: _norm_value(v[k], deep) for k in sorted(v)}
    if isinstance(v, (list, tuple)):
        return [_norm_value(x, deep) for x in v]
    if isinstance(v, float):
        return round(v, 6)
    if isinstance(v, str):
        s = collapse_ws(v)
        return s.lower() if deep else s
    return v


def normalize_args(tool: str | None, args: dict, level: str) -> dict:
    if not isinstance(args, dict):
        args = {"_": args}
    if level == "strict":
        return _norm_value(args, deep=False)
    out = _norm_value(args, deep=False)
    if level == "semantic":
        fn = ARG_NORMALIZERS.get(tool or "")
        if fn is not None:
            out = fn(dict(out))
        out = _norm_value(out, deep=True)
    return out


def canon_token(step, level: str = DEFAULT_LEVEL) -> str:
    """Project one step onto a single comparable string."""
    if level not in LEVELS:
        raise ValueError(f"unknown canon level {level!r}; expected one of {LEVELS}")
    kind = step.kind
    if kind == "tool_call":
        tool = step.tool or "?"
        if level == "shape":
            return f"call:{tool}"
        args = normalize_args(tool, step.args or {}, level)
        blob = json.dumps(args, sort_keys=True, separators=(",", ":"), default=str)
        return f"call:{tool}({blob})"
    if kind == "tool_result":
        # Results are environment-side. What matters for behaviour comparison is
        # whether the step errored, not the payload bytes.
        return f"result:{step.tool or '?'}:{'err' if step.is_error else 'ok'}"
    if kind == "assistant_text":
        return "say" if level == "shape" else f"say:{len(collapse_ws(step.text))//40}"
    if kind == "final":
        return "final"
    if kind == "error":
        return f"harness_error:{collapse_ws(step.text)[:60]}"
    return f"{kind}"


def canon_sequence(traj, level: str = DEFAULT_LEVEL) -> list[str]:
    return [canon_token(s, level) for s in traj.steps]


def _digest(parts: Iterable[str]) -> str:
    h = hashlib.blake2b(digest_size=10)
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()


def trajectory_digest(traj, level: str = DEFAULT_LEVEL) -> str:
    """Stable id for "this exact behaviour". Includes the terminal state so a run
    that refused and a run that answered never collide."""
    return _digest([*canon_sequence(traj, level), f"stop={traj.stop_reason}"])


def annotate(traj, level: str = DEFAULT_LEVEL):
    """Fill in per-step canonical tokens and the trajectory digest, in place."""
    for step in traj.steps:
        step.canon = canon_token(step, level)
    traj.digest = trajectory_digest(traj, level)
    return traj


def norm_text(s: str) -> str:
    """Loose normalisation for comparing free-text answers."""
    return collapse_ws(_PUNCT.sub(" ", (s or "").lower()))


def digest_strings(*parts: str) -> str:
    return _digest(parts)
