"""Tool registry: schemas, dispatch, validation, and fault injection.

A tool here is a pure function of ``(World, args)``. The registry is the single
source of truth for three consumers that must not drift apart:

* the mock executor (this module),
* the tool schemas sent to the live model,
* the canonicalisation rules used when diffing trajectories.

Dispatch deliberately does the boring hostile things a real tool surface does -
reject unknown tools, reject unknown arguments, reject wrong types - because
"agent recovers from its own malformed call" is behaviour worth testing, and a
permissive mock would hide it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

from . import canon
from .world import World

ToolFn = Callable[[World, dict], Any]


@dataclass
class ToolSpec:
    name: str
    description: str
    properties: dict[str, dict]
    required: list[str]
    fn: ToolFn
    destructive: bool = False  # mutates or removes state
    outward: bool = False  # leaves the org (email, webhook)
    tags: tuple[str, ...] = ()

    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": self.properties,
            "required": list(self.required),
            "additionalProperties": False,
        }


REGISTRY: dict[str, ToolSpec] = {}


def tool(name: str, description: str, properties: dict, required: Iterable[str],
         destructive: bool = False, outward: bool = False,
         tags: Iterable[str] = ()) -> Callable[[ToolFn], ToolFn]:
    def deco(fn: ToolFn) -> ToolFn:
        if name in REGISTRY:
            raise ValueError(f"tool {name!r} registered twice")
        REGISTRY[name] = ToolSpec(
            name=name, description=description, properties=properties,
            required=list(required), fn=fn, destructive=destructive,
            outward=outward, tags=tuple(tags),
        )
        return fn
    return deco


def get(name: str) -> Optional[ToolSpec]:
    return REGISTRY.get(name)


def names() -> list[str]:
    return sorted(REGISTRY)


def destructive_names() -> set[str]:
    return {n for n, s in REGISTRY.items() if s.destructive}


def outward_names() -> set[str]:
    return {n for n, s in REGISTRY.items() if s.outward}


def anthropic_tool_defs(only: Optional[Iterable[str]] = None) -> list[dict]:
    """Tool definitions for the Messages API.

    ``strict: True`` is a top-level field on the tool (not on ``tool_choice``) and
    requires ``additionalProperties: false`` plus ``required`` - which
    ``ToolSpec.input_schema`` always emits. Order is sorted and stable so the
    ``tools`` block stays byte-identical across requests; a varying tool list is
    one of the classic silent prompt-cache invalidators.
    """
    wanted = sorted(only) if only is not None else names()
    defs = []
    for n in wanted:
        spec = REGISTRY[n]
        defs.append({
            "name": spec.name,
            "description": spec.description,
            "strict": True,
            "input_schema": spec.input_schema(),
        })
    return defs


_TYPE_MAP = {
    "string": str, "integer": int, "number": (int, float),
    "boolean": bool, "array": list, "object": dict,
}


def validate_args(spec: ToolSpec, args: dict) -> Optional[str]:
    """Minimal JSON-Schema check. Returns an error string, or None if valid."""
    if not isinstance(args, dict):
        return f"arguments must be an object, got {type(args).__name__}"
    for key in spec.required:
        if key not in args:
            return f"missing required argument {key!r}"
    for key, value in args.items():
        prop = spec.properties.get(key)
        if prop is None:
            allowed = ", ".join(sorted(spec.properties)) or "(none)"
            return f"unknown argument {key!r}; accepted arguments are: {allowed}"
        expected = _TYPE_MAP.get(prop.get("type", "string"), object)
        # bool is a subclass of int in Python; do not let True satisfy "integer".
        if prop.get("type") in ("integer", "number") and isinstance(value, bool):
            return f"argument {key!r} must be {prop['type']}, got boolean"
        if not isinstance(value, expected):
            return f"argument {key!r} must be {prop.get('type')}, got {type(value).__name__}"
        if "enum" in prop and value not in prop["enum"]:
            return f"argument {key!r} must be one of {prop['enum']}"
    return None


def call_tool(world: World, name: str, args: dict) -> tuple[Any, bool]:
    """Execute a tool against the world. Returns ``(result, is_error)``.

    Errors are returned, never raised: the agent should see a ``tool_result`` with
    ``is_error: true`` and get the chance to recover, which is exactly what the
    error-recovery tasks measure.
    """
    spec = REGISTRY.get(name)
    if spec is None:
        return (f"Error: no such tool {name!r}. Available tools: "
                f"{', '.join(names())}"), True

    problem = validate_args(spec, args)
    if problem is not None:
        return f"Error: invalid arguments for {name}: {problem}", True

    nth = world.bump(name)
    injected = world.check_fault(name, nth)
    if injected is not None:
        return f"Error: {injected}", True

    try:
        return spec.fn(world, args), False
    except LookupError as exc:  # not-found style failures the agent can act on
        return f"Error: {exc}", True
    except ValueError as exc:  # bad input the agent can correct
        return f"Error: {exc}", True


@canon.register_arg_normalizer("search_docs")
def _norm_search(args: dict) -> dict:
    # Query wording and word order are not behavioural differences; treat
    # "refund policy days" and "days policy refund" as the same call.
    if isinstance(args.get("query"), str):
        args["query"] = canon.word_bag(args["query"])
    args.pop("limit", None)  # a different page size is not a different plan
    return args


for _path_tool in ("read_file", "write_file", "delete_file", "list_files"):
    canon.register_arg_normalizer(_path_tool)(
        lambda args: {**args, "path": canon.norm_path(args["path"])}
        if isinstance(args.get("path"), str) else args
    )
