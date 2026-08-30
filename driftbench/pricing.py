"""Model price table and cost accounting.

Prices are USD per 1M tokens, first-party Anthropic API rates. Bedrock and Vertex
are partner-operated with separate pricing and are deliberately not modelled here.

Cached reads bill at ~0.1x the input rate; cache writes at ~1.25x.
"""

from __future__ import annotations

CACHE_READ_MULTIPLIER = 0.10
CACHE_WRITE_MULTIPLIER = 1.25

# model id -> (input $/1M, output $/1M)
PRICES: dict[str, tuple[float, float]] = {
    "claude-fable-5": (10.00, 50.00),
    "claude-mythos-5": (10.00, 50.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-4-6": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    # The offline fixture agent and heuristic judge cost nothing. Keeping them in
    # the table means cost reporting needs no special-casing.
    "scripted": (0.0, 0.0),
    "heuristic": (0.0, 0.0),
}

# Batch API runs asynchronously at 50% of standard rates. The judge is a good fit
# for it (single-shot, not latency sensitive); the agent loop is not, because each
# turn depends on the previous tool result.
BATCH_DISCOUNT = 0.50


def price_for(model: str) -> tuple[float, float]:
    if model in PRICES:
        return PRICES[model]
    # Unknown model: charge nothing rather than guess, but make it visible.
    return (0.0, 0.0)


def cost_usd(
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    batch: bool = False,
) -> float:
    """Cost of one request in USD."""
    in_rate, out_rate = price_for(model)
    total = (
        input_tokens * in_rate
        + cache_read_tokens * in_rate * CACHE_READ_MULTIPLIER
        + cache_write_tokens * in_rate * CACHE_WRITE_MULTIPLIER
        + output_tokens * out_rate
    ) / 1_000_000
    if batch:
        total *= BATCH_DISCOUNT
    return total


def is_known(model: str) -> bool:
    return model in PRICES
