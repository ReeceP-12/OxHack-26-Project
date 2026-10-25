"""Scale the base token budget with the length of the conversation.

A fixed budget means that in a long chat the same few thousand tokens are shared by ever more messages, so old
context is squeezed harder and harder. This lets the base budget grow with the number of user prompts, but
never so much that the prompt approaches the naive (send-everything) size, which would throw away the saving.

    scaled = base + gain * (prompts - START) ** p        (p = 0.5 sqrt, 1 linear, 2 quadratic)
    effective = base + room * tanh(raw / room),  room = min(MAX, FRAC * naive_tokens) - base

The ceiling is the guard that matters (approached smoothly, never hit as a corner): a quadratic curve left alone overtakes the naive size (which itself
grows linearly), so it is bounded to FRAC of what naive would send.

Environment (all optional):
    MEMORY_SCALE         off | sqrt | linear | quad          (default: see DEFAULT_MODE)
    MEMORY_SCALE_START   prompts before scaling begins        (default 30)
    MEMORY_SCALE_GAIN    tokens per step, overrides the mode's default gain
    MEMORY_BUDGET_MAX    hard ceiling on the base budget      (default 6000)
    MEMORY_SCALE_FRAC    ceiling as a fraction of naive size  (default 0.5)
"""
import math
import os

DEFAULT_MODE = "sqrt"
_POWER = {"sqrt": 0.5, "linear": 1.0, "quad": 2.0}
_GAIN = {"sqrt": 120.0, "linear": 12.0, "quad": 0.25}    # tokens added per (prompt beyond START) ** p
_OFF = ("off", "fixed", "none", "0", "false", "no")


def _num(name, default):
    try:
        return float(os.environ.get(name, "").strip() or default)
    except ValueError:
        return float(default)


def scale_settings():
    mode = (os.environ.get("MEMORY_SCALE", "").strip().lower() or DEFAULT_MODE)
    if mode in _OFF:
        mode = "off"
    elif mode not in _POWER:
        mode = DEFAULT_MODE
    return {"mode": mode, "start": _num("MEMORY_SCALE_START", 30),
            "gain": _num("MEMORY_SCALE_GAIN", _GAIN.get(mode, 0)),
            "max": _num("MEMORY_BUDGET_MAX", 6000), "frac": _num("MEMORY_SCALE_FRAC", 0.5)}


def scaled_budget(base, n_prompts, naive_tokens, settings=None):
    """Effective base budget for a conversation with n_prompts user messages and naive_tokens in total."""
    s = settings or scale_settings()
    if s["mode"] == "off":
        return int(base)
    k = max(0.0, n_prompts - s["start"])
    raw = s["gain"] * (k ** _POWER[s["mode"]])
    ceiling = min(s["max"], s["frac"] * naive_tokens)
    room = ceiling - base
    if room <= 0 or raw <= 0:
        return int(base)
    # Soft ceiling: base + room * tanh(raw / room). It follows the chosen curve while raw is small (tanh(x) ~ x)
    # and bends smoothly towards the ceiling instead of hitting it as a hard corner, so the chart has no kink.
    return int(base + room * math.tanh(raw / room))
