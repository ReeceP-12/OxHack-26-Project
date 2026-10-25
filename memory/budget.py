"""Scale the base token budget with the length of the conversation.

A fixed budget means that in a long chat the same few thousand tokens are shared by ever more messages, so old
context is squeezed harder and harder. This lets the base budget grow with the number of user prompts, but
never so much that the prompt approaches the naive (send-everything) size, which would throw away the saving.

    scaled = base + gain * (prompts - START) ** p        (p = 0.5 sqrt, 1 linear, 2 quadratic; or gain * ln(1 + k) for mode log)
    effective = base + room * tanh(raw / room),  room = min(MAX, FRAC * naive_tokens) - base

The ceiling is the guard that matters (approached smoothly, never hit as a corner): a quadratic curve left alone overtakes the naive size (which itself
grows linearly), so it is bounded to FRAC of what naive would send.

Environment (all optional):
    MEMORY_SCALE         off | sqrt | linear | quad | log | smooth | arctan | root          (default: see DEFAULT_MODE)
    MEMORY_SCALE_START   prompts before scaling begins        (default 30; 0 for root, arctan and smooth)
    MEMORY_SCALE_END     smooth/arctan: prompt at which the ceiling is reached (default 250)
    MEMORY_SCALE_HALF    root only: prompts past START at which half of the extra budget is reached (default 100)
    MEMORY_SCALE_BEND    arctan only: prompts over which the early rise bends (default 40; smaller = bigger jump at the start)
    MEMORY_SCALE_ARC     arctan only: 0 to 1, share of arctan vs a straight line to the ceiling at END (default 1 = pure arctan)
    MEMORY_SCALE_GAIN    tokens per step, overrides the mode's default gain
    MEMORY_BUDGET_MAX    hard ceiling on the base budget      (default 6000)
    MEMORY_SCALE_FRAC    ceiling as a fraction of naive size  (default 0.5; 0 = off, which is the default for root, arctan and smooth)
"""
import math
import os

DEFAULT_MODE = "sqrt"
_POWER = {"sqrt": 0.5, "linear": 1.0, "quad": 2.0, "log": None, "smooth": None, "arctan": None, "root": None}   # log: gain * ln(1 + k); smooth: smoothstep to a cap, see below
_GAIN = {"sqrt": 120.0, "linear": 12.0, "quad": 0.25, "log": 350.0, "smooth": 0.0, "arctan": 0.0, "root": 0.0}    # tokens added per (prompt beyond START) ** p
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
    # The steep-start curves (root, arctan, smooth) begin at prompt 0 by default: a late START would leave the budget flat
    # and then make it jump. The older gentle curves (sqrt, linear, quad, log) keep their START of 30.
    steep = mode in ("root", "arctan", "smooth")
    default_start = 0 if steep else 30
    default_frac = 0 if steep else 0.5      # steep modes: no full-history guard (see _clamped); 0 = guard off
    return {"mode": mode, "start": _num("MEMORY_SCALE_START", default_start),
            "gain": _num("MEMORY_SCALE_GAIN", _GAIN.get(mode, 0)),
            "max": _num("MEMORY_BUDGET_MAX", 6000), "frac": _num("MEMORY_SCALE_FRAC", default_frac),
            "end": _num("MEMORY_SCALE_END", 250), "bend": _num("MEMORY_SCALE_BEND", 40),
            "arc": min(1.0, max(0.0, _num("MEMORY_SCALE_ARC", 1.0))),
            "half": _num("MEMORY_SCALE_HALF", 100)}


def _clamped(base, s, naive_tokens, f):
    """base + (MAX - base) * f, held back by the full-history guard (never more than FRAC of what naive would send).

    The curve f is applied to the FULL room (MAX - base), and the guard only CLIPS the result. Scaling the room by the
    guard instead would multiply the curve by a growing number (naive grows with every turn), which bends the start of
    the chart upwards (a 'U') even though f itself is concave down."""
    value = base + max(0.0, s["max"] - base) * f
    if s["frac"] <= 0:                                      # guard off (the default for the steep-start modes)
        return int(value)
    return int(min(value, max(float(base), s["frac"] * naive_tokens)))


def scaled_budget(base, n_prompts, naive_tokens, settings=None):
    """Effective base budget for a conversation with n_prompts user messages and naive_tokens in total."""
    s = settings or scale_settings()
    if s["mode"] == "off":
        return int(base)
    k = max(0.0, n_prompts - s["start"])
    if s["mode"] == "root":
        # Square-root start, hard-cap finish. u = sqrt(k / HALF) rises like a square root (very steep at the start,
        # infinite slope at k = 0), and f = u / (1 + u) bends it into an asymptote at the ceiling:
        #   budget = base + room * u / (1 + u)       reaches HALF the extra room at k = HALF prompts past START,
        # then creeps towards the cap without ever passing it (about 90% of the room at 81 x HALF prompts).
        u = math.sqrt(k / max(1.0, s.get("half", 100)))
        return _clamped(base, s, naive_tokens, u / (1 + u))
    if s["mode"] == "arctan":
        # Plain arctan on x >= 0 (x = prompts beyond START): a large jump right at the start that keeps bending
        # and flattens towards the ceiling, which it approaches but never passes:
        #   f = (2 / pi) * atan(k / BEND)          f(0) = 0, f -> 1 as k grows
        # ARC < 1 blends in a straight line that reaches the ceiling at prompt `end` (ARC = 1, the default, is pure arctan).
        arc, bend = s.get("arc", 1.0), max(1.0, s.get("bend", 40))
        span = max(1.0, s.get("end", 250) - s["start"])
        f = arc * (2 / math.pi) * math.atan(k / bend) + (1 - arc) * min(1.0, k / span)
        return _clamped(base, s, naive_tokens, f)
    if s["mode"] == "smooth":
        # Only the concave-down HALF of a smoothstep S-curve (the part after its midpoint, rescaled): it rises
        # steeply straight away, keeps slowing, and lands on the ceiling EXACTLY at prompt `end` with zero slope,
        # a true hard cap with no corner. f(t) = 1.5 t - 0.5 t^3 on t in [0, 1]. No exp/tanh, just a cubic.
        span = max(1.0, s.get("end", 250) - s["start"])
        t = min(1.0, k / span)
        return _clamped(base, s, naive_tokens, 1.5 * t - 0.5 * t ** 3)
    # "log" is gain * ln(1 + k): zero at k = 0, never negative, fast at first and ever slower (diminishing returns).
    raw = s["gain"] * (math.log1p(k) if s["mode"] == "log" else k ** _POWER[s["mode"]])
    ceiling = min(s["max"], s["frac"] * naive_tokens) if s["frac"] > 0 else s["max"]
    room = ceiling - base
    if room <= 0 or raw <= 0:
        return int(base)
    # Soft ceiling: base + room * tanh(raw / room). It follows the chosen curve while raw is small (tanh(x) ~ x)
    # and bends smoothly towards the ceiling instead of hitting it as a hard corner, so the chart has no kink.
    return int(base + room * math.tanh(raw / room))
