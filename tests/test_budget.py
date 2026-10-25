"""Run with:  python tests/test_budget.py   (or pytest). No model download needed."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tests"))
from memory.budget import scaled_budget, scale_settings
from memory import tiers as T
from memory.context_manager import ContextManager
from test_memory import FakeEmbedder


def _cfg(mode, **kw):
    s = {"mode": mode, "start": 30, "gain": {"sqrt": 120.0, "linear": 12.0, "quad": 0.25}.get(mode, 0), "max": 6000, "frac": 0.5}
    s.update(kw)
    return s


def test_off_and_short_chats_keep_the_base_budget():
    assert scaled_budget(1500, 500, 10 ** 6, _cfg("off")) == 1500
    for mode in ("sqrt", "linear", "quad"):
        assert scaled_budget(1500, 10, 50_000, _cfg(mode)) == 1500
        assert scaled_budget(1500, 30, 50_000, _cfg(mode)) == 1500          # nothing added at or before START


def test_budget_never_below_base_and_grows_with_length():
    for mode in ("sqrt", "linear", "quad"):
        prev = 0
        for n in (10, 40, 80, 160, 320):
            b = scaled_budget(1500, n, 700 + 95 * n * 10, _cfg(mode))      # large naive size: only START/MAX bind
            assert b >= 1500 and b >= prev, (mode, n, b, prev)
            prev = b


def test_never_approaches_the_naive_size():
    """The guard that makes quadratic safe: the budget stays at or below frac * naive once it has grown."""
    for mode in ("sqrt", "linear", "quad"):
        for n in (50, 100, 200, 400):
            naive = 700 + 95 * n
            b = scaled_budget(1500, n, naive, _cfg(mode))
            assert b <= max(1500, 0.5 * naive) + 1, (mode, n, b, naive)


def test_hard_ceiling_and_curve_shapes():
    assert scaled_budget(1500, 10_000, 10 ** 9, _cfg("quad")) == 6000
    assert scaled_budget(1500, 10_000, 10 ** 9, _cfg("quad", max=4000)) == 4000
    n, naive = 230, 10 ** 9                                                # 200 prompts past START
    sq, li, qu = (scaled_budget(1500, n, naive, _cfg(m)) for m in ("sqrt", "linear", "quad"))
    assert sq < li < qu                                                     # sqrt < linear < quad


def test_environment_parsing_and_bad_values(monkeypatch=None):
    saved = {k: os.environ.get(k) for k in ("MEMORY_SCALE", "MEMORY_SCALE_START", "MEMORY_SCALE_GAIN", "MEMORY_BUDGET_MAX")}
    try:
        os.environ["MEMORY_SCALE"] = "OFF"
        assert scale_settings()["mode"] == "off"
        os.environ["MEMORY_SCALE"] = "banana"
        assert scale_settings()["mode"] == "sqrt"                          # unknown mode falls back to the default
        os.environ["MEMORY_SCALE"] = "linear"; os.environ["MEMORY_SCALE_START"] = "x"; os.environ["MEMORY_SCALE_GAIN"] = "20"
        s = scale_settings()
        assert s["start"] == 30 and s["gain"] == 20.0
    finally:
        for k, v in saved.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v


def _long_convo(turns):
    m = [{"role": "system", "content": "You are a helpful voice assistant. " * 20}]
    for i in range(turns):
        m += [{"role": "user", "content": f"Tell me about topic number {i}, please include plenty of detail."},
              {"role": "assistant", "content": f"Here is an answer about topic {i}. " + "More detail follows here. " * 12}]
    m.append({"role": "user", "content": "What were we saying?"})
    return m


def test_manager_uses_the_scaled_budget_and_reports_it():
    m = _long_convo(120)
    saved = os.environ.get("MEMORY_SCALE")
    try:
        os.environ["MEMORY_SCALE"] = "off"
        _, t_off = ContextManager(FakeEmbedder(), fact_clf=None).build(m, 1500)
        assert T.LAST_INFO["scale"]["mode"] == "off" and T.LAST_INFO["scale"]["scaled_base"] == 1500
        os.environ["MEMORY_SCALE"] = "sqrt"
        _, t_on = ContextManager(FakeEmbedder(), fact_clf=None).build(m, 1500)
        info = T.LAST_INFO["scale"]
        assert info["mode"] == "sqrt" and info["prompts"] == 121 and info["scaled_base"] > 1500, info
        assert T.LAST_INFO["base_budget"] == info["scaled_base"]
        kept = lambda t: sum(x["tier"] != "archived" for x in t)
        assert kept(t_on) > kept(t_off)                                    # a bigger budget keeps more of the chat
    finally:
        if saved is None: os.environ.pop("MEMORY_SCALE", None)
        else: os.environ["MEMORY_SCALE"] = saved


def test_log_mode_is_zero_at_start_never_negative_and_concave():
    st = {"mode": "log", "start": 30, "gain": 350.0, "max": 6000, "frac": 0.5}
    big = 10 ** 9                                             # ceiling out of the way
    vals = [scaled_budget(1500, n, big, st) for n in range(0, 400)]
    assert vals[0] == 1500 and vals[30] == 1500               # nothing before START, never below base
    assert all(b >= a for a, b in zip(vals, vals[1:]))        # non-decreasing
    steps = [vals[i + 1] - vals[i] for i in range(31, 300)]
    assert steps[0] > steps[-1] and all(abs(x) < 400 for x in steps)   # fast then slower, no jumps
    assert max(vals) < 6000


def test_smooth_mode_is_concave_down_from_the_start_and_lands_on_the_cap():
    st = {"mode": "smooth", "start": 30, "gain": 0.0, "max": 6000, "frac": 0.5, "end": 250}
    big = 10 ** 9
    vals = {n: scaled_budget(1500, n, big, st) for n in range(0, 500)}
    assert vals[0] == vals[30] == 1500                       # nothing before START
    assert vals[250] == 6000 and vals[499] == 6000           # hits the cap at END, then flat
    assert all(vals[n + 1] >= vals[n] for n in range(499))   # never decreases
    assert max(vals.values()) == 6000                        # hard cap, never above
    steps = [vals[n + 1] - vals[n] for n in range(30, 250)]
    assert steps[0] >= 25                                    # steep immediately: no slow, concave-up start
    assert all(b <= a + 1 for a, b in zip(steps, steps[1:])) # concave down all the way (rounding aside)
    assert steps[-1] < 5                                     # lands gently, no corner at the cap
    # the full-history guard still applies: a short chat cannot be given more than half of what naive would send
    assert scaled_budget(1500, 250, 4000, st) <= 2000


def test_arctan_mode_jumps_at_the_start_then_flattens_and_never_passes_the_cap():
    st = {"mode": "arctan", "start": 30, "gain": 0.0, "max": 6000, "frac": 0.5, "end": 250, "bend": 40, "arc": 1.0}
    big = 10 ** 9
    vals = {n: scaled_budget(1500, n, big, st) for n in range(0, 800)}
    assert vals[0] == vals[30] == 1500                       # nothing before START
    assert all(vals[n + 1] >= vals[n] for n in range(799)) and max(vals.values()) < 6000   # approaches, never passes
    steps = [vals[n + 1] - vals[n] for n in range(30, 400)]
    assert steps[0] >= 60                                    # large increase straight away
    assert all(b <= a + 1 for a, b in zip(steps, steps[1:])) # concave down: ever smaller gains
    assert steps[-1] < steps[0] / 20                         # flattens out later
    assert vals[100] > 3000 and vals[400] > vals[200]        # most of the budget arrives early, still creeping up
    line = dict(st, arc=0.0)                                 # arc=0: a straight line that lands on the cap at END
    assert scaled_budget(1500, 250, big, line) == 6000 and scaled_budget(1500, 140, big, line) == 3750
    assert scaled_budget(1500, 250, 4000, st) <= 2000         # the full-history guard still applies


def test_root_mode_is_steep_at_the_start_and_approaches_the_cap_asymptotically():
    st = {"mode": "root", "start": 30, "gain": 0.0, "max": 6000, "frac": 0.5, "end": 250, "bend": 40, "arc": 1.0, "half": 100}
    big = 10 ** 9
    vals = {n: scaled_budget(1500, n, big, st) for n in range(0, 2000)}
    assert vals[0] == vals[30] == 1500                       # nothing before START, never negative
    assert all(vals[n + 1] >= vals[n] for n in range(1999)) and max(vals.values()) < 6000   # asymptote, hard cap
    assert vals[31] - vals[30] >= 300                        # very steep straight away (square-root start)
    steps = [vals[n + 1] - vals[n] for n in range(30, 600)]
    assert all(b <= a + 1 for a, b in zip(steps, steps[1:])) # concave down: ever slower
    assert steps[400] < steps[0] / 30                        # flattens out
    assert abs(vals[130] - (1500 + 4500 / 2)) <= 1           # HALF prompts past START = half the extra room
    assert vals[400] > vals[200] > vals[100]                 # but never stops creeping up
    assert scaled_budget(1500, 250, 4000, st) <= 2000        # the full-history guard still applies


def test_curve_shape_is_not_bent_by_the_growing_history_guard():
    """With the history growing every turn (about 180 tokens per prompt, as in tools/demo_long.py) the new modes must
    still be concave down from the very first prompt: no linear start, no 'U'."""
    for mode, extra in (("root", {"half": 100}), ("arctan", {"bend": 40, "arc": 1.0}), ("smooth", {"end": 250})):
        st = {"mode": mode, "start": 30, "gain": 0.0, "max": 6000, "frac": 0.5, **extra}
        vals = [scaled_budget(1500, n, 700 + 180 * n, st) for n in range(30, 400)]
        steps = [b - a for a, b in zip(vals, vals[1:])]
        assert steps[0] >= 30, (mode, steps[0])
        assert all(b <= a + 2 for a, b in zip(steps, steps[1:])), (mode, "curves upwards somewhere")


def test_steep_start_modes_default_to_starting_at_prompt_zero():
    from memory.budget import scale_settings
    old = os.environ.get("MEMORY_SCALE_START")
    os.environ.pop("MEMORY_SCALE_START", None)
    try:
        for mode, start in (("root", 0), ("arctan", 0), ("smooth", 0), ("sqrt", 30), ("quad", 30), ("log", 30)):
            os.environ["MEMORY_SCALE"] = mode
            assert scale_settings()["start"] == start, mode
        os.environ["MEMORY_SCALE"], os.environ["MEMORY_SCALE_START"] = "root", "12"
        assert scale_settings()["start"] == 12                   # an explicit setting still wins
    finally:
        os.environ.pop("MEMORY_SCALE_START", None)
        if old is not None:
            os.environ["MEMORY_SCALE_START"] = old


def test_steep_modes_have_no_guard_by_default_so_the_curve_is_pure():
    from memory.budget import scale_settings
    for k in ("MEMORY_SCALE_START", "MEMORY_SCALE_FRAC"):
        os.environ.pop(k, None)
    for mode in ("root", "arctan", "smooth"):
        os.environ["MEMORY_SCALE"] = mode
        st = scale_settings()
        assert st["frac"] == 0 and st["start"] == 0, mode
        # a history that grows 180 tokens per prompt, as in tools/demo_long.py: the budget is concave down from prompt 1
        vals = [scaled_budget(1500, n, 700 + 180 * n, st) for n in range(0, 400)]
        steps = [b - a for a, b in zip(vals, vals[1:])]
        assert vals[0] == 1500 and steps[0] >= 25 and all(b <= a + 2 for a, b in zip(steps, steps[1:])), (mode, steps[:6])
    os.environ["MEMORY_SCALE"], os.environ["MEMORY_SCALE_FRAC"] = "root", "0.5"       # an explicit guard still works
    assert scaled_budget(1500, 100, 4000, scale_settings()) == 2000
    os.environ.pop("MEMORY_SCALE_FRAC")
    os.environ["MEMORY_SCALE"] = "quad"                                                # older modes keep the 0.5 guard
    assert scale_settings()["frac"] == 0.5


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok ", name)
