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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok ", name)
