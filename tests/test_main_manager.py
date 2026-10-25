"""Run with:  python tests/test_main_manager.py   (or pytest). No model download needed.

Regression for "every message is marked full": main.py used to default to PASS-THROUGH unless MANAGER=memory
was set, so anyone who forgot the variable saw nothing shrunk and no warning.
"""
import os, sys, types
os.environ['MEMORY_SCALE'] = 'off'   # these tests assume a fixed base budget
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tests"))

try:
    import fastapi, httpx  # noqa: F401
except ImportError:                       # keep the test runnable without the web stack installed
    class _App:
        def __init__(self, *a, **k): pass
        def get(self, *a, **k): return lambda f: f
        def post(self, *a, **k): return lambda f: f
    fa = types.ModuleType("fastapi"); fa.FastAPI, fa.Request = _App, object
    fr = types.ModuleType("fastapi.responses"); fr.JSONResponse = fr.StreamingResponse = object
    hx = types.ModuleType("httpx"); hx.AsyncClient = hx.Timeout = hx.Limits = object
    sys.modules.update({"fastapi": fa, "fastapi.responses": fr, "httpx": hx})

import main
from test_memory import FakeEmbedder
from memory.context_manager import ContextManager

KEYS = ("MANAGER", "STANDIN", "MEMORY_BUDGET")


def _env(**kw):
    for k in KEYS:
        os.environ.pop(k, None)
    for k, v in kw.items():
        os.environ[k] = v
    main._MEMORY_ERR = None
    main._MEMORY_CM = None


def _convo(n=40):
    m = [{"role": "system", "content": "You are a helpful voice assistant."},
         {"role": "user", "content": "I'm severely allergic to peanuts."},
         {"role": "assistant", "content": "Understood."}]
    for i in range(n):
        m += [{"role": "user", "content": f"Tell me about topic {i}. " + "Please include plenty of extra detail. " * 8},
              {"role": "assistant", "content": f"Here is a long answer about topic {i}. " + "More detail follows here. " * 10}]
    m.append({"role": "user", "content": "Which dish should I order?"})
    return m


def _fake_memory():
    main._MEMORY_CM = ContextManager(FakeEmbedder(), fact_clf=None)
    return main._MEMORY_CM


def test_default_manager_is_memory_not_passthrough():
    _env()
    assert main._manager_name() == "memory"
    _env(STANDIN="1")
    assert main._manager_name() == "standin"
    _env(MANAGER="standin")
    assert main._manager_name() == "standin"


def test_passthrough_must_be_asked_for():
    for v in ("none", "off", "passthrough"):
        _env(MANAGER=v)
        assert main._manager_name() == ""
        msgs = _convo(5)
        sent, tiers = main.manager_build(msgs, 8000)
        assert sent == msgs and all(t["tier"] == "full" for t in tiers)


def test_default_run_actually_shrinks_a_long_conversation():
    """The bug: with MANAGER unset every tier was 'full' and nothing was dropped."""
    _env()
    main._memory_cm = _fake_memory
    msgs = _convo()
    sent, tiers = main.manager_build(msgs, 8000)
    kinds = {t["tier"] for t in tiers}
    # Coverage-first keeps most messages in a shorter form, so compare tokens and tiers, not message counts.
    assert sum(t["tokens"] for t in tiers) < sum(t["tokens_full"] for t in tiers), "nothing was shrunk"
    assert kinds - {"full"}, f"every tier is full: {kinds}"
    assert tiers[1]["tier"] in ("full", "summary", "gist"), tiers[1]       # the allergy is kept


def test_default_falls_back_loudly_if_memory_cannot_load():
    _env()
    def boom(): raise ImportError("No module named 'sentence_transformers'")
    main._memory_cm = boom
    msgs = _convo(5)
    sent, tiers = main.manager_build(msgs, 8000)
    assert sent == msgs and all(t["tier"] == "full" for t in tiers)
    st = main.manager_status()
    assert st["name"] == "memory" and "sentence_transformers" in st["error"]


def test_explicitly_requested_memory_fails_fast():
    _env(MANAGER="memory")
    def boom(): raise ImportError("missing dependency")
    main._memory_cm = boom
    try:
        main.manager_build(_convo(5), 8000)
    except ImportError:
        return
    raise AssertionError("MANAGER=memory should not silently fall back to pass-through")


def test_status_reports_classifier_and_budget():
    _env(MEMORY_BUDGET="900")
    main._memory_cm = _fake_memory
    main.manager_build(_convo(3), 8000)
    st = main.manager_status()
    assert st["name"] == "memory" and st["error"] is None and st["budget"] == 900
    assert st["fact_classifier"] == {"loaded": False, "threshold": None}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok ", name)
