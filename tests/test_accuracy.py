"""Run with:  python tests/test_accuracy.py   (checks the statistics helpers and the conversation builder)"""
import os, sys, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.accuracy import wilson, mcnemar_exact, window_to, make_conversation, survives, FACTS, UPDATE


def test_wilson_is_sane():
    lo, hi = wilson(0, 10); assert lo == 0.0 and 0.2 < hi < 0.35
    lo, hi = wilson(10, 10); assert hi == 1.0 and 0.65 < lo < 0.8
    lo, hi = wilson(50, 100); assert 0.39 < lo < 0.41 and 0.59 < hi < 0.61
    assert wilson(0, 0) == (0.0, 0.0)


def test_mcnemar_exact_known_values():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0
    assert abs(mcnemar_exact(10, 0) - 2 / 1024) < 1e-12
    assert abs(mcnemar_exact(0, 10) - mcnemar_exact(10, 0)) < 1e-12


def test_conversation_plants_five_scored_facts_and_ends_on_a_user_prompt():
    for n in (50, 100, 200):
        msgs, planted = make_conversation(random.Random(3), n)
        assert msgs[0]["role"] == "system" and msgs[-1]["role"] == "user"
        assert sum(1 for m in msgs if m["role"] == "user") == n
        assert len(planted) == 5 and sum(1 for f, _ in planted if f is UPDATE) == 1
        assert any(f["crit"] for f, _ in planted)
        for f, i in planted:                                   # every scored fact is really in the history
            assert survives(msgs, f), f["id"]


def test_filler_never_contains_a_fact_key_by_accident():
    msgs, planted = make_conversation(random.Random(4), 200)
    planted_keys = {k for f, _ in planted for k in f["key"]}      # peanut and peanut_p share the key "peanut"
    for f in FACTS + [UPDATE]:
        if not set(f["key"]) & planted_keys:
            assert not survives(msgs, f), f"{f['id']} key leaks into the filler"


def test_window_matches_the_token_budget_and_starts_on_a_user_message():
    msgs, _ = make_conversation(random.Random(5), 100)
    count = lambda t: len(t) // 4 + 4
    w = window_to(msgs, 1200, count)
    assert sum(count(m["content"]) for m in w) <= 1200
    assert w[0]["role"] == "system" and w[1]["role"] == "user" and w[-1] is msgs[-1]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok ", name)
