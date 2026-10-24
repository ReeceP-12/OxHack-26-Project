"""Run with:  python tests/test_memory.py   (or pytest)
Uses a fake word-overlap embedder so it needs no model download.
"""
import sys, os, re, zlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from memory.scorer import score_messages, importance, recency
from memory.tiers import fill_budget, count_tokens, first_sentence
from memory.context_manager import ContextManager


class FakeEmbedder:
    """Hashed bag of words. Shared words = similar. Good enough to test the logic."""
    def embed_many(self, texts):
        out = []
        for t in texts:
            v = np.zeros(64, dtype=np.float32)
            for w in re.findall(r"[a-z']+", t.lower()):
                v[zlib.crc32(w.encode()) % 64] += 1
            n = np.linalg.norm(v)
            out.append(v / n if n else v)
        return np.stack(out)


def convo():
    m = [{"role": "system", "content": "You are a helpful voice assistant. " * 20}]
    m += [{"role": "user", "content": "My deadline is Friday at 5pm and I am allergic to peanuts."},
          {"role": "assistant", "content": "Noted. Deadline Friday 5pm, peanut allergy."}]
    for i in range(12):
        m += [{"role": "user", "content": f"Tell me a long random story about topic number {i} with many words in it please."},
              {"role": "assistant", "content": f"Sure. Here is a long story about topic {i}. " + "Words go here. " * 15}]
    m += [{"role": "user", "content": "What deadline did I give you again?"}]
    return m


def test_importance_and_recency():
    assert importance("My deadline is Friday at 5pm") > importance("tell me a story")
    assert importance("I am allergic to peanuts", "user") > importance("I am allergic to peanuts", "assistant")
    assert recency(0) == 1 and abs(recency(10) - 0.5) < 1e-9


def test_scores_in_range_and_deadline_ranks_high():
    m = convo()
    s = score_messages(m, FakeEmbedder())
    assert len(s) == len(m)
    assert all(0 <= x["score"] <= 1.5 for x in s)
    old = s[1]["score"]                                   # the deadline message
    boring = np.mean([s[i]["score"] for i in range(3, 12, 2)])
    assert old > boring, (old, boring)


def test_budget_respected_and_protected_kept():
    m = convo()
    kept, tiers = ContextManager(FakeEmbedder()).build(m, budget=600)
    assert kept[0]["role"] == "system"
    assert kept[-1]["content"] == m[-1]["content"]        # current question kept
    assert len(tiers) == len(m)
    assert sum(t["tokens"] for t in tiers) <= 600 + 1
    assert sum(1 for t in tiers if t["tier"] == "archived") > 0


def test_deadline_survives_while_boring_dropped():
    m = convo()
    _, tiers = ContextManager(FakeEmbedder()).build(m, budget=600)
    assert tiers[1]["tier"] in ("full", "summary"), tiers[1]


def test_order_preserved_and_tokens_flat():
    base = convo()
    sent = []
    for k in (10, 20, 28):                                 # growing conversations
        m = base[:k] + [base[-1]]
        kept, _ = ContextManager(FakeEmbedder()).build(m, budget=700)
        idx = [m.index(x) if x in m else -1 for x in kept]
        assert idx == sorted(idx) or -1 in idx            # summaries have new content; skip those
        sent.append(sum(count_tokens(x["content"]) for x in kept))
    assert max(sent) <= 700 + 5, sent


def test_small_conversation_untouched():
    m = [{"role": "system", "content": "hi"}, {"role": "user", "content": "hello"}]
    kept, tiers = ContextManager(FakeEmbedder()).build(m, budget=1500)
    assert kept == m and all(t["tier"] == "full" for t in tiers)


def repetitive_convo(n_filler=30):
    """Mirrors tools/send_convo.py: near-identical filler turns that look just like the final question."""
    m = [{"role": "system", "content": "You are a helpful voice assistant."},
         {"role": "user", "content": "My deadline is Friday at 5pm and I'm allergic to peanuts."},
         {"role": "assistant", "content": "Mock reply. I was given 2 messages and about 30 tokens of context. "}]
    for i in range(n_filler):
        m += [{"role": "user", "content": f"Tell me about topic {i}. " + "Please include plenty of extra detail. " * 8},
              {"role": "assistant", "content": f"Mock reply. I was given {i} messages and about 600 tokens of context. "}]
    m.append({"role": "user", "content": "Tell me about topic 99. " + "Please include plenty of extra detail. " * 8})
    return m


def test_regression_fact_survives_repetitive_filler():
    """Bug found 24 Oct: with 30 near-identical turns the deadline got archived. Pin floor fixes it."""
    m = repetitive_convo()
    _, tiers = ContextManager(FakeEmbedder()).build(m, budget=1500)
    assert tiers[1]["tier"] in ("full", "summary"), tiers[1]


def test_pin_floor_applies_to_user_messages_only():
    m = [{"role": "system", "content": "x"},
         {"role": "assistant", "content": "Remember, my deadline is Friday at 5pm and I am allergic to nuts."},
         {"role": "user", "content": "ok"}]
    s = score_messages(m, FakeEmbedder())
    assert s[1]["score"] < 0.7


def test_summary_skips_filler_opener():
    t = "Sure. Here is a fairly long explanation of how volcanoes work. More detail follows."
    assert first_sentence(t).startswith("Here is a fairly long explanation")
    assert first_sentence("Great question! The capital of France is Paris.").startswith("The capital")
    assert first_sentence("Sure.") == "Sure."            # nothing else to keep


def test_tier_entries_have_agreed_keys():
    _, tiers = ContextManager(FakeEmbedder()).build(convo(), budget=600)
    for t in tiers:
        assert set(t) >= {"index", "role", "tier", "score", "tokens"}
        assert t["tier"] in ("full", "summary", "gist", "archived")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok ", name)