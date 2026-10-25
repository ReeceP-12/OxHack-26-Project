"""Run with:  python tests/test_memory.py   (or pytest)
Uses a fake word-overlap embedder so it needs no model download.
"""
import sys, os, re, zlib
os.environ['MEMORY_SCALE'] = 'off'   # these tests assume a fixed base budget
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from memory.scorer import score_messages, importance, recency, PIN_IMPORTANCE
from memory.tiers import fill_budget, count_tokens, first_sentence
import memory.tiers as T
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
    info = T.LAST_INFO                                    # dynamic budget: base + the pinned facts lane
    assert info["pinned_tokens"] <= info["pin_cap"]
    assert sum(t["tokens"] for t in tiers) <= info["effective_budget"] + 1
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
        assert sent[-1] <= T.LAST_INFO["effective_budget"] + 5, (sent, T.LAST_INFO)
    assert max(sent) <= 700 + T.PIN_CAP + 5, sent          # bounded: never more than base + the pin cap


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


def test_single_sentence_messages_shrink_to_gist_instead_of_vanishing():
    """A one-sentence message cannot be summarised shorter, so it used to jump straight to archived."""
    m = [{"role": "system", "content": "s"}]
    for i in range(14):
        m += [{"role": "user", "content": f"Please tell me something interesting about subject number {i} in the world today"},
              {"role": "assistant", "content": f"Subject {i} has a long and interesting history that people have studied for years"}]
    m.append({"role": "user", "content": "ok go on"})
    kept, tiers = ContextManager(FakeEmbedder()).build(m, budget=330)
    gists = [t for t in tiers if t["tier"] == "gist"]
    assert gists, [t["tier"] for t in tiers]
    assert all(t["tokens"] < t["tokens_full"] for t in gists)
    sent = sum(count_tokens(x["content"]) for x in kept)
    assert sent <= T.LAST_INFO["effective_budget"] + 20, sent   # protected tail may overshoot slightly, as before
    g = next(x for x in kept if x["content"].endswith("..."))
    assert len(g["content"].split()) <= 9


def test_system_message_is_not_embedded():
    class Spy(FakeEmbedder):
        def __init__(self): self.seen = []
        def embed_many(self, texts):
            self.seen += list(texts); return super().embed_many(texts)
    spy = Spy()
    m = [{"role": "system", "content": "SYSTEM MARKER " * 50}, {"role": "user", "content": "hello there"}]
    s = score_messages(m, spy)
    assert not any("SYSTEM MARKER" in t for t in spy.seen)
    assert len(s) == 2
    kept, _ = ContextManager(Spy()).build(m, budget=600)
    assert kept[0]["role"] == "system"



STRONG_FACTS = [
    "I'm severely allergic to peanuts.", "I\u2019m severely allergic to peanuts.", "I have a peanut allergy.",
    "My son is allergic to eggs.", "I'm vegan.", "I can't eat gluten.", "I'm on insulin for my diabetes.",
    "My name is Tom.", "Call me Tom", "My wife's name is Anna.", "I have two kids.",
    "My dog is called Biscuit and he's a three year old beagle.",
    "I live in Oxford.", "I work as a nurse at the hospital.",
]
NOT_FACTS = [
    "Tell me a story about a dragon.", "Can you recommend a vegan restaurant near me?",
    "What's the best food for a dog?", "My friend told me a funny story about a cat.",
    "Is a peanut allergy common?", "I think pizza is great.", "I am so bored.",
    "Tell me about topic 5.", "How do I train a puppy?", "What does the word allergen mean?",
    "I have a question about volcanoes.", "I'm going to the shops later.",
]


def test_personal_facts_are_pinned():
    for t in STRONG_FACTS:
        assert importance(t) >= PIN_IMPORTANCE, t


def test_ordinary_chatter_is_not_pinned():
    for t in NOT_FACTS:
        assert importance(t) < PIN_IMPORTANCE, t


def test_regression_two_weak_hits_do_not_pin_by_accident():
    # "am" used to count as a time-of-day word, so any "I am ..." plus one digit was pinned.
    assert importance("I am bored, tell me about topic 5.") < PIN_IMPORTANCE


def test_regression_allergy_and_pet_survive_without_word_overlap():
    """Teammate B, 24 Oct: allergy/pet were archived and only came back because the question reused their words."""
    m = [{"role": "system", "content": "You are a helpful voice assistant."},
         {"role": "user", "content": "I'm severely allergic to peanuts."},
         {"role": "assistant", "content": "Understood, I will keep that in mind."},
         {"role": "user", "content": "My dog is called Biscuit and he's a three year old beagle."},
         {"role": "assistant", "content": "Lovely. Biscuit sounds fun."}]
    for i in range(40):
        m += [{"role": "user", "content": f"Tell me about topic {i}. " + "Please include plenty of extra detail. " * 8},
              {"role": "assistant", "content": f"Here is a long answer about topic {i}. " + "More detail follows here. " * 10}]
    m.append({"role": "user", "content": "Is this pad thai safe for me, and what should I get my pet?"})
    _, tiers = ContextManager(FakeEmbedder()).build(m, budget=1500)
    assert tiers[1]["tier"] in ("full", "summary"), tiers[1]
    assert tiers[3]["tier"] in ("full", "summary"), tiers[3]

def _chat(first_fact, n_filler):
    m = [{"role": "system", "content": "You are a helpful voice assistant."}]
    if first_fact:
        m += [{"role": "user", "content": first_fact}, {"role": "assistant", "content": "Understood, I will keep that in mind."}]
    for t in range(n_filler):
        m += [{"role": "user", "content": f"Tell me a long random story about topic number {t} with many words in it please."},
              {"role": "assistant", "content": "Sure. Here is a long story. " + "Words go here. " * 15}]
    m.append({"role": "user", "content": "Anything I should remember about my health?"})
    return m


def test_dynamic_budget_steps_up_only_for_a_stated_fact():
    cm = ContextManager(FakeEmbedder())
    cm.build(_chat(None, 30), 1500)
    assert T.LAST_INFO["pinned_tokens"] == 0 and T.LAST_INFO["effective_budget"] == 1500
    kept, tiers = cm.build(_chat("I have a heart condition, so please avoid anything strenuous.", 30), 1500)
    i = T.LAST_INFO
    assert tiers[1]["pinned"] and tiers[1]["critical"], tiers[1]       # found by the critical patterns
    assert any("heart" in x["content"] for x in kept)                  # survives 30 turns of filler
    assert 0 < i["pinned_tokens"] <= 60 and i["effective_budget"] == 1500 + i["pinned_tokens"], i


def test_pin_lane_is_capped_and_critical_is_never_evicted():
    nouns = ("river garage ladder violin castle engine pillow quartz meadow harbor lantern marble orchard pepper saddle "
             "thimble velvet walnut anchor basket cobalt dragon ember falcon glacier hammock island jungle kettle lemon "
             "magnet nectar oyster pebble quilt rocket spruce tundra umbrella voyage willow yacht zephyr bronze candle "
             "donkey eagle forest").split()
    m = [{"role": "system", "content": "s"},
         {"role": "user", "content": "I have a heart condition, so please avoid anything strenuous."}, {"role": "assistant", "content": "Ok."}]
    for k, w in enumerate(nouns):
        m += [{"role": "user", "content": f"My {w} is booked for Friday {k + 1} March at {1000 + k * 37} and it is due then."}, {"role": "assistant", "content": "Noted."}]
    m.append({"role": "user", "content": "What should I avoid?"})
    kept, tiers = ContextManager(FakeEmbedder()).build(m, 1500)
    i = T.LAST_INFO
    assert i["pinned_tokens"] <= i["pin_cap"] and i["pin_overflow"] > 0 and i["critical_count"] == 1, i
    assert any("heart" in x["content"] for x in kept), "critical fact was evicted by ordinary ones"


def test_updated_fact_replaces_old_but_critical_facts_are_not_merged():
    m = [{"role": "system", "content": "s"},
         {"role": "user", "content": "My flight is BA 249 and it leaves at 6:40am."}, {"role": "assistant", "content": "Ok."},
         {"role": "user", "content": "Update: my flight is now BA 251 and it leaves at 7:10am."}, {"role": "assistant", "content": "Ok."},
         {"role": "user", "content": "I am allergic to peanuts."}, {"role": "assistant", "content": "Ok."},
         {"role": "user", "content": "I am allergic to shellfish too."}, {"role": "assistant", "content": "Ok."}]
    for k in range(6):
        m += [{"role": "user", "content": f"Filler {k}."}, {"role": "assistant", "content": "Fine."}]
    kept, tiers = ContextManager(FakeEmbedder()).build(m, 250)
    txt = " ".join(x["content"] for x in kept)
    assert "251" in txt and "peanuts" in txt and "shellfish" in txt, txt
    assert not tiers[1]["pinned"], "the older flight message should have been superseded"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok ", name)