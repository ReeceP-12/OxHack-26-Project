"""Run:  python standin/test_standin.py   (from the repo root)"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from standin.standin import build, _tokens

FILLER = ["Tell me about the weather in {c}.", "What is a good book about {c}?", "How do trains work in {c}?",
          "Can you describe the food in {c}?", "What music is popular in {c}?"]
CITIES = ["Paris", "Tokyo", "Lima", "Cairo", "Oslo", "Rome", "Seoul", "Delhi", "Perth", "Quito"]

def convo(n_turns):
    msgs = [{"role": "system", "content": "You are a helpful voice assistant. " * 20}]
    for t in range(n_turns):
        if t == 1: u = "Remember that my project deadline is Friday the 13th."
        elif t == 3: u = "I am allergic to peanuts, so never suggest them."
        else: u = FILLER[t % 5].format(c=CITIES[t % 10]) + " Please give a fairly detailed answer, with a few examples."
        msgs.append({"role": "user", "content": u})
        msgs.append({"role": "assistant", "content": "Sure. " + ("Here is a detailed answer with examples. " * 6)})
    return msgs

def check_dynamic():
    """Pinned lane: the effective budget steps up when a critical fact arrives, stays bounded, never loses it."""
    from standin import standin as S
    base = 1500
    def chat(first_fact, n_filler):
        m = [{"role": "system", "content": "You are a helpful voice assistant."}]
        if first_fact:
            m += [{"role": "user", "content": first_fact}, {"role": "assistant", "content": "Understood, I will keep that in mind."}]
        for t in range(n_filler):
            m += [{"role": "user", "content": FILLER[t % 5].format(c=CITIES[t % 10]) + " Please give a fairly detailed answer, with a few examples."},
                  {"role": "assistant", "content": "Sure. " + ("Here is a detailed answer with examples. " * 6)}]
        m.append({"role": "user", "content": "Anything I should remember about my health?"})
        return m
    # 1. no facts: nothing pinned, effective budget == base
    S.build(chat(None, 40), base); i0 = dict(S.LAST_INFO)
    assert i0["pinned_tokens"] == 0 and i0["effective_budget"] == base, i0
    # 2. a health fact: pinned, critical, survives 40 filler turns, budget rises by only its small cost
    out, tiers = S.build(chat("I have a heart condition, so please avoid anything strenuous.", 40), base); i1 = dict(S.LAST_INFO)
    assert "heart" in " ".join(m["content"] for m in out)
    assert tiers[1]["pinned"] and tiers[1]["critical"], tiers[1]
    assert 0 < i1["pinned_tokens"] <= 60 and i1["effective_budget"] == base + i1["pinned_tokens"], i1
    # 3. flood of ordinary numeric facts: lane stays under the cap, critical fact is never evicted
    m = chat("I have a heart condition, so please avoid anything strenuous.", 10)
    for k in range(80):
        m.insert(3 + 2 * k, {"role": "user", "content": f"Note {k}: code {k * 7919} unit {k} gate {k + 40} room {k + 3}."})
        m.insert(4 + 2 * k, {"role": "assistant", "content": "Noted."})
    out, tiers = S.build(m, base); i2 = dict(S.LAST_INFO)
    assert i2["pinned_tokens"] <= i2["pin_cap"], i2
    assert "heart" in " ".join(x["content"] for x in out), "critical fact evicted by ordinary ones"
    assert i2["pin_overflow"] > 0, i2
    sent = sum(_tokens(x["content"]) for x in out)
    print(f"flood: sent={sent} pinned={i2['pinned_tokens']}/{i2['pin_cap']} overflow={i2['pin_overflow']} critical_kept={i2['critical_count']}")
    # 4. an updated ordinary fact replaces the older one, critical facts are never merged
    m = [{"role": "system", "content": "s"},
         {"role": "user", "content": "My flight is BA 249 and it leaves at 6:40am."}, {"role": "assistant", "content": "Ok."},
         {"role": "user", "content": "Update: my flight is now BA 251 and it leaves at 7:10am."}, {"role": "assistant", "content": "Ok."},
         {"role": "user", "content": "I am allergic to peanuts."}, {"role": "assistant", "content": "Ok."},
         {"role": "user", "content": "I am allergic to shellfish too."}, {"role": "assistant", "content": "Ok."}] + \
        [{"role": "user", "content": f"Filler {i}."} if i % 2 == 0 else {"role": "assistant", "content": "Fine."} for i in range(8)]
    out, tiers = S.build(m, 300)
    txt = " ".join(x["content"] for x in out)
    assert "251" in txt and "peanuts" in txt and "shellfish" in txt, txt
    assert not tiers[1]["pinned"], "older flight message should have been superseded"
    # 5. distinct ordinary facts really fill the lane to the cap; the critical one still survives
    nouns = ("river garage ladder violin castle engine pillow quartz meadow harbor lantern marble orchard pepper saddle thimble "
             "velvet walnut anchor basket cobalt dragon ember falcon glacier hammock island jungle kettle lemon magnet nectar "
             "oyster pebble quilt rocket spruce tundra umbrella voyage willow yacht zephyr bronze candle donkey eagle forest").split()
    m = [{"role": "system", "content": "s"},
         {"role": "user", "content": "I have a heart condition, so please avoid anything strenuous."}, {"role": "assistant", "content": "Ok."}]
    for k, w in enumerate(nouns):
        m += [{"role": "user", "content": f"My {w} number is {1000 + k * 37} and the {w} code is {k + 9}."}, {"role": "assistant", "content": "Noted."}]
    m.append({"role": "user", "content": "What should I avoid?"})
    out, tiers = S.build(m, base); i5 = dict(S.LAST_INFO)
    assert i5["pinned_tokens"] <= i5["pin_cap"] and i5["pin_overflow"] > 0 and i5["critical_count"] == 1, i5
    assert "heart" in " ".join(x["content"] for x in out)
    print("cap check: pinned", i5["pinned_tokens"], "/", i5["pin_cap"], "overflow", i5["pin_overflow"])
    print("dynamic checks ok")


if __name__ == "__main__":
    budget = 1500
    print("turn  naive_tokens  ours_tokens")
    for n in (5, 10, 20, 30, 40, 50):
        msgs = convo(n)
        msgs.append({"role": "user", "content": "Quick check, what deadline did I give you and what should you avoid suggesting?"})
        out, tiers = build(msgs, budget)
        naive = sum(_tokens(m["content"]) for m in msgs)
        ours = sum(_tokens(m["content"]) for m in out)
        print(f"{n:>4}  {naive:>12}  {ours:>11}")
    texts = " ".join(m["content"] for m in out)
    print("deadline kept:", "Friday" in texts, "| peanut kept:", "peanut" in texts)
    from collections import Counter
    print(Counter(t["tier"] for t in tiers))
    assert "Friday" in texts and "peanut" in texts
    assert out[0]["role"] == "system" and out[-1] == msgs[-1]
    assert len(tiers) == len(msgs)
    print("ok")
    check_dynamic()
