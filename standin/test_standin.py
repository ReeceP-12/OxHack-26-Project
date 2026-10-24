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
