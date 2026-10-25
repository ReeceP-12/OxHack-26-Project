"""A long, realistic conversation for the dashboard: 150 prompts, 40 to 80 word messages and replies.

Why: in mock mode the proxy's replies are one short line, so the history stays tiny, the memory budget never
binds, and the dashboard looks the same as naive. This script supplies its own realistic assistant replies, so
the history grows like a real call. Facts are planted at turns 2, 10 and 25 and asked about at the end.

    python tools/demo_long.py                       # 150 prompts against http://localhost:8000
    python tools/demo_long.py --turns 250 --base http://localhost:8000
Run the proxy first, then watch the dashboard (Use fake data UNticked).
Try:  MEMORY_SCALE=off  vs  MEMORY_SCALE=quad  (restart the proxy between runs).
"""
import argparse
import random
import httpx

FACTS = {
    2: "Quick heads up, I'm severely allergic to peanuts, so keep that in mind for anything food related.",
    10: "My project deadline is Friday the 14th, please remember that.",
    25: "My dog is called Biscuit and he's a three year old beagle.",
}
TOPICS = ["octopuses", "the history of the bicycle", "how tides work", "learning guitar chords", "northern lights",
          "volcanoes", "making a decent cup of coffee", "how vaccines work", "board games for four people",
          "why the sky is blue", "migrating birds", "keeping a houseplant alive", "how airplanes stay up",
          "the deepest part of the ocean", "mental arithmetic tricks", "how compasses work", "why cats purr"]
PADS = ["I would like a reasonably detailed answer with a couple of examples, as if I had never come across it. ",
        "Keep it friendly and practical, and mention anything surprising that most people get wrong about it. ",
        "I only have a few minutes, so give me the key points first and then a bit of the background. ",
        "Explain it the way you would to a curious teenager, with one good analogy if you can think of one. ",
        "Start from the very basics, then tell me what an expert would find interesting about it. "]
REPLY = ("Happy to help with that. {t} is a topic with a lot of interesting detail, so here is a short tour. "
         "First, the basic idea is simpler than it sounds, and most of the confusion comes from a few common "
         "myths. Second, there are two or three examples that make it concrete, and each one shows a different "
         "side of it. Finally, if you want to go further, I can suggest where to look next or go through any "
         "part of this in more depth, whichever you prefer.")
QUESTIONS = [("What am I allergic to?", "peanut"), ("What is my project deadline?", "14"), ("What is my dog's name?", "biscuit")]


_RNG = random.Random(7)   # fixed seed: varied but repeatable, so the tier map has no repeating pattern
_PLAN = [(_RNG.randrange(len(TOPICS)), _RNG.randrange(len(PADS))) for _ in range(1000)]


def user_text(i):
    t, p = _PLAN[i]
    return FACTS.get(i) or f"Tell me about {TOPICS[t]}, number {i}. {PADS[p]}"


def reply_text(i):
    return REPLY.format(t=TOPICS[_PLAN[i][0]].capitalize())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--turns", type=int, default=150)
    a = ap.parse_args()
    httpx.post(a.base + "/debug/reset", timeout=10)
    httpx.get(a.base + "/debug/mode/on", timeout=10)
    msgs = [{"role": "system", "content": "You are a calm, polite voice assistant."}]
    for i in range(a.turns):
        msgs.append({"role": "user", "content": user_text(i)})
        r = httpx.post(a.base + "/v1/chat/completions", json={"messages": msgs}, timeout=60)
        r.raise_for_status()
        msgs.append({"role": "assistant", "content": reply_text(i)})   # our own realistic reply, not the mock's
        if (i + 1) % 25 == 0:
            t = httpx.get(a.base + "/debug/state", timeout=10).json()["turns"][-1]
            info = t.get("manager_info") or {}
            print(f"turn {i + 1:3d}: naive {t['tokens_in']:6d}  sent {t['tokens_sent']:5d}  "
                  f"budget {info.get('effective_budget', '?')}  scale {info.get('scale', {}).get('scaled_base', '?')}")
    tiers = httpx.get(a.base + "/debug/state", timeout=10).json()["turns"][-1].get("tiers", [])
    for idx, label in ((2 * 2 + 1, "allergy"), (2 * 10 + 1, "deadline"), (2 * 25 + 1, "dog")):
        print(f"planted {label:8s} message tier at the end:",
              next((x["tier"] for x in tiers if x["index"] == idx), "?"))


if __name__ == "__main__":
    main()
