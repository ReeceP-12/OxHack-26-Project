"""Eyeball the scores with the REAL MiniLM.  Run from repo root:  python -m memory.try_scorer"""
from memory.context_manager import ContextManager

msgs = [{"role": "system", "content": "You are a helpful voice assistant."},
        {"role": "user", "content": "My deadline is Friday at 5pm and I'm allergic to peanuts."},
        {"role": "assistant", "content": "Got it, Friday 5pm and a peanut allergy."}]
for t in ["the history of bicycles", "how volcanoes work", "a recipe for pancakes",
          "why the sky is blue", "tips for learning guitar", "how tides work"]:
    msgs += [{"role": "user", "content": f"Tell me about {t}."},
             {"role": "assistant", "content": f"Sure. Here is a fairly long explanation of {t}. " + "More detail here. " * 12}]
msgs.append({"role": "user", "content": "Sorry, when was that deadline I mentioned?"})

kept, tiers = ContextManager().build(msgs, budget=350)
print(f"{'idx':>3} {'role':<9} {'tier':<9} {'score':>5} {'tok':>4}  text")
for t in tiers:
    print(f"{t['index']:>3} {t['role']:<9} {t['tier']:<9} {t['score']:>5} {t['tokens']:>4}  {msgs[t['index']]['content'][:50]}")
print(f"\n{len(msgs)} messages in, {len(kept)} sent")
