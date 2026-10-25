"""Compare budget curves on long conversations (no model, no proxy): tokens sent versus context retained.

    python tools/sim_scaling.py                 # fake embedder, a few seconds
    python tools/sim_scaling.py --real          # real MiniLM (slower, more faithful similarity)
    python tools/sim_scaling.py --lengths 50 100 200 400

"callback" = a late question about an early, non-fact topic (octopuses). It passes if that old exchange is still
sent in full or as a summary. "any" = share of all messages still present in some form (full, summary or gist).
"""
import argparse, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tests"))

from bench.conversation import build_turns, CHATTER
from memory.scorer import text_of
from memory.tiers import count_tokens
from memory.context_manager import ContextManager

CALLBACK = "Going back to octopuses, what were you telling me about them?"


def make_convo(n_turns):
    turns = list(build_turns())
    k = 0
    while len(turns) < n_turns:
        turns.append(CHATTER[k % len(CHATTER)] + f" (round {k // len(CHATTER) + 2})")
        k += 1
    turns = turns[:n_turns]
    msgs = [{"role": "system", "content": "You are a calm, polite assistant. " * 18}]     # ~700 tokens, like ElevenLabs
    for t in turns:
        msgs += [{"role": "user", "content": t},
                 {"role": "assistant", "content": f"Sure. {t} Here is more detail about that. "
                                                  + "It is a fairly long explanation with plenty of words. " * 5}]
    msgs.append({"role": "user", "content": CALLBACK})
    return msgs


def run(embedder, lengths, modes, base):
    print(f"base budget {base}; system message ~700 tokens; replies ~85 tokens\\n")
    print(f"{'turns':>5} {'mode':>7} {'base→eff':>11} {'sent':>6} {'naive':>6} {'sent%':>6} {'any%':>5} {'callback':>9}")
    for n in lengths:
        msgs = make_convo(n)
        naive = sum(count_tokens(text_of(m)) for m in msgs)
        old_u = next(i for i, m in enumerate(msgs) if "octopuses" in m["content"] and m["role"] == "user")
        for mode in modes:
            os.environ["MEMORY_SCALE"] = mode
            cm = ContextManager(embedder, fact_clf=None)
            kept, tiers = cm.build(msgs, base)
            from memory import tiers as T
            eff = T.LAST_INFO.get("scale", {}).get("scaled_base", base)
            sent = sum(t["tokens"] for t in tiers)
            body = [t for t in tiers if t["role"] != "system"]
            anyp = sum(t["tier"] != "archived" for t in body) / len(body)
            cb = all(tiers[i]["tier"] in ("full", "summary") for i in (old_u, old_u + 1))
            print(f"{n:5} {mode:>7} {base:>5}→{eff:<5} {sent:6} {naive:6} {100 * sent / naive:5.0f}% {100 * anyp:4.0f}% "
                  f"{'kept' if cb else 'LOST':>9}")
        print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--lengths", type=int, nargs="+", default=[25, 50, 100, 200, 400])
    ap.add_argument("--modes", nargs="+", default=["off", "sqrt", "linear", "quad"])
    ap.add_argument("--base", type=int, default=1500)
    a = ap.parse_args()
    if a.real:
        from memory.scorer import Embedder
        emb = Embedder()
    else:
        from test_memory import FakeEmbedder
        emb = FakeEmbedder()
    run(emb, a.lengths, a.modes, a.base)
