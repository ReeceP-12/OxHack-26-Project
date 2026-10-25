"""Where do the memory manager's milliseconds go?  Run it on the laptop that shows ~600 ms per turn.

    python tools/profile_manager.py            real MiniLM (needs the model; run from the repo root)
    python tools/profile_manager.py --fake     no model, just checks the script and the non-model code

It prints, per turn: total manager time, time inside the embedder, how many texts were actually encoded
(cache misses), time scoring (without embedding) and time filling the budget. Three scenarios:
  A  stable system message (what we expect from ElevenLabs)
  B  system message that changes every turn (e.g. it contains the time). If this is slow and A is fast,
     that is the cause: the ~700 token system text gets re-encoded every turn.
  C  warm micro-benchmarks: one short text, one long text, a batch of three.
Paste the whole output back.
"""
import argparse, os, platform, re, statistics as st, sys, time, zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser()
ap.add_argument("--fake", action="store_true")
ap.add_argument("--turns", type=int, default=30)
args = ap.parse_args()

t0 = time.perf_counter()
import memory.context_manager as cmmod
from memory.context_manager import ContextManager
print(f"python {platform.python_version()} on {platform.system()}, import memory: {1000 * (time.perf_counter() - t0):.0f} ms")
try:
    import torch
    print(f"torch {torch.__version__}, threads {torch.get_num_threads()}")
except Exception as e:
    print("torch not importable:", e)

if args.fake:
    import numpy as np

    class Emb:
        def __init__(self): self._cache = {}
        def embed_many(self, texts):
            for t in dict.fromkeys(texts):
                if t not in self._cache:
                    time.sleep(0.002)
                    v = np.zeros(64, dtype=np.float32)
                    for w in re.findall(r"[a-z']+", t.lower()): v[zlib.crc32(w.encode()) % 64] += 1
                    self._cache[t] = v / (np.linalg.norm(v) or 1)
            return np.stack([self._cache[t] for t in texts])
    emb = Emb()
else:
    from memory.scorer import Embedder
    emb = Embedder()
    t = time.perf_counter(); emb.embed_many(["warmup"])
    print(f"model load + first encode (cold): {1000 * (time.perf_counter() - t):.0f} ms")

T = {"embed": 0.0, "score": 0.0, "fill": 0.0, "new": 0}
_embed, _score, _fill = emb.embed_many, cmmod.score_messages, cmmod.fill_budget

def timed_embed(texts):
    before = len(emb._cache)
    t = time.perf_counter(); r = _embed(texts); T["embed"] += time.perf_counter() - t
    T["new"] += len(emb._cache) - before
    return r

def timed_score(*a, **k):
    t = time.perf_counter(); r = _score(*a, **k); T["score"] += time.perf_counter() - t
    return r

def timed_fill(*a, **k):
    t = time.perf_counter(); r = _fill(*a, **k); T["fill"] += time.perf_counter() - t
    return r

emb.embed_many, cmmod.score_messages, cmmod.fill_budget = timed_embed, timed_score, timed_fill

SYSTEM = ("You are a friendly voice assistant. Keep answers short and natural for speech. " * 12)[:2800]
USER = "Can you tell me something interesting about topic number {i}, and keep it fairly short please?"
ASSIST = "Sure. Here is a short answer about topic {i}. " + "It has a long history and a few surprising details. " * 4

def scenario(name, system_changes):
    emb._cache.clear()
    cm = ContextManager(embedder=emb)
    msgs = [{"role": "system", "content": SYSTEM}]
    rows = []
    for i in range(args.turns):
        if system_changes:
            msgs[0] = {"role": "system", "content": SYSTEM + f" Current time: 12:{i:02d}:00."}
        msgs.append({"role": "user", "content": USER.format(i=i)})
        for k in T: T[k] = 0
        t = time.perf_counter(); cm.build(msgs, 1500); total = time.perf_counter() - t
        rows.append((i + 1, len(msgs), total * 1000, T["embed"] * 1000, T["new"],
                     (T["score"] - T["embed"]) * 1000, T["fill"] * 1000))
        msgs.append({"role": "assistant", "content": ASSIST.format(i=i)})
    print(f"\nScenario {name}")
    print("turn  msgs  total_ms  embed_ms  new_texts  score_ms  fill_ms")
    for r in rows[:3] + rows[-2:]:
        print(f"{r[0]:>4} {r[1]:>5} {r[2]:>9.1f} {r[3]:>9.1f} {r[4]:>10} {r[5]:>9.1f} {r[6]:>8.1f}")
    tot = [r[2] for r in rows]
    print(f"median {st.median(tot):.1f} ms | max {max(tot):.1f} ms | turn 1 {rows[0][2]:.1f} ms | "
          f"embed share {100 * sum(r[3] for r in rows) / sum(tot):.0f}%")

scenario("A: stable system message", False)
scenario("B: system message changes every turn", True)

print("\nScenario C: warm micro-benchmarks (median of 5, texts are new each time)")
def bench(label, make):
    ts = []
    for k in range(5):
        texts = make(k); t = time.perf_counter(); emb.embed_many(texts); ts.append((time.perf_counter() - t) * 1000)
    print(f"  {label:34s} {st.median(ts):7.1f} ms")
bench("1 short text (15 words)", lambda k: [f"short sentence number {k} about volcanoes and rivers today please"])
bench("1 long text (~700 tokens)", lambda k: [SYSTEM + f" variant {k}"])
bench("3 short texts in one batch", lambda k: [f"alpha {k} {j} beta gamma delta epsilon zeta" for j in range(3)])
