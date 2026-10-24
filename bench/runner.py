"""Run the scripted conversation against an OpenAI-compatible endpoint and log per-turn metrics.

Usage:
  python -m bench.runner --url http://localhost:8000/v1/chat/completions --label ours
  python -m bench.runner --url http://localhost:8000/v1/chat/completions --label naive
(The proxy decides whether to manage context. Runner always sends the FULL history.)
"""
import argparse, csv, time, requests
from bench.conversation import build_turns, RECALL

def chat(url, model, messages):
    t0 = time.perf_counter()
    r = requests.post(url, json={"model": model, "messages": messages}, timeout=60)
    r.raise_for_status()
    latency = time.perf_counter() - t0
    data = r.json()
    reply = data["choices"][0]["message"]["content"]
    prompt_tokens = (data.get("usage") or {}).get("prompt_tokens", -1)
    return reply, latency, prompt_tokens

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--model", default="gemini-flash")
    a = ap.parse_args()

    history, rows = [], []
    for i, text in enumerate(build_turns()):
        history.append({"role": "user", "content": text})
        reply, lat, ptok = chat(a.url, a.model, history)
        history.append({"role": "assistant", "content": reply})
        rows.append({"label": a.label, "turn": i, "latency_s": round(lat, 3), "prompt_tokens": ptok})
        print(f"turn {i:2d}  {lat:5.2f}s  prompt_tokens={ptok}")

    score = 0
    for q, kws in RECALL:
        # ask on a copy so recall questions don't pollute each other
        reply, _, _ = chat(a.url, a.model, history + [{"role": "user", "content": q}])
        ok = any(k in reply.lower() for k in kws)
        score += ok
        print(f"[{'OK' if ok else 'MISS'}] {q} -> {reply[:80]!r}")
    print(f"\nRECALL {a.label}: {score}/{len(RECALL)}")

    with open(f"bench/results_{a.label}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)

if __name__ == "__main__":
    main()
