"""Run the scripted conversation against the proxy and log per-turn metrics.

Usage (from repo root, proxy running on :8000):
  python -m bench.runner --mode off --label naive
  python -m bench.runner --mode on  --label ours
The runner flips the proxy's switch (POST /debug/mode), resets the turn log
(POST /debug/reset), always sends the FULL history, then saves /debug/state.
"""
import argparse, csv, json, time, requests
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
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--mode", choices=["on", "off"], required=True,
                    help="on = context manager, off = naive")
    ap.add_argument("--label", required=True)
    ap.add_argument("--model", default="gemini-flash")
    a = ap.parse_args()
    url = a.base + "/v1/chat/completions"

    # set mode and clear the turn log so runs don't mix
    r = requests.post(a.base + "/debug/mode", json={"manager_on": a.mode == "on"}, timeout=10)
    r.raise_for_status()
    requests.post(a.base + "/debug/reset", timeout=10).raise_for_status()
    print(f"mode={a.mode}, log reset")

    history, rows = [], []
    for i, text in enumerate(build_turns()):
        history.append({"role": "user", "content": text})
        reply, lat, ptok = chat(url, a.model, history)
        history.append({"role": "assistant", "content": reply})
        rows.append({"label": a.label, "mode": a.mode, "turn": i,
                     "latency_s": round(lat, 3), "prompt_tokens": ptok})
        print(f"turn {i:2d}  {lat:5.2f}s  prompt_tokens={ptok}")

    # save the tier map / turn records before recall questions add extra turns
    try:
        state = requests.get(a.base + "/debug/state", timeout=10).json()
        with open(f"bench/state_{a.label}.json", "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        print("could not save /debug/state:", e)

    score = 0
    for q, kws in RECALL:
        reply, _, _ = chat(url, a.model, history + [{"role": "user", "content": q}])
        ok = any(k in reply.lower() for k in kws)
        score += ok
        print(f"[{'OK' if ok else 'MISS'}] {q} -> {reply[:80]!r}")
    print(f"\nRECALL {a.label}: {score}/{len(RECALL)}")

    with open(f"bench/results_{a.label}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)

if __name__ == "__main__":
    main()
