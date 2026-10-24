"""Plot naive vs ours from the proxy's saved /debug/state files.

Usage (from repo root, after both runs):
  python -m bench.plot
  python -m bench.plot --naive naive --ours ours      (labels used with runner --label)

Reads bench/state_<label>.json. Each file has {"manager_on":..., "turns":[...]},
and each turn has tokens_in, tokens_sent, ttft_ms, total_ms, model.
The proxy's global "turn" counter keeps running between runs, so we use each
record's POSITION in the list as the turn number.

Writes bench/tokens_per_turn.png and bench/latency_per_turn.png.
"""
import argparse, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load(label):
    path = f"bench/state_{label}.json"
    if not os.path.exists(path):
        print(f"missing {path}, skipping {label}")
        return None
    with open(path) as f:
        return json.load(f)["turns"]

def col(turns, key):
    return [t[key] for t in turns]

def summary(name, turns):
    ts, ti = col(turns, "tokens_sent"), col(turns, "tokens_in")
    ttft = sorted(col(turns, "ttft_ms"))
    total = sorted(col(turns, "total_ms"))
    p50 = lambda xs: xs[len(xs) // 2]
    print(f"{name:6s} turns={len(turns)}  tokens_sent total={sum(ts)}  "
          f"(tokens_in total={sum(ti)})  last-turn sent={ts[-1]}  "
          f"ttft median={p50(ttft)}ms  total median={p50(total)}ms")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--naive", default="naive")
    ap.add_argument("--ours", default="ours")
    a = ap.parse_args()

    runs = {"naive": load(a.naive), "ours": load(a.ours)}
    runs = {k: v for k, v in runs.items() if v}
    if not runs:
        raise SystemExit("no state files found: run the benchmark first")

    colours = {"naive": "#c0392b", "ours": "#2471a3"}
    mock = any(t.get("model") == "mock" for v in runs.values() for t in v)
    note = "  [MOCK MODEL, latency not meaningful]" if mock else ""

    for name, turns in runs.items():
        summary(name, turns)
    if "naive" in runs and "ours" in runs:
        n, o = sum(col(runs["naive"], "tokens_sent")), sum(col(runs["ours"], "tokens_sent"))
        if n:
            print(f"tokens saved by ours: {100 * (n - o) / n:.1f}%  ({n} -> {o})")

    # Chart 1: tokens actually sent to the model, per turn
    plt.figure(figsize=(9, 5))
    for name, turns in runs.items():
        plt.plot(range(len(turns)), col(turns, "tokens_sent"),
                 label=name, color=colours[name], linewidth=2)
    plt.xlabel("Turn")
    plt.ylabel("Tokens sent to the model")
    plt.title("Prompt size per turn: naive vs ours")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("bench/tokens_per_turn.png", dpi=150)
    plt.close()

    # Chart 2: time to first token (the number that matters for voice), per turn
    plt.figure(figsize=(9, 5))
    for name, turns in runs.items():
        plt.plot(range(len(turns)), col(turns, "ttft_ms"),
                 label=name, color=colours[name], linewidth=2)
    plt.xlabel("Turn")
    plt.ylabel("Time to first token (ms)")
    plt.title("Latency per turn: naive vs ours" + note)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("bench/latency_per_turn.png", dpi=150)
    plt.close()
    print("wrote bench/tokens_per_turn.png and bench/latency_per_turn.png")

if __name__ == "__main__":
    main()
