"""Plot naive vs ours from the proxy's saved /debug/state files.

Usage (from repo root, after both runs):
  python -m bench.plot
  python -m bench.plot --naive naive --ours ours      (labels used with runner --label)

Reads bench/state_<label>.json. Each file has {"manager_on":..., "turns":[...]},
and each turn has tokens_in, tokens_sent, ttft_ms, total_ms, model.
The proxy's global "turn" counter keeps running between runs, so we use each
record's POSITION in the list as the turn number.

Writes bench/tokens_per_turn.png, bench/tokens_cumulative.png and bench/latency_per_turn.png.
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
        turns = json.load(f)["turns"]
    # A call Gemini refused (and the runner retried) is logged by the proxy without timings.
    # Drop those records: the successful retry is the real turn, and keeping both would
    # count the same turn's tokens twice.
    good = [t for t in turns if t.get("ttft_ms") is not None and t.get("total_ms") is not None]
    if len(good) != len(turns):
        print(f"{label}: ignored {len(turns) - len(good)} refused/failed call records "
              f"({len(good)} turns kept)")
    return good

def col(turns, key):
    return [t[key] for t in turns]

def summary(name, turns):
    ts, ti = col(turns, "tokens_sent"), col(turns, "tokens_in")
    ttft = sorted(col(turns, "ttft_ms"))
    total = sorted(col(turns, "total_ms"))
    p50 = lambda xs: xs[len(xs) // 2]
    mgr = [t["manager_ms"] for t in turns if t.get("manager_ms") is not None]
    mean = lambda xs: sum(xs) / len(xs)
    extra = ""
    if mgr:
        mgr = sorted(mgr)
        extra = (f"  manager_ms median={p50(mgr):.1f} max={mgr[-1]:.1f}"
                 f"  (ttft+manager mean={mean(col(turns, 'ttft_ms')) + mean(mgr):.0f}ms)")
    print(f"{name:6s} turns={len(turns)}  tokens_sent total={sum(ts)}  "
          f"(tokens_in total={sum(ti)})  last-turn sent={ts[-1]}  "
          f"ttft median={p50(ttft)}ms mean={mean(ttft):.0f}ms  total median={p50(total)}ms" + extra)

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

    # Chart 2: cumulative tokens sent (running total). This is the "area under" chart 1,
    # and it is where a saving becomes obvious: naive curves up, a flat manager is a straight line.
    plt.figure(figsize=(9, 5))
    for name, turns in runs.items():
        total, running = 0, []
        for t in turns:
            total += t["tokens_sent"]
            running.append(total)
        plt.plot(range(len(turns)), running, label=name, color=colours[name], linewidth=2)
    plt.xlabel("Turn")
    plt.ylabel("Total tokens sent so far")
    plt.title("Cumulative tokens over the conversation: naive vs ours")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("bench/tokens_cumulative.png", dpi=150)
    plt.close()

    # Chart 3: time to first token (the number that matters for voice), per turn
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
    # Chart 4: the memory manager's own cost per turn (ttft does not include it)
    if any(t.get("manager_ms") is not None for v in runs.values() for t in v):
        plt.figure(figsize=(9, 5))
        for name, turns in runs.items():
            plt.plot(range(len(turns)), [t.get("manager_ms") or 0 for t in turns],
                     label=name, color=colours[name], linewidth=2)
        plt.xlabel("Turn")
        plt.ylabel("Memory manager time (ms)")
        plt.title("Cost of the memory manager per turn")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig("bench/manager_ms_per_turn.png", dpi=150)
        plt.close()
    print("wrote bench/tokens_per_turn.png, bench/tokens_cumulative.png, bench/latency_per_turn.png "
          "(and bench/manager_ms_per_turn.png when manager_ms exists)")

if __name__ == "__main__":
    main()
