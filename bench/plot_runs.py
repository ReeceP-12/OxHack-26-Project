"""Plot several runs per condition: mean line with a min-to-max band.

Usage (from repo root):
  python -m bench.plot_runs --naive naive1 naive2 naive3 --ours v2b-test1 v2b-test2 v2b-test3 --tag v2

Reads bench/state_<label>.json for every label, the same files bench.plot reads. The turn number is the
record's position (retried calls are merged, as in bench.plot). Runs are cut to the shortest run so every
point is the mean of the same number of runs.

Writes into bench/charts_<tag>/ (so charts from different versions never overwrite each other):
  tokens_per_turn.png, tokens_cumulative.png, latency_per_turn.png, manager_ms_per_turn.png
and prints the mean, min and max of the totals.
"""
import argparse, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from bench.plot import load, col

COLOURS = {"naive": "#c0392b", "ours": "#2471a3"}


def series(runs, key):
    n = min(len(r) for r in runs)
    return [[(t.get(key) or 0) for t in r[:n]] for r in runs]


def band(ax, runs, key, name, cumulative=False):
    ys = series(runs, key)
    if cumulative:
        out = []
        for y in ys:
            tot, run = 0, []
            for v in y:
                tot += v
                run.append(tot)
            out.append(run)
        ys = out
    n = len(ys[0])
    mean = [sum(y[i] for y in ys) / len(ys) for i in range(n)]
    lo = [min(y[i] for y in ys) for i in range(n)]
    hi = [max(y[i] for y in ys) for i in range(n)]
    c = COLOURS[name]
    ax.fill_between(range(n), lo, hi, color=c, alpha=0.2, linewidth=0)
    ax.plot(range(n), mean, color=c, linewidth=2, label=(f"{name} (mean of {len(ys)} runs)" if len(ys) > 1 else f"{name} (1 run)"))


def chart(groups, key, title, ylabel, path, cumulative=False):
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, runs in groups.items():
        band(ax, runs, key, name, cumulative)
    ax.set_xlabel("Turn")
    ax.set_ylabel(ylabel)
    many = any(len(r) > 1 for r in groups.values())
    ax.set_title(title + ("\nline = mean, shaded band = min to max across runs" if many else "\nsingle run each"), fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def stats(xs):
    return f"mean {sum(xs) / len(xs):,.0f} (min {min(xs):,.0f}, max {max(xs):,.0f})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--naive", nargs="+", required=True)
    ap.add_argument("--ours", nargs="+", required=True)
    ap.add_argument("--tag", default="runs")
    a = ap.parse_args()

    groups = {}
    for name, labels in (("naive", a.naive), ("ours", a.ours)):
        runs = [load(l) for l in labels]
        runs = [r for r in runs if r]
        if not runs:
            raise SystemExit(f"no state files found for {name}")
        groups[name] = runs
        totals = [sum(col(r, "tokens_sent")) for r in runs]
        last = [r[-1]["tokens_sent"] for r in runs]
        ttft = [sum(col(r, "ttft_ms")) / len(r) for r in runs]
        print(f"{name}: {len(runs)} runs, turns per run {[len(r) for r in runs]}")
        print(f"  total tokens   : {stats(totals)}")
        print(f"  last-turn      : {stats(last)}")
        print(f"  ttft mean (ms) : {stats(ttft)}")

    out = f"bench/charts_{a.tag}"
    os.makedirs(out, exist_ok=True)
    chart(groups, "tokens_sent", "Prompt size per turn: naive vs ours", "Tokens sent to the model",
          f"{out}/tokens_per_turn.png")
    chart(groups, "tokens_sent", "Cumulative tokens over the conversation: naive vs ours",
          "Total tokens sent so far", f"{out}/tokens_cumulative.png", cumulative=True)
    chart(groups, "ttft_ms", "Time to first token per turn: naive vs ours", "Time to first token (ms)",
          f"{out}/latency_per_turn.png")
    if any(t.get("manager_ms") is not None for r in groups["ours"] for t in r):
        only = {"ours": groups["ours"]}
        chart(only, "manager_ms", "Cost of the memory manager per turn", "Memory manager time (ms)",
              f"{out}/manager_ms_per_turn.png")
    print(f"wrote charts to {out}/")


if __name__ == "__main__":
    main()
