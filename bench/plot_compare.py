"""Response-time chart: naive vs v0 vs v2, each the mean of several runs (band = min to max).

Usage (from repo root):
  python -m bench.plot_compare --naive naive1 naive2 naive3 --v0 ours1 ours2 ours3 --v2 v2b-test1 v2b-test2 v2b-test3

Reads bench/state_<label>.json (the same files bench.plot reads; retried calls are merged).
Writes bench/charts_compare/response_time_ttft.png          (time to first LLM token, per turn)
       bench/charts_compare/response_time_with_manager.png  (the same plus the memory manager's own time)
and prints mean, min and max of the per-run mean for each group.

Time to first token is measured at the proxy: it is the wait for the model to start answering, NOT the time to
first spoken audio. The second chart adds the manager's time because that is paid before the model is called.
"""
import argparse, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from bench.plot import load

COLOURS = {"naive": "#c0392b", "v0": "#7d8c8d", "v2": "#2471a3"}


def per_turn(run, with_manager):
    return [(t.get("ttft_ms") or 0) + ((t.get("manager_ms") or 0) if with_manager else 0) for t in run]


def draw(groups, with_manager, title, path):
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, runs in groups.items():
        n = min(len(r) for r in runs)
        ys = [per_turn(r, with_manager)[:n] for r in runs]
        mean = [sum(y[i] for y in ys) / len(ys) for i in range(n)]
        lo = [min(y[i] for y in ys) for i in range(n)]
        hi = [max(y[i] for y in ys) for i in range(n)]
        ax.fill_between(range(n), lo, hi, color=COLOURS[name], alpha=0.15, linewidth=0)
        ax.plot(range(n), mean, color=COLOURS[name], linewidth=2, label=f"{name} (mean of {len(ys)} runs)")
    ax.set_xlabel("Turn")
    ax.set_ylabel("Milliseconds")
    ax.set_title(title + "\nline = mean, shaded band = min to max across runs", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--naive", nargs="+", required=True)
    ap.add_argument("--v0", nargs="+", required=True)
    ap.add_argument("--v2", nargs="+", required=True)
    a = ap.parse_args()

    groups = {}
    for name, labels in (("naive", a.naive), ("v0", a.v0), ("v2", a.v2)):
        runs = [r for r in (load(l) for l in labels) if r]
        if not runs:
            raise SystemExit(f"no state files found for {name}")
        groups[name] = runs
        for wm, tag in ((False, "first-token"), (True, "first-token + manager")):
            means = [sum(per_turn(r, wm)) / len(r) for r in runs]
            print(f"{name:5s} {tag:22s}: mean {sum(means) / len(means):,.0f} ms "
                  f"(min {min(means):,.0f}, max {max(means):,.0f})  [{len(runs)} runs]")

    out = "bench/charts_compare"
    os.makedirs(out, exist_ok=True)
    draw(groups, False, "Time to first token per turn: naive vs v0 vs v2", f"{out}/response_time_ttft.png")
    draw(groups, True, "Time to first token plus memory-manager time: naive vs v0 vs v2",
         f"{out}/response_time_with_manager.png")
    print(f"wrote charts to {out}/")


if __name__ == "__main__":
    main()
