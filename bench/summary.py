"""Summarise repeated runs: mean, min and max across runs, for naive vs ours.

Usage (from repo root, after several runs of each mode):
  python -m bench.summary --naive naive1 naive2 naive3 --ours ours1 ours2 ours3

Each label is a run made with:  python -m bench.runner --mode off|on --label <label>
It reads bench/state_<label>.json (tokens, time to first token, manager time) and
bench/replies_<label>.json (recall answers). Runs that came from other people's machines
work too: copy their state_*.json and replies_*.json into bench/.

A run with failed turns or errored recall questions is listed, but you decide whether to
trust it. Refused calls that the runner retried are ignored (see plot.py).
"""
import argparse, json, os

def dedupe_retries(turns):
    """A turn the runner retried (refused call or empty reply) leaves extra records in the proxy
    log. Retries send the same history, so they share `messages_in` with the final, successful
    attempt. Keep only the LAST record of each run of identical messages_in."""
    out = []
    for t in turns:
        if out and t.get("messages_in") is not None and out[-1].get("messages_in") == t.get("messages_in"):
            out[-1] = t
        else:
            out.append(t)
    return out

def load_state(label):
    with open(f"bench/state_{label}.json") as f:
        turns = json.load(f)["turns"]
    good = [t for t in turns if t.get("ttft_ms") is not None and t.get("total_ms") is not None]
    return dedupe_retries(good)

def load_recall(label):
    path = f"bench/replies_{label}.json"
    if not os.path.exists(path):
        return None, None, 0
    with open(path) as f:
        answers = json.load(f)
    return sum(1 for r in answers if r["correct"]), len(answers), \
        sum(1 for r in answers if not r["reply"].strip())

def run_stats(label):
    turns = load_state(label)
    mean = lambda xs: sum(xs) / len(xs) if xs else None
    mgr = [t["manager_ms"] for t in turns if t.get("manager_ms") is not None]
    correct, answered, empty = load_recall(label)
    return {
        "label": label,
        "turns": len(turns),
        "tokens_total": sum(t["tokens_sent"] for t in turns),
        "tokens_last": turns[-1]["tokens_sent"],
        "ttft_mean": mean([t["ttft_ms"] for t in turns]),
        "manager_ms_mean": mean(mgr),
        "recall": correct,
        "answered": answered,
        "empty": empty,
    }

def spread(values, fmt="{:.0f}"):
    vals = [v for v in values if v is not None]
    if not vals:
        return "-"
    return (f"mean {fmt.format(sum(vals) / len(vals))}  "
            f"(min {fmt.format(min(vals))}, max {fmt.format(max(vals))})")

def report(name, labels):
    runs = []
    for lab in labels:
        try:
            runs.append(run_stats(lab))
        except FileNotFoundError:
            print(f"  missing files for {lab}, skipped")
    print(f"\n{name}: {len(runs)} runs ({', '.join(r['label'] for r in runs)})")
    if not runs:
        return runs
    for r in runs:
        flag = "" if r["turns"] == 50 else f"   <-- only {r['turns']} turns recorded"
        if r["empty"]:
            flag += f"   <-- {r['empty']} EMPTY reply (model returned nothing; not counted as forgetting)"
        if r["answered"] is not None and r["answered"] < 5:
            flag += f"   <-- only {r['answered']} recall questions answered"
        print(f"  {r['label']:10s} recall {r['recall']}/{r['answered']}  tokens {r['tokens_total']}  "
              f"last turn {r['tokens_last']}  ttft mean {r['ttft_mean']:.0f} ms{flag}")
    print("  recall (of 5)      :", spread([r["recall"] for r in runs], "{:.1f}"))
    print("  total tokens sent  :", spread([r["tokens_total"] for r in runs]))
    print("  last-turn tokens   :", spread([r["tokens_last"] for r in runs]))
    print("  ttft mean (ms)     :", spread([r["ttft_mean"] for r in runs]))
    if any(r["manager_ms_mean"] is not None for r in runs):
        print("  manager ms (mean)  :", spread([r["manager_ms_mean"] for r in runs], "{:.1f}"))
    return runs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--naive", nargs="+", required=True)
    ap.add_argument("--ours", nargs="+", required=True)
    a = ap.parse_args()
    n = report("naive", a.naive)
    o = report("ours", a.ours)
    if n and o:
        mean = lambda xs: sum(xs) / len(xs)
        nt, ot = mean([r["tokens_total"] for r in n]), mean([r["tokens_total"] for r in o])
        nl, ol = mean([r["ttft_mean"] for r in n]), mean([r["ttft_mean"] for r in o])
        print(f"\nmean tokens saved by ours: {100 * (nt - ot) / nt:.1f}%")
        print(f"mean ttft change (ours vs naive): {100 * (ol - nl) / nl:+.1f}%  "
              "(only trust this when the min/max ranges do not overlap)")

if __name__ == "__main__":
    main()
