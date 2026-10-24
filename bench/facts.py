"""Per-fact report: where did each planted fact end up, and was it recalled?

Usage (from repo root, after a run with the memory manager on):
  python -m bench.facts --label ours

Reads bench/state_<label>.json (the proxy's per-turn records, including `tiers`),
bench/transcript_<label>.json and bench/replies_<label>.json (both written by the runner).
For each planted fact it prints the tier the fact's message had on the LAST turn
(full / summary / archived), its score, and whether the matching recall question was
answered correctly. A fact that is "archived" was dropped from the prompt entirely.

Naive runs have no tiers (every message is sent in full), so there is nothing to show.
"""
import argparse, json
from bench.conversation import PLANTED, RECALL

# which recall question (index into RECALL) checks each planted turn
QUESTION_FOR = {1: 0, 4: 1, 30: 1, 7: 2, 11: 3, 15: 4, 40: 4}

def load(path):
    with open(path) as f:
        return json.load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    a = ap.parse_args()

    state = load(f"bench/state_{a.label}.json")
    transcript = load(f"bench/transcript_{a.label}.json")
    try:
        answers = {r["question"]: r for r in load(f"bench/replies_{a.label}.json")}
    except FileNotFoundError:
        answers = {}

    last = next((t for t in reversed(state["turns"]) if t.get("tiers")), None)
    if not last:
        raise SystemExit("no tiers in the state file: run with the memory manager on (MANAGER=memory, --mode on)")
    tiers = last["tiers"]

    # tiers cover every message the proxy received on that turn (the runner's system
    # message comes first, so it shifts every index by `offset`); the transcript has the
    # same messages without the system message and with the final reply on the end.
    offset = last["messages_in"] - (len(transcript) - 1)

    print(f"{a.label}: tiers from turn record {last.get('turn')} "
          f"({last['messages_in']} messages in, {last['messages_sent']} sent)\n")
    print(f"{'planted at':>10}  {'tier':9} {'score':>6} {'tokens':>6}  recalled?  fact")
    for turn, text in sorted(PLANTED.items()):
        try:
            ti = next(i for i, m in enumerate(transcript) if m["role"] == "user" and m["content"] == text) + offset
            entry = tiers[ti]
            tier, score, tok = entry["tier"], entry.get("score"), entry.get("tokens")
        except (StopIteration, IndexError, KeyError):
            tier, score, tok = "not found", None, None
        q = RECALL[QUESTION_FOR[turn]][0]
        rec = answers.get(q)
        recalled = "n/a" if rec is None else ("yes" if rec["correct"] else "NO")
        s = f"{score:.2f}" if isinstance(score, (int, float)) else "-"
        print(f"turn {turn:>5}  {tier:9} {s:>6} {tok if tok is not None else '-':>6}  {recalled:9}  {text[:55]}")
    print("\nNote: turn 4 (old deadline) is superseded by turn 30, so its tier matters less.")

if __name__ == "__main__":
    main()
