"""For each recall question: which tier was the relevant planted fact in WHEN IT WAS ASKED?

facts.py shows the tiers on the last chatter turn (before the recall questions). A fact can be
archived then and still be brought back when the question about it arrives, because the
manager scores everything against the newest user message. This script shows that.

Usage (from repo root):
  python -m bench.recall_tiers --label ours1
It reads bench/state_<label>_recall.json when the runner saved one (new runs do), otherwise it
asks the live proxy (/debug/state), which only works until the next run resets the log.
"""
import argparse, json, os, requests
from bench.conversation import PLANTED, RECALL, RECALL_PARAPHRASE

# recall question index -> the planted turns that answer it
FACTS_FOR = {0: [1], 1: [30, 4], 2: [7], 3: [11], 4: [15, 40]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--paraphrase", action="store_true",
                    help="the run was made with --paraphrase: also show the reworded questions")
    a = ap.parse_args()

    path = f"bench/state_{a.label}_recall.json"
    if os.path.exists(path):
        turns = json.load(open(path))["turns"]
        print(f"(from {path})")
    else:
        turns = requests.get(a.base + "/debug/state", timeout=10).json()["turns"]
        print("(from the live proxy log: only valid until the next run resets it)")
    transcript = json.load(open(f"bench/transcript_{a.label}.json"))

    good = [t for t in turns if t.get("tiers") and t.get("ttft_ms") is not None]
    questions = list(RECALL) + (list(RECALL_PARAPHRASE) if a.paraphrase else [])
    recall_turns = good[-len(questions):]
    if len(recall_turns) < len(questions):
        raise SystemExit("could not find the recall-question turns in the state")

    for n, (q, _) in enumerate(questions):
        qi = n % len(RECALL)          # reworded questions repeat the same five facts, in order
        rt = recall_turns[n]
        offset = rt["messages_in"] - len(transcript) - 1   # system message(s) before the history; 1 for the question
        label = " (reworded)" if n >= len(RECALL) else ""
        print(f"\n{q}{label}   ({rt['messages_in']} messages in, {rt['messages_sent']} sent)")
        for turn in FACTS_FOR[qi]:
            text = PLANTED[turn]
            try:
                ti = next(i for i, m in enumerate(transcript)
                          if m["role"] == "user" and m["content"] == text) + offset
                e = rt["tiers"][ti]
                print(f"   fact from turn {turn:>2}: {e['tier']:9} score {e.get('score', 0):.2f}  tokens sent {e.get('tokens')}")
            except (StopIteration, IndexError, KeyError):
                print(f"   fact from turn {turn:>2}: not found")

if __name__ == "__main__":
    main()
