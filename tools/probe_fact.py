"""Offline probe: how does the fact detector see a sentence? No proxy or API keys needed.

    python tools/probe_fact.py                              # a built-in set, incl. "im allergic to peanuts"
    python tools/probe_fact.py "im allergic to peanuts" "my dog is called Rex"

For each sentence it prints:
  regex     importance from the regex rules (0.9 = strong personal fact, 0.4/0.8 = weak hits)
  clf       probability from the learned classifier (pinned when >= its threshold)
  pinned    True if the regex OR the classifier flags it (importance >= 0.8)
  alone     the message's score as the ONLY thing said so far
  in chat   its score after 20 turns of unrelated chatter (the pin floor, 0.7, is what keeps it alive)

A low "clf" with a high "regex" is fine: the two are OR-ed. Only pinned=False is a real miss.
Needs the MiniLM model (downloads ~90MB once, from Hugging Face).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from memory.scorer import Embedder, importance, score_messages, PIN_IMPORTANCE
from memory.fact_classifier import FactClassifier

DEFAULT = ["im allergic to peanuts", "I'm allergic to peanuts.", "Im allergic to peanuts",
           "peanuts make me really ill", "I can't have nuts", "my dog is called Biscuit",
           "recommend a vegan restaurant", "is a peanut allergy common?", "what's the weather like"]
CHATTER = ["What's a good way to learn guitar chords?", "Tell me something about octopuses.",
           "Why is the sky blue?", "How do bees make honey?", "Suggest a short stretching routine."] * 4

def main():
    sents = sys.argv[1:] or DEFAULT
    emb, clf = Embedder(), FactClassifier.load()
    print(f"classifier: {'loaded, threshold ' + str(clf.threshold) if clf else 'NOT LOADED (regex only) - is memory/fact_clf.npz there?'}\n")
    print(f"{'sentence':40s} {'regex':>5s} {'clf':>5s} {'pinned':>6s} {'alone':>6s} {'in chat':>7s}")
    probs = clf.probs(sents, emb) if clf else [float('nan')] * len(sents)
    for s, p in zip(sents, probs):
        rx = importance(s)
        pinned = rx >= PIN_IMPORTANCE or (clf is not None and p >= clf.threshold)
        alone = score_messages([{"role": "user", "content": s}], emb, clf)[0]["score"]
        chat = [{"role": "user", "content": s}]
        for c in CHATTER:
            chat += [{"role": "user", "content": c}, {"role": "assistant", "content": "Sure, here is a short answer."}]
        chat.append({"role": "user", "content": "ok go on"})
        in_chat = score_messages(chat, emb, clf)[0]["score"]
        print(f"{s[:40]:40s} {rx:5.2f} {p:5.2f} {str(pinned):>6s} {alone:6.2f} {in_chat:7.2f}")

if __name__ == "__main__":
    main()
