"""The one function the proxy calls."""
import os

from memory.scorer import Embedder, score_messages
from memory.tiers import fill_budget
from memory.fact_classifier import FactClassifier

DEFAULT_BUDGET = 1500   # total tokens sent, including ElevenLabs' ~700 token system message


def _off(name):
    return os.environ.get(name, "").strip().lower() in ("off", "0", "false", "no")


class ContextManager:
    def __init__(self, embedder=None, keep_last=4, fact_clf="auto"):
        """fact_clf: "auto" loads memory/fact_clf.npz if it exists AND was trained with this
        embedder's model; None disables it (regex only); or pass a FactClassifier.

        Environment switches (only in "auto" mode), handy for the benchmark:
          FACT_CLF=off          ignore the classifier file (regex only)
          FACT_THRESHOLD=0.5    override the threshold saved in the file, no retraining needed
        """
        self.embedder = embedder or Embedder()
        self.keep_last = keep_last
        if fact_clf == "auto":
            fact_clf = None if _off("FACT_CLF") else FactClassifier.load()
            if fact_clf is not None and getattr(self.embedder, "model_name", None) != fact_clf.model_name:
                fact_clf = None
            override = os.environ.get("FACT_THRESHOLD", "").strip()
            if fact_clf is not None and override:
                try:
                    fact_clf.threshold = float(override)
                except ValueError:
                    print(f"FACT_THRESHOLD={override!r} is not a number, using {fact_clf.threshold}", flush=True)
        self.fact_clf = fact_clf

    def build(self, messages, budget=DEFAULT_BUDGET):
        """messages: OpenAI-style list. Returns (reduced_messages, tiers)."""
        scores = score_messages(messages, self.embedder, self.fact_clf)
        return fill_budget(messages, scores, budget, keep_last=self.keep_last)
