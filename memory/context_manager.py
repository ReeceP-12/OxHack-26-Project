"""The one function the proxy calls."""
from memory.scorer import Embedder, score_messages
from memory.tiers import fill_budget

DEFAULT_BUDGET = 1500   # total tokens sent, including ElevenLabs' ~700 token system message


class ContextManager:
    def __init__(self, embedder=None, keep_last=4):
        self.embedder = embedder or Embedder()
        self.keep_last = keep_last

    def build(self, messages, budget=DEFAULT_BUDGET):
        """messages: OpenAI-style list. Returns (reduced_messages, tiers)."""
        scores = score_messages(messages, self.embedder)
        return fill_budget(messages, scores, budget, keep_last=self.keep_last)
