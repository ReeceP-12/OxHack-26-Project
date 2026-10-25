"""Learned "is this a personal fact worth keeping?" detector.

A logistic regression on the MiniLM sentence embeddings we already compute for similarity, so
inference is one small matrix product and needs no extra model and NO LLM call. It is applied
per sentence (a fact buried in a long message still counts) and OR-ed with the regex rules in
scorer.py, because missing an allergy costs far more than pinning one extra message.

Weights live in memory/fact_clf.npz, produced by:  python -m memory.train_fact_classifier
If that file is missing the manager silently uses the regex alone.
"""
import os
import re
import numpy as np

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fact_clf.npz")
_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def split_sentences(text):
    """Sentences worth classifying. Very short fragments ("ok.") are kept: they are cheap and are
    labelled as non-facts in the training data."""
    parts = [p.strip() for p in _SPLIT.split(text.replace("\u2019", "'")) if p and p.strip()]
    return parts or [text.strip()]


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def fit_logistic(X, y, l2=0.1, pos_weight=1.5, iters=30):
    """L2-regularised logistic regression by Newton's method (IRLS). numpy only, no tuning of a
    learning rate, converges in a handful of steps. Returns (w, b); the intercept is not penalised."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, d = X.shape
    A = np.hstack([X, np.ones((n, 1))])
    sw = np.where(y == 1, pos_weight, 1.0)
    reg = np.eye(d + 1) * l2
    reg[-1, -1] = 0.0
    theta = np.zeros(d + 1)
    for _ in range(iters):
        p = _sigmoid(A @ theta)
        grad = A.T @ (sw * (p - y)) + reg @ theta
        W = sw * p * (1 - p)
        H = (A * W[:, None]).T @ A + reg + 1e-8 * np.eye(d + 1)
        step = np.linalg.solve(H, grad)
        theta -= step
        if np.max(np.abs(step)) < 1e-6:
            break
    return theta[:-1].astype(np.float32), float(theta[-1])


class FactClassifier:
    def __init__(self, w, b, threshold=0.4, model_name="all-MiniLM-L6-v2"):
        self.w = np.asarray(w, dtype=np.float32)
        self.b = float(b)
        self.threshold = float(threshold)
        self.model_name = model_name
        self._cache = {}

    # ---- persistence ----
    def save(self, path=DEFAULT_PATH):
        np.savez(path, w=self.w, b=self.b, threshold=self.threshold, model_name=self.model_name)

    @classmethod
    def load(cls, path=DEFAULT_PATH):
        """Returns None if the file is missing or unreadable (the manager then uses regex only)."""
        try:
            d = np.load(path, allow_pickle=False)
            return cls(d["w"], float(d["b"]), float(d["threshold"]), str(d["model_name"]))
        except Exception:
            return None

    # ---- inference ----
    def prob_embeddings(self, E):
        return _sigmoid(np.asarray(E, dtype=np.float32) @ self.w + self.b)

    def probs(self, texts, embedder):
        """Probability that each text states a personal fact (max over its sentences)."""
        todo = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if todo:
            sents = [split_sentences(t) for t in todo]
            flat = [s for group in sents for s in group]
            p = self.prob_embeddings(embedder.embed_many(flat))
            k = 0
            for t, group in zip(todo, sents):
                self._cache[t] = float(p[k:k + len(group)].max())
                k += len(group)
        return [self._cache[t] for t in texts]
