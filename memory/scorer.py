"""Scores every message in a conversation from 0 to 1 for "how worth keeping".

No LLM calls here. The only model is MiniLM, which runs locally in a few ms.

score = W_SIM*similarity + W_REC*recency + W_IMP*importance + W_REF*referenced
"""
import re
import numpy as np

# Starting guesses. Tune these once the benchmark runs.
W_SIM, W_REC, W_IMP, W_REF = 0.4, 0.2, 0.3, 0.1
HALF_LIFE_TURNS = 10      # a message this many turns old has recency 0.5
REF_THRESHOLD = 0.6       # later message counts as "referring back" above this similarity
REF_SATURATE = 3          # this many later references = referenced score of 1.0
PIN_IMPORTANCE = 0.8      # user messages at least this important are treated as facts to keep
PIN_FLOOR = 0.7           # ...and never score below this, however off-topic they are now


def text_of(msg):
    """OpenAI content can be a string or a list of parts. Always return a string."""
    c = msg.get("content")
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    return " ".join(p.get("text", "") for p in c if isinstance(p, dict))


class Embedder:
    """MiniLM wrapper with a cache, so each distinct text is embedded once ever.

    ElevenLabs resends the whole history every turn, so without the cache we
    would re-embed everything every turn.
    """

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._cache = {}

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_many(self, texts):
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            vecs = self._load().encode(missing, normalize_embeddings=True)
            for t, v in zip(missing, vecs):
                self._cache[t] = np.asarray(v, dtype=np.float32)
        return np.stack([self._cache[t] for t in texts])


# ---------- the four sub-scores ----------

def recency(age_turns, half_life=HALF_LIFE_TURNS):
    """1.0 for the newest message, halving every `half_life` turns."""
    return 0.5 ** (age_turns / half_life)


_IMPORTANT = [
    r"\d",                                                       # any number
    r"\b(mon|tues|wednes|thurs|fri|satur|sun)day\b",             # weekdays
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
    r"\b(today|tomorrow|tonight|deadline|due|by then|o'?clock|am|pm)\b",
    r"\bmy (name|email|phone|address|birthday|budget|number)\b",
    r"\b(i am|i'm|im) (allergic|vegan|vegetarian|diabetic|lactose|gluten)\b",
    r"\ballerg",
    r"\b(remember|don'?t forget|keep in mind|important|note that)\b",
    r"\b(never|always|must|can'?t|cannot|don'?t|do not|only)\b",
    r"\bmy [a-z]+ (is|are|was)\b",                               # "my X is ..."
    r"\b(i prefer|i like|i hate|i need|i want)\b",
]
_IMPORTANT_RE = [re.compile(p, re.I) for p in _IMPORTANT]


def importance(text, role="user"):
    """0 to 1 from regex hits. Assistant messages count less (the user's facts matter most)."""
    hits = sum(1 for r in _IMPORTANT_RE if r.search(text))
    s = min(1.0, 0.4 * hits)
    return s if role == "user" else 0.5 * s


def referenced(sim_matrix, i):
    """How many LATER messages are about the same thing as message i."""
    later = sim_matrix[i, i + 1:]
    count = int((later > REF_THRESHOLD).sum())
    return min(1.0, count / REF_SATURATE)


# ---------- combine ----------

def score_messages(messages, embedder):
    """Returns one dict per message with the four sub-scores and the total.

    "Current" = the last user message. Every other message is compared with it.
    """
    n = len(messages)
    if n == 0:
        return []
    texts = [text_of(m) for m in messages]
    embs = embedder.embed_many(texts)                  # (n, 384), already normalised
    sim = embs @ embs.T                                # cosine similarity of every pair

    cur = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=n - 1)

    out = []
    for i, m in enumerate(messages):
        age_turns = (n - 1 - i) / 2                    # two messages (user+assistant) per turn
        parts = {
            "similarity": max(0.0, float(sim[i, cur])),
            "recency": recency(age_turns),
            "importance": importance(texts[i], m.get("role", "user")),
            "referenced": referenced(sim, i),
        }
        parts["score"] = (W_SIM * parts["similarity"] + W_REC * parts["recency"]
                          + W_IMP * parts["importance"] + W_REF * parts["referenced"])
        # A stated fact ("my deadline is Friday", "I'm allergic to...") must survive even when the
        # conversation has moved on, so it gets a score floor. Only user messages can be pinned.
        if m.get("role") == "user" and parts["importance"] >= PIN_IMPORTANCE:
            parts["score"] = max(parts["score"], PIN_FLOOR)
        out.append(parts)
    return out
