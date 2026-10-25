"""Scores every message in a conversation from 0 to 1 for "how worth keeping".

No LLM calls here. The only model is MiniLM, which runs locally in a few ms.

score = W_SIM*similarity + W_REC*recency + W_IMP*importance + W_REF*referenced
"""
import os
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
    r"\b(today|tomorrow|tonight|deadline|due|by then|o'?clock)\b",
    r"\bmy (name|email|phone|address|birthday|budget|number)\b",
    r"\b(i am|i'm|im) (allergic|vegan|vegetarian|diabetic|lactose|gluten)\b",
    r"\ballerg",
    r"\b(remember|don'?t forget|keep in mind|important|note that)\b",
    r"\b(never|always|must|can'?t|cannot|don'?t|do not|only)\b",
    r"\bmy [a-z]+ (is|are|was)\b",                               # "my X is ..."
    r"\b(i prefer|i like|i hate|i need|i want)\b",
]
_IMPORTANT.append(r"\b\w+ years? old\b")                      # ages
_IMPORTANT_RE = [re.compile(p, re.I) for p in _IMPORTANT]

# ---- strong facts: ONE match is enough to pin (importance 0.9) ------------------------------
# Personal facts a voice agent must never lose: allergies, medical and dietary needs, names,
# family, pets, where the user lives or works. They require a first-person statement ("I'm...",
# "my dog is...") so that questions like "recommend a vegan restaurant" or "is a peanut allergy
# common?" do not get pinned.
_STATE = (r"(?:\bi(?:'m| am)\b|\bim\b|\bi have\b|\bi've\b|\bi get\b|\bi react\b"
          r"|\bmy [a-z]+ (?:is|has|are|was)\b|\bwe(?:'re| are)\b)")
_HEALTH = (r"(?:allerg\w*|intoleran\w*|anaphyla\w*|epipen|vegan|vegetarian|pescatarian|halal|kosher"
           r"|gluten[- ]free|dairy[- ]free|nut[- ]free|lactose|coeliac|celiac|diabet\w*|asthma\w*"
           r"|epilep\w*|pregnan\w*|medication|insulin|disabilit\w*|wheelchair|dyslex\w*|autis\w*)")
_REL = (r"(?:wife|husband|partner|girlfriend|boyfriend|spouse|fianc\w+|mum|mom|mother|dad|father|son"
        r"|daughter|brother|sister|kids?|child|children|baby|boss|manager|colleague|dog|cat|pet|puppy"
        r"|kitten|parrot|rabbit|horse|grandma|grandmother|grandpa|grandfather|uncle|aunt|cousin"
        r"|nephew|niece|flatmate|roommate)")
_STRONG = [
    _STATE + r"[^.!?]{0,40}\b" + _HEALTH + r"\b",                                 # "I'm severely allergic"
    r"\b(?:allergic|allergy|allergies|intolerant) (?:to|of)\b",                    # "allergic to X"
    r"\bi (?:can'?t|cannot|mustn'?t|must not|don'?t|do not) (?:eat|drink)\b",       # "I can't eat gluten"
    r"\b(?:my name is|my name's|call me|i go by|my surname)\b",                    # names
    r"\bmy " + _REL + r"(?:'s)? (?:is|are|was|were|has|had|called|named|name|birthday|age)\b",
    r"\b(?:i have|i've got|we have|we've got|i own)\s+(?:an?|two|three|four|five|\d+)\s+(?:\w+\s+){0,2}" + _REL + r"\b",
    r"\bi (?:live|work|study)\s+(?:in|at|for|as|near)\b",                         # where / what
    r"\bi(?:'m| am) from\b|\bi was born\b",
]
_STRONG_RE = [re.compile(p, re.I) for p in _STRONG]
if os.environ.get("FACT_STRONG_REGEX", "").strip().lower() in ("off", "0", "false", "no"):
    _STRONG_RE = []   # ablation: only the learned classifier can pin personal facts
STRONG_IMPORTANCE = 0.9


def importance(text, role="user"):
    """0 to 1. Regex hits (0.4 each, capped), or 0.9 for a strong personal fact.

    Assistant messages count half (the user's own words are what must be remembered).
    """
    text = text.replace("\u2019", "'")        # voice transcripts often use curly apostrophes
    hits = sum(1 for r in _IMPORTANT_RE if r.search(text))
    s = min(1.0, 0.4 * hits)
    if any(r.search(text) for r in _STRONG_RE):
        s = max(s, STRONG_IMPORTANCE)
    return s if role == "user" else 0.5 * s


def referenced(sim_matrix, i):
    """How many LATER messages are about the same thing as message i."""
    later = sim_matrix[i, i + 1:]
    count = int((later > REF_THRESHOLD).sum())
    return min(1.0, count / REF_SATURATE)


# ---------- combine ----------

def _importance_with_clf(text, role, fact_prob, fact_clf):
    """Regex importance, raised to STRONG_IMPORTANCE when the learned detector flags a user fact.
    OR-ed, never replaced: a missed allergy costs far more than one extra pinned message."""
    imp = importance(text, role)
    if role == "user" and fact_clf is not None and fact_prob >= fact_clf.threshold:
        imp = max(imp, STRONG_IMPORTANCE)
    return imp


def score_messages(messages, embedder, fact_clf=None):
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

    # Learned fact detector (optional): P(this user message states a personal fact worth keeping).
    fact_p = [0.0] * n
    if fact_clf is not None:
        user_idx = [i for i, m in enumerate(messages) if m.get("role") == "user"]
        for i, pr in zip(user_idx, fact_clf.probs([texts[i] for i in user_idx], embedder)):
            fact_p[i] = pr

    out = []
    for i, m in enumerate(messages):
        age_turns = (n - 1 - i) / 2                    # two messages (user+assistant) per turn
        parts = {
            "similarity": max(0.0, float(sim[i, cur])),
            "recency": recency(age_turns),
            "importance": _importance_with_clf(texts[i], m.get("role", "user"), fact_p[i], fact_clf),
            "referenced": referenced(sim, i),
        }
        parts["fact_prob"] = fact_p[i]
        parts["score"] = (W_SIM * parts["similarity"] + W_REC * parts["recency"]
                          + W_IMP * parts["importance"] + W_REF * parts["referenced"])
        # A stated fact ("my deadline is Friday", "I'm allergic to...") must survive even when the
        # conversation has moved on, so it gets a score floor. Only user messages can be pinned.
        if m.get("role") == "user" and parts["importance"] >= PIN_IMPORTANCE:
            parts["score"] = max(parts["score"], PIN_FLOOR)
        out.append(parts)
    return out
