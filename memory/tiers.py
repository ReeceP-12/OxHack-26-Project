"""Greedy fill: keep the highest scoring messages that fit the token budget."""
import re

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text):
        return len(_ENC.encode(text)) + 4              # +4 ~ per-message overhead
except Exception:                                      # tiktoken missing or no network for its files
    def count_tokens(text):
        return len(text) // 4 + 4                      # same rough guess main.py uses

from memory.scorer import text_of

SUMMARY_MAX_TOKENS = 30


_FILLER = {"sure", "okay", "ok", "of course", "certainly", "absolutely", "great question",
           "got it", "noted", "alright", "right", "yes", "yeah", "no problem", "happy to help",
           "good question", "understood", "thanks", "thank you", "hello", "hi"}


def _is_filler(sentence):
    words = re.sub(r"[^\w\s']", "", sentence.lower()).split()
    return len(words) <= 2 or " ".join(words) in _FILLER


def first_sentence(text, max_tokens=SUMMARY_MAX_TOKENS):
    """Cheap summary: the first sentence that says something, capped in length. No LLM.

    Skips filler openers like "Sure." or "Great question!" so the summary keeps content.
    Falls back to the very first sentence if everything looks like filler.
    """
    sentences = [x for x in re.split(r"(?<=[.!?])\s+", text.strip()) if x]
    if not sentences:
        return ""
    s = next((x for x in sentences if not _is_filler(x)), sentences[0])
    words = s.split()
    while len(words) > 3 and count_tokens(" ".join(words)) > max_tokens:
        words = words[: int(len(words) * 0.8)]
    return " ".join(words)


def fill_budget(messages, scores, budget, keep_last=4):
    """Returns (kept_messages_in_original_order, tiers).

    Protected (always sent in full): every system message and the last `keep_last` messages.
    Everything else competes by score:
      full text if it fits, else first-sentence summary if that fits, else archived (dropped).
    """
    n = len(messages)
    texts = [text_of(m) for m in messages]
    full_tok = [count_tokens(t) for t in texts]

    protected = {i for i, m in enumerate(messages) if m.get("role") == "system"}
    protected |= set(range(max(0, n - keep_last), n))

    tier = {i: "full" for i in protected}
    used = sum(full_tok[i] for i in protected)
    sent_text = {i: texts[i] for i in protected}
    sent_tok = {i: full_tok[i] for i in protected}

    candidates = sorted((i for i in range(n) if i not in protected),
                        key=lambda i: scores[i]["score"], reverse=True)
    for i in candidates:
        if used + full_tok[i] <= budget:
            tier[i], sent_text[i], sent_tok[i] = "full", texts[i], full_tok[i]
            used += full_tok[i]
            continue
        short = first_sentence(texts[i])
        short_tok = count_tokens(short)
        if short and short_tok < full_tok[i] and used + short_tok <= budget:
            tier[i], sent_text[i], sent_tok[i] = "summary", short, short_tok
            used += short_tok
        else:
            tier[i] = "archived"
            sent_tok[i] = 0

    kept, tiers = [], []
    for i, m in enumerate(messages):
        if tier[i] != "archived":
            new = dict(m)
            if tier[i] == "summary":
                new["content"] = sent_text[i]
            kept.append(new)
        tiers.append({
            "index": i,
            "role": m.get("role", "user"),
            "tier": tier[i],
            "score": round(scores[i]["score"], 3),
            "tokens": sent_tok[i],            # tokens actually sent (0 if archived)
            "tokens_full": full_tok[i],       # tokens in the original message
        })
    return kept, tiers