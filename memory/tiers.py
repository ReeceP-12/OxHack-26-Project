"""Greedy fill: keep the highest scoring messages that fit the token budget."""
import os
import re

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text):
        return len(_ENC.encode(text)) + 4              # +4 ~ per-message overhead
except Exception:                                      # tiktoken missing or no network for its files
    def count_tokens(text):
        return len(text) // 4 + 4                      # same rough guess main.py uses

from memory.scorer import text_of, PIN_FLOOR

SUMMARY_MAX_TOKENS = 30
# Coverage first: give every candidate its cheapest form, then upgrade the best ones to full text with what
# is left. Without it the budget is spent on full text and the lower scoring messages jump straight to
# archived (no summaries or gists ever appear). Set MEMORY_COVERAGE=0 for the old full-first behaviour.
COVERAGE_FIRST = os.environ.get("MEMORY_COVERAGE", "1").strip().lower() not in ("0", "off", "false", "no")
GIST_WORDS = 8             # gist = the first few words, for messages a summary cannot shrink


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


def gist(text, max_words=GIST_WORDS):
    """Last-resort shrink for a message the summary cannot shorten (typically one short sentence):
    keep the first few words and cut the rest. No LLM."""
    words = text.split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]) + "..."


def fill_budget(messages, scores, budget, keep_last=4):
    """Returns (kept_messages_in_original_order, tiers).

    Protected (always sent in full): every system message and the last `keep_last` messages.
    Everything else competes by score:
      full text, else first-sentence summary, else a gist (first few words), else archived (dropped).
    With COVERAGE_FIRST (default) everyone gets their cheapest form first and the best are then upgraded.
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

    def forms(i):
        """The ways message i can be sent, biggest first: (tier, text, tokens)."""
        out = [("full", texts[i], full_tok[i])]
        short = first_sentence(texts[i])
        short_tok = count_tokens(short)
        if short and short_tok < full_tok[i]:
            out.append(("summary", short, short_tok))
        # Gists are chopped mid-sentence, so only the user's own messages get one. Showing the model many
        # truncated ASSISTANT turns invites it to imitate them (cut-off answers); those summarise or archive.
        if messages[i].get("role") == "user":
            g = gist(texts[i])
            g_tok = count_tokens(g)
            if g and g_tok < out[-1][2]:
                out.append(("gist", g, g_tok))
        return out

    def take_biggest(i):
        for t, txt, tok in forms(i):                       # biggest form that fits
            if used_box[0] + tok <= budget:
                tier[i], sent_text[i], sent_tok[i] = t, txt, tok
                used_box[0] += tok
                return
        tier[i] = "archived"
        sent_tok[i] = 0

    used_box = [used]
    if COVERAGE_FIRST:
        # Facts and strongly relevant messages (score >= PIN_FLOOR) keep full-first treatment: shrinking a
        # stated fact to a gist would throw the fact away. Everything else gets coverage first.
        priority = [i for i in candidates if scores[i]["score"] >= PIN_FLOOR]
        rest = [i for i in candidates if scores[i]["score"] < PIN_FLOOR]
        for i in priority:
            take_biggest(i)
        used = used_box[0]
        # Pass 1: the rest, best score first, in their cheapest form while the budget lasts.
        opts = {i: forms(i) for i in rest}
        for i in rest:
            t, txt, tok = opts[i][-1]
            if used + tok <= budget:
                tier[i], sent_text[i], sent_tok[i] = t, txt, tok
                used += tok
            else:
                tier[i] = "archived"
                sent_tok[i] = 0
        # Pass 2: best score first, upgrade to the biggest form that still fits.
        for i in rest:
            if tier[i] == "archived":
                continue
            for t, txt, tok in opts[i]:                    # biggest first
                if tok <= sent_tok[i]:
                    break                                  # nothing bigger left
                if used - sent_tok[i] + tok <= budget:
                    used += tok - sent_tok[i]
                    tier[i], sent_text[i], sent_tok[i] = t, txt, tok
                    break
    else:
        for i in candidates:
            take_biggest(i)

    kept, tiers = [], []
    for i, m in enumerate(messages):
        if tier[i] != "archived":
            new = dict(m)
            if tier[i] in ("summary", "gist"):
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