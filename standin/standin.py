"""Stand-in context manager (Reece's v0 safety net). Deterministic, no LLM, no heavy deps.

Same contract as the real one:  build(messages, budget) -> (messages, tiers)
Swap in Teammate A's version when it lands. Turn this on with  set STANDIN=1  (see main.py).

How it works:
  1. System messages, the last KEEP_RECENT messages, and tool traffic are always kept in full.
  2. Every other message gets a score: similarity to the latest user message (bag of words),
     importance (numbers, days, "my X is", remember, must, allergy...), and recency.
  3. Fill the budget greedily by score: full text, else first-sentence summary, else a short gist
     if the message looks important, else archived (dropped from the prompt).
  4. Important messages never drop below gist.
Archived messages are rescored every turn, so a later question that matches one pulls it back.
"""
import math
import re

KEEP_RECENT = 6
W_SIM, W_IMP, W_REC = 0.5, 0.35, 0.15
SUMMARY_CHARS = 140
GIST_WORDS = 10

STOP = set("""a an the and or but if so to of in on at for with is are was were be been am i you he she it we they
me my your our this that these those do does did have has had not no yes ok okay just really very can could would
should will what when where who how why about from by as up out there then than too also""".split())

IMPORTANT = re.compile(
    r"(\d|\bmy\s+\w+(\s+\w+)?\s+(is|are|was)\b|\bremember\b|\bdon'?t forget\b|\bmust\b|\bnever\b|\balways\b"
    r"|\ballerg|\bdeadline\b|\bdue\b|\bbirthday\b|\bpassword\b|\bbudget\b|\bmonday\b|\btuesday\b|\bwednesday\b"
    r"|\bthursday\b|\bfriday\b|\bsaturday\b|\bsunday\b|\btomorrow\b|\bi am\b|\bi'm\b|\bmy name\b)",
    re.I,
)


def _text(m: dict) -> str:
    c = m.get("content") or ""
    if isinstance(c, list):
        c = " ".join(p.get("text", "") for p in c if isinstance(p, dict))
    return c


def _tokens(text: str) -> int:
    return len(text) // 4 + 4


def _words(text: str) -> dict:
    out: dict = {}
    for w in re.findall(r"[a-z0-9']+", text.lower()):
        if w not in STOP and len(w) > 1:
            out[w] = out.get(w, 0) + 1
    return out


def _cos(a: dict, b: dict) -> float:
    if not a or not b:
        return 0.0
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def _importance(text: str) -> float:
    hits = len(IMPORTANT.findall(text))
    return min(1.0, hits / 2)


def _summary(text: str) -> str:
    first = re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)[0]
    return first if len(first) <= SUMMARY_CHARS else first[:SUMMARY_CHARS].rsplit(" ", 1)[0] + "..."


def _gist(text: str) -> str:
    return " ".join(text.split()[:GIST_WORDS]) + ("..." if len(text.split()) > GIST_WORDS else "")


def _with_content(m: dict, content: str, tag: str) -> dict:
    return {**m, "content": f"[{tag}] {content}"}


def build(messages: list[dict], budget: int):
    n = len(messages)
    texts = [_text(m) for m in messages]
    costs = [_tokens(t) for t in texts]
    tiers = [{"index": i, "role": m.get("role", ""), "tier": "archived", "score": 0.0, "tokens": 0}
             for i, m in enumerate(messages)]
    chosen: dict[int, dict] = {}  # index -> message to send

    def keep_full(i: int) -> None:
        chosen[i] = messages[i]
        tiers[i].update(tier="full", tokens=costs[i])

    # Always-full: system, tool traffic, recent tail.
    for i, m in enumerate(messages):
        if m.get("role") in ("system", "developer", "tool") or m.get("tool_calls") or i >= n - KEEP_RECENT:
            tiers[i]["score"] = 1.0
            keep_full(i)

    used = sum(tiers[i]["tokens"] for i in chosen)
    last_user = next((texts[i] for i in range(n - 1, -1, -1) if messages[i].get("role") == "user"), "")
    q = _words(last_user)

    rest = [i for i in range(n) if i not in chosen]
    scored = []
    for i in rest:
        sim = _cos(q, _words(texts[i]))
        imp = _importance(texts[i])
        rec = (i + 1) / n
        score = W_SIM * sim + W_IMP * imp + W_REC * rec
        tiers[i]["score"] = round(score, 3)
        scored.append((score, imp, i))
    scored.sort(reverse=True)

    for score, imp, i in scored:
        if not texts[i].strip():
            continue
        if used + costs[i] <= budget:
            keep_full(i)
            used += costs[i]
            continue
        s = _summary(texts[i])
        sc = _tokens(s) + 2
        if sc < costs[i] and used + sc <= budget:
            chosen[i] = _with_content(messages[i], s, "summary")
            tiers[i].update(tier="summary", tokens=sc)
            used += sc
            continue
        g = _gist(texts[i])
        gc = _tokens(g) + 2
        if imp >= 0.5:  # important facts never go below gist, even over budget
            chosen[i] = _with_content(messages[i], g, "gist")
            tiers[i].update(tier="gist", tokens=gc)
            used += gc
        # else stays archived

    out = [chosen[i] for i in sorted(chosen)]
    return out, tiers
