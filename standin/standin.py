"""Stand-in context manager. Deterministic, no LLM, no heavy deps.

Same contract as the real one:  build(messages, budget) -> (messages, tiers)
Turn this on with  MANAGER=standin  (see main.py).

Two lanes, so the context size is dynamic but still bounded:
  Working context  The base budget (STANDIN_BUDGET). System messages, tool traffic and the last
                   KEEP_RECENT messages are always sent in full, then the rest is filled greedily by
                   score (similarity to the latest user message, importance, recency):
                   full text, else first-sentence summary, else gist, else archived.
  Pinned lane      User messages that state a durable fact (numbers, days, "my X is", allergies,
                   health...) are stored in short form on top of the base budget, up to PIN_CAP tokens
                   (STANDIN_PIN_CAP). So the effective budget grows a little when a critical fact
                   arrives:  effective = base + pinned tokens,  never more than base + PIN_CAP.
                   Over the cap it degrades gracefully: critical facts (health, allergy, safety) rank
                   first and are never evicted by ordinary ones; lower ranked pins shrink to a gist
                   before anything leaves; a pin that still does not fit goes back into the working pool
                   (archived, but a later question that matches it can pull it back). A repeated or
                   updated ordinary fact replaces the older one; critical facts are never merged.

build() also records LAST_INFO (base, cap, pinned tokens, effective budget) for /debug/state.
"""
import math
import re

import os

KEEP_RECENT = 6
PIN_CAP_DEFAULT = 500      # most tokens the pinned lane may add on top of the base budget
PIN_FULL_MAX = 45          # a pinned message longer than this is shortened before it is stored
SUPERSEDE_SIM = 0.5        # an ordinary pin this similar to a later one is replaced by it
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


CRITICAL = re.compile(
    r"(\ballerg|\banaphyl|\bheart\b|\bcardiac|\bpacemaker|\bdiabet|\basthma|\bepilep|\bseizure"
    r"|\bpregnan|\bmedication|\bprescri|\bcancer|\bmigraine"
    r"|\bi (have|got|suffer from|am on|take)\b.{0,40}\b(disease|condition|disorder|syndrome|intoleran\w*))",
    re.I,
)

LAST_INFO: dict = {}


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


def build(messages: list[dict], budget: int, pin_cap: int | None = None):
    global LAST_INFO
    if pin_cap is None:
        pin_cap = int(os.environ.get("STANDIN_PIN_CAP", PIN_CAP_DEFAULT))
    n = len(messages)
    texts = [_text(m) for m in messages]
    costs = [_tokens(t) for t in texts]
    tiers = [{"index": i, "role": m.get("role", ""), "tier": "archived", "score": 0.0, "tokens": 0,
              "pinned": False, "critical": False} for i, m in enumerate(messages)]
    chosen: dict[int, dict] = {}  # index -> message to send

    def keep_full(i: int) -> None:
        chosen[i] = messages[i]
        tiers[i].update(tier="full", tokens=costs[i])

    # Always-full: system, tool traffic, recent tail.
    for i, m in enumerate(messages):
        if m.get("role") in ("system", "developer", "tool") or m.get("tool_calls") or i >= n - KEEP_RECENT:
            tiers[i]["score"] = 1.0
            keep_full(i)
    used = sum(tiers[i]["tokens"] for i in chosen)  # base budget use, protected messages included

    # ---- pinned lane -------------------------------------------------------------------------
    pin_idx = [i for i in range(n) if i not in chosen and messages[i].get("role") == "user"
               and texts[i].strip() and (CRITICAL.search(texts[i]) or _importance(texts[i]) >= 0.5)]
    crit = {i: bool(CRITICAL.search(texts[i])) for i in pin_idx}
    words = {i: _words(texts[i]) for i in pin_idx}
    overflow: set[int] = set()
    # A repeated or updated ordinary fact replaces the older one. Critical facts are never merged.
    for j in pin_idx:
        for i in pin_idx:
            if i < j and not crit[i] and i not in overflow and _cos(words[i], words[j]) >= SUPERSEDE_SIM:
                overflow.add(i)
    ranked = sorted((i for i in pin_idx if i not in overflow),
                    key=lambda i: (crit[i], _importance(texts[i]), i), reverse=True)
    pinned_used = 0
    for i in ranked:
        short = texts[i] if costs[i] <= PIN_FULL_MAX else _summary(texts[i])
        sc = costs[i] if short is texts[i] else _tokens(short) + 2
        g = _gist(texts[i])
        gc = _tokens(g) + 2
        if pinned_used + sc <= pin_cap:
            use, tag, cost = short, ("full" if short is texts[i] else "summary"), sc
        elif pinned_used + gc <= pin_cap:
            use, tag, cost = g, "gist", gc
        else:
            overflow.add(i)  # does not fit even as a gist: back to the working pool
            continue
        chosen[i] = messages[i] if tag == "full" else _with_content(messages[i], use, tag)
        tiers[i].update(tier=tag, tokens=cost, score=1.0, pinned=True, critical=crit[i])
        pinned_used += cost

    # ---- working pool ------------------------------------------------------------------------
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
        scored.append((score, i))
    scored.sort(reverse=True)
    for score, i in scored:
        if not texts[i].strip():
            continue
        if used + costs[i] <= budget:
            keep_full(i)
            used += costs[i]
            continue
        sm = _summary(texts[i])
        sc = _tokens(sm) + 2
        if sc < costs[i] and used + sc <= budget:
            chosen[i] = _with_content(messages[i], sm, "summary")
            tiers[i].update(tier="summary", tokens=sc)
            used += sc
            continue
        g = _gist(texts[i])
        gc = _tokens(g) + 2
        if used + gc <= budget and gc < costs[i]:
            chosen[i] = _with_content(messages[i], g, "gist")
            tiers[i].update(tier="gist", tokens=gc)
            used += gc
        # else stays archived

    n_crit = sum(1 for t in tiers if t["pinned"] and t["critical"])
    LAST_INFO = {"base_budget": budget, "pin_cap": pin_cap, "pinned_tokens": pinned_used,
                 "pinned_count": sum(1 for t in tiers if t["pinned"]), "critical_count": n_crit,
                 "pin_overflow": len(overflow), "effective_budget": budget + pinned_used}
    out = [chosen[i] for i in sorted(chosen)]
    return out, tiers
