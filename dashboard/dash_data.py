"""Pure helpers for the dashboard: fake data and the tier map HTML. No Streamlit in here,
so it can be tested on its own.

The tier entry shape is the one proposed in docs/TEAMMATE_GUIDE.md:
    {"index": 4, "role": "user", "tier": "full|summary|gist|archived", "score": 0.73, "tokens": 41}
Each turn in /debug/state may carry them as turn["tiers"].
"""
import html
import math
import random

TIERS = ["full", "summary", "gist", "archived"]
TIER_COLOURS = {
    "full": "#2e9e5b",      # green: sent in full
    "summary": "#e0a800",   # amber: shrunk to a summary
    "gist": "#e07b39",      # orange: a few words / stored fact
    "archived": "#8a8f98",  # grey: out of the prompt, still searchable
}
TIER_MEANING = {
    "full": "Full text",
    "summary": "Summary",
    "gist": "Gist",
    "archived": "Archived (out of prompt)",
}
PINNED_RING = "#4da3ff"    # blue outline: a fact the manager pinned
CRITICAL_RING = "#ff4d6d"  # red outline: a pinned fact marked critical
SYSTEM_TOKENS = 700  # ElevenLabs injects roughly this much system content every turn


def make_fake_turns(n_turns: int = 50, seed: int = 7) -> list[dict]:
    """A believable /debug/state turn list. Naive tokens climb, ours flatten, tiers shift over time."""
    rng = random.Random(seed)
    max_msgs = 2 * n_turns + 1
    base_tokens = [SYSTEM_TOKENS] + [rng.randint(18, 70) for _ in range(max_msgs - 1)]
    relevance = [1.0] + [rng.random() * 0.6 for _ in range(max_msgs - 1)]
    for i in (3, 5, 11):  # planted facts: a deadline, an allergy, a name
        if i < max_msgs:
            relevance[i] = 0.95
    turns = []
    for k in range(1, n_turns + 1):
        m = 2 * k + 1
        tiers, sent_tokens, sent_msgs = [], SYSTEM_TOKENS, 1
        for i in range(m):
            role = "system" if i == 0 else ("user" if i % 2 == 1 else "assistant")
            recency = math.exp(-(m - 1 - i) / 8)
            score = round(min(1.0, 0.55 * recency + 0.45 * relevance[i]), 2)
            if i == 0 or i >= m - 4 or score >= 0.6:
                tier = "full"
            elif score >= 0.4:
                tier = "summary"
            elif score >= 0.25 or relevance[i] > 0.9:
                tier = "gist"
            else:
                tier = "archived"
            tok = base_tokens[i]
            cost = {"full": tok, "summary": round(tok * 0.3), "gist": 6, "archived": 0}[tier]
            if i > 0:
                sent_tokens += cost
                sent_msgs += tier != "archived"
            pinned = role == "user" and relevance[i] > 0.9
            tiers.append({"index": i, "role": role, "tier": tier, "score": score, "tokens": tok,
                          "pinned": pinned, "critical": pinned and i == 5})  # message 5 plays the allergy
        pinned_tokens = sum(e["tokens"] for e in tiers if e["pinned"])
        tokens_in = sum(base_tokens[:m])
        ttft_ours = 650 + rng.randint(-60, 90)
        ttft_naive = 800 + 14 * k + rng.randint(-60, 90)
        turns.append({
            "turn": k, "messages_in": m, "messages_sent": sent_msgs,
            "tokens_in": tokens_in, "tokens_sent": sent_tokens,
            "manager_on": True, "ttft_ms": ttft_ours, "naive_ttft_ms": ttft_naive,
            "total_ms": ttft_ours + 1400, "model": "fake", "tiers": tiers,
            "manager_ms": round(12 + 0.15 * m + rng.random() * 6, 2),
            "manager_info": {"base_budget": 1500, "pinned_tokens": pinned_tokens,
                             "effective_budget": 1500 + min(500, pinned_tokens), "pin_cap": 500},
        })
    return turns


def tier_counts(tiers: list[dict]) -> dict:
    counts = {t: 0 for t in TIERS}
    for e in tiers:
        counts[e.get("tier", "full")] = counts.get(e.get("tier", "full"), 0) + 1
    return counts


def pin_counts(tiers: list[dict]) -> tuple[int, int]:
    """(pinned, critical). A missing flag counts as false."""
    pinned = sum(1 for e in tiers if e.get("pinned"))
    critical = sum(1 for e in tiers if e.get("critical"))
    return pinned, critical


def tier_map_html(tiers: list[dict]) -> str:
    """A wrapped row of coloured squares, one per original message. Hover for details."""
    cells = []
    for e in tiers:
        tier = e.get("tier", "full")
        colour = TIER_COLOURS.get(tier, "#444")
        pinned, critical = bool(e.get("pinned")), bool(e.get("critical"))
        flags = " | CRITICAL" if critical else (" | pinned" if pinned else "")
        tip = (f"#{e.get('index', '?')} {e.get('role', '')} | {tier} | "
               f"score {float(e.get('score', 0)):.2f} | {e.get('tokens', 0)} tokens{flags}")
        if critical:
            ring, label = f"3px solid {CRITICAL_RING}", "!"
        elif pinned:
            ring, label = f"3px solid {PINNED_RING}", "P"
        else:
            ring, label = "3px solid transparent", "S" if e.get("role") == "system" else ""
        cells.append(
            f'<div title="{html.escape(tip)}" style="box-sizing:border-box;width:26px;height:26px;'
            f'border-radius:5px;background:{colour};border:{ring};color:#fff;font:600 11px sans-serif;'
            f'display:flex;align-items:center;justify-content:center;">{label}</div>'
        )
    return '<div style="display:flex;flex-wrap:wrap;gap:4px;">' + "".join(cells) + "</div>"


def legend_html() -> str:
    items = []
    for t in TIERS:
        items.append(
            f'<span style="display:inline-flex;align-items:center;margin-right:18px;">'
            f'<span style="width:14px;height:14px;border-radius:3px;background:{TIER_COLOURS[t]};'
            f'display:inline-block;margin-right:6px;"></span>{TIER_MEANING[t]}</span>'
        )
    for ring, text in ((PINNED_RING, "Pinned fact"), (CRITICAL_RING, "Critical fact")):
        items.append(
            f'<span style="display:inline-flex;align-items:center;margin-right:18px;">'
            f'<span style="width:14px;height:14px;border-radius:3px;border:3px solid {ring};'
            f'box-sizing:border-box;display:inline-block;margin-right:6px;"></span>{text}</span>'
        )
    return "<div style='margin:4px 0 10px 0;'>" + "".join(items) + "</div>"
