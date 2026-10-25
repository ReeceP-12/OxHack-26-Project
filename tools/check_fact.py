"""Is a fact the regex cannot see pinned, and does it survive 30 turns of chatter?

Needs the proxy running (the memory manager is now the default):
    export MOCK_LLM=1
    uvicorn main:app --port 8000
then, from the repo root, in another terminal:
    python tools/check_fact.py
    python tools/check_fact.py "I react badly to shellfish."     # try your own sentence

"Pinned" means the fact's score is at the pin floor (0.7): the regex or the classifier flagged it. A short
fact can also survive unpinned when there is spare budget, so the score, not the tier, is the real signal.
For a control, restart the proxy with FACT_CLF=off FACT_STRONG_REGEX=0 and run it again: the score drops to ~0.05.
"""
import sys
import httpx

B = "http://localhost:8000"
FACT = sys.argv[1] if len(sys.argv) > 1 else "Honestly, peanuts make me really ill."
PIN_SCORE = 0.7   # memory/scorer.py PIN_FLOOR


def main():
    try:
        info = httpx.get(f"{B}/", timeout=5).json()
    except httpx.HTTPError as e:
        raise SystemExit(f"Cannot reach the proxy at {B} ({e.__class__.__name__}). Is uvicorn running? "
                         "It needs a few seconds to load MiniLM before it answers.")
    mgr = info.get("manager") or {}          # older proxies do not report this
    clf = mgr.get("fact_classifier") or {}
    print(f"manager       : {mgr.get('name', '?')}  | classifier loaded: {clf.get('loaded', '?')}"
          f"  threshold: {clf.get('threshold', '?')}")
    if mgr.get("name") == "none" or mgr.get("error"):
        raise SystemExit(f"The memory manager is NOT running ({mgr.get('error') or 'MANAGER=none'}), so every message "
                         "is sent in full and this test proves nothing. Fix that and retry.")

    httpx.post(f"{B}/debug/reset")
    httpx.get(f"{B}/debug/mode/on")

    msgs = [{"role": "system", "content": "You are a helpful voice assistant."},
            {"role": "user", "content": FACT}]
    for i in range(30):
        msgs.append({"role": "user", "content": f"Tell me about topic {i}. " + "Please include plenty of extra detail. " * 8})
        resp = httpx.post(f"{B}/v1/chat/completions", json={"messages": msgs}, timeout=60)
        if resp.status_code != 200:
            raise SystemExit(f"turn {i}: HTTP {resp.status_code}: {resp.text[:300]}")
        msgs.append({"role": "assistant", "content": resp.json()["choices"][0]["message"]["content"]})

    turns = httpx.get(f"{B}/debug/state").json()["turns"]
    tiers = turns[-1]["tiers"]
    fact = tiers[1]
    pinned = fact["score"] >= PIN_SCORE

    print(f"fact sentence : {FACT!r}")
    print(f"last turn     : {turns[-1]['messages_sent']} of {turns[-1]['messages_in']} messages sent, "
          f"{turns[-1]['tokens_sent']} tokens")
    print(f"fact message  : tier={fact['tier']}  score={fact['score']}  pinned={'yes' if pinned else 'no'}")

    if all(t["tier"] == "full" for t in tiers):
        print("\nNOTHING WAS SHRUNK, so this test proves nothing. Check GET / : manager should be 'memory'.")
    elif pinned and fact["tier"] in ("full", "summary"):
        print("\nPASS: the fact was pinned and survived 30 turns of unrelated chatter.")
    elif fact["tier"] == "archived":
        print("\nFAIL: the fact was archived. Is FACT_CLF set to 'off', and does memory/fact_clf.npz exist?")
    else:
        print(f"\nNOT PINNED: the fact survived only as '{fact['tier']}' because the budget had room. "
              "The classifier did not flag it. A tighter MEMORY_BUDGET (e.g. 700) would drop it.")


if __name__ == "__main__":
    main()
