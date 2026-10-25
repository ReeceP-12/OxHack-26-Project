"""Does a fact the regex cannot see survive 30 turns of chatter?

Needs the proxy running with the manager on:
    export MOCK_LLM=1 MANAGER=memory
    uvicorn main:app --port 8000
then, from the repo root, in another terminal:
    python tools/check_fact.py
    python tools/check_fact.py "I react badly to shellfish."     # try your own sentence
"""
import sys
import httpx

B = "http://localhost:8000"
FACT = sys.argv[1] if len(sys.argv) > 1 else "Honestly, peanuts make me really ill."


def main():
    try:
        httpx.get(f"{B}/", timeout=5)
    except httpx.HTTPError as e:
        raise SystemExit(f"Cannot reach the proxy at {B} ({e.__class__.__name__}). Is uvicorn running? "
                         "It needs a few seconds to load MiniLM before it answers.")
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
    sent, total = turns[-1]["messages_sent"], turns[-1]["messages_in"]
    fact = tiers[1]

    print(f"fact sentence : {FACT!r}")
    print(f"last turn     : {sent} of {total} messages sent, {turns[-1]['tokens_sent']} tokens")
    print(f"fact message  : tier={fact['tier']}  score={fact['score']}")

    if not any(t["tier"] == "archived" for t in tiers):
        print("\nNOTHING WAS DROPPED, so this test proves nothing. The manager is not running: "
              "start uvicorn with  export MANAGER=memory  (and make sure the switch is on).")
    elif fact["tier"] in ("full", "summary"):
        print("\nPASS: the fact survived 30 turns of unrelated chatter.")
    else:
        print("\nFAIL: the fact was archived. With the classifier loaded and threshold 0.4 this sentence should "
              "be pinned. Check FACT_CLF is not 'off' and that memory/fact_clf.npz exists.")


if __name__ == "__main__":
    main()