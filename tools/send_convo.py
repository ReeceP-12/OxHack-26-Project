import httpx
B = "http://localhost:8000"
def run():
    httpx.post(f"{B}/debug/reset")
    msgs = [{"role": "system", "content": "You are a helpful voice assistant."},
            {"role": "user", "content": "My deadline is Friday at 5pm and I'm allergic to peanuts."}]
    for i in range(30):
        msgs.append({"role": "user", "content": f"Tell me about topic {i}. " + "Please include plenty of extra detail. " * 8})
        r = httpx.post(f"{B}/v1/chat/completions", json={"messages": msgs}, timeout=30).json()
        msgs.append({"role": "assistant", "content": r["choices"][0]["message"]["content"]})
    t = httpx.get(f"{B}/debug/state").json()["turns"]
    print([x["tokens_sent"] for x in t])

    last = t[-1].get("tiers", [])
    print("  deadline msg:", next((x["tier"] for x in last if x["index"] == 1), "n/a (manager off)"))
httpx.get(f"{B}/debug/mode/on");  print("manager ON "); run()
httpx.get(f"{B}/debug/mode/off"); print("manager OFF"); run()
