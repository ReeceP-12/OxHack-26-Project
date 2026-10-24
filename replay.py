"""Replays the last request the proxy saw straight at Gemini, no proxy, and times it.
Needs the proxy to have run with  set DEBUG_DUMP=1  so last_request.json exists.
Usage (key set in this window): python replay.py [model]"""
import os, sys, json, time, httpx

URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
KEY = os.environ["GEMINI_API_KEY"]
model = sys.argv[1] if len(sys.argv) > 1 else "gemini-3.5-flash-lite"
saved = json.load(open("last_request.json", encoding="utf-8"))
print("incoming fields:", saved["incoming_keys"])


def attempt(label, body):
    body = {**body, "model": model}
    t0 = time.time()
    try:
        with httpx.stream("POST", URL, json=body, timeout=30,
                          headers={"Authorization": f"Bearer {KEY}"}) as r:
            if r.status_code != 200:
                print(f"{label}: HTTP {r.status_code} {r.read().decode()[:200]!r}"); return
            for line in r.iter_lines():
                if line.startswith("data:") and "[DONE]" not in line:
                    print(f"{label}: first chunk {(time.time()-t0)*1000:.0f} ms"); return
            print(f"{label}: no chunks")
    except Exception as e:
        print(f"{label}: FAILED after {time.time()-t0:.1f}s {e!r}")


full = saved["forwarded"]
attempt("exact forwarded request", full)
no_tools = {k: v for k, v in full.items() if k not in ("tools", "tool_choice")}
attempt("without tools          ", no_tools)
no_effort = {k: v for k, v in no_tools.items() if k != "reasoning_effort"}
attempt("no tools, no effort    ", no_effort)
