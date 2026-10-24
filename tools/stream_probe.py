"""Shows exactly what a model streams back, chunk by chunk, with timings.
Uses the last real ElevenLabs request if last_request.json exists (run the proxy with DEBUG_DUMP=1
to create it), otherwise a simple message. Key must be set: GEMINI_API_KEY.

    python tools/stream_probe.py [model] [--simple]
"""
import json, os, sys, time, httpx

URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
KEY = os.environ["GEMINI_API_KEY"]
args = [a for a in sys.argv[1:] if not a.startswith("--")]
model = args[0] if args else "gemma-4-26b-a4b-it"

body = {"messages": [{"role": "user", "content": "Say hi in five words."}], "stream": True}
if "--simple" not in sys.argv and os.path.exists("last_request.json"):
    body = json.load(open("last_request.json", encoding="utf-8"))["forwarded"]
    print("using last_request.json:", len(body["messages"]), "messages, fields:", sorted(body))
body = {**body, "model": model, "stream": True}
if model.startswith("gemma"):
    body.pop("reasoning_effort", None)

t0 = time.time()
first_content = None
n = 0
with httpx.stream("POST", URL, json=body, timeout=60, headers={"Authorization": f"Bearer {KEY}"}) as r:
    print("HTTP", r.status_code, "headers after", f"{(time.time()-t0)*1000:.0f} ms")
    if r.status_code != 200:
        print(r.read().decode()[:600]); sys.exit(1)
    for line in r.iter_lines():
        if not line.strip():
            continue
        n += 1
        ms = (time.time() - t0) * 1000
        if line.strip() == "data: [DONE]":
            print(f"{ms:7.0f} ms  [DONE]"); break
        try:
            obj = json.loads(line[5:])
            ch = (obj.get("choices") or [{}])[0]
            delta = ch.get("delta") or {}
            text = delta.get("content") or ""
            if text and first_content is None:
                first_content = ms
            extra = {k: (str(v)[:40]) for k, v in delta.items() if k not in ("content", "role") and v}
            print(f"{ms:7.0f} ms  content={text!r:40} finish={ch.get('finish_reason')} {extra if extra else ''}")
        except ValueError:
            print(f"{ms:7.0f} ms  RAW {line[:120]}")
print(f"\nchunks: {n}, first CONTENT at: {first_content and round(first_content)} ms, total: {(time.time()-t0)*1000:.0f} ms")
