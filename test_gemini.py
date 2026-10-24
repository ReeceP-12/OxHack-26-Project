import os, time, httpx

URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
KEY = os.environ["GEMINI_API_KEY"]
MODELS = ["gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-3.7-flash",
          "gemini-3.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-flash",
          "gemini-flash-lite-latest", "gemma-4-26b-a4b-it"]


def run(model, extra, label):
    body = {"model": model, "stream": True,
            "messages": [{"role": "user", "content": "Say hi in five words."}], **extra}
    t0 = time.time()
    try:
        with httpx.stream("POST", URL, json=body, timeout=10,
                          headers={"Authorization": f"Bearer {KEY}"}) as r:
            if r.status_code != 200:
                print(f"{model:28} {label:10} HTTP {r.status_code} {r.read().decode()[:120]!r}")
                return
            for line in r.iter_lines():
                if line.startswith("data:") and "[DONE]" not in line:
                    print(f"{model:28} {label:10} OK first chunk {(time.time()-t0)*1000:.0f} ms")
                    return
            print(f"{model:28} {label:10} stream ended with no chunks")
    except Exception as e:
        print(f"{model:28} {label:10} FAILED after {time.time()-t0:.1f}s {e!r}")


for m in MODELS:
    run(m, {}, "no-effort")
print()
for m in MODELS[:4]:
    run(m, {"reasoning_effort": "low"}, "low")
