"""Finds out how to switch a model's thinking off. For each variant it prints:
  first chunk, time until the first ANSWER text (not thinking), thinking size, total time.
Runs the real saved ElevenLabs request (last_request.json) and a simple prompt.

    python tools/probe_thinking.py [model]
"""
import json, os, sys, time, httpx

URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
KEY = os.environ["GEMINI_API_KEY"]
model = sys.argv[1] if len(sys.argv) > 1 else "gemma-4-26b-a4b-it"

prompts = {"simple": {"messages": [{"role": "user", "content": "Say hi in five words."}]}}
if os.path.exists("last_request.json"):
    real = json.load(open("last_request.json", encoding="utf-8"))["forwarded"]
    real = {k: v for k, v in real.items() if k != "reasoning_effort"}
    prompts["real"] = real

variants = {
    "default (no field)": {},
    "reasoning_effort=none": {"reasoning_effort": "none"},
    "reasoning_effort=minimal": {"reasoning_effort": "minimal"},
    "reasoning_effort=low": {"reasoning_effort": "low"},
    "thinking_budget=0": {"extra_body": {"google": {"thinking_config": {"thinking_budget": 0, "include_thoughts": False}}}},
}


def run(base, extra):
    body = {**base, **extra, "model": model, "stream": True}
    t0 = time.time()
    first = answer = None
    thought_chars = answer_chars = 0
    try:
        with httpx.stream("POST", URL, json=body, timeout=40, headers={"Authorization": f"Bearer {KEY}"}) as r:
            if r.status_code != 200:
                return f"HTTP {r.status_code} {r.read().decode()[:160]!r}"
            for line in r.iter_lines():
                if not line.startswith("data:") or "[DONE]" in line:
                    continue
                ms = (time.time() - t0) * 1000
                first = first or ms
                try:
                    d = (json.loads(line[5:]).get("choices") or [{}])[0].get("delta") or {}
                except ValueError:
                    continue
                text = d.get("content") or ""
                is_thought = bool((((d.get("extra_content") or {}).get("google") or {}).get("thought")))
                if is_thought or text.startswith("<thought>"):
                    thought_chars += len(text)
                else:
                    text = text.replace("</thought>", "")
                    if text.strip() and answer is None:
                        answer = ms
                    answer_chars += len(text)
    except Exception as e:
        return f"FAILED {e!r}"
    f = lambda v: "-" if v is None else f"{v:5.0f}"
    return (f"first chunk {f(first)} ms | first ANSWER {f(answer)} ms | thinking {thought_chars:5d} chars | "
            f"answer {answer_chars:4d} chars | total {(time.time()-t0)*1000:5.0f} ms")


print("model:", model)
for pname, base in prompts.items():
    print(f"\n--- prompt: {pname}")
    for vname, extra in variants.items():
        print(f"{vname:26} {run(base, extra)}", flush=True)
