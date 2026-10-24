"""Fake OpenAI-style endpoint for testing runner.py before the real proxy exists.

Run from repo root:  uvicorn bench.stub_server:app --port 8001
It echoes canned replies and remembers facts, so recall questions can pass.
"""
from fastapi import FastAPI

app = FastAPI()

KNOWN = {
    "allerg": "You're allergic to peanuts.",
    "deadline": "Your deadline is Friday the 14th.",
    "dog": "Your dog is called Biscuit.",
    "budget": "Your budget is 1200 pounds.",
    "flight": "Your flight is BA 249.",
}

@app.post("/v1/chat/completions")
def chat(body: dict):
    msgs = body["messages"]
    last = msgs[-1]["content"].lower()
    reply = "Okay, noted."
    for key, ans in KNOWN.items():
        if key in last and "?" in last:
            reply = ans
    # rough token estimate: 4 chars per token over the whole history
    tokens = sum(len(m["content"]) for m in msgs) // 4
    return {
        "choices": [{"message": {"role": "assistant", "content": reply}}],
        "usage": {"prompt_tokens": tokens},
    }

# --- mimic the real proxy's debug endpoints so runner.py can be tested ---
STATE = {"manager_on": True, "turns": []}

@app.post("/debug/mode")
def set_mode(body: dict):
    STATE["manager_on"] = bool(body["manager_on"])
    return {"manager_on": STATE["manager_on"]}

@app.post("/debug/reset")
def reset():
    STATE["turns"] = []
    return {"ok": True}

@app.get("/debug/state")
def state():
    return STATE
