"""Context-managing proxy: ElevenLabs -> this -> an OpenAI-compatible LLM (Gemini by default).

Run:
    set GEMINI_API_KEY=your_key_here        (cmd; use `export` on Mac/Linux)
    uvicorn main:app --port 8000
Tunnel it for the live voice demo:
    ngrok http 8000 --url https://<static-domain>.ngrok-free.dev
and give ElevenLabs  https://<static-domain>.ngrok-free.dev/v1

No key, or Gemini is flaky? Run with  set MOCK_LLM=1  and the proxy replies with a canned answer
instead of calling a model. Context logic and token counts still work, so memory, benchmark and
dashboard work can carry on. See docs/KNOWN_ISSUES.md.

Environment variables (all optional except a key unless MOCK_LLM=1):
    GEMINI_API_KEY      key for Gemini (default provider)
    UPSTREAM_BASE_URL   use any other OpenAI-compatible provider instead, e.g. https://openrouter.ai/api/v1
    UPSTREAM_API_KEY    key for that provider
    REAL_MODELS         comma separated model chain; ElevenLabs' model name is ignored
    REASONING_EFFORT    Gemini models, default "low"
    GEMMA_EFFORT        Gemma models, default "minimal" (the only value that stops its visible thinking)
    HEDGE_DELAY         seconds with no reply before also trying the next model (default 1.5)
    MAX_PARALLEL        most upstream requests in flight per turn (default 2)
    TOTAL_TIMEOUT       give up on a turn after this many seconds (default 7; ElevenLabs quits at ~8)
    MANAGER             which context manager runs when the switch is ON: "memory" (Teammate A's, memory/),
                        "standin" (standin/standin.py), or unset = pass-through placeholder. STANDIN=1 still means standin.
    MEMORY_BUDGET       token budget for A's manager (default 1500)
    STANDIN_BUDGET      token budget for the stand-in (default 1500)
    CONTEXT_MANAGER     on/off at startup (default on). Flip live with /debug/mode/on|off
    MOCK_LLM            1 = never call a model
    DEBUG_DUMP          1 = save each request to last_request.json for replay.py
"""
import asyncio
import json
import os
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse


def _flag(name: str, default: str = "") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "on", "yes")


# ---- config ---------------------------------------------------------------------------------
UPSTREAM_BASE = os.environ.get("UPSTREAM_BASE_URL", "").rstrip("/")
IS_GEMINI = not UPSTREAM_BASE
BASE = UPSTREAM_BASE or "https://generativelanguage.googleapis.com/v1beta/openai"
API_KEY = os.environ.get("UPSTREAM_API_KEY", "") if UPSTREAM_BASE else os.environ.get("GEMINI_API_KEY", "")

# Models are tried in order. A model may be listed more than once: when an upstream is overloaded,
# individual requests hang at random, so racing copies of one model helps (see docs/KNOWN_ISSUES.md).
DEFAULT_CHAIN = "gemma-4-26b-a4b-it"
MODELS = [m.strip() for m in os.environ.get("REAL_MODELS", DEFAULT_CHAIN).split(",") if m.strip()]
REASONING_EFFORT = os.environ.get("REASONING_EFFORT", "low")
GEMMA_EFFORT = os.environ.get("GEMMA_EFFORT", "minimal")
HEDGE_DELAY = float(os.environ.get("HEDGE_DELAY", "1.5"))
MAX_PARALLEL = max(1, int(os.environ.get("MAX_PARALLEL", "2")))
TOTAL_TIMEOUT = float(os.environ.get("TOTAL_TIMEOUT", "7"))
MOCK_LLM = _flag("MOCK_LLM")
DEBUG_DUMP = _flag("DEBUG_DUMP")
BUDGET = 8000  # token budget handed to the context manager
RETRY_STATUS = (500, 502, 503, 504)  # upstream errors worth one quick retry of the same model

# Only forward fields the upstream endpoint understands (ElevenLabs adds extras).
FORWARD_FIELDS = ("messages", "temperature", "max_tokens", "tools", "tool_choice", "top_p")

# ---- shared state ---------------------------------------------------------------------------
# The switch. True = our ContextManager shapes the context. False = naive pass-through baseline.
STATE = {"manager_on": os.environ.get("CONTEXT_MANAGER", "on").strip().lower() not in ("off", "0", "false", "no")}
TURN_LOG: list[dict] = []  # one entry per request, newest last; dashboard and benchmark read this
MAX_LOG = 5000
_turn_counter = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    if _manager_name() == "memory":
        _memory_cm()  # fail fast and warm up now, not on the first live turn
    # One shared client: connections (and TLS handshakes) are reused across turns.
    app.state.client = httpx.AsyncClient(
        timeout=httpx.Timeout(30, connect=5),
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
    )
    yield
    await app.state.client.aclose()


app = FastAPI(lifespan=lifespan)


# ---- context hooks --------------------------------------------------------------------------
try:  # accurate counts if tiktoken is installed (same counter A's manager uses), else 4 chars per token
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:
    _ENC = None


def est_tokens(messages: list[dict]) -> int:
    """Token estimate: tiktoken when available, otherwise 4 chars per token."""
    total = 0
    for m in messages:
        c = m.get("content") or ""
        if isinstance(c, list):  # content parts
            c = " ".join(p.get("text", "") for p in c if isinstance(p, dict))
        total += (len(_ENC.encode(c)) if _ENC else len(c) // 4) + 4
    return total


def _msg_tokens(m: dict) -> int:
    return est_tokens([m])


def full_tiers(messages: list[dict]) -> list[dict]:
    """Tier info when nothing is shrunk: every message is kept in full."""
    return [{"index": i, "role": m.get("role", ""), "tier": "full", "score": 1.0, "tokens": _msg_tokens(m)}
            for i, m in enumerate(messages)]


_MEMORY_CM = None


def _manager_name() -> str:
    name = os.environ.get("MANAGER", "").strip().lower()
    if not name and _flag("STANDIN"):
        name = "standin"
    return name


def _memory_cm():
    """Teammate A's ContextManager, imported lazily so the proxy still starts without memory/."""
    global _MEMORY_CM
    if _MEMORY_CM is None:
        from memory.context_manager import ContextManager
        _MEMORY_CM = ContextManager()
        _MEMORY_CM.build([{"role": "user", "content": "warmup"}], 1500)  # load models before the first real turn
    return _MEMORY_CM


def manager_build(messages: list[dict], budget: int):
    """Context manager hook. Returns (messages, tiers).

      messages: the reduced OpenAI-style list to send to the model.
      tiers: one entry per ORIGINAL message (index into the incoming list):
             {"index": 4, "role": "user", "tier": "full|summary|gist|archived", "score": 0.73, "tokens": 41}
             "tokens" is the cost of that message as it appears in the sent prompt (0 if archived).
    Returning just a list is still accepted (everything is then reported as full).

    Choose the manager with MANAGER=memory|standin (unset = pass-through)."""
    name = _manager_name()
    if name == "memory":
        return _memory_cm().build(messages, int(os.environ.get("MEMORY_BUDGET", "1500")))
    if name == "standin":
        from standin.standin import build as standin_build
        return standin_build(messages, int(os.environ.get("STANDIN_BUDGET", "1500")))
    return messages, full_tiers(messages)


def build_context(messages: list[dict], budget: int) -> tuple[list[dict], list[dict]]:
    if not STATE["manager_on"]:
        return messages, full_tiers(messages)  # naive baseline: send everything untouched
    out = manager_build(messages, budget)
    if isinstance(out, tuple):
        sent, tiers = out
        return sent, tiers
    return out, full_tiers(messages)  # old-style manager that returns a bare list


# ---- debug endpoints ------------------------------------------------------------------------
@app.get("/")
def root():
    return {"ok": True, "mock": MOCK_LLM, "manager_on": STATE["manager_on"], "models": MODELS}


@app.get("/debug/state")
def debug_state():
    return {"manager_on": STATE["manager_on"], "turns": TURN_LOG}


def _set_mode(on: bool) -> dict:
    STATE["manager_on"] = on
    print(f"== context manager {'ON' if on else 'OFF (naive)'}", flush=True)
    return {"manager_on": on}


@app.get("/debug/mode")
def get_mode():
    return {"manager_on": STATE["manager_on"]}


@app.post("/debug/mode")
async def set_mode(request: Request):
    """Body: {"manager_on": true|false}. Use this from scripts."""
    return _set_mode(bool((await request.json()).get("manager_on")))


@app.get("/debug/mode/on")
def mode_on():
    return _set_mode(True)


@app.get("/debug/mode/off")
def mode_off():
    return _set_mode(False)


@app.post("/debug/reset")
def reset_log():
    TURN_LOG.clear()
    return {"turns": 0}


# ---- upstream racing ------------------------------------------------------------------------
async def _close(r) -> None:
    try:
        await r.aclose()
    except Exception:
        pass


async def _attempt(client: httpx.AsyncClient, model: str, payload: dict, headers: dict) -> dict:
    """One upstream try. Resolves once the FIRST chunk has arrived: {"ok": True, r, it, first},
    or {"ok": False, status, err} on failure. Only raises on cancellation (and cleans up first)."""
    r = None
    try:
        body = {**payload, "model": model}
        if model.startswith("gemma"):
            # Gemma 4 thinks out loud by default (streams <thought>... as reply text, 5 to 14 s of delay).
            # "minimal" is the only value it accepts that turns that off (none/low/budget return 400).
            body["reasoning_effort"] = GEMMA_EFFORT
        req = client.build_request("POST", f"{BASE}/chat/completions", json=body, headers=headers)
        r = await client.send(req, stream=True)
        if r.status_code != 200:
            err = (await r.aread()).decode(errors="replace")
            await _close(r)
            return {"ok": False, "model": model, "status": r.status_code, "err": err[:200]}
        it = r.aiter_bytes()
        first = await it.__anext__()
        return {"ok": True, "model": model, "r": r, "it": it, "first": first}
    except asyncio.CancelledError:
        if r is not None:
            await _close(r)
        raise
    except StopAsyncIteration:
        await _close(r)
        return {"ok": False, "model": model, "status": None, "err": "empty stream"}
    except Exception as e:
        if r is not None:
            await _close(r)
        return {"ok": False, "model": model, "status": None, "err": repr(e)}


async def race(client: httpx.AsyncClient, payload: dict, headers: dict, t0: float) -> dict | None:
    """Start the first model; if it is slow, also start the next; retry a 503 once; first model
    to deliver a first chunk wins and the rest are cancelled. Returns the winner or None."""
    queue = list(MODELS)
    tasks: dict[asyncio.Task, str] = {}
    retried: set[str] = set()
    winner = None

    def launch() -> None:
        m = queue.pop(0)
        print(f"   trying {m}", flush=True)
        tasks[asyncio.create_task(_attempt(client, m, payload, headers))] = m

    try:
        launch()
        while tasks and winner is None:
            remaining = t0 + TOTAL_TIMEOUT - time.perf_counter()
            if remaining <= 0:
                print("   turn deadline hit", flush=True)
                break
            done, _ = await asyncio.wait(list(tasks), timeout=min(HEDGE_DELAY, remaining),
                                         return_when=asyncio.FIRST_COMPLETED)
            if not done:  # everything in flight is slow: hedge with the next model
                if queue and len(tasks) < MAX_PARALLEL:
                    print(f"   no reply after {HEDGE_DELAY}s, hedging", flush=True)
                    launch()
                continue
            for t in done:
                m = tasks.pop(t)
                res = t.result()
                if res["ok"]:
                    if winner is None:
                        winner = res
                    else:
                        await _close(res["r"])
                    continue
                print(f"   {m} failed: {res['status']} {res['err']}", flush=True)
                if res["status"] in RETRY_STATUS and m not in retried:
                    retried.add(m)  # overload/internal errors often clear fast: retry same model once
                    queue.insert(0, m)
            while winner is None and queue and len(tasks) < MAX_PARALLEL:
                launch()
    finally:
        for t in tasks:
            t.cancel()
        for res in await asyncio.gather(*tasks, return_exceptions=True):
            if isinstance(res, dict) and res.get("ok"):
                await _close(res["r"])
    return winner


def _sse(obj: dict) -> bytes:
    return f"data: {json.dumps(obj)}\n\n".encode()


async def _mock_stream(entry: dict, n_msgs: int, t0: float):
    text = f"Mock reply. I was given {n_msgs} messages and about {entry['tokens_sent']} tokens of context."
    entry["ttft_ms"], entry["model"] = 5, "mock"
    print(f"<- mock reply ({n_msgs} msgs sent)", flush=True)
    cid = f"chatcmpl-mock-{entry['turn']}"
    base = {"id": cid, "object": "chat.completion.chunk", "model": "mock"}
    for w in text.split(" "):
        yield _sse({**base, "choices": [{"index": 0, "delta": {"content": w + " "}, "finish_reason": None}]})
        await asyncio.sleep(0.01)
    yield _sse({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
    yield b"data: [DONE]\n\n"
    entry["total_ms"] = round((time.perf_counter() - t0) * 1000)


async def _real_stream(client, payload: dict, headers: dict, entry: dict, t0: float):
    winner = await race(client, payload, headers, t0)
    if winner is None:
        print("all models failed", flush=True)
        yield _sse({"error": {"message": "all upstream models unavailable", "code": 503}})
        return
    entry["ttft_ms"] = round((time.perf_counter() - t0) * 1000)
    entry["model"] = winner["model"]
    print(f"<- {winner['model']} first chunk after {entry['ttft_ms']} ms", flush=True)
    try:
        yield winner["first"]
        async for chunk in winner["it"]:
            yield chunk
    finally:
        await _close(winner["r"])
    entry["total_ms"] = round((time.perf_counter() - t0) * 1000)
    print({k: v for k, v in entry.items() if k != "tiers"}, flush=True)


async def _collect_json(gen, entry: dict, model_hint: str):
    """Drain an SSE stream into one OpenAI-style chat.completion JSON reply (for stream=false callers).
    Assembles text only; tool calls are not assembled here (the live ElevenLabs path streams)."""
    raw = b"".join([c async for c in gen]).decode("utf-8", errors="replace")
    text, finish, error = [], "stop", None
    for line in raw.splitlines():
        if not line.startswith("data:") or line.strip() == "data: [DONE]":
            continue
        try:
            obj = json.loads(line[5:].strip())
        except ValueError:
            continue
        if "error" in obj:
            error = obj["error"]
            continue
        for ch in obj.get("choices") or []:
            text.append((ch.get("delta") or {}).get("content") or "")
            finish = ch.get("finish_reason") or finish
    if error is not None:
        return JSONResponse({"error": error}, status_code=int(error.get("code") or 503))
    content = "".join(text)
    completion = len(content) // 4 + 1
    return JSONResponse({
        "id": f"chatcmpl-{entry['turn']}",
        "object": "chat.completion",
        "model": entry.get("model") or model_hint,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish}],
        "usage": {"prompt_tokens": entry["tokens_sent"], "completion_tokens": completion,
                  "total_tokens": entry["tokens_sent"] + completion},
    })


# ---- the one endpoint ElevenLabs calls ------------------------------------------------------
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    global _turn_counter
    body = await request.json()
    original = body.get("messages", [])
    t_mgr = time.perf_counter()
    messages, tiers = build_context(original, budget=BUDGET)
    manager_ms = round((time.perf_counter() - t_mgr) * 1000, 2)  # what the memory manager itself costs

    payload = {k: body[k] for k in FORWARD_FIELDS if k in body}
    payload["messages"] = messages
    payload["stream"] = True
    if IS_GEMINI and REASONING_EFFORT:
        payload["reasoning_effort"] = REASONING_EFFORT

    if DEBUG_DUMP:
        try:
            with open("last_request.json", "w", encoding="utf-8") as f:
                json.dump({"incoming_keys": sorted(body), "incoming": body, "forwarded": payload}, f, indent=2)
        except OSError as e:
            print("   could not save last_request.json", e, flush=True)

    _turn_counter += 1
    entry = {
        "turn": _turn_counter,
        "messages_in": len(original),
        "messages_sent": len(messages),
        "tokens_in": est_tokens(original),
        "tokens_sent": est_tokens(messages),
        "manager_on": STATE["manager_on"],
        "tiers": tiers,
        "manager_ms": manager_ms,
        "ttft_ms": None,
        "total_ms": None,
    }
    TURN_LOG.append(entry)
    if len(TURN_LOG) > MAX_LOG:
        del TURN_LOG[: len(TURN_LOG) - MAX_LOG]

    t0 = time.perf_counter()
    print(f"-> turn {entry['turn']}: {len(original)} msgs in, {len(messages)} sent"
          f"{' (mock)' if MOCK_LLM else ''}", flush=True)

    if MOCK_LLM:
        gen = _mock_stream(entry, len(messages), t0)
    else:
        headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
        gen = _real_stream(request.app.state.client, payload, headers, entry, t0)
    if body.get("stream"):  # ElevenLabs streams
        return StreamingResponse(gen, media_type="text/event-stream")
    return await _collect_json(gen, entry, "mock" if MOCK_LLM else MODELS[0])  # standard OpenAI non-stream reply
