# Session context: proxy and ElevenLabs wiring (Sat 24 Oct 2026, updated ~17:20)

**Update 17:20:** the Gemini timeout is understood but not fixed: Google is overloaded or hanging for Reece's key. The proxy now retries, races models and has a `MOCK_LLM=1` mode so everyone can keep working. Context manager on/off switch added. Full details in `docs/KNOWN_ISSUES.md`. Sections below are the 14:20 snapshot where not contradicted.

Read this alongside `CONTEXT.md` (the idea, architecture, prizes, timeline). This file is only what has happened since: what is built, what works, what is broken, and how it was set up. Keep it updated.

## Where we are in the plan
- Step 0 (verify the ElevenLabs custom LLM hook): **hook works end to end (one real reply seen through ElevenLabs), but Gemini is slow or hangs most of the time. See `docs/KNOWN_ISSUES.md`.**
- Step 1 (agree interfaces): written down in `docs/TEAMMATE_GUIDE.md`. Needs a quick confirm from the team.
- Step 2 (naive pass-through proxy): **written and hardened**: 503 retry, model racing, 7 s cap, mock mode, on/off switch. A full reply has been seen through ElevenLabs, but unreliably.
- Step 3 (`ContextManager.build`): not started. Placeholder `manager_build()` in `main.py` returns the messages unchanged. `build_context()` calls it unless the switch is off.

## What exists in the repo
- `main.py` (repo root for now, move to `proxy/` when `memory/` lands): FastAPI app.
  - `POST /v1/chat/completions`: takes OpenAI-style request, calls `build_context()`, streams from Gemini's OpenAI-compatible endpoint, passes chunks straight back.
  - `GET /debug/state`: list of per-turn records: `turn`, `messages_in`, `messages_sent`, `tokens_in`, `tokens_sent`, `ttft_ms`, `total_ms`.
  - `GET /`: returns `{"ok": true}`.
  - Ignores the model name ElevenLabs sends. Real model comes from env var `REAL_MODEL` (default `gemini-3.8-flash`).
  - Sends `reasoning_effort` (env var `REASONING_EFFORT`, default `low`) to keep time to first token down.
  - Prints `-> turn N: sending X msgs to MODEL` when a request arrives and `<- first chunk after N ms` when Gemini starts replying. No `<-` line means Gemini never answered.
  - Token counts are a rough 4-chars-per-token estimate. Swap for tiktoken.
- `.gitignore` was broken (leading spaces on every line, so nothing was ignored). Rewritten: `.venv/`, `__pycache__/`, `.env`, `*.pyc`.
- A `.pyc` file was committed by mistake. Fix: `git rm --cached -r __pycache__`.
- `requirements.txt` is a pip freeze but was missing `httpx` (proxy imports it). Re-freeze after `pip install httpx`.
- Changes to `main.py` and `.gitignore` may still be uncommitted. Check `git status`.

## ElevenLabs setup (agent called "speech engine")
- Product: **ElevenAgents** (not ElevenCreative).
- Blank agent, short system prompt, no knowledge base, no RAG. Keeps the benchmark clean: the only difference between naive and ours should be our proxy.
- Voice mode: **Custom** (not Realtime).
- LLM: primary row, pick **Custom LLM**. It is at the bottom of the model list.
  - Endpoint type: **Chat Completions** (not Responses).
  - Server URL: our ngrok URL ending in `/v1`. ElevenLabs appends `/chat/completions` itself, so do not add it.
  - Model ID: anything (`gemini-2.5-flash` was typed in). The proxy ignores it.
  - Secret: a made-up string. The proxy does not check it yet. Should add a check before sharing the URL widely.
  - Limit token usage: 5000 (suggested in their docs).
  - Delete the four backup LLMs so a failure is visible and never silently skips our proxy.
- Click **Publish** (top right) after changes, then use **Preview**. Typing a message in Preview is a quicker test than voice.
- Their first-turn request was 3 messages and about 695 estimated tokens. So ElevenLabs injects its own system content (~700 tokens baseline). Remember this when reading the benchmark: the naive baseline starts above zero.

## ngrok setup
- ngrok is a command line tool, no app window. Run it from cmd.
- One-time: install from the Microsoft Store, then `ngrok config add-authtoken <token>` from the dashboard.
- The free account gives one **static domain**. It does not change on restart, so paste it into ElevenLabs once. The domain belongs to Reece's account. Ask him for it, it is not stored in this repo on purpose (the tunnel has no auth yet).
- **Run the tunnel against port 8000, not 80.** ngrok's setup page shows `ngrok http 80 ...` and copying that breaks everything with `ERR_NGROK_8012` (502 Bad Gateway). This has happened twice.
  ```
  ngrok http 8000 --url https://<static-domain>.ngrok-free.dev
  ```
  Check the Forwarding line says `-> http://localhost:8000`.
- Only one machine can be the live voice endpoint at a time. That is Reece's laptop. Everyone else tests locally.

## Problems hit so far, and the fix
| Symptom | Cause | Fix |
|---|---|---|
| Custom LLM not visible on the LLM page | It is inside the model picker, not a separate field | Click the Primary row, scroll to the bottom |
| ngrok "opens a terminal and nothing happens" | It is a CLI | Run it from cmd |
| `ModuleNotFoundError: httpx` | Not in requirements | `pip install httpx` |
| ElevenLabs: `502 ... ERR_NGROK_8012` | ngrok forwarding to port 80 | Restart ngrok with port 8000 |
| Gemini 404: `gemini-2.5-flash is no longer available to new users` | New keys cannot use it | Use `gemini-3.8-flash` (now the default in `main.py`) |
| ElevenLabs: `LLM Cascade Error: TimeoutError` | **Open, Google side, see KNOWN_ISSUES.md.** Requests reach the proxy, `ttft_ms` and `total_ms` stay `null`, no `Gemini error` line printed. Gemini accepts the request and does not reply in time | In progress, see below |

## Open issue: TimeoutError (Reece owns this)
Evidence: uvicorn shows `POST /v1/chat/completions 200` (the 200 is sent before Gemini is contacted, so it proves nothing). `/debug/state` shows turns with `ttft_ms: null`. No `Gemini error` line, so Google is not rejecting the call. One earlier turn did get an answer back through ElevenLabs, so the path can work.

Likely cause: a thinking model with slow first token. Already tried: `reasoning_effort` set to `low`. Next checks:
1. Look at the uvicorn lines `-> turn` and `<- first chunk after N ms`.
2. Call Gemini directly with curl (outside the proxy) and time it. If it is slow or errors, the model or key is the problem.
3. If `reasoning_effort` gives a 400, set `REASONING_EFFORT` differently or remove the field.
4. Try a faster model name via `set REAL_MODEL=<name>` before starting uvicorn.
5. Fallback: raise ElevenLabs soft timeout, or move to the own voice loop (memory code is identical either way).

## Environment gotchas (from CONTEXT.md, still true)
- Windows laptop, VS Code terminal defaults to cmd.
- MSYS2 Python shadows the real one. Real Python: `C:\Users\reece\AppData\Local\Python\bin\python.exe`. Create the venv with that full path, activate with `.venv\Scripts\activate`.
- The venv is currently using Python 3.14.
- Gemini API key lives only in an environment variable: `set GEMINI_API_KEY=...`. Never in the repo, never in ElevenLabs. If it was ever pasted somewhere public, rotate it in AI Studio.

## Decisions made this session
- Blank ElevenLabs agent, Custom voice mode, Chat Completions endpoint.
- Proxy is the single point everything calls. Memory logic plugs in through `build_context()` / `ContextManager.build`.
- Keep the first memory version dumb (embed, score, fill budget greedily). No gist tiers, SQLite archive or tuned weights until it runs end to end.
- Do a tiny Mem0 comparison early. If we cannot beat it on tokens, lean on latency and the visibility story.

## Next actions
1. Reece: fix the timeout, confirm one full reply comes back through ElevenLabs, commit and push.
2. Reece: add the secret check to the proxy (rejects requests without the ElevenLabs secret).
3. Teammate A: `memory/` v0 (`ContextManager.build`).
4. Teammate B: `bench/` 50-turn script and recall questions.
5. Teammate C: dashboard against a fake `/debug/state` payload.
6. Whoever has a minute: ask an organiser about the theme and the Citadel "Handmade" category. Install Delta early.
