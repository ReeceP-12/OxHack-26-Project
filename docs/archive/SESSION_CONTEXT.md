> **Archived (hackathon session notes, outdated).** Current docs: `README.md`, `docs/HOW_IT_WORKS.md`, `docs/RESULTS.md`.

# Session context (Sat 24 Oct 2026, updated ~21:35)

Read this alongside `CONTEXT.md` (the idea, architecture, prizes, timeline) and `docs/KNOWN_ISSUES.md` (the model and API problems, with the working recipe). This file is what has happened since CONTEXT.md was written: what is built, what works, what is broken, how it is set up, and what is next. Keep it updated.

## How to work with Reece
Casual and direct. Honest assessments, not reassurance. Never use em dashes in anything written for him. Do not do unsolicited repeated check-ins. He is at OxfordHACK 2026 (ends Sunday 4:30pm, clocks go back overnight).

## Where we are in the plan
- Step 0 (ElevenLabs custom LLM hook): **works end to end.** Replies come back through ElevenLabs Preview with the working setup below.
- Step 1 (agree interfaces): written in `docs/TEAMMATE_GUIDE.md`. **The tiers shape is still not confirmed with the team.**
- Step 2 (proxy): **done and hardened.** See "The proxy".
- Step 3 (real `manager_build`): **not on main.** Still the placeholder that returns the messages unchanged. As of 21:05 there was no `memory/` branch on GitHub from Teammate A.
- Benchmark (B): in progress on branch `bench`.
- Dashboard and pitch (C): owner arrived late, nothing visible on GitHub yet.

## Team and repo state
- GitHub: `ReeceP-12/OxHack-26-Project`. `main` is Reece's proxy work. Branch `bench` is Teammate B (Xavisanz), last push ~21:30: runner, 50-turn conversation (planted facts, distinct filler, a contradiction, a repeated fact), failed-turn handling, plot script, cumulative chart. B has been merging `main` into `bench`.
- Teammate A (memory): in progress locally, unconfirmed. Critical path. Plan said v0 end to end by 6pm, so this is behind.
- Teammate C (dashboard and pitch): can build now against live `/debug/state` (tokens per turn, ttft, model, manager flag). The tier map needs the tiers interface.
- Reece: proxy, ElevenLabs wiring, integration, unblocking, prize admin.
- Git notes: `LF will be replaced by CRLF` warnings are harmless. `requirements.txt` shows as modified from line-ending noise. `last_request.json` and `.venv` are git-ignored.

## The proxy (`main.py`)
FastAPI app. ElevenLabs sends the full message list each turn; the proxy calls `build_context()`, forwards to an OpenAI-compatible model and streams back.
- `POST /v1/chat/completions`: streams (SSE) when `stream` is true (ElevenLabs). Without `stream` it returns a normal OpenAI JSON reply with `usage.prompt_tokens` (the bench runner uses this).
- `GET /debug/state`: `{"manager_on": bool, "turns": [...]}`. Each turn: `turn`, `messages_in`, `messages_sent`, `tokens_in`, `tokens_sent`, `manager_on`, `model`, `ttft_ms`, `total_ms`. Token counts are a rough 4-chars-per-token estimate. The log is capped at 5000 turns.
- Context manager switch: `GET /debug/mode`, `GET /debug/mode/on`, `GET /debug/mode/off`, `POST /debug/mode` with `{"manager_on": false}`. Off means naive pass-through. `POST /debug/reset` clears the log. Startup state from `CONTEXT_MANAGER=on|off`.
- `build_context()` calls `manager_build(messages, budget)` unless the switch is off. **Teammate A's code goes in `manager_build`.** It currently returns just the message list; the agreed future shape is `(messages, tiers)`.
- Upstream racing: retries 500, 502, 503 and 504 once on the same model, starts the next copy after `HEDGE_DELAY` of silence, up to `MAX_PARALLEL` in flight, first to deliver a first chunk wins, the rest are cancelled, hard cap `TOTAL_TIMEOUT`. A shared HTTP client reuses connections.
- Mock mode: `MOCK_LLM=1` never calls a model and returns a canned reply ("Mock reply. I was given N messages and about T tokens of context."). Token counts and context logic still work, so memory, benchmark and dashboard work can proceed. Recall tests and real latency need a real model.
- Env vars (all optional): `GEMINI_API_KEY`, `UPSTREAM_BASE_URL` and `UPSTREAM_API_KEY` (any OpenAI-compatible provider), `REAL_MODELS` (comma separated chain, default `gemma-4-26b-a4b-it`; ElevenLabs' model name is ignored), `REASONING_EFFORT` (Gemini models, default `low`), `GEMMA_EFFORT` (default `minimal`), `HEDGE_DELAY` (default 1.5), `MAX_PARALLEL` (default 2), `TOTAL_TIMEOUT` (default 7), `CONTEXT_MANAGER`, `MOCK_LLM`, `DEBUG_DUMP` (saves each request to `last_request.json`).
- Tools in `tools/`: `test_gemini.py` (which models answer and how fast), `probe_thinking.py` (how to switch a model's thinking off), `stream_probe.py` (raw chunk timings of a real request), `soak.py` (success rate over N requests), `replay.py` (replays the last saved request).

## The working API setup (read this first if the model is misbehaving)
- Gemini 3.x Flash models were overloaded or hung for every key tried, including a teammate's (which also had 403 permission errors on some models). Typical symptoms: `503 UNAVAILABLE ... high demand`, or no first byte for 10 s or more.
- **`gemma-4-26b-a4b-it` on the same Gemini API key works.** Two catches:
  1. Gemma 4 thinks out loud by default: it streams `<thought>...` as reply text and takes 5 to 14 s before the answer. `reasoning_effort: "minimal"` is the only value that turns this off (`none`, `low` and `thinking_budget` return 400). The proxy sends it automatically for any model whose name starts with `gemma`.
  2. Individual requests still hang or return 500 at random, so the proxy races copies.
- Recommended live settings:
  ```
  set REAL_MODELS=gemma-4-26b-a4b-it,gemma-4-26b-a4b-it,gemma-4-26b-a4b-it
  set MAX_PARALLEL=3
  set HEDGE_DELAY=1.5
  uvicorn main:app --port 8000
  ```
- Measured with `python tools/soak.py` (15 sequential requests, 6 s window): 15/15 answered, median 3.4 s, slowest 4.7 s (full replies, not first byte). Full ElevenLabs Preview also worked afterwards.
- **For benchmark runs** use one model copy and no racing so naive and ours are comparable: `set REAL_MODELS=gemma-4-26b-a4b-it`, `set MAX_PARALLEL=1`, `set TOTAL_TIMEOUT=30`.
- Other providers were tried and blocked from Reece's network: Groq signup refused, Cerebras needs a card, OpenRouter showed "Unable to verify". Any of them works through `UPSTREAM_BASE_URL`, `UPSTREAM_API_KEY` and `REAL_MODELS` if a teammate gets a key from another network.
- A fully local model (Ollama with Gemma) was discussed and not needed.

## ElevenLabs setup (agent called "speech engine")
- Product: ElevenAgents. Blank agent, short system prompt, no knowledge base, no RAG, so the only difference between naive and ours is our proxy.
- Voice mode: Custom. LLM: primary row, pick Custom LLM (bottom of the list). Endpoint type: Chat Completions. Server URL: the ngrok URL ending in `/v1` (ElevenLabs appends `/chat/completions`). Model ID: anything, the proxy ignores it. Secret: a made-up string, **the proxy does not check it yet**. Limit token usage 5000. Backup LLMs deleted so a failure is visible.
- Click Publish after changes, then use Preview. Typing in Preview is quicker than voice.
- ElevenLabs gives up on an LLM attempt after roughly 6 to 8 s (the panel showed "8.0 s lost to failed LLM attempts") and retries, so the first byte must arrive inside that.
- Its first-turn request is 3 messages and about 699 estimated tokens (ElevenLabs injects roughly 700 tokens of system content). It sends `messages`, `model`, `stream`, `stream_options`, `temperature`, `max_tokens` and `tools`. The naive baseline starts above zero.

## ngrok
- CLI only, run from cmd. Free account has one static domain that belongs to Reece, so it is not stored in the repo. Run `ngrok http 8000 --url https://<static-domain>.ngrok-free.dev` and check the Forwarding line says `-> http://localhost:8000`. Port 80 (ngrok's default example) gives `ERR_NGROK_8012`.
- Only Reece's laptop is the live voice endpoint. Everyone else tests locally.

## Problems hit and the fix
| Symptom | Cause | Fix |
|---|---|---|
| ElevenLabs `LLM Cascade Error: TimeoutError` | Gemini Flash overloaded or hanging; Gemma thinking out loud | Use Gemma with `minimal` and racing (above) |
| Gemini 503 `high demand` | Google capacity | Same as above, or another provider |
| `ModuleNotFoundError: httpx` | Missing from requirements | Added to `requirements.txt` |
| `ERR_NGROK_8012` / 502 | ngrok on port 80 | Restart ngrok on 8000 |
| `[Errno 10048] ... only one usage of each socket address` | Old uvicorn still on port 8000 | Ctrl+C in the old terminal, or `netstat -ano \| findstr :8000` then `taskkill /PID <pid> /F` |
| Bench runner `JSONDecodeError` against the proxy | Proxy only streamed | Proxy now returns JSON when `stream` is not set |
| `<thought>` text in replies, 5 to 14 s delay | Gemma 4 thinking | `reasoning_effort: "minimal"` for Gemma |
| `python` runs MSYS2 | PATH order | Use `C:\Users\reece\AppData\Local\Python\bin\python.exe` to make the venv, activate with `.venv\Scripts\activate` |

## Decisions made
- Blank ElevenLabs agent, Custom voice mode, Chat Completions endpoint.
- The proxy is the single point everything calls. Memory plugs in through `manager_build()`.
- Keep the first memory version dumb (embed, score, fill budget greedily). No gist tiers, SQLite archive or tuned weights until it runs end to end.
- A tiny Mem0 comparison early if time; if we cannot beat it on tokens, lean on latency and the visibility story.
- Defaults stay plain (one model); the flaky-upstream settings are an opt-in recipe, not hard-coded workarounds.
- Mock mode is a development tool, not a substitute for a working model in the demo.

## Open items and next actions
1. **Teammate A:** push a rough `memory/` branch now and give an ETA for the real `manager_build()` on main. Offer to split the work (scorer vs tiers plus glue and tests).
2. **Tiers interface:** confirm in the team chat, then wire it in. Per message: `{"index": 4, "role": "user", "tier": "full|summary|gist|archived", "score": 0.73, "tokens": 41}`. `manager_build` returns `(messages, tiers)`, the proxy stores the tiers on the turn record, naive mode marks everything `full`. A and C both depend on it.
3. **Secret check on the proxy** (`PROXY_SECRET`, reject on mismatch, open when unset). First log only the incoming header names on one real ElevenLabs request to confirm how the secret arrives, so it does not lock out the agent.
4. **B:** run the naive baseline now (one model copy, no racing, `TOTAL_TIMEOUT=30`), then "ours" when A lands, then the plot. Recall needs a real model.
5. **C:** dashboard against live `/debug/state`, tier map when the interface lands, pitch outline.
6. **Reece, admin:** ask an organiser the theme and what counts as Citadel "Best Handmade Project", install Delta (judging looks at collaboration in it), consider deploying the proxy to Vultr (prize, and it removes the laptop and ngrok single point of failure), record a fallback demo take as soon as it runs cleanly.
7. **Housekeeping:** swap the 4-chars-per-token estimate for tiktoken if benchmark accuracy needs it. Rotate the Gemini key that was pasted into chat (it sat in a local notes file). Keep the ngrok URL and keys out of the repo and the team chat.

## Environment gotchas
- Windows laptop, VS Code terminal defaults to cmd. Real Python is `C:\Users\reece\AppData\Local\Python\bin\python.exe`, venv uses 3.14. `set X=...` is cmd syntax; `set X=` with nothing after deletes the variable.
- Gemini key lives only in an environment variable (`set GEMINI_API_KEY=...`), never in the repo or ElevenLabs.
- Timeline: feature freeze Sunday 1pm, event ends 4:30pm. Drop list in order: native hot path, Mem0 comparison, Tiger Data, Delta. Never drop the recall test or the dashboard.
