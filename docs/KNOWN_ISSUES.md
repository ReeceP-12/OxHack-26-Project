# Known issues (Sat 24 Oct 2026, updated ~18:35)

Read this before debugging the proxy. Nothing here is a setup mistake on your side.

## 0. Working setup found (read this first)

The Gemini Flash models were overloaded or hanging for every key we tried. **`gemma-4-26b-a4b-it` on the same Gemini API key works**, with two catches:

1. **Gemma 4 thinks out loud by default.** It streams `<thought>...` as reply text and can take 5 to 14 s before the real answer. The proxy now sends `reasoning_effort: "minimal"` to Gemma models, the only value that turns this off (`none`, `low` and `thinking_budget` all return 400). Override with `GEMMA_EFFORT`.
2. **Individual requests still hang or return 500 at random.** The proxy races copies and retries 500, 502, 503 and 504 once. Recommended live settings:
```
set REAL_MODELS=gemma-4-26b-a4b-it,gemma-4-26b-a4b-it,gemma-4-26b-a4b-it
set MAX_PARALLEL=3
set HEDGE_DELAY=1.5
uvicorn main:app --port 8000
```
Measured with `python tools/soak.py` (15 sequential requests, 6 s window): 15/15 answered, median 3.4 s, slowest 4.7 s (full replies). Re-run it to check the current state. Tools: `tools/test_gemini.py` (which models answer), `tools/probe_thinking.py` (how to switch thinking off), `tools/stream_probe.py` (raw chunk timings), `tools/soak.py` (success rate), `tools/replay.py`.

The older notes below are still true for the Flash models.

## 1. Gemini Flash models slow or hanging (workaround in section 0)

**Symptom:** ElevenLabs shows `custom_llm_error: LLM Cascade Error: TimeoutError`, or a reply arrives after about 12 seconds ("8.0 s lost to failed LLM attempts").

**What we found:**
- Google returns `503 UNAVAILABLE: This model is currently experiencing high demand`, or never starts replying.
- Of the models on Reece's key, only `gemini-3.5-flash-lite` ever answered, in roughly 3 to 4 seconds. `gemini-3.8-flash`, `3.7-flash`, `3.5-flash` hung or returned 503.
- Even three parallel requests to flash-lite all hung at one point, so this is Google capacity, not our proxy code.
- ElevenLabs gives up on an attempt after about 8 seconds and retries, so the first token must arrive within that.
- Groq and Cerebras signups were blocked (Cerebras wanted a card), and OpenRouter showed "Unable to verify" from Reece's network.

**Settings for flaky periods.** The defaults are one model, up to 2 requests in flight. When an upstream is this overloaded, race several copies of the one model that works:
```
set REAL_MODELS=gemini-3.5-flash-lite,gemini-3.5-flash-lite,gemini-3.5-flash-lite
set MAX_PARALLEL=3
uvicorn main:app --port 8000
```

**What the proxy does about it:** retries a 503 once, starts a second request after 1.5 s of silence, runs up to `MAX_PARALLEL` requests in flight, takes whichever answers first, and gives up cleanly at 7 s. Tune with `HEDGE_DELAY`, `MAX_PARALLEL`, `TOTAL_TIMEOUT`.

**What you can do:**
- Use your own Gemini key (see the teammate guide). If it answers quickly, you are fine.
- No working key, or Gemini is slow for you too? Run the proxy with `MOCK_LLM=1`. See below.
- To try another provider: `UPSTREAM_BASE_URL`, `UPSTREAM_API_KEY`, `REAL_MODELS` (any OpenAI-compatible endpoint, for example OpenRouter).
- To test whether Gemini is healthy right now: `python tools/test_gemini.py` (about 30 seconds).

## 2. Mock mode (use it while real models are flaky)

```
set MOCK_LLM=1
uvicorn main:app --port 8000
```

The proxy never calls a model. It streams a canned reply such as "Mock reply. I was given 5 messages and about 699 tokens of context." The context logic still runs, so `/debug/state` shows real `tokens_in`, `tokens_sent`, `messages_sent` and the `manager_on` flag.

**You can do with mock mode:** build and test `manager_build()` (Teammate A), build the dashboard (C), run the benchmark's token and message counts (B), check the proxy end to end through ElevenLabs.

**You cannot do with mock mode:** judge answer quality, run the recall test (it needs a real model to answer), or measure real latency. Do those with a real key once one works.

## 3. Context manager switch

- `GET /debug/mode/off` = naive pass-through baseline, `GET /debug/mode/on` = our manager. `GET /debug/mode` reads it.
- Scripts: `POST /debug/mode` with `{"manager_on": false}`.
- Start state: `set CONTEXT_MANAGER=off` before uvicorn. Default is on.
- `POST /debug/reset` clears the turn log between benchmark runs.
- Each turn in `/debug/state` records `manager_on` and `model`.
- Put the real memory logic in `manager_build()`, not `build_context()`, so the switch keeps working.

## 4. Benchmark note

The proxy answers requests without `"stream": true` with a normal OpenAI JSON reply (text and `usage.prompt_tokens`), so the bench runner works unchanged. ElevenLabs streams.


When benchmarking naive against ours, pin one model and one request at a time so both runs are comparable:
`set REAL_MODELS=gemini-3.5-flash-lite` and `set MAX_PARALLEL=1`. Hedging can otherwise make different turns use different models. Also raise `TOTAL_TIMEOUT` (for example 30) for scripted runs, since the 7 s default is tuned for the live ElevenLabs demo.

## 5. Other

- The proxy has no secret check yet. Do not share the ngrok URL widely.
- Set `DEBUG_DUMP=1` to save each request to `last_request.json` (git-ignored) for `tools/replay.py`. Diagnostic scripts live in `tools/`.
