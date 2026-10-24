# Known issues (Sat 24 Oct 2026, ~17:15)

Read this before debugging the proxy. Nothing here is a setup mistake on your side.

## 1. Gemini is slow or hangs for Reece's key (open)

**Symptom:** ElevenLabs shows `custom_llm_error: LLM Cascade Error: TimeoutError`, or a reply arrives after about 12 seconds ("8.0 s lost to failed LLM attempts").

**What we found:**
- Google returns `503 UNAVAILABLE: This model is currently experiencing high demand`, or never starts replying.
- Of the models on Reece's key, only `gemini-3.5-flash-lite` ever answered, in roughly 3 to 4 seconds. `gemini-3.8-flash`, `3.7-flash`, `3.5-flash` hung or returned 503.
- Even three parallel requests to flash-lite all hung at one point, so this is Google capacity, not our proxy code.
- ElevenLabs gives up on an attempt after about 8 seconds and retries, so the first token must arrive within that.
- Groq and Cerebras signups were blocked (Cerebras wanted a card), and OpenRouter showed "Unable to verify" from Reece's network.

**What the proxy does about it:** retries a 503 once, starts a second request after 1.5 s of silence, runs up to 3 requests in flight, takes whichever answers first, and gives up cleanly at 7 s. Tune with `HEDGE_DELAY`, `MAX_PARALLEL`, `TOTAL_TIMEOUT`.

**What you can do:**
- Use your own Gemini key (see the teammate guide). If it answers quickly, you are fine.
- No working key, or Gemini is slow for you too? Run the proxy with `MOCK_LLM=1`. See below.
- To try another provider: `UPSTREAM_BASE_URL`, `UPSTREAM_API_KEY`, `REAL_MODELS` (any OpenAI-compatible endpoint, for example OpenRouter).
- To test whether Gemini is healthy right now: `python test_gemini.py` (about 30 seconds).

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

When benchmarking naive against ours, pin one model and one request at a time so both runs are comparable:
`set REAL_MODELS=gemini-3.5-flash-lite` and `set MAX_PARALLEL=1`. Hedging can otherwise make different turns use different models. Also raise `TOTAL_TIMEOUT` (for example 30) for scripted runs, since the 7 s default is tuned for the live ElevenLabs demo.

## 5. Other

- `requirements.txtpip` in the repo is a stray typo file. Delete it.
- The proxy has no secret check yet. Do not share the ngrok URL widely.
- Set `DEBUG_DUMP=1` to save each request to `last_request.json` (git-ignored) for `replay.py`.
