# Teammate guide

Read `CONTEXT.md` first (the idea and plan), then this. Goal of this file: you are running and productive within about 20 minutes, and you know exactly what to build.

**One sentence:** memory for voice agents with no LLM in the memory loop, so it adds almost no latency, plus a live dashboard showing what it kept, shrank and forgot.

## 1. Get set up (about 15 min)

You do **not** need ElevenLabs or ngrok. Only Reece's laptop runs the live voice demo. Everyone else tests the proxy locally.

1. Clone the repo and make your own branch. Never commit straight to `main`.
   ```
   git clone https://github.com/ReeceP-12/OxHack-26-Project.git
   cd OxHack-26-Project
   git checkout -b <your-name>/<task>
   ```
2. Make a virtual environment with a real Python 3.11 or newer.
   - Windows (cmd): `python -m venv .venv` then `.venv\Scripts\activate`
   - Mac or Linux: `python3 -m venv .venv` then `source .venv/bin/activate`
   - Windows gotcha: if `where python` lists an MSYS2 path first, the venv will have no `Scripts` folder. Use the full path to the real Python instead.
3. Install: `pip install -r requirements.txt` and `pip install httpx` (if `httpx` is not in the file yet).
4. Get your own Gemini API key from Google AI Studio (aistudio.google.com, or mlh.link/gemini). Own key means no shared rate limits.
5. Set it as an environment variable. **Never put it in a file in the repo.** `.env` is gitignored but an environment variable is safer.
   - Windows (cmd): `set GEMINI_API_KEY=your_key`
   - Mac or Linux: `export GEMINI_API_KEY=your_key`
6. Run the proxy: `uvicorn main:app --port 8000`
   - No working key, or Gemini is slow? Run `set MOCK_LLM=1` first (Mac or Linux: `export MOCK_LLM=1`). The proxy then returns a canned reply and never calls a model. Token counts and context logic still work. See `docs/KNOWN_ISSUES.md`.
7. Check it:
   - Open `http://localhost:8000/debug/state`. You should see `{"manager_on":true,"turns":[]}`.
   - Send a test chat (Windows cmd):
     ```
     curl -N http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"messages\":[{\"role\":\"user\",\"content\":\"say hi\"}]}"
     ```
     You should see streamed `data: {...}` chunks. Then `/debug/state` shows one turn.
   - If you get a `Gemini error`, paste it into the team chat. Reece is currently debugging a slow first reply from Gemini, so a slow response is a known issue, not your setup.

## 2. How it fits together

```
ElevenLabs agent (speech in/out)
        |  OpenAI-style chat request
        v
FastAPI proxy (main.py) -> ContextManager.build -> Gemini Flash
        |
        v
   /debug/state  ->  Streamlit dashboard
```

Every turn, ElevenLabs sends the **whole message list** to our proxy. The proxy calls `build_context(messages, budget)`, which returns a shorter list, and sends that to Gemini. Today `manager_build` in `main.py` returns the list unchanged. Teammate A replaces it. `build_context` wraps it and skips it when the on/off switch is off (the naive baseline), so put your code in `manager_build`. The point of the project is that our shorter list is faster and cheaper but still remembers the important things.

## 3. The three interfaces (do not change without telling everyone)

1. **`POST /v1/chat/completions`**: OpenAI chat completions format in, server-sent events stream out. Already written.
2. **`GET /debug/state`**: returns `{"turns": [...]}`. Each turn today has: `turn`, `messages_in`, `messages_sent`, `tokens_in`, `tokens_sent`, `manager_on`, `model`, `ttft_ms` (time to first token), `total_ms`. `GET /debug/mode/on|off` flips the context manager live for naive-versus-ours runs, and `POST /debug/reset` clears the log.
   - Proposed addition (agree it together): a `tiers` list per turn, one entry per original message:
     `{"index": 4, "role": "user", "tier": "full|summary|gist|archived", "score": 0.73, "tokens": 41}`
     The dashboard reads this. A fake payload in this shape lets the dashboard be built without waiting for real data.
3. **`ContextManager.build(messages, budget) -> (messages, tiers)`**: the one function everything calls. `messages` is the OpenAI-style list, `budget` is a token count. Returns the reduced list and the tier info for `/debug/state`. (The placeholder currently returns just the list. Switch to returning both when A lands.)

## 4. Who builds what

### Teammate A: `memory/` (the core, everything depends on it)
Build v0 **dumb**, get it running end to end, then improve. Order:
1. `memory/scorer.py`: for each turn compute a score from similarity to the current turn (sentence-transformers `all-MiniLM-L6-v2`, local), recency decay, an importance flag (regex for numbers, dates, names, "my X is...", "remember", constraints), and later "was referenced again".
2. `memory/tiers.py`: given scores and a token budget (use tiktoken), fill greedily. Full text for top turns, a first-sentence summary for the next band, drop the rest. Never drop the system message or the last few turns.
3. `memory/context_manager.py`: `ContextManager.build(messages, budget)` that calls the two above.
4. Plug into `manager_build()` in `main.py` on your branch and check that tokens per turn goes flat while a naive run climbs.
Skip until v0 works: gist tier, SQLite archive, tuned weights, pinned facts. Rules: no LLM calls inside the memory loop. Embedding every turn must stay in the low milliseconds. Write a few unit tests with a tiny hand-made conversation.

### Teammate B: `bench/` (proves it works, no dependencies on anyone)
1. A scripted 50-turn conversation (a JSON or Python list of user messages).
2. Plant facts early (a deadline, an allergy, a name) and ask recall questions late ("what deadline did I give you?"). Include a fact that is irrelevant for 30 turns and then matters, because that is what proves we reduce before we delete.
3. A runner that replays the script against the proxy twice (naive baseline vs ours), records tokens per turn, latency per turn, and whether each recall answer was right, and writes results to a CSV or JSON.
4. Score recall out of N. Be honest: report quality tradeoffs, do not hide them.
Remember the baseline already includes about 700 tokens of ElevenLabs system content on every turn in the live demo. Local runs through `curl` or a script will not have that, so say which one you measured.

### Teammate C: `dashboard/` and the pitch
1. Streamlit app that reads `/debug/state` (URL configurable, default `http://localhost:8000`).
2. Build it against a **fake JSON payload** first so it is ready before real data exists.
3. Show: tokens per turn (naive vs ours), time to first token per turn, and the tier map (a row of turns coloured by tier: full, summary, gist, archived, with the score on hover). Auto-refresh every second or two. The tier map changing live is the demo.
4. Start the pitch outline: the problem (history is resent every turn, tokens grow roughly quadratically, latency matters more in voice), our angle (no LLM in the memory loop, visible forgetting), honest numbers.

### Reece: proxy, ElevenLabs wiring, integration, unblocking everyone
Gemini slowness is a known open issue (see `docs/KNOWN_ISSUES.md`). Next: a secret check on the proxy.

## 5. Team rules
- Your own branch, small commits, push often. Open a pull request into `main` when your piece runs.
- Do not edit `main.py` on `main`. If you need a change there, tell Reece or do it on your branch and PR it.
- No keys, tokens or tunnel URLs in the repo or in commit messages.
- Do not leave the ElevenLabs agent running idle. Free tier is limited.
- Agree any change to the three interfaces in the team chat first.
- Delta: judging looks at how we collaborate in it. Install from delta.dev/download and run real work through shared threads.

## 6. Timeline (event ends Sunday 4:30pm, clocks go back overnight)
- Now to 2pm: everyone cloned, running, interfaces agreed.
- 2pm to 6pm: v0 end to end in plain Python.
- 6pm to 11pm: tune scorer, benchmark running, dashboard v1.
- Overnight: sleep in shifts, leave benchmarks running. A rested demo beats a 4am bug.
- Sunday 8am to 12pm: real numbers, naive vs ours, polish.
- Sunday 1pm: feature freeze. Record a fallback demo video, rehearse the pitch twice.
- Drop list if time gets tight, in order: native hot path, Mem0 comparison, Tiger Data, Delta. **Never drop the recall test or the dashboard.**

## 7. Where to go when stuck
- Proxy or ElevenLabs problems: Reece.
- Read `docs/SESSION_CONTEXT.md` for what is already broken and why before debugging something that is already known.
- Quick questions about the plan: `CONTEXT.md`.
