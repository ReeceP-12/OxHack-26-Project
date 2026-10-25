# Deterministic Memory for Voice Agents

> Remember what matters. No language model in the memory loop.

Built at OxfordHACK 2026.

**Demo video:** https://youtu.be/pfISXdMw4ME

## The problem

A voice agent resends the entire conversation to the language model on every turn, because the model has no memory of its own. So the longer you talk, the more tokens each turn costs, and in voice a slow reply is heard as an awkward pause. The usual fix, keeping only the last few messages, forgets important things said early on, like an allergy.

## What it does

This is a proxy that sits between an ElevenLabs voice agent and the language model. On every turn it decides what the model gets to see:

- Every message is scored from 0 to 1 using similarity to the current question, recency, importance and whether later messages refer back to it.
- Messages are kept in full, shrunk to a one-sentence summary, cut to a short gist, or left out of that turn's prompt.
- Critical facts (health, allergies, safety) go into a pinned lane on top of the token budget, so they cannot be squeezed out.
- A live dashboard shows what was kept, shrunk and forgotten, message by message.

There is no language model anywhere in the memory loop. It uses sentence embeddings, pattern matching and one tiny classifier, so it is free, repeatable and runs locally.

In our live voice demo we tell the agent about a heart condition, talk about unrelated things for many turns, then ask what exercise to do. The agent takes the condition into account without being reminded.

## Why it is different

| | Last-N window | LLM memory (e.g. Mem0) | This project |
|---|---|---|---|
| Memory step cost | Free | Extra LLM calls | Local maths |
| Same answer every time | Yes | No | Yes |
| Keeps critical facts | No | Depends on the summary | Pinned lane |
| Shows what it dropped | No | Not per message | Live dashboard |
| Recall on our 50-turn test | 0/5 | Not tested | 5/5 |

These are typical designs, not a benchmark of any named product. Tools like Mem0 are largely about long-term memory across sessions. This project is about keeping one live call small and safe.

## How it works in short

```
ElevenLabs agent  ->  FastAPI proxy (main.py)  ->  Gemma via the Gemini API
                          |  memory/  (scorer, tiers, fact classifier, budget)
                          +-> /debug/state  ->  Streamlit dashboard
```

1. ElevenLabs sends the full message list to our OpenAI-compatible endpoint.
2. The memory manager scores every message and packs them into a token budget. Each message gets its smallest form first, so the whole conversation is represented, then the best ones are upgraded to bigger forms with the space left.
3. Pinned facts live in their own lane on top of the base budget.
4. The reduced prompt goes to the model, and the reply streams back. Nothing is stored: because the history is resent every turn, a message left out today can return when it becomes relevant again.

Details: [`docs/HOW_IT_WORKS.md`](docs/HOW_IT_WORKS.md).

## Results

| Measure | Result |
|---|---|
| Tokens vs naive | 58% fewer on average over 50 turns (naive 189,739, ours 79,082; ranges do not overlap). On the last turn: 2,227 vs 7,932 tokens |
| Recall (out of 5) | Ours 5 and naive 5 in every run, and ours also 5 on 5 reworded questions. A last-20-messages window scored 0 |
| Speed | Mean time to first LLM token 1,815 ms vs 2,310 ms for naive on the 50-turn script (small and suggestive, and it weakened on a longer 107-turn test). Manager cost about 100 ms per turn |

Method: 50-turn scripted conversation with 7 planted facts, 5 recall questions plus 5 reworded ones, 3 runs per mode with min and max reported, base budget 1500, default budget scaling, gemini-3.5-flash-lite. One script and one model. Both naive and ours score 5/5, so this shows no loss of recall, not a gain. The window baseline scores 0 because the facts are planted earlier than its 20-message window. Offline accuracy tests at 50, 100 and 200 prompts are in [`docs/RESULTS.md`](docs/RESULTS.md).

## Honest limits

- Compression can lose detail, so recall is measured, not assumed.
- The scoring weights are hand set, and the fact detector's training set is small.
- Most tuning happened near our own test conversation. A held-out conversation is the strongest next test.
- The matching is English only.
- The proxy runs on a laptop behind a tunnel for the demo.

## Try it

See the **Quickstart** in the technical reference below (section 2). It runs with canned replies and no API key, or with a Gemini key for real answers, and it starts the live dashboard. For the voice demo, point an ElevenLabs custom LLM agent at the proxy URL ending in `/v1`.

## Team

- Reece Pugh: proxy, ElevenLabs integration, project lead
- Tom (GitHub: whatsaxis): memory manager and statistics
- Alex (GitHub: Ulef-maker): dashboard and statistics
- Xavisanz (GitHub): benchmark and statistics

## Built with

Python, FastAPI, ElevenLabs, Gemma on the Gemini API, sentence-transformers (MiniLM), NumPy, tiktoken, Streamlit.

---

## Technical reference

An ElevenLabs voice agent resends the whole chat every turn, so every reply gets slower and dearer. This proxy sits in
between and **shrinks the history before it reaches the model**: stated facts (allergies, deadlines, names) are kept,
the rest is summarised or dropped. There is **no LLM in the memory loop**, so it adds almost no latency.

```
ElevenLabs  ->  main.py (FastAPI proxy)  ->  Gemini / Gemma
                  |  memory/  scorer, tiers, fact classifier, budget curves
                  +-> /debug/state  ->  dashboard/app.py (Streamlit)
```

**Headline result** (`docs/RESULTS.md`): at 50, 100 and 200 prompts the manager kept 100% of critical facts (allergy,
medication) in every test case while sending 6 to 46% of the tokens that resending everything would. A same-cost window
of recent messages kept 0 to 4% of the planted facts.

| read this | for |
|---|---|
| **this file** | setup, running, every setting, tools, troubleshooting |
| [`docs/HOW_IT_WORKS.md`](docs/HOW_IT_WORKS.md) | scoring, tiers, pinned facts, the fact classifier, the budget curves |
| [`docs/RESULTS.md`](docs/RESULTS.md) | the accuracy tests and their results, caveats |
| [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md) | Gemini/Gemma flakiness and the recipe that works, mock mode |
| [`docs/archive/`](docs/archive/) | hackathon-era planning notes (outdated, kept for history) |

---

## 1. Setup (once)

```bash
git clone https://github.com/ReeceP-12/OxHack-26-Project.git && cd OxHack-26-Project
python3 -m venv .venv && source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt                            # includes sentence-transformers (MiniLM) and tiktoken
```
The first run downloads MiniLM (about 90 MB). Run everything from the repo root with the venv active.
On Mac/Linux use `export NAME=value`; in Windows cmd use `set NAME=value`.
**Environment variables are read when uvicorn starts. Restart it after changing any of them, in the same terminal.**

## 2. Quickstart: run everything (3 terminals)

**Terminal 1: the proxy** with our recommended configuration (stop any running uvicorn first with Ctrl+C)
```bash
source run_arctan.sh        # clears leftover MEMORY_*/FACT_* settings, sets the config below, starts uvicorn on :8000
```
`run_arctan.sh` uses mock replies (`MOCK_LLM=1`, no API key). For real models open it and replace that line with
`export GEMINI_API_KEY=your_key` (and see `docs/KNOWN_ISSUES.md` for the recipe that stops Gemini hanging).
Check it: `curl localhost:8000/` should show `"manager": {"name": "memory", ... "fact_classifier": {"loaded": true, "threshold": 0.4}}`.

**Terminal 2: the dashboard**
```bash
streamlit run dashboard/app.py          # http://localhost:8501 ; untick "Use fake data"
```
**Terminal 3: feed it a conversation**
```bash
curl -X POST localhost:8000/debug/reset         # clear the turn log
python tools/demo_long.py                       # 150 realistic prompts: the budget and tiers show up
python tools/demo_long.py --turns 250           # longer
python -m bench.runner --mode on --label demo   # or the scripted 50-turn benchmark conversation
```
For the live voice demo, tunnel the proxy and give ElevenLabs `https://<domain>/v1`:
`ngrok http 8000 --url https://<static-domain>.ngrok-free.dev`.

### What `run_arctan.sh` sets
| variable | value | meaning |
|---|---|---|
| `MEMORY_SCALE` | `arctan` | the budget grows with the number of prompts: a big jump at the start, flattening towards the cap |
| `MEMORY_BUDGET` | `1500` | base token budget at prompt 0 |
| `MEMORY_BUDGET_MAX` | `6000` | the hard cap the budget approaches (never passes) |
| `MEMORY_SCALE_BEND` | `40` | smaller = bigger early jump and earlier flattening |
| `MEMORY_SCALE_ARC` | `1` | 1 = pure arctan; lower blends in a straight line that reaches the cap at `MEMORY_SCALE_END` |
| `MEMORY_SCALE_START` / `_FRAC` | `0` / `0` | growth starts at prompt 0, no full-history guard (the defaults for arctan) |

Budget this gives (tokens): prompt 1 = 1571, 10 = 2201, 50 = 4067, 100 = 4909, 200 = 5434, 400 = 5714.

## 3. All settings (environment variables)

### Which manager and how big
| variable | default | meaning |
|---|---|---|
| `MANAGER` | `memory` | `memory` (ours), `standin` (stand-in manager), `none` (pass-through; everything "full") |
| `CONTEXT_MANAGER` | `on` | start with the manager on/off; flip live with `/debug/mode/on` and `/debug/mode/off` |
| `MEMORY_BUDGET` | `1500` | base token budget sent to the model (system message included) |
| `MEMORY_PIN_CAP` | `500` | most tokens the pinned-facts lane may add on top of the base budget |
| `MEMORY_COVERAGE` | `1` | `0` = old full-text-first filling (messages jump straight to archived; not recommended) |
| `PAIR_FACTOR` | `0.9` | a reply scores at least this x its question, so Q&A pairs stay together (`0` = off) |
| `STANDIN_BUDGET`, `STANDIN_PIN_CAP` | `1500`, `500` | the same two settings for the stand-in manager |

### Budget scaling (memory/budget.py; curves explained in `docs/HOW_IT_WORKS.md`)
| variable | default | meaning |
|---|---|---|
| `MEMORY_SCALE` | `sqrt` | `off` (fixed; use for headline benchmarks), `arctan`, `root`, `smooth`, `log`, `sqrt`, `linear`, `quad` |
| `MEMORY_BUDGET_MAX` | `6000` | hard ceiling on the base budget |
| `MEMORY_SCALE_START` | `30` (`0` for `arctan`, `root`, `smooth`) | prompts before scaling begins |
| `MEMORY_SCALE_FRAC` | `0.5` (`0` = off, the default for `arctan`, `root`, `smooth`) | ceiling as a fraction of what naive would send |
| `MEMORY_SCALE_GAIN` | per mode | tokens per step (sqrt 120, linear 12, quad 0.25, log 350) |
| `MEMORY_SCALE_BEND` | `40` | `arctan`: prompts over which the early rise bends |
| `MEMORY_SCALE_ARC` | `1` | `arctan`: share of arctan vs a straight line to the cap at `END` |
| `MEMORY_SCALE_HALF` | `100` | `root`: prompts past START at which half the extra budget is reached |
| `MEMORY_SCALE_END` | `250` | `smooth`/`arctan` blend: prompt at which the cap is reached |

The chosen value per turn is in `/debug/state` under `manager_info.scale`.

### Fact detection (learned classifier + regex, OR-ed)
| variable | default | meaning |
|---|---|---|
| `FACT_CLF` | on | `off` = ignore the classifier file (regex only) |
| `FACT_THRESHOLD` | from `memory/fact_clf.npz` (0.4) | lower catches more facts, pins more chatter; no retraining needed |
| `FACT_STRONG_REGEX` | on | `0`/`off` = disable the strong regex (to test the classifier alone) |

Retrain: `python -m memory.train_fact_classifier` (options `--threshold 0.5 --l2 0.1 --pos-weight 1.5`).

### Model / upstream (proxy)
| variable | default | meaning |
|---|---|---|
| `MOCK_LLM` | off | `1` = never call a model, canned reply |
| `GEMINI_API_KEY` | | key for Gemini (default provider) |
| `UPSTREAM_BASE_URL`, `UPSTREAM_API_KEY` | | use any other OpenAI-compatible provider (e.g. OpenRouter) |
| `REAL_MODELS` | `gemma-4-26b-a4b-it` | comma-separated model chain; the same model may be listed twice to race copies |
| `REASONING_EFFORT` | `low` | Gemini models |
| `GEMMA_EFFORT` | `minimal` | Gemma models (the only value that stops visible thinking) |
| `HEDGE_DELAY` | `1.5` | seconds without a reply before also trying the next model |
| `MAX_PARALLEL` | `2` | most upstream requests in flight per turn |
| `TOTAL_TIMEOUT` | `7` | give up on a turn after this many seconds (ElevenLabs quits at ~8) |
| `DEBUG_DUMP` | off | `1` = save each request to `last_request.json` (for `replay.py`, `stream_probe.py`) |

## 4. Debug endpoints
| endpoint | what it does |
|---|---|
| `GET /` | health: mock?, manager name, classifier loaded + threshold, budget |
| `GET /debug/state` | `{manager_on, manager, turns:[...]}`; each turn has `tokens_in`, `tokens_sent`, `ttft_ms`, `tiers`, `manager_info` |
| `GET /debug/mode`, `/debug/mode/on`, `/debug/mode/off` | read / switch the manager on or off live |
| `POST /debug/mode` with `{"manager_on": true}` | same, for scripts |
| `POST /debug/reset` | clear the turn log between runs |

## 5. Benchmarks and accuracy
```bash
# scripted 50-turn conversation (5 recall questions), proxy running
python -m bench.runner --mode off --label naive
python -m bench.runner --mode on  --label ours
python -m bench.runner --mode off --label window20 --window 20     # crude "forget old messages" baseline
python -m bench.plot --naive naive --ours ours                     # writes bench/*.png and prints % saved

# does it forget important things? (offline statistics; design and results in docs/RESULTS.md)
python -m bench.accuracy                                  # real MiniLM + classifier, no API keys
python -m bench.accuracy --trials 100 --lengths 50 100 200 400 --scale arctan
python -m bench.accuracy --embedder fake --trials 10      # no model download (classifier skipped)
python -m bench.accuracy --live --live-trials 10          # also ask a real model through the running proxy
```
Use `MEMORY_SCALE=off` on the proxy for the headline 50-turn comparison so numbers stay comparable.

## 6. Tools (`tools/`)
| command | purpose |
|---|---|
| `python tools/demo_long.py [--turns 250]` | realistic long chat for the dashboard |
| `python tools/probe_fact.py "im allergic to peanuts"` | how the regex and classifier see a sentence, offline; no args = a built-in set |
| `python tools/probe_topic.py [--gap 80]` | does a much-discussed topic survive, and come back when you return to it |
| `python tools/check_fact.py "I react badly to shellfish."` | is a fact pinned and does it survive 30 turns of chatter, through the running proxy |
| `python tools/sim_scaling.py --lengths 20 50 100 200 --modes off sqrt linear quad [--real]` | budget-scaling simulation table |
| `python tools/send_convo.py` | quick ON vs OFF comparison on a 30-turn chat |
| `python tools/profile_manager.py [--fake]` | where the manager's milliseconds go |
| `python tools/soak.py [N] [window_s]` | reliability of proxy + model as ElevenLabs sees it |
| `python tools/replay.py`, `stream_probe.py`, `probe_thinking.py`, `test_gemini.py` | model latency / streaming / thinking diagnostics (need a key; some need `DEBUG_DUMP=1`) |
| `python -m memory.try_scorer` | eyeball scores with the real MiniLM |

## 7. Tests
```bash
for t in memory fact_classifier budget main_manager accuracy; do python tests/test_$t.py || break; done
```
The memory, budget and accuracy tests need no model download. `test_main_manager` prints a harmless "pip install" note
if the proxy dependencies are missing.

## 8. FAQ
- **Nothing is archived / the dashboard looks the same as naive.** Archiving happens only when even the cheapest form of
  every message does not fit. In mock mode the replies are one line, so the history stays tiny; run `tools/demo_long.py`.
  A growing budget makes archiving rarer still. For a demo of cuts use `MEMORY_SCALE=off MEMORY_BUDGET=400`.
- **Why is every pinned fact scored exactly 0.70?** That is the pin floor. The score only decides whether a fact is
  protected; health facts are ranked above the rest inside the pinned lane (`critical` flag). See `docs/HOW_IT_WORKS.md`.
- **If we talk about something a lot, does it stay?** Only briefly: recency and similarity protect it while it is the
  topic, and a small "referenced" bonus helps, but it is not pinned. Details and a probe in `docs/HOW_IT_WORKS.md`.
- **"Use fake data" looks great but real data doesn't.** The fake data is scripted (its latency is made up). Use
  `tools/demo_long.py` against the real proxy.

## 9. Troubleshooting
- **Everything shows as "full"**: `curl localhost:8000/` and read `manager.name`; `none` means pass-through or the memory
  import failed (the message says why).
- **503 / `Illegal header value b'Bearer '`**: no API key and `MOCK_LLM` not set in that terminal. `export MOCK_LLM=1`
  or `export GEMINI_API_KEY=...`, then restart uvicorn.
- **Dashboard empty**: the proxy has seen no requests since it started; run `tools/demo_long.py`.
- **Connection refused**: uvicorn not running yet (MiniLM takes a few seconds to load) or a different port.
- **A setting seems ignored**: restart uvicorn in the same terminal where you exported it, or use `source run_arctan.sh`
  which clears stale settings first. Check with `env | grep MEMORY`.
- Gemini/Gemma hanging or slow: `docs/KNOWN_ISSUES.md`.
