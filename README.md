# OxHack-26-Project: memory for voice agents

An ElevenLabs voice agent resends the whole chat every turn. This proxy sits in between and **shrinks the history
before it reaches the model**, keeping what matters (allergies, deadlines, names) and compressing the rest, with
**no LLM in the memory loop** (so it adds almost no latency).

```
ElevenLabs  ->  main.py (FastAPI proxy)  ->  Gemini / Gemma
                  |  memory/ (scorer, tiers, fact classifier, budget scaling)
                  +-> /debug/state  ->  dashboard/app.py (Streamlit)
```

## 1. Setup (once)

```bash
git clone <repo> && cd OxHack-26-Project
python3 -m venv .venv && source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt                            # includes sentence-transformers (MiniLM) and tiktoken
```
The first run downloads MiniLM (about 90 MB). Run everything below from the repo root with the venv active.
On Mac/Linux use `export NAME=value`; in Windows cmd use `set NAME=value`. **Restart uvicorn after changing any variable.**

## 2. Run the proxy

```bash
# no API key needed: canned replies, but all context logic and token counts are real
export MOCK_LLM=1
uvicorn main:app --port 8000

# real models
export GEMINI_API_KEY=your_key
uvicorn main:app --port 8000

# tunnel for the live ElevenLabs demo (give ElevenLabs https://<domain>/v1)
ngrok http 8000 --url https://<static-domain>.ngrok-free.dev
```
Check it: `curl localhost:8000/` should show `"manager": {"name": "memory", ... "fact_classifier": {"loaded": true, "threshold": 0.4}}`.
If the name is `none`, every message is sent in full (nothing is being saved).

## 3. Run the dashboard

```bash
streamlit run dashboard/app.py          # opens http://localhost:8501; proxy URL defaults to http://localhost:8000
```
Untick **Use fake data** to see the real proxy. Fake data is scripted (its latency is made up). Feed the real
dashboard with one of:
```bash
python tools/demo_long.py                         # 150 realistic prompts, best for showing the budget and tiers
python tools/demo_long.py --turns 250
python -m bench.runner --mode on --label demo     # the 50-turn scripted benchmark conversation
```

## 4. Settings (environment variables)

### Which manager and how big
| variable | default | meaning |
|---|---|---|
| `MANAGER` | `memory` | `memory` (ours), `standin` (stand-in manager), `none` (pass-through; everything "full") |
| `CONTEXT_MANAGER` | `on` | start with the manager on/off; flip live with `/debug/mode/on` and `/debug/mode/off` |
| `MEMORY_BUDGET` | `1500` | base token budget sent to the model (system message included) |
| `MEMORY_PIN_CAP` | `500` | most tokens the pinned-facts lane may add on top of the base budget |
| `MEMORY_COVERAGE` | `1` | `0` = old full-text-first filling (messages jump straight to archived; not recommended) |
| `PAIR_FACTOR` | `0.9` | a reply scores at least this x its question, so Q&A pairs stay together (`0` = off) |
| `STANDIN_BUDGET`, `STANDIN_PIN_CAP` | `1500`, `500` | same two settings for the stand-in manager |

### Budget scaling with conversation length (memory/budget.py)
The base budget grows as the chat gets longer, then eases (softly, no corner) into a ceiling.
`budget = base + room * tanh(raw / room)`, where `raw = GAIN * (prompts - START)^p` and `room = min(MAX, FRAC * full history size) - base`.

| variable | default | meaning |
|---|---|---|
| `MEMORY_SCALE` | `sqrt` | `off` (fixed budget, use for headline benchmarks), `sqrt`, `linear`, `quad` |
| `MEMORY_SCALE_START` | `30` | prompts before scaling begins |
| `MEMORY_SCALE_GAIN` | per mode | tokens per step (sqrt 120, linear 12, quad 0.25) |
| `MEMORY_BUDGET_MAX` | `6000` | hard ceiling on the base budget |
| `MEMORY_SCALE_FRAC` | `0.5` | ceiling as a fraction of what naive would send (stops it overtaking naive) |

Rule of thumb: `sqrt` is gentle; `quad` keeps more context in very long calls (recommended past ~100 prompts).
The chosen value per turn is in `/debug/state` under `manager_info.scale`.

### Fact detection (learned classifier + regex, OR-ed)
| variable | default | meaning |
|---|---|---|
| `FACT_CLF` | on | `off` = ignore the classifier file (regex only) |
| `FACT_THRESHOLD` | from `memory/fact_clf.npz` (0.4) | lower catches more facts, pins more chatter; no retraining needed |
| `FACT_STRONG_REGEX` | on | `0`/`off` = disable the strong regex (use with `FACT_CLF` on to test the classifier alone) |

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

## 5. Debug endpoints
| endpoint | what it does |
|---|---|
| `GET /` | health: mock?, manager name, classifier loaded + threshold, budget |
| `GET /debug/state` | `{manager_on, manager, turns:[...]}`; each turn has `tokens_in`, `tokens_sent`, `ttft_ms`, `tiers`, `manager_info` |
| `GET /debug/mode`, `/debug/mode/on`, `/debug/mode/off` | read / switch the manager on or off live |
| `POST /debug/mode` with `{"manager_on": true}` | same, for scripts |
| `POST /debug/reset` | clear the turn log between runs |

Tiers per message: `full` (sent as is), `summary` (first sentence), `gist` (first few words, user messages only), `archived` (dropped).
A message is archived only when even the cheapest form of everything cannot fit the budget; pinned facts never are.

## 6. Benchmarks and accuracy

```bash
# scripted 50-turn conversation (5 recall questions), proxy running
python -m bench.runner --mode off --label naive
python -m bench.runner --mode on  --label ours
python -m bench.runner --mode off --label window20 --window 20     # crude "forget old messages" baseline
python -m bench.plot --naive naive --ours ours                     # writes bench/*.png and prints % saved

# does it forget important things? (statistics; see bench/accuracy.py for the design)
python -m bench.accuracy                                  # offline, real MiniLM + classifier, no API keys
python -m bench.accuracy --trials 100 --lengths 50 100 200 400
python -m bench.accuracy --embedder fake --trials 10      # no model download (classifier skipped)
python -m bench.accuracy --live --live-trials 10          # also ask a real model through the running proxy
```
Use `MEMORY_SCALE=off` on the proxy for the headline 50-turn comparison so numbers stay comparable.

## 7. Tools (`tools/`)
| command | purpose |
|---|---|
| `python tools/probe_fact.py "im allergic to peanuts"` | how the regex and classifier see a sentence, offline; no args = a built-in set |
| `python tools/check_fact.py "I react badly to shellfish."` | is a fact pinned and does it survive 30 turns of chatter, through the running proxy |
| `python tools/sim_scaling.py --lengths 20 50 100 200 --modes off sqrt linear quad [--real]` | budget-scaling simulation table |
| `python tools/demo_long.py [--turns 250]` | realistic long chat for the dashboard |
| `python tools/send_convo.py` | quick ON vs OFF comparison on a 30-turn chat |
| `python tools/profile_manager.py [--fake]` | where the manager's milliseconds go |
| `python tools/soak.py [N] [window_s]` | reliability of proxy + model as ElevenLabs sees it |
| `python tools/replay.py [model]`, `python tools/stream_probe.py [model]`, `python tools/probe_thinking.py [model]` | model latency / streaming / thinking diagnostics (need `DEBUG_DUMP=1` and a key) |
| `python -m memory.try_scorer` | eyeball scores with the real MiniLM |

## 8. Tests
```bash
python tests/test_memory.py && python tests/test_fact_classifier.py && python tests/test_budget.py \
  && python tests/test_main_manager.py && python tests/test_accuracy.py
```

## 9. Troubleshooting
- **Everything shows as "full"**: `curl localhost:8000/` and read `manager.name`; `none` means pass-through or the memory import failed (the message says why).
- **Dashboard empty**: the proxy has seen no requests since it started; run `tools/demo_long.py`.
- **Dashboard looks great only with fake data**: fake data is scripted. In mock mode replies are tiny, so the history never gets big enough to need cuts; use `tools/demo_long.py`.
- **Connection refused**: uvicorn not running yet (MiniLM takes a few seconds to load) or a different port.
- **Env var seems ignored**: restart uvicorn in the same terminal where you exported it.
- Detailed notes: `memory/README.md`, `docs/KNOWN_ISSUES.md`, `docs/TEAMMATE_GUIDE.md`.
