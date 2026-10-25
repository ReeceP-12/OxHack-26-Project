# How the memory manager works

The voice agent resends the whole chat every turn. `memory/` returns a shorter list that still remembers what
matters. **No LLM calls**: only a small local embedding model (MiniLM), a few regexes and one tiny classifier.
Everything here is recomputed every turn from the full history, so nothing is ever deleted for good.

- [Use it from Python](#use-it-from-python)
- [Tiers](#tiers)
- [How a message is scored](#how-a-message-is-scored)
- [What does it remember, and what does it forget?](#what-does-it-remember-and-what-does-it-forget)
- [Learned fact detector](#learned-fact-detector)
- [The budget, and how it grows with the chat](#the-budget-and-how-it-grows-with-the-chat)
- [Tuning, checks and known limits](#tuning-checks-and-known-limits)

## Use it from Python

```python
from memory.context_manager import ContextManager

cm = ContextManager()                      # create once; loads MiniLM (slow the first time)
kept, tiers = cm.build(messages, budget=1500)
```
- `messages`: OpenAI-style list of `{"role": ..., "content": ...}`.
- `budget`: base token budget for everything sent (system message included), counted with tiktoken.
  The scaling modes below raise it as the chat grows; pinned facts add up to `MEMORY_PIN_CAP` on top.
- `kept`: the reduced list, in original order. Send this to the model.
- `tiers`: one entry per ORIGINAL message, for `/debug/state` and the dashboard:
```json
{"index": 4, "role": "user", "tier": "full", "score": 0.73, "tokens": 41, "tokens_full": 41, "pinned": false, "critical": false}
```
`tokens` is what was actually sent (0 if archived); `tokens_full` is the original size; `pinned` marks a stated fact in
the pinned lane; `critical` marks a health or safety fact (allergy, medication...), which is ranked above ordinary facts.

## Tiers

| tier | meaning |
|---|---|
| `full` | sent unchanged |
| `summary` | cut to its first meaningful sentence |
| `gist` | first 8 words, for a short message a summary cannot shrink (user messages only) |
| `archived` | not sent this turn (can come back next turn if it scores higher) |

Filling works coverage-first: every candidate first gets its cheapest form, then the best-scoring are upgraded to
full text with what is left. A message is archived only when even the cheapest forms of everything do not fit, so with
a generous or growing budget **nothing is archived**. That is by design, not a fault. System messages and the last 4
messages are always sent in full.

## How a message is scored

Score is 0 to about 1, recomputed every turn:
```
score = 0.4*similarity + 0.2*recency + 0.3*importance + 0.1*referenced
```
- **similarity**: meaning-closeness to the latest user message (MiniLM cosine). The main lever.
- **recency**: halves every 10 turns.
- **importance**: regex hits (numbers, dates, "my X is", "remember", "never/always"...), or 0.9 for a strong personal fact
  (allergy, medication, name, family, pet, where you live), OR-ed with the learned classifier.
- **referenced**: share of later messages about the same thing, saturating at 3 (so at most +0.1).
- **pin floor**: a user message with importance >= 0.8 never scores below 0.7, so stated facts survive when the chat
  moves on. This is why every pinned fact shows a score of exactly 0.70: the score only decides *whether* a fact is
  protected, not how much it matters.
- **pairing** (`PAIR_FACTOR`, 0.9): a reply scores at least 0.9 x the question before it, so question and answer stay
  together instead of leaving a comb of questions with no answers.

**Pinned lane.** Pinned facts are sent on top of the base budget, up to `MEMORY_PIN_CAP` (500 tokens). If they outgrow
it, they are ranked by (critical, importance, recency): critical facts first and never evicted by ordinary ones,
never merged; ordinary facts that don't fit are shrunk to a summary, then a gist, then returned to the normal pool.
A later, very similar ordinary fact (an update, like a changed flight number) replaces the older one.

## What does it remember, and what does it forget?

- **Stated facts** (allergies, deadlines, names, numbers): pinned, kept through the whole call.
- **Recent turns**: always kept (last 4 messages), and recency favours the last few turns.
- **A topic you talked about a lot but never called important** (say five questions about black holes): there is no
  "frequency" protection, only the small *referenced* bonus. While you are still on the topic those questions score
  high (similar to the current question and recent). Once you move on, they shrink to summaries, then gists, then
  archived as the budget fills. If you later return to the topic the current question is similar to them again, so
  they can come back, but with a small budget they may not win a place. Check with
  `python tools/probe_topic.py` (real MiniLM) and tell us if returning to a topic does not bring it back.
- **Updated facts**: the newest version wins; the old one is dropped when they are similar enough.

## Learned fact detector

Regex misses paraphrases ("peanuts make me ill", "I react badly to shellfish"). A small logistic regression on the MiniLM
embeddings we already compute flags "the user states a personal fact worth keeping" per sentence. It is OR-ed with the
regex (never replaces it), only applies to user messages, and costs one small matrix product per turn.

```
python -m memory.train_fact_classifier                      # needs the real MiniLM; writes memory/fact_clf.npz (commit it)
python -m memory.train_fact_classifier --threshold 0.3      # lower = fewer missed facts, more extra pins
```
- Data: `memory/fact_data.py` (templates plus hand-written; a held-out set with different phrasing). The training script
  prints precision and recall for the regex, the classifier and both combined, and lists every miss. Add examples for
  anything it gets wrong, then retrain.
- No `fact_clf.npz`, or one trained for a different embedder: the manager silently uses regex only.
- Switches (set before starting uvicorn): `FACT_CLF=off` regex only | default classifier OR regex |
  `FACT_STRONG_REGEX=0` classifier only | `FACT_THRESHOLD=0.5` override the saved threshold (default 0.4).
- Caveat: the training data was written by us. Real voice transcripts are messier. See `docs/RESULTS.md`.

## The budget, and how it grows with the chat

A fixed budget is shared by more and more messages as a chat lengthens, so old context is squeezed harder
(`python tools/sim_scaling.py`: at 100 turns a fixed 1500 keeps under half the messages in any form). The base budget
therefore grows with the number of user prompts (`memory/budget.py`). Choose a curve with `MEMORY_SCALE`:

| mode | shape | notes |
|---|---|---|
| `off` | fixed budget | use for the headline naive-vs-ours benchmark |
| `arctan` | `(2/pi) atan(prompts / BEND)` | **recommended**: big jump at the start, flattens towards the cap, never passes it |
| `root` | `u/(1+u)`, `u = sqrt(prompts / HALF)` | square-root start, slower asymptote to the cap |
| `smooth` | `1.5t - 0.5t^3` | steep start, lands exactly on the cap at prompt `END`, then flat |
| `log` | `GAIN * ln(1 + prompts - START)` | never negative, front-loaded, no cap of its own |
| `sqrt`, `linear`, `quad` | `GAIN * (prompts - START)^p` | older curves (START 30), softly capped; `sqrt` is the code default |

Budget in tokens (base 1500, cap 6000, steep modes starting at prompt 0, `quad` from prompt 30 with a growing history):

| prompts | 1 | 5 | 10 | 20 | 50 | 100 | 150 | 200 | 300 | 400 | 800 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `arctan` | 1571 | 1856 | 2201 | 2828 | 4067 | 4909 | 5253 | 5434 | 5620 | 5714 | 5856 |
| `root` | 1909 | 2322 | 2581 | 2890 | 3363 | 3750 | 3977 | 4136 | 4352 | 4500 | 4824 |
| `smooth` | 1526 | 1634 | 1769 | 2038 | 2832 | 4056 | 5064 | 5748 | 6000 | 6000 | 6000 |
| `quad` | 1500 | 1500 | 1500 | 1500 | 1599 | 2695 | 4488 | 5651 | 5997 | 5999 | 6000 |

For `arctan`, `root` and `smooth`: scaling starts at prompt 0 and there is no full-history guard by default, so the chart
is the pure curve. A budget larger than the chat just means nothing needs cutting. The older modes cap the budget at
`0.5 x the full history` (`MEMORY_SCALE_FRAC`) because an uncapped quadratic would overtake the linearly growing full
history and save nothing. The chosen value per turn is in `/debug/state` under `manager_info.scale`, and charted on the
dashboard. All settings: see `README.md`.

## Tuning, checks and known limits

Tuning constants are at the top of `scorer.py` (`W_SIM, W_REC, W_IMP, W_REF`, `HALF_LIFE_TURNS`, `REF_THRESHOLD`,
`PIN_IMPORTANCE`, `PIN_FLOOR`, `PAIR_FACTOR`) and `tiers.py` (`PIN_CAP`, `PIN_FULL_MAX`, `CRITICAL`).

```
python tests/test_memory.py          # no model download needed (fake embedder)
python -m memory.try_scorer          # prints a score table with the real MiniLM
python tools/probe_fact.py "im allergic to peanuts"      # how one sentence is seen
python -m bench.accuracy             # does it forget important things? (see docs/RESULTS.md)
```
Known limits:
- Tool-call messages (`role: "tool"`) are not specially protected, and dropping messages can break strict
  user/assistant alternation.
- The `critical` keyword list in `tiers.py` is narrower than the classifier: a health fact worded without those keywords
  ("I take insulin", "shellfish puts me in hospital") is pinned but ranked as ordinary when the lane overflows.
- The first request loads MiniLM (the proxy warms it up at start).
- Weights are guesses tuned on our own scripted conversations and synthetic test chats.
