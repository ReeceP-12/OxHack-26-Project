# memory/: the context manager

Takes the full chat history ElevenLabs sends every turn and returns a shorter list that still
remembers what matters. **No LLM calls**, only a small local embedding model (MiniLM) and regex.

## Use it

```python
from memory.context_manager import ContextManager

cm = ContextManager()                      # create once; loads MiniLM (slow the first time)
kept, tiers = cm.build(messages, budget=1500)
```

- `messages`: OpenAI-style list of `{"role": ..., "content": ...}`.
- `budget`: token budget for everything sent (system message included). Counted with tiktoken.
- `kept`: the reduced list, in original order. Send this to the model.
- `tiers`: one entry per ORIGINAL message, for `/debug/state` and the dashboard:

```json
{"index": 4, "role": "user", "tier": "full", "score": 0.73, "tokens": 41, "tokens_full": 41}
```

| tier | meaning |
|---|---|
| `full` | sent unchanged |
| `summary` | cut to its first meaningful sentence |
| `archived` | not sent this turn (can come back next turn if it scores higher) |
| `gist` | reserved, not built yet |

`tokens` is what was actually sent (0 if archived). `tokens_full` is the original size.

## How a message is scored (0 to about 1, recomputed every turn)

```
score = 0.4*similarity + 0.2*recency + 0.3*importance + 0.1*referenced
```
- similarity: meaning-closeness to the latest user message (MiniLM cosine)
- recency: halves every 10 turns
- importance: regex hits (numbers, dates, "my X is", allergies, "remember", "never/always"...)
- referenced: how many later messages are about the same thing
- **pin floor**: a user message with importance >= 0.8 never scores below 0.7, so stated facts survive
  even when the conversation moves on (weights alone could not fix this; see the regression test)

Always sent in full: system messages and the last 4 messages. The rest are filled greedily by score:
full text if it fits, else first-sentence summary, else archived.

## Tuning (all constants at the top of `scorer.py`)
`W_SIM, W_REC, W_IMP, W_REF`, `HALF_LIFE_TURNS`, `REF_THRESHOLD`, `PIN_IMPORTANCE`, `PIN_FLOOR`.
Budget lives in `context_manager.py` (`DEFAULT_BUDGET`) and in the call from `main.py`.

## Check it
```
python tests/test_memory.py        # no model download needed (uses a fake embedder)
python -m memory.try_scorer        # prints a score table with the real MiniLM
python tools/send_convo.py         # needs the proxy running; ON should go flat, OFF should climb
```

## Known limits
- Tool-call messages (`role: "tool"`) are not specially protected.
- Dropping messages can break strict user/assistant alternation.
- First request loads MiniLM (warm it up at startup).
- Weights and the 1500 budget are guesses until the recall benchmark exists.