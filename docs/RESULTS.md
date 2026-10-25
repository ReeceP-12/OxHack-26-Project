# Results: does the memory manager forget important things?

**Short answer.** At 50, 100 and 200 prompts the manager kept **100% of critical facts (allergy, medication)** in
every one of 247 test cases, while sending **6 to 46% of the tokens** that naive resending would. A window of
recent messages with the same token cost kept **0 to 4%** of the planted facts. The weak spot is facts phrased in
a way the regex cannot see: the learned classifier recovers most of them (77% at 200 prompts, 100% at 50 and 100
with the scaled budget), but not all.

Run on: real MiniLM (`all-MiniLM-L6-v2`), tiktoken token counts, learned fact classifier at threshold 0.4,
base budget 1500, 40 random conversations per length, seed 1 (`python -m bench.accuracy`).

---

## 1. How it was tested (`bench/accuracy.py`)

If a fact is not in the prompt the model receives, no model can recall it. So the test asks, for every important
fact: **is it still in the prompt that would be sent at the end of the call?**

1. Build random conversations of 50, 100 and 200 prompts with realistic 40 to 80 word messages and replies.
2. Plant 5 facts in the first 35% of each call, at random positions:
   a health-critical fact (allergy, shellfish, medication), 3 others drawn from ten templates (deadline, dog, budget,
   flight, sister, a meeting), and one fact that is **corrected later** ("My locker code is 4417" ... "it is now 9023";
   only the new value counts). Some are **paraphrased so the regex cannot see them** ("Peanuts make me really ill",
   "shellfish puts me in hospital", "the call with the Henderson account has been pencilled in ...").
3. Pass the whole history through each condition and check whether each fact's key phrase is in the sent prompt.

| condition | what it is |
|---|---|
| `naive` | resend everything. 100% by construction; it is the **cost** baseline |
| `window-fixed` | only the most recent messages, with the **same token count** as `ours-fixed` sent |
| `window-scaled` | only the most recent messages, with the same token count as `ours-scaled` sent |
| `ours-fixed` | the memory manager, fixed budget (`MEMORY_SCALE=off`) |
| `ours-scaled` | the memory manager, budget scaled with call length (`MEMORY_SCALE=quad`) |
| `ours-scaled-noclf` | as `ours-scaled` but regex only (no learned classifier) |

The same-cost window is the fair rival: identical tokens, but it forgets by age instead of by importance.
Statistics: survival rate with a 95% Wilson interval; paired exact McNemar test on the same fact instances.
"Kept" counts of 200 = 40 conversations x 5 facts; critical-only counts are smaller.

## 2. Results

### 2.1 All planted facts

| prompts | condition | kept | rate | 95% CI | tokens sent | % of naive |
|---|---|---|---|---|---|---|
| 50 | naive | 200/200 | 100.0% | 98-100% | 7065 | 100% |
| 50 | window-fixed | 0/200 | 0.0% | 0-2% | 1651 | 23% |
| 50 | window-scaled | 8/200 | 4.0% | 2-8% | 3183 | 45% |
| 50 | ours-fixed | 188/200 | 94.0% | 90-97% | 1715 | 24% |
| 50 | ours-scaled | 200/200 | 100.0% | 98-100% | 3250 | 46% |
| 50 | ours-scaled-noclf | 160/200 | 80.0% | 74-85% | 3156 | 45% |
| 100 | naive | 200/200 | 100.0% | 98-100% | 14340 | 100% |
| 100 | window-fixed | 0/200 | 0.0% | 0-2% | 1670 | 12% |
| 100 | window-scaled | 1/200 | 0.5% | 0-3% | 5952 | 42% |
| 100 | ours-fixed | 187/200 | 93.5% | 89-96% | 1755 | 12% |
| 100 | ours-scaled | 200/200 | 100.0% | 98-100% | 6019 | 42% |
| 100 | ours-scaled-noclf | 160/200 | 80.0% | 74-85% | 5883 | 41% |
| 200 | naive | 200/200 | 100.0% | 98-100% | 28893 | 100% |
| 200 | window-fixed | 0/200 | 0.0% | 0-2% | 1690 | 6% |
| 200 | window-scaled | 0/200 | 0.0% | 0-2% | 6186 | 21% |
| 200 | ours-fixed | 187/200 | 93.5% | 89-96% | 1787 | 6% |
| 200 | ours-scaled | 187/200 | 93.5% | 89-96% | 6288 | 22% |
| 200 | ours-scaled-noclf | 91/200 | 45.5% | 39-52% | 6119 | 21% |

### 2.2 Critical facts only (allergy, medication)

| prompts | condition | kept | rate | 95% CI | tokens sent | % of naive |
|---|---|---|---|---|---|---|
| 50 | naive | 83/83 | 100.0% | 96-100% | 7066 | 100% |
| 50 | window-fixed | 0/83 | 0.0% | 0-4% | 1651 | 23% |
| 50 | window-scaled | 0/83 | 0.0% | 0-4% | 3187 | 45% |
| 50 | ours-fixed | 83/83 | 100.0% | 96-100% | 1715 | 24% |
| 50 | ours-scaled | 83/83 | 100.0% | 96-100% | 3251 | 46% |
| 50 | ours-scaled-noclf | 83/83 | 100.0% | 96-100% | 3153 | 45% |
| 100 | naive | 80/80 | 100.0% | 95-100% | 14342 | 100% |
| 100 | window-fixed | 0/80 | 0.0% | 0-5% | 1679 | 12% |
| 100 | window-scaled | 0/80 | 0.0% | 0-5% | 5962 | 42% |
| 100 | ours-fixed | 80/80 | 100.0% | 95-100% | 1761 | 12% |
| 100 | ours-scaled | 80/80 | 100.0% | 95-100% | 6025 | 42% |
| 100 | ours-scaled-noclf | 80/80 | 100.0% | 95-100% | 5879 | 41% |
| 200 | naive | 84/84 | 100.0% | 96-100% | 28894 | 100% |
| 200 | window-fixed | 0/84 | 0.0% | 0-4% | 1693 | 6% |
| 200 | window-scaled | 0/84 | 0.0% | 0-4% | 6188 | 21% |
| 200 | ours-fixed | 84/84 | 100.0% | 96-100% | 1787 | 6% |
| 200 | ours-scaled | 84/84 | 100.0% | 96-100% | 6288 | 22% |
| 200 | ours-scaled-noclf | 28/84 | 33.3% | 24-44% | 6118 | 21% |

### 2.3 Paraphrased facts the regex cannot see (classifier test)

| prompts | condition | kept | rate | 95% CI | tokens sent | % of naive |
|---|---|---|---|---|---|---|
| 50 | naive | 56/56 | 100.0% | 94-100% | 7064 | 100% |
| 50 | window-fixed | 0/56 | 0.0% | 0-6% | 1649 | 23% |
| 50 | window-scaled | 0/56 | 0.0% | 0-6% | 3187 | 45% |
| 50 | ours-fixed | 44/56 | 78.6% | 66-87% | 1715 | 24% |
| 50 | ours-scaled | 56/56 | 100.0% | 94-100% | 3250 | 46% |
| 50 | ours-scaled-noclf | 56/56 | 100.0% | 94-100% | 3151 | 45% |
| 100 | naive | 47/47 | 100.0% | 92-100% | 14340 | 100% |
| 100 | window-fixed | 0/47 | 0.0% | 0-8% | 1675 | 12% |
| 100 | window-scaled | 0/47 | 0.0% | 0-8% | 5956 | 42% |
| 100 | ours-fixed | 34/47 | 72.3% | 58-83% | 1756 | 12% |
| 100 | ours-scaled | 47/47 | 100.0% | 92-100% | 6020 | 42% |
| 100 | ours-scaled-noclf | 47/47 | 100.0% | 92-100% | 5876 | 41% |
| 200 | naive | 56/56 | 100.0% | 94-100% | 28895 | 100% |
| 200 | window-fixed | 0/56 | 0.0% | 0-6% | 1677 | 6% |
| 200 | window-scaled | 0/56 | 0.0% | 0-6% | 6176 | 21% |
| 200 | ours-fixed | 43/56 | 76.8% | 64-86% | 1783 | 6% |
| 200 | ours-scaled | 43/56 | 76.8% | 64-86% | 6284 | 22% |
| 200 | ours-scaled-noclf | 9/56 | 16.1% | 9-28% | 6113 | 21% |

### 2.4 The corrected fact (only the NEW value counts)

| prompts | condition | kept | rate | 95% CI | tokens sent | % of naive |
|---|---|---|---|---|---|---|
| 50 | naive | 40/40 | 100.0% | 91-100% | 7065 | 100% |
| 50 | window-fixed | 0/40 | 0.0% | 0-9% | 1651 | 23% |
| 50 | window-scaled | 8/40 | 20.0% | 10-35% | 3183 | 45% |
| 50 | ours-fixed | 40/40 | 100.0% | 91-100% | 1715 | 24% |
| 50 | ours-scaled | 40/40 | 100.0% | 91-100% | 3250 | 46% |
| 50 | ours-scaled-noclf | 0/40 | 0.0% | 0-9% | 3156 | 45% |
| 100 | naive | 40/40 | 100.0% | 91-100% | 14340 | 100% |
| 100 | window-fixed | 0/40 | 0.0% | 0-9% | 1670 | 12% |
| 100 | window-scaled | 1/40 | 2.5% | 0-13% | 5952 | 42% |
| 100 | ours-fixed | 40/40 | 100.0% | 91-100% | 1755 | 12% |
| 100 | ours-scaled | 40/40 | 100.0% | 91-100% | 6019 | 42% |
| 100 | ours-scaled-noclf | 0/40 | 0.0% | 0-9% | 5883 | 41% |
| 200 | naive | 40/40 | 100.0% | 91-100% | 28893 | 100% |
| 200 | window-fixed | 0/40 | 0.0% | 0-9% | 1690 | 6% |
| 200 | window-scaled | 0/40 | 0.0% | 0-9% | 6186 | 21% |
| 200 | ours-fixed | 40/40 | 100.0% | 91-100% | 1787 | 6% |
| 200 | ours-scaled | 40/40 | 100.0% | 91-100% | 6288 | 22% |
| 200 | ours-scaled-noclf | 0/40 | 0.0% | 0-9% | 6119 | 21% |

### 2.5 Paired test: ours vs a window with the same token cost (exact McNemar)

| prompts | ours | window | only ours right | only window right | p |
|---|---|---|---|---|---|
| 50 | ours-fixed | window-fixed | 188 | 0 | 5.098e-57 |
| 50 | ours-scaled | window-scaled | 192 | 0 | 3.186e-58 |
| 100 | ours-fixed | window-fixed | 187 | 0 | 1.02e-56 |
| 100 | ours-scaled | window-scaled | 199 | 0 | 2.489e-60 |
| 200 | ours-fixed | window-fixed | 187 | 0 | 1.02e-56 |
| 200 | ours-scaled | window-scaled | 187 | 0 | 1.02e-56 |

## 3. What this means

- **Ours vs the same-cost window.** In every pair there is no fact the window kept that ours dropped
  ("only window right" = 0), and ours was right on 187 to 199 facts the window forgot. The difference is not luck.
- **Critical facts.** 100% kept at every length, in all 247 critical cases, with and without scaling.
- **The learned classifier matters.** With regex only, critical facts fall to 33% and paraphrased facts to 16% at 200
  prompts, and the corrected value is lost 100% of the time at every length. (At 50 and 100 prompts regex-only still
  looks fine for critical facts because the budget has room; it fails only once space gets tight.)
- **Cost.** Naive cannot lose a fact but sends 4 to 16 times as many tokens at these lengths.
- **Scaling.** At 50 and 100 prompts the scaled budget lifts retention from about 94% to 100%, for about double the
  tokens of the fixed budget (still 42 to 46% of naive). At 200 prompts it does not help the paraphrased facts.

## 4. Known weak spot and caveats

- About a quarter of paraphrased facts at 200 prompts (13 of 56) were never recognised as facts, so neither budget
  setting keeps them. This is a classifier miss, not a budget problem. To see which facts:
  `python -c "import csv,collections as c; d=c.Counter(); [d.update([r['fact']]) for r in csv.DictReader(open('bench/accuracy_facts.csv')) if r['cond']=='ours-scaled' and r['length']=='200' and r['kept']=='0']; print(d)"`
- **This measures presence in the prompt, not whether the model then answers correctly.** Presence is necessary; use
  `python -m bench.accuracy --live` (proxy running with real models) for end-to-end recall.
- The window baseline always loses here by construction, since every fact is planted in the first 35% of the call.
  It is the right same-cost rival but not a clever one.
- The test conversations are synthetic (repetitive filler). Real calls are messier.
- Some topics in the test sentences (shellfish, insulin, a sister's name) also appear in the classifier's training
  data. The paraphrased numbers may be slightly optimistic until tested on sentences written fresh.
- 40 conversations per length gives intervals of a few points; use `--trials 100` before quoting exact figures.

## 5. Budget scaling simulation (`tools/sim_scaling.py`, no real model)

Base 1500, ~700-token system message, ~85-token replies. "any%" = share of messages still present in some form;
"callback" = whether a fact planted early and asked about at the end survived. (This simulation predates the soft
ceiling, so exact budgets are a little different now; the shape is the same.)

| prompts | mode | base to effective | sent | full history | sent % | any% | callback |
|---|---|---|---|---|---|---|---|
| 50 | fixed | 1500 to 1500 | 1631 | 5438 | 30% | 93% | kept |
| 50 | sqrt | 1500 to 2049 | 2180 | 5438 | 40% | 100% | kept |
| 50 | linear | 1500 to 1752 | 1883 | 5438 | 35% | 100% | kept |
| 50 | quad | 1500 to 1610 | 1729 | 5438 | 32% | 100% | kept |
| 100 | fixed | 1500 to 1500 | 1696 | 10863 | 16% | 48% | **LOST** |
| 100 | sqrt | 1500 to 2511 | 2707 | 10863 | 25% | 88% | kept |
| 100 | linear | 1500 to 2352 | 2548 | 10863 | 23% | 82% | kept |
| 100 | quad | 1500 to 2760 | 2956 | 10863 | 27% | 98% | kept |
| 200 | fixed | 1500 to 1500 | 1696 | 21662 | 8% | 24% | **LOST** |
| 200 | sqrt | 1500 to 3069 | 3265 | 21662 | 15% | 55% | **LOST** |
| 200 | linear | 1500 to 3552 | 3748 | 21662 | 17% | 64% | **LOST** |
| 200 | quad | 1500 to 6000 | 6196 | 21662 | 29% | 100% | kept |

A fixed budget loses an early fact by 100 prompts; sqrt and linear lose it by 200; quad keeps it. The budget is
capped at `min(MEMORY_BUDGET_MAX, 0.5 x full history)` and eases into the cap smoothly, because an uncapped
quadratic would overtake the (linear) full-history size and save nothing.

## 6. Reproduce

```bash
source .venv/bin/activate
python tests/test_accuracy.py                          # sanity-check the statistics code
python -m bench.accuracy --trials 100                  # offline; writes bench/accuracy_summary.md + accuracy_facts.csv
python tools/sim_scaling.py --lengths 20 50 100 200 --modes off sqrt linear quad
# live model recall (proxy running with real models and MEMORY_SCALE=quad):
python -m bench.accuracy --live --live-trials 10
```
Full list of commands and settings: `README.md`. How the manager works: `docs/HOW_IT_WORKS.md`.
