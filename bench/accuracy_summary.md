# Do important facts survive? (offline retrieval test)

seed 1, 40 conversations per length, base budget 1500, scaling 'quad', embedder real. A fact counts as kept when its key phrase is in the prompt that would be sent.

### All planted facts

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

### Critical facts only (allergy, medication)

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

### Paraphrased facts the regex cannot see (classifier test)

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

### Updated fact (only the NEW value counts)

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

### Paired test: ours vs a window with the same token cost (exact McNemar)

| prompts | ours | window | only ours right | only window right | p |
|---|---|---|---|---|---|
| 50 | ours-fixed | window-fixed | 188 | 0 | 5.098e-57 |
| 50 | ours-scaled | window-scaled | 192 | 0 | 3.186e-58 |
| 100 | ours-fixed | window-fixed | 187 | 0 | 1.02e-56 |
| 100 | ours-scaled | window-scaled | 199 | 0 | 2.489e-60 |
| 200 | ours-fixed | window-fixed | 187 | 0 | 1.02e-56 |
| 200 | ours-scaled | window-scaled | 187 | 0 | 1.02e-56 |

Read it as: p < 0.05 with 'only ours right' much larger than 'only window right' means ours keeps important facts the same-cost window forgets. Naive is 100% by construction; its cost is the last column.
