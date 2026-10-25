"""Does the memory manager forget important things? A statistical test, with and without a live model.

Question: when a user states a fact early in a long call, is it still in the prompt the model receives at the end?
If the fact is not in the prompt, no model can recall it. If it is, recall depends only on the model.

How it works (offline, no API keys, repeatable with --seed):
  1. Build many random conversations of L prompts (default 50, 100 and 200) with realistic 40 to 80 word
     messages and replies. In each, 5 facts are planted early (some health critical, some ordinary, some
     paraphrased so the regex cannot see them, one that is later UPDATED), at random positions.
  2. Pass the final history through each condition and check, for every planted fact, whether its key phrase
     survives in what would be sent:
        naive            everything is resent (100% by construction; it is the cost baseline)
        window-fixed     only the most recent messages, with the SAME token count as ours-fixed sent
        window-scaled    only the most recent messages, with the same token count as ours-scaled sent
        ours-fixed       the memory manager, fixed budget (MEMORY_SCALE=off)
        ours-scaled      the memory manager, budget scaled with length (MEMORY_SCALE=quad)
        ours-scaled-noclf  as ours-scaled but without the learned classifier (regex only)
     The token-matched window is the fair rival: same cost, but it forgets by age instead of by importance.
  3. Report survival rate with a 95% Wilson interval, tokens sent, and an exact McNemar test (paired, same
     fact instances) of ours vs the token-matched window.

    python -m bench.accuracy                          # real MiniLM + classifier, 40 trials per length
    python -m bench.accuracy --trials 100 --lengths 50 100 200 400
    python -m bench.accuracy --embedder fake --trials 10     # no model download; classifier is skipped
    python -m bench.accuracy --live --live-trials 10         # ALSO ask a real model through the running proxy

Writes bench/accuracy_facts.csv (one row per fact and condition) and bench/accuracy_summary.md.
--live needs the proxy running with real models (see README). It sends only the final request per question
(one call per fact per condition), not the whole conversation turn by turn, so it is cheap.
"""
import argparse, csv, math, os, random, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---- facts ----------------------------------------------------------------------------------------
# crit = health/safety, para = phrased so the regex cannot see it (only the learned classifier can pin it).
# Keys are matched in lower case against the sent prompt.
FACTS = [
    dict(id="peanut",   crit=True,  para=False, key=["peanut"],
         say="Quick heads up, I'm severely allergic to peanuts, so keep that in mind for anything food related.",
         ask="What am I allergic to?"),
    dict(id="peanut_p", crit=True,  para=True,  key=["peanut"],
         say="Peanuts make me really ill, so please never suggest them to me.",
         ask="Which nut should you never suggest to me?"),
    dict(id="shellfish", crit=True, para=True,  key=["shellfish"],
         say="Just so you know, shellfish puts me in hospital.",
         ask="Which food puts me in hospital?"),
    dict(id="insulin",  crit=True,  para=False, key=["insulin"],
         say="I take insulin twice a day, so regular mealtimes really matter to me.",
         ask="What medicine do I take twice a day?"),
    dict(id="deadline", crit=False, para=False, key=["14th"],
         say="My project deadline is Friday the 14th, please remember that.",
         ask="What is my project deadline?"),
    dict(id="dog",      crit=False, para=False, key=["biscuit"],
         say="My dog is called Biscuit and he's a three year old beagle.",
         ask="What is my dog called?"),
    dict(id="budget",   crit=False, para=False, key=["1200"],
         say="My budget for the trip is 1200 pounds in total.",
         ask="What is my total trip budget?"),
    dict(id="flight",   crit=False, para=False, key=["ba 249"],
         say="My flight is BA 249 and it leaves at 6:40am.",
         ask="What is my flight number?"),
    dict(id="sister",   crit=False, para=False, key=["priya"],
         say="My sister Priya is getting married in June, so I need that weekend free.",
         ask="What is my sister's name?"),
    dict(id="meeting",  crit=False, para=True,  key=["henderson"],
         say="The call with the Henderson account has been pencilled in for half past three on Thursday.",
         ask="Which account is my Thursday call with?"),
]
# a fact that is later CHANGED: only the new value counts
UPDATE = dict(id="locker", crit=False, para=False, key=["9023"], update=True,
              say="My locker code is 4417.", say2="Update: my locker code has changed, it is now 9023.",
              ask="What is my locker code now?")

TOPICS = ["octopuses", "the history of the bicycle", "how tides work", "learning guitar chords", "the northern lights",
          "volcanoes", "making a decent cup of coffee", "how vaccines work", "board games for four people",
          "why the sky is blue", "migrating birds", "keeping a houseplant alive", "how airplanes stay up",
          "the deepest part of the ocean", "mental arithmetic tricks", "how compasses work", "why cats purr"]
PADS = ["I would like a reasonably detailed answer with a couple of examples, as if I had never come across it. ",
        "Keep it friendly and practical, and mention anything surprising that most people get wrong about it. ",
        "I only have a few minutes, so give me the key points first and then a bit of the background. ",
        "Explain it the way you would to a curious teenager, with one good analogy if you can think of one. ",
        "Start from the very basics, then tell me what an expert would find interesting about it. "]
REPLY = ("Happy to help with that. {t} is a topic with a lot of interesting detail, so here is a short tour. "
         "First, the basic idea is simpler than it sounds, and most of the confusion comes from a few common "
         "myths. Second, there are two or three examples that make it concrete, and each one shows a different "
         "side of it. Finally, if you want to go further, I can suggest where to look next or go through any "
         "part of this in more depth, whichever you prefer.")
SYSTEM = "You are a calm, polite voice assistant."


def make_conversation(rng, n_prompts):
    """Returns (messages, planted) where planted = [(fact, prompt_index)] and messages ends on a user prompt."""
    first = rng.choice([f for f in FACTS if f["crit"]])                       # always plant one critical fact
    chosen = [first] + rng.sample([f for f in FACTS if f is not first], 3)    # plus 3 others, plus the update
    late = max(5, int(n_prompts * 0.35))                                      # facts are planted in the first 35%
    slots = rng.sample(range(1, late), 5)
    at = {s: (f, f["say"]) for f, s in zip(chosen, slots[:4])}
    upd_first = slots[4]
    upd_second = rng.randrange(int(n_prompts * 0.4), int(n_prompts * 0.6))    # the correction comes later
    at[upd_first] = (UPDATE, UPDATE["say"])
    at[upd_second] = (UPDATE, UPDATE["say2"])
    msgs = [{"role": "system", "content": SYSTEM}]
    for i in range(n_prompts):
        text = at[i][1] if i in at else f"Tell me about {rng.choice(TOPICS)}, number {i}. {rng.choice(PADS)}"
        msgs.append({"role": "user", "content": text})
        if i < n_prompts - 1:
            msgs.append({"role": "assistant", "content": REPLY.format(t=rng.choice(TOPICS).capitalize())})
    planted = [(f, i) for i, (f, _) in sorted(at.items()) if i != upd_first]   # the update is scored on its NEW value
    return msgs, planted


# ---- conditions -----------------------------------------------------------------------------------
def window_to(messages, tokens, count):
    """System message + the most recent messages that fit `tokens` (always starting on a user message)."""
    sysm = [m for m in messages if m["role"] == "system"]
    body = [m for m in messages if m["role"] != "system"]
    used = sum(count(m["content"]) for m in sysm)
    keep = []
    for m in reversed(body):
        c = count(m["content"])
        if used + c > tokens:
            break
        keep.append(m)
        used += c
    keep.reverse()
    while keep and keep[0]["role"] != "user":
        keep = keep[1:]
    return sysm + keep


def survives(sent, fact):
    text = " ".join(m["content"] for m in sent if m["role"] != "system").lower()
    return any(k in text for k in fact["key"])


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(b, c):
    """Two-sided exact p for b vs c discordant pairs (b: only A right, c: only B right)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


# ---- main -----------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=40, help="conversations per length")
    ap.add_argument("--lengths", type=int, nargs="+", default=[50, 100, 200], help="prompts per conversation")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--embedder", choices=["real", "fake"], default="real")
    ap.add_argument("--budget", type=int, default=1500)
    ap.add_argument("--scale", default="quad", help="scaling mode for the 'scaled' conditions")
    ap.add_argument("--live", action="store_true", help="also ask a real model through the running proxy")
    ap.add_argument("--live-trials", type=int, default=10, help="conversations per length for --live")
    ap.add_argument("--base", default="http://localhost:8000")
    a = ap.parse_args()

    from memory.context_manager import ContextManager
    from memory.tiers import count_tokens
    if a.embedder == "fake":
        from tests.test_memory import FakeEmbedder
        emb, clf_mode = FakeEmbedder(), None
        print("NOTE: fake embedder, so the learned classifier is skipped. Use the real one for the real numbers.")
    else:
        from memory.scorer import Embedder
        emb, clf_mode = Embedder(), "auto"
    cm = ContextManager(emb, fact_clf=clf_mode)
    cm_noclf = ContextManager(emb, fact_clf=None)
    print(f"classifier: {'loaded, threshold ' + str(cm.fact_clf.threshold) if cm.fact_clf else 'not in use'}")

    def run(cmx, msgs, mode):
        os.environ["MEMORY_SCALE"] = mode
        return cmx.build(msgs, a.budget)[0]

    def toks(ms):
        return sum(count_tokens(m["content"]) for m in ms)

    CONDS = ["naive", "window-fixed", "window-scaled", "ours-fixed", "ours-scaled", "ours-scaled-noclf"]
    rows, live_rows = [], []
    t0 = time.time()
    for L in a.lengths:
        rng = random.Random(f"{a.seed}-{L}")
        for trial in range(a.trials):
            msgs, planted = make_conversation(rng, L)
            sent = {"naive": msgs}
            sent["ours-fixed"] = run(cm, msgs, "off")
            sent["ours-scaled"] = run(cm, msgs, a.scale)
            sent["ours-scaled-noclf"] = run(cm_noclf, msgs, a.scale)
            sent["window-fixed"] = window_to(msgs, toks(sent["ours-fixed"]), count_tokens)
            sent["window-scaled"] = window_to(msgs, toks(sent["ours-scaled"]), count_tokens)
            for f, idx in planted:
                for c in CONDS:
                    rows.append(dict(length=L, trial=trial, fact=f["id"], crit=int(f["crit"]), para=int(f["para"]),
                                     update=int(bool(f.get("update"))), cond=c, kept=int(survives(sent[c], f)),
                                     tokens=toks(sent[c]), naive_tokens=toks(msgs)))
            if a.live and trial < a.live_trials:
                live_rows += live_trial(a, L, trial, msgs, planted)
        print(f"  length {L}: {a.trials} trials done ({time.time() - t0:.0f}s)", flush=True)

    with open("bench/accuracy_facts.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    out = []

    def P(s=""):
        print(s); out.append(s)

    def table(title, pick):
        P(f"\n### {title}\n")
        P("| prompts | condition | kept | rate | 95% CI | tokens sent | % of naive |")
        P("|---|---|---|---|---|---|---|")
        for L in a.lengths:
            for c in CONDS:
                sel = [r for r in rows if r["length"] == L and r["cond"] == c and pick(r)]
                if not sel:
                    continue
                k, n = sum(r["kept"] for r in sel), len(sel)
                lo, hi = wilson(k, n)
                tk = sum(r["tokens"] for r in sel) / n
                nv = sum(r["naive_tokens"] for r in sel) / n
                P(f"| {L} | {c} | {k}/{n} | {100 * k / n:.1f}% | {100 * lo:.0f}-{100 * hi:.0f}% | {tk:.0f} | {100 * tk / nv:.0f}% |")

    P("# Do important facts survive? (offline retrieval test)")
    P(f"\nseed {a.seed}, {a.trials} conversations per length, base budget {a.budget}, scaling '{a.scale}', "
      f"embedder {a.embedder}. A fact counts as kept when its key phrase is in the prompt that would be sent.")
    table("All planted facts", lambda r: True)
    table("Critical facts only (allergy, medication)", lambda r: r["crit"] == 1)
    table("Paraphrased facts the regex cannot see (classifier test)", lambda r: r["para"] == 1)
    table("Updated fact (only the NEW value counts)", lambda r: r["update"] == 1)

    P("\n### Paired test: ours vs a window with the same token cost (exact McNemar)\n")
    P("| prompts | ours | window | only ours right | only window right | p |")
    P("|---|---|---|---|---|---|")
    key = lambda r: (r["trial"], r["fact"], r["update"])
    for L in a.lengths:
        for o, wnd in (("ours-fixed", "window-fixed"), ("ours-scaled", "window-scaled")):
            A = {key(r): r["kept"] for r in rows if r["length"] == L and r["cond"] == o}
            B = {key(r): r["kept"] for r in rows if r["length"] == L and r["cond"] == wnd}
            b = sum(1 for k in A if A[k] and not B[k]); c = sum(1 for k in A if B[k] and not A[k])
            P(f"| {L} | {o} | {wnd} | {b} | {c} | {mcnemar_exact(b, c):.4g} |")
    P("\nRead it as: p < 0.05 with 'only ours right' much larger than 'only window right' means ours keeps "
      "important facts the same-cost window forgets. Naive is 100% by construction; its cost is the last column.")

    if live_rows:
        P("\n# Live model recall (through the proxy)\n")
        P("| prompts | system | correct | rate | 95% CI |")
        P("|---|---|---|---|---|")
        for L in a.lengths:
            for s in ("naive", "ours"):
                sel = [r for r in live_rows if r["length"] == L and r["system"] == s and not r["error"]]
                if not sel:
                    continue
                k, n = sum(r["ok"] for r in sel), len(sel)
                lo, hi = wilson(k, n)
                P(f"| {L} | {s} | {k}/{n} | {100 * k / n:.1f}% | {100 * lo:.0f}-{100 * hi:.0f}% |")
            A = {(r["trial"], r["fact"]): r["ok"] for r in live_rows if r["length"] == L and r["system"] == "ours" and not r["error"]}
            B = {(r["trial"], r["fact"]): r["ok"] for r in live_rows if r["length"] == L and r["system"] == "naive" and not r["error"]}
            both = A.keys() & B.keys()
            b = sum(1 for k in both if A[k] and not B[k]); c = sum(1 for k in both if B[k] and not A[k])
            P(f"\nlength {L}: ours right / naive wrong = {b}, naive right / ours wrong = {c}, exact McNemar p = {mcnemar_exact(b, c):.4g}\n")
        errs = sum(1 for r in live_rows if r["error"])
        if errs:
            P(f"({errs} live requests failed and were left out. Do not trust a run with many.)")

    with open("bench/accuracy_summary.md", "w") as fh:
        fh.write("\n".join(out) + "\n")
    print("\nwrote bench/accuracy_summary.md and bench/accuracy_facts.csv")


def live_trial(a, L, trial, msgs, planted):
    """Ask each planted fact's question through the proxy twice: manager OFF (naive) then ON (ours)."""
    import requests
    from bench.conversation import score_reply
    url = a.base + "/v1/chat/completions"
    rows = []
    for f, _ in planted:
        for name, on in (("naive", False), ("ours", True)):
            requests.post(a.base + "/debug/mode", json={"manager_on": on}, timeout=10).raise_for_status()
            body = {"model": "gemini-flash", "messages": msgs + [{"role": "assistant", "content": REPLY.format(t="That")},
                                                                  {"role": "user", "content": f["ask"]}]}
            err, ok = None, 0
            try:
                r = requests.post(url, json=body, timeout=60)
                data = r.json()
                if r.status_code != 200 or "error" in data or not data.get("choices"):
                    err = f"HTTP {r.status_code}"
                else:
                    ok = int(score_reply(data["choices"][0]["message"]["content"], f["key"])[0])
            except Exception as e:
                err = str(e)[:80]
            rows.append(dict(length=L, trial=trial, fact=f["id"], system=name, ok=ok, error=err))
    return rows


if __name__ == "__main__":
    main()
