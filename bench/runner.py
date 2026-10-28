"""Run the scripted conversation against the proxy and log per-turn metrics.

Usage (from repo root, proxy running on :8000):
  python -m bench.runner --mode off --label naive
  python -m bench.runner --mode on  --label ours
  python -m bench.runner --mode off --label window20 --window 20
The last one is a crude "forget old messages" baseline: it sends only the last N
messages each turn (N=0 means send everything). It needs no proxy support, and it
shows the benchmark can detect forgetting. Compare it with: python -m bench.plot --ours window20
The runner flips the proxy's switch (POST /debug/mode), resets the turn log
(POST /debug/reset), always sends the FULL history, then saves /debug/state.

An error reply (HTTP error, timeout, or an error body such as a Gemini 503)
counts as a FAILED turn, not a reply: it is logged with failed=1, its user
message is dropped from the history, and it never counts as a recall answer.
"""
import argparse, csv, json, re, time, requests
from bench.conversation import build_turns, build_turns_from, RECALL, RECALL_PARAPHRASE, score_reply

def chat(url, model, messages, timeout=60):
    """Returns (reply, latency, prompt_tokens, error). error is None on success."""
    t0 = time.perf_counter()
    try:
        r = requests.post(url, json={"model": model, "messages": messages}, timeout=timeout)
    except requests.RequestException as e:
        return None, time.perf_counter() - t0, -1, f"request failed: {e}"
    latency = time.perf_counter() - t0
    if r.status_code != 200:
        return None, latency, -1, f"HTTP {r.status_code}: {r.text[:100]}"
    try:
        data = r.json()
        if "error" in data or not data.get("choices"):
            return None, latency, -1, f"error body: {str(data)[:100]}"
        reply = data["choices"][0]["message"]["content"]
    except (ValueError, KeyError, TypeError) as e:
        return None, latency, -1, f"bad response: {e}"
    if not reply or not reply.strip():
        # The model returned nothing (seen once in ours2 on the allergy question). That is not
        # an answer, so it is treated like a refused call and retried, and counted in "retries".
        return None, latency, -1, "empty reply from the model"
    prompt_tokens = (data.get("usage") or {}).get("prompt_tokens", -1)
    return reply, latency, prompt_tokens, None

def chat_retry(url, model, messages, retries, wait):
    """Like chat(), but a failed call is tried again up to `retries` more times after
    `wait` seconds (Gemini sometimes refuses a call for a moment: busy or a per-minute limit).
    Returns (reply, latency, prompt_tokens, error, retries_used). Latency is the last attempt only."""
    used = 0
    while True:
        reply, lat, ptok, err = chat(url, model, messages)
        if not err or used >= retries:
            return reply, lat, ptok, err, used
        used += 1
        print(f"        retry {used}/{retries} in {wait:.0f}s after: {err[:60]}")
        time.sleep(wait)

# Sent as the first message of EVERY request, in every mode, so the comparison stays fair.
# Without it the model drifted into a shouting persona when its history was cut down.
# Pass --system "" to turn it off. Whoever writes manager_build() must keep this message.
SYSTEM_DEFAULT = ("You are a calm, polite assistant. Answer politely and naturally. "
                  "Never shout or use capital letters for emphasis.")

def prep(messages, n, system):
    """Window the history (last_n), then put the system message first when there is one."""
    msgs = last_n(messages, n)
    return ([{"role": "system", "content": system}] + msgs) if system else msgs

def is_shouting(reply):
    """True if a reply looks like the model has drifted into a shouting or theatrical persona:
    lots of all-caps words, or repeated '?!'. Normal replies use neither."""
    words = re.findall(r"[A-Za-z']+", reply)
    caps = [w for w in words if len(w) >= 3 and w.isupper()]
    return reply.count("?!") >= 2 or (len(words) >= 10 and len(caps) / len(words) > 0.2)

def last_n(messages, n):
    """The last n messages (0 = all), but never starting on an assistant reply.

    A conversation that opens with the model's own reply and no question before it is
    unnatural and can push the model off its normal behaviour, so any leading assistant
    messages are dropped (a window of 20 therefore becomes 19 messages)."""
    msgs = messages[-n:] if n else messages
    while msgs and msgs[0]["role"] != "user":
        msgs = msgs[1:]
    return msgs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--mode", choices=["on", "off"], required=True,
                    help="on = context manager, off = naive")
    ap.add_argument("--label", required=True)
    ap.add_argument("--model", default="gemini-flash")
    ap.add_argument("--system", default=SYSTEM_DEFAULT,
                    help='system message sent with every request ("" to disable)')
    ap.add_argument("--retries", type=int, default=3,
                    help="extra attempts for a call that fails (0 = never retry)")
    ap.add_argument("--retry-wait", type=float, default=8.0,
                    help="seconds to wait before each retry")
    ap.add_argument("--paraphrase", action="store_true",
                    help="after the 5 recall questions, also ask 5 reworded questions (RECALL_PARAPHRASE)")
    ap.add_argument("--window", type=int, default=0,
                    help="send only the last N messages each turn (0 = send everything)")
    ap.add_argument("--questions", default="",
                    help="text file, one question per line, used as filler instead of the built-in chatter "
                         "(100 lines + 7 planted facts = 107 turns)")
    a = ap.parse_args()
    url = a.base + "/v1/chat/completions"

    # set mode and clear the turn log so runs don't mix
    r = requests.post(a.base + "/debug/mode", json={"manager_on": a.mode == "on"}, timeout=10)
    r.raise_for_status()
    requests.post(a.base + "/debug/reset", timeout=10).raise_for_status()
    print(f"mode={a.mode}, window={a.window or 'all'}, system prompt={'on' if a.system else 'off'}, log reset")

    history, rows, failed_turns, shouts, total_retries = [], [], 0, 0, 0
    turns = build_turns_from(a.questions) if a.questions else build_turns()
    print(f"{len(turns)} turns" + (f" (filler from {a.questions})" if a.questions else ""))
    for i, text in enumerate(turns):
        history.append({"role": "user", "content": text})
        reply, lat, ptok, err, used = chat_retry(url, a.model, prep(history, a.window, a.system),
                                                 a.retries, a.retry_wait)
        total_retries += used
        if err:
            history.pop()  # failed turn: drop the user message, keep history consistent
            failed_turns += 1
            print(f"turn {i:2d}  FAILED  {err}")
        else:
            history.append({"role": "assistant", "content": reply})
            if is_shouting(reply):
                shouts += 1
            print(f"turn {i:2d}  {lat:5.2f}s  prompt_tokens={ptok}" + ("  SHOUTING" if is_shouting(reply) else ""))
        rows.append({"label": a.label, "mode": a.mode, "window": a.window, "system": int(bool(a.system)), "turn": i,
                     "latency_s": round(lat, 3), "prompt_tokens": ptok,
                     "failed": int(bool(err))})

    # full transcript of the 50 turns, so drift can be diagnosed afterwards
    with open(f"bench/transcript_{a.label}.json", "w") as f:
        json.dump(history, f, indent=2)

    # save the tier map / turn records before recall questions add extra turns
    try:
        state = requests.get(a.base + "/debug/state", timeout=10).json()
        with open(f"bench/state_{a.label}.json", "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        print("could not save /debug/state:", e)

    score, errors, hedges, answers = 0, 0, 0, []
    p_score, p_errors = 0, 0
    question_sets = [("original", RECALL)]
    if a.paraphrase:
        question_sets.append(("paraphrase", RECALL_PARAPHRASE))
    for qset, questions in question_sets:
        for q, kws in questions:
            reply, _, _, err, used = chat_retry(
                url, a.model,
                prep(history + [{"role": "user", "content": q}], a.window, a.system),
                a.retries, a.retry_wait)
            total_retries += used
            if err:
                if qset == "original":
                    errors += 1
                else:
                    p_errors += 1
                print(f"[ERROR] {q} -> {err}")
                continue
            ok, hedged = score_reply(reply, kws)
            if qset == "original":
                score += ok
            else:
                p_score += ok
            hedges += hedged
            answers.append({"set": qset, "question": q, "correct": bool(ok),
                            "hedged": bool(hedged), "reply": reply})
            tag = "OK" if ok else ("MISS (hedged: keyword present but reply refused)" if hedged else "MISS")
            print(f"[{tag}]{' (reworded)' if qset == 'paraphrase' else ''} {q} -> {reply[:80]!r}")
    # state again, now including the 5 recall-question turns (recall_tiers.py reads this)
    try:
        state = requests.get(a.base + "/debug/state", timeout=10).json()
        with open(f"bench/state_{a.label}_recall.json", "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        print("could not save the recall-time state:", e)
    print(f"\nRECALL {a.label}: {score}/{len(RECALL)}"
          f"  (failed turns: {failed_turns}, errored recall questions: {errors}, hedged replies: {hedges}, shouting replies: {shouts}, retries used: {total_retries})")
    if a.paraphrase:
        print(f"PARAPHRASE {a.label}: {p_score}/{len(RECALL_PARAPHRASE)}  (errored reworded questions: {p_errors})")
    if total_retries:
        print(f"NOTE: {total_retries} calls were retried (refused or empty reply the first time). "
              "Report this number next to the results.")
    if failed_turns or errors:
        print("WARNING: this run had errors. Do not use it for the final numbers.")
    if shouts:
        print("WARNING: the model drifted into shouting. Read bench/transcript_" + a.label + ".json before trusting this run.")

    with open(f"bench/replies_{a.label}.json", "w") as f:
        json.dump(answers, f, indent=2)

    with open(f"bench/results_{a.label}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)

if __name__ == "__main__":
    main()
