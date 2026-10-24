"""Measures how reliable the proxy + model is, the way ElevenLabs would see it.
Sends N sequential requests (non-streaming) and counts how many answer within the window.
The proxy must be running (with your real settings). ElevenLabs gives up after roughly 6 to 8 s.

    python tools/soak.py [N=15] [window_seconds=6]
"""
import sys, time, statistics, httpx

n = int(sys.argv[1]) if len(sys.argv) > 1 else 15
window = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0
msgs = [{"role": "system", "content": "You are a helpful voice assistant. Keep replies short."},
        {"role": "user", "content": "What is the capital of France?"}]
ok_times, fails = [], 0
for i in range(n):
    t0 = time.time()
    try:
        r = httpx.post("http://localhost:8000/v1/chat/completions", json={"messages": msgs}, timeout=window)
        dt = time.time() - t0
        if r.status_code == 200:
            ok_times.append(dt); print(f"{i+1:2d}  OK    {dt:4.1f}s  {r.json()['choices'][0]['message']['content'][:50]!r}")
        else:
            fails += 1; print(f"{i+1:2d}  FAIL  {dt:4.1f}s  HTTP {r.status_code}")
    except Exception as e:
        fails += 1; print(f"{i+1:2d}  FAIL  {time.time()-t0:4.1f}s  {type(e).__name__} (no answer inside {window}s)")
print(f"\n{len(ok_times)}/{n} answered inside {window}s ({100*len(ok_times)//n}%)")
if ok_times:
    ok_times.sort()
    print(f"median {statistics.median(ok_times):.1f}s, slowest {ok_times[-1]:.1f}s")
