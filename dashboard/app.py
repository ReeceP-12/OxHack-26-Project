"""Live dashboard for the voice memory proxy.

Run (from the repo root, venv active, proxy running on port 8000):
    streamlit run dashboard/app.py

No proxy yet? Tick "Use fake data" in the sidebar. "Simulate a live conversation" adds one fake turn
per refresh so you can rehearse the demo.
"""
import time

import pandas as pd
import requests
import streamlit as st

from dash_data import (TIER_COLOURS, TIER_MEANING, TIERS, legend_html, make_fake_turns,
                  tier_counts, tier_map_html, pin_counts)

st.set_page_config(page_title="Voice memory dashboard", layout="wide")

# ---- sidebar --------------------------------------------------------------------------------
st.sidebar.header("Settings")
base_url = st.sidebar.text_input("Proxy URL", "http://localhost:8000")
refresh = st.sidebar.slider("Refresh every (seconds)", 1, 10, 2)
auto = st.sidebar.checkbox("Auto refresh", value=True)
use_fake = st.sidebar.checkbox("Use fake data", value=False)
simulate = st.sidebar.checkbox("Simulate a live conversation (fake data only)", value=True)

st.title("What the memory kept, shrank and forgot")
st.caption("Each turn the voice agent resends the whole conversation. The manager decides what to keep.")

@st.cache_data
def get_fake() -> list:
    return make_fake_turns(50)


FAKE = get_fake()


def fetch_state() -> dict:
    """Return the /debug/state payload, or fake data. Raises on a real network problem."""
    if use_fake:
        n = st.session_state.get("fake_n", 8)
        if simulate:
            n = n + 1 if n < len(FAKE) else 8  # add a turn per refresh, loop at the end
        st.session_state["fake_n"] = n
        return {"manager_on": True, "turns": FAKE[:n]}
    r = requests.get(base_url.rstrip("/") + "/debug/state", timeout=2)
    r.raise_for_status()
    return r.json()


def build_frame(turns: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(turns)
    for col in ("tokens_in", "tokens_sent", "ttft_ms", "manager_on"):
        if col not in df.columns:
            df[col] = None
    if "manager_ms" not in df.columns:
        df["manager_ms"] = None
    for col in ("tokens_in", "tokens_sent", "ttft_ms", "manager_ms"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.set_index("turn")
    on = df["manager_on"].map(lambda v: True if v is None or v != v else bool(v))
    out = pd.DataFrame(index=df.index)
    out["Naive (everything resent)"] = df["tokens_in"]
    out["Ours (after memory manager)"] = df["tokens_sent"]
    out["TTFT, manager ON"] = df["ttft_ms"].where(on)
    out["TTFT, manager OFF (naive)"] = df["ttft_ms"].where(~on)
    out["Memory manager cost (ms)"] = df["manager_ms"].where(on)
    # manager_info is null in naive mode, so a missing value just means "no budget info this turn"
    info = pd.DataFrame([(t.get("manager_info") or {}) for t in turns], index=df.index)
    for key, label in (("base_budget", "Base budget"), ("effective_budget", "Effective budget")):
        out[label] = pd.to_numeric(info[key], errors="coerce") if key in info.columns else float("nan")
    return out


def show_budget_chart(budget: pd.DataFrame) -> None:
    """Step line with a y axis that does not start at zero, so small steps above the base are visible."""
    try:
        import altair as alt
        long = (budget.reset_index().melt("turn", var_name="Line", value_name="Tokens").dropna())
        chart = (alt.Chart(long).mark_line(interpolate="step-after")
                 .encode(x=alt.X("turn:Q", title="Turn"),
                         y=alt.Y("Tokens:Q", scale=alt.Scale(zero=False)),
                         color=alt.Color("Line:N"),
                         tooltip=["turn", "Line", "Tokens"]))
        st.altair_chart(chart)
    except Exception:
        st.line_chart(budget)


def show(state: dict, follow: bool, pick_turn: int) -> None:
    turns = state.get("turns", [])
    if not turns:
        st.info("No turns yet. Talk to the agent (or send a test request to the proxy) and they appear here.")
        return
    frame = build_frame(turns)
    last = turns[-1]
    t_in, t_sent = last.get("tokens_in") or 0, last.get("tokens_sent") or 0
    saved = (1 - t_sent / t_in) * 100 if t_in else 0.0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Turns so far", len(turns))
    c2.metric("Tokens, naive", f"{t_in:,}")
    c3.metric("Tokens, ours", f"{t_sent:,}", f"-{saved:.0f}%" if saved > 0 else None, delta_color="inverse")
    mode_on = state.get("manager_on", last.get("manager_on", True))
    c4.metric("Memory manager", "ON" if mode_on else "OFF (naive)")
    cost = frame["Memory manager cost (ms)"].dropna()
    c5.metric("Manager cost (ms)", f"{cost.iloc[-1]:.0f}" if len(cost) else "n/a",
              f"median {cost.median():.0f}" if len(cost) else None, delta_color="off")

    left, right = st.columns(2)
    with left:
        st.subheader("Tokens sent per turn")
        st.line_chart(frame[["Naive (everything resent)", "Ours (after memory manager)"]])
        budget = frame[["Base budget", "Effective budget"]].dropna(how="all")
        if not budget.empty:
            st.subheader("Token budget per turn")
            st.caption("Steps up above the base when the manager pins a stated fact.")
            show_budget_chart(budget)
    with right:
        st.subheader("Time to first token (ms)")
        ttft = frame[["TTFT, manager ON", "TTFT, manager OFF (naive)"]].dropna(how="all")
        if ttft.empty:
            st.caption("No latency recorded yet.")
        else:
            st.line_chart(ttft)
        if len(cost):
            st.subheader("What the memory manager costs per turn (ms)")
            st.line_chart(frame[["Memory manager cost (ms)"]])

    st.subheader("Tier map")
    with_tiers = [t for t in turns if t.get("tiers")]
    if not with_tiers:
        st.info("The proxy is not returning tier info yet (the memory manager has not been plugged in). "
                "Tick 'Use fake data' in the sidebar to preview this section.")
        return
    numbers = [t["turn"] for t in with_tiers]
    pick = numbers[-1] if follow or pick_turn not in numbers else pick_turn
    chosen = next(t for t in with_tiers if t["turn"] == pick)
    counts = tier_counts(chosen["tiers"])
    n_pinned, n_critical = pin_counts(chosen["tiers"])
    cols = st.columns(len(TIERS) + 2)
    for col, tier in zip(cols, TIERS):
        col.metric(TIER_MEANING[tier], counts.get(tier, 0))
    cols[len(TIERS)].metric("Pinned facts", n_pinned)
    cols[len(TIERS) + 1].metric("Critical facts", n_critical)
    st.markdown(legend_html(), unsafe_allow_html=True)
    st.markdown(tier_map_html(chosen["tiers"]), unsafe_allow_html=True)
    st.caption(f"Turn {pick}: one square per message in the original conversation, oldest first. "
               "Hover a square for its score and token count.")


st.sidebar.divider()
follow = st.sidebar.checkbox("Tier map follows the latest turn", value=True)
pick_turn = st.sidebar.number_input("Tier map turn (when not following)", min_value=1, value=1, step=1)

placeholder = st.empty()


def render() -> None:
    with placeholder.container():
        try:
            state = fetch_state()
        except Exception as e:  # proxy down, wrong URL, etc.
            st.error(f"Cannot reach the proxy at {base_url}: {e}")
            st.caption("Start it with: uvicorn main:app --port 8000 (add 'set MOCK_LLM=1' first if you have no key). "
                       "Or tick 'Use fake data' in the sidebar.")
            return
        show(state, follow, int(pick_turn))


# Redraw in place inside one long run. (Re-running the whole script every few seconds makes
# Streamlit grey the page out while it waits, which flickers.) Changing a widget restarts the loop.
render()
while auto:
    time.sleep(refresh)
    render()
