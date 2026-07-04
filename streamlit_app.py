import streamlit as st
import pandas as pd
import json
from huggingface_hub import HfFileSystem
import config
from us_calendar import next_trading_day

st.set_page_config(page_title="OT-FM Engine", layout="wide")

st.markdown("""
<style>
.main-header { font-size:2.4rem; font-weight:700; color:#1a1a2e; margin-bottom:0.3rem; }
.sub-header  { font-size:1.1rem; color:#555; margin-bottom:1.5rem; }
.uni-title   { font-size:1.4rem; font-weight:600; margin-top:1rem; margin-bottom:0.8rem;
               padding-left:0.5rem; border-left:5px solid #533483; }
.etf-card    { background:linear-gradient(135deg,#1a1a2e 0%,#533483 100%); color:white;
               border-radius:14px; padding:1rem; margin:0.4rem; text-align:center;
               box-shadow:0 4px 6px rgba(0,0,0,0.2); }
.win-card    { background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%); color:white;
               border-radius:14px; padding:1rem; margin:0.4rem; text-align:center;
               box-shadow:0 4px 6px rgba(0,0,0,0.2); }
.etf-ticker  { font-size:1.3rem; font-weight:bold; }
.etf-score   { font-size:0.88rem; margin-top:0.25rem; opacity:0.9; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🌀 OT-FM Engine</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Tong et al. (2023) Flow Matching with Optimal Coupling · '
    'Exact 1D optimal transport pairing (sort-based, no Sinkhorn) · '
    'Straight-line ODE, analytical backprop · '
    'Multi-window cross-sectional z-score</div>',
    unsafe_allow_html=True)

st.sidebar.markdown("## OT-FM Engine")
st.sidebar.markdown(f"**Next Trading Day:** `{next_trading_day()}`")
st.sidebar.markdown(f"**Windows:** {config.WINDOWS}")
st.sidebar.markdown(
    f"**Network:** {config.N_HIDDEN} hidden layers x {config.HIDDEN_DIM} | "
    f"integration steps={config.N_INTEGRATION_STEPS}")
st.sidebar.markdown(
    f"**Training:** epochs={config.OTFM_EPOCHS} | lr={config.OTFM_LR} | "
    f"batch={config.OTFM_BATCH_SIZE}")
st.sidebar.markdown(
    f"**Weights:** Endpoint {config.WEIGHT_ENDPOINT:.0%} | "
    f"Drift {config.WEIGHT_DRIFT:.0%} | "
    f"Straightness {config.WEIGHT_STRAIGHTNESS:.0%}")

HF_TOKEN    = config.HF_TOKEN
OUTPUT_REPO = config.OUTPUT_REPO


@st.cache_data(ttl=3600)
def list_repo_files():
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        return [f["name"] for f in fs.ls(f"datasets/{OUTPUT_REPO}",
                                          detail=True, recursive=True)
                if f["type"] == "file"]
    except Exception as e:
        return [f"Error: {e}"]


def find_latest(files, prefix):
    matches = sorted([f for f in files if f.endswith(".json") and prefix in f],
                     reverse=True)
    return matches[0] if matches else None


@st.cache_data(ttl=3600)
def load_json(path):
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        with fs.open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}


files     = list_repo_files()
tab1_path = find_latest(files, "otfm_engine_2")
tab2_path = find_latest(files, "otfm_engine_windows_")

if not tab1_path:
    st.error("No results found. Run trainer.py first.")
    st.stop()

data1 = load_json(tab1_path)
if "error" in data1:
    st.error(f"Error loading data: {data1['error']}")
    st.stop()

data2      = load_json(tab2_path) if tab2_path else None
universes1 = data1["universes"]
universes2 = data2["universes"] if data2 and "error" not in data2 else None

st.sidebar.markdown(f"**Run date:** `{data1.get('run_date','?')}`")

tab1, tab2 = st.tabs(["🏆 Best Window per ETF", "🔍 Explore by Window"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("🏆 Top ETFs — Optimal Transport Flow Matching Signal")

    with st.expander("OT-FM Methodology", expanded=True):
        st.markdown("""
**Flow Matching** trains a vector field to transform a noise sample x0 into
a data sample x1 along a straight-line path:

```
x_t = (1-t)*x0 + t*x1,   u_t = x1 - x0
L   = E [ || v_theta(x_t, t, c) - u_t ||^2 ]
```

Sampling integrates the ODE `dx/dt = v_theta(x,t,c)` from x(0) to x(1) —
no reverse-time SDE, no denoising chain.

**Vanilla CFM pairs x0 and x1 arbitrarily.** OT-FM instead pairs them
**within each minibatch via optimal transport**, minimizing total squared
transport cost first — straighter, non-crossing paths, faster convergence,
fewer integration steps needed. This is the "frontier" improvement over
vanilla CFM.

**Exact 1D coupling (no Sinkhorn needed):** since the target here is a
scalar forward return, optimal transport under squared cost has a closed
form — sort both the noise batch and the data batch, pair by rank:

```
sort(x0)  paired with  sort(x1),  rank-for-rank
```

This is exactly optimal, not approximated. The conditioning state c
(lagged returns + macro context) travels with its original x1 through
the re-sort — only the noise-to-data assignment changes.

**Signal:**

```
score = 0.45 * endpoint + 0.35 * drift + 0.20 * straightness * sign(endpoint)
```

- `endpoint` — x(1) from integrating the ODE from x(0)=0: the generated forecast
- `drift` — initial velocity v_theta(0,0,c): direction & magnitude of departure
- `straightness` — 1/(1+std of velocity along the path): directly validates
  the OT-FM benefit — a well-trained OT-coupled flow should be nearly straight

**Distinct from CFM:** independent/arbitrary pairing vs. exact optimal pairing.
**Distinct from DDB:** DDB pins both x(0) and x(1) to real financial values;
here x(0) is noise and x(1) is generated by the flow, not analytically pinned.
        """)

    for universe_name, uni_data in universes1.items():
        top_etfs = uni_data.get("top_etfs", [])
        if not top_etfs:
            continue
        st.markdown(
            f'<div class="uni-title">{universe_name.replace("_"," ").title()}</div>',
            unsafe_allow_html=True)
        cols = st.columns(3)
        for idx, etf in enumerate(top_etfs):
            with cols[idx]:
                st.markdown(f"""
<div class="etf-card">
  <div class="etf-ticker">{etf['ticker']}</div>
  <div class="etf-score">OT-FM score = {etf['otfm_score']:.4f}</div>
  <div class="etf-score">best window = {etf.get('best_window','N/A')}d</div>
</div>
""", unsafe_allow_html=True)

        with st.expander(f"Full ranking — {universe_name}"):
            full = uni_data.get("full_scores", {})
            if full:
                rows = []
                for t, info in full.items():
                    score = info.get("score", info) if isinstance(info, dict) else info
                    win   = info.get("best_window", "N/A") if isinstance(info, dict) else "N/A"
                    rows.append({"ETF": t, "OT-FM Score": score, "Best Window (d)": win})
                df = pd.DataFrame(rows).sort_values("OT-FM Score", ascending=False)
                st.dataframe(df, use_container_width=True, hide_index=True)
        st.divider()

    st.caption(
        f"Run date: {data1.get('run_date','?')} · "
        "Tong et al. (2023) Minibatch Optimal Transport Flow Matching · "
        "Scores are cross-sectional z-scores.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("🔍 Explore OT-FM Rankings by Window")

    if not universes2:
        st.warning("Window-level detail not found. Re-run trainer.")
        st.stop()

    all_wins = set()
    for ud in universes2.values():
        all_wins.update(ud.get("windows", {}).keys())
    win_options = sorted([int(w) for w in all_wins])

    if not win_options:
        st.error("No window data available.")
        st.stop()

    default_idx  = win_options.index(252) if 252 in win_options else 0
    selected_win = st.selectbox(
        "Select lookback window",
        options=win_options,
        index=default_idx,
        format_func=lambda w: f"{w}d  (~{round(w/21)} months)",
    )
    win_key = str(selected_win)

    with st.expander("Window guidance", expanded=False):
        st.markdown("""
- **63d** — short training set; few OT-coupled minibatches; reactive, noisier
- **126d** — 6-month window; recommended minimum for a stable flow
- **252d** — 1-year window; most stable vector field; recommended primary signal
- **504d** — 2-year window; structural regime flow; slow-moving signal
        """)

    st.markdown(f"### OT-FM Rankings at **{selected_win}d** window")

    for universe_name in ["FI_COMMODITIES", "EQUITY_SECTORS", "COMBINED"]:
        label = {
            "FI_COMMODITIES": "🏦 FI & Commodities",
            "EQUITY_SECTORS": "📈 Equity Sectors",
            "COMBINED":       "🌐 Combined",
        }.get(universe_name, universe_name)

        st.markdown(f'<div class="uni-title">{label}</div>', unsafe_allow_html=True)

        uni_data = universes2.get(universe_name, {})
        win_data = uni_data.get("windows", {}).get(win_key)

        if not win_data:
            st.info(f"No data for {universe_name} at {selected_win}d.")
            st.divider()
            continue

        cols = st.columns(3)
        for idx, etf in enumerate(win_data.get("top_etfs", [])):
            with cols[idx]:
                st.markdown(f"""
<div class="win-card">
  <div class="etf-ticker">{etf['ticker']}</div>
  <div class="etf-score">OT-FM score = {etf['otfm_score']:.4f}</div>
  <div class="etf-score">window = {selected_win}d</div>
</div>
""", unsafe_allow_html=True)

        with st.expander(f"Full ranking — {label} @ {selected_win}d"):
            rows = win_data.get("full_ranking", [])
            if rows:
                df = pd.DataFrame(rows, columns=["ETF", "OT-FM Score"])
                df.insert(0, "Rank", range(1, len(df) + 1))
                st.dataframe(df, use_container_width=True, hide_index=True)

        st.divider()

    st.caption(f"Window: {selected_win}d · Run date: {data2.get('run_date','?')}")
