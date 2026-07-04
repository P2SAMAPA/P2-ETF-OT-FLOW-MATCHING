# 🌀 P2-ETF-OT-FLOW-MATCHING

**Optimal Transport Flow Matching Engine — Tong et al. (2023)**

Part of the **P2Quant Engine Suite** · [P2SAMAPA](https://github.com/P2SAMAPA)

---

## What This Engine Does

This engine trains a **flow matching** vector field that transforms noise
into a forecast distribution of forward returns, exactly as CFM does
elsewhere in this suite — but with one key difference: instead of pairing
noise samples with data samples arbitrarily, it pairs them **via optimal
transport within each training minibatch**. This straightens the learned
paths, speeds convergence, and reduces the number of ODE integration steps
needed at inference — the central "frontier" improvement OT-FM contributes
over vanilla flow matching.

---

## Theory

### Flow Matching

```
x_t = (1-t)*x0 + t*x1,   t in [0,1]
u_t = x1 - x0             (constant target velocity along the path)
L   = E [ || v_theta(x_t, t, c) - u_t ||^2 ]
```

Sampling integrates `dx/dt = v_theta(x,t,c)` from x(0) to x(1) — no
reverse-time SDE, no long denoising chain.

### Vanilla CFM vs. OT-FM

| | Vanilla CFM (elsewhere in this suite) | **OT-FM (this engine)** |
|---|---|---|
| x0 ↔ x1 pairing | Independent / arbitrary | **Optimal transport, minimized cost** |
| Path crossings | Many | **Minimized** |
| Convergence | Slower | **Faster** |
| Integration steps needed | More | **Fewer** |

### Exact 1D Optimal Transport

The target here (forward return) is a **scalar**, so optimal transport
under squared-error cost has an exact closed form — the classical monotone
rearrangement theorem:

```
sort(x0)  paired with  sort(x1),  rank-for-rank
```

No Sinkhorn iteration, no approximate solver, no extra dependency. The
coupling implemented here is exactly optimal, not approximated. The
conditioning state c (lagged returns + macro context) travels with its
original x1 through the re-sort — only the noise-to-data assignment is
re-optimized.

### Architecture

A small tanh MLP `v_theta([x, t, c]) -> velocity`, built from scratch with
manual forward/backward and Adam — the same complexity class as DDB's
score network in this suite, not the larger transformer used by the
Decision Transformer engine.

### Score Construction

```
score = 0.45 * endpoint + 0.35 * drift + 0.20 * straightness * sign(endpoint)
```

| Component | Meaning |
|-----------|---------|
| endpoint | x(1) from integrating the ODE from x(0)=0 — the generated forecast itself |
| drift | v_theta(0,0,c) — initial departure direction & magnitude |
| straightness | 1/(1+std of velocity along the path) — directly validates the OT-FM benefit: a well-trained OT-coupled flow should be nearly straight |

---

## Distinction from Other Generative Engines

| Engine | x(0) | x(1) | Coupling |
|--------|------|------|----------|
| SCORE-DIFFUSION | Noise | Data | N/A (reverse SDE) |
| CFM | Noise | Data | Independent / arbitrary |
| DDB | Today's return | Macro-implied target | Both endpoints pinned |
| **OT-FM (this engine)** | **Noise** | **Data** | **Exact optimal (1D)** |

The key distinction from CFM: identical straight-line path family, but the
noise-to-data assignment is optimized rather than arbitrary, which is what
makes the learned paths straighter and the model faster to train and to
sample from.

---

## Universes & Windows

| Universe | Tickers |
|---|---|
| FI_COMMODITIES | TLT, VCIT, LQD, HYG, VNQ, GLD, SLV |
| EQUITY_SECTORS | SPY, QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, GDX, XME, IWF, XSD, XBI, IWM, IWD, IWO, XLB, XLRE |
| COMBINED | All of the above |

**Windows:** `63d · 126d · 252d · 504d`

---

## Repository Structure

```
P2-ETF-OT-FLOW-MATCHING/
├── config.py          # Universes, OT-FM hyperparameters, score weights
├── data_manager.py    # HuggingFace loader
├── otfm_engine.py      # Core: vector field MLP, exact 1D OT coupling, analytical backprop
├── trainer.py          # Orchestrator
├── push_results.py     # HfApi.upload_file wrapper
├── streamlit_app.py     # Two-tab Streamlit dashboard
├── us_calendar.py      # US trading calendar helper
├── requirements.txt
└── .github/
    └── workflows/
        └── daily.yml   # Single job
```

---

## Setup

```bash
git clone https://github.com/P2SAMAPA/P2-ETF-OT-FLOW-MATCHING
cd P2-ETF-OT-FLOW-MATCHING
pip install -r requirements.txt

export HF_TOKEN=hf_...
python trainer.py
streamlit run streamlit_app.py
```

**Required GitHub secret:** `HF_TOKEN`

**Required HuggingFace dataset repo:** `P2SAMAPA/p2-etf-otfm-results`

---

## References

- Lipman, Y. et al. (2023). Flow Matching for Generative Modeling. ICLR 2023.
- Tong, A. et al. (2023). Improving and Generalizing Flow-Based Generative
  Models with Minibatch Optimal Transport. TMLR 2024.
- Pooladian, A-A. et al. (2023). Multisample Flow Matching: Straightening
  Flows with Minibatch Couplings. ICML 2023.
- Villani, C. (2003). Topics in Optimal Transportation. AMS.
