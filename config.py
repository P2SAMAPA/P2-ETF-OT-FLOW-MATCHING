import os

HF_TOKEN    = os.environ.get("HF_TOKEN", "")
DATA_REPO   = "P2SAMAPA/fi-etf-macro-signal-master-data"
OUTPUT_REPO = "P2SAMAPA/p2-etf-otfm-results"

UNIVERSES = {
    "FI_COMMODITIES": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"],
    "EQUITY_SECTORS": [
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI",
        "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
    "COMBINED": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI",
        "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
}

MACRO_COLS_CORE     = ["VIX", "DXY", "T10Y2Y"]
MACRO_COLS_EXTENDED = ["IG_SPREAD", "HY_SPREAD"]

# ── Rolling windows (trading days) ────────────────────────────────────────────
WINDOWS = [63, 126, 252, 504]

# ── OT-FM hyperparameters ─────────────────────────────────────────────────────
# Lipman et al. (2023) "Flow Matching for Generative Modeling" trains a vector
# field v_theta(x,t) to transform a noise sample x0 into a data sample x1 along
# a straight-line path x_t = (1-t)*x0 + t*x1, with constant target velocity
# u_t = x1 - x0. Standard (vanilla) CFM pairs each x0 with an independently /
# arbitrarily drawn x1 — CFM elsewhere in this suite does exactly this.
#
# OT-FM (Tong et al. 2023, "Improving and Generalizing Flow-Based Generative
# Models with Minibatch Optimal Transport") instead pairs x0 and x1 within
# each minibatch via OPTIMAL TRANSPORT, minimizing total squared transport
# cost before computing the flow matching loss. This reduces path crossings,
# yields straighter learned trajectories, and needs fewer ODE integration
# steps at inference — the "frontier" improvement over vanilla CFM.
#
# Because the target variable here (a forward return) is SCALAR, optimal
# transport under squared-error cost has an exact closed form: sort both the
# noise minibatch and the data minibatch and pair by rank (the classical 1D
# optimal transport / monotone rearrangement result). No Sinkhorn, no
# approximate solver needed — the coupling used here is exactly optimal,
# not approximated.

N_LAGS       = 10     # lagged return features included in the conditioning state c
HIDDEN_DIM   = 32
N_HIDDEN     = 2       # number of hidden layers in the from-scratch tanh MLP

OTFM_BATCH_SIZE = 32    # minibatch size for the OT coupling + training step
OTFM_EPOCHS     = 60
OTFM_LR         = 3e-3

N_INTEGRATION_STEPS = 8   # Euler steps used to integrate the ODE at inference
PRED_HORIZON        = 21  # forward return horizon defining the data distribution x1

# ── Score construction ────────────────────────────────────────────────────────
# endpoint     : x(1) obtained by integrating dx/dt = v_theta(x,t,c) from
#                x(0)=0 to t=1 — the generated forward-return forecast itself
# drift        : initial velocity v_theta(0, 0, c) — direction & magnitude of
#                the flow's departure, before any integration drift accumulates
# straightness : 1 / (1 + std of velocity along the integrated path) — how
#                constant the learned velocity is along the trajectory for
#                today's conditioning; OT coupling is specifically what makes
#                this high, so it directly validates the OT-FM benefit itself

WEIGHT_ENDPOINT     = 0.45
WEIGHT_DRIFT         = 0.35
WEIGHT_STRAIGHTNESS  = 0.20

TOP_N = 3
