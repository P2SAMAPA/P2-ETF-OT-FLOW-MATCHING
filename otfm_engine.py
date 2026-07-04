"""
otfm_engine.py — Optimal Transport Flow Matching Engine
==========================================================

Theory
------
**Flow Matching (Lipman et al. 2023)**

Trains a vector field v_theta(x,t) to transform a noise sample x0 into a
data sample x1 along a straight-line probability path:

    x_t = (1-t)*x0 + t*x1,   t in [0,1]
    u_t = x1 - x0            (constant target velocity along the path)

    L = E [ || v_theta(x_t, t, c) - u_t ||^2 ]

Once trained, sampling is done by integrating the ODE dx/dt = v_theta(x,t,c)
from x(0)=x0 to x(1) — no reverse-time SDE, no thousands of denoising steps.

**Vanilla CFM vs. OT-FM**

Vanilla CFM (used elsewhere in this suite) pairs each noise sample x0 with
an independently / arbitrarily drawn data sample x1. This works, but the
resulting straight-line paths can cross each other a great deal, which
makes the marginal vector field the model has to learn curve around those
crossings — slower training, more integration steps needed at inference.

OT-FM (Tong et al. 2023, "Improving and Generalizing Flow-Based Generative
Models with Minibatch Optimal Transport") instead pairs x0 and x1 WITHIN
EACH MINIBATCH via optimal transport, minimizing total squared transport
cost before computing the flow matching loss. Straighter, non-crossing
paths result — faster convergence, fewer integration steps, more reliable
extrapolation. This is the "frontier" improvement over vanilla CFM.

**Exact 1D optimal transport via sorting**

The target variable here (a forward return) is SCALAR. Under squared-error
cost, the optimal transport plan between two 1D distributions has an exact
closed form: sort both samples and pair by rank (the classical monotone
rearrangement theorem — see e.g. Villani, "Optimal Transport: Old and
New"). No Sinkhorn iteration, no approximate solver, no extra dependency —
the coupling implemented below is exactly optimal, not approximated:

    sort(x0) paired with sort(x1), rank-for-rank

The conditioning context c_i "belongs" to its data point x1_i (it is the
market state that preceded that realized return) and travels with x1_i
through the re-sort — only the noise-to-data assignment is re-optimized,
never the (c_i, x1_i) relationship itself.

**Architecture**

A small tanh MLP v_theta([x, t, c]) -> velocity, built from scratch with
manual forward/backward and Adam — matching this suite's established
from-scratch modelling pattern (same complexity class as DDB's score
network, not the larger transformer used in the Decision Transformer).

**Score construction**

    score = 0.45*endpoint + 0.35*drift + 0.20*straightness*sign(endpoint)

| Component     | Meaning                                                              |
|----------------|-----------------------------------------------------------------------|
| endpoint       | x(1) from integrating the ODE from x(0)=0 — the generated forecast    |
| drift          | v_theta(0,0,c) — initial departure direction & magnitude              |
| straightness   | 1/(1+std of velocity along the path) — validates the OT-FM benefit    |

**Distinction from other generative engines in the suite**

| Engine            | x(0)        | x(1)                | Coupling                  |
|--------------------|-------------|----------------------|---------------------------|
| SCORE-DIFFUSION    | Noise       | Data                 | N/A (reverse SDE)         |
| CFM                | Noise       | Data                 | Independent / arbitrary   |
| DDB                | Today's ret.| Macro-implied target | Both endpoints pinned     |
| **OT-FM (this)**   | **Noise**   | **Data**             | **Exact optimal (1D)**    |

References
----------
- Lipman, Y. et al. (2023). Flow Matching for Generative Modeling. ICLR 2023.
- Tong, A. et al. (2023). Improving and Generalizing Flow-Based Generative
  Models with Minibatch Optimal Transport. TMLR 2024.
- Pooladian, A-A. et al. (2023). Multisample Flow Matching: Straightening
  Flows with Minibatch Couplings. ICML 2023.
- Villani, C. (2003). Topics in Optimal Transportation. AMS.
"""

import numpy as np
import pandas as pd
from typing import List

import config


# ── Basic differentiable layers (manual forward/backward) ─────────────────────

class Linear:
    def __init__(self, in_d: int, out_d: int, rng: np.random.Generator):
        scale = np.sqrt(2.0 / in_d)
        self.W = rng.normal(0, scale, (in_d, out_d))
        self.b = np.zeros(out_d)

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.X = X
        return X @ self.W + self.b

    def backward(self, dY: np.ndarray):
        X = self.X
        X2  = X.reshape(-1, X.shape[-1])
        dY2 = dY.reshape(-1, dY.shape[-1])
        dW  = X2.T @ dY2
        db  = dY2.sum(axis=0)
        dX  = dY @ self.W.T
        return dX, dW, db


class VectorFieldMLP:
    """v_theta([x, t, c]) -> scalar velocity. tanh MLP, manual backprop."""

    def __init__(self, cond_dim: int, rng: np.random.Generator):
        in_dim = 2 + cond_dim   # x, t, conditioning
        H = config.HIDDEN_DIM
        self.layers = [Linear(in_dim, H, rng)]
        for _ in range(config.N_HIDDEN - 1):
            self.layers.append(Linear(H, H, rng))
        self.out = Linear(H, 1, rng)

    def forward(self, x: np.ndarray, t: np.ndarray, c: np.ndarray) -> np.ndarray:
        """x:(B,) t:(B,) c:(B,cond_dim) -> velocity:(B,)"""
        inp = np.concatenate([x[:, None], t[:, None], c], axis=1)
        self._acts = [inp]
        h = inp
        for layer in self.layers:
            z = layer.forward(h)
            h = np.tanh(z)
            self._acts.append(h)
        v = self.out.forward(h)
        return v.squeeze(-1)

    def backward(self, dV: np.ndarray):
        dOut, dOutW, dOutb = self.out.backward(dV[:, None])
        grads = {"out": (dOutW, dOutb), "layers": []}

        dh = dOut
        for i in reversed(range(len(self.layers))):
            h = self._acts[i + 1]
            dz = dh * (1 - h ** 2)
            dh, dW, db = self.layers[i].backward(dz)
            grads["layers"].insert(0, (dW, db))
        return grads

    # ── Adam ────────────────────────────────────────────────────────────────

    def _param_list(self):
        params = [(self.out, "W"), (self.out, "b")]
        for layer in self.layers:
            params += [(layer, "W"), (layer, "b")]
        return params

    def init_adam(self):
        return [(np.zeros_like(getattr(o, a)), np.zeros_like(getattr(o, a)))
                for o, a in self._param_list()]

    def apply_adam(self, grads, state, step, lr,
                    b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        flat = [grads["out"][0], grads["out"][1]]
        for dW, db in grads["layers"]:
            flat += [dW, db]

        params = self._param_list()
        for i, ((obj, attr), grad) in enumerate(zip(params, flat)):
            m, v = state[i]
            m[:] = b1 * m + (1 - b1) * grad
            v[:] = b2 * v + (1 - b2) * grad ** 2
            mh = m / (1 - b1 ** step)
            vh = v / (1 - b2 ** step)
            update = lr * mh / (np.sqrt(vh) + eps)
            setattr(obj, attr, getattr(obj, attr) - update)


# ── Exact 1D optimal transport coupling ────────────────────────────────────────

def ot_couple_1d(x0: np.ndarray, x1: np.ndarray, c: np.ndarray):
    """
    Exact optimal transport coupling between two 1D samples under squared
    cost: sort both and pair by rank (monotone rearrangement theorem).
    Conditioning c travels with its original x1 through the re-sort.
    """
    order0 = np.argsort(x0)
    order1 = np.argsort(x1)
    x0_sorted = x0[order0]
    x1_sorted = x1[order1]
    c_sorted  = c[order1]
    return x0_sorted, x1_sorted, c_sorted


# ── Training ───────────────────────────────────────────────────────────────────

def _train_otfm(x1_data: np.ndarray, cond: np.ndarray, rng: np.random.Generator) -> VectorFieldMLP:
    """x1_data: (N,) realized forward returns. cond: (N, cond_dim) conditioning states."""
    N = len(x1_data)
    B = config.OTFM_BATCH_SIZE
    if N < B:
        raise ValueError("insufficient samples for OT-FM training")

    model = VectorFieldMLP(cond_dim=cond.shape[1], rng=rng)
    state = model.init_adam()
    step = 0

    for epoch in range(config.OTFM_EPOCHS):
        idx = rng.permutation(N)
        epoch_loss, n_b = 0.0, 0

        for i in range(0, N, B):
            bi = idx[i:i + B]
            if len(bi) < 4:
                continue

            x1_b = x1_data[bi]
            c_b  = cond[bi]
            x0_b = rng.normal(0, 1, size=len(bi))

            x0_c, x1_c, c_c = ot_couple_1d(x0_b, x1_b, c_b)

            t = rng.uniform(0, 1, size=len(bi))
            x_t = (1 - t) * x0_c + t * x1_c
            u_t = x1_c - x0_c

            pred = model.forward(x_t, t, c_c)
            resid = pred - u_t
            loss = float(np.mean(resid ** 2))

            grads = model.backward(2.0 * resid / resid.size)
            step += 1
            model.apply_adam(grads, state, step, lr=config.OTFM_LR)

            epoch_loss += loss
            n_b += 1

        if (epoch + 1) % 15 == 0:
            print(f"    epoch {epoch+1}/{config.OTFM_EPOCHS}  loss={epoch_loss/max(n_b,1):.6f}")

    return model


def _integrate_path(model: VectorFieldMLP, c: np.ndarray, n_steps: int):
    """Integrate dx/dt = v_theta(x,t,c) from x(0)=0 to x(1). Returns (x1, velocities)."""
    x = np.array([0.0])
    dt = 1.0 / n_steps
    velocities = []
    for k in range(n_steps):
        t = np.array([k * dt])
        v = model.forward(x, t, c[None, :])
        velocities.append(float(v[0]))
        x = x + v * dt
    return float(x[0]), np.array(velocities)


# ── Main scoring function ─────────────────────────────────────────────────────

def compute_otfm_scores(
    prices:    pd.DataFrame,
    macro_df:  pd.DataFrame,
    tickers:   List[str],
    window:    int,
) -> pd.Series:
    """
    Train an Optimal Transport Flow Matching vector field per ETF and
    extract a generative forecast signal. Returns cross-sectional z-scores.
    """
    avail = [t for t in tickers if t in prices.columns]
    if not avail:
        return pd.Series(dtype=float)

    L, H = config.N_LAGS, config.PRED_HORIZON
    min_rows = window + H + L + 5
    if len(prices) < min_rows:
        return pd.Series(dtype=float)

    common   = prices.index.intersection(macro_df.index) if not macro_df.empty else prices.index
    prices_a = prices.loc[common]
    macro_a  = macro_df.loc[common] if not macro_df.empty else pd.DataFrame(index=common)

    macro_vals = macro_a.values.astype(np.float64) if not macro_a.empty else np.zeros((len(common), 0))
    if macro_vals.shape[1] > 0:
        m_mu  = np.nanmean(macro_vals, axis=0, keepdims=True)
        m_std = np.nanstd(macro_vals,  axis=0, keepdims=True) + 1e-8
        macro_norm = np.nan_to_num((macro_vals - m_mu) / m_std, 0.0)
    else:
        macro_norm = np.zeros((len(common), 0))

    rng = np.random.default_rng(42)
    raw_scores = {}

    for ticker in avail:
        ps = prices_a[ticker].dropna()
        if len(ps) < min_rows:
            continue

        log_ret = np.log(ps / ps.shift(1)).dropna().values
        mac = macro_norm[-len(log_ret):]
        if len(mac) < len(log_ret):
            log_ret = log_ret[-len(mac):]

        T = len(log_ret)
        start = max(L, T - window - H)
        end = T - H
        if end - start < config.OTFM_BATCH_SIZE * 2:
            continue

        # ── Build (conditioning state, forward return) training pairs ─────────
        states = []
        for t in range(start, end):
            lag = log_ret[t - L:t]
            mu, sd = lag.mean(), lag.std() + 1e-8
            s = np.concatenate([(lag - mu) / sd, mac[t]])
            states.append(s)
        states = np.array(states)
        fwd = np.array([log_ret[t:t + H].mean() for t in range(start, end)])

        n = min(len(states), len(fwd))
        states, fwd = states[-n:], fwd[-n:]
        if n < config.OTFM_BATCH_SIZE * 2:
            continue

        # Normalize the forward-return target for stable training; rescale
        # the generated endpoint back to raw-return units afterward.
        fwd_mu, fwd_sd = fwd.mean(), fwd.std() + 1e-8
        fwd_norm = (fwd - fwd_mu) / fwd_sd

        print(f"    Training OT-FM for {ticker} (N={n}, cond_dim={states.shape[1]})")
        try:
            model = _train_otfm(fwd_norm, states, rng)
        except Exception as e:
            print(f"    Failed {ticker}: {e}")
            continue

        # ── Inference: integrate the ODE for today's conditioning state ───────
        c_today = states[-1]
        x1_gen, velocities = _integrate_path(model, c_today, config.N_INTEGRATION_STEPS)

        endpoint = float(x1_gen * fwd_sd + fwd_mu)          # back to raw-return units
        drift    = float(velocities[0] * fwd_sd)             # initial departure, same scale
        straightness = float(1.0 / (1.0 + np.std(velocities)))

        print(f"    {ticker}: endpoint={endpoint:.5f}  drift={drift:.5f}  "
              f"straightness={straightness:.4f}")

        composite = (
            config.WEIGHT_ENDPOINT     * endpoint
            + config.WEIGHT_DRIFT        * drift
            + config.WEIGHT_STRAIGHTNESS * straightness * np.sign(endpoint if endpoint != 0 else 1.0)
        )
        raw_scores[ticker] = composite

    if not raw_scores:
        return pd.Series(dtype=float)

    scores = pd.Series(raw_scores)
    mu, std = scores.mean(), scores.std()
    if std < 1e-10:
        return pd.Series(0.0, index=scores.index)
    return (scores - mu) / std
