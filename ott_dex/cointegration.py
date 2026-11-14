"""
cointegration.py — Screening and spread econometrics.

* Pearson correlation (paper, Section 3.1)
* Rolling pairwise Engle–Granger pass rate (paper, Table 2) — kept as baseline
* EXTENSION: multivariate Johansen trace test on the whole bucket
* EXTENSION: Ornstein–Uhlenbeck estimation of each spread (θ, μ, σ, half-life),
  which replaces the paper's ad-hoc "average reversion time" mr.
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

Pair = Tuple[str, str]


def all_pairs(symbols) -> List[Pair]:
    return list(combinations(list(symbols), 2))


def spread(logp: pd.DataFrame, pair: Pair) -> pd.Series:
    """s_ij,t = log P_i,t − log P_j,t   (paper's spread; β = 1, see notebook)."""
    i, j = pair
    return logp[i] - logp[j]


# --------------------------------------------------------------------------
# Correlation and Engle–Granger (baseline, as in the paper)
# --------------------------------------------------------------------------
def rolling_screen(logp: pd.DataFrame, window: int = 42, pvalue: float = 0.05) -> pd.DataFrame:
    """
    Split the sample into non-overlapping windows (default 1 week of 4h bars).
    For each pair report the average Pearson correlation of log prices and the
    share of windows in which the Engle–Granger test rejects "no cointegration".
    """
    out = []
    for pair in all_pairs(logp.columns):
        cors, passes = [], []
        for s in range(0, len(logp) - window + 1, window):
            w = logp.iloc[s:s + window]
            x, y = w[pair[0]].values, w[pair[1]].values
            if np.std(x) == 0 or np.std(y) == 0:
                continue
            cors.append(np.corrcoef(x, y)[0, 1])
            try:
                passes.append(coint(x, y, trend="c")[1] < pvalue)
            except Exception:
                passes.append(False)
        out.append({"pair": f"{pair[0]}:{pair[1]}", "corr": np.mean(cors) if cors else np.nan,
                    "eg_pass_rate": np.mean(passes) if passes else np.nan, "n_windows": len(passes)})
    return pd.DataFrame(out).set_index("pair")


# --------------------------------------------------------------------------
# EXTENSION 1a — Johansen multivariate cointegration
# --------------------------------------------------------------------------
def johansen(logp: pd.DataFrame, det_order: int = 0, k_ar_diff: int = 1) -> Dict:
    """
    Johansen trace test on the n log-price series of the bucket.
    rank r = number of independent cointegrating relations. With n assets that
    all track the same ETH price we expect r = n − 1 (one common stochastic trend).
    Returns the table of trace statistics, the selected rank and the
    cointegrating vectors normalised on the first asset.

    Note: with near-identical series, the last trace test (H0: r ≤ n−1) over-rejects
    in finite samples (≈ 20–40 % in our simulations instead of 5 %), so r = n is
    sometimes reported. r = n would mean ETH itself is stationary, which is not
    plausible; we read r ∈ {n−1, n} as "all spreads stationary".
    """
    res = coint_johansen(logp.values, det_order, k_ar_diff)
    n = logp.shape[1]
    tab = pd.DataFrame({"H0: rank <=": range(n), "trace_stat": res.lr1,
                        "crit_90": res.cvt[:, 0], "crit_95": res.cvt[:, 1], "crit_99": res.cvt[:, 2]})
    tab["reject_95"] = tab["trace_stat"] > tab["crit_95"]
    rank = int(tab["reject_95"].cumprod().sum())
    vecs = pd.DataFrame(np.real(res.evec[:, :max(rank, 1)]), index=logp.columns)
    vecs = vecs / vecs.iloc[0]
    return {"table": tab, "rank": rank, "vectors": vecs, "eigenvalues": res.eig}


# --------------------------------------------------------------------------
# EXTENSION 1b — Ornstein–Uhlenbeck fit of a spread
# --------------------------------------------------------------------------
def fit_ou(s: pd.Series, dt_hours: float = 4.0) -> Dict[str, float]:
    """
    ds_t = θ (μ − s_t) dt + σ dW_t.  Exact discretisation on a grid of step Δ:
        s_{t+1} = a + b s_t + ε,   b = e^{−θΔ},  a = μ(1 − b),
        Var(ε) = σ² (1 − b²) / (2θ).
    Estimated by OLS of s_{t+1} on s_t.
    Returns θ (per hour), μ, σ_eq (stationary st.dev.), half-life (hours) and b.
    """
    x = s.values[:-1]
    y = s.values[1:]
    X = np.column_stack([np.ones_like(x), x])
    (a, b), *_ = np.linalg.lstsq(X, y, rcond=None)
    eps = y - (a + b * x)
    b = float(np.clip(b, 1e-6, 0.999999))
    theta = -np.log(b) / dt_hours
    mu = a / (1 - b)
    sig_eps = eps.std(ddof=2)
    sigma_eq = sig_eps / np.sqrt(1 - b * b)
    return {"theta": theta, "mu": mu, "sigma_eq": sigma_eq,
            "half_life_h": np.log(2) / theta, "b": b, "sigma_eps": sig_eps}


def fit_ou_all(logp: pd.DataFrame, dt_hours: float = 4.0) -> pd.DataFrame:
    rows = {}
    for p in all_pairs(logp.columns):
        rows[f"{p[0]}:{p[1]}"] = fit_ou(spread(logp, p), dt_hours)
    return pd.DataFrame(rows).T


def screen_assets(screen_tab: pd.DataFrame, min_corr: float, min_eg_pass: float) -> List[Pair]:
    """Keep the pairs that pass both the correlation and Engle–Granger filters."""
    keep = screen_tab[(screen_tab["corr"] >= min_corr) & (screen_tab["eg_pass_rate"] >= min_eg_pass)]
    return [tuple(p.split(":")) for p in keep.index]
