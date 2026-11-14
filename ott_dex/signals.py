"""
signals.py — Turning a spread into a z-score.

* static   : paper's rule. z = (s − mean_formation) / std_formation, fixed bands.
* ewma     : EXTENSION. Exponentially weighted mean / st.dev., updated each bar.
* kalman   : EXTENSION. State-space model of the spread
             s_t = m_t + x_t                              (observation)
             m_t = m_{t−1} + η_t,   η ~ N(0, σ²_m)        (slow fair value, random walk)
             x_t = φ x_{t−1} + ν_t, ν ~ N(0, σ²_x)        (mean-reverting deviation = discrete OU)
             Parameters (σ²_m, σ²_x, φ) are fitted by maximum likelihood on the
             formation window only, then the Kalman filter is run causally.
             Dynamic band: centre m̂_{t|t}, half-width per unit z
                 std_t = sqrt( Var_stat(x) + P^{m}_{t|t} )
             so the band widens when the filter is unsure about the fair value
             (e.g. right after a de-peg) and tightens when it is confident.

All z-scores at time t use information up to t only (no look-ahead).
Each function returns a DataFrame with columns: z, mean, std (band half-width
per 1 z), so that the trading band in spread units is mean ± z_open·std.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.structural import UnobservedComponents


def static_z(s: pd.Series, formation_end) -> pd.DataFrame:
    f = s.loc[:formation_end]
    m, sd = f.mean(), f.std()
    return pd.DataFrame({"z": (s - m) / sd, "mean": m, "std": sd}, index=s.index)


def ewma_z(s: pd.Series, halflife: int = 18) -> pd.DataFrame:
    m = s.ewm(halflife=halflife, min_periods=halflife).mean().shift(1)
    sd = s.ewm(halflife=halflife, min_periods=halflife).std().shift(1)
    return pd.DataFrame({"z": (s - m) / sd, "mean": m, "std": sd}, index=s.index)


SCALE = 1e4  # work in basis points: helps the MLE optimiser


def fit_state_space(s_form: pd.Series, max_level_ratio: float = 0.02) -> dict:
    """
    MLE of  s = m (random walk) + x (AR(1))  on formation data.
    Guard: the fair value must move slowly relative to the deviation, otherwise
    the filter "explains away" every deviation and no signal is left; we cap
    σ²_m ≤ max_level_ratio · Var_stat(x).
    """
    y = s_form.values * SCALE
    mod = UnobservedComponents(y, level="rwalk", autoregressive=1)
    res = mod.fit(disp=False, maxiter=200)
    par = dict(zip(mod.param_names, res.params))
    phi = float(np.clip(par["ar.L1"], 0.0, 0.995))
    s2x = float(max(par["sigma2.ar"], 1e-8))
    var_x = s2x / (1 - phi**2)
    s2m = float(min(max(par["sigma2.level"], 1e-10), max_level_ratio * var_x))
    return {"phi": phi, "sigma2_x": s2x, "sigma2_m": s2m, "var_x": var_x,
            "init_level": float(np.mean(y[:20]))}


def kalman_z(s: pd.Series, kp: dict) -> pd.DataFrame:
    """Causal 2-state Kalman filter with fixed parameters `kp` (bps units)."""
    y = s.values * SCALE
    n = len(y)
    T = np.array([[1.0, 0.0], [0.0, kp["phi"]]])
    Qm = np.diag([kp["sigma2_m"], kp["sigma2_x"]])
    Z = np.array([1.0, 1.0])
    a = np.array([kp["init_level"], 0.0])
    P = np.diag([kp["var_x"], kp["var_x"]])
    z, mean, std = np.empty(n), np.empty(n), np.empty(n)
    for t in range(n):
        a = T @ a                                     # predict
        P = T @ P @ T.T + Qm
        F = Z @ P @ Z
        K = P @ Z / F
        a = a + K * (y[t] - Z @ a)                    # update with s_t
        P = P - np.outer(K, Z @ P)
        mean[t] = a[0]
        std[t] = np.sqrt(kp["var_x"] + P[0, 0])
        z[t] = (y[t] - a[0]) / std[t]
    z[:10] = 0.0                                      # burn-in
    return pd.DataFrame({"z": z, "mean": mean / SCALE, "std": std / SCALE}, index=s.index)


def make_z(s: pd.Series, method: str, formation_end, ewma_halflife: int = 18, kalman_params=None):
    if method == "static":
        return static_z(s, formation_end)
    if method == "ewma":
        return ewma_z(s, ewma_halflife)
    if method == "kalman":
        kp = kalman_params or fit_state_space(s.loc[:formation_end])
        return kalman_z(s, kp)
    raise ValueError(method)
