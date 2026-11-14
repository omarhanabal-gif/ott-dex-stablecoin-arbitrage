"""
optimizer.py — Principal allocation across simultaneous spread signals.

Decision variable: w_n ∈ [0, w_max] = fraction of portfolio NAV committed to
pair n (the direction d_n ∈ {+1, −1} is given by the sign of the signal, so the
long/short columns of the paper's W matrix collapse into one non-negative
number per pair; market neutrality in ETH units is enforced by execution:
the ETH bought in one pool is sold in the other within the same bar).

Paper (mean–variance, eq. 6):
    max  Σ_n w_n EP_n  −  λ · w' Σ w
Extension (CVaR, Rockafellar & Uryasev 2000):
    max  Σ_n w_n EP_n  −  λ · CVaR_α(L(w))
    CVaR_α = min_ζ  ζ + 1/((1−α)K) Σ_k max(L_k(w) − ζ, 0)
    L_k(w) = −Σ_n w_n r_{k,n}  (loss in scenario k, from historical h-bar
                                spread moves in the formation window)

Both include
    − Σ_n κ_n w_n²     (convex price impact: slippage grows with size / depth)
and the constraints
    0 ≤ w_n ≤ w_max
    Σ_{n spends c} w_n ≤ avail_c      (paper's "trading weights" constraint, eq. 4)
    Σ_n w_n ≤ 1                       (no leverage)

EP_n (EXTENSION of the paper's mr × r): the OU expected reversion over the
horizon h, net of round-trip fees:
    EP_n = |s_n − m_n| · (1 − e^{−θ_n h}) − c_n
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cvxpy as cp
import numpy as np
import pandas as pd


@dataclass
class Candidate:
    pair: tuple            # (i, j)
    direction: int         # +1 : spread too high -> sell ETH for i, buy ETH with j
    spend: str             # stablecoin spent at opening
    ep: float              # expected net return per unit notional
    kappa: float           # quadratic impact coefficient (in weight units)
    scen: pd.Series        # dated scenario P&L per unit notional


_CACHE: dict = {}


def _build(n: int, groups: tuple, risk: str, K: int):
    """
    Compile the problem ONCE per shape (n candidates, which candidates share a
    spending currency, risk type, K scenarios) with cvxpy Parameters (DPP);
    later calls only update parameter values -> ~10x faster than rebuilding.
    """
    w = cp.Variable(n, nonneg=True)
    ep, kap = cp.Parameter(n), cp.Parameter(n, nonneg=True)
    lam = cp.Parameter(nonneg=True)
    R = cp.Parameter((K, n))
    avail = cp.Parameter(len(groups), nonneg=True)
    obj = ep @ w - cp.sum(cp.multiply(kap, cp.square(w)))
    cons = [w <= cp.Parameter(name="wmax", nonneg=True, value=0.5), cp.sum(w) <= 1]
    wmax = cons[0].args[1]
    for g, idx in enumerate(groups):
        cons.append(cp.sum(w[list(idx)]) <= avail[g])
    if risk == "variance":                       # λ w'Σw = ||R w||² with R = sqrt(λ/(K−1)) R_c
        obj = obj - cp.sum_squares(R @ w)        # R holds sqrt(λ/(K−1)) · centred scenarios
        alpha = None
    else:                                        # Rockafellar–Uryasev CVaR
        alpha = cp.Parameter(nonneg=True)        # holds λ / ((1−α) K)  (one parameter keeps DPP)
        zeta, u = cp.Variable(), cp.Variable(K, nonneg=True)
        cons.append(u >= -R @ w - zeta)
        obj = obj - lam * zeta - alpha * cp.sum(u)
    prob = cp.Problem(cp.Maximize(obj), cons)
    assert prob.is_dpp(), "optimizer problem must be DPP to be cached"
    return prob, dict(w=w, ep=ep, kap=kap, lam=lam, R=R, avail=avail, alpha=alpha, wmax=wmax)


def cvar_loss(r: np.ndarray, alpha: float) -> float:
    """Sample CVaR_α of the loss −r, exactly as the Rockafellar–Uryasev LP defines it."""
    L = -np.asarray(r)
    K = len(L)
    excess = np.maximum(L[None, :] - L[:, None], 0).sum(axis=1)     # for each candidate ζ = L_j
    return float(np.min(L + excess / ((1 - alpha) * K)))


def _solve_one(c: Candidate, r: np.ndarray, avail: dict, risk: str, lam: float,
               alpha: float, w_max: float) -> np.ndarray:
    """
    Closed form for a single signal (the most frequent case).
    Risk terms are homogeneous in w ≥ 0:  CVaR(w·r) = w·CVaR(r),  Var(w·r) = w²·Var(r), so
        CVaR    : max_w  w (EP − λ·CVaR(r)) − κ w²      →  w* = (EP − λ·CVaR) / (2κ)
        variance: max_w  w EP − (κ + λ·Var(r)) w²       →  w* = EP / (2(κ + λ·Var))
    then clipped to the box/budget constraints (identical to the cvxpy solution).
    """
    cap = min(w_max, 1.0, max(avail.get(c.spend, 0.0), 0.0))
    if risk == "cvar":
        slope, curv = c.ep - max(lam, 0) * cvar_loss(r, alpha), max(c.kappa, 0.0)
    else:
        slope, curv = c.ep, max(c.kappa, 0.0) + max(lam, 0) * float(np.var(r, ddof=1))
    if slope <= 0:
        w = 0.0
    elif curv <= 0:
        w = cap
    else:
        w = min(slope / (2 * curv), cap)
    return np.array([w])


def solve(cands: List[Candidate], avail: dict, risk: str = "cvar", lam: float = 1.0,
          alpha: float = 0.95, w_max: float = 0.5, max_scen: int = 300) -> np.ndarray:
    n = len(cands)
    if n == 0:
        return np.zeros(0)
    # joint scenarios: align every pair's scenario series on common dates (K × n)
    R = pd.concat([pd.Series(c.scen).reset_index(drop=True) if not isinstance(c.scen, pd.Series)
                   else c.scen for c in cands], axis=1, join="inner").values
    # Overlapping h-bar scenarios on 15-min data are highly redundant: an evenly
    # spaced subsample of fixed size keeps the tail shape and the problem shape.
    R = R[np.linspace(0, R.shape[0] - 1, max_scen).astype(int)]
    K = R.shape[0]
    if risk not in ("variance", "cvar"):
        raise ValueError(risk)
    if n == 1:
        return _solve_one(cands[0], R[:, 0], avail, risk, lam, alpha, w_max)
    spends = sorted({c.spend for c in cands})
    groups = tuple(tuple(k for k, c in enumerate(cands) if c.spend == cur) for cur in spends)
    key = (n, groups, risk, K)
    if key not in _CACHE:
        _CACHE[key] = _build(n, groups, risk, K)
    prob, P = _CACHE[key]
    P["ep"].value = np.array([c.ep for c in cands])
    P["kap"].value = np.array([max(c.kappa, 0.0) for c in cands])
    P["lam"].value = max(lam, 0.0)
    P["avail"].value = np.array([max(avail.get(cur, 0.0), 0.0) for cur in spends])
    P["wmax"].value = w_max
    if risk == "variance":
        P["R"].value = np.sqrt(max(lam, 0.0) / (K - 1)) * (R - R.mean(axis=0))
    else:
        P["R"].value = R
        P["alpha"].value = max(lam, 0.0) / ((1 - alpha) * K)
    try:
        prob.solve(solver=cp.CLARABEL)
    except Exception:
        try:
            prob.solve()
        except Exception:
            return np.zeros(n)
    w = P["w"].value
    if w is None:
        return np.zeros(n)
    return np.clip(np.asarray(w).ravel(), 0, w_max)
