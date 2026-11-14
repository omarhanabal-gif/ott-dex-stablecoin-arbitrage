"""
Automated checks of the time-series pipeline.   Run:  pytest -q

Each test checks a property the backtest relies on:
  * econometrics recover known parameters (OU half-life, Johansen rank);
  * signals are causal (no look-ahead);
  * AMM swap maths behave as expected (fees, slippage);
  * the closed-form allocation equals the cvxpy solution;
  * the book is market-neutral (P&L does not depend on the ETH price level);
  * the walk-forward only trades out-of-sample.
"""
import numpy as np
import pandas as pd
import pytest

from ott_dex import backtest as bt
from ott_dex import cointegration as ci
from ott_dex import data, frictions as fr, metrics as mt, optimizer as opt, signals as sg
from ott_dex.config import POOLS, Params

RNG = np.random.default_rng(42)


# --------------------------------------------------------------------------
# Econometrics
# --------------------------------------------------------------------------
def test_ou_fit_recovers_half_life():
    b, n = 0.9, 20_000                                 # AR(1) coefficient per bar
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = b * x[t - 1] + RNG.normal(0, 1e-4)
    est = ci.fit_ou(pd.Series(x), dt_hours=0.25)
    true_hl = np.log(2) / (-np.log(b)) * 0.25         # hours
    assert est["half_life_h"] == pytest.approx(true_hl, rel=0.15)
    assert est["b"] == pytest.approx(b, abs=0.02)


def test_johansen_finds_common_trend():
    n = 3000
    trend = np.cumsum(RNG.normal(0, 0.01, n))          # one common stochastic trend (ETH)
    logp = pd.DataFrame({c: trend + RNG.normal(0, 5e-4, n) for c in ("A", "B", "C")})
    # Theory: n − 1 = 2 relations. In finite samples the last trace test (H0: r ≤ n−1)
    # over-rejects, so rank 3 also occurs; what matters is r ≥ n − 1 (all spreads stationary).
    assert ci.johansen(logp)["rank"] >= 2


def test_engle_granger_screen_passes_cointegrated_pairs():
    n = 2000
    trend = np.cumsum(RNG.normal(0, 0.01, n))
    logp = pd.DataFrame({"A": trend + RNG.normal(0, 5e-4, n), "B": trend + RNG.normal(0, 5e-4, n)})
    tab = ci.rolling_screen(logp, window=400)
    assert tab.loc["A:B", "eg_pass_rate"] >= 0.8
    assert tab.loc["A:B", "corr"] > 0.9


# --------------------------------------------------------------------------
# Signals: causality (no look-ahead)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("method", ["static", "ewma", "kalman"])
def test_signals_are_causal(method):
    n, t_cut = 1500, 1000
    s = pd.Series(np.cumsum(RNG.normal(0, 1e-5, n)) + RNG.normal(0, 3e-4, n),
                  index=pd.date_range("2026-01-01", periods=n, freq="15min"))
    form_end = s.index[700]
    kp = sg.fit_state_space(s.loc[:form_end]) if method == "kalman" else None
    z1 = sg.make_z(s, method, form_end, 20, kp)["z"]
    s2 = s.copy()
    s2.iloc[t_cut + 1:] += 0.05                        # change only the FUTURE
    z2 = sg.make_z(s2, method, form_end, 20, kp)["z"]
    pd.testing.assert_series_equal(z1.iloc[: t_cut + 1], z2.iloc[: t_cut + 1])


# --------------------------------------------------------------------------
# Frictions
# --------------------------------------------------------------------------
def test_swap_without_frictions_is_exact():
    r = fr.buy_eth(3000.0, price=3000.0, depth=None, fee=0.0)
    assert r.out == pytest.approx(1.0)
    assert r.fee_cost == 0 and r.slip_cost == pytest.approx(0, abs=1e-9)


def test_round_trip_loses_fees_plus_slippage():
    price, depth, fee, amount = 3000.0, 1e6, 0.0005, 50_000.0
    b = fr.buy_eth(amount, price, depth, fee)
    s = fr.sell_eth(b.out, price, depth, fee)
    loss = amount - s.out
    assert loss > 2 * fee * amount * 0.99              # at least the two fees
    assert b.slip_cost > 0 and s.slip_cost > 0         # price impact on both legs
    # deeper pool -> less slippage
    assert fr.buy_eth(amount, price, 1e8, fee).slip_cost < b.slip_cost


def test_gas_cost_formula():
    assert fr.gas_cost_usd(150_000, 2.0, 3000.0) == pytest.approx(150_000 * 2e-9 * 3000)


# --------------------------------------------------------------------------
# Optimiser
# --------------------------------------------------------------------------
@pytest.mark.parametrize("risk", ["cvar", "variance"])
def test_closed_form_matches_cvxpy(risk):
    for _ in range(5):
        r = pd.Series(RNG.normal(0.004, 0.003, 300))
        c = opt.Candidate(("A", "B"), 1, "B", 0.002, RNG.uniform(0.01, 0.3), r)
        lam, av = RNG.uniform(0, 0.3), {"B": RNG.uniform(0.1, 1), "D": 1.0}
        fast = opt.solve([c], av, risk, lam, 0.95, 0.5)[0]
        dummy = opt.Candidate(("C", "D"), 1, "D", -1.0, 0.1, pd.Series(np.zeros(300)))
        slow = opt.solve([c, dummy], av, risk, lam, 0.95, 0.5)[0]   # forces the cvxpy path
        assert fast == pytest.approx(slow, abs=1e-4)


def test_optimizer_respects_constraints():
    cands = [opt.Candidate((f"X{k}", "S"), 1, "S", 0.01, 0.0, pd.Series(RNG.normal(0.01, 1e-3, 300)))
             for k in range(3)]
    w = opt.solve(cands, {"S": 0.4}, "cvar", 0.1, 0.95, 0.5)
    assert np.all(w >= -1e-8) and np.all(w <= 0.5 + 1e-8)
    assert w.sum() <= 0.4 + 1e-6                       # all three spend currency S


# --------------------------------------------------------------------------
# Backtest engine
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def synth():
    return data.make_synthetic(n_bars=4200, bar_minutes=15, seed=3)


def test_book_is_market_neutral(synth):
    """Scaling every ETH price by k (ETH moves, cross-rates unchanged) must not change P&L."""
    close, opn, _, gas, depth = synth
    p = Params(use_dex_frictions=False)                # gas is priced in ETH, so switch it off here
    res = []
    for k in (1.0, 2.5):
        r = bt.walk_forward(close * k, opn * k, gas["gas_gwei"], depth, POOLS, p, tune=False, verbose=False)
        res.append(r["nav"])
    pd.testing.assert_series_equal(res[0], res[1], rtol=1e-9, check_names=False)


def test_walk_forward_trades_only_out_of_sample(synth):
    close, opn, _, gas, depth = synth
    p = Params()
    r = bt.walk_forward(close, opn, gas["gas_gwei"], depth, POOLS, p, tune=False, verbose=False)
    first_oos = close.index[p.formation_bars]
    assert r["nav"].index[0] == first_oos
    if len(r["trades"]):
        assert (pd.to_datetime(r["trades"]["open"]) > first_oos).all()   # executed after a signal
        assert {"pnl_usd", "fee_usd", "slip_usd", "gas_usd"} <= set(r["trades"].columns)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def test_max_drawdown_and_win_loss():
    nav = pd.Series([100, 110, 99, 120, 90, 130.0])
    assert mt.perf(nav, 35040)["max_drawdown"] == pytest.approx(90 / 120 - 1)
    trades = pd.DataFrame({"pnl_usd": [5, -2, 3, -1, 4], "hold_hours": 1.0,
                           "fee_usd": 0.0, "slip_usd": 0.0, "gas_usd": 0.0})
    st = mt.trade_stats(trades)
    assert st["win_loss_ratio"] == pytest.approx(3 / 2)      # paper's definition (counts)
    assert st["profit_factor"] == pytest.approx(12 / 3)
