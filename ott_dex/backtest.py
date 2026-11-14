"""
backtest.py — Event-driven backtest engine with DEX frictions + walk-forward.

Timeline of one bar t (execution_lag = 1, the default):
  1. orders decided at the close of bar t−1 are executed at the OPEN of bar t
     (closing orders first, then opening orders), paying fee + slippage + gas;
  2. the portfolio is marked to market at the CLOSE of bar t;
  3. signals are computed from the close of bar t and new orders are queued.
No information from bar t is used to trade at a price earlier than bar t.

Spread trade on pair (i, j), spread s = log P_i − log P_j (P = price of ETH):
  direction +1 (s too high: ETH expensive in i)
      open : spend j → buy ETH in pool j → sell that ETH in pool i → receive i
      close: spend all i received → buy ETH in pool i → sell in pool j → get j back
  direction −1 is the mirror image.
The ETH bought is sold in the same bar, so the book never holds ETH
(the paper's market-neutrality constraint, eq. 5, holds by construction).
Net P&L = j received at close − j spent at open (valued in USDC) − gas.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import cointegration as ci
from . import signals as sg
from .config import NUMERAIRE, Params, PoolSpec
from .frictions import buy_eth, gas_cost_usd, sell_eth
from .optimizer import Candidate, solve


# ==========================================================================
# Model fitted on a formation window
# ==========================================================================
def fit_model(close: pd.DataFrame, form_slice: slice, eval_end, p: Params) -> dict:
    """
    Everything that is estimated on formation data only:
    screening, Johansen, OU parameters, Kalman (Q, R), z-scores (causal),
    and historical h-bar scenario returns for the risk term.
    """
    logp = np.log(close)
    lf = logp.iloc[form_slice]
    form_end = lf.index[-1]
    screen = ci.rolling_screen(lf, p.eg_window_bars)
    pairs = ci.screen_assets(screen, p.min_corr, p.min_eg_pass)
    try:
        jo = ci.johansen(lf)
    except Exception:
        jo = {"rank": np.nan, "table": None, "vectors": None}
    model = {"pairs": {}, "screen": screen, "johansen": jo, "form_end": form_end}
    for pair in pairs:
        s = ci.spread(logp.loc[lf.index[0]:eval_end], pair)
        ou = ci.fit_ou(s.loc[:form_end], p.bar_hours)
        hl_bars = max(ou["half_life_h"] / p.bar_hours, 1.0)
        kp = sg.fit_state_space(s.loc[:form_end]) if p.signal == "kalman" else None
        zdf = sg.make_z(s, p.signal, form_end, p.ewma_halflife_bars, kp)
        h = int(np.clip(round(p.ep_horizon_halflives * hl_bars), 1, 60))
        # Scenarios for the risk term: historical OU forecast errors over h bars,
        # e_k = (s_{k+h} − s_k) − (μ − s_k)(1 − e^{−θh}), from formation data only.
        sf = s.loc[:form_end].values
        decay = 1 - np.exp(-np.log(2) / hl_bars * h)
        scen = pd.Series((sf[h:] - sf[:-h]) - (ou["mu"] - sf[:-h]) * decay,
                         index=s.loc[:form_end].index[:-h])   # dated -> joint scenarios across pairs
        model["pairs"][pair] = {"s": s, "z": zdf, "ou": ou, "hl_bars": hl_bars,
                                "h": h, "scen": scen, "kalman": kp,
                                "zv": zdf["z"].to_numpy(), "sv": zdf["std"].to_numpy()}  # fast access
    return model


# ==========================================================================
# Engine
# ==========================================================================
@dataclass
class Engine:
    p: Params
    pools: Dict[str, PoolSpec]
    depth: Dict[str, float]
    symbols: List[str]
    bal: Dict[str, float] = field(default_factory=dict)
    positions: Dict[tuple, dict] = field(default_factory=dict)
    trades: List[dict] = field(default_factory=list)
    nav_hist: List[tuple] = field(default_factory=list)
    costs: Dict[str, float] = field(default_factory=lambda: {"fee": 0.0, "slippage": 0.0, "gas": 0.0})
    pending: List[tuple] = field(default_factory=list)

    def __post_init__(self):
        if not self.bal:
            self.bal = {c: self.p.capital_per_stable for c in self.symbols}

    # ---------------- helpers ----------------
    def _fee(self, c):
        return self.pools[c].fee if self.p.use_dex_frictions else self.p.flat_fee

    def _depth(self, c):
        return self.depth.get(c) if self.p.use_dex_frictions else None

    def _rates(self, px) -> dict:
        """USDC value of 1 unit of each stablecoin, implied by ETH prices (px: dict or Series)."""
        num = px[NUMERAIRE]
        return {c: num / px[c] for c in self.symbols}

    def nav(self, px) -> float:
        r = self._rates(px)
        return float(sum(self.bal[c] * r[c] for c in self.symbols))

    def _gas_usd(self, gas_gwei: float, px: pd.Series) -> float:
        if not self.p.use_dex_frictions:
            return 0.0
        return gas_cost_usd(self.p.gas_units_per_swap, gas_gwei, px[NUMERAIRE])

    def _two_swaps(self, spend, recv, amount, px, gas_gwei):
        """spend `amount` of `spend` → ETH → `recv`. Returns received amount and costs (USDC)."""
        r = self._rates(px)
        b = buy_eth(amount, px[spend], self._depth(spend), self._fee(spend))
        s = sell_eth(b.out, px[recv], self._depth(recv), self._fee(recv))
        fee = b.fee_cost * r[spend] + s.fee_cost * r[recv]
        slip = b.slip_cost * r[spend] + s.slip_cost * r[recv]
        gas = 2 * self._gas_usd(gas_gwei, px)
        self.bal[spend] -= amount
        self.bal[recv] += s.out
        self.bal[NUMERAIRE] -= gas                     # gas paid from the USDC sleeve
        self.costs["fee"] += fee
        self.costs["slippage"] += slip
        self.costs["gas"] += gas
        return s.out, fee, slip, gas

    # ---------------- actions ----------------
    def _open(self, pair, direction, notional_usd, px, gas_gwei, ts, z):
        i, j = pair
        spend, recv = (j, i) if direction > 0 else (i, j)
        amount = min(notional_usd / self._rates(px)[spend], self.bal[spend])
        if amount <= 0:
            return
        got, fee, slip, gas = self._two_swaps(spend, recv, amount, px, gas_gwei)
        self.positions[pair] = {"dir": direction, "spend": spend, "recv": recv, "spent": amount,
                                "got": got, "t_open": ts, "z_open": z, "bars": 0,
                                "fee": fee, "slip": slip, "gas": gas,
                                "notional_usd": amount * self._rates(px)[spend]}

    def _close(self, pair, px, gas_gwei, ts, reason):
        pos = self.positions.pop(pair)
        amount = min(pos["got"], max(self.bal[pos["recv"]], 0.0))
        back, fee, slip, gas = self._two_swaps(pos["recv"], pos["spend"], amount, px, gas_gwei)
        r = self._rates(px)[pos["spend"]]
        pnl = (back - pos["spent"]) * r - pos["gas"] - gas
        self.trades.append({"pair": f"{pair[0]}:{pair[1]}", "direction": pos["dir"],
                            "open": pos["t_open"], "close": ts, "bars": pos["bars"],
                            "hold_hours": pos["bars"] * self.p.bar_hours,
                            "notional_usd": pos["notional_usd"], "pnl_usd": pnl,
                            "fee_usd": pos["fee"] + fee, "slip_usd": pos["slip"] + slip,
                            "gas_usd": pos["gas"] + gas, "exit": reason})

    # ---------------- decision ----------------
    def _decide(self, model, t_idx, px, gas_gwei, ts):
        p = self.p
        orders = []
        mp = model["pairs"]
        # 1) exits
        for pair, pos in list(self.positions.items()):
            pos["bars"] += 1
            if pair not in mp:
                orders.append(("close", pair, "dropped")); continue
            z = mp[pair]["zv"][t_idx]
            hl = mp[pair]["hl_bars"]
            if abs(z) < p.z_close or np.sign(z) != np.sign(pos["z_open"]):
                orders.append(("close", pair, "converged"))
            elif abs(z) > p.z_stop:
                orders.append(("close", pair, "stop_loss"))
            elif pos["bars"] >= p.max_hold_halflives * hl:
                orders.append(("close", pair, "time_stop"))
        closing = {o[1] for o in orders}
        # 2) entries
        nav = self.nav(px)
        rates = self._rates(px)
        gas1 = self._gas_usd(gas_gwei, px)
        cands = []
        for pair, m in mp.items():
            if pair in self.positions and pair not in closing:
                continue
            if pair in closing:
                continue
            z = m["zv"][t_idx]
            if not np.isfinite(z) or abs(z) < p.z_open or abs(z) > p.z_stop:
                continue
            d = 1 if z > 0 else -1
            i, j = pair
            spend = j if d > 0 else i
            dev = abs(z) * m["sv"][t_idx]
            theta_bar = np.log(2) / m["hl_bars"]
            gross = dev * (1 - np.exp(-theta_bar * m["h"]))
            fees = 2 * (self._fee(i) + self._fee(j))
            if p.entry_filter:
                ep = gross - fees
                if ep <= 0:                           # friction-aware entry filter
                    continue
            else:
                ep = gross                            # paper: costs ignored when sizing
            Di, Dj = self._depth(i), self._depth(j)
            kappa = 2 * nav * (1 / Di + 1 / Dj) if (Di and Dj) else 0.0
            # scenario P&L per unit notional = expected net return + signed OU forecast error
            cands.append(Candidate(pair, d, spend, ep, kappa, ep - d * m["scen"]))
        if cands:
            avail = {c: self.bal[c] * rates[c] / nav for c in self.symbols}
            w = solve(cands, avail, p.risk, p.lam, p.cvar_alpha, p.w_max_pair)
            for c, wk in zip(cands, w):
                if wk < 1e-4:
                    continue
                notional = wk * nav
                exp_profit = notional * c.ep - c.kappa * wk**2 * nav - 4 * gas1
                if p.entry_filter and exp_profit <= 0:   # gas makes the trade uneconomic
                    continue
                z = mp[c.pair]["zv"][t_idx]
                orders.append(("open", c.pair, c.direction, notional, z))
        return orders

    def _execute(self, orders, px, gas_gwei, ts):
        for o in orders:
            if o[0] == "close" and o[1] in self.positions:
                self._close(o[1], px, gas_gwei, ts, o[2])
        for o in orders:
            if o[0] == "open" and o[1] not in self.positions:
                self._open(o[1], o[2], o[3], px, gas_gwei, ts, o[4])

    # ---------------- main loop ----------------
    def run(self, close: pd.DataFrame, opn: pd.DataFrame, gas: pd.Series, model: dict,
            start: int, end: int, record: bool = True):
        """Trade bars [start, end) of `close`. z-series in `model` share close.index."""
        idx = close.index
        syms = list(close.columns)
        C, O, G = close.to_numpy(), opn.to_numpy(), np.asarray(gas, dtype=float)   # fast row access
        for t in range(start, end):
            ts = idx[t]
            g = float(G[t])
            if self.pending:                            # 1) execute at this bar's open
                self._execute(self.pending, dict(zip(syms, O[t])), g, ts)
                self.pending = []
            px = dict(zip(syms, C[t]))
            if record:                                  # 2) mark to market
                self.nav_hist.append((ts, self.nav(px)))
            zpos = model["offset"] + t                  # 3) decide on the close
            orders = self._decide(model, zpos, px, g, ts)
            if self.p.execution_lag == 0:
                self._execute(orders, px, g, ts)
            else:
                self.pending = orders
        return self

    def close_all(self, close, gas, t):
        px = dict(zip(close.columns, close.to_numpy()[t]))
        for pair in list(self.positions):
            self._close(pair, px, float(gas.iloc[t]), close.index[t], "end_of_test")
        self.nav_hist.append((close.index[t], self.nav(px)))


# ==========================================================================
# Walk-forward driver
# ==========================================================================
def _sharpe(nav: pd.Series) -> float:
    r = nav.pct_change().dropna()
    return float(r.mean() / r.std()) if len(r) > 2 and r.std() > 0 else -np.inf


def _attach_offset(model: dict, close_index, eval_start_ts) -> dict:
    """z-series start at the formation start; map global bar t -> row in z."""
    any_pair = next(iter(model["pairs"].values()), None)
    if any_pair is None:
        model["offset"] = 0
        return model
    z0 = any_pair["z"].index[0]
    model["offset"] = -close_index.get_loc(z0)
    return model


def grid_search(close, opn, gas, model, p: Params, pools, depth, form_slice) -> tuple:
    """Choose (z_open, z_close) on the FORMATION window only (fixes the paper's look-ahead)."""
    best, best_score = (p.z_open, p.z_close), -np.inf
    if not model["pairs"]:
        return best, pd.DataFrame()
    rows = []
    for zo in p.z_open_grid:
        for zc in p.z_close_grid:
            if zc >= zo:
                continue
            q = replace(p, z_open=zo, z_close=zc)
            e = Engine(q, pools, depth, list(close.columns))
            e.run(close, opn, gas, model, form_slice.start + 10, form_slice.stop)
            nav = pd.Series(dict(e.nav_hist))
            score = _sharpe(nav) if len(e.trades) >= 2 else -np.inf
            rows.append({"z_open": zo, "z_close": zc, "trades": len(e.trades),
                         "pnl_usd": nav.iloc[-1] - nav.iloc[0], "sharpe_bar": score})
            if score > best_score:
                best, best_score = (zo, zc), score
    return best, pd.DataFrame(rows)


def walk_forward(close: pd.DataFrame, opn: pd.DataFrame, gas: pd.Series, depth: Dict[str, float],
                 pools: Dict[str, PoolSpec], p: Params, tune: bool = True, verbose: bool = True) -> dict:
    """
    Rolling scheme:  [formation F bars | trade T bars] → shift by T → refit → ...
    Parameters (screening, OU, Kalman, thresholds) are always estimated on the
    formation window that precedes the trading window: every traded bar is
    out-of-sample.
    """
    n = len(close)
    F, T = p.formation_bars, p.trading_bars
    engine = Engine(p, pools, depth, list(close.columns))
    blocks = []
    start = 0
    while start + F < n:
        form = slice(start, start + F)
        t0, t1 = start + F, min(start + F + T, n)
        model = fit_model(close, form, close.index[t1 - 1], p)
        model = _attach_offset(model, close.index, close.index[t0])
        zo, zc = p.z_open, p.z_close
        grid = pd.DataFrame()
        if tune and model["pairs"]:
            (zo, zc), grid = grid_search(close, opn, gas, model, p, pools, depth, form)
        engine.p = replace(p, z_open=zo, z_close=zc)
        engine.run(close, opn, gas, model, t0, t1)
        blocks.append({"formation": (close.index[form.start], close.index[form.stop - 1]),
                       "trading": (close.index[t0], close.index[t1 - 1]),
                       "pairs": [f"{a}:{b}" for a, b in model["pairs"]],
                       "johansen_rank": model["johansen"]["rank"], "z_open": zo, "z_close": zc,
                       "half_lives_h": {f"{a}:{b}": round(m["hl_bars"] * p.bar_hours, 1)
                                        for (a, b), m in model["pairs"].items()},
                       "grid": grid})
        if verbose:
            print(f"[{close.index[t0]:%Y-%m-%d} → {close.index[t1-1]:%Y-%m-%d}] pairs={blocks[-1]['pairs']} "
                  f"rank={blocks[-1]['johansen_rank']} z=({zo},{zc}) trades so far={len(engine.trades)}")
        start += T
    engine.close_all(close, gas, n - 1)
    nav = pd.Series(dict(engine.nav_hist)).sort_index()
    nav = nav[~nav.index.duplicated(keep="last")]
    trades = pd.DataFrame(engine.trades)
    return {"nav": nav, "trades": trades, "costs": dict(engine.costs), "blocks": blocks,
            "params": p, "final_balances": dict(engine.bal)}


# ==========================================================================
# Benchmarks
# ==========================================================================
def benchmarks(close: pd.DataFrame, start_ts, p: Params) -> pd.DataFrame:
    """Buy-and-hold ETH and passive holding of the initial stablecoin basket (USDC value)."""
    c = close.loc[start_ts:]
    rates = c[NUMERAIRE].values[:, None] / c.values
    init = p.capital_per_stable * len(close.columns)
    hold = pd.Series((p.capital_per_stable * rates).sum(axis=1), index=c.index)
    eth_units = init / c[NUMERAIRE].iloc[0]
    bh = eth_units * c[NUMERAIRE]
    return pd.DataFrame({"buy_hold_ETH": bh, "hold_stables": hold})
