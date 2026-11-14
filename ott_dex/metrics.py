"""
metrics.py — Performance and trade statistics (paper's Tables 5–9 + drawdown).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def perf(nav: pd.Series, bars_per_year: float, rf_annual: float = 0.04) -> dict:
    r = nav.pct_change().dropna()
    years = len(r) / bars_per_year
    total = nav.iloc[-1] / nav.iloc[0] - 1
    ann = (1 + total) ** (1 / years) - 1 if years > 0 else np.nan
    vol = r.std() * np.sqrt(bars_per_year)
    rf_bar = (1 + rf_annual) ** (1 / bars_per_year) - 1
    ex = r - rf_bar
    sharpe = ex.mean() / r.std() * np.sqrt(bars_per_year) if r.std() > 0 else np.nan
    sharpe0 = r.mean() / r.std() * np.sqrt(bars_per_year) if r.std() > 0 else np.nan
    down = r[r < 0].std() * np.sqrt(bars_per_year)
    sortino = (r.mean() * bars_per_year - rf_annual) / down if down and down > 0 else np.nan
    dd = nav / nav.cummax() - 1
    mdd = dd.min()
    return {"total_return": total, "ann_return": ann, "ann_vol": vol,
            f"sharpe_rf{int(rf_annual*100)}%": sharpe, "sharpe_rf0": sharpe0,
            "sortino": sortino, "max_drawdown": mdd,
            "calmar": ann / abs(mdd) if mdd < 0 else np.nan}


def trade_stats(trades: pd.DataFrame) -> dict:
    if trades is None or trades.empty:
        return {"n_trades": 0}
    pnl = trades["pnl_usd"]
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    return {
        "n_trades": len(trades),
        "pct_winning": len(wins) / len(pnl),
        "pct_losing": len(losses) / len(pnl),
        "win_loss_ratio": len(wins) / len(losses) if len(losses) else np.inf,   # paper's definition
        "avg_win_usd": wins.mean() if len(wins) else 0.0,
        "avg_loss_usd": losses.mean() if len(losses) else 0.0,
        "payoff_ratio": wins.mean() / abs(losses.mean()) if len(wins) and len(losses) and losses.mean() != 0 else np.nan,
        "largest_win_usd": pnl.max(), "largest_loss_usd": pnl.min(),
        "profit_factor": wins.sum() / abs(losses.sum()) if losses.sum() != 0 else np.inf,
        "avg_holding_hours": trades["hold_hours"].mean(),
        "total_fees_usd": trades["fee_usd"].sum(),
        "total_slippage_usd": trades["slip_usd"].sum(),
        "total_gas_usd": trades["gas_usd"].sum(),
    }


def drawdown(nav: pd.Series) -> pd.Series:
    return nav / nav.cummax() - 1


def summary_table(results: dict, bars_per_year: float, rf: float) -> pd.DataFrame:
    """results = {name: walk_forward output or NAV Series}."""
    rows = {}
    for name, res in results.items():
        if isinstance(res, pd.Series):
            rows[name] = perf(res, bars_per_year, rf)
        else:
            rows[name] = {**perf(res["nav"], bars_per_year, rf), **trade_stats(res["trades"])}
    return pd.DataFrame(rows).T


def active_nav(nav: pd.Series, hold: pd.Series) -> pd.Series:
    """
    Strategy NAV minus the passive 'hold the initial stablecoin basket' NAV.
    Marking USDT/DAI balances in USDC adds FX noise that has nothing to do
    with the arbitrage; the active series isolates the trading P&L
    (the paper's Table 4 makes the same separation for fiat FX moves).
    """
    h = hold.reindex(nav.index).ffill()
    return nav - h + h.iloc[0]
