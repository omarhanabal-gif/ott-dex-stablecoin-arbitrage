"""
frictions.py — Real-world DEX execution costs.

1. Swap fee      : the pool's fee tier f (0.01 %, 0.05 %, 0.3 %, 1 %) is taken
                   from the input amount, as in Uniswap.
2. Slippage      : constant-product x·y = k on the pool's *virtual* reserves.
                   For a Uniswap V3 pool inside its active tick range this is
                   exact; R_stable = depth, R_eth = depth / P.
                   amount_out = R_out · a(1−f) / (R_in + a(1−f))
3. Gas           : gas_units × (base fee + tip) [gwei] × 1e-9 × P(ETH) → USD.

The functions return the amount received AND a cost breakdown, so the
backtest can report how much P&L each friction eats.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SwapResult:
    out: float          # amount received
    fee_cost: float     # cost of the fee, in units of the *stablecoin* involved
    slip_cost: float    # price-impact cost, in stablecoin units


def cp_out(amount_in: float, r_in: float, r_out: float, fee: float) -> float:
    a = amount_in * (1.0 - fee)
    return r_out * a / (r_in + a)


def buy_eth(stable_in: float, price: float, depth: float | None, fee: float) -> SwapResult:
    """Spend `stable_in` stablecoins to buy ETH. price = stablecoins per ETH (mid)."""
    ideal_eth = stable_in / price
    if depth is None or depth == float("inf"):         # paper's frictionless book
        eth = stable_in * (1 - fee) / price
    else:
        eth = cp_out(stable_in, depth, depth / price, fee)
    fee_cost = stable_in * fee
    slip_cost = (ideal_eth - eth) * price - fee_cost
    return SwapResult(eth, fee_cost, max(slip_cost, 0.0))


def sell_eth(eth_in: float, price: float, depth: float | None, fee: float) -> SwapResult:
    """Sell `eth_in` ETH for stablecoins."""
    ideal = eth_in * price
    if depth is None or depth == float("inf"):
        out = eth_in * (1 - fee) * price
    else:
        out = cp_out(eth_in, depth / price, depth, fee)
    fee_cost = ideal * fee
    slip_cost = ideal - out - fee_cost
    return SwapResult(out, fee_cost, max(slip_cost, 0.0))


def gas_cost_usd(gas_units: int, gas_gwei: float, eth_usd: float) -> float:
    return gas_units * gas_gwei * 1e-9 * eth_usd


def round_trip_cost_bps(notional: float, fee_a: float, fee_b: float, depth_a: float, depth_b: float,
                        gas_usd_per_swap: float) -> float:
    """
    Approximate cost (in bps of notional) of opening AND closing one spread
    trade = 4 swaps. Used by the friction-aware entry filter and the optimiser.
    """
    fees = 2 * (fee_a + fee_b)
    slip = 2 * (notional / depth_a + notional / depth_b) if depth_a and depth_b else 0.0
    gas = 4 * gas_usd_per_swap / notional
    return 1e4 * (fees + slip + gas)
