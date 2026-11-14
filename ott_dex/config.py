"""
config.py — Research design in one place.

Universe  : Stablecoin arbitrage anchored on ETH (Uniswap V3, Ethereum mainnet).
            The paper's "fiat currencies quoted against ETH" become
            "stablecoins quoted against ETH". ETH is the intermediate asset,
            stablecoins are the bucket we hold and rotate between.
Data      : CoinGecko on-chain API (DEX OHLCV), 15-minute bars (configurable).
Frictions : per-pool fee tier, constant-product slippage on Uniswap V3
            virtual reserves, stochastic mainnet gas (base fee + tip).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

# --------------------------------------------------------------------------
# Tokens / pools
# --------------------------------------------------------------------------
WETH = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
NETWORK = "eth"  # GeckoTerminal network id for Ethereum mainnet


@dataclass(frozen=True)
class PoolSpec:
    """One Uniswap V3 pool ETH <-> stablecoin."""
    symbol: str             # stablecoin symbol (the "currency" of the bucket)
    address: str            # pool contract address
    fee: float              # swap-fee tier as a fraction (0.0005 = 0.05 %)
    stable_decimals: int    # ERC-20 decimals of the stablecoin
    stable_is_token0: bool  # Uniswap orders tokens by address


# Main universe: the deepest 0.05 % pools.
POOLS: Dict[str, PoolSpec] = {
    "USDC": PoolSpec("USDC", "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640", 0.0005, 6, True),
    "USDT": PoolSpec("USDT", "0x11b815efb8f581194ae79006d24e0d814b7697f6", 0.0005, 6, False),
    "DAI":  PoolSpec("DAI",  "0x60594a405d53811d3bc4766596efd80fd545a270", 0.0005, 18, True),
}

# Alternative fee tiers (0.3 %) for the fee-tier sensitivity analysis.
POOLS_030: Dict[str, PoolSpec] = {
    "USDC": PoolSpec("USDC", "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8", 0.003, 6, True),
    "USDT": PoolSpec("USDT", "0x4e68ccd3e89f51c3074ca5072bbac773960dfa36", 0.003, 6, False),
    "DAI":  PoolSpec("DAI",  "0xc2e9f25be6257c210d7adf0d4cd6e3e881ba25f8", 0.003, 18, True),
}

NUMERAIRE = "USDC"  # P&L is reported in USDC (≈ USD)

# Public Ethereum RPC endpoints used for gas history and pool liquidity.
RPC_URLS: Tuple[str, ...] = (
    "https://ethereum-rpc.publicnode.com",
    "https://eth.llamarpc.com",
    "https://cloudflare-eth.com",
)

# Uniswap V3 mainnet subgraph (optional source; needs a free The Graph API key)
UNISWAP_V3_SUBGRAPH_ID = "5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV"


# --------------------------------------------------------------------------
# Strategy / backtest parameters
# --------------------------------------------------------------------------
@dataclass
class Params:
    # sampling
    bar_minutes: int = 15                   # 15-minute bars (paper: 1/5/60 min main, 15/60/720 appendix)
    # walk-forward split (in days, converted to bars below)
    formation_days: float = 30.0            # formation window
    trading_days: float = 10.0              # traded out-of-sample, then refit
    # screening
    min_corr: float = 0.90
    min_eg_pass: float = 0.30               # share of weekly windows passing Engle–Granger
    eg_window_days: float = 7.0             # Engle–Granger windows of 1 week
    # signals
    signal: str = "kalman"                  # "kalman" | "ewma" | "static"
    ewma_halflife_hours: float = 12.0
    z_open_grid: Tuple[float, ...] = (2.0, 2.5, 3.0)
    z_close_grid: Tuple[float, ...] = (0.0, 0.5)
    z_open: float = 2.0                     # default if grid search finds nothing
    z_close: float = 0.5
    z_stop: float = 6.0                     # stop-loss: spread keeps diverging
    max_hold_halflives: float = 6.0         # time stop, in OU half-lives
    # optimiser
    risk: str = "cvar"                      # "cvar" | "variance"
    lam: float = 0.5                        # risk-aversion λ
    cvar_alpha: float = 0.95
    w_max_pair: float = 0.5                 # max fraction of NAV per pair
    ep_horizon_halflives: float = 3.0       # EP horizon h, in half-lives
    entry_filter: bool = True               # EXTENSION: trade only if E[profit] > fees + slippage + gas
                                            # (False = paper: open on every threshold crossing)
    # frictions
    use_dex_frictions: bool = True          # False => paper's flat fee, no slippage/gas
    flat_fee: float = 0.001                 # paper's 0.1 % (used only if use_dex_frictions=False)
    gas_units_per_swap: int = 150_000       # Uniswap V3 single-hop swap
    execution_lag: int = 1                  # execute at next bar's open
    # capital
    capital_per_stable: float = 250_000.0   # starting balance in each stablecoin
    rf_annual: float = 0.04                 # risk-free rate for Sharpe (paper uses 4 %)

    # ---- derived quantities (everything in bars) ----
    @property
    def bar_hours(self) -> float:
        return self.bar_minutes / 60.0

    @property
    def bars_per_day(self) -> float:
        return 24.0 / self.bar_hours

    @property
    def formation_bars(self) -> int:
        return int(round(self.formation_days * self.bars_per_day))

    @property
    def trading_bars(self) -> int:
        return int(round(self.trading_days * self.bars_per_day))

    @property
    def eg_window_bars(self) -> int:
        return int(round(self.eg_window_days * self.bars_per_day))

    @property
    def ewma_halflife_bars(self) -> int:
        return max(int(round(self.ewma_halflife_hours / self.bar_hours)), 2)

    @property
    def bars_per_year(self) -> float:
        return 365.0 * self.bars_per_day


def with_fee(pools: Dict[str, PoolSpec], fee: float) -> Dict[str, PoolSpec]:
    """Same pools with a hypothetical fee tier (fee-tier sensitivity analysis)."""
    from dataclasses import replace as _r
    return {k: _r(v, fee=fee) for k, v in pools.items()}
