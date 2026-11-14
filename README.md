# OTT-DEX — Extending the Optimal Trading Technique to DEX Stablecoin Arbitrage

Group project, Algorithmic Trading. Extension of Yang & Malik (2024), *Optimal Market-Neutral Multivariate Pair Trading on the Cryptocurrency Platform*, IJFS 12:77.

**Authors:** Member 1 · Member 2 · Member 3 · Member 4 *(replace with your names)*

> **Main finding.** On Uniswap V3 (May–Nov 2026, 15-min bars, out-of-sample), the paper's OTT rule loses money once real DEX frictions are modelled (−$46k with the paper's flat 0.1 % fee; −$1.4k with real fee tiers, slippage and gas). A friction-aware entry filter turns it into a small but consistent profit (94–100 % winning trades), yet capacity is tiny: ETH/stablecoin cross-pool mispricings close within one 15-minute bar.

## Research design (unique, not the paper's setup)

| Component | Choice |
|---|---|
| Universe | **Stablecoin arbitrage**: ETH priced in USDC, USDT, DAI (ETH is only the intermediate) |
| Data | **Uniswap V3** on-chain swaps via the **CoinGecko API** (on-chain endpoints, free Demo key); gas and pool liquidity via Ethereum RPC |
| Sampling / window | **15-minute** bars, last ~6 months (May → Nov 2026); a 4-hour run is kept as robustness check |
| Cointegration | Engle–Granger (baseline) + **Johansen** + **Ornstein–Uhlenbeck** |
| Signals | **Kalman-filter state-space** dynamic bands (EWMA also available) |
| Optimisation | **mean–CVaR** with a convex price-impact penalty (mean–variance kept as baseline) |
| Frictions | **pool fee tier**, **x·y=k slippage** on V3 virtual reserves, **stochastic gas** |
| Validation | **walk-forward**: 30-day formation / 10-day out-of-sample trading, refit each block |

## Repository structure

```
ott-dex/
├── ott_dex/                         # the pipeline, one module per stage
│   ├── config.py         # pools, fee tiers, all parameters (Params dataclass)
│   ├── data.py           # DATA ACQUISITION: CoinGecko/GeckoTerminal OHLCV, The Graph, RPC gas & liquidity, caching
│   ├── cointegration.py  # COINTEGRATION TESTING: correlation, Engle–Granger, Johansen, OU fit
│   ├── signals.py        # static / EWMA / Kalman state-space z-scores
│   ├── frictions.py      # fee tiers, constant-product slippage, gas cost
│   ├── optimizer.py      # OPTIMISATION SOLVER: mean–variance & mean–CVaR (closed form + cvxpy)
│   ├── backtest.py       # BACKTEST ENGINE + walk-forward driver + benchmarks
│   └── metrics.py        # Sharpe, max drawdown, win/loss ratio, active P&L
├── notebooks/
│   └── OTT_DEX_stablecoins.ipynb    # step-by-step walkthrough (executed, with outputs)
├── tests/test_pipeline.py           # automated checks (pytest)
├── data/raw/                        # the exact data used in the report (CoinGecko 15-min & 4h OHLCV, gas, pool depth)
├── results/                         # main results, 15-min bars (tables, NAVs, trades)
├── results_4h/                      # robustness run, 4-hour bars
├── report/                          # LaTeX report (5 pages) + figures
└── requirements.txt
```

## How to reproduce the results

```bash
pip install -r requirements.txt
pytest -q                                   # 15 automated checks, ~10 s
jupyter lab notebooks/OTT_DEX_stablecoins.ipynb   # then Run All
```
**No API key is needed to reproduce the report**: the study window is frozen (20 May 14:30 → 14 Nov 2026 14:15 UTC) and all inputs are in `data/raw/`, so the notebook reads them from disk. A full run (5 variants + sensitivity) takes about 5–10 minutes.

To download fresh data instead: get a free CoinGecko Demo key (coingecko.com/en/api → Developer Dashboard), set `COINGECKO_API_KEY` in Step 0 (or `export COINGECKO_API_KEY=...`), switch to the "latest 6 months" line in Step 2, and delete `data/raw/pool_depth.json`. The first run downloads the 15-min prices (≈2 min) and the hourly gas history (≈5–10 min, ~4,300 RPC calls in 8 threads).

## Automated tests (`tests/test_pipeline.py`)
OU estimation recovers a known half-life · Johansen and Engle–Granger detect a common trend · every signal (static, EWMA, Kalman) is causal (changing the future does not change past z-scores) · swap maths (fees, slippage, gas) · the closed-form allocation equals the cvxpy solution · constraints are respected · the book is market-neutral (scaling the ETH price leaves P&L unchanged) · the walk-forward trades only out-of-sample · metrics (max drawdown, win/loss ratio).

To change the sampling frequency use `Params(bar_minutes=60)` (or 240 for 4h) and fetch the matching candles in Step 2.

`DATA_MODE = "synthetic"` runs the whole pipeline offline on simulated data. Use it **only to test the code**; its numbers mean nothing.

## Known limitations (state them in the report)
* The free CoinGecko Demo plan only serves ~6 months of OHLCV. For a longer window, use `data.fetch_subgraph_hourly()` with a free The Graph API key.
* Pool depth is read at one block (saved in `data/raw/pool_depth.json`) and held constant, because the free RPC is not an archive node.
* Execution is at the next bar's open price; MEV, sandwich attacks and failed transactions are not modelled.
* Gas is paid from the USDC sleeve. Gas is sampled hourly (one block per request on the free node) and forward-filled to 15 min.
