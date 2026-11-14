"""
data.py — Data acquisition module.

Sources
-------
1. GeckoTerminal public API (default, no key): pool OHLCV from on-chain swaps.
   Free tier = last ~6 months of history, 30 calls/min, <=1000 candles/call.
2. Uniswap V3 subgraph on The Graph (optional, free API key): hourly pool data
   with longer history and the pool's active liquidity.
3. Ethereum JSON-RPC (public node): historical base fee + priority fee
   (eth_feeHistory) and current pool liquidity (eth_call).
4. A synthetic generator, used ONLY to test the pipeline offline.

Every downloaded series is cached to CSV under data/raw so the notebook is
reproducible and does not hammer the APIs.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import requests

from .config import (NETWORK, RPC_URLS, UNISWAP_V3_SUBGRAPH_ID, WETH, PoolSpec)

GT_BASE = "https://api.geckoterminal.com/api/v2"
GT_HEADERS = {"accept": "application/json;version=20230302"}

# CoinGecko official API, on-chain (DEX) endpoints = same data as GeckoTerminal.
#   Demo plan (free key from coingecko.com/en/api): base below + header x-cg-demo-api-key
#   Paid plan: https://pro-api.coingecko.com/api/v3/onchain + header x-cg-pro-api-key
CG_DEMO_BASE = "https://api.coingecko.com/api/v3/onchain"
CG_PRO_BASE = "https://pro-api.coingecko.com/api/v3/onchain"


def _endpoint(source: str, api_key: str | None):
    """Base URL + headers for the chosen provider."""
    if source == "geckoterminal":
        return GT_BASE, GT_HEADERS
    if source == "coingecko":
        if not api_key:
            raise ValueError("source='coingecko' needs a CoinGecko API key (free Demo key on coingecko.com/en/api)")
        return CG_DEMO_BASE, {"accept": "application/json", "x-cg-demo-api-key": api_key}
    if source == "coingecko_pro":
        return CG_PRO_BASE, {"accept": "application/json", "x-cg-pro-api-key": api_key}
    raise ValueError(source)


# ==========================================================================
# 1. CoinGecko / GeckoTerminal on-chain OHLCV
# ==========================================================================
def _get_json(url: str, params: dict | None = None, retries: int = 6, sleep: float = 2.2,
              headers: dict | None = None) -> dict:
    """GET with polite retry / back-off (free tiers allow ~30 calls/min)."""
    for k in range(retries):
        r = requests.get(url, params=params, headers=headers or GT_HEADERS, timeout=30)
        if r.status_code == 200:
            time.sleep(sleep)
            return r.json()
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(sleep * (2 ** k))
            continue
        r.raise_for_status()
    raise RuntimeError(f"GET failed after {retries} retries: {url}")


def fetch_pool_info(pool: PoolSpec, source: str = "geckoterminal", api_key: str | None = None) -> dict:
    """Pool metadata (name, reserve in USD) — used as a sanity check."""
    base, hdr = _endpoint(source, api_key)
    js = _get_json(f"{base}/networks/{NETWORK}/pools/{pool.address}", headers=hdr)
    a = js["data"]["attributes"]
    return {"symbol": pool.symbol, "name": a.get("name"),
            "reserve_usd": float(a.get("reserve_in_usd") or np.nan),
            "fee_tier": pool.fee, "address": pool.address}


def fetch_pool_ohlcv(pool: PoolSpec, start: pd.Timestamp, end: pd.Timestamp,
                     timeframe: str = "hour", aggregate: int = 4,
                     cache_dir: str | Path = "data/raw", refresh: bool = False,
                     source: str = "geckoterminal", api_key: str | None = None,
                     max_pages: int = 40) -> pd.DataFrame:
    """
    Price of 1 ETH expressed in the pool's stablecoin, as OHLCV candles.

    source = "coingecko"     : CoinGecko API, on-chain endpoint (free Demo key)
             "coingecko_pro" : CoinGecko paid plan (history beyond 6 months)
             "geckoterminal" : CoinGecko's public GeckoTerminal API (no key)
    All three serve the same on-chain candles built from Uniswap swaps.

    We ask for `currency=token` and `token=<WETH address>`, i.e.
    the price of WETH in units of the other token of the pool (the stablecoin).
    This keeps the de-peg information: a USD-denominated price would hide it.
    """
    cache = Path(cache_dir) / f"gt_{pool.symbol}_{pool.address[:8]}_{timeframe}{aggregate}.csv"
    if cache.exists() and not refresh:
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        if df.index.min() <= start + pd.Timedelta(hours=aggregate) and df.index.max() >= end - pd.Timedelta(days=1):
            return _complete_bars(df, _bar_h(timeframe, aggregate)).loc[start:end]

    base, hdr = _endpoint(source, api_key)
    url = f"{base}/networks/{NETWORK}/pools/{pool.address}/ohlcv/{timeframe}"
    before = int(end.timestamp())
    rows, prev_oldest = [], None
    for page in range(1, max_pages + 1):
        js = _get_json(url, params={"aggregate": aggregate, "before_timestamp": before,
                                    "limit": 1000, "currency": "token", "token": WETH}, headers=hdr)
        lst = js["data"]["attributes"]["ohlcv_list"]
        if not lst:
            break
        oldest = min(r[0] for r in lst)
        # Guard: if the API ignores before_timestamp or has no older data, it returns the
        # same page again -> stop instead of looping forever.
        if prev_oldest is not None and oldest >= prev_oldest:
            print(f"  {pool.symbol}: API returned no older candles; history starts at "
                  f"{pd.to_datetime(prev_oldest, unit='s')}")
            break
        rows.extend(lst)
        prev_oldest = oldest
        print(f"  {pool.symbol}: page {page}, {len(rows)} candles, back to {pd.to_datetime(oldest, unit='s')}")
        if oldest <= start.timestamp() or len(lst) < 1000:
            break
        before = int(oldest)
    else:
        print(f"  {pool.symbol}: stopped after {max_pages} pages")

    # Not enough history from the keyed CoinGecko endpoint -> same data via GeckoTerminal
    if (source != "geckoterminal" and prev_oldest is not None
            and prev_oldest > start.timestamp() + 7 * 86400):
        print(f"  {pool.symbol}: {source} only returned data back to {pd.to_datetime(prev_oldest, unit='s')}; "
              f"retrying via GeckoTerminal (CoinGecko's public on-chain API)")
        return fetch_pool_ohlcv(pool, start, end, timeframe, aggregate, cache_dir, refresh,
                                "geckoterminal", None, max_pages)

    if not rows:
        raise RuntimeError(f"No OHLCV returned for {pool.symbol}. "
                           "Free CoinGecko/GeckoTerminal plans only serve the last ~6 months.")
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("ts").sort_values("ts")
    df.index = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert(None)
    df = df.drop(columns="ts").astype(float)

    # Orientation check: ETH is worth >> 1 stablecoin. If we got the inverse, flip it.
    if df["close"].median() < 1.0:
        inv = 1.0 / df[["open", "high", "low", "close"]]
        df["open"], df["close"] = inv["open"], inv["close"]
        df["high"], df["low"] = inv["low"], inv["high"]

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(cache)
    return _complete_bars(df, _bar_h(timeframe, aggregate)).loc[start:end]


def _bar_h(timeframe: str, aggregate: int) -> float:
    return {"minute": aggregate / 60.0, "hour": float(aggregate), "day": 24.0 * aggregate}[timeframe]


def _complete_bars(df: pd.DataFrame, bar_hours: float) -> pd.DataFrame:
    """Drop the candle that is still forming (its close lies in the future)."""
    now = pd.Timestamp.now("UTC").tz_localize(None)
    return df[df.index + pd.Timedelta(hours=bar_hours) <= now]


# ==========================================================================
# 2. Uniswap V3 subgraph (optional, longer history)
# ==========================================================================
def fetch_subgraph_hourly(pool: PoolSpec, api_key: str, start: pd.Timestamp, end: pd.Timestamp,
                          bar_hours: int = 4, cache_dir: str | Path = "data/raw") -> pd.DataFrame:
    """
    Hourly pool data from The Graph, resampled to `bar_hours`.
    Returns ETH price in stablecoin (OHLC from hourly closes), volumeUSD,
    tvlUSD and the pool's active liquidity L (useful for slippage).
    """
    cache = Path(cache_dir) / f"tg_{pool.symbol}_{pool.address[:8]}_{bar_hours}h.csv"
    if cache.exists():
        return pd.read_csv(cache, index_col=0, parse_dates=True).loc[start:end]
    url = f"https://gateway.thegraph.com/api/{api_key}/subgraphs/id/{UNISWAP_V3_SUBGRAPH_ID}"
    q = """query($pool:String!,$t0:Int!,$t1:Int!){
      poolHourDatas(first:1000, orderBy:periodStartUnix, orderDirection:asc,
        where:{pool:$pool, periodStartUnix_gte:$t0, periodStartUnix_lt:$t1}){
        periodStartUnix token0Price token1Price liquidity tvlUSD volumeUSD }}"""
    t0, t1, rows = int(start.timestamp()), int(end.timestamp()), []
    while t0 < t1:
        r = requests.post(url, json={"query": q, "variables": {"pool": pool.address, "t0": t0, "t1": t1}},
                          timeout=60)
        r.raise_for_status()
        batch = r.json()["data"]["poolHourDatas"]
        if not batch:
            break
        rows += batch
        t0 = int(batch[-1]["periodStartUnix"]) + 1
        if len(batch) < 1000:
            break
    df = pd.DataFrame(rows)
    df.index = pd.to_datetime(df["periodStartUnix"].astype(int), unit="s")
    # token0Price = price of token0 in token1 ... ETH-in-stable is whichever is > 1
    p0, p1 = df["token0Price"].astype(float), df["token1Price"].astype(float)
    px = p0 if p0.median() > 1 else p1
    out = pd.DataFrame({"px": px, "liquidity": df["liquidity"].astype(float),
                        "tvl_usd": df["tvlUSD"].astype(float), "volume": df["volumeUSD"].astype(float)})
    rule = f"{bar_hours}h"
    res = pd.DataFrame({
        "open": out["px"].resample(rule).first(), "high": out["px"].resample(rule).max(),
        "low": out["px"].resample(rule).min(), "close": out["px"].resample(rule).last(),
        "volume": out["volume"].resample(rule).sum(), "tvl_usd": out["tvl_usd"].resample(rule).last(),
        "liquidity": out["liquidity"].resample(rule).last()}).dropna(subset=["close"])
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    res.to_csv(cache)
    return res.loc[start:end]


# ==========================================================================
# 3. Ethereum RPC: gas history and pool liquidity
# ==========================================================================
class RPC:
    """Tiny JSON-RPC client that falls back across public endpoints."""

    def __init__(self, urls: Iterable[str] = RPC_URLS):
        self.urls = list(urls)
        self.s = requests.Session()

    def call(self, method: str, params: list, retries: int = 3):
        last = None
        for url in self.urls:
            for k in range(retries):
                try:
                    r = self.s.post(url, json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
                                    timeout=30)
                    js = r.json()
                    if "result" in js and (js["result"] is not None or method == "eth_getBlockByNumber"):
                        return js["result"]
                    last = js.get("error")
                except Exception as e:  # network error -> retry / next endpoint
                    last = e
                time.sleep(0.5 * (k + 1))
        raise RuntimeError(f"RPC {method} failed on all endpoints: {last}")

    def block(self, tag) -> dict:
        tag = tag if isinstance(tag, str) else hex(int(tag))
        return self.call("eth_getBlockByNumber", [tag, False])


def _block_at(rpc: RPC, ts: float, latest: Optional[dict] = None) -> int:
    """Block number whose timestamp is ~ts (post-Merge slots are 12 s). Never beyond the chain head."""
    latest = latest or rpc.block("latest")
    bn, bt = int(latest["number"], 16), int(latest["timestamp"], 16)
    if ts >= bt:                                   # time not reached yet -> newest block
        return bn
    guess = min(bn - int((bt - ts) / 12), bn)
    for _ in range(8):
        b = rpc.block(guess)
        if b is None:                              # node does not know this block: step back
            guess = min(guess, bn) - 10
            continue
        t = int(b["timestamp"], 16)
        step = int((ts - t) / 12)
        if abs(step) < 3:
            return guess
        guess = min(guess + step, bn)
    return guess


def _fee_sample(rpc: RPC, block: int, n: int):
    """
    Mean base fee and mean median tip (gwei) over the `n` blocks ending at `block`.
    Free public nodes cap eth_feeHistory (some at 10 blocks): shrink `n` until the
    node accepts it. Last resort: base fee from the block header, tip unknown (NaN).
    Returns (base, tip, n_that_worked).
    """
    for k in sorted({n, 10, 5, 1}, reverse=True):
        if k > n:
            continue
        try:
            fh = rpc.call("eth_feeHistory", [hex(k), hex(block), [50]], retries=1)
            base = np.array([int(x, 16) for x in fh["baseFeePerGas"][:-1]]) / 1e9
            rw = fh.get("reward") or []
            tip = np.array([int(r[0], 16) for r in rw]) / 1e9 if rw else np.array([np.nan])
            return float(base.mean()), float(np.nanmean(tip)) if np.isfinite(tip).any() else np.nan, k
        except Exception:
            continue
    blk = rpc.block(block)
    return int(blk["baseFeePerGas"], 16) / 1e9, np.nan, 1


def fetch_gas_history(index: pd.DatetimeIndex, bar_hours: float = 1.0, blocks_per_sample: int = 50,
                      cache_dir: str | Path = "data/raw", rpc: Optional[RPC] = None,
                      n_threads: int = 8) -> pd.DataFrame:
    """
    Gas price (gwei) at the CLOSE of every bar of `index` (bars of `bar_hours`):
    mean base fee and mean median priority fee over the `blocks_per_sample`
    blocks before the close, via eth_feeHistory (fewer blocks if the node caps it).

    For 15-minute strategies, pass an HOURLY index and forward-fill onto the
    15-minute grid: base fee moves by at most 12.5 % per block, so hourly
    sampling captures its dynamics at a quarter of the RPC calls.
    Requests run in `n_threads` parallel threads (one RPC client per thread).
    """
    tag = f"{int(round(bar_hours * 60))}m"
    cache = Path(cache_dir) / f"gas_{index[0]:%Y%m%d%H}_{index[-1]:%Y%m%d%H}_{tag}.csv"
    if cache.exists():
        return pd.read_csv(cache, index_col=0, parse_dates=True)
    rpc = rpc or RPC()
    close_ts = np.asarray(((index + pd.Timedelta(hours=bar_hours)) - pd.Timestamp("1970-01-01"))
                          // pd.Timedelta("1s"), dtype=float)
    latest = rpc.block("latest")
    head_ts = int(latest["timestamp"], 16)
    close_ts = np.minimum(close_ts, head_ts)       # the last bar may still be open
    b0, b1 = _block_at(rpc, close_ts[0], latest), _block_at(rpc, close_ts[-1], latest)
    # linear interpolation of block numbers between the two exact anchors
    blocks = np.round(np.interp(close_ts, [close_ts[0], close_ts[-1]], [b0, b1])).astype(int)

    # find once how many blocks per request the node accepts
    _, _, n_ok = _fee_sample(rpc, int(blocks[0]), blocks_per_sample)
    print(f"  gas: {len(blocks)} samples, {n_ok} block(s) per request, {n_threads} threads")

    import threading
    from concurrent.futures import ThreadPoolExecutor
    local = threading.local()
    shared = None if type(rpc) is RPC else rpc      # real client -> one per thread

    def work(i):
        cli = shared
        if cli is None:
            if not hasattr(local, "rpc"):
                local.rpc = RPC(rpc.urls)
            cli = local.rpc
        for attempt in range(5):                   # public nodes throttle bursts: back off and retry
            try:
                base, tip, _ = _fee_sample(cli, int(blocks[i]), n_ok)
                return index[i], base, tip
            except Exception:
                time.sleep(1.0 * (attempt + 1))
        return index[i], np.nan, np.nan            # filled from neighbours below

    rows, done = [], 0
    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        for r in ex.map(work, range(len(blocks))):
            rows.append(r)
            done += 1
            if done % 500 == 0:
                print(f"  gas {done}/{len(blocks)}")
    df = pd.DataFrame(rows, columns=["ts", "base_fee_gwei", "tip_gwei"]).set_index("ts").sort_index()
    n_miss = int(df["base_fee_gwei"].isna().sum())
    if n_miss:
        print(f"  gas: {n_miss} samples failed after retries -> interpolated from neighbours")
        df["base_fee_gwei"] = df["base_fee_gwei"].interpolate(limit_direction="both")
    if df["tip_gwei"].isna().all():
        df["tip_gwei"] = 0.0
    df["tip_gwei"] = df["tip_gwei"].fillna(df["tip_gwei"].median())
    df["gas_gwei"] = df["base_fee_gwei"] + df["tip_gwei"]
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(cache)
    return df


def fetch_pool_depth(pool: PoolSpec, rpc: Optional[RPC] = None) -> float:
    """
    Virtual reserve of the stablecoin side of a Uniswap V3 pool (in stablecoin units).

    Inside the active tick range a V3 pool behaves like x·y = k on *virtual*
    reserves:  x_v = L / sqrt(P),  y_v = L · sqrt(P)   (raw token units),
    where L = liquidity() and sqrt(P) = sqrtPriceX96 / 2**96 from slot0().
    """
    rpc = rpc or RPC()
    L = int(rpc.call("eth_call", [{"to": pool.address, "data": "0x1a686502"}, "latest"]), 16)
    slot0 = rpc.call("eth_call", [{"to": pool.address, "data": "0x3850c7bd"}, "latest"])
    sqrtp = int(slot0[2:66], 16) / 2**96
    raw = L / sqrtp if pool.stable_is_token0 else L * sqrtp
    return raw / 10**pool.stable_decimals


# ==========================================================================
# 4. Panel construction
# ==========================================================================
def build_panel(ohlcv: Dict[str, pd.DataFrame], max_ffill: int = 2, freq: str = "15min"):
    """
    Align all pools on a common time grid.
    Returns (close, open, volume) DataFrames, columns = stablecoin symbols,
    values = price of 1 ETH in that stablecoin.
    `freq` is the bar size ("15min", "1h", "4h"…). Gaps (bars with no swap) are
    forward-filled for at most `max_ffill` bars.
    """
    close = pd.concat({k: v["close"] for k, v in ohlcv.items()}, axis=1).sort_index()
    opn = pd.concat({k: v["open"] for k, v in ohlcv.items()}, axis=1).sort_index()
    vol = pd.concat({k: v["volume"] for k, v in ohlcv.items()}, axis=1).sort_index()
    grid = pd.date_range(close.index[0], close.index[-1], freq=freq)
    close = close.reindex(grid).ffill(limit=max_ffill)
    opn = opn.reindex(grid)
    opn = opn.fillna(close.shift(1)).ffill(limit=max_ffill)  # no swap in bar -> open = prev close
    vol = vol.reindex(grid).fillna(0.0)
    ok = close.notna().all(axis=1) & opn.notna().all(axis=1)
    return close[ok], opn[ok], vol[ok]


# ==========================================================================
# 5. Synthetic data (offline testing only — never used for reported results)
# ==========================================================================
def make_synthetic(n_bars: int = 17000, bar_minutes: int = 15, seed: int = 7,
                   symbols=("USDC", "USDT", "DAI")):
    """
    Simulated ETH prices in several stablecoins:
      ETH/USD   ~ GBM (80 % annual vol)
      stable/USD = 1 + OU deviation (a few bps) + rare de-peg jumps
      pool noise = OU microstructure noise (a few bps)
    Plus a log-OU gas price and fixed pool depths.
    """
    rng = np.random.default_rng(seed)
    bar_hours = bar_minutes / 60
    dt = bar_hours / (24 * 365)
    idx = pd.date_range("2026-05-15", periods=n_bars, freq=f"{bar_minutes}min")
    eth = 3000 * np.exp(np.cumsum(-0.5 * 0.8**2 * dt + 0.8 * np.sqrt(dt) * rng.standard_normal(n_bars)))

    def ou(n, theta, sigma, x0=0.0):
        x = np.empty(n); x[0] = x0
        a = np.exp(-theta)
        s = sigma * np.sqrt(1 - a * a)
        for t in range(1, n):
            x[t] = a * x[t - 1] + s * rng.standard_normal()
        return x

    close, opn = {}, {}
    for k, sym in enumerate(symbols):
        dev = ou(n_bars, 0.15, 4e-4 * (1 + k))          # peg deviation
        if sym != "USDC":                                 # occasional de-peg episodes
            for _ in range(3):
                t0 = rng.integers(50, n_bars - 60)
                dev[t0:t0 + 30] += -abs(rng.normal(0.004, 0.002)) * np.exp(-np.arange(30) / 8)
        micro = ou(n_bars, 0.5, 12e-4 * (1 + 0.8 * k))     # pool-specific noise
        px = eth / (1 + dev) * np.exp(micro)
        close[sym] = px
        opn[sym] = np.r_[px[0], px[:-1]] * np.exp(rng.normal(0, 2e-4, n_bars))
    close = pd.DataFrame(close, index=idx)
    opn = pd.DataFrame(opn, index=idx)
    vol = pd.DataFrame(1e6, index=idx, columns=list(symbols))
    gas = np.exp(np.log(2.0) + ou(n_bars, 0.1, 0.6))
    gas = pd.DataFrame({"base_fee_gwei": gas * 0.9, "tip_gwei": gas * 0.1, "gas_gwei": gas}, index=idx)
    depth = {"USDC": 150e6, "USDT": 40e6, "DAI": 3e6}
    return close, opn, vol, gas, {s: depth.get(s, 10e6) for s in symbols}
