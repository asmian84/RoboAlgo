"""
data_engine/crypto_fetcher.py
────────────────────────────────────────────────────────────────────────────────
Crypto data feed — spot OHLCV + perp funding + stablecoin supply.

Purpose: feed cross-asset risk signals into the equity regime detector.
Not for trading crypto — for extracting signal from crypto to trade equities.

Sources (all keyless public REST):
  Spot OHLCV       → Coinbase Exchange → Binance → yfinance (fallback chain)
  Perp funding     → Binance Futures /fapi/v1/premiumIndex
  Stablecoin cap   → CoinGecko /coins/markets

All functions fail silently to None so the caller can degrade gracefully.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ── Cache ──────────────────────────────────────────────────────────────────
# Regime detection runs every scan; crypto doesn't move fast enough to need
# sub-minute freshness. 10-min bars cache is plenty.
_OHLCV_CACHE: dict[str, tuple[pd.DataFrame, float]] = {}
_FUNDING_CACHE: dict[str, tuple[float, float]] = {}
_STABLECOIN_CACHE: dict[str, tuple[dict, float]] = {}

_OHLCV_TTL = 600.0       # 10 min
_FUNDING_TTL = 300.0     # 5 min
_STABLECOIN_TTL = 1800.0 # 30 min

_HTTP_TIMEOUT = 5.0


# ── Symbol translation ────────────────────────────────────────────────────
# Accept short names ("BTC", "ETH", "XRP", ...) from callers and translate
# to each venue's canonical form. Most venues follow the same convention:
#     Coinbase / yfinance → "{SYM}-USD"
#     Binance spot / perp  → "{SYM}USDT"
# Overrides handle the exceptions (e.g. Binance uses MATIC → POL rename).

_PAIR_OVERRIDES: dict[str, dict[str, str]] = {
    # symbol → {venue: override}
    "MATIC": {"binance_spot": "POLUSDT", "binance_perp": "POLUSDT"},
}

_UNSUPPORTED_ON_COINBASE = {"BNB", "TRX", "TON"}
_UNSUPPORTED_ON_BINANCE_US = set()  # informational; we use the .com API


def _coinbase_pair(sym: str) -> str | None:
    if sym.upper() in _UNSUPPORTED_ON_COINBASE:
        return None
    ov = _PAIR_OVERRIDES.get(sym.upper(), {}).get("coinbase")
    return ov or f"{sym.upper()}-USD"


def _binance_spot_pair(sym: str) -> str | None:
    ov = _PAIR_OVERRIDES.get(sym.upper(), {}).get("binance_spot")
    return ov or f"{sym.upper()}USDT"


def _binance_perp_pair(sym: str) -> str | None:
    ov = _PAIR_OVERRIDES.get(sym.upper(), {}).get("binance_perp")
    return ov or f"{sym.upper()}USDT"


def _yf_pair(sym: str) -> str | None:
    ov = _PAIR_OVERRIDES.get(sym.upper(), {}).get("yfinance")
    return ov or f"{sym.upper()}-USD"


# ─────────────────────────────────────────────────────────────────────────────
#  Spot OHLCV
# ─────────────────────────────────────────────────────────────────────────────

def _coinbase_daily(symbol: str, bars: int) -> pd.DataFrame | None:
    """Fetch daily candles from Coinbase Exchange public API."""
    pair = _coinbase_pair(symbol)
    if not pair:
        return None
    try:
        url = f"https://api.exchange.coinbase.com/products/{pair}/candles"
        r = requests.get(url, params={"granularity": 86400}, timeout=_HTTP_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        # Coinbase format: [[time, low, high, open, close, volume], ...] newest first
        if not data or not isinstance(data, list):
            return None
        rows = []
        for c in data:
            if len(c) < 6:
                continue
            rows.append({
                "date":   pd.Timestamp(int(c[0]), unit="s", tz="UTC"),
                "low":    float(c[1]),
                "high":   float(c[2]),
                "open":   float(c[3]),
                "close":  float(c[4]),
                "volume": float(c[5]),
            })
        if not rows:
            return None
        df = pd.DataFrame(rows).sort_values("date").set_index("date")
        return df[["open", "high", "low", "close", "volume"]].iloc[-bars:]
    except Exception as e:
        logger.debug("Coinbase daily failed for %s: %s", symbol, e)
        return None


def _binance_daily(symbol: str, bars: int) -> pd.DataFrame | None:
    """Fetch daily klines from Binance spot public API."""
    pair = _binance_spot_pair(symbol)
    if not pair:
        return None
    try:
        url = "https://api.binance.com/api/v3/klines"
        r = requests.get(
            url,
            params={"symbol": pair, "interval": "1d", "limit": min(bars, 1000)},
            timeout=_HTTP_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
        # Binance kline: [openTime, open, high, low, close, volume, closeTime, ...]
        if not data or not isinstance(data, list):
            return None
        rows = []
        for k in data:
            if len(k) < 6:
                continue
            rows.append({
                "date":   pd.Timestamp(int(k[0]), unit="ms", tz="UTC"),
                "open":   float(k[1]),
                "high":   float(k[2]),
                "low":    float(k[3]),
                "close":  float(k[4]),
                "volume": float(k[5]),
            })
        if not rows:
            return None
        df = pd.DataFrame(rows).sort_values("date").set_index("date")
        return df[["open", "high", "low", "close", "volume"]]
    except Exception as e:
        logger.debug("Binance daily failed for %s: %s", symbol, e)
        return None


def _yfinance_daily(symbol: str, bars: int) -> pd.DataFrame | None:
    """Last-resort fallback via yfinance."""
    pair = _yf_pair(symbol)
    if not pair:
        return None
    try:
        import yfinance as yf
        period = "3mo" if bars <= 90 else "1y"
        df = yf.download(pair, period=period, interval="1d", progress=False, timeout=10)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        return df[cols].iloc[-bars:]
    except Exception as e:
        logger.debug("yfinance crypto fallback failed for %s: %s", symbol, e)
        return None


_MIN_FETCH_BARS = 60


def fetch_crypto_daily(symbol: str, bars: int = 60, force_refresh: bool = False) -> pd.DataFrame | None:
    """
    Fetch daily OHLCV for a crypto asset.

    Always fetches at least _MIN_FETCH_BARS from source so we have enough
    history to detect a valid response even when the caller only wants a
    handful of recent bars (e.g. the crypto-lead check asks for 5).

    Args:
        symbol: Short name — "BTC", "ETH", "SOL"
        bars:   Number of trailing daily bars to return

    Returns:
        DataFrame indexed by UTC date with columns [open, high, low, close, volume],
        or None if all sources fail.
    """
    sym = symbol.upper()
    key = sym  # cache keyed by symbol only; we slice per-call
    fetch_n = max(bars, _MIN_FETCH_BARS)
    now = time.time()
    if not force_refresh and key in _OHLCV_CACHE:
        df, ts = _OHLCV_CACHE[key]
        if now - ts < _OHLCV_TTL and len(df) >= bars:
            return df.iloc[-bars:].copy()

    for src in (_coinbase_daily, _binance_daily, _yfinance_daily):
        df = src(sym, fetch_n)
        if df is not None and len(df) >= min(10, fetch_n):
            _OHLCV_CACHE[key] = (df, now)
            return df.iloc[-bars:].copy()
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  Perp funding rate (positioning / leverage tell)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_funding_rate(symbol: str = "BTC", force_refresh: bool = False) -> float | None:
    """
    Return the current perpetual-futures funding rate, annualized as a percent.

    Positive extreme (>+40%) means longs are paying heavily to stay long —
    speculative leverage is stretched. Negative extreme (<-10%) means shorts
    are paying — often a capitulation / squeeze setup.

    Source: Binance USD-M futures premiumIndex (funds every 8h).
    Formula: rate_per_8h * 3 * 365 * 100 = annualized %.

    Returns None if the endpoint is unreachable (e.g. geo-blocked).
    """
    sym = symbol.upper()
    pair = _binance_perp_pair(sym)
    if not pair:
        return None

    now = time.time()
    if not force_refresh and sym in _FUNDING_CACHE:
        rate, ts = _FUNDING_CACHE[sym]
        if now - ts < _FUNDING_TTL:
            return rate

    try:
        r = requests.get(
            "https://fapi.binance.com/fapi/v1/premiumIndex",
            params={"symbol": pair},
            timeout=_HTTP_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
        per_8h = float(data.get("lastFundingRate", 0.0))
        annualized_pct = per_8h * 3 * 365 * 100
        _FUNDING_CACHE[sym] = (annualized_pct, now)
        return annualized_pct
    except Exception as e:
        logger.debug("Binance funding failed for %s: %s", symbol, e)
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  Stablecoin supply (sidelined-capital tell)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_stablecoin_supply(force_refresh: bool = False) -> dict | None:
    """
    Return stablecoin market cap + 24h change for USDT + USDC.

    Interpretation:
      Rapid expansion (>2% in 24h) = capital exiting risk into stables
        → risk-off flow, bearish for equities
      Rapid contraction (<-2%) = capital redeploying into risk
        → risk-on flow, bullish for equities

    Source: CoinGecko free public API (rate-limited but generous).

    Returns:
        {
          "usdt_cap": float,
          "usdc_cap": float,
          "total_cap": float,
          "change_24h_pct": float,  # weighted-average 24h delta
        }
        or None on failure.
    """
    now = time.time()
    if not force_refresh and "stables" in _STABLECOIN_CACHE:
        data, ts = _STABLECOIN_CACHE["stables"]
        if now - ts < _STABLECOIN_TTL:
            return dict(data)

    try:
        r = requests.get(
            "https://api.coingecko.com/api/v3/coins/markets",
            params={"vs_currency": "usd", "ids": "tether,usd-coin"},
            timeout=_HTTP_TIMEOUT,
        )
        r.raise_for_status()
        arr = r.json()
        if not isinstance(arr, list) or not arr:
            return None

        by_id = {c.get("id"): c for c in arr if isinstance(c, dict)}
        usdt = by_id.get("tether", {})
        usdc = by_id.get("usd-coin", {})
        usdt_cap = float(usdt.get("market_cap") or 0.0)
        usdc_cap = float(usdc.get("market_cap") or 0.0)
        total = usdt_cap + usdc_cap
        if total <= 0:
            return None

        # Weighted-average 24h change across the two
        usdt_chg = float(usdt.get("market_cap_change_percentage_24h") or 0.0)
        usdc_chg = float(usdc.get("market_cap_change_percentage_24h") or 0.0)
        change_24h_pct = (usdt_chg * usdt_cap + usdc_chg * usdc_cap) / total

        result = {
            "usdt_cap": usdt_cap,
            "usdc_cap": usdc_cap,
            "total_cap": total,
            "change_24h_pct": change_24h_pct,
        }
        _STABLECOIN_CACHE["stables"] = (result, now)
        return dict(result)
    except Exception as e:
        logger.debug("CoinGecko stablecoin fetch failed: %s", e)
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  Composite risk read — used by the equity regime detector
# ─────────────────────────────────────────────────────────────────────────────

def compute_crypto_risk_read(spy_closes: np.ndarray | None = None) -> dict[str, Any]:
    """
    Combine BTC/ETH trend, funding, stablecoin flow, and (optionally) BTC-vs-SPY
    correlation into a single risk read for the equity regime detector.

    Args:
        spy_closes: Optional trailing SPY daily closes (numpy array). When
                    provided, we compute rolling 20-day BTC-vs-SPY correlation
                    and flag correlation breaks.

    Returns a dict — all keys always present, values may be None on partial
    data-source failures. Never raises.

    Keys:
      btc_trend:                "UP" | "DOWN" | "NEUTRAL"
      eth_trend:                "UP" | "DOWN" | "NEUTRAL"
      btc_price:                float | None
      btc_spy_correlation:      float | None  (-1..1)
      correlation_broken:       bool
      funding_rate_annualized_pct: float | None
      funding_state:            "STRETCHED_LONG" | "STRETCHED_SHORT" | "NEUTRAL"
      stablecoin_flow:          "EXPANDING" | "CONTRACTING" | "STABLE"
      stablecoin_change_24h_pct: float | None
      risk_signal:              "CONFIRM_LONG" | "CONFIRM_SHORT" | "DIVERGE" | "STRETCHED" | "NEUTRAL"
      sources_ok:               dict[str, bool]  — which upstream sources returned data
    """
    out: dict[str, Any] = {
        "btc_trend": "NEUTRAL",
        "eth_trend": "NEUTRAL",
        "btc_price": None,
        "btc_spy_correlation": None,
        "correlation_broken": False,
        "funding_rate_annualized_pct": None,
        "funding_state": "NEUTRAL",
        "stablecoin_flow": "STABLE",
        "stablecoin_change_24h_pct": None,
        "risk_signal": "NEUTRAL",
        "sources_ok": {"btc": False, "eth": False, "funding": False, "stables": False},
    }

    # ── BTC trend + optional correlation
    btc = fetch_crypto_daily("BTC", bars=60)
    if btc is not None and len(btc) >= 20:
        out["sources_ok"]["btc"] = True
        closes = btc["close"].values.astype(float)
        price = float(closes[-1])
        ma20 = float(np.mean(closes[-20:]))
        ma50 = float(np.mean(closes[-50:])) if len(closes) >= 50 else ma20
        out["btc_price"] = price
        if price > ma20 > ma50:
            out["btc_trend"] = "UP"
        elif price < ma20 < ma50:
            out["btc_trend"] = "DOWN"

        if spy_closes is not None and len(spy_closes) >= 21 and len(closes) >= 21:
            # Align tail lengths — take last N of each
            n = min(len(closes), len(spy_closes), 40)
            btc_ret = np.diff(closes[-n:]) / closes[-n:-1]
            spy_ret = np.diff(spy_closes[-n:]) / spy_closes[-n:-1]
            m = min(len(btc_ret), len(spy_ret))
            if m >= 20:
                recent_corr = float(np.corrcoef(btc_ret[-20:], spy_ret[-20:])[0, 1])
                out["btc_spy_correlation"] = recent_corr
                if m >= 40:
                    prior_corr = float(np.corrcoef(btc_ret[-40:-20], spy_ret[-40:-20])[0, 1])
                    # Break = prior was positively correlated but current has flipped/collapsed
                    if prior_corr > 0.4 and recent_corr < 0.1:
                        out["correlation_broken"] = True

    # ── ETH trend
    eth = fetch_crypto_daily("ETH", bars=60)
    if eth is not None and len(eth) >= 20:
        out["sources_ok"]["eth"] = True
        e_closes = eth["close"].values.astype(float)
        e_price = float(e_closes[-1])
        e_ma20 = float(np.mean(e_closes[-20:]))
        e_ma50 = float(np.mean(e_closes[-50:])) if len(e_closes) >= 50 else e_ma20
        if e_price > e_ma20 > e_ma50:
            out["eth_trend"] = "UP"
        elif e_price < e_ma20 < e_ma50:
            out["eth_trend"] = "DOWN"

    # ── Funding rate
    funding = fetch_funding_rate("BTC")
    if funding is not None:
        out["sources_ok"]["funding"] = True
        out["funding_rate_annualized_pct"] = round(funding, 2)
        if funding > 40:
            out["funding_state"] = "STRETCHED_LONG"
        elif funding < -10:
            out["funding_state"] = "STRETCHED_SHORT"

    # ── Stablecoin flow
    stables = fetch_stablecoin_supply()
    if stables:
        out["sources_ok"]["stables"] = True
        chg = stables["change_24h_pct"]
        out["stablecoin_change_24h_pct"] = round(chg, 3)
        # Stablecoin cap EXPANDING = capital moving from risk → cash-equivalent
        if chg > 2.0:
            out["stablecoin_flow"] = "EXPANDING"
        elif chg < -2.0:
            out["stablecoin_flow"] = "CONTRACTING"

    # ── Composite risk signal
    #    Rules of thumb:
    #      Both crypto trending up + neutral funding + contracting stables → CONFIRM_LONG
    #      Both crypto trending down + expanding stables → CONFIRM_SHORT
    #      Funding stretched either way → STRETCHED (positioning warning)
    #      Trends diverge or correlation break → DIVERGE
    up_votes = sum(1 for t in (out["btc_trend"], out["eth_trend"]) if t == "UP")
    down_votes = sum(1 for t in (out["btc_trend"], out["eth_trend"]) if t == "DOWN")

    if out["funding_state"] in ("STRETCHED_LONG", "STRETCHED_SHORT"):
        out["risk_signal"] = "STRETCHED"
    elif out["correlation_broken"]:
        out["risk_signal"] = "DIVERGE"
    elif up_votes == 2 and out["stablecoin_flow"] != "EXPANDING":
        out["risk_signal"] = "CONFIRM_LONG"
    elif down_votes == 2 or (down_votes >= 1 and out["stablecoin_flow"] == "EXPANDING"):
        out["risk_signal"] = "CONFIRM_SHORT"

    return out


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ─────────────────────────────────────────────────────────────────────────────
#  Smoke test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    print("── fetch_crypto_daily('BTC', 60) ──")
    btc = fetch_crypto_daily("BTC", 60)
    if btc is not None:
        print(f"  {len(btc)} bars, latest close = ${btc['close'].iloc[-1]:,.0f}")
    else:
        print("  FAILED — no source returned data")

    print("── fetch_crypto_daily('ETH', 60) ──")
    eth = fetch_crypto_daily("ETH", 60)
    if eth is not None:
        print(f"  {len(eth)} bars, latest close = ${eth['close'].iloc[-1]:,.0f}")
    else:
        print("  FAILED")

    print("── fetch_funding_rate('BTC') ──")
    fr = fetch_funding_rate("BTC")
    print(f"  {fr:+.2f}% annualized" if fr is not None else "  FAILED")

    print("── fetch_stablecoin_supply() ──")
    s = fetch_stablecoin_supply()
    if s:
        print(f"  total_cap=${s['total_cap']/1e9:.1f}B  change_24h={s['change_24h_pct']:+.2f}%")
    else:
        print("  FAILED")

    print("── compute_crypto_risk_read() ──")
    read = compute_crypto_risk_read()
    for k, v in read.items():
        print(f"  {k}: {v}")
