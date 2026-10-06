"""
Crypto trade-signal scanner.

Parallel to the equity `scanner.py` but crypto-native:
  • Universe    = top-N by market cap (CoinGecko) with a hardcoded fallback list
  • Data feed   = data_engine.crypto_fetcher (24/7, keyless public APIs)
  • Regime      = BTC-driven (not SPY); reuses compute_crypto_risk_read
  • Scoring     = simplified trend/momentum/volatility composite tuned for crypto
                  volatility (equity scoring assumes tighter ranges).

Emits the same SSE event shapes as scan_universe_streaming so the frontend
can share the TradesPage rendering code with a different endpoint URL.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Generator

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Universe ────────────────────────────────────────────────────────────────

# Hardcoded fallback — top majors by long-term relevance. Used when CoinGecko
# is unreachable so scans never fail on universe resolution.
_FALLBACK_UNIVERSE = [
    "BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "AVAX", "LINK",
    "DOT", "MATIC", "LTC", "BCH", "ATOM", "UNI", "ETC", "XLM",
    "FIL", "APT", "ARB", "OP", "INJ", "SUI", "NEAR", "AAVE",
    "MKR", "IMX", "GRT", "SAND", "MANA", "AXS",
]


def get_crypto_universe(limit: int = 30) -> list[str]:
    """
    Return top-N crypto tickers by market cap.

    Prefers CoinGecko for a fresh top-N list; falls back to the hardcoded
    majors when the API is unreachable. Returns short-form tickers (BTC, ETH…)
    matching what crypto_fetcher understands.
    """
    try:
        import requests
        # Over-fetch: filters below drop stablecoins/wrapped/malformed tickers.
        r = requests.get(
            "https://api.coingecko.com/api/v3/coins/markets",
            params={"vs_currency": "usd", "order": "market_cap_desc",
                    "per_page": min(max(limit * 2, 30), 100), "page": 1},
            timeout=5,
        )
        r.raise_for_status()
        arr = r.json()
        if isinstance(arr, list) and arr:
            # Drop stablecoins (flat by design) + wrapped/staked derivatives
            # (redundant with underlying) so signals surface something tradeable.
            skip_ids = {
                "tether", "usd-coin", "dai", "true-usd", "first-digital-usd",
                "ethena-usde", "usds", "paypal-usd", "gemini-dollar", "frax",
                "wrapped-bitcoin", "wrapped-steth", "staked-ether", "lido-staked-ether",
                "coinbase-wrapped-btc", "renbtc", "wrapped-eeth",
            }
            skip_symbols = {"USDT", "USDC", "DAI", "TUSD", "FDUSD", "USDE", "USDS",
                            "PYUSD", "GUSD", "FRAX", "WBTC", "WSTETH", "STETH",
                            "CBBTC", "WEETH"}
            out: list[str] = []
            for c in arr:
                if c.get("id") in skip_ids:
                    continue
                sym = str(c.get("symbol", "")).upper()
                if not sym or sym in skip_symbols or sym in out:
                    continue
                # Filter tickers with symbols that break venue pair formats
                # (underscores, dots, or > 6 chars are almost always exchange-
                # specific derivatives, not spot-listed on Coinbase/Binance).
                if "_" in sym or "." in sym or len(sym) > 6:
                    continue
                out.append(sym)
                if len(out) >= limit:
                    break
            if out:
                return out
    except Exception as exc:
        logger.debug("CoinGecko universe fetch failed: %s — using fallback", exc)

    return list(_FALLBACK_UNIVERSE[:limit])


# ── Scoring ─────────────────────────────────────────────────────────────────

def _score_symbol(symbol: str, bars: pd.DataFrame, funding_state: str) -> dict[str, Any]:
    """
    Compute a simplified crypto trade signal from daily bars + funding context.

    Returns a TradeRec-shaped dict compatible with the frontend TradeRec type.
    """
    closes = bars["close"].values.astype(float)
    highs  = bars["high"].values.astype(float)
    lows   = bars["low"].values.astype(float)
    price  = float(closes[-1])

    ma20 = float(np.mean(closes[-20:]))
    ma50 = float(np.mean(closes[-50:])) if len(closes) >= 50 else ma20

    # ── Trend (0-100)
    if price > ma20 > ma50:
        trend_score, direction = 90.0, "LONG"
    elif price > ma20 and ma20 <= ma50:
        trend_score, direction = 65.0, "LONG"
    elif price < ma20 < ma50:
        trend_score, direction = 90.0, "SHORT"
    elif price < ma20 and ma20 >= ma50:
        trend_score, direction = 65.0, "SHORT"
    else:
        trend_score, direction = 40.0, "LONG" if price >= ma20 else "SHORT"

    # ── Momentum: 7-day return normalized. ±5% mapped to 0-100.
    ret_7d = (price / float(closes[-8]) - 1.0) if len(closes) >= 8 else 0.0
    momentum_raw = ret_7d * 100 if direction == "LONG" else -ret_7d * 100
    momentum_score = max(0.0, min(100.0, 50.0 + momentum_raw * 10))  # ±5% → 0/100

    # ── ATR + volatility bucket
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])),
    )
    atr = float(np.mean(tr[-14:])) if len(tr) >= 14 else float(np.mean(tr) if len(tr) else 0.0)
    atr_pct = (atr / price * 100) if price else 0.0
    # Sweet spot for setups is 2-6% daily ATR. Score peaks there, drops at edges.
    if 2.0 <= atr_pct <= 6.0:
        vol_score = 85.0
    elif 1.0 <= atr_pct < 2.0 or 6.0 < atr_pct <= 10.0:
        vol_score = 65.0
    else:
        vol_score = 40.0

    # ── Funding alignment: stretched long funding penalizes long setups
    funding_score = 60.0
    if funding_state == "STRETCHED_LONG":
        funding_score = 30.0 if direction == "LONG" else 80.0
    elif funding_state == "STRETCHED_SHORT":
        funding_score = 30.0 if direction == "SHORT" else 80.0

    # ── Composite (weighted average)
    components = {
        "trend":     round(trend_score, 1),
        "momentum":  round(momentum_score, 1),
        "volatility": round(vol_score, 1),
        "funding":   round(funding_score, 1),
    }
    score = (
        trend_score * 0.35 +
        momentum_score * 0.25 +
        vol_score * 0.20 +
        funding_score * 0.20
    )
    score = round(score, 1)

    # ── Verdict
    if score >= 70:
        verdict = "BUY" if direction == "LONG" else "SELL"
    elif score >= 55:
        verdict = "HOLD"
    else:
        verdict = "NO_TRADE"

    # ── Confidence label
    if score >= 80:
        confidence = "HIGH"
    elif score >= 65:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    # ── Levels (ATR-based; 1.5× stop, 3× target for 2:1 R:R)
    stop_dist   = 1.5 * atr
    target_dist = 3.0 * atr
    if direction == "LONG":
        stop, target = price - stop_dist, price + target_dist
    else:
        stop, target = price + stop_dist, price - target_dist
    rr = round((target_dist / stop_dist) if stop_dist else 0.0, 2)

    entry_notes: list[str] = []
    if funding_state == "STRETCHED_LONG":
        entry_notes.append("Perp funding stretched long — reduce size or wait for reset")
    elif funding_state == "STRETCHED_SHORT":
        entry_notes.append("Perp funding stretched short — squeeze risk (mean-reversion)")
    if trend_score >= 85:
        entry_notes.append(f"Strong {direction.lower()} trend (price/MA20/MA50 aligned)")
    if atr_pct > 8:
        entry_notes.append(f"High volatility (ATR {atr_pct:.1f}% of price) — widen stops")

    return {
        "symbol":        symbol,
        "current_price": price,
        "atr":           atr,
        "atr_pct":       round(atr_pct, 2),
        "direction":     direction,
        "score":         score,
        "confidence":    confidence,
        "verdict":       verdict,
        "entry":         round(price, 4),
        "stop":          round(stop, 4),
        "target":        round(target, 4),
        "risk_reward":   rr,
        "components":    components,
        "entry_notes":   entry_notes,
        "rejection_reason": "" if verdict != "NO_TRADE" else f"Score {score} below 55 threshold",
    }


def _position_from_rec(rec: dict[str, Any], account_size: float, risk_pct: float = 1.0) -> dict[str, Any]:
    """Sizing: risk 1% of account per trade; position = risk / (entry - stop)."""
    entry = rec["entry"]
    stop  = rec["stop"]
    stop_dist = abs(entry - stop)
    risk_amount = account_size * (risk_pct / 100.0)
    if stop_dist <= 0:
        rec.update({"position_size": 0.0, "position_value": 0.0, "risk_amount": risk_amount,
                    "projected_profit": 0.0, "projected_profit_pct": 0.0})
        return rec
    position_size  = risk_amount / stop_dist
    position_value = position_size * entry
    projected_profit = abs(rec["target"] - entry) * position_size
    projected_pct    = (projected_profit / account_size * 100) if account_size else 0.0
    rec.update({
        "position_size":         round(position_size, 6),
        "position_value":        round(position_value, 2),
        "risk_amount":           round(risk_amount, 2),
        "projected_profit":      round(projected_profit, 2),
        "projected_profit_pct":  round(projected_pct, 2),
        "profile":               "SWING",
        "level_source":          "atr",
        "component_summary":     ", ".join(f"{k}={v}" for k, v in rec["components"].items()),
        "bonuses":               [],
    })
    return rec


# ── Regime ─────────────────────────────────────────────────────────────────

def _detect_crypto_regime() -> dict[str, Any]:
    """
    Crypto-native regime derived from BTC trend + funding + stablecoin flow.

    Same output keys as the equity `_detect_market_regime` where possible so
    the frontend can render a single RegimeInfo component for both scanners.
    """
    from data_engine.crypto_fetcher import compute_crypto_risk_read

    result: dict[str, Any] = {
        "regime": "UNKNOWN",
        "bias":   "NEUTRAL",
        "hedges": [],  # spot-only crypto — no inverse tokens surface here
        "btc_trend": "NEUTRAL",
        "eth_trend": "NEUTRAL",
        "modifiers": [],
    }
    try:
        crypto = compute_crypto_risk_read()
        result["crypto"]    = crypto
        result["btc_trend"] = crypto.get("btc_trend", "NEUTRAL")
        result["eth_trend"] = crypto.get("eth_trend", "NEUTRAL")

        risk = crypto.get("risk_signal", "NEUTRAL")
        if risk == "CONFIRM_LONG":
            result["regime"], result["bias"] = "BULL", "LONG"
        elif risk == "CONFIRM_SHORT":
            result["regime"], result["bias"] = "BEAR", "SHORT"
        elif risk == "STRETCHED":
            result["regime"], result["bias"] = "STRETCHED", "NEUTRAL"
            result["modifiers"].append(
                f"funding stretched — {crypto.get('funding_state', '')} "
                f"at {crypto.get('funding_rate_annualized_pct', 0):.1f}%"
            )
        elif risk == "DIVERGE":
            result["regime"], result["bias"] = "DIVERGE", "NEUTRAL"
            result["modifiers"].append("BTC-SPY correlation broken — sentiment shift")
        else:
            # Base on BTC trend alone
            bt = result["btc_trend"]
            if bt == "UP":
                result["regime"], result["bias"] = "BULL", "LONG"
            elif bt == "DOWN":
                result["regime"], result["bias"] = "BEAR", "SHORT"
            else:
                result["regime"], result["bias"] = "RANGE", "NEUTRAL"
        logger.info(
            "Crypto regime: %s bias=%s BTC=%s ETH=%s funding=%s risk=%s",
            result["regime"], result["bias"], result["btc_trend"],
            result["eth_trend"], crypto.get("funding_state"), risk,
        )
    except Exception as e:
        logger.warning("Crypto regime detection failed: %s", e)
    return result


# ── Public entry points ────────────────────────────────────────────────────

def scan_crypto_universe(
    symbols: list[str],
    profile: str = "SWING",
    account_size: float = 10000.0,
) -> dict[str, Any]:
    """Non-streaming crypto scan — returns full result dict."""
    from data_engine.crypto_fetcher import fetch_crypto_daily, fetch_funding_rate

    start = time.time()
    regime = _detect_crypto_regime()
    funding_state = regime.get("crypto", {}).get("funding_state", "NEUTRAL")

    recommendations: list[dict] = []
    errors: list[dict] = []

    for sym in symbols:
        try:
            bars = fetch_crypto_daily(sym, bars=60)
            if bars is None or len(bars) < 20:
                errors.append({"symbol": sym, "error": "insufficient data"})
                continue
            # Per-symbol funding (majors only — falls back to BTC state)
            sym_funding = funding_state
            if sym in ("ETH",):
                fr = fetch_funding_rate(sym)
                if fr is not None:
                    if fr > 40: sym_funding = "STRETCHED_LONG"
                    elif fr < -10: sym_funding = "STRETCHED_SHORT"
                    else: sym_funding = "NEUTRAL"

            rec = _score_symbol(sym, bars, sym_funding)
            _position_from_rec(rec, account_size)
            recommendations.append(rec)
        except Exception as e:
            logger.warning("Crypto scoring failed for %s: %s", sym, e)
            errors.append({"symbol": sym, "error": str(e)})

    recommendations.sort(key=lambda r: r["score"], reverse=True)
    top_trades = [r for r in recommendations if r["verdict"] in ("BUY", "SELL")]
    watchlist  = [r for r in recommendations if r["verdict"] == "HOLD"]
    no_trade   = [r for r in recommendations if r["verdict"] == "NO_TRADE"]

    return {
        "scan_time":       round(time.time() - start, 2),
        "symbols_scanned": len(symbols),
        "profile":         profile,
        "account_size":    account_size,
        "regime":          regime,
        "recommendations": recommendations,
        "top_trades":      top_trades,
        "watchlist":       watchlist,
        "no_trade":        no_trade,
        "errors":          errors,
    }


def stream_scan_crypto(
    symbols: list[str],
    profile: str = "SWING",
    account_size: float = 10000.0,
) -> Generator[dict[str, Any], None, None]:
    """
    SSE-friendly generator. Emits the same event types as the equity streamer:
      regime → discovery → progress[N] → trade[?] → result
    """
    from data_engine.crypto_fetcher import fetch_crypto_daily, fetch_funding_rate

    start = time.time()

    regime = _detect_crypto_regime()
    yield {"type": "regime", **regime}

    yield {
        "type": "discovery",
        "movers_found": 0,      # crypto skips full-market pre-screen for now
        "hedges_added": 0,
        "total_symbols": len(symbols),
        "regime": regime["regime"],
        "bias":   regime["bias"],
    }

    funding_state_default = regime.get("crypto", {}).get("funding_state", "NEUTRAL")
    total = len(symbols)
    recommendations: list[dict] = []
    errors: list[dict] = []

    for i, sym in enumerate(symbols):
        try:
            bars = fetch_crypto_daily(sym, bars=60)
            if bars is None or len(bars) < 20:
                errors.append({"symbol": sym, "error": "insufficient data"})
                yield {"type": "progress", "completed": i + 1, "total": total,
                       "symbol": sym, "verdict": "ERROR", "score": 0}
                continue

            sym_funding = funding_state_default
            if sym in ("ETH",):
                fr = fetch_funding_rate(sym)
                if fr is not None:
                    if fr > 40: sym_funding = "STRETCHED_LONG"
                    elif fr < -10: sym_funding = "STRETCHED_SHORT"
                    else: sym_funding = "NEUTRAL"

            rec = _score_symbol(sym, bars, sym_funding)
            _position_from_rec(rec, account_size)
            recommendations.append(rec)

            yield {"type": "progress", "completed": i + 1, "total": total,
                   "symbol": sym, "verdict": rec["verdict"], "score": rec["score"]}
            if rec["verdict"] in ("BUY", "SELL"):
                yield {"type": "trade", "trade": rec}
        except Exception as e:
            logger.warning("Crypto scoring failed for %s: %s", sym, e)
            errors.append({"symbol": sym, "error": str(e)})
            yield {"type": "progress", "completed": i + 1, "total": total,
                   "symbol": sym, "verdict": "ERROR", "score": 0}

    recommendations.sort(key=lambda r: r["score"], reverse=True)
    top_trades = [r for r in recommendations if r["verdict"] in ("BUY", "SELL")]
    watchlist  = [r for r in recommendations if r["verdict"] == "HOLD"]
    no_trade   = [r for r in recommendations if r["verdict"] == "NO_TRADE"]

    yield {
        "type":            "result",
        "scan_time":       round(time.time() - start, 2),
        "symbols_scanned": total,
        "profile":         profile,
        "account_size":    account_size,
        "regime":          regime,
        "recommendations": recommendations,
        "top_trades":      top_trades,
        "watchlist":       watchlist,
        "no_trade":        no_trade,
        "errors":          errors,
    }
