"""Options data router — powered by Alpha Vantage."""
from __future__ import annotations

import os
import time
from typing import Any

import httpx
from fastapi import APIRouter

AV_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "UPSSIQEAMIJFI6RS")
AV_BASE = "https://www.alphavantage.co/query"

router = APIRouter()

# In-memory cache: symbol → (data, timestamp)
_options_cache: dict[str, tuple[dict, float]] = {}
_OPTIONS_TTL = 900  # 15 minutes


async def _fetch_options(symbol: str) -> dict[str, Any]:
    """Fetch and analyze options data for a symbol (async httpx)."""
    now = time.time()
    cached = _options_cache.get(symbol.upper())
    if cached and now - cached[1] < _OPTIONS_TTL:
        return cached[0]

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            # Try realtime options first
            r = await client.get(
                AV_BASE,
                params={"function": "REALTIME_OPTIONS", "symbol": symbol.upper(), "apikey": AV_API_KEY},
            )
            data = r.json()

            # Fallback to historical if realtime not available
            if "data" not in data or not data["data"]:
                r2 = await client.get(
                    AV_BASE,
                    params={"function": "HISTORICAL_OPTIONS", "symbol": symbol.upper(), "apikey": AV_API_KEY},
                )
                data = r2.json()

        if "Note" in data or "Information" in data:
            return {"error": "rate_limited", "symbol": symbol}

        options = data.get("data", [])
        if not options:
            return {"error": "no_data", "symbol": symbol}

        # ── Parse options ──────────────────────────────────────────────────
        calls = []
        puts = []

        for opt in options:
            try:
                contract = {
                    "contract_id": opt.get("contractID", ""),
                    "expiration": opt.get("expiration", ""),
                    "strike": float(opt.get("strike", 0)),
                    "type": opt.get("type", "").lower(),  # "call" or "put"
                    "last": float(opt.get("last", 0) or 0),
                    "mark": float(opt.get("mark", opt.get("last", 0)) or 0),
                    "bid": float(opt.get("bid", 0) or 0),
                    "ask": float(opt.get("ask", 0) or 0),
                    "volume": int(float(opt.get("volume", 0) or 0)),
                    "open_interest": int(float(opt.get("open_interest", 0) or 0)),
                    "implied_volatility": float(opt.get("implied_volatility", 0) or 0),
                    "delta": float(opt.get("delta", 0) or 0),
                    "gamma": float(opt.get("gamma", 0) or 0),
                    "theta": float(opt.get("theta", 0) or 0),
                    "vega": float(opt.get("vega", 0) or 0),
                }
                if contract["type"] == "call":
                    calls.append(contract)
                elif contract["type"] == "put":
                    puts.append(contract)
            except (ValueError, TypeError):
                continue

        if not calls and not puts:
            return {"error": "no_valid_options", "symbol": symbol}

        # ── Analytics ──────────────────────────────────────────────────────

        # Put/Call Ratio (volume-based)
        total_call_vol = sum(c["volume"] for c in calls)
        total_put_vol = sum(p["volume"] for p in puts)
        pc_ratio = round(total_put_vol / max(total_call_vol, 1), 3)

        # Put/Call Ratio (open interest)
        total_call_oi = sum(c["open_interest"] for c in calls)
        total_put_oi = sum(p["open_interest"] for p in puts)
        pc_ratio_oi = round(total_put_oi / max(total_call_oi, 1), 3)

        # PC ratio interpretation
        if pc_ratio > 1.3:
            pc_signal = "bearish"  # heavy put buying = fear
        elif pc_ratio < 0.7:
            pc_signal = "bullish"  # heavy call buying = complacency or bullish
        else:
            pc_signal = "neutral"

        # Max Pain (strike where total options value is minimized for holders at expiration)
        # Get the nearest expiration
        all_expirations = sorted(set(o["expiration"] for o in calls + puts))
        near_exp = all_expirations[0] if all_expirations else None

        max_pain_strike = None
        if near_exp:
            near_calls = [c for c in calls if c["expiration"] == near_exp]
            near_puts = [p for p in puts if p["expiration"] == near_exp]
            all_strikes = sorted(set(c["strike"] for c in near_calls + near_puts))

            min_pain = float("inf")
            for test_strike in all_strikes:
                call_pain = sum(
                    max(0, test_strike - c["strike"]) * c["open_interest"]
                    for c in near_calls
                )
                put_pain = sum(
                    max(0, p["strike"] - test_strike) * p["open_interest"]
                    for p in near_puts
                )
                total_pain = call_pain + put_pain
                if total_pain < min_pain:
                    min_pain = total_pain
                    max_pain_strike = test_strike

        # Unusual Options Activity (volume > 3× open interest)
        unusual = []
        for opt in calls + puts:
            if opt["open_interest"] > 0 and opt["volume"] > opt["open_interest"] * 3 and opt["volume"] > 100:
                unusual.append({
                    "type": opt["type"],
                    "strike": opt["strike"],
                    "expiration": opt["expiration"],
                    "volume": opt["volume"],
                    "open_interest": opt["open_interest"],
                    "vol_oi_ratio": round(opt["volume"] / max(opt["open_interest"], 1), 2),
                    "implied_volatility": opt["implied_volatility"],
                    "delta": opt["delta"],
                })
        unusual.sort(key=lambda x: x["vol_oi_ratio"], reverse=True)
        unusual = unusual[:10]  # top 10

        # IV Skew (near-term puts vs calls IV comparison)
        if near_exp:
            near_calls_iv = [c["implied_volatility"] for c in near_calls if c["implied_volatility"] > 0]
            near_puts_iv = [p["implied_volatility"] for p in near_puts if p["implied_volatility"] > 0]
            avg_call_iv = sum(near_calls_iv) / len(near_calls_iv) if near_calls_iv else 0
            avg_put_iv = sum(near_puts_iv) / len(near_puts_iv) if near_puts_iv else 0
            iv_skew = round(avg_put_iv - avg_call_iv, 4)
            # Positive skew = puts more expensive = bearish fear
            # Negative skew = calls more expensive = bullish demand
        else:
            iv_skew = 0
            avg_call_iv = 0
            avg_put_iv = 0

        # Top strikes by open interest (key levels)
        all_options_by_oi = sorted(calls + puts, key=lambda x: x["open_interest"], reverse=True)[:20]
        key_strikes = sorted(set(o["strike"] for o in all_options_by_oi))

        # Highest volume calls and puts
        top_calls = sorted(calls, key=lambda x: x["volume"], reverse=True)[:5]
        top_puts = sorted(puts, key=lambda x: x["volume"], reverse=True)[:5]

        result = {
            "symbol": symbol.upper(),
            "put_call_ratio": pc_ratio,
            "put_call_ratio_oi": pc_ratio_oi,
            "pc_signal": pc_signal,
            "total_call_volume": total_call_vol,
            "total_put_volume": total_put_vol,
            "total_call_oi": total_call_oi,
            "total_put_oi": total_put_oi,
            "max_pain": max_pain_strike,
            "nearest_expiration": near_exp,
            "iv_skew": iv_skew,
            "avg_call_iv": round(avg_call_iv, 4),
            "avg_put_iv": round(avg_put_iv, 4),
            "key_strikes": key_strikes[:15],  # most important price levels
            "unusual_activity": unusual,
            "top_calls": top_calls,
            "top_puts": top_puts,
            "expirations": all_expirations[:8],
            "total_contracts": len(calls) + len(puts),
        }

        _options_cache[symbol.upper()] = (result, now)
        return result

    except Exception as e:
        return {"error": str(e), "symbol": symbol}


@router.get("/{symbol}")
async def get_options_data(symbol: str):
    """Get options analytics for a symbol from Alpha Vantage."""
    return await _fetch_options(symbol.upper())


@router.get("/{symbol}/chain")
async def get_options_chain(symbol: str, expiration: str = ""):
    """Get full options chain for a symbol, optionally filtered by expiration."""
    data = await _fetch_options(symbol.upper())
    if "error" in data:
        return data

    # Return the full chain data
    return {
        "symbol": symbol.upper(),
        "expiration": expiration or data.get("nearest_expiration"),
        "put_call_ratio": data.get("put_call_ratio"),
        "max_pain": data.get("max_pain"),
        "key_strikes": data.get("key_strikes", []),
        "unusual_activity": data.get("unusual_activity", []),
    }
