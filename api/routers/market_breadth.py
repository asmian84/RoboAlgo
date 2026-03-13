"""
Market Breadth Indicators
GET /api/market/breadth   → VIX, McClellan Oscillator, Fear/Greed, Advance-Decline

McClellan Oscillator:
  ratio = (advances - declines) / (advances + declines)   [Ratio-Adjusted]
  MCO   = 19-day EMA(ratio) − 39-day EMA(ratio)  × 1000
  Summation Index = cumulative sum of MCO

Reference: https://chartschool.stockcharts.com/table-of-contents/market-indicators/mcclellan-oscillator
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter

router = APIRouter()

# Simple TTL cache — breadth data changes once per day
_cache: dict[str, Any] = {}
_cache_ts: float = 0.0
_CACHE_TTL = 15 * 60  # 15 minutes

# ── Large-cap proxy basket for advance/decline (replaces defunct ^NYADV/^NYDEC) ─
# ~100 stocks spread across all 11 GICS sectors for representative breadth
_BREADTH_TICKERS = [
    # Technology
    'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META', 'AMZN', 'AMD', 'INTC', 'CSCO', 'ORCL',
    'CRM', 'ADBE', 'QCOM', 'TXN', 'AVGO', 'AMAT', 'MU', 'NOW', 'INTU', 'PANW',
    # Financials
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'V', 'MA', 'AXP', 'BLK', 'SCHW',
    'USB', 'PNC', 'COF', 'CB', 'PRU', 'MET', 'MMC', 'AON', 'TFC', 'BK',
    # Healthcare
    'JNJ', 'UNH', 'LLY', 'PFE', 'ABBV', 'ABT', 'TMO', 'DHR', 'AMGN', 'BMY',
    'CVS', 'CI', 'ELV', 'ISRG', 'ZTS', 'REGN', 'GILD', 'BSX', 'VRTX', 'MDT',
    # Consumer Discretionary
    'HD', 'LOW', 'MCD', 'SBUX', 'NKE', 'TJX', 'BKNG', 'GM', 'F', 'ORLY',
    # Consumer Staples
    'WMT', 'COST', 'PG', 'KO', 'PEP', 'CL', 'MO', 'PM', 'MDLZ', 'STZ',
    # Energy
    'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'MPC', 'HAL', 'VLO', 'OXY', 'PSX',
    # Industrials
    'CAT', 'DE', 'BA', 'GE', 'HON', 'UPS', 'FDX', 'LMT', 'RTX', 'ETN',
    # Materials, Utilities, Real Estate, Communication
    'LIN', 'APD', 'NEE', 'DUK', 'SO', 'PLD', 'AMT', 'NFLX', 'DIS', 'CMCSA',
]


def _ema(series: list[float], period: int) -> list[float]:
    """Single-pass EMA without pandas dependency."""
    k = 2.0 / (period + 1)
    out = [series[0]]
    for v in series[1:]:
        out.append(v * k + out[-1] * (1 - k))
    return out


def _compute_mcclellan(adv: list[float], dec: list[float]) -> dict[str, Any]:
    """Compute Ratio-Adjusted McClellan Oscillator and Summation Index."""
    if len(adv) < 40 or len(dec) < 40:
        return {"mco": None, "summation": None, "mco_series": [], "sum_series": []}

    # Ratio-adjusted: avoid division by zero
    ratio = [
        (a - d) / (a + d) if (a + d) > 0 else 0.0
        for a, d in zip(adv, dec)
    ]
    ratio_1000 = [r * 1000 for r in ratio]  # scale to readable range

    ema19 = _ema(ratio_1000, 19)
    ema39 = _ema(ratio_1000, 39)
    mco   = [e19 - e39 for e19, e39 in zip(ema19, ema39)]

    # Summation Index = running total of MCO
    summation: list[float] = []
    total = 0.0
    for m in mco:
        total += m
        summation.append(total)

    return {
        "mco":        round(mco[-1], 2),
        "summation":  round(summation[-1], 2),
        "mco_series": [round(v, 2) for v in mco[-90:]],   # last 90 days for chart
        "sum_series": [round(v, 2) for v in summation[-90:]],
    }


@router.get("/breadth")
def get_market_breadth():
    """
    Market breadth data:
    - VIX level + direction
    - McClellan Oscillator (NYSE ratio-adjusted) + Summation Index
    - Simplified Fear/Greed composite score (VIX 50% + MCO 30% + SPY momentum 20%)
    """
    global _cache, _cache_ts

    now = time.time()
    if _cache and now - _cache_ts < _CACHE_TTL:
        return _cache

    try:
        import yfinance as yf
        import pandas as pd

        result: dict[str, Any] = {}

        # ── VIX ─────────────────────────────────────────────────────────────
        try:
            vix_hist = yf.Ticker("^VIX").history(period="5d")
            if not vix_hist.empty:
                vix_now   = float(vix_hist["Close"].iloc[-1])
                vix_prev  = float(vix_hist["Close"].iloc[-2]) if len(vix_hist) >= 2 else vix_now
                vix_chg   = vix_now - vix_prev
                result["vix"]           = round(vix_now, 2)
                result["vix_change"]    = round(vix_chg, 2)
                result["vix_direction"] = (
                    "extreme_fear" if vix_now >= 30 else
                    "fear"         if vix_now >= 20 else
                    "normal"       if vix_now >= 15 else
                    "complacency"
                )
            else:
                result["vix"] = None
                result["vix_direction"] = "unknown"
        except Exception:
            result["vix"] = None
            result["vix_direction"] = "unknown"

        # ── Advance-Decline proxy → McClellan Oscillator ─────────────────────
        # ^NYADV / ^NYDEC are delisted from Yahoo Finance; we derive A/D from
        # a curated 100-stock large-cap basket (one fast batch download).
        try:
            df_multi = yf.download(
                _BREADTH_TICKERS,
                period="8mo",
                auto_adjust=True,
                progress=False,
            )
            # yfinance ≥ 0.2 returns (metric, ticker) MultiIndex columns
            close_df = (
                df_multi["Close"]
                if isinstance(df_multi.columns, pd.MultiIndex)
                else df_multi
            )
            close_df = close_df.dropna(how="all")

            if len(close_df) >= 42:
                rets      = close_df.pct_change().dropna(how="all").iloc[1:]
                adv_list  = (rets > 0).sum(axis=1).astype(float).tolist()
                dec_list  = (rets < 0).sum(axis=1).astype(float).tolist()
                date_list = [str(d.date()) for d in rets.index]

                mco_data = _compute_mcclellan(adv_list, dec_list)
                result.update({
                    "mco":         mco_data["mco"],
                    "mco_sum":     mco_data["summation"],
                    "mco_series":  mco_data["mco_series"],
                    "sum_series":  mco_data["sum_series"],
                    "mco_dates":   date_list[-90:],
                    "mco_direction": (
                        "bullish" if mco_data["mco"] is not None and mco_data["mco"] > 0
                        else "bearish"
                    ),
                    "adv_latest":  round(adv_list[-1]) if adv_list else None,
                    "dec_latest":  round(dec_list[-1]) if dec_list else None,
                })
            else:
                result["mco"] = None
                result["mco_sum"] = None
                result["mco_direction"] = "unknown"
        except Exception:
            result["mco"] = None
            result["mco_sum"] = None
            result["mco_direction"] = "unknown"

        # ── SPY 125-day momentum ─────────────────────────────────────────────
        try:
            spy_hist = yf.Ticker("SPY").history(period="8mo")
            if len(spy_hist) >= 126:
                spy_now    = float(spy_hist["Close"].iloc[-1])
                spy_ma125  = float(spy_hist["Close"].iloc[-126:].mean())
                spy_mom    = (spy_now / spy_ma125 - 1) * 100
                result["spy_momentum"]  = round(spy_mom, 2)
                result["spy_above_ma"]  = spy_now > spy_ma125
            else:
                result["spy_momentum"] = None
        except Exception:
            result["spy_momentum"] = None

        # ── Fear / Greed — CNN official index (real data) ────────────────────
        try:
            import httpx as _httpx
            _headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                "Accept": "application/json",
            }
            with _httpx.Client(timeout=5, headers=_headers) as _hc:
                _r = _hc.get("https://production.dataviz.cnn.io/index/fearandgreed/graphdata")
            _r.raise_for_status()
            _data = _r.json()
            fg = float(_data["fear_and_greed"]["score"])
            result["fear_greed"] = int(round(fg))
            result["fear_greed_label"] = (
                "Extreme Fear"  if fg < 20 else
                "Fear"          if fg < 40 else
                "Neutral"       if fg < 60 else
                "Greed"         if fg < 80 else
                "Extreme Greed"
            )
            result["fear_greed_direction"] = "bearish" if fg < 45 else "bullish" if fg > 55 else "neutral"
        except Exception:
            # Fallback: synthetic composite (VIX 50% + MCO 30% + SPY mom 20%)
            try:
                scores: list[tuple[float, float]] = []
                vix_val = result.get("vix")
                if vix_val is not None:
                    v_score = max(0.0, min(100.0, 100.0 - (vix_val - 10.0) * 2.0))
                    scores.append((v_score, 0.50))
                mco_val = result.get("mco")
                if mco_val is not None:
                    m_score = max(0.0, min(100.0, 50.0 + mco_val))
                    scores.append((m_score, 0.30))
                spy_mom = result.get("spy_momentum")
                if spy_mom is not None:
                    s_score = max(0.0, min(100.0, 50.0 + spy_mom * 10.0))
                    scores.append((s_score, 0.20))
                if scores:
                    total_weight = sum(w for _, w in scores)
                    fg = sum(v * w for v, w in scores) / total_weight
                    result["fear_greed"] = int(round(fg))
                    result["fear_greed_label"] = (
                        "Extreme Fear"  if fg < 20 else
                        "Fear"          if fg < 40 else
                        "Neutral"       if fg < 60 else
                        "Greed"         if fg < 80 else
                        "Extreme Greed"
                    )
                    result["fear_greed_direction"] = "bearish" if fg < 45 else "bullish" if fg > 55 else "neutral"
                else:
                    result["fear_greed"] = None
                    result["fear_greed_label"] = "N/A"
            except Exception:
                result["fear_greed"] = None
                result["fear_greed_label"] = "N/A"

        _cache = result
        _cache_ts = now
        return result

    except Exception as e:
        return {"error": str(e)}
