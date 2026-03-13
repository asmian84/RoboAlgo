"""
Sniper Entry Scanner — Find absolute bottoms with capitulation confirmation
Returns TWO lists:
  1. Negative Gamma Entries (options-aware: shorts trapped)
  2. Technical Entries (pure price action: RSI/MACD capitulation)
"""

from fastapi import APIRouter, Query, BackgroundTasks
import logging
import asyncio
import time
from typing import Optional

router = APIRouter(tags=["sniper_entry"])
logger = logging.getLogger(__name__)

# Global cache
_last_scan: dict = {"negative_gamma": [], "technical": []}
_last_scan_time: float = 0
_scan_in_progress: bool = False
_SCAN_TTL = 5 * 60  # 5 min cache


def _run_sniper_scan(
    top_n: int = 300,
    min_volume: float = 100_000,
) -> dict:
    """
    Find stocks at absolute bottoms with capitulation confirmation.
    Returns both negative gamma + technical entries.
    """
    from data_engine.alphavantage_fetcher import load_ohlcv_from_db
    from config.settings import get_all_instruments
    import numpy as np

    syms = get_all_instruments()
    logger.info("Sniper entry scan: loading price data for %d symbols …", len(syms))

    price_data = load_ohlcv_from_db(syms, lookback_days=400)

    neg_gamma_entries = []
    technical_entries = []

    for sym, df in price_data.items():
        if df is None or len(df) < 30:
            continue

        try:
            close = df['close'].values
            volume = df['volume'].values
            high = df['high'].values
            low = df['low'].values

            if volume[-1] < min_volume:
                continue

            current_price = close[-1]
            if current_price <= 0:
                continue

            n = len(close)

            # ── WILDER'S RSI (proper calculation) ──
            delta = np.diff(close)
            gain = np.where(delta > 0, delta, 0.0)
            loss = np.where(delta < 0, -delta, 0.0)
            period = 14
            if len(gain) >= period:
                avg_gain = np.mean(gain[:period])
                avg_loss = np.mean(loss[:period])
                for g, l in zip(gain[period:], loss[period:]):
                    avg_gain = (avg_gain * (period - 1) + g) / period
                    avg_loss = (avg_loss * (period - 1) + l) / period
                rsi = 100 - (100 / (1 + avg_gain / avg_loss)) if avg_loss > 0 else 100.0
            else:
                rsi = 50.0

            # ── SUPPORT / LOCAL LOW ──
            bars_back = min(60, n)
            bars_back_200 = min(n, n)
            low_60  = np.min(low[-bars_back:])
            low_all = np.min(low)
            distance_from_60_low  = ((current_price - low_60)  / low_60  * 100) if low_60  > 0 else 0
            distance_from_all_low = ((current_price - low_all) / low_all * 100) if low_all > 0 else 0

            # Proximity buckets: at_low (<10%), near_low (10-25%), bouncing (25-40%)
            at_local_low   = distance_from_60_low  < 10
            near_52w_low   = distance_from_all_low < 15
            bouncing_off   = distance_from_all_low < 40

            if not bouncing_off:
                continue

            # ── OVERSOLD FLAGS ──
            rsi_extreme  = rsi < 25
            rsi_oversold = rsi < 35
            rsi_weak     = rsi < 45

            # ── MACD (EMA-based) ──
            def ema(arr, span):
                k = 2 / (span + 1)
                e = arr[0]
                for v in arr[1:]:
                    e = v * k + e * (1 - k)
                return e

            if n >= 26:
                ema12 = ema(close, 12)
                ema26 = ema(close, 26)
                macd_val = ema12 - ema26
                # Signal: 9-period EMA of MACD (approximate)
                macd_series = [ema(close[:i], 12) - ema(close[:i], 26) for i in range(26, n+1)]
                signal_val = ema(np.array(macd_series), 9) if len(macd_series) >= 9 else macd_val
                macd_below_signal = macd_val < signal_val
                macd_bouncing = close[-1] > close[-5] if n >= 5 else False
                macd_histogram_rising = (macd_val - signal_val) > (macd_series[-2] - signal_val) if len(macd_series) >= 2 else False
            else:
                macd_below_signal = False
                macd_bouncing = close[-1] > close[-3] if n >= 3 else False
                macd_histogram_rising = False

            # ── VOLUME CAPITULATION ──
            recent_vol   = np.mean(volume[-3:])
            vol_20       = np.mean(volume[-min(20, n):])
            vol_60       = np.mean(volume[-min(60, n):])
            volume_dried_up    = recent_vol < vol_20 * 0.75
            volume_spike_down  = recent_vol > vol_20 * 1.5   # big sell volume = capitulation
            vol_accumulating   = recent_vol > vol_20 * 1.2   # rising volume = accumulation

            # ── BB WIDTH (volatility squeeze) ──
            sma20 = np.mean(close[-min(20, n):])
            std20 = np.std(close[-min(20, n):])
            bb_pct = (4 * std20 / sma20 * 100) if sma20 > 0 else 100
            volatility_compressed = bb_pct < 20  # loosened — high-vol market

            # ── ENTRY SCORE (0-100) ──
            # Support score: closer to low = better
            if at_local_low:
                support_score = 100 - distance_from_60_low * 5
            elif near_52w_low:
                support_score = 75 - distance_from_all_low * 2
            else:
                support_score = max(0, 50 - distance_from_all_low)

            # Capitulation score: RSI oversold + volume signals
            capitulation_score = (
                (35 if rsi_extreme else 20 if rsi_oversold else 10 if rsi_weak else 0) +
                (20 if volume_dried_up else 0) +
                (15 if volume_spike_down else 0) +
                (15 if macd_below_signal else 0) +
                (10 if not macd_bouncing else 0)  # Still falling = setup building
            )

            # Bounce score: early signs of reversal
            bounce_score = (
                (25 if macd_bouncing else 0) +
                (20 if vol_accumulating else 0) +
                (20 if macd_histogram_rising else 0) +
                (15 if rsi > 30 and rsi_oversold else 0)   # RSI recovering from oversold
            )

            entry_quality = (
                support_score     * 0.30 +
                capitulation_score * 0.45 +
                bounce_score      * 0.25
            )

            # Minimum bar: must have SOME oversold + near a low
            if capitulation_score < 10 or entry_quality < 15:
                continue

            # ── NEGATIVE GAMMA PROXY ──
            negative_gamma_proxy = (
                (at_local_low or near_52w_low) and
                rsi_oversold and
                volatility_compressed
            )

            # ── Build result ──
            result = {
                'symbol': sym,
                'current_price': float(current_price),
                'entry_quality': float(entry_quality),
                'support_score': float(support_score),
                'capitulation_score': float(capitulation_score),
                'bounce_score': float(bounce_score),
                'rsi': float(rsi),
                'distance_from_60_low_pct': float(distance_from_60_low),
                'distance_from_all_low_pct': float(distance_from_all_low),
                'volume_dried_up': bool(volume_dried_up),
                'macd_bouncing': bool(macd_bouncing),
                'bb_width_pct': float(bb_pct),
                'volume_dried_up': bool(volume_dried_up),
                'macd_bouncing': bool(macd_bouncing),
                'notes': [
                    f"RSI: {rsi:.0f}{'  ⚡ extreme oversold' if rsi_extreme else '  oversold' if rsi_oversold else ''}",
                    f"Price: {distance_from_60_low:.1f}% from 60d low  /  {distance_from_all_low:.1f}% from all-time low",
                    f"Cap: {capitulation_score:.0f}  Bounce: {bounce_score:.0f}  Quality: {entry_quality:.0f}",
                ]
            }

            if negative_gamma_proxy:
                result['entry_type'] = 'negative_gamma'
                neg_gamma_entries.append(result)
            else:
                result['entry_type'] = 'technical'
                technical_entries.append(result)

        except Exception as e:
            logger.debug("Sniper scan error for %s: %s", sym, e)
            continue

    # Sort both lists by entry quality
    neg_gamma_entries.sort(key=lambda x: x['entry_quality'], reverse=True)
    technical_entries.sort(key=lambda x: x['entry_quality'], reverse=True)

    # Rank
    for i, r in enumerate(neg_gamma_entries[:top_n], 1):
        r['rank'] = i
    for i, r in enumerate(technical_entries[:top_n], 1):
        r['rank'] = i

    logger.info(
        "Sniper scan: found %d negative gamma + %d technical entries",
        len(neg_gamma_entries),
        len(technical_entries)
    )

    return {
        "negative_gamma": neg_gamma_entries[:top_n],
        "technical": technical_entries[:top_n],
    }


@router.get("/scan")
async def scan_sniper_entries(
    background_tasks: BackgroundTasks,
    top_n: int = Query(default=200, ge=1, le=5000),
    min_volume: float = Query(default=100_000, ge=0),
    force: bool = Query(default=False),
):
    """
    Scan for sniper entry points at absolute bottoms.
    Returns two lists:
      - negative_gamma: shorts trapped, gamma squeeze coming
      - technical: pure RSI/MACD capitulation with volume confirmation
    """
    global _last_scan, _last_scan_time, _scan_in_progress

    now = time.time()
    if not force and _last_scan["negative_gamma"] and (now - _last_scan_time) < _SCAN_TTL:
        return {
            "source": "cache",
            "scan_age_s": round(now - _last_scan_time, 0),
            "count_neg_gamma": len(_last_scan["negative_gamma"]),
            "count_technical": len(_last_scan["technical"]),
            "results": {
                "negative_gamma": _last_scan["negative_gamma"][:top_n],
                "technical": _last_scan["technical"][:top_n],
            },
        }

    if _scan_in_progress:
        return {
            "source": "scan_in_progress",
            "message": "Scan is running — try again in 30 s",
            "results": _last_scan,
        }

    _scan_in_progress = True
    try:
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: _run_sniper_scan(top_n, min_volume),
        )
        _last_scan = results
        _last_scan_time = time.time()
        return {
            "source": "fresh",
            "count_neg_gamma": len(results["negative_gamma"]),
            "count_technical": len(results["technical"]),
            "results": results,
        }
    except Exception as exc:
        logger.exception("Sniper entry scan failed: %s", exc)
        return {
            "source": "error",
            "message": str(exc),
            "results": {"negative_gamma": [], "technical": []},
        }
    finally:
        _scan_in_progress = False
