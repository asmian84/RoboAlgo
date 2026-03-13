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

router = APIRouter(prefix="/sniper-entry", tags=["sniper_entry"])
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
        if df is None or len(df) < 50:
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

            # ── SUPPORT / LOCAL LOW ──
            # Find lowest point in last 60 bars
            low_60 = np.min(low[-60:])
            low_200 = np.min(low[-200:])
            distance_from_60_low = ((current_price - low_60) / low_60 * 100) if low_60 > 0 else 0
            distance_from_200_low = ((current_price - low_200) / low_200 * 100) if low_200 > 0 else 0

            # Must be CLOSE to a low (within 3%)
            at_local_low = distance_from_60_low < 3
            near_52w_low = distance_from_200_low < 5

            if not (at_local_low or near_52w_low):
                continue

            # ── CAPITULATION SIGNALS ──
            # 1. RSI extreme oversold
            delta = np.diff(close)
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            avg_gain = np.mean(gain[-14:])
            avg_loss = np.mean(loss[-14:])
            rs = avg_gain / avg_loss if avg_loss > 0 else 0
            rsi = 100 - (100 / (1 + rs)) if rs > 0 else 100

            rsi_oversold = rsi < 25
            extreme_oversold = rsi < 15

            # 2. MACD - below signal and starting to diverge
            sma12 = np.mean(close[-12:])
            sma26 = np.mean(close[-26:])
            macd = sma12 - sma26
            signal = np.mean([sma12 - sma26 for sma12, sma26 in
                            zip(np.convolve(close, np.ones(12)/12, mode='valid'),
                                np.convolve(close, np.ones(26)/26, mode='valid'))][-9:])
            macd_below_signal = macd < signal
            macd_bouncing = (close[-1] - close[-5]) > 0  # Price up last 5 bars

            # 3. Volume Capitulation
            recent_vol = np.mean(volume[-5:])
            vol_20 = np.mean(volume[-20:])
            vol_60 = np.mean(volume[-60:])
            volume_dried_up = recent_vol < vol_20 * 0.7  # Current vol < 70% of 20-day avg
            vol_down_days = np.sum(np.diff(close[-5:]) < 0)  # Multiple down days

            # 4. Time in oversold
            rsi_oversold_days = 0
            for i in range(1, 21):
                delta_i = np.diff(close[-i-14:-i])
                gain_i = np.where(delta_i > 0, delta_i, 0)
                loss_i = np.where(delta_i < 0, -delta_i, 0)
                avg_gain_i = np.mean(gain_i)
                avg_loss_i = np.mean(loss_i)
                rs_i = avg_gain_i / avg_loss_i if avg_loss_i > 0 else 0
                rsi_i = 100 - (100 / (1 + rs_i)) if rs_i > 0 else 100
                if rsi_i < 25:
                    rsi_oversold_days += 1
                else:
                    break

            # 5. Bounce setup (volume increasing on up days)
            last_5_up_vol = []
            for i in range(1, 6):
                if close[-i] > close[-(i+1)]:
                    last_5_up_vol.append(volume[-i])
            up_vol_avg = np.mean(last_5_up_vol) if last_5_up_vol else 0
            vol_avg_overall = np.mean(volume[-20:])
            bounce_vol_strong = up_vol_avg > vol_avg_overall * 0.8

            # ── ENTRY SCORE (0-100) ──
            # Components:
            support_score = (100 - distance_from_60_low * 10) if distance_from_60_low < 5 else (100 - distance_from_200_low * 5)
            capitulation_score = (
                (40 if extreme_oversold else 25 if rsi_oversold else 0) +
                (15 if macd_below_signal else 0) +
                (20 if volume_dried_up else 0) +
                (10 if vol_down_days >= 2 else 0)
            )
            bounce_score = (
                (15 if macd_bouncing else 0) +
                (15 if bounce_vol_strong else 0)
            )
            time_score = min(rsi_oversold_days * 2, 20)  # Max 20 points

            entry_quality = (support_score * 0.25 + capitulation_score * 0.40 + bounce_score * 0.20 + time_score * 0.15)

            # Must have minimum capitulation
            if capitulation_score < 30:
                continue

            # ── NEGATIVE GAMMA CHECK ──
            # Simple proxy: volatility compression + price at low = gamma trap
            sma20 = np.mean(close[-20:])
            std20 = np.std(close[-20:])
            bb_width = 2 * std20
            bb_pct = (bb_width / sma20) * 100 if sma20 > 0 else 100

            volatility_compressed = bb_pct < 8
            negative_gamma_proxy = (
                at_local_low and
                rsi_oversold and
                volatility_compressed and
                capitulation_score > 40
            )

            # ── Build result ──
            result = {
                'symbol': sym,
                'current_price': float(current_price),
                'entry_quality': float(entry_quality),
                'support_score': float(support_score),
                'capitulation_score': float(capitulation_score),
                'bounce_score': float(bounce_score),
                'time_score': float(time_score),
                'rsi': float(rsi),
                'rsi_oversold_days': int(rsi_oversold_days),
                'distance_from_60_low_pct': float(distance_from_60_low),
                'distance_from_200_low_pct': float(distance_from_200_low),
                'volume_dried_up': bool(volume_dried_up),
                'macd_bouncing': bool(macd_bouncing),
                'bb_width_pct': float(bb_pct),
                'notes': [
                    f"RSI: {rsi:.0f} (oversold {rsi_oversold_days}d)",
                    f"Price: {distance_from_60_low:.1f}% from 60-day low",
                    f"Cap Score: {capitulation_score:.0f}/100"
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
