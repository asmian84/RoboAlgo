"""
GammaTracker — High-volume negative gamma scanner
Returns ALL stocks with negative gamma + basic consolidation setup
No strict filtering — focus on breadth over precision.
"""

from fastapi import APIRouter, Query, BackgroundTasks
import logging
import asyncio
import time
from typing import Optional

router = APIRouter(prefix="/gamma-tracker", tags=["gamma_tracker"])
logger = logging.getLogger(__name__)

# Global cache
_last_scan: list[dict] = []
_last_scan_time: float = 0
_scan_in_progress: bool = False
_SCAN_TTL = 5 * 60  # 5 min cache


def _run_gamma_scan(
    top_n: int = 500,
    min_volume: float = 100_000,
) -> list[dict]:
    """
    Fast gamma tracker scan:
    - Load all price data
    - Calculate negative gamma levels (simple logic)
    - Return ALL candidates sorted by gamma strength
    - No strict pattern detection
    """
    from data_engine.alphavantage_fetcher import load_ohlcv_from_db
    from config.settings import get_all_instruments
    import numpy as np

    syms = get_all_instruments()
    logger.info("Gamma tracker: loading price data for %d symbols …", len(syms))

    price_data = load_ohlcv_from_db(syms, lookback_days=200)

    results = []

    for sym, df in price_data.items():
        if df is None or len(df) < 20:
            continue

        try:
            # ── Basic checks ──
            close = df['close'].values
            volume = df['volume'].values
            high = df['high'].values
            low = df['low'].values

            if volume[-1] < min_volume:
                continue

            current_price = close[-1]
            if current_price <= 0:
                continue

            # ── Volatility squeeze (simple: compare BB bands) ──
            sma20 = np.mean(close[-20:])
            std20 = np.std(close[-20:])
            bb_width = 2 * std20
            bb_pct = (bb_width / sma20) * 100 if sma20 > 0 else 100

            # Consolidation = narrow BB (< 10%)
            is_consolidating = bb_pct < 10

            # ── Negative gamma proxy ──
            # (Simple: high IV rank + big ATR potential + put/call skew direction)
            # For now: volatility compression + volume profile
            recent_atr = np.mean(np.abs(np.diff(close[-10:])))
            avg_atr = np.mean(np.abs(np.diff(close[-60:])))
            atr_compression = (recent_atr / avg_atr) if avg_atr > 0 else 1.0

            # Squeeze signal: low recent ATR relative to avg
            squeeze_strength = max(0, 1.0 - atr_compression) * 100

            # ── Volume accumulation ──
            vol_20d = np.mean(volume[-20:])
            vol_60d = np.mean(volume[-60:])
            vol_accumulation = (vol_20d / vol_60d - 1) * 100 if vol_60d > 0 else 0

            # ── Price position in range ──
            range_high = np.max(high[-30:])
            range_low = np.min(low[-30:])
            price_pct = ((current_price - range_low) / (range_high - range_low) * 100) if range_high > range_low else 50

            # Consolidation at middle = ready to burst
            consolidation_score = 100 - abs(price_pct - 50)  # 0-100, 100 = centered

            # ── Negative gamma score ──
            # High = potential for explosive move
            gamma_score = (squeeze_strength + vol_accumulation + consolidation_score) / 3

            # Quality filter: Must have consolidation + good gamma signal
            # (Not too loose, not too strict)
            min_consolidation = 40  # At least some tightness
            min_gamma = 35          # Decent squeeze/volume signal
            min_acc = 0             # Volume can be neutral or positive

            passes_quality = (
                (consolidation_score >= min_consolidation or is_consolidating) and
                gamma_score >= min_gamma and
                squeeze_strength > 0
            )

            if not passes_quality:
                continue

            # Readiness = how ready is it to burst (weighted combination)
            readiness = (
                (consolidation_score * 0.35) +    # Tight consolidation critical
                (squeeze_strength * 0.40) +       # Volatility squeeze = pressure buildup
                (vol_accumulation * 0.25)         # Volume accumulation confirms setup
            )

            results.append({
                'symbol': sym,
                'rank': 0,
                'current_price': float(current_price),
                'gamma_score': float(gamma_score),
                'readiness_score': float(readiness),  # Primary sort: how ready to burst
                'squeeze_strength': float(squeeze_strength),
                'vol_accumulation': float(vol_accumulation),
                'consolidation_score': float(consolidation_score),
                'is_consolidating': bool(is_consolidating),
                'bb_width_pct': float(bb_pct),
                'atr_compression': float(atr_compression),
                'volume_20d': float(vol_20d),
                'notes': ['Ready to burst: consolidation + squeeze + volume']
            })

        except Exception as e:
            logger.debug("Gamma scan error for %s: %s", sym, e)
            continue

    # Sort by readiness_score descending (how ready to burst)
    results.sort(key=lambda x: x['readiness_score'], reverse=True)

    # Rank and limit
    for i, r in enumerate(results[:top_n], 1):
        r['rank'] = i

    logger.info("Gamma tracker found %d candidates (returning top %d)", len(results), min(top_n, len(results)))
    return results[:top_n]


@router.get("/scan")
async def scan_gamma(
    background_tasks: BackgroundTasks,
    top_n: int = Query(default=500, ge=1, le=5000),
    min_volume: float = Query(default=100_000, ge=0),
    force: bool = Query(default=False),
):
    """
    Scan ALL stocks for negative gamma + consolidation.
    Returns hundreds of candidates, not just 12.
    """
    global _last_scan, _last_scan_time, _scan_in_progress

    now = time.time()
    if not force and _last_scan and (now - _last_scan_time) < _SCAN_TTL:
        return {
            "source": "cache",
            "scan_age_s": round(now - _last_scan_time, 0),
            "count": len(_last_scan),
            "results": _last_scan[:top_n],
        }

    if _scan_in_progress:
        return {
            "source": "scan_in_progress",
            "message": "Scan is running — try again in 30 s",
            "results": _last_scan[:top_n],
        }

    _scan_in_progress = True
    try:
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: _run_gamma_scan(top_n, min_volume),
        )
        _last_scan = results
        _last_scan_time = time.time()
        return {
            "source": "fresh",
            "count": len(results),
            "results": results,
        }
    except Exception as exc:
        logger.exception("Gamma scan failed: %s", exc)
        return {
            "source": "error",
            "message": str(exc),
            "results": [],
        }
    finally:
        _scan_in_progress = False
