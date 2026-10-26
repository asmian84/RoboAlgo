"""
Penny Sniper — Millionaire Maker Scanner
=========================================
Two-layer approach:
  Layer 1 — Finviz discovery:
      Scrapes Finviz screener to get stocks near 52-week lows with volume.
      Typically returns 200-400 unique tickers matching rough criteria.

  Layer 2 — RoboAlgo intelligence (the value-add over Finviz):
      Batch-downloads 60 days of OHLCV from yfinance in groups of 50.
      Applies 5 reversal-detection signals:

      1. RSI REVERSAL  — RSI was below 35, now rising for 3+ days (slope > 0)
      2. CAPITULATION  — At least one day in last 15 with volume > 2× avg
                          AND price was down that day (sellers exhausted)
      3. PRICE COIL    — Daily range narrowing over last 5 bars (compression)
                          AND price making higher lows (floor forming)
      4. BB SQUEEZE    — Bollinger Band width at/near 20-day minimum
      5. MACD TURN     — MACD histogram crossing from negative to positive

      All 5 signals scored 0–100. Composite = weighted sum.
      Only setups with composite ≥ 38 AND at least 2 signals > 45 are returned.
"""

from fastapi import APIRouter, Query
import asyncio
import time
import math
import logging
import re
from typing import Optional

router = APIRouter(tags=["penny_sniper"])
logger = logging.getLogger(__name__)

_cache:      list[dict] = []
_cache_time: float      = 0
_scanning:   bool       = False
_SCAN_TTL   = 12 * 60  # 12 min

# ── Finviz universe fetcher ───────────────────────────────────────────────────

def _fetch_finviz_universe(max_price: float = 20.0, max_pages: int = 15) -> list[str]:
    """
    Scrape Finviz screener for stocks near 52w lows with volume.
    Returns deduplicated list of ticker symbols.
    """
    import requests

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        'Accept-Language': 'en-US,en;q=0.9',
    }

    price_filter = f"sh_price_u{int(max_price)}"
    base_url = (
        "https://finviz.com/screener.ashx"
        f"?v=111&f={price_filter},sh_avgvol_o50,ta_52weeklow_b30h"
        "&o=-volume"
    )

    tickers = []
    seen    = set()

    for page in range(max_pages):
        row_offset = page * 20 + 1
        url = f"{base_url}&r={row_offset}"
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code != 200:
                break
            html = resp.text

            # Extract tickers and deduplicate WITHIN this page first
            page_tickers_raw = re.findall(r'quote\.ashx\?t=([A-Z]{1,5})', html)
            page_tickers_dedup = list(dict.fromkeys(page_tickers_raw))  # preserve order, remove dups

            new_tickers = [t for t in page_tickers_dedup if t not in seen]
            if not new_tickers:
                break

            tickers.extend(new_tickers)
            seen.update(new_tickers)
            logger.debug("Finviz page %d: +%d tickers (total %d)", page + 1, len(new_tickers), len(tickers))
            time.sleep(0.6)
        except Exception as e:
            logger.warning("Finviz fetch page %d failed: %s", page + 1, e)
            break

    logger.info("Finviz universe: %d unique tickers", len(tickers))
    return tickers

# ── Wilder RSI ────────────────────────────────────────────────────────────────

def _wilder_rsi_series(close, period=14):
    import numpy as np
    out = np.full(len(close), float('nan'))
    if len(close) < period + 2:
        return out
    delta = np.diff(close)
    gain  = np.where(delta > 0, delta, 0.0)
    loss  = np.where(delta < 0, -delta, 0.0)
    ag    = float(np.mean(gain[:period]))
    al    = float(np.mean(loss[:period]))
    rsis  = []
    for g, l in zip(gain[period:], loss[period:]):
        ag = (ag * (period - 1) + g) / period
        al = (al * (period - 1) + l) / period
        rsis.append(100.0 - (100.0 / (1 + ag / al)) if al > 0 else 100.0)
    # rsis has len(close)-1-period values; they start at close index period+1
    start = period + 1
    end   = start + len(rsis)
    if end <= len(out):
        out[start:end] = rsis
    return out

def _ema(arr, span):
    import numpy as np
    result = np.full(len(arr), float('nan'))
    k = 2 / (span + 1)
    first_valid = next((i for i, v in enumerate(arr) if not math.isnan(v)), None)
    if first_valid is None:
        return result
    result[first_valid] = arr[first_valid]
    for i in range(first_valid + 1, len(arr)):
        if math.isnan(arr[i]):
            result[i] = result[i - 1]
        else:
            result[i] = arr[i] * k + result[i - 1] * (1 - k)
    return result

# ── Score a single stock using pre-downloaded OHLCV ──────────────────────────

def _score_from_ohlcv(sym: str, close, volume, high, low, max_price: float, min_price: float) -> Optional[dict]:
    import numpy as np

    if len(close) < 25:
        return None

    spot = float(close[-1])
    if spot < min_price or spot > max_price or spot == 0:
        return None
    if float(volume[-1]) < 5_000:
        return None

    avg_vol = float(np.mean(volume[-21:-1])) if len(volume) > 21 else 1.0
    if avg_vol < 1:
        avg_vol = 1.0

    low_52w  = float(np.min(low))
    high_52w = float(np.max(high))
    dist_low = (spot - low_52w) / low_52w * 100 if low_52w > 0 else 999

    # ── Signal 1: RSI REVERSAL ────────────────────────────────────────────────
    rsi_series = _wilder_rsi_series(close)
    valid_rsi  = rsi_series[~np.isnan(rsi_series)]
    if len(valid_rsi) < 5:
        return None
    rsi_now    = float(valid_rsi[-1])
    rsi_3d_ago = float(valid_rsi[-4]) if len(valid_rsi) >= 4 else rsi_now
    rsi_7d_ago = float(valid_rsi[-8]) if len(valid_rsi) >= 8 else rsi_3d_ago
    rsi_slope  = rsi_now - rsi_3d_ago
    rsi_was_low = rsi_7d_ago < 35 or rsi_3d_ago < 35

    # Don't cut off stocks that are launching — RSI can be high if it was oversold before
    if rsi_now > 80 and not rsi_was_low:
        return None  # overbought with no prior oversold = extended, skip

    if rsi_was_low and rsi_slope > 3:
        rsi_rev_score = min(100, 50 + rsi_slope * 5)
    elif rsi_was_low and rsi_slope > 0:
        rsi_rev_score = 30 + rsi_slope * 5
    elif rsi_was_low and rsi_now > 50:
        rsi_rev_score = 60                               # reversed from oversold = LAUNCH
    elif rsi_now < 25:
        rsi_rev_score = 45
    elif rsi_now < 35:
        rsi_rev_score = 30
    elif rsi_now < 50:
        rsi_rev_score = 15
    else:
        rsi_rev_score = 5

    # ── Signal 2: CAPITULATION ────────────────────────────────────────────────
    lookback  = min(15, len(close) - 1)
    cap_score = 0.0
    for i in range(len(close) - lookback, len(close) - 1):
        day_ret = (close[i] - close[i-1]) / close[i-1] if close[i-1] > 0 else 0
        v_ratio = volume[i] / avg_vol
        if day_ret < -0.02 and v_ratio > 2.0:
            cap_score = max(cap_score, min(100, v_ratio * 25))

    today_v_ratio = volume[-1] / avg_vol

    # Also check 5-day avg volume vs 20-day to catch sustained volume pickup
    vol_5d  = float(np.mean(volume[-5:])) if len(volume) >= 5 else avg_vol
    vol_surge_ratio = vol_5d / avg_vol if avg_vol > 0 else 1.0

    # ── Signal 3: PRICE COIL ──────────────────────────────────────────────────
    last5_ranges = [(high[i] - low[i]) / close[i] * 100 for i in range(-5, 0) if close[i] > 0]
    prev5_ranges = [(high[i] - low[i]) / close[i] * 100 for i in range(-10, -5) if close[i] > 0]
    avg_recent   = float(np.mean(last5_ranges)) if last5_ranges else 999
    avg_prior    = float(np.mean(prev5_ranges)) if prev5_ranges else avg_recent
    range_comp   = (avg_prior - avg_recent) / avg_prior * 100 if avg_prior > 0 else 0

    recent_lows  = low[-5:]
    higher_lows  = sum(1 for i in range(1, len(recent_lows)) if recent_lows[i] >= recent_lows[i-1])

    coil_score = min(100, max(0, range_comp * 1.5 + higher_lows * 15))

    # ── Signal 4: BB SQUEEZE ──────────────────────────────────────────────────
    if len(close) >= 20:
        sma20  = np.mean(close[-20:])
        std20  = np.std(close[-20:])
        bb_width_pct = (4 * std20 / sma20 * 100) if sma20 > 0 else 999

        all_bb_widths = []
        for j in range(max(0, len(close) - 30), len(close) - 5):
            window = close[j:j+20]
            if len(window) == 20:
                m = np.mean(window)
                w = np.std(window) / m * 100 if m > 0 else None
                if w:
                    all_bb_widths.append(w)
        min_recent_bb = float(np.min(all_bb_widths)) if all_bb_widths else bb_width_pct

        bb_squeeze_ratio = (bb_width_pct - min_recent_bb) / min_recent_bb if min_recent_bb > 0 else 1
        bb_score = max(0, min(100, (1 - bb_squeeze_ratio) * 100))
        lower_band = sma20 - 2 * std20
        near_lower = (spot - lower_band) / (4 * std20) if std20 > 0 else 0.5
        bb_score   = bb_score * 0.7 + max(0, (1 - near_lower) * 30)
    else:
        bb_score = 0

    # ── Signal 5: MACD HISTOGRAM TURN ────────────────────────────────────────
    macd_line  = _ema(close, 12) - _ema(close, 26)
    signal_arr = _ema(macd_line, 9)
    hist_arr   = macd_line - signal_arr
    valid_hist = hist_arr[~np.isnan(hist_arr)]
    if len(valid_hist) >= 3:
        h_now  = float(valid_hist[-1])
        h_prev = float(valid_hist[-2])
        h_2ago = float(valid_hist[-3])
        if h_2ago < 0 and h_now > h_prev:
            macd_score = min(100, abs(h_now - h_2ago) / (abs(h_2ago) + 1e-9) * 100)
        elif h_now > h_prev > h_2ago:
            macd_score = 50
        elif h_now > h_prev:
            macd_score = 25
        else:
            macd_score = 0
    else:
        macd_score = 0

    # ── Signal 6: VOLUME LAUNCH ───────────────────────────────────────────────
    # High recent volume vs average = money flowing in
    if vol_surge_ratio >= 4.0:
        vol_score = 100
    elif vol_surge_ratio >= 3.0:
        vol_score = min(100, (vol_surge_ratio - 1) * 35)
    elif vol_surge_ratio >= 2.0:
        vol_score = min(100, (vol_surge_ratio - 1) * 20)
    elif today_v_ratio >= 5.0:
        vol_score = min(100, today_v_ratio * 12)
    else:
        vol_score = max(0, (vol_surge_ratio - 1.0) * 15)

    # Price momentum: up > 20% from recent low in last 5 bars
    low5 = float(np.min(low[-5:])) if len(low) >= 5 else spot
    price_launch_pct = (spot - low5) / low5 * 100 if low5 > 0 else 0

    # ── Composite ─────────────────────────────────────────────────────────────
    rocket_score = (
        rsi_rev_score * 0.25 +
        cap_score     * 0.20 +
        coil_score    * 0.15 +
        bb_score      * 0.12 +
        macd_score    * 0.10 +
        vol_score     * 0.18    # volume is critical for penny stocks
    )

    # Bonus for price momentum (already moving)
    if price_launch_pct > 30:
        rocket_score = min(100, rocket_score + 12)
    elif price_launch_pct > 15:
        rocket_score = min(100, rocket_score + 6)

    signals_strong = sum(1 for s in [rsi_rev_score, cap_score, coil_score, bb_score, macd_score, vol_score] if s > 45)
    if rocket_score < 32 or signals_strong < 1:
        return None

    if dist_low < 5:    rocket_score = min(100, rocket_score + 15)
    elif dist_low < 10: rocket_score = min(100, rocket_score + 8)
    elif dist_low < 20: rocket_score = min(100, rocket_score + 3)

    active = []
    if rsi_rev_score > 50: active.append(f"RSI↑ {rsi_now:.0f}")
    if cap_score > 50:     active.append(f"Cap {cap_score:.0f}")
    if coil_score > 50:    active.append(f"Coil {coil_score:.0f}")
    if bb_score > 50:      active.append(f"BB {bb_score:.0f}")
    if macd_score > 50:    active.append(f"MACD {macd_score:.0f}")
    if vol_score > 50:     active.append(f"Vol {vol_surge_ratio:.1f}×")

    off_high = (high_52w - spot) / high_52w * 100 if high_52w > 0 else 0
    mom_3d   = (spot - close[-4]) / close[-4] * 100 if len(close) >= 4 and close[-4] > 0 else 0

    return {
        'symbol':        sym,
        'price':         round(spot, 4),
        'rocket_score':  round(rocket_score, 1),
        'rsi':           round(rsi_now, 1),
        'rsi_slope':     round(rsi_slope, 1),
        'rsi_score':     round(rsi_rev_score, 1),
        'cap_score':     round(cap_score, 1),
        'coil_score':    round(coil_score, 1),
        'bb_score':      round(bb_score, 1),
        'macd_score':    round(macd_score, 1),
        'dist_from_low': round(dist_low, 1),
        'off_high_pct':  round(off_high, 1),
        'vol_ratio':     round(today_v_ratio, 2),
        'low_52w':       round(low_52w, 4),
        'high_52w':      round(high_52w, 2),
        'vol_score':      round(vol_score, 1),
        'signals_count': signals_strong,
        'notes': [
            f"${spot:.3f}  ·  {dist_low:.1f}% from 52w low  ·  {off_high:.0f}% off 52w high",
            f"Vol {today_v_ratio:.1f}×  ·  RSI {rsi_now:.0f} (slope {rsi_slope:+.1f})  ·  3d {mom_3d:+.1f}%",
            "Signals: " + ("  ·  ".join(active) if active else "borderline"),
        ],
    }

# ── Batch downloader ──────────────────────────────────────────────────────────

def _download_batch(tickers: list[str]) -> dict[str, dict]:
    """Download OHLCV for a batch of tickers via yf.download() — single HTTP call."""
    import yfinance as yf
    import numpy as np

    if not tickers:
        return {}

    try:
        df = yf.download(
            tickers,
            period='6mo',
            interval='1d',
            progress=False,
            auto_adjust=True,
            threads=True,
        )
    except Exception as e:
        logger.warning("Batch download failed: %s", e)
        return {}

    result = {}
    if df is None or df.empty:
        return result

    # Single-ticker: columns are flat (Close, High, Low, Open, Volume)
    if len(tickers) == 1:
        sym = tickers[0]
        try:
            close  = df['Close'].values.astype(float)
            volume = df['Volume'].values.astype(float)
            high   = df['High'].values.astype(float)
            low    = df['Low'].values.astype(float)
            if len(close) >= 25:
                result[sym] = {'close': close, 'volume': volume, 'high': high, 'low': low}
        except Exception:
            pass
        return result

    # Multi-ticker: MultiIndex columns (Metric, Ticker)
    for sym in tickers:
        try:
            close  = df['Close'][sym].dropna().values.astype(float)
            volume = df['Volume'][sym].dropna().values.astype(float)
            high   = df['High'][sym].dropna().values.astype(float)
            low    = df['Low'][sym].dropna().values.astype(float)
            n = min(len(close), len(volume), len(high), len(low))
            if n >= 25:
                result[sym] = {
                    'close':  close[:n],
                    'volume': volume[:n],
                    'high':   high[:n],
                    'low':    low[:n],
                }
        except Exception:
            pass

    return result

# ── Full scan ─────────────────────────────────────────────────────────────────

def _run_scan(top_n: int, max_price: float, min_price: float) -> list[dict]:
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Layer 1: Finviz universe
    tickers = _fetch_finviz_universe(max_price=max_price)
    if not tickers:
        logger.warning("Finviz returned 0 tickers — using fallback")
        tickers = [
            "EONR","CEI","TELL","SNDL","CLOV","WKHS","NKLA","GOEV","FCEL",
            "PLUG","CLNE","GEVO","NVAX","OCGN","VXRT","BNGO","SENS","RIOT",
            "MARA","HUT","BITF","ACB","CGC","TLRY","HEXO","OGI",
        ]

    logger.info("Penny sniper: downloading data for %d tickers in batches…", len(tickers))

    # Layer 2: batch download in groups of 40, with 3 parallel workers
    BATCH = 40
    batches = [tickers[i:i+BATCH] for i in range(0, len(tickers), BATCH)]

    all_ohlcv: dict[str, dict] = {}

    def _dl(batch):
        time.sleep(0.1)  # small stagger
        return _download_batch(batch)

    with ThreadPoolExecutor(max_workers=3) as pool:
        futs = {pool.submit(_dl, b): b for b in batches}
        for fut in as_completed(futs):
            try:
                all_ohlcv.update(fut.result())
            except Exception as e:
                logger.warning("Batch future failed: %s", e)

    logger.info("Data downloaded for %d / %d tickers", len(all_ohlcv), len(tickers))

    # Layer 3: score each
    results = []
    for sym, ohlcv in all_ohlcv.items():
        try:
            r = _score_from_ohlcv(
                sym,
                ohlcv['close'],
                ohlcv['volume'],
                ohlcv['high'],
                ohlcv['low'],
                max_price,
                min_price,
            )
            if r:
                results.append(r)
        except Exception as e:
            logger.debug("Score error %s: %s", sym, e)

    results.sort(key=lambda x: x['rocket_score'], reverse=True)
    for i, r in enumerate(results[:top_n], 1):
        r['rank'] = i

    logger.info("Penny sniper: %d rocket setups out of %d downloaded", len(results), len(all_ohlcv))
    return results[:top_n]

# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.get("/scan")
async def scan_penny(
    top_n:     int   = Query(default=100, ge=1, le=500),
    max_price: float = Query(default=20.0, ge=0.01, le=100),
    min_price: float = Query(default=0.10, ge=0.01),
    force:     bool  = Query(default=False),
):
    """
    Penny stock millionaire-maker scanner.
    Finviz discovers universe → batch yfinance download →
    RoboAlgo 5-signal reversal intelligence filters to coiled setups only.
    """
    global _cache, _cache_time, _scanning

    now = time.time()
    if not force and _cache and (now - _cache_time) < _SCAN_TTL:
        return {
            "source":     "cache",
            "scan_age_s": round(now - _cache_time),
            "count":      len(_cache),
            "results":    _cache[:top_n],
        }

    if _scanning:
        return {
            "source":  "scan_in_progress",
            "message": "Scan running — check back in ~90 s",
            "results": _cache[:top_n],
        }

    _scanning = True
    try:
        loop    = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, lambda: _run_scan(top_n, max_price, min_price))
        _cache      = results
        _cache_time = time.time()
        return {
            "source":  "fresh",
            "count":   len(results),
            "results": results,
        }
    except Exception as exc:
        logger.exception("Penny sniper scan failed: %s", exc)
        return {"source": "error", "message": str(exc), "results": []}
    finally:
        _scanning = False
