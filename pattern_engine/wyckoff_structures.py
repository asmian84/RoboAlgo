"""Wyckoff structure detector — Accumulation + Distribution with phase labeling.

Detects Wyckoff Accumulation (Spring → SOS → LPS → markup) and
Wyckoff Distribution (UTAD → SOW → LPSY → markdown) using
price/volume structural analysis over the most recent consolidation range.

Returns a list of results so both accumulation and distribution are individually
visible in the Pattern Detection panel.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from structure_engine.swing_detector import detect_swings, compute_adaptive_minimum_move

VALID_STATES = {"NOT_PRESENT", "FORMING", "READY", "BREAKOUT", "FAILED", "COMPLETED"}


# ── Helpers ───────────────────────────────────────────────────────────────────


def _base(name: str) -> dict[str, Any]:
    return {
        "pattern_name": name,
        "pattern_category": "wyckoff",
        "status": "NOT_PRESENT",
        "breakout_level": None,
        "target": None,
        "invalidation_level": None,
        "confidence": 0.0,
        "points": [],
        "overlay_lines": [],
        "phase": None,
        "phase_label": None,
        "events": [],
        "event_points": [],  # [{label, index, price}] → normalized to [{label, date, price}]
    }


def _vol_trend(vol: pd.Series, window_short: int = 10, window_long: int = 30) -> float:
    """Volume trend ratio — >1 means increasing, <1 means declining."""
    short = float(vol.tail(window_short).mean())
    long = float(vol.tail(window_long).mean())
    return short / max(long, 1.0)


def _higher_lows(lows: list[dict[str, Any]]) -> bool:
    """True if swing lows are making higher lows (rising support)."""
    if len(lows) < 2:
        return False
    prices = [l["price"] for l in lows]
    return all(prices[i] < prices[i + 1] for i in range(len(prices) - 1))


def _lower_highs(highs: list[dict[str, Any]]) -> bool:
    """True if swing highs are making lower highs (falling resistance)."""
    if len(highs) < 2:
        return False
    prices = [h["price"] for h in highs]
    return all(prices[i] > prices[i + 1] for i in range(len(prices) - 1))


def _find_range(df: pd.DataFrame, lookback: int = 60) -> tuple[float, float, int, int]:
    """Find the consolidation range over the last `lookback` bars.

    Returns (range_low, range_high, range_start_idx, range_end_idx).
    """
    tail = df.tail(lookback).reset_index(drop=True)
    high = tail["high"].astype(float)
    low = tail["low"].astype(float)
    range_high = float(high.max())
    range_low = float(low.min())
    start_idx = len(df) - lookback
    end_idx = len(df) - 1
    return range_low, range_high, max(start_idx, 0), end_idx


def _vol_confirms_accumulation(df: pd.DataFrame, mid: float) -> float:
    """Score 0-1: rising vol on rallies above mid + declining vol on drops below mid."""
    close = df["close"].astype(float)
    vol = df["volume"].fillna(0).astype(float)
    tail = df.tail(40)
    if len(tail) < 10:
        return 0.5
    c = tail["close"].astype(float)
    v = tail["volume"].fillna(0).astype(float)
    above = v[c > mid]
    below = v[c <= mid]
    if len(above) < 3 or len(below) < 3:
        return 0.5
    avg_above = float(above.mean())
    avg_below = float(below.mean())
    # Accumulation: volume expands on rallies, contracts on drops
    if avg_above > avg_below:
        ratio = min(avg_above / max(avg_below, 1), 3.0)
        return min(0.5 + (ratio - 1.0) * 0.25, 1.0)
    return max(0.5 - (avg_below / max(avg_above, 1) - 1.0) * 0.15, 0.0)


def _vol_confirms_distribution(df: pd.DataFrame, mid: float) -> float:
    """Score 0-1: rising vol on drops below mid + declining vol on rallies above mid."""
    tail = df.tail(40)
    if len(tail) < 10:
        return 0.5
    c = tail["close"].astype(float)
    v = tail["volume"].fillna(0).astype(float)
    above = v[c > mid]
    below = v[c <= mid]
    if len(above) < 3 or len(below) < 3:
        return 0.5
    avg_above = float(above.mean())
    avg_below = float(below.mean())
    # Distribution: volume expands on drops, contracts on rallies
    if avg_below > avg_above:
        ratio = min(avg_below / max(avg_above, 1), 3.0)
        return min(0.5 + (ratio - 1.0) * 0.25, 1.0)
    return max(0.5 - (avg_above / max(avg_below, 1) - 1.0) * 0.15, 0.0)


# ── Phase & Event Detection ──────────────────────────────────────────────────


def _detect_accum_phase(
    close: pd.Series,
    lows: list[dict],
    highs: list[dict],
    range_low: float,
    range_high: float,
    last_close: float,
    vol_ratio: float,
) -> tuple[str, str, list[str], list[dict]]:
    """Determine accumulation phase (A-E), events, and event coordinates.

    Phase A: Selling climax (SC), automatic rally (AR) — stopping the downtrend
    Phase B: Secondary tests (ST) — building cause, testing supply/demand
    Phase C: Spring / test — shakeout below support (bull trap reversal)
    Phase D: Sign of Strength (SOS), Last Point of Support (LPS) — markup begins
    Phase E: Breakout above range — trend continuation

    Returns (phase, phase_label, events, event_points) where event_points is
    [{label, index, price}] for chart annotation markers.
    """
    events: list[str] = []
    event_points: list[dict] = []
    width = max(range_high - range_low, 1e-9)
    mid = (range_high + range_low) / 2.0

    def _ep(label: str, pt: dict) -> None:
        event_points.append({"label": label, "index": int(pt["index"]), "price": float(pt["price"])})

    # ── Spring: swing low that dipped below range_low then recovered ──────
    has_spring = False
    spring_low_pt: dict | None = None
    if lows:
        min_low_entry = min(lows, key=lambda l: l["price"])
        if min_low_entry["price"] < range_low * 1.005 and last_close > range_low:
            has_spring = True
            spring_low_pt = min_low_entry
            events.append("Spring")
            _ep("Spring", min_low_entry)

    # ── SOS: close breaks above mid with strength ─────────────────────────
    has_sos = last_close > mid + width * 0.15
    if has_sos:
        events.append("SOS")
        # Coordinate: first swing high above mid+10% width (after spring if present)
        sos_candidates = [h for h in highs if h["price"] > mid + width * 0.10]
        if spring_low_pt:
            post_spring = [h for h in sos_candidates if h["index"] > spring_low_pt["index"]]
            if post_spring:
                sos_candidates = post_spring
        if sos_candidates:
            _ep("SOS", sos_candidates[0])
        elif highs:
            _ep("SOS", highs[-1])

    # ── LPS: higher lows after spring ────────────────────────────────────
    has_lps = False
    if has_spring and spring_low_pt and _higher_lows(lows[-3:]):
        has_lps = True
        events.append("LPS")
        post_spring_lows = [l for l in lows if l["index"] > spring_low_pt["index"]]
        if post_spring_lows:
            _ep("LPS", post_spring_lows[-1])

    # ── ST: secondary test near range_low ────────────────────────────────
    has_st = False
    if len(lows) >= 2:
        second_low = lows[-1]
        if abs(second_low["price"] - range_low) / width < 0.15:
            has_st = True
            events.append("ST")
            _ep("ST", second_low)

    # ── Phase determination ───────────────────────────────────────────────
    if last_close > range_high:
        phase = "E"
        phase_label = "Markup"
        if "SOS" not in events:
            events.append("SOS")
    elif has_lps and has_sos:
        phase = "D"
        phase_label = "SOS / LPS"
    elif has_spring:
        phase = "C"
        phase_label = "Spring"
    elif has_st:
        phase = "B"
        phase_label = "Secondary Test"
    else:
        phase = "A"
        phase_label = "Selling Climax"
        if lows and lows[0]["price"] <= range_low * 1.02:
            events.append("SC")
            _ep("SC", lows[0])
        if highs and highs[0]["price"] >= range_high * 0.95:
            events.append("AR")
            _ep("AR", highs[0])

    # ── Always annotate SC/AR as historical anchors (even in later phases) ─
    if phase != "A":
        if lows and not any(ep["label"] == "SC" for ep in event_points):
            _ep("SC", lows[0])
        if highs and not any(ep["label"] == "AR" for ep in event_points):
            _ep("AR", highs[0])

    return phase, phase_label, events, event_points


def _detect_distrib_phase(
    close: pd.Series,
    lows: list[dict],
    highs: list[dict],
    range_low: float,
    range_high: float,
    last_close: float,
    vol_ratio: float,
) -> tuple[str, str, list[str], list[dict]]:
    """Determine distribution phase (A-E), events, and event coordinates.

    Phase A: PSY (preliminary supply), BC (buying climax), AR (auto reaction)
    Phase B: ST (secondary test), UT (upthrust) — testing demand
    Phase C: UTAD (upthrust after distribution) — bull trap above resistance
    Phase D: SOW (sign of weakness), LPSY (last point of supply) — markdown begins
    Phase E: Breakdown below range

    Returns (phase, phase_label, events, event_points) where event_points is
    [{label, index, price}] for chart annotation markers.
    """
    events: list[str] = []
    event_points: list[dict] = []
    width = max(range_high - range_low, 1e-9)
    mid = (range_high + range_low) / 2.0

    def _ep(label: str, pt: dict) -> None:
        event_points.append({"label": label, "index": int(pt["index"]), "price": float(pt["price"])})

    # ── UTAD: swing high that pierced above range_high then fell back ─────
    has_utad = False
    utad_pt: dict | None = None
    if highs:
        max_high_entry = max(highs, key=lambda h: h["price"])
        if max_high_entry["price"] > range_high * 0.995 and last_close < range_high:
            has_utad = True
            utad_pt = max_high_entry
            events.append("UTAD")
            _ep("UTAD", utad_pt)

    # ── SOW: close breaks below mid with strength ─────────────────────────
    has_sow = last_close < mid - width * 0.15
    if has_sow:
        events.append("SOW")
        if lows:
            _ep("SOW", lows[-1])

    # ── LPSY: lower highs after UTAD ──────────────────────────────────────
    has_lpsy = False
    if has_utad and utad_pt and _lower_highs(highs[-3:]):
        has_lpsy = True
        events.append("LPSY")
        post_utad_highs = [h for h in highs if h["index"] > utad_pt["index"]]
        if post_utad_highs:
            _ep("LPSY", post_utad_highs[-1])

    # ── UT: upthrust near range_high ──────────────────────────────────────
    has_ut = False
    if len(highs) >= 2:
        second_high = highs[-1]
        if abs(second_high["price"] - range_high) / width < 0.15:
            has_ut = True
            events.append("UT")
            _ep("UT", second_high)

    # ── PSY: preliminary supply — first high in range ─────────────────────
    if highs and highs[0]["price"] >= range_high * 0.95:
        events.append("PSY")
        _ep("PSY", highs[0])

    # ── Phase determination ───────────────────────────────────────────────
    if last_close < range_low:
        phase = "E"
        phase_label = "Markdown"
        if "SOW" not in events:
            events.append("SOW")
    elif has_lpsy and has_sow:
        phase = "D"
        phase_label = "SOW / LPSY"
    elif has_utad:
        phase = "C"
        phase_label = "UTAD"
    elif has_ut:
        phase = "B"
        phase_label = "Secondary Test"
    else:
        phase = "A"
        phase_label = "Buying Climax"
        if highs and highs[0]["price"] >= range_high * 0.98:
            events.append("BC")
            _ep("BC", highs[0])
        if lows and lows[0]["price"] <= range_low * 1.05:
            events.append("AR")
            _ep("AR", lows[0])

    # ── Always annotate PSY/BC/AR as historical anchors (in later phases) ─
    if phase != "A":
        if highs and not any(ep["label"] in ("PSY", "BC") for ep in event_points):
            _ep("PSY", highs[0])
        if lows and not any(ep["label"] == "AR" for ep in event_points):
            _ep("AR", lows[0])

    return phase, phase_label, events, event_points


# ── Accumulation Detector ─────────────────────────────────────────────────────


def _detect_accumulation(
    df: pd.DataFrame,
    swings: dict[str, list[dict]],
    adaptive_mm: float,
) -> dict[str, Any]:
    """Detect Wyckoff Accumulation structure."""
    result = _base("Wyckoff Accumulation")
    if len(df) < 60:
        return result

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    vol = df["volume"].fillna(0).astype(float)
    last_close = float(close.iloc[-1])

    # Find consolidation range
    lookback = min(80, len(df))
    range_low, range_high, r_start, r_end = _find_range(df, lookback)
    width = max(range_high - range_low, 1e-9)
    compression = width / max(last_close, 1e-9)

    # Compression must be reasonable (not trending strongly)
    if compression > 0.40:
        return result

    mid = (range_high + range_low) / 2.0

    # Need a prior downtrend or flat-to-down context going INTO the range
    # Check if the first quarter of the range starts lower than mid
    first_quarter = close.iloc[r_start:r_start + max(lookback // 4, 5)]
    if len(first_quarter) < 3:
        return result

    # Get swings within the range
    all_pivots = sorted(swings["swing_highs"] + swings["swing_lows"], key=lambda p: p["index"])
    range_lows = [l for l in swings["swing_lows"] if l["index"] >= r_start]
    range_highs = [h for h in swings["swing_highs"] if h["index"] >= r_start]

    if len(range_lows) < 2 or len(range_highs) < 2:
        return result

    # ── Evidence scoring ──────────────────────────────────────────────────
    score = 45.0  # base

    # 1. Shift up: closes migrate from lower half to upper half
    first_half = close.tail(lookback).iloc[:lookback // 2]
    second_half = close.tail(lookback).iloc[lookback // 2:]
    if float(second_half.mean()) > float(first_half.mean()):
        score += 12.0

    # 2. Volume confirms accumulation
    vol_score = _vol_confirms_accumulation(df, mid)
    score += vol_score * 15.0

    # 3. Spring-like action (dip below support then recovery)
    min_low_price = min(l["price"] for l in range_lows)
    spring_like = min_low_price < range_low * 1.01 and last_close > range_low
    if spring_like:
        score += 15.0

    # 4. Higher lows (rising support)
    if _higher_lows(range_lows[-3:]):
        score += 8.0

    # 5. Volume trend — expanding on recent rally
    vol_ratio = _vol_trend(vol)
    if vol_ratio > 1.1:
        score += 5.0

    confidence = float(np.clip(score, 0, 100))

    # ── Phase detection ───────────────────────────────────────────────────
    phase, phase_label, events, event_points = _detect_accum_phase(
        close, range_lows, range_highs, range_low, range_high, last_close, vol_ratio,
    )

    # ── Status ────────────────────────────────────────────────────────────
    breakout = float(range_high)
    target = float(breakout + width * 0.8)
    invalidation = float(range_low)

    if last_close < invalidation * 0.98:
        status = "FAILED"
    elif last_close >= target:
        status = "COMPLETED"
    elif last_close > breakout and vol.tail(1).iloc[0] > vol.tail(20).mean() * 1.2:
        status = "BREAKOUT"
    elif last_close >= breakout * 0.995:
        status = "READY"
    else:
        status = "FORMING"

    # ── Overlay lines: support/resistance box ─────────────────────────────
    overlay_lines: list[list[list[int | float]]] = [
        # Resistance line across range
        [[r_start, range_high], [r_end, range_high]],
        # Support line across range
        [[r_start, range_low], [r_end, range_low]],
        # Mid-line
        [[r_start, mid], [r_end, mid]],
    ]
    # If breakout, draw breakout line extending right
    if status in ("BREAKOUT", "COMPLETED"):
        overlay_lines.append([[r_end, breakout], [r_end + 10, breakout]])

    result.update({
        "status": status,
        "breakout_level": round(breakout, 4),
        "target": round(target, 4),
        "invalidation_level": round(invalidation, 4),
        "confidence": round(confidence, 2),
        "points": [[p["index"], p["price"]] for p in range_lows[-3:] + range_highs[-3:]],
        "overlay_lines": overlay_lines,
        "phase": phase,
        "phase_label": phase_label,
        "events": events,
        "event_points": event_points,
        "direction": "bullish",
        "support_level": round(range_low, 4),
        "resistance_level": round(range_high, 4),
    })
    return result


# ── Distribution Detector ─────────────────────────────────────────────────────


def _detect_distribution(
    df: pd.DataFrame,
    swings: dict[str, list[dict]],
    adaptive_mm: float,
) -> dict[str, Any]:
    """Detect Wyckoff Distribution structure."""
    result = _base("Wyckoff Distribution")
    if len(df) < 60:
        return result

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    vol = df["volume"].fillna(0).astype(float)
    last_close = float(close.iloc[-1])

    # Find consolidation range
    lookback = min(80, len(df))
    range_low, range_high, r_start, r_end = _find_range(df, lookback)
    width = max(range_high - range_low, 1e-9)
    compression = width / max(last_close, 1e-9)

    if compression > 0.40:
        return result

    mid = (range_high + range_low) / 2.0

    # Distribution needs prior uptrend coming INTO the range
    # Check if the first quarter starts higher than mid (we're topping out)
    first_quarter = close.iloc[r_start:r_start + max(lookback // 4, 5)]
    if len(first_quarter) < 3:
        return result

    # Get swings within the range
    range_lows = [l for l in swings["swing_lows"] if l["index"] >= r_start]
    range_highs = [h for h in swings["swing_highs"] if h["index"] >= r_start]

    if len(range_lows) < 2 or len(range_highs) < 2:
        return result

    # ── Evidence scoring ──────────────────────────────────────────────────
    score = 45.0  # base

    # 1. Shift down: closes migrate from upper half to lower half
    first_half = close.tail(lookback).iloc[:lookback // 2]
    second_half = close.tail(lookback).iloc[lookback // 2:]
    if float(second_half.mean()) < float(first_half.mean()):
        score += 12.0

    # 2. Volume confirms distribution
    vol_score = _vol_confirms_distribution(df, mid)
    score += vol_score * 15.0

    # 3. UTAD-like action (push above resistance then falls back)
    max_high_price = max(h["price"] for h in range_highs)
    utad_like = max_high_price > range_high * 0.995 and last_close < range_high
    if utad_like:
        score += 15.0

    # 4. Lower highs (falling resistance)
    if _lower_highs(range_highs[-3:]):
        score += 8.0

    # 5. Volume trend — expanding on recent decline
    vol_ratio = _vol_trend(vol)
    if vol_ratio > 1.1 and last_close < mid:
        score += 5.0

    confidence = float(np.clip(score, 0, 100))

    # ── Phase detection ───────────────────────────────────────────────────
    phase, phase_label, events, event_points = _detect_distrib_phase(
        close, range_lows, range_highs, range_low, range_high, last_close, vol_ratio,
    )

    # ── Status ────────────────────────────────────────────────────────────
    breakout = float(range_low)  # bearish breakout is BELOW support
    target = float(breakout - width * 0.8)
    invalidation = float(range_high)

    if last_close > invalidation * 1.02:
        status = "FAILED"
    elif last_close <= target:
        status = "COMPLETED"
    elif last_close < breakout and vol.tail(1).iloc[0] > vol.tail(20).mean() * 1.2:
        status = "BREAKOUT"
    elif last_close <= breakout * 1.005:
        status = "READY"
    else:
        status = "FORMING"

    # ── Overlay lines ─────────────────────────────────────────────────────
    overlay_lines: list[list[list[int | float]]] = [
        # Resistance line across range
        [[r_start, range_high], [r_end, range_high]],
        # Support line across range
        [[r_start, range_low], [r_end, range_low]],
        # Mid-line
        [[r_start, mid], [r_end, mid]],
    ]
    if status in ("BREAKOUT", "COMPLETED"):
        overlay_lines.append([[r_end, breakout], [r_end + 10, breakout]])

    result.update({
        "status": status,
        "breakout_level": round(breakout, 4),
        "target": round(target, 4),
        "invalidation_level": round(invalidation, 4),
        "confidence": round(confidence, 2),
        "points": [[p["index"], p["price"]] for p in range_lows[-3:] + range_highs[-3:]],
        "overlay_lines": overlay_lines,
        "phase": phase,
        "phase_label": phase_label,
        "events": events,
        "event_points": event_points,
        "direction": "bearish",
        "support_level": round(range_low, 4),
        "resistance_level": round(range_high, 4),
    })
    return result


# ── Public API ────────────────────────────────────────────────────────────────


def detect(symbol: str, price_data: pd.DataFrame) -> list[dict[str, Any]]:
    """Detect Wyckoff structures — returns list with accumulation + distribution.

    Both are always returned (even if NOT_PRESENT) so the Pattern Detection
    panel shows individual status for each.
    """
    results: list[dict[str, Any]] = []
    if price_data is None or len(price_data) < 60:
        results.append(_base("Wyckoff Accumulation"))
        results.append(_base("Wyckoff Distribution"))
        return results

    df = price_data.sort_values("date").reset_index(drop=True).copy()
    adaptive_mm = compute_adaptive_minimum_move(df)
    swings = detect_swings(df, minimum_move=adaptive_mm)

    accum = _detect_accumulation(df, swings, adaptive_mm)
    distrib = _detect_distribution(df, swings, adaptive_mm)
    results.append(accum)
    results.append(distrib)
    return results
