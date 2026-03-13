"""Chair Pattern detector — deterministic, multi-timeframe, scanner-ready.

Structure:
  Phase 1 — Backrest (Impulse):  strong directional move ≥ 12% in ≤ 20 bars
  Phase 2 — Seat Base (Pullback): 20–50% Fibonacci retracement
  Phase 3 — Seat (Consolidation): 5–25 bars, flat resistance, rising support,
                                   range ≤ 40% of impulse height, volume contraction
  Phase 4 — Breakout:             close > seat high + volume ≥ 1.5× avg + full body

Supported timeframes: 1H, 4H, 1D (and any intraday bar passed in).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from pattern_engine.common import base_result
from structure_engine.swing_detector import detect_swings, compute_adaptive_minimum_move


# ── Helpers ───────────────────────────────────────────────────────────────────

def _polyfit_slope(xs: list[float], ys: list[float]) -> float:
    """Return slope of best-fit line through (xs, ys); 0.0 on degenerate input."""
    if len(xs) < 2:
        return 0.0
    try:
        slope, _ = np.polyfit(xs, ys, 1)
        return float(slope)
    except Exception:
        return 0.0


# ── Main detector ─────────────────────────────────────────────────────────────

def detect(symbol: str, price_data: pd.DataFrame) -> dict[str, Any]:
    """Detect Chair Pattern and return a structured result dict.

    Returns base_result (NOT_PRESENT) immediately on structural failure.
    Sets status to FAILED when price invalidates the seat.
    Sets status to FORMING while the pattern builds.
    Sets status to READY when seat is complete and price is near resistance.
    Sets status to BREAKOUT when confirmed breakout occurs.
    Sets status to COMPLETED when price reaches target.
    """
    result = base_result("Chair Pattern")
    if price_data is None or len(price_data) < 30:
        return result

    df = price_data.sort_values("date").reset_index(drop=True).copy()
    closes = df["close"].astype(float).values
    highs  = df["high"].astype(float).values
    lows   = df["low"].astype(float).values
    opens  = df["open"].astype(float).values
    vols   = df["volume"].fillna(0.0).astype(float).values if "volume" in df.columns else np.zeros(len(df))
    n      = len(df)

    # ── Trend pre-condition: price above 50 MA ─────────────────────────────
    # Chair only forms in bullish context
    if n >= 50:
        ma50 = float(np.mean(closes[-50:]))
        if closes[-1] < ma50 * 0.98:   # 2% tolerance
            return result

    # ── Swing detection ────────────────────────────────────────────────────
    adaptive_mm = compute_adaptive_minimum_move(df)
    swings      = detect_swings(df, minimum_move=adaptive_mm)
    swing_lows  = swings["swing_lows"]
    swing_highs = swings["swing_highs"]
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return result

    # ── Trend validation: higher highs + higher lows ───────────────────────
    recent_hh = [h["price"] for h in swing_highs[-3:]]
    recent_hl = [l["price"] for l in swing_lows[-3:]]
    if len(recent_hh) >= 2 and recent_hh[-1] <= recent_hh[-2]:
        return result   # not making higher highs
    if len(recent_hl) >= 2 and recent_hl[-1] <= recent_hl[-2]:
        return result   # not making higher lows

    # ── Average volatility (for impulse strength) ──────────────────────────
    lookback = min(20, n - 1)
    if lookback >= 2:
        daily_rets = np.diff(closes[-lookback - 1:]) / np.maximum(closes[-lookback - 1:-1], 1e-9)
        avg_vol_pct = float(np.std(daily_rets)) * 100
    else:
        avg_vol_pct = 1.5
    avg_vol_pct = max(avg_vol_pct, 0.3)   # floor to avoid division by zero

    # ── Phase 1 — Impulse (backrest) ──────────────────────────────────────
    # Find the best qualifying impulse from the last 10 swing lows
    impulse_low  = None
    impulse_high = None
    best_score   = -1.0

    for low in swing_lows[-10:]:
        candidates = [
            h for h in swing_highs
            if h["index"] > low["index"]
            and (h["index"] - low["index"]) <= 20   # ≤ 20 candles
        ]
        for high in candidates:
            price_move_pct = (high["price"] - low["price"]) / max(low["price"], 1e-9) * 100
            if price_move_pct < 12.0:               # ≥ 12% required
                continue
            impulse_strength = price_move_pct / avg_vol_pct
            if impulse_strength < 2.0:              # strength filter
                continue
            # Prefer higher-score impulses (move% dominated, penalise slow)
            duration = high["index"] - low["index"]
            score = price_move_pct + impulse_strength * 0.5 - duration * 0.1
            if score > best_score:
                best_score    = score
                impulse_low   = low
                impulse_high  = high

    if impulse_low is None or impulse_high is None:
        return result

    impulse_height   = impulse_high["price"] - impulse_low["price"]
    impulse_move_pct = impulse_height / max(impulse_low["price"], 1e-9) * 100
    impulse_strength = impulse_move_pct / avg_vol_pct

    # Volume expansion check during impulse
    imp_start = int(impulse_low["index"])
    imp_end   = int(impulse_high["index"])
    seg_vols  = vols[imp_start:imp_end + 1]
    pre_vols  = vols[max(0, imp_start - 10):imp_start]
    vol_expanded = (
        float(np.mean(seg_vols)) > float(np.mean(pre_vols)) * 1.1
        if len(pre_vols) > 0 and len(seg_vols) > 0 else True
    )

    # ── Phase 2 — Pullback (seat base) ────────────────────────────────────
    post_lows = [l for l in swing_lows if l["index"] > impulse_high["index"]]
    if not post_lows:
        # Still in impulse or no pullback observed yet
        result["status"] = "FORMING"
        result["points"] = [
            [imp_start,             float(impulse_low["price"])],
            [imp_end,               float(impulse_high["price"])],
        ]
        return result

    pullback_low  = min(post_lows, key=lambda x: x["price"])
    retracement   = (impulse_high["price"] - pullback_low["price"]) / max(impulse_height, 1e-9)

    if retracement > 0.60:
        result["status"]             = "FAILED"
        result["invalidation_level"] = round(float(pullback_low["price"]), 4)
        result["points"] = [
            [imp_start,                      float(impulse_low["price"])],
            [imp_end,                        float(impulse_high["price"])],
            [int(pullback_low["index"]),     float(pullback_low["price"])],
        ]
        return result

    if retracement < 0.20:
        result["status"] = "FORMING"
        result["points"] = [
            [imp_start, float(impulse_low["price"])],
            [imp_end,   float(impulse_high["price"])],
        ]
        return result

    # ── Phase 3 — Seat formation (consolidation) ──────────────────────────
    seat_start   = int(pullback_low["index"])
    last_idx     = n - 1
    seat_candles = last_idx - seat_start

    if seat_candles < 5:
        result["status"] = "FORMING"
        result["points"] = [
            [imp_start,                  float(impulse_low["price"])],
            [imp_end,                    float(impulse_high["price"])],
            [int(pullback_low["index"]), float(pullback_low["price"])],
        ]
        return result

    if seat_candles > 25:
        # Seat took too long — pattern invalidated by structure age
        return result

    seat_slice  = df.iloc[seat_start:]
    seat_h      = float(seat_slice["high"].max())
    seat_l      = float(seat_slice["low"].min())
    seat_range  = seat_h - seat_l

    if seat_range > impulse_height * 0.40:
        return result   # seat too wide

    # Flat resistance
    seat_highs_ = [h for h in swing_highs if h["index"] >= seat_start]
    if len(seat_highs_) >= 2:
        xh         = [float(p["index"]) for p in seat_highs_[-4:]]
        yh         = [float(p["price"]) for p in seat_highs_[-4:]]
        res_slope  = _polyfit_slope(xh, yh)
        mean_yh    = float(np.mean(yh))
        flat_res   = abs(res_slope) / max(mean_yh, 1e-9) < 0.002
    else:
        flat_res   = True    # single high or no data — assume flat

    # Slightly rising support
    seat_lows_  = [l for l in swing_lows if l["index"] >= seat_start]
    if len(seat_lows_) >= 2:
        xl         = [float(p["index"]) for p in seat_lows_[-4:]]
        yl         = [float(p["price"]) for p in seat_lows_[-4:]]
        sup_slope  = _polyfit_slope(xl, yl)
        sup_int    = float(np.polyfit(xl, yl, 1)[1]) if len(xl) >= 2 else seat_l
        rising_sup = sup_slope >= 0
        invalidation = float(sup_slope * last_idx + sup_int)
    else:
        rising_sup   = True
        invalidation = seat_l

    # Volume contraction during seat vs impulse
    imp_vols  = vols[imp_start:imp_end + 1]
    seat_vols = vols[seat_start:last_idx + 1]
    imp_vol_mean  = float(np.mean(imp_vols))  if len(imp_vols)  > 0 else 0.0
    seat_vol_mean = float(np.mean(seat_vols)) if len(seat_vols) > 0 else 0.0
    vol_contracted = seat_vol_mean < imp_vol_mean * 0.90 if imp_vol_mean > 0 else True

    # Breakout parameters
    breakout = seat_h
    target   = breakout + impulse_height
    result["breakout_level"]     = round(breakout, 4)
    result["target"]             = round(target, 4)
    result["projected_target"]   = round(target, 4)
    result["invalidation_level"] = round(invalidation, 4)

    # ── Phase 4 — Breakout gate ────────────────────────────────────────────
    close      = float(closes[-1])
    last_vol   = float(vols[-1])
    avg_vol_20 = float(np.mean(vols[-20:])) if n >= 20 else float(np.mean(vols)) if len(vols) > 0 else 0.0
    body_size  = abs(float(closes[-1]) - float(opens[-1]))
    avg_body   = float(np.mean(np.abs(closes[-20:] - opens[-20:]))) if n >= 20 else body_size

    vol_ok             = last_vol >= avg_vol_20 * 1.5 if avg_vol_20 > 0 else True
    body_ok            = body_size >= avg_body
    breakout_confirmed = close > breakout and vol_ok and body_ok

    # ── Status ────────────────────────────────────────────────────────────
    if close < invalidation:
        status = "FAILED"
    elif close >= target:
        status = "COMPLETED"
    elif breakout_confirmed:
        status = "BREAKOUT"
    elif close >= breakout * 0.995:
        status = "READY"
    else:
        status = "FORMING"

    # ── Confidence score (0–100) ──────────────────────────────────────────
    # 0.30 impulse_strength + 0.20 retracement + 0.20 consolidation + 0.15 volume + 0.15 breakout

    impulse_strength_score = float(np.clip(impulse_strength / 6.0, 0.0, 1.0))

    # Ideal retracement 30–40%; penalise extremes
    retracement_score = float(np.clip(1.0 - abs(retracement - 0.35) / 0.25, 0.0, 1.0))

    # Consolidation: range tightness + structure quality
    range_quality       = float(np.clip(1.0 - seat_range / (impulse_height * 0.40), 0.0, 1.0))
    consolidation_score = range_quality * 0.6 + (0.2 if flat_res else 0.0) + (0.2 if rising_sup else 0.0)

    volume_score = 1.0 if vol_contracted else 0.3

    if status == "BREAKOUT":
        breakout_score = 1.0
    elif status in ("COMPLETED",):
        breakout_score = 0.9
    elif status == "READY":
        breakout_score = 0.6
    elif status == "FORMING":
        breakout_score = 0.3
    else:   # FAILED
        breakout_score = 0.0

    confidence = float(np.clip(
        (0.30 * impulse_strength_score
         + 0.20 * retracement_score
         + 0.20 * consolidation_score
         + 0.15 * volume_score
         + 0.15 * breakout_score) * 100,
        0.0, 100.0,
    ))

    # ── Chart overlay geometry ─────────────────────────────────────────────
    # overlay_lines: list of independent line pairs [[coord,price],[coord,price]]
    # coords can be date strings (for full-chart channel lines) or bar indices.
    pb_idx = int(pullback_low["index"])
    pb_px  = float(pullback_low["price"])
    imp_lo_px = float(impulse_low["price"])
    imp_hi_px = float(impulse_high["price"])

    # ── Full-chart upper/lower channel trendlines ─────────────────────────
    # Convert bar index to date string for full-chart spanning lines.
    def idx_to_date(i: int) -> str:
        i_clamped = int(max(0, min(n - 1, i)))
        d = df["date"].iloc[i_clamped]
        return str(d)[:10]   # "YYYY-MM-DD"

    def future_date(extra_bars: int) -> str:
        """Project a date string `extra_bars` beyond the last bar."""
        from datetime import datetime, timedelta
        try:
            last_d  = datetime.fromisoformat(str(df["date"].iloc[-1])[:10])
            # Estimate average bar spacing from last 10 bars
            if n >= 11:
                prev_d = datetime.fromisoformat(str(df["date"].iloc[-11])[:10])
                avg_days = max(1, round((last_d - prev_d).days / 10))
            else:
                avg_days = 1
            future = last_d + timedelta(days=avg_days * extra_bars)
            return future.strftime("%Y-%m-%d")
        except Exception:
            return idx_to_date(n - 1)

    # Forecast extension: ~25% of total bar count, min 30 bars
    n_forecast = max(30, int(n * 0.25))

    first_date  = idx_to_date(0)
    last_date   = idx_to_date(n - 1)
    fcast_date  = future_date(n_forecast)

    # Upper channel: linear regression through all swing highs
    # → slopes from earliest to latest and projects forward
    all_sh_x = [float(h["index"]) for h in swing_highs]
    all_sh_y = [float(h["price"]) for h in swing_highs]
    all_sl_x = [float(l["index"]) for l in swing_lows]
    all_sl_y = [float(l["price"]) for l in swing_lows]

    def channel_endpoints(xs, ys):
        """Return (at_first, at_last, at_forecast) prices for a channel line."""
        if len(xs) < 2:
            return None
        try:
            slope, intercept = np.polyfit(xs, ys, 1)
        except Exception:
            return None
        at_first    = slope * 0            + intercept
        at_last     = slope * (n - 1)      + intercept
        at_forecast = slope * (n - 1 + n_forecast) + intercept
        return at_first, at_last, at_forecast

    upper_pts = channel_endpoints(all_sh_x, all_sh_y)
    lower_pts = channel_endpoints(all_sl_x, all_sl_y)

    overlay_lines: list = [
        # 1. Backrest (impulse up) — steep solid line
        [[imp_start, imp_lo_px],  [imp_end, imp_hi_px]],
        # 2. Pullback (impulse high → pullback low)
        [[imp_end, imp_hi_px],    [pb_idx, pb_px]],
        # 3. Seat resistance — horizontal solid line at seat_h
        [[pb_idx, seat_h],        [last_idx, seat_h]],
        # 4. Seat support — horizontal solid line at seat_l
        [[pb_idx, seat_l],        [last_idx, seat_l]],
        # 5. Recovery (pullback low → current seat high) — upward through seat
        [[pb_idx, pb_px],         [last_idx, seat_h]],
        # 6. Right edge of seat box (vertical at current bar)
        [[last_idx, seat_l],      [last_idx, seat_h]],
        # 7. Target projection — extends from seat high toward target
        [[last_date, seat_h],     [fcast_date, target]],
    ]

    overlay_line_roles = [
        "impulse",     # 1. Backrest — bright green (the impulse move)
        "pullback",    # 2. Pullback from peak — orange
        "resistance",  # 3. Flat seat resistance — amber
        "support",     # 4. Flat seat support — amber
        "recovery",    # 5. Rising recovery through seat — light green dashed
        "box_edge",    # 6. Right-edge vertical of seat box — dim
        "target",      # 7. Target projection — purple dashed
    ]

    # Append full-chart channel lines (date-string coords, span entire chart + forecast)
    if upper_pts:
        overlay_lines.extend([
            [[first_date, upper_pts[0]], [fcast_date, upper_pts[2]]],   # 8. Upper channel
        ])
        overlay_line_roles.extend(["channel_upper"])

    if lower_pts:
        # Primary lower channel
        overlay_lines.extend([
            [[first_date, lower_pts[0]], [fcast_date, lower_pts[2]]],   # 9. Lower channel
        ])
        overlay_line_roles.extend(["channel_lower"])

        # Secondary parallel lower lines (angel/fan lines, spaced by ATR)
        # Add 1 and 2 ATR-width offsets below the primary lower channel
        atr_val = float(np.mean(highs[-20:] - lows[-20:]))
        if atr_val > 0:
            for mult, role in [(1.5, "channel_lower_2"), (3.0, "channel_lower_3")]:
                offset = atr_val * mult
                overlay_lines.append(
                    [[first_date, lower_pts[0] - offset], [fcast_date, lower_pts[2] - offset]]
                )
                overlay_line_roles.append(role)

    result.update({
        "status":     status,
        "confidence": round(confidence, 2),
        "probability": round(confidence, 2),   # legacy field
        # Extra scanner fields
        "impulse_start_index": imp_start,
        "impulse_end_index":   imp_end,
        "seat_high":           round(seat_h, 4),
        "seat_low":            round(seat_l, 4),
        # Legacy zigzag trace (fallback for non-chair renderers)
        "points": [
            [imp_start, imp_lo_px],
            [imp_end,   imp_hi_px],
            [pb_idx,    pb_px],
            [last_idx,  seat_h],
        ],
        # Full chair geometry for solid trendline overlay
        "overlay_lines": overlay_lines,
        # Per-segment role labels for frontend styling
        "overlay_line_roles": overlay_line_roles,
    })
    return result


# ── Backward-compatible helper ─────────────────────────────────────────────────

def detect_chair_pattern(df: pd.DataFrame) -> dict[str, Any] | None:
    """Backward-compatible wrapper used by existing scanners.

    Returns a legacy-format dict when a BREAKOUT is detected, else None.
    """
    out = detect("UNKNOWN", df)
    if out.get("status") in ("BREAKOUT", "READY"):
        return {
            "pattern":        "chair",
            "support_line":   [],
            "resistance_line": [],
            "breakout_level": out.get("breakout_level"),
            "target":         out.get("target"),
        }
    return None
