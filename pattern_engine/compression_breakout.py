"""Triangle pattern detectors — Symmetrical, Ascending, Descending.

Each triangle type is detected independently using OLS regression through
pivot highs and lows.  All three are always returned (possibly NOT_PRESENT).

  - Symmetrical Triangle : falling resistance + rising support → neutral breakout
  - Ascending Triangle   : flat resistance   + rising support → bullish breakout
  - Descending Triangle  : falling resistance + flat support  → bearish breakdown

Multi-scale scan: tries 3 minimum_move scales so both short-term (weeks) and
long-term (months) triangles are caught.  The highest-confidence result per
triangle type wins across all scales.

Visual output per pattern:
  - Resistance trendline + support trendline
  - Coloured fill between trendlines (indigo for sym, green for asc, red for desc)
  - Breakout extension line (dashed)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from pattern_engine.common import (
    base_result,
    composite_probability,
    liquidity_alignment_score,
    market_regime_score,
    momentum_score,
    status_from_levels,
    volume_confirmation,
)
from structure_engine.swing_detector import detect_swings, compute_adaptive_minimum_move

# Slope normalised to mean price must be below this to count as "flat"
_FLAT_THRESHOLD   = 0.0003
# Minimum compression: channel must narrow to < 80% of original width
_MIN_COMPRESSION  = 0.20
# Minimum height as fraction of mean price
_MIN_HEIGHT_RATIO = 0.012


def _is_flat(slope: float, mean_price: float) -> bool:
    if mean_price <= 0:
        return abs(slope) < _FLAT_THRESHOLD
    return abs(slope / mean_price) < _FLAT_THRESHOLD


def _build_overlay(first_idx: int, last_idx: int,
                   res_slope: float, res_int: float,
                   sup_slope: float, sup_int: float,
                   bullish: bool) -> list:
    fwd_idx = last_idx + 20
    resistance_end = float(res_slope * last_idx + res_int)
    support_end    = float(sup_slope * last_idx + sup_int)
    if bullish:
        ext_price = float(res_slope * fwd_idx + res_int)
        ext_seg   = [[last_idx, resistance_end], [fwd_idx, ext_price]]
    else:
        ext_price = float(sup_slope * fwd_idx + sup_int)
        ext_seg   = [[last_idx, support_end], [fwd_idx, ext_price]]
    return [
        [[first_idx, float(res_slope * first_idx + res_int)], [last_idx, resistance_end]],
        [[first_idx, float(sup_slope * first_idx + sup_int)], [last_idx, support_end]],
        ext_seg,
    ]


def _score(df: pd.DataFrame, breakout: float, invalidation: float,
           target: float, bullish: bool,
           compression_ratio: float) -> tuple[float, str]:
    """Compute probability and status for a detected triangle."""
    sq = float(np.clip(60 + compression_ratio * 40, 0, 100))
    prob = composite_probability(
        structure_quality=sq,
        volume=volume_confirmation(df),
        liquidity=liquidity_alignment_score(df, breakout),
        regime=market_regime_score(df),
        momentum=momentum_score(df),
    )
    status = status_from_levels(df["close"], breakout, invalidation, target, bullish=bullish)
    return round(prob, 2), status


def _detect_symmetrical(df: pd.DataFrame, highs: list, lows: list) -> dict[str, Any]:
    """Symmetrical Triangle: falling resistance + rising support."""
    result = base_result("Symmetrical Triangle")
    if len(highs) < 2 or len(lows) < 2:
        return result

    last_idx   = len(df) - 1
    mean_price = max(float(df["close"].tail(60).mean()), 1.0)

    xh = np.array([p["index"] for p in highs], dtype=float)
    yh = np.array([p["price"] for p in highs], dtype=float)
    xl = np.array([p["index"] for p in lows],  dtype=float)
    yl = np.array([p["price"] for p in lows],  dtype=float)

    res_slope, res_int = np.polyfit(xh, yh, 1)
    sup_slope, sup_int = np.polyfit(xl, yl, 1)

    # Falling resistance + rising support
    if res_slope >= 0 or sup_slope <= 0:
        return result
    if _is_flat(res_slope, mean_price) or _is_flat(sup_slope, mean_price):
        return result

    breakout_line = float(res_slope * last_idx + res_int)
    support_line  = float(sup_slope * last_idx + sup_int)
    if breakout_line <= support_line:
        return result

    width_now  = breakout_line - support_line
    width_hist = float(np.mean(yh) - np.mean(yl))
    if width_hist <= 0 or width_now < 1e-9:
        return result
    if width_now / mean_price < _MIN_HEIGHT_RATIO:
        return result
    compression_ratio = max(0.0, (width_hist - width_now) / width_hist)
    if compression_ratio < _MIN_COMPRESSION:
        return result

    # Symmetrical: breakout direction uncertain — bias bullish by convention
    breakout     = breakout_line
    invalidation = support_line
    target       = float(breakout + width_hist)
    prob, status = _score(df, breakout, invalidation, target, True, compression_ratio)

    first_idx = int(min(xh[0], xl[0]))
    overlay_lines = _build_overlay(first_idx, last_idx,
                                   res_slope, res_int,
                                   sup_slope, sup_int, True)
    result.update({
        "pattern_name":       "Symmetrical Triangle",
        "status":             status,
        "breakout_level":     round(breakout, 4),
        "invalidation_level": round(invalidation, 4),
        "projected_target":   round(target, 4),
        "probability":        prob,
        "confidence":         prob,
        "direction":          "neutral",
        "points": [[p["index"], p["price"]] for p in highs[-2:] + lows[-2:]],
        "point_labels":       ["H1", "H2", "L1", "L2"],
        "overlay_lines":      overlay_lines,
        "overlay_line_roles": ["resistance", "support", "neckline_ext"],
        "fill_zone": {
            "upper":   0,
            "lower":   1,
            "color":   "#818cf8",    # indigo — neutral direction
            "opacity": 0.20,
        },
    })
    return result


def _detect_ascending_tri(df: pd.DataFrame, highs: list, lows: list) -> dict[str, Any]:
    """Ascending Triangle: flat resistance + rising support → bullish."""
    result = base_result("Ascending Triangle")
    if len(highs) < 2 or len(lows) < 2:
        return result

    last_idx   = len(df) - 1
    mean_price = max(float(df["close"].tail(60).mean()), 1.0)

    xh = np.array([p["index"] for p in highs], dtype=float)
    yh = np.array([p["price"] for p in highs], dtype=float)
    xl = np.array([p["index"] for p in lows],  dtype=float)
    yl = np.array([p["price"] for p in lows],  dtype=float)

    res_slope, res_int = np.polyfit(xh, yh, 1)
    sup_slope, sup_int = np.polyfit(xl, yl, 1)

    # Flat resistance + rising support
    if not _is_flat(res_slope, mean_price):
        return result
    if sup_slope <= 0 or _is_flat(sup_slope, mean_price):
        return result

    breakout_line = float(res_slope * last_idx + res_int)
    support_line  = float(sup_slope * last_idx + sup_int)
    if breakout_line <= support_line:
        return result

    width_now  = breakout_line - support_line
    width_hist = float(np.mean(yh) - np.mean(yl))
    if width_hist <= 0 or width_now < 1e-9:
        return result
    if width_now / mean_price < _MIN_HEIGHT_RATIO:
        return result
    compression_ratio = max(0.0, (width_hist - width_now) / width_hist)
    if compression_ratio < _MIN_COMPRESSION:
        return result

    breakout     = breakout_line
    invalidation = support_line
    target       = float(breakout + width_hist)
    prob, status = _score(df, breakout, invalidation, target, True, compression_ratio)

    first_idx = int(min(xh[0], xl[0]))
    overlay_lines = _build_overlay(first_idx, last_idx,
                                   res_slope, res_int,
                                   sup_slope, sup_int, True)
    result.update({
        "pattern_name":       "Ascending Triangle",
        "status":             status,
        "breakout_level":     round(breakout, 4),
        "invalidation_level": round(invalidation, 4),
        "projected_target":   round(target, 4),
        "probability":        prob,
        "confidence":         prob,
        "direction":          "bullish",
        "points": [[p["index"], p["price"]] for p in highs[-2:] + lows[-2:]],
        "point_labels":       ["H1", "H2", "L1", "L2"],
        "overlay_lines":      overlay_lines,
        "overlay_line_roles": ["resistance", "support", "neckline_ext"],
        "fill_zone": {
            "upper":   0,
            "lower":   1,
            "color":   "#22c55e",    # green — bullish
            "opacity": 0.20,
        },
    })
    return result


def _detect_descending_tri(df: pd.DataFrame, highs: list, lows: list) -> dict[str, Any]:
    """Descending Triangle: falling resistance + flat support → bearish."""
    result = base_result("Descending Triangle")
    if len(highs) < 2 or len(lows) < 2:
        return result

    last_idx   = len(df) - 1
    mean_price = max(float(df["close"].tail(60).mean()), 1.0)

    xh = np.array([p["index"] for p in highs], dtype=float)
    yh = np.array([p["price"] for p in highs], dtype=float)
    xl = np.array([p["index"] for p in lows],  dtype=float)
    yl = np.array([p["price"] for p in lows],  dtype=float)

    res_slope, res_int = np.polyfit(xh, yh, 1)
    sup_slope, sup_int = np.polyfit(xl, yl, 1)

    # Falling resistance + flat support
    if res_slope >= 0 or _is_flat(res_slope, mean_price):
        return result
    if not _is_flat(sup_slope, mean_price):
        return result

    breakout_line = float(res_slope * last_idx + res_int)
    support_line  = float(sup_slope * last_idx + sup_int)
    if breakout_line <= support_line:
        return result

    width_now  = breakout_line - support_line
    width_hist = float(np.mean(yh) - np.mean(yl))
    if width_hist <= 0 or width_now < 1e-9:
        return result
    if width_now / mean_price < _MIN_HEIGHT_RATIO:
        return result
    compression_ratio = max(0.0, (width_hist - width_now) / width_hist)
    if compression_ratio < _MIN_COMPRESSION:
        return result

    breakout     = support_line
    invalidation = breakout_line
    target       = float(breakout - width_hist)
    prob, status = _score(df, breakout, invalidation, target, False, compression_ratio)

    first_idx = int(min(xh[0], xl[0]))
    overlay_lines = _build_overlay(first_idx, last_idx,
                                   res_slope, res_int,
                                   sup_slope, sup_int, False)
    result.update({
        "pattern_name":       "Descending Triangle",
        "status":             status,
        "breakout_level":     round(breakout, 4),
        "invalidation_level": round(invalidation, 4),
        "projected_target":   round(target, 4),
        "probability":        prob,
        "confidence":         prob,
        "direction":          "bearish",
        "points": [[p["index"], p["price"]] for p in highs[-2:] + lows[-2:]],
        "point_labels":       ["H1", "H2", "L1", "L2"],
        "overlay_lines":      overlay_lines,
        "overlay_line_roles": ["resistance", "support", "neckline_ext"],
        "fill_zone": {
            "upper":   0,
            "lower":   1,
            "color":   "#f87171",    # red — bearish
            "opacity": 0.20,
        },
    })
    return result


def detect(symbol: str, price_data: pd.DataFrame) -> list[dict[str, Any]]:
    """Return [Symmetrical Triangle, Ascending Triangle, Descending Triangle].

    Each entry is the best result found across multiple swing-detection scales.
    Entries that are not detected have status NOT_PRESENT.
    """
    defaults = [
        base_result("Symmetrical Triangle"),
        base_result("Ascending Triangle"),
        base_result("Descending Triangle"),
    ]
    if price_data is None or len(price_data) < 50:
        return defaults

    df          = price_data.sort_values("date").reset_index(drop=True).copy()
    adaptive_mm = compute_adaptive_minimum_move(df)

    best_sym  = defaults[0]
    best_asc  = defaults[1]
    best_desc = defaults[2]

    for scale in [0.8, 1.5, 3.0]:
        mm     = adaptive_mm * scale
        swings = detect_swings(df, minimum_move=mm)
        highs  = swings["swing_highs"][-5:]
        lows   = swings["swing_lows"][-5:]

        r = _detect_symmetrical(df, highs, lows)
        if float(r.get("confidence", 0.0)) > float(best_sym.get("confidence", 0.0)):
            best_sym = r

        r = _detect_ascending_tri(df, highs, lows)
        if float(r.get("confidence", 0.0)) > float(best_asc.get("confidence", 0.0)):
            best_asc = r

        r = _detect_descending_tri(df, highs, lows)
        if float(r.get("confidence", 0.0)) > float(best_desc.get("confidence", 0.0)):
            best_desc = r

    return [best_sym, best_asc, best_desc]
