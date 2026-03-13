"""Chart pattern detector — full pattern suite.

Orchestrates all sub-pattern detectors and returns the combined list.
Each sub-detector returns either a single dict or a list[dict].

Pattern families:
  Trend Continuation  : Chair, Bull Flag, Bear Flag, Bullish/Bearish Pennant,
                        Ascending Channel, Descending Channel, Cup & Handle
  Compression/Breakout: Symmetrical / Ascending / Descending Triangle,
                        Rising / Falling Wedge
  Reversal            : Head & Shoulders, Inv. Head & Shoulders,
                        Double Top/Bottom, Triple Top/Bottom,
                        Rounding Bottom/Top
  Range / Expansion   : Rectangle, Megaphone
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from pattern_engine.ascending_channel import detect as detect_channel
from pattern_engine.bear_flag import detect as detect_bear_flag
from pattern_engine.bull_flag import detect as detect_bull_flag
from pattern_engine.chair_pattern import detect as detect_chair
from pattern_engine.compression_breakout import detect as detect_compression
from pattern_engine.cup_handle import detect as detect_cup
from pattern_engine.descending_channel import detect as detect_descending_channel
from pattern_engine.double_top_bottom import detect as detect_double_patterns
from pattern_engine.expansion_pattern import detect as detect_megaphone
from pattern_engine.head_shoulders import detect as detect_head_shoulders
from pattern_engine.pennant import detect as detect_pennants
from pattern_engine.rectangle_pattern import detect as detect_rectangle
from pattern_engine.rounding_pattern import detect as detect_rounding
from pattern_engine.wedge_pattern import detect as detect_wedges

VALID_STATES = {"NOT_PRESENT", "FORMING", "READY", "BREAKOUT", "FAILED", "COMPLETED"}


def _normalize(raw: dict[str, Any], fallback_name: str) -> dict[str, Any]:
    status = raw.get("status", "NOT_PRESENT")
    if status not in VALID_STATES:
        status = "NOT_PRESENT"
    normalized: dict[str, Any] = {
        "pattern_name":     raw.get("pattern_name", fallback_name),
        "pattern_category": "chart",
        "status":           status,
        "breakout_level":   raw.get("breakout_level"),
        "target":           raw.get("target", raw.get("projected_target")),
        "confidence":       float(raw.get("confidence", raw.get("probability", 0.0)) or 0.0),
        "points":           raw.get("points", []),
    }
    # Pass through all enriched fields from sub-detectors
    for key in (
        "overlay_lines", "overlay_line_roles", "direction",
        "invalidation_level", "projected_target", "probability",
        "support_level", "resistance_level",
        "point_labels", "event_points",
        "cup_left_price", "cup_right_price", "cup_bottom_price",
        # Harmonic-specific
        "ratios", "prz_low", "prz_high",
        # Shaded fill zone between two trendlines (channels, wedges, triangles)
        "fill_zone",
    ):
        if key in raw:
            normalized[key] = raw[key]
    return normalized


def _norm_list(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize a list returned by a multi-result detector."""
    return [_normalize(item, item.get("pattern_name", "Unknown")) for item in items]


def detect_all(
    symbol: str, price_data: pd.DataFrame, resolution_minutes: int = 0,
) -> list[dict[str, Any]]:
    """Return ALL chart sub-patterns as a list (including NOT_PRESENT ones).

    Used by PatternService._run_all_detectors so every sub-pattern is
    individually visible in the Pattern Detection panel.

    Timeframe routing:
    - Chair Pattern: requires >= 30m bars (skip on 1m / 5m / 15m)
    """
    results: list[dict[str, Any]] = []

    # ── Trend-continuation ─────────────────────────────────────────────────
    if resolution_minutes == 0 or resolution_minutes >= 30:
        results.append(_normalize(detect_chair(symbol, price_data),           "Chair Pattern"))

    results.append(_normalize(detect_bull_flag(symbol, price_data),           "Bull Flag"))
    results.append(_normalize(detect_bear_flag(symbol, price_data),           "Bear Flag"))
    results += _norm_list(detect_pennants(symbol, price_data))               # Bullish + Bearish Pennant
    results.append(_normalize(detect_cup(symbol, price_data),                 "Cup & Handle"))
    results.append(_normalize(detect_channel(symbol, price_data),             "Ascending Channel"))
    results.append(_normalize(detect_descending_channel(symbol, price_data),  "Descending Channel"))

    # ── Compression / Triangle / Wedge ────────────────────────────────────
    results += _norm_list(detect_compression(symbol, price_data))             # Sym + Asc + Desc Triangle
    results += _norm_list(detect_wedges(symbol, price_data))                  # Rising + Falling Wedge

    # ── Reversal patterns ─────────────────────────────────────────────────
    results += _norm_list(detect_head_shoulders(symbol, price_data))         # H&S + Inv. H&S
    results += _norm_list(detect_double_patterns(symbol, price_data))        # DT, DB, TT, TB
    results += _norm_list(detect_rounding(symbol, price_data))               # Rounding Bottom + Top

    # ── Range / Expansion ─────────────────────────────────────────────────
    results.append(_normalize(detect_rectangle(symbol, price_data),          "Rectangle"))
    results.append(_normalize(detect_megaphone(symbol, price_data),          "Megaphone"))

    return results


def detect(symbol: str, price_data: pd.DataFrame) -> dict[str, Any]:
    """Legacy entry-point: return only the highest-confidence chart pattern."""
    candidates = [c for c in detect_all(symbol, price_data) if c.get("status") != "NOT_PRESENT"]
    candidates.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)
    return candidates[0] if candidates else {
        "pattern_name":     "Chart Pattern",
        "pattern_category": "chart",
        "status":           "NOT_PRESENT",
        "breakout_level":   None,
        "target":           None,
        "confidence":       0.0,
        "points":           [],
    }

