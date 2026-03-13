"""Price-Time Confluence Node Detector.

Combines signals from structure_engine, pattern_engine, cycle_engine,
and geometry_engine to identify Decision Nodes — specific price-time
zones where multiple analysis methods converge.

A Decision Node represents a high-probability zone for a price reaction.
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from cycle_engine.cycle_projection import project_cycle
from geometry_engine.square_of_9 import sq9_nearest_levels
from geometry_engine.square_of_144 import sq144_nearest_levels
from geometry_engine.price_time_symmetry import compute_price_time_symmetry
from structure_engine.swing_detector import detect_swings, compute_adaptive_minimum_move

logger = logging.getLogger("confluence_engine.node_detector")


def _cluster_levels(
    levels: list[float],
    tolerance_pct: float = 0.015,
) -> list[dict]:
    """Cluster nearby price levels into zones.

    Groups levels within tolerance_pct of each other and returns
    the midpoint, width, and number of contributing signals per cluster.
    """
    if not levels:
        return []

    sorted_levels = sorted(levels)
    clusters: list[list[float]] = [[sorted_levels[0]]]

    for lv in sorted_levels[1:]:
        cluster_mid = np.mean(clusters[-1])
        if abs(lv - cluster_mid) / max(cluster_mid, 1e-9) <= tolerance_pct:
            clusters[-1].append(lv)
        else:
            clusters.append([lv])

    result = []
    for c in clusters:
        result.append({
            "price_mid": round(float(np.mean(c)), 4),
            "price_low": round(float(min(c)), 4),
            "price_high": round(float(max(c)), 4),
            "n_signals": len(c),
        })

    return sorted(result, key=lambda x: x["n_signals"], reverse=True)


def detect_confluence_nodes(
    df: pd.DataFrame,
    symbol: str = "",
    patterns: list[dict] | None = None,
) -> list[dict[str, Any]]:
    """Detect price-time confluence nodes for a symbol.

    Combines:
    - Cycle projections (FFT + wavelet + Hilbert → next peak/trough)
    - Geometry levels (Square-of-9, Square-of-144)
    - Price-time symmetry
    - Pattern breakout/target levels
    - Swing structure (recent support/resistance)

    Returns list of Decision Nodes sorted by confluence score.
    """
    nodes: list[dict[str, Any]] = []

    if df is None or len(df) < 60:
        return nodes

    df = df.sort_values("date").reset_index(drop=True).copy()
    close = df["close"].astype(float).values
    current_price = float(close[-1])
    last_date_str = str(df["date"].iloc[-1])

    # ── Collect price levels from all engines ─────────────────────────────

    support_levels: list[float] = []
    resistance_levels: list[float] = []
    signal_descriptions: list[str] = []

    # 1. Cycle projections
    try:
        cycle_data = project_cycle(df)
        if cycle_data["next_trough_price"] and cycle_data["next_trough_price"] > 0:
            support_levels.append(cycle_data["next_trough_price"])
            signal_descriptions.append(
                f"Cycle trough projected at {cycle_data['next_trough_price']:.2f} "
                f"({cycle_data.get('dominant_cycle_length', 0):.0f}-day cycle)"
            )
        if cycle_data["next_peak_price"] and cycle_data["next_peak_price"] > 0:
            resistance_levels.append(cycle_data["next_peak_price"])
            signal_descriptions.append(
                f"Cycle peak projected at {cycle_data['next_peak_price']:.2f}"
            )
    except Exception as exc:
        logger.debug("Cycle projection failed: %s", exc)
        cycle_data = {}

    # 2. Square-of-9 levels
    try:
        sq9_sup, sq9_res = sq9_nearest_levels(current_price)
        support_levels.append(sq9_sup)
        resistance_levels.append(sq9_res)
        signal_descriptions.append(f"Sq-of-9: S={sq9_sup:.2f} R={sq9_res:.2f}")
    except Exception as exc:
        logger.debug("Square-of-9 failed: %s", exc)

    # 3. Square-of-144 levels
    try:
        sq144_sup, sq144_res = sq144_nearest_levels(current_price)
        support_levels.append(sq144_sup)
        resistance_levels.append(sq144_res)
        signal_descriptions.append(f"Sq-of-144: S={sq144_sup:.2f} R={sq144_res:.2f}")
    except Exception as exc:
        logger.debug("Square-of-144 failed: %s", exc)

    # 4. Swing structure levels
    try:
        adaptive_mm = compute_adaptive_minimum_move(df)
        swings = detect_swings(df, minimum_move=adaptive_mm)
        for sl in swings["swing_lows"][-3:]:
            support_levels.append(float(sl["price"]))
        for sh in swings["swing_highs"][-3:]:
            resistance_levels.append(float(sh["price"]))
        signal_descriptions.append("Swing structure S/R levels")
    except Exception as exc:
        logger.debug("Swing detection failed: %s", exc)

    # 5. Pattern levels
    if patterns:
        for p in patterns:
            status = p.get("status", "NOT_PRESENT")
            if status in ("NOT_PRESENT", "FAILED"):
                continue
            bl = p.get("breakout_level")
            tgt = p.get("target", p.get("projected_target"))
            inv = p.get("invalidation_level")
            if bl:
                resistance_levels.append(float(bl))
                signal_descriptions.append(f"Pattern {p.get('pattern_name', '?')}: breakout={bl:.2f}")
            if tgt:
                resistance_levels.append(float(tgt))
            if inv:
                support_levels.append(float(inv))

    # ── Cluster levels into zones ─────────────────────────────────────────

    support_clusters = _cluster_levels(support_levels)
    resistance_clusters = _cluster_levels(resistance_levels)

    # ── Build confluence nodes ────────────────────────────────────────────

    # Price-time symmetry adds to confidence
    try:
        symmetry = compute_price_time_symmetry(df)
        symmetry_score = symmetry.get("symmetry_score", 0.0)
    except Exception:
        symmetry_score = 0.0

    # Cycle alignment adds to confidence
    cycle_alignment = cycle_data.get("cycle_alignment_score", 0.0)

    # Time window from cycle projection
    next_peak_date = cycle_data.get("next_peak_date")
    next_trough_date = cycle_data.get("next_trough_date")

    for cluster in support_clusters[:3]:  # top 3 support zones
        n_sigs = cluster["n_signals"]
        base_score = min(n_sigs * 20, 60)  # 20 points per converging signal, cap 60
        bonus = symmetry_score * 0.2 + cycle_alignment * 0.2
        score = float(np.clip(base_score + bonus, 0, 100))

        time_start = next_trough_date if next_trough_date else date.today() + timedelta(days=1)
        time_end = time_start + timedelta(days=int(cycle_data.get("dominant_cycle_length", 10) * 0.3)) if isinstance(time_start, date) else time_start

        component_scores = {
            "swing_structure": min(n_sigs, 3) * 20,
            "cycle": round(cycle_alignment, 1),
            "geometry": 20 if n_sigs >= 2 else 0,
            "symmetry": round(symmetry_score, 1),
        }

        nodes.append({
            "symbol": symbol,
            "price_low": cluster["price_low"],
            "price_high": cluster["price_high"],
            "time_window_start": str(time_start),
            "time_window_end": str(time_end),
            "confluence_score": round(score, 2),
            "component_scores": json.dumps(component_scores),
            "supporting_signals": json.dumps(signal_descriptions[:8]),
            "node_type": "support",
            "direction": "bullish",
            "status": "upcoming",
            "n_signals": n_sigs,
        })

    for cluster in resistance_clusters[:3]:  # top 3 resistance zones
        n_sigs = cluster["n_signals"]
        base_score = min(n_sigs * 20, 60)
        bonus = symmetry_score * 0.2 + cycle_alignment * 0.2
        score = float(np.clip(base_score + bonus, 0, 100))

        time_start = next_peak_date if next_peak_date else date.today() + timedelta(days=1)
        time_end = time_start + timedelta(days=int(cycle_data.get("dominant_cycle_length", 10) * 0.3)) if isinstance(time_start, date) else time_start

        component_scores = {
            "swing_structure": min(n_sigs, 3) * 20,
            "cycle": round(cycle_alignment, 1),
            "geometry": 20 if n_sigs >= 2 else 0,
            "symmetry": round(symmetry_score, 1),
        }

        nodes.append({
            "symbol": symbol,
            "price_low": cluster["price_low"],
            "price_high": cluster["price_high"],
            "time_window_start": str(time_start),
            "time_window_end": str(time_end),
            "confluence_score": round(score, 2),
            "component_scores": json.dumps(component_scores),
            "supporting_signals": json.dumps(signal_descriptions[:8]),
            "node_type": "resistance",
            "direction": "bearish",
            "status": "upcoming",
            "n_signals": n_sigs,
        })

    nodes.sort(key=lambda n: n["confluence_score"], reverse=True)
    return nodes
