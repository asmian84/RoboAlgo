"""
Elliott Wave Counter — institutional-grade wave detection engine.

Uses adaptive zigzag pivot detection → monowave segmentation → impulse/corrective
wave pattern matching with strict rule enforcement and Fibonacci scoring.

Three absolute Elliott Wave rules (violations = instant disqualification):
  Rule 1: Wave 2 never retraces more than 100% of Wave 1
  Rule 2: Wave 3 is never the shortest impulse wave (1, 3, 5)
  Rule 3: Wave 4 never enters the price territory of Wave 1

Fibonacci scoring (guidelines, not rules — affect confidence):
  - Wave 2 retraces 50–61.8% of Wave 1 (ideal)
  - Wave 3 extends 161.8–261.8% of Wave 1 (ideal)
  - Wave 4 retraces 23.6–38.2% of Wave 3 (ideal)
  - Wave 5 extends 61.8–100% of Wave 1 from Wave 4 end (ideal)

Multi-degree labeling:
  - Primary:      (1)(2)(3)(4)(5)  /  (A)(B)(C)
  - Intermediate:  1 2 3 4 5       /   A B C
  - Minor:         i ii iii iv v   /   a b c

References: Frost & Prechter "Elliott Wave Principle", Neely "Mastering Elliott Wave"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

# Fibonacci ratios used in scoring
FIB_RATIOS = {
    "0.236": 0.236,
    "0.382": 0.382,
    "0.500": 0.500,
    "0.618": 0.618,
    "0.786": 0.786,
    "1.000": 1.000,
    "1.272": 1.272,
    "1.618": 1.618,
    "2.000": 2.000,
    "2.618": 2.618,
    "3.618": 3.618,
    "4.236": 4.236,
}

# Wave labels per degree
DEGREE_LABELS = {
    "primary":      {"impulse": ["(1)", "(2)", "(3)", "(4)", "(5)"],
                     "corrective": ["(A)", "(B)", "(C)"],
                     "triangle": ["(A)", "(B)", "(C)", "(D)", "(E)"],
                     "diagonal": ["(1)", "(2)", "(3)", "(4)", "(5)"],
                     "double_correction": ["(W)", "(X)", "(Y)"]},
    "intermediate": {"impulse": ["1", "2", "3", "4", "5"],
                     "corrective": ["A", "B", "C"],
                     "triangle": ["A", "B", "C", "D", "E"],
                     "diagonal": ["1", "2", "3", "4", "5"],
                     "double_correction": ["W", "X", "Y"]},
    "minor":        {"impulse": ["i", "ii", "iii", "iv", "v"],
                     "corrective": ["a", "b", "c"],
                     "triangle": ["a", "b", "c", "d", "e"],
                     "diagonal": ["i", "ii", "iii", "iv", "v"],
                     "double_correction": ["w", "x", "y"]},
}


class WaveType(Enum):
    # Motive
    IMPULSE          = "impulse"
    IMPULSE_EXT_W1   = "impulse_ext_w1"
    IMPULSE_EXT_W3   = "impulse_ext_w3"   # most common in stocks
    IMPULSE_EXT_W5   = "impulse_ext_w5"   # most common in commodities
    TRUNCATED_5TH    = "truncated_5th"
    # Corrective
    ZIGZAG               = "zigzag"
    FLAT_REGULAR         = "flat_regular"
    FLAT_EXPANDED        = "flat_expanded"
    FLAT_RUNNING         = "flat_running"
    CONTRACTING_TRIANGLE = "contracting_triangle"
    EXPANDING_TRIANGLE   = "expanding_triangle"
    # Diagonals
    ENDING_DIAGONAL  = "ending_diagonal"
    LEADING_DIAGONAL = "leading_diagonal"
    # Legacy / fallback
    CORRECTIVE        = "corrective"
    TRIANGLE          = "triangle"
    DIAGONAL          = "diagonal"
    DOUBLE_CORRECTION = "double_correction"
    UNKNOWN           = "unknown"


class WaveDirection(Enum):
    UP   = "up"
    DOWN = "down"


@dataclass
class Pivot:
    """A swing high or swing low point."""
    idx: int
    price: float
    is_high: bool
    strength: int = 1

    @property
    def is_low(self) -> bool:
        return not self.is_high


@dataclass
class MonoWave:
    """A single directional price move between two consecutive pivots."""
    start: Pivot
    end:   Pivot
    direction:             WaveDirection = field(init=False)
    price_change:          float         = field(init=False)
    bar_count:             int           = field(init=False)
    # Layer 2 classification (set by classify_internal_structure)
    internal_structure:    str = field(default="indeterminate")  # "impulsive"|"corrective"|"indeterminate"
    classification_method: str = field(default="unknown")        # "recursive"|"heuristic"|"unknown"
    sub_pivot_count:       int = field(default=0)

    def __post_init__(self):
        self.direction    = WaveDirection.UP if self.end.price > self.start.price else WaveDirection.DOWN
        self.price_change = abs(self.end.price - self.start.price)
        self.bar_count    = abs(self.end.idx - self.start.idx)


@dataclass
class WaveCount:
    """A complete or partial wave count."""
    waves:           list[MonoWave]
    wave_type:       WaveType
    direction:       WaveDirection
    degree:          str   = "intermediate"
    confidence:      float = 0.0
    fib_score:       float = 0.0
    rule_violations: list[str]          = field(default_factory=list)
    labels:          list[str]          = field(default_factory=list)
    is_complete:     bool  = False
    sub_counts:      list[Optional["WaveCount"]] = field(default_factory=list)
    # v2 additions
    extension_wave:      Optional[int]  = None   # 1, 3, or 5
    invalidation_level:  float          = 0.0
    alternation_quality: str            = "unknown"  # "good"|"fair"|"poor"|"unknown"
    position_bonus:      float          = 0.0


# ── Zigzag Pivot Detection ───────────────────────────────────────────────────

def detect_pivots(
    df: pd.DataFrame,
    atr_multiplier: float = 0.5,
    min_lookback: int = 3,
    max_lookback: int = 10,
) -> list[Pivot]:
    """
    Adaptive zigzag pivot detection using ATR-scaled thresholds.

    Unlike fixed-period pivots, this adapts to volatility:
    - High volatility → larger swings required → fewer, more significant pivots
    - Low volatility → smaller swings matter → more granular pivots

    Returns alternating high/low pivots (no two consecutive highs or lows).
    """
    high = df["high"].astype(float).values
    low = df["low"].astype(float).values
    close = df["close"].astype(float).values
    n = len(df)

    if n < 20:
        return []

    # Compute ATR for adaptive threshold
    atr = _compute_atr(high, low, close, period=14)
    threshold = atr * atr_multiplier

    # Phase 1: Find ALL local extremes with adaptive lookback
    raw_pivots: list[Pivot] = []

    for lookback in range(min_lookback, max_lookback + 1):
        for i in range(lookback, n - lookback):
            # Check swing high
            is_high = True
            for j in range(1, lookback + 1):
                if high[i] <= high[i - j] or high[i] <= high[i + j]:
                    is_high = False
                    break
            if is_high:
                raw_pivots.append(Pivot(idx=i, price=float(high[i]), is_high=True, strength=lookback))

            # Check swing low
            is_low = True
            for j in range(1, lookback + 1):
                if low[i] >= low[i - j] or low[i] >= low[i + j]:
                    is_low = False
                    break
            if is_low:
                raw_pivots.append(Pivot(idx=i, price=float(low[i]), is_high=False, strength=lookback))

    if not raw_pivots:
        return []

    # Phase 2: Deduplicate — keep strongest pivot per cluster
    raw_pivots.sort(key=lambda p: p.idx)
    deduped: list[Pivot] = []
    i = 0
    while i < len(raw_pivots):
        cluster = [raw_pivots[i]]
        j = i + 1
        # Cluster pivots within 3 bars of each other with same direction
        while j < len(raw_pivots) and raw_pivots[j].idx - raw_pivots[i].idx <= 3:
            if raw_pivots[j].is_high == raw_pivots[i].is_high:
                cluster.append(raw_pivots[j])
            j += 1
        # Keep the one with highest strength, breaking ties by most extreme price
        best = max(cluster, key=lambda p: (p.strength, p.price if p.is_high else -p.price))
        deduped.append(best)
        i = j if j > i + 1 else i + 1

    # Phase 3: Enforce alternation (no two consecutive highs or lows)
    # This is the zigzag filter — guarantees clean wave structure
    alternating = _enforce_alternation(deduped, high, low, threshold)

    # Phase 4: Filter insignificant moves (< threshold ATR)
    significant = _filter_insignificant(alternating, threshold)

    return significant


def _compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
    """Compute Average True Range."""
    n = len(high)
    if n < period + 1:
        return float(np.mean(high - low)) if n > 0 else 1.0

    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )

    # Use last `period` bars for current ATR
    return float(np.mean(tr[-period:]))


def _enforce_alternation(pivots: list[Pivot], high: np.ndarray, low: np.ndarray, threshold: float) -> list[Pivot]:
    """Ensure alternating high/low pattern. When consecutive same-type pivots
    exist, keep the most extreme one."""
    if len(pivots) < 2:
        return pivots

    result: list[Pivot] = [pivots[0]]

    for i in range(1, len(pivots)):
        curr = pivots[i]
        prev = result[-1]

        if curr.is_high != prev.is_high:
            # Already alternating — good
            result.append(curr)
        else:
            # Same type — keep the more extreme one
            if curr.is_high:
                if curr.price > prev.price:
                    result[-1] = curr
            else:
                if curr.price < prev.price:
                    result[-1] = curr

    return result


def _filter_insignificant(pivots: list[Pivot], threshold: float) -> list[Pivot]:
    """Remove pivots that create moves smaller than threshold."""
    if len(pivots) < 3:
        return pivots

    result = list(pivots)
    changed = True
    max_passes = 20  # prevent infinite loop
    passes = 0

    while changed and len(result) >= 3 and passes < max_passes:
        changed = False
        passes += 1
        new_result: list[Pivot] = [result[0]]
        i = 1
        while i < len(result):
            move = abs(result[i].price - new_result[-1].price)
            if move < threshold * 0.3 and i < len(result) - 1:
                # Skip this pivot and next (merge the insignificant zigzag)
                i += 2
                changed = True
            else:
                new_result.append(result[i])
                i += 1
        result = new_result

    return result


# ── Macro Percentage-Based Pivot Detection ──────────────────────────────────

def detect_macro_pivots(
    df: pd.DataFrame,
    min_swing_pct: float = 0.15,
) -> list[Pivot]:
    """
    Detect major pivots using percentage-based swing thresholds.

    For macro (weekly/monthly) analysis, ATR-based detection finds too many
    small pivots. This function finds only the MAJOR turning points where
    price reversed by at least `min_swing_pct` (e.g., 15% = 0.15).

    Returns alternating high/low pivots representing the grand cycle structure.
    """
    high = df["high"].astype(float).values
    low = df["low"].astype(float).values
    n = len(df)

    if n < 20:
        return []

    # Zigzag algorithm with percentage threshold
    # Track current direction and last confirmed pivot
    pivots: list[Pivot] = []
    direction = 0  # 0=undecided, 1=up, -1=down

    # Start with the absolute low as first candidate
    curr_high_idx = 0
    curr_high_price = float(high[0])
    curr_low_idx = 0
    curr_low_price = float(low[0])

    for i in range(1, n):
        h = float(high[i])
        lo = float(low[i])

        if h > curr_high_price:
            curr_high_idx = i
            curr_high_price = h

        if lo < curr_low_price:
            curr_low_idx = i
            curr_low_price = lo

        if direction == 0:
            # Determine initial direction
            up_pct = (curr_high_price - curr_low_price) / max(curr_low_price, 0.01)
            if up_pct >= min_swing_pct:
                # First significant move is up — start with the low
                pivots.append(Pivot(idx=curr_low_idx, price=curr_low_price, is_high=False, strength=10))
                direction = 1
                curr_high_idx = curr_low_idx
                curr_high_price = float(high[curr_low_idx])
                # Re-scan for high from low point
                for j in range(curr_low_idx, i + 1):
                    if float(high[j]) > curr_high_price:
                        curr_high_idx = j
                        curr_high_price = float(high[j])
            down_pct = (curr_high_price - curr_low_price) / max(curr_high_price, 0.01)
            if direction == 0 and down_pct >= min_swing_pct:
                pivots.append(Pivot(idx=curr_high_idx, price=curr_high_price, is_high=True, strength=10))
                direction = -1
                curr_low_idx = curr_high_idx
                curr_low_price = float(low[curr_high_idx])
                for j in range(curr_high_idx, i + 1):
                    if float(low[j]) < curr_low_price:
                        curr_low_idx = j
                        curr_low_price = float(low[j])

        elif direction == 1:
            # Trending up — update high, check for reversal
            reversal_pct = (curr_high_price - lo) / max(curr_high_price, 0.01)
            if reversal_pct >= min_swing_pct:
                # Confirmed high pivot
                pivots.append(Pivot(idx=curr_high_idx, price=curr_high_price, is_high=True, strength=10))
                direction = -1
                curr_low_idx = i
                curr_low_price = lo

        elif direction == -1:
            # Trending down — update low, check for reversal
            reversal_pct = (h - curr_low_price) / max(curr_low_price, 0.01)
            if reversal_pct >= min_swing_pct:
                # Confirmed low pivot
                pivots.append(Pivot(idx=curr_low_idx, price=curr_low_price, is_high=False, strength=10))
                direction = 1
                curr_high_idx = i
                curr_high_price = h

    # Add final pivot (current position)
    if pivots:
        if direction == 1:
            pivots.append(Pivot(idx=curr_high_idx, price=curr_high_price, is_high=True, strength=10))
        elif direction == -1:
            pivots.append(Pivot(idx=curr_low_idx, price=curr_low_price, is_high=False, strength=10))

    return pivots


# ── MonoWave Builder ─────────────────────────────────────────────────────────

def build_monowaves(pivots: list[Pivot]) -> list[MonoWave]:
    """
    Convert a list of alternating pivots into a list of MonoWaves.
    Each MonoWave spans from pivot[i] to pivot[i+1].
    internal_structure defaults to "indeterminate" — Layer 2 fills this in.
    """
    if len(pivots) < 2:
        return []
    return [MonoWave(start=pivots[i], end=pivots[i + 1]) for i in range(len(pivots) - 1)]


# ── MonoWave Cache ───────────────────────────────────────────────────────────
# Keyed by (df_id, atr_multiplier) where df_id = id(df).
# Avoids re-classifying sub-wave structure on repeated analyze() calls.

_monowave_cache: dict[tuple, list[MonoWave]] = {}


def clear_monowave_cache() -> None:
    """Clear the monowave cache. Call when loading new price data."""
    _monowave_cache.clear()


def get_cached_monowaves(
    df: pd.DataFrame,
    atr_multiplier: float = 0.5,
) -> list[MonoWave]:
    """
    Return MonoWaves for this DataFrame, using cache when available.
    internal_structure is populated by classify_internal_structure() inside here.
    """
    cache_key = (id(df), atr_multiplier)
    if cache_key in _monowave_cache:
        return _monowave_cache[cache_key]

    pivots = detect_pivots(df, atr_multiplier=atr_multiplier)
    monowaves = build_monowaves(pivots)

    # Classify each monowave's internal structure (Layer 2)
    for mw in monowaves:
        structure, method, count = classify_internal_structure(
            df, mw.start.idx, mw.end.idx
        )
        mw.internal_structure    = structure
        mw.classification_method = method
        mw.sub_pivot_count       = count

    _monowave_cache[cache_key] = monowaves
    return monowaves


def classify_internal_structure(
    df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    min_bars_for_recursion: int = 8,
) -> tuple[str, str, int]:
    """Stub — implemented in Task 5."""
    return "indeterminate", "unknown", 0




# ── v2 Detection Engine ──────────────────────────────────────────────────────
# Functions below are the v2 bottom-up pipeline:
#   Layer 2: classify_internal_structure()  — already stubbed above
#   Layer 3: find_impulse_candidates(), find_zigzag_candidates(),
#            find_flat_candidates(), find_triangle_candidates(),
#            find_diagonal_candidates()     (Tasks 6-8)
#   Layer 4: validate_and_score()           (Task 9)
#   Layer 5: rank_counts(), analyze_multi_degree()  (Task 10)
# ────────────────────────────────────────────────────────────────────────────



def _validate_impulse_rules(waves: list[MonoWave], direction: WaveDirection) -> list[str]:
    """
    Validate the three absolute Elliott Wave rules for a 5-wave impulse.
    Returns list of violations (empty = valid).
    """
    violations = []

    w1, w2, w3, w4, w5 = waves

    # Rule 1: Wave 2 never retraces more than 100% of Wave 1
    if direction == WaveDirection.UP:
        if w2.end.price <= w1.start.price:
            violations.append("Rule 1: Wave 2 retraces > 100% of Wave 1")
    else:
        if w2.end.price >= w1.start.price:
            violations.append("Rule 1: Wave 2 retraces > 100% of Wave 1")

    # Rule 2: Wave 3 is never the shortest of waves 1, 3, 5
    w1_len = w1.price_change
    w3_len = w3.price_change
    w5_len = w5.price_change
    if w3_len < w1_len and w3_len < w5_len:
        violations.append("Rule 2: Wave 3 is the shortest impulse wave")

    # Rule 3: Wave 4 never overlaps Wave 1 price territory
    if direction == WaveDirection.UP:
        if w4.end.price < w1.end.price:
            violations.append("Rule 3: Wave 4 overlaps Wave 1 territory")
    else:
        if w4.end.price > w1.end.price:
            violations.append("Rule 3: Wave 4 overlaps Wave 1 territory")

    # Additional structural validation
    # Waves 1, 3, 5 must move in the impulse direction
    if direction == WaveDirection.UP:
        if w1.direction != WaveDirection.UP or w3.direction != WaveDirection.UP or w5.direction != WaveDirection.UP:
            violations.append("Structural: Impulse waves not in correct direction")
        if w2.direction != WaveDirection.DOWN or w4.direction != WaveDirection.DOWN:
            violations.append("Structural: Corrective waves not opposing")
    else:
        if w1.direction != WaveDirection.DOWN or w3.direction != WaveDirection.DOWN or w5.direction != WaveDirection.DOWN:
            violations.append("Structural: Impulse waves not in correct direction")
        if w2.direction != WaveDirection.UP or w4.direction != WaveDirection.UP:
            violations.append("Structural: Corrective waves not opposing")

    return violations



# ── Elliott Wave Channel Computation ───────────────────────────────────────
#
# Two channel types for impulse waves:
#
# 1. BASE CHANNEL (available from wave 2 onward):
#    - Lower line: wave 1 start → wave 2 end (connecting the two lows for bullish)
#    - Upper line: parallel through wave 1 end (the high)
#    Used to project wave 3's target and confirm impulse development.
#
# 2. ACCELERATION CHANNEL (available from wave 4 onward):
#    - Lower line: wave 2 end → wave 4 end (connecting the corrective lows)
#    - Upper line: parallel through wave 3 end (the peak)
#    Wave 5 rides inside this channel; breakout below signals ABC correction.
#
# For corrective waves:
#    - A-C line with parallel through B
#    Breakout of corrective channel signals new impulse.

def compute_ew_channels(
    impulse_wc: Optional[WaveCount],
    correction_wc: Optional[WaveCount] = None,
) -> list[dict[str, Any]]:
    """
    Compute channel lines for an impulse (and optionally its linked correction).

    Returns a list of channel dicts, each containing:
      - type: "base" | "acceleration" | "corrective"
      - lower: {start_idx, start_price, end_idx, end_price}  (the anchor line)
      - upper: {start_idx, start_price, end_idx, end_price}  (the parallel)
      - direction: "up" | "down"
      - breakout: null | {idx, price, new_phase}  (if breakout detected)
    """
    channels = []

    if impulse_wc and impulse_wc.wave_type == WaveType.IMPULSE and len(impulse_wc.waves) >= 2:
        waves = impulse_wc.waves
        is_up = impulse_wc.direction == WaveDirection.UP

        # ── Base channel (waves 1-2) ──────────────────────────────────
        w1 = waves[0]
        w2 = waves[1]

        # Anchor line: connects the two retracement endpoints
        # For bullish: w1.start (low) → w2.end (higher low) = support
        # Parallel through w1.end (the first peak)
        anchor_start_idx = w1.start.idx
        anchor_start_price = w1.start.price
        anchor_end_idx = w2.end.idx
        anchor_end_price = w2.end.price

        parallel_ref_idx = w1.end.idx
        parallel_ref_price = w1.end.price

        # Compute parallel offset: distance from anchor line at parallel_ref_idx
        if anchor_end_idx != anchor_start_idx:
            slope = (anchor_end_price - anchor_start_price) / (anchor_end_idx - anchor_start_idx)
            anchor_at_ref = anchor_start_price + slope * (parallel_ref_idx - anchor_start_idx)
            offset = parallel_ref_price - anchor_at_ref

            # Extend the channel to the last wave end
            last_wave = waves[-1]
            extend_idx = last_wave.end.idx + max(10, int((last_wave.end.idx - anchor_start_idx) * 0.3))

            channels.append({
                "type": "base",
                "direction": "up" if is_up else "down",
                "lower": {
                    "start_idx": anchor_start_idx,
                    "start_price": round(anchor_start_price, 4),
                    "end_idx": extend_idx,
                    "end_price": round(anchor_start_price + slope * (extend_idx - anchor_start_idx), 4),
                },
                "upper": {
                    "start_idx": anchor_start_idx,
                    "start_price": round(anchor_start_price + offset, 4),
                    "end_idx": extend_idx,
                    "end_price": round(anchor_start_price + offset + slope * (extend_idx - anchor_start_idx), 4),
                },
                "breakout": None,
            })

        # ── Acceleration channel (waves 2-4) ──────────────────────────
        if len(waves) >= 4:
            w3 = waves[2]
            w4 = waves[3]

            # Anchor line: w2.end → w4.end (connecting corrective endpoints)
            a2_start_idx = w2.end.idx
            a2_start_price = w2.end.price
            a2_end_idx = w4.end.idx
            a2_end_price = w4.end.price

            # Parallel through w3.end
            a2_ref_idx = w3.end.idx
            a2_ref_price = w3.end.price

            if a2_end_idx != a2_start_idx:
                a2_slope = (a2_end_price - a2_start_price) / (a2_end_idx - a2_start_idx)
                a2_at_ref = a2_start_price + a2_slope * (a2_ref_idx - a2_start_idx)
                a2_offset = a2_ref_price - a2_at_ref

                last_wave = waves[-1]
                a2_extend = last_wave.end.idx + max(15, int((last_wave.end.idx - a2_start_idx) * 0.4))

                # Detect channel breakout after wave 5
                breakout_info = None
                if len(waves) >= 5:
                    w5 = waves[4]
                    # After wave 5, check if price broke the channel
                    # For bullish: breakout below the 2-4 line signals correction
                    # For bearish: breakout above the 2-4 line signals correction
                    w5_end_on_anchor = a2_start_price + a2_slope * (w5.end.idx - a2_start_idx)
                    if is_up and w5.end.price < w5_end_on_anchor:
                        breakout_info = {
                            "idx": w5.end.idx,
                            "price": round(w5.end.price, 4),
                            "new_phase": "corrective",
                            "label": "CHANNEL BREAK → ABC",
                        }
                    elif not is_up and w5.end.price > w5_end_on_anchor:
                        breakout_info = {
                            "idx": w5.end.idx,
                            "price": round(w5.end.price, 4),
                            "new_phase": "corrective",
                            "label": "CHANNEL BREAK → ABC",
                        }

                channels.append({
                    "type": "acceleration",
                    "direction": "up" if is_up else "down",
                    "lower": {
                        "start_idx": a2_start_idx,
                        "start_price": round(a2_start_price, 4),
                        "end_idx": a2_extend,
                        "end_price": round(a2_start_price + a2_slope * (a2_extend - a2_start_idx), 4),
                    },
                    "upper": {
                        "start_idx": a2_start_idx,
                        "start_price": round(a2_start_price + a2_offset, 4),
                        "end_idx": a2_extend,
                        "end_price": round(a2_start_price + a2_offset + a2_slope * (a2_extend - a2_start_idx), 4),
                    },
                    "breakout": breakout_info,
                })

    # ── Corrective channel (A-B-C) ────────────────────────────────────
    if correction_wc and len(correction_wc.waves) >= 2:
        c_waves = correction_wc.waves
        is_corr_up = correction_wc.direction == WaveDirection.UP

        wA = c_waves[0]
        wB = c_waves[1]

        # Anchor: A start → B end (or A end → C end if 3 waves)
        if len(c_waves) >= 3:
            wC = c_waves[2]
            # A-C line (connecting the two impulse-direction ends)
            c_anchor_start_idx = wA.end.idx
            c_anchor_start_price = wA.end.price
            c_anchor_end_idx = wC.end.idx
            c_anchor_end_price = wC.end.price
            # Parallel through B end
            c_ref_idx = wB.end.idx
            c_ref_price = wB.end.price
            c_extend = wC.end.idx + max(15, int((wC.end.idx - wA.start.idx) * 0.5))
        else:
            # Only A-B: use A.start → B.end as anchor, parallel through A.end
            c_anchor_start_idx = wA.start.idx
            c_anchor_start_price = wA.start.price
            c_anchor_end_idx = wB.end.idx
            c_anchor_end_price = wB.end.price
            c_ref_idx = wA.end.idx
            c_ref_price = wA.end.price
            c_extend = wB.end.idx + max(15, int((wB.end.idx - wA.start.idx) * 0.5))

        if c_anchor_end_idx != c_anchor_start_idx:
            c_slope = (c_anchor_end_price - c_anchor_start_price) / (c_anchor_end_idx - c_anchor_start_idx)
            c_at_ref = c_anchor_start_price + c_slope * (c_ref_idx - c_anchor_start_idx)
            c_offset = c_ref_price - c_at_ref

            # Detect breakout of corrective channel → new impulse
            breakout_info = None
            if len(c_waves) >= 3:
                wC_end = c_waves[-1]
                # After C, if price breaks the B-side parallel, new impulse starts
                c_end_on_parallel = c_anchor_start_price + c_offset + c_slope * (wC_end.end.idx - c_anchor_start_idx)
                if is_corr_up and wC_end.end.price > c_end_on_parallel:
                    breakout_info = {
                        "idx": wC_end.end.idx,
                        "price": round(wC_end.end.price, 4),
                        "new_phase": "impulse",
                        "label": "CHANNEL BREAK → NEW IMPULSE",
                    }
                elif not is_corr_up and wC_end.end.price < c_end_on_parallel:
                    breakout_info = {
                        "idx": wC_end.end.idx,
                        "price": round(wC_end.end.price, 4),
                        "new_phase": "impulse",
                        "label": "CHANNEL BREAK → NEW IMPULSE",
                    }

            channels.append({
                "type": "corrective",
                "direction": "up" if is_corr_up else "down",
                "lower": {
                    "start_idx": c_anchor_start_idx,
                    "start_price": round(c_anchor_start_price, 4),
                    "end_idx": c_extend,
                    "end_price": round(c_anchor_start_price + c_slope * (c_extend - c_anchor_start_idx), 4),
                },
                "upper": {
                    "start_idx": c_anchor_start_idx,
                    "start_price": round(c_anchor_start_price + c_offset, 4),
                    "end_idx": c_extend,
                    "end_price": round(c_anchor_start_price + c_offset + c_slope * (c_extend - c_anchor_start_idx), 4),
                },
                "breakout": breakout_info,
            })

    return channels
