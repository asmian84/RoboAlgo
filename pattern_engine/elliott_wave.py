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
    """
    Determine whether the price move from start_idx to end_idx is:
      - "impulsive"    : internally structured as 5 waves
      - "corrective"   : internally structured as 3 waves / overlapping
      - "indeterminate": too few bars or ambiguous

    Returns (structure, method, sub_pivot_count) where:
      method = "recursive" | "heuristic" | "unknown"

    Hybrid approach (Brown):
      - If (end_idx - start_idx) >= min_bars_for_recursion: recursive sub-pivot detection
      - Otherwise: proportion heuristic (max counter-move ratio)
    """
    bar_count = end_idx - start_idx

    if bar_count < 4:
        return "indeterminate", "unknown", 0

    if bar_count < min_bars_for_recursion:
        return _classify_by_heuristic(df, start_idx, end_idx)

    return _classify_by_recursion(df, start_idx, end_idx)


def _classify_by_heuristic(
    df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
) -> tuple[str, str, int]:
    """
    Brown's box method: if max counter-move > 38.2% of total move → corrective.
    If max counter-move < 10% → impulsive. Between 10–38.2% → indeterminate.
    """
    sl = slice(start_idx, end_idx + 1)
    closes = df["close"].iloc[sl].values.astype(float)

    if len(closes) < 3:
        return "indeterminate", "unknown", 0

    total_move = abs(closes[-1] - closes[0])
    if total_move < 1e-9:
        return "indeterminate", "heuristic", 0

    direction = 1 if closes[-1] > closes[0] else -1
    peak = closes[0]
    max_retrace_ratio = 0.0

    for price in closes[1:]:
        if direction == 1:
            if price > peak:
                peak = price
            retrace = (peak - price) / total_move
        else:
            if price < peak:
                peak = price
            retrace = (price - peak) / total_move
        if retrace > max_retrace_ratio:
            max_retrace_ratio = retrace

    if max_retrace_ratio > 0.382:
        return "corrective", "heuristic", 0
    elif max_retrace_ratio < 0.10:
        return "impulsive", "heuristic", 0
    return "indeterminate", "heuristic", 0


def _classify_by_recursion(
    df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
) -> tuple[str, str, int]:
    """
    Detect sub-pivots within the bar range using a tighter ATR multiplier.
    If ≥4 sub-pivots satisfy 5-wave rules → impulsive.
    If structure is overlapping / 3-wave → corrective.
    """
    # Slice the DataFrame to the monowave range
    sub_df = df.iloc[start_idx: end_idx + 1].reset_index(drop=True)

    if len(sub_df) < 8:
        return _classify_by_heuristic(df, start_idx, end_idx)

    # Use a tighter ATR multiplier to find sub-pivots within this wave
    # Also use smaller lookback for finer granularity
    sub_pivots = _detect_sub_pivots(sub_df)

    # Add boundary pivots: the start and end of the move are implicit pivots
    # that detect_pivots doesn't include (it only finds internal extremes)
    sub_pivots = _add_boundary_pivots(sub_df, sub_pivots)

    if len(sub_pivots) < 2:
        # Fall back to heuristic if we can't find structure
        return _classify_by_heuristic(df, start_idx, end_idx)

    if len(sub_pivots) < 4:
        # 2–3 sub-pivots: check if structure is 3-wave corrective
        if _is_three_wave(sub_pivots, sub_df):
            return "corrective", "recursive", len(sub_pivots)
        return "indeterminate", "recursive", len(sub_pivots)

    # 4+ sub-pivots: check for 5-wave impulse structure
    if _is_five_wave_impulse(sub_pivots, sub_df):
        return "impulsive", "recursive", len(sub_pivots)

    if _has_heavy_overlap(sub_pivots):
        return "corrective", "recursive", len(sub_pivots)

    # Ambiguous — use heuristic as tiebreaker on the raw price data
    heuristic_result, _, _ = _classify_by_heuristic(df, start_idx, end_idx)
    if heuristic_result != "indeterminate":
        return heuristic_result, "recursive", len(sub_pivots)

    return "indeterminate", "recursive", len(sub_pivots)


def _add_boundary_pivots(sub_df: pd.DataFrame, sub_pivots: list[Pivot]) -> list[Pivot]:
    """
    Add the start and end of the sub-DataFrame as boundary pivots if they
    complement the alternation pattern. The start/end of a monowave are
    always pivots at the parent level — detect_pivots only finds internal ones.
    """
    n = len(sub_df)
    if n < 2:
        return sub_pivots

    close = sub_df["close"].values
    high = sub_df["high"].values
    low = sub_df["low"].values

    result = list(sub_pivots)

    # Determine overall direction of this monowave
    overall_up = close[-1] > close[0]

    # Start boundary: if moving up, start is a LOW; if down, start is a HIGH
    if overall_up:
        start_pivot = Pivot(idx=0, price=float(low[0]), is_high=False, strength=0)
    else:
        start_pivot = Pivot(idx=0, price=float(high[0]), is_high=True, strength=0)

    # End boundary: if moving up, end is a HIGH; if down, end is a LOW
    if overall_up:
        end_pivot = Pivot(idx=n - 1, price=float(high[n - 1]), is_high=True, strength=0)
    else:
        end_pivot = Pivot(idx=n - 1, price=float(low[n - 1]), is_high=False, strength=0)

    # Only add start if it doesn't conflict with the first found pivot
    if not result or result[0].is_high != start_pivot.is_high:
        if not result or abs(result[0].idx - start_pivot.idx) > 1:
            result.insert(0, start_pivot)

    # Only add end if it doesn't conflict with the last found pivot
    if not result or result[-1].is_high != end_pivot.is_high:
        if not result or abs(result[-1].idx - end_pivot.idx) > 1:
            result.append(end_pivot)

    return result


def _detect_sub_pivots(sub_df: pd.DataFrame) -> list[Pivot]:
    """
    Find sub-pivots within a monowave's bar range.
    Uses progressively looser settings until we get at least 4 pivots,
    or gives up and returns whatever we found.
    """
    n = len(sub_df)

    # For shorter sub-DataFrames, detect_pivots returns [] (needs ≥20 bars).
    # Use a simplified local pivot detector instead.
    if n < 20:
        return _simple_pivot_detect(sub_df)

    # Try tighter settings first, then loosen
    for atr_mult in (0.2, 0.3, 0.5):
        try:
            pivots = detect_pivots(sub_df, atr_multiplier=atr_mult, min_lookback=2, max_lookback=4)
            if len(pivots) >= 4:
                return pivots
        except Exception:
            continue

    # Final attempt with smallest multiplier
    try:
        return detect_pivots(sub_df, atr_multiplier=0.15, min_lookback=1, max_lookback=3)
    except Exception:
        return _simple_pivot_detect(sub_df)


def _simple_pivot_detect(sub_df: pd.DataFrame) -> list[Pivot]:
    """
    Ultra-simple pivot detection for short windows (<20 bars).
    Finds local highs/lows with lookback=1, enforces alternation.
    """
    high = sub_df["high"].astype(float).values
    low = sub_df["low"].astype(float).values
    n = len(sub_df)

    if n < 3:
        return []

    pivots: list[Pivot] = []
    for i in range(1, n - 1):
        if high[i] >= high[i - 1] and high[i] >= high[i + 1]:
            pivots.append(Pivot(idx=i, price=float(high[i]), is_high=True, strength=1))
        if low[i] <= low[i - 1] and low[i] <= low[i + 1]:
            pivots.append(Pivot(idx=i, price=float(low[i]), is_high=False, strength=1))

    if not pivots:
        return []

    # Sort by index
    pivots.sort(key=lambda p: (p.idx, not p.is_high))

    # Enforce alternation: no consecutive highs or lows
    alternating: list[Pivot] = [pivots[0]]
    for p in pivots[1:]:
        if p.is_high != alternating[-1].is_high:
            alternating.append(p)
        else:
            # Keep the more extreme one
            if p.is_high and p.price > alternating[-1].price:
                alternating[-1] = p
            elif not p.is_high and p.price < alternating[-1].price:
                alternating[-1] = p

    return alternating


def _is_five_wave_impulse(sub_pivots: list[Pivot], sub_df: pd.DataFrame) -> bool:
    """
    Check if sub_pivots contain a valid 5-wave structure.
    Uses a sliding window of 6 pivots (5 waves).
    """
    n = len(sub_pivots)
    # Try every 6-pivot window
    for i in range(n - 5):
        window = sub_pivots[i: i + 6]
        # Build MonoWaves from the 6 pivots (5 waves)
        mws = build_monowaves(window)
        if len(mws) != 5:
            continue
        # Check alternating direction
        if not all(mws[j].direction != mws[j + 1].direction for j in range(4)):
            continue
        # Determine direction from first wave
        direction = mws[0].direction
        # Apply 3 inviolable rules
        violations = _validate_impulse_rules(mws, direction)
        if not violations:
            return True
    return False


def _is_three_wave(sub_pivots: list[Pivot], sub_df: pd.DataFrame) -> bool:
    """
    Check if sub_pivots look like a 3-wave (A-B-C) corrective structure.
    B must not retrace all of A, and structure should be alternating.
    """
    if len(sub_pivots) < 4:
        # Need at least 4 pivots for 3 waves (start, A-end, B-end, C-end)
        # With fewer, try using what we have
        if len(sub_pivots) == 3:
            mws = build_monowaves(sub_pivots)
            if len(mws) < 2:
                return False
            # 2 waves: check if they alternate direction (A-B pattern at minimum)
            return mws[0].direction != mws[1].direction
        return False

    # Use first 4 pivots: start, A-end, B-end, C-end
    mws = build_monowaves(sub_pivots[:4])
    if len(mws) < 3:
        return False
    a, b, c = mws[0], mws[1], mws[2]
    # Must alternate direction
    if a.direction == b.direction or b.direction == c.direction:
        return False
    # B must be counter-trend but not retrace all of A
    if a.direction == WaveDirection.UP:
        b_retraces_all = b.end.price <= a.start.price
    else:
        b_retraces_all = b.end.price >= a.start.price
    return not b_retraces_all


def _has_heavy_overlap(sub_pivots: list[Pivot]) -> bool:
    """
    Check for overlapping wave structure (hallmark of corrective pattern).
    If waves frequently revisit the same price territory, it's corrective.

    Method: For each pair of same-direction waves (i, i+2), check if the
    later wave's range overlaps significantly with the earlier wave's range.
    """
    if len(sub_pivots) < 5:
        return False

    mws = build_monowaves(sub_pivots)
    if len(mws) < 4:
        return False

    overlap_count = 0
    total_checks = 0

    for i in range(len(mws) - 2):
        w_a = mws[i]
        w_b = mws[i + 2]  # Same direction wave (2 positions later)

        if w_a.direction != w_b.direction:
            continue

        total_checks += 1

        # Check price range overlap
        a_lo = min(w_a.start.price, w_a.end.price)
        a_hi = max(w_a.start.price, w_a.end.price)
        b_lo = min(w_b.start.price, w_b.end.price)
        b_hi = max(w_b.start.price, w_b.end.price)

        overlap = max(0, min(a_hi, b_hi) - max(a_lo, b_lo))
        a_range = a_hi - a_lo

        if a_range > 0 and overlap / a_range > 0.30:
            overlap_count += 1

    if total_checks == 0:
        return False

    return overlap_count / total_checks > 0.50




# ── v2 Detection Engine ──────────────────────────────────────────────────────
# Functions below are the v2 bottom-up pipeline:
#   Layer 2: classify_internal_structure()  — already stubbed above
#   Layer 3: find_impulse_candidates(), find_zigzag_candidates(),
#            find_flat_candidates(), find_triangle_candidates(),
#            find_diagonal_candidates()     (Tasks 6-8)
#   Layer 4: validate_and_score()           (Task 9)
#   Layer 5: rank_counts(), analyze_multi_degree()  (Task 10)
# ────────────────────────────────────────────────────────────────────────────


# ── Layer 3: Pattern Candidate Finders ───────────────────────────────────────

def find_impulse_candidates(
    monowaves: list[MonoWave],
    is_commodity: bool = False,
) -> list[dict]:
    """
    Scan MonoWave sequences for valid impulse wave candidates.

    For each window of 5 consecutive MonoWaves:
    1. Check alternating direction
    2. Check internal structures (actionary=impulsive, corrective=corrective)
       — "indeterminate" is treated permissively
    3. Apply 3 inviolable rules (hard gate)
    4. Detect which wave is extended (W3 bias for stocks, W5 bias for commodities)

    Returns list of candidate dicts (not yet scored by Layer 4).
    """
    candidates = []
    n = len(monowaves)

    for i in range(n - 4):
        seq = monowaves[i: i + 5]
        w1, w2, w3, w4, w5 = seq

        # Must alternate direction
        if not (w1.direction != w2.direction and
                w2.direction != w3.direction and
                w3.direction != w4.direction and
                w4.direction != w5.direction):
            continue

        # Actionary (1,3,5) must be impulsive or indeterminate
        # Corrective (2,4) must be corrective or indeterminate
        if w1.internal_structure == "corrective":
            continue
        if w3.internal_structure == "corrective":
            continue
        if w5.internal_structure == "corrective":
            continue
        if w2.internal_structure == "impulsive":
            continue
        if w4.internal_structure == "impulsive":
            continue

        # Hard gate: 3 inviolable rules
        direction = w1.direction
        violations = _validate_impulse_rules([w1, w2, w3, w4, w5], direction)
        if violations:
            continue

        # Detect extension
        sizes = [w1.price_change, w3.price_change, w5.price_change]
        max_size = max(sizes)
        ext_idx = sizes.index(max_size)  # 0=w1, 1=w3, 2=w5
        ext_wave = [1, 3, 5][ext_idx]

        # Only label as extended if dominant wave > 1.618× the smaller of the other two
        other_sizes = [s for j, s in enumerate(sizes) if j != ext_idx]
        min_other = min(other_sizes)
        is_extended = max_size > 1.618 * min_other if min_other > 0 else False

        # Commodity bias: if W5 > W3, favor W5 extension
        if is_commodity and w5.price_change > w3.price_change:
            ext_wave = 5
            is_extended = True

        wave_type = _get_impulse_wave_type(ext_wave if is_extended else None)

        # Truncation check: W5 fails to exceed W3 end, but only valid if W3 > 2.618×W1
        is_truncated = False
        if direction == WaveDirection.UP and w5.end.price < w3.end.price:
            if w3.price_change > 2.618 * w1.price_change:
                is_truncated = True
                wave_type = WaveType.TRUNCATED_5TH
            else:
                continue  # invalid: W5 below W3 without the extension condition
        elif direction == WaveDirection.DOWN and w5.end.price > w3.end.price:
            if w3.price_change > 2.618 * w1.price_change:
                is_truncated = True
                wave_type = WaveType.TRUNCATED_5TH
            else:
                continue

        candidates.append({
            "waves":        seq,
            "start_idx":    i,
            "wave_type":    wave_type,
            "extension_wave": ext_wave if is_extended else None,
            "is_truncated": is_truncated,
            "direction":    direction,
        })

    return candidates


def _get_impulse_wave_type(ext_wave: Optional[int]) -> WaveType:
    return {
        1:    WaveType.IMPULSE_EXT_W1,
        3:    WaveType.IMPULSE_EXT_W3,
        5:    WaveType.IMPULSE_EXT_W5,
        None: WaveType.IMPULSE,
    }[ext_wave]


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
