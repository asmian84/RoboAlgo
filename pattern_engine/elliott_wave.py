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


# ── Impulse Wave Detection ───────────────────────────────────────────────────

def find_impulse_waves(
    pivots: list[Pivot],
    direction: WaveDirection = WaveDirection.UP,
) -> list[WaveCount]:
    """
    Find all valid 5-wave impulse patterns in the pivot sequence.

    Scans for 5-wave structures (alternating direction monowaves) and validates
    them against Elliott's three absolute rules. Returns all valid candidates
    ranked by Fibonacci confidence score.
    """
    if len(pivots) < 6:  # Need at least 6 pivots for 5 waves
        return []

    candidates: list[WaveCount] = []

    # For bullish impulse: start at a low, go high-low-high-low-high (5 moves)
    # For bearish impulse: start at a high, go low-high-low-high-low
    start_is_high = direction == WaveDirection.DOWN
    is_up = direction == WaveDirection.UP

    # Pre-build alternating-direction index lookup for non-greedy search
    n_pivots = len(pivots)

    # Pre-compute global extreme indices for W3 inclusion
    _global_high_idx = max(range(n_pivots), key=lambda k: pivots[k].price if pivots[k].is_high else -float('inf')) if n_pivots > 0 else 0
    _global_low_idx  = min(range(n_pivots), key=lambda k: pivots[k].price if not pivots[k].is_high else float('inf')) if n_pivots > 0 else 0

    def _next_options(after_idx: int, want_high: bool, max_branch: int = 8,
                      include_global_extreme: bool = False) -> list[int]:
        """Return up to max_branch nearest (chronological) pivot indices
        > after_idx with matching direction. Walking nearest-first keeps
        wave 1-4 building local structure rather than jumping to far extremes.

        When include_global_extreme=True, the global highest high (want_high=True)
        or global lowest low (want_high=False) is always included as a candidate
        even if beyond the max_branch window — this ensures extended W3 structures
        (common in crypto) are never missed."""
        opts = []
        for k in range(after_idx + 1, n_pivots):
            if pivots[k].is_high == want_high:
                opts.append(k)
                if len(opts) >= max_branch:
                    break

        if include_global_extreme:
            ext_idx = _global_high_idx if want_high else _global_low_idx
            if ext_idx > after_idx and ext_idx not in opts:
                opts.append(ext_idx)
                opts.sort()  # keep chronological order

        return opts

    def _emit(seq_indices: list[int]):
        full_seq = [pivots[i] for i in seq_indices]
        waves = [MonoWave(start=full_seq[i], end=full_seq[i + 1]) for i in range(5)]
        if _validate_impulse_rules(waves, direction):
            return
        fib_score = _score_impulse_fibonacci(waves, direction)
        confidence = _compute_impulse_confidence(waves, fib_score, direction)
        labels = DEGREE_LABELS["intermediate"]["impulse"]
        candidates.append(WaveCount(
            waves=waves,
            wave_type=WaveType.IMPULSE,
            direction=direction,
            confidence=confidence,
            fib_score=fib_score,
            labels=labels,
            is_complete=True,
        ))

    # ── Non-greedy 5-wave search with branch-and-prune ──────────────────
    # For each starting pivot, explore multiple choices for waves 1-4 (each
    # capped to a small set of candidates), then exhaustively scan wave 5
    # endpoints. This catches extended structures whose true wave 5 lands on
    # the chart's structural extreme (e.g. ATH) instead of the first
    # alternating pivot.
    for start_i in range(n_pivots - 5):
        p0 = pivots[start_i]
        if p0.is_high != start_is_high:
            continue

        # Wave 1 candidates (opposite direction of start)
        for i1 in _next_options(start_i, want_high=is_up, max_branch=6):
            p1 = pivots[i1]
            # Wave 1 must move in the impulse direction
            if is_up and p1.price <= p0.price:
                continue
            if not is_up and p1.price >= p0.price:
                continue

            # Wave 2 candidates
            for i2 in _next_options(i1, want_high=not is_up, max_branch=6):
                p2 = pivots[i2]
                # Wave 2 must not retrace beyond wave 1 start (≤100%)
                if is_up and p2.price <= p0.price:
                    continue
                if not is_up and p2.price >= p0.price:
                    continue

                # Wave 3 candidates — must extend beyond wave 1 end.
                # W3 is typically the longest wave in an impulse so we search
                # broadly (max_branch=20) to catch extended structures like
                # BTC's $108K peak which can be many pivots away from W2.
                # include_global_extreme=True also ensures the absolute ATH/ATL
                # is always considered.
                for i3 in _next_options(i2, want_high=is_up, max_branch=20,
                                        include_global_extreme=True):
                    p3 = pivots[i3]
                    if is_up and p3.price <= p1.price:
                        continue
                    if not is_up and p3.price >= p1.price:
                        continue

                    # Wave 4 candidates — must not enter wave 1 territory
                    for i4 in _next_options(i3, want_high=not is_up, max_branch=6):
                        p4 = pivots[i4]
                        if is_up and p4.price <= p1.price:
                            continue  # overlap with wave 1
                        if not is_up and p4.price >= p1.price:
                            continue

                        # Wave 5 candidates — must extend beyond wave 3 end.
                        # Scan ALL subsequent same-direction pivots (don't cap)
                        # so we surface structures terminating at the chart's
                        # actual high/low.
                        w5_dir = is_up
                        for i5 in range(i4 + 1, n_pivots):
                            if pivots[i5].is_high != w5_dir:
                                continue
                            p5 = pivots[i5]
                            if is_up and p5.price <= p3.price:
                                continue
                            if not is_up and p5.price >= p3.price:
                                continue
                            _emit([start_i, i1, i2, i3, i4, i5])

    # Sort by confidence descending
    candidates.sort(key=lambda c: c.confidence, reverse=True)

    # Deduplicate overlapping counts — keep highest confidence
    return _deduplicate_counts(candidates)


def find_partial_impulse(
    pivots: list[Pivot],
    direction: WaveDirection = WaveDirection.UP,
) -> list[WaveCount]:
    """
    Find partial (in-progress) impulse waves.

    Detects waves 1-2 complete (looking for wave 3 entry),
    waves 1-2-3 complete (in wave 4 pullback),
    waves 1-2-3-4 complete (looking for wave 5).
    """
    if len(pivots) < 3:
        return []

    candidates: list[WaveCount] = []
    start_is_high = direction == WaveDirection.DOWN

    for start_i in range(len(pivots) - 2):
        p0 = pivots[start_i]
        if p0.is_high != start_is_high:
            continue

        # Try to build as many waves as possible
        sequence = [p0]
        j = start_i + 1

        while len(sequence) < 6 and j < len(pivots):
            expected_high = (len(sequence) % 2 == 1) if direction == WaveDirection.UP else (len(sequence) % 2 == 0)
            if pivots[j].is_high == expected_high:
                sequence.append(pivots[j])
            j += 1

        num_waves = len(sequence) - 1
        if num_waves < 2:
            continue
        if num_waves >= 5:
            continue  # This is a complete impulse, handled by find_impulse_waves

        waves = []
        for k in range(num_waves):
            waves.append(MonoWave(start=sequence[k], end=sequence[k + 1]))

        # Validate rules for completed waves so far
        violations = _validate_partial_impulse_rules(waves, direction)
        if violations:
            continue

        fib_score = _score_partial_fibonacci(waves, direction)
        confidence = _compute_partial_confidence(waves, fib_score, num_waves)

        labels = DEGREE_LABELS["intermediate"]["impulse"][:num_waves]

        wc = WaveCount(
            waves=waves,
            wave_type=WaveType.IMPULSE,
            direction=direction,
            confidence=confidence,
            fib_score=fib_score,
            labels=labels,
            is_complete=False,
        )
        candidates.append(wc)

    candidates.sort(key=lambda c: c.confidence, reverse=True)
    return _deduplicate_counts(candidates)


# ── Corrective Wave Detection ────────────────────────────────────────────────

def find_corrective_waves(
    pivots: list[Pivot],
    direction: WaveDirection = WaveDirection.DOWN,
) -> list[WaveCount]:
    """
    Find ABC corrective patterns.

    direction = overall correction direction (DOWN after bullish impulse, UP after bearish).

    Supports:
    - Zigzag (5-3-5): sharp, deep correction
    - Flat (3-3-5): sideways correction
    - Expanded flat: Wave B exceeds Wave A start
    """
    if len(pivots) < 4:  # Need 4 pivots for 3 waves (A-B-C)
        return []

    candidates: list[WaveCount] = []
    start_is_high = direction == WaveDirection.DOWN  # Correction starts at a high after uptrend

    for start_i in range(len(pivots) - 3):
        p0 = pivots[start_i]
        if p0.is_high != start_is_high:
            continue

        sequence = [p0]
        j = start_i + 1

        while len(sequence) < 4 and j < len(pivots):
            expected_high = (len(sequence) % 2 == 0) if direction == WaveDirection.DOWN else (len(sequence) % 2 == 1)
            if pivots[j].is_high == expected_high:
                sequence.append(pivots[j])
            j += 1

        if len(sequence) < 4:
            continue

        waves = [MonoWave(start=sequence[k], end=sequence[k + 1]) for k in range(3)]

        # Validate corrective rules
        if not _validate_corrective(waves, direction):
            continue

        fib_score = _score_corrective_fibonacci(waves)
        confidence = _compute_corrective_confidence(waves, fib_score)

        labels = DEGREE_LABELS["intermediate"]["corrective"]

        wc = WaveCount(
            waves=waves,
            wave_type=WaveType.CORRECTIVE,
            direction=direction,
            confidence=confidence,
            fib_score=fib_score,
            labels=labels,
            is_complete=True,
        )
        candidates.append(wc)

    candidates.sort(key=lambda c: c.confidence, reverse=True)
    return _deduplicate_counts(candidates)


# ── Triangle Wave Detection ────────────────────────────────────────────────

def find_triangle_waves(
    pivots: list[Pivot],
    direction: WaveDirection = WaveDirection.DOWN,
) -> list[WaveCount]:
    """
    Find corrective triangle patterns (a-b-c-d-e).

    Triangles have 5 waves with converging or diverging trendlines.
    Types detected:
      - Contracting (symmetrical): highs declining, lows rising
      - Ascending: highs flat, lows rising
      - Descending: highs declining, lows flat
      - Expanding: highs rising, lows declining
    """
    if len(pivots) < 6:  # Need 6 pivots for 5 waves (a,b,c,d,e)
        return []

    candidates: list[WaveCount] = []

    # For bearish triangle: starts at a high (a down, b up, c down, d up, e down)
    # For bullish triangle: starts at a low (a up, b down, c up, d down, e up)
    start_is_high = direction == WaveDirection.DOWN

    for start_i in range(len(pivots) - 5):
        p0 = pivots[start_i]
        if p0.is_high != start_is_high:
            continue

        # Collect 6 alternating pivots
        sequence = [p0]
        j = start_i + 1

        while len(sequence) < 6 and j < len(pivots):
            expected_high = (len(sequence) % 2 == 0) if direction == WaveDirection.DOWN else (len(sequence) % 2 == 1)
            if pivots[j].is_high == expected_high:
                sequence.append(pivots[j])
            j += 1

        if len(sequence) < 6:
            continue

        # Build 5 monowaves
        waves = [MonoWave(start=sequence[k], end=sequence[k + 1]) for k in range(5)]

        # Validate triangle structure
        if not _validate_triangle(waves, direction):
            continue

        # Score using Fibonacci relationships
        fib_score = _score_triangle_fibonacci(waves)
        confidence = min(90.0, fib_score * 0.5 + 25.0)

        labels = DEGREE_LABELS["intermediate"]["triangle"]

        wc = WaveCount(
            waves=waves,
            wave_type=WaveType.TRIANGLE,
            direction=direction,
            confidence=confidence,
            fib_score=fib_score,
            labels=labels,
            is_complete=True,
        )
        candidates.append(wc)

    candidates.sort(key=lambda c: c.confidence, reverse=True)
    return _deduplicate_counts(candidates)


def _validate_triangle(waves: list[MonoWave], direction: WaveDirection) -> bool:
    """
    Validate triangle structure: 5 waves alternating direction, with
    converging or diverging trendlines. Each successive wave should
    generally stay within the boundaries of wave A.
    """
    if len(waves) != 5:
        return False

    wa, wb, wc, wd, we = waves

    # Waves must alternate direction
    if direction == WaveDirection.DOWN:
        expected_dirs = [WaveDirection.DOWN, WaveDirection.UP, WaveDirection.DOWN, WaveDirection.UP, WaveDirection.DOWN]
    else:
        expected_dirs = [WaveDirection.UP, WaveDirection.DOWN, WaveDirection.UP, WaveDirection.DOWN, WaveDirection.UP]

    for w, exp_dir in zip(waves, expected_dirs):
        if w.direction != exp_dir:
            return False

    # Waves should stay within the boundaries of wave A (with some tolerance)
    a_high = max(wa.start.price, wa.end.price)
    a_low = min(wa.start.price, wa.end.price)
    a_range = a_high - a_low
    tolerance = a_range * 0.15  # 15% tolerance

    for w in waves[1:]:
        if w.end.price > a_high + tolerance or w.end.price < a_low - tolerance:
            return False
        if w.start.price > a_high + tolerance or w.start.price < a_low - tolerance:
            return False

    # Check for converging or expanding pattern
    price_changes = [w.price_change for w in waves]

    # Contracting: successive waves should generally decrease
    decreasing_count = sum(1 for i in range(1, 5) if price_changes[i] < price_changes[i - 1])
    # Expanding: successive waves should generally increase
    increasing_count = sum(1 for i in range(1, 5) if price_changes[i] > price_changes[i - 1])

    # Accept if mostly contracting (3+ out of 4) or mostly expanding (3+ out of 4)
    if decreasing_count < 3 and increasing_count < 3:
        return False

    return True


def _score_triangle_fibonacci(waves: list[MonoWave]) -> float:
    """Score triangle Fibonacci relationships."""
    scores = []
    wa, wb, wc, wd, we = waves

    # Wave B typically retraces 50-78.6% of A
    b_retrace = wb.price_change / max(wa.price_change, 1e-10)
    d = _nearest_fib_distance(b_retrace, [0.500, 0.618, 0.786])
    scores.append(max(0, 100 - d * 200))

    # Wave C typically retraces 61.8-78.6% of B
    c_retrace = wc.price_change / max(wb.price_change, 1e-10)
    d = _nearest_fib_distance(c_retrace, [0.618, 0.786])
    scores.append(max(0, 100 - d * 200))

    # Wave D typically retraces 61.8-78.6% of C
    d_retrace = wd.price_change / max(wc.price_change, 1e-10)
    d2 = _nearest_fib_distance(d_retrace, [0.618, 0.786])
    scores.append(max(0, 100 - d2 * 200))

    # Wave E typically retraces 61.8-78.6% of D
    e_retrace = we.price_change / max(wd.price_change, 1e-10)
    d3 = _nearest_fib_distance(e_retrace, [0.618, 0.786])
    scores.append(max(0, 100 - d3 * 200))

    return float(np.mean(scores)) if scores else 0.0


# ── Ending Diagonal Detection ──────────────────────────────────────────────

def find_ending_diagonal(
    pivots: list[Pivot],
    direction: WaveDirection = WaveDirection.UP,
) -> list[WaveCount]:
    """
    Find ending diagonal (wedge) patterns -- 5-wave structure where
    wave 4 IS allowed to overlap wave 1 (unlike regular impulse).
    Trendlines converge. Common in wave 5 or wave C positions.
    """
    if len(pivots) < 6:
        return []

    candidates: list[WaveCount] = []
    start_is_high = direction == WaveDirection.DOWN

    for start_i in range(len(pivots) - 5):
        p0 = pivots[start_i]
        if p0.is_high != start_is_high:
            continue

        sequence = [p0]
        j = start_i + 1

        while len(sequence) < 6 and j < len(pivots):
            expected_high = (len(sequence) % 2 == 1) if direction == WaveDirection.UP else (len(sequence) % 2 == 0)
            if pivots[j].is_high == expected_high:
                sequence.append(pivots[j])
            j += 1

        if len(sequence) < 6:
            continue

        waves = [MonoWave(start=sequence[k], end=sequence[k + 1]) for k in range(5)]

        # Validate diagonal rules
        if not _validate_diagonal(waves, direction):
            continue

        fib_score = _score_diagonal_fibonacci(waves)
        confidence = min(85.0, fib_score * 0.45 + 20.0)

        labels = DEGREE_LABELS["intermediate"]["diagonal"]

        wc = WaveCount(
            waves=waves,
            wave_type=WaveType.DIAGONAL,
            direction=direction,
            confidence=confidence,
            fib_score=fib_score,
            labels=labels,
            is_complete=True,
        )
        candidates.append(wc)

    candidates.sort(key=lambda c: c.confidence, reverse=True)
    return _deduplicate_counts(candidates)


def _validate_diagonal(waves: list[MonoWave], direction: WaveDirection) -> bool:
    """
    Validate ending diagonal structure:
    - 5 waves in correct directional alternation
    - Wave 4 CAN overlap wave 1 (key difference from impulse)
    - Trendlines through 1-3-5 and 2-4 should converge (wedge shape)
    - Each successive motive wave should be shorter: 1 > 3 > 5
    - Each successive corrective wave should be shorter: 2 > 4
    """
    if len(waves) != 5:
        return False

    w1, w2, w3, w4, w5 = waves

    # Direction check: motive waves in direction, corrective waves opposite
    if direction == WaveDirection.UP:
        if w1.direction != WaveDirection.UP or w3.direction != WaveDirection.UP or w5.direction != WaveDirection.UP:
            return False
        if w2.direction != WaveDirection.DOWN or w4.direction != WaveDirection.DOWN:
            return False
    else:
        if w1.direction != WaveDirection.DOWN or w3.direction != WaveDirection.DOWN or w5.direction != WaveDirection.DOWN:
            return False
        if w2.direction != WaveDirection.UP or w4.direction != WaveDirection.UP:
            return False

    # Rule 2 still applies: Wave 3 must not be the shortest of 1, 3, 5
    if w3.price_change < w1.price_change and w3.price_change < w5.price_change:
        return False

    # Diagonal characteristic: successive motive waves should shorten (1 > 3 > 5)
    if not (w1.price_change > w3.price_change and w3.price_change > w5.price_change):
        # Allow some tolerance: at least 1 > 5 and general trend of shortening
        if not (w1.price_change > w5.price_change):
            return False

    # Diagonal characteristic: successive corrective waves should shorten (2 > 4)
    # Allow tolerance — at least don't require 4 much larger than 2
    if w4.price_change > w2.price_change * 1.3:
        return False

    # Convergence check: the range of successive waves should narrow
    # Wave 1 range should be larger than wave 5 range
    if w5.price_change > w1.price_change * 1.1:
        return False

    return True


def _score_diagonal_fibonacci(waves: list[MonoWave]) -> float:
    """Score diagonal Fibonacci relationships."""
    w1, w2, w3, w4, w5 = waves
    scores = []

    # Wave 2 retracement of Wave 1 (ideal: 61.8-78.6%)
    w2_retrace = w2.price_change / max(w1.price_change, 1e-10)
    d = _nearest_fib_distance(w2_retrace, [0.618, 0.786])
    scores.append(max(0, 100 - d * 200))

    # Wave 3 relative to Wave 1 (ideal: 61.8-78.6% of Wave 1)
    w3_ratio = w3.price_change / max(w1.price_change, 1e-10)
    d = _nearest_fib_distance(w3_ratio, [0.618, 0.786, 1.000])
    scores.append(max(0, 100 - d * 150))

    # Wave 4 retracement of Wave 3 (ideal: 61.8-78.6%)
    w4_retrace = w4.price_change / max(w3.price_change, 1e-10)
    d = _nearest_fib_distance(w4_retrace, [0.618, 0.786])
    scores.append(max(0, 100 - d * 200))

    # Wave 5 relative to Wave 1 (ideal: 38.2-61.8% of Wave 1)
    w5_ratio = w5.price_change / max(w1.price_change, 1e-10)
    d = _nearest_fib_distance(w5_ratio, [0.382, 0.500, 0.618])
    scores.append(max(0, 100 - d * 200))

    return float(np.mean(scores)) if scores else 0.0


# ── Double Correction (WXY) Detection ──────────────────────────────────────

def find_double_correction(
    pivots: list[Pivot],
    direction: WaveDirection = WaveDirection.DOWN,
) -> list[WaveCount]:
    """
    Find WXY double correction patterns.
    W = ABC, X = connector (counter-trend move), Y = ABC
    Total: 7 waves (A-B-C of W, then X, then A-B-C of Y)
    """
    if len(pivots) < 8:  # Need 8 pivots for 7 waves
        return []

    candidates: list[WaveCount] = []
    start_is_high = direction == WaveDirection.DOWN

    for start_i in range(len(pivots) - 7):
        p0 = pivots[start_i]
        if p0.is_high != start_is_high:
            continue

        # Collect 8 alternating pivots for 7 waves
        sequence = [p0]
        j = start_i + 1

        while len(sequence) < 8 and j < len(pivots):
            expected_high = (len(sequence) % 2 == 0) if direction == WaveDirection.DOWN else (len(sequence) % 2 == 1)
            if pivots[j].is_high == expected_high:
                sequence.append(pivots[j])
            j += 1

        if len(sequence) < 8:
            continue

        # Build 7 monowaves
        all_waves = [MonoWave(start=sequence[k], end=sequence[k + 1]) for k in range(7)]

        # Split into W (waves 0-2), X (wave 3), Y (waves 4-6)
        w_waves = all_waves[0:3]  # ABC of W
        x_wave = all_waves[3]      # X connector
        y_waves = all_waves[4:7]  # ABC of Y

        # Validate W as ABC
        if not _validate_corrective(w_waves, direction):
            continue

        # Validate X is counter-trend
        if direction == WaveDirection.DOWN:
            if x_wave.direction != WaveDirection.UP:
                continue
        else:
            if x_wave.direction != WaveDirection.DOWN:
                continue

        # Validate Y as ABC
        if not _validate_corrective(y_waves, direction):
            continue

        # X wave should retrace 38.2-78.6% of W
        w_total_change = abs(w_waves[-1].end.price - w_waves[0].start.price)
        x_retrace_ratio = x_wave.price_change / max(w_total_change, 1e-10)
        if x_retrace_ratio < 0.2 or x_retrace_ratio > 1.0:
            continue

        # Score
        fib_score = _score_double_correction_fibonacci(w_waves, x_wave, y_waves, w_total_change)
        confidence = min(85.0, fib_score * 0.45 + 20.0)

        labels = DEGREE_LABELS["intermediate"]["double_correction"]

        # For the WaveCount, store all 7 waves but label as W, X, Y
        wc = WaveCount(
            waves=all_waves,
            wave_type=WaveType.DOUBLE_CORRECTION,
            direction=direction,
            confidence=confidence,
            fib_score=fib_score,
            labels=labels,
            is_complete=True,
        )
        candidates.append(wc)

    candidates.sort(key=lambda c: c.confidence, reverse=True)
    return _deduplicate_counts(candidates)


def _score_double_correction_fibonacci(
    w_waves: list[MonoWave],
    x_wave: MonoWave,
    y_waves: list[MonoWave],
    w_total_change: float,
) -> float:
    """Score WXY double correction Fibonacci relationships."""
    scores = []

    # Score W sub-structure (ABC)
    w_fib = _score_corrective_fibonacci(w_waves)
    scores.append(w_fib)

    # X wave retracement of W (ideal: 38.2%, 50%, 61.8%, 78.6%)
    x_retrace = x_wave.price_change / max(w_total_change, 1e-10)
    d = _nearest_fib_distance(x_retrace, [0.382, 0.500, 0.618, 0.786])
    scores.append(max(0, 100 - d * 200))

    # Score Y sub-structure (ABC)
    y_fib = _score_corrective_fibonacci(y_waves)
    scores.append(y_fib)

    # Y relative to W (ideal: 61.8%, 100%, 161.8%)
    y_total_change = abs(y_waves[-1].end.price - y_waves[0].start.price)
    y_ratio = y_total_change / max(w_total_change, 1e-10)
    d = _nearest_fib_distance(y_ratio, [0.618, 1.000, 1.618])
    scores.append(max(0, 100 - d * 150))

    return float(np.mean(scores)) if scores else 0.0


# ── Rule Validation ──────────────────────────────────────────────────────────

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


def _validate_partial_impulse_rules(waves: list[MonoWave], direction: WaveDirection) -> list[str]:
    """Validate rules for partial impulse (2-4 waves)."""
    violations = []

    if len(waves) >= 2:
        w1, w2 = waves[0], waves[1]
        # Rule 1
        if direction == WaveDirection.UP:
            if w2.end.price <= w1.start.price:
                violations.append("Rule 1: Wave 2 retraces > 100% of Wave 1")
        else:
            if w2.end.price >= w1.start.price:
                violations.append("Rule 1: Wave 2 retraces > 100% of Wave 1")

    if len(waves) >= 4:
        w1, w3, w4 = waves[0], waves[2], waves[3]
        # Rule 3
        if direction == WaveDirection.UP:
            if w4.end.price < w1.end.price:
                violations.append("Rule 3: Wave 4 overlaps Wave 1 territory")
        else:
            if w4.end.price > w1.end.price:
                violations.append("Rule 3: Wave 4 overlaps Wave 1 territory")

    return violations


def _validate_corrective(waves: list[MonoWave], direction: WaveDirection) -> bool:
    """Validate basic ABC corrective structure."""
    if len(waves) != 3:
        return False

    wa, wb, wc = waves

    # Wave A and C should be in the correction direction
    # Wave B should be counter-trend
    if direction == WaveDirection.DOWN:
        if wa.direction != WaveDirection.DOWN or wc.direction != WaveDirection.DOWN:
            return False
        if wb.direction != WaveDirection.UP:
            return False
        # Wave B should not exceed Wave A start (unless expanded flat)
        # We allow it for flexibility but penalize in scoring
    else:
        if wa.direction != WaveDirection.UP or wc.direction != WaveDirection.UP:
            return False
        if wb.direction != WaveDirection.DOWN:
            return False

    return True


# ── Fibonacci Scoring ────────────────────────────────────────────────────────

def _nearest_fib_distance(ratio: float, target_fibs: list[float]) -> float:
    """How close a ratio is to the nearest target Fibonacci level. Lower = better."""
    return min(abs(ratio - f) for f in target_fibs)


def _score_impulse_fibonacci(waves: list[MonoWave], direction: WaveDirection) -> float:
    """
    Score how well the impulse matches ideal Fibonacci relationships.
    Returns 0-100 score.
    """
    w1, w2, w3, w4, w5 = waves
    scores = []

    # Wave 2 retracement of Wave 1 (ideal: 50%, 61.8%, 78.6%)
    w2_retrace = w2.price_change / max(w1.price_change, 1e-10)
    d = _nearest_fib_distance(w2_retrace, [0.382, 0.500, 0.618, 0.786])
    scores.append(max(0, 100 - d * 200))

    # Wave 3 extension of Wave 1 (ideal: 161.8%, 200%, 261.8%, 300%, 361.8%)
    # 3.0× and 3.618× are common in crypto/volatile assets with extended W3
    w3_ext = w3.price_change / max(w1.price_change, 1e-10)
    d = _nearest_fib_distance(w3_ext, [1.000, 1.618, 2.000, 2.618, 3.000, 3.618])
    scores.append(max(0, 100 - d * 100))

    # Wave 4 retracement of Wave 3 (ideal: 23.6%, 38.2%, 50%, 61.8%)
    w4_retrace = w4.price_change / max(w3.price_change, 1e-10)
    d = _nearest_fib_distance(w4_retrace, [0.236, 0.382, 0.500, 0.618])
    scores.append(max(0, 100 - d * 200))

    # Wave 5 extension from Wave 4 end relative to Wave 1
    # (ideal: 61.8%, 100%, 161.8%; crypto/volatile: 2.0, 2.618, 3.0, 3.618)
    w5_ext = w5.price_change / max(w1.price_change, 1e-10)
    d = _nearest_fib_distance(w5_ext, [0.618, 1.000, 1.618, 2.000, 2.618, 3.000, 3.618])
    scores.append(max(0, 100 - d * 150))

    # Wave 3 should be longest (guideline, not rule — bonus)
    if w3.price_change >= w1.price_change and w3.price_change >= w5.price_change:
        scores.append(100)
    else:
        scores.append(40)

    # Alternation guideline: Wave 2 and Wave 4 should differ in character
    # (one deep/sharp, other shallow/sideways)
    w2_depth = w2.price_change / max(w1.price_change, 1e-10)
    w4_depth = w4.price_change / max(w3.price_change, 1e-10)
    alternation = abs(w2_depth - w4_depth)
    scores.append(min(100, alternation * 300))

    # Weighted average (Wave 3 extension matters most)
    weights = [1.0, 1.5, 1.0, 0.8, 1.2, 0.5]
    return float(np.average(scores, weights=weights))


def _score_partial_fibonacci(waves: list[MonoWave], direction: WaveDirection) -> float:
    """Score partial impulse Fibonacci relationships."""
    scores = []

    if len(waves) >= 2:
        w1, w2 = waves[0], waves[1]
        w2_retrace = w2.price_change / max(w1.price_change, 1e-10)
        d = _nearest_fib_distance(w2_retrace, [0.382, 0.500, 0.618, 0.786])
        scores.append(max(0, 100 - d * 200))

    if len(waves) >= 3:
        w1, w3 = waves[0], waves[2]
        w3_ext = w3.price_change / max(w1.price_change, 1e-10)
        d = _nearest_fib_distance(w3_ext, [1.000, 1.618, 2.000, 2.618, 3.000, 3.618])
        scores.append(max(0, 100 - d * 100))

    if len(waves) >= 4:
        w3, w4 = waves[2], waves[3]
        w4_retrace = w4.price_change / max(w3.price_change, 1e-10)
        d = _nearest_fib_distance(w4_retrace, [0.236, 0.382, 0.500])
        scores.append(max(0, 100 - d * 200))

    return float(np.mean(scores)) if scores else 0.0


def _score_corrective_fibonacci(waves: list[MonoWave]) -> float:
    """Score ABC corrective Fibonacci relationships."""
    wa, wb, wc = waves
    scores = []

    # Wave B retracement of Wave A (ideal: 38.2%, 50%, 61.8%, 78.6%)
    b_retrace = wb.price_change / max(wa.price_change, 1e-10)
    d = _nearest_fib_distance(b_retrace, [0.382, 0.500, 0.618, 0.786])
    scores.append(max(0, 100 - d * 200))

    # Wave C relative to Wave A (ideal: 61.8%, 100%, 161.8%)
    c_ratio = wc.price_change / max(wa.price_change, 1e-10)
    d = _nearest_fib_distance(c_ratio, [0.618, 1.000, 1.618])
    scores.append(max(0, 100 - d * 150))

    return float(np.mean(scores)) if scores else 0.0


# ── Confidence Computation ───────────────────────────────────────────────────

def _compute_impulse_confidence(waves: list[MonoWave], fib_score: float, direction: WaveDirection) -> float:
    """
    Compute overall confidence for a complete impulse wave count.

    Factors:
    1. Fibonacci score (40%)
    2. Proportionality — wave durations should be proportional (15%)
    3. Wave 3 strength — should be clearly the strongest (20%)
    4. Channeling — waves 1-3-5 endpoints should roughly channel (15%)
    5. Volume/momentum alignment (10%) — not available here, placeholder
    """
    scores = []

    # 1. Fibonacci score
    scores.append(fib_score * 0.40)

    # 2. Proportionality — Wave 3 typically longest in time too
    w1, w2, w3, w4, w5 = waves
    durations = [w.bar_count for w in waves]
    if durations[2] >= max(durations[0], durations[4]):
        scores.append(15.0)  # Wave 3 longest in time
    elif durations[2] >= min(durations[0], durations[4]):
        scores.append(8.0)
    else:
        scores.append(3.0)

    # 3. Wave 3 strength
    changes = [w.price_change for w in waves]
    impulse_changes = [changes[0], changes[2], changes[4]]
    if changes[2] == max(impulse_changes):
        scores.append(20.0)
    elif changes[2] > np.median(impulse_changes):
        scores.append(12.0)
    else:
        scores.append(5.0)

    # 4. Channeling — draw line from end of W1 to end of W5, see if W3 end is near
    if direction == WaveDirection.UP:
        # Channel line from W1 end to W5 end
        w1_end = w1.end.price
        w5_end = w5.end.price
        w3_end = w3.end.price
        w1_idx = w1.end.idx
        w5_idx = w5.end.idx
        w3_idx = w3.end.idx
        if w5_idx != w1_idx:
            expected_w3 = w1_end + (w5_end - w1_end) * (w3_idx - w1_idx) / (w5_idx - w1_idx)
            channel_err = abs(w3_end - expected_w3) / max(abs(w5_end - w1_end), 1e-10)
            scores.append(max(0, 15.0 - channel_err * 50))
        else:
            scores.append(7.0)
    else:
        scores.append(10.0)

    # 5. Placeholder for volume confirmation
    scores.append(7.0)

    return min(100.0, sum(scores))


def _compute_partial_confidence(waves: list[MonoWave], fib_score: float, num_waves: int) -> float:
    """Confidence for partial impulse — lower base, scales with completeness."""
    base = fib_score * 0.5
    completeness_bonus = num_waves * 8.0  # 2 waves = +16, 3 = +24, 4 = +32
    return min(85.0, base + completeness_bonus)


def _compute_corrective_confidence(waves: list[MonoWave], fib_score: float) -> float:
    """Confidence for corrective pattern."""
    return min(90.0, fib_score * 0.6 + 20.0)


# ── Deduplication ────────────────────────────────────────────────────────────

def _deduplicate_counts(candidates: list[WaveCount]) -> list[WaveCount]:
    """Remove truly duplicate wave counts, keeping highest confidence.

    Two counts are duplicates only when they share 3+ pivot points (i.e. they
    are essentially the same structural reading with minor variations). Time-
    based overlap alone is NOT sufficient — structurally different counts in
    the same time range represent different valid interpretations and must
    both survive for downstream scoring to pick the right one.

    Separates UP and DOWN directions so they never block each other.
    """
    if not candidates:
        return []

    result: list[WaveCount] = []
    # Track pivot sets per direction
    used_pivot_sets: dict[str, list[set[tuple[int, float]]]] = {}

    for wc in candidates:
        dir_key = wc.direction.value if hasattr(wc.direction, 'value') else str(wc.direction)

        # Build this count's structural signature: W3, W4, and W5 endpoints
        # define the key structure. Counts with the same W3/W4/W5 are truly
        # the same reading. Counts sharing W3/W5 but with different W4 are
        # structurally different (different correction depth) and must survive.
        w3_end = (wc.waves[2].end.idx, round(wc.waves[2].end.price, 0)) if len(wc.waves) > 2 else None
        w4_end = (wc.waves[3].end.idx, round(wc.waves[3].end.price, 0)) if len(wc.waves) > 3 else None
        w5_end = (wc.waves[-1].end.idx, round(wc.waves[-1].end.price, 0)) if wc.waves else None
        this_sig = (w3_end, w4_end, w5_end)

        is_dup = False
        for existing_sig in used_pivot_sets.get(dir_key, []):
            if this_sig == existing_sig:
                is_dup = True
                break

        if not is_dup:
            result.append(wc)
            used_pivot_sets.setdefault(dir_key, []).append(this_sig)

    return result


# ── Multi-Degree Analysis ────────────────────────────────────────────────────

def analyze_multi_degree(
    df: pd.DataFrame,
    primary_atr_mult: float = 1.5,
    minor_atr_mult: float = 0.3,
    macro_mode: bool = False,
) -> dict[str, Any]:
    """
    Run Elliott Wave analysis at multiple degrees (timeframe-like zoom levels).

    Returns the best wave counts at each degree:
    - Primary (major swings, fewer pivots)
    - Intermediate (default)
    - Minor (more granular swings)

    macro_mode: When True (weekly/monthly), uses wider lookback windows
    to find grand supercycle waves spanning the full chart.
    """
    high = df["high"].astype(float).values
    low = df["low"].astype(float).values
    close = df["close"].astype(float).values

    results: dict[str, Any] = {
        "primary": {"impulse": [], "corrective": [], "partial": []},
        "intermediate": {"impulse": [], "corrective": [], "partial": []},
        "minor": {"impulse": [], "corrective": [], "partial": []},
    }

    # Intermediate ATR mult is midpoint between primary and minor, clamped
    inter_atr_mult = round(max(0.4, min(0.8, (primary_atr_mult + minor_atr_mult) / 2 + 0.15)), 2)

    if macro_mode:
        # ── MACRO: Use percentage-based pivots for the primary degree ──
        # Run ALL swing thresholds and union their pivots — coarse thresholds
        # find grand cycle structure, finer ones surface key intermediate
        # turning points (and ensure the chart's global extreme always lands
        # in some pivot list). Wave detection on the merged list catches both
        # grand cycle counts and counts that terminate at the actual ATH/ATL.
        merged: dict[tuple[int, bool], "Pivot"] = {}
        for swing_pct in [0.35, 0.25, 0.20, 0.15]:
            for p in detect_macro_pivots(df, min_swing_pct=swing_pct):
                key = (p.idx, p.is_high)
                if key not in merged:
                    merged[key] = p
        # Force-include the global high and global low — these are the
        # anchor points the user can SEE on the chart and the structure
        # MUST be able to terminate there.
        n_bars = len(df)
        if n_bars > 0:
            high_arr = df["high"].astype(float).values
            low_arr = df["low"].astype(float).values
            gh_idx = int(high_arr.argmax())
            gl_idx = int(low_arr.argmin())
            merged.setdefault((gh_idx, True), Pivot(idx=gh_idx, price=float(high_arr[gh_idx]), is_high=True, strength=10))
            merged.setdefault((gl_idx, False), Pivot(idx=gl_idx, price=float(low_arr[gl_idx]), is_high=False, strength=10))
        macro_pivots = sorted(merged.values(), key=lambda p: (p.idx, 0 if p.is_high else 1))

        if len(macro_pivots) >= 6:
            for direction in [WaveDirection.UP, WaveDirection.DOWN]:
                impulse = find_impulse_waves(macro_pivots, direction)
                corrective_dir = WaveDirection.DOWN if direction == WaveDirection.UP else WaveDirection.UP
                corrective = find_corrective_waves(macro_pivots, corrective_dir)
                partial = find_partial_impulse(macro_pivots, direction)
                results["primary"]["impulse"].extend(impulse)
                results["primary"]["corrective"].extend(corrective)
                results["primary"]["partial"].extend(partial)

            # Keep many macro candidates so anchor scoring downstream can
            # surface the structurally meaningful one (which may not have
            # the highest raw confidence — e.g. extended wave counts in
            # crypto often have unusual Fibonacci ratios but are correct).
            for key in ["impulse", "corrective", "partial"]:
                results["primary"][key].sort(key=lambda c: c.confidence, reverse=True)
                results["primary"][key] = results["primary"][key][:50]

        # ATR-based detection for intermediate and minor degrees
        degree_configs = [
            ("intermediate", inter_atr_mult, 6, 18),
            ("minor", minor_atr_mult, 3, 10),
        ]
    else:
        degree_configs = [
            ("primary", primary_atr_mult, 5, 15),
            ("intermediate", inter_atr_mult, 3, 10),
            ("minor", minor_atr_mult, 2, 6),
        ]

    for degree, atr_mult, min_lb, max_lb in degree_configs:
        pivots = detect_pivots(df, atr_multiplier=atr_mult, min_lookback=min_lb, max_lookback=max_lb)

        if len(pivots) < 4:
            continue

        # Try both directions
        for direction in [WaveDirection.UP, WaveDirection.DOWN]:
            impulse = find_impulse_waves(pivots, direction)
            corrective_dir = WaveDirection.DOWN if direction == WaveDirection.UP else WaveDirection.UP
            corrective = find_corrective_waves(pivots, corrective_dir)
            partial = find_partial_impulse(pivots, direction)

            triangles = find_triangle_waves(pivots, corrective_dir)
            diagonals = find_ending_diagonal(pivots, direction)
            double_corr = find_double_correction(pivots, corrective_dir)

            results[degree]["impulse"].extend(impulse)
            results[degree]["corrective"].extend(corrective)
            results[degree]["corrective"].extend(triangles)
            results[degree]["corrective"].extend(diagonals)
            results[degree]["corrective"].extend(double_corr)
            results[degree]["partial"].extend(partial)

        # Sort each by confidence
        for key in ["impulse", "corrective", "partial"]:
            results[degree][key].sort(key=lambda c: c.confidence, reverse=True)
            results[degree][key] = results[degree][key][:3]  # Keep top 3 per type

    return results


# ── Public API ───────────────────────────────────────────────────────────────

def detect_elliott_waves(
    symbol: str,
    df: pd.DataFrame,
    resolution_minutes: int = 0,
) -> dict[str, Any]:
    """
    Main entry point — run full Elliott Wave analysis on a DataFrame.

    Args:
        resolution_minutes: 0 = daily, 1/5/15/60/240 = intraday.
            Used to adapt ATR multipliers for better pivot detection
            at each timeframe.

    Returns a structured dict with wave counts at multiple degrees,
    current wave position assessment, and trade signals.
    """
    if df is None or len(df) < 50:
        return {"symbol": symbol, "error": "Insufficient data (need 50+ bars)", "waves": []}

    # Adapt ATR multipliers to bar count and timeframe.
    # Daily bars with 500+ bars have large ATR values — use lower multipliers
    # so pivots aren't over-filtered and 5-wave structures are found.
    n = len(df)
    if resolution_minutes == 0:
        # Daily timeframe — scale down for large bar counts
        if n >= 500:
            primary_mult = 1.0
            minor_mult   = 0.20
        elif n >= 250:
            primary_mult = 1.2
            minor_mult   = 0.25
        else:
            primary_mult = 1.5
            minor_mult   = 0.30
    elif resolution_minutes <= 15:
        # Fast intraday (1m, 5m, 15m) — need tighter thresholds
        primary_mult = 1.0
        minor_mult   = 0.20
    elif resolution_minutes <= 60:
        # 1H — moderate
        primary_mult = 1.2
        minor_mult   = 0.25
    elif resolution_minutes <= 240:
        # 4H — standard
        primary_mult = 1.3
        minor_mult   = 0.25
    elif resolution_minutes >= 10080:
        # Weekly / Monthly — macro-scale: need very wide lookback to find
        # grand supercycle waves spanning years, not just months
        if n >= 400:
            primary_mult = 2.5   # only massive swings
            minor_mult   = 0.8
        elif n >= 200:
            primary_mult = 2.0
            minor_mult   = 0.5
        else:
            primary_mult = 1.5
            minor_mult   = 0.30
    else:
        # Fallback
        primary_mult = 1.3
        minor_mult   = 0.25

    # Run multi-degree analysis on full dataset
    macro = resolution_minutes >= 10080
    analysis = analyze_multi_degree(df, primary_atr_mult=primary_mult, minor_atr_mult=minor_mult, macro_mode=macro)

    # For daily data with many bars, also run a focused analysis on the recent
    # window (last 250 bars ≈ 1 year). This picks up current wave structure
    # that gets diluted in the full 3-year dataset.
    if resolution_minutes == 0 and n > 300:
        recent_df = df.tail(250).reset_index(drop=True)
        recent_analysis = analyze_multi_degree(
            recent_df,
            primary_atr_mult=max(primary_mult - 0.2, 0.6),
            minor_atr_mult=max(minor_mult - 0.05, 0.15),
        )
        # Merge recent results: re-index pivots to full DataFrame coordinates
        # and deduplicate against existing counts
        offset = n - 250

        def _wave_signature(wc_obj):
            """Unique key for a wave count based on start/end bar indices."""
            if not wc_obj.waves:
                return None
            return (wc_obj.waves[0].start.idx, wc_obj.waves[-1].end.idx,
                    len(wc_obj.waves), wc_obj.wave_type, wc_obj.direction)

        for degree in ["primary", "intermediate", "minor"]:
            for key in ["impulse", "corrective", "partial"]:
                # Build set of existing signatures for dedup
                existing_sigs = {_wave_signature(wc) for wc in analysis[degree][key]}
                for wc in recent_analysis[degree][key]:
                    # Shift all pivot indices to align with the full DataFrame
                    for w in wc.waves:
                        w.start = Pivot(
                            idx=w.start.idx + offset,
                            price=w.start.price,
                            is_high=w.start.is_high,
                            strength=w.start.strength,
                        )
                        w.end = Pivot(
                            idx=w.end.idx + offset,
                            price=w.end.price,
                            is_high=w.end.is_high,
                            strength=w.end.strength,
                        )
                    sig = _wave_signature(wc)
                    if sig not in existing_sigs:
                        analysis[degree][key].append(wc)
                        existing_sigs.add(sig)
            # Re-sort and keep top results after merge — keep MORE for the
            # primary degree so macro anchor scoring downstream can find the
            # structurally meaningful count (highest raw confidence isn't
            # always the right answer for grand cycle counts).
            for key in ["impulse", "corrective", "partial"]:
                analysis[degree][key].sort(key=lambda c: c.confidence, reverse=True)
                cap = 25 if degree == "primary" else 5
                analysis[degree][key] = analysis[degree][key][:cap]

    # Determine current position (what wave are we likely in?)
    current_position = _assess_current_position(df, analysis)

    # Build output waves list (all degrees combined, sorted by relevance)
    all_waves = _build_wave_list(df, analysis)

    # Get the best overall count
    best_count = all_waves[0] if all_waves else None

    return {
        "symbol": symbol,
        "bar_count": len(df),
        "waves": all_waves,
        "current_position": current_position,
        "best_count": _serialize_wave_count(best_count) if best_count else None,
    }


def _assess_current_position(df: pd.DataFrame, analysis: dict) -> dict[str, Any]:
    """
    Determine what wave we're most likely in right now.

    Returns a rich context dict with:
      - phase: "impulse" or "correction" (the macro structure)
      - wave_number: which wave within that phase (1-5 or A/B/C)
      - bias: "bullish" or "bearish" (trade direction)
      - strength: "strong" / "moderate" / "cautious" / "weak"
      - action: actionable trade signal text
      - next_move: what to expect next in the sequence
      - pattern: "bullish_impulse", "bearish_impulse", "bullish_correction", etc.
    """
    last_idx = len(df) - 1
    # Scale proximity thresholds by bar count — with 780 daily bars,
    # a wave ending 60 bars ago is still "recent" (~3 months)
    partial_proximity = max(5, int(last_idx * 0.05))   # 5% of bars
    complete_proximity = max(10, int(last_idx * 0.10))  # 10% of bars

    # Check intermediate degree first (most tradeable)
    for degree in ["intermediate", "primary", "minor"]:
        partials = analysis[degree]["partial"]
        impulses = analysis[degree]["impulse"]
        correctives = analysis[degree]["corrective"]

        # ── Partial impulse reaching near current bar ────────────────
        for wc in partials:
            last_wave_end = wc.waves[-1].end.idx
            if last_wave_end >= last_idx - partial_proximity:
                num_waves = len(wc.waves)
                is_up = wc.direction == WaveDirection.UP
                trend_word = "BULLISH" if is_up else "BEARISH"
                labels = DEGREE_LABELS[degree]["impulse"]
                next_label = labels[num_waves] if num_waves < 5 else "Complete"

                if num_waves == 2:
                    # Completed waves 1+2, currently in Wave 3
                    action = "WATCH — Wave 3 forming (strongest wave)"
                    strength = "strong"
                    next_move = "Wave 3 targets 1.618× of Wave 1"
                    wave_number = 3
                    sub_phase = "motive"  # wave 3 is motive
                elif num_waves == 3:
                    # Completed waves 1+2+3, currently in Wave 4 pullback
                    action = "WAIT — Wave 4 pullback in progress"
                    strength = "cautious"
                    next_move = "Wave 4 retraces 23.6-38.2% of Wave 3, then Wave 5 begins"
                    wave_number = 4
                    sub_phase = "corrective"  # wave 4 is corrective within impulse
                elif num_waves == 4:
                    # Completed waves 1-4, currently in Wave 5
                    action = "SETUP — Wave 5 final push"
                    strength = "moderate"
                    next_move = "Wave 5 completes impulse, then ABC correction follows"
                    wave_number = 5
                    sub_phase = "motive"  # wave 5 is motive (final)
                else:
                    action = "OBSERVE"
                    strength = "weak"
                    next_move = "Unclear wave position"
                    wave_number = num_waves + 1
                    sub_phase = "unknown"

                bias = f"{strength}_{'bullish' if is_up else 'bearish'}"
                pattern = f"{'bullish' if is_up else 'bearish'}_impulse"

                return {
                    "degree": degree,
                    "phase": "impulse",
                    "wave_number": wave_number,
                    "wave_in_progress": f"Wave {wave_number}",
                    "current_label": next_label,
                    "direction": wc.direction.value,
                    "confidence": round(wc.confidence, 1),
                    "action": action,
                    "bias": bias,
                    "strength": strength,
                    "trend": trend_word,
                    "pattern": pattern,
                    "sub_phase": sub_phase,
                    "next_move": next_move,
                    "type": "partial_impulse",
                    "waves_complete": num_waves,
                    "summary": f"{trend_word} IMPULSE · Wave {wave_number} of 5",
                }

        # ── Complete impulse near current bar ────────────────────────
        for wc in impulses:
            last_wave_end = wc.waves[-1].end.idx
            if last_wave_end >= last_idx - complete_proximity:
                is_up = wc.direction == WaveDirection.UP
                # After a bullish impulse completes → bearish correction expected
                corr_bias = "bearish" if is_up else "bullish"
                trend_word = "BEARISH" if is_up else "BULLISH"  # correction goes opposite

                return {
                    "degree": degree,
                    "phase": "correction",
                    "wave_number": 0,  # A wave starting
                    "wave_in_progress": "Wave A starting",
                    "current_label": "(A)" if degree == "primary" else "A",
                    "direction": "down" if is_up else "up",
                    "confidence": round(wc.confidence, 1),
                    "action": "REVERSE — Impulse complete, ABC correction starting",
                    "bias": corr_bias,
                    "strength": "strong",
                    "trend": trend_word,
                    "pattern": f"{corr_bias}_correction",
                    "sub_phase": "corrective",
                    "next_move": "Wave A targets 38.2-50% retrace of impulse, then B bounce, then C lower",
                    "type": "impulse_complete",
                    "waves_complete": 5,
                    "summary": f"{trend_word} CORRECTION · Wave A of ABC",
                }

        # ── Complete correction near current bar ─────────────────────
        for wc in correctives:
            last_wave_end = wc.waves[-1].end.idx
            if last_wave_end >= last_idx - complete_proximity:
                is_down = wc.direction == WaveDirection.DOWN

                # Handle triangle patterns
                if wc.wave_type == WaveType.TRIANGLE:
                    num_waves = len(wc.waves)
                    # Determine which wave of the triangle we're nearest to
                    tri_labels = DEGREE_LABELS[degree]["triangle"]
                    current_wave_label = tri_labels[num_waves - 1] if num_waves <= 5 else tri_labels[-1]
                    trend_word = "BEARISH" if is_down else "BULLISH"

                    return {
                        "degree": degree,
                        "phase": "correction",
                        "wave_number": num_waves,
                        "wave_in_progress": f"Wave {current_wave_label} of triangle",
                        "current_label": current_wave_label,
                        "direction": wc.direction.value,
                        "confidence": round(wc.confidence, 1),
                        "action": "WAIT — Triangle correction nearing completion, breakout imminent",
                        "bias": "bearish" if is_down else "bullish",
                        "strength": "moderate",
                        "trend": trend_word,
                        "pattern": "triangle_correction",
                        "sub_phase": "corrective",
                        "next_move": "Triangle completion → strong breakout in direction of prior trend",
                        "type": "triangle",
                        "waves_complete": num_waves,
                        "summary": f"TRIANGLE · Wave {current_wave_label} of a-b-c-d-e",
                    }

                # Handle ending diagonal patterns
                if wc.wave_type == WaveType.DIAGONAL:
                    num_waves = len(wc.waves)
                    is_up = wc.direction == WaveDirection.UP
                    trend_word = "BULLISH" if is_up else "BEARISH"
                    diag_labels = DEGREE_LABELS[degree]["diagonal"]
                    current_wave_label = diag_labels[num_waves - 1] if num_waves <= 5 else diag_labels[-1]

                    return {
                        "degree": degree,
                        "phase": "impulse",
                        "wave_number": num_waves,
                        "wave_in_progress": f"Wave {current_wave_label} of diagonal",
                        "current_label": current_wave_label,
                        "direction": wc.direction.value,
                        "confidence": round(wc.confidence, 1),
                        "action": "CAUTION — Ending diagonal suggests trend exhaustion",
                        "bias": "bearish" if is_up else "bullish",
                        "strength": "cautious",
                        "trend": trend_word,
                        "pattern": "ending_diagonal",
                        "sub_phase": "motive",
                        "next_move": "Diagonal completion → sharp reversal expected",
                        "type": "diagonal",
                        "waves_complete": num_waves,
                        "summary": f"ENDING DIAGONAL · Wave {current_wave_label} of 5",
                    }

                # Handle double correction patterns
                if wc.wave_type == WaveType.DOUBLE_CORRECTION:
                    num_waves = len(wc.waves)
                    trend_word = "BEARISH" if is_down else "BULLISH"
                    # Determine WXY position: waves 1-3 = W, wave 4 = X, waves 5-7 = Y
                    if num_waves <= 3:
                        wxy_label = "W"
                    elif num_waves == 4:
                        wxy_label = "X"
                    else:
                        wxy_label = "Y"

                    return {
                        "degree": degree,
                        "phase": "correction",
                        "wave_number": num_waves,
                        "wave_in_progress": f"Wave {wxy_label} of WXY",
                        "current_label": wxy_label,
                        "direction": wc.direction.value,
                        "confidence": round(wc.confidence, 1),
                        "action": "WAIT — Double correction extending, more sideways/down action",
                        "bias": "bearish" if is_down else "bullish",
                        "strength": "cautious",
                        "trend": trend_word,
                        "pattern": "double_correction",
                        "sub_phase": "corrective",
                        "next_move": "Y wave completion → new impulse phase begins",
                        "type": "double_correction",
                        "waves_complete": num_waves,
                        "summary": f"DOUBLE CORRECTION · Wave {wxy_label} connecting W and Y",
                    }

                # Standard ABC correction
                impulse_bias = "bullish" if is_down else "bearish"
                trend_word = "BULLISH" if is_down else "BEARISH"

                return {
                    "degree": degree,
                    "phase": "impulse",
                    "wave_number": 1,
                    "wave_in_progress": "New Wave 1 starting",
                    "current_label": "1" if degree == "intermediate" else "(1)",
                    "direction": "up" if is_down else "down",
                    "confidence": round(wc.confidence, 1),
                    "action": "SETUP — Correction complete, new impulse starting",
                    "bias": impulse_bias,
                    "strength": "strong",
                    "trend": trend_word,
                    "pattern": f"{impulse_bias}_impulse",
                    "sub_phase": "motive",
                    "next_move": "New impulse Wave 1 forming — early trend entry",
                    "type": "correction_complete",
                    "waves_complete": 3,
                    "summary": f"{trend_word} IMPULSE · Wave 1 of 5 (new trend)",
                }

    return {
        "degree": "unknown",
        "phase": "unknown",
        "wave_number": 0,
        "wave_in_progress": "No clear count",
        "current_label": "?",
        "direction": "neutral",
        "confidence": 0,
        "action": "NO CLEAR WAVE STRUCTURE",
        "bias": "neutral",
        "strength": "none",
        "trend": "NEUTRAL",
        "pattern": "none",
        "sub_phase": "unknown",
        "next_move": "",
        "type": "none",
        "waves_complete": 0,
        "summary": "No clear Elliott Wave structure detected",
    }


def _build_wave_list(df: pd.DataFrame, analysis: dict) -> list[WaveCount]:
    """Combine all wave counts from all degrees, sorted by confidence."""
    all_counts: list[WaveCount] = []

    for degree in ["primary", "intermediate", "minor"]:
        for wc in analysis[degree]["impulse"]:
            wc.degree = degree
            all_counts.append(wc)
        for wc in analysis[degree]["corrective"]:
            wc.degree = degree
            all_counts.append(wc)
        for wc in analysis[degree]["partial"]:
            wc.degree = degree
            all_counts.append(wc)

    # Sort: impulse (complete or partial) > corrective, then by recency + confidence
    # Partial impulses near current bar are very valuable (they tell you what wave you're in)
    last_bar = len(df) - 1

    # Peak-alignment: a wave count is only meaningful if its pivots actually
    # land on the real swing extremes of the chart window. Otherwise the
    # drawn zigzag visually ignores the obvious top/bottom, which is exactly
    # what users complain about. Compute the chart's absolute high/low and
    # give counts whose pivots hit those extrema a strong bonus.
    try:
        highs = df["high"].to_numpy() if "high" in df.columns else df["High"].to_numpy()
        lows  = df["low"].to_numpy()  if "low"  in df.columns else df["Low"].to_numpy()
    except Exception:
        highs, lows = None, None

    # Focus peak alignment on the most recent window where the user actually
    # looks (≈ last 60% of bars). Earlier grand extremes would otherwise
    # dominate even when the meaningful current structure is elsewhere.
    peak_window_start = int(last_bar * 0.40)
    if highs is not None and lows is not None and len(highs) > peak_window_start + 10:
        recent_high_idx = peak_window_start + int(highs[peak_window_start:].argmax())
        recent_low_idx  = peak_window_start + int(lows[peak_window_start:].argmin())
        recent_high = float(highs[recent_high_idx])
        recent_low  = float(lows[recent_low_idx])
    else:
        recent_high_idx = recent_low_idx = -1
        recent_high = recent_low = 0.0

    def peak_bonus(wc: WaveCount) -> float:
        """Return 0..1 bonus — higher when the wave's pivots include the
        recent swing high (for up moves) or swing low (for down moves)."""
        if not wc.waves or recent_high_idx < 0:
            return 0.0
        is_up = wc.direction == WaveDirection.UP
        target_idx = recent_high_idx if is_up else recent_low_idx
        target_price = recent_high if is_up else recent_low
        if target_price <= 0:
            return 0.0
        # Allow a small tolerance: 1% of price and 5 bars for alignment.
        price_tol = max(target_price * 0.01, 1e-6)
        idx_tol = 5
        best = 0.0
        for w in wc.waves:
            for p in (w.start, w.end):
                if abs(p.idx - target_idx) <= idx_tol and abs(p.price - target_price) <= price_tol:
                    # Close match — full bonus
                    return 1.0
                # Partial credit: if the pivot price is within 1.5% of extreme
                price_err = abs(p.price - target_price) / target_price
                if price_err < 0.015:
                    best = max(best, 1.0 - price_err / 0.015)
        return best

    def sort_key(wc: WaveCount) -> tuple:
        is_impulse = wc.wave_type == WaveType.IMPULSE
        is_diagonal = wc.wave_type == WaveType.DIAGONAL
        is_corrective = wc.wave_type in (WaveType.CORRECTIVE, WaveType.TRIANGLE, WaveType.DOUBLE_CORRECTION)
        type_rank = 0 if (is_impulse and wc.is_complete) else \
                    1 if (is_impulse and not wc.is_complete) else \
                    2 if (is_diagonal and wc.is_complete) else \
                    3 if (is_corrective and wc.is_complete) else 4
        # Recency bonus: waves ending closer to current bar sort earlier
        end_idx = wc.waves[-1].end.idx if wc.waves else 0
        recency = end_idx / max(last_bar, 1)  # 0..1, higher = more recent
        peak = peak_bonus(wc)
        # Peak alignment is weighted HEAVILY — a wave that visibly ignores
        # the chart extreme is worse than a lower-confidence one that hits it.
        # wc.confidence is 0..100, normalize to 0..1 to match recency/peak scale.
        conf_norm = wc.confidence / 100.0
        score = conf_norm * 0.35 + recency * 0.25 + peak * 0.40
        return (type_rank, -score)

    all_counts.sort(key=sort_key)
    return all_counts


def _compute_projections(wc: WaveCount) -> list[dict[str, Any]]:
    """
    Compute Fibonacci projection targets for the NEXT expected wave.

    For partial impulses:
    - After Wave 2: project Wave 3 targets (1.618x, 2.0x, 2.618x of Wave 1)
    - After Wave 3: project Wave 4 targets (0.236x, 0.382x retrace of Wave 3)
    - After Wave 4: project Wave 5 targets (0.618x, 1.0x, 1.618x of Wave 1 from Wave 4)

    For complete impulses:
    - Project ABC correction targets (0.382x, 0.500x, 0.618x retrace of entire impulse)

    For complete correctives:
    - Project new Wave 1 impulse target
    """
    projections: list[dict[str, Any]] = []
    waves = wc.waves

    if wc.wave_type == WaveType.IMPULSE and not wc.is_complete:
        num = len(waves)
        w1 = waves[0]
        is_up = wc.direction == WaveDirection.UP

        if num == 2:
            # Project Wave 3 — the strongest wave
            w2_end = waves[1].end.price
            w1_size = w1.price_change
            for fib, label in [(1.618, "W3 · 1.618"), (2.0, "W3 · 2.0"), (2.618, "W3 · 2.618")]:
                target = w2_end + w1_size * fib if is_up else w2_end - w1_size * fib
                projections.append({
                    "label": label,
                    "price": round(target, 2),
                    "fib": fib,
                    "from_idx": waves[1].end.idx,
                    "type": "target",
                    "wave": 3,
                })

        elif num == 3:
            # Project Wave 4 pullback
            w3 = waves[2]
            w3_size = w3.price_change
            for fib, label in [(0.236, "W4 · 0.236"), (0.382, "W4 · 0.382"), (0.500, "W4 · 0.500")]:
                target = w3.end.price - w3_size * fib if is_up else w3.end.price + w3_size * fib
                projections.append({
                    "label": label,
                    "price": round(target, 2),
                    "fib": fib,
                    "from_idx": w3.end.idx,
                    "type": "retrace",
                    "wave": 4,
                })

        elif num == 4:
            # Project Wave 5 targets
            w4_end = waves[3].end.price
            w1_size = w1.price_change
            for fib, label in [(0.618, "W5 · 0.618"), (1.0, "W5 · 1.0"), (1.618, "W5 · 1.618")]:
                target = w4_end + w1_size * fib if is_up else w4_end - w1_size * fib
                projections.append({
                    "label": label,
                    "price": round(target, 2),
                    "fib": fib,
                    "from_idx": waves[3].end.idx,
                    "type": "target",
                    "wave": 5,
                })

    elif wc.wave_type == WaveType.IMPULSE and wc.is_complete:
        # ── Predictability watch after a complete impulse ──────────────────
        # Two scenarios after Wave 5 prints:
        #   1) ABC correction unfolds (most common immediate next move)
        #   2) Trend continues with a new impulse to extension targets
        # Project both so the user can watch which scenario plays out and
        # have explicit invalidation/target levels for each.
        is_up = wc.direction == WaveDirection.UP
        impulse_start = waves[0].start.price
        impulse_end = waves[-1].end.price
        impulse_size = abs(impulse_end - impulse_start)
        last_idx = waves[-1].end.idx

        # ── ABC corrective targets ──
        # Wave A: typical 0.382 / 0.5 / 0.618 retrace of the full impulse
        # Wave B: 0.5 / 0.618 bounce of Wave A
        # Wave C: extends to 1.0× / 1.618× of Wave A from Wave B
        for fib, label, wlabel in [
            (0.382, "Wave A · 0.382", "A"),
            (0.500, "Wave A · 0.500", "A"),
            (0.618, "Wave A · 0.618", "A"),
        ]:
            target = impulse_end - impulse_size * fib if is_up else impulse_end + impulse_size * fib
            projections.append({
                "label": label,
                "price": round(target, 2),
                "fib": fib,
                "from_idx": last_idx,
                "type": "retrace",
                "wave": "A",
                "scenario": "correction",
            })

        # Wave B bounce — measured from the typical Wave A target (0.382),
        # which is the most common A-leg retracement in real markets.
        wave_a_target = impulse_end - impulse_size * 0.382 if is_up else impulse_end + impulse_size * 0.382
        wave_a_size = abs(impulse_end - wave_a_target)
        for fib, label in [(0.500, "Wave B · 0.500"), (0.618, "Wave B · 0.618")]:
            target = wave_a_target + wave_a_size * fib if is_up else wave_a_target - wave_a_size * fib
            projections.append({
                "label": label,
                "price": round(target, 2),
                "fib": fib,
                "from_idx": last_idx,
                "type": "bounce",
                "wave": "B",
                "scenario": "correction",
            })

        # Wave C terminal targets — projected from Wave B end as fib × Wave A
        wave_b_target = wave_a_target + wave_a_size * 0.500 if is_up else wave_a_target - wave_a_size * 0.500
        for fib, label in [(1.0, "Wave C · 1.0"), (1.618, "Wave C · 1.618")]:
            target = wave_b_target - wave_a_size * fib if is_up else wave_b_target + wave_a_size * fib
            projections.append({
                "label": label,
                "price": round(target, 2),
                "fib": fib,
                "from_idx": last_idx,
                "type": "target",
                "wave": "C",
                "scenario": "correction",
            })

        # ── Continuation targets (if trend extends past Wave 5) ──
        # Treats the completed impulse as wave 1 of a larger degree.
        for fib, label in [(1.272, "Cont · 1.272"), (1.618, "Cont · 1.618")]:
            target = impulse_end + impulse_size * (fib - 1.0) if is_up else impulse_end - impulse_size * (fib - 1.0)
            projections.append({
                "label": label,
                "price": round(target, 2),
                "fib": fib,
                "from_idx": last_idx,
                "type": "extension",
                "wave": "Cont",
                "scenario": "continuation",
            })

        # Invalidation level: a move beyond Wave 5 end in the impulse direction
        # negates the corrective scenario; a move below Wave 4 confirms it.
        wave4_end = waves[3].end.price
        projections.append({
            "label": "Correction Trigger (W4 break)",
            "price": round(wave4_end, 2),
            "fib": 0.0,
            "from_idx": last_idx,
            "type": "trigger",
            "wave": "A",
            "scenario": "correction",
        })

    elif wc.wave_type == WaveType.CORRECTIVE and wc.is_complete:
        # Complete correction — project new impulse Wave 1
        is_down_correction = wc.direction == WaveDirection.DOWN
        corr_start = waves[0].start.price
        corr_end = waves[-1].end.price
        corr_size = abs(corr_start - corr_end)
        last_idx = waves[-1].end.idx

        for fib, label in [(1.0, "New W1 · 1.0"), (1.618, "New W1 · 1.618")]:
            if is_down_correction:
                target = corr_end + corr_size * fib  # new impulse up
            else:
                target = corr_end - corr_size * fib  # new impulse down
            projections.append({
                "label": label,
                "price": round(target, 2),
                "fib": fib,
                "from_idx": last_idx,
                "type": "target",
                "wave": 1,
            })

    return projections


def _serialize_wave_count(wc: WaveCount) -> dict[str, Any]:
    """Convert a WaveCount to a JSON-serializable dict."""
    degree_labels = DEGREE_LABELS.get(wc.degree, DEGREE_LABELS["intermediate"])
    # Choose the correct label set based on wave type
    type_to_label_key = {
        WaveType.IMPULSE: "impulse",
        WaveType.CORRECTIVE: "corrective",
        WaveType.TRIANGLE: "triangle",
        WaveType.DIAGONAL: "diagonal",
        WaveType.DOUBLE_CORRECTION: "double_correction",
    }
    label_key = type_to_label_key.get(wc.wave_type, "corrective")
    label_set = degree_labels.get(label_key, degree_labels["corrective"])

    wave_data = []
    for i, w in enumerate(wc.waves):
        label = label_set[i] if i < len(label_set) else str(i + 1)
        wave_data.append({
            "label": label,
            "start_idx": w.start.idx,
            "start_price": round(w.start.price, 4),
            "end_idx": w.end.idx,
            "end_price": round(w.end.price, 4),
            "direction": w.direction.value,
            "price_change": round(w.price_change, 4),
            "bar_count": w.bar_count,
        })

    # Compute Fibonacci projections for next expected wave
    projections = _compute_projections(wc)

    return {
        "type": wc.wave_type.value,
        "degree": wc.degree,
        "direction": wc.direction.value,
        "confidence": round(wc.confidence, 1),
        "fib_score": round(wc.fib_score, 1),
        "is_complete": wc.is_complete,
        "waves": wave_data,
        "labels": [w["label"] for w in wave_data],
        "projections": projections,
    }


def serialize_all_waves(wave_counts: list[WaveCount]) -> list[dict[str, Any]]:
    """Serialize a list of wave counts for API response."""
    return [_serialize_wave_count(wc) for wc in wave_counts[:10]]  # Top 10


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
