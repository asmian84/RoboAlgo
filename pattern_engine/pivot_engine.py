"""
RoboAlgo — Pivot Engine
Vectorised pivot-point detection and structural-pattern classifier.

Detects:
  • Symmetric / Ascending / Descending Triangles
  • Rising / Falling Wedges
  • Bull / Bear Flags
  • Horizontal Channels
  • Cup and Handle

Algorithm:
  1. ZigZag pivot detection (vectorised NumPy).
  2. Trendline regression over last N pivots.
  3. Geometric classification of upper/lower trendline pair.
  4. Confidence scoring based on touch count, R², convergence.

Performance target: ≤ 2 ms per symbol on 400-bar series.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Pivot:
    idx:      int
    price:    float
    kind:     Literal["H", "L"]  # high or low


@dataclass
class TrendLine:
    slope:    float     # price-per-bar
    intercept: float
    r2:       float     # goodness of fit over touch points
    touches:  int
    start_idx: int
    end_idx:  int

    def price_at(self, idx: int) -> float:
        return self.intercept + self.slope * idx


@dataclass
class PatternResult:
    pattern_name:      str
    pattern_category:  str = "chart"
    direction:         str = "neutral"    # bullish | bearish | neutral
    confidence:        float = 0.0        # 0-100
    breakout_level:    float | None = None
    invalidation_level: float | None = None
    target:            float | None = None
    upper_line:        TrendLine | None = None
    lower_line:        TrendLine | None = None
    points:            list[dict] = field(default_factory=list)
    bars_forming:      int = 0


# ─────────────────────────────────────────────────────────────────────────────
#  Pivot detection (vectorised)
# ─────────────────────────────────────────────────────────────────────────────

def find_pivots(
    high: np.ndarray,
    low:  np.ndarray,
    left_bars:  int = 5,
    right_bars: int = 5,
) -> list[Pivot]:
    """
    ZigZag pivot detection using a sliding window comparison.

    A bar i is a pivot HIGH if:  high[i] == max(high[i-L : i+R+1])
    A bar i is a pivot LOW  if:  low[i]  == min(low[i-L  : i+R+1])

    Returns list of Pivot sorted by idx.
    """
    n       = len(high)
    pivots: list[Pivot] = []
    L, R    = left_bars, right_bars

    # Pre-compute rolling max/min with stride tricks for speed
    # We use a simple vectorised approach: compare each bar to its window.
    for i in range(L, n - R):
        h_win = high[i - L : i + R + 1]
        l_win = low[i  - L : i + R + 1]
        if high[i] == h_win.max():
            pivots.append(Pivot(idx=i, price=float(high[i]), kind="H"))
        if low[i] == l_win.min():
            pivots.append(Pivot(idx=i, price=float(low[i]),  kind="L"))

    # Sort and deduplicate overlapping pivots (prefer the more extreme one)
    pivots.sort(key=lambda p: p.idx)
    deduplicated: list[Pivot] = []
    seen_idx: set[int] = set()
    for pv in pivots:
        if pv.idx not in seen_idx:
            deduplicated.append(pv)
            seen_idx.add(pv.idx)

    return deduplicated


def _trendline(pivots: list[Pivot], n_bars: int) -> TrendLine | None:
    """Least-squares trendline through a list of pivots."""
    if len(pivots) < 2:
        return None
    xs = np.array([p.idx   for p in pivots], dtype=float)
    ys = np.array([p.price for p in pivots], dtype=float)
    # Normalise x to reduce numerical error
    x0 = xs[0]
    xn = xs - x0
    if xn.std() < 1e-9:
        return None
    slope, intercept_n = np.polyfit(xn, ys, 1)
    intercept = intercept_n - slope * x0
    # R²
    y_pred = slope * xs + intercept
    ss_res = np.sum((ys - y_pred) ** 2)
    ss_tot = np.sum((ys - ys.mean()) ** 2)
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 1e-10 else 1.0
    return TrendLine(
        slope      = float(slope),
        intercept  = float(intercept),
        r2         = float(max(0, r2)),
        touches    = len(pivots),
        start_idx  = int(xs[0]),
        end_idx    = int(xs[-1]),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Pattern classifiers
# ─────────────────────────────────────────────────────────────────────────────

class PivotEngine:
    """
    Detects structural chart patterns from OHLCV data.

    Usage:
        engine = PivotEngine()
        results = engine.detect(high, low, close, volume)
    """

    def __init__(
        self,
        left_bars:  int = 5,
        right_bars: int = 5,
        min_pivots: int = 4,
        lookback:   int = 120,
    ):
        self.left_bars  = left_bars
        self.right_bars = right_bars
        self.min_pivots = min_pivots
        self.lookback   = lookback

    # ── Public ────────────────────────────────────────────────────────────

    def detect(
        self,
        high:   np.ndarray,
        low:    np.ndarray,
        close:  np.ndarray,
        volume: np.ndarray | None = None,
        dates:  list | None = None,
    ) -> list[PatternResult]:
        """
        Run all detectors on the provided bar data.
        Returns a list of PatternResult, best-confidence first.
        """
        # Use only the last `lookback` bars
        h = high[-self.lookback:]
        l = low[-self.lookback:]
        c = close[-self.lookback:]
        v = volume[-self.lookback:] if volume is not None else np.ones(len(h))

        pivots = find_pivots(h, l, self.left_bars, self.right_bars)
        if len(pivots) < self.min_pivots:
            return []

        highs = [p for p in pivots if p.kind == "H"]
        lows  = [p for p in pivots if p.kind == "L"]

        results: list[PatternResult] = []

        detectors = [
            self._detect_triangle,
            self._detect_wedge,
            self._detect_flag,
            self._detect_channel,
            self._detect_cup_handle,
        ]
        for detector in detectors:
            try:
                r = detector(highs, lows, c, v)
                if r:
                    results.append(r)
            except Exception as exc:
                logger.debug("pivot detector %s failed: %s", detector.__name__, exc)

        results.sort(key=lambda r: r.confidence, reverse=True)
        return results

    # ── Triangle ──────────────────────────────────────────────────────────

    def _detect_triangle(
        self,
        highs: list[Pivot],
        lows:  list[Pivot],
        close: np.ndarray,
        volume: np.ndarray,
    ) -> PatternResult | None:
        if len(highs) < 2 or len(lows) < 2:
            return None

        upper = _trendline(highs[-4:], len(close))
        lower = _trendline(lows[-4:],  len(close))
        if upper is None or lower is None:
            return None

        apex_idx = self._apex(upper, lower)
        if apex_idx is None or apex_idx < len(close) - 1:
            return None   # apex not in future

        current_idx = len(close) - 1
        breakout_level    = upper.price_at(current_idx)
        invalidation_level = lower.price_at(current_idx)

        upper_slope_n = upper.slope / max(close[-1], 1e-6)
        lower_slope_n = lower.slope / max(close[-1], 1e-6)

        # Classify triangle type
        sym_th = 0.0003
        if abs(upper_slope_n) < sym_th and abs(lower_slope_n) < sym_th:
            name = "Symmetric Triangle"
            direction = "neutral"
        elif abs(upper_slope_n) < sym_th and lower_slope_n > sym_th:
            name = "Ascending Triangle"
            direction = "bullish"
        elif upper_slope_n < -sym_th and abs(lower_slope_n) < sym_th:
            name = "Descending Triangle"
            direction = "bearish"
        elif upper_slope_n < -sym_th and lower_slope_n > sym_th:
            name = "Symmetric Triangle"
            direction = "neutral"
        else:
            return None

        # Confidence: based on R², touch count, convergence progress
        bars_forming = max(highs[-1].idx, lows[-1].idx) - min(highs[0].idx, lows[0].idx)
        convergence  = 1.0 - (current_idx / max(apex_idx, current_idx + 1))
        conf = (
            0.35 * (upper.r2 + lower.r2) / 2 * 100
            + 0.25 * min((len(highs) + len(lows)) / 8, 1) * 100
            + 0.25 * min(convergence * 2, 1) * 100
            + 0.15 * (1 - abs(upper.r2 - lower.r2)) * 100
        )

        # Volume decreasing inside pattern is confirmation
        if len(volume) >= 10:
            v_early = volume[-bars_forming - 5 : -bars_forming + 5].mean() if bars_forming > 5 else volume[:5].mean()
            v_late  = volume[-5:].mean()
            if v_late < v_early * 0.8:
                conf = min(conf + 8, 100)

        height = (breakout_level - invalidation_level)
        target  = breakout_level + height if direction == "bullish" else invalidation_level - height

        return PatternResult(
            pattern_name       = name,
            direction          = direction,
            confidence         = round(conf, 1),
            breakout_level     = round(breakout_level, 4),
            invalidation_level = round(invalidation_level, 4),
            target             = round(target, 4),
            upper_line         = upper,
            lower_line         = lower,
            bars_forming       = bars_forming,
            points = [
                {"idx": p.idx, "price": p.price, "kind": p.kind}
                for p in sorted(highs[-4:] + lows[-4:], key=lambda x: x.idx)
            ],
        )

    # ── Wedge ─────────────────────────────────────────────────────────────

    def _detect_wedge(
        self,
        highs: list[Pivot],
        lows:  list[Pivot],
        close: np.ndarray,
        volume: np.ndarray,
    ) -> PatternResult | None:
        if len(highs) < 2 or len(lows) < 2:
            return None

        upper = _trendline(highs[-4:], len(close))
        lower = _trendline(lows[-4:],  len(close))
        if upper is None or lower is None:
            return None

        upper_slope_n = upper.slope / max(close[-1], 1e-6)
        lower_slope_n = lower.slope / max(close[-1], 1e-6)

        # Wedge: both lines slope same direction but converge
        same_dir_up   = upper_slope_n > 0.0002 and lower_slope_n > 0.0002
        same_dir_down = upper_slope_n < -0.0002 and lower_slope_n < -0.0002
        if not (same_dir_up or same_dir_down):
            return None

        converging = abs(upper_slope_n) != abs(lower_slope_n)
        if not converging:
            return None

        current_idx = len(close) - 1
        breakout_level    = upper.price_at(current_idx)
        invalidation_level = lower.price_at(current_idx)

        if same_dir_up:
            name = "Rising Wedge"
            direction = "bearish"    # rising wedge breaks down
        else:
            name = "Falling Wedge"
            direction = "bullish"    # falling wedge breaks up

        bars_forming = max(highs[-1].idx, lows[-1].idx) - min(highs[0].idx, lows[0].idx)
        conf = (
            0.40 * (upper.r2 + lower.r2) / 2 * 100
            + 0.30 * min((len(highs) + len(lows)) / 6, 1) * 100
            + 0.30 * min(bars_forming / 20, 1) * 100
        )

        height = abs(breakout_level - invalidation_level)
        target = (invalidation_level - height) if direction == "bearish" else (breakout_level + height)

        return PatternResult(
            pattern_name       = name,
            direction          = direction,
            confidence         = round(conf, 1),
            breakout_level     = round(breakout_level, 4),
            invalidation_level = round(invalidation_level, 4),
            target             = round(target, 4),
            upper_line         = upper,
            lower_line         = lower,
            bars_forming       = bars_forming,
            points = [
                {"idx": p.idx, "price": p.price, "kind": p.kind}
                for p in sorted(highs[-4:] + lows[-4:], key=lambda x: x.idx)
            ],
        )

    # ── Flag ──────────────────────────────────────────────────────────────

    def _detect_flag(
        self,
        highs: list[Pivot],
        lows:  list[Pivot],
        close: np.ndarray,
        volume: np.ndarray,
    ) -> PatternResult | None:
        """
        Flag detection:
          • Sharp prior move (pole) of ≥ 5% in ≤ 15 bars
          • Followed by tight consolidation (flag) in opposite direction
        """
        n = len(close)
        if n < 30:
            return None

        # Detect pole: biggest move in last 60 bars
        pole_window = min(60, n - 1)
        segment     = close[-(pole_window + 1):]
        pct_moves   = np.diff(segment) / segment[:-1] * 100

        # Bullish flag: strong up-move pole, then slight consolidation
        # Bearish flag: strong down-move pole, then slight consolidation

        # Find highest 5-bar gain
        bull_gains = np.array([segment[i + 5] / segment[i] - 1 for i in range(len(segment) - 5)])
        bear_gains = np.array([segment[i] / segment[i + 5] - 1 for i in range(len(segment) - 5)])

        bull_pole_idx = int(np.argmax(bull_gains))
        bear_pole_idx = int(np.argmax(bear_gains))
        bull_move = float(bull_gains[bull_pole_idx]) if len(bull_gains) else 0.0
        bear_move = float(bear_gains[bear_pole_idx]) if len(bear_gains) else 0.0

        direction: str
        pole_move: float
        pole_start: int
        pole_end:   int

        if bull_move >= 0.05 and bull_move >= bear_move:
            direction  = "bullish"
            pole_move  = bull_move
            pole_start = -(pole_window - bull_pole_idx + 5)
            pole_end   = -(pole_window - bull_pole_idx)
        elif bear_move >= 0.05:
            direction  = "bearish"
            pole_move  = bear_move
            pole_start = -(pole_window - bear_pole_idx + 5)
            pole_end   = -(pole_window - bear_pole_idx)
        else:
            return None

        # Flag body: bars after the pole
        flag_bars = close[pole_end:]
        if len(flag_bars) < 5:
            return None

        flag_range = (flag_bars.max() - flag_bars.min()) / max(flag_bars.mean(), 1e-6)
        if flag_range > 0.06:
            return None    # too wide — not a flag

        # Slight counter-trend drift is expected
        flag_drift = (flag_bars[-1] - flag_bars[0]) / max(flag_bars[0], 1e-6)
        if direction == "bullish" and flag_drift > 0.01:
            return None    # rising flag on bullish is not valid
        if direction == "bearish" and flag_drift < -0.01:
            return None

        current_price = float(close[-1])
        breakout_level    = float(flag_bars.max()) if direction == "bullish" else float(flag_bars.min())
        invalidation_level = float(flag_bars.min()) if direction == "bullish" else float(flag_bars.max())
        pole_height       = abs(pole_move * current_price)
        target = breakout_level + pole_height if direction == "bullish" else breakout_level - pole_height

        # Volume: pole should have higher volume than flag
        v_pole = float(volume[pole_end - 5 : pole_end].mean()) if len(volume) > 5 else 1.0
        v_flag = float(volume[pole_end:].mean())
        vol_conf = min((v_pole / max(v_flag, 1e-6)) / 2, 1) * 100

        conf = (
            0.35 * min(pole_move / 0.10, 1) * 100
            + 0.30 * (1 - flag_range / 0.06) * 100
            + 0.20 * vol_conf
            + 0.15 * min(len(flag_bars) / 10, 1) * 100
        )

        name = "Bull Flag" if direction == "bullish" else "Bear Flag"
        return PatternResult(
            pattern_name       = name,
            direction          = direction,
            confidence         = round(conf, 1),
            breakout_level     = round(breakout_level, 4),
            invalidation_level = round(invalidation_level, 4),
            target             = round(target, 4),
            bars_forming       = len(flag_bars),
            points             = [],
        )

    # ── Channel ───────────────────────────────────────────────────────────

    def _detect_channel(
        self,
        highs: list[Pivot],
        lows:  list[Pivot],
        close: np.ndarray,
        volume: np.ndarray,
    ) -> PatternResult | None:
        if len(highs) < 3 or len(lows) < 3:
            return None

        upper = _trendline(highs[-5:], len(close))
        lower = _trendline(lows[-5:],  len(close))
        if upper is None or lower is None:
            return None

        # Channel: parallel lines (slope within 15% of each other)
        slope_diff = abs(upper.slope - lower.slope) / max(abs(upper.slope) + abs(lower.slope), 1e-9)
        if slope_diff > 0.15:
            return None

        current_idx = len(close) - 1
        upper_price = upper.price_at(current_idx)
        lower_price = lower.price_at(current_idx)

        upper_slope_n = upper.slope / max(close[-1], 1e-6)
        if upper_slope_n > 0.0001:
            name = "Ascending Channel"
            direction = "bullish"
        elif upper_slope_n < -0.0001:
            name = "Descending Channel"
            direction = "bearish"
        else:
            name = "Horizontal Channel"
            direction = "neutral"

        bars_forming = max(highs[-1].idx, lows[-1].idx) - min(highs[0].idx, lows[0].idx)
        conf = (
            0.35 * (upper.r2 + lower.r2) / 2 * 100
            + 0.35 * (1 - slope_diff) * 100
            + 0.30 * min((upper.touches + lower.touches) / 8, 1) * 100
        )
        height = upper_price - lower_price
        target = upper_price + height if direction == "bullish" else lower_price - height

        return PatternResult(
            pattern_name       = name,
            direction          = direction,
            confidence         = round(conf, 1),
            breakout_level     = round(upper_price, 4),
            invalidation_level = round(lower_price, 4),
            target             = round(target, 4),
            upper_line         = upper,
            lower_line         = lower,
            bars_forming       = bars_forming,
            points = [
                {"idx": p.idx, "price": p.price, "kind": p.kind}
                for p in sorted(highs[-5:] + lows[-5:], key=lambda x: x.idx)
            ],
        )

    # ── Cup and Handle ────────────────────────────────────────────────────

    def _detect_cup_handle(
        self,
        highs: list[Pivot],
        lows:  list[Pivot],
        close: np.ndarray,
        volume: np.ndarray,
    ) -> PatternResult | None:
        """
        Cup and Handle:
          • U-shaped correction (cup): 10-65 bar depth, ≥ 12% decline
          • Small consolidation after recovery (handle): < 50% of cup depth
          • Volume: heavier on left side of cup, lighter during handle
        """
        n = len(close)
        if n < 50:
            return None

        # Find leftmost high, deepest low, recovery high
        if not highs or not lows:
            return None

        # Simple heuristic: find the biggest valley in last 100 bars
        window = close[-100:]
        if len(window) < 30:
            return None

        # Left rim: find the first major high
        left_high_idx = int(np.argmax(window[:len(window) // 2]))
        left_high     = float(window[left_high_idx])

        # Cup bottom
        cup_segment   = window[left_high_idx:]
        if len(cup_segment) < 10:
            return None
        bottom_idx_rel = int(np.argmin(cup_segment))
        bottom_idx     = left_high_idx + bottom_idx_rel
        bottom_price   = float(window[bottom_idx])

        cup_depth = (left_high - bottom_price) / max(left_high, 1e-6)
        if cup_depth < 0.12:
            return None    # not deep enough

        # Right rim: close must recover to within 3% of left high
        right_segment  = window[bottom_idx:]
        if len(right_segment) < 5:
            return None
        right_high     = float(right_segment.max())
        recovery_ratio = right_high / left_high
        if recovery_ratio < 0.97:
            return None    # hasn't recovered

        # Handle: last portion should be a small pullback
        handle_segment = window[bottom_idx + int(np.argmax(right_segment)):]
        if len(handle_segment) < 3:
            return None
        handle_drop = (handle_segment[0] - handle_segment.min()) / max(handle_segment[0], 1e-6)
        if handle_drop > cup_depth * 0.5:
            return None    # handle too deep

        breakout_level    = float(right_high)
        invalidation_level = float(handle_segment.min())
        target             = breakout_level + (breakout_level - bottom_price)

        # Volume: left side should be higher
        mid  = (left_high_idx + bottom_idx) // 2
        v_left  = volume[-100 + left_high_idx : -100 + mid].mean() if mid > left_high_idx else 1.0
        v_right = volume[-100 + mid : -100 + bottom_idx + 5].mean() if bottom_idx > mid else 1.0
        vol_ok  = v_left > v_right

        conf = (
            0.30 * min(cup_depth / 0.35, 1) * 100
            + 0.30 * recovery_ratio * 100
            + 0.25 * (1 - handle_drop / max(cup_depth * 0.5, 0.01)) * 100
            + 0.15 * (100 if vol_ok else 30)
        )

        return PatternResult(
            pattern_name       = "Cup and Handle",
            direction          = "bullish",
            confidence         = round(conf, 1),
            breakout_level     = round(breakout_level, 4),
            invalidation_level = round(invalidation_level, 4),
            target             = round(target, 4),
            bars_forming       = len(window) - left_high_idx,
            points             = [],
        )

    # ── Utilities ─────────────────────────────────────────────────────────

    @staticmethod
    def _apex(upper: TrendLine, lower: TrendLine) -> int | None:
        """Index where two trendlines intersect (can be future)."""
        dslope = upper.slope - lower.slope
        if abs(dslope) < 1e-10:
            return None
        idx = (lower.intercept - upper.intercept) / dslope
        return int(idx)

    @staticmethod
    def result_to_dict(r: PatternResult) -> dict:
        return {
            "pattern_name":       r.pattern_name,
            "pattern_category":   r.pattern_category,
            "direction":          r.direction,
            "confidence":         r.confidence,
            "breakout_level":     r.breakout_level,
            "invalidation_level": r.invalidation_level,
            "target":             r.target,
            "bars_forming":       r.bars_forming,
            "points":             r.points,
        }
