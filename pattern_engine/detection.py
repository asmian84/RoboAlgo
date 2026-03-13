"""
RoboAlgo — Chart Pattern Detection Engine
Algorithmically detects classical chart patterns.

Patterns detected:
  - Flag (bull/bear)           - Consolidation after sharp move
  - Ascending/Descending Triangle - Flat top + rising lows (or flat bottom + falling highs)
  - Symmetric Triangle         - Converging highs/lows
  - Wedge (rising/falling)     - Both lines slope same direction
  - Channel (up/down)          - Parallel trendlines
  - Head & Shoulders           - 3-peak reversal pattern
  - Inverse Head & Shoulders   - 3-trough reversal pattern

Each pattern returns: pattern_type, direction, breakout_price, target_price, stop_price, confidence
"""

import logging
from typing import Optional
from datetime import timedelta

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from sqlalchemy import select, desc
from sqlalchemy.dialects.postgresql import insert as pg_insert
from tqdm import tqdm

from database.connection import get_session
from database.models import Instrument, PriceData, ChartPattern

logger = logging.getLogger(__name__)

# ── Parameters ─────────────────────────────────────────────────────────────────
MIN_PATTERN_BARS  = 5    # minimum bars to form a pattern
MAX_PATTERN_BARS  = 60   # maximum lookback for pattern detection
PIVOT_ORDER       = 5    # argrelextrema order for swing point detection
FLAG_MAX_RETRACE  = 0.5  # flag retracement maximum (% of pole)
MIN_CONFIDENCE    = 0.40 # minimum confidence to store pattern


class ChartPatternEngine:
    """
    Detects chart patterns from OHLCV data.

    Usage:
        engine = ChartPatternEngine()
        engine.compute_and_store()
        patterns = engine.get_latest("SOXL")
    """

    # ── Public API ─────────────────────────────────────────────────────────────

    def compute_and_store(self, symbol: Optional[str] = None) -> int:
        """Detect and store chart patterns for one or all instruments."""
        with get_session() as session:
            if symbol:
                instruments = session.execute(
                    select(Instrument).where(Instrument.symbol == symbol.upper())
                ).scalars().all()
            else:
                instruments = session.execute(select(Instrument)).scalars().all()

            total = 0
            for inst in tqdm(instruments, desc="Chart patterns"):
                try:
                    rows = self._process_instrument(session, inst)
                    if rows:
                        self._upsert(session, rows)
                        total += len(rows)
                except Exception as e:
                    logger.warning(f"Pattern detection failed for {inst.symbol}: {e}")
            logger.info(f"Chart pattern engine: stored {total} pattern rows.")
            return total

    def get_latest(self, symbol: str, limit: int = 5) -> list[dict]:
        """Return the most recent chart patterns for a symbol."""
        with get_session() as session:
            instr = session.execute(
                select(Instrument).where(Instrument.symbol == symbol.upper())
            ).scalar_one_or_none()
            if not instr:
                return []
            rows = session.execute(
                select(ChartPattern)
                .where(ChartPattern.instrument_id == instr.id)
                .order_by(desc(ChartPattern.date))
                .limit(limit)
            ).scalars().all()
            return [self._row_to_dict(r) for r in rows]

    def get_pattern_score(self, symbol: str) -> float:
        """Return best pattern confidence score (0–100) for confluence."""
        patterns = self.get_latest(symbol, limit=3)
        if not patterns:
            return 50.0
        # Use the highest-confidence recent pattern
        best = max(patterns, key=lambda p: p.get("pattern_score", 0))
        return float(best.get("pattern_score", 50.0))

    # ── Internal Processing ────────────────────────────────────────────────────

    def _process_instrument(self, session, instrument) -> list[dict]:
        """Detect patterns for one instrument using sliding window."""
        price_rows = session.execute(
            select(PriceData)
            .where(PriceData.instrument_id == instrument.id)
            .order_by(PriceData.date)
        ).scalars().all()

        if len(price_rows) < MAX_PATTERN_BARS + 5:
            return []

        prices = pd.DataFrame([{
            "date": r.date, "open": r.open, "high": r.high,
            "low": r.low, "close": r.close, "volume": r.volume,
        } for r in price_rows]).set_index("date")

        return self._detect_all_patterns(instrument.id, prices)

    def _detect_all_patterns(self, instrument_id: int, df: pd.DataFrame) -> list[dict]:
        """Run all pattern detectors over the price history."""
        all_records = []
        dates = df.index.tolist()

        # Use rolling 60-bar window, step every 5 bars for performance
        for end_i in range(MAX_PATTERN_BARS, len(dates), 5):
            start_i = max(0, end_i - MAX_PATTERN_BARS)
            window  = df.iloc[start_i:end_i + 1]
            date    = dates[end_i]

            detected = []
            detected.extend(self._detect_flag(instrument_id, date, window))
            detected.extend(self._detect_triangle(instrument_id, date, window))
            detected.extend(self._detect_wedge(instrument_id, date, window))
            detected.extend(self._detect_channel(instrument_id, date, window))
            detected.extend(self._detect_head_and_shoulders(instrument_id, date, window))

            # Only keep patterns with sufficient confidence
            for p in detected:
                if p.get("pattern_confidence", 0) >= MIN_CONFIDENCE:
                    all_records.append(p)

        return all_records

    # ── Flag Pattern ───────────────────────────────────────────────────────────

    def _detect_flag(self, inst_id: int, date, window: pd.DataFrame) -> list[dict]:
        """
        Bull/Bear Flag: Sharp pole move followed by tight consolidation channel.
        Breakout target = flag breakout + pole height.
        """
        if len(window) < 15:
            return []

        close = window["close"].values
        high  = window["high"].values
        low   = window["low"].values

        # Find the "pole" — sharp move in last 20 bars
        results = []
        for pole_len in [5, 7, 10]:
            if len(close) < pole_len + 8:
                continue

            pole = close[-(pole_len + 8):-8]
            flag_section = close[-8:]

            pole_return = (pole[-1] - pole[0]) / max(abs(pole[0]), 1e-8)
            if abs(pole_return) < 0.05:  # pole must be ≥5% move
                continue

            direction = "bullish" if pole_return > 0 else "bearish"

            # Flag: tight range consolidation (max range < 50% of pole move)
            flag_range = flag_section.max() - flag_section.min()
            flag_retrace = flag_range / max(abs(pole[-1] - pole[0]), 1e-8)

            if flag_retrace > FLAG_MAX_RETRACE:
                continue

            # Check slight counter-trend slope in flag
            flag_x = np.arange(len(flag_section))
            flag_slope = np.polyfit(flag_x, flag_section, 1)[0]
            counter_trend = (flag_slope < 0 and direction == "bullish") or (
                flag_slope > 0 and direction == "bearish"
            )

            pole_height = abs(pole[-1] - pole[0])
            if direction == "bullish":
                breakout_price = float(np.max(flag_section[-3:]))
                target_price   = breakout_price + pole_height
                stop_price     = float(np.min(flag_section))
            else:
                breakout_price = float(np.min(flag_section[-3:]))
                target_price   = breakout_price - pole_height
                stop_price     = float(np.max(flag_section))

            confidence = 0.5
            if counter_trend:
                confidence += 0.15
            if flag_retrace < 0.30:
                confidence += 0.15
            if abs(pole_return) > 0.10:
                confidence += 0.10

            results.append({
                "instrument_id":     inst_id,
                "date":              date,
                "pattern_type":      "flag",
                "direction":         direction,
                "sub_type":          f"{'bull' if direction == 'bullish' else 'bear'}_flag",
                "start_date":        window.index[-(pole_len + 8)],
                "bars_in_pattern":   pole_len + 8,
                "breakout_price":    round(breakout_price, 4),
                "target_price":      round(target_price, 4),
                "stop_price":        round(stop_price, 4),
                "pattern_confidence": round(min(confidence, 0.95), 4),
                "pattern_score":     round(min(confidence * 100, 95), 2),
            })

        return results[:1]  # return best flag if found

    # ── Triangle Pattern ───────────────────────────────────────────────────────

    def _detect_triangle(self, inst_id: int, date, window: pd.DataFrame) -> list[dict]:
        """
        Ascending: flat top + rising lows → bullish breakout above flat top
        Descending: flat bottom + falling highs → bearish break below flat bottom
        Symmetric: converging highs/lows → breakout in direction of trend
        """
        if len(window) < 20:
            return []

        high  = window["high"].values
        low   = window["low"].values
        close = window["close"].values[-1]
        bars  = len(window)

        # Find swing pivots
        swing_hi_idx = argrelextrema(window["high"].values,  np.greater, order=PIVOT_ORDER)[0]
        swing_lo_idx = argrelextrema(window["low"].values,   np.less,   order=PIVOT_ORDER)[0]

        if len(swing_hi_idx) < 2 or len(swing_lo_idx) < 2:
            return []

        # Fit trendlines to swing highs and lows
        hi_prices = high[swing_hi_idx]
        lo_prices = low[swing_lo_idx]

        hi_slope = np.polyfit(swing_hi_idx, hi_prices, 1)[0] / max(hi_prices.mean(), 1e-8)
        lo_slope = np.polyfit(swing_lo_idx, lo_prices, 1)[0] / max(lo_prices.mean(), 1e-8)

        results = []

        # Ascending triangle: flat top (|hi_slope| < 0.001) + rising lows
        if abs(hi_slope) < 0.001 and lo_slope > 0.0005:
            flat_top       = float(hi_prices.max())
            breakout_price = flat_top
            target_price   = flat_top + (flat_top - lo_prices.min())
            stop_price     = float(lo_prices[-1]) if len(lo_prices) else flat_top * 0.97
            confidence     = min(0.55 + len(swing_hi_idx) * 0.05, 0.85)
            results.append(self._make_triangle_record(
                inst_id, date, window, "ascending_triangle", "bullish",
                breakout_price, target_price, stop_price, confidence
            ))

        # Descending triangle: flat bottom + falling highs
        elif abs(lo_slope) < 0.001 and hi_slope < -0.0005:
            flat_bottom    = float(lo_prices.min())
            breakout_price = flat_bottom
            target_price   = flat_bottom - (hi_prices.max() - flat_bottom)
            stop_price     = float(hi_prices[-1]) if len(hi_prices) else flat_bottom * 1.03
            confidence     = min(0.55 + len(swing_lo_idx) * 0.05, 0.85)
            results.append(self._make_triangle_record(
                inst_id, date, window, "descending_triangle", "bearish",
                breakout_price, target_price, stop_price, confidence
            ))

        # Symmetric triangle: converging highs/lows
        elif hi_slope < -0.0003 and lo_slope > 0.0003:
            apex  = (hi_prices[-1] + lo_prices[-1]) / 2
            width = hi_prices[0] - lo_prices[0]
            # Direction determined by prior trend
            prior_trend = (close - window["close"].values[0]) / max(abs(window["close"].values[0]), 1e-8)
            direction  = "bullish" if prior_trend > 0 else "bearish"
            breakout_price = float(hi_prices[-1] if direction == "bullish" else lo_prices[-1])
            target_price   = breakout_price + (width * 0.75 * (1 if direction == "bullish" else -1))
            stop_price     = float(lo_prices[-1] if direction == "bullish" else hi_prices[-1])
            confidence     = 0.55
            results.append(self._make_triangle_record(
                inst_id, date, window, "symmetric_triangle", direction,
                breakout_price, target_price, stop_price, confidence
            ))

        return results

    def _make_triangle_record(
        self, inst_id: int, date, window: pd.DataFrame, sub_type: str,
        direction: str, breakout: float, target: float, stop: float, confidence: float
    ) -> dict:
        return {
            "instrument_id":     inst_id,
            "date":              date,
            "pattern_type":      "triangle",
            "direction":         direction,
            "sub_type":          sub_type,
            "start_date":        window.index[0],
            "bars_in_pattern":   len(window),
            "breakout_price":    round(breakout, 4),
            "target_price":      round(target, 4),
            "stop_price":        round(stop, 4),
            "pattern_confidence": round(confidence, 4),
            "pattern_score":     round(confidence * 100, 2),
        }

    # ── Wedge Pattern ──────────────────────────────────────────────────────────

    def _detect_wedge(self, inst_id: int, date, window: pd.DataFrame) -> list[dict]:
        """
        Rising wedge: both lines slope up, but converging → bearish breakout
        Falling wedge: both lines slope down, converging → bullish breakout
        """
        if len(window) < 20:
            return []

        high  = window["high"].values
        low   = window["low"].values
        x     = np.arange(len(window))

        hi_slope, hi_int = np.polyfit(x, high, 1)
        lo_slope, lo_int = np.polyfit(x, low,  1)

        hi_slope_pct = hi_slope / max(high.mean(), 1e-8)
        lo_slope_pct = lo_slope / max(low.mean(), 1e-8)

        results = []

        # Rising wedge: both slopes up, lower line rising faster → converging
        if hi_slope_pct > 0.001 and lo_slope_pct > hi_slope_pct * 1.1:
            breakout_price = float(np.polyval([lo_slope, lo_int], len(window)))
            height         = high.max() - low.min()
            target_price   = breakout_price - height * 0.7
            stop_price     = float(high[-3:].max())
            confidence     = 0.60
            results.append({
                "instrument_id":     inst_id,
                "date":              date,
                "pattern_type":      "wedge",
                "direction":         "bearish",
                "sub_type":          "rising_wedge",
                "start_date":        window.index[0],
                "bars_in_pattern":   len(window),
                "breakout_price":    round(breakout_price, 4),
                "target_price":      round(target_price, 4),
                "stop_price":        round(stop_price, 4),
                "pattern_confidence": confidence,
                "pattern_score":     round(confidence * 100, 2),
            })

        # Falling wedge: both slopes down, upper line falling faster → converging
        elif hi_slope_pct < -0.001 and lo_slope_pct < hi_slope_pct * 1.1:
            breakout_price = float(np.polyval([hi_slope, hi_int], len(window)))
            height         = high.max() - low.min()
            target_price   = breakout_price + height * 0.7
            stop_price     = float(low[-3:].min())
            confidence     = 0.60
            results.append({
                "instrument_id":     inst_id,
                "date":              date,
                "pattern_type":      "wedge",
                "direction":         "bullish",
                "sub_type":          "falling_wedge",
                "start_date":        window.index[0],
                "bars_in_pattern":   len(window),
                "breakout_price":    round(breakout_price, 4),
                "target_price":      round(target_price, 4),
                "stop_price":        round(stop_price, 4),
                "pattern_confidence": confidence,
                "pattern_score":     round(confidence * 100, 2),
            })

        return results

    # ── Channel Pattern ────────────────────────────────────────────────────────

    def _detect_channel(self, inst_id: int, date, window: pd.DataFrame) -> list[dict]:
        """
        Parallel trendlines (up-channel or down-channel).
        Breakout from channel = signal in breakout direction.
        """
        if len(window) < 20:
            return []

        high  = window["high"].values
        low   = window["low"].values
        close = window["close"].values
        x     = np.arange(len(window))

        hi_slope, hi_int = np.polyfit(x, high, 1)
        lo_slope, lo_int = np.polyfit(x, low,  1)

        slope_diff = abs(hi_slope - lo_slope) / max(abs(hi_slope + lo_slope) / 2, 1e-8)
        if slope_diff > 0.5:  # slopes must be roughly parallel
            return []

        avg_slope    = (hi_slope + lo_slope) / 2
        avg_slope_pct = avg_slope / max(close.mean(), 1e-8)

        channel_height = np.mean(high - low)
        if channel_height <= 0:
            return []

        if avg_slope_pct > 0.0005:
            direction     = "bullish"  # up-channel breakout above → continuation
            sub_type      = "ascending_channel"
            breakout_price = float(np.polyval([hi_slope, hi_int], len(window)))
            target_price   = breakout_price + channel_height
            stop_price     = float(np.polyval([lo_slope, lo_int], len(window)))
        elif avg_slope_pct < -0.0005:
            direction     = "bearish"
            sub_type      = "descending_channel"
            breakout_price = float(np.polyval([lo_slope, lo_int], len(window)))
            target_price   = breakout_price - channel_height
            stop_price     = float(np.polyval([hi_slope, hi_int], len(window)))
        else:
            return []  # flat channel — not a strong setup

        confidence = 0.50 + min(len(window) / 100, 0.20)

        return [{
            "instrument_id":     inst_id,
            "date":              date,
            "pattern_type":      "channel",
            "direction":         direction,
            "sub_type":          sub_type,
            "start_date":        window.index[0],
            "bars_in_pattern":   len(window),
            "breakout_price":    round(breakout_price, 4),
            "target_price":      round(target_price, 4),
            "stop_price":        round(stop_price, 4),
            "pattern_confidence": round(confidence, 4),
            "pattern_score":     round(confidence * 100, 2),
        }]

    # ── Head & Shoulders ───────────────────────────────────────────────────────

    def _detect_head_and_shoulders(self, inst_id: int, date, window: pd.DataFrame) -> list[dict]:
        """
        Head & Shoulders: 3-peak pattern with middle peak highest.
        Inverse H&S: 3-trough pattern with middle trough lowest.
        """
        if len(window) < 30:
            return []

        high  = window["high"].values
        low   = window["low"].values
        close = window["close"].values

        results = []

        # H&S (bearish reversal)
        peaks_idx = argrelextrema(high, np.greater, order=PIVOT_ORDER)[0]
        if len(peaks_idx) >= 3:
            # Take last 3 peaks
            l, h_peak, r = peaks_idx[-3], peaks_idx[-2], peaks_idx[-1]
            lv, hv, rv   = high[l], high[h_peak], high[r]

            # Head must be highest, shoulders roughly equal
            if hv > lv and hv > rv:
                shoulder_symmetry = abs(lv - rv) / max(hv, 1e-8)
                neckline = (low[l:h_peak].min() + low[h_peak:r].min()) / 2
                pattern_height = hv - neckline

                if shoulder_symmetry < 0.15 and pattern_height > 0:
                    confidence = 0.60 + (0.15 - shoulder_symmetry) / 0.15 * 0.20
                    results.append({
                        "instrument_id":     inst_id,
                        "date":              date,
                        "pattern_type":      "head_and_shoulders",
                        "direction":         "bearish",
                        "sub_type":          "head_and_shoulders",
                        "start_date":        window.index[int(l)],
                        "bars_in_pattern":   int(r - l),
                        "breakout_price":    round(float(neckline), 4),
                        "target_price":      round(float(neckline - pattern_height), 4),
                        "stop_price":        round(float(hv * 1.01), 4),
                        "pattern_confidence": round(float(confidence), 4),
                        "pattern_score":     round(float(confidence * 100), 2),
                    })

        # Inverse H&S (bullish reversal)
        troughs_idx = argrelextrema(low, np.less, order=PIVOT_ORDER)[0]
        if len(troughs_idx) >= 3:
            l, h_trough, r = troughs_idx[-3], troughs_idx[-2], troughs_idx[-1]
            lv, hv, rv     = low[l], low[h_trough], low[r]

            if hv < lv and hv < rv:
                shoulder_symmetry = abs(lv - rv) / max(abs(hv), 1e-8)
                neckline = (high[l:h_trough].max() + high[h_trough:r].max()) / 2
                pattern_height = neckline - hv

                if shoulder_symmetry < 0.15 and pattern_height > 0:
                    confidence = 0.60 + (0.15 - shoulder_symmetry) / 0.15 * 0.20
                    results.append({
                        "instrument_id":     inst_id,
                        "date":              date,
                        "pattern_type":      "head_and_shoulders",
                        "direction":         "bullish",
                        "sub_type":          "inverse_head_and_shoulders",
                        "start_date":        window.index[int(l)],
                        "bars_in_pattern":   int(r - l),
                        "breakout_price":    round(float(neckline), 4),
                        "target_price":      round(float(neckline + pattern_height), 4),
                        "stop_price":        round(float(hv * 0.99), 4),
                        "pattern_confidence": round(float(confidence), 4),
                        "pattern_score":     round(float(confidence * 100), 2),
                    })

        return results

    # ── Persistence ────────────────────────────────────────────────────────────

    def _upsert(self, session, records: list[dict]):
        if not records:
            return
        # ChartPattern has no unique constraint on (instrument_id, date, pattern_type)
        # Use insert with on_conflict_do_nothing for simplicity
        for record in records:
            stmt = pg_insert(ChartPattern).values([record])
            stmt = stmt.on_conflict_do_nothing()
            try:
                session.execute(stmt)
            except Exception:
                pass
        session.commit()

    def _row_to_dict(self, row: ChartPattern) -> dict:
        return {
            "date":              str(row.date),
            "pattern_type":      row.pattern_type,
            "direction":         row.direction,
            "sub_type":          row.sub_type,
            "start_date":        str(row.start_date) if row.start_date else None,
            "bars_in_pattern":   row.bars_in_pattern,
            "breakout_price":    row.breakout_price,
            "target_price":      row.target_price,
            "stop_price":        row.stop_price,
            "pattern_confidence": row.pattern_confidence,
            "pattern_score":     row.pattern_score,
        }
