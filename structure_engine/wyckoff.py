"""
RoboAlgo — Wyckoff Market Structure Engine
Identifies Wyckoff phases and structural events using price/volume analysis.

Wyckoff Phases:
  Accumulation (A–E): Smart money absorbing supply, spring/secondary tests
  Markup: Trend initiation after accumulation
  Distribution (A–E): Smart money distributing, upthrust/UTAD
  Markdown: Trend initiation after distribution

Key Events:
  Spring (SC/Spring): False break below support in accumulation, then reversal
  Upthrust (UT/UTAD): False break above resistance in distribution, then reversal
  Secondary Test (ST): Retest of prior support/resistance with lower volume
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import select, desc
from sqlalchemy.dialects.postgresql import insert as pg_insert
from tqdm import tqdm

from database.connection import get_session
from database.models import Instrument, PriceData, Indicator, WyckoffPhase

logger = logging.getLogger(__name__)

# ── Parameters ─────────────────────────────────────────────────────────────────
SWING_LOOKBACK       = 10   # bars to identify swing highs/lows
TREND_LOOKBACK       = 50   # bars for trend analysis
VOLUME_LOOKBACK      = 20   # bars for volume average
PHASE_MIN_BARS       = 15   # minimum bars to classify a phase
SPRING_PENETRATION   = 0.02  # 2% — spring must penetrate support by this
UPTHRUST_PENETRATION = 0.02  # 2% — upthrust must penetrate resistance by this


class WyckoffEngine:
    """
    Identifies Wyckoff phases from price and volume patterns.

    Usage:
        engine = WyckoffEngine()
        engine.compute_and_store()
        phase = engine.get_latest("SOXL")
    """

    # ── Public API ─────────────────────────────────────────────────────────────

    def compute_and_store(self, symbol: Optional[str] = None) -> int:
        """Compute Wyckoff phases for one or all instruments."""
        with get_session() as session:
            if symbol:
                instruments = session.execute(
                    select(Instrument).where(Instrument.symbol == symbol.upper())
                ).scalars().all()
            else:
                instruments = session.execute(select(Instrument)).scalars().all()

            total = 0
            for inst in tqdm(instruments, desc="Wyckoff phases"):
                try:
                    rows = self._process_instrument(session, inst)
                    if rows:
                        self._upsert(session, rows)
                        total += len(rows)
                except Exception as e:
                    logger.warning(f"Wyckoff engine failed for {inst.symbol}: {e}")
            logger.info(f"Wyckoff engine: stored {total} phase rows.")
            return total

    def get_latest(self, symbol: str) -> Optional[dict]:
        """Return the latest Wyckoff phase for a symbol."""
        with get_session() as session:
            instr = session.execute(
                select(Instrument).where(Instrument.symbol == symbol.upper())
            ).scalar_one_or_none()
            if not instr:
                return None
            row = session.execute(
                select(WyckoffPhase)
                .where(WyckoffPhase.instrument_id == instr.id)
                .order_by(desc(WyckoffPhase.date))
                .limit(1)
            ).scalar_one_or_none()
            return self._row_to_dict(row) if row else None

    def phase_score_for_direction(self, symbol: str, direction: str = "up") -> float:
        """
        Return 0–100 score indicating how well the current Wyckoff phase
        supports a trade in the given direction.
        """
        latest = self.get_latest(symbol)
        if not latest:
            return 50.0

        phase = latest.get("phase", "Unknown")
        spring = latest.get("spring_detected", False)
        upthrust = latest.get("upthrust_detected", False)
        confidence = latest.get("confidence", 0.5)

        score = 50.0
        if direction == "up":
            if phase == "Accumulation":
                score = 70.0
                if spring:
                    score += 15  # spring = strong buy setup
                if latest.get("sub_phase") in ("Phase C", "Phase D"):
                    score += 10  # best buy zones in Wyckoff
            elif phase == "Markup":
                score = 80.0
            elif phase == "Distribution":
                score = 25.0
                if upthrust:
                    score -= 10  # upthrust = distribution confirmation
            elif phase == "Markdown":
                score = 15.0
        else:
            if phase == "Distribution":
                score = 70.0
                if upthrust:
                    score += 15
                if latest.get("sub_phase") in ("Phase C", "Phase D"):
                    score += 10
            elif phase == "Markdown":
                score = 80.0
            elif phase == "Accumulation":
                score = 25.0
            elif phase == "Markup":
                score = 15.0

        # Confidence-weight the score
        score = score * confidence + 50.0 * (1 - confidence)
        return float(np.clip(score, 0, 100))

    # ── Internal Processing ────────────────────────────────────────────────────

    def _process_instrument(self, session, instrument) -> list[dict]:
        """Compute Wyckoff phases for one instrument."""
        price_rows = session.execute(
            select(PriceData)
            .where(PriceData.instrument_id == instrument.id)
            .order_by(PriceData.date)
        ).scalars().all()

        if len(price_rows) < TREND_LOOKBACK + PHASE_MIN_BARS:
            return []

        ind_rows = session.execute(
            select(Indicator)
            .where(Indicator.instrument_id == instrument.id)
            .order_by(Indicator.date)
        ).scalars().all()

        prices = pd.DataFrame([{
            "date": r.date, "open": r.open, "high": r.high,
            "low": r.low, "close": r.close, "volume": r.volume,
        } for r in price_rows]).set_index("date")

        indicators = pd.DataFrame([{
            "date": r.date, "rsi": r.rsi, "atr": r.atr,
            "ma50": r.ma50, "ma200": r.ma200,
        } for r in ind_rows]).set_index("date")

        df = prices.join(indicators, how="left").dropna(subset=["close"])
        return self._compute_phases(instrument.id, df)

    def _compute_phases(self, instrument_id: int, df: pd.DataFrame) -> list[dict]:
        """Identify Wyckoff phases across full price history."""
        close  = df["close"]
        high   = df["high"]
        low    = df["low"]
        volume = df["volume"].fillna(0)
        ma50   = df.get("ma50",   pd.Series(np.nan, index=df.index)).ffill()
        ma200  = df.get("ma200",  pd.Series(np.nan, index=df.index)).ffill()

        vol_avg   = volume.rolling(VOLUME_LOOKBACK).mean()
        vol_trend = self._volume_trend(volume, lookback=VOLUME_LOOKBACK)

        # Swing highs/lows for support/resistance
        swing_highs, swing_lows = self._compute_swings(high, low, SWING_LOOKBACK)

        # Rolling support/resistance from recent swings
        support    = swing_lows.rolling(TREND_LOOKBACK).max().ffill()
        resistance = swing_highs.rolling(TREND_LOOKBACK).min().ffill()

        records = []
        dates = df.index.tolist()
        prev_phase = "Unknown"

        for i, date in enumerate(dates):
            if i < TREND_LOOKBACK:
                continue

            cur_close  = float(close.iloc[i])
            cur_high   = float(high.iloc[i])
            cur_low    = float(low.iloc[i])
            cur_vol    = float(volume.iloc[i])
            cur_vol_av = float(vol_avg.iloc[i]) if not pd.isna(vol_avg.iloc[i]) else 1.0
            sup        = float(support.iloc[i])    if not pd.isna(support.iloc[i])    else cur_low
            res        = float(resistance.iloc[i]) if not pd.isna(resistance.iloc[i]) else cur_high
            ma50_val   = float(ma50.iloc[i])  if not pd.isna(ma50.iloc[i])  else cur_close
            ma200_val  = float(ma200.iloc[i]) if not pd.isna(ma200.iloc[i]) else cur_close
            v_trend    = vol_trend.iloc[i]

            # Determine phase
            phase, sub_phase, confidence = self._classify_phase(
                close.iloc[max(0, i-TREND_LOOKBACK):i+1].values,
                volume.iloc[max(0, i-TREND_LOOKBACK):i+1].values,
                ma50_val, ma200_val, sup, res, cur_close
            )

            # Detect spring (false breakdown then recovery)
            spring = self._detect_spring(
                cur_low, cur_close, sup, volume.iloc[max(0, i-5):i+1].values,
                vol_avg.iloc[i] if not pd.isna(vol_avg.iloc[i]) else 1.0, phase
            )

            # Detect upthrust (false breakout then rejection)
            upthrust = self._detect_upthrust(
                cur_high, cur_close, res, volume.iloc[max(0, i-5):i+1].values,
                vol_avg.iloc[i] if not pd.isna(vol_avg.iloc[i]) else 1.0, phase
            )

            # Detect secondary test (retest with lower volume)
            secondary_test = self._detect_secondary_test(
                close.iloc[max(0, i-10):i+1].values,
                low.iloc[max(0, i-10):i+1].values,
                high.iloc[max(0, i-10):i+1].values,
                volume.iloc[max(0, i-10):i+1].values,
                phase, spring, upthrust
            )

            phase_score = self._compute_phase_score(
                phase, sub_phase, spring, upthrust, secondary_test, confidence
            )

            records.append({
                "instrument_id":         instrument_id,
                "date":                  date,
                "phase":                 phase,
                "sub_phase":             sub_phase,
                "spring_detected":       spring,
                "upthrust_detected":     upthrust,
                "secondary_test_detected": secondary_test,
                "support_level":         round(sup, 4),
                "resistance_level":      round(res, 4),
                "volume_trend":          v_trend,
                "confidence":            round(confidence, 4),
                "phase_score":           round(phase_score, 2),
            })

        return records

    def _classify_phase(
        self, closes: np.ndarray, volumes: np.ndarray,
        ma50: float, ma200: float, support: float,
        resistance: float, current_price: float
    ) -> tuple[str, str, float]:
        """Classify the current Wyckoff phase based on price structure."""
        if len(closes) < 10:
            return "Unknown", "Phase A", 0.3

        # Price trend
        trend_50  = current_price / ma50  - 1 if ma50  > 0 else 0
        trend_200 = current_price / ma200 - 1 if ma200 > 0 else 0

        # Price momentum (linear regression slope)
        x = np.arange(min(len(closes), 30))
        y = closes[-len(x):]
        if len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]
            norm_slope = slope / (current_price + 1e-8)  # normalize by price
        else:
            norm_slope = 0

        # Volume trend (is volume confirming price?)
        vol_recent = volumes[-5:].mean() if len(volumes) >= 5 else volumes.mean()
        vol_prior  = volumes[-20:-5].mean() if len(volumes) >= 20 else volumes.mean()
        vol_ratio  = vol_recent / max(vol_prior, 1e-8)

        # Range contraction (trading range = possible accumulation/distribution)
        price_range = closes.max() - closes.min()
        range_pct   = price_range / max(current_price, 1e-8)

        # Price position within range
        range_pos = (current_price - closes.min()) / max(price_range, 1e-8) if price_range > 0 else 0.5

        # Classification logic
        phase = "Unknown"
        sub_phase = "Phase A"
        confidence = 0.5

        if trend_200 > 0.05 and norm_slope > 0:
            # Uptrend zone
            if trend_50 > 0.02:
                phase = "Markup"
                confidence = min(0.5 + abs(norm_slope) * 500, 0.9)
            else:
                phase = "Accumulation"
                sub_phase = "Phase D"
                confidence = 0.6
        elif trend_200 < -0.05 and norm_slope < 0:
            # Downtrend zone
            if trend_50 < -0.02:
                phase = "Markdown"
                confidence = min(0.5 + abs(norm_slope) * 500, 0.9)
            else:
                phase = "Distribution"
                sub_phase = "Phase D"
                confidence = 0.6
        elif range_pct < 0.15:
            # Trading range (compression zone)
            if trend_200 >= 0:
                phase = "Accumulation"
                sub_phase = self._determine_accumulation_subphase(
                    range_pos, vol_ratio, closes
                )
                confidence = 0.65
            else:
                phase = "Distribution"
                sub_phase = self._determine_distribution_subphase(
                    range_pos, vol_ratio, closes
                )
                confidence = 0.65
        else:
            # Transitional — guess based on trend
            if norm_slope > 0:
                phase = "Accumulation"
                confidence = 0.4
            elif norm_slope < 0:
                phase = "Distribution"
                confidence = 0.4

        return phase, sub_phase, confidence

    def _determine_accumulation_subphase(
        self, range_pos: float, vol_ratio: float, closes: np.ndarray
    ) -> str:
        """Determine A/B/C/D/E sub-phase of accumulation."""
        if range_pos < 0.2:
            return "Phase A"  # Stopping action — climactic selling
        elif range_pos < 0.4:
            return "Phase B"  # Building cause — secondary test
        elif range_pos < 0.6:
            return "Phase C"  # Test (spring zone) — most powerful
        elif range_pos < 0.8:
            return "Phase D"  # Mark up begins — last point of support
        else:
            return "Phase E"  # Mark up — leaving range

    def _determine_distribution_subphase(
        self, range_pos: float, vol_ratio: float, closes: np.ndarray
    ) -> str:
        """Determine A/B/C/D/E sub-phase of distribution."""
        if range_pos > 0.8:
            return "Phase A"  # Stopping action — climactic buying
        elif range_pos > 0.6:
            return "Phase B"  # Building cause — secondary tests
        elif range_pos > 0.4:
            return "Phase C"  # Upthrust zone — most powerful
        elif range_pos > 0.2:
            return "Phase D"  # Mark down begins
        else:
            return "Phase E"  # Mark down — leaving range

    def _detect_spring(
        self, cur_low: float, cur_close: float, support: float,
        volumes: np.ndarray, vol_avg: float, phase: str
    ) -> bool:
        """
        Detect spring event: low penetrated support but closed back above.
        Best in Accumulation Phase C.
        """
        if phase not in ("Accumulation", "Unknown"):
            return False
        penetration = (support - cur_low) / max(support, 1e-8)
        closed_above = cur_close > support
        low_vol = len(volumes) > 0 and volumes[-1] < vol_avg * 1.2  # spring on lower volume

        return (
            penetration >= SPRING_PENETRATION and
            penetration < 0.10 and  # not a full breakdown
            closed_above and
            low_vol
        )

    def _detect_upthrust(
        self, cur_high: float, cur_close: float, resistance: float,
        volumes: np.ndarray, vol_avg: float, phase: str
    ) -> bool:
        """
        Detect upthrust: high penetrated resistance but closed back below.
        Best in Distribution Phase C.
        """
        if phase not in ("Distribution", "Unknown"):
            return False
        penetration = (cur_high - resistance) / max(resistance, 1e-8)
        closed_below = cur_close < resistance
        high_vol = len(volumes) > 0 and volumes[-1] > vol_avg * 1.2  # upthrust on higher volume

        return (
            penetration >= UPTHRUST_PENETRATION and
            penetration < 0.10 and
            closed_below and
            high_vol
        )

    def _detect_secondary_test(
        self, closes: np.ndarray, lows: np.ndarray, highs: np.ndarray,
        volumes: np.ndarray, phase: str, spring: bool, upthrust: bool
    ) -> bool:
        """
        Detect secondary test (ST): retest of prior support/resistance
        with diminishing volume — confirms prior spring or upthrust.
        """
        if len(closes) < 5 or not (spring or upthrust):
            return False

        # Volume should be declining on the retest
        vol_recent = volumes[-2:].mean() if len(volumes) >= 2 else volumes[-1]
        vol_prior  = volumes[-5:-2].mean() if len(volumes) >= 5 else volumes.mean()
        vol_declining = vol_recent < vol_prior * 0.85  # 15% lower volume on retest

        # Price should be near prior pivot (within 2%)
        price_range = closes.max() - closes.min()
        near_prior  = price_range / max(closes[-1], 1e-8) < 0.05

        return vol_declining and near_prior

    def _compute_phase_score(
        self, phase: str, sub_phase: str, spring: bool,
        upthrust: bool, secondary_test: bool, confidence: float
    ) -> float:
        """Compute 0–100 Wyckoff phase confluence score."""
        base_scores = {
            "Markup":        80.0,
            "Accumulation":  65.0,
            "Distribution":  40.0,
            "Markdown":      20.0,
            "Unknown":       50.0,
        }
        score = base_scores.get(phase, 50.0)

        # Sub-phase bonus (Phase C is the sweet spot)
        if sub_phase == "Phase C":
            score += 10
        elif sub_phase == "Phase D":
            score += 5

        # Event bonuses
        if spring and phase == "Accumulation":
            score += 10
        if upthrust and phase == "Distribution":
            score -= 15  # bearish signal
        if secondary_test:
            score += 5

        # Confidence weighting
        score = score * confidence + 50.0 * (1 - confidence)
        return float(np.clip(score, 0, 100))

    def _volume_trend(self, volume: pd.Series, lookback: int = 20) -> pd.Series:
        """Classify volume trend as rising/falling/neutral."""
        vol_avg  = volume.rolling(lookback).mean()
        vol_prev = volume.rolling(lookback).mean().shift(lookback // 2)
        ratio    = vol_avg / vol_prev.replace(0, np.nan)
        result   = pd.Series("neutral", index=volume.index)
        result[ratio > 1.1]  = "rising"
        result[ratio < 0.90] = "falling"
        return result

    def _compute_swings(
        self, high: pd.Series, low: pd.Series, lookback: int
    ) -> tuple[pd.Series, pd.Series]:
        """Identify swing highs and lows using rolling window pivots."""
        swing_highs = pd.Series(np.nan, index=high.index)
        swing_lows  = pd.Series(np.nan, index=low.index)

        for i in range(lookback, len(high)):
            window_high = high.iloc[i-lookback:i+1]
            window_low  = low.iloc[i-lookback:i+1]
            mid = lookback // 2
            if high.iloc[i-mid] == window_high.max():
                swing_highs.iloc[i-mid] = high.iloc[i-mid]
            if low.iloc[i-mid] == window_low.min():
                swing_lows.iloc[i-mid] = low.iloc[i-mid]

        return swing_highs.ffill(), swing_lows.ffill()

    def _upsert(self, session, records: list[dict]):
        if not records:
            return
        stmt = pg_insert(WyckoffPhase).values(records)
        stmt = stmt.on_conflict_do_update(
            constraint="uq_wyckoff_inst_date",
            set_={
                "phase":                   stmt.excluded.phase,
                "sub_phase":               stmt.excluded.sub_phase,
                "spring_detected":         stmt.excluded.spring_detected,
                "upthrust_detected":       stmt.excluded.upthrust_detected,
                "secondary_test_detected": stmt.excluded.secondary_test_detected,
                "support_level":           stmt.excluded.support_level,
                "resistance_level":        stmt.excluded.resistance_level,
                "volume_trend":            stmt.excluded.volume_trend,
                "confidence":              stmt.excluded.confidence,
                "phase_score":             stmt.excluded.phase_score,
            }
        )
        session.execute(stmt)
        session.commit()

    def _row_to_dict(self, row: WyckoffPhase) -> dict:
        return {
            "date":                  str(row.date),
            "phase":                 row.phase,
            "sub_phase":             row.sub_phase,
            "spring_detected":       row.spring_detected,
            "upthrust_detected":     row.upthrust_detected,
            "secondary_test_detected": row.secondary_test_detected,
            "support_level":         row.support_level,
            "resistance_level":      row.resistance_level,
            "volume_trend":          row.volume_trend,
            "confidence":            row.confidence,
            "phase_score":           row.phase_score,
        }
