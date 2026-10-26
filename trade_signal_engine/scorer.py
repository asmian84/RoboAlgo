"""
UnifiedScorer — 18-component weighted scorer with hard gates and SMC integration.

Wires ALL existing engines + new SMC engine into one score per symbol.
Each component outputs a 0-1 value (0.5 = neutral, >0.5 = bullish, <0.5 = bearish).
Final score = weighted sum × 100 + bonuses, clamped to 0-100.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ScoredSignal:
    symbol: str
    profile: str              # SCALP | SWING | POSITION
    score: float              # 0-100
    direction: str            # LONG | SHORT | NO_TRADE
    confidence: str           # HIGH | MEDIUM | LOW | NONE
    components: dict[str, float] = field(default_factory=dict)
    component_details: dict[str, Any] = field(default_factory=dict)
    rejection_reason: str = ""
    bonuses: list[str] = field(default_factory=list)


UNIFIED_WEIGHTS = {
    # Direction Agreement (35%)
    "mtf_alignment":       0.15,
    "trend_alignment":     0.10,
    "wave_phase":          0.05,
    "structure_shift":     0.05,
    # Smart Money (20%)
    "fvg_confluence":      0.06,
    "order_block":         0.06,
    "premium_discount":    0.04,
    "liquidity_sweep":     0.04,
    # Institutional Flow (15%)
    "gex_signal":          0.08,
    "dealer_positioning":  0.04,
    "catalyst":            0.03,
    # Structure Quality (15%)
    "pattern_score":       0.05,
    "level_confluence":    0.05,
    "breakout_strength":   0.03,
    "relative_strength":   0.02,
    # Timing (15%)
    "cycle_alignment":     0.05,
    "exhaustion_check":    0.05,
    "vol_compression":     0.05,
}

TRADE_PROFILES = {
    "SCALP": {
        "mtf_model": "DAY_TRADE",
        "bar_tf": "5m",
        "smc_tf": "5m",
        "structure_tf": "15m",
        "stop_atr_mult": 0.75,
        "target_rr": 2.5,
    },
    "SWING": {
        "mtf_model": "SWING_TRADE",
        "bar_tf": "1h",
        "smc_tf": "1h",
        "structure_tf": "4h",
        "stop_atr_mult": 1.5,
        "target_rr": 3.0,
    },
    "POSITION": {
        "mtf_model": "INVESTMENT",
        "bar_tf": "1d",
        "smc_tf": "4h",
        "structure_tf": "1d",
        "stop_atr_mult": 2.0,
        "target_rr": 4.0,
    },
}


class UnifiedScorer:
    """Score a symbol using 18 weighted components + hard gates + boosters."""

    def __init__(self):
        self._weights = UNIFIED_WEIGHTS

    def score(self, sd) -> ScoredSignal:
        """
        Score a SymbolData object.

        Args:
            sd: SymbolData from collector.py (has .symbol, .profile, .bars,
                .options_chain, .current_price)

        Returns:
            ScoredSignal with score 0-100, direction, confidence, and
            component breakdown
        """
        profile_cfg = TRADE_PROFILES.get(sd.profile, TRADE_PROFILES["SWING"])
        components: dict[str, float] = {}
        details: dict[str, Any] = {}
        bonuses: list[str] = []

        # Get the primary analysis DataFrame
        bar_tf = profile_cfg["bar_tf"]
        smc_tf = profile_cfg["smc_tf"]
        primary_df = sd.bars.get(bar_tf)
        if primary_df is None or (
            isinstance(primary_df, pd.DataFrame) and primary_df.empty
        ):
            primary_df = self._best_available_df(sd)

        if primary_df is None or len(primary_df) < 10:
            return ScoredSignal(
                symbol=sd.symbol,
                profile=sd.profile,
                score=0,
                direction="NO_TRADE",
                confidence="NONE",
                rejection_reason="Insufficient price data",
            )

        current_price = sd.current_price or float(primary_df["close"].iloc[-1])

        # Compute ATR
        atr = self._compute_atr(primary_df)

        # ── 1. MTF Alignment (15%) ──────────────────────────────────
        mtf_score = 0.5
        mtf_bias = "NEUTRAL"
        try:
            from mtf_alignment.mtf_alignment_engine import MTFAlignmentEngine

            mtf_engine = MTFAlignmentEngine(
                symbol=sd.symbol, model=profile_cfg["mtf_model"]
            )
            # Build timeframe dict from available bars
            tf_map = {}
            for tf_key, df in sd.bars.items():
                label = self._tf_to_mtf_label(tf_key)
                if label:
                    tf_map[label] = df
            if tf_map:
                mtf_result = mtf_engine.run(tf_map)
                mtf_score = mtf_result.get("alignment_score", 0.5)
                mtf_bias = mtf_result.get("bias_direction", "NEUTRAL")
                details["mtf"] = mtf_result
        except Exception as e:
            logger.debug("MTF engine unavailable: %s", e)
        components["mtf_alignment"] = mtf_score

        # ── MTF PENALTY (soft — no longer a hard kill) ─────────────
        # If MTF is 0 (RANGE/NEUTRAL), fall back to trend direction
        # and apply a penalty later instead of rejecting outright.
        mtf_penalty = 0.0
        if mtf_score < 0.20:
            mtf_penalty = 8.0   # moderate penalty, not a kill
        elif mtf_score < 0.40:
            mtf_penalty = 4.0   # slight penalty

        # Determine target direction: MTF bias → trend fallback
        if mtf_bias in ("LONG", "SHORT"):
            target_dir = "long" if mtf_bias == "LONG" else "short"
        else:
            # Fallback: use simple trend from price vs MA20/MA50
            trend_score = self._score_trend(primary_df)
            target_dir = "long" if trend_score >= 0.50 else "short"

        # ── 2. Trend Alignment (10%) ────────────────────────────────
        components["trend_alignment"] = self._score_trend(primary_df)

        # ── 3. Wave Phase (5%) ──────────────────────────────────────
        wave_score = 0.5
        wave_phase = "TRANSITION"
        wave_direction = "NEUTRAL"
        wave_confidence = 0.0
        try:
            from wave_phase.wave_phase_engine import detect_wave_phase

            wave_result = detect_wave_phase(primary_df, symbol=sd.symbol)
            wave_phase = wave_result.get("wave_phase", "TRANSITION")
            wave_direction = wave_result.get("direction", "NEUTRAL")
            wave_confidence = wave_result.get("confidence", 0.0)
            if wave_phase == "IMPULSE":
                wave_score = 0.75 if wave_direction == "UP" else 0.25
            elif wave_phase == "CORRECTION":
                wave_score = 0.35
            details["wave"] = wave_result
        except Exception as e:
            logger.debug("Wave phase unavailable: %s", e)
        components["wave_phase"] = wave_score

        # ── PENALTY: Correction + Long + High Confidence ────────────
        correction_penalty = 0.0
        if wave_phase == "CORRECTION" and target_dir == "long" and wave_confidence > 0.6:
            correction_penalty = 12.0
            bonuses.append("penalty: correction phase active (-12)")

        # ── 4. SMC Components ──────────────────────────────────────
        smc_result = None
        smc_df = sd.bars.get(smc_tf)
        if smc_df is None or (
            isinstance(smc_df, pd.DataFrame) and smc_df.empty
        ):
            smc_df = primary_df
        try:
            from smc_engine.smc_scorer import score_smc

            if len(smc_df) >= 20:
                smc_result = score_smc(smc_df, current_price=current_price)

                # Structure shift (5%)
                if "CHoCH" in smc_result.latest_shift:
                    if "bullish" in smc_result.latest_shift:
                        components["structure_shift"] = 0.82
                    else:
                        components["structure_shift"] = 0.18
                elif "BOS" in smc_result.latest_shift:
                    if "bullish" in smc_result.latest_shift:
                        components["structure_shift"] = 0.70
                    else:
                        components["structure_shift"] = 0.30
                else:
                    components["structure_shift"] = 0.50

                # FVG confluence (6%)
                components["fvg_confluence"] = (
                    smc_result.smc_score if smc_result.fvg_count > 0 else 0.5
                )

                # Order block (6%)
                components["order_block"] = (
                    smc_result.smc_score if smc_result.ob_count > 0 else 0.5
                )

                # Premium/Discount (4%)
                if smc_result.zone == "DISCOUNT":
                    components["premium_discount"] = 0.75
                elif smc_result.zone == "PREMIUM":
                    components["premium_discount"] = 0.25
                else:
                    components["premium_discount"] = 0.50

                details["smc"] = {
                    "sniper_entry": smc_result.sniper_entry,
                    "entry_zone_top": smc_result.entry_zone_top,
                    "entry_zone_bottom": smc_result.entry_zone_bottom,
                    "bias": smc_result.smc_bias,
                    "in_ote": smc_result.in_ote,
                    "zone": smc_result.zone,
                    "fvg_count": smc_result.fvg_count,
                    "ob_count": smc_result.ob_count,
                    "latest_shift": smc_result.latest_shift,
                }
        except Exception as e:
            logger.debug("SMC engine unavailable: %s", e)

        # Defaults for SMC components if not set
        components.setdefault("structure_shift", 0.50)
        components.setdefault("fvg_confluence", 0.50)
        components.setdefault("order_block", 0.50)
        components.setdefault("premium_discount", 0.50)

        # ── PENALTY: Premium zone for longs / Discount for shorts ──
        zone_penalty = 0.0
        if smc_result:
            if smc_result.zone == "PREMIUM" and target_dir == "long":
                zone_penalty = 8.0
                bonuses.append("penalty: PREMIUM zone for longs (-8)")
            if smc_result.zone == "DISCOUNT" and target_dir == "short":
                zone_penalty = 8.0
                bonuses.append("penalty: DISCOUNT zone for shorts (-8)")

        # ── 5. Liquidity Sweep (4%) ────────────────────────────────
        components["liquidity_sweep"] = self._score_liquidity(primary_df, atr)

        # ── 6. Time Exhaustion (5%) ────────────────────────────────
        exhaustion_detected = False
        exhaustion_dir = "NONE"
        try:
            from time_exhaustion.exhaustion_engine import detect_exhaustion

            exh_result = detect_exhaustion(
                primary_df, timeframe=bar_tf, symbol=sd.symbol
            )
            exhaustion_detected = exh_result.get("exhaustion_detected", False)
            exhaustion_dir = exh_result.get("direction", "NONE")
            exh_strength = exh_result.get("exhaustion_strength", 0.0)
            components["exhaustion_check"] = (
                1.0 - exh_strength if exhaustion_detected else 0.60
            )
            details["exhaustion"] = exh_result
        except Exception as e:
            logger.debug("Exhaustion engine unavailable: %s", e)
            components["exhaustion_check"] = 0.5

        # ── PENALTY: Exhaustion in trade direction ──────────────────
        exhaustion_penalty = 0.0
        if exhaustion_detected:
            if (exhaustion_dir == "UP" and target_dir == "long") or (
                exhaustion_dir == "DOWN" and target_dir == "short"
            ):
                exhaustion_penalty = 10.0
                bonuses.append(f"penalty: exhaustion in {exhaustion_dir} (-10)")

        # ── 7. GEX Signal (8%) ─────────────────────────────────────
        gex_profile = None
        if sd.options_chain:
            try:
                from options_engine.gamma_exposure import GammaExposureEngine

                gex_engine = GammaExposureEngine()
                gex_profile = gex_engine.compute(sd.options_chain)
                gex_score = gex_profile.gex_score / 100.0
                components["gex_signal"] = gex_score
                details["gex"] = {
                    "gamma_regime": gex_profile.gamma_regime,
                    "gex_score": gex_profile.gex_score,
                    "call_wall": gex_profile.call_wall,
                    "put_wall": gex_profile.put_wall,
                    "zero_gamma": gex_profile.zero_gamma,
                }
            except Exception as e:
                logger.debug("GEX engine unavailable: %s", e)
        components.setdefault("gex_signal", 0.50)

        # ── 8. Dealer Positioning (4%) ─────────────────────────────
        if gex_profile:
            try:
                from options_engine.dealer_positioning import (
                    DealerPositioningEngine,
                )

                dp_engine = DealerPositioningEngine()
                dp_result = dp_engine.compute(gex_profile, atr=atr)
                dp_score = dp_result.positioning_score / 100.0
                components["dealer_positioning"] = dp_score
                details["dealer"] = {
                    "regime": dp_result.dealer_regime,
                    "squeeze_risk": dp_result.squeeze_risk,
                    "flow_bias": dp_result.flow_bias,
                }
            except Exception as e:
                logger.debug("Dealer positioning unavailable: %s", e)
        components.setdefault("dealer_positioning", 0.50)

        # ── 9. Catalyst (3%) ───────────────────────────────────────
        try:
            from context_engine.catalyst_engine import CatalystEngine

            cat_engine = CatalystEngine()
            cat_result = cat_engine.score_symbol(sd.symbol)
            cat_score_raw = cat_result.get("catalyst_score", 0.0)
            cat_sentiment = cat_result.get("sentiment", "neutral")
            if cat_sentiment == "bullish":
                components["catalyst"] = 0.5 + cat_score_raw * 0.4
            elif cat_sentiment == "bearish":
                components["catalyst"] = 0.5 - cat_score_raw * 0.4
            else:
                components["catalyst"] = 0.50
            details["catalyst"] = cat_result
        except Exception as e:
            logger.debug("Catalyst engine unavailable: %s", e)
            components.setdefault("catalyst", 0.50)

        # ── 10. Pattern Score (5%) ─────────────────────────────────
        components.setdefault("pattern_score", 0.50)  # TODO: wire PatternService

        # ── 11. Level Confluence (5%) ──────────────────────────────
        try:
            from signal_engine.level_engine import LevelEngine

            level_engine = LevelEngine()
            level_result = level_engine.compute(sd.symbol)
            gems = level_result.get("gems", [])
            pounce = level_result.get("pounce", [])
            if pounce:
                components["level_confluence"] = 0.85
            elif gems:
                components["level_confluence"] = 0.70
            else:
                components["level_confluence"] = 0.50
            details["levels"] = {
                "gem_count": len(gems),
                "pounce_count": len(pounce),
                "pounce_ideas": [
                    g.get("trade_idea") for g in pounce if g.get("trade_idea")
                ],
            }
        except Exception as e:
            logger.debug("Level engine unavailable: %s", e)
            components.setdefault("level_confluence", 0.50)

        # ── 12. Breakout Strength (3%) ─────────────────────────────
        components["breakout_strength"] = self._score_breakout(primary_df, atr)

        # ── 13. Relative Strength vs SPY (2%) ──────────────────────
        components["relative_strength"] = self._score_relative_strength(
            sd.symbol, primary_df
        )

        # ── 14. Cycle Alignment (5%) ───────────────────────────────
        components.setdefault(
            "cycle_alignment", 0.50
        )  # TODO: wire FFT/wavelet

        # ── 15. Vol Compression (5%) ───────────────────────────────
        components["vol_compression"] = self._score_vol_compression(primary_df)

        # ── Compute weighted score ─────────────────────────────────
        weighted_sum = sum(
            components.get(k, 0.5) * w for k, w in self._weights.items()
        )
        raw_score = weighted_sum * 100  # 0-100 scale

        # ── PENALTY: No structure confirmation (very soft) ─────────
        structure_val = components.get("structure_shift", 0.50)
        if abs(structure_val - 0.50) < 0.05:  # neutral = no CHoCH/BOS
            raw_score *= 0.92  # reduce by 8% (was 30%)
            bonuses.append("penalty: no structure confirmation (-8%)")

        # ── SCORE BOOSTERS ─────────────────────────────────────────
        bonus = 0
        if smc_result:
            if smc_result.fvg_count > 0 and smc_result.ob_count > 0:
                bonus += 15
                bonuses.append("FVG+OB overlap (+15)")
            if (
                "CHoCH" in smc_result.latest_shift
                and components.get("liquidity_sweep", 0.5) > 0.65
            ):
                bonus += 10
                bonuses.append("liquidity sweep + CHoCH (+10)")
            if smc_result.in_ote:
                bonus += 8
                bonuses.append("in OTE zone (+8)")

        # ── Apply soft penalties (replaces hard gates) ─────────────
        total_penalty = mtf_penalty + correction_penalty + zone_penalty + exhaustion_penalty
        final_score = min(100, max(0, raw_score + bonus - total_penalty))

        # ── Determine direction and confidence ─────────────────────
        # Aggressive thresholds: surface every trade the market offers
        if final_score >= 55:
            direction = "LONG" if target_dir == "long" else "SHORT"
            confidence = "HIGH" if final_score >= 80 else "MEDIUM" if final_score >= 65 else "LOW"
        elif final_score >= 35:
            direction = "LONG" if target_dir == "long" else "SHORT"
            confidence = "LOW"
        else:
            direction = "NO_TRADE"
            confidence = "NONE"

        return ScoredSignal(
            symbol=sd.symbol,
            profile=sd.profile,
            score=round(final_score, 1),
            direction=direction,
            confidence=confidence,
            components=components,
            component_details=details,
            bonuses=bonuses,
        )

    # ── Helper methods ─────────────────────────────────────────────

    def _best_available_df(self, sd) -> Optional[pd.DataFrame]:
        """Get the best available DataFrame from SymbolData."""
        for tf in ["1h", "5m", "15m", "4h", "1d", "1wk"]:
            if tf in sd.bars and not sd.bars[tf].empty:
                return sd.bars[tf]
        return None

    def _compute_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Compute ATR-14."""
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        if len(df) < 2:
            return 1.0
        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1]),
            ),
        )
        atr = (
            float(np.mean(tr[-period:]))
            if len(tr) >= period
            else float(np.mean(tr))
        )
        return max(atr, 0.01)

    def _tf_to_mtf_label(self, tf: str) -> Optional[str]:
        """Map yfinance timeframe strings to MTF engine labels."""
        mapping = {
            "5m": "5m",
            "15m": "15m",
            "1h": "1h",
            "4h": "4h",
            "1d": "daily",
            "1wk": "weekly",
        }
        return mapping.get(tf)

    def _score_trend(self, df: pd.DataFrame) -> float:
        """Score trend alignment from MA20/50."""
        if len(df) < 50:
            return 0.5
        closes = df["close"].values
        ma20 = np.mean(closes[-20:])
        ma50 = np.mean(closes[-50:])
        price = closes[-1]

        score = 0.5
        if price > ma50:
            score += 0.15
        if price > ma20:
            score += 0.10
        if ma20 > ma50:
            score += 0.10
        if price < ma50:
            score -= 0.15
        if price < ma20:
            score -= 0.10
        if ma20 < ma50:
            score -= 0.10

        return max(0.0, min(1.0, score))

    def _score_liquidity(self, df: pd.DataFrame, atr: float) -> float:
        """Score liquidity sweep (stop hunt) detection."""
        if len(df) < 20:
            return 0.5
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values

        # Check if recent bar swept below prior low then closed above
        recent_low = min(lows[-20:-1])
        if lows[-1] < recent_low and closes[-1] > recent_low:
            sweep_size = (recent_low - lows[-1]) / atr
            return min(0.90, 0.65 + sweep_size * 0.15)

        # Check sweep above prior high then closed below
        recent_high = max(highs[-20:-1])
        if highs[-1] > recent_high and closes[-1] < recent_high:
            sweep_size = (highs[-1] - recent_high) / atr
            return min(0.90, 0.65 + sweep_size * 0.15)

        return 0.50

    def _score_breakout(self, df: pd.DataFrame, atr: float) -> float:
        """Score breakout strength."""
        if len(df) < 20:
            return 0.5
        closes = df["close"].values
        volumes = (
            df["volume"].values if "volume" in df.columns else np.ones(len(df))
        )

        high_20 = max(df["high"].values[-20:])
        low_20 = min(df["low"].values[-20:])
        price = closes[-1]

        # Breakout above 20-bar high
        if price >= high_20:
            vol_ratio = (
                volumes[-1] / np.mean(volumes[-20:])
                if np.mean(volumes[-20:]) > 0
                else 1
            )
            return min(1.0, 0.60 + vol_ratio * 0.10)

        # Breakdown below 20-bar low
        if price <= low_20:
            vol_ratio = (
                volumes[-1] / np.mean(volumes[-20:])
                if np.mean(volumes[-20:]) > 0
                else 1
            )
            return max(0.0, 0.40 - vol_ratio * 0.10)

        return 0.50

    def _score_relative_strength(
        self, symbol: str, df: pd.DataFrame
    ) -> float:
        """Score relative strength vs SPY."""
        if symbol in ("SPY", "SPX") or len(df) < 20:
            return 0.50
        # Simple: compare % change over 20 bars
        pct_change = (df["close"].iloc[-1] / df["close"].iloc[-20] - 1) * 100
        if pct_change > 5:
            return 0.75
        elif pct_change > 0:
            return 0.60
        elif pct_change > -5:
            return 0.40
        else:
            return 0.25

    def _score_vol_compression(self, df: pd.DataFrame) -> float:
        """Score volatility compression (BB squeeze)."""
        if len(df) < 20:
            return 0.5
        closes = df["close"].values
        ma20 = np.mean(closes[-20:])
        std20 = np.std(closes[-20:])
        bb_width = (std20 * 2) / ma20 if ma20 > 0 else 0

        # Also check ATR compression
        atr_now = self._compute_atr(df.iloc[-20:])
        atr_prior = (
            self._compute_atr(df.iloc[-40:-20]) if len(df) >= 40 else atr_now
        )
        atr_ratio = atr_now / atr_prior if atr_prior > 0 else 1.0

        # Low BB width + declining ATR = coiled spring
        if bb_width < 0.03 and atr_ratio < 0.8:
            return 0.85  # highly compressed
        elif bb_width < 0.05:
            return 0.70
        elif bb_width > 0.10:
            return 0.30  # already expanded
        return 0.50
