"""
RoboAlgo — Confluence Engine
The central decision layer. Aggregates all analysis engines into a single score
with a full human-readable decision trace.

Weight Table:
  volatility compression  = 25%
  breakout strength       = 20%
  trend alignment         = 15%
  liquidity structure     = 15%
  patterns / harmonics    = 10%
  Wyckoff phase           = 10%
  Gann projections        =  5%

Signal Tiers:
  confluence ≥ 80  → HIGH   (executable signal)
  60–79           → MEDIUM  (executable signal)
  50–59           → WATCH   (monitor, no trade)
  < 50            → NONE    (no action)

Decision Trace Example:
  TRADE SIGNAL — SOXL
  Volatility Regime: HIGH_VOL
  Compression: BB width percentile = 12%
  Range: High = 52.30  Low = 49.80
  Breakout: Price broke above range high  Volume = 1.9× average
  Pattern: Bull flag
  Wyckoff Phase: Accumulation → Markup
  Expected Move: +14%
  Confluence Score: 87%
"""

import json
import logging
from datetime import date as DateType
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import select, desc
from sqlalchemy.dialects.postgresql import insert as pg_insert
from tqdm import tqdm

from database.connection import get_session
from database.models import (
    Instrument, Indicator, PriceData, Signal,
    VolatilityRegime, RangeCompression, BreakoutSignal,
    LiquidityLevel, WyckoffPhase, HarmonicPattern, ChartPattern,
    ConfluenceScore, PatternDetection,
)
from config.settings import VOL_LOW, VOL_NORMAL, VOL_HIGH
from range_engine.compression import RangeCompressionEngine
from range_engine.breakout import BreakoutEngine
from projection_engine.expected_move import ExpectedMoveEngine
from structure_engine.liquidity import LiquidityEngine
from structure_engine.wyckoff import WyckoffEngine
from pattern_engine.detection import ChartPatternEngine
from pattern_engine.harmonics import HarmonicEngine
from projection_engine.gann import GannEngine

logger = logging.getLogger(__name__)

# ── Weights (10 components — extended from original 7) ─────────────────────────
WEIGHTS = {
    "vol_compression":      0.20,
    "breakout":             0.15,
    "trend":                0.12,
    "liquidity":            0.10,
    "pattern":              0.08,
    "wyckoff":              0.08,
    "gann":                 0.04,
    "cycle_alignment":      0.10,   # FFT/wavelet/Hilbert cycle phase alignment
    "price_time_symmetry":  0.08,   # Gann geometry price-time balance
    "harmonic_confluence":  0.05,   # Advanced harmonic pattern zone proximity
}

# Signal tier thresholds
TIER_HIGH   = 80.0
TIER_MEDIUM = 60.0
TIER_WATCH  = 50.0


class ConfluenceEngine:
    """
    The master signal orchestrator.
    Pulls data from all analysis engines and produces a confluence-scored signal
    with full decision trace.

    Usage:
        engine = ConfluenceEngine()
        result = engine.score_symbol("SOXL")   # single symbol, live
        engine.compute_and_store()             # all instruments, batch
    """

    def __init__(self):
        self._compression_engine  = RangeCompressionEngine()
        self._breakout_engine     = BreakoutEngine()
        self._expected_move       = ExpectedMoveEngine()
        self._liquidity_engine    = LiquidityEngine()
        self._wyckoff_engine      = WyckoffEngine()
        self._chart_pattern       = ChartPatternEngine()
        self._harmonic_engine     = HarmonicEngine()
        self._gann_engine         = GannEngine()

    # ── Public API ─────────────────────────────────────────────────────────────

    def score_symbol(self, symbol: str) -> dict:
        """
        Compute full confluence score for a single symbol in real time.
        Uses latest data from all engines. For symbols not in the instrument
        database, falls back to score_symbol_live() which fetches from yfinance.
        Returns complete signal dict with decision trace.
        """
        sym = symbol.upper()
        with get_session() as session:
            instr = session.execute(
                select(Instrument).where(Instrument.symbol == sym)
            ).scalar_one_or_none()
            if not instr:
                # Not in pipeline — compute live from yfinance
                return self.score_symbol_live(sym)

            # Load all components
            price   = self._latest_price(session, instr.id)
            ind     = self._latest_indicator(session, instr.id)
            vol     = self._latest_vol_regime(session, instr.id)
            comp    = self._latest_compression(session, instr.id)
            bo      = self._latest_breakout(session, instr.id)
            liq     = self._latest_liquidity(session, instr.id)
            wyck    = self._latest_wyckoff(session, instr.id)
            chart_p    = self._latest_chart_patterns(session, instr.id)
            harm_p     = self._latest_harmonic_patterns(session, instr.id)
            pat_detect = self._latest_pattern_detections(session, instr.id)
            signal     = self._latest_signal(session, instr.id)

        return self._compute_confluence(
            sym, instr, price, ind, vol, comp, bo, liq, wyck,
            chart_p, harm_p, pat_detect, signal
        )

    def score_symbol_live(self, symbol: str) -> dict:
        """
        Compute a full confluence score for any symbol via yfinance.
        Used automatically when the symbol is not in the instrument database.
        Fetches 2 years of daily OHLCV, computes indicators in-memory, and
        runs the same scoring pipeline as the DB-backed path.

        Components with limited data (Wyckoff, Liquidity, Harmonics) default
        to 50/100 neutral. Cycle, Gann, and Price-Time Symmetry are computed
        live from the fetched price series.  The result is flagged live_computed=True.
        """
        import types
        import yfinance as yf
        from api.routers.prices import _compute_indicators

        sym = symbol.upper()

        # ── 1. Fetch OHLCV ────────────────────────────────────────────────────
        try:
            hist = yf.Ticker(sym).history(period="2y", interval="1d", auto_adjust=True)
            if hist is None or hist.empty or len(hist) < 50:
                return {"error": f"No price data available for {sym}"}
            hist.index   = hist.index.tz_localize(None) if hist.index.tzinfo else hist.index
            hist.columns = [c.lower() for c in hist.columns]
        except Exception as exc:
            logger.warning(f"Live confluence fetch failed for {sym}: {exc}")
            return {"error": f"Could not fetch price data for {sym}"}

        # ── 2. Compute indicators in-memory ───────────────────────────────────
        ind_df = _compute_indicators(hist)
        row    = ind_df.iloc[-1]
        last   = hist.iloc[-1]

        def _f(val, default=0.0) -> float:
            """Safe float; returns default for NaN/None."""
            try:
                v = float(val)
                return default if v != v else v   # NaN check
            except Exception:
                return default

        cur_price = _f(last["close"], 1.0)
        cur_atr   = _f(row["atr"], cur_price * 0.02) or cur_price * 0.02

        # ── 3. Indicator adapter ──────────────────────────────────────────────
        ind = types.SimpleNamespace(
            atr            = cur_atr,
            rsi            = _f(row["rsi"],            50.0),
            macd_histogram = _f(row["macd_histogram"],  0.0),
            ma50           = _f(row["ma50"],           cur_price),
            ma200          = _f(row["ma200"],          cur_price),
            bb_width       = _f(row["bb_width"],        0.05),
            bb_upper       = _f(row["bb_upper"],       cur_price * 1.05),
            bb_lower       = _f(row["bb_lower"],       cur_price * 0.95),
        )

        # ── 4. Volatility adapter ─────────────────────────────────────────────
        bb_s   = ind_df["bb_width"].dropna()
        atr_s  = ind_df["atr"].dropna()
        bb_pct = float(bb_s.rank(pct=True).iloc[-1])  if len(bb_s)  >= 20 else 0.5
        atr_pct= float(atr_s.rank(pct=True).iloc[-1]) if len(atr_s) >= 20 else 0.5
        is_comp= bb_pct < 0.25
        is_exp = (len(bb_s) >= 5 and _f(bb_s.iloc[-1]) > _f(bb_s.iloc[-5]) * 1.15)

        vol = types.SimpleNamespace(
            regime                = VOL_NORMAL,
            bb_width_percentile   = bb_pct,
            atr_percentile        = atr_pct,
            is_compression        = is_comp,
            is_expansion          = is_exp,
            compression_range_high= None,
            compression_range_low = None,
        )

        # ── 5. Compression adapter ────────────────────────────────────────────
        below_mask    = bb_s < bb_s.quantile(0.25) if len(bb_s) > 0 else pd.Series(dtype=bool)
        comp_duration = int(below_mask[::-1].cumprod().sum()) if (is_comp and len(below_mask) > 0) else 0
        recent        = hist.tail(max(comp_duration, 5))
        comp_hi       = float(recent["high"].max())
        comp_lo       = float(recent["low"].min())

        comp = types.SimpleNamespace(
            is_compressed       = is_comp,
            compression_duration= comp_duration,
            compression_range_low= comp_lo,
            range_high          = comp_hi,
            range_low           = comp_lo,
            range_mid           = (comp_hi + comp_lo) / 2,
        )

        # ── 6. Breakout adapter ───────────────────────────────────────────────
        high_20    = float(hist["high"].rolling(20).max().iloc[-2])
        vol_avg20  = float(hist["volume"].rolling(20).mean().iloc[-1]) or 1
        vol_now    = float(hist["volume"].iloc[-1])
        vol_ratio  = vol_now / vol_avg20

        price_break    = cur_price > high_20
        vol_break      = vol_ratio > 1.2
        momentum_break = _f(row["macd_histogram"]) > 0 and _f(row["rsi"]) > 50
        triggers_met   = int(price_break) + int(vol_break) + int(momentum_break)

        if triggers_met >= 2:
            bo_strength = float(np.clip(
                50 + max(cur_price - high_20, 0) / cur_atr * 20 + (vol_ratio - 1) * 15,
                0, 100
            ))
        elif triggers_met == 1:
            bo_strength = 35.0
        else:
            bo_strength = 20.0

        bo = types.SimpleNamespace(
            triggers_met          = triggers_met,
            breakout_strength     = bo_strength,
            breakout_direction    = "up" if price_break else "none",
            volume_ratio          = round(vol_ratio, 2),
            price_trigger         = price_break,
            volume_trigger        = vol_break,
            momentum_trigger      = momentum_break,
            compression_range_low = comp_lo,
        )

        # ── 7. Component scores ───────────────────────────────────────────────
        vol_comp_score            = self._score_vol_compression(vol, comp)
        breakout_score            = self._score_breakout(bo)
        trend_score               = self._score_trend(ind, cur_price)
        liq_score                 = 50.0   # no liquidity engine data for live symbols
        pattern_score             = 50.0   # no pattern DB data
        wyckoff_score             = 50.0   # no Wyckoff data
        gann_score                = self._gann_engine.compute_gann_score(sym)
        cycle_alignment_score     = self._score_cycle_alignment(sym, cur_price)
        price_time_symmetry_score = self._score_price_time_symmetry(sym)
        harmonic_confluence_score = 40.0   # no harmonic PRZ data

        # ── 8. Weighted total ─────────────────────────────────────────────────
        confluence = (
            WEIGHTS["vol_compression"]      * vol_comp_score
            + WEIGHTS["breakout"]           * breakout_score
            + WEIGHTS["trend"]              * trend_score
            + WEIGHTS["liquidity"]          * liq_score
            + WEIGHTS["pattern"]            * pattern_score
            + WEIGHTS["wyckoff"]            * wyckoff_score
            + WEIGHTS["gann"]               * gann_score
            + WEIGHTS["cycle_alignment"]    * cycle_alignment_score
            + WEIGHTS["price_time_symmetry"]* price_time_symmetry_score
            + WEIGHTS["harmonic_confluence"]* harmonic_confluence_score
        )
        confluence = round(float(np.clip(confluence, 0, 100)), 2)

        if confluence >= TIER_HIGH:
            signal_tier = "HIGH"
        elif confluence >= TIER_MEDIUM:
            signal_tier = "MEDIUM"
        elif confluence >= TIER_WATCH:
            signal_tier = "WATCH"
        else:
            signal_tier = "NONE"

        # ── 9. Expected move + trade plan ─────────────────────────────────────
        em   = self._expected_move.calculate(
            symbol=sym, atr=cur_atr, entry_price=cur_price,
            instrument_type=None, compression_duration=comp_duration,
        )
        plan = self._build_trade_plan(cur_price, cur_atr, em, bo, comp)

        # ── 10. Decision trace ────────────────────────────────────────────────
        trace = self._build_decision_trace(
            symbol=sym,
            vol_regime=VOL_NORMAL,
            vol_comp_score=vol_comp_score,
            breakout_score=breakout_score,
            trend_score=trend_score,
            liq_score=liq_score,
            pattern_score=pattern_score,
            wyckoff_score=wyckoff_score,
            gann_score=gann_score,
            cycle_alignment_score=cycle_alignment_score,
            price_time_symmetry_score=price_time_symmetry_score,
            harmonic_confluence_score=harmonic_confluence_score,
            confluence=confluence,
            signal_tier=signal_tier,
            vol=vol, comp=comp, bo=bo,
            liq=None, wyck=None,
            chart_patterns=[], harmonic_patterns=[],
            pattern_detections=[],
            em=em, plan=plan, cur_price=cur_price,
        )

        return {
            "symbol":            sym,
            "date":              str(hist.index[-1].date()),
            "confluence_score":  confluence,
            "signal_tier":       signal_tier,
            "volatility_regime": VOL_NORMAL,
            "is_compression":    is_comp,
            "is_breakout":       triggers_met >= 2,
            "expected_move_pct": em.get("expected_move_pct", 0),
            "entry_price":       cur_price,
            "target_price":      plan["target"],
            "stop_price":        plan["stop"],
            "component_scores": {
                "vol_compression":       round(vol_comp_score, 2),
                "breakout":              round(breakout_score, 2),
                "trend":                 round(trend_score, 2),
                "liquidity":             round(liq_score, 2),
                "pattern":               round(pattern_score, 2),
                "wyckoff":               round(wyckoff_score, 2),
                "gann":                  round(gann_score, 2),
                "cycle_alignment":       round(cycle_alignment_score, 2),
                "price_time_symmetry":   round(price_time_symmetry_score, 2),
                "harmonic_confluence":   round(harmonic_confluence_score, 2),
            },
            "decision_trace":    trace,
            "live_computed":     True,   # fetched from yfinance, not pipeline DB
        }

    def compute_and_store(self, symbol: Optional[str] = None) -> int:
        """Batch compute and store confluence scores for all/one instrument."""
        with get_session() as session:
            if symbol:
                instruments = session.execute(
                    select(Instrument).where(Instrument.symbol == symbol.upper())
                ).scalars().all()
            else:
                instruments = session.execute(select(Instrument)).scalars().all()

            total = 0
            for inst in tqdm(instruments, desc="Confluence scoring"):
                try:
                    price   = self._latest_price(session, inst.id)
                    ind     = self._latest_indicator(session, inst.id)
                    vol     = self._latest_vol_regime(session, inst.id)
                    comp    = self._latest_compression(session, inst.id)
                    bo      = self._latest_breakout(session, inst.id)
                    liq     = self._latest_liquidity(session, inst.id)
                    wyck    = self._latest_wyckoff(session, inst.id)
                    chart_p    = self._latest_chart_patterns(session, inst.id)
                    harm_p     = self._latest_harmonic_patterns(session, inst.id)
                    pat_detect = self._latest_pattern_detections(session, inst.id)
                    signal     = self._latest_signal(session, inst.id)

                    if not price or not ind:
                        continue

                    result = self._compute_confluence(
                        inst.symbol, inst, price, ind, vol, comp, bo, liq, wyck,
                        chart_p, harm_p, pat_detect, signal
                    )

                    if "error" not in result:
                        self._upsert(session, inst.id, result)
                        total += 1
                except Exception as e:
                    logger.warning(f"Confluence failed for {inst.symbol}: {e}")

            logger.info(f"Confluence engine: scored {total} instruments.")
            return total

    def get_top_signals(
        self, min_tier: str = "MEDIUM", limit: int = 20
    ) -> list[dict]:
        """Return top confluence signals above a given tier."""
        tier_map = {"HIGH": TIER_HIGH, "MEDIUM": TIER_MEDIUM, "WATCH": TIER_WATCH}
        min_score = tier_map.get(min_tier.upper(), TIER_MEDIUM)

        with get_session() as session:
            rows = session.execute(
                select(ConfluenceScore, Instrument.symbol)
                .join(Instrument, Instrument.id == ConfluenceScore.instrument_id)
                .where(ConfluenceScore.confluence_score >= min_score)
                .order_by(desc(ConfluenceScore.date), desc(ConfluenceScore.confluence_score))
            ).all()

            seen = set()
            results = []
            for cs, sym in rows:
                if sym not in seen:
                    seen.add(sym)
                    d = self._cs_to_dict(cs)
                    d["symbol"] = sym
                    results.append(d)
                if len(results) >= limit:
                    break
            return results

    # ── Core Computation ───────────────────────────────────────────────────────

    def _compute_confluence(
        self, symbol: str, instr, price, ind, vol, comp, bo, liq, wyck,
        chart_patterns: list, harmonic_patterns: list,
        pattern_detections: list, signal,
    ) -> dict:
        """
        Compute weighted confluence score from all engine inputs.
        Returns full signal dict with decision trace.
        """
        if not price or not ind:
            return {"error": "Missing price/indicator data"}

        cur_price    = float(price.close or 0)
        cur_atr      = float(ind.atr or 0)
        vol_regime   = vol.regime    if vol  else VOL_NORMAL
        is_comp      = bool(vol.is_compression) if vol else False
        is_expansion = bool(vol.is_expansion)   if vol else False

        # Gate: LOW_VOL → suppress signal entirely
        if vol_regime == VOL_LOW and not is_expansion:
            return self._build_gated_result(symbol, cur_price, vol_regime, price.date)

        # ── Component Scores (each 0–100) ──────────────────────────────────────

        # 1. Volatility Compression Score (25%)
        vol_comp_score = self._score_vol_compression(vol, comp)

        # 2. Breakout Score (20%)
        breakout_score = self._score_breakout(bo)

        # 3. Trend Alignment Score (15%)
        trend_score = self._score_trend(ind, cur_price)

        # 4. Liquidity Score (15%)
        liq_score = float(liq.liquidity_score) if liq else 50.0

        # 5. Pattern Score (8%) — blended from 3 sources:
        #    • ChartPattern table        (35%)
        #    • HarmonicPattern table     (20%)
        #    • PatternDetection table    (45%) — 35+ detectors, highest quality
        chart_score    = max((p.pattern_score or 0) for p in chart_patterns)    if chart_patterns    else 50.0
        harmonic_score = max((p.pattern_score or 0) for p in harmonic_patterns) if harmonic_patterns else 50.0
        detect_score   = self._score_pattern_detections(pattern_detections)
        pattern_score  = (chart_score * 0.35 + harmonic_score * 0.20 + detect_score * 0.45)

        # 6. Wyckoff Score (10%)
        wyckoff_score = self._score_wyckoff(wyck, is_comp)

        # 7. Gann Score (4%)
        gann_score = self._gann_engine.compute_gann_score(symbol)

        # 8. Cycle Alignment Score (10%) — FFT/wavelet consensus
        cycle_alignment_score = self._score_cycle_alignment(symbol, cur_price)

        # 9. Price-Time Symmetry Score (8%) — Gann geometry balance
        price_time_symmetry_score = self._score_price_time_symmetry(symbol)

        # 10. Harmonic Confluence Score (5%) — proximity to PRZ zones
        harmonic_confluence_score = self._score_harmonic_confluence(
            harmonic_patterns, cur_price, cur_atr
        )

        # ── Weighted Total (10 components) ────────────────────────────────────
        confluence = (
            WEIGHTS["vol_compression"]     * vol_comp_score
            + WEIGHTS["breakout"]          * breakout_score
            + WEIGHTS["trend"]             * trend_score
            + WEIGHTS["liquidity"]         * liq_score
            + WEIGHTS["pattern"]           * pattern_score
            + WEIGHTS["wyckoff"]           * wyckoff_score
            + WEIGHTS["gann"]              * gann_score
            + WEIGHTS["cycle_alignment"]   * cycle_alignment_score
            + WEIGHTS["price_time_symmetry"] * price_time_symmetry_score
            + WEIGHTS["harmonic_confluence"] * harmonic_confluence_score
        )
        confluence = round(float(np.clip(confluence, 0, 100)), 2)

        # ── Signal Tier ────────────────────────────────────────────────────────
        if confluence >= TIER_HIGH:
            signal_tier = "HIGH"
        elif confluence >= TIER_MEDIUM:
            signal_tier = "MEDIUM"
        elif confluence >= TIER_WATCH:
            signal_tier = "WATCH"
        else:
            signal_tier = "NONE"

        # ── Expected Move ─────────────────────────────────────────────────────
        comp_duration = int(comp.compression_duration) if comp else 0
        em = self._expected_move.calculate(
            symbol=symbol,
            atr=cur_atr,
            entry_price=cur_price,
            instrument_type=instr.instrument_type if instr else None,
            compression_duration=comp_duration,
        )

        # ── Trade Plan ────────────────────────────────────────────────────────
        plan = self._build_trade_plan(cur_price, cur_atr, em, bo, comp)

        # ── Decision Trace ────────────────────────────────────────────────────
        trace = self._build_decision_trace(
            symbol=symbol,
            vol_regime=vol_regime,
            vol_comp_score=vol_comp_score,
            breakout_score=breakout_score,
            trend_score=trend_score,
            liq_score=liq_score,
            pattern_score=pattern_score,
            wyckoff_score=wyckoff_score,
            gann_score=gann_score,
            cycle_alignment_score=cycle_alignment_score,
            price_time_symmetry_score=price_time_symmetry_score,
            harmonic_confluence_score=harmonic_confluence_score,
            confluence=confluence,
            signal_tier=signal_tier,
            vol=vol, comp=comp, bo=bo, liq=liq, wyck=wyck,
            chart_patterns=chart_patterns,
            harmonic_patterns=harmonic_patterns,
            pattern_detections=pattern_detections,
            em=em, plan=plan, cur_price=cur_price,
        )

        return {
            "symbol":               symbol,
            "date":                 str(price.date),
            "confluence_score":     confluence,
            "signal_tier":          signal_tier,
            "volatility_regime":    vol_regime,
            "is_compression":       is_comp,
            "is_breakout":          bool(bo and bo.triggers_met and bo.triggers_met >= 2),
            "expected_move_pct":    em.get("expected_move_pct", 0),
            "expected_move_display": em.get("expected_move_pct_display", "N/A"),
            "passes_move_filter":   em.get("passes_filter", False),
            "entry_price":          plan["entry"],
            "add_price":            plan["add"],
            "scale_price":          plan["scale"],
            "target_price":         plan["target"],
            "stop_price":           plan["stop"],
            "component_scores": {
                "vol_compression":     round(vol_comp_score, 2),
                "breakout":            round(breakout_score, 2),
                "trend":               round(trend_score, 2),
                "liquidity":           round(liq_score, 2),
                "pattern":             round(pattern_score, 2),
                "wyckoff":             round(wyckoff_score, 2),
                "gann":                round(gann_score, 2),
                "cycle_alignment":     round(cycle_alignment_score, 2),
                "price_time_symmetry": round(price_time_symmetry_score, 2),
                "harmonic_confluence": round(harmonic_confluence_score, 2),
            },
            "decision_trace":       trace,
        }

    # ── Component Scorers ──────────────────────────────────────────────────────

    def _score_vol_compression(self, vol, comp) -> float:
        """0–100: how well compressed volatility is (energy buildup)."""
        score = 50.0
        if vol:
            bb_pct  = float(vol.bb_width_percentile or 0.5)
            atr_pct = float(vol.atr_percentile or 0.5)
            score   = (1 - bb_pct) * 50 + (1 - atr_pct) * 50
        if comp and comp.is_compressed:
            score += 10
            score += min(float(comp.compression_duration or 0) / 2, 15)  # duration bonus
        if vol and vol.is_expansion:
            score = 90.0  # expansion after compression = max score
        return float(np.clip(score, 0, 100))

    def _score_breakout(self, bo) -> float:
        """0–100: breakout strength."""
        if not bo:
            return 30.0
        if not bo.triggers_met or bo.triggers_met < 2:
            return 20.0
        return float(np.clip(bo.breakout_strength or 0, 0, 100))

    def _score_trend(self, ind, price: float) -> float:
        """0–100: trend alignment from indicators."""
        score = 50.0
        if not ind:
            return score

        # MA alignment
        ma50  = float(ind.ma50  or price)
        ma200 = float(ind.ma200 or price)

        if price > ma50 > ma200:
            score += 20   # bullish alignment
        elif price < ma50 < ma200:
            score += 15   # bearish alignment (still trend)

        # RSI
        rsi = float(ind.rsi or 50)
        if 55 <= rsi <= 75:
            score += 10   # healthy bullish momentum
        elif 25 <= rsi <= 45:
            score += 5    # oversold recovery zone

        # MACD
        macd = float(ind.macd_histogram or 0)
        if macd > 0:
            score += 10
        elif macd < 0:
            score -= 5

        return float(np.clip(score, 0, 100))

    def _score_wyckoff(self, wyck, is_compression: bool) -> float:
        """0–100: Wyckoff phase confluence score."""
        if not wyck:
            return 50.0
        base = float(wyck.phase_score or 50)
        if is_compression and wyck.phase == "Accumulation":
            base += 15   # compression in accumulation = prime setup
        return float(np.clip(base, 0, 100))

    def _score_pattern_detections(self, detections: list) -> float:
        """
        0–100: aggregate score from PatternDetection table.

        Strategy:
          • Only READY, BREAKOUT, FORMING detections count (skip NOT_PRESENT/FAILED)
          • Best single-pattern confidence = 70% of score
          • Breadth bonus: +3 per additional confirming detection (cap 30 pts)
          • Status multiplier: BREAKOUT ×1.15, READY ×1.0, FORMING ×0.75
          • Direction diversity: conflicting bullish/bearish penalised by −10

        Returns neutral 50.0 when no detections are available.
        """
        if not detections:
            return 50.0

        active_statuses = {"READY", "BREAKOUT", "FORMING"}
        status_mult     = {"BREAKOUT": 1.15, "READY": 1.0, "FORMING": 0.75}

        scored: list[float] = []
        bull_count = 0
        bear_count = 0

        for d in detections:
            status = (d.status or "").upper()
            if status not in active_statuses:
                continue
            conf  = float(d.confidence or 0)          # 0–100
            mult  = status_mult.get(status, 1.0)
            score = min(conf * mult, 100.0)
            scored.append(score)

            direction = (d.direction or "").lower()
            if direction == "bullish":
                bull_count += 1
            elif direction == "bearish":
                bear_count += 1

        if not scored:
            return 45.0   # detections exist but all FAILED/NOT_PRESENT → slight negative signal

        best_score   = max(scored)
        breadth_pts  = min((len(scored) - 1) * 3, 30)
        conflict_pen = -10.0 if (bull_count > 0 and bear_count > 0) else 0.0

        raw = best_score * 0.70 + breadth_pts + conflict_pen + 30.0   # 30-pt base
        return float(np.clip(raw, 0, 100))

    def _score_cycle_alignment(self, symbol: str, cur_price: float) -> float:
        """0–100: Cycle alignment from FFT/wavelet/Hilbert analysis.

        High score when cycle phase is near trough (buy zone) or
        when the alignment_score from the cycle engine is strong.
        """
        try:
            from api.routers.cycles import _fetch_price_data
            from cycle_engine.cycle_projection import project_cycle
            df = _fetch_price_data(symbol)
            if df.empty:
                return 50.0
            result = project_cycle(df)
            alignment = float(result.get("cycle_alignment_score", 0.5))
            phase = float(result.get("cycle_phase", 0.5))

            # Phase near trough (0.7-1.0 or 0.0-0.3) = buying opportunity
            # Phase near peak (0.3-0.7) = caution
            if phase <= 0.3 or phase >= 0.7:
                phase_score = 70.0
            else:
                phase_score = 35.0

            return float(np.clip(alignment * 60 + phase_score * 0.4, 0, 100))
        except Exception:
            return 50.0

    def _score_price_time_symmetry(self, symbol: str) -> float:
        """0–100: Price-time symmetry from Gann geometry engine.

        High score when recent swings show balanced price-time ratios
        (symmetry_ratio near 1.0) indicating potential reversal zones.
        """
        try:
            from api.routers.cycles import _fetch_price_data
            from geometry_engine.price_time_symmetry import compute_price_time_symmetry
            df = _fetch_price_data(symbol)
            if df.empty:
                return 50.0
            result = compute_price_time_symmetry(df)
            zones = result.get("symmetry_zones", [])
            if not zones:
                return 40.0

            # Score based on how many recent symmetry zones exist
            # and how close symmetry_ratio is to 1.0
            best_ratio = min(abs(z.get("symmetry_ratio", 2.0) - 1.0) for z in zones)
            # ratio_diff of 0 = perfect symmetry → 100, ratio_diff of 1.0 → 30
            ratio_score = max(100 - best_ratio * 70, 30)

            # Bonus for multiple zones
            count_bonus = min(len(zones) * 5, 20)

            return float(np.clip(ratio_score + count_bonus, 0, 100))
        except Exception:
            return 50.0

    def _score_harmonic_confluence(
        self, harmonic_patterns: list, cur_price: float, cur_atr: float
    ) -> float:
        """0–100: Harmonic pattern PRZ proximity score.

        High score when price is within or near a Potential Reversal Zone (PRZ)
        of an active harmonic pattern (Gartley, Bat, Butterfly, Crab, Cypher).
        """
        if not harmonic_patterns or cur_atr <= 0:
            return 40.0

        best_score = 40.0
        for p in harmonic_patterns:
            prz_low  = float(getattr(p, 'prz_low', 0) or 0)
            prz_high = float(getattr(p, 'prz_high', 0) or 0)
            conf     = float(getattr(p, 'confidence', 0) or getattr(p, 'pattern_score', 0) or 0)

            if prz_low <= 0 or prz_high <= 0:
                continue

            # Distance from PRZ in ATR units
            if cur_price < prz_low:
                dist_atr = (prz_low - cur_price) / cur_atr
            elif cur_price > prz_high:
                dist_atr = (cur_price - prz_high) / cur_atr
            else:
                dist_atr = 0  # inside PRZ

            # Score: inside PRZ = 90+, within 1 ATR = 70-90, beyond = decay
            if dist_atr == 0:
                proximity = 95.0
            elif dist_atr <= 1.0:
                proximity = 90.0 - dist_atr * 20
            elif dist_atr <= 3.0:
                proximity = 70.0 - (dist_atr - 1.0) * 15
            else:
                proximity = max(40.0 - (dist_atr - 3.0) * 5, 20)

            # Weight by confidence
            score = proximity * 0.7 + conf * 30
            best_score = max(best_score, score)

        return float(np.clip(best_score, 0, 100))

    # ── Trade Plan Builder ─────────────────────────────────────────────────────

    def _build_trade_plan(
        self, price: float, atr: float, em: dict, bo, comp
    ) -> dict:
        """Build ATR-based trade plan levels."""
        # Targets based on expected move multiplier
        mult   = em.get("multiplier", 2.5)
        target = price + atr * mult

        # If breakout, use compression range for stop
        if bo and bo.compression_range_low:
            stop = float(bo.compression_range_low) * 0.99  # 1% below range low
        else:
            stop = price - atr * 1.5

        return {
            "entry":  round(price, 4),
            "add":    round(price - atr * 0.5, 4),    # add zone
            "scale":  round(price + atr * 2.0, 4),    # first scale-out
            "target": round(target, 4),                # full target
            "stop":   round(stop, 4),                  # invalidation
        }

    # ── Decision Trace Builder ─────────────────────────────────────────────────

    def _build_decision_trace(self, **kwargs) -> str:
        """
        Build a full human-readable decision trace.
        Returns JSON string stored in DB.
        """
        sym      = kwargs["symbol"]
        vr       = kwargs["vol_regime"]
        conf     = kwargs["confluence"]
        tier     = kwargs["signal_tier"]
        vol      = kwargs["vol"]
        comp     = kwargs["comp"]
        bo       = kwargs["bo"]
        liq      = kwargs["liq"]
        wyck     = kwargs["wyck"]
        chart_p    = kwargs["chart_patterns"]
        harm_p     = kwargs["harmonic_patterns"]
        pat_detect = kwargs.get("pattern_detections", [])
        em         = kwargs["em"]
        plan     = kwargs["plan"]
        price    = kwargs["cur_price"]

        lines = [
            f"═══════════════════════════════════════════",
            f"  TRADE SIGNAL — {sym}",
            f"═══════════════════════════════════════════",
            f"",
            f"  Volatility Regime:  {vr}",
        ]

        # Compression details
        if comp and comp.is_compressed:
            lines.append(
                f"  Compression:        BB width pct = {(vol.bb_width_percentile or 0)*100:.1f}%  "
                f"| Duration = {comp.compression_duration or 0} bars"
            )
            lines.append(
                f"  Range:              High = {comp.range_high or 0:.2f}  "
                f"Low = {comp.range_low or 0:.2f}  "
                f"Mid = {comp.range_mid or 0:.2f}"
            )
        elif vol and vol.compression_range_high:
            lines.append(
                f"  Prior Range:        High = {vol.compression_range_high:.2f}  "
                f"Low = {vol.compression_range_low or 0:.2f}"
            )

        # Breakout details
        if bo and bo.triggers_met and bo.triggers_met >= 2:
            direction = bo.breakout_direction or "up"
            vol_ratio = bo.volume_ratio or 0
            triggers  = []
            if bo.price_trigger:    triggers.append("Price")
            if bo.volume_trigger:   triggers.append("Volume")
            if bo.momentum_trigger: triggers.append("Momentum")
            lines.append(
                f"  Breakout:           Price broke {'above' if direction == 'up' else 'below'} range "
                f"| Volume = {vol_ratio:.1f}× avg | Triggers: {', '.join(triggers)}"
            )
            lines.append(
                f"  Breakout Strength:  {bo.breakout_strength or 0:.1f}/100"
            )

        # Pattern context
        best_pattern = None
        if chart_p:
            best_pattern = max(chart_p, key=lambda p: p.pattern_confidence or 0)
            lines.append(
                f"  Chart Pattern:      {(best_pattern.pattern_type or '').replace('_', ' ').title()} "
                f"({(best_pattern.direction or '').title()}) "
                f"| Confidence = {(best_pattern.pattern_confidence or 0)*100:.0f}%"
            )

        if harm_p:
            best_harm = max(harm_p, key=lambda p: p.confidence or 0)
            lines.append(
                f"  Harmonic:           {best_harm.pattern_type} "
                f"({(best_harm.direction or '').title()}) "
                f"| PRZ = {best_harm.prz_low:.2f}–{best_harm.prz_high:.2f}"
            )

        # PatternDetection engine results — top 3 by confidence
        active_pd = [
            d for d in (pat_detect or [])
            if (d.status or "").upper() in {"READY", "BREAKOUT", "FORMING"}
        ]
        if active_pd:
            top_pd = sorted(active_pd, key=lambda d: d.confidence or 0, reverse=True)[:3]
            pd_parts = [
                f"{d.pattern_name} ({(d.direction or '').title()}, "
                f"{d.status}, {(d.confidence or 0):.0f}%)"
                for d in top_pd
            ]
            lines.append(f"  Detectors ({len(active_pd)} active): {' | '.join(pd_parts)}")

        # Wyckoff phase
        if wyck:
            event = ""
            if wyck.spring_detected:         event = " ← Spring detected!"
            elif wyck.upthrust_detected:      event = " ← Upthrust detected!"
            elif wyck.secondary_test_detected: event = " ← Secondary Test"
            lines.append(
                f"  Wyckoff Phase:      {wyck.phase} / {wyck.sub_phase}{event} "
                f"| Confidence = {(wyck.confidence or 0)*100:.0f}%"
            )

        # Liquidity
        if liq:
            liq_notes = []
            if liq.swept_prev_high: liq_notes.append("Swept prev high")
            if liq.swept_prev_low:  liq_notes.append("Swept prev low")
            if liq.above_vwap:      liq_notes.append("Above VWAP")
            if liq.near_vol_node:   liq_notes.append("At volume node")
            if liq_notes:
                lines.append(f"  Liquidity:          {' | '.join(liq_notes)}")

        # Expected move + targets
        lines.extend([
            f"",
            f"  Expected Move:      {em.get('expected_move_pct_display', 'N/A')} "
            f"({'✓ PASSES' if em.get('passes_filter') else '✗ FAILS 10% minimum'})",
            f"  Multiplier:         {em.get('multiplier', 0):.2f}× ATR",
            f"",
            f"  ── Trade Plan ──────────────────────────",
            f"  Entry:              ${plan['entry']:.2f}",
            f"  Add/Accumulate:     ${plan['add']:.2f}",
            f"  Scale Out:          ${plan['scale']:.2f}",
            f"  Full Target:        ${plan['target']:.2f}",
            f"  Stop Loss:          ${plan['stop']:.2f}",
            f"  Risk/Reward:        {abs((plan['target']-plan['entry'])/(plan['entry']-plan['stop']+1e-8)):.1f}:1",
            f"",
            f"  ── Confluence Scores (10 components) ──",
            f"  Vol Compression:    {kwargs['vol_comp_score']:.1f}/100  (wt: 20%)",
            f"  Breakout Strength:  {kwargs['breakout_score']:.1f}/100  (wt: 15%)",
            f"  Trend Alignment:    {kwargs['trend_score']:.1f}/100  (wt: 12%)",
            f"  Cycle Alignment:    {kwargs.get('cycle_alignment_score', 50):.1f}/100  (wt: 10%)",
            f"  Liquidity:          {kwargs['liq_score']:.1f}/100  (wt: 10%)",
            f"  Price-Time Symm:    {kwargs.get('price_time_symmetry_score', 50):.1f}/100  (wt:  8%)",
            f"  Patterns:           {kwargs['pattern_score']:.1f}/100  (wt:  8%)",
            f"  Wyckoff Phase:      {kwargs['wyckoff_score']:.1f}/100  (wt:  8%)",
            f"  Harmonic Conf.:     {kwargs.get('harmonic_confluence_score', 50):.1f}/100  (wt:  5%)",
            f"  Gann Projection:    {kwargs['gann_score']:.1f}/100  (wt:  4%)",
            f"  ────────────────────────────────────────",
            f"  CONFLUENCE SCORE:   {conf:.1f}/100  → {tier}",
            f"═══════════════════════════════════════════",
        ])

        trace_dict = {
            "text": "\n".join(lines),
            "symbol": sym,
            "confluence": conf,
            "tier": tier,
            "vol_regime": vr,
            "expected_move_pct": em.get("expected_move_pct", 0),
            "passes_move_filter": em.get("passes_filter", False),
        }
        return json.dumps(trace_dict, ensure_ascii=False)[:8000]

    # ── Gated result (LOW_VOL) ─────────────────────────────────────────────────

    def _build_gated_result(self, symbol: str, price: float, vol_regime: str, date) -> dict:
        trace = json.dumps({
            "text": (
                f"═══════════════════════════════\n"
                f"  SIGNAL GATED — {symbol}\n"
                f"═══════════════════════════════\n"
                f"  Volatility Regime: {vol_regime}\n"
                f"  Reason: LOW_VOL — insufficient volatility expansion.\n"
                f"  Action: Symbol added to watchlist.\n"
                f"          Wait for compression to resolve.\n"
                f"═══════════════════════════════"
            ),
            "symbol": symbol, "tier": "NONE", "vol_regime": vol_regime,
        })
        return {
            "symbol":            symbol,
            "date":              str(date),
            "confluence_score":  0.0,
            "signal_tier":       "NONE",
            "volatility_regime": vol_regime,
            "is_compression":    True,
            "is_breakout":       False,
            "expected_move_pct": 0,
            "entry_price":       price,
            "decision_trace":    trace,
            "gated":             True,
        }

    # ── DB Helpers ─────────────────────────────────────────────────────────────

    def _latest_price(self, session, instr_id: int):
        return session.execute(
            select(PriceData).where(PriceData.instrument_id == instr_id)
            .order_by(desc(PriceData.date)).limit(1)
        ).scalar_one_or_none()

    def _latest_indicator(self, session, instr_id: int):
        return session.execute(
            select(Indicator).where(Indicator.instrument_id == instr_id)
            .order_by(desc(Indicator.date)).limit(1)
        ).scalar_one_or_none()

    def _latest_vol_regime(self, session, instr_id: int):
        return session.execute(
            select(VolatilityRegime).where(VolatilityRegime.instrument_id == instr_id)
            .order_by(desc(VolatilityRegime.date)).limit(1)
        ).scalar_one_or_none()

    def _latest_compression(self, session, instr_id: int):
        return session.execute(
            select(RangeCompression).where(RangeCompression.instrument_id == instr_id)
            .order_by(desc(RangeCompression.date)).limit(1)
        ).scalar_one_or_none()

    def _latest_breakout(self, session, instr_id: int):
        return session.execute(
            select(BreakoutSignal).where(BreakoutSignal.instrument_id == instr_id)
            .order_by(desc(BreakoutSignal.date)).limit(1)
        ).scalar_one_or_none()

    def _latest_liquidity(self, session, instr_id: int):
        return session.execute(
            select(LiquidityLevel).where(LiquidityLevel.instrument_id == instr_id)
            .order_by(desc(LiquidityLevel.date)).limit(1)
        ).scalar_one_or_none()

    def _latest_wyckoff(self, session, instr_id: int):
        return session.execute(
            select(WyckoffPhase).where(WyckoffPhase.instrument_id == instr_id)
            .order_by(desc(WyckoffPhase.date)).limit(1)
        ).scalar_one_or_none()

    def _latest_chart_patterns(self, session, instr_id: int):
        return session.execute(
            select(ChartPattern).where(ChartPattern.instrument_id == instr_id)
            .order_by(desc(ChartPattern.date)).limit(5)
        ).scalars().all()

    def _latest_harmonic_patterns(self, session, instr_id: int):
        return session.execute(
            select(HarmonicPattern).where(HarmonicPattern.instrument_id == instr_id)
            .order_by(desc(HarmonicPattern.date)).limit(3)
        ).scalars().all()

    def _latest_pattern_detections(self, session, instr_id: int):
        """
        Fetch the most-recent 20 PatternDetection rows for an instrument.
        20 rows = enough breadth across all 35+ detectors without being
        dominated by stale formations.  Ordered most-recent date first,
        then by confidence descending so the best signals surface first.
        """
        return session.execute(
            select(PatternDetection)
            .where(PatternDetection.instrument_id == instr_id)
            .order_by(desc(PatternDetection.date), desc(PatternDetection.confidence))
            .limit(20)
        ).scalars().all()

    def _latest_signal(self, session, instr_id: int):
        return session.execute(
            select(Signal).where(Signal.instrument_id == instr_id)
            .order_by(desc(Signal.date)).limit(1)
        ).scalar_one_or_none()

    def _upsert(self, session, instrument_id: int, result: dict):
        cs = result.get("component_scores", {})
        record = {
            "instrument_id":           instrument_id,
            "date":                    result.get("date"),
            "vol_compression_score":   cs.get("vol_compression", 0),
            "breakout_score":          cs.get("breakout", 0),
            "trend_score":             cs.get("trend", 0),
            "liquidity_score":         cs.get("liquidity", 0),
            "pattern_score":           cs.get("pattern", 0),
            "wyckoff_score":           cs.get("wyckoff", 0),
            "gann_score":              cs.get("gann", 0),
            "cycle_alignment_score":   cs.get("cycle_alignment", 0),
            "price_time_symmetry_score": cs.get("price_time_symmetry", 0),
            "harmonic_confluence_score": cs.get("harmonic_confluence", 0),
            "confluence_score":    result.get("confluence_score", 0),
            "signal_tier":         result.get("signal_tier", "NONE"),
            "volatility_regime":   result.get("volatility_regime", "NORMAL_VOL"),
            "is_compression":      result.get("is_compression", False),
            "is_breakout":         result.get("is_breakout", False),
            "expected_move_pct":   result.get("expected_move_pct", 0),
            "entry_price":         result.get("entry_price"),
            "add_price":           result.get("add_price"),
            "scale_price":         result.get("scale_price"),
            "target_price":        result.get("target_price"),
            "stop_price":          result.get("stop_price"),
            "decision_trace":      result.get("decision_trace"),
        }
        stmt = pg_insert(ConfluenceScore).values([record])
        stmt = stmt.on_conflict_do_update(
            constraint="uq_confluence_inst_date",
            set_={k: stmt.excluded[k] for k in record if k not in ("instrument_id", "date")}
        )
        session.execute(stmt)
        session.commit()

    def _cs_to_dict(self, cs: ConfluenceScore) -> dict:
        trace_text = ""
        if cs.decision_trace:
            try:
                trace_text = json.loads(cs.decision_trace).get("text", "")
            except Exception:
                trace_text = cs.decision_trace
        return {
            "date":             str(cs.date),
            "confluence_score": cs.confluence_score,
            "signal_tier":      cs.signal_tier,
            "volatility_regime": cs.volatility_regime,
            "is_compression":   cs.is_compression,
            "is_breakout":      cs.is_breakout,
            "expected_move_pct": cs.expected_move_pct,
            "entry_price":      cs.entry_price,
            "add_price":        cs.add_price,
            "scale_price":      cs.scale_price,
            "target_price":     cs.target_price,
            "stop_price":       cs.stop_price,
            "component_scores": {
                "vol_compression":     cs.vol_compression_score,
                "breakout":            cs.breakout_score,
                "trend":               cs.trend_score,
                "liquidity":           cs.liquidity_score,
                "pattern":             cs.pattern_score,
                "wyckoff":             cs.wyckoff_score,
                "gann":                cs.gann_score,
                "cycle_alignment":     getattr(cs, 'cycle_alignment_score', None),
                "price_time_symmetry": getattr(cs, 'price_time_symmetry_score', None),
                "harmonic_confluence": getattr(cs, 'harmonic_confluence_score', None),
            },
            "decision_trace_text": trace_text,
        }
