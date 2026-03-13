"""
RoboAlgo — Setup Quality Scorer
Composite 0-100 scoring model for trade setup quality.

Aggregates ten signal factors:
  1. confluence_score        (28%) — primary gate from ConfluenceScore table
  2. breakout_score          (18%) — latest BreakoutSignal strength
  3. momentum_acceleration   (13%) — RSI 3-day acceleration from Feature table
  4. volume_participation    (13%) — directional volume quality from Feature table
  5. correlation_exposure    (08%) — lower cross-asset corr → higher score
  6. market_state            (04%) — EXPANSION=1.0, TREND=0.85, COMP=0.65, CHAOS=0.20
  7. volatility_regime       (04%) — LOW=0.90, NORMAL=0.75, HIGH=0.40
  8. breakout_quality        (04%) — BreakoutQualityScore gate (< 60 → zero)
  9. liquidity_shelf         (03%) — LiquidityShelfScore absorption zone bonus
 10. liquidity_alignment     (05%) — LiquidityMapEngine directional alignment

Grades:
  A  ≥ 78   — strong edge, high confidence
  B  62-77  — good setup, proceed with standard sizing
  C  48-61  — moderate, consider reduced size or wait for confirmation
  D  < 48   — weak or misaligned, skip or paper-track only

Output
------
{
  "symbol":       str,
  "date":         str,         # ISO date of the score
  "quality_score": float,      # 0-100
  "quality_grade": str,        # A/B/C/D
  "factors": {
    "confluence_score":        float | None,
    "breakout_score":          float | None,
    "momentum_acceleration":   float | None,   # -1 to +1
    "volume_participation":    float | None,   # -1 to +1
    "correlation_exposure":    float | None,   # -1 to +1
    "market_state":            str   | None,
    "volatility_regime":       str   | None,
    "liquidity_alignment":     float | None,   # 0-100
  },
  "weights": { ... },          # applied weight for each factor
  "breakdown": { ... },        # weighted contribution per factor
  "computed_at": str,
}
"""

import json
import logging
from datetime import datetime, date, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# ── Weight table ──────────────────────────────────────────────────────────────
# v3: added liquidity_alignment (5%); reduced correlation/market_state/
#     vol_regime/breakout_quality/liquidity_shelf by 1% each to compensate.
WEIGHTS = {
    "confluence":          0.28,
    "breakout":            0.18,
    "momentum_accel":      0.13,
    "volume_part":         0.13,
    "correlation":         0.08,   # -1%
    "market_state":        0.04,   # -1%
    "vol_regime":          0.04,   # -1%
    "breakout_quality":    0.04,   # -1%
    "liquidity_shelf":     0.03,   # -1%
    "liquidity_alignment": 0.05,   # NEW — LiquidityMapEngine directional alignment
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"

# ── Gating thresholds ─────────────────────────────────────────────────────────
BREAKOUT_QUALITY_GATE  = 60.0  # score < 60 → reject
LIQUIDITY_SHELF_GATE   = 40.0  # score < 40 during breakout → penalise

# ── Context weight tables ─────────────────────────────────────────────────────
MARKET_STATE_WEIGHTS = {
    "EXPANSION":   1.00,
    "TREND":       0.85,
    "COMPRESSION": 0.65,
    "CHAOS":       0.20,
}

VOL_REGIME_WEIGHTS = {
    "LOW_VOL":    0.90,
    "NORMAL_VOL": 0.75,
    "HIGH_VOL":   0.40,
}

# ── Grade thresholds ──────────────────────────────────────────────────────────
def _grade(score: float) -> str:
    if score >= 78:
        return "A"
    if score >= 62:
        return "B"
    if score >= 48:
        return "C"
    return "D"


class SetupQualityScorer:
    """
    Scores a trade setup's quality (0-100) by pulling the latest available
    data for a symbol from the relevant DB tables.
    """

    # ── Public API ─────────────────────────────────────────────────────────────

    def score_symbol(self, symbol: str) -> dict:
        """
        Compute the latest setup quality score for *symbol*.
        Uses the most recent data available in each engine's table.
        """
        factors = self._gather_factors(symbol)
        return self._compute_score(symbol, factors)

    def compute_and_store_all(self, as_of_date: Optional[date] = None):
        """
        Compute and persist SetupQualityScore for every instrument.
        Called by the pipeline runner.
        """
        from database.connection import get_session
        from database.models import Instrument, SetupQualityScore
        from sqlalchemy import select
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        target_date = as_of_date or date.today()

        with get_session() as session:
            instruments = session.execute(select(Instrument)).scalars().all()

        for inst in instruments:
            try:
                factors = self._gather_factors(inst.symbol, as_of_date=target_date)
                result  = self._compute_score(inst.symbol, factors, as_of_date=target_date)
                self._persist(result, inst.id, target_date)
            except Exception as e:
                logger.warning("SetupQualityScorer failed for %s: %s", inst.symbol, e)

    def get_scores(
        self,
        min_grade: str = "D",
        limit:     int  = 50,
        as_of_date: Optional[date] = None,
    ) -> list[dict]:
        """
        Return top setup quality scores from the DB, ordered by quality_score desc.
        min_grade filter: 'A', 'B', 'C', or 'D' (includes that grade and above).
        """
        from database.connection import get_session
        from database.models import SetupQualityScore
        from sqlalchemy import select, desc

        grade_order = ["A", "B", "C", "D"]
        min_idx     = grade_order.index(min_grade) if min_grade in grade_order else 3
        allowed     = grade_order[:min_idx + 1]

        target_date = as_of_date or date.today()

        with get_session() as session:
            rows = session.execute(
                select(SetupQualityScore)
                .where(
                    SetupQualityScore.date == target_date,
                    SetupQualityScore.quality_grade.in_(allowed),
                )
                .order_by(desc(SetupQualityScore.quality_score))
                .limit(limit)
            ).scalars().all()

        return [self._row_to_dict(r) for r in rows]

    # ── Factor gathering ───────────────────────────────────────────────────────

    def _gather_factors(self, symbol: str, as_of_date: Optional[date] = None) -> dict:
        from database.connection import get_session
        from database.models import (
            Instrument, Feature, ConfluenceScore, VolatilityRegime,
            MarketState,
        )
        from sqlalchemy import select, desc

        # Also import BreakoutSignal from range_engine table if it exists
        try:
            from database.models import BreakoutSignal
            has_breakout_model = True
        except ImportError:
            has_breakout_model = False

        cutoff = as_of_date or date.today()

        factors: dict = {
            "confluence_score":      None,
            "breakout_score":        None,
            "momentum_acceleration": None,
            "volume_participation":  None,
            "breakout_quality":      None,   # BreakoutQualityScore 0-100
            "liquidity_shelf":       None,   # LiquidityShelfScore 0-100
            "liquidity_alignment":   None,   # LiquidityMapEngine 0-100
            "correlation_exposure":  None,
            "market_state":          None,
            "volatility_regime":     None,
        }

        with get_session() as session:
            # ── Instrument id ──────────────────────────────────────────────
            inst = session.execute(
                select(Instrument).where(Instrument.symbol == symbol)
            ).scalar_one_or_none()
            if inst is None:
                return factors
            iid = inst.id

            # ── Latest Feature row (v2 factors) ───────────────────────────
            feat = session.execute(
                select(Feature)
                .where(Feature.instrument_id == iid, Feature.date <= cutoff)
                .order_by(desc(Feature.date))
                .limit(1)
            ).scalar_one_or_none()
            if feat:
                factors["momentum_acceleration"] = feat.momentum_acceleration
                factors["volume_participation"]  = feat.volume_participation
                factors["correlation_exposure"]  = feat.correlation_exposure

            # ── Latest ConfluenceScore ─────────────────────────────────────
            conf = session.execute(
                select(ConfluenceScore)
                .where(ConfluenceScore.instrument_id == iid, ConfluenceScore.date <= cutoff)
                .order_by(desc(ConfluenceScore.date))
                .limit(1)
            ).scalar_one_or_none()
            if conf:
                factors["confluence_score"] = conf.confluence_score

            # ── Latest BreakoutSignal ──────────────────────────────────────
            if has_breakout_model:
                try:
                    brk = session.execute(
                        select(BreakoutSignal)
                        .where(
                            BreakoutSignal.instrument_id == iid,
                            BreakoutSignal.date <= cutoff,
                        )
                        .order_by(desc(BreakoutSignal.date))
                        .limit(1)
                    ).scalar_one_or_none()
                    if brk:
                        factors["breakout_score"] = getattr(brk, "breakout_strength", None)
                except Exception:
                    pass  # table may not exist yet

            # ── Latest MarketState ─────────────────────────────────────────
            ms = session.execute(
                select(MarketState)
                .where(MarketState.instrument_id == iid, MarketState.date <= cutoff)
                .order_by(desc(MarketState.date))
                .limit(1)
            ).scalar_one_or_none()
            if ms:
                factors["market_state"] = ms.state

            # ── Latest VolatilityRegime ────────────────────────────────────
            vr = session.execute(
                select(VolatilityRegime)
                .where(VolatilityRegime.instrument_id == iid, VolatilityRegime.date <= cutoff)
                .order_by(desc(VolatilityRegime.date))
                .limit(1)
            ).scalar_one_or_none()
            if vr:
                factors["volatility_regime"] = vr.regime

            # ── BreakoutQualityScore ───────────────────────────────────────
            try:
                from database.models import BreakoutQualityScore
                bqs = session.execute(
                    select(BreakoutQualityScore.breakout_quality_score)
                    .where(
                        BreakoutQualityScore.instrument_id == iid,
                        BreakoutQualityScore.date <= cutoff,
                    )
                    .order_by(desc(BreakoutQualityScore.date))
                    .limit(1)
                ).scalar_one_or_none()
                if bqs is not None:
                    factors["breakout_quality"] = float(bqs)
            except Exception:
                pass

            # ── LiquidityShelfScore ────────────────────────────────────────
            try:
                from database.models import LiquidityShelfScore
                lss = session.execute(
                    select(LiquidityShelfScore.liquidity_shelf_score)
                    .where(
                        LiquidityShelfScore.instrument_id == iid,
                        LiquidityShelfScore.date <= cutoff,
                    )
                    .order_by(desc(LiquidityShelfScore.date))
                    .limit(1)
                ).scalar_one_or_none()
                if lss is not None:
                    factors["liquidity_shelf"] = float(lss)
            except Exception:
                pass

        # ── LiquidityMapEngine alignment (live call — no DB table needed) ──
        try:
            from structure_engine.liquidity_map import LiquidityMapEngine
            # Determine breakout direction from market_state / breakout_score
            direction = "long"
            if (factors.get("momentum_acceleration") or 0) < -0.2:
                direction = "short"
            lmap = LiquidityMapEngine()
            align = lmap.get_alignment_score(symbol, direction=direction, as_of_date=cutoff)
            factors["liquidity_alignment"] = align
        except Exception:
            pass  # non-fatal: alignment falls back to neutral 0.5

        return factors

    # ── Scoring ────────────────────────────────────────────────────────────────

    def _compute_score(
        self,
        symbol:      str,
        factors:     dict,
        as_of_date:  Optional[date] = None,
    ) -> dict:
        """
        Combine gathered factors into a 0-100 quality score.
        Any missing factor falls back to a neutral contribution (0.5 × weight).
        """
        score_date = as_of_date or date.today()

        breakdown: dict[str, float] = {}

        # 1. Confluence (30%) — normalise 0-100 score to 0-1 unit
        if factors["confluence_score"] is not None:
            c_norm = min(max(factors["confluence_score"] / 100.0, 0.0), 1.0)
        else:
            c_norm = 0.5  # neutral fallback
        breakdown["confluence"] = round(c_norm * WEIGHTS["confluence"] * 100, 2)

        # 2. Breakout strength (20%) — normalise 0-100 to 0-1; 0 if no breakout
        if factors["breakout_score"] is not None:
            b_norm = min(max(factors["breakout_score"] / 100.0, 0.0), 1.0)
        else:
            b_norm = 0.0  # no breakout → zero contribution
        breakdown["breakout"] = round(b_norm * WEIGHTS["breakout"] * 100, 2)

        # 3. Momentum acceleration (15%) — map -1→+1 to 0→1
        if factors["momentum_acceleration"] is not None:
            ma_norm = (min(max(factors["momentum_acceleration"], -1.0), 1.0) + 1.0) / 2.0
        else:
            ma_norm = 0.5
        breakdown["momentum_accel"] = round(ma_norm * WEIGHTS["momentum_accel"] * 100, 2)

        # 4. Volume participation (15%) — map -1→+1 to 0→1
        if factors["volume_participation"] is not None:
            vp_norm = (min(max(factors["volume_participation"], -1.0), 1.0) + 1.0) / 2.0
        else:
            vp_norm = 0.5
        breakdown["volume_part"] = round(vp_norm * WEIGHTS["volume_part"] * 100, 2)

        # 5. Correlation exposure (10%) — lower corr → higher score
        #    map corr -1→+1 to score 1→0 (invert), then to 0→1 range
        if factors["correlation_exposure"] is not None:
            corr  = min(max(factors["correlation_exposure"], -1.0), 1.0)
            ce_norm = (1.0 - corr) / 2.0  # corr=+1 → 0, corr=-1 → 1
        else:
            ce_norm = 0.5
        breakdown["correlation"] = round(ce_norm * WEIGHTS["correlation"] * 100, 2)

        # 6. Market state (5%)
        ms_w = MARKET_STATE_WEIGHTS.get(factors.get("market_state") or "", 0.65)
        breakdown["market_state"] = round(ms_w * WEIGHTS["market_state"] * 100, 2)

        # 7. Volatility regime (5%)
        vr_w = VOL_REGIME_WEIGHTS.get(factors.get("volatility_regime") or "", 0.75)
        breakdown["vol_regime"] = round(vr_w * WEIGHTS["vol_regime"] * 100, 2)

        # 8. Breakout quality (5%) — 0 if score < gate, proportional above gate
        bq = factors.get("breakout_quality")
        if bq is not None:
            bq_val = float(bq)
            if bq_val < BREAKOUT_QUALITY_GATE:
                bq_norm = 0.0   # hard gate: below 60 → zero contribution
            else:
                bq_norm = min((bq_val - BREAKOUT_QUALITY_GATE) / (100.0 - BREAKOUT_QUALITY_GATE), 1.0)
        else:
            bq_norm = 0.5  # neutral when not yet computed
        breakdown["breakout_quality"] = round(bq_norm * WEIGHTS["breakout_quality"] * 100, 2)

        # 9. Liquidity shelf (3%) — bonus for confirmed absorption zone
        ls = factors.get("liquidity_shelf")
        if ls is not None:
            ls_val = float(ls)
            # Penalty if breakout detected but shelf is weak
            has_breakout = (factors.get("breakout_score") or 0) > 50
            if has_breakout and ls_val < LIQUIDITY_SHELF_GATE:
                ls_norm = 0.2  # penalise: weak shelf during breakout
            else:
                ls_norm = min(ls_val / 100.0, 1.0)
        else:
            ls_norm = 0.5
        breakdown["liquidity_shelf"] = round(ls_norm * WEIGHTS["liquidity_shelf"] * 100, 2)

        # 10. Liquidity alignment (5%) — directional alignment with nearest pool
        la = factors.get("liquidity_alignment")
        if la is not None:
            la_norm = min(max(float(la) / 100.0, 0.0), 1.0)
        else:
            la_norm = 0.5  # neutral when map not available
        breakdown["liquidity_alignment"] = round(la_norm * WEIGHTS["liquidity_alignment"] * 100, 2)

        # ── Total ─────────────────────────────────────────────────────────────
        quality_score = round(sum(breakdown.values()), 1)
        quality_grade = _grade(quality_score)

        return {
            "symbol":        symbol,
            "date":          score_date.isoformat(),
            "quality_score": quality_score,
            "quality_grade": quality_grade,
            "factors":       factors,
            "weights":       WEIGHTS,
            "breakdown":     breakdown,
            "computed_at":   datetime.utcnow().isoformat() + "Z",
        }

    # ── Persistence ────────────────────────────────────────────────────────────

    def _persist(self, result: dict, instrument_id: int, score_date: date):
        from database.connection import get_session
        from database.models import SetupQualityScore
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        record = {
            "instrument_id":        instrument_id,
            "date":                 score_date,
            "symbol":               result["symbol"],
            "confluence_score":     result["factors"].get("confluence_score"),
            "breakout_score":       result["factors"].get("breakout_score"),
            "momentum_acceleration":result["factors"].get("momentum_acceleration"),
            "volume_participation": result["factors"].get("volume_participation"),
            "correlation_exposure": result["factors"].get("correlation_exposure"),
            "market_state":         result["factors"].get("market_state"),
            "volatility_regime":    result["factors"].get("volatility_regime"),
            "quality_score":        result["quality_score"],
            "quality_grade":        result["quality_grade"],
            "score_breakdown":      json.dumps(result["breakdown"]),
        }
        update_cols = [
            "symbol", "confluence_score", "breakout_score",
            "momentum_acceleration", "volume_participation",
            "correlation_exposure", "market_state", "volatility_regime",
            "quality_score", "quality_grade", "score_breakdown",
        ]

        with get_session() as session:
            stmt = pg_insert(SetupQualityScore).values([record])
            stmt = stmt.on_conflict_do_update(
                constraint="uq_setup_quality_inst_date",
                set_={col: stmt.excluded[col] for col in update_cols},
            )
            session.execute(stmt)
            session.commit()

    # ── Utility ────────────────────────────────────────────────────────────────

    @staticmethod
    def _row_to_dict(row) -> dict:
        return {
            "symbol":               row.symbol,
            "date":                 str(row.date),
            "quality_score":        row.quality_score,
            "quality_grade":        row.quality_grade,
            "confluence_score":     row.confluence_score,
            "breakout_score":       row.breakout_score,
            "momentum_acceleration":row.momentum_acceleration,
            "volume_participation": row.volume_participation,
            "correlation_exposure": row.correlation_exposure,
            "market_state":         row.market_state,
            "volatility_regime":    row.volatility_regime,
            "score_breakdown":      json.loads(row.score_breakdown) if row.score_breakdown else {},
        }
