"""
RoboAlgo — Market Opportunity Radar
Identifies early-stage compression setups forming BEFORE breakouts occur.

The radar scans all tracked instruments and scores them for opportunity,
surfacing symbols where energy is building and institutional accumulation
is detectable — the optimal window to prepare watchlist entries.

Score: OpportunityScore  0–100

Components
----------
compression_score  (40%)
    From the RangeCompression table. Measures how much volatility has
    contracted vs historical norms. High score = deeply compressed.
    Pulled directly from the stored compression_score column (0–100).

shelf_score  (35%)
    From the LiquidityShelfScore table (or live computation).
    Measures institutional absorption at a key price level.
    High score = active accumulation zone present.

proximity_score  (25%)
    From LiquidityMapEngine: how close is current price to the nearest
    strong liquidity cluster?  Close proximity → breakout is imminent.
    Score = max(nearest_above.liquidity_score, nearest_below.liquidity_score)
            × distance_discount, where distance_discount decays with %distance.

Radar identifies a setup as "early" when:
  - compression_score ≥ 50  (volatility contraction underway)
  - shelf_score ≥ 40         (absorption activity present)
  - market_state is COMPRESSION or TREND (not yet EXPANSION)

Output
------
{
  symbol, date, opportunity_score,
  compression_score, shelf_score, proximity_score,
  market_state, is_early_stage,
  computed_at
}
"""

import logging
from datetime import datetime, date
from typing import Optional

from sqlalchemy import select, desc

from database.connection import get_session
from database.models import Instrument, RangeCompression, MarketState

logger = logging.getLogger(__name__)

# ── Component weights ──────────────────────────────────────────────────────────
WEIGHTS = {
    "compression": 0.40,
    "shelf":       0.35,
    "proximity":   0.25,
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9

# ── Constants ──────────────────────────────────────────────────────────────────
EARLY_COMPRESSION_THRESHOLD = 50.0   # min compression_score to qualify as "early"
EARLY_SHELF_THRESHOLD       = 35.0   # min shelf_score to qualify as "early"
EARLY_STAGE_STATES          = {"COMPRESSION", "TREND"}   # states eligible for early setup


class OpportunityRadarEngine:
    """
    Scans all instruments and ranks them by opportunity score.

    The highest-scoring symbols are in deep compression with active
    institutional absorption — ideal candidates to watch for a breakout.

    Usage:
        radar = OpportunityRadarEngine()
        results = radar.scan_all()          # returns sorted list
        top10  = results[:10]               # top 10 opportunities
    """

    def score_symbol(
        self,
        symbol:     str,
        as_of_date: Optional[date] = None,
    ) -> dict:
        """
        Compute the opportunity score for *symbol*.

        Returns:
            {
              symbol, date, opportunity_score,
              compression_score, shelf_score, proximity_score,
              market_state, is_early_stage, computed_at
            }
        """
        target_date = as_of_date or date.today()

        compression_score = self._get_compression_score(symbol, target_date)
        shelf_score       = self._get_shelf_score(symbol, target_date)
        proximity_score   = self._get_proximity_score(symbol)
        market_state      = self._get_market_state(symbol, target_date)

        opportunity_score = round(
            compression_score * WEIGHTS["compression"] +
            shelf_score       * WEIGHTS["shelf"]       +
            proximity_score   * WEIGHTS["proximity"],
            1,
        )

        is_early_stage = (
            compression_score >= EARLY_COMPRESSION_THRESHOLD and
            shelf_score       >= EARLY_SHELF_THRESHOLD       and
            market_state in EARLY_STAGE_STATES
        )

        return {
            "symbol":            symbol.upper(),
            "date":              target_date.isoformat(),
            "opportunity_score": opportunity_score,
            "compression_score": round(compression_score, 1),
            "shelf_score":       round(shelf_score, 1),
            "proximity_score":   round(proximity_score, 1),
            "market_state":      market_state,
            "is_early_stage":    is_early_stage,
            "computed_at":       datetime.utcnow().isoformat() + "Z",
        }

    def scan_all(
        self,
        as_of_date:  Optional[date] = None,
        limit:       int = 25,
        early_only:  bool = False,
    ) -> list[dict]:
        """
        Score all tracked instruments and return sorted by opportunity_score desc.

        Args:
            as_of_date:  Evaluate as of this date (default: today).
            limit:       Max number of symbols to return.
            early_only:  If True, return only early-stage setups.
        """
        target_date = as_of_date or date.today()

        with get_session() as session:
            instruments = session.execute(select(Instrument)).scalars().all()

        results = []
        for inst in instruments:
            try:
                result = self.score_symbol(inst.symbol, target_date)
                if early_only and not result["is_early_stage"]:
                    continue
                results.append(result)
            except Exception as e:
                logger.warning("OpportunityRadar failed for %s: %s", inst.symbol, e)

        results.sort(key=lambda x: x["opportunity_score"], reverse=True)
        return results[:limit]

    # ── Factor collectors ─────────────────────────────────────────────────────

    def _get_compression_score(self, symbol: str, cutoff: date) -> float:
        """Return the latest stored compression_score (0–100). 0 if not found."""
        try:
            with get_session() as session:
                inst = session.execute(
                    select(Instrument).where(Instrument.symbol == symbol.upper())
                ).scalar_one_or_none()
                if inst is None:
                    return 0.0

                row = session.execute(
                    select(RangeCompression.compression_score)
                    .where(
                        RangeCompression.instrument_id == inst.id,
                        RangeCompression.date <= cutoff,
                    )
                    .order_by(desc(RangeCompression.date))
                    .limit(1)
                ).scalar_one_or_none()

            return float(row) if row is not None else 0.0
        except Exception:
            return 0.0

    def _get_shelf_score(self, symbol: str, cutoff: date) -> float:
        """Return the latest stored liquidity_shelf_score (0–100). 0 if not found."""
        try:
            from database.models import LiquidityShelfScore
            with get_session() as session:
                inst = session.execute(
                    select(Instrument).where(Instrument.symbol == symbol.upper())
                ).scalar_one_or_none()
                if inst is None:
                    return 0.0

                row = session.execute(
                    select(LiquidityShelfScore.liquidity_shelf_score)
                    .where(
                        LiquidityShelfScore.instrument_id == inst.id,
                        LiquidityShelfScore.date <= cutoff,
                    )
                    .order_by(desc(LiquidityShelfScore.date))
                    .limit(1)
                ).scalar_one_or_none()

            if row is not None:
                return float(row)

            # Fallback: live computation
            from structure_engine.liquidity_shelf import LiquidityShelfEngine
            result = LiquidityShelfEngine().detect_liquidity_shelf(symbol, cutoff)
            return float(result.get("liquidity_shelf_score", 0.0))
        except Exception:
            return 0.0

    def _get_proximity_score(self, symbol: str) -> float:
        """
        Score how close current price is to the nearest strong liquidity level.
        High score = price is near a level where a move is likely imminent.
        """
        try:
            from structure_engine.liquidity_map import LiquidityMapEngine
            lmap = LiquidityMapEngine()
            result = lmap.build_liquidity_map(symbol)

            levels = result.get("liquidity_levels", [])
            if not levels:
                return 0.0

            # Pick the nearest level (above or below) with the best score
            nearest = min(levels, key=lambda l: l["distance_pct"])
            base_score    = nearest["liquidity_score"]
            # Discount for distance: within 1% = full, 5% = 0
            distance_pct  = nearest["distance_pct"]  # already as %
            dist_discount = max(0.0, 1.0 - distance_pct / 5.0)

            return round(base_score * dist_discount, 1)
        except Exception:
            return 0.0

    def _get_market_state(self, symbol: str, cutoff: date) -> str:
        """Return the latest market state for this symbol."""
        try:
            with get_session() as session:
                inst = session.execute(
                    select(Instrument).where(Instrument.symbol == symbol.upper())
                ).scalar_one_or_none()
                if inst is None:
                    return "UNKNOWN"

                row = session.execute(
                    select(MarketState.state)
                    .where(
                        MarketState.instrument_id == inst.id,
                        MarketState.date <= cutoff,
                    )
                    .order_by(desc(MarketState.date))
                    .limit(1)
                ).scalar_one_or_none()

            return str(row) if row else "UNKNOWN"
        except Exception:
            return "UNKNOWN"
