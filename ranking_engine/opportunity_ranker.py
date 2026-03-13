"""
RoboAlgo — Opportunity Ranker
Post-processes RocketScanner output with additional context:
  • Sector/regime clustering
  • Risk-adjusted ranking (ATR-normalised expected value)
  • Diversity filtering (no more than 2 same sector in top 20)
  • Enrichment with market-breadth context
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_SECTOR_MAP: dict[str, str] = {
    # Semiconductors
    "NVDA": "semi", "AMD": "semi", "AVGO": "semi", "QCOM": "semi",
    "TXN": "semi", "MU": "semi", "AMAT": "semi", "LRCX": "semi",
    "SOXL": "semi", "SOXS": "semi",
    # Mega-cap tech
    "AAPL": "megacap", "MSFT": "megacap", "AMZN": "megacap",
    "GOOGL": "megacap", "META": "megacap", "TSLA": "megacap",
    # Fintech / Crypto
    "COIN": "crypto", "MSTR": "crypto", "MARA": "crypto",
    "RIOT": "crypto", "IREN": "crypto",
    # Biotech
    "LABU": "biotech", "LABD": "biotech",
    # Energy
    "GUSH": "energy", "DRIP": "energy", "XOM": "energy", "CVX": "energy",
    # Indices
    "TQQQ": "index", "SQQQ": "index", "UPRO": "index", "SPXU": "index",
    "SPY": "index", "QQQ": "index", "IWM": "index",
}


def _get_sector(symbol: str) -> str:
    return _SECTOR_MAP.get(symbol.upper(), "other")


@dataclass
class RankedCandidate:
    """Enriched candidate with ranking metadata."""
    rank:                int
    symbol:              str
    rocket_score:        float
    adjusted_score:      float          # risk-adjusted rocket score
    pattern_name:        str
    direction:           str
    breakout_level:      float | None
    invalidation_level:  float | None
    target:              float | None
    risk_reward:         float          # (target - current) / (current - invalidation)
    expected_value:      float          # confidence × RR − (1 − confidence)
    gamma_levels:        dict
    gamma_regime:        str
    squeeze_active:      bool
    sector:              str
    component_scores:    dict
    current_price:       float
    current_atr:         float
    notes:               list[str] = field(default_factory=list)


class OpportunityRanker:
    """
    Ranks and filters RocketCandidate objects into the final signal list.

    Usage:
        ranker = OpportunityRanker()
        ranked = ranker.rank(candidates, top_n=20)
    """

    def __init__(
        self,
        max_per_sector:   int   = 3,
        min_risk_reward:  float = 1.5,
        top_n:            int   = 20,
    ):
        self.max_per_sector  = max_per_sector
        self.min_risk_reward = min_risk_reward
        self.top_n           = top_n

    def rank(
        self,
        candidates: list,       # list[RocketCandidate]
        market_regime: str = "NORMAL",  # BULL | BEAR | NORMAL
    ) -> list[RankedCandidate]:
        """
        Enrich, filter and rank candidates.

        Steps:
          1. Compute risk/reward and expected value per candidate.
          2. Apply adjusted_score = rocket_score × regime_multiplier × rr_multiplier.
          3. Apply sector diversity cap.
          4. Return top_n sorted by adjusted_score.
        """
        enriched: list[tuple[float, RankedCandidate]] = []

        regime_mult = {"BULL": 1.05, "BEAR": 0.90, "NORMAL": 1.0}.get(market_regime, 1.0)

        for c in candidates:
            sector = _get_sector(c.symbol)
            rr     = self._risk_reward(c)
            ev     = self._expected_value(c, rr)

            # Adjusted score
            rr_mult  = min(rr / self.min_risk_reward, 1.5)   # cap at 1.5×
            adj_score = min(c.rocket_score * regime_mult * rr_mult, 100)

            rc = RankedCandidate(
                rank               = 0,   # filled below
                symbol             = c.symbol,
                rocket_score       = c.rocket_score,
                adjusted_score     = round(adj_score, 1),
                pattern_name       = c.pattern_name,
                direction          = c.direction,
                breakout_level     = c.breakout_level,
                invalidation_level = c.invalidation_level,
                target             = c.target,
                risk_reward        = round(rr, 2),
                expected_value     = round(ev, 3),
                gamma_levels       = c.gamma_levels,
                gamma_regime       = c.gamma_regime,
                squeeze_active     = c.squeeze_active,
                sector             = sector,
                component_scores   = {
                    "pattern_quality": c.pattern_quality,
                    "gamma_score":     c.gamma_score,
                    "vol_squeeze":     c.vol_squeeze_score,
                    "volume_accum":    c.volume_score,
                    "trend_align":     c.trend_score,
                },
                current_price = c.current_price,
                current_atr   = c.current_atr,
                notes         = c.notes,
            )
            enriched.append((adj_score, rc))

        # Sort by adjusted_score descending
        enriched.sort(key=lambda x: x[0], reverse=True)

        # Sector diversity cap
        sector_count: dict[str, int] = {}
        final: list[RankedCandidate] = []
        for _, rc in enriched:
            count = sector_count.get(rc.sector, 0)
            if count >= self.max_per_sector:
                continue
            sector_count[rc.sector] = count + 1
            final.append(rc)
            if len(final) >= self.top_n:
                break

        # Assign ranks
        for i, rc in enumerate(final, start=1):
            rc.rank = i

        return final

    @staticmethod
    def _risk_reward(c) -> float:
        """R:R = (target - current) / (current - invalidation)."""
        if c.target is None or c.invalidation_level is None or c.current_price <= 0:
            return 1.0
        if c.direction == "bullish":
            reward = c.target - c.current_price
            risk   = c.current_price - c.invalidation_level
        else:
            reward = c.current_price - c.target
            risk   = c.invalidation_level - c.current_price
        if risk <= 0:
            return 1.0
        return max(reward / risk, 0.0)

    @staticmethod
    def _expected_value(c, rr: float) -> float:
        """EV = p_win × RR − p_lose where p = confidence / 100."""
        p = (c.pattern_quality or 50) / 100
        return p * rr - (1 - p)

    @staticmethod
    def to_dict(rc: RankedCandidate) -> dict:
        return {
            "rank":              rc.rank,
            "symbol":            rc.symbol,
            "rocket_score":      rc.rocket_score,
            "adjusted_score":    rc.adjusted_score,
            "pattern_name":      rc.pattern_name,
            "direction":         rc.direction,
            "breakout_level":    rc.breakout_level,
            "invalidation_level": rc.invalidation_level,
            "target":            rc.target,
            "risk_reward":       rc.risk_reward,
            "expected_value":    rc.expected_value,
            "gamma_levels":      rc.gamma_levels,
            "gamma_regime":      rc.gamma_regime,
            "squeeze_active":    rc.squeeze_active,
            "sector":            rc.sector,
            "component_scores":  rc.component_scores,
            "current_price":     rc.current_price,
            "current_atr":       rc.current_atr,
            "notes":             rc.notes,
        }
