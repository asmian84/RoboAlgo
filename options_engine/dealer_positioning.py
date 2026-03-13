"""
RoboAlgo — Dealer Positioning Engine
Synthesises GEX + options-flow signals into a single dealer-positioning regime.

Outputs per symbol:
  • dealer_regime   — NET_LONG_GAMMA | NET_SHORT_GAMMA | TRANSITIONING
  • squeeze_risk    — LOW | MEDIUM | HIGH  (gamma squeeze potential)
  • gamma_levels    — dict with call_wall, put_wall, zero_gamma, max_pain
  • flow_bias       — BULLISH | BEARISH | NEUTRAL (from PC ratio + wall position)
  • positioning_score — 0-100 for scanner integration
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from options_engine.gamma_exposure import GEXProfile

logger = logging.getLogger(__name__)


@dataclass
class DealerPositioning:
    symbol:              str
    dealer_regime:       str    # NET_LONG_GAMMA | NET_SHORT_GAMMA | TRANSITIONING
    squeeze_risk:        str    # LOW | MEDIUM | HIGH
    flow_bias:           str    # BULLISH | BEARISH | NEUTRAL
    positioning_score:   float  # 0-100
    gamma_levels:        dict
    notes:               list[str]


class DealerPositioningEngine:
    """
    Translates a GEXProfile into actionable dealer-positioning signals.

    Regime logic:
      NET_LONG_GAMMA:
        total_net_gex > 0 AND spot between put_wall / call_wall
        → dealers long gamma → sell rallies / buy dips → mean-reverting
        → LOW priority for rocket scanner (no explosive potential)

      NET_SHORT_GAMMA:
        total_net_gex < 0 OR spot outside walls
        → dealers short gamma → buy rallies / sell dips → trending
        → HIGH priority for rocket scanner (explosive potential)

      TRANSITIONING:
        spot within 1 ATR of zero_gamma → about to flip
        → MEDIUM-HIGH priority (gamma flip imminent)

    Squeeze-risk logic:
      HIGH:  spot < put_wall AND gamma_regime == NEGATIVE
             → put-selling dealers forced to buy as price falls
      MEDIUM: flip_dist < 2% or pc_ratio > 1.5
      LOW:  otherwise
    """

    def compute(self, profile: GEXProfile, atr: float = 0.0) -> DealerPositioning:
        spot  = profile.spot
        notes: list[str] = []

        # ── Dealer regime ──────────────────────────────────────────────────
        flip_dist_pct = profile.gex_flip_dist
        if profile.gamma_regime == "POSITIVE" and spot > profile.put_wall and spot < profile.call_wall:
            dealer_regime = "NET_LONG_GAMMA"
            notes.append("Spot pinned between walls — dealer dampening active")
        elif profile.gamma_regime == "NEGATIVE":
            dealer_regime = "NET_SHORT_GAMMA"
            notes.append("Dealers short gamma — moves will accelerate")
        elif flip_dist_pct < 2.0:
            dealer_regime = "TRANSITIONING"
            notes.append(f"Gamma flip zone: {flip_dist_pct:.1f}% from zero-gamma")
        else:
            dealer_regime = "NET_LONG_GAMMA"

        # ── Squeeze risk ──────────────────────────────────────────────────
        if spot < profile.put_wall and profile.gamma_regime == "NEGATIVE":
            squeeze_risk = "HIGH"
            notes.append("Put wall violated in negative-gamma regime — squeeze risk HIGH")
        elif flip_dist_pct < 2.0 or profile.pc_ratio > 1.5:
            squeeze_risk = "MEDIUM"
        else:
            squeeze_risk = "LOW"

        # ── Flow bias ─────────────────────────────────────────────────────
        if profile.pc_ratio < 0.7:
            flow_bias = "BULLISH"    # heavy call buying
            notes.append(f"Call-heavy flow (PC ratio {profile.pc_ratio:.2f})")
        elif profile.pc_ratio > 1.3:
            flow_bias = "BEARISH"
            notes.append(f"Put-heavy flow (PC ratio {profile.pc_ratio:.2f})")
        else:
            flow_bias = "NEUTRAL"

        # ── Positioning score ─────────────────────────────────────────────
        # Rocket scanner wants SHORT_GAMMA + bullish flow + high squeeze risk
        regime_pts = {
            "NET_SHORT_GAMMA":  100,
            "TRANSITIONING":    70,
            "NET_LONG_GAMMA":   20,
        }.get(dealer_regime, 20)

        squeeze_pts = {"HIGH": 100, "MEDIUM": 60, "LOW": 20}.get(squeeze_risk, 20)

        bias_pts = {"BULLISH": 100, "NEUTRAL": 50, "BEARISH": 0}.get(flow_bias, 50)

        pos_score = 0.45 * regime_pts + 0.35 * squeeze_pts + 0.20 * bias_pts

        gamma_levels = {
            "call_wall":   profile.call_wall,
            "put_wall":    profile.put_wall,
            "zero_gamma":  profile.zero_gamma,
            "max_pain":    profile.max_pain,
        }

        return DealerPositioning(
            symbol             = profile.symbol,
            dealer_regime      = dealer_regime,
            squeeze_risk       = squeeze_risk,
            flow_bias          = flow_bias,
            positioning_score  = round(min(pos_score, 100), 1),
            gamma_levels       = gamma_levels,
            notes              = notes,
        )

    @staticmethod
    def to_dict(dp: DealerPositioning) -> dict:
        return {
            "symbol":             dp.symbol,
            "dealer_regime":      dp.dealer_regime,
            "squeeze_risk":       dp.squeeze_risk,
            "flow_bias":          dp.flow_bias,
            "positioning_score":  dp.positioning_score,
            "gamma_levels":       dp.gamma_levels,
            "notes":              dp.notes,
        }
