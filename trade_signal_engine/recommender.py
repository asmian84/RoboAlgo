"""
TradeRecommender — converts a ScoredSignal into an actionable trade recommendation.

Entry priority:
  1. SMC sniper entry (FVG/OB zone from smc_result)
  2. LevelEngine pounce ideas (gem trade ideas)
  3. ATR-based fallback

Position sizing: Fixed fractional risk (1-2% of account).
Profit projection: Based on position size × (target - entry).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


TRADE_PROFILES = {
    "SCALP": {"stop_atr_mult": 0.75, "target_rr": 2.5, "max_risk_pct": 1.0},
    "SWING": {"stop_atr_mult": 1.5, "target_rr": 3.0, "max_risk_pct": 1.5},
    "POSITION": {"stop_atr_mult": 2.0, "target_rr": 4.0, "max_risk_pct": 2.0},
}


@dataclass
class TradeRecommendation:
    symbol: str
    profile: str
    direction: str              # LONG | SHORT
    verdict: str                # BUY | SELL | HOLD | NO_TRADE
    score: float                # 0-100 from scorer
    confidence: str             # HIGH | MEDIUM | LOW | NONE

    # Precision levels
    entry: float = 0.0
    stop: float = 0.0
    target: float = 0.0
    risk_reward: float = 0.0
    level_source: str = ""      # "smc_sniper" | "level_pounce" | "atr_fallback"

    # Position sizing
    account_size: float = 10000.0
    risk_amount: float = 0.0    # $ at risk
    position_size: int = 0      # shares
    position_value: float = 0.0

    # Profit projection
    projected_profit: float = 0.0
    projected_profit_pct: float = 0.0

    # Context
    rejection_reason: str = ""
    entry_notes: list[str] = field(default_factory=list)
    component_summary: dict[str, Any] = field(default_factory=dict)


class TradeRecommender:
    """Convert scored signals into actionable trade recommendations."""

    def recommend(
        self,
        signal,  # ScoredSignal from scorer.py
        atr: float = 0.0,
        current_price: float = 0.0,
        account_size: float = 10000.0,
    ) -> TradeRecommendation:
        """
        Generate a trade recommendation from a scored signal.

        Args:
            signal: ScoredSignal from UnifiedScorer
            atr: ATR value (if 0, estimated from score)
            current_price: Current price (if 0, use entry estimate)
            account_size: Account size for position sizing

        Returns:
            TradeRecommendation with entry/stop/target/sizing
        """
        profile_cfg = TRADE_PROFILES.get(signal.profile, TRADE_PROFILES["SWING"])

        rec = TradeRecommendation(
            symbol=signal.symbol,
            profile=signal.profile,
            direction=signal.direction,
            verdict="NO_TRADE",
            score=signal.score,
            confidence=signal.confidence,
            account_size=account_size,
        )

        # NO_TRADE signals
        if signal.direction == "NO_TRADE":
            rec.verdict = "NO_TRADE"
            rec.rejection_reason = signal.rejection_reason or "Score too low or hard gate triggered"
            return rec

        # Default ATR estimate if not provided
        if atr <= 0 and current_price > 0:
            atr = current_price * 0.02  # 2% default estimate

        if current_price <= 0:
            rec.verdict = "NO_TRADE"
            rec.rejection_reason = "No current price available"
            return rec

        stop_mult = profile_cfg["stop_atr_mult"]
        target_rr = profile_cfg["target_rr"]
        max_risk_pct = profile_cfg["max_risk_pct"]

        # ── Priority 1: SMC sniper entry ─────────────────────────
        smc_detail = signal.component_details.get("smc", {})
        sniper_entry = smc_detail.get("sniper_entry", 0)

        if sniper_entry > 0 and abs(sniper_entry - current_price) / current_price < 0.05:
            # SMC entry within 5% of current price — use it
            rec.entry = round(sniper_entry, 2)
            zone_bottom = smc_detail.get("entry_zone_bottom", sniper_entry)
            zone_top = smc_detail.get("entry_zone_top", sniper_entry)

            if signal.direction == "LONG":
                rec.stop = round(zone_bottom - atr * 0.3, 2)
                risk_per_share = rec.entry - rec.stop
                rec.target = round(rec.entry + risk_per_share * target_rr, 2)
            else:
                rec.stop = round(zone_top + atr * 0.3, 2)
                risk_per_share = rec.stop - rec.entry
                rec.target = round(rec.entry - risk_per_share * target_rr, 2)

            rec.level_source = "smc_sniper"
            rec.entry_notes.append(f"SMC sniper entry at FVG/OB zone ({smc_detail.get('zone', 'N/A')})")
            if smc_detail.get("in_ote"):
                rec.entry_notes.append("Entry in OTE zone (62-79% fib retracement)")

        # ── Priority 2: LevelEngine pounce ideas ─────────────────
        elif signal.component_details.get("levels", {}).get("pounce_ideas"):
            pounce_ideas = signal.component_details["levels"]["pounce_ideas"]
            # Use the first pounce idea
            idea = pounce_ideas[0]
            if idea:
                rec.entry = idea.get("entry", current_price)
                rec.stop = idea.get("stop", 0)
                rec.target = idea.get("target", 0)
                rec.level_source = "level_pounce"
                rec.entry_notes.append("Entry from LevelEngine gem pounce zone")

        # ── Priority 3: ATR fallback ─────────────────────────────
        else:
            rec.entry = round(current_price, 2)
            if signal.direction == "LONG":
                rec.stop = round(current_price - atr * stop_mult, 2)
                risk_per_share = rec.entry - rec.stop
                rec.target = round(current_price + risk_per_share * target_rr, 2)
            else:
                rec.stop = round(current_price + atr * stop_mult, 2)
                risk_per_share = rec.stop - rec.entry
                rec.target = round(current_price - risk_per_share * target_rr, 2)
            rec.level_source = "atr_fallback"
            rec.entry_notes.append(f"ATR-based entry ({stop_mult}x ATR stop, {target_rr}:1 R:R)")

        # ── Compute risk:reward ──────────────────────────────────
        if rec.entry > 0 and rec.stop > 0 and rec.target > 0:
            if signal.direction == "LONG":
                risk = abs(rec.entry - rec.stop)
                reward = abs(rec.target - rec.entry)
            else:
                risk = abs(rec.stop - rec.entry)
                reward = abs(rec.entry - rec.target)

            rec.risk_reward = round(reward / risk, 1) if risk > 0 else 0.0

        # ── Position sizing (fixed fractional) ───────────────────
        risk_per_share = abs(rec.entry - rec.stop) if rec.entry > 0 and rec.stop > 0 else atr
        if risk_per_share > 0:
            max_risk_dollars = account_size * (max_risk_pct / 100)
            rec.risk_amount = round(max_risk_dollars, 2)
            rec.position_size = max(1, int(max_risk_dollars / risk_per_share))
            rec.position_value = round(rec.position_size * rec.entry, 2)

        # ── Profit projection ────────────────────────────────────
        if rec.position_size > 0 and rec.target > 0 and rec.entry > 0:
            if signal.direction == "LONG":
                rec.projected_profit = round(rec.position_size * (rec.target - rec.entry), 2)
            else:
                rec.projected_profit = round(rec.position_size * (rec.entry - rec.target), 2)
            rec.projected_profit_pct = round(rec.projected_profit / account_size * 100, 2)

        # ── Verdict ──────────────────────────────────────────────
        # Surface every trade — HIGH/MEDIUM = strong signal, LOW = speculative
        if signal.direction in ("LONG", "SHORT") and signal.score >= 55:
            rec.verdict = "BUY" if signal.direction == "LONG" else "SELL"
        elif signal.direction in ("LONG", "SHORT") and signal.score >= 35:
            rec.verdict = "HOLD"  # watchlist — has potential
        else:
            rec.verdict = "NO_TRADE"

        # ── Component summary for display ────────────────────────
        rec.component_summary = {
            "top_bullish": [],
            "top_bearish": [],
            "hard_gates_passed": True,
        }
        for comp_name, comp_val in signal.components.items():
            if comp_val > 0.65:
                rec.component_summary["top_bullish"].append(f"{comp_name}: {comp_val:.2f}")
            elif comp_val < 0.35:
                rec.component_summary["top_bearish"].append(f"{comp_name}: {comp_val:.2f}")

        return rec
