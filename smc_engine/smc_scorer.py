"""
SMC Composite Scorer — combines FVG + OB + CHoCH/BOS + Premium/Discount
into a single directional signal.

Returns:
  - smc_score: 0-1 (0.5 = neutral)
  - smc_bias: "LONG" | "SHORT" | "NEUTRAL"
  - entry_zone: precise price zone for entry (FVG/OB overlap)
  - confluence_detail: what's aligned and what's not
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from smc_engine.fair_value_gap import detect_fvgs, nearest_unfilled_fvg, FVG
from smc_engine.order_block import detect_order_blocks, nearest_unmitigated_ob, OrderBlock
from smc_engine.structure import detect_structure_shifts, latest_structure, StructureShift
from smc_engine.premium_discount import analyze_zone, ZoneAnalysis


@dataclass
class SMCResult:
    smc_score: float                      # 0-1 (>0.5 = bullish, <0.5 = bearish)
    smc_bias: str                         # LONG | SHORT | NEUTRAL
    entry_zone_top: float                 # precise entry zone upper
    entry_zone_bottom: float              # precise entry zone lower
    sniper_entry: float                   # exact best entry price (FVG/OB midpoint)
    fvg_count: int
    ob_count: int
    latest_shift: str                     # "CHoCH bullish" | "BOS bearish" | "none"
    zone: str                             # PREMIUM | DISCOUNT | EQUILIBRIUM
    in_ote: bool
    detail: dict[str, Any] = field(default_factory=dict)


def score_smc(df: pd.DataFrame, current_price: float = 0.0) -> SMCResult:
    """
    Run all SMC detectors and produce a composite result.

    Args:
        df: OHLCV DataFrame (any timeframe)
        current_price: override current price (uses last close if 0)

    Returns:
        SMCResult with composite score, bias, and precision entry zones
    """
    if current_price <= 0:
        current_price = float(df["close"].iloc[-1])

    # ── Run detectors ─────────────────────────────────────────────────────────
    fvgs = detect_fvgs(df, lookback=80)
    obs = detect_order_blocks(df, lookback=80)
    shifts = detect_structure_shifts(df, lookback=80)
    zone = analyze_zone(df)

    # ── Score components ──────────────────────────────────────────────────────
    score = 0.5
    bias_votes: list[float] = []  # >0.5 = bullish, <0.5 = bearish

    # 1. Structure shifts (most important)
    shift = latest_structure(shifts)
    shift_label = "none"
    if shift:
        shift_label = f"{shift.shift_type} {shift.direction}"
        if shift.shift_type == "CHoCH":
            # CHoCH is the reversal signal — strong directional
            if shift.direction == "bullish":
                bias_votes.append(0.85 * shift.confidence)
            else:
                bias_votes.append(0.15 + 0.35 * (1 - shift.confidence))
        elif shift.shift_type == "BOS":
            # BOS confirms the trend
            if shift.direction == "bullish":
                bias_votes.append(0.70)
            else:
                bias_votes.append(0.30)

    # 2. Premium/Discount zone
    if zone.zone == "DISCOUNT":
        bias_votes.append(0.70)  # favorable for longs
    elif zone.zone == "PREMIUM":
        bias_votes.append(0.30)  # favorable for shorts
    else:
        bias_votes.append(0.50)

    if zone.in_ote:
        bias_votes.append(0.75)  # OTE = optimal long entry

    # 3. FVG proximity
    nearest_bull_fvg = nearest_unfilled_fvg(fvgs, current_price, "bullish")
    nearest_bear_fvg = nearest_unfilled_fvg(fvgs, current_price, "bearish")

    bull_fvg_dist = abs(current_price - nearest_bull_fvg.midpoint) / current_price if nearest_bull_fvg else 1.0
    bear_fvg_dist = abs(current_price - nearest_bear_fvg.midpoint) / current_price if nearest_bear_fvg else 1.0

    if bull_fvg_dist < 0.01:  # within 1% of bullish FVG
        bias_votes.append(0.75)
    if bear_fvg_dist < 0.01:
        bias_votes.append(0.25)

    # 4. Order block proximity
    nearest_bull_ob = nearest_unmitigated_ob(obs, current_price, "bullish")
    nearest_bear_ob = nearest_unmitigated_ob(obs, current_price, "bearish")

    if nearest_bull_ob:
        ob_dist = abs(current_price - nearest_bull_ob.zone_top) / current_price
        if ob_dist < 0.015:  # within 1.5% of bullish OB
            bias_votes.append(0.70 + 0.15 * nearest_bull_ob.strength)
    if nearest_bear_ob:
        ob_dist = abs(current_price - nearest_bear_ob.zone_bottom) / current_price
        if ob_dist < 0.015:
            bias_votes.append(0.30 - 0.15 * nearest_bear_ob.strength)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    if bias_votes:
        score = sum(bias_votes) / len(bias_votes)

    if score > 0.58:
        smc_bias = "LONG"
    elif score < 0.42:
        smc_bias = "SHORT"
    else:
        smc_bias = "NEUTRAL"

    # ── Precision entry zone ──────────────────────────────────────────────────
    entry_top = current_price
    entry_bottom = current_price
    sniper_entry = current_price

    if smc_bias == "LONG":
        # Best entry = FVG + OB overlap, or closest FVG, or closest OB
        candidates = []
        if nearest_bull_fvg and not nearest_bull_fvg.filled:
            candidates.append((nearest_bull_fvg.midpoint, nearest_bull_fvg.top, nearest_bull_fvg.bottom))
        if nearest_bull_ob and not nearest_bull_ob.mitigated:
            candidates.append((
                (nearest_bull_ob.zone_top + nearest_bull_ob.zone_bottom) / 2,
                nearest_bull_ob.zone_top, nearest_bull_ob.zone_bottom,
            ))
        if zone.in_ote:
            candidates.append(((zone.ote_top + zone.ote_bottom) / 2, zone.ote_top, zone.ote_bottom))

        if candidates:
            # Pick the one closest to current price
            candidates.sort(key=lambda c: abs(c[0] - current_price))
            sniper_entry, entry_top, entry_bottom = candidates[0]

    elif smc_bias == "SHORT":
        candidates = []
        if nearest_bear_fvg and not nearest_bear_fvg.filled:
            candidates.append((nearest_bear_fvg.midpoint, nearest_bear_fvg.top, nearest_bear_fvg.bottom))
        if nearest_bear_ob and not nearest_bear_ob.mitigated:
            candidates.append((
                (nearest_bear_ob.zone_top + nearest_bear_ob.zone_bottom) / 2,
                nearest_bear_ob.zone_top, nearest_bear_ob.zone_bottom,
            ))
        if candidates:
            candidates.sort(key=lambda c: abs(c[0] - current_price))
            sniper_entry, entry_top, entry_bottom = candidates[0]

    return SMCResult(
        smc_score=round(score, 3),
        smc_bias=smc_bias,
        entry_zone_top=round(entry_top, 4),
        entry_zone_bottom=round(entry_bottom, 4),
        sniper_entry=round(sniper_entry, 4),
        fvg_count=len([f for f in fvgs if not f.filled]),
        ob_count=len([o for o in obs if not o.mitigated]),
        latest_shift=shift_label,
        zone=zone.zone,
        in_ote=zone.in_ote,
        detail={
            "zone_pct": zone.zone_pct,
            "equilibrium": zone.equilibrium,
            "swing_high": zone.swing_high,
            "swing_low": zone.swing_low,
        },
    )
