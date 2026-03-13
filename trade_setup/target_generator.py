"""Target Generator — computes profit targets using risk multiples and liquidity zones.

Target cascade
--------------
Target 1 = entry + 1.5 × risk  (in the direction of the trade)
Target 2 = entry + 3.0 × risk
Target 3 = nearest untested liquidity zone in the direction of the trade
           (falls back to entry + 5 × risk if no zone is near)

``risk`` is defined as |entry - stop_loss| (the risk in price terms).

Liquidity zones are consumed from LiquidityMapEngine output — they are NOT
recomputed here.
"""

from __future__ import annotations

from trade_setup.entry_logic import DIRECTION_LONG, DIRECTION_SHORT

# R-multiple milestones for targets 1 and 2
_TARGET_1_R = 1.5
_TARGET_2_R = 3.0
_TARGET_3_R = 5.0   # fallback when no zone available for T3


def generate_targets(
    entry: float,
    stop_loss: float,
    direction: str,
    liquidity_zones: list[dict] | None = None,
    max_targets: int = 3,
) -> list[float]:
    """Generate a list of profit targets.

    Args:
        entry:           Entry price.
        stop_loss:       Stop loss price.
        direction:       ``"LONG"`` or ``"SHORT"``.
        liquidity_zones: Zone list from LiquidityMapEngine (sorted price desc).
                         Used to locate the T3 zone target.
        max_targets:     Maximum number of targets to return (default 3).

    Returns:
        Sorted list of target prices.
        LONG:  ascending order (nearest → farthest).
        SHORT: descending order (nearest → farthest).
    """
    risk = abs(entry - stop_loss)
    if risk <= 0:
        return []

    sign = 1 if direction == DIRECTION_LONG else -1

    t1 = round(entry + sign * _TARGET_1_R * risk, 4)
    t2 = round(entry + sign * _TARGET_2_R * risk, 4)
    t3 = _zone_target(entry, direction, risk, liquidity_zones)

    targets = [t1, t2, t3][:max_targets]

    # Sort so the nearest target is first
    targets.sort(reverse=(direction == DIRECTION_SHORT))
    return targets


def compute_risk_reward(
    entry: float,
    stop_loss: float,
    target: float,
    direction: str,
) -> float:
    """Compute the risk-reward ratio for a given target.

    Args:
        entry:     Entry price.
        stop_loss: Stop loss price.
        target:    One of the profit targets.
        direction: ``"LONG"`` or ``"SHORT"``.

    Returns:
        Risk-reward ratio (e.g. 3.2 means reward is 3.2× the risk).
        Returns 0.0 if risk is zero.
    """
    risk = abs(entry - stop_loss)
    if risk <= 0:
        return 0.0

    if direction == DIRECTION_LONG:
        reward = target - entry
    else:
        reward = entry - target

    return round(max(reward / risk, 0.0), 2)


# ── Internal helpers ───────────────────────────────────────────────────────────

def _zone_target(
    entry: float,
    direction: str,
    risk: float,
    liquidity_zones: list[dict] | None,
) -> float:
    """Find the nearest strong liquidity zone in the trade direction for T3.

    Prefers zones with ``strength >= 0.5`` that are at least 2× risk away
    from entry (so T3 is always beyond T2 = 3R).

    Falls back to entry ± ``_TARGET_3_R × risk`` when no suitable zone exists.
    """
    fallback = round(
        entry + (1 if direction == DIRECTION_LONG else -1) * _TARGET_3_R * risk,
        4,
    )

    if not liquidity_zones:
        return fallback

    min_distance = risk * 2   # must be at least 2× risk away

    if direction == DIRECTION_LONG:
        # Zones above entry, prefer lowest (nearest)
        candidates = [
            z for z in liquidity_zones
            if z["price"] > entry + min_distance
            and z.get("strength", 0) >= 0.5
            and z.get("side") == "high"
        ]
        if candidates:
            return round(min(c["price"] for c in candidates), 4)
    else:
        # Zones below entry, prefer highest (nearest)
        candidates = [
            z for z in liquidity_zones
            if z["price"] < entry - min_distance
            and z.get("strength", 0) >= 0.5
            and z.get("side") == "low"
        ]
        if candidates:
            return round(max(c["price"] for c in candidates), 4)

    return fallback
