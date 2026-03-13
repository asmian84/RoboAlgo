"""Model agreement scorer for trading signals.

Measures directional consensus across multiple independent analysis
engines: pattern detection, trend indicators, cycle phase, and
market force.  Higher agreement = more trustworthy signal.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("signal_confidence_engine.model_agreement")


def compute_model_agreement(symbol: str) -> float:
    """Score multi-model directional agreement for *symbol* on a 0-100 scale.

    Sources
    -------
    1. Pattern engine direction (bullish / bearish / neutral).
    2. Trend direction from latest indicator (price > MA50 > MA200 = bullish).
    3. Cycle phase from FFT / wavelet / Hilbert projection
       (phase < 0.3 = bullish, > 0.7 = bearish, else neutral).
    4. Market force bias from physics engine net force
       (net > 0 = bullish, net < 0 = bearish).

    Each source contributes +1 (bullish), -1 (bearish), or 0 (neutral / error).
    Agreement = abs(sum_of_votes) / number_of_votes, scaled to 0-100.
    """
    sym = symbol.upper()
    votes: list[int] = []

    # ── 1. Pattern engine direction ──────────────────────────────────────
    try:
        from pattern_engine.service import PatternService
        patterns = PatternService().detect_for_symbol(sym, resolution_minutes=0)
        if patterns:
            # Take direction of top-confidence pattern
            top = patterns[0]
            direction = top.get("direction", "neutral")
            if direction == "bullish":
                votes.append(1)
            elif direction == "bearish":
                votes.append(-1)
            else:
                votes.append(0)
        else:
            votes.append(0)
    except Exception:
        votes.append(0)

    # ── 2. Trend direction from latest indicator ─────────────────────────
    try:
        from sqlalchemy import select, desc

        from database.connection import get_session
        from database.models import Indicator, Instrument, PriceData

        with get_session() as session:
            inst = session.execute(
                select(Instrument).where(Instrument.symbol == sym)
            ).scalar_one_or_none()
            if inst is not None:
                ind = session.execute(
                    select(Indicator)
                    .where(Indicator.instrument_id == inst.id)
                    .order_by(desc(Indicator.date))
                    .limit(1)
                ).scalar_one_or_none()

                price_row = session.execute(
                    select(PriceData)
                    .where(PriceData.instrument_id == inst.id)
                    .order_by(desc(PriceData.date))
                    .limit(1)
                ).scalar_one_or_none()

                if ind and price_row and ind.ma50 and ind.ma200:
                    close = float(price_row.close)
                    ma50 = float(ind.ma50)
                    ma200 = float(ind.ma200)
                    if close > ma50 > ma200:
                        votes.append(1)   # bullish
                    elif close < ma50 < ma200:
                        votes.append(-1)  # bearish
                    else:
                        votes.append(0)   # neutral / mixed
                else:
                    votes.append(0)
            else:
                votes.append(0)
    except Exception:
        votes.append(0)

    # ── 3. Cycle phase ───────────────────────────────────────────────────
    try:
        from api.routers.cycles import _fetch_price_data
        from cycle_engine.cycle_projection import project_cycle

        df = _fetch_price_data(sym)
        if not df.empty:
            cycle_data = project_cycle(df)
            phase = cycle_data.get("cycle_phase", 0.5)
            if phase < 0.3:
                votes.append(1)   # bullish (near trough)
            elif phase > 0.7:
                votes.append(-1)  # bearish (near peak)
            else:
                votes.append(0)
        else:
            votes.append(0)
    except Exception:
        votes.append(0)

    # ── 4. Market force bias ─────────────────────────────────────────────
    try:
        from api.routers.cycles import _fetch_price_data as _fetch_pd
        from physics_engine.net_force import compute_net_force

        df = _fetch_pd(sym)
        if not df.empty:
            forces = compute_net_force(df)
            net = forces.get("net_force", 0.0)
            if net > 0:
                votes.append(1)
            elif net < 0:
                votes.append(-1)
            else:
                votes.append(0)
        else:
            votes.append(0)
    except Exception:
        votes.append(0)

    # ── Compute agreement score ──────────────────────────────────────────
    n = len(votes)
    if n == 0:
        return 50.0

    agreement = abs(sum(votes)) / n  # 0.0 to 1.0
    return max(0.0, min(100.0, agreement * 100.0))
