"""Feature stability scorer for trading signals.

Measures how stable key technical indicators have been over recent
bars.  Stable features indicate a consistent market environment
where signal quality is higher.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("signal_confidence_engine.feature_stability")


def compute_feature_stability(symbol: str) -> float:
    """Score feature stability for *symbol* on a 0-100 scale.

    Logic
    -----
    1. Fetch the last 20 indicator rows from the database.
    2. Compute the coefficient of variation (std / mean) for RSI, ATR,
       and MACD histogram.
    3. Lower CoV = more stable = higher score.
    4. score = 100 - min(avg_cov * 200, 80)
    5. Clip to [0, 100].
    """
    import numpy as np
    from sqlalchemy import select, desc

    from database.connection import get_session
    from database.models import Indicator, Instrument

    sym = symbol.upper()

    with get_session() as session:
        inst = session.execute(
            select(Instrument).where(Instrument.symbol == sym)
        ).scalar_one_or_none()
        if inst is None:
            return 50.0

        rows = session.execute(
            select(Indicator)
            .where(Indicator.instrument_id == inst.id)
            .order_by(desc(Indicator.date))
            .limit(20)
        ).scalars().all()

        if len(rows) < 10:
            return 50.0

        # Extract series
        rsi_vals = [float(r.rsi) for r in rows if r.rsi is not None]
        atr_vals = [float(r.atr) for r in rows if r.atr is not None]
        macd_hist_vals = [float(r.macd_histogram) for r in rows if r.macd_histogram is not None]

        covs: list[float] = []

        for vals in (rsi_vals, atr_vals, macd_hist_vals):
            if len(vals) < 5:
                continue
            arr = np.array(vals, dtype=float)
            mean = np.mean(arr)
            std = np.std(arr)
            if abs(mean) > 1e-9:
                covs.append(abs(std / mean))
            else:
                # Mean near zero (e.g. MACD histogram oscillating around 0)
                # Use std relative to max absolute value as proxy
                max_abs = np.max(np.abs(arr))
                if max_abs > 1e-9:
                    covs.append(std / max_abs)
                else:
                    covs.append(0.0)  # all zeros = perfectly stable

        if not covs:
            return 50.0

        avg_cov = float(np.mean(covs))
        score = 100.0 - min(avg_cov * 200.0, 80.0)
        return float(np.clip(score, 0.0, 100.0))
