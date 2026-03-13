"""Position Sizer — computes position size from account risk parameters.

Formula
-------
    risk_amount   = account_size × risk_percent
    position_size = risk_amount / (entry_price − stop_loss)   # shares / contracts

All sizing is based on fixed-fractional risk management:
    • ``risk_percent`` controls the fraction of the account risked per trade.
    • ``max_position_pct`` caps position size as a fraction of account equity
      (prevents over-sizing on very tight stops).

The position is capped so that position_size × entry_price ≤ account_size × max_position_pct.

IMPORTANT: No external engine calls.  This is pure arithmetic using inputs
supplied by setup_engine.py.
"""

from __future__ import annotations

# Defaults
_DEFAULT_RISK_PERCENT    = 0.01    # 1% of account per trade
_DEFAULT_MAX_POSITION    = 0.20    # max 20% of account in a single position
_DEFAULT_ACCOUNT_SIZE    = 100_000.0


def compute_position_size(
    entry: float,
    stop_loss: float,
    account_size: float = _DEFAULT_ACCOUNT_SIZE,
    risk_percent: float = _DEFAULT_RISK_PERCENT,
    max_position_pct: float = _DEFAULT_MAX_POSITION,
) -> dict:
    """Calculate position size and associated risk metrics.

    Args:
        entry:            Entry price per share / unit.
        stop_loss:        Stop loss price per share / unit.
        account_size:     Total account equity in currency units.
        risk_percent:     Fraction of account to risk per trade (e.g. 0.01 = 1%).
        max_position_pct: Maximum fraction of account in one position (e.g. 0.20).

    Returns:
        dict::

            {
                "position_size":      float,  # number of shares / units
                "risk_amount":        float,  # dollar risk
                "position_value":     float,  # position_size × entry
                "position_pct":       float,  # position_value / account_size
                "risk_per_share":     float,  # |entry - stop_loss|
                "risk_percent_used":  float,  # actual risk% after cap
            }
    """
    risk_per_share = abs(entry - stop_loss)

    if risk_per_share <= 0 or entry <= 0 or account_size <= 0:
        return _zero_size()

    risk_amount = account_size * risk_percent
    raw_size    = risk_amount / risk_per_share

    # Apply max position cap
    max_size_by_value = (account_size * max_position_pct) / entry
    position_size     = min(raw_size, max_size_by_value)
    position_size     = max(round(position_size, 2), 0.0)

    position_value  = round(position_size * entry, 2)
    actual_risk     = round(position_size * risk_per_share, 2)
    risk_pct_used   = round(actual_risk / account_size, 6)
    position_pct    = round(position_value / account_size, 4)

    return {
        "position_size":     position_size,
        "risk_amount":       actual_risk,
        "position_value":    position_value,
        "position_pct":      position_pct,
        "risk_per_share":    round(risk_per_share, 4),
        "risk_percent_used": risk_pct_used,
    }


def _zero_size() -> dict:
    return {
        "position_size":     0.0,
        "risk_amount":       0.0,
        "position_value":    0.0,
        "position_pct":      0.0,
        "risk_per_share":    0.0,
        "risk_percent_used": 0.0,
    }
