"""
RoboAlgo — Quantile-Based Price Forecast Engine

Computes forward price distributions using historical return quantiles
scaled by rolling volatility.
"""

import pandas as pd


def compute_quantile_forecast(df: pd.DataFrame, horizon_days: int = 20) -> dict:
    """Compute quantile-based price forecasts from historical returns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns [date, open, high, low, close, volume].
    horizon_days : int
        Forecast horizon in trading days (default 20).

    Returns
    -------
    dict
        Quantile-based price forecast including p10 through p90 levels,
        expected price, and volatility estimates.
    """
    import numpy as np

    close = df["close"].values.astype(np.float64)
    if len(close) < 22:
        current = float(close[-1]) if len(close) > 0 else 0.0
        return {
            "current_price": current,
            "horizon_days": horizon_days,
            "expected_price": current,
            "p10": current, "p25": current, "p50": current,
            "p75": current, "p90": current,
            "annualized_vol": 0.0,
            "daily_vol": 0.0,
        }

    # Log returns (vectorized)
    log_returns = np.log(close[1:] / close[:-1])

    # Rolling volatility (20-bar window) — use the last window for current vol
    window = 20
    if len(log_returns) >= window:
        # Vectorized rolling std via stride tricks
        shape = (len(log_returns) - window + 1, window)
        strides = (log_returns.strides[0], log_returns.strides[0])
        rolling = np.lib.stride_tricks.as_strided(log_returns, shape=shape, strides=strides)
        rolling_vol = np.std(rolling, axis=1, ddof=1)
        daily_vol = float(rolling_vol[-1])
    else:
        daily_vol = float(np.std(log_returns, ddof=1))

    annualized_vol = daily_vol * np.sqrt(252)

    # Scale returns by sqrt(horizon_days) for time-scaling
    scaled_returns = log_returns * np.sqrt(horizon_days)

    # Compute quantiles of scaled return distribution
    quantiles = np.quantile(scaled_returns, [0.10, 0.25, 0.50, 0.75, 0.90])

    current_price = float(close[-1])

    # Convert from log returns to absolute prices
    p10 = current_price * np.exp(float(quantiles[0]))
    p25 = current_price * np.exp(float(quantiles[1]))
    p50 = current_price * np.exp(float(quantiles[2]))
    p75 = current_price * np.exp(float(quantiles[3]))
    p90 = current_price * np.exp(float(quantiles[4]))

    return {
        "current_price": current_price,
        "horizon_days": horizon_days,
        "expected_price": p50,
        "p10": round(p10, 4),
        "p25": round(p25, 4),
        "p50": round(p50, 4),
        "p75": round(p75, 4),
        "p90": round(p90, 4),
        "annualized_vol": round(annualized_vol, 6),
        "daily_vol": round(daily_vol, 6),
    }
