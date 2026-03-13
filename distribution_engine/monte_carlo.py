"""
RoboAlgo — Monte Carlo Price Path Simulation

Generates forward price distributions via Geometric Brownian Motion
with drift and volatility fitted from historical returns.
"""

import pandas as pd


def monte_carlo_forecast(
    df: pd.DataFrame,
    horizon_days: int = 20,
    n_simulations: int = 1000,
) -> dict:
    """Monte Carlo price path simulation for probability distributions.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns [date, open, high, low, close, volume].
    horizon_days : int
        Forecast horizon in trading days (default 20).
    n_simulations : int
        Number of simulated price paths (default 1000).

    Returns
    -------
    dict
        Monte Carlo forecast including quantile prices, up/down probabilities,
        min/max prices, and 5 representative paths (p10/p25/p50/p75/p90).
    """
    import numpy as np

    close = df["close"].values.astype(np.float64)
    if len(close) < 2:
        current = float(close[-1]) if len(close) > 0 else 0.0
        return {
            "current_price": current,
            "horizon_days": horizon_days,
            "n_simulations": n_simulations,
            "expected_price": current,
            "p10": current, "p25": current, "p50": current,
            "p75": current, "p90": current,
            "probability_up": 0.5,
            "probability_down": 0.5,
            "max_price": current,
            "min_price": current,
            "paths_summary": [],
        }

    # Daily log returns
    log_returns = np.log(close[1:] / close[:-1])

    # Fit drift (mu) and volatility (sigma) from returns
    mu = float(np.mean(log_returns))
    sigma = float(np.std(log_returns, ddof=1))

    current_price = float(close[-1])

    # Reproducible RNG
    rng = np.random.default_rng(42)

    # Generate all random shocks at once: (n_simulations, horizon_days)
    Z = rng.standard_normal((n_simulations, horizon_days))

    # GBM increments: S(t+dt) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
    # dt = 1 (daily)
    drift = mu - 0.5 * sigma ** 2
    daily_increments = np.exp(drift + sigma * Z)

    # Build price paths: cumulative product along time axis
    # paths shape: (n_simulations, horizon_days + 1) including starting price
    paths = np.empty((n_simulations, horizon_days + 1), dtype=np.float64)
    paths[:, 0] = current_price
    paths[:, 1:] = current_price * np.cumprod(daily_increments, axis=1)

    # Final prices (last column)
    final_prices = paths[:, -1]

    # Quantiles of final price distribution
    quantiles = np.quantile(final_prices, [0.10, 0.25, 0.50, 0.75, 0.90])

    # Probabilities
    probability_up = float(np.mean(final_prices > current_price))
    probability_down = float(np.mean(final_prices < current_price))

    # Representative paths: find the path whose final price is closest to each quantile
    target_quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]
    paths_summary = []
    for q in target_quantiles:
        target_price = float(np.quantile(final_prices, q))
        idx = int(np.argmin(np.abs(final_prices - target_price)))
        paths_summary.append(paths[idx].tolist())

    return {
        "current_price": current_price,
        "horizon_days": horizon_days,
        "n_simulations": n_simulations,
        "expected_price": round(float(np.mean(final_prices)), 4),
        "p10": round(float(quantiles[0]), 4),
        "p25": round(float(quantiles[1]), 4),
        "p50": round(float(quantiles[2]), 4),
        "p75": round(float(quantiles[3]), 4),
        "p90": round(float(quantiles[4]), 4),
        "probability_up": round(probability_up, 4),
        "probability_down": round(probability_down, 4),
        "max_price": round(float(np.max(final_prices)), 4),
        "min_price": round(float(np.min(final_prices)), 4),
        "paths_summary": paths_summary,
    }
