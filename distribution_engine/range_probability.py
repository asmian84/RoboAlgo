"""
RoboAlgo — Range Probability & Master Price Distribution Forecast

Computes probability of reaching specific price targets using both
quantile regression and Monte Carlo simulation, and provides a
blended master forecast.
"""

import pandas as pd


def compute_range_probability(
    df: pd.DataFrame,
    targets: list[float] | None = None,
) -> dict:
    """Compute probability of reaching specific price targets.

    Combines Monte Carlo path simulation with quantile-based estimation
    to determine the likelihood of touching each target price.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns [date, open, high, low, close, volume].
    targets : list[float] | None
        Specific price targets to evaluate. If None, auto-generates
        targets at -10%, -5%, 0%, +5%, +10% of the current price.

    Returns
    -------
    dict
        Per-target probability of reaching the price, ATR, and daily range %.
    """
    import numpy as np
    from .monte_carlo import monte_carlo_forecast

    close = df["close"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)

    current_price = float(close[-1])

    # Auto-generate targets if not provided
    if targets is None:
        targets = [
            round(current_price * m, 4)
            for m in [0.90, 0.95, 1.00, 1.05, 1.10]
        ]

    # Compute ATR (14-period) for context
    if len(close) >= 15:
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1]),
            ),
        )
        atr = float(np.mean(tr[-14:]))
    else:
        atr = float(np.mean(high - low)) if len(close) > 0 else 0.0

    # Daily range %
    daily_ranges = (high - low) / close
    daily_range_pct = float(np.mean(daily_ranges[-20:])) if len(daily_ranges) >= 20 else float(np.mean(daily_ranges))

    # Run Monte Carlo to get full path matrix
    mc_result = monte_carlo_forecast(df, horizon_days=20, n_simulations=1000)

    # Re-generate paths for touch analysis (need the full path matrix)
    log_returns = np.log(close[1:] / close[:-1])
    mu = float(np.mean(log_returns))
    sigma = float(np.std(log_returns, ddof=1))

    rng = np.random.default_rng(42)
    Z = rng.standard_normal((1000, 20))
    drift = mu - 0.5 * sigma ** 2
    daily_increments = np.exp(drift + sigma * Z)

    paths = np.empty((1000, 21), dtype=np.float64)
    paths[:, 0] = current_price
    paths[:, 1:] = current_price * np.cumprod(daily_increments, axis=1)

    # Evaluate each target
    target_results = []
    for target in targets:
        if target >= current_price:
            direction = "above"
            # Fraction of paths where max price along path >= target
            path_max = np.max(paths, axis=1)
            p_reach = float(np.mean(path_max >= target))

            # Median days to reach: among paths that reach, find first crossing
            reached_mask = path_max >= target
            if np.any(reached_mask):
                reached_paths = paths[reached_mask]
                # First index where price >= target for each path
                crossings = np.argmax(reached_paths >= target, axis=1)
                days_median = float(np.median(crossings))
            else:
                days_median = float("nan")
        else:
            direction = "below"
            # Fraction of paths where min price along path <= target
            path_min = np.min(paths, axis=1)
            p_reach = float(np.mean(path_min <= target))

            # Median days to reach
            reached_mask = path_min <= target
            if np.any(reached_mask):
                reached_paths = paths[reached_mask]
                crossings = np.argmax(reached_paths <= target, axis=1)
                days_median = float(np.median(crossings))
            else:
                days_median = float("nan")

        target_results.append({
            "price": round(target, 4),
            "probability_reach": round(p_reach, 4),
            "direction": direction,
            "days_to_reach_median": round(days_median, 2) if not np.isnan(days_median) else None,
        })

    return {
        "current_price": current_price,
        "targets": target_results,
        "atr": round(atr, 4),
        "daily_range_pct": round(daily_range_pct, 6),
    }


def forecast_price_distribution(df: pd.DataFrame, horizon_days: int = 20) -> dict:
    """Master forecast combining quantile and Monte Carlo methods.

    Blends quantile-based and Monte Carlo price forecasts to produce
    a single consensus distribution, including up/down probabilities
    and representative simulated paths.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns [date, open, high, low, close, volume].
    horizon_days : int
        Forecast horizon in trading days (default 20).

    Returns
    -------
    dict
        Blended forecast with quantile levels, probabilities, volatility,
        and representative price paths.
    """
    from .quantile_model import compute_quantile_forecast
    from .monte_carlo import monte_carlo_forecast

    quantile = compute_quantile_forecast(df, horizon_days)
    mc = monte_carlo_forecast(df, horizon_days)

    # Blend: average the quantile estimates from both methods
    blended = {
        "symbol": "",  # filled by caller
        "horizon_days": horizon_days,
        "current_price": quantile["current_price"],
        "expected_price": round((quantile["expected_price"] + mc["expected_price"]) / 2, 4),
        "p10": round((quantile["p10"] + mc["p10"]) / 2, 4),
        "p25": round((quantile["p25"] + mc["p25"]) / 2, 4),
        "p50": round((quantile["p50"] + mc["p50"]) / 2, 4),
        "p75": round((quantile["p75"] + mc["p75"]) / 2, 4),
        "p90": round((quantile["p90"] + mc["p90"]) / 2, 4),
        "probability_up": mc["probability_up"],
        "probability_down": mc["probability_down"],
        "annualized_vol": quantile["annualized_vol"],
        "daily_vol": quantile["daily_vol"],
        "paths_summary": mc.get("paths_summary", []),
        "method": "quantile_mc_blend",
    }
    return blended
