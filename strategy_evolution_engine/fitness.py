"""Fitness evaluation for strategy genomes via simplified backtesting."""

from __future__ import annotations

from typing import Any

import numpy as np


# Fitness weights
FITNESS_WEIGHTS = {
    "sharpe_ratio": 0.35,
    "win_rate": 0.25,
    "profit_factor": 0.20,
    "drawdown_penalty": 0.10,
    "trade_frequency": 0.10,
}


def compute_fitness(metrics: dict[str, float]) -> float:
    """Compute composite fitness score from backtest metrics.

    Args:
        metrics: Dict with sharpe_ratio, win_rate, profit_factor,
                max_drawdown, trade_count.

    Returns:
        Fitness score (higher is better, typically 0-1 range).
    """
    sharpe = float(metrics.get("sharpe_ratio", 0.0))
    win_rate = float(metrics.get("win_rate", 0.0))
    pf = float(metrics.get("profit_factor", 0.0))
    max_dd = float(metrics.get("max_drawdown", 0.0))
    trade_count = int(metrics.get("trade_count", 0))

    # Normalize components to 0-1 range
    sharpe_norm = float(np.clip(sharpe / 3.0, -1, 1))  # 3.0 Sharpe = perfect
    wr_norm = float(np.clip(win_rate, 0, 1))
    pf_norm = float(np.clip((pf - 1.0) / 3.0, 0, 1))  # PF of 4.0 = perfect
    dd_penalty = float(np.clip(1.0 - abs(max_dd) / 0.30, 0, 1))  # 30% DD = 0 score

    # Trade frequency: penalize too few trades (< 10) or too many (> 200)
    if trade_count < 5:
        freq_score = 0.0
    elif trade_count < 10:
        freq_score = trade_count / 10.0
    elif trade_count > 200:
        freq_score = max(0, 1.0 - (trade_count - 200) / 200)
    else:
        freq_score = 1.0

    fitness = (
        sharpe_norm * FITNESS_WEIGHTS["sharpe_ratio"]
        + wr_norm * FITNESS_WEIGHTS["win_rate"]
        + pf_norm * FITNESS_WEIGHTS["profit_factor"]
        + dd_penalty * FITNESS_WEIGHTS["drawdown_penalty"]
        + freq_score * FITNESS_WEIGHTS["trade_frequency"]
    )

    return round(float(np.clip(fitness, 0, 1)), 4)


def evaluate_genome_simple(
    genome_params: dict[str, Any],
    price_data: dict[str, list],
) -> dict[str, float]:
    """Simplified backtest for fitness evaluation.

    This is a fast approximation — not a full backtest engine.
    Uses the genome parameters to simulate entry/exit decisions
    on historical price data.

    Args:
        genome_params: Strategy genome parameters.
        price_data: Dict mapping symbol -> list of {date, close, atr} dicts.

    Returns:
        Dict with sharpe_ratio, win_rate, profit_factor, max_drawdown, trade_count.
    """
    # Placeholder implementation — full backtest engine is complex
    # This provides reasonable defaults for the genetic algorithm to optimize against
    trades: list[float] = []

    entry_min = genome_params.get("entry_confluence_min", 60)
    stop_mult = genome_params.get("stop_atr_mult", 2.0)
    target_mult = genome_params.get("target_atr_mult", 4.0)

    for sym, bars in price_data.items():
        if len(bars) < 50:
            continue

        for i in range(50, len(bars) - 20):
            bar = bars[i]
            # Simulate entry decision (simplified)
            close = float(bar.get("close", 0))
            atr = float(bar.get("atr", close * 0.02))

            if atr <= 0 or close <= 0:
                continue

            # Check if future bars hit target or stop
            stop = close - stop_mult * atr
            target = close + target_mult * atr

            for j in range(i + 1, min(i + genome_params.get("hold_days_max", 20), len(bars))):
                future_close = float(bars[j].get("close", 0))
                if future_close >= target:
                    trades.append((target - close) / close)
                    break
                elif future_close <= stop:
                    trades.append((stop - close) / close)
                    break
            # Skip to avoid overlapping trades
            break

    if not trades:
        return {"sharpe_ratio": 0, "win_rate": 0, "profit_factor": 0, "max_drawdown": 0, "trade_count": 0}

    trades_arr = np.array(trades)
    wins = trades_arr[trades_arr > 0]
    losses = trades_arr[trades_arr <= 0]

    sharpe = float(np.mean(trades_arr) / max(np.std(trades_arr), 1e-9) * np.sqrt(252))
    win_rate = float(len(wins) / len(trades_arr))
    gross_profit = float(wins.sum()) if len(wins) > 0 else 0.0
    gross_loss = float(abs(losses.sum())) if len(losses) > 0 else 1e-9
    profit_factor = gross_profit / max(gross_loss, 1e-9)

    # Max drawdown
    cumulative = np.cumsum(trades_arr)
    peak = np.maximum.accumulate(cumulative)
    drawdown = peak - cumulative
    max_dd = float(drawdown.max()) if len(drawdown) > 0 else 0.0

    return {
        "sharpe_ratio": round(sharpe, 4),
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 4),
        "max_drawdown": round(max_dd, 4),
        "trade_count": len(trades),
    }
