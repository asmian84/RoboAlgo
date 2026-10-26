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

# A standard deviation / loss total below this is treated as zero, i.e. the
# ratio it would sit under is undefined rather than "very large".
_EPS = 1e-9

# Reported profit factor when there are winning trades but no losing ones.
# That ratio is mathematically infinite; a finite stand-in keeps the stored /
# served value JSON- and DB-safe. compute_fitness saturates at PF = 4.0, so
# any cap >= 4 leaves fitness (and therefore ranking) unchanged.
PROFIT_FACTOR_CAP = 10.0


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
    if "sharpe_undefined_sign" in metrics:
        # Sharpe was undefined (zero return variance) and is reported as 0.0.
        # Before that fix the same case surfaced as +/-1e8 and was clipped to
        # +/-1 here, so an all-win run got full Sharpe credit and an all-loss
        # run got full penalty. Keep exactly that fitness: use the sign of the
        # mean return instead of the (now neutral) reported 0.0, otherwise an
        # all-loss genome would start to outrank a mostly-losing one.
        sharpe_norm = float(np.sign(metrics["sharpe_undefined_sign"]))
    else:
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

    # INERT: `entry_min` is read here but never used below, so entries are NOT
    # filtered by confluence -- every bar is a candidate entry. Changing
    # entry_confluence_min has zero effect on any metric. Left as-is on purpose
    # (wiring it in would change which strategies get evolved/recommended; that
    # is an operator decision). See docs/fitness_findings.md.
    entry_min = genome_params.get("entry_confluence_min", 60)  # noqa: F841
    stop_mult = genome_params.get("stop_atr_mult", 2.0)
    target_mult = genome_params.get("target_atr_mult", 4.0)

    for sym, bars in price_data.items():
        if len(bars) < 50:
            continue

        # NOTE: this loop previously ended with an unconditional `break` in the
        # outer body ("skip to avoid overlapping trades"), so every symbol
        # contributed at most ONE trade regardless of history length. That made
        # sharpe/win_rate/profit_factor/trade_count rest on ~1 observation per
        # symbol and let compute_fitness's trade_frequency term (0.0 for < 5
        # trades) be driven by that artifact rather than by the genome.
        # Non-overlap is now enforced with a cursor: after a closed trade we
        # resume at the bar after its exit; otherwise we advance one bar.
        i = 50
        while i < len(bars) - 20:
            bar = bars[i]
            # Simulate entry decision (simplified)
            close = float(bar.get("close", 0))
            atr = float(bar.get("atr", close * 0.02))

            if atr <= 0 or close <= 0:
                i += 1
                continue

            # Check if future bars hit target or stop
            stop = close - stop_mult * atr
            target = close + target_mult * atr

            exit_idx = None
            for j in range(i + 1, min(i + genome_params.get("hold_days_max", 20), len(bars))):
                future_close = float(bars[j].get("close", 0))
                if future_close >= target:
                    trades.append((target - close) / close)
                    exit_idx = j
                    break
                elif future_close <= stop:
                    trades.append((stop - close) / close)
                    exit_idx = j
                    break

            # Closed trade: next entry must come after its exit bar (no overlap).
            # No exit triggered: step to the next bar.
            i = exit_idx + 1 if exit_idx is not None else i + 1

    if not trades:
        return {"sharpe_ratio": 0, "win_rate": 0, "profit_factor": 0, "max_drawdown": 0, "trade_count": 0}

    trades_arr = np.array(trades)
    wins = trades_arr[trades_arr > 0]
    losses = trades_arr[trades_arr <= 0]

    # Exits are fixed ATR multiples, so every trade return is one of only two
    # values (target-hit or stop-hit). A run where all trades share one outcome
    # (or a single trade) has zero variance, so Sharpe is UNDEFINED, not
    # infinite. Dividing by max(std, 1e-9) used to manufacture a meaningless
    # +/-1e8-scale number (e.g. -634980314.6555 for 13 straight stop-outs).
    # Report 0.0 and flag the direction so compute_fitness can keep its scoring.
    mean_ret = float(np.mean(trades_arr))
    std_ret = float(np.std(trades_arr))
    sharpe_undefined = std_ret < _EPS
    sharpe = 0.0 if sharpe_undefined else mean_ret / std_ret * float(np.sqrt(252))

    win_rate = float(len(wins) / len(trades_arr))
    gross_profit = float(wins.sum()) if len(wins) > 0 else 0.0
    gross_loss = float(abs(losses.sum())) if len(losses) > 0 else 0.0
    # Same pattern as Sharpe: with no losing trades the ratio's denominator is
    # 0 and the old 1e-9 floor produced a ~1e7-1e9 "profit factor".
    if gross_loss < _EPS:
        profit_factor = PROFIT_FACTOR_CAP if gross_profit > 0 else 0.0
    else:
        profit_factor = gross_profit / gross_loss

    # Max drawdown
    cumulative = np.cumsum(trades_arr)
    peak = np.maximum.accumulate(cumulative)
    drawdown = peak - cumulative
    max_dd = float(drawdown.max()) if len(drawdown) > 0 else 0.0

    result: dict[str, float] = {
        "sharpe_ratio": round(sharpe, 4),
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 4),
        "max_drawdown": round(max_dd, 4),
        "trade_count": len(trades),
    }
    if sharpe_undefined:
        # Only present when Sharpe is undefined; consumed by compute_fitness.
        result["sharpe_undefined_sign"] = float(np.sign(mean_ret))
    return result
