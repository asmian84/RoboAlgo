"""
signal_engine/backtest_engine.py
────────────────────────────────────────────────────────────────────────────────
Vectorised backtester for concrete, rule-based strategies.

Each strategy has:
  • clear entry trigger (boolean mask on daily bars)
  • fixed stop / target in ATR units
  • time-limit exit (max holding days)

Returns trade log + equity curve + aggregate stats so we can PROVE
which setups make money before the scanner shows them to the user.

Strategies:
  1. pullback_to_ma   — uptrend pullback to MA20/MA50 support
  2. squeeze_breakout  — BB squeeze → volume breakout
  3. oversold_bounce   — RSI<30 + near BB lower + still above MA200
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RISK_PER_TRADE = 1000.0   # fixed dollar risk per trade for equity curve


# ── Trade record ──────────────────────────────────────────────────────────────

@dataclass
class Trade:
    strategy:    str
    symbol:      str
    entry_date:  str
    entry_price: float
    stop:        float
    target:      float
    exit_date:   str   = ""
    exit_price:  float = 0.0
    exit_reason: str   = ""       # "target" | "stop" | "timeout"
    pnl_pct:     float = 0.0
    pnl_dollar:  float = 0.0
    rr_actual:   float = 0.0
    hold_days:   int   = 0

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy, "symbol": self.symbol,
            "entry_date": self.entry_date, "entry_price": round(self.entry_price, 2),
            "stop": round(self.stop, 2), "target": round(self.target, 2),
            "exit_date": self.exit_date, "exit_price": round(self.exit_price, 2),
            "exit_reason": self.exit_reason,
            "pnl_pct": round(self.pnl_pct, 2),
            "pnl_dollar": round(self.pnl_dollar, 2),
            "rr_actual": round(self.rr_actual, 2),
            "hold_days": self.hold_days,
            "won": self.pnl_pct > 0,
        }


# ── Helper: vectorised ATR ───────────────────────────────────────────────────

def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """14-period ATR as a numpy array, same length as input (first `period` values are NaN)."""
    prev_c = np.roll(close, 1)
    prev_c[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_c), np.abs(low - prev_c)))
    atr = pd.Series(tr).rolling(period).mean().to_numpy()
    return atr


def _rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder RSI as numpy array."""
    delta = np.diff(close, prepend=close[0])
    gain  = np.where(delta > 0, delta, 0.0)
    loss  = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, min_periods=period).mean().to_numpy()
    avg_loss = pd.Series(loss).ewm(alpha=1/period, min_periods=period).mean().to_numpy()
    rs  = avg_gain / np.maximum(avg_loss, 1e-10)
    rsi = 100 - 100 / (1 + rs)
    return rsi


# ── Strategy 1: Pullback to MA ───────────────────────────────────────────────

def _pullback_entries(df: pd.DataFrame, atr_arr: np.ndarray) -> list[int]:
    """
    Buy when:
      • Price is in uptrend (close > MA50)
      • Today's low touches or dips below MA20
      • Close still above MA50 (not a breakdown)
    """
    close = df["Close"].to_numpy()
    low   = df["Low"].to_numpy()
    ma20  = pd.Series(close).rolling(20).mean().to_numpy()
    ma50  = pd.Series(close).rolling(50).mean().to_numpy()

    entries = []
    cooldown = 0
    for i in range(50, len(close)):
        if cooldown > 0:
            cooldown -= 1
            continue
        if np.isnan(ma20[i]) or np.isnan(ma50[i]) or np.isnan(atr_arr[i]):
            continue
        # Uptrend: close above MA50
        if close[i] <= ma50[i]:
            continue
        # Pullback: low dips to/below MA20 (but close still above MA50)
        if low[i] <= ma20[i] * 1.002:
            entries.append(i)
            cooldown = 5   # don't re-enter for 5 bars
    return entries


# ── Strategy 2: Squeeze Breakout ──────────────────────────────────────────────

def _squeeze_entries(df: pd.DataFrame, atr_arr: np.ndarray) -> list[int]:
    """
    Buy when:
      • BB width < its 20-period average (compression)
      • Today's close breaks above upper Bollinger Band
      • Volume > 1.5× its 20-day average
    """
    close  = df["Close"].to_numpy()
    volume = df["Volume"].to_numpy().astype(float)

    ma20   = pd.Series(close).rolling(20).mean().to_numpy()
    std20  = pd.Series(close).rolling(20).std().to_numpy()
    bb_upper = ma20 + 2 * std20
    bb_width = 4 * std20 / np.maximum(ma20, 1e-6)  # normalised width
    avg_width = pd.Series(bb_width).rolling(20).mean().to_numpy()
    avg_vol   = pd.Series(volume).rolling(20).mean().to_numpy()

    entries = []
    cooldown = 0
    for i in range(40, len(close)):
        if cooldown > 0:
            cooldown -= 1
            continue
        if np.isnan(avg_width[i]) or np.isnan(atr_arr[i]):
            continue
        # Squeeze: BB width < average
        if bb_width[i] >= avg_width[i]:
            continue
        # Breakout: close above upper BB
        if close[i] <= bb_upper[i]:
            continue
        # Volume confirmation
        if avg_vol[i] <= 0 or volume[i] < 1.5 * avg_vol[i]:
            continue
        entries.append(i)
        cooldown = 10
    return entries


# ── Strategy 3: Oversold Bounce ───────────────────────────────────────────────

def _oversold_entries(df: pd.DataFrame, atr_arr: np.ndarray) -> list[int]:
    """
    Buy when:
      • RSI(14) < 30 (oversold)
      • Price at or below lower Bollinger Band
      • Still above MA200 (not a secular downtrend / value trap)
    """
    close = df["Close"].to_numpy()
    rsi   = _rsi(close)
    ma200 = pd.Series(close).rolling(200).mean().to_numpy()
    ma20  = pd.Series(close).rolling(20).mean().to_numpy()
    std20 = pd.Series(close).rolling(20).std().to_numpy()
    bb_lower = ma20 - 2 * std20

    entries = []
    cooldown = 0
    for i in range(200, len(close)):
        if cooldown > 0:
            cooldown -= 1
            continue
        if np.isnan(ma200[i]) or np.isnan(atr_arr[i]):
            continue
        # Oversold
        if rsi[i] >= 30:
            continue
        # Near lower BB
        if close[i] > bb_lower[i] * 1.01:
            continue
        # Not in secular downtrend
        if close[i] < ma200[i]:
            continue
        entries.append(i)
        cooldown = 5
    return entries


# ── Simulate trades ──────────────────────────────────────────────────────────

def _simulate(
    df: pd.DataFrame,
    entry_indices: list[int],
    atr_arr: np.ndarray,
    strategy_name: str,
    symbol: str,
    stop_atr: float,      # stop distance in ATR units below entry
    target_atr: float,     # target distance in ATR units above entry
    max_hold: int,         # max bars to hold
) -> list[Trade]:
    """Simulate long trades from entry indices with fixed stop/target."""
    close = df["Close"].to_numpy()
    high  = df["High"].to_numpy()
    low   = df["Low"].to_numpy()
    dates = df.index

    trades: list[Trade] = []
    for idx in entry_indices:
        if idx + 1 >= len(close):
            continue
        entry = float(close[idx])  # enter at today's close
        atr_val = float(atr_arr[idx])
        if atr_val <= 0 or entry <= 0:
            continue

        stop   = entry - stop_atr * atr_val
        target = entry + target_atr * atr_val
        risk   = entry - stop
        if risk <= 0:
            continue

        trade = Trade(
            strategy=strategy_name,
            symbol=symbol,
            entry_date=str(dates[idx].date()) if hasattr(dates[idx], 'date') else str(dates[idx]),
            entry_price=entry,
            stop=stop,
            target=target,
        )

        # Walk forward bar by bar
        for j in range(idx + 1, min(idx + 1 + max_hold, len(close))):
            # Check stop first (conservative — assume worst case)
            if float(low[j]) <= stop:
                trade.exit_price  = stop
                trade.exit_date   = str(dates[j].date()) if hasattr(dates[j], 'date') else str(dates[j])
                trade.exit_reason = "stop"
                trade.hold_days   = j - idx
                break
            # Check target
            if float(high[j]) >= target:
                trade.exit_price  = target
                trade.exit_date   = str(dates[j].date()) if hasattr(dates[j], 'date') else str(dates[j])
                trade.exit_reason = "target"
                trade.hold_days   = j - idx
                break
        else:
            # Timeout — exit at close of last bar
            last_j = min(idx + max_hold, len(close) - 1)
            trade.exit_price  = float(close[last_j])
            trade.exit_date   = str(dates[last_j].date()) if hasattr(dates[last_j], 'date') else str(dates[last_j])
            trade.exit_reason = "timeout"
            trade.hold_days   = last_j - idx

        trade.pnl_pct    = round((trade.exit_price / entry - 1) * 100, 2)
        trade.pnl_dollar = round(RISK_PER_TRADE * (trade.exit_price - entry) / risk, 2)
        trade.rr_actual  = round((trade.exit_price - entry) / risk, 2) if risk > 0 else 0.0
        trades.append(trade)

    return trades


# ── Aggregate stats ──────────────────────────────────────────────────────────

def _stats(trades: list[Trade]) -> dict:
    """Compute aggregate performance metrics from a trade list."""
    if not trades:
        return {
            "total": 0, "wins": 0, "losses": 0, "win_rate": 0,
            "avg_pnl_pct": 0, "avg_win_pct": 0, "avg_loss_pct": 0,
            "profit_factor": 0, "total_pnl": 0, "avg_rr": 0,
            "avg_hold_days": 0, "best_trade": 0, "worst_trade": 0,
            "max_drawdown_pct": 0,
        }

    pnls   = [t.pnl_dollar for t in trades]
    pnl_pcts = [t.pnl_pct for t in trades]
    wins   = [t for t in trades if t.pnl_dollar > 0]
    losses = [t for t in trades if t.pnl_dollar <= 0]

    gross_profit = sum(t.pnl_dollar for t in wins)
    gross_loss   = abs(sum(t.pnl_dollar for t in losses))

    # Equity curve + max drawdown
    equity = np.cumsum(pnls)
    running_max = np.maximum.accumulate(equity)
    drawdowns = equity - running_max
    max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

    return {
        "total":            len(trades),
        "wins":             len(wins),
        "losses":           len(losses),
        "win_rate":         round(len(wins) / len(trades) * 100, 1),
        "avg_pnl_pct":      round(np.mean(pnl_pcts), 2),
        "avg_win_pct":      round(np.mean([t.pnl_pct for t in wins]), 2)   if wins   else 0,
        "avg_loss_pct":     round(np.mean([t.pnl_pct for t in losses]), 2) if losses  else 0,
        "profit_factor":    round(gross_profit / max(gross_loss, 1), 2),
        "total_pnl":        round(sum(pnls), 2),
        "avg_rr":           round(np.mean([t.rr_actual for t in trades]), 2),
        "avg_hold_days":    round(np.mean([t.hold_days for t in trades]), 1),
        "best_trade":       round(max(pnl_pcts), 2),
        "worst_trade":      round(min(pnl_pcts), 2),
        "max_drawdown":     round(max_dd, 2),
    }


def _equity_curve(trades: list[Trade]) -> list[dict]:
    """Build cumulative equity curve for charting."""
    if not trades:
        return []
    cumulative = 0.0
    curve = []
    for t in trades:
        cumulative += t.pnl_dollar
        curve.append({
            "date":   t.exit_date,
            "pnl":    round(t.pnl_dollar, 2),
            "equity": round(cumulative, 2),
            "symbol": t.symbol,
        })
    return curve


# ── Main engine ───────────────────────────────────────────────────────────────

STRATEGIES = {
    "pullback_to_ma": {
        "name":       "Pullback to MA",
        "desc":       "Buy uptrend pullback to 20-day MA (above 50 MA). Stop 1 ATR below, target 2 ATR above.",
        "entry_fn":   _pullback_entries,
        "stop_atr":   1.0,
        "target_atr": 2.0,
        "max_hold":   10,
    },
    "squeeze_breakout": {
        "name":       "Squeeze Breakout",
        "desc":       "Buy BB squeeze breakout on volume spike. Stop at BB mid, target 3 ATR above.",
        "entry_fn":   _squeeze_entries,
        "stop_atr":   1.0,
        "target_atr": 3.0,
        "max_hold":   15,
    },
    "oversold_bounce": {
        "name":       "Oversold Bounce",
        "desc":       "Buy RSI<30 near lower BB, still above 200 MA. Stop 1.5 ATR below, target 2 ATR above.",
        "entry_fn":   _oversold_entries,
        "stop_atr":   1.5,
        "target_atr": 2.0,
        "max_hold":   5,
    },
}


class BacktestEngine:
    """
    Run rule-based strategy backtests on any symbol using yfinance data.

    Usage:
        engine = BacktestEngine()
        result = engine.run("AAPL")                      # all strategies, 1 year
        result = engine.run("TSLA", strategies=["squeeze_breakout"])
        result = engine.run_universe(["AAPL", "NVDA", "TSLA"])
    """

    def run(
        self,
        symbol: str,
        strategies: list[str] | None = None,
        period: str = "2y",
    ) -> dict:
        """
        Backtest one or more strategies on a single symbol.

        Returns:
            {
              symbol, period,
              strategies: {
                strategy_key: { stats, trades, equity_curve }
              },
              combined: { stats, equity_curve }
            }
        """
        import yfinance as yf

        sym = symbol.upper()
        strat_keys = strategies or list(STRATEGIES.keys())

        try:
            hist = yf.Ticker(sym).history(period=period, interval="1d")
            if hist.empty or len(hist) < 60:
                return {"error": f"Insufficient data for {sym}"}
        except Exception as e:
            return {"error": f"Data fetch failed: {e}"}

        h = hist["High"].to_numpy()
        l = hist["Low"].to_numpy()
        c = hist["Close"].to_numpy()
        atr_arr = _atr(h, l, c)

        all_trades: list[Trade] = []
        results: dict = {}

        for key in strat_keys:
            if key not in STRATEGIES:
                continue
            cfg     = STRATEGIES[key]
            entries = cfg["entry_fn"](hist, atr_arr)
            trades  = _simulate(
                hist, entries, atr_arr,
                strategy_name=cfg["name"],
                symbol=sym,
                stop_atr=cfg["stop_atr"],
                target_atr=cfg["target_atr"],
                max_hold=cfg["max_hold"],
            )
            results[key] = {
                "name":         cfg["name"],
                "description":  cfg["desc"],
                "stats":        _stats(trades),
                "trades":       [t.to_dict() for t in trades],
                "equity_curve": _equity_curve(trades),
            }
            all_trades.extend(trades)

        # Sort combined trades by date
        all_trades.sort(key=lambda t: t.entry_date)

        return {
            "symbol":     sym,
            "period":     period,
            "bars":       len(hist),
            "strategies": results,
            "combined": {
                "stats":        _stats(all_trades),
                "equity_curve": _equity_curve(all_trades),
                "trade_count":  len(all_trades),
            },
        }

    def run_universe(
        self,
        symbols: list[str],
        strategies: list[str] | None = None,
        period: str = "2y",
    ) -> dict:
        """Backtest across multiple symbols. Returns per-symbol + aggregate."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        all_trades: list[Trade] = []
        per_symbol: dict = {}

        def _run_one(sym):
            return sym, self.run(sym, strategies, period)

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(_run_one, s) for s in symbols]
            for fut in as_completed(futures, timeout=120):
                sym, result = fut.result()
                if "error" not in result:
                    per_symbol[sym] = {
                        "combined_stats": result["combined"]["stats"],
                        # Per-strategy stats for quality lookup in pounce scanner
                        "per_strategy": {
                            key: strat["stats"]
                            for key, strat in result["strategies"].items()
                        },
                    }
                    # Collect all trades for aggregate
                    for key, strat in result["strategies"].items():
                        for td in strat["trades"]:
                            all_trades.append(Trade(**{
                                k: td[k] for k in [
                                    "strategy", "symbol", "entry_date", "entry_price",
                                    "stop", "target", "exit_date", "exit_price",
                                    "exit_reason", "pnl_pct", "pnl_dollar",
                                    "rr_actual", "hold_days",
                                ]
                            }))

        all_trades.sort(key=lambda t: t.entry_date)

        # Per-strategy aggregate
        strat_agg: dict = {}
        for key in (strategies or list(STRATEGIES.keys())):
            strat_trades = [t for t in all_trades if t.strategy == STRATEGIES.get(key, {}).get("name", key)]
            strat_agg[key] = {
                "name":  STRATEGIES[key]["name"] if key in STRATEGIES else key,
                "stats": _stats(strat_trades),
            }

        return {
            "symbols_tested":  len(per_symbol),
            "total_trades":    len(all_trades),
            "per_symbol":      per_symbol,
            "per_strategy":    strat_agg,
            "combined": {
                "stats":        _stats(all_trades),
                "equity_curve": _equity_curve(all_trades),
            },
        }
