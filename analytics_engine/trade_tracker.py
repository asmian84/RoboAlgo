"""
analytics_engine/trade_tracker.py
────────────────────────────────────────────────────────────────────────────────
Trade Performance Tracker.

Tracks trades taken, computes:
  • win rate
  • expectancy (EV per trade in $)
  • average win / average loss
  • profit factor
  • equity curve
  • performance by GEX regime (positive vs negative gamma)
  • performance by signal confidence level

Storage: JSON file (simple, persistent, no DB dependency)
         Plugs into PostgreSQL when ready via TradeLifecycle model.

Data model per trade:
  {
    "id":           str (uuid4),
    "symbol":       str,
    "signal_type":  "long" | "short",
    "confidence":   "high" | "medium" | "low",
    "score":        float,
    "entry_price":  float,
    "exit_price":   float | None,
    "shares":       int,
    "stop_price":   float,
    "target_price": float,
    "dollar_risk":  float,
    "gex_regime":   str,         # POSITIVE | NEGATIVE | NEUTRAL
    "catalyst_type": str,
    "opened_at":    str (ISO),
    "closed_at":    str | None,
    "pnl":          float | None,
    "status":       "open" | "won" | "lost" | "scratch",
    "exit_reason":  str,         # "stop" | "target" | "manual" | ""
  }
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_STORE = Path(__file__).parent.parent / "data" / "trades.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TradeTracker:
    """
    Lightweight trade journal with performance analytics.

    All trades are persisted to a JSON file.  The in-memory list is the
    source of truth for analytics; file is written on every mutation.
    """

    def __init__(self, store_path: str | Path | None = None):
        self.store_path = Path(store_path or _DEFAULT_STORE)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self._trades: list[dict] = self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> list[dict]:
        if self.store_path.exists():
            try:
                with open(self.store_path) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning("Could not load trades: %s", e)
        return []

    def _save(self) -> None:
        with open(self.store_path, "w") as f:
            json.dump(self._trades, f, indent=2)

    # ── Trade lifecycle ───────────────────────────────────────────────────────

    def open_trade(
        self,
        symbol:       str,
        signal_type:  str,
        confidence:   str,
        score:        float,
        entry_price:  float,
        shares:       int,
        stop_price:   float,
        target_price: float,
        dollar_risk:  float,
        gex_regime:   str = "NEUTRAL",
        catalyst_type: str = "none",
        **kwargs,
    ) -> str:
        """Record a new trade. Returns trade_id."""
        trade_id = str(uuid.uuid4())[:8]
        trade = {
            "id":            trade_id,
            "symbol":        symbol.upper(),
            "signal_type":   signal_type,
            "confidence":    confidence,
            "score":         round(score, 4),
            "entry_price":   entry_price,
            "exit_price":    None,
            "shares":        shares,
            "stop_price":    stop_price,
            "target_price":  target_price,
            "dollar_risk":   dollar_risk,
            "gex_regime":    gex_regime,
            "catalyst_type": catalyst_type,
            "opened_at":     _now(),
            "closed_at":     None,
            "pnl":           None,
            "status":        "open",
            "exit_reason":   "",
        }
        trade.update(kwargs)
        self._trades.append(trade)
        self._save()
        logger.info("Opened trade %s: %s %s @ $%.2f", trade_id, signal_type, symbol, entry_price)
        return trade_id

    def close_trade(
        self,
        trade_id:    str,
        exit_price:  float,
        exit_reason: str = "manual",
    ) -> dict | None:
        """Close a trade and compute P&L."""
        trade = self._get(trade_id)
        if trade is None:
            logger.warning("Trade %s not found", trade_id)
            return None
        if trade["status"] != "open":
            logger.warning("Trade %s already closed", trade_id)
            return trade

        entry  = trade["entry_price"]
        shares = trade["shares"]
        mult   = 1 if trade["signal_type"] == "long" else -1
        pnl    = (exit_price - entry) * shares * mult

        trade["exit_price"]  = exit_price
        trade["closed_at"]   = _now()
        trade["pnl"]         = round(pnl, 2)
        trade["exit_reason"] = exit_reason

        if abs(pnl) < 0.01 * trade.get("dollar_risk", 1):
            trade["status"] = "scratch"
        elif pnl > 0:
            trade["status"] = "won"
        else:
            trade["status"] = "lost"

        self._save()
        logger.info("Closed trade %s: PnL = $%.2f (%s)", trade_id, pnl, trade["status"])
        return trade

    def update_trade(self, trade_id: str, **fields) -> None:
        """Update arbitrary fields on a trade."""
        trade = self._get(trade_id)
        if trade:
            trade.update(fields)
            self._save()

    def _get(self, trade_id: str) -> dict | None:
        for t in self._trades:
            if t["id"] == trade_id:
                return t
        return None

    # ── Queries ───────────────────────────────────────────────────────────────

    @property
    def open_trades(self) -> list[dict]:
        return [t for t in self._trades if t["status"] == "open"]

    @property
    def closed_trades(self) -> list[dict]:
        return [t for t in self._trades if t["status"] in ("won", "lost", "scratch")]

    def get_all(self) -> list[dict]:
        return list(self._trades)

    # ── Analytics ─────────────────────────────────────────────────────────────

    def compute_stats(self, trades: list[dict] | None = None) -> dict[str, Any]:
        """
        Compute full performance statistics.

        Returns:
          win_rate, expectancy, avg_win, avg_loss, profit_factor,
          total_pnl, max_drawdown, equity_curve, sharpe_ratio
        """
        closed = trades if trades is not None else self.closed_trades
        if not closed:
            return self._empty_stats()

        pnls     = [t["pnl"] for t in closed if t.get("pnl") is not None]
        wins     = [p for p in pnls if p > 0]
        losses   = [p for p in pnls if p < 0]
        scratches = [p for p in pnls if p == 0]

        n         = len(pnls)
        win_rate  = len(wins) / n if n > 0 else 0.0
        avg_win   = float(np.mean(wins))   if wins   else 0.0
        avg_loss  = float(np.mean(losses)) if losses else 0.0
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

        profit_factor = (
            abs(sum(wins)) / abs(sum(losses))
            if losses and sum(losses) != 0
            else float("inf")
        )

        # Equity curve
        equity_curve = list(np.cumsum(pnls).tolist())

        # Max drawdown
        peak        = float("-inf")
        max_dd      = 0.0
        running_pnl = 0.0
        for p in pnls:
            running_pnl += p
            peak = max(peak, running_pnl)
            dd   = peak - running_pnl
            max_dd = max(max_dd, dd)

        # Sharpe (simplified — daily PnL as returns)
        if len(pnls) >= 2:
            sharpe = (float(np.mean(pnls)) / float(np.std(pnls)) * np.sqrt(252)
                      if np.std(pnls) > 0 else 0.0)
        else:
            sharpe = 0.0

        return {
            "total_trades":   n,
            "wins":           len(wins),
            "losses":         len(losses),
            "scratches":      len(scratches),
            "win_rate":       round(win_rate, 4),
            "avg_win":        round(avg_win, 2),
            "avg_loss":       round(avg_loss, 2),
            "expectancy":     round(expectancy, 2),
            "profit_factor":  round(profit_factor, 3),
            "total_pnl":      round(sum(pnls), 2),
            "max_drawdown":   round(max_dd, 2),
            "sharpe_ratio":   round(sharpe, 3),
            "equity_curve":   [round(v, 2) for v in equity_curve],
        }

    def stats_by_regime(self) -> dict[str, dict]:
        """Break down performance by GEX regime."""
        regimes = set(t.get("gex_regime", "NEUTRAL") for t in self.closed_trades)
        return {
            regime: self.compute_stats([
                t for t in self.closed_trades
                if t.get("gex_regime") == regime
            ])
            for regime in regimes
        }

    def stats_by_confidence(self) -> dict[str, dict]:
        """Break down performance by signal confidence level."""
        levels = ("high", "medium", "low")
        return {
            level: self.compute_stats([
                t for t in self.closed_trades
                if t.get("confidence") == level
            ])
            for level in levels
        }

    def stats_by_symbol(self) -> dict[str, dict]:
        """Per-symbol performance breakdown."""
        symbols = set(t["symbol"] for t in self.closed_trades)
        return {
            sym: self.compute_stats([
                t for t in self.closed_trades if t["symbol"] == sym
            ])
            for sym in sorted(symbols)
        }

    @staticmethod
    def _empty_stats() -> dict[str, Any]:
        return {
            "total_trades": 0, "wins": 0, "losses": 0, "scratches": 0,
            "win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
            "expectancy": 0.0, "profit_factor": 0.0, "total_pnl": 0.0,
            "max_drawdown": 0.0, "sharpe_ratio": 0.0, "equity_curve": [],
        }

    def report(self) -> str:
        """Human-readable performance report."""
        stats = self.compute_stats()
        lines = [
            "=" * 50,
            "ROBOALGO PERFORMANCE REPORT",
            "=" * 50,
            f"Total Trades : {stats['total_trades']}",
            f"Win Rate     : {stats['win_rate']:.1%}",
            f"Avg Win      : ${stats['avg_win']:,.2f}",
            f"Avg Loss     : ${stats['avg_loss']:,.2f}",
            f"Expectancy   : ${stats['expectancy']:,.2f} per trade",
            f"Profit Factor: {stats['profit_factor']:.2f}",
            f"Total P&L    : ${stats['total_pnl']:,.2f}",
            f"Max Drawdown : ${stats['max_drawdown']:,.2f}",
            f"Sharpe Ratio : {stats['sharpe_ratio']:.2f}",
            "",
            "Open Trades  :",
        ]
        for t in self.open_trades:
            lines.append(f"  {t['symbol']:6} {t['signal_type']:5} @ ${t['entry_price']}")
        lines.append("=" * 50)
        return "\n".join(lines)


# ── Example usage ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile
    tracker = TradeTracker(store_path=tempfile.mktemp(suffix=".json"))

    # Open 3 trades
    id1 = tracker.open_trade("NVDA", "long",  "high",   0.88, 875.0, 5,  840.0, 945.0, 175.0, "NEGATIVE")
    id2 = tracker.open_trade("SPY",  "long",  "medium", 0.74, 502.0, 10, 494.0, 518.0, 80.0,  "POSITIVE")
    id3 = tracker.open_trade("QQQ",  "short", "medium", 0.71, 435.0, 8,  443.0, 417.0, 64.0,  "NEGATIVE")

    # Close them with outcomes
    tracker.close_trade(id1, 940.0, "target")    # win
    tracker.close_trade(id2, 494.0, "stop")      # loss
    tracker.close_trade(id3, 419.0, "target")    # win (short)

    print(tracker.report())
    print("\nBy GEX Regime:")
    for regime, stats in tracker.stats_by_regime().items():
        print(f"  {regime}: {stats['win_rate']:.0%} win rate | EV=${stats['expectancy']:.2f}")
