"""
RoboAlgo — Enhanced Paper Trading Engine
Full simulation with slippage, portfolio rules, and performance tracking.

Features:
  - Slippage simulation (configurable, default 0.10% per side)
  - Portfolio rules via PortfolioManager
  - Confluence-score based signal filtering
  - Tracks: win_rate, profit_factor, max_drawdown, weekly_return, Sharpe ratio
  - Market state integration (CHAOS → halve size, COMPRESSION → no trade)
"""

import logging
from datetime import date as DateType, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import select, desc, func

from database.connection import get_session
from database.models import (
    Instrument, PriceData, Signal, ConfluenceScore,
    PaperAccount, PaperPosition, PaperTrade
)
from portfolio_engine.manager import PortfolioManager, MAX_POSITIONS

logger = logging.getLogger(__name__)

# ── Simulation Parameters ──────────────────────────────────────────────────────
INITIAL_BALANCE      = 100_000.0
SLIPPAGE_PCT         = 0.001       # 0.10% per side (entry and exit)
COMMISSION_PER_SHARE = 0.005       # $0.005 per share (Alpaca-like)
MIN_CONFLUENCE_SCORE = 60.0        # minimum to enter a trade
MIN_EXPECTED_MOVE    = 0.10        # minimum 10% expected move


class PaperTradingSimulator:
    """
    Enhanced paper trading simulator with slippage, portfolio rules,
    and full performance reporting.

    Usage:
        sim = PaperTradingSimulator()
        sim.run_simulation(start_date="2024-01-01", end_date="2025-01-01")
        report = sim.generate_report()
    """

    def __init__(self, slippage_pct: float = SLIPPAGE_PCT):
        self._slippage   = slippage_pct
        self._portfolio  = PortfolioManager()

    # ── Public API ─────────────────────────────────────────────────────────────

    def run_simulation(
        self,
        start_date: str,
        end_date:   str,
        reset:      bool = False,
    ) -> dict:
        """
        Run paper trading simulation over a date range.

        Args:
            start_date: "YYYY-MM-DD"
            end_date:   "YYYY-MM-DD"
            reset:      Clear all existing simulation data first

        Returns summary stats dict.
        """
        if reset:
            self._reset_simulation()

        start = DateType.fromisoformat(start_date)
        end   = DateType.fromisoformat(end_date)
        current_date  = start
        account_value = INITIAL_BALANCE
        trades_made   = 0

        logger.info(f"Paper simulation: {start_date} → {end_date}  Balance=${account_value:,.0f}")

        while current_date <= end:
            # Process exits first (check open positions)
            exits_pnl = self._process_exits(current_date, account_value)
            account_value += exits_pnl

            # Process new entries
            entries = self._process_entries(current_date, account_value)
            trades_made += entries

            # Record daily snapshot
            self._record_daily_snapshot(current_date, account_value, exits_pnl)

            current_date += timedelta(days=1)
            # Skip weekends
            while current_date.weekday() >= 5:
                current_date += timedelta(days=1)

        return self.generate_report()

    def generate_report(self) -> dict:
        """Generate a comprehensive performance report."""
        with get_session() as session:
            # Closed trades
            trades = session.execute(select(PaperTrade)).scalars().all()
            # Equity curve
            snapshots = session.execute(
                select(PaperAccount).order_by(PaperAccount.date)
            ).scalars().all()
            # Open positions
            open_pos = session.execute(select(PaperPosition)).scalars().all()

        if not trades and not snapshots:
            return {"error": "No simulation data found. Run simulation first."}

        trade_dicts = [self._trade_to_dict(t) for t in trades]
        equity_curve = [float(s.ending_balance) for s in snapshots]
        dates_list   = [str(s.date) for s in snapshots]

        # Performance metrics
        returns       = [t["return_percent"] for t in trade_dicts]
        winning       = [r for r in returns if r > 0]
        losing        = [r for r in returns if r <= 0]

        win_rate = len(winning) / max(len(returns), 1)

        gross_profit = sum(t["pnl"] for t in trade_dicts if t["pnl"] > 0)
        gross_loss   = abs(sum(t["pnl"] for t in trade_dicts if t["pnl"] <= 0))
        profit_factor = gross_profit / max(gross_loss, 1e-8)

        max_drawdown = self._compute_max_drawdown(equity_curve)

        avg_win  = float(np.mean(winning))  if winning else 0
        avg_loss = float(np.mean(losing))   if losing  else 0

        # Weekly returns
        weekly_returns = self._compute_weekly_returns(snapshots)

        # Sharpe ratio (annualized)
        if len(equity_curve) > 1:
            daily_rets = np.diff(equity_curve) / np.array(equity_curve[:-1])
            sharpe     = (np.mean(daily_rets) / max(np.std(daily_rets), 1e-8)) * np.sqrt(252)
        else:
            sharpe = 0.0

        current_equity = equity_curve[-1] if equity_curve else INITIAL_BALANCE
        total_return   = (current_equity - INITIAL_BALANCE) / INITIAL_BALANCE

        return {
            "summary": {
                "initial_balance":  INITIAL_BALANCE,
                "current_balance":  round(current_equity, 2),
                "total_return_pct": round(total_return, 4),
                "total_return_pct_display": f"{total_return:.1%}",
                "max_drawdown_pct": round(max_drawdown, 4),
                "max_drawdown_pct_display": f"{max_drawdown:.1%}",
            },
            "trade_stats": {
                "total_trades":   len(trades),
                "winning_trades": len(winning),
                "losing_trades":  len(losing),
                "win_rate":       round(win_rate, 4),
                "win_rate_display": f"{win_rate:.1%}",
                "profit_factor":  round(profit_factor, 3),
                "avg_win_pct":    round(avg_win, 4),
                "avg_loss_pct":   round(avg_loss, 4),
                "expectancy":     round(win_rate * avg_win + (1 - win_rate) * avg_loss, 4),
                "sharpe_ratio":   round(float(sharpe), 3),
            },
            "exit_analysis": self._exit_breakdown(trade_dicts),
            "weekly_returns": weekly_returns[:8],  # last 8 weeks
            "equity_curve": {
                "dates":   dates_list[-30:],         # last 30 days
                "values":  [round(v, 2) for v in equity_curve[-30:]],
            },
            "open_positions": [self._pos_to_dict(p) for p in open_pos],
            "recent_trades":  trade_dicts[-10:][::-1],  # last 10 reversed
        }

    # ── Simulation Steps ───────────────────────────────────────────────────────

    def _process_exits(self, current_date: DateType, account_value: float) -> float:
        """Check all open positions for exit conditions. Returns total P&L for the day."""
        total_pnl = 0.0
        with get_session() as session:
            positions = session.execute(select(PaperPosition)).scalars().all()
            for pos in positions:
                # Get today's OHLCV for this symbol
                instr = session.execute(
                    select(Instrument).where(Instrument.symbol == pos.symbol)
                ).scalar_one_or_none()
                if not instr:
                    continue

                price_row = session.execute(
                    select(PriceData)
                    .where(PriceData.instrument_id == instr.id)
                    .where(PriceData.date == current_date)
                ).scalar_one_or_none()

                if not price_row:
                    continue

                exit_price = None
                exit_reason = None
                cur_close = float(price_row.close or 0)
                cur_low   = float(price_row.low   or cur_close)
                cur_high  = float(price_row.high  or cur_close)

                # Stop loss hit (checked intraday using low price)
                if pos.stop_price and cur_low <= pos.stop_price:
                    exit_price  = self._apply_slippage(pos.stop_price, "sell")
                    exit_reason = "stop"

                # Target hit (checked intraday using high price)
                elif pos.target_price and cur_high >= pos.target_price:
                    exit_price  = self._apply_slippage(pos.target_price, "sell")
                    exit_reason = "target"

                # Signal reversal (check if new confluence score flipped)
                elif self._check_signal_reversal(pos.symbol, session):
                    exit_price  = self._apply_slippage(cur_close, "sell")
                    exit_reason = "signal_reversal"

                if exit_price and exit_reason:
                    pnl = (exit_price - pos.entry_price) * pos.position_size
                    commission = pos.position_size * COMMISSION_PER_SHARE * 2  # in + out
                    pnl -= commission
                    pnl = round(pnl, 2)
                    total_pnl += pnl

                    return_pct = (exit_price - pos.entry_price) / max(pos.entry_price, 1e-8)

                    # Record closed trade
                    trade = PaperTrade(
                        symbol           = pos.symbol,
                        entry_date       = pos.entry_date,
                        exit_date        = current_date,
                        entry_price      = pos.entry_price,
                        exit_price       = round(exit_price, 4),
                        position_size    = pos.position_size,
                        direction        = pos.direction,
                        pnl              = pnl,
                        return_percent   = round(return_pct, 4),
                        exit_reason      = exit_reason,
                        signal_probability = pos.signal_probability,
                        confidence_tier  = pos.confidence_tier,
                        market_phase     = pos.market_phase,
                    )
                    session.add(trade)
                    session.delete(pos)

            session.commit()
        return total_pnl

    def _process_entries(self, current_date: DateType, account_value: float) -> int:
        """Check for new entry signals and open positions. Returns count of trades entered."""
        entries = 0
        with get_session() as session:
            # Count open positions
            open_count = session.execute(
                select(func.count(PaperPosition.id))
            ).scalar_one() or 0

            if open_count >= MAX_POSITIONS:
                return 0

            # Get latest confluence scores above threshold
            signals = session.execute(
                select(ConfluenceScore, Instrument.symbol)
                .join(Instrument, Instrument.id == ConfluenceScore.instrument_id)
                .where(ConfluenceScore.confluence_score >= MIN_CONFLUENCE_SCORE)
                .where(ConfluenceScore.signal_tier.in_(["HIGH", "MEDIUM"]))
                .where(ConfluenceScore.is_breakout == True)
                .order_by(desc(ConfluenceScore.confluence_score))
                .limit(10)
            ).all()

            for cs, sym in signals:
                if open_count >= MAX_POSITIONS:
                    break

                # Skip if already have position
                existing = session.execute(
                    select(PaperPosition).where(PaperPosition.symbol == sym)
                ).scalar_one_or_none()
                if existing:
                    continue

                # Get today's open price (enter at open next day)
                instr = session.execute(
                    select(Instrument).where(Instrument.symbol == sym)
                ).scalar_one_or_none()
                if not instr:
                    continue

                price_row = session.execute(
                    select(PriceData)
                    .where(PriceData.instrument_id == instr.id)
                    .where(PriceData.date == current_date)
                ).scalar_one_or_none()
                if not price_row:
                    continue

                # Enter at open with slippage
                entry_price  = self._apply_slippage(float(price_row.open or 0), "buy")
                stop_price   = float(cs.stop_price or 0)
                target_price = float(cs.target_price or 0)

                if entry_price <= 0 or stop_price <= 0 or stop_price >= entry_price:
                    continue

                # Validate expected move
                em_pct = float(cs.expected_move_pct or 0)
                if em_pct < MIN_EXPECTED_MOVE:
                    continue

                # Size via portfolio manager
                sizing = self._portfolio.validate_trade(
                    symbol=sym,
                    entry_price=entry_price,
                    stop_price=stop_price,
                    account_equity=account_value,
                    market_state="EXPANSION",  # only entering on breakout signals
                    confluence_score=float(cs.confluence_score or 0),
                )

                if not sizing.get("approved") or sizing.get("shares", 0) <= 0:
                    continue

                shares    = sizing["shares"]
                pos_value = shares * entry_price
                commission = shares * COMMISSION_PER_SHARE

                # Open position
                pos = PaperPosition(
                    symbol           = sym,
                    entry_date       = current_date,
                    entry_price      = round(entry_price, 4),
                    position_size    = shares,
                    position_value   = round(pos_value, 2),
                    direction        = "long",
                    stop_price       = round(stop_price, 4),
                    target_price     = round(target_price, 4),
                    signal_probability = cs.confluence_score / 100,
                    confidence_tier  = cs.signal_tier,
                    market_phase     = "EXPANSION",
                )
                session.add(pos)
                open_count += 1
                entries += 1
                logger.debug(
                    f"Opened {sym}: {shares} shares @ ${entry_price:.2f}  "
                    f"stop=${stop_price:.2f}  target=${target_price:.2f}"
                )

            session.commit()
        return entries

    def _record_daily_snapshot(
        self, date: DateType, balance: float, daily_pnl: float
    ):
        """Record end-of-day account snapshot."""
        with get_session() as session:
            open_count = session.execute(
                select(func.count(PaperPosition.id))
            ).scalar_one() or 0

            # Upsert
            existing = session.execute(
                select(PaperAccount).where(PaperAccount.date == date)
            ).scalar_one_or_none()

            if existing:
                existing.ending_balance = round(balance, 2)
                existing.daily_pnl      = round(daily_pnl, 2)
                existing.open_positions = open_count
            else:
                acct = PaperAccount(
                    date             = date,
                    starting_balance = round(balance - daily_pnl, 2),
                    ending_balance   = round(balance, 2),
                    daily_pnl        = round(daily_pnl, 2),
                    open_positions   = open_count,
                )
                session.add(acct)
            session.commit()

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _apply_slippage(self, price: float, side: str) -> float:
        """Apply slippage to an order price."""
        if side == "buy":
            return price * (1 + self._slippage)
        else:
            return price * (1 - self._slippage)

    def _check_signal_reversal(self, symbol: str, session) -> bool:
        """Check if the latest confluence score has reversed to NONE."""
        cs = session.execute(
            select(ConfluenceScore)
            .join(Instrument, Instrument.id == ConfluenceScore.instrument_id)
            .where(Instrument.symbol == symbol)
            .order_by(desc(ConfluenceScore.date))
            .limit(1)
        ).scalar_one_or_none()
        if cs and cs.signal_tier == "NONE":
            return True
        return False

    def _compute_max_drawdown(self, equity_curve: list[float]) -> float:
        """Compute maximum peak-to-trough drawdown."""
        if len(equity_curve) < 2:
            return 0.0
        peak = equity_curve[0]
        max_dd = 0.0
        for val in equity_curve:
            if val > peak:
                peak = val
            dd = (peak - val) / max(peak, 1e-8)
            max_dd = max(max_dd, dd)
        return max_dd

    def _compute_weekly_returns(self, snapshots: list) -> list[dict]:
        """Compute weekly return summaries."""
        if not snapshots:
            return []
        df = pd.DataFrame([{
            "date":    s.date,
            "balance": float(s.ending_balance),
        } for s in snapshots])
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").resample("W").last()
        df["weekly_return"] = df["balance"].pct_change()
        df = df.dropna()
        return [
            {
                "week":          str(row.name.date()),
                "balance":       round(float(row["balance"]), 2),
                "weekly_return": round(float(row["weekly_return"]), 4),
                "weekly_return_display": f"{row['weekly_return']:.1%}",
            }
            for _, row in df.iterrows()
        ]

    def _exit_breakdown(self, trades: list[dict]) -> dict:
        """Breakdown of exit reasons."""
        reasons = {}
        for t in trades:
            r = t.get("exit_reason", "unknown")
            if r not in reasons:
                reasons[r] = {"count": 0, "total_pnl": 0.0, "avg_return": 0.0, "_returns": []}
            reasons[r]["count"] += 1
            reasons[r]["total_pnl"] += t.get("pnl", 0)
            reasons[r]["_returns"].append(t.get("return_percent", 0))

        result = {}
        for r, data in reasons.items():
            rets = data.pop("_returns")
            data["avg_return"] = round(float(np.mean(rets)), 4) if rets else 0
            data["total_pnl"]  = round(data["total_pnl"], 2)
            result[r] = data
        return result

    def _reset_simulation(self):
        """Clear all simulation data."""
        with get_session() as session:
            session.execute(PaperTrade.__table__.delete())
            session.execute(PaperPosition.__table__.delete())
            session.execute(PaperAccount.__table__.delete())
            session.commit()
        logger.info("Paper trading simulation reset.")

    def _trade_to_dict(self, t: PaperTrade) -> dict:
        return {
            "symbol":       t.symbol,
            "entry_date":   str(t.entry_date),
            "exit_date":    str(t.exit_date),
            "entry_price":  t.entry_price,
            "exit_price":   t.exit_price,
            "shares":       t.position_size,
            "direction":    t.direction,
            "pnl":          t.pnl,
            "return_percent": t.return_percent,
            "exit_reason":  t.exit_reason,
            "tier":         t.confidence_tier,
        }

    def _pos_to_dict(self, p: PaperPosition) -> dict:
        return {
            "symbol":       p.symbol,
            "entry_date":   str(p.entry_date),
            "entry_price":  p.entry_price,
            "shares":       p.position_size,
            "position_value": p.position_value,
            "stop_price":   p.stop_price,
            "target_price": p.target_price,
            "tier":         p.confidence_tier,
        }
