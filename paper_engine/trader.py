"""
RoboAlgo — Paper Trading Engine
Simulates trading using RoboAlgo signals without risking real capital.

Rules:
  - Starting balance:    $100,000
  - Risk per trade:      2% of current account balance
  - Position sizing:     risk_per_trade / ATR_distance_to_stop
  - Max open positions:  5
  - Max exposure/trade:  25% of account
  - Entry:               next day open after signal
  - Exit:                target reached | stop hit | signal reverses
  - Direction:           long for bull ETFs, long for bear ETFs (inverse exposure built-in)
"""

import logging
from datetime import date, timedelta
from typing import Optional

import yfinance as yf
import pandas as pd
from sqlalchemy import select, delete, desc
from sqlalchemy.dialects.postgresql import insert as pg_insert

from database.connection import get_session
from database.models import (
    Instrument, PriceData, Indicator, Signal,
    PaperAccount, PaperPosition, PaperTrade,
)

logger = logging.getLogger(__name__)

STARTING_BALANCE = 100_000.0
RISK_PCT         = 0.02          # 2% risk per trade
MAX_POSITIONS    = 5
MAX_EXPOSURE_PCT = 0.25          # 25% of account per trade
STOP_ATR_MULT    = 1.5           # stop = entry - 1.5 ATR


class PaperTrader:
    """
    Simulates paper trades using RoboAlgo signals.

    Usage:
        trader = PaperTrader()
        summary = trader.run_simulation(
            start_date=date(2025, 1, 1),
            end_date=date(2025, 3, 1),
            reset=True,
        )
    """

    def __init__(self):
        self.session = get_session()

    def close(self):
        self.session.close()

    # ── Public API ─────────────────────────────────────────────────────────

    def run_simulation(
        self,
        start_date: date,
        end_date: date,
        reset: bool = False,
    ) -> dict:
        """
        Run paper trading simulation over a date range.

        Args:
            start_date:  First simulation day (inclusive)
            end_date:    Last simulation day (inclusive)
            reset:       Clear all existing paper data before running

        Returns:
            Summary dict with performance stats.
        """
        if reset:
            self._reset()

        balance = self._get_current_balance()
        sim_date = start_date

        while sim_date <= end_date:
            # Process exits first (check existing positions)
            balance = self._process_exits(sim_date, balance)

            # Then check for new entries (using yesterday's signal → today's open)
            balance = self._process_entries(sim_date, balance)

            # Record daily snapshot
            self._record_daily_snapshot(sim_date, balance)

            sim_date += timedelta(days=1)

        return self.get_summary(start_date, end_date)

    def get_summary(self, start_date: Optional[date] = None, end_date: Optional[date] = None) -> dict:
        """Return performance summary for the simulation period."""
        trades = self._get_closed_trades(start_date, end_date)
        snapshots = self._get_snapshots(start_date, end_date)
        positions = self._get_open_positions()

        if not snapshots:
            return {
                "starting_balance": STARTING_BALANCE,
                "ending_balance": STARTING_BALANCE,
                "total_return_pct": 0.0,
                "total_pnl": 0.0,
                "number_of_trades": 0,
                "win_rate": 0.0,
                "average_return_pct": 0.0,
                "largest_win": 0.0,
                "largest_loss": 0.0,
                "max_drawdown_pct": 0.0,
                "open_positions": 0,
                "trades": [],
            }

        starting_balance = snapshots[0]["starting_balance"]
        ending_balance   = snapshots[-1]["ending_balance"]
        total_pnl        = ending_balance - starting_balance
        total_return_pct = (total_pnl / starting_balance) * 100.0

        # Win rate
        wins  = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] <= 0]
        win_rate = len(wins) / len(trades) * 100.0 if trades else 0.0

        # Average return
        avg_return = sum(t["return_percent"] for t in trades) / len(trades) if trades else 0.0
        largest_win  = max((t["pnl"] for t in trades), default=0.0)
        largest_loss = min((t["pnl"] for t in trades), default=0.0)

        # Max drawdown from equity curve
        max_drawdown_pct = self._compute_max_drawdown(snapshots)

        return {
            "starting_balance":   round(starting_balance, 2),
            "ending_balance":     round(ending_balance, 2),
            "total_return_pct":   round(total_return_pct, 2),
            "total_pnl":          round(total_pnl, 2),
            "number_of_trades":   len(trades),
            "win_rate":           round(win_rate, 1),
            "average_return_pct": round(avg_return, 2),
            "largest_win":        round(largest_win, 2),
            "largest_loss":       round(largest_loss, 2),
            "max_drawdown_pct":   round(max_drawdown_pct, 2),
            "open_positions":     len(positions),
            "trades":             trades,
            "daily_equity":       snapshots,
        }

    def get_open_positions(self) -> list[dict]:
        return self._get_open_positions()

    def get_trade_history(self, limit: int = 200) -> list[dict]:
        return self._get_closed_trades(limit=limit)

    def reset(self):
        self._reset()

    # ── Simulation Steps ────────────────────────────────────────────────────

    def _process_exits(self, sim_date: date, balance: float) -> float:
        """Check all open positions for exit conditions on sim_date."""
        positions = self.session.execute(select(PaperPosition)).scalars().all()

        for pos in positions:
            if pos.entry_date >= sim_date:
                continue  # can't exit same day as entry

            # Get today's OHLC
            ohlc = self._get_price(pos.symbol, sim_date)
            if not ohlc:
                continue

            open_p, high_p, low_p, close_p = ohlc

            exit_price  = None
            exit_reason = None

            # Check if target or stop was hit (use high/low for intraday check)
            if pos.target_price and high_p >= pos.target_price:
                exit_price  = pos.target_price
                exit_reason = "target"
            elif pos.stop_price and low_p <= pos.stop_price:
                exit_price  = pos.stop_price
                exit_reason = "stop"
            else:
                # Check signal reversal — if today's signal is bearish, exit at close
                sig = self._get_latest_signal(pos.symbol, sim_date)
                if sig and sig.probability < 0.40:
                    exit_price  = close_p
                    exit_reason = "signal_reversal"

            if exit_price is not None:
                pnl    = (exit_price - pos.entry_price) * pos.position_size
                ret_pct = (exit_price / pos.entry_price - 1.0) * 100.0

                # Store trade
                trade = PaperTrade(
                    symbol=pos.symbol,
                    entry_date=pos.entry_date,
                    exit_date=sim_date,
                    entry_price=pos.entry_price,
                    exit_price=round(exit_price, 4),
                    position_size=pos.position_size,
                    direction=pos.direction,
                    pnl=round(pnl, 2),
                    return_percent=round(ret_pct, 2),
                    exit_reason=exit_reason,
                    signal_probability=pos.signal_probability,
                    confidence_tier=pos.confidence_tier,
                    market_phase=pos.market_phase,
                )
                self.session.add(trade)
                self.session.delete(pos)
                balance += pnl
                logger.info(f"EXIT {pos.symbol} @ {exit_price:.2f} | {exit_reason} | PnL: ${pnl:.2f}")

        self.session.commit()
        return balance

    def _process_entries(self, sim_date: date, balance: float) -> float:
        """Check yesterday's signals and open new positions at today's open."""
        # How many slots are free?
        open_count = self.session.execute(
            select(PaperPosition)
        ).scalars().all()
        available_slots = MAX_POSITIONS - len(open_count)
        if available_slots <= 0:
            return balance

        # Symbols already held
        held_symbols = {p.symbol for p in open_count}

        # Get yesterday's HIGH-confidence signals (prob ≥ 0.70)
        yesterday = sim_date - timedelta(days=1)
        candidates = self._get_signals_for_date(yesterday, min_prob=0.70)

        for sig_dict in candidates:
            if available_slots <= 0:
                break

            sym = sig_dict["symbol"]
            if sym in held_symbols:
                continue

            # Get today's open price (entry)
            ohlc = self._get_price(sym, sim_date)
            if not ohlc:
                continue

            entry_price = ohlc[0]  # open
            if not entry_price or entry_price <= 0:
                continue

            # ATR for position sizing
            atr = self._get_atr(sym, yesterday)
            if not atr or atr <= 0:
                continue

            # Position sizing: risk 2% of balance / ATR distance
            risk_dollars   = balance * RISK_PCT
            atr_distance   = atr * STOP_ATR_MULT
            raw_shares     = int(risk_dollars / atr_distance)

            # Cap at 25% of account
            max_value = balance * MAX_EXPOSURE_PCT
            max_shares = int(max_value / entry_price)
            shares = min(raw_shares, max_shares)
            if shares < 1:
                continue

            position_value = shares * entry_price
            if position_value > balance * MAX_EXPOSURE_PCT:
                continue

            stop_price   = round(entry_price - atr_distance, 4)
            target_price = sig_dict.get("sell_price") or round(entry_price + atr * 4.0, 4)

            pos = PaperPosition(
                symbol=sym,
                entry_date=sim_date,
                entry_price=round(entry_price, 4),
                position_size=shares,
                position_value=round(position_value, 2),
                direction="long",
                stop_price=stop_price,
                target_price=round(target_price, 4),
                signal_probability=sig_dict["probability"],
                confidence_tier=sig_dict["confidence_tier"],
                market_phase=sig_dict["market_phase"],
            )
            self.session.add(pos)
            held_symbols.add(sym)
            available_slots -= 1
            logger.info(
                f"ENTER {sym} @ {entry_price:.2f} | {shares} shares | "
                f"stop={stop_price:.2f} target={target_price:.2f} | "
                f"prob={sig_dict['probability']:.0%}"
            )

        self.session.commit()
        return balance

    # ── Data Helpers ────────────────────────────────────────────────────────

    def _get_price(self, symbol: str, on_date: date) -> Optional[tuple]:
        """Return (open, high, low, close) for symbol on date from DB."""
        row = self.session.execute(
            select(PriceData)
            .join(Instrument, PriceData.instrument_id == Instrument.id)
            .where(Instrument.symbol == symbol)
            .where(PriceData.date == on_date)
            .limit(1)
        ).scalar_one_or_none()

        if not row or row.close is None:
            return None
        return (
            float(row.open  or row.close),
            float(row.high  or row.close),
            float(row.low   or row.close),
            float(row.close),
        )

    def _get_atr(self, symbol: str, on_date: date) -> Optional[float]:
        """Return ATR for symbol on/before date."""
        row = self.session.execute(
            select(Indicator)
            .join(Instrument, Indicator.instrument_id == Instrument.id)
            .where(Instrument.symbol == symbol)
            .where(Indicator.date <= on_date)
            .order_by(desc(Indicator.date))
            .limit(1)
        ).scalar_one_or_none()
        return float(row.atr) if row and row.atr else None

    def _get_latest_signal(self, symbol: str, on_date: date) -> Optional[Signal]:
        """Return the most recent signal for symbol on/before date."""
        instr = self.session.execute(
            select(Instrument).where(Instrument.symbol == symbol)
        ).scalar_one_or_none()
        if not instr:
            return None
        return self.session.execute(
            select(Signal)
            .where(Signal.instrument_id == instr.id)
            .where(Signal.date <= on_date)
            .order_by(desc(Signal.date))
            .limit(1)
        ).scalar_one_or_none()

    def _get_signals_for_date(self, sig_date: date, min_prob: float = 0.70) -> list[dict]:
        """Return all signals for a specific date sorted by probability desc."""
        rows = self.session.execute(
            select(Signal, Instrument.symbol)
            .join(Instrument, Signal.instrument_id == Instrument.id)
            .where(Signal.date == sig_date)
            .where(Signal.probability >= min_prob)
            .order_by(desc(Signal.probability))
        ).all()

        return [
            {
                "symbol":           sym,
                "probability":      float(sig.probability),
                "confidence_tier":  sig.confidence_tier,
                "market_phase":     sig.market_phase,
                "buy_price":        sig.buy_price,
                "sell_price":       sig.sell_price,
                "accumulate_price": sig.accumulate_price,
                "scale_price":      sig.scale_price,
            }
            for sig, sym in rows
        ]

    def _get_current_balance(self) -> float:
        """Return current account balance from latest snapshot, or starting balance."""
        last = self.session.execute(
            select(PaperAccount).order_by(desc(PaperAccount.date)).limit(1)
        ).scalar_one_or_none()
        return float(last.ending_balance) if last else STARTING_BALANCE

    def _record_daily_snapshot(self, on_date: date, ending_balance: float):
        """Upsert daily account snapshot."""
        existing = self.session.execute(
            select(PaperAccount).where(PaperAccount.date == on_date)
        ).scalar_one_or_none()

        prev = self.session.execute(
            select(PaperAccount)
            .where(PaperAccount.date < on_date)
            .order_by(desc(PaperAccount.date))
            .limit(1)
        ).scalar_one_or_none()

        starting = float(prev.ending_balance) if prev else STARTING_BALANCE
        daily_pnl = ending_balance - starting
        open_pos_count = self.session.execute(select(PaperPosition)).scalars().all()

        if existing:
            existing.ending_balance = round(ending_balance, 2)
            existing.daily_pnl      = round(daily_pnl, 2)
            existing.open_positions = len(open_pos_count)
        else:
            snap = PaperAccount(
                date=on_date,
                starting_balance=round(starting, 2),
                ending_balance=round(ending_balance, 2),
                daily_pnl=round(daily_pnl, 2),
                open_positions=len(open_pos_count),
            )
            self.session.add(snap)
        self.session.commit()

    # ── Result Fetchers ─────────────────────────────────────────────────────

    def _get_closed_trades(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        limit: int = 1000,
    ) -> list[dict]:
        q = select(PaperTrade).order_by(desc(PaperTrade.exit_date))
        if start_date:
            q = q.where(PaperTrade.exit_date >= start_date)
        if end_date:
            q = q.where(PaperTrade.exit_date <= end_date)
        q = q.limit(limit)
        rows = self.session.execute(q).scalars().all()
        return [
            {
                "symbol":             r.symbol,
                "entry_date":         str(r.entry_date),
                "exit_date":          str(r.exit_date),
                "entry_price":        r.entry_price,
                "exit_price":         r.exit_price,
                "position_size":      r.position_size,
                "direction":          r.direction,
                "pnl":                r.pnl,
                "return_percent":     r.return_percent,
                "exit_reason":        r.exit_reason,
                "confidence_tier":    r.confidence_tier,
                "market_phase":       r.market_phase,
                "signal_probability": r.signal_probability,
            }
            for r in rows
        ]

    def _get_open_positions(self) -> list[dict]:
        rows = self.session.execute(select(PaperPosition)).scalars().all()
        return [
            {
                "symbol":             r.symbol,
                "entry_date":         str(r.entry_date),
                "entry_price":        r.entry_price,
                "position_size":      r.position_size,
                "position_value":     r.position_value,
                "direction":          r.direction,
                "stop_price":         r.stop_price,
                "target_price":       r.target_price,
                "confidence_tier":    r.confidence_tier,
                "market_phase":       r.market_phase,
                "signal_probability": r.signal_probability,
            }
            for r in rows
        ]

    def _get_snapshots(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> list[dict]:
        q = select(PaperAccount).order_by(PaperAccount.date)
        if start_date:
            q = q.where(PaperAccount.date >= start_date)
        if end_date:
            q = q.where(PaperAccount.date <= end_date)
        rows = self.session.execute(q).scalars().all()
        return [
            {
                "date":             str(r.date),
                "starting_balance": r.starting_balance,
                "ending_balance":   r.ending_balance,
                "daily_pnl":        r.daily_pnl,
                "open_positions":   r.open_positions,
            }
            for r in rows
        ]

    # ── Metrics ─────────────────────────────────────────────────────────────

    def _compute_max_drawdown(self, snapshots: list[dict]) -> float:
        """Compute max drawdown % from equity curve."""
        if not snapshots:
            return 0.0
        equity = [s["ending_balance"] for s in snapshots]
        peak = equity[0]
        max_dd = 0.0
        for val in equity:
            if val > peak:
                peak = val
            dd = (peak - val) / peak * 100.0
            if dd > max_dd:
                max_dd = dd
        return round(max_dd, 2)

    # ── Reset ────────────────────────────────────────────────────────────────

    def _reset(self):
        """Clear all paper trading data."""
        self.session.execute(delete(PaperTrade))
        self.session.execute(delete(PaperPosition))
        self.session.execute(delete(PaperAccount))
        self.session.commit()
        logger.info("Paper trading data reset to zero.")
