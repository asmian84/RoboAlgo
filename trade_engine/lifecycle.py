"""
RoboAlgo — Trade Lifecycle Engine
Manages the progression of signals through their lifecycle states.

Lifecycle States:
  SETUP   → Signal identified. Watching for confirmation.
  TRIGGER → Entry conditions met. Awaiting execution.
  ENTRY   → Position entered. Tracking active trade.
  ACTIVE  → Trade in market. Monitoring tiers and stop.
  EXIT    → Trade closed. Outcome recorded.

State transitions:
  SETUP   →(confluence confirmed)→ TRIGGER
  TRIGGER →(market opens, price confirms)→ ENTRY
  ENTRY   →(order filled)→ ACTIVE
  ACTIVE  →(tier hit / stop hit / reversal)→ EXIT

Downstream effects:
  EXIT records feed back into analytics_engine/expectancy.py
  Negative EV setup types are downgraded to WATCH in confluence engine.
"""

import logging
from datetime import datetime
from typing import Optional, Literal

from sqlalchemy import select, desc

from database.connection import get_session
from database.models import Instrument, TradeLifecycle

logger = logging.getLogger(__name__)

# Valid lifecycle state machine
VALID_TRANSITIONS = {
    "SETUP":   ["TRIGGER", "EXIT"],       # EXIT from SETUP = cancelled
    "TRIGGER": ["ENTRY",   "EXIT"],       # EXIT from TRIGGER = missed entry
    "ENTRY":   ["ACTIVE",  "EXIT"],       # EXIT from ENTRY = immediate stop
    "ACTIVE":  ["EXIT"],
    "EXIT":    [],                         # terminal
}

ExitReason = Literal[
    "STOP_HIT", "TIER1_HIT", "TIER2_HIT", "TIER3_HIT",
    "SIGNAL_REVERSAL", "MANUAL", "CANCELLED", "TIME_STOP"
]


class TradeLifecycleEngine:
    """
    Creates and advances trade lifecycle records.

    Usage:
        engine = TradeLifecycleEngine()

        # Create a new SETUP
        trade_id = engine.create_setup(
            symbol="SOXL",
            setup_type="compression_breakout",
            market_state="EXPANSION",
            entry_price=62.50,
            stop_price=60.00,
            tier1_sell=65.00,
            tier2_sell=70.00,
            tier3_hold=80.00,
            confluence_score=84.0,
        )

        # Advance to TRIGGER
        engine.advance(trade_id, "TRIGGER")

        # Advance to ENTRY (position taken)
        engine.advance(trade_id, "ENTRY", position_size=200)

        # Mark as ACTIVE
        engine.advance(trade_id, "ACTIVE")

        # Close with exit data
        engine.close_trade(
            trade_id,
            exit_price=67.50,
            exit_reason="TIER1_HIT",
        )
    """

    def create_setup(
        self,
        symbol: str,
        setup_type: str,
        market_state: str,
        entry_price: float,
        stop_price: float,
        tier1_sell: float,
        tier2_sell: float,
        tier3_hold: float,
        confluence_score: float = 0.0,
        position_size: int = 0,
        notes: Optional[str] = None,
    ) -> int:
        """Create a new trade lifecycle record in SETUP state. Returns trade_id."""
        with get_session() as session:
            record = TradeLifecycle(
                symbol            = symbol.upper(),
                setup_type        = setup_type,
                market_state      = market_state,
                state             = "SETUP",
                entry_price       = entry_price,
                stop_price        = stop_price,
                tier1_sell        = tier1_sell,
                tier2_sell        = tier2_sell,
                tier3_hold        = tier3_hold,
                target_price      = tier2_sell,   # general target = tier2
                confluence_score  = confluence_score,
                position_size     = position_size,
                setup_timestamp   = datetime.utcnow(),
                notes             = notes,
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            trade_id = record.id

        logger.info(f"Trade lifecycle SETUP created: {symbol} {setup_type} id={trade_id}")
        return trade_id

    def advance(
        self,
        trade_id: int,
        new_state: str,
        position_size: Optional[int] = None,
        executed_entry_price: Optional[float] = None,
    ) -> bool:
        """
        Advance a trade to the next lifecycle state.
        Returns True if successful, False if transition is invalid.
        """
        with get_session() as session:
            trade = session.get(TradeLifecycle, trade_id)
            if not trade:
                logger.error(f"Trade id={trade_id} not found")
                return False

            current = trade.state
            allowed = VALID_TRANSITIONS.get(current, [])
            if new_state not in allowed:
                logger.warning(
                    f"Invalid transition: {current} → {new_state} for trade id={trade_id}"
                )
                return False

            trade.state = new_state

            if new_state == "TRIGGER":
                trade.trigger_timestamp = datetime.utcnow()
            elif new_state == "ENTRY":
                trade.entry_timestamp = datetime.utcnow()
                if position_size is not None:
                    trade.position_size = position_size
                if executed_entry_price is not None:
                    trade.entry_price = executed_entry_price
            elif new_state == "EXIT":
                trade.exit_timestamp = datetime.utcnow()

            session.commit()

        logger.info(f"Trade id={trade_id} advanced: {current} → {new_state}")
        return True

    def close_trade(
        self,
        trade_id: int,
        exit_price: float,
        exit_reason: str,
        notes: Optional[str] = None,
    ) -> Optional[dict]:
        """
        Close a trade (advance to EXIT) and record P&L.
        Returns summary dict with pnl, return_pct, holding_period.
        """
        with get_session() as session:
            trade = session.get(TradeLifecycle, trade_id)
            if not trade:
                logger.error(f"Trade id={trade_id} not found")
                return None

            if trade.state == "EXIT":
                logger.warning(f"Trade id={trade_id} already in EXIT state")
                return None

            entry_px   = float(trade.entry_price or 0)
            shares     = int(trade.position_size or 0)
            pnl        = (exit_price - entry_px) * shares
            return_pct = (exit_price - entry_px) / max(entry_px, 1e-8) if entry_px > 0 else 0.0

            holding_days = None
            if trade.entry_timestamp:
                holding_days = (datetime.utcnow() - trade.entry_timestamp).days

            trade.state           = "EXIT"
            trade.exit_price      = exit_price
            trade.exit_reason     = exit_reason
            trade.exit_timestamp  = datetime.utcnow()
            trade.pnl             = round(pnl, 4)
            trade.return_percent  = round(return_pct, 6)
            trade.holding_period  = holding_days
            if notes:
                trade.notes = notes

            session.commit()

        logger.info(
            f"Trade id={trade_id} closed: {exit_reason} "
            f"exit={exit_price:.4f} pnl={pnl:.2f}"
        )

        return {
            "trade_id":      trade_id,
            "symbol":        trade.symbol,
            "setup_type":    trade.setup_type,
            "market_state":  trade.market_state,
            "exit_reason":   exit_reason,
            "entry_price":   entry_px,
            "exit_price":    exit_price,
            "pnl":           round(pnl, 4),
            "return_pct":    round(return_pct, 6),
            "holding_days":  holding_days,
        }

    def get_active_trades(self, symbol: Optional[str] = None) -> list[dict]:
        """Return all open trades (SETUP/TRIGGER/ENTRY/ACTIVE)."""
        with get_session() as session:
            q = select(TradeLifecycle).where(
                TradeLifecycle.state.in_(["SETUP", "TRIGGER", "ENTRY", "ACTIVE"])
            )
            if symbol:
                q = q.where(TradeLifecycle.symbol == symbol.upper())
            q = q.order_by(desc(TradeLifecycle.setup_timestamp))
            rows = session.execute(q).scalars().all()
            return [self._row_to_dict(r) for r in rows]

    def get_trade_history(
        self,
        symbol: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """Return completed trades (EXIT state)."""
        with get_session() as session:
            q = select(TradeLifecycle).where(TradeLifecycle.state == "EXIT")
            if symbol:
                q = q.where(TradeLifecycle.symbol == symbol.upper())
            q = q.order_by(desc(TradeLifecycle.exit_timestamp)).limit(limit)
            rows = session.execute(q).scalars().all()
            return [self._row_to_dict(r) for r in rows]

    def get_trade(self, trade_id: int) -> Optional[dict]:
        """Return a single trade by ID."""
        with get_session() as session:
            trade = session.get(TradeLifecycle, trade_id)
            return self._row_to_dict(trade) if trade else None

    def get_open_position_count(self) -> int:
        """Return count of positions currently in ACTIVE state."""
        with get_session() as session:
            from sqlalchemy import func
            result = session.execute(
                select(func.count()).select_from(TradeLifecycle)
                .where(TradeLifecycle.state == "ACTIVE")
            ).scalar()
            return int(result or 0)

    # ── Internal ───────────────────────────────────────────────────────────────

    def _row_to_dict(self, row: TradeLifecycle) -> dict:
        return {
            "id":             row.id,
            "symbol":         row.symbol,
            "setup_type":     row.setup_type,
            "market_state":   row.market_state,
            "state":          row.state,
            "entry_price":    row.entry_price,
            "stop_price":     row.stop_price,
            "tier1_sell":     row.tier1_sell,
            "tier2_sell":     row.tier2_sell,
            "tier3_hold":     row.tier3_hold,
            "exit_price":     row.exit_price,
            "exit_reason":    row.exit_reason,
            "position_size":  row.position_size,
            "pnl":            row.pnl,
            "return_percent": row.return_percent,
            "holding_period": row.holding_period,
            "confluence_score": row.confluence_score,
            "setup_at":       str(row.setup_timestamp) if row.setup_timestamp else None,
            "trigger_at":     str(row.trigger_timestamp) if row.trigger_timestamp else None,
            "entry_at":       str(row.entry_timestamp) if row.entry_timestamp else None,
            "exit_at":        str(row.exit_timestamp) if row.exit_timestamp else None,
            "notes":          row.notes,
        }
