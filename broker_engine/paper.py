"""
RoboAlgo — Paper Broker Adapter
Local, zero-credential simulated broker backed by the existing
`paper_account` / `paper_positions` / `paper_trades` tables.

Why this exists:
  The auto-trader only had an Alpaca adapter, which needs API keys.
  PaperBroker slots into the same BrokerBase contract so the auto-trader
  loop/governor/reconciler work completely unchanged — just point
  `broker="paper"` at it and it simulates fills against real market data
  (via data_engine.market_data_service, falling back to the latest stored
  price_data close) with zero external credentials required.

Accounting model (works within the existing schema — no new columns):
  - `paper_account` is treated as a running CASH ledger, upserted once per
    calendar day. `ending_balance` is *cash*, not equity. Equity/portfolio
    value is always derived on read as cash + sum(position market value) —
    it is never itself persisted, so it can't drift out of sync with cash.
  - `daily_pnl` accumulates *realized* P&L booked today (position opens
    move cash but are not realized P&L; only closes are).
  - Opening a position moves cash immediately (buy: cash -= qty*price;
    short sell: cash += qty*price). Closing moves cash the opposite way
    and books a `paper_trades` row with the realized P&L.
  - `mark_to_market()` is what actually closes positions whose stop/target
    has been crossed — call it periodically (the PAPER page and the
    auto-trader loop both should).

Order-level bookkeeping (order_id / client_order_id / status) has no home
in the position/trade tables, so it lives in-memory on the broker instance.
Fills are immediate (no resting order book), so idempotency for a given
`client_order_id` is guaranteed for the lifetime of a single PaperBroker
instance — which is how the auto-trader loop uses it per cycle, backed
further by auto_trader.audit's own client_order_id de-dupe.
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import date, datetime
from typing import Optional

from sqlalchemy import select, desc, delete, func

from broker_engine.base import BrokerBase, AccountInfo, Position, OrderResult, Quote
from database.connection import get_session
from database.models import (
    Instrument, PriceData, PaperAccount, PaperPosition, PaperTrade,
)

logger = logging.getLogger(__name__)

_DEFAULT_STARTING_BALANCE = 100_000.0


class PaperBroker(BrokerBase):
    """
    Local simulated broker. No network credentials required.

    Prices come from data_engine.market_data_service (yfinance/Finnhub),
    falling back to the latest stored price_data close. Never raises on
    quote failure — callers get None/0-valued quotes instead.
    """

    def __init__(self):
        # In-memory order registry — see module docstring for why.
        self._orders: dict[str, OrderResult] = {}       # order_id -> OrderResult
        self._client_index: dict[str, str] = {}          # client_order_id -> order_id

    @property
    def name(self) -> str:
        return "paper"

    @property
    def is_paper(self) -> bool:
        # Always True. This broker can never report live — auto_trader's
        # assert_mode_safe relies on that guarantee.
        return True

    # ── Bootstrap / lifecycle ──────────────────────────────────────────────

    @staticmethod
    def _starting_balance() -> float:
        try:
            return float(os.getenv("PAPER_STARTING_BALANCE", str(_DEFAULT_STARTING_BALANCE)))
        except (TypeError, ValueError):
            return _DEFAULT_STARTING_BALANCE

    def _get_or_create_today_row(self, session) -> PaperAccount:
        """Bootstrap the account on first use; roll cash forward day-to-day."""
        today = date.today()
        row = session.execute(
            select(PaperAccount).where(PaperAccount.date == today)
        ).scalar_one_or_none()
        if row:
            return row

        prev = session.execute(
            select(PaperAccount).order_by(desc(PaperAccount.date)).limit(1)
        ).scalar_one_or_none()
        starting = float(prev.ending_balance) if prev else self._starting_balance()

        open_count = session.execute(
            select(func.count(PaperPosition.id))
        ).scalar_one() or 0

        row = PaperAccount(
            date=today,
            starting_balance=round(starting, 2),
            ending_balance=round(starting, 2),
            daily_pnl=0.0,
            open_positions=open_count,
        )
        session.add(row)
        session.commit()
        session.refresh(row)
        return row

    def reset(self, starting_balance: Optional[float] = None) -> AccountInfo:
        """Wipe all paper positions/trades and restore the opening cash balance."""
        bal = float(starting_balance) if starting_balance is not None else self._starting_balance()
        with get_session() as session:
            session.execute(delete(PaperTrade))
            session.execute(delete(PaperPosition))
            session.execute(delete(PaperAccount))
            session.add(PaperAccount(
                date=date.today(),
                starting_balance=round(bal, 2),
                ending_balance=round(bal, 2),
                daily_pnl=0.0,
                open_positions=0,
            ))
            session.commit()
        self._orders.clear()
        self._client_index.clear()
        logger.info("PaperBroker reset — starting balance $%.2f", bal)
        return self.get_account()

    # ── Market Data ─────────────────────────────────────────────────────────

    def _get_price(self, symbol: str) -> Optional[float]:
        """Best-effort current price. Never raises — returns None on total failure."""
        sym = symbol.upper()

        try:
            from data_engine.market_data_service import get_quote as _svc_get_quote
            data = _svc_get_quote(sym)
            if data and data.get("price"):
                price = float(data["price"])
                if price > 0:
                    return price
        except Exception as exc:
            logger.debug("PaperBroker: live quote failed for %s: %s", sym, exc)

        # Fallback: latest stored daily close.
        try:
            with get_session() as session:
                instr = session.execute(
                    select(Instrument).where(Instrument.symbol == sym)
                ).scalar_one_or_none()
                if instr:
                    row = session.execute(
                        select(PriceData)
                        .where(PriceData.instrument_id == instr.id)
                        .where(PriceData.close.isnot(None))
                        .order_by(desc(PriceData.date))
                        .limit(1)
                    ).scalar_one_or_none()
                    if row and row.close:
                        return float(row.close)
        except Exception as exc:
            logger.debug("PaperBroker: price_data fallback failed for %s: %s", sym, exc)

        return None

    def get_quote(self, symbol: str) -> Quote:
        price = self._get_price(symbol)
        if price is None:
            return Quote(symbol=symbol.upper(), bid=0.0, ask=0.0, last=0.0, volume=0)
        return Quote(
            symbol=symbol.upper(), bid=price, ask=price, last=price,
            volume=0, timestamp=datetime.utcnow(),
        )

    def get_quotes(self, symbols: list[str]) -> dict[str, Quote]:
        return {s.upper(): self.get_quote(s) for s in symbols}

    # ── Account ─────────────────────────────────────────────────────────────

    def get_account(self) -> AccountInfo:
        with get_session() as session:
            row = self._get_or_create_today_row(session)
            cash = float(row.ending_balance)

        positions = self.get_positions()
        mv = sum(p.market_value for p in positions)
        return AccountInfo(
            account_id="paper-1",
            cash=round(cash, 2),
            portfolio_val=round(cash + mv, 2),
            buying_power=round(cash, 2),
            broker="paper",
            currency="USD",
            is_paper=True,
        )

    def get_positions(self) -> list[Position]:
        with get_session() as session:
            rows = session.execute(select(PaperPosition)).scalars().all()
            snapshot = [
                (r.symbol, r.position_size, r.entry_price, r.direction) for r in rows
            ]

        results: list[Position] = []
        for symbol, qty, entry_price, direction in snapshot:
            price = self._get_price(symbol)
            if price is None:
                price = float(entry_price)
            mv = qty * price
            cost_basis = qty * entry_price
            if direction == "short":
                upnl = cost_basis - mv
            else:
                upnl = mv - cost_basis
            upct = (upnl / cost_basis * 100.0) if cost_basis else 0.0
            results.append(Position(
                symbol=symbol, qty=qty, avg_cost=entry_price,
                market_value=round(mv, 2), unrealized_pnl=round(upnl, 2),
                unrealized_pct=round(upct, 2), side=direction,
            ))
        return results

    def get_position(self, symbol: str) -> Optional[Position]:
        sym = symbol.upper()
        for p in self.get_positions():
            if p.symbol.upper() == sym:
                return p
        return None

    # ── Orders ──────────────────────────────────────────────────────────────

    def _execute_fill(
        self, symbol: str, qty: int, side: str, fill_price: Optional[float],
        stop_price: Optional[float] = None, target_price: Optional[float] = None,
        client_order_id: Optional[str] = None, order_type: str = "market",
    ) -> OrderResult:
        sym = symbol.upper()
        side = side.lower()

        if client_order_id and client_order_id in self._client_index:
            return self._orders[self._client_index[client_order_id]]

        order_id = str(uuid.uuid4())

        def _reject(reason: str) -> OrderResult:
            result = OrderResult(
                order_id=order_id, symbol=sym, qty=int(qty) if qty else 0, side=side,
                order_type=order_type, status="rejected", broker="paper",
                client_order_id=client_order_id, submitted_at=datetime.utcnow(),
                raw={"reason": reason},
            )
            self._orders[order_id] = result
            if client_order_id:
                self._client_index[client_order_id] = order_id
            return result

        try:
            qty = int(qty)
        except (TypeError, ValueError):
            return _reject("invalid qty")
        if qty <= 0:
            return _reject("qty must be positive")
        if fill_price is None or fill_price <= 0:
            return _reject("no valid fill price")

        fill_price = round(float(fill_price), 4)

        try:
            with get_session() as session:
                acct = self._get_or_create_today_row(session)
                pos = session.execute(
                    select(PaperPosition).where(PaperPosition.symbol == sym)
                ).scalar_one_or_none()

                filled_qty = qty
                status = "filled"

                closing = pos is not None and (
                    (side == "sell" and pos.direction == "long")
                    or (side == "buy" and pos.direction == "short")
                )

                if closing:
                    close_qty = min(qty, pos.position_size)
                    filled_qty = close_qty
                    if close_qty < qty:
                        status = "partial"
                        logger.warning(
                            "PaperBroker: %s %s qty=%d exceeds held %d — filling %d only",
                            side, sym, qty, pos.position_size, close_qty,
                        )

                    if pos.direction == "long":
                        pnl = round((fill_price - pos.entry_price) * close_qty, 2)
                        cash_delta = fill_price * close_qty
                        ret_pct = round((fill_price / pos.entry_price - 1.0) * 100.0, 4) if pos.entry_price else 0.0
                    else:
                        pnl = round((pos.entry_price - fill_price) * close_qty, 2)
                        cash_delta = -fill_price * close_qty
                        ret_pct = round((pos.entry_price / fill_price - 1.0) * 100.0, 4) if fill_price else 0.0

                    acct.ending_balance = round(float(acct.ending_balance) + cash_delta, 2)
                    acct.daily_pnl = round(float(acct.daily_pnl or 0.0) + pnl, 2)

                    session.add(PaperTrade(
                        symbol=sym, entry_date=pos.entry_date, exit_date=date.today(),
                        entry_price=pos.entry_price, exit_price=fill_price,
                        position_size=close_qty, direction=pos.direction,
                        pnl=pnl, return_percent=ret_pct, exit_reason="manual_close",
                        signal_probability=pos.signal_probability,
                        confidence_tier=pos.confidence_tier, market_phase=pos.market_phase,
                    ))

                    if close_qty >= pos.position_size:
                        session.delete(pos)
                    else:
                        pos.position_size -= close_qty
                        pos.position_value = round(pos.position_size * pos.entry_price, 2)

                else:
                    direction = "long" if side == "buy" else "short"
                    cost = round(fill_price * qty, 2)

                    if direction == "long" and cost > float(acct.ending_balance):
                        return _reject(
                            f"insufficient cash: need {cost:.2f}, have {float(acct.ending_balance):.2f}"
                        )

                    cash_delta = -cost if direction == "long" else cost
                    acct.ending_balance = round(float(acct.ending_balance) + cash_delta, 2)

                    if pos and pos.direction == direction:
                        total_cost = pos.entry_price * pos.position_size + fill_price * qty
                        new_qty = pos.position_size + qty
                        pos.entry_price = round(total_cost / new_qty, 4)
                        pos.position_size = new_qty
                        pos.position_value = round(new_qty * pos.entry_price, 2)
                        if stop_price is not None:
                            pos.stop_price = round(float(stop_price), 4)
                        if target_price is not None:
                            pos.target_price = round(float(target_price), 4)
                    else:
                        session.add(PaperPosition(
                            symbol=sym, entry_date=date.today(), entry_price=fill_price,
                            position_size=qty, position_value=cost, direction=direction,
                            stop_price=round(float(stop_price), 4) if stop_price is not None else None,
                            target_price=round(float(target_price), 4) if target_price is not None else None,
                        ))

                open_count = session.execute(
                    select(func.count(PaperPosition.id))
                ).scalar_one() or 0
                acct.open_positions = open_count
                session.commit()

                result = OrderResult(
                    order_id=order_id, symbol=sym, qty=qty, side=side, order_type=order_type,
                    status=status, filled_price=fill_price,
                    limit_price=fill_price if order_type in ("limit", "bracket") else None,
                    stop_price=float(stop_price) if stop_price is not None else None,
                    filled_qty=filled_qty, submitted_at=datetime.utcnow(), broker="paper",
                    client_order_id=client_order_id, raw={},
                )
        except Exception as exc:
            logger.error("PaperBroker: fill failed for %s: %s", sym, exc, exc_info=True)
            return _reject(str(exc))

        self._orders[order_id] = result
        if client_order_id:
            self._client_index[client_order_id] = order_id
        return result

    def place_market_order(self, symbol: str, qty: int, side: str) -> OrderResult:
        fill_price = self._get_price(symbol)
        return self._execute_fill(symbol, qty, side, fill_price, order_type="market")

    def place_limit_order(
        self, symbol: str, qty: int, side: str, limit_price: float,
        time_in_force: str = "day",
    ) -> OrderResult:
        # Simplified simulator: no resting order book, so a limit order fills
        # immediately at the limit price (time_in_force is accepted but unused).
        return self._execute_fill(symbol, qty, side, limit_price, order_type="limit")

    def place_stop_order(
        self, symbol: str, qty: int, side: str, stop_price: float,
    ) -> OrderResult:
        # Simplified simulator: treated as immediately triggered.
        return self._execute_fill(symbol, qty, side, stop_price, order_type="stop")

    def place_bracket_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        limit_price: float,
        take_profit: float,
        stop_loss: float,
        client_order_id: Optional[str] = None,
    ) -> OrderResult:
        """
        Fill immediately at `limit_price`, open a position, and record
        `stop_loss`/`take_profit` on it so mark_to_market() can close it
        later when price crosses either. Idempotent on client_order_id.
        """
        return self._execute_fill(
            symbol, qty, side, limit_price,
            stop_price=stop_loss, target_price=take_profit,
            client_order_id=client_order_id, order_type="bracket",
        )

    def get_order_by_client_id(self, client_order_id: str) -> Optional[OrderResult]:
        order_id = self._client_index.get(client_order_id)
        return self._orders.get(order_id) if order_id else None

    def cancel_order(self, order_id: str) -> bool:
        order = self._orders.get(order_id)
        if not order or order.status != "pending":
            return False
        order.status = "cancelled"
        return True

    def cancel_all_orders(self) -> int:
        cancelled = 0
        for order in self._orders.values():
            if order.status == "pending":
                order.status = "cancelled"
                cancelled += 1
        return cancelled

    def get_order_status(self, order_id: str) -> OrderResult:
        order = self._orders.get(order_id)
        if order is None:
            raise ValueError(f"PaperBroker: unknown order_id {order_id!r}")
        return order

    def get_open_orders(self) -> list[OrderResult]:
        return [o for o in self._orders.values() if o.status == "pending"]

    # ── Mark-to-market ──────────────────────────────────────────────────────

    def mark_to_market(self) -> dict:
        """
        Re-price every open position. Close any whose stop or target has
        been crossed, booking a paper_trades row with realized P&L and
        updating account cash. Returns a summary dict.
        """
        checked = 0
        skipped = 0
        closed: list[dict] = []

        try:
            with get_session() as session:
                acct = self._get_or_create_today_row(session)
                positions = session.execute(select(PaperPosition)).scalars().all()

                for pos in positions:
                    checked += 1
                    price = self._get_price(pos.symbol)
                    if price is None or price <= 0:
                        skipped += 1
                        continue

                    hit_stop = hit_target = False
                    if pos.direction == "long":
                        if pos.stop_price is not None and price <= pos.stop_price:
                            hit_stop = True
                        elif pos.target_price is not None and price >= pos.target_price:
                            hit_target = True
                    else:
                        if pos.stop_price is not None and price >= pos.stop_price:
                            hit_stop = True
                        elif pos.target_price is not None and price <= pos.target_price:
                            hit_target = True

                    if not (hit_stop or hit_target):
                        continue

                    exit_price = pos.stop_price if hit_stop else pos.target_price
                    exit_reason = "stop" if hit_stop else "target"

                    if pos.direction == "long":
                        pnl = round((exit_price - pos.entry_price) * pos.position_size, 2)
                        cash_delta = exit_price * pos.position_size
                        ret_pct = round((exit_price / pos.entry_price - 1.0) * 100.0, 4) if pos.entry_price else 0.0
                    else:
                        pnl = round((pos.entry_price - exit_price) * pos.position_size, 2)
                        cash_delta = -exit_price * pos.position_size
                        ret_pct = round((pos.entry_price / exit_price - 1.0) * 100.0, 4) if exit_price else 0.0

                    acct.ending_balance = round(float(acct.ending_balance) + cash_delta, 2)
                    acct.daily_pnl = round(float(acct.daily_pnl or 0.0) + pnl, 2)

                    session.add(PaperTrade(
                        symbol=pos.symbol, entry_date=pos.entry_date, exit_date=date.today(),
                        entry_price=pos.entry_price, exit_price=round(exit_price, 4),
                        position_size=pos.position_size, direction=pos.direction,
                        pnl=pnl, return_percent=ret_pct, exit_reason=exit_reason,
                        signal_probability=pos.signal_probability,
                        confidence_tier=pos.confidence_tier, market_phase=pos.market_phase,
                    ))
                    session.delete(pos)
                    closed.append({
                        "symbol": pos.symbol, "exit_reason": exit_reason,
                        "exit_price": round(exit_price, 4), "pnl": pnl,
                    })

                open_count = session.execute(
                    select(func.count(PaperPosition.id))
                ).scalar_one() or 0
                acct.open_positions = open_count
                session.commit()
        except Exception as exc:
            logger.error("PaperBroker: mark_to_market failed: %s", exc, exc_info=True)
            return {"checked": checked, "closed": closed, "skipped_no_quote": skipped, "error": str(exc)}

        return {
            "checked": checked,
            "closed": closed,
            "skipped_no_quote": skipped,
            "still_open": checked - len(closed),
        }

    # ── Connectivity ─────────────────────────────────────────────────────────

    def is_market_open(self) -> bool:
        try:
            from market_regime.regime_engine import _is_equity_market_open
            return _is_equity_market_open()
        except Exception as exc:
            logger.warning("PaperBroker: market hours check failed: %s", exc)
            return False

    def get_clock(self) -> dict:
        return {
            "is_open": self.is_market_open(),
            "next_open": None,
            "next_close": None,
            "timestamp": datetime.utcnow().isoformat(),
        }
