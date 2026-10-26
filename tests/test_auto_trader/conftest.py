"""
tests/test_auto_trader/conftest.py
────────────────────────────────────────────────────────────────────────────────
Shared test doubles and helpers for the auto_trader safety-component tests.

FakeBroker      — an in-memory BrokerBase implementation. Every abstract method
                  is implemented; `fail_on` lets a test force a specific call to
                  raise, to exercise the reconciler's degrade-instead-of-raise
                  paths.
RecordingAudit  — a drop-in replacement for `auto_trader.audit` that appends to
                  plain lists instead of touching the database. Pass it as
                  `audit_mod=RecordingAudit()` to `reconciler.reconcile()`.
snapshot()      — builds a `PortfolioSnapshot` with sane defaults, overridable
                  via kwargs.

NOTE on SQLite: `database/models.py` declares `ix_paper_positions_symbol` twice,
so a bare `Base.metadata.create_all()` crashes on SQLite. None of the tests in
this package touch the database (reconciler tests inject RecordingAudit;
config-gate tests never import auto_trader.audit at all), so no engine/tables
are created here. If a future test genuinely needs auto_trader's own tables,
create ONLY the three auto_trader tables, e.g.:

    Base.metadata.create_all(engine, tables=[
        AutoTraderCycle.__table__, AutoTraderDecision.__table__,
        AutoTraderDiscrepancy.__table__,
    ])

never a bare `Base.metadata.create_all(engine)`.
"""
from __future__ import annotations

from typing import Optional

from broker_engine.base import AccountInfo, BrokerBase, OrderResult, Position, Quote
from auto_trader.governor import PortfolioSnapshot


class FakeBroker(BrokerBase):
    """In-memory BrokerBase double for auto_trader tests."""

    def __init__(self, *, is_paper: bool = True, account: Optional[AccountInfo] = None):
        self._positions: dict[str, Position] = {}
        self._orders: list[OrderResult] = []
        self._is_paper = is_paper
        self._account = account or AccountInfo(
            account_id="fake-account",
            cash=100_000.0,
            portfolio_val=100_000.0,
            buying_power=100_000.0,
            broker="fake",
            is_paper=is_paper,
        )
        # Any method name placed in here raises RuntimeError when called.
        self.fail_on: set[str] = set()
        # Every place_bracket_order call, including client_order_id, in order.
        self.bracket_calls: list[dict] = []
        self._next_order_id = 1

    # ── failure injection ───────────────────────────────────────────────────
    def _check_fail(self, name: str) -> None:
        if name in self.fail_on:
            raise RuntimeError(f"FakeBroker: simulated failure in {name}")

    def _new_order_id(self) -> str:
        oid = f"fake-order-{self._next_order_id}"
        self._next_order_id += 1
        return oid

    # ── Account ─────────────────────────────────────────────────────────────
    def get_account(self) -> AccountInfo:
        self._check_fail("get_account")
        return self._account

    def get_positions(self) -> list[Position]:
        self._check_fail("get_positions")
        return list(self._positions.values())

    def get_position(self, symbol: str) -> Optional[Position]:
        self._check_fail("get_position")
        return self._positions.get(symbol.upper())

    # ── Market data ─────────────────────────────────────────────────────────
    def get_quote(self, symbol: str) -> Quote:
        self._check_fail("get_quote")
        return Quote(symbol=symbol, bid=100.0, ask=100.1, last=100.05, volume=1_000_000)

    def get_quotes(self, symbols: list[str]) -> dict[str, Quote]:
        self._check_fail("get_quotes")
        return {s: self.get_quote(s) for s in symbols}

    # ── Orders ──────────────────────────────────────────────────────────────
    def place_market_order(self, symbol: str, qty: int, side: str) -> OrderResult:
        self._check_fail("place_market_order")
        order = OrderResult(
            order_id=self._new_order_id(), symbol=symbol, qty=qty, side=side,
            order_type="market", status="pending", broker=self.name,
        )
        self._orders.append(order)
        return order

    def place_limit_order(
        self, symbol: str, qty: int, side: str, limit_price: float,
        time_in_force: str = "day",
    ) -> OrderResult:
        self._check_fail("place_limit_order")
        order = OrderResult(
            order_id=self._new_order_id(), symbol=symbol, qty=qty, side=side,
            order_type="limit", status="pending", limit_price=limit_price,
            broker=self.name,
        )
        self._orders.append(order)
        return order

    def place_stop_order(
        self, symbol: str, qty: int, side: str, stop_price: float,
    ) -> OrderResult:
        self._check_fail("place_stop_order")
        order = OrderResult(
            order_id=self._new_order_id(), symbol=symbol, qty=qty, side=side,
            order_type="stop", status="pending", stop_price=stop_price,
            broker=self.name,
        )
        self._orders.append(order)
        return order

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
        self._check_fail("place_bracket_order")
        self.bracket_calls.append({
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "limit_price": limit_price,
            "take_profit": take_profit,
            "stop_loss": stop_loss,
            "client_order_id": client_order_id,
        })
        order = OrderResult(
            order_id=self._new_order_id(), symbol=symbol, qty=qty, side=side,
            order_type="bracket", status="pending", limit_price=limit_price,
            broker=self.name, client_order_id=client_order_id,
            raw={"take_profit": take_profit, "stop_loss": stop_loss},
        )
        self._orders.append(order)
        return order

    def cancel_order(self, order_id: str) -> bool:
        self._check_fail("cancel_order")
        for o in self._orders:
            if o.order_id == order_id:
                o.status = "cancelled"
                return True
        return False

    def cancel_all_orders(self) -> int:
        self._check_fail("cancel_all_orders")
        count = 0
        for o in self._orders:
            if o.status == "pending":
                o.status = "cancelled"
                count += 1
        return count

    def get_order_status(self, order_id: str) -> OrderResult:
        self._check_fail("get_order_status")
        for o in self._orders:
            if o.order_id == order_id:
                return o
        raise KeyError(f"no such order_id: {order_id}")

    def get_open_orders(self) -> list[OrderResult]:
        self._check_fail("get_open_orders")
        return [o for o in self._orders if o.status == "pending"]

    def get_order_by_client_id(self, client_order_id: str) -> Optional[OrderResult]:
        self._check_fail("get_order_by_client_id")
        for o in self._orders:
            if o.client_order_id == client_order_id:
                return o
        return None

    # ── Connectivity ────────────────────────────────────────────────────────
    def is_market_open(self) -> bool:
        self._check_fail("is_market_open")
        return True

    def get_clock(self) -> dict:
        self._check_fail("get_clock")
        return {"is_open": True, "next_open": None, "next_close": None}

    # ── Broker metadata ─────────────────────────────────────────────────────
    @property
    def name(self) -> str:
        return "fake"

    @property
    def is_paper(self) -> bool:
        return self._is_paper

    @is_paper.setter
    def is_paper(self, value: bool) -> None:
        self._is_paper = value


class RecordingAudit:
    """Drop-in `audit_mod` double: records calls instead of touching the DB."""

    def __init__(self):
        self.decisions: list[dict] = []
        self.discrepancies: list[dict] = []
        self.cycles_started: list[dict] = []
        self.cycles_finished: list[dict] = []

    def log_decision(self, cycle_id, symbol, action, reason, **fields) -> None:
        self.decisions.append({
            "cycle_id": cycle_id, "symbol": symbol, "action": action,
            "reason": reason, **fields,
        })

    def log_discrepancy(self, cycle_id, kind, symbol, expected, actual, resolution, detail="") -> None:
        self.discrepancies.append({
            "cycle_id": cycle_id, "kind": kind, "symbol": symbol,
            "expected": expected, "actual": actual, "resolution": resolution,
            "detail": detail,
        })

    def start_cycle(self, cycle_id, mode, broker, equity_start, day_start_equity) -> None:
        self.cycles_started.append({
            "cycle_id": cycle_id, "mode": mode, "broker": broker,
            "equity_start": equity_start, "day_start_equity": day_start_equity,
        })

    def finish_cycle(self, cycle_id, status, **kwargs) -> None:
        self.cycles_finished.append({"cycle_id": cycle_id, "status": status, **kwargs})


def snapshot(**overrides) -> PortfolioSnapshot:
    """Build a PortfolioSnapshot with safe defaults, overridable via kwargs."""
    defaults = dict(
        equity=100_000.0,
        cash=50_000.0,
        positions=[],
        day_start_equity=100_000.0,
        consecutive_losses=0,
    )
    defaults.update(overrides)
    return PortfolioSnapshot(**defaults)
