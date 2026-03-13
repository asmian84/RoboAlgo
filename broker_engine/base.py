"""
RoboAlgo — Broker Engine: Abstract Base
All broker adapters implement this interface.

Supported brokers:
  - Alpaca   (primary — commission-free, API-first, paper + live)
  - IBKR     (institutional — Interactive Brokers TWS/Gateway)
  - Questrade (Canadian retail — REST API)

Usage:
    from broker_engine.factory import get_broker
    broker = get_broker("alpaca")
    account = broker.get_account()
    result  = broker.place_market_order("TQQQ", qty=100, side="buy")
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class AccountInfo:
    account_id:    str
    cash:          float
    portfolio_val: float
    buying_power:  float
    broker:        str
    currency:      str = "USD"
    is_paper:      bool = True


@dataclass
class Position:
    symbol:          str
    qty:             int
    avg_cost:        float
    market_value:    float
    unrealized_pnl:  float
    unrealized_pct:  float
    side:            str = "long"  # long | short


@dataclass
class OrderResult:
    order_id:      str
    symbol:        str
    qty:           int
    side:          str           # buy | sell
    order_type:    str           # market | limit | stop
    status:        str           # filled | pending | partial | rejected | cancelled
    filled_price:  Optional[float] = None
    limit_price:   Optional[float] = None
    stop_price:    Optional[float] = None
    filled_qty:    int = 0
    submitted_at:  Optional[datetime] = None
    broker:        str = ""
    raw:           dict = field(default_factory=dict)  # raw broker response


@dataclass
class Quote:
    symbol:     str
    bid:        float
    ask:        float
    last:       float
    volume:     int
    timestamp:  Optional[datetime] = None

    @property
    def mid(self) -> float:
        return round((self.bid + self.ask) / 2.0, 4)


class BrokerBase(ABC):
    """
    Abstract broker interface. All live/paper broker adapters implement this contract.

    The paper_engine.PaperTrader bypasses this and uses direct DB simulation.
    This interface is for eventual live trading.
    """

    # ── Account ─────────────────────────────────────────────────────────────

    @abstractmethod
    def get_account(self) -> AccountInfo:
        """Return current account balance, buying power, and metadata."""

    @abstractmethod
    def get_positions(self) -> list[Position]:
        """Return all currently open positions."""

    @abstractmethod
    def get_position(self, symbol: str) -> Optional[Position]:
        """Return open position for a specific symbol, or None if flat."""

    # ── Market Data ─────────────────────────────────────────────────────────

    @abstractmethod
    def get_quote(self, symbol: str) -> Quote:
        """Return current bid/ask/last quote for a symbol."""

    @abstractmethod
    def get_quotes(self, symbols: list[str]) -> dict[str, Quote]:
        """Return quotes for multiple symbols in a single call."""

    # ── Orders ──────────────────────────────────────────────────────────────

    @abstractmethod
    def place_market_order(self, symbol: str, qty: int, side: str) -> OrderResult:
        """
        Place an immediate market order.
        side: "buy" | "sell"
        """

    @abstractmethod
    def place_limit_order(
        self, symbol: str, qty: int, side: str, limit_price: float,
        time_in_force: str = "day",
    ) -> OrderResult:
        """
        Place a limit order.
        time_in_force: "day" | "gtc" | "ioc" | "fok"
        """

    @abstractmethod
    def place_stop_order(
        self, symbol: str, qty: int, side: str, stop_price: float,
    ) -> OrderResult:
        """Place a stop (stop-market) order."""

    @abstractmethod
    def place_bracket_order(
        self,
        symbol:      str,
        qty:         int,
        side:        str,
        limit_price: float,
        take_profit: float,
        stop_loss:   float,
    ) -> OrderResult:
        """
        Place a bracket order: entry limit + auto take-profit + auto stop-loss.
        This is the preferred order type for RoboAlgo trade execution.
        """

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order. Returns True if successfully cancelled."""

    @abstractmethod
    def cancel_all_orders(self) -> int:
        """Cancel all open orders. Returns count of cancelled orders."""

    @abstractmethod
    def get_order_status(self, order_id: str) -> OrderResult:
        """Fetch current status of an order by ID."""

    @abstractmethod
    def get_open_orders(self) -> list[OrderResult]:
        """Return all currently open/pending orders."""

    # ── Connectivity ─────────────────────────────────────────────────────────

    @abstractmethod
    def is_market_open(self) -> bool:
        """Return True if the market is currently open for trading."""

    @abstractmethod
    def get_clock(self) -> dict:
        """Return market hours info: {is_open, next_open, next_close}."""

    # ── Broker metadata ──────────────────────────────────────────────────────

    @property
    @abstractmethod
    def name(self) -> str:
        """Broker name identifier: 'alpaca' | 'ibkr' | 'questrade'"""

    @property
    @abstractmethod
    def is_paper(self) -> bool:
        """True if this is a paper/simulated trading account."""
