"""
RoboAlgo — Alpaca Broker Adapter
Commission-free trading via Alpaca Markets API.

Setup:
  pip install alpaca-py
  .env: ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER=true

Paper: https://paper-api.alpaca.markets
Live:  https://api.alpaca.markets

Docs: https://docs.alpaca.markets/
"""

import os
import logging
from datetime import datetime
from typing import Optional

from broker_engine.base import BrokerBase, AccountInfo, Position, OrderResult, Quote

logger = logging.getLogger(__name__)


class AlpacaBroker(BrokerBase):
    """
    Alpaca Markets adapter using alpaca-py SDK.
    Supports both paper and live trading accounts.
    """

    def __init__(self):
        self._api_key    = os.getenv("ALPACA_API_KEY")
        self._secret_key = os.getenv("ALPACA_SECRET_KEY")
        self._paper      = os.getenv("ALPACA_PAPER", "true").lower() != "false"
        self._client     = None
        self._data_client = None

        if not self._api_key or not self._secret_key:
            raise EnvironmentError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env"
            )

    @property
    def name(self) -> str:
        return "alpaca"

    @property
    def is_paper(self) -> bool:
        return self._paper

    def _get_client(self):
        if self._client is None:
            try:
                from alpaca.trading.client import TradingClient
                self._client = TradingClient(
                    api_key=self._api_key,
                    secret_key=self._secret_key,
                    paper=self._paper,
                )
            except ImportError:
                raise RuntimeError(
                    "alpaca-py not installed. Run: pip install alpaca-py"
                )
        return self._client

    def _get_data_client(self):
        if self._data_client is None:
            try:
                from alpaca.data.historical import StockHistoricalDataClient
                self._data_client = StockHistoricalDataClient(
                    api_key=self._api_key,
                    secret_key=self._secret_key,
                )
            except ImportError:
                raise RuntimeError(
                    "alpaca-py not installed. Run: pip install alpaca-py"
                )
        return self._data_client

    # ── Account ─────────────────────────────────────────────────────────────

    def get_account(self) -> AccountInfo:
        client = self._get_client()
        acct = client.get_account()
        return AccountInfo(
            account_id=str(acct.account_number),
            cash=float(acct.cash),
            portfolio_val=float(acct.portfolio_value),
            buying_power=float(acct.buying_power),
            broker="alpaca",
            currency="USD",
            is_paper=self._paper,
        )

    def get_positions(self) -> list[Position]:
        client = self._get_client()
        positions = client.get_all_positions()
        return [self._map_position(p) for p in positions]

    def get_position(self, symbol: str) -> Optional[Position]:
        client = self._get_client()
        try:
            pos = client.get_open_position(symbol.upper())
            return self._map_position(pos)
        except Exception:
            return None

    # ── Market Data ─────────────────────────────────────────────────────────

    def get_quote(self, symbol: str) -> Quote:
        try:
            from alpaca.data.requests import StockLatestQuoteRequest
            dc = self._get_data_client()
            req = StockLatestQuoteRequest(symbol_or_symbols=[symbol.upper()])
            quotes = dc.get_stock_latest_quote(req)
            q = quotes[symbol.upper()]
            return Quote(
                symbol=symbol.upper(),
                bid=float(q.bid_price or 0),
                ask=float(q.ask_price or 0),
                last=float(q.ask_price or q.bid_price or 0),
                volume=int(q.bid_size + q.ask_size),
                timestamp=q.timestamp,
            )
        except Exception as e:
            logger.warning(f"Alpaca quote failed for {symbol}: {e}")
            return Quote(symbol=symbol, bid=0, ask=0, last=0, volume=0)

    def get_quotes(self, symbols: list[str]) -> dict[str, Quote]:
        return {s: self.get_quote(s) for s in symbols}

    # ── Orders ──────────────────────────────────────────────────────────────

    def place_market_order(self, symbol: str, qty: int, side: str) -> OrderResult:
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        client = self._get_client()
        req = MarketOrderRequest(
            symbol=symbol.upper(),
            qty=qty,
            side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        order = client.submit_order(req)
        return self._map_order(order)

    def place_limit_order(
        self, symbol: str, qty: int, side: str, limit_price: float,
        time_in_force: str = "day",
    ) -> OrderResult:
        from alpaca.trading.requests import LimitOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        client = self._get_client()
        tif_map = {"day": TimeInForce.DAY, "gtc": TimeInForce.GTC,
                   "ioc": TimeInForce.IOC, "fok": TimeInForce.FOK}
        req = LimitOrderRequest(
            symbol=symbol.upper(),
            qty=qty,
            side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
            limit_price=round(limit_price, 2),
            time_in_force=tif_map.get(time_in_force, TimeInForce.DAY),
        )
        order = client.submit_order(req)
        return self._map_order(order)

    def place_stop_order(
        self, symbol: str, qty: int, side: str, stop_price: float,
    ) -> OrderResult:
        from alpaca.trading.requests import StopOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        client = self._get_client()
        req = StopOrderRequest(
            symbol=symbol.upper(),
            qty=qty,
            side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
            stop_price=round(stop_price, 2),
            time_in_force=TimeInForce.DAY,
        )
        order = client.submit_order(req)
        return self._map_order(order)

    def place_bracket_order(
        self, symbol: str, qty: int, side: str,
        limit_price: float, take_profit: float, stop_loss: float,
    ) -> OrderResult:
        """
        Bracket order: entry limit + auto take-profit limit + auto stop-loss.
        Preferred order type for RoboAlgo — enters and sets exits in one shot.
        """
        from alpaca.trading.requests import (
            LimitOrderRequest, TakeProfitRequest, StopLossRequest
        )
        from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
        client = self._get_client()
        req = LimitOrderRequest(
            symbol=symbol.upper(),
            qty=qty,
            side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
            limit_price=round(limit_price, 2),
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=round(take_profit, 2)),
            stop_loss=StopLossRequest(stop_price=round(stop_loss, 2)),
        )
        order = client.submit_order(req)
        return self._map_order(order)

    def cancel_order(self, order_id: str) -> bool:
        client = self._get_client()
        try:
            client.cancel_order_by_id(order_id)
            return True
        except Exception as e:
            logger.warning(f"Cancel order {order_id} failed: {e}")
            return False

    def cancel_all_orders(self) -> int:
        client = self._get_client()
        cancelled = client.cancel_orders()
        return len(cancelled) if cancelled else 0

    def get_order_status(self, order_id: str) -> OrderResult:
        client = self._get_client()
        order = client.get_order_by_id(order_id)
        return self._map_order(order)

    def get_open_orders(self) -> list[OrderResult]:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        client = self._get_client()
        orders = client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN))
        return [self._map_order(o) for o in orders]

    # ── Connectivity ─────────────────────────────────────────────────────────

    def is_market_open(self) -> bool:
        client = self._get_client()
        clock = client.get_clock()
        return clock.is_open

    def get_clock(self) -> dict:
        client = self._get_client()
        clock = client.get_clock()
        return {
            "is_open":    clock.is_open,
            "next_open":  clock.next_open.isoformat() if clock.next_open else None,
            "next_close": clock.next_close.isoformat() if clock.next_close else None,
            "timestamp":  clock.timestamp.isoformat() if clock.timestamp else None,
        }

    # ── Mappers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _map_position(pos) -> Position:
        return Position(
            symbol=pos.symbol,
            qty=int(float(pos.qty)),
            avg_cost=float(pos.avg_entry_price or 0),
            market_value=float(pos.market_value or 0),
            unrealized_pnl=float(pos.unrealized_pl or 0),
            unrealized_pct=float(pos.unrealized_plpc or 0) * 100,
            side=str(pos.side),
        )

    @staticmethod
    def _map_order(order) -> OrderResult:
        return OrderResult(
            order_id=str(order.id),
            symbol=order.symbol,
            qty=int(float(order.qty or 0)),
            side=str(order.side),
            order_type=str(order.order_type),
            status=str(order.status),
            filled_price=float(order.filled_avg_price) if order.filled_avg_price else None,
            limit_price=float(order.limit_price) if order.limit_price else None,
            stop_price=float(order.stop_price) if order.stop_price else None,
            filled_qty=int(float(order.filled_qty or 0)),
            submitted_at=order.submitted_at,
            broker="alpaca",
        )
