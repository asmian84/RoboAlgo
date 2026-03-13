"""
RoboAlgo — IBKR Broker Adapter (Interactive Brokers)
Connects via ib_insync to TWS or IB Gateway.

Setup:
  pip install ib_insync
  .env: IBKR_HOST=127.0.0.1, IBKR_PORT=7497 (paper TWS) or 7496 (live TWS)
        IBKR_CLIENT_ID=1

TWS paper port:  7497
TWS live port:   7496
Gateway paper:   4002
Gateway live:    4001

Docs: https://ib-insync.readthedocs.io/
"""

import os
import logging
from typing import Optional

from broker_engine.base import BrokerBase, AccountInfo, Position, OrderResult, Quote

logger = logging.getLogger(__name__)


class IBKRBroker(BrokerBase):
    """
    Interactive Brokers adapter via ib_insync.
    Requires TWS or IB Gateway running locally on IBKR_PORT.
    """

    def __init__(self):
        self._host       = os.getenv("IBKR_HOST",      "127.0.0.1")
        self._port       = int(os.getenv("IBKR_PORT",  "7497"))
        self._client_id  = int(os.getenv("IBKR_CLIENT_ID", "1"))
        self._paper_mode = self._port in (7497, 4002)   # Paper ports
        self._ib         = None

    @property
    def name(self) -> str:
        return "ibkr"

    @property
    def is_paper(self) -> bool:
        return self._paper_mode

    def _connect(self):
        if self._ib is None or not self._ib.isConnected():
            try:
                from ib_insync import IB
                self._ib = IB()
                self._ib.connect(
                    host=self._host,
                    port=self._port,
                    clientId=self._client_id,
                    timeout=15,
                )
                logger.info(
                    f"IBKR connected on {self._host}:{self._port} "
                    f"(paper={self._paper_mode})"
                )
            except ImportError:
                raise RuntimeError(
                    "ib_insync not installed. Run: pip install ib_insync"
                )
            except Exception as e:
                raise ConnectionError(
                    f"Cannot connect to IBKR TWS/Gateway on {self._host}:{self._port}. "
                    f"Make sure TWS or IB Gateway is running and API connections are enabled. "
                    f"Error: {e}"
                )
        return self._ib

    def disconnect(self):
        if self._ib and self._ib.isConnected():
            self._ib.disconnect()
            self._ib = None

    # ── Account ─────────────────────────────────────────────────────────────

    def get_account(self) -> AccountInfo:
        ib = self._connect()
        summary = {v.tag: v.value for v in ib.accountSummary()}
        return AccountInfo(
            account_id=summary.get("AccountCode", "IBKR"),
            cash=float(summary.get("CashBalance", 0)),
            portfolio_val=float(summary.get("NetLiquidation", 0)),
            buying_power=float(summary.get("BuyingPower", 0)),
            broker="ibkr",
            currency="USD",
            is_paper=self._paper_mode,
        )

    def get_positions(self) -> list[Position]:
        ib = self._connect()
        raw_positions = ib.positions()
        result = []
        for pos in raw_positions:
            if pos.position == 0:
                continue
            result.append(Position(
                symbol=pos.contract.symbol,
                qty=int(pos.position),
                avg_cost=float(pos.avgCost),
                market_value=float(pos.position * pos.avgCost),
                unrealized_pnl=0.0,   # requires market data subscription
                unrealized_pct=0.0,
                side="long" if pos.position > 0 else "short",
            ))
        return result

    def get_position(self, symbol: str) -> Optional[Position]:
        positions = self.get_positions()
        return next((p for p in positions if p.symbol.upper() == symbol.upper()), None)

    # ── Market Data ─────────────────────────────────────────────────────────

    def get_quote(self, symbol: str) -> Quote:
        from ib_insync import Stock
        ib = self._connect()
        contract = Stock(symbol.upper(), "SMART", "USD")
        ib.qualifyContracts(contract)
        ticker = ib.reqMktData(contract, "", False, False)
        ib.sleep(1)  # wait for data
        return Quote(
            symbol=symbol.upper(),
            bid=float(ticker.bid or 0),
            ask=float(ticker.ask or 0),
            last=float(ticker.last or ticker.close or 0),
            volume=int(ticker.volume or 0),
        )

    def get_quotes(self, symbols: list[str]) -> dict[str, Quote]:
        return {s: self.get_quote(s) for s in symbols}

    # ── Orders ──────────────────────────────────────────────────────────────

    def place_market_order(self, symbol: str, qty: int, side: str) -> OrderResult:
        from ib_insync import Stock, MarketOrder
        ib = self._connect()
        contract = Stock(symbol.upper(), "SMART", "USD")
        action = "BUY" if side.lower() == "buy" else "SELL"
        order = MarketOrder(action=action, totalQuantity=qty)
        trade = ib.placeOrder(contract, order)
        ib.sleep(1)
        return self._map_trade(trade, symbol)

    def place_limit_order(
        self, symbol: str, qty: int, side: str, limit_price: float,
        time_in_force: str = "day",
    ) -> OrderResult:
        from ib_insync import Stock, LimitOrder
        ib = self._connect()
        contract = Stock(symbol.upper(), "SMART", "USD")
        action = "BUY" if side.lower() == "buy" else "SELL"
        tif = "DAY" if time_in_force == "day" else "GTC"
        order = LimitOrder(action=action, totalQuantity=qty,
                           lmtPrice=round(limit_price, 2), tif=tif)
        trade = ib.placeOrder(contract, order)
        ib.sleep(1)
        return self._map_trade(trade, symbol)

    def place_stop_order(
        self, symbol: str, qty: int, side: str, stop_price: float,
    ) -> OrderResult:
        from ib_insync import Stock, StopOrder
        ib = self._connect()
        contract = Stock(symbol.upper(), "SMART", "USD")
        action = "BUY" if side.lower() == "buy" else "SELL"
        order = StopOrder(action=action, totalQuantity=qty,
                          stopPrice=round(stop_price, 2))
        trade = ib.placeOrder(contract, order)
        ib.sleep(1)
        return self._map_trade(trade, symbol)

    def place_bracket_order(
        self, symbol: str, qty: int, side: str,
        limit_price: float, take_profit: float, stop_loss: float,
    ) -> OrderResult:
        from ib_insync import Stock, BracketOrder
        ib = self._connect()
        contract = Stock(symbol.upper(), "SMART", "USD")
        action = "BUY" if side.lower() == "buy" else "SELL"
        bracket = ib.bracketOrder(
            action=action,
            quantity=qty,
            limitPrice=round(limit_price, 2),
            takeProfitPrice=round(take_profit, 2),
            stopLossPrice=round(stop_loss, 2),
        )
        trades = [ib.placeOrder(contract, o) for o in bracket]
        ib.sleep(1)
        return self._map_trade(trades[0], symbol)   # return parent order

    def cancel_order(self, order_id: str) -> bool:
        ib = self._connect()
        open_trades = ib.openTrades()
        for trade in open_trades:
            if str(trade.order.orderId) == order_id:
                ib.cancelOrder(trade.order)
                return True
        return False

    def cancel_all_orders(self) -> int:
        ib = self._connect()
        open_trades = ib.openTrades()
        for trade in open_trades:
            ib.cancelOrder(trade.order)
        return len(open_trades)

    def get_order_status(self, order_id: str) -> OrderResult:
        ib = self._connect()
        for trade in ib.openTrades():
            if str(trade.order.orderId) == order_id:
                return self._map_trade(trade, trade.contract.symbol)
        raise ValueError(f"Order {order_id} not found in open trades.")

    def get_open_orders(self) -> list[OrderResult]:
        ib = self._connect()
        trades = ib.openTrades()
        return [self._map_trade(t, t.contract.symbol) for t in trades]

    # ── Connectivity ─────────────────────────────────────────────────────────

    def is_market_open(self) -> bool:
        ib = self._connect()
        # Check using current time vs NYSE hours (simplified)
        from datetime import datetime, time
        import pytz
        now = datetime.now(pytz.timezone("America/New_York"))
        if now.weekday() >= 5:  # Saturday/Sunday
            return False
        return time(9, 30) <= now.time() <= time(16, 0)

    def get_clock(self) -> dict:
        from datetime import datetime, timedelta
        import pytz
        tz = pytz.timezone("America/New_York")
        now = datetime.now(tz)
        return {
            "is_open":    self.is_market_open(),
            "timestamp":  now.isoformat(),
            "broker":     "ibkr",
        }

    # ── Mapper ───────────────────────────────────────────────────────────────

    @staticmethod
    def _map_trade(trade, symbol: str) -> OrderResult:
        order  = trade.order
        status = trade.orderStatus
        return OrderResult(
            order_id=str(order.orderId),
            symbol=symbol.upper(),
            qty=int(order.totalQuantity),
            side=order.action.lower(),
            order_type=order.orderType.lower(),
            status=status.status.lower(),
            filled_price=float(status.avgFillPrice) if status.avgFillPrice else None,
            limit_price=float(order.lmtPrice) if hasattr(order, "lmtPrice") else None,
            stop_price=float(order.auxPrice) if hasattr(order, "auxPrice") else None,
            filled_qty=int(status.filled),
            broker="ibkr",
        )
