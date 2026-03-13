"""
RoboAlgo — Questrade Broker Adapter
Canadian retail broker via Questrade REST API.

Setup:
  pip install questrade-api  (or use requests directly)
  .env: QUESTRADE_REFRESH_TOKEN=<your_token>

Tokens: https://login.questrade.com/APIAccess/userapps.aspx
Docs:   https://www.questrade.com/api/documentation/getting-started

Note:
  Questrade OAuth access tokens expire every 30 minutes.
  This adapter handles automatic token refresh via the refresh token.
  Questrade only supports Canadian & US equities (no crypto).
  PDT rules apply (Pattern Day Trader) for US margin accounts.
"""

import os
import logging
import time
from typing import Optional
import requests

from broker_engine.base import BrokerBase, AccountInfo, Position, OrderResult, Quote

logger = logging.getLogger(__name__)

_QT_AUTH_URL = "https://login.questrade.com/oauth2/token"


class QuestradeBroker(BrokerBase):
    """
    Questrade broker adapter using direct REST API calls.
    Automatically refreshes OAuth tokens before expiry.
    """

    def __init__(self):
        self._refresh_token = os.getenv("QUESTRADE_REFRESH_TOKEN")
        self._access_token  = None
        self._api_server    = None
        self._token_expiry  = 0  # epoch timestamp

        if not self._refresh_token:
            raise EnvironmentError(
                "QUESTRADE_REFRESH_TOKEN must be set in .env. "
                "Generate at: https://login.questrade.com/APIAccess/userapps.aspx"
            )

        # Authenticate on init
        self._refresh_access_token()

    @property
    def name(self) -> str:
        return "questrade"

    @property
    def is_paper(self) -> bool:
        # Questrade does not offer a true paper trading environment
        return False

    # ── Auth ─────────────────────────────────────────────────────────────────

    def _refresh_access_token(self):
        """Exchange refresh token for a new access token."""
        resp = requests.post(
            _QT_AUTH_URL,
            params={"grant_type": "refresh_token", "refresh_token": self._refresh_token},
        )
        resp.raise_for_status()
        data = resp.json()
        self._access_token  = data["access_token"]
        self._refresh_token = data["refresh_token"]   # new refresh token
        self._api_server    = data["api_server"]
        self._token_expiry  = time.time() + data.get("expires_in", 1800) - 60

        # Persist new refresh token (so it doesn't expire)
        if os.path.exists(".env"):
            with open(".env", "r") as f:
                content = f.read()
            new_content = []
            for line in content.splitlines():
                if line.startswith("QUESTRADE_REFRESH_TOKEN="):
                    new_content.append(f"QUESTRADE_REFRESH_TOKEN={self._refresh_token}")
                else:
                    new_content.append(line)
            with open(".env", "w") as f:
                f.write("\n".join(new_content) + "\n")

    def _ensure_token_valid(self):
        if time.time() >= self._token_expiry:
            self._refresh_access_token()

    def _headers(self) -> dict:
        self._ensure_token_valid()
        return {"Authorization": f"Bearer {self._access_token}"}

    def _get(self, path: str, **params) -> dict:
        url = f"{self._api_server}v1/{path}"
        resp = requests.get(url, headers=self._headers(), params=params)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, body: dict) -> dict:
        url = f"{self._api_server}v1/{path}"
        resp = requests.post(url, headers=self._headers(), json=body)
        resp.raise_for_status()
        return resp.json()

    def _delete(self, path: str) -> dict:
        url = f"{self._api_server}v1/{path}"
        resp = requests.delete(url, headers=self._headers())
        resp.raise_for_status()
        return resp.json() if resp.content else {}

    # ── Account ─────────────────────────────────────────────────────────────

    def get_account(self) -> AccountInfo:
        data = self._get("accounts")
        acct = data["accounts"][0]
        balances = self._get(f"accounts/{acct['number']}/balances")
        combined = next(
            (b for b in balances["combinedBalances"] if b["currency"] == "USD"),
            balances["combinedBalances"][0] if balances["combinedBalances"] else {}
        )
        return AccountInfo(
            account_id=str(acct["number"]),
            cash=float(combined.get("cash", 0)),
            portfolio_val=float(combined.get("totalEquity", 0)),
            buying_power=float(combined.get("buyingPower", 0)),
            broker="questrade",
            currency=combined.get("currency", "USD"),
            is_paper=False,
        )

    def get_positions(self) -> list[Position]:
        data = self._get("accounts")
        acct_num = data["accounts"][0]["number"]
        pos_data = self._get(f"accounts/{acct_num}/positions")
        return [self._map_position(p) for p in pos_data.get("positions", [])]

    def get_position(self, symbol: str) -> Optional[Position]:
        positions = self.get_positions()
        return next((p for p in positions if p.symbol.upper() == symbol.upper()), None)

    # ── Market Data ─────────────────────────────────────────────────────────

    def get_quote(self, symbol: str) -> Quote:
        # First, get symbol ID
        sym_data = self._get("symbols/search", prefix=symbol.upper())
        symbols  = sym_data.get("symbols", [])
        if not symbols:
            return Quote(symbol=symbol, bid=0, ask=0, last=0, volume=0)
        sym_id = symbols[0]["symbolId"]
        q_data = self._get(f"markets/quotes/{sym_id}")
        q = q_data["quotes"][0]
        return Quote(
            symbol=symbol.upper(),
            bid=float(q.get("bidPrice") or 0),
            ask=float(q.get("askPrice") or 0),
            last=float(q.get("lastTradePrice") or 0),
            volume=int(q.get("volume") or 0),
        )

    def get_quotes(self, symbols: list[str]) -> dict[str, Quote]:
        return {s: self.get_quote(s) for s in symbols}

    # ── Orders ──────────────────────────────────────────────────────────────

    def _get_account_number(self) -> str:
        data = self._get("accounts")
        return str(data["accounts"][0]["number"])

    def _get_symbol_id(self, symbol: str) -> int:
        sym_data = self._get("symbols/search", prefix=symbol.upper())
        symbols  = sym_data.get("symbols", [])
        if not symbols:
            raise ValueError(f"Symbol not found on Questrade: {symbol}")
        return symbols[0]["symbolId"]

    def place_market_order(self, symbol: str, qty: int, side: str) -> OrderResult:
        acct = self._get_account_number()
        sym_id = self._get_symbol_id(symbol)
        body = {
            "accountNumber": acct,
            "symbolId": sym_id,
            "quantity": qty,
            "action": "Buy" if side.lower() == "buy" else "Sell",
            "orderType": "Market",
            "timeInForce": "Day",
        }
        result = self._post(f"accounts/{acct}/orders", body)
        return self._map_order(result.get("orders", [{}])[0], symbol)

    def place_limit_order(
        self, symbol: str, qty: int, side: str, limit_price: float,
        time_in_force: str = "day",
    ) -> OrderResult:
        acct = self._get_account_number()
        sym_id = self._get_symbol_id(symbol)
        tif_map = {"day": "Day", "gtc": "GoodTillCanceled"}
        body = {
            "accountNumber": acct,
            "symbolId": sym_id,
            "quantity": qty,
            "action": "Buy" if side.lower() == "buy" else "Sell",
            "orderType": "Limit",
            "limitPrice": round(limit_price, 2),
            "timeInForce": tif_map.get(time_in_force, "Day"),
        }
        result = self._post(f"accounts/{acct}/orders", body)
        return self._map_order(result.get("orders", [{}])[0], symbol)

    def place_stop_order(
        self, symbol: str, qty: int, side: str, stop_price: float,
    ) -> OrderResult:
        acct = self._get_account_number()
        sym_id = self._get_symbol_id(symbol)
        body = {
            "accountNumber": acct,
            "symbolId": sym_id,
            "quantity": qty,
            "action": "Buy" if side.lower() == "buy" else "SELL",
            "orderType": "Stop",
            "stopPrice": round(stop_price, 2),
            "timeInForce": "Day",
        }
        result = self._post(f"accounts/{acct}/orders", body)
        return self._map_order(result.get("orders", [{}])[0], symbol)

    def place_bracket_order(
        self, symbol: str, qty: int, side: str,
        limit_price: float, take_profit: float, stop_loss: float,
    ) -> OrderResult:
        """
        Questrade doesn't natively support bracket orders.
        Simulated: entry limit + separate stop and limit-profit orders.
        """
        # Place entry
        entry = self.place_limit_order(symbol, qty, side, limit_price)
        # Companion exit orders (submitted immediately; attach manually)
        exit_side = "sell" if side.lower() == "buy" else "buy"
        self.place_limit_order(symbol, qty, exit_side, take_profit, "gtc")
        self.place_stop_order(symbol, qty, exit_side, stop_loss)
        return entry

    def cancel_order(self, order_id: str) -> bool:
        acct = self._get_account_number()
        try:
            self._delete(f"accounts/{acct}/orders/{order_id}")
            return True
        except Exception as e:
            logger.warning(f"Cancel order {order_id} failed: {e}")
            return False

    def cancel_all_orders(self) -> int:
        acct = self._get_account_number()
        orders = self._get(f"accounts/{acct}/orders", stateFilter="Open")
        count = 0
        for o in orders.get("orders", []):
            if self.cancel_order(str(o["id"])):
                count += 1
        return count

    def get_order_status(self, order_id: str) -> OrderResult:
        acct = self._get_account_number()
        data = self._get(f"accounts/{acct}/orders/{order_id}")
        return self._map_order(data.get("orders", [{}])[0], "")

    def get_open_orders(self) -> list[OrderResult]:
        acct = self._get_account_number()
        data = self._get(f"accounts/{acct}/orders", stateFilter="Open")
        return [self._map_order(o, o.get("symbol", "")) for o in data.get("orders", [])]

    # ── Connectivity ─────────────────────────────────────────────────────────

    def is_market_open(self) -> bool:
        from datetime import datetime, time
        import pytz
        now = datetime.now(pytz.timezone("America/New_York"))
        if now.weekday() >= 5:
            return False
        return time(9, 30) <= now.time() <= time(16, 0)

    def get_clock(self) -> dict:
        from datetime import datetime
        import pytz
        now = datetime.now(pytz.timezone("America/New_York"))
        return {"is_open": self.is_market_open(), "timestamp": now.isoformat(), "broker": "questrade"}

    # ── Mappers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _map_position(p: dict) -> Position:
        qty = int(p.get("openQuantity", 0))
        return Position(
            symbol=p.get("symbol", ""),
            qty=abs(qty),
            avg_cost=float(p.get("averageEntryPrice", 0)),
            market_value=float(p.get("currentMarketValue", 0)),
            unrealized_pnl=float(p.get("openPnl", 0)),
            unrealized_pct=float(p.get("openPnl", 0)) / max(float(p.get("currentMarketValue", 1)), 1) * 100,
            side="long" if qty >= 0 else "short",
        )

    @staticmethod
    def _map_order(o: dict, symbol: str) -> OrderResult:
        status_map = {
            "FilledAll":   "filled",
            "FilledPart":  "partial",
            "Pending":     "pending",
            "Cancelled":   "cancelled",
            "Rejected":    "rejected",
        }
        return OrderResult(
            order_id=str(o.get("id", "")),
            symbol=o.get("symbol", symbol),
            qty=int(o.get("totalQuantity", 0)),
            side=o.get("action", "").lower(),
            order_type=o.get("orderType", "").lower(),
            status=status_map.get(o.get("state", ""), o.get("state", "unknown").lower()),
            filled_price=float(o["avgExecPrice"]) if o.get("avgExecPrice") else None,
            limit_price=float(o["limitPrice"]) if o.get("limitPrice") else None,
            stop_price=float(o["stopPrice"]) if o.get("stopPrice") else None,
            filled_qty=int(o.get("filledQuantity", 0)),
            broker="questrade",
        )
