"""
auto_trader.reconciler — the broker is always the source of truth.

Every cycle starts here: read what the broker actually holds, diff it against
what auto_trader believed going into the cycle (`last_known`, captured at the
end of the previous cycle), and log the difference. This module never
*corrects* anything by placing or cancelling a broker order -- whatever the
broker reports is simply adopted. The only side effects are: an audit-trail
row per discrepancy, an EXIT decision + closed_since_last entry for anything
that vanished from the broker since we last looked (this is what feeds the
consecutive-loss halt), and a plain-data snapshot of current broker state for
the rest of the cycle to act on.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Optional

from auto_trader import audit
from auto_trader.governor import PortfolioSnapshot, PositionView
from broker_engine.base import AccountInfo, OrderResult, Position

RESOLUTION = "adopted broker state"


def client_order_id_for(cycle_id: str, symbol: str, side: str) -> str:
    """
    Deterministic idempotency key for a (cycle, symbol, side) triple.

    Hashing is case-insensitive on symbol/side, so "tqqq"/"TQQQ" and
    "buy"/"BUY" collide to the same id -- a retried cycle can check the
    broker's open orders for this id before ever placing a duplicate.
    """
    key = f"{cycle_id}|{symbol.upper()}|{side.lower()}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return f"ra-{digest[:24]}"


def snapshot_broker(broker) -> tuple[AccountInfo, list[Position], list[OrderResult]]:
    """
    Read account / positions / open orders straight from the broker.

    Each leg is wrapped in its own try/except: a failure reading open orders
    must never lose an otherwise-good positions read (and vice versa). A leg
    that raises comes back as the exception instance itself rather than being
    silently swallowed, so reconcile() can log exactly what failed and decide
    what "empty" means for the rest of the cycle.
    """
    try:
        account: Any = broker.get_account()
    except Exception as exc:  # broker adapters can raise anything
        account = exc

    try:
        positions: Any = broker.get_positions() or []
    except Exception as exc:
        positions = exc

    try:
        open_orders: Any = broker.get_open_orders() or []
    except Exception as exc:
        open_orders = exc

    return account, positions, open_orders


def build_portfolio_snapshot(
    account: AccountInfo,
    positions: list[Position],
    day_start_equity: float,
    consecutive_losses: int,
) -> PortfolioSnapshot:
    """Map broker-shaped account/positions onto the governor's plain-data view."""
    return PortfolioSnapshot(
        equity=account.portfolio_val,
        cash=account.cash,
        positions=[
            PositionView(
                symbol=p.symbol,
                qty=p.qty,
                market_value=p.market_value,
                unrealized_pnl=p.unrealized_pnl,
                side=p.side,
            )
            for p in positions
        ],
        day_start_equity=day_start_equity,
        consecutive_losses=consecutive_losses,
    )


def order_already_exists(open_orders: list[OrderResult], client_order_id: str) -> bool:
    """True if some open order already carries this client_order_id."""
    return any(o.client_order_id == client_order_id for o in open_orders)


@dataclass
class ReconcileResult:
    account: Optional[AccountInfo]
    positions: list[Position]
    open_orders: list[OrderResult]
    discrepancies: list[dict] = field(default_factory=list)
    closed_since_last: list[dict] = field(default_factory=list)


def reconcile(broker, cycle_id, last_known=None, audit_mod=audit) -> ReconcileResult:
    """
    Broker-is-truth reconciliation for one cycle.

    last_known: {symbol: {"qty":.., "market_value":.., "unrealized_pnl":..}}
    captured at the end of the previous cycle (None/{} on the first cycle).

    Never places or cancels a broker order. Every discrepancy is resolved by
    adopting whatever the broker reports -- the only "correction" that happens
    lands in auto_trader's own bookkeeping (closed_since_last / audit rows),
    never back on the broker itself.
    """
    last_known = last_known or {}
    discrepancies: list[dict] = []
    closed_since_last: list[dict] = []

    raw_account, raw_positions, raw_open_orders = snapshot_broker(broker)

    if isinstance(raw_account, Exception):
        discrepancies.append({
            "kind": "broker_read_error",
            "call": "get_account",
            "error": str(raw_account),
        })
        audit_mod.log_discrepancy(
            cycle_id, "broker_read_error", "ACCOUNT",
            expected="account read", actual=str(raw_account),
            resolution="continued with no account data",
        )
        account: Optional[AccountInfo] = None
    else:
        account = raw_account

    if isinstance(raw_positions, Exception):
        discrepancies.append({
            "kind": "broker_read_error",
            "call": "get_positions",
            "error": str(raw_positions),
        })
        audit_mod.log_discrepancy(
            cycle_id, "broker_read_error", "PORTFOLIO",
            expected="positions read", actual=str(raw_positions),
            resolution="continued with empty list",
        )
        positions: list[Position] = []
    else:
        positions = raw_positions

    if isinstance(raw_open_orders, Exception):
        discrepancies.append({
            "kind": "broker_read_error",
            "call": "get_open_orders",
            "error": str(raw_open_orders),
        })
        audit_mod.log_discrepancy(
            cycle_id, "broker_read_error", "PORTFOLIO",
            expected="open orders read", actual=str(raw_open_orders),
            resolution="continued with empty list",
        )
        open_orders: list[OrderResult] = []
    else:
        open_orders = raw_open_orders

    broker_by_symbol = {p.symbol.upper(): p for p in positions}
    last_known_by_symbol = {s.upper(): v for s, v in last_known.items()}

    # last_known vs broker: closed-externally / qty-mismatch
    for symbol, prior in last_known_by_symbol.items():
        broker_pos = broker_by_symbol.get(symbol)

        if broker_pos is None:
            discrepancies.append({
                "kind": "position_closed_externally",
                "symbol": symbol,
                "last_known": prior,
            })
            audit_mod.log_discrepancy(
                cycle_id, "position_closed_externally", symbol,
                expected=prior, actual=None, resolution=RESOLUTION,
            )

            realized_pnl = prior.get("unrealized_pnl")
            qty = prior.get("qty")
            closed_since_last.append({
                "symbol": symbol,
                "qty": qty,
                "realized_pnl": realized_pnl,
                "estimated": True,
            })
            audit_mod.log_decision(
                cycle_id, symbol, "EXIT",
                "position closed externally or by bracket exit",
                realized_pnl=realized_pnl, qty=qty,
            )
            continue

        if broker_pos.qty != prior.get("qty"):
            discrepancies.append({
                "kind": "qty_mismatch",
                "symbol": symbol,
                "last_known_qty": prior.get("qty"),
                "broker_qty": broker_pos.qty,
            })
            audit_mod.log_discrepancy(
                cycle_id, "qty_mismatch", symbol,
                expected=prior.get("qty"), actual=broker_pos.qty,
                resolution=RESOLUTION,
            )

    # broker positions unseen in last_known
    for symbol, broker_pos in broker_by_symbol.items():
        if symbol not in last_known_by_symbol:
            discrepancies.append({
                "kind": "position_unknown_locally",
                "symbol": symbol,
                "qty": broker_pos.qty,
                "market_value": broker_pos.market_value,
                "unrealized_pnl": broker_pos.unrealized_pnl,
            })
            audit_mod.log_discrepancy(
                cycle_id, "position_unknown_locally", symbol,
                expected=None,
                actual={
                    "qty": broker_pos.qty,
                    "market_value": broker_pos.market_value,
                    "unrealized_pnl": broker_pos.unrealized_pnl,
                },
                resolution=RESOLUTION,
            )

    return ReconcileResult(
        account=account,
        positions=positions,
        open_orders=open_orders,
        discrepancies=discrepancies,
        closed_since_last=closed_since_last,
    )
