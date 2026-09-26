"""
tests/test_auto_trader/test_reconciler.py
────────────────────────────────────────────────────────────────────────────────
Unit tests for auto_trader.reconciler — "the broker is always the source of
truth". Every test here is DB-free: reconcile() is always called with an
injected RecordingAudit() so nothing ever touches the real database.
"""
from __future__ import annotations

from broker_engine.base import AccountInfo, OrderResult, Position

from auto_trader.reconciler import (
    RESOLUTION,
    client_order_id_for,
    order_already_exists,
    reconcile,
)

from .conftest import FakeBroker, RecordingAudit


def _account(**overrides) -> AccountInfo:
    defaults = dict(
        account_id="acct-1", cash=50_000.0, portfolio_val=100_000.0,
        buying_power=50_000.0, broker="fake", is_paper=True,
    )
    defaults.update(overrides)
    return AccountInfo(**defaults)


def _position(symbol="TQQQ", qty=10, avg_cost=100.0, market_value=1_050.0,
              unrealized_pnl=50.0, unrealized_pct=5.0, side="long") -> Position:
    return Position(
        symbol=symbol, qty=qty, avg_cost=avg_cost, market_value=market_value,
        unrealized_pnl=unrealized_pnl, unrealized_pct=unrealized_pct, side=side,
    )


class TestClientOrderIdFor:
    def test_deterministic(self):
        a = client_order_id_for("cycle-1", "TQQQ", "buy")
        b = client_order_id_for("cycle-1", "TQQQ", "buy")
        assert a == b

    def test_case_insensitive_symbol_and_side(self):
        lower = client_order_id_for("cycle-1", "tqqq", "buy")
        upper = client_order_id_for("cycle-1", "TQQQ", "BUY")
        assert lower == upper

    def test_cycle_sensitive(self):
        a = client_order_id_for("cycle-1", "TQQQ", "buy")
        b = client_order_id_for("cycle-2", "TQQQ", "buy")
        assert a != b

    def test_ra_prefix(self):
        cid = client_order_id_for("cycle-1", "TQQQ", "buy")
        assert cid.startswith("ra-")


class TestPositionClosedExternally:
    def test_emits_discrepancy_and_exit_decision_with_realized_pnl(self):
        broker = FakeBroker()
        broker._account = _account()
        # broker now reports NO positions -- TQQQ vanished since last cycle
        broker._positions = {}
        last_known = {"TQQQ": {"qty": 10, "unrealized_pnl": 75.0}}
        audit_mod = RecordingAudit()

        result = reconcile(broker, "cycle-1", last_known=last_known, audit_mod=audit_mod)

        assert len(audit_mod.discrepancies) == 1
        disc = audit_mod.discrepancies[0]
        assert disc["kind"] == "position_closed_externally"
        assert disc["symbol"] == "TQQQ"
        assert disc["resolution"] == RESOLUTION

        exit_decisions = [d for d in audit_mod.decisions if d["action"] == "EXIT"]
        assert len(exit_decisions) == 1
        assert exit_decisions[0]["symbol"] == "TQQQ"
        assert exit_decisions[0]["realized_pnl"] == 75.0

        assert len(result.closed_since_last) == 1
        assert result.closed_since_last[0]["realized_pnl"] == 75.0


class TestPositionUnknownLocally:
    def test_position_unknown_locally(self):
        broker = FakeBroker()
        broker._account = _account()
        broker._positions = {"TQQQ": _position(symbol="TQQQ")}
        audit_mod = RecordingAudit()

        result = reconcile(broker, "cycle-1", last_known={}, audit_mod=audit_mod)

        kinds = [d["kind"] for d in audit_mod.discrepancies]
        assert "position_unknown_locally" in kinds
        unknown = next(d for d in audit_mod.discrepancies if d["kind"] == "position_unknown_locally")
        assert unknown["symbol"] == "TQQQ"
        assert unknown["resolution"] == RESOLUTION
        assert len(result.positions) == 1


class TestQtyMismatch:
    def test_qty_mismatch_resolves_to_broker_qty(self):
        broker = FakeBroker()
        broker._account = _account()
        broker._positions = {"TQQQ": _position(symbol="TQQQ", qty=15)}
        last_known = {"TQQQ": {"qty": 10}}
        audit_mod = RecordingAudit()

        reconcile(broker, "cycle-1", last_known=last_known, audit_mod=audit_mod)

        mismatches = [d for d in audit_mod.discrepancies if d["kind"] == "qty_mismatch"]
        assert len(mismatches) == 1
        mismatch = mismatches[0]
        assert mismatch["symbol"] == "TQQQ"
        assert mismatch["expected"] == 10
        assert mismatch["actual"] == 15
        assert mismatch["resolution"] == RESOLUTION


class TestOrderAlreadyExists:
    def test_true_when_client_order_id_present(self):
        open_orders = [
            OrderResult(
                order_id="o1", symbol="TQQQ", qty=10, side="buy",
                order_type="bracket", status="pending", client_order_id="ra-abc123",
            ),
        ]
        assert order_already_exists(open_orders, "ra-abc123") is True

    def test_false_when_client_order_id_absent(self):
        open_orders = [
            OrderResult(
                order_id="o1", symbol="TQQQ", qty=10, side="buy",
                order_type="bracket", status="pending", client_order_id="ra-abc123",
            ),
        ]
        assert order_already_exists(open_orders, "ra-does-not-exist") is False


class TestBrokerReadFailureDegrades:
    def test_get_open_orders_failure_degrades_instead_of_raising(self):
        broker = FakeBroker()
        broker._account = _account()
        broker._positions = {"TQQQ": _position(symbol="TQQQ")}
        broker.fail_on = {"get_open_orders"}
        audit_mod = RecordingAudit()

        # Must not raise.
        result = reconcile(broker, "cycle-1", last_known={}, audit_mod=audit_mod)

        assert result.open_orders == []
        # positions read still succeeded despite the open-orders failure
        assert len(result.positions) == 1

        read_error = [d for d in result.discrepancies if d["kind"] == "broker_read_error"]
        assert len(read_error) == 1
        assert read_error[0]["call"] == "get_open_orders"

        audit_read_errors = [d for d in audit_mod.discrepancies if d["kind"] == "broker_read_error"]
        assert len(audit_read_errors) == 1
        assert audit_read_errors[0]["resolution"] == "continued with empty list"
