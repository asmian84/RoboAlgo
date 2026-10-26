"""
tests/test_auto_trader/test_governor.py
────────────────────────────────────────────────────────────────────────────────
Unit tests for auto_trader.governor — the zero-I/O risk policy engine.

One test proves each rule fires with the exact `verdict.rule` tag, plus tests
for precedence, a clean approval path, and a couple of edge cases (governor
never resizes an order, side-aware stop validation, day_start_equity=0).
"""
from __future__ import annotations

import dataclasses

from auto_trader.governor import (
    CandidateOrder,
    GovernorVerdict,
    PositionView,
    RiskLimits,
    check_halts,
    check_order,
    evaluate,
)

from .conftest import snapshot


def _candidate(**overrides) -> CandidateOrder:
    """A clean, fully-valid long candidate — override fields per test."""
    defaults = dict(
        symbol="TQQQ",
        side="buy",
        qty=10,
        entry=100.0,
        stop=95.0,
        target=110.0,
        notional=1_000.0,
        risk_amount=50.0,
        score=0.8,
        confidence=0.8,
    )
    defaults.update(overrides)
    return CandidateOrder(**defaults)


LIMITS = RiskLimits()  # defaults: max_open_positions=5, max_total_exposure_pct=60,
# per_trade_risk_pct=1.0, daily_loss_limit_pct=3.0, consecutive_loss_halt=3,
# max_position_size_pct=20.0


class TestHaltRules:
    def test_kill_switch(self):
        snap = snapshot()
        verdict = evaluate(_candidate(), snap, LIMITS, kill_switch_on=True)
        assert verdict.approved is False
        assert verdict.rule == "kill_switch"

    def test_daily_loss_limit(self):
        # day pnl = (96000 - 100000) / 100000 * 100 = -4% <= -3%
        snap = snapshot(equity=96_000.0, day_start_equity=100_000.0)
        verdict = evaluate(_candidate(), snap, LIMITS, kill_switch_on=False)
        assert verdict.approved is False
        assert verdict.rule == "daily_loss_limit"

    def test_consecutive_loss_halt(self):
        snap = snapshot(
            equity=100_000.0, day_start_equity=100_000.0, consecutive_losses=3,
        )
        verdict = evaluate(_candidate(), snap, LIMITS, kill_switch_on=False)
        assert verdict.approved is False
        assert verdict.rule == "consecutive_loss_halt"


class TestOrderRules:
    def test_invalid_order(self):
        snap = snapshot()
        # buy with stop >= entry is invalid
        candidate = _candidate(side="buy", entry=100.0, stop=105.0)
        verdict = check_order(candidate, snap, LIMITS)
        assert verdict.approved is False
        assert verdict.rule == "invalid_order"

    def test_duplicate_position(self):
        snap = snapshot(positions=[
            PositionView(symbol="TQQQ", qty=5, market_value=500.0, unrealized_pnl=0.0),
        ])
        candidate = _candidate(symbol="tqqq")  # lowercase — must still match uppercase
        verdict = check_order(candidate, snap, LIMITS)
        assert verdict.approved is False
        assert verdict.rule == "duplicate_position"

    def test_max_open_positions(self):
        existing = [
            PositionView(symbol=f"SYM{i}", qty=1, market_value=100.0, unrealized_pnl=0.0)
            for i in range(LIMITS.max_open_positions)
        ]
        snap = snapshot(positions=existing)
        candidate = _candidate(symbol="TQQQ")  # not in existing symbols
        verdict = check_order(candidate, snap, LIMITS)
        assert verdict.approved is False
        assert verdict.rule == "max_open_positions"

    def test_max_position_size(self):
        # equity 100_000 * 20% = 20_000 max position size
        snap = snapshot(equity=100_000.0, cash=100_000.0)
        candidate = _candidate(notional=25_000.0, risk_amount=50.0)
        verdict = check_order(candidate, snap, LIMITS)
        assert verdict.approved is False
        assert verdict.rule == "max_position_size"

    def test_per_trade_risk(self):
        # equity 100_000 * 1% = 1_000 max risk; notional stays under max_position_size
        snap = snapshot(equity=100_000.0, cash=100_000.0)
        candidate = _candidate(notional=5_000.0, risk_amount=1_500.0)
        verdict = check_order(candidate, snap, LIMITS)
        assert verdict.approved is False
        assert verdict.rule == "per_trade_risk"

    def test_max_total_exposure(self):
        # existing gross exposure 50_000 + notional 15_000 = 65_000 > 60% of 100_000
        snap = snapshot(
            equity=100_000.0,
            cash=100_000.0,
            positions=[
                PositionView(symbol="SPY", qty=100, market_value=50_000.0, unrealized_pnl=0.0),
            ],
        )
        candidate = _candidate(symbol="TQQQ", notional=15_000.0, risk_amount=500.0)
        verdict = check_order(candidate, snap, LIMITS)
        assert verdict.approved is False
        assert verdict.rule == "max_total_exposure"

    def test_insufficient_buying_power(self):
        snap = snapshot(equity=100_000.0, cash=1_000.0)
        candidate = _candidate(notional=1_500.0, risk_amount=500.0)
        verdict = check_order(candidate, snap, LIMITS)
        assert verdict.approved is False
        assert verdict.rule == "insufficient_buying_power"


class TestPrecedence:
    def test_rule_precedence(self):
        # A snapshot that would trip kill_switch, daily_loss_limit, AND
        # consecutive_loss_halt simultaneously -- first match wins.
        snap = snapshot(
            equity=96_000.0, day_start_equity=100_000.0, consecutive_losses=5,
        )

        kill_switch_verdict = check_halts(snap, LIMITS, kill_switch_on=True)
        assert kill_switch_verdict.rule == "kill_switch"

        daily_loss_verdict = check_halts(snap, LIMITS, kill_switch_on=False)
        assert daily_loss_verdict.rule == "daily_loss_limit"

        no_daily_loss_snap = snapshot(
            equity=100_000.0, day_start_equity=100_000.0, consecutive_losses=5,
        )
        consecutive_verdict = check_halts(no_daily_loss_snap, LIMITS, kill_switch_on=False)
        assert consecutive_verdict.rule == "consecutive_loss_halt"


class TestCleanApproval:
    def test_clean_candidate_approved(self):
        snap = snapshot()
        verdict = evaluate(_candidate(), snap, LIMITS, kill_switch_on=False)
        assert verdict.approved is True
        assert verdict.rule is None


class TestGovernorNeverResizes:
    def test_governor_never_adjusts_qty(self):
        # GovernorVerdict simply has no qty field to adjust...
        field_names = {f.name for f in dataclasses.fields(GovernorVerdict)}
        assert "qty" not in field_names

        # ...and the candidate object itself is never mutated by evaluate().
        candidate = _candidate(qty=10, notional=25_000.0, risk_amount=1_500.0)
        snap = snapshot(equity=100_000.0, cash=100_000.0)
        verdict = evaluate(candidate, snap, LIMITS, kill_switch_on=False)
        assert verdict.approved is False
        assert candidate.qty == 10


class TestSideAwareStopValidation:
    def test_short_side_stop_above_entry(self):
        # A short (sell) with its stop BELOW entry is nonsensical for a short
        # and must be rejected as invalid_order.
        snap = snapshot()
        candidate = _candidate(
            symbol="TQQQ", side="sell", entry=100.0, stop=95.0, target=90.0,
        )
        verdict = check_order(candidate, snap, LIMITS)
        assert verdict.approved is False
        assert verdict.rule == "invalid_order"


class TestDayStartEquityGuard:
    def test_day_start_equity_zero_guarded(self):
        snap = snapshot(equity=100_000.0, day_start_equity=0.0, consecutive_losses=0)
        # Must not raise (e.g. ZeroDivisionError) and must not spuriously halt.
        verdict = check_halts(snap, LIMITS, kill_switch_on=False)
        assert verdict.approved is True
        assert verdict.rule is None
