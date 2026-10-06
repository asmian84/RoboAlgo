"""
scripts/auto_trader_smoke.py
────────────────────────────────────────────────────────────────────────────────
End-to-end smoke test for the auto_trader loop — NO Alpaca credentials,
NO Redis, and NO live market required.

Proves, with a single `python scripts/auto_trader_smoke.py` invocation, that
the loop actually:
  1. scans (via a canned, no-network scanner double),
  2. risk-gates each candidate through the real governor rules,
  3. dry-run "places" the one candidate that clears every rule,
  4. rejects the one that would exceed max_position_size,
  5. never lets a NO_TRADE verdict reach the governor at all, and
  6. refuses to re-place (SKIPs) a duplicate client_order_id on a second
     run with the same cycle_id — i.e. it is idempotent.

Nothing here touches a real broker, a real database, or Redis:
  * `auto_trader.audit.get_session` is monkeypatched to point at a throwaway
    SQLite file containing only the three auto_trader tables (see the NOTE
    in tests/test_auto_trader/conftest.py — a bare `Base.metadata.create_all()`
    crashes on SQLite because database/models.py declares
    `ix_paper_positions_symbol` twice; pre-existing, out of scope here).
  * `auto_trader.loop.get_broker` is monkeypatched to return the in-memory
    `FakeBroker` test double from tests/test_auto_trader/conftest.py.
  * `trade_signal_engine.scanner.scan_universe` is monkeypatched to return a
    canned, hand-built recommendations list.
  * `auto_trader.config.load` is monkeypatched to force
    enabled=True, dry_run=True, ignore_market_hours=True.
  * `ROBOALGO_AUTOTRADER_ALLOW_NO_REDIS=1` is set before any auto_trader
    import so the kill switch / run flag fall back to a local file store
    instead of requiring a live Redis server.

This script only *observes* auto_trader / api / tests / database — it does
not modify any of them. Exits 0 on success, 1 on any assertion failure.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# auto_trader.state logs a WARNING on every single Redis connection attempt
# (expected here — there is no Redis server) which would otherwise drown out
# the decision table below. Silence just that logger; nothing else.
logging.getLogger("auto_trader.state").setLevel(logging.ERROR)

# Allow `python scripts/auto_trader_smoke.py` to be run directly from the
# repo root without requiring PYTHONPATH — sys.path[0] is otherwise just
# the scripts/ directory, which breaks every project-root import below.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Must happen BEFORE importing any auto_trader module — auto_trader.state
# reads this env var lazily, but setting it up front keeps the ordering
# obviously correct regardless of import order below.
os.environ["ROBOALGO_AUTOTRADER_ALLOW_NO_REDIS"] = "1"

import dataclasses
import tempfile
import traceback

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from auto_trader import audit, config, loop, state
from broker_engine.base import AccountInfo
from database.models import (
    AutoTraderCycle,
    AutoTraderDecision,
    AutoTraderDiscrepancy,
    Base,
)
from tests.test_auto_trader.conftest import FakeBroker
import trade_signal_engine.scanner as scanner_mod

CLEAN_SYMBOL = "AAPL"       # should sail through every governor rule
OVERSIZED_SYMBOL = "MSFT"   # notional > 20% of equity -> max_position_size
NO_TRADE_SYMBOL = "TSLA"    # verdict NO_TRADE -> must never reach governor

CYCLE_ID = "smoke-cycle-0001"

failures: list[str] = []


def check(condition: bool, description: str) -> None:
    if condition:
        print(f"  [ok]   {description}")
    else:
        print(f"  [FAIL] {description}")
        failures.append(description)


def canned_scan_universe(symbols=None, profile="SWING", account_size=10_000.0,
                          max_workers=4, **_kwargs) -> dict:
    """No-network stand-in for trade_signal_engine.scanner.scan_universe."""
    return {
        "symbols_scanned": 3,
        "regime": {"regime": "NEUTRAL", "confidence": "HIGH"},
        "recommendations": [
            {
                # Clean BUY: small size, tiny risk, no existing position.
                "symbol": CLEAN_SYMBOL, "verdict": "BUY", "direction": "long",
                "score": 90.0, "confidence": 0.85,
                "entry": 100.0, "stop": 95.0, "target": 115.0,
                "position_size": 10, "position_value": 1_000.0, "risk_amount": 50.0,
            },
            {
                # Oversized BUY: notional 3,000 > 20% of 10,000 equity (2,000).
                "symbol": OVERSIZED_SYMBOL, "verdict": "BUY", "direction": "long",
                "score": 80.0, "confidence": 0.75,
                "entry": 100.0, "stop": 95.0, "target": 115.0,
                "position_size": 30, "position_value": 3_000.0, "risk_amount": 150.0,
            },
            {
                # NO_TRADE: filtered out before the governor ever sees it.
                "symbol": NO_TRADE_SYMBOL, "verdict": "NO_TRADE", "direction": "long",
                "score": 10.0, "confidence": 0.10,
                "entry": 250.0, "stop": 240.0, "target": 260.0,
                "position_size": 0, "position_value": 0.0, "risk_amount": 0.0,
            },
        ],
    }


def print_cycle_row(cycle_id: str) -> None:
    row = next((c for c in audit.recent_cycles(20) if c["cycle_id"] == cycle_id), None)
    print("\n--- Cycle row ---")
    if row is None:
        print(f"  (no cycle row found for cycle_id={cycle_id!r})")
        return
    for key in (
        "cycle_id", "status", "mode", "broker", "symbols_scanned", "candidates",
        "orders_placed", "orders_rejected", "equity_start", "equity_end",
        "day_start_equity", "regime", "halt_reason", "error",
    ):
        print(f"  {key:<18}: {row.get(key)}")


def print_decisions_table(cycle_id: str, label: str) -> list[dict]:
    decisions = audit.cycle_decisions(cycle_id)
    print(f"\n--- Decisions ({label}) ---")
    header = f"{'symbol':<8}{'action':<9}{'governor_rule':<22}{'client_order_id':<28}reason"
    print(f"  {header}")
    print(f"  {'-' * len(header)}")
    for d in decisions:
        symbol = d["symbol"] or "-"
        action = d["action"] or "-"
        rule = d["governor_rule"] or "-"
        coid = d["client_order_id"] or "-"
        reason = d["reason"] or "-"
        print(f"  {symbol:<8}{action:<9}{rule:<22}{coid:<28}{reason}")
    if not decisions:
        print("  (no decision rows)")
    return decisions


def main() -> int:
    tmp_fd, tmp_path = tempfile.mkstemp(prefix="auto_trader_smoke_", suffix=".db")
    os.close(tmp_fd)

    try:
        # ── 1. Temp SQLite DB with ONLY the three auto_trader tables ────────
        engine = create_engine(f"sqlite:///{tmp_path}")
        Base.metadata.create_all(engine, tables=[
            AutoTraderCycle.__table__,
            AutoTraderDecision.__table__,
            AutoTraderDiscrepancy.__table__,
        ])
        SessionLocal = sessionmaker(bind=engine)
        audit.get_session = lambda: SessionLocal()

        # ── 2. Fake broker: ~$10k equity, no open positions ─────────────────
        fake_account = AccountInfo(
            account_id="smoke-account", cash=10_000.0, portfolio_val=10_000.0,
            buying_power=10_000.0, broker="fake", is_paper=True,
        )
        fake_broker = FakeBroker(is_paper=True, account=fake_account)
        loop.get_broker = lambda *_a, **_kw: fake_broker

        # ── 3. Canned, no-network scanner ────────────────────────────────────
        scanner_mod.scan_universe = canned_scan_universe

        # ── 4. Forced-safe config: enabled, dry_run, ignore market hours ────
        base_cfg = config.load()
        forced_cfg = dataclasses.replace(
            base_cfg,
            enabled=True,
            dry_run=True,
            ignore_market_hours=True,
            symbols=[CLEAN_SYMBOL, OVERSIZED_SYMBOL, NO_TRADE_SYMBOL],
        )
        config.load = lambda: forced_cfg

        # ── 5. Kill switch OFF, running ON ───────────────────────────────────
        state.set_kill_switch(False)
        state.set_running(True)

        # ── 6. Run cycle #1 ───────────────────────────────────────────────
        print("=== Cycle 1: first run ===")
        result1 = loop.run_cycle(cycle_id=CYCLE_ID)
        print(f"run_cycle() -> {result1}")

        print_cycle_row(CYCLE_ID)
        decisions1 = print_decisions_table(CYCLE_ID, "after cycle 1")

        clean_decisions = [d for d in decisions1 if d["symbol"] == CLEAN_SYMBOL]
        oversized_decisions = [d for d in decisions1 if d["symbol"] == OVERSIZED_SYMBOL]
        no_trade_decisions = [d for d in decisions1 if d["symbol"] == NO_TRADE_SYMBOL]

        print("\n=== Assertions: cycle 1 ===")
        check(
            len(clean_decisions) == 1
            and clean_decisions[0]["action"] == "DRY_RUN"
            and (clean_decisions[0]["client_order_id"] or "").startswith("ra-"),
            "clean BUY (AAPL) produced exactly one DRY_RUN decision with "
            "client_order_id starting 'ra-'",
        )
        check(
            len(oversized_decisions) == 1
            and oversized_decisions[0]["action"] == "REJECT"
            and oversized_decisions[0]["governor_rule"] == "max_position_size",
            "oversized BUY (MSFT) produced exactly one REJECT with "
            "governor_rule == 'max_position_size'",
        )
        check(
            len(no_trade_decisions) == 0,
            "NO_TRADE candidate (TSLA) never reached the governor "
            "(no decision row at all)",
        )

        clean_coid = clean_decisions[0]["client_order_id"] if clean_decisions else None

        # ── 7. Run cycle #2 with the SAME cycle_id -> idempotency check ─────
        print("\n=== Cycle 2: re-run with SAME cycle_id (idempotency) ===")
        result2 = loop.run_cycle(cycle_id=CYCLE_ID)
        print(f"run_cycle() -> {result2}")

        decisions2 = print_decisions_table(CYCLE_ID, "cumulative after cycle 2")

        skip_decisions = [
            d for d in decisions2
            if d["symbol"] == CLEAN_SYMBOL
            and d["action"] == "SKIP"
            and "duplicate client_order_id" in (d["reason"] or "")
        ]
        print("\n=== Assertions: cycle 2 (idempotency) ===")
        check(
            len(skip_decisions) == 1 and skip_decisions[0]["client_order_id"] == clean_coid,
            "re-running run_cycle with the SAME cycle_id produced a SKIP / "
            "'duplicate client_order_id' decision for the clean BUY (AAPL), "
            "reusing the same client_order_id",
        )

    except Exception:
        traceback.print_exc()
        failures.append("unhandled exception raised during smoke test (see traceback above)")
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    print()
    if failures:
        print(f"SMOKE FAILED: {'; '.join(failures)}")
        return 1

    print("SMOKE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
