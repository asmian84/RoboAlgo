"""
RoboAlgo — One-time migration: create auto trader audit tables.

Run once after deploying the auto trader audit feature:
    python scripts/migrate_auto_trader.py

Creates auto_trader_cycles, auto_trader_decisions, and
auto_trader_discrepancies via Base.metadata.create_all() scoped to only
those three tables (checkfirst=True — safe to re-run).

NOTE: a bare Base.metadata.create_all() over the *whole* metadata will
crash on SQLite because of a pre-existing duplicate index name collision
on ix_paper_positions_symbol (database/models.py, PaperPosition). This
script intentionally passes an explicit tables=[...] list to avoid ever
touching that codepath.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# `python scripts/migrate_auto_trader.py` puts scripts/ on sys.path, not the
# repo root, so the project packages are not importable. Bootstrap the root
# rather than requiring callers to set PYTHONPATH or remember -m, since the
# runbook lists this script as the very first step.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from database.connection import get_engine
from database.models import (
    Base,
    AutoTraderCycle,
    AutoTraderDecision,
    AutoTraderDiscrepancy,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def run():
    engine = get_engine()
    tables = [
        AutoTraderCycle.__table__,
        AutoTraderDecision.__table__,
        AutoTraderDiscrepancy.__table__,
    ]
    Base.metadata.create_all(engine, tables=tables, checkfirst=True)
    for table in tables:
        log.info(f"  ✓  {table.name}")

    log.info("Migration complete.")


if __name__ == "__main__":
    run()
