"""
auto_trader.audit — writes and queries the decision trail.

Every write and query here is best-effort. A logging/audit failure must
never interrupt a live trading cycle, so every public function wraps its
database work in try/except and logs on failure instead of raising.

Tables (see database/models.py):
    AutoTraderCycle       — one run of the auto trader loop
    AutoTraderDecision    — per-symbol decision made during a cycle
    AutoTraderDiscrepancy — reconciliation mismatch found during a cycle
"""

import json
import logging
from datetime import datetime

from sqlalchemy import select

from database.connection import get_session
from database.models import AutoTraderCycle, AutoTraderDecision, AutoTraderDiscrepancy

logger = logging.getLogger(__name__)

# Columns on AutoTraderDecision that log_decision() may set via **fields.
# (cycle_id, symbol, action, reason are already explicit positional params.)
_DECISION_EXTRA_FIELDS = {
    "ts", "verdict", "direction", "score", "confidence", "entry", "stop",
    "target", "qty", "risk_amount", "realized_pnl", "governor_rule",
    "broker_order_id", "client_order_id", "payload",
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _iso(value):
    """datetime -> isoformat string, tolerating None / bad values."""
    if value is None:
        return None
    try:
        return value.isoformat()
    except Exception:
        return str(value)


def _load_json(value):
    """JSON text -> python object, guarded against bad/empty payloads."""
    if not value:
        return None
    try:
        return json.loads(value)
    except Exception:
        return None


def _stringify(value, maxlen=200):
    if value is None:
        return None
    if not isinstance(value, str):
        try:
            value = json.dumps(value, default=str)
        except Exception:
            value = str(value)
    if maxlen and len(value) > maxlen:
        value = value[:maxlen]
    return value


def _cycle_to_dict(row: AutoTraderCycle) -> dict:
    return {
        "id": row.id,
        "cycle_id": row.cycle_id,
        "started_at": _iso(row.started_at),
        "finished_at": _iso(row.finished_at),
        "mode": row.mode,
        "broker": row.broker,
        "status": row.status,
        "symbols_scanned": row.symbols_scanned,
        "candidates": row.candidates,
        "orders_placed": row.orders_placed,
        "orders_rejected": row.orders_rejected,
        "equity_start": row.equity_start,
        "equity_end": row.equity_end,
        "day_start_equity": row.day_start_equity,
        "regime": row.regime,
        "positions": _load_json(row.positions_json),
        "halt_reason": row.halt_reason,
        "error": row.error,
        "created_at": _iso(row.created_at),
    }


def _decision_to_dict(row: AutoTraderDecision) -> dict:
    return {
        "id": row.id,
        "cycle_id": row.cycle_id,
        "ts": _iso(row.ts),
        "symbol": row.symbol,
        "action": row.action,
        "verdict": row.verdict,
        "direction": row.direction,
        "score": row.score,
        "confidence": row.confidence,
        "entry": row.entry,
        "stop": row.stop,
        "target": row.target,
        "qty": row.qty,
        "risk_amount": row.risk_amount,
        "realized_pnl": row.realized_pnl,
        "reason": row.reason,
        "governor_rule": row.governor_rule,
        "broker_order_id": row.broker_order_id,
        "client_order_id": row.client_order_id,
        "payload": _load_json(row.payload),
    }


def _discrepancy_to_dict(row: AutoTraderDiscrepancy) -> dict:
    return {
        "id": row.id,
        "cycle_id": row.cycle_id,
        "ts": _iso(row.ts),
        "kind": row.kind,
        "symbol": row.symbol,
        "expected": row.expected,
        "actual": row.actual,
        "resolution": row.resolution,
        "detail": row.detail,
    }


# ---------------------------------------------------------------------------
# writes
# ---------------------------------------------------------------------------

def new_cycle_id(mode: str) -> str:
    """f"{utcnow:%Y%m%dT%H%M%SZ}-{mode}" — never raises."""
    try:
        return f"{datetime.utcnow():%Y%m%dT%H%M%SZ}-{mode}"
    except Exception:
        logger.error("audit.new_cycle_id failed for mode=%r", mode, exc_info=True)
        return f"unknown-{mode}"


def start_cycle(cycle_id, mode, broker, equity_start, day_start_equity) -> None:
    """Insert the opening row for a new auto trader cycle."""
    session = None
    try:
        session = get_session()
        session.add(AutoTraderCycle(
            cycle_id=cycle_id,
            mode=mode,
            broker=broker,
            status="running",
            equity_start=equity_start,
            day_start_equity=day_start_equity,
        ))
        session.commit()
    except Exception:
        logger.error("audit.start_cycle failed cycle_id=%r", cycle_id, exc_info=True)
        if session is not None:
            try:
                session.rollback()
            except Exception:
                pass
    finally:
        if session is not None:
            try:
                session.close()
            except Exception:
                pass


def finish_cycle(cycle_id, status, *, equity_end=None, symbols_scanned=0, candidates=0,
                  orders_placed=0, orders_rejected=0, regime=None, positions=None,
                  halt_reason=None, error=None) -> None:
    """Update the cycle row with final counters, equity, and outcome."""
    session = None
    try:
        session = get_session()
        row = session.execute(
            select(AutoTraderCycle).where(AutoTraderCycle.cycle_id == cycle_id)
        ).scalar_one_or_none()
        if row is None:
            logger.error("audit.finish_cycle: no cycle row for cycle_id=%r", cycle_id)
            return
        row.finished_at = datetime.utcnow()
        row.status = status
        row.equity_end = equity_end
        row.symbols_scanned = symbols_scanned
        row.candidates = candidates
        row.orders_placed = orders_placed
        row.orders_rejected = orders_rejected
        row.regime = regime
        if positions is not None:
            try:
                row.positions_json = json.dumps(positions, default=str)
            except Exception:
                row.positions_json = None
        row.halt_reason = halt_reason
        row.error = error
        session.commit()
    except Exception:
        logger.error("audit.finish_cycle failed cycle_id=%r", cycle_id, exc_info=True)
        if session is not None:
            try:
                session.rollback()
            except Exception:
                pass
    finally:
        if session is not None:
            try:
                session.close()
            except Exception:
                pass


def log_decision(cycle_id, symbol, action, reason, **fields) -> None:
    """Record one per-symbol decision. Unknown **fields keys are ignored."""
    session = None
    try:
        kwargs = {k: v for k, v in fields.items() if k in _DECISION_EXTRA_FIELDS}
        if "payload" in kwargs and kwargs["payload"] is not None:
            payload = kwargs["payload"]
            if not isinstance(payload, str):
                try:
                    payload = json.dumps(payload, default=str)
                except Exception:
                    payload = str(payload)
            kwargs["payload"] = payload

        session = get_session()
        session.add(AutoTraderDecision(
            cycle_id=cycle_id,
            symbol=symbol,
            action=action,
            reason=reason,
            **kwargs,
        ))
        session.commit()
    except Exception:
        logger.error(
            "audit.log_decision failed cycle_id=%r symbol=%r action=%r",
            cycle_id, symbol, action, exc_info=True,
        )
        if session is not None:
            try:
                session.rollback()
            except Exception:
                pass
    finally:
        if session is not None:
            try:
                session.close()
            except Exception:
                pass


def log_discrepancy(cycle_id, kind, symbol, expected, actual, resolution, detail="") -> None:
    """Record a reconciliation mismatch found during a cycle."""
    session = None
    try:
        session = get_session()
        session.add(AutoTraderDiscrepancy(
            cycle_id=cycle_id,
            kind=kind,
            symbol=symbol,
            expected=_stringify(expected),
            actual=_stringify(actual),
            resolution=resolution,
            detail=detail,
        ))
        session.commit()
    except Exception:
        logger.error(
            "audit.log_discrepancy failed cycle_id=%r kind=%r symbol=%r",
            cycle_id, kind, symbol, exc_info=True,
        )
        if session is not None:
            try:
                session.rollback()
            except Exception:
                pass
    finally:
        if session is not None:
            try:
                session.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# queries
# ---------------------------------------------------------------------------

def has_placed(client_order_id: str) -> bool:
    """True if a decision with this client_order_id was already logged."""
    session = None
    try:
        session = get_session()
        row = session.execute(
            select(AutoTraderDecision.id)
            .where(AutoTraderDecision.client_order_id == client_order_id)
            .limit(1)
        ).first()
        return row is not None
    except Exception:
        logger.error("audit.has_placed failed client_order_id=%r", client_order_id, exc_info=True)
        return False
    finally:
        if session is not None:
            try:
                session.close()
            except Exception:
                pass


def day_start_equity(default: float) -> float:
    """equity_start of the earliest cycle started today (UTC), else default."""
    session = None
    try:
        session = get_session()
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        row = session.execute(
            select(AutoTraderCycle.equity_start)
            .where(AutoTraderCycle.started_at >= today_start)
            .order_by(AutoTraderCycle.started_at.asc())
            .limit(1)
        ).first()
        if row is not None and row[0] is not None:
            return row[0]
        return default
    except Exception:
        logger.error("audit.day_start_equity failed", exc_info=True)
        return default
    finally:
        if session is not None:
            try:
                session.close()
            except Exception:
                pass


def consecutive_losses(lookback: int = 10) -> int:
    """Count leading losing EXITs (most recent first) up to `lookback` rows."""
    session = None
    try:
        session = get_session()
        rows = session.execute(
            select(AutoTraderDecision.realized_pnl)
            .where(AutoTraderDecision.action == "EXIT")
            .where(AutoTraderDecision.realized_pnl.isnot(None))
            .order_by(AutoTraderDecision.ts.desc())
            .limit(lookback)
        ).scalars().all()
        count = 0
        for pnl in rows:
            if pnl is not None and pnl < 0:
                count += 1
            else:
                break
        return count
    except Exception:
        logger.error("audit.consecutive_losses failed", exc_info=True)
        return 0
    finally:
        if session is not None:
            try:
                session.close()
            except Exception:
                pass


def recent_cycles(limit: int = 50) -> list:
    """Most recent cycles, newest first, as JSON-serializable dicts."""
    session = None
    try:
        session = get_session()
        rows = session.execute(
            select(AutoTraderCycle)
            .order_by(AutoTraderCycle.started_at.desc())
            .limit(limit)
        ).scalars().all()
        return [_cycle_to_dict(r) for r in rows]
    except Exception:
        logger.error("audit.recent_cycles failed", exc_info=True)
        return []
    finally:
        if session is not None:
            try:
                session.close()
            except Exception:
                pass


def cycle_decisions(cycle_id: str) -> list:
    """All decisions for one cycle, oldest first, as JSON-serializable dicts."""
    session = None
    try:
        session = get_session()
        rows = session.execute(
            select(AutoTraderDecision)
            .where(AutoTraderDecision.cycle_id == cycle_id)
            .order_by(AutoTraderDecision.ts.asc())
        ).scalars().all()
        return [_decision_to_dict(r) for r in rows]
    except Exception:
        logger.error("audit.cycle_decisions failed cycle_id=%r", cycle_id, exc_info=True)
        return []
    finally:
        if session is not None:
            try:
                session.close()
            except Exception:
                pass


def recent_decisions(limit=200, action=None, symbol=None) -> list:
    """Most recent decisions, newest first, optionally filtered."""
    session = None
    try:
        session = get_session()
        stmt = select(AutoTraderDecision)
        if action is not None:
            stmt = stmt.where(AutoTraderDecision.action == action)
        if symbol is not None:
            stmt = stmt.where(AutoTraderDecision.symbol == symbol)
        stmt = stmt.order_by(AutoTraderDecision.ts.desc()).limit(limit)
        rows = session.execute(stmt).scalars().all()
        return [_decision_to_dict(r) for r in rows]
    except Exception:
        logger.error(
            "audit.recent_decisions failed action=%r symbol=%r", action, symbol, exc_info=True,
        )
        return []
    finally:
        if session is not None:
            try:
                session.close()
            except Exception:
                pass


def recent_discrepancies(limit=100) -> list:
    """Most recent reconciliation discrepancies, newest first."""
    session = None
    try:
        session = get_session()
        rows = session.execute(
            select(AutoTraderDiscrepancy)
            .order_by(AutoTraderDiscrepancy.ts.desc())
            .limit(limit)
        ).scalars().all()
        return [_discrepancy_to_dict(r) for r in rows]
    except Exception:
        logger.error("audit.recent_discrepancies failed", exc_info=True)
        return []
    finally:
        if session is not None:
            try:
                session.close()
            except Exception:
                pass
