"""
lab_engine/verdict_log.py
────────────────────────────────────────────────────────────────────────────────
Experiment verdict log — deliberately its own SQLite file, NOT the main DB.

Why separate: the Lab is a hiring trial. Keeping its data in one droppable
file means retiring an experiment is `rm lab_experiments.db`, with zero
migration against the production Postgres schema.

Every panel run logs one row per persona PLUS one row for the live scanner's
own verdict on the same symbol at the same instant. That side-by-side is the
whole point — you cannot fire a persona without knowing what the incumbent
scored on the identical setup.

Scoring is forward-looking: rows store `price_at_verdict`, and `scorecard()`
marks each row against the current price to compute hit rate per source.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

_DB_PATH = os.getenv(
    "LAB_DB_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lab_experiments.db"),
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiment_verdicts (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id            TEXT    NOT NULL,
    ts                TEXT    NOT NULL,
    symbol            TEXT    NOT NULL,
    source            TEXT    NOT NULL,
    verdict           TEXT    NOT NULL,
    score             REAL,
    confidence        REAL,
    provider          TEXT,
    price_at_verdict  REAL,
    meta              TEXT
);
CREATE INDEX IF NOT EXISTS idx_ev_symbol ON experiment_verdicts(symbol);
CREATE INDEX IF NOT EXISTS idx_ev_source ON experiment_verdicts(source);
CREATE INDEX IF NOT EXISTS idx_ev_run    ON experiment_verdicts(run_id);
"""


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(_DB_PATH, timeout=10)
    c.row_factory = sqlite3.Row
    c.executescript(_SCHEMA)
    return c


def db_path() -> str:
    return _DB_PATH


def log_run(
    run_id: str,
    symbol: str,
    rows: list[dict[str, Any]],
    provider: str,
    price: float | None,
) -> int:
    """
    Persist one panel run.

    `rows` items need: source, verdict, score, confidence, and optionally meta.
    Returns the number of rows written. Never raises — a logging failure must
    not break the API response the user is waiting on.
    """
    ts = datetime.now(timezone.utc).isoformat()
    try:
        with _conn() as c:
            c.executemany(
                """INSERT INTO experiment_verdicts
                   (run_id, ts, symbol, source, verdict, score, confidence,
                    provider, price_at_verdict, meta)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                [
                    (
                        run_id, ts, symbol.upper(), r["source"], r["verdict"],
                        r.get("score"), r.get("confidence"), provider, price,
                        json.dumps(r.get("meta") or {}),
                    )
                    for r in rows
                ],
            )
        return len(rows)
    except Exception as exc:
        logger.warning("verdict log write failed for %s: %s", symbol, exc)
        return 0


def recent_runs(limit: int = 50) -> list[dict[str, Any]]:
    """Most recent panel runs, newest first — one entry per run_id."""
    try:
        with _conn() as c:
            rows = c.execute(
                """SELECT run_id, symbol, provider, MIN(ts) AS ts,
                          COUNT(*) AS n_verdicts, MAX(price_at_verdict) AS price
                   FROM experiment_verdicts
                   GROUP BY run_id
                   ORDER BY ts DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]
    except Exception as exc:
        logger.warning("recent_runs failed: %s", exc)
        return []


def scorecard(min_age_hours: float = 0.0) -> dict[str, Any]:
    """
    Hit rate per source, marked against current prices.

    A verdict "hits" when the market moved its way: BULLISH and price up,
    BEARISH and price down. NEUTRAL rows are tracked but excluded from hit
    rate — a neutral call cannot be right or wrong directionally.

    `min_age_hours` filters out verdicts too fresh to have been tested yet.
    Returns per-source stats plus the sample size, so a 100% hit rate on
    n=2 is visibly not the same as 60% on n=200.
    """
    try:
        with _conn() as c:
            rows = [dict(r) for r in c.execute(
                "SELECT * FROM experiment_verdicts ORDER BY ts ASC"
            ).fetchall()]
    except Exception as exc:
        logger.warning("scorecard read failed: %s", exc)
        return {"sources": [], "total_verdicts": 0, "error": str(exc)}

    if not rows:
        return {"sources": [], "total_verdicts": 0}

    now = datetime.now(timezone.utc)
    symbols = sorted({r["symbol"] for r in rows})
    prices = _current_prices(symbols)

    by_source: dict[str, dict[str, Any]] = {}
    for r in rows:
        px_now = prices.get(r["symbol"])
        px_then = r["price_at_verdict"]
        try:
            age_h = (now - datetime.fromisoformat(r["ts"])).total_seconds() / 3600
        except Exception:
            age_h = 0.0
        if age_h < min_age_hours:
            continue

        s = by_source.setdefault(r["source"], {
            "source": r["source"], "n": 0, "scored": 0, "hits": 0,
            "bullish": 0, "bearish": 0, "neutral": 0, "sum_return": 0.0,
        })
        s["n"] += 1
        v = r["verdict"]
        if v == "BULLISH":
            s["bullish"] += 1
        elif v == "BEARISH":
            s["bearish"] += 1
        elif v == "NEUTRAL":
            s["neutral"] += 1

        if v in ("BULLISH", "BEARISH") and px_now and px_then:
            ret = (px_now - px_then) / px_then
            directional = ret if v == "BULLISH" else -ret
            s["scored"] += 1
            s["sum_return"] += directional
            if directional > 0:
                s["hits"] += 1

    out = []
    for s in by_source.values():
        scored = s["scored"]
        out.append({
            **s,
            "hit_rate":  round(s["hits"] / scored * 100, 1) if scored else None,
            "avg_return": round(s["sum_return"] / scored * 100, 2) if scored else None,
        })
    out.sort(key=lambda x: (x["hit_rate"] is None, -(x["hit_rate"] or 0)))

    return {
        "sources": out,
        "total_verdicts": len(rows),
        "symbols_tracked": len(symbols),
        "min_age_hours": min_age_hours,
        "note": "Hit rate excludes NEUTRAL verdicts. Small n is not evidence — "
                "check the `scored` column before retiring anything.",
    }


def _current_prices(symbols: list[str]) -> dict[str, float]:
    """Best-effort current prices for marking verdicts. Missing = skipped."""
    out: dict[str, float] = {}
    try:
        from data_engine.market_data_service import get_quotes_batch
        for sym, q in (get_quotes_batch(symbols) or {}).items():
            if q and q.get("price"):
                out[sym] = float(q["price"])
    except Exception as exc:
        logger.warning("price marking failed: %s", exc)
    return out


def purge(source: str | None = None) -> int:
    """Delete logged verdicts — all, or just one retired source. Returns rows removed."""
    try:
        with _conn() as c:
            if source:
                cur = c.execute("DELETE FROM experiment_verdicts WHERE source = ?", (source,))
            else:
                cur = c.execute("DELETE FROM experiment_verdicts")
            return cur.rowcount
    except Exception as exc:
        logger.warning("purge failed: %s", exc)
        return 0
