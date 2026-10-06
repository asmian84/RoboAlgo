"""
auto_trader/health.py
────────────────────────────────────────────────────────────────────────────────
Read-only health and decay reporting for the auto-trader loop.

Both public functions here read EXCLUSIVELY through the `auto_trader.audit`
query helpers and `auto_trader.state` — no new SQL, no direct DB access. Like
`audit` itself, both functions are best-effort: a health-check failure must
never raise, it must degrade to a safe/empty default instead.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from auto_trader import audit, config as config_mod, state

logger = logging.getLogger(__name__)

# Celery beat dispatches an auto-trader cycle every 15 minutes; a cycle is
# considered stale once its age exceeds 2x that interval.
_BEAT_INTERVAL_SECONDS = 15 * 60
_STALE_THRESHOLD_SECONDS = 2 * _BEAT_INTERVAL_SECONDS  # 1800s


def _parse_iso(value):
    """ISO-format string (as produced by audit._iso) -> aware datetime, or None."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _market_probably_open(now_utc: datetime) -> bool:
    """
    Deliberately simplified US-equities market-hours check: Mon-Fri,
    09:30-16:00 America/New_York, with NO holiday calendar and no early-close
    handling. This is a heuristic input to the `stale` flag below, not a
    trading gate — a false "open" on a market holiday only makes staleness
    detection slightly too eager (it can never place or block an order).
    """
    try:
        from zoneinfo import ZoneInfo
        eastern = now_utc.astimezone(ZoneInfo("America/New_York"))
    except Exception:
        # If zoneinfo/tzdata isn't available for some reason, fall back to
        # UTC rather than raising — worst case the heuristic is a few hours off.
        eastern = now_utc
    if eastern.weekday() >= 5:  # Saturday, Sunday
        return False
    minutes = eastern.hour * 60 + eastern.minute
    return (9 * 60 + 30) <= minutes <= (16 * 60)


def loop_health() -> dict:
    """
    Snapshot of the auto-trader loop's health.

    `stale` is True only when BOTH:
      - `age_seconds` (time since the last cycle started) exceeds 2x the beat
        interval (1800s), AND
      - the market would plausibly be open right now, per the simplified
        weekday + 09:30-16:00 America/New_York check in
        `_market_probably_open` (no holiday calendar — see its docstring).

    If there has never been a cycle, `age_seconds` is None and `stale` is
    True whenever the market is plausibly open (there should be cycles by
    now and there are none).

    Never raises: every sub-lookup is individually guarded and degrades to a
    safe default (empty list / None / fail-safe kill_switch=True) on error.
    """
    now = datetime.now(timezone.utc)

    try:
        cycles = audit.recent_cycles(limit=50)
    except Exception:
        logger.error("auto_trader.health.loop_health: recent_cycles failed", exc_info=True)
        cycles = []

    last_cycle = cycles[0] if cycles else None
    last_cycle_id = last_cycle.get("cycle_id") if last_cycle else None
    last_cycle_at = last_cycle.get("started_at") if last_cycle else None

    started = _parse_iso(last_cycle_at)
    age_seconds = (now - started).total_seconds() if started is not None else None

    market_open = _market_probably_open(now)
    if age_seconds is None:
        stale = bool(market_open)
    else:
        stale = age_seconds > _STALE_THRESHOLD_SECONDS and market_open

    consecutive_failed_cycles = 0
    for c in cycles:
        if c.get("status") == "failed":
            consecutive_failed_cycles += 1
        else:
            break

    try:
        running = state.running_enabled()
    except Exception:
        logger.error("auto_trader.health.loop_health: running_enabled failed", exc_info=True)
        running = False

    try:
        kill_switch = state.kill_switch_on()
    except Exception:
        logger.error("auto_trader.health.loop_health: kill_switch_on failed", exc_info=True)
        kill_switch = True  # fail-safe, matches state.py's own fail-safe default

    try:
        redis_available = state.redis_available()
    except Exception:
        logger.error("auto_trader.health.loop_health: redis_available failed", exc_info=True)
        redis_available = False

    try:
        heartbeat = state.get_heartbeat()
    except Exception:
        logger.error("auto_trader.health.loop_health: get_heartbeat failed", exc_info=True)
        heartbeat = None

    mode = None
    broker = None
    try:
        cfg = config_mod.load()
        mode = cfg.mode
        broker = cfg.broker
    except Exception:
        logger.error("auto_trader.health.loop_health: config.load failed", exc_info=True)

    is_paper = None
    broker_error = None
    try:
        from broker_engine.factory import get_broker
        b = get_broker(broker)
        is_paper = b.is_paper
    except Exception as exc:
        is_paper = None
        broker_error = str(exc)

    return {
        "last_cycle_id": last_cycle_id,
        "last_cycle_at": last_cycle_at,
        "age_seconds": age_seconds,
        "stale": stale,
        "consecutive_failed_cycles": consecutive_failed_cycles,
        "running": running,
        "kill_switch": kill_switch,
        "mode": mode,
        "broker": broker,
        "is_paper": is_paper,
        "broker_error": broker_error,
        "redis_available": redis_available,
        "heartbeat": heartbeat,
    }


def decay_report(cycles: int = 20) -> dict:
    """
    Approval-rate / rejection / score-decay report over the last `cycles`
    auto-trader cycles.

    Every statistic carries its own sample size (`n`) alongside it so a
    small-sample result (e.g. 100% approval on n=2) can never be presented
    as if it had the weight of a larger one. Never raises: an empty/no-data
    DB yields zeroed-out stats with n=0, not an exception.

    Returns:
        {
          "cycles_requested": int,
          "cycles_found": int,
          "approval_rate": {"rate": float|None, "n": int, "placed": int, "rejected": int},
          "rejection_histogram": {"n": int, "counts": {governor_rule: count}},
          "score_by_outcome": {
              "placed":   {"avg": float|None, "n": int},
              "rejected": {"avg": float|None, "n": int},
          },
          "symbols_scanned_trend": [{"cycle_id", "started_at", "symbols_scanned"}, ...],
        }
    """
    try:
        cycle_rows = audit.recent_cycles(limit=cycles)
    except Exception:
        logger.error("auto_trader.health.decay_report: recent_cycles failed", exc_info=True)
        cycle_rows = []

    cycle_ids = {c.get("cycle_id") for c in cycle_rows if c.get("cycle_id")}

    try:
        # Pull a generously large window of decisions and then filter down to
        # the cycles we actually selected, since recent_decisions() has no
        # cycle-set filter of its own.
        decisions = audit.recent_decisions(limit=max(cycles * 50, 200))
    except Exception:
        logger.error("auto_trader.health.decay_report: recent_decisions failed", exc_info=True)
        decisions = []

    if cycle_ids:
        decisions = [d for d in decisions if d.get("cycle_id") in cycle_ids]

    # auto_trader/loop.py logs these action values verbatim: "PLACE" when an
    # order is actually sent to the broker, "REJECT" when the governor blocks
    # a candidate. ("DRY_RUN" and "SKIP" are neither an approval nor a
    # governor rejection, so they're excluded from this ratio.)
    placed = [d for d in decisions if d.get("action") == "PLACE"]
    rejected = [d for d in decisions if d.get("action") == "REJECT"]

    total_considered = len(placed) + len(rejected)
    approval_rate = (len(placed) / total_considered) if total_considered else None

    rejection_histogram: dict = {}
    for d in rejected:
        rule = d.get("governor_rule") or "unknown"
        rejection_histogram[rule] = rejection_histogram.get(rule, 0) + 1

    def _avg_score(rows):
        scores = [r.get("score") for r in rows if r.get("score") is not None]
        if not scores:
            return {"avg": None, "n": 0}
        return {"avg": sum(scores) / len(scores), "n": len(scores)}

    symbols_scanned_trend = [
        {
            "cycle_id": c.get("cycle_id"),
            "started_at": c.get("started_at"),
            "symbols_scanned": c.get("symbols_scanned"),
        }
        for c in cycle_rows
    ]

    return {
        "cycles_requested": cycles,
        "cycles_found": len(cycle_rows),
        "approval_rate": {
            "rate": approval_rate,
            "n": total_considered,
            "placed": len(placed),
            "rejected": len(rejected),
        },
        "rejection_histogram": {
            "n": len(rejected),
            "counts": rejection_histogram,
        },
        "score_by_outcome": {
            "placed": _avg_score(placed),
            "rejected": _avg_score(rejected),
        },
        "symbols_scanned_trend": symbols_scanned_trend,
    }
