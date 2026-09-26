"""
api/routers/trading_control.py
────────────────────────────────────────────────────────────────────────────────
Operator control surface for the autonomous trading loop: start/stop, kill
switch, config, and live audit trail (cycles / decisions / discrepancies).

Endpoints:
  GET  /api/auto-trader/status                     Current run/kill/broker state
  POST /api/auto-trader/start                       [GUARDED] Enable the run loop
  POST /api/auto-trader/stop                        Disable the run loop
  POST /api/auto-trader/kill                        Kill switch — see asymmetry below
  GET  /api/auto-trader/config                      Current auto-trader config
  POST /api/auto-trader/config                      [GUARDED] Save a partial config
  POST /api/auto-trader/cycle/run                   [GUARDED] Dispatch or run a cycle
  GET  /api/auto-trader/cycles                      Recent cycles
  GET  /api/auto-trader/cycles/{cycle_id}/decisions Decisions for one cycle
  GET  /api/auto-trader/decisions                   Recent decisions (filterable)
  GET  /api/auto-trader/discrepancies               Recent reconciliation discrepancies
  GET  /api/auto-trader/health                      Loop health/staleness snapshot
  GET  /api/auto-trader/decay                       Approval-rate / rejection decay report

Auth asymmetry — deliberate, read before changing:
  This API has no authentication layer anywhere (zero `Depends()` across every
  other router; auth is frontend-only, enforced by Supabase in the browser).
  These endpoints are different: they can place real orders, so risk-INCREASING
  actions (start, config writes, dispatching a live cycle, turning the kill
  switch OFF) are gated behind `X-RoboAlgo-Control-Token`. Risk-REDUCING actions
  (stop, turning the kill switch ON) are intentionally left unauthenticated so
  an operator can never be locked out of stopping the system — a missing or
  wrong token must never be able to block a stop/kill call.

  The gate itself fails closed: if ROBOALGO_CONTROL_TOKEN is not set in the
  environment, every guarded endpoint returns 503 rather than silently running
  unauthenticated or silently no-op'ing.
"""
from __future__ import annotations

import hmac
import logging
import os

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Guard ─────────────────────────────────────────────────────────────────────

def _require_token(
    x_control_token: str | None = Header(None, alias="X-RoboAlgo-Control-Token"),
) -> None:
    """
    Dependency for risk-increasing endpoints.

    503 if ROBOALGO_CONTROL_TOKEN isn't configured (gate fails closed, not open).
    403 if the caller's token is missing or doesn't match.
    """
    expected = os.getenv("ROBOALGO_CONTROL_TOKEN")
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="ROBOALGO_CONTROL_TOKEN is not configured; risk-increasing controls are disabled",
        )
    if not x_control_token or not hmac.compare_digest(x_control_token, expected):
        raise HTTPException(status_code=403, detail="Invalid or missing control token")


# ── Pydantic models ───────────────────────────────────────────────────────────

class KillSwitchPayload(BaseModel):
    on: bool
    reason: str = ""


# ── Status ────────────────────────────────────────────────────────────────────

@router.get("/status")
def get_status():
    """
    Snapshot of run state, kill switch, broker/mode, and the last cycle.

    Never raises on a broker configuration problem (e.g. missing Alpaca keys):
    `is_paper` degrades to null and `broker_error` carries the reason.
    """
    from auto_trader import audit, config as config_mod, state

    cfg = config_mod.load()

    is_paper = None
    broker_error = None
    try:
        from broker_engine.factory import get_broker
        broker = get_broker(cfg.broker)
        is_paper = broker.is_paper
    except Exception as exc:
        broker_error = str(exc)

    try:
        cycles = audit.recent_cycles(limit=1)
        last_cycle = cycles[0] if cycles else None
    except Exception as exc:
        logger.warning("trading_control.status: recent_cycles failed: %s", exc)
        last_cycle = None

    return {
        "enabled": cfg.enabled,
        "running": state.running_enabled(),
        "kill_switch": state.kill_switch_on(),
        # Why it is in this state and who set it. Informational only — the
        # authoritative value is `kill_switch` above, which fails safe.
        "kill_meta": state.get_kill_switch_meta(),
        "mode": cfg.mode,
        "broker": cfg.broker,
        "is_paper": is_paper,
        "broker_error": broker_error,
        "redis_available": state.redis_available(),
        "lock": state.current_lock(),
        "last_cycle": last_cycle,
    }


# ── Run control ───────────────────────────────────────────────────────────────

@router.post("/start", dependencies=[Depends(_require_token)])
def start():
    """[GUARDED] Enable the run loop."""
    from auto_trader import state
    state.set_running(True, actor="api")
    return {"running": True}


@router.post("/stop")
def stop():
    """Disable the run loop. Deliberately unauthenticated — see module docstring."""
    from auto_trader import state
    state.set_running(False, actor="api")
    return {"running": False}


@router.post("/kill")
def kill_switch(
    payload: KillSwitchPayload,
    x_control_token: str | None = Header(None, alias="X-RoboAlgo-Control-Token"),
):
    """
    Set the kill switch.

    Asymmetric on purpose: turning the kill switch ON (`on=true`, i.e. stop
    trading) is unauthenticated so an operator can never be locked out.
    Turning it OFF (`on=false`, i.e. resume trading) is risk-increasing and
    requires the control token.
    """
    from auto_trader import state

    if not payload.on:
        _require_token(x_control_token)

    state.set_kill_switch(payload.on, reason=payload.reason, actor="api")
    return {"kill_switch": payload.on, "reason": payload.reason}


# ── Config ────────────────────────────────────────────────────────────────────

@router.get("/config")
def get_config():
    """Current auto-trader config."""
    from auto_trader import config as config_mod
    return config_mod.as_dict(config_mod.load())


@router.post("/config", dependencies=[Depends(_require_token)])
def update_config(payload: dict):
    """[GUARDED] Merge `payload` onto the current config and persist it."""
    from auto_trader import config as config_mod

    try:
        cfg = config_mod.save(payload)
    except config_mod.AutoTraderConfigError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return config_mod.as_dict(cfg)


# ── Cycle execution ───────────────────────────────────────────────────────────

@router.post("/cycle/run", dependencies=[Depends(_require_token)])
def trigger_cycle(inline: bool = Query(False, description="Run synchronously instead of via Celery")):
    """
    [GUARDED] Kick off one auto-trader cycle.

    `inline=false` (default) dispatches the `auto_trader_cycle` Celery task
    and returns its task id. `inline=true` runs the cycle synchronously in
    this process and is only permitted when the config has dry_run=True.
    """
    from auto_trader import config as config_mod

    if not inline:
        try:
            from infrastructure.celery_app import app as celery_app
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Celery is not available: {exc}",
            )
        async_result = celery_app.send_task("auto_trader_cycle")
        return {"task_id": async_result.id}

    cfg = config_mod.load()
    if not cfg.dry_run:
        raise HTTPException(
            status_code=400,
            detail="inline=true is only permitted when the auto-trader config has dry_run=True",
        )

    from auto_trader.loop import run_cycle
    return run_cycle(dry_run=True)


# ── Audit trail ───────────────────────────────────────────────────────────────

@router.get("/cycles")
def list_cycles(limit: int = Query(50, ge=1, le=500)):
    from auto_trader import audit
    return {"cycles": audit.recent_cycles(limit=limit)}


@router.get("/cycles/{cycle_id}/decisions")
def get_cycle_decisions(cycle_id: str):
    from auto_trader import audit
    return {"cycle_id": cycle_id, "decisions": audit.cycle_decisions(cycle_id)}


@router.get("/decisions")
def list_decisions(
    limit: int = Query(200, ge=1, le=1000),
    action: str | None = Query(None),
    symbol: str | None = Query(None),
):
    from auto_trader import audit
    return {"decisions": audit.recent_decisions(limit=limit, action=action, symbol=symbol)}


@router.get("/discrepancies")
def list_discrepancies(limit: int = Query(100, ge=1, le=1000)):
    from auto_trader import audit
    return {"discrepancies": audit.recent_discrepancies(limit=limit)}


# ── Health / decay ────────────────────────────────────────────────────────────

@router.get("/health")
def health():
    """
    Loop health snapshot: last cycle, staleness, run/kill state, broker mode,
    redis, heartbeat. Read-only and unauthenticated. Degrades to safe
    defaults rather than raising if the DB has no auto-trader tables yet.
    """
    from auto_trader.health import loop_health
    return loop_health()


@router.get("/decay")
def decay(cycles: int = Query(20, ge=1, le=500)):
    """
    Approval-rate / rejection-histogram / score-decay report over the last
    `cycles` cycles. Read-only and unauthenticated. Every statistic carries
    its own sample size (`n`) so a small-sample result is never presented as
    if it had the weight of a larger one.
    """
    from auto_trader.health import decay_report
    return decay_report(cycles)
