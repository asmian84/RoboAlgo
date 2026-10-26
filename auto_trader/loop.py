"""
auto_trader/loop.py — the cycle orchestrator.

Single public entry point: `run_cycle()`. Ties together config, state (kill
switch / run flag), the reconciler (broker-is-truth), the governor (risk
policy), a trade-signal scan, and the audit trail into one autonomous pass:

    load config -> safety gates -> reconcile broker state -> halt check ->
    scan for candidates -> per-candidate risk-gate + place -> record outcome

Safety posture (do not weaken any of these without a corresponding plan
change):
  * `cfg.enabled=False` (the shipped default) short-circuits before any I/O
    or DB write.
  * The kill switch and run flag are re-checked at the very top, and the
    kill switch is re-checked again on every iteration of the candidate
    loop, since a long-running cycle must react to a kill mid-flight.
  * Redis is what makes the kill switch cross-process. If Redis is not
    reachable, a non-dry-run request is forcibly downgraded to dry_run —
    the kill switch cannot be trusted to stop a live order in that state.
  * This function NEVER raises. Every path returns a status dict, and the
    top-level handler converts any unexpected exception into a "failed"
    cycle record instead of propagating (a raised exception plus a Celery
    retry would silently re-run a trading cycle).
"""

from __future__ import annotations

import logging
import time
import traceback

from auto_trader import audit, config, governor, reconciler, state
from auto_trader.config import AutoTraderConfigError
from auto_trader.governor import CandidateOrder
from broker_engine.factory import get_broker

logger = logging.getLogger(__name__)


def _last_known_positions() -> dict:
    """
    Reconstruct the {symbol: {"qty", "market_value", "unrealized_pnl"}} map
    the reconciler expects as `last_known`, from the previous cycle's
    persisted `positions` payload. Tolerates a missing/empty history or a
    malformed payload (audit.recent_cycles already guards bad JSON; this
    just guards the shape once it's back in Python).
    """
    try:
        cycles = audit.recent_cycles(1)
    except Exception:
        logger.warning("loop: failed to load recent_cycles for last_known", exc_info=True)
        return {}
    if not cycles:
        return {}
    positions = cycles[0].get("positions")
    if not isinstance(positions, dict):
        return {}
    return positions


def _positions_payload(positions) -> dict:
    """Broker Position objects -> the plain-dict shape stored on the cycle row
    (and re-read as `last_known` by the next cycle's reconcile())."""
    return {
        p.symbol: {
            "qty": p.qty,
            "market_value": p.market_value,
            "unrealized_pnl": p.unrealized_pnl,
        }
        for p in positions
    }


def _default_symbols(cfg) -> list[str]:
    """Resolve the scan universe when cfg.symbols is empty."""
    if cfg.asset_class == "crypto":
        from trade_signal_engine.crypto_scanner import get_crypto_universe
        return get_crypto_universe(limit=cfg.max_symbols)

    # Fast path proven in api/routers/trade_signals.py: curated tickers
    # instead of the slow full-universe download.
    from trade_signal_engine.universe import get_options_universe, OPTIONS_FOCUS, _BROAD_UNIVERSE
    return list(dict.fromkeys(OPTIONS_FOCUS + _BROAD_UNIVERSE[:80]))[: cfg.max_symbols]


def _coerce_confidence(raw) -> float:
    """Scanner confidence may be a numeric score or a string band
    ("LOW"/"MEDIUM"/"HIGH"/"NONE"). Coerce anything non-numeric to 0.0."""
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def run_cycle(cycle_id: str | None = None, *, dry_run: bool | None = None) -> dict:
    cid = None
    try:
        cfg = config.load()

        if not cfg.enabled:
            return {"status": "disabled"}

        if state.kill_switch_on():
            return {"status": "killed"}

        if not state.running_enabled():
            return {"status": "stopped"}

        cid = cycle_id or audit.new_cycle_id(cfg.mode)

        try:
            broker = get_broker(cfg.broker)
            config.assert_mode_safe(broker, cfg)
        except AutoTraderConfigError as exc:
            logger.error("loop: config unsafe for cycle %s: %s", cid, exc)
            audit.start_cycle(cid, cfg.mode, cfg.broker, None, None)
            audit.finish_cycle(cid, "failed", error=str(exc))
            return {"status": "failed", "cycle_id": cid, "error": str(exc)}

        # NEW SAFETY GATE: Redis is what makes the kill switch cross-process.
        # Without it the kill switch is only host-local, so a non-dry-run
        # cycle on this host cannot be trusted to be stoppable — force dry.
        effective_dry = cfg.dry_run if dry_run is None else dry_run
        if not effective_dry and not state.redis_available():
            effective_dry = True
            audit.log_decision(
                cid, "-", "NO_ACTION",
                "redis unavailable - forced dry_run; kill switch cannot be "
                "trusted across processes",
            )

        # Market-hours gate. Inaction must still be audited.
        if not cfg.ignore_market_hours and not broker.is_market_open():
            audit.start_cycle(cid, cfg.mode, broker.name, None, None)
            audit.log_decision(cid, "-", "NO_ACTION", "market closed")
            audit.finish_cycle(cid, "skipped")
            return {"status": "skipped", "cycle_id": cid}

        last_known = _last_known_positions()
        rec = reconciler.reconcile(broker, cid, last_known)

        dse = audit.day_start_equity(rec.account.portfolio_val)
        cl = audit.consecutive_losses()
        audit.start_cycle(cid, cfg.mode, broker.name, rec.account.portfolio_val, dse)

        snap = reconciler.build_portfolio_snapshot(rec.account, rec.positions, dse, cl)
        halt = governor.check_halts(snap, cfg.risk, state.kill_switch_on())
        if not halt.approved:
            audit.log_decision(cid, "-", "HALT", halt.reason, governor_rule=halt.rule)
            audit.finish_cycle(
                cid, "halted", equity_end=snap.equity, halt_reason=halt.rule,
                positions=_positions_payload(rec.positions),
            )
            return {"status": "halted", "cycle_id": cid, "halt_reason": halt.rule, "reason": halt.reason}

        # ── Scan ────────────────────────────────────────────────────────────
        try:
            symbols = cfg.symbols or _default_symbols(cfg)
            if cfg.asset_class == "crypto":
                from trade_signal_engine.crypto_scanner import scan_crypto_universe
                result = scan_crypto_universe(
                    symbols=symbols,
                    profile=cfg.profile,
                    account_size=snap.equity,
                )
            else:
                from trade_signal_engine.scanner import scan_universe
                result = scan_universe(
                    symbols=symbols,
                    profile=cfg.profile,
                    account_size=snap.equity,
                    max_workers=cfg.max_workers,
                )
        except Exception as exc:
            logger.error("loop: scan failed for cycle %s: %s", cid, exc, exc_info=True)
            audit.finish_cycle(cid, "failed", equity_end=snap.equity, error=str(exc))
            return {"status": "failed", "cycle_id": cid, "error": str(exc)}

        candidates = sorted(
            [r for r in result["recommendations"] if r["verdict"] in ("BUY", "SELL")],
            key=lambda r: r["score"],
            reverse=True,
        )[: cfg.max_new_positions_per_cycle]

        if not candidates:
            audit.log_decision(
                cid, "-", "NO_ACTION",
                f"scanned {result['symbols_scanned']} symbols, 0 BUY/SELL verdicts",
            )
            audit.finish_cycle(
                cid, "completed",
                equity_end=snap.equity,
                symbols_scanned=result.get("symbols_scanned", 0),
                candidates=0,
                orders_placed=0,
                orders_rejected=0,
                regime=result.get("regime", {}).get("regime"),
                positions=_positions_payload(rec.positions),
            )
            return {
                "status": "completed", "cycle_id": cid,
                "symbols_scanned": result.get("symbols_scanned", 0),
                "candidates": 0, "orders_placed": 0, "orders_rejected": 0,
            }

        # ── Per-candidate loop ──────────────────────────────────────────────
        orders_placed = 0
        orders_rejected = 0
        # Set when the candidate loop is cut short by a halt. The cycle must
        # NOT be recorded as "completed" in that case — an operator reading
        # the audit trail has to be able to tell a clean finish from a kill.
        halted_rule: str | None = None

        for r in candidates:
            if state.kill_switch_on():
                audit.log_decision(
                    cid, "-", "HALT", "kill switch tripped mid-cycle",
                    governor_rule="kill_switch",
                )
                halted_rule = "kill_switch"
                break

            mid_halt = governor.check_halts(snap, cfg.risk, state.kill_switch_on())
            if not mid_halt.approved:
                audit.log_decision(cid, "-", "HALT", mid_halt.reason, governor_rule=mid_halt.rule)
                halted_rule = mid_halt.rule
                break

            symbol = r["symbol"]
            side = "buy" if r["direction"] in ("long", "LONG", "BUY") else "sell"
            confidence = _coerce_confidence(r.get("confidence"))

            cand = CandidateOrder(
                symbol=symbol,
                side=side,
                qty=int(r["position_size"]),
                entry=r["entry"],
                stop=r["stop"],
                target=r["target"],
                notional=r["position_value"],
                risk_amount=r["risk_amount"],
                score=r["score"],
                confidence=confidence,
            )

            v = governor.evaluate(cand, snap, cfg.risk, state.kill_switch_on())
            if not v.approved:
                audit.log_decision(
                    cid, symbol, "REJECT", v.reason,
                    governor_rule=v.rule, verdict=r["verdict"], direction=r["direction"],
                    score=r["score"], confidence=confidence, entry=r["entry"], stop=r["stop"],
                    target=r["target"], qty=cand.qty, risk_amount=r["risk_amount"],
                )
                orders_rejected += 1
                state.set_heartbeat({
                    "cycle_id": cid, "symbol": symbol, "action": "REJECT",
                    "orders_placed": orders_placed, "orders_rejected": orders_rejected,
                    "ts": time.time(),
                })
                continue

            coid = reconciler.client_order_id_for(cid, symbol, side)

            if (
                audit.has_placed(coid)
                or reconciler.order_already_exists(rec.open_orders, coid)
                or broker.get_order_by_client_id(coid) is not None
            ):
                audit.log_decision(cid, symbol, "SKIP", "duplicate client_order_id", client_order_id=coid)
                state.set_heartbeat({
                    "cycle_id": cid, "symbol": symbol, "action": "SKIP",
                    "orders_placed": orders_placed, "orders_rejected": orders_rejected,
                    "ts": time.time(),
                })
                continue

            if effective_dry:
                audit.log_decision(
                    cid, symbol, "DRY_RUN", "dry_run enabled", client_order_id=coid,
                    verdict=r["verdict"], direction=r["direction"], score=r["score"],
                    confidence=confidence, entry=r["entry"], stop=r["stop"], target=r["target"],
                    qty=cand.qty, risk_amount=r["risk_amount"],
                )
                state.set_heartbeat({
                    "cycle_id": cid, "symbol": symbol, "action": "DRY_RUN",
                    "orders_placed": orders_placed, "orders_rejected": orders_rejected,
                    "ts": time.time(),
                })
                continue

            try:
                order = broker.place_bracket_order(
                    symbol, cand.qty, side,
                    limit_price=cand.entry, take_profit=cand.target, stop_loss=cand.stop,
                    client_order_id=coid,
                )
            except Exception as exc:
                audit.log_decision(
                    cid, symbol, "SKIP", str(exc),
                    governor_rule="broker_error", client_order_id=coid,
                )
            else:
                audit.log_decision(
                    cid, symbol, "PLACE", "order placed via bracket order",
                    broker_order_id=order.order_id, client_order_id=coid,
                    verdict=r["verdict"], direction=r["direction"], score=r["score"],
                    confidence=confidence, entry=r["entry"], stop=r["stop"], target=r["target"],
                    qty=cand.qty, risk_amount=r["risk_amount"],
                )
                orders_placed += 1
            finally:
                # The order may have landed even if place_bracket_order raised —
                # re-reconcile against real broker state before the next candidate.
                rec = reconciler.reconcile(broker, cid, last_known=None)
                snap = reconciler.build_portfolio_snapshot(rec.account, rec.positions, dse, cl)

            state.set_heartbeat({
                "cycle_id": cid, "symbol": symbol, "action": "PLACE",
                "orders_placed": orders_placed, "orders_rejected": orders_rejected,
                "ts": time.time(),
            })

        final_status = "halted" if halted_rule else "completed"
        audit.finish_cycle(
            cid, final_status,
            equity_end=snap.equity,
            symbols_scanned=result.get("symbols_scanned", 0),
            candidates=len(candidates),
            orders_placed=orders_placed,
            orders_rejected=orders_rejected,
            regime=result.get("regime", {}).get("regime"),
            positions=_positions_payload(rec.positions),
            halt_reason=halted_rule,
        )
        return {
            "status": final_status, "cycle_id": cid,
            "symbols_scanned": result.get("symbols_scanned", 0),
            "candidates": len(candidates),
            "orders_placed": orders_placed,
            "orders_rejected": orders_rejected,
            "halt_reason": halted_rule,
        }

    except Exception:
        tb = traceback.format_exc()
        logger.error("loop: run_cycle crashed:\n%s", tb)
        if cid is not None:
            audit.finish_cycle(cid, "failed", error=tb)
        return {"status": "failed"}
