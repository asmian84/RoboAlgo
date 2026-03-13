"""
RoboAlgo - Backtest Engine
Evaluates historical signal performance using forward price data.
Entry = buy_price, Stop = accumulate_price (-1 ATR),
T1 = scale_price (+2 ATR), Target = sell_price (+4 ATR).
"""

import time
from collections import defaultdict

from fastapi import APIRouter
from sqlalchemy import desc, select

from database.connection import get_session
from database.models import Instrument, PriceData, Signal

router = APIRouter()

FORWARD_DAYS = 20  # look 20 trading days forward (~1 calendar month)

# Simple module-level cache (1-hour TTL)
_cache: dict = {"result": None, "ts": 0.0}
_CACHE_TTL = 3600.0


def _compute_backtest() -> dict:
    session = get_session()
    try:
        # ── Load all price data ──────────────────────────────────────────────
        price_rows = list(session.execute(
            select(
                PriceData.instrument_id, PriceData.date,
                PriceData.high, PriceData.low,
            ).order_by(PriceData.instrument_id, PriceData.date)
        ).all())

        by_instr: dict[int, list] = defaultdict(list)
        for row in price_rows:
            by_instr[row.instrument_id].append(row)

        # Precompute forward max-high / min-low for each (instrument_id, date)
        fwd_stats: dict[tuple, tuple] = {}
        for instr_id, rows in by_instr.items():
            n = len(rows)
            highs = [r.high if r.high else 0.0 for r in rows]
            lows  = [r.low  if r.low  else float("inf") for r in rows]
            dates = [r.date for r in rows]
            for i, d in enumerate(dates):
                end = min(i + 1 + FORWARD_DAYS, n)
                fwd_h = highs[i + 1 : end]
                fwd_l = lows[i + 1 : end]
                fwd_stats[(instr_id, d)] = (
                    max(fwd_h) if fwd_h else None,
                    min(fwd_l) if fwd_l else None,
                )

        # ── Load all signals ─────────────────────────────────────────────────
        sig_rows = list(session.execute(
            select(
                Signal.instrument_id, Signal.date, Signal.probability,
                Signal.confidence_tier, Signal.market_phase,
                Signal.buy_price, Signal.accumulate_price,
                Signal.scale_price, Signal.sell_price,
                Instrument.symbol,
            )
            .join(Instrument, Signal.instrument_id == Instrument.id)
            .where(Signal.buy_price.isnot(None))
            .where(Signal.sell_price.isnot(None))
        ).all())

        # ── Classify each signal ─────────────────────────────────────────────
        results = []
        for row in sig_rows:
            fwd_max_h, fwd_min_l = fwd_stats.get((row.instrument_id, row.date), (None, None))
            entry = row.buy_price or 1.0
            stop  = row.accumulate_price
            t1    = row.scale_price
            tgt   = row.sell_price

            if fwd_max_h is None or fwd_min_l is None:
                outcome = "open"
            elif stop and fwd_min_l <= stop:
                outcome = "stop"
            elif tgt and fwd_max_h >= tgt:
                outcome = "target"
            elif t1 and fwd_max_h >= t1:
                outcome = "t1"
            else:
                outcome = "open"

            results.append({
                "symbol":          row.symbol,
                "date":            row.date,
                "probability":     row.probability,
                "confidence_tier": row.confidence_tier,
                "market_phase":    row.market_phase,
                "outcome":         outcome,
                "t1_return":       round((t1  - entry) / entry * 100, 1) if t1  else None,
                "tgt_return":      round((tgt - entry) / entry * 100, 1) if tgt else None,
                "stop_return":     round((stop - entry) / entry * 100, 1) if stop else None,
            })

        # ── Aggregate helper ─────────────────────────────────────────────────
        def _stats(subset: list) -> dict:
            closed = [r for r in subset if r["outcome"] != "open"]
            wins   = [r for r in subset if r["outcome"] in ("t1", "target")]
            tgts   = [r for r in subset if r["outcome"] == "target"]
            stops  = [r for r in subset if r["outcome"] == "stop"]
            nc = len(closed) or 1

            win_rets  = [r["t1_return"]  for r in wins  if r["t1_return"]  is not None]
            tgt_rets  = [r["tgt_return"] for r in tgts  if r["tgt_return"] is not None]
            stop_rets = [r["stop_return"] for r in stops if r["stop_return"] is not None]

            return {
                "total":           len(subset),
                "closed":          len(closed),
                "wins_t1":         len(wins),
                "wins_target":     len(tgts),
                "stops":           len(stops),
                "open":            len(subset) - len(closed),
                "win_rate":        round(len(wins)  / nc * 100, 1),
                "target_rate":     round(len(tgts)  / nc * 100, 1),
                "stop_rate":       round(len(stops) / nc * 100, 1),
                "avg_t1_return":   round(sum(win_rets)  / len(win_rets),  1) if win_rets  else None,
                "avg_tgt_return":  round(sum(tgt_rets)  / len(tgt_rets),  1) if tgt_rets  else None,
                "avg_stop_return": round(sum(stop_rets) / len(stop_rets), 1) if stop_rets else None,
            }

        # ── By probability bucket ────────────────────────────────────────────
        BUCKETS = [
            ("95-100%", 0.95, 1.01),
            ("90-95%",  0.90, 0.95),
            ("80-90%",  0.80, 0.90),
            ("70-80%",  0.70, 0.80),
            ("<70%",    0.00, 0.70),
        ]
        by_prob = {}
        for label, lo, hi in BUCKETS:
            sub = [r for r in results if lo <= r["probability"] < hi]
            by_prob[label] = _stats(sub)

        # ── By market phase ──────────────────────────────────────────────────
        phases: dict[str, list] = defaultdict(list)
        for r in results:
            phases[r["market_phase"]].append(r)
        by_phase = {ph: _stats(recs) for ph, recs in phases.items()}

        # ── By confidence tier ───────────────────────────────────────────────
        tiers: dict[str, list] = defaultdict(list)
        for r in results:
            tiers[r["confidence_tier"]].append(r)
        by_tier = {t: _stats(recs) for t, recs in tiers.items()}

        return {
            "forward_days": FORWARD_DAYS,
            "overall":       _stats(results),
            "by_probability": by_prob,
            "by_phase":      by_phase,
            "by_tier":       by_tier,
        }
    finally:
        session.close()


@router.get("/stats")
def get_backtest_stats():
    """Aggregated backtest stats across all historical signals (cached 1h)."""
    now = time.time()
    if _cache["result"] is None or (now - _cache["ts"]) > _CACHE_TTL:
        _cache["result"] = _compute_backtest()
        _cache["ts"] = now
    return _cache["result"]


@router.get("/ticker/{symbol}")
def get_ticker_backtest(symbol: str):
    """All historical signals for a single ticker with forward-price outcomes."""
    symbol = symbol.upper()
    session = get_session()
    try:
        instr = session.execute(
            select(Instrument).where(Instrument.symbol == symbol)
        ).scalar_one_or_none()
        if not instr:
            return {"symbol": symbol, "signals": [], "stats": None}

        # Load price data for this instrument
        price_rows = list(session.execute(
            select(PriceData.date, PriceData.high, PriceData.low)
            .where(PriceData.instrument_id == instr.id)
            .order_by(PriceData.date)
        ).all())

        highs  = [r.high if r.high else 0.0    for r in price_rows]
        lows   = [r.low  if r.low  else 9e9    for r in price_rows]
        dates  = [r.date                         for r in price_rows]
        date_idx = {d: i for i, d in enumerate(dates)}

        # Load all signals for this instrument
        sig_rows = list(session.execute(
            select(Signal)
            .where(Signal.instrument_id == instr.id)
            .where(Signal.buy_price.isnot(None))
            .order_by(desc(Signal.date))
        ).scalars())

        records = []
        for sig in sig_rows:
            i = date_idx.get(sig.date)
            if i is not None:
                end = min(i + 1 + FORWARD_DAYS, len(price_rows))
                fwd_h = highs[i + 1 : end]
                fwd_l = lows[i + 1 : end]
                fwd_max_h = max(fwd_h) if fwd_h else None
                fwd_min_l = min(fwd_l) if fwd_l else None
            else:
                fwd_max_h = fwd_min_l = None

            entry = sig.buy_price or 1.0
            stop  = sig.accumulate_price
            t1    = sig.scale_price
            tgt   = sig.sell_price

            if fwd_max_h is None or fwd_min_l is None:
                outcome = "open"
            elif stop and fwd_min_l <= stop:
                outcome = "stop"
            elif tgt and fwd_max_h >= tgt:
                outcome = "target"
            elif t1 and fwd_max_h >= t1:
                outcome = "t1"
            else:
                outcome = "open"

            records.append({
                "date":            sig.date.isoformat(),
                "probability":     round(float(sig.probability) * 100, 1),
                "confidence_tier": sig.confidence_tier,
                "market_phase":    sig.market_phase,
                "buy_price":       round(entry, 2),
                "accumulate_price": round(stop, 2) if stop else None,
                "scale_price":     round(t1, 2)   if t1   else None,
                "sell_price":      round(tgt, 2)  if tgt  else None,
                "outcome":         outcome,
                "t1_return":  round((t1   - entry) / entry * 100, 1) if t1   else None,
                "tgt_return": round((tgt  - entry) / entry * 100, 1) if tgt  else None,
                "stop_return":round((stop - entry) / entry * 100, 1) if stop else None,
            })

        # Aggregate stats for this ticker
        closed = [r for r in records if r["outcome"] != "open"]
        wins   = [r for r in records if r["outcome"] in ("t1", "target")]
        stops  = [r for r in records if r["outcome"] == "stop"]
        tgts   = [r for r in records if r["outcome"] == "target"]
        nc     = len(closed) or 1

        win_rets  = [r["t1_return"]  for r in wins  if r["t1_return"]  is not None]
        tgt_rets  = [r["tgt_return"] for r in tgts  if r["tgt_return"] is not None]
        stop_rets = [r["stop_return"] for r in stops if r["stop_return"] is not None]

        stats = {
            "total":           len(records),
            "closed":          len(closed),
            "wins_t1":         len(wins),
            "wins_target":     len(tgts),
            "stops":           len(stops),
            "open":            len(records) - len(closed),
            "win_rate":        round(len(wins)  / nc * 100, 1),
            "target_rate":     round(len(tgts)  / nc * 100, 1),
            "stop_rate":       round(len(stops) / nc * 100, 1),
            "avg_t1_return":   round(sum(win_rets)  / len(win_rets),  1) if win_rets  else None,
            "avg_tgt_return":  round(sum(tgt_rets)  / len(tgt_rets),  1) if tgt_rets  else None,
            "avg_stop_return": round(sum(stop_rets) / len(stop_rets), 1) if stop_rets else None,
        }

        return {"symbol": symbol, "forward_days": FORWARD_DAYS, "stats": stats, "signals": records}
    finally:
        session.close()


_BUCKET_RANGES = {
    "95-100%": (0.95, 1.01),
    "90-95%":  (0.90, 0.95),
    "80-90%":  (0.80, 0.90),
    "70-80%":  (0.70, 0.80),
    "<70%":    (0.00, 0.70),
}


@router.get("/drill")
def get_backtest_drill(bucket: str = "95-100%", phase: str = ""):
    """
    Returns individual signal rows for a given probability bucket + optional phase filter.
    Returns only the most recent signal per symbol (current state), sorted by probability desc.
    """
    b = _BUCKET_RANGES.get(bucket)
    if not b:
        return {"bucket": bucket, "signals": []}
    min_prob, max_prob = b

    session = get_session()
    try:
        # Load price data for forward-outcome calculation
        price_rows = list(session.execute(
            select(
                PriceData.instrument_id, PriceData.date,
                PriceData.high, PriceData.low,
            ).order_by(PriceData.instrument_id, PriceData.date)
        ).all())

        by_instr: dict[int, list] = defaultdict(list)
        for row in price_rows:
            by_instr[row.instrument_id].append(row)

        fwd_stats: dict[tuple, tuple] = {}
        for instr_id, rows in by_instr.items():
            n = len(rows)
            highs = [r.high if r.high else 0.0 for r in rows]
            lows  = [r.low  if r.low  else float("inf") for r in rows]
            dates = [r.date for r in rows]
            for i, d in enumerate(dates):
                end = min(i + 1 + FORWARD_DAYS, n)
                fwd_h = highs[i + 1 : end]
                fwd_l = lows[i + 1 : end]
                fwd_stats[(instr_id, d)] = (
                    max(fwd_h) if fwd_h else None,
                    min(fwd_l) if fwd_l else None,
                )

        # Build query — latest signal per symbol in this bucket
        q = (
            select(
                Signal.instrument_id, Signal.date, Signal.probability,
                Signal.confidence_tier, Signal.market_phase,
                Signal.buy_price, Signal.accumulate_price,
                Signal.scale_price, Signal.sell_price,
                Instrument.symbol,
            )
            .join(Instrument, Signal.instrument_id == Instrument.id)
            .where(Signal.buy_price.isnot(None))
            .where(Signal.probability >= min_prob)
            .where(Signal.probability < max_prob)
        )
        if phase:
            q = q.where(Signal.market_phase == phase)

        sig_rows = list(session.execute(q.order_by(desc(Signal.date), desc(Signal.probability))).all())

        # Keep latest per symbol
        seen: set[str] = set()
        records = []
        for row in sig_rows:
            if row.symbol in seen:
                continue
            seen.add(row.symbol)

            fwd_max_h, fwd_min_l = fwd_stats.get((row.instrument_id, row.date), (None, None))
            entry = row.buy_price or 1.0
            stop  = row.accumulate_price
            t1    = row.scale_price
            tgt   = row.sell_price

            if fwd_max_h is None or fwd_min_l is None:
                outcome = "open"
            elif stop and fwd_min_l <= stop:
                outcome = "stop"
            elif tgt and fwd_max_h >= tgt:
                outcome = "target"
            elif t1 and fwd_max_h >= t1:
                outcome = "t1"
            else:
                outcome = "open"

            records.append({
                "symbol":          row.symbol,
                "date":            row.date.isoformat(),
                "probability":     round(float(row.probability) * 100, 1),
                "confidence_tier": row.confidence_tier,
                "market_phase":    row.market_phase,
                "buy_price":       round(entry, 2),
                "outcome":         outcome,
                "t1_return":       round((t1   - entry) / entry * 100, 1) if t1   else None,
                "tgt_return":      round((tgt  - entry) / entry * 100, 1) if tgt  else None,
                "stop_return":     round((stop - entry) / entry * 100, 1) if stop else None,
            })

        records.sort(key=lambda r: r["probability"], reverse=True)
        return {"bucket": bucket, "phase": phase, "signals": records}
    finally:
        session.close()


@router.get("/similar/{symbol}")
def get_similar_setups(symbol: str):
    """Win rate for historical setups similar to the current signal for this symbol."""
    symbol = symbol.upper()
    session = get_session()
    try:
        latest = session.execute(
            select(Signal, Instrument.symbol)
            .join(Instrument, Signal.instrument_id == Instrument.id)
            .where(Instrument.symbol == symbol)
            .order_by(desc(Signal.date))
            .limit(1)
        ).first()

        if not latest:
            return {"symbol": symbol, "found": False}

        sig, sym = latest
        phase = sig.market_phase
        prob  = sig.probability

        stats = get_backtest_stats()

        # Match probability bucket
        prob_bucket, prob_stats = None, {}
        for label, lo, hi in [
            ("95-100%", 0.95, 1.01), ("90-95%", 0.90, 0.95),
            ("80-90%",  0.80, 0.90), ("70-80%", 0.70, 0.80),
            ("<70%",    0.00, 0.70),
        ]:
            if lo <= prob < hi:
                prob_bucket = label
                prob_stats  = stats["by_probability"].get(label, {})
                break

        phase_stats = stats["by_phase"].get(phase, {})

        return {
            "symbol":       symbol,
            "found":        True,
            "phase":        phase,
            "probability":  round(prob * 100, 1),
            "prob_bucket":  prob_bucket,
            "phase_stats":  phase_stats,
            "prob_stats":   prob_stats,
        }
    finally:
        session.close()
