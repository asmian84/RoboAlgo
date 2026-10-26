"""
scanner_engine/finder.py
────────────────────────────────────────────────────────────────────────────────
The Finder — a confluence screener.

Composes EXISTING data (indicators, range_compressions, breakout_signals,
price_data). Does not reimplement indicator math, compression detection, or
breakout logic — those live in indicator_engine/, range_engine/, and the
breakout pipeline stages respectively.

Public API:
    FinderCriteria   — dataclass of tunable screening thresholds
    PRESETS          — {"signal", "pennyking", "maker"} -> FinderCriteria
    scan(criteria, limit=100) -> list[dict]
    health() -> dict   — per-table freshness, so a stale backfill is visible

Performance note: every check below is driven by data loaded in bulk (one
query per table, latest-row-per-instrument or last-N-bars-per-instrument via
a window function), never a per-instrument round trip. The one exception is
the live confluence call, which is deliberately slow and per-symbol — it is
only ever invoked for candidates that have already passed every other check.

Wide-universe scanning:
    scan_wide(criteria, limit=100, max_deep=400, universe=None) -> dict
        `scan()` above only ever sees the ~234 instruments that have
        precomputed rows in indicators/range_compressions/breakout_signals —
        almost entirely large caps and leveraged ETFs, so sub-$5 / sub-$20
        presets return nothing there. `scan_wide` instead scans the full
        live US ticker universe (~13,786 symbols via
        trade_signal_engine.universe._get_all_exchange_tickers) with NO DB
        reads: a cheap yfinance bulk 1-day screen narrows the universe by
        price/volume, then a deep per-symbol OHLCV pass (live, ~250 daily
        bars) computes every check that scan() would have pulled from the
        DB. It is slower (minutes, not milliseconds) and confluence is
        skipped entirely (see scan_wide's docstring), but it is the only
        path that can ever find a real penny stock.

        Stage 1 prefers TradingView's server-side screener (one request,
        full universe, ~seconds) via data_engine.tradingview_provider and
        falls back to the yfinance batch loop above when that is
        unavailable. `stats["stage1_source"]` says which one ran. TradingView
        is trusted for symbol / price / volume ONLY — see _stage1_screen.
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from statistics import mean

import numpy as np
import pandas as pd
import yfinance as yf
from sqlalchemy import func, select

from database.connection import get_session
from database.models import BreakoutSignal, Indicator, Instrument, PriceData, RangeCompression
from range_engine.compression import (
    ATR_THRESHOLD,
    BB_WIDTH_THRESHOLD,
    COMPRESSION_LOOKBACK,
    RANGE_LONG,
    RANGE_SHORT,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FinderCriteria:
    min_price: float | None = None
    max_price: float | None = None
    min_volume: int = 100_000
    require_golden_cross: bool = True     # ma50 > ma200 AND close > ma50
    require_consolidation: bool = True    # compressed recently
    max_gap_pct: float = 3.0              # reject if any recent gap exceeds this
    gap_lookback: int = 20
    min_confluence: float = 0.0           # 0 = skip the (slow) live confluence call
    require_breakout_ready: bool = False
    consolidation_lookback: int = 30      # bars back to look for compression
    min_compression_score: float = 50.0
    # Reject an instrument whose newest bar is older than this many calendar
    # days. Without it `scan()` happily evaluates a delisted ticker's last
    # known rows as if they were today's: CFLT stopped trading on 2026-03-05
    # and was still being returned as the TOP result seven months later, with
    # a confident-looking entry/stop/target built on a dead price. Nothing in
    # the output said the quote was old, and the health panel counted it
    # inside "230/234 current".
    #
    # 5 days absorbs a Friday bar read on the Tuesday after a Monday holiday
    # without letting a genuinely dead symbol through.
    max_bar_age_days: int = 5


# Gap tolerance is calibrated PER PRICE BAND, because "no gaps" is not one
# number across asset classes. Measured over deep-analysed candidates in each
# band (max gap in the trailing 20 bars):
#
#     large caps ($20+)   median ~1-2%
#     $5-20               median  6.0%
#     sub-$5              median  6.1%   (tail out to 442%)
#
# A single 3% threshold is a large-cap number. Applied to penny names it
# rejected ~80% of the universe on volatility alone, before any signal logic
# ran — which is why pennyking returned nothing from a pool that did contain
# genuine setups. These values sit just above each band's observed median, so
# the filter still rejects the gappy half of the band rather than waving
# everything through.
#
# Note what is deliberately NOT relaxed: require_golden_cross. Only ~6% of
# penny names are above their golden cross, and that is the signal being asked
# for, not an obstacle to it.
PRESETS: dict[str, FinderCriteria] = {
    "signal": FinderCriteria(
        max_gap_pct=3.0,
    ),
    "pennyking": FinderCriteria(
        max_price=5.0,
        min_volume=500_000,
        max_gap_pct=8.0,
    ),
    "maker": FinderCriteria(
        min_price=5.0,
        max_price=20.0,
        min_volume=250_000,
        max_gap_pct=6.0,
    ),
}


# ── Bulk loaders ──────────────────────────────────────────────────────────────
# Each of these is exactly one round trip to the DB, regardless of universe size.

def _load_instruments(session) -> list[tuple[int, str]]:
    rows = session.execute(select(Instrument.id, Instrument.symbol)).all()
    return [(r.id, r.symbol) for r in rows]


def _load_latest_indicators(session) -> dict[int, Indicator]:
    """Latest indicator row per instrument (group-by-max-date, joined back)."""
    latest = (
        select(Indicator.instrument_id, func.max(Indicator.date).label("max_date"))
        .group_by(Indicator.instrument_id)
        .subquery()
    )
    rows = session.execute(
        select(Indicator).join(
            latest,
            (Indicator.instrument_id == latest.c.instrument_id)
            & (Indicator.date == latest.c.max_date),
        )
    ).scalars().all()
    return {row.instrument_id: row for row in rows}


def _last_n_rows(session, model, n: int, columns) -> dict[int, list]:
    """
    Last N rows per instrument for `model`, via a window function — one query.
    Returns {instrument_id: [row, ...]} sorted ascending by date.
    """
    ranked = (
        select(
            *columns,
            func.row_number()
            .over(partition_by=model.instrument_id, order_by=model.date.desc())
            .label("rn"),
        )
        .subquery()
    )
    rows = session.execute(select(ranked).where(ranked.c.rn <= n)).all()
    grouped: dict[int, list] = defaultdict(list)
    for row in rows:
        grouped[row.instrument_id].append(row)
    for instrument_id in grouped:
        grouped[instrument_id].sort(key=lambda r: r.date)
    return grouped


def _load_recent_prices(session, n: int) -> dict[int, list]:
    cols = [PriceData.instrument_id, PriceData.date, PriceData.open, PriceData.close, PriceData.volume]
    return _last_n_rows(session, PriceData, n, cols)


def _load_recent_compressions(session, n: int) -> dict[int, list]:
    cols = [
        RangeCompression.instrument_id, RangeCompression.date,
        RangeCompression.is_compressed, RangeCompression.compression_score,
        RangeCompression.compression_duration,
        # The consolidation bounds ARE the trade: entry at the high, stop at
        # the low, target a range-height above. Without these columns the
        # screener can only say "interesting" and never "here is the trade".
        RangeCompression.range_high, RangeCompression.range_low,
    ]
    return _last_n_rows(session, RangeCompression, n, cols)


def _load_recent_breakouts(session, n: int) -> dict[int, list]:
    cols = [
        BreakoutSignal.instrument_id, BreakoutSignal.date,
        BreakoutSignal.breakout_strength, BreakoutSignal.triggers_met,
    ]
    return _last_n_rows(session, BreakoutSignal, n, cols)


# ── Trade levels ──────────────────────────────────────────────────────────────

# A price within this many percent of the entry level counts as "at entry": close enough that
# the buy-stop is about to work or just has. Beyond it the setup is clearly below or past entry.
AT_ENTRY_PCT = 0.5

# The five states a listed setup can be in, relative to its own levels. Checked in this order.
ENTRY_STATUSES = ("past_target", "below_stop", "past_entry", "at_entry", "below_entry")


def trade_levels(range_high, range_low, price) -> dict:
    """Derive entry / stop / target from the consolidation the screener matched.

    A screener that only returns "this looks interesting" leaves the actual
    decision unmade. For a consolidation-breakout — which is exactly the setup
    these presets select for — the structure already defines the trade:

        entry   the top of the range. You are buying the break, not the range.
        stop    the bottom of the range. Below it the consolidation has failed
                and the reason for the trade no longer exists.
        target  one range-height above entry — the measured move, the standard
                projection for a range break.

    Honest note on risk_reward: with a full-range stop this is ~1.0 by
    construction, because risk and reward are both one range height. That is
    the arithmetic, not a flattering number — a trader wanting better than 1:1
    tightens the stop (range_mid is the usual choice, which doubles it), and
    that is their call to make, not something to bake in so the screen looks
    better than it is. range_mid is returned so the UI can offer it.

    `risk_reward` is the PLANNED ratio, measured from the entry level. It is not
    what a buyer gets once the setup is past its trigger, and replaying the
    Finder over the whole listed universe (docs/FINDER_VALIDATION_2026-10-06.md)
    showed that is the usual case: 55-57% of fills happened ABOVE the displayed
    entry, so the realised ratio averaged 0.68, not 1.0. Two things therefore
    come with the levels:

        entry_status  where the price is relative to the setup:
                        below_entry  not triggered; a buy-stop at entry is still working
                        at_entry     within AT_ENTRY_PCT of entry, either side
                        past_entry   already through the trigger; chasing
                        below_stop   at or under the stop: the consolidation has failed
                        past_target  at or over the target: the measured move is done
        rr_real       reward-to-risk from `fill_ref`, the price a buyer would actually
                      pay: the entry level while untriggered, the current price once
                      past it. None for below_stop / past_target, where there is no
                      trade left to take. risk_pct / reward_pct are the same two legs
                      as a percentage of fill_ref.

    Returns all-None when there is no consolidation to measure from, rather
    than inventing levels off a price alone.
    """
    try:
        rh = float(range_high) if range_high is not None else None
        rl = float(range_low) if range_low is not None else None
        px = float(price) if price is not None else None
    except (TypeError, ValueError):
        rh = rl = px = None

    if rh is None or rl is None or rh <= rl:
        return {
            "entry": None, "stop": None, "target": None, "risk_reward": None,
            "range_high": None, "range_low": None, "range_mid": None,
            "entry_status": None, "fill_ref": None, "rr_real": None,
            "risk_pct": None, "reward_pct": None,
        }

    height = rh - rl
    entry = rh
    stop = rl
    target = rh + height
    risk = entry - stop
    rr = round((target - entry) / risk, 2) if risk > 0 else None

    status = fill_ref = rr_real = risk_pct = reward_pct = None
    if px:
        above_entry_pct = (px - entry) / entry * 100       # + means price is already past the trigger
        if px >= target:
            status = "past_target"
        elif px <= stop:
            status = "below_stop"
        elif above_entry_pct > AT_ENTRY_PCT:
            status = "past_entry"
        elif above_entry_pct >= -AT_ENTRY_PCT:
            status = "at_entry"
        else:
            status = "below_entry"
        if status in ("below_entry", "at_entry", "past_entry"):
            fill_ref = max(px, entry)                       # you cannot buy the break below the break
            rr_real = round((target - fill_ref) / (fill_ref - stop), 2)
            risk_pct = round((fill_ref - stop) / fill_ref * 100, 2)
            reward_pct = round((target - fill_ref) / fill_ref * 100, 2)

    return {
        "entry": round(entry, 4),
        "stop": round(stop, 4),
        "target": round(target, 4),
        "risk_reward": rr,
        "range_high": round(rh, 4),
        "range_low": round(rl, 4),
        "range_mid": round((rh + rl) / 2, 4),
        # How far price sits below the trigger — 0 means it is breaking out now,
        # negative means it is already through.
        "distance_to_entry_pct": round((entry - px) / px * 100, 2) if px else None,
        "entry_status": status,
        "fill_ref": round(fill_ref, 4) if fill_ref is not None else None,
        "rr_real": rr_real,
        "risk_pct": risk_pct,
        "reward_pct": reward_pct,
    }


# ── Per-candidate evaluation ──────────────────────────────────────────────────

def _evaluate(symbol: str, criteria: FinderCriteria, prices: list, indicator, compressions: list, breakouts: list) -> dict | None:
    if not prices:
        return None
    latest = prices[-1]
    price = latest.close
    volume = latest.volume
    if price is None:
        return None

    checks: dict[str, bool] = {}
    required: list[str] = []

    # 1. price_band — always evaluated
    price_ok = True
    if criteria.min_price is not None:
        price_ok = price_ok and price >= criteria.min_price
    if criteria.max_price is not None:
        price_ok = price_ok and price <= criteria.max_price
    checks["price_band"] = price_ok
    required.append("price_band")

    # 2. liquidity — always evaluated
    vol_window = [p.volume for p in prices[-20:] if p.volume is not None]
    avg_volume_20 = mean(vol_window) if vol_window else 0.0
    checks["liquidity"] = avg_volume_20 >= criteria.min_volume
    required.append("liquidity")

    # 3. golden_cross — always computed, gated only if required
    ma50 = indicator.ma50 if indicator is not None else None
    ma200 = indicator.ma200 if indicator is not None else None
    golden_cross = ma50 is not None and ma200 is not None and ma50 > ma200 and price > ma50
    checks["golden_cross"] = golden_cross
    if criteria.require_golden_cross:
        required.append("golden_cross")

    # 4. consolidating — always computed, gated only if required
    window = compressions[-criteria.consolidation_lookback:] if compressions else []
    compressed_bars = [b for b in window if b.is_compressed and (b.compression_score or 0) >= criteria.min_compression_score]
    consolidating = len(compressed_bars) > 0
    checks["consolidating"] = consolidating
    if criteria.require_consolidation:
        required.append("consolidating")

    if compressed_bars:
        driver = max(compressed_bars, key=lambda b: b.compression_score or 0)
        compression_score = driver.compression_score
        compression_duration = driver.compression_duration
    elif window:
        driver = window[-1]
        compression_score = driver.compression_score
        compression_duration = driver.compression_duration
    else:
        compression_score = None
        compression_duration = None

    # 5. no_gaps — always evaluated
    gap_window = prices[-(criteria.gap_lookback + 1):]
    gaps = []
    for i in range(1, len(gap_window)):
        prev_close = gap_window[i - 1].close
        cur_open = gap_window[i].open
        if prev_close and cur_open is not None:
            gaps.append(abs(cur_open - prev_close) / prev_close * 100)
    gap_max_pct = max(gaps) if gaps else 0.0
    checks["no_gaps"] = gap_max_pct <= criteria.max_gap_pct
    required.append("no_gaps")

    # 6. breakout_ready — only checked (and only gates) when required
    breakout_strength = breakouts[-1].breakout_strength if breakouts else None
    if criteria.require_breakout_ready:
        recent_price_dates = {p.date for p in prices[-10:]}
        breakout_ready = any(b.date in recent_price_dates for b in breakouts)
        checks["breakout_ready"] = breakout_ready
        required.append("breakout_ready")

    # Fast checks done — bail before the expensive confluence call if this
    # candidate already doesn't qualify.
    if not all(checks[k] for k in required):
        return None

    confluence = None
    if criteria.min_confluence > 0:
        try:
            from confluence_engine.score import ConfluenceEngine

            live = ConfluenceEngine().score_symbol_live(symbol)
            confluence = live.get("confluence_score") if isinstance(live, dict) else None
        except Exception as exc:
            logger.warning("live confluence failed for %s: %s", symbol, exc)
            confluence = None
        checks["confluence"] = confluence is not None and confluence >= criteria.min_confluence
        required.append("confluence")
        if not checks["confluence"]:
            return None

    passed = sum(1 for v in checks.values() if v)
    failed = len(checks) - passed

    levels = trade_levels(
        getattr(driver, "range_high", None) if compression_score is not None else None,
        getattr(driver, "range_low", None) if compression_score is not None else None,
        price,
    )

    return {
        "symbol": symbol,
        "price": price,
        "volume": volume,
        "checks": checks,
        "passed": passed,
        "failed": failed,
        "ma50": ma50,
        "ma200": ma200,
        "avg_volume_20": avg_volume_20,
        "compression_score": compression_score,
        "compression_duration": compression_duration,
        "gap_max_pct": gap_max_pct,
        "breakout_strength": breakout_strength,
        "confluence": confluence,
        **levels,
    }


# ── Public entry point ────────────────────────────────────────────────────────

def attach_setup_evidence(results: list[dict], daily_by_symbol: dict | None = None) -> dict:
    """Add `setup_evidence` to each result: which APPROVED learned setups it matches, and what the
    evidence behind each says. Tags only. It never re-ranks, filters or drops a result, and it never
    raises: on failure each result carries `setup_evidence: None` plus an error, and the returned status
    says so. See setup_engine/apply.py for what "approved" means and what the tags do and do not claim."""
    try:
        from setup_engine import apply as setup_apply
        return setup_apply.annotate(results, daily_by_symbol)
    except Exception as exc:                                    # noqa: BLE001
        logger.warning("setup evidence unavailable: %s", exc)
        for r in results:
            r["setup_evidence"] = None
            r["setup_evidence_error"] = f"{type(exc).__name__}: {exc}"
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _db_daily_bars(symbols: list[str], bars: int = 600) -> dict:
    """Recent daily bars for a few tracked instruments, straight from the price table."""
    if not symbols:
        return {}
    with get_session() as session:
        rows = session.execute(
            select(Instrument.symbol, PriceData.date, PriceData.open, PriceData.high,
                   PriceData.low, PriceData.close, PriceData.volume)
            .join(PriceData, PriceData.instrument_id == Instrument.id)
            .where(Instrument.symbol.in_(symbols), PriceData.close.isnot(None))
        ).all()
    if not rows:
        return {}
    df = pd.DataFrame(rows, columns=["symbol", "date", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"])
    out = {}
    for sym, g in df.groupby("symbol"):
        out[sym] = g.set_index("date").drop(columns="symbol").sort_index().astype(float).iloc[-bars:]
    return out


def scan(criteria: FinderCriteria, limit: int = 100) -> list[dict]:
    with get_session() as session:
        instruments = _load_instruments(session)
        indicators_by_id = _load_latest_indicators(session)

        price_n = max(20, criteria.gap_lookback) + 1
        prices_by_id = _load_recent_prices(session, price_n)
        compressions_by_id = _load_recent_compressions(session, criteria.consolidation_lookback)
        breakouts_by_id = _load_recent_breakouts(session, 10)

        cutoff = date.today() - timedelta(days=criteria.max_bar_age_days)

        results = []
        stale_symbols: list[str] = []
        for instrument_id, symbol in instruments:
            prices = prices_by_id.get(instrument_id, [])

            # Recency gate, before any signal logic. `prices` is ascending by
            # date, so [-1] is the newest bar this instrument has. A symbol
            # that stopped trading keeps its last rows forever and every
            # downstream check happily passes on them — the screener cannot
            # tell "quiet consolidation" from "delisted".
            if prices and criteria.max_bar_age_days > 0:
                newest = prices[-1].date
                if newest < cutoff:
                    stale_symbols.append(f"{symbol} ({newest})")
                    continue

            candidate = _evaluate(
                symbol,
                criteria,
                prices,
                indicators_by_id.get(instrument_id),
                compressions_by_id.get(instrument_id, []),
                breakouts_by_id.get(instrument_id, []),
            )
            if candidate is not None:
                results.append(candidate)

        if stale_symbols:
            logger.warning(
                "Finder excluded %d instrument(s) with no bar in the last %d "
                "days: %s. These are usually delisted or renamed and should be "
                "removed from the tracked universe.",
                len(stale_symbols), criteria.max_bar_age_days,
                ", ".join(sorted(stale_symbols)),
            )

    results.sort(
        key=lambda r: (r["compression_score"] or 0, r["breakout_strength"] or 0),
        reverse=True,
    )
    results = results[:limit]
    attach_setup_evidence(results, _db_daily_bars([r["symbol"] for r in results]))
    return results


def _quiet_yfinance():
    """Delisted symbols are routine at 13k-ticker scale and yfinance logs a
    warning + prints a summary line per failure. Silence its logger for the
    duration of a bulk call — we count failures ourselves instead."""
    yf_logger = logging.getLogger("yfinance")
    prev_level = yf_logger.level
    yf_logger.setLevel(logging.CRITICAL)
    return yf_logger, prev_level


# ── Stage 1: cheap bulk screen (price/volume only, no history) ────────────────

_STAGE1_BATCH = 300
# (timestamp, survivors, failed, source) — source is "tradingview" | "yfinance".
_STAGE1_CACHE: dict[tuple, tuple[float, list[dict], int, str]] = {}
_STAGE1_CACHE_TTL = 900  # 15 min

# A provider price may sit this far outside the requested band before it is
# treated as a violation. Pure float-noise tolerance, not a fuzzy margin.
_BAND_EPS = 1e-9


def _tradingview_stage1(criteria: FinderCriteria) -> tuple[list[dict], int] | None:
    """Stage-1 survivors from TradingView, or None if the caller should fall
    back to yfinance. NEVER raises.

    *** CORPORATE-ACTION GUARD — wrong-trade defence ***
    TradingView is trusted here for exactly three things: symbol, price and
    volume. Its PRECOMPUTED INDICATORS (SMA50 / SMA200 / RSI / change) are NOT
    corporate-action-consistent with its own close, and the corruption is
    silent and symbol-dependent. Verified live on 2026-10-06, same request:

        symbol  close    SMA50    SMA200   RSI    change
        CTVA    12.39    77.32    77.88    6.38   +3.94%   <-- inconsistent
        SPY    774.83   764.42   720.95   58.83   +0.67%
        MSFT   525.18   490.59   433.07   66.17   +1.48%
        NVDA   238.90   218.92   201.02   66.90   +2.12%
        AAPL   332.89   322.41   289.45   53.79   -0.24%

    NYSE:CTVA closes at 12.39 but its "SMA50" is 77.32 — the averages were
    computed on pre-adjustment prices, so a screen reading them would see an
    ~84%-below-the-mean, RSI-6 extreme-oversold name that was actually UP 3.9%.
    SPY/MSFT/NVDA/AAPL are self-consistent, which is why this is dangerous:
    it only bites whichever symbols had a corporate action. CTVA is the #1
    survivor of a $1-$20 screen, so this is live, not hypothetical.

    So this function takes symbol/price/volume and nothing else.
    `_stage2_deep_scan` / `_evaluate_wide` remain the ONLY source of signal
    math: they pull real ~1y adjusted OHLCV and compute every indicator
    themselves, and they re-check the price band against that fresh close.

    Belt and braces on top of the provider's own band filter: any row whose
    price falls outside the requested band by more than _BAND_EPS (or that is
    malformed) is dropped and counted into `failed`, so a server that quietly
    ignored part of the filter cannot widen the funnel.
    """
    try:
        from data_engine import tradingview_provider

        result = tradingview_provider.screen(criteria)
        if result is None:
            return None
        raw, provider_failed = result

        survivors: list[dict] = []
        failed = int(provider_failed)
        for row in raw:
            try:
                symbol = row["symbol"]
                price = float(row["price"])
                volume = float(row["volume"])
                if not isinstance(symbol, str) or not symbol or not price > 0 or not volume >= 0:
                    raise ValueError("malformed provider row")
                if criteria.min_price is not None and price < criteria.min_price - _BAND_EPS:
                    raise ValueError("below price band")
                if criteria.max_price is not None and price > criteria.max_price + _BAND_EPS:
                    raise ValueError("above price band")
            except (KeyError, TypeError, ValueError):
                failed += 1
                continue
            survivors.append({"symbol": symbol, "price": price, "volume": volume})

        if not survivors:
            return None
        return survivors, failed
    except Exception as exc:  # noqa: BLE001 — a provider problem must never break the scan
        logger.warning("stage1: tradingview provider error, falling back: %s: %s", type(exc).__name__, exc)
        return None


# TradingView's market size changes slowly; measure it rarely.
_TV_UNIVERSE_CACHE: tuple[float, int] | None = None
_TV_UNIVERSE_TTL = 6 * 3600


def _tradingview_universe_size() -> int | None:
    """How many US stocks TradingView screened, or None if unknown. NEVER raises.

    `provider.screen()` returns only the rows that survived the band, not the
    size of the market it searched, so the figure `stats.universe_size` needs
    is not in the stage-1 result. The only way to get it through the
    provider's public API is an unfiltered screen (no price band, any
    volume): rows returned = stocks examined. That is a multi-page walk
    (~12 requests, a few seconds), so it is measured at most once per
    _TV_UNIVERSE_TTL and best-effort — a failure just means the caller falls
    back to a floored estimate rather than failing the scan.
    """
    global _TV_UNIVERSE_CACHE
    now = time.time()
    if _TV_UNIVERSE_CACHE is not None and now - _TV_UNIVERSE_CACHE[0] < _TV_UNIVERSE_TTL:
        return _TV_UNIVERSE_CACHE[1]
    try:
        from data_engine import tradingview_provider

        result = tradingview_provider.screen(
            FinderCriteria(min_price=None, max_price=None, min_volume=-1)
        )
        if result is None:
            return None
        rows, failed = result
        size = len(rows) + int(failed)
    except Exception as exc:  # noqa: BLE001 — a stat must never break the scan
        logger.info("stage1: tradingview universe size unavailable: %s: %s", type(exc).__name__, exc)
        return None
    if size <= 0:
        return None
    _TV_UNIVERSE_CACHE = (now, size)
    return size


def _stage1_screen(
    universe: list[str],
    criteria: FinderCriteria,
    allow_provider: bool = True,
) -> tuple[list[dict], int, str]:
    """Stage-1 price/volume screen. Returns (survivors, failed_count, source)
    where survivors are [{"symbol", "price", "volume"}, ...] and source is
    "tradingview" or "yfinance" — the path that actually produced them.

    Preferred path: TradingView's server-side screener (one request over the
    whole market, see _tradingview_stage1 for the trust boundary). It ignores
    `universe` by design — it screens the full market in one call — so it is
    skipped when `allow_provider` is False (an explicit caller-supplied
    universe must be honoured, and the provider cannot do that).

    Fallback path (TradingView returned None / empty, or is disallowed): one
    cheap 1-day yfinance bulk download per batch of ~300 tickers over
    `universe`. Filters on price band + volume only (no history, no
    indicators). This loop is unchanged.

    Cached in-process for _STAGE1_CACHE_TTL, keyed primarily by the
    price/volume band (plus a cheap universe fingerprint so two different
    universes with the same band never collide). The cache is consulted
    BEFORE the provider, so both paths are cached; a cached fallback result
    keeps its "yfinance" source label until it expires.
    """
    cache_key = (
        criteria.min_price,
        criteria.max_price,
        criteria.min_volume,
        len(universe),
        hash(tuple(universe)),
        allow_provider,
    )
    now = time.time()
    cached = _STAGE1_CACHE.get(cache_key)
    if cached is not None and now - cached[0] < _STAGE1_CACHE_TTL:
        _, survivors, failed, source = cached
        return survivors, failed, source

    if allow_provider:
        tv = _tradingview_stage1(criteria)
        if tv is not None:
            tv_survivors, tv_failed = tv
            _STAGE1_CACHE[cache_key] = (now, tv_survivors, tv_failed, "tradingview")
            return tv_survivors, tv_failed, "tradingview"
        logger.info(
            "stage1: TradingView unavailable or returned nothing; "
            "falling back to the yfinance batch screen over %d tickers",
            len(universe),
        )

    survivors: list[dict] = []
    failed = 0
    yf_logger, prev_level = _quiet_yfinance()
    try:
        for i in range(0, len(universe), _STAGE1_BATCH):
            batch = universe[i:i + _STAGE1_BATCH]
            try:
                df = yf.download(batch, period="1d", group_by="column", threads=True, progress=False)
            except Exception as exc:
                logger.debug("stage1 batch download failed at offset %d: %s", i, exc)
                failed += len(batch)
                continue

            if df is None or df.empty or "Close" not in df.columns.get_level_values(0):
                failed += len(batch)
                continue

            closes = df["Close"]
            volumes = df["Volume"] if "Volume" in df.columns.get_level_values(0) else None

            for symbol in batch:
                try:
                    if symbol not in closes.columns:
                        failed += 1
                        continue
                    close_series = closes[symbol].dropna()
                    if close_series.empty:
                        failed += 1
                        continue
                    price = float(close_series.iloc[-1])
                    if price <= 0:
                        failed += 1
                        continue
                    volume = 0.0
                    if volumes is not None and symbol in volumes.columns:
                        vol_series = volumes[symbol].dropna()
                        if not vol_series.empty:
                            volume = float(vol_series.iloc[-1])

                    if criteria.min_price is not None and price < criteria.min_price:
                        continue
                    if criteria.max_price is not None and price > criteria.max_price:
                        continue
                    if volume < criteria.min_volume:
                        continue
                    survivors.append({"symbol": symbol, "price": price, "volume": volume})
                except Exception as exc:
                    logger.debug("stage1 parse failed for %s: %s", symbol, exc)
                    failed += 1
                    continue
    finally:
        yf_logger.setLevel(prev_level)

    _STAGE1_CACHE[cache_key] = (now, survivors, failed, "yfinance")
    return survivors, failed, "yfinance"


# ── Stage 2: deep per-symbol OHLCV analysis (survivors only) ──────────────────

_STAGE2_BATCH = 100
_STAGE2_PERIOD = "1y"          # ~250 daily bars
_BREAKOUT_NEAR_PCT = 2.0       # "near" the consolidation high, in percent


def _wide_compression(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Mirror range_engine.compression's range-ratio + percentile scoring,
    computed live from OHLCV (wide mode has no DB-computed ATR/BB-width
    percentiles to read, so they're derived here from the same window
    lengths and the same 40/35/25 weighting, to keep scores comparable
    between the DB path and this one).

    Returns (compression_score, is_compressed, compression_duration) Series
    aligned to df.index.
    """
    high, low, close = df["high"], df["low"], df["close"]

    range_10 = high.rolling(RANGE_SHORT).max() - low.rolling(RANGE_SHORT).min()
    range_30 = high.rolling(RANGE_LONG).max() - low.rolling(RANGE_LONG).min()
    range_ratio = (range_10 / range_30.replace(0, np.nan)).clip(0, 2)

    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr = true_range.rolling(14).mean()
    atr_norm = atr / close.replace(0, np.nan)

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_width = (4 * std20) / sma20.replace(0, np.nan)  # (upper - lower) / middle, 2-sigma bands

    lookback = min(len(df), COMPRESSION_LOOKBACK)
    min_periods = max(10, lookback // 5)

    def _pct_rank(s: pd.Series) -> pd.Series:
        return s.rolling(lookback, min_periods=min_periods).apply(
            lambda x: float(np.sum(x[:-1] <= x[-1]) / max(len(x) - 1, 1)), raw=True
        )

    atr_pct = _pct_rank(atr_norm)
    bb_pct = _pct_rank(bb_width)

    range_score = 1 - range_ratio.clip(0, 1)
    score = (
        (1 - bb_pct.fillna(0.5).clip(0, 1)) * 40
        + (1 - atr_pct.fillna(0.5).clip(0, 1)) * 35
        + range_score.fillna(0) * 25
    ).clip(0, 100)

    trigger_bb = bb_pct.fillna(1.0) < BB_WIDTH_THRESHOLD
    trigger_atr = atr_pct.fillna(1.0) < ATR_THRESHOLD
    trigger_range = range_ratio.fillna(2.0) < 1.0
    triggers_met = trigger_bb.astype(int) + trigger_atr.astype(int) + trigger_range.astype(int)
    is_compressed = triggers_met >= 2

    duration = pd.Series(0, index=df.index)
    consecutive = 0
    for pos, flag in enumerate(is_compressed.tolist()):
        consecutive = consecutive + 1 if flag else 0
        duration.iloc[pos] = consecutive

    return score, is_compressed, duration


def _evaluate_wide(symbol: str, criteria: FinderCriteria, df: pd.DataFrame) -> dict | None:
    """Same contract as _evaluate(), but every check is computed live from a
    single symbol's OHLCV DataFrame (columns: open, high, low, close,
    volume; ascending date index) instead of DB rows."""
    df = df.dropna(subset=["close"])
    if df.empty:
        return None

    price = float(df["close"].iloc[-1])
    volume = float(df["volume"].iloc[-1]) if pd.notna(df["volume"].iloc[-1]) else 0.0
    if price <= 0:
        return None

    checks: dict[str, bool] = {}
    required: list[str] = []

    # 1. price_band
    price_ok = True
    if criteria.min_price is not None:
        price_ok = price_ok and price >= criteria.min_price
    if criteria.max_price is not None:
        price_ok = price_ok and price <= criteria.max_price
    checks["price_band"] = price_ok
    required.append("price_band")

    # 2. liquidity — 20-bar average volume
    vol_window = df["volume"].tail(20).dropna()
    avg_volume_20 = float(vol_window.mean()) if not vol_window.empty else 0.0
    checks["liquidity"] = avg_volume_20 >= criteria.min_volume
    required.append("liquidity")

    # 3. golden_cross — SMA50 > SMA200 and close > SMA50
    sma50 = df["close"].rolling(50).mean()
    sma200 = df["close"].rolling(200).mean()
    ma50 = float(sma50.iloc[-1]) if pd.notna(sma50.iloc[-1]) else None
    ma200 = float(sma200.iloc[-1]) if pd.notna(sma200.iloc[-1]) else None
    golden_cross = ma50 is not None and ma200 is not None and ma50 > ma200 and price > ma50
    checks["golden_cross"] = golden_cross
    if criteria.require_golden_cross:
        required.append("golden_cross")

    # 4. consolidating — mirrors scan()'s compression window logic
    score_series, is_compressed_series, duration_series = _wide_compression(df)
    window = min(criteria.consolidation_lookback, len(df))
    score_window = score_series.tail(window)
    compressed_window = is_compressed_series.tail(window)
    duration_window = duration_series.tail(window)

    compressed_positions = [
        pos for pos in range(len(score_window))
        if compressed_window.iloc[pos] and (score_window.iloc[pos] or 0) >= criteria.min_compression_score
    ]
    consolidating = len(compressed_positions) > 0
    checks["consolidating"] = consolidating
    if criteria.require_consolidation:
        required.append("consolidating")

    if compressed_positions:
        best = max(compressed_positions, key=lambda pos: score_window.iloc[pos] or 0)
        compression_score = float(score_window.iloc[best]) if pd.notna(score_window.iloc[best]) else None
        compression_duration = int(duration_window.iloc[best])
    elif len(score_window) and pd.notna(score_window.iloc[-1]):
        compression_score = float(score_window.iloc[-1])
        compression_duration = int(duration_window.iloc[-1])
    else:
        compression_score = None
        compression_duration = None

    # 5. no_gaps
    gap_window = df.tail(criteria.gap_lookback + 1)
    gaps = []
    opens = gap_window["open"].tolist()
    closes = gap_window["close"].tolist()
    for i in range(1, len(gap_window)):
        prev_close, cur_open = closes[i - 1], opens[i]
        if prev_close and cur_open is not None and pd.notna(prev_close) and pd.notna(cur_open):
            gaps.append(abs(cur_open - prev_close) / prev_close * 100)
    gap_max_pct = max(gaps) if gaps else 0.0
    checks["no_gaps"] = gap_max_pct <= criteria.max_gap_pct
    required.append("no_gaps")

    # 6. breakout_ready — close near/above the recent consolidation high.
    # Always computed (breakout_strength is informational either way); only
    # gates the result when explicitly required.
    prior_high_window = df["high"].iloc[-(RANGE_SHORT + 1):-1]
    recent_high = float(prior_high_window.max()) if not prior_high_window.empty else None
    if recent_high:
        breakout_strength = (price - recent_high) / recent_high * 100
        breakout_ready = price >= recent_high * (1 - _BREAKOUT_NEAR_PCT / 100)
    else:
        breakout_strength = None
        breakout_ready = False
    if criteria.require_breakout_ready:
        checks["breakout_ready"] = breakout_ready
        required.append("breakout_ready")

    if not all(checks[k] for k in required):
        return None

    # confluence is deliberately skipped in wide mode: it is a slow, live,
    # per-symbol call, and running it over hundreds of stage-2 survivors
    # would take hours. Always None here — never gates a wide-mode result.
    passed = sum(1 for v in checks.values() if v)
    failed = len(checks) - passed

    # Same consolidation-derived levels as the fast path. The wide path has no
    # RangeCompression row to read bounds from, so use the rolling window it
    # already computed for the breakout check — recent_high is the same
    # boundary, and the matching low completes the range.
    prior_low_window = df["low"].iloc[-(RANGE_SHORT + 1):-1]
    recent_low = float(prior_low_window.min()) if not prior_low_window.empty else None
    levels = trade_levels(recent_high, recent_low, price)

    return {
        "symbol": symbol,
        "price": price,
        "volume": volume,
        "checks": checks,
        "passed": passed,
        "failed": failed,
        **levels,
        "ma50": ma50,
        "ma200": ma200,
        "avg_volume_20": avg_volume_20,
        "compression_score": compression_score,
        "compression_duration": compression_duration,
        "gap_max_pct": gap_max_pct,
        "breakout_strength": breakout_strength,
        "confluence": None,
    }


def _stage2_deep_scan(candidates: list[dict], criteria: FinderCriteria) -> tuple[list[dict], int]:
    """Deep OHLCV pass over stage-1 survivors, batched. Returns
    (results, failed_count)."""
    results: list[dict] = []
    failed = 0
    yf_logger, prev_level = _quiet_yfinance()
    try:
        for i in range(0, len(candidates), _STAGE2_BATCH):
            batch = candidates[i:i + _STAGE2_BATCH]
            symbols = [c["symbol"] for c in batch]
            try:
                df = yf.download(
                    symbols, period=_STAGE2_PERIOD, group_by="column", threads=True, progress=False,
                )
            except Exception as exc:
                logger.debug("stage2 batch download failed at offset %d: %s", i, exc)
                failed += len(batch)
                continue

            if df is None or df.empty or "Close" not in df.columns.get_level_values(0):
                failed += len(batch)
                continue

            opens = df["Open"] if "Open" in df.columns.get_level_values(0) else None
            highs = df["High"] if "High" in df.columns.get_level_values(0) else None
            lows = df["Low"] if "Low" in df.columns.get_level_values(0) else None
            closes = df["Close"]
            volumes = df["Volume"] if "Volume" in df.columns.get_level_values(0) else None

            for symbol in symbols:
                try:
                    if symbol not in closes.columns:
                        failed += 1
                        continue
                    sym_df = pd.DataFrame({
                        "open": opens[symbol] if opens is not None and symbol in opens.columns else np.nan,
                        "high": highs[symbol] if highs is not None and symbol in highs.columns else np.nan,
                        "low": lows[symbol] if lows is not None and symbol in lows.columns else np.nan,
                        "close": closes[symbol],
                        "volume": volumes[symbol] if volumes is not None and symbol in volumes.columns else np.nan,
                    }).dropna(subset=["close"])
                    if sym_df.empty:
                        failed += 1
                        continue
                    candidate = _evaluate_wide(symbol, criteria, sym_df)
                    if candidate is not None:
                        results.append(candidate)
                except Exception as exc:
                    logger.debug("stage2 evaluate failed for %s: %s", symbol, exc)
                    failed += 1
                    continue
    finally:
        yf_logger.setLevel(prev_level)

    return results, failed


def scan_wide(
    criteria: FinderCriteria,
    limit: int = 100,
    max_deep: int = 400,
    universe: list[str] | None = None,
    max_universe: int = 3000,
) -> dict:
    """
    Live, wide-universe screen — does NOT depend on the precomputed
    indicators/range_compressions/breakout_signals tables that scan() reads,
    so it is the only path that can find sub-$5 / $5-$20 names outside the
    ~234-instrument DB universe.

    Two stages:
      1. Price band + volume screen only, cached in-process ~15 min per
         price/volume band. Preferred source is TradingView's server-side
         screener (data_engine.tradingview_provider): one request screens the
         whole market in seconds, so `max_universe` does NOT apply to it. If
         TradingView is unavailable (or the caller passed an explicit
         `universe`, which it cannot honour) this falls back to a cheap 1-day
         bulk yfinance download over the universe (default: the live
         ~13,786-symbol US ticker list from
         trade_signal_engine.universe._get_all_exchange_tickers), which IS
         capped by `max_universe`. TradingView supplies symbol/price/volume
         only; it never supplies indicators (see _tradingview_stage1).
      2. Deep ~250-bar OHLCV pass, live, over the top `max_deep` stage-1
         survivors ranked by dollar volume (price × volume — most liquid
         first, since illiquid penny names are untradeable regardless of
         setup). Every check scan() would read from the DB is recomputed
         here directly from OHLCV.

    confluence is skipped entirely in wide mode (see _evaluate_wide) — it is
    always None in the returned results and never gates a match, regardless
    of criteria.min_confluence.

    Returns {"results": [...], "stats": {...}}. Each result has the same
    shape scan() returns. `stats` always reports universe_size so a scan's
    true coverage is visible rather than assumed, and `stage1_source`
    ("tradingview" | "yfinance") so a scan never silently misrepresents which
    path produced its candidates.
    """
    start = time.time()

    explicit_universe = universe is not None
    if universe is None:
        from trade_signal_engine.universe import _get_all_exchange_tickers

        universe = _get_all_exchange_tickers()
    universe = list(universe)

    # Cap the interactive sweep. The full US listing is ~13,786 tickers and
    # stage 1 downloads roughly 300 per 4.5s, so an uncapped scan takes ~5
    # minutes — long past the point a browser fetch gives up and far past the
    # point anyone waits after a click. Worse, sweeping the whole listing
    # back-to-back gets the host rate-limited by Yahoo, after which EVERY
    # symbol returns empty and the screen looks broken rather than slow.
    #
    # The cap costs coverage, so it is reported in stats rather than hidden:
    # universe_size is what was actually screened, universe_available is what
    # existed. Pass max_universe=0 for the full sweep when a slow, complete
    # run is genuinely wanted (a scheduled job rather than a click).
    #
    # The cap exists for the yfinance fallback only. The TradingView path
    # screens the whole market server-side in one request, so when it serves
    # stage 1 the cap is moot and is reported as not applied (see stats below).
    universe_available = len(universe)
    if max_universe and universe_available > max_universe:
        universe = universe[:max_universe]

    if criteria.min_confluence > 0:
        logger.warning(
            "scan_wide: min_confluence=%.1f requested but confluence is skipped in wide-scan "
            "mode (live per-symbol calls over hundreds of survivors would take hours); "
            "confluence=None for every result and never gates a match here.",
            criteria.min_confluence,
        )

    # An explicit caller-supplied universe must be honoured, and TradingView
    # cannot screen an arbitrary symbol list — so only the default universe
    # may use the provider.
    stage1_survivors, stage1_failed, stage1_source = _stage1_screen(
        universe, criteria, allow_provider=not explicit_universe,
    )

    if stage1_source == "tradingview":
        # Nothing was capped: TradingView screened its whole US stock market
        # server-side, so `max_universe` did not apply. The ticker listing
        # fetched above belongs to the yfinance fallback — it can be a
        # different size AND a different membership (it excludes OTC and
        # includes ETFs; TradingView's `type == stock` is the reverse), and
        # when the listing source is down it is only a ~361-name static list,
        # which would report a "universe" smaller than the survivor count.
        # So report TradingView's own market size. If that measurement is
        # unavailable, fall back to the listing length but never below the
        # rows TradingView demonstrably returned.
        tv_size = _tradingview_universe_size()
        if tv_size is None:
            tv_size = max(universe_available, len(stage1_survivors) + stage1_failed)
        universe_size = universe_available = tv_size
        universe_capped = False
    else:
        universe_size = len(universe)
        universe_capped = len(universe) < universe_available

    stage1_survivors = sorted(
        stage1_survivors, key=lambda s: s["price"] * s["volume"], reverse=True
    )
    deep_candidates = stage1_survivors[:max_deep]

    results, stage2_failed = _stage2_deep_scan(deep_candidates, criteria)

    results.sort(
        key=lambda r: (r["compression_score"] or 0, r["breakout_strength"] or 0),
        reverse=True,
    )

    stats = {
        # Which stage-1 path produced the candidates. Never omit this: the two
        # paths have different universes (see _tradingview_stage1 / ETF note).
        "stage1_source": stage1_source,
        "universe_size": universe_size,
        # What existed vs what was screened. If these differ the scan was
        # capped, and the caller should say so rather than implying full
        # market coverage.
        "universe_available": universe_available,
        "universe_capped": universe_capped,
        "stage1_survivors": len(stage1_survivors),
        "deep_analyzed": len(deep_candidates),
        "matched": len(results),
        "elapsed_seconds": round(time.time() - start, 1),
        "failed_downloads": stage1_failed + stage2_failed,
    }

    final = results[:limit]
    # Tag the hits with what the learned, validated setups say about them. Bars for these few stocks are
    # fetched in one batched call; if that fails the scan still returns, and says so.
    stats["setup_evidence"] = attach_setup_evidence(final)
    return {"results": final, "stats": stats}


def health() -> dict:
    """Per-table freshness. Makes a stale backfill VISIBLE instead of a silent
    empty scan.

    Two kinds of table, reported differently on purpose:

    "dense" tables (prices, indicators, compressions) carry one row per
    instrument per bar, so "how many instruments are current as of the latest
    date" is a real coverage number — if it drops, ingestion is broken.

    "event" tables (breakout_signals) only get a row when something actually
    happens. Across this universe that is roughly 70 events a MONTH, so on any
    normal day one or two instruments are "current" out of 232. Scoring that
    as coverage made a healthy market look like a 99%-broken pipeline and the
    UI showed a permanent DATA MAY BE STALE warning, which trains you to
    ignore the one warning that matters. Event tables report recent activity
    instead, and no coverage ratio.
    """
    dense_tables = (
        ("price_data", PriceData),
        ("indicators", Indicator),
        ("range_compressions", RangeCompression),
    )
    event_tables = (
        ("breakout_signals", BreakoutSignal),
    )
    result: dict = {}
    with get_session() as session:
        for name, model in dense_tables:
            latest_date = session.execute(select(func.max(model.date))).scalar()
            total_instruments = session.execute(
                select(func.count(func.distinct(model.instrument_id)))
            ).scalar() or 0

            if latest_date is None:
                result[name] = {
                    "kind": "dense",
                    "latest_date": None,
                    "instruments_current": 0,
                    "instruments_total": total_instruments,
                }
                continue

            per_instrument_max = (
                select(model.instrument_id, func.max(model.date).label("max_date"))
                .group_by(model.instrument_id)
                .subquery()
            )
            current_count = session.execute(
                select(func.count()).select_from(per_instrument_max)
                .where(per_instrument_max.c.max_date == latest_date)
            ).scalar() or 0

            result[name] = {
                "kind": "dense",
                "latest_date": str(latest_date),
                "instruments_current": current_count,
                "instruments_total": total_instruments,
            }

        for name, model in event_tables:
            latest_date = session.execute(select(func.max(model.date))).scalar()
            total_instruments = session.execute(
                select(func.count(func.distinct(model.instrument_id)))
            ).scalar() or 0

            cutoff = date.today() - timedelta(days=30)
            events_30d = session.execute(
                select(func.count()).select_from(model).where(model.date >= cutoff)
            ).scalar() or 0
            instruments_30d = session.execute(
                select(func.count(func.distinct(model.instrument_id)))
                .where(model.date >= cutoff)
            ).scalar() or 0

            result[name] = {
                "kind": "event",
                "latest_date": str(latest_date) if latest_date else None,
                "events_30d": events_30d,
                "instruments_30d": instruments_30d,
                "instruments_total": total_instruments,
            }
    return result
