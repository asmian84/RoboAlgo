"""
Stage 1 — daily OHLCV ingestion into `price_data`.

This is the stage the Celery beat schedule calls first every weekday, and the
one every downstream stage depends on. It was referenced by
`infrastructure/celery_app.py`'s stage_map but the module never existed, so
`run_pipeline_stage(1)` raised ImportError on every scheduled run and price
data silently stopped updating.

Ingestion itself is not reimplemented here — `data_engine.downloader.MarketDataDownloader`
already handles source fallback (Finnhub → yfinance) and upserts rows via
ON CONFLICT, so re-running is idempotent. This stage adds the thing a scheduled
job needs and the downloader lacks: an incremental window.

`MarketDataDownloader` always pulls from DATA_START_DATE (2010-01-01). Doing that
nightly for every instrument means re-downloading 16 years of bars to learn one
new row. Instead we look at what is already stored and fetch only from shortly
before the newest bar, so a healthy daily run is cheap and a long outage
backfills automatically without anyone picking a date by hand.

Usage:
    from pipeline.stage1_prices import run
    run()                      # incremental — normal scheduled call
    run(symbol="AAPL")         # one instrument
    run(full=True)             # ignore stored state, re-pull from DATA_START_DATE
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

from sqlalchemy import func, select

from database.connection import get_session
from database.models import Instrument, PriceData

logger = logging.getLogger(__name__)

# Re-fetch a few days behind the newest stored bar on every run. Exchanges
# revise late prints and split/dividend adjustments land retroactively; the
# downloader upserts, so overlapping rows are corrections rather than duplicates.
_OVERLAP_DAYS = 5

# Below this many stored bars an instrument is treated as new and given full
# history rather than an incremental window.
_MIN_BARS_FOR_INCREMENTAL = 20

# Fraction of instruments allowed to be older than the chosen start date.
#
# The naive choice is the OLDEST last-bar date, so no symbol can gap. But a
# universe always accumulates tickers that are delisted, renamed or acquired
# (SQ -> XYZ, CFLT, SMAR ...). Those can never advance, so with the oldest date
# they pin the window to the day they died and every nightly run re-downloads
# months of data for all 235 symbols — forever, while still looking healthy.
#
# Taking a low percentile instead tolerates a minority of dead tickers while
# still behaving correctly during a real outage: if ingestion genuinely stops,
# EVERY symbol shares the same stale date, so the percentile and the minimum
# are the same value and the full backfill happens as intended. The symbols
# excluded are, by construction, the ones with no new data to miss.
_STALE_TOLERANCE = 0.10


def _incremental_start(session, symbols: list[str]) -> str | None:
    """Earliest date we must re-fetch from to leave no gap.

    Returns the OLDEST last-bar date across the instruments being ingested,
    minus the overlap. Oldest rather than newest on purpose: the downloader
    takes a single start date for the whole run, so anchoring on the newest
    symbol would skip rows for every symbol that lags it.

    `symbols` must be the exact set this run will download — never "everything
    in the database". The instruments table outlives the configured universe
    (delisted or dropped tickers keep their rows), and those orphans are never
    refreshed because the downloader only fetches the configured list. Letting
    one permanently-stale orphan into this calculation would pin the start date
    to its last bar forever, quietly turning every nightly run into a
    months-long re-download that still looks like it is working.

    Returns None when there is nothing usable to be incremental from, meaning
    the caller should fall back to full history.
    """
    if not symbols:
        return None

    stmt = (
        select(Instrument.symbol, func.max(PriceData.date), func.count(PriceData.id))
        .join(PriceData, PriceData.instrument_id == Instrument.id)
        .where(Instrument.symbol.in_([s.upper() for s in symbols]))
        .group_by(Instrument.symbol)
    )

    rows = session.execute(stmt).all()
    if not rows:
        return None

    # Ignore thinly-populated instruments — they need a full pull, and letting
    # one of them set the anchor would drag every other symbol back to 2010.
    last_dates = sorted(
        r[1] for r in rows if r[1] is not None and r[2] >= _MIN_BARS_FOR_INCREMENTAL
    )
    if not last_dates:
        return None

    # Oldest date that still covers (1 - _STALE_TOLERANCE) of the universe.
    # Index 0 is the plain minimum, which is what a small universe collapses to.
    idx = int(len(last_dates) * _STALE_TOLERANCE)
    anchor = last_dates[min(idx, len(last_dates) - 1)]

    return (anchor - timedelta(days=_OVERLAP_DAYS)).isoformat()


def run(symbol: str | None = None, full: bool = False, **_kwargs) -> int:
    """Ingest daily OHLCV. Returns the number of instruments successfully stored.

    Args:
        symbol: Restrict to one instrument. None processes the configured universe.
        full:   Ignore what is already stored and re-pull from DATA_START_DATE.

    Never raises on a per-symbol failure — the downloader counts those and the
    run continues, so one delisted ticker cannot stop the nightly ingest.
    """
    from data_engine.downloader import MarketDataDownloader

    downloader = MarketDataDownloader()
    # download_all(None) falls back to the configured universe, so resolve it
    # here too — the incremental window must be computed over exactly the
    # symbols this run will fetch, nothing wider.
    symbols = [symbol.upper()] if symbol else None
    target = symbols if symbols is not None else list(downloader.instruments)

    session = get_session()
    try:
        start = None if full else _incremental_start(session, target)
    finally:
        session.close()

    if start:
        downloader.start_date = start
        logger.info(
            "Stage 1: incremental ingest from %s (%s)",
            start, symbol or f"{len(downloader.instruments)} instruments",
        )
    else:
        logger.info(
            "Stage 1: full ingest from %s (%s)",
            downloader.start_date, symbol or f"{len(downloader.instruments)} instruments",
        )

    result = downloader.download_all(symbols)
    stored, failed = result.get("success", 0), result.get("failed", 0)

    if failed:
        logger.warning("Stage 1: %d instruments stored, %d failed", stored, failed)
    else:
        logger.info("Stage 1: %d instruments stored", stored)

    return stored


def freshness(limit: int = 10) -> dict:
    """Report how current `price_data` is — for health checks and debugging.

    Returns the newest stored bar, how many days back that is, and the most
    stale instruments, so a stalled ingest is visible without opening a DB shell.
    """
    session = get_session()
    try:
        rows = session.execute(
            select(Instrument.symbol, func.max(PriceData.date))
            .join(PriceData, PriceData.instrument_id == Instrument.id)
            .group_by(Instrument.symbol)
            .order_by(func.max(PriceData.date).asc())
        ).all()
    finally:
        session.close()

    if not rows:
        return {"instruments": 0, "newest": None, "age_days": None, "stalest": []}

    newest = max(r[1] for r in rows if r[1] is not None)
    return {
        "instruments": len(rows),
        "newest": newest.isoformat(),
        "age_days": (date.today() - newest).days,
        "stalest": [
            {"symbol": s, "last": d.isoformat()} for s, d in rows[:limit] if d is not None
        ],
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    import json
    print("Before:", json.dumps(freshness(5), indent=2))
    count = run()
    print(f"Stored {count} instruments")
    print("After: ", json.dumps(freshness(5), indent=2))
