"""Cycle metric endpoints — basic cycle metrics + advanced FFT/wavelet/Hilbert analysis."""

import time
from typing import Any

from fastapi import APIRouter, HTTPException
import pandas as pd
from sqlalchemy import select

from database.connection import get_session
from database.models import Instrument, CycleMetric, PriceData

router = APIRouter()

# ── Advanced analysis cache ───────────────────────────────────────────────────
_adv_cache: dict[str, tuple[dict, float]] = {}
_ADV_TTL = 300.0


def _fetch_price_data(symbol: str) -> pd.DataFrame:
    """Fetch price data from DB with yfinance fallback."""
    with get_session() as session:
        inst = session.execute(
            select(Instrument).where(Instrument.symbol == symbol.upper())
        ).scalar_one_or_none()
        if inst is None:
            try:
                import yfinance as yf
                df = yf.Ticker(symbol.upper()).history(period="2y", interval="1d")
                if df.empty:
                    return pd.DataFrame()
                df = df.rename(columns={"Open": "open", "High": "high", "Low": "low",
                                         "Close": "close", "Volume": "volume"})
                df.index.name = "date"
                df = df.reset_index()
                df["date"] = df["date"].dt.strftime("%Y-%m-%d")
                return df[["date", "open", "high", "low", "close", "volume"]].tail(500).reset_index(drop=True)
            except Exception:
                return pd.DataFrame()

        rows = session.execute(
            select(PriceData.date, PriceData.open, PriceData.high,
                   PriceData.low, PriceData.close, PriceData.volume)
            .where(PriceData.instrument_id == inst.id)
            .order_by(PriceData.date.desc())
            .limit(500)
        ).all()

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
    df["date"] = df["date"].astype(str)
    return df.sort_values("date").reset_index(drop=True)


@router.get("/heatmap/latest")
def get_cycle_heatmap():
    """Latest cycle data across all instruments for heatmap view."""
    session = get_session()
    try:
        instruments = session.execute(
            select(Instrument).order_by(Instrument.symbol)
        ).scalars().all()

        result = []
        for inst in instruments:
            latest = session.execute(
                select(CycleMetric)
                .where(CycleMetric.instrument_id == inst.id)
                .order_by(CycleMetric.date.desc())
                .limit(1)
            ).scalar()

            if latest:
                result.append({
                    "symbol": inst.symbol,
                    "date": str(latest.date),
                    "cycle_length": latest.cycle_length,
                    "cycle_phase": latest.cycle_phase,
                    "cycle_strength": latest.cycle_strength,
                })

        return result
    finally:
        session.close()


@router.get("/{symbol}")
def get_cycles(symbol: str, limit: int = 500):
    """Cycle metrics for a symbol."""
    session = get_session()
    try:
        inst_id = session.execute(
            select(Instrument.id).where(Instrument.symbol == symbol.upper())
        ).scalar()
        if inst_id is None:
            raise HTTPException(404, f"Instrument not found: {symbol}")

        df = pd.read_sql(
            select(CycleMetric.date, CycleMetric.cycle_length,
                   CycleMetric.cycle_phase, CycleMetric.cycle_strength)
            .where(CycleMetric.instrument_id == inst_id)
            .order_by(CycleMetric.date),
            session.bind,
        )
        if df.empty:
            return []
        df = df.tail(limit)
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        return df.where(df.notna(), None).to_dict(orient="records")
    finally:
        session.close()


@router.get("/{symbol}/advanced")
def get_cycle_advanced(symbol: str) -> dict[str, Any]:
    """Advanced cycle analysis — FFT, wavelet, Hilbert phase, and projection.

    Combines multiple spectral methods to find dominant cycles,
    compute instantaneous phase, and project the next peak/trough.
    Results are cached for 5 minutes.
    """
    sym = symbol.upper()
    now = time.time()

    cached, ts = _adv_cache.get(sym, (None, 0.0))
    if cached is not None and (now - ts) < _ADV_TTL:
        return cached

    df = _fetch_price_data(sym)
    if df.empty:
        return {"symbol": sym, "error": "No price data available"}

    from cycle_engine.cycle_projection import project_cycle
    result = project_cycle(df)
    result["symbol"] = sym

    _adv_cache[sym] = (result, now)
    return result
