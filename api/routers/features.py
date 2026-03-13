"""Feature endpoints."""

from fastapi import APIRouter, HTTPException
import pandas as pd
from sqlalchemy import select, func

from database.connection import get_session
from database.models import Instrument, Feature

router = APIRouter()

FEATURE_COLS = [
    "trend_strength", "momentum", "volatility_percentile", "volume_ratio",
    "cycle_phase", "macd_norm", "bb_position", "price_to_ma50",
    "return_5d", "return_20d",
]


@router.get("/matrix/latest")
def get_feature_matrix():
    """Latest feature values across all instruments."""
    session = get_session()
    try:
        instruments = session.execute(
            select(Instrument).order_by(Instrument.symbol)
        ).scalars().all()

        result = []
        for inst in instruments:
            latest = session.execute(
                select(Feature)
                .where(Feature.instrument_id == inst.id)
                .order_by(Feature.date.desc())
                .limit(1)
            ).scalar()

            if latest:
                result.append({
                    "symbol": inst.symbol,
                    "date": str(latest.date),
                    "trend_strength": latest.trend_strength,
                    "momentum": latest.momentum,
                    "volatility_percentile": latest.volatility_percentile,
                    "volume_ratio": latest.volume_ratio,
                    "cycle_phase": latest.cycle_phase,
                    "macd_norm": latest.macd_norm,
                    "bb_position": latest.bb_position,
                    "price_to_ma50": latest.price_to_ma50,
                    "return_5d": latest.return_5d,
                    "return_20d": latest.return_20d,
                })

        return result
    finally:
        session.close()


@router.get("/{symbol}")
def get_features(symbol: str, limit: int = 500):
    """Feature vectors for a symbol."""
    session = get_session()
    try:
        inst_id = session.execute(
            select(Instrument.id).where(Instrument.symbol == symbol.upper())
        ).scalar()
        if inst_id is None:
            raise HTTPException(404, f"Instrument not found: {symbol}")

        df = pd.read_sql(
            select(Feature.date, *[getattr(Feature, c) for c in FEATURE_COLS])
            .where(Feature.instrument_id == inst_id)
            .order_by(Feature.date),
            session.bind,
        )
        if df.empty:
            return []
        df = df.tail(limit)
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        return df.where(df.notna(), None).to_dict(orient="records")
    finally:
        session.close()
