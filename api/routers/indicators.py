"""Indicator endpoints."""

from fastapi import APIRouter, HTTPException

from data_engine.loader import DataLoader

router = APIRouter()
loader = DataLoader()


@router.get("/{symbol}")
def get_indicators(symbol: str, limit: int = 500):
    """Technical indicators for a symbol."""
    df = loader.get_indicators(symbol.upper())
    if df.empty:
        raise HTTPException(404, f"No indicators for {symbol}")
    df = df.tail(limit).reset_index()
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df.where(df.notna(), None).to_dict(orient="records")
