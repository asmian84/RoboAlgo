"""
RoboAlgo — Price Distribution Forecast API

Endpoints for forward price distribution forecasts combining
quantile regression and Monte Carlo simulation.
"""

from fastapi import APIRouter

router = APIRouter()

_cache: dict[str, tuple[dict, float]] = {}
_TTL = 300.0


@router.get("/{symbol}")
def get_price_distribution(symbol: str, horizon: int = 20):
    """Price distribution forecast for a symbol."""
    import time

    sym = symbol.upper()
    cache_key = f"{sym}_{horizon}"
    now = time.time()

    cached, ts = _cache.get(cache_key, (None, 0.0))
    if cached is not None and (now - ts) < _TTL:
        return cached

    from api.routers.cycles import _fetch_price_data

    df = _fetch_price_data(sym)
    if df.empty:
        return {"symbol": sym, "error": "No price data"}

    from distribution_engine.range_probability import forecast_price_distribution

    result = forecast_price_distribution(df, horizon)
    result["symbol"] = sym

    _cache[cache_key] = (result, now)
    return result
