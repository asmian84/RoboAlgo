"""The replay must reproduce what the production engines actually stored.

Skipped when the database is unreachable or empty. Compares recomputed
features against the `indicators` and `range_compressions` tables for real
instruments over their recent history.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.filterwarnings("ignore")


def _engine():
    try:
        from sqlalchemy import text
        from database.connection import get_engine
        eng = get_engine()
        with eng.connect() as c:
            c.execute(text("select 1"))
        return eng
    except Exception as exc:                       # no database in this environment
        pytest.skip(f"database unavailable: {type(exc).__name__}")


@pytest.mark.parametrize("symbol", ["SPY", "AAPL", "XOM"])
def test_features_match_the_stored_engine_output(symbol):
    from sqlalchemy import text
    from strategy_validation.finder_backtest import compute_features

    eng = _engine()
    px = pd.read_sql(text("select p.date, p.open, p.high, p.low, p.close, p.volume from price_data p "
                          "join instruments i on i.id=p.instrument_id where i.symbol=:s order by p.date"),
                     eng, params={"s": symbol}, index_col="date", parse_dates=["date"])
    if len(px) < 600:
        pytest.skip(f"{symbol} has too little history in this database")
    ind = pd.read_sql(text("select x.date, x.ma50, x.ma200 from indicators x join instruments i "
                           "on i.id=x.instrument_id where i.symbol=:s order by x.date"),
                      eng, params={"s": symbol}, index_col="date", parse_dates=["date"])
    comp = pd.read_sql(text("select r.date, r.is_compressed, r.compression_score, r.range_high, r.range_low "
                            "from range_compressions r join instruments i on i.id=r.instrument_id "
                            "where i.symbol=:s order by r.date"),
                       eng, params={"s": symbol}, index_col="date", parse_dates=["date"])
    f = compute_features(px)
    mine = pd.DataFrame({"ma50": f.ma50, "ma200": f.ma200, "score": f.score, "is_comp": f.is_comp,
                         "rh": f.range_high, "rl": f.range_low, "valid": f.valid},
                        index=pd.DatetimeIndex(f.dates))

    j = mine.join(ind, rsuffix="_db", how="inner").dropna(subset=["ma50", "ma50_db"])
    assert len(j) > 300
    assert np.allclose(j.ma50, j.ma50_db, rtol=1e-9) and np.allclose(j.ma200.dropna(), j.ma200_db.dropna(), rtol=1e-9)

    c = mine[mine.valid].join(comp, how="inner").iloc[-1000:]
    assert len(c) > 300
    assert (c.is_comp == c.is_compressed.astype(bool)).mean() == 1.0
    assert np.allclose(c.score, c.compression_score.astype(float), atol=0.011)
    assert np.allclose(c.rh, c.range_high.astype(float), atol=1e-4)
    assert np.allclose(c.rl, c.range_low.astype(float), atol=1e-4)
