"""The delisted-data ingest must not corrupt the survivor cache or let a dead stock's outcome vanish."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Parquet is a research-script dependency, not part of the deployed app (requirements.txt omits it on purpose).
# Without it these tests skip with a reason instead of erroring.
pytest.importorskip("pyarrow", reason="the ingest and dataset builder read/write parquet; pip install pyarrow")

ROOT = Path(__file__).resolve().parents[2]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _bars(symbol, n=900, start="2014-01-02", seed=0, fall_after=None):
    rng = np.random.default_rng(seed)
    close = 20 * np.exp(np.cumsum(rng.normal(0.0003, 0.02, n)))
    if fall_after:
        close[fall_after:] *= 0.1
    df = pd.DataFrame({"symbol": symbol, "date": pd.bdate_range(start, periods=n), "open": close * 1.001,
                       "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 3e6})
    return df


@pytest.fixture
def world(tmp_path):
    surv = tmp_path / "surv"
    surv.mkdir()
    pd.concat([_bars("AAA", seed=1), _bars("ZZZ", seed=2)]).to_parquet(surv / "batch_0000.parquet", index=False)
    return tmp_path, surv


def test_a_vendor_file_is_ingested_collisions_are_skipped_and_survivors_untouched(world):
    tmp, surv = world
    vendor = tmp / "vendor"
    vendor.mkdir()
    dead = _bars("DEAD", seed=3, fall_after=700).drop(columns="symbol").rename(columns={"date": "Date"})
    dead.to_csv(vendor / "DEAD.csv", index=False)                                  # one file per symbol, symbol from the name
    clash = _bars("AAA", seed=9)                                                    # a DIFFERENT company that reused AAA
    short = _bars("TINY", n=50, seed=4)
    pd.concat([clash, short]).rename(columns={"date": "timestamp"}).to_csv(vendor / "combined.csv", index=False)
    before = (surv / "batch_0000.parquet").read_bytes()

    ing = _load("ingest_delisted_history")
    into = tmp / "all"
    assert ing.main(["--source", str(vendor), "--survivors", str(surv), "--into", str(into)]) == 0

    assert (surv / "batch_0000.parquet").read_bytes() == before                     # survivors never modified
    assert (into / "batch_0000.parquet").is_symlink()                               # linked, not copied
    out = pd.read_parquet(into / "batch_delisted_0000.parquet")
    assert set(out["symbol"]) == {"DEAD"}                                           # AAA collided, TINY too short
    assert json.loads((into / "terminal.json").read_text()) == ["DEAD"]
    man = json.loads((into / "manifest_delisted.json").read_text())
    assert man["skipped_ticker_collisions"] == ["AAA"] and man["skipped_too_short"] == ["TINY"]


def test_incomplete_rows_and_alternative_column_names_are_handled(world):
    tmp, surv = world
    d = _bars("OLD", seed=5)
    d = d.rename(columns={"symbol": "Ticker", "date": "t", "open": "o", "high": "h", "low": "l", "close": "c", "volume": "v"})
    d["t"] = d["t"].astype("int64") // 10**6                                        # epoch milliseconds, Polygon-style
    d.loc[10:14, ["o", "h", "l", "c"]] = np.nan
    f = tmp / "poly.parquet"
    d.to_parquet(f, index=False)
    ing = _load("ingest_delisted_history")
    into = tmp / "all2"
    ing.main(["--source", str(f), "--survivors", str(surv), "--into", str(into)])
    out = pd.read_parquet(into / "batch_delisted_0000.parquet")
    assert len(out) == 900 - 5 and out["date"].min() >= pd.Timestamp("2014-01-01")
    assert out["close"].notna().all()


def test_a_row_with_good_prices_but_no_date_is_dropped(world):
    # The price filters cannot catch this row (its prices are valid); only the dropna on `date` does.
    tmp, surv = world
    d = _bars("GONE", seed=9)
    d["date"] = d["date"].dt.strftime("%Y-%m-%d")
    d.loc[20, "date"] = ""
    f = tmp / "gone.csv"
    d.to_csv(f, index=False)
    ing = _load("ingest_delisted_history")
    into = tmp / "all_nodate"
    ing.main(["--source", str(f), "--survivors", str(surv), "--into", str(into)])
    out = pd.read_parquet(into / "batch_delisted_0000.parquet")
    assert len(out) == 900 - 1
    assert out["date"].notna().all()


def test_a_missing_required_column_stops_with_a_clear_message(world):
    tmp, surv = world
    bad = tmp / "bad.csv"
    pd.DataFrame({"date": ["2020-01-01"], "close": [1.0]}).to_csv(bad, index=False)
    ing = _load("ingest_delisted_history")
    with pytest.raises(SystemExit, match="missing columns"):
        ing.main(["--source", str(bad), "--survivors", str(surv), "--into", str(tmp / "all3")])


def test_the_dataset_builder_keeps_a_dead_stocks_collapse_instead_of_dropping_it(world):
    tmp, surv = world
    vendor = tmp / "vendor"
    vendor.mkdir()
    _bars("DEAD", seed=3, fall_after=700).to_csv(vendor / "DEAD.csv", index=False)
    ing = _load("ingest_delisted_history")
    into = tmp / "all"
    ing.main(["--source", str(vendor), "--survivors", str(surv), "--into", str(into)])
    build = _load("build_setup_dataset")
    from setup_engine import features as ft
    obs_dates = ft.observation_dates("2015-06-01", "2017-06-30")
    path = into / "batch_delisted_0000.parquet"
    rows_dead, _, _ = build._batch((path, obs_dates, frozenset({"DEAD"})))
    rows_surv, _, _ = build._batch((path, obs_dates, frozenset()))
    assert rows_dead["fwd_ret_6m"].notna().sum() > rows_surv["fwd_ret_6m"].notna().sum()
    assert (rows_dead["fwd_ret_6m"] < -0.5).any()                                    # the collapse is in the data
