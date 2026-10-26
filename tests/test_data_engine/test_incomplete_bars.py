"""A bar with no prices must never reach price_data.

Regression for 2026-10-06: yfinance returned a row with a volume but NaN open/high/low/close for
115 of 230 instruments. It was stored, every downstream stage silently skipped those instruments,
and the Finder lost half its universe while the freshness check (which only reads dates) stayed
green. The upsert also does DO UPDATE, so such a row could overwrite a good stored bar.
"""
import numpy as np
import pandas as pd
import pytest

import data_engine.downloader as dl


def _bars():
    return pd.DataFrame({
        "date": pd.to_datetime(["2026-10-01", "2026-10-02", "2026-10-05", "2026-10-06"]),
        "open": [10.0, 10.1, 10.2, np.nan],
        "high": [10.5, 10.6, 10.7, np.nan],
        "low": [9.9, 10.0, 10.1, np.nan],
        "close": [10.4, 10.5, 10.6, np.nan],
        "volume": [1e6, 1e6, 1e6, 3.9e6],          # the partial row still has a volume
    })


def test_a_row_with_volume_but_no_prices_is_dropped_and_counted():
    clean, dropped = dl.drop_incomplete_bars(_bars())
    assert dropped == 1 and len(clean) == 3
    assert clean["date"].max() == pd.Timestamp("2026-10-05")


@pytest.mark.parametrize("field", ["open", "high", "low", "close"])
def test_any_single_missing_price_field_disqualifies_the_bar(field):
    df = _bars().iloc[:3].copy()
    df.loc[1, field] = np.nan
    clean, dropped = dl.drop_incomplete_bars(df)
    assert dropped == 1 and list(clean["date"]) == [pd.Timestamp("2026-10-01"), pd.Timestamp("2026-10-05")]


def test_a_missing_volume_alone_is_not_disqualifying():
    df = _bars().iloc[:3].copy()
    df.loc[1, "volume"] = np.nan
    clean, dropped = dl.drop_incomplete_bars(df)
    assert dropped == 0 and len(clean) == 3


def test_complete_data_passes_through_untouched():
    df = _bars().iloc[:3]
    clean, dropped = dl.drop_incomplete_bars(df)
    assert dropped == 0 and clean.equals(df)


class _Capture:
    """Stands in for sqlalchemy's pg_insert: records what would have been written."""
    written = None

    def __init__(self, *_):
        pass

    def values(self, records):
        type(self).written = records
        self.excluded = type("E", (), {k: None for k in ("open", "high", "low", "close", "volume")})()
        return self

    def on_conflict_do_update(self, **_):
        return self


class _Session:
    def execute(self, _):
        pass

    def commit(self):
        pass


def test_the_store_path_never_writes_the_partial_row(monkeypatch):
    monkeypatch.setattr(dl, "pg_insert", _Capture)
    _Capture.written = None
    downloader = dl.MarketDataDownloader.__new__(dl.MarketDataDownloader)      # skip __init__: no network
    downloader._store_prices(_Session(), 7, _bars())
    assert [r["date"] for r in _Capture.written] == list(pd.to_datetime(["2026-10-01", "2026-10-02", "2026-10-05"]))
    assert all(r["close"] is not None for r in _Capture.written)


def test_an_all_partial_batch_writes_nothing(monkeypatch):
    monkeypatch.setattr(dl, "pg_insert", _Capture)
    _Capture.written = "untouched"
    df = _bars().iloc[3:]
    dl.MarketDataDownloader.__new__(dl.MarketDataDownloader)._store_prices(_Session(), 7, df)
    assert _Capture.written == "untouched"
