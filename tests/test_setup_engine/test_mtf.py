"""setup_engine.mtf: a weekly or monthly bar must never be used before it is complete.

The failure being guarded against: a multi-timeframe screen that looks good because a weekly or
monthly column quietly contains information from after the day it is evaluated on. The classic
leaks: the in-progress week counted as a finished bar; a daily row reading its OWN (unfinished)
week instead of the last completed one; a week carrying the month that CONTAINS it (a month that has
not ended yet) instead of the last completed month; a "prior 26-week high" or "prior 10-week
volume" that includes the bar being tested; a benchmark whose later bars leak into the stock's
relative-strength columns. Each leak is built as a trap here and mutation-tested. The calendar
rules (Friday weeks, last-weekday months, holiday weeks becoming usable one day late) are pinned
on dates where they matter (Good Friday 2024, a month ending on a Sunday, Memorial Day 2021).
No test touches the network.
"""
import numpy as np
import pandas as pd
import pytest

from setup_engine import mtf
from setup_engine.mtf import (MTF_COLUMNS, SIGNATURES, add_xs_ranks, best_run, ensure_benchmarks,
                              indicators_mtf, is_liquid, load_benchmark, map_universe, month_label,
                              monthly_bars, mtf_features, week_label, weekly_bars)
from setup_engine.screen import indicators

DATE_COLS = ["w_last_day", "m_label"]


def _daily(n=1800, seed=2, start="2014-01-02"):
    rng = np.random.default_rng(seed)
    close = 20 * np.exp(np.cumsum(rng.normal(0.0004, 0.02, n)))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.008, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.008, n)))
    vol = rng.integers(500_000, 3_000_000, n).astype(float)
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx)


def _from_weekly(closes, volumes=None, start="2020-01-06"):
    """Five identical business-day bars per week (o = h = l = c = the week's close); volume is per DAY."""
    closes = np.asarray(closes, dtype=float)
    vols = np.full(len(closes), 100.0) if volumes is None else np.asarray(volumes, dtype=float)
    idx = pd.bdate_range(start, periods=5 * len(closes))
    c = np.repeat(closes, 5)
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": np.repeat(vols, 5)}, index=idx)


def _ts(s):
    return pd.Timestamp(s)


def _same(a, b, msg=""):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    assert a.shape == b.shape, msg
    assert np.allclose(a, b, rtol=1e-12, equal_nan=True), msg


@pytest.fixture(scope="module")
def pair():
    return _daily(1500, seed=2), _daily(1500, seed=7)


@pytest.fixture(scope="module")
def full(pair):
    d, b = pair
    return mtf_features(d, b)


# ── (a) calendar labels ───────────────────────────────────────────────────────

def test_week_label_is_the_friday_ending_the_week():
    assert list(week_label(["2024-03-28", "2022-10-14"])) == [_ts("2024-03-29"), _ts("2022-10-14")]
    # the weekend belongs to the NEXT week
    assert list(week_label(["2024-03-30", "2024-03-31", "2024-03-25"])) == [_ts("2024-04-05"), _ts("2024-04-05"),
                                                                            _ts("2024-03-29")]
    assert week_label(_ts("2024-03-28"))[0] == _ts("2024-03-29")      # a scalar works too


def test_month_label_is_the_last_weekday_of_the_month():
    assert month_label(["2024-03-05"])[0] == _ts("2024-03-29")        # 31 March 2024 is a Sunday
    assert month_label(["2021-05-10"])[0] == _ts("2021-05-31")        # a Monday
    assert month_label(["2022-04-04"])[0] == _ts("2022-04-29")        # 30 April 2022 is a Saturday
    assert month_label(["2024-02-14"])[0] == _ts("2024-02-29")        # a Thursday


# ── (b) weekly bars: aggregation and completeness ─────────────────────────────

def _hand_daily():
    idx = pd.bdate_range("2024-04-01", "2024-04-12")                  # Mon 1 April .. Fri 12 April
    i = np.arange(len(idx))
    o = 10.0 + i
    return pd.DataFrame({"open": o, "high": o + 2, "low": o - 1, "close": o + 0.5, "volume": 100.0 * (i + 1)},
                        index=idx)


def test_weekly_bars_aggregate_by_hand():
    d = _hand_daily()
    wb = weekly_bars(d)
    assert list(wb.index) == [_ts("2024-04-05"), _ts("2024-04-12")]
    assert wb.index.name == "week"
    w1, w2 = wb.iloc[0], wb.iloc[1]
    o, h, l, c, v = (d[k].tolist() for k in ("open", "high", "low", "close", "volume"))
    assert w1["open"] == o[0] and w1["close"] == c[4]
    assert w1["high"] == max(h[0:5]) and w1["low"] == min(l[0:5])
    assert w1["volume"] == sum(v[0:5]) == 1500.0
    assert w1["dvol"] == pytest.approx(sum(ci * vi for ci, vi in zip(c[0:5], v[0:5])))
    assert w1["n_days"] == 5 and w1["last_day"] == _ts("2024-04-05")
    assert w2["open"] == o[5] and w2["close"] == c[9]
    assert w2["high"] == max(h[5:10]) and w2["low"] == min(l[5:10])
    assert w2["volume"] == sum(v[5:10]) == 4000.0
    assert w2["dvol"] == pytest.approx(sum(ci * vi for ci, vi in zip(c[5:10], v[5:10])))
    assert w2["n_days"] == 5 and w2["last_day"] == _ts("2024-04-12")


def test_weekly_bars_never_include_the_week_in_progress():
    d = _hand_daily()
    assert list(weekly_bars(d.iloc[:8]).index) == [_ts("2024-04-05")]     # data ends Wednesday 10 April
    assert list(weekly_bars(d.iloc[:9]).index) == [_ts("2024-04-05")]     # Thursday
    assert list(weekly_bars(d.iloc[:10]).index) == [_ts("2024-04-05"), _ts("2024-04-12")]   # Friday
    assert list(weekly_bars(d, as_of="2024-04-10").index) == [_ts("2024-04-05")]
    # rows after as_of are dropped BEFORE aggregating: the week-1 bar is the same either way
    assert weekly_bars(d, as_of="2024-04-10").iloc[0].equals(weekly_bars(d).iloc[0])


def test_monthly_bars_label_by_last_weekday_and_exclude_the_month_in_progress():
    d = _daily(160, start="2024-01-01")                                   # through ~mid-August
    mb = monthly_bars(d)
    assert mb.index.name == "month"
    assert _ts("2024-03-29") in mb.index and _ts("2024-03-31") not in mb.index
    assert mb.index[-1] <= d.index[-1]
    assert mb.index[-1] == _ts("2024-07-31")
    mar = d.loc["2024-03-01":"2024-03-31"]
    assert mb.loc["2024-03-29", "close"] == mar["close"].iloc[-1]
    assert mb.loc["2024-03-29", "n_days"] == len(mar)


def test_bars_reject_a_bad_index_or_missing_columns():
    d = _daily(50)
    with pytest.raises(ValueError):
        weekly_bars(pd.concat([d, d.iloc[:3]]).sort_index())            # duplicate dates
    with pytest.raises(ValueError):
        weekly_bars(d.iloc[::-1])                                       # decreasing
    with pytest.raises(ValueError):
        monthly_bars(d.drop(columns="volume"))


# ── (c) holidays ──────────────────────────────────────────────────────────────

def test_a_holiday_friday_week_is_usable_only_from_the_next_trading_day():
    d = _daily(60, start="2024-03-01")                                    # covers March and April 2024
    d = d.drop(index=_ts("2024-03-29"))                                   # Good Friday
    wb = weekly_bars(d)
    w = wb.loc["2024-03-29"]
    assert w["n_days"] == 4 and w["last_day"] == _ts("2024-03-28")
    # data ends on the Thursday: the week is not complete (its Friday label is still in the future)
    assert _ts("2024-03-29") not in weekly_bars(d.loc[:"2024-03-28"]).index
    assert weekly_bars(d.loc[:"2024-03-28"]).index[-1] == _ts("2024-03-22")
    # through the next trading day it is
    assert _ts("2024-03-29") in weekly_bars(d.loc[:"2024-04-01"]).index
    assert _ts("2024-03-29") in weekly_bars(d.loc[:"2024-04-02"]).index


# ── (d) truncation invariance of the weekly table ─────────────────────────────

def test_column_list_matches_the_contract(full):
    assert list(full.columns) == list(MTF_COLUMNS) + DATE_COLS
    assert len(set(MTF_COLUMNS)) == len(MTF_COLUMNS) == 45
    assert not set(DATE_COLS) & set(MTF_COLUMNS)
    assert full.index.name == "week"
    assert all(full[c].dtype == np.float64 for c in MTF_COLUMNS)
    assert not np.isinf(full[list(MTF_COLUMNS)].to_numpy()).any()
    assert full.index.equals(week_label(full.index))                    # every row label is a Friday


@pytest.mark.parametrize("cut", [600, 601, 900, 1300])                  # 600, 900, 1300 are Thursdays; 601 a Friday
def test_weekly_table_ignores_the_future(pair, full, cut):
    d, b = pair
    part = mtf_features(d.iloc[: cut + 1], b)                           # bench passed FULL both times
    want = full[full.index <= d.index[cut]]
    assert part.index.equals(want.index)
    assert list(part.columns) == list(MTF_COLUMNS) + DATE_COLS
    assert part.index[-1] <= d.index[cut]
    for col in MTF_COLUMNS:
        _same(part[col], want[col], f"{col} changed when later bars were removed (cut {cut})")
    for col in DATE_COLS:
        assert part[col].equals(want[col]), f"{col} changed when later bars were removed (cut {cut})"


@pytest.mark.parametrize("cut", [600, 601, 900, 1300])
def test_weekly_table_ignores_the_benchmarks_future_too(pair, full, cut):
    """The benchmark cut at the same date gives the same table: no benchmark bar after a week's label
    (not even the month containing that week) reaches the rs_ / bm_ columns of that week."""
    d, b = pair
    part = mtf_features(d.iloc[: cut + 1], b.iloc[: cut + 1])
    want = full[full.index <= d.index[cut]]
    assert part.index.equals(want.index)
    for col in MTF_COLUMNS:
        _same(part[col], want[col], f"{col} depends on benchmark bars after the week (cut {cut})")
    for col in DATE_COLS:
        assert part[col].equals(want[col])


def test_the_table_is_not_vacuous(full):
    for col in MTF_COLUMNS:
        assert full[col].notna().sum() > 100, f"{col} is almost all NaN on a 300-week series"


# ── (e) truncation invariance of the daily mapping ────────────────────────────

@pytest.mark.parametrize("cut", [600, 601, 900, 1300])
def test_daily_table_ignores_the_future(pair, cut):
    d, b = pair
    ind_full = indicators_mtf(d, b)
    part = indicators_mtf(d.iloc[: cut + 1], b)
    assert list(part.columns) == list(ind_full.columns)
    assert part.index.equals(d.index[: cut + 1])
    for col in ind_full.columns:
        _same(part[col], ind_full[col].iloc[: cut + 1], f"{col} changed when later bars were removed (cut {cut})")


def test_the_daily_table_has_the_screen_columns_then_the_mtf_columns(pair):
    d, b = pair
    ind = indicators_mtf(d.iloc[:400], b)
    from setup_engine.screen import COLUMNS
    assert list(ind.columns) == list(COLUMNS) + list(MTF_COLUMNS)
    assert all(dt == np.float64 for dt in ind.dtypes)
    base = indicators(d.iloc[:400])
    for col in COLUMNS:
        _same(ind[col], base[col], col)


# ── (f) weekly semantics by hand ──────────────────────────────────────────────

def test_weekly_semantics_on_a_flat_series_with_one_breakout_week():
    d = _from_weekly([10] * 40 + [12], volumes=[100] * 40 + [300])
    wf = mtf_features(d)
    assert len(wf) == 41 and wf.index[-1] == _ts("2020-01-10") + pd.Timedelta(weeks=40)
    assert wf["w_prev_hi26"].iloc[40] == 10.0
    broke = (wf["w_close"] > wf["w_prev_hi26"]).to_numpy()
    assert list(np.flatnonzero(broke)) == [40]
    assert wf["w_volx"].iloc[40] == pytest.approx(3.0)
    assert wf["w_sma30"].iloc[40] == pytest.approx((29 * 10 + 12) / 30)
    assert wf["w_slope30"].iloc[40] == pytest.approx(wf["w_sma30"].iloc[40] / 10.0 - 1.0)
    assert wf["w_volume"].iloc[40] == 1500.0 and wf["w_n"].iloc[40] == 41 and wf["w_n_days"].iloc[40] == 5


def test_52_week_columns_are_nan_until_the_window_is_full():
    wf = mtf_features(_from_weekly([10] * 60))
    assert wf["w_dd52"].iloc[:51].isna().all() and np.isfinite(wf["w_dd52"].iloc[51])
    assert wf["w_hi52"].iloc[:51].isna().all()
    assert wf["w_prev_hi52"].iloc[:52].isna().all() and np.isfinite(wf["w_prev_hi52"].iloc[52])
    assert wf["w_sma30"].iloc[:29].isna().all() and np.isfinite(wf["w_sma30"].iloc[29])
    assert wf["w_dd52"].iloc[51] == 0.0
    assert (wf["w_dd26"].dropna() <= 0).all()


def test_weekly_formulas_by_hand_on_a_random_series(pair):
    d, _ = pair
    wb = weekly_bars(d)
    wf = mtf_features(d)
    c, v = wb["close"].to_numpy(), wb["volume"].to_numpy()
    n = len(c)
    assert wf["w_ret13"].iloc[13] == pytest.approx(c[13] / c[0] - 1.0, rel=1e-12) and wf["w_ret13"].iloc[:13].isna().all()
    assert wf["w_ret26"].iloc[:26].isna().all()
    for i in (26, 100, n - 1):
        assert wf["w_ret13"].iloc[i] == pytest.approx(c[i] / c[i - 13] - 1.0, rel=1e-12)
        assert wf["w_ret26"].iloc[i] == pytest.approx(c[i] / c[i - 26] - 1.0, rel=1e-12)
        assert wf["w_prev_close"].iloc[i] == c[i - 1]
    for i in (60, 150, n - 1):
        assert wf["w_hi26"].iloc[i] == c[i - 25: i + 1].max()
        assert wf["w_prev_hi26"].iloc[i] == c[i - 26: i].max()
        assert wf["w_dd26"].iloc[i] == pytest.approx(c[i] / c[i - 25: i + 1].max() - 1.0, rel=1e-12)
        assert wf["w_hi52"].iloc[i] == c[i - 51: i + 1].max()
        assert wf["w_prev_hi52"].iloc[i] == c[i - 52: i].max()
        assert wf["w_dd52"].iloc[i] == pytest.approx(c[i] / c[i - 51: i + 1].max() - 1.0, rel=1e-12)
        sma = c[i - 29: i + 1].mean()
        assert wf["w_sma30"].iloc[i] == pytest.approx(sma, rel=1e-12)
        assert wf["w_prev_sma30"].iloc[i] == pytest.approx(c[i - 30: i].mean(), rel=1e-12)
        assert wf["w_px_sma30"].iloc[i] == pytest.approx(c[i] / sma - 1.0, rel=1e-12)
        assert wf["w_slope30"].iloc[i] == pytest.approx(sma / c[i - 33: i - 3].mean() - 1.0, rel=1e-12)
        assert wf["w_volx"].iloc[i] == pytest.approx(v[i] / v[i - 10: i].mean(), rel=1e-12)
        assert wf["w_n"].iloc[i] == i + 1
    assert wf["w_n"].iloc[0] == 1


def test_dollar_volume_is_close_times_volume_and_its_13_week_median():
    vols = [100.0] * 13
    vols[5] = 10_000.0                                    # one spike week inside the window
    wf = mtf_features(_from_weekly([10] * 13, volumes=vols))
    assert wf["w_dvol"].iloc[0] == pytest.approx(5 * 10 * 100.0)
    assert wf["w_dvol"].iloc[5] == pytest.approx(5 * 10 * 10_000.0)
    assert wf["w_dvol13"].iloc[:12].isna().all()
    assert wf["w_dvol13"].iloc[12] == pytest.approx(5 * 10 * 100.0)       # the median ignores the spike


def test_monthly_formulas_by_hand():
    d = _daily(1300, seed=11, start="2014-01-02")
    mb = monthly_bars(d)
    mf = mtf.monthly_features(mb)
    c = mb["close"].to_numpy()
    assert mf.index.equals(mb.index) and (mf["m_label"].to_numpy() == mb.index.to_numpy()).all()
    for i in (24, 40, len(c) - 1):
        sma = c[i - 9: i + 1].mean()
        assert mf["m_close"].iloc[i] == c[i] and mf["m_prev_close"].iloc[i] == c[i - 1]
        assert mf["m_sma10"].iloc[i] == pytest.approx(sma, rel=1e-12)
        assert mf["m_prev_sma10"].iloc[i] == pytest.approx(c[i - 10: i].mean(), rel=1e-12)
        assert mf["m_px_sma10"].iloc[i] == pytest.approx(c[i] / sma - 1.0, rel=1e-12)
        assert mf["m_hi24"].iloc[i] == c[i - 23: i + 1].max()
        assert mf["m_dd24"].iloc[i] == pytest.approx(c[i] / c[i - 23: i + 1].max() - 1.0, rel=1e-12)
    assert mf["m_sma10"].iloc[:9].isna().all() and np.isfinite(mf["m_sma10"].iloc[9])
    assert mf["m_hi24"].iloc[:23].isna().all() and np.isfinite(mf["m_hi24"].iloc[23])


def test_the_prior_high_excludes_the_week_being_tested():
    # a breakout week: close 15 after 40 weeks at 10. The prior-26 high stays 10, so it is a NEW high;
    # the including-this-week high is 15, and the drawdown from it is 0.
    wf = mtf_features(_from_weekly([10] * 40 + [15]))
    assert wf["w_prev_hi26"].iloc[40] == 10.0 and wf["w_hi26"].iloc[40] == 15.0
    assert wf["w_dd26"].iloc[40] == 0.0 and wf["w_prev_hi52"].isna().iloc[40]


def test_higher_low_compares_the_recent_low_with_the_26_weeks_before_it():
    closes = [10] * 30 + [8] * 15 + [9] * 15      # the 13-week low (9) is above the prior 26-week low (8)
    wf = mtf_features(_from_weekly(closes))
    assert wf["w_higher_low"].iloc[:38].isna().all()
    assert wf["w_higher_low"].iloc[-1] == 1.0
    assert wf["w_low_prior"].iloc[-1] == 8.0 and wf["w_low13"].iloc[-1] == 9.0
    wf2 = mtf_features(_from_weekly([10] * 30 + [9] * 15 + [8] * 15))
    assert wf2["w_higher_low"].iloc[-1] == 0.0


# ── (g) monthly: the last COMPLETED month ─────────────────────────────────────

def test_a_week_carries_the_last_completed_month_not_the_month_it_is_in():
    d = _daily(130, seed=5, start="2024-01-01")
    d = d.loc[:"2024-05-31"]
    wf = mtf_features(d)
    feb = d.loc["2024-02-29", "close"]
    mar = d.loc["2024-03-29", "close"]
    assert wf.loc["2024-03-22", "m_close"] == feb and wf.loc["2024-03-22", "m_label"] == _ts("2024-02-29")
    assert wf.loc["2024-03-29", "m_close"] == mar and wf.loc["2024-03-29", "m_label"] == _ts("2024-03-29")
    assert wf.loc["2024-03-15", "m_close"] == feb
    # the month label is never after the week label
    assert (wf["m_label"].dropna() <= wf.index[wf["m_label"].notna()]).all()
    assert wf.loc["2024-04-05", "m_prev_close"] == feb


def test_a_holiday_month_end_still_maps_to_its_own_label():
    d = _daily(130, seed=5, start="2024-01-01").loc[:"2024-05-31"].drop(index=_ts("2024-03-29"))
    wf = mtf_features(d)
    assert wf.loc["2024-03-29", "m_label"] == _ts("2024-03-29")         # label 29 March <= week label 29 March
    assert wf.loc["2024-03-29", "m_close"] == d.loc["2024-03-28", "close"]
    assert wf.loc["2024-03-22", "m_close"] == d.loc["2024-02-29", "close"]


def test_a_stale_month_is_not_carried_across_a_long_gap():
    d = _daily(200, seed=5, start="2024-01-01").loc[:"2024-08-30"]
    d = d.drop(index=d.loc["2024-03-01":"2024-06-30"].index)           # a four-month halt
    wf = mtf_features(d)
    assert wf.loc["2024-07-05", "m_close"] != wf.loc["2024-07-05", "m_close"]       # NaN: February is > 62 days old
    assert np.isnan(wf.loc["2024-07-05", "m_px_sma10"]) and pd.isna(wf.loc["2024-07-05", "m_label"])
    assert wf.loc["2024-08-02", "m_close"] == d.loc["2024-07-31", "close"]          # July completed


# ── (h) daily mapping ─────────────────────────────────────────────────────────

def test_a_daily_row_carries_the_last_complete_week():
    d = _daily(300, seed=4, start="2023-06-01")
    ind = indicators_mtf(d)
    assert ind.loc["2024-03-22", "w_close"] == d.loc["2024-03-22", "close"]          # Friday: its own week
    assert ind.loc["2024-03-21", "w_close"] == d.loc["2024-03-15", "close"]          # Thursday: the week before
    assert ind.loc["2024-03-25", "w_close"] == d.loc["2024-03-22", "close"]          # Monday: last Friday
    wf = mtf_features(d)
    assert ind.loc["2024-03-22", "w_sma30"] == wf.loc["2024-03-22", "w_sma30"]       # a Friday row IS its week's row
    assert ind.loc["2024-03-21", "w_sma30"] == wf.loc["2024-03-15", "w_sma30"]


def test_good_friday_removed_the_holiday_week_is_carried_from_the_next_trading_day():
    d = _daily(300, seed=4, start="2023-06-01").drop(index=_ts("2024-03-29"))
    ind = indicators_mtf(d)
    assert ind.loc["2024-03-28", "w_close"] == d.loc["2024-03-22", "close"]          # Thursday: still the week of 22 March
    assert ind.loc["2024-04-01", "w_close"] == d.loc["2024-03-28", "close"]          # Monday: week 29 March (4 days)
    assert ind.loc["2024-04-01", "w_n_days"] == 4


def test_a_stock_that_stops_trading_does_not_carry_a_stale_week_forever():
    d = _daily(200, seed=3, start="2022-01-03")
    d = d.drop(index=d.loc["2022-07-04":"2022-08-15"].index)                         # a six-week halt
    ind = indicators_mtf(d)
    first_after = d.loc["2022-08-16":].index[0]
    assert np.isnan(ind.loc[first_after, "w_close"])                                 # the last week is > 13 days old
    assert np.isfinite(ind.loc["2022-06-30", "w_close"])
    assert np.isfinite(ind["w_close"].iloc[-1])


# ── (i) relative strength ─────────────────────────────────────────────────────

def test_relative_strength_against_a_benchmark_that_is_exactly_half(pair):
    d, _ = pair
    b = d.copy()
    b[["open", "high", "low", "close"]] = b[["open", "high", "low", "close"]] / 2.0
    wf = mtf_features(d, b)
    rs = wf["rs_line"]
    assert rs.notna().all() and np.allclose(rs, 2.0, rtol=1e-12)
    ret = wf["rs_ret26"].dropna()
    assert len(ret) == len(wf) - 26 and np.allclose(ret, 0.0, atol=1e-12)
    for col in ("rs_new_hi26", "rs_new_hi52", "rs_lead26"):
        v = wf[col].dropna()
        assert len(v) > 100 and (v == 0.0).all(), col
    assert wf["rs_prev_hi26"].iloc[:26].isna().all() and np.isfinite(wf["rs_prev_hi26"].iloc[26])
    assert wf["rs_prev_hi52"].iloc[:52].isna().all() and np.isfinite(wf["rs_prev_hi52"].iloc[52])
    assert wf["bm_w_px_sma30"].notna().sum() > 100 and wf["bm_m_px_sma10"].notna().sum() > 100


def test_without_a_benchmark_every_rs_and_bm_column_is_nan(pair):
    d, _ = pair
    wf = mtf_features(d)
    cols = [c for c in MTF_COLUMNS if c.startswith(("rs_", "bm_"))]
    assert len(cols) == 9 and wf[cols].isna().all().all()
    assert wf[[c for c in MTF_COLUMNS if c not in cols]].notna().any().all()
    ind = indicators_mtf(d.iloc[:500])
    assert ind[cols].isna().all().all()


def test_rs_leads_when_the_rs_line_makes_a_high_before_price_does():
    # price drops from 13 to 10 and stays there, but the benchmark falls much faster: the RS line makes a
    # new 26-week high while price is still far below its prior 26-week high (13, set 17+ weeks earlier)
    d = _from_weekly([13] * 30 + [10] * 17)
    bench = _from_weekly([130] * 30 + list(np.linspace(100, 40, 17)))
    wf = mtf_features(d, bench)
    assert wf["rs_new_hi26"].iloc[-1] == 1.0
    assert wf["w_close"].iloc[-1] <= wf["w_prev_hi26"].iloc[-1] == 13.0
    assert wf["rs_lead26"].iloc[-1] == 1.0
    assert wf["rs_lead26"].iloc[:26].isna().all()                     # NaN until the 26-week RS window is full
    # price making a new high at the same time is not "RS before price"
    d2 = _from_weekly([10] * 45 + [11, 12])
    b2 = _from_weekly([100] * 45 + [100, 100])
    wf2 = mtf_features(d2, b2)
    assert wf2["rs_new_hi26"].iloc[-1] == 1.0 and wf2["w_close"].iloc[-1] > wf2["w_prev_hi26"].iloc[-1]
    assert wf2["rs_lead26"].iloc[-1] == 0.0


def test_a_benchmark_with_missing_weeks_leaves_nan_rs_for_those_weeks(pair):
    d, b = pair
    b2 = b.drop(index=b.loc["2017-01-02":"2017-03-31"].index)
    wf = mtf_features(d, b2)
    gap = wf.loc["2017-01-13":"2017-03-24"]
    assert len(gap) > 5 and gap["rs_line"].isna().all() and gap["bm_w_px_sma30"].isna().all()
    assert wf["rs_line"].notna().sum() > 200


# ── (j) liquidity ─────────────────────────────────────────────────────────────

def _liq(price=10.0, dvol=1e7, n=100):
    return pd.DataFrame({"w_close": [price], "w_dvol13": [dvol], "w_n": [n]})


def test_is_liquid_boundaries():
    assert is_liquid(_liq(price=2.0))[0] and not is_liquid(_liq(price=1.999))[0]
    assert is_liquid(_liq(dvol=5e6))[0] and not is_liquid(_liq(dvol=5e6 - 1))[0]
    assert not is_liquid(_liq(n=59))[0] and is_liquid(_liq(n=60))[0]
    assert is_liquid(_liq())[0]
    for bad in (_liq(price=np.nan), _liq(dvol=np.nan), _liq(n=np.nan)):
        assert not is_liquid(bad)[0]
    out = is_liquid(pd.concat([_liq(), _liq(price=1.0), _liq(n=np.nan)], ignore_index=True))
    assert out.dtype == bool and list(out) == [True, False, False]


# ── (k) the ex-post best run ──────────────────────────────────────────────────

def _weekly_close(vals, start="2022-01-07"):
    return pd.Series(vals, index=pd.date_range(start, periods=len(vals), freq="7D"), dtype=float)


def test_best_run_finds_the_largest_low_to_peak_multiple():
    s = _weekly_close([10, 8, 5, 6, 12, 9, 20, 15])
    r = best_run(s, "2022-01-01", min_weeks=8)
    assert r["low_week"] == s.index[2] and r["low_close"] == 5.0
    assert r["peak_week"] == s.index[6] and r["peak_close"] == 20.0
    assert r["mult"] == 4.0 and r["weeks"] == 4
    assert r["mdd_in_run"] == pytest.approx(-0.25)
    assert r["dd_into_low"] == pytest.approx(5.0 / 10.0 - 1.0)
    assert r["max_gap_days"] == 7


def test_best_run_ranks_lows_by_multiple_not_by_points():
    s = _weekly_close([100, 150, 2, 6])                  # +50 points from 100, but x3 from 2
    r = best_run(s, "2022-01-01", min_weeks=4)
    assert r["low_close"] == 2.0 and r["peak_close"] == 6.0 and r["mult"] == 3.0


def test_best_run_start_min_weeks_and_data_gaps():
    s = _weekly_close([10, 8, 5, 6, 12, 9, 20, 15])
    assert best_run(s, "2022-01-01", min_weeks=9) is None            # too short
    assert best_run(s, s.index[3], min_weeks=3) is not None
    later = best_run(s, s.index[3], min_weeks=3)
    assert later["low_close"] == 6.0 and later["dd_into_low"] == pytest.approx(6.0 / 10.0 - 1.0)   # prior 730 days: full series
    assert best_run(s, "2022-01-01", min_weeks=8)["dd_into_low"] is not None
    first = best_run(_weekly_close([5, 6, 12, 9, 20]), "2022-01-01", min_weeks=5)
    assert first["dd_into_low"] is None                              # nothing before the low
    gappy = pd.concat([_weekly_close([5, 6]), _weekly_close([30, 31], start="2022-06-03")])
    g = best_run(gappy, "2022-01-01", min_weeks=4)
    assert g["max_gap_days"] > 100                                   # a halted / stitched series is flagged


# ── (l) benchmarks ────────────────────────────────────────────────────────────

def _fake_bench(sym, start):
    idx = pd.bdate_range("2020-01-01", periods=40)
    c = np.linspace(100, 110, 40)
    return pd.DataFrame({"open": c, "high": c + 1, "low": c - 1, "close": c, "volume": 1e6}, index=idx)


def test_ensure_benchmarks_fetches_only_what_is_missing(tmp_path):
    pytest.importorskip("pyarrow")
    calls = []

    def fetch(sym, start):
        calls.append((sym, start))
        return _fake_bench(sym, start)

    pre = tmp_path / "SPY.parquet"
    _fake_bench("SPY", None).rename_axis("date").to_parquet(pre)
    paths = ensure_benchmarks(("SPY", "QQQ"), dest=tmp_path, start="2001-01-01", fetch=fetch)
    assert calls == [("QQQ", "2001-01-01")]
    assert set(paths) == {"SPY", "QQQ"} and all(p.exists() for p in paths.values())
    assert paths["QQQ"] == tmp_path / "QQQ.parquet"
    ensure_benchmarks(("SPY", "QQQ"), dest=tmp_path, fetch=fetch)
    assert len(calls) == 1                                           # second call: nothing missing, nothing fetched
    back = load_benchmark("QQQ", dest=tmp_path)
    assert isinstance(back.index, pd.DatetimeIndex) and len(back) == 40
    assert list(back.columns) == ["open", "high", "low", "close", "volume"] and back.index.name == "date"
    assert back["close"].iloc[-1] == pytest.approx(110.0)
    clipped = load_benchmark("QQQ", dest=tmp_path, end="2020-01-10")
    assert clipped.index[-1] == _ts("2020-01-10") and len(clipped) == 8


def test_ensure_benchmarks_with_every_file_present_does_not_fetch_or_import(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    for s in ("SPY", "QQQ"):
        _fake_bench(s, None).rename_axis("date").to_parquet(tmp_path / f"{s}.parquet")

    def boom(sym, start):
        raise AssertionError("fetched although every file exists")

    monkeypatch.setattr(mtf, "_yf_fetch", boom)
    assert set(ensure_benchmarks(("SPY", "QQQ"), dest=tmp_path)) == {"SPY", "QQQ"}


def test_ensure_benchmarks_empty_fetch_raises_with_the_symbol(tmp_path):
    with pytest.raises(ValueError, match="XYZ"):
        ensure_benchmarks(("XYZ",), dest=tmp_path, fetch=lambda s, st: pd.DataFrame())
    assert not (tmp_path / "XYZ.parquet").exists()


def test_the_benchmark_constants():
    assert mtf.BENCHMARKS == ("SPY", "QQQ", "SMH", "TSM", "ASML", "ARM")
    assert mtf.BENCH_DIR.name == "benchmarks" and mtf.DEFAULT_HISTORY.name == "history_2000"


# ── (m) cross-sectional ranks ─────────────────────────────────────────────────

def test_xs_ranks_use_liquid_rows_only():
    wk1, wk2 = _ts("2024-01-05"), _ts("2024-01-12")
    panel = pd.DataFrame({
        "week": [wk1] * 5 + [wk2] * 3,
        "symbol": list("ABCDE") + list("ABC"),
        "w_ret13": [0.10, 0.20, 0.30, 0.40, 9.99, 0.5, 0.1, np.nan],
        "w_ret26": [0.01, 0.02, 0.03, 0.04, 9.99, 0.3, 0.2, 0.1],
        "liquid": [True, True, True, True, False, True, True, True],
    })
    out = add_xs_ranks(panel)
    assert out is not panel and "xs_rank_ret13" not in panel.columns
    r = out["xs_rank_ret13"].to_numpy()
    assert np.allclose(r[:4], [0.25, 0.5, 0.75, 1.0])              # the 9.99 illiquid row does not move anyone
    assert np.isnan(r[4])                                           # and gets no rank
    assert r[5] == 1.0 and r[6] == 0.5 and np.isnan(r[7])           # NaN return: no rank, others unaffected
    assert np.allclose(out["xs_rank_ret26"].to_numpy()[[0, 1, 2, 3]], [0.25, 0.5, 0.75, 1.0])
    assert np.allclose(out["xs_rank_ret26"].to_numpy()[5:], [1.0, 2 / 3, 1 / 3])
    assert np.isnan(out["xs_rank_ret26"].iloc[4])
    assert mtf.XS_COLUMNS == ("xs_rank_ret13", "xs_rank_ret26")


def test_xs_ranks_average_ties_and_work_on_a_duplicate_index():
    panel = pd.DataFrame({"week": [_ts("2024-01-05")] * 4, "w_ret13": [0.1, 0.2, 0.2, 0.3],
                          "w_ret26": [1.0, 2.0, 3.0, 4.0], "liquid": [True] * 4}, index=[0, 0, 1, 1])
    out = add_xs_ranks(panel)
    assert np.allclose(out["xs_rank_ret13"], [0.25, 0.625, 0.625, 1.0])


# ── (n) the universe driver ───────────────────────────────────────────────────

def _fn_len(symbol, daily):
    return (symbol, len(daily))


def _fn_scaled(symbol, daily, k):
    return None if symbol == "CCC" else (symbol, len(daily) * k)


def _write_batch(path, stocks):
    parts = []
    for sym, (n, seed) in stocks.items():
        d = _daily(n, seed=seed, start="2020-01-02").rename_axis("date").reset_index()
        d.insert(1, "symbol", sym)
        parts.append(d)
    pd.concat(parts, ignore_index=True).to_parquet(path)


@pytest.fixture()
def history(tmp_path):
    pytest.importorskip("pyarrow")
    _write_batch(tmp_path / "batch_0000.parquet", {"AAA": (200, 1), "BBB": (50, 2)})
    _write_batch(tmp_path / "batch_0001.parquet", {"CCC": (150, 3), "DDD": (130, 4)})
    return tmp_path


def test_map_universe_yields_one_list_per_batch_and_skips_short_stocks(history):
    out = list(map_universe(history, _fn_len, progress=False))
    assert [sorted(b) for b in out] == [[("AAA", 200)], [("CCC", 150), ("DDD", 130)]]   # BBB (50 rows) skipped
    assert out[0] == [("AAA", 200)]


def test_map_universe_limit_batches_args_and_none_results(history):
    assert list(map_universe(history, _fn_len, limit_batches=1, progress=False)) == [[("AAA", 200)]]
    out = list(map_universe(history, _fn_scaled, fn_args=(3,), progress=False))
    assert out == [[("AAA", 600)], [("DDD", 390)]]                                      # CCC returned None


def test_map_universe_with_a_fork_pool_matches_the_sequential_map(history):
    seq = list(map_universe(history, _fn_len, progress=False))
    par = list(map_universe(history, _fn_len, workers=2, progress=False))
    assert sorted(sorted(b) for b in par) == sorted(sorted(b) for b in seq)


def test_map_universe_prints_progress(history, capsys):
    list(map_universe(history, _fn_len))
    out = capsys.readouterr().out
    assert "1/2 batches" in out and "2/2 batches" in out


def test_data_end_and_load_daily_read_the_cache(history):
    assert mtf.data_end(history) == _daily(200, seed=1, start="2020-01-02").index[-1]
    assert mtf.data_end(history, limit_batches=1) == _daily(200, seed=1, start="2020-01-02").index[-1]
    got = mtf.load_daily(history, ["AAA", "DDD", "ZZZ"])
    assert set(got) == {"AAA", "DDD"}
    assert len(got["AAA"]) == 200 and len(got["DDD"]) == 130
    assert isinstance(got["AAA"].index, pd.DatetimeIndex) and got["AAA"].index.tz is None
    assert list(got["AAA"].columns) == ["open", "high", "low", "close", "volume"]
    _same(got["AAA"]["close"], _daily(200, seed=1, start="2020-01-02")["close"])


# ── (o) RSI and ATR parity with the screener ──────────────────────────────────

def _manual_wilder_rsi(close, n=14):
    c = np.asarray(close, dtype=float)
    out = np.full(len(c), np.nan)
    ag = al = 0.0                                      # the first bar's NaN change counts as zero gain and loss
    for i in range(1, len(c)):
        ch = c[i] - c[i - 1]
        ag = ag * (n - 1) / n + max(ch, 0.0) / n
        al = al * (n - 1) / n + max(-ch, 0.0) / n
        if i >= n - 1 and al > 0:
            out[i] = 100.0 - 100.0 / (1.0 + ag / al)
    return out


def test_weekly_rsi_and_atr_equal_the_screeners_wilder_formulas(pair):
    d, _ = pair
    wb = weekly_bars(d)
    wf = mtf_features(d)
    _same(wf["w_rsi14"], _manual_wilder_rsi(wb["close"].to_numpy()), "w_rsi14 vs an independent Wilder loop")
    ref = indicators(wb[["open", "high", "low", "close", "volume"]])
    _same(wf["w_rsi14"], ref["rsi14"], "w_rsi14 vs screen.indicators on weekly bars")
    _same(wf["w_atr_pct"], ref["atr_pct"], "w_atr_pct vs screen.indicators on weekly bars")
    assert wf["w_rsi14"].notna().sum() > 250 and wf["w_atr_pct"].notna().sum() > 250
    mb = monthly_bars(d)
    mref = indicators(mb[["open", "high", "low", "close", "volume"]])
    mf = mtf.monthly_features(mb)
    _same(mf["m_rsi14"], mref["rsi14"], "m_rsi14 vs screen.indicators on monthly bars")
    _same(mf["m_rsi14"], _manual_wilder_rsi(mb["close"].to_numpy()), "m_rsi14 vs an independent Wilder loop")


# ── contract extras ───────────────────────────────────────────────────────────

def test_a_short_history_gives_an_empty_table_with_the_right_columns():
    d = _daily(3, start="2024-04-01")                                  # Monday..Wednesday: no complete week
    wf = mtf_features(d)
    assert len(wf) == 0 and list(wf.columns) == list(MTF_COLUMNS) + DATE_COLS
    ind = indicators_mtf(d)
    assert len(ind) == 3 and ind[list(MTF_COLUMNS)].isna().all().all()


def test_the_weekly_table_respects_as_of(pair):
    d, b = pair
    a = d.index[700]
    got = mtf_features(d, b, as_of=a)
    assert got.index[-1] <= a
    want = mtf_features(d.loc[:a], b)
    assert got.index.equals(want.index)
    for col in MTF_COLUMNS:
        _same(got[col], want[col], col)


def test_a_benchmark_is_clipped_to_as_of_before_its_bars_are_built():
    # the benchmark keeps going after the stock's data ends; its later bars must not reach any rs_ column
    d = _daily(300, seed=1, start="2020-01-02")
    b_long = _daily(500, seed=9, start="2020-01-02")
    a = d.index[-1]
    got = mtf_features(d, b_long)
    cut = mtf_features(d, b_long.loc[:a])
    for col in MTF_COLUMNS:
        _same(got[col], cut[col], col)
    # and moving only the FUTURE bench values changes nothing
    b_alt = b_long.copy()
    b_alt.loc[b_alt.index > a, ["open", "high", "low", "close"]] *= 3.0
    alt = mtf_features(d, b_alt)
    for col in MTF_COLUMNS:
        _same(got[col], alt[col], col)


# ── pre-registered signatures ─────────────────────────────────────────────────

def test_signatures_are_written_exactly_as_pre_registered():
    assert list(SIGNATURES) == ["B0", "S1", "S2", "S3", "S4"]
    assert SIGNATURES["B0"] == {"Weekly SMA30": "Price above"}
    assert SIGNATURES["S1"] == {"Performance Rank (13 Weeks)": "Top 20%", "Weekly SMA30": "Price above",
                                "Weekly SMA30 Slope": "Rising"}
    assert SIGNATURES["S2"] == {"Weekly High (52w)": "30% or more below", "Weekly SMA30": "Price above",
                                "Performance Rank (26 Weeks)": "Top 20%"}
    assert SIGNATURES["S3"] == {"Weekly High (26w)": "New High", "Weekly Volume": "Over 1.5x 10-week average",
                                "Weekly SMA30": "Price above", "Weekly SMA30 Slope": "Rising",
                                "Monthly SMA10": "Price above"}
    assert SIGNATURES["S4"] == {**SIGNATURES["S3"], "RS Line": "New 26-week high",
                                "Benchmark Weekly SMA30": "Price above"}
    assert len(SIGNATURES["S4"]) == len(SIGNATURES["S3"]) + 2
    assert set(mtf.SIGNATURE_LABELS) == set(SIGNATURES)
    assert mtf.SIGNATURE_LABELS["S2"] == "beaten-down reclaim"
