"""setup_engine.features: past-only, timeframe-agnostic, correctly labelled.

A learner that is shown the future in its inputs will "discover" setups that do not exist.
The tests here are the guard rails against that, and each is mutation-tested.
"""
import numpy as np
import pandas as pd
import pytest

from setup_engine import features as ft


def _daily(n=1800, seed=2, start="2014-01-02"):
    rng = np.random.default_rng(seed)
    close = 20 * np.exp(np.cumsum(rng.normal(0.0004, 0.02, n)))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.008, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.008, n)))
    vol = rng.integers(500_000, 3_000_000, n).astype(float)
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx)


# ── no look-ahead ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("tf", ["D", "W", "M"])
@pytest.mark.parametrize("cut", [0.45, 0.7, 0.9])
def test_bar_features_ignore_the_future(tf, cut):
    bars = ft.resample_bars(_daily(), tf)
    k = int(len(bars) * cut)
    full = ft.bar_features(bars)
    part = ft.bar_features(bars.iloc[: k + 1])
    a, b = full.iloc[: k + 1].to_numpy(), part.to_numpy()
    assert np.allclose(a, b, equal_nan=True), "a feature changed when later bars were removed"


def test_an_observation_row_is_identical_with_or_without_later_data():
    """The strongest check: rebuild the dataset from daily data CUT at each observation date."""
    d = _daily(1500)
    obs = ft.observation_dates("2016-06-01", "2019-06-01")
    full = ft.build_observations("X", d, obs, min_dollar_volume=0, with_labels=False)
    feature_cols = [c for c in full.columns if c[:2] in ("D_", "W_", "M_")]
    assert len(full) > 20
    for t in full["date"].iloc[[3, 10, 17, len(full) // 2, len(full) - 2]]:
        cut = ft.build_observations("X", d.loc[:t], pd.DatetimeIndex([t]), min_dollar_volume=0, with_labels=False)
        a = full.loc[full["date"] == t, feature_cols].to_numpy()[0]
        b = cut[feature_cols].to_numpy()[0]
        assert np.allclose(a, b, equal_nan=True), f"row for {t.date()} depends on data after {t.date()}"


def test_weekly_and_monthly_bars_are_complete_on_the_dates_they_are_used():
    d = _daily(900)
    wk = ft.resample_bars(d, "W")
    mo = ft.resample_bars(d, "M")
    assert all(ts.dayofweek == 4 for ts in wk.index)                         # labelled by their Friday
    assert all(ts == ts + pd.offsets.MonthEnd(0) for ts in mo.index)         # labelled by month-end
    # a bar's close is the close of its last trading day, never a later one
    t = wk.index[40]
    assert wk.loc[t, "close"] == d.loc[:t, "close"].iloc[-1]


def test_a_month_in_progress_is_not_visible_mid_month():
    d = _daily(1200)
    obs = pd.DatetimeIndex(["2017-03-17"])                                  # a Friday in mid-March
    row = ft.build_observations("X", d, obs, min_dollar_volume=0, with_labels=False).iloc[0]
    feb = ft.bar_features(ft.resample_bars(d.loc[:"2017-03-17"], "M")).loc["2017-02-28"]
    assert row["M_mom10"] == pytest.approx(feb["mom10"])                    # the last COMPLETED month, February


def test_a_stock_that_stopped_trading_is_not_observed_after_it_stopped():
    d = _daily(900)
    last = d.index[-1]
    obs = pd.DatetimeIndex([last + pd.Timedelta(weeks=8), last - pd.Timedelta(weeks=8)])
    out = ft.build_observations("X", d, obs, min_dollar_volume=0, with_labels=False)
    assert list(out["date"]) == [last - pd.Timedelta(weeks=8)]


# ── observation dates ─────────────────────────────────────────────────────────

def test_observation_dates_are_fridays_four_weeks_apart():
    obs = ft.observation_dates("2010-03-01", "2012-03-01")
    assert all(t.dayofweek == 4 for t in obs)
    assert set(np.diff(obs.values).astype("timedelta64[D]").astype(int)) == {28}
    assert obs[0] >= pd.Timestamp("2010-03-01") and obs[-1] <= pd.Timestamp("2012-03-01")


# ── labels ────────────────────────────────────────────────────────────────────

def _path(n, doubles_at):
    c = np.full(n, 10.0)
    c[doubles_at:] = 20.0
    return pd.Series(c, index=pd.bdate_range("2010-01-04", periods=n))


def test_forward_max_sees_a_double_inside_the_next_three_years():
    c = _path(1500, doubles_at=400)
    lab = ft.forward_labels(c, np.array([0, 100, 300]))
    assert list(lab["fwd_max_3y"]) == pytest.approx([1.0, 1.0, 1.0])     # 400 is within 756 bars of each


def test_one_year_return_is_the_price_a_year_later_not_the_peak():
    c = _path(1500, doubles_at=400)
    lab = ft.forward_labels(c, np.array([0, 300]))
    assert lab.loc[0, "fwd_ret_1y"] == pytest.approx(0.0)         # still 10 at +252
    assert lab.loc[0, "fwd_ret_2y"] == pytest.approx(1.0)         # 20 at +504
    assert lab.loc[1, "fwd_ret_1y"] == pytest.approx(1.0)         # 20 at 300+252=552
    assert lab.loc[1, "fwd_max_3y"] == pytest.approx(1.0)


def test_the_run_before_today_is_not_a_forward_label():
    c = _path(1500, doubles_at=400)
    lab = ft.forward_labels(c, np.array([450]))                   # already doubled; nothing further happens
    assert lab.loc[0, "fwd_max_3y"] == pytest.approx(0.0)


def test_a_window_that_is_not_fully_played_out_is_unlabelled():
    c = _path(1000, doubles_at=700)                               # a double inside bar 0's window
    lab = ft.forward_labels(c, np.array([0, 100, 400]))
    assert lab.loc[0, "fwd_max_3y"] == pytest.approx(1.0)         # 0 + 756 <= 999: fully observable
    assert np.isnan(lab.loc[2, "fwd_max_3y"])                     # 400 + 756 > 999: not yet known
    assert np.isnan(lab.loc[2, "fwd_ret_3y"])


def test_a_young_stock_keeps_its_rows_and_each_feature_goes_missing_only_when_its_own_window_is_short():
    d = _daily(520)                                                     # ~2 years: ~24 monthly bars
    obs = ft.observation_dates("2015-06-01", "2015-12-31")
    out = ft.build_observations("YOUNG", d, obs, min_dollar_volume=0, with_labels=False)
    assert out is not None and len(out) > 3
    assert out["M_c_ma50"].isna().all() and out["M_dist_hi50"].isna().all()    # need 50 monthly bars: not yet
    assert out["M_c_ma10"].notna().any()                                       # a 10-bar window is full by month 10
    assert out["W_mom30"].notna().all() and out["D_mom30"].notna().all()


def test_a_stock_that_only_falls_has_a_negative_forward_max_not_zero():
    # forward max is over the FUTURE closes only. If today's close leaked into the window, a stock
    # that never again trades above today would read 0% instead of its true (negative) best case.
    c = pd.Series(np.concatenate([np.full(10, 10.0), np.full(1500, 5.0)]), index=pd.bdate_range("2010-01-04", periods=1510))
    lab = ft.forward_labels(c, np.array([9]))                       # the last bar at 10; every later close is 5
    assert lab.loc[0, "fwd_max_3y"] == pytest.approx(-0.5)


def test_the_forward_window_ends_exactly_three_years_out():
    n = 1600
    base = np.full(n, 10.0)
    on_edge, past_edge = base.copy(), base.copy()
    on_edge[100 + ft.H3] = 30.0           # exactly the last bar of the window (t + 756)
    past_edge[100 + ft.H3 + 1] = 30.0     # one bar too late
    idx = pd.bdate_range("2010-01-04", periods=n)
    assert ft.forward_labels(pd.Series(on_edge, idx), np.array([100])).loc[0, "fwd_max_3y"] == pytest.approx(2.0)
    assert ft.forward_labels(pd.Series(past_edge, idx), np.array([100])).loc[0, "fwd_max_3y"] == pytest.approx(0.0)


def test_stale_weekly_and_monthly_bars_after_a_data_hole_are_not_used():
    d = _daily(1400)
    hole = d.loc["2016-01-04":"2016-05-31"].index                       # five months with no data at all
    d = d.drop(hole)
    t = pd.Timestamp("2016-06-10")                                     # a Friday, just after trading resumes
    row = ft.build_observations("X", d, pd.DatetimeIndex([t]), min_dollar_volume=0, with_labels=False)
    assert row is not None and len(row) == 1
    last_month_label = ft.resample_bars(d.loc[:t], "M").index[-2]       # the last COMPLETED month: December
    assert (t - last_month_label).days > 35
    assert row[[c for c in row.columns if c.startswith("M_")]].isna().all(axis=None), "stale monthly bar leaked in"


def test_relative_strength_ranks_within_each_date_only():
    obs = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-03"] * 3 + ["2020-02-28"] * 3),
        "symbol": list("ABCABC"),
        "W_mom30": [0.1, 0.5, 0.3, 9.0, 0.0, 0.2],          # a huge value on a LATER date must not move earlier ranks
    })
    out = ft.add_relative_strength(obs)
    first = out[out["date"] == "2020-01-03"].set_index("symbol")["rs_W_mom30"]
    assert first["B"] == 1.0 and first["A"] == pytest.approx(1 / 3) and first["C"] == pytest.approx(2 / 3)
    later = out[out["date"] == "2020-02-28"].set_index("symbol")["rs_W_mom30"]
    assert later["A"] == 1.0 and later["B"] == pytest.approx(1 / 3)


def test_six_month_labels_see_126_bars_and_nothing_else():
    n = 1000
    c = np.full(n, 10.0)
    c[100 + ft.H6] = 20.0                 # exactly the last bar of the window from bar 100
    c[300 + ft.H6 + 1] = 20.0             # one bar too late for bar 300
    s = pd.Series(c, index=pd.bdate_range("2010-01-04", periods=n))
    lab = ft.forward_labels(s, np.array([100, 300]))
    assert lab.loc[0, "fwd_max_6m"] == pytest.approx(1.0) and lab.loc[0, "fwd_ret_6m"] == pytest.approx(1.0)
    assert lab.loc[1, "fwd_max_6m"] == pytest.approx(0.0) and lab.loc[1, "fwd_ret_6m"] == pytest.approx(0.0)


def test_a_six_month_window_that_has_not_played_out_is_unlabelled():
    s = pd.Series(np.full(300, 10.0), index=pd.bdate_range("2010-01-04", periods=300))
    lab = ft.forward_labels(s, np.array([100, 190]))
    assert lab.loc[0, "fwd_max_6m"] == pytest.approx(0.0)         # 100 + 126 <= 299
    assert np.isnan(lab.loc[1, "fwd_max_6m"]) and np.isnan(lab.loc[1, "fwd_ret_6m"])    # 190 + 126 > 299


# ── delisted stocks: the outcome must not vanish with the data ───────────────────────

def _dying(n=600, collapse_at=500):
    d = _daily(n, seed=7)
    d.loc[d.index[collapse_at:], ["open", "high", "low", "close"]] *= 0.15      # an 85% collapse, then trading ends
    return d


def test_a_delisted_stocks_forward_window_is_held_at_its_last_price_not_dropped():
    d = _dying()
    obs = ft.observation_dates("2014-03-01", "2015-12-31")
    surv = ft.build_observations("X", d, obs, min_dollar_volume=0, with_labels=True, terminal=False)
    dead = ft.build_observations("X", d, obs, min_dollar_volume=0, with_labels=True, terminal=True)
    # treated as a survivor, the observations just before the end have NO six-month label (window incomplete)
    assert surv["fwd_ret_6m"].isna().any() and dead["fwd_ret_6m"].notna().all()
    # and the collapse shows up as the loss it was
    before_collapse = dead[(dead["date"] >= d.index[400]) & (dead["date"] < d.index[500])]
    assert len(before_collapse) >= 3 and (before_collapse["fwd_ret_6m"] < -0.5).all()
    after = dead[dead["date"] > d.index[505]]                                      # already collapsed: no second crash
    assert len(after) and (after["fwd_ret_6m"] > -0.5).all()


def test_holding_at_the_last_price_never_changes_a_feature():
    d = _dying()
    obs = ft.observation_dates("2014-03-01", "2015-12-31")
    a = ft.build_observations("X", d, obs, min_dollar_volume=0, with_labels=False, terminal=False)
    b = ft.build_observations("X", d, obs, min_dollar_volume=0, with_labels=False, terminal=True)
    cols = [c for c in a.columns if c[:2] in ("D_", "W_", "M_")]
    assert np.allclose(a[cols].to_numpy(), b[cols].to_numpy(), equal_nan=True)


def test_hold_at_last_price_extends_the_series_flat():
    c = pd.Series(np.arange(1.0, 11.0), index=pd.bdate_range("2020-01-06", periods=10))
    h = ft.hold_at_last_price(c, bars=5)
    assert len(h) == 15 and (h.iloc[10:] == 10.0).all() and (h.iloc[:10].to_numpy() == c.to_numpy()).all()
    assert h.index.is_monotonic_increasing and h.index[10] > c.index[-1]
