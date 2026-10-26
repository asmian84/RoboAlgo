"""setup_engine.screen: a screener must not use the future, and must never let a NaN pass.

The failure being guarded against: a screen that "finds" setups because a column quietly
contains information from after the bar it is evaluated on (a volume average that includes the
trigger day, a "new high" measured against a window that includes today, a centred window), or
because a warm-up bar with no data at all (NaN) passes a filter written as the negation of
another. Both produce a screener, and a backtest of it, that looks better than anything
tradable. Each guard below is built as a trap and is mutation-tested.
"""
import numpy as np
import pandas as pd
import pytest

from setup_engine import screen as S
from setup_engine.screen import COLUMNS, Screen, available_filters, indicators


def _daily(n=1800, seed=2, start="2014-01-02"):
    rng = np.random.default_rng(seed)
    close = 20 * np.exp(np.cumsum(rng.normal(0.0004, 0.02, n)))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.008, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.008, n)))
    vol = rng.integers(500_000, 3_000_000, n).astype(float)
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx)


def _bars(n, open_=10.0, high=10.0, low=10.0, close=10.0, volume=100.0):
    """n flat bars; each argument is a scalar or an array of length n."""
    idx = pd.bdate_range("2020-01-01", periods=n)
    cols = {"open": open_, "high": high, "low": low, "close": close, "volume": volume}
    return pd.DataFrame({k: np.broadcast_to(np.asarray(v, dtype=float), (n,)).copy() for k, v in cols.items()},
                        index=idx)


def _nan_ind(n=5):
    return pd.DataFrame(np.nan, index=range(n), columns=list(COLUMNS))


def _all_options():
    return [(f, o) for f, opts in available_filters().items() for o in opts]


def _same_spec(got, want):
    assert len(got) == len(want), (got, want)
    for g, w in zip(got, want):
        if isinstance(w, str):
            assert g == w, (got, want)
        else:
            assert g == pytest.approx(w, rel=1e-12, abs=1e-12), (got, want)


@pytest.fixture(scope="module")
def full():
    d = _daily(1500)
    return d, indicators(d)


# ── no look-ahead ─────────────────────────────────────────────────────────────

def test_the_table_has_exactly_the_documented_columns_as_float64():
    d = _daily(700)
    ind = indicators(d)
    assert list(ind.columns) == list(COLUMNS)
    assert set(ind.columns) == set(COLUMNS)
    assert ind.index.equals(d.index)
    assert all(dt == np.float64 for dt in ind.dtypes)
    assert not np.isinf(ind.to_numpy()).any()


@pytest.mark.parametrize("cut", [600, 900, 1300])
def test_every_column_ignores_the_future(full, cut):
    d, ind = full
    part = indicators(d.iloc[: cut + 1])
    assert set(part.columns) == set(COLUMNS)
    for col in COLUMNS:
        a, b = ind[col].iloc[: cut + 1].to_numpy(), part[col].to_numpy()
        assert np.allclose(a, b, rtol=1e-12, equal_nan=True), f"{col} changed when later bars were removed"


@pytest.mark.parametrize("cut", [600, 900, 1300])
def test_every_example_filter_ignores_the_future(full, cut):
    d, ind = full
    part = indicators(d.iloc[: cut + 1])
    options = _all_options()
    assert len(options) > 200
    fired = 0
    for f, o in options:
        scr = Screen({f: o})
        m_full = scr.mask(ind)[: cut + 1]
        m_cut = scr.mask(part)
        assert np.array_equal(m_full, m_cut), f"{f}: {o} flipped when later bars were removed"
        fired += int(scr.mask(ind).any())
    assert fired > 0.5 * len(options), "most options never fire on the random walk: the check is vacuous"


def test_rolling_windows_are_nan_until_their_own_window_is_full(full):
    _, ind = full
    first_valid = {
        "open": 0, "high": 0, "low": 0, "close": 0, "volume": 0,
        "prev_close": 1, "change": 1, "gap": 1,
        "sma20": 19, "sma50": 49, "sma200": 199, "prev_sma20": 20, "prev_sma50": 50, "prev_sma200": 200,
        "rsi14": 13, "atr14": 13, "atr_pct": 13,
        "avg_vol20": 20, "relvol": 20, "dollar_vol20": 20,
        "high_20": 19, "high_50": 49, "high_52w": 251, "high_2y": 503,
        "low_20": 19, "low_50": 49, "low_52w": 251, "low_2y": 503,
        "prev_high_20": 20, "prev_high_50": 50, "prev_high_52w": 252, "prev_high_2y": 504,
        "prev_low_20": 20, "prev_low_50": 50, "prev_low_52w": 252, "prev_low_2y": 504,
        "pct_below_high_20": 19, "pct_below_high_50": 49, "pct_below_high_52w": 251, "pct_below_high_2y": 503,
        "pct_above_low_20": 19, "pct_above_low_50": 49, "pct_above_low_52w": 251, "pct_above_low_2y": 503,
        "perf_week": 5, "perf_month": 21, "perf_quarter": 63, "perf_half": 126, "perf_year": 252,
        "low_recent": 125, "low_prior": 377, "higher_low": 377,
    }
    assert set(first_valid) == set(COLUMNS)
    for col, k in first_valid.items():
        valid = ind[col].notna().to_numpy()
        assert valid[k:].all(), f"{col} has NaN after bar {k}"
        assert not valid[:k].any(), f"{col} is not NaN before bar {k}: its window is not full yet"


# ── NaN never matches ─────────────────────────────────────────────────────────

def test_nan_never_matches_any_example_option():
    ind = _nan_ind(5)
    for f, o in _all_options():
        assert not Screen({f: o}).mask(ind).any(), f"{f}: {o} matched an all-NaN row"


def test_each_comparison_rejects_nan_on_its_own_without_help_from_valid():
    """Guards 'Under' written as not-Over: the evaluator must be NaN-safe by itself."""
    ind = _nan_ind(5)
    for f, opts in S.FILTERS.items():
        for o, spec in opts.items():
            assert not S._evaluate(spec, ind).any(), f"{f}: {o} evaluates True on NaN"


def test_an_infinite_value_never_matches():
    ind = _nan_ind(3)
    ind["close"] = [np.inf, 6.0, -np.inf]
    scr = Screen({"Price": "Over $5"})
    assert scr.mask(ind).tolist() == [False, True, False]
    assert scr.valid(ind).tolist() == [False, True, False]


def test_inf_is_replaced_by_nan():
    ind = indicators(_bars(60, volume=0.0))            # avg_vol20 == 0 -> volume / 0 would be inf
    assert ind["avg_vol20"].iloc[30] == 0.0
    assert ind["relvol"].isna().all()
    assert not np.isinf(ind.to_numpy()).any()


# ── semantics on hand-built frames ────────────────────────────────────────────

def test_relative_volume_and_dollar_volume_use_only_prior_bars():
    vol = np.r_[np.full(25, 100.0), 300.0]
    ind = indicators(_bars(26, volume=vol))
    assert np.isnan(ind["relvol"].iloc[19])
    assert ind["relvol"].iloc[20] == 1.0
    assert ind["relvol"].iloc[25] == 3.0                # 300 / mean of bars 5..24, not including itself
    assert ind["dollar_vol20"].iloc[25] == 1000.0       # 20 prior bars of 10 * 100; the spike's 3000 excluded


def test_a_spike_cannot_lift_itself_over_a_liquidity_filter():
    vol = np.r_[np.full(25, 100.0), 10_000.0]
    ind = indicators(_bars(26, volume=vol))
    assert not Screen({"Dollar Volume 20d": "Over 5000"}).mask(ind)[25]
    assert not Screen({"Average Volume": "Over 1000"}).mask(ind)[25]
    assert Screen({"Relative Volume": "Over 5"}).mask(ind)[25]


def test_new_high_is_measured_against_yesterdays_window_not_todays():
    def frame(close25, high25):
        close = np.full(26, 10.0)
        high = np.full(26, 10.0)
        close[25], high[25] = close25, high25
        return indicators(_bars(26, open_=10.0, high=high, low=9.0, close=close))
    scr = Screen({"20-Day High": "New High"})
    assert scr.mask(frame(10.5, 10.6))[25]              # 10.5 > the prior 20 highs (10); today's 10.6 must not count
    assert not scr.mask(frame(9.9, 10.6))[25]
    assert not scr.mask(frame(10.0, 10.0))[25]          # equal is not a new high
    assert not scr.mask(frame(10.5, 10.6))[:25].any()


def test_new_low_is_measured_against_yesterdays_window_not_todays():
    close = np.full(26, 10.0)
    low = np.full(26, 10.0)
    close[25], low[25] = 9.5, 9.4
    ind = indicators(_bars(26, open_=10.0, high=11.0, low=low, close=close))
    scr = Screen({"20-Day Low": "New Low"})
    assert scr.mask(ind)[25]
    assert not scr.mask(ind)[:25].any()


def test_two_year_high_boundary_and_warmup():
    o = np.r_[np.full(100, 100.0), np.full(500, 30.0)]
    ind = indicators(_bars(600, open_=o, high=o, low=o, close=o))
    scr = Screen({"2-Year High": "70% or more below"})
    m = scr.mask(ind)
    assert m[550]                                       # 30 is exactly 70% below 100
    assert not m[:503].any()                            # high_2y is NaN until 504 bars exist
    assert m[503:].all()
    close = o.copy()
    high = o.copy()
    close[550] = high[550] = 30.01
    assert not scr.mask(indicators(_bars(600, open_=o, high=high, low=o, close=close)))[550]


def test_comparison_boundaries_and_tolerance():
    ind = _nan_ind(5)
    ind["pct_below_high_2y"] = [0.7 - 1e-12, 0.7, 0.7 + 1e-12, 0.7 - 1e-6, 0.69]
    assert Screen({"2-Year High": "70% or more below"}).mask(ind).tolist() == [True, True, True, False, False]
    ind = _nan_ind(5)
    ind["change"] = [-0.05 + 1e-12, -0.05, -0.05 - 1e-12, -0.05 + 1e-6, -0.04]
    assert Screen({"Change": "Down 5%"}).mask(ind).tolist() == [True, True, True, False, False]
    ind = _nan_ind(5)
    ind["close"] = [5.0, 5.0 + 1e-9, 4.0, 20.0, 20.01]
    assert Screen({"Price": "Over $5"}).mask(ind).tolist() == [False, True, False, True, True]      # strict
    assert Screen({"Price": "$5 to $20"}).mask(ind).tolist() == [True, True, False, True, False]    # inclusive
    ind = _nan_ind(4)
    ind["close"] = [11.0 - 1e-12, 11.0 - 1e-6, 9.0 + 1e-12, 9.0 + 1e-6]
    ind["sma20"] = 10.0
    assert Screen({"SMA20": "Price 10% above SMA20"}).mask(ind).tolist() == [True, False, False, False]
    assert Screen({"SMA20": "Price 10% below SMA20"}).mask(ind).tolist() == [False, False, True, False]


def test_monthly_low_stand_in_boundaries():
    low = np.full(500, 10.0)
    low[100] = 5.0
    ind = indicators(_bars(500, high=10.0, low=low, close=10.0))
    hl = ind["higher_low"].to_numpy()
    assert np.isnan(hl[300])
    assert (hl[377:478] == 1.0).all()                   # bar 100 is still inside bars t-377..t-126 up to t=477
    assert (hl[478:] == 0.0).all()                      # and falls out at t=478
    higher = Screen({"Monthly Low": "Higher low"}).mask(ind)
    lower = Screen({"Monthly Low": "Lower low"}).mask(ind)
    assert higher[477] and not higher[478]
    assert not lower[478] and not lower[477]            # equal lows are neither higher nor lower
    assert not higher[300] and not lower[300]
    assert not higher[:377].any() and not lower.any()


def test_lower_low_fires_when_the_recent_low_is_below_the_prior_low():
    low = np.full(500, 10.0)
    low[450] = 5.0
    ind = indicators(_bars(500, high=10.0, low=low, close=10.0))
    assert Screen({"Monthly Low": "Lower low"}).mask(ind)[450:500].all()
    assert not Screen({"Monthly Low": "Higher low"}).mask(ind)[450]


def test_price_crossing_sma20():
    close = np.r_[np.full(25, 10.0), np.full(5, 11.0)]
    ind = indicators(_bars(30, open_=close, high=close, low=close, close=close))
    up = Screen({"SMA20": "Price crossed above SMA20"}).mask(ind)
    assert up[25] and not up[26]
    assert np.flatnonzero(up).tolist() == [25]
    assert not Screen({"SMA20": "Price crossed below SMA20"}).mask(ind).any()
    down = np.r_[np.full(25, 11.0), np.full(5, 10.0)]
    ind = indicators(_bars(30, open_=down, high=down, low=down, close=down))
    cb = Screen({"SMA20": "Price crossed below SMA20"}).mask(ind)
    assert np.flatnonzero(cb).tolist() == [25]
    assert not Screen({"SMA20": "Price crossed above SMA20"}).mask(ind).any()


def test_sma_crossing_sma():
    close = np.r_[np.full(60, 10.0), 10.0 + 0.5 * np.arange(1, 21)]
    ind = indicators(_bars(80, open_=close, high=close, low=close, close=close))
    expected = ((ind["sma20"] > ind["sma50"]) & (ind["prev_sma20"] <= ind["prev_sma50"])).to_numpy()
    got = Screen({"SMA20": "SMA20 crossed above SMA50"}).mask(ind)
    assert np.flatnonzero(got).tolist() == [60]
    assert np.array_equal(got, expected)
    assert not Screen({"SMA50": "SMA50 crossed above SMA20"}).mask(ind).any()
    assert Screen({"SMA20": "SMA20 above SMA50"}).mask(ind)[60:].all()
    assert Screen({"SMA50": "SMA50 below SMA20"}).mask(ind)[60:].all()


def test_derived_columns_follow_their_definitions(full):
    d, ind = full
    c, h, l = d["close"].to_numpy(), d["high"].to_numpy(), d["low"].to_numpy()
    assert np.allclose(ind["atr_pct"], ind["atr14"] / ind["close"], equal_nan=True)
    for key, w in S.WINDOWS.items():
        hi = pd.Series(h).rolling(w).max().to_numpy()                      # independent of the module
        lo = pd.Series(l).rolling(w).min().to_numpy()
        assert np.allclose(ind[f"high_{key}"], hi, equal_nan=True)
        assert np.allclose(ind[f"low_{key}"], lo, equal_nan=True)
        assert np.allclose(ind[f"prev_high_{key}"].iloc[1:], hi[:-1], equal_nan=True)
        assert np.allclose(ind[f"prev_low_{key}"].iloc[1:], lo[:-1], equal_nan=True)
        assert np.allclose(ind[f"pct_below_high_{key}"], 1 - c / hi, equal_nan=True)
        assert np.allclose(ind[f"pct_above_low_{key}"], c / lo - 1, equal_nan=True)
    for n in (20, 50, 200):
        sma = pd.Series(c).rolling(n).mean().to_numpy()
        assert np.allclose(ind[f"sma{n}"], sma, equal_nan=True)
        assert np.allclose(ind[f"prev_sma{n}"].iloc[1:], sma[:-1], equal_nan=True)
    assert np.allclose(ind["change"].iloc[1:], c[1:] / c[:-1] - 1)
    assert np.allclose(ind["gap"].iloc[1:], d["open"].to_numpy()[1:] / c[:-1] - 1)


def test_perf_gap_and_change_by_hand():
    close = 10.0 + np.arange(40.0)
    open_ = close + 0.5
    ind = indicators(_bars(40, open_=open_, high=close + 1, low=close - 1, close=close))
    assert ind["change"].iloc[5] == pytest.approx(15.0 / 14.0 - 1.0)
    assert ind["gap"].iloc[5] == pytest.approx(15.5 / 14.0 - 1.0)
    assert ind["perf_week"].iloc[10] == pytest.approx(20.0 / 15.0 - 1.0)
    assert ind["perf_month"].iloc[30] == pytest.approx(40.0 / 19.0 - 1.0)
    assert np.isnan(ind["perf_month"].iloc[20])
    assert np.isnan(ind["change"].iloc[0]) and np.isnan(ind["gap"].iloc[0])
    m = Screen({"Performance (Month)": "Over 200%"}).mask(ind)
    assert m[21] and not m[22] and not m[:21].any()    # 31/10-1 = 2.1 but 32/11-1 = 1.909


def test_gap_and_change_filters():
    open_ = np.full(12, 10.0)
    open_[5], open_[7] = 10.3, 9.6
    ind = indicators(_bars(12, open_=open_, high=11.0, low=9.0, close=10.0))
    gap_up, gap_down = Screen({"Gap": "Up 2%"}).mask(ind), Screen({"Gap": "Down 2%"}).mask(ind)
    assert np.flatnonzero(gap_up).tolist() == [5]
    assert np.flatnonzero(gap_down).tolist() == [7]
    assert np.flatnonzero(Screen({"Gap": "Up"}).mask(ind)).tolist() == [5]
    assert np.flatnonzero(Screen({"Gap": "Down"}).mask(ind)).tolist() == [7]
    close = np.full(12, 10.0)
    close[3] = 10.5
    ind = indicators(_bars(12, open_=10.0, high=11.0, low=9.0, close=close))
    assert np.flatnonzero(Screen({"Change": "Up 5%"}).mask(ind)).tolist() == [3]
    assert np.flatnonzero(Screen({"Change": "Down 4%"}).mask(ind)).tolist() == [4]     # 10 / 10.5 - 1 = -4.76%
    assert np.flatnonzero(Screen({"Change": "Up"}).mask(ind)).tolist() == [3]
    assert np.flatnonzero(Screen({"Change": "Down"}).mask(ind)).tolist() == [4]


def test_the_rules_are_anded_and_nan_rows_drop_out():
    ind = _nan_ind(4)
    ind["close"] = [4.0, 6.0, 8.0, 10.0]
    ind["relvol"] = [3.0, 3.0, 1.0, np.nan]
    scr = Screen({"Price": "Over $5", "Relative Volume": "Over 2"})
    assert scr.mask(ind).tolist() == [False, True, False, False]
    assert scr.valid(ind).tolist() == [True, True, True, False]
    assert scr.fields() == ["close", "relvol"]


def test_an_empty_screen_passes_every_bar():
    scr = Screen({})
    ind = _nan_ind(3)
    assert scr.mask(ind).tolist() == [True, True, True]
    assert scr.valid(ind).tolist() == [True, True, True]
    assert scr.fields() == []
    assert scr.describe() == "(no rules)"


def test_fields_are_unique_and_in_order_of_first_use():
    scr = Screen.from_text("Price: Over $5; SMA20: Price crossed above SMA20; SMA50: Price above SMA50")
    assert scr.fields() == ["close", "sma20", "prev_close", "prev_sma20", "sma50"]


# ── option grammar ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("flt,opt,spec", [
    ("Price", "Over $5", ("gt", "close", 5.0)),
    ("Price", "under $50", ("lt", "close", 50.0)),
    ("Price", "$5 to $20", ("between", "close", 5.0, 20.0)),
    ("Average Volume", "Over 1M", ("gt", "avg_vol20", 1e6)),
    ("Average Volume", "100K to 1M", ("between", "avg_vol20", 1e5, 1e6)),
    ("Current Volume", "Over 5M", ("gt", "volume", 5e6)),
    ("Dollar Volume 20d", "Over $10M", ("gt", "dollar_vol20", 1e7)),
    ("Dollar Volume 20d", "Under 1B", ("lt", "dollar_vol20", 1e9)),
    ("Relative Volume", "Over 1.5", ("gt", "relvol", 1.5)),
    ("RSI (14)", "Overbought (70)", ("gt", "rsi14", 70.0)),
    ("RSI (14)", "Oversold (30)", ("lt", "rsi14", 30.0)),
    ("RSI (14)", "Over 50", ("gt", "rsi14", 50.0)),
    ("RSI (14)", "Under 50", ("lt", "rsi14", 50.0)),
    ("ATR %", "Over 5%", ("gt", "atr_pct", 0.05)),
    ("ATR %", "Under 2%", ("lt", "atr_pct", 0.02)),
    ("Average True Range", "Under 0.5", ("lt", "atr14", 0.5)),
    ("Change", "Up", ("gt", "change", 0.0)),
    ("Change", "Down", ("lt", "change", 0.0)),
    ("Change", "Up 5%", ("ge", "change", 0.05)),
    ("Change", "Down 5%", ("le", "change", -0.05)),
    ("Gap", "Down 10%", ("le", "gap", -0.10)),
    ("Performance (Week)", "Down", ("lt", "perf_week", 0.0)),
    ("Performance (Month)", "Over 50%", ("gt", "perf_month", 0.5)),
    ("Performance (Quarter)", "Under -20%", ("lt", "perf_quarter", -0.2)),
    ("Performance (Half Year)", "Up", ("gt", "perf_half", 0.0)),
    ("Performance (Year)", "Under -75%", ("lt", "perf_year", -0.75)),
    ("SMA50", "Price above SMA50", ("gt_col", "close", "sma50")),
    ("SMA50", "Price below SMA50", ("lt_col", "close", "sma50")),
    ("SMA200", "Price crossed above SMA200", ("cross_above", "close", "sma200", "prev_close", "prev_sma200")),
    ("SMA20", "Price crossed below SMA20", ("cross_below", "close", "sma20", "prev_close", "prev_sma20")),
    ("SMA200", "Price 10% above SMA200", ("ge_ratio", "close", "sma200", 1.1)),
    ("SMA20", "Price 10% below SMA20", ("le_ratio", "close", "sma20", 0.9)),
    ("SMA20", "SMA20 above SMA50", ("gt_col", "sma20", "sma50")),
    ("SMA20", "SMA20 below SMA200", ("lt_col", "sma20", "sma200")),
    ("SMA50", "SMA50 crossed below SMA200", ("cross_below", "sma50", "sma200", "prev_sma50", "prev_sma200")),
    ("SMA200", "SMA200 crossed above SMA20", ("cross_above", "sma200", "sma20", "prev_sma200", "prev_sma20")),
    ("20-Day High", "New High", ("gt_col", "close", "prev_high_20")),
    ("52-Week High", "0-5% below", ("between", "pct_below_high_52w", 0.0, 0.05)),
    ("2-Year High", "70% or more below", ("ge", "pct_below_high_2y", 0.7)),
    ("50-Day High", "0-10% below", ("between", "pct_below_high_50", 0.0, 0.10)),
    ("50-Day Low", "New Low", ("lt_col", "close", "prev_low_50")),
    ("2-Year Low", "200% or more above", ("ge", "pct_above_low_2y", 2.0)),
    ("20-Day Low", "0-3% above", ("between", "pct_above_low_20", 0.0, 0.03)),
    ("52-Week Low", "20% or more above", ("ge", "pct_above_low_52w", 0.2)),
    ("Monthly Low", "Higher low", ("gt_col", "low_recent", "low_prior")),
    ("Monthly Low", "Lower low", ("lt_col", "low_recent", "low_prior")),
])
def test_option_parses_to_the_documented_spec(flt, opt, spec):
    _same_spec(Screen({flt: opt})._specs[0], spec)


def test_dropdown_examples_are_the_documented_lists():
    f = available_filters()
    assert set(f["Price"]) == {f"Under ${x}" for x in (1, 2, 5, 10, 20, 50)} | {
        f"Over ${x}" for x in (1, 2, 5, 10, 20, 50, 100)} | {"$1 to $5", "$1 to $10", "$5 to $20", "$10 to $50"}
    assert len(f["SMA20"]) == len(f["SMA50"]) == len(f["SMA200"]) == 14
    assert f["Monthly Low"] == ["Higher low", "Lower low"]
    assert "Overbought (80)" in f["RSI (14)"] and "Oversold (20)" in f["RSI (14)"]
    assert "90% or more below" in f["2-Year High"] and "200% or more above" in f["52-Week Low"]
    for name in ("Performance (Week)", "Performance (Month)", "Performance (Quarter)",
                 "Performance (Half Year)", "Performance (Year)"):
        assert "Under -75%" in f[name] and "Over 100%" in f[name]
    assert "Up 20%" in f["Change"] and "Up 20%" not in f["Gap"]


def test_a_percent_filter_requires_the_percent_sign_and_a_dollar_filter_does_not():
    for flt, opt in [("ATR %", "Over 2"), ("Performance (Week)", "Over 10"), ("Change", "Up 5"),
                     ("52-Week High", "0-5 below"), ("2-Year High", "70 or more below")]:
        with pytest.raises(ValueError):
            Screen({flt: opt})
    Screen({"Price": "Over 5"})
    Screen({"Dollar Volume 20d": "Over 5M"})


def test_nonsense_options_are_rejected_rather_than_silently_matching_nothing():
    for flt, opt in [("Change", "Down -5%"), ("Price", "$20 to $5"), ("SMA20", "SMA20 above SMA20"),
                     ("SMA20", "Price above SMA50"), ("2-Year High", "-10% or more below")]:
        with pytest.raises(ValueError):
            Screen({flt: opt})


def test_labels_are_case_and_whitespace_insensitive():
    a = Screen({"Price": "Over $5", "52-Week High": "0-5% below"})
    b = Screen({"  price ": " OVER   $5 ", "52-WEEK   high": "0-5%  BELOW"})
    ind = indicators(_daily(700))
    assert np.array_equal(a.mask(ind), b.mask(ind))
    assert b.rules == {"  price ": " OVER   $5 ", "52-WEEK   high": "0-5%  BELOW"}


# ── errors and text form ──────────────────────────────────────────────────────

def test_unknown_filter_lists_the_valid_names():
    with pytest.raises(ValueError, match="Price") as e:
        Screen({"Foo": "Bar"})
    assert "Foo" in str(e.value) and "Monthly Low" in str(e.value)


def test_unknown_option_lists_examples_and_patterns():
    with pytest.raises(ValueError, match=r"Over \$1") as e:
        Screen({"Price": "Over one dollar"})
    assert "Over <N>" in str(e.value) and "<N> to <N>" in str(e.value)


def test_from_text_rules():
    scr = Screen.from_text("52-Week High: 70% or more below; Relative Volume: Over 2")
    assert scr.rules == {"52-Week High": "70% or more below", "Relative Volume": "Over 2"}
    scr = Screen.from_text("Price: Over $5\n\n ; Change: Up ;\n")
    assert scr.rules == {"Price": "Over $5", "Change": "Up"}
    with pytest.raises(ValueError, match="no ':'"):
        Screen.from_text("Price Over $5")
    with pytest.raises(ValueError, match="given twice"):
        Screen.from_text("Price: Over $5; price: Under $50")
    with pytest.raises(ValueError, match="given twice"):
        Screen({"Price": "Over $5", "PRICE": "Under $50"})


def test_describe_contains_each_rule_and_its_formula():
    rules = {"Price": "Over $5", "SMA20": "Price crossed above SMA20", "2-Year High": "70% or more below",
             "Monthly Low": "Higher low"}
    text = Screen(rules).describe()
    lines = text.splitlines()
    assert len(lines) == len(rules)
    for f, o in rules.items():
        assert f"{f}: {o}" in text
    assert "[close > 5]" in text
    assert "[close > sma20 and prev_close <= prev_sma20]" in text
    assert "[pct_below_high_2y >= 0.7]" in text
    assert "[low_recent > low_prior]" in text
    assert "(calendar-free stand-in: lowest low of last 126 bars vs the 252 before)" in lines[3]
    assert "calendar-free" not in "\n".join(lines[:3])
    assert "1000000" in Screen({"Average Volume": "Over 1M"}).describe()


# ── parity with the project's indicator code ──────────────────────────────────

def test_atr_and_rsi_match_the_indicator_calculator(full):
    try:
        from indicator_engine.calculator import IndicatorCalculator
        calc = IndicatorCalculator()
        calc.compute_atr, calc.compute_rsi
    except Exception as exc:                                    # database/sqlalchemy stack not importable
        pytest.skip(f"IndicatorCalculator unavailable: {exc}")
    d, ind = full
    atr = calc.compute_atr(d["high"], d["low"], d["close"])
    rsi = calc.compute_rsi(d["close"])
    assert np.allclose(ind["atr14"].to_numpy(), atr.to_numpy(), equal_nan=True)
    assert np.allclose(ind["rsi14"].to_numpy(), rsi.to_numpy(), equal_nan=True)
    assert ind["atr14"].notna().sum() > 1400 and ind["rsi14"].notna().sum() > 1400


def test_rsi_is_nan_not_100_while_the_average_loss_is_zero_like_the_calculator():
    try:
        from indicator_engine.calculator import IndicatorCalculator
        calc = IndicatorCalculator()
    except Exception as exc:
        pytest.skip(f"IndicatorCalculator unavailable: {exc}")
    close = 10.0 + 0.1 * np.arange(60)
    d = _bars(60, open_=close, high=close, low=close, close=close)
    want = calc.compute_rsi(d["close"]).to_numpy()
    got = indicators(d)["rsi14"].to_numpy()
    assert np.isnan(want[30:]).all()                    # the calculator's own quirk
    assert np.allclose(got, want, equal_nan=True)
    assert not Screen({"RSI (14)": "Overbought (70)"}).mask(indicators(d)).any()


# ── inputs ────────────────────────────────────────────────────────────────────

def test_a_date_column_is_accepted_and_a_bad_index_is_rejected():
    d = _daily(80)
    a = indicators(d)
    b = indicators(d.reset_index(names="date"))
    assert a.index.equals(b.index)
    assert np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True)
    with pytest.raises(ValueError):
        indicators(pd.concat([d.iloc[:5], d.iloc[:5]]))              # duplicate dates
    with pytest.raises(ValueError):
        indicators(d.iloc[::-1])                                     # decreasing
    with pytest.raises(ValueError):
        indicators(d.drop(columns="volume"))


def test_clean_daily_drops_bad_bars_and_fills_volume():
    idx = pd.to_datetime(["2020-01-03", "2020-01-01", "2020-01-02", "2020-01-02", "2020-01-06",
                          "2020-01-07", "2020-01-08", "2020-01-09"])
    df = pd.DataFrame({
        "open":   [10.0, 10.0, 10.0, 11.0, 10.0, 10.0, np.nan, 10.0],
        "high":   [11.0, 11.0, 11.0, 12.0, 11.0,  9.0, 11.0, 11.0],
        "low":    [9.0,   9.0,  9.0, 10.0,  9.0, 10.0, 9.0, 0.0],
        "close":  [10.0, 10.0, 10.0, 11.5,  0.0, 10.0, 10.0, 10.0],
        "volume": [np.nan, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]}, index=idx)
    out = S.clean_daily(df)
    assert out.index.is_monotonic_increasing and out.index.is_unique
    assert list(out.index.strftime("%Y-%m-%d")) == ["2020-01-01", "2020-01-02", "2020-01-03"]
    assert out.loc["2020-01-02", "close"] == 11.5                    # duplicate: the last row is kept
    assert out.loc["2020-01-03", "volume"] == 0.0                    # NaN volume filled
