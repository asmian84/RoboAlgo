"""setup_engine.screen x setup_engine.mtf: every multi-timeframe filter, on real mtf tables, must be causal and NaN-safe.

The failure being guarded against: a weekly/monthly/RS filter that "finds" setups because the table
under it quietly knows the future. ``test_mtf.py`` pins the columns and ``test_screen_mtf.py`` pins the
filter definitions on hand-made frames; neither proves that the PAIR is causal. The traps here:
* a daily row that reads its OWN in-progress week (a Thursday seeing Friday's close), which makes a
  filter evaluated on the table cut at Thursday differ from the same filter on the full table;
* a Friday row that maps to the PREVIOUS week (causal but stale), which no truncation test can see
  because the shifted table is still a function of the past, so the Friday row is compared with the
  weekly table directly;
* a filter that matches a row whose inputs are NaN (warm-up, no benchmark, an illiquid stock's rank);
* a cross-sectional rank that an illiquid stock can move, or that is computed before the liquidity cut;
* a pre-registered signature that does not even construct, or that is not nested the way it is written
  (S4 is S3 plus two rules; S3 and S1 contain "Weekly SMA30: Price above").
All data is synthetic with fixed seeds; no test touches the network or the ledger.
"""
import numpy as np
import pandas as pd
import pytest

from setup_engine import mtf
from setup_engine import screen as S
from setup_engine.mtf import MTF_COLUMNS, SIGNATURES, XS_COLUMNS, add_xs_ranks, indicators_mtf, is_liquid, mtf_features
from setup_engine.screen import MTF_FILTER_COLUMNS, MTF_FILTERS, Screen

CUTS = (600, 900, 1300)         # every cut falls on a Thursday: the Friday that completes the week is NOT yet known


def _daily(n=1800, seed=2, start="2014-01-02"):
    rng = np.random.default_rng(seed)
    close = 20 * np.exp(np.cumsum(rng.normal(0.0004, 0.02, n)))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.008, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.008, n)))
    vol = rng.integers(500_000, 3_000_000, n).astype(float)
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx)


def _rank_free_options():
    """Every (filter, example) of MTF_FILTERS except the two cross-sectional Performance Rank filters."""
    return [(f, o) for f, options in MTF_FILTERS.items() if not f.startswith("Performance Rank") for o in options]


def _all_options():
    return [(f, o) for f, options in MTF_FILTERS.items() for o in options]


@pytest.fixture(scope="module")
def data():
    d, b = _daily(1500, seed=2), _daily(1500, seed=7)
    return d, b


@pytest.fixture(scope="module")
def tables(data):
    """Full tables and the same tables rebuilt from data cut at each date (the benchmark stays FULL)."""
    d, b = data
    full_ind, full_wk = indicators_mtf(d, b), mtf_features(d, b)
    cut_ind, cut_wk = {}, {}
    for cut in CUTS:
        cut_ind[cut] = indicators_mtf(d.iloc[: cut + 1], b)
        wk = mtf_features(d.iloc[: cut + 1], b)
        cut_wk[cut] = (wk, full_wk[full_wk.index <= d.index[cut]])
    return {"full_ind": full_ind, "full_wk": full_wk, "cut_ind": cut_ind, "cut_wk": cut_wk}


# ── (1) the table has every column the filters read ───────────────────────────

def test_every_non_rank_filter_column_is_produced_by_mtf():
    missing = set(MTF_FILTER_COLUMNS) - set(XS_COLUMNS) - set(MTF_COLUMNS)
    assert not missing, f"screen reads columns that mtf never produces: {sorted(missing)}"
    assert set(XS_COLUMNS) <= set(MTF_FILTER_COLUMNS)                 # and the rank columns are the study's job


# ── (2) no look-ahead, every filter, both table shapes ────────────────────────

@pytest.mark.parametrize("filter_name,option", _rank_free_options())
def test_mtf_filter_mask_is_unchanged_when_the_future_is_cut_off(tables, data, filter_name, option):
    d, _ = data
    screen = Screen({filter_name: option})
    full_mask = screen.mask(tables["full_ind"])
    for cut in CUTS:
        got = screen.mask(tables["cut_ind"][cut])
        assert len(got) == cut + 1
        assert np.array_equal(full_mask[: cut + 1], got), (
            f"{filter_name}: {option} on indicators_mtf changes when data after {d.index[cut].date()} is removed")
        trunc, restricted = tables["cut_wk"][cut]
        assert trunc.index.equals(restricted.index), f"weekly index differs at {d.index[cut].date()}"
        assert np.array_equal(screen.mask(restricted), screen.mask(trunc)), (
            f"{filter_name}: {option} on mtf_features changes when data after {d.index[cut].date()} is removed")


def test_the_cut_dates_leave_the_cut_week_unfinished(data):
    """The truncation test only has teeth if the last week of every cut is still in progress."""
    d, _ = data
    for cut in CUTS:
        assert mtf.week_label(d.index[cut])[0] > d.index[cut]


# ── (3) not vacuous ───────────────────────────────────────────────────────────

def test_most_mtf_options_fire_somewhere_so_the_invariance_test_is_not_vacuous(tables):
    options = _rank_free_options()
    for name, table in (("indicators_mtf", tables["full_ind"]), ("mtf_features", tables["full_wk"]),
                        ("indicators_mtf cut at 1300", tables["cut_ind"][1300])):
        fired = [Screen({f: o}).mask(table).any() for f, o in options]
        assert sum(fired) > 0.5 * len(options), f"only {sum(fired)}/{len(options)} options ever match on {name}"


def test_both_directions_of_the_basic_trend_filters_fire(tables):
    """A filter pair that is always on one side would make the truncation comparison trivially equal."""
    t = tables["full_ind"]
    for f, a, b in (("Weekly SMA30", "Price above", "Price below"), ("Weekly SMA30 Slope", "Rising", "Falling"),
                    ("Monthly SMA10", "Price above", "Price below"), ("Weekly Low", "Higher low", "Lower low"),
                    ("Benchmark Weekly SMA30", "Price above", "Price below")):
        up, down = Screen({f: a}).mask(t), Screen({f: b}).mask(t)
        assert up.any() and down.any(), f
        assert not (up & down).any(), f


# ── (4) NaN never matches ─────────────────────────────────────────────────────

def test_an_all_nan_table_matches_no_mtf_option():
    cols = sorted(set(MTF_COLUMNS) | set(XS_COLUMNS) | set(MTF_FILTER_COLUMNS))
    nan = pd.DataFrame(np.nan, index=range(6), columns=cols)
    for f, o in _all_options():
        assert not Screen({f: o}).mask(nan).any(), f"{f}: {o} matches an all-NaN row"


def test_no_option_matches_a_row_with_a_nan_input_on_a_real_table(tables):
    """On a real table the warm-up rows and the benchmark-less rows are NaN: those must never pass."""
    for table in (tables["full_ind"], tables["full_wk"]):
        for f, o in _rank_free_options():
            screen = Screen({f: o})
            m = screen.mask(table)
            needed = table[screen.fields()].to_numpy(dtype="float64")
            assert np.isfinite(needed[m]).all(), f"{f}: {o} matched a row with a NaN input"


def test_without_a_benchmark_every_relative_strength_option_matches_nothing(data):
    d, _ = data
    t = indicators_mtf(d)                                              # no benchmark
    for f in ("RS vs Benchmark (26w)", "RS Line", "Benchmark Weekly SMA30", "Benchmark Monthly SMA10"):
        for o in MTF_FILTERS[f]:
            assert not Screen({f: o}).mask(t).any(), f"{f}: {o} matched with no benchmark"


# ── (5) Performance Rank on a panel built with add_xs_ranks ───────────────────

def _rank_panel():
    """Two weeks. Liquid stocks have distinct returns; illiquid stocks (price, dollar volume, history) and a
    liquid stock with no return are scattered in with returns that would top every ranking."""
    rng = np.random.default_rng(11)
    rows = []
    for week, n_liquid in ((pd.Timestamp("2024-03-01"), 20), (pd.Timestamp("2024-03-08"), 10)):
        r13 = rng.permutation(np.arange(1, n_liquid + 1)) / 100.0
        r26 = rng.permutation(np.arange(1, n_liquid + 1)) / 50.0
        for i in range(n_liquid):
            rows.append({"week": week, "sym": f"L{i}", "w_close": 10.0, "w_dvol13": 1e7, "w_n": 100.0,
                         "w_ret13": r13[i], "w_ret26": r26[i]})
        rows += [{"week": week, "sym": "cheap", "w_close": 1.5, "w_dvol13": 1e7, "w_n": 100.0,
                  "w_ret13": 9.0, "w_ret26": 9.0},
                 {"week": week, "sym": "thin", "w_close": 10.0, "w_dvol13": 1e6, "w_n": 100.0,
                  "w_ret13": 9.0, "w_ret26": 9.0},
                 {"week": week, "sym": "young", "w_close": 10.0, "w_dvol13": 1e7, "w_n": 30.0,
                  "w_ret13": 9.0, "w_ret26": 9.0},
                 {"week": week, "sym": "noret", "w_close": 10.0, "w_dvol13": 1e7, "w_n": 100.0,
                  "w_ret13": np.nan, "w_ret26": np.nan}]
    panel = pd.DataFrame(rows)
    panel["liquid"] = is_liquid(panel)
    return add_xs_ranks(panel)


def _expected_rank(panel, col, side, pct):
    """Hand rule: among liquid rows with a finite return in one week, ordinal k of n (1 = worst);
    Top P <=> k/n >= 1-P, Bottom P <=> k/n <= P, done in integers (P = 20 means 20%)."""
    out = np.zeros(len(panel), dtype=bool)
    for _, g in panel[panel["liquid"] & np.isfinite(panel[col])].groupby("week"):
        order = g[col].sort_values().index
        n = len(order)
        for k, label in enumerate(order, 1):
            keep = 100 * k >= (100 - pct) * n if side == "Top" else 100 * k <= pct * n
            out[panel.index.get_loc(label)] = keep
    return out


@pytest.mark.parametrize("n_weeks,col,source", [("13", "xs_rank_ret13", "w_ret13"), ("26", "xs_rank_ret26", "w_ret26")])
@pytest.mark.parametrize("side,option,pct", [("Top", "Top 20%", 20), ("Bottom", "Bottom 20%", 20),
                                             ("Top", "Top 50%", 50), ("Top", "Top 10%", 10)])
def test_performance_rank_selects_exactly_the_liquid_quantile(n_weeks, col, source, side, option, pct):
    panel = _rank_panel()
    got = Screen({f"Performance Rank ({n_weeks} Weeks)": option}).mask(panel)
    want = _expected_rank(panel, source, side, pct)
    assert want.any()
    assert np.array_equal(got, want), panel.loc[got != want, ["week", "sym", source, col]]
    illiquid = ~panel["liquid"].to_numpy()
    assert not got[illiquid].any()                                     # cheap / thin / young / no return: never selected
    assert panel.loc[illiquid, col].isna().all()


def test_an_illiquid_stock_does_not_move_the_ranks_of_the_liquid_ones():
    panel = _rank_panel()
    liquid = panel[panel["liquid"]].copy()
    alone = add_xs_ranks(liquid.drop(columns=["xs_rank_ret13", "xs_rank_ret26"]))
    both = panel.loc[panel["liquid"]]
    assert np.allclose(both["xs_rank_ret13"], alone["xs_rank_ret13"], equal_nan=True)
    assert np.allclose(both["xs_rank_ret26"], alone["xs_rank_ret26"], equal_nan=True)
    first = both[both["week"] == both["week"].min()]
    assert sorted(first["xs_rank_ret13"].dropna().round(10)) == [round(k / 20, 10) for k in range(1, 21)]
    assert first.loc[first["sym"] == "noret", "xs_rank_ret13"].isna().all()    # liquid, but no return: no rank


# ── (6) the pre-registered signatures ─────────────────────────────────────────

@pytest.fixture(scope="module")
def panel_of_stocks():
    """Twelve random-walk stocks against one benchmark: weekly tables stacked, liquidity and ranks added."""
    bench = _daily(1500, seed=100)
    parts = []
    for seed in range(12):
        wk = mtf_features(_daily(1500, seed=seed + 1), bench)
        wk = wk.reset_index()
        wk["sym"] = f"S{seed}"
        parts.append(wk)
    panel = pd.concat(parts, ignore_index=True)
    panel["liquid"] = is_liquid(panel)
    return add_xs_ranks(panel)


def test_every_signature_evaluates_on_a_weekly_panel_with_ranks(panel_of_stocks):
    assert set(SIGNATURES) == {"B0", "S1", "S2", "S3", "S4"}
    assert set(mtf.SIGNATURE_LABELS) == set(SIGNATURES)
    masks = {}
    for name, rules in SIGNATURES.items():
        screen = Screen(rules)
        masks[name] = screen.mask(panel_of_stocks)
        assert masks[name].dtype == bool and len(masks[name]) == len(panel_of_stocks), name
    assert masks["B0"].any()                                           # the market-beta baseline fires
    assert masks["B0"].sum() < len(panel_of_stocks)                    # ... and is not everything
    assert masks["S1"].any()                                           # a top-quintile leader exists in twelve stocks
    # nesting as written: S1, S3 contain "Weekly SMA30: Price above"; S4 is S3 plus two rules
    assert not (masks["S1"] & ~masks["B0"]).any()
    assert not (masks["S3"] & ~masks["B0"]).any()
    assert not (masks["S4"] & ~masks["S3"]).any()
    # NaN safety on the study's own panel: a signature never selects a row with a NaN input
    for name, rules in SIGNATURES.items():
        screen = Screen(rules)
        needed = panel_of_stocks[screen.fields()].to_numpy(dtype="float64")
        assert np.isfinite(needed[masks[name]]).all(), name


def test_rank_free_signatures_evaluate_on_a_daily_table_and_rank_signatures_say_what_is_missing(data):
    d, b = data
    t = indicators_mtf(d, b)
    for name in ("B0", "S3", "S4"):
        m = Screen(SIGNATURES[name]).mask(t)
        assert len(m) == len(t)
    assert Screen(SIGNATURES["B0"]).mask(t).any()
    assert not (Screen(SIGNATURES["S4"]).mask(t) & ~Screen(SIGNATURES["S3"]).mask(t)).any()
    for name in ("S1", "S2"):
        with pytest.raises(ValueError, match="add_xs_ranks"):
            Screen(SIGNATURES[name]).mask(t)


# ── (7) Friday rows carry their own week, Thursday rows the week before ───────

def test_a_friday_row_carries_its_own_week_and_a_thursday_row_the_week_before(tables, data):
    d, _ = data
    ind, wk = tables["full_ind"], tables["full_wk"]
    fridays = [dt for dt in wk.index if dt in ind.index]
    assert len(fridays) > 200
    for col in ("w_close", "w_sma30", "w_n"):
        got = ind.loc[fridays, col].to_numpy()
        want = wk.loc[fridays, col].to_numpy()
        assert np.allclose(got, want, rtol=1e-12, equal_nan=True), f"Friday rows read the wrong week for {col}"
    # the Thursday before each Friday still carries the week before (its own week is not complete yet)
    pairs = [(dt - pd.Timedelta(days=1), prev) for prev, dt in zip(wk.index[:-1], wk.index[1:])
             if (dt - pd.Timedelta(days=1)) in ind.index]
    thursdays = [t for t, _ in pairs]
    previous = [p for _, p in pairs]
    for col in ("w_close", "w_sma30"):
        got = ind.loc[thursdays, col].to_numpy()
        want = wk.loc[previous, col].to_numpy()
        assert np.allclose(got, want, rtol=1e-12, equal_nan=True), f"Thursday rows read the wrong week for {col}"
    # and a filter on the daily table agrees with the same filter on the weekly table at every Friday
    for f, o in (("Weekly SMA30", "Price above"), ("Weekly High (26w)", "New High"), ("Weekly SMA30 Slope", "Rising")):
        screen = Screen({f: o})
        daily_m = pd.Series(screen.mask(ind), index=ind.index).loc[fridays].to_numpy()
        weekly_m = pd.Series(screen.mask(wk), index=wk.index).loc[fridays].to_numpy()
        assert np.array_equal(daily_m, weekly_m), f"{f}: {o} differs between the daily and the weekly table"
