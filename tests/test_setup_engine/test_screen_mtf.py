"""setup_engine.screen, multi-timeframe vocabulary: it must extend the screener without bending it.

The failures being guarded against:
* an MTF option that reads the wrong column or the wrong side (a "30% or more below" written as
  ``w_dd52 <= +0.3`` selects nothing, a "Top 20%" written as ``rank >= 0.2`` selects the BOTTOM
  80%), because the weekly/RS columns cannot be compared with a daily table;
* a warm-up week (NaN) passing a weekly filter, because "Price below" was written as "not Price
  above" (negation turns NaN into True);
* an MTF filter leaking into ``FILTERS`` / ``available_filters()``, which every daily consumer
  walks against ``indicators()`` tables that have no weekly columns;
* a screen on a table without the weekly/monthly/RS columns silently matching nothing (or dying
  with a bare KeyError) instead of saying which module produces them.
These tests build the MTF columns by hand (this file does not import ``setup_engine.mtf``); the
column names are the contract shared with that module. Each guard is mutation-tested.
"""
import numpy as np
import pandas as pd
import pytest

from setup_engine import screen as S
from setup_engine.screen import (COLUMNS, FILTERS, MTF_FILTER_COLUMNS, MTF_FILTERS, Screen,
                                 available_filters, indicators)

# the Column contract (setup_engine/mtf.py MTF_COLUMNS + the cross-sectional ranks), copied
_CONTRACT_COLUMNS = (
    "w_open", "w_high", "w_low", "w_close", "w_volume", "w_dvol", "w_n_days", "w_n", "w_prev_close",
    "w_ret13", "w_ret26", "w_hi26", "w_prev_hi26", "w_dd26", "w_hi52", "w_prev_hi52", "w_dd52",
    "w_sma30", "w_prev_sma30", "w_slope30", "w_px_sma30", "w_volx", "w_dvol13", "w_rsi14",
    "w_atr_pct", "w_low13", "w_low_prior", "w_higher_low", "m_close", "m_prev_close", "m_sma10",
    "m_prev_sma10", "m_px_sma10", "m_rsi14", "m_hi24", "m_dd24", "rs_line", "rs_ret26",
    "rs_prev_hi26", "rs_prev_hi52", "rs_new_hi26", "rs_new_hi52", "rs_lead26", "bm_w_px_sma30",
    "bm_m_px_sma10", "xs_rank_ret13", "xs_rank_ret26",
)

# the pre-registered signatures (setup_engine/mtf.py SIGNATURES), copied from the plan
_S3 = {"Weekly High (26w)": "New High", "Weekly Volume": "Over 1.5x 10-week average",
       "Weekly SMA30": "Price above", "Weekly SMA30 Slope": "Rising", "Monthly SMA10": "Price above"}
SIGNATURES = {
    "B0": {"Weekly SMA30": "Price above"},
    "S1": {"Performance Rank (13 Weeks)": "Top 20%", "Weekly SMA30": "Price above",
           "Weekly SMA30 Slope": "Rising"},
    "S2": {"Weekly High (52w)": "30% or more below", "Weekly SMA30": "Price above",
           "Performance Rank (26 Weeks)": "Top 20%"},
    "S3": dict(_S3),
    "S4": {**_S3, "RS Line": "New 26-week high", "Benchmark Weekly SMA30": "Price above"},
}

# every MTF column, set so that ALL of S1..S4 hold on a single row
_PASSING = {
    "w_close": 12.0, "w_sma30": 10.0, "w_prev_close": 11.0, "w_prev_sma30": 10.5, "w_slope30": 0.02,
    "m_close": 13.0, "m_sma10": 11.0, "m_prev_close": 12.0, "m_prev_sma10": 10.8,
    "w_prev_hi26": 11.5, "w_dd26": 0.0, "w_prev_hi52": 11.5, "w_dd52": -0.4,
    "w_volx": 2.0, "w_rsi14": 65.0, "m_rsi14": 60.0, "w_low13": 9.0, "w_low_prior": 8.0,
    "rs_ret26": 0.3, "rs_line": 1.3, "rs_prev_hi26": 1.2, "rs_prev_hi52": 1.25, "rs_lead26": 0.0,
    "bm_w_px_sma30": 0.03, "bm_m_px_sma10": 0.02, "xs_rank_ret13": 0.9, "xs_rank_ret26": 0.85,
}


def _daily(n=1800, seed=2, start="2014-01-02"):
    rng = np.random.default_rng(seed)
    close = 20 * np.exp(np.cumsum(rng.normal(0.0004, 0.02, n)))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.008, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.008, n)))
    vol = rng.integers(500_000, 3_000_000, n).astype(float)
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx)


def _mtf_nan(n=5):
    return pd.DataFrame(np.nan, index=range(n), columns=sorted(MTF_FILTER_COLUMNS))


def _row(**over):
    """One row where every MTF signature holds; ``over`` changes columns."""
    vals = {**_PASSING, **over}
    return pd.DataFrame({c: [vals[c]] for c in sorted(MTF_FILTER_COLUMNS)})


def _col_table(**cols):
    """A table with only the given columns (the filters under test read nothing else)."""
    return pd.DataFrame({k: np.asarray(v, dtype="float64") for k, v in cols.items()})


def _all_mtf_options():
    return [(f, o) for f, opts in MTF_FILTERS.items() for o in opts]


def _same_spec(got, want):
    assert len(got) == len(want), (got, want)
    for g, w in zip(got, want):
        if isinstance(w, str):
            assert g == w, (got, want)
        else:
            assert g == pytest.approx(w, rel=1e-12, abs=1e-12), (got, want)


# ── the vocabulary parses ─────────────────────────────────────────────────────

def test_every_mtf_example_parses_to_the_spec_stored_in_mtf_filters():
    assert len(MTF_FILTERS) == 15 and sum(len(o) for o in MTF_FILTERS.values()) == 69
    for flt, opt in _all_mtf_options():
        _same_spec(Screen({flt: opt})._specs[0], MTF_FILTERS[flt][opt])


@pytest.mark.parametrize("flt,opt,spec", [
    ("Weekly SMA30", "Price above", ("gt_col", "w_close", "w_sma30")),
    ("Weekly SMA30", "Price below", ("lt_col", "w_close", "w_sma30")),
    ("Weekly SMA30", "Price crossed above", ("cross_above", "w_close", "w_sma30", "w_prev_close", "w_prev_sma30")),
    ("Weekly SMA30", "Price crossed below", ("cross_below", "w_close", "w_sma30", "w_prev_close", "w_prev_sma30")),
    ("Weekly SMA30", "Price 10% above", ("ge_ratio", "w_close", "w_sma30", 1.1)),
    ("Weekly SMA30", "Price 10% below", ("le_ratio", "w_close", "w_sma30", 0.9)),
    ("Weekly SMA30 Slope", "Rising", ("gt", "w_slope30", 0.0)),
    ("Weekly SMA30 Slope", "Falling", ("lt", "w_slope30", 0.0)),
    ("Monthly SMA10", "Price above", ("gt_col", "m_close", "m_sma10")),
    ("Monthly SMA10", "Price crossed below", ("cross_below", "m_close", "m_sma10", "m_prev_close", "m_prev_sma10")),
    ("Weekly High (26w)", "New High", ("gt_col", "w_close", "w_prev_hi26")),
    ("Weekly High (52w)", "New High", ("gt_col", "w_close", "w_prev_hi52")),
    ("Weekly High (52w)", "30% or more below", ("le", "w_dd52", -0.3)),
    ("Weekly High (26w)", "20% or more below", ("le", "w_dd26", -0.2)),
    ("Weekly High (52w)", "0-5% below", ("between", "w_dd52", -0.05, 0.0)),
    ("Weekly Volume", "Over 1.5x 10-week average", ("gt", "w_volx", 1.5)),
    ("Weekly Volume", "Under 1x 10-week average", ("lt", "w_volx", 1.0)),
    ("Weekly RSI (14)", "Over 60", ("gt", "w_rsi14", 60.0)),
    ("Monthly RSI (14)", "Under 40", ("lt", "m_rsi14", 40.0)),
    ("Weekly Low", "Higher low", ("gt_col", "w_low13", "w_low_prior")),
    ("Weekly Low", "Lower low", ("lt_col", "w_low13", "w_low_prior")),
    ("RS vs Benchmark (26w)", "Over 25%", ("gt", "rs_ret26", 0.25)),
    ("RS vs Benchmark (26w)", "Under -25%", ("lt", "rs_ret26", -0.25)),
    ("RS Line", "New 26-week high", ("gt_col", "rs_line", "rs_prev_hi26")),
    ("RS Line", "New 52-week high", ("gt_col", "rs_line", "rs_prev_hi52")),
    ("RS Line", "New 26-week high before price", ("gt", "rs_lead26", 0.5)),
    ("Benchmark Weekly SMA30", "Price above", ("gt", "bm_w_px_sma30", 0.0)),
    ("Benchmark Monthly SMA10", "Price below", ("lt", "bm_m_px_sma10", 0.0)),
    ("Performance Rank (13 Weeks)", "Top 20%", ("ge", "xs_rank_ret13", 0.8)),
    ("Performance Rank (26 Weeks)", "Bottom 20%", ("le", "xs_rank_ret26", 0.2)),
    ("Performance Rank (26 Weeks)", "Top 10%", ("ge", "xs_rank_ret26", 0.9)),
])
def test_mtf_option_parses_to_the_documented_spec(flt, opt, spec):
    _same_spec(Screen({flt: opt})._specs[0], spec)


def test_options_beyond_the_dropdown_parse_and_nonsense_is_rejected():
    assert Screen({"Weekly Volume": "Over 2.5x 10-week average"})._specs[0] == ("gt", "w_volx", 2.5)
    assert Screen({"Weekly High (52w)": "7% or more below"})._specs[0][2] == pytest.approx(-0.07)
    assert Screen({"Performance Rank (13 Weeks)": "Top 33%"})._specs[0][2] == pytest.approx(0.67)
    for flt, opt in [("Performance Rank (13 Weeks)", "Top 20"),            # the % sign is required
                     ("Performance Rank (13 Weeks)", "Top -5%"),           # negative percentage
                     ("Weekly High (52w)", "-10% or more below"),
                     ("Weekly High (52w)", "0-5% above"),
                     ("Weekly Volume", "Over 1.5"),                        # not a 10-week multiple
                     ("Weekly SMA30", "Price above SMA30"),                # the daily wording
                     ("Monthly SMA10", "Price 10% above"),                 # monthly has no % forms
                     ("RS Line", "New 13-week high"),
                     ("Weekly SMA30 Slope", "Up")]:
        with pytest.raises(ValueError):
            Screen({flt: opt})


# ── NaN never matches ─────────────────────────────────────────────────────────

def test_nan_never_matches_any_mtf_example_option():
    ind = _mtf_nan(5)
    for flt, opt in _all_mtf_options():
        assert not Screen({flt: opt}).mask(ind).any(), f"{flt}: {opt} matched an all-NaN row"


def test_each_mtf_comparison_rejects_nan_on_its_own_without_help_from_valid():
    """Guards 'Price below' written as not-'Price above': the evaluator must be NaN-safe itself."""
    ind = _mtf_nan(5)
    for flt, opts in MTF_FILTERS.items():
        for opt, spec in opts.items():
            assert not S._evaluate(spec, ind).any(), f"{flt}: {opt} evaluates True on NaN"


def test_one_nan_input_is_enough_to_drop_the_row():
    ind = _row(w_prev_close=10.0, w_prev_sma30=10.5)          # a genuine cross above on row 0
    ind = pd.concat([ind, ind], ignore_index=True)
    ind.loc[1, "w_prev_sma30"] = np.nan                       # row 1 loses only the cross rule's 4th input
    scr = Screen({"Weekly SMA30": "Price crossed above"})
    assert scr.mask(ind).tolist() == [True, False]
    assert scr.valid(ind).tolist() == [True, False]


# ── boundaries ────────────────────────────────────────────────────────────────

def test_thirty_percent_below_the_52_week_high_boundaries():
    ind = _col_table(w_dd52=[-0.3 - 1e-12, -0.3, -0.3 + 1e-6, -0.29, -0.31, 0.0])
    got = Screen({"Weekly High (52w)": "30% or more below"}).mask(ind)
    assert got.tolist() == [True, True, False, False, True, False]


def test_zero_to_five_percent_below_boundaries():
    ind = _col_table(w_dd26=[0.0, -0.05, -0.0500001, -0.01, 1e-6, -0.2])
    got = Screen({"Weekly High (26w)": "0-5% below"}).mask(ind)
    assert got.tolist() == [True, True, False, True, False, False]


def test_top_and_bottom_rank_boundaries():
    ind = _col_table(xs_rank_ret13=[0.8, 0.8 - 1e-6, 1.0, 0.2, 0.05])
    assert Screen({"Performance Rank (13 Weeks)": "Top 20%"}).mask(ind).tolist() == [True, False, True, False, False]
    assert Screen({"Performance Rank (13 Weeks)": "Bottom 20%"}).mask(ind).tolist() == [False, False, False, True, True]
    assert Screen({"Performance Rank (13 Weeks)": "Top 20%"}).mask(_col_table(xs_rank_ret13=[0.8 - 1e-12])).tolist() == [True]
    # the two horizons read their own column
    assert Screen({"Performance Rank (26 Weeks)": "Top 20%"}).fields() == ["xs_rank_ret26"]


def test_weekly_sma30_cross_above_is_true_only_where_the_previous_pair_was_at_or_below():
    ind = _col_table(
        w_close=[11, 11, 11, 10, 9, 11, 11],
        w_sma30=[10, 10, 10, 10, 10, 10, 10],
        w_prev_close=[9, 10, 11, 9, 9, 9, np.nan],
        w_prev_sma30=[10, 10, 10, 10, 10, np.nan, 10])
    got = Screen({"Weekly SMA30": "Price crossed above"}).mask(ind)
    #            below   equal   above   not-yet  both-below  prev sma NaN  prev close NaN
    assert got.tolist() == [True, True, False, False, False, False, False]
    got_below = Screen({"Weekly SMA30": "Price crossed below"}).mask(ind)
    assert got_below.tolist() == [False, False, False, False, False, False, False]
    dn = _col_table(w_close=[9, 9, 9, 10], w_sma30=[10, 10, 10, 10],
                    w_prev_close=[11, 10, 9, 11], w_prev_sma30=[10, 10, 10, 10])
    assert Screen({"Weekly SMA30": "Price crossed below"}).mask(dn).tolist() == [True, True, False, False]


def test_price_percent_above_and_below_the_weekly_sma30():
    ind = _col_table(w_close=[11.0, 10.99, 9.0, 9.01], w_sma30=[10.0] * 4)
    assert Screen({"Weekly SMA30": "Price 10% above"}).mask(ind).tolist() == [True, False, False, False]
    assert Screen({"Weekly SMA30": "Price 10% below"}).mask(ind).tolist() == [False, False, True, False]


def test_strict_comparisons_exclude_equality():
    ind = _col_table(w_volx=[1.5, 1.5001, 1.4999], rs_line=[1.2, 1.2, 1.2], rs_prev_hi26=[1.2, 1.1, 1.3],
                     w_slope30=[0.0, 0.001, -0.001], w_low13=[5.0, 5.0, 5.0], w_low_prior=[5.0, 4.0, 6.0])
    assert Screen({"Weekly Volume": "Over 1.5x 10-week average"}).mask(ind).tolist() == [False, True, False]
    assert Screen({"RS Line": "New 26-week high"}).mask(ind).tolist() == [False, True, False]
    assert Screen({"Weekly SMA30 Slope": "Rising"}).mask(ind).tolist() == [False, True, False]
    assert Screen({"Weekly SMA30 Slope": "Falling"}).mask(ind).tolist() == [False, False, True]
    assert Screen({"Weekly Low": "Higher low"}).mask(ind).tolist() == [False, True, False]
    assert Screen({"Weekly Low": "Lower low"}).mask(ind).tolist() == [False, False, True]


def test_rs_lead_and_benchmark_filters():
    ind = _col_table(rs_lead26=[1.0, 0.0, np.nan], bm_w_px_sma30=[0.01, -0.01, 0.0],
                     bm_m_px_sma10=[0.5, -0.5, np.nan], rs_ret26=[0.30, 0.10, -0.30])
    assert Screen({"RS Line": "New 26-week high before price"}).mask(ind).tolist() == [True, False, False]
    assert Screen({"Benchmark Weekly SMA30": "Price above"}).mask(ind).tolist() == [True, False, False]
    assert Screen({"Benchmark Weekly SMA30": "Price below"}).mask(ind).tolist() == [False, True, False]
    assert Screen({"Benchmark Monthly SMA10": "Price below"}).mask(ind).tolist() == [False, True, False]
    assert Screen({"RS vs Benchmark (26w)": "Over 25%"}).mask(ind).tolist() == [True, False, False]
    assert Screen({"RS vs Benchmark (26w)": "Under -25%"}).mask(ind).tolist() == [False, False, True]


def test_monthly_sma10_and_rsi_filters():
    ind = _col_table(m_close=[12, 9, 11], m_sma10=[10, 10, 10], m_prev_close=[9, 11, 11],
                     m_prev_sma10=[10, 10, 10], m_rsi14=[71, 49.9, 30], w_rsi14=[55, 45, 25])
    assert Screen({"Monthly SMA10": "Price above"}).mask(ind).tolist() == [True, False, True]
    assert Screen({"Monthly SMA10": "Price crossed above"}).mask(ind).tolist() == [True, False, False]
    assert Screen({"Monthly SMA10": "Price crossed below"}).mask(ind).tolist() == [False, True, False]
    assert Screen({"Monthly RSI (14)": "Over 70"}).mask(ind).tolist() == [True, False, False]
    assert Screen({"Weekly RSI (14)": "Under 30"}).mask(ind).tolist() == [False, False, True]


# ── the daily vocabulary is untouched ─────────────────────────────────────────

def test_filters_and_available_filters_are_unchanged_by_the_mtf_vocabulary():
    assert set(available_filters()) & set(MTF_FILTERS) == set()
    assert set(FILTERS) & set(MTF_FILTERS) == set()
    assert len(S.FILTERS) == 27 and len(available_filters()) == 27
    assert sum(len(o) for o in FILTERS.values()) == 259
    assert list(FILTERS) == [
        "Price", "Average Volume", "Current Volume", "Dollar Volume 20d", "Relative Volume", "RSI (14)",
        "ATR %", "Average True Range", "Change", "Gap", "Performance (Week)", "Performance (Month)",
        "Performance (Quarter)", "Performance (Half Year)", "Performance (Year)", "SMA20", "SMA50",
        "SMA200", "20-Day High", "50-Day High", "52-Week High", "2-Year High", "20-Day Low",
        "50-Day Low", "52-Week Low", "2-Year Low", "Monthly Low"]
    assert available_filters() == {k: list(v) for k, v in FILTERS.items()}
    assert len(COLUMNS) == 52 and set(COLUMNS).isdisjoint(MTF_FILTER_COLUMNS)


def test_include_mtf_adds_the_mtf_filters_after_the_daily_ones():
    both = available_filters(include_mtf=True)
    assert "Weekly SMA30" in both and "Performance Rank (26 Weeks)" in both
    assert list(both) == list(FILTERS) + list(MTF_FILTERS)
    assert len(both) == 27 + 15
    assert all(both[name] == list(opts) for name, opts in MTF_FILTERS.items())
    assert available_filters(include_mtf=False) == available_filters()


def test_a_screen_can_mix_daily_and_mtf_filters():
    ind = _row()
    for c in ("close", "dollar_vol20"):
        ind[c] = [50.0]
    scr = Screen({"Price": "Over $5", "Dollar Volume 20d": "Over 1M", "Weekly SMA30": "Price above"})
    assert scr.fields() == ["close", "dollar_vol20", "w_close", "w_sma30"]
    ind["dollar_vol20"] = [2e6]
    assert scr.mask(ind).tolist() == [True]
    ind["dollar_vol20"] = [5e5]
    assert scr.mask(ind).tolist() == [False]


# ── a table without the MTF columns ───────────────────────────────────────────

def test_an_mtf_screen_on_a_daily_only_table_names_the_module_that_makes_the_columns():
    ind = indicators(_daily(300))
    scr = Screen({"Weekly SMA30": "Price above"})
    with pytest.raises(ValueError, match="indicators_mtf") as e:
        scr.mask(ind)
    assert "w_close" in str(e.value) and "w_sma30" in str(e.value)
    with pytest.raises(ValueError, match="indicators_mtf"):
        scr.valid(ind)
    with pytest.raises(ValueError, match="mtf_features"):
        Screen({"Price": "Over $5", "RS Line": "New 26-week high"}).mask(ind)
    with pytest.raises(ValueError, match="add_xs_ranks"):
        Screen({"Performance Rank (13 Weeks)": "Top 20%"}).mask(_mtf_nan(3).drop(columns="xs_rank_ret13"))


def test_a_missing_daily_column_still_fails_exactly_as_before():
    ind = indicators(_daily(300)).drop(columns="close")
    with pytest.raises(KeyError):
        Screen({"Price": "Over $5"}).mask(ind)
    assert Screen({}).mask(ind).all()                         # an empty screen still passes everything


# ── names, errors and text ────────────────────────────────────────────────────

def test_mtf_names_and_options_are_case_and_whitespace_insensitive():
    a = Screen({"Weekly SMA30": "Price above", "Performance Rank (13 Weeks)": "Top 20%"})
    b = Screen({"  weekly   sma30 ": " PRICE   Above ", "performance rank (13 WEEKS)": "top  20 %"})
    ind = _row()
    assert a.mask(ind).tolist() == b.mask(ind).tolist() == [True]
    assert a._specs == b._specs


def test_duplicate_mtf_filter_is_an_error_even_with_different_case():
    with pytest.raises(ValueError, match="given twice"):
        Screen({"Weekly SMA30": "Price above", "weekly sma30": "Price below"})
    with pytest.raises(ValueError, match="given twice"):
        Screen.from_text("Weekly SMA30: Price above; WEEKLY SMA30: Price below")
    assert Screen.from_text("Weekly SMA30: Price above; Weekly SMA30 Slope: Rising").fields() == [
        "w_close", "w_sma30", "w_slope30"]


def test_unknown_filter_lists_the_mtf_names_too():
    with pytest.raises(ValueError, match="Weekly SMA30") as e:
        Screen({"Foo": "Bar"})
    msg = str(e.value)
    assert "Price" in msg and "Monthly Low" in msg and "Performance Rank (13 Weeks)" in msg


def test_unknown_mtf_option_lists_examples_and_patterns():
    with pytest.raises(ValueError, match="Weekly SMA30") as e:
        Screen({"Weekly SMA30": "Price sideways"})
    msg = str(e.value)
    assert "Price crossed above" in msg and "Price <P> above" in msg


def test_describe_marks_the_completed_bar_and_the_benchmark_dependence():
    text = Screen({"Weekly SMA30": "Price above"}).describe()
    assert "(last COMPLETED weekly bar)" in text and "[w_close > w_sma30]" in text
    assert "(last COMPLETED monthly bar)" in Screen({"Monthly SMA10": "Price above"}).describe()
    assert "(needs a benchmark; NaN never matches)" in Screen({"RS Line": "New 26-week high"}).describe()
    assert "(cross-sectional rank among liquid stocks that week, added by the study)" in Screen(
        {"Performance Rank (13 Weeks)": "Top 20%"}).describe()
    for flt, opts in MTF_FILTERS.items():
        line = Screen({flt: next(iter(opts))}).describe()
        assert any(n in line for n in (
            "(last COMPLETED weekly bar)", "(last COMPLETED monthly bar)",
            "(needs a benchmark; NaN never matches)",
            "(cross-sectional rank among liquid stocks that week, added by the study)")), flt


def test_describe_leaves_the_existing_notes_alone():
    rules = {"Price": "Over $5", "Monthly Low": "Higher low", "Weekly Low": "Higher low"}
    lines = Screen(rules).describe().splitlines()
    assert lines[0] == "Price: Over $5  [close > 5]"
    assert "calendar-free stand-in" in lines[1] and "COMPLETED" not in lines[1]
    assert lines[2].endswith("(last COMPLETED weekly bar)") and "calendar-free" not in lines[2]


# ── the column contract ───────────────────────────────────────────────────────

def test_mtf_filter_columns_are_exactly_the_contract_set():
    assert isinstance(MTF_FILTER_COLUMNS, frozenset)
    assert MTF_FILTER_COLUMNS == frozenset({
        "w_close", "w_sma30", "w_prev_close", "w_prev_sma30", "w_slope30", "m_close", "m_sma10",
        "m_prev_close", "m_prev_sma10", "w_prev_hi26", "w_dd26", "w_prev_hi52", "w_dd52", "w_volx",
        "w_rsi14", "m_rsi14", "w_low13", "w_low_prior", "rs_ret26", "rs_line", "rs_prev_hi26",
        "rs_prev_hi52", "rs_lead26", "bm_w_px_sma30", "bm_m_px_sma10", "xs_rank_ret13",
        "xs_rank_ret26"})
    assert MTF_FILTER_COLUMNS <= set(_CONTRACT_COLUMNS)
    assert set(_PASSING) == set(MTF_FILTER_COLUMNS)


# ── the pre-registered signatures ─────────────────────────────────────────────

@pytest.mark.parametrize("name", sorted(SIGNATURES))
def test_every_signature_constructs_as_a_screen_over_known_columns(name):
    scr = Screen(SIGNATURES[name])
    assert set(scr.fields()) <= MTF_FILTER_COLUMNS
    assert set(SIGNATURES[name]) <= set(MTF_FILTERS)
    assert scr.mask(_row()).tolist() == [True]


def test_s4_is_s3_plus_the_two_market_rules():
    assert SIGNATURES["S4"] == {**SIGNATURES["S3"], "RS Line": "New 26-week high",
                                "Benchmark Weekly SMA30": "Price above"}
    assert len(SIGNATURES["S3"]) == 5 and len(SIGNATURES["S4"]) == 7


@pytest.mark.parametrize("name,breaks", [
    ("B0", [dict(w_close=9.0)]),
    ("S1", [dict(xs_rank_ret13=0.79), dict(w_close=9.0), dict(w_slope30=-0.01)]),
    ("S2", [dict(w_dd52=-0.29), dict(w_close=9.0), dict(xs_rank_ret26=0.79)]),
    ("S3", [dict(w_prev_hi26=12.0), dict(w_volx=1.5), dict(w_close=9.0), dict(w_slope30=0.0),
            dict(m_close=10.0)]),
    ("S4", [dict(w_prev_hi26=12.0), dict(w_volx=1.4), dict(w_close=9.0), dict(w_slope30=-0.5),
            dict(m_close=10.0), dict(rs_line=1.2), dict(bm_w_px_sma30=-0.01)]),
])
def test_a_signature_fails_when_any_single_condition_fails(name, breaks):
    scr = Screen(SIGNATURES[name])
    assert scr.mask(_row()).tolist() == [True]
    for change in breaks:
        assert scr.mask(_row(**change)).tolist() == [False], (name, change)


def test_s4_needs_the_market_rules_that_s3_does_not():
    assert Screen(SIGNATURES["S3"]).mask(_row(rs_line=1.2, bm_w_px_sma30=-0.05)).tolist() == [True]
    assert Screen(SIGNATURES["S4"]).mask(_row(rs_line=1.2, bm_w_px_sma30=-0.05)).tolist() == [False]
