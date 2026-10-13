"""setup_engine.artifacts: the corporate-action artifact detector.

The failures being guarded against: (1) the rule flagging real moves (a news crash on a volume burst) or
missing a plain re-quote at a new nominal price; (2) a boundary that silently moves (exactly 50%, exactly
$1M of prior dollar volume, exactly 10 prior days); (3) LOOK-AHEAD, where the volume baseline includes the
very day being tested and so hides its own spike; (4) a contamination window that is off by one at either
end, which would keep or drop the wrong trades in a sensitivity run.
"""
import numpy as np
import pandas as pd

from setup_engine import artifacts as A


def _frame(n=40, close=10.0, volume=200_000.0):
    """n business days of flat close and volume: prior dollar volume 2e6, prior 20-day mean volume 200k."""
    idx = pd.bdate_range("2020-01-01", periods=n, name="date")
    return pd.DataFrame({"close": np.full(n, close), "volume": np.full(n, volume)}, index=idx)


def _jump(to, at=30, **kw):
    """Flat frame whose close is `to` from row `at` on (so only row `at` has a return)."""
    df = _frame(**kw)
    df.iloc[at:, df.columns.get_loc("close")] = to
    return df


def _days(df):
    return list(A.artifact_days(df))


# ── the rule ────────────────────────────────────────────────────────────────────

def test_a_ten_to_one_requote_on_quiet_volume_is_flagged_on_that_day_only():
    """If this fails, the plain unadjusted-reverse-split case is missed (or the next flat days are flagged too)."""
    df = _jump(100.0)
    assert _days(df) == [df.index[30]]


def test_the_same_jump_on_a_five_times_volume_burst_is_not_flagged():
    """If this fails, real news moves (which draw volume) are thrown away as artifacts."""
    df = _jump(100.0)
    df.iloc[30, df.columns.get_loc("volume")] = 1_000_000.0
    assert _days(df) == []


def test_a_move_just_under_fifty_percent_is_not_flagged():
    """If this fails, MIN_ABS_RET is looser than 0.50."""
    assert _days(_jump(14.99)) == []


def test_a_move_of_exactly_plus_fifty_percent_is_flagged():
    """If this fails, the return threshold became strict (> instead of >=): 10 -> 15 is exactly 0.5 in float."""
    df = _jump(15.0)
    assert _days(df) == [df.index[30]]


def test_a_fall_of_exactly_fifty_percent_is_flagged():
    """If this fails, only up-moves are checked (the sign of the return matters)."""
    df = _jump(5.0)
    assert _days(df) == [df.index[30]]


def test_prior_dollar_volume_of_exactly_one_million_passes():
    """If this fails, the dollar-volume floor became strict: 10.0 x 100_000 is exactly 1e6."""
    df = _jump(100.0)
    df.iloc[29, df.columns.get_loc("volume")] = 100_000.0
    assert _days(df) == [df.index[30]]


def test_prior_dollar_volume_just_under_one_million_is_not_flagged():
    """If this fails, thin stocks (prior dollar volume 999,990) are scanned although the floor excludes them."""
    df = _jump(100.0)
    df.iloc[29, df.columns.get_loc("volume")] = 99_999.0
    assert _days(df) == []


def test_a_jump_with_only_nine_prior_days_is_not_flagged_and_with_ten_it_is():
    """If this fails, VOL_MIN_DAYS moved: the volume baseline would be trusted on too little history."""
    nine = _jump(100.0, at=9)
    ten = _jump(100.0, at=10)
    assert _days(nine) == []
    assert _days(ten) == [ten.index[10]]


def test_volume_below_twice_the_prior_mean_is_flagged_and_above_it_is_not():
    """If this fails, the volume test is missing or inverted: 390k < 2 x 200k is quiet, 410k is a burst."""
    quiet = _jump(100.0)
    quiet.iloc[30, quiet.columns.get_loc("volume")] = 390_000.0
    burst = _jump(100.0)
    burst.iloc[30, burst.columns.get_loc("volume")] = 410_000.0
    assert _days(quiet) == [quiet.index[30]]
    assert _days(burst) == []


def test_volume_of_exactly_twice_the_prior_mean_is_not_flagged():
    """If this fails, the volume test became <= : 400k is exactly 2 x 200k and counts as a burst (strict <)."""
    df = _jump(100.0)
    df.iloc[30, df.columns.get_loc("volume")] = 400_000.0
    assert _days(df) == []


def test_the_volume_baseline_excludes_the_day_itself():
    """LOOK-AHEAD GUARD. A same-day mean would be (19 x 200k + 410k) / 20 = 210.5k, 2x = 421k > 410k, which
    would flag a day that is a 2.05x burst against its true prior mean of 200k. Prior-only: not flagged."""
    df = _jump(100.0)
    df.iloc[30, df.columns.get_loc("volume")] = 410_000.0
    assert _days(df) == []


def test_a_date_column_gives_the_same_answer_as_a_date_index():
    """If this fails, frames read from parquet (date as a column) are scanned differently from indexed ones."""
    df = _jump(100.0)
    flat = df.reset_index()
    assert list(A.artifact_days(flat)) == list(A.artifact_days(df))
    assert A.artifact_days(flat).name == "date"


def test_missing_prices_or_volume_never_flag_a_day():
    """If this fails, NaN comparisons leak into the flags and a gap in the data looks like a split."""
    df = _jump(100.0)
    df.iloc[30, df.columns.get_loc("volume")] = np.nan
    assert _days(df) == []
    df2 = _jump(100.0)
    df2.iloc[29, df2.columns.get_loc("close")] = np.nan
    assert _days(df2) == []


def test_a_flat_stock_has_no_artifact_days_and_the_result_is_an_empty_index():
    """If this fails, the detector invents days on quiet data."""
    out = A.artifact_days(_frame())
    assert len(out) == 0 and isinstance(out, pd.DatetimeIndex)


# ── contamination windows ───────────────────────────────────────────────────────

_D = pd.DatetimeIndex(["2020-01-10"])


def test_contaminated_includes_both_ends_of_the_window():
    """If this fails, a window that STARTS or ENDS on the artifact day misses it (off-by-one)."""
    assert A.contaminated(_D, "2020-01-10", "2020-02-01") is True       # start inclusive
    assert A.contaminated(_D, "2020-01-01", "2020-01-10") is True       # end inclusive
    assert A.contaminated(_D, "2020-01-11", "2020-02-01") is False      # starts the day after
    assert A.contaminated(_D, "2020-01-01", "2020-01-09") is False      # ends the day before


def test_contaminated_with_no_artifact_days_is_false():
    """If this fails, a stock with no flagged days is reported as contaminated."""
    assert A.contaminated(pd.DatetimeIndex([]), "2000-01-01", "2030-01-01") is False


def test_contaminated_mask_agrees_with_the_scalar_rule_on_hand_cases():
    """If this fails, the vectorised path (used on 100k+ rows) disagrees with the readable one."""
    days = {"A": pd.DatetimeIndex(["2020-03-20", "2020-01-10", "2021-06-01"]),    # unsorted on purpose
            "B": pd.DatetimeIndex([]), "C": pd.DatetimeIndex(["2019-12-31"])}
    syms = ["A", "A", "A", "B", "C", "ZZZ"]
    starts = pd.to_datetime(["2020-01-10", "2020-01-11", "2020-03-21", "2000-01-01", "2020-01-01", "2000-01-01"])
    ends = pd.to_datetime(["2020-01-10", "2020-03-19", "2021-05-31", "2030-01-01", "2020-12-31", "2030-01-01"])
    expected = [True,           # window is exactly the artifact day 2020-01-10
                False,          # 2020-01-11 .. 2020-03-19: between the two days
                False,          # 2020-03-21 .. 2021-05-31: after 03-20, before 06-01
                False,          # symbol with no days
                False,          # artifact day is before the window
                False]          # unknown symbol
    got = A.contaminated_mask(days, syms, starts, ends)
    assert got.dtype == bool and list(got) == expected
    for s, a, b, e in zip(syms, starts, ends, expected):
        assert A.contaminated(days.get(s, pd.DatetimeIndex([])), a, b) is e
    # and the positive twin: shift one window so it catches 2021-06-01 on its LAST day
    assert list(A.contaminated_mask(days, ["A"], pd.to_datetime(["2020-03-21"]), pd.to_datetime(["2021-06-01"]))) == [True]


def test_contaminated_mask_on_no_rows_returns_an_empty_bool_array():
    """If this fails, an empty trade table crashes the sensitivity run."""
    out = A.contaminated_mask({"A": _D}, [], pd.to_datetime([]), pd.to_datetime([]))
    assert out.dtype == bool and len(out) == 0


def test_contaminated_mask_never_flags_a_missing_date():
    """If this fails, a NaT start or end (a row with no date) is treated as contaminated."""
    out = A.contaminated_mask({"A": _D}, ["A", "A"], pd.to_datetime([pd.NaT, "2020-01-01"]),
                              pd.to_datetime(["2030-01-01", pd.NaT]))
    assert list(out) == [False, False]
