"""scripts/scanner_recall.py must turn each existing scanner into flags that mean what the scanner means, must
not read the future or flatter a scanner, and the runner must report the precision proxy honestly.

The failures these tests guard against, each mutation-tested:
  * the turnaround bottom state read on the breakout bar itself instead of the bar before (the breakout lifts the
    close and so changes the state it is meant to be conditioned on);
  * a daily flag attributed to the wrong week (a Sunday-ending week puts a Wednesday flag on the following
    Sunday, which matches no Friday label, so the recall window silently loses it);
  * a recall window that is wider or narrower than the pre-registered 8 weeks before / 52 weeks after the low;
  * a missing value treated as a match in a registered rule (a stock without enough history would then be
    "flagged" by a rule that cannot be evaluated on it);
  * every Finder preset silently evaluated as the "signal" preset;
  * the constants drifting away from the pre-registered turnaround study or the dataset the registered rules
    were learned on.
Part 2 (the runner) adds:
  * the "how many OTHER stocks flagged it that week" count including stocks that are not liquid (an illiquid
    stock flagged by the same scanner would make a scanner look more selective than it is), or the cohort
    names themselves / extras outside the ranking universe;
  * the registered rules' rs_ ranks computed among the cohort names only (every cohort name would then rank
    near the top of a universe of 46 winners, so every rs_ rule would fire on them);
  * the low week taken from the first week of data instead of the ex-post best run;
  * the recall table reading bars after the data's last date, or a stock with a bad bar crashing a long run
    silently (it is counted and reported) or a cohort name's failure being swallowed.
No test touches the network, the price cache or the real ledger (learn.REGISTRY_DIR is redirected to a tmp dir
holding a COPY of the registry, so reading the registered setups still works; the tests that run the whole
script write parquet and so skip without pyarrow).
"""
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from setup_engine import features as FT
from setup_engine import learn as L
from setup_engine import mtf
from setup_engine.screen import Screen, clean_daily, indicators

ROOT = Path(__file__).resolve().parents[2]
REAL_REGISTRY = ROOT / "setup_engine" / "registry" / "registry.json"


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


R = _load("scanner_recall", "scanner_recall.py")


@pytest.fixture(autouse=True)
def _registry_in_tmp(tmp_path, monkeypatch):
    """Redirect the registry/ledger dir to tmp, seeded with a copy of the real registry (read-only use)."""
    d = tmp_path / "registry"
    d.mkdir()
    shutil.copy(REAL_REGISTRY, d / "registry.json")
    monkeypatch.setattr(L, "REGISTRY_DIR", d)


def _finder_or_skip():
    try:
        from scanner_engine.finder import PRESETS
        from indicator_engine.calculator import IndicatorCalculator  # noqa: F401
        from strategy_validation import finder_backtest as FB
    except Exception as exc:                                    # noqa: BLE001 - any import failure means skip
        pytest.skip(f"the Finder or its indicator engine cannot be imported here: {exc}")
    return PRESETS, FB


# ── synthetic stocks ──────────────────────────────────────────────────────────────────────────────

def _bars(o, h, l, c, v, start="2010-01-04"):
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v},
                        index=pd.bdate_range(start, periods=len(c))).astype(float)


def _turnaround_stock(n=760, start="2010-01-04", extra_spikes=()):
    """Copy of tests/test_study/test_study_turnaround.py::_turnaround_stock.

    Flat at 100, falls to 20 by bar 400 (low 19), basing at 22, then a high-volume breakout on bar 700."""
    o, h, l, c = (np.empty(n) for _ in range(4))
    v = np.full(n, 1e6)
    o[:300] = c[:300] = 100.0
    h[:300], l[:300] = 101.0, 99.0
    c[300:401] = np.linspace(100.0, 20.0, 101)
    o[300:401] = c[300:401]
    h[300:401], l[300:401] = c[300:401] * 1.01, c[300:401] * 0.99
    l[400] = 19.0
    o[401:700] = c[401:700] = 22.0
    h[401:700], l[401:700] = 22.2, 21.8
    o[700], h[700], l[700], c[700], v[700] = 22.0, 33.0, 22.0, 32.0, 5e6
    o[701:] = c[701:] = 32.0
    h[701:], l[701:] = 32.2, 31.8
    for s, hi in extra_spikes:                      # (bar, high): a breakout inside the base
        o[s], h[s], l[s], c[s], v[s] = 22.0, hi, 22.0, hi - 0.1, 5e6
    return _bars(o, h, l, c, v, start)


def _walk(n, seed, p0=10.0, drift=0.0008, sig=0.01, vol=1e6, start="2015-01-05"):
    """A seeded random walk with realistic-looking bars. At p0=3, drift=0.0012, seed=0 it crosses the Finder's
    sub-$5 and $5-20 price bands, so the three presets flag different days."""
    rng = np.random.default_rng(seed)
    close = p0 * np.exp(np.cumsum(rng.normal(drift, sig, n)))
    open_ = np.r_[p0, close[:-1]] * np.exp(rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, n)))
    volume = vol * rng.uniform(0.8, 1.2, n)
    return _bars(open_, high, low, close, volume, start)


# ── 1. the pre-registered constants ───────────────────────────────────────────────────────────────

def test_constants_equal_the_turnaround_study_and_the_dataset_they_were_learned_on():
    S = _load("study_turnaround_for_recall", "study_turnaround.py")
    assert R.TURNAROUND_BOTTOM == S.BOTTOM == {"2-Year High": "70% or more below", "Monthly Low": "Higher low"}
    assert R.TURNAROUND_TRIGGER == S.TRIGGER == {"20-Day High": "New High", "Relative Volume": "Over 2"}
    assert R.TURNAROUND_LIQUIDITY == S.LIQUIDITY == {"Price": "Over $1", "Dollar Volume 20d": "Over 1M"}
    primary = S.CONFIGS[0]
    assert (R.TURNAROUND_BOTTOM, R.TURNAROUND_TRIGGER, R.TURNAROUND_LIQUIDITY) == \
        (primary.bottom, primary.trigger, primary.liquidity)
    assert primary.bottom_lag == 1
    for rules in (R.TURNAROUND_BOTTOM, R.TURNAROUND_TRIGGER, R.TURNAROUND_LIQUIDITY):
        Screen(rules)                                                  # every screen constructs
    B = _load("build_setup_dataset_for_recall", "build_setup_dataset.py")
    assert R.REGISTRY_MIN_DOLLAR_VOLUME == B.MIN_DOLLAR_VOLUME == 5e5


def test_the_recall_window_and_presets_are_the_pre_registered_ones():
    assert (R.RECALL_BEFORE_WEEKS, R.RECALL_AFTER_WEEKS) == (8, 52)
    assert R.FINDER_PRESETS == ("signal", "pennyking", "maker")
    assert "golden cross" in R.FINDER_DESCRIPTION and "compress" in R.FINDER_DESCRIPTION


def test_the_finder_presets_named_here_exist_in_the_finder():
    presets, _ = _finder_or_skip()
    assert set(R.FINDER_PRESETS) <= set(presets)


# ── 2. the turnaround rule: the bottom is read on the bar BEFORE the breakout ─────────────────────

def test_turnaround_flags_the_breakout_bar_and_nothing_else():
    df = _turnaround_stock()
    flags = R.turnaround_daily_flags(df)
    assert flags.dtype == bool and len(flags) == len(df)
    assert np.flatnonzero(flags).tolist() == [700]
    # why the same-bar reading would miss it: the breakout lifts the close to 68.3% below the 2-year high
    ind = indicators(clean_daily(df))
    assert ind["pct_below_high_2y"].iloc[699] > 0.7 >= ind["pct_below_high_2y"].iloc[700]


def test_turnaround_flags_equal_the_study_c0_arm_on_a_stock_with_two_breakouts():
    S = _load("study_turnaround_for_recall", "study_turnaround.py")
    df = _turnaround_stock(extra_spikes=((600, 23.0),))
    flags = R.turnaround_daily_flags(df)
    c0 = S.arm_masks(indicators(clean_daily(df)), S.CONFIGS[0], None)["C0"]
    assert np.flatnonzero(flags).tolist() == [600, 700]
    assert np.array_equal(flags, c0)


def test_turnaround_flags_do_not_change_when_later_bars_are_cut_off():
    df = _turnaround_stock(extra_spikes=((600, 23.0),))
    full = R.turnaround_daily_flags(df)
    for cut in (601, 650, 701):
        assert np.array_equal(R.turnaround_daily_flags(df.iloc[:cut]), full[:cut])


# ── 3. the Finder presets ─────────────────────────────────────────────────────────────────────────

def test_finder_flags_are_all_false_below_the_engines_minimum_history():
    _, FB = _finder_or_skip()
    short = _walk(FB.MIN_ROWS - 1, seed=0, p0=3.0, drift=0.0012)
    flags = R.finder_daily_flags(short)
    assert list(flags) == list(R.FINDER_PRESETS)
    for arr in flags.values():
        assert arr.dtype == bool and len(arr) == len(short) and not arr.any()


def test_finder_flags_equal_build_signals_for_each_preset_and_the_presets_really_differ():
    presets, FB = _finder_or_skip()
    df = _walk(700, seed=0, p0=3.0, drift=0.0012)
    flags = R.finder_daily_flags(df)
    feats = FB.compute_features(df)
    for p in R.FINDER_PRESETS:
        want = FB.build_signals(feats, FB.Rule.from_criteria(presets[p])).signal
        assert flags[p].dtype == bool and len(flags[p]) == len(df)
        assert np.array_equal(flags[p], want), p
    # the test only means something if the presets disagree: price bands and gap limits must matter here
    for p in R.FINDER_PRESETS:
        assert flags[p].sum() > 0, f"{p} never fires on the test series"
    assert not np.array_equal(flags["signal"], flags["pennyking"])
    assert not np.array_equal(flags["signal"], flags["maker"])
    assert not np.array_equal(flags["pennyking"], flags["maker"])


def test_finder_flags_accept_a_subset_of_presets_and_reject_an_unknown_one():
    _finder_or_skip()
    df = _walk(700, seed=0, p0=3.0, drift=0.0012)
    only = R.finder_daily_flags(df, presets=("maker",))
    assert list(only) == ["maker"] and only["maker"].sum() > 0
    with pytest.raises(ValueError, match="nonsense"):
        R.finder_daily_flags(df, presets=("nonsense",))


def test_finder_flags_do_not_change_when_later_bars_are_cut_off():
    _finder_or_skip()
    df = _walk(700, seed=0, p0=3.0, drift=0.0012)
    full = R.finder_daily_flags(df)
    for cut in (450, 600):
        part = R.finder_daily_flags(df.iloc[:cut])
        for p in R.FINDER_PRESETS:
            assert np.array_equal(part[p], full[p][:cut]), (p, cut)


# ── 4. from days to weeks ─────────────────────────────────────────────────────────────────────────

def test_a_wednesday_flag_belongs_to_that_weeks_friday():
    dates = pd.bdate_range("2024-01-01", periods=30)                  # Monday 2024-01-01
    arr = np.zeros(len(dates), dtype=bool)
    arr[list(dates).index(pd.Timestamp("2024-01-03"))] = True        # a Wednesday
    any_w, first_w = R.to_weekly(dates, {"x": arr})
    assert any_w.index.name == "week"
    assert pd.Timestamp("2024-01-05") in any_w.index
    assert any_w.index.tolist() == list(pd.date_range("2024-01-05", periods=6, freq="7D"))
    assert any_w["x"].tolist() == [True, False, False, False, False, False]
    assert first_w.loc["2024-01-05", "x"] == pd.Timestamp("2024-01-03")
    assert first_w["x"].iloc[1:].isna().all()
    assert any_w["x"].dtype == bool


def test_two_flags_in_one_week_make_one_true_week_with_the_earlier_first_date():
    dates = pd.bdate_range("2024-01-01", periods=15)
    arr = np.zeros(len(dates), dtype=bool)
    for d in ("2024-01-09", "2024-01-11"):                           # Tuesday and Thursday of one week
        arr[list(dates).index(pd.Timestamp(d))] = True
    other = np.zeros(len(dates), dtype=bool)
    other[0] = True
    any_w, first_w = R.to_weekly(dates, {"a": arr, "b": other})
    assert any_w["a"].tolist() == [False, True, False]
    assert any_w["a"].sum() == 1
    assert first_w.loc["2024-01-12", "a"] == pd.Timestamp("2024-01-09")
    assert first_w["a"].isna().sum() == 2
    assert any_w["b"].tolist() == [True, False, False]
    assert first_w.loc["2024-01-05", "b"] == pd.Timestamp("2024-01-01")


def test_to_weekly_rejects_a_flag_array_of_the_wrong_length():
    dates = pd.bdate_range("2024-01-01", periods=10)
    with pytest.raises(ValueError, match="10"):
        R.to_weekly(dates, {"x": np.zeros(9, dtype=bool)})


# ── 5. the recall window ──────────────────────────────────────────────────────────────────────────

def _weekly(true_at=()):
    idx = pd.date_range("2021-01-01", "2023-12-29", freq="W-FRI", name="week")
    s = pd.Series(False, index=idx)
    for d in true_at:
        s.loc[d] = True
    return s


LOW = pd.Timestamp("2022-06-03")                                      # a Friday


def test_first_flag_window_boundaries_are_eight_weeks_before_and_fifty_two_after():
    w = pd.Timedelta(weeks=1)
    assert R.first_flag(_weekly([LOW - 9 * w]), LOW) is None
    assert R.first_flag(_weekly([LOW - 8 * w]), LOW) == LOW - 8 * w
    assert R.first_flag(_weekly([LOW + 52 * w]), LOW) == LOW + 52 * w
    assert R.first_flag(_weekly([LOW + 53 * w]), LOW) is None
    assert R.first_flag(_weekly([LOW]), LOW) == LOW
    assert R.first_flag(_weekly(), LOW) is None


def test_first_flag_returns_the_earliest_flag_inside_the_window():
    w = pd.Timedelta(weeks=1)
    s = _weekly([LOW - 9 * w, LOW - 8 * w, LOW + 5 * w, LOW + 53 * w])
    assert R.first_flag(s, LOW) == LOW - 8 * w
    assert R.first_flag(s, LOW, before=0, after=10) == LOW + 5 * w
    assert R.first_flag(s, LOW, before=9, after=52) == LOW - 9 * w


# ── 6. registered setups: flags from the rule, NaN never flags ───────────────────────────────────

def test_registry_flags_follow_the_rule_and_a_missing_value_never_flags():
    setups = [{"id": "r1-07", "rule": [{"feature": "W_mom30", "op": "<=", "value": -0.7218}]},
              {"id": "two", "rule": [{"feature": "W_mom30", "op": ">", "value": -0.7207},
                                     {"feature": "W_dist_hi50", "op": "<=", "value": -0.7233}]}]
    obs = pd.DataFrame({"W_mom30": [-0.9, -0.7218, -0.72, np.nan, 0.1, -0.5],
                        "W_dist_hi50": [-0.8, -0.8, -0.8, -0.8, np.nan, -0.5]})
    f = R.registry_flags(obs, setups)
    assert list(f.columns) == ["r1-07", "two"] and f.index.equals(obs.index)
    assert f.dtypes.eq(bool).all()
    assert f["r1-07"].tolist() == [True, True, False, False, False, False]
    assert f["two"].tolist() == [False, False, True, False, False, False]
    assert f["r1-07"].tolist() == (obs["W_mom30"] <= -0.7218).tolist()    # NaN compares False in pandas too
    all_nan = pd.DataFrame({"W_mom30": [np.nan] * 3, "W_dist_hi50": [np.nan] * 3})
    assert not R.registry_flags(all_nan, setups).to_numpy().any()


def test_the_registered_r1_07_rule_is_the_one_these_tests_assume():
    by_id = {s["id"]: s for s in R.registry_setups()}
    assert by_id["r1-07"]["rule"] == [{"feature": "W_mom30", "op": "<=", "value": -0.7218}]
    obs = pd.DataFrame({"W_mom30": [-0.9, -0.7218, -0.72, np.nan]})
    assert R.registry_flags(obs, [by_id["r1-07"]])["r1-07"].tolist() == [True, True, False, False]


def test_registry_flags_name_the_missing_column_when_ranks_were_not_added():
    setups = [s for s in R.registry_setups() if any(c["feature"].startswith("rs_") for c in s["rule"])]
    assert setups
    with pytest.raises(ValueError, match="add_relative_strength"):
        R.registry_flags(pd.DataFrame({"W_mom30": [0.0]}), setups[:1])


# ── 7. the registry's setups and features ─────────────────────────────────────────────────────────

def test_registry_setups_returns_the_twenty_evaluable_setups():
    setups = R.registry_setups()
    ids = [s["id"] for s in setups]
    assert len(ids) == len(set(ids)) == 20
    empty_textbook = {"r1-textbook_compression_golden_cross", "r1-textbook_stage2_trend",
                      "r1-textbook_tight_base_breakout"}
    assert not empty_textbook & set(ids)
    assert all(s["rule"] for s in setups)
    assert sum(s["status"] == "rejected_robustness" for s in setups) == 17
    assert sum(s["status"] == "approved" for s in setups) == 3
    # all 23 registered entries are accounted for
    assert len(L.load_registry()["setups"]) == 23


def test_registry_setups_drops_entries_with_an_empty_rule_and_keeps_order():
    reg = {"setups": [{"id": "a", "rule": [{"feature": "W_mom30", "op": ">", "value": 0}]},
                      {"id": "b", "rule": []},
                      {"id": "c", "rule": [{"feature": "D_mom50", "op": "<=", "value": 1}]},
                      {"id": "d"}]}
    assert [s["id"] for s in R.registry_setups(reg)] == ["a", "c"]


def test_registry_features_are_the_sorted_union_without_the_cross_sectional_ranks():
    setups = [{"id": "a", "rule": [{"feature": "W_mom30", "op": ">", "value": 0},
                                   {"feature": "rs_W_mom30", "op": "<=", "value": 0.5}]},
              {"id": "b", "rule": [{"feature": "D_atr_pctl100", "op": ">", "value": 0.5},
                                   {"feature": "W_mom30", "op": "<=", "value": 1},
                                   {"feature": "rs_W_mom50", "op": ">", "value": 0.1}]}]
    assert R.registry_features(setups) == ["D_atr_pctl100", "W_mom30"]
    feats = R.registry_features(R.registry_setups())
    assert feats == sorted(set(feats)) and feats
    assert not any(f.startswith("rs_") for f in feats)
    assert set(feats) <= set(f"{tf}_{b}" for tf in FT.TIMEFRAMES for b in FT.BAR_FEATURES)


# ── 8. registry observations: no labels, only the given Fridays ──────────────────────────────────

def test_registry_obs_has_no_forward_columns_and_only_the_given_fridays():
    df = _walk(1000, seed=3, p0=10.0)
    fridays = pd.date_range("2016-01-01", "2018-12-28", freq="W-FRI")
    obs = R.registry_obs("AAA", df, fridays)
    assert obs is not None and len(obs) > 50
    assert not [c for c in obs.columns if c.startswith("fwd_")]
    assert obs["date"].isin(fridays).all() and obs["date"].is_unique
    assert (obs["symbol"] == "AAA").all()
    assert obs.columns.is_unique
    want = ["symbol", "date", "dvol50", *R.registry_features(R.registry_setups())]
    assert [c for c in obs.columns if c in want] == [c for c in want if c in obs.columns]
    assert set(want) <= set(obs.columns)
    assert set(FT.RS_SOURCES) <= set(obs.columns)
    assert set(obs.columns) == set(want) | set(FT.RS_SOURCES)


def test_registry_obs_features_argument_selects_columns_and_a_bad_one_is_refused():
    df = _walk(1000, seed=3, p0=10.0)
    fridays = pd.date_range("2016-01-01", "2018-12-28", freq="W-FRI")
    obs = R.registry_obs("AAA", df, fridays, features=["D_mom30"])
    assert list(obs.columns) == ["symbol", "date", "dvol50", "D_mom30", *FT.RS_SOURCES]
    with pytest.raises(ValueError, match="not_a_feature"):
        R.registry_obs("AAA", df, fridays, features=["not_a_feature"])


def test_registry_obs_applies_the_dollar_volume_floor_and_needs_enough_history():
    fridays = pd.date_range("2016-01-01", "2018-12-28", freq="W-FRI")
    thin = _walk(1000, seed=3, p0=10.0, vol=1000.0)                   # ~1e4 dollars a day, below the 5e5 floor
    assert R.registry_obs("THIN", thin, fridays) is None
    assert R.registry_obs("TINY", _walk(100, seed=3), fridays) is None


def test_registry_obs_agree_with_features_build_observations_on_the_kept_columns():
    df = _walk(1000, seed=5, p0=10.0)
    fridays = pd.date_range("2016-01-01", "2018-12-28", freq="W-FRI")
    got = R.registry_obs("AAA", df, fridays)
    ref = FT.build_observations("AAA", df, fridays, 5e5, with_labels=False)
    pd.testing.assert_frame_equal(got, ref[list(got.columns)].reset_index(drop=True))


def test_registry_flags_run_end_to_end_on_ranked_observations():
    fridays = pd.date_range("2016-01-01", "2018-12-28", freq="W-FRI")
    panel = pd.concat([R.registry_obs(f"S{i}", _walk(1000, seed=10 + i, p0=10.0, drift=-0.0003 * (i % 3)), fridays)
                       for i in range(6)], ignore_index=True)
    ranked = FT.add_relative_strength(panel)
    setups = R.registry_setups()
    flags = R.registry_flags(ranked, setups)
    assert list(flags.columns) == [s["id"] for s in setups] and len(flags) == len(ranked)
    assert flags.dtypes.eq(bool).all()


# ═════════════════════════════════════════════════════════════════════════════════════════════════
# Part 2: the runner
# ═════════════════════════════════════════════════════════════════════════════════════════════════

COHORT_46 = [
    "AMD", "NVDA", "MSFT", "AMZN", "TSLA", "AVGO", "MU", "QCOM", "INTC", "TXN", "AMAT", "LRCX", "KLAC", "MRVL",
    "ON", "NXPI", "MCHP", "ADI", "MPWR", "TER", "SMCI", "TSM", "ASML", "ARM", "ALAB", "CRDO", "COHR", "LITE",
    "ANET", "VRT", "WDC", "STX", "DELL", "GLW", "CIEN", "CLS", "SNDK", "NBIS", "PLTR", "APP", "META", "GOOGL",
    "AAPL", "SPY", "QQQ", "SMH"]


def _strict_json(path):
    def bad(c):
        raise ValueError(f"non-strict JSON constant {c}")
    return json.loads(Path(path).read_text(), parse_constant=bad)


def _wk(date):
    return pd.Timestamp(mtf.week_label([pd.Timestamp(date)])[0])


# ── constants and labels ──────────────────────────────────────────────────────────────────────────

def test_the_private_week_label_equals_the_mtf_week_label_on_random_business_dates():
    rng = np.random.default_rng(11)
    days = pd.bdate_range("2005-01-03", "2026-10-02")
    dates = pd.DatetimeIndex(days[rng.integers(0, len(days), 600)])
    assert dates.dayofweek.nunique() == 5                      # Monday to Friday are all exercised
    assert R._week_label(dates).equals(mtf.week_label(dates))
    assert R._week_label(dates).dayofweek.isin([4]).all()


def test_the_cohort_is_the_46_names_of_the_case_study_in_order():
    assert list(R.COHORT_RECALL) == COHORT_46
    assert len(set(R.COHORT_RECALL)) == len(R.COHORT_RECALL) == 46
    assert set(R.USER_NAMES) <= set(R.COHORT_RECALL)
    assert set(R.NOT_IN_CACHE) <= set(R.COHORT_RECALL)
    assert set(R.BENCH_NAMES) <= set(R.NOT_IN_CACHE)
    assert (R.RUN_START, R.RECALL_START, R.RS_BENCH) == ("2021-10-08", "2021-06-01", "SPY")
    assert pd.Timestamp(R.RECALL_START) < pd.Timestamp(R.RUN_START) - pd.Timedelta(weeks=R.RECALL_BEFORE_WEEKS)


def test_the_signature_inputs_cover_every_column_the_five_signatures_read():
    assert list(R.SIGNATURE_NAMES) == ["S1", "S2", "S3", "S4", "B0"] and set(R.SIGNATURE_NAMES) == set(mtf.SIGNATURES)
    need = {c for n in R.SIGNATURE_NAMES for c in Screen(mtf.SIGNATURES[n]).fields()}
    assert need - set(mtf.XS_COLUMNS) <= set(R.MTF_INPUT_COLUMNS)
    assert {"w_close", "w_ret13", "w_ret26"} <= set(R.MTF_INPUT_COLUMNS)
    assert not set(R.MTF_INPUT_COLUMNS) & set(mtf.XS_COLUMNS)
    assert set(R.MTF_INPUT_COLUMNS) <= set(mtf.MTF_COLUMNS)


def test_scanner_keys_are_3_finder_1_turnaround_20_registry_5_mtf_in_that_order():
    setups = R.registry_setups()
    keys = R.scanner_keys(setups)
    assert len(keys) == len(set(keys)) == 29
    assert keys[:4] == ["finder_signal", "finder_pennyking", "finder_maker", "turnaround"]
    assert keys[4:24] == [f"registry:{s['id']}" for s in setups]
    assert keys[24:] == ["mtf:S1", "mtf:S2", "mtf:S3", "mtf:S4", "mtf:B0"]


def test_output_paths_a_limited_run_never_defaults_to_docs(tmp_path):
    assert R.output_paths(None, False) == (R.OUT_JSON, R.OUT_MD)
    assert R.output_paths(None, True) == (R.SMOKE_DIR / "scanner_recall.json", R.SMOKE_DIR / "scanner_recall.md")
    assert "docs" not in R.SMOKE_DIR.parts and R.SMOKE_DIR.parts[-2:] == ("mtf", "recall_smoke")
    assert R.output_paths(str(tmp_path), True) == (tmp_path / "scanner_recall.json", tmp_path / "scanner_recall.md")
    assert R.output_paths(str(tmp_path), False)[0].parent == tmp_path


def test_jsonable_gives_strict_json_for_nan_inf_timestamps_and_numpy():
    obj = {"a": float("nan"), "b": np.float64("inf"), "c": pd.Timestamp("2022-10-14"), "d": pd.NaT,
           "e": np.int64(3), "f": np.bool_(True), "g": [np.float32(1.5), None], "h": {1: 2}}
    out = json.loads(json.dumps(R._jsonable(obj), allow_nan=False))
    assert out == {"a": None, "b": None, "c": "2022-10-14", "d": None, "e": 3, "f": True, "g": [1.5, None],
                   "h": {"1": 2}}


# ── hand panels ───────────────────────────────────────────────────────────────────────────────────

def _blank_weekly(symbol, weeks, liquid=True, in_universe=True, **cols):
    """A panel fragment shaped like recall_rows' output: every input column NaN, every flag False, plus overrides."""
    weeks = pd.DatetimeIndex(weeks)
    d = {"symbol": symbol, "week": weeks}
    for c in R.MTF_INPUT_COLUMNS:
        d[c] = np.nan
    d["liquid"] = liquid
    for k in ("finder_signal", "finder_pennyking", "finder_maker", "turnaround"):
        d[k] = False
        d[f"first_{k}"] = pd.NaT
    d.update(cols)
    df = pd.DataFrame({k: (v if np.ndim(v) else [v] * len(weeks)) for k, v in d.items()})
    df["in_universe"] = in_universe
    return df


def _part(weekly=None, registry=None, symbol="X"):
    return {"symbol": symbol, "weekly": weekly, "registry": registry, "info": None, "error": None}


def test_weekly_ranks_use_the_liquid_universe_only_and_extras_are_ranked_against_it():
    w = pd.Timestamp("2023-06-02")
    rows = [_blank_weekly(f"U{i}", [w], w_ret13=0.125 * (i + 1), w_ret26=0.5 * (i + 1)) for i in range(4)]
    rows.append(_blank_weekly("ILLIQ", [w], liquid=False, w_ret13=9.9, w_ret26=9.9))
    rows.append(_blank_weekly("EXTRA", [w], in_universe=False, w_ret13=0.375, w_ret26=1.0))
    rows.append(_blank_weekly("EXTRA_ILLIQ", [w], liquid=False, in_universe=False, w_ret13=0.375, w_ret26=1.0))
    out = R.add_weekly_ranks(pd.concat(rows, ignore_index=True)).set_index("symbol")
    assert out.loc[["U0", "U1", "U2", "U3"], "xs_rank_ret13"].tolist() == [0.25, 0.5, 0.75, 1.0]
    assert out.loc[["U0", "U1", "U2", "U3"], "xs_rank_ret26"].tolist() == [0.25, 0.5, 0.75, 1.0]
    assert np.isnan(out.loc["ILLIQ", "xs_rank_ret13"])                    # an illiquid stock has no rank ...
    assert out.loc["EXTRA", "xs_rank_ret13"] == 0.75                      # ... a liquid extra: 3 of the 4 universe values are <= its own
    assert out.loc["EXTRA", "xs_rank_ret26"] == 0.5                       # (1.0 vs 0.5 / 1 / 1.5 / 2: two are <= 1.0)
    assert np.isnan(out.loc["EXTRA_ILLIQ", "xs_rank_ret13"])


def test_extra_ranks_of_an_unseen_week_or_a_missing_value_are_nan():
    base = pd.DataFrame({"week": pd.to_datetime(["2023-01-06"] * 3), "v": [1.0, 2.0, 3.0]})
    extra = pd.DataFrame({"week": pd.to_datetime(["2023-01-06", "2023-01-13", "2023-01-06"]), "v": [2.5, 2.5, np.nan]})
    got = R.extra_ranks(base, extra, "week", {"v": "r"})["r"].to_numpy()
    assert got[0] == pytest.approx(2 / 3) and np.isnan(got[1]) and np.isnan(got[2])


def _obs(symbol, date, in_universe=True, **cols):
    d = {"symbol": symbol, "date": pd.Timestamp(date), "dvol50": 1e7}
    for c in FT.RS_SOURCES:
        d[c] = np.nan
    d.update(cols)
    df = pd.DataFrame([d])
    df["in_universe"] = in_universe
    return df


def test_registry_ranks_are_taken_over_every_stock_not_over_the_cohort_names():
    """NVDA has the LOWEST momentum of five stocks. Ranked among all five it is at 0.2 and the rule
    rs_W_mom30 > 0.5 does not flag it; ranked among the cohort only it would be 1.0 and always flagged."""
    w = pd.Timestamp("2023-06-02")
    syms = ["NVDA", "B", "C", "D", "E"]
    parts = [_part(_blank_weekly(sym, [w]), _obs(sym, w, W_mom30=float(i + 1)), sym) for i, sym in enumerate(syms)]
    parts.append(_part(_blank_weekly("NOOBS", [w]), None, "NOOBS"))                 # no Friday observation
    setups = [{"id": "top", "rule": [{"feature": "rs_W_mom30", "op": ">", "value": 0.5}]}]
    panel = R.assemble(parts, setups).set_index("symbol")
    assert panel.loc[syms, "registry:top"].tolist() == [False, False, True, True, True]
    assert panel.loc["NOOBS", "registry:top"] == False                            # noqa: E712 - numpy bool
    assert panel["registry:top"].dtype == bool


def test_a_registry_extra_is_ranked_against_the_universe_and_does_not_move_it():
    w = pd.Timestamp("2023-06-02")
    uni = pd.concat([_obs(f"U{i}", w, W_mom30=float(i + 1)) for i in range(4)], ignore_index=True)
    ex = _obs("EXTRA", w, in_universe=False, W_mom30=3.0)
    ranked = R.rank_registry(pd.concat([uni, ex], ignore_index=True)).set_index("symbol")
    assert ranked.loc[[f"U{i}" for i in range(4)], "rs_W_mom30"].tolist() == [0.25, 0.5, 0.75, 1.0]
    assert ranked.loc["EXTRA", "rs_W_mom30"] == 0.75
    assert ranked["rs_D_mom50"].isna().all()


def test_assemble_evaluates_the_signatures_with_the_ranks_and_never_matches_nan():
    w = pd.Timestamp("2023-06-02")
    rows = []
    for i in range(10):          # ten liquid stocks above a rising 30-week SMA; ranks 0.1 ... 1.0 on w_ret13
        rows.append(_blank_weekly(f"U{i}", [w], w_close=11.0, w_sma30=10.0, w_slope30=0.01, w_ret13=0.05 * (i + 1),
                                  w_ret26=0.1))
    rows.append(_blank_weekly("NANSTOCK", [w]))
    panel = R.assemble([_part(r, None, r.symbol.iloc[0]) for r in rows], []).set_index("symbol")
    assert panel.loc[[f"U{i}" for i in range(10)], "mtf:B0"].all()
    assert not panel.loc["NANSTOCK", ["mtf:B0", "mtf:S1", "mtf:S2", "mtf:S3", "mtf:S4"]].any()
    assert panel.loc[[f"U{i}" for i in range(10)], "mtf:S1"].tolist() == [False] * 7 + [True] * 3   # Top 20%: ranks 0.8, 0.9, 1.0
    assert not panel[["mtf:S2", "mtf:S3", "mtf:S4"]].to_numpy().any()


def test_assemble_without_any_weekly_row_is_an_error():
    with pytest.raises(ValueError, match="no weekly rows"):
        R.assemble([_part(None, None)], [])


# ── the precision proxy: only liquid stocks of the ranking universe are counted ───────────────────

def _v_shaped_closes(n=80, low_at=30):
    return np.r_[np.linspace(100.0, 50.0, low_at + 1), np.linspace(50.0, 150.0, n - low_at)[1:]]


def _proxy_panel(d_in_universe=True):
    weeks = pd.date_range("2021-10-08", periods=80, freq="W-FRI")
    close = _v_shaped_closes()
    low_week = weeks[30]
    flag_week, other_week = weeks[33], weeks[34]
    flag = np.zeros(80, dtype=bool)
    flag[[33, 40]] = True
    frames = [_blank_weekly("D", weeks, in_universe=d_in_universe, w_close=close, scanner_x=flag)]
    one = lambda sym, week, **kw: _blank_weekly(sym, [week], **kw)            # noqa: E731
    frames += [one("A", flag_week, scanner_x=True), one("C", flag_week, scanner_x=False),
               one("B_ILLIQUID", flag_week, liquid=False, scanner_x=True),
               one("E_EXTRA", flag_week, in_universe=False, scanner_x=True),
               one("F_LATER", other_week, scanner_x=True)]
    return pd.concat(frames, ignore_index=True), low_week, flag_week, close


def test_other_flagged_counts_liquid_universe_stocks_only_and_excludes_the_name_itself():
    weekly, low_week, flag_week, close = _proxy_panel()
    counts = R.weekly_counts(weekly, ["scanner_x"])
    assert counts.loc[flag_week, "scanner_x"] == 2          # A and D; not the illiquid B, not the extra E
    assert counts.loc[flag_week, "n_liquid"] == 3           # A, C, D
    rep = R.cohort_report(weekly, counts, "D", ["scanner_x"], in_cache=True)
    cell = rep["scanners"]["scanner_x"]
    assert rep["run"]["low_week"] == low_week and rep["run"]["low_close"] == 50.0
    assert cell["flagged_in_window"] is True and cell["week"] == flag_week
    assert cell["weeks_from_low"] == 3
    assert cell["pct_off_low"] == pytest.approx(close[33] / 50.0 - 1.0)
    assert cell["n_flagged_liquid"] == 2 and cell["n_liquid_universe"] == 3
    assert isinstance(cell["n_other_flagged"], int) and cell["n_other_flagged"] == 1   # only A
    assert R.format_cell(cell) == f"+3w ({close[33] / 50.0 - 1.0:+.0%}) / 1"


def test_a_name_outside_the_ranking_universe_does_not_subtract_itself():
    weekly, _, flag_week, _ = _proxy_panel(d_in_universe=False)
    counts = R.weekly_counts(weekly, ["scanner_x"])
    assert counts.loc[flag_week, "scanner_x"] == 1           # only A: D is an extra now
    cell = R.cohort_report(weekly, counts, "D", ["scanner_x"])["scanners"]["scanner_x"]
    assert (cell["n_flagged_liquid"], cell["n_other_flagged"]) == (1, 1)


def test_the_recall_window_is_eight_weeks_before_and_fifty_two_after_the_found_low():
    weeks = pd.date_range("2021-10-08", periods=120, freq="W-FRI")
    close = np.r_[_v_shaped_closes(80, 30), np.full(40, 150.0)]
    low_week = weeks[30]

    def cell(flag_idx):
        flag = np.zeros(120, dtype=bool)
        flag[flag_idx] = True
        w = _blank_weekly("D", weeks, w_close=close, scanner_x=flag)
        return R.cohort_report(w, R.weekly_counts(w, ["scanner_x"]), "D", ["scanner_x"])["scanners"]["scanner_x"]
    assert cell([30 - 8])["weeks_from_low"] == -8 and cell([30 + 52])["weeks_from_low"] == 52
    for outside in (30 - 9, 30 + 53):
        c = cell([outside])
        assert c["flagged_in_window"] is False and c["ever_flagged"] is True
        assert R.format_cell(c) == "-"
    assert cell([])["ever_flagged"] is False
    assert low_week == weeks[30]


def test_flag_rates_are_flagged_stock_weeks_over_liquid_stock_weeks():
    weeks = pd.date_range("2023-01-06", periods=2, freq="W-FRI")
    rows = [_blank_weekly(f"S{i}", weeks, scanner_x=[i < 2, i < 1]) for i in range(4)]
    rows.append(_blank_weekly("ILLIQ", weeks, liquid=False, scanner_x=True))
    counts = R.weekly_counts(pd.concat(rows, ignore_index=True), ["scanner_x"])
    assert counts["scanner_x"].tolist() == [2, 1] and counts["n_liquid"].tolist() == [4, 4]
    rate = R.flag_rates(counts, ["scanner_x"])["scanner_x"]
    assert rate["flag_rate"] == pytest.approx(3 / 8) and rate["max_flagged_in_a_week"] == 2


def test_format_cell_and_the_learned_count():
    flagged = {"flagged_in_window": True, "weeks_from_low": -2, "pct_off_low": 0.05, "n_other_flagged": 0}
    assert R.format_cell(flagged) == "-2w (+5%) / 0"
    assert R.format_cell({"flagged_in_window": False}) == "-" and R.format_cell(None) == "n/a"
    setups = R.registry_setups()
    learned = [s for s in setups if s["status"] != "approved"]
    assert len(learned) == 17
    scanners = {k: {"flagged_in_window": False} for k in R.scanner_keys(setups)}
    for s in learned[:3]:
        scanners[f"registry:{s['id']}"] = flagged
    rep = {"symbol": "XYZ", "run": {"low_week": pd.Timestamp("2022-10-14"), "mult": 3.0}, "scanners": scanners}
    a, b = R.table_blocks([rep], setups)
    assert a["header"][-1] == "learned (of 17)" and a["rows"][0][-1] == "3/17"
    assert a["rows"][0][:3] == ["XYZ", "2022-10-14", "3.0x"]
    assert len(a["rows"][0]) == len(a["header"]) and len(b["rows"][0]) == len(b["header"])
    none = {"symbol": "NONE", "run": None, "scanners": {}}
    assert R.table_blocks([none], setups)[0]["rows"][0][1:4] == ["n/a", "n/a", "n/a"]


# ── one stock: recall_rows ────────────────────────────────────────────────────────────────────────

def _fridays(a, b):
    return pd.date_range(a, b, freq="W-FRI")


def test_recall_rows_weekly_rows_and_observations_do_not_change_when_later_bars_are_cut_off():
    _finder_or_skip()
    df = _walk(1400, seed=0, p0=3.0, drift=0.0012, start="2019-01-07")
    spy = _walk(1400, seed=9, p0=100.0, start="2019-01-07")
    cohort = frozenset({"AAA"})
    full = R.recall_rows("AAA", df, spy, df.index[-1], _fridays(R.RECALL_START, df.index[-1]), cohort)
    cut = pd.Timestamp("2023-06-02")
    part = R.recall_rows("AAA", df[df.index <= cut], spy, cut, _fridays(R.RECALL_START, cut), cohort)
    fw = full["weekly"][full["weekly"]["week"] <= cut].reset_index(drop=True)
    pw = part["weekly"]
    assert pw["week"].max() == cut and len(fw) == len(pw) > 100
    assert list(fw.columns) == list(pw.columns)
    for c in fw.columns:
        if fw[c].dtype.kind == "f":
            assert np.allclose(fw[c], pw[c], rtol=1e-12, equal_nan=True), c
        else:
            assert fw[c].equals(pw[c]), c
    assert any(pw[f"finder_{p}"].any() for p in R.FINDER_PRESETS)              # the check is not vacuous
    fo = full["registry"][full["registry"]["date"] <= cut].reset_index(drop=True)
    po = part["registry"]
    assert len(fo) == len(po) > 100
    pd.testing.assert_frame_equal(fo, po)


def test_recall_rows_clips_the_stock_and_the_benchmark_to_as_of():
    _finder_or_skip()
    df = _walk(900, seed=1, p0=10.0, start="2019-01-07")
    spy = _walk(900, seed=2, p0=100.0, start="2019-01-07")
    as_of = pd.Timestamp("2021-12-15")                                      # a Wednesday
    r = R.recall_rows("AAA", df, spy, as_of, _fridays(R.RECALL_START, as_of), frozenset({"AAA"}))
    assert r["weekly"]["week"].max() == pd.Timestamp("2021-12-10")          # the Friday before; the part-formed week is absent
    assert (r["registry"]["date"] <= as_of).all()
    assert r["info"]["last_bar"] == as_of and r["info"]["first_bar"] == df.index[0]


def test_recall_rows_keeps_liquid_stocks_and_cohort_names_and_nothing_else():
    _finder_or_skip()
    spy = _walk(1400, seed=2, p0=100.0, start="2019-01-07")
    liquid = _walk(1400, seed=1, p0=20.0, start="2019-01-07", vol=1e6)
    illiquid = _walk(1400, seed=1, p0=20.0, start="2019-01-07", vol=1000.0)
    as_of = liquid.index[-1]
    fr = _fridays(R.RECALL_START, as_of)
    keep = R.recall_rows("L", liquid, spy, as_of, fr, frozenset())
    assert keep["weekly"]["liquid"].all() and len(keep["weekly"]) > 100 and keep["info"] is None
    skip = R.recall_rows("I", illiquid, spy, as_of, fr, frozenset())
    assert skip["weekly"] is None and skip["registry"] is None and skip["error"] is None      # scanned, nothing kept
    named = R.recall_rows("I", illiquid, spy, as_of, fr, frozenset({"I"}))
    assert len(named["weekly"]) > 100 and not named["weekly"]["liquid"].any()
    assert named["info"]["n_bars"] == 1400
    assert R.recall_rows("S", liquid.iloc[:100], spy, as_of, fr, frozenset()) is None          # < MIN_DAILY_ROWS
    assert not [c for c in keep["weekly"].columns if c.startswith("fwd_")]
    assert {"finder_signal", "finder_pennyking", "finder_maker", "turnaround"} <= set(keep["weekly"].columns)


def test_a_failing_stock_is_reported_but_a_failing_cohort_name_stops_the_run(monkeypatch):
    _finder_or_skip()
    spy = _walk(900, seed=2, p0=100.0, start="2019-01-07")
    df = _walk(900, seed=1, p0=20.0, start="2019-01-07")

    def boom(*a, **k):
        raise RuntimeError("bad bar")
    monkeypatch.setattr(R, "turnaround_daily_flags", boom)
    fr = _fridays(R.RECALL_START, df.index[-1])
    r = R.recall_rows("BAD", df, spy, df.index[-1], fr, frozenset())
    assert r["weekly"] is None and r["error"] == "RuntimeError: bad bar"
    with pytest.raises(RuntimeError, match="bad bar"):
        R.recall_rows("NVDA", df, spy, df.index[-1], fr, frozenset({"NVDA"}))


# ── the whole script on a tiny cache ──────────────────────────────────────────────────────────────

def _synthetic_nvda(n=1900, start="2019-01-02"):
    """Rises to 60, falls 80% to a single low on Friday 2022-10-14, bases, then breaks out on heavy volume."""
    rng = np.random.default_rng(5)
    idx = pd.bdate_range(start, periods=n)
    days = np.asarray((idx - idx[0]).days, dtype=float)
    anchors = [(idx[0], 20.0), ("2021-10-01", 60.0), ("2022-10-14", 12.0), ("2023-03-31", 16.0),
               ("2024-04-01", 90.0), (idx[-1], 95.0)]
    xp = [float((pd.Timestamp(d) - idx[0]).days) for d, _ in anchors]
    close = np.exp(np.interp(days, xp, np.log([v for _, v in anchors])) + rng.normal(0, 0.004, n))
    low_bar = idx.get_loc(pd.Timestamp("2022-10-14"))
    close[low_bar] = 10.0
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * 1.004
    low = np.minimum(open_, close) * 0.996
    vol = np.full(n, 600_000.0)
    vol[(idx >= "2023-04-03") & (idx <= "2023-06-30")] = 1_800_000.0
    return _bars(open_, high, low, close, vol, start)


NVDA_LOW_WEEK = pd.Timestamp("2022-10-14")


def _write_cache(tmp_path, batches, bench_names=mtf.BENCHMARKS, n=1900):
    pytest.importorskip("pyarrow")
    hist = tmp_path / "history"
    hist.mkdir()
    for i, stocks in enumerate(batches):
        frames = [df.rename_axis("date").reset_index().assign(symbol=sym) for sym, df in stocks.items()]
        pd.concat(frames)[["date", "symbol", "open", "high", "low", "close", "volume"]].to_parquet(
            hist / f"batch_{i:04d}.parquet")
    bench = tmp_path / "bench"
    bench.mkdir()
    for k, sym in enumerate(bench_names):
        _walk(n, seed=100 + k, p0=100.0, drift=0.0006, vol=5e6, start="2019-01-02").rename_axis("date").to_parquet(
            bench / f"{sym}.parquet")
    return hist, bench


def _stocks(n=1900):
    fillers = {f"F{i:02d}": _walk(n, seed=20 + i, p0=15.0 + i, drift=0.0003 * (i % 4), vol=700_000, start="2019-01-02")
               for i in range(14)}
    fillers["ILLIQ"] = _walk(n, seed=77, p0=20.0, vol=1000.0, start="2019-01-02")
    return fillers


def _run(tmp_path, hist, bench, *extra, out="out"):
    out_dir = tmp_path / out
    rc = R.main(["--history", str(hist), "--bench-dir", str(bench), "--out-dir", str(out_dir), *extra])
    return rc, out_dir


@pytest.mark.parametrize("workers", [1, 2])
def test_end_to_end_on_a_tiny_cache(tmp_path, workers):
    _finder_or_skip()
    fillers = _stocks()
    names = sorted(fillers)
    hist, bench = _write_cache(tmp_path, [{"NVDA": _synthetic_nvda(), **{k: fillers[k] for k in names[:7]}},
                                          {k: fillers[k] for k in names[7:]}])
    rc, out = _run(tmp_path, hist, bench, "--symbols", "NVDA", "--workers", str(workers))
    assert rc == 0
    res = _strict_json(out / "scanner_recall.json")
    assert (out / "scanner_recall.md").read_text().startswith("# Scanner recall")
    assert res["cohort"] == ["NVDA"] and res["missing"] == [] and res["errors"]["n"] == 0
    assert res["limited_run"] is False and res["data_end"] == str(pd.read_parquet(hist / "batch_0000.parquet")["date"].max().date())
    nv = res["cohort_results"][0]
    assert nv["symbol"] == "NVDA" and nv["in_cache"] is True
    # the ex-post low is the engineered bottom, not the first week of data
    assert nv["run"]["low_week"] == "2022-10-14" and nv["run"]["low_close"] == 10.0
    assert nv["run"]["mult"] > 5 and nv["data_gap_warning"] is False
    assert set(nv["scanners"]) == set(res["scanner_keys"]) and len(res["scanner_keys"]) == 29
    flagged = [c for c in nv["scanners"].values() if c["flagged_in_window"]]
    assert flagged, "at least the market-beta baseline should flag NVDA after its recovery"
    for c in flagged:
        assert isinstance(c["n_other_flagged"], int) and c["n_other_flagged"] >= 0
        assert c["n_other_flagged"] <= c["n_liquid_universe"]
    assert nv["scanners"]["mtf:B0"]["flagged_in_window"] is True
    assert nv["scanners"]["mtf:B0"]["weeks_from_low"] >= -8
    # the table has the name, and the same cells as the JSON
    names_in_table = [row[0] for block in res["tables"] for row in block["rows"]]
    assert names_in_table == ["NVDA", "NVDA"]
    assert "NVDA" in (out / "scanner_recall.md").read_text()
    # NVDA + 14 liquid fillers + the illiquid one were scanned; only liquid stocks and NVDA leave a weekly row
    assert res["sizes"]["n_stocks"] == 16 and res["sizes"]["n_extras"] == 0
    assert res["sizes"]["n_weekly_rows"] < 16 * 280
    assert res["sizes"]["n_liquid_rows"] > 0
    assert not (tmp_path / "registry" / "trials.jsonl").exists()                  # the ledger was not touched
    assert not (out / "trials.jsonl").exists()


def test_cohort_names_outside_the_limited_batches_and_the_benchmark_directory_are_extras(tmp_path):
    _finder_or_skip()
    fillers = _stocks()
    names = sorted(fillers)
    hist, bench = _write_cache(tmp_path, [{k: fillers[k] for k in names}, {"NVDA": _synthetic_nvda()}])
    rc1, out1 = _run(tmp_path, hist, bench, "--limit-batches", "1", "--symbols", "NVDA", "--workers", "1", out="o1")
    rc2, out2 = _run(tmp_path, hist, bench, "--limit-batches", "1", "--symbols", "NVDA", "TSM", "SPY", "--workers", "1",
                     out="o2")
    assert rc1 == rc2 == 0
    r1, r2 = _strict_json(out1 / "scanner_recall.json"), _strict_json(out2 / "scanner_recall.json")
    assert r1["limited_run"] is True and r1["sizes"]["n_extras"] == 1
    assert r2["sizes"]["n_extras"] == 3 and r2["missing"] == []
    by = {x["symbol"]: x for x in r2["cohort_results"]}
    assert by["NVDA"]["in_cache"] is True and by["TSM"]["in_cache"] is False and by["SPY"]["in_cache"] is False
    assert by["NVDA"]["run"]["low_week"] == "2022-10-14"                           # found in batch 2 although only batch 1 is scanned
    assert by["TSM"]["run"] is not None
    # extras are ranked against the universe but are not part of it: the universe counts do not move
    assert r1["sizes"]["n_liquid_rows"] == r2["sizes"]["n_liquid_rows"] and r1["sizes"]["n_stocks"] == r2["sizes"]["n_stocks"]
    assert r1["flag_rates"] == r2["flag_rates"]


def test_a_stock_that_fails_is_counted_and_the_run_continues(tmp_path, monkeypatch):
    _finder_or_skip()
    fillers = _stocks()
    names = sorted(fillers)
    hist, bench = _write_cache(tmp_path, [{"NVDA": _synthetic_nvda(), **{k: fillers[k] for k in names[:7]}},
                                          {k: fillers[k] for k in names[7:]}])
    real = R.registry_obs

    def flaky(symbol, daily, fridays, features=None):
        if symbol == "F03":
            raise ValueError("corrupt bar")
        return real(symbol, daily, fridays, features)
    monkeypatch.setattr(R, "registry_obs", flaky)
    rc, out = _run(tmp_path, hist, bench, "--symbols", "NVDA", "--workers", "1")
    res = _strict_json(out / "scanner_recall.json")
    assert rc == 0 and res["errors"]["n"] == 1
    assert res["errors"]["first"] == [{"symbol": "F03", "error": "ValueError: corrupt bar"}]
    assert res["sizes"]["n_stocks"] == 15


def test_main_refuses_a_name_that_is_not_in_the_cohort(tmp_path):
    with pytest.raises(SystemExit, match="NOTACOHORTNAME"):
        R.main(["--history", str(tmp_path), "--symbols", "NOTACOHORTNAME"])
