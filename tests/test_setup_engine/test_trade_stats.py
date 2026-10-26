"""setup_engine.trade_stats: the numbers that decide whether a simulated strategy is believed.

The failure being guarded against: an interval that is far too narrow because forty trades opened in one
month were counted as forty independent observations; an average of cluster averages instead of the
ratio of sums (a one-trade month outweighing a ninety-nine-trade month); a control pool that quietly
wraps a missing volatility reading into the top bucket; and a survivorship estimate inflated by ETFs and
SPAC shells. Each is built as a hand-checked case below.
"""
import numpy as np
import pandas as pd
import pytest

from setup_engine import trade_stats as T


# ── dates and buckets ──────────────────────────────────────────────────────────

def test_the_day_axis_is_anchored_at_2000_and_covers_through_2026():
    assert T.N_DAYS == 9862
    assert T.day_index("2000-01-01")[0] == 0
    assert T.day_index("2026-12-31")[0] == T.N_DAYS - 1
    assert list(T.day_index(["2000-01-02", "2000-01-10"])) == [1, 9]


@pytest.mark.parametrize("bad", ["1999-12-31", "2027-01-01", pd.NaT])
def test_a_date_off_the_axis_is_refused_rather_than_wrapped(bad):
    with pytest.raises(ValueError):
        T.day_index([bad])


def test_day_index_accepts_a_series_and_a_datetime_index():
    s = pd.Series(pd.to_datetime(["2000-01-03", "2001-01-01"]))
    assert list(T.day_index(s)) == [2, 366]
    assert list(T.day_index(pd.DatetimeIndex(s))) == [2, 366]


def test_a_missing_volatility_reading_is_bucket_minus_one_not_the_top_bucket():
    # np.digitize(NaN, edges) is 6, the TOP bucket; a trade with no ATR must join no pool at all.
    b = T.vol_bucket(np.array([0.01, 0.05, 0.5, np.nan, np.inf]))
    assert list(b) == [0, 3, 6, -1, -1]


def test_bucket_edges_belong_to_the_bucket_above():
    assert list(T.vol_bucket(np.array([0.0199999, 0.02, 0.0399999, 0.04, 0.12]))) == [0, 1, 2, 3, 6]
    assert T.N_BUCKETS == len(T.ATR_PCT_EDGES) + 1
    assert T.vol_bucket(np.array([0.05])).dtype == np.int64


def test_month_and_half_year_keys():
    d = ["2020-01-15", "2020-06-30", "2020-07-01", "2021-12-31"]
    assert list(T.month_key(d)) == [202001, 202006, 202007, 202112]
    assert list(T.half_year_key(d)) == [20201, 20201, 20202, 20212]


# ── summaries ──────────────────────────────────────────────────────────────────

def test_summarize_on_a_hand_checked_set():
    s = T.summarize([-1, -1, 2, 4, 0])
    assert s["n"] == 5
    assert s["win_rate"] == pytest.approx(0.4)
    assert s["avg_win"] == pytest.approx(3.0)
    assert s["avg_loss"] == pytest.approx(-1.0)
    assert s["expectancy"] == pytest.approx(0.8)
    assert s["median"] == pytest.approx(0.0)
    assert s["profit_factor"] == pytest.approx(3.0)          # wins 6 / losses 2
    assert s["share_ge_3r"] == pytest.approx(0.2)


def test_summarize_of_nothing_is_n_zero_and_none_everywhere():
    for empty in ([], [np.nan, np.inf]):
        s = T.summarize(empty)
        assert s["n"] == 0
        assert all(v is None for k, v in s.items() if k != "n")


def test_summarize_with_no_losses_has_no_profit_factor_and_no_wins_has_no_avg_win():
    assert T.summarize([1.0, 2.0])["profit_factor"] is None
    assert T.summarize([1.0, 2.0])["avg_loss"] is None
    assert T.summarize([-1.0, -2.0])["avg_win"] is None
    assert T.summarize([-1.0, -2.0])["profit_factor"] == pytest.approx(0.0)


def test_summarize_ignores_non_finite_values():
    assert T.summarize([1.0, np.nan, 3.0, np.inf])["n"] == 2


def test_the_winsorized_mean_is_not_carried_by_one_huge_trade():
    x = np.r_[np.ones(99), 1000.0]
    s = T.summarize(x)
    assert s["expectancy"] == pytest.approx(10.99)
    q99 = np.quantile(x, 0.99)
    assert s["winsorized_mean"] == pytest.approx((99 + q99) / 100)
    assert s["winsorized_mean"] < 2.0


# ── cluster bootstrap ──────────────────────────────────────────────────────────

def _ci_width(r):
    return r["hi"] - r["lo"]


def test_one_cluster_of_identical_values_is_one_observation_not_fifty():
    # 30 months, every trade in a month has the same result. Treating trades as independent pretends
    # to have 1500 observations; there are 30. The honest interval must be far wider.
    rng = np.random.default_rng(1)
    vals = rng.normal(size=30)
    x = np.repeat(vals, 50)
    months = np.repeat(np.arange(30), 50)
    clustered = T.cluster_bootstrap(x, months, n_boot=2000, seed=3)
    naive = T.cluster_bootstrap(x, np.arange(len(x)), n_boot=2000, seed=3)
    assert clustered["n"] == 1500 and clustered["n_clusters"] == 30
    assert naive["n_clusters"] == 1500
    assert _ci_width(clustered) >= 4 * _ci_width(naive)
    assert clustered["lo"] <= clustered["mean"] <= clustered["hi"]


def test_the_mean_is_a_ratio_of_sums_not_a_mean_of_cluster_means():
    # Cluster A: one trade of +10. Cluster B: ninety-nine trades of 0. The mean trade is 0.1, not 5.
    x = np.r_[10.0, np.zeros(99)]
    cl = np.r_[0, np.ones(99, dtype=int)]
    r = T.cluster_bootstrap(x, cl, n_boot=2000, seed=0)
    assert r["mean"] == pytest.approx(0.1)
    # With two clusters a draw is AA (20/2 = 10), AB or BA (10/100 = 0.1) or BB (0), each a ratio of
    # sums, with probabilities 1/4, 1/2, 1/4. A mean of cluster means would give 10, 5 and 0 instead.
    assert r["lo"] == pytest.approx(0.0) and r["hi"] == pytest.approx(10.0)
    mid = T.cluster_bootstrap(x, cl, n_boot=2000, seed=0, alpha=0.9)      # the central 10% of draws
    assert mid["lo"] == pytest.approx(0.1) and mid["hi"] == pytest.approx(0.1)


def test_the_interval_is_the_ratio_of_sums_resampling_used_for_stocks_in_learn():
    # Same construction as learn.excess_return: integers(0, K, (n_boot, K)); sum / count per draw.
    rng0 = np.random.default_rng(5)
    x = rng0.normal(0.2, 1.0, size=240)
    cl = np.repeat(np.arange(12), 20)                     # first-appearance order == sorted order
    s = np.array([x[cl == k].sum() for k in range(12)])
    c = np.array([(cl == k).sum() for k in range(12)], float)
    rng = np.random.default_rng(7)
    idx = rng.integers(0, 12, size=(500, 12))
    boots = s[idx].sum(axis=1) / c[idx].sum(axis=1)
    r = T.cluster_bootstrap(x, cl, n_boot=500, seed=7)
    assert r["lo"] == pytest.approx(np.quantile(boots, 0.025))
    assert r["hi"] == pytest.approx(np.quantile(boots, 0.975))
    assert r["p_le_0"] == pytest.approx((boots <= 0).mean())
    assert r["mean"] == pytest.approx(x.mean())


def test_fewer_than_two_clusters_give_a_mean_but_no_interval():
    r = T.cluster_bootstrap([1.0, 2.0, 3.0], ["a", "a", "a"])
    assert r["mean"] == pytest.approx(2.0)
    assert r["n"] == 3 and r["n_clusters"] == 1
    assert r["lo"] is None and r["hi"] is None and r["p_le_0"] is None


def test_non_finite_values_are_dropped_and_empty_input_is_all_none():
    r = T.cluster_bootstrap([1.0, np.nan, 3.0, np.inf, 5.0], [1, 1, 2, 2, 3])
    assert r["n"] == 3 and r["n_clusters"] == 3
    assert r["mean"] == pytest.approx(3.0)
    e = T.cluster_bootstrap([np.nan], [1])
    assert e["n"] == 0 and e["n_clusters"] == 0 and e["mean"] is None and e["lo"] is None


def test_the_bootstrap_is_reproducible_and_does_not_depend_on_the_chunk_size(monkeypatch):
    rng = np.random.default_rng(2)
    x = rng.normal(size=3000)
    cl = np.arange(3000) // 3                              # 1000 clusters
    a = T.cluster_bootstrap(x, cl, n_boot=400, seed=4)
    assert a == T.cluster_bootstrap(x, cl, n_boot=400, seed=4)
    monkeypatch.setattr(T, "_CHUNK_ELEMS", 50_000)         # forces several blocks
    b = T.cluster_bootstrap(x, cl, n_boot=400, seed=4)
    assert (a["lo"], a["hi"], a["p_le_0"]) == (b["lo"], b["hi"], b["p_le_0"])


def test_a_longer_interval_when_more_certainty_is_asked_for():
    rng = np.random.default_rng(8)
    x, cl = rng.normal(size=600), np.arange(600) // 10
    assert _ci_width(T.cluster_bootstrap(x, cl, alpha=0.01)) > _ci_width(T.cluster_bootstrap(x, cl, alpha=0.20))


# ── paired cluster bootstrap ───────────────────────────────────────────────────

def test_a_constant_shift_between_arms_has_a_zero_width_interval():
    # xb = xa - 1 on the same clusters: the difference is exactly 1 in EVERY draw only if the two arms
    # are resampled jointly. Independent resampling would give a wide interval.
    rng = np.random.default_rng(3)
    xa = rng.normal(size=300)
    cl = np.arange(300) // 10
    r = T.cluster_bootstrap_diff(xa, cl, xa - 1.0, cl, n_boot=1000, seed=1)
    assert r["mean"] == pytest.approx(1.0)
    assert r["lo"] == pytest.approx(1.0) and r["hi"] == pytest.approx(1.0)
    assert r["p_le_0"] == pytest.approx(0.0)
    assert r["n_a"] == 300 and r["n_b"] == 300 and r["n_clusters"] == 30


def test_identical_arms_have_a_difference_interval_covering_zero():
    rng = np.random.default_rng(4)
    x, cl = rng.normal(size=200), np.arange(200) // 8
    r = T.cluster_bootstrap_diff(x, cl, x, cl, n_boot=500, seed=2)
    assert r["mean"] == pytest.approx(0.0)
    assert r["lo"] <= 0.0 <= r["hi"]


def test_a_draw_with_no_trades_in_one_arm_is_skipped_not_fatal():
    # Arm A lives in clusters 0-2, arm B in 3-5: about 3% of draws contain none of one arm.
    xa, ca = np.array([1.0, 2.0, 3.0]), np.array([0, 1, 2])
    xb, cb = np.array([0.5, 0.5, 0.5]), np.array([3, 4, 5])
    r = T.cluster_bootstrap_diff(xa, ca, xb, cb, n_boot=2000, seed=0)
    assert r["mean"] == pytest.approx(1.5)
    assert np.isfinite(r["lo"]) and np.isfinite(r["hi"]) and np.isfinite(r["p_le_0"])
    assert r["n_clusters"] == 6


def test_an_empty_arm_gives_no_difference():
    r = T.cluster_bootstrap_diff([1.0, 2.0], [0, 1], [np.nan], [0])
    assert r["mean"] is None and r["lo"] is None and r["n_b"] == 0


def test_each_arms_mean_in_the_difference_is_a_ratio_of_sums():
    xa, ca = np.r_[10.0, np.zeros(99)], np.r_[0, np.ones(99, dtype=int)]
    xb, cb = np.array([0.0, 0.0]), np.array([0, 1])
    r = T.cluster_bootstrap_diff(xa, ca, xb, cb, n_boot=2000, seed=0, alpha=0.9)
    assert r["mean"] == pytest.approx(0.1)
    assert r["lo"] == pytest.approx(0.1) and r["hi"] == pytest.approx(0.1)    # the middle draws are AB / BA


# ── control pools ──────────────────────────────────────────────────────────────

D = 7000          # an arbitrary day on the axis


def _pool():
    acc = T.PoolAccumulator()
    acc.add("c", [D, D, D], [2, 2, 2], [1.0, -1.0, 3.0])
    acc.add("c", [D], [5], [2.0])
    return acc


def test_pool_statistics_for_one_day_and_bucket():
    a = _pool().arr["c"]
    assert a.shape == (T.N_DAYS, T.N_BUCKETS, len(T.POOL_STATS))
    cell = dict(zip(T.POOL_STATS, a[D, 2]))
    assert cell == {"n": 3, "sum_r": 3.0, "sum_r2": 11.0, "n_win": 2, "sum_win": 4.0,
                    "n_loss": 1, "sum_loss": -1.0, "n_ge3": 1}
    mean, n = T.pool_means(a, [D], bucket=2)
    assert n[0] == 3 and mean[0] == pytest.approx(1.0)


def test_the_pool_mean_over_all_buckets_pools_every_trade_of_the_day():
    a = _pool().arr["c"]
    mean, n = T.pool_means(a, [D])
    assert n[0] == 4 and mean[0] == pytest.approx(1.25)
    mean5, n5 = T.pool_means(a, [D], bucket=5)
    assert n5[0] == 1 and mean5[0] == pytest.approx(2.0)


def test_an_empty_pool_has_a_nan_mean_and_zero_n():
    a = _pool().arr["c"]
    mean, n = T.pool_means(a, [D + 1, D], bucket=[2, 3])
    assert list(n) == [0, 0] and np.isnan(mean).all()


def test_bucket_minus_one_matches_nothing_even_when_the_top_bucket_has_trades():
    acc = T.PoolAccumulator()
    acc.add("c", [D], [6], [5.0])                          # the TOP bucket holds a trade on this day
    mean, n = T.pool_means(acc.arr["c"], [D, D], bucket=[-1, 6])
    assert list(n) == [0, 1]
    assert np.isnan(mean[0]) and mean[1] == pytest.approx(5.0)


def test_nan_r_and_bucket_minus_one_are_ignored_on_add():
    acc = T.PoolAccumulator()
    acc.add("c", [D, D, D, D], [2, -1, 2, 2], [1.0, 9.0, np.nan, np.inf])
    a = acc.arr["c"]
    assert a[..., 0].sum() == 1 and a[D, 2, 1] == pytest.approx(1.0)


def test_add_refuses_a_day_off_the_axis_and_mismatched_lengths():
    acc = T.PoolAccumulator()
    with pytest.raises(ValueError):
        acc.add("c", [T.N_DAYS], [1], [1.0])
    with pytest.raises(ValueError):
        acc.add("c", [D, D], [1], [1.0, 2.0])
    with pytest.raises(ValueError):
        acc.add("c", [D], [T.N_BUCKETS], [1.0])


def test_merging_pools_adds_them_and_keeps_keys_unique_to_either_side():
    a, b = _pool(), _pool()
    b.add("only_b", [D], [0], [1.0])
    out = a.merge(b)
    assert out is a
    assert a.arr["c"][D, 2, 0] == 6 and a.arr["c"][D, 5, 1] == pytest.approx(4.0)
    assert a.arr["only_b"][D, 0, 0] == 1
    assert b.arr["c"][D, 2, 0] == 3                        # the source is not modified
    a.arr["only_b"][D, 0, 0] = 99
    assert b.arr["only_b"][D, 0, 0] == 1                   # and not aliased


def test_excess_is_the_trade_minus_its_own_days_pool():
    a = _pool().arr["c"]
    ex = T.excess_vs_pool([3.0, 3.0, 3.0], [D, D, D + 1], a, bucket=[2, 5, 2])
    assert ex[0] == pytest.approx(2.0)                     # 3 - 1.0
    assert ex[1] == pytest.approx(1.0)                     # 3 - 2.0
    assert np.isnan(ex[2])                                 # nothing on that day


def test_matched_control_summary_by_hand():
    acc = T.PoolAccumulator()
    acc.add("c", [D, D, D], [2, 2, 2], [1.0, -1.0, 3.0])     # day D: n 3, sum 3, 2 wins (sum 4), 1 loss, 1 >= 3R
    acc.add("c", [D, D], [3, 3], [2.0, 2.0])                 # another bucket, must not be used
    acc.add("c", [D + 1, D + 1], [2, 2], [-2.0, -2.0])       # day D+1: n 2, sum -4, 2 losses
    pool = acc.arr["c"]
    # three treatment trades; the third has no control at all
    s = T.matched_control_summary(pool, [D, D + 1, D + 5], bucket=2)
    assert s["n_input"] == 3 and s["n_matched"] == 2
    assert s["expectancy"] == pytest.approx(np.mean([3 / 3, -4 / 2]))           # -0.5
    assert s["win_rate"] == pytest.approx(np.mean([2 / 3, 0.0]))                 # 1/3
    assert s["avg_win"] == pytest.approx((4 / 3) / (2 / 3))                      # 2.0
    assert s["avg_loss"] == pytest.approx((-1 / 3 - 4 / 2) / (1 / 3 + 2 / 2))    # -1.75
    assert s["profit_factor"] == pytest.approx((4 / 3) / (1 / 3 + 4 / 2))        # 4/7
    assert s["share_ge_3r"] == pytest.approx(np.mean([1 / 3, 0.0]))              # 1/6
    assert s["median_pool_size"] == pytest.approx(2.5) and s["min_pool_size"] == 2
    assert s["median"] is None                              # not recoverable from sums


def test_matched_control_summary_over_all_buckets_and_when_nothing_matches():
    acc = T.PoolAccumulator()
    acc.add("c", [D, D, D], [2, 2, 2], [1.0, -1.0, 3.0])
    acc.add("c", [D], [5], [2.0])
    s = T.matched_control_summary(acc.arr["c"], [D])
    assert s["expectancy"] == pytest.approx(1.25) and s["median_pool_size"] == 4
    none = T.matched_control_summary(acc.arr["c"], [D + 9], bucket=2)
    assert none["n_matched"] == 0 and none["expectancy"] is None and none["min_pool_size"] is None


def test_a_pool_of_only_winners_has_no_loss_stats():
    acc = T.PoolAccumulator()
    acc.add("c", [D], [1], [1.0])
    s = T.matched_control_summary(acc.arr["c"], [D], bucket=1)
    assert s["avg_loss"] is None and s["profit_factor"] is None and s["avg_win"] == pytest.approx(1.0)


def test_a_pool_of_the_wrong_shape_is_refused():
    with pytest.raises(ValueError):
        T.pool_means(np.zeros((3, 3, 3)), [0])


# ── sizing arithmetic ──────────────────────────────────────────────────────────

def test_breakeven_share_of_total_losses():
    f = T.breakeven_f(0.3, -1.0)
    assert f == pytest.approx(0.3 / 1.3)
    assert T.adjusted_expectancy(0.3, f, -1.0) == pytest.approx(0.0, abs=1e-12)
    assert T.adjusted_expectancy(0.3, 0.0, -1.0) == pytest.approx(0.3)
    assert T.adjusted_expectancy(0.3, 0.5, -1.0) < 0


@pytest.mark.parametrize("E,L", [(0.0, -1.0), (-0.2, -1.0), (0.3, 0.0), (0.3, 1.0), (None, -1.0), (0.3, None)])
def test_no_breakeven_when_there_is_no_edge_or_no_loss_to_absorb(E, L):
    assert T.breakeven_f(E, L) is None


# ── survivorship ───────────────────────────────────────────────────────────────

DELISTED = pd.DataFrame({
    "name": ["Alpha Corp", "Beta Inc", "Some Index ETF", "X Acquisition Corp", "Gamma Inc", "Delta Inc"],
    "assetType": ["Stock", "Stock", "ETF", "Stock", "Stock", "Stock"],
    "ipoDate": ["2005-01-01", "2010-03-01", "2001-01-01", "2020-02-01", "2015-06-01", "2008-01-01"],
    "delistingDate": ["2012-06-30", "2011-05-01", "2015-01-01", "2022-08-01", "2021-09-01", "2010-01-01"],
})
SPANS = pd.DataFrame({
    "symbol": ["S1", "S2", "S3", "S4", "S5"],
    "first_date": ["2000-01-03", "2009-05-01", "2011-02-01", "2019-01-01", "2001-01-01"],
    "last_date": ["2026-10-01", "2026-10-01", "2026-10-01", "2026-10-01", "2010-12-31"],
})


def test_delisted_share_by_year_on_a_hand_table_excludes_etfs_and_spacs():
    r = T.delisted_share_by_year(DELISTED, SPANS, [2010, 2011, 2021])
    assert list(r.index) == [2010, 2011, 2021] and r.index.name == "year"
    assert r.loc[2010, "survivors_alive"] == 3 and r.loc[2010, "delisted_alive"] == 3     # ETF would make it 4
    assert r.loc[2010, "f_stock"] == pytest.approx(3 / 6)
    assert r.loc[2011, "survivors_alive"] == 3 and r.loc[2011, "delisted_alive"] == 2
    assert r.loc[2011, "f_stock"] == pytest.approx(2 / 5)
    assert r.loc[2021, "survivors_alive"] == 4 and r.loc[2021, "delisted_alive"] == 1     # the SPAC would make it 2
    assert r.loc[2021, "f_stock"] == pytest.approx(1 / 5)
    assert "f_bottom_scenario" not in r.columns


def test_a_listing_that_is_delisted_on_jan_first_is_alive_that_year():
    r = T.delisted_share_by_year(DELISTED, SPANS, [2010])        # Delta delists 2010-01-01
    assert r.loc[2010, "delisted_alive"] == 3


def test_the_bottom_scenario_counts_delistings_in_the_next_three_years():
    r = T.delisted_share_by_year(DELISTED, SPANS, [2010, 2011, 2021, 2024], bottom_counts={2010: 6, 2011: 2, 2021: 0, 2024: 0})
    assert r.loc[2010, "delisted_near"] == 3 and r.loc[2010, "f_bottom_scenario"] == pytest.approx(3 / 9)
    assert r.loc[2011, "delisted_near"] == 2 and r.loc[2011, "f_bottom_scenario"] == pytest.approx(2 / 4)
    assert r.loc[2021, "delisted_near"] == 1 and r.loc[2021, "f_bottom_scenario"] == pytest.approx(1.0)
    assert r.loc[2024, "delisted_near"] == 0 and pd.isna(r.loc[2024, "f_bottom_scenario"])    # 0 / 0
    assert r.loc[2024, "f_stock"] == pytest.approx(0.0)


def test_a_year_with_no_bottom_count_has_no_scenario_value():
    r = T.delisted_share_by_year(DELISTED, SPANS, [2010, 2011], bottom_counts={2010: 6})
    assert pd.isna(r.loc[2011, "f_bottom_scenario"])


def test_the_delisted_list_is_flagged_incomplete_before_2013():
    r = T.delisted_share_by_year(DELISTED, SPANS, [2010, 2012, 2013, 2021])
    assert list(r["list_incomplete"]) == [True, True, False, False]


def test_the_non_common_pattern_matches_the_shell_names_and_not_ordinary_ones():
    s = pd.Series(["Foo Acquisition Corp", "Bar Warrants", "Baz Units", "Qux Rights", "Zed Merger Sub",
                   "Abc Preferred", "Def Pfd", "Ghi When Issued", "Apple Inc", "United Rentals", "Rightmove"])
    hit = s.str.contains(T.NON_COMMON_RE, case=False, regex=True).tolist()
    assert hit == [True] * 8 + [False] * 3


def test_a_missing_column_in_the_listing_table_is_reported():
    with pytest.raises(ValueError, match="name"):
        T.delisted_share_by_year(DELISTED.drop(columns=["name"]), SPANS, [2010])
