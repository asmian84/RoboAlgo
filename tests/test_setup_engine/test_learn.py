"""setup_engine.learn: it must find real setups, and above all must NOT find fake ones.

The failure being guarded against: "learning" that rewards a setup for being observed in a good
year, for being a volatile stock, or for being one lucky stock sampled many times. Each is built
as a synthetic trap below, and each must come out as no edge.
"""
import numpy as np
import pandas as pd
import pytest

from setup_engine import learn as L


def _obs(n_stocks=300, n_dates=24, seed=0, planted=True, date_rates=None):
    """Synthetic panel. With planted=True, runners are far likelier when A is high and B is low."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2002-01-04", periods=n_dates, freq="28D")
    rows = []
    for s in range(n_stocks):
        for k, d in enumerate(dates):
            a, b, c = rng.uniform(), rng.uniform(), rng.normal()
            p = 0.02
            if planted and a > 0.6 and b < 0.3:
                p = 0.40
            if date_rates is not None:
                p = date_rates[k]
            rows.append((f"S{s}", d, a, b, c, p))
    df = pd.DataFrame(rows, columns=["symbol", "date", "A", "B", "C", "p"])
    df["runner"] = (rng.uniform(size=len(df)) < df["p"]).astype(int)
    df["fwd_max_3y"] = np.where(df["runner"] == 1, 5.0, 0.2)
    df["ret"] = 0.1
    df["bad"] = 0
    return df


# ── splits ─────────────────────────────────────────────────────────────────────

def test_the_configured_splits_are_purged():
    L.check_purged()


def test_overlapping_stages_are_refused():
    bad = {"train": ("2001-01-01", "2012-12-31"), "validate": ("2014-01-01", "2016-12-31")}   # 2012 + 3y > 2014
    with pytest.raises(ValueError, match="not purged"):
        L.check_purged(bad)


# ── rules ──────────────────────────────────────────────────────────────────────

def test_a_missing_value_never_satisfies_a_condition():
    df = pd.DataFrame({"x": [np.nan, 1.0, -1.0]})
    assert list(L.Condition("x", "<=", 0.0).mask(df)) == [False, False, True]
    assert list(L.Condition("x", ">", 0.0).mask(df)) == [False, True, False]


def test_rules_from_a_tree_partition_the_complete_cases_exactly_once():
    d = _obs(120)
    rules = L.fit_rules(d, ["A", "B", "C"], depth=3, min_leaf=50)
    assert len(rules) >= 3
    hits = np.sum([r.mask(d) for r in rules], axis=0)
    assert (hits == 1).all(), "every row must fall in exactly one leaf"


def test_the_learner_recovers_a_planted_setup():
    d = _obs(400)
    rules = L.fit_rules(d, ["A", "B", "C"], depth=3, min_leaf=100)
    rate = L.base_rate_by_date(d)
    scored = [(L.evaluate(d, r.mask(d), rate, n_boot=50), r) for r in rules]
    big = [(e, r) for e, r in scored if e["n_obs"] >= 500]          # support: ignore tiny leaves that split on noise
    e, best = max(big, key=lambda t: t[0]["lift"] or 0)
    thr = {c.feature: c.value for c in best.conditions if c.feature in ("A", "B")}
    assert thr["A"] == pytest.approx(0.6, abs=0.03) and thr["B"] == pytest.approx(0.3, abs=0.03)
    assert e["lift"] > 4                                            # planted odds are ~6x the base rate


def test_rules_round_trip_through_json():
    r = L.Rule((L.Condition("W_mom30", ">", 0.12), L.Condition("D_atr_ratio", "<=", 0.9)))
    assert L.Rule.from_json(r.to_json()) == r


# ── the traps ──────────────────────────────────────────────────────────────────

def test_trap_a_rule_that_only_lives_in_good_years_has_no_lift():
    # No setup at all: runner odds depend ONLY on the date (2% most years, 20% in four boom years).
    rates = [0.20 if k in (3, 4, 5, 6) else 0.02 for k in range(24)]
    d = _obs(400, planted=False, date_rates=rates)
    boom = d["date"].isin(sorted(d["date"].unique())[3:7]).to_numpy()
    naive = d.loc[boom, "runner"].mean() / d["runner"].mean()
    adj = L.evaluate(d, boom, n_boot=300)
    assert naive > 3, "the trap must look attractive without the date adjustment"
    assert adj["lift"] == pytest.approx(1.0, abs=0.15)
    assert adj["lift_lo"] < 1.0 < adj["lift_hi"]


def test_trap_one_lucky_stock_is_not_a_setup():
    d = _obs(60, planted=False)
    d["runner"] = 0
    lucky = d["symbol"] == "S0"
    d.loc[lucky, "runner"] = 1                                   # all the runners belong to a single stock
    m = (lucky | (d["symbol"] == "S1")).to_numpy()               # a rule that picks that stock (and one other)
    e = L.evaluate(d, m, n_boot=1000)
    assert e["n_stocks"] == 2 and e["lift"] > 5
    assert e["lift_lo"] < 1.0, "resampling stocks must show the result hangs on one name"


def test_trap_high_volatility_alone_is_not_a_setup():
    rng = np.random.default_rng(3)
    d = _obs(500, planted=False, seed=3)
    d["W_range10"] = rng.uniform(size=len(d))
    p = np.where(d["W_range10"] > 0.66, 0.15, 0.02)              # volatile stocks run more, no other structure
    d["runner"] = (rng.uniform(size=len(d)) < p).astype(int)
    rule_mask = (d["W_range10"] > 0.66).to_numpy()
    overall = L.evaluate(d, rule_mask, n_boot=100)
    within = L.lift_within_volatility_terciles(d, rule_mask)
    assert overall["lift"] > 2                                    # it looks like a setup (15% vs ~6% overall)
    assert within["high_vol"]["lift"] == pytest.approx(1.0, abs=0.2)   # but is only the volatility


def test_a_real_setup_survives_the_volatility_check():
    rng = np.random.default_rng(4)
    d = _obs(600, planted=True, seed=4)
    d["W_range10"] = rng.uniform(size=len(d))                     # independent of the planted setup
    m = ((d["A"] > 0.6) & (d["B"] < 0.3)).to_numpy()
    within = L.lift_within_volatility_terciles(d, m)
    assert all(v["lift"] is not None and v["lift"] > 3 for v in within.values())


# ── the ledger ─────────────────────────────────────────────────────────────────

def test_the_ledger_only_ever_grows(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "REGISTRY_DIR", tmp_path)
    assert L.trial_count() == 0
    L.log_trial({"model": "m1", "rule": "A > 0.5"})
    L.log_trial({"model": "m1", "rule": "B <= 0.2"})
    assert L.trial_count() == 2
    L.log_trial({"model": "m2", "rule": "C > 0"})
    assert L.trial_count() == 3
    lines = (tmp_path / "trials.jsonl").read_text().splitlines()
    assert lines[0].count("m1") == 1 and "ts" in lines[0]        # earlier entries untouched


# ── type versus timing ─────────────────────────────────────────────────────────

def _typed_panel(kind, n_stocks=500, n_dates=40, seed=11):
    """kind='type': 20% of stocks are runner-prone ALL the time and always show high A. No timing signal.
       kind='timing': every stock is alike, but within a stock runs follow low B. No stock-type signal."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2002-01-04", periods=n_dates, freq="28D")
    prone = rng.uniform(size=n_stocks) < 0.2
    rows = []
    for s in range(n_stocks):
        for d in dates:
            a = rng.uniform(0.6, 1.0) if (kind == "type" and prone[s]) else rng.uniform(0.0, 0.6) if kind == "type" else rng.uniform()
            b = rng.uniform()
            if kind == "both":     # a dominant stock-type effect AND a weaker timing effect, together
                a = rng.uniform(0.6, 1.0) if prone[s] else rng.uniform(0.0, 0.6)
                p = min(0.6, (0.10 if prone[s] else 0.01) * (4 if b < 0.2 else 1))
            else:
                p = (0.20 if prone[s] else 0.01) if kind == "type" else (0.30 if b < 0.2 else 0.02)
            rows.append((f"S{s}", d, a, b, p))
    df = pd.DataFrame(rows, columns=["symbol", "date", "A", "B", "p"])
    df["runner"] = (rng.uniform(size=len(df)) < df["p"]).astype(int)
    df["fwd_max_3y"] = np.where(df["runner"] == 1, 5.0, 0.2)
    df["ret"] = np.where(df["runner"] == 1, 2.0, 0.0)
    df["bad"] = 0
    return df


def test_trap_a_stock_type_signal_has_big_cross_sectional_lift_but_no_timing_lift():
    d = _typed_panel("type")
    m = (d["A"] > 0.6).to_numpy()                                   # picks out the runner-prone stocks, always
    e = L.evaluate(d, m, n_boot=400)
    assert e["lift"] > 3, "it must look like a setup when judged against the market"
    assert e["timing_lift"] == pytest.approx(1.0, abs=0.05), "but it says nothing about WHEN a given stock runs"
    assert e["p_timing_le_1"] > 0.5, "and it must not be significant"


def test_a_real_timing_signal_has_a_large_timing_lift():
    d = _typed_panel("timing")
    m = (d["B"] < 0.2).to_numpy()
    e = L.evaluate(d, m, n_boot=400)
    # true effect is ~3.95x. The measure runs slightly low because each stock's own run rate includes the
    # very runs being credited to the rule, which is the conservative direction.
    assert 3.0 < e["timing_lift"] < 4.2 and e["timing_lo"] > 2.5


def test_the_timing_learner_finds_the_moment_and_ignores_the_stock_type():
    d = _typed_panel("type")
    d["B"] = np.random.default_rng(5).uniform(size=len(d))          # noise
    rules = L.fit_rules(d, ["A", "B"], depth=3, min_leaf=300, target="timing")
    rate = L.base_rate_by_date(d)
    lifts = [L.evaluate(d, r.mask(d), rate, n_boot=100)["timing_lift"] or 0 for r in rules]
    assert max(lifts) < 1.6, "with no timing signal in the data, no rule should look like timing"


def test_the_timing_learner_recovers_a_planted_timing_signal():
    d = _typed_panel("timing")
    rules = L.fit_rules(d, ["A", "B"], depth=2, min_leaf=300, target="timing")
    rate = L.base_rate_by_date(d)
    ev = [(L.evaluate(d, r.mask(d), rate, n_boot=100), r) for r in rules]
    e, best = max(((e, r) for e, r in ev if e["n_obs"] >= 800), key=lambda t: t[0]["timing_lift"] or 0)
    assert any(c.feature == "B" and c.op == "<=" and c.value == pytest.approx(0.2, abs=0.05) for c in best.conditions)
    assert e["timing_lift"] > 3


def test_a_classifier_learner_is_fooled_by_stock_type_and_the_timing_learner_is_not():
    d = _typed_panel("type")
    clf = L.fit_rules(d, ["A"], depth=1, min_leaf=300, target="type_and_timing")
    assert any(c.feature == "A" for r in clf for c in r.conditions)       # it latches onto the type feature
    rate = L.base_rate_by_date(d)
    best_clf = max(L.evaluate(d, r.mask(d), rate, n_boot=100)["timing_lift"] or 0 for r in clf)
    assert best_clf < 1.4                                                 # ...and its rule has no timing value


def test_with_type_and_timing_together_the_two_learners_split_on_different_things():
    d = _typed_panel("both")
    clf = L.fit_rules(d, ["A", "B"], depth=2, min_leaf=300, target="type_and_timing")
    tim = L.fit_rules(d, ["A", "B"], depth=2, min_leaf=300, target="timing")
    # a rule's first condition is the tree's root split
    assert {r.conditions[0].feature for r in clf} == {"A"}, "the classifier follows the dominant stock-type effect"
    assert {r.conditions[0].feature for r in tim} == {"B"}, "the timing learner follows the moment, not the kind of stock"


# ── robustness and the registry ──────────────────────────────────────────────────

def _liq_panel(effect_in_liquid: bool, n=1200, seed=33):
    rng = np.random.default_rng(seed)
    d = _typed_panel("timing", n_stocks=n, seed=seed)                 # a genuine timing effect on B < 0.2
    liquid = rng.uniform(size=n) < 0.5
    d["dvol50"] = np.where(d["symbol"].str[1:].astype(int).map(lambda i: liquid[i]), 50e6, 1e6)
    if not effect_in_liquid:                                           # the effect exists only in the illiquid half
        liq = d["dvol50"] >= 10e6
        d.loc[liq, "runner"] = (rng.uniform(size=int(liq.sum())) < 0.03).astype(int)
    d["ret"] = np.where(d["runner"] == 1, 4.0, 0.2)
    d["bad"] = (d["ret"] <= -0.5).astype(int)
    return d


def test_a_lift_that_exists_only_in_illiquid_stocks_fails_robustness():
    d = _liq_panel(effect_in_liquid=False)
    r = L.robustness(d, (d["B"] < 0.2).to_numpy(), n_boot=300)
    assert not r["passed"] and any("liquid" in x for x in r["reasons"])


def test_a_lift_that_holds_in_liquid_stocks_passes_robustness():
    d = _liq_panel(effect_in_liquid=True)
    r = L.robustness(d, (d["B"] < 0.2).to_numpy(), n_boot=300)
    assert r["passed"], r["reasons"]
    assert r["liquid_timing_lift"] > 2 and r["liquid_lo"] > 1


def test_a_rule_that_mostly_finds_collapsing_stocks_fails_even_with_a_big_lift():
    d = _liq_panel(effect_in_liquid=True)
    m = (d["B"] < 0.2).to_numpy()
    d["ret"] = np.where(m & (d["runner"] == 0), -0.8, d["ret"])                   # non-runners the rule flags lose 80%
    d["bad"] = (d["ret"] <= -0.5).astype(int)
    r = L.robustness(d, m, n_boot=300)
    assert not r["passed"] and any("end badly" in x for x in r["reasons"])


def test_the_scan_never_surfaces_an_unapproved_setup_by_default():
    reg = {"setups": [
        {"id": "ok", "status": "approved", "rule": [{"feature": "x", "op": ">", "value": 0.0}]},
        {"id": "meh", "status": "rejected_robustness", "rule": [{"feature": "x", "op": ">", "value": 0.0}]}]}
    cur = pd.DataFrame({"symbol": ["A", "B"], "x": [1.0, -1.0], "close": [5.0, 6.0], "dvol50": [1e6, 1e6]})
    assert list(L.scan_current(cur, reg)["setup"]) == ["ok"]
    assert set(L.scan_current(cur, reg, include_unapproved=True)["setup"]) == {"ok", "meh"}
    assert L.scan_current(cur, {"setups": [reg["setups"][1]]}).empty


# ── objectives and the excess-return check ──────────────────────────────────────

def test_every_objective_is_purged():
    for o in L.OBJECTIVES.values():
        L.check_purged(obj=o)


def test_slice_period_follows_the_objective():
    obs = pd.DataFrame({
        "symbol": list("AAAA"), "date": pd.to_datetime(["2005-01-07", "2014-01-10", "2019-01-11", "2024-06-07"]),
        "fwd_max_3y": [5.0, 0.1, 0.2, np.nan], "fwd_ret_3y": [4.0, 0.0, 0.0, np.nan],
        "fwd_max_6m": [0.6, 0.1, 0.9, 0.2], "fwd_ret_6m": [0.5, 0.0, -0.4, 0.1], "dvol50": 1e6})
    tr3 = L.slice_period(obs, "train", L.OBJ_3Y)
    assert list(tr3["runner"]) == [1] and list(tr3["ret"]) == [4.0]              # +400% needed for a 3y runner
    tr6 = L.slice_period(obs, "train", L.OBJ_6M)
    assert list(tr6["runner"]) == [0]                                           # +60% is NOT a 6m runner: it takes +100%
    te6 = L.slice_period(obs, "test", L.OBJ_6M)
    assert list(te6["runner"]) == [0, 0] and list(te6["bad"]) == [1, 0]         # 2019 ran +90% then ended -40% (bad); 2024 no run
    assert L.slice_period(obs, "test", L.OBJ_3Y).empty                          # unlabelled 3y windows are excluded


def _panel_for_excess(flag_ret, other_ret, runner_flag=0.2, runner_other=0.05, n=600, seed=9):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2002-01-04", periods=20, freq="28D")
    rows = []
    for s_ in range(n):
        flagged = s_ < n // 5
        for d in dates:
            run = rng.uniform() < (runner_flag if flagged else runner_other)
            ret = (flag_ret if run else -0.8) if flagged else (other_ret if run else 0.0)   # flagged non-runners lose 80%
            rows.append((f"S{s_}", d, int(run), ret, int(flagged)))
    df = pd.DataFrame(rows, columns=["symbol", "date", "runner", "ret", "flag"])
    df["bad"] = (df["ret"] <= -0.3).astype(int)
    return df


def test_more_hits_that_earn_no_more_is_not_an_edge():
    # flagged stocks hit a big move 4x as often, but the ones that do not hit lose 80%: round one's finding
    d = _panel_for_excess(flag_ret=2.0, other_ret=2.0)
    e = L.excess_return(d, d["flag"].to_numpy().astype(bool), n_boot=300)
    assert e["excess_mean"] < 0 and e["excess_hi"] < 0.05


def test_a_rule_that_really_earns_more_has_a_positive_excess():
    rng = np.random.default_rng(2)
    d = _panel_for_excess(2.0, 2.0)
    flagged = d["flag"].to_numpy().astype(bool)
    d.loc[flagged, "ret"] = rng.normal(0.30, 0.2, flagged.sum())                  # earns +30% on average, others ~0
    d.loc[~flagged, "ret"] = rng.normal(0.0, 0.2, (~flagged).sum())
    e = L.excess_return(d, flagged, n_boot=300)
    # the same date's average includes the flagged stocks themselves (20% of the panel at +30% -> +6%), so
    # the true excess over that baseline is 0.30 - 0.06 = 0.24, and that is the conservative reading
    assert e["excess_lo"] > 0.15 and e["excess_mean"] == pytest.approx(0.24, abs=0.04)


def test_one_enormous_winner_cannot_make_a_rule_look_profitable():
    rng = np.random.default_rng(3)
    d = _panel_for_excess(2.0, 2.0)
    flagged = d["flag"].to_numpy().astype(bool)
    d["ret"] = rng.normal(0.0, 0.2, len(d))
    d.loc[d.index[flagged][0], "ret"] = 500.0                                    # a single 500x print among the flagged
    e = L.excess_return(d, flagged, n_boot=300)
    assert e["excess_mean"] < 0.02, "winsorising must cap the influence of one outlier"


def test_flagging_only_boom_dates_earns_no_excess_over_the_dates_average():
    rng = np.random.default_rng(4)
    dates = pd.date_range("2002-01-04", periods=20, freq="28D")
    rows = [(f"S{i}", d, 0, (0.5 if k < 5 else 0.0) + rng.normal(0, 0.1)) for i in range(400) for k, d in enumerate(dates)]
    d = pd.DataFrame(rows, columns=["symbol", "date", "runner", "ret"])
    d["bad"] = 0
    boom = d["date"].isin(dates[:5]).to_numpy()                                  # the first five dates all did well
    e = L.excess_return(d, boom, n_boot=300)
    assert abs(e["excess_mean"]) < 0.03


def test_robustness_rejects_a_rule_that_hits_often_but_does_not_earn_more():
    d = _panel_for_excess(2.0, 2.0)
    d["dvol50"] = 50e6
    r = L.robustness(d, d["flag"].to_numpy().astype(bool), n_boot=300)
    assert not r["passed"] and any("out-earn" in x for x in r["reasons"])


def test_the_excess_interval_is_a_ratio_so_stocks_with_many_observations_do_not_distort_it():
    # One date. 50 stocks with 100 observations each and 1,000 stocks with one. Everything flagged earns
    # +0.20, everything else 0, so the excess is a constant 0.20 - (the date's average). Resampling stocks
    # changes how many observations are drawn; a ratio of sums stays put, a sum over a fixed count does not.
    rng = np.random.default_rng(6)
    d0 = pd.Timestamp("2002-01-04")
    rows = [(f"B{i}", d0) for i in range(50) for _ in range(100)] + [(f"S{i}", d0) for i in range(1000)]
    d = pd.DataFrame(rows, columns=["symbol", "date"])
    flagged = (d["symbol"].str.startswith("B") | (d["symbol"].str[1:].astype(int, errors="ignore").where(d["symbol"].str.startswith("S"), 0).astype(int) < 500)).to_numpy()
    d["runner"] = 0
    d["ret"] = np.where(flagged, 0.20, 0.0) + rng.normal(0, 0.001, len(d))
    d["bad"] = 0
    e = L.excess_return(d, flagged, n_boot=400)
    assert e["excess_mean"] > 0
    assert (e["excess_hi"] - e["excess_lo"]) < 0.05 * e["excess_mean"], "interval blew up: not a ratio of sums"


def test_excess_mode_judges_a_trend_setup_on_what_it_earns_not_on_how_often_it_doubles():
    rng = np.random.default_rng(8)
    d = _panel_for_excess(2.0, 2.0)
    d["dvol50"] = 50e6
    flagged = d["flag"].to_numpy().astype(bool)
    # a "trend" setup: fewer big runs than average, but steadier and higher average return
    d["runner"] = np.where(flagged, (rng.uniform(size=len(d)) < 0.02).astype(int), d["runner"])
    d["ret"] = np.where(flagged, rng.normal(0.12, 0.10, len(d)), rng.normal(0.0, 0.25, len(d)))
    d["bad"] = (d["ret"] <= -0.3).astype(int)
    timing = L.robustness(d, flagged, n_boot=300, primary="timing")
    excess = L.robustness(d, flagged, n_boot=300, primary="excess")
    assert not timing["passed"]                                                                  # it does not time runs
    assert excess["passed"], excess["reasons"]                                                    # but it does out-earn
    assert excess["liquid_excess_lo"] > 0


def test_excess_mode_still_rejects_a_setup_that_does_not_out_earn_among_liquid_stocks():
    d = _panel_for_excess(2.0, 2.0)
    d["dvol50"] = 50e6
    r = L.robustness(d, d["flag"].to_numpy().astype(bool), n_boot=300, primary="excess")
    assert not r["passed"]


def test_excess_mode_rejects_a_setup_that_out_earns_only_among_illiquid_stocks():
    rng = np.random.default_rng(10)
    n, dates = 1200, pd.date_range("2002-01-04", periods=20, freq="28D")
    rows = [(f"S{i}", d, i < n // 5, 50e6 if i % 2 else 1e6) for i in range(n) for d in dates]
    d = pd.DataFrame(rows, columns=["symbol", "date", "flag", "dvol50"])
    illiquid = d["dvol50"] < 10e6
    d["ret"] = rng.normal(0.0, 0.2, len(d))
    d.loc[d["flag"] & illiquid, "ret"] += 0.25                      # the edge exists only where nothing can be traded
    d["runner"] = 0
    d["bad"] = (d["ret"] <= -0.3).astype(int)
    m = d["flag"].to_numpy()
    r = L.robustness(d, m, n_boot=300, primary="excess")
    assert not r["passed"] and any("among liquid stocks" in x for x in r["reasons"])
    assert r["excess_lo"] > 0                                        # overall it looks fine, which is the trap
