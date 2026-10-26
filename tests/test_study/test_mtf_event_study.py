"""scripts/mtf_event_study.py must judge a multi-timeframe signature against what ALL liquid stocks did on the
SAME market-confirmation day, and must not read the future, lean on a rising market, or let in-sample
years decide.

The failures these tests guard against, each mutation-tested:
  * a market "event" that is really a 3-week dip (streak length not required), or a drawdown measured from
    an all-time high that was set years before the window instead of the prior 52 weeks;
  * a rising market credited to the signature because the control mean is pooled over all weeks instead of
    matched to each trade's own event week;
  * 2022 (the in-sample cross-sections the signatures were chosen on) or immature events leaking into the
    held-out verdict set;
  * the verdict judged at the 95% level instead of the Bonferroni 98.75% level;
  * placebo weeks drawn near a real event (they would share its market);
  * stock features that change when later bars are cut off, or illiquid stocks reaching the analysed rows;
  * a limited smoke run writing to the real trial ledger.
No test touches the network or the real ledger (learn.REGISTRY_DIR is redirected to tmp_path).
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from setup_engine import learn as L
from setup_engine import mtf
from setup_engine.screen import Screen

ROOT = Path(__file__).resolve().parents[2]


def _load():
    spec = importlib.util.spec_from_file_location("mtf_event_study", ROOT / "scripts" / "mtf_event_study.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mtf_event_study"] = mod
    spec.loader.exec_module(mod)
    return mod


S = _load()


@pytest.fixture(autouse=True)
def _ledger_in_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "REGISTRY_DIR", tmp_path / "registry")


# ── synthetic benches ─────────────────────────────────────────────────────────────────────────────

def _bench(closes, start="2010-01-04"):
    """Five identical business-day bars per weekly close (o = h = l = c)."""
    closes = np.asarray(closes, dtype=float)
    idx = pd.bdate_range(start, periods=5 * len(closes))
    c = np.repeat(closes, 5)
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1000.0}, index=idx)


def _path(start, segments):
    """Weekly closes: from ``start``, linear legs of (weeks, end value)."""
    out = [float(start)]
    for n, end in segments:
        out += list(np.linspace(out[-1], end, n + 1)[1:])
    return np.array(out)


def _crossings(closes):
    """Independent re-computation: (k, streak length, drawdown vs prior 52 weeks, vs all-time high) per cross above."""
    c = pd.Series(closes)
    sma = c.rolling(30, min_periods=30).mean()
    out = []
    for k in range(1, len(c)):
        if c[k] > sma[k] and c[k - 1] < sma[k - 1]:
            j = k - 1
            while j >= 0 and np.isfinite(sma[j]) and c[j] < sma[j]:
                j -= 1
            low = c[j + 1:k].min()
            out.append((k, k - 1 - j, low / c[max(0, k - 52):k].max() - 1, low / c[:k].max() - 1))
    return out


def _event_idx(ev, start="2010-01-04"):
    first = mtf.week_label([pd.Timestamp(start)])[0]
    return [int((w - first).days // 7) for w in ev["week"]]


# ── 1. the event rule ─────────────────────────────────────────────────────────────────────────────

def test_one_event_at_the_first_close_back_above_the_line_after_a_long_decline():
    closes = _path(100, [(60, 200), (12, 150), (40, 260)])
    cr = _crossings(closes)
    assert len(cr) == 1 and cr[0][1] >= S.EV_MIN_BELOW                        # the construction is what we think it is
    ev = S.confirmation_events(_bench(closes))
    assert _event_idx(ev) == [cr[0][0]] and len(ev) == 1
    assert ev["below_weeks"].iloc[0] == cr[0][1]
    assert ev["drawdown"].iloc[0] == pytest.approx(-0.25)
    assert list(ev.columns) == ["week", "last_day", "below_weeks", "drawdown", "year", "in_sample", "mature", "episode"]


def test_a_dip_of_seven_weeks_is_not_an_event():
    closes = _path(100, [(60, 200), (4, 150), (10, 215), (30, 260)])
    cr = _crossings(closes)
    assert len(cr) == 1 and cr[0][1] == 7 and cr[0][2] <= -S.EV_MIN_DD         # 7 weeks below, deep enough: only the length blocks it
    assert len(S.confirmation_events(_bench(closes))) == 0


def test_a_ten_percent_dip_is_not_an_event_even_after_a_long_streak():
    closes = _path(100, [(60, 200), (12, 178), (6, 178), (20, 230)])
    cr = _crossings(closes)
    assert len(cr) == 1 and cr[0][1] >= S.EV_MIN_BELOW and cr[0][2] > -S.EV_MIN_DD   # long enough, too shallow
    assert len(S.confirmation_events(_bench(closes))) == 0


def test_two_separate_ten_week_dips_give_two_events_in_different_episodes():
    closes = _path(100, [(60, 200), (10, 150), (50, 260), (10, 195), (40, 300)])
    ev = S.confirmation_events(_bench(closes))
    assert _event_idx(ev) == [k for k, *_ in _crossings(closes)] and len(ev) == 2
    assert ev["episode"].tolist() == [0, 1]
    assert (ev["week"].iloc[1] - ev["week"].iloc[0]).days / 7 >= S.EPISODE_GAP_WEEKS


def test_events_less_than_26_weeks_apart_share_an_episode():
    closes = _path(100, [(60, 200), (10, 150), (6, 210), (4, 110), (6, 110), (50, 330)])
    cr = _crossings(closes)
    assert [k for k, *_ in cr] == [74, 95] and all(x[1] >= S.EV_MIN_BELOW and x[2] <= -S.EV_MIN_DD for x in cr)
    ev = S.confirmation_events(_bench(closes))
    assert _event_idx(ev) == [74, 95]
    assert (ev["week"].iloc[1] - ev["week"].iloc[0]).days / 7 == 21 < S.EPISODE_GAP_WEEKS
    assert ev["episode"].tolist() == [0, 0]


def test_a_one_week_pop_above_the_line_inside_a_decline_only_fires_at_the_final_cross():
    closes = np.r_[_path(100, [(60, 200)]), [190, 170, 150, 200], np.linspace(150, 120, 9), np.linspace(125, 260, 40)]
    cr = _crossings(closes)
    assert len(cr) == 2 and cr[0][1] < S.EV_MIN_BELOW <= cr[1][1]            # the pop follows a short streak, the final cross a long one
    ev = S.confirmation_events(_bench(closes))
    assert _event_idx(ev) == [cr[1][0]]


def test_the_drawdown_is_measured_against_the_prior_52_weeks_not_the_all_time_high():
    t = np.arange(150)
    closes = np.r_[_path(100, [(60, 200), (30, 100)]), 100 + 3 * np.sin(2 * np.pi * t / 24)]
    cr = _crossings(closes)
    late = [x for x in cr[1:] if x[1] >= S.EV_MIN_BELOW]
    assert len(late) >= 4 and all(x[2] > -S.EV_MIN_DD > x[3] for x in late)   # long enough, shallow vs 52w, deep vs the old peak
    ev = S.confirmation_events(_bench(closes))
    assert _event_idx(ev) == [cr[0][0]]                                      # only the genuine post-crash recovery


def test_mature_means_156_more_weekly_bars_exist_and_in_sample_is_the_year():
    closes = _path(100, [(60, 200), (12, 150), (40, 260), (200, 600)])
    ev = S.confirmation_events(_bench(closes))
    k = _event_idx(ev)[0]
    assert bool(ev["mature"].iloc[0]) == (k + S.MATURE_WEEKS <= len(closes) - 1)
    short = S.confirmation_events(_bench(closes[:k + 100]))
    assert len(short) == 1 and not short["mature"].iloc[0]
    assert ev["in_sample"].iloc[0] == (ev["year"].iloc[0] in S.IN_SAMPLE_YEARS)


def test_events_use_only_complete_weekly_bars_up_to_as_of():
    closes = _path(100, [(60, 200), (12, 150), (40, 260)])
    b = _bench(closes)
    k = _event_idx(S.confirmation_events(b))[0]
    cutoff = b.index[5 * k + 2]                       # Wednesday of the event week: the week is not complete
    assert len(S.confirmation_events(b, cutoff)) == 0
    assert len(S.confirmation_events(b, b.index[5 * k + 4])) == 1


# ── 2. constants ──────────────────────────────────────────────────────────────────────────────────

def test_the_pre_registered_constants_and_signatures():
    assert set(S.SIGNATURE_NAMES) == set(mtf.SIGNATURES)
    assert S.VERDICT_SIGNATURES == ("S1", "S2", "S3", "S4") and S.PRIMARY_EXIT == "R3"
    assert S.ALPHA_VERDICT == 0.0125 and S.ALPHA == 0.05 and S.MIN_EVENTS_WITH_TRADES == 5
    assert S.HORIZONS == (26, 52, 156) and S.N_PLACEBO == 40 and S.IN_SAMPLE_YEARS == (2022,)
    assert set(S.EXITS) == {"R1", "R2", "R3"} and S.OUTCOMES[2] == "ret_R3"
    for name in S.SIGNATURE_NAMES:
        assert isinstance(Screen(mtf.SIGNATURES[name]), Screen)
    for col in S.SIGNATURE_COLUMNS:
        assert col in mtf.MTF_COLUMNS
    doc = " ".join(S.__doc__.split())
    assert " ".join(S.EXPLORATION_DISCLOSURE.split()) in doc               # the disclosure is in the docstring verbatim


# ── hand panels for analyze ───────────────────────────────────────────────────────────────────────

def _blank_columns():
    return (["symbol", "week"] + list(S.SIGNATURE_COLUMNS) + list(S.OUTCOMES) + [f"fmax_{S.HORIZONS[-1]}"])


def _panel(rows):
    """rows: dicts with week, symbol and any of the columns (the rest NaN)."""
    df = pd.DataFrame(rows).reindex(columns=_blank_columns())
    df["week"] = pd.to_datetime(df["week"])
    return df


def _events(*specs):
    """specs: (week, in_sample, mature)"""
    return pd.DataFrame({"week": pd.to_datetime([s[0] for s in specs]), "in_sample": [s[1] for s in specs],
                         "mature": [s[2] for s in specs], "episode": list(range(len(specs)))})


NO_EVENTS = _events()
NO_SPANS = pd.DataFrame(columns=["symbol", "first_date", "last_date"])


def _above(week, sym, r, above=True, **kw):
    d = {"week": week, "symbol": sym, "w_close": 11.0 if above else 9.0, "w_sma30": 10.0, "ret_R3": r}
    d.update(kw)
    return d


# ── 3. the date-matched control ───────────────────────────────────────────────────────────────────

def test_the_control_is_the_mean_of_all_liquid_stocks_on_the_same_event_week_not_the_pooled_mean():
    A, B = "2005-01-07", "2006-03-03"
    rows = [_above(A, "a1", 0.3), _above(A, "a2", 0.0, above=False), _above(A, "a3", 0.0, above=False), _above(A, "a4", 0.0, above=False),
            _above(B, "b1", 0.5), _above(B, "b2", 0.5), _above(B, "b3", 0.5), _above(B, "b4", 0.4, above=False)]
    res = S.analyze(_panel(rows), {"spy": _events((A, False, True), (B, False, True)), "smh": NO_EVENTS}, [], NO_SPANS,
                    None, n_boot=50, seed=1)
    blk = res["signatures"]["B0"]["spy"]["ret_R3"]["held_out_mature"]
    # date-matched: week A control 0.075, week B control 0.475 -> excess (0.225 + 3 * 0.025) / 4
    assert blk["mean_excess"] == pytest.approx(0.075)
    assert blk["mean_r"] == pytest.approx(0.45) and blk["mean_control"] == pytest.approx(0.375)
    assert blk["mean_excess"] != pytest.approx(0.45 - 2.2 / 8)               # the pooled-mean version would say 0.175
    pe = {e["week"].date().isoformat(): e for e in res["signatures"]["B0"]["per_event"]["spy"]}
    assert pe[A]["control_mean"] == pytest.approx(0.075) and pe[A]["excess"] == pytest.approx(0.225)
    assert pe[B]["control_mean"] == pytest.approx(0.475) and pe[B]["excess"] == pytest.approx(0.025)
    assert pe[A]["n_liquid"] == 4 and pe[A]["n_sig"] == 1 and pe[B]["n_sig"] == 3


def test_the_verdict_interval_is_wider_than_the_95_percent_interval():
    rng = np.random.default_rng(3)
    rows = []
    for k, wk in enumerate(pd.date_range("2003-01-03", periods=10, freq="52W-FRI")):
        for s in range(12):
            sig = s < 4
            rows.append(_above(wk, f"s{s}", float(rng.normal(0.1 if sig else 0.0, 0.2)), above=sig))
    evs = _events(*[(w, False, True) for w in sorted({r["week"] for r in rows})])
    res = S.analyze(_panel(rows), {"spy": evs, "smh": NO_EVENTS}, [], NO_SPANS, None, n_boot=2000, seed=5)
    blk = res["signatures"]["B0"]["spy"]["ret_R3"]["held_out_mature"]
    assert blk["ci_event_verdict"]["lo"] < blk["ci_event"]["lo"] and blk["ci_event_verdict"]["hi"] > blk["ci_event"]["hi"]
    assert blk["ci_event_verdict"]["mean"] == pytest.approx(blk["ci_event"]["mean"])


# ── 4. the verdict ────────────────────────────────────────────────────────────────────────────────

def _vin(mean=0.05, lo=0.01, n_ev=10, n_pos=6, gentle=0.02, harsh=0.01, lo95=None):
    per = [{"n_sig": 3, "excess": 0.1 if i < n_pos else -0.1, "split": "held_out_mature"} for i in range(n_ev)]
    return {"excess_event": {"mean": mean, "lo": lo if lo95 is None else lo95},
            "excess_event_verdict": {"mean": mean, "lo": lo}, "per_event": per,
            "adj_excess_gentle": gentle, "adj_excess_harsh": harsh}


def test_the_verdict_passes_only_when_all_three_conditions_hold():
    v = S.verdict(_vin())
    assert v["verdict"] == "edge" and all(v["checks"].values())
    # boundary: five events with trades and a strict majority (3 of 5) is enough
    assert S.verdict(_vin(n_ev=5, n_pos=3))["verdict"] == "edge"


@pytest.mark.parametrize("kw, failed", [
    (dict(lo=-0.001), "1_"), (dict(mean=-0.01, lo=-0.05), "1_"),
    (dict(n_ev=4, n_pos=4), "2_"), (dict(n_ev=10, n_pos=5), "2_"), (dict(n_ev=5, n_pos=2), "2_"),
    (dict(gentle=-0.01), "3_"), (dict(harsh=-0.01), "3_"),
])
def test_the_verdict_fails_for_each_condition_individually(kw, failed):
    v = S.verdict(_vin(**kw))
    assert v["verdict"] == "no demonstrated edge"
    bad = [k for k, ok in v["checks"].items() if not ok]
    assert len(bad) == 1 and bad[0].startswith(failed)


def test_events_without_trades_do_not_count_toward_the_minimum():
    vin = _vin(n_ev=4, n_pos=4)
    vin["per_event"] += [{"n_sig": 0, "excess": None, "split": "held_out_mature"}] * 6
    assert S.verdict(vin)["verdict"] == "no demonstrated edge"


def test_the_verdict_uses_the_bonferroni_interval_not_the_95_percent_one():
    v = S.verdict(_vin(lo=-0.01, lo95=0.02))                              # 95% lower bound > 0, 98.75% lower bound < 0
    assert v["verdict"] == "no demonstrated edge" and not v["checks"]["1_event_clustered_excess_positive_and_ci_lo_positive"]


def test_a_missing_input_makes_the_verdict_incomplete():
    v = S.verdict(_vin(harsh=None))
    assert v["verdict"].startswith("incomplete (missing:") and "adj_excess_harsh" in v["verdict"] and v["checks"] is None
    v = S.verdict({**_vin(), "excess_event_verdict": {"mean": None, "lo": None}})
    assert v["verdict"].startswith("incomplete (missing:") and "event-clustered excess" in v["verdict"]


def test_2022_and_immature_events_are_outside_the_held_out_verdict_set():
    held, i22, imm = "2005-01-07", "2022-08-12", "2025-05-16"
    rows = []
    for wk in (held, i22, imm):
        rows += [_above(wk, f"{wk}-sig", 0.5), _above(wk, f"{wk}-x", 0.0, above=False)]
    evs = _events((held, False, True), (i22, True, True), (imm, False, False))
    res = S.analyze(_panel(rows), {"spy": evs, "smh": NO_EVENTS}, [], NO_SPANS, None, n_boot=50, seed=1)
    b = res["signatures"]["B0"]["spy"]["ret_R3"]
    assert b["held_out_mature"]["n_trades"] == 1 and b["held_out_mature"]["n_events_with_trades"] == 1
    assert b["in_sample_2022"]["n_trades"] == 1 and b["immature"]["n_trades"] == 1 and b["all"]["n_trades"] == 3
    assert [e["split"] for e in res["signatures"]["B0"]["per_event"]["spy"]] == ["held_out_mature", "in_sample_2022", "immature"]
    vin = res["signatures"]["S1"]["verdict_input"]
    assert [e["split"] for e in vin["per_event"]] == ["held_out_mature"]
    assert S.event_split(True, True) == "in_sample_2022" and S.event_split(False, False) == "immature"
    # no delisted table -> the survivorship input is missing -> incomplete, never a silent pass
    assert res["verdict"]["S1"]["verdict"].startswith("incomplete (missing:")


def test_survivorship_arithmetic_by_hand():
    sv = S.survivorship([2015, 2015], [0.5, 0.3], [0.1, 0.1], {2015: 0.2})
    assert sv["f"] == pytest.approx(0.2) and sv["f_sig_harsh"] == pytest.approx(0.3)
    assert sv["adj_sig_gentle"] == pytest.approx(0.8 * 0.4 + 0.2 * -0.5)
    assert sv["adj_ctl_gentle"] == pytest.approx(0.8 * 0.1 + 0.2 * -0.5)
    assert sv["adj_excess_gentle"] == pytest.approx(0.24)
    assert sv["adj_sig_harsh"] == pytest.approx(0.7 * 0.4 - 0.3) and sv["adj_ctl_harsh"] == pytest.approx(0.8 * 0.1 - 0.2)
    assert sv["adj_excess_harsh"] == pytest.approx(0.10)
    assert sv["breakeven_f_sig"] == pytest.approx((0.4 + 0.12) / 1.4)         # at that share the harsh excess is zero
    assert sv["breakeven_mult"] == pytest.approx(sv["breakeven_f_sig"] / 0.2)
    be = sv["breakeven_f_sig"]
    assert (1 - be) * 0.4 + be * S.L_HARSH == pytest.approx(sv["adj_ctl_harsh"])
    # years without a share are skipped; no usable year -> no adjustment
    assert S.survivorship([2015, 1999], [0.5, 0.3], [0.1, 0.1], {2015: 0.2})["f"] == pytest.approx(0.2)
    assert S.survivorship([1999], [0.5], [0.1], {2015: 0.2})["adj_excess_harsh"] is None
    assert S.survivorship([], [], [], {})["adj_excess_gentle"] is None


def test_the_reference_delisted_share_floors_the_years_before_2013():
    spans = pd.DataFrame({"symbol": [f"S{i}" for i in range(10)], "first_date": pd.Timestamp("2000-01-03"),
                          "last_date": pd.Timestamp("2026-10-06")})
    delisted = pd.DataFrame({"name": ["A Inc", "B Inc", "C Inc"], "assetType": "Stock",
                             "ipoDate": ["2004-01-01", "2010-01-01", "2010-01-01"],
                             "delistingDate": ["2005-06-01", "2016-06-01", "2016-06-01"]})
    t = S.survivorship_table(delisted, spans, range(2003, 2020))
    f_stock = t["table"]["f_stock"]
    assert f_stock[2005] == pytest.approx(1 / 11) and f_stock[2008] == 0.0 and f_stock[2014] == pytest.approx(2 / 12)
    assert f_stock[2018] == 0.0
    assert t["f_ref"] == pytest.approx(4 * (2 / 12) / 7)                       # 2013-2016 at 2/12, 2017-2019 at 0
    assert t["f_used"][2005] == pytest.approx(t["f_ref"])                      # 1/11 is below f_ref: the floor binds
    assert t["f_used"][2008] == pytest.approx(t["f_ref"])
    assert t["f_used"][2014] == pytest.approx(2 / 12) and t["f_used"][2018] == 0.0   # post-2013 years keep their own share


# ── 5. placebo weeks ──────────────────────────────────────────────────────────────────────────────

def test_placebo_weeks_are_reproducible_mature_and_far_from_every_event():
    weeks = pd.date_range("2002-01-04", periods=1100, freq="W-FRI")
    events = [weeks[100], weeks[260], weeks[420], weeks[700]]
    as_of = weeks[-1]
    a = S.placebo_weeks(weeks, events, as_of)
    b = S.placebo_weeks(weeks, events, as_of)
    assert a == b and a == sorted(a) and len(a) == len(set(a)) == S.N_PLACEBO
    last = len(weeks) - 1
    for w in a:
        assert w >= pd.Timestamp(S.PLACEBO_START)
        assert weeks.get_loc(w) + S.MATURE_WEEKS <= last                           # mature
        assert min(abs((w - e).days) for e in events) > S.PLACEBO_EXCLUSION_WEEKS * 7
    assert len(S.placebo_weeks(weeks, events, as_of, n=5)) == 5
    short = pd.date_range("2010-01-01", periods=200, freq="W-FRI")               # only 44 mature candidates remain
    assert len(S.placebo_weeks(short, [], short[-1], n=100)) == 200 - S.MATURE_WEEKS


# ── 6. stock rows: liquid only, ranks among liquid ────────────────────────────────────────────────

def _stock(n=900, seed=1, start="2012-01-02", price=50.0, vol=1_000_000.0):
    rng = np.random.default_rng(seed)
    close = price * np.exp(np.cumsum(rng.normal(0.0004, 0.015, n)))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, n)))
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": np.full(n, vol)}, index=idx)


def test_stock_rows_returns_only_liquid_rows_and_always_a_span():
    spy = _stock(900, seed=9)
    weeks = [mtf.week_label([spy.index[k]])[0] for k in (400, 500, 600)]
    ok = S.stock_rows("LIQ", _stock(900, seed=1), spy, weeks, spy.index[-1])
    assert list(ok["rows"]["week"]) == weeks and ok["rows"]["symbol"].eq("LIQ").all()
    assert ok["span"] == ("LIQ", spy.index[0], spy.index[-1])
    wf = mtf.mtf_features(_stock(900, seed=1), spy)
    assert mtf.is_liquid(wf.loc[weeks]).all()
    illiquid = S.stock_rows("PENNY", _stock(900, seed=2, vol=100.0), spy, weeks, spy.index[-1])
    assert illiquid["rows"] is None and illiquid["span"][0] == "PENNY"
    young = S.stock_rows("NEW", _stock(900, seed=3), spy, [mtf.week_label([spy.index[100]])[0]], spy.index[-1])
    assert young["rows"] is None                                                # fewer than 60 weekly bars


def test_stock_rows_has_features_and_outcomes_for_every_exit_and_horizon():
    spy = _stock(1600, seed=9)
    wk = [mtf.week_label([spy.index[400]])[0]]
    row = S.stock_rows("LIQ", _stock(1600, seed=1), spy, wk, spy.index[-1])["rows"].iloc[0]
    for name in S.EXITS:
        assert np.isfinite(row[f"ret_{name}"]) and row[f"reason_{name}"] in ("stop", "gap_stop", "r1_weekly_sma30",
                                                                           "r2_monthly_sma10", "r3_trail30", "time", "end")
        assert np.isfinite(row[f"weeks_{name}"]) and np.isfinite(row[f"mfe_{name}"])
    for h in S.HORIZONS:
        assert np.isfinite(row[f"hret_{h}"])
    assert np.isfinite(row["fmax_156"]) and row["fmax_156"] > -1.0
    for col in S.SIGNATURE_COLUMNS:
        assert col in row.index


def test_the_cross_sectional_rank_is_computed_among_the_analysed_liquid_rows_only():
    wk = "2005-01-07"
    rows = [dict(_above(wk, f"s{i}", 0.1), w_ret13=0.01 * i, w_ret26=0.01 * i, w_slope30=0.01) for i in range(10)]
    res = S.analyze(_panel(rows), {"spy": _events((wk, False, True)), "smh": NO_EVENTS}, [], NO_SPANS, None, n_boot=20, seed=1)
    # S1 = top 20% of w_ret13 (rank >= 0.8, inclusive) + above the SMA + rising slope: of ten liquid rows the
    # ranks are 0.1 .. 1.0, so exactly the rows ranked 0.8, 0.9 and 1.0
    assert res["signatures"]["S1"]["n_mask_rows"] == 3
    assert res["signatures"]["S1"]["spy"]["ret_R3"]["held_out_mature"]["n_trades"] == 3


# ── 7. causality of the stock features ────────────────────────────────────────────────────────────

def test_cutting_off_later_daily_bars_leaves_the_event_week_features_unchanged():
    daily, spy = _stock(1100, seed=4), _stock(1100, seed=5)
    cut = 700
    wk = mtf.week_label([daily.index[cut]])[0]
    cut = daily.index.get_loc(wk)                                             # the event week's last trading day
    full = S.stock_rows("X", daily, spy, [wk], daily.index[-1])["rows"]
    trunc = S.stock_rows("X", daily.iloc[:cut + 1], spy, [wk], daily.index[cut])["rows"]
    assert len(full) == len(trunc) == 1
    for col in S.SIGNATURE_COLUMNS:
        assert np.allclose(full[col].to_numpy(float), trunc[col].to_numpy(float), rtol=1e-12, equal_nan=True), col
    assert np.isnan(trunc["ret_R3"].iloc[0]) and np.isfinite(full["ret_R3"].iloc[0])   # the outcome needs later bars, the features do not


# ── strict JSON ───────────────────────────────────────────────────────────────────────────────────

def test_jsonable_is_strict():
    o = {"a": float("nan"), "b": np.float64("inf"), "c": np.int64(3), "d": pd.Timestamp("2020-02-03"), 5: [np.bool_(True), (1, 2)],
         "e": np.datetime64("2021-01-02"), "f": pd.NaT}
    j = json.dumps(S._jsonable(o), allow_nan=False)
    assert json.loads(j) == {"a": None, "b": None, "c": 3, "d": "2020-02-03", "5": [True, [1, 2]], "e": "2021-01-02", "f": None}


# ── 8. main on a tmp history ──────────────────────────────────────────────────────────────────────

def _write_world(tmp_path):
    pytest.importorskip("pyarrow")
    hist, bench = tmp_path / "hist", tmp_path / "bench"
    hist.mkdir()
    bench.mkdir()
    start, n_weeks = "2008-01-07", 520
    closes = _path(100, [(60, 200), (14, 150), (60, 300), (386, 600)])[:n_weeks]
    for sym, scale in (("SPY", 1.0), ("SMH", 0.5)):
        b = _bench(closes * scale, start)
        b.index.name = "date"
        b.to_parquet(bench / f"{sym}.parquet")

    def long(d, sym):
        x = d.reset_index().rename(columns={"index": "date"})
        x.insert(1, "symbol", sym)
        return x

    n = 5 * n_weeks
    stocks = [long(_stock(n, seed=100 + i, start=start), f"T{i:02d}") for i in range(24)]
    pd.concat(stocks[:12]).to_parquet(hist / "batch_0000.parquet", index=False)
    pd.concat(stocks[12:]).to_parquet(hist / "batch_0001.parquet", index=False)
    csv = tmp_path / "delisted.csv"
    pd.DataFrame({"symbol": ["D1", "D2"], "name": ["Alpha Inc", "Beta Corp"], "exchange": ["NYSE"] * 2,
                  "assetType": ["Stock", "Stock"], "ipoDate": ["2004-01-01", "2006-05-01"],
                  "delistingDate": ["2010-03-01", "2013-08-01"], "status": ["Delisted"] * 2}).to_csv(csv, index=False)
    return hist, bench, csv, closes


def _strict(text):
    def bad(token):
        raise ValueError(f"non-strict JSON token {token}")
    return json.loads(text, parse_constant=bad)


def test_main_limited_run_writes_no_ledger_and_strict_json_and_full_run_writes_sixteen_records(tmp_path, capsys):
    hist, bench, csv, closes = _write_world(tmp_path)
    assert len(S.confirmation_events(_bench(closes, "2008-01-07"))) >= 1          # the synthetic market has a qualifying dip
    out, rows_out = tmp_path / "res.json", tmp_path / "rows.parquet"
    base = ["--history", str(hist), "--bench-dir", str(bench), "--delisted", str(csv), "--workers", "1",
            "--out", str(out), "--rows-out", str(rows_out), "--n-placebo", "3"]
    assert L.trial_count() == 0

    assert S.main(base + ["--limit-batches", "1"]) == 0
    assert L.trial_count() == 0, "a limited run must not touch the ledger"
    js = _strict(out.read_text())
    assert js["limited_run"] is True and js["ledger_trials_before"] == 0 and js["ledger_trials_after"] == 0
    assert set(js["verdict"]) == set(S.VERDICT_SIGNATURES)
    for v in js["verdict"].values():
        assert v["verdict"].split(" ")[0] in ("edge", "no", "incomplete")
    assert js["sizes"]["spy"]["n_events"] >= 1 and js["sizes"]["rows_total"] > 0
    assert js["disclosure"] == S.EXPLORATION_DISCLOSURE and js["verdict_rule"] == S.VERDICT_RULE
    text = capsys.readouterr().out
    assert "VERDICT S1" in text and "SPY events" in text and "ledger trials: 0 -> 0" in text
    assert rows_out.exists()

    assert S.main(base) == 0
    lines = [json.loads(x) for x in (tmp_path / "registry" / "trials.jsonl").read_text().splitlines()]
    assert len(lines) == 16 and L.trial_count() == 16
    assert lines[0]["stage"] == S.LEDGER_STAGE and lines[0]["event"] == "preregistered"
    assert set(lines[0]["signatures"]) == set(S.SIGNATURE_NAMES) and lines[0]["disclosure"] == S.EXPLORATION_DISCLOSURE
    assert [(x["signature"], x["exit"]) for x in lines[1:]] == [(s, e) for s in S.SIGNATURE_NAMES for e in S.EXITS]
    assert all(x["stage"] == S.LEDGER_STAGE and "n_trades_heldout" in x and "excess_mean" in x and "excess_lo" in x
               and "data_end" in x for x in lines[1:])
    js = _strict(out.read_text())
    assert js["ledger_trials_before"] == 0 and js["ledger_trials_after"] == 16 and js["limited_run"] is False
    assert js["signatures"]["B0"]["survivorship"] is not None                     # the delisted table was used
