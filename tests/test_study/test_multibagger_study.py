"""scripts/multibagger_study.py: the forward-multiple arithmetic must be exact, and the study must not quietly flatter itself.

The failures being guarded against: a forward window that includes the start month (every stock then "never
loses"), an eligibility boundary that moves, a trailing high of the wrong length, a dead (terminal) stock that
silently vanishes instead of being held flat, a monthly close taken from the wrong day, and a results file
that is not strict JSON or that hides the survivorship caveat. Every expected number below is computed by hand
from the formulas that build the synthetic grid, never from the code under test.
"""
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


M = _load("multibagger_study")

IDX = pd.period_range("2000-01", "2026-09", freq="M")     # 321 months
T = len(IDX)
HZ = M.H


def P(s):
    return pd.Period(s, "M")


def pos(s):
    """Month number counted from 2000-01."""
    return (P(s) - IDX[0]).n


def grid(**series):
    return pd.DataFrame(series, index=IDX)


# ── synthetic world: 3 winners, 20 decayers, 200 flat stocks (223 stocks, all eligible every month) ──

def winner_path():
    """100 through 2004-12; log-linear to 2.0 at 2005-12; log-linear to 120 at 2010-12; 120 afterwards."""
    k = np.arange(T)
    a, b, c = pos("2004-12"), pos("2005-12"), pos("2010-12")
    out = np.full(T, 100.0)
    seg = (k >= a) & (k <= b)
    out[seg] = 100.0 * np.exp(math.log(2.0 / 100.0) * (k[seg] - a) / (b - a))
    seg = (k > b) & (k < c)
    out[seg] = 2.0 * np.exp(math.log(120.0 / 2.0) * (k[seg] - b) / (c - b))
    out[k >= c] = 120.0
    return out


def w_close(year):
    """Hand formula for a winner's December close in ``year``."""
    if year <= 2004:
        return 100.0
    if year == 2005:
        return 2.0
    if year < 2010:
        return 2.0 * 60 ** ((year - 2005) / 5)
    return 120.0


@pytest.fixture(scope="module")
def world():
    k = np.arange(T)
    c, d = {}, {}
    for i in range(1, 4):
        c[f"W{i}"], d[f"W{i}"] = winner_path(), 5e5
    for i in range(1, 21):
        c[f"D{i}"], d[f"D{i}"] = 50.0 * 0.99 ** k, 5e5
    for i in range(1, 201):
        c[f"N{i}"], d[f"N{i}"] = 20.0 * np.where(k % 2 == 0, 1.02, 1 / 1.02), 1e6
    C, DV = grid(**c), grid(**d)
    F = {**M.features(C, DV), **M.forward(C)}
    dec = [P(f"{y}-12") for y in range(2001, 2016)]
    return {"C": C, "DV": DV, "F": F, "S": M.starts(C, DV, dec, F)}


# ── forward arithmetic ───────────────────────────────────────────────────────────────────────────────

def test_forward_excludes_start_month_and_includes_month_i_plus_h_small_grid():
    """A window that starts at i, or stops before i+H, gives wrong multiples for every stock."""
    C = pd.DataFrame({"X": [10.0, 20.0, 5.0, 40.0, 1.0]}, index=pd.period_range("2020-01", periods=5, freq="M"))
    f = M.forward(C, H=2)
    assert f["fmax"]["X"].iloc[:3].tolist() == [2.0, 2.0, 8.0]      # max(20,5)/10, max(5,40)/20, max(40,1)/5
    assert f["fend"]["X"].iloc[:3].tolist() == [0.5, 2.0, 0.2]       # 5/10, 40/20, 1/5
    assert f["fmax"]["X"].iloc[3:].isna().all() and f["fend"]["X"].iloc[3:].isna().all()


def test_forward_ignores_missing_months_inside_the_window():
    """A gap month must neither poison the window max with NaN nor be treated as zero."""
    C = pd.DataFrame({"X": [10.0, np.nan, 30.0, np.nan]}, index=pd.period_range("2020-01", periods=4, freq="M"))
    f = M.forward(C, H=2)
    assert f["fmax"]["X"].iloc[0] == 3.0
    assert np.isnan(f["fend"]["X"].iloc[1])          # start month itself missing
    assert f["fend"]["X"].iloc[0] == 3.0


def test_forward_horizon_inclusive_at_i_plus_h_and_exclusive_after(world):
    """fmax uses months i+1 .. i+H: a spike at i+H counts, a spike at i+H+1 does not."""
    i = pos("2003-12")
    on = np.full(T, 10.0)
    on[i + HZ] = 30.0
    off = np.full(T, 10.0)
    off[i + HZ + 1] = 30.0
    C = grid(ON=on, OFF=off)
    f = M.forward(C)
    assert f["fmax"].loc[P("2003-12"), "ON"] == pytest.approx(3.0)
    assert f["fend"].loc[P("2003-12"), "ON"] == pytest.approx(3.0)
    assert f["fmax"].loc[P("2003-12"), "OFF"] == pytest.approx(1.0)
    assert f["fend"].loc[P("2003-12"), "OFF"] == pytest.approx(1.0)


def test_forward_declining_stock_has_fmax_below_one_because_start_month_is_excluded():
    """If the start month were in the window a declining stock would show fmax == 1.0 and never 'lose'."""
    C = grid(DEC=100.0 * 0.9 ** np.arange(T))
    f = M.forward(C)
    valid = f["fmax"]["DEC"].dropna()
    assert len(valid) == T - HZ
    assert (valid < 1.0).all()
    assert valid.iloc[0] == pytest.approx(0.9)       # the next month is the best the window offers


def test_forward_is_nan_when_window_runs_past_the_grid():
    """The last H start months have no complete window and must be NaN, not a partial answer."""
    f = M.forward(grid(X=np.full(T, 10.0)))
    assert f["fmax"].loc[P("2016-09"), "X"] == pytest.approx(1.0)    # 2016-09 + 120 months = 2026-09, the last month
    assert np.isnan(f["fmax"].loc[P("2016-10"), "X"])
    assert np.isnan(f["fend"].loc[P("2016-10"), "X"])


# ── hand-checked winners ─────────────────────────────────────────────────────────────────────────────

def test_winner_forward_values_match_the_formulas(world):
    """The 50x winner's fmax, dd60 and neighbours follow directly from the log-linear construction."""
    F = world["F"]
    assert F["fmax"].loc[P("2005-12"), "W1"] == pytest.approx(60.0, abs=1e-9)
    assert F["dd60"].loc[P("2005-12"), "W1"] == pytest.approx(2.0 / 100.0 - 1, abs=1e-12)
    c2006 = 2.0 * 60 ** (12 / 60)
    assert world["C"].loc[P("2006-12"), "W1"] == pytest.approx(c2006, rel=1e-9)
    assert F["fmax"].loc[P("2006-12"), "W1"] == pytest.approx(120.0 / c2006, rel=1e-9)
    assert F["fmax"].loc[P("2004-12"), "W1"] == pytest.approx(1.2)


def test_base_rates_pooled_and_by_year_match_hand_counts(world):
    """The pooled share is 3 / (15 x 223); 2005 alone is 3 / 223; the other thresholds follow from the same formulas."""
    br = M.base_rates(world["S"])
    n_all = 15 * 223
    assert br["pooled"]["n"] == n_all and br["pooled"]["n_stocks"] == 223
    assert br["pooled"]["share_fmax_ge_50x"] == pytest.approx(3 / n_all)
    assert br["by_year"]["2005"]["n"] == 223
    assert br["by_year"]["2005"]["share_fmax_ge_50x"] == pytest.approx(3 / 223)
    assert br["by_year"]["2006"]["share_fmax_ge_50x"] == 0.0
    assert list(br["by_year"]) == [str(y) for y in range(2001, 2016)]
    # a winner's peak multiple from Dec y is 120 / close(y); the end multiple is the same (the tail is flat at 120)
    for m in M.PEAK_MULTS:
        n_years = sum(1 for y in range(2001, 2016) if 120.0 / w_close(y) >= m)
        assert br["pooled"][f"share_fmax_ge_{m}x"] == pytest.approx(3 * n_years / n_all), m
    for m in M.END_MULTS:
        n_years = sum(1 for y in range(2001, 2016) if 120.0 / w_close(y) >= m)
        assert br["pooled"][f"share_fend_ge_{m}x"] == pytest.approx(3 * n_years / n_all), m
    assert [br["pooled"][f"share_fmax_ge_{m}x"] * n_all for m in (5, 10, 20, 50, 100)] == pytest.approx([12, 9, 6, 3, 0])


def test_winners_list_has_the_three_winners_at_their_best_start(world):
    """winners() must name each 50x stock once, at the start with the largest multiple."""
    w = M.winners(world["S"])
    assert [x["symbol"] for x in w] == ["W1", "W2", "W3"]
    assert all(x["start"] == "2005-12" for x in w)
    assert all(x["multiple"] == pytest.approx(60.0, abs=1e-9) for x in w)
    assert all(x["n_starts"] == 1 for x in w)
    assert M.winners(world["S"], mult=1000) == []


def test_precision_table_matches_hand_counts(world):
    """n / n_50x / precision / recall / lift per profile, derived from which December starts each rule selects."""
    S = world["S"]
    prec = {r["profile"]: r for r in M.precision(S)}
    assert list(prec) == list(M.PROFILES) and len(prec) == 5
    n_all, pooled = 15 * 223, 3 / (15 * 223)
    # D1..D20 stay above $5 through 2015-12 (50 x 0.99**191 = 7.3); only W at 2005-12 (2.0) and 2006-12 (4.53) are under $5
    for name in ("price < $5", "price < $5 and >=70% below 60m high"):
        r = prec[name]
        assert (r["n"], r["n_50x"]) == (6, 3)
        assert r["precision_50x"] == 0.5 and r["recall_50x"] == 1.0 and r["precision_10x"] == 1.0
        assert r["lift_50x"] == pytest.approx(0.5 / pooled)
    r = prec["price < $10 and >=80% below 60m high"]            # 2007-12 closes at 10.3, outside
    assert (r["n"], r["n_50x"], r["recall_50x"]) == (6, 3, 1.0)
    r = prec["up >=100% in 12 months"]                            # W doubles each December 2006..2010: 60**0.2 = 2.27
    assert (r["n"], r["n_50x"]) == (15, 0)
    assert r["precision_50x"] == 0.0 and r["recall_50x"] == 0.0 and r["lift_50x"] == 0.0
    assert r["precision_10x"] == pytest.approx(6 / 15)            # peaks of 26.5x (2006) and 11.6x (2007) for 3 stocks
    r = prec["dollar volume < $2M/day"]                           # every stock trades under $2M
    assert (r["n"], r["n_50x"]) == (n_all, 3)
    assert r["precision_50x"] == pytest.approx(pooled) and r["lift_50x"] == pytest.approx(1.0)
    assert r["precision_10x"] == pytest.approx(9 / n_all)


def test_profile_of_winners_is_the_2005_trough(world):
    """The winners' median start is the $2.00, 98%-below-high month; 'all' covers every start."""
    S = world["S"]
    pw = M.profile(S, (S["fmax"] >= M.WINNER).to_numpy())
    assert pw["n"] == 3 and pw["n_stocks"] == 3
    assert pw["median_close"] == pytest.approx(2.0)
    assert pw["median_dd60"] == pytest.approx(-0.98)
    assert pw["median_ret12"] == pytest.approx(2.0 / 100.0 - 1)
    assert pw["median_vol12"] == pytest.approx(0.0, abs=1e-6)      # a steady log-linear decline has no dispersion
    allp = M.profile(S, np.ones(len(S), dtype=bool))
    assert allp["n"] == 15 * 223 and allp["median_close"] > 0
    assert M.profile(S, np.zeros(len(S), dtype=bool))["median_close"] is None


# ── eligibility, trailing high, terminal ─────────────────────────────────────────────────────────────

def test_eligibility_boundaries_price_and_dollar_volume_are_inclusive():
    """$1.00 and $300,000 are eligible; $0.999 and $299,999 are not."""
    C = grid(P_EDGE=1.0, P_LOW=0.999, V_EDGE=5.0, V_LOW=5.0, OK=20.0)
    DV = grid(P_EDGE=5e5, P_LOW=5e5, V_EDGE=300_000.0, V_LOW=299_999.0, OK=1e6)
    F = {**M.features(C, DV), **M.forward(C)}
    S = M.starts(C, DV, [P("2003-12")], F)
    assert sorted(S["symbol"]) == ["OK", "P_EDGE", "V_EDGE"]
    assert S.attrs["n_incomplete"] == 0


def test_dd60_uses_a_sixty_month_window_that_includes_the_current_month():
    """Peak 100 at 2001-01, 20 from 2002-01: 59 months later the peak is still in view (-0.8); a 36-month window would say 0."""
    k = np.arange(T)
    s = np.full(T, 20.0)
    s[k < pos("2001-01")] = 50.0
    s[pos("2001-01")] = 100.0
    s[(k > pos("2001-01")) & (k < pos("2002-01"))] = 60.0
    C = grid(X=s)
    dd = M.features(C, grid(X=1e6))["dd60"]["X"]
    assert pos("2005-12") - pos("2001-01") == 59
    assert dd.loc[P("2005-12")] == pytest.approx(20.0 / 100.0 - 1)
    assert dd.loc[P("2006-01")] == pytest.approx(20.0 / 60.0 - 1)  # month 60: the 100 has left; the 60s of 2001 are still in view
    assert dd.loc[P("2006-12")] == pytest.approx(0.0)              # now the window is 2002-01..2006-12: all 20s
    assert np.isnan(dd.iloc[pos("2000-11")])                       # fewer than 12 months of history
    assert dd.iloc[pos("2000-12")] == pytest.approx(0.0)           # 12 months of 50s: the first defined value


def _terminal_world():
    t = np.full(T, np.nan)
    i = pos("2003-12")
    t[i:i + 13] = np.linspace(10.0, 12.0, 13)                     # peak 12 at 2004-12
    t[i + 13:pos("2006-01")] = 10.0
    t[pos("2006-01"):pos("2008-01")] = 6.0
    t[pos("2008-01"):pos("2008-06") + 1] = 4.0                     # last month 2008-06 at price 4
    C = grid(T=t, F1=20.0)
    DV = grid(T=5e5, F1=1e6)
    DV["T"] = np.where(np.isnan(t), np.nan, 5e5)
    return C, DV


def test_terminal_stock_is_held_flat_and_included_otherwise_excluded_as_incomplete():
    """A delisted stock must keep its terrible outcome (fend 0.4) instead of dropping out for 'incomplete data'."""
    C, DV = _terminal_world()
    month = [P("2003-12")]
    held = M.hold_terminal_flat(C, frozenset({"T"}))
    F = {**M.features(held, DV), **M.forward(held)}
    S = M.starts(held, DV, month, F)
    row = S[S["symbol"] == "T"].iloc[0]
    assert row["fend"] == pytest.approx(0.4)                       # 4 / 10, the flat tail at 2013-12
    assert row["fmax"] == pytest.approx(1.2)                       # the 12 at 2004-12, the tail never exceeds it
    assert sorted(S["symbol"]) == ["F1", "T"] and S.attrs["n_incomplete"] == 0

    F2 = {**M.features(C, DV), **M.forward(C)}
    S2 = M.starts(C, DV, month, F2)
    assert sorted(S2["symbol"]) == ["F1"]
    assert S2.attrs["n_incomplete"] == 1


def test_hold_terminal_flat_fills_only_after_the_last_valid_month_and_only_terminal_symbols():
    """Interior gaps stay missing and non-terminal symbols are untouched."""
    idx = pd.period_range("2020-01", periods=6, freq="M")
    C = pd.DataFrame({"T": [1.0, np.nan, 3.0, 4.0, np.nan, np.nan], "S": [1.0, 2.0, np.nan, np.nan, np.nan, np.nan],
                      "GONE": np.nan}, index=idx)
    out = M.hold_terminal_flat(C, frozenset({"T", "GONE", "NOT_IN_GRID"}))
    assert out["T"].iloc[0] == 1.0 and np.isnan(out["T"].iloc[1])
    assert out["T"].iloc[2:].tolist() == [3.0, 4.0, 4.0, 4.0]
    assert out["S"].iloc[2:].isna().all()
    assert out["GONE"].isna().all()
    assert C["T"].iloc[4:].isna().all()                            # the input is not modified


# ── wide / monthly_from_daily / last_complete_month ──────────────────────────────────────────────────

def test_wide_grid_is_contiguous_so_gaps_do_not_shift_the_horizon():
    """A stock with a missing month must still have one grid row per calendar month."""
    panel = pd.DataFrame({"symbol": ["X", "X", "Y"], "month": [P("2020-01"), P("2020-04"), P("2020-03")],
                          "close": [1.0, 2.0, 3.0], "dvol": [9.0, 9.0, 9.0]})
    w = M.wide(panel, "close")
    assert list(w.index) == [P("2020-01"), P("2020-02"), P("2020-03"), P("2020-04")]
    assert w["X"].isna().tolist() == [False, True, True, False]
    assert w["Y"].tolist()[2] == 3.0 and w["Y"].isna().sum() == 3


def test_monthly_from_daily_takes_the_last_day_close_max_high_min_low_and_mean_dollar_volume():
    """The month's close is the LAST trading day's, whatever the row order; broken bars are dropped first."""
    d = pd.DataFrame({
        "symbol": "X",
        "date": pd.to_datetime(["2020-01-06", "2020-01-02", "2020-02-03", "2020-01-03", "2020-02-04", "2020-01-07"]),
        "open": [3.0, 1.0, 4.0, 2.0, 5.0, np.nan],
        "high": [3.5, 1.5, 4.5, 2.5, 5.5, 9.0],
        "low": [2.5, 0.8, 3.5, 1.5, 4.5, 1.0],
        "close": [3.0, 1.0, 4.0, 2.0, 5.0, 100.0],
        "volume": [300.0, 100.0, 40.0, 200.0, 50.0, 1e9]})
    m = M.monthly_from_daily(d)
    assert list(m.columns) == ["symbol", "month", "close", "high", "low", "dvol", "n_days"]
    assert m["month"].tolist() == [P("2020-01"), P("2020-02")]
    jan, feb = m.iloc[0], m.iloc[1]
    assert jan["close"] == 3.0 and jan["high"] == 3.5 and jan["low"] == 0.8 and jan["n_days"] == 3
    assert jan["dvol"] == pytest.approx((1 * 100 + 2 * 200 + 3 * 300) / 3)
    assert feb["close"] == 5.0 and feb["n_days"] == 2 and feb["dvol"] == pytest.approx((4 * 40 + 5 * 50) / 2)


def test_monthly_from_daily_drops_impossible_bars_and_handles_empty():
    """A bar with high < low, a zero price or an unusable row must not reach the panel."""
    d = pd.DataFrame({"symbol": ["X", "X", "X"], "date": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]),
                      "open": [1.0, 1.0, 0.0], "high": [1.0, 0.5, 1.0], "low": [1.0, 0.9, 1.0],
                      "close": [1.0, 1.0, 1.0], "volume": [10.0, 10.0, 10.0]})
    m = M.monthly_from_daily(d)
    assert m["n_days"].tolist() == [1]                              # only the first bar is valid
    assert len(M.monthly_from_daily(d.iloc[1:])) == 0


def test_last_complete_month_depends_on_the_month_label():
    """Data through 2026-10-07 has no complete October; data through the last weekday of September completes September."""
    assert M.last_complete_month(pd.Timestamp("2026-10-07")) == P("2026-09")
    assert M.last_complete_month(pd.Timestamp("2026-09-30")) == P("2026-09")
    assert M.last_complete_month(pd.Timestamp("2026-09-29")) == P("2026-08")
    assert M.last_complete_month(pd.Timestamp("2026-10-30")) == P("2026-10")


# ── fallen variants (hand-built three-stock grid) ────────────────────────────────────────────────────

def _fallen_grid():
    """t counts months from 2003-12 (t = 0). Y turns up slowly, Z falls again, V is a 100x."""
    t = np.arange(T) - pos("2003-12")
    y = np.select([t <= 0, t <= 12, t == 13, t == 14], [100.0, 5.0, 5.5, 9.0], default=40.0)
    z = np.select([t <= 0, t <= 12], [100.0, 8.0], default=3.5)
    v = np.select([t <= 0, t <= 12], [100.0, 4.0], default=400.0)
    return grid(Y=y, Z=z, V=v), grid(Y=1e6, Z=1e6, V=1e6)


def test_fallen_variants_exact_counts_and_outcomes_on_a_hand_built_grid():
    """A / B / C / control sizes and their outcome shares, derived by hand from the three price paths."""
    C, DV = _fallen_grid()
    F = {**M.features(C, DV), **M.forward(C)}
    v = M.fallen_variants(C, DV, F)
    # A: Y t=1..14 (14), Z t=1..59 (59), V t=1..12 (12)
    assert (v["A"]["n"], v["A"]["n_stocks"]) == (85, 3)
    assert (v["B"]["n"], v["B"]["n_stocks"]) == (2, 1)            # Y at t=13 (5.5 > 5.05) and t=14 (9 > 5.45)
    assert (v["C"]["n"], v["C"]["n_stocks"]) == (1, 1)            # only t=14 has a 3-month return >= 20% (0.8; t=13 is 0.1)
    assert (v["control"]["n"], v["control"]["n_stocks"]) == (36, 3)   # t=1..9 for Y, V; t=1..9 and 13..21 for Z

    a_end = [40 / 5.0] * 12 + [40 / 5.5, 40 / 9.0] + [3.5 / 8.0] * 12 + [1.0] * 47 + [100.0] * 12
    assert len(a_end) == 85
    A = v["A"]
    assert A["share_10x"] == pytest.approx(12 / 85) and A["share_20x"] == pytest.approx(12 / 85)
    assert A["share_50x"] == pytest.approx(12 / 85)               # V's 100x
    assert A["share_fell50"] == pytest.approx(12 / 85)            # Z t=1..12: 3.5 / 8 = 0.4375
    assert A["share_end_below"] == pytest.approx(12 / 85)
    assert A["median_end"] == pytest.approx(1.0)
    assert A["mean_end_winsor"] == pytest.approx(sum(a_end) / 85)  # 99th pct = 100: the clip does not bite

    a, b = 40 / 9.0, 40 / 5.5
    assert v["B"]["mean_end_winsor"] == pytest.approx((1.01 * a + 0.99 * b) / 2)   # the larger value is clipped to 0.01a + 0.99b
    assert v["B"]["mean_end_winsor"] < (a + b) / 2
    assert v["B"]["median_end"] == pytest.approx((a + b) / 2)
    assert v["B"]["share_10x"] == 0.0 and v["B"]["share_fell50"] == 0.0 and v["B"]["share_end_below"] == 0.0
    assert v["C"]["mean_end_winsor"] == pytest.approx(a) and v["C"]["median_end"] == pytest.approx(a)

    ctl = v["control"]
    assert ctl["share_10x"] == pytest.approx(9 / 36) and ctl["share_fell50"] == pytest.approx(9 / 36)
    assert ctl["share_end_below"] == pytest.approx(9 / 36)
    assert ctl["median_end"] == pytest.approx((1.0 + 8.0) / 2)
    assert ctl["mean_end_winsor"] == pytest.approx((9 * 0.4375 + 9 * 1.0 + 9 * 8.0 + 9 * 100.0) / 36)


def test_fallen_variants_agree_with_an_independent_pure_python_recount():
    """Recount A/B/C/control and their outcomes with plain loops over the three price lists (no pandas rolling)."""
    C, DV = _fallen_grid()
    F = {**M.features(C, DV), **M.forward(C)}
    v = M.fallen_variants(C, DV, F)
    first, last = pos("2001-01"), pos("2015-12")
    got = {k: [] for k in M.VARIANTS}
    for sym in C.columns:
        s = [float(x) for x in C[sym]]
        for i in range(first, last + 1):
            if i + HZ >= len(s):
                continue
            hi = max(s[max(0, i - 59):i + 1])
            sma = sum(s[i - 9:i + 1]) / 10
            r3 = s[i] / s[i - 3] - 1
            if not (1.0 <= s[i] <= 10.0 and s[i] / hi - 1 <= -0.70):
                continue
            rec = {"fend": s[i + HZ] / s[i], "fmax": max(s[i + 1:i + HZ + 1]) / s[i], "fmin": min(s[i + 1:i + 37]) / s[i]}
            got["A"].append(rec)
            if s[i] > sma:
                got["B"].append(rec)
                if r3 >= 0.20:
                    got["C"].append(rec)
            if s[i] < sma:
                got["control"].append(rec)
    for name in M.VARIANTS:
        recs = got[name]
        assert v[name]["n"] == len(recs), name
        assert v[name]["share_10x"] == pytest.approx(sum(r["fmax"] >= 10 for r in recs) / len(recs)), name
        assert v[name]["share_fell50"] == pytest.approx(sum(r["fmin"] <= 0.5 for r in recs) / len(recs)), name
        assert v[name]["share_end_below"] == pytest.approx(sum(r["fend"] < 1 for r in recs) / len(recs)), name
        assert v[name]["median_end"] == pytest.approx(float(np.median([r["fend"] for r in recs]))), name


def test_winsor_mean_clips_only_the_upper_tail_at_the_99th_percentile():
    """[1]*99 + [101]: the 99th percentile is 2.0, so the mean is 1.01, not 2.0; an empty sample gives None."""
    x = np.array([1.0] * 99 + [101.0])
    assert M._winsor_mean(x) == pytest.approx(1.01)
    assert M._winsor_mean(np.array([0.0] + [1.0] * 99)) == pytest.approx(0.99)      # the low outlier is NOT lifted
    assert M._winsor_mean(np.array([])) is None


# ── today's pool and fundamentals ────────────────────────────────────────────────────────────────────

def _pool_world():
    k = np.arange(T)
    fall = np.where(k < pos("2024-01"), 100.0, 5.0)
    C = grid(FALL=fall, DEAD=fall, THIN=fall, UP=20.0, CHEAP=np.where(k < pos("2024-01"), 100.0, 0.9))
    DV = grid(FALL=3e6, DEAD=3e6, THIN=999_999.0, UP=3e6, CHEAP=3e6)
    return C, DV


def test_today_pool_applies_the_fallen_rule_the_volume_floor_and_excludes_terminal_stocks():
    """FALL qualifies; DEAD is terminal; THIN is under $1M a day; UP is not fallen; CHEAP is under $1."""
    C, DV = _pool_world()
    F = {**M.features(C, DV), **M.forward(C)}
    pool = M.today_pool(C, DV, F, P("2026-09"), frozenset({"DEAD"}))
    assert pool["symbol"].tolist() == ["FALL"]
    assert list(pool.columns) == ["symbol", "close", "dvol", "dd60", "above_sma10", "ret3"]
    assert pool["close"].iloc[0] == 5.0 and pool["dd60"].iloc[0] == pytest.approx(-0.95)
    assert not bool(pool["above_sma10"].iloc[0])                   # flat at 5.0: not above its own average
    both = M.today_pool(C, DV, F, P("2026-09"), frozenset())
    assert both["symbol"].tolist() == ["DEAD", "FALL"]
    assert len(M.today_pool(C, DV, F, P("2030-01"), frozenset())) == 0


def test_fundamentals_check_scores_five_tests_and_never_raises():
    """Missing fields fail their test; a fetch that raises gives a row that fails all fundamentals but keeps the trend flag."""
    pool = pd.DataFrame({"symbol": ["GOOD", "NOCASH", "BOOM", "NODATA"], "above_sma10": [True, True, False, True]})
    info = {
        "GOOD": {"marketCap": 5e8, "revenueGrowth": 0.15, "grossMargins": 0.4, "totalCash": 100.0, "totalDebt": 50.0},
        "NOCASH": {"marketCap": 3e8, "revenueGrowth": 0.10, "grossMargins": 0.25, "totalCash": 10.0, "totalDebt": 50.0,
                   "freeCashflow": 5.0},
    }

    def fetch(sym):
        if sym == "BOOM":
            raise RuntimeError("network down")
        return info.get(sym)

    out = M.fundamentals_check(pool, fetch=fetch).set_index("symbol")
    assert out.loc["GOOD", "n_pass"] == 5
    assert bool(out.loc["NOCASH", "pass_cap"]) and bool(out.loc["NOCASH", "pass_growth"]) and bool(out.loc["NOCASH", "pass_margin"])
    assert bool(out.loc["NOCASH", "pass_cash"])                    # net cash is negative but free cash flow is positive
    assert out.loc["BOOM", "n_pass"] == 0 and not bool(out.loc["BOOM", "pass_trend"])
    assert out.loc["NODATA", "n_pass"] == 1                        # only the trend flag
    assert len(M.fundamentals_check(pool.iloc[:0], fetch=fetch)) == 0


# ── plumbing: panel build, strict JSON, end to end ───────────────────────────────────────────────────

def _daily(symbol, dates, seed, base=20.0, volume=1e6):
    rng = np.random.default_rng(seed)
    close = base * np.exp(np.cumsum(rng.normal(0.0, 0.01, len(dates))))
    return pd.DataFrame({"symbol": symbol, "date": dates, "open": close, "high": close * 1.01, "low": close * 0.99,
                         "close": close, "volume": volume})


def _history(tmp_path, terminal=None):
    pytest.importorskip("pyarrow")
    hist = tmp_path / "hist"
    hist.mkdir()
    full = pd.bdate_range(end="2026-09-30", periods=7000)
    fallen = _daily("FALLEN", full, 3, base=1.0, volume=5e5)
    px = np.where(full < pd.Timestamp("2023-06-01"), 100.0, 5.0)
    for col in ("open", "high", "low", "close"):
        fallen[col] = px * {"open": 1.0, "high": 1.01, "low": 0.99, "close": 1.0}[col]
    dead_dates = full[full <= pd.Timestamp("2008-06-30")]
    pd.concat([_daily("AAA", full, 1), fallen]).to_parquet(hist / "batch_0000.parquet", index=False)
    pd.concat([_daily("BBB", full, 2, base=30.0), _daily("TTT", dead_dates, 4, base=15.0)]).to_parquet(
        hist / "batch_0001.parquet", index=False)
    if terminal is not None:
        (hist / "terminal.json").write_text(json.dumps(terminal))
    return hist


def _strict(text):
    return json.loads(text, parse_constant=lambda c: pytest.fail(f"non-strict JSON constant {c}"))


def test_build_panel_reads_every_batch_and_respects_limit(tmp_path):
    """build_panel covers all batch files (in-process, workers=1); --limit keeps only the first batch files."""
    hist = _history(tmp_path)
    one = M.build_panel(hist, workers=1)
    assert one.groupby("symbol")["month"].apply(lambda m: m.is_monotonic_increasing).all()
    assert sorted(one["symbol"].unique()) == ["AAA", "BBB", "FALLEN", "TTT"]
    only = M.build_panel(hist, workers=1, limit=1)
    assert sorted(only["symbol"].unique()) == ["AAA", "FALLEN"]
    assert M.load_terminal(hist) == frozenset()
    with pytest.raises(FileNotFoundError):
        M.build_panel(tmp_path / "nowhere")


def test_jsonable_turns_non_finite_numbers_and_periods_into_strict_json():
    """NaN / inf must become null and Periods strings, or json.dumps(allow_nan=False) raises."""
    obj = {"a": float("nan"), "b": np.float64("inf"), "c": P("2020-01"), "d": [np.int64(3), np.bool_(True)]}
    text = json.dumps(M._jsonable(obj), allow_nan=False)
    assert _strict(text) == {"a": None, "b": None, "c": "2020-01", "d": [3, True]}


def test_main_end_to_end_with_a_terminal_stock_is_strict_json_and_leads_with_survivorship(tmp_path, capsys):
    """main() returns 0, writes strict JSON and a pool CSV, announces the survivorship correction FIRST, and keeps terminal stocks out of the pool."""
    hist = _history(tmp_path, terminal=["TTT"])
    out, pool = tmp_path / "r.json", tmp_path / "p.csv"
    rc = M.main(["--history", str(hist), "--out", str(out), "--pool-out", str(pool), "--workers", "1"])
    assert rc == 0
    text = out.read_text()
    assert "NaN" not in text and "Infinity" not in text
    res = _strict(text)
    assert res["data"]["n_terminal"] == 1 and res["data"]["n_symbols"] == 4
    assert res["data"]["survivorship"] == "survivorship-corrected (1 delisted stocks held flat)"
    assert res["data"]["last_complete_month"] == "2026-09" and res["data"]["grid"][1] == "2026-09"
    assert list(res["base_rates"]["by_year"]) == [str(y) for y in range(2001, 2016)]
    assert len(res["precision"]) == 5 and set(res["fallen_variants"]) == {"A", "B", "C", "control"}
    assert res["pool"]["n"] == 1 and res["pool"]["candidates"][0]["symbol"] == "FALLEN"
    assert "no demonstrated edge" in res["pool"]["note"] and res["preregistration"].startswith("Fixed before")
    assert pd.read_csv(pool)["symbol"].tolist() == ["FALLEN"]
    first = capsys.readouterr().out.splitlines()[0]
    assert first.startswith("survivorship-corrected")


@pytest.mark.parametrize("terminal", [None, [], ["ZZZ"]])
def test_main_without_terminal_stocks_says_survivors_only_first(tmp_path, capsys, terminal):
    """No terminal.json (or none present in the data) must print SURVIVORS ONLY first and never claim a correction."""
    hist = _history(tmp_path, terminal=terminal)
    out = tmp_path / "r.json"
    rc = M.main(["--history", str(hist), "--out", str(out), "--pool-out", str(tmp_path / "p.csv"), "--workers", "1"])
    assert rc == 0
    stdout = capsys.readouterr().out
    assert stdout.splitlines()[0].startswith("SURVIVORS ONLY")
    assert "survivorship-corrected" not in stdout
    res = _strict(out.read_text())
    assert res["data"]["n_terminal"] == 0 and res["data"]["survivorship"].startswith("SURVIVORS ONLY")
    assert "survivorship-corrected" not in out.read_text()


def test_main_limit_reads_only_the_first_batch_file(tmp_path, capsys):
    """--limit N is a smoke test over the first N batch files and is labelled as such."""
    hist = _history(tmp_path, terminal=["TTT"])
    out = tmp_path / "r.json"
    rc = M.main(["--history", str(hist), "--limit", "1", "--out", str(out), "--pool-out", str(tmp_path / "p.csv"), "--workers", "1"])
    assert rc == 0
    res = _strict(out.read_text())
    assert res["data"]["n_symbols"] == 2 and res["data"]["limited_run"] is True and res["data"]["n_terminal"] == 0
    assert "limited run" in capsys.readouterr().out
