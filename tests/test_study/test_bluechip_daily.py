"""scripts/bluechip_daily.py must describe TODAY'S S&P 500 state without reading a partial bar, ranking against
stocks outside the index, or dressing a small tilt up as a forecast.

The failures these tests guard against, each mutation-tested where it is a logic rule:
  * a bar for a day that is still in progress (or a Friday bar that has not closed) leaking into the weekly state
    or into last_close, so the report changes with the time of day;
  * percentile ranks computed over every downloaded stock instead of the index members, so an outsider with the
    strongest returns shifts every real member's rank;
  * group boundaries off by one comparison (50% extended, 8% off the high, rank 0.80), which silently moves
    names between "leader" and "extended";
  * a full-confluence name double counted as a plain breakout leader;
  * the reference stop level computed from the wrong fraction;
  * NaN or Infinity in the JSON (invalid for strict parsers), a fundamentals failure aborting the whole report, a
    missing "not investment advice" caveat.
Everything is synthetic: fake fetchers, fake yfinance module, no network, nothing written outside tmp_path.
"""
import importlib.util
import json
import sys
import types
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from setup_engine import mtf

ROOT = Path(__file__).resolve().parents[2]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


B = _load("bluechip_daily")


# ── synthetic world ───────────────────────────────────────────────────────────────────────────────

IDX = pd.bdate_range(end="2026-10-09", periods=700)        # last bar: Friday 2026-10-09


def _frame(close, volume=2e6, index=IDX):
    close = np.asarray(close, dtype="float64")
    prev = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame({"open": prev, "high": np.maximum(prev, close) * 1.003, "low": np.minimum(prev, close) * 0.997,
                         "close": close, "volume": np.full(len(close), float(volume))}, index=index)


def _walk(seed, n=700, start=50.0, sigma=0.015, mu=0.0003):
    rng = np.random.default_rng(seed)
    return start * np.exp(np.cumsum(rng.normal(mu, sigma, n)))


def _leader(seed, fast_last=None, rate=1.004):
    """Flat at 50 for 500 bars, then +0.4% a day; with ``fast_last`` the last bars rise 1.2% a day instead."""
    rng = np.random.default_rng(seed)
    close = 50.0 * (1 + rng.normal(0, 0.002, 700))
    growth = np.full(200, rate)
    if fast_last:
        growth[-fast_last:] = 1.012
    close[500:] = 50.0 * np.cumprod(growth) * (1 + rng.normal(0, 0.002, 200))
    fr = _frame(close)
    fr.iloc[-5:, fr.columns.get_loc("volume")] *= 3
    return fr


def _world():
    """SPY plus twelve members: ten random walks, LEADER and EXTENDED."""
    rng = np.random.default_rng(1)
    spy = _frame(100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.008, 700))), volume=5e7)
    daily = {f"RW{i}": _frame(_walk(seed)) for i, seed in enumerate(range(2, 12))}
    daily["LEADER"] = _leader(20)
    daily["EXTENDED"] = _leader(21, fast_last=100)
    return spy, daily


def _universe(symbols, sectors=("Tech", "Health")):
    return pd.DataFrame({
        "symbol": list(symbols), "raw_symbol": list(symbols), "security": [f"{s} Inc" for s in symbols],
        "sector": [sectors[i % len(sectors)] for i in range(len(symbols))], "sub_industry": "Software",
        "date_added": pd.Timestamp("2000-01-03"), "cik": "0000000001"})


@pytest.fixture(scope="module")
def world():
    spy, daily = _world()
    uni = _universe(daily)
    as_of = B.last_complete_day(spy.index, "2026-10-12")
    P = B.screen(daily, spy, uni, as_of)
    return spy, daily, uni, as_of, P


# ── calendar ──────────────────────────────────────────────────────────────────────────────────────

def test_last_complete_day_is_strictly_before_today():
    """Today's bar is partial whether or not the market has closed."""
    idx = pd.bdate_range("2026-10-05", "2026-10-12")                        # Mon .. Mon
    assert B.last_complete_day(idx, "2026-10-12") == pd.Timestamp("2026-10-09")
    assert B.last_complete_day(idx, "2026-10-13") == pd.Timestamp("2026-10-12")
    assert B.last_complete_day(idx, pd.Timestamp("2026-10-12 15:30")) == pd.Timestamp("2026-10-09")
    with pytest.raises(ValueError):
        B.last_complete_day(idx, "2026-10-05")


def test_asof_week_is_the_latest_complete_friday():
    """Off by a week and every symbol is skipped as stale, or the screen shows a week that is not complete."""
    assert B.asof_week(pd.Timestamp("2026-10-09")) == pd.Timestamp("2026-10-09")
    assert B.asof_week(pd.Timestamp("2026-10-08")) == pd.Timestamp("2026-10-02")
    assert B.asof_week(pd.Timestamp("2026-10-10")) == pd.Timestamp("2026-10-09")        # Saturday
    assert B.asof_week(pd.Timestamp("2026-10-12")) == pd.Timestamp("2026-10-09")        # Monday
    assert B.asof_week(pd.Timestamp("2026-10-05")) == pd.Timestamp("2026-10-02")


def test_asof_week_matches_the_last_weekly_row_mtf_builds_for_every_weekday():
    """asof_week must agree with what mtf_features really returns, or every stock is skipped as stale_week."""
    idx = pd.bdate_range(end="2026-10-12", periods=400)
    fr = _frame(_walk(5, n=400), index=idx)
    for as_of in pd.date_range("2026-10-05", "2026-10-12"):
        wf = mtf.mtf_features(fr[fr.index <= as_of], as_of=as_of)
        assert wf.index[-1] == B.asof_week(as_of), as_of


# ── groups on a hand-built panel ──────────────────────────────────────────────────────────────────

def _hand_panel():
    #  sym  S1     S2     S3     S4     px_sma30  off_high52  rank26
    spec = [
        ("A", True, True, True, True, 0.10, -0.01, 0.95),       # every signature: full confluence, not extended
        ("B", False, False, True, False, 0.10, -0.01, 0.50),    # S3 without S4
        ("C", True, False, False, False, 0.50, -0.01, 0.90),    # extended boundary: exactly +50%
        ("D", True, False, False, False, 0.4999, -0.01, 0.90),  # just under: not extended
        ("E", True, False, False, False, 0.10, -0.08, 0.90),    # exactly 8% off the high: not "near" (strict >)
        ("F", False, True, False, False, 0.10, -0.01, 0.40),    # beaten-down reclaim only
        ("G", True, False, False, False, 0.2499, -0.0799, 0.80),  # on every boundary but inside
        ("H", True, False, False, False, 0.10, -0.01, 0.7999),  # rank just under the cut
        ("I", True, False, False, False, 0.25, -0.01, 0.90),    # exactly +25%: not "not extended" (strict <)
        ("N", True, False, False, False, np.nan, np.nan, np.nan),  # NaN never satisfies a comparison
    ]
    df = pd.DataFrame(spec, columns=["symbol", "S1", "S2", "S3", "S4", "w_px_sma30", "off_high52", "xs_rank_ret26"])
    df["B0"] = True
    df["security"] = df["symbol"] + " Inc"
    df["sector"] = "Tech"
    df["w_close"] = 100.0
    df["trail30"] = 70.0
    df["xs_rank_ret13"] = df["xs_rank_ret26"]
    df["w_ret26"] = 0.1
    df["bm_w_px_sma30"] = 0.04
    return df


def test_groups_have_exactly_the_members_the_rules_say():
    """Boundary comparisons (>=, >, <) and the S3/S4 split, on rows whose membership is known by hand."""
    G = B.groups(_hand_panel())
    names = lambda k: set(G[k]["symbol"])
    assert names("full_confluence") == {"A"}
    assert names("breakout_leaders") == {"B"}
    assert "A" not in names("breakout_leaders")
    assert names("beaten_down_reclaim") == {"A", "F"}
    assert names("leaders_not_extended") == {"A", "G"}
    assert names("extended") == {"C"}
    assert "D" not in names("extended") and "E" not in names("leaders_not_extended")


def test_groups_are_sorted_by_26_week_rank_best_first_and_carry_the_documented_columns():
    """Unsorted output hides the strongest names; a missing column breaks the markdown/JSON consumers."""
    G = B.groups(_hand_panel())
    assert list(G["leaders_not_extended"]["symbol"]) == ["A", "G"]                         # 0.95 then 0.80
    assert list(G["beaten_down_reclaim"]["symbol"]) == ["A", "F"]                           # 0.95 then 0.40
    assert list(G["full_confluence"].columns) == ["symbol", "security", "sector", "w_close", "off_high52",
                                                  "w_px_sma30", "xs_rank_ret13", "xs_rank_ret26", "trail30"]


def test_sector_leadership_counts_and_median_by_hand():
    """Sector counts, the strict '> 0' for above the 30-week average, and the sort by median return."""
    P = pd.DataFrame({
        "symbol": ["X1", "X2", "Y1", "Y2"], "sector": ["Tech", "Tech", "Health", "Health"],
        "w_px_sma30": [0.10, 0.20, 0.02, 0.0], "w_ret26": [0.30, 0.10, 0.70, 0.50],
        "S1": [True, False, False, True]})
    t = B.sector_table(P)
    assert list(t.columns) == ["sector", "n", "above_30w", "share_above_30w", "s1_leaders", "median_ret26"]
    assert list(t["sector"]) == ["Health", "Tech"]                                      # 0.60 before 0.20
    h, tech = t.iloc[0], t.iloc[1]
    assert (h["n"], h["above_30w"], h["share_above_30w"], h["s1_leaders"]) == (2, 1, 0.5, 1)
    assert h["median_ret26"] == pytest.approx(0.60)
    assert (tech["n"], tech["above_30w"], tech["share_above_30w"], tech["s1_leaders"]) == (2, 2, 1.0, 1)
    assert tech["median_ret26"] == pytest.approx(0.20)


def test_regime_reads_the_benchmark_and_the_breadth():
    """The headline line: SPY against its 30-week average and the share of names above theirs."""
    P = _hand_panel()
    r = B.groups(P)["regime"]
    assert r["benchmark"] == "SPY" and r["bench_px_sma30"] == pytest.approx(0.04) and r["bench_above_30w"] is True
    assert r["n"] == 10 and r["share_above_30w"] == pytest.approx(9 / 10)                # N is NaN: not above
    line = B.regime_line(r)
    assert "SPY is 4.0% above its 30-week average" in line and "90%" in line


def test_groups_of_an_empty_panel_do_not_crash():
    """A day on which every member is skipped must still produce a report."""
    G = B.groups(B._empty_panel())
    assert all(len(G[g]) == 0 for g in B.STOCK_GROUPS)
    assert G["regime"]["n"] == 0 and G["regime"]["bench_above_30w"] is False
    assert "no stocks screened" in B.regime_line(G["regime"])


# ── screen() end to end on synthetic daily data ───────────────────────────────────────────────────

def test_trail30_is_seventy_percent_of_the_weekly_close(world):
    """The reference level must be 0.70 x the latest weekly close (100 -> 70)."""
    spy, _, _, _, P = world
    np.testing.assert_allclose(P["trail30"], 0.7 * P["w_close"])
    P2 = B.screen({"FLAT": _frame(np.full(700, 100.0))}, spy, _universe(["FLAT"]), pd.Timestamp("2026-10-09"))
    assert P2.loc[0, "w_close"] == pytest.approx(100.0)
    assert P2.loc[0, "trail30"] == pytest.approx(70.0)


def test_screen_flags_the_planted_leader_and_the_extended_name(world):
    """The pipeline must see a steady climber as an RS leader and a parabolic one as extended."""
    _, daily, _, as_of, P = world
    assert len(P) == 12 and P.attrs["skipped"] == {}
    row = P.set_index("symbol")
    assert row.loc["LEADER", "S1"]
    rank13 = row["xs_rank_ret13"]
    assert rank13["EXTENDED"] == rank13.max() == 1.0                                       # the parabolic name is first
    assert rank13["LEADER"] == rank13.drop("EXTENDED").max()                               # the steady climber is second
    assert row.loc["EXTENDED", "S1"] and row.loc["EXTENDED", "w_px_sma30"] >= 0.5
    G = B.groups(P)
    assert "EXTENDED" in set(G["extended"]["symbol"])
    assert row.loc["LEADER", "w_px_sma30"] < 0.5 and "LEADER" not in set(G["extended"]["symbol"])
    assert (P["week"] == B.asof_week(as_of)).all()
    assert list(P["symbol"]) == sorted(P["symbol"])


def test_every_group_equals_the_rule_recomputed_from_the_panel(world):
    """The grouping code against an independent restatement of the rules, written with literal numbers."""
    *_, P = world
    G = B.groups(P)
    price_above = (P["w_close"] > P["w_sma30"]).to_numpy()
    s1 = ((P["xs_rank_ret13"] >= 0.8) & price_above & (P["w_slope30"] > 0)).to_numpy()
    s2 = ((P["w_dd52"] <= -0.30) & price_above & (P["xs_rank_ret26"] >= 0.8)).to_numpy()
    assert (P["S1"].to_numpy() == s1).all() and (P["S2"].to_numpy() == s2).all()
    assert (P["B0"].to_numpy() == price_above).all()
    sym = P["symbol"]
    assert set(G["full_confluence"]["symbol"]) == set(sym[P["S4"]])
    assert set(G["breakout_leaders"]["symbol"]) == set(sym[P["S3"] & ~P["S4"]])
    assert set(G["beaten_down_reclaim"]["symbol"]) == set(sym[P["S2"]])
    near = P["S1"] & (P["w_dd52"] > -0.08) & (P["w_px_sma30"] < 0.25) & (P["xs_rank_ret26"] >= 0.80)
    assert set(G["leaders_not_extended"]["symbol"]) == set(sym[near])
    assert set(G["extended"]["symbol"]) == set(sym[P["S1"] & (P["w_px_sma30"] >= 0.50)])
    assert G["regime"]["n"] == 12
    assert G["regime"]["share_above_30w"] == pytest.approx(float((P["w_px_sma30"] > 0).mean()))


def test_confluence_and_reclaim_names_reach_their_groups_through_the_pipeline():
    """With a rising market a steady leader is full confluence (S4) and must NOT also be listed as a plain
    breakout leader; a stock that fell 60% and recovered strongly is a beaten-down reclaim (S2)."""
    rng = np.random.default_rng(1)
    spy = _frame(100.0 * np.exp(np.cumsum(rng.normal(0.0015, 0.008, 700))), volume=5e7)
    close = np.full(700, 50.0)
    close[450:520] = 50.0 * (20.0 / 50.0) ** (np.arange(1, 71) / 70)                  # 50 -> 20
    close[520:570] = 20.0
    close[570:] = 20.0 * (34.0 / 20.0) ** (np.arange(1, 131) / 130)                     # 20 -> 34 in 26 weeks
    close = close * (1 + np.random.default_rng(30).normal(0, 0.002, 700))
    daily = {f"RW{i}": _frame(_walk(seed)) for i, seed in enumerate(range(2, 12))}
    daily["LEADER"], daily["RECLAIM"] = _leader(20), _frame(close)
    as_of = B.last_complete_day(spy.index, "2026-10-12")
    P = B.screen(daily, spy, _universe(daily), as_of)
    G = B.groups(P)
    assert set(G["full_confluence"]["symbol"]) == {"LEADER"}
    assert "LEADER" not in set(G["breakout_leaders"]["symbol"])
    assert set(G["beaten_down_reclaim"]["symbol"]) == {"RECLAIM"}
    assert G["regime"]["bench_above_30w"] is True


def test_ranks_are_within_the_index_an_outsider_cannot_move_them(world):
    """An outsider with the strongest returns, downloaded but not a member, must not appear or shift any rank."""
    spy, daily, uni, as_of, P = world
    outsider = _frame(50.0 * np.cumprod(np.full(700, 1.02)))
    with_outsider = dict(daily, OUTSIDER=outsider)
    P2 = B.screen(with_outsider, spy, uni, as_of)
    assert "OUTSIDER" not in set(P2["symbol"])
    for col in ("xs_rank_ret13", "xs_rank_ret26"):
        pd.testing.assert_series_equal(P2[col], P[col])
    pd.testing.assert_frame_equal(P2, P)


def test_a_partial_bar_for_today_changes_nothing(world):
    """A bar dated today (close x2) must not reach the weekly state or last_close."""
    spy, daily, uni, as_of, P = world

    def with_partial(fr):
        extra = fr.iloc[[-1]].copy()
        extra.index = pd.DatetimeIndex([pd.Timestamp("2026-10-12")])
        extra[["open", "high", "low", "close"]] *= 2
        return pd.concat([fr, extra])

    daily2 = {s: with_partial(f) for s, f in daily.items()}
    spy2 = with_partial(spy)
    as_of2 = B.last_complete_day(spy2.index, "2026-10-12")
    assert as_of2 == as_of == pd.Timestamp("2026-10-09")
    P2 = B.screen(daily2, spy2, uni, as_of2)
    pd.testing.assert_frame_equal(P2, P)
    assert (P2["last_close"] == P["last_close"]).all() and (P2["w_close"] == P["w_close"]).all()


def test_a_friday_bar_that_is_still_forming_does_not_complete_the_week(world):
    """Run on Friday itself: the Friday bar is partial, so the weekly state must be last week's."""
    spy, daily, uni, _, _ = world
    as_of = B.last_complete_day(spy.index, "2026-10-09")
    assert as_of == pd.Timestamp("2026-10-08")
    full = B.screen(daily, spy, uni, as_of)                                            # frames still hold Friday's bar
    cut = B.screen({s: f[f.index <= as_of] for s, f in daily.items()}, spy[spy.index <= as_of], uni, as_of)
    assert len(full) == 12 and (full["week"] == pd.Timestamp("2026-10-02")).all()
    pd.testing.assert_frame_equal(full, cut)


def test_skipped_members_are_reported_with_a_reason(world):
    """Short history, a stale (halted) stock and a missing download must be listed, not silently ignored."""
    spy, daily, uni, as_of, _ = world
    d = dict(daily)
    d["SHORT"] = _frame(_walk(40, n=100), index=IDX[-100:])
    d["STALE"] = _frame(_walk(41, n=500), index=pd.bdate_range(end="2026-08-28", periods=500))
    u = _universe([*daily, "SHORT", "STALE", "GONE"])
    P = B.screen(d, spy, u, as_of)
    assert P.attrs["skipped"] == {"GONE": "no_data", "SHORT": "short", "STALE": "stale_week"}
    assert len(P) == 12


def test_a_future_add_date_keeps_a_symbol_out_of_the_screen(world):
    """members_asof is applied: a name added after the as-of day is not screened."""
    spy, daily, uni, as_of, _ = world
    u = uni.copy()
    u.loc[u["symbol"] == "LEADER", "date_added"] = as_of + pd.Timedelta(days=1)
    P = B.screen(daily, spy, u, as_of)
    assert "LEADER" not in set(P["symbol"]) and len(P) == 11


def test_a_screen_with_nothing_to_show_still_returns_an_empty_panel(world):
    """All symbols missing -> an empty panel with the full column set, not an exception."""
    spy, _, uni, as_of, P = world
    P0 = B.screen({}, spy, uni, as_of)
    assert len(P0) == 0 and set(P.columns) <= set(P0.columns)
    assert set(P0.attrs["skipped"].values()) == {"no_data"}


# ── fetchers ──────────────────────────────────────────────────────────────────────────────────────

def test_fetch_prices_batches_cleans_and_drops_empty_frames():
    """Batch size, tz-aware indexes, capitalised columns, bad bars and all-NaN frames."""
    calls = []
    idx = pd.date_range("2026-09-01 04:00", periods=4, freq="D", tz="UTC")

    def raw(close, bad=False):
        df = pd.DataFrame({"Open": close, "High": np.array(close) * 1.01, "Low": np.array(close) * 0.99,
                           "Close": close, "Adj Close": close, "Volume": 1000.0}, index=idx)
        if bad:
            df.iloc[1, df.columns.get_loc("High")] = 0.5                              # high < low
            df.iloc[2, :] = np.nan
        return df

    def fake(symbols, period):
        calls.append((list(symbols), period))
        frames = {"A": raw([10.0, 11, 12, 13]), "B": raw([5.0, 6, 7, 8], bad=True),
                  "C": pd.DataFrame(np.nan, index=idx, columns=["Open", "High", "Low", "Close", "Volume"])}
        return {s: frames[s] for s in symbols if s in frames}

    out = B.fetch_prices(["A", "B", "C"], period="2y", batch=2, fetch=fake)
    assert calls == [(["A", "B"], "2y"), (["C"], "2y")]
    assert set(out) == {"A", "B"}                                                      # C had no usable bar
    a = out["A"]
    assert list(a.columns) == ["open", "high", "low", "close", "volume"]
    assert a.index.tz is None and (a.index == a.index.normalize()).all() and len(a) == 4
    assert len(out["B"]) == 2                                                          # the bad bar and the NaN row are gone


def test_default_prices_fetch_reads_a_yfinance_multiindex_without_the_network(monkeypatch):
    """The real yfinance call shape (ticker-first MultiIndex, Adj Close present, one failed ticker)."""
    seen = {}
    idx = pd.bdate_range("2026-09-01", periods=5)

    def download(symbols, **kw):
        seen["symbols"], seen["kw"] = list(symbols), kw
        cols = pd.MultiIndex.from_product([["AAA", "BBB"], ["Open", "High", "Low", "Close", "Adj Close", "Volume"]])
        df = pd.DataFrame(np.nan, index=idx, columns=cols)
        for f, v in (("Open", 10.0), ("High", 11.0), ("Low", 9.0), ("Close", 10.5), ("Adj Close", 10.4), ("Volume", 1e5)):
            df[("AAA", f)] = v
        return df                                                                      # BBB failed: all NaN; CCC absent

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(download=download))
    out = B.fetch_prices(["AAA", "BBB", "CCC"], period="4y")
    assert set(out) == {"AAA"} and len(out["AAA"]) == 5 and out["AAA"]["close"].iloc[0] == 10.5
    assert seen["kw"]["auto_adjust"] is False and seen["kw"]["group_by"] == "ticker" and seen["kw"]["period"] == "4y"


def test_fundamentals_fill_missing_fields_with_nan_and_parse_the_earnings_date():
    """A partial info dict must not fail; the calendar's first earnings date is kept."""
    def fake(sym):
        if sym == "AAA":
            return ({"marketCap": 5e11, "forwardPE": "Infinity", "revenueGrowth": 0.12}, {"Earnings Date": [date(2026, 10, 30)]})
        if sym == "CCC":
            return ({}, pd.DataFrame({"Earnings Date": [pd.Timestamp("2026-11-02", tz="UTC")]}))
        raise RuntimeError("throttled")

    f = B.fundamentals(["AAA", "BBB", "CCC"], fetch=fake).set_index("symbol")
    assert list(f.columns) == [*B.FUND_FIELDS, "next_earnings"]
    assert f.loc["AAA", "marketCap"] == 5e11 and f.loc["AAA", "revenueGrowth"] == 0.12
    assert np.isnan(f.loc["AAA", "forwardPE"]) and np.isnan(f.loc["AAA", "totalDebt"])        # "Infinity" is not a number
    assert f.loc["AAA", "next_earnings"] == pd.Timestamp("2026-10-30")
    assert f.loc["BBB"].drop("next_earnings").isna().all() and pd.isna(f.loc["BBB", "next_earnings"])   # raised: a NaN row
    assert f.loc["CCC", "next_earnings"] == pd.Timestamp("2026-11-02")


def test_default_fundamentals_fetch_survives_a_broken_ticker(monkeypatch):
    """The default yfinance path must turn any exception into an empty row."""
    class Ticker:
        def __init__(self, s):
            self.s = s

        @property
        def info(self):
            if self.s == "BAD":
                raise RuntimeError("429")
            return {"marketCap": 1e9}

        calendar = {"Earnings Date": []}

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(Ticker=Ticker))
    f = B.fundamentals(["OK", "BAD"]).set_index("symbol")
    assert f.loc["OK", "marketCap"] == 1e9 and pd.isna(f.loc["OK", "next_earnings"])
    assert np.isnan(f.loc["BAD", "marketCap"])


# ── output ────────────────────────────────────────────────────────────────────────────────────────

def test_md_table_escapes_pipes_blanks_nan_and_formats_numbers():
    """A raw '|' in a company name breaks the table; NaN must be blank; prices get 2 decimals."""
    df = pd.DataFrame({"symbol": ["A|B", "C"], "w_close": [12.3456, np.nan], "xs_rank_ret26": [0.5, 1.0], "n": [3, 4]})
    assert B._md_table(df).splitlines() == [
        "| symbol | w_close | xs_rank_ret26 | n |", "|---|---|---|---|",
        "| A\\|B | 12.35 | 0.500 | 3 |", "| C |  | 1.000 | 4 |"]
    assert B._md_table(df.iloc[0:0]) == "(none)"


def _write_world(tmp_path, daily, spy):
    prices = tmp_path / "prices"
    prices.mkdir()
    for s, f in daily.items():
        f.to_pickle(prices / f"{s}.pkl")
    spy.to_pickle(prices / "SPY.pkl")
    return prices


def test_main_offline_writes_three_files_with_strict_json_and_the_caveat(tmp_path, world, capsys):
    """End to end with no network: three outputs, strict JSON (no NaN/Infinity), the 'not investment advice' caveat."""
    spy, daily, uni, _, _ = world
    prices = _write_world(tmp_path, daily, spy)
    uni_csv = tmp_path / "sp500_2026-10-12.csv"
    uni.to_csv(uni_csv, index=False)
    out = tmp_path / "out"
    rc = B.main(["--no-network", "--prices-from", str(prices), "--universe", str(uni_csv), "--out-dir", str(out),
                 "--no-fundamentals", "--today", "2026-10-12"])
    assert rc == 0
    files = sorted(p.name for p in out.iterdir())
    assert files == ["bluechip_2026-10-09.csv", "bluechip_2026-10-09.json", "bluechip_2026-10-09.md"]
    text = (out / "bluechip_2026-10-09.json").read_text()
    assert "NaN" not in text and "Infinity" not in text
    obj = json.loads(text)
    assert obj["as_of"] == "2026-10-09" and obj["asof_week"] == "2026-10-09"
    assert list(obj["groups"]) == list(B.STOCK_GROUPS)
    assert obj["caveats"] == B.CAVEATS and obj["skipped"] == {}
    assert "EXTENDED" in [r["symbol"] for r in obj["groups"]["extended"]]
    assert set(obj["regime"]) == {"benchmark", "bench_px_sma30", "bench_above_30w", "share_above_30w", "n"}
    md = (out / "bluechip_2026-10-09.md").read_text()
    assert "not investment advice" in md and B.CAVEATS in md and "Regime:" in md
    assert "## Extended" in md and "## Sector leadership" in md
    csv = pd.read_csv(out / "bluechip_2026-10-09.csv")
    assert len(csv) == 12 and "trail30" in csv.columns
    printed = capsys.readouterr().out
    assert "as_of 2026-10-09" in printed and "Regime:" in printed and "extended=" in printed


def test_main_limit_takes_the_first_symbols_of_the_sorted_universe(tmp_path, world):
    """--limit N must cut the sorted universe, not the dict order of the price files."""
    spy, daily, uni, _, _ = world
    prices = _write_world(tmp_path, daily, spy)
    uni_csv = tmp_path / "u.csv"
    uni.to_csv(uni_csv, index=False)
    rc = B.main(["--no-network", "--prices-from", str(prices), "--universe", str(uni_csv), "--limit", "3",
                 "--out-dir", str(tmp_path / "o"), "--no-fundamentals", "--today", "2026-10-12"])
    assert rc == 0
    obj = json.loads((tmp_path / "o" / "bluechip_2026-10-09.csv").with_suffix(".json").read_text())
    csv = pd.read_csv(tmp_path / "o" / "bluechip_2026-10-09.csv")
    assert list(csv["symbol"]) == sorted(uni["symbol"])[:3] and obj["regime"]["n"] == 3


def test_main_offline_without_prices_dir_or_spy_fails_cleanly(tmp_path, world, capsys):
    """--no-network without --prices-from is a usage error; a prices dir without SPY returns 2, not a traceback."""
    spy, daily, uni, _, _ = world
    uni_csv = tmp_path / "u.csv"
    uni.to_csv(uni_csv, index=False)
    with pytest.raises(SystemExit):
        B.main(["--no-network", "--universe", str(uni_csv), "--out-dir", str(tmp_path / "o")])
    empty = tmp_path / "empty"
    empty.mkdir()
    rc = B.main(["--no-network", "--prices-from", str(empty), "--universe", str(uni_csv), "--out-dir",
                 str(tmp_path / "o"), "--no-fundamentals", "--today", "2026-10-12"])
    assert rc == 2 and "SPY" in capsys.readouterr().err
    assert not (tmp_path / "o").exists()


def test_write_outputs_survives_nan_in_the_panel_and_merges_fundamentals(tmp_path, world):
    """NaN anywhere in the panel or the fundamentals must become JSON null; fundamentals reach the CSV."""
    *_, P = world
    P = P.copy()
    P.attrs.update(world[-1].attrs)
    P.loc[0, "w_ret26"] = np.nan
    P.loc[0, "w_close"] = np.nan
    G = B.groups(P)
    fund = B.fundamentals(["EXTENDED"], fetch=lambda s: ({"marketCap": 1e9, "forwardPE": np.nan}, None))
    paths = B.write_outputs(P, G, "2026-10-09", tmp_path / "o", fund)
    text = paths["json"].read_text()
    assert "NaN" not in text and "Infinity" not in text
    json.loads(text)
    csv = pd.read_csv(paths["csv"]).set_index("symbol")
    assert csv.loc["EXTENDED", "marketCap"] == 1e9 and np.isnan(csv.loc["LEADER", "marketCap"])
    assert set(paths) == {"csv", "md", "json"}
