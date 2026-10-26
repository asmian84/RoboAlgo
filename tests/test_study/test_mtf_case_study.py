"""scripts/mtf_case_study.py describes the big winners at their lows. It must describe them against what was
knowable at the time and compare them with the right universe, and it must never become a source of evidence.

The failures these tests guard against, each mutation-tested:
  * a confirmation milestone searched from the START OF DATA instead of from the low week (a stock that was
    already above its 30-week SMA years before the low would report an "early confirmation" that is not one);
  * a percentile computed over ALL rows of the week instead of the LIQUID universe (an illiquid outlier would
    shift every name's rank), or a percentile that ignores the float32 rounding of the stored universe;
  * a benchmark (or any series) NOT clipped to the price cache's last date: yfinance runs to today, so a ride
    or a weekly bar would use days the cache does not have;
  * a small universe (fewer than MIN_UNIVERSE liquid stocks) silently producing a percentile;
  * the cohort tables missing from the top of the HTML index, or the ledger touched by a description-only script.
No test touches the network, the price cache or the real ledger (learn.REGISTRY_DIR is redirected to tmp_path;
the real trials.jsonl is compared byte for byte before and after the end-to-end run).
"""
import importlib.util
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from setup_engine import learn as L
from setup_engine import mtf

ROOT = Path(__file__).resolve().parents[2]
REAL_LEDGER = ROOT / "setup_engine" / "registry" / "trials.jsonl"


def _load():
    spec = importlib.util.spec_from_file_location("mtf_case_study", ROOT / "scripts" / "mtf_case_study.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mtf_case_study"] = mod
    spec.loader.exec_module(mod)
    return mod


CS = _load()


@pytest.fixture(autouse=True)
def _ledger_in_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "REGISTRY_DIR", tmp_path / "registry")


# ── synthetic stocks ──────────────────────────────────────────────────────────────────────────────

def _from_weekly(closes, volumes=None, start="2020-01-06"):
    """Five identical business days per weekly close (o = h = l = c), Monday to Friday."""
    closes = np.asarray(closes, dtype=float)
    days = pd.bdate_range(start, periods=5 * len(closes))
    c = np.repeat(closes, 5)
    v = np.full(len(c), 1e6) if volumes is None else np.repeat(np.asarray(volumes, dtype=float), 5)
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": v}, index=days)


def _winner_closes():
    """70 weeks rising 80 -> 120, 30 weeks falling to 60, 8 flat weeks, then 60 weeks rising 3x to 180."""
    lead = 80.0 * (120.0 / 80.0) ** (np.arange(70) / 69.0)          # idx 0..69
    fall = 120.0 * 0.5 ** (np.arange(1, 31) / 30.0)                 # idx 70..99, ends at 60
    flat = np.full(8, 60.0)                                         # idx 100..107
    rise = 60.0 * 3.0 ** (np.arange(1, 61) / 60.0)                  # idx 108..167, ends at 180
    return np.concatenate([lead, fall, flat, rise])


LOW_IDX = 99


@pytest.fixture
def winner(monkeypatch):
    cl = _winner_closes()
    daily = _from_weekly(cl)
    labels = mtf.weekly_bars(daily).index
    monkeypatch.setattr(CS, "RUN_START", str(labels[69].date()))
    bench = _from_weekly(np.full(len(cl), 100.0))
    return cl, daily, labels, bench


# ── 1. the cohort ─────────────────────────────────────────────────────────────────────────────────

def test_cohort_has_46_unique_names_and_every_user_name():
    assert len(CS.COHORT) == 46 and len(set(CS.COHORT)) == 46
    assert set(CS.USER_NAMES) <= set(CS.COHORT)
    assert set(CS.NOT_IN_CACHE) <= set(CS.COHORT)
    assert CS.RS_BENCH == "SPY" and CS.RUN_START == "2021-10-08"


# ── 2. milestones on a synthetic winner ───────────────────────────────────────────────────────────

def test_milestones_are_searched_from_the_low_week_and_pct_off_low_is_exact(winner):
    cl, daily, labels, bench = winner
    c = CS.case("SYN", daily, bench, pd.DataFrame(), daily.index[-1])
    run = c["run"]
    assert run["low_week"] == labels[LOW_IDX] and run["low_close"] == pytest.approx(60.0)
    assert run["peak_week"] == labels[-1] and run["mult"] == pytest.approx(3.0) and run["weeks"] == 68
    low = c["milestones"]["low"]
    assert low["week"] == labels[LOW_IDX] and low["weeks_after_low"] == 0 and low["pct_off_low"] == 0.0

    # the lead-in was already above its SMA, rising and breaking out years before the low: none of that counts
    k_above = next(k for k in range(LOW_IDX, len(cl)) if cl[k] > cl[k - 29:k + 1].mean())
    assert k_above > LOW_IDX
    ab = c["milestones"]["w_above_sma30"]
    assert ab["week"] == labels[k_above] and ab["weeks_after_low"] == k_above - LOW_IDX
    assert ab["pct_off_low"] == pytest.approx(cl[k_above] / 60.0 - 1.0)

    k_brk = next(k for k in range(LOW_IDX, len(cl)) if cl[k] > cl[k - 26:k].max())
    assert c["milestones"]["brk26"]["week"] == labels[k_brk]
    assert c["milestones"]["brk26"]["pct_off_low"] == pytest.approx(cl[k_brk] / 60.0 - 1.0)

    for key in CS.MILESTONES:
        rec = c["milestones"][key]
        if rec is not None:
            assert rec["week"] >= run["low_week"], key
    assert c["milestones"]["vol_expansion"] is None          # constant volume never expands

    conf = c["milestones"]["first_confluence"]
    m10 = c["milestones"]["m_above_sma10"]
    assert conf["week"] >= m10["week"] and conf["week"] >= ab["week"]
    assert conf["week"] >= c["milestones"]["sma30_rising"]["week"]
    wf = c["_wf"]
    k = wf.index.get_loc(conf["week"])
    ok = ((wf["w_close"] > wf["w_sma30"]) & (wf["w_slope30"] > 0) & (wf["m_close"] > wf["m_sma10"])
          & (wf["m_label"] >= run["low_week"]))
    assert ok.iloc[k] and not ok.iloc[LOW_IDX:k].any()


def test_rides_start_at_the_first_confluence_week_and_data_flags_are_reported(winner):
    cl, daily, labels, bench = winner
    c = CS.case("SYN", daily, bench, pd.DataFrame(), daily.index[-1])
    conf_week = c["milestones"]["first_confluence"]["week"]
    for name in CS.RIDES:
        r = c["rides"][name]
        assert r is not None and r["signal_week"] == conf_week
        assert r["entry_date"] > r["signal_date"] and np.isfinite(r["ret_net"])
    assert c["hold"]["low_to_peak"] == pytest.approx(3.0) and c["hold"]["confluence_after_peak"] is False
    assert c["last_day"] == daily.index[-1]
    assert not any("data gap" in w for w in c["warnings"])                  # no gap in the run ...
    assert any("short history" in w for w in c["warnings"])                 # ... but only 99 weeks of history before the low
    assert c["liquid_at_low"] is True


def test_case_refuses_data_that_runs_past_the_data_end(winner):
    cl, daily, labels, bench = winner
    with pytest.raises(ValueError, match="clip every series to data_end"):
        CS.case("SYN", daily, bench, pd.DataFrame(), daily.index[-10])


def test_a_gappy_short_history_name_is_flagged(monkeypatch):
    cl = _winner_closes()[40:]                       # starts 40 weeks before the low's neighbourhood: < 2 years of history
    daily = _from_weekly(cl)
    daily = daily[(daily.index < daily.index[400]) | (daily.index > daily.index[400] + pd.Timedelta(days=60))]
    labels = mtf.weekly_bars(daily).index
    monkeypatch.setattr(CS, "RUN_START", str(labels[10].date()))
    c = CS.case("GAP", daily, _from_weekly(np.full(len(cl), 100.0)), pd.DataFrame(), daily.index[-1])
    text = " ".join(c["warnings"])
    assert "short history" in text and "data gap" in text


# ── 3. percentiles ────────────────────────────────────────────────────────────────────────────────

def test_percentile_of_by_hand_and_the_small_universe_rule(monkeypatch):
    week = pd.DataFrame({"w_ret26": [1.0, 2.0, 3.0, 4.0], "liquid": [True] * 4})
    assert CS.MIN_UNIVERSE == 200 and CS.percentile_of(week, "w_ret26", 3.0) is None     # too few stocks
    monkeypatch.setattr(CS, "MIN_UNIVERSE", 4)
    assert CS.percentile_of(week, "w_ret26", 3.0) == 0.75
    assert CS.percentile_of(week, "w_ret26", 0.5) == 0.0 and CS.percentile_of(week, "w_ret26", 9.0) == 1.0
    assert CS.percentile_of(week, "w_ret26", float("nan")) is None
    assert CS.percentile_of(week, "nope", 3.0) is None
    monkeypatch.setattr(CS, "MIN_UNIVERSE", 5)
    assert CS.percentile_of(week, "w_ret26", 3.0) is None


def test_percentile_uses_liquid_rows_only_and_ignores_non_finite_values(monkeypatch):
    monkeypatch.setattr(CS, "MIN_UNIVERSE", 4)
    week = pd.DataFrame({"w_ret26": [1.0, 2.0, 3.0, 4.0, -50.0, np.nan],
                         "liquid": [True, True, True, True, False, True]})
    assert CS.percentile_of(week, "w_ret26", 3.0) == 0.75        # the illiquid -50 must not count (all rows: 0.8)
    view = CS.PanelView(week.assign(week=pd.Timestamp("2022-10-14")))
    assert view.percentile("2022-10-14", "w_ret26", 3.0) == 0.75
    assert view.percentile("2022-10-21", "w_ret26", 3.0) is None  # a week that is not in the panel


def test_percentile_respects_the_float32_rounding_of_the_stored_universe(monkeypatch):
    monkeypatch.setattr(CS, "MIN_UNIVERSE", 3)
    week = pd.DataFrame({"x": np.array([0.1, 0.2, 0.3], dtype="float32"), "liquid": True})
    assert CS.percentile_of(week, "x", 0.3) == 1.0               # float64 0.3 > float32(0.3): must still count itself


def test_xs_ranks_are_percentiles_against_the_panel(monkeypatch):
    monkeypatch.setattr(CS, "MIN_UNIVERSE", 10)
    w1, w2 = pd.Timestamp("2022-10-14"), pd.Timestamp("2022-10-21")
    panel = pd.DataFrame({"week": [w1] * 10, "liquid": True, "w_ret13": np.arange(10) / 10.0,
                          "w_ret26": np.arange(10) / 10.0})
    wf = pd.DataFrame({"w_ret13": [0.85, 0.85], "w_ret26": [-1.0, np.nan]}, index=pd.DatetimeIndex([w1, w2]))
    out = CS._fill_xs(wf, CS.PanelView(panel))
    assert out["xs_rank_ret13"].iloc[0] == pytest.approx(0.9)    # values <= 0.85: 0.0 .. 0.8
    assert np.isnan(out["xs_rank_ret13"].iloc[1])                # week missing from the panel
    assert out["xs_rank_ret26"].iloc[0] == 0.0 and np.isnan(out["xs_rank_ret26"].iloc[1])


# ── 4. the panel ──────────────────────────────────────────────────────────────────────────────────

def test_panel_rows_keep_only_liquid_weeks_since_the_start():
    cl = 50.0 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.03, 200)))
    daily = _from_weekly(cl, volumes=np.full(200, 2e6))
    out = CS.panel_rows("LIQ", daily, _from_weekly(np.full(200, 100.0)), daily.index[-1], "2020-06-01")
    assert out is not None and (out["symbol"] == "LIQ").all() and out["liquid"].all()
    assert out["week"].min() >= pd.Timestamp("2020-06-01")
    assert (out["w_close"] >= 2.0).all() and out["w_close"].dtype == np.float32
    wf = mtf.mtf_features(daily, _from_weekly(np.full(200, 100.0)))
    assert len(out) == int((mtf.is_liquid(wf) & (wf.index >= pd.Timestamp("2020-06-01"))).sum())
    assert out["week"].min() >= wf.index[59]                     # w_n >= 60 weeks of history

    penny = _from_weekly(np.full(200, 1.0), volumes=np.full(200, 1e9))
    assert CS.panel_rows("PEN", penny, _from_weekly(np.full(200, 100.0)), penny.index[-1], "2020-06-01") is None


# ── 5. summaries ──────────────────────────────────────────────────────────────────────────────────

def _fake_case(symbol, low_week, mult, ret_r3):
    ms = {k: None for k in CS.MILESTONES}
    state = {"values": {c: 1.0 for c in CS.PCT_COLUMNS}, "pct": {c: 0.5 for c in CS.PCT_COLUMNS}}
    ms["low"] = {"week": pd.Timestamp(low_week), "weeks_after_low": 0, "w_close": 1.0, "pct_off_low": 0.0, **state}
    ride = {"signal_week": pd.Timestamp(low_week), "ret_net": ret_r3, "ret_gross": ret_r3 + 0.01,
            "reason": "r3_trail30", "weeks_held": 50.0}
    return {"symbol": symbol, "group": CS.group_of(symbol), "warnings": [], "milestones": ms,
            "signatures": {s: None for s in mtf.SIGNATURES}, "state_at_low": state,
            "rides": {"R1": None, "R2": None, "R3": ride},
            "run": {"low_week": pd.Timestamp(low_week), "peak_week": pd.Timestamp("2025-01-03"), "mult": mult,
                    "weeks": 100, "mdd_in_run": -0.3, "dd_into_low": -0.5, "max_gap_days": 7}}


def test_summaries_sort_runs_median_the_rides_and_count_same_week_lows():
    cases = {"AMD": _fake_case("AMD", "2022-10-14", 2.0, 0.5), "NVDA": _fake_case("NVDA", "2022-10-14", 4.0, 1.5),
             "MSFT": _fake_case("MSFT", "2022-11-04", 1.5, -0.1), "SPY": _fake_case("SPY", "2022-10-14", 1.3, 0.1)}
    s = CS.summaries(cases)
    assert [r["symbol"] for r in s["runs"]] == ["NVDA", "AMD", "MSFT", "SPY"]
    assert s["lows_calendar"][0]["week"] == pd.Timestamp("2022-10-14") and s["lows_calendar"][0]["n"] == 2
    assert s["lows_calendar"][0]["symbols"] == ["AMD", "NVDA"]                  # the benchmark ETF is not a "name"
    r3 = s["rides"]["user"]["R3"]
    assert r3["n"] == 3 and r3["median_ret_net"] == pytest.approx(0.5)
    assert r3["share_ge_100pct"] == pytest.approx(1 / 3)
    assert r3["median_capture"] == pytest.approx(np.median([1.5 / 2.0, 2.5 / 4.0, 0.9 / 1.5]))
    assert s["rides"]["user"]["R1"]["n"] == 0 and s["rides"]["user"]["R1"]["median_ret_net"] is None
    assert s["milestones"]["user"]["low"]["n_reached"] == 3 and s["milestones"]["user"]["brk26"]["n_reached"] == 0
    assert s["state_at_low"]["user"]["w_dd52"]["median_pct"] == 0.5
    tables = CS.build_tables(s, cases)
    assert {t["name"] for t in tables} >= {"runs", "milestones", "state_at_low", "rides", "signatures", "lows_calendar"}
    assert all(CS.text_table(t) for t in tables)


def test_chart_header_lines_are_wrapped_inside_the_canvas():
    line = "After the low: " + " | ".join(f"milestone number {i} +{i}w/+{i}0%" for i in range(12))
    out = CS._wrap([line, "short"])
    assert out[-1] == "short" and len(out) > 2
    assert all(len(x) <= CS.HEADER_WIDTH_CHARS for x in out)
    assert sum(x.count("milestone number") for x in out) == 12               # nothing is lost by wrapping


# ── 6. main, end to end ───────────────────────────────────────────────────────────────────────────

def _write_world(tmp_path, n_stocks=30, extra_bench_days=20):
    pytest.importorskip("pyarrow")
    hist = tmp_path / "history"
    hist.mkdir()
    days = pd.bdate_range("2019-01-02", "2024-12-31")
    rng = np.random.default_rng(7)
    names = ["NVDA", "AMD"] + [f"S{i:02d}" for i in range(n_stocks - 2)]
    frames = []
    for s in names:
        c = 50.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.018, len(days))))
        frames.append(pd.DataFrame({"date": days, "symbol": s, "open": c * 0.999, "high": c * 1.01, "low": c * 0.99,
                                    "close": c, "volume": np.full(len(days), 2e6)}))
    half = n_stocks // 2
    pd.concat(frames[:half]).to_parquet(hist / "batch_0000.parquet", index=False)
    pd.concat(frames[half:]).to_parquet(hist / "batch_0001.parquet", index=False)
    bdir = tmp_path / "bench"
    bdir.mkdir()
    bdays = pd.bdate_range("2019-01-02", periods=len(days) + extra_bench_days)       # runs PAST the cache's last date
    for s in mtf.BENCHMARKS:
        c = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(bdays))))
        pd.DataFrame({"open": c, "high": c * 1.005, "low": c * 0.995, "close": c, "volume": 1e8},
                     index=pd.DatetimeIndex(bdays, name="date")).to_parquet(bdir / f"{s}.parquet")
    return hist, bdir, days[-1]


def _strict(text):
    def boom(x):
        raise ValueError(f"non-finite constant {x} in JSON")
    return json.loads(text, parse_constant=boom)


def test_main_end_to_end_writes_strict_json_charts_and_index_and_leaves_the_ledger_alone(tmp_path, monkeypatch, capsys):
    hist, bdir, last = _write_world(tmp_path)
    monkeypatch.setattr(CS, "MIN_UNIVERSE", 5)
    before = REAL_LEDGER.read_bytes()
    out = tmp_path / "out"
    rc = CS.main(["--history", str(hist), "--bench-dir", str(bdir), "--out-dir", str(out),
                  "--symbols", "NVDA", "AMD", "SPY", "--workers", "1"])
    assert rc == 0
    js = _strict((out / "results.json").read_text())
    assert js["data_end"] == str(last.date()) and set(js["cases"]) == {"NVDA", "AMD", "SPY"}
    assert js["missing_symbols"] == [] and js["panel"]["n_weeks"] > 0 and js["limited_run"] is False

    # nothing may run past the cache's last date, although the benchmark files do (a bench not clipped would)
    for sym, c in js["cases"].items():
        assert c["last_day"] <= js["data_end"] and c["last_week"] <= js["data_end"], sym
        for r in (c["rides"] or {}).values():
            if r:
                assert r["exit_date"] <= js["data_end"], sym
    assert js["cases"]["SPY"]["last_day"] == str(last.date())
    assert not any(k.startswith("_") for k in js["cases"]["NVDA"])

    nv = js["cases"]["NVDA"]
    assert nv["run"]["low_week"] and nv["milestones"]["low"]["pct"]["w_ret26"] is not None     # percentiles exist
    assert set(nv["signatures"]) == set(mtf.SIGNATURES) and set(nv["rides"]) == {"R1", "R2", "R3"}
    assert set(js["summaries"]) >= {"runs", "milestones", "state_at_low", "rides", "signatures", "lows_calendar"}

    for sym in ("NVDA", "AMD", "SPY"):
        root = ET.fromstring((out / f"{sym}.svg").read_text(encoding="utf-8"))
        assert root.tag.endswith("svg")
    html = (out / "index.html").read_text(encoding="utf-8")
    assert "NVDA" in html and "Cohort summary" in html and "<script" not in html
    assert html.index("Cohort summary") < html.index('<section id="e0">')                       # tables ABOVE the charts
    assert html.index("(a) Best run") < html.index("(d) Ride outcomes") < html.index('<section id="e0">')
    text = capsys.readouterr().out
    assert "(a) Best run" in text and "(d) Ride outcomes" in text and "wrote 3 charts" in text

    assert REAL_LEDGER.read_bytes() == before, "the real ledger must not change"
    assert not (tmp_path / "registry" / "trials.jsonl").exists() and L.trial_count() == 0


def test_a_limited_run_defaults_to_the_smoke_directory_not_docs(tmp_path, monkeypatch):
    hist, bdir, _ = _write_world(tmp_path)
    monkeypatch.setattr(CS, "SMOKE_DIR", tmp_path / "smoke")
    monkeypatch.setattr(CS, "OUT_DIR", tmp_path / "docs_out")
    monkeypatch.setattr(CS, "MIN_UNIVERSE", 5)
    assert CS.main(["--history", str(hist), "--bench-dir", str(bdir), "--limit-batches", "1",
                    "--symbols", "NVDA", "SPY", "--workers", "1"]) == 0
    assert (tmp_path / "smoke" / "results.json").exists() and not (tmp_path / "docs_out").exists()
    js = _strict((tmp_path / "smoke" / "results.json").read_text())
    assert js["limited_run"] is True and js["batch_files_for_panel"] == 1
    assert "NVDA" in js["cases"]                         # the cohort is read from ALL batches, the panel from 1
