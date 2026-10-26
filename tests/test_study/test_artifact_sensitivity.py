"""scripts/artifact_sensitivity.py: BEFORE must be the published sample, AFTER must drop exactly the contaminated rows.

The failures being guarded against: (1) a BEFORE that does not reproduce the published sample (censored
"end" trades kept, other configurations mixed in, a different date column for the month clusters), which
would make the whole before/after comparison meaningless; (2) a contamination window that is off by a day
or by the wrong length, which silently drops the wrong trades; (3) the date-matched control being
recomputed AFTER the drop, which would quietly turn "remove artifacts from the treated sample" into
"remove them from the market as well"; (4) signature masks that stop being the published ones.
Everything is hand-computed from small tables; only the main() tests touch parquet, and only under tmp_path.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from setup_engine import mtf
from setup_engine import trade_stats as TS
from setup_engine.screen import Screen

ROOT = Path(__file__).resolve().parents[2]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


S = _load("artifact_sensitivity")
T = pd.Timestamp


# ── turnaround, by hand ─────────────────────────────────────────────────────────────

def _trade(symbol, signal, exit_, r, reason="target", config="primary", arm="C0"):
    return {"symbol": symbol, "signal_date": T(signal), "exit_date": T(exit_), "r_net": r,
            "reason": reason, "config": config, "arm": arm}


def _trades():
    return pd.DataFrame([
        _trade("A", "2020-03-02", "2020-04-01", 1.0),
        _trade("B", "2020-03-02", "2020-04-01", -1.0),
        _trade("C", "2020-06-01", "2020-07-01", 2.0),
        _trade("D", "2020-06-01", "2020-07-01", 9.0, reason="end"),             # censored: never in a published number
        _trade("E", "2020-06-01", "2020-07-01", 5.0, config="cost_2pct"),       # another configuration
        _trade("F", "2020-06-01", "2020-07-01", 7.0, arm="C1"),                 # another arm
    ])


def _days(**kw):
    return {k: pd.DatetimeIndex(v) for k, v in kw.items()}


def test_before_is_the_published_sample_and_after_drops_the_contaminated_trade():
    """If this fails, BEFORE no longer reproduces the published C0 sample or AFTER drops the wrong trade.
    B's signal is 2020-03-02; an artifact on 2020-01-15 is 47 days earlier (inside 60 days, outside 6)."""
    out = S.turnaround_sensitivity(_trades(), _days(B=["2020-01-15"]), n_boot=200)
    assert out["before"]["n"] == 3 and out["before"]["mean_r"] == pytest.approx(2 / 3)
    assert out["after"]["n"] == 2 and out["after"]["mean_r"] == pytest.approx(1.5)
    assert out["n_dropped"] == 1 and out["dropped_mean_r"] == pytest.approx(-1.0)
    assert [(d["symbol"], d["signal_date"]) for d in out["dropped"]] == [("B", T("2020-03-02"))]
    assert out["before"]["month"]["n_clusters"] == 2 and out["before"]["stock"]["n_clusters"] == 3
    assert out["after"]["month"]["n_clusters"] == 2 and out["after"]["stock"]["n_clusters"] == 2


def test_a_trade_whose_exit_day_is_the_artifact_day_is_dropped():
    """If this fails, the window end is exclusive and a trade that closed ON the artifact day is kept."""
    out = S.turnaround_sensitivity(_trades(), _days(C=["2020-07-01"]), n_boot=200)
    assert out["n_dropped"] == 1 and out["dropped"][0]["symbol"] == "C"
    assert out["after"]["n"] == 2 and out["after"]["mean_r"] == pytest.approx(0.0)


def test_the_pre_signal_window_is_exactly_sixty_days_inclusive():
    """If this fails, PRE_DAYS moved: 2020-03-02 minus 60 days is 2020-01-02 (in); 2020-01-01 is out."""
    inside = S.turnaround_sensitivity(_trades(), _days(B=["2020-01-02"]), n_boot=200)
    outside = S.turnaround_sensitivity(_trades(), _days(B=["2020-01-01"]), n_boot=200)
    assert inside["n_dropped"] == 1
    assert outside["n_dropped"] == 0 and outside["after"]["n"] == 3


def test_an_artifact_in_another_stock_or_after_the_exit_drops_nothing():
    """If this fails, contamination leaks across symbols or past the exit date."""
    out = S.turnaround_sensitivity(_trades(), _days(A=["2020-04-02"], Z=["2020-03-15"]), n_boot=200)
    assert out["n_dropped"] == 0 and out["after"]["n"] == out["before"]["n"] == 3
    assert out["dropped"] == [] and out["dropped_mean_r"] is None


def test_the_month_clusters_use_the_signal_date_not_the_exit_date():
    """If this fails, BEFORE no longer matches the published intervals (study_turnaround clusters on signal_date).
    Signals fall in March and April (2 clusters); both trades exit in April (1 cluster if exit_date were used)."""
    t = pd.DataFrame([_trade("P", "2020-03-02", "2020-04-15", 1.0), _trade("Q", "2020-04-01", "2020-04-15", 2.0)])
    out = S.turnaround_sensitivity(t, {}, n_boot=200)
    assert out["before"]["month"]["n_clusters"] == 2


def test_before_intervals_equal_a_direct_bootstrap_of_the_published_sample():
    """If this fails, the cluster keys, alpha or seed differ from the published study's _cis call."""
    rng = np.random.default_rng(3)
    sig = pd.Timestamp("2015-01-05") + pd.to_timedelta(rng.integers(0, 900, 80), unit="D")
    t = pd.DataFrame({"symbol": [f"S{i % 25}" for i in range(80)], "signal_date": sig,
                      "exit_date": sig + pd.Timedelta(days=30), "r_net": rng.normal(0.1, 1.5, 80),
                      "reason": "trail", "config": "primary", "arm": "C0"})
    out = S.turnaround_sensitivity(t, {}, n_boot=300)
    r = t["r_net"].to_numpy()
    assert out["before"]["month"] == TS.cluster_bootstrap(r, TS.month_key(t["signal_date"]), 300, 20261007, 0.05)
    assert out["before"]["stock"] == TS.cluster_bootstrap(r, pd.factorize(t["symbol"])[0], 300, 20261007, 0.05)
    assert out["after"] == out["before"]                      # nothing to drop: AFTER is the same sample


def test_the_default_seeds_are_the_seeds_of_the_published_studies():
    """If this fails, BEFORE intervals stop reproducing docs/turnaround_study_results.json and
    docs/mtf_event_study_results.json (the means would still match, the intervals would not)."""
    assert S.SEED_TURNAROUND == _load("study_turnaround").SEED
    assert S.SEED_EVENTS == _load("mtf_event_study").SEED
    assert S.ALPHA_EVENT == pytest.approx(0.05 / 4)


# ── events, by hand ─────────────────────────────────────────────────────────────────

HELD = "2001-05-18"


def _rows():
    """Held week 2001-05-18: A 0.10 (S1), B 0.30 (S1, will be contaminated), C -0.10 (not S1); D sits in a
    week that is not held-out (ignored by BEFORE and AFTER, and not part of the control of the held week)."""
    return pd.DataFrame({"symbol": ["A", "B", "C", "D"],
                         "week": [T(HELD), T(HELD), T(HELD), T("2001-05-25")],
                         S.OUTCOME: [0.10, 0.30, -0.10, 5.0]})


S1 = {"S1": np.array([True, True, False, True])}


def test_events_before_and_after_by_hand():
    """If this fails, the held-out filter, the date-matched control or the contamination drop is wrong.
    control = (0.10 + 0.30 - 0.10) / 3 = 0.10; B (excess +0.20) is contaminated by 2002-01-04."""
    out = S.event_sensitivity(_rows(), _days(B=["2002-01-04"]), masks=S1, n_boot=200)
    s = out["signatures"]["S1"]
    assert s["before"]["n"] == 2 and s["before"]["mean_ret"] == pytest.approx(0.20)
    assert s["before"]["mean_excess"] == pytest.approx(0.10)
    assert s["after"]["n"] == 1 and s["after"]["mean_ret"] == pytest.approx(0.10)
    assert s["after"]["mean_excess"] == pytest.approx(0.0)
    assert s["n_dropped"] == 1 and s["dropped_mean_ret"] == pytest.approx(0.30)
    assert s["dropped_mean_excess"] == pytest.approx(0.20)
    assert out["control"]["n_rows"] == 3 and out["control"]["n_contaminated"] == 1
    assert out["control"]["share_contaminated"] == pytest.approx(1 / 3)


def test_an_artifact_four_years_after_the_event_is_outside_the_window():
    """If this fails, the exposure window is longer than 156 weeks."""
    out = S.event_sensitivity(_rows(), _days(B=["2005-06-01"]), masks=S1, n_boot=200)
    s = out["signatures"]["S1"]
    assert s["after"]["n"] == s["before"]["n"] == 2 and s["n_dropped"] == 0


def test_the_event_window_is_156_weeks_inclusive_at_both_ends():
    """If this fails, EVENT_WEEKS_FWD moved or an end became exclusive: 2001-05-18 + 156 weeks = 2004-05-14."""
    def dropped(day, who="B"):
        return S.event_sensitivity(_rows(), _days(**{who: [day]}), masks=S1, n_boot=200)["signatures"]["S1"]["n_dropped"]
    assert dropped("2004-05-14") == 1            # last day inside
    assert dropped("2004-05-15") == 0            # one day later
    assert dropped("2001-05-18") == 1            # the event week itself (start inclusive)
    assert dropped("2001-05-17") == 0            # the day before


def test_the_control_is_the_full_table_mean_even_for_non_finite_outcomes():
    """If this fails, NaN outcomes leak into the control mean or into the BEFORE sample."""
    rows = _rows()
    rows.loc[2, S.OUTCOME] = np.nan                           # C has no outcome now: control = (0.10 + 0.30) / 2
    out = S.event_sensitivity(rows, {}, masks={"S1": np.array([True, True, True, False])}, n_boot=200)
    s = out["signatures"]["S1"]["before"]
    assert s["n"] == 2 and s["mean_excess"] == pytest.approx(0.0)
    assert out["control"]["n_rows"] == 2


def test_event_interval_clusters_by_week_and_uses_the_given_alpha():
    """If this fails, the interval is clustered by stock/row or ignores the Bonferroni alpha."""
    rows = pd.DataFrame({"symbol": list("ABCDEF"), S.OUTCOME: [0.1, 0.3, 0.0, 0.5, 0.2, 0.2],
                         "week": [T("2001-05-18")] * 3 + [T("2001-12-07")] * 3})
    masks = {"S1": np.ones(6, dtype=bool)}
    out = S.event_sensitivity(rows, {}, masks=masks, n_boot=300)
    ci = out["signatures"]["S1"]["before"]["ci"]
    x = rows[S.OUTCOME].to_numpy() - np.array([0.4 / 3] * 3 + [0.9 / 3] * 3)       # same-week all-stock means
    ref = TS.cluster_bootstrap(x, pd.to_datetime(rows["week"]).to_numpy("datetime64[ns]").astype("int64"), 300, 20261009, 0.05 / 4)
    assert ci["n_clusters"] == 2 and ci["mean"] == pytest.approx(0.0, abs=1e-9)
    assert ci["lo"] == pytest.approx(ref["lo"], abs=1e-6) and ci["hi"] == pytest.approx(ref["hi"], abs=1e-6)
    wide = S.event_sensitivity(rows, {}, masks=masks, n_boot=300, alpha=0.5)["signatures"]["S1"]["before"]["ci"]
    assert wide["lo"] >= ci["lo"] and wide["hi"] <= ci["hi"]


def test_weeks_outside_the_held_out_list_are_never_analysed():
    """If this fails, in-sample or immature event weeks leak into the held-out numbers."""
    out = S.event_sensitivity(_rows(), {}, masks={"S1": np.ones(4, dtype=bool)}, weeks=("2001-05-25",), n_boot=200)
    s = out["signatures"]["S1"]["before"]
    assert s["n"] == 1 and s["mean_ret"] == pytest.approx(5.0)


# ── signature masks ─────────────────────────────────────────────────────────────────

def _signature_frame():
    """Three stocks in one week; all columns any signature reads."""
    rng = np.random.default_rng(5)
    n = 3
    df = pd.DataFrame({"symbol": list("XYZ"), "week": T("2019-02-15"),
                       "w_ret13": [0.50, 0.10, 0.30], "w_ret26": [0.2, 0.6, 0.4],
                       "w_close": [10.0, 10.0, 8.0], "w_sma30": [9.0, 9.0, 9.0], "w_slope30": [0.1, 0.1, 0.1],
                       "w_dd52": rng.uniform(-0.5, 0, n), "w_prev_hi26": [9.0, 20.0, 20.0],
                       "w_volx": [2.0, 1.0, 2.0], "m_close": [10.0] * n, "m_sma10": [9.0] * n,
                       "rs_line": [1.0, 1.0, 1.0], "rs_prev_hi26": [0.9] * n, "bm_w_px_sma30": [0.05] * n,
                       S.OUTCOME: [0.1, 0.2, 0.3]})
    return df


def test_signature_masks_are_the_published_masks_with_every_row_liquid():
    """If this fails, rows are no longer ranked as liquid (all ranks NaN, S1/S2 never true) or the screens changed.
    X has the top 13-week return and sits above a rising SMA30: S1 only for X; B0 (above SMA30) for X and Y."""
    rows = _signature_frame()
    got = S.signature_masks(rows)
    assert list(got) == ["S1", "S2", "S3", "S4", "B0"]
    assert list(got["S1"]) == [True, False, False]
    assert list(got["B0"]) == [True, True, False]
    panel = mtf.add_xs_ranks(rows.assign(liquid=True))
    for s in got:
        assert list(got[s]) == list(Screen(mtf.SIGNATURES[s]).mask(panel))
        assert got[s].dtype == bool and len(got[s]) == 3


# ── by-year summary ─────────────────────────────────────────────────────────────────

def test_summary_by_year_counts_by_hand():
    """If this fails, the diagnostic table counts the wrong year, double-counts a stock or skips an empty year."""
    days = _days(A=["2019-05-01", "2019-05-02", "2021-03-01"], B=["2019-07-01"])
    tb = pd.DataFrame({"signal_date": [T("2019-06-01"), T("2021-01-01")]})
    rb = pd.DataFrame({"week": [T("2020-05-29")]})
    out = S.summary_by_year(days, tb, rb)
    assert list(out.index) == [2019, 2020, 2021] and out.index.name == "year"
    assert list(out.columns) == ["artifact_days", "stocks_flagged", "c0_trades_dropped", "event_rows_dropped"]
    assert out["artifact_days"].tolist() == [3, 0, 1]
    assert out["stocks_flagged"].tolist() == [2, 0, 1]
    assert out["c0_trades_dropped"].tolist() == [1, 0, 1]
    assert out["event_rows_dropped"].tolist() == [0, 1, 0]


def test_summary_by_year_spans_the_years_the_price_data_covers():
    """If this fails, quiet years at the edges of the data vanish from the table instead of showing zeros."""
    out = S.summary_by_year(_days(A=["2019-05-01"]), pd.DataFrame(columns=["signal_date"]), pd.DataFrame(columns=["week"]),
                            years=range(2017, 2021))
    assert list(out.index) == [2017, 2018, 2019, 2020]
    assert out["artifact_days"].tolist() == [0, 0, 1, 0] and out["stocks_flagged"].tolist() == [0, 0, 1, 0]


def test_summary_by_year_with_nothing_flagged_is_an_empty_table_not_an_error():
    """If this fails, a --limit run with no flagged days crashes instead of printing an empty table."""
    out = S.summary_by_year({}, pd.DataFrame(columns=["signal_date"]), pd.DataFrame(columns=["week"]))
    assert len(out) == 0 and list(out.columns) == ["artifact_days", "stocks_flagged", "c0_trades_dropped", "event_rows_dropped"]


# ── scanning the cache ──────────────────────────────────────────────────────────────

def _daily_frames(n_symbols=6):
    idx = pd.bdate_range("2019-01-01", periods=120, name="date")
    out = {}
    for k in range(n_symbols):
        close = np.full(len(idx), 10.0)
        if k % 2 == 0:
            close[60 + k:] = 100.0                              # a quiet 10:1 re-quote
        out[f"S{k}"] = pd.DataFrame({"close": close, "volume": np.full(len(idx), 200_000.0)}, index=idx)
    return out


def test_compute_artifact_days_flags_the_requoted_stocks_only():
    """If this fails, the scan returns days for clean stocks or misses a re-quote."""
    frames = _daily_frames()
    got = S.compute_artifact_days(frames, workers=1)
    assert sorted(got) == ["S0", "S2", "S4"]
    assert list(got["S2"]) == [frames["S2"].index[62]]


def test_the_fork_pool_returns_the_same_days_as_the_in_process_scan():
    """If this fails, parallel runs (--workers 8) differ from serial ones."""
    frames = _daily_frames()
    serial = S.compute_artifact_days(frames, workers=1)
    pooled = S.compute_artifact_days(frames, workers=2)
    assert sorted(serial) == sorted(pooled)
    for k in serial:
        assert list(serial[k]) == list(pooled[k])


# ── main (parquet, tmp_path only) ───────────────────────────────────────────────────

def _world(tmp_path):
    """history with B (re-quoted on idx 200), C (clean), Y (re-quoted on idx 300), X (clean); a trades file
    and an event-rows file over the same four stocks."""
    idx = pd.bdate_range("2018-06-01", periods=500)
    frames = []
    for sym, jump in (("B", 200), ("C", None), ("X", None), ("Y", 300)):
        close = np.full(len(idx), 10.0)
        if jump is not None:
            close[jump:] = 100.0
        frames.append(pd.DataFrame({"date": idx, "symbol": sym, "open": close, "high": close, "low": close,
                                    "close": close, "volume": 200_000.0}))
    hist = tmp_path / "hist"
    hist.mkdir()
    pd.concat(frames).to_parquet(hist / "batch_0000.parquet", index=False)
    trades = pd.DataFrame([
        _trade("B", idx[190], idx[210], -1.0),                      # contains B's re-quote on idx[200]
        _trade("C", idx[100], idx[120], 2.0),                       # clean
        _trade("B", idx[300], idx[320], 4.0, reason="end"),         # censored: excluded from everything
        _trade("C", idx[100], idx[120], 8.0, config="cost_2pct"),   # other configuration
    ])
    trades.to_parquet(tmp_path / "trades.parquet", index=False)
    rng = np.random.default_rng(11)
    weeks, syms = [T("2019-02-15"), T("2020-05-29")], ["B", "C", "X", "Y"]
    n = len(weeks) * len(syms)
    ev = _signature_frame().iloc[[0] * n].reset_index(drop=True)
    ev["symbol"] = syms * len(weeks)
    ev["week"] = [w for w in weeks for _ in syms]
    ev[S.OUTCOME] = rng.normal(0.1, 0.3, n)
    ev.to_parquet(tmp_path / "event_rows.parquet", index=False)
    return hist, tmp_path / "trades.parquet", tmp_path / "event_rows.parquet"


def _argv(tmp_path, hist, trades, ev, *extra):
    return ["--history", str(hist), "--trades", str(trades), "--event-rows", str(ev),
            "--out", str(tmp_path / "a.json"), "--workers", "1", "--n-boot", "100", *extra]


def test_main_writes_strict_json_with_before_and_after(tmp_path, capsys):
    """If this fails, the end-to-end run (parquet in, strict JSON out) is broken or emits NaN/Infinity."""
    pytest.importorskip("pyarrow")
    hist, trades, ev = _world(tmp_path)
    assert S.main(_argv(tmp_path, hist, trades, ev)) == 0
    text = (tmp_path / "a.json").read_text()

    def refuse(const):
        raise AssertionError(f"non-strict JSON constant {const}")
    doc = json.loads(text, parse_constant=refuse)
    assert "NaN" not in text and "Infinity" not in text
    assert doc["post_hoc"] is True and set(doc) >= {"rule", "post_hoc", "turnaround", "events", "by_year"}
    tr = doc["turnaround"]
    assert tr["before"]["n"] == 2 and tr["after"]["n"] == 1 and tr["n_dropped"] == 1
    assert tr["before"]["mean_r"] == pytest.approx(0.5) and tr["after"]["mean_r"] == pytest.approx(2.0)
    sig = doc["events"]["signatures"]
    assert sorted(sig) == ["B0", "S1", "S2", "S3", "S4"]
    assert sig["B0"]["before"]["n"] == 8                          # 4 stocks x 2 held-out weeks, all above their SMA30
    assert sig["B0"]["after"]["n"] == 6 and sig["B0"]["n_dropped"] == 2   # B and Y in week 2019-02-15
    assert doc["events"]["control"]["n_contaminated"] == 2
    assert doc["data"]["n_stocks_flagged"] == 2 and doc["data"]["n_flagged_days"] == 2
    assert [r["year"] for r in doc["by_year"]] == sorted(r["year"] for r in doc["by_year"])
    out = capsys.readouterr().out
    assert "TURNAROUND" in out and "EVENTS" in out


def test_main_summary_only_prints_the_table_and_writes_nothing(tmp_path, capsys):
    """If this fails, --summary-only writes a result file (it must only print) or loses the count columns."""
    pytest.importorskip("pyarrow")
    hist, trades, ev = _world(tmp_path)
    assert S.main(_argv(tmp_path, hist, trades, ev, "--summary-only")) == 0
    assert not (tmp_path / "a.json").exists()
    out = capsys.readouterr().out
    for col in ("artifact_days", "stocks_flagged", "c0_trades_dropped", "event_rows_dropped"):
        assert col in out
    assert "TURNAROUND" not in out


def test_main_limit_uses_the_first_symbols_of_each_part(tmp_path, capsys):
    """If this fails, --limit does not subset (the smoke command would scan the whole universe)."""
    pytest.importorskip("pyarrow")
    hist, trades, ev = _world(tmp_path)
    assert S.main(_argv(tmp_path, hist, trades, ev, "--limit", "1")) == 0
    doc = json.loads((tmp_path / "a.json").read_text())
    assert doc["data"]["limited_run"] is True and doc["data"]["n_stocks_requested"] == 1     # B only: first sorted symbol of both parts
    assert doc["turnaround"]["before"]["n"] == 1 and doc["turnaround"]["n_dropped"] == 1
