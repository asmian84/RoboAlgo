"""scripts/run_screen.py: the CLI must screen the LATEST bar of live stocks, and nothing else.

The failure being guarded against: a screen of "today" that lists a stock which stopped trading
(its last, stale bar still satisfies the rule), that evaluates the "previous bar" rule on the
current bar (so a bottom-then-breakout screen can never fire, or fires on the wrong bar), or that
cuts each stock's history so short that the 2-year-high column is NaN and the bottom rule silently
matches nothing. Each guard below is built as a trap and is mutation-tested.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from setup_engine.screen import COLUMNS, Screen, clean_daily, indicators

ROOT = Path(__file__).resolve().parents[2]
AS_OF = pd.Timestamp("2026-09-29")
SPIKE = Screen({"Relative Volume": "Over 2"})


def _load():
    spec = importlib.util.spec_from_file_location("run_screen", ROOT / "scripts" / "run_screen.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_screen"] = mod
    spec.loader.exec_module(mod)
    return mod


RS = _load()


def _flat(n=60, end=AS_OF, spike_at=None):
    """Flat 10.0 bars with volume 1e6 ending at ``end``; volume 5e6 at position ``spike_at``
    (negative counts from the end)."""
    idx = pd.bdate_range(end=end, periods=n)
    vol = np.full(n, 1e6)
    if spike_at is not None:
        vol[spike_at] = 5e6
    return pd.DataFrame({"open": 10.0, "high": 10.1, "low": 9.9, "close": 10.0, "volume": vol}, index=idx)


def _stale(spike_at=-1, days=3):
    """A stock whose last bar (the spike) is ``days`` business days before AS_OF."""
    return _flat(end=AS_OF - pd.offsets.BDay(days), spike_at=spike_at)


def _walk(n, seed=2, start="2014-01-02"):
    rng = np.random.default_rng(seed)
    close = 20 * np.exp(np.cumsum(rng.normal(0.0004, 0.02, n)))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.008, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.008, n)))
    vol = rng.integers(500_000, 3_000_000, n).astype(float)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol},
                        index=pd.bdate_range(start, periods=n))


def _long(stocks):
    """{symbol: daily frame} -> the long layout of a history batch file."""
    parts = [df.rename_axis("date").reset_index().assign(symbol=sym) for sym, df in stocks.items()]
    return pd.concat(parts, ignore_index=True)[["date", "symbol", "open", "high", "low", "close", "volume"]]


def _history(tmp_path, batches):
    """Write one batch_NNNN.parquet per {symbol: frame} dict in ``batches``."""
    pytest.importorskip("pyarrow", reason="the history cache is parquet; pip install pyarrow")
    hist = tmp_path / "history"
    hist.mkdir()
    for k, stocks in enumerate(batches):
        _long(stocks).to_parquet(hist / f"batch_{k:04d}.parquet", index=False)
    return hist


def _universe():
    """SPIKER spikes on the as-of bar, FLATTY is flat, STALEY spiked on ITS last bar, 3 days early."""
    return {"SPIKER": _flat(spike_at=-1), "FLATTY": _flat(), "STALEY": _stale()}


# ── screen_latest ─────────────────────────────────────────────────────────────

def test_only_stocks_that_traded_on_the_as_of_date_are_screened():
    out = RS.screen_latest(list(_universe().items()), SPIKE, AS_OF)
    assert list(out["symbol"]) == ["SPIKER"]
    assert list(out.columns) == ["symbol", "date", "close", "relvol"]
    assert out.loc[0, "date"] == AS_OF
    assert out.loc[0, "relvol"] == pytest.approx(5.0)
    assert out.loc[0, "close"] == pytest.approx(10.0)


def test_a_stale_stock_is_not_even_evaluated():
    _, n = RS.evaluate_frames(list(_universe().items()), SPIKE, AS_OF)
    assert n == 2                        # SPIKER and FLATTY; STALEY's last bar is 3 days before as_of


def test_bars_after_the_as_of_date_do_not_exist_for_the_screen():
    """A stock that spikes the day AFTER as_of must not match as of as_of."""
    later = _flat(n=61, end=AS_OF + pd.offsets.BDay(1), spike_at=-1)
    assert later.index[-2] == AS_OF
    out = RS.screen_latest([("LATE", later)], SPIKE, AS_OF)
    assert out.empty
    assert list(RS.screen_latest([("LATE", later)], SPIKE, AS_OF + pd.offsets.BDay(1))["symbol"]) == ["LATE"]


def test_the_prior_rule_is_read_on_the_previous_bar():
    stocks = {"NOW": _flat(spike_at=-1), "YESTERDAY": _flat(spike_at=-2), "FLATTY": _flat()}
    frames = list(stocks.items())
    nothing = Screen({})
    now = RS.screen_latest(frames, SPIKE, AS_OF)
    prior = RS.screen_latest(frames, nothing, AS_OF, prior_screen=SPIKE)
    assert list(now["symbol"]) == ["NOW"]                # spiked on the as-of bar only
    assert list(prior["symbol"]) == ["YESTERDAY"]        # spiked on the previous bar only
    assert "prior_relvol" in prior.columns and prior.loc[0, "prior_relvol"] == pytest.approx(5.0)
    # both rules together: spiked yesterday AND quiet now ("Under 1": today's volume vs a mean lifted by the spike)
    both = RS.screen_latest(frames, Screen({"Relative Volume": "Under 1"}), AS_OF, prior_screen=SPIKE)
    assert list(both["symbol"]) == ["YESTERDAY"]
    assert both.loc[0, "relvol"] == pytest.approx(1e6 / 1.2e6)
    assert both.loc[0, "prior_relvol"] == pytest.approx(5.0)


def test_results_are_sorted_by_symbol():
    stocks = {k: _flat(spike_at=-1) for k in ("ZZZ", "AAA", "MMM")}
    out = RS.screen_latest(list(stocks.items()), SPIKE, AS_OF)
    assert list(out["symbol"]) == ["AAA", "MMM", "ZZZ"]


def test_a_700_bar_stock_has_a_finite_two_year_high_distance():
    """TAIL_BARS must cover the 504-bar window or every 2-year rule silently matches nothing."""
    n = 700
    close = np.r_[np.full(300, 100.0), np.full(400, 20.0)]
    df = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
                       "volume": 1e6}, index=pd.bdate_range(end=AS_OF, periods=n))
    last, _ = RS.latest_rows([("SLUMP", df)], AS_OF)
    assert np.isfinite(last.loc[0, "pct_below_high_2y"])
    assert np.isfinite(last.loc[0, "low_prior"])         # the Monthly Low stand-in needs 378 bars
    out = RS.screen_latest([("SLUMP", df)], Screen({"2-Year High": "70% or more below"}), AS_OF)
    assert list(out["symbol"]) == ["SLUMP"]


def test_the_tail_cut_changes_no_column_of_the_last_row():
    """Indicators on the last TAIL_BARS bars equal indicators on the whole history (Wilder converges)."""
    df = _walk(1500).rename_axis(None)
    as_of = df.index[-1]
    last, prior = RS.latest_rows([("LONG", df)], as_of)
    full = indicators(clean_daily(df))
    for got, row in ((last.loc[0, list(COLUMNS)], full.iloc[-1]), (prior.loc[0, list(COLUMNS)], full.iloc[-2])):
        assert np.allclose(got.to_numpy(dtype=float), row.to_numpy(), rtol=1e-9, equal_nan=True)
    assert prior.loc[0, "date"] == df.index[-2]


def test_cache_dates_reads_only_the_date_column(tmp_path, monkeypatch):
    hist = _history(tmp_path, [{"AAA": _flat(n=30)}, {"BBB": _flat(n=45, end=AS_OF + pd.offsets.BDay(2))}])
    seen = []
    real = pd.read_parquet

    def spy(path, *a, **kw):
        seen.append(kw.get("columns"))
        return real(path, *a, **kw)

    monkeypatch.setattr(pd, "read_parquet", spy)
    dates = RS.cache_dates(hist)
    assert seen == [["date"], ["date"]]
    assert dates.is_monotonic_increasing and dates.is_unique
    assert dates[-1] == AS_OF + pd.offsets.BDay(2)
    assert len(dates) == len(pd.bdate_range(dates[0], dates[-1]))


# ── main ──────────────────────────────────────────────────────────────────────

def test_main_lists_the_match_and_not_the_others(tmp_path, capsys):
    hist = _history(tmp_path, [_universe()])
    rc = RS.main(["--history", str(hist), "--rules", "Relative Volume: Over 2", "--workers", "1"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "cache last date: 2026-09-29 | as of: 2026-09-29 | stocks evaluated: 2 | matches: 1" in out
    assert "SPIKER" in out
    assert "FLATTY" not in out and "STALEY" not in out
    assert "screen:" in out and "relvol > 2" in out


def test_main_as_of_picks_the_latest_cache_date_on_or_before(tmp_path, capsys):
    hist = _history(tmp_path, [{"SPIKER": _flat(n=80, end=AS_OF + pd.offsets.BDay(10), spike_at=-11)}])
    # a Saturday: the effective date is the Friday before it
    rc = RS.main(["--history", str(hist), "--rules", "Relative Volume: Over 2", "--workers", "1",
                  "--as-of", str((AS_OF + pd.Timedelta(days=4)).date())])
    out = capsys.readouterr().out
    assert rc == 0
    assert "cache last date: 2026-10-13 | as of: 2026-10-02 |" in out
    assert "SPIKER" not in out                            # the spike was on 2026-09-29, 3 bars earlier
    rc = RS.main(["--history", str(hist), "--rules", "Relative Volume: Over 2", "--workers", "1",
                  "--as-of", "2026-09-29"])
    out = capsys.readouterr().out
    assert rc == 0 and "as of: 2026-09-29" in out and "SPIKER" in out


def test_main_as_of_before_the_cache_is_an_error(tmp_path, capsys):
    hist = _history(tmp_path, [_universe()])
    rc = RS.main(["--history", str(hist), "--rules", "Relative Volume: Over 2", "--as-of", "1990-01-01"])
    captured = capsys.readouterr()
    assert rc == 2
    assert "no cache date" in captured.err
    assert captured.out == ""


def test_main_prior_rules_are_applied_and_described(tmp_path, capsys):
    stocks = {"YESTERDAY": _flat(spike_at=-2), "NOW": _flat(spike_at=-1)}
    hist = _history(tmp_path, [stocks])
    rc = RS.main(["--history", str(hist), "--workers", "1", "--rules", "Relative Volume: Under 1",
                  "--prior-rules", "Relative Volume: Over 2"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "YESTERDAY" in out and "NOW" not in out.split("prior screen")[1]
    assert "prior screen" in out and "prior_relvol" in out


def test_main_bad_rules_return_2_and_name_the_valid_choices(tmp_path, capsys):
    hist = _history(tmp_path, [_universe()])
    assert RS.main(["--history", str(hist), "--rules", "Foo: Bar"]) == 2
    err = capsys.readouterr().err
    assert "valid" in err.lower() and "Price" in err
    assert RS.main(["--history", str(hist), "--rules", "Price: Over one dollar"]) == 2
    assert "Over $1" in capsys.readouterr().err
    assert RS.main(["--history", str(hist), "--prior-rules", "Relative Volume Over 2"]) == 2      # no colon
    assert "':'" in capsys.readouterr().err


def test_main_needs_at_least_one_rule(capsys):
    assert RS.main([]) == 2
    assert "--rules" in capsys.readouterr().err


def test_main_missing_history_is_an_error(tmp_path, capsys):
    assert RS.main(["--history", str(tmp_path / "nope"), "--rules", "Relative Volume: Over 2"]) == 2
    assert "batch_" in capsys.readouterr().err


def test_main_list_filters_needs_no_history(tmp_path, capsys):
    rc = RS.main(["--list-filters", "--history", str(tmp_path / "nope")])
    out = capsys.readouterr().out
    assert rc == 0
    assert "52-Week High" in out and "Monthly Low" in out and "Higher low" in out


def test_main_workers_give_the_same_answer_as_sequential(tmp_path, capsys):
    hist = _history(tmp_path, [{"SPIKER": _flat(spike_at=-1), "FLATTY": _flat()},
                               {"SPIKE2": _flat(spike_at=-1), "STALEY": _stale()}])
    argv = ["--history", str(hist), "--rules", "Relative Volume: Over 2"]
    assert RS.main(argv + ["--workers", "1"]) == 0
    seq = capsys.readouterr().out
    assert RS.main(argv + ["--workers", "2"]) == 0
    par = capsys.readouterr().out
    assert seq == par
    assert "matches: 2" in seq and "SPIKE2" in seq and "SPIKER" in seq


def test_main_limit_batches_and_csv(tmp_path, capsys):
    hist = _history(tmp_path, [{"SPIKER": _flat(spike_at=-1)}, {"SPIKE2": _flat(spike_at=-1)}])
    csv = tmp_path / "out.csv"
    rc = RS.main(["--history", str(hist), "--rules", "Relative Volume: Over 2", "--workers", "1",
                  "--limit-batches", "1", "--csv", str(csv)])
    out = capsys.readouterr().out
    assert rc == 0 and "SPIKER" in out and "SPIKE2" not in out
    saved = pd.read_csv(csv)
    assert list(saved["symbol"]) == ["SPIKER"]
    assert saved.loc[0, "date"] == "2026-09-29"
    assert list(saved.columns) == ["symbol", "date", "close", "relvol"]


def test_main_reads_timezone_aware_dates(tmp_path, capsys):
    pytest.importorskip("pyarrow", reason="the history cache is parquet; pip install pyarrow")
    hist = tmp_path / "history"
    hist.mkdir()
    long = _long({"SPIKER": _flat(spike_at=-1)})
    long["date"] = long["date"].dt.tz_localize("UTC")
    long.to_parquet(hist / "batch_0000.parquet", index=False)
    rc = RS.main(["--history", str(hist), "--rules", "Relative Volume: Over 2", "--workers", "1"])
    out = capsys.readouterr().out
    assert rc == 0 and "SPIKER" in out and "as of: 2026-09-29" in out


def test_main_says_so_when_nothing_matches(tmp_path, capsys):
    hist = _history(tmp_path, [{"FLATTY": _flat()}])
    rc = RS.main(["--history", str(hist), "--rules", "Relative Volume: Over 2", "--workers", "1"])
    out = capsys.readouterr().out
    assert rc == 0 and "matches: 0" in out and "(no matches)" in out
