"""
The pre-registered Kronos test must measure what it says it measures. These tests pin the rules that a
plausible bug would quietly break, with no torch, no network and no price cache:

  * look-ahead: the universe at an origin is built from bars up to the origin close only, and the realized
    return runs from the origin close to the CLOSE OF ORIGIN + 20 trading days;
  * universe boundaries: 400 bars exactly is enough, 399 is not; a last close of exactly $1 is kept; the
    dollar-volume rank uses the last 60 bars and cuts at top 500;
  * the IC: sign, a hand-computed value, ties, and a constant side returning None;
  * the decile spread: top versus bottom, and top versus the universe mean;
  * the verdict: a lower bound of exactly 0 is NOT a PASS, and fewer than 12 clean origins is UNDERPOWERED;
  * the placebo: shuffled ranks score about zero, so a real IC is the pipeline and not an artefact;
  * output: the JSON is strict (no NaN, no Infinity) and the write is atomic;
  * import: setup_engine.kronos_eval loads with torch unavailable, so the test suite never needs the model.

Everything is synthetic and in memory.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from setup_engine import kronos_eval as K

ROOT = Path(__file__).resolve().parents[2]


def _load(name):
    """Load scripts/<name>.py as a module (house pattern: scripts are not a package)."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ── synthetic prices ───────────────────────────────────────────────────────────────────────────────

def frame(n=520, start="2020-01-01", base=100.0, vol=1_000_000, drift=0.0003, seed=0):
    """A deterministic daily OHLCV frame on business days."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    close = base * np.exp(np.cumsum(rng.normal(drift, 0.01, n)))
    return pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99,
                         "close": close, "volume": float(vol)}, index=idx)


def with_bar(daily, when, row):
    """A copy of ``daily`` with one extra OHLCV bar appended to every symbol's frame."""
    out = {}
    for sym, d in daily.items():
        extra = pd.DataFrame([row], index=pd.DatetimeIndex([pd.Timestamp(when)]), columns=list(d.columns))
        out[sym] = pd.concat([d, extra])
    return out


# ── origins ───────────────────────────────────────────────────────────────────────────────────────

def test_month_end_days_takes_the_last_trading_day_of_the_month():
    days = pd.bdate_range("2025-01-01", "2025-03-31")
    days = days[days != pd.Timestamp("2025-01-31")]          # a holiday on a month end
    ends = K.month_end_days(days)
    assert ends == [pd.Timestamp("2025-01-30"), pd.Timestamp("2025-02-28"), pd.Timestamp("2025-03-31")]


def test_origin_windows_split_and_drop_immature_origins():
    days = pd.bdate_range("2023-01-01", "2026-10-06")
    w = K.origin_windows(days)
    assert all(t >= K.CLEAN_START for t in w["clean"])
    assert w["clean"][0] == pd.Timestamp("2025-07-31")
    assert all(K.INFO_START <= t <= K.INFO_END for t in w["informational"])
    assert pd.Timestamp("2023-01-31") in w["informational"]
    # the last month end has too few forward days to have a realized 20-day return
    assert pd.Timestamp("2026-09-30") in w["immature"]
    assert pd.Timestamp("2026-09-30") not in w["all"]


def test_origin_window_needs_twenty_forward_days_exactly():
    days = pd.bdate_range("2025-06-01", "2025-08-15")
    w = K.origin_windows(days, horizon=20)
    # 2025-07-31 has only 11 trading days after it in this calendar
    assert pd.Timestamp("2025-07-31") in w["immature"]
    days = pd.bdate_range("2025-06-01", "2025-09-02")        # 21 after 07-31
    w = K.origin_windows(days, horizon=20)
    assert pd.Timestamp("2025-07-31") in w["clean"]


# ── look-ahead ────────────────────────────────────────────────────────────────────────────────────

def test_universe_ignores_every_bar_after_the_origin():
    daily = {f"S{i}": frame(seed=i) for i in range(5)}
    origin = pd.Timestamp(daily["S0"].index[-1])          # the last bar: plenty of history, origin in the frame
    before = K.universe_candidates(daily, origin)
    assert len(before) == 5                                # the test is not vacuous
    after = K.universe_candidates(with_bar(daily, origin + pd.Timedelta(days=1), [1, 1, 1, 999.0, 1e12]), origin)
    pd.testing.assert_frame_equal(before, after)           # a huge future bar moves nothing


def test_realized_return_runs_to_origin_plus_twenty_and_skips_the_origin_itself():
    days = pd.bdate_range("2021-01-01", periods=600)
    close = np.arange(600, dtype=float) + 100.0
    d = pd.DataFrame({"open": close, "high": close, "low": close, "close": close,
                      "volume": 1.0}, index=days)
    origin = days[300]
    assert K.realized_return(d, origin) == pytest.approx(close[320] / close[300] - 1.0)


def test_forecast_input_ends_at_the_origin_and_ignores_the_future():
    S = _load("kronos_eval")
    d = frame(n=520)
    origin = d.index[450]
    x, x_ts, y_ts = S.forecast_inputs(d, origin)
    assert len(x) == 400 and x.index[-1] == origin
    assert x_ts.iloc[-1] == origin and len(y_ts) == 20
    assert y_ts.iloc[0] == d.index[451]                    # the first bar AFTER the origin
    later = pd.DataFrame([d.iloc[-1].values], index=[d.index[-1] + pd.Timedelta(days=1)],
                         columns=d.columns)
    x2, _, _ = S.forecast_inputs(pd.concat([d, later]), origin)
    pd.testing.assert_frame_equal(x, x2, check_freq=False)  # a future bar cannot move the input
    assert list(x.columns) == ["open", "high", "low", "close", "volume"]
    assert S.forecast_inputs(d.iloc[:300], d.index[299])[0] is None    # too short to forecast


def test_realized_return_is_none_without_a_full_forward_window():
    d = frame(n=100)
    assert K.realized_return(d, d.index[80]) is None          # only 19 bars after it
    assert K.realized_return(d, pd.Timestamp("1999-01-04")) is None   # not a bar of the series


def test_momentum_and_reversal_use_the_registered_windows():
    days = pd.bdate_range("2021-01-01", periods=600)
    close = np.arange(600, dtype=float) + 100.0
    d = pd.DataFrame({"open": close, "high": close, "low": close, "close": close,
                      "volume": 1.0}, index=days)
    i = 400
    assert K.momentum(d, days[i]) == pytest.approx(close[i - 21] / close[i - 252] - 1.0)
    assert K.reversal(d, days[i]) == pytest.approx(-(close[i] / close[i - 21] - 1.0))
    assert K.momentum(d, days[100]) is None


# ── universe boundaries ──────────────────────────────────────────────────────────────────────────

def test_universe_bar_and_price_boundaries():
    origin = pd.Timestamp("2022-06-30")
    daily = {
        "EXACT400": frame(n=400, base=10.0, seed=1),
        "SHORT399": frame(n=399, base=10.0, seed=2),
        "DOLLAR": frame(n=400, base=1.0, seed=3),
        "PENNY": frame(n=400, base=0.99, seed=4),
    }
    daily["DOLLAR"].loc[:, ["open", "high", "low", "close"]] = 1.0
    daily["PENNY"].loc[:, ["open", "high", "low", "close"]] = 0.99
    c = K.universe_candidates(daily, origin).set_index("symbol")
    assert "SHORT399" not in c.index
    assert "EXACT400" in c.index
    assert "DOLLAR" in c.index                     # $1 exactly is kept
    assert "PENNY" not in c.index                  # 0.99 is not


def test_universe_rank_uses_the_last_sixty_bars_and_cuts_at_top_n():
    origin = pd.Timestamp("2022-06-30")
    daily = {f"S{i}": frame(n=400, vol=10 ** (6 + i), seed=i) for i in range(5)}
    # S0 traded enormously 61-400 bars ago but is quiet in the last 60: its dollar volume must be the recent one
    old = frame(n=400, vol=1, seed=9)
    old.iloc[:340, old.columns.get_loc("volume")] = 1e12
    daily["OLD"] = old
    c = K.universe_candidates(daily, origin).set_index("symbol")
    assert c.loc["OLD", "dvol"] == pytest.approx(np.mean(
        (old["close"] * old["volume"]).to_numpy()[-60:]))
    picked = K.select_universe(K.universe_candidates(daily, origin), top_n=2)
    assert picked == ["S4", "S3"]


def test_forecast_return_is_the_mean_over_sample_paths():
    assert K.forecast_return([110.0, 90.0], 100.0) == pytest.approx(0.0)
    assert K.forecast_return(np.array([100.0, 110.0]), 100.0) == pytest.approx(0.05)
    with pytest.raises(ValueError):
        K.forecast_return([], 100.0)


# ── IC and deciles ──────────────────────────────────────────────────────────────────────────────

def test_ic_sign_is_what_the_metric_means():
    assert K.spearman_ic([1, 2, 3, 4, 5], [10, 20, 30, 40, 50]) == pytest.approx(1.0)
    assert K.spearman_ic([1, 2, 3, 4, 5], [50, 40, 30, 20, 10]) == pytest.approx(-1.0)


def test_ic_hand_computed_case_and_ties():
    # ranks of x are 1..5; y ranks are 2,1,4,3,5. Pearson of those ranks is 8 / sqrt(10*10) = 0.8
    assert K.spearman_ic([1, 2, 3, 4, 5], [2, 1, 4, 3, 5]) == pytest.approx(0.8)
    # x tied at the bottom: ranks 1.5,1.5,3,4 against 1,2,3,4 -> 4.5 / sqrt(4.5 * 5)
    assert K.spearman_ic([1, 1, 2, 3], [1, 2, 3, 4]) == pytest.approx(4.5 / np.sqrt(4.5 * 5.0))


def test_ic_is_none_when_it_cannot_be_computed():
    assert K.spearman_ic([1, 2], [1, 2]) is None                  # fewer than 3 pairs
    assert K.spearman_ic([1, 1, 1, 1], [1, 2, 3, 4]) is None      # constant forecast
    assert K.spearman_ic([1, 2, 3], [1, np.nan, 3]) is None       # one pair left after cleaning


def test_decile_spread_top_minus_bottom_and_universe():
    f = np.arange(20, dtype=float)
    r = np.arange(20, dtype=float)
    d = K.decile_spread(f, r)
    assert d["k"] == 2 and d["n"] == 20
    assert d["top"] == pytest.approx(18.5)
    assert d["bottom"] == pytest.approx(0.5)
    assert d["spread"] == pytest.approx(18.0)
    assert d["universe_mean"] == pytest.approx(9.5)
    assert d["top_minus_universe"] == pytest.approx(9.0)


def test_decile_spread_drops_non_finite_rows():
    d = K.decile_spread([5.0, 4.0, 3.0, np.nan], [1.0, 0.0, -1.0, 9.0])
    assert d["n"] == 3 and d["k"] == 1
    assert d["top"] == pytest.approx(1.0) and d["bottom"] == pytest.approx(-1.0)


# ── placebo ─────────────────────────────────────────────────────────────────────────────────────

def test_placebo_scores_about_zero():
    rng = np.random.default_rng(7)
    ics = []
    for i in range(40):
        realized = rng.normal(0, 0.08, 300)
        forecast = realized + rng.normal(0, 1.0, 300)             # a strong true signal
        ics.append(K.spearman_ic(K.placebo_forecast(forecast, seed=K.PLACEBO_SEED + i), realized))
    assert np.mean(ics) == pytest.approx(0.0, abs=0.02)
    assert abs(np.mean(ics)) < 0.05


# ── verdict ─────────────────────────────────────────────────────────────────────────────────────

def test_verdict_boundary_zero_lower_bound_is_not_a_pass():
    pos = {"lo": 0.01, "hi": 0.05}
    zero = {"lo": 0.0, "hi": 0.05}
    assert K.evaluate_verdict(pos, pos, 14)["pass"] is True
    assert K.evaluate_verdict(zero, pos, 14)["pass"] is False
    assert K.evaluate_verdict(pos, zero, 14)["pass"] is False
    assert K.evaluate_verdict({"lo": None}, pos, 14)["pass"] is False
    assert K.evaluate_verdict(pos, pos, 14)["verdict"] == "PASS"
    assert K.evaluate_verdict(zero, pos, 14)["verdict"] == "no demonstrated edge"


def test_verdict_flags_an_underpowered_window():
    pos = {"lo": 0.01, "hi": 0.05}
    assert K.evaluate_verdict(pos, pos, 11)["underpowered"] is True
    assert K.evaluate_verdict(pos, pos, 12)["underpowered"] is False


# ── output ──────────────────────────────────────────────────────────────────────────────────────

def _result():
    """A minimal but complete result dict, the shape ``write_outputs`` is given at the end of a run."""
    ci = {"mean": 0.01, "lo": -0.02, "hi": 0.04, "p_le_0": 0.3, "n": 14, "n_clusters": 14}
    signal = {**ci, "n_origins": 14, "mean_names": 480.0}
    row = {"origin": pd.Timestamp("2026-01-30"), "n": 480, "n_realized": 478, "n_ic": 478, "k": 48,
           "ic_kronos": 0.02, "ic_momentum": 0.01, "ic_reversal": -0.03, "ic_placebo": 0.0,
           "spread_kronos": 0.004, "top_kronos": 0.02, "bottom_kronos": 0.016,
           "spread_momentum": 0.001, "spread_reversal": -0.002, "spread_placebo": 0.0,
           "top_minus_universe_kronos": 0.001, "top_minus_universe_momentum": 0.0,
           "top_minus_universe_reversal": 0.0, "top_minus_universe_placebo": 0.0}
    window = {"n_origins": 14, "per_origin": [row], "signals": {n: dict(signal) for n in K.SIGNALS},
              "spreads": {n: dict(ci) for n in K.SIGNALS},
              "top_minus_universe": {n: dict(ci) for n in K.SIGNALS},
              "kronos_minus_momentum": {**ci, "n_a": 14, "n_b": 14}}
    return {"generated": "2026-10-20", "data_end": pd.Timestamp("2026-10-06"),
            "kronos": {"commit": "0" * 40, "commit_date": "2026-04-13", "model": "m", "tokenizer": "t",
                       "device": "mps", "lookback": 400, "horizon": 20, "T": 1.0, "top_p": 0.9,
                       "sample_count": 5, "top_n": 500, "min_price": 1.0, "min_bars": 400, "seeded": 1},
            "origins": {"clean": [pd.Timestamp("2026-01-30")], "informational": []},
            "universe": {"2026-01-30": 480}, "clean": window, "informational": window,
            "verdict": {"verdict": "no demonstrated edge", "pass": False, "underpowered": False,
                        "n_clean_origins": 14},
            "contamination": {"info_minus_clean": ci, "note": "flat"},
            "caveats": ["survivors only"], "nan": float("nan"), "inf": float("inf")}


def test_json_output_is_strict_and_the_write_is_atomic(tmp_path):
    S = _load("kronos_eval")
    good = {"a": np.float64(0.5), "b": [np.int64(2)], "c": None, "d": float("inf"), "e": float("nan")}
    assert json.loads(json.dumps(S._jsonable(good), allow_nan=False)) == \
        {"a": 0.5, "b": [2], "c": None, "d": None, "e": None}
    paths = S.write_outputs(_result(), tmp_path / "kronos_eval_2026-10-20")
    text = Path(paths["json"]).read_text()
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text)["nan"] is None and json.loads(text)["inf"] is None
    assert json.loads(text)["data_end"] == "2026-10-06"
    assert "no demonstrated edge" in Path(paths["md"]).read_text()
    assert not list(tmp_path.glob("*.tmp"))                      # nothing left half-written


def test_module_imports_without_torch():
    import subprocess
    code = (
        "import sys\n"
        "class Ban:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] == 'torch':\n"
        "            raise ImportError('torch is banned in this test')\n"
        "        return None\n"
        "sys.meta_path.insert(0, Ban())\n"
        "import setup_engine.kronos_eval as K\n"
        "assert 'torch' not in sys.modules\n"
        "print(K.LOOKBACK, K.HORIZON)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "400 20"
