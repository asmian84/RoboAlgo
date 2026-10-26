"""setup_engine.trade_sim: the exit rules must fill where a real order could fill, and never peek.

The failure being guarded against: a backtest that flatters a setup. Each test below is one way
that happens (entering at a price nobody could get, sizing the stop from a bar that had not
happened, letting a trailing stop use the bar it is tested against or move down, crediting the
target on a bar that also stopped out, paying a gap-through-stop at the stop price, taking
overlapping trades, forgetting the cost), built on a hand-made price path whose answer is known
exactly. The jitted kernels and the pure-Python fallback must also agree.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from setup_engine import trade_sim as ts
from setup_engine.trade_sim import ExitModel, REASONS, TRADE_COLUMNS, simulate, simulate_each

BASE = ExitModel(2.0, None, None, 120, 0.0)


def _ind(rows, atr=1.0):
    """rows: (open, high, low, close) per bar. atr: scalar or one value per bar."""
    a = np.array(rows, dtype=float)
    n = len(a)
    atr = np.full(n, float(atr)) if np.isscalar(atr) else np.asarray(atr, dtype=float)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({"open": a[:, 0], "high": a[:, 1], "low": a[:, 2], "close": a[:, 3], "atr14": atr},
                        index=idx)


FLAT = (10.0, 10.1, 9.9, 10.0)


def _one(df, model=BASE, t=0):
    out = simulate(df, [t], model, cooldown=0)
    assert len(out) == 1
    return out.iloc[0]


def _walk_prices(n=400, seed=5, jumps=False):
    """Random walk with a Wilder ATR, as the screen computes it."""
    rng = np.random.default_rng(seed)
    close = 20 * np.exp(np.cumsum(rng.standard_t(3, n) * 0.012 if jumps else rng.normal(0, 0.015, n)))
    open_ = close * (1 + (rng.standard_t(2, n) * 0.01 if jumps else rng.normal(0, 0.004, n)))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, n)))
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close},
                      index=pd.bdate_range("2018-01-01", periods=n))
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    df["atr14"] = tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    return df


# ── entry and stop ─────────────────────────────────────────────────────────────

def test_entry_is_the_next_open_and_the_stop_uses_the_atr_known_at_the_signal():
    df = _ind([(10, 10, 10, 10), (11, 11.5, 10.5, 11), (11, 11.2, 10.8, 11)], atr=[1, 5, 5])
    tr = _one(df)
    assert tr["entry"] == pytest.approx(11.0)             # the next open, not the signal close of 10
    assert tr["entry_date"] == df.index[1]
    assert tr["signal_date"] == df.index[0]
    assert tr["stop"] == pytest.approx(9.0)               # 11 - 2 * atr[0]; atr[1] = 5 would give 1
    assert tr["risk"] == pytest.approx(2.0)
    assert tr["reason"] == "end"
    assert tr["r_gross"] == pytest.approx(0.0)
    assert tr["atr_pct"] == pytest.approx(1 / 10)         # ATR at the signal over the signal close


def test_the_walk_itself_stops_at_entry_minus_the_atr_known_at_the_signal():
    """The reported entry/stop columns are one thing; the level the path is tested against is another."""
    df = _ind([(10, 10, 10, 10), (11, 11.2, 8.9, 10), (10, 10.1, 9.9, 10)], atr=[1, 5, 5])
    tr = _one(df)
    assert tr["exit_i"] == 1                              # low 8.9 is through the stop at 11 - 2 = 9
    assert tr["exit"] == pytest.approx(9.0)
    assert tr["reason"] == "stop"
    assert tr["r_gross"] == pytest.approx(-1.0)
    ok = _ind([(10, 10, 10, 10), (11, 11.2, 9.1, 10), (10, 10.1, 9.9, 10)], atr=[1, 5, 5])
    assert _one(ok)["reason"] == "end"                    # 9.1 is above the stop: it must NOT be stopped


def test_a_clean_stop_loses_one_r_plus_the_cost():
    df = _ind([(10, 10, 10, 10), (10, 10.5, 9.5, 10), (9.8, 9.9, 7.5, 8)])
    tr = _one(df, ExitModel(2.0, None, None, 120, 0.01))
    assert tr["exit"] == pytest.approx(8.0)
    assert tr["reason"] == "stop"
    assert tr["r_gross"] == pytest.approx(-1.0)
    assert tr["cost_r"] == pytest.approx(0.05)            # 1% of 10, over a risk of 2
    assert tr["r_net"] == pytest.approx(-1.05)


def test_a_gap_through_the_stop_fills_at_the_open_not_the_stop():
    df = _ind([(10, 10, 10, 10), (10, 10.5, 9.5, 10), (7, 7.2, 6.5, 7)])
    tr = _one(df)
    assert tr["exit"] == pytest.approx(7.0)
    assert tr["reason"] == "gap_stop"
    assert tr["r_gross"] == pytest.approx(-1.5)


def test_the_stop_wins_when_one_bar_could_hit_both_stop_and_target():
    df = _ind([(10, 10, 10, 10), (10, 10.2, 9.8, 10), (10, 15, 7.5, 12)])
    tr = _one(df, ExitModel(2.0, None, 2.0, 120, 0.0))
    assert tr["exit"] == pytest.approx(8.0)
    assert tr["reason"] == "stop"
    assert tr["r_gross"] == pytest.approx(-1.0)


def test_a_stop_on_the_entry_bar_is_taken():
    df = _ind([(10, 10, 10, 10), (10, 10.2, 7.9, 9), (9, 9.1, 8.9, 9)])
    tr = _one(df)
    assert tr["exit_i"] == 1
    assert tr["bars"] == 0
    assert tr["exit"] == pytest.approx(8.0)
    assert tr["reason"] == "stop"


# ── target ─────────────────────────────────────────────────────────────────────

def test_the_target_fills_at_the_target_and_a_favourable_gap_fills_at_the_open():
    m = ExitModel(2.0, None, 2.0, 120, 0.0)
    hit = _ind([(10, 10, 10, 10), (10, 10.2, 9.8, 10), (10.5, 14.5, 10, 14)])
    tr = _one(hit, m)
    assert (tr["exit"], tr["reason"]) == (pytest.approx(14.0), "target")
    assert tr["r_gross"] == pytest.approx(2.0)
    gap = _ind([(10, 10, 10, 10), (10, 10.2, 9.8, 10), (15, 16, 14.8, 15.5)])
    tr = _one(gap, m)
    assert (tr["exit"], tr["reason"]) == (pytest.approx(15.0), "target")
    assert tr["r_gross"] == pytest.approx(2.5)


# ── trailing stop ──────────────────────────────────────────────────────────────

def test_the_trail_ratchets_up_never_down_and_uses_only_the_prior_bar():
    rows = [(10, 10, 10, 10), (10, 10, 9.5, 10), (11, 14, 10.9, 13.5),
            (13, 13.2, 12, 12.5), (11.5, 11.8, 10.5, 11), (11, 11, 11, 11)]
    df = _ind(rows, atr=[1, 1, 1, 2, 2, 2])
    tr = _one(df, ExitModel(2.0, 3.0, None, 120, 0.0))
    # bar 3: stop = 14 - 3*1 = 11. bar 4: 14 - 3*2 = 8 would LOWER it, so it stays 11 and bar 4 (low 10.5) exits.
    assert tr["exit_i"] == 4
    assert tr["exit"] == pytest.approx(11.0)
    assert tr["reason"] == "trail"
    assert tr["r_gross"] == pytest.approx(0.5)
    assert tr["bars"] == 3
    assert tr["mfe_r"] == pytest.approx(2.0)              # best high 14, entry 10, risk 2


# ── time and end ───────────────────────────────────────────────────────────────

def test_the_time_stop_exits_at_the_close_max_bars_after_entry():
    df = _ind([FLAT] * 8)
    tr = _one(df, ExitModel(2.0, None, None, 3, 0.0))
    assert tr["exit_i"] == 4                              # entry bar 1, plus 3
    assert tr["exit"] == pytest.approx(10.0)
    assert tr["reason"] == "time"
    assert tr["bars"] == 3


def test_a_trade_still_open_at_the_end_of_the_data_exits_at_the_last_close():
    df = _ind([FLAT] * 4)
    tr = _one(df)
    assert tr["exit_i"] == 3
    assert tr["reason"] == "end"


# ── invalid inputs ─────────────────────────────────────────────────────────────

def test_a_signal_on_the_last_bar_makes_no_trade_and_keeps_the_columns():
    out = simulate(_ind([FLAT] * 5), [4], BASE)
    assert len(out) == 0
    assert tuple(out.columns) == TRADE_COLUMNS


def test_no_trade_without_a_usable_atr_or_with_a_stop_below_zero():
    nan_atr = _ind([FLAT] * 6, atr=[np.nan] + [1.0] * 5)
    assert len(simulate(nan_atr, [0], BASE, cooldown=0)) == 0
    zero_atr = _ind([FLAT] * 6, atr=[0.0] + [1.0] * 5)
    assert len(simulate(zero_atr, [0], BASE, cooldown=0)) == 0
    assert not simulate_each(zero_atr, np.ones(6, dtype=bool), BASE)["valid"][0]    # risk 0 would divide by zero
    assert not simulate_each(nan_atr, np.ones(6, dtype=bool), BASE)["valid"][0]
    huge = _ind([FLAT] * 6, atr=[6.0] + [1.0] * 5)        # entry 10 - 2*6 < 0
    assert len(simulate(huge, [0], BASE, cooldown=0)) == 0
    each = simulate_each(huge, np.ones(6, dtype=bool), BASE)
    assert not each["valid"][0] and each["valid"][1]       # bar 1 has atr 1: fine
    assert np.isnan(each["r_gross"][0]) and each["exit_i"][0] == -1 and each["reason"][0] == -1
    assert not each["valid"][5]                            # last bar: no next bar


def test_out_of_range_signal_indices_are_refused():
    with pytest.raises(ValueError, match="out of range"):
        simulate(_ind([FLAT] * 5), [5], BASE)
    with pytest.raises(ValueError, match="out of range"):
        simulate(_ind([FLAT] * 5), [-1], BASE)


def test_the_exit_model_refuses_nonsense():
    for bad in ({"stop_atr": 0.0}, {"trail_atr": 0.0}, {"target_r": -1.0}, {"max_bars": -1}, {"cost_pct": -0.01}):
        with pytest.raises(ValueError):
            ExitModel(**bad)
    ExitModel(trail_atr=None, target_r=None)


# ── overlap and cooldown ───────────────────────────────────────────────────────

def _stopped_at_5():
    rows = [FLAT] * 60
    rows[5] = (10.0, 10.1, 7.5, 10.0)
    return _ind(rows)


def test_signals_inside_a_trade_or_its_cooldown_are_skipped():
    df = _stopped_at_5()
    sigs = [0, 1, 2, 10, 24, 30]
    cool = simulate(df, sigs, BASE, cooldown=20)
    assert list(cool["signal_i"]) == [0, 30]               # trade 0 exits on bar 5, eligible again from bar 25
    assert cool.iloc[0]["exit_i"] == 5
    none = simulate(df, sigs, BASE, cooldown=0)
    assert list(none["signal_i"]) == [0, 10]               # the bar-10 trade then runs to the end of the data
    assert list(simulate(df, [0, 5], BASE, cooldown=0)["signal_i"]) == [0, 5]   # a signal ON the exit bar is eligible
    assert list(simulate(df, [0, 24, 25], BASE, cooldown=20)["signal_i"]) == [0, 25]


def test_duplicate_and_unsorted_signals_are_handled():
    df = _stopped_at_5()
    a = simulate(df, [30, 0, 0, 10, 30], BASE, cooldown=0)
    b = simulate(df, [0, 10, 30], BASE, cooldown=0)
    pd.testing.assert_frame_equal(a, b)


# ── simulate_each ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("model", [ExitModel(2.0, 3.0, None, 120, 0.01),
                                   ExitModel(1.5, 2.5, 3.0, 60, 0.0),
                                   ExitModel(2.0, None, 1.0, 20, 0.01)])
def test_simulate_each_equals_simulate_one_trade_at_a_time(model):
    df = _walk_prices()
    n = len(df)
    starts = np.random.default_rng(11).choice(n, 40, replace=False)
    starts = np.union1d(starts, [0, 3, n - 2, n - 1])      # include bars without ATR and the last bar
    mask = np.zeros(n, dtype=bool)
    mask[starts] = True
    each = simulate_each(df, mask, model)
    assert each["valid"].sum() > 25
    assert not each["valid"][0] and not each["valid"][n - 1]
    assert not each["valid"][~mask].any()
    for s in starts:
        one = simulate(df, [s], model, cooldown=0)
        if not each["valid"][s]:
            assert len(one) == 0
            assert np.isnan(each["r_gross"][s]) and np.isnan(each["inv_risk"][s])
            continue
        assert len(one) == 1
        row = one.iloc[0]
        assert each["r_gross"][s] == pytest.approx(row["r_gross"])
        assert each["exit_i"][s] == row["exit_i"]
        assert REASONS[each["reason"][s]] == row["reason"]
        assert each["inv_risk"][s] == pytest.approx(row["entry"] / row["risk"])
        assert model.cost_pct * each["inv_risk"][s] == pytest.approx(row["cost_r"])


def test_simulate_each_trades_are_independent_even_when_they_overlap():
    df = _stopped_at_5()
    each = simulate_each(df, np.isin(np.arange(60), [0, 1, 2, 10]), BASE)
    assert list(np.flatnonzero(each["valid"])) == [0, 1, 2, 10]
    assert list(each["exit_i"][[0, 1, 2, 10]]) == [5, 5, 5, 59]
    with pytest.raises(ValueError, match="shape"):
        simulate_each(df, np.ones(10, dtype=bool), BASE)


# ── reasons ────────────────────────────────────────────────────────────────────

def test_every_reason_is_a_known_reason_and_the_columns_are_complete():
    seen = set()
    for jumps, model in [(False, ExitModel(2.0, 3.0, None, 120, 0.01)), (False, ExitModel(2.0, None, 2.0, 15, 0.01)),
                         (True, ExitModel(1.0, 2.0, 3.0, 30, 0.01)), (True, ExitModel(2.0, None, None, 40, 0.01))]:
        df = _walk_prices(1200, seed=9, jumps=jumps)
        out = simulate(df, np.arange(20, len(df) - 1, 7), model, cooldown=0, symbol="ZZ")
        assert len(out) > 30
        assert tuple(out.columns) == TRADE_COLUMNS
        assert set(out["reason"]) <= set(REASONS)
        assert (out["symbol"] == "ZZ").all()
        assert np.allclose(out["r_net"], out["r_gross"] - out["cost_r"])
        assert (out["bars"] == out["exit_i"] - out["entry_i"]).all()
        assert (out["mfe_r"] >= out["r_gross"] - 1e-12).all()      # the best price is never below the exit
        seen |= set(out["reason"])
    assert seen == set(REASONS), f"the sweep should exercise every exit, saw only {sorted(seen)}"


# ── no look-ahead ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cut", [200, 500, 800])
def test_a_closed_trade_is_identical_with_or_without_later_bars(cut):
    df = _walk_prices(1000, seed=3, jumps=True)
    model = ExitModel(1.5, 2.5, 4.0, 40, 0.01)
    sigs = np.arange(20, len(df) - 1, 5)
    full = simulate(df, sigs, model, cooldown=3)
    part = simulate(df.iloc[:cut], sigs[sigs < cut], model, cooldown=3)
    done = full[full["exit_i"] < cut - 1].reset_index(drop=True)
    assert len(done) > 3
    pd.testing.assert_frame_equal(done, part.iloc[:len(done)].reset_index(drop=True))


# ── jit and pure python agree ──────────────────────────────────────────────────

def _frames_for_parity():
    df = _walk_prices(900, seed=21, jumps=True)
    model = ExitModel(1.5, 2.5, 4.0, 40, 0.01)
    sigs = np.arange(20, len(df) - 1, 3)
    mask = np.zeros(len(df), dtype=bool)
    mask[sigs] = True
    return df, model, sigs, mask


def _compute(mod, df, model, sigs, mask):
    return (mod.simulate(df, sigs, model, cooldown=5, symbol="Q"), mod.simulate_each(df, mask, model))


def _assert_same(a, b):
    pd.testing.assert_frame_equal(a[0], b[0], check_exact=True)
    assert a[1].keys() == b[1].keys()
    for k in a[1]:
        np.testing.assert_array_equal(a[1][k], b[1][k])


def test_the_pure_python_kernels_give_exactly_the_jitted_results(monkeypatch):
    if not ts.JIT:
        pytest.skip("numba is not installed: the pure-Python kernels are the only path")
    df, model, sigs, mask = _frames_for_parity()
    jitted = _compute(ts, df, model, sigs, mask)
    assert len(jitted[0]) > 40
    monkeypatch.setattr(ts, "_walk", ts._walk.py_func)             # _run_* look _walk up in module globals
    monkeypatch.setattr(ts, "_run_signals", ts._run_signals.py_func)
    monkeypatch.setattr(ts, "_run_each", ts._run_each.py_func)
    _assert_same(jitted, _compute(ts, df, model, sigs, mask))


def test_the_module_still_works_when_numba_cannot_be_imported(monkeypatch):
    """Load a second copy of the module with numba blocked: the except-branch of the import."""
    monkeypatch.setitem(sys.modules, "numba", None)               # makes `from numba import njit` raise
    spec = importlib.util.spec_from_file_location("trade_sim_nojit", Path(ts.__file__))
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "trade_sim_nojit", mod)
    spec.loader.exec_module(mod)
    assert mod.JIT is False
    df, model, sigs, mask = _frames_for_parity()
    _assert_same(_compute(ts, df, model, sigs, mask), _compute(mod, df, model, sigs, mask))
