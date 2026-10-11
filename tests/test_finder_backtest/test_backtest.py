"""Tests for strategy_validation.finder_backtest.

Three groups: (1) the features are correct and use no future data, (2) the order
model does what its docstring says, on bars small enough to check by hand,
(3) the portfolio arithmetic. A backtest that is wrong in the optimistic
direction looks exactly like a good strategy, so each rule is pinned.
"""
import numpy as np
import pandas as pd
import pytest

from strategy_validation import finder_backtest as fb
from strategy_validation.finder_backtest import (
    ExecParams, Features, Rule, SignalSet, Trade, build_portfolio, build_signals,
    compute_features, period_stats, rolling_percentile_rank, simulate_symbol)


def _walk(n=700, seed=1, start=50.0):
    rng = np.random.default_rng(seed)
    close = start * np.exp(np.cumsum(rng.normal(0.0004, 0.015, n)))
    open_ = close * (1 + rng.normal(0, 0.004, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, n)))
    vol = rng.integers(200_000, 2_000_000, n).astype(float)
    idx = pd.bdate_range("2019-01-01", periods=n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx)


# ── (1) features ──────────────────────────────────────────────────────────────

def test_percentile_rank_matches_the_engine():
    from volatility_engine.regime import VolatilityRegimeEngine
    rng = np.random.default_rng(3)
    x = rng.normal(size=600)
    x[::37] = np.nan                                    # the engine tolerates gaps; so must the copy
    ref = VolatilityRegimeEngine()._rolling_percentile(pd.Series(x), 252).values
    mine = rolling_percentile_rank(x, 252)
    both = ~np.isnan(ref) & ~np.isnan(mine)
    assert both.sum() > 300
    assert np.array_equal(np.isnan(ref), np.isnan(mine))
    assert np.allclose(ref[both], mine[both], atol=1e-9)


@pytest.mark.parametrize("cut", [330, 450, 620])
def test_features_and_signals_at_t_do_not_depend_on_the_future(cut):
    df = _walk(700)
    full = compute_features(df)
    part = compute_features(df.iloc[:cut + 1])
    rule = Rule(min_volume=0, max_gap_pct=100.0)
    sf, sp = build_signals(full, rule), build_signals(part, rule)
    for name in ("ma50", "ma200", "score", "range_high", "range_low", "avg_vol20", "gap_pct"):
        a, b = getattr(full, name)[: cut + 1], getattr(part, name)
        assert np.allclose(a, b, equal_nan=True), f"{name} changed when future bars were removed"
    assert np.array_equal(full.is_comp[: cut + 1], part.is_comp)
    assert np.array_equal(sf.signal[: cut + 1], sp.signal)
    assert np.allclose(sf.entry[: cut + 1], sp.entry, equal_nan=True)


def test_signal_requires_every_condition():
    df = _walk(700, seed=5)
    f = compute_features(df)
    loose = build_signals(f, Rule(min_volume=0, max_gap_pct=1e9))
    assert loose.signal.sum() > 0, "synthetic data should produce some signals"
    # every condition the Finder checks must be able to veto on its own
    assert build_signals(f, Rule(min_volume=1e12, max_gap_pct=1e9)).signal.sum() == 0     # liquidity
    assert build_signals(f, Rule(min_volume=0, max_gap_pct=-1.0)).signal.sum() == 0       # gaps
    assert build_signals(f, Rule(min_price=1e9, min_volume=0, max_gap_pct=1e9)).signal.sum() == 0   # price band
    assert build_signals(f, Rule(min_volume=0, max_gap_pct=1e9, min_compression_score=101)).signal.sum() == 0
    s = loose
    on = s.signal
    assert np.all(s.entry[on] > s.stop[on])
    assert np.allclose(s.target[on] - s.entry[on], s.entry[on] - s.stop[on])   # target = one range-height up


def test_too_little_history_produces_no_features():
    assert compute_features(_walk(fb.MIN_ROWS - 1)) is None


# ── (2) the order model, on hand-built bars ───────────────────────────────────

def _ss(o, h, l, c, sig_at=0, E=100.0, S=90.0, T=110.0, score=80.0, drivers=None):
    n = len(c)
    z = np.zeros(n)
    f = Features(dates=pd.bdate_range("2024-01-01", periods=n).values.astype("datetime64[ns]"),
                 open=np.array(o, float), high=np.array(h, float), low=np.array(l, float),
                 close=np.array(c, float), volume=np.full(n, 1e6), ma50=z, ma200=z, avg_vol20=z,
                 gap_pct=z, valid=np.ones(n, bool), is_comp=np.ones(n, bool), score=np.full(n, score),
                 range_high=z, range_low=z)
    sig = np.zeros(n, bool)
    for s in ([sig_at] if np.isscalar(sig_at) else sig_at):
        sig[s] = True
    drv = np.arange(n) if drivers is None else np.asarray(drivers)
    return SignalSet(f, sig, np.full(n, E), np.full(n, S), np.full(n, T), np.full(n, score), drv)


EX = ExecParams(valid_days=5, max_hold=5)


def test_breakout_fills_at_entry_and_exits_at_target():
    #       0      1      2      3
    ss = _ss(o=[99, 99, 104, 108], h=[99, 101, 109, 111], l=[98, 98, 103, 107], c=[99, 100, 108, 110])
    (t,) = simulate_symbol("X", ss, EX)
    assert (t.fill_i, t.fill_px) == (1, 100.0)           # opened below entry, high reached it
    assert (t.exit_i, t.exit_px, t.reason) == (3, 110.0, "target")
    assert t.gross == pytest.approx(0.10)
    assert np.prod(1 + t.ret.astype(float)) - 1 == pytest.approx(t.gross, abs=1e-6)  # path compounds to the trade


def test_gap_up_open_pays_the_gap_not_the_entry():
    ss = _ss(o=[99, 103, 104], h=[99, 105, 111], l=[98, 102, 103], c=[99, 104, 110])
    (t,) = simulate_symbol("X", ss, EX)
    assert t.fill_px == 103.0                             # filled at the open, 3% above the 100 level


def test_gap_past_the_target_is_no_trade():
    ss = _ss(o=[99, 112], h=[99, 115], l=[98, 111], c=[99, 114])
    assert simulate_symbol("X", ss, EX) == []


def test_stop_beats_target_when_one_bar_could_hit_both():
    ss = _ss(o=[99, 100, 100], h=[99, 101, 112], l=[98, 99, 88], c=[99, 100, 100])
    (t,) = simulate_symbol("X", ss, EX)
    assert (t.reason, t.exit_px) == ("stop", 90.0)        # the pessimistic reading


def test_gap_down_through_the_stop_exits_at_the_open_not_the_stop():
    ss = _ss(o=[99, 100, 85], h=[99, 101, 86], l=[98, 99, 84], c=[99, 100, 85])
    (t,) = simulate_symbol("X", ss, EX)
    assert (t.reason, t.exit_px) == ("stop_gap", 85.0)    # you do not get the stop price on a gap
    assert t.gross == pytest.approx(-0.15)


def test_time_stop_exits_at_the_close():
    n = 12
    ss = _ss(o=[99] + [100] * 11, h=[99] + [101] * 11, l=[98] + [99] * 11, c=[99] + [100] * 11)
    (t,) = simulate_symbol("X", ss, ExecParams(valid_days=5, max_hold=4))
    assert (t.fill_i, t.reason, t.exit_i) == (1, "time", 5)


def test_order_expires_unfilled():
    ss = _ss(o=[99] * 8, h=[99.5] * 8, l=[98] * 8, c=[99] * 8)
    assert simulate_symbol("X", ss, ExecParams(valid_days=3, max_hold=5)) == []


def test_order_cancelled_if_price_closes_below_the_stop_first():
    ss = _ss(o=[99, 95, 88, 100, 101], h=[99, 96, 89, 101, 102], l=[98, 89, 87, 99, 100], c=[99, 90, 88, 100, 101])
    assert simulate_symbol("X", ss, EX) == []              # closed at 88 < stop 90 before ever reaching 100


def test_untriggered_only_skips_setups_already_past_entry():
    ss = _ss(o=[105, 106, 108], h=[106, 107, 111], l=[104, 105, 107], c=[105, 106, 110])
    assert simulate_symbol("X", ss, ExecParams(5, 5, untriggered_only=True)) == []
    assert len(simulate_symbol("X", ss, ExecParams(5, 5, untriggered_only=False))) == 1


def test_one_order_per_setup_no_reentry_on_stale_levels():
    # signal on bar 0 and again on bar 4 with the SAME driver; the first trade is stopped out
    o = [99, 100, 85, 99, 100, 101, 102]
    h = [99, 101, 86, 99.5, 101, 102, 103]
    l = [98, 99, 84, 98, 99, 100, 101]
    c = [99, 100, 85, 99, 100, 101, 102]
    ss = _ss(o, h, l, c, sig_at=[0, 3], drivers=[7] * 7)
    assert len(simulate_symbol("X", ss, EX)) == 1
    ss2 = _ss(o, h, l, c, sig_at=[0, 3], drivers=[1, 2, 3, 4, 5, 6, 7])   # a NEW setup may trade
    assert len(simulate_symbol("X", ss2, EX)) == 2


def test_positions_in_one_symbol_never_overlap():
    rng = np.random.default_rng(9)
    df = _walk(900, seed=11)
    ss = build_signals(compute_features(df), Rule(min_volume=0, max_gap_pct=1e9))
    trades = simulate_symbol("X", ss, ExecParams())
    assert len(trades) > 3
    for a, b in zip(trades, trades[1:]):
        assert b.fill_i >= a.exit_i


# ── (3) portfolio arithmetic ──────────────────────────────────────────────────

def _trade(sym, start, ret, score=50.0):
    d = pd.bdate_range("2024-01-01", periods=60).values.astype("datetime64[ns]")
    r = np.asarray(ret, np.float32)
    return Trade(sym, score, 0, 0, len(r) - 1, 100.0, 100.0 * float(np.prod(1 + r.astype(float))), "x",
                 d[start:start + len(r)], r)


CAL = pd.bdate_range("2024-01-01", periods=60).values.astype("datetime64[ns]")


def test_cost_is_charged_once_on_the_exit_bar():
    daily, _ = build_portfolio([_trade("A", 2, [0.01, 0.01, 0.01])], CAL, cost_rt=0.005, max_positions=None)
    assert daily[2] == pytest.approx(0.01) and daily[3] == pytest.approx(0.01)
    assert daily[4] == pytest.approx(0.01 - 0.005)


def test_equal_weight_averages_what_is_open():
    daily, _ = build_portfolio([_trade("A", 2, [0.02, 0.02]), _trade("B", 2, [0.04, 0.04])], CAL, 0.0, None)
    assert daily[2] == pytest.approx(0.03)


def test_slot_cap_takes_the_best_ranked_signals_first():
    ts = [_trade(s, 2, [0.01, 0.01], score=sc) for s, sc in (("lo", 10), ("mid", 50), ("hi", 90))]
    _, taken = build_portfolio(ts, CAL, 0.0, max_positions=2)
    assert {t.symbol for t in taken} == {"hi", "mid"}


def test_slot_cap_frees_a_slot_after_the_exit():
    ts = [_trade("A", 2, [0.01, 0.01]), _trade("B", 2, [0.01, 0.01]), _trade("C", 6, [0.01])]
    _, taken = build_portfolio(ts, CAL, 0.0, max_positions=2)
    assert {t.symbol for t in taken} == {"A", "B", "C"}     # C starts after A and B have exited


def test_capped_portfolio_holds_cash_in_empty_slots():
    daily, _ = build_portfolio([_trade("A", 2, [0.10])], CAL, 0.0, max_positions=10)
    assert daily[2] == pytest.approx(0.01)                   # 10% return on a 1/10 slot


def test_period_stats_split_and_drawdown():
    daily = np.zeros(len(CAL))
    daily[5], daily[6], daily[7] = 0.10, -0.20, 0.05
    s = period_stats(daily, CAL, [], 0.0, "2024-01-01")
    assert s["max_drawdown_pct"] == pytest.approx(20.0)
    assert s["n_obs"] == len(CAL)
    later = period_stats(daily, CAL, [], 0.0, str(CAL[6])[:10])
    assert later["n_obs"] < s["n_obs"]


def test_optimistic_bound_credits_the_target_on_a_both_hit_bar_but_the_default_does_not():
    ss = _ss(o=[99, 100, 100], h=[99, 101, 112], l=[98, 99, 88], c=[99, 100, 100])
    (pess,) = simulate_symbol("X", ss, EX)
    (opt,) = simulate_symbol("X", ss, ExecParams(valid_days=5, max_hold=5, stop_first=False))
    assert (pess.reason, opt.reason) == ("stop", "target")
    assert opt.gross > 0 > pess.gross


def test_optimistic_bound_is_never_worse_than_the_pessimistic_default():
    df = _walk(1200, seed=21)
    ss = build_signals(compute_features(df), Rule(min_volume=0, max_gap_pct=1e9))
    pess = simulate_symbol("X", ss, ExecParams())
    opt = simulate_symbol("X", ss, ExecParams(stop_first=False))
    assert len(pess) > 5
    assert sum(t.gross for t in opt) >= sum(t.gross for t in pess) - 1e-9
