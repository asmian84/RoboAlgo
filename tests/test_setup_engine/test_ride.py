"""setup_engine.ride: a "let it ride" backtest must fill where a real order could fill, and never peek.

The failure being guarded against: a ride simulator that flatters the signal. Each test below is
one way that happens (entering at the signal close instead of the next open, filling a rule that
fired on a close AT that close, letting a close rule outrank the stop on the day the stop was
touched, measuring the trailing exit from a close the trade never held, evaluating a monthly rule
on a month that ended before the entry, forgetting the cost or the time cap, sizing the stop from
the signal price), each built on a hand-made price path whose answer is known exactly.

The jitted kernel and the pure-Python fallback must agree, and `ride()` is also compared with a
plain loop-based reference implementation (`_reference_ride`, written below from the rule text and
sharing no code or lookup arrays with ride.py) on several hundred seeded random-walk rides.
"""
import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from setup_engine import ride
from setup_engine.ride import (RIDE_COLUMNS, RIDE_R1, RIDE_R2, RIDE_R3, REASONS, RideExit, forward_max_close,
                               horizon_return, prepare, ride_many)

START = "2020-01-06"   # a Monday: week w of a flat business-day frame occupies positions 5w .. 5w+4


# ── helpers ────────────────────────────────────────────────────────────────────

def _flat(n, px=10.0):
    idx = pd.bdate_range(START, periods=n)
    return pd.DataFrame({k: np.full(n, float(px)) for k in ("open", "high", "low", "close")}, index=idx)


def _weekly_path(closes):
    """Five identical bars (o=h=l=c) per weekly close."""
    px = np.repeat(np.asarray(closes, dtype=float), 5)
    return pd.DataFrame({k: px.copy() for k in ("open", "high", "low", "close")},
                        index=pd.bdate_range(START, periods=len(px)))


def _bar(d, pos, o, h, l, c):
    assert h >= max(o, c) and l <= min(o, c), "test bug: an invalid bar"
    for col, v in zip(("open", "high", "low", "close"), (o, h, l, c)):
        d.iloc[pos, d.columns.get_loc(col)] = float(v)


def _monday(w):
    return 5 * w


def _friday(w):
    return 5 * w + 4


def _bars(daily):
    """Weekly / monthly frames (close, last_day) with the calendar label rules, as mtf.weekly_bars / monthly_bars.

    Week label = the Friday ending the week; month label = the last WEEKDAY of the month. Only
    bars whose label is <= the last daily date are kept (a bar still in progress is not complete).
    """
    idx = daily.index
    wl = idx.to_period("W-FRI").end_time.normalize()
    me = idx.to_period("M").end_time.normalize()
    ml = me - pd.to_timedelta(np.maximum(me.weekday - 4, 0), unit="D")
    days = pd.Series(idx, index=idx)
    out = []
    for lab, name in ((wl, "week"), (ml, "month")):
        b = pd.DataFrame({"close": daily["close"].groupby(lab).last(), "last_day": days.groupby(lab).max()})
        b.index.name = name
        out.append(b[b.index <= idx[-1]])
    return tuple(out)


def _rd(daily, **kw):
    return prepare(daily, *_bars(daily), **kw)


def _same(a, b):
    return a == b or (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b))


# ── T1 entry at the next open ──────────────────────────────────────────────────

def test_entry_is_the_next_daily_open_not_the_signal_close():
    d = _flat(200)
    sig = 20
    _bar(d, _friday(sig) + 1, 10.5, 10.5, 10.0, 10.0)          # the day after the signal week opens at 10.5
    r = ride.ride(_rd(d), sig, replace(RIDE_R3, max_weeks=999))
    assert r["entry_i"] == _friday(sig) + 1
    assert r["entry"] == 10.5
    assert r["stop"] == pytest.approx(10.5 * 0.65)
    assert r["reason"] == "end"                                 # nothing fired: still open at the last bar
    assert r["exit_i"] == 199 and r["exit"] == 10.0              # exit = the last close
    assert r["ret_gross"] == pytest.approx(10.0 / 10.5 - 1)
    assert r["signal_date"] == d.index[_friday(sig)] and r["entry_date"] == d.index[_friday(sig) + 1]


# ── T2 a close rule fills at the NEXT open ─────────────────────────────────────

def test_r1_weekly_close_below_sma30_fills_at_the_next_open_not_the_close():
    d = _flat(200)
    _bar(d, _friday(37), 10, 10, 9, 9)                          # week 37 closes at 9: below the 30-week SMA
    _bar(d, _monday(38), 8.8, 8.8, 8.8, 8.8)                    # and Monday opens at 8.8
    r = ride.ride(_rd(d), 34, RIDE_R1)
    assert r["reason"] == "r1_weekly_sma30"
    assert r["exit"] == 8.8                                      # the open, not the 9 close that triggered it
    assert r["exit_i"] == _monday(38) and r["exit_date"] == d.index[_monday(38)]
    assert r["ret_gross"] == pytest.approx(8.8 / 10 - 1)


def test_r1_does_not_fire_while_the_weekly_close_equals_the_sma():
    r = ride.ride(_rd(_flat(200)), 34, RIDE_R1)                  # closes == SMA30 forever
    assert r["reason"] == "end"


# ── T3 stop and gap-through-stop ───────────────────────────────────────────────

def test_stop_fills_at_the_stop_and_is_sized_from_the_entry_open_not_the_signal_close():
    d = _flat(100)
    _bar(d, _friday(10), 12, 12, 12, 12)                        # signal week closes at 12 ...
    # ... and the next open is 10 (gap down): the stop is 10 * 0.65 = 6.5, not 12 * 0.65 = 7.8
    _bar(d, _monday(14) + 2, 9, 9, 6.4, 7)                      # a day that trades down through 6.5
    r = ride.ride(_rd(d), 10, RIDE_R3)
    assert r["entry"] == 10.0 and r["stop"] == pytest.approx(6.5)
    assert r["reason"] == "stop" and r["exit"] == pytest.approx(6.5)
    assert r["exit_i"] == _monday(14) + 2
    assert r["ret_gross"] == pytest.approx(-0.35)


def test_a_day_that_opens_below_the_stop_fills_at_the_open():
    d = _flat(100)
    _bar(d, _monday(14) + 2, 6.0, 6.2, 5.9, 6.0)
    r = ride.ride(_rd(d), 10, RIDE_R3)
    assert r["reason"] == "gap_stop" and r["exit"] == 6.0       # worse than the 6.5 stop: the gap is paid
    assert r["ret_gross"] == pytest.approx(-0.40)


def test_the_entry_bar_is_tested_like_any_other_bar():
    d = _flat(100)
    entry = _friday(10) + 1
    _bar(d, entry, 10, 10, 6.0, 8)
    r = ride.ride(_rd(d), 10, RIDE_R3)
    assert r["reason"] == "stop" and r["exit_i"] == entry == r["entry_i"] and r["exit"] == pytest.approx(6.5)


# ── T4 the stop outranks a close rule on the same day ──────────────────────────

def test_stop_is_checked_before_the_close_rule_on_the_same_day():
    d = _flat(200)
    _bar(d, _friday(36), 10, 10, 6.4, 6.6)                      # touches the stop AND closes below the SMA30
    r = ride.ride(_rd(d), 34, RIDE_R1)
    assert r["reason"] == "stop" and r["exit"] == pytest.approx(6.5)
    assert r["exit_i"] == _friday(36)                           # same day, not the next open


def test_a_decision_taken_at_yesterdays_close_fills_at_todays_open_even_below_the_stop():
    d = _flat(200)
    _bar(d, _friday(36), 10, 10, 9, 9)                          # R1 fires at this close
    _bar(d, _monday(37), 6.0, 6.0, 6.0, 6.0)                    # and the next open is below the 6.5 stop
    r = ride.ride(_rd(d), 34, RIDE_R1)
    assert r["exit"] == 6.0 and r["exit_i"] == _monday(37)
    assert r["reason"] == "r1_weekly_sma30"                     # the exit was already decided; it fills at the open


# ── T5 R3 ignores closes from before the entry ─────────────────────────────────

def test_r3_trail_ignores_the_signal_weeks_close_and_everything_before_the_entry():
    closes = [10] * 5 + [30] + [10, 11, 12, 9, 8.3, 8.0]
    d = _weekly_path(closes)                                     # signal week 5 closes at 30; entry opens at 10
    r = ride.ride(_rd(d), 5, RIDE_R3)
    assert r["entry"] == 10.0
    # 9 is above 0.7 * 12 = 8.4: still held. 8.3 is below it: decided, filled at the next open (8.0)
    assert r["reason"] == "r3_trail30"
    assert r["exit_i"] == _monday(11) and r["exit"] == 8.0
    assert r["mfe_close"] == pytest.approx(0.2)                  # best weekly close HELD was 12, not 30
    assert r["mfe"] == pytest.approx(0.2)


def test_r3_does_not_fire_on_a_close_above_the_trail_line():
    closes = [10] * 5 + [30] + [10, 11, 12, 9]
    r = ride.ride(_rd(_weekly_path(closes)), 5, RIDE_R3)
    assert r["reason"] == "end" and r["exit"] == 9.0


# ── T6 the time cap ────────────────────────────────────────────────────────────

def test_time_cap_exits_at_the_open_after_the_nth_weekly_close():
    d = _weekly_path([10 + w for w in range(40)])                # rising: no trail exit ever
    r = ride.ride(_rd(d), 5, RideExit(r3=True, max_weeks=5))
    # weekly closes after the signal week: weeks 6..10. The 5th (week 10) decides; fill at week 11's open
    assert r["reason"] == "time"
    assert r["exit_i"] == _monday(11) and r["exit"] == 21.0
    assert r["entry"] == 16.0
    assert r["weeks_held"] == 5.0


def test_time_cap_applies_with_no_close_rule_switched_on():
    d = _weekly_path([10 + w for w in range(40)])
    r = ride.ride(_rd(d), 5, RideExit(max_weeks=3))
    assert r["reason"] == "time" and r["exit_i"] == _monday(9)


# ── T7 cost ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cost", [0.0, 0.01, 0.025])
def test_cost_is_a_flat_round_trip_fraction_off_the_gross_return(cost):
    d = _weekly_path([10] * 10 + [12] * 10)                      # enter at 10, still open at 12
    r = ride.ride(_rd(d), 4, replace(RIDE_R3, cost_rt=cost))
    assert r["ret_gross"] == pytest.approx(0.2)
    assert r["ret_net"] == pytest.approx(0.2 - cost)


def test_default_cost_is_one_percent():
    r = ride.ride(_rd(_weekly_path([10] * 10 + [12] * 10)), 4, RIDE_R3)
    assert r["ret_net"] == pytest.approx(r["ret_gross"] - 0.01)


# ── T8 R2 monthly rule ─────────────────────────────────────────────────────────

def _month_frame():
    d = _flat(420)
    wk, mo = _bars(d)
    return d, wk, mo


def test_r2_monthly_close_below_sma10_after_the_entry_exits_at_the_next_open():
    d, wk, mo = _month_frame()
    end14 = mo["last_day"].iloc[14]                              # 15th month-end: 14 months of 10 came before
    p = d.index.get_loc(end14)
    _bar(d, p, 10, 10, 9, 9)                                     # month-end close 9 < SMA10 = 9.9
    _bar(d, p + 1, 8.8, 8.8, 8.8, 8.8)
    wk, mo = _bars(d)
    sig = int((wk["last_day"] <= mo["last_day"].iloc[13]).sum()) - 1   # last week before the previous month-end
    assert wk["last_day"].iloc[sig] < end14
    r = ride.ride(prepare(d, wk, mo), sig, RIDE_R2)
    assert r["reason"] == "r2_monthly_sma10"
    assert r["exit_i"] == p + 1 and r["exit"] == 8.8             # the open after the month-end, not its 9 close


def test_r2_ignores_a_month_that_ended_before_the_entry_day():
    d, wk, mo = _month_frame()
    end14 = mo["last_day"].iloc[14]
    p = d.index.get_loc(end14)
    _bar(d, p, 10, 10, 9, 9)                                     # this month-end IS below its SMA10 ...
    wk, mo = _bars(d)
    sig = int((wk["last_day"] < end14).sum())                    # ... but the signal week ends on or after it
    assert wk["last_day"].iloc[sig] >= end14
    r = ride.ride(prepare(d, wk, mo), sig, RIDE_R2)
    assert r["entry_i"] > p
    assert r["reason"] == "end" and r["exit"] == 10.0            # later months (10 > SMA10 = 9.9) never trigger


def test_a_weekly_rule_outranks_the_monthly_rule_decided_on_the_same_close():
    d, wk, mo = _month_frame()
    p = d.index.get_loc(mo["last_day"].iloc[9])                  # 2020-10-30: a Friday AND a month-end
    assert d.index[p].dayofweek == 4
    _bar(d, p, 10, 10, 9, 9)                                     # below the 30-week SMA and below the 10-month SMA
    _bar(d, p + 1, 8.8, 8.8, 8.8, 8.8)
    wk, mo = _bars(d)
    r = ride.ride(prepare(d, wk, mo), 38, RideExit(r1=True, r2=True))
    assert r["reason"] == "r1_weekly_sma30" and r["exit_i"] == p + 1 and r["exit"] == 8.8
    r2 = ride.ride(prepare(d, wk, mo), 38, RIDE_R2)              # with only R2 on, the monthly rule is what fires
    assert r2["reason"] == "r2_monthly_sma10" and r2["exit_i"] == p + 1


# ── mfe: which bars count ──────────────────────────────────────────────────────

def test_mfe_includes_the_high_of_the_bar_that_hits_the_stop_but_not_the_bar_we_sell_the_open_of():
    d = _flat(100)
    _bar(d, _monday(14) + 2, 10, 11, 6.4, 7)                    # trades up to 11, then down through the 6.5 stop
    r = ride.ride(_rd(d), 10, RIDE_R3)
    assert r["reason"] == "stop" and r["mfe"] == pytest.approx(0.1)

    d = _flat(200)
    _bar(d, _friday(36), 10, 10, 9, 9)                          # R1 decided at this close
    _bar(d, _monday(37), 8.8, 20, 8.8, 15)                      # we sell at the 8.8 open: the 20 high is never reached
    r = ride.ride(_rd(d), 34, RIDE_R1)
    assert r["reason"] == "r1_weekly_sma30" and r["exit"] == 8.8
    assert r["mfe"] == pytest.approx(0.0)
    assert r["mfe_close"] == pytest.approx(0.0)                 # weekly closes held: 10, 10, 9 -> best 10 = the entry


# ── T9 no next bar ─────────────────────────────────────────────────────────────

def test_signal_on_the_last_week_has_nothing_to_enter_on():
    d = _flat(60)
    rd = _rd(d)
    last = len(rd.w_pos) - 1
    assert rd.w_pos[last] == len(d) - 1
    assert ride.ride(rd, last, RIDE_R3) is None
    out = ride_many(rd, [last], RIDE_R3)
    assert out.empty and list(out.columns) == list(RIDE_COLUMNS)
    both = ride_many(rd, [3, last], RIDE_R3)
    assert len(both) == 1 and both["signal_week"].iloc[0] == rd.week_labels[3]


# ── T10 holiday week ───────────────────────────────────────────────────────────

def test_a_holiday_week_is_decided_at_the_thursday_close_and_fills_at_the_next_monday_open():
    d = _flat(200)
    d = d.drop(d.index[_friday(36)])                             # no Friday in week 36
    thu = _friday(36) - 1
    _bar(d, thu, 10, 10, 9, 9)
    mon = thu + 1                                                # the Friday is gone: the next bar is Monday
    _bar(d, mon, 8.8, 8.8, 8.8, 8.8)
    wk, mo = _bars(d)
    assert wk["last_day"].iloc[36].dayofweek == 3                # Thursday
    r = ride.ride(prepare(d, wk, mo), 34, RIDE_R1)
    assert r["reason"] == "r1_weekly_sma30"
    assert d.index[r["exit_i"]].dayofweek == 0 and r["exit_i"] == mon
    assert r["exit"] == 8.8


# ── T11 fixed horizons and forward max close ───────────────────────────────────

def test_horizon_return_and_forward_max_close_by_hand_and_nan_when_incomplete():
    closes = [10, 10, 10, 12, 15, 11, 14, 13]                    # 8 weeks
    rd = _rd(_weekly_path(closes))
    # signal week 1: entry at week 2's open (10). horizon 3: sell at the open after week 4 = week 5's open (11)
    assert horizon_return(rd, 1, 3) == pytest.approx(11 / 10 - 1 - 0.01)
    assert horizon_return(rd, 1, 3, cost_rt=0.0) == pytest.approx(0.1)
    # week 1 + 5 = 6 -> sell at week 7's open (13)
    assert horizon_return(rd, 1, 5) == pytest.approx(13 / 10 - 1 - 0.01)
    # signal week 1 + 6 = 7 is the last week: no bar after it to sell on -> NaN
    assert math.isnan(horizon_return(rd, 1, 6))
    assert math.isnan(horizon_return(rd, 1, 7))
    # forward max close: weeks 2..4 -> max(10, 12, 15) / 10 - 1
    assert forward_max_close(rd, 1, 3) == pytest.approx(0.5)
    # weeks 2..7 -> max(.., 15, .., 14, 13) = 15
    assert forward_max_close(rd, 1, 6) == pytest.approx(0.5)
    # full windows only: week 1 + 7 = 8 does not exist
    assert math.isnan(forward_max_close(rd, 1, 7))
    # the last complete week's close is known even though it cannot be sold on: horizon NaN, label not.
    # signal week 4: entry at week 5's open (11); weeks 5..7 close 11, 14, 13 -> max 14
    assert forward_max_close(rd, 4, 3) == pytest.approx(14 / 11 - 1)
    assert math.isnan(horizon_return(rd, 4, 3))


def test_horizon_arguments_are_validated():
    rd = _rd(_weekly_path([10] * 6))
    with pytest.raises(ValueError):
        horizon_return(rd, 1, 0)
    with pytest.raises(ValueError):
        forward_max_close(rd, 1, 0)
    with pytest.raises(ValueError):
        horizon_return(rd, 99, 1)


# ── input checks, ride_many ────────────────────────────────────────────────────

def test_prepare_rejects_a_last_day_that_is_not_a_daily_date():
    d = _flat(60)
    wk, mo = _bars(d)
    bad = wk.copy()
    bad.iloc[2, bad.columns.get_loc("last_day")] = pd.Timestamp("1999-01-01")
    with pytest.raises(ValueError, match="last_day"):
        prepare(d, bad, mo)
    with pytest.raises(ValueError, match="monthly"):
        prepare(d, wk, mo.assign(last_day=mo["last_day"] + pd.Timedelta(days=1)))
    with pytest.raises(ValueError):
        prepare(d.iloc[::-1], wk, mo)


def test_ride_many_is_ride_per_valid_week_and_columns_are_fixed():
    d = _weekly_path([10 + 0.1 * w for w in range(60)])
    rd = _rd(d)
    weeks = [3, 7, 20, len(rd.w_pos) - 1]
    out = ride_many(rd, weeks, RIDE_R3)
    assert list(out.columns) == list(RIDE_COLUMNS) and len(out) == 3
    for i, sw in enumerate(weeks[:3]):
        r = ride.ride(rd, sw, RIDE_R3)
        for col in RIDE_COLUMNS:
            assert _same(out[col].iloc[i], r[col]) or out[col].iloc[i] == pytest.approx(r[col]), col
    assert set(out["reason"]) <= set(REASONS)


def test_a_ride_exit_built_for_other_sma_windows_is_refused():
    rd = _rd(_flat(60))
    with pytest.raises(ValueError, match="SMA windows"):
        ride.ride(rd, 3, RideExit(r1=True, sma_w=20))


def test_constants():
    assert REASONS == ("stop", "gap_stop", "r1_weekly_sma30", "r2_monthly_sma10", "r3_trail30", "time", "end")
    assert RIDE_COLUMNS[0] == "signal_week" and RIDE_COLUMNS[-1] == "reason" and len(RIDE_COLUMNS) == 15
    assert (RIDE_R1.r1, RIDE_R1.r2, RIDE_R1.r3) == (True, False, False)
    assert (RIDE_R2.r1, RIDE_R2.r2, RIDE_R2.r3) == (False, True, False)
    assert (RIDE_R3.r1, RIDE_R3.r2, RIDE_R3.r3) == (False, False, True)
    e = RideExit()
    assert (e.trail_pct, e.stop_pct, e.max_weeks, e.cost_rt, e.sma_w, e.sma_m) == (0.30, 0.35, 156, 0.01, 30, 10)


# ── random walks: parity and the independent reference ─────────────────────────

def _random_daily(seed, n=1300, drop=0.02):
    """Fat-tailed random walk with overnight gaps, occasional big gaps, and random missing days (holidays)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2012-01-02", periods=n)
    keep = rng.random(n) > drop
    keep[0] = keep[-1] = True
    idx = idx[keep]
    m = len(idx)
    mu = rng.uniform(-0.0004, 0.0016)
    gap = rng.standard_t(3, m) * 0.008 + (rng.random(m) < 0.012) * rng.normal(0, 0.10, m)
    intra = rng.standard_t(4, m) * 0.012 + mu
    logc = np.log(40.0) + np.cumsum(gap + intra)
    c = np.exp(logc)
    o = np.exp(logc - intra)
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.007, m)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.007, m)))
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c}, index=idx)


def _kernel_args(rd, sw, ex):
    return (rd.o, rd.h, rd.l, rd.c, rd.wk_at, rd.w_close, rd.w_sma, rd.mo_at, rd.m_close, rd.m_sma,
            int(rd.w_pos[sw]), sw, bool(ex.r1), bool(ex.r2), bool(ex.r3),
            float(ex.trail_pct), float(ex.stop_pct), int(ex.max_weeks))


CONFIGS = [
    RIDE_R1, RIDE_R2, RIDE_R3,
    RideExit(r1=True, r3=True, stop_pct=0.15, trail_pct=0.20, max_weeks=26),
    RideExit(r2=True, r3=True, stop_pct=0.25, max_weeks=52),
]


def test_jit_kernel_and_pure_python_kernel_agree_on_random_walk_rides():
    py = getattr(ride._ride_kernel, "py_func", ride._ride_kernel)
    rng = np.random.default_rng(7)
    checked = 0
    for seed in range(10):
        rd = _rd(_random_daily(100 + seed))
        K = len(rd.w_pos)
        for sw in rng.choice(K - 1, 3, replace=False):
            for ex in (CONFIGS[0], CONFIGS[1], CONFIGS[2], CONFIGS[3 + int(sw) % 2]):
                a = ride._ride_kernel(*_kernel_args(rd, int(sw), ex))
                b = py(*_kernel_args(rd, int(sw), ex))
                assert tuple(a) == tuple(b), (seed, sw, ex)
                checked += 1
    assert checked >= 30


def _reference_ride(daily, weekly, monthly, sw, ex):
    """The ride rules, restated as a plain day-by-day loop from the rule text (no arrays shared with ride.py).

    Rule text: the signal is the close of weekly bar `sw` (its last_day); buy at the next day's
    open; the stop is 35% (stop_pct) under that open. Every day from the entry on: (1) if an exit
    was decided at yesterday's close, sell at today's open; (2) track the highest high; (3) if the
    open is at or under the stop, sell at the open, else if the low is at or under the stop, sell
    at the stop; (4) if a weekly bar completes today, record its close as held, then: sell
    tomorrow if (R1) the close is under the average of the last 30 weekly closes, else if (R3) it
    is at or under 70% of the highest weekly close held, else if the weeks since the signal week
    reach the cap; (5) if a month completes today and nothing is decided, sell tomorrow if (R2) its
    close is under the average of the last 10 monthly closes. Otherwise sell at the last close.
    """
    days = list(daily.index)
    day_no = {d: i for i, d in enumerate(days)}
    op, hi, lo, cl = (daily[c].tolist() for c in ("open", "high", "low", "close"))
    week_end = {d: k for k, d in enumerate(weekly["last_day"])}
    month_end = {d: k for k, d in enumerate(monthly["last_day"])}
    wclose, mclose = weekly["close"].tolist(), monthly["close"].tolist()

    first = day_no[weekly["last_day"].iloc[sw]] + 1
    if first >= len(days):
        return None
    entry = op[first]
    stop = entry * (1 - ex.stop_pct)
    decided = None
    best_high = -math.inf
    held = []                                   # weekly closes completed while in the trade

    def out(i, px, why):
        return {"entry_i": first, "exit_i": i, "exit": px, "reason": why, "entry": entry,
                "mfe": best_high / entry - 1, "mfe_close": (max(held) / entry - 1) if held else math.nan}

    for i in range(first, len(days)):
        if decided is not None:
            return out(i, op[i], decided)
        best_high = max(best_high, hi[i])
        if op[i] <= stop:
            return out(i, op[i], "gap_stop")
        if lo[i] <= stop:
            return out(i, stop, "stop")
        today = days[i]
        if today in week_end:
            k = week_end[today]
            held.append(wclose[k])
            if ex.r1 and k + 1 >= ex.sma_w and wclose[k] < sum(wclose[k + 1 - ex.sma_w: k + 1]) / ex.sma_w:
                decided = "r1_weekly_sma30"
            elif ex.r3 and wclose[k] <= (1 - ex.trail_pct) * max(held):
                decided = "r3_trail30"
            elif k - sw >= ex.max_weeks:
                decided = "time"
        if decided is None and ex.r2 and today in month_end:
            k = month_end[today]
            if k + 1 >= ex.sma_m and mclose[k] < sum(mclose[k + 1 - ex.sma_m: k + 1]) / ex.sma_m:
                decided = "r2_monthly_sma10"
    return out(len(days) - 1, cl[-1], "end")


def test_ride_matches_an_independent_loop_reference_on_seeded_random_walks():
    rng = np.random.default_rng(2026)
    n_rides = n_base = 0
    reasons = set()
    for seed in range(8):
        d = _random_daily(seed)
        wk, mo = _bars(d)
        rd = prepare(d, wk, mo)
        K = len(wk)
        weeks = sorted(set(rng.choice(K - 1, 12, replace=False).tolist()) | {K - 1})
        for sw in weeks:
            for ci, ex in enumerate(CONFIGS):
                ref = _reference_ride(d, wk, mo, sw, ex)
                got = ride.ride(rd, sw, ex)
                if ref is None:
                    assert got is None
                    continue
                assert got is not None
                ctx = (seed, sw, ci)
                assert got["entry_i"] == ref["entry_i"], ctx
                assert got["exit_i"] == ref["exit_i"], ctx
                assert got["reason"] == ref["reason"], ctx
                assert got["entry"] == ref["entry"], ctx
                assert got["exit"] == pytest.approx(ref["exit"], rel=1e-12), ctx
                assert got["ret_gross"] == pytest.approx(ref["exit"] / ref["entry"] - 1, rel=1e-12, abs=1e-15), ctx
                assert got["ret_net"] == pytest.approx(got["ret_gross"] - ex.cost_rt), ctx
                assert got["mfe"] == pytest.approx(ref["mfe"], rel=1e-12, abs=1e-15), ctx
                assert got["mfe_close"] == pytest.approx(ref["mfe_close"], rel=1e-12, abs=1e-15, nan_ok=True), ctx
                n_rides += 1
                n_base += ci < 3
                reasons.add(got["reason"])
    assert n_base >= 200 and n_rides >= 400
    assert reasons == set(REASONS), f"the random walks never produced: {set(REASONS) - reasons}"
