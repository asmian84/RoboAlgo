"""
Weekly "let it ride" simulator: buy a stock after a weekly signal and hold it until a
weekly/monthly rule says the trend is over, with a hard stop underneath.

setup_engine/trade_sim.py measures a screen with ATR stops on DAILY bars. This module answers a
different question: if you buy a stock the day after a weekly signal and hold it for months or
years, how much of the move do you keep under rules that only look at COMPLETED weekly or
monthly bars? It takes the daily bars plus the weekly and monthly bar tables (what
``setup_engine.mtf.weekly_bars`` / ``monthly_bars`` return; this module does not import mtf)
and simulates one long trade per signal week. Results are in plain returns, not R.

Exit rules (any combination; the stop and the time cap are always on)
* R1  weekly close below the 30-week SMA of weekly closes.
* R2  monthly close below the 10-month SMA of monthly closes.
* R3  weekly close at or below (1 - trail_pct) of the highest weekly close held (a 30% trail).
* stop       entry open * (1 - stop_pct), checked on every daily bar.
* time cap   max_weeks completed weekly bars after the signal week.

NO LOOK-AHEAD
-------------
* A weekly or monthly bar is used only once it is COMPLETE: its close is looked at on its
  ``last_day`` (the last trading day of that week/month) and never earlier. A week whose Friday is
  a holiday is therefore decided at the Thursday close.
* A rule that fires on a close cannot trade on that close. The exit is FILLED AT THE NEXT DAILY
  OPEN. (Filling at the close that triggered the rule would let the rule see the price it trades at.)
* R3 only looks at weekly closes of weeks that completed on or after the entry day. The signal
  week's own close, and anything before it, is never the "highest close held": a gap down from a
  big weekly close would otherwise stop the trade out before it began.
* R2 only looks at month-ends on or after the entry day. A month that ended before the entry
  was decided by someone who was not in the trade.

THE ORDER MODEL
---------------
* Signal at the close of a weekly bar (its ``last_day``). Entry at the OPEN of the next daily
  bar. A gap on the entry bar is paid. The entry bar is tested like any other bar.
* Per daily bar from the entry bar on, in this order:
  1. an exit decided at the previous close fills now, at this open (reason = the rule that fired);
  2. the running high is updated (for ``mfe``);
  3. open at or below the stop exits at the open ("gap_stop"); low at or below the stop exits at
     the stop ("stop"). The stop is checked BEFORE any close-based rule on the same bar, so a bar
     that touches the stop and also closes below the SMA exits at the stop;
  4. if a weekly bar completes today: R1, else R3, else the time cap (first one that holds is
     decided at this close, in that priority; highest weekly close held is updated first);
  5. if a month completes today and nothing was decided yet: R2.
* No exit by the last bar: exit at the last close ("end"). Such a trade is still open in
  reality; treat "end" rows as censored if that matters.
* A close-based decision taken on the last bar of the data also ends as "end": there is no next
  open to fill at.
* ``cost_rt`` is a flat round-trip fraction of the entry price subtracted from the gross return.

KNOWN BIASES (not fixable here, reported alongside every result)
----------------------------------------------------------------
Survivorship (today's listings only), Yahoo's retroactive split adjustment, and no dividend
adjustment. Stops are assumed fillable at the stop price inside the bar (only open gaps cost
extra), so real fills on thin stocks are worse: the result is an UPPER bound. The "mfe" fields
are ex-post descriptions of the path, never a feature.

Kernels run under numba when it is installed and as plain Python otherwise. Both give identical
results (tested). Inputs must already be cleaned (positive prices, high >= low).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from numba import njit

    def jit(f):
        return njit(cache=False)(f)

    JIT = True
except Exception:  # numba missing or broken: same code, just slower
    def jit(f):
        return f

    JIT = False

REASONS = ("stop", "gap_stop", "r1_weekly_sma30", "r2_monthly_sma10", "r3_trail30", "time", "end")
RIDE_COLUMNS = ("signal_week", "signal_date", "entry_i", "exit_i", "entry_date", "exit_date",
                "entry", "stop", "exit", "ret_gross", "ret_net", "weeks_held", "mfe", "mfe_close",
                "reason")

# reason codes returned by the kernel (index into REASONS)
_STOP, _GAP_STOP, _R1, _R2, _R3, _TIME, _END = 0, 1, 2, 3, 4, 5, 6


@dataclass(frozen=True)
class RideExit:
    """How a ride is managed. r1/r2/r3 switch the close-based rules on; the stop and the cap are always on."""
    r1: bool = False                # exit when the weekly close < sma_w-week SMA
    r2: bool = False                # exit when the monthly close < sma_m-month SMA
    r3: bool = False                # exit when the weekly close <= (1 - trail_pct) * highest weekly close held
    trail_pct: float = 0.30
    stop_pct: float = 0.35          # initial stop below the entry open
    max_weeks: int = 156            # time cap: completed weekly bars after the signal week
    cost_rt: float = 0.01           # flat round-trip cost, fraction of the entry price
    sma_w: int = 30
    sma_m: int = 10

    def __post_init__(self):
        if not 0 < self.trail_pct < 1:
            raise ValueError(f"trail_pct must be in (0, 1), got {self.trail_pct}")
        if not 0 < self.stop_pct < 1:
            raise ValueError(f"stop_pct must be in (0, 1), got {self.stop_pct}")
        if self.max_weeks < 0:
            raise ValueError(f"max_weeks must be >= 0, got {self.max_weeks}")
        if self.cost_rt < 0:
            raise ValueError(f"cost_rt must be >= 0, got {self.cost_rt}")
        if self.sma_w < 1 or self.sma_m < 1:
            raise ValueError("sma_w and sma_m must be >= 1")


RIDE_R1 = RideExit(r1=True)
RIDE_R2 = RideExit(r2=True)
RIDE_R3 = RideExit(r3=True)


@dataclass
class RideData:
    """One stock's arrays, prepared once and reused for every signal week and exit rule."""
    o: np.ndarray                   # daily open/high/low/close, float64
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    dates: np.ndarray               # datetime64[ns], one per daily bar
    w_pos: np.ndarray               # int64: daily position of each week's last_day
    w_close: np.ndarray             # float64: weekly closes
    w_sma: np.ndarray               # float64: rolling(sma_w, min_periods=sma_w) mean of weekly closes
    wk_at: np.ndarray               # int64, len n: -1, or the week index whose close is this day
    m_pos: np.ndarray               # same, for months
    m_close: np.ndarray
    m_sma: np.ndarray
    mo_at: np.ndarray
    week_labels: pd.DatetimeIndex
    sma_w: int = 30                 # the windows the SMAs above were built with (checked against RideExit)
    sma_m: int = 10


# ── Kernel (contiguous float64 / int64 arrays and scalars only; NaN is "none") ─────

@jit
def _ride_kernel(o, h, l, c, wk_at, w_close, w_sma, mo_at, m_close, m_sma,
                 s_pos, s_week, use_r1, use_r2, use_r3, trail_pct, stop_pct, max_weeks):
    """Walk one ride signalled at daily position s_pos (the last day of weekly bar s_week).

    Returns (exit_i, exit_px, reason_code, max_high, peak_weekly_close). The caller has
    already checked that s_pos + 1 < n. ``peak`` is -inf when no weekly close was held.
    """
    n = len(o)
    entry_i = s_pos + 1
    stop = o[entry_i] * (1.0 - stop_pct)
    pending = -1                         # reason of an exit decided at a close, filled at the next open
    mh = -math.inf
    peak = -math.inf
    for p in range(entry_i, n):
        # 1. a close-based exit decided yesterday fills at today's open
        if pending >= 0:
            return p, o[p], pending, mh, peak
        # 2. best price reached while in the trade (for MFE)
        if h[p] > mh:
            mh = h[p]
        # 3. the stop, BEFORE any close-based rule on the same bar
        if o[p] <= stop:
            return p, o[p], _GAP_STOP, mh, peak
        if l[p] <= stop:
            return p, stop, _STOP, mh, peak
        # 4. a weekly bar completes today
        k = wk_at[p]
        if k >= 0:
            if w_close[k] > peak:
                peak = w_close[k]
            if use_r1 and not math.isnan(w_sma[k]) and w_close[k] < w_sma[k]:
                pending = _R1
            elif use_r3 and w_close[k] <= (1.0 - trail_pct) * peak:
                pending = _R3
            elif k - s_week >= max_weeks:
                pending = _TIME
        # 5. a month completes today (months that ended before the entry are never visited)
        m = mo_at[p]
        if pending < 0 and m >= 0 and use_r2 and not math.isnan(m_sma[m]) and m_close[m] < m_sma[m]:
            pending = _R2
    return n - 1, c[n - 1], _END, mh, peak


# ── Public API ─────────────────────────────────────────────────────────────────

def _positions(index: pd.DatetimeIndex, last_day, what: str) -> np.ndarray:
    ld = pd.DatetimeIndex(pd.to_datetime(np.asarray(last_day)))
    pos = index.get_indexer(ld)
    if (pos < 0).any():
        bad = ld[pos < 0][0]
        raise ValueError(f"{what}: last_day {bad.date()} is not a date in the daily index")
    if len(pos) > 1 and not (np.diff(pos) > 0).all():
        raise ValueError(f"{what}: last_day must be strictly increasing")
    return np.ascontiguousarray(pos, dtype=np.int64)


def _bar_arrays(daily: pd.DataFrame, bars: pd.DataFrame, window: int, what: str):
    for col in ("close", "last_day"):
        if col not in bars.columns:
            raise ValueError(f"{what} is missing column {col!r}")
    pos = _positions(daily.index, bars["last_day"], what)
    close = np.ascontiguousarray(bars["close"].to_numpy(dtype=np.float64))
    sma = np.ascontiguousarray(
        pd.Series(close).rolling(window, min_periods=window).mean().to_numpy(dtype=np.float64))
    at = np.full(len(daily), -1, dtype=np.int64)
    at[pos] = np.arange(len(pos), dtype=np.int64)
    return pos, close, sma, at


def prepare(daily: pd.DataFrame, weekly: pd.DataFrame, monthly: pd.DataFrame,
            sma_w: int = 30, sma_m: int = 10) -> RideData:
    """Arrays for one stock. ``weekly`` / ``monthly`` are indexed by bar label and have ``close`` and ``last_day``.

    Every last_day must be a date in ``daily.index`` (ValueError otherwise), and last_days must
    increase. Only COMPLETE bars may be passed in: this module does not know "today".
    """
    if not isinstance(daily.index, pd.DatetimeIndex) or not daily.index.is_unique \
            or not daily.index.is_monotonic_increasing:
        raise ValueError("daily needs a unique, increasing DatetimeIndex")
    if len(daily) == 0:
        raise ValueError("daily is empty")
    for col in ("open", "high", "low", "close"):
        if col not in daily.columns:
            raise ValueError(f"daily is missing column {col!r}")
    w_pos, w_close, w_sma, wk_at = _bar_arrays(daily, weekly, sma_w, "weekly")
    m_pos, m_close, m_sma, mo_at = _bar_arrays(daily, monthly, sma_m, "monthly")
    f = lambda col: np.ascontiguousarray(daily[col].to_numpy(dtype=np.float64))  # noqa: E731
    return RideData(o=f("open"), h=f("high"), l=f("low"), c=f("close"),
                    dates=daily.index.to_numpy().astype("datetime64[ns]"),
                    w_pos=w_pos, w_close=w_close, w_sma=w_sma, wk_at=wk_at,
                    m_pos=m_pos, m_close=m_close, m_sma=m_sma, mo_at=mo_at,
                    week_labels=pd.DatetimeIndex(weekly.index), sma_w=int(sma_w), sma_m=int(sma_m))


def _check_week(rd: RideData, signal_week: int) -> int:
    sw = int(signal_week)
    if not 0 <= sw < len(rd.w_pos):
        raise ValueError(f"signal_week {signal_week} out of range for {len(rd.w_pos)} weekly bars")
    return sw


def ride(rd: RideData, signal_week: int, ex: RideExit):
    """One trade signalled at the close of weekly bar ``signal_week`` (an integer position in the weekly table).

    Returns a dict with RIDE_COLUMNS, or None when there is no daily bar after the signal week to
    enter on. ``signal_week`` in the result is the week LABEL; ``mfe`` is the best high reached
    while held, ``mfe_close`` the best weekly close held (NaN if no weekly bar completed while in
    the trade).
    """
    if ex.sma_w != rd.sma_w or ex.sma_m != rd.sma_m:
        raise ValueError(f"RideExit uses SMA windows ({ex.sma_w}, {ex.sma_m}) but the data was "
                         f"prepared with ({rd.sma_w}, {rd.sma_m})")
    sw = _check_week(rd, signal_week)
    n = len(rd.o)
    s_pos = int(rd.w_pos[sw])
    if s_pos + 1 >= n:
        return None
    xi, xp, code, mh, peak = _ride_kernel(
        rd.o, rd.h, rd.l, rd.c, rd.wk_at, rd.w_close, rd.w_sma, rd.mo_at, rd.m_close, rd.m_sma,
        s_pos, sw, bool(ex.r1), bool(ex.r2), bool(ex.r3),
        float(ex.trail_pct), float(ex.stop_pct), int(ex.max_weeks))
    xi = int(xi)
    ei = s_pos + 1
    entry = float(rd.o[ei])
    exit_px = float(xp)
    gross = exit_px / entry - 1.0
    entry_date = pd.Timestamp(rd.dates[ei])
    exit_date = pd.Timestamp(rd.dates[xi])
    return {
        "signal_week": rd.week_labels[sw],
        "signal_date": pd.Timestamp(rd.dates[s_pos]),
        "entry_i": ei,
        "exit_i": xi,
        "entry_date": entry_date,
        "exit_date": exit_date,
        "entry": entry,
        "stop": entry * (1.0 - ex.stop_pct),
        "exit": exit_px,
        "ret_gross": gross,
        "ret_net": gross - ex.cost_rt,
        "weeks_held": (exit_date - entry_date).days / 7,
        "mfe": float(mh) / entry - 1.0,
        "mfe_close": math.nan if not math.isfinite(peak) else float(peak) / entry - 1.0,
        "reason": REASONS[int(code)],
    }


def ride_many(rd: RideData, signal_weeks, ex: RideExit) -> pd.DataFrame:
    """One row (RIDE_COLUMNS) per signal week that has a next bar to enter on; an empty frame if none."""
    rows = []
    for sw in signal_weeks:
        r = ride(rd, sw, ex)
        if r is not None:
            rows.append(r)
    if not rows:
        return pd.DataFrame(columns=list(RIDE_COLUMNS))
    return pd.DataFrame(rows, columns=list(RIDE_COLUMNS))


def horizon_return(rd: RideData, signal_week: int, weeks: int, cost_rt: float = 0.01) -> float:
    """Fixed-horizon return: buy at the open after the signal week, sell at the open after week ``signal_week + weeks``.

    NaN when that week does not exist or has no daily bar after it (incomplete window).
    """
    if weeks < 1:
        raise ValueError(f"weeks must be >= 1, got {weeks}")
    sw = _check_week(rd, signal_week)
    n = len(rd.o)
    k = sw + int(weeks)
    if k >= len(rd.w_pos):
        return math.nan
    e0 = int(rd.w_pos[sw]) + 1
    e1 = int(rd.w_pos[k]) + 1
    if e1 >= n or e0 >= n:
        return math.nan
    return float(rd.o[e1] / rd.o[e0] - 1.0 - cost_rt)


def forward_max_close(rd: RideData, signal_week: int, weeks: int) -> float:
    """Best weekly close in the ``weeks`` weeks after the signal week, relative to the entry open.

    FULL windows only (NaN unless ``signal_week + weeks`` exists), as features.forward_labels does.
    Ex-post by construction: a label, never a feature.
    """
    if weeks < 1:
        raise ValueError(f"weeks must be >= 1, got {weeks}")
    sw = _check_week(rd, signal_week)
    n = len(rd.o)
    if sw + int(weeks) >= len(rd.w_close):
        return math.nan
    e0 = int(rd.w_pos[sw]) + 1
    if e0 >= n:
        return math.nan
    return float(rd.w_close[sw + 1: sw + int(weeks) + 1].max() / rd.o[e0] - 1.0)
