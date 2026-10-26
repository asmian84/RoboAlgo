"""
Trade simulation for ATR-based exits, so a screen can be measured in R-multiples.

A screen (setup_engine/screen.py) says "these bars qualify". Whether trading them
makes money depends on the exit rule, so this module takes a set of signal bars and
simulates one long trade per signal under an ``ExitModel`` (ATR stop, optional ATR
trailing stop, optional R target, time stop, flat cost). Results are in R, where
1R is the initial risk ``stop_atr * ATR(t)``, so trades on a $2 stock and a $200
stock are comparable.

NO LOOK-AHEAD
-------------
* A signal on bar t is decided at the close of t. Everything that sets the trade up
  (the ATR that sizes the stop, the signal itself) uses bars <= t. The ATR of the
  entry bar is NOT used.
* The trailing stop in force during bar e is built from bars <= e-1 only, so it is
  known at the open of e. A bar's own high can never raise the stop that the same
  bar is tested against.
* ``tests/test_setup_engine/test_trade_sim`` checks that every trade that closed
  before a cut is identical whether or not later bars exist.

THE ORDER MODEL (stated because a backtest is only as honest as its fills)
--------------------------------------------------------------------------
* Entry: at the OPEN of bar t+1 (market-on-open after the signal). No fill
  improvement; a gap on the entry bar is paid.
* Initial stop = entry - stop_atr * ATR(t). Risk (1R) = stop_atr * ATR(t).
* Trailing stop (optional): at the start of bar e > entry bar, stop = max(stop,
  highest high of bars entry..e-1 - trail_atr * ATR(e-1)). It only ratchets up.
* Per bar, in this order: open at or below the stop exits at the open ("gap_stop");
  low at or below the stop exits at the stop ("stop", or "trail" if the stop has been
  raised above its initial level); open at or above the target exits at the open
  (favourable gap, as in ``strategy_validation.finder_backtest``); high at or above
  the target exits at the target; ``max_bars`` bars after entry exits at the close
  ("time"). **If one bar could hit both stop and target, the stop is assumed
  first.** The entry bar is tested like any other bar.
* No exit by the last bar: exit at the last close ("end"). Such a trade is still
  open in reality; treat the "end" rows as censored if that matters.
* Costs: ``cost_pct`` is a flat round-trip fraction of the entry price, charged in
  R as ``cost_pct * entry / risk``.
* One position at a time per symbol: after an exit at bar x, a signal on bar t' is
  taken only if t' >= x + cooldown. With cooldown=0 this matches the re-entry
  convention of ``finder_backtest`` (the next signal at or after the exit bar).
* A signal is skipped when the stop would sit at or below zero (very high ATR
  against a low price), when ATR(t) is missing or not positive, or when there is
  no next bar to enter on.

KNOWN BIASES (not fixable here, reported alongside every result)
----------------------------------------------------------------
Survivorship (today's listings only), Yahoo's retroactive split adjustment
(distorts historical price bands and ATR for reverse-split stocks), and no
dividend adjustment. Stops are assumed fillable at the stop price inside the bar
(only open gaps cost extra), so real fills on thin stocks are worse. The result is
an UPPER bound.

Kernels run under numba when it is installed and as plain Python otherwise. Both
give identical results (tested). Inputs must already be cleaned (positive prices,
high >= low), as ``setup_engine.screen.clean_daily`` produces.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

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

REASONS = ("stop", "gap_stop", "trail", "target", "time", "end")
TRADE_COLUMNS = ("symbol", "signal_i", "entry_i", "exit_i", "signal_date", "entry_date", "exit_date",
                 "entry", "stop", "exit", "risk", "r_gross", "cost_r", "r_net", "bars", "reason",
                 "mfe_r", "atr_pct")

# reason codes returned by the kernels (index into REASONS)
_STOP, _GAP_STOP, _TRAIL, _TARGET, _TIME, _END = 0, 1, 2, 3, 4, 5


@dataclass(frozen=True)
class ExitModel:
    """How a trade is managed. ``None`` for trail_atr / target_r switches that exit off."""
    stop_atr: float = 2.0                   # initial stop distance, in ATR(t); also defines 1R
    trail_atr: Optional[float] = 3.0        # trailing stop distance in ATR, or None
    target_r: Optional[float] = None        # profit target in R, or None
    max_bars: int = 120                     # time stop: bars after the entry bar
    cost_pct: float = 0.01                  # flat round-trip cost, fraction of the entry price

    def __post_init__(self):
        if not self.stop_atr > 0:
            raise ValueError(f"stop_atr must be positive, got {self.stop_atr}")
        if self.trail_atr is not None and not self.trail_atr > 0:
            raise ValueError(f"trail_atr must be positive or None, got {self.trail_atr}")
        if self.target_r is not None and not self.target_r > 0:
            raise ValueError(f"target_r must be positive or None, got {self.target_r}")
        if self.max_bars < 0:
            raise ValueError(f"max_bars must be >= 0, got {self.max_bars}")
        if self.cost_pct < 0:
            raise ValueError(f"cost_pct must be >= 0, got {self.cost_pct}")


# ── Kernels (contiguous float64 / int64 arrays and scalars only; None is NaN) ──

@jit
def _walk(o, h, l, c, atr, t, stop_atr, trail_atr, target_r, max_bars):
    """Walk one trade signalled on bar t. Returns (exit_i, exit_px, reason_code, max_high).

    The caller has already checked t+1 < n, atr[t] > 0 and that the initial stop is > 0.
    """
    n = len(o)
    entry_i = t + 1
    E = o[entry_i]
    risk = stop_atr * atr[t]
    stop = E - risk
    stop0 = stop
    trail_on = not math.isnan(trail_atr)
    target_on = not math.isnan(target_r)
    tgt = E + target_r * risk            # NaN when the target is off
    hh = -math.inf
    mh = -math.inf
    for e in range(entry_i, n):
        # 1. trailing stop, from information through bar e-1 only; it only ratchets up
        if e > entry_i and trail_on:
            if h[e - 1] > hh:
                hh = h[e - 1]
            cand = hh - trail_atr * atr[e - 1]
            if cand > stop:              # NaN never passes, so a missing ATR cannot move the stop
                stop = cand
        # 2. best price reached while in the trade (for MFE)
        if h[e] > mh:
            mh = h[e]
        # 3. gap through the stop: filled at the open
        if o[e] <= stop:
            return e, o[e], _GAP_STOP, mh
        # 4. stop touched inside the bar. Checked BEFORE the target: stop-first on a both-hit bar
        if l[e] <= stop:
            if stop > stop0:
                return e, stop, _TRAIL, mh
            return e, stop, _STOP, mh
        # 5-6. target: a favourable gap fills at the open, otherwise at the target
        if target_on and o[e] >= tgt:
            return e, o[e], _TARGET, mh
        if target_on and h[e] >= tgt:
            return e, tgt, _TARGET, mh
        # 7. time stop
        if e - entry_i >= max_bars:
            return e, c[e], _TIME, mh
    return n - 1, c[n - 1], _END, mh


@jit
def _run_signals(o, h, l, c, atr, sig_idx, stop_atr, trail_atr, target_r, max_bars, cooldown):
    """One position at a time over sorted, unique signal bars. Returns (k, t, exit_i, exit_px, code, max_high)."""
    n = len(o)
    m = len(sig_idx)
    t_out = np.empty(m, dtype=np.int64)
    xi_out = np.empty(m, dtype=np.int64)
    xp_out = np.empty(m, dtype=np.float64)
    rc_out = np.empty(m, dtype=np.int64)
    mh_out = np.empty(m, dtype=np.float64)
    k = 0
    next_ok = 0
    for j in range(m):
        t = sig_idx[j]
        if t < next_ok:                      # still in a trade, or cooling down
            continue
        if t + 1 >= n:                       # nothing to enter on
            continue
        if not (atr[t] > 0.0):               # missing or non-positive ATR
            continue
        if not (o[t + 1] - stop_atr * atr[t] > 0.0):   # stop at or below zero (also rejects NaN)
            continue
        xi, xp, rc, mh = _walk(o, h, l, c, atr, t, stop_atr, trail_atr, target_r, max_bars)
        t_out[k] = t
        xi_out[k] = xi
        xp_out[k] = xp
        rc_out[k] = rc
        mh_out[k] = mh
        k += 1
        next_ok = max(xi + cooldown, t + 1)
    return k, t_out, xi_out, xp_out, rc_out, mh_out


@jit
def _run_each(o, h, l, c, atr, starts, stop_atr, trail_atr, target_r, max_bars):
    """Every start is its own trade: no cooldown, no overlap check. Starts must already be valid."""
    m = len(starts)
    xi_out = np.empty(m, dtype=np.int64)
    xp_out = np.empty(m, dtype=np.float64)
    rc_out = np.empty(m, dtype=np.int64)
    mh_out = np.empty(m, dtype=np.float64)
    for j in range(m):
        xi, xp, rc, mh = _walk(o, h, l, c, atr, starts[j], stop_atr, trail_atr, target_r, max_bars)
        xi_out[j] = xi
        xp_out[j] = xp
        rc_out[j] = rc
        mh_out[j] = mh
    return xi_out, xp_out, rc_out, mh_out


# ── Public API ─────────────────────────────────────────────────────────────────

def _arrays(ind: pd.DataFrame):
    for col in ("open", "high", "low", "close", "atr14"):
        if col not in ind.columns:
            raise ValueError(f"ind is missing column {col!r}")
    return tuple(np.ascontiguousarray(ind[col].to_numpy(dtype=np.float64))
                 for col in ("open", "high", "low", "close", "atr14"))


def _nan_if_none(x) -> float:
    return math.nan if x is None else float(x)


def simulate(ind: pd.DataFrame, signal_idx, model: ExitModel, cooldown: int = 20,
             symbol: str = "") -> pd.DataFrame:
    """One row (TRADE_COLUMNS) per trade taken, one position at a time.

    ``ind`` needs open/high/low/close/atr14 columns and an increasing DatetimeIndex.
    ``signal_idx`` are integer positions of the signal bars; duplicates are ignored.
    """
    if not isinstance(ind.index, pd.DatetimeIndex) or not ind.index.is_monotonic_increasing:
        raise ValueError("ind needs an increasing DatetimeIndex")
    o, h, l, c, atr = _arrays(ind)
    n = len(o)
    sig = np.unique(np.asarray(signal_idx, dtype=np.int64))
    if len(sig) and (sig[0] < 0 or sig[-1] >= n):
        raise ValueError(f"signal_idx out of range for {n} bars")
    k, t, xi, xp, rc, mh = _run_signals(o, h, l, c, atr, sig, float(model.stop_atr),
                                        _nan_if_none(model.trail_atr), _nan_if_none(model.target_r),
                                        int(model.max_bars), int(cooldown))
    t, xi, xp, rc, mh = t[:k], xi[:k], xp[:k], rc[:k], mh[:k]
    ei = t + 1
    entry = o[ei]
    risk = model.stop_atr * atr[t]
    r_gross = (xp - entry) / risk
    cost_r = model.cost_pct * entry / risk
    idx = ind.index
    return pd.DataFrame({
        "symbol": np.full(k, symbol, dtype=object),
        "signal_i": t,
        "entry_i": ei,
        "exit_i": xi,
        "signal_date": idx[t],
        "entry_date": idx[ei],
        "exit_date": idx[xi],
        "entry": entry,
        "stop": entry - risk,
        "exit": xp,
        "risk": risk,
        "r_gross": r_gross,
        "cost_r": cost_r,
        "r_net": r_gross - cost_r,
        "bars": xi - ei,
        "reason": np.array(REASONS, dtype=object)[rc],
        "mfe_r": (mh - entry) / risk,
        "atr_pct": atr[t] / c[t],
    }, columns=list(TRADE_COLUMNS))


def simulate_each(ind: pd.DataFrame, start_mask, model: ExitModel) -> dict:
    """Every True bar of ``start_mask`` is an independent trade (no cooldown, no overlap).

    This is for control pools, where each eligible bar is a sample. Returns arrays of
    length n: ``valid``; ``r_gross`` (NaN where invalid); ``inv_risk`` = entry / risk, so
    cost in R is ``cost_pct * inv_risk``; ``exit_i`` and ``reason`` (index into REASONS),
    both -1 where invalid. A start is valid iff it is in the mask, a next bar exists,
    ATR(t) > 0 and the initial stop is > 0.
    """
    o, h, l, c, atr = _arrays(ind)
    n = len(o)
    mask = np.asarray(start_mask, dtype=bool)
    if mask.shape != (n,):
        raise ValueError(f"start_mask must have shape ({n},), got {mask.shape}")
    valid = np.zeros(n, dtype=bool)
    if n > 1:
        with np.errstate(invalid="ignore"):
            valid[:-1] = mask[:-1] & (atr[:-1] > 0.0) & (o[1:] - model.stop_atr * atr[:-1] > 0.0)
    r_gross = np.full(n, np.nan)
    inv_risk = np.full(n, np.nan)
    exit_i = np.full(n, -1, dtype=np.int64)
    reason = np.full(n, -1, dtype=np.int64)
    starts = np.ascontiguousarray(np.flatnonzero(valid), dtype=np.int64)
    if len(starts):
        xi, xp, rc, _ = _run_each(o, h, l, c, atr, starts, float(model.stop_atr),
                                  _nan_if_none(model.trail_atr), _nan_if_none(model.target_r),
                                  int(model.max_bars))
        entry = o[starts + 1]
        risk = model.stop_atr * atr[starts]
        r_gross[starts] = (xp - entry) / risk
        inv_risk[starts] = entry / risk
        exit_i[starts] = xi
        reason[starts] = rc
    return {"valid": valid, "r_gross": r_gross, "inv_risk": inv_risk, "exit_i": exit_i, "reason": reason}
