"""Corporate-action artifact detector: find days whose price jump is probably a split or reverse split
the data vendor did not adjust, not a real move.

THE RULE (POST-HOC). A day is flagged when ALL of
    (1) the close moved by at least MIN_ABS_RET (50%) against the previous close, up or down;
    (2) the PREVIOUS day's dollar volume (close x volume) was at least MIN_PRIOR_DVOL ($1M), so thin
        penny-stock noise is not scanned;
    (3) the day's own volume was BELOW VOL_SPIKE (2x) the mean volume of the VOL_WIN (20) days before it
        (needs at least VOL_MIN_DAYS of them). A real crash or squeeze draws a volume burst; a pure
        re-quote of the same company at a new nominal price does not.
The volume baseline is the mean of days d-20 .. d-1 and NEVER includes day d (a look-ahead guard: the
day's own spike would otherwise lift its own baseline and hide itself).

POST-HOC. This rule was chosen after looking at suspicious bars in the price cache, not before the
studies it is used on. It is a sensitivity tool, not a pre-registered test, and nothing here changes a
published verdict by itself.

WHAT IT MISCLASSIFIES (read before trusting a result that uses it).
  * A real crash or spike on news during a trading halt: the stock reopens -50% on volume that is still
    under 2x a prior average already inflated by the run-up. Trial failures and fraud halts are flagged
    as "artifacts", which removes genuine losers and FLATTERS a dropped-sample result.
  * Gap-ups on low volume (buy-out announcements in thin names) are flagged the same way.
  * ADR ratio changes, share-class consolidations and unadjusted spin-offs ARE what the rule wants, but
    it only catches them when their volume stays quiet.
  * It only sees the CURRENT cache. Yahoo adjusts old splits retroactively, so an old reverse-split stock
    shows no jump at all (its price band is shifted, not broken) and is invisible here. "No artifact
    found" does not mean "clean".
  * The dollar-volume floor uses a split-adjusted close, so it is not what traded at the time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_ABS_RET = 0.50
MIN_PRIOR_DVOL = 1_000_000
VOL_SPIKE = 2.0
VOL_WIN = 20
VOL_MIN_DAYS = 10


def artifact_days(daily: pd.DataFrame) -> pd.DatetimeIndex:
    """Dates of the bars that look like an unadjusted split (see the module docstring for the rule).

    ``daily`` is indexed by date, or carries a ``date`` column; it needs ``close`` and ``volume``.
    Missing values never flag a day. The result is named ``date``.
    """
    df = daily
    if not isinstance(df.index, pd.DatetimeIndex) and "date" in df.columns:
        df = df.set_index(pd.DatetimeIndex(pd.to_datetime(df["date"])))
    close = df["close"].astype("float64")
    vol = df["volume"].astype("float64")
    ret = close / close.shift(1) - 1.0
    prior_dvol = (close * vol).shift(1)
    prior_mean = vol.rolling(VOL_WIN, min_periods=VOL_MIN_DAYS).mean().shift(1)    # days d-20 .. d-1, never day d
    with np.errstate(invalid="ignore"):
        flag = (ret.abs() >= MIN_ABS_RET) & (prior_dvol >= MIN_PRIOR_DVOL) & (vol < VOL_SPIKE * prior_mean)
    return pd.DatetimeIndex(df.index[flag.to_numpy(dtype=bool)], name="date")


def contaminated(days, start, end) -> bool:
    """True if any artifact day ``d`` has ``start <= d <= end`` (both ends inclusive)."""
    arr = np.asarray(pd.DatetimeIndex(days), dtype="datetime64[ns]")
    lo, hi = pd.Timestamp(start).to_datetime64(), pd.Timestamp(end).to_datetime64()
    return bool(((arr >= lo) & (arr <= hi)).any())


def contaminated_mask(days_by_symbol: dict, symbols, starts, ends) -> np.ndarray:
    """Vectorised ``contaminated``: one bool per (symbol, start, end) triple. Unknown symbol -> False.

    Per symbol the artifact days are sorted once; ``searchsorted(start, 'left')`` finds the first day on
    or after the start, which is inside the window iff it is <= end. NaT never contaminates.
    """
    syms = np.asarray(symbols, dtype=object)
    lo = np.asarray(pd.to_datetime(starts), dtype="datetime64[ns]")
    hi = np.asarray(pd.to_datetime(ends), dtype="datetime64[ns]")
    out = np.zeros(len(syms), dtype=bool)
    if not len(syms):
        return out
    codes, uniques = pd.factorize(pd.Series(syms, dtype=object))
    for k, sym in enumerate(uniques):
        days = days_by_symbol.get(sym)
        if days is None or len(days) == 0:
            continue
        arr = np.sort(np.asarray(pd.DatetimeIndex(days), dtype="datetime64[ns]"))
        idx = np.flatnonzero(codes == k)
        j = np.searchsorted(arr, lo[idx], side="left")
        ok = j < len(arr)
        hit = np.zeros(len(idx), dtype=bool)
        hit[ok] = arr[j[ok]] <= hi[idx][ok]
        out[idx] = hit
    return out
