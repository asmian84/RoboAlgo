"""
A pre-registered test of the open-source Kronos candlestick foundation model on US stocks. Pure
functions only: origins, universe, forecast returns, cross-sectional IC, decile spreads, baselines, the
verdict. Nothing here imports torch and nothing here touches the network; the predictor is injected, so
the offline tests run against a fake.

PRE-REGISTRATION (fixed before any data was seen; not edited after the first run).

WHAT IS TESTED. Kronos-small (NeoQuasar/Kronos-small) with Kronos-Tokenizer-base, ZERO-SHOT, no
fine-tuning. It is a decoder-only transformer that generates future OHLCV candles from past candles.
Lookback 400 daily bars, horizon 20 trading days, T=1.0, top_p=0.9, sample_count=5. The forecast return
for a stock at an origin is the mean over the sample paths of (predicted close at day 20 / last observed
close) - 1. ``predict_batch`` averages the 5 decoded paths inside the model, so the mean ratio equals
mean(close at day 20) / last close - 1; both forms are the same number.

ORIGINS. Every month-end trading day of the shared calendar. CLEAN window: origins from 2025-07-31
through the last month-end that has 20 trading days of realized forward data in the cache. Kronos's
weights went public on 2025-06-30 and its pretraining range is NOT disclosed, so any US data before
2025-07-01 may have been seen in training and is treated as contaminated. INFORMATIONAL window: month
ends 2023-01 through 2024-12, reported separately and labelled as possibly seen in training. It never
enters the verdict.

UNIVERSE at each origin: the top 500 US stocks by 60-day average dollar volume computed from data up to
and including the origin, last close >= $1, at least 400 bars of history. No look-ahead: only bars up to
the origin close enter the universe filter or the model input.

REALIZED RETURN. Close at origin + 20 trading days / close at origin - 1, from the cache, close to close.

METRICS per origin, cross-sectional: Spearman rank IC between forecast return and realized return; top
decile minus bottom decile mean realized return; top decile minus the universe mean. Decile size
k = max(1, floor(n * 10% + 0.5)).

BASELINES on the same universe and origins: 12-1 momentum (close[t-21] / close[t-252] - 1), 1-month
reversal (-(close[t] / close[t-21] - 1)), and a shuffled-rank placebo with a fixed seed.

STATISTICS. Month-clustered bootstrap, 2000 draws, seed 0, 95% (setup_engine.trade_stats), for the mean
IC of each signal and for (Kronos IC - momentum IC). With about 14 clean origins the intervals are wide
and the test is weak.

VERDICT (mechanical, see ``evaluate_verdict``). PASS only if the lower bound of the 95% interval of the
Kronos mean IC is above 0 in the CLEAN window AND the lower bound of the interval of (Kronos IC -
momentum IC) is above 0. A lower bound of exactly 0 is not a PASS. Otherwise the verdict is "no
demonstrated edge". "UNDERPOWERED" is reported when the clean window holds fewer than 12 origins.

COSTS. Decile spreads are GROSS. No transaction costs, no slippage, no borrow. A monthly top-decile
strategy would pay all of them.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from setup_engine import trade_stats as TS

# ── pre-registered choices ────────────────────────────────────────────────────────────────────────

LOOKBACK = 400                 # daily bars given to the model
HORIZON = 20                   # trading days forecast, and the realized-return horizon
SAMPLE_COUNT = 5
TEMPERATURE = 1.0
TOP_P = 0.9

TOP_N = 500                    # universe size
DVOL_WINDOW = 60               # "60-day average dollar volume"
MIN_PRICE = 1.0                # last close >= $1
MIN_BARS = 400                 # at least 400 bars of history at the origin

CLEAN_START = pd.Timestamp("2025-07-31")     # weights public 2025-06-30: earlier data may be contaminated
INFO_START = pd.Timestamp("2023-01-01")
INFO_END = pd.Timestamp("2024-12-31")

MIN_CLEAN_ORIGINS = 12         # below this the window is UNDERPOWERED
N_BOOT = 2000
SEED = 0
ALPHA = 0.05
DECILE = 0.10
PLACEBO_SEED = 20261020

MOM_LONG = 252                 # 12-1 momentum: close[t-21] / close[t-252] - 1
MOM_SKIP = 21
REV_WINDOW = 21                # 1-month reversal: -(close[t] / close[t-21] - 1)

SIGNALS = ("kronos", "momentum", "reversal", "placebo")

# ── calendar and origins ──────────────────────────────────────────────────────────────────────────

def trading_days(dates) -> pd.DatetimeIndex:
    """Sorted unique, timezone-naive trading days."""
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return pd.DatetimeIndex(np.unique(idx.values))


def month_end_days(calendar) -> list[pd.Timestamp]:
    """The last trading day of every calendar month present in ``calendar``."""
    days = trading_days(calendar)
    if len(days) == 0:
        return []
    s = pd.Series(days, index=days)
    return [pd.Timestamp(t) for t in s.groupby([days.year, days.month]).max()]


def origin_windows(calendar, horizon: int = HORIZON) -> dict:
    """Month-end origins with ``horizon`` trading days of realized data after them, split into windows.

    Returns {"clean": [...], "informational": [...], "immature": [...], "all": [...]}. An origin without
    enough forward trading days in the calendar is "immature" and is dropped from both windows: its
    realized return would not exist.
    """
    days = trading_days(calendar)
    ends = month_end_days(days)
    pos = days.get_indexer(pd.DatetimeIndex(ends)) if ends else np.array([], dtype=np.int64)
    mature, immature = [], []
    for t, p in zip(ends, pos):
        if p >= 0 and p + horizon < len(days):
            mature.append(pd.Timestamp(t))
        else:
            immature.append(pd.Timestamp(t))
    clean = [t for t in mature if t >= CLEAN_START]
    info = [t for t in mature if INFO_START <= t <= INFO_END]
    return {"clean": clean, "informational": info, "immature": immature, "all": mature}


def _position(index: pd.DatetimeIndex, when) -> int:
    """Position of ``when`` in ``index``, or -1 if it is not a bar of that series."""
    t = pd.Timestamp(when)
    i = int(index.searchsorted(t))
    return i if i < len(index) and index[i] == t else -1


# ── universe ──────────────────────────────────────────────────────────────────────────────────────

def symbol_stats(frame: pd.DataFrame, origins, dvol_window: int = DVOL_WINDOW, min_price: float = MIN_PRICE,
                 min_bars: int = MIN_BARS) -> pd.DataFrame:
    """One symbol's universe inputs at each origin, using only bars up to and including the origin close.

    Columns: origin, n_bars, last_close, dvol (mean close*volume over the last ``dvol_window`` bars),
    last_date (the symbol's last bar at or before the origin). Rows where the symbol has fewer than
    ``min_bars`` bars at the origin, a last close below ``min_price``, or a non-finite number are dropped.
    ``origins`` must be increasing; a bar after an origin never enters that origin's numbers.
    """
    origins = pd.DatetimeIndex(pd.to_datetime(list(origins))).sort_values()
    out = pd.DataFrame(columns=["origin", "n_bars", "last_close", "dvol", "last_date"])
    if len(frame) == 0 or len(origins) == 0:
        return out
    close = frame["close"].to_numpy(dtype=float)
    dvol = close * frame["volume"].to_numpy(dtype=float)
    pos = np.searchsorted(frame.index.values.astype("datetime64[ns]"),
                          origins.values.astype("datetime64[ns]"), side="right") - 1
    n = pos + 1
    need = max(int(min_bars), int(dvol_window))
    ok = (pos >= 0) & (n >= need)
    if not ok.any():
        return out
    pos, n = pos[ok], n[ok]
    # a direct windowed mean, not a cumsum difference: dollar volume spans many orders of magnitude and the
    # subtraction cancels away the small recent sum
    avg = np.array([dvol[p + 1 - dvol_window:p + 1].mean() for p in pos])
    last = close[pos]
    rows = pd.DataFrame({"origin": origins[ok], "n_bars": n.astype(np.int64), "last_close": last,
                         "dvol": avg, "last_date": frame.index.values[pos]})
    rows = rows[np.isfinite(rows["last_close"]) & np.isfinite(rows["dvol"]) & (rows["last_close"] >= min_price)]
    return rows.reset_index(drop=True)[out.columns]


def universe_candidates(daily: dict, origin, dvol_window: int = DVOL_WINDOW, min_price: float = MIN_PRICE,
                        min_bars: int = MIN_BARS) -> pd.DataFrame:
    """Per-symbol universe inputs at ``origin`` for a whole panel of symbols (see ``symbol_stats``).

    Columns: symbol, n_bars, last_close, dvol, last_date. Only bars up to the origin close are used.
    """
    rows = []
    for sym, frame in daily.items():
        s = symbol_stats(frame, [origin], dvol_window, min_price, min_bars)
        if len(s):
            rows.append(s.assign(symbol=str(sym)))
    cols = ["symbol", "n_bars", "last_close", "dvol", "last_date"]
    return pd.concat(rows, ignore_index=True)[cols] if rows else pd.DataFrame(columns=cols)


def select_universe(candidates: pd.DataFrame, top_n: int = TOP_N) -> list[str]:
    """The ``top_n`` symbols by dollar volume, largest first; ties broken by symbol so the set is stable."""
    if len(candidates) == 0:
        return []
    c = candidates.sort_values(["dvol", "symbol"], ascending=[False, True], kind="stable")
    return [str(s) for s in c["symbol"].head(top_n)]


# ── returns ───────────────────────────────────────────────────────────────────────────────────────

def realized_return(frame: pd.DataFrame, origin, horizon: int = HORIZON) -> Optional[float]:
    """Close at origin + ``horizon`` trading days (in that series) / close at origin - 1.

    The origin itself is never the endpoint: the forward window starts at the NEXT bar. None when the
    origin is not a bar of the series or fewer than ``horizon`` bars follow it.
    """
    i = _position(frame.index, origin)
    if i < 0 or i + horizon >= len(frame):
        return None
    c0 = float(frame["close"].iloc[i])
    c1 = float(frame["close"].iloc[i + horizon])
    if not (np.isfinite(c0) and np.isfinite(c1)) or c0 <= 0:
        return None
    return c1 / c0 - 1.0


def forecast_return(pred_close, last_close: float) -> float:
    """Mean over sample paths of (predicted close / last observed close) - 1.

    ``pred_close`` is either one number (an already averaged path) or an array of sample-path closes at
    the horizon. Dividing the constant ``last_close`` out first is exact, so both forms agree.
    """
    p = np.asarray(pred_close, dtype=float).ravel()
    p = p[np.isfinite(p)]
    if len(p) == 0 or not np.isfinite(last_close) or last_close <= 0:
        raise ValueError("forecast_return: no usable predicted close, or a non-positive last close")
    return float(np.mean(p / float(last_close) - 1.0))


def momentum(frame: pd.DataFrame, origin, long: int = MOM_LONG, skip: int = MOM_SKIP) -> Optional[float]:
    """12-1 momentum at the origin: close[t-skip] / close[t-long] - 1. None without enough history."""
    i = _position(frame.index, origin)
    if i < long:
        return None
    a, b = float(frame["close"].iloc[i - skip]), float(frame["close"].iloc[i - long])
    if not (np.isfinite(a) and np.isfinite(b)) or b <= 0:
        return None
    return a / b - 1.0


def reversal(frame: pd.DataFrame, origin, window: int = REV_WINDOW) -> Optional[float]:
    """1-month reversal at the origin: -(close[t] / close[t-window] - 1). None without enough history."""
    i = _position(frame.index, origin)
    if i < window:
        return None
    a, b = float(frame["close"].iloc[i]), float(frame["close"].iloc[i - window])
    if not (np.isfinite(a) and np.isfinite(b)) or b <= 0:
        return None
    return -(a / b - 1.0)


# ── cross-sectional statistics ────────────────────────────────────────────────────────────────────

def average_rank(x) -> np.ndarray:
    """Ranks from 1 to n with ties sharing their mean rank (the Spearman convention)."""
    a = np.asarray(x, dtype=float).ravel()
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    starts = ends - counts + 1
    return ((starts + ends) / 2.0)[inv]


def spearman_ic(x, y) -> Optional[float]:
    """Spearman rank correlation of paired values; None with fewer than 3 usable pairs or a constant side."""
    a = np.asarray(x, dtype=float).ravel()
    b = np.asarray(y, dtype=float).ravel()
    if len(a) != len(b):
        raise ValueError("spearman_ic: x and y must have the same length")
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    if len(a) < 3:
        return None
    ra, rb = average_rank(a), average_rank(b)
    da, db = ra - ra.mean(), rb - rb.mean()
    den = float(np.sqrt((da * da).sum() * (db * db).sum()))
    return None if den <= 0 else float((da * db).sum() / den)


def decile_count(n: int, decile: float = DECILE) -> int:
    """Names per tail: max(1, floor(n * decile + 0.5))."""
    return max(1, int(np.floor(n * decile + 0.5)))


def decile_spread(forecast, realized, decile: float = DECILE) -> dict:
    """Top-decile minus bottom-decile mean realized return, and top decile minus the universe mean.

    Names are ordered by forecast, largest first; ties keep the caller's order (sort the rows by symbol
    before calling, so the result is deterministic). Non-finite rows are dropped first.
    """
    f = np.asarray(forecast, dtype=float).ravel()
    r = np.asarray(realized, dtype=float).ravel()
    if len(f) != len(r):
        raise ValueError("decile_spread: forecast and realized must have the same length")
    keep = np.isfinite(f) & np.isfinite(r)
    f, r = f[keep], r[keep]
    n = len(f)
    if n == 0:
        return {"n": 0, "k": 0, "top": None, "bottom": None, "spread": None, "universe_mean": None,
                "top_minus_universe": None}
    k = min(decile_count(n, decile), n)
    order = np.argsort(-f, kind="stable")
    top = float(r[order[:k]].mean())
    bottom = float(r[order[-k:]].mean())
    universe = float(r.mean())
    return {"n": int(n), "k": int(k), "top": top, "bottom": bottom, "spread": top - bottom,
            "universe_mean": universe, "top_minus_universe": top - universe}


def per_origin(forecast, realized, decile: float = DECILE) -> dict:
    """One origin's cross-sectional result: IC plus the two spreads."""
    d = decile_spread(forecast, realized, decile)
    return {"ic": spearman_ic(forecast, realized), **d}


def placebo_forecast(values, seed: int = PLACEBO_SEED) -> np.ndarray:
    """A fixed-seed permutation of the forecast values: the same ranks, randomly matched to outcomes."""
    a = np.asarray(values, dtype=float).ravel()
    return np.random.default_rng(seed).permutation(a)


# ── bootstrap and verdict ────────────────────────────────────────────────────────────────────────

def mean_ic(rows: list[dict]) -> list[float]:
    """The per-origin IC values, origin order, missing ICs dropped."""
    return [float(r["ic"]) for r in rows if r.get("ic") is not None and np.isfinite(r["ic"])]


def cluster_summary(values, clusters, n_boot: int = N_BOOT, seed: int = SEED, alpha: float = ALPHA) -> dict:
    """Mean with a month-clustered percentile interval (setup_engine.trade_stats, reused unchanged)."""
    return TS.cluster_bootstrap(values, clusters, n_boot=n_boot, seed=seed, alpha=alpha)


def diff_summary(values_a, clusters_a, values_b, clusters_b, n_boot: int = N_BOOT, seed: int = SEED,
                 alpha: float = ALPHA) -> dict:
    """(mean of arm A - mean of arm B) with a joint month-clustered interval."""
    return TS.cluster_bootstrap_diff(values_a, clusters_a, values_b, clusters_b, n_boot=n_boot, seed=seed, alpha=alpha)


def _positive_lower_bound(ci) -> bool:
    """True only when the interval exists and its lower bound is strictly above zero."""
    lo = (ci or {}).get("lo")
    return bool(lo is not None and np.isfinite(lo) and lo > 0.0)


def evaluate_verdict(kronos_ci, diff_ci, n_clean_origins: int, min_origins: int = MIN_CLEAN_ORIGINS) -> dict:
    """The pre-registered rule, applied mechanically.

    PASS = the 95% lower bound of the Kronos mean IC is above 0 AND the 95% lower bound of
    (Kronos IC - momentum IC) is above 0. A lower bound of exactly 0 is not above 0. UNDERPOWERED is a
    separate flag (fewer than ``min_origins`` clean origins); it never changes the arithmetic.
    """
    ic_ok = _positive_lower_bound(kronos_ci)
    diff_ok = _positive_lower_bound(diff_ci)
    passed = ic_ok and diff_ok
    return {"verdict": "PASS" if passed else "no demonstrated edge", "pass": passed,
            "kronos_ic_lower_bound_above_zero": ic_ok, "beats_momentum_lower_bound_above_zero": diff_ok,
            "underpowered": bool(n_clean_origins < min_origins),
            "n_clean_origins": int(n_clean_origins), "min_clean_origins": int(min_origins)}
