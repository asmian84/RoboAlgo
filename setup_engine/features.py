"""
Scale-free chart features that work on any timeframe, for learning which setups precede big runs.

THE IDEA. A setup is a SHAPE, not a price level or a calendar span: a base that has tightened,
a trend that has turned up, volume that has dried and then returned. If a shape matters, it
should matter on daily, weekly and monthly bars alike. So every feature here is a ratio measured
in BARS, and the same bar-count windows (10 / 30 / 50) are used on every timeframe. Daily,
weekly and monthly bars then see the "same" setup literally the same way, and the learner can
tell us whether the shape really is timeframe-invariant instead of us assuming it.

NO LOOK-AHEAD. A feature at bar i uses bars <= i only. `tests/test_setup_engine` checks that
truncating the future leaves every feature unchanged, and mutation-tests it. Bars are labelled
by the END of their period and an observation date only sees bars labelled on or before it, so
a weekly bar is never used before its Friday and a monthly bar never before its month-end.

WHAT IS DELIBERATELY NOT A FEATURE. Price level. Yahoo split-adjusts history retroactively, so
a stock that traded at $20 and later split 10:1 shows as $2 in the past, and "penny" would be
learned from split history rather than from anything tradable. Liquidity (dollar volume, which
split-adjustment cancels out of) is used as a FILTER instead.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from strategy_validation.finder_backtest import rolling_percentile_rank

# one shared set of windows, in bars, on every timeframe
N_SHORT, N_MID, N_LONG = 10, 30, 50

TIMEFRAMES = ("D", "W", "M")
_RESAMPLE = {"W": "W-FRI", "M": "ME"}
# a bar older than this (in calendar days) at an observation date means the stock was not trading then
_MAX_BAR_AGE = {"D": 5, "W": 9, "M": 35}

BAR_FEATURES = [
    "c_ma10", "c_ma30", "c_ma50", "ma10_30", "ma30_50", "ma50_slope",
    "dist_hi50", "dist_lo50", "base_depth30", "range10", "contraction",
    "atr_ratio", "atr_pctl100", "vol_dry", "up_down_vol",
    "mom10", "mom30", "mom50", "new_hi_cnt10", "gap_max20", "bars_since_hi50",
]

AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


def resample_bars(daily: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Daily OHLCV -> bars of a timeframe, labelled by the END of the period."""
    if tf == "D":
        return daily
    out = daily.resample(_RESAMPLE[tf], label="right", closed="right").agg(AGG)
    return out.dropna(subset=["close"])


def bar_features(b: pd.DataFrame) -> pd.DataFrame:
    """Features at every bar of `b` (open/high/low/close/volume), each using that bar and earlier only."""
    o, h, l, c, v = b["open"], b["high"], b["low"], b["close"], b["volume"].astype(float)
    n = len(b)
    f = pd.DataFrame(index=b.index)

    def sma(s, k):
        return s.rolling(k, min_periods=k).mean()

    s10, s30, s50 = sma(c, N_SHORT), sma(c, N_MID), sma(c, N_LONG)
    f["c_ma10"] = c / s10 - 1                    # extension above the short average
    f["c_ma30"] = c / s30 - 1
    f["c_ma50"] = c / s50 - 1
    f["ma10_30"] = s10 / s30 - 1                 # short over mid: the golden-cross state
    f["ma30_50"] = s30 / s50 - 1
    f["ma50_slope"] = s50 / s50.shift(N_SHORT) - 1   # is the long average rising?

    hi50 = h.rolling(N_LONG, min_periods=N_LONG).max()
    lo50 = l.rolling(N_LONG, min_periods=N_LONG).min()
    hi30 = h.rolling(N_MID, min_periods=N_MID).max()
    lo30 = l.rolling(N_MID, min_periods=N_MID).min()
    hi10 = h.rolling(N_SHORT, min_periods=N_SHORT).max()
    lo10 = l.rolling(N_SHORT, min_periods=N_SHORT).min()
    f["dist_hi50"] = c / hi50 - 1                # <= 0: how far below the 50-bar high
    f["dist_lo50"] = c / lo50 - 1                # how far above the 50-bar low
    f["base_depth30"] = 1 - lo30 / hi30          # depth of the last 30 bars' range
    f["range10"] = (hi10 - lo10) / c             # tightness right now
    f["contraction"] = (hi10 - lo10) / (hi30 - lo30).replace(0, np.nan)   # is the range shrinking?

    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr5, atr30 = tr.rolling(5, min_periods=5).mean(), tr.rolling(N_MID, min_periods=N_MID).mean()
    f["atr_ratio"] = atr5 / atr30.replace(0, np.nan)             # volatility contraction
    f["atr_pctl100"] = rolling_percentile_rank(
        (tr.rolling(N_SHORT, min_periods=N_SHORT).mean() / c).values, lookback=100, min_periods=50)

    f["vol_dry"] = v.rolling(5, min_periods=5).mean() / v.rolling(N_MID, min_periods=N_MID).mean().replace(0, np.nan)
    up = v.where(c > pc, 0.0)
    dn = v.where(c < pc, 0.0)
    f["up_down_vol"] = (up.rolling(N_MID, min_periods=N_MID).sum()
                        / dn.rolling(N_MID, min_periods=N_MID).sum().replace(0, np.nan)).clip(0, 20)   # accumulation vs distribution

    f["mom10"] = c / c.shift(N_SHORT) - 1
    f["mom30"] = c / c.shift(N_MID) - 1
    f["mom50"] = c / c.shift(N_LONG) - 1
    hi_close50 = c.rolling(N_LONG, min_periods=N_LONG).max()
    f["new_hi_cnt10"] = (c >= hi_close50).astype(float).rolling(N_SHORT, min_periods=N_SHORT).sum()
    f["gap_max20"] = (o / pc - 1).abs().rolling(20, min_periods=20).max()

    is_hi = (h >= hi50).to_numpy()
    last_hi = pd.Series(np.where(is_hi, np.arange(n), np.nan)).ffill().to_numpy()
    f["bars_since_hi50"] = np.arange(n) - last_hi
    f.loc[hi50.isna().to_numpy(), "bars_since_hi50"] = np.nan
    return f[BAR_FEATURES].replace([np.inf, -np.inf], np.nan)


# ── observation dates, labels, and the per-ticker dataset ────────────────────────

_ANCHOR = pd.Timestamp("2000-01-07")      # a Friday


def observation_dates(start: str, end: str, every_weeks: int = 4) -> pd.DatetimeIndex:
    """Global Friday observation dates, every `every_weeks`. Fridays so a weekly bar is complete on them."""
    k0 = max(0, int(np.ceil((pd.Timestamp(start) - _ANCHOR).days / (7 * every_weeks))))
    k1 = int((pd.Timestamp(end) - _ANCHOR).days // (7 * every_weeks))
    return pd.DatetimeIndex([_ANCHOR + pd.Timedelta(weeks=every_weeks * k) for k in range(k0, k1 + 1)])


H6, H1, H2, H3 = 126, 252, 504, 756     # six months, one, two and three years of daily bars


def forward_labels(close: pd.Series, idx0: np.ndarray) -> pd.DataFrame:
    """Forward outcomes from the daily bar at each position in idx0. NaN unless the FULL window exists.

    A negative can only be called once the whole window has played out, so partial windows are
    never labelled, even where a positive is already known: labelling early positives and leaving
    unfinished negatives out would inflate the runner rate.
    """
    c = close.to_numpy(dtype=float)
    n = len(c)
    rev = pd.Series(c[::-1])
    fwd_max3 = rev.rolling(H3, min_periods=H3).max().to_numpy()[::-1]       # max of c[t .. t+H3-1]
    # exclude today: max over c[t+1 .. t+H3] = fwd_max3 at t+1
    out = pd.DataFrame(index=np.arange(len(idx0)), dtype=float)
    base = c[idx0]
    nxt = idx0 + 1
    ok3 = nxt + H3 - 1 < n
    fm = np.full(len(idx0), np.nan)
    fm[ok3] = fwd_max3[nxt[ok3]]
    out["fwd_max_3y"] = fm / base - 1
    # six-month outcomes: far less exposed to survivorship than the 3-year ones, because a window this
    # short rarely contains much of a stock's slide to delisting
    fwd_max6 = rev.rolling(H6, min_periods=H6).max().to_numpy()[::-1]
    ok6 = nxt + H6 - 1 < n
    fm6 = np.full(len(idx0), np.nan)
    fm6[ok6] = fwd_max6[nxt[ok6]]
    out["fwd_max_6m"] = fm6 / base - 1
    for name, h in (("fwd_ret_6m", H6), ("fwd_ret_1y", H1), ("fwd_ret_2y", H2), ("fwd_ret_3y", H3)):
        t = idx0 + h
        r = np.full(len(idx0), np.nan)
        ok = t < n
        r[ok] = c[t[ok]] / base[ok] - 1
        out[name] = r
    return out


def hold_at_last_price(close: pd.Series, bars: int = H3 + 5) -> pd.Series:
    """Extend a series past its final bar by holding the last close flat, for LABELS only.

    A delisted stock does not stop existing at its last trade. Someone holding it ends with whatever it
    last traded at (or, for a bankruptcy, usually far less). If observations whose forward window runs
    past that last bar were simply dropped, the data would lose precisely the observations that carry the
    damaging outcomes, and the survivorship bias would survive in the very dataset built to remove it.
    Holding the last close is the neutral choice and is OPTIMISTIC for a bankruptcy; the true terminal
    value is not in a price series.
    """
    last = close.index[-1]
    pad = pd.bdate_range(last + pd.Timedelta(days=1), periods=bars)
    return pd.concat([close, pd.Series(float(close.iloc[-1]), index=pad)])


def build_observations(symbol: str, daily: pd.DataFrame, obs_dates: pd.DatetimeIndex,
                       min_dollar_volume: float = 5e5, with_labels: bool = True,
                       terminal: bool = False) -> Optional[pd.DataFrame]:
    """One row per observation date for one ticker: features on every timeframe, liquidity, labels.

    terminal=True marks a stock known to have been DELISTED (not one whose data merely stops): its
    forward outcomes are computed with the last price held flat (see `hold_at_last_price`)."""
    daily = daily[~daily.index.duplicated(keep="last")].sort_index()
    if len(daily) < 120:
        return None
    close = daily["close"]
    dvol = (close * daily["volume"]).rolling(N_LONG, min_periods=N_LONG).mean()

    i0 = daily.index.searchsorted(obs_dates, side="right") - 1
    valid = i0 >= 0
    age = np.full(len(obs_dates), np.inf)
    age[valid] = (obs_dates[valid] - daily.index[i0[valid]]).days
    keep = valid & (age <= _MAX_BAR_AGE["D"])
    if not keep.any():
        return None
    od, i0 = obs_dates[keep], i0[keep]

    cols = {"symbol": symbol, "date": od, "dvol50": dvol.to_numpy()[i0]}
    for tf in TIMEFRAMES:
        bars = resample_bars(daily, tf)
        # No minimum bar count here, on purpose. A feature goes missing only when ITS OWN window is
        # not yet full (a 50-bar feature on a stock with 30 monthly bars). Gating on the total number
        # of bars made availability depend on how much data exists AFTER the observation date, which
        # is look-ahead even though no feature value was wrong; the truncation test caught it. It also
        # keeps young stocks, where recent runners come from.
        feats = bar_features(bars)
        j = bars.index.searchsorted(od, side="right") - 1
        jv = j >= 0
        bage = np.full(len(od), np.inf)
        bage[jv] = (od[jv] - bars.index[j[jv]]).days
        ok = jv & (bage <= _MAX_BAR_AGE[tf])
        arr = feats.to_numpy()[np.clip(j, 0, None)]
        arr[~ok] = np.nan
        for k, name in enumerate(BAR_FEATURES):
            cols[f"{tf}_{name}"] = arr[:, k]
    df = pd.DataFrame(cols)
    df = df[df["dvol50"] >= min_dollar_volume]
    if df.empty:
        return None
    if with_labels:
        lab_close = hold_at_last_price(close) if terminal else close
        lab = forward_labels(lab_close, i0[(dvol.to_numpy()[i0] >= min_dollar_volume)])
        df = pd.concat([df.reset_index(drop=True), lab], axis=1)
    return df.reset_index(drop=True)


def current_features(symbol: str, daily: pd.DataFrame) -> Optional[pd.Series]:
    """Features as of the LAST available bar, for scanning today. Weekly/monthly use the latest bar
    even if still forming (flagged by `partial_W` / `partial_M`)."""
    daily = daily[~daily.index.duplicated(keep="last")].sort_index()
    if len(daily) < 120:
        return None
    last = daily.index[-1]
    row = {"symbol": symbol, "date": last,
           "dvol50": float((daily["close"] * daily["volume"]).rolling(N_LONG, min_periods=N_LONG).mean().iloc[-1]),
           "close": float(daily["close"].iloc[-1])}
    for tf in TIMEFRAMES:
        bars = resample_bars(daily, tf)
        f = bar_features(bars).iloc[-1]
        for name in BAR_FEATURES:
            row[f"{tf}_{name}"] = f[name]
        if tf != "D":
            row[f"partial_{tf}"] = bool(bars.index[-1] > last)
    return pd.Series(row)


# ── cross-sectional features ─────────────────────────────────────────────────────

RS_SOURCES = ("D_mom50", "W_mom30", "W_mom50", "M_mom30")


def add_relative_strength(obs: pd.DataFrame) -> pd.DataFrame:
    """Percentile rank (0-1) of each momentum feature against every other stock on the SAME date.

    "Is it outperforming the market" is the oldest filter for leaders. It is computed within a
    single date, so it uses no information from any other date and cannot leak the future.
    """
    obs = obs.copy()
    g = obs.groupby("date")
    for src in RS_SOURCES:
        if src in obs.columns:
            obs[f"rs_{src}"] = g[src].rank(pct=True)
    return obs
