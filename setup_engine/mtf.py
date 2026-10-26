"""
Multi-timeframe (weekly and monthly) features, relative strength, and the plumbing the studies share.

``mtf_features(daily, bench_daily)`` turns one stock's daily bars into a WEEKLY table, one row per
COMPLETE week (Friday label), with weekly trend / breakout / volume columns, the last completed
MONTH's trend columns, and relative-strength columns against a benchmark. ``indicators_mtf`` maps
the same columns onto the daily index (next to ``screen.indicators``) so ``screen.Screen`` can
evaluate weekly/monthly filters on any trading day. ``add_xs_ranks`` adds the cross-sectional
momentum ranks a study computes across stocks. ``SIGNATURES`` holds the pre-registered screens.

THE NO-LOOK-AHEAD RULE
----------------------
A weekly or monthly bar is used only once it is COMPLETE. A bar is labelled by the last WEEKDAY
of its period (Friday for a week, the last weekday of the calendar month for a month) and is
complete at evaluation date d iff its label <= d. The rule is calendar-based, so completeness never
depends on rows that do not exist yet: a week whose Friday is a holiday (Good Friday) becomes
usable on the daily table only from the next trading day, which is conservative, never early.
Every column at week label L is a function of daily bars dated <= L only; every rolling window is
NaN until it is full (``min_periods`` equals the window); volume averages use the PRIOR weeks
(shifted by one) so a volume spike cannot lift itself over its own threshold. A benchmark is clipped
to the same ``as_of`` before its bars are built. ``tests/test_setup_engine/test_mtf.py`` rebuilds
every table from data cut at several dates and requires every column to be unchanged.
Columns whose name ends in ``_label`` / ``_last_day`` are dates, not features, and are never
mapped onto the daily index. ``best_run`` is the one function that uses the future, by
construction: it is an EX-POST label for case studies and must never feed a feature.

THE ORDER MODEL
---------------
None here. A weekly signal is known at the weekly close (the label Friday, or the last trading day
of that week); it can be acted on from the NEXT daily open at the earliest. Fills, stops and costs
belong to ``setup_engine.ride`` / ``setup_engine.trade_sim``.

KNOWN BIASES (not fixable here, reported alongside every result)
----------------------------------------------------------------
Survivorship (a universe of today's listings flatters every screen), Yahoo's retroactive split
adjustment (a reverse-split stock shows a price history it never traded at, which distorts the
price and dollar-volume bands in ``is_liquid``), and no dividend adjustment (closes are not total
return). A benchmark whose data stops before ``as_of`` contributes its last available bar of the
final week as if that week were complete (stale, not future); callers clip every series to the
data end first.
"""
from __future__ import annotations

import multiprocessing as mp
import time
from pathlib import Path
from typing import Callable, Iterator, Optional

import numpy as np
import pandas as pd

from setup_engine import screen as S

# ── parameters ────────────────────────────────────────────────────────────────

WEEK_FREQ = "W-FRI"
SMA_W = 30              # weekly SMA window (weeks)
SLOPE_W = 4             # slope of the weekly SMA: sma / sma.shift(SLOPE_W) - 1
SMA_M = 10              # monthly SMA window (months)
RSI_N = 14
ATR_N = 14
VOLX_W = 10             # weekly volume vs the PRIOR 10 weeks
DVOL_W = 13             # median weekly dollar volume window
LOW_RECENT_W = 13       # "higher low": lowest low of the last 13 weeks ...
LOW_PRIOR_W = 26        # ... vs the lowest low of the 26 weeks before those
RS_W = 26               # relative-strength window
RET_SHORT_W = 13        # w_ret13 / xs_rank_ret13
RET_LONG_W = 26         # w_ret26 / xs_rank_ret26
HI_SHORT_W = 26         # w_hi26 / w_prev_hi26 / w_dd26
HI_LONG_W = 52          # w_hi52 / w_prev_hi52 / w_dd52
RS_LONG_W = 52          # rs_prev_hi52
M_HI = 24               # monthly high window (months)
MAX_WEEK_AGE_DAYS = 13  # a daily row never carries a weekly bar older than this (halted / delisted stock)
MAX_MONTH_AGE_DAYS = 62  # a week never carries a monthly bar older than this
LIQ_MIN_PRICE = 2.0
LIQ_MIN_DVOL13 = 5e6
LIQ_MIN_WEEKS = 60
MIN_DAILY_ROWS = 120    # map_universe skips stocks with fewer daily rows
BENCH_DIR = Path.home() / ".roboalgo" / "benchmarks"
BENCHMARKS = ("SPY", "QQQ", "SMH", "TSM", "ASML", "ARM")
BENCH_START = "2000-01-01"
DEFAULT_HISTORY = Path.home() / ".roboalgo" / "history_2000"

MTF_COLUMNS = (
    "w_open", "w_high", "w_low", "w_close", "w_volume", "w_dvol", "w_n_days", "w_n", "w_prev_close",
    "w_ret13", "w_ret26", "w_hi26", "w_prev_hi26", "w_dd26", "w_hi52", "w_prev_hi52", "w_dd52",
    "w_sma30", "w_prev_sma30", "w_slope30", "w_px_sma30", "w_volx", "w_dvol13", "w_rsi14", "w_atr_pct",
    "w_low13", "w_low_prior", "w_higher_low",
    "m_close", "m_prev_close", "m_sma10", "m_prev_sma10", "m_px_sma10", "m_rsi14", "m_hi24", "m_dd24",
    "rs_line", "rs_ret26", "rs_prev_hi26", "rs_prev_hi52", "rs_new_hi26", "rs_new_hi52", "rs_lead26",
    "bm_w_px_sma30", "bm_m_px_sma10",
)
XS_COLUMNS = ("xs_rank_ret13", "xs_rank_ret26")
_DATE_COLUMNS = ("w_last_day", "m_label")
_OHLCV = ("open", "high", "low", "close", "volume")

# Pre-registered screens (written before the event study runs; never edited after). Rule keys and
# options are the vocabulary of setup_engine.screen's multi-timeframe filters.
SIGNATURES = {
    "B0": {"Weekly SMA30": "Price above"},
    "S1": {"Performance Rank (13 Weeks)": "Top 20%", "Weekly SMA30": "Price above",
           "Weekly SMA30 Slope": "Rising"},
    "S2": {"Weekly High (52w)": "30% or more below", "Weekly SMA30": "Price above",
           "Performance Rank (26 Weeks)": "Top 20%"},
    "S3": {"Weekly High (26w)": "New High", "Weekly Volume": "Over 1.5x 10-week average",
           "Weekly SMA30": "Price above", "Weekly SMA30 Slope": "Rising", "Monthly SMA10": "Price above"},
    "S4": {"Weekly High (26w)": "New High", "Weekly Volume": "Over 1.5x 10-week average",
           "Weekly SMA30": "Price above", "Weekly SMA30 Slope": "Rising", "Monthly SMA10": "Price above",
           "RS Line": "New 26-week high", "Benchmark Weekly SMA30": "Price above"},
}
SIGNATURE_LABELS = {
    "B0": "market beta baseline",
    "S1": "RS leader",
    "S2": "beaten-down reclaim",
    "S3": "breakout leader",
    "S4": "full confluence",
}


# ── calendar labels ───────────────────────────────────────────────────────────

def _as_index(dates) -> pd.DatetimeIndex:
    if np.ndim(dates) == 0:
        dates = [dates]
    return pd.DatetimeIndex(dates)


def week_label(dates) -> pd.DatetimeIndex:
    """The Friday ending each date's week (Saturday and Sunday belong to the NEXT week)."""
    return _as_index(dates).to_period(WEEK_FREQ).end_time.normalize()


def month_label(dates) -> pd.DatetimeIndex:
    """The last WEEKDAY of each date's calendar month (a month ending on a weekend rolls back to Friday)."""
    me = _as_index(dates).to_period("M").end_time.normalize()
    return me - pd.to_timedelta(np.maximum(me.weekday - 4, 0), unit="D")


# ── bars ──────────────────────────────────────────────────────────────────────

_BAR_COLUMNS = ["open", "high", "low", "close", "volume", "dvol", "n_days", "last_day"]


def _empty_bars(name: str) -> pd.DataFrame:
    out = pd.DataFrame({"open": [], "high": [], "low": [], "close": [], "volume": [], "dvol": [],
                        "n_days": np.array([], dtype="int64"),
                        "last_day": np.array([], dtype="datetime64[ns]")},
                       index=pd.DatetimeIndex([], name=name))
    return out[_BAR_COLUMNS]


def _bars(daily: pd.DataFrame, labeller: Callable, name: str, as_of) -> pd.DataFrame:
    df = S._as_indexed(daily)
    if len(df) == 0:
        return _empty_bars(name)
    as_of = df.index[-1] if as_of is None else pd.Timestamp(as_of)
    df = df[df.index <= as_of]
    if len(df) == 0:
        return _empty_bars(name)
    c = df["close"].astype("float64")
    tmp = pd.DataFrame({"open": df["open"].astype("float64"), "high": df["high"].astype("float64"),
                        "low": df["low"].astype("float64"), "close": c,
                        "volume": df["volume"].astype("float64"),
                        "dvol": c * df["volume"].astype("float64"),
                        "last_day": df.index})
    g = tmp.groupby(labeller(df.index), sort=True)
    out = g.agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
                volume=("volume", "sum"), dvol=("dvol", "sum"), n_days=("close", "size"),
                last_day=("last_day", "max"))
    out.index = pd.DatetimeIndex(out.index, name=name)
    out = out[out.index <= as_of]
    out["n_days"] = out["n_days"].astype("int64")
    return out[_BAR_COLUMNS]


def weekly_bars(daily: pd.DataFrame, as_of=None) -> pd.DataFrame:
    """Daily bars -> COMPLETE weekly bars, indexed by the Friday label (index name "week").

    Rows of ``daily`` after ``as_of`` (default: the last row) are dropped first, then only labels
    <= ``as_of`` are kept, so the in-progress week never appears. ``daily`` needs a unique,
    increasing DatetimeIndex (``screen.clean_daily`` output); ValueError otherwise.
    """
    return _bars(daily, week_label, "week", as_of)


def monthly_bars(daily: pd.DataFrame, as_of=None) -> pd.DataFrame:
    """Daily bars -> COMPLETE monthly bars, indexed by the last-weekday-of-month label ("month")."""
    return _bars(daily, month_label, "month", as_of)


# ── indicators (copies of screen.indicators' formulas) ────────────────────────

def _wilder_rsi(close: pd.Series) -> pd.Series:
    """Wilder RSI(14), identical to ``screen.indicators``: NaN (not 100) while the average loss is 0."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / RSI_N, min_periods=RSI_N, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / RSI_N, min_periods=RSI_N, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _wilder_atr(h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    """Wilder ATR(14), identical to ``screen.indicators`` (previous close = c.shift(1))."""
    prev_close = c.shift(1)
    tr = pd.concat([h - l, (h - prev_close).abs(), (l - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / ATR_N, min_periods=ATR_N, adjust=False).mean()


def _clean_inf(df: pd.DataFrame, cols) -> pd.DataFrame:
    cols = list(cols)
    df[cols] = df[cols].astype("float64").replace([np.inf, -np.inf], np.nan)
    return df


def weekly_features(wb: pd.DataFrame) -> pd.DataFrame:
    """Weekly bars -> the ``w_*`` columns (plus ``w_last_day``), same index. Row i uses weeks 0..i only."""
    idx = wb.index
    o, h, l, c, v, dv = (wb[k].astype("float64") for k in ("open", "high", "low", "close", "volume", "dvol"))
    f: dict[str, pd.Series] = {}
    f["w_open"], f["w_high"], f["w_low"], f["w_close"], f["w_volume"], f["w_dvol"] = o, h, l, c, v, dv
    f["w_n_days"] = wb["n_days"].astype("float64")
    f["w_n"] = pd.Series(np.arange(1, len(wb) + 1, dtype="float64"), index=idx)
    f["w_prev_close"] = c.shift(1)
    f["w_ret13"] = c / c.shift(RET_SHORT_W) - 1.0
    f["w_ret26"] = c / c.shift(RET_LONG_W) - 1.0
    for k, name in ((HI_SHORT_W, "26"), (HI_LONG_W, "52")):
        hi = c.rolling(k, min_periods=k).max()          # includes this week
        f[f"w_hi{name}"] = hi
        f[f"w_prev_hi{name}"] = hi.shift(1)             # max of the PRIOR k weekly closes
        f[f"w_dd{name}"] = c / hi - 1.0
    sma = c.rolling(SMA_W, min_periods=SMA_W).mean()
    f["w_sma30"] = sma
    f["w_prev_sma30"] = sma.shift(1)
    f["w_slope30"] = sma / sma.shift(SLOPE_W) - 1.0
    f["w_px_sma30"] = c / sma - 1.0
    f["w_volx"] = v / v.rolling(VOLX_W, min_periods=VOLX_W).mean().shift(1)     # PRIOR 10 weeks
    f["w_dvol13"] = dv.rolling(DVOL_W, min_periods=DVOL_W).median()
    f["w_rsi14"] = _wilder_rsi(c)
    f["w_atr_pct"] = _wilder_atr(h, l, c) / c
    low13 = l.rolling(LOW_RECENT_W, min_periods=LOW_RECENT_W).min()
    low_prior = l.rolling(LOW_PRIOR_W, min_periods=LOW_PRIOR_W).min().shift(LOW_RECENT_W)
    f["w_low13"] = low13
    f["w_low_prior"] = low_prior
    a, b = low13.to_numpy(), low_prior.to_numpy()
    f["w_higher_low"] = pd.Series(np.where(np.isnan(a) | np.isnan(b), np.nan, (a > b).astype("float64")),
                                  index=idx)
    names = [k for k in MTF_COLUMNS if k.startswith("w_")]
    out = pd.DataFrame({k: f[k] for k in names}, index=idx)
    out = _clean_inf(out, names)
    out["w_last_day"] = pd.DatetimeIndex(wb["last_day"]) if len(wb) else pd.DatetimeIndex([])
    out.index.name = "week"
    return out


def monthly_features(mb: pd.DataFrame) -> pd.DataFrame:
    """Monthly bars -> the ``m_*`` columns plus ``m_label`` (the month's own label), same index."""
    idx = mb.index
    c = mb["close"].astype("float64")
    sma = c.rolling(SMA_M, min_periods=SMA_M).mean()
    hi = c.rolling(M_HI, min_periods=M_HI).max()
    f = {"m_close": c, "m_prev_close": c.shift(1), "m_sma10": sma, "m_prev_sma10": sma.shift(1),
         "m_px_sma10": c / sma - 1.0, "m_rsi14": _wilder_rsi(c), "m_hi24": hi, "m_dd24": c / hi - 1.0}
    names = [k for k in MTF_COLUMNS if k.startswith("m_")]
    out = _clean_inf(pd.DataFrame({k: f[k] for k in names}, index=idx), names)
    out["m_label"] = pd.DatetimeIndex(idx)
    out.index.name = "month"
    return out


# ── mapping between timeframes ────────────────────────────────────────────────

def _map_positions(src_labels: pd.DatetimeIndex, target: pd.DatetimeIndex, max_age_days: int) -> np.ndarray:
    """Position in ``src_labels`` of the latest label <= each target date; -1 if none or older than max_age_days."""
    n = len(target)
    if len(src_labels) == 0 or n == 0:
        return np.full(n, -1, dtype="int64")
    j = src_labels.searchsorted(target, side="right") - 1
    t = np.asarray(target, dtype="datetime64[ns]").astype("int64")
    s = np.asarray(src_labels, dtype="datetime64[ns]").astype("int64")
    age_days = (t - s[np.maximum(j, 0)]) // 86_400_000_000_000
    return np.where((j >= 0) & (age_days <= max_age_days), j, -1).astype("int64")


def _gather(values: np.ndarray, j: np.ndarray) -> np.ndarray:
    """values[j] row-wise, NaN rows where j < 0."""
    out = np.full((len(j), values.shape[1]), np.nan)
    ok = j >= 0
    out[ok] = values[j[ok]]
    return out


def _gather_dates(labels: pd.DatetimeIndex, j: np.ndarray) -> pd.DatetimeIndex:
    out = np.full(len(j), np.datetime64("NaT"), dtype="datetime64[ns]")
    ok = j >= 0
    out[ok] = np.asarray(labels, dtype="datetime64[ns]")[j[ok]]
    return pd.DatetimeIndex(out)


def _empty_mtf() -> pd.DataFrame:
    out = pd.DataFrame({k: np.array([], dtype="float64") for k in MTF_COLUMNS}, index=pd.DatetimeIndex([], name="week"))
    for k in _DATE_COLUMNS:
        out[k] = pd.DatetimeIndex([])
    return out


def _rs_columns(w_close: pd.Series, w_prev_hi26: pd.Series, bench_wc: pd.Series) -> dict[str, pd.Series]:
    """Relative-strength columns of a stock against a benchmark's weekly close aligned on the same labels."""
    rs = (w_close / bench_wc).replace([np.inf, -np.inf], np.nan)
    prev26 = rs.rolling(RS_W, min_periods=RS_W).max().shift(1)
    prev52 = rs.rolling(RS_LONG_W, min_periods=RS_LONG_W).max().shift(1)
    a, p26, p52 = rs.to_numpy(), prev26.to_numpy(), prev52.to_numpy()
    wc, ph = w_close.to_numpy(), w_prev_hi26.to_numpy()
    with np.errstate(invalid="ignore"):
        new26 = np.where(np.isnan(a) | np.isnan(p26), np.nan, (a > p26).astype("float64"))
        new52 = np.where(np.isnan(a) | np.isnan(p52), np.nan, (a > p52).astype("float64"))
        finite = np.isfinite(a) & np.isfinite(p26) & np.isfinite(wc) & np.isfinite(ph)
        lead = np.where(finite, ((a > p26) & (wc <= ph)).astype("float64"), np.nan)
    idx = w_close.index
    return {"rs_line": rs, "rs_ret26": rs / rs.shift(RS_W) - 1.0, "rs_prev_hi26": prev26, "rs_prev_hi52": prev52,
            "rs_new_hi26": pd.Series(new26, index=idx), "rs_new_hi52": pd.Series(new52, index=idx),
            "rs_lead26": pd.Series(lead, index=idx)}


def mtf_features(daily: pd.DataFrame, bench_daily: Optional[pd.DataFrame] = None, as_of=None) -> pd.DataFrame:
    """One stock's daily bars -> the weekly MTF table (columns ``MTF_COLUMNS`` + ``w_last_day``, ``m_label``).

    One row per COMPLETE week (index = Friday label, name "week"). Each week also carries the LAST
    COMPLETED month (latest month label <= the week label; NaN if that month is more than
    ``MAX_MONTH_AGE_DAYS`` older). With ``bench_daily`` the ``rs_*`` and ``bm_*`` columns are
    filled (the benchmark is clipped to the same ``as_of`` before its bars are built); without it
    they exist and are all NaN. Float columns are float64 with inf replaced by NaN.
    """
    df = S._as_indexed(daily)
    if len(df) == 0:
        return _empty_mtf()
    as_of = df.index[-1] if as_of is None else pd.Timestamp(as_of)
    wf = weekly_features(weekly_bars(df, as_of))
    if len(wf) == 0:
        return _empty_mtf()
    mf = monthly_features(monthly_bars(df, as_of))
    m_names = [k for k in MTF_COLUMNS if k.startswith("m_")]
    jm = _map_positions(mf.index, wf.index, MAX_MONTH_AGE_DAYS)
    mapped = pd.DataFrame(_gather(mf[m_names].to_numpy(dtype="float64"), jm), index=wf.index, columns=m_names)

    cols: dict[str, pd.Series] = {k: wf[k] for k in wf.columns}
    cols.update({k: mapped[k] for k in m_names})
    cols["m_label"] = pd.Series(_gather_dates(mf.index, jm), index=wf.index)
    nan = pd.Series(np.nan, index=wf.index)
    rs_names = [k for k in MTF_COLUMNS if k.startswith("rs_")]
    cols.update({k: nan for k in rs_names + ["bm_w_px_sma30", "bm_m_px_sma10"]})
    if bench_daily is not None:
        bb = S._as_indexed(bench_daily)
        bb = bb[bb.index <= as_of]
        bw = weekly_features(weekly_bars(bb, as_of))
        if len(bw):
            bench_wc = bw["w_close"].reindex(wf.index)
            cols.update(_rs_columns(wf["w_close"], wf["w_prev_hi26"], bench_wc))
            cols["bm_w_px_sma30"] = bw["w_px_sma30"].reindex(wf.index)
            bmf = monthly_features(monthly_bars(bb, as_of))
            jb = _map_positions(bmf.index, wf.index, MAX_MONTH_AGE_DAYS)
            cols["bm_m_px_sma10"] = pd.Series(_gather(bmf[["m_px_sma10"]].to_numpy(dtype="float64"), jb)[:, 0],
                                              index=wf.index)
    float_cols = list(MTF_COLUMNS)
    out = pd.DataFrame({k: cols[k] for k in float_cols + list(_DATE_COLUMNS)}, index=wf.index)
    out = _clean_inf(out, float_cols)
    out.index.name = "week"
    return out


def indicators_mtf(daily: pd.DataFrame, bench: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """``screen.indicators(daily)`` joined with ``MTF_COLUMNS`` mapped onto the daily index (float64).

    For date d the weekly row is the latest COMPLETE week (label <= d); NaN when there is none or
    it is more than ``MAX_WEEK_AGE_DAYS`` old. The weekly table is built with ``as_of`` = the last
    daily row, so a Friday row carries its own week and a Thursday row carries the week before.
    """
    ind = S.indicators(daily)
    wf = mtf_features(daily, bench)
    j = _map_positions(wf.index, ind.index, MAX_WEEK_AGE_DAYS)
    mapped = pd.DataFrame(_gather(wf[list(MTF_COLUMNS)].to_numpy(dtype="float64"), j),
                          index=ind.index, columns=list(MTF_COLUMNS))
    return pd.concat([ind, mapped], axis=1)


# ── liquidity and cross-sectional ranks ───────────────────────────────────────

def is_liquid(df: pd.DataFrame) -> np.ndarray:
    """Weekly liquidity filter: price >= LIQ_MIN_PRICE, median 13-week dollar volume >= LIQ_MIN_DVOL13,
    at least LIQ_MIN_WEEKS weekly bars of history. NaN never passes."""
    c = df["w_close"].to_numpy(dtype="float64")
    d = df["w_dvol13"].to_numpy(dtype="float64")
    n = df["w_n"].to_numpy(dtype="float64")
    with np.errstate(invalid="ignore"):
        return (np.isfinite(c) & np.isfinite(d) & np.isfinite(n)
                & (c >= LIQ_MIN_PRICE) & (d >= LIQ_MIN_DVOL13) & (n >= LIQ_MIN_WEEKS))


def add_xs_ranks(panel: pd.DataFrame, week_col: str = "week", liquid_col: str = "liquid") -> pd.DataFrame:
    """Copy of ``panel`` with ``xs_rank_ret13`` / ``xs_rank_ret26``.

    Within each week the rank (percentile, ties averaged) of ``w_ret13`` / ``w_ret26`` among rows
    that are liquid AND have a finite value; every other row is NaN, so an illiquid stock neither
    gets a rank nor moves anyone else's.
    """
    out = panel.copy()
    liq = out[liquid_col].fillna(False).to_numpy(dtype=bool)
    weeks = out[week_col].to_numpy()
    for src, dst in (("w_ret13", "xs_rank_ret13"), ("w_ret26", "xs_rank_ret26")):
        vals = out[src].to_numpy(dtype="float64")
        use = np.flatnonzero(liq & np.isfinite(vals))
        res = np.full(len(out), np.nan)
        if len(use):
            tmp = pd.DataFrame({"w": weeks[use], "v": vals[use]})
            res[use] = tmp.groupby("w")["v"].rank(pct=True, method="average").to_numpy()
        out[dst] = res
    return out


# ── ex-post run label (uses the future by construction) ───────────────────────

def best_run(close: pd.Series, start, min_weeks: int = 52) -> Optional[dict]:
    """The largest low-to-later-peak run in a WEEKLY close series since ``start``, or None.

    THIS IS AN EX-POST LABEL: it picks the low that has the highest later peak, so it uses the
    future by construction. It is for case studies ("what did the winners look like at their low")
    and must never be a feature or a signal.

    The low is the week with the largest ratio (best later close / its close); the peak is the
    highest close from that low on. ``dd_into_low`` is the low relative to the highest close in the
    730 days before it (None if there is none); ``max_gap_days`` is the largest gap between
    consecutive weekly labels inside the run (a large value flags halted or stitched data).
    """
    close = close.astype("float64").dropna()
    wc = close[close.index >= pd.Timestamp(start)]
    if len(wc) < min_weeks or len(wc) == 0:
        return None
    arr = wc.to_numpy()
    fut_max = np.maximum.accumulate(arr[::-1])[::-1]
    ratio = np.r_[fut_max[1:] / arr[:-1], 1.0]
    li = int(np.argmax(ratio))
    pk = li + int(np.argmax(arr[li:]))
    run = wc.iloc[li: pk + 1]
    low_week, low_close = wc.index[li], float(arr[li])
    prior = close[(close.index < low_week) & (close.index >= low_week - pd.Timedelta(days=730))]
    gaps = np.diff(np.asarray(run.index, dtype="datetime64[ns]").astype("int64")) // 86_400_000_000_000
    return {"low_week": low_week, "low_close": low_close,
            "peak_week": wc.index[pk], "peak_close": float(arr[pk]),
            "mult": float(arr[pk] / low_close), "weeks": int(pk - li),
            "mdd_in_run": float((run / run.cummax() - 1.0).min()),
            "dd_into_low": float(low_close / prior.max() - 1.0) if len(prior) else None,
            "max_gap_days": int(gaps.max()) if len(gaps) else 0}


# ── benchmarks ────────────────────────────────────────────────────────────────

def _yf_fetch(symbol: str, start: str) -> pd.DataFrame:
    """Daily OHLCV for one symbol from Yahoo (split-adjusted, not dividend-adjusted), lowercase columns."""
    import yfinance as yf

    raw = yf.Ticker(symbol).history(start=start, auto_adjust=False, actions=False)
    df = raw[["Open", "High", "Low", "Close", "Volume"]].rename(columns=str.lower)
    df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
    df.index.name = "date"
    return df.dropna(subset=["close"])


def ensure_benchmarks(symbols=BENCHMARKS, dest=BENCH_DIR, start: str = BENCH_START,
                      fetch: Optional[Callable[[str, str], pd.DataFrame]] = None) -> dict[str, Path]:
    """Make sure ``dest/<SYM>.parquet`` exists for every symbol; fetch only the missing ones.

    When every file exists nothing is imported or fetched. ``fetch(symbol, start)`` returns a daily
    frame (DatetimeIndex; open/high/low/close/volume); the default is Yahoo via ``_yf_fetch``.
    ValueError("no data for SYM") if a fetch comes back empty.
    """
    dest = Path(dest)
    paths: dict[str, Path] = {}
    for sym in symbols:
        path = dest / f"{sym}.parquet"
        if not path.exists():
            df = (fetch or _yf_fetch)(sym, start)
            if df is None or len(df) == 0:
                raise ValueError(f"no data for {sym}")
            out = df[list(_OHLCV)].astype("float64").copy()
            out.index = pd.DatetimeIndex(out.index)
            out.index.name = "date"
            dest.mkdir(parents=True, exist_ok=True)
            out.to_parquet(path)
        paths[sym] = path
    return paths


def load_benchmark(symbol: str, dest=BENCH_DIR, end=None) -> pd.DataFrame:
    """A benchmark's daily bars (cleaned), clipped to dates <= ``end``."""
    df = pd.read_parquet(Path(dest) / f"{symbol}.parquet")
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index(pd.to_datetime(df["date"])).drop(columns="date")
    df = S.clean_daily(df[list(_OHLCV)].astype("float64"))
    if end is not None:
        df = df[df.index <= pd.Timestamp(end)]
    return df


# ── the daily cache ───────────────────────────────────────────────────────────

def _batch_files(history, limit_batches: Optional[int] = None) -> list[Path]:
    files = sorted(Path(history).expanduser().glob("batch_*.parquet"))
    return files[:limit_batches] if limit_batches is not None else files


def _naive_dates(series: pd.Series) -> pd.Series:
    d = pd.to_datetime(series)
    return d.dt.tz_convert(None) if d.dt.tz is not None else d


def load_daily(history, symbols) -> dict[str, pd.DataFrame]:
    """Cleaned daily bars for the given symbols, read from ALL batch files of the cache."""
    import pyarrow.parquet as pq

    want = list(symbols)
    parts: dict[str, list[pd.DataFrame]] = {}
    for f in _batch_files(history):
        tbl = pq.read_table(f, filters=[("symbol", "in", want)])
        if not tbl.num_rows:
            continue
        raw = tbl.to_pandas()
        raw["date"] = _naive_dates(raw["date"])
        for sym, g in raw.groupby("symbol"):
            parts.setdefault(sym, []).append(g.set_index("date")[list(_OHLCV)].astype("float64"))
    return {sym: S.clean_daily(pd.concat(p)) for sym, p in parts.items()}


def data_end(history, limit_batches: Optional[int] = None) -> pd.Timestamp:
    """The last date in the (optionally limited) cache, reading only the date column."""
    import pyarrow.parquet as pq

    best = None
    for f in _batch_files(history, limit_batches):
        d = _naive_dates(pq.read_table(f, columns=["date"]).to_pandas()["date"])
        if len(d):
            m = d.max()
            best = m if best is None or m > best else best
    if best is None:
        raise ValueError(f"no batch files under {history}")
    return pd.Timestamp(best)


def _universe_batch(args) -> list:
    """Worker: (path, fn, fn_args) -> the non-None ``fn(symbol, daily, *fn_args)`` results of one batch."""
    path, fn, fn_args = args
    raw = pd.read_parquet(path)
    raw["date"] = _naive_dates(raw["date"])
    out = []
    for sym, g in raw.groupby("symbol"):
        daily = S.clean_daily(g.set_index("date")[list(_OHLCV)].astype("float64"))
        if len(daily) < MIN_DAILY_ROWS:
            continue
        r = fn(sym, daily, *fn_args)
        if r is not None:
            out.append(r)
    return out


def map_universe(history, fn: Callable, fn_args: tuple = (), workers: int = 1,
                 limit_batches: Optional[int] = None, progress: bool = True) -> Iterator[list]:
    """Run ``fn(symbol, daily, *fn_args)`` over every stock of the cache; yield one list of results per batch.

    Stocks with fewer than ``MIN_DAILY_ROWS`` cleaned rows are skipped; ``None`` results are dropped.
    ``workers <= 1`` maps sequentially; otherwise a fork pool is used and batches arrive in completion
    order (``fn`` must be a picklable module-level function).
    """
    files = _batch_files(history, limit_batches)
    tasks = [(str(f), fn, tuple(fn_args)) for f in files]
    t0 = time.time()
    pool = None
    if workers <= 1:
        it = map(_universe_batch, tasks)
    else:
        pool = mp.get_context("fork").Pool(workers)
        it = pool.imap_unordered(_universe_batch, tasks)
    try:
        for k, res in enumerate(it, 1):
            if progress:
                print(f"  {k}/{len(tasks)} batches ({time.time() - t0:.0f}s)", flush=True)
            yield res
    except BaseException:
        if pool is not None:
            pool.terminate()
        raise
    else:
        if pool is not None:
            pool.close()
    finally:
        if pool is not None:
            pool.join()
