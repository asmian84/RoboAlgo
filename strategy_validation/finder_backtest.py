"""
Historical replay of the Finder, so its rule can be put through the validation gate.

The Finder (scanner_engine/finder.py) ranks setups: golden cross, a recent
compressed range, no big gaps, enough liquidity. It prints an entry, stop and
target for each. Nothing in the repo ever measured whether trading those levels
makes money. This module replays the rule day by day over price history and
simulates the orders, so the result can be handed to ``strategy_validation.gates``.

WHAT IS REPLICATED, AND HOW IT IS CHECKED
-----------------------------------------
Features are the project's own: ``IndicatorCalculator`` for ATR, Bollinger and
the moving averages; the rolling percentile rank and compression score are
vectorised copies of ``VolatilityRegimeEngine._rolling_percentile`` and
``RangeCompressionEngine``. ``tests/test_finder_backtest`` checks the copies
against the engines, and checks that a feature at day t is identical whether or
not later bars exist (no look-ahead).

THE ORDER MODEL (stated because a backtest is only as honest as its fills)
--------------------------------------------------------------------------
* A signal on day t (decided at the close) places a buy-stop at ``entry`` (the
  top of the compressed range), valid for the next ``valid_days`` bars.
* Fill: at the open if the bar opens at or above entry (gap-up pays the gap),
  otherwise at ``entry`` if the bar's high reaches it.
* No fill if the bar opens at or beyond the TARGET (nobody buys past the
  target), or if price closes below the STOP before the order triggers.
* After a fill, per bar: open at or below the stop exits at the open; low at or
  below the stop exits at the stop; open at or above the target exits at the
  open; high at or above the target exits at the target. **If one bar could hit
  both, the stop is assumed first.** A time stop exits at the close after
  ``max_hold`` bars.
* One order per setup: a given compressed-range "driver" bar is armed once, so a
  stopped-out setup cannot re-enter on the same stale levels.
* Costs are a flat round-trip fraction supplied by the caller.

KNOWN BIASES (not fixable here, reported alongside every result)
----------------------------------------------------------------
Survivorship (today's listings only), Yahoo's retroactive split adjustment
(distorts historical price bands for reverse-split stocks), and parameters that
were set by judgement from recent data. The result is an UPPER bound.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

RANGE_SHORT = 10
RANGE_LONG = 30
PCT_LOOKBACK = 252
PCT_MIN_PERIODS = PCT_LOOKBACK // 2
BB_WIDTH_THRESHOLD = 0.20
ATR_THRESHOLD = 0.25
MIN_ROWS = RANGE_LONG + PCT_LOOKBACK  # the engine skips instruments with fewer price rows


# ── Rule and execution parameters ─────────────────────────────────────────────

@dataclass(frozen=True)
class Rule:
    """The Finder's screening rule. Mirrors scanner_engine.finder.FinderCriteria."""
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    min_volume: float = 100_000
    max_gap_pct: float = 3.0
    gap_lookback: int = 20
    consolidation_lookback: int = 30
    min_compression_score: float = 50.0

    @classmethod
    def from_criteria(cls, c) -> "Rule":
        return cls(c.min_price, c.max_price, c.min_volume, c.max_gap_pct,
                   c.gap_lookback, c.consolidation_lookback, c.min_compression_score)


@dataclass(frozen=True)
class ExecParams:
    valid_days: int = 10          # how long the buy-stop stays working
    max_hold: int = 30            # time stop, in bars after the fill
    untriggered_only: bool = False  # only signal while close <= entry ("ready to run")
    # False = OPTIMISTIC bound, used only to measure how much the pessimistic default matters:
    # a bar that could hit both stop and target is credited to the target, and a bar that
    # fills mid-range is assumed to have made its low BEFORE the breakout. Never a candidate.
    stop_first: bool = True


# ── Feature math (vectorised copies of the engines) ───────────────────────────

def rolling_percentile_rank(x: np.ndarray, lookback: int = PCT_LOOKBACK,
                            min_periods: int = PCT_MIN_PERIODS) -> np.ndarray:
    """Rank of x[i] inside the trailing window ending at i, as a 0-1 fraction.

    Copy of VolatilityRegimeEngine._rolling_percentile: the fraction of valid
    values in the window strictly below x[i]; NaN when x[i] is NaN or the window
    holds fewer than ``min_periods`` valid values. The window ends at i, so it
    never sees a later bar.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    out = np.full(n, np.nan)
    if n == 0:
        return out
    padded = np.concatenate([np.full(lookback - 1, np.nan), x])
    win = sliding_window_view(padded, lookback)
    last = win[:, -1]
    valid = ~np.isnan(win)
    nvalid = valid.sum(axis=1)
    less = (np.where(valid, win, np.inf) < last[:, None]).sum(axis=1)
    ok = (~np.isnan(last)) & (nvalid >= min_periods)
    out[ok] = less[ok] / nvalid[ok]
    return np.round(out, 4)


_CALC = None


def _calc():
    global _CALC
    if _CALC is None:
        from indicator_engine.calculator import IndicatorCalculator
        _CALC = IndicatorCalculator()
    return _CALC


@dataclass
class Features:
    """Rule-independent per-bar features for one ticker."""
    dates: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    ma50: np.ndarray
    ma200: np.ndarray
    avg_vol20: np.ndarray
    gap_pct: np.ndarray          # |open - prev close| / prev close, in percent
    valid: np.ndarray            # a compression row exists on this bar
    is_comp: np.ndarray
    score: np.ndarray            # compression_score, rounded to 2dp like the engine
    range_high: np.ndarray       # rolling 10-bar high, 4dp
    range_low: np.ndarray        # rolling 10-bar low, 4dp


def compute_features(df: pd.DataFrame) -> Optional[Features]:
    """df: ascending date index with open/high/low/close/volume columns."""
    if len(df) < MIN_ROWS:
        return None
    from config.settings import INDICATOR_PARAMS as P

    c, h, l = df["close"], df["high"], df["low"]
    calc = _calc()
    atr = calc.compute_atr(h, l, c, P["atr_period"])
    _, _, _, bb_width = calc.compute_bollinger(c, P["bb_period"], P["bb_std"])
    ma50 = calc.compute_sma(c, P["sma_short"])
    ma200 = calc.compute_sma(c, P["sma_long"])

    atr_pct = (atr / c.replace(0, np.nan)).round(6)
    atr_rank = rolling_percentile_rank(atr_pct.values)
    bb_rank = rolling_percentile_rank(bb_width.ffill().values)

    r10 = h.rolling(RANGE_SHORT).max() - l.rolling(RANGE_SHORT).min()
    r30 = h.rolling(RANGE_LONG).max() - l.rolling(RANGE_LONG).min()
    ratio = (r10 / r30.replace(0, np.nan)).clip(0, 2)

    n = len(df)
    idx = np.arange(n)
    valid = (idx >= RANGE_LONG) & ~np.isnan(atr_rank) & ~np.isnan(bb_rank) & ~np.isnan(ratio.values)

    trig = ((bb_rank < BB_WIDTH_THRESHOLD).astype(int)
            + (atr_rank < ATR_THRESHOLD).astype(int)
            + (ratio.values < 1.0).astype(int))
    is_comp = valid & (trig >= 2)

    # RangeCompressionEngine._compute_daily_compression_score
    bb_s = pd.Series(bb_rank).fillna(0.5).clip(0, 1)
    atr_s = pd.Series(atr_rank).fillna(0.5).clip(0, 1)
    ratio_s = (r10.fillna(0) / r30.replace(0, np.nan)).clip(0, 2).fillna(1.0)
    range_score = 1 - ratio_s.clip(0, 1)
    score = ((1 - bb_s) * 40 + (1 - atr_s) * 35 + range_score.values * 25).clip(0, 100).round(2)

    prev_close = c.shift(1)
    gap = ((df["open"] - prev_close).abs() / prev_close.where(prev_close > 0) * 100)

    return Features(
        dates=df.index.values,
        open=df["open"].values.astype(float), high=h.values.astype(float),
        low=l.values.astype(float), close=c.values.astype(float),
        volume=df["volume"].values.astype(float),
        ma50=ma50.values, ma200=ma200.values,
        avg_vol20=df["volume"].rolling(20, min_periods=1).mean().values,
        gap_pct=gap.values,
        valid=valid, is_comp=is_comp, score=np.asarray(score),
        range_high=h.rolling(RANGE_SHORT).max().round(4).values,
        range_low=l.rolling(RANGE_SHORT).min().round(4).values,
    )


@dataclass
class SignalSet:
    feats: Features
    signal: np.ndarray       # bool per bar: the Finder would list this ticker
    entry: np.ndarray
    stop: np.ndarray
    target: np.ndarray
    score: np.ndarray        # the driver bar's compression score (the Finder's sort key)
    driver: np.ndarray       # bar index of the compressed "driver" bar, -1 if none


def build_signals(f: Features, rule: Rule) -> SignalSet:
    """Apply a Rule to precomputed features. Every value at bar t uses bars <= t."""
    n = len(f.close)
    c = f.close

    # last `consolidation_lookback` compression ROWS ending at each bar, as the Finder reads them
    rows = np.flatnonzero(f.valid)
    m = len(rows)
    has_cons = np.zeros(n, dtype=bool)
    driver = np.full(n, -1, dtype=np.int64)
    if m:
        k = rule.consolidation_lookback
        masked = np.where(f.is_comp[rows] & (f.score[rows] >= rule.min_compression_score),
                          f.score[rows], -np.inf)
        padded = np.concatenate([np.full(k - 1, -np.inf), masked])
        win = sliding_window_view(padded, k)               # window ending at each compressed row
        best = win.argmax(axis=1)                          # first max, like Python's max()
        found = np.isfinite(win.max(axis=1))
        drv_compact = np.arange(m) - (k - 1) + best        # index into `rows`
        last_row = np.searchsorted(rows, np.arange(n), side="right") - 1   # latest row at or before t
        ok_row = last_row >= 0
        lr = np.where(ok_row, last_row, 0)
        has_cons = ok_row & found[lr]
        driver = np.where(has_cons, rows[np.clip(drv_compact[lr], 0, m - 1)], -1)

    band = np.ones(n, dtype=bool)
    if rule.min_price is not None:
        band &= c >= rule.min_price
    if rule.max_price is not None:
        band &= c <= rule.max_price
    liquid = f.avg_vol20 >= rule.min_volume
    golden = (f.ma50 > f.ma200) & (c > f.ma50)
    gap_max = pd.Series(f.gap_pct).rolling(rule.gap_lookback, min_periods=1).max().fillna(0.0).values
    no_gaps = gap_max <= rule.max_gap_pct
    enough_history = np.arange(n) >= MIN_ROWS - 1

    signal = band & liquid & golden & has_cons & no_gaps & enough_history

    safe_driver = np.where(driver >= 0, driver, 0)
    rh = np.where(driver >= 0, f.range_high[safe_driver], np.nan)
    rl = np.where(driver >= 0, f.range_low[safe_driver], np.nan)
    height = rh - rl
    levels_ok = np.isfinite(rh) & np.isfinite(rl) & (rh > rl)     # scanner_engine.finder.trade_levels
    signal &= levels_ok
    return SignalSet(
        feats=f, signal=signal,
        entry=np.where(levels_ok, rh, np.nan),
        stop=np.where(levels_ok, rl, np.nan),
        target=np.where(levels_ok, rh + height, np.nan),
        score=np.where(driver >= 0, f.score[safe_driver], np.nan),
        driver=driver,
    )


# ── Order simulation ──────────────────────────────────────────────────────────

@dataclass
class Trade:
    symbol: str
    score: float
    sig_i: int
    fill_i: int
    exit_i: int
    fill_px: float
    exit_px: float
    reason: str
    dates: np.ndarray            # the bars held, fill..exit inclusive
    ret: np.ndarray              # gross return for each of those bars (float32)
    entry: float = float("nan")  # the levels the order was placed with, kept so realised
    stop: float = float("nan")   # reward-to-risk can be compared with the displayed 1:1
    target: float = float("nan")

    @property
    def gross(self) -> float:
        return self.exit_px / self.fill_px - 1.0


def simulate_symbol(symbol: str, ss: SignalSet, ex: ExecParams) -> list[Trade]:
    f = ss.feats
    o, h, l, c = f.open, f.high, f.low, f.close
    n = len(c)
    sig_idx = np.flatnonzero(ss.signal)
    used_drivers: set[int] = set()
    trades: list[Trade] = []
    ptr = 0
    while ptr < len(sig_idx):
        t = int(sig_idx[ptr])
        drv = int(ss.driver[t])
        E, S, T = ss.entry[t], ss.stop[t], ss.target[t]
        if drv in used_drivers or (ex.untriggered_only and c[t] > E):
            ptr += 1
            continue
        used_drivers.add(drv)

        # -- working buy-stop
        fill_i, fill_px = -1, np.nan
        last = min(t + ex.valid_days, n - 1)
        resume = last
        d = t + 1
        while d <= last:
            if o[d] >= T:                       # gapped past the target: no trade
                resume = d
                break
            if o[d] >= E:
                fill_i, fill_px = d, o[d]
                break
            if h[d] >= E:
                fill_i, fill_px = d, E
                break
            if c[d] < S:                        # setup failed before it triggered
                resume = d
                break
            d += 1
        if fill_i < 0:
            ptr = int(np.searchsorted(sig_idx, max(resume, t + 1), side="left"))
            continue

        # -- manage the position
        exit_i, exit_px, reason = -1, np.nan, ""
        mid_bar_fill = o[fill_i] < E                     # filled at the entry level, not at a gap-up open
        if ex.stop_first and l[fill_i] <= S:             # same bar could stop us out: assume it did
            exit_i, exit_px, reason = fill_i, S, "stop"
        elif h[fill_i] >= T:
            exit_i, exit_px, reason = fill_i, T, "target"
        elif (not ex.stop_first) and not mid_bar_fill and l[fill_i] <= S:
            exit_i, exit_px, reason = fill_i, S, "stop"
        e = fill_i + 1
        while exit_i < 0 and e < n:
            if o[e] <= S:
                exit_i, exit_px, reason = e, o[e], "stop_gap"
            elif ex.stop_first and l[e] <= S:   # stop checked before target: conservative
                exit_i, exit_px, reason = e, S, "stop"
            elif o[e] >= T:
                exit_i, exit_px, reason = e, o[e], "target_gap"
            elif h[e] >= T:
                exit_i, exit_px, reason = e, T, "target"
            elif l[e] <= S:                     # reached only when the target was credited first
                exit_i, exit_px, reason = e, S, "stop"
            elif e - fill_i >= ex.max_hold:
                exit_i, exit_px, reason = e, c[e], "time"
            else:
                e += 1
        if exit_i < 0:
            exit_i, exit_px, reason = n - 1, c[n - 1], "open_end"

        # -- daily return path, fill..exit
        span = np.arange(fill_i, exit_i + 1)
        path = np.empty(len(span), dtype=np.float64)
        if exit_i == fill_i:
            path[0] = exit_px / fill_px - 1.0
        else:
            path[0] = c[fill_i] / fill_px - 1.0
            if len(span) > 2:
                path[1:-1] = c[span[1:-1]] / c[span[1:-1] - 1] - 1.0
            path[-1] = exit_px / c[exit_i - 1] - 1.0
        trades.append(Trade(symbol, float(ss.score[t]), t, fill_i, exit_i, float(fill_px),
                            float(exit_px), reason, f.dates[span], path.astype(np.float32),
                            float(E), float(S), float(T)))
        ptr = int(np.searchsorted(sig_idx, max(exit_i, t + 1), side="left"))
    return trades


def run_symbol(symbol: str, df: pd.DataFrame, rules: dict[str, Rule],
               variants: dict[str, tuple[str, ExecParams]]) -> dict[str, list[Trade]]:
    """One ticker through every rule and execution variant. Features computed once."""
    f = compute_features(df)
    out = {name: [] for name in variants}
    if f is None:
        return out
    sets = {rn: build_signals(f, r) for rn, r in rules.items()}
    for name, (rule_name, ex) in variants.items():
        out[name] = simulate_symbol(symbol, sets[rule_name], ex)
    return out


# ── Portfolios and statistics ─────────────────────────────────────────────────

def build_portfolio(trades: Iterable[Trade], calendar: np.ndarray, cost_rt: float,
                    max_positions: Optional[int] = None) -> tuple[np.ndarray, list[Trade]]:
    """Daily portfolio return and the trades actually taken.

    max_positions=None: every signal is taken, equal-weighted across whatever is
    open that day (fully invested, no cash drag). It measures the screen's edge
    over the WHOLE universe with no selection. Not tradable as such.

    max_positions=K: at most K positions, each 1/K of equity, filled in the
    Finder's own rank order (highest compression score first) when slots are
    short. Cash earns nothing. This is what a trader could actually run.

    The round-trip cost is charged on the exit bar.
    """
    ordered = sorted(trades, key=lambda t: (t.dates[0], -t.score))
    n = len(calendar)
    total = np.zeros(n)
    count = np.zeros(n)
    taken: list[Trade] = []
    active: list[int] = []        # heap of exit calendar indices
    w = 1.0 / max_positions if max_positions else 1.0
    for t in ordered:
        gi = np.searchsorted(calendar, t.dates)
        if max_positions:
            while active and active[0] < gi[0]:
                heapq.heappop(active)
            if len(active) >= max_positions:
                continue
            heapq.heappush(active, int(gi[-1]))
        r = t.ret.astype(np.float64)
        r[-1] -= cost_rt
        np.add.at(total, gi, r * w)
        np.add.at(count, gi, 1.0)
        taken.append(t)
    if max_positions:
        return total, taken
    return np.divide(total, count, out=np.zeros(n), where=count > 0), taken


def period_stats(daily: np.ndarray, calendar: np.ndarray, taken: list[Trade], cost_rt: float,
                 start: str, end: Optional[str] = None) -> dict:
    """Sharpe / drawdown / hit-rate for one date window (inputs for PerformanceStats)."""
    lo = np.datetime64(start)
    hi = np.datetime64(end) if end else np.datetime64("2262-01-01")
    m = (calendar >= lo) & (calendar <= hi)
    r = daily[m]
    out = {"n_obs": int(m.sum()), "sharpe": None, "max_drawdown_pct": None, "hit_rate": None,
           "n_trades": 0, "skew": 0.0, "kurtosis": 3.0, "total_return_pct": None,
           "gross_expectancy_pct": None, "net_expectancy_pct": None, "profit_factor": None}
    if len(r) > 2 and r.std(ddof=1) > 0:
        out["sharpe"] = float(r.mean() / r.std(ddof=1) * np.sqrt(252))
        from scipy import stats
        out["skew"] = float(stats.skew(r))
        out["kurtosis"] = float(stats.kurtosis(r, fisher=False))
    if len(r):
        eq = np.cumprod(1.0 + r)
        peak = np.maximum.accumulate(eq)
        out["max_drawdown_pct"] = float(((peak - eq) / peak).max() * 100)
        out["total_return_pct"] = float((eq[-1] - 1) * 100)
    inwin = [t for t in taken if lo <= t.dates[-1] <= hi]
    out["n_trades"] = len(inwin)
    if inwin:
        gross = np.array([t.gross for t in inwin])
        net = gross - cost_rt
        out["hit_rate"] = float((net > 0).mean())
        out["gross_expectancy_pct"] = float(gross.mean() * 100)
        out["net_expectancy_pct"] = float(net.mean() * 100)
        wins, losses = net[net > 0].sum(), -net[net < 0].sum()
        out["profit_factor"] = float(wins / losses) if losses > 0 else None
    return out
