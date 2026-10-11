"""
Does buying a beaten-down stock on its breakout day make money? One pre-registered test over the whole
price cache.

    python scripts/study_turnaround.py                          # full study, writes docs/turnaround_study_results.json
    python scripts/study_turnaround.py --limit-batches 2 --no-ledger    # smoke test on two batches, writes nothing real
    python scripts/study_turnaround.py --inspect NAUT           # also print the signals and trades of one stock

THE QUESTION. A stock is in a BOTTOM state when it sits 70% or more below its 2-year high AND its lowest
low of the last 126 bars is above the lowest low of the 252 bars before those (it stopped making new
lows: a "higher low"). On a TRIGGER day it closes at a new 20-day high on more than twice its usual
volume. C0 is every trigger day that follows a bottom-state day, bought at the next open and managed
with an ATR stop, a trailing stop and a time stop, in R-multiples, NET of a flat cost.

PRE-REGISTERED. Every constant, the 12 configurations and the verdict rule below are fixed before any
result is seen. Nothing is tuned. The primary configuration decides the verdict; the other eleven only
show whether the answer is fragile (they never change the verdict). One record of the plan and one per
configuration are appended to the trial ledger (setup_engine/registry/trials.jsonl) on a full run, so
the count of attempts stays honest.

CONTROLS (the point of the design). A strategy that "works" in a rising market proves nothing, so the
treatment is compared with trades that could have been picked on the SAME DATE, with the same exit
rule and the same cost, so the market regime cancels:
    C0   trigger day AND bottom state on the prior bar        (the treatment, one position at a time)
    C1   trigger day WITHOUT a bottom state                   (does the bottom add anything to the trigger?)
    C2   bottom state, NO trigger, same date                  (is it the trigger or just the beaten-down stock?)
    C3   every liquid stock on the same date                  (the market that day)
    C3v  same date AND same ATR% bucket                       (the market that day, for stocks this volatile)
Control pools hold EVERY eligible stock-day as an independent trade (no cooldown), as sums and counts per
(date, volatility bucket). Each treatment trade's excess is its net R minus its pool's mean net R.

NO LOOK-AHEAD. The BOTTOM state is read on the bar BEFORE the trigger (bottom_lag=1): the breakout bar
itself lifts the close and would otherwise help to leave the bottom state (the configuration
bottom_same_bar reproduces the original, same-bar brief to show the difference). The volatility bucket
uses ATR% of the bar before the signal. Liquidity uses the prior 20 bars. Entry is the next open.
Indicators are tested to be unchanged when later bars are cut off (setup_engine/screen.py).

ORDER MODEL. See setup_engine/trade_sim.py: entry at the next open, stop-first on a bar that could hit
both stop and target, gaps filled at the open, flat round-trip cost. The signal window ends 2026-03-31
so that a 120-bar trade has closed before the data ends. A trade still open at the data end ("end") is
censored, so it is counted and EXCLUDED from every headline statistic, control pool and verdict input,
and the output says how many there were.

CLUSTERING. Trades opened in the same month share one market, so confidence intervals resample whole
calendar months, whole half-years, or whole stocks (setup_engine/trade_stats.py). They are still
somewhat too narrow, because overlapping holds and regimes link neighbouring months.

VERDICT RULE (applied to the primary configuration only; implemented in ``verdict``):
    'edge' only if ALL of
      (1) C0 net E[R] > 0;
      (2) the C0-minus-C3v excess has a 95% interval whose LOWER bound is > 0 under BOTH month and
          half-year clustering;
      (3) the C0-minus-C2 excess mean is > 0;
      (4) adjusted_expectancy(E, f_ref, -1.3) > 0, where f_ref is the C0-trade-weighted share of
          listed common stocks that are missing from today's-listings price data (f_stock), over signal
          years 2013 onward. E is the C0 net E[R]; f_ref of the trades is assumed to end at -1.3R.
    otherwise 'no demonstrated edge'. If an input cannot be computed: 'incomplete (missing: ...)'.

KNOWN BIASES (not fixable here, reported with every result). Survivorship: the price cache holds only
today's listings, so every average is flattered; the survivorship block sizes the hole and the
adjusted expectancy charges for it, and the stock-count share is a LOWER bound for a screen that selects
beaten-down stocks. Yahoo adjusts old prices retroactively for splits (reverse-split stocks show price
bands they never traded at). No dividend adjustment. Stops are assumed fillable inside the bar, which
flatters thin stocks. The result is an UPPER bound.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import multiprocessing as mp
import os
import sys
import time
import warnings
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from setup_engine import learn as L
from setup_engine import trade_stats as TS
from setup_engine.screen import Screen, clean_daily, indicators
from setup_engine.trade_sim import REASONS, TRADE_COLUMNS, ExitModel, simulate, simulate_each

# ── pre-registered choices (fixed before any result is seen) ──────────────────────────────────────

DEFAULT_HISTORY = Path.home() / ".roboalgo" / "history_2000"
DELISTED_CSV = Path.home() / ".roboalgo" / "delisted" / "av_delisted.csv"
OUT_JSON = ROOT / "docs" / "turnaround_study_results.json"
SMOKE_OUT = Path.home() / ".roboalgo" / "turnaround" / "smoke_results.json"
TRADES_OUT = Path.home() / ".roboalgo" / "turnaround" / "trades.parquet"

SIGNAL_START = "2001-01-01"
SIGNAL_END = "2026-03-31"          # all primary trades resolve inside the data (120 bars before 2026-10-06 is ~2026-04-14)
ERAS = {"2001-2009": ("2001-01-01", "2009-12-31"),
        "2010-2017": ("2010-01-01", "2017-12-31"),
        "2018-2026": ("2018-01-01", "2026-03-31")}

N_BOOT = 2000
SEED = 20261007
ALPHA = 0.05
L_SCENARIOS = (-1.0, -1.3)         # total-loss outcome of a trade in a delisted stock, in R
LEDGER_STAGE = "turnaround_study"
INSPECT = ("NAUT",)

BOTTOM = {"2-Year High": "70% or more below", "Monthly Low": "Higher low"}
TRIGGER = {"20-Day High": "New High", "Relative Volume": "Over 2"}
LIQUIDITY = {"Price": "Over $1", "Dollar Volume 20d": "Over 1M"}
EXIT_PRIMARY = ExitModel(2.0, 3.0, None, 120, 0.01)
EXIT_SECONDARY = ExitModel(2.0, None, 3.0, 60, 0.01)

F_REF_FROM_YEAR = 2013             # the delisted list is thin before this year
MIN_ROWS = 30                      # a stock needs this many clean bars to be processed at all


@dataclass(frozen=True)
class Config:
    name: str
    role: str                      # primary | secondary | cost | stability
    bottom: dict
    trigger: dict
    liquidity: dict
    exit: ExitModel
    bottom_lag: int = 1            # bars between the bottom-state reading and the trigger bar
    cooldown: int = 20             # bars after an exit before the next treatment trade may open


_PRIMARY = Config("primary", "primary", BOTTOM, TRIGGER, LIQUIDITY, EXIT_PRIMARY)
CONFIGS = (
    _PRIMARY,
    replace(_PRIMARY, name="exit_target3r_60", role="secondary", exit=EXIT_SECONDARY),
    replace(_PRIMARY, name="cost_0.5pct", role="cost", exit=replace(EXIT_PRIMARY, cost_pct=0.005)),
    replace(_PRIMARY, name="cost_2pct", role="cost", exit=replace(EXIT_PRIMARY, cost_pct=0.02)),
    replace(_PRIMARY, name="bottom_60pct", role="stability",
            bottom={"2-Year High": "60% or more below", "Monthly Low": "Higher low"}),
    replace(_PRIMARY, name="bottom_80pct", role="stability",
            bottom={"2-Year High": "80% or more below", "Monthly Low": "Higher low"}),
    replace(_PRIMARY, name="relvol_1.5", role="stability",
            trigger={"20-Day High": "New High", "Relative Volume": "Over 1.5"}),
    replace(_PRIMARY, name="relvol_3", role="stability",
            trigger={"20-Day High": "New High", "Relative Volume": "Over 3"}),
    replace(_PRIMARY, name="liquidity_5M", role="stability",
            liquidity={"Price": "Over $1", "Dollar Volume 20d": "Over 5M"}),
    replace(_PRIMARY, name="no_price_filter", role="stability", liquidity={"Dollar Volume 20d": "Over 1M"}),
    replace(_PRIMARY, name="bottom_same_bar", role="stability", bottom_lag=0),
    replace(_PRIMARY, name="cooldown_0", role="stability", cooldown=0),
)

VERDICT_L = L_SCENARIOS[-1]        # -1.3
VERDICT_RULE = (
    "'edge' only if ALL of: (1) C0 net E[R] > 0; (2) the C0-minus-C3v excess has a 95% CI whose lower "
    "bound is > 0 under BOTH month and half-year clustering; (3) the C0-minus-C2 excess mean is > 0; "
    f"(4) adjusted_expectancy(E, f_ref, {VERDICT_L}) > 0, where f_ref is the C0-trade-weighted share of "
    f"listed common stocks missing from the price data (f_stock) over signal years >= {F_REF_FROM_YEAR}. "
    "Otherwise 'no demonstrated edge'. A missing input gives 'incomplete (missing: ...)'. "
    "Stability configurations never change the verdict."
)

_END = REASONS.index("end")


# ── masks ─────────────────────────────────────────────────────────────────────────────────────────

def _lag(a, k: int) -> np.ndarray:
    """a shifted k bars later (the value at bar t is a[t-k]); the first k bars are False."""
    a = np.asarray(a, dtype=bool)
    if k < 0:
        raise ValueError("lag must be >= 0")
    if k == 0:
        return a.copy()
    out = np.zeros_like(a)
    out[k:] = a[:-k]
    return out


def arm_masks(ind: pd.DataFrame, cfg: Config, window) -> dict:
    """Boolean arrays over the bars of ``ind`` for each arm. ``window`` = (start, end) or None.

    base is every bar where the bottom reading (on the prior bar), the trigger columns and the
    liquidity columns are all valid, ATR is positive, a next bar exists and the bar is in the signal
    window. Negations (~B_lag, ~T) are safe ONLY because base includes the validity masks: NaN never
    reaches a negation.
    """
    Bs, Ts, Ls = Screen(cfg.bottom), Screen(cfg.trigger), Screen(cfg.liquidity)
    B, Bv = Bs.mask(ind), Bs.valid(ind)
    B_lag, Bv_lag = _lag(B, cfg.bottom_lag), _lag(Bv, cfg.bottom_lag)
    T, Tv, Lq = Ts.mask(ind), Ts.valid(ind), Ls.mask(ind)
    n = len(ind)
    has_next = np.ones(n, dtype=bool)
    has_next[-1:] = False
    if window is None:
        in_window = np.ones(n, dtype=bool)
    else:
        lo, hi = pd.Timestamp(window[0]), pd.Timestamp(window[1])
        in_window = np.asarray((ind.index >= lo) & (ind.index <= hi))
    atr_ok = ind["atr14"].to_numpy(dtype=float) > 0.0
    base = Bv_lag & Tv & Lq & atr_ok & has_next & in_window
    return {"C0": base & B_lag & T,
            "C1": base & ~B_lag & T,
            "C2pool": base & B_lag & ~T,
            "C3pool": base,
            "bottom": B & Bv,
            "liquid": Lq}


# ── per-stock and per-batch processing ────────────────────────────────────────────────────────────

def _empty_trades() -> pd.DataFrame:
    return pd.DataFrame(columns=list(TRADE_COLUMNS) + ["config", "arm", "vol_bucket"])


def _exit_key(m: ExitModel) -> tuple:
    return (m.stop_atr, m.trail_atr, m.target_r, m.max_bars)


def process_stock(symbol, daily: pd.DataFrame, configs, acc: "TS.PoolAccumulator", window=None) -> dict:
    """One stock: simulate C0/C1 trades for every config and add the C2/C3 control pools to ``acc``.

    Returns {"trades", "span": (symbol, first, last) or None, "bottom_years": set, "pool_end": {config: n}}.
    """
    out = {"trades": _empty_trades(), "span": None, "bottom_years": set(), "pool_end": {}}
    df = clean_daily(daily)
    if len(df) < MIN_ROWS:
        return out
    ind = indicators(df)
    n = len(ind)
    out["span"] = (symbol, ind.index[0], ind.index[-1])
    masks = {cfg.name: arm_masks(ind, cfg, window) for cfg in configs}
    # bucket on the bar BEFORE the signal, so the trigger bar's own range cannot move the stock to a
    # higher-volatility bucket than the one it was in when the setup formed
    bucket = TS.vol_bucket(ind["atr_pct"].shift(1).to_numpy(dtype=float))
    day = TS.day_index(ind.index)

    # one independent-trade simulation per distinct exit rule, over the union of the pools that use it
    groups: dict = {}
    for cfg in configs:
        groups.setdefault(_exit_key(cfg.exit), []).append(cfg)
    each = {}
    for key, cfgs in groups.items():
        union = np.zeros(n, dtype=bool)
        for c in cfgs:
            union |= masks[c.name]["C3pool"]
        each[key] = simulate_each(ind, union, cfgs[0].exit)

    parts = []
    for cfg in configs:
        m, e = masks[cfg.name], each[_exit_key(cfg.exit)]
        # NET R with this config's own cost: the controls and the treatment pay the same cost
        r_net = e["r_gross"] - cfg.exit.cost_pct * e["inv_risk"]
        ok = e["valid"] & np.isfinite(r_net)
        resolved = ok & (e["reason"] != _END)
        for arm, pool_mask in (("C2", m["C2pool"]), ("C3", m["C3pool"])):
            sel = pool_mask & resolved & (bucket >= 0)
            acc.add(f"{cfg.name}|{arm}", day[sel], bucket[sel], r_net[sel])
        out["pool_end"][cfg.name] = int((m["C3pool"] & ok & ~resolved).sum())
        for arm in ("C0", "C1"):
            idx = np.flatnonzero(m[arm])
            if len(idx) == 0:
                continue
            tr = simulate(ind, idx, cfg.exit, cfg.cooldown, symbol)
            if tr.empty:
                continue
            tr["config"] = cfg.name
            tr["arm"] = arm
            tr["vol_bucket"] = bucket[tr["signal_i"].to_numpy()]
            parts.append(tr)
    if parts:
        out["trades"] = pd.concat(parts, ignore_index=True)

    pc = next((c for c in configs if c.role == "primary"), configs[0])
    mp_ = masks[pc.name]
    out["bottom_years"] = {int(y) for y in np.unique(ind.index.year[mp_["bottom"] & mp_["liquid"]])}
    return out


def process_frames(frames, configs, window=None) -> dict:
    """Process an iterable of (symbol, daily frame) into one batch result."""
    acc = TS.PoolAccumulator()
    trades, spans = [], []
    bottom_counts, pool_end = Counter(), Counter()
    n_stocks = 0
    for symbol, df in frames:
        r = process_stock(symbol, df, configs, acc, window)
        if r["span"] is None:
            continue
        n_stocks += 1
        spans.append(r["span"])
        bottom_counts.update(r["bottom_years"])
        pool_end.update(r["pool_end"])
        if len(r["trades"]):
            trades.append(r["trades"])
    return {"trades": pd.concat(trades, ignore_index=True) if trades else _empty_trades(),
            "acc": acc,
            "spans": pd.DataFrame(spans, columns=["symbol", "first_date", "last_date"]),
            "bottom_counts": bottom_counts, "n_stocks": n_stocks, "pool_end": pool_end}


def process_batch(args) -> dict:
    """Worker: (path, configs, window) -> batch result. Reads one parquet batch file."""
    path, configs, window = args
    raw = pd.read_parquet(path)
    raw["date"] = pd.to_datetime(raw["date"])
    if raw["date"].dt.tz is not None:
        raw["date"] = raw["date"].dt.tz_convert(None)
    frames = ((sym, g.set_index("date").drop(columns="symbol").astype(float)) for sym, g in raw.groupby("symbol"))
    return process_frames(frames, configs, window)


def merge(results) -> dict:
    """Combine batch results (any iterable, consumed once; pool arrays are added in place as they arrive)."""
    acc = TS.PoolAccumulator()
    trades, spans = [], []
    bottom_counts, pool_end = Counter(), Counter()
    n_stocks = n_batches = 0
    for r in results:
        acc.merge(r["acc"])
        if len(r["trades"]):
            trades.append(r["trades"])
        if len(r["spans"]):
            spans.append(r["spans"])
        bottom_counts.update(r["bottom_counts"])
        pool_end.update(r["pool_end"])
        n_stocks += r["n_stocks"]
        n_batches += 1
    return {"trades": pd.concat(trades, ignore_index=True) if trades else _empty_trades(),
            "acc": acc,
            "spans": pd.concat(spans, ignore_index=True) if spans else pd.DataFrame(columns=["symbol", "first_date", "last_date"]),
            "bottom_counts": bottom_counts, "n_stocks": n_stocks, "pool_end": pool_end, "n_batches": n_batches}


# ── analysis ──────────────────────────────────────────────────────────────────────────────────────

def _excess(r, day, pool, bucket=None) -> np.ndarray:
    """Each trade's net R minus the mean net R of the control pool on ITS OWN date (and bucket)."""
    return TS.excess_vs_pool(r, day, pool, bucket)


def config_spec(cfg: Config) -> dict:
    return {"name": cfg.name, "role": cfg.role, "bottom": dict(cfg.bottom), "trigger": dict(cfg.trigger),
            "liquidity": dict(cfg.liquidity), "exit": dataclasses.asdict(cfg.exit),
            "bottom_lag": cfg.bottom_lag, "cooldown": cfg.cooldown}


def _cis(x, month, half, stock, n_boot, seed) -> dict:
    return {"month": TS.cluster_bootstrap(x, month, n_boot, seed, ALPHA),
            "half": TS.cluster_bootstrap(x, half, n_boot, seed, ALPHA),
            "stock": TS.cluster_bootstrap(x, stock, n_boot, seed, ALPHA)}


def _finite_mean(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else None


def _slice_table(t: pd.DataFrame, labels, ex_v) -> dict:
    out = {}
    for lab in sorted(set(labels)):
        s = np.asarray(labels) == lab
        r = t["r_net"].to_numpy(dtype=float)[s]
        out[str(lab)] = {"n": int(s.sum()), "expectancy": _finite_mean(r),
                         "win_rate": float((r[np.isfinite(r)] > 0).mean()) if np.isfinite(r).any() else None,
                         "excess_C3v": _finite_mean(ex_v[s])}
    return out


def _arm_block(t: pd.DataFrame, acc, name: str, arm: str, n_boot: int, seed: int) -> tuple:
    """Headline numbers for one arm of one config, EXCLUDING trades still open at the data end.

    Returns (block, resolved trades, excess arrays {"C3","C3v","C2"} aligned to the resolved trades).
    """
    t = t.copy()
    t["signal_date"] = pd.to_datetime(t["signal_date"])
    open_ = t["reason"].to_numpy() == "end"
    r_all = t["r_net"].to_numpy(dtype=float)
    res = t[~open_].reset_index(drop=True)
    r = res["r_net"].to_numpy(dtype=float)
    dates = res["signal_date"]
    day = TS.day_index(dates) if len(res) else np.zeros(0, dtype=np.int64)
    bkt = res["vol_bucket"].to_numpy(dtype=np.int64) if len(res) else np.zeros(0, dtype=np.int64)
    pools = {k: acc.arr.get(f"{name}|{k}") for k in ("C2", "C3")}
    if pools["C3"] is None:
        pools = {k: np.zeros((TS.N_DAYS, TS.N_BUCKETS, len(TS.POOL_STATS))) for k in ("C2", "C3")}
    ex = {"C3": _excess(r, day, pools["C3"], None),
          "C3v": _excess(r, day, pools["C3"], bkt),
          "C2": _excess(r, day, pools["C2"], None)}
    month, half = TS.month_key(dates), TS.half_year_key(dates)
    stock = pd.factorize(res["symbol"])[0]
    by_era = {}
    for era, (lo, hi) in ERAS.items():
        s = np.asarray((dates >= pd.Timestamp(lo)) & (dates <= pd.Timestamp(hi)))
        by_era[era] = {"n": int(s.sum()), "expectancy": _finite_mean(r[s]),
                       "win_rate": float((r[s][np.isfinite(r[s])] > 0).mean()) if np.isfinite(r[s]).any() else None,
                       "excess_C3v": _finite_mean(ex["C3v"][s])}
    block = {"n_trades": int(len(t)),
             "n_open_at_end_excluded": int(open_.sum()),
             "n": int(len(res)),
             "n_stocks": int(res["symbol"].nunique()),
             "net": TS.summarize(r),
             "gross": TS.summarize(res["r_gross"].to_numpy(dtype=float)),
             "net_including_open_trades": TS.summarize(r_all),
             "reasons": {str(k): int(v) for k, v in Counter(res["reason"]).items()},
             "ci": _cis(r, month, half, stock, n_boot, seed),
             "by_era": by_era,
             "by_year": _slice_table(res, np.asarray(dates.dt.year), ex["C3v"]) if len(res) else {}}
    aux = {"res": res, "day": day, "bucket": bkt, "month": month, "half": half, "stock": stock,
           "excess": ex, "pools": pools, "r": r}
    return block, aux


def _excess_block(x, aux, n_boot, seed) -> dict:
    ci = _cis(x, aux["month"], aux["half"], aux["stock"], n_boot, seed)
    ci["n_matched"] = int(np.isfinite(x).sum())
    return ci


def _top_bottom(res: pd.DataFrame) -> tuple:
    cols = ["symbol", "signal_date", "entry", "exit", "r_net", "reason", "exit_date", "bars"]
    if res.empty:
        return [], []
    s = res.sort_values("r_net", ascending=False)
    return s.head(15)[cols].to_dict("records"), s.tail(5)[cols].to_dict("records")


def _weighted_f(years, f_by_year, from_year=None):
    """Mean of f over the C0 trades' signal years, weighted by trades per year; years with no f are skipped."""
    num = den = 0.0
    for y, w in Counter(years).items():
        if from_year is not None and y < from_year:
            continue
        f = f_by_year.get(y)
        if f is None or not np.isfinite(f):
            continue
        num += w * f
        den += w
    return float(num / den) if den > 0 else None


def _survivorship_block(res0: pd.DataFrame, E, table: pd.DataFrame) -> dict:
    years = [int(y) for y in pd.to_datetime(res0["signal_date"]).dt.year] if len(res0) else []
    f_stock = {int(y): v for y, v in table["f_stock"].items()}
    f_bottom = {int(y): v for y, v in table["f_bottom_scenario"].items()} if "f_bottom_scenario" in table else {}
    f = {"f_T_stock": _weighted_f(years, f_stock),
         "f_T_bottom": _weighted_f(years, f_bottom),
         "f_ref": _weighted_f(years, f_stock, F_REF_FROM_YEAR)}
    return {**f,
            "breakeven_f": {str(Lv): TS.breakeven_f(E, Lv) for Lv in L_SCENARIOS},
            "adjusted_expectancy": {fn: {str(Lv): TS.adjusted_expectancy(E, fv, Lv) for Lv in L_SCENARIOS}
                                    for fn, fv in f.items()}}


def _get(d, *path):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return None
        d = d[p]
    return d


def verdict(primary: dict) -> dict:
    """Apply VERDICT_RULE to the primary configuration's analysis dict."""
    E = _get(primary, "C0", "net", "expectancy")
    lo_m = _get(primary, "excess", "C0_vs_C3v", "month", "lo")
    lo_h = _get(primary, "excess", "C0_vs_C3v", "half", "lo")
    ex2 = _get(primary, "excess", "C0_vs_C2", "month", "mean")
    f_ref = _get(primary, "survivorship", "f_ref")
    inputs = {"C0 net E[R]": E, "C0-C3v excess month CI lo": lo_m, "C0-C3v excess half-year CI lo": lo_h,
              "C0-C2 excess mean": ex2, "f_ref": f_ref}
    missing = [k for k, v in inputs.items() if v is None]
    if missing:
        return {"verdict": f"incomplete (missing: {', '.join(missing)})", "inputs": inputs, "checks": None}
    adj = TS.adjusted_expectancy(E, f_ref, VERDICT_L)
    checks = {"1_net_expectancy_positive": bool(E > 0),
              "2_excess_vs_C3v_ci_lo_positive_month_and_half": bool(lo_m > 0 and lo_h > 0),
              "3_excess_vs_C2_mean_positive": bool(ex2 > 0),
              f"4_adjusted_expectancy_f_ref_L{VERDICT_L}_positive": bool(adj > 0)}
    inputs["adjusted_expectancy"] = adj
    return {"verdict": "edge" if all(checks.values()) else "no demonstrated edge", "inputs": inputs, "checks": checks}


def analyze(merged: dict, configs, delisted=None, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Everything the report needs, one block per configuration, plus survivorship and the verdict."""
    trades, acc = merged["trades"], merged["acc"]
    table = None
    if delisted is not None:
        spans = merged["spans"].assign(first_date=pd.to_datetime(merged["spans"]["first_date"]),
                                       last_date=pd.to_datetime(merged["spans"]["last_date"]))
        years = list(range(int(SIGNAL_START[:4]), int(SIGNAL_END[:4]) + 1))
        table = TS.delisted_share_by_year(delisted, spans, years, dict(merged["bottom_counts"]))
    out = {"configs": {}}
    for cfg in configs:
        t = trades[trades["config"] == cfg.name] if len(trades) else trades
        t0, t1 = t[t["arm"] == "C0"], t[t["arm"] == "C1"]
        c0, a0 = _arm_block(t0, acc, cfg.name, "C0", n_boot, seed)
        c1, a1 = _arm_block(t1, acc, cfg.name, "C1", n_boot, seed)
        c0["top15"], c0["bottom5"] = _top_bottom(a0["res"])
        ex0, ex1 = a0["excess"], a1["excess"]
        res = {"spec": config_spec(cfg), "C0": c0, "C1": c1,
               "pool_trades_open_at_end_excluded": int(merged["pool_end"].get(cfg.name, 0)),
               "pool_n": {k: (int(acc.arr[f"{cfg.name}|{k}"][:, :, 0].sum()) if f"{cfg.name}|{k}" in acc.arr else 0)
                          for k in ("C2", "C3")},
               "controls": {"C3": TS.matched_control_summary(a0["pools"]["C3"], a0["day"], None),
                            "C3v": TS.matched_control_summary(a0["pools"]["C3"], a0["day"], a0["bucket"]),
                            "C2": TS.matched_control_summary(a0["pools"]["C2"], a0["day"], None)},
               "excess": {"C0_vs_C3": _excess_block(ex0["C3"], a0, n_boot, seed),
                          "C0_vs_C3v": _excess_block(ex0["C3v"], a0, n_boot, seed),
                          "C0_vs_C2": _excess_block(ex0["C2"], a0, n_boot, seed),
                          "C1_vs_C3": _excess_block(ex1["C3"], a1, n_boot, seed),
                          "C1_vs_C3v": _excess_block(ex1["C3v"], a1, n_boot, seed)},
               "C0_minus_C1_vs_C3v": {
                   "month": TS.cluster_bootstrap_diff(ex0["C3v"], a0["month"], ex1["C3v"], a1["month"], n_boot, seed, ALPHA),
                   "half": TS.cluster_bootstrap_diff(ex0["C3v"], a0["half"], ex1["C3v"], a1["half"], n_boot, seed, ALPHA)}}
        if table is not None:
            res["survivorship"] = _survivorship_block(a0["res"], c0["net"]["expectancy"], table)
        out["configs"][cfg.name] = res
    if table is not None:
        out["survivorship"] = {
            "by_year": {int(y): row for y, row in table.to_dict("index").items()},
            "note": ("f_stock counts listed common stocks missing from the price cache by number of stocks. For a screen "
                     "that selects beaten-down stocks it is a LOWER bound: those are the stocks that disappear. "
                     "f_bottom_scenario assumes every stock that was delisted within two years was in the bottom state."),
            "L_scenarios": list(L_SCENARIOS)}
    prim = next((c for c in configs if c.role == "primary"), None)
    if prim is not None:
        out["verdict"] = verdict(out["configs"][prim.name])
        out["stability"] = _stability(configs, out["configs"], prim.name)
    return out


def _stability(configs, results: dict, primary_name: str) -> dict:
    """How many stability configs share the sign of the primary C0-minus-C3v excess (never changes the verdict)."""
    def excess(name):
        return _get(results[name], "excess", "C0_vs_C3v", "month", "mean")

    sign = excess(primary_name)
    stab = {c.name: excess(c.name) for c in configs if c.role == "stability"}
    same = [k for k, v in stab.items() if sign is not None and v is not None and np.sign(v) == np.sign(sign)]
    return {"primary_excess_C0_vs_C3v": sign, "n_configs": len(stab), "n_same_sign": len(same), "excess_by_config": stab}


# ── one stock, by hand ────────────────────────────────────────────────────────────────────────────

def inspect_symbol(history, symbol: str, configs) -> dict:
    """The raw C0 signals and trades of ONE stock under 'primary' and 'bottom_same_bar', so data errors
    and the effect of the cooldown can be eyeballed. Reads every batch file (ignores --limit-batches)."""
    import pyarrow.parquet as pq
    files = sorted(Path(history).expanduser().glob("batch_*.parquet"))
    parts = []
    for f in files:
        tbl = pq.read_table(f, filters=[("symbol", "=", symbol)])
        if tbl.num_rows:
            parts.append(tbl.to_pandas())
    out = {"symbol": symbol, "found": bool(parts), "configs": {}}
    if not parts:
        return out
    raw = pd.concat(parts, ignore_index=True)
    raw["date"] = pd.to_datetime(raw["date"])
    if raw["date"].dt.tz is not None:
        raw["date"] = raw["date"].dt.tz_convert(None)
    df = clean_daily(raw.set_index("date").drop(columns="symbol").astype(float))
    if len(df) < MIN_ROWS:
        out["found"] = False
        return out
    ind = indicators(df)
    for cfg in configs:
        if cfg.name not in ("primary", "bottom_same_bar"):
            continue
        idx = np.flatnonzero(arm_masks(ind, cfg, None)["C0"])
        tr = simulate(ind, idx, cfg.exit, cfg.cooldown, symbol) if len(idx) else pd.DataFrame(columns=list(TRADE_COLUMNS))
        taken = set(tr["signal_i"].tolist())
        out["configs"][cfg.name] = {
            "raw_signals": [{"date": ind.index[i], "status": "taken" if int(i) in taken else "suppressed"} for i in idx],
            "trades": [{**{k: row[k] for k in ("signal_date", "entry_date", "exit_date", "entry", "exit", "r_net", "reason", "bars")},
                        "resolved": row["reason"] != "end"} for _, row in tr.iterrows()]}
    return out


# ── output ────────────────────────────────────────────────────────────────────────────────────────

def _jsonable(o):
    """Recursively convert to strict-JSON types: non-finite floats -> None, numpy -> Python, dates -> ISO."""
    if o is None or isinstance(o, (str, bool)):
        return o
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, (pd.Timestamp, np.datetime64)):
        t = pd.Timestamp(o)
        return None if pd.isna(t) else t.date().isoformat()
    if isinstance(o, dict):
        return {str(_jsonable(k) if not isinstance(k, str) else k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return [_jsonable(v) for v in o.tolist()]
    if isinstance(o, pd.DataFrame):
        return _jsonable(o.to_dict("records"))
    if isinstance(o, Path):
        return str(o)
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return _jsonable(dataclasses.asdict(o))
    if o is pd.NaT:
        return None
    return str(o)


def _n(x, nd=3):
    return "n/a" if x is None else f"{x:+.{nd}f}"


def _ci(d, nd=3):
    if not d or d.get("lo") is None:
        return "[n/a]"
    return f"[{d['lo']:+.{nd}f}, {d['hi']:+.{nd}f}]"


def format_table(analysis: dict) -> str:
    head = (f"{'name':<18} {'role':<9} {'C0 n':>6} {'stocks':>6} {'E net':>7} {'CI month':>18} "
            f"{'excess C3v [CI month]':>28} {'excess C2':>10} {'C1 n':>8} {'C1 E':>7}")
    lines = [head, "-" * len(head)]
    for name, r in analysis["configs"].items():
        c0, c1, ex = r["C0"], r["C1"], r["excess"]
        lines.append(
            f"{name:<18} {r['spec']['role']:<9} {c0['n']:>6} {c0['n_stocks']:>6} {_n(c0['net']['expectancy']):>7} "
            f"{_ci(c0['ci']['month']):>18} "
            f"{(_n(ex['C0_vs_C3v']['month']['mean']) + ' ' + _ci(ex['C0_vs_C3v']['month'])):>28} "
            f"{_n(ex['C0_vs_C2']['month']['mean']):>10} {c1['n']:>8} {_n(c1['net']['expectancy']):>7}")
    return "\n".join(lines)


def _warm_jit() -> None:
    """Compile the numba kernels BEFORE forking workers, so each worker does not compile its own copy."""
    n = 40
    rng = np.random.default_rng(0)
    c = 10 + np.cumsum(rng.normal(0, 0.1, n))
    ind = pd.DataFrame({"open": c, "high": c + 0.2, "low": c - 0.2, "close": c, "atr14": np.full(n, 0.4)},
                       index=pd.bdate_range("2020-01-01", periods=n))
    for m in (EXIT_PRIMARY, EXIT_SECONDARY):
        simulate(ind, [3, 10], m, 5)
        mask = np.zeros(n, dtype=bool)
        mask[[3, 10]] = True
        simulate_each(ind, mask, m)


def _data_end(files) -> pd.Timestamp:
    return max(pd.Timestamp(pd.read_parquet(f, columns=["date"])["date"].max()) for f in files)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--history", default=str(DEFAULT_HISTORY))
    ap.add_argument("--limit-batches", type=int, default=None, help="smoke test: only the first N batch files (no ledger)")
    ap.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    ap.add_argument("--out", default=None, help=f"results JSON (default {OUT_JSON}, or {SMOKE_OUT} with --limit-batches)")
    ap.add_argument("--no-ledger", action="store_true", help="do not write to the trial ledger")
    ap.add_argument("--delisted", default=str(DELISTED_CSV))
    ap.add_argument("--inspect", action="append", default=None, metavar="SYM",
                    help=f"also report the signals and trades of one stock (default {', '.join(INSPECT)})")
    ap.add_argument("--trades-out", default=None, help=f"trades parquet (default {TRADES_OUT})")
    args = ap.parse_args(argv)

    limited = args.limit_batches is not None
    use_ledger = (not limited) and (not args.no_ledger)
    out_path = Path(args.out).expanduser() if args.out else (SMOKE_OUT if limited else OUT_JSON)
    trades_out = Path(args.trades_out).expanduser() if args.trades_out else (
        TRADES_OUT.with_name("smoke_trades.parquet") if limited else TRADES_OUT)
    history = Path(args.history).expanduser()
    files = sorted(history.glob("batch_*.parquet"))
    if limited:
        files = files[: args.limit_batches]
    if not files:
        raise SystemExit(f"no batch files in {history}")
    data_end = _data_end(files)
    window = (SIGNAL_START, SIGNAL_END)
    configs = list(CONFIGS)
    delisted_path = Path(args.delisted).expanduser()
    delisted = pd.read_csv(delisted_path) if delisted_path.exists() else None
    if delisted is None:
        print(f"note: {delisted_path} not found, survivorship terms (and so the verdict) will be incomplete", flush=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.time()
        _warm_jit()
        before = L.trial_count()
        if use_ledger:
            L.log_trial({"stage": LEDGER_STAGE, "event": "preregistered", "configs": [config_spec(c) for c in configs],
                         "window": list(window), "verdict_rule": VERDICT_RULE})
        tasks = [(str(f), configs, window) for f in files]

        def _results():
            if args.workers <= 1:
                it = map(process_batch, tasks)
                pool = None
            else:
                pool = mp.get_context("fork").Pool(args.workers)
                it = pool.imap_unordered(process_batch, tasks)
            try:
                for k, r in enumerate(it, 1):
                    print(f"  {k}/{len(tasks)} batches, {r['n_stocks']} stocks ({time.time() - t0:.0f}s)", flush=True)
                    yield r
            finally:
                if pool is not None:
                    pool.close()
                    pool.join()

        merged = merge(_results())
        print(f"scanned {merged['n_stocks']} stocks in {len(files)} batch file(s), data through {data_end.date()}; analysing...",
              flush=True)
        analysis = analyze(merged, configs, delisted)
        inspect = [inspect_symbol(history, s, configs) for s in (list(INSPECT) + (args.inspect or []))]

        analysis["data_end"] = data_end
        analysis["n_stocks_scanned"] = merged["n_stocks"]
        analysis["limited_run"] = limited
        analysis["batch_files"] = len(files)
        analysis["window"] = list(window)
        analysis["verdict_rule"] = VERDICT_RULE
        analysis["inspect"] = inspect
        analysis["ledger_trials_before"] = before
        if use_ledger:
            for cfg in configs:
                r = analysis["configs"][cfg.name]
                L.log_trial({"stage": LEDGER_STAGE, "config": cfg.name, "role": cfg.role, "spec": config_spec(cfg),
                             "n_trades": r["C0"]["n"], "E": r["C0"]["net"]["expectancy"],
                             "excess_C3v_mean": r["excess"]["C0_vs_C3v"]["month"]["mean"],
                             "excess_C3v_lo": r["excess"]["C0_vs_C3v"]["month"]["lo"],
                             "data_end": str(data_end.date()), "n_stocks_scanned": merged["n_stocks"]})
        analysis["ledger_trials_after"] = L.trial_count()

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(_jsonable(analysis), indent=1, allow_nan=False))
        trades_out.parent.mkdir(parents=True, exist_ok=True)
        merged["trades"].to_parquet(trades_out, index=False)

    print()
    print(format_table(analysis))
    n_open = {k: (v["C0"]["n_open_at_end_excluded"], v["C1"]["n_open_at_end_excluded"]) for k, v in analysis["configs"].items()}
    if any(a or b for a, b in n_open.values()):
        print("\ntrades still open at the data end are EXCLUDED from every statistic above (C0, C1): "
              + ", ".join(f"{k} {a}/{b}" for k, (a, b) in n_open.items() if a or b))
    else:
        print("\ntrades still open at the data end excluded from the statistics above: none")
    v = analysis["verdict"]
    st = analysis["stability"]
    print(f"\nVERDICT (primary): {v['verdict']}")
    if v["checks"]:
        print("  checks: " + "; ".join(f"{k}={'PASS' if ok else 'FAIL'}" for k, ok in v["checks"].items()))
    print(f"  stability: {st['n_same_sign']} of {st['n_configs']} stability configs share the sign of the primary "
          f"C0-vs-C3v excess ({_n(st['primary_excess_C0_vs_C3v'])})")
    if limited:
        print("  (limited run: a smoke test of the machinery, NOT the study)")
    for ins in inspect:
        if not ins["found"]:
            print(f"inspect {ins['symbol']}: not found in the cache")
            continue
        for name, blk in ins["configs"].items():
            sig = ", ".join(f"{pd.Timestamp(s['date']).date()} {s['status']}" for s in blk["raw_signals"]) or "none"
            print(f"inspect {ins['symbol']} [{name}] raw C0 signals: {sig}")
            for tr in blk["trades"]:
                print(f"    trade {pd.Timestamp(tr['entry_date']).date()} -> {pd.Timestamp(tr['exit_date']).date()} "
                      f"{tr['entry']:.2f} -> {tr['exit']:.2f} net {tr['r_net']:+.2f}R {tr['reason']}"
                      f"{'' if tr['resolved'] else ' (OPEN, excluded)'}")
    print(f"wrote {out_path} and {trades_out} | ledger trials: {before} -> {analysis['ledger_trials_after']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
