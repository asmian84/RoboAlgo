"""
Replay the Finder over its ENTIRE universe and put the result through the
validation gate.

    python scripts/validate_finder.py                 # everything
    python scripts/validate_finder.py --only db       # the tracked large-cap/ETF group
    python scripts/validate_finder.py --only broad    # the full listed universe (needs history cache)

Needs the history cache from scripts/download_universe_history.py for --only broad.

WHAT IS TESTED
  signal    : the "signal" preset over the ~234 instruments the database tracks (2010 onward)
  pennyking : the sub-$5 preset over the whole listed US universe (2018 onward)
  maker     : the $5-20 preset over the whole listed US universe
  A realism variant, penny_floor1, drops sub-$1 names, which the live preset allows.

THE TRIAL COUNT. Every configuration evaluated here is declared and counted, because
looking at several and reporting the best is exactly the selection effect the gate
corrects for. Per preset: the base rule, two hold-period variants, and an
"only while still below entry" variant. Plus the $1-floor variant. That is 13
configurations, and every candidate is deflated against all 13.

THE SPLIT. Rules are fixed (nothing is fitted), so "in sample" is simply the earlier
period. In sample ends 2023-12-31; out of sample is 2024-01-01 onward. The preset
thresholds were set by judgement from recent data, not by optimising performance,
but they are not independent of that data, so the OOS window is not perfectly clean.

TWO PORTFOLIOS, both reported, because they answer different questions:
  uncapped : every signal taken, equal-weighted. Measures the screen's edge over the
             whole universe with no selection. Not tradable as such.
  top10    : at most 10 positions, filled in the Finder's own ranking. Tradable.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from strategy_validation.finder_backtest import (
    ExecParams, Rule, build_portfolio, period_stats, run_symbol)

HISTORY = Path.home() / ".roboalgo" / "history"
IS_END = "2023-12-31"
OOS_START = "2024-01-01"
IS_START = {"db": "2012-01-01", "broad": "2019-03-01"}
# Round-trip cost as a fraction. ASSUMPTIONS, not measurements: spread + slippage grow as price falls.
COST = {"signal": 0.0010, "maker": 0.0040, "pennyking": 0.0100, "penny_floor1": 0.0100}
TOP_K = 10

EXEC = {
    "base":    ExecParams(valid_days=10, max_hold=30),
    "h20":     ExecParams(valid_days=10, max_hold=20),
    "h60":     ExecParams(valid_days=10, max_hold=60),
    "untrig":  ExecParams(valid_days=10, max_hold=30, untriggered_only=True),
}


# Sensitivity switch, set by scripts/finder_sensitivity.py before it forks workers: replay with the
# OPPOSITE (optimistic) fill conventions to bound how much the pessimistic defaults matter.
OPTIMISTIC = False


def rules_and_variants(group: str):
    from dataclasses import replace
    from scanner_engine.finder import PRESETS
    if group == "db":
        rules = {"signal": Rule.from_criteria(PRESETS["signal"])}
        variants = {f"signal.{k}": ("signal", v) for k, v in EXEC.items()}
    else:
        pk, mk = Rule.from_criteria(PRESETS["pennyking"]), Rule.from_criteria(PRESETS["maker"])
        floor = Rule(**{**pk.__dict__, "min_price": 1.0})
        rules = {"pennyking": pk, "maker": mk, "penny_floor1": floor}
        variants = {f"{r}.{k}": (r, v) for r in ("pennyking", "maker") for k, v in EXEC.items()}
        variants["penny_floor1.base"] = ("penny_floor1", EXEC["base"])
    if OPTIMISTIC:
        variants = {k: (r, replace(ex, stop_first=False)) for k, (r, ex) in variants.items()}
    return rules, variants


# ── data loading ──────────────────────────────────────────────────────────────

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[(df["close"] > 0) & (df["open"] > 0) & (df["high"] >= df["low"]) & (df["low"] > 0)]
    return df.dropna(subset=["open", "high", "low", "close"]).fillna({"volume": 0.0})


def load_db() -> dict[str, pd.DataFrame]:
    from sqlalchemy import text
    from database.connection import get_engine
    q = text("""select i.symbol, p.date, p.open, p.high, p.low, p.close, p.volume
                from price_data p join instruments i on i.id = p.instrument_id order by i.symbol, p.date""")
    raw = pd.read_sql(q, get_engine(), parse_dates=["date"])
    return {s: clean(g.set_index("date").drop(columns="symbol").astype(float)) for s, g in raw.groupby("symbol")}


def spy_calendar() -> np.ndarray:
    from sqlalchemy import text
    from database.connection import get_engine
    d = pd.read_sql(text("select p.date from price_data p join instruments i on i.id=p.instrument_id "
                         "where i.symbol='SPY' order by p.date"), get_engine(), parse_dates=["date"])["date"]
    return d.values.astype("datetime64[ns]")


# ── per-batch worker ──────────────────────────────────────────────────────────

def _work_batch(args):
    path, group = args
    rules, variants = rules_and_variants(group)
    raw = pd.read_parquet(path)
    raw["date"] = pd.to_datetime(raw["date"]).dt.tz_localize(None)
    out = {v: [] for v in variants}
    cov = {"scanned": 0, "short_history": 0, "with_signal": {r: 0 for r in rules}}
    for sym, g in raw.groupby("symbol"):
        df = clean(g.set_index("date").drop(columns="symbol").astype(float))
        res = run_symbol(sym, df, rules, variants)
        if len(df) < 282:
            cov["short_history"] += 1
            continue
        cov["scanned"] += 1
        for v, trades in res.items():
            out[v].extend(trades)
        for r in rules:
            if res.get(f"{r}.base"):
                cov["with_signal"][r] += 1
    return out, cov


def run_broad() -> tuple[dict, dict]:
    files = sorted(HISTORY.glob("batch_*.parquet"))
    if not files:
        raise SystemExit(f"no history in {HISTORY}; run scripts/download_universe_history.py first")
    _, variants = rules_and_variants("broad")
    trades = {v: [] for v in variants}
    cov = {"scanned": 0, "short_history": 0, "with_signal": {}}
    t0 = time.time()
    with mp.get_context("fork").Pool(min(14, mp.cpu_count())) as pool:
        for k, (out, c) in enumerate(pool.imap_unordered(_work_batch, [(f, "broad") for f in files]), 1):
            for v, tr in out.items():
                trades[v].extend(tr)
            cov["scanned"] += c["scanned"]
            cov["short_history"] += c["short_history"]
            for r, n in c["with_signal"].items():
                cov["with_signal"][r] = cov["with_signal"].get(r, 0) + n
            print(f"  broad: {k}/{len(files)} batches, {cov['scanned']} tickers scanned "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return trades, cov


def run_db() -> tuple[dict, dict]:
    data = load_db()
    rules, variants = rules_and_variants("db")
    trades = {v: [] for v in variants}
    cov = {"scanned": 0, "short_history": 0, "with_signal": {"signal": 0}}
    for sym, df in data.items():
        if len(df) < 282:
            cov["short_history"] += 1
            continue
        cov["scanned"] += 1
        res = run_symbol(sym, df, rules, variants)
        for v, tr in res.items():
            trades[v].extend(tr)
        if res.get("signal.base"):
            cov["with_signal"]["signal"] += 1
    print(f"  db: {cov['scanned']} instruments scanned, {cov['short_history']} too short", flush=True)
    return trades, cov


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate_group(group: str, trades: dict, cov: dict, calendar: np.ndarray) -> dict:
    results = {}
    start = IS_START[group]
    for name, tr in trades.items():
        preset = name.split(".")[0]
        row = {"preset": preset, "n_trades_all": len(tr), "tickers_traded": len({t.symbol for t in tr}),
               "exit_reasons": dict(pd.Series([t.reason for t in tr]).value_counts()) if tr else {}}
        for kind, k in (("uncapped", None), ("top10", TOP_K)):
            daily, taken = build_portfolio(tr, calendar, COST[preset], k)
            row[kind] = {
                "in_sample": period_stats(daily, calendar, taken, COST[preset], start, IS_END),
                "out_of_sample": period_stats(daily, calendar, taken, COST[preset], OOS_START),
                "by_year": {str(y): period_stats(daily, calendar, taken, COST[preset], f"{y}-01-01", f"{y}-12-31")
                            for y in range(int(start[:4]), int(calendar[-1].astype("datetime64[Y]").astype(int)) + 1971)},
            }
        results[name] = row
        print(f"  {name:<22} trades {len(tr):>7}  OOS Sharpe uncapped "
              f"{(row['uncapped']['out_of_sample']['sharpe'] or 0):>6.2f} | top10 "
              f"{(row['top10']['out_of_sample']['sharpe'] or 0):>6.2f}", flush=True)
    return {"coverage": cov, "configs": results}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--only", choices=["db", "broad", "all"], default="all")
    ap.add_argument("--out", default=str(ROOT / "docs" / "finder_validation_results.json"))
    args = ap.parse_args()

    cal_full = spy_calendar()
    out = {"split": {"in_sample_end": IS_END, "oos_start": OOS_START, "is_start": IS_START},
           "costs_round_trip": COST, "top_k": TOP_K, "groups": {}}
    if args.only in ("db", "all"):
        print("== DB group (tracked large caps + ETFs)", flush=True)
        trades, cov = run_db()
        out["groups"]["db"] = evaluate_group("db", trades, cov, cal_full[cal_full >= np.datetime64(IS_START["db"])])
    if args.only in ("broad", "all"):
        print("== Broad group (whole listed universe)", flush=True)
        trades, cov = run_broad()
        out["groups"]["broad"] = evaluate_group("broad", trades, cov, cal_full[cal_full >= np.datetime64("2018-01-01")])
    Path(args.out).write_text(json.dumps(out, indent=1, default=lambda o: None if o is None else str(o)))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
