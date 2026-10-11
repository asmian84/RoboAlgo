"""
Context for the Finder validation: what the verdicts are measured against, and how much
the modelling choices matter.

    python scripts/finder_sensitivity.py            # writes docs/finder_sensitivity.json

  1. OPTIMISTIC BOUND. The replay is pessimistic on purpose (a bar that could hit both stop
     and target is a stop). Rerun with the opposite convention. If the screen fails even
     then, the verdict does not depend on that choice. These runs are a bound, not
     candidates, and are NOT part of the trial ledger.
  2. BUY-AND-HOLD SPY over the same windows. A Sharpe means nothing without the market's.
  3. EVERY GATE RULE, not just the first to fail, plus the deflated Sharpe itself.
  4. COST BREAK-EVEN: gross expectancy per trade against the assumed round-trip cost.
  5. COVERAGE: how much of the universe was actually scanned.
"""
from __future__ import annotations

import json
import sys
import warnings
from dataclasses import replace
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import validate_finder as vf
from strategy_validation.finder_backtest import build_portfolio, period_stats
from strategy_validation.gates import (PerformanceStats, StrategyCandidate, ValidationLimits, evaluate)

BASES = ["signal.base", "pennyking.base", "maker.base", "penny_floor1.base"]


def spy_stats(start: str) -> dict:
    from sqlalchemy import text
    from database.connection import get_engine
    px = pd.read_sql(text("select p.date, p.close from price_data p join instruments i on i.id=p.instrument_id "
                          "where i.symbol='SPY' order by p.date"), get_engine(), parse_dates=["date"], index_col="date")["close"]
    r = px.pct_change().dropna()
    cal = r.index.values.astype("datetime64[ns]")
    out = {}
    for tag, a, b in (("in_sample", start, vf.IS_END), ("out_of_sample", vf.OOS_START, None)):
        out[tag] = period_stats(r.values, cal, [], 0.0, a, b)
    return out


def main() -> int:
    cal_full = vf.spy_calendar()
    res = json.loads((ROOT / "docs" / "finder_validation_results.json").read_text())
    out = {"coverage": {}, "benchmark_spy": {}, "scorecard": {}, "cost_breakeven": {}, "optimistic_bound": {}}

    # 5. coverage ---------------------------------------------------------------
    manifest = json.loads((vf.HISTORY / "manifest.json").read_text())
    bars = sum(pq.ParquetFile(f).metadata.num_rows for f in sorted(vf.HISTORY.glob("batch_*.parquet")))
    out["coverage"] = {
        "listed_us_stocks_on_download_date": manifest["listed_today"],
        "preferred_shares_not_fetchable": manifest["skipped_preferreds"],
        "symbols_with_price_history": manifest["with_data"],
        "symbols_no_data_returned": manifest["no_data"],
        "tickers_scanned_with_the_full_rule": res["groups"]["broad"]["coverage"]["scanned"],
        "tickers_too_new_for_the_engine (<282 bars)": res["groups"]["broad"]["coverage"]["short_history"],
        "ticker_days_evaluated_broad": int(bars),
        "tracked_db_instruments_scanned": res["groups"]["db"]["coverage"]["scanned"],
        "tickers_with_at_least_one_signal": {**res["groups"]["db"]["coverage"]["with_signal"],
                                             **res["groups"]["broad"]["coverage"]["with_signal"]},
    }

    # 2. benchmark --------------------------------------------------------------
    out["benchmark_spy"] = {"db_window": spy_stats(vf.IS_START["db"]), "broad_window": spy_stats(vf.IS_START["broad"])}

    # 3. full scorecard + DSR ---------------------------------------------------
    configs = {}
    for g in res["groups"].values():
        configs.update(g["configs"])
    lim = ValidationLimits()
    relaxed = ValidationLimits(min_oos_sharpe=-99, max_drawdown_pct=1e9, min_hit_rate=0.0, min_trades=0, min_dsr=0.0)
    for kind in ("uncapped", "top10"):
        ledger = [(c[kind]["out_of_sample"]["sharpe"] or 0.0) for c in configs.values()]
        for name in BASES:
            o, i = configs[name][kind]["out_of_sample"], configs[name][kind]["in_sample"]
            ps = lambda d: PerformanceStats(d["sharpe"] or 0.0, d["max_drawdown_pct"] or 0.0, d["hit_rate"] or 0.0,
                                            d["n_trades"], d["n_obs"], d["skew"], d["kurtosis"])
            cand = StrategyCandidate(ps(i), ps(o), n_trials=len(configs), name=name, trial_sharpes=ledger)
            v = evaluate(cand, relaxed)           # relaxed limits so the DSR is always computed
            m = v.metrics
            out["scorecard"][f"{name}|{kind}"] = {
                "oos_sharpe": o["sharpe"], "needs_>=": lim.min_oos_sharpe, "sharpe_ok": (o["sharpe"] or 0) >= lim.min_oos_sharpe,
                "oos_max_dd_pct": o["max_drawdown_pct"], "dd_limit": lim.max_drawdown_pct, "dd_ok": (o["max_drawdown_pct"] or 0) <= lim.max_drawdown_pct,
                "oos_hit_rate": o["hit_rate"], "hit_ok": (o["hit_rate"] or 0) >= lim.min_hit_rate,
                "oos_trades": o["n_trades"], "trades_ok": o["n_trades"] >= lim.min_trades,
                "dsr": m.get("dsr"), "benchmark_sharpe_to_beat": m.get("benchmark_sharpe"), "dsr_ok": (m.get("dsr") or 0) >= lim.min_dsr,
                "in_sample_sharpe": i["sharpe"], "oos_over_is": m.get("oos_is_ratio"),
            }
            cb = configs[name][kind]["out_of_sample"]
            out["cost_breakeven"][f"{name}|{kind}"] = {
                "oos_gross_expectancy_pct_per_trade": cb["gross_expectancy_pct"],
                "assumed_round_trip_cost_pct": vf.COST[name.split('.')[0]] * 100,
                "oos_net_expectancy_pct_per_trade": cb["net_expectancy_pct"], "oos_profit_factor": cb["profit_factor"]}

    # 1. optimistic bound -------------------------------------------------------
    vf.OPTIMISTIC = True            # forked workers inherit this
    for group, cal_start in (("db", vf.IS_START["db"]), ("broad", "2018-01-01")):
        rules, variants = vf.rules_and_variants(group)
        variants = {k: v for k, v in variants.items() if k in BASES}
        cal = cal_full[cal_full >= np.datetime64(cal_start)]
        trades = {k: [] for k in variants}
        if group == "db":
            for sym, df in vf.load_db().items():
                if len(df) < 282:
                    continue
                for k, tr in vf.run_symbol(sym, df, rules, variants).items():
                    trades[k].extend(tr)
        else:
            import multiprocessing as mp
            files = sorted(vf.HISTORY.glob("batch_*.parquet"))
            with mp.get_context("fork").Pool(14) as pool:
                for o, _ in pool.imap_unordered(vf._work_batch, [(f, "broad") for f in files]):
                    for k in variants:
                        trades[k].extend(o.get(k, []))
        for k, tr in trades.items():
            preset = k.split(".")[0]
            row = {}
            for kind, K in (("uncapped", None), ("top10", vf.TOP_K)):
                daily, taken = build_portfolio(tr, cal, vf.COST[preset], K)
                row[kind] = {"in_sample": period_stats(daily, cal, taken, vf.COST[preset], vf.IS_START[group], vf.IS_END),
                             "out_of_sample": period_stats(daily, cal, taken, vf.COST[preset], vf.OOS_START)}
            out["optimistic_bound"][k] = row
            print(f"  optimistic {k:<20} OOS Sharpe uncapped {row['uncapped']['out_of_sample']['sharpe'] or 0:5.2f}"
                  f" | top10 {row['top10']['out_of_sample']['sharpe'] or 0:5.2f}", flush=True)

    Path(ROOT / "docs" / "finder_sensitivity.json").write_text(json.dumps(out, indent=1, default=str))
    print("wrote docs/finder_sensitivity.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
