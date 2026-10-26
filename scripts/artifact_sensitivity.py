"""
Do the likely corporate-action artifacts in the price cache drive the turnaround and event-study results? A post-hoc sensitivity run, not a new test.

    python scripts/artifact_sensitivity.py --workers 8                   # full run, writes docs/artifact_sensitivity_results.json
    python scripts/artifact_sensitivity.py --summary-only --limit 300    # by-year table of flagged days, writes nothing
    python scripts/artifact_sensitivity.py --out /tmp/a.json             # write somewhere else

WHAT IT DOES. setup_engine.artifacts flags days whose close moved 50% or more on quiet volume (the
signature of a split or reverse split the data vendor did not adjust). Two published results are then
recomputed with every observation that overlaps such a day REMOVED, and shown next to the original (BEFORE)
numbers, which must reproduce the published ones to the last digit:
    TURNAROUND  the primary configuration, arm C0 (docs/turnaround_study_results.json). A trade is dropped
                if a flagged day of its stock lies in [signal_date - PRE_DAYS days, exit_date], both ends
                inclusive. Trades still open at the data end (reason "end") were excluded from every
                published statistic and are excluded here too. Month and stock clusters use the SAME date
                column as the published study (signal_date), so the BEFORE intervals match it.
    EVENTS      the held-out mature SPY events, outcome ret_R3, signatures S1-S4 and B0
                (docs/mtf_event_study_results.json). A stock-event row is dropped if a flagged day lies in
                [event week, event week + 156 weeks]. The signature masks, the cross-sectional ranks and the
                date-matched control (the mean outcome of ALL stocks that event week) are computed ONCE on
                the FULL table and never recomputed after a drop, so AFTER differs from BEFORE only through
                which treated rows remain.

PRE-REGISTRATION. NONE: this is POST-HOC. The artifact rule was chosen after looking at suspicious bars,
so it is a sensitivity tool and must not be read as a second test. The published verdicts stand or fall
on their own pre-registered rules; this run can only say how much of a published number depended on rows
that look like artifacts. The constants below are fixed in this file before the run; the bootstrap seeds
are the seeds of the published studies, so BEFORE is bit-for-bit comparable.

KNOWN BIASES (not fixable here, reported with every result).
  * The rule misclassifies. Real crashes or spikes on news during a trading halt (volume under twice a
    prior average already inflated by the run-up) and gap-ups on low volume (buy-outs in thin names) are
    flagged as "artifacts"; dropping them removes genuine big losers or winners. The dropped trades' mean
    is reported next to AFTER: if the dropped trades are mostly losers, AFTER is flattered, not cleaned.
  * Unadjusted old splits are invisible: Yahoo adjusts them retroactively, so the price band is shifted
    rather than broken. "No artifact found" does not mean "clean".
  * The control stays at the FULL same-week mean while the treated sample is cleaned. If control rows
    contain artifacts too, AFTER excess is biased by the control's own contamination. The share of
    contaminated control rows is reported so the size of that asymmetry is visible.
  * Survivorship, retroactive split adjustment and the narrow intervals of the published studies carry
    over unchanged; with about ten independent market moments every event interval is crude.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from setup_engine import artifacts as A
from setup_engine import mtf
from setup_engine import trade_stats as TS
from setup_engine.screen import Screen

# ── fixed choices (POST-HOC sensitivity; see the module docstring) ────────────────────────────────

TRADES = Path.home() / ".roboalgo" / "turnaround" / "trades.parquet"
EVENT_ROWS = Path.home() / ".roboalgo" / "mtf" / "event_rows.parquet"
HISTORY = mtf.DEFAULT_HISTORY
OUT_JSON = ROOT / "docs" / "artifact_sensitivity_results.json"

CONFIG = "primary"                  # turnaround configuration that decided the published verdict
ARM = "C0"                          # the treatment arm
PRE_DAYS = 60                       # calendar days before the signal date that still count against a trade
EVENT_WEEKS_FWD = 156               # an event row is exposed for this many weeks after its event week
SIGS = ("S1", "S2", "S3", "S4", "B0")
OUTCOME = "ret_R3"                  # the primary exit of the event study
HELD_OUT_WEEKS = ("2001-05-18", "2001-12-07", "2002-11-29", "2003-03-21", "2008-05-02",
                  "2009-04-24", "2010-08-06", "2011-10-28", "2019-02-15", "2020-05-29")

N_BOOT = 2000
SEED_TURNAROUND = 20261007          # the seed of scripts/study_turnaround.py, so BEFORE reproduces its intervals
SEED_EVENTS = 20261009              # the seed of scripts/mtf_event_study.py, so BEFORE reproduces its intervals
ALPHA = 0.05
ALPHA_EVENT = 0.05 / 4              # Bonferroni over the four verdict signatures (98.75% interval)
N_DROPPED_LISTED = 50               # how many dropped trades are listed in the JSON

TRADE_COLUMNS = ["symbol", "signal_date", "exit_date", "r_net", "reason", "config", "arm"]
RULE_TEXT = ("A day is an artifact when |close/previous close - 1| >= MIN_ABS_RET, the previous day's dollar "
             "volume >= MIN_PRIOR_DVOL and the day's volume < VOL_SPIKE x the mean volume of the VOL_WIN days "
             "before it (at least VOL_MIN_DAYS of them). POST-HOC: chosen after seeing suspicious bars.")


# ── helpers ───────────────────────────────────────────────────────────────────────────────────────

def _finite_mean(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else None


def _week_mean(wk, v):
    """Per row: the mean of the finite ``v`` over the rows of the same week (all rows); NaN if none."""
    ok = np.isfinite(v)
    s = pd.Series(np.where(ok, v, 0.0)).groupby(wk).transform("sum").to_numpy()
    c = pd.Series(ok.astype("float64")).groupby(wk).transform("sum").to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c > 0, s / c, np.nan)


def _week_int(weeks) -> np.ndarray:
    return pd.to_datetime(weeks).to_numpy(dtype="datetime64[ns]").astype("int64")


# ── turnaround ────────────────────────────────────────────────────────────────────────────────────

def select_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """The published sample: configuration CONFIG, arm ARM, resolved trades only (reason != "end")."""
    t = trades[(trades["config"] == CONFIG) & (trades["arm"] == ARM) & (trades["reason"] != "end")]
    return t.reset_index(drop=True)


def trade_bad_mask(t: pd.DataFrame, days_by_symbol: dict) -> np.ndarray:
    """True for trades with a flagged day in [signal_date - PRE_DAYS days, exit_date] (both inclusive)."""
    starts = pd.to_datetime(t["signal_date"]) - pd.Timedelta(days=PRE_DAYS)
    ends = pd.to_datetime(t["exit_date"])
    return A.contaminated_mask(days_by_symbol, t["symbol"].to_numpy(), starts, ends)


def _trade_block(t: pd.DataFrame, n_boot: int, seed: int) -> dict:
    r = t["r_net"].to_numpy(dtype=float)
    month = TS.month_key(pd.to_datetime(t["signal_date"]))       # the SAME date column as the published study
    stock = pd.factorize(t["symbol"])[0]
    return {"n": int(len(t)), "mean_r": _finite_mean(r),
            "month": TS.cluster_bootstrap(r, month, n_boot, seed, ALPHA),
            "stock": TS.cluster_bootstrap(r, stock, n_boot, seed, ALPHA)}


def turnaround_sensitivity(trades: pd.DataFrame, days_by_symbol: dict, n_boot: int = N_BOOT, seed=None) -> dict:
    """BEFORE (all resolved CONFIG/ARM trades) vs AFTER (trades with no flagged day in their window removed)."""
    seed = SEED_TURNAROUND if seed is None else seed
    t = select_trades(trades)
    bad = trade_bad_mask(t, days_by_symbol)
    kept, gone = t[~bad].reset_index(drop=True), t[bad].reset_index(drop=True)
    listed = gone.sort_values(["signal_date", "symbol"]).head(N_DROPPED_LISTED)
    return {"config": CONFIG, "arm": ARM, "pre_days": PRE_DAYS,
            "before": _trade_block(t, n_boot, seed),
            "after": _trade_block(kept, n_boot, seed),
            "n_dropped": int(bad.sum()),
            "dropped_mean_r": _finite_mean(gone["r_net"]),
            "dropped_win_rate": (float((gone["r_net"].to_numpy(dtype=float) > 0).mean()) if len(gone) else None),
            "dropped": [{"symbol": row["symbol"], "signal_date": row["signal_date"], "exit_date": row["exit_date"],
                         "r_net": row["r_net"]} for _, row in listed.iterrows()]}


# ── events ────────────────────────────────────────────────────────────────────────────────────────

def signature_masks(rows: pd.DataFrame) -> dict:
    """{signature: bool mask over ``rows``}, built exactly as the published study built them: rows are liquid
    by construction, ranks are within each event week over the FULL table."""
    panel = mtf.add_xs_ranks(rows.assign(liquid=True))
    return {s: Screen(mtf.SIGNATURES[s]).mask(panel) for s in SIGS}


def event_bad_mask(rows: pd.DataFrame, days_by_symbol: dict) -> np.ndarray:
    """True for event rows with a flagged day in [week, week + EVENT_WEEKS_FWD weeks] (both inclusive)."""
    weeks = pd.to_datetime(rows["week"])
    ends = weeks + pd.Timedelta(weeks=EVENT_WEEKS_FWD)
    return A.contaminated_mask(days_by_symbol, rows["symbol"].to_numpy(), weeks, ends)


def event_sensitivity(rows: pd.DataFrame, days_by_symbol: dict, weeks=HELD_OUT_WEEKS, masks=None,
                      n_boot: int = N_BOOT, seed=None, alpha: float = ALPHA_EVENT) -> dict:
    """Per signature: BEFORE (mask & held-out weeks & finite outcome) vs AFTER (also not contaminated).

    The masks and the date-matched control are computed on the FULL table before any row is dropped.
    Each block holds n, mean_ret, mean_excess (outcome minus the control) and the event-clustered interval.
    """
    seed = SEED_EVENTS if seed is None else seed
    if masks is None:
        masks = signature_masks(rows)
    ret = rows[OUTCOME].to_numpy(dtype="float64")
    fin = np.isfinite(ret)
    wk = pd.to_datetime(rows["week"]).to_numpy(dtype="datetime64[ns]").astype("int64")
    control = _week_mean(wk, ret)                                  # FULL table, once; never after a drop
    excess = ret - control
    held = np.isin(wk, _week_int(list(weeks)))
    bad = event_bad_mask(rows, days_by_symbol)

    def block(sel):
        x = excess[sel]
        return {"n": int(sel.sum()), "mean_ret": _finite_mean(ret[sel]), "mean_excess": _finite_mean(x),
                "ci": TS.cluster_bootstrap(x, wk[sel], n_boot, seed, alpha)}

    ctl_rows = held & fin
    out = {"outcome": OUTCOME, "weeks": [str(pd.Timestamp(w).date()) for w in weeks], "alpha": alpha,
           "horizon_weeks": EVENT_WEEKS_FWD,
           "control": {"n_rows": int(ctl_rows.sum()), "n_contaminated": int((ctl_rows & bad).sum()),
                       "share_contaminated": (float((ctl_rows & bad).sum() / ctl_rows.sum()) if ctl_rows.any() else None)},
           "signatures": {}}
    for s, m in masks.items():
        m = np.asarray(m, dtype=bool)
        before, after, gone = m & held & fin, m & held & fin & ~bad, m & held & fin & bad
        out["signatures"][s] = {"before": block(before), "after": block(after),
                                "n_dropped": int(gone.sum()),
                                "dropped_mean_ret": _finite_mean(ret[gone]),
                                "dropped_mean_excess": _finite_mean(excess[gone])}
    return out


# ── by-year summary ───────────────────────────────────────────────────────────────────────────────

def summary_by_year(days_by_symbol: dict, trades_bad: pd.DataFrame, rows_bad: pd.DataFrame, years=None) -> pd.DataFrame:
    """Per calendar year: artifact_days (flagged bars), stocks_flagged (distinct stocks with one that year),
    c0_trades_dropped (dropped trades, by signal_date year), event_rows_dropped (dropped event rows, by week year).
    ``trades_bad`` has a ``signal_date`` column, ``rows_bad`` a ``week`` column. Years are contiguous, spanning
    the data and, if given, ``years`` (the years the price data covers, so quiet years show as zeros)."""
    cols = ["artifact_days", "stocks_flagged", "c0_trades_dropped", "event_rows_dropped"]
    syms, flag_years = [], []
    for sym, days in days_by_symbol.items():
        d = pd.DatetimeIndex(days)
        syms.extend([sym] * len(d))
        flag_years.extend(d.year.tolist())
    flagged = pd.DataFrame({"symbol": syms, "year": np.asarray(flag_years, dtype=np.int64)})
    t_years = pd.to_datetime(trades_bad["signal_date"]).dt.year if len(trades_bad) else pd.Series([], dtype=np.int64)
    r_years = pd.to_datetime(rows_bad["week"]).dt.year if len(rows_bad) else pd.Series([], dtype=np.int64)
    extra = np.asarray([] if years is None else list(years), dtype=np.int64)
    all_years = np.concatenate([flagged["year"].to_numpy(), t_years.to_numpy(dtype=np.int64),
                                r_years.to_numpy(dtype=np.int64), extra])
    if not len(all_years):
        return pd.DataFrame(columns=cols, index=pd.Index([], name="year", dtype=np.int64)).astype(np.int64)
    idx = pd.Index(range(int(all_years.min()), int(all_years.max()) + 1), name="year")
    out = pd.DataFrame(index=idx)
    out["artifact_days"] = flagged.groupby("year").size().reindex(idx, fill_value=0)
    out["stocks_flagged"] = flagged.groupby("year")["symbol"].nunique().reindex(idx, fill_value=0)
    out["c0_trades_dropped"] = t_years.value_counts().reindex(idx, fill_value=0)
    out["event_rows_dropped"] = r_years.value_counts().reindex(idx, fill_value=0)
    return out.astype(np.int64)[cols]


# ── scanning the cache ────────────────────────────────────────────────────────────────────────────

_DAILY: dict = {}


def _flag_chunk(symbols) -> dict:
    out = {}
    for s in symbols:
        d = A.artifact_days(_DAILY[s])
        if len(d):
            out[s] = d
    return out


def compute_artifact_days(daily_by_symbol: dict, workers: int = 1) -> dict:
    """{symbol: DatetimeIndex of flagged days}, only symbols with at least one. A fork pool maps symbol chunks
    when workers > 1 (the frames are inherited by the children, not pickled); workers <= 1 runs in-process."""
    global _DAILY
    syms = sorted(daily_by_symbol)
    if workers <= 1 or len(syms) < 2:
        _DAILY = daily_by_symbol
        try:
            return _flag_chunk(syms)
        finally:
            _DAILY = {}
    chunks = [list(c) for c in np.array_split(np.array(syms, dtype=object), min(len(syms), workers * 4))]
    _DAILY = daily_by_symbol
    try:
        with mp.get_context("fork").Pool(workers) as pool:
            parts = pool.map(_flag_chunk, chunks)
    finally:
        _DAILY = {}
    return {k: v for part in parts for k, v in part.items()}


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


def _f(x, nd=4):
    return "n/a" if x is None else f"{x:+.{nd}f}"


def _ci_text(ci):
    return "[n/a]" if not ci or ci.get("lo") is None else f"[{ci['lo']:+.4f}, {ci['hi']:+.4f}]"


def _print_turnaround(tr: dict) -> None:
    print(f"\nTURNAROUND  {tr['config']} / {tr['arm']}  (resolved trades; drop if a flagged day lies in "
          f"[signal-{tr['pre_days']}d, exit])")
    print(f"  {'':8s}{'n':>7s}{'mean R':>10s}   {'month 95% CI':26s}{'stock 95% CI':26s}")
    for name in ("before", "after"):
        b = tr[name]
        print(f"  {name:8s}{b['n']:7d}{_f(b['mean_r']):>10s}   {_ci_text(b['month']):26s}{_ci_text(b['stock']):26s}")
    print(f"  dropped {tr['n_dropped']} trades: mean R {_f(tr['dropped_mean_r'])}, win rate "
          f"{'n/a' if tr['dropped_win_rate'] is None else format(tr['dropped_win_rate'], '.3f')}")


def _print_events(ev: dict) -> None:
    print(f"\nEVENTS  held-out mature SPY events, {ev['outcome']}, excess vs the same-week all-stock mean, "
          f"{100 * (1 - ev['alpha']):.2f}% event-clustered interval")
    c = ev["control"]
    print(f"  control rows in the held-out weeks: {c['n_rows']}, contaminated {c['n_contaminated']} "
          f"({'n/a' if c['share_contaminated'] is None else format(c['share_contaminated'], '.3f')})")
    print(f"  {'sig':4s}{'':8s}{'n':>7s}{'mean ret':>10s}{'excess':>10s}   {'interval':24s}")
    for s, d in ev["signatures"].items():
        for name in ("before", "after"):
            b = d[name]
            print(f"  {s:4s}{name:8s}{b['n']:7d}{_f(b['mean_ret']):>10s}{_f(b['mean_excess']):>10s}   {_ci_text(b['ci']):24s}")
        print(f"  {'':4s}dropped {d['n_dropped']} rows: mean ret {_f(d['dropped_mean_ret'])}, "
              f"mean excess {_f(d['dropped_mean_excess'])}")


# ── main ──────────────────────────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--history", default=str(HISTORY), help=f"daily cache directory (default {HISTORY})")
    ap.add_argument("--trades", default=str(TRADES), help=f"turnaround trades parquet (default {TRADES})")
    ap.add_argument("--event-rows", default=str(EVENT_ROWS), help=f"event-study rows parquet (default {EVENT_ROWS})")
    ap.add_argument("--out", default=str(OUT_JSON), help=f"result JSON (default {OUT_JSON})")
    ap.add_argument("--summary-only", action="store_true", help="print the by-year table of flagged days and exit; writes nothing")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N symbols (sorted) of each part")
    ap.add_argument("--workers", type=int, default=1, help="fork-pool workers for the artifact scan (default 1)")
    ap.add_argument("--n-boot", type=int, default=N_BOOT, help=f"bootstrap draws (default {N_BOOT})")
    ap.add_argument("--seed", type=int, default=None, help="override BOTH published bootstrap seeds")
    args = ap.parse_args(argv)

    t0 = time.time()
    trades = select_trades(pd.read_parquet(args.trades, columns=TRADE_COLUMNS))
    rows = pd.read_parquet(args.event_rows)
    t_syms, r_syms = sorted(trades["symbol"].unique()), sorted(rows["symbol"].unique())
    if args.limit is not None:
        t_syms, r_syms = t_syms[:args.limit], r_syms[:args.limit]
        trades = trades[trades["symbol"].isin(t_syms)].reset_index(drop=True)
        rows = rows[rows["symbol"].isin(r_syms)].reset_index(drop=True)
    wanted = sorted(set(t_syms) | set(r_syms))
    print(f"trades {len(trades)} ({len(t_syms)} stocks), event rows {len(rows)} ({len(r_syms)} stocks); "
          f"scanning {len(wanted)} stocks in {args.history}", flush=True)

    daily = mtf.load_daily(args.history, wanted)
    days = compute_artifact_days(daily, args.workers)
    n_with_data = len(daily)
    data_years = (range(min(d.index[0].year for d in daily.values()), max(d.index[-1].year for d in daily.values()) + 1)
                  if daily else [])
    del daily
    n_days = int(sum(len(v) for v in days.values()))
    print(f"{n_with_data} stocks had daily bars; {n_days} flagged days in {len(days)} stocks ({time.time() - t0:.0f}s)")

    tb, rb = trade_bad_mask(trades, days), event_bad_mask(rows, days)
    by_year = summary_by_year(days, trades[tb], rows[rb], data_years)
    print("\nFLAGGED DAYS BY YEAR (c0_trades_dropped by signal year, event_rows_dropped by event-week year)")
    print(by_year.to_string())
    if args.summary_only:
        return 0

    n_boot = args.n_boot
    turn = turnaround_sensitivity(trades, days, n_boot, args.seed)
    ev = event_sensitivity(rows, days, HELD_OUT_WEEKS, None, n_boot, args.seed, ALPHA_EVENT)
    _print_turnaround(turn)
    _print_events(ev)

    result = {"rule": {"text": RULE_TEXT, "min_abs_ret": A.MIN_ABS_RET, "min_prior_dvol": A.MIN_PRIOR_DVOL,
                       "vol_spike": A.VOL_SPIKE, "vol_win": A.VOL_WIN, "vol_min_days": A.VOL_MIN_DAYS},
              "post_hoc": True,
              "data": {"history": args.history, "n_stocks_requested": len(wanted), "n_stocks_with_daily": n_with_data,
                       "n_stocks_flagged": len(days), "n_flagged_days": n_days, "limited_run": args.limit is not None,
                       "seed_turnaround": SEED_TURNAROUND if args.seed is None else args.seed,
                       "seed_events": SEED_EVENTS if args.seed is None else args.seed, "n_boot": n_boot},
              "turnaround": turn, "events": ev, "by_year": by_year.reset_index()}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_jsonable(result), indent=1, allow_nan=False))
    print(f"\nwrote {out} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
