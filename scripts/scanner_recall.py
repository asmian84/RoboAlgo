"""
Would the scanners this repo already has have flagged the big 2021-2026 winners near their lows? The
adapters turn each scanner into a per-stock, per-week flag; the runner replays them over the whole price
cache and reads the flags around each named winner's ex-post low.

    python scripts/scanner_recall.py                               # full run, writes docs/mtf_case_study/scanner_recall.{json,md}
    python scripts/scanner_recall.py --limit-batches 3 --out-dir ~/.roboalgo/mtf/recall_smoke    # smoke test, writes nothing under docs/
    python scripts/scanner_recall.py --symbols NVDA AMD            # report only these cohort names

THE QUESTION. For a stock whose best run began at a low week L (chosen EX POST, see scripts/mtf_case_study.py),
did any existing scanner flag it in the window [L - RECALL_BEFORE_WEEKS, L + RECALL_AFTER_WEEKS]? Three
families are adapted, each exactly as the repo already defines it, with nothing re-tuned:

  FINDER       the three Finder presets ("signal", "pennyking", "maker"), replayed over price history with
               strategy_validation/finder_backtest.py (the same replay that scripts/validate_finder.py
               gates). FINDER_DESCRIPTION says what a flag means.
  SETUPS       the 20 registered chart setups that have a rule (setup_engine/registry/registry.json): 17
               learned round-1 rules (rejected on robustness), 3 round-2 textbook rules (approved,
               provisional). The 3 round-1 textbook entries have EMPTY rule lists (rejected, no edge): they
               cannot be evaluated and are not listed. A registered rule is read on Friday observation dates
               from setup_engine/features.build_observations, with the cross-sectional rs_ ranks computed
               against the universe the rules were learned on (50-day dollar volume >= 5e5).
  TURNAROUND   the pre-registered rule of scripts/study_turnaround.py (a stock 70%+ below its 2-year high
               with a higher low, breaking out to a 20-day high on more than twice its usual volume), as
               RAW daily signals: no cooldown, no signal window, no exit simulation.

PRE-REGISTERED. The presets, the rule texts, the 8-week lead and the 52-week lag of the recall window are
fixed here before any recall number is computed. Recall is a DESCRIPTION of the scanners against hindsight
winners: a high recall proves nothing about an edge (a scanner that flags everything has perfect recall), so
the runner must report each scanner's flag RATE over the whole liquid universe next to its recall.

NO LOOK-AHEAD. Every flag at bar t uses bars <= t only: the Finder features and the screen indicators are
causal (each is tested on its own), the turnaround bottom state is read on the bar BEFORE the trigger bar,
and a registry observation at a Friday sees a weekly bar only once its Friday has closed and a monthly bar
only once its month has ended (setup_engine/features.py). The recall window is anchored on the ex-post low,
so it is an evaluation window, never a trading rule. Flags are attributed to the week (the Friday ending the
week) containing the bar; a stock flagged on a Wednesday is flagged "in that week" with its first flagged
date kept, so the runner can tell a flag that came before the low from one that came after.

ORDER MODEL. Not used here: these are signals, not trades. A signal at a close would be traded at the next
open in every simulation in this repo.

KNOWN BIASES (reported with every result). Survivorship: the price cache holds today's listings only, so
winners are over-represented among the stocks that exist to be flagged. Yahoo adjusts old prices
retroactively for splits (reverse-split stocks show price bands they never traded at; the Finder price band
and the turnaround "Price over $1" filter are affected). No dividend adjustment. The registry rs_ ranks are
computed over the stocks the caller supplies, so a smoke run on a few batches ranks against a smaller
universe than the rules were learned on. Part-formed final weeks (data ending mid-week) get a label in the
future; flags there are real bars but belong to an incomplete week.

THE RUNNER. Every stock of the cache is processed once (setup_engine.mtf.map_universe, fork pool) into weekly
rows: the weekly multi-timeframe table (setup_engine/mtf.py) from RECALL_START on, a liquid flag, the
Finder / turnaround flags attributed to their week, and the registry's Friday observations. Rows are kept for
LIQUID stocks (mtf.is_liquid) and for the cohort names. The parent ranks across stocks (mtf.add_xs_ranks for
Performance Rank; features.add_relative_strength for the registered rules' rs_ columns, over EVERY stock's
observations of that Friday, not over the cohort), evaluates the five pre-registered MTF signatures
(mtf.SIGNATURES: S1-S4 and the market-beta baseline B0) and joins the registry flags. For each cohort name the
ex-post low week is mtf.best_run on its weekly closes since RUN_START (the same function as the case study),
and for each scanner the first flagged week in [low - 8, low + 52] weeks gives weeks-from-low, the price
move off the low's close and the PRECISION PROXY: how many OTHER liquid stocks the same scanner flagged that
week (a scanner that flags a quarter of the market is not telling you much). Each scanner's overall flag
rate over the liquid universe is reported next to its recall.

The six cohort names that are not in the stock cache (TSM, ASML, ARM, SPY, QQQ, SMH) come from the benchmark
directory; cohort names that sit in batches outside a --limit-batches smoke run are loaded from ALL batches.
Both kinds are "extras": they are scanned with the same function but are NOT part of the ranking universe or
of the precision counts, and their Performance Rank / rs_ ranks are the share of that week's universe values
at or below theirs. The multi-timeframe columns and the benchmark (SPY) are clipped to the cache's last date
(yfinance series run to today). The Finder replay runs over each stock's full history: measured at a few
milliseconds per stock, it does not need restricting.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from setup_engine import mtf
from setup_engine.screen import Screen, indicators

# ── pre-registered choices (fixed before any result is seen) ──────────────────────────────────────

FINDER_PRESETS = ("signal", "pennyking", "maker")

# Copies of scripts/study_turnaround.py BOTTOM / TRIGGER / LIQUIDITY (tests/test_study/test_scanner_recall.py
# compares them with the script, so a drift in either is caught).
TURNAROUND_BOTTOM = {"2-Year High": "70% or more below", "Monthly Low": "Higher low"}
TURNAROUND_TRIGGER = {"20-Day High": "New High", "Relative Volume": "Over 2"}
TURNAROUND_LIQUIDITY = {"Price": "Over $1", "Dollar Volume 20d": "Over 1M"}

# = scripts/build_setup_dataset.MIN_DOLLAR_VOLUME: the 50-day dollar-volume floor of the universe the
# registered rules' rs_ ranks were learned on.
REGISTRY_MIN_DOLLAR_VOLUME = 5e5

RECALL_BEFORE_WEEKS = 8      # a flag this many weeks before the low week still counts
RECALL_AFTER_WEEKS = 52      # ... and this many weeks after it

FINDER_DESCRIPTION = (
    "A Finder flag on day t means ALL of: a golden cross (50-day average above the 200-day average and the "
    "close above the 50-day average); a compressed range among the last 30 compression rows with a "
    "compression score >= 50; no open gap above the preset's max_gap_pct in the last 20 bars; a 20-day "
    "average volume >= the preset's min_volume; the close inside the preset's price band; at least "
    "finder_backtest.MIN_ROWS bars of history; and valid trade levels (entry, stop, target) taken from the "
    "compressed range's driver bar. Presets: signal (no price band, 100k volume, 3% gaps), pennyking "
    "(price <= 5, 500k volume, 8% gaps), maker (price 5-20, 250k volume, 6% gaps)."
)


# ── the runner's pre-registered choices (fixed before any recall number is computed) ──────────────

# The 46 named winners / benchmarks: the same ordered unique union as scripts/mtf_case_study.COHORT.
USER_NAMES = ("AMD", "NVDA", "MSFT", "AMZN", "TSLA")
CHIPS = ("NVDA", "AMD", "AVGO", "MU", "QCOM", "INTC", "TXN", "AMAT", "LRCX", "KLAC", "MRVL", "ON", "NXPI",
         "MCHP", "ADI", "MPWR", "TER", "SMCI", "TSM", "ASML", "ARM", "ALAB", "CRDO", "COHR", "LITE")
AI_INFRA = ("ANET", "VRT", "WDC", "STX", "DELL", "GLW", "CIEN", "CLS", "SNDK", "NBIS", "PLTR", "APP")
MEGA = ("META", "GOOGL", "AAPL")
BENCH_NAMES = ("SPY", "QQQ", "SMH")
COHORT_RECALL = tuple(dict.fromkeys(USER_NAMES + CHIPS + AI_INFRA + MEGA + BENCH_NAMES))
NOT_IN_CACHE = ("TSM", "ASML", "ARM", "SPY", "QQQ", "SMH")     # read from mtf.BENCH_DIR, not the stock cache

RUN_START = "2021-10-08"       # best_run looks for the low from this week on (the case study's RUN_START)
RECALL_START = "2021-06-01"    # weekly rows, Friday observations and flag attribution start here
RS_BENCH = "SPY"               # benchmark of the rs_ / bm_ columns (S4)
SIGNATURE_NAMES = ("S1", "S2", "S3", "S4", "B0")   # keys of mtf.SIGNATURES, in table order
DATA_GAP_DAYS = 30             # a run with a larger gap between weekly labels is flagged (halted / stitched data)
DEFAULT_HISTORY = mtf.DEFAULT_HISTORY
BENCH_DIR = mtf.BENCH_DIR
OUT_JSON = ROOT / "docs" / "mtf_case_study" / "scanner_recall.json"
OUT_MD = ROOT / "docs" / "mtf_case_study" / "scanner_recall.md"
SMOKE_DIR = Path.home() / ".roboalgo" / "mtf" / "recall_smoke"
FULL_UNIVERSE_STOCKS = 11519   # stocks in the full cache (manifest), for the runtime projection
PROJECTION_WORKERS = 12


def _signature_inputs() -> tuple:
    """Columns of the weekly table the five signatures read (cross-sectional ranks excluded) plus the ones
    the report needs: w_close (price off the low), w_ret13 / w_ret26 (the Performance Rank sources)."""
    cols = ["w_close", "w_ret13", "w_ret26"]
    for name in SIGNATURE_NAMES:
        cols += [c for c in Screen(mtf.SIGNATURES[name]).fields() if c not in mtf.XS_COLUMNS]
    return tuple(dict.fromkeys(cols))


MTF_INPUT_COLUMNS = _signature_inputs()

# Short column labels for the three approved round-2 rules; any other id is shown as is.
REGISTRY_SHORT = {"r2-textbook_compression_golden_cross": "r2:compr_gc",
                  "r2-textbook_stage2_trend": "r2:stage2",
                  "r2-textbook_tight_base_breakout": "r2:tight_base"}


# ── week labels ───────────────────────────────────────────────────────────────────────────────────

def _week_label(dates) -> pd.DatetimeIndex:
    """The Friday ending the week of each date. Identical to setup_engine.mtf.week_label."""
    return pd.DatetimeIndex(dates).to_period("W-FRI").end_time.normalize()


# ── Finder presets ────────────────────────────────────────────────────────────────────────────────

def finder_daily_flags(daily: pd.DataFrame, presets: Sequence[str] = FINDER_PRESETS) -> dict[str, np.ndarray]:
    """{preset: bool array aligned to the rows of ``daily``}: the days the Finder would have listed the stock.

    ``daily`` needs an increasing DatetimeIndex and open/high/low/close/volume (clean it first; the arrays are
    aligned to its rows, so this function does not drop any). Fewer than finder_backtest.MIN_ROWS rows gives
    all-False arrays, as the engine skips such instruments. Imports are inside the function: the Finder
    presets live in scanner_engine.finder, which pulls sqlalchemy and the database package.
    """
    from scanner_engine.finder import PRESETS
    from strategy_validation.finder_backtest import Rule, build_signals, compute_features

    unknown = [p for p in presets if p not in PRESETS]
    if unknown:
        raise ValueError(f"unknown Finder preset(s) {unknown}; available: {sorted(PRESETS)}")
    feats = compute_features(daily)
    if feats is None:
        return {p: np.zeros(len(daily), dtype=bool) for p in presets}
    return {p: np.asarray(build_signals(feats, Rule.from_criteria(PRESETS[p])).signal, dtype=bool)
            for p in presets}


# ── the turnaround rule ───────────────────────────────────────────────────────────────────────────

def _lag(a, k: int) -> np.ndarray:
    """a shifted k bars later (the value at bar t is a[t-k]); the first k bars are False.

    Copy of scripts/study_turnaround._lag.
    """
    a = np.asarray(a, dtype=bool)
    if k < 0:
        raise ValueError("lag must be >= 0")
    if k == 0:
        return a.copy()
    out = np.zeros_like(a)
    out[k:] = a[:-k]
    return out


def turnaround_daily_flags(daily: pd.DataFrame) -> np.ndarray:
    """Bool array aligned to the rows of ``daily``: bottom state on the PRIOR bar AND trigger AND liquid.

    The raw rule of scripts/study_turnaround.py (its primary configuration): no cooldown, no signal window,
    no requirement that a next bar exists. ``daily`` must be clean (unique increasing index); the array is
    aligned to its rows.
    """
    ind = indicators(daily)
    bottom, trigger, liquidity = Screen(TURNAROUND_BOTTOM), Screen(TURNAROUND_TRIGGER), Screen(TURNAROUND_LIQUIDITY)
    b_lag = _lag(bottom.mask(ind), 1)
    bv_lag = _lag(bottom.valid(ind), 1)
    return b_lag & bv_lag & trigger.mask(ind) & liquidity.mask(ind)


# ── the setup registry ────────────────────────────────────────────────────────────────────────────

def registry_setups(reg: Optional[dict] = None) -> list[dict]:
    """The registered setups that HAVE a rule (20 of 23). Entries with an empty rule list are not evaluable."""
    if reg is None:
        from setup_engine import learn as L
        reg = L.load_registry()
    return [s for s in reg.get("setups", []) if s.get("rule")]


def registry_features(setups: Sequence[dict]) -> list[str]:
    """Union of the per-bar features the rules read, sorted. The rs_ ranks are excluded: they are computed
    across stocks from features.RS_SOURCES, not read from a single stock's bars."""
    feats = {c["feature"] for s in setups for c in s["rule"] if not c["feature"].startswith("rs_")}
    return sorted(feats)


def registry_obs(symbol: str, daily: pd.DataFrame, fridays: pd.DatetimeIndex,
                 features: Optional[Sequence[str]] = None) -> Optional[pd.DataFrame]:
    """One stock's registry inputs on the given Friday observation dates, or None.

    features.build_observations with the registry's dollar-volume floor and NO labels (so no fwd_ column and
    nothing from after the observation date). Keeps symbol, date, dvol50, the registry features and every
    features.RS_SOURCES column (the caller ranks those across stocks with features.add_relative_strength).
    ``features`` defaults to registry_features(registry_setups()); a runner should compute that once and pass
    it, instead of re-reading the registry for every stock.
    """
    from setup_engine import features as FT

    obs = FT.build_observations(symbol, daily, fridays, REGISTRY_MIN_DOLLAR_VOLUME, with_labels=False)
    if obs is None:
        return None
    wanted = list(features) if features is not None else registry_features(registry_setups())
    missing = [c for c in wanted if c not in obs.columns]
    if missing:
        raise ValueError(f"registry features {missing} are not produced by features.build_observations")
    cols = list(dict.fromkeys(["symbol", "date", "dvol50", *wanted, *FT.RS_SOURCES]))
    return obs[[c for c in cols if c in obs.columns]].reset_index(drop=True)


def registry_flags(obs_ranked: pd.DataFrame, setups: Sequence[dict]) -> pd.DataFrame:
    """Bool frame, one column per setup id, aligned to ``obs_ranked``: the rows where the setup's rule holds.

    ``obs_ranked`` must already carry the rs_ columns (features.add_relative_strength over the whole panel).
    A missing value never satisfies a condition (learn.Condition.mask compares NaN as False).
    """
    from setup_engine import learn as L

    out = {}
    for s in setups:
        rule = L.Rule.from_json(s["rule"])
        missing = [c.feature for c in rule.conditions if c.feature not in obs_ranked.columns]
        if missing:
            raise ValueError(f"setup {s['id']} reads {missing}, which are not in the table "
                             "(did you run features.add_relative_strength over the whole panel?)")
        out[s["id"]] = np.asarray(rule.mask(obs_ranked), dtype=bool)
    return pd.DataFrame(out, index=obs_ranked.index, columns=[s["id"] for s in setups])


# ── from days to weeks ────────────────────────────────────────────────────────────────────────────

def to_weekly(dates, flags: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily flags -> weekly. Returns (any_flag, first_date), both indexed by week label ("week", the Friday).

    any_flag: bool per flag name, True where any day of the week is flagged. first_date: the earliest flagged
    date of that week (NaT if none). One row per week that has at least one bar in ``dates``.
    """
    dates = pd.DatetimeIndex(dates)
    wk = _week_label(dates)
    any_cols, first_cols = {}, {}
    for name, arr in flags.items():
        arr = np.asarray(arr, dtype=bool)
        if len(arr) != len(dates):
            raise ValueError(f"flag {name!r} has {len(arr)} values for {len(dates)} dates")
        any_cols[name] = pd.Series(arr, index=dates).groupby(wk).any()
        first_cols[name] = pd.Series(dates.where(arr), index=dates).groupby(wk).min()
    any_df = pd.DataFrame(any_cols, index=pd.Index(sorted(set(wk)), name="week"), columns=list(flags))
    first_df = pd.DataFrame(first_cols, index=any_df.index, columns=list(flags))
    any_df = any_df.astype(bool)
    return any_df, first_df.astype("datetime64[ns]")


def first_flag(weekly_flags: pd.Series, low_week, before: int = RECALL_BEFORE_WEEKS,
               after: int = RECALL_AFTER_WEEKS) -> Optional[pd.Timestamp]:
    """The first week label L that is flagged with low_week - before weeks <= L <= low_week + after weeks, else None.

    ``weekly_flags`` is a bool Series indexed by week label. Both ends of the window are inclusive.
    """
    low_week = pd.Timestamp(low_week)
    lo, hi = low_week - pd.Timedelta(weeks=before), low_week + pd.Timedelta(weeks=after)
    hit = weekly_flags.index[weekly_flags.fillna(False).astype(bool).to_numpy()]
    hit = hit[(hit >= lo) & (hit <= hi)]
    return pd.Timestamp(hit.min()) if len(hit) else None




# ═════════════════════════════════════════════════════════════════════════════════════════════════
# The runner
# ═════════════════════════════════════════════════════════════════════════════════════════════════

# ── one stock -> weekly rows and Friday observations ──────────────────────────────────────────────

def scanner_keys(setups: Sequence[dict]) -> list[str]:
    """Every scanner column, in table order: finder_<preset> x3, turnaround, registry:<id> x20, mtf:<name> x5."""
    return ([f"finder_{p}" for p in FINDER_PRESETS] + ["turnaround"]
            + [f"registry:{s['id']}" for s in setups] + [f"mtf:{n}" for n in SIGNATURE_NAMES])


def recall_rows(symbol: str, daily: pd.DataFrame, spy: Optional[pd.DataFrame], as_of, fridays: pd.DatetimeIndex,
                cohort, reg_features: Optional[Sequence[str]] = None) -> Optional[dict]:
    """The map_universe function: one stock -> {"symbol", "weekly", "registry", "info", "error"}, or None.

    None only when the stock has fewer than mtf.MIN_DAILY_ROWS bars up to ``as_of``; a stock with nothing to
    keep (never liquid, no registry observation) returns the record with weekly = registry = None, so the
    caller can still count it as scanned.

    weekly: one row per COMPLETE week from RECALL_START on that is liquid (mtf.is_liquid) or belongs to a
    cohort name, with the columns MTF_INPUT_COLUMNS, ``liquid``, the daily Finder / turnaround flags attributed
    to their week (finder_<preset>, turnaround: True if any day of the week is flagged) and the first flagged
    date of each week (first_<flag>, NaT if none). registry: the stock's registry inputs on ``fridays``
    (registry_obs; every stock with a dollar volume of 5e5 or more, liquid or not, because the registered
    rules' rs_ ranks are learned against that universe). info: first / last bar for a cohort name.
    ``daily`` must be clean; it is clipped to ``as_of`` (and ``spy`` is clipped by mtf_features), so nothing
    after the cache's last date is read. A non-cohort stock that raises is returned as {"error": ...} and
    counted by the caller; a cohort name that raises propagates.
    """
    as_of = pd.Timestamp(as_of)
    daily = daily[daily.index <= as_of]
    if len(daily) < mtf.MIN_DAILY_ROWS:
        return None
    try:
        return _recall_rows(symbol, daily, spy, as_of, fridays, symbol in cohort, reg_features)
    except Exception as exc:                          # noqa: BLE001 - counted and reported, never silent
        if symbol in cohort:
            raise
        return {"symbol": symbol, "weekly": None, "registry": None, "info": None,
                "error": f"{type(exc).__name__}: {exc}"}


def _recall_rows(symbol, daily, spy, as_of, fridays, in_cohort, reg_features) -> Optional[dict]:
    wf = mtf.mtf_features(daily, spy, as_of)
    wf = wf[wf.index >= pd.Timestamp(RECALL_START)]
    liquid = mtf.is_liquid(wf) if len(wf) else np.zeros(0, dtype=bool)

    flags = {f"finder_{p}": a for p, a in finder_daily_flags(daily).items()}
    flags["turnaround"] = turnaround_daily_flags(daily)
    any_w, first_w = to_weekly(daily.index, flags)

    weekly = None
    keep = liquid | in_cohort if len(wf) else liquid
    if keep.any():
        w = wf.loc[keep, list(MTF_INPUT_COLUMNS)].copy()
        w["liquid"] = liquid[keep]
        for k in flags:
            w[k] = any_w[k].reindex(w.index, fill_value=False).to_numpy(dtype=bool)
            w[f"first_{k}"] = first_w[k].reindex(w.index).to_numpy()
        w.insert(0, "week", w.index.to_numpy())
        w.insert(0, "symbol", symbol)
        weekly = w.reset_index(drop=True)

    obs = registry_obs(symbol, daily, fridays, reg_features)
    info = {"first_bar": daily.index[0], "last_bar": daily.index[-1], "n_bars": len(daily)} if in_cohort else None
    return {"symbol": symbol, "weekly": weekly, "registry": obs, "info": info, "error": None}


# ── cross-sectional ranks ─────────────────────────────────────────────────────────────────────────

def extra_ranks(base: pd.DataFrame, extra: pd.DataFrame, key: str, cols: dict) -> pd.DataFrame:
    """Copy of ``extra`` with, for each ``src -> dst`` in ``cols``, dst = the share of ``base`` values at or
    below the row's own value among the base rows with the same ``key`` (NaN if the value is not finite or the
    base has no finite value for that key). The rows of ``extra`` are ranked AGAINST the base universe and
    never enter it, so they cannot move anyone else's rank."""
    out = extra.copy()
    for src, dst in cols.items():
        b_key = base[key].to_numpy()
        b_val = base[src].to_numpy(dtype="float64")
        ok = np.isfinite(b_val)
        sorted_vals = {k: np.sort(g) for k, g in pd.Series(b_val[ok]).groupby(b_key[ok])}
        e_key, e_val = out[key].to_numpy(), out[src].to_numpy(dtype="float64")
        res = np.full(len(out), np.nan)
        for i in range(len(out)):
            arr = sorted_vals.get(e_key[i])
            if arr is not None and len(arr) and np.isfinite(e_val[i]):
                res[i] = np.searchsorted(arr, e_val[i], side="right") / len(arr)
        out[dst] = res
    return out


def add_weekly_ranks(weekly: pd.DataFrame) -> pd.DataFrame:
    """xs_rank_ret13 / xs_rank_ret26 (mtf.add_xs_ranks) among the liquid rows of the ranking universe
    (``in_universe``); liquid extras are ranked against that universe (extra_ranks); everything else NaN."""
    base_mask = weekly["liquid"].to_numpy(dtype=bool) & weekly["in_universe"].to_numpy(dtype=bool)
    out = mtf.add_xs_ranks(weekly.assign(_base=base_mask), "week", "_base").drop(columns="_base")
    extra = weekly["liquid"].to_numpy(dtype=bool) & ~weekly["in_universe"].to_numpy(dtype=bool)
    if extra.any():
        ranked = extra_ranks(out[base_mask], out[extra], "week",
                             {"w_ret13": "xs_rank_ret13", "w_ret26": "xs_rank_ret26"})
        out.loc[extra, ["xs_rank_ret13", "xs_rank_ret26"]] = ranked[["xs_rank_ret13", "xs_rank_ret26"]].to_numpy()
    return out


def rank_registry(obs: pd.DataFrame) -> pd.DataFrame:
    """features.add_relative_strength over EVERY stock's observations of each Friday (``in_universe`` rows);
    extras are ranked against that universe. The result carries the rs_<source> columns the rules read."""
    from setup_engine import features as FT

    base = obs[obs["in_universe"].to_numpy(dtype=bool)]
    ranked = FT.add_relative_strength(base)
    extra = obs[~obs["in_universe"].to_numpy(dtype=bool)]
    if len(extra):
        ex = extra_ranks(base, extra, "date", {src: f"rs_{src}" for src in FT.RS_SOURCES if src in obs.columns})
        ranked = pd.concat([ranked, ex], ignore_index=True)
    return ranked.reset_index(drop=True)


# ── all stocks -> one panel with a flag per scanner ───────────────────────────────────────────────

def assemble(parts: Sequence[dict], setups: Sequence[dict]) -> pd.DataFrame:
    """Concatenate the per-stock results (each tagged with ``in_universe`` by ``collect``) into the weekly
    panel: ranks, the five MTF signatures, and the registry flags joined on (symbol, week). A registry flag is
    False where the stock has no Friday observation (dollar volume below 5e5, or too little history)."""
    frames = [p["weekly"] for p in parts if p["weekly"] is not None and len(p["weekly"])]
    if not frames:
        raise ValueError("no weekly rows: nothing is liquid in the data supplied")
    weekly = add_weekly_ranks(pd.concat(frames, ignore_index=True))
    for name in SIGNATURE_NAMES:
        weekly[f"mtf:{name}"] = Screen(mtf.SIGNATURES[name]).mask(weekly)

    obs_frames = [p["registry"] for p in parts if p["registry"] is not None and len(p["registry"])]
    reg_cols = [f"registry:{s['id']}" for s in setups]
    if obs_frames and setups:
        ranked = rank_registry(pd.concat(obs_frames, ignore_index=True))
        flags = registry_flags(ranked, setups)
        flags.columns = reg_cols
        keyed = pd.concat([ranked[["symbol", "date"]].rename(columns={"date": "week"}), flags], axis=1)
        weekly = weekly.merge(keyed, on=["symbol", "week"], how="left")
        weekly[reg_cols] = weekly[reg_cols].eq(True)           # NaN (no observation that Friday) -> False
    else:
        for c in reg_cols:
            weekly[c] = False
    return weekly


def weekly_counts(weekly: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    """Per week label: ``n_liquid`` and, per scanner, the number of flagged stocks, counting ONLY liquid rows
    of the ranking universe (an illiquid stock, or an extra, never raises a count)."""
    base = weekly[weekly["liquid"].to_numpy(dtype=bool) & weekly["in_universe"].to_numpy(dtype=bool)]
    g = base.groupby("week")
    out = g[list(keys)].sum().astype("int64")
    out.insert(0, "n_liquid", g.size().astype("int64"))
    return out


def flag_rates(counts: pd.DataFrame, keys: Sequence[str]) -> dict:
    """Each scanner's flag rate over the liquid universe: flagged stock-weeks / liquid stock-weeks."""
    total = int(counts["n_liquid"].sum())
    out = {}
    for k in keys:
        per_week = counts[k]
        out[k] = {"flagged_stock_weeks": int(per_week.sum()),
                  "flag_rate": float(per_week.sum() / total) if total else None,
                  "median_flagged_per_week": float(per_week.median()) if len(per_week) else None,
                  "max_flagged_in_a_week": int(per_week.max()) if len(per_week) else None}
    return out


# ── reading the flags around each name's low ──────────────────────────────────────────────────────

def _scanner_cell(sub: pd.DataFrame, counts: pd.DataFrame, key: str, low_week, low_close: float) -> dict:
    lo = low_week - pd.Timedelta(weeks=RECALL_BEFORE_WEEKS)
    hi = low_week + pd.Timedelta(weeks=RECALL_AFTER_WEEKS)
    window = counts.loc[(counts.index >= lo) & (counts.index <= hi), key]
    flags = sub[key].astype(bool)
    wk = first_flag(flags, low_week)
    cell = {"flagged_in_window": wk is not None, "ever_flagged": bool(flags.any()),
            "week": None, "first_date": None, "weeks_from_low": None, "pct_off_low": None,
            "n_flagged_liquid": None, "n_liquid_universe": None, "n_other_flagged": None,
            "median_flagged_in_window": float(window.median()) if len(window) else None}
    if wk is None:
        return cell
    own_counted = bool(sub.at[wk, "liquid"]) and bool(sub.at[wk, "in_universe"])
    n_flagged = int(counts.at[wk, key]) if wk in counts.index else 0
    first_col = f"first_{key}"
    cell.update(week=wk,
                first_date=sub.at[wk, first_col] if first_col in sub.columns else wk,
                weeks_from_low=int(round((wk - low_week).days / 7)),
                pct_off_low=float(sub.at[wk, "w_close"] / low_close - 1.0),
                n_flagged_liquid=n_flagged,
                n_liquid_universe=int(counts.at[wk, "n_liquid"]) if wk in counts.index else 0,
                n_other_flagged=n_flagged - int(own_counted))
    return cell


def cohort_report(weekly: pd.DataFrame, counts: pd.DataFrame, symbol: str, keys: Sequence[str],
                  info: Optional[dict] = None, in_cache: Optional[bool] = None) -> dict:
    """One cohort name: its ex-post best run (mtf.best_run on its weekly closes since RUN_START) and, per
    scanner, the first flag in [low - RECALL_BEFORE_WEEKS, low + RECALL_AFTER_WEEKS] with its precision proxy."""
    sub = weekly[weekly["symbol"] == symbol].sort_values("week").set_index("week")
    rec = {"symbol": symbol, "in_cache": in_cache, "weekly_rows": int(len(sub)),
           "history_start": (info or {}).get("first_bar"), "last_bar": (info or {}).get("last_bar"),
           "run": None, "scanners": {}}
    run = mtf.best_run(sub["w_close"], RUN_START) if len(sub) else None
    if run is None:
        return rec
    rec["run"] = {k: run[k] for k in ("low_week", "low_close", "peak_week", "peak_close", "mult", "weeks",
                                      "mdd_in_run", "max_gap_days")}
    rec["data_gap_warning"] = bool(run["max_gap_days"] > DATA_GAP_DAYS)
    for k in keys:
        rec["scanners"][k] = _scanner_cell(sub, counts, k, run["low_week"], run["low_close"])
    return rec


# ── tables ────────────────────────────────────────────────────────────────────────────────────────

def format_cell(cell: Optional[dict]) -> str:
    """"+Nw (+P%) / K" (weeks from the low, price vs the low's close, other liquid stocks flagged that week),
    "-" if the scanner did not flag the name in the window, "n/a" if the name has no run."""
    if cell is None:
        return "n/a"
    if not cell["flagged_in_window"]:
        return "-"
    return f"{cell['weeks_from_low']:+d}w ({cell['pct_off_low']:+.0%}) / {cell['n_other_flagged']}"


def _short(setup_id: str) -> str:
    return REGISTRY_SHORT.get(setup_id, setup_id)


def table_blocks(reports: Sequence[dict], setups: Sequence[dict]) -> list[dict]:
    """The recall table as two blocks of the same rows: the repo's existing scanners, then the MTF signatures."""
    approved = [s for s in setups if s.get("status") == "approved"]
    learned = [s for s in setups if s.get("status") != "approved"]

    def head(r):
        run = r["run"]
        if run is None:
            return [r["symbol"], "n/a", "n/a"]
        gap = "*" if r.get("data_gap_warning") else ""
        return [r["symbol"] + gap, run["low_week"].date().isoformat(), f"{run['mult']:.1f}x"]

    def learned_cell(r):
        if r["run"] is None:
            return "n/a"
        n = sum(r["scanners"][f"registry:{s['id']}"]["flagged_in_window"] for s in learned)
        return f"{n}/{len(learned)}"

    a_cols = ([(f"finder:{p}", f"finder_{p}") for p in FINDER_PRESETS] + [("turnaround", "turnaround")]
              + [(_short(s["id"]), f"registry:{s['id']}") for s in approved])
    block_a = {"title": "Existing scanners", "header": ["name", "low week", "run"] + [h for h, _ in a_cols]
               + [f"learned (of {len(learned)})"], "rows": []}
    block_b = {"title": "Multi-timeframe signatures (pre-registered)",
               "header": ["name", "low week", "run"] + [f"mtf:{n}" for n in SIGNATURE_NAMES], "rows": []}
    for r in reports:
        sc = r["scanners"]
        block_a["rows"].append(head(r) + [format_cell(sc.get(k)) for _, k in a_cols] + [learned_cell(r)])
        block_b["rows"].append(head(r) + [format_cell(sc.get(f"mtf:{n}")) for n in SIGNATURE_NAMES])
    return [block_a, block_b]


def render_console(blocks: Sequence[dict]) -> str:
    out = []
    for b in blocks:
        cols = [b["header"]] + b["rows"]
        widths = [max(len(str(row[i])) for row in cols) for i in range(len(b["header"]))]
        out.append(b["title"])
        out.append("  ".join(str(h).ljust(w) for h, w in zip(b["header"], widths)))
        out.append("  ".join("-" * w for w in widths))
        out.extend("  ".join(str(c).ljust(w) for c, w in zip(row, widths)) for row in b["rows"])
        out.append("")
    return "\n".join(out)


def render_markdown_tables(blocks: Sequence[dict]) -> str:
    out = []
    for b in blocks:
        out.append(f"### {b['title']}\n")
        out.append("| " + " | ".join(b["header"]) + " |")
        out.append("|" + "|".join(" --- " for _ in b["header"]) + "|")
        out.extend("| " + " | ".join(str(c).replace("|", "/") for c in row) + " |" for row in b["rows"])
        out.append("")
    return "\n".join(out)


def scanner_summary(reports: Sequence[dict], rates: dict, keys: Sequence[str]) -> list[dict]:
    """Per scanner: how many cohort names it flagged in the window / ever, next to its overall flag rate."""
    have = [r for r in reports if r["run"] is not None]
    rows = []
    for k in keys:
        cells = [r["scanners"][k] for r in have]
        rows.append({"scanner": k, "names_with_run": len(have),
                     "flagged_in_window": sum(c["flagged_in_window"] for c in cells),
                     "ever_flagged": sum(c["ever_flagged"] for c in cells),
                     "flag_rate": rates[k]["flag_rate"],
                     "median_flagged_per_week": rates[k]["median_flagged_per_week"]})
    return rows


def render_summary(rows: Sequence[dict]) -> str:
    head = ["scanner", "in window", "ever", "flag rate", "median flagged/wk"]
    body = [[r["scanner"], f"{r['flagged_in_window']}/{r['names_with_run']}", f"{r['ever_flagged']}/{r['names_with_run']}",
             "n/a" if r["flag_rate"] is None else f"{r['flag_rate']:.2%}",
             "n/a" if r["median_flagged_per_week"] is None else f"{r['median_flagged_per_week']:.0f}"] for r in rows]
    widths = [max(len(str(x[i])) for x in [head] + body) for i in range(len(head))]
    lines = ["  ".join(str(h).ljust(w) for h, w in zip(head, widths)), "  ".join("-" * w for w in widths)]
    lines += ["  ".join(str(c).ljust(w) for c, w in zip(row, widths)) for row in body]
    return "\n".join(lines)


def markdown_report(result: dict, blocks: Sequence[dict], summary: Sequence[dict]) -> str:
    sizes = result["sizes"]
    lines = [
        "# Scanner recall: would our scanners have flagged the big winners near their lows?",
        "",
        f"Data through {result['data_end']}. Weekly rows from {RECALL_START}. This is a DESCRIPTION of existing "
        "scanners against hindsight winners: the 46 names were chosen because they went up (selection on the "
        "outcome), the low of each is found ex post (`setup_engine.mtf.best_run` on weekly closes since "
        f"{RUN_START}), and the price cache holds today's listings only (survivorship). A high recall is not "
        "evidence of an edge: a scanner that flags everything has perfect recall.",
        "",
        "## What each scanner looks for",
        "",
        f"- **Finder presets** (`finder:signal`, `finder:pennyking`, `finder:maker`): {FINDER_DESCRIPTION}",
        "- **turnaround**: the pre-registered rule of `scripts/study_turnaround.py` as raw daily signals (70% or "
        "more below the 2-year high with a higher low on the prior bar, then a new 20-day high on more than "
        "twice the usual volume; price over $1, 20-day dollar volume over 1M).",
        "- **registry**: the 20 registered chart setups that have a rule, read on Friday observations with "
        "cross-sectional ranks against every stock with a 50-day dollar volume of 5e5 or more. The 3 approved "
        "round-2 textbook rules (provisional) have their own columns; `learned` counts how many of the 17 round-1 "
        "learned rules (all rejected on robustness) flagged the name in the window. The 3 round-1 textbook "
        "entries have empty rules and cannot be evaluated.",
        "- **mtf:S1..S4, mtf:B0**: the five pre-registered multi-timeframe signatures of `setup_engine/mtf.py`:",
    ]
    for n in SIGNATURE_NAMES:
        lines.append(f"  - `{n}` {mtf.SIGNATURE_LABELS[n]}: " + "; ".join(f"{k} = {v}" for k, v in mtf.SIGNATURES[n].items()))
    lines += [
        "",
        "## How to read a cell",
        "",
        f"`+Nw (+P%) / K`: the first week flagged in [low - {RECALL_BEFORE_WEEKS} weeks, low + {RECALL_AFTER_WEEKS} weeks] "
        "was N weeks after the low week (negative = before); the weekly close that week was P% above the low's "
        "close; K is how many OTHER liquid stocks the same scanner flagged that same week. `-` means no flag "
        "in the window. K is a precision proxy: a scanner that flags hundreds of stocks a week is not singling "
        "the name out. Liquid means `mtf.is_liquid`: weekly close of 2 or more, median 13-week dollar volume "
        "of 5e6 or more, at least 60 weekly bars. `*` after a name marks a run that spans a gap of more than "
        f"{DATA_GAP_DAYS} days between weekly bars (halted or stitched data: do not interpret).",
        "",
        f"Universe: {sizes['n_stocks']} stocks scanned, {sizes['n_weekly_rows']} weekly rows kept, "
        f"{sizes['n_liquid_rows']} liquid stock-weeks over {sizes['n_weeks']} weeks "
        f"(median {sizes['median_liquid_per_week']:.0f} liquid stocks a week). Cohort names that are not "
        "in the stock cache, or that sit outside a limited run's batches, are ranked against the universe but "
        "are not part of it.",
        "",
        "## Results",
        "",
        render_markdown_tables(blocks),
        "## Recall next to the flag rate",
        "",
        "| scanner | flagged in window | ever flagged | flag rate (liquid stock-weeks) | median flagged / week |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in summary:
        rate = "n/a" if r["flag_rate"] is None else f"{r['flag_rate']:.2%}"
        med = "n/a" if r["median_flagged_per_week"] is None else f"{r['median_flagged_per_week']:.0f}"
        lines.append(f"| {r['scanner']} | {r['flagged_in_window']}/{r['names_with_run']} | "
                     f"{r['ever_flagged']}/{r['names_with_run']} | {rate} | {med} |")
    lines += [
        "",
        "## Caveats",
        "",
        "- Ex-post selection: the names and their lows are chosen with hindsight. Nothing here validates a scanner.",
        "- Survivorship: today's listings only; delisted losers that every scanner may have flagged are absent, "
        "so the flag rates and K are lower bounds on how noisy a scanner was in real time.",
        "- Yahoo adjusts old prices retroactively for splits and the data has no dividend adjustment.",
        "- The registered rules were learned on weekly observations of 2000-2024 data; this table reads them on "
        "2021-2026 Fridays against the full cache's rank universe.",
        "- Weeks are complete weeks only: flags on the final, part-formed week are not attributed to any row.",
    ]
    if result["missing"]:
        lines.append(f"- No data for: {', '.join(result['missing'])}.")
    if result["errors"]["n"]:
        lines.append(f"- {result['errors']['n']} stock(s) failed to process and were skipped (see the JSON).")
    return "\n".join(lines) + "\n"


# ── output ────────────────────────────────────────────────────────────────────────────────────────

def _jsonable(o):
    """Recursively convert to strict-JSON types: non-finite floats -> None, numpy -> Python, dates -> ISO.

    Copy of scripts/study_turnaround._jsonable."""
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


def output_paths(out_dir, limited: bool) -> tuple[Path, Path]:
    """(json path, markdown path). A limited run never defaults to docs/: it goes to SMOKE_DIR."""
    if out_dir:
        d = Path(out_dir).expanduser()
        return d / OUT_JSON.name, d / OUT_MD.name
    if limited:
        return SMOKE_DIR / OUT_JSON.name, SMOKE_DIR / OUT_MD.name
    return OUT_JSON, OUT_MD


# ── main ──────────────────────────────────────────────────────────────────────────────────────────

def _collect(results, in_universe: bool, parts: list, errors: list, infos: dict) -> int:
    """Tag each per-stock result and file it. Returns the number of stocks that processed without error."""
    n = 0
    for batch in results:
        for r in batch if isinstance(batch, list) else [batch]:
            if r is None:
                continue
            if r.get("error"):
                errors.append({"symbol": r["symbol"], "error": r["error"]})
                continue
            n += 1
            for part in ("weekly", "registry"):
                if r[part] is not None:
                    r[part] = r[part].assign(in_universe=in_universe)
            if r["info"] is not None:
                infos[r["symbol"]] = r["info"]
            parts.append(r)
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--history", default=str(DEFAULT_HISTORY))
    ap.add_argument("--bench-dir", default=str(BENCH_DIR))
    ap.add_argument("--limit-batches", type=int, default=None,
                    help="smoke test: scan only the first N batch files (output goes to the smoke directory)")
    ap.add_argument("--workers", type=int, default=min(PROJECTION_WORKERS, os.cpu_count() or 1))
    ap.add_argument("--out-dir", default=None,
                    help=f"directory for scanner_recall.json / .md (default {OUT_JSON.parent}, or {SMOKE_DIR} with --limit-batches)")
    ap.add_argument("--symbols", nargs="+", default=None, help="report only these cohort names (default: all 46)")
    args = ap.parse_args(argv)

    names = tuple(dict.fromkeys(args.symbols)) if args.symbols else COHORT_RECALL
    unknown = [n for n in names if n not in COHORT_RECALL]
    if unknown:
        raise SystemExit(f"not cohort names: {unknown}")
    limited = args.limit_batches is not None
    json_path, md_path = output_paths(args.out_dir, limited)
    history = Path(args.history).expanduser()
    bench_dir = Path(args.bench_dir).expanduser()
    t_start = time.time()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        as_of = mtf.data_end(history)
        mtf.ensure_benchmarks(tuple(dict.fromkeys((RS_BENCH,) + tuple(n for n in names if n in NOT_IN_CACHE))),
                              dest=bench_dir)
        spy = mtf.load_benchmark(RS_BENCH, bench_dir, end=as_of)
        setups = registry_setups()
        reg_features = registry_features(setups)
        fridays = pd.date_range(pd.Timestamp(RECALL_START), as_of, freq="W-FRI")
        cohort = frozenset(names)
        fn_args = (spy, as_of, fridays, cohort, reg_features)
        from scanner_engine.finder import PRESETS  # noqa: F401 - import (sqlalchemy, database) once, before forking

        parts, errors, infos = [], [], {}
        print(f"data through {as_of.date()}; scanning {'the first ' + str(args.limit_batches) + ' batch(es)' if limited else 'all batches'} "
              f"with {args.workers} worker(s)", flush=True)
        t0 = time.time()
        n_stocks = _collect(mtf.map_universe(history, recall_rows, fn_args, args.workers, args.limit_batches),
                            True, parts, errors, infos)
        t_universe = time.time() - t0

        # cohort names the universe pass did not see: other batches of a limited run, and the benchmark directory
        todo = [n for n in names if n not in infos]
        extras: dict[str, pd.DataFrame] = {}
        in_cache_extra = [n for n in todo if n not in NOT_IN_CACHE]
        if in_cache_extra and limited:
            extras.update(mtf.load_daily(history, in_cache_extra))
        from_bench = {n for n in todo if n in NOT_IN_CACHE and (bench_dir / f"{n}.parquet").exists()}
        for n in sorted(from_bench, key=todo.index):
            extras[n] = mtf.load_benchmark(n, bench_dir, end=as_of)
        _collect(([recall_rows(n, d, spy, as_of, fridays, cohort, reg_features)] for n, d in extras.items()),
                 False, parts, errors, infos)
        extra_names = set(extras) & set(infos)

        t1 = time.time()
        setup_keys = scanner_keys(setups)
        weekly = assemble(parts, setups)
        counts = weekly_counts(weekly, setup_keys)
        rates = flag_rates(counts, setup_keys)
        reports = [cohort_report(weekly, counts, n, setup_keys, infos.get(n),
                                 in_cache=(n in infos and n not in from_bench)) for n in names]
        t_assemble = time.time() - t1
    missing = [n for n in names if n not in infos]
    summary = scanner_summary(reports, rates, setup_keys)
    blocks = table_blocks(reports, setups)

    liquid_univ = weekly[weekly["liquid"].to_numpy(dtype=bool) & weekly["in_universe"].to_numpy(dtype=bool)]
    sizes = {"n_stocks": n_stocks, "n_weekly_rows": int(len(weekly)), "n_liquid_rows": int(len(liquid_univ)),
             "n_weeks": int(counts.shape[0]), "median_liquid_per_week": float(counts["n_liquid"].median()),
             "n_registry_obs": int(sum(len(p["registry"]) for p in parts if p["registry"] is not None)),
             "n_cohort_reported": len(reports) - len(missing), "n_extras": len(extra_names)}
    per_stock = t_universe * min(args.workers, max(1, len(mtf._batch_files(history, args.limit_batches)))) / max(n_stocks, 1)
    timing = {"universe_pass_seconds": t_universe, "assemble_seconds": t_assemble, "workers": args.workers,
              "worker_seconds_per_stock": per_stock,
              "projected_full_universe_seconds": per_stock * FULL_UNIVERSE_STOCKS / PROJECTION_WORKERS}
    result = {"data_end": as_of, "limited_run": limited, "limit_batches": args.limit_batches,
              "recall_start": RECALL_START, "run_start": RUN_START,
              "window_weeks": {"before": RECALL_BEFORE_WEEKS, "after": RECALL_AFTER_WEEKS},
              "finder_description": FINDER_DESCRIPTION, "cohort": list(names), "missing": missing,
              "registry": {s["id"]: {"status": s.get("status"), "objective": s.get("objective"),
                                     "rule_text": s.get("rule_text")} for s in setups},
              "signatures": {n: {"label": mtf.SIGNATURE_LABELS[n], "rules": mtf.SIGNATURES[n]} for n in SIGNATURE_NAMES},
              "scanner_keys": setup_keys, "sizes": sizes, "timing": timing,
              "errors": {"n": len(errors), "first": errors[:25]},
              "scanner_summary": summary, "flag_rates": rates, "cohort_results": reports,
              "tables": blocks}

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(_jsonable(result), allow_nan=False, indent=1))
    md_path.write_text(markdown_report(_jsonable(result), blocks, summary))

    print(f"\nscanned {n_stocks} stocks with enough history ({len(errors)} failed), {sizes['n_weekly_rows']} weekly rows kept, "
          f"{sizes['n_liquid_rows']} liquid stock-weeks over {sizes['n_weeks']} weeks "
          f"(median {sizes['median_liquid_per_week']:.0f} liquid per week), {sizes['n_registry_obs']} registry observations")
    print(f"cohort reported: {sizes['n_cohort_reported']}/{len(names)}"
          + (f"; no data for {', '.join(missing)}" if missing else "")
          + "   (* = run spans a data gap of more than "
          f"{DATA_GAP_DAYS} days; cell = weeks from the low (price vs the low's close) / other liquid stocks flagged that week)\n")
    print(render_console(blocks))
    print("Recall next to the flag rate\n" + render_summary(summary) + "\n")
    if errors:
        print("failed stocks (first 10): " + "; ".join(f"{e['symbol']}: {e['error']}" for e in errors[:10]))
    print(f"timing: universe pass {t_universe:.0f}s ({args.workers} workers, {per_stock * 1000:.0f} worker-ms per stock), "
          f"assembly {t_assemble:.0f}s, total {time.time() - t_start:.0f}s; projected full pass "
          f"{timing['projected_full_universe_seconds'] / 60:.1f} min with {PROJECTION_WORKERS} workers "
          f"({FULL_UNIVERSE_STOCKS} stocks, excluding assembly)")
    print(f"wrote {json_path} and {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
