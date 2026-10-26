"""
After a MARKET confirmation event, do stocks that match a multi-timeframe signature, bought at the next
open and ridden, beat ALL liquid stocks bought the same day with the same exit? One pre-registered
out-of-sample event study over the whole price cache.

    python scripts/mtf_event_study.py                               # full study, writes docs/mtf_event_study_results.json
    python scripts/mtf_event_study.py --limit-batches 3 --no-ledger # smoke test, writes nothing real

PRE-REGISTERED. Every constant, the event rule, the five signatures (setup_engine.mtf.SIGNATURES), the
three exits, the control, the clustering and the verdict rule below are fixed before any result is
seen. Nothing is tuned afterwards. Only S1-S4 on the primary exit (R3), SPY events, held-out mature
events decide the verdict; B0, the other exits, the fixed horizons, the SMH events, 2022 and the
placebo weeks are descriptive and never change it. One record of the plan and one per (signature, exit)
are appended to the trial ledger (setup_engine/registry/trials.jsonl) on a full run, so the count of
attempts stays honest.

THE QUESTION. A MARKET EVENT is a week whose SPY close is back above its 30-week SMA after at least 8
consecutive weeks below it, with the low of that streak at least 15% under the highest weekly close of
the 52 weeks before the event (a bear-market recovery, not a 3-week dip; ``confirmation_events``). On the
event week's close every liquid stock that matches a signature is bought at the NEXT daily open and
ridden under an exit rule (setup_engine/ride.py). Do stocks that look like S1-S4 on that day earn more
than every liquid stock bought that day under the same exit?

EXPLORATION DISCLOSURE (verbatim). "The signatures S1-S4 were chosen after looking at the 2022-10-14 ..
2023-02-03 cross-sections of the AI/chip winners (planner fact list). 2022 events are therefore IN-SAMPLE
and excluded from the verdict; the 2025 lows were seen in the case-study exploration (not used to pick
signatures) and the 2025 event is immature, so it is reported but outside the verdict."

CONTROLS (the point of the design). A strategy that "works" in a rising market proves nothing, so each
trade is compared with the DATE-MATCHED control: the mean outcome of ALL liquid stocks with a finite
outcome on the SAME event week, under the SAME exit and the same cost. Trade excess x_i = r_i minus that
week's control mean (the control includes the signature's own stocks: it is "the market that day"). B0
("price above the 30-week SMA", market beta) is a secondary contrast: x_i = r_i minus the mean of the B0
stocks that week. PLACEBO WEEKS are random mature weeks far from any SPY or SMH event: a signature whose
excess is as large there is a generic factor (momentum), not a bottom-timing rule; placebo weeks are not
bottoms, so only EXCESS (never raw returns) is compared.

NO LOOK-AHEAD. Weekly and monthly bars are used only once COMPLETE (setup_engine/mtf.py). The event is
decided on COMPLETE weekly bars of the benchmark clipped to the cache's last date (yfinance runs to
today, the cache ends earlier; every series is clipped to ``data_end`` first). Signatures read the
event week's own completed bar and same-week cross-sectional ranks among liquid stocks. The order model
is setup_engine/ride.py: signal at the weekly close, entry at the next daily open, exits filled at the
next open after the deciding close, a hard stop underneath, flat round-trip cost. The ex-post labels
(fmax_156 and the ran-5x shares) never feed a feature.

CLUSTERING. Stocks bought on the same event week share one market, so the primary interval resamples
whole EVENTS (setup_engine/trade_stats.cluster_bootstrap, mean as a ratio of sums). The interval for the
verdict is two-sided at the Bonferroni level ALPHA_VERDICT = 0.05 / 4 (98.75%). Intervals clustered by
stock and by EPISODE (events less than 26 weeks apart: 2002-11/2003-03, 2022-08/2022-11) are reported
next to it. With only about ten independent market moments all of these are crude and too narrow.

VERDICT RULE (see VERDICT_RULE; implemented in ``verdict``). SURVIVORSHIP: today's listings only. The
adjustment is an ASSUMPTION, not a measurement (``survivorship``).

KNOWN BIASES (not fixable here, reported with every result). Survivorship (today's listings; every mean
and hit rate is an upper bound and the control is flattered too; the delisted list is thin before 2013),
Yahoo's retroactive split adjustment (distorts old price/dollar-volume bands), no dividend adjustment.
Stops are assumed fillable inside the bar, which flatters thin stocks. R1/R2 exits interact with the
signatures (signatures need price above the 30-week/10-month SMA, controls often do not), so R1/R2
excess is exposure-confounded; PRIMARY_EXIT is R3, whose rule does not depend on the entry state.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from setup_engine import learn as L
from setup_engine import mtf
from setup_engine import ride as R
from setup_engine import trade_stats as TS
from setup_engine.screen import Screen

# ── pre-registered choices (fixed before any result is seen) ──────────────────────────────────────

DEFAULT_HISTORY = mtf.DEFAULT_HISTORY
BENCH_DIR = mtf.BENCH_DIR
DELISTED_CSV = Path.home() / ".roboalgo" / "delisted" / "av_delisted.csv"
OUT_JSON = ROOT / "docs" / "mtf_event_study_results.json"
SMOKE_OUT = Path.home() / ".roboalgo" / "mtf" / "event_smoke.json"
TRADES_OUT = Path.home() / ".roboalgo" / "mtf" / "event_rows.parquet"

EVENT_BENCH = "SPY"
SECONDARY_BENCH = "SMH"
RS_BENCH = "SPY"
EV_SMA_W = 30                      # weekly SMA window of the benchmark
EV_MIN_BELOW = 8                   # consecutive weekly closes below the SMA before the cross
EV_MIN_DD = 0.15                   # the streak's lowest close vs the highest close of the prior 52 weeks
EV_DD_LOOKBACK = 52
EPISODE_GAP_WEEKS = 26             # events less than this many weeks apart share an episode

IN_SAMPLE_YEARS = (2022,)
MATURE_WEEKS = 156                 # an event is mature when this many weekly bars exist after it

SIGNATURE_NAMES = ("S1", "S2", "S3", "S4", "B0")
VERDICT_SIGNATURES = ("S1", "S2", "S3", "S4")
EXITS = {"R1": R.RIDE_R1, "R2": R.RIDE_R2, "R3": R.RIDE_R3}
PRIMARY_EXIT = "R3"
HORIZONS = (26, 52, 156)           # fixed-horizon returns, in weeks after the signal week
RUN_MULT = 5.0                     # "ran 5x": best weekly close within 156 weeks >= RUN_MULT x the entry open
OUTCOMES = tuple(f"ret_{x}" for x in EXITS) + tuple(f"hret_{h}" for h in HORIZONS)
PRIMARY_OUTCOME = f"ret_{PRIMARY_EXIT}"
SPLITS = ("held_out_mature", "in_sample_2022", "immature", "all")
EVENT_SETS = ("spy", "smh", "placebo")
# every column any signature reads from the weekly table (the rank columns are added by the study)
SIGNATURE_COLUMNS = tuple(dict.fromkeys(
    ["w_ret13", "w_ret26", "w_close"]
    + [c for s in SIGNATURE_NAMES for c in Screen(mtf.SIGNATURES[s]).fields() if c not in mtf.XS_COLUMNS]))

N_PLACEBO = 40
PLACEBO_SEED = 20261009
PLACEBO_EXCLUSION_WEEKS = 26
PLACEBO_START = "2001-06-01"

N_BOOT = 2000
SEED = 20261009
ALPHA = 0.05
ALPHA_VERDICT = 0.05 / 4           # Bonferroni over the four verdict signatures
MIN_EVENTS_WITH_TRADES = 5

F_REF_FROM_YEAR = 2013             # the delisted list is thin before this year
L_GENTLE = -0.5                    # return of a trade in a stock that was delisted: gentle assumption
L_HARSH = -1.0                     # ... harsh assumption (total loss)
HARSH_SIG_MULT = 1.5               # harsh: the signature's missing share is this multiple of the universe's
TABLE_FIRST_YEAR = 2001

LEDGER_STAGE = "mtf_event_study"

EXPLORATION_DISCLOSURE = (
    "The signatures S1-S4 were chosen after looking at the 2022-10-14 .. 2023-02-03 cross-sections of the "
    "AI/chip winners (planner fact list). 2022 events are therefore IN-SAMPLE and excluded from the verdict; "
    "the 2025 lows were seen in the case-study exploration (not used to pick signatures) and the 2025 event "
    "is immature, so it is reported but outside the verdict.")

EVENT_RULE = (
    f"Week k (COMPLETE weekly bars of the benchmark, clipped to the data end) is an EVENT iff close[k] > "
    f"SMA{EV_SMA_W}[k] and close[k-1] < SMA{EV_SMA_W}[k-1]; the streak of consecutive weeks with close < SMA "
    f"ending at k-1 has length >= {EV_MIN_BELOW} (a week with NaN SMA ends the streak); and min(close over "
    f"the streak) / max(close over weeks k-{EV_DD_LOOKBACK}..k-1) - 1 <= -{EV_MIN_DD}. Re-arm is implicit: "
    f"a new event needs a fresh streak of >= {EV_MIN_BELOW} weeks below. Episodes: events less than "
    f"{EPISODE_GAP_WEEKS} weeks apart share an id. Mature: k + {MATURE_WEEKS} <= the last benchmark week.")

VERDICT_RULE = (
    f"Applied to each of {', '.join(VERDICT_SIGNATURES)} on exit {PRIMARY_EXIT}, {EVENT_BENCH} events, the "
    f"held-out mature events only (2022 and immature events excluded). 'edge' only if ALL of: (1) the "
    f"event-clustered mean excess vs the date-matched control is > 0 AND the lower bound of its "
    f"{100 * (1 - ALPHA_VERDICT):.2f}% interval (Bonferroni over four signatures) is > 0; (2) the signature "
    f"had trades in >= {MIN_EVENTS_WITH_TRADES} held-out events AND its per-event excess is > 0 in a strict "
    f"majority of those events; (3) the survivorship-adjusted excess is > 0 under BOTH the gentle "
    f"(L={L_GENTLE}) and the harsh (L={L_HARSH}, signature missing share x{HARSH_SIG_MULT}) assumptions. "
    f"Otherwise 'no demonstrated edge'. A missing input gives 'incomplete (missing: ...)'. B0, the other "
    f"exits, the horizons, SMH events, 2022 and the placebo never change the verdict.")

SPLIT_IN_SAMPLE = "in_sample_2022"
SPLIT_IMMATURE = "immature"
SPLIT_HELD_OUT = "held_out_mature"


# ── events ────────────────────────────────────────────────────────────────────────────────────────

def confirmation_events(bench_daily: pd.DataFrame, as_of=None) -> pd.DataFrame:
    """The market confirmation events of a benchmark (rule: EVENT_RULE) on COMPLETE weekly bars up to ``as_of``.

    Columns: week, last_day, below_weeks, drawdown, year, in_sample, mature, episode (one row per event).
    The 52-week look-back uses the weeks that exist when fewer than 52 precede the event.
    """
    wb = mtf.weekly_bars(bench_daily, as_of)
    cols = ["week", "last_day", "below_weeks", "drawdown", "year", "in_sample", "mature", "episode"]
    n = len(wb)
    if n == 0:
        return pd.DataFrame(columns=cols)
    c = wb["close"].to_numpy(dtype="float64")
    sma = pd.Series(c).rolling(EV_SMA_W, min_periods=EV_SMA_W).mean().to_numpy()
    rows = []
    for k in range(1, n):
        if not (np.isfinite(sma[k]) and np.isfinite(sma[k - 1])):
            continue
        if not (c[k] > sma[k] and c[k - 1] < sma[k - 1]):
            continue
        j = k - 1
        while j >= 0 and np.isfinite(sma[j]) and c[j] < sma[j]:
            j -= 1
        streak = k - 1 - j
        if streak < EV_MIN_BELOW:
            continue
        low = float(c[j + 1:k].min())
        prior_hi = float(c[max(0, k - EV_DD_LOOKBACK):k].max())
        dd = low / prior_hi - 1.0
        if dd > -EV_MIN_DD:
            continue
        wk = wb.index[k]
        rows.append({"week": wk, "last_day": wb["last_day"].iloc[k], "below_weeks": int(streak),
                     "drawdown": float(dd), "year": int(wk.year), "in_sample": int(wk.year) in IN_SAMPLE_YEARS,
                     "mature": bool(k + MATURE_WEEKS <= n - 1)})
    out = pd.DataFrame(rows, columns=[c_ for c_ in cols if c_ != "episode"])
    ep, prev, cur = [], None, -1
    for wk in out["week"]:
        if prev is None or (wk - prev).days / 7 >= EPISODE_GAP_WEEKS:
            cur += 1
        ep.append(cur)
        prev = wk
    out["episode"] = ep
    return out[cols].reset_index(drop=True)


def event_split(in_sample: bool, mature: bool) -> str:
    """Which part of the evidence an event belongs to: in-sample years first, then immature, else the verdict set."""
    if in_sample:
        return SPLIT_IN_SAMPLE
    if not mature:
        return SPLIT_IMMATURE
    return SPLIT_HELD_OUT


def placebo_weeks(bench_weeks, events_all, as_of, n: int = N_PLACEBO) -> list:
    """``n`` random mature weeks (>= PLACEBO_START) not within PLACEBO_EXCLUSION_WEEKS of any event, sorted.

    ``bench_weeks`` are the benchmark's complete weekly labels; ``events_all`` the SPY and SMH event weeks.
    Reproducible: the draw uses PLACEBO_SEED.
    """
    weeks = pd.DatetimeIndex(bench_weeks)
    weeks = weeks[weeks <= pd.Timestamp(as_of)]
    last = len(weeks) - 1
    cand = (weeks >= pd.Timestamp(PLACEBO_START)) & (np.arange(len(weeks)) + MATURE_WEEKS <= last)
    ev = pd.DatetimeIndex(events_all)
    if len(ev) and len(weeks):
        gap_days = np.abs((weeks.values[:, None] - ev.values[None, :]) / np.timedelta64(1, "D")).min(axis=1)
        cand &= ~(gap_days <= PLACEBO_EXCLUSION_WEEKS * 7)
    idx = np.flatnonzero(cand)
    rng = np.random.default_rng(PLACEBO_SEED)
    chosen = rng.choice(idx, size=min(int(n), len(idx)), replace=False) if len(idx) else np.array([], dtype=int)
    return [pd.Timestamp(weeks[i]) for i in sorted(chosen)]


# ── one stock ─────────────────────────────────────────────────────────────────────────────────────

def stock_rows(symbol, daily: pd.DataFrame, spy: pd.DataFrame, weeks, as_of):
    """The map_universe function: one stock's liquid rows on the study weeks, with features and outcomes.

    Returns {"rows": DataFrame | None, "span": (symbol, first_date, last_date)} (the span feeds the
    survivorship table even when no row qualifies). Features at week W use daily bars <= W only; the
    outcome columns (ret_*, reason_*, weeks_*, mfe_*, hret_*, fmax_156) are ex-post.
    """
    as_of = pd.Timestamp(as_of)
    daily = daily[daily.index <= as_of]
    if len(daily) == 0:
        return None
    span = (symbol, daily.index[0], daily.index[-1])
    wf = mtf.mtf_features(daily, spy, as_of)
    if len(wf) == 0:
        return {"rows": None, "span": span}
    pos = wf.index.get_indexer(pd.DatetimeIndex(weeks))
    liquid = mtf.is_liquid(wf)
    keep = np.flatnonzero((pos >= 0) & liquid[np.maximum(pos, 0)])
    if len(keep) == 0:
        return {"rows": None, "span": span}
    wb = mtf.weekly_bars(daily, as_of)
    mb = mtf.monthly_bars(daily, as_of)
    rd = R.prepare(daily, wb, mb)
    feats = {c: wf[c].to_numpy(dtype="float64") for c in SIGNATURE_COLUMNS}
    recs = []
    for i in keep:
        p = int(pos[i])
        row = {"symbol": symbol, "week": wf.index[p]}
        for c in SIGNATURE_COLUMNS:
            row[c] = float(feats[c][p])
        for name, ex in EXITS.items():
            r = R.ride(rd, p, ex)
            row[f"ret_{name}"] = math.nan if r is None else float(r["ret_net"])
            row[f"reason_{name}"] = None if r is None else r["reason"]
            row[f"weeks_{name}"] = math.nan if r is None else float(r["weeks_held"])
            row[f"mfe_{name}"] = math.nan if r is None else float(r["mfe"])
        for h in HORIZONS:
            row[f"hret_{h}"] = R.horizon_return(rd, p, h, EXITS[PRIMARY_EXIT].cost_rt)
        row[f"fmax_{HORIZONS[-1]}"] = R.forward_max_close(rd, p, HORIZONS[-1])
        recs.append(row)
    return {"rows": pd.DataFrame(recs), "span": span}


# ── survivorship ──────────────────────────────────────────────────────────────────────────────────

def survivorship_table(delisted: pd.DataFrame, spans: pd.DataFrame, years) -> dict:
    """The delisted share by year and the shares the adjustment uses.

    f_ref = mean f_stock over the table's years >= F_REF_FROM_YEAR; f_used(y) = max(f_stock(y), f_ref)
    when the delisted list is incomplete for y (before 2013), else f_stock(y).
    """
    spans = spans.assign(first_date=pd.to_datetime(spans["first_date"]), last_date=pd.to_datetime(spans["last_date"]))
    table = TS.delisted_share_by_year(delisted, spans, list(years))
    f_stock = {int(y): float(v) for y, v in table["f_stock"].items()}
    ref = [v for y, v in f_stock.items() if y >= F_REF_FROM_YEAR and np.isfinite(v)]
    f_ref = float(np.mean(ref)) if ref else None
    f_used = {}
    for y, v in f_stock.items():
        if bool(table.loc[y, "list_incomplete"]) and f_ref is not None:
            f_used[y] = max(v, f_ref) if np.isfinite(v) else f_ref
        else:
            f_used[y] = v
    return {"table": table, "f_ref": f_ref, "f_used": f_used}


def survivorship(years, r, ctl, f_used: dict) -> dict:
    """Survivorship-adjusted excess of one signature's verdict-set trades (an ASSUMPTION, not a measurement).

    ``years`` is the event year of each trade, ``r`` its return, ``ctl`` its week's control mean.
    f = trade-weighted mean of f_used over the trades (years without f are skipped). Gentle: a share f of
    both the signature and the control ends at L_GENTLE. Harsh: the signature's share is
    min(1, HARSH_SIG_MULT * f), the control's f, both ending at L_HARSH. ``breakeven_f_sig`` is the
    signature missing share at which the harsh adjusted excess is zero (raw solution: a value outside
    [0, 1] means it never / always holds); ``breakeven_mult`` is that share over f.
    """
    years = np.asarray(years, dtype=int)
    r = np.asarray(r, dtype=float)
    ctl = np.asarray(ctl, dtype=float)
    out = {"n_trades": int(len(r)), "f": None, "f_sig_harsh": None, "mean_sig": None, "mean_ctl": None,
           "adj_sig_gentle": None, "adj_ctl_gentle": None, "adj_excess_gentle": None,
           "adj_sig_harsh": None, "adj_ctl_harsh": None, "adj_excess_harsh": None,
           "breakeven_f_sig": None, "breakeven_mult": None}
    if len(r) == 0:
        return out
    fs = np.array([f_used.get(int(y), np.nan) for y in years], dtype=float)
    ok = np.isfinite(fs)
    if not ok.any():
        return out
    f = float(fs[ok].mean())
    m_sig, m_ctl = float(r.mean()), float(ctl.mean())
    f_sig = min(1.0, HARSH_SIG_MULT * f)
    adj_sig_g, adj_ctl_g = (1 - f) * m_sig + f * L_GENTLE, (1 - f) * m_ctl + f * L_GENTLE
    adj_sig_h, adj_ctl_h = (1 - f_sig) * m_sig + f_sig * L_HARSH, (1 - f) * m_ctl + f * L_HARSH
    denom = m_sig - L_HARSH
    be = (m_sig - adj_ctl_h) / denom if denom > 0 else None
    out.update(f=f, f_sig_harsh=f_sig, mean_sig=m_sig, mean_ctl=m_ctl,
               adj_sig_gentle=adj_sig_g, adj_ctl_gentle=adj_ctl_g, adj_excess_gentle=adj_sig_g - adj_ctl_g,
               adj_sig_harsh=adj_sig_h, adj_ctl_harsh=adj_ctl_h, adj_excess_harsh=adj_sig_h - adj_ctl_h,
               breakeven_f_sig=be, breakeven_mult=(be / f if (be is not None and f > 0) else None))
    return out


# ── verdict ───────────────────────────────────────────────────────────────────────────────────────

def verdict(block: dict) -> dict:
    """Apply VERDICT_RULE to one signature's verdict input.

    ``block`` needs: ``excess_event_verdict`` (cluster_bootstrap dict at ALPHA_VERDICT over event clusters:
    mean, lo), ``per_event`` (the held-out events' rows: n_sig, excess), ``adj_excess_gentle`` and
    ``adj_excess_harsh``. ``excess_event`` (the 95% interval) is deliberately NOT used.
    """
    ci = block.get("excess_event_verdict") or {}
    mean, lo = ci.get("mean"), ci.get("lo")
    per_event = block.get("per_event")
    adj_g, adj_h = block.get("adj_excess_gentle"), block.get("adj_excess_harsh")
    with_trades = None
    if per_event is not None:
        with_trades = [e for e in per_event if (e.get("n_sig") or 0) > 0 and e.get("excess") is not None]
    inputs = {"event-clustered excess mean": mean, f"event-clustered excess CI lo ({100 * (1 - ALPHA_VERDICT):.2f}%)": lo,
              "per-event table": with_trades, "adj_excess_gentle": adj_g, "adj_excess_harsh": adj_h}
    missing = [k for k, v in inputs.items() if v is None]
    if missing:
        return {"verdict": f"incomplete (missing: {', '.join(missing)})", "inputs": _shallow(inputs), "checks": None}
    k_pos = sum(1 for e in with_trades if e["excess"] > 0)
    checks = {"1_event_clustered_excess_positive_and_ci_lo_positive": bool(mean > 0 and lo > 0),
              "2_enough_events_and_strict_majority_positive": bool(len(with_trades) >= MIN_EVENTS_WITH_TRADES
                                                                  and 2 * k_pos > len(with_trades)),
              "3_adjusted_excess_positive_gentle_and_harsh": bool(adj_g > 0 and adj_h > 0)}
    ins = _shallow(inputs)
    ins.update({"n_events_with_trades": len(with_trades), "n_events_positive": int(k_pos)})
    return {"verdict": "edge" if all(checks.values()) else "no demonstrated edge", "inputs": ins, "checks": checks}


def _shallow(inputs: dict) -> dict:
    return {k: (len(v) if isinstance(v, list) else v) for k, v in inputs.items()}


# ── analysis ──────────────────────────────────────────────────────────────────────────────────────

def _week_mean(wk, v, include=None):
    """Per row: the mean of the finite ``v`` (restricted to ``include``) over the rows of the same week; NaN if none."""
    ok = np.isfinite(v) if include is None else (np.isfinite(v) & include)
    s = pd.Series(np.where(ok, v, 0.0)).groupby(wk).transform("sum").to_numpy()
    c = pd.Series(ok.astype("float64")).groupby(wk).transform("sum").to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c > 0, s / c, np.nan)


def _event_meta(events: pd.DataFrame) -> pd.DataFrame:
    meta = events[["week", "in_sample", "mature", "episode"]].copy().reset_index(drop=True)
    meta["split"] = [event_split(bool(a), bool(b)) for a, b in zip(meta["in_sample"], meta["mature"])]
    return meta


def _placebo_meta(placebo) -> pd.DataFrame:
    weeks = [pd.Timestamp(w) for w in placebo]
    return pd.DataFrame({"week": pd.DatetimeIndex(weeks), "in_sample": False, "mature": True,
                         "episode": np.arange(len(weeks)), "split": SPLIT_HELD_OUT})


def _mean(x):
    return float(np.mean(x)) if len(x) else None


def _share(x, thresh):
    return float(np.mean(x >= thresh)) if len(x) else None


def _ran5(fmax):
    f = fmax[np.isfinite(fmax)]
    return (float(np.mean(f >= RUN_MULT - 1.0)), int(len(f))) if len(f) else (None, 0)


def analyze(rows: pd.DataFrame, events: dict, placebo, spans, delisted, n_boot: int = N_BOOT,
            seed: int = SEED) -> dict:
    """Everything the report needs: blocks per signature x event set x outcome x split, per-event tables,
    the placebo comparison, survivorship, and the verdicts.

    ``rows``: the stock_rows output concatenated (liquid rows only). ``events``: {"spy": events frame,
    "smh": events frame}. ``placebo``: list of weeks. ``spans``: frame symbol, first_date, last_date.
    ``delisted``: the listing-status frame or None (then the verdicts are 'incomplete').
    """
    panel = mtf.add_xs_ranks(rows.assign(liquid=True))
    n = len(panel)
    wk = panel["week"].to_numpy(dtype="datetime64[ns]").astype("int64")
    sym = panel["symbol"].to_numpy()
    fmax = panel[f"fmax_{HORIZONS[-1]}"].to_numpy(dtype="float64")
    vals = {o: panel[o].to_numpy(dtype="float64") for o in OUTCOMES}
    ctl = {o: _week_mean(wk, vals[o]) for o in OUTCOMES}
    masks = {s: Screen(mtf.SIGNATURES[s]).mask(panel) for s in SIGNATURE_NAMES}
    b0 = {o: _week_mean(wk, vals[o], masks["B0"]) for o in OUTCOMES}

    metas = {"spy": _event_meta(events["spy"]), "smh": _event_meta(events["smh"]), "placebo": _placebo_meta(placebo)}
    ev_idx, ev_split, ev_epi = {}, {}, {}
    for name, meta in metas.items():
        lookup = {int(t): j for j, t in enumerate(meta["week"].to_numpy(dtype="datetime64[ns]").astype("int64"))}
        idx = np.array([lookup.get(int(t), -1) for t in wk], dtype=np.int64)
        ev_idx[name] = idx
        if len(meta) and n:
            j = np.maximum(idx, 0)
            ev_split[name] = np.where(idx >= 0, meta["split"].to_numpy(dtype=object)[j], "")
            ev_epi[name] = np.where(idx >= 0, meta["episode"].to_numpy(dtype=np.int64)[j], -1)
        else:
            ev_split[name] = np.full(n, "", dtype=object)
            ev_epi[name] = np.full(n, -1, dtype=np.int64)

    def in_split(name, split):
        base = ev_idx[name] >= 0
        return base if split == "all" else base & (ev_split[name] == split)

    # survivorship pieces
    years_all = [int(w.year) for m in metas.values() for w in m["week"]]
    sv_tab = None
    if delisted is not None and len(years_all):
        sv_tab = survivorship_table(delisted, spans, range(TABLE_FIRST_YEAR, max(years_all) + 1))

    def block(sel, o, name, sig):
        r, c = vals[o][sel], ctl[o][sel]
        x = r - c
        ev, stk, epi = wk[sel], sym[sel], ev_epi[name][sel]
        out = {"n_trades": int(sel.sum()), "n_events_with_trades": int(len(np.unique(ev))),
               "n_stocks": int(len(np.unique(stk))), "mean_r": _mean(r), "median_r": float(np.median(r)) if len(r) else None,
               "share_ge_1": _share(r, 1.0), "share_ge_3": _share(r, 3.0), "mean_control": _mean(c),
               "mean_excess": _mean(x),
               "ci_event": TS.cluster_bootstrap(x, ev, n_boot, seed, ALPHA),
               "ci_event_verdict": TS.cluster_bootstrap(x, ev, n_boot, seed, ALPHA_VERDICT),
               "ci_stock": TS.cluster_bootstrap(x, stk, n_boot, seed, ALPHA),
               "ci_episode": TS.cluster_bootstrap(x, epi, n_boot, seed, ALPHA)}
        if sig == "B0":
            out["vs_B0"] = None
        else:
            out["vs_B0"] = TS.cluster_bootstrap(r - b0[o][sel], ev, n_boot, seed, ALPHA)
        return out

    def run_stats(sel_rows, sig_mask):
        sig_share, sig_n = _ran5(fmax[sel_rows & sig_mask])
        base_share, base_n = _ran5(fmax[sel_rows])
        return {"ran5x_sig": sig_share, "ran5x_sig_n": sig_n, "ran5x_base": base_share, "ran5x_base_n": base_n}

    res = {"signatures": {}}
    for s in SIGNATURE_NAMES:
        sm = masks[s]
        entry = {"label": mtf.SIGNATURE_LABELS[s], "rules": dict(mtf.SIGNATURES[s]), "n_mask_rows": int(sm.sum())}
        for name in EVENT_SETS:
            per_out = {}
            for o in OUTCOMES:
                fin = np.isfinite(vals[o])
                per_out[o] = {}
                for sp in SPLITS:
                    rows_in = in_split(name, sp)
                    b = block(rows_in & sm & fin, o, name, s)
                    b.update(run_stats(rows_in, sm))
                    per_out[o][sp] = b
            entry[name] = per_out
        # per-event tables on the primary outcome
        entry["per_event"] = {}
        po = PRIMARY_OUTCOME
        for name, meta in metas.items():
            tab = []
            for j, m in meta.iterrows():
                in_e = ev_idx[name] == j
                fin = np.isfinite(vals[po])
                sig_rows = in_e & sm & fin
                n_liq = int(in_e.sum())
                n_sig = int(sig_rows.sum())
                ctl_mean = _mean(vals[po][in_e & fin])
                sig_mean = _mean(vals[po][sig_rows])
                sr, _ = _ran5(fmax[in_e & sm])
                br, _ = _ran5(fmax[in_e])
                tab.append({"week": m["week"], "in_sample": bool(m["in_sample"]), "mature": bool(m["mature"]),
                            "split": m["split"], "n_liquid": n_liq, "n_sig": n_sig, "sig_mean": sig_mean,
                            "control_mean": ctl_mean,
                            "excess": (sig_mean - ctl_mean) if (sig_mean is not None and ctl_mean is not None) else None,
                            "sig_ran5x": sr, "base_ran5x": br})
            entry["per_event"][name] = tab
        # placebo comparison: SPY held-out event excess vs placebo-week excess
        comp = {}
        for o in OUTCOMES:
            fin = np.isfinite(vals[o])
            se = in_split("spy", SPLIT_HELD_OUT) & sm & fin
            sp_ = in_split("placebo", "all") & sm & fin
            comp[o] = TS.cluster_bootstrap_diff(vals[o][se] - ctl[o][se], wk[se], vals[o][sp_] - ctl[o][sp_], wk[sp_],
                                                n_boot, seed, ALPHA)
        entry["placebo_comparison"] = comp
        # survivorship and the verdict input (SPY, held-out mature, primary outcome)
        sel_v = in_split("spy", SPLIT_HELD_OUT) & sm & np.isfinite(vals[po])
        sv = None
        if sv_tab is not None:
            yrs = pd.DatetimeIndex(wk[sel_v].astype("datetime64[ns]")).year
            sv = survivorship(yrs, vals[po][sel_v], ctl[po][sel_v], sv_tab["f_used"])
            sv["f_ref"] = sv_tab["f_ref"]
        entry["survivorship"] = sv
        vin = {"excess_event_verdict": entry["spy"][po][SPLIT_HELD_OUT]["ci_event_verdict"],
               "per_event": [e for e in entry["per_event"]["spy"] if e["split"] == SPLIT_HELD_OUT],
               "adj_excess_gentle": None if sv is None else sv["adj_excess_gentle"],
               "adj_excess_harsh": None if sv is None else sv["adj_excess_harsh"]}
        entry["verdict_input"] = vin
        if s in VERDICT_SIGNATURES:
            entry["verdict"] = verdict(vin)
        res["signatures"][s] = entry

    res["verdict"] = {s: res["signatures"][s]["verdict"] for s in VERDICT_SIGNATURES}
    res["survivorship_table"] = None
    if sv_tab is not None:
        res["survivorship_table"] = {"by_year": {int(y): row for y, row in sv_tab["table"].to_dict("index").items()},
                                     "f_ref": sv_tab["f_ref"], "f_used": sv_tab["f_used"],
                                     "L_gentle": L_GENTLE, "L_harsh": L_HARSH, "harsh_sig_mult": HARSH_SIG_MULT}
    sizes = {}
    for name, meta in metas.items():
        sizes[name] = {"n_events": int(len(meta)),
                       **{f"n_events_{sp}": int((meta["split"] == sp).sum()) for sp in SPLITS if sp != "all"},
                       "n_stock_events": int((ev_idx[name] >= 0).sum()),
                       "distinct_stocks": int(len(np.unique(sym[ev_idx[name] >= 0])))}
    sizes["rows_total"] = int(n)
    sizes["distinct_stocks_total"] = int(len(np.unique(sym))) if n else 0
    res["sizes"] = sizes
    return res


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


def format_events(name: str, ev: pd.DataFrame) -> str:
    lines = [f"{name} events ({len(ev)}): week  below  drawdown  split  episode"]
    for _, e in ev.iterrows():
        lines.append(f"  {e['week'].date()}  {e['below_weeks']:>3}w  {e['drawdown']:+.1%}  "
                     f"{event_split(bool(e['in_sample']), bool(e['mature'])):<16} ep{e['episode']}")
    return "\n".join(lines)


def format_signature_table(analysis: dict) -> str:
    head = (f"{'sig':<4} {'ev w/trades':>11} {'trades':>7} {'stocks':>6} {'mean R3':>8} "
            f"{'excess [CI ' + format(100 * (1 - ALPHA_VERDICT), '.2f') + '%]':>30} {'ev pos':>7} {'adj harsh':>9} "
            f"{'ran5x sig/base':>16}")
    lines = [head, "-" * len(head)]
    for s in SIGNATURE_NAMES:
        e = analysis["signatures"][s]
        b = e["spy"][PRIMARY_OUTCOME][SPLIT_HELD_OUT]
        pe = [x for x in e["verdict_input"]["per_event"] if (x["n_sig"] or 0) > 0 and x["excess"] is not None]
        pos = sum(1 for x in pe if x["excess"] > 0)
        sv = e["survivorship"]
        rs = "n/a" if b["ran5x_sig"] is None else f"{b['ran5x_sig']:.1%}"
        rb = "n/a" if b["ran5x_base"] is None else f"{b['ran5x_base']:.1%}"
        lines.append(
            f"{s:<4} {b['n_events_with_trades']:>11} {b['n_trades']:>7} {b['n_stocks']:>6} {_n(b['mean_r']):>8} "
            f"{(_n(b['mean_excess']) + ' ' + _ci(b['ci_event_verdict'])):>30} {pos:>3}/{len(pe):<3} "
            f"{_n(None if sv is None else sv['adj_excess_harsh']):>9} {(rs + ' / ' + rb):>16}")
    return "\n".join(lines)


def _warm_jit() -> None:
    """Compile the ride kernel BEFORE forking workers, so each worker does not compile its own copy."""
    idx = pd.bdate_range("2020-01-06", periods=400)
    c = 10 + np.cumsum(np.random.default_rng(0).normal(0, 0.1, len(idx)))
    daily = pd.DataFrame({"open": c, "high": c + 0.2, "low": c - 0.2, "close": c, "volume": 1e6}, index=idx)
    rd = R.prepare(daily, mtf.weekly_bars(daily), mtf.monthly_bars(daily))
    for ex in EXITS.values():
        R.ride(rd, 3, ex)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--history", default=str(DEFAULT_HISTORY))
    ap.add_argument("--bench-dir", default=str(BENCH_DIR))
    ap.add_argument("--limit-batches", type=int, default=None, help="smoke test: only the first N batch files (no ledger)")
    ap.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    ap.add_argument("--out", default=None, help=f"results JSON (default {OUT_JSON}, or {SMOKE_OUT} with --limit-batches)")
    ap.add_argument("--no-ledger", action="store_true", help="do not write to the trial ledger")
    ap.add_argument("--delisted", default=str(DELISTED_CSV))
    ap.add_argument("--rows-out", default=None, help=f"stock-event rows parquet (default {TRADES_OUT})")
    ap.add_argument("--n-placebo", type=int, default=N_PLACEBO)
    args = ap.parse_args(argv)

    limited = args.limit_batches is not None
    use_ledger = (not limited) and (not args.no_ledger)
    out_path = Path(args.out).expanduser() if args.out else (SMOKE_OUT if limited else OUT_JSON)
    rows_out = Path(args.rows_out).expanduser() if args.rows_out else (
        TRADES_OUT.with_name("event_rows_smoke.parquet") if limited else TRADES_OUT)
    history = Path(args.history).expanduser()
    bench_dir = Path(args.bench_dir).expanduser()
    if not sorted(history.glob("batch_*.parquet")):
        raise SystemExit(f"no batch files in {history}")
    delisted_path = Path(args.delisted).expanduser()
    delisted = pd.read_csv(delisted_path) if delisted_path.exists() else None
    if delisted is None:
        print(f"note: {delisted_path} not found, survivorship terms (and so the verdicts) will be incomplete", flush=True)

    data_end = mtf.data_end(history)
    mtf.ensure_benchmarks((EVENT_BENCH, SECONDARY_BENCH), dest=bench_dir)
    spy = mtf.load_benchmark(EVENT_BENCH, bench_dir, end=data_end)
    smh = mtf.load_benchmark(SECONDARY_BENCH, bench_dir, end=data_end)
    ev_spy = confirmation_events(spy, data_end)
    ev_smh = confirmation_events(smh, data_end)
    bench_weeks = mtf.weekly_bars(spy, data_end).index
    all_event_weeks = list(ev_spy["week"]) + list(ev_smh["week"])
    placebo = placebo_weeks(bench_weeks, all_event_weeks, data_end, args.n_placebo)
    weeks = sorted(set(all_event_weeks) | set(placebo))
    print(f"data through {data_end.date()}; {len(ev_spy)} {EVENT_BENCH} events, {len(ev_smh)} {SECONDARY_BENCH} events, "
          f"{len(placebo)} placebo weeks", flush=True)
    print(format_events(EVENT_BENCH, ev_spy), flush=True)
    print(format_events(SECONDARY_BENCH, ev_smh), flush=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.time()
        before = L.trial_count()
        if use_ledger:
            L.log_trial({"stage": LEDGER_STAGE, "event": "preregistered",
                         "signatures": {s: dict(mtf.SIGNATURES[s]) for s in SIGNATURE_NAMES},
                         "exits": {k: dataclasses.asdict(v) for k, v in EXITS.items()},
                         "event_rule": EVENT_RULE,
                         "events": {EVENT_BENCH: [str(w.date()) for w in ev_spy["week"]],
                                    SECONDARY_BENCH: [str(w.date()) for w in ev_smh["week"]]},
                         "placebo_weeks": [str(w.date()) for w in placebo],
                         "verdict_rule": VERDICT_RULE, "disclosure": EXPLORATION_DISCLOSURE})
        _warm_jit()
        parts, spans = [], []
        n_stocks = 0
        for batch in mtf.map_universe(history, stock_rows, (spy, weeks, data_end), workers=args.workers,
                                      limit_batches=args.limit_batches):
            for item in batch:
                n_stocks += 1
                spans.append(item["span"])
                if item["rows"] is not None and len(item["rows"]):
                    parts.append(item["rows"])
        rows = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
            columns=["symbol", "week"] + list(SIGNATURE_COLUMNS) + [f"{p}_{x}" for x in EXITS for p in ("ret", "reason", "weeks", "mfe")]
            + [f"hret_{h}" for h in HORIZONS] + [f"fmax_{HORIZONS[-1]}"])
        span_df = pd.DataFrame(spans, columns=["symbol", "first_date", "last_date"])
        print(f"scanned {n_stocks} stocks, {len(rows)} liquid stock-event rows; analysing... ({time.time() - t0:.0f}s)", flush=True)
        analysis = analyze(rows, {"spy": ev_spy, "smh": ev_smh}, placebo, span_df, delisted)

        analysis.update({"data_end": data_end, "limited_run": limited, "n_stocks_scanned": n_stocks,
                         "history": str(history), "preregistration": {
                             "signatures": {s: dict(mtf.SIGNATURES[s]) for s in SIGNATURE_NAMES},
                             "exits": {k: dataclasses.asdict(v) for k, v in EXITS.items()},
                             "primary_exit": PRIMARY_EXIT, "event_rule": EVENT_RULE, "verdict_rule": VERDICT_RULE,
                             "disclosure": EXPLORATION_DISCLOSURE, "alpha": ALPHA, "alpha_verdict": ALPHA_VERDICT,
                             "n_boot": N_BOOT, "seed": SEED, "placebo_seed": PLACEBO_SEED},
                         "events": {EVENT_BENCH: ev_spy, SECONDARY_BENCH: ev_smh, "placebo_weeks": placebo},
                         "verdict_rule": VERDICT_RULE, "disclosure": EXPLORATION_DISCLOSURE,
                         "ledger_trials_before": before})
        if use_ledger:
            for s in SIGNATURE_NAMES:
                for x in EXITS:
                    b = analysis["signatures"][s]["spy"][f"ret_{x}"][SPLIT_HELD_OUT]
                    L.log_trial({"stage": LEDGER_STAGE, "signature": s, "exit": x,
                                 "n_trades_heldout": b["n_trades"], "excess_mean": b["ci_event_verdict"]["mean"],
                                 "excess_lo": b["ci_event_verdict"]["lo"], "alpha": ALPHA_VERDICT,
                                 "data_end": str(data_end.date())})
        analysis["ledger_trials_after"] = L.trial_count()

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(_jsonable(analysis), indent=1, allow_nan=False))
        rows_out.parent.mkdir(parents=True, exist_ok=True)
        rows.to_parquet(rows_out, index=False)

    print()
    print(format_signature_table(analysis))
    for s in SIGNATURE_NAMES:
        e = analysis["signatures"][s]
        pc = e["placebo_comparison"][PRIMARY_OUTCOME]
        pl = e["placebo"][PRIMARY_OUTCOME]["all"]
        i22 = e["spy"][PRIMARY_OUTCOME][SPLIT_IN_SAMPLE]
        print(f"{s} placebo: excess {_n(pl['mean_excess'])} over {pl['n_events_with_trades']} weeks; "
              f"events-minus-placebo {_n(pc['mean'])} {_ci(pc)} | 2022 (in-sample, excluded): "
              f"n={i22['n_trades']} excess {_n(i22['mean_excess'])} {_ci(i22['ci_event'])}")
    print()
    for s in VERDICT_SIGNATURES:
        v = analysis["verdict"][s]
        print(f"VERDICT {s} ({mtf.SIGNATURE_LABELS[s]}): {v['verdict']}")
        if v["checks"]:
            print("  checks: " + "; ".join(f"{k}={'PASS' if ok else 'FAIL'}" for k, ok in v["checks"].items()))
    if limited:
        print("  (limited run: a smoke test of the machinery, NOT the study)")
    print(f"wrote {out_path} and {rows_out} | ledger trials: {before} -> {analysis['ledger_trials_after']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
