"""
What did the big 2021-2026 winners look like on weekly and monthly charts at their lows, at the first
confirmations and along the run, and how unremarkable were they against ALL liquid stocks that week?

    python scripts/mtf_case_study.py                              # full run, writes docs/mtf_case_study/
    python scripts/mtf_case_study.py --limit-batches 3 --workers 8 --out-dir ~/.roboalgo/mtf/case_smoke   # smoke
    python scripts/mtf_case_study.py --symbols NVDA AMD SPY       # a few names only

THE QUESTION. For each of 46 names (the five the user named, the chip and AI-infrastructure leaders, three
mega caps and three benchmark ETFs) find the best run since RUN_START: the weekly close with the highest
later weekly close (the LOW), and the peak after it. Then describe the stock at the low (every PCT_COLUMNS
value and its PERCENTILE among all liquid stocks that same week) and the first week after the low in which
each confirmation appeared: weekly close above the 30-week SMA, a rising 30-week SMA, monthly close above the
10-month SMA, a 26-week closing breakout, the RS line (vs SPY) at a 52-week high, a volume expansion, and
"first confluence" (weekly and monthly trend agree). Each milestone is reported with its week, the weeks
after the low and the percent off the low. The five pre-registered signatures of setup_engine/mtf.py are
reported with the first week each fires in [low - 8 weeks, low + 52 weeks] (it can be BEFORE the low).
Rides (setup_engine/ride.py: R1 weekly SMA30, R2 monthly SMA10, R3 30% trail, all with a 35% stop) are
simulated from the first-confluence week: signal at the weekly close, entry at the next daily open.
One SVG chart per name and an HTML index (cohort tables first) are written next to results.json.

EX-POST SELECTION. The winners are chosen BECAUSE they won, and the low is the week with the best later
close: both use the future by construction (setup_engine.mtf.best_run). Nothing here is evidence of an edge.
Whether a signature beats all liquid stocks bought the same day is the job of scripts/mtf_event_study.py,
which is pre-registered and out of sample. This script only describes. No ledger is touched.

NO LOOK-AHEAD (for the descriptions, not for the labels). A weekly or monthly bar is used only once it is
COMPLETE (label <= the evaluation date). Every series, benchmarks included, is clipped to the price
cache's last date (``data_end``) before anything is computed (yfinance runs to today). Percentiles and
cross-sectional ranks are same-week cross-sections of the liquid universe (price >= 2, median 13-week
dollar volume >= 5e6, >= 60 weeks of history), so the name is compared with what was investable that week.
The name's own value is part of that universe when it is liquid.

PERCENTILE DEFINITION (one rule for every name, in the cache or not): the share of the week's liquid
universe values that are <= the name's value (finite values only); None when fewer than MIN_UNIVERSE
liquid values exist that week. A LOW percentile of w_dd52 or w_ret26 therefore means "deeper or weaker
than most", a high one "shallower or stronger". The Performance Rank columns the signatures need are
filled the same way for every name (percentile of w_ret13 / w_ret26 against the panel), so TSM, ASML,
ARM and the benchmark ETFs, which are not in the cache, are ranked against the same universe.

BIASES. Survivorship (today's listings only: the universe the percentiles are taken against is the
survivors, which flatters every "how unremarkable" statement toward ordinary-looking lows); Yahoo's
retroactive split adjustment; no dividend adjustment; a name whose history is short or has a data gap
(NBIS: Yandex history and a halt; SNDK: a 2025 spin-off; ARM/ALAB/CRDO: IPOs) is flagged in its case and must
not be interpreted without the flag; the window of 3 batches in a smoke run gives a small universe.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from setup_engine import mtf
from setup_engine import ride as RD
from setup_engine import svgchart as SV
from setup_engine.screen import Screen

# ── pre-registered choices (fixed before any result is seen) ──────────────────────────────────────

USER_NAMES = ("AMD", "NVDA", "MSFT", "AMZN", "TSLA")
CHIPS = ("NVDA", "AMD", "AVGO", "MU", "QCOM", "INTC", "TXN", "AMAT", "LRCX", "KLAC", "MRVL", "ON", "NXPI",
         "MCHP", "ADI", "MPWR", "TER", "SMCI", "TSM", "ASML", "ARM", "ALAB", "CRDO", "COHR", "LITE")
AI_INFRA = ("ANET", "VRT", "WDC", "STX", "DELL", "GLW", "CIEN", "CLS", "SNDK", "NBIS", "PLTR", "APP")
MEGA = ("META", "GOOGL", "AAPL")
BENCH_NAMES = ("SPY", "QQQ", "SMH")
COHORT = tuple(dict.fromkeys(USER_NAMES + CHIPS + AI_INFRA + MEGA + BENCH_NAMES))
NOT_IN_CACHE = ("TSM", "ASML", "ARM", "SPY", "QQQ", "SMH")     # loaded from mtf.BENCH_DIR
RS_BENCH = "SPY"
RUN_START = "2021-10-08"
PANEL_START = "2020-06-01"
CHART_START = "2019-01-01"
PCT_COLUMNS = ("w_ret13", "w_ret26", "w_dd52", "w_px_sma30", "w_slope30", "w_volx", "rs_ret26", "w_rsi14",
               "m_px_sma10", "xs_rank_ret13", "xs_rank_ret26")
VOL_EXPANSION = 1.5
RIDES = {"R1": RD.RIDE_R1, "R2": RD.RIDE_R2, "R3": RD.RIDE_R3}
MIN_UNIVERSE = 200                  # fewer liquid values than this in a week -> no percentile
WINDOW_BEFORE_WEEKS = 8             # signature window: [low - 8 weeks, low + 52 weeks]
WINDOW_AFTER_WEEKS = 52
GAP_WARN_DAYS = 30                  # a gap between consecutive weekly bars inside the run longer than this
SHORT_HISTORY_DAYS = 730            # less history than this before the low -> "short history"
LOWS_CALENDAR_TOP = 5
OUT_DIR = ROOT / "docs" / "mtf_case_study"
SMOKE_DIR = Path.home() / ".roboalgo" / "mtf" / "case_smoke"

MILESTONES = ("low", "w_above_sma30", "sma30_rising", "m_above_sma10", "brk26", "rs_new_hi52",
              "vol_expansion", "first_confluence")
MILESTONE_LABELS = {"low": "Low", "w_above_sma30": "30w reclaim", "sma30_rising": "30w SMA rising",
                    "m_above_sma10": "10m reclaim", "brk26": "26w breakout", "rs_new_hi52": "RS 52w high",
                    "vol_expansion": "volume x1.5 up-week", "first_confluence": "CONFLUENCE"}
SIGNATURE_KEYS = tuple(f"sig_{s}" for s in mtf.SIGNATURES)
GROUPS_REPORTED = ("chips_ai", "user", "all_stocks")
HEADER_WIDTH_CHARS = 135            # chart header lines are wrapped at " | " to stay inside the canvas
CHART_SIGNATURES = ("S1", "S2", "S3", "S4")      # B0 is "price above the 30-week SMA": same as the reclaim marker
# columns the panel keeps: percentile inputs + everything a signature reads (cross-sectional ranks excluded)
PANEL_KEEP = tuple(dict.fromkeys(
    [c for c in PCT_COLUMNS if c not in mtf.XS_COLUMNS] + ["w_ret13", "w_ret26"]
    + [c for rules in mtf.SIGNATURES.values() for c in Screen(rules).fields() if c not in mtf.XS_COLUMNS]))


# ── helpers ───────────────────────────────────────────────────────────────────────────────────────

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
        return {str(k): _jsonable(v) for k, v in o.items() if not str(k).startswith("_")}
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


def _clip(df: pd.DataFrame, end) -> pd.DataFrame:
    return df[df.index <= pd.Timestamp(end)]


def _fin(x):
    """float(x) or None if missing / not finite."""
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _median(xs):
    v = [x for x in (_fin(i) for i in xs) if x is not None]
    return float(np.median(v)) if v else None


def _d(x) -> str:
    if x is None:
        return "-"
    try:
        t = pd.Timestamp(x)
    except (TypeError, ValueError):
        return str(x)
    return "-" if pd.isna(t) else t.date().isoformat()


def _pc(x, nd: int = 0) -> str:
    x = _fin(x)
    return "n/a" if x is None else f"{x * 100:+.{nd}f}%"


def _pctile(x) -> str:
    x = _fin(x)
    return "n/a" if x is None else f"{x * 100:.0f}"


def group_of(symbol: str) -> str:
    if symbol in USER_NAMES:
        return "user"
    if symbol in CHIPS or symbol in AI_INFRA:
        return "chips_ai"
    if symbol in MEGA:
        return "mega"
    if symbol in BENCH_NAMES:
        return "bench"
    return "other"


def _in_group(symbol: str, group: str) -> bool:
    if group == "all_stocks":
        return group_of(symbol) != "bench"
    if group == "chips_ai":
        return symbol in CHIPS or symbol in AI_INFRA
    if group == "user":
        return symbol in USER_NAMES
    raise ValueError(f"unknown group {group!r}")


# ── percentiles against the liquid universe of a week ─────────────────────────────────────────────

def _universe_values(panel_week: pd.DataFrame, column: str) -> np.ndarray:
    """Finite values of ``column`` among the LIQUID rows of one week's panel slice."""
    v = panel_week[column].to_numpy()
    if "liquid" in panel_week.columns:
        v = v[panel_week["liquid"].fillna(False).to_numpy(dtype=bool)]
    return v[np.isfinite(v)]


def _rank(sorted_vals: np.ndarray, value) -> float | None:
    """Share of ``sorted_vals`` <= value (None if value is missing or the universe is too small)."""
    x = _fin(value)
    if x is None or len(sorted_vals) < MIN_UNIVERSE:
        return None
    return float(np.searchsorted(sorted_vals, sorted_vals.dtype.type(x), side="right") / len(sorted_vals))


def percentile_of(panel_week: pd.DataFrame, column: str, value) -> float | None:
    """Percentile of ``value`` among the liquid rows of one week (see the module docstring)."""
    if column not in panel_week.columns:
        return None
    return _rank(np.sort(_universe_values(panel_week, column)), value)


class PanelView:
    """The panel grouped by week with the sorted universe values cached, so 46 names x 300 weeks stay cheap."""

    def __init__(self, panel: pd.DataFrame | None):
        self.groups: dict = {}
        self.n_rows = 0
        if panel is not None and len(panel):
            self.n_rows = len(panel)
            self.groups = {pd.Timestamp(w): g for w, g in panel.groupby("week", sort=True)}
        self._sorted: dict = {}

    def percentile(self, week, column: str, value) -> float | None:
        key = (pd.Timestamp(week), column)
        if key not in self._sorted:
            g = self.groups.get(key[0])
            self._sorted[key] = (None if g is None or column not in g.columns
                                 else np.sort(_universe_values(g, column)))
        vals = self._sorted[key]
        return None if vals is None else _rank(vals, value)

    def sizes(self) -> dict:
        n = [int(g["liquid"].sum()) if "liquid" in g.columns else len(g) for g in self.groups.values()]
        return {"n_rows": int(self.n_rows), "n_weeks": len(n),
                "median_liquid_per_week": float(np.median(n)) if n else None,
                "min_liquid_per_week": int(min(n)) if n else None,
                "weeks_with_percentiles": int(sum(1 for x in n if x >= MIN_UNIVERSE))}


# ── the liquid-universe panel ─────────────────────────────────────────────────────────────────────

def panel_rows(symbol: str, daily: pd.DataFrame, bench: pd.DataFrame, as_of, start) -> pd.DataFrame | None:
    """One stock's LIQUID weekly rows since ``start`` (the ``map_universe`` function); None if it has none."""
    wf = mtf.mtf_features(daily, bench, as_of)
    if len(wf) == 0:
        return None
    sel = (wf.index >= pd.Timestamp(start)) & mtf.is_liquid(wf)
    if not sel.any():
        return None
    out = wf.loc[sel, list(PANEL_KEEP)].astype("float32")
    out.insert(0, "week", wf.index[sel])
    out.insert(0, "symbol", symbol)
    out["liquid"] = True
    return out.reset_index(drop=True)


def build_panel(history, bench: pd.DataFrame, as_of, workers: int = 1, limit_batches=None) -> pd.DataFrame:
    """All liquid weekly rows of the (limited) cache since PANEL_START, with the cross-sectional ranks added."""
    frames = []
    for res in mtf.map_universe(history, panel_rows, (bench, as_of, PANEL_START), workers=workers,
                                limit_batches=limit_batches):
        frames.extend(res)
    if not frames:
        return pd.DataFrame({"symbol": [], "week": pd.DatetimeIndex([]), "liquid": np.array([], dtype=bool),
                             **{c: np.array([], dtype="float32") for c in PANEL_KEEP}})
    panel = pd.concat(frames, ignore_index=True)
    return mtf.add_xs_ranks(panel, "week", "liquid")


# ── one case ──────────────────────────────────────────────────────────────────────────────────────

def _fill_xs(wf: pd.DataFrame, view: PanelView) -> pd.DataFrame:
    """Copy of the weekly table with xs_rank_ret13 / xs_rank_ret26 = percentile against the panel (NaN if none)."""
    out = wf.copy()
    for src, dst in (("w_ret13", "xs_rank_ret13"), ("w_ret26", "xs_rank_ret26")):
        vals = out[src].to_numpy(dtype="float64")
        res = np.full(len(out), np.nan)
        for i, (w, v) in enumerate(zip(out.index, vals)):
            p = view.percentile(w, src, v)
            if p is not None:
                res[i] = p
        out[dst] = res
    return out


def _first_true(cond: np.ndarray, start: int):
    idx = np.flatnonzero(cond[start:])
    return None if len(idx) == 0 else int(start + idx[0])


def find_milestones(wf: pd.DataFrame, li: int) -> dict:
    """Position (in ``wf``) of the first week k >= li where each milestone condition holds, else None.

    ``m_above_sma10`` needs the monthly bar to have COMPLETED on or after the low week (a month that
    ended before the low says nothing about the recovery); ``first_confluence`` uses the same monthly test.
    """
    low_week = wf.index[li]
    f = {k: wf[k].to_numpy(dtype="float64") for k in
         ("w_close", "w_sma30", "w_slope30", "m_close", "m_sma10", "w_prev_hi26", "rs_line", "rs_prev_hi52",
          "w_volx", "w_prev_close")}
    m_fresh = (wf["m_label"] >= low_week).to_numpy(dtype=bool)
    with np.errstate(invalid="ignore"):
        above = f["w_close"] > f["w_sma30"]
        rising = f["w_slope30"] > 0
        m_above = (f["m_close"] > f["m_sma10"]) & m_fresh
        conds = {
            "w_above_sma30": above,
            "sma30_rising": rising,
            "m_above_sma10": m_above,
            "brk26": f["w_close"] > f["w_prev_hi26"],
            "rs_new_hi52": f["rs_line"] > f["rs_prev_hi52"],
            "vol_expansion": (f["w_volx"] >= VOL_EXPANSION) & (f["w_close"] > f["w_prev_close"]),
            "first_confluence": above & rising & m_above,
        }
    out = {"low": li}
    for k, c in conds.items():
        out[k] = _first_true(c, li)
    return out


def _record(wf2: pd.DataFrame, k: int, low_week, low_close: float, view: PanelView, with_state: bool = True) -> dict:
    week = wf2.index[k]
    close = float(wf2["w_close"].iloc[k])
    rec = {"week": week, "weeks_after_low": int(round((week - low_week).days / 7)), "w_close": close,
           "pct_off_low": close / low_close - 1.0}
    if with_state:
        vals = {c: _fin(wf2[c].iloc[k]) for c in PCT_COLUMNS}
        rec["values"] = vals
        rec["pct"] = {c: view.percentile(week, c, vals[c]) for c in PCT_COLUMNS}
    return rec


def _ride_dict(r: dict | None) -> dict | None:
    if r is None:
        return None
    keep = ("signal_week", "signal_date", "entry_date", "exit_date", "entry", "exit", "ret_gross", "ret_net",
            "weeks_held", "mfe", "mfe_close", "reason")
    return {k: r[k] for k in keep}


def case(symbol: str, daily: pd.DataFrame, bench: pd.DataFrame, panel, as_of) -> dict:
    """The full description of one name: run, milestones, state at the low, signatures, rides, data flags.

    ``daily`` must already be clipped to ``as_of`` (ValueError otherwise): a series that runs past the
    data end would let the ride simulation see bars the study does not have. ``panel`` is the liquid
    universe (DataFrame or PanelView); an empty panel just gives no percentiles.
    """
    as_of = pd.Timestamp(as_of)
    if daily.index[-1] > as_of:
        raise ValueError(f"{symbol}: daily data runs to {daily.index[-1].date()}, past the data end "
                         f"{as_of.date()}: clip every series to data_end first")
    view = panel if isinstance(panel, PanelView) else PanelView(panel)
    wf = mtf.mtf_features(daily, bench, as_of)
    out = {"symbol": symbol, "group": group_of(symbol), "history_start": daily.index[0],
           "last_day": daily.index[-1], "last_week": wf.index[-1] if len(wf) else None,
           "n_weeks": int(len(wf)), "warnings": [], "run": None, "milestones": {}, "signatures": {},
           "state_at_low": None, "rides": {}, "hold": None, "liquid_at_low": None}
    run = mtf.best_run(wf["w_close"], RUN_START) if len(wf) else None
    if run is None:
        out["warnings"].append("no run found (fewer than 52 weekly closes since the run start)")
        out["_wf"] = wf
        return out
    out["run"] = run
    low_week, low_close = run["low_week"], run["low_close"]
    li = int(wf.index.get_loc(low_week))
    wf2 = _fill_xs(wf, view)
    if run["max_gap_days"] > GAP_WARN_DAYS:
        out["warnings"].append(f"data gap: {run['max_gap_days']} days between weekly bars inside the run")
    if (low_week - daily.index[0]).days < SHORT_HISTORY_DAYS:
        out["warnings"].append(f"short history: first bar {daily.index[0].date()} "
                               f"(drawdown into the low and 52-week statistics are partial)")
    out["liquid_at_low"] = bool(mtf.is_liquid(wf.iloc[[li]])[0])

    pos = find_milestones(wf2, li)
    for key in MILESTONES:
        out["milestones"][key] = None if pos[key] is None else _record(wf2, pos[key], low_week, low_close, view)
    low_rec = out["milestones"]["low"]
    out["state_at_low"] = {"values": low_rec["values"], "pct": low_rec["pct"]}

    lo_w = low_week - pd.Timedelta(weeks=WINDOW_BEFORE_WEEKS)
    hi_w = low_week + pd.Timedelta(weeks=WINDOW_AFTER_WEEKS)
    window = np.asarray((wf2.index >= lo_w) & (wf2.index <= hi_w))
    for name, rules in mtf.SIGNATURES.items():
        fires = Screen(rules).mask(wf2) & window
        k = _first_true(fires, 0)
        out["signatures"][name] = None if k is None else _record(wf2, k, low_week, low_close, view, with_state=False)

    k_conf = pos["first_confluence"]
    out["rides"] = {name: None for name in RIDES}
    if k_conf is not None:
        wb, mb = mtf.weekly_bars(daily, as_of), mtf.monthly_bars(daily, as_of)
        if not wb.index.equals(wf.index):
            raise ValueError(f"{symbol}: weekly bars and weekly features disagree on the index")
        rd = RD.prepare(daily, wb, mb)
        entry = None
        for name, ex in RIDES.items():
            r = RD.ride(rd, k_conf, ex)
            out["rides"][name] = _ride_dict(r)
            entry = entry if entry is not None or r is None else r["entry"]
        out["hold"] = {"low_to_peak": run["mult"],
                       "confluence_to_peak": None if entry is None else run["peak_close"] / entry,
                       "confluence_after_peak": bool(wf.index[k_conf] > run["peak_week"])}
    else:
        out["hold"] = {"low_to_peak": run["mult"], "confluence_to_peak": None, "confluence_after_peak": False}
    out["_wf"] = wf2
    return out


# ── summaries ─────────────────────────────────────────────────────────────────────────────────────

def _ms(c: dict, key: str):
    if key.startswith("sig_"):
        return c["signatures"].get(key[4:])
    return c["milestones"].get(key)


def summaries(cases: dict) -> dict:
    """The cohort tables (a)-(f) from the per-name cases."""
    with_run = {s: c for s, c in cases.items() if c["run"] is not None}
    runs = sorted(({"symbol": s, "group": c["group"], "low_week": c["run"]["low_week"],
                    "peak_week": c["run"]["peak_week"], "mult": c["run"]["mult"], "weeks": c["run"]["weeks"],
                    "mdd_in_run": c["run"]["mdd_in_run"], "dd_into_low": c["run"]["dd_into_low"],
                    "max_gap_days": c["run"]["max_gap_days"], "warnings": list(c["warnings"])}
                   for s, c in with_run.items()), key=lambda r: -r["mult"])
    out = {"runs": runs, "milestones": {}, "state_at_low": {}, "rides": {}, "signatures": {}, "lows_calendar": []}
    for g in GROUPS_REPORTED:
        members = {s: c for s, c in with_run.items() if _in_group(s, g)}
        ms = {}
        for key in MILESTONES + SIGNATURE_KEYS:
            recs = [r for r in (_ms(c, key) for c in members.values()) if r is not None]
            ms[key] = {"n": len(members), "n_reached": len(recs),
                       "median_weeks_after_low": _median(r["weeks_after_low"] for r in recs),
                       "median_pct_off_low": _median(r["pct_off_low"] for r in recs)}
        out["milestones"][g] = ms
        out["state_at_low"][g] = {
            col: {"n": sum(1 for c in members.values() if _fin(c["state_at_low"]["pct"].get(col)) is not None),
                  "median_pct": _median(c["state_at_low"]["pct"].get(col) for c in members.values()),
                  "median_value": _median(c["state_at_low"]["values"].get(col) for c in members.values())}
            for col in PCT_COLUMNS}
        rides = {}
        for name in RIDES:
            rs = [(c["rides"].get(name), c["run"]["mult"]) for c in members.values()]
            rs = [(r, m) for r, m in rs if r is not None]
            rides[name] = {
                "n": len(rs),
                "median_ret_net": _median(r["ret_net"] for r, _ in rs),
                "share_ge_100pct": (sum(1 for r, _ in rs if r["ret_net"] >= 1.0) / len(rs)) if rs else None,
                "median_capture": _median((1.0 + r["ret_net"]) / m for r, m in rs),
                "median_weeks_held": _median(r["weeks_held"] for r, _ in rs),
                "reasons": dict(Counter(r["reason"] for r, _ in rs))}
        out["rides"][g] = rides
    for s, c in cases.items():
        out["signatures"][s] = {k[4:]: (None if _ms(c, k) is None else _ms(c, k)["weeks_after_low"])
                                for k in SIGNATURE_KEYS} if c["run"] is not None else {}
    by_week: dict = {}
    for s, c in with_run.items():
        if c["group"] != "bench":
            by_week.setdefault(c["run"]["low_week"], []).append(s)
    top = sorted(by_week.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:LOWS_CALENDAR_TOP]
    out["lows_calendar"] = [{"week": w, "n": len(v), "symbols": sorted(v)} for w, v in top]
    return out


# ── tables (one source for the console and the HTML) ──────────────────────────────────────────────

def build_tables(summ: dict, cases: dict) -> list[dict]:
    T = []
    T.append({"name": "runs", "console": True,
              "caption": "(a) Best run since the run start, sorted by multiple (low close -> peak close; "
                         "ex-post, selected on the outcome)",
              "header": ["symbol", "group", "low week", "peak week", "mult", "weeks", "mdd in run", "below 2y high",
                         "flags"],
              "rows": [[r["symbol"], r["group"], _d(r["low_week"]), _d(r["peak_week"]), f"x{r['mult']:.2f}",
                        str(r["weeks"]), _pc(r["mdd_in_run"]), _pc(r["dd_into_low"]), "; ".join(r["warnings"])]
                       for r in summ["runs"]]})
    rows = []
    for key in MILESTONES + SIGNATURE_KEYS:
        row = [MILESTONE_LABELS.get(key, key)]
        for g in GROUPS_REPORTED:
            m = summ["milestones"][g][key]
            w, p = m["median_weeks_after_low"], m["median_pct_off_low"]
            row.append(f"{m['n_reached']}/{m['n']}  {'n/a' if w is None else f'{w:+.0f}w'}  {_pc(p)}")
        rows.append(row)
    T.append({"name": "milestones", "console": True,
              "caption": "(b) First week each confirmation appeared: names reached / names, median weeks after "
                         "the low, median percent off the low (signature rows: first fire in [low-8w, low+52w])",
              "header": ["milestone"] + list(GROUPS_REPORTED), "rows": rows})
    rows = []
    for col in PCT_COLUMNS:
        row = [col]
        for g in GROUPS_REPORTED:
            s = summ["state_at_low"][g][col]
            row.append(f"{_pctile(s['median_pct'])} (n={s['n']})")
        rows.append(row)
    T.append({"name": "state_at_low", "console": True,
              "caption": "(c) State at the low: median PERCENTILE among all liquid stocks that week "
                         "(0 = lowest value, 100 = highest; n = names with a percentile)",
              "header": ["column"] + list(GROUPS_REPORTED), "rows": rows})
    rows = []
    for name in RIDES:
        for g in GROUPS_REPORTED:
            r = summ["rides"][g][name]
            rows.append([name, g, str(r["n"]), _pc(r["median_ret_net"]), _pc(r["share_ge_100pct"]),
                         "n/a" if r["median_capture"] is None else f"{r['median_capture'] * 100:.0f}%",
                         "n/a" if r["median_weeks_held"] is None else f"{r['median_weeks_held']:.0f}",
                         ", ".join(f"{k}:{v}" for k, v in sorted(r["reasons"].items()))])
    T.append({"name": "rides", "console": True,
              "caption": "(d) Ride outcomes from the first-confluence week (entry next open, 1% round trip): "
                         "median net return, share >= +100%, median capture = (1+ret)/run multiple",
              "header": ["exit", "group", "n", "median ret", ">= +100%", "capture", "weeks held", "exit reasons"],
              "rows": rows})
    sigs = list(mtf.SIGNATURES)
    rows = [[s] + [("-" if summ["signatures"][s].get(n) is None else f"{summ['signatures'][s][n]:+d}w")
                   for n in sigs] for s in summ["signatures"] if summ["signatures"][s]]
    T.append({"name": "signatures", "console": True,
              "caption": "(e) Signatures: first week each fired in [low-8w, low+52w], in weeks after the low "
                         "('-' = never in the window)",
              "header": ["symbol"] + sigs, "rows": rows})
    T.append({"name": "lows_calendar", "console": True,
              "caption": "(f) Lows calendar: how many names bottomed in the same week (top weeks, benchmarks excluded)",
              "header": ["low week", "names", "symbols"],
              "rows": [[_d(r["week"]), str(r["n"]), " ".join(r["symbols"])] for r in summ["lows_calendar"]]})
    rows = []
    for s, c in sorted(cases.items(), key=lambda kv: -(kv[1]["run"]["mult"] if kv[1]["run"] else 0)):
        if c["state_at_low"] is None:
            continue
        rows.append([s] + [_pctile(c["state_at_low"]["pct"].get(col)) for col in PCT_COLUMNS])
    T.append({"name": "state_by_name", "console": False,
              "caption": "(g) State at the low by name: percentile among liquid stocks that week",
              "header": ["symbol"] + list(PCT_COLUMNS), "rows": rows})
    return T


def text_table(t: dict) -> str:
    cols = [t["header"]] + t["rows"]
    widths = [max(len(str(r[i])) for r in cols) for i in range(len(t["header"]))]
    line = lambda r: "  ".join(str(x).ljust(w) for x, w in zip(r, widths)).rstrip()  # noqa: E731
    return "\n".join([t["caption"], line(t["header"]), line(["-" * w for w in widths])] + [line(r) for r in t["rows"]])


def _esc(s) -> str:
    return SV._t(s)


def html_tables(tables: list[dict]) -> str:
    parts = ['<div class="cohort">', "<h2>Cohort summary</h2>"]
    for t in tables:
        parts.append(f"<h3>{_esc(t['caption'])}</h3><table><thead><tr>"
                     + "".join(f"<th>{_esc(h)}</th>" for h in t["header"]) + "</tr></thead><tbody>")
        for r in t["rows"]:
            parts.append("<tr>" + "".join(f"<td>{_esc(x)}</td>" for x in r) + "</tr>")
        parts.append("</tbody></table>")
    parts.append("</div>")
    return "".join(parts)


_TABLE_CSS = ("table{border-collapse:collapse;font-size:12px;margin:4px 0 14px}"
              "th,td{border:1px solid #d3d8df;padding:2px 7px;text-align:right}"
              "th:first-child,td:first-child{text-align:left}h3{font-size:14px;margin:16px 0 2px}")


def write_index(path, entries, title, intro_lines, tables: list[dict]) -> Path:
    """svgchart.write_html_index, then the cohort tables inserted ABOVE the per-name sections."""
    p = SV.write_html_index(path, entries, title, intro_lines)
    html = p.read_text(encoding="utf-8")
    block = html_tables(tables)
    html = html.replace("</style>", _TABLE_CSS + "</style>", 1)
    for marker in ("<nav>", '<section id="e0">', "</body>"):
        if marker in html:
            html = html.replace(marker, block + marker, 1)
            break
    p.write_text(html, encoding="utf-8")
    return p


# ── charts ────────────────────────────────────────────────────────────────────────────────────────

def _entry_lines(c: dict) -> list[str]:
    """The summary lines under a name in the HTML index (also the chart header content)."""
    lines = []
    run = c["run"]
    if run is None:
        return list(c["warnings"]) or ["no run"]
    lines.append(f"Run: low {_d(run['low_week'])} close {run['low_close']:.2f} -> peak {_d(run['peak_week'])} "
                 f"close {run['peak_close']:.2f} = x{run['mult']:.2f} in {run['weeks']} weeks")
    lines.append(f"Drawdown inside the run {_pc(run['mdd_in_run'])}; the low was {_pc(run['dd_into_low'])} vs the "
                 f"prior 2-year high")
    ms = []
    for key in MILESTONES[1:]:
        r = c["milestones"].get(key)
        ms.append(f"{MILESTONE_LABELS[key]} " + ("never" if r is None else f"+{r['weeks_after_low']}w/{_pc(r['pct_off_low'])}"))
    lines.append("After the low: " + " | ".join(ms))
    sg = []
    for n in mtf.SIGNATURES:
        r = c["signatures"].get(n)
        sg.append(f"{n} " + ("-" if r is None else f"{r['weeks_after_low']:+d}w/{_pc(r['pct_off_low'])}"))
    lines.append("Signatures first fired in [low-8w, low+52w]: " + " | ".join(sg))
    st = c["state_at_low"]["pct"]
    lines.append("At the low, percentile among liquid stocks: " + ", ".join(
        f"{k} {_pctile(st.get(k))}" for k in ("w_ret26", "w_dd52", "rs_ret26", "w_rsi14", "w_px_sma30")))
    rd = []
    for name in RIDES:
        r = c["rides"].get(name)
        rd.append(f"{name} " + ("n/a" if r is None else f"{_pc(r['ret_net'])} ({r['reason']}, {r['weeks_held']:.0f}w)"))
    hold = c["hold"] or {}
    lines.append("Ride from first confluence: " + " | ".join(rd)
                 + f" | buy at the low, hold to the peak x{run['mult']:.2f}"
                 + ("" if hold.get("confluence_to_peak") is None else f" | from the confluence entry x{hold['confluence_to_peak']:.2f}"))
    lines.extend(f"WARNING: {w}" for w in c["warnings"])
    return lines


def _wrap(lines: list[str], width: int = HEADER_WIDTH_CHARS) -> list[str]:
    """Split lines longer than ``width`` at their " | " separators (continuation lines are indented)."""
    out = []
    for line in lines:
        if len(line) <= width or " | " not in line:
            out.append(line)
            continue
        cur = ""
        for part in line.split(" | "):
            if cur and len(cur) + 3 + len(part) > width:
                out.append(cur)
                cur = "    " + part
            else:
                cur = part if not cur else cur + " | " + part
        out.append(cur)
    return out


def chart(symbol: str, c: dict, daily: pd.DataFrame, as_of, out_dir) -> Path:
    """Write {SYMBOL}.svg: weekly candles since CHART_START with the low, every confirmation, the peak and the R3 ride."""
    wb = mtf.weekly_bars(daily, as_of)
    bars = wb.loc[wb.index >= pd.Timestamp(CHART_START), ["open", "high", "low", "close", "volume"]]
    wf = c["_wf"]
    markers = []
    run = c["run"]
    if run is not None:
        low = c["milestones"]["low"]
        markers.append(SV.Marker(run["low_week"], f"Low {run['low_close']:.2f}", "low"))
        for key in MILESTONES[1:]:
            r = c["milestones"].get(key)
            if r is not None:
                markers.append(SV.Marker(r["week"], f"{MILESTONE_LABELS[key]} {_pc(r['pct_off_low'])}", "confirm"))
        for n in CHART_SIGNATURES:
            r = c["signatures"].get(n)
            if r is not None:
                markers.append(SV.Marker(r["week"], f"{n} fires {_pc(r['pct_off_low'])}", "event"))
        markers.append(SV.Marker(run["peak_week"], f"Peak x{run['mult']:.1f} ({run['peak_close']:.2f})", "peak"))
        r3 = c["rides"].get("R3")
        if r3 is not None:
            markers.append(SV.Marker(r3["entry_date"], "R3 entry", "entry"))
            markers.append(SV.Marker(r3["exit_date"], f"R3 exit {_pc(r3['ret_net'])} ({r3['reason']})", "exit"))
    svg = SV.weekly_chart_svg(
        bars, sma30=wf["w_sma30"], sma10m=wf["m_sma10"], rs_line=wf["rs_line"], markers=markers,
        title=f"{symbol}  weekly chart (log scale)",
        header_lines=_wrap([f"Orange line = 30-week SMA, purple = 10-month SMA, bottom panel = RS line vs {RS_BENCH}"]
                           + _entry_lines(c)))
    return SV.write_svg(Path(out_dir) / f"{symbol}.svg", svg)


# ── main ──────────────────────────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1] if __doc__ else None)
    ap.add_argument("--history", default=str(mtf.DEFAULT_HISTORY))
    ap.add_argument("--bench-dir", default=str(mtf.BENCH_DIR))
    ap.add_argument("--limit-batches", type=int, default=None,
                    help="smoke: build the percentile panel from the first N batch files only (cohort data is read from all)")
    ap.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    ap.add_argument("--out-dir", default=None, help=f"default {OUT_DIR}, or {SMOKE_DIR} with --limit-batches")
    ap.add_argument("--symbols", nargs="+", default=None, help="only these names (default: the 46-name cohort)")
    args = ap.parse_args(argv)

    history, bench_dir = Path(args.history).expanduser(), Path(args.bench_dir).expanduser()
    limited = args.limit_batches is not None
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else (SMOKE_DIR if limited else OUT_DIR)
    cohort = tuple(dict.fromkeys(args.symbols)) if args.symbols else COHORT

    data_end = mtf.data_end(history)
    need_bench = tuple(dict.fromkeys((RS_BENCH,) + tuple(s for s in NOT_IN_CACHE if s in cohort)))
    mtf.ensure_benchmarks(need_bench, bench_dir)
    print(f"data end {data_end.date()}; cohort {len(cohort)} names; "
          f"panel from {'the first ' + str(args.limit_batches) + ' batches' if limited else 'all batches'}", flush=True)

    in_cache = [s for s in cohort if s not in NOT_IN_CACHE]
    loaded = mtf.load_daily(history, in_cache) if in_cache else {}
    daily: dict = {s: _clip(d, data_end) for s, d in loaded.items() if len(d)}
    for s in cohort:
        if s in NOT_IN_CACHE:
            daily[s] = mtf.load_benchmark(s, bench_dir, end=data_end)
    daily = {s: d for s, d in daily.items() if len(d)}
    missing = [s for s in cohort if s not in daily]
    if missing:
        print(f"not found (skipped): {', '.join(missing)}", flush=True)
    spy = daily[RS_BENCH] if RS_BENCH in daily else mtf.load_benchmark(RS_BENCH, bench_dir, end=data_end)

    panel = build_panel(history, spy, data_end, args.workers, args.limit_batches)
    view = PanelView(panel)
    sizes = view.sizes()
    print(f"panel: {sizes['n_rows']} liquid stock-weeks, {sizes['n_weeks']} weeks, median "
          f"{sizes['median_liquid_per_week']} liquid stocks per week ({sizes['weeks_with_percentiles']} weeks "
          f">= {MIN_UNIVERSE} for percentiles)", flush=True)

    cases = {}
    for s in cohort:
        if s in daily:
            cases[s] = case(s, daily[s], spy, view, data_end)
    summ = summaries(cases)
    tables = build_tables(summ, cases)

    out_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    order = sorted(cases, key=lambda s: -(cases[s]["run"]["mult"] if cases[s]["run"] else 0))
    for s in order:
        chart(s, cases[s], daily[s], data_end, out_dir)
        entries.append({"symbol": s, "svg_file": f"{s}.svg", "summary_lines": _entry_lines(cases[s])})
    intro = [
        f"Descriptive and EX POST: the names were chosen because they won and each low is the week with the best "
        f"later weekly close, so nothing here is evidence of an edge (that is what the out-of-sample event study tests).",
        f"Data end {data_end.date()} (every series, benchmarks included, is clipped to it); weekly and monthly bars "
        f"are used only once complete; run start {RUN_START}.",
        f"Percentiles compare each name with ALL liquid stocks that week (price >= 2, 13-week median dollar volume "
        f">= 5e6, >= 60 weeks): panel of {sizes['n_rows']} stock-weeks over {sizes['n_weeks']} weeks, median "
        f"{sizes['median_liquid_per_week']} liquid stocks per week" + (" (smoke run: first batches only)" if limited else "") + ".",
        "Survivorship: today's listings only. Yahoo split-adjusts retroactively; no dividends. "
        "Names with a data gap or short history carry a WARNING line.",
    ]
    write_index(out_dir / "index.html", entries, "Multi-timeframe case study: the 2021-2026 winners", intro, tables)

    results = {"data_end": data_end, "limited_run": limited, "batch_files_for_panel": args.limit_batches,
               "run_start": RUN_START, "panel_start": PANEL_START, "rs_bench": RS_BENCH, "cohort": list(cohort),
               "missing_symbols": missing, "min_universe": MIN_UNIVERSE, "panel": sizes,
               "signatures": mtf.SIGNATURES, "cases": cases, "summaries": summ}
    (out_dir / "results.json").write_text(json.dumps(_jsonable(results), indent=1, allow_nan=False))

    print()
    for t in tables:
        if t["console"]:
            print(text_table(t))
            print()
    print(f"wrote {len(entries)} charts, index.html and results.json to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
