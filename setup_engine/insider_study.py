"""
Pure logic of the pre-registered insider-cluster (long) and dilution-plus-burn (short) study on fallen
US stocks. No file is read and no network is touched here: prices, Form 4 frames and XBRL company facts
are passed in by the caller (scripts/insider_study.py), so every rule below is testable on a hand-built
frame. The Form 345 loader lives in data_engine/sec_insider.py and is NOT imported here; this module
codes to the frozen frame layouts (see FRAME LAYOUTS) and nothing else.

THE TWO QUESTIONS (both pre-registered, nothing is tuned):
  LONG. Among FALLEN stock-months, do those with an insider purchase CLUSTER in the last 90 days earn a
  higher 12-month return than fallen stocks without any insider purchase, in the same month?
  SHORT. Among FALLEN stock-months, do those with heavy dilution AND a short cash runway (dilution plus
  burn) suffer a higher rate of "wipeout" (the lowest close of the next 12 months at most half the
  origin close) than fallen stocks without both?

DEFINITIONS (C1-C12 are the planner clarifications that close look-ahead and mislabeling holes).
  C1 FALLEN = multibagger variant A exactly: month-end close in [$1, $10] inclusive, 60-month drawdown
     (dd60) <= -70%, AND mean daily dollar volume of the month >= $300,000. Origins 2006-01 to 2023-09.
  C2 ORIGIN-CLOSE TIMING. The month-end close is the close of the last trading day, and a Form 4 accepted
     after 4pm that day is public only after that close. A stock-month is TREATED iff a cluster event
     has event_date >= month_end - 89 days AND event_date < last_weekday(month_end). Control exclusion
     uses the full window [month_end - 89, month_end] (the stricter one). Short-arm XBRL reads use
     pit_date = last_weekday(month_end) - 1 day. last_weekday: Saturday steps back 1 day, Sunday 2.
     Exchange holidays are ignored (a known, documented simplification).
  C3 A cluster needs >= 2 distinct qualifying owners AND >= 2 distinct accession numbers (one joint
     filing is never a cluster). C4 only Forms 4 and 5 count as purchases (no amendments). C5 a filing's
     value is the sum of its code-P non-derivative rows, shares x price; qualifies at >= $5,000.
     (C3-C5 are applied by data_engine/sec_insider.py, whose output this module consumes.)
  C6 COVERAGE. A stock-month enters the long arm only if its symbol maps to exactly ONE issuer CIK at
     pit_date and every calendar quarter touched by the 90-day window has at least one filing in the
     loaded data. Excluded rows are counted by reason (GROUPS).
  C7 SHORT arm: classified only if BOTH dilution and burn are computable, else "unknown", excluded from
     both groups. A value whose period end is more than 550 days before the date it is read for counts
     as missing (sec_edgar's latest-value-however-old rule would otherwise leak stale tags).
  C8 DILUTION = shares_now / shares_12m_ago >= 1.20, as a RATIO (now/then - 1 >= 0.20 fails on 1.2M
     over 1.0M by floating point). Both counts come from ONE concept (dei first, else us-gaap).
  C9 BURN reads the latest ANNUAL operating cash flow row (duration 350 to 380 days) known at pit_date,
     never a later quarterly row. OCF >= 0 means burn is False. Otherwise burn is cash / |OCF| < 1.0 year.
  C10 PROVISIONAL iff no delisted names are in the price set (decided by the script).
  C11 Origin eligibility and dd60 come from the RAW month grid; forward outcomes come from the
     held-flat grid, so a held-flat month after a delisting is never an origin. An outcome at horizon h
     needs a finite close at i+h; fmin12 skips gaps but is NaN unless the close at i+12 is finite.
  C12 Secondary rows are descriptive (alpha 0.05) and never enter the verdicts.

STATISTICS. Mean excess return over the same-month control mean, bootstrap over whole calendar months
(setup_engine.trade_stats.cluster_bootstrap, 2000 draws, seed 0). ALPHA = 0.025 gives the 1.25% and
98.75% quantiles: a two-sided 97.5% interval (Bonferroni over the 2 primary tests). The lower bound is
the "lo" key. A lower bound of exactly 0 is not a PASS.

FRAME LAYOUTS (dates are datetime64[ns] at midnight; CIKs are 10-digit zero-padded str).
  purchases: accession, filing_date, issuer_cik, symbol, owner_cik, value, trade_date
  events:    issuer_cik, symbol, event_date, n_owners, accession
  stock months (sm): symbol, month (Period[M]), month_end, close, r6, r12, r36, fmin12, wipe12; the
  script adds pit_date, cik, ambiguous; assign_long adds group; the short arm adds dilution, burn,
  short_class.

KNOWN BIASES (reported with every result by the script). Survivors only unless delisted prices are
loaded; hold-flat understates bankruptcies; overlapping windows of the same stock; XBRL coverage starts
about 2011 for small filers; no costs, no borrow for the short side.
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from setup_engine import trade_stats as TS

ROOT = Path(__file__).resolve().parents[1]


def _edgar():
    """data_engine/sec_edgar.py loaded BY PATH (stdlib + numpy + pandas only). Importing
    ``data_engine`` would pull in the database layer, yfinance and config.settings. Registered in
    sys.modules BEFORE exec_module (python 3.14 dataclass trap) under the key shared with sec_insider."""
    name = "roboalgo_sec_edgar"
    mod = sys.modules.get(name)
    if mod is not None:
        return mod
    spec = importlib.util.spec_from_file_location(name, ROOT / "data_engine" / "sec_edgar.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return mod


E = _edgar()

# ── pre-registered choices (fixed before any data was seen) ───────────────────────────────────────

ORIGINS = ("2006-01", "2023-09")
TREAT_WINDOW_DAYS = 90            # [month_end - 89, last_weekday) for events; [month_end - 89, month_end] for control exclusion
HORIZONS = (6, 12, 36)
PRIMARY_H = 12
WIPEOUT_WIN = 12
WIPEOUT_LEVEL = 0.5
DILUTION_MIN_RATIO = 1.20
RUNWAY_MAX_YEARS = 1.0
ANNUAL_DAYS = (350, 380)
MAX_STALE_DAYS = 550
SHARES_LAG_DAYS = 365
N_BOOT = 2000
SEED = 0
ALPHA = 0.025
ALPHA_SECONDARY = 0.05
FALLEN_PRICE = (1.0, 10.0)        # = multibagger variant A; a test asserts equality
FALLEN_DD = -0.70
MIN_DVOL = 300_000
PROVISIONAL = "PROVISIONAL: survivors only (delisted names missing)"
PASS = "PASS"
NO_EDGE = "no demonstrated edge"
GROUPS = ("cluster", "single", "control", "excluded_unmapped", "excluded_ambiguous", "excluded_no_coverage")
SHORT_CLASSES = ("flagged", "unflagged", "unknown")

_KEY_SHIFT = 1 << 32              # composite (cik code, day) keys for np.searchsorted
_DAY_OFFSET = 1 << 30
_QUARTER_RE = re.compile(r"^(\d{4})q([1-4])$")


# ── dates ─────────────────────────────────────────────────────────────────────────────────────────

def _dates(x) -> pd.Series:
    """Dates as a datetime64 Series normalized to midnight. A Series keeps its index."""
    if isinstance(x, pd.Series):
        return pd.to_datetime(x).dt.normalize()
    return pd.Series(pd.DatetimeIndex(pd.to_datetime(x))).dt.normalize()


def _days(dates) -> np.ndarray:
    """Whole days since 1970-01-01 as int64 (NaT gives a meaningless number: callers mask it first)."""
    arr = np.asarray(pd.DatetimeIndex(dates).values).astype("datetime64[D]")
    return arr.astype(np.int64)


def last_weekdays(month_end: pd.Series) -> pd.Series:
    """The last weekday on or before each date: Saturday steps back 1 day, Sunday 2. Holidays are ignored."""
    s = _dates(month_end)
    wd = s.dt.weekday.to_numpy(dtype=float)               # NaN for NaT
    back = np.where(wd == 5, 1, np.where(wd == 6, 2, 0))
    return s - pd.Series(pd.to_timedelta(back, unit="D"), index=s.index)


def pit_dates(month_end: pd.Series) -> pd.Series:
    """The day BEFORE the last weekday: filings of that last weekday may have come after the close."""
    return last_weekdays(month_end) - pd.Timedelta(days=1)


# ── prices: fallen universe, forward outcomes, stock-months ───────────────────────────────────────

def fallen_mask(C, DV, dd60, price=FALLEN_PRICE, dd=FALLEN_DD, min_dvol=MIN_DVOL) -> pd.DataFrame:
    """Multibagger variant A: price in [lo, hi] inclusive, dd60 <= dd, dollar volume >= min_dvol. NaN is False."""
    DV = DV.reindex(index=C.index, columns=C.columns)
    dd60 = dd60.reindex(index=C.index, columns=C.columns)
    return (C >= price[0]) & (C <= price[1]) & (dd60 <= dd) & (DV >= min_dvol)


def _check_contiguous(index) -> None:
    if len(index) > 1:
        ok = isinstance(index, pd.PeriodIndex) and bool((np.diff(index.asi8) == 1).all())
        if not ok:
            raise ValueError("forward_outcomes needs a contiguous monthly PeriodIndex (use multibagger wide())")


def forward_outcomes(C, horizons=HORIZONS, wipe_win=WIPEOUT_WIN, wipe_level=WIPEOUT_LEVEL) -> dict:
    """Forward outcomes per origin month i on the (held-flat) grid ``C``.

    r{h} = C[i+h] / C[i] - 1, NaN unless C[i+h] is finite. fmin{w} = min(C[i+1..i+w]) / C[i]: the origin
    month is EXCLUDED, gaps are skipped (np.fmin), and the value is NaN unless C[i+w] is finite.
    wipe{w} is 1.0 when fmin <= wipe_level (inclusive), else 0.0, and NaN where fmin is NaN.
    """
    _check_contiguous(C.index)
    a = C.to_numpy(dtype=float)
    T, N = a.shape
    out = {}

    def frame(x):
        return pd.DataFrame(x, index=C.index, columns=C.columns)

    with np.errstate(invalid="ignore", divide="ignore"):
        for h in horizons:
            r = np.full((T, N), np.nan)
            if T > h:
                r[: T - h] = a[h:] / a[: T - h] - 1.0
            r[~np.isfinite(r)] = np.nan
            out[f"r{h}"] = frame(r)
        lo = np.full((T, N), np.nan)
        for k in range(1, wipe_win + 1):
            if k >= T:
                break
            lo[: T - k] = np.fmin(lo[: T - k], a[k:])
        end = np.full((T, N), np.nan)
        if T > wipe_win:
            end[: T - wipe_win] = a[wipe_win:]
        fmin = lo / a
        fmin[~np.isfinite(end) | ~np.isfinite(fmin)] = np.nan
        wipe = np.where(np.isfinite(fmin), (fmin <= wipe_level).astype(float), np.nan)
    out[f"fmin{wipe_win}"] = frame(fmin)
    out[f"wipe{wipe_win}"] = frame(wipe)
    return out


SM_COLUMNS = ["symbol", "month", "month_end", "close", "r6", "r12", "r36", "fmin12", "wipe12"]


def stock_months(C_raw, DV, dd60, C_held, origins=ORIGINS) -> pd.DataFrame:
    """One row per (symbol, month) that is FALLEN on the raw grid and inside ``origins`` (both inclusive).

    Outcomes come from forward_outcomes(C_held). Rows with NaN outcomes are KEPT (counted at stats time).
    month_end is the calendar month end (midnight).
    """
    lo, hi = pd.Period(origins[0], "M"), pd.Period(origins[1], "M")
    grid = dict(index=C_raw.index, columns=C_raw.columns)
    m = fallen_mask(C_raw, DV, dd60).reindex(**grid).fillna(False).to_numpy(dtype=bool)
    in_range = np.asarray((C_raw.index >= lo) & (C_raw.index <= hi))
    m = m & in_range[:, None]
    oc = {k: v.reindex(**grid).to_numpy(dtype=float) for k, v in forward_outcomes(C_held).items()}
    rows, cols = np.nonzero(m)
    close = C_raw.to_numpy(dtype=float)[rows, cols]
    df = pd.DataFrame({
        "symbol": np.asarray(C_raw.columns, dtype=object)[cols],
        "month": C_raw.index[rows],
        "month_end": C_raw.index.end_time.normalize()[rows],
        "close": close,
        **{k: oc[k][rows, cols] for k in ("r6", "r12", "r36", "fmin12", "wipe12")},
    })
    return df[SM_COLUMNS].reset_index(drop=True)


# ── long arm: treatment assignment ────────────────────────────────────────────────────────────────

class _CikDates:
    """Sorted composite (cik, day) keys, so per-CIK window membership is two np.searchsorted calls."""

    def __init__(self, ciks, dates):
        ciks = np.asarray(ciks, dtype=object)
        dates = pd.DatetimeIndex(pd.to_datetime(dates))
        ok = np.asarray(pd.notna(ciks)) & ~np.asarray(dates.isna())
        ciks, dates = ciks[ok], dates[ok]
        self.code = {c: i for i, c in enumerate(pd.unique(ciks))}
        codes = pd.Series(ciks).map(self.code).to_numpy(dtype=np.int64)
        self.keys = np.sort(codes * _KEY_SHIFT + _days(dates) + _DAY_OFFSET)

    def has(self, ciks, lo, hi, hi_inclusive: bool) -> np.ndarray:
        """True per query row when this CIK has a date in [lo, hi] (hi_inclusive) or [lo, hi)."""
        q = pd.Series(np.asarray(ciks, dtype=object)).map(self.code)
        known = q.notna().to_numpy()
        base = q.fillna(0).to_numpy(dtype=np.int64) * _KEY_SHIFT
        i = np.searchsorted(self.keys, base + _days(lo) + _DAY_OFFSET, side="left")
        j = np.searchsorted(self.keys, base + _days(hi) + _DAY_OFFSET, side="right" if hi_inclusive else "left")
        return known & (j > i)


def _quarter_ordinals(labels) -> np.ndarray:
    out = []
    for lab in labels:
        m = _QUARTER_RE.match(str(lab).strip().lower())
        if not m:
            raise ValueError(f"quarter label {lab!r} is not of the form 'YYYYqN'")
        out.append(int(m.group(1)) * 4 + int(m.group(2)) - 1)
    return np.array(sorted(set(out)), dtype=np.int64)


def _missing_cik(cik: pd.Series) -> np.ndarray:
    return (cik.isna() | cik.map(lambda x: isinstance(x, str) and x.strip() == "")).to_numpy(dtype=bool)


def assign_long(sm, purchases, events, quarters_present, window_days=TREAT_WINDOW_DAYS) -> pd.DataFrame:
    """Copy of ``sm`` plus ``group``. Precedence, highest first:
    ambiguous symbol -> excluded_ambiguous; no CIK -> excluded_unmapped; a quarter of the window absent
    from ``quarters_present`` -> excluded_no_coverage; a cluster EVENT of the same issuer with
    month_end - 89d <= event_date < last_weekday(month_end) -> cluster; else a PURCHASE of the same
    issuer with month_end - 89d <= filing_date <= month_end -> single; else control.
    ``sm`` needs month_end, cik (10-digit str or None) and ambiguous (bool).
    """
    need = [c for c in ("month_end", "cik", "ambiguous") if c not in sm.columns]
    if need:
        raise ValueError(f"assign_long: sm is missing column(s) {need}")
    out = sm.copy()
    if len(out) == 0:
        out["group"] = pd.Series([], dtype=object, index=out.index)
        return out
    me = _dates(out["month_end"]).reset_index(drop=True)
    if me.isna().any():
        raise ValueError("assign_long: month_end must not be missing")
    lw = last_weekdays(me)
    lo = me - pd.Timedelta(days=int(window_days) - 1)
    cik = np.asarray(out["cik"], dtype=object)

    ev = _CikDates(events["issuer_cik"], events["event_date"]) if len(events) else _CikDates([], [])
    pu = _CikDates(purchases["issuer_cik"], purchases["filing_date"]) if len(purchases) else _CikDates([], [])
    is_cluster = ev.has(cik, lo, lw, hi_inclusive=False)
    is_single = pu.has(cik, lo, me, hi_inclusive=True)

    present = _quarter_ordinals(quarters_present)
    q_lo = lo.dt.year.to_numpy(dtype=np.int64) * 4 + lo.dt.quarter.to_numpy(dtype=np.int64) - 1
    q_hi = me.dt.year.to_numpy(dtype=np.int64) * 4 + me.dt.quarter.to_numpy(dtype=np.int64) - 1
    no_cov = np.zeros(len(me), dtype=bool)
    for k in range(int((q_hi - q_lo).max()) + 1):
        q = q_lo + k
        no_cov |= (q <= q_hi) & ~np.isin(q, present)

    group = np.full(len(me), "control", dtype=object)
    group[is_single] = "single"
    group[is_cluster] = "cluster"
    group[no_cov] = "excluded_no_coverage"
    group[_missing_cik(pd.Series(cik))] = "excluded_unmapped"
    group[out["ambiguous"].fillna(False).to_numpy(dtype=bool)] = "excluded_ambiguous"
    out["group"] = group
    return out


# ── long arm: statistics ──────────────────────────────────────────────────────────────────────────

def _control_pool(control, col) -> np.ndarray:
    pool = TS.PoolAccumulator()
    pool.add("c", TS.day_index(control["month_end"]), np.zeros(len(control), dtype=int),
             control[col].to_numpy(dtype=float))
    return pool.arr["c"]


def excess_same_month(treated, control, col) -> np.ndarray:
    """Each treated row's ``col`` minus the mean of ``col`` over control rows with the same month_end.
    NaN where that month has no (finite) control. Empty inputs give an all-NaN array."""
    if len(treated) == 0:
        return np.zeros(0, dtype=float)
    if len(control) == 0:
        return np.full(len(treated), np.nan)
    return TS.excess_vs_pool(treated[col].to_numpy(dtype=float), TS.day_index(treated["month_end"]),
                             _control_pool(control, col), bucket=None)


def _boot_month(treated, ex, n_boot, seed, alpha) -> dict:
    return TS.cluster_bootstrap(ex, TS.month_key(treated["month_end"]), n_boot, seed, alpha)


def long_arm(sm, reused=frozenset(), n_boot=N_BOOT, seed=SEED, alpha=ALPHA) -> dict:
    """Primary: mean 12-month excess of cluster rows over same-month controls, month-clustered bootstrap.
    Everything under ``secondary`` is descriptive (ALPHA_SECONDARY) and never enters a verdict."""
    cl = sm[sm["group"] == "cluster"]
    si = sm[sm["group"] == "single"]
    ct = sm[sm["group"] == "control"]
    ex = excess_same_month(cl, ct, "r12")
    fin = np.isfinite(ex)
    primary = _boot_month(cl, ex, n_boot, seed, alpha)
    primary.update(alpha=alpha, n_stocks=int(pd.Series(cl["symbol"].to_numpy()[fin]).nunique()))
    robust = TS.cluster_bootstrap(ex, cl["symbol"].to_numpy(), n_boot, seed, alpha)

    r12_ok = np.isfinite(cl["r12"].to_numpy(dtype=float))
    ok = fin & r12_ok
    if ok.any() and len(ct):
        control_summary = TS.matched_control_summary(_control_pool(ct, "r12"), TS.day_index(cl["month_end"][ok]))
    else:
        control_summary = TS.matched_control_summary(np.zeros((TS.N_DAYS, TS.N_BUCKETS, len(TS.POOL_STATS))),
                                                     np.zeros(0, dtype=np.int64))
    counts = {"n_cluster": int(len(cl)), "n_cluster_with_r12": int(r12_ok.sum()), "n_matched": int(fin.sum()),
              "n_control": int(len(ct)), "n_single": int(len(si)),
              "n_excluded_unmapped": int((sm["group"] == "excluded_unmapped").sum()),
              "n_excluded_ambiguous": int((sm["group"] == "excluded_ambiguous").sum()),
              "n_excluded_no_coverage": int((sm["group"] == "excluded_no_coverage").sum())}

    a2 = ALPHA_SECONDARY
    cl2, ct2 = cl[~cl["symbol"].isin(set(reused))], ct[~ct["symbol"].isin(set(reused))]
    secondary = {
        "r6": _boot_month(cl, excess_same_month(cl, ct, "r6"), n_boot, seed, a2),
        "r36": _boot_month(cl, excess_same_month(cl, ct, "r36"), n_boot, seed, a2),
        "single_r12": _boot_month(si, excess_same_month(si, ct, "r12"), n_boot, seed, a2),
        "wipe12_cluster_minus_control": TS.cluster_bootstrap_diff(
            cl["wipe12"].to_numpy(dtype=float), TS.month_key(cl["month_end"]),
            ct["wipe12"].to_numpy(dtype=float), TS.month_key(ct["month_end"]), n_boot, seed, a2),
        "no_reused_r12": _boot_month(cl2, excess_same_month(cl2, ct2, "r12"), n_boot, seed, a2),
    }
    return {"primary": primary, "robust_stock_clusters": robust, "control_summary": control_summary,
            "counts": counts, "secondary": secondary}


# ── short arm: XBRL point-in-time flags ───────────────────────────────────────────────────────────

def _finite(x) -> bool:
    return x is not None and bool(np.isfinite(x))


def _fresh(end, read_date, max_stale_days) -> bool:
    """A period end is usable only if it is at most max_stale_days before the date it is read for (C7)."""
    return end is not None and not pd.isna(end) and (pd.Timestamp(read_date) - pd.Timestamp(end)) <= pd.Timedelta(days=int(max_stale_days))


def annual_ocf(usd, d, max_stale_days=MAX_STALE_DAYS):
    """(period end, value) of the latest ANNUAL (350 to 380 day) operating cash flow row filed on or
    before ``d``, restatements filed after ``d`` invisible (E.pit). (None, nan) if there is none or its
    period end is more than ``max_stale_days`` before ``d``. Quarterly rows are never read (C9)."""
    d = pd.Timestamp(d).normalize()
    p = E.pit(usd, d, E.OCF)
    p = p[p["start"].notna()]
    if not p.empty:
        dur = (p["end"] - p["start"]).dt.days
        p = p[(dur >= ANNUAL_DAYS[0]) & (dur <= ANNUAL_DAYS[1])]
    if p.empty:
        return None, float("nan")
    row = p.iloc[-1]                                           # E.pit orders by (end, start)
    if not _fresh(row["end"], d, max_stale_days):
        return None, float("nan")
    return row["end"], float(row["val"])


def dilution_flag(shr, d, lag_days=SHARES_LAG_DAYS, max_stale_days=MAX_STALE_DAYS):
    """True iff shares_now / shares_then >= 1.20 (ratio form, C8); None if not computable.

    Both counts come from ONE concept: dei first, else us-gaap, the first with a usable pair. A pair is
    usable when both counts are finite, ``then`` > 0, and each period end is within ``max_stale_days`` of
    the date it was read for (now at ``d``, then at ``d - lag_days``).
    """
    d = pd.Timestamp(d).normalize()
    d_then = d - pd.Timedelta(days=int(lag_days))
    for concept in (E.SHARES_DEI, E.SHARES_GAAP):
        end_now, now = E.as_of(shr, d, concept)
        end_then, then = E.as_of(shr, d_then, concept)
        if not (_finite(now) and _finite(then)) or then <= 0:
            continue
        if not (_fresh(end_now, d, max_stale_days) and _fresh(end_then, d_then, max_stale_days)):
            continue
        return bool(now / then >= DILUTION_MIN_RATIO)
    return None


def burn_flag(usd, d, max_stale_days=MAX_STALE_DAYS):
    """True iff cash runway is under RUNWAY_MAX_YEARS: cash / |annual OCF| < 1.0. OCF >= 0 gives False
    whatever the cash. None when the OCF or (for a negative OCF) the cash is missing or stale."""
    d = pd.Timestamp(d).normalize()
    end, ocf = annual_ocf(usd, d, max_stale_days)
    if end is None or not _finite(ocf):
        return None
    if ocf >= 0:
        return False
    cash_end, cash = E._cash_total(usd, d)
    if cash_end is None or not _finite(cash) or not _fresh(cash_end, d, max_stale_days):
        return None
    return bool(cash / abs(ocf) < RUNWAY_MAX_YEARS)


def _known(x) -> bool:
    return x is not None and x is not pd.NA and not (isinstance(x, float) and np.isnan(x))


def classify_short(dilution, burn) -> str:
    """flagged = both known and both True; unflagged = both known otherwise; unknown = anything missing."""
    if not (_known(dilution) and _known(burn)):
        return "unknown"
    return "flagged" if (bool(dilution) and bool(burn)) else "unflagged"


def short_flags(facts, month_ends) -> pd.DataFrame:
    """Per month end: pit_date, dilution, burn, short_class, in input order (a Series input keeps its index).
    The facts table is built ONCE per call. ``facts`` None gives "unknown" for every row."""
    me = _dates(month_ends)
    pit = pit_dates(me)
    dil, brn = [None] * len(me), [None] * len(me)
    if facts is not None:
        tbl = E.facts_table(facts, concepts={E.OCF, E.CASH, *E.CASH_ADD, E.SHARES_DEI, E.SHARES_GAAP})
        usd = tbl[tbl["unit"] == "USD"]
        shr = tbl[tbl["unit"] == "shares"]
        for i, d in enumerate(pit):
            if pd.isna(d):
                continue
            dil[i] = dilution_flag(shr, d)
            brn[i] = burn_flag(usd, d)
    return pd.DataFrame({"month_end": me, "pit_date": pit,
                         "dilution": pd.Series(dil, index=me.index, dtype=object),
                         "burn": pd.Series(brn, index=me.index, dtype=object),
                         "short_class": [classify_short(a, b) for a, b in zip(dil, brn)]},
                        index=me.index)


def short_arm(sm, n_boot=N_BOOT, seed=SEED, alpha=ALPHA) -> dict:
    """Wipeout rate of flagged minus unflagged stock-months (unknown rows excluded), month-clustered
    bootstrap of the difference. Needs month_end, symbol, short_class, wipe12."""
    cls = sm["short_class"]
    classified = cls.isin(["flagged", "unflagged"])
    has_out = np.isfinite(sm["wipe12"].to_numpy(dtype=float))
    a = sm[(cls == "flagged") & has_out]
    b = sm[(cls == "unflagged") & has_out]
    xa, xb = a["wipe12"].to_numpy(dtype=float), b["wipe12"].to_numpy(dtype=float)
    primary = TS.cluster_bootstrap_diff(xa, TS.month_key(a["month_end"]), xb, TS.month_key(b["month_end"]),
                                        n_boot, seed, alpha)
    primary["alpha"] = alpha
    robust = TS.cluster_bootstrap_diff(xa, a["symbol"].to_numpy(), xb, b["symbol"].to_numpy(),
                                       n_boot, seed, alpha)
    me_class = _dates(sm.loc[classified, "month_end"])
    years = me_class.dt.year.value_counts().sort_index()
    return {"primary": primary, "robust_stock_clusters": robust,
            "rates": {"flagged": float(xa.mean()) if len(xa) else None,
                      "unflagged": float(xb.mean()) if len(xb) else None},
            "counts": {"n_flagged": int(len(a)), "n_unflagged": int(len(b)),
                       "n_unknown": int((cls == "unknown").sum()),
                       "n_no_outcome": int((classified.to_numpy() & ~has_out).sum())},
            "coverage_by_year": {int(y): int(n) for y, n in years.items()},
            "first_classified_month": me_class.min().strftime("%Y-%m") if len(me_class) else None}


# ── verdict ───────────────────────────────────────────────────────────────────────────────────────

def verdict(lo, *, provisional: bool, not_run_reason=None) -> str:
    """Mechanical verdict from the lower bound ``lo``: PASS only if lo > 0 (exactly 0 is not a PASS).
    PROVISIONAL text comes first when the price set has no delisted names."""
    if not_run_reason:
        core = f"not run: {not_run_reason}"
    elif lo is None or not np.isfinite(lo):
        core = f"{NO_EDGE} (interval not computable)"
    elif lo > 0:
        core = PASS
    else:
        core = NO_EDGE
    return f"{PROVISIONAL} | {core}" if provisional else core
