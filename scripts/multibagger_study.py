"""
Which stocks became 50x multibaggers within ten years, what did they look like at the start, and does a "fallen but turning" profile pick them out?

    python scripts/multibagger_study.py                                 # full study, writes docs/multibagger_study_results.json
    python scripts/multibagger_study.py --limit 2 --workers 4           # smoke test on two batch files (writes under ~/.roboalgo/multibagger)
    python scripts/multibagger_study.py --fundamentals                  # also run the (network) fundamentals check on today's pool

THE QUESTION. For every stock-month in the price cache, what is the highest MONTH-END close reached in the
next 120 months, as a multiple of today's close (``fmax``), and the close 120 months later (``fend``)?
Part 1 counts how often 5x, 10x, 20x, 50x and 100x happen (December starts, 2001-2015) and profiles the
50x winners at their best start. Part 2 asks how well simple profiles ("price under $5", "70% below the
60-month high", ...) pick the winners out (precision, recall, lift). Part 3 follows four "fallen stock"
variants month by month (A: $1-$10 and 70% or more below the 60-month high; B: A and above its 10-month
average; C: B and up 20% in 3 months; control: A and BELOW its 10-month average) and reports how often
they reach 10x/20x/50x, fall another 50%, or end below the start. Part 4 lists today's pool of such
stocks. The pool is a list of candidates with NO demonstrated edge, not a recommendation.

PRE-REGISTRATION. Fixed before any result was seen (they are the module constants and are copied into the
JSON): horizon H = 120 months; December starts 2001-2015 for the base rates and the precision table; every
month 2001-01 to 2015-12 for the variants; eligibility = month-end close >= $1.00 and mean daily dollar
volume of the month >= $300,000; peak multiples 5/10/20/50/100, end multiples 10/50, winner = 50x; the
60-month high includes the current month; fallen = close in [$1, $10] and >= 70% below it; trend = close
above the 10-month average; turn = 3-month return >= 20%; "fell another 50%" = the lowest close of the next 36
months <= 0.5x the start; the mean end multiple is winsorised at its 99th percentile (upper tail only);
today's pool = the same fallen rule with dollar volume >= $1,000,000 in the last complete month; the pool
fundamentals check = market cap >= $300M, revenue growth >= 10%, gross margin >= 25%, net cash >= 0 or
positive free cash flow, close above the 10-month average. Nothing is tuned.

KNOWN BIASES (not fixable here, reported with every result). Survivors only unless terminal.json is
present in the history directory: stocks that were delisted are missing from the cache, so every rate is
an upper bound, and for the "fallen" profiles (the stocks that died are exactly the ones missing) the
bias is the worst possible. Stocks listed in terminal.json are held FLAT at their last month-end close
through the end of the grid: optimistic for bankruptcies, which really went to about zero, and roughly
right for cash takeovers. Monthly starts of the same stock overlap, so n counts stock-months and
``n_stocks`` is the honest sample size. Closes are split-adjusted by Yahoo (retroactively, so a reverse-split
stock can show a price it never traded at) and not dividend-adjusted. Month-end closes miss intra-month
highs, so ``fmax`` understates the intra-month peak, but nobody sells at the monthly top, so it is an upper
bound on what was capturable. The last (partial) month of the data is dropped. No rate here is a probability.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import multiprocessing as mp
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from setup_engine import mtf

# ── pre-registered choices (fixed before any result is seen) ──────────────────────────────────────

DEFAULT_HISTORY = Path.home() / ".roboalgo" / "history_2000"
OUT_JSON = ROOT / "docs" / "multibagger_study_results.json"
POOL_DIR = Path.home() / ".roboalgo" / "multibagger"

H = 120                            # forward horizon in calendar months
DEC_START_YEARS = (2001, 2015)     # December starts for the base rates, winners and precision table
MONTHLY_START = ("2001-01", "2015-12")   # every month in this range for the fallen variants
MIN_PRICE = 1.00                   # eligibility: month-end close at least this
MIN_DVOL = 300_000                 # eligibility: mean daily dollar volume of the month at least this
PEAK_MULTS = (5, 10, 20, 50, 100)  # share of starts whose highest forward close reaches this multiple
END_MULTS = (10, 50)               # share of starts whose close H months later is at least this multiple
WINNER = 50                        # a "multibagger" for the winners list and the precision table
HI_WIN = 60                        # months in the trailing high (includes the current month)
HI_MIN_PERIODS = 12                # months of history needed before a trailing high exists
FALLEN_DD = -0.70                  # fallen: this far below the trailing high or worse
FALLEN_PRICE = (1.0, 10.0)         # fallen: close inside this range (inclusive)
SMA_M = 10                         # months in the trend average
RET3_MIN = 0.20                    # variant C: 3-month return at least this
RET_LAGS = {"ret3": 3, "ret12": 12, "ret36": 36}
VOL_WIN = 12                       # months of log returns in the volatility feature
FELL_WIN = 36                      # months ahead in which "fell another 50%" is looked for
FELL_LEVEL = 0.5                   # ... the lowest close must be at most this multiple of the start
WINSOR_Q = 0.99                    # the mean end multiple is clipped at this quantile (upper tail only)
POOL_MIN_DVOL = 1_000_000          # today's pool: dollar volume at least this
FUND = {"min_cap": 3e8, "min_rev_growth": 0.10, "min_gross_margin": 0.25}
SMOKE_NAME = "smoke.json"          # default --out under POOL_DIR for a --limit run

# Precision table: ordered name -> rule on the long start table S (columns close, dvol, dd60, ret12, ...).
PROFILES = {
    "price < $5": lambda S: S["close"] < 5,
    "price < $5 and >=70% below 60m high": lambda S: (S["close"] < 5) & (S["dd60"] <= -0.70),
    "price < $10 and >=80% below 60m high": lambda S: (S["close"] < 10) & (S["dd60"] <= -0.80),
    "up >=100% in 12 months": lambda S: S["ret12"] >= 1.0,
    "dollar volume < $2M/day": lambda S: S["dvol"] < 2e6,
}

SURVIVORS_ONLY = ("SURVIVORS ONLY: stocks listed today; dead stocks are missing, so every rate is an upper bound")
POOL_NOTE = ("Candidates only: a list of stocks that fit the fallen profile today, with no demonstrated edge "
             "(see fallen_variants and the survivorship note). Not a recommendation.")
PREREGISTRATION = (
    f"Fixed before any result was seen. Horizon {H} months; December starts {DEC_START_YEARS[0]}-{DEC_START_YEARS[1]} for "
    f"base rates and precision, every month {MONTHLY_START[0]} to {MONTHLY_START[1]} for the variants; eligible = month-end close "
    f">= ${MIN_PRICE:.2f} and mean daily dollar volume >= ${MIN_DVOL:,}; peak multiples {list(PEAK_MULTS)}, end multiples "
    f"{list(END_MULTS)}, winner = {WINNER}x; fallen = close ${FALLEN_PRICE[0]:.0f}-${FALLEN_PRICE[1]:.0f} and >= {abs(FALLEN_DD):.0%} below the "
    f"{HI_WIN}-month high; trend = close above the {SMA_M}-month average; turn = 3-month return >= {RET3_MIN:.0%}; fell another "
    f"50% = lowest close of the next {FELL_WIN} months <= {FELL_LEVEL}x; mean end multiple winsorised at the {WINSOR_Q:.0%} quantile "
    f"(upper only); today's pool needs dollar volume >= ${POOL_MIN_DVOL:,}; fundamentals {FUND}. Nothing is tuned.")

PANEL_COLUMNS = ["symbol", "month", "close", "high", "low", "dvol", "n_days"]
START_FEATURES = ("dd60", "ret12", "ret36", "vol12", "sma10", "ret3")
START_FORWARD = ("fmax", "fend", "fmin36")
START_COLUMNS = ["symbol", "month", "close", "dvol", *START_FEATURES, *START_FORWARD]
PROFILE_FIELDS = ("close", "dvol", "dd60", "ret12", "ret36", "vol12")
VARIANTS = ("A", "B", "C", "control")
POOL_COLUMNS = ["symbol", "close", "dvol", "dd60", "above_sma10", "ret3"]
FUND_INPUTS = ("marketCap", "revenueGrowth", "grossMargins", "totalCash", "totalDebt", "freeCashflow")
FUND_CHECKS = ("pass_cap", "pass_growth", "pass_margin", "pass_cash", "pass_trend")
BATCH_COLUMNS = ["symbol", "date", "open", "high", "low", "close", "volume"]


# ── monthly panel ─────────────────────────────────────────────────────────────────────────────────

def monthly_from_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Daily bars (symbol, date, open, high, low, close, volume) -> one row per (symbol, month).

    Rows with a missing or impossible bar are dropped first. ``close`` is the close of the LAST trading day
    of the month, ``dvol`` the mean of close x volume over the month's days.
    """
    d = df[BATCH_COLUMNS].copy()
    d["date"] = pd.to_datetime(d["date"])
    ok = (d[["open", "high", "low", "close"]].notna().all(axis=1)
          & (d["close"] > 0) & (d["open"] > 0) & (d["low"] > 0) & (d["high"] >= d["low"]))
    d = d[ok].copy()
    if d.empty:
        return pd.DataFrame({"symbol": pd.Series(dtype=object), "month": pd.PeriodIndex([], freq="M"),
                             "close": pd.Series(dtype=float), "high": pd.Series(dtype=float),
                             "low": pd.Series(dtype=float), "dvol": pd.Series(dtype=float),
                             "n_days": pd.Series(dtype="int64")})[PANEL_COLUMNS]
    d["volume"] = d["volume"].fillna(0.0)
    d = d.sort_values(["symbol", "date"], kind="mergesort").drop_duplicates(["symbol", "date"], keep="last")
    d["month"] = d["date"].dt.to_period("M")
    d["dv"] = d["close"] * d["volume"]
    out = (d.groupby(["symbol", "month"], sort=True)
            .agg(close=("close", "last"), high=("high", "max"), low=("low", "min"),
                 dvol=("dv", "mean"), n_days=("close", "size"))
            .reset_index())
    return out[PANEL_COLUMNS]


def _panel_batch(path: str) -> pd.DataFrame:
    """Read one batch parquet and reduce it to monthly rows."""
    df = pd.read_parquet(path, columns=BATCH_COLUMNS)
    dates = pd.to_datetime(df["date"])
    if dates.dt.tz is not None:
        dates = dates.dt.tz_localize(None)
    df["date"] = dates
    return monthly_from_daily(df)


def _batch_files(history, limit=None) -> list:
    files = sorted(Path(history).expanduser().glob("batch_*.parquet"))
    return files[:limit] if limit is not None else files


def build_panel(history, workers: int = 1, limit: int | None = None) -> pd.DataFrame:
    """Monthly panel over the first ``limit`` sorted batch files of ``history`` (a fork pool when workers > 1)."""
    paths = [str(f) for f in _batch_files(history, limit)]
    if not paths:
        raise FileNotFoundError(f"no batch_*.parquet files in {history}")
    if workers > 1 and len(paths) > 1:
        with mp.get_context("fork").Pool(min(workers, len(paths))) as pool:
            parts = pool.map(_panel_batch, paths)
    else:
        parts = [_panel_batch(p) for p in paths]
    panel = pd.concat(parts, ignore_index=True)
    panel = panel.drop_duplicates(["symbol", "month"], keep="last")
    return panel.sort_values(["symbol", "month"]).reset_index(drop=True)


def _data_end(files) -> pd.Timestamp:
    """The latest date in the given batch files (only the date column is read)."""
    return max(pd.Timestamp(pd.read_parquet(f, columns=["date"])["date"].max()) for f in files)


def load_terminal(history) -> frozenset:
    """Symbols listed in history/terminal.json (delisted stocks added by ingest_delisted_history.py), else empty."""
    path = Path(history).expanduser() / "terminal.json"
    if not path.exists():
        return frozenset()
    return frozenset(json.loads(path.read_text()))


def wide(panel: pd.DataFrame, col: str) -> pd.DataFrame:
    """Panel column as a month x symbol grid. The month index is CONTIGUOUS, so i + H is H calendar months."""
    months = pd.period_range(panel["month"].min(), panel["month"].max(), freq="M")
    return panel.pivot(index="month", columns="symbol", values=col).reindex(months)


def hold_terminal_flat(C: pd.DataFrame, terminal) -> pd.DataFrame:
    """Hold each terminal symbol's last close flat from its last valid month to the end of the grid (gaps untouched)."""
    out = C.copy()
    for sym in terminal:
        if sym not in out.columns:
            continue
        s = out[sym]
        last = s.last_valid_index()
        if last is None:
            continue
        out.loc[out.index > last, sym] = s.loc[last]
    return out


# ── features and forward outcomes ─────────────────────────────────────────────────────────────────

def features(C: pd.DataFrame, DV: pd.DataFrame) -> dict:
    """Trailing features at each month, all aligned to ``C`` and using only data up to and including that month."""
    out = {"dd60": C / C.rolling(HI_WIN, min_periods=HI_MIN_PERIODS).max() - 1}
    for name, lag in RET_LAGS.items():
        out[name] = C / C.shift(lag) - 1
    out["vol12"] = np.log(C).diff().rolling(VOL_WIN, min_periods=VOL_WIN).std(ddof=1) * np.sqrt(12)
    out["sma10"] = C.rolling(SMA_M, min_periods=SMA_M).mean()
    out["dvol"] = DV
    return out


def forward(C: pd.DataFrame, H: int = H) -> dict:
    """Forward outcomes per start month i: fmax = max(C[i+1..i+H]) / C[i], fend = C[i+H] / C[i], fmin36 = min(C[i+1..i+FELL_WIN]) / C[i].

    The start month is EXCLUDED and month i+H INCLUDED. A window that runs past the end of the grid gives NaN.
    """
    a = C.to_numpy(dtype=float)
    T, N = a.shape
    hi = np.full((T, N), np.nan)
    lo = np.full((T, N), np.nan)
    for k in range(1, H + 1):
        if k >= T:
            break
        hi[: T - k] = np.fmax(hi[: T - k], a[k:])
    for k in range(1, FELL_WIN + 1):
        if k >= T:
            break
        lo[: T - k] = np.fmin(lo[: T - k], a[k:])
    with np.errstate(invalid="ignore", divide="ignore"):
        fmax = hi / a
        fmin = lo / a
        fend = np.full((T, N), np.nan)
        if T > H:
            fend[: T - H] = a[H:] / a[: T - H]
    fmax[max(T - H, 0):] = np.nan
    fmin[max(T - FELL_WIN, 0):] = np.nan
    mk = lambda x: pd.DataFrame(x, index=C.index, columns=C.columns)  # noqa: E731
    return {"fmax": mk(fmax), "fend": mk(fend), "fmin36": mk(fmin)}


def starts(C: pd.DataFrame, DV: pd.DataFrame, months, F: dict) -> pd.DataFrame:
    """Long table of eligible (symbol, start month) rows with their features and forward outcomes.

    ``F`` is ``{**features(C, DV), **forward(C)}``. Eligible = close >= MIN_PRICE and dvol >= MIN_DVOL and a
    finite ``fend``. The count of rows dropped ONLY because ``fend`` is not finite goes to ``attrs["n_incomplete"]``.
    """
    rows = C.index.get_indexer(pd.PeriodIndex(list(months), freq="M"))
    rows = rows[rows >= 0]
    c = C.to_numpy(dtype=float)[rows]
    dv = DV.to_numpy(dtype=float)[rows]
    fend = F["fend"].to_numpy(dtype=float)[rows]
    eligible = (c >= MIN_PRICE) & (dv >= MIN_DVOL)
    finite = np.isfinite(fend)
    n_incomplete = int((eligible & ~finite).sum())
    r, s = np.nonzero(eligible & finite)
    out = {"symbol": np.asarray(C.columns)[s], "month": C.index[rows][r], "close": c[r, s], "dvol": dv[r, s]}
    for name in (*START_FEATURES, *START_FORWARD):
        out[name] = F[name].to_numpy(dtype=float)[rows][r, s]
    S = pd.DataFrame(out)[START_COLUMNS]
    S.attrs["n_incomplete"] = n_incomplete
    return S


# ── analysis ──────────────────────────────────────────────────────────────────────────────────────

def _share(mask: np.ndarray, n: int):
    return float(np.sum(mask)) / n if n else None


def _rate_block(S: pd.DataFrame) -> dict:
    n = len(S)
    out = {"n": n, "n_stocks": int(S["symbol"].nunique())}
    for m in PEAK_MULTS:
        out[f"share_fmax_ge_{m}x"] = _share(S["fmax"].to_numpy() >= m, n)
    for m in END_MULTS:
        out[f"share_fend_ge_{m}x"] = _share(S["fend"].to_numpy() >= m, n)
    return out


def base_rates(S: pd.DataFrame) -> dict:
    """Share of starts whose peak/end multiple reaches each threshold, pooled and by start year."""
    years = S["month"].dt.year if len(S) else pd.Series([], dtype=int)
    by_year = {}
    for y in range(DEC_START_YEARS[0], DEC_START_YEARS[1] + 1):
        by_year[str(y)] = _rate_block(S[(years == y).to_numpy()])
    return {"pooled": _rate_block(S), "by_year": by_year}


def winners(S: pd.DataFrame, mult: float = WINNER) -> list:
    """Distinct symbols whose fmax reaches ``mult`` from at least one start, each with its best start."""
    w = S[S["fmax"] >= mult]
    if w.empty:
        return []
    best = w.loc[w.groupby("symbol")["fmax"].idxmax()]
    n_starts = w.groupby("symbol").size()
    out = [{"symbol": r.symbol, "start": str(r.month), "multiple": float(r.fmax), "close": float(r.close),
            "dvol": float(r.dvol), "dd60": float(r.dd60), "n_starts": int(n_starts[r.symbol])}
           for r in best.itertuples()]
    return sorted(out, key=lambda d: (-d["multiple"], d["symbol"]))


def profile(S: pd.DataFrame, mask) -> dict:
    """Medians of the start characteristics over the rows selected by ``mask``."""
    sel = S[np.asarray(mask, dtype=bool)]
    out = {"n": len(sel), "n_stocks": int(sel["symbol"].nunique())}
    for f in PROFILE_FIELDS:
        out[f"median_{f}"] = float(sel[f].median()) if len(sel) else None
    return out


def precision(S: pd.DataFrame) -> list:
    """For each profile: how many starts it selects, how many of them are 50x, and the 50x/10x hit rates and lift."""
    n_all = len(S)
    win = (S["fmax"] >= WINNER).to_numpy()
    total_win = int(win.sum())
    pooled = total_win / n_all if n_all else None
    out = []
    for name, rule in PROFILES.items():
        m = np.asarray(rule(S), dtype=bool)
        n = int(m.sum())
        n_win = int((m & win).sum())
        p50 = n_win / n if n else None
        out.append({"profile": name, "n": n, "n_50x": n_win, "precision_50x": p50,
                    "recall_50x": n_win / total_win if total_win else None,
                    "precision_10x": _share((S["fmax"].to_numpy() >= 10) & m, n) if n else None,
                    "lift_50x": p50 / pooled if (p50 is not None and pooled) else None})
    return out


def _winsor_mean(x: np.ndarray, q: float = WINSOR_Q) -> float | None:
    """Mean of ``x`` with the upper tail clipped at its ``q`` quantile (the lower tail is left alone)."""
    if len(x) == 0:
        return None
    return float(np.minimum(x, np.quantile(x, q)).mean())


def fallen_variants(C: pd.DataFrame, DV: pd.DataFrame, F: dict) -> dict:
    """Outcomes of the four fallen-stock variants over every month in MONTHLY_START with a full forward window."""
    lo, hi = pd.Period(MONTHLY_START[0], "M"), pd.Period(MONTHLY_START[1], "M")
    sel = np.nonzero((C.index >= lo) & (C.index <= hi))[0]

    def arr(X):
        return X.to_numpy(dtype=float)[sel]

    c, dv = arr(C), arr(DV)
    dd, sma, ret3 = arr(F["dd60"]), arr(F["sma10"]), arr(F["ret3"])
    fmax, fend, fmin = arr(F["fmax"]), arr(F["fend"]), arr(F["fmin36"])
    A = ((c >= FALLEN_PRICE[0]) & (c <= FALLEN_PRICE[1]) & (dd <= FALLEN_DD) & (dv >= MIN_DVOL) & np.isfinite(fend))
    B = A & (c > sma)
    masks = {"A": A, "B": B, "C": B & (ret3 >= RET3_MIN), "control": A & (c < sma)}
    out = {}
    for name in VARIANTS:
        m = masks[name]
        n = int(m.sum())
        e = fend[m]
        out[name] = {
            "n": n, "n_stocks": int(np.unique(np.nonzero(m)[1]).size),
            "share_10x": _share(fmax[m] >= 10, n), "share_20x": _share(fmax[m] >= 20, n),
            "share_50x": _share(fmax[m] >= 50, n),
            "share_fell50": _share(fmin[m] <= FELL_LEVEL, n), "share_end_below": _share(e < 1, n),
            "mean_end_winsor": _winsor_mean(e), "median_end": float(np.median(e)) if n else None}
    return out


def last_complete_month(max_date) -> pd.Period:
    """The month of ``max_date`` if the data reaches that month's last weekday, else the previous month."""
    d = pd.Timestamp(max_date).normalize()
    if d >= mtf.month_label(d)[0]:
        return d.to_period("M")
    return d.to_period("M") - 1


def today_pool(C: pd.DataFrame, DV: pd.DataFrame, F: dict, month, terminal) -> pd.DataFrame:
    """Listed (non-terminal) stocks that fit the fallen profile at ``month`` and trade at least POOL_MIN_DVOL a day."""
    month = pd.Period(month, "M")
    if month not in C.index:
        return pd.DataFrame(columns=POOL_COLUMNS)
    i = C.index.get_loc(month)
    c, dv = C.iloc[i], DV.iloc[i]
    dd, sma, r3 = F["dd60"].iloc[i], F["sma10"].iloc[i], F["ret3"].iloc[i]
    m = ((c >= FALLEN_PRICE[0]) & (c <= FALLEN_PRICE[1]) & (dv >= POOL_MIN_DVOL) & (dd <= FALLEN_DD)
         & ~c.index.isin(list(terminal)))
    out = pd.DataFrame({"symbol": c.index[m], "close": c[m].to_numpy(), "dvol": dv[m].to_numpy(),
                        "dd60": dd[m].to_numpy(), "above_sma10": (c[m] > sma[m]).to_numpy(), "ret3": r3[m].to_numpy()})
    return out.sort_values(["dvol", "symbol"], ascending=[False, True]).reset_index(drop=True)[POOL_COLUMNS]


def _yf_info(symbol: str) -> dict:
    import yfinance as yf
    return yf.Ticker(symbol).info


def _num(x) -> float:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return f if math.isfinite(f) else float("nan")


def fundamentals_check(pool: pd.DataFrame, fetch=None) -> pd.DataFrame:
    """Five pass/fail checks per pool symbol from ``fetch(symbol) -> info dict`` (yfinance by default).

    A missing field fails its check. Never raises on a bad fetch (that row simply fails every fundamentals check).
    """
    fetch = fetch or _yf_info
    rows = []
    for sym, above in zip(pool["symbol"], pool["above_sma10"]):
        try:
            info = fetch(sym) or {}
        except Exception:
            info = {}
        v = {k: _num(info.get(k)) for k in FUND_INPUTS}
        net_cash_ok = bool(np.isfinite(v["totalCash"]) and np.isfinite(v["totalDebt"]) and v["totalCash"] - v["totalDebt"] >= 0)
        fcf_ok = bool(np.isfinite(v["freeCashflow"]) and v["freeCashflow"] > 0)
        checks = {"pass_cap": bool(v["marketCap"] >= FUND["min_cap"]),
                  "pass_growth": bool(v["revenueGrowth"] >= FUND["min_rev_growth"]),
                  "pass_margin": bool(v["grossMargins"] >= FUND["min_gross_margin"]),
                  "pass_cash": net_cash_ok or fcf_ok, "pass_trend": bool(above)}
        rows.append({"symbol": sym, **v, **checks, "n_pass": int(sum(checks.values()))})
    cols = ["symbol", *FUND_INPUTS, *FUND_CHECKS, "n_pass"]
    return pd.DataFrame(rows, columns=cols)


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


def _pct(x, nd=2) -> str:
    return "n/a" if x is None else f"{100 * x:.{nd}f}%"


def _x(x, nd=2) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}x"


def format_base_rates(br: dict) -> str:
    cols = [f"fmax>={m}x" for m in PEAK_MULTS] + [f"fend>={m}x" for m in END_MULTS]
    head = f"{'start':<8} {'n':>6} {'stocks':>6} " + " ".join(f"{c:>10}" for c in cols)
    lines = [head, "-" * len(head)]
    blocks = [(y, b) for y, b in br["by_year"].items()] + [("pooled", br["pooled"])]
    for label, b in blocks:
        vals = [b[f"share_fmax_ge_{m}x"] for m in PEAK_MULTS] + [b[f"share_fend_ge_{m}x"] for m in END_MULTS]
        lines.append(f"{label:<8} {b['n']:>6} {b['n_stocks']:>6} " + " ".join(f"{_pct(v):>10}" for v in vals))
    return "\n".join(lines)


def format_precision(rows: list) -> str:
    head = f"{'profile':<40} {'n':>7} {'n_50x':>6} {'prec 50x':>9} {'recall':>8} {'prec 10x':>9} {'lift':>8}"
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(f"{r['profile']:<40} {r['n']:>7} {r['n_50x']:>6} {_pct(r['precision_50x']):>9} "
                     f"{_pct(r['recall_50x'], 1):>8} {_pct(r['precision_10x']):>9} {_x(r['lift_50x'], 1):>8}")
    return "\n".join(lines)


def format_variants(v: dict) -> str:
    head = (f"{'variant':<8} {'n':>7} {'stocks':>6} {'10x':>8} {'20x':>8} {'50x':>8} {'fell50':>8} "
            f"{'end<1':>8} {'mean end*':>10} {'median end':>11}")
    lines = [head, "-" * len(head)]
    for name in VARIANTS:
        r = v[name]
        lines.append(f"{name:<8} {r['n']:>7} {r['n_stocks']:>6} {_pct(r['share_10x']):>8} {_pct(r['share_20x']):>8} "
                     f"{_pct(r['share_50x']):>8} {_pct(r['share_fell50']):>8} {_pct(r['share_end_below']):>8} "
                     f"{_x(r['mean_end_winsor']):>10} {_x(r['median_end']):>11}")
    lines.append("* mean end multiple winsorised at the 99th percentile (upper tail only)")
    return "\n".join(lines)


def _constants() -> dict:
    return {"H": H, "DEC_START_YEARS": DEC_START_YEARS, "MONTHLY_START": MONTHLY_START, "MIN_PRICE": MIN_PRICE,
            "MIN_DVOL": MIN_DVOL, "PEAK_MULTS": PEAK_MULTS, "END_MULTS": END_MULTS, "WINNER": WINNER,
            "HI_WIN": HI_WIN, "HI_MIN_PERIODS": HI_MIN_PERIODS, "FALLEN_DD": FALLEN_DD, "FALLEN_PRICE": FALLEN_PRICE,
            "SMA_M": SMA_M, "RET3_MIN": RET3_MIN, "RET_LAGS": RET_LAGS, "VOL_WIN": VOL_WIN, "FELL_WIN": FELL_WIN,
            "FELL_LEVEL": FELL_LEVEL, "WINSOR_Q": WINSOR_Q, "POOL_MIN_DVOL": POOL_MIN_DVOL, "FUND": FUND,
            "PROFILES": list(PROFILES)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--history", default=str(DEFAULT_HISTORY))
    ap.add_argument("--limit", type=int, default=None, help="smoke test: only the first N batch files")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--out", default=None, help=f"results JSON (default {OUT_JSON}, or {POOL_DIR / SMOKE_NAME} with --limit)")
    ap.add_argument("--pool-out", default=None, help=f"pool CSV (default {POOL_DIR}/pool_<month>.csv)")
    ap.add_argument("--fundamentals", action="store_true", help="also run the fundamentals check on the pool (network)")
    args = ap.parse_args(argv)

    limited = args.limit is not None
    history = Path(args.history).expanduser()
    files = _batch_files(history, args.limit)
    if not files:
        raise SystemExit(f"no batch_*.parquet files in {history}")
    out_path = Path(args.out).expanduser() if args.out else (POOL_DIR / SMOKE_NAME if limited else OUT_JSON)

    data_end = _data_end(files)
    month_lc = last_complete_month(data_end)
    panel = build_panel(history, workers=args.workers, limit=args.limit)
    panel = panel[panel["month"] <= month_lc]           # the last, partial month is not a month-end close
    terminal = frozenset(load_terminal(history) & set(panel["symbol"]))
    survivorship = (f"survivorship-corrected ({len(terminal)} delisted stocks held flat)" if terminal else SURVIVORS_ONLY)
    print(survivorship, flush=True)

    C = hold_terminal_flat(wide(panel, "close"), terminal)
    DV = wide(panel, "dvol")
    F = {**features(C, DV), **forward(C)}
    dec_months = [pd.Period(f"{y}-12", "M") for y in range(DEC_START_YEARS[0], DEC_START_YEARS[1] + 1)]
    S = starts(C, DV, dec_months, F)
    n_incomplete = S.attrs["n_incomplete"]

    br = base_rates(S)
    win = winners(S)
    prof = {"winners": profile(S, (S["fmax"] >= WINNER).to_numpy()), "all": profile(S, np.ones(len(S), dtype=bool))}
    prec = precision(S)
    var = fallen_variants(C, DV, F)
    pool = today_pool(C, DV, F, month_lc, terminal)
    fund = fundamentals_check(pool) if args.fundamentals else None

    pool_path = Path(args.pool_out).expanduser() if args.pool_out else (
        POOL_DIR / f"pool_{month_lc}{'_smoke' if limited else ''}.csv")
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    (pool if fund is None else pool.merge(fund, on="symbol", how="left")).to_csv(pool_path, index=False)

    result = {
        "preregistration": PREREGISTRATION,
        "constants": _constants(),
        "data": {"history": str(history), "n_symbols": int(panel["symbol"].nunique()), "n_terminal": len(terminal),
                 "survivorship": survivorship, "grid": [str(C.index[0]), str(C.index[-1])],
                 "last_complete_month": str(month_lc), "n_incomplete": n_incomplete, "limited_run": limited},
        "base_rates": br, "winners_50x": win, "profile": prof, "precision": prec, "fallen_variants": var,
        "pool": {"month": str(month_lc), "note": POOL_NOTE, "n": len(pool), "candidates": pool,
                 "fundamentals": fund}}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_jsonable(result), indent=1, allow_nan=False))

    print(f"data: {history} | {result['data']['n_symbols']} symbols | grid {result['data']['grid'][0]} to "
          f"{result['data']['grid'][1]} | last complete month {month_lc} | {n_incomplete} eligible starts dropped "
          f"for an incomplete forward window", flush=True)
    print(f"\nBASE RATES (December starts {DEC_START_YEARS[0]}-{DEC_START_YEARS[1]}, {H} months ahead)\n{format_base_rates(br)}")
    print(f"\n{WINNER}x WINNERS: {len(win)} stocks; best starts: "
          + ", ".join(f"{w['symbol']} {w['start']} ({w['multiple']:.0f}x)" for w in win[:10]))
    print(f"\nPRECISION OF SIMPLE PROFILES (starts that reach {WINNER}x)\n{format_precision(prec)}")
    print(f"\nFALLEN VARIANTS (every month {MONTHLY_START[0]} to {MONTHLY_START[1]})\n{format_variants(var)}")
    print(f"\nTODAY'S POOL ({month_lc}): {len(pool)} candidates, no demonstrated edge"
          + (f"; {int((fund['n_pass'] == len(FUND_CHECKS)).sum())} pass all fundamentals checks" if fund is not None else ""))
    if limited:
        print("(limited run: a smoke test of the machinery, NOT the study)")
    print(f"wrote {out_path} and {pool_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
