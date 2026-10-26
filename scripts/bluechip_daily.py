"""
Which S&P 500 stocks look like leaders right now? A screen of today's index on the last complete day.

    python scripts/bluechip_daily.py                                  # download prices, report in ~/.roboalgo/reports
    python scripts/bluechip_daily.py --limit 25 --no-fundamentals --out-dir /tmp/bc      # quick live smoke test
    python scripts/bluechip_daily.py --no-network --prices-from DIR --universe sp500.csv --today 2026-10-12

WHAT IT DOES. It takes the current S&P 500 list (setup_engine.universe), the last ~4 years of daily bars
(yfinance, split-adjusted, not dividend-adjusted), cuts every frame at the last COMPLETE day (today's bar is
treated as partial, whether or not the market has closed) and builds the weekly multi-timeframe state of each
stock (setup_engine.mtf). It then flags the pre-registered signatures of the multi-timeframe study: S1 (RS
leader), S2 (beaten-down reclaim), S3 (breakout leader), S4 (full confluence) and the B0 baseline, and sorts
the names into groups (full_confluence, breakout_leaders, beaten_down_reclaim, leaders_not_extended,
extended) plus a sector table and a one-line market regime. Fundamentals (market cap, forward P/E, growth,
margins, next earnings date) come from yfinance for the names in the groups and are reference data only.

READ THIS BEFORE USING IT.
  * Ranks are WITHIN the S&P 500, not across all stocks: "top 20%" here is the top 20% of an already elite,
    survivorship-filtered set, so it is not the same thing as the study's "top 20% of liquid stocks".
  * The signals are WEEKLY. They change only when a new weekly bar completes (Fridays). On Monday to Thursday
    the screen repeats last Friday's weekly state; only last_close and chg_since_week move. On a week whose
    Friday was a holiday the last complete day is a Thursday and the screen still shows the previous week.
  * Price patterns are a small tilt, not a source of returns (see CAVEATS in the output).
  * trail30 is 0.70 x the latest weekly close, a REFERENCE level only. It is not the study's trailing stop,
    which trails the highest close since entry.
  * This is a research screen, not investment advice.

PRE-REGISTRATION. The signature rules are the ones in setup_engine.mtf.SIGNATURES, fixed before the event
study ran and never edited. The grouping thresholds below (TRAIL_FRAC, NEAR_HIGH, NOT_EXT_PX_SMA30, RANK26_MIN,
EXTENDED_PX_SMA30) were chosen once, before this script was ever run, and are not tuned. Nothing here is tested
for edge: the groups are descriptions of today's state, not a strategy.

KNOWN BIASES. The member list is TODAY'S list filtered by add date, so the screen is valid for today and
survivorship-biased for any past date (removed members are absent). Yahoo adjusts old prices retroactively
for splits. Volume is as Yahoo reports it. Fundamentals are whatever Yahoo returns today, with no point-in-time
guarantee, and are empty for names it throttles.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from setup_engine import mtf
from setup_engine.screen import Screen, clean_daily
from setup_engine.universe import load_sp500, members_asof, refresh_sp500

# ── fixed choices (not tuned) ─────────────────────────────────────────────────────────────────────

BENCH = "SPY"
MIN_DAILY_ROWS = 400          # a stock needs this many clean daily bars up to the as-of day
PERIOD = "4y"
BATCH = 125                   # symbols per yfinance download call
TRAIL_FRAC = 0.70             # trail30 = TRAIL_FRAC x latest weekly close (reference level only)
NEAR_HIGH = -0.08             # "not far from the high": off_high52 strictly above this
NOT_EXT_PX_SMA30 = 0.25       # "not extended": weekly close less than 25% above its 30-week average
RANK26_MIN = 0.80             # "leader" by 26-week return: rank within the S&P 500 at least this
EXTENDED_PX_SMA30 = 0.50      # "extended": weekly close 50% or more above its 30-week average
SIGS = ("S1", "S2", "S3", "S4", "B0")
OUT_DIR = Path.home() / ".roboalgo" / "reports"
PRICE_CACHE = Path.home() / ".roboalgo" / "bluechip" / "prices"
FUND_FIELDS = ("marketCap", "forwardPE", "revenueGrowth", "earningsGrowth", "operatingMargins", "returnOnEquity",
               "totalCash", "totalDebt")
OHLCV = ("open", "high", "low", "close", "volume")
STOCK_COLS = ("symbol", "security", "sector", "w_close", "off_high52", "w_px_sma30", "xs_rank_ret13",
              "xs_rank_ret26", "trail30")
STOCK_GROUPS = ("full_confluence", "breakout_leaders", "beaten_down_reclaim", "leaders_not_extended", "extended")
SECTOR_COLS = ("sector", "n", "above_30w", "share_above_30w", "s1_leaders", "median_ret26")
GROUP_TITLES = {
    "full_confluence": "Full confluence (S4)",
    "breakout_leaders": "Breakout leaders (S3, not S4)",
    "beaten_down_reclaim": "Beaten-down reclaim (S2)",
    "leaders_not_extended": (f"Leaders, not extended (S1, within {-NEAR_HIGH:.0%} of the 52-week high, under "
                             f"{NOT_EXT_PX_SMA30:.0%} above the 30-week average, 26-week rank at least {RANK26_MIN:.2f})"),
    "extended": f"Extended (S1, {EXTENDED_PX_SMA30:.0%} or more above the 30-week average)",
}
CAVEATS = ("Price patterns are a small tilt, not a source of returns. In this project's tests the leader patterns "
           "tilted returns by about +1-2% per half-year in 2018-2026 and showed nothing before 2018. Ranks are "
           "within the S&P 500 only. This is a research screen, not investment advice.")
NOTES = ("The signals are weekly: they change only when a weekly bar completes (Fridays). last_close is the last "
         "complete day. trail30 is 0.70 x the latest weekly close, a reference level only, not the study's "
         "trailing stop.")
_FORMATS = {"w_close": ".2f", "trail30": ".2f"}


# ── calendar ──────────────────────────────────────────────────────────────────────────────────────

def last_complete_day(daily_index: pd.DatetimeIndex, today) -> pd.Timestamp:
    """The latest index date STRICTLY before ``today``: today's bar is treated as partial (the market may
    be open, or Yahoo's bar may still be forming). ValueError if there is none."""
    t = pd.Timestamp(today).normalize()
    idx = pd.DatetimeIndex(daily_index).normalize()
    idx = idx[idx < t]
    if len(idx) == 0:
        raise ValueError(f"no daily bar before {t.date()}")
    return pd.Timestamp(idx.max())


def asof_week(as_of) -> pd.Timestamp:
    """The latest COMPLETE week label (a Friday) at ``as_of``: the day itself on a Friday, otherwise the Friday
    before. This is exactly the last row ``mtf.mtf_features(..., as_of=as_of)`` can return, because the
    weekly table drops a week whose Friday label is after ``as_of``."""
    as_of = pd.Timestamp(as_of)
    if as_of.weekday() == 4:
        return as_of.normalize()
    return pd.Timestamp(mtf.week_label(as_of)[0]) - pd.Timedelta(days=7)


# ── the screen ────────────────────────────────────────────────────────────────────────────────────

def _empty_panel() -> pd.DataFrame:
    """The panel with every column and no rows (nothing passed the data checks)."""
    cols: dict[str, pd.Series] = {"symbol": pd.Series(dtype=object), "week": pd.Series(dtype="datetime64[ns]")}
    for c in (*mtf.MTF_COLUMNS, "last_close", "chg_since_week", *mtf.XS_COLUMNS, "off_high52", "trail30"):
        cols[c] = pd.Series(dtype="float64")
    for c in ("w_last_day", "m_label"):
        cols[c] = pd.Series(dtype="datetime64[ns]")
    for c in ("liquid", *SIGS):
        cols[c] = pd.Series(dtype=bool)
    for c in ("security", "sector", "sub_industry"):
        cols[c] = pd.Series(dtype=object)
    return pd.DataFrame(cols)


def screen(daily_by_symbol: dict, spy: pd.DataFrame, universe_df: pd.DataFrame, as_of) -> pd.DataFrame:
    """One row per screened S&P 500 member: its latest complete weekly state, ranks, the five signatures.

    Only symbols that are members of the index at ``as_of`` (``members_asof``) are processed, so ranks are
    percentiles WITHIN the index and a downloaded outsider can never move them. Every frame is cut at
    ``as_of`` before anything is computed. Skipped members are returned in ``P.attrs["skipped"]`` as
    ``{symbol: reason}``: "no_data" (no frame), "short" (fewer than MIN_DAILY_ROWS clean bars) or
    "stale_week" (the stock's latest weekly bar is not the index's latest complete week: halted or delisted).
    """
    as_of = pd.Timestamp(as_of).normalize()
    week = asof_week(as_of)
    members = set(members_asof(universe_df, as_of)["symbol"])
    spy_c = clean_daily(spy)
    spy_c = spy_c[spy_c.index <= as_of]
    skipped: dict[str, str] = {}
    parts = []
    for sym in sorted(members):
        frame = daily_by_symbol.get(sym)
        if frame is None or len(frame) == 0:
            skipped[sym] = "no_data"
            continue
        d = clean_daily(frame)
        d = d[d.index <= as_of]
        if len(d) < MIN_DAILY_ROWS:
            skipped[sym] = "short"
            continue
        wf = mtf.mtf_features(d, spy_c, as_of=as_of)
        if len(wf) == 0:
            skipped[sym] = "short"
            continue
        if wf.index[-1] != week:
            skipped[sym] = "stale_week"
            continue
        last = wf.iloc[[-1]].reset_index()                       # the "week" index becomes a column
        last.insert(0, "symbol", sym)
        last_close = float(d["close"].iloc[-1])
        last["last_close"] = last_close
        last["chg_since_week"] = last_close / last["w_close"] - 1.0
        parts.append(last)
    spy_wf = mtf.mtf_features(spy_c, as_of=as_of)
    spy_px = float(spy_wf["w_px_sma30"].iloc[-1]) if len(spy_wf) and spy_wf.index[-1] == week else float("nan")
    if parts:
        P = pd.concat(parts, ignore_index=True)
        P["liquid"] = mtf.is_liquid(P)
        P = mtf.add_xs_ranks(P, week_col="week")
        for s in SIGS:
            P[s] = Screen(mtf.SIGNATURES[s]).mask(P)
        P["off_high52"] = P["w_dd52"]
        P["trail30"] = TRAIL_FRAC * P["w_close"]
        meta = universe_df[["symbol", "security", "sector", "sub_industry"]].drop_duplicates("symbol")
        P = P.merge(meta, on="symbol", how="left")
    else:
        P = _empty_panel()
    P = P.sort_values("symbol", kind="stable").reset_index(drop=True)
    P.attrs["skipped"] = skipped
    P.attrs["spy_px_sma30"] = spy_px
    return P


def _pick(P: pd.DataFrame, mask) -> pd.DataFrame:
    """The STOCK_COLS rows where ``mask`` holds, best 26-week rank first (ties keep symbol order)."""
    out = P.loc[np.asarray(mask, dtype=bool), list(STOCK_COLS)]
    return out.sort_values("xs_rank_ret26", ascending=False, kind="stable", na_position="last").reset_index(drop=True)


def sector_table(P: pd.DataFrame) -> pd.DataFrame:
    """Per sector: stocks, how many are above their 30-week average, S1 leaders, median 26-week return."""
    rows = []
    for sector, g in P.assign(sector=P["sector"].fillna("(unknown)")).groupby("sector", sort=True):
        above = int((g["w_px_sma30"] > 0).sum())
        rows.append({"sector": sector, "n": int(len(g)), "above_30w": above,
                     "share_above_30w": above / len(g), "s1_leaders": int(g["S1"].astype(bool).sum()),
                     "median_ret26": float(g["w_ret26"].median())})
    out = pd.DataFrame(rows, columns=list(SECTOR_COLS))
    return out.sort_values("median_ret26", ascending=False, kind="stable", na_position="last").reset_index(drop=True)


def groups(P: pd.DataFrame) -> dict:
    """Sort the screened names into groups. Keys: full_confluence, breakout_leaders, beaten_down_reclaim,
    leaders_not_extended, extended (stock tables, STOCK_COLS, best 26-week rank first), sector_leadership
    (table) and regime (dict)."""
    s1, s2, s3, s4 = (P[s].to_numpy(dtype=bool) for s in ("S1", "S2", "S3", "S4"))
    off, px, r26 = (pd.to_numeric(P[c], errors="coerce").to_numpy(dtype="float64")
                    for c in ("off_high52", "w_px_sma30", "xs_rank_ret26"))
    with np.errstate(invalid="ignore"):                       # NaN compares False, which is what we want
        not_extended = s1 & (off > NEAR_HIGH) & (px < NOT_EXT_PX_SMA30) & (r26 >= RANK26_MIN)
        extended = s1 & (px >= EXTENDED_PX_SMA30)
        above = px > 0
    bm = pd.to_numeric(P["bm_w_px_sma30"], errors="coerce").dropna() if "bm_w_px_sma30" in P else pd.Series(dtype=float)
    spy_px = float(bm.iloc[-1]) if len(bm) else float(P.attrs.get("spy_px_sma30", float("nan")))
    n = int(len(P))
    return {
        "full_confluence": _pick(P, s4),
        "breakout_leaders": _pick(P, s3 & ~s4),
        "beaten_down_reclaim": _pick(P, s2),
        "leaders_not_extended": _pick(P, not_extended),
        "extended": _pick(P, extended),
        "sector_leadership": sector_table(P),
        "regime": {"benchmark": BENCH, "bench_px_sma30": spy_px, "bench_above_30w": bool(spy_px > 0),
                   "share_above_30w": float(above.mean()) if n else float("nan"), "n": n},
    }


def regime_line(regime: dict) -> str:
    px, share = regime["bench_px_sma30"], regime["share_above_30w"]
    bench = (f"{regime['benchmark']} is {abs(px) * 100:.1f}% {'above' if px > 0 else 'below'} its 30-week average"
             if math.isfinite(px) else f"{regime['benchmark']} vs its 30-week average: n/a")
    breadth = (f"{share * 100:.0f}% of the {regime['n']} screened stocks are above theirs"
               if math.isfinite(share) else "no stocks screened")
    return f"Regime: {bench}; {breadth}."


# ── data ──────────────────────────────────────────────────────────────────────────────────────────

def _tidy(df: pd.DataFrame):
    """A raw OHLCV frame -> the cleaned daily frame the screen expects (lower-case open..volume, tz-naive
    normalized DatetimeIndex, bad bars dropped), or None when nothing usable is left."""
    if df is None or len(df) == 0:
        return None
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    if not isinstance(df.index, pd.DatetimeIndex) and "date" in df.columns:
        df = df.set_index(pd.to_datetime(df["date"]))
    if not all(c in df.columns for c in OHLCV) or not isinstance(df.index, pd.DatetimeIndex):
        return None
    df = df[list(OHLCV)]
    idx = df.index
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df.index.name = "date"
    df = df.dropna(how="all")
    if len(df) == 0:
        return None
    df = clean_daily(df)
    return df if len(df) else None


def _yf_batch(symbols: list, period: str) -> dict:
    """yfinance download of one batch -> {symbol: frame with lower-case open..volume}. ``Adj Close`` is ignored:
    prices are split-adjusted but NOT dividend-adjusted, the same as the cached history."""
    import yfinance as yf

    raw = yf.download(list(symbols), period=period, auto_adjust=False, group_by="ticker", threads=True,
                      progress=False)
    out: dict = {}
    if raw is None or len(raw) == 0:
        return out
    for s in symbols:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                if s not in raw.columns.get_level_values(0):
                    continue
                sub = raw[s]
            elif len(symbols) == 1:
                sub = raw
            else:
                continue
            out[s] = sub[["Open", "High", "Low", "Close", "Volume"]].rename(columns=str.lower)
        except Exception:
            continue
    return out


def fetch_prices(symbols, period: str = PERIOD, batch: int = BATCH, fetch=None) -> dict:
    """Daily bars for ``symbols`` in batches of ``batch``: ``{symbol: cleaned frame}``. Symbols with no usable
    bars are absent. ``fetch(list_of_symbols, period) -> {symbol: raw OHLCV frame}`` is injectable."""
    fetch = fetch or _yf_batch
    symbols = list(symbols)
    out: dict = {}
    for i in range(0, len(symbols), batch):
        chunk = symbols[i:i + batch]
        for sym, raw in fetch(chunk, period).items():
            df = _tidy(raw)
            if df is not None:
                out[sym] = df
    return out


def _num(x) -> float:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return f if math.isfinite(f) else float("nan")


def _next_earnings(cal) -> pd.Timestamp:
    """The first 'Earnings Date' of a yfinance calendar (dict or DataFrame), else NaT."""
    try:
        v = None
        if isinstance(cal, dict):
            v = cal.get("Earnings Date")
        elif isinstance(cal, pd.DataFrame):
            if "Earnings Date" in cal.index:
                v = cal.loc["Earnings Date"].iloc[0]
            elif "Earnings Date" in cal.columns:
                v = cal["Earnings Date"].iloc[0]
        if isinstance(v, (list, tuple, np.ndarray, pd.Series, pd.Index)):
            v = v[0] if len(v) else None
        t = pd.to_datetime(v, errors="coerce")
        if pd.isna(t):
            return pd.NaT
        t = pd.Timestamp(t)
        return t.tz_localize(None).normalize() if t.tzinfo is not None else t.normalize()
    except Exception:
        return pd.NaT


def _yf_fund(symbol: str):
    try:
        import yfinance as yf

        t = yf.Ticker(symbol)
        return t.info, t.calendar
    except Exception:
        return {}, None


def fundamentals(symbols, fetch=None) -> pd.DataFrame:
    """Reference fundamentals: ``symbol``, FUND_FIELDS (NaN when missing) and ``next_earnings`` (NaT when
    unknown). ``fetch(symbol) -> (info dict, calendar)`` is injectable. Never raises: a failing symbol is a row
    of NaN."""
    fetch = fetch or _yf_fund
    rows = []
    for sym in symbols:
        try:
            info, cal = fetch(sym)
        except Exception:
            info, cal = {}, None
        info = info if isinstance(info, dict) else {}
        row = {"symbol": sym}
        for f in FUND_FIELDS:
            row[f] = _num(info.get(f))
        row["next_earnings"] = _next_earnings(cal)
        rows.append(row)
    out = pd.DataFrame(rows, columns=["symbol", *FUND_FIELDS, "next_earnings"])
    out["next_earnings"] = pd.to_datetime(out["next_earnings"])
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


def _cell(v, col: str) -> str:
    if v is None or (not isinstance(v, (str, bool)) and pd.isna(v)):
        return ""
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return format(float(v), _FORMATS.get(col, ".3f"))
    if isinstance(v, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(v).date().isoformat()
    return " ".join(str(v).split()).replace("|", "\\|")


def _md_table(df: pd.DataFrame) -> str:
    """A GitHub markdown table (pandas' to_markdown needs tabulate, which is not installed)."""
    if len(df) == 0:
        return "(none)"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for rec in df.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(_cell(v, c) for v, c in zip(rec, cols)) + " |")
    return "\n".join(lines)


def write_outputs(P: pd.DataFrame, G: dict, as_of, out_dir, fund=None) -> dict:
    """Write bluechip_<YYYY-MM-DD>.csv / .md / .json into ``out_dir``; returns {"csv", "md", "json"} paths.

    The CSV holds every screened row (fundamentals merged in where fetched), the markdown one table per
    group plus the sector table and the CAVEATS, the JSON the same content with strict types (no NaN)."""
    as_of = pd.Timestamp(as_of).normalize()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"bluechip_{as_of:%Y-%m-%d}"
    paths = {"csv": out_dir / f"{stem}.csv", "md": out_dir / f"{stem}.md", "json": out_dir / f"{stem}.json"}
    skipped = dict(P.attrs.get("skipped", {}))
    week = asof_week(as_of)

    table = P.copy()
    if fund is not None and len(fund):
        table = table.merge(fund, on="symbol", how="left")
    table.to_csv(paths["csv"], index=False)

    regime = G["regime"]
    reasons: dict[str, int] = {}
    for r in skipped.values():
        reasons[r] = reasons.get(r, 0) + 1
    skip_txt = ", ".join(f"{k}: {v}" for k, v in sorted(reasons.items())) or "none"
    md = [f"# S&P 500 blue-chip screen, as of {as_of:%Y-%m-%d}", "",
          f"Weekly state of the week ending {week:%Y-%m-%d}. {NOTES}", "",
          regime_line(regime), "",
          f"Screened {len(P)} stocks; skipped {len(skipped)} ({skip_txt}).", ""]
    for name in STOCK_GROUPS:
        md += [f"## {GROUP_TITLES[name]} ({len(G[name])})", "", _md_table(G[name]), ""]
    md += ["## Sector leadership", "", _md_table(G["sector_leadership"]), "", "## Caveats", "", CAVEATS, ""]
    paths["md"].write_text("\n".join(md), encoding="utf-8")

    obj = {"as_of": as_of, "asof_week": week, "regime": regime,
           "groups": {name: G[name] for name in STOCK_GROUPS},
           "sector_leadership": G["sector_leadership"], "skipped": skipped, "caveats": CAVEATS}
    paths["json"].write_text(json.dumps(_jsonable(obj), indent=1, allow_nan=False), encoding="utf-8")
    return paths


# ── main ──────────────────────────────────────────────────────────────────────────────────────────

def _read_prices(prices_from: Path, symbols) -> dict:
    """``<SYMBOL>.pkl`` or ``<SYMBOL>.parquet`` from a directory (offline mode); missing files are skipped."""
    out = {}
    for sym in symbols:
        pkl, pq = prices_from / f"{sym}.pkl", prices_from / f"{sym}.parquet"
        if pkl.exists():
            raw = pd.read_pickle(pkl)
        elif pq.exists():
            raw = pd.read_parquet(pq)
        else:
            continue
        df = _tidy(raw)
        if df is not None:
            out[sym] = df
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR, help=f"report directory (default {OUT_DIR})")
    ap.add_argument("--limit", type=int, default=None, help="only the first N symbols of the sorted universe")
    ap.add_argument("--no-fundamentals", action="store_true", help="skip the yfinance fundamentals lookup")
    ap.add_argument("--no-network", action="store_true",
                    help="never touch the network: needs --prices-from, uses --universe or the newest cached list")
    ap.add_argument("--prices-from", type=Path, default=None,
                    help="directory with <SYMBOL>.pkl or <SYMBOL>.parquet daily bars (SPY included)")
    ap.add_argument("--universe", type=Path, default=None, help="S&P 500 csv (default: the cached/refreshed list)")
    ap.add_argument("--today", default=None, help="YYYY-MM-DD, default today; the screen uses the last day BEFORE it")
    ap.add_argument("--workers", type=int, default=1, help="accepted for symmetry with the other scripts; unused")
    args = ap.parse_args(argv)

    if args.no_network and args.prices_from is None:
        ap.error("--no-network needs --prices-from DIR")
    today = pd.Timestamp(args.today) if args.today else pd.Timestamp(date.today())

    if args.universe is not None:
        universe = load_sp500(args.universe)
    elif args.no_network:
        universe = load_sp500()
    else:
        universe = load_sp500(refresh_sp500())
    if args.limit is not None:
        universe = universe.sort_values("symbol").head(args.limit)
    universe = universe.reset_index(drop=True)
    symbols = sorted(set(universe["symbol"]))

    if args.no_network:
        prices = _read_prices(args.prices_from, [*symbols, BENCH])
    else:
        prices = fetch_prices([*symbols, BENCH])
        PRICE_CACHE.mkdir(parents=True, exist_ok=True)
        for sym, df in prices.items():
            df.to_pickle(PRICE_CACHE / f"{sym}.pkl")
    if BENCH not in prices:
        print(f"no price data for {BENCH}; cannot build the market regime", file=sys.stderr)
        return 2
    spy = prices.pop(BENCH)
    as_of = last_complete_day(spy.index, today)
    prices = {s: df[df.index <= as_of] for s, df in prices.items()}
    spy = spy[spy.index <= as_of]

    P = screen(prices, spy, universe, as_of)
    G = groups(P)

    fund = None
    if not (args.no_fundamentals or args.no_network):           # fundamentals need the network
        names = sorted(set().union(*(set(G[g]["symbol"]) for g in STOCK_GROUPS)))
        fund = fundamentals(names)
    paths = write_outputs(P, G, as_of, args.out_dir, fund)

    print(f"as_of {as_of:%Y-%m-%d}  (weekly state: week ending {asof_week(as_of):%Y-%m-%d})")
    print(regime_line(G["regime"]))
    print(f"screened {len(P)}, skipped {len(P.attrs['skipped'])}")
    print("groups: " + "  ".join(f"{g}={len(G[g])}" for g in STOCK_GROUPS))
    for kind, p in paths.items():
        print(f"wrote {kind}: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
