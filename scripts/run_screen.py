"""
Run a Finviz-style screen over the cached daily history and list the stocks that match on one date.

    python scripts/run_screen.py --rules "52-Week High: 70% or more below; Relative Volume: Over 2"
    python scripts/run_screen.py --as-of 2026-09-29 \\
        --prior-rules "2-Year High: 70% or more below; Monthly Low: Higher low" \\
        --rules "20-Day High: New High; Relative Volume: Over 2; Price: Over $1; Dollar Volume 20d: Over 1M"
    python scripts/run_screen.py --list-filters

Filter names and option labels are the ones in ``setup_engine.screen`` (``--list-filters`` prints the
dropdown). Rules are ``Filter: Option`` pairs separated by ``;`` or newlines.

``--rules`` is evaluated on the stock's bar at the as-of date. ``--prior-rules`` is evaluated on the
stock's PREVIOUS bar (the row before the as-of bar), and a stock must satisfy both. That is the
turnaround study's convention: the "bottom" is read one bar before the breakout bar, so the breakout
cannot lift the close out of the bottom state it is supposed to start from.

THE NO-LOOK-AHEAD RULE
----------------------
Every number for the as-of date is computed from bars up to and including that date, and nothing
later: bars after ``--as-of`` are dropped BEFORE the indicators are built. The screen says which
stocks qualify at that close; it can be acted on from the next open at the earliest. A stock is
evaluated only if its last bar on or before the as-of date IS the as-of date, so a halted or
delisted name that last traded earlier never shows up with stale numbers.

THE ORDER MODEL
---------------
None. A screen only says which stocks qualify. Fills, stops and costs belong to
``setup_engine.trade_sim``.

WINDOW
------
Only the last TAIL_BARS (800) clean bars of each stock are used. That covers the longest look-back
(the 2-year high needs 504 bars; the "Monthly Low" stand-in needs 378) and lets the Wilder averages
(RSI, ATR) converge to double precision, while keeping a full-universe scan cheap.

KNOWN BIASES (reported alongside every result)
----------------------------------------------
Survivorship: the cache is the tickers listed on the download date, so a stock that was delisted
after the as-of date is absent from an old as-of screen. Yahoo splits are applied retroactively, so
a reverse-split stock shows a price history it never traded at, which distorts price and
dollar-volume bands. Closes are not dividend-adjusted. Bars are cleaned with
``setup_engine.screen.clean_daily`` (the same cleaner as the setup dataset).

Needs the cache from scripts/download_universe_history.py --start 2000-01-01 (pyarrow is a
research-script dependency, imported lazily through pandas).
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from setup_engine.screen import COLUMNS, Screen, available_filters, clean_daily, indicators

DEFAULT_HISTORY = Path.home() / ".roboalgo" / "history_2000"
TAIL_BARS = 800          # covers the 504-bar and 378-bar windows; Wilder averages converge to double precision


# ── Reading the cache ─────────────────────────────────────────────────────────

def batch_files(history, limit_batches: int | None = None) -> list[Path]:
    """The ``batch_*.parquet`` files of a history cache, sorted; the first N when ``limit_batches``."""
    files = sorted(Path(history).expanduser().glob("batch_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no batch_*.parquet files in {Path(history).expanduser()}")
    return files[:limit_batches] if limit_batches else files


def _naive_dates(series: pd.Series) -> pd.Series:
    """Datetimes without a timezone (tz-aware values are converted to UTC first)."""
    dates = pd.to_datetime(series)
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_convert(None)
    return dates


def cache_dates(history, limit_batches: int | None = None) -> pd.DatetimeIndex:
    """Every distinct date in the cache, sorted. Reads ONLY the ``date`` column of each batch."""
    seen: set = set()
    for f in batch_files(history, limit_batches):
        seen.update(_naive_dates(pd.read_parquet(f, columns=["date"])["date"]).unique())
    return pd.DatetimeIndex(sorted(seen))


def _frames(path):
    """(symbol, daily bars indexed by date) for every stock in one batch file."""
    raw = pd.read_parquet(path)
    raw["date"] = _naive_dates(raw["date"])
    for symbol, g in raw.groupby("symbol"):
        yield symbol, g.set_index("date").drop(columns="symbol").astype(float)


# ── Screening ─────────────────────────────────────────────────────────────────

def latest_rows(frames, as_of) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The indicator rows of every stock that traded ON ``as_of``: (last rows, prior rows).

    ``frames`` is an iterable of ``(symbol, daily bars with a DatetimeIndex)``. Each stock is cleaned
    with ``clean_daily``, cut to the bars on or before ``as_of`` (nothing later reaches an indicator),
    and skipped unless its last bar IS ``as_of`` (stale or halted names drop out) and it has a
    previous bar. Indicators are built on the last TAIL_BARS bars. Both frames have one row per
    stock, in the same order, with ``symbol`` and ``date`` columns followed by the indicator columns;
    the prior row is the stock's own previous bar.
    """
    as_of = pd.Timestamp(as_of)
    symbols, last_dates, prior_dates, last_vals, prior_vals = [], [], [], [], []
    for symbol, df in frames:
        df = clean_daily(df)
        df = df[df.index <= as_of]
        if len(df) < 2 or df.index[-1] != as_of:
            continue
        ind = indicators(df.tail(TAIL_BARS))
        symbols.append(symbol)
        last_dates.append(ind.index[-1])
        prior_dates.append(ind.index[-2])
        last_vals.append(ind.iloc[-1].to_numpy())
        prior_vals.append(ind.iloc[-2].to_numpy())

    def build(dates, vals):
        body = (pd.DataFrame(np.vstack(vals), columns=list(COLUMNS)) if vals
                else pd.DataFrame(columns=list(COLUMNS), dtype="float64"))
        head = pd.DataFrame({"symbol": symbols, "date": pd.to_datetime(dates)})
        return pd.concat([head, body], axis=1)

    return build(last_dates, last_vals), build(prior_dates, prior_vals)


def evaluate_frames(frames, screen: Screen, as_of, prior_screen: Screen | None = None):
    """``(matches, n_evaluated)``: ``screen`` on the as-of bar AND ``prior_screen`` on the bar before."""
    last, prior = latest_rows(frames, as_of)
    n_evaluated = len(last)
    keep = screen.mask(last)
    if prior_screen is not None:
        keep = keep & prior_screen.mask(prior)
    cols: list[str] = []
    for c in ["close"] + screen.fields():
        if c not in cols:
            cols.append(c)
    out = last.loc[keep, ["symbol", "date"] + cols].reset_index(drop=True)
    if prior_screen is not None:
        extra = prior.loc[keep, prior_screen.fields()].reset_index(drop=True)
        out = pd.concat([out, extra.add_prefix("prior_")], axis=1)
    return out, n_evaluated


def screen_latest(frames, screen: Screen, as_of, prior_screen: Screen | None = None) -> pd.DataFrame:
    """Stocks matching ``screen`` on ``as_of`` (and ``prior_screen`` on the previous bar), by symbol.

    Columns: symbol, date, close, the screen's fields, then ``prior_<field>`` for each field the
    prior screen reads.
    """
    out, _ = evaluate_frames(frames, screen, as_of, prior_screen)
    return out.sort_values("symbol").reset_index(drop=True)


def _batch(args):
    """Pool worker: screen one batch file. Returns (matches, n_evaluated)."""
    path, rules, prior_rules, as_of = args
    return evaluate_frames(_frames(path), Screen(rules),
                           as_of, Screen(prior_rules) if prior_rules is not None else None)


# ── Output ────────────────────────────────────────────────────────────────────

def _fmt(x) -> str:
    if isinstance(x, float):
        return f"{x:,.0f}" if abs(x) >= 1000 else f"{x:.4g}"
    return str(x)


def _table(df: pd.DataFrame) -> str:
    shown = df.copy()
    shown["date"] = shown["date"].dt.strftime("%Y-%m-%d")
    return shown.to_string(index=False, float_format=_fmt)


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run a Finviz-style screen over the cached daily history.")
    ap.add_argument("--rules", default="", help='"Filter: Option; Filter: Option" on the as-of bar')
    ap.add_argument("--prior-rules", default=None,
                    help="rules evaluated on the stock's previous bar (must ALSO hold)")
    ap.add_argument("--history", default=str(DEFAULT_HISTORY))
    ap.add_argument("--as-of", default="latest",
                    help='"latest" (default) or a date; the effective date is the latest cache date <= it')
    ap.add_argument("--list-filters", action="store_true", help="print every filter and its example options")
    ap.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1),
                    help="processes over batches (1 = sequential)")
    ap.add_argument("--limit-batches", type=int, default=None, help="only the first N batch files")
    ap.add_argument("--csv", default=None, help="also write the matches to this CSV")
    return ap


def _print_filters() -> None:
    for name, options in available_filters().items():
        print(f"{name}: {', '.join(options)}")


def _fail(message: str) -> int:
    print(message, file=sys.stderr)
    return 2


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    if args.list_filters:
        _print_filters()
        return 0
    if not args.rules.strip() and not (args.prior_rules or "").strip():
        return _fail("nothing to screen: give --rules and/or --prior-rules (see --list-filters)")
    try:
        screen = Screen.from_text(args.rules)
        prior_screen = Screen.from_text(args.prior_rules) if args.prior_rules is not None else None
    except ValueError as e:
        return _fail(f"error: {e}")

    try:
        files = batch_files(args.history, args.limit_batches)
    except FileNotFoundError as e:
        return _fail(f"error: {e}")
    dates = cache_dates(args.history, args.limit_batches)
    if len(dates) == 0:
        return _fail(f"error: no dates in {args.history}")
    cache_last = dates[-1]
    if str(args.as_of).strip().lower() == "latest":
        as_of = cache_last
    else:
        try:
            wanted = pd.Timestamp(args.as_of)
        except (ValueError, TypeError):
            return _fail(f"error: cannot read --as-of {args.as_of!r} (use YYYY-MM-DD or 'latest')")
        usable = dates[dates <= wanted]
        if len(usable) == 0:
            return _fail(f"error: no cache date on or before {wanted.date()} (cache starts {dates[0].date()})")
        as_of = usable[-1]

    jobs = [(f, screen.rules, prior_screen.rules if prior_screen is not None else None, as_of) for f in files]
    workers = max(1, min(args.workers, len(jobs)))
    if workers == 1:
        results = [_batch(j) for j in jobs]
    else:
        with mp.get_context("fork").Pool(workers) as pool:
            results = list(pool.imap_unordered(_batch, jobs))
    n_evaluated = sum(n for _, n in results)
    parts = [m for m, _ in results if len(m)]
    if parts:
        matches = pd.concat(parts, ignore_index=True).sort_values("symbol").reset_index(drop=True)
    else:
        matches = pd.DataFrame(columns=["symbol", "date", "close"])

    print(f"cache last date: {cache_last.date()} | as of: {as_of.date()} | "
          f"stocks evaluated: {n_evaluated} | matches: {len(matches)}")
    print("screen:")
    for line in screen.describe().splitlines():
        print(f"  {line}")
    if prior_screen is not None:
        print("prior screen (the stock's previous bar):")
        for line in prior_screen.describe().splitlines():
            print(f"  {line}")
    if len(matches):
        print(_table(matches))
    else:
        print("(no matches)")
    if args.csv:
        out = matches.copy()
        if len(out):
            out["date"] = out["date"].dt.strftime("%Y-%m-%d")
        out.to_csv(args.csv, index=False)
        print(f"wrote {len(out)} rows to {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
