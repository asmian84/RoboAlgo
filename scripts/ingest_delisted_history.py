"""
Merge price history for DELISTED stocks into the learning cache, so the learning can be rerun without
survivorship bias.

    python scripts/ingest_delisted_history.py --source ~/data/delisted_prices/   # a directory of CSV/parquet
    python scripts/ingest_delisted_history.py --source one_big_file.parquet --into ~/.roboalgo/history_all

WHY THIS EXISTS. Every learning round so far used stocks that are listed TODAY. Stocks that crashed and
were delisted are missing, so the crashed stocks that remain are the ones that recovered, and any pattern
built on crashes is contaminated. The free sources checked (Yahoo, AlphaVantage, FMP, Finnhub, Stooq)
do not serve prices for delisted tickers, so this needs a vendor dataset that does. It accepts any of them:
one file per symbol, or one combined file with a symbol column.

WHAT IT ACCEPTS. CSV or parquet with a symbol (combined files only), a date, open, high, low, close and
volume. Common column names are recognised case-insensitively (ticker, t, o/h/l/c/v, ...). A numeric date
column is read as epoch milliseconds (Polygon-style). Use --map to override, e.g. --map close=adj_close.
Prices should be split-adjusted like the survivor cache; dividend-adjusted is fine if the survivors are too.

WHAT IT DOES TO KEEP THE RESULT HONEST.
  * Rows with a missing open/high/low/close are dropped (same rule as the survivor pipeline).
  * A symbol already in the survivor cache is SKIPPED and reported, not merged. Tickers are re-used, and a
    delisted company's history must never be spliced onto a different company that took the same symbol.
  * Every ingested symbol is written to terminal.json. `build_setup_dataset.py` then holds its last price
    flat through the forward window, so a stock that died keeps its terrible outcome instead of having
    those observations dropped for "incomplete data".
  * The survivor batches are symlinked, not copied, and never modified.

Then:  python scripts/build_setup_dataset.py --history <into>  &&  python scripts/learn_setups.py ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

DEFAULT_SURVIVORS = Path.home() / ".roboalgo" / "history_2000"
DEFAULT_INTO = Path.home() / ".roboalgo" / "history_all"
ALIASES = {
    "symbol": ["symbol", "ticker", "ticker_symbol", "sym", "permno_ticker"],
    "date": ["date", "timestamp", "datetime", "time", "t"],
    "open": ["open", "o", "adj_open"],
    "high": ["high", "h", "adj_high"],
    "low": ["low", "l", "adj_low"],
    "close": ["close", "c", "adj_close", "closeprice"],
    "volume": ["volume", "v", "vol", "adj_volume"],
}
BATCH = 200


def read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def standardise(df: pd.DataFrame, mapping: dict[str, str], default_symbol: str | None) -> pd.DataFrame:
    cols = {c.lower().strip(): c for c in df.columns}
    out = {}
    for want, names in ALIASES.items():
        chosen = mapping.get(want)
        if chosen is not None:
            if chosen.lower() not in cols:
                raise SystemExit(f"--map {want}={chosen}: no such column. Columns: {list(df.columns)}")
            out[want] = df[cols[chosen.lower()]]
            continue
        for n in names:
            if n in cols:
                out[want] = df[cols[n]]
                break
    if "symbol" not in out:
        if default_symbol is None:
            raise SystemExit(f"no symbol column found and the file name gives none. Columns: {list(df.columns)}")
        out["symbol"] = pd.Series(default_symbol, index=df.index)
    missing = [k for k in ("date", "open", "high", "low", "close") if k not in out]
    if missing:
        raise SystemExit(f"missing columns {missing}. Columns: {list(df.columns)}. Use --map NAME=column.")
    res = pd.DataFrame(out)
    if "volume" not in res:
        res["volume"] = 0.0
    d = res["date"]
    res["date"] = pd.to_datetime(d, unit="ms") if pd.api.types.is_numeric_dtype(d) and d.abs().max() > 1e11 else pd.to_datetime(d)
    res["date"] = res["date"].dt.tz_localize(None).dt.normalize()
    res["symbol"] = res["symbol"].astype(str).str.upper().str.strip()
    for c in ("open", "high", "low", "close", "volume"):
        res[c] = pd.to_numeric(res[c], errors="coerce")
    res["volume"] = res["volume"].fillna(0.0)
    return res[["symbol", "date", "open", "high", "low", "close", "volume"]]


def survivor_symbols(survivors: Path) -> set[str]:
    syms: set[str] = set()
    for f in sorted(survivors.glob("batch_*.parquet")):
        syms |= set(pd.read_parquet(f, columns=["symbol"])["symbol"].unique())
    return syms


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--source", required=True, help="a file, or a directory of CSV/parquet files")
    ap.add_argument("--survivors", default=str(DEFAULT_SURVIVORS))
    ap.add_argument("--into", default=str(DEFAULT_INTO))
    ap.add_argument("--map", action="append", default=[], metavar="NAME=column", help="override a column, e.g. close=adj_close")
    ap.add_argument("--min-bars", type=int, default=120, help="skip symbols with fewer daily bars than this")
    args = ap.parse_args(argv)

    src, surv, into = Path(args.source).expanduser(), Path(args.survivors).expanduser(), Path(args.into).expanduser()
    mapping = dict(m.split("=", 1) for m in args.map)
    files = [src] if src.is_file() else sorted(p for p in src.rglob("*") if p.suffix.lower() in (".csv", ".parquet"))
    if not files:
        raise SystemExit(f"no CSV or parquet files under {src}")
    if not list(surv.glob("batch_*.parquet")):
        raise SystemExit(f"no survivor cache in {surv}; run scripts/download_universe_history.py first")

    frames = []
    for f in files:
        frames.append(standardise(read_any(f), mapping, default_symbol=f.stem.split(".")[0].upper()))
    data = pd.concat(frames, ignore_index=True)
    before = len(data)
    data = data.dropna(subset=["open", "high", "low", "close", "date"])
    data = data[(data["close"] > 0) & (data["open"] > 0) & (data["high"] >= data["low"]) & (data["low"] > 0)]
    dropped = before - len(data)

    existing = survivor_symbols(surv)
    collide = sorted(set(data["symbol"]) & existing)
    data = data[~data["symbol"].isin(existing)]
    counts = data.groupby("symbol").size()
    short = sorted(counts[counts < args.min_bars].index)
    data = data[~data["symbol"].isin(short)]
    symbols = sorted(data["symbol"].unique())

    into.mkdir(parents=True, exist_ok=True)
    for f in sorted(surv.glob("batch_*.parquet")):
        link = into / f.name
        if not link.exists():
            os.symlink(f, link)
    for k in range(0, len(symbols), BATCH):
        part = data[data["symbol"].isin(symbols[k:k + BATCH])].sort_values(["symbol", "date"])
        part.to_parquet(into / f"batch_delisted_{k // BATCH:04d}.parquet", index=False)
    (into / "terminal.json").write_text(json.dumps(sorted(symbols)))
    manifest = {"survivor_cache": str(surv), "ingested_symbols": len(symbols), "rows": int(len(data)),
                "dropped_incomplete_rows": int(dropped), "skipped_ticker_collisions": collide,
                "skipped_too_short": short, "source": str(src),
                "note": "terminal.json symbols are held at their last price through forward windows"}
    (into / "manifest_delisted.json").write_text(json.dumps(manifest, indent=1))
    print(f"ingested {len(symbols):,} delisted symbols ({len(data):,} rows) into {into}")
    print(f"  dropped {dropped:,} incomplete rows | skipped {len(collide)} ticker collisions with the survivor cache"
          f" | skipped {len(short)} with under {args.min_bars} bars")
    if collide:
        print("  collisions (not merged, to avoid splicing two different companies):", ", ".join(collide[:15]), "..." if len(collide) > 15 else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
