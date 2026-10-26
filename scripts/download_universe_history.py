"""
Download daily history for the WHOLE listed US stock universe, for backtesting.

Why this exists: validating a screener means replaying its rule over every
ticker it could have scanned, not over the handful that happen to be in the
database. The database tracks ~234 curated instruments. The Finder's wide scan
covers ~11.9k. A backtest on the 234 says little about the penny and small-cap
presets, which live almost entirely outside them.

The universe list comes from the TradingView provider (data_engine/
tradingview_provider.py), the same source the live wide scan uses. Prices come
from yfinance with auto_adjust=False: OHLC are split-adjusted but NOT
dividend-adjusted, which is what the live Finder sees.

KNOWN LIMITS. These bias a backtest on this data and are reported, not fixed:

  * SURVIVORSHIP. The universe is whatever is listed TODAY. Companies that were
    delisted, went bankrupt or were acquired are absent, and penny stocks
    delist constantly. Results are therefore an UPPER bound on real-world
    performance. A strategy that fails here fails for real; one that passes has
    not yet been shown to work.
  * REVERSE SPLITS. Yahoo split-adjusts history retroactively, so a stock that
    was $0.15 and later reverse-split 1:20 shows as $3.00 in the past. Price-band
    and volume filters then see the wrong historical values.
  * Preferred shares (tickers containing "/") cannot be fetched and are skipped.

Resumable: each batch is written to its own parquet file; rerunning skips the
batches already on disk. Output: ~/.roboalgo/history/ (outside the repo).

    python scripts/download_universe_history.py            # everything, from 2018
    python scripts/download_universe_history.py --limit 400   # smoke test
    python scripts/download_universe_history.py --start 2000-01-01 --out-dir ~/.roboalgo/history_2000
                                                          # long history, for multi-year studies
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import pandas as pd

OUT_DIR = Path.home() / ".roboalgo" / "history"
BATCH = 200
START = "2018-01-01"


def universe() -> list[str]:
    from data_engine import tradingview_provider as tv
    from scanner_engine.finder import FinderCriteria

    res = tv.screen(FinderCriteria(min_price=None, max_price=None, min_volume=0))
    if not res:
        raise SystemExit("TradingView provider returned nothing; cannot build the universe list.")
    return sorted({r["symbol"] for r in res[0]})


def fetch_batch(symbols: list[str], start: str = START, attempts: int = 3) -> pd.DataFrame:
    """One multi-ticker download -> long frame. Retries on an empty result,
    which is how a rate limit usually shows up."""
    import yfinance as yf

    last_err = None
    for k in range(attempts):
        try:
            raw = yf.download(symbols, start=start, interval="1d", auto_adjust=False,
                              group_by="ticker", threads=True, progress=False)
            if raw is not None and len(raw):
                frames = []
                for s in symbols:
                    if s not in raw.columns.get_level_values(0):
                        continue
                    d = raw[s][["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
                    if d.empty:
                        continue
                    d = d.copy()
                    d.columns = ["open", "high", "low", "close", "volume"]
                    d.insert(0, "symbol", s)
                    d.index.name = "date"
                    frames.append(d.reset_index())
                if frames:
                    return pd.concat(frames, ignore_index=True)
        except Exception as exc:  # network, parsing, rate limit
            last_err = exc
        time.sleep(4 * (k + 1))
    if last_err:
        print(f"   gave up on a batch: {type(last_err).__name__}: {last_err}", flush=True)
    return pd.DataFrame()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--limit", type=int, default=0, help="only the first N symbols (smoke test)")
    ap.add_argument("--start", default=START, help="first date to fetch (default %(default)s)")
    ap.add_argument("--out-dir", default=str(OUT_DIR), help="cache directory (default %(default)s)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    listed = universe()
    skipped = [s for s in listed if "/" in s]
    symbols = [s.replace(".", "-") for s in listed if "/" not in s]
    if args.limit:
        symbols = symbols[: args.limit]
    batches = [symbols[i : i + BATCH] for i in range(0, len(symbols), BATCH)]
    print(f"universe: {len(listed)} listed | {len(skipped)} skipped (preferreds) | "
          f"{len(symbols)} to fetch in {len(batches)} batches -> {out_dir}", flush=True)

    got = 0
    t0 = time.time()
    for i, b in enumerate(batches):
        f = out_dir / f"batch_{i:04d}.parquet"
        if f.exists():
            got += pd.read_parquet(f, columns=["symbol"])["symbol"].nunique()
            continue
        df = fetch_batch(b, args.start)
        if len(df):
            df.to_parquet(f, index=False)
            got += df["symbol"].nunique()
        print(f"  batch {i + 1:>3}/{len(batches)}  symbols with data so far: {got:>6}  "
              f"({time.time() - t0:.0f}s)", flush=True)
        time.sleep(1.0)

    manifest = {
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "start": args.start,
        "listed_today": len(listed),
        "skipped_preferreds": len(skipped),
        "requested": len(symbols),
        "with_data": got,
        "no_data": len(symbols) - got,
        "adjustment": "split-adjusted, not dividend-adjusted",
        "survivorship_warning": "universe = tickers listed on the download date; delisted names are absent",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
