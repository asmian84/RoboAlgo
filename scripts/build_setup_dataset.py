"""
Build the learning dataset: one row per (stock, observation date) with scale-free features on
daily / weekly / monthly bars and the forward outcome, over the whole listed universe.

    python scripts/build_setup_dataset.py                    # uses ~/.roboalgo/history_2000
    python scripts/build_setup_dataset.py --history ~/.roboalgo/history_2000 --out ~/.roboalgo/setups

Writes  obs.parquet      every observation with features AND forward labels (NaN where the
                         window has not played out yet), for learning and testing
        current.parquet  the latest bar of every stock, features only, for scanning today

Needs the cache from scripts/download_universe_history.py --start 2000-01-01.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from setup_engine import features as ft

DEFAULT_HISTORY = Path.home() / ".roboalgo" / "history_2000"
DEFAULT_OUT = Path.home() / ".roboalgo" / "setups"
MIN_DOLLAR_VOLUME = 5e5       # 50-day average daily traded value; price-agnostic, so split history cannot distort it


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[(df["close"] > 0) & (df["open"] > 0) & (df["high"] >= df["low"]) & (df["low"] > 0)]
    return df.dropna(subset=["open", "high", "low", "close"]).fillna({"volume": 0.0})


def _batch(args):
    path, obs_dates = args[0], args[1]
    terminal = args[2] if len(args) > 2 else frozenset()
    raw = pd.read_parquet(path)
    raw["date"] = pd.to_datetime(raw["date"]).dt.tz_localize(None)
    rows, current, n = [], [], 0
    for sym, g in raw.groupby("symbol"):
        df = clean(g.set_index("date").drop(columns="symbol").astype(float))
        if len(df) < 120:
            continue
        n += 1
        is_dead = sym in terminal
        o = ft.build_observations(sym, df, obs_dates, MIN_DOLLAR_VOLUME, with_labels=True, terminal=is_dead)
        if o is not None:
            rows.append(o)
        if not is_dead:                          # a delisted stock is not on today's scan
            c = ft.current_features(sym, df)
            if c is not None:
                current.append(c)
    return (pd.concat(rows, ignore_index=True) if rows else None,
            pd.DataFrame(current) if current else None, n)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--history", default=str(DEFAULT_HISTORY))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--start", default="2001-01-01", help="first observation date")
    args = ap.parse_args()
    hist, out = Path(args.history).expanduser(), Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(hist.glob("batch_*.parquet"))
    if not files:
        raise SystemExit(f"no history in {hist}")
    tpath = hist / "terminal.json"
    terminal = frozenset(json.loads(tpath.read_text())) if tpath.exists() else frozenset()
    if terminal:
        print(f"{len(terminal):,} delisted symbols: their last price is held through forward windows", flush=True)

    last = max(pd.read_parquet(f, columns=["date"])["date"].max() for f in files[:3])
    obs_dates = ft.observation_dates(args.start, str(pd.Timestamp(last).date()))
    print(f"{len(files)} batches | {len(obs_dates)} observation dates {obs_dates[0].date()} .. {obs_dates[-1].date()}", flush=True)

    t0 = time.time()
    obs, cur, scanned = [], [], 0
    with mp.get_context("fork").Pool(min(14, mp.cpu_count())) as pool:
        for k, (o, c, n) in enumerate(pool.imap_unordered(_batch, [(f, obs_dates, terminal) for f in files]), 1):
            if o is not None: obs.append(o)
            if c is not None: cur.append(c)
            scanned += n
            print(f"  {k}/{len(files)} batches, {scanned} stocks ({time.time() - t0:.0f}s)", flush=True)

    obs = ft.add_relative_strength(pd.concat(obs, ignore_index=True))
    cur = pd.concat(cur, ignore_index=True)
    cur = cur[cur["date"] >= pd.Timestamp(last) - pd.Timedelta(days=5)]          # still trading
    # Rank among LIQUID stocks only, exactly as the training observations were. Ranking everything would
    # make "stronger than 70% of stocks" mean something different today than it did when the rules were learned.
    cur = cur[cur["dvol50"] >= MIN_DOLLAR_VOLUME]
    cur = cur.assign(date=pd.Timestamp(last))
    cur = ft.add_relative_strength(cur)
    from setup_engine.apply import build_reference
    ref = build_reference(cur)
    print(f"relative-strength reference written: {ref['n_stocks']:,} liquid stocks as of {ref['asof']}")
    obs.to_parquet(out / "obs.parquet", index=False)
    cur.to_parquet(out / "current.parquet", index=False)
    lab = obs["fwd_max_3y"].notna()
    print(f"obs: {len(obs):,} rows, {obs['symbol'].nunique():,} stocks | labelled: {lab.sum():,} "
          f"| runners (>=3x within 3y): {(obs.loc[lab, 'fwd_max_3y'] >= 2.0).mean():.2%}")
    print(f"current: {len(cur):,} stocks as of {pd.Timestamp(last).date()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
