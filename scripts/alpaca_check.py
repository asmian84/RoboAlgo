"""
Go/no-go: can Alpaca supply price history for delisted US stocks (or must we keep using the Tiingo loader)?

    export ALPACA_API_KEY=... ALPACA_SECRET_KEY=...      # paper-account keys are enough
    python scripts/alpaca_check.py                        # free plan: 40 delisted stocks, seed 7
    python scripts/alpaca_check.py --plan plus --sample 100
    python scripts/alpaca_check.py --out /tmp/check.json --feed iex

WHAT IT DOES, in four steps. (1) Reads the market clock: proves the credentials and the host. (2) Fetches AAPL
daily bars since 2016-01-04: proves the data subscription and the history start. (3) Lists Alpaca's inactive and
active US-equity assets once each (the exchange column tells OTC rows apart; an active asset with the symbol of a
probed stock flags a re-used ticker). (4) Draws a seeded sample from the AlphaVantage delisted list
(~/.roboalgo/delisted/av_delisted.csv, common stocks delisted 2017-2025, the same filters as the Tiingo job),
asks Alpaca for each one's daily bars from 2016 to 30 days past the delisting date, and tests whether the bars
cover the stock's life. Each stock is asked first with the symbol mapped as of the day before its delisting
(asof), then, if that returns nothing or an error, with a plain request. The result is a JSON report with one
row per stock and a verdict.

PRE-REGISTRATION. Fixed before any Alpaca data was seen: sample 40, seed 7, delisting years 2017-2025
(tiingo_provider.select_universe filters), start tolerance 60 days from max(ipo, 2016-01-04), end tolerance 45
days around the delisting date, PASS share 0.5 (>=), feed sip, adjustment split, asof = delisting date - 1 day
then a plain retry. Nothing is tuned after the data is seen. PASS means at least half of the sampled stocks are
covered; it is a statement about post-2016 delistings, not a replacement verdict for the Tiingo loader.

KNOWN BIASES. The date test is not an identity test: tickers are re-used, so a series that fits the dates may
belong to another company (ticker_now_active and the per-stock rows expose the risk). Alpaca history starts in
2016, so 'life' means life since 2016. The inactive_non_otc subset (stocks Alpaca lists as inactive on a non-OTC
exchange) is conditioned on Alpaca already knowing the stock and is reported for information only; the verdict
uses the whole AlphaVantage sample, so names that were OTC at delisting count as failures. The asof parameter is
unverified live: if Alpaca rejects it, every row falls back to the plain request and asof_error shows why.

Credentials come from the environment only and are never printed or written (the report lists the NAMES of the
variables that were set, nothing else). The module does not import config.settings, so keys that live only in
.env are not seen: export them. Exit 0 whatever the verdict; 2 when credentials are missing or rejected (a
rejected credential also writes a NOT_RUN report).
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd


def _load(name: str, rel: str):
    # data_engine/__init__ pulls in the database layer and config.settings, so load the modules by path.
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ap = _load("_alpaca_check_provider", "data_engine/alpaca_provider.py")
tp = _load("_alpaca_check_tiingo", "data_engine/tiingo_provider.py")

AV_CSV = Path.home() / ".roboalgo" / "delisted" / "av_delisted.csv"
REPORT_DIR = Path.home() / ".roboalgo" / "reports"
SAMPLE_N = 40
SEED = 7
DELIST_FROM_YEAR = 2017
DELIST_TO_YEAR = 2025
DATA_START = "2016-01-04"               # first trading day of 2016; Alpaca history starts in 2016
PROBE_START = "2016-01-01"
AAPL_START = "2016-01-04"
TAIL_DAYS = 30
START_TOL_DAYS = 60
END_TOL_DAYS = 45
PASS_SHARE = 0.5
ADJUSTMENT = "split"
VERDICT_PASS = "Alpaca supplies delisted history for at least half of the sampled stocks"
VERDICT_FAIL = "Alpaca cannot supply delisted history; use the Tiingo loader"
CAVEAT = ("Tickers are re-used: covers_life tests dates, not identity, so a bar series may belong to another company "
          "that held the ticker. History starts in 2016, so 'life' means life since 2016-01-04.")


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


def _write_json(path: Path, obj) -> None:
    """Atomic: write a temp file next to the target, then replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(_jsonable(obj), indent=1, allow_nan=False))
    os.replace(tmp, path)


# ── the probe ─────────────────────────────────────────────────────────────────────────────────────

def covers_life(first, last, ipo, delist) -> bool:
    """True when the bars start within START_TOL_DAYS of max(ipo, 2016-01-04) and end within END_TOL_DAYS of the delisting."""
    if first is None or last is None or pd.isna(first) or pd.isna(last):
        return False
    first, last, ipo, delist = (pd.Timestamp(x) for x in (first, last, ipo, delist))
    anchor = max(ipo, pd.Timestamp(DATA_START))
    return bool(first <= anchor + pd.Timedelta(days=START_TOL_DAYS) and abs((last - delist).days) <= END_TOL_DAYS)


def probe_universe(av: pd.DataFrame) -> pd.DataFrame:
    """AV delisted rows for the probe: tiingo_provider.select_universe over 2017-2025 (common stocks, [A-Z]{1,5},
    listed > 365 days, snapshot-day rows excluded). Independent of Alpaca's own asset list."""
    return tp.select_universe(av, DELIST_FROM_YEAR, DELIST_TO_YEAR)


def alpaca_status(symbol: str, inactive: pd.DataFrame) -> str:
    """'inactive' (a non-OTC inactive asset), 'inactive_otc' (only OTC rows; data endpoints return nothing for OTC), 'not_listed'."""
    if inactive is None or len(inactive) == 0 or "symbol" not in inactive:
        return "not_listed"
    rows = inactive[inactive["symbol"].astype(str) == symbol]
    if len(rows) == 0:
        return "not_listed"
    if (rows["exchange"].astype(str).str.upper() != "OTC").any():
        return "inactive"
    return "inactive_otc"


def probe_stock(client, row: dict, feed: str) -> dict:
    """Ask for one stock's bars (asof the day before delisting, then plain) and test whether they cover its life."""
    sym = str(row["symbol"])
    ipo, delist = pd.Timestamp(row["ipoDate"]), pd.Timestamp(row["delistingDate"])
    end = (delist + pd.Timedelta(days=TAIL_DAYS)).date().isoformat()
    asof = (delist - pd.Timedelta(days=1)).date().isoformat()
    out = {"symbol": sym, "name": row.get("name"), "ipo": ipo, "delist": delist, "via": None, "bars": 0,
           "first": None, "last": None, "covers_life": False, "asof_error": None, "plain_error": None}
    for label, kw in (("asof", {"asof": asof}), ("plain", {})):
        try:
            df = client.bars([sym], PROBE_START, end, timeframe="1Day", adjustment=ADJUSTMENT, feed=feed, **kw)
        except ap.AlpacaAuthError:
            raise
        except ap.AlpacaError as e:
            out[f"{label}_error"] = str(e)
            continue
        if df is not None and len(df):
            out.update(via=label, bars=int(len(df)), first=pd.Timestamp(df["date"].min()), last=pd.Timestamp(df["date"].max()))
            out["covers_life"] = covers_life(out["first"], out["last"], ipo, delist)
            break
    return out


def _share(a: int, b: int):
    return a / b if b else None


def summarize(rows: list) -> dict:
    n = len(rows)
    covers = sum(1 for r in rows if r["covers_life"])
    anyb = sum(1 for r in rows if r["bars"] > 0)
    inact = [r for r in rows if r["alpaca_status"] == "inactive"]
    k = sum(1 for r in inact if r["covers_life"])
    return {"sampled": n, "covers_life": covers, "share_covers_life": _share(covers, n),
            "any_bars": anyb, "share_any_bars": _share(anyb, n),
            "via_asof": sum(1 for r in rows if r["via"] == "asof"), "via_plain": sum(1 for r in rows if r["via"] == "plain"),
            "errors": sum(1 for r in rows if r["asof_error"] or r["plain_error"]),
            "inactive_non_otc": {"n": len(inact), "covers_life": k, "share": _share(k, len(inact))},
            "inactive_otc": sum(1 for r in rows if r["alpaca_status"] == "inactive_otc"),
            "not_listed": sum(1 for r in rows if r["alpaca_status"] == "not_listed"),
            "ticker_now_active": sum(1 for r in rows if r["ticker_now_active"])}


def verdict(n_covers: int, n_sampled: int) -> str:
    return "PASS" if n_sampled > 0 and n_covers >= PASS_SHARE * n_sampled else "FAIL"


def main(argv=None, client=None, today: Optional[str] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    p.add_argument("--sample", type=int, default=SAMPLE_N)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--out", default=None, help="report path (default ~/.roboalgo/reports/alpaca_check_<date>.json)")
    p.add_argument("--plan", choices=sorted(ap.PLANS), default="free")
    p.add_argument("--feed", choices=["iex", "sip"], default="sip")
    p.add_argument("--delisted-csv", default=str(AV_CSV))
    args = p.parse_args(argv)
    today = today or pd.Timestamp.now(tz="UTC").date().isoformat()
    out = Path(args.out).expanduser() if args.out else REPORT_DIR / f"alpaca_check_{today}.json"
    env_set = [n for n in ap.ENV_NAMES if (os.environ.get(n) or "").strip()]           # names only, never values
    if client is None:
        try:
            client = ap.AlpacaClient(plan=args.plan)
        except ap.AlpacaConfigError as e:
            print(f"alpaca_check: {e}", file=sys.stderr)
            return 2
    csv = Path(args.delisted_csv).expanduser()
    if not csv.exists():
        raise SystemExit(f"no delisted list at {csv}")
    report = {"script": "alpaca_check", "run_date_utc": today,
              "pre_registered": {"sample": args.sample, "seed": args.seed, "delisted_years": [DELIST_FROM_YEAR, DELIST_TO_YEAR],
                                 "start_tol_days": START_TOL_DAYS, "end_tol_days": END_TOL_DAYS, "pass_share": PASS_SHARE,
                                 "adjustment": ADJUSTMENT, "feed": args.feed, "probe_start": PROBE_START, "tail_days": TAIL_DAYS},
              "env_vars_set": env_set, "plan": args.plan, "trading_base": getattr(client, "trading_base", None), "caveat": CAVEAT}
    try:
        report["clock"] = client.clock()
        aapl = client.bars(["AAPL"], AAPL_START, today, timeframe="1Day", adjustment=ADJUSTMENT, feed=args.feed)
        first = pd.Timestamp(aapl["date"].min()) if len(aapl) else None
        # first_ok: AAPL's first bar is within the start tolerance of the requested start (a late first bar = truncated history)
        report["aapl"] = {"rows": int(len(aapl)), "first": first, "last": pd.Timestamp(aapl["date"].max()) if len(aapl) else None,
                          "first_ok": bool(first is not None and first <= pd.Timestamp(AAPL_START) + pd.Timedelta(days=START_TOL_DAYS))}
        inactive = client.assets(status="inactive", asset_class="us_equity")
        active = client.assets(status="active", asset_class="us_equity")
        n_otc = int((inactive["exchange"].astype(str).str.upper() == "OTC").sum()) if len(inactive) else 0
        report["assets"] = {"inactive": int(len(inactive)), "inactive_otc": n_otc, "active": int(len(active))}
        active_set = set(active["symbol"].astype(str)) if len(active) else set()
        av = pd.read_csv(csv, dtype=str, keep_default_na=False)          # keep_default_na: "NA" is a ticker
        universe = probe_universe(av)
        sample = universe.sample(n=min(args.sample, len(universe)), random_state=args.seed).sort_values("symbol", kind="mergesort")
        rows = []
        for r in sample.to_dict("records"):
            row = probe_stock(client, r, args.feed)
            row["alpaca_status"] = alpaca_status(row["symbol"], inactive)
            row["ticker_now_active"] = row["symbol"] in active_set
            rows.append(row)
    except ap.AlpacaAuthError as e:
        report.update(error=str(e), verdict="NOT_RUN", verdict_text="credentials or data subscription rejected")
        _write_json(out, report)
        print(f"alpaca_check: {e}", file=sys.stderr)
        return 2
    s = summarize(rows)
    s["universe"] = int(len(universe))
    report["probe"] = {**s, "stocks": rows}
    v = verdict(s["covers_life"], s["sampled"])
    report.update(verdict=v, verdict_text=VERDICT_PASS if v == "PASS" else VERDICT_FAIL, requests=getattr(client, "requests", None))
    _write_json(out, report)
    share_txt = "n/a" if s["share_covers_life"] is None else f"{s['share_covers_life']:.1%}"
    print(f"alpaca_check: plan {args.plan}, feed {args.feed}, host {report['trading_base']}")
    print(f"AAPL daily bars since {AAPL_START}: {report['aapl']['rows']} (first {report['aapl']['first']})")
    print(f"delisted probe: {s['covers_life']}/{s['sampled']} cover their life since 2016 ({share_txt}); any bar {s['any_bars']}/{s['sampled']}; "
          f"inactive non-OTC subset {s['inactive_non_otc']['covers_life']}/{s['inactive_non_otc']['n']}")
    print(f"VERDICT {v}: {report['verdict_text']}")
    print(CAVEAT)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
