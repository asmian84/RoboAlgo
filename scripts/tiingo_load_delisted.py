"""
Load price history for delisted US stocks from Tiingo, split-adjust it, and stage it for ingest_delisted_history.py.

    python scripts/tiingo_load_delisted.py --dry-run                      # no token, no network: sizes the job
    TIINGO_TOKEN=... python scripts/tiingo_load_delisted.py --plan free --max-requests 20 --limit-symbols 8
    TIINGO_TOKEN=... python scripts/tiingo_load_delisted.py --plan power
    python scripts/ingest_delisted_history.py --source ~/.roboalgo/tiingo/out

WHAT IT DOES. Takes the AlphaVantage list of delisted listings (~/.roboalgo/delisted/av_delisted.csv), keeps
common stocks delisted in [--from-year, --to-year] that are not already in the survivor cache, and for each
one asks Tiingo for the ticker, then (only if that fails the identity test) for the ticker + "Q" that OTC
tickers carry after a bankruptcy. A ticker is accepted only if its history starts near our listing date and
ends near our delisting date (data_engine/tiingo_provider.identify): tickers are re-used, and a delisted
company's history must never be spliced onto a different company. Prices are RAW from Tiingo and divided by
the product of the later splitFactor values here (split-adjusted, not dividend-adjusted, like the survivor
cache). The result is written under <root>/out/ as delisted_NNNN.parquet with OUR symbol, ready for the ingest.

RESUME AND LIMITS. Everything is cached: <root>/raw/<TICKER>.json holds the raw metadata and prices, so a
rerun asks Tiingo for nothing it already has; <root>/state.json holds one entry per decided symbol (written
after every symbol, atomically); <root>/usage.json counts this month's unique tickers and today's requests
against the plan (free: 45/hour, 900/day, 480 unique symbols/month; power: 9000/hour, 90000/day). A 429
sleeps to the next hour boundary and retries up to MAX_RETRY times, then stops with the state saved. A server
error retries with a growing sleep, then stops. A 401/403 stops at once (nothing is recorded: a wrong token
must not turn every symbol into a permanent "no data").
The token is read from the TIINGO_TOKEN environment variable only, sent as a header, and never printed or
written to any file.

PRE-REGISTRATION. Fixed before any Tiingo data was seen: the identity tolerances (45 days at the end, 800
for the Q ticker, 60 days at the start, 365 for an ipoDate before 2000-01-01), the universe filters and the
request budget. The 2000-01-01 cut-off is an ASSUMPTION about when AlphaVantage's ipoDate stops being a
first-trade date; the run reports how many symbols used the wide tolerance. Nothing is tuned.

KNOWN BIASES. The identity rule has no lower bound on the start date, so a predecessor that held the ticker
before our IPO is accepted if the dates otherwise fit (state records start_before_ipo_days to expose it).
A symbol whose delisting date differs from Tiingo's end date by more than 45 days is rejected even if it is
the right company. Q tickers add months of post-bankruptcy penny prices. If Tiingo misses a split the
adjusted prices contain a false jump: run the artifact detector on the result before trusting any
delisted-inclusive number. On the free plan (480 symbols a month) a universe of thousands of stocks takes
many months; the dry run prints the estimate.
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

# data_engine/__init__ imports the database layer, yfinance and config.settings (which loads .env into the
# environment); this script needs none of it, so the provider module is loaded by path.
_spec = importlib.util.spec_from_file_location("tiingo_provider", ROOT / "data_engine" / "tiingo_provider.py")
tp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tp)

AV_CSV = Path.home() / ".roboalgo" / "delisted" / "av_delisted.csv"
ROOT_DIR = Path.home() / ".roboalgo" / "tiingo"
SURVIVORS = Path.home() / ".roboalgo" / "history_2000"
OUT_BATCH = 200
MAX_RETRY = 3
EST_REQ_PER_SYMBOL = 2.3          # plan estimate: meta + prices for the ticker, plus the Q ticker for those that fail; measure on the first live run
EST_TICKERS_PER_SYMBOL = 2.0      # upper bound: the ticker and its Q ticker; this is what the monthly unique-symbol cap counts
FINAL = frozenset(tp.STATUSES)
AUTH_FAIL = (401, 403)
NEXT_HINT = "next: python scripts/ingest_delisted_history.py --source ~/.roboalgo/tiingo/out"


class _Stop(Exception):
    """End the run now with the state saved; ``reason`` is machine-readable, the message goes to the console."""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


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


def _read_json(path: Path, default):
    return json.loads(path.read_text()) if path.exists() else default


# ── the loader ────────────────────────────────────────────────────────────────────────────────────

def _roll_usage(usage: dict, t: datetime) -> None:
    month, day = t.strftime("%Y-%m"), t.strftime("%Y-%m-%d")
    if usage.get("month") != month:
        usage["month"], usage["tickers"] = month, []
    if usage.get("day") != day:
        usage["day"], usage["day_count"] = day, 0
    usage.setdefault("tickers", [])
    usage.setdefault("day_count", 0)


def _span(meta: dict) -> str:
    return f"{str(meta.get('startDate'))[:10]}..{str(meta.get('endDate'))[:10]} '{meta.get('name')}'"


def run(selected, *, token, plan, max_requests, root, fetch, clock=time.time, sleep=time.sleep, now=None) -> dict:
    """Decide every symbol in ``selected`` (a select_universe frame) that state.json has not decided yet.

    fetch(url, headers) -> (status, body bytes); clock/sleep drive the throttle; now() -> datetime gives the
    month and day for usage.json. Returns counts and, if it stopped early, ``stopped`` (a reason) and ``message``.
    """
    root = Path(root)
    raw_dir, state_path, usage_path = root / "raw", root / "state.json", root / "usage.json"
    raw_dir.mkdir(parents=True, exist_ok=True)
    caps = tp.PLANS[plan]
    now = now or (lambda: datetime.now(timezone.utc))
    headers = tp.auth_headers(token)
    state = _read_json(state_path, {})
    usage = _read_json(usage_path, {})
    _roll_usage(usage, now())
    limiter = tp.RateLimiter(caps["per_hour"], caps["per_day"], clock=clock, sleep=sleep)
    limiter.day_count = int(usage["day_count"])
    result = {"requests": 0, "processed": 0, "already_final": 0, "counts": {s: 0 for s in tp.STATUSES},
              "via_q": 0, "wide_tol": 0, "stopped": None, "message": ""}

    def request(url: str, ticker: str):
        """One HTTP request through the throttle; returns (status, parsed json or None)."""
        for attempt in range(MAX_RETRY + 1):
            if result["requests"] >= max_requests:
                raise _Stop("max_requests", f"reached --max-requests {max_requests}")
            _roll_usage(usage, now())
            if ticker not in usage["tickers"] and len(usage["tickers"]) >= caps["per_month_symbols"]:
                raise _Stop("monthly_symbol_cap", f"this month's {caps['per_month_symbols']} unique symbols on the {plan} plan are used up")
            try:
                limiter.before_request()
            except tp.DailyCapReached as e:
                raise _Stop("daily_cap", str(e))
            try:
                status, body = fetch(url, headers)
            except OSError:
                status, body = 599, b""                                    # network trouble: retried like a server error
            result["requests"] += 1
            if ticker not in usage["tickers"]:
                usage["tickers"].append(ticker)
            usage["day_count"] += 1
            _write_json(usage_path, usage)
            if status in AUTH_FAIL:
                raise _Stop("auth", f"Tiingo answered HTTP {status}: check TIINGO_TOKEN and the plan; nothing was recorded for this symbol")
            if status == 429:
                if attempt == MAX_RETRY:
                    raise _Stop("rate_limited", f"still HTTP 429 after {MAX_RETRY} retries")
                limiter.on_429()
                continue
            if status >= 500:
                if attempt == MAX_RETRY:
                    raise _Stop("server_error", f"HTTP {status} after {MAX_RETRY} retries")
                sleep(2 ** attempt)
                continue
            try:
                data = json.loads(body) if body else None
            except ValueError:
                data = None
            return status, data
        raise AssertionError("unreachable")

    def attempt(ticker: str, ours: dict, is_q: bool) -> dict:
        path = raw_dir / f"{ticker}.json"
        raw = _read_json(path, None)
        if raw is None:
            status, data = request(tp.meta_url(ticker), ticker)
            if status == 200 and isinstance(data, dict):
                raw = {"meta": data, "prices": None}
            elif status == 404:
                raw = {"meta": None, "prices": None}
            else:
                return {"status": "no_data", "reason": f"{ticker}: metadata HTTP {status}"}
            _write_json(path, raw)
        meta = raw.get("meta")
        verdict = tp.identify(meta, ours, is_q)
        if verdict != "ok":
            if not meta:
                return {"status": verdict, "reason": f"{ticker}: not in Tiingo"}
            return {"status": verdict, "reason": f"{ticker}: Tiingo {_span(meta)} vs ours {str(ours['ipoDate'])[:10]}..{str(ours['delistingDate'])[:10]} '{ours['name']}'"}
        if raw.get("prices") is None:
            status, data = request(tp.prices_url(ticker, str(meta["startDate"])[:10], str(meta["endDate"])[:10]), ticker)
            if status != 200 or not isinstance(data, list) or not data:
                why = f"prices HTTP {status}" if status != 200 else "prices empty"
                return {"status": "no_data", "reason": f"{ticker}: {why}"}
            raw["prices"] = data
            _write_json(path, raw)
        df = tp.to_split_adjusted(raw["prices"])
        if df.empty:
            return {"status": "no_data", "reason": f"{ticker}: prices empty"}
        return {"status": "ok", "reason": "", "ticker_used": ticker, "rows": int(len(df)),
                "first": df["date"].iloc[0].date().isoformat(), "last": df["date"].iloc[-1].date().isoformat(),
                "start_before_ipo_days": tp.start_before_ipo_days(meta, ours), "wide_tol": bool(tp.wide_tolerance(ours))}

    order = {"reused": 3, "mismatch": 2, "no_data": 1}
    for ours in selected.to_dict("records"):
        sym = ours["symbol"]
        if state.get(sym, {}).get("status") in FINAL:
            result["already_final"] += 1
            continue
        try:
            a = attempt(sym, ours, False)
            if a["status"] != "ok":
                q = attempt(sym + "Q", ours, True)
                if q["status"] == "ok":
                    a = q
                else:
                    best = max((a, q), key=lambda x: order[x["status"]])
                    a = {"status": best["status"], "reason": f"{a['reason']} | {q['reason']}"}
        except _Stop as stop:
            result["stopped"], result["message"] = stop.reason, str(stop)
            break
        entry = {"status": a["status"], "ticker_used": a.get("ticker_used"), "reason": a["reason"],
                 "rows": a.get("rows", 0), "first": a.get("first"), "last": a.get("last"),
                 "start_before_ipo_days": a.get("start_before_ipo_days"), "wide_tol": a.get("wide_tol")}
        state[sym] = entry
        _write_json(state_path, state)
        result["processed"] += 1
        result["counts"][entry["status"]] += 1
        if entry["status"] == "ok":
            result["via_q"] += int(entry["ticker_used"] != sym)
            result["wide_tol"] += int(bool(entry["wide_tol"]))
    return result


def write_out(state: dict, root) -> list[Path]:
    """Rebuild <root>/out/delisted_NNNN.parquet from EVERY ok symbol in ``state`` (old output files are removed).

    Prices come from the raw cache of the ticker that was accepted, are split-adjusted by
    to_split_adjusted, and carry OUR symbol (not the Q ticker). At most OUT_BATCH symbols per file.
    """
    root = Path(root)
    out = root / "out"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("delisted_*.parquet"):
        old.unlink()
    symbols = sorted(s for s, e in state.items() if e.get("status") == "ok")
    frames = []
    for s in symbols:
        raw = _read_json(root / "raw" / f"{state[s]['ticker_used']}.json", None)
        if not raw or not raw.get("prices"):
            raise FileNotFoundError(f"{s}: state says ok but raw/{state[s]['ticker_used']}.json has no prices; delete its state entry to refetch")
        df = tp.to_split_adjusted(raw["prices"])
        df.insert(0, "symbol", s)
        frames.append(df)
    paths = []
    for k in range(0, len(symbols), OUT_BATCH):
        part = pd.concat(frames[k:k + OUT_BATCH], ignore_index=True)
        part = part.drop_duplicates(["symbol", "date"], keep="last").sort_values(["symbol", "date"]).reset_index(drop=True)
        p = out / f"delisted_{k // OUT_BATCH:04d}.parquet"
        part.to_parquet(p, index=False)
        paths.append(p)
    return paths


# ── command line ──────────────────────────────────────────────────────────────────────────────────

def _survivor_symbols(survivors: Path) -> set[str]:
    files = sorted(survivors.glob("batch_*.parquet"))
    if not files:
        raise SystemExit(f"no survivor cache in {survivors}; run scripts/download_universe_history.py first")
    syms: set[str] = set()
    for f in files:
        syms |= set(pd.read_parquet(f, columns=["symbol"])["symbol"].unique())
    return syms


def _dry_run(universe: pd.DataFrame, n_decided: int, pending: pd.DataFrame, plan: str) -> None:
    caps = tp.PLANS[plan]
    n, p = len(universe), len(pending)
    req = p * EST_REQ_PER_SYMBOL
    tickers = p * EST_TICKERS_PER_SYMBOL
    hours_hour = req / caps["per_hour"]
    hours_day = req / caps["per_day"] * 24 if req > caps["per_day"] else 0.0                  # a cap only binds once the job exceeds it
    hours_month = tickers / caps["per_month_symbols"] * 30 * 24 if tickers > caps["per_month_symbols"] else 0.0
    binding = max((hours_hour, "hourly cap"), (hours_day, "daily cap"), (hours_month, "monthly unique-symbol cap"))
    print("Tiingo delisted loader: DRY RUN (no token needed, no network)")
    print(f"selected: {n:,} delisted common stocks not in the survivor cache | already decided in state: {n_decided:,} | pending: {p:,}")
    print("pending by delisting year: " + ", ".join(f"{y}: {c}" for y, c in pending["delistingDate"].dt.year.value_counts().sort_index().items()))
    print(f"estimated requests: ~{req:,.0f} ({EST_REQ_PER_SYMBOL} per symbol; a plan estimate, measured on the first live run)")
    print(f"plan {plan}: {caps['per_hour']:,}/hour, {caps['per_day']:,}/day, {caps['per_month_symbols']:,} unique symbols/month")
    print(f"  hours at the hourly cap {hours_hour:,.1f} | at the daily cap {hours_day:,.1f} | at the monthly symbol cap {hours_month:,.1f}"
          f" (0 = does not bind; <= {EST_TICKERS_PER_SYMBOL:g} tickers per symbol)")
    print(f"estimated elapsed: ~{binding[0]:,.1f} hours (~{binding[0] / 24:,.1f} days), limited by the {binding[1]}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--plan", choices=sorted(tp.PLANS), default="free")
    ap.add_argument("--max-requests", type=int, default=None, help="stop after this many HTTP requests (default: no limit)")
    ap.add_argument("--limit-symbols", type=int, default=None, help="decide at most this many NEW symbols this run")
    ap.add_argument("--from-year", type=int, default=2010)
    ap.add_argument("--to-year", type=int, default=2025)
    ap.add_argument("--av-csv", default=str(AV_CSV))
    ap.add_argument("--survivors", default=str(SURVIVORS))
    ap.add_argument("--root", default=str(ROOT_DIR))
    ap.add_argument("--dry-run", action="store_true", help="no token, no network, no writes: size the job and exit")
    args = ap.parse_args(argv)

    root = Path(args.root).expanduser()
    av = pd.read_csv(Path(args.av_csv).expanduser(), dtype=str, keep_default_na=False)   # keep_default_na: "NA" is a ticker
    universe = tp.select_universe(av, args.from_year, args.to_year, exclude=_survivor_symbols(Path(args.survivors).expanduser()))
    state = _read_json(root / "state.json", {})
    decided = universe["symbol"].map(lambda s: state.get(s, {}).get("status") in FINAL)
    pending = universe[~decided]
    if args.limit_symbols is not None:
        pending = pending.head(args.limit_symbols)

    if args.dry_run:
        _dry_run(universe, int(decided.sum()), pending, args.plan)
        return 0

    token = os.environ.get("TIINGO_TOKEN", "").strip()
    if not token:
        raise SystemExit("set TIINGO_TOKEN")
    print(f"deciding {len(pending):,} of {len(universe):,} selected symbols ({int(decided.sum()):,} already decided) on the {args.plan} plan")
    res = run(pending, token=token, plan=args.plan, max_requests=args.max_requests if args.max_requests is not None else sys.maxsize,
              root=root, fetch=lambda url, headers: tp.default_fetch(url, headers))
    state = _read_json(root / "state.json", {})
    paths = write_out(state, root)
    sel = {s: state[s] for s in universe["symbol"] if s in state}
    total = {st: sum(1 for e in sel.values() if e["status"] == st) for st in tp.STATUSES}
    print(f"requests this run: {res['requests']:,} | decided this run: {res['processed']:,} | {res['counts']}")
    via_q = sum(1 for s, e in sel.items() if e["status"] == "ok" and e["ticker_used"] != s)
    wide = sum(1 for e in sel.values() if e["status"] == "ok" and e.get("wide_tol"))
    print(f"decided so far (selected universe): {total} | via the Q ticker: {via_q:,} | used the wide start tolerance: {wide:,}")
    if res["stopped"]:
        print(f"STOPPED EARLY ({res['stopped']}): {res['message']}; state is saved, rerun to continue")
    print(f"wrote {len(paths)} file(s) under {root / 'out'} ({total['ok']:,} symbols)")
    print(NEXT_HINT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
