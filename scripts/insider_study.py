"""
Insider purchase clusters (long) and dilution with cash burn (short) on fallen US stocks: a pre-registered test.

    python scripts/insider_study.py --dry-run     # which inputs are present; computes nothing; exit 0
    python scripts/insider_study.py               # the study; writes ~/.roboalgo/reports/insider_study_results.json
    python scripts/insider_study.py --workers 8   # the short arm reads one companyfacts file per CIK in a process pool

THIS DOCSTRING IS THE PRE-REGISTRATION. It repeats docs/INSIDER_STUDY_PREREG.md and was fixed before any SEC
insider data or SEC companyfacts data was seen. The code is built and tested offline on hand-made fixtures. No
parameter is tuned after a result, and no neighbouring setting is tried to rescue a negative one.

QUESTION. Both arms use the same universe of fallen US stocks (below).
  LONG.  When at least two different directors or officers of a fallen stock each buy at least $5,000 of its
         shares on the open market within 30 days, does the stock earn a higher 12-month return than fallen
         stocks with no insider purchase in the same period?
  SHORT. Among fallen stocks, do those that diluted their share count by 20% or more over the past year AND burn
         cash with less than a year of runway lose half their price or more within 12 months more often than
         the stocks that are not flagged?
Each arm has one primary test, so there are 2 primary tests. Everything else is descriptive.

DEFINITIONS (C1 to C12 close the look-ahead and mislabeling holes of the brief; all are fixed).
  Universe. Origins are month-ends 2006-01 to 2023-09. FALLEN is the multibagger study's variant A exactly (C1):
    month-end close from $1 to $10 inclusive, at least 70% below the trailing 60-month high (dd60 <= -0.70),
    AND mean daily dollar volume of the month >= $300,000. The constants are copied and a test asserts equality.
  Qualifying purchase (C4, C5). A code-P row of the NON-derivative table of an original Form 4 or Form 5 (amend-
    ments 4/A and 5/A are excluded) whose reporting owner is a director or an officer. The value of an (owner,
    filing) is the sum over the filing's code-P rows of shares x price; rows with a non-finite or non-positive
    share count or price add nothing. The filing qualifies at a value of $5,000.00 or more (inclusive).
  Cluster (C3). At least 2 distinct qualifying owners AND at least 2 distinct accession numbers among the
    qualifying filings of one issuer (by CIK, never by ticker) within 30 calendar days of FILING dates, window
    [filing date - 29 days, filing date]. One joint Form 4 is never a cluster. The EVENT DATE is the filing
    date of the filing that completes the rule, never the trade date: the trade can precede its filing, and
    only the filing is public.
  Treatment window and the origin close (C2). The month-end close is the close of the last trading day, and a
    Form 4 accepted after 4pm that day is public only after that close. For month-end me, last_weekday(me) is me,
    or me - 1 day on a Saturday, or me - 2 days on a Sunday (exchange holidays are ignored). A stock-month is
    TREATED iff its issuer has a cluster event with me - 89 days <= event date < last_weekday(me): a window of
    90 days, lower bound inclusive. A SINGLE stock-month has a qualifying purchase with me - 89 days <= filing
    date <= me but is not treated. The CONTROL has no qualifying purchase at all in [me - 89 days, me] (the
    stricter window), so a single purchase never sits in the control.
  Symbol to CIK and coverage (C6). A price symbol maps to an issuer at pit_date = last_weekday(me) - 1 day if
    some Form 3/4/5 was filed under that symbol with first_filed <= pit_date and last_filed >= pit_date - 365
    days. Two or more CIKs gives "ambiguous", none gives "unmapped"; both are excluded and counted. Every calendar
    quarter touched by [me - 89 days, me] must have at least one filing in the loaded data, else the row is
    "excluded_no_coverage" (a missing zip must not turn treated months into controls). Foreign private issuers
    never file Form 4, so a ticker with no filing at all is unmapped and never sits in the control forever.
  Short features (C7, C8, C9), read from SEC companyfacts point-in-time by FILED date at pit_date.
    Dilution is True iff shares_now / shares_12m_ago >= 1.20, as a RATIO (now / then - 1 >= 0.20 fails at
    1.2M over 1.0M in floating point). 12 months earlier is pit_date - 365 days. Both counts come from ONE concept:
    dei EntityCommonStockSharesOutstanding first, else us-gaap CommonStockSharesOutstanding.
    Burn is True iff the latest ANNUAL operating cash flow (duration 350 to 380 days, filed on or before
    pit_date; never a later quarter) is negative and cash / |operating cash flow| < 1.0 year. Cash is the cash
    concept plus the first available add-on. Operating cash flow >= 0 gives False whatever the cash.
    A value whose period end is more than 550 days before the date it is read for counts as missing.
    FLAGGED = both known and both True. UNFLAGGED = both known, not both True. UNKNOWN = anything missing; unknown
    rows are excluded from both groups, so missing data never lands in the unflagged group.
  Outcomes. On the grid where a delisted name is held flat at its last close, r_h = C[i+h] / C[i] - 1 needs a
    finite C[i+h] (C11). Origin eligibility and dd60 come from the RAW grid, so a held-flat month after a
    delisting is never an origin. Primary 1 (long): r12 minus the mean r12 of the controls with the same month-end
    (no matching on size, price or volatility). Primary 2 (short): the wipeout rate of flagged minus the wipeout
    rate of unflagged, where a wipeout is min(C[i+1..i+12]) <= 0.5 x C[i] (gaps skipped, undefined unless
    C[i+12] is finite).
  Secondary rows (C12) are descriptive, use alpha 0.05 and never enter a verdict: r6 and r36 excess, the single-
    purchase r12 excess, the long-arm wipeout difference (cluster minus control), the long arm with re-used
    symbols removed.

STATISTICS. Month-clustered bootstrap (whole month-ends resampled), 2000 draws, seed 0, setup_engine/trade_stats.py.
alpha = 0.025 takes the 1.25% and 98.75% quantiles: a two-sided 97.5% interval (Bonferroni over the 2 primary
tests). The lower bound is the interval's lower end. The same quantities clustered by stock are reported as
robustness at 97.5% for both primary outcomes; they do not enter a verdict.

VERDICT RULE (mechanical, applied to the two primaries only).
LONG PASS iff the 97.5% lower bound of the 12-month excess return is > 0; SHORT PASS iff the 97.5% lower bound of (flagged wipeout rate minus unflagged wipeout rate) is > 0; otherwise no demonstrated edge.
A lower bound of exactly 0 is not a PASS. If the interval cannot be computed the verdict is "no demonstrated edge
(interval not computable)". Without a companyfacts directory the short verdict is "not run: no companyfacts
directory". "No demonstrated edge" is the whole negative result; it does not say the effect is zero.

PROVISIONAL (C10). The result is PROVISIONAL iff no delisted names are in the price set: no delisted symbol is
loaded from the Tiingo directory's delisted_*.parquet files or listed in the history directory's terminal.json.
The run decides this from the symbols actually loaded, so an empty terminal.json or Tiingo files holding no
symbols do not lift it. Then the JSON sets provisional to true, the first caveat is the text below, and every
verdict line starts with it:
    PROVISIONAL: survivors only (delisted names missing)
Direction of the bias: unknown for the long arm (dead insider-buying names and dead controls are both missing);
the unfavourable case for the short arm (the flagged names most likely to die are the ones missing).

KNOWN BIASES. Survivors only unless delisted prices are loaded; holding a delisted name flat understates bankruptcies.
The Form 345 layout and the companyfacts layout were UNVERIFIED when this was written (the first real file is the
check: the loader fails with the columns it found). The ticker join relies on the as-reported symbol matching the
price cache; the output reports the mapped share by year. XBRL tagging by small companies starts around 2011, so
the short arm's effective window starts around 2012 (see coverage_by_year). Share counts are as reported, not
split-adjusted. One event treats up to 3 consecutive month-ends and 12-month windows overlap, which month
clustering does not absorb (hence the stock-clustered robustness rows). Exchange holidays and the filing time of
day are not modelled. No costs, borrow or capacity are modelled. No result is claimed here.

INPUTS. --history (batch_*.parquet, and terminal.json if present), --insider-dir (Form 345 quarterly zips),
--facts-dir (CIK##########.json: the data_engine/sec_edgar.py cache layout {"fetched_at", "data"} or a raw SEC
companyfacts file; read with NO network and the cache age ignored), --tiingo-dir (delisted_*.parquet). This script
NEVER touches the network and never calls the download helper.

EXIT CODES. 0 normal run or dry run; 2 a required input (history, insider) is missing; 3 the insider data does not
match the assumed format (the message lists the columns found). None of these paths prints a traceback.
setup_engine/registry/trials.jsonl is deliberately not appended by this script.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from setup_engine import insider_study as IS
from setup_engine import mtf


def _load(name: str, path: Path):
    """Module loaded BY PATH and registered before exec_module (python 3.14 dataclass trap). An already
    registered copy from the same file is reused. ``data_engine/__init__`` is never imported: it pulls in the
    database layer, yfinance and config.settings."""
    old = sys.modules.get(name)
    if old is not None and Path(getattr(old, "__file__", "") or "").resolve() == path.resolve():
        return old
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return mod


M = _load("multibagger_study", ROOT / "scripts" / "multibagger_study.py")
SI = _load("roboalgo_sec_insider", ROOT / "data_engine" / "sec_insider.py")

# ── defaults (fixed) ──────────────────────────────────────────────────────────────────────────────

DEFAULT_HISTORY = mtf.DEFAULT_HISTORY
DEFAULT_INSIDER = SI.INSIDER_DIR
DEFAULT_FACTS = Path.home() / ".roboalgo" / "sec" / "facts"
DEFAULT_TIINGO = Path.home() / ".roboalgo" / "tiingo" / "out"
DEFAULT_OUT = Path.home() / ".roboalgo" / "reports" / "insider_study_results.json"

CI_LABEL = f"{100 * (1 - IS.ALPHA):g}%"                 # "97.5%": the label of both primary intervals and both robustness rows
STUDY = "insider purchase clusters (long) and dilution with cash burn (short) on fallen US stocks"
NOT_RUN_NO_DIR = "no companyfacts directory"
NOT_RUN_NO_FILES = "no companyfacts files in the directory"
EXIT_OK, EXIT_MISSING, EXIT_FORMAT = 0, 2, 3

PREREGISTRATION = (
    f"Fixed before any SEC insider or companyfacts data was seen (docs/INSIDER_STUDY_PREREG.md). Fallen = month-end "
    f"close ${IS.FALLEN_PRICE[0]:.0f}-${IS.FALLEN_PRICE[1]:.0f}, at least {abs(IS.FALLEN_DD):.0%} below the 60-month high, "
    f"mean daily dollar volume >= ${IS.MIN_DVOL:,} (multibagger variant A); origins {IS.ORIGINS[0]} to {IS.ORIGINS[1]}. "
    f"Qualifying purchase = open-market code P on an original Form 4 or 5 by a director or officer, >= ${SI.MIN_VALUE:,.0f} "
    f"per filing. Cluster = >= {SI.CLUSTER_MIN_OWNERS} owners and >= 2 filings within {SI.CLUSTER_WINDOW_DAYS} calendar days "
    f"of FILING dates (event date = filing date). Treated = a cluster event in the {IS.TREAT_WINDOW_DAYS} days up to the "
    f"month-end close, not on the last weekday; control = no qualifying purchase in that window. Short flag = shares "
    f"up >= {round((IS.DILUTION_MIN_RATIO - 1) * 100):d}% over {IS.SHARES_LAG_DAYS} days (ratio form) AND cash runway under "
    f"{IS.RUNWAY_MAX_YEARS:g} year on the latest annual operating cash flow; a missing value is unknown, never unflagged. "
    f"Primary outcomes: 12-month excess return of treated over same-month controls (long); wipeout rate (lowest of the "
    f"next {IS.WIPEOUT_WIN} closes <= {IS.WIPEOUT_LEVEL:g}x) of flagged minus unflagged (short). Month-clustered "
    f"bootstrap, {IS.N_BOOT} draws, seed {IS.SEED}, alpha {IS.ALPHA} = a {CI_LABEL} interval. LONG PASS iff the {CI_LABEL} "
    f"lower bound > 0; SHORT PASS iff the {CI_LABEL} lower bound > 0; otherwise no demonstrated edge. PROVISIONAL "
    f"when no delisted names are in the price set. Nothing is tuned.")

CAVEATS = (
    "Delisted names held flat at the last close understate bankruptcy losses.",
    "The Form 345 and companyfacts layouts were unverified when the code was written; check the coverage block.",
    "Ticker to CIK join is as-reported and point-in-time; see coverage.mapped_share_by_year and n_reused_symbols.",
    "Small-company XBRL starts around 2011: the short arm's effective window starts around 2012 (short.coverage_by_year).",
    "Share counts are as reported, not split-adjusted; debt and leases are not in the burn test.",
    "One event treats up to 3 consecutive month-ends and 12-month windows overlap; see the stock-clustered robustness rows.",
    "Exchange holidays and filing time of day are not modelled; no costs, borrow or capacity.",
)


# ── inputs ────────────────────────────────────────────────────────────────────────────────────────

def _count(d: Path, pattern: str) -> int:
    return sum(1 for _ in d.glob(pattern)) if d.is_dir() else 0


def inputs_report(history, insider_dir, facts_dir, tiingo_dir) -> dict:
    """Which inputs exist, judged by file names plus the symbol count in terminal.json. Required: history, insider.
    The provisional flag here is a first read for the dry run; run() re-decides it from the symbols actually loaded."""
    h, ins = Path(history).expanduser(), Path(insider_dir).expanduser()
    f, t = Path(facts_dir).expanduser(), Path(tiingo_dir).expanduser()
    n_batches = _count(h, "batch_*.parquet")
    terminal_json = (h / "terminal.json").is_file()
    try:
        n_terminal = len(M.load_terminal(h)) if terminal_json else 0
    except (ValueError, TypeError):          # an unreadable terminal.json lists no delisted names
        n_terminal = 0
    insider = SI.inventory(ins)
    n_tiingo = _count(t, "delisted_*.parquet")
    rep = {
        "history": {"path": str(h), "exists": h.is_dir(), "n_batches": n_batches, "terminal_json": terminal_json,
                    "n_terminal": n_terminal},
        "insider": insider,
        "facts": {"path": str(f), "exists": f.is_dir(), "n_files": _count(f, "CIK*.json")},
        "tiingo": {"path": str(t), "exists": t.is_dir(), "n_files": n_tiingo},
        "provisional": not (n_tiingo > 0 or n_terminal > 0),
    }
    rep["missing_required"] = [name for name, ok in (("history", n_batches > 0), ("insider", insider["n_zips"] > 0))
                               if not ok]
    return rep


def _status_line(name: str, ok: bool, path: str, detail: str) -> str:
    return f"  {name:<8} {'present' if ok else 'MISSING'}  {path}  ({detail})"


def print_inputs(rep: dict) -> None:
    h, i, f, t = rep["history"], rep["insider"], rep["facts"], rep["tiingo"]
    if h["exists"]:
        h_detail = f"{h['n_batches']} batch files, " + ("terminal.json present" if h["terminal_json"] else "no terminal.json")
    else:
        h_detail = "directory not found"
    i_detail = (f"{i['n_zips']} zips, quarters {i['quarters'][0]} to {i['quarters'][-1]}" if i["quarters"]
                else f"{i['n_zips']} zips") if i["exists"] else "directory not found"
    f_detail = f"{f['n_files']} CIK files" if f["exists"] else "directory not found"
    t_detail = f"{t['n_files']} delisted_*.parquet files" if t["exists"] else "directory not found"
    print(_status_line("history", h["n_batches"] > 0, h["path"], h_detail))
    print(_status_line("insider", i["n_zips"] > 0, i["path"], i_detail))
    print(_status_line("facts", f["n_files"] > 0, f["path"], f_detail))
    print(_status_line("tiingo", t["n_files"] > 0, t["path"], t_detail))
    if rep["provisional"]:
        print(IS.PROVISIONAL)


# ── prices ────────────────────────────────────────────────────────────────────────────────────────

def load_prices(history, tiingo_dir, workers):
    """(C_raw, DV, C_held, terminal). Survivor rows win a duplicate stock-month over a Tiingo row; Tiingo symbols are
    terminal (held flat at the last close). Months after the last complete month are dropped."""
    history = Path(history).expanduser()
    data_end = M._data_end(M._batch_files(history))
    panel = M.build_panel(history, workers)
    terminal = set(M.load_terminal(history))
    tdir = Path(tiingo_dir).expanduser()
    tfiles = sorted(tdir.glob("delisted_*.parquet")) if tdir.is_dir() else []
    if tfiles:
        extra = pd.concat([M._panel_batch(str(f)) for f in tfiles], ignore_index=True)
        panel = pd.concat([panel, extra], ignore_index=True).drop_duplicates(["symbol", "month"], keep="first")
        terminal |= set(extra["symbol"])
    panel = panel[panel["month"] <= M.last_complete_month(data_end)]
    panel = panel.sort_values(["symbol", "month"]).reset_index(drop=True)
    C_raw, DV = M.wide(panel, "close"), M.wide(panel, "dvol")
    return C_raw, DV, M.hold_terminal_flat(C_raw, frozenset(terminal)), frozenset(terminal)


# ── companyfacts ──────────────────────────────────────────────────────────────────────────────────

def _read_facts(facts_dir, cik):
    """(facts or None, status) with status in {"ok", "missing", "unreadable"}. No network; cache age ignored."""
    path = Path(facts_dir) / f"CIK{cik}.json"
    if not path.is_file():
        return None, "missing"
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, "unreadable"
    if isinstance(blob, dict) and "fetched_at" in blob and isinstance(blob.get("data"), dict):
        blob = blob["data"]
    return (blob, "ok") if isinstance(blob, dict) else (None, "unreadable")


def load_facts(facts_dir, cik):
    """Companyfacts for a 10-digit CIK from facts_dir/CIK##########.json, unwrapping the EdgarClient cache
    {"fetched_at", "data"}. None if the file is missing or unreadable. NO network, and the TTL is ignored on purpose."""
    return _read_facts(facts_dir, cik)[0]


def _short_one(task):
    """Worker: all month-ends of one CIK -> flags per input row. Module level so a process pool can pickle it."""
    cik, rows, month_ends, facts_dir = task
    facts, status = _read_facts(facts_dir, cik)
    me = pd.DatetimeIndex(month_ends)
    ue = me.unique()
    fl = IS.short_flags(facts, ue).iloc[ue.get_indexer(me)]
    return cik, rows, list(fl["dilution"]), list(fl["burn"]), list(fl["short_class"]), status


def _fmt_secs(s: float) -> str:
    return f"{s:.0f}s" if s < 90 else (f"{s / 60:.1f}m" if s < 5400 else f"{s / 3600:.1f}h")


class Progress:
    """Prints 'label: done/total (pct) elapsed .. eta ..' at most every ``every`` seconds, and at the end."""

    def __init__(self, label: str, total: int, every: float = 10.0, clock=time.monotonic, out=print):
        self.label, self.total, self.every, self.clock, self.out = label, total, every, clock, out
        self.t0 = self.last = clock()

    def update(self, done: int) -> None:
        now = self.clock()
        if done < self.total and now - self.last < self.every:
            return
        self.last = now
        elapsed = now - self.t0
        eta = elapsed / done * (self.total - done) if done else 0.0
        pct = 100.0 * done / self.total if self.total else 100.0
        self.out(f"  {self.label}: {done}/{self.total} ({pct:.0f}%) elapsed {_fmt_secs(elapsed)} eta {_fmt_secs(eta)}")


def short_features(sub: pd.DataFrame, facts_dir, workers: int = 1, every: float = 10.0):
    """Dilution, burn and short_class per row of ``sub`` (needs cik and month_end), one task per CIK.
    Returns (dilution, burn, short_class, stats). A process pool is used when workers > 1."""
    n = len(sub)
    dil, brn = np.full(n, None, dtype=object), np.full(n, None, dtype=object)
    cls = np.full(n, "unknown", dtype=object)
    me = sub["month_end"].to_numpy()
    tasks = [(cik, rows, me[rows], str(facts_dir)) for cik, rows in sub.groupby("cik").indices.items()]
    prog = Progress("short arm, CIKs", len(tasks), every=every)
    if workers > 1 and len(tasks) > 1:
        chunk = max(1, min(25, len(tasks) // (workers * 8)))
        with mp.get_context("fork").Pool(min(workers, len(tasks))) as pool:
            results = []
            for i, res in enumerate(pool.imap_unordered(_short_one, tasks, chunksize=chunk), 1):
                results.append(res)
                prog.update(i)
    else:
        results = []
        for i, task in enumerate(tasks, 1):
            results.append(_short_one(task))
            prog.update(i)
    status = {"ok": 0, "missing": 0, "unreadable": 0}
    for _cik, rows, d, b, c, st in results:
        status[st] += 1
        for r, dv, bv, cv in zip(rows, d, b, c):
            dil[r], brn[r], cls[r] = dv, bv, cv
    stats = {"n_rows_with_cik": int(n), "n_ciks": len(tasks), "n_facts_found": status["ok"],
             "n_facts_missing": status["missing"], "n_facts_unreadable": status["unreadable"]}
    return dil, brn, cls, stats


# ── the study ─────────────────────────────────────────────────────────────────────────────────────

def _constants() -> dict:
    return {
        "ORIGINS": IS.ORIGINS, "FALLEN_PRICE": IS.FALLEN_PRICE, "FALLEN_DD": IS.FALLEN_DD, "MIN_DVOL": IS.MIN_DVOL,
        "TREAT_WINDOW_DAYS": IS.TREAT_WINDOW_DAYS, "HORIZONS": IS.HORIZONS, "PRIMARY_H": IS.PRIMARY_H,
        "WIPEOUT_WIN": IS.WIPEOUT_WIN, "WIPEOUT_LEVEL": IS.WIPEOUT_LEVEL,
        "DILUTION_MIN_RATIO": IS.DILUTION_MIN_RATIO, "RUNWAY_MAX_YEARS": IS.RUNWAY_MAX_YEARS,
        "ANNUAL_DAYS": IS.ANNUAL_DAYS, "MAX_STALE_DAYS": IS.MAX_STALE_DAYS, "SHARES_LAG_DAYS": IS.SHARES_LAG_DAYS,
        "N_BOOT": IS.N_BOOT, "SEED": IS.SEED, "ALPHA": IS.ALPHA, "ALPHA_SECONDARY": IS.ALPHA_SECONDARY,
        "MIN_VALUE": SI.MIN_VALUE, "CLUSTER_WINDOW_DAYS": SI.CLUSTER_WINDOW_DAYS,
        "CLUSTER_MIN_OWNERS": SI.CLUSTER_MIN_OWNERS, "SYMBOL_LOOKBACK_DAYS": SI.SYMBOL_LOOKBACK_DAYS,
        "PURCHASE_FORMS": SI.PURCHASE_FORMS, "QUALIFYING_ROLES": sorted(SI.QUALIFYING_ROLES),
    }


def _say(msg: str) -> None:
    print(msg, flush=True)


def run(args, rep: dict) -> dict:
    """The whole study on the real inputs. Raises SI.FormatError when the insider zips do not match the layout."""
    prov = bool(rep["provisional"])
    _say("loading insider data ...")
    purchases, spans, qstats = SI.load_dir(args.insider_dir)
    events = SI.clusters(purchases)
    _say(f"  {len(purchases)} qualifying (owner, filing) purchases, {len(events)} cluster events, {len(spans)} symbol spans")

    _say("loading prices ...")
    C_raw, DV, C_held, terminal = load_prices(args.history, args.tiingo_dir, args.workers)
    prov = prov or not terminal              # C10: decided by the delisted symbols actually loaded, not file names
    dd60 = M.features(C_raw, DV)["dd60"]
    sm = IS.stock_months(C_raw, DV, dd60, C_held)
    _say(f"  {len(sm)} fallen stock-months, {len(terminal)} terminal symbols")

    sm["pit_date"] = IS.pit_dates(sm["month_end"])
    mp_ = SI.map_symbols(spans, pd.DataFrame({"symbol": sm["symbol"], "date": sm["pit_date"]}))
    sm["cik"] = mp_["cik"].to_numpy(dtype=object)
    sm["ambiguous"] = mp_["ambiguous"].to_numpy(dtype=bool)
    covered = SI.covered_quarters(spans)
    sm = IS.assign_long(sm, purchases, events, covered)
    long = IS.long_arm(sm, reused=SI.reused_symbols(spans))
    long["group_counts"] = {g: int((sm["group"] == g).sum()) for g in IS.GROUPS}

    mapped = (sm["cik"].notna() & ~sm["ambiguous"]).to_numpy()
    by_year = pd.Series(mapped).groupby(sm["month_end"].dt.year.to_numpy()).mean()

    facts_dir = Path(args.facts_dir).expanduser()
    if not facts_dir.is_dir():
        short, short_reason = {"ran": False, "reason": NOT_RUN_NO_DIR}, NOT_RUN_NO_DIR
    elif rep["facts"]["n_files"] == 0:
        short, short_reason = {"ran": False, "reason": NOT_RUN_NO_FILES}, NOT_RUN_NO_FILES
    else:
        _say("short arm: reading companyfacts per CIK ...")
        sub = sm[sm["cik"].notna()].reset_index(drop=True)
        sub["dilution"], sub["burn"], sub["short_class"], fstats = short_features(sub, facts_dir, args.workers)
        short = {"ran": True, **IS.short_arm(sub), "facts": fstats}
        short_reason = None

    verdicts = {"long": IS.verdict(long["primary"]["lo"], provisional=prov),
                "short": (IS.verdict(None, provisional=prov, not_run_reason=short_reason) if short_reason
                          else IS.verdict(short["primary"]["lo"], provisional=prov))}
    return {
        "study": STUDY,
        "generated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "provisional": prov,
        "caveats": ([IS.PROVISIONAL] if prov else []) + list(CAVEATS),
        "preregistration": PREREGISTRATION,
        "constants": _constants(),
        "inputs": rep,
        "coverage": {
            "n_fallen_stock_months": int(len(sm)),
            "mapped_share_by_year": {int(y): float(v) for y, v in by_year.items()},
            "n_reused_symbols": len(SI.reused_symbols(spans)),
            "insider_quarters": sorted(covered),
            "quarter_stats": qstats,
            "n_purchases": int(len(purchases)),
            "n_cluster_events": int(len(events)),
            "n_terminal_symbols": len(terminal),
        },
        "long": long,
        "short": short,
        "verdicts": verdicts,
    }


# ── output ────────────────────────────────────────────────────────────────────────────────────────

def write_json(obj, path) -> None:
    """Strict JSON (no NaN or Infinity), written to a temp file and then moved into place."""
    path = Path(path)
    text = json.dumps(M._jsonable(obj), indent=2, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:+.2f}%"


def _interval_line(label: str, d: dict) -> str:
    return (f"  {label}: mean {_pct(d.get('mean'))}, {CI_LABEL} interval [{_pct(d.get('lo'))}, {_pct(d.get('hi'))}], "
            f"n={d.get('n')}, clusters={d.get('n_clusters')}")


def print_result(result: dict, out_path) -> None:
    lg, sh, v = result["long"], result["short"], result["verdicts"]
    if result["provisional"]:
        print(IS.PROVISIONAL)
    c = lg["counts"]
    print(f"LONG  groups: cluster {c['n_cluster']} (with r12 {c['n_cluster_with_r12']}, matched {c['n_matched']}), "
          f"control {c['n_control']}, single {c['n_single']}, excluded: unmapped {c['n_excluded_unmapped']}, "
          f"ambiguous {c['n_excluded_ambiguous']}, no coverage {c['n_excluded_no_coverage']}")
    print(_interval_line("primary (excess r12, month clusters)", lg["primary"]))
    print(_interval_line("robustness (excess r12, stock clusters)", lg["robust_stock_clusters"]))
    if sh.get("ran"):
        k = sh["counts"]
        print(f"SHORT groups: flagged {k['n_flagged']}, unflagged {k['n_unflagged']}, unknown {k['n_unknown']}, "
              f"no outcome {k['n_no_outcome']}; first classified month {sh['first_classified_month']}")
        print(_interval_line("primary (wipeout rate flagged minus unflagged, month clusters)", sh["primary"]))
        print(_interval_line("robustness (wipeout difference, stock clusters)", sh["robust_stock_clusters"]))
    else:
        print(f"SHORT not run: {sh.get('reason')}")
    print(f"LONG  verdict: {v['long']}")
    print(f"SHORT verdict: {v['short']}")
    print(f"wrote {out_path}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--history", default=str(DEFAULT_HISTORY), help="directory with batch_*.parquet (and terminal.json)")
    ap.add_argument("--insider-dir", default=str(DEFAULT_INSIDER), help="directory with the Form 345 quarterly zips")
    ap.add_argument("--facts-dir", default=str(DEFAULT_FACTS), help="directory with CIK##########.json companyfacts")
    ap.add_argument("--tiingo-dir", default=str(DEFAULT_TIINGO), help="directory with delisted_*.parquet")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="results JSON")
    ap.add_argument("--workers", type=int, default=1, help="process pool size for the short arm and the price panel")
    ap.add_argument("--dry-run", action="store_true", help="report which inputs are present and compute nothing")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    args.workers = max(1, int(args.workers))
    args.history = str(Path(args.history).expanduser())
    args.insider_dir = str(Path(args.insider_dir).expanduser())
    args.facts_dir = str(Path(args.facts_dir).expanduser())
    args.tiingo_dir = str(Path(args.tiingo_dir).expanduser())
    out_path = Path(args.out).expanduser()

    rep = inputs_report(args.history, args.insider_dir, args.facts_dir, args.tiingo_dir)
    print_inputs(rep)
    if args.dry_run:
        print("dry run: nothing computed")
        return EXIT_OK
    if rep["missing_required"]:
        print(f"cannot run: missing required input(s): {', '.join(rep['missing_required'])}")
        return EXIT_MISSING
    try:
        result = run(args, rep)
    except SI.FormatError as e:
        print(f"insider data format mismatch: {e}")
        return EXIT_FORMAT
    write_json(result, out_path)
    print_result(result, out_path)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
