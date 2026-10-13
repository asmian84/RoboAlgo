"""
RoboAlgo - SEC EDGAR provider: point-in-time company fundamentals and dilution filings.

What it answers, for a company and a date: how much cash, how much debt, how fast is it
burning cash (runway in months), how fast is the share count growing, revenue growth and
gross margin, and which dilution-type filings (shelf registrations and prospectus
supplements) it made recently. Everything comes from the SEC's free XBRL "companyfacts"
and "submissions" endpoints.

*** POINT-IN-TIME RULE: EVERYTHING IS KEYED ON THE `filed` DATE ***
-----------------------------------------------------------------
A figure is "known" on a date only if the filing that carried it was filed on or before that
date. companyfacts keeps a separate copy of each fact from every filing that reported it, so
for each period (start, end) we keep the copy with the LATEST `filed` that is still <= the
date (tie on `filed`: the greater accession number). A restatement filed after the date is
invisible. Never use the `frame` field (the "CY2020Q1I" labels): the SEC attaches it to ONE
copy per period, the most recent restatement, so it leaks the future. facts_table() drops it.

Quarterly values are derived by differencing year-to-date figures (Q2 = 6M - 3M, Q3 = 9M - 6M,
Q4 = FY - 9M), because cash-flow statements are reported year-to-date and 10-K filings carry no
fourth-quarter column. A direct 3-month row, when present, wins over the difference.

Known limits (each one fails to NaN, not to a wrong number, unless noted)
-------------------------------------------------------------------------
* 52/53-week years are fine (durations are binned 80-100, 170-190, 260-285, 350-380 days), but a
  change of fiscal year end (transition periods with odd lengths) drops those quarters, so ttm()
  is NaN around it. Expect this for small caps.
* Only the us-gaap and dei taxonomies are read. Foreign private issuers (20-F/40-F, ifrs-full)
  return almost nothing. Money is read from unit "USD" only, shares from unit "shares" only.
* The dei cover-page share count is dated near the filing date, not the balance-sheet date, and a
  multi-class company can report one class only. share_growth_yoy is approximate.
* Revenue concepts changed around 2018 (ASC 606). Both revenue windows are computed from the SAME
  concept, so a filer that switched concepts shows NaN growth rather than a mixed-concept number.
* The latest value of a concept is used however old its period end is: a tag a company stopped
  using years ago still reports its last value. period_end in the snapshot is the date of the
  CASH figure, so compare it with `as_of` before trusting a snapshot.
* Debt is best effort. Operating and finance leases and convertible notes tagged under other
  concepts are missed, so "net cash" can be too rosy for small caps that fund themselves with
  converts, which are exactly the dilution-prone names this is meant to flag.
* 424B3 filings are mostly resale registrations and merger prospectuses, not at-the-market sales,
  and 8-K and Form 4 counts are not dilution signals. potential_atm (a shelf registration PLUS a
  prospectus supplement, 424B5 being the meaningful one) is a pointer to go read the filings,
  not a finding.

Network and credentials
-----------------------
SEC_USER_AGENT must be set in the environment to "Name contact@example.org" (a real contact:
the SEC answers 403 without one). There is NO default and no address is stored in this code.
Requests are throttled to MAX_PER_SEC (the SEC's published limit is 10 per second) and cached on
disk. Every network function goes through an injectable `fetch(url, headers) -> (status, body)`
so tests never touch the network.
"""
from __future__ import annotations

import gzip
import json
import os
import re
import time
import zlib
from datetime import date as _today_date
from pathlib import Path

import numpy as np
import pandas as pd

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
SUBM_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
CACHE_DIR = Path.home() / ".roboalgo" / "sec"
TTL = {"tickers": 86400, "facts": 86400, "submissions": 21600}   # seconds
MAX_PER_SEC = 8
SHELF_FORMS = ("S-3", "S-3ASR")
SUPPLEMENT_FORMS = ("424B5", "424B3")
DILUTION_FORMS = SHELF_FORMS + SUPPLEMENT_FORMS
FORMS_DEFAULT = ("S-3", "S-3ASR", "424B5", "424B3", "8-K", "4")

CASH = "CashAndCashEquivalentsAtCarryingValue"
CASH_ADD = ("ShortTermInvestments", "MarketableSecuritiesCurrent",
            "AvailableForSaleSecuritiesDebtSecuritiesCurrent")
OCF = "NetCashProvidedByUsedInOperatingActivities"
REVENUE = ("RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet")
GROSS = "GrossProfit"
COST = ("CostOfRevenue", "CostOfGoodsAndServicesSold")
SHARES_DEI = "EntityCommonStockSharesOutstanding"
SHARES_GAAP = "CommonStockSharesOutstanding"
LTD_NONCURRENT = "LongTermDebtNoncurrent"
LTD = "LongTermDebt"
LTD_CURRENT = "LongTermDebtCurrent"
DEBT_CURRENT = "DebtCurrent"
ST_BORROW = "ShortTermBorrowings"

TAXONOMIES = ("us-gaap", "dei")
# (class, min days, max days), inclusive. Class 1 = a 3-month quarter, 2 = 6M YTD, 3 = 9M YTD, 4 = 12M.
DURATION_BINS = ((1, 80, 100), (2, 170, 190), (3, 260, 285), (4, 350, 380))
QUARTER_GAP = (80, 100)          # days between consecutive quarter ends
FACT_COLUMNS = ["concept", "taxonomy", "unit", "start", "end", "fy", "fp", "form", "filed", "val", "accn"]
FILING_COLUMNS = ["form", "filingDate", "accessionNumber", "primaryDocument"]

_SNAPSHOT_CONCEPTS = frozenset(
    (CASH, OCF, GROSS, SHARES_DEI, SHARES_GAAP, LTD_NONCURRENT, LTD, LTD_CURRENT, DEBT_CURRENT, ST_BORROW)
    + CASH_ADD + REVENUE + COST)

_NAN = float("nan")


# -- identity of the caller -------------------------------------------------------------------

def user_agent() -> str:
    """The User-Agent the SEC requires, read from SEC_USER_AGENT. There is deliberately no default."""
    ua = os.environ.get("SEC_USER_AGENT", "").strip()
    if not ua or "@" not in ua:
        raise RuntimeError("SEC_USER_AGENT must be set to 'Name contact@example.org' "
                           "(the SEC returns 403 without a contact in the User-Agent)")
    return ua


# -- network: throttle, fetch, cache ----------------------------------------------------------

class Throttle:
    """Spaces calls at least 1/max_per_sec apart. Time sources are injectable for tests."""

    def __init__(self, max_per_sec: float = MAX_PER_SEC, clock=time.monotonic, sleep=time.sleep):
        self.gap = 1.0 / max_per_sec
        self.clock = clock
        self.sleep = sleep
        self.next_ok = 0.0

    def wait(self) -> None:
        now = self.clock()
        if now < self.next_ok:
            self.sleep(self.next_ok - now)
            now = self.clock()
        self.next_ok = max(now, self.next_ok) + self.gap


def _decode_body(body: bytes, encoding: str | None) -> bytes:
    """Undo gzip/deflate if the server used it (by header, or by the gzip magic bytes)."""
    enc = (encoding or "").lower()
    if "gzip" in enc or body[:2] == b"\x1f\x8b":
        return gzip.decompress(body)
    if "deflate" in enc:
        try:
            return zlib.decompress(body)
        except zlib.error:
            return zlib.decompress(body, -zlib.MAX_WBITS)
    return body


def default_fetch(url: str, headers: dict) -> tuple[int, bytes]:
    """GET `url` -> (status, body bytes). HTTP errors are returned, not raised."""
    import urllib.error
    import urllib.request
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, _decode_body(resp.read(), resp.headers.get("Content-Encoding"))
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            raw = _decode_body(raw, e.headers.get("Content-Encoding") if e.headers else None)
        except Exception:
            pass
        return e.code, raw


def _safe_key(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(key))


class EdgarClient:
    """Throttled, disk-cached JSON client. `fetch(url, headers) -> (status, body)` is injectable."""

    def __init__(self, cache_dir=CACHE_DIR, fetch=None, clock=time.monotonic, wall=time.time, sleep=time.sleep):
        self.cache_dir = Path(cache_dir)
        self.fetch = fetch or default_fetch
        self.wall = wall
        self.throttle = Throttle(MAX_PER_SEC, clock=clock, sleep=sleep)

    def _path(self, kind: str, key: str) -> Path:
        return self.cache_dir / kind / f"{_safe_key(key)}.json"

    def get_json(self, url: str, kind: str, key: str):
        """Cached JSON for `url`. A cache entry younger than TTL[kind] seconds is returned with no request."""
        if kind not in TTL:
            raise ValueError(f"unknown cache kind {kind!r}; expected one of {sorted(TTL)}")
        path = self._path(kind, key)
        if path.exists():
            try:
                blob = json.loads(path.read_text(encoding="utf-8"))
                if self.wall() - float(blob["fetched_at"]) < TTL[kind]:
                    return blob["data"]
            except (ValueError, KeyError, TypeError, OSError):
                pass                                     # unreadable cache entry: treat as stale
        headers = {"User-Agent": user_agent(), "Accept-Encoding": "gzip, deflate"}
        self.throttle.wait()
        status, body = self.fetch(url, headers)
        if status != 200:
            hint = (" (the SEC rejects requests without a contact in SEC_USER_AGENT)" if status == 403 else "")
            raise RuntimeError(f"SEC request failed with HTTP {status} for {url}{hint}")
        data = json.loads(body)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps({"fetched_at": self.wall(), "data": data}), encoding="utf-8")
        os.replace(tmp, path)
        return data


# -- identifiers ------------------------------------------------------------------------------

def pad_cik(x) -> str:
    """CIK as a 10-digit zero-padded string. Accepts ints, whole floats and strings such as 'CIK0000001234'."""
    if isinstance(x, (float, np.floating)):
        x = "" if np.isnan(x) else str(int(x))
    return re.sub(r"\D", "", str(x)).zfill(10)


def _norm_ticker(t) -> str:
    return str(t).strip().upper().replace(".", "-")


def ticker_to_cik(ticker, table=None, client=None):
    """10-digit CIK for a ticker, or None. 'BRK.B' and 'BRK-B' both match the SEC's 'BRK-B'.

    `table` is the parsed company_tickers.json ({"0": {"cik_str", "ticker", "title"}, ...}); if
    omitted it is fetched (cached for a day) through `client`.
    """
    if table is None:
        table = (client or EdgarClient()).get_json(TICKERS_URL, "tickers", "company_tickers")
    entries = table.values() if isinstance(table, dict) else table
    want = _norm_ticker(ticker)
    for e in entries:
        if _norm_ticker(e.get("ticker", "")) == want:
            return pad_cik(e["cik_str"])
    return None


def cik_from_sp500(df: pd.DataFrame, table=None, client=None) -> pd.Series:
    """CIK per row: df['cik'] when present and non-empty, else looked up from df['symbol'] (None if unknown).

    The ticker table is fetched at most once, and only if some row needs the fallback.
    """
    out, tbl = [], table
    for i in range(len(df)):
        raw = df["cik"].iloc[i] if "cik" in df.columns else None
        digits = "" if raw is None or (isinstance(raw, float) and np.isnan(raw)) else re.sub(r"\D", "", str(raw))
        if digits:
            out.append(pad_cik(digits))
            continue
        if tbl is None:
            tbl = (client or EdgarClient()).get_json(TICKERS_URL, "tickers", "company_tickers")
        out.append(ticker_to_cik(df["symbol"].iloc[i], table=tbl))
    return pd.Series(out, index=df.index, dtype=object, name="cik")


# -- XBRL facts -------------------------------------------------------------------------------

def company_facts(cik, client=None) -> dict:
    """The raw companyfacts JSON for a CIK (cached for a day)."""
    padded = pad_cik(cik)
    return (client or EdgarClient()).get_json(FACTS_URL.format(cik=padded), "facts", f"CIK{padded}")


def submissions(cik, client=None) -> dict:
    """The raw submissions JSON for a CIK (cached for six hours)."""
    padded = pad_cik(cik)
    return (client or EdgarClient()).get_json(SUBM_URL.format(cik=padded), "submissions", f"CIK{padded}")


def facts_table(facts: dict, concepts=None) -> pd.DataFrame:
    """Flatten companyfacts into one row per reported value.

    Columns: concept, taxonomy, unit, start (NaT for instants), end, fy, fp, form, filed, val, accn.
    There is deliberately NO `frame` column: frames reflect later restatements and would leak the
    future into a point-in-time read. Only the us-gaap and dei taxonomies are read. `concepts`
    (an iterable of names) restricts the output to those concepts.
    """
    want = None if concepts is None else set(concepts)
    rows = []
    for tax in TAXONOMIES:
        for concept, body in ((facts or {}).get("facts", {}).get(tax, {}) or {}).items():
            if want is not None and concept not in want:
                continue
            for unit, items in (body.get("units", {}) or {}).items():
                for it in items:
                    if "end" not in it or "val" not in it or "filed" not in it:
                        continue
                    rows.append((concept, tax, unit, it.get("start"), it["end"], it.get("fy"), it.get("fp"),
                                 it.get("form"), it["filed"], it["val"], it.get("accn") or ""))
    df = pd.DataFrame(rows, columns=FACT_COLUMNS)
    for c in ("start", "end", "filed"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    df["val"] = pd.to_numeric(df["val"], errors="coerce")
    df["fy"] = pd.to_numeric(df["fy"], errors="coerce").astype("Int64")
    df = df[df["end"].notna() & df["filed"].notna() & df["val"].notna()]
    return df.reset_index(drop=True)


def _ts(d) -> pd.Timestamp:
    return pd.Timestamp(d).normalize()


def pit(table: pd.DataFrame, date, concept: str) -> pd.DataFrame:
    """The values of `concept` that were public on `date`, one row per period (start, end).

    Only rows with filed <= date are visible. Per period the row with the LATEST `filed` wins (tie:
    the greater accession number), so a restatement filed after `date` is invisible.
    """
    d = _ts(date)
    t = table[(table["concept"] == concept) & (table["filed"] <= d)]
    t = t.sort_values(["filed", "accn"], kind="mergesort")
    t = t.drop_duplicates(subset=["start", "end"], keep="last")
    return t.sort_values(["end", "start"], kind="mergesort").reset_index(drop=True)


def as_of(table: pd.DataFrame, date, concept: str, kind: str = "instant"):
    """(period end, value) of the latest period of `concept` known on `date`; (None, nan) if none.

    kind="instant" reads balance-sheet style rows (no start); kind="duration" reads rows with a
    start and, on a tie of period end, takes the longest period.
    """
    p = pit(table, date, concept)
    if kind == "instant":
        p = p[p["start"].isna()]
    elif kind == "duration":
        p = p[p["start"].notna()]
    else:
        raise ValueError("kind must be 'instant' or 'duration'")
    if p.empty:
        return None, _NAN
    if kind == "duration":
        p = p.assign(_dur=(p["end"] - p["start"]).dt.days).sort_values(["end", "_dur"], kind="mergesort")
    row = p.iloc[-1]
    return row["end"], float(row["val"])


def _dur_class(days: int) -> int:
    for cls, lo, hi in DURATION_BINS:
        if lo <= days <= hi:
            return cls
    return 0


def quarterly(table: pd.DataFrame, date, concept: str) -> pd.Series:
    """Single-quarter values of a flow concept known on `date`, indexed by quarter end.

    A direct 3-month row (start, end) is used when present. Otherwise the quarter is the
    year-to-date figure minus the previous year-to-date figure that has the same fiscal-year
    start (Q2 = 6M - 3M, Q3 = 9M - 6M, Q4 = FY - 9M). Rows whose duration fits no bin are ignored.
    """
    p = pit(table, date, concept)
    p = p[p["start"].notna()]
    out: dict[pd.Timestamp, float] = {}
    if not p.empty:
        p = p.assign(cls=[_dur_class(int(x)) for x in (p["end"] - p["start"]).dt.days])
        p = p[p["cls"] > 0]
        for r in p[p["cls"] == 1].itertuples():
            out[r.end] = float(r.val)
        for r in p[p["cls"] > 1].itertuples():
            if r.end in out:
                continue                                  # a direct 3M row wins over a difference
            prev = p[(p["start"] == r.start) & (p["cls"] == r.cls - 1) & (p["end"] < r.end)]
            if prev.empty:
                continue
            prev = prev.iloc[-1]                          # latest end of the previous YTD step
            if not (QUARTER_GAP[0] <= (r.end - prev["end"]).days <= QUARTER_GAP[1]):
                continue
            out[r.end] = float(r.val) - float(prev["val"])
    s = pd.Series(out, dtype=float)
    s.index = pd.DatetimeIndex(s.index, name="end")
    return s.sort_index()


def ttm(table: pd.DataFrame, date, concept: str) -> float:
    """Sum of the last four single-quarter values, or NaN unless they are four consecutive quarters."""
    q = quarterly(table, date, concept)
    if len(q) < 4:
        return _NAN
    last = q.iloc[-4:]
    gaps = np.diff(last.index.values).astype("timedelta64[D]").astype(int)
    if not all(QUARTER_GAP[0] <= g <= QUARTER_GAP[1] for g in gaps):
        return _NAN
    return float(last.sum())


# -- snapshot ---------------------------------------------------------------------------------

def _val_at_end(table, date, concept, end) -> float:
    """Instant value of `concept` for exactly the period `end` as known on `date`, else NaN."""
    p = pit(table, date, concept)
    p = p[p["start"].isna() & (p["end"] == end)]
    return float(p.iloc[-1]["val"]) if not p.empty else _NAN


def _cash_total(usd, d):
    end, cash = as_of(usd, d, CASH)
    if end is None:
        return None, _NAN
    for c in CASH_ADD:
        extra = _val_at_end(usd, d, c, end)
        if np.isfinite(extra):
            return end, cash + extra                      # only the FIRST available add-on
    return end, cash


def _debt(usd, d) -> float:
    def v(c):
        return as_of(usd, d, c)[1]

    lt_nc, lt, lt_cur = v(LTD_NONCURRENT), v(LTD), v(LTD_CURRENT)
    if np.isfinite(lt_nc):
        long_part = lt_nc
    elif np.isfinite(lt) and np.isfinite(lt_cur):
        long_part = lt - lt_cur
    else:
        long_part = lt                                    # may be NaN
    cur = v(DEBT_CURRENT)
    if np.isfinite(cur):
        short_part = cur
    else:
        parts = [x for x in (lt_cur, v(ST_BORROW)) if np.isfinite(x)]
        short_part = float(sum(parts)) if parts else _NAN
    if not np.isfinite(long_part) and not np.isfinite(short_part):
        return _NAN
    return float(np.nansum([long_part, short_part]))


def _shares_now(shr, d) -> float:
    """Shares outstanding known on `d`: the dei cover-page count if any, else the us-gaap balance-sheet count."""
    for c in (SHARES_DEI, SHARES_GAAP):
        v = as_of(shr, d, c)[1]
        if np.isfinite(v):
            return v
    return _NAN


def _shares_pair(shr, d_now, d_then):
    """(shares at d_now, shares at d_then), each point in time, both from ONE concept (dei first, else us-gaap)."""
    for c in (SHARES_DEI, SHARES_GAAP):
        a, b = as_of(shr, d_now, c)[1], as_of(shr, d_then, c)[1]
        if np.isfinite(a) and np.isfinite(b):
            return a, b
    return _NAN, _NAN


def _revenue_concept(usd, d):
    best, best_end = None, None
    for c in REVENUE:
        end, _ = as_of(usd, d, c, kind="duration")
        if end is not None and (best_end is None or end > best_end):
            best, best_end = c, end
    return best


def snapshot(cik, date, client=None, facts=None) -> dict:
    """Point-in-time fundamentals for `cik` on `date` (only filings filed on or before `date`).

    Keys: as_of, period_end (date of the cash figure), cash_total, debt, op_cash_flow_ttm,
    burn_per_month, runway_months (inf when operating cash flow is not negative), shares_out,
    share_growth_yoy, revenue_concept, revenue_ttm, rev_growth_yoy, gross_margin_ttm. Anything the
    filings do not support is NaN (period_end is None). Pass `facts` to skip the network.
    """
    d = _ts(date)
    if facts is None:
        facts = company_facts(cik, client=client)
    table = facts_table(facts, concepts=_SNAPSHOT_CONCEPTS)
    usd = table[table["unit"] == "USD"]
    shr = table[table["unit"] == "shares"]

    period_end, cash_total = _cash_total(usd, d)
    debt = _debt(usd, d)

    ocf = ttm(usd, d, OCF)
    burn = max(0.0, -ocf) / 12.0 if np.isfinite(ocf) else _NAN
    if not np.isfinite(burn):
        runway = _NAN
    elif burn > 0:
        runway = cash_total / burn if np.isfinite(cash_total) else _NAN
    else:
        runway = float("inf")

    d_prev = d - pd.Timedelta(days=365)
    shares_out = _shares_now(shr, d)
    shares_a, shares_b = _shares_pair(shr, d, d_prev)
    share_growth = shares_a / shares_b - 1.0 if np.isfinite(shares_b) and shares_b != 0 else _NAN

    rev_c = _revenue_concept(usd, d)
    rev_ttm = rev_prev = _NAN
    if rev_c is not None:
        rev_ttm, rev_prev = ttm(usd, d, rev_c), ttm(usd, d_prev, rev_c)
    rev_growth = rev_ttm / rev_prev - 1.0 if np.isfinite(rev_ttm) and np.isfinite(rev_prev) and rev_prev != 0 else _NAN

    gross_margin = _NAN
    if np.isfinite(rev_ttm) and rev_ttm != 0:
        gp = ttm(usd, d, GROSS)
        if np.isfinite(gp):
            gross_margin = gp / rev_ttm
        else:
            for c in COST:
                cost = ttm(usd, d, c)
                if np.isfinite(cost):
                    gross_margin = (rev_ttm - cost) / rev_ttm
                    break

    return {"as_of": d, "period_end": period_end, "cash_total": cash_total, "debt": debt,
            "op_cash_flow_ttm": ocf, "burn_per_month": burn, "runway_months": runway,
            "shares_out": shares_out, "share_growth_yoy": share_growth,
            "revenue_concept": rev_c, "revenue_ttm": rev_ttm, "rev_growth_yoy": rev_growth,
            "gross_margin_ttm": gross_margin}


# -- filings ----------------------------------------------------------------------------------

def recent_filings(cik, forms=FORMS_DEFAULT, days: int = 365, today=None, client=None, subm=None) -> pd.DataFrame:
    """Filings of the given forms with filingDate in [today - days, today] (both inclusive), newest first.

    Columns: form, filingDate, accessionNumber, primaryDocument. The submissions "recent" block
    holds only the latest ~1000 filings; older pages are not read, so for a very active filer a
    long window can be truncated.
    """
    if subm is None:
        subm = submissions(cik, client=client)
    recent = ((subm or {}).get("filings", {}) or {}).get("recent", {}) or {}
    n = len(recent.get("form", []) or [])

    def col(name):
        vals = list(recent.get(name, []) or [])
        return vals + [None] * (n - len(vals))

    df = pd.DataFrame({"form": col("form"), "filingDate": pd.to_datetime(col("filingDate"), errors="coerce"),
                       "accessionNumber": col("accessionNumber"), "primaryDocument": col("primaryDocument")},
                      columns=FILING_COLUMNS)
    end = _ts(today if today is not None else _today_date.today())
    start = end - pd.Timedelta(days=int(days))
    keep = df["form"].isin(set(forms)) & (df["filingDate"] >= start) & (df["filingDate"] <= end)
    return df[keep].sort_values("filingDate", ascending=False, kind="mergesort").reset_index(drop=True)


def dilution_flags(filings: pd.DataFrame) -> dict:
    """Counts of dilution-type filings: n_shelf (S-3, S-3ASR), n_supplements (424B5, 424B3), their sum,
    and potential_atm = at least one shelf AND at least one supplement.

    This is a pointer to go read the filings, not a finding: 424B3 is mostly resale registrations
    and merger prospectuses, and a shelf can be unused. 8-K and Form 4 are not counted.
    """
    forms = filings["form"] if filings is not None and "form" in filings.columns else pd.Series([], dtype=object)
    n_shelf = int(forms.isin(SHELF_FORMS).sum())
    n_supp = int(forms.isin(SUPPLEMENT_FORMS).sum())
    return {"n_shelf": n_shelf, "n_supplements": n_supp, "n_dilution_forms": n_shelf + n_supp,
            "potential_atm": bool(n_shelf >= 1 and n_supp >= 1)}
