"""
RoboAlgo - SEC Form 3/4/5 insider data sets: open-market purchase clusters and a point-in-time symbol map.

What it answers: on which dates did two or more directors or officers of the same company buy
its stock on the open market (a "cluster"), and which company (CIK) did a ticker belong to on a
given date. The input is the SEC's quarterly "Form 3, 4 and 5 data sets" (zip files of tab
separated tables), put by hand in INSIDER_DIR. Nothing here touches the network except the
optional download_quarters() helper, which no study path calls.

*** FORMAT IS UNVERIFIED ***
---------------------------
No data set was fetched while this module was written. The URL pattern was wrong and is now
verified and corrected (see DATASET_URLS); the member names
(SUBMISSION, REPORTINGOWNER, NONDERIV_TRANS), the column names, the date format and the
encoding of RPTOWNER_RELATIONSHIP are all assumptions taken from memory of the SEC's published
layout. The first real file is the verification. read_table() fails loudly with the columns it
actually found, so a mismatch shows up as a FormatError and not as an empty result. Possible
surprises: the relationship split into flag columns, ISO dates, CRLF line ends, embedded tabs
(those rows are skipped, and stats report how many rows each table gave).

*** DEFINITIONS (pre-registered in docs/INSIDER_STUDY_PREREG.md) ***
-----------------------------------------------------------------
* A purchase row is a non-derivative transaction with code P (open-market or private purchase)
  on a Form 4 or Form 5. Amendments (4/A, 5/A) are NOT counted, so a late amendment can never
  complete a cluster with a stale trade. Forms 3 never carry purchases.
* Value per (owner, filing) is the sum over that filing's code-P rows of shares x price. A row
  with a non-finite or non-positive share count or price adds nothing. The filing qualifies when
  that sum is at least $5,000.00 (rounded to 1e-6 dollars to defeat float noise at the boundary).
* Only directors and officers count. A ten percent owner alone does not. A relationship such as
  "Director, TenPercentOwner" counts because it names a director.
* The event date is the FILING date, never the trade date. A trade can be reported up to two
  business days late, and only the filing makes it public.
* A cluster needs at least 2 distinct qualifying owners AND at least 2 distinct accession
  numbers inside a window of 30 calendar days (filing dates at most 29 days apart, inclusive).
  One Form 4 can list several reporting owners, and its transaction rows belong to the accession,
  not to an owner, so a single joint filing is never a cluster.
* The cluster date is the first date on which the window held the second qualifying owner.

Look-ahead rules
----------------
map_symbols() uses only filings made on or before the date asked (first_filed <= date), and only
those within the last SYMBOL_LOOKBACK_DAYS. A ticker linked to two or more CIKs in that window is
reported as ambiguous and the caller is expected to exclude it.

Spans are one row per (symbol, issuer CIK, source zip) and are not merged across zips on purpose,
so a ticker that moved from one company to another is not smeared into a single interval.

Imports and credentials
-----------------------
Module level imports are stdlib, numpy and pandas only. NEVER `from data_engine import ...`:
data_engine/__init__ pulls in the database layer, yfinance and config.settings (which loads .env).
sec_edgar is loaded lazily BY PATH inside the download helper. SEC_USER_AGENT must be set to
"Name contact@example.org" for downloads; there is no default and no address is stored here.
"""
from __future__ import annotations

import csv
import importlib.util
import os
import re
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INSIDER_DIR = Path.home() / ".roboalgo" / "sec" / "insider"
# Verified 2026-10-26 against the SEC's "Insider Transactions Data Sets" index page. The pre-registered
# guess (".../form-345-data-sets/...") returns 404 for every quarter; see the amendment in the prereg.
# 2006q1 to 2026q1 sit under structureddata, 2026q2 onward under datastandardsinnovation, so both are tried.
DATASET_URLS = ("https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{year}q{q}_form345.zip",
                "https://www.sec.gov/files/datastandardsinnovation/data/insider-transactions-data-sets/{year}q{q}_form345.zip")
TABLES = {"SUBMISSION": ("ACCESSION_NUMBER", "FILING_DATE", "DOCUMENT_TYPE", "ISSUERCIK", "ISSUERTRADINGSYMBOL"),
          "REPORTINGOWNER": ("ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNER_RELATIONSHIP"),
          "NONDERIV_TRANS": ("ACCESSION_NUMBER", "TRANS_CODE", "TRANS_SHARES", "TRANS_PRICEPERSHARE")}
OPTIONAL = {"NONDERIV_TRANS": ("TRANS_DATE",)}
PURCHASE_FORMS = ("4", "5")
PURCHASE_CODE = "P"
QUALIFYING_ROLES = frozenset({"director", "officer"})
MIN_VALUE = 5_000.0
CLUSTER_WINDOW_DAYS = 30          # 30 calendar days inclusive: max filing-date difference 29
CLUSTER_MIN_OWNERS = 2
SYMBOL_LOOKBACK_DAYS = 365
MISSING_SYMBOLS = frozenset({"", "NONE", "N/A", "NA", "NULL", "-"})
DATE_FORMATS = ("%d-%b-%Y", "%Y-%m-%d", "%m/%d/%Y")
PURCHASE_COLUMNS = ["accession", "filing_date", "issuer_cik", "symbol", "owner_cik", "value", "trade_date"]
EVENT_COLUMNS = ["issuer_cik", "symbol", "event_date", "n_owners", "accession"]
SPAN_COLUMNS = ["symbol", "issuer_cik", "first_filed", "last_filed", "n_filings"]

_ZIP_MAGIC = b"PK\x03\x04"
_QUARTER_LABEL = re.compile(r"^\d{4}q[1-4]$")
_QUARTER_IN_NAME = re.compile(r"(\d{4})\s*q([1-4])", re.IGNORECASE)
_EDGAR_KEY = "roboalgo_sec_edgar"


class FormatError(ValueError):
    """The data set does not look like the layout this module expects (see the UNVERIFIED note)."""


# -- scalar and vector normalizers ------------------------------------------------------------

def normalize_symbol(s) -> str:
    """Upper case, '.' to '-', and "" for None, NaN and placeholder values such as "NONE" or "N/A"."""
    if s is None or (not isinstance(s, str) and pd.isna(s)):
        return ""
    out = str(s).strip().upper().replace(".", "-")
    return "" if out in MISSING_SYMBOLS else out


def pad_cik(x) -> str:
    """Digits only, zero padded to 10. "" if there are no digits."""
    if x is None or (not isinstance(x, str) and pd.isna(x)):
        return ""
    digits = re.sub(r"\D", "", str(x))
    return digits.zfill(10) if digits else ""


def _pad_ciks(s: pd.Series) -> pd.Series:
    digits = s.fillna("").astype(str).str.replace(r"\D", "", regex=True)
    return digits.str.zfill(10).where(digits != "", "")


def _norm_symbols(s: pd.Series) -> pd.Series:
    out = s.fillna("").astype(str).str.strip().str.upper().str.replace(".", "-", regex=False)
    return out.where(~out.isin(MISSING_SYMBOLS), "")


def parse_dates(s: pd.Series) -> pd.Series:
    """Parse with each DATE_FORMATS entry in turn, filling gaps in order. Unparseable gives NaT."""
    txt = s.fillna("").astype(str).str.strip()
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    for fmt in DATE_FORMATS:
        todo = out.isna()
        if not todo.any():
            break
        got = pd.to_datetime(txt[todo], format=fmt, errors="coerce")
        out = out.copy()
        out[todo] = got
    return out.astype("datetime64[ns]").dt.normalize()


def qualifying_role(relationship) -> bool:
    """True iff any token of the relationship text names a director or an officer."""
    if relationship is None or (not isinstance(relationship, str) and pd.isna(relationship)):
        return False
    for tok in re.split(r"[,;|/]", str(relationship)):
        if re.sub(r"[^a-z]", "", tok.lower()) in QUALIFYING_ROLES:
            return True
    return False


def _numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.fillna("").astype(str).str.strip().str.replace(",", "", regex=False), errors="coerce")


# -- reading the zip --------------------------------------------------------------------------

def find_member(names: list[str], table: str) -> str:
    """The zip member whose base name (no directories, no .tsv/.txt) equals `table`, ignoring case."""
    hits = []
    for name in names:
        if name.endswith("/"):
            continue
        base = name.replace("\\", "/").rsplit("/", 1)[-1]
        low = base.lower()
        for ext in (".tsv", ".txt"):
            if low.endswith(ext):
                base = base[: -len(ext)]
                break
        if base.lower() == table.lower():
            hits.append(name)
    if len(hits) != 1:
        what = "no member" if not hits else f"{len(hits)} members"
        raise FormatError(f"{table}: {what} matched; members found: {sorted(names)}")
    return hits[0]


def read_table(zf: zipfile.ZipFile, table: str, required: tuple, optional: tuple = ()) -> pd.DataFrame:
    """One tab separated member as strings, with headers upper-cased. Missing required columns raise FormatError."""
    member = find_member(zf.namelist(), table)
    try:
        with zf.open(member) as fh:
            df = pd.read_csv(fh, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE,
                             encoding="utf-8", encoding_errors="replace", on_bad_lines="skip")
    except pd.errors.EmptyDataError as e:
        raise FormatError(f"{table}: member {member!r} is empty") from e
    df.columns = [str(c).strip().upper() for c in df.columns]
    df = df.loc[:, ~pd.Index(df.columns).duplicated()]
    found = list(df.columns)
    missing = [c for c in required if c not in found]
    if missing:
        raise FormatError(f"{table}: missing column(s) {missing}; columns found: {sorted(found)}")
    keep = list(required) + [c for c in optional if c in found]
    return df[keep].copy()


def _empty_purchases() -> pd.DataFrame:
    return pd.DataFrame({"accession": pd.Series(dtype=object), "filing_date": pd.Series(dtype="datetime64[ns]"),
                         "issuer_cik": pd.Series(dtype=object), "symbol": pd.Series(dtype=object),
                         "owner_cik": pd.Series(dtype=object), "value": pd.Series(dtype=float),
                         "trade_date": pd.Series(dtype="datetime64[ns]")})[PURCHASE_COLUMNS]


def _empty_spans() -> pd.DataFrame:
    return pd.DataFrame({"symbol": pd.Series(dtype=object), "issuer_cik": pd.Series(dtype=object),
                         "first_filed": pd.Series(dtype="datetime64[ns]"),
                         "last_filed": pd.Series(dtype="datetime64[ns]"),
                         "n_filings": pd.Series(dtype="int64")})[SPAN_COLUMNS]


def _empty_events() -> pd.DataFrame:
    return pd.DataFrame({"issuer_cik": pd.Series(dtype=object), "symbol": pd.Series(dtype=object),
                         "event_date": pd.Series(dtype="datetime64[ns]"),
                         "n_owners": pd.Series(dtype="int64"),
                         "accession": pd.Series(dtype=object)})[EVENT_COLUMNS]


def load_quarter(path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """One quarterly zip -> (purchases, spans, stats). See the module docstring for the definitions."""
    path = Path(path)
    try:
        with zipfile.ZipFile(path) as zf:
            sub = read_table(zf, "SUBMISSION", TABLES["SUBMISSION"])
            own = read_table(zf, "REPORTINGOWNER", TABLES["REPORTINGOWNER"])
            trn = read_table(zf, "NONDERIV_TRANS", TABLES["NONDERIV_TRANS"], OPTIONAL["NONDERIV_TRANS"])
    except zipfile.BadZipFile as e:
        raise FormatError(f"{path.name}: not a readable zip file ({e})") from e
    stats = {"file": path.name, "rows": {"SUBMISSION": len(sub), "REPORTINGOWNER": len(own),
                                         "NONDERIV_TRANS": len(trn)},
             "bad_dates": 0, "no_cik": 0, "no_owner_cik": 0}

    # SUBMISSION: one row per filing
    sub["accession"] = sub["ACCESSION_NUMBER"].str.strip()
    sub["filing_date"] = parse_dates(sub["FILING_DATE"])
    sub["doc"] = sub["DOCUMENT_TYPE"].str.strip().str.upper()
    sub["issuer_cik"] = _pad_ciks(sub["ISSUERCIK"])
    sub["symbol"] = _norm_symbols(sub["ISSUERTRADINGSYMBOL"])
    bad = sub["filing_date"].isna()
    stats["bad_dates"] = int(bad.sum())
    sub = sub[~bad]
    nocik = sub["issuer_cik"] == ""
    stats["no_cik"] = int(nocik.sum())
    sub = sub[~nocik].drop_duplicates("accession", keep="first")

    # spans use every document type that has a symbol, a date and a CIK
    sp = sub[sub["symbol"] != ""]
    if len(sp):
        spans = (sp.groupby(["symbol", "issuer_cik"], sort=True)
                 .agg(first_filed=("filing_date", "min"), last_filed=("filing_date", "max"),
                      n_filings=("accession", "count")).reset_index())[SPAN_COLUMNS]
        spans["n_filings"] = spans["n_filings"].astype("int64")
    else:
        spans = _empty_spans()

    # NONDERIV_TRANS: code-P rows summed per accession
    trn["accession"] = trn["ACCESSION_NUMBER"].str.strip()
    p = trn[trn["TRANS_CODE"].str.strip().str.upper() == PURCHASE_CODE].copy()
    if len(p) == 0:
        return _empty_purchases(), spans, stats
    shares, price = _numeric(p["TRANS_SHARES"]), _numeric(p["TRANS_PRICEPERSHARE"])
    usable = np.isfinite(shares) & np.isfinite(price) & (shares > 0) & (price > 0)
    p["amount"] = np.where(usable, shares * price, 0.0)
    p["trade_date"] = parse_dates(p["TRANS_DATE"]) if "TRANS_DATE" in p.columns else pd.NaT
    per_acc = p.groupby("accession", sort=False).agg(value=("amount", "sum"), trade_date=("trade_date", "max"))
    per_acc["value"] = per_acc["value"].round(6)
    per_acc = per_acc[per_acc["value"] >= MIN_VALUE].reset_index()

    # only Forms 4 and 5 (no amendments), joined to the issuer and the qualifying owners
    filings = sub[sub["doc"].isin(PURCHASE_FORMS)][["accession", "filing_date", "issuer_cik", "symbol"]]
    own["accession"] = own["ACCESSION_NUMBER"].str.strip()
    own["owner_cik"] = _pad_ciks(own["RPTOWNERCIK"])
    roles = {r: qualifying_role(r) for r in own["RPTOWNER_RELATIONSHIP"].unique()}
    own = own[own["RPTOWNER_RELATIONSHIP"].map(roles).astype(bool)]
    noown = own["owner_cik"] == ""
    stats["no_owner_cik"] = int(noown.sum())
    own = own[~noown][["accession", "owner_cik"]]

    out = per_acc.merge(filings, on="accession", how="inner").merge(own, on="accession", how="inner")
    if len(out) == 0:
        return _empty_purchases(), spans, stats
    out = (out[PURCHASE_COLUMNS].drop_duplicates(["owner_cik", "accession"], keep="first")
           .sort_values(["issuer_cik", "filing_date", "accession", "owner_cik"], kind="mergesort")
           .reset_index(drop=True))
    return out, spans, stats


def _zip_paths(insider_dir) -> list[Path]:
    d = Path(insider_dir)
    if not d.is_dir():
        return []
    return sorted((p for p in d.iterdir() if p.is_file() and p.suffix.lower() == ".zip"), key=lambda p: p.name)


def load_dir(insider_dir) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Every *.zip in the directory, in name order. Purchases are de-duplicated; spans are kept per zip."""
    paths = _zip_paths(insider_dir)
    if not paths:
        raise FileNotFoundError(f"no .zip files in {insider_dir}")
    pur, spn, stats = [], [], []
    for p in paths:
        a, b, st = load_quarter(p)
        pur.append(a)
        spn.append(b)
        stats.append(st)
    purchases = pd.concat(pur, ignore_index=True)
    purchases = (purchases.drop_duplicates(["owner_cik", "accession"], keep="first")
                 .sort_values(["issuer_cik", "filing_date", "accession", "owner_cik"], kind="mergesort")
                 .reset_index(drop=True))[PURCHASE_COLUMNS]
    spans = pd.concat(spn, ignore_index=True)[SPAN_COLUMNS]
    return purchases, spans, stats


def inventory(insider_dir) -> dict:
    """What is in the directory, judged by file names only. It never opens a zip."""
    d = Path(insider_dir)
    names = [p.name for p in _zip_paths(d)]
    quarters, unrecognized = set(), []
    for n in names:
        m = _QUARTER_IN_NAME.search(n)
        if m:
            quarters.add(f"{m.group(1)}q{m.group(2)}")
        else:
            unrecognized.append(n)
    return {"path": str(d), "exists": d.is_dir(), "n_zips": len(names), "zips": names,
            "quarters": sorted(quarters), "unrecognized": unrecognized}


# -- clusters and the symbol map --------------------------------------------------------------

def clusters(purchases: pd.DataFrame, window_days: int = CLUSTER_WINDOW_DAYS,
             min_owners: int = CLUSTER_MIN_OWNERS) -> pd.DataFrame:
    """Cluster events by issuer CIK, dated at the FILING date on which the window first held enough owners.

    For each purchase row j the window is every row of the same issuer with a filing date in
    [fd_j - (window_days - 1) days, fd_j]. It is date based: same-day rows are in, later dates never are.
    An event needs >= min_owners distinct owners AND >= 2 distinct accessions in the window.
    """
    if len(purchases) == 0:
        return _empty_events()
    p = purchases[["issuer_cik", "symbol", "filing_date", "accession", "owner_cik"]].dropna(subset=["filing_date"])
    p = p.sort_values(["issuer_cik", "filing_date", "accession"], kind="mergesort")
    back = np.timedelta64(window_days - 1, "D")
    rows = []
    for cik, g in p.groupby("issuer_cik", sort=True):
        if len(g) < 2 or g["owner_cik"].nunique() < min_owners:
            continue
        dates = g["filing_date"].to_numpy()
        owners = g["owner_cik"].to_numpy()
        accs = g["accession"].to_numpy()
        syms = g["symbol"].to_numpy()
        lo = np.searchsorted(dates, dates - back, side="left")
        hi = np.searchsorted(dates, dates, side="right")
        for j in range(len(g)):
            a, b = lo[j], hi[j]
            if b - a < 2:
                continue
            ow = set(owners[a:b])
            if len(ow) < min_owners:
                continue
            if len(set(accs[a:b])) < 2:
                continue
            rows.append((cik, syms[j], dates[j], len(ow), accs[j]))
    if not rows:
        return _empty_events()
    ev = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    ev["event_date"] = pd.to_datetime(ev["event_date"]).astype("datetime64[ns]")
    ev["n_owners"] = ev["n_owners"].astype("int64")
    return ev.drop_duplicates(["issuer_cik", "event_date"], keep="first").reset_index(drop=True)


def covered_quarters(spans: pd.DataFrame) -> set[str]:
    """Lowercase 'YYYYqN' labels of every first_filed and last_filed in the spans."""
    if len(spans) == 0:
        return set()
    d = pd.concat([spans["first_filed"], spans["last_filed"]]).dropna()
    return set(d.dt.to_period("Q").astype(str).str.lower().unique())


def reused_symbols(spans: pd.DataFrame) -> set[str]:
    """Symbols linked to two or more distinct issuer CIKs anywhere in the data."""
    if len(spans) == 0:
        return set()
    n = spans.groupby("symbol")["issuer_cik"].nunique()
    return set(n[n >= 2].index)


def map_symbols(spans: pd.DataFrame, keys: pd.DataFrame, lookback_days: int = SYMBOL_LOOKBACK_DAYS) -> pd.DataFrame:
    """Which CIK did each (symbol, date) key belong to, using only filings made on or before the date.

    A span is a candidate iff it has the (normalized) symbol, first_filed <= date and
    last_filed >= date - lookback_days. One distinct CIK gives (cik, False), two or more give
    (None, True), none gives (None, False). Output keeps the input index and row order, the input
    symbol and date columns, and adds `cik` (object: str or None) and `ambiguous` (bool).
    """
    n = len(keys)
    out = keys[["symbol", "date"]].copy()
    ciks = np.full(n, None, dtype=object)
    amb = np.zeros(n, dtype=bool)
    if n and len(spans):
        k = pd.DataFrame({"_row": np.arange(n), "_sym": _norm_symbols(keys["symbol"]).to_numpy(),
                          "date": pd.to_datetime(keys["date"]).to_numpy()})
        sp = spans[["symbol", "issuer_cik", "first_filed", "last_filed"]]
        m = k.merge(sp, left_on="_sym", right_on="symbol", how="inner")
        m = m[(m["first_filed"] <= m["date"]) & (m["last_filed"] >= m["date"] - pd.Timedelta(days=lookback_days))]
        if len(m):
            g = m.groupby("_row")["issuer_cik"].agg(["nunique", "min"])
            one = g[g["nunique"] == 1]
            ciks[one.index.to_numpy()] = one["min"].to_numpy()
            amb[g[g["nunique"] >= 2].index.to_numpy()] = True
    out["cik"] = pd.Series(ciks, index=out.index, dtype=object)
    out["ambiguous"] = amb
    return out


# -- download helper (UNVERIFIED, not wired to any study path) ---------------------------------

def _edgar():
    """data_engine/sec_edgar.py loaded by path, once. (data_engine/__init__ would import the database layer.)"""
    mod = sys.modules.get(_EDGAR_KEY)
    if mod is not None:
        return mod
    spec = importlib.util.spec_from_file_location(_EDGAR_KEY, ROOT / "data_engine" / "sec_edgar.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_EDGAR_KEY] = mod                       # before exec: required for @dataclass on Python 3.14
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(_EDGAR_KEY, None)
        raise
    return mod


def download_quarters(quarters: list[str], dest=INSIDER_DIR, fetch=None, clock=time.monotonic,
                      sleep=time.sleep) -> list[Path]:
    """Fetch the quarterly zips named like '2015q1' into `dest`. UNVERIFIED: the URL was never tested.

    Returns the path of every requested quarter's file (already present or newly written). Existing
    files are skipped without a request. The User-Agent comes from sec_edgar.user_agent(), which raises
    RuntimeError when SEC_USER_AGENT is unset, before any request. A status other than 200, or a body
    that is not a zip, raises RuntimeError mentioning UNVERIFIED and writes nothing.
    """
    for q in quarters:
        if not isinstance(q, str) or not _QUARTER_LABEL.match(q):
            raise ValueError(f"bad quarter label {q!r}; expected like '2015q1'")
    dest = Path(dest)
    paths = [dest / f"{q}_form345.zip" for q in quarters]
    todo = [(q, p) for q, p in zip(quarters, paths) if not p.exists()]
    if not todo:
        return paths
    E = _edgar()
    headers = {"User-Agent": E.user_agent()}
    fetch = fetch or E.default_fetch
    throttle = E.Throttle(E.MAX_PER_SEC, clock=clock, sleep=sleep)
    dest.mkdir(parents=True, exist_ok=True)
    for q, final in todo:
        tried = []
        for template in DATASET_URLS:
            url = template.format(year=q[:4], q=q[5])
            throttle.wait()
            status, body = fetch(url, headers)
            tried.append(f"HTTP {status} from {url}")
            if status == 200 and body[:4] == _ZIP_MAGIC:
                break
        else:
            raise RuntimeError(f"{q}: no candidate URL returned a zip ({'; '.join(tried)}). Check the SEC's "
                               f"Insider Transactions Data Sets page for the current path.")
        tmp = final.with_name(final.name + ".tmp")
        tmp.write_bytes(body)
        os.replace(tmp, final)
    return paths
