"""
Tiingo end-of-day provider for DELISTED stocks: URL builders, split adjustment, the identity rule that
decides whether a Tiingo ticker is really the company we asked about, and a request throttle.

CONTRACT USED (Authorization: Token <token> on every call; the token is never logged or stored):
    GET {BASE}/{ticker}                       -> {"ticker","name","exchangeCode","startDate","endDate",...}; 404 if unknown
    GET {BASE}/{ticker}/prices?startDate=&endDate=&format=json
        -> [{"date": "2015-02-27T00:00:00.000Z", "open","high","low","close","volume",
             "adjOpen","adjHigh","adjLow","adjClose","adjVolume","divCash","splitFactor"}, ...]
The metadata call was exercised by an earlier by-hand coverage probe; the price schema above is Tiingo's
documented one. No live call is made by this module's tests (every network function takes an injectable
``fetch``).

WHY OUR OWN SPLIT ADJUSTMENT. Tiingo's adj* fields also adjust for dividends. The survivor cache is
split-adjusted only (Yahoo auto_adjust=False), so mixing the two would put a dividend step between the
survivors and the delisted stocks. We take the RAW open/high/low/close/volume and divide by the product
of the LATER splitFactor values ourselves. Dividends are NOT adjusted. If Tiingo misses a split (it happens
for OTC names) this adjustment inherits the gap: run the artifact detector on the result before trusting it.

IDENTITY RULE (``identify``). A ticker is re-used by later companies, so "Tiingo has data for AAA" proves
nothing. A match is accepted only if the price history starts near our listing date AND ends near our
delisting date. Known weaknesses, kept visible rather than hidden:
  * No lower bound on the start: a predecessor that held the ticker BEFORE our IPO is not caught.
  * ipoDate before OLD_IPO_BEFORE is treated as unreliable (a long-listed company: Tiingo may start later
    than the vendor's listing date), so the start tolerance widens to START_TOL_OLD_DAYS. That cut-off is
    an assumption, not a measured fact.
  * AV delistingDate and Tiingo endDate can differ by more than END_TOL_DAYS when trading moved to OTC
    before the formal delisting; a good match is then called "reused" (or "mismatch" via the Q ticker).
"""
from __future__ import annotations

import re
import time
from typing import Callable, Optional
from urllib.parse import quote

import numpy as np
import pandas as pd

BASE = "https://api.tiingo.com/tiingo/daily"
SNAPSHOT_CUTOFF = "2026-10-01"      # AV rows delisted on/after this are the vendor's snapshot-day artefacts, not delistings
END_TOL_DAYS = 45
Q_END_TOL_DAYS = 800                # a bankrupt "XYZQ" ticker keeps trading OTC for a long tail after our delisting date
START_TOL_DAYS = 60
START_TOL_OLD_DAYS = 365
OLD_IPO_BEFORE = "2000-01-01"       # ASSUMPTION: an AV ipoDate before this is unreliable (see module docstring)
NAME_STOP = {"corp", "corporation", "company", "holdings", "holding", "group", "limited", "class", "common", "stock",
             "shares", "incorporated", "international", "trust"}
PLANS = {
    "free": {"per_hour": 45, "per_day": 900, "per_month_symbols": 480},
    "power": {"per_hour": 9000, "per_day": 90000, "per_month_symbols": 100000},
}
STATUSES = ("ok", "reused", "no_data", "mismatch")
PRICE_COLUMNS = ["date", "open", "high", "low", "close", "volume"]


class DailyCapReached(RuntimeError):
    """The plan's per-day request cap is used up; waiting an hour does not help."""


def meta_url(ticker: str) -> str:
    return f"{BASE}/{quote(str(ticker), safe='')}"


def prices_url(ticker: str, start: str, end: str) -> str:
    """start / end are 'YYYY-MM-DD'."""
    return f"{BASE}/{quote(str(ticker), safe='')}/prices?startDate={start}&endDate={end}&format=json"


def auth_headers(token: str) -> dict:
    """Request headers. The token goes in the Authorization header only, never in a URL, a log or a file."""
    return {"Authorization": f"Token {token}", "Content-Type": "application/json"}


def default_fetch(url: str, headers: dict) -> tuple[int, bytes]:
    """The one real network call: (status, body). HTTP errors come back as (code, body), not exceptions."""
    import urllib.error
    import urllib.request
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return int(r.status), r.read()
    except urllib.error.HTTPError as e:
        return int(e.code), e.read()


# ── prices ────────────────────────────────────────────────────────────────────────────────────────────

def to_split_adjusted(rows: list[dict]) -> pd.DataFrame:
    """Raw Tiingo daily rows -> split-adjusted (NOT dividend-adjusted) frame: date, open, high, low, close, volume.

    A row's price is divided by the product of the splitFactor of every LATER row (its own factor is not
    included: the split happens on that row's date, so that day's price is already post-split); volume is
    multiplied by the same product. splitFactor missing, 0, negative or NaN counts as 1.0. The adj* fields
    are never read.
    """
    if not rows:
        return pd.DataFrame({c: pd.Series(dtype="datetime64[ns]" if c == "date" else "float64") for c in PRICE_COLUMNS})
    df = pd.DataFrame(rows)
    date = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None).dt.normalize()
    sf = pd.to_numeric(df["splitFactor"], errors="coerce") if "splitFactor" in df else pd.Series(1.0, index=df.index)
    sf = sf.where(sf.notna() & (sf > 0), 1.0)
    out = pd.DataFrame({"date": date})
    for c in ("open", "high", "low", "close", "volume"):
        out[c] = pd.to_numeric(df[c], errors="coerce") if c in df else np.nan
    out["_sf"] = sf.to_numpy(dtype=float)
    out = out.sort_values("date", kind="mergesort").reset_index(drop=True)
    sf_sorted = out.pop("_sf").to_numpy()
    suffix = sf_sorted[::-1].cumprod()[::-1]                      # product of sf over rows >= t
    factor = np.append(suffix[1:], 1.0)                            # product over rows > t (own sf excluded)
    for c in ("open", "high", "low", "close"):
        out[c] = out[c] / factor
    out["volume"] = out["volume"] * factor
    return out[PRICE_COLUMNS]


# ── identity ──────────────────────────────────────────────────────────────────────────────────────────

def name_overlap(a: str, b: str) -> Optional[bool]:
    """Do two company names share a distinctive word (>= 4 letters, generic words removed)? None if either is empty."""
    a = "" if a is None or (isinstance(a, float) and np.isnan(a)) else str(a).strip()
    b = "" if b is None or (isinstance(b, float) and np.isnan(b)) else str(b).strip()
    if not a or not b:
        return None
    wa = set(re.findall(r"[a-z]{4,}", a.lower())) - NAME_STOP
    wb = set(re.findall(r"[a-z]{4,}", b.lower())) - NAME_STOP
    return bool(wa & wb)


def _ts(x) -> Optional[pd.Timestamp]:
    if x is None:
        return None
    t = pd.to_datetime(x, errors="coerce", utc=True)
    if pd.isna(t):
        return None
    return t.tz_localize(None).normalize()


def wide_tolerance(ours: dict) -> bool:
    """True when our ipoDate is old enough that the wide start tolerance applies."""
    ipo = _ts(ours.get("ipoDate"))
    return ipo is not None and ipo < pd.Timestamp(OLD_IPO_BEFORE)


def start_before_ipo_days(meta: dict, ours: dict) -> Optional[int]:
    """Days by which Tiingo's first date precedes our ipoDate (positive = Tiingo starts earlier; large values
    hint at a predecessor company that held the ticker, which the identity rule cannot exclude)."""
    ipo, start = _ts(ours.get("ipoDate")), _ts(meta.get("startDate"))
    return None if ipo is None or start is None else int((ipo - start).days)


def identify(meta: Optional[dict], ours: dict, is_q: bool = False) -> str:
    """Is the Tiingo ticker described by ``meta`` the company in ``ours`` (name, ipoDate, delistingDate)?

    Returns 'no_data' | 'ok' | 'reused' | 'mismatch':
      no_data   Tiingo has no metadata, or no start/end date.
      ok        history starts within the start tolerance of our ipoDate AND ends within the end window of our
                delistingDate (+-END_TOL_DAYS; the Q-suffixed ticker may end up to Q_END_TOL_DAYS after).
      reused    dates fail AND (the history runs more than END_TOL_DAYS past our delisting - a later company
                kept trading under this ticker - or the names share no distinctive word).
      mismatch  dates fail but nothing says it is another company.
    """
    if not meta:
        return "no_data"
    start, end = _ts(meta.get("startDate")), _ts(meta.get("endDate"))
    if start is None or end is None:
        return "no_data"
    ipo, delist = _ts(ours.get("ipoDate")), _ts(ours.get("delistingDate"))
    if ipo is None or delist is None:
        return "mismatch"                                   # cannot be placed in time; select_universe never lets this through
    tol = START_TOL_OLD_DAYS if wide_tolerance(ours) else START_TOL_DAYS
    start_ok = start <= ipo + pd.Timedelta(days=tol)
    hi = delist + pd.Timedelta(days=Q_END_TOL_DAYS if is_q else END_TOL_DAYS)
    end_ok = delist - pd.Timedelta(days=END_TOL_DAYS) <= end <= hi
    if start_ok and end_ok:
        return "ok"
    if (not is_q and end > delist + pd.Timedelta(days=END_TOL_DAYS)) or name_overlap(meta.get("name"), ours.get("name")) is False:
        return "reused"
    return "mismatch"


# ── universe ──────────────────────────────────────────────────────────────────────────────────────────

def select_universe(av: pd.DataFrame, from_year: int = 2010, to_year: int = 2025, exclude=frozenset()) -> pd.DataFrame:
    """AlphaVantage delisted-listing rows worth asking Tiingo about, oldest delisting first.

    Kept: assetType == 'Stock'; a common-stock name (no SPAC / warrant / unit / right / preferred wording, the
    project's NON_COMMON_RE); a plain 1-5 capital-letter symbol; delisted before SNAPSHOT_CUTOFF (later dates
    are the snapshot day, not real delistings); listed for more than 365 days; delisted in
    [from_year, to_year]; symbol not in ``exclude`` (the survivor cache: ingest would skip it anyway).
    ipoDate and delistingDate come back as datetimes.
    """
    from setup_engine.trade_stats import NON_COMMON_RE
    d = av.copy()
    d["ipoDate"] = pd.to_datetime(d["ipoDate"], errors="coerce")
    d["delistingDate"] = pd.to_datetime(d["delistingDate"], errors="coerce")
    d["name"] = d["name"].fillna("")
    sym = d["symbol"].astype(str)
    keep = (
        (d["assetType"] == "Stock")
        & ~d["name"].astype(str).str.contains(NON_COMMON_RE, case=False, regex=True, na=False)
        & sym.str.fullmatch(r"[A-Z]{1,5}")
        & (d["delistingDate"] < pd.Timestamp(SNAPSHOT_CUTOFF))
        & ((d["delistingDate"] - d["ipoDate"]).dt.days > 365)
        & d["delistingDate"].dt.year.between(from_year, to_year)
        & ~sym.isin(set(exclude))
    )
    return d[keep].sort_values(["delistingDate", "symbol"], kind="mergesort").reset_index(drop=True)


# ── throttle ──────────────────────────────────────────────────────────────────────────────────────────

class RateLimiter:
    """Hourly and daily request budget for one Tiingo plan.

    before_request() is called before EVERY HTTP request. When the current clock hour already holds
    ``per_hour`` requests it sleeps to the next hour boundary (+1 s), then counts the request. When the
    day (UTC) holds ``per_day`` it raises DailyCapReached instead: sleeping an hour cannot help.
    on_429() sleeps to the next hour boundary: the server said the budget is gone, whatever our count says.
    ``day_count`` may be set by the caller to carry the count of an earlier process the same day.
    """

    def __init__(self, per_hour: int, per_day: int, clock: Callable[[], float] = time.time,
                 sleep: Callable[[float], None] = time.sleep):
        self.per_hour, self.per_day, self.clock, self.sleep = per_hour, per_day, clock, sleep
        now = clock()
        self._hour, self._hour_count = int(now // 3600), 0
        self._day = int(now // 86400)
        self.day_count = 0

    def _roll(self, now: float) -> None:
        if int(now // 3600) != self._hour:
            self._hour, self._hour_count = int(now // 3600), 0
        if int(now // 86400) != self._day:
            self._day, self.day_count = int(now // 86400), 0

    def _to_next_hour(self) -> float:
        return 3600.0 - self.clock() % 3600.0 + 1.0

    def before_request(self) -> None:
        self._roll(self.clock())
        if self.day_count >= self.per_day:
            raise DailyCapReached(f"daily cap of {self.per_day} requests reached")
        if self._hour_count >= self.per_hour:
            self.sleep(self._to_next_hour())
            now = self.clock()
            self._hour, self._hour_count = int(now // 3600), 0
            self._roll(now)
        self._hour_count += 1
        self.day_count += 1

    def on_429(self) -> None:
        self.sleep(self._to_next_hour())
        now = self.clock()
        self._hour, self._hour_count = int(now // 3600), 0
        self._roll(now)
