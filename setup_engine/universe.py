"""The S&P 500 member list: scraped from Wikipedia, cached on disk, filtered by the date a name was added.

Standard library + pandas only. Network access goes through an injectable ``fetch(url, headers) -> str``
so nothing here needs the internet in a test.

THE LIST IS TODAY'S LIST. Wikipedia carries the CURRENT constituents with the date each was added.
``members_asof`` filters that list by the add date; it is NOT the historical membership (names that
were removed since are absent). That is fine for a screen of today's index and survivorship-biased for
any backtest, which must not use it as a point-in-time universe.
"""
from __future__ import annotations

import html as html_lib
import re
from datetime import date
from pathlib import Path

import pandas as pd

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
USER_AGENT = "RoboAlgoResearch/1.0"
UNIVERSE_DIR = Path.home() / ".roboalgo" / "universe"
MAX_AGE_DAYS = 7
REQUIRED = ("Symbol", "Security", "GICS Sector", "GICS Sub-Industry", "Date added", "CIK")
OUT_COLUMNS = ("symbol", "raw_symbol", "security", "sector", "sub_industry", "date_added", "cik")
_FILE_GLOB = "sp500_????-??-??.csv"
_FILE_RE = re.compile(r"^sp500_(\d{4}-\d{2}-\d{2})\.csv$")


def normalize_symbol(s: str) -> str:
    """Yahoo's spelling of a ticker: upper case, "." -> "-" (``BRK.B`` -> ``BRK-B``)."""
    return s.strip().upper().replace(".", "-")


def _cell_text(s: str) -> str:
    s = re.sub(r"<[^>]+>", "", s)
    s = html_lib.unescape(s)
    s = re.sub(r"\[\d+\]", "", s)                   # footnote markers such as "[3]"
    return " ".join(s.split())


def _cik(s: str) -> str:
    digits = re.sub(r"\D", "", s or "")
    return digits.zfill(10) if digits else ""


def parse_sp500(html: str) -> pd.DataFrame:
    """The ``id="constituents"`` table of the Wikipedia page -> one row per member.

    Columns: symbol (Yahoo spelling), raw_symbol, security, sector, sub_industry, date_added (datetime,
    NaT when blank), cik (10 digits, "" when blank). ValueError when the table or a required column is
    missing, so a page redesign fails loudly instead of returning an empty universe.
    """
    m = re.search(r'<table\b[^>]*\bid\s*=\s*"constituents"[^>]*>(.*?)</table>', html, re.S | re.I)
    if m is None:
        raise ValueError('S&P 500 constituents table (id="constituents") not found in the page')
    body = m.group(1)
    header: list[str] | None = None
    data: list[list[str]] = []
    for row in re.findall(r"<tr\b[^>]*>(.*?)</tr>", body, re.S | re.I):
        cells = re.findall(r"<t([dh])\b[^>]*>(.*?)</t[dh]>", row, re.S | re.I)
        if not cells:
            continue
        kinds = [k.lower() for k, _ in cells]
        texts = [_cell_text(t) for _, t in cells]
        if header is None and all(k == "h" for k in kinds):
            header = texts
        elif "d" in kinds:
            data.append(texts)
    header = header or []
    missing = [c for c in REQUIRED if c not in header]
    if missing:
        raise ValueError(f"constituents table is missing columns {missing}")
    rows = [r for r in data if len(r) == len(header)]
    col = {name: header.index(name) for name in REQUIRED}
    out = pd.DataFrame({
        "symbol": [normalize_symbol(r[col["Symbol"]]) for r in rows],
        "raw_symbol": [r[col["Symbol"]] for r in rows],
        "security": [r[col["Security"]] for r in rows],
        "sector": [r[col["GICS Sector"]] for r in rows],
        "sub_industry": [r[col["GICS Sub-Industry"]] for r in rows],
        "date_added": pd.to_datetime(pd.Series([r[col["Date added"]] for r in rows], dtype="object"),
                                     errors="coerce"),
        "cik": [_cik(r[col["CIK"]]) for r in rows],
    })
    return out[list(OUT_COLUMNS)]


def fetch_sp500(fetch=None) -> str:
    """The raw HTML of the Wikipedia page. ``fetch(url, headers) -> str`` is injectable; the default uses
    urllib with ``{"User-Agent": USER_AGENT}`` and a 30 second timeout."""
    headers = {"User-Agent": USER_AGENT}
    if fetch is not None:
        return fetch(WIKI_URL, headers)
    import urllib.request

    req = urllib.request.Request(WIKI_URL, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8")


def _snapshots(dest: Path) -> list[tuple[pd.Timestamp, Path]]:
    """(date from the file NAME, path) of every cached snapshot, oldest first."""
    out = []
    if dest.exists():
        for p in sorted(dest.glob(_FILE_GLOB)):
            m = _FILE_RE.match(p.name)
            if m:
                out.append((pd.Timestamp(m.group(1)), p))
    return out


def refresh_sp500(dest_dir=UNIVERSE_DIR, max_age_days: int = MAX_AGE_DAYS, fetch=None, today=None) -> Path:
    """The newest cached snapshot if it is younger than ``max_age_days``, else download and cache a new one.

    A snapshot is ``sp500_YYYY-MM-DD.csv``; its age is taken from the date in its NAME. The page is only
    fetched when there is no snapshot or the newest one is ``max_age_days`` old or older.
    """
    dest = Path(dest_dir)
    today_ts = pd.Timestamp(date.today() if today is None else today).normalize()
    snaps = _snapshots(dest)
    if snaps:
        newest_date, newest_path = snaps[-1]
        if (today_ts - newest_date).days < max_age_days:
            return newest_path
    df = parse_sp500(fetch_sp500(fetch))
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"sp500_{today_ts:%Y-%m-%d}.csv"
    df.to_csv(path, index=False)
    return path


def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Accept a snapshot saved with Wikipedia's own headers (Symbol, Security, ...) as well as ours."""
    if "symbol" in df.columns:
        return df
    if not all(c in df.columns for c in REQUIRED):
        raise ValueError(f"universe file has neither the columns {list(OUT_COLUMNS)} nor {list(REQUIRED)}")
    return pd.DataFrame({
        "symbol": [normalize_symbol(s) for s in df["Symbol"]],
        "raw_symbol": df["Symbol"].str.strip(),
        "security": df["Security"],
        "sector": df["GICS Sector"],
        "sub_industry": df["GICS Sub-Industry"],
        "date_added": df["Date added"],
        "cik": df["CIK"],
    })


def load_sp500(path=None, dest_dir=UNIVERSE_DIR) -> pd.DataFrame:
    """``path`` (or the newest ``sp500_*.csv`` in ``dest_dir``) -> the member table.

    ``date_added`` is parsed to datetime (NaT when blank) and ``cik`` is kept as a 10-character string.
    FileNotFoundError when there is no snapshot.
    """
    if path is None:
        files = sorted(Path(dest_dir).glob("sp500_*.csv")) if Path(dest_dir).exists() else []
        if not files:
            raise FileNotFoundError(f"no sp500_*.csv in {dest_dir}; run refresh_sp500() first")
        path = files[-1]
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df = _standardize(df)
    df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
    df["cik"] = [_cik(c) for c in df["cik"]]
    return df[list(OUT_COLUMNS)].reset_index(drop=True)


def members_asof(df: pd.DataFrame, date) -> pd.DataFrame:
    """Rows added on or before ``date`` (a blank add date counts as a member).

    THIS IS THE CURRENT LIST FILTERED BY ADD DATE, NOT HISTORICAL MEMBERSHIP: names that left the index
    are absent. Valid for today's screen; survivorship-biased for any backtest.
    """
    d = pd.Timestamp(date)
    added = df["date_added"]
    return df[(added <= d) | added.isna()]
