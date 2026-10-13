"""setup_engine.universe: the S&P 500 list must parse the page faithfully and must not pass off today's list as history.

The failures being guarded against: a ticker spelled the Wikipedia way ("BRK.B") that Yahoo does not know, HTML
entities and footnote markers left in the names, a blank add date silently dropping a member, a member added ON
the as-of date being treated as not yet in the index, and a stale cache being served forever (or a fresh one
re-downloaded on every run). No test touches the network: the fetch is a fake that also checks the headers.
"""
import pandas as pd
import pytest

from setup_engine import universe as U

_HEADER = ("<tr><th>Symbol</th><th>Security</th><th>GICS Sector</th><th>GICS Sub-Industry</th>"
           "<th>Headquarters Location</th><th>Date added</th><th>CIK</th><th>Founded</th></tr>")


def _row(symbol_cell, security, added, cik):
    return (f"<tr><td>{symbol_cell}</td><td>{security}</td><td>Tech</td><td>Software</td><td>Somewhere</td>"
            f"<td>{added}</td><td>{cik}</td><td>1999</td></tr>")


def _page(header=_HEADER):
    other = '<table class="wikitable" id="other"><tr><th>Symbol</th></tr><tr><td>ZZZ</td></tr></table>'
    rows = [
        _row("AAA", "Alpha Inc", "2010-01-04", "12345"),
        _row("BRK.B", "Berkshire &amp; Co", "2010-02-16", "0001067983"),
        _row("CCC", "Gamma [3] Ltd", "", ""),
        _row("DDD", "D&#39;Arcy Corp", "2020-06-30", "99"),
        _row('<a href="/wiki/EEE">EEE</a>', "<b>Epsilon</b>\n Systems", "2020-07-01", "7"),
    ]
    return ("<html><body>" + other + '<table class="wikitable sortable" id="constituents">' + header
            + "".join(rows) + "</table></body></html>")


@pytest.fixture
def members():
    return U.parse_sp500(_page())


# ── parse ─────────────────────────────────────────────────────────────────────

def test_parse_returns_every_member_with_yahoo_symbols_and_clean_text(members):
    """Wrong symbol spelling, unescaped entities, footnote markers or markup left in a cell."""
    assert list(members.columns) == ["symbol", "raw_symbol", "security", "sector", "sub_industry", "date_added", "cik"]
    assert list(members["symbol"]) == ["AAA", "BRK-B", "CCC", "DDD", "EEE"]
    assert members.loc[1, "raw_symbol"] == "BRK.B"
    assert members.loc[1, "security"] == "Berkshire & Co"
    assert members.loc[3, "security"] == "D'Arcy Corp"
    assert members.loc[2, "security"] == "Gamma Ltd"                       # "[3]" footnote removed, spaces collapsed
    assert members.loc[4, "security"] == "Epsilon Systems"                 # tags removed, newline collapsed
    assert members.loc[4, "symbol"] == "EEE"                               # symbol wrapped in a link


def test_parse_dates_and_ciks_are_typed_and_padded(members):
    """A blank add date must become NaT (not crash or become 1970), a CIK must keep its leading zeros."""
    assert members.loc[0, "date_added"] == pd.Timestamp("2010-01-04")
    assert pd.isna(members.loc[2, "date_added"])
    assert str(members["date_added"].dtype).startswith("datetime64")
    assert list(members["cik"]) == ["0000012345", "0001067983", "", "0000000099", "0000000007"]


def test_parse_ignores_the_other_table_on_the_page(members):
    """The unrelated table placed before the constituents must not leak a ZZZ row."""
    assert "ZZZ" not in set(members["symbol"])
    assert len(members) == 5


def test_a_page_without_the_constituents_table_is_an_error_not_an_empty_universe():
    """A page redesign must fail loudly."""
    with pytest.raises(ValueError, match="constituents"):
        U.parse_sp500("<html><table id='other'><tr><td>x</td></tr></table></html>")


def test_a_header_without_a_required_column_is_an_error():
    """A renamed column must fail loudly instead of shifting every field."""
    bad = _HEADER.replace("<th>GICS Sector</th>", "<th>Sector</th>")
    with pytest.raises(ValueError, match="GICS Sector"):
        U.parse_sp500(_page(bad))


def test_normalize_symbol_uses_dashes_and_upper_case():
    """Yahoo spells share classes with a dash."""
    assert U.normalize_symbol(" brk.b ") == "BRK-B"
    assert U.normalize_symbol("BF.B") == "BF-B"
    assert U.normalize_symbol("AAPL") == "AAPL"


# ── members_asof ──────────────────────────────────────────────────────────────

def test_members_asof_includes_the_add_date_itself_and_blank_dates(members):
    """A name added on the as-of date is in; a blank add date is kept; a later add is out (boundary check)."""
    assert set(U.members_asof(members, "2020-06-30")["symbol"]) == {"AAA", "BRK-B", "CCC", "DDD"}
    assert set(U.members_asof(members, "2020-06-29")["symbol"]) == {"AAA", "BRK-B", "CCC"}
    assert set(U.members_asof(members, "2020-07-01")["symbol"]) == {"AAA", "BRK-B", "CCC", "DDD", "EEE"}


# ── fetch + refresh ───────────────────────────────────────────────────────────

class _FakeFetch:
    def __init__(self):
        self.calls = []

    def __call__(self, url, headers):
        self.calls.append((url, dict(headers)))
        assert headers == {"User-Agent": "RoboAlgoResearch/1.0"}, "only the research User-Agent may be sent"
        return _page()


def _touch(dest, name):
    dest.mkdir(parents=True, exist_ok=True)
    p = dest / name
    p.write_text("symbol\n")
    return p


def test_fetch_sp500_sends_only_the_user_agent_to_the_wikipedia_url():
    """A header leak (cookies, keys) or the wrong page."""
    f = _FakeFetch()
    html = U.fetch_sp500(f)
    assert html == _page()
    assert f.calls == [(U.WIKI_URL, {"User-Agent": "RoboAlgoResearch/1.0"})]


def test_refresh_serves_a_fresh_cache_without_fetching(tmp_path):
    """Re-downloading the page on every run (6 days old, limit 7)."""
    p = _touch(tmp_path, "sp500_2026-10-06.csv")
    f = _FakeFetch()
    assert U.refresh_sp500(tmp_path, fetch=f, today="2026-10-12") == p
    assert f.calls == []


def test_refresh_downloads_when_the_cache_is_stale(tmp_path):
    """Serving an 8-day-old list forever."""
    _touch(tmp_path, "sp500_2026-10-06.csv")
    f = _FakeFetch()
    out = U.refresh_sp500(tmp_path, fetch=f, today="2026-10-14")
    assert out == tmp_path / "sp500_2026-10-14.csv"
    assert len(f.calls) == 1
    again = U.load_sp500(out)
    assert list(again["symbol"]) == ["AAA", "BRK-B", "CCC", "DDD", "EEE"]


def test_refresh_downloads_when_the_cache_is_exactly_max_age_old(tmp_path):
    """Off-by-one at the age limit: 7 days old with max_age_days=7 is stale."""
    _touch(tmp_path, "sp500_2026-10-06.csv")
    f = _FakeFetch()
    out = U.refresh_sp500(tmp_path, max_age_days=7, fetch=f, today="2026-10-13")
    assert len(f.calls) == 1
    assert out.name == "sp500_2026-10-13.csv"


def test_refresh_with_no_cache_dir_creates_it_and_fetches(tmp_path):
    """First run: the cache directory does not exist yet."""
    dest = tmp_path / "new" / "universe"
    f = _FakeFetch()
    out = U.refresh_sp500(dest, fetch=f, today="2026-10-12")
    assert out == dest / "sp500_2026-10-12.csv" and out.exists() and len(f.calls) == 1


def test_refresh_reads_the_cache_age_from_the_file_name_not_the_mtime(tmp_path):
    """A file touched today but named for an old date is old."""
    _touch(tmp_path, "sp500_2026-01-02.csv")                      # mtime is now, name says January
    f = _FakeFetch()
    U.refresh_sp500(tmp_path, fetch=f, today="2026-10-12")
    assert len(f.calls) == 1


# ── load ──────────────────────────────────────────────────────────────────────

def test_load_picks_the_newest_snapshot_and_keeps_ciks_as_strings(tmp_path):
    """The older list read by mistake; leading zeros of the CIK lost on the CSV round trip."""
    old = U.parse_sp500(_page()).iloc[:2]
    old.to_csv(tmp_path / "sp500_2026-09-01.csv", index=False)
    U.parse_sp500(_page()).to_csv(tmp_path / "sp500_2026-10-01.csv", index=False)
    df = U.load_sp500(dest_dir=tmp_path)
    assert len(df) == 5
    assert df.loc[0, "cik"] == "0000012345"
    assert df.loc[2, "cik"] == "" and pd.isna(df.loc[2, "date_added"])
    assert pd.api.types.is_datetime64_any_dtype(df["date_added"])
    assert len(U.load_sp500(tmp_path / "sp500_2026-09-01.csv")) == 2          # an explicit path wins


def test_load_without_any_snapshot_raises(tmp_path):
    """A silent empty universe when nothing was downloaded."""
    with pytest.raises(FileNotFoundError):
        U.load_sp500(dest_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        U.load_sp500(dest_dir=tmp_path / "missing")


def test_load_accepts_a_snapshot_saved_with_wikipedia_headers(tmp_path):
    """A hand-saved copy of the Wikipedia table (Symbol, Security, ...) must load as if parsed by us."""
    raw = pd.DataFrame({"Symbol": ["BRK.B", "AAA"], "Security": ["Berkshire", "Alpha"],
                        "GICS Sector": ["Financials", "Tech"], "GICS Sub-Industry": ["x", "y"],
                        "Date added": ["2010-02-16", ""], "CIK": ["1067983", ""]})
    p = tmp_path / "sp500_2026-10-12.csv"
    raw.to_csv(p, index=False)
    df = U.load_sp500(dest_dir=tmp_path)
    assert list(df["symbol"]) == ["BRK-B", "AAA"]
    assert list(df["raw_symbol"]) == ["BRK.B", "AAA"]
    assert list(df["cik"]) == ["0001067983", ""]
    assert df.loc[0, "date_added"] == pd.Timestamp("2010-02-16") and pd.isna(df.loc[1, "date_added"])
    assert list(df["sector"]) == ["Financials", "Tech"]
    pd.DataFrame({"a": [1]}).to_csv(tmp_path / "sp500_2026-10-13.csv", index=False)
    with pytest.raises(ValueError, match="neither"):
        U.load_sp500(dest_dir=tmp_path)
