"""The Tiingo provider must not splice the wrong company onto a delisted stock or leave a split step in its prices.

Two failures this guards against. (1) A reused ticker accepted as "ours": a later company's history, or a
history that runs years past our delisting date, would be ingested as the dead company and corrupt every
outcome the learning derives from it. (2) A split applied to the wrong side (own factor included, volume not
scaled, adj* fields used): a false 50% jump or a dividend step between delisted and survivor prices.
No test touches the network: every request function is injected or monkeypatched.
"""
import importlib.util
import io
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load_provider():
    # data_engine/__init__ pulls in the database layer and config.settings, so load the module by path.
    spec = importlib.util.spec_from_file_location("tiingo_provider_under_test", ROOT / "data_engine" / "tiingo_provider.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["tiingo_provider_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


tp = _load_provider()


def _row(date, close, volume, split=1.0, **extra):
    row = {"date": date, "open": close, "high": close, "low": close, "close": close, "volume": volume,
           "adjOpen": 1.0, "adjHigh": 1.0, "adjLow": 1.0, "adjClose": 1.0, "adjVolume": 1, "divCash": 0.0, "splitFactor": split}
    row.update(extra)
    return row


# ── split adjustment ──────────────────────────────────────────────────────────────────────────────

def test_split_adjust_by_hand_two_for_one_then_one_for_four_reverse_gives_flat_200():
    """Breaks if the split factor is applied to the wrong side, includes the split day, or volume is not scaled."""
    rows = [
        _row("2020-01-02T00:00:00.000Z", 100.0, 1000),
        _row("2020-01-03T00:00:00.000Z", 100.0, 1000),
        _row("2020-01-06T00:00:00.000Z", 50.0, 2000, split=2.0),       # 2-for-1 on this day: price halves, volume doubles
        _row("2020-01-07T00:00:00.000Z", 50.0, 2000),
        _row("2020-01-08T00:00:00.000Z", 200.0, 500, split=0.25),      # 1-for-4 reverse: price x4, volume /4
    ]
    out = tp.to_split_adjusted(rows)
    assert list(out.columns) == ["date", "open", "high", "low", "close", "volume"]
    assert out["close"].tolist() == pytest.approx([200.0] * 5)
    assert out["open"].tolist() == pytest.approx([200.0] * 5)
    assert out["high"].tolist() == pytest.approx([200.0] * 5)
    assert out["low"].tolist() == pytest.approx([200.0] * 5)
    assert out["volume"].tolist() == pytest.approx([500.0] * 5)        # the adj* garbage (1.0) is never read
    assert out["date"].iloc[0] == pd.Timestamp("2020-01-02") and out["date"].dt.tz is None


def test_split_adjust_leaves_the_latest_prices_untouched_and_never_includes_a_rows_own_factor():
    """Breaks if the last row is scaled, or the split day's own factor leaks into that day's price."""
    rows = [_row("2021-03-01", 10.0, 100), _row("2021-03-02", 5.0, 200, split=2.0), _row("2021-03-03", 5.5, 150)]
    out = tp.to_split_adjusted(rows)
    assert out["close"].tolist() == pytest.approx([5.0, 5.0, 5.5])
    assert out["volume"].tolist() == pytest.approx([200.0, 200.0, 150.0])


@pytest.mark.parametrize("bad", [0, 0.0, None, float("nan"), -2.0])
def test_a_missing_zero_nan_or_negative_split_factor_counts_as_one(bad):
    """Breaks if a bad splitFactor zeroes or NaNs the whole earlier history."""
    rows = [_row("2021-03-01", 10.0, 100), _row("2021-03-02", 10.0, 100, split=bad), _row("2021-03-03", 10.0, 100)]
    out = tp.to_split_adjusted(rows)
    assert out["close"].tolist() == pytest.approx([10.0, 10.0, 10.0])
    assert out["volume"].tolist() == pytest.approx([100.0] * 3)


def test_split_adjust_sorts_unordered_input_and_tolerates_a_missing_split_column():
    """Breaks if the cumulative product runs in input order instead of date order."""
    rows = [_row("2020-01-08", 200.0, 500, split=0.25), _row("2020-01-02", 100.0, 1000), _row("2020-01-06", 50.0, 2000, split=2.0)]
    out = tp.to_split_adjusted(rows)
    assert out["date"].is_monotonic_increasing
    assert out["close"].tolist() == pytest.approx([200.0, 200.0, 200.0])
    plain = [{k: v for k, v in _row("2020-01-02", 7.0, 5).items() if k != "splitFactor"}]
    assert tp.to_split_adjusted(plain)["close"].tolist() == [7.0]


def test_split_adjust_of_no_rows_is_an_empty_frame_with_the_right_columns():
    """Breaks if an empty price list crashes the loader instead of becoming 'no data'."""
    out = tp.to_split_adjusted([])
    assert out.empty and list(out.columns) == ["date", "open", "high", "low", "close", "volume"]


def test_split_adjust_does_not_adjust_for_dividends():
    """Breaks if divCash leaks into prices: the survivor cache is split-adjusted only."""
    rows = [_row("2020-01-02", 10.0, 100, divCash=5.0), _row("2020-01-03", 10.0, 100)]
    assert tp.to_split_adjusted(rows)["close"].tolist() == [10.0, 10.0]


# ── identity ──────────────────────────────────────────────────────────────────────────────────────

OURS = {"name": "Apex Mining Corp", "ipoDate": "2005-03-01", "delistingDate": "2015-03-02"}


def _meta(start="2005-03-01", end="2015-03-02", name="Apex Mining Corp"):
    return {"ticker": "AAA", "name": name, "startDate": start, "endDate": end}


def test_a_ticker_without_metadata_or_dates_is_no_data():
    """Breaks if a 404 or a half-filled metadata record is treated as a candidate."""
    assert tp.identify(None, OURS) == "no_data"
    assert tp.identify({}, OURS) == "no_data"
    assert tp.identify(_meta(end=None), OURS) == "no_data"
    assert tp.identify(_meta(start=None), OURS) == "no_data"


@pytest.mark.parametrize("meta,is_q,expected", [
    (_meta(), False, "ok"),
    (_meta(start="2005-03-01T00:00:00.000Z", end="2015-03-02T00:00:00.000Z"), False, "ok"),     # full timestamps parse
    (_meta(end="2015-04-16"), False, "ok"),                                                   # delisting + 45 days
    (_meta(end="2015-04-17"), False, "reused"),                                               # +46: it kept trading as someone else
    (_meta(end="2015-01-16"), False, "ok"),                                                   # delisting - 45 days
    (_meta(end="2015-01-15"), False, "mismatch"),                                             # ended well before our delisting
    (_meta(end="2018-06-01", name="Apex Biotech Inc"), False, "reused"),                      # the date rule alone decides
    (_meta(end="2017-05-10"), True, "ok"),                                                    # Q ticker: delisting + 800 days (asserted below)
    (_meta(end="2017-05-11"), True, "mismatch"),                                              # +801: the S-only 'reused' rule does not apply
    (_meta(end="2015-04-17"), True, "ok"),
    (_meta(start="2005-04-30"), False, "ok"),                                                 # ipo + 60 days
    (_meta(start="2005-05-01"), False, "mismatch"),                                           # ipo + 61 days
    (_meta(start="2005-05-01", name="Zeta Robotics Inc"), False, "reused"),                   # dates fail and the names differ
    (_meta(start="2005-05-01", name=""), False, "mismatch"),                                  # no name: the name rule is skipped
])
def test_identify_table_around_every_tolerance_boundary(meta, is_q, expected):
    """Breaks if a date tolerance or the reused/mismatch split moves by even a day."""
    assert tp.identify(meta, OURS, is_q) == expected


def test_the_q_ticker_boundary_dates_in_the_table_are_exactly_800_and_801_days_after_delisting():
    """Guards the table above against a calendar slip (2015-03-02 + 800 days is 2017-05-10, not 05-11)."""
    delist = pd.Timestamp(OURS["delistingDate"])
    assert (pd.Timestamp("2017-05-10") - delist).days == 800 and (pd.Timestamp("2017-05-11") - delist).days == 801
    assert (pd.Timestamp("2015-04-16") - delist).days == 45 and (pd.Timestamp("2015-01-16") - delist).days == -45


def test_an_ipo_date_before_2000_gets_the_wide_start_tolerance_and_one_on_or_after_does_not():
    """Breaks if the old-IPO assumption (AV ipoDate unreliable before 2000) is lost or applied to later IPOs."""
    old = {"name": "Apex Mining Corp", "ipoDate": "1990-01-02", "delistingDate": "2015-03-02"}
    assert tp.identify(_meta(start="1990-12-28"), old) == "ok"
    assert tp.identify(_meta(start="1991-01-02"), old) == "ok"          # ipo + 365
    assert tp.identify(_meta(start="1991-01-03"), old) == "mismatch"    # ipo + 366
    edge = dict(old, ipoDate="2000-01-01")
    assert tp.identify(_meta(start="2000-03-02"), edge) == "mismatch"   # 61 days: the cut-off date itself is NOT 'old'
    assert tp.identify(_meta(start="2000-03-01"), edge) == "ok"
    before = dict(old, ipoDate="1999-12-31")
    assert tp.identify(_meta(start="2000-03-02"), before) == "ok"
    assert tp.wide_tolerance(before) and not tp.wide_tolerance(edge)


def test_start_before_ipo_days_reports_how_much_earlier_tiingo_starts():
    """Breaks if a predecessor company's early history cannot be spotted afterwards in the state file."""
    assert tp.start_before_ipo_days(_meta(start="2003-03-01"), OURS) == (pd.Timestamp("2005-03-01") - pd.Timestamp("2003-03-01")).days
    assert tp.start_before_ipo_days(_meta(start="2005-03-11"), OURS) == -10


def test_name_overlap_uses_distinctive_words_only():
    """Breaks if generic words (corp, holdings, ...) count as evidence of the same company."""
    assert tp.name_overlap("Apex Mining Corp", "Apex Biotech Inc") is True
    assert tp.name_overlap("Apex Mining Corp", "Zeta Robotics Inc") is False
    assert tp.name_overlap("Alpha Holdings Group", "Beta Holdings Group") is False       # only stop words in common
    assert tp.name_overlap("", "Apex") is None and tp.name_overlap("Apex", None) is None
    assert tp.name_overlap(float("nan"), "Apex") is None
    assert tp.name_overlap("Big Co", "Big Co") is False                                  # no 4-letter word on either side


# ── universe ──────────────────────────────────────────────────────────────────────────────────────

def _av_row(symbol, name="Alpha Corp", asset="Stock", ipo="2005-01-03", delist="2012-06-01"):
    return {"symbol": symbol, "name": name, "exchange": "NYSE", "assetType": asset, "ipoDate": ipo, "delistingDate": delist, "status": "Delisted"}


def test_select_universe_keeps_only_real_delisted_common_stocks_in_range():
    """Breaks if SPAC shells, ETFs, snapshot-day rows, short-lived listings or survivors enter the Tiingo job."""
    av = pd.DataFrame([
        _av_row("GOODA"),
        _av_row("AA", name="Alcoa When Issued"),                                    # AA is a fine symbol but the name says when-issued
        _av_row("AA-W"),                                                            # not [A-Z]{1,5}
        _av_row("TOOLONG"),                                                         # 7 letters
        _av_row("lower"),
        _av_row("WARR", name="Zeta Warrants Class A"),
        _av_row("ETFX", asset="ETF"),
        _av_row("SNAP", delist="2026-10-05"),                                       # the vendor's snapshot day, not a delisting
        _av_row("SHRT", ipo="2011-08-05", delist="2012-06-01"),                     # listed ~300 days
        _av_row("EXACT", ipo="2011-06-01", delist="2012-05-31"),                    # listed exactly 365 days: not > 365
        _av_row("OLDY", delist="2009-12-31"),
        _av_row("FIRST", delist="2010-01-04"),                                      # from_year boundary
        _av_row("LAST", delist="2025-12-31"),                                       # to_year boundary
        _av_row("NEXT", delist="2026-01-02"),
        _av_row("SURV"),                                                            # in the survivor cache
        _av_row("GOODB", name=None),                                                # a missing name is not evidence of a SPAC
        _av_row("NODATE", ipo=None),
    ])
    out = tp.select_universe(av, 2010, 2025, exclude=frozenset({"SURV"}))
    assert list(out["symbol"]) == ["FIRST", "GOODA", "GOODB", "LAST"]
    assert out["delistingDate"].is_monotonic_increasing and pd.api.types.is_datetime64_any_dtype(out["ipoDate"])


# ── throttle ──────────────────────────────────────────────────────────────────────────────────────

class _FakeTime:
    def __init__(self, t):
        self.t, self.sleeps = float(t), []

    def clock(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += s


def test_the_third_request_in_an_hour_sleeps_to_the_next_hour_boundary_plus_one_second():
    """Breaks if the hourly budget is ignored (Tiingo would answer 429 and the monthly allowance burns on retries)."""
    ft = _FakeTime(7200.0)
    lim = tp.RateLimiter(2, 100, clock=ft.clock, sleep=ft.sleep)
    lim.before_request()
    lim.before_request()
    assert ft.sleeps == []
    lim.before_request()
    assert ft.sleeps == [3601.0]
    lim.before_request()                                                         # new hour, second request of it
    assert ft.sleeps == [3601.0]
    lim.before_request()                                                         # hour is full again; the clock is now 10801.0
    assert ft.sleeps == [3601.0, 3600.0]


def test_a_new_clock_hour_resets_the_hourly_count_without_sleeping():
    """Breaks if the loader sleeps although the hour has already rolled over."""
    ft = _FakeTime(7200.0)
    lim = tp.RateLimiter(2, 100, clock=ft.clock, sleep=ft.sleep)
    lim.before_request()
    lim.before_request()
    ft.t += 3600.0
    lim.before_request()
    lim.before_request()
    assert ft.sleeps == []


def test_the_daily_cap_raises_instead_of_sleeping_and_resets_on_the_next_day():
    """Breaks if a run keeps asking after the day's budget is gone, or never recovers on the next day."""
    ft = _FakeTime(86400.0 * 3)
    lim = tp.RateLimiter(100, 3, clock=ft.clock, sleep=ft.sleep)
    for _ in range(3):
        lim.before_request()
    with pytest.raises(tp.DailyCapReached):
        lim.before_request()
    assert ft.sleeps == []
    ft.t += 86400.0
    lim.before_request()


def test_a_day_count_carried_over_from_an_earlier_process_counts_against_the_cap():
    """Breaks if a restart forgets what was already requested today."""
    ft = _FakeTime(86400.0 * 3)
    lim = tp.RateLimiter(100, 3, clock=ft.clock, sleep=ft.sleep)
    lim.day_count = 2
    lim.before_request()
    with pytest.raises(tp.DailyCapReached):
        lim.before_request()


def test_a_429_sleeps_to_the_next_hour_and_clears_the_hourly_count():
    """Breaks if a 429 is answered with a short retry that burns the rest of the allowance."""
    ft = _FakeTime(7300.0)
    lim = tp.RateLimiter(1, 100, clock=ft.clock, sleep=ft.sleep)
    lim.before_request()
    lim.on_429()
    assert ft.sleeps == [3501.0]
    lim.before_request()                                                         # count was cleared: no second sleep
    assert ft.sleeps == [3501.0]


# ── urls and transport ────────────────────────────────────────────────────────────────────────────

def test_urls_and_auth_headers():
    """Breaks if the endpoint shape changes or the token ends up anywhere but the Authorization header."""
    assert tp.meta_url("AAA") == "https://api.tiingo.com/tiingo/daily/AAA"
    assert tp.prices_url("AAAQ", "2005-03-01", "2015-02-27") == \
        "https://api.tiingo.com/tiingo/daily/AAAQ/prices?startDate=2005-03-01&endDate=2015-02-27&format=json"
    h = tp.auth_headers("TOK")
    assert h == {"Authorization": "Token TOK", "Content-Type": "application/json"}
    assert "TOK" not in tp.meta_url("AAA") + tp.prices_url("AAA", "2000-01-01", "2001-01-01")


def test_default_fetch_returns_http_errors_as_a_status_instead_of_raising(monkeypatch):
    """Breaks if a 404 for an unknown ticker aborts the run instead of meaning 'no such ticker'."""
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["auth"], seen["timeout"] = req.get_header("Authorization"), timeout
        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, io.BytesIO(b'{"detail":"Not found."}'))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    status, body = tp.default_fetch(tp.meta_url("NOPE"), tp.auth_headers("TOK"))
    assert (status, body) == (404, b'{"detail":"Not found."}')
    assert seen == {"auth": "Token TOK", "timeout": 30}


def test_default_fetch_returns_status_and_body_on_success(monkeypatch):
    """Breaks if the success path returns the response object instead of (status, bytes)."""
    class _Resp(io.BytesIO):
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: _Resp(b"[1]"))
    assert tp.default_fetch("https://example.invalid/x", {}) == (200, b"[1]")


def test_the_module_imports_no_network_or_parquet_library_at_top_level():
    """Breaks the project rule that library modules import urllib.request / pyarrow / requests / yfinance lazily."""
    src = (ROOT / "data_engine" / "tiingo_provider.py").read_text().split("\n")
    top = [ln for ln in src if ln.startswith(("import ", "from "))]
    banned = ("pyarrow", "yfinance", "requests", "urllib.request")
    assert not [ln for ln in top if any(b in ln for b in banned)], top
