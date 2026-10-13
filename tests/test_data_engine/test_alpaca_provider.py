"""The Alpaca provider must not leak a credential, silently use unadjusted or IEX-only bars, or lose a page of data.

Failure groups this guards against. (1) Secrets: a key or token printed to a log, put in a URL or echoed in an
exception message. (2) Request shape: a missing `adjustment`, the wrong default feed, a window that includes
future data on the free plan, symbol lists above 100, corporate-action windows above 90 days. (3) Data loss:
a lost page, a repeated page token, a 429 that is not retried, a date shifted by the bar's time-of-day stamp.
(4) Auth flow: an OAuth token that is never cached, never refreshed, or refreshed forever after a 401.
No test touches the network: every request goes through an injected fake `fetch` or a monkeypatched urlopen.
"""
import importlib.util
import io
import json
import logging
import sys
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load_provider():
    # data_engine/__init__ pulls in the database layer and config.settings, so load the module by path.
    spec = importlib.util.spec_from_file_location("alpaca_provider_under_test", ROOT / "data_engine" / "alpaca_provider.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["alpaca_provider_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


ap = _load_provider()

KEYS = {"ALPACA_API_KEY": "TEST-KEY-ID", "ALPACA_SECRET_KEY": "TEST-SECRET"}
OAUTH = {"ALPACA_CLIENT_ID": "TEST-CLIENT-ID", "ALPACA_CLIENT_SECRET": "TEST-CLIENT-SECRET"}
SECRETS = ("TEST-KEY-ID", "TEST-SECRET", "TEST-CLIENT-ID", "TEST-CLIENT-SECRET")
UTC_NOW = pd.Timestamp("2026-10-13T15:00:00Z")
CLOCK_URL = "https://paper-api.alpaca.markets/v2/clock"


class FakeTime:
    """A controllable clock: sleeping advances it, so the throttle and the backoff see consistent time."""

    def __init__(self):
        self.t = 1000.0
        self.sleeps = []

    def now(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += s


class FakeHTTP:
    """Pops one (status, payload) per call and records the call; an exhausted list raises IndexError (never hangs)."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, method, url, headers, body=None):
        self.calls.append({"method": method, "url": url, "headers": dict(headers), "body": body})
        status, payload = self.responses.pop(0)
        return status, payload if isinstance(payload, bytes) else json.dumps(payload).encode()


def make(responses, env=KEYS, plan="free"):
    ft, http = FakeTime(), FakeHTTP(responses)
    c = ap.AlpacaClient(plan=plan, fetch=http, now=ft.now, sleep=ft.sleep, utcnow=lambda: UTC_NOW, env=dict(env))
    return c, http, ft


def q(url):
    return parse_qs(urlsplit(url).query)


def page(bars, token=None):
    return {"bars": bars, "next_page_token": token}


def bar(t, c=10.0):
    return {"t": t, "o": c, "h": c + 1, "l": c - 1, "c": c, "v": 1000, "n": 7, "vw": c}


def _bearer(call):
    return call["headers"]["Authorization"]


# ── authentication ────────────────────────────────────────────────────────────────────────────────

def test_the_key_and_bearer_headers_are_exactly_alpacas_and_a_keys_client_sends_them():
    """Breaks if a header name is wrong (every request then answers 401) or the keys client sends no auth."""
    assert ap.key_headers("TEST-KEY-ID", "TEST-SECRET") == {"APCA-API-KEY-ID": "TEST-KEY-ID", "APCA-API-SECRET-KEY": "TEST-SECRET"}
    assert ap.bearer_headers("TOK-1") == {"Authorization": "Bearer TOK-1"}
    c, http, _ = make([(200, {"is_open": False})])
    assert c.clock() == {"is_open": False}
    assert http.calls[0]["headers"] == ap.key_headers("TEST-KEY-ID", "TEST-SECRET")
    assert http.calls[0]["method"] == "GET"
    assert http.calls[0]["url"] == CLOCK_URL


def test_credentials_precedence_missing_and_partial_cases_and_the_error_texts_hold_no_values():
    """Breaks if keys lose precedence over OAuth, a blank value counts as set, or an error text echoes a secret."""
    assert ap.credentials({**KEYS, **OAUTH}).scheme == "keys"
    assert ap.credentials(OAUTH).scheme == "oauth"
    assert ap.credentials({"ALPACA_API_KEY": "  ", **OAUTH}).scheme == "oauth"
    assert repr(ap.credentials(KEYS)) == "Credentials(scheme='keys')"
    with pytest.raises(ap.AlpacaConfigError) as e:
        ap.credentials({})
    for name in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "ALPACA_CLIENT_ID", "ALPACA_CLIENT_SECRET"):
        assert name in str(e.value)
    with pytest.raises(ap.AlpacaConfigError) as e:
        ap.credentials({"ALPACA_API_KEY": "TEST-KEY-ID"})
    assert str(e.value) == "ALPACA_API_KEY is set but ALPACA_SECRET_KEY is not"
    assert "TEST-KEY-ID" not in str(e.value)
    with pytest.raises(ap.AlpacaConfigError) as e:
        ap.credentials({"ALPACA_CLIENT_ID": "TEST-CLIENT-ID"})
    assert "TEST-CLIENT-ID" not in str(e.value)


@pytest.mark.parametrize("value, host", [
    (None, ap.TRADING_PAPER), ("true", ap.TRADING_PAPER), ("FALSE", ap.TRADING_LIVE), ("False", ap.TRADING_LIVE),
    ("false", ap.TRADING_LIVE), ("no", ap.TRADING_PAPER), ("0", ap.TRADING_PAPER), (" false", ap.TRADING_PAPER),
])
def test_trading_base_goes_live_only_for_the_exact_word_false_in_any_case(value, host):
    """Breaks if the paper/live rule drifts from broker_engine/alpaca.py (unset must be paper, ' false' must be paper)."""
    env = {} if value is None else {"ALPACA_PAPER": value}
    assert ap.trading_base(env) == host


def test_a_client_with_paper_false_calls_the_live_trading_host():
    """Breaks if the client ignores ALPACA_PAPER and reads the wrong account host."""
    c, http, _ = make([(200, {"is_open": True})], env={**KEYS, "ALPACA_PAPER": "false"})
    c.clock()
    assert http.calls[0]["url"] == "https://api.alpaca.markets/v2/clock"


def test_no_secret_reaches_a_log_a_url_a_repr_or_an_exception_text(caplog):
    """Breaks if a key, a secret or a token appears in the log, a URL, a repr or an error message (a body echo is scrubbed)."""
    caplog.set_level(logging.DEBUG)
    c, http, _ = make([(200, {"is_open": False}), (401, {"message": "no"}),
                       (422, {"message": "bad key TEST-SECRET / TEST-KEY-ID"})])
    c.clock()
    with pytest.raises(ap.AlpacaAuthError) as auth:
        c.clock()
    with pytest.raises(ap.AlpacaError) as other:
        c.clock()
    assert "***" in str(other.value) and "HTTP 422" in str(other.value)
    for s in SECRETS:
        assert s not in repr(c) and s not in repr(c._creds)
        assert s not in str(auth.value) and s not in str(other.value)
        assert s not in caplog.text
        assert all(s not in call["url"] for call in http.calls)
    assert "/v2/clock" in caplog.text

    caplog.clear()
    c2, http2, _ = make([(200, {"access_token": "TOK-1", "expires_in": 900}), (200, {"is_open": True})], env=OAUTH)
    c2.clock()
    for s in SECRETS + ("TOK-1",):
        assert s not in caplog.text and s not in repr(c2)
        assert all(s not in call["url"] for call in http2.calls)
    assert "/v2/clock" in caplog.text


def test_the_oauth_token_request_is_a_form_post_and_the_next_call_carries_the_bearer():
    """Breaks if the token exchange sends the wrong grant, the wrong body or content type, or the bearer is not used."""
    c, http, _ = make([(200, {"access_token": "TOK-1", "expires_in": 900, "token_type": "Bearer"}), (200, {"is_open": True})], env=OAUTH)
    c.clock()
    first = http.calls[0]
    assert first["method"] == "POST" and first["url"] == ap.OAUTH_TOKEN_URL
    assert first["headers"] == {"Content-Type": "application/x-www-form-urlencoded"}
    assert parse_qs(first["body"].decode()) == {"grant_type": ["client_credentials"], "client_id": ["TEST-CLIENT-ID"],
                                                "client_secret": ["TEST-CLIENT-SECRET"]}
    assert http.calls[1]["headers"] == {"Authorization": "Bearer TOK-1"}


def test_the_token_is_cached_until_expires_in_minus_30_seconds():
    """Breaks if the token is fetched on every call, never refreshed, or refreshed before/after the 30 s skew."""
    c, http, ft = make([(200, {"access_token": "TOK-1", "expires_in": 900}), (200, {"is_open": True}), (200, {"is_open": True}),
                        (200, {"access_token": "TOK-2", "expires_in": 900}), (200, {"is_open": True})], env=OAUTH)
    c.clock()
    assert c.token_requests == 1
    ft.t = 1869.0                       # 1000 + 900 - 30 = 1870: one second before expiry
    c.clock()
    assert c.token_requests == 1 and _bearer(http.calls[-1]) == "Bearer TOK-1"
    ft.t = 1870.0
    c.clock()
    assert c.token_requests == 2 and _bearer(http.calls[-1]) == "Bearer TOK-2"


def test_a_token_without_expires_in_lasts_300_seconds_less_the_skew():
    """Breaks if the default lifetime is not 300 s (1000 + 300 - 30 = 1270)."""
    c, http, ft = make([(200, {"access_token": "TOK-1"}), (200, {"is_open": True}), (200, {"is_open": True}),
                        (200, {"access_token": "TOK-2"}), (200, {"is_open": True})], env=OAUTH)
    c.clock()
    ft.t = 1269.0
    c.clock()
    assert c.token_requests == 1
    ft.t = 1270.0
    c.clock()
    assert c.token_requests == 2 and _bearer(http.calls[-1]) == "Bearer TOK-2"


def test_a_401_on_oauth_refreshes_the_token_once_and_a_second_401_raises():
    """Breaks if an expired token is not renewed, or a bad credential loops forever instead of raising."""
    c, http, _ = make([(200, {"access_token": "TOK-1", "expires_in": 900}), (401, {}),
                       (200, {"access_token": "TOK-2", "expires_in": 900}), (200, {"is_open": True})], env=OAUTH)
    assert c.clock() == {"is_open": True}
    assert c.token_requests == 2 and len(http.calls) == 4
    assert _bearer(http.calls[3]) == "Bearer TOK-2"

    c, http, _ = make([(200, {"access_token": "TOK-1", "expires_in": 900}), (401, {}),
                       (200, {"access_token": "TOK-2", "expires_in": 900}), (401, {})], env=OAUTH)
    with pytest.raises(ap.AlpacaAuthError):
        c.clock()
    assert len(http.calls) == 4


def test_a_403_never_refreshes_and_a_keys_401_raises_at_once_without_sleeping():
    """Breaks if a missing subscription (403) triggers token churn, or a bad key is retried."""
    c, http, _ = make([(200, {"access_token": "TOK-1", "expires_in": 900}), (403, {})], env=OAUTH)
    with pytest.raises(ap.AlpacaAuthError):
        c.clock()
    assert len(http.calls) == 2
    c, http, ft = make([(401, {})])
    with pytest.raises(ap.AlpacaAuthError):
        c.clock()
    assert len(http.calls) == 1 and ft.sleeps == []


# ── throttle, retries, errors ─────────────────────────────────────────────────────────────────────

def test_the_throttle_spaces_requests_by_the_plan_rate_and_unknown_plans_are_refused():
    """Breaks if calls are not spaced 0.4 s (free) / 0.1 s (plus), a gap already long enough still sleeps, or a plan typo passes."""
    c, _, ft = make([(200, {})] * 3)
    for _ in range(3):
        c.clock()
    assert ft.sleeps == pytest.approx([0.4, 0.4])
    c, _, ft = make([(200, {})] * 3, plan="plus")
    for _ in range(3):
        c.clock()
    assert ft.sleeps == pytest.approx([0.1, 0.1])
    c, _, ft = make([(200, {})] * 3)
    for _ in range(3):
        c.clock()
        ft.t += 1.0
    assert ft.sleeps == []
    with pytest.raises(ValueError):
        make([], plan="gold")


def test_429_and_5xx_retry_with_the_1_2_4_backoff_then_raise_the_right_error():
    """Breaks if a 429 is not retried, the backoff schedule changes, or the final error class is wrong."""
    c, http, ft = make([(429, {}), (200, {"is_open": False})])
    assert c.clock() == {"is_open": False}
    assert ft.sleeps == [1] and c.requests == 2

    c, http, ft = make([(429, {})] * 4)
    with pytest.raises(ap.AlpacaRateLimited):
        c.clock()
    assert ft.sleeps == [1, 2, 4] and len(http.calls) == 4

    c, http, ft = make([(503, {}), (500, {}), (200, {})])
    c.clock()
    assert ft.sleeps == [1, 2]

    c, http, ft = make([(502, {})] * 4)
    with pytest.raises(ap.AlpacaServerError):
        c.clock()
    assert ft.sleeps == [1, 2, 4]


def test_a_404_is_none_for_asset_and_an_error_for_everything_else():
    """Breaks if an unknown ticker aborts a probe, or a 404 on bars is silently read as 'no data'."""
    c, http, _ = make([(404, {"message": "not found"})])
    assert c.asset("ZZZZ") is None
    assert http.calls[0]["url"] == "https://paper-api.alpaca.markets/v2/assets/ZZZZ"
    c, _, _ = make([(404, {"message": "not found"})])
    with pytest.raises(ap.AlpacaError):
        c.bars(["AAPL"], "2020-03-02", "2020-03-03")


# ── assets ────────────────────────────────────────────────────────────────────────────────────────

def _asset(i, sym, exch):
    return {"id": i, "class": "us_equity", "exchange": exch, "symbol": sym, "name": f"{sym} Inc", "status": "inactive",
            "tradable": False, "marginable": True}


def test_assets_keep_the_exchange_column_and_send_only_the_requested_filters():
    """Breaks if OTC rows cannot be told apart (no exchange), an extra field leaks in, or a filter is not sent."""
    rows = [_asset("a1", "AAA", "NYSE"), _asset("a2", "BBB", "OTC"), _asset("a3", "CCC", "NASDAQ")]
    c, http, _ = make([(200, rows), (200, rows), (200, [])])
    df = c.assets()
    assert list(df.columns) == ap.ASSET_COLUMNS
    assert df["exchange"].tolist() == ["NYSE", "OTC", "NASDAQ"]
    assert q(http.calls[0]["url"]) == {"status": ["inactive"], "asset_class": ["us_equity"]}
    c.assets(status="active", exchange="NYSE")
    assert q(http.calls[1]["url"]) == {"status": ["active"], "asset_class": ["us_equity"], "exchange": ["NYSE"]}
    empty = c.assets()
    assert len(empty) == 0 and list(empty.columns) == ap.ASSET_COLUMNS


def test_an_unknown_asset_status_is_refused_before_any_request():
    """Breaks if a status typo reaches the API instead of failing fast."""
    c, http, _ = make([])
    with pytest.raises(ValueError):
        c.assets(status="all")
    assert http.calls == []


# ── bars: request shape ───────────────────────────────────────────────────────────────────────────

def test_bars_query_defaults_always_send_adjustment_and_the_iex_feed_and_the_day_boundaries():
    """Breaks if adjustment is omitted (API default is raw), the default feed changes, or the end day is cut off."""
    c, http, _ = make([(200, page({})), (200, page({})), (200, page({}))])
    c.bars(["AAPL"], "2020-03-02", "2020-03-03")
    url = http.calls[0]["url"]
    assert url.startswith("https://data.alpaca.markets/v2/stocks/bars?")
    assert q(url) == {"symbols": ["AAPL"], "timeframe": ["1Day"], "start": ["2020-03-02T00:00:00Z"], "end": ["2020-03-03T23:59:59Z"],
                      "adjustment": ["split"], "feed": ["iex"], "limit": ["10000"], "sort": ["asc"]}
    c.bars(["AAPL"], "2020-03-02", "2020-03-03", adjustment="raw")
    assert q(http.calls[1]["url"])["adjustment"] == ["raw"]
    c.bars(["AAPL"], "2020-03-02", "2020-03-03", asof="2019-06-13")
    assert q(http.calls[2]["url"])["asof"] == ["2019-06-13"]


@pytest.mark.parametrize("symbols, start, end, kw", [
    (["AAPL"], "2020-03-02", "2020-03-03", {"adjustment": "splits"}),
    (["AAPL"], "2020-03-02", "2020-03-03", {"feed": "SIP"}),
    (["AAPL"], "2020-03-02", "2020-03-03", {"feed": "delayed"}),
    (["AAPL"], "2020-03-02", "2020-03-03", {"chunk": 0}),
    (["AAPL"], "2020-03-02", "2020-03-03", {"chunk": 101}),
    ([], "2020-03-02", "2020-03-03", {}),
    (["A,B"], "2020-03-02", "2020-03-03", {}),
    ([""], "2020-03-02", "2020-03-03", {}),
    (["AAPL"], "2020-03-05", "2020-03-02", {}),
])
def test_invalid_bar_arguments_raise_value_error_and_make_no_request(symbols, start, end, kw):
    """Breaks if a bad adjustment, feed, chunk, symbol list or date order reaches the network."""
    c, http, _ = make([])
    with pytest.raises(ValueError):
        c.bars(symbols, start, end, **kw)
    assert http.calls == []


def test_paging_follows_the_token_to_the_end_and_merges_and_sorts_the_pages():
    """Breaks if only the first page is read, a page token is not sent back, or the merged bars are unsorted."""
    c, http, _ = make([
        (200, page({"AAPL": [bar("2020-03-02T05:00:00Z", 1)]}, "P2")),
        (200, page({"AAPL": [bar("2020-03-03T05:00:00Z", 2)], "MSFT": [bar("2020-03-02T05:00:00Z", 3)]}, "P3")),
        (200, page({"MSFT": [bar("2020-03-03T05:00:00Z", 4)]}, None)),
    ])
    df = c.bars(["AAPL", "MSFT"], "2020-03-02", "2020-03-03")
    assert len(http.calls) == 3
    assert "page_token" not in q(http.calls[0]["url"])
    assert q(http.calls[1]["url"])["page_token"] == ["P2"]
    assert q(http.calls[2]["url"])["page_token"] == ["P3"]
    assert df["symbol"].tolist() == ["AAPL", "AAPL", "MSFT", "MSFT"]
    assert df["close"].tolist() == [1.0, 2.0, 3.0, 4.0]


def test_a_repeated_page_token_stops_the_loop_with_an_error():
    """Breaks if a server that repeats its token makes the client loop forever."""
    c, _, _ = make([(200, page({}, "P2")), (200, page({}, "P2"))])
    with pytest.raises(ap.AlpacaError, match="repeated a page token"):
        c.bars(["AAPL"], "2020-03-02", "2020-03-03")


def test_symbols_go_out_in_chunks_of_at_most_100_deduplicated_and_upper_cased():
    """Breaks if 250 symbols go in one request, the chunk size changes, or case/whitespace duplicates are sent twice."""
    syms = [f"S{i:03d}" for i in range(250)]
    c, http, _ = make([(200, page({}))] * 3)
    df = c.bars(syms, "2020-03-02", "2020-03-03")
    sent = [q(call["url"])["symbols"][0].split(",") for call in http.calls]
    assert [len(s) for s in sent] == [100, 100, 50]
    assert sent[0] == syms[:100]
    assert len(df) == 0 and list(df.columns) == ap.BAR_COLUMNS
    assert df.attrs["adjustment"] == "split"
    c, http, _ = make([(200, page({}))])
    c.bars(["aapl", " AAPL ", "AAPL"], "2020-03-02", "2020-03-03")
    assert len(http.calls) == 1 and q(http.calls[0]["url"])["symbols"] == ["AAPL"]


# ── bars: result frame ────────────────────────────────────────────────────────────────────────────

def test_daily_timestamps_become_naive_midnight_dates_and_hourly_ones_keep_their_time():
    """Breaks if a 04:00/05:00Z daily stamp shifts the date, the dates stay tz-aware, or a field is mis-mapped."""
    c, _, _ = make([(200, page({"AAPL": [bar("2020-03-02T05:00:00Z"), bar("2020-07-01T04:00:00Z")]}))])
    df = c.bars(["AAPL"], "2020-03-02", "2020-07-01")
    assert pd.api.types.is_datetime64_any_dtype(df["date"]) and df["date"].dt.tz is None
    assert df["date"].tolist() == [pd.Timestamp("2020-03-02"), pd.Timestamp("2020-07-01")]
    first = df.iloc[0]
    assert (first["open"], first["high"], first["low"], first["close"]) == (10.0, 11.0, 9.0, 10.0)
    assert (first["volume"], first["trades"], first["vwap"]) == (1000.0, 7.0, 10.0)
    assert df.attrs == {"adjustment": "split", "feed": "iex", "timeframe": "1Day"}
    c, _, _ = make([(200, page({"AAPL": [bar("2020-03-02T15:00:00Z")]}))])
    hourly = c.bars(["AAPL"], "2020-03-02", "2020-03-02", timeframe="1Hour")
    assert hourly["date"].iloc[0] == pd.Timestamp("2020-03-02 15:00:00")


@pytest.mark.parametrize("plan, feed, start, end, expected_end", [
    ("free", "iex", "2020-03-02", "2026-10-13", "2026-10-13T14:44:00Z"),
    ("free", "sip", "2020-03-02", "2026-10-13", "2026-10-13T14:44:00Z"),
    ("free", "iex", "2020-03-02", "2026-10-13T14:50:30Z", "2026-10-13T14:44:00Z"),
    ("free", "iex", "2020-03-02", "2026-10-12", "2026-10-12T23:59:59Z"),
    ("plus", "iex", "2020-03-02", "2026-10-13", "2026-10-13T23:59:59Z"),
])
def test_the_free_plan_trims_end_to_16_minutes_before_now_for_every_feed(plan, feed, start, end, expected_end):
    """Breaks if the free plan asks for data newer than 16 minutes (403 on SIP), or a paid plan or an old end is clipped."""
    c, http, _ = make([(200, page({}))], plan=plan)
    c.bars(["AAPL"], start, end, feed=feed)
    assert q(http.calls[0]["url"])["end"] == [expected_end]


def test_a_start_after_the_trimmed_end_raises_without_a_request():
    """Breaks if an empty window after the trim is sent to the API."""
    c, http, _ = make([])
    with pytest.raises(ValueError):
        c.bars(["AAPL"], "2026-10-13T14:50:00Z", "2026-10-13")
    assert http.calls == []


@pytest.mark.parametrize("payload", [page(None), page({}), page({"AAPL": []})])
def test_no_bars_give_an_empty_frame_with_the_columns_and_attrs(payload):
    """Breaks if an empty answer raises, loses the column order, or drops the adjustment record."""
    c, _, _ = make([(200, payload)])
    df = c.bars(["AAPL"], "2020-03-02", "2020-03-03")
    assert len(df) == 0 and list(df.columns) == ap.BAR_COLUMNS
    assert df.attrs == {"adjustment": "split", "feed": "iex", "timeframe": "1Day"}


# ── corporate actions ─────────────────────────────────────────────────────────────────────────────

WINDOWS = [("2024-01-01", "2024-03-30"), ("2024-03-31", "2024-06-28"), ("2024-06-29", "2024-07-18")]


def test_ca_windows_are_90_calendar_days_inclusive_with_no_gap_and_no_overlap():
    """Breaks if a window exceeds the endpoint's 90-day limit, leaves a gap, or repeats a day."""
    assert ap.ca_windows("2024-01-01", "2024-07-18") == WINDOWS
    assert ap.ca_windows("2024-01-01", "2024-03-30") == [("2024-01-01", "2024-03-30")]
    assert ap.ca_windows("2024-01-01", "2024-03-31") == [("2024-01-01", "2024-03-30"), ("2024-03-31", "2024-03-31")]
    assert ap.ca_windows("2024-05-05", "2024-05-05") == [("2024-05-05", "2024-05-05")]
    with pytest.raises(ValueError):
        ap.ca_windows("2024-02-01", "2024-01-01")
    # calendar guard for the hand values above (2024 is a leap year)
    assert (pd.Timestamp("2024-03-30") - pd.Timestamp("2024-01-01")).days == 89
    assert (pd.Timestamp("2024-06-28") - pd.Timestamp("2024-03-31")).days == 89
    assert (pd.Timestamp("2024-07-18") - pd.Timestamp("2024-01-01")).days == 199


def _ca(i, typ, sym, ex, cash, old="1", new="1", record=None, payable=None):
    return {"id": i, "ca_type": typ, "initiating_symbol": sym, "target_symbol": sym, "ex_date": ex, "record_date": record,
            "payable_date": payable, "cash": cash, "old_rate": old, "new_rate": new}


def test_corporate_actions_request_one_call_per_window_and_parse_the_rows():
    """Breaks if windows are not requested separately, the type casing/list changes, or the frame loses columns or types."""
    c, http, _ = make([
        (200, [_ca("ca1", "split", "XYZ", "2024-02-01", "0", "1", "4", record="2024-01-25", payable="2024-01-31")]),
        (200, [_ca("ca2", "dividend", "ABC", "2024-05-10", "0.25")]),
        (200, []),
    ])
    df = c.corporate_actions("2024-01-01", "2024-07-18")
    assert len(http.calls) == 3
    for call, (a, b) in zip(http.calls, WINDOWS):
        u = urlsplit(call["url"])
        assert u.netloc == "paper-api.alpaca.markets" and u.path == "/v2/corporate_actions/announcements"
        assert q(call["url"])["since"] == [a] and q(call["url"])["until"] == [b]
        assert q(call["url"])["ca_types"] == ["Split,Dividend,Merger,Spinoff"]
        assert "symbol" not in q(call["url"])
    assert list(df.columns) == ap.CA_COLUMNS
    assert df["id"].tolist() == ["ca1", "ca2"]
    assert df["split_from"].iloc[0] == 1.0 and df["split_to"].iloc[0] == 4.0
    assert df["cash"].iloc[1] == 0.25
    assert df["ex_date"].iloc[0] == pd.Timestamp("2024-02-01")


def test_corporate_actions_per_symbol_sends_each_symbol_for_each_window():
    """Breaks if the symbol filter is dropped or the symbol loop and the window loop are mixed up."""
    c, http, _ = make([(200, [])] * 6)
    c.corporate_actions("2024-01-01", "2024-07-18", symbols=["XYZ", "ABC"])
    assert len(http.calls) == 6
    assert [q(call["url"])["symbol"] for call in http.calls] == [["XYZ"]] * 3 + [["ABC"]] * 3


def test_corporate_actions_refuse_unknown_types_collapse_duplicate_ids_and_state_they_are_unverified():
    """Breaks if a type typo reaches the API, a row announced in two windows counts twice, or the UNVERIFIED warning is lost."""
    c, http, _ = make([])
    with pytest.raises(ValueError):
        c.corporate_actions("2024-01-01", "2024-01-31", types=("delisting",))
    assert http.calls == []
    row = _ca("ca1", "split", "XYZ", "2024-03-30", "0", "1", "2")
    c, _, _ = make([(200, [row]), (200, [row]), (200, [])])
    assert len(c.corporate_actions("2024-01-01", "2024-07-18")) == 1
    assert "UNVERIFIED" in ap.AlpacaClient.corporate_actions.__doc__


# ── transport ─────────────────────────────────────────────────────────────────────────────────────

def test_default_fetch_returns_http_errors_as_a_status_instead_of_raising(monkeypatch):
    """Breaks if a 404 for an unknown ticker aborts the run instead of arriving as (404, body)."""
    def fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, io.BytesIO(b'{"message":"nope"}'))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert ap.default_fetch("GET", "https://example.invalid/x", {}) == (404, b'{"message":"nope"}')


def test_default_fetch_sends_method_body_user_agent_and_a_30_second_timeout_and_returns_status_and_bytes(monkeypatch):
    """Breaks if the OAuth POST loses its method or body, the user agent changes, or the timeout is dropped."""
    seen = {}

    class _Resp(io.BytesIO):
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        seen.update(method=req.get_method(), data=req.data, agent=req.get_header("User-agent"), timeout=timeout)
        return _Resp(b"[1]")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert ap.default_fetch("POST", "https://example.invalid/x", {"X-Test": "1"}, b"x") == (200, b"[1]")
    assert seen == {"method": "POST", "data": b"x", "agent": ap.USER_AGENT, "timeout": 30}


def test_the_module_imports_no_network_or_parquet_library_at_top_level():
    """Breaks the project rule that library modules import urllib.request / pyarrow / requests / yfinance lazily."""
    src = (ROOT / "data_engine" / "alpaca_provider.py").read_text().split("\n")
    top = [ln for ln in src if ln.startswith(("import ", "from "))]
    banned = ("pyarrow", "yfinance", "requests", "urllib.request")
    assert not [ln for ln in top if any(b in ln for b in banned)], top
