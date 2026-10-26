"""
Tests for data_engine.tradingview_provider.

Offline tests use hardcoded fixtures and a stubbed `requests.post` (no network).
The live test is marked `network`; deselect it with:

    pytest tests/test_tradingview_provider -m "not network"
"""
from __future__ import annotations

import pytest
import requests

from data_engine import tradingview_provider as tv
from scanner_engine.finder import FinderCriteria


@pytest.fixture(autouse=True)
def _clear_cache():
    tv._CACHE.clear()
    yield
    tv._CACHE.clear()


class _Resp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


# Rows: 3 good, plus bad ones that must be counted as failed.
FIXTURE = {
    "totalCount": 9,
    "data": [
        {"s": "NASDAQ:WBD", "d": ["WBD", 12.5, 435637442]},
        {"s": "NYSE:F", "d": ["F", 11.0, 90000000]},
        {"s": "NASDAQ:SOFI", "d": ["SOFI", 8.25, 55000000.0]},
        {"s": "NYSE:ZERO", "d": ["ZERO", 0, 5000000]},             # close <= 0
        {"s": "NYSE:NEG", "d": ["NEG", -3.0, 5000000]},            # close <= 0
        {"s": "NYSE:STR", "d": ["STR", "n/a", 5000000]},           # non-numeric close
        {"s": "NYSE:NOVOL", "d": ["NOVOL", 5.0, None]},            # non-numeric volume
        {"s": "NYSE:SHORT", "d": ["SHORT", 5.0]},                  # truncated row
        {"s": "NYSE:BAD"},                                         # no "d"
    ],
}


def test_parse_fixture_offline(monkeypatch):
    calls = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append(json)
        return _Resp(FIXTURE)

    monkeypatch.setattr(tv.requests, "post", fake_post)
    criteria = FinderCriteria(min_price=1, max_price=20, min_volume=1_000_000)
    out = tv.screen(criteria)

    assert out is not None
    survivors, failed = out
    # Survivor shape matches _stage1_screen: {"symbol": str, "price": float, "volume": float}
    assert survivors == [
        {"symbol": "WBD", "price": 12.5, "volume": 435637442.0},
        {"symbol": "F", "price": 11.0, "volume": 90000000.0},
        {"symbol": "SOFI", "price": 8.25, "volume": 55000000.0},
    ]
    for s in survivors:
        assert set(s) == {"symbol", "price", "volume"}
        assert isinstance(s["symbol"], str)
        assert isinstance(s["price"], float)
        assert isinstance(s["volume"], float)
        assert ":" not in s["symbol"]  # taken from d[0], not the exchange-prefixed "s"
    assert failed == 6  # ZERO, NEG, STR, NOVOL, SHORT, BAD

    # Request contract: exactly name/close/volume, nothing precomputed.
    assert len(calls) == 1
    body = calls[0]
    assert body["columns"] == ["name", "close", "volume"]
    assert body["filter"] == [
        {"left": "type", "operation": "equal", "right": "stock"},
        {"left": "close", "operation": "in_range", "right": [1, 20]},
        {"left": "volume", "operation": "greater", "right": 1_000_000},
    ]
    assert body["range"] == [0, 1000]


def test_no_indicator_columns_requested():
    """Guard the CTVA hazard: SMA/RSI/change columns must never be requested."""
    for col in tv._COLUMNS:
        assert not any(tok in col.upper() for tok in ("SMA", "EMA", "RSI", "MACD", "CHANGE"))
    assert tv._COLUMNS == ["name", "close", "volume"]


def test_filter_single_sided_bounds():
    only_min = tv._build_filter(FinderCriteria(min_price=3, max_price=None, min_volume=10))
    only_max = tv._build_filter(FinderCriteria(min_price=None, max_price=9, min_volume=10))
    neither = tv._build_filter(FinderCriteria(min_price=None, max_price=None, min_volume=10))
    assert {"left": "close", "operation": "greater", "right": 3} in only_min
    assert {"left": "close", "operation": "less", "right": 9} in only_max
    assert [f["left"] for f in neither] == ["type", "volume"]


def test_pagination_and_sleep(monkeypatch):
    def page(n, start):
        return [{"s": f"X:T{start + i}", "d": [f"T{start + i}", 5.0, 2_000_000]} for i in range(n)]

    pages = [
        {"totalCount": 1500, "data": page(1000, 0)},
        {"totalCount": 1500, "data": page(500, 1000)},
    ]
    seen_ranges, sleeps = [], []

    def fake_post(url, json=None, headers=None, timeout=None):
        seen_ranges.append(json["range"])
        return _Resp(pages[len(seen_ranges) - 1])

    monkeypatch.setattr(tv.requests, "post", fake_post)
    monkeypatch.setattr(tv.time, "sleep", lambda s: sleeps.append(s))
    out = tv.screen(FinderCriteria(min_price=1, max_price=20, min_volume=1_000_000))

    assert out is not None
    survivors, failed = out
    assert len(survivors) == 1500 and failed == 0
    assert seen_ranges == [[0, 1000], [1000, 2000]]
    assert sleeps == [0.25]


def test_page_cap_returns_none(monkeypatch):
    rows = [{"s": f"X:T{i}", "d": [f"T{i}", 5.0, 2_000_000]} for i in range(1000)]
    n = []

    def fake_post(url, json=None, headers=None, timeout=None):
        n.append(1)
        return _Resp({"totalCount": 10**6, "data": rows})

    monkeypatch.setattr(tv.requests, "post", fake_post)
    monkeypatch.setattr(tv.time, "sleep", lambda s: None)
    assert tv.screen(FinderCriteria(min_price=1, max_price=20)) is None
    assert len(n) == 20


def test_cache_hit_skips_network(monkeypatch):
    n = []

    def fake_post(url, json=None, headers=None, timeout=None):
        n.append(1)
        return _Resp(FIXTURE)

    monkeypatch.setattr(tv.requests, "post", fake_post)
    c = FinderCriteria(min_price=1, max_price=20, min_volume=1_000_000)
    first = tv.screen(c)
    second = tv.screen(c)
    assert first == second and len(n) == 1
    # Mutating a returned result must not poison the cache.
    first[0][0]["price"] = -1.0
    assert tv.screen(c)[0][0]["price"] == 12.5


@pytest.mark.parametrize(
    "behaviour",
    [
        lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError("down")),
        lambda *a, **k: (_ for _ in ()).throw(requests.Timeout("slow")),
        lambda *a, **k: _Resp({}, status=429),
        lambda *a, **k: _Resp({}, status=500),
        lambda *a, **k: _Resp(ValueError("not json")),
        lambda *a, **k: _Resp([1, 2, 3]),
        lambda *a, **k: _Resp({"nope": 1}),
        lambda *a, **k: _Resp({"totalCount": 1, "data": "oops"}),
        lambda *a, **k: _Resp({"totalCount": "many", "data": []}),
    ],
    ids=["conn", "timeout", "429", "500", "bad-json", "list", "no-data", "data-not-list", "total-not-int"],
)
def test_failures_return_none_never_raise(monkeypatch, behaviour):
    monkeypatch.setattr(tv.requests, "post", behaviour)
    assert tv.screen(FinderCriteria(min_price=1, max_price=20)) is None


def test_failure_is_not_cached(monkeypatch):
    monkeypatch.setattr(tv.requests, "post", lambda *a, **k: _Resp({}, status=500))
    c = FinderCriteria(min_price=1, max_price=20)
    assert tv.screen(c) is None
    monkeypatch.setattr(tv.requests, "post", lambda *a, **k: _Resp(FIXTURE))
    assert tv.screen(c) is not None


@pytest.mark.network
def test_live_screen_1_to_20_dollars_volume_1m():
    """Hits the real endpoint. Deselect with: -m "not network"."""
    c = FinderCriteria(min_price=1.0, max_price=20.0, min_volume=1_000_000)
    out = tv.screen(c)
    if out is None:
        pytest.fail("tradingview screen returned None (endpoint unreachable or changed)")
    survivors, failed = out
    assert 300 < len(survivors) < 5000
    for s in survivors:
        assert 1.0 <= s["price"] <= 20.0, s
        assert s["price"] > 0, s
        assert s["volume"] >= 1_000_000, s
        assert isinstance(s["symbol"], str) and s["symbol"] and ":" not in s["symbol"]
