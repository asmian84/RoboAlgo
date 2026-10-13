"""The Tiingo loader must hand the ingest only prices of the SAME company, split-adjusted, and must survive
rate limits and restarts without losing work or leaking the token.

Failures this guards against: (1) a reused ticker (BBB) written out as if it were the dead company; (2) the
bankruptcy "Q" ticker (CCCQ) never tried, so a dead stock silently drops out of the survivorship-corrected
data; (3) a 429 treated as "no data", which would mark good stocks as permanently missing; (4) a rerun that
asks the vendor again for what it already holds (the free plan has 480 symbols a month); (5) the token written
into a state file, a cache file, a parquet file or the console.
No test touches the network: the fake Tiingo below answers every request and checks the Authorization header.
"""
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
TOKEN = "TEST-TOKEN-XYZ"
BASE = "https://api.tiingo.com/tiingo/daily"
NOW = datetime(2026, 10, 12, 12, 0, tzinfo=timezone.utc)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _iso(d):
    return pd.Timestamp(d).strftime("%Y-%m-%dT00:00:00.000Z")


def _row(date, close, volume, split=1.0):
    return {"date": _iso(date), "open": close, "high": close, "low": close, "close": close, "volume": volume,
            "adjOpen": 1.0, "adjHigh": 1.0, "adjLow": 1.0, "adjClose": 1.0, "adjVolume": 1, "divCash": 0.0, "splitFactor": split}


def _aaa_prices():
    """300 business days ending 2015-02-27: 2-for-1 on day 100, 1-for-4 reverse on day 200; adjusted close is 200 throughout."""
    rows = []
    for i, d in enumerate(pd.bdate_range(end="2015-02-27", periods=300)):
        close, vol = (100.0, 1000) if i < 100 else (50.0, 2000) if i < 200 else (200.0, 500)
        rows.append(_row(d, close, vol, split=2.0 if i == 100 else 0.25 if i == 200 else 1.0))
    return rows


def _flat_prices(end, close=20.0, n=300):
    return [_row(d, close, 100_000) for d in pd.bdate_range(end=end, periods=n)]


AV_ROWS = [
    {"symbol": "AAA", "name": "Apex Mining Corp", "exchange": "NYSE", "assetType": "Stock", "ipoDate": "2005-03-01", "delistingDate": "2015-03-02", "status": "Delisted"},
    {"symbol": "BBB", "name": "Apex Mining Corp", "exchange": "NYSE", "assetType": "Stock", "ipoDate": "2006-01-03", "delistingDate": "2016-06-01", "status": "Delisted"},
    {"symbol": "CCC", "name": "Cobalt Systems Corp", "exchange": "NASDAQ", "assetType": "Stock", "ipoDate": "2007-02-01", "delistingDate": "2017-09-01", "status": "Delisted"},
    {"symbol": "ZZZ", "name": "Zeta Zone Inc", "exchange": "NYSE", "assetType": "Stock", "ipoDate": "2001-01-02", "delistingDate": "2018-01-02", "status": "Delisted"},
    {"symbol": "ETFX", "name": "Some Index Fund", "exchange": "NYSE", "assetType": "ETF", "ipoDate": "2008-01-02", "delistingDate": "2014-01-02", "status": "Delisted"},
]


def _routes():
    ccc_end = (pd.Timestamp("2017-09-01") + pd.Timedelta(days=200)).date().isoformat()          # the Q ticker keeps trading OTC
    bbb_end = (pd.Timestamp("2016-06-01") + pd.DateOffset(years=3)).date().isoformat()          # a later company under the same ticker
    return {
        "/AAA": (200, {"ticker": "AAA", "name": "Apex Mining Corp", "startDate": "2005-03-01", "endDate": "2015-02-27"}),
        "/AAA/prices": (200, _aaa_prices()),
        "/BBB": (200, {"ticker": "BBB", "name": "Apex Biotech Inc", "startDate": "2006-01-03", "endDate": bbb_end}),
        "/BBBQ": (404, {"detail": "Not found."}),
        "/CCC": (404, {"detail": "Not found."}),
        "/CCCQ": (200, {"ticker": "CCCQ", "name": "Cobalt Systems Corp", "startDate": "2007-02-01", "endDate": ccc_end}),
        "/CCCQ/prices": (200, _flat_prices(ccc_end)),
    }


class FakeTiingo:
    """dict url-suffix -> (status, json) or a list of them (consumed in order, the last one repeats)."""

    def __init__(self, routes, token=TOKEN):
        self.routes, self.token, self.calls = routes, token, []

    def __call__(self, url, headers):
        assert headers.get("Authorization") == f"Token {self.token}"
        assert self.token not in url
        path = url.split("?")[0][len(BASE):]
        self.calls.append(path)
        resp = self.routes.get(path, (404, {"detail": "Not found."}))
        if isinstance(resp, list):
            resp = resp.pop(0) if len(resp) > 1 else resp[0]
        return resp[0], json.dumps(resp[1]).encode()


class FakeTime:
    def __init__(self, t=100_000.0):
        self.t, self.sleeps = float(t), []

    def clock(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += s


@pytest.fixture
def ld():
    return _load("tiingo_load_delisted")


def _selected(ld, rows=None, only=None):
    av = pd.DataFrame(rows or AV_ROWS)
    sel = ld.tp.select_universe(av, 2010, 2025, exclude=frozenset({"ZZZ"}))
    return sel if only is None else sel[sel["symbol"].isin(only)].reset_index(drop=True)


def _run(ld, tmp, fetch, selected=None, max_requests=10_000, ft=None, plan="free"):
    ft = ft or FakeTime()
    res = ld.run(_selected(ld) if selected is None else selected, token=TOKEN, plan=plan, max_requests=max_requests,
                 root=tmp / "tiingo", fetch=fetch, clock=ft.clock, sleep=ft.sleep, now=lambda: NOW)
    return res, ft


def _state(tmp):
    return json.loads((tmp / "tiingo" / "state.json").read_text())


def _survivors(tmp):
    pytest.importorskip("pyarrow")
    surv = tmp / "surv"
    surv.mkdir()
    rng = np.random.default_rng(3)
    close = 20 * np.exp(np.cumsum(rng.normal(0.0003, 0.02, 300)))
    pd.DataFrame({"symbol": "ZZZ", "date": pd.bdate_range("2014-01-02", periods=300), "open": close, "high": close * 1.01,
                  "low": close * 0.99, "close": close, "volume": 3e6}).to_parquet(surv / "batch_0000.parquet", index=False)
    return surv


def _csv(tmp):
    p = tmp / "av.csv"
    pd.DataFrame(AV_ROWS).to_csv(p, index=False)
    return p


# ── the round trip ────────────────────────────────────────────────────────────────────────────────

def test_round_trip_keeps_the_dead_company_via_its_q_ticker_and_drops_the_reused_ticker(ld, tmp_path):
    """Breaks if CCCQ is not tried, BBB is accepted, or the split-adjusted prices do not survive the ingest."""
    pytest.importorskip("pyarrow")
    fake = FakeTiingo(_routes())
    res, _ = _run(ld, tmp_path, fake)
    assert res["stopped"] is None and res["requests"] == 7                  # AAA 2, BBB 1 + BBBQ 1, CCC 1 + CCCQ 2
    assert res["counts"] == {"ok": 2, "reused": 1, "no_data": 0, "mismatch": 0} and res["via_q"] == 1
    assert "/BBB/prices" not in fake.calls and "/BBBQ/prices" not in fake.calls

    state = _state(tmp_path)
    assert state["CCC"]["ticker_used"] == "CCCQ" and state["AAA"]["ticker_used"] == "AAA"
    assert state["BBB"]["status"] == "reused" and state["BBB"]["ticker_used"] is None
    assert state["AAA"]["rows"] == 300 and state["AAA"]["last"] == "2015-02-27"

    paths = ld.write_out(state, tmp_path / "tiingo")
    assert [p.name for p in paths] == ["delisted_0000.parquet"]

    ing = _load("ingest_delisted_history")
    surv, into = _survivors(tmp_path), tmp_path / "into"
    assert ing.main(["--source", str(tmp_path / "tiingo" / "out"), "--survivors", str(surv), "--into", str(into)]) == 0
    out = pd.read_parquet(into / "batch_delisted_0000.parquet")
    assert set(out["symbol"]) == {"AAA", "CCC"}                              # OUR symbol, not CCCQ; BBB and ZZZ absent
    assert json.loads((into / "terminal.json").read_text()) == ["AAA", "CCC"]
    aaa = out[out["symbol"] == "AAA"]
    assert len(aaa) == 300 and np.allclose(aaa["close"], 200.0) and np.allclose(aaa["open"], 200.0)
    assert np.allclose(aaa["volume"], 500.0)
    assert len(out[out["symbol"] == "CCC"]) == 300


def test_state_entries_record_the_evidence_for_each_decision(ld, tmp_path):
    """Breaks if the reason for a rejection, the start-before-ipo gap or the wide-tolerance flag is not kept."""
    _run(ld, tmp_path, FakeTiingo(_routes()))
    state = _state(tmp_path)
    assert state["AAA"]["start_before_ipo_days"] == 0 and state["AAA"]["wide_tol"] is False
    assert state["AAA"]["reason"] == "" and state["AAA"]["first"] == pd.bdate_range(end="2015-02-27", periods=300)[0].date().isoformat()
    assert "BBB: Tiingo 2006-01-03..2019-06-01 'Apex Biotech Inc'" in state["BBB"]["reason"] and "BBBQ: not in Tiingo" in state["BBB"]["reason"]


def test_a_stock_listed_before_2000_uses_the_wide_start_tolerance_and_it_is_counted(ld, tmp_path):
    """Breaks if the old-IPO assumption is invisible in the run summary and the state."""
    row = {"symbol": "OLDX", "name": "Old Rail Corp", "exchange": "NYSE", "assetType": "Stock", "ipoDate": "1995-01-03", "delistingDate": "2012-05-01", "status": "Delisted"}
    routes = {"/OLDX": (200, {"ticker": "OLDX", "name": "Old Rail Corp", "startDate": "1995-11-01", "endDate": "2012-04-30"}),
              "/OLDX/prices": (200, _flat_prices("2012-04-30"))}
    res, _ = _run(ld, tmp_path, FakeTiingo(routes), selected=_selected(ld, rows=[row]))
    assert res["counts"]["ok"] == 1 and res["wide_tol"] == 1
    assert _state(tmp_path)["OLDX"]["start_before_ipo_days"] == -302 and _state(tmp_path)["OLDX"]["wide_tol"] is True


def test_a_prices_request_that_fails_makes_the_ticker_no_data_and_the_q_ticker_is_tried(ld, tmp_path):
    """Breaks if a non-200 prices answer is treated as success, or stops the Q fallback."""
    routes = _routes()
    routes["/AAA/prices"] = (404, {"detail": "Not found."})
    fake = FakeTiingo(routes)
    res, _ = _run(ld, tmp_path, fake, selected=_selected(ld, only=["AAA"]))
    assert res["counts"]["no_data"] == 1 and "/AAAQ" in fake.calls
    assert "AAA: prices HTTP 404" in _state(tmp_path)["AAA"]["reason"]


# ── resume ────────────────────────────────────────────────────────────────────────────────────────

def test_a_second_run_makes_no_requests_and_even_a_lost_state_file_costs_none(ld, tmp_path):
    """Breaks if resume asks the vendor again for symbols already decided or tickers already cached."""
    fake = FakeTiingo(_routes())
    _run(ld, tmp_path, fake)
    first_state, n_calls = _state(tmp_path), len(fake.calls)

    res, _ = _run(ld, tmp_path, fake)
    assert res["requests"] == 0 and res["processed"] == 0 and res["already_final"] == 3 and len(fake.calls) == n_calls

    (tmp_path / "tiingo" / "state.json").unlink()
    res, _ = _run(ld, tmp_path, fake)                                        # the raw cache alone is enough
    assert res["requests"] == 0 and res["processed"] == 3 and len(fake.calls) == n_calls
    assert _state(tmp_path) == first_state


def test_max_requests_stops_mid_symbol_and_the_next_run_resumes_from_the_raw_cache(ld, tmp_path):
    """Breaks if the cap is ignored, a half-finished symbol is recorded as final, or its cached metadata is requested again."""
    fake = FakeTiingo(_routes())
    res, _ = _run(ld, tmp_path, fake, max_requests=3)                         # AAA 2 + BBB meta 1; BBBQ would be the 4th
    assert res["stopped"] == "max_requests" and res["requests"] == 3
    assert list(_state(tmp_path)) == ["AAA"]
    assert (tmp_path / "tiingo" / "raw" / "BBB.json").exists()

    res, _ = _run(ld, tmp_path, fake)
    assert res["stopped"] is None and res["requests"] == 4                    # BBBQ 1 + CCC 1 + CCCQ 2; BBB metadata came from the cache
    assert fake.calls.count("/BBB") == 1
    assert set(_state(tmp_path)) == {"AAA", "BBB", "CCC"}


def test_state_is_saved_after_every_symbol_so_an_interrupt_loses_at_most_the_symbol_in_progress(ld, tmp_path):
    """Breaks if progress is only saved at the end: a Ctrl-C or crash after hours of requests would lose every decision."""
    inner, n = FakeTiingo(_routes()), {"i": 0}

    def interrupted(url, headers):
        n["i"] += 1
        if n["i"] == 3:                                                       # BBB's metadata, right after AAA completed
            raise KeyboardInterrupt
        return inner(url, headers)

    with pytest.raises(KeyboardInterrupt):
        _run(ld, tmp_path, interrupted)
    assert list(_state(tmp_path)) == ["AAA"]
    res, _ = _run(ld, tmp_path, inner)
    assert res["requests"] == 5 and set(_state(tmp_path)) == {"AAA", "BBB", "CCC"}


# ── rate limits and errors ────────────────────────────────────────────────────────────────────────

def test_a_429_sleeps_to_the_next_hour_and_is_retried_instead_of_becoming_no_data(ld, tmp_path):
    """Breaks if a 429 is treated as an answer (the stock would be marked missing for good)."""
    routes = _routes()
    routes["/AAA"] = [(429, {"detail": "rate"}), routes["/AAA"]]
    fake = FakeTiingo(routes)
    ft = FakeTime(7300.0)                                                     # 3500 s to the next hour boundary
    res, ft = _run(ld, tmp_path, fake, selected=_selected(ld, only=["AAA"]), ft=ft)
    assert _state(tmp_path)["AAA"]["status"] == "ok" and res["requests"] == 3
    assert any(s >= 3500.0 for s in ft.sleeps), ft.sleeps
    assert fake.calls == ["/AAA", "/AAA", "/AAA/prices"]


def test_a_429_that_never_clears_stops_the_run_after_the_retries_with_nothing_recorded(ld, tmp_path):
    """Breaks if the loader hammers a closed door forever or records the symbol as decided."""
    routes = _routes()
    routes["/AAA"] = (429, {"detail": "rate"})
    res, ft = _run(ld, tmp_path, FakeTiingo(routes), selected=_selected(ld, only=["AAA"]))
    assert res["stopped"] == "rate_limited" and res["requests"] == ld.MAX_RETRY + 1
    assert len(ft.sleeps) == ld.MAX_RETRY
    assert not (tmp_path / "tiingo" / "state.json").exists()


def test_a_server_error_is_retried_with_growing_sleeps_then_stops_without_recording(ld, tmp_path):
    """Breaks if a transient 5xx becomes a permanent 'no data' or is retried without waiting."""
    routes = _routes()
    routes["/AAA"] = [(500, {}), routes["/AAA"]]
    res, ft = _run(ld, tmp_path, FakeTiingo(routes), selected=_selected(ld, only=["AAA"]))
    assert _state(tmp_path)["AAA"]["status"] == "ok" and ft.sleeps == [1]

    routes["/AAA"] = (503, {})
    res, ft = _run(ld, tmp_path / "again", FakeTiingo(routes), selected=_selected(ld, only=["AAA"]))
    assert res["stopped"] == "server_error" and ft.sleeps == [1, 2, 4]
    assert not (tmp_path / "again" / "tiingo" / "state.json").exists()


def test_a_network_error_is_retried_like_a_server_error(ld, tmp_path):
    """Breaks if one dropped connection kills the run or marks the symbol missing."""
    inner = FakeTiingo(_routes())
    calls = {"n": 0}

    def flaky(url, headers):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionResetError("reset")
        return inner(url, headers)

    res, ft = _run(ld, tmp_path, flaky, selected=_selected(ld, only=["AAA"]))
    assert _state(tmp_path)["AAA"]["status"] == "ok" and ft.sleeps == [1]


@pytest.mark.parametrize("status", [401, 403])
def test_a_rejected_token_stops_at_once_and_records_nothing(ld, tmp_path, status):
    """Breaks if a wrong token turns every symbol into a permanent 'no data' in the state file."""
    routes = _routes()
    routes["/AAA"] = (status, {"detail": "Invalid token."})
    res, _ = _run(ld, tmp_path, FakeTiingo(routes))
    assert res["stopped"] == "auth" and res["requests"] == 1 and res["processed"] == 0
    assert not (tmp_path / "tiingo" / "state.json").exists()


def test_the_monthly_unique_symbol_cap_stops_the_run_and_usage_lists_the_tickers(ld, tmp_path, monkeypatch):
    """Breaks if the loader burns past the plan's monthly symbol allowance (Tiingo would block the whole account)."""
    monkeypatch.setitem(ld.tp.PLANS, "free", dict(ld.tp.PLANS["free"], per_month_symbols=2))
    res, _ = _run(ld, tmp_path, FakeTiingo(_routes()))
    assert res["stopped"] == "monthly_symbol_cap"
    usage = json.loads((tmp_path / "tiingo" / "usage.json").read_text())
    assert usage["month"] == "2026-10" and usage["tickers"] == ["AAA", "BBB"]
    assert list(_state(tmp_path)) == ["AAA"]


def test_usage_counts_this_months_tickers_and_todays_requests(ld, tmp_path):
    """Breaks if the usage file under-counts, which would let a restart overspend the daily cap."""
    _run(ld, tmp_path, FakeTiingo(_routes()))
    usage = json.loads((tmp_path / "tiingo" / "usage.json").read_text())
    assert usage == {"month": "2026-10", "tickers": ["AAA", "BBB", "BBBQ", "CCC", "CCCQ"], "day": "2026-10-12", "day_count": 7}


def test_the_daily_cap_is_enforced_across_restarts(ld, tmp_path, monkeypatch):
    """Breaks if a restart forgets today's requests and the account runs over its daily cap."""
    monkeypatch.setitem(ld.tp.PLANS, "free", dict(ld.tp.PLANS["free"], per_day=3))
    res, _ = _run(ld, tmp_path, FakeTiingo(_routes()))
    assert res["stopped"] == "daily_cap" and res["requests"] == 3 and list(_state(tmp_path)) == ["AAA"]
    res, _ = _run(ld, tmp_path, FakeTiingo(_routes()))                        # same day, a new process
    assert res["stopped"] == "daily_cap" and res["requests"] == 0


# ── token, dry run, command line ──────────────────────────────────────────────────────────────────

def test_the_token_is_never_written_to_a_file_or_printed(ld, tmp_path, monkeypatch, capsys):
    """Breaks if the token reaches state.json, usage.json, the raw cache, the parquet output or the console."""
    pytest.importorskip("pyarrow")
    surv, csv = _survivors(tmp_path), _csv(tmp_path)
    monkeypatch.setenv("TIINGO_TOKEN", TOKEN)
    monkeypatch.setattr(ld.tp, "default_fetch", FakeTiingo(_routes()))
    assert ld.main(["--av-csv", str(csv), "--survivors", str(surv), "--root", str(tmp_path / "tiingo")]) == 0
    shown = capsys.readouterr()
    files = [p for p in (tmp_path / "tiingo").rglob("*") if p.is_file()]
    names = {p.name for p in files}
    assert {"state.json", "usage.json", "AAA.json", "CCCQ.json", "delisted_0000.parquet"} <= names
    for p in files:
        assert TOKEN.encode() not in p.read_bytes(), p
    assert TOKEN not in shown.out + shown.err
    assert shown.out.strip().splitlines()[-1] == "next: python scripts/ingest_delisted_history.py --source ~/.roboalgo/tiingo/out"


def test_the_dry_run_needs_no_token_makes_no_request_and_writes_nothing(ld, tmp_path, monkeypatch, capsys):
    """Breaks if a dry run calls the network, demands a token, or creates files under the root."""
    surv, csv = _survivors(tmp_path), _csv(tmp_path)
    monkeypatch.delenv("TIINGO_TOKEN", raising=False)

    def boom(*a, **k):
        raise AssertionError("the dry run must not fetch")

    monkeypatch.setattr(ld.tp, "default_fetch", boom)
    assert ld.main(["--dry-run", "--av-csv", str(csv), "--root", str(tmp_path / "tiingo"), "--survivors", str(surv)]) == 0
    out = capsys.readouterr().out
    assert "selected: 3 delisted common stocks" in out and "pending: 3" in out
    assert "2015: 1, 2016: 1, 2017: 1" in out and "estimated requests: ~7" in out and "limited by the" in out
    assert not (tmp_path / "tiingo").exists()


def test_without_a_token_a_real_run_stops_with_a_clear_message(ld, tmp_path, monkeypatch):
    """Breaks if a missing TIINGO_TOKEN produces a stack trace or, worse, an unauthenticated run."""
    surv, csv = _survivors(tmp_path), _csv(tmp_path)
    monkeypatch.delenv("TIINGO_TOKEN", raising=False)
    with pytest.raises(SystemExit, match="set TIINGO_TOKEN"):
        ld.main(["--av-csv", str(csv), "--root", str(tmp_path / "tiingo"), "--survivors", str(surv)])
    assert not (tmp_path / "tiingo").exists()


def test_limit_symbols_counts_new_symbols_so_a_rerun_makes_progress(ld, tmp_path, monkeypatch, capsys):
    """Breaks if --limit-symbols 1 re-selects the already decided first symbol forever."""
    surv, csv = _survivors(tmp_path), _csv(tmp_path)
    monkeypatch.setenv("TIINGO_TOKEN", TOKEN)
    fake = FakeTiingo(_routes())
    monkeypatch.setattr(ld.tp, "default_fetch", fake)
    args = ["--av-csv", str(csv), "--survivors", str(surv), "--root", str(tmp_path / "tiingo"), "--limit-symbols", "1"]
    assert ld.main(args) == 0
    assert list(_state(tmp_path)) == ["AAA"]
    assert ld.main(args) == 0
    assert list(_state(tmp_path)) == ["AAA", "BBB"]
    assert "STOPPED" not in capsys.readouterr().out


def test_an_early_stop_is_announced_and_the_output_is_still_written(ld, tmp_path, monkeypatch, capsys):
    """Breaks if a rate-limit stop looks like a clean finish, or loses the symbols decided before it."""
    surv, csv = _survivors(tmp_path), _csv(tmp_path)
    monkeypatch.setenv("TIINGO_TOKEN", TOKEN)
    monkeypatch.setattr(ld.tp, "default_fetch", FakeTiingo(_routes()))
    assert ld.main(["--av-csv", str(csv), "--survivors", str(surv), "--root", str(tmp_path / "tiingo"), "--max-requests", "2"]) == 0
    out = capsys.readouterr().out
    assert "STOPPED EARLY (max_requests)" in out and "wrote 1 file(s)" in out
    assert pd.read_parquet(tmp_path / "tiingo" / "out" / "delisted_0000.parquet")["symbol"].unique().tolist() == ["AAA"]


# ── write_out ─────────────────────────────────────────────────────────────────────────────────────

def test_write_out_replaces_old_files_and_splits_by_symbol_count(ld, tmp_path, monkeypatch):
    """Breaks if a stale batch survives a rebuild (a removed symbol would linger) or the batch size is ignored."""
    pytest.importorskip("pyarrow")
    _run(ld, tmp_path, FakeTiingo(_routes()))
    out = tmp_path / "tiingo" / "out"
    out.mkdir(parents=True, exist_ok=True)
    (out / "delisted_0009.parquet").write_bytes(b"stale")
    (out / "keep.txt").write_text("not ours")
    monkeypatch.setattr(ld, "OUT_BATCH", 1)
    paths = ld.write_out(_state(tmp_path), tmp_path / "tiingo")
    assert [p.name for p in paths] == ["delisted_0000.parquet", "delisted_0001.parquet"]
    assert not (out / "delisted_0009.parquet").exists() and (out / "keep.txt").exists()
    assert [pd.read_parquet(p)["symbol"].unique().tolist() for p in paths] == [["AAA"], ["CCC"]]


def test_write_out_dedupes_a_repeated_date_and_sorts(ld, tmp_path):
    """Breaks if duplicate vendor rows reach the ingest and double-count a bar."""
    pytest.importorskip("pyarrow")
    root = tmp_path / "tiingo"
    (root / "raw").mkdir(parents=True)
    rows = [_row("2020-01-03", 10.0, 100), _row("2020-01-02", 10.0, 100), _row("2020-01-03", 11.0, 100)]
    (root / "raw" / "DUP.json").write_text(json.dumps({"meta": {}, "prices": rows}))
    paths = ld.write_out({"DUP": {"status": "ok", "ticker_used": "DUP"}, "NOPE": {"status": "reused", "ticker_used": None}}, root)
    df = pd.read_parquet(paths[0])
    assert df["date"].tolist() == [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03")] and df["close"].tolist() == [10.0, 11.0]


def test_write_out_refuses_an_ok_symbol_whose_raw_cache_is_gone(ld, tmp_path):
    """Breaks if a deleted cache file silently drops a symbol from the output."""
    root = tmp_path / "tiingo"
    (root / "raw").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="GONE"):
        ld.write_out({"GONE": {"status": "ok", "ticker_used": "GONE"}}, root)
