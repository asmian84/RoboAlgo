"""The Alpaca go/no-go must not call a vendor good because of a loose threshold, a flattering sample or a leaked key.

Failure groups this guards against. (1) The pre-registered rules drifting: the start/end tolerances, the 0.5
PASS share (>=), the delisting-year window, the AV-based sampling frame (OTC and unknown names must stay in the
primary sample). (2) The call shape: sip feed, split adjustment, the asof request first and a plain retry second,
a tail of 30 days past the delisting. (3) The report: strict JSON, the NAMES of the exported variables and never
a value, a NOT_RUN report and exit 2 when the credentials are missing or rejected. (4) Reproducibility of the sample.
No test touches the network: the client is a fake injected into main().
"""
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load():
    # The script loads the provider modules by path itself (data_engine/__init__ pulls in the DB layer).
    spec = importlib.util.spec_from_file_location("alpaca_check_under_test", ROOT / "scripts" / "alpaca_check.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["alpaca_check_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


chk = _load()

SYMS = [f"S{chr(65 + i // 26)}{chr(65 + i % 26)}" for i in range(40)]     # SAA..SAZ, SBA..SBN (sorted order == list order)
IPO, DELIST = "2010-05-03", "2019-06-14"


def write_av(d: Path) -> Path:
    """40 good delisted rows plus 5 that the universe filters must exclude."""
    base = {"exchange": "NYSE", "assetType": "Stock", "ipoDate": IPO, "delistingDate": DELIST, "status": "Delisted"}
    rows = [dict(base, symbol=s, name=f"Company {s} Inc") for s in SYMS]
    rows += [dict(base, symbol="SNAP", name="Snap Day Inc", delistingDate="2026-10-06"),     # snapshot-day row
             dict(base, symbol="OLDD", name="Old Delisting Inc", delistingDate="2016-12-30"),
             dict(base, symbol="WRNT", name="Zeta Warrants Inc"),
             dict(base, symbol="ETFX", name="Some Fund", assetType="ETF"),
             dict(base, symbol="SHRT", name="Short Listing Inc", ipoDate="2018-09-01")]          # 286 days listed
    d.mkdir(parents=True, exist_ok=True)
    p = d / "av.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def frame(sym, dates):
    return pd.DataFrame({"symbol": [sym] * len(dates), "date": pd.to_datetime(list(dates))})


class FakeClient:
    """Stands in for AlpacaClient: `covered` stocks answer the asof request, `plain_only` ones only the plain retry."""
    plan, trading_base = "free", "https://paper-api.alpaca.markets"

    def __init__(self, covered=(), plain_only=(), inactive=None, active=(), errors=None, auth_fail=False, aapl_first="2016-01-04"):
        self.covered, self.plain_only = set(covered), set(plain_only)
        self.inactive = inactive if inactive is not None else pd.DataFrame(columns=["symbol", "exchange"])
        self.active = pd.DataFrame({"symbol": list(active), "exchange": ["NYSE"] * len(active)})
        self.errors, self.auth_fail, self.calls, self.requests = dict(errors or {}), auth_fail, [], 0
        self.aapl_first = aapl_first

    def clock(self):
        if self.auth_fail:
            raise chk.ap.AlpacaAuthError("Alpaca answered HTTP 401 for /v2/clock: check the credentials and the data subscription")
        return {"is_open": False, "timestamp": "2026-10-13T15:00:00Z"}

    def assets(self, status="inactive", asset_class="us_equity", exchange=None):
        return self.inactive if status == "inactive" else self.active

    def bars(self, symbols, start, end, timeframe="1Day", adjustment="split", feed="iex", chunk=100, asof=None):
        sym = symbols[0]
        self.calls.append({"symbol": sym, "start": start, "end": end, "timeframe": timeframe, "adjustment": adjustment, "feed": feed, "asof": asof})
        if sym == "AAPL":
            return frame(sym, [self.aapl_first, "2026-10-12"])
        if asof is not None and sym in self.errors:
            raise self.errors[sym]
        hit = (sym in self.covered and asof is not None) or (sym in self.plain_only and asof is None)
        return frame(sym, ["2016-01-04", "2019-06-13"] if hit else [])


def run(d: Path, client, *extra):
    out = d / "report.json"
    code = chk.main(["--delisted-csv", str(write_av(d)), "--out", str(out), *extra], client=client, today="2026-10-13")
    return code, (json.loads(out.read_text()) if out.exists() else None), out


# ── the pre-registered rules ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("first, last, ipo, delist, expected", [
    ("2016-03-04", "2019-06-13", IPO, DELIST, True),        # start exactly 60 days after 2016-01-04
    ("2016-03-05", "2019-06-13", IPO, DELIST, False),       # 61 days
    ("2016-01-04", "2019-04-30", IPO, DELIST, True),        # ends exactly 45 days before the delisting
    ("2016-01-04", "2019-04-29", IPO, DELIST, False),       # 46
    ("2016-01-04", "2019-07-29", IPO, DELIST, True),        # ends exactly 45 days after
    ("2016-01-04", "2019-07-30", IPO, DELIST, False),       # 46
    ("2018-04-02", "2019-06-13", "2018-02-01", DELIST, True),     # IPO after 2016: the anchor is the IPO date, +60 days
    ("2018-04-03", "2019-06-13", "2018-02-01", DELIST, False),
    (None, "2019-06-13", IPO, DELIST, False),
])
def test_covers_life_at_every_tolerance_boundary(first, last, ipo, delist, expected):
    """Breaks if a tolerance (60 days start, 45 days end), the 2016 anchor for old IPOs, or the no-bars case changes."""
    assert chk.covers_life(first, last, ipo, delist) is expected


def test_the_calendar_arithmetic_behind_the_covers_life_boundaries():
    """Breaks if a hand-computed boundary date above is wrong (2016 is a leap year)."""
    assert (pd.Timestamp("2016-03-04") - pd.Timestamp("2016-01-04")).days == 60
    assert (pd.Timestamp("2019-06-14") - pd.Timestamp("2019-04-30")).days == 45
    assert (pd.Timestamp("2019-07-29") - pd.Timestamp("2019-06-14")).days == 45
    assert (pd.Timestamp("2018-04-02") - pd.Timestamp("2018-02-01")).days == 60
    assert (pd.Timestamp("2019-07-14") - pd.Timestamp("2019-06-14")).days == 30


def test_the_probe_universe_keeps_only_real_delisted_common_stocks_of_2017_to_2025(tmp_path):
    """Breaks if the snapshot-day, pre-2017, warrant, ETF or short-listing rows enter the sampling frame."""
    av = pd.read_csv(write_av(tmp_path), dtype=str, keep_default_na=False)
    assert sorted(chk.probe_universe(av)["symbol"]) == SYMS


@pytest.mark.parametrize("covers, sampled, expected", [(20, 40, "PASS"), (19, 40, "FAIL"), (0, 0, "FAIL"), (1, 2, "PASS"), (0, 1, "FAIL")])
def test_the_verdict_passes_at_exactly_half_and_never_on_an_empty_sample(covers, sampled, expected):
    """Breaks if the threshold becomes strict (>), moves off 0.5, or an empty sample passes."""
    assert chk.verdict(covers, sampled) == expected


# ── the run ───────────────────────────────────────────────────────────────────────────────────────

def test_a_run_where_half_are_covered_passes_and_reports_every_count(tmp_path):
    """Breaks if 20 of 40 stops being a PASS, or a summary count, share or the AAPL block is wrong."""
    code, rep, _ = run(tmp_path, FakeClient(covered=SYMS[:20]))
    assert code == 0
    assert rep["verdict"] == "PASS" and rep["verdict_text"] == chk.VERDICT_PASS
    p = rep["probe"]
    assert (p["sampled"], p["universe"], p["covers_life"], p["share_covers_life"]) == (40, 40, 20, 0.5)
    assert (p["any_bars"], p["share_any_bars"], p["via_asof"], p["via_plain"], p["errors"]) == (20, 0.5, 20, 0, 0)
    assert len(p["stocks"]) == 40
    assert rep["aapl"] == {"rows": 2, "first": "2016-01-04", "last": "2026-10-12", "first_ok": True}
    assert rep["caveat"] == chk.CAVEAT


def test_a_run_where_fewer_than_half_are_covered_fails_with_the_tiingo_advice(tmp_path):
    """Breaks if 19 of 40 passes, or the FAIL text stops pointing back to the Tiingo loader."""
    code, rep, _ = run(tmp_path, FakeClient(covered=SYMS[:19]))
    assert code == 0 and rep["verdict"] == "FAIL" and rep["verdict_text"] == chk.VERDICT_FAIL
    assert rep["probe"]["covers_life"] == 19 and rep["probe"]["share_covers_life"] == 0.475


def test_otc_and_unlisted_names_stay_in_the_primary_sample_and_are_only_counted_separately(tmp_path):
    """Breaks if OTC rows are mistaken for non-OTC, or the primary verdict is conditioned on Alpaca knowing the stock."""
    inactive = pd.DataFrame({"symbol": SYMS[:39], "exchange": ["OTC", "OTC"] + ["NYSE"] * 37})
    code, rep, _ = run(tmp_path, FakeClient(covered=SYMS[:20], inactive=inactive, active=["SAC"]))
    p = rep["probe"]
    assert p["inactive_non_otc"] == {"n": 37, "covers_life": 18, "share": pytest.approx(18 / 37)}
    assert (p["inactive_otc"], p["not_listed"], p["ticker_now_active"]) == (2, 1, 1)
    by = {r["symbol"]: r for r in p["stocks"]}
    assert by["SAA"]["alpaca_status"] == "inactive_otc"
    assert by["SBN"]["alpaca_status"] == "not_listed"
    assert by["SAC"]["alpaca_status"] == "inactive" and by["SAC"]["ticker_now_active"] is True
    assert rep["assets"] == {"inactive": 39, "inactive_otc": 2, "active": 1}
    assert rep["verdict"] == "PASS" and p["covers_life"] == 20


def test_the_bars_requests_use_sip_split_adjustment_the_asof_day_before_and_a_30_day_tail(tmp_path):
    """Breaks if the probe uses the IEX feed, drops asof, asks for the wrong window, or skips the plain retry."""
    client = FakeClient(covered=SYMS[:20])
    run(tmp_path, client)
    aapl = [c for c in client.calls if c["symbol"] == "AAPL"]
    assert aapl == [{"symbol": "AAPL", "start": "2016-01-04", "end": "2026-10-13", "timeframe": "1Day", "adjustment": "split",
                     "feed": "sip", "asof": None}]
    saa = [c for c in client.calls if c["symbol"] == "SAA"]
    assert saa == [{"symbol": "SAA", "start": "2016-01-01", "end": "2019-07-14", "timeframe": "1Day", "adjustment": "split",
                    "feed": "sip", "asof": "2019-06-13"}]
    sau = [c for c in client.calls if c["symbol"] == "SAU"]
    assert [c["asof"] for c in sau] == ["2019-06-13", None]


def test_a_stock_found_only_by_the_plain_retry_counts_and_is_labelled_plain(tmp_path):
    """Breaks if the plain fallback is not tried after an empty asof answer, or its hit is not counted."""
    code, rep, _ = run(tmp_path, FakeClient(covered=SYMS[:20], plain_only={"SAU"}))
    p = rep["probe"]
    assert p["covers_life"] == 21 and p["via_plain"] == 1
    assert {r["symbol"]: r for r in p["stocks"]}["SAU"]["via"] == "plain"


def test_a_non_auth_error_is_recorded_for_that_stock_and_the_run_continues(tmp_path):
    """Breaks if one rejected asof parameter aborts the whole probe or is silently swallowed."""
    err = chk.ap.AlpacaError("Alpaca answered HTTP 422 for /v2/stocks/bars: bad asof")
    code, rep, _ = run(tmp_path, FakeClient(covered=SYMS[:20], errors={"SAV": err}))
    sav = {r["symbol"]: r for r in rep["probe"]["stocks"]}["SAV"]
    assert code == 0 and "422" in sav["asof_error"] and sav["plain_error"] is None
    assert sav["via"] is None and sav["bars"] == 0
    assert rep["probe"]["errors"] == 1 and rep["probe"]["sampled"] == 40


@pytest.mark.parametrize("aapl_first, ok", [("2016-03-04", True), ("2016-03-05", False), ("2020-06-01", False)])
def test_the_aapl_history_start_flag_is_false_when_the_first_bar_is_late(tmp_path, aapl_first, ok):
    """Breaks if first_ok stays True for truncated history (it must compare with the start plus the 60-day tolerance)."""
    code, rep, _ = run(tmp_path, FakeClient(covered=SYMS[:20], aapl_first=aapl_first))
    assert rep["aapl"]["first_ok"] is ok


# ── the report and the credentials ────────────────────────────────────────────────────────────────

def test_the_report_is_strict_json_lists_only_variable_names_and_leaks_no_value(tmp_path, monkeypatch, capsys):
    """Breaks if NaN reaches the report, an exported key value is written or printed, or the names list changes."""
    for name in chk.ap.ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ALPACA_API_KEY", "TEST-KEY-ID")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "TEST-SECRET")
    code, rep, out = run(tmp_path, FakeClient(covered=SYMS[:20]))
    text = out.read_text()
    json.loads(text)
    assert "NaN" not in text and "Infinity" not in text
    cap = capsys.readouterr()
    for secret in ("TEST-KEY-ID", "TEST-SECRET"):
        assert secret not in text and secret not in cap.out and secret not in cap.err
    assert rep["env_vars_set"] == ["ALPACA_API_KEY", "ALPACA_SECRET_KEY"]
    assert "VERDICT PASS" in cap.out


def test_missing_credentials_exit_2_with_the_names_to_set_and_write_no_report(tmp_path, monkeypatch, capsys):
    """Breaks if a run without credentials crashes, exits 0, writes a report, or does not say which variables to set."""
    for name in chk.ap.ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    out = tmp_path / "r.json"
    assert chk.main(["--out", str(out)]) == 2
    assert not out.exists()
    assert "ALPACA_API_KEY" in capsys.readouterr().err


def test_rejected_credentials_exit_2_and_write_a_not_run_report(tmp_path):
    """Breaks if a 401 is read as a FAIL verdict about Alpaca's data, or exits 0."""
    code, rep, _ = run(tmp_path, FakeClient(auth_fail=True))
    assert code == 2 and rep["verdict"] == "NOT_RUN" and "401" in rep["error"]


def test_the_seeded_sample_is_reproducible_and_sorted(tmp_path):
    """Breaks if the same seed draws different stocks on a rerun, or the rows lose their symbol order."""
    _, a, _ = run(tmp_path / "a", FakeClient(), "--sample", "5", "--seed", "7")
    _, b, _ = run(tmp_path / "b", FakeClient(), "--sample", "5", "--seed", "7")
    sa, sb = [r["symbol"] for r in a["probe"]["stocks"]], [r["symbol"] for r in b["probe"]["stocks"]]
    assert sa == sb and len(sa) == 5 and sa == sorted(sa)
