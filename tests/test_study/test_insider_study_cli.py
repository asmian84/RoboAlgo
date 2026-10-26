"""scripts/insider_study.py: the CLI must report its inputs, fail cleanly, and never flatter a result.

The failures being guarded against: a run that loads prices before it knows an input is missing, a traceback where an
exit code belongs, a result file that is not strict JSON, a survivors-only result that is not stamped PROVISIONAL, a
stamp that stays on after delisted prices are loaded, a docstring that drifts from docs/INSIDER_STUDY_PREREG.md, a
stray network call, and the download helper being wired into the study. The end-to-end runs use a synthetic world
whose every expected number is worked out by hand in the comments, never taken from the code under test.

Synthetic world (months 2001-01 to 2026-09, one bar per month on the last weekday, volume 1e6):
  AAA, BBB, CCC close 50.0 through 2013-12 and 5.0 from 2014-01; AAA closes 6.0 from 2015-07.
  A stock is FALLEN from 2014-01 through 2018-11 (dd60 = 5/50 - 1 = -0.9 until the 50.0 leaves the 60-month window).
  Insider zips 2015q1 and 2015q2 hold only code-A awards (never purchases) for every issuer, plus two AAA purchases
  of exactly $5,000 each by two different people (filed 2015-05-04 and 2015-05-15): one cluster event, 2015-05-15.
  Coverage needs the quarters of [month_end - 89 days, month_end], so only 2015-03 to 2015-06 are covered.
  TREATED = AAA at 2015-05 (window opens 2015-03-03, last weekday 2015-05-29) and 2015-06; both have
  r12 = 6.0 / 5.0 - 1 = 0.2 against controls whose r12 is 0.0 (constant 5.0).
  CONTROLS = 2015-03 and 2015-04 (AAA, BBB, CCC) plus 2015-05 and 2015-06 (BBB, CCC) = 10.
"""
import ast
import importlib.util
import json
import re
import socket
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load_script():
    name = "insider_study_script"                  # never "insider_study": that is setup_engine.insider_study
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "insider_study.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


S = _load_script()
IS, SI, M = S.IS, S.SI, S.M
PROV = "PROVISIONAL: survivors only (delisted names missing)"


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network access attempted in an offline test")

    monkeypatch.setattr(socket.socket, "connect", boom)


def _strict(text):
    def bad(x):
        raise AssertionError(f"non-strict JSON constant {x}")

    return json.loads(text, parse_constant=bad)


# ── synthetic world ───────────────────────────────────────────────────────────────────────────────

def _last_weekday(p):
    me = p.end_time.normalize()
    return me - pd.Timedelta(days=int({5: 1, 6: 2}.get(me.weekday(), 0)))


def _bars(symbol, close_of, first="2001-01", last="2026-09"):
    rows = []
    for p in pd.period_range(first, last, freq="M"):
        c = float(close_of(p))
        rows.append((symbol, _last_weekday(p), c, c, c, c, 1e6))
    return pd.DataFrame(rows, columns=["symbol", "date", "open", "high", "low", "close", "volume"])


def _fall(p, *, lift=None):
    if p <= pd.Period("2013-12", "M"):
        return 50.0
    if lift is not None and p >= pd.Period(lift, "M"):
        return 6.0
    return 5.0


def _zip(path, tables):
    """tables: {member name: (header list, row lists)}, written tab separated."""
    with zipfile.ZipFile(path, "w") as zf:
        for member, (header, rows) in tables.items():
            zf.writestr(member, "\n".join("\t".join(r) for r in [header, *rows]) + "\n")


def _insider_zip(path, filings):
    """filings: (accession, filed, doc, issuer cik, symbol, owner cik, relationship, [(code, shares, price)])."""
    sub, own, trn = [], [], []
    for acc, filed, doc, cik, sym, owner, rel, trades in filings:
        sub.append([acc, filed, doc, cik, sym, "x"])
        own.append([acc, owner, rel, "x"])
        for code, shares, price in trades:
            trn.append([acc, filed, code, str(shares), str(price), "x"])
    _zip(path, {
        "SUBMISSION.tsv": (["ACCESSION_NUMBER", "FILING_DATE", "DOCUMENT_TYPE", "ISSUERCIK", "ISSUERTRADINGSYMBOL", "EXTRA"], sub),
        "REPORTINGOWNER.tsv": (["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNER_RELATIONSHIP", "EXTRA"], own),
        "NONDERIV_TRANS.tsv": (["ACCESSION_NUMBER", "TRANS_DATE", "TRANS_CODE", "TRANS_SHARES", "TRANS_PRICEPERSHARE", "EXTRA"], trn)})


AWARD = [("A", 100, 0.0)]
ISSUERS = (("1001", "AAA"), ("1002", "BBB"), ("1003", "CCC"), ("1007", "DED"))


def _make_world(tmp_path, tiingo=False):
    hist, ins = tmp_path / "history", tmp_path / "insider"
    hist.mkdir()
    ins.mkdir()
    bars = pd.concat([_bars("AAA", lambda p: _fall(p, lift="2015-07")), _bars("BBB", _fall), _bars("CCC", _fall)],
                     ignore_index=True)
    bars.to_parquet(hist / "batch_0000.parquet", index=False)
    q1 = [(f"0000000001-15-{i:06d}", "15-JAN-2015", "4", cik, sym, str(9100 + i), "Director", AWARD)
          for i, (cik, sym) in enumerate(ISSUERS, 1)]
    q2 = [(f"0000000002-15-{i:06d}", "15-APR-2015", "4", cik, sym, str(9200 + i), "Director", AWARD)
          for i, (cik, sym) in enumerate(ISSUERS[1:], 1)]
    q2 += [("0000000002-15-000101", "04-MAY-2015", "4", "1001", "AAA", "9001", "Director", [("P", 1000, 5.00)]),
           ("0000000002-15-000102", "15-MAY-2015", "4", "1001", "AAA", "9002", "Officer", [("P", 1000, 5.00)])]
    _insider_zip(ins / "2015q1_form345.zip", q1)
    _insider_zip(ins / "2015q2_form345.zip", q2)
    t = tmp_path / "tiingo"
    if tiingo:
        t.mkdir()
        ded = _bars("DED", lambda p: _fall(p), last="2015-08")
        ded.to_parquet(t / "delisted_0000.parquet", index=False)
    return {"history": hist, "insider": ins, "facts": tmp_path / "facts", "tiingo": t, "out": tmp_path / "out" / "r.json"}


def _argv(w, *extra):
    return ["--history", str(w["history"]), "--insider-dir", str(w["insider"]), "--facts-dir", str(w["facts"]),
            "--tiingo-dir", str(w["tiingo"]), "--out", str(w["out"]), *extra]


def _run(w, capsys, *extra):
    rc = S.main(_argv(w, *extra))
    return rc, capsys.readouterr().out


# ── dry run, missing inputs, format mismatch ──────────────────────────────────────────────────────

def _heavy_forbidden(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("heavy loading before the input check")

    monkeypatch.setattr(S.M, "build_panel", boom)
    monkeypatch.setattr(S.SI, "load_dir", boom)


def test_cli_dry_run_all_missing(tmp_path, capsys, monkeypatch):
    _heavy_forbidden(monkeypatch)
    rc = S.main(["--dry-run", "--history", str(tmp_path / "h"), "--insider-dir", str(tmp_path / "i"),
                 "--facts-dir", str(tmp_path / "f"), "--tiingo-dir", str(tmp_path / "t")])
    out = capsys.readouterr().out
    assert rc == 0
    assert out.count("MISSING") >= 4
    assert "PROVISIONAL: survivors only" in out
    assert "dry run: nothing computed" in out
    assert "Traceback" not in out
    assert out.index("PROVISIONAL: survivors only") < out.index("dry run: nothing computed")


def test_cli_dry_run_reports_present_inputs(tmp_path, capsys):
    w = _make_world(tmp_path)
    rc, out = _run(w, capsys, "--dry-run")
    assert rc == 0
    lines = out.splitlines()
    by_name = {ln.split()[0]: ln for ln in lines if ln.startswith("  ")}
    assert "present" in by_name["history"] and "1 batch files, no terminal.json" in by_name["history"]
    assert "present" in by_name["insider"] and "2 zips, quarters 2015q1 to 2015q2" in by_name["insider"]
    assert "MISSING" in by_name["facts"] and "MISSING" in by_name["tiingo"]
    assert not w["out"].exists()


def test_cli_missing_insider_exits_2(tmp_path, capsys, monkeypatch):
    _heavy_forbidden(monkeypatch)
    hist = tmp_path / "h"
    hist.mkdir()
    (hist / "batch_0000.parquet").write_bytes(b"")             # contents are never read at this stage
    out_path = tmp_path / "r.json"
    rc = S.main(["--history", str(hist), "--insider-dir", str(tmp_path / "none"), "--facts-dir", str(tmp_path / "f"),
                 "--tiingo-dir", str(tmp_path / "t"), "--out", str(out_path)])
    out = capsys.readouterr().out
    assert rc == 2
    assert "cannot run: missing required input(s): insider" in out
    assert "Traceback" not in out
    assert not out_path.exists()


def test_cli_missing_history_and_insider_lists_both(tmp_path, capsys, monkeypatch):
    _heavy_forbidden(monkeypatch)
    rc = S.main(["--history", str(tmp_path / "h"), "--insider-dir", str(tmp_path / "i"),
                 "--facts-dir", str(tmp_path / "f"), "--tiingo-dir", str(tmp_path / "t")])
    out = capsys.readouterr().out
    assert rc == 2
    assert "cannot run: missing required input(s): history, insider" in out


def test_cli_format_mismatch_exits_3(tmp_path, capsys):
    w = _make_world(tmp_path)
    for z in w["insider"].iterdir():
        z.unlink()
    _zip(w["insider"] / "2015q1_form345.zip", {
        "SUBMISSION.tsv": (["ACCESSION_NUMBER", "FILING_DATE", "DOCUMENT_TYPE", "ISSUERCIK"], [["a", "05-JAN-2015", "4", "1001"]]),
        "REPORTINGOWNER.tsv": (["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNER_RELATIONSHIP"], [["a", "9", "Director"]]),
        "NONDERIV_TRANS.tsv": (["ACCESSION_NUMBER", "TRANS_CODE", "TRANS_SHARES", "TRANS_PRICEPERSHARE"], [["a", "P", "1", "1"]])})
    rc, out = _run(w, capsys)
    assert rc == 3
    assert "insider data format mismatch" in out
    assert "ISSUERTRADINGSYMBOL" in out and "columns found" in out
    assert "Traceback" not in out
    assert not w["out"].exists()


def test_cli_corrupt_zip_exits_3(tmp_path, capsys):
    w = _make_world(tmp_path)
    (w["insider"] / "2015q3_form345.zip").write_bytes(b"this is not a zip")
    rc, out = _run(w, capsys)
    assert rc == 3
    assert "insider data format mismatch" in out and "Traceback" not in out


# ── inputs report, facts loader, writer, progress ─────────────────────────────────────────────────

def test_inputs_report_counts_names_only_and_applies_the_provisional_rule(tmp_path):
    h, i, f, t = (tmp_path / n for n in ("h", "i", "f", "t"))
    for d in (h, i, f, t):
        d.mkdir()
    (h / "batch_0000.parquet").write_bytes(b"")
    (h / "batch_0001.parquet").write_bytes(b"")
    (h / "manifest.json").write_text("{}")
    (i / "2015q1_form345.zip").write_bytes(b"")
    (f / "CIK0000001001.json").write_text("{}")
    (f / "CIK0000001002.json").write_text("{}")
    (f / "other.json").write_text("{}")
    (t / "other.parquet").write_bytes(b"")
    rep = S.inputs_report(h, i, f, t)
    assert rep["history"]["n_batches"] == 2 and rep["history"]["terminal_json"] is False
    assert rep["insider"]["n_zips"] == 1 and rep["insider"]["quarters"] == ["2015q1"]
    assert rep["facts"]["n_files"] == 2
    assert rep["tiingo"]["n_files"] == 0
    assert rep["provisional"] is True and rep["missing_required"] == []
    (t / "delisted_0000.parquet").write_bytes(b"")                       # delisted prices: not provisional
    assert S.inputs_report(h, i, f, t)["provisional"] is False
    (t / "delisted_0000.parquet").unlink()
    (h / "terminal.json").write_text("[]")                              # empty terminal.json: still provisional
    assert S.inputs_report(h, i, f, t)["provisional"] is True
    (h / "terminal.json").write_text("{not json")                       # unreadable: lists no names, provisional
    assert S.inputs_report(h, i, f, t)["provisional"] is True
    (h / "terminal.json").write_text('["DEAD"]')                        # one delisted name listed: not provisional
    rep = S.inputs_report(h, i, f, t)
    assert rep["provisional"] is False and rep["history"]["n_terminal"] == 1
    none = S.inputs_report(tmp_path / "a", tmp_path / "b", tmp_path / "c", tmp_path / "d")
    assert none["missing_required"] == ["history", "insider"] and none["provisional"] is True
    assert none["insider"]["exists"] is False and none["facts"]["exists"] is False


def test_load_facts_reads_cache_layout_and_raw_layout_without_network(tmp_path):
    facts = {"facts": {"dei": {}}}
    (tmp_path / "CIK0000001002.json").write_text(json.dumps({"fetched_at": 0, "data": facts}))   # ancient: TTL is ignored
    (tmp_path / "CIK0000001003.json").write_text(json.dumps(facts))
    (tmp_path / "CIK0000001001.json").write_text("{not json")
    assert S.load_facts(tmp_path, "0000001002") == facts
    assert S.load_facts(tmp_path, "0000001003") == facts
    assert S.load_facts(tmp_path, "0000001001") is None
    assert S.load_facts(tmp_path, "0000009999") is None
    assert S.load_facts(tmp_path / "no_such_dir", "0000001002") is None


def test_write_json_is_strict(tmp_path):
    path = tmp_path / "sub" / "x.json"
    S.write_json({"a": float("nan"), "b": np.float64(np.inf), "c": pd.Timestamp("2015-01-31"), 2015: np.int64(3)}, path)
    assert _strict(path.read_text()) == {"a": None, "b": None, "c": "2015-01-31", "2015": 3}
    assert [p.name for p in path.parent.iterdir()] == ["x.json"]         # the temp file was moved into place
    S.write_json({"a": 1}, path)                                         # overwrites atomically
    assert _strict(path.read_text()) == {"a": 1}


def test_progress_prints_an_eta():
    ticks = iter([0.0, 5.0, 10.0, 20.0])
    lines = []
    prog = S.Progress("work", 10, every=7.0, clock=lambda: next(ticks), out=lines.append)
    prog.update(1)                           # t=5: under the interval, silent
    prog.update(5)                           # t=10: 10 s for 5 of 10 items, so 10 s to go
    prog.update(10)                          # the last item always prints
    assert len(lines) == 2
    assert "5/10 (50%)" in lines[0] and "elapsed 10s" in lines[0] and "eta 10s" in lines[0]
    assert "10/10 (100%)" in lines[1] and "eta 0s" in lines[1]


# ── end to end on the synthetic world ─────────────────────────────────────────────────────────────

def _e2e_checks(res, *, n_control, n_fallen):
    lg = res["long"]
    c = lg["counts"]
    assert c["n_cluster"] == 2 and c["n_cluster_with_r12"] == 2 and c["n_matched"] == 2
    assert c["n_control"] == n_control and c["n_single"] == 0 and c["n_excluded_ambiguous"] == 0
    assert lg["primary"]["mean"] == pytest.approx(0.2)                   # AAA r12 0.2 against controls at 0.0
    assert lg["primary"]["lo"] == pytest.approx(0.2) and lg["primary"]["hi"] == pytest.approx(0.2)
    assert lg["primary"]["n"] == 2 and lg["primary"]["n_clusters"] == 2 and lg["primary"]["alpha"] == 0.025
    assert lg["primary"]["n_stocks"] == 1
    cov = res["coverage"]
    assert cov["n_fallen_stock_months"] == n_fallen
    assert cov["insider_quarters"] == ["2015q1", "2015q2"] and len(cov["quarter_stats"]) == 2
    assert cov["n_purchases"] == 2 and cov["n_cluster_events"] == 1
    assert sum(lg["group_counts"].values()) == n_fallen
    assert res["constants"]["ALPHA"] == 0.025 and res["constants"]["MIN_VALUE"] == 5000.0
    assert "97.5%" in res["preregistration"]


def test_cli_e2e_provisional(tmp_path, capsys, monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError("the download helper must never be called by the CLI")

    monkeypatch.setattr(SI, "download_quarters", forbidden)
    w = _make_world(tmp_path)
    rc, out = _run(w, capsys)
    assert rc == 0, out
    res = _strict(w["out"].read_text())
    assert res["provisional"] is True
    assert res["caveats"][0].startswith("PROVISIONAL") and res["caveats"][0] == PROV
    _e2e_checks(res, n_control=10, n_fallen=177)                         # 3 stocks x 59 months (2014-01 to 2018-11)
    assert res["verdicts"]["long"] == f"{PROV} | PASS"                   # both clusters give 0.2, so lo = 0.2 > 0
    assert res["verdicts"]["short"] == f"{PROV} | not run: no companyfacts directory"
    assert res["short"] == {"ran": False, "reason": "no companyfacts directory"}
    assert res["inputs"]["provisional"] is True and res["inputs"]["facts"]["exists"] is False
    lines = out.splitlines()
    first_result = next(i for i, ln in enumerate(lines) if ln.startswith("LONG  groups"))
    assert lines[first_result - 1] == PROV                               # the stamp opens the result block
    assert f"LONG  verdict: {PROV} | PASS" in out
    assert "Traceback" not in out
    assert not list(w["out"].parent.glob("*.tmp"))


def test_cli_e2e_with_tiingo_not_provisional(tmp_path, capsys):
    w = _make_world(tmp_path, tiingo=True)
    rc, out = _run(w, capsys)
    assert rc == 0, out
    res = _strict(w["out"].read_text())
    assert res["provisional"] is False
    assert PROV not in res["caveats"] and not res["caveats"][0].startswith("PROVISIONAL")
    # DED (fallen 2014-01 to 2015-08 on the raw grid, held flat afterwards, so r12 = 0.0) adds one control in each of
    # 2015-03 to 2015-06, and 20 fallen stock-months.
    _e2e_checks(res, n_control=14, n_fallen=197)
    assert res["verdicts"]["long"] == "PASS"
    assert res["verdicts"]["short"] == "not run: no companyfacts directory"
    assert res["coverage"]["n_terminal_symbols"] == 1 and res["inputs"]["tiingo"]["n_files"] == 1
    assert PROV not in out


def test_cli_tiingo_files_without_delisted_symbols_stay_provisional(tmp_path, capsys, monkeypatch):
    """Breaks if delisted FILES rather than delisted SYMBOLS actually loaded lift the PROVISIONAL stamp (C10)."""
    real = S.load_prices
    monkeypatch.setattr(S, "load_prices", lambda *a, **k: (*real(*a, **k)[:3], frozenset()))
    w = _make_world(tmp_path, tiingo=True)
    rc, out = _run(w, capsys)
    assert rc == 0, out
    res = _strict(w["out"].read_text())
    assert res["inputs"]["provisional"] is False                         # the file-name read said "not provisional"
    assert res["provisional"] is True and res["caveats"][0] == PROV       # the loaded symbols overrule it
    assert res["verdicts"]["long"].startswith(PROV)


def test_held_flat_names_are_not_origins_in_the_cli_universe(tmp_path, capsys):
    """DED has no bars after 2015-08: its held-flat months must not become origins (20 months, not 20 + 100)."""
    w = _make_world(tmp_path, tiingo=True)
    rc, _ = _run(w, capsys)
    res = _strict(w["out"].read_text())
    assert rc == 0 and res["coverage"]["n_fallen_stock_months"] - 177 == 20


def _facts(shares, ocf=None, cash=None):
    def rows(items, extra=None):
        return [{"end": e, "val": v, "filed": f, "form": "10-K", "fy": 2014, "fp": "FY", "accn": f"a-{f}",
                 **({"start": s} if s else {})} for (s, e, v, f) in items]

    gaap = {}
    if ocf:
        gaap["NetCashProvidedByUsedInOperatingActivities"] = {"units": {"USD": rows(ocf)}}
    if cash:
        gaap["CashAndCashEquivalentsAtCarryingValue"] = {"units": {"USD": rows(cash)}}
    return {"facts": {"dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": rows(shares)}}},
                      "us-gaap": gaap}}


def test_cli_e2e_short_arm_with_facts_and_a_process_pool(tmp_path, capsys):
    w = _make_world(tmp_path)
    w["facts"].mkdir()
    diluted = [(None, "2014-03-31", 1_000_000, "2014-04-15"), (None, "2015-03-31", 2_000_000, "2015-04-15")]
    flat = [(None, "2014-03-31", 1_000_000, "2014-04-15"), (None, "2015-03-31", 1_000_000, "2015-04-15")]
    # BBB: shares doubled and a runway of 500,000 / 1,000,000 = 0.5 years -> FLAGGED from the 2015-04 month-end on.
    bbb = _facts(diluted, ocf=[("2014-01-01", "2014-12-31", -1_000_000, "2015-03-01")],
                 cash=[(None, "2014-12-31", 500_000, "2015-03-01")])
    # CCC: no dilution and positive operating cash flow -> UNFLAGGED (raw companyfacts layout, no cache wrapper).
    ccc = _facts(flat, ocf=[("2014-01-01", "2014-12-31", 100_000, "2015-03-01")])
    (w["facts"] / "CIK0000001002.json").write_text(json.dumps({"fetched_at": 0, "data": bbb}))
    (w["facts"] / "CIK0000001003.json").write_text(json.dumps(ccc))
    (w["facts"] / "CIK0000001001.json").write_text("{not json")          # AAA: unreadable -> unknown, never unflagged
    rc, out = _run(w, capsys, "--workers", "2")
    assert rc == 0, out
    res = _strict(w["out"].read_text())
    sh = res["short"]
    assert sh["ran"] is True
    # BBB and CCC map from 2015-01 to 2016-03 (last Form 4 filed 2015-04-15, plus 365 days); the shares a year earlier
    # exist from the 2015-04 month-end, so 2015-01 to 2015-03 are unknown. Flagged and unflagged: 2015-04 to 2016-03 = 12 each.
    assert sh["counts"] == {"n_flagged": 12, "n_unflagged": 12, "n_unknown": 22, "n_no_outcome": 0}
    assert sh["facts"] == {"n_rows_with_cik": 46, "n_ciks": 3, "n_facts_found": 2, "n_facts_missing": 0,
                           "n_facts_unreadable": 1}
    assert sh["coverage_by_year"] == {"2015": 18, "2016": 6} and sh["first_classified_month"] == "2015-04"
    assert sh["rates"] == {"flagged": 0.0, "unflagged": 0.0}              # the price never moves, so no wipeouts
    assert sh["primary"]["mean"] == 0.0 and sh["primary"]["lo"] == 0.0 and sh["primary"]["alpha"] == 0.025
    assert sh["robust_stock_clusters"]["n_clusters"] == 2
    assert res["verdicts"]["short"] == f"{PROV} | no demonstrated edge"   # a lower bound of exactly 0 is not a PASS
    assert res["verdicts"]["long"] == f"{PROV} | PASS"
    robust = [ln for ln in out.splitlines() if "robustness" in ln]
    assert len(robust) == 2 and all("97.5% interval" in ln for ln in robust)
    assert "eta" in out                                                   # the progress line


def test_cli_universe_with_no_fallen_stock_still_writes_strict_json(tmp_path, capsys):
    w = _make_world(tmp_path)
    _bars("AAA", lambda p: 50.0).to_parquet(w["history"] / "batch_0000.parquet", index=False)
    rc, out = _run(w, capsys)
    assert rc == 0, out
    res = _strict(w["out"].read_text())
    assert res["coverage"]["n_fallen_stock_months"] == 0
    assert res["verdicts"]["long"] == f"{PROV} | no demonstrated edge (interval not computable)"
    assert res["long"]["counts"]["n_cluster"] == 0 and "Traceback" not in out


def test_cli_never_wires_the_download_helper():
    tree = ast.parse((ROOT / "scripts" / "insider_study.py").read_text())
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert "download_quarters" not in names and "default_fetch" not in names and "EdgarClient" not in names


# ── the docstring is the pre-registration ─────────────────────────────────────────────────────────

VERDICT_SENTENCE = ("LONG PASS iff the 97.5% lower bound of the 12-month excess return is > 0; SHORT PASS iff the 97.5% "
                    "lower bound of (flagged wipeout rate minus unflagged wipeout rate) is > 0; otherwise no "
                    "demonstrated edge.")


def test_prereg_text_in_sync():
    doc = (ROOT / "docs" / "INSIDER_STUDY_PREREG.md").read_text()
    for needle in ("97.5%", "no demonstrated edge", "PROVISIONAL: survivors only", "filing date", "$5,000", "30",
                   "90 days", "20%", VERDICT_SENTENCE):
        assert needle in S.__doc__, f"missing from the script docstring: {needle!r}"
        assert needle in doc, f"missing from docs/INSIDER_STUDY_PREREG.md: {needle!r}"
    assert S.PREREGISTRATION and "97.5%" in S.PREREGISTRATION and "$5,000" in S.PREREGISTRATION
    assert "30 calendar days" in S.PREREGISTRATION and "90 days" in S.PREREGISTRATION and "20%" in S.PREREGISTRATION
    assert IS.PROVISIONAL in S.__doc__ and S.CI_LABEL == "97.5%"
    for c in ("C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12"):
        assert re.search(rf"\b{c}\b", S.__doc__), f"{c} not referenced in the docstring"
    assert "EXIT CODES" in S.__doc__ and "KNOWN BIASES" in S.__doc__


NEW_FILES = ("data_engine/sec_insider.py", "setup_engine/insider_study.py", "scripts/insider_study.py",
             "docs/INSIDER_STUDY_PREREG.md", "tests/test_data_engine/test_sec_insider.py",
             "tests/test_study/test_insider_study.py", "tests/test_study/test_insider_study_cli.py")


def test_new_files_hygiene():
    email = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
    em_dash = chr(0x2014)
    for rel in NEW_FILES:
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert em_dash not in text, f"em dash in {rel}"
        for m in email.findall(text):
            assert m.rstrip(".").endswith(("example.org", "example.com", "example.invalid")), f"{m!r} in {rel}"
