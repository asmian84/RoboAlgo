"""setup_engine.apply: tagging scan hits with approved setups, and never failing quietly.

The two ways this goes wrong are tagging with a setup that was not approved (a rejected pattern
quietly becomes a buy signal) and failing silently (an empty tag list reads as "matches nothing").
"""
import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from setup_engine import apply as A
from setup_engine import learn as L


def _daily(n=700, drift=0.0006, seed=1, start="2023-01-02"):
    rng = np.random.default_rng(seed)
    close = 20 * np.exp(np.cumsum(rng.normal(drift, 0.015, n)))
    open_ = close * (1 + rng.normal(0, 0.004, n))
    high = np.maximum(open_, close) * 1.004
    low = np.minimum(open_, close) * 0.996
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": np.full(n, 2_000_000.0)}, index=pd.bdate_range(start, periods=n))


def _ref(asof=None):
    levels = [float(q) for q in A._QUANTILES]
    vals = list(np.linspace(-0.5, 0.5, len(levels)))                      # a simple, known distribution
    return {"asof": str(asof or date.today()), "n_stocks": 1000, "levels": levels,
            "features": {s: {"values": vals} for s in ("D_mom50", "W_mom30", "W_mom50", "M_mom30")}}


def _reg(status="approved", rule=None):
    rule = rule if rule is not None else [{"feature": "W_mom30", "op": ">", "value": 0.0}]
    return {"setups": [{"id": "r9-up", "rule_text": "uptrend", "status": status, "rule": rule, "provisional": True,
                        "excess_by_era": {"test": {"excess_mean": 0.04, "excess_lo": 0.03, "excess_hi": 0.05}},
                        "robustness": {"liquid_excess_mean": 0.02}}]}


# ── references and ranks ───────────────────────────────────────────────────────

def test_rank_interpolates_against_the_reference():
    ref = _ref()
    assert A.rank_against(ref, "W_mom50", 0.0) == pytest.approx(0.5, abs=0.01)
    assert A.rank_against(ref, "W_mom50", -0.5) == pytest.approx(0.0, abs=0.01)
    assert A.rank_against(ref, "W_mom50", 9.0) == pytest.approx(1.0)               # above everything: top rank, not an error
    assert np.isnan(A.rank_against(ref, "W_mom50", float("nan")))
    assert np.isnan(A.rank_against(ref, "no_such_feature", 0.1))


def test_a_reference_round_trips_through_disk(tmp_path):
    rng = np.random.default_rng(0)
    cur = pd.DataFrame({"date": pd.Timestamp("2026-10-06"), **{s: rng.normal(0, 0.3, 500) for s in ("D_mom50", "W_mom30", "W_mom50", "M_mom30")}})
    path = tmp_path / "ref.json"
    ref = A.build_reference(cur, path)
    again = A.load_reference(path)
    assert again["asof"] == "2026-10-06" and again["n_stocks"] == 500
    assert A.rank_against(again, "W_mom50", float(cur["W_mom50"].median())) == pytest.approx(0.5, abs=0.02)


# ── which setups are applied ─────────────────────────────────────────────────────

def test_only_approved_setups_are_applied():
    assert [s["id"] for s in A.approved_setups(_reg("approved"))] == ["r9-up"]
    assert A.approved_setups(_reg("rejected_robustness")) == []
    assert A.approved_setups(_reg("approved", rule=[])) == []                        # no rule to evaluate, nothing to apply
    assert A.approved_setups({"setups": []}) == []


def test_a_rejected_setup_is_never_tagged_even_though_it_would_match():
    results = [{"symbol": "UP"}]
    st = A.annotate(results, {"UP": _daily(drift=0.002)}, reg=_reg("rejected_robustness"), ref=_ref())
    assert results[0]["setup_evidence"] == [] and st["approved_setups"] == 0


# ── tagging ─────────────────────────────────────────────────────────────────────

def test_a_matching_stock_is_tagged_and_a_non_matching_one_is_not():
    results = [{"symbol": "UP"}, {"symbol": "DOWN"}]
    daily = {"UP": _daily(drift=0.003, seed=2), "DOWN": _daily(drift=-0.003, seed=3)}
    st = A.annotate(results, daily, reg=_reg(), ref=_ref())
    assert st["ok"] and st["annotated"] == 2
    assert [c["id"] for c in results[0]["setup_evidence"]] == ["r9-up"]
    assert results[1]["setup_evidence"] == []
    card = results[0]["setup_evidence"][0]
    assert card["provisional"] is True and card["excess_6m_test"] == 0.04 and card["excess_6m_test_ci"] == [0.03, 0.05]


def test_annotation_never_reorders_filters_or_drops_results():
    results = [{"symbol": s, "marker": i} for i, s in enumerate(["B", "A", "C"])]
    daily = {s: _daily(drift=d, seed=i) for i, (s, d) in enumerate(zip("BAC", (0.003, -0.003, 0.003)))}
    A.annotate(results, daily, reg=_reg(), ref=_ref())
    assert [r["symbol"] for r in results] == ["B", "A", "C"] and [r["marker"] for r in results] == [0, 1, 2]


def test_bars_are_fetched_only_for_symbols_the_caller_did_not_supply():
    seen = []
    def fetch(symbols):
        seen.extend(symbols)
        return {s: _daily(seed=5) for s in symbols}
    results = [{"symbol": "HAVE"}, {"symbol": "NEED"}]
    A.annotate(results, {"HAVE": _daily(seed=4)}, reg=_reg(), ref=_ref(), fetch=fetch)
    assert seen == ["NEED"]


# ── failure must be visible ───────────────────────────────────────────────────────

def test_too_little_history_is_marked_unevaluable_not_silently_empty():
    results = [{"symbol": "NEW"}]
    st = A.annotate(results, {"NEW": _daily(n=200)}, reg=_reg(), ref=_ref())
    assert results[0]["setup_evidence"] is None and "history" in results[0]["setup_evidence_error"]
    assert st["unevaluable"] == 1 and st["ok"]


def test_a_missing_reference_fails_loudly_on_every_result_and_never_raises():
    results = [{"symbol": "A"}, {"symbol": "B"}]
    st = A.annotate(results, {"A": _daily(), "B": _daily()}, reg=_reg(), ref=None) if A.load_reference() is None else None
    if st is None:   # a real reference exists on this machine: force the missing case explicitly
        orig = A.load_reference
        A.load_reference = lambda path=None: None
        try:
            st = A.annotate(results, {"A": _daily(), "B": _daily()}, reg=_reg())
        finally:
            A.load_reference = orig
    assert st["ok"] is False and "reference" in st["error"]
    assert all(r["setup_evidence"] is None and "reference" in r["setup_evidence_error"] for r in results)


def test_a_fetch_failure_fails_loudly_and_the_scan_survives():
    def boom(symbols):
        raise ConnectionError("yahoo is down")
    results = [{"symbol": "X"}]
    st = A.annotate(results, {}, reg=_reg(), ref=_ref(), fetch=boom)
    assert st["ok"] is False and "yahoo is down" in st["error"]
    assert results[0]["setup_evidence"] is None and "yahoo is down" in results[0]["setup_evidence_error"]


def test_a_stale_reference_is_flagged():
    old = date.today() - timedelta(days=A.STALE_DAYS + 5)
    st = A.annotate([{"symbol": "A"}], {"A": _daily()}, reg=_reg(), ref=_ref(asof=old))
    assert st["reference_stale"] is True
    fresh = A.annotate([{"symbol": "A"}], {"A": _daily()}, reg=_reg(), ref=_ref())
    assert fresh["reference_stale"] is False


def test_no_results_is_a_clean_no_op():
    assert A.annotate([], {}, reg=_reg(), ref=_ref())["ok"] is True


# ── the Finder hook ────────────────────────────────────────────────────────────────

def test_the_finder_hook_swallows_an_annotation_crash_and_says_so(monkeypatch):
    from scanner_engine import finder
    monkeypatch.setattr(A, "annotate", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("kaboom")))
    results = [{"symbol": "A"}, {"symbol": "B"}]
    st = finder.attach_setup_evidence(results)
    assert st["ok"] is False and "kaboom" in st["error"]
    assert all(r["setup_evidence"] is None and "kaboom" in r["setup_evidence_error"] for r in results)
