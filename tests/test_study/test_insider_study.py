"""setup_engine/insider_study.py: the insider-cluster and dilution-plus-burn study must not look into the future.

The failures being guarded against: treating a stock-month as "cluster" because of a Form 4 that became public
only after the origin close, a 90-day window that is 91 days (or 89), a dilution test that misses exactly 20%
through floating point (1.2M / 1.0M - 1 is 0.19999999999999996), a runway boundary that moves, a missing value
silently counted as "not diluted", a quarterly cash-flow row read instead of the annual one, a stale XBRL tag
read as current, a forward window that includes the origin month, a held-flat month after a delisting used as
an origin, and a verdict that calls a lower bound of exactly 0 a pass. Every expected number below is computed
by hand from the formulas that build the synthetic frames, never from the code under test. The frames follow the
frozen layouts of data_engine/sec_insider.py, which is NOT imported here, and nothing touches the network.
"""
import ast
import importlib.util
import re
import socket
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from setup_engine import insider_study as IS
from setup_engine import trade_stats as TS

E = IS.E
ROOT = Path(__file__).resolve().parents[2]
NAN = float("nan")
Q3 = {"2015q1", "2015q2", "2015q3"}


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network access attempted in an offline test")

    monkeypatch.setattr(socket.socket, "connect", boom)


def ts(s):
    return pd.Timestamp(s)


def P(s):
    return pd.Period(s, "M")


# -- constants -----------------------------------------------------------------------------------

def _load_multibagger():
    name = "multibagger_study_for_insider_constants"
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "multibagger_study.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_constants_match_preregistration_and_multibagger():
    M = _load_multibagger()
    assert IS.TREAT_WINDOW_DAYS == 90
    assert IS.ALPHA == 0.025 and IS.ALPHA_SECONDARY == 0.05
    assert IS.N_BOOT == 2000 and IS.SEED == 0
    assert IS.ORIGINS == ("2006-01", "2023-09")
    assert IS.DILUTION_MIN_RATIO == 1.20 and IS.RUNWAY_MAX_YEARS == 1.0
    assert IS.HORIZONS == (6, 12, 36) and IS.PRIMARY_H == 12
    assert IS.WIPEOUT_WIN == 12 and IS.WIPEOUT_LEVEL == 0.5
    assert IS.ANNUAL_DAYS == (350, 380) and IS.MAX_STALE_DAYS == 550 and IS.SHARES_LAG_DAYS == 365
    assert IS.FALLEN_PRICE == M.FALLEN_PRICE
    assert IS.FALLEN_DD == M.FALLEN_DD
    assert IS.MIN_DVOL == M.MIN_DVOL
    assert IS.PROVISIONAL == "PROVISIONAL: survivors only (delisted names missing)"
    assert IS.GROUPS == ("cluster", "single", "control", "excluded_unmapped", "excluded_ambiguous",
                         "excluded_no_coverage")
    assert IS.SHORT_CLASSES == ("flagged", "unflagged", "unknown")


# -- dates -----------------------------------------------------------------------------------------

def test_last_weekday_and_pit_date():
    me = pd.Series(pd.to_datetime(["2015-05-31", "2015-02-28", "2015-06-30"]))
    lw = IS.last_weekdays(me)
    assert lw.tolist() == [ts("2015-05-29"), ts("2015-02-27"), ts("2015-06-30")]
    assert IS.pit_dates(me).tolist() == [ts("2015-05-28"), ts("2015-02-26"), ts("2015-06-29")]
    assert ts("2015-05-31").weekday() == 6 and ts("2015-02-28").weekday() == 5      # Sunday, Saturday


def test_last_weekday_keeps_the_index():
    me = pd.Series(pd.to_datetime(["2015-05-31", "2015-06-30"]), index=[7, 3])
    assert IS.last_weekdays(me).index.tolist() == [7, 3]


# -- prices ----------------------------------------------------------------------------------------

def test_forward_outcomes_hand_computed():
    idx = pd.period_range("2010-01", "2013-12", freq="M")                 # 48 months
    up = np.arange(1, 49, dtype=float)                                    # C[i] = i + 1
    dn = np.full(48, 95.0)
    dn[0] = 100.0
    dn[1:13] = [90, 80, 70, 60, 55, 50, 51, 60, 70, 80, 90, 95]
    dn2 = dn.copy()
    dn2[6] = 50.01                                                        # the 50 becomes 50.01
    gap = np.full(48, 10.0)
    gap[12] = NAN
    C = pd.DataFrame({"UP": up, "DN": dn, "DN2": dn2, "GAP": gap}, index=idx)
    F = IS.forward_outcomes(C)
    assert set(F) == {"r6", "r12", "r36", "fmin12", "wipe12"}
    first = P("2010-01")
    # UP at the first month: the origin month is excluded from the fmin window
    assert F["r6"].loc[first, "UP"] == pytest.approx(6.0)                 # 7 / 1 - 1
    assert F["r12"].loc[first, "UP"] == pytest.approx(12.0)               # 13 / 1 - 1
    assert F["r36"].loc[first, "UP"] == pytest.approx(36.0)               # 37 / 1 - 1
    assert F["fmin12"].loc[first, "UP"] == pytest.approx(2.0)             # min(C[1..12]) / C[0] = 2 / 1
    assert F["wipe12"].loc[first, "UP"] == 0.0
    assert F["fmin12"].loc[P("2012-12"), "UP"] == pytest.approx(37 / 36)  # i = 35: min(C[36..47]) = 37, C[35] = 36
    assert F["r12"].loc[P("2012-12"), "UP"] == pytest.approx(48 / 36 - 1)
    assert np.isnan(F["r12"].loc[P("2013-01"), "UP"])                     # i = 36: C[48] does not exist
    assert F["r6"].loc[P("2013-06"), "UP"] == pytest.approx(48 / 42 - 1)  # i = 41
    assert np.isnan(F["r6"].loc[P("2013-07"), "UP"])                      # i = 42: C[48] does not exist
    assert F["r36"].loc[P("2010-12"), "UP"] == pytest.approx(48 / 12 - 1)  # i = 11
    assert np.isnan(F["r36"].loc[P("2011-01"), "UP"])
    # DN: the lowest close of the next 12 months is 50 = exactly half of 100, and <= is inclusive
    assert F["fmin12"].loc[first, "DN"] == pytest.approx(0.5)
    assert F["wipe12"].loc[first, "DN"] == 1.0
    assert F["r12"].loc[first, "DN"] == pytest.approx(-0.05)              # 95 / 100 - 1
    # DN2: 50.01 is above half
    assert F["fmin12"].loc[first, "DN2"] == pytest.approx(0.5001)
    assert F["wipe12"].loc[first, "DN2"] == 0.0
    # GAP: C[12] missing -> every 12-month outcome is NaN, the 6-month one is not
    assert np.isnan(F["r12"].loc[first, "GAP"])
    assert np.isnan(F["fmin12"].loc[first, "GAP"])
    assert np.isnan(F["wipe12"].loc[first, "GAP"])
    assert F["r6"].loc[first, "GAP"] == pytest.approx(0.0)
    assert F["r36"].loc[first, "GAP"] == pytest.approx(0.0)


def test_forward_outcomes_requires_contiguous_months():
    idx = pd.PeriodIndex(["2010-01", "2010-03", "2010-04"], freq="M")
    with pytest.raises(ValueError, match="contiguous"):
        IS.forward_outcomes(pd.DataFrame({"X": [1.0, 2.0, 3.0]}, index=idx))


def test_forward_outcomes_skips_gap_inside_the_window():
    idx = pd.period_range("2010-01", periods=14, freq="M")
    c = np.full(14, 10.0)
    c[3] = NAN                                                            # a missing month inside the window
    c[5] = 4.0                                                            # the real low
    F = IS.forward_outcomes(pd.DataFrame({"X": c}, index=idx))
    assert F["fmin12"].iloc[0, 0] == pytest.approx(0.4)                   # NaN neither poisons nor counts as 0
    assert F["wipe12"].iloc[0, 0] == 1.0


def test_fallen_mask_boundaries():
    cases = [(1.00, -0.80, 300000, True), (0.99, -0.80, 1e6, False), (10.00, -0.70, 1e6, True),
             (10.01, -0.80, 1e6, False), (5.0, -0.6999, 1e6, False), (5.0, -0.80, 299999, False),
             (NAN, -0.80, 1e6, False), (5.0, NAN, 1e6, False), (5.0, -0.80, NAN, False)]
    idx = pd.PeriodIndex(["2010-01"], freq="M")
    cols = [f"S{i}" for i in range(len(cases))]
    C = pd.DataFrame([[c[0] for c in cases]], index=idx, columns=cols)
    D = pd.DataFrame([[c[1] for c in cases]], index=idx, columns=cols)
    V = pd.DataFrame([[c[2] for c in cases]], index=idx, columns=cols)
    got = IS.fallen_mask(C, V, D).iloc[0].tolist()
    assert got == [c[3] for c in cases]


def test_held_flat_month_is_not_an_origin():
    idx = pd.period_range("2001-01", "2007-12", freq="M")
    raw = np.full(len(idx), np.nan)
    raw[:60] = 50.0                                                       # 2001-01 .. 2005-12
    raw[60:66] = 5.0                                                      # 2006-01 .. 2006-06, then delisted
    held = raw.copy()
    held[66:] = 5.0                                                       # held flat at the last close
    C_raw = pd.DataFrame({"DEAD": raw}, index=idx)
    C_held = pd.DataFrame({"DEAD": held}, index=idx)
    DV = pd.DataFrame({"DEAD": np.full(len(idx), 1e6)}, index=idx)
    dd60 = C_raw / C_raw.rolling(60, min_periods=12).max() - 1
    sm = IS.stock_months(C_raw, DV, dd60, C_held)
    assert sm["month"].tolist() == [P(f"2006-0{m}") for m in range(1, 7)]
    last = sm[sm["month"] == P("2006-06")].iloc[0]
    assert last["r12"] == 0.0                                             # from the held-flat grid: 5 / 5 - 1
    assert P("2006-07") not in set(sm["month"])
    assert list(sm.columns) == IS.SM_COLUMNS
    assert str(sm["month"].dtype) == "period[M]"
    assert sm["month_end"].iloc[-1] == ts("2006-06-30")
    assert sm["month_end"].dtype == "datetime64[ns]"
    # even when dd60 is (wrongly) computed on the held-flat grid and is finite after the delisting, the RAW grid
    # decides eligibility: the held-flat months 2006-07 .. 2007-12 are never origins
    dd60_held = C_held / C_held.rolling(60, min_periods=12).max() - 1
    assert dd60_held.loc[P("2007-06"), "DEAD"] <= -0.7
    sm_held = IS.stock_months(C_raw, DV, dd60_held, C_held)
    assert sm_held["month"].tolist() == [P(f"2006-0{m}") for m in range(1, 7)]


def test_stock_months_origin_range_and_nan_outcomes_kept():
    idx = pd.period_range("2005-01", "2024-12", freq="M")
    C = pd.DataFrame({"X": np.full(len(idx), 5.0)}, index=idx)
    DV = pd.DataFrame({"X": np.full(len(idx), 1e6)}, index=idx)
    dd = pd.DataFrame({"X": np.full(len(idx), -0.8)}, index=idx)
    sm = IS.stock_months(C, DV, dd, C)
    assert sm["month"].iloc[0] == P("2006-01") and sm["month"].iloc[-1] == P("2023-09")   # both inclusive
    assert len(sm) == 213
    assert sm["r12"].notna().all()                                         # 2024-09 exists on this grid
    # a grid that ends 2024-03: origins from 2023-04 on have no 12-month outcome, but the rows are KEPT
    short = idx[idx <= P("2024-03")]
    sm2 = IS.stock_months(C.loc[short], DV.loc[short], dd.loc[short], C.loc[short])
    assert len(sm2) == 213
    assert sm2.loc[sm2["month"] <= P("2023-03"), "r12"].notna().all()
    assert sm2.loc[sm2["month"] >= P("2023-04"), "r12"].isna().all()
    assert sm2.loc[sm2["month"] >= P("2023-04"), "wipe12"].isna().all()
    custom = IS.stock_months(C, DV, dd, C, origins=("2023-08", "2023-12"))
    assert custom["month"].tolist() == [P("2023-08"), P("2023-09"), P("2023-10"), P("2023-11"), P("2023-12")]


# -- long arm: treatment assignment ------------------------------------------------------------------

EV_COLS = ["issuer_cik", "symbol", "event_date", "n_owners", "accession"]
PU_COLS = ["accession", "filing_date", "issuer_cik", "symbol", "owner_cik", "value", "trade_date"]
CIK_A, CIK_B = "0000001001", "0000002002"


def _events(dates, cik=CIK_A):
    dates = list(dates)
    return pd.DataFrame({"issuer_cik": [cik] * len(dates), "symbol": ["AAA"] * len(dates),
                         "event_date": pd.to_datetime(dates), "n_owners": [2] * len(dates),
                         "accession": [f"E{i}" for i in range(len(dates))]}, columns=EV_COLS)


def _purchases(dates, cik=CIK_A):
    dates = list(dates)
    return pd.DataFrame({"accession": [f"P{i}" for i in range(len(dates))], "filing_date": pd.to_datetime(dates),
                         "issuer_cik": [cik] * len(dates), "symbol": ["AAA"] * len(dates),
                         "owner_cik": [f"900{i}" for i in range(len(dates))], "value": [6000.0] * len(dates),
                         "trade_date": pd.to_datetime(dates)}, columns=PU_COLS)


def _sm_row(month, cik=CIK_A, ambiguous=False):
    p = P(month)
    return pd.DataFrame([{"symbol": "AAA", "month": p, "month_end": p.end_time.normalize(), "close": 5.0,
                          "r6": 0.0, "r12": 0.0, "r36": 0.0, "fmin12": 1.0, "wipe12": 0.0, "cik": cik,
                          "ambiguous": ambiguous}])


def _group(month, events=(), purchases=(), cik=CIK_A, ambiguous=False, quarters=Q3, event_cik=CIK_A):
    out = IS.assign_long(_sm_row(month, cik=cik, ambiguous=ambiguous), _purchases(purchases, cik=event_cik),
                         _events(events, cik=event_cik), quarters)
    return out["group"].iloc[0]


def test_treatment_window_boundary():
    assert (ts("2015-06-30") - ts("2015-04-02")).days == 89
    assert _group("2015-06", events=["2015-04-02"], purchases=["2015-04-02"]) == "cluster"     # 89 days: in
    assert _group("2015-06", events=["2015-04-01"], purchases=["2015-04-01"]) == "control"     # 90 days: out
    assert _group("2015-06", events=["2015-04-01"], purchases=["2015-04-02"]) == "single"      # purchase in, event out


def test_event_on_last_weekday_not_usable():
    # Tuesday month end: an event filed on the month-end day itself is public only after the close
    assert _group("2015-06", events=["2015-06-30"], purchases=["2015-06-30"]) == "single"
    assert _group("2015-06", events=["2015-06-29"], purchases=["2015-06-29"]) == "cluster"
    # Sunday month end 2015-05-31: the last weekday is Friday 05-29
    assert _group("2015-05", events=["2015-05-29"], purchases=["2015-05-29"]) == "single"
    assert _group("2015-05", events=["2015-05-28"], purchases=["2015-05-28"]) == "cluster"


def test_no_look_ahead_after_month_end():
    assert _group("2015-06", events=["2015-07-01"], purchases=["2015-07-01"]) == "control"


def test_single_purchase_is_not_control():
    assert _group("2015-06", purchases=["2015-05-15"]) == "single"
    assert _group("2015-06") == "control"


def test_reused_ticker_other_cik_ignored():
    # the row's issuer is 0000002002; the cluster belongs to 0000001001 (another company that used the symbol)
    g = _group("2015-06", cik=CIK_B, events=["2015-06-01"], purchases=["2015-06-01"], event_cik=CIK_A)
    assert g == "control"


def test_exclusions():
    ev, pu = ["2015-06-01"], ["2015-06-01"]
    assert _group("2015-06", ambiguous=True, events=ev, purchases=pu) == "excluded_ambiguous"
    assert _group("2015-06", cik=None, events=ev, purchases=pu) == "excluded_unmapped"
    assert _group("2015-06", cik=None, ambiguous=True) == "excluded_ambiguous"           # ambiguous outranks unmapped
    # 2015-02 reaches back to 2014-12-01, so 2014q4 must be present
    assert _group("2015-02", events=["2015-01-20"], purchases=["2015-01-20"]) == "excluded_no_coverage"
    assert _group("2015-02", events=["2015-01-20"], purchases=["2015-01-20"],
                  quarters=Q3 | {"2014q4"}) == "cluster"
    # the middle quarter missing is also a hole
    assert _group("2015-06", events=ev, purchases=pu, quarters={"2015q1", "2015q3"}) == "excluded_no_coverage"
    assert _group("2015-06", cik="", events=ev, purchases=pu) == "excluded_unmapped"


def test_assign_long_many_rows_keeps_order_and_input():
    sm = pd.concat([_sm_row("2015-06"), _sm_row("2015-06", cik=CIK_B), _sm_row("2015-05"),
                    _sm_row("2015-06", cik=None)], ignore_index=True)
    ev = _events(["2015-05-10"], cik=CIK_A)
    pu = pd.concat([_purchases(["2015-05-10"], cik=CIK_A), _purchases(["2015-05-20"], cik=CIK_B)], ignore_index=True)
    out = IS.assign_long(sm, pu, ev, Q3)
    assert out["group"].tolist() == ["cluster", "single", "cluster", "excluded_unmapped"]
    assert "group" not in sm.columns                                       # the input is not modified
    with pytest.raises(ValueError, match="ambiguous"):
        IS.assign_long(sm.drop(columns=["ambiguous"]), pu, ev, Q3)
    with pytest.raises(ValueError, match="YYYYqN"):
        IS.assign_long(sm, pu, ev, {"2015-Q1"})


def test_assign_long_empty_inputs():
    out = IS.assign_long(_sm_row("2015-06"), _purchases([]), _events([]), Q3)
    assert out["group"].tolist() == ["control"]
    empty = IS.assign_long(_sm_row("2015-06").iloc[0:0], _purchases([]), _events([]), Q3)
    assert "group" in empty.columns and len(empty) == 0


# -- long arm: statistics --------------------------------------------------------------------------

def _lrow(symbol, me, r12, group, r6=0.0, r36=0.0, wipe12=0.0):
    return {"symbol": symbol, "month": P(me[:7]), "month_end": ts(me), "close": 5.0, "r6": r6, "r12": r12,
            "r36": r36, "fmin12": 1.0, "wipe12": wipe12, "group": group}


def test_long_arm_hand_computed():
    sm = pd.DataFrame([
        _lrow("C1", "2015-06-30", 0.30, "cluster"), _lrow("C2", "2015-07-31", -0.10, "cluster"),
        _lrow("C3", "2015-08-31", 0.05, "cluster"),
        _lrow("K1", "2015-06-30", 0.10, "control"), _lrow("K2", "2015-06-30", 0.20, "control"),
        _lrow("K3", "2015-07-31", 0.00, "control")])
    res = IS.long_arm(sm)
    expected = np.array([0.30 - (0.10 + 0.20) / 2, -0.10 - 0.0])           # August has no control: dropped
    ref = TS.cluster_bootstrap(expected, [201506, 201507], 2000, 0, 0.025)
    p = res["primary"]
    assert p["mean"] == pytest.approx(0.025, abs=1e-12)
    assert p["n"] == 2 and p["n_clusters"] == 2
    assert p["lo"] == pytest.approx(ref["lo"], abs=1e-12) and p["hi"] == pytest.approx(ref["hi"], abs=1e-12)
    assert p["alpha"] == 0.025
    assert p["n_stocks"] == 2
    c = res["counts"]
    assert c["n_cluster"] == 3 and c["n_matched"] == 2 and c["n_control"] == 3 and c["n_cluster_with_r12"] == 3
    assert c["n_single"] == 0 and c["n_excluded_unmapped"] == 0
    cs = res["control_summary"]
    assert cs["n_matched"] == 2
    assert cs["expectancy"] == pytest.approx((0.15 + 0.0) / 2, abs=1e-12)  # mean of the control means


def test_long_arm_secondary_counts_and_reused():
    sm = pd.DataFrame([
        _lrow("R1", "2015-06-30", 0.40, "cluster", r6=0.10, r36=0.50, wipe12=1.0),
        _lrow("C2", "2015-06-30", 0.20, "cluster", r6=0.00, r36=0.30, wipe12=0.0),
        _lrow("C3", "2015-07-31", 0.10, "cluster", r6=0.05, r36=0.20, wipe12=0.0),
        _lrow("K1", "2015-06-30", 0.10, "control", r6=0.00, r36=0.10, wipe12=0.0),
        _lrow("K2", "2015-06-30", 0.30, "control", r6=0.20, r36=0.30, wipe12=1.0),
        _lrow("K3", "2015-07-31", 0.00, "control", r6=0.05, r36=0.00, wipe12=0.0),
        _lrow("R9", "2015-07-31", 0.20, "control", r6=0.15, r36=0.10, wipe12=0.0),
        _lrow("S1", "2015-06-30", 0.50, "single"),
        _lrow("U1", "2015-06-30", 0.00, "excluded_unmapped"),
        _lrow("A1", "2015-06-30", 0.00, "excluded_ambiguous"),
        _lrow("N1", "2015-06-30", 0.00, "excluded_no_coverage")])
    res = IS.long_arm(sm, reused=frozenset({"R1", "R9"}))
    c = res["counts"]
    assert c == {"n_cluster": 3, "n_cluster_with_r12": 3, "n_matched": 3, "n_control": 4, "n_single": 1,
                 "n_excluded_unmapped": 1, "n_excluded_ambiguous": 1, "n_excluded_no_coverage": 1}
    months = [201506, 201506, 201507]
    ex12 = [0.40 - 0.20, 0.20 - 0.20, 0.10 - 0.10]                         # June control mean 0.2, July 0.1
    assert res["primary"]["mean"] == pytest.approx(np.mean(ex12), abs=1e-12)
    rob = TS.cluster_bootstrap(ex12, ["R1", "C2", "C3"], 2000, 0, 0.025)   # primary alpha, stock clusters
    assert res["robust_stock_clusters"]["lo"] == pytest.approx(rob["lo"], abs=1e-12)
    assert res["robust_stock_clusters"]["n_clusters"] == 3
    s = res["secondary"]
    ex6 = [0.10 - 0.10, 0.00 - 0.10, 0.05 - 0.10]
    ex36 = [0.50 - 0.20, 0.30 - 0.20, 0.20 - 0.05]
    for key, ex in (("r6", ex6), ("r36", ex36)):
        ref = TS.cluster_bootstrap(ex, months, 2000, 0, 0.05)
        assert s[key]["mean"] == pytest.approx(ref["mean"], abs=1e-12)
        assert s[key]["lo"] == pytest.approx(ref["lo"], abs=1e-12)
    ref = TS.cluster_bootstrap([0.50 - 0.20], [201506], 2000, 0, 0.05)     # one single row in one month
    assert s["single_r12"]["mean"] == pytest.approx(0.30, abs=1e-12) and s["single_r12"]["lo"] is None
    assert ref["lo"] is None
    wd = TS.cluster_bootstrap_diff([1.0, 0.0, 0.0], months, [0.0, 1.0, 0.0, 0.0],
                                   [201506, 201506, 201507, 201507], 2000, 0, 0.05)
    assert s["wipe12_cluster_minus_control"]["mean"] == pytest.approx(wd["mean"], abs=1e-12)
    assert s["wipe12_cluster_minus_control"]["lo"] == pytest.approx(wd["lo"], abs=1e-12)
    # reused symbols R1 and R9 removed from both arms: controls K1, K2 (June mean 0.2) and K3 (July 0.0)
    nr = TS.cluster_bootstrap([0.20 - 0.20, 0.10 - 0.0], [201506, 201507], 2000, 0, 0.05)
    assert s["no_reused_r12"]["mean"] == pytest.approx(nr["mean"], abs=1e-12) == pytest.approx(0.05, abs=1e-12)
    assert s["no_reused_r12"]["lo"] == pytest.approx(nr["lo"], abs=1e-12)


def test_long_arm_no_cluster_rows_does_not_raise():
    sm = pd.DataFrame([_lrow("K1", "2015-06-30", 0.10, "control"), _lrow("K2", "2015-07-31", 0.0, "control")])
    res = IS.long_arm(sm)
    assert res["primary"]["mean"] is None and res["primary"]["lo"] is None and res["primary"]["n"] == 0
    assert res["counts"]["n_cluster"] == 0 and res["counts"]["n_control"] == 2
    assert res["control_summary"]["n_matched"] == 0
    assert IS.verdict(res["primary"]["lo"], provisional=False) == "no demonstrated edge (interval not computable)"


def test_excess_same_month_empty_frames():
    tr = pd.DataFrame([_lrow("C1", "2015-06-30", 0.3, "cluster")])
    ct = pd.DataFrame([_lrow("K1", "2015-06-30", 0.1, "control")])
    assert IS.excess_same_month(tr.iloc[0:0], ct, "r12").shape == (0,)
    out = IS.excess_same_month(tr, ct.iloc[0:0], "r12")
    assert out.shape == (1,) and np.isnan(out[0])
    assert IS.excess_same_month(tr, ct, "r12")[0] == pytest.approx(0.2)
    nan_ct = ct.assign(r12=NAN)                                            # a NaN control is not a control
    assert np.isnan(IS.excess_same_month(tr, nan_ct, "r12")[0])


# -- short arm -------------------------------------------------------------------------------------

def _row(val, end, filed, start=None, form="10-Q", accn=None, fy=2020, fp="Q1"):
    r = {"end": end, "val": val, "accn": accn or f"acc-{filed}-{end}-{start}", "fy": fy, "fp": fp,
         "form": form, "filed": filed}
    if start:
        r["start"] = start
    return r


def _facts(gaap=None, dei=None, units=None):
    """companyfacts JSON. gaap/dei map concept -> rows; units maps concept -> unit (default USD / shares)."""
    units = units or {}

    def block(d, default_unit):
        return {c: {"label": c, "description": "", "units": {units.get(c, default_unit): rows}}
                for c, rows in (d or {}).items()}

    return {"cik": 1234, "entityName": "Synthetic Test Co",
            "facts": {"us-gaap": block(gaap, "USD"), "dei": block(dei, "shares")}}


def _split(facts):
    t = E.facts_table(facts)
    return t[t["unit"] == "USD"], t[t["unit"] == "shares"]


def facts_1234():
    return _facts(
        gaap={E.OCF: [_row(-1_000_000, "2014-12-31", "2015-03-01", start="2014-01-01", form="10-K", accn="o-1"),
                      _row(50_000, "2015-03-31", "2015-04-15", start="2015-01-01", accn="o-2")],
              E.CASH: [_row(999_999, "2014-12-31", "2015-03-01", form="10-K", accn="c-1"),
                       _row(1_000_000, "2015-03-31", "2015-04-15", accn="c-2")]},
        dei={E.SHARES_DEI: [_row(1_000_000, "2014-03-31", "2014-04-15", accn="s-1"),
                            _row(1_200_000, "2015-03-31", "2015-04-15", accn="s-2")]})


def facts_5678():
    return _facts(
        gaap={E.OCF: [_row(-2_000_000, "2014-12-31", "2015-03-01", start="2014-01-01", form="10-K", accn="o-1")],
              E.CASH: [_row(500_000, "2015-03-31", "2015-04-15", accn="c-1")]},
        dei={E.SHARES_DEI: [_row(1_000_000, "2014-03-31", "2014-04-15", accn="s-1"),
                            _row(1_500_000, "2015-03-31", "2015-04-15", accn="s-2")]})


def facts_7777():
    return _facts(dei={E.SHARES_DEI: [_row(1_000_000, "2014-03-31", "2014-04-15", accn="s-1"),
                                      _row(2_000_000, "2015-03-31", "2015-04-30", accn="s-2")]})


def test_short_flags_hand_computed():
    fl = IS.short_flags(facts_1234(), [ts("2015-03-31"), ts("2015-04-30")])
    assert list(fl.columns) == ["month_end", "pit_date", "dilution", "burn", "short_class"]
    assert fl["pit_date"].tolist() == [ts("2015-03-30"), ts("2015-04-29")]
    # 2015-03-31: the 2015-03-31 share count was filed 2015-04-15, so there is no count to compare
    assert fl["dilution"].iloc[0] is None
    assert fl["burn"].iloc[0] is True                                     # cash 999,999 / 1,000,000 < 1
    assert fl["short_class"].iloc[0] == "unknown"
    # 2015-04-30: ratio exactly 1.2; runway exactly 1.0; the quarterly +50,000 OCF row is ignored
    assert fl["dilution"].iloc[1] is True
    assert fl["burn"].iloc[1] is False
    assert fl["short_class"].iloc[1] == "unflagged"
    f5 = IS.short_flags(facts_5678(), [ts("2015-04-30")])
    assert (f5["dilution"].iloc[0], f5["burn"].iloc[0], f5["short_class"].iloc[0]) == (True, True, "flagged")
    none = IS.short_flags(None, [ts("2015-03-31"), ts("2015-04-30")])
    assert none["short_class"].tolist() == ["unknown", "unknown"]
    assert none["dilution"].tolist() == [None, None] and none["burn"].tolist() == [None, None]


def test_short_flags_keeps_input_order_and_series_index():
    me = pd.Series([ts("2015-04-30"), ts("2015-03-31")], index=[10, 20])
    fl = IS.short_flags(facts_1234(), me)
    assert fl.index.tolist() == [10, 20]
    assert fl["short_class"].tolist() == ["unflagged", "unknown"]


def test_dilution_boundary_exactly_20pct():
    d = ts("2015-04-29")
    assert IS.dilution_flag(_split(_facts(dei={E.SHARES_DEI: [
        _row(1_000_000, "2014-03-31", "2014-04-15"), _row(1_200_000, "2015-03-31", "2015-04-15")]}))[1], d) is True
    assert IS.dilution_flag(_split(_facts(dei={E.SHARES_DEI: [
        _row(1_000_000, "2014-03-31", "2014-04-15"), _row(1_199_999, "2015-03-31", "2015-04-15")]}))[1], d) is False
    assert 1_200_000 / 1_000_000 - 1 < 0.20                                # the trap the ratio form avoids


def test_dilution_uses_one_concept_and_rejects_bad_then():
    d = ts("2015-04-29")
    # dei has only the NOW count (5M); us-gaap has both (1.0M then, 1.1M now). Mixing would give 5.0x.
    mixed = _facts(gaap={E.SHARES_GAAP: [_row(1_000_000, "2014-03-31", "2014-04-15"),
                                         _row(1_100_000, "2015-03-31", "2015-04-15")]},
                   dei={E.SHARES_DEI: [_row(5_000_000, "2015-03-31", "2015-04-15")]},
                   units={E.SHARES_GAAP: "shares"})
    assert IS.dilution_flag(_split(mixed)[1], d) is False
    zero_then = _facts(dei={E.SHARES_DEI: [_row(0, "2014-03-31", "2014-04-15"),
                                           _row(1_000, "2015-03-31", "2015-04-15")]})
    assert IS.dilution_flag(_split(zero_then)[1], d) is None
    only_now = _facts(dei={E.SHARES_DEI: [_row(1_000_000, "2015-03-31", "2015-04-15")]})
    assert IS.dilution_flag(_split(only_now)[1], d) is None


def test_filing_on_last_weekday_not_visible():
    fl = IS.short_flags(facts_7777(), [ts("2015-04-30"), ts("2015-05-31")])
    # 2015-04-30: pit 2015-04-29, the 2.0M count was filed 2015-04-30 and is invisible, so the ratio is 1.0
    assert fl["dilution"].iloc[0] is False
    # 2015-05-31 (Sunday): pit 2015-05-28, the count is public
    assert fl["dilution"].iloc[1] is True
    assert fl["short_class"].tolist() == ["unknown", "unknown"]            # no cash-flow data: burn is unknown


def test_stale_value_is_missing():
    fl = IS.short_flags(facts_1234(), [ts("2016-12-31")])                  # pit 2016-12-29
    assert fl["dilution"].iloc[0] is None                                  # latest period end 2015-03-31 is 639 days old
    assert fl["short_class"].iloc[0] == "unknown"
    shr = _split(facts_1234())[1]
    # a period end exactly 550 days before the date is still usable (at that date the count a year earlier is
    # the same row, so the ratio is 1.0 and the answer is False), 551 days is stale and the answer is None
    assert IS.dilution_flag(shr, ts("2015-03-31") + pd.Timedelta(days=550)) is False
    assert IS.dilution_flag(shr, ts("2015-03-31") + pd.Timedelta(days=551)) is None


def test_annual_ocf_picks_annual_row_and_no_restatement_look_ahead():
    gaap = {E.OCF: [
        _row(-500_000, "2013-12-31", "2014-03-01", start="2013-01-01", form="10-K", accn="a"),
        _row(-1_000_000, "2014-12-31", "2015-03-01", start="2014-01-01", form="10-K", accn="b"),
        _row(-900_000, "2014-12-31", "2015-06-01", start="2014-01-01", form="10-K/A", accn="c"),   # restated later
        _row(-700_000, "2015-09-30", "2015-11-01", start="2015-01-01", accn="d")]}                  # 9M: not annual
    usd = _split(_facts(gaap=gaap))[0]
    assert IS.annual_ocf(usd, ts("2015-02-15")) == (ts("2013-12-31"), -500_000.0)     # FY2014 not yet filed
    assert IS.annual_ocf(usd, ts("2015-03-01")) == (ts("2014-12-31"), -1_000_000.0)   # filed on the date: visible
    assert IS.annual_ocf(usd, ts("2015-04-29")) == (ts("2014-12-31"), -1_000_000.0)   # the restatement is invisible
    assert IS.annual_ocf(usd, ts("2015-12-01")) == (ts("2014-12-31"), -900_000.0)     # visible once filed
    assert IS.annual_ocf(usd, ts("2013-12-31")) == (None, pytest.approx(NAN, nan_ok=True))
    # staleness: a period end 550 days before the date is the last usable one
    stale_usd = _split(_facts(gaap={E.OCF: [_row(-1.0, "2014-12-31", "2015-03-01", start="2014-01-01")]}))[0]
    end = ts("2014-12-31")
    assert IS.annual_ocf(stale_usd, end + pd.Timedelta(days=550))[0] == end            # 550 days: usable
    assert IS.annual_ocf(stale_usd, end + pd.Timedelta(days=551))[0] is None           # 551 days: stale


def _usd(ocf=None, cash=None, ocf_start="2014-01-01", ocf_end="2014-12-31", extra_cash=None):
    gaap = {}
    if ocf is not None:
        gaap[E.OCF] = [_row(ocf, ocf_end, "2015-03-01", start=ocf_start, form="10-K")]
    cash_rows = []
    if cash is not None:
        cash_rows.append(_row(cash, "2015-03-31", "2015-04-15"))
    if cash_rows:
        gaap[E.CASH] = cash_rows
    if extra_cash is not None:
        gaap[E.CASH_ADD[0]] = [_row(extra_cash, "2015-03-31", "2015-04-15")]
    return _split(_facts(gaap=gaap))[0]


def test_runway_boundary_exactly_one_year():
    d = ts("2015-04-29")
    assert IS.burn_flag(_usd(ocf=-1e6, cash=999_999), d) is True
    assert IS.burn_flag(_usd(ocf=-1e6, cash=1_000_000), d) is False        # exactly one year is not "under one year"
    assert IS.burn_flag(_usd(ocf=0, cash=1.0), d) is False
    assert IS.burn_flag(_usd(ocf=5, cash=1.0), d) is False
    assert IS.burn_flag(_usd(ocf=5), d) is False                           # OCF >= 0 is known, whatever the cash
    assert IS.burn_flag(_usd(cash=1.0), d) is None                         # no OCF
    assert IS.burn_flag(_usd(ocf=-1e6), d) is None                         # no cash
    # only a 90-day OCF row: no annual row, so nothing is known
    assert IS.burn_flag(_usd(ocf=-1e6, cash=1.0, ocf_start="2015-01-01", ocf_end="2015-03-31"), d) is None
    # the first available cash add-on is part of the cash total
    assert IS.burn_flag(_usd(ocf=-1e6, cash=600_000, extra_cash=400_000), d) is False
    assert IS.burn_flag(_usd(ocf=-1e6, cash=600_000, extra_cash=399_999), d) is True
    # a stale annual OCF (period end more than 550 days before the date) is missing
    assert IS.burn_flag(_usd(ocf=-1e6, cash=1.0), ts("2014-12-31") + pd.Timedelta(days=551)) is None
    # a stale cash figure (period end 668 days before the date) with a fresh negative OCF is missing too
    old_cash = _split(_facts(gaap={
        E.OCF: [_row(-1e6, "2014-12-31", "2015-03-01", start="2014-01-01", form="10-K")],
        E.CASH: [_row(1.0, "2013-06-30", "2013-08-01")]}))[0]
    assert IS.burn_flag(old_cash, d) is None


def test_missing_data_never_matches():
    cases = [((True, True), "flagged"), ((True, False), "unflagged"), ((False, False), "unflagged"),
             ((False, True), "unflagged"), ((True, None), "unknown"), ((False, None), "unknown"),
             ((None, True), "unknown"), ((None, None), "unknown"), ((NAN, True), "unknown"),
             ((True, pd.NA), "unknown")]
    for (dil, burn), want in cases:
        assert IS.classify_short(dil, burn) == want, (dil, burn)


def _srow(symbol, me, cls, wipe12):
    return {"symbol": symbol, "month": P(me[:7]), "month_end": ts(me), "short_class": cls, "wipe12": wipe12}


def test_short_arm_hand_computed():
    sm = pd.DataFrame([
        _srow("X1", "2015-04-30", "flagged", 1.0), _srow("X2", "2015-05-31", "flagged", 0.0),
        _srow("Y1", "2015-04-30", "unflagged", 0.0), _srow("Y2", "2015-04-30", "unflagged", 0.0),
        _srow("Y3", "2015-05-31", "unflagged", 1.0), _srow("Y4", "2015-05-31", "unflagged", 0.0),
        _srow("Z1", "2015-05-31", "unknown", 1.0)])
    res = IS.short_arm(sm)
    ref = TS.cluster_bootstrap_diff([1, 0], [201504, 201505], [0, 0, 1, 0],
                                    [201504, 201504, 201505, 201505], 2000, 0, 0.025)
    p = res["primary"]
    assert p["mean"] == pytest.approx(0.25)                                # 0.5 - 0.25
    assert p["n_a"] == 2 and p["n_b"] == 4
    assert p["lo"] == pytest.approx(ref["lo"], abs=1e-12) and p["hi"] == pytest.approx(ref["hi"], abs=1e-12)
    assert p["alpha"] == 0.025
    assert res["counts"]["n_unknown"] == 1 and res["counts"]["n_flagged"] == 2 and res["counts"]["n_unflagged"] == 4
    assert res["rates"] == {"flagged": 0.5, "unflagged": 0.25}
    rob = TS.cluster_bootstrap_diff([1, 0], ["X1", "X2"], [0, 0, 1, 0], ["Y1", "Y2", "Y3", "Y4"], 2000, 0, 0.025)
    assert res["robust_stock_clusters"]["lo"] == pytest.approx(rob["lo"], abs=1e-12)
    assert res["robust_stock_clusters"]["hi"] == pytest.approx(rob["hi"], abs=1e-12)


def test_short_arm_counts_coverage_and_no_outcome():
    sm = pd.DataFrame([
        _srow("A", "2014-12-31", "flagged", 1.0), _srow("B", "2015-01-31", "unflagged", 0.0),
        _srow("C", "2015-02-28", "unflagged", NAN), _srow("D", "2015-02-28", "unknown", 1.0)])
    res = IS.short_arm(sm)
    assert res["counts"] == {"n_flagged": 1, "n_unflagged": 1, "n_unknown": 1, "n_no_outcome": 1}
    assert res["coverage_by_year"] == {2014: 1, 2015: 2}
    assert res["first_classified_month"] == "2014-12"
    assert res["primary"]["n_a"] == 1 and res["primary"]["n_b"] == 1
    none = IS.short_arm(sm.assign(short_class="unknown"))
    assert none["primary"]["mean"] is None and none["first_classified_month"] is None
    assert none["coverage_by_year"] == {} and none["rates"] == {"flagged": None, "unflagged": None}


# -- verdict ---------------------------------------------------------------------------------------

def test_verdict_boundary_zero_is_not_pass():
    assert IS.verdict(0.0, provisional=False) == "no demonstrated edge"
    assert IS.verdict(-0.01, provisional=False) == "no demonstrated edge"
    assert IS.verdict(1e-12, provisional=False) == "PASS"
    assert IS.verdict(None, provisional=False) == "no demonstrated edge (interval not computable)"
    assert IS.verdict(NAN, provisional=False) == "no demonstrated edge (interval not computable)"
    assert IS.verdict(0.01, provisional=True) == "PROVISIONAL: survivors only (delisted names missing) | PASS"
    assert IS.verdict(0.0, provisional=True) == "PROVISIONAL: survivors only (delisted names missing) | no demonstrated edge"
    assert IS.verdict(None, provisional=False, not_run_reason="no companyfacts directory") == \
        "not run: no companyfacts directory"
    assert IS.verdict(None, provisional=True, not_run_reason="no companyfacts directory") == \
        "PROVISIONAL: survivors only (delisted names missing) | not run: no companyfacts directory"


# -- hygiene ---------------------------------------------------------------------------------------

def test_new_files_have_no_em_dash_and_no_email_and_no_sec_insider_import():
    for rel in ("setup_engine/insider_study.py", "tests/test_study/test_insider_study.py"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert chr(0x2014) not in text, rel
        assert re.findall(r"[\w.+-]+@[\w-]+\.[\w.]+", text) == [], rel
    src = (ROOT / "setup_engine/insider_study.py").read_text(encoding="utf-8")
    imported = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
            imported |= {f"{node.module}.{a.name}" for a in node.names}
    assert not [m for m in imported if m.startswith("data_engine")], imported      # database layer, yfinance
    assert not [m for m in imported if "multibagger" in m or "sec_insider" in m], imported
    assert src.count("spec_from_file_location") == 1                              # only sec_edgar is loaded by path
