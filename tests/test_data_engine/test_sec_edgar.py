"""data_engine.sec_edgar: a fundamentals snapshot must only contain what was public on the date asked.

The failure being guarded against: a back-dated read that quietly uses a restatement filed a year
later (look-ahead), a trailing-twelve-month figure built from the wrong quarters (a year-to-date
number mistaken for a quarter, or quarters with a hole in them), a runway computed from a stale or
mis-signed cash burn, or a client that hammers the SEC, ignores its cache, or sends no identity.
All data below is SYNTHETIC ("Synthetic Test Co", CIK 1234) and nothing here touches the network.
"""
import ast
import gzip
import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_engine import sec_edgar as E

ROOT = Path(__file__).resolve().parents[2]
UA = "Unit Test unit@example.invalid"
CASH, OCF = E.CASH, E.OCF


# -- synthetic companyfacts -------------------------------------------------------------------

def _row(val, end, filed, start=None, form="10-Q", accn=None, frame=None, fy=2020, fp="Q1"):
    r = {"end": end, "val": val, "accn": accn or f"acc-{filed}-{end}-{start}", "fy": fy, "fp": fp,
         "form": form, "filed": filed}
    if start:
        r["start"] = start
    if frame:
        r["frame"] = frame
    return r


def _facts(gaap=None, dei=None, units=None):
    """companyfacts JSON. gaap/dei map concept -> rows; units maps concept -> unit (default USD / shares)."""
    units = units or {}

    def block(d, default_unit):
        return {c: {"label": c, "description": "", "units": {units.get(c, default_unit): rows}}
                for c, rows in (d or {}).items()}

    return {"cik": 1234, "entityName": "Synthetic Test Co",
            "facts": {"us-gaap": block(gaap, "USD"), "dei": block(dei, "shares")}}


def _ytd(vals, year=2020, filed=None):
    """Year-to-date rows for a fiscal year starting Jan 1: vals = [3M, 6M, 9M, 12M] (None skips a row)."""
    ends = [f"{year}-03-31", f"{year}-06-30", f"{year}-09-30", f"{year}-12-31"]
    filed = filed or [f"{year}-05-01", f"{year}-08-01", f"{year}-11-01", f"{year + 1}-02-15"]
    forms = ["10-Q", "10-Q", "10-Q", "10-K"]
    return [_row(v, e, f, start=f"{year}-01-01", form=fm, accn=f"acc-{year}-{i}")
            for i, (v, e, f, fm) in enumerate(zip(vals, ends, filed, forms)) if v is not None]


def _table(gaap=None, dei=None, units=None):
    return E.facts_table(_facts(gaap, dei, units))


def _ts(s):
    return pd.Timestamp(s)


class FakeClock:
    """Monotonic clock whose sleep() advances it, so throttling is testable without waiting."""

    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def _client(tmp_path, fetch, clock=None, wall=None):
    clock = clock or FakeClock()
    return E.EdgarClient(cache_dir=tmp_path / "sec", fetch=fetch, clock=clock, wall=wall or (lambda: 0.0),
                         sleep=clock.sleep)


@pytest.fixture
def ua(monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", UA)
    return UA


# -- point in time ----------------------------------------------------------------------------

def _restated_cash():
    return _table({CASH: [
        _row(100, "2020-03-31", "2020-05-01", accn="acc-A"),
        _row(90, "2020-03-31", "2021-05-01", accn="acc-B", frame="CY2020Q1I"),
    ]})


@pytest.mark.parametrize("date,expected", [
    ("2020-06-01", 100.0),
    ("2021-04-30", 100.0),     # the day before the restatement was filed
    ("2021-05-01", 90.0),      # the filing day itself is visible
    ("2021-06-01", 90.0),
])
def test_a_restatement_is_invisible_until_the_day_it_is_filed(date, expected):
    """Look-ahead: using a restated value on a date before it was filed."""
    end, val = E.as_of(_restated_cash(), date, CASH)
    assert end == _ts("2020-03-31") and val == expected


def test_a_value_is_invisible_before_it_is_filed():
    """Look-ahead: a figure appearing in a read dated before its first filing."""
    end, val = E.as_of(_restated_cash(), "2020-04-30", CASH)
    assert end is None and np.isnan(val)


@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_same_filing_date_takes_the_greater_accession_number(order):
    """Two copies filed the same day must resolve deterministically, whatever the input order."""
    rows = [_row(5, "2020-03-31", "2020-05-01", accn="0001-A"), _row(7, "2020-03-31", "2020-05-01", accn="0001-B")]
    t = _table({CASH: [rows[i] for i in order]})
    assert E.as_of(t, "2020-06-01", CASH)[1] == 7.0


def test_the_latest_period_known_wins_and_periods_are_kept_separately():
    """as_of must return the newest period end, while pit keeps one row per period (not one overall)."""
    t = _table({CASH: [_row(1, "2019-12-31", "2020-02-01"), _row(2, "2020-03-31", "2020-05-01"),
                       _row(3, "2020-06-30", "2020-08-01")]})
    assert E.as_of(t, "2020-06-01", CASH) == (_ts("2020-03-31"), 2.0)
    assert list(E.pit(t, "2020-06-01", CASH)["val"]) == [1.0, 2.0]


def test_duration_lookup_takes_the_longest_period_on_a_tie():
    """Several durations end on the same date (3M and 12M at year end); the full-year one is wanted."""
    t = _table({OCF: [_row(25, "2020-12-31", "2021-02-15", start="2020-10-01"),
                      _row(70, "2020-12-31", "2021-02-15", start="2020-01-01")]})
    assert E.as_of(t, "2021-03-01", OCF, kind="duration") == (_ts("2020-12-31"), 70.0)
    assert E.as_of(t, "2021-03-01", OCF, kind="instant")[0] is None
    with pytest.raises(ValueError):
        E.as_of(t, "2021-03-01", OCF, kind="bogus")


# -- facts_table ------------------------------------------------------------------------------

def test_facts_table_has_the_documented_columns_and_no_frame():
    """The `frame` field reflects later restatements; it must never reach the table."""
    t = _table({CASH: [_row(90, "2020-03-31", "2021-05-01", frame="CY2020Q1I"),
                       _row(5, "2020-01-01", "2020-05-01", start="2020-01-01")]})
    assert list(t.columns) == ["concept", "taxonomy", "unit", "start", "end", "fy", "fp", "form", "filed", "val", "accn"]
    assert "frame" not in t.columns
    assert t["start"].isna().sum() == 1 and t["start"].notna().sum() == 1
    for c in ("start", "end", "filed"):
        assert str(t[c].dtype).startswith("datetime64")


def test_facts_table_reads_only_us_gaap_and_dei_and_honours_the_concept_filter():
    """IFRS and other taxonomies are out of scope; a concept filter must really filter."""
    f = _facts({CASH: [_row(1, "2020-03-31", "2020-05-01")], "Other": [_row(2, "2020-03-31", "2020-05-01")]},
               {E.SHARES_DEI: [_row(3, "2020-03-31", "2020-05-01")]})
    f["facts"]["ifrs-full"] = {"Cash": {"units": {"USD": [_row(9, "2020-03-31", "2020-05-01")]}}}
    t = E.facts_table(f)
    assert sorted(t["concept"]) == sorted([CASH, "Other", E.SHARES_DEI])
    assert set(t["taxonomy"]) == {"us-gaap", "dei"}
    assert list(E.facts_table(f, concepts=[CASH])["concept"]) == [CASH]


def test_facts_table_of_nothing_is_an_empty_table_with_the_right_columns():
    """Empty or odd input must not raise, and downstream date comparisons must still work on it."""
    for facts in ({}, {"facts": {}}, {"facts": {"us-gaap": {}}}, _facts()):
        t = E.facts_table(facts)
        assert t.empty and "filed" in t.columns
        assert E.as_of(t, "2021-01-01", CASH)[0] is None
        assert E.quarterly(t, "2021-01-01", OCF).empty


def test_rows_without_a_value_or_filing_date_are_dropped():
    """A fact with no val/filed cannot be placed in time, so it must not become a NaN row."""
    good = _row(1, "2020-03-31", "2020-05-01")
    no_val = {k: v for k, v in _row(2, "2020-06-30", "2020-08-01").items() if k != "val"}
    no_filed = {k: v for k, v in _row(3, "2020-09-30", "2020-11-01").items() if k != "filed"}
    assert list(_table({CASH: [good, no_val, no_filed]})["val"]) == [1.0]


# -- quarterly and ttm ------------------------------------------------------------------------

def test_quarterly_differences_year_to_date_rows_into_quarters():
    """Cash-flow rows are year-to-date; Q2 = 6M-3M, Q3 = 9M-6M, Q4 = FY-9M (a wrong Q4 shows as 50, not 25)."""
    t = _table({OCF: _ytd([10, 25, 45, 70])})
    q = E.quarterly(t, "2021-03-01", OCF)
    assert list(q.index) == [_ts("2020-03-31"), _ts("2020-06-30"), _ts("2020-09-30"), _ts("2020-12-31")]
    assert list(q.values) == [10.0, 15.0, 20.0, 25.0]


def test_ttm_is_the_sum_of_four_consecutive_known_quarters_else_nan():
    """Trailing figures must not be built from a partial year."""
    t = _table({OCF: _ytd([10, 25, 45, 70])})
    assert E.ttm(t, "2021-03-01", OCF) == 70.0
    assert np.isnan(E.ttm(t, "2020-12-01", OCF))          # the 10-K is not filed yet: only three quarters known


def test_ttm_rolls_forward_across_fiscal_years():
    """The last four quarters can straddle two fiscal years (Q4 of one, Q1-Q3 of the next)."""
    t = _table({OCF: _ytd([10, 25, 45, 70], year=2020) + _ytd([12, 30, None, None], year=2021)})
    # known on 2021-09-01: Q3'20=20, Q4'20=25, Q1'21=12, Q2'21=18
    assert list(E.quarterly(t, "2021-09-01", OCF).values[-4:]) == [20.0, 25.0, 12.0, 18.0]
    assert E.ttm(t, "2021-09-01", OCF) == 75.0


def test_a_direct_three_month_row_beats_the_year_to_date_difference():
    """When a filer reports Q2 directly (16), it must be used instead of 6M-3M (15)."""
    rows = _ytd([10, 25, 45, 70]) + [_row(16, "2020-06-30", "2020-08-01", start="2020-04-01", accn="acc-q2")]
    t = _table({OCF: rows})
    q = E.quarterly(t, "2021-03-01", OCF)
    assert q[_ts("2020-06-30")] == 16.0
    assert q[_ts("2020-09-30")] == 20.0 and q[_ts("2020-12-31")] == 25.0   # the YTD differences are unaffected
    assert E.ttm(t, "2021-03-01", OCF) == 71.0


def test_a_missing_year_to_date_step_leaves_a_hole_not_a_wrong_quarter():
    """6M and 12M with no 9M row: FY-6M is a six-month figure and must not be called a quarter."""
    t = _table({OCF: _ytd([10, 25, None, 70])})
    q = E.quarterly(t, "2021-03-01", OCF)
    assert list(q.index) == [_ts("2020-03-31"), _ts("2020-06-30")]
    assert np.isnan(E.ttm(t, "2021-03-01", OCF))


def test_a_missing_first_quarter_leaves_no_second_quarter_difference():
    """6M with no 3M row cannot be split, so no Q2 appears (and ttm is nan rather than short)."""
    t = _table({OCF: _ytd([None, 25, 45, 70])})
    q = E.quarterly(t, "2021-03-01", OCF)
    assert list(q.index) == [_ts("2020-09-30"), _ts("2020-12-31")] and list(q.values) == [20.0, 25.0]
    assert np.isnan(E.ttm(t, "2021-03-01", OCF))


def test_a_hole_inside_the_last_four_quarters_gives_nan_but_an_older_hole_does_not():
    """The last four values must be four consecutive quarters; a skipped quarter older than that is irrelevant."""
    rows = [_row(10, "2020-03-31", "2020-05-01", start="2020-01-01"),
            _row(20, "2020-09-30", "2020-11-01", start="2020-07-01"),       # Q2'20 is missing: an old hole
            _row(30, "2020-12-31", "2021-02-15", start="2020-10-01"),
            _row(40, "2021-03-31", "2021-05-01", start="2021-01-01"),
            _row(50, "2021-06-30", "2021-08-01", start="2021-04-01")]
    t = _table({OCF: rows})
    assert len(E.quarterly(t, "2021-09-01", OCF)) == 5
    assert E.ttm(t, "2021-09-01", OCF) == 140.0              # 20 + 30 + 40 + 50: consecutive, the old hole is behind them
    rows += [_row(70, "2021-12-31", "2022-02-15", start="2021-10-01")]      # Q3'21 is missing: a hole inside the window
    assert np.isnan(E.ttm(_table({OCF: rows}), "2022-03-01", OCF))


@pytest.mark.parametrize("days,counted", [(79, False), (80, True), (91, True), (100, True), (101, False)])
def test_three_month_duration_bins_are_inclusive(days, counted):
    """52/53-week quarters (91 days) count; anything not roughly a quarter (79, 101 days) must not."""
    end = _ts("2020-06-30")
    t = _table({OCF: [_row(5, "2020-06-30", "2020-08-01", start=str((end - pd.Timedelta(days=days)).date()))]})
    assert (end in E.quarterly(t, "2020-09-01", OCF).index) is counted


@pytest.mark.parametrize("days,counted", [(169, False), (170, True), (190, True), (191, False)])
def test_six_month_duration_bins_are_inclusive(days, counted):
    """The 6M year-to-date bin edges (170-190 days) decide whether Q2 exists."""
    start = _ts("2020-01-01")
    end = start + pd.Timedelta(days=days)
    rows = [_row(10, "2020-03-31", "2020-05-01", start="2020-01-01"),
            _row(25, str(end.date()), "2020-08-01", start="2020-01-01")]
    q = E.quarterly(_table({OCF: rows}), "2020-12-01", OCF)
    assert (end in q.index) is counted


def test_odd_durations_are_ignored_and_never_become_quarters():
    """A fiscal-year change leaves a 150-day transition period: it must drop out, not be bucketed."""
    rows = _ytd([10, 25, 45, 70]) + [_row(99, "2021-05-31", "2021-07-01", start="2020-12-31")]
    q = E.quarterly(_table({OCF: rows}), "2021-09-01", OCF)
    assert _ts("2021-05-31") not in q.index and len(q) == 4


def test_differencing_uses_the_figures_known_on_the_date():
    """A 6M figure restated later must not leak into a Q2 computed for an earlier date."""
    rows = _ytd([10, 25, 45, 70]) + [_row(31, "2020-06-30", "2021-06-01", start="2020-01-01", accn="acc-restate")]
    t = _table({OCF: rows})
    assert E.quarterly(t, "2021-03-01", OCF)[_ts("2020-06-30")] == 15.0
    assert E.quarterly(t, "2021-06-01", OCF)[_ts("2020-06-30")] == 21.0


# -- snapshot: runway, cash, debt -------------------------------------------------------------

def _snap_facts(ocf_ytd, cash=1200, extra=None, dei=None):
    gaap = {OCF: _ytd(ocf_ytd)}
    if cash is not None:
        gaap[CASH] = [_row(cash, "2020-12-31", "2021-02-15", form="10-K")]
    gaap.update(extra or {})
    return _facts(gaap, dei)


def test_runway_is_cash_over_monthly_burn():
    """Cash 1200 and trailing operating cash flow -2400 -> burn 200 a month -> 6 months."""
    s = E.snapshot(1234, "2021-03-01", facts=_snap_facts([-600, -1200, -1800, -2400]))
    assert s["op_cash_flow_ttm"] == -2400.0
    assert s["burn_per_month"] == pytest.approx(200.0)
    assert s["runway_months"] == pytest.approx(6.0)
    assert s["cash_total"] == 1200.0 and s["period_end"] == _ts("2020-12-31") and s["as_of"] == _ts("2021-03-01")


def test_positive_operating_cash_flow_means_no_burn_and_infinite_runway():
    """A cash-generating company must not get a negative or zero-division runway."""
    s = E.snapshot(1234, "2021-03-01", facts=_snap_facts([100, 250, 450, 700]))
    assert s["burn_per_month"] == 0.0 and s["runway_months"] == float("inf")


def test_runway_is_nan_when_the_burn_or_the_cash_is_unknown():
    """No operating-cash-flow history -> unknown, not infinite; burning with no cash figure -> unknown."""
    no_ocf = E.snapshot(1234, "2021-03-01", facts=_facts({CASH: [_row(1200, "2020-12-31", "2021-02-15")]}))
    assert np.isnan(no_ocf["burn_per_month"]) and np.isnan(no_ocf["runway_months"])
    no_cash = E.snapshot(1234, "2021-03-01", facts=_snap_facts([-600, -1200, -1800, -2400], cash=None))
    assert no_cash["burn_per_month"] == pytest.approx(200.0)
    assert np.isnan(no_cash["cash_total"]) and np.isnan(no_cash["runway_months"]) and no_cash["period_end"] is None


def test_runway_uses_only_what_was_filed_by_the_date():
    """A restated annual cash flow filed later changes the runway only from its filing date on."""
    extra_rows = _ytd([-600, -1200, -1800, -2400]) + [
        _row(-1200, "2020-12-31", "2021-06-01", start="2020-01-01", form="10-K/A", accn="acc-restate")]
    f = _facts({OCF: extra_rows, CASH: [_row(1200, "2020-12-31", "2021-02-15")]})
    assert E.snapshot(1234, "2021-03-01", facts=f)["runway_months"] == pytest.approx(6.0)
    # restated: Q4 = -1200 - (-1800) = +600, ttm = -600 - 600 - 600 + 600 = -1200 -> burn 100 -> 12 months
    assert E.snapshot(1234, "2021-06-01", facts=f)["runway_months"] == pytest.approx(12.0)


def test_cash_total_adds_only_the_first_addon_with_the_same_period_end():
    """Cash 1000 + the first matching add-on (200). Later add-ons and other-period add-ons are not added."""
    extra = {"ShortTermInvestments": [_row(500, "2020-09-30", "2020-11-01")],                 # wrong period end
             "MarketableSecuritiesCurrent": [_row(200, "2020-12-31", "2021-02-15")],           # first usable
             "AvailableForSaleSecuritiesDebtSecuritiesCurrent": [_row(999, "2020-12-31", "2021-02-15")]}
    s = E.snapshot(1234, "2021-03-01", facts=_snap_facts([-1, -2, -3, -4], cash=1000, extra=extra))
    assert s["cash_total"] == 1200.0
    extra["ShortTermInvestments"] = [_row(300, "2020-12-31", "2021-02-15")]                    # now it IS the first
    assert E.snapshot(1234, "2021-03-01", facts=_snap_facts([-1, -2, -3, -4], cash=1000, extra=extra))["cash_total"] == 1300.0


def _debt(extra):
    rows = {k: [_row(v, "2020-12-31", "2021-02-15")] for k, v in extra.items()}
    return E.snapshot(1234, "2021-03-01", facts=_facts(rows))["debt"]


@pytest.mark.parametrize("parts,expected", [
    ({"LongTermDebtNoncurrent": 300, "DebtCurrent": 50}, 350.0),
    ({"LongTermDebtNoncurrent": 300, "LongTermDebtCurrent": 40, "ShortTermBorrowings": 10}, 350.0),
    ({"LongTermDebt": 400, "LongTermDebtCurrent": 100, "ShortTermBorrowings": 20}, 420.0),   # (400-100)+(100+20)
    ({"LongTermDebt": 400}, 400.0),
    ({"ShortTermBorrowings": 20}, 20.0),
    ({"LongTermDebtNoncurrent": 300}, 300.0),
])
def test_debt_is_assembled_from_the_best_available_parts(parts, expected):
    """Best-effort debt: noncurrent portion plus current portion, never double counting the current part."""
    assert _debt(parts) == expected


def test_debt_is_nan_when_no_debt_concept_is_reported():
    """No debt tags at all means unknown, not zero."""
    assert np.isnan(_debt({}))


def test_money_in_another_currency_is_ignored():
    """A cash figure in EUR must not be read as dollars."""
    f = _facts({CASH: [_row(1200, "2020-12-31", "2021-02-15")]}, units={CASH: "EUR"})
    assert np.isnan(E.snapshot(1234, "2021-03-01", facts=f)["cash_total"])


def test_missing_concepts_give_nan_fields_and_no_exception():
    """A company with no usable XBRL (or none yet filed) yields a row of NaNs."""
    for facts in ({}, {"facts": {}}, _facts(), _facts({"Unrelated": [_row(1, "2020-03-31", "2020-05-01")]})):
        s = E.snapshot(1234, "2021-03-01", facts=facts)
        for k in ("cash_total", "debt", "op_cash_flow_ttm", "burn_per_month", "runway_months", "shares_out",
                  "share_growth_yoy", "revenue_ttm", "rev_growth_yoy", "gross_margin_ttm"):
            assert np.isnan(s[k]), k
        assert s["period_end"] is None and s["revenue_concept"] is None and s["as_of"] == _ts("2021-03-01")


# -- snapshot: shares, revenue, margin --------------------------------------------------------

def _shares_rows():
    return [_row(100, "2019-03-31", "2019-05-01"), _row(110, "2020-03-31", "2020-05-01")]


def test_share_growth_compares_point_in_time_counts_a_year_apart():
    """110 shares now vs 100 a year earlier -> +10%."""
    s = E.snapshot(1234, "2020-06-01", facts=_facts(dei={E.SHARES_DEI: _shares_rows()}))
    assert s["shares_out"] == 110.0
    assert s["share_growth_yoy"] == pytest.approx(0.10)


def test_share_growth_ignores_a_prior_year_count_filed_after_the_prior_date():
    """The year-ago count must be the one known a year ago, not a later-filed copy of it."""
    rows = _shares_rows() + [_row(104, "2019-03-31", "2019-12-01", accn="acc-late")]   # filed after 2019-06-01
    s = E.snapshot(1234, "2020-06-01", facts=_facts(dei={E.SHARES_DEI: rows}))
    assert s["share_growth_yoy"] == pytest.approx(0.10)


def test_dei_shares_are_preferred_over_us_gaap_shares():
    """The cover-page count (dei) wins when both exist; us-gaap is only the fallback."""
    gaap = {E.SHARES_GAAP: [_row(105, "2020-03-31", "2020-05-01")]}
    both = E.snapshot(1234, "2020-06-01", facts=_facts(gaap, {E.SHARES_DEI: _shares_rows()},
                                                       units={E.SHARES_GAAP: "shares"}))
    assert both["shares_out"] == 110.0
    only_gaap = E.snapshot(1234, "2020-06-01", facts=_facts(gaap, units={E.SHARES_GAAP: "shares"}))
    assert only_gaap["shares_out"] == 105.0 and np.isnan(only_gaap["share_growth_yoy"])


def _rev_facts(extra=None, rev_concept="Revenues"):
    gaap = {rev_concept: _ytd([100, 220, 360, 520], 2020) + _ytd([90, 190, 300, 420], 2019)}
    gaap.update(extra or {})
    return _facts(gaap)


def test_revenue_growth_and_gross_margin_from_gross_profit():
    """Revenue ttm 520 vs 420 a year earlier (+23.8%); gross profit ttm 208 -> margin 40%."""
    f = _rev_facts({E.GROSS: _ytd([40, 90, 150, 208], 2020)})
    s = E.snapshot(1234, "2021-03-01", facts=f)
    assert s["revenue_concept"] == "Revenues" and s["revenue_ttm"] == 520.0
    assert s["rev_growth_yoy"] == pytest.approx(520 / 420 - 1)
    assert s["gross_margin_ttm"] == pytest.approx(0.4)


@pytest.mark.parametrize("cost_concept", ["CostOfRevenue", "CostOfGoodsAndServicesSold"])
def test_gross_margin_falls_back_to_revenue_minus_cost(cost_concept):
    """With no GrossProfit, margin is (revenue - cost)/revenue = (520 - 312)/520 = 40%."""
    f = _rev_facts({cost_concept: _ytd([60, 130, 210, 312], 2020)})
    assert E.snapshot(1234, "2021-03-01", facts=f)["gross_margin_ttm"] == pytest.approx(0.4)


def test_revenue_concept_is_the_one_with_the_most_recent_period():
    """A retired tag (SalesRevenueNet, last period 2018) must lose to the tag still in use."""
    f = _rev_facts({"SalesRevenueNet": _ytd([1, 2, 3, 4], 2018)})
    assert E.snapshot(1234, "2021-03-01", facts=f)["revenue_concept"] == "Revenues"


def test_revenue_concept_tie_goes_to_the_earlier_listed_concept():
    """Equal latest period: the preferred concept (ASC 606 tag) is chosen, deterministically."""
    f = _rev_facts({E.REVENUE[0]: _ytd([7, 14, 21, 28], 2020)})
    assert E.snapshot(1234, "2021-03-01", facts=f)["revenue_concept"] == E.REVENUE[0]


def test_revenue_growth_is_nan_when_the_chosen_concept_lacks_the_prior_year():
    """Never mix two concepts across the year-over-year gap: the prior year is under another tag -> nan."""
    f = _facts({"Revenues": _ytd([100, 220, 360, 520], 2020), "SalesRevenueNet": _ytd([90, 190, 300, 420], 2019)})
    s = E.snapshot(1234, "2021-03-01", facts=f)
    assert s["revenue_concept"] == "Revenues" and s["revenue_ttm"] == 520.0
    assert np.isnan(s["rev_growth_yoy"])


# -- user agent -------------------------------------------------------------------------------

def test_user_agent_unset_raises_with_no_default(monkeypatch):
    """No contact -> no request. A baked-in default would send someone else's identity to the SEC."""
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    with pytest.raises(RuntimeError, match="SEC_USER_AGENT"):
        E.user_agent()


@pytest.mark.parametrize("value", ["", "   ", "NoAtSign"])
def test_user_agent_without_a_contact_raises(monkeypatch, value):
    """The SEC returns 403 without a contact address; fail early with the fix in the message."""
    monkeypatch.setenv("SEC_USER_AGENT", value)
    with pytest.raises(RuntimeError, match="contact@example.org"):
        E.user_agent()


def test_user_agent_with_a_contact_is_returned_unchanged(monkeypatch):
    """A valid value passes through untouched."""
    monkeypatch.setenv("SEC_USER_AGENT", UA)
    assert E.user_agent() == UA


def test_the_client_refuses_to_make_a_request_without_a_user_agent(tmp_path, monkeypatch):
    """Without SEC_USER_AGENT the fetch function must never be called."""
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)

    def fetch(url, headers):
        raise AssertionError("a request was attempted without a User-Agent")

    with pytest.raises(RuntimeError, match="SEC_USER_AGENT"):
        _client(tmp_path, fetch).get_json("https://x.invalid/a", "facts", "k")


def test_no_address_is_hard_coded_in_the_module_source():
    """The only address-shaped string allowed in the module is the RFC 2606 example in the error message."""
    src = (ROOT / "data_engine" / "sec_edgar.py").read_text(encoding="utf-8")
    assert set(re.findall(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", src)) == {"contact@example.org"}


def test_the_module_imports_no_network_or_parquet_library_at_top_level():
    """Library modules import urllib.request (and friends) only inside the function that needs it."""
    tree = ast.parse((ROOT / "data_engine" / "sec_edgar.py").read_text(encoding="utf-8"))
    names = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
    assert not {n for n in names if n.split(".")[0] in {"urllib", "requests", "pyarrow", "yfinance"}}


# -- throttle ---------------------------------------------------------------------------------

def test_throttle_spaces_nine_requests_at_least_an_eighth_of_a_second_apart(tmp_path, ua):
    """The SEC limit is 10 requests a second; nine rapid calls must take at least a full second."""
    clock, starts = FakeClock(), []

    def fetch(url, headers):
        starts.append(clock())
        return 200, b"{}"

    c = _client(tmp_path, fetch, clock=clock)
    for i in range(9):
        c.get_json(f"https://x.invalid/{i}", "facts", f"k{i}")
    gaps = np.diff(starts)
    assert len(starts) == 9 and (gaps >= 0.125 - 1e-12).all()
    assert starts[-1] - starts[0] >= 1.0 - 1e-12


def test_throttle_does_not_sleep_for_a_first_call_or_for_calls_that_are_already_spaced():
    """Slow callers must not be delayed further."""
    clock = FakeClock()
    sleeps = []
    t = E.Throttle(8, clock=clock, sleep=lambda s: (sleeps.append(s), clock.sleep(s)))
    t.wait()
    clock.t += 1.0
    t.wait()
    clock.t += 5.0
    t.wait()
    assert sleeps == []
    t.wait()                                 # an immediate second call must wait the full gap
    assert sleeps == [pytest.approx(0.125)]


# -- cache and client -------------------------------------------------------------------------

def _counting_fetch(payload=None):
    calls = []

    def fetch(url, headers):
        calls.append((url, headers))
        return 200, json.dumps(payload if payload is not None else {"ok": True}).encode()

    fetch.calls = calls
    return fetch


class _Wall:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


@pytest.mark.parametrize("fn,age,refetched", [
    ("facts", 0, False), ("facts", 3600, False), ("facts", 86399, False), ("facts", 86400, True),
    ("facts", 86401, True),
    ("submissions", 21599, False), ("submissions", 21600, True), ("submissions", 21601, True),
])
def test_cache_entries_expire_after_their_ttl(tmp_path, ua, fn, age, refetched):
    """facts live a day and submissions six hours: stale data would hide new filings, a missing cache hammers the SEC."""
    fetch, wall = _counting_fetch(), _Wall()
    c = _client(tmp_path, fetch, wall=wall)
    call = (lambda: E.company_facts(1234, client=c)) if fn == "facts" else (lambda: E.submissions(1234, client=c))
    call()
    wall.t = age
    call()
    assert len(fetch.calls) == (2 if refetched else 1)


def test_cache_files_hold_the_fetch_time_and_the_data(tmp_path, ua):
    """The on-disk format is {"fetched_at", "data"} under cache_dir/kind/key.json."""
    fetch, wall = _counting_fetch({"a": 1}), _Wall()
    wall.t = 1234.5
    E.company_facts(1234, client=_client(tmp_path, fetch, wall=wall))
    blob = json.loads((tmp_path / "sec" / "facts" / "CIK0000001234.json").read_text())
    assert blob == {"fetched_at": 1234.5, "data": {"a": 1}}


def test_a_corrupt_cache_file_is_refetched_not_fatal(tmp_path, ua):
    """A half-written cache file must cost one request, not an exception."""
    fetch = _counting_fetch({"a": 1})
    p = tmp_path / "sec" / "facts"
    p.mkdir(parents=True)
    (p / "CIK0000001234.json").write_text("{not json")
    assert E.company_facts(1234, client=_client(tmp_path, fetch)) == {"a": 1}
    assert len(fetch.calls) == 1


def test_requests_carry_the_user_agent_and_encoding_headers_and_nothing_else(tmp_path, ua):
    """Exactly the identity header the SEC asks for, plus compression; no other headers leak."""
    fetch = _counting_fetch()
    E.company_facts(1234, client=_client(tmp_path, fetch))
    url, headers = fetch.calls[0]
    assert headers == {"User-Agent": UA, "Accept-Encoding": "gzip, deflate"}
    assert "CIK0000001234.json" in url and url.startswith("https://data.sec.gov/api/xbrl/companyfacts/")


def test_submissions_url_and_cache_key(tmp_path, ua):
    """Submissions use their own endpoint and cache directory."""
    fetch = _counting_fetch()
    E.submissions(1234, client=_client(tmp_path, fetch))
    assert fetch.calls[0][0] == "https://data.sec.gov/submissions/CIK0000001234.json"
    assert (tmp_path / "sec" / "submissions" / "CIK0000001234.json").exists()


def test_a_non_200_status_raises_and_is_not_cached(tmp_path, ua):
    """A 403/500 must surface (naming SEC_USER_AGENT for 403) and must not poison the cache."""
    calls = []

    def fetch(url, headers):
        calls.append(url)
        return (403 if len(calls) == 1 else 500), b"denied"

    c = _client(tmp_path, fetch)
    with pytest.raises(RuntimeError, match=r"403.*SEC_USER_AGENT"):
        E.company_facts(1234, client=c)
    with pytest.raises(RuntimeError, match="500"):
        E.company_facts(1234, client=c)
    assert len(calls) == 2 and not (tmp_path / "sec" / "facts" / "CIK0000001234.json").exists()


def test_an_unknown_cache_kind_is_rejected(tmp_path, ua):
    """A typo in `kind` must fail loudly instead of caching forever or never."""
    with pytest.raises(ValueError):
        _client(tmp_path, _counting_fetch()).get_json("https://x.invalid", "factz", "k")


def test_snapshot_through_a_client_fetches_the_facts_once(tmp_path, ua):
    """snapshot(cik, date) with no `facts` goes through the cached client."""
    fetch = _counting_fetch(_snap_facts([-600, -1200, -1800, -2400]))
    c = _client(tmp_path, fetch)
    s1 = E.snapshot(1234, "2021-03-01", client=c)
    E.snapshot("0000001234", "2020-12-01", client=c)
    assert s1["runway_months"] == pytest.approx(6.0) and len(fetch.calls) == 1


def test_default_fetch_gunzips_and_returns_http_errors_without_raising(monkeypatch):
    """The real fetch (with urlopen replaced, no network): gzip bodies are unzipped, HTTP errors become (code, body)."""
    import urllib.error
    import urllib.request

    seen = {}

    class Resp:
        status = 200
        headers = {"Content-Encoding": "gzip"}

        def read(self):
            return gzip.compress(b'{"a": 1}')

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def ok(req, timeout=None):
        seen["ua"] = req.get_header("User-agent")
        seen["url"] = req.full_url
        return Resp()

    monkeypatch.setattr(urllib.request, "urlopen", ok)
    status, body = E.default_fetch("https://data.invalid/x", {"User-Agent": UA})
    assert status == 200 and json.loads(body) == {"a": 1}
    assert seen == {"ua": UA, "url": "https://data.invalid/x"}

    def boom(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 403, "Forbidden", {}, io.BytesIO(b"nope"))

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    assert E.default_fetch("https://data.invalid/x", {"User-Agent": UA}) == (403, b"nope")


# -- identifiers ------------------------------------------------------------------------------

@pytest.mark.parametrize("x", [1234, "1234", "0001234", "CIK0000001234", 1234.0, np.int64(1234)])
def test_pad_cik_gives_ten_digits(x):
    """CIKs arrive as ints, floats (from a CSV with gaps) or strings; all must normalise the same way."""
    assert E.pad_cik(x) == "0000001234"


def test_facts_url_contains_the_padded_cik(tmp_path, ua):
    """The companyfacts endpoint needs the 10-digit CIK."""
    fetch = _counting_fetch()
    E.company_facts(1234, client=_client(tmp_path, fetch))
    assert fetch.calls[0][0].endswith("CIK0000001234.json")


TICKERS = {"0": {"cik_str": 111, "ticker": "AAA", "title": "Alpha Inc"},
           "1": {"cik_str": 222, "ticker": "BRK-B", "title": "Class B Co"}}


@pytest.mark.parametrize("ticker,expected", [("AAA", "0000000111"), ("aaa", "0000000111"), (" AAA ", "0000000111"),
                                             ("BRK.B", "0000000222"), ("BRK-B", "0000000222"), ("ZZZ", None)])
def test_ticker_to_cik_matches_case_and_share_class_separators(ticker, expected):
    """Dots and dashes in share-class tickers must both resolve; unknown tickers give None."""
    assert E.ticker_to_cik(ticker, table=TICKERS) == expected


def test_ticker_table_is_fetched_once_and_cached(tmp_path, ua):
    """Looking up many tickers must cost one request for the ticker table."""
    fetch = _counting_fetch(TICKERS)
    c = _client(tmp_path, fetch)
    assert E.ticker_to_cik("AAA", client=c) == "0000000111"
    assert E.ticker_to_cik("BRK.B", client=c) == "0000000222"
    assert len(fetch.calls) == 1 and fetch.calls[0][0] == E.TICKERS_URL


def test_cik_from_sp500_prefers_the_listed_cik_and_falls_back_to_the_ticker_table():
    """A CIK in the S&P table wins; blanks, None and NaN fall back to the ticker; unknown stays None."""
    df = pd.DataFrame({"symbol": ["AAA", "BRK-B", "AAA", "ZZZ"], "cik": ["0000000999", "", None, np.nan]},
                      index=[10, 11, 12, 13])
    out = E.cik_from_sp500(df, table=TICKERS)
    assert list(out.index) == [10, 11, 12, 13]
    assert list(out) == ["0000000999", "0000000222", "0000000111", None]
    assert list(E.cik_from_sp500(pd.DataFrame({"symbol": ["AAA"]}), table=TICKERS)) == ["0000000111"]


def test_cik_from_sp500_makes_no_request_when_every_cik_is_present():
    """The ticker table is only needed for the fallback."""
    def boom(url, headers):
        raise AssertionError("no request expected")

    df = pd.DataFrame({"symbol": ["AAA"], "cik": ["0000000111"]})
    client = E.EdgarClient(fetch=boom)
    assert list(E.cik_from_sp500(df, client=client)) == ["0000000111"]


# -- filings ----------------------------------------------------------------------------------

def _subm():
    rows = [("S-3", "2021-03-01"), ("424B5", "2020-03-01"), ("424B5", "2020-02-29"), ("S-3", "2021-03-02"),
            ("8-K", "2020-09-10"), ("4", "2021-01-05"), ("10-K", "2020-12-01"), ("S-3ASR", "2020-06-01"),
            ("424B3", "2020-07-01"), ("4", "2020-10-01")]
    return {"filings": {"recent": {
        "form": [r[0] for r in rows],
        "filingDate": [r[1] for r in rows],
        "accessionNumber": [f"acc-{i}" for i in range(len(rows))],
        "primaryDocument": [f"doc{i}.htm" for i in range(len(rows) - 1)],     # one short on purpose
    }}}


def test_recent_filings_window_is_inclusive_at_both_ends_and_filters_forms():
    """today=2021-03-01, 365 days -> [2020-03-01, 2021-03-01]; 02-29 and a future date are out; 10-K is not a default form."""
    f = E.recent_filings(1234, today="2021-03-01", subm=_subm())
    assert list(f.columns) == ["form", "filingDate", "accessionNumber", "primaryDocument"]
    assert sorted(f["form"]) == sorted(["S-3", "424B5", "8-K", "4", "4", "S-3ASR", "424B3"])
    assert f["filingDate"].min() == _ts("2020-03-01") and f["filingDate"].max() == _ts("2021-03-01")
    assert list(f["filingDate"]) == sorted(f["filingDate"], reverse=True)          # newest first
    top = f.iloc[0]
    assert (top["form"], top["accessionNumber"], top["primaryDocument"]) == ("S-3", "acc-0", "doc0.htm")


def test_recent_filings_form_and_day_arguments():
    """A custom form list and window length change what is returned."""
    only_8k = E.recent_filings(1234, forms=("8-K",), today="2021-03-01", subm=_subm())
    assert list(only_8k["form"]) == ["8-K"]
    today_only = E.recent_filings(1234, days=0, today="2021-03-01", subm=_subm())
    assert list(today_only["form"]) == ["S-3"]
    ten_k = E.recent_filings(1234, forms=("10-K",), today="2021-03-01", subm=_subm())
    assert list(ten_k["accessionNumber"]) == ["acc-6"] and ten_k["primaryDocument"].iloc[0] == "doc6.htm"


def test_recent_filings_pads_a_short_document_array_and_survives_empty_input():
    """A short column (the last filing has no document) becomes None; an empty submissions block gives an empty frame."""
    last = E.recent_filings(1234, forms=("4",), days=400, today="2021-03-01", subm=_subm())
    assert last["primaryDocument"].isna().sum() == 1 and "doc5.htm" in set(last["primaryDocument"])
    empty = E.recent_filings(1234, today="2021-03-01", subm={"filings": {"recent": {}}})
    assert empty.empty and list(empty.columns) == ["form", "filingDate", "accessionNumber", "primaryDocument"]
    assert E.recent_filings(1234, today="2021-03-01", subm={}).empty


def test_recent_filings_goes_through_the_client(tmp_path, ua):
    """With no `subm`, the submissions endpoint is fetched (and cached) through the client."""
    fetch = _counting_fetch(_subm())
    c = _client(tmp_path, fetch)
    f = E.recent_filings(1234, today="2021-03-01", client=c)
    E.recent_filings(1234, today="2021-03-01", client=c)
    assert len(f) == 7 and len(fetch.calls) == 1 and fetch.calls[0][0].endswith("submissions/CIK0000001234.json")


def test_dilution_flags_count_shelves_and_supplements_exactly():
    """In the fake window: S-3 + S-3ASR = 2 shelves, 424B5 + 424B3 = 2 supplements, 8-K and Form 4 are not counted."""
    flags = E.dilution_flags(E.recent_filings(1234, today="2021-03-01", subm=_subm()))
    assert flags == {"n_shelf": 2, "n_supplements": 2, "n_dilution_forms": 4, "potential_atm": True}


@pytest.mark.parametrize("forms,expected", [
    (["S-3"], (1, 0, False)),
    (["424B5"], (0, 1, False)),
    (["S-3ASR", "424B3"], (1, 1, True)),
    (["S-3", "424B5", "424B5", "8-K", "4"], (1, 2, True)),
    (["8-K", "4", "10-K"], (0, 0, False)),
    ([], (0, 0, False)),
])
def test_potential_atm_needs_both_a_shelf_and_a_supplement(forms, expected):
    """A shelf alone (or a supplement alone) is not the shelf-plus-supplement pattern of an at-the-market program."""
    flags = E.dilution_flags(pd.DataFrame({"form": forms}))
    assert (flags["n_shelf"], flags["n_supplements"], flags["potential_atm"]) == expected
    assert flags["n_dilution_forms"] == expected[0] + expected[1]


def test_dilution_flags_accepts_an_empty_frame_without_columns():
    """No filings at all must give zeros, not a KeyError."""
    assert E.dilution_flags(pd.DataFrame()) == {"n_shelf": 0, "n_supplements": 0, "n_dilution_forms": 0,
                                                "potential_atm": False}
