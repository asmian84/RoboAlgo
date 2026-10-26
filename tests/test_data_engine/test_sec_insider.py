"""data_engine.sec_insider: an insider purchase cluster must be dated when it became PUBLIC, by real insiders.

The failures being guarded against: a cluster dated at the trade instead of the filing (look-ahead of up
to two business days per trade), one joint Form 4 or one owner filing twice counted as a cluster, a ten
percent owner or an amended filing completing a cluster, a $4,995 buy passing a $5,000 floor (or $5,000.00
failing it), a 30-day window that is really 31 days, and a ticker mapped to a company using filings made
after the date asked. All data below is SYNTHETIC and nothing here touches the network: every test runs
with socket.connect patched to raise, and the download helper only ever sees an injected fake fetch.
"""
import ast
import importlib.util
import socket
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
UA = "Unit Test unit@example.invalid"


def _load_module():
    name = "roboalgo_sec_insider"
    spec = importlib.util.spec_from_file_location(name, ROOT / "data_engine" / "sec_insider.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod                           # before exec_module: Python 3.14 needs it for @dataclass
    spec.loader.exec_module(mod)
    return mod


SI = _load_module()


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network access attempted in a unit test")
    monkeypatch.setattr(socket.socket, "connect", boom)


# -- synthetic Form 345 quarter ---------------------------------------------------------------

def _acc(short):
    return f"0000000001-15-0000{short}"


# (acc, FILING_DATE, DOCUMENT_TYPE, ISSUERCIK, ISSUERTRADINGSYMBOL)
SUB_ROWS = [
    ("A1", "05-JAN-2015", "4", "1001", "AAA"), ("A2", "03-FEB-2015", "4", "1001", "AAA"),
    ("A3", "10-FEB-2015", "4", "1001", "AAA"),
    ("B1", "05-JAN-2015", "4", "1002", "BBB"), ("B2", "04-FEB-2015", "4", "1002", "BBB"),
    ("C1", "10-JAN-2015", "4", "1003", "CCC"), ("C2", "12-JAN-2015", "4", "1003", "CCC"),
    ("C3", "15-JAN-2015", "4", "1003", "CCC"),
    ("D1", "10-FEB-2015", "4", "1004", "DDD"), ("D2", "20-FEB-2015", "4/A", "1004", "DDD"),
    ("E1", "02-MAR-2015", "4", "1005", "EEE"), ("E2", "03-MAR-2015", "4", "1005", "EEE"),
    ("F1", "05-JAN-2015", "4", "1006", "FFF"), ("F2", "20-JAN-2015", "4", "1006", "FFF"),
    ("R1", "07-JAN-2015", "3", "2001", "RUSE"), ("R2", "09-MAR-2015", "4", "2002", "ruse"),
    ("N1", "09-MAR-2015", "4", "2003", "NONE"), ("K1", "11-MAR-2015", "4", "2004", "BRK.B"),
]
# (acc, RPTOWNERCIK, RPTOWNER_RELATIONSHIP)
OWN_ROWS = [
    ("A1", "9001", "Director"), ("A2", "9002", "Officer"), ("A3", "9003", "Director"),
    ("B1", "9011", "Director"), ("B2", "9012", "Director"),
    ("C1", "9021", "Director"), ("C2", "9022", "TenPercentOwner"), ("C3", "9023", "Officer"),
    ("D1", "9031", "Director"), ("D1", "9032", "Director"), ("D2", "9033", "Officer"),
    ("E1", "9041", "Director"), ("E2", "9042", "Director"),
    ("F1", "9051", "Director"), ("F2", "9051", "Director"),
    ("R1", "9061", "Director"), ("R2", "9062", "Officer"), ("N1", "9063", "Director"), ("K1", "9064", "Director"),
]
# (acc, TRANS_DATE, TRANS_CODE, TRANS_SHARES, TRANS_PRICEPERSHARE)
TRN_ROWS = [
    ("A1", "02-JAN-2015", "P", "600", "5.00"), ("A1", "02-JAN-2015", "P", "400", "5.00"),
    ("A2", "06-JAN-2015", "P", "2000", "3.00"), ("A3", "06-FEB-2015", "P", "1000", "7.00"),
    ("B1", "02-JAN-2015", "P", "1000", "10.00"), ("B2", "20-JAN-2015", "P", "1000", "10.00"),
    ("C1", "08-JAN-2015", "P", "999", "5.00"), ("C2", "09-JAN-2015", "P", "10000", "2.00"),
    ("C3", "13-JAN-2015", "P", "2000", "3.00"),
    ("D1", "09-FEB-2015", "P", "3000", "4.00"), ("D2", "18-FEB-2015", "P", "2000", "4.00"),
    ("E1", "27-FEB-2015", " p ", "1000", "6.00"), ("E2", "02-MAR-2015", "S", "1000", "10.00"),
    ("F1", "02-JAN-2015", "P", "1000", "6.00"), ("F2", "16-JAN-2015", "P", "1000", "6.00"),
    ("R2", "05-MAR-2015", "S", "500", "2.00"), ("K1", "10-MAR-2015", "A", "100", "0"),
]
# (issuer, owner, acc, filing_date, value, trade_date) exactly as the plan hand-computed them
EXPECTED_PURCHASES = [
    ("1001", "9001", "A1", "2015-01-05", 5000.0, "2015-01-02"),
    ("1001", "9002", "A2", "2015-02-03", 6000.0, "2015-01-06"),
    ("1001", "9003", "A3", "2015-02-10", 7000.0, "2015-02-06"),
    ("1002", "9011", "B1", "2015-01-05", 10000.0, "2015-01-02"),
    ("1002", "9012", "B2", "2015-02-04", 10000.0, "2015-01-20"),
    ("1003", "9023", "C3", "2015-01-15", 6000.0, "2015-01-13"),
    ("1004", "9031", "D1", "2015-02-10", 12000.0, "2015-02-09"),
    ("1004", "9032", "D1", "2015-02-10", 12000.0, "2015-02-09"),
    ("1005", "9041", "E1", "2015-03-02", 6000.0, "2015-02-27"),
    ("1006", "9051", "F1", "2015-01-05", 6000.0, "2015-01-02"),
    ("1006", "9051", "F2", "2015-01-20", 6000.0, "2015-01-16"),
]


def _zip(path, members):
    """members: {member name: (header list, rows)}; each table is written as tab-joined lines."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, (header, rows) in members.items():
            lines = ["\t".join(header)] + ["\t".join(str(c) for c in r) for r in rows]
            zf.writestr(name, "\n".join(lines) + "\n")
    return Path(path)


def _write(path, sub=None, own=None, trn=None, lower=False, drop_sym=False, drop_table=None, with_trans_date=True):
    sub = SUB_ROWS if sub is None else sub
    own = OWN_ROWS if own is None else own
    trn = TRN_ROWS if trn is None else trn
    sub_h = ["ACCESSION_NUMBER", "FILING_DATE", "DOCUMENT_TYPE", "ISSUERCIK", "ISSUERTRADINGSYMBOL", "XTRA"]
    sub_r = [[_acc(a), f, d, c, s, "x"] for a, f, d, c, s in sub]
    if drop_sym:
        sub_h = sub_h[:4] + sub_h[5:]
        sub_r = [r[:4] + r[5:] for r in sub_r]
    own_h = ["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNER_RELATIONSHIP", "XTRA"]
    own_r = [[_acc(a), c, r, "x"] for a, c, r in own]
    trn_h = ["ACCESSION_NUMBER", "TRANS_DATE", "TRANS_CODE", "TRANS_SHARES", "TRANS_PRICEPERSHARE", "XTRA"]
    trn_r = [[_acc(a), d, c, s, p, "x"] for a, d, c, s, p in trn]
    if not with_trans_date:
        trn_h = trn_h[:1] + trn_h[2:]
        trn_r = [r[:1] + r[2:] for r in trn_r]
    tables = {"SUBMISSION": (sub_h, sub_r), "REPORTINGOWNER": (own_h, own_r), "NONDERIV_TRANS": (trn_h, trn_r)}
    if drop_table:
        del tables[drop_table]
    members = {}
    for name, (h, r) in tables.items():
        if lower:
            members[f"2015q1/{name.lower()}.tsv"] = ([c.lower() for c in h], r)
        else:
            members[f"{name}.tsv"] = (h, r)
    return _zip(path, members)


@pytest.fixture(scope="module")
def loaded(tmp_path_factory):
    p = _write(tmp_path_factory.mktemp("q1") / "2015q1_form345.zip")
    return SI.load_quarter(p)


def _tuples(df):
    return [(r.issuer_cik, r.owner_cik, r.accession, r.filing_date.strftime("%Y-%m-%d"), r.value,
             r.trade_date.strftime("%Y-%m-%d") if pd.notna(r.trade_date) else None)
            for r in df.itertuples(index=False)]


def _expected(rows=EXPECTED_PURCHASES):
    return [(i.zfill(10), o.zfill(10), _acc(a), f, v, t) for i, o, a, f, v, t in rows]


def _owners(purchases, issuer):
    return set(purchases.loc[purchases["issuer_cik"] == issuer.zfill(10), "owner_cik"])


# -- purchases -------------------------------------------------------------------------------

def test_expected_purchases_exact(loaded):
    purchases, _, stats = loaded
    assert list(purchases.columns) == SI.PURCHASE_COLUMNS
    assert len(purchases) == 11
    assert _tuples(purchases) == _expected()
    assert (purchases.loc[purchases["issuer_cik"] == "0000001001", "symbol"] == "AAA").all()
    assert stats["file"] == "2015q1_form345.zip"
    assert stats["rows"] == {"SUBMISSION": 18, "REPORTINGOWNER": 19, "NONDERIV_TRANS": 17}
    assert stats["bad_dates"] == 0


def test_value_boundary_aggregated_5000(loaded):
    purchases, _, _ = loaded
    got = purchases.set_index("owner_cik")["value"]
    assert "0000009001" in got.index                   # exactly $5,000.00 must pass a ">= 5,000" floor
    assert got["0000009001"] == 5000.0                 # 600 x 5.00 + 400 x 5.00
    assert "0000009021" not in got.index               # 999 x 5.00 = 4,995.00


def test_ten_percent_owner_only_is_excluded(loaded):
    purchases, _, _ = loaded
    assert "0000009022" not in set(purchases["owner_cik"])      # $20,000 but a ten percent owner
    assert SI.clusters(purchases).query("issuer_cik == '0000001003'").empty
    assert SI.qualifying_role("Director") is True
    assert SI.qualifying_role("Officer") is True
    assert SI.qualifying_role("TenPercentOwner") is False
    assert SI.qualifying_role("Director, TenPercentOwner") is True
    assert SI.qualifying_role("Other") is False
    assert SI.qualifying_role("") is False
    assert SI.qualifying_role(None) is False
    assert SI.qualifying_role("TenPercentOwner; Officer") is True
    assert SI.qualifying_role("Officer/Director") is True


def test_event_date_is_filing_date_not_trade_date(loaded):
    purchases, _, _ = loaded
    ev = SI.clusters(purchases)
    first = ev[ev["issuer_cik"] == "0000001001"]["event_date"].min()
    assert first == pd.Timestamp("2015-02-03")         # A2's filing date, not its 2015-01-06 trade date
    # 1002 traded 18 days apart (2015-01-02 and 2015-01-20) but FILED 30 days apart: no cluster
    assert ev[ev["issuer_cik"] == "0000001002"].empty


def test_cluster_window_boundary(loaded):
    purchases, _, _ = loaded
    ev = SI.clusters(purchases)
    assert ev[ev["issuer_cik"] == "0000001002"].empty            # 2015-01-05 and 2015-02-04: 30 days apart, out
    assert (ev["issuer_cik"] == "0000001001").any()              # 2015-01-05 and 2015-02-03: 29 days apart, in
    assert SI.CLUSTER_WINDOW_DAYS == 30 and SI.CLUSTER_MIN_OWNERS == 2


def test_joint_filing_is_not_a_cluster(loaded):
    purchases, _, _ = loaded
    d1 = purchases[purchases["accession"] == _acc("D1")]
    assert set(d1["owner_cik"]) == {"0000009031", "0000009032"}
    assert SI.clusters(purchases).query("issuer_cik == '0000001004'").empty


def test_amendment_is_excluded(loaded):
    purchases, spans, _ = loaded
    assert "0000009033" not in set(purchases["owner_cik"])       # D2 is a 4/A
    assert SI.clusters(purchases).query("issuer_cik == '0000001004'").empty
    assert "4/A" not in SI.PURCHASE_FORMS
    # ...but an amendment still counts as a filing for the symbol spans (see test_spans_and_reused_symbols)
    assert spans.set_index("symbol").loc["DDD", "last_filed"] == pd.Timestamp("2015-02-20")


def test_same_owner_twice_is_not_a_cluster(loaded):
    purchases, _, _ = loaded
    f = purchases[purchases["issuer_cik"] == "0000001006"]
    assert len(f) == 2 and f["owner_cik"].nunique() == 1 and f["accession"].nunique() == 2
    assert SI.clusters(purchases).query("issuer_cik == '0000001006'").empty


def test_code_is_case_and_space_insensitive_and_only_P_counts(loaded):
    purchases, _, _ = loaded
    owners = set(purchases["owner_cik"])
    assert "0000009041" in owners                      # E1 wrote " p "
    assert "0000009042" not in owners                  # E2 is a sale (S)
    assert "0000009064" not in owners                  # K1 is an award (A)
    assert "0000009062" not in owners                  # R2 is a sale (S)


def test_expected_events_exact(loaded):
    purchases, _, _ = loaded
    ev = SI.clusters(purchases)
    assert list(ev.columns) == SI.EVENT_COLUMNS
    assert len(ev) == 2
    assert ev["issuer_cik"].tolist() == ["0000001001", "0000001001"]
    assert ev["event_date"].tolist() == [pd.Timestamp("2015-02-03"), pd.Timestamp("2015-02-10")]
    assert ev["n_owners"].tolist() == [2, 2]
    assert ev["accession"].tolist() == [_acc("A2"), _acc("A3")]
    assert ev["symbol"].tolist() == ["AAA", "AAA"]
    assert ev["event_date"].min() == pd.Timestamp("2015-02-03")  # the completion date for 1001


def test_non_positive_and_unparseable_rows_add_nothing(tmp_path):
    sub = [("X1", "05-JAN-2015", "4", "4001", "XXX"), ("X2", "06-JAN-2015", "4", "4001", "XXX"),
           ("X3", "07-JAN-2015", "4", "4001", "XXX")]
    own = [("X1", "9101", "Director"), ("X2", "9102", "Director"), ("X3", "9103", "Director")]
    trn = [("X1", "02-JAN-2015", "P", "1000", "0"), ("X1", "02-JAN-2015", "P", "-1000", "6.00"),
           ("X1", "02-JAN-2015", "P", "abc", "6.00"), ("X1", "02-JAN-2015", "P", "1000", ""),
           ("X1", "02-JAN-2015", "P", "1000", "6.00"),
           ("X2", "03-JAN-2015", "P", "1000", "0"), ("X2", "03-JAN-2015", "P", "-5000", "5.00"),
           ("X2", "03-JAN-2015", "P", "inf", "5.00"),
           ("X3", "04-JAN-2015", "P", "1,000", "5.00")]
    purchases, _, _ = SI.load_quarter(_write(tmp_path / "2015q1_form345.zip", sub, own, trn))
    assert _tuples(purchases) == [("0000004001", "0000009101", _acc("X1"), "2015-01-05", 6000.0, "2015-01-02"),
                                  ("0000004001", "0000009103", _acc("X3"), "2015-01-07", 5000.0, "2015-01-04")]


def test_trans_date_column_is_optional(tmp_path):
    purchases, _, _ = SI.load_quarter(_write(tmp_path / "2015q1_form345.zip", with_trans_date=False))
    assert len(purchases) == 11 and purchases["trade_date"].isna().all()
    assert purchases["filing_date"].notna().all()


# -- spans and symbol map --------------------------------------------------------------------

def test_spans_and_reused_symbols(loaded):
    _, spans, _ = loaded
    assert list(spans.columns) == SI.SPAN_COLUMNS
    assert len(spans) == 9
    got = {(r.symbol, r.issuer_cik): (r.first_filed.strftime("%Y-%m-%d"), r.last_filed.strftime("%Y-%m-%d"), r.n_filings)
           for r in spans.itertuples(index=False)}
    assert got == {
        ("AAA", "0000001001"): ("2015-01-05", "2015-02-10", 3),
        ("BBB", "0000001002"): ("2015-01-05", "2015-02-04", 2),
        ("CCC", "0000001003"): ("2015-01-10", "2015-01-15", 3),
        ("DDD", "0000001004"): ("2015-02-10", "2015-02-20", 2),      # the 4/A is included
        ("EEE", "0000001005"): ("2015-03-02", "2015-03-03", 2),
        ("FFF", "0000001006"): ("2015-01-05", "2015-01-20", 2),
        ("RUSE", "0000002001"): ("2015-01-07", "2015-01-07", 1),
        ("RUSE", "0000002002"): ("2015-03-09", "2015-03-09", 1),
        ("BRK-B", "0000002004"): ("2015-03-11", "2015-03-11", 1),
    }
    assert "NONE" not in set(spans["symbol"]) and "" not in set(spans["symbol"])
    assert SI.reused_symbols(spans) == {"RUSE"}
    assert SI.covered_quarters(spans) == {"2015q1"}


def test_symbol_map_point_in_time(loaded):
    _, spans, _ = loaded
    keys = pd.DataFrame({
        "symbol": ["AAA", "RUSE", "RUSE", "AAA", "AAA", "AAA", "brk.b", "ZZZ"],
        "date": pd.to_datetime(["2015-03-30", "2015-01-30", "2015-03-30", "2014-12-30", "2016-02-10",
                                "2016-02-11", "2015-03-30", "2015-03-30"])})
    keys.index = range(10, 18)
    out = SI.map_symbols(spans, keys)
    assert list(out.columns) == ["symbol", "date", "cik", "ambiguous"]
    assert list(out.index) == list(keys.index)                       # input order and index preserved
    assert out["symbol"].tolist() == keys["symbol"].tolist()
    assert out["cik"].tolist() == ["0000001001", "0000002001", None, None, "0000001001", None, "0000002004", None]
    assert out["ambiguous"].tolist() == [False, False, True, False, False, False, False, False]
    assert out["ambiguous"].dtype == bool
    # a longer look-back reaches the 2015-02-10 filing of AAA from 2016-02-11
    assert SI.map_symbols(spans, keys.iloc[[5]], lookback_days=366)["cik"].tolist() == ["0000001001"]


def test_symbol_map_empty_inputs(loaded):
    _, spans, _ = loaded
    empty_keys = pd.DataFrame({"symbol": pd.Series(dtype=object), "date": pd.Series(dtype="datetime64[ns]")})
    out = SI.map_symbols(spans, empty_keys)
    assert len(out) == 0 and list(out.columns) == ["symbol", "date", "cik", "ambiguous"]
    keys = pd.DataFrame({"symbol": ["AAA"], "date": pd.to_datetime(["2015-03-30"])})
    out = SI.map_symbols(SI._empty_spans(), keys)
    assert out["cik"].tolist() == [None] and out["ambiguous"].tolist() == [False]


# -- direct unit tests of clusters() ---------------------------------------------------------

def _p(rows):
    """rows: (issuer, owner, accession, filing_date); builds a purchases frame by hand."""
    return pd.DataFrame({"accession": [r[2] for r in rows], "filing_date": pd.to_datetime([r[3] for r in rows]),
                         "issuer_cik": [r[0] for r in rows], "symbol": ["SYM"] * len(rows),
                         "owner_cik": [r[1] for r in rows], "value": [6000.0] * len(rows),
                         "trade_date": pd.NaT})[SI.PURCHASE_COLUMNS]


def test_clusters_window_parameter_and_boundary():
    two = _p([("I", "o1", "a1", "2015-01-01"), ("I", "o2", "a2", "2015-01-03")])
    assert len(SI.clusters(two, window_days=3)) == 1             # difference 2 < 3
    far = _p([("I", "o1", "a1", "2015-01-01"), ("I", "o2", "a2", "2015-01-04")])
    assert len(SI.clusters(far, window_days=3)) == 0             # difference 3 = window_days: out
    assert len(SI.clusters(far, window_days=4)) == 1


def test_clusters_same_day_rows_and_owner_count():
    same_day = _p([("I", "o1", "a1", "2015-03-02"), ("I", "o2", "a2", "2015-03-02"), ("I", "o3", "a3", "2015-03-02")])
    ev = SI.clusters(same_day)
    assert len(ev) == 1                                           # one event per (issuer, date)
    assert ev.loc[0, "n_owners"] == 3 and ev.loc[0, "accession"] == "a1"
    assert ev.loc[0, "event_date"] == pd.Timestamp("2015-03-02")
    assert len(SI.clusters(same_day, min_owners=4)) == 0


def test_clusters_later_rows_never_join_an_earlier_window():
    rows = _p([("I", "o1", "a1", "2015-01-01"), ("I", "o2", "a2", "2015-01-20"), ("I", "o3", "a3", "2015-03-01")])
    ev = SI.clusters(rows)
    assert ev["event_date"].tolist() == [pd.Timestamp("2015-01-20")]
    other_issuers = _p([("I", "o1", "a1", "2015-01-01"), ("J", "o2", "a2", "2015-01-02")])
    assert len(SI.clusters(other_issuers)) == 0                   # different companies never combine


def test_clusters_empty_input():
    ev = SI.clusters(SI._empty_purchases())
    assert len(ev) == 0 and list(ev.columns) == SI.EVENT_COLUMNS


# -- layout tolerance and loud failures ------------------------------------------------------

def test_member_and_header_case_insensitive(tmp_path, loaded):
    purchases, spans, _ = SI.load_quarter(_write(tmp_path / "2015q1_form345.zip", lower=True))
    assert _tuples(purchases) == _expected()
    assert len(spans) == 9


def test_find_member_rules():
    assert SI.find_member(["x/SUBMISSION.TSV", "y.csv"], "submission") == "x/SUBMISSION.TSV"
    assert SI.find_member(["NONDERIV_TRANS.txt", "other.tsv"], "NONDERIV_TRANS") == "NONDERIV_TRANS.txt"
    assert SI.find_member(["d/", "d/REPORTINGOWNER"], "REPORTINGOWNER") == "d/REPORTINGOWNER"
    with pytest.raises(SI.FormatError, match="more|2 members"):
        SI.find_member(["SUBMISSION.tsv", "old/submission.txt"], "SUBMISSION")
    with pytest.raises(SI.FormatError, match="SUBMISSION_EXTRA.tsv"):
        SI.find_member(["SUBMISSION_EXTRA.tsv"], "SUBMISSION")


def test_date_formats(tmp_path):
    s = pd.Series(["05-JAN-2015", "05-jan-2015", "2015-01-05", "01/05/2015", " 05-Jan-2015 ", "garbage", ""])
    got = SI.parse_dates(s)
    assert got.tolist()[:5] == [pd.Timestamp("2015-01-05")] * 5
    assert pd.isna(got.iloc[5]) and pd.isna(got.iloc[6])
    assert str(got.dtype) == "datetime64[ns]"
    sub = SUB_ROWS + [("G1", "garbage", "4", "3001", "GGG"), ("G2", "05-JAN-2015", "4", "", "HHH")]
    own = OWN_ROWS + [("G1", "9071", "Director"), ("G2", "9072", "Director")]
    trn = TRN_ROWS + [("G1", "03-JAN-2015", "P", "10000", "5.00"), ("G2", "03-JAN-2015", "P", "10000", "5.00")]
    purchases, spans, stats = SI.load_quarter(_write(tmp_path / "2015q1_form345.zip", sub, own, trn))
    assert stats["bad_dates"] == 1                                # the garbage FILING_DATE row
    assert stats["no_cik"] == 1                                   # the empty issuer CIK row
    assert _tuples(purchases) == _expected()                      # neither row leaks into purchases
    assert "GGG" not in set(spans["symbol"]) and "HHH" not in set(spans["symbol"])


def test_missing_column_fails_loudly(tmp_path):
    p = _write(tmp_path / "2015q1_form345.zip", drop_sym=True)
    with pytest.raises(SI.FormatError) as ei:
        SI.load_quarter(p)
    msg = str(ei.value)
    assert "ISSUERTRADINGSYMBOL" in msg and "columns found" in msg and "ACCESSION_NUMBER" in msg


def test_missing_table_fails_loudly(tmp_path):
    p = _write(tmp_path / "2015q1_form345.zip", drop_table="REPORTINGOWNER")
    with pytest.raises(SI.FormatError) as ei:
        SI.load_quarter(p)
    msg = str(ei.value)
    assert "REPORTINGOWNER" in msg and "SUBMISSION.tsv" in msg and "NONDERIV_TRANS.tsv" in msg


def test_unreadable_zip_is_a_format_error(tmp_path):
    bad = tmp_path / "2015q1_form345.zip"
    bad.write_bytes(b"this is not a zip file")
    with pytest.raises(SI.FormatError, match="not a readable zip"):
        SI.load_quarter(bad)


def test_inventory(tmp_path):
    d = tmp_path / "ins"
    d.mkdir()
    (d / "2015q1_form345.zip").write_bytes(b"not opened")
    (d / "2015Q2_FORM345.ZIP").write_bytes(b"not opened either")
    (d / "notes.txt").write_text("x")
    inv = SI.inventory(d)
    assert inv["exists"] is True and inv["n_zips"] == 2
    assert inv["quarters"] == ["2015q1", "2015q2"]
    assert inv["zips"] == ["2015Q2_FORM345.ZIP", "2015q1_form345.zip"] and inv["unrecognized"] == []
    assert inv["path"] == str(d)
    (d / "mystery.zip").write_bytes(b"")
    assert SI.inventory(d)["unrecognized"] == ["mystery.zip"]
    gone = SI.inventory(tmp_path / "nope")
    assert gone["exists"] is False and gone["n_zips"] == 0 and gone["quarters"] == [] and gone["zips"] == []


def test_load_dir_empty_missing_and_dedupe(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "readme.txt").write_text("x")
    with pytest.raises(FileNotFoundError):
        SI.load_dir(empty)
    with pytest.raises(FileNotFoundError):
        SI.load_dir(tmp_path / "missing")
    d = tmp_path / "two"
    d.mkdir()
    _write(d / "2015q1_form345.zip")
    _write(d / "2015q2_form345.zip")                              # the same filings again: must de-duplicate
    purchases, spans, stats = SI.load_dir(d)
    assert _tuples(purchases) == _expected()
    assert len(spans) == 18                                       # spans stay one row per source zip
    assert [s["file"] for s in stats] == ["2015q1_form345.zip", "2015q2_form345.zip"]
    assert SI.reused_symbols(spans) == {"RUSE"}


def test_normalizers():
    assert SI.normalize_symbol(" brk.b ") == "BRK-B"
    for missing in (None, float("nan"), "", "  ", "none", "N/A", "na", "NULL", "-", "."):
        assert SI.normalize_symbol(missing) == ""
    assert SI.pad_cik("1001") == "0000001001"
    assert SI.pad_cik("CIK 0000001001") == "0000001001"
    assert SI.pad_cik("") == "" and SI.pad_cik(None) == "" and SI.pad_cik("abc") == ""
    s = pd.Series([" brk.b ", "NONE", "ruse", "", "n/a"])
    assert SI._norm_symbols(s).tolist() == [SI.normalize_symbol(x) for x in s]


def test_module_is_import_light_and_pure_ascii_prose():
    src = (ROOT / "data_engine" / "sec_insider.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    allowed = {"__future__", "csv", "importlib", "os", "re", "sys", "time", "zipfile", "pathlib", "numpy", "pandas"}
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert {a.name.split(".")[0] for a in node.names} <= allowed
        elif isinstance(node, ast.ImportFrom):
            assert node.module.split(".")[0] in allowed
    for node in ast.walk(tree):                                   # nowhere, not even lazily
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("data_engine")
        if isinstance(node, ast.Import):
            assert not any(a.name.startswith("data_engine") for a in node.names)
    assert chr(0x2014) not in src
    assert chr(0x2014) not in Path(__file__).read_text(encoding="utf-8")


# -- download helper (UNVERIFIED; only ever exercised with a fake fetch) ---------------------

def _tiny_zip_bytes(tmp_path):
    p = _zip(tmp_path / "tiny.zip", {"SUBMISSION.tsv": (["A"], [["1"]])})
    return p.read_bytes()


def _must_not_fetch(url, headers):
    raise AssertionError("fetch must not be called")


def test_download_requires_user_agent(tmp_path, monkeypatch):
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    with pytest.raises(RuntimeError, match="SEC_USER_AGENT"):
        SI.download_quarters(["2015q1"], tmp_path / "dl", fetch=_must_not_fetch)
    assert not (tmp_path / "dl" / "2015q1_form345.zip").exists()


def test_download_writes_zip_and_sends_ua(tmp_path, monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", UA)
    body = _tiny_zip_bytes(tmp_path)
    calls, sleeps = [], []

    def fetch(url, headers):
        calls.append((url, dict(headers)))
        return 200, body

    dest = tmp_path / "dl"
    paths = SI.download_quarters(["2015q1", "2015q2"], dest, fetch=fetch, clock=lambda: 0.0, sleep=sleeps.append)
    assert paths == [dest / "2015q1_form345.zip", dest / "2015q2_form345.zip"]
    assert (dest / "2015q1_form345.zip").read_bytes() == body
    assert calls[0][0] == SI.DATASET_URLS[0].format(year=2015, q=1)
    assert calls[1][0] == SI.DATASET_URLS[0].format(year=2015, q=2)
    assert calls[0][1]["User-Agent"] == UA
    assert len(sleeps) == 1 and sleeps[0] > 0                     # the second request was throttled
    assert [p.name for p in dest.iterdir() if p.name.endswith(".tmp")] == []


def test_download_rejects_non_zip_and_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", UA)
    dest = tmp_path / "dl"
    for status, body in ((200, b"<html>"), (404, b"")):
        with pytest.raises(RuntimeError, match="no candidate URL"):
            SI.download_quarters(["2015q1"], dest, fetch=lambda u, h, s=status, b=body: (s, b),
                                 clock=lambda: 0.0, sleep=lambda s: None)
        assert not dest.exists() or list(dest.iterdir()) == []   # no file and no .tmp left behind
    # a zip served with an error status is still an error
    with pytest.raises(RuntimeError, match="no candidate URL"):
        SI.download_quarters(["2015q1"], dest, fetch=lambda u, h: (500, _tiny_zip_bytes(tmp_path)),
                             clock=lambda: 0.0, sleep=lambda s: None)
    assert list(dest.iterdir()) == []
    # an existing file is skipped without calling fetch
    dest.mkdir(exist_ok=True)
    keep = dest / "2015q1_form345.zip"
    keep.write_bytes(b"already here")
    assert SI.download_quarters(["2015q1"], dest, fetch=_must_not_fetch) == [keep]
    assert keep.read_bytes() == b"already here"
    for bad in ("2015q5", "15q1", "2015Q1", "2015q1; rm", ""):
        with pytest.raises(ValueError):
            SI.download_quarters([bad], dest, fetch=_must_not_fetch)


def test_socket_guard_is_active():
    with pytest.raises(AssertionError, match="network access"):
        socket.socket().connect(("192.0.2.1", 80))
