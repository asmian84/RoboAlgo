"""setup_engine.svgchart: a hand-written chart must stay valid XML, must not lie, and must stay readable.

The failures being guarded against:
* markup injection / broken files: a label like "A & B <x>" (or a title with quotes) written
  unescaped makes the SVG unparseable, and one bad file takes a whole HTML index of charts with it;
* "nan" in the output: a coordinate computed from a missing value is written as the text "nan",
  which browsers silently render as nothing or as a line to the corner;
* a moving average that is joined ACROSS its gap, drawing a value that never existed;
* a log axis that is not log (the point of the log axis is that 10 -> 100 -> 1000 is evenly spaced);
* a marker snapped to the NEXT bar, i.e. an event drawn on a bar that had not happened yet;
* a flat price series dividing by zero in the scale;
* marker labels written on top of each other, which is what makes a chart unreadable.
Each guard below is built as a trap and is mutation-tested.
"""
import re
import xml.etree.ElementTree as ET
from html.parser import HTMLParser

import numpy as np
import pandas as pd
import pytest

from setup_engine import svgchart as sc
from setup_engine.svgchart import MARKER_COLOURS, Marker, weekly_chart_svg, write_html_index, write_svg

NS = "{http://www.w3.org/2000/svg}"


def _bars(n=60, start="2020-01-03", seed=1, base=20.0, volume=True):
    """n weekly bars labelled on Fridays: a seeded log random walk with o/h/l/c consistent."""
    rng = np.random.default_rng(seed)
    close = base * np.exp(np.cumsum(rng.normal(0.003, 0.04, n)))
    open_ = np.r_[base, close[:-1]] * (1 + rng.normal(0, 0.004, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.01, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.01, n)))
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close},
                      index=pd.date_range(start, periods=n, freq="W-FRI", name="week"))
    if volume:
        df["volume"] = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return df


def _flat(closes, start="2022-01-07"):
    """Bars with open = high = low = close = the given values, one per Friday, no volume column."""
    c = np.asarray(closes, dtype=float)
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c},
                        index=pd.date_range(start, periods=len(c), freq="W-FRI", name="week"))


class _TagCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))


def _root(svg):
    return ET.fromstring(svg)


def _cls(el):
    return el.get("class", "").split()


def _find(root, tag, cls=None):
    return [e for e in root.iter(NS + tag) if cls is None or cls in _cls(e)]


def _points(poly):
    return [tuple(float(v) for v in p.split(",")) for p in poly.get("points").split()]


def _markers(root):
    return [g for g in _find(root, "g") if "marker" in _cls(g)]


# ------------------------------------------------------------------------- 1. valid XML

def test_output_parses_as_xml_with_an_svg_root_in_every_mode():
    bars = _bars(120)
    sma = bars["close"].rolling(30).mean()
    rs = bars["close"] / (bars["close"].iloc[0] * 2)
    for kw in (dict(), dict(candles=False), dict(log_scale=False), dict(sma30=sma, sma10m=sma, rs_line=rs),
               dict(sma30=sma, rs_line=rs, log_scale=False, candles=False)):
        svg = weekly_chart_svg(bars, title="T", header_lines=["a", "b"], **kw)
        root = _root(svg)
        assert root.tag.endswith("svg")
        assert root.tag == NS + "svg"
        assert root.get("width") == "1100"


def test_output_is_deterministic():
    bars = _bars(80)
    m = [Marker(bars.index[10], "low", "low")]
    assert weekly_chart_svg(bars, markers=m, title="x") == weekly_chart_svg(bars, markers=m, title="x")


# ------------------------------------------------------------------------- 2. markers

def test_low_confirm_peak_make_exactly_three_marker_groups_with_their_kinds_and_labels():
    bars = _bars(100)
    labels = {"low": "Low 12.50", "confirm": "Confirm +20%", "peak": "Peak 88.00 (x7.0)"}
    markers = [Marker(bars.index[10], labels["low"], "low"),
               Marker(bars.index[40], labels["confirm"], "confirm"),
               Marker(bars.index[90], labels["peak"], "peak")]
    root = _root(weekly_chart_svg(bars, markers=markers))
    groups = _markers(root)
    assert len(groups) == 3
    assert sorted(g.get("data-kind") for g in groups) == ["confirm", "low", "peak"]
    for kind, label in labels.items():
        (g,) = [g for g in groups if "marker-" + kind in _cls(g)]
        texts = [t.text for t in g.iter(NS + "text")]
        assert texts == [label]
        assert g.find(NS + "title").text.startswith(label)
        assert g.find(NS + "line").get("stroke") == MARKER_COLOURS[kind]
        assert g.find(NS + "circle").get("fill") == MARKER_COLOURS[kind]


def test_unknown_marker_kind_is_drawn_as_an_event_and_does_not_break_the_xml():
    bars = _bars(30)
    root = _root(weekly_chart_svg(bars, markers=[Marker(bars.index[5], "x", 'we"ird <kind>')]))
    (g,) = _markers(root)
    assert g.find(NS + "line").get("stroke") == MARKER_COLOURS["event"]
    assert g.get("data-kind") == 'we"ird <kind>'


def test_marker_dots_sit_on_the_close_of_their_bar():
    bars = _bars(40)
    root = _root(weekly_chart_svg(bars, candles=False, markers=[Marker(bars.index[20], "m", "low")]))
    (g,) = _markers(root)
    (poly,) = _find(root, "polyline", "close")
    x, y = _points(poly)[20]
    assert float(g.find(NS + "circle").get("cx")) == x
    assert float(g.find(NS + "circle").get("cy")) == y
    assert float(g.find(NS + "line").get("x1")) == x


# ------------------------------------------------------------------------- 3. escaping

def test_markup_characters_in_every_text_are_escaped_and_round_trip():
    bars = _bars(40)
    nasty = 'A & B <x> "q" \'s\''
    title = 'Title "with" quotes & <tags>'
    svg = weekly_chart_svg(bars, title=title, header_lines=[nasty],
                           markers=[Marker(bars.index[10], nasty, "low")])
    root = _root(svg)                                   # raises if anything was written raw
    texts = [t.text for t in root.iter(NS + "text")]
    assert nasty in texts and title in texts
    assert "<x>" not in svg and "<tags>" not in svg
    assert _root(svg).get("aria-label") == title


def test_control_characters_that_xml_forbids_are_removed_rather_than_breaking_the_file():
    bars = _bars(30)
    svg = weekly_chart_svg(bars, title="bad\x00\x08title", markers=[Marker(bars.index[5], "la\x0bbel", "low")])
    root = _root(svg)
    assert "badtitle" in [t.text for t in root.iter(NS + "text")]
    assert "label" in [t.text for t in root.iter(NS + "text")]


# ------------------------------------------------------------------------- 4. NaN handling

def test_nan_in_an_overlay_breaks_the_line_in_two_and_nan_is_never_written():
    bars = _bars(60)
    sma = bars["close"].rolling(5, min_periods=1).mean()
    sma.iloc[28:32] = np.nan
    svg = weekly_chart_svg(bars, sma30=sma)
    root = _root(svg)
    polys = _find(root, "polyline", "sma30")
    assert len(polys) == 2
    assert len(_points(polys[0])) == 28 and len(_points(polys[1])) == 28
    assert "nan" not in svg.lower() and "inf" not in re.sub(r"info|infinite", "", svg.lower())


def test_leading_nan_from_a_warmup_window_just_starts_the_line_late():
    bars = _bars(60)
    sma = bars["close"].rolling(30, min_periods=30).mean()
    svg = weekly_chart_svg(bars, sma30=sma, sma10m=sma.shift(5))
    root = _root(svg)
    assert len(_find(root, "polyline", "sma30")) == 1
    assert len(_points(_find(root, "polyline", "sma30")[0])) == 31
    assert "nan" not in svg.lower()


def test_a_bar_with_a_nan_price_is_skipped_not_written_as_nan():
    bars = _bars(50)
    full = _root(weekly_chart_svg(bars))
    bars.iloc[20, bars.columns.get_loc("close")] = np.nan
    bars.iloc[30, bars.columns.get_loc("low")] = np.nan
    svg = weekly_chart_svg(bars)
    root = _root(svg)
    n_rects = lambda r: sum(len(g.findall(NS + "rect")) for g in _find(r, "g", "candles"))
    assert n_rects(full) == 50 and n_rects(root) == 48
    assert "nan" not in svg.lower()
    # close line mode breaks at the skipped bar
    assert len(_find(_root(weekly_chart_svg(bars, candles=False)), "polyline", "close")) == 3


def test_nan_in_rs_and_volume_never_reaches_the_output():
    bars = _bars(50)
    rs = pd.Series(np.linspace(1, 2, 50), index=bars.index)
    rs.iloc[10:20] = np.nan
    bars.iloc[5, bars.columns.get_loc("volume")] = np.nan
    svg = weekly_chart_svg(bars, rs_line=rs)
    assert len(_find(_root(svg), "polyline", "rs")) == 2
    assert "nan" not in svg.lower()


# ------------------------------------------------------------------------- 5. log scale

def _close_ys(svg):
    (poly,) = _find(_root(svg), "polyline", "close")
    return [y for _, y in _points(poly)]


def test_log_scale_spaces_10_100_1000_evenly_and_linear_scale_does_not():
    bars = _flat([10, 100, 1000])
    y = _close_ys(weekly_chart_svg(bars, candles=False, log_scale=True))
    assert abs((y[0] - y[1]) - (y[1] - y[2])) < 1e-6
    assert y[0] > y[1] > y[2]                                   # higher price, smaller y
    yl = _close_ys(weekly_chart_svg(bars, candles=False, log_scale=False))
    assert abs((yl[0] - yl[1]) - (yl[1] - yl[2])) > 10
    assert yl[0] - yl[1] < yl[1] - yl[2]


def test_log_scale_ignores_bars_it_cannot_draw():
    bars = _flat([10, 0.0, 100, -5, 1000])
    svg = weekly_chart_svg(bars, candles=False)
    assert "nan" not in svg.lower()
    assert len(_find(_root(svg), "polyline", "close")) == 3     # 10 | 100 | 1000, split by the bad bars
    _root(weekly_chart_svg(bars, candles=False, log_scale=False))


# ------------------------------------------------------------------------- 6. flat prices

@pytest.mark.parametrize("kw", [dict(), dict(candles=False), dict(log_scale=False),
                                dict(candles=False, log_scale=False)])
def test_constant_prices_do_not_crash_and_stay_inside_the_panel(kw):
    bars = _flat([25.0] * 30)
    bars["volume"] = 0.0
    sma = pd.Series(25.0, index=bars.index)
    rs = pd.Series(1.0, index=bars.index)
    svg = weekly_chart_svg(bars, sma30=sma, rs_line=rs, **kw)
    root = _root(svg)
    assert "nan" not in svg.lower()
    frame = _find(root, "rect", "frame-price")[0]
    top, bottom = float(frame.get("y")), float(frame.get("y")) + float(frame.get("height"))
    for poly in _find(root, "polyline", "sma30"):
        assert all(top < y < bottom for _, y in _points(poly))


def test_a_single_bar_and_all_zero_linear_prices_draw():
    _root(weekly_chart_svg(_flat([7.0])))
    _root(weekly_chart_svg(_flat([0.0, 0.0, 0.0]), log_scale=False))


# ------------------------------------------------------------------------- 7. marker snapping

def test_a_mid_week_marker_snaps_back_to_the_previous_friday_not_the_next_bar():
    bars = _bars(30)                                           # Fridays from 2020-01-03
    friday, nxt = bars.index[10], bars.index[11]
    mid = friday + pd.Timedelta(days=3)                        # a Monday
    mon = _root(weekly_chart_svg(bars, candles=False, markers=[Marker(mid, "m", "event")]))
    fri = _root(weekly_chart_svg(bars, candles=False, markers=[Marker(friday, "m", "event")]))
    xs = [x for x, _ in _points(_find(mon, "polyline", "close")[0])]
    (g,) = _markers(mon)
    assert float(g.find(NS + "line").get("x1")) == xs[10] != xs[11]
    assert g.get("data-bar") == friday.date().isoformat()
    assert g.get("data-date") == mid.date().isoformat()
    assert float(_markers(fri)[0].find(NS + "line").get("x1")) == xs[10]
    assert nxt > mid                                           # the trap: the next label is later


def test_a_marker_after_the_last_bar_snaps_to_the_last_bar():
    bars = _bars(20)
    root = _root(weekly_chart_svg(bars, candles=False, markers=[Marker(bars.index[-1] + pd.Timedelta(days=40), "m")]))
    xs = [x for x, _ in _points(_find(root, "polyline", "close")[0])]
    assert float(_markers(root)[0].find(NS + "line").get("x1")) == xs[-1]


def test_a_marker_before_the_first_bar_is_dropped_and_counted_in_a_comment():
    bars = _bars(20)
    early = bars.index[0] - pd.Timedelta(days=1)
    svg = weekly_chart_svg(bars, markers=[Marker(early, "gone", "low"), Marker(bars.index[3], "kept", "peak"),
                                          Marker(early - pd.Timedelta(days=90), "gone2")])
    root = _root(svg)
    assert [g.get("data-kind") for g in _markers(root)] == ["peak"]
    assert "gone" not in [t.text for t in root.iter(NS + "text")]
    assert "<!-- dropped 2 marker(s) dated before the first bar -->" in svg
    assert "dropped" not in weekly_chart_svg(bars, markers=[Marker(bars.index[3], "k")])


# ------------------------------------------------------------------------- 8. html index

def test_write_html_index_has_each_symbol_the_escaped_summary_and_the_image(tmp_path):
    entries = [dict(symbol="NVDA", svg_file="charts/NVDA.svg", summary_lines=["low 0.5 -> peak 240 & more", "a <b> line"]),
               dict(symbol="AMD", svg_file="charts/AMD.svg", summary_lines=[])]
    path = write_html_index(tmp_path / "out" / "index.html", entries, "Case <study> & more", intro_lines=["intro & note"])
    html = path.read_text(encoding="utf-8")
    assert html.startswith("<!DOCTYPE html>")
    for needle in ("<h2>NVDA</h2>", "<h2>AMD</h2>", "low 0.5 -&gt; peak 240 &amp; more", "a &lt;b&gt; line",
                   '<img src="charts/NVDA.svg" loading="lazy"', '<img src="charts/AMD.svg" loading="lazy"',
                   "<title>Case &lt;study&gt; &amp; more</title>", "intro &amp; note"):
        assert needle in html, needle
    assert html.count("<section") == 2
    assert "<script" not in html.lower() and "<b>" not in html
    parser = _TagCollector()
    parser.feed(html)
    assert [a["src"] for t, a in parser.tags if t == "img"] == ["charts/NVDA.svg", "charts/AMD.svg"]
    assert [a["loading"] for t, a in parser.tags if t == "img"] == ["lazy", "lazy"]
    assert [t for t, _ in parser.tags if t in ("script", "iframe", "object")] == []


def test_write_html_index_quotes_a_file_name_with_spaces_and_handles_no_entries(tmp_path):
    p = write_html_index(tmp_path / "i.html", [dict(symbol="X", svg_file="my charts/a b.svg", summary_lines=["s"])], "t")
    assert 'src="my%20charts/a%20b.svg"' in p.read_text(encoding="utf-8")
    p2 = write_html_index(tmp_path / "empty.html", [], "nothing")
    assert "<section" not in p2.read_text(encoding="utf-8")


def test_write_svg_round_trips_and_creates_the_directory(tmp_path):
    svg = weekly_chart_svg(_bars(20), title="é — ü")
    p = write_svg(tmp_path / "a" / "b" / "c.svg", svg)
    assert p.read_text(encoding="utf-8") == svg
    ET.parse(p)


# ------------------------------------------------------------------------- 9. empty input

def test_empty_or_unusable_bars_raise_value_error():
    with pytest.raises(ValueError):
        weekly_chart_svg(pd.DataFrame(columns=["open", "high", "low", "close", "volume"]))
    with pytest.raises(ValueError, match="missing"):
        weekly_chart_svg(_bars(10).drop(columns=["close"]))
    nan_bars = _bars(10)
    nan_bars[["open", "high", "low", "close"]] = np.nan
    with pytest.raises(ValueError, match="plottable"):
        weekly_chart_svg(nan_bars)
    with pytest.raises(ValueError, match="increasing"):
        weekly_chart_svg(_bars(10).iloc[::-1])


# ------------------------------------------------------------------------- readability

def _label_boxes(root):
    """(x0, x1, y) of every marker label, using the module's own width estimate."""
    boxes = []
    for t in _find(root, "text", "marker-label"):
        x = float(t.get("x"))
        boxes.append((x, x + sc.LABEL_CHAR_W * len(t.text), float(t.get("y")), t.text))
    return boxes


def test_close_markers_get_their_labels_stacked_so_no_two_labels_overlap():
    bars = _bars(120)
    markers = [Marker(bars.index[50], "Low 10.25 (2022-10-14)", "low"),
               Marker(bars.index[51], "Confirm: close over 30w SMA", "confirm"),
               Marker(bars.index[52], "Entry next open", "entry"),
               Marker(bars.index[54], "Peak", "peak"),
               Marker(bars.index[110], "Exit", "exit")]
    root = _root(weekly_chart_svg(bars, markers=markers))
    boxes = _label_boxes(root)
    assert len(boxes) == 5
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            same_row = abs(a[2] - b[2]) < 1
            overlap = a[0] < b[1] and b[0] < a[1]
            assert not (same_row and overlap), (a, b)
    assert len({round(y) for _, _, y, _ in boxes}) >= 3          # the crowded ones really are stacked
    far = [b for b in boxes if b[3] == "Exit"][0]
    assert far[2] == min(y for _, _, y, _ in boxes)               # an uncrowded marker uses the top row


def test_well_separated_markers_share_a_single_row():
    bars = _bars(200)
    markers = [Marker(bars.index[20], "Low", "low"), Marker(bars.index[100], "Confirm", "confirm"),
               Marker(bars.index[180], "Peak", "peak")]
    boxes = _label_boxes(_root(weekly_chart_svg(bars, markers=markers)))
    assert len({round(y) for _, _, y, _ in boxes}) == 1


def test_a_marker_line_does_not_run_through_another_markers_label():
    bars = _bars(120)
    markers = [Marker(bars.index[i], f"Marker number {i}", "event") for i in (40, 41, 44, 47, 50, 53)]
    root = _root(weekly_chart_svg(bars, markers=markers))
    labels = {t.text: t for t in _find(root, "text", "marker-label")}
    for g in _markers(root):
        line = g.find(NS + "line")
        lx, y_start = float(line.get("x1")), float(line.get("y1"))
        own = g.find(NS + "text").text
        for text, t in labels.items():
            if text == own:
                continue
            x0, y = float(t.get("x")), float(t.get("y"))
            crosses = x0 - 3 < lx < x0 + sc.LABEL_CHAR_W * len(text) + 3 and y > y_start
            assert not crosses, (own, text)


def test_a_label_near_the_right_edge_stays_inside_the_canvas():
    bars = _bars(60)
    root = _root(weekly_chart_svg(bars, markers=[Marker(bars.index[-1], "A fairly long label at the end", "peak")]))
    ((x0, x1, _, _),) = _label_boxes(root)
    assert x1 <= 1100 - sc.MARGIN_RIGHT + 1e-6 and x0 >= sc.MARGIN_LEFT


def test_many_markers_grow_the_canvas_instead_of_squeezing_the_panels():
    bars = _bars(120)
    markers = [Marker(bars.index[60 + i], f"A long marker label number {i}", "event") for i in range(30)]
    root = _root(weekly_chart_svg(bars, markers=markers, height=400))
    price = float(_find(root, "rect", "frame-price")[0].get("height"))
    assert price > 100 and int(root.get("height")) > 400
    assert root.get("viewBox") == f"0 0 1100 {root.get('height')}"


def test_log_price_ticks_are_1_2_5_per_decade_and_inside_the_axis():
    ax = sc._Axis(0.09, 245.0, 0.0, 300.0, True)
    ticks = sc._price_ticks(ax)
    assert ticks == [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
    assert all(ax.lo <= t <= ax.hi for t in ticks)
    bars = _bars(300, base=0.5)
    bars[["open", "high", "low", "close"]] *= np.r_[np.ones(150), np.full(150, 40.0)][:, None]
    texts = [t.text for t in _find(_root(weekly_chart_svg(bars)), "text", "ytick")]
    assert {"1", "10"} <= set(texts)


def test_a_narrow_log_range_falls_back_to_round_linear_ticks():
    ax = sc._Axis(100.0, 120.0, 0.0, 300.0, True)
    ticks = sc._price_ticks(ax)
    assert len(ticks) >= 3 and all(100.0 <= t <= 120.0 for t in ticks)
    assert ticks == sorted(ticks)
    assert sc._fmt_price(1234.0) == "1,234" and sc._fmt_price(0.05) == "0.05" and sc._fmt_price(2.5) == "2.5"


def test_every_year_is_ticked_when_there_is_room_and_each_tick_is_on_the_first_bar_of_its_year():
    bars = _bars(1350, start="2000-01-07", seed=3)             # about 26 years in ~976 px
    root = _root(weekly_chart_svg(bars))
    ticks = _find(root, "text", "xtick")
    assert [int(t.text) for t in ticks] == list(range(2000, 2026))
    first_bar = {int(y): i for i, y in reversed(list(enumerate(bars.index.year)))}
    poly = _find(_root(weekly_chart_svg(bars, candles=False)), "polyline", "close")[0]
    xs = [x for x, _ in _points(poly)]
    for t in ticks:
        assert float(t.get("x")) == pytest.approx(xs[first_bar[int(t.text)]], abs=0.01)
    short = _bars(260, start="2020-01-03", seed=3)             # 5 years
    assert [t.text for t in _find(_root(weekly_chart_svg(short)), "text", "xtick")] == \
        ["2020", "2021", "2022", "2023", "2024"]


def test_year_ticks_are_thinned_on_a_very_long_chart_so_the_labels_never_touch():
    bars = _bars(2400, start="1980-01-04", seed=3)             # about 46 years: 1 year is ~20 px
    ticks = _find(_root(weekly_chart_svg(bars)), "text", "xtick")
    years = [int(t.text) for t in ticks]
    assert len(years) >= 15 and all(y % 2 == 0 for y in years)
    xs = [float(t.get("x")) for t in ticks]
    assert min(b - a for a, b in zip(xs, xs[1:])) >= 34


def test_a_chart_spanning_under_three_years_ticks_months_with_the_year_on_january():
    bars = _bars(70, start="2023-09-01", seed=3)
    texts = [t.text for t in _find(_root(weekly_chart_svg(bars)), "text", "xtick")]
    assert "Jan 2024" in texts and any(t in ("Oct", "Nov", "Dec", "Feb", "Mar") for t in texts)
    assert not any(re.fullmatch(r"\d{4}", t) for t in texts)


def test_a_series_starting_mid_year_does_not_get_a_tick_for_that_partial_year():
    bars = _bars(600, start="2010-07-02", seed=3)
    years = [int(t.text) for t in _find(_root(weekly_chart_svg(bars)), "text", "xtick")]
    assert 2010 not in years and years[0] == 2011


def test_the_relative_strength_panel_exists_only_when_there_is_an_rs_line():
    bars = _bars(80)
    assert not _find(_root(weekly_chart_svg(bars)), "polyline", "rs")
    assert not _find(_root(weekly_chart_svg(bars)), "rect", "frame-rs")
    root = _root(weekly_chart_svg(bars, rs_line=bars["close"] / 50))
    assert len(_find(root, "polyline", "rs")) == 1 and _find(root, "rect", "frame-rs")
    no_vol = _root(weekly_chart_svg(_bars(80, volume=False)))
    assert not _find(no_vol, "rect", "frame-volume")


def test_panel_heights_follow_the_58_14_18_split_of_the_plot_height():
    root = _root(weekly_chart_svg(_bars(80), rs_line=pd.Series(1.0, index=_bars(80).index)))
    h = {k: float(_find(root, "rect", "frame-" + k)[0].get("height")) for k in ("price", "volume", "rs")}
    assert h["price"] / h["rs"] == pytest.approx(0.58 / 0.18, rel=1e-3)
    assert h["volume"] / h["price"] == pytest.approx(0.14 / 0.58, rel=1e-3)


def test_candle_up_is_hollow_green_and_down_is_filled_red_when_bars_are_wide_enough():
    bars = _bars(40)
    root = _root(weekly_chart_svg(bars))
    up = _find(root, "g", "up")[0]
    down = [g for g in _find(root, "g", "candles") if "down" in _cls(g)][0]
    assert up.get("fill") == "#ffffff" and up.get("stroke") == sc.UP_COLOUR
    assert down.get("fill") == down.get("stroke") == sc.DOWN_COLOUR
    n_up = int((bars["close"] >= bars["open"]).sum())
    assert len(up.findall(NS + "rect")) == n_up and len(up.findall(NS + "line")) == n_up


def test_a_line_forced_across_a_label_is_drawn_before_that_label_so_the_text_stays_on_top():
    bars = _bars(120)
    # long labels at the right edge are pulled left over their neighbours' lines
    markers = [Marker(bars.index[i], f"Marker with a rather long label number {i}", "event")
               for i in (118, 114, 110, 106, 103, 100)]           # input order must not decide drawing order
    root = _root(weekly_chart_svg(bars, markers=markers))
    groups = _markers(root)
    order = {g.get("data-date"): k for k, g in enumerate(groups)}
    texts = {g.get("data-date"): g.find(NS + "text") for g in groups}
    lines = {g.get("data-date"): g.find(NS + "line") for g in groups}
    crossings = 0
    for d_line, line in lines.items():
        lx, y_start = float(line.get("x1")), float(line.get("y1"))
        for d_text, t in texts.items():
            if d_text == d_line:
                continue
            x0, y = float(t.get("x")), float(t.get("y"))
            if x0 < lx < x0 + sc.LABEL_CHAR_W * len(t.text) and y > y_start:
                crossings += 1
                assert order[d_line] < order[d_text], (d_line, d_text)
    assert crossings > 0                                       # the scenario really forces a crossing
    assert all(t.get("stroke") == "#ffffff" for t in texts.values())    # halo
