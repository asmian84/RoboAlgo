"""
Pure-Python SVG charts of weekly bars, and a static HTML index to page through them.

No plotting library: the SVG is written by hand from numpy/pandas, so a chart costs nothing to
install, opens in any browser, and can be read back by a test with ``xml.etree``. One chart is a
log-price candlestick panel (30-week SMA and the monthly 10-month SMA as overlays), a volume panel
and an optional relative-strength panel, with year ticks along the bottom and labelled markers
(low, confirmation, peak, entry, exit) hanging from a label strip above the candles.

THE NO-LOOK-AHEAD RULE
----------------------
This module computes nothing: every bar, average and RS value is drawn as given. The caller is
responsible for passing COMPLETE weekly/monthly bars (``setup_engine.mtf`` builds them: a bar is
used only once it is complete) and for passing overlays that were computed from completed bars.
Markers are the one thing that is allowed to use the future: a "low" or "peak" marker is an
EX-POST label from a case study, drawn so a person can see what the screen would have been up
against. It is never a feature and never an input to a rule. A marker whose date is not a bar label
snaps BACKWARDS to the latest bar label <= the date (never to the next bar, which would draw an
event on a bar that had not happened yet); a marker before the first bar is dropped, and the drop is
counted in an XML comment.

THE ORDER MODEL
---------------
None. A marker's dot sits on the CLOSE of the weekly bar it snaps to. That is where a person
reads the chart, not where an order would have filled (signal at a weekly close, entry at the next
daily open; see ``setup_engine.trade_sim``).

KNOWN BIASES (the picture inherits them from the data it is given)
-------------------------------------------------------------------
Survivorship (today's listings only), Yahoo's retroactive split adjustment (a stock can be drawn at
prices it never traded at), and no dividend adjustment. On a log axis a bar with a non-positive
price cannot be drawn and is skipped, like a bar with a NaN price.

SAFETY
------
Every text and attribute goes through ``xml.sax.saxutils.escape`` / ``quoteattr`` after control
characters that XML 1.0 forbids are removed, and no coordinate can be NaN or infinite (a coordinate
that is would raise instead of being written as "nan"). Polylines break at missing values rather
than joining across them. The HTML index contains no JavaScript.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote
from xml.sax.saxutils import escape, quoteattr

import numpy as np
import pandas as pd

SVG_NS = "http://www.w3.org/2000/svg"

# colour per marker kind; an unknown kind is drawn as "event"
MARKER_COLOURS = {
    "low": "#1565c0",
    "confirm": "#00897b",
    "peak": "#ad1457",
    "entry": "#2e7d32",
    "exit": "#c62828",
    "event": "#546e7a",
}

UP_COLOUR = "#1b8a4b"
DOWN_COLOUR = "#c62828"
SMA30_COLOUR = "#f08c00"
SMA10M_COLOUR = "#6a1b9a"
RS_COLOUR = "#1c7ed6"
INK = "#1f2933"
MUTED = "#6b7785"
GRID = "#e6e9ee"
GRID_MAJOR = "#d3d8df"
FRAME = "#b8bfc9"
FONT = "-apple-system, 'Helvetica Neue', Helvetica, Arial, sans-serif"

# layout (pixels) and panel shares of the plot height (title/label strip and x axis excluded)
MARGIN_LEFT = 62
MARGIN_RIGHT = 60
AXIS_H = 30
LANE_H = 15
MIN_PLOT_H = 240
PRICE_SHARE, VOLUME_SHARE, RS_SHARE, GAP_SHARE = 0.58, 0.14, 0.18, 0.05
RANGE_MARGIN = 0.03         # breathing room above/below the data, as a share of the (log) range
LABEL_CHAR_W = 6.4          # estimated width of one label character at LABEL_FONT_PX
LABEL_FONT_PX = 11
LABEL_PAD = 8               # minimum horizontal gap between two labels on the same lane
MIN_TICK_GAP = 16           # minimum pixel gap between two y-axis ticks

_LOG_MANTISSAS = ((1, 2, 3, 4, 5, 6, 7, 8, 9), (1, 2, 3, 5, 7), (1, 2, 5), (1, 3), (1,))
_XML_BAD = re.compile("[^\u0009\u000a\u000d -퟿-�\U00010000-\U0010ffff]")
_KIND_SAFE = re.compile(r"[^A-Za-z0-9_-]")
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


@dataclass(frozen=True)
class Marker:
    """A labelled date on the chart. ``kind`` picks the colour from ``MARKER_COLOURS``."""

    date: pd.Timestamp
    label: str
    kind: str = "event"


# --------------------------------------------------------------------------------------- text

def _clean(s) -> str:
    """str(s) without the control characters XML 1.0 cannot carry."""
    return _XML_BAD.sub("", str(s))


def _t(s) -> str:
    """Text-node safe string: control characters removed, then &, < and > escaped."""
    return escape(_clean(s))


def _a(s) -> str:
    """Attribute value, quoted and escaped (includes the surrounding quotes)."""
    return quoteattr(_clean(s))


def _f(x) -> str:
    """A finite number as a short string. A NaN/inf coordinate is a bug, never output."""
    x = float(x)
    if not math.isfinite(x):
        raise ValueError(f"non-finite coordinate {x!r}: refusing to write it into the SVG")
    s = f"{x:.2f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def _fmt_price(v: float) -> str:
    if v >= 1000:
        return f"{v:,.0f}"
    return f"{v:.4g}"


def _fmt_big(v: float) -> str:
    for div, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(v) >= div:
            return f"{v / div:.3g}{suffix}"
    return f"{v:.0f}"


# ------------------------------------------------------------------------------------- scales

class _Axis:
    """Maps a value onto pixel y: ``hi`` at ``top``, ``lo`` at ``bottom`` (log or linear)."""

    def __init__(self, lo: float, hi: float, top: float, bottom: float, log: bool):
        self.lo, self.hi, self.top, self.bottom, self.log = lo, hi, top, bottom, log
        self._a = math.log(lo) if log else lo
        self._b = math.log(hi) if log else hi

    def y(self, v: float) -> float:
        t = math.log(v) if self.log else v
        return self.top + (self._b - t) / (self._b - self._a) * (self.bottom - self.top)


def _axis_range(lo: float, hi: float, log: bool) -> tuple[float, float]:
    """Pad a degenerate range (flat prices), then add a small margin on both ends."""
    if hi <= lo:
        if log or lo > 0:
            lo, hi = lo * 0.95, hi * 1.05
        else:
            pad = 0.05 * abs(lo) if lo != 0 else 1.0
            lo, hi = lo - pad, hi + pad
    if log:
        span = math.log(hi / lo)
        return lo * math.exp(-RANGE_MARGIN * span), hi * math.exp(RANGE_MARGIN * span)
    span = hi - lo
    return lo - RANGE_MARGIN * span, hi + RANGE_MARGIN * span


def _nice_ticks(lo: float, hi: float, target: int = 6) -> list[float]:
    """Round-number ticks (1, 2, 2.5, 5 x 10^k) inside [lo, hi] for a linear axis."""
    span = hi - lo
    if not span > 0:
        return [lo]
    raw = span / target
    mag = 10.0 ** math.floor(math.log10(raw))
    step = mag * 10
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * mag:
            step = m * mag
            break
    k0 = math.ceil(lo / step - 1e-9)
    out, k = [], k0
    while k * step <= hi * (1 + 1e-12) + 1e-12:
        out.append(round(k * step, 10))
        k += 1
        if len(out) > 100:
            break
    return out


def _price_ticks(axis: _Axis, min_gap: float = MIN_TICK_GAP) -> list[float]:
    """Y ticks for a price (or RS) axis. Log axes use 1-2-5 per decade, falling back to denser or
    sparser mantissa sets until the ticks are neither a handful nor a comb; a range too narrow for
    any of them (100 to 120) falls back to linear round numbers."""
    lo, hi = axis.lo, axis.hi
    if not axis.log:
        return _nice_ticks(lo, hi)
    e0, e1 = math.floor(math.log10(lo)), math.ceil(math.log10(hi))
    candidates = []
    for mants in _LOG_MANTISSAS:
        ticks = [float(f"{m}e{e}") for e in range(e0, e1 + 1) for m in mants]
        ticks = [t for t in ticks if lo <= t <= hi]
        if len(ticks) >= 2:
            gap = min(abs(axis.y(a) - axis.y(b)) for a, b in zip(ticks, ticks[1:]))
        else:
            gap = float("inf")
        candidates.append((ticks, gap))
    for ticks, gap in candidates:
        if 4 <= len(ticks) <= 12 and gap >= min_gap:
            return ticks
    for ticks, gap in candidates:
        if 3 <= len(ticks) <= 14 and gap >= min_gap:
            return ticks
    return _nice_ticks(lo, hi)


def _x_ticks(labels: pd.DatetimeIndex, step_x: float) -> list[tuple[int, str, bool]]:
    """(bar index, text, is_major) x ticks. Years once the chart spans 3+ years (thinned to every
    2/5/10 years so the labels never touch); otherwise months, with the year on each January."""
    n = len(labels)
    years = labels.year.to_numpy()
    months = labels.month.to_numpy()
    first_of_year = np.r_[True, years[1:] != years[:-1]]
    if (labels[-1] - labels[0]).days >= 3 * 365:
        # a year is ticked at its first bar, but the first bar of the data only if it IS in January
        ok = first_of_year & ((np.arange(n) > 0) | (months == 1))
        idx = np.flatnonzero(ok)
        for every in (1, 2, 5, 10, 20):
            if step_x * 52 * every >= 34:           # a 4-digit label is about 27 px wide
                break
        return [(int(i), str(int(years[i])), True) for i in idx if int(years[i]) % every == 0]
    first_of_month = np.r_[True, (years[1:] != years[:-1]) | (months[1:] != months[:-1])]
    ok = first_of_month & ((np.arange(n) > 0) | (labels.day.to_numpy() <= 7))
    idx = np.flatnonzero(ok)
    for every in (1, 2, 3, 6, 12):
        if step_x * 4.345 * every >= 52:
            break
    out = []
    for i in idx:
        m = int(months[i])
        if (m - 1) % every:
            continue
        out.append((int(i), f"{_MONTHS[m - 1]} {int(years[i])}" if m == 1 else _MONTHS[m - 1], m == 1))
    return out


# --------------------------------------------------------------------------------- polylines

def _segments(xs, ys) -> list[list[tuple[float, float]]]:
    """Runs of finite points. A missing value ENDS the run (the line is not joined across it)."""
    segs: list[list[tuple[float, float]]] = []
    cur: list[tuple[float, float]] = []
    for x, y in zip(xs, ys):
        if math.isfinite(y):
            cur.append((x, y))
        elif cur:
            segs.append(cur)
            cur = []
    if cur:
        segs.append(cur)
    return segs


def _polylines(cls: str, xs, ys, colour: str, width: float = 1.4, opacity: float = 1.0) -> list[str]:
    out = []
    for seg in _segments(xs, ys):
        if len(seg) == 1:                       # a lone point still shows as a dot (round caps)
            seg = seg * 2
        pts = " ".join(f"{_f(x)},{_f(y)}" for x, y in seg)
        out.append(
            f'<polyline class="{cls}" points="{pts}" fill="none" stroke="{colour}" '
            f'stroke-width="{width}" stroke-opacity="{opacity}" stroke-linejoin="round" '
            f'stroke-linecap="round"/>'
        )
    return out


# ------------------------------------------------------------------------------ marker layout

def _layout_labels(items: list[tuple[float, float]], left: float, right: float,
                   pad: float = LABEL_PAD) -> tuple[list[tuple[float, int]], int]:
    """Place marker labels in lanes (lane 0 is the top row) so no two labels overlap and no
    marker's vertical line runs through another marker's label.

    ``items`` is (marker x, label width). A label starts at its marker's x (pulled left if it would
    run off the right edge). Markers are placed left to right; each takes the first lane where its
    text clears every label already in that lane, no earlier marker's line crosses its text from
    above, and its own line does not cross a label below it. Returns ([(text x, lane)] in input
    order, number of lanes).
    """
    order = sorted(range(len(items)), key=lambda i: items[i][0])
    placed: list[tuple[float, float, float, int]] = []      # text x0, text x1, line x, lane
    result: dict[int, tuple[float, int]] = {}
    max_lane = -1
    for i in order:
        x, w = items[i]
        tx = max(left, min(x, right - w))
        te = tx + w
        lane = 0
        while lane <= max_lane:
            clash = False
            for ptx, pte, px, pl in placed:
                if pl == lane and tx < pte + pad and ptx < te + pad:
                    clash = True                                # text on text
                elif pl < lane and tx - 3 < px < te + 3:
                    clash = True                                # a line from above through this text
                elif pl > lane and ptx - 3 < x < pte + 3:
                    clash = True                                # this line through a label below
                if clash:
                    break
            if not clash:
                break
            lane += 1
        placed.append((tx, te, x, lane))
        result[i] = (tx, lane)
        max_lane = max(max_lane, lane)
    return [result[i] for i in range(len(items))], max_lane + 1


# ------------------------------------------------------------------------------------ chart

def _prepare(bars) -> tuple[pd.DatetimeIndex, dict[str, np.ndarray]]:
    if not isinstance(bars, pd.DataFrame) or len(bars) == 0:
        raise ValueError("bars is empty: there is nothing to draw")
    missing = [k for k in ("open", "high", "low", "close") if k not in bars.columns]
    if missing:
        raise ValueError(f"bars needs open/high/low/close columns; missing {missing}")
    try:
        idx = pd.DatetimeIndex(bars.index)
    except (TypeError, ValueError) as exc:
        raise ValueError("bars must be indexed by week label (dates)") from exc
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    if not (idx.is_monotonic_increasing and idx.is_unique):
        raise ValueError("bars index must be unique and increasing")
    cols = {k: bars[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close")}
    if "volume" in bars.columns:
        cols["volume"] = bars["volume"].to_numpy(dtype="float64")
    return idx, cols


def _overlay(series, idx: pd.DatetimeIndex, positive: bool) -> np.ndarray | None:
    """A caller's overlay on the bars' index as float64; unusable values (and, on a log axis,
    non-positive ones) become NaN. None stays None."""
    if series is None:
        return None
    if isinstance(series, pd.Series):
        s = series.reindex(idx)
        arr = s.to_numpy(dtype="float64")
    else:
        arr = np.asarray(series, dtype="float64")
        if arr.shape != (len(idx),):
            raise ValueError("an overlay given without an index must have one value per bar")
    arr = np.where(np.isfinite(arr), arr, np.nan)
    if positive:
        arr = np.where(arr > 0, arr, np.nan)
    return arr


def weekly_chart_svg(bars: pd.DataFrame, *, sma30=None, sma10m=None, rs_line=None, markers=(),
                     title: str = "", header_lines=(), width: int = 1100, height: int = 660,
                     log_scale: bool = True, candles: bool = True) -> str:
    """One stock's weekly chart as a standalone SVG string.

    ``bars``: DataFrame indexed by week label with open/high/low/close (and volume). ``sma30``,
    ``sma10m`` (already mapped to weeks by the caller) and ``rs_line`` are Series on the same
    index. A bar with a NaN price is skipped. ``markers`` are ``Marker`` objects; see the module
    docstring for snapping. ``height`` is the whole canvas; if the marker label strip is tall
    enough to squeeze the panels under ``MIN_PLOT_H`` pixels the canvas grows instead.
    """
    idx, arr = _prepare(bars)
    n = len(idx)
    width, height = int(width), int(height)
    o, h, lo_, c = arr["open"], arr["high"], arr["low"], arr["close"]
    vol = arr.get("volume")

    ok = np.isfinite(o) & np.isfinite(h) & np.isfinite(lo_) & np.isfinite(c)
    if log_scale:
        ok &= (o > 0) & (h > 0) & (lo_ > 0) & (c > 0)
    if not ok.any():
        raise ValueError("no plottable bar: every bar has a NaN (or, on a log axis, non-positive) price")
    s30 = _overlay(sma30, idx, log_scale)
    s10 = _overlay(sma10m, idx, log_scale)

    # ------------------------------------------------------------ horizontal geometry
    left, right = float(MARGIN_LEFT), float(width - MARGIN_RIGHT)
    plot_w = right - left
    step_x = plot_w / n
    xs = left + (np.arange(n) + 0.5) * step_x

    # ------------------------------------------------------------ markers: snap, then lanes
    snapped: list[tuple[Marker, int]] = []
    dropped = 0
    for m in markers:
        d = pd.Timestamp(m.date)
        if d.tzinfo is not None:
            d = d.tz_localize(None)
        i = int(idx.searchsorted(d, side="right")) - 1
        if i < 0:
            dropped += 1
            continue
        snapped.append((m, i))
    label_w = [LABEL_CHAR_W * len(_clean(m.label)) for m, _ in snapped]
    lanes, n_lanes = _layout_labels([(float(xs[i]), w) for (_, i), w in zip(snapped, label_w)],
                                    left, right)

    # ------------------------------------------------------------ vertical geometry
    header_lines = [str(s) for s in header_lines]
    header_h = 12 + 22 + 16 * len(header_lines) + 6
    strip_h = n_lanes * LANE_H + 6 if n_lanes else 0

    rs_arr = _overlay(rs_line, idx, log_scale)
    has_rs = rs_arr is not None and bool(np.isfinite(rs_arr).any())
    has_vol = vol is not None and bool(np.isfinite(vol).any())
    plot_h = height - header_h - strip_h - AXIS_H
    if plot_h < MIN_PLOT_H:
        height += MIN_PLOT_H - plot_h
        plot_h = MIN_PLOT_H
    panels = [("price", PRICE_SHARE)]
    if has_vol:
        panels.append(("volume", VOLUME_SHARE))
    if has_rs:
        panels.append(("rs", RS_SHARE))
    gap = GAP_SHARE * plot_h
    avail = plot_h - gap * (len(panels) - 1)
    total_share = sum(s for _, s in panels)
    geom: dict[str, tuple[float, float]] = {}
    y = float(header_h + strip_h)
    for name, share in panels:
        ph = avail * share / total_share
        geom[name] = (y, y + ph)
        y += ph + gap
    plot_bottom = geom[panels[-1][0]][1]

    # ------------------------------------------------------------ price axis
    lows = [lo_[ok], o[ok], c[ok]]
    highs = [h[ok], o[ok], c[ok]]
    for s in (s30, s10):
        if s is not None and np.isfinite(s).any():
            lows.append(s[np.isfinite(s)])
            highs.append(s[np.isfinite(s)])
    pmin = float(min(a.min() for a in lows))
    pmax = float(max(a.max() for a in highs))
    plo, phi = _axis_range(pmin, pmax, log_scale)
    p_top, p_bot = geom["price"]
    pax = _Axis(plo, phi, p_top, p_bot, log_scale)

    out: list[str] = []
    out.append(f'<rect width="{width}" height="{height}" fill="#ffffff"/>')

    # ------------------------------------------------------------ header
    out.append(f'<text x="{_f(left)}" y="28" font-size="17" font-weight="700" fill="{INK}">{_t(title)}</text>')
    for k, line in enumerate(header_lines):
        out.append(f'<text x="{_f(left)}" y="{_f(46 + 16 * k)}" font-size="12" fill="{MUTED}">{_t(line)}</text>')
    legend = []
    if s30 is not None and np.isfinite(s30).any():
        legend.append(("30-week SMA", SMA30_COLOUR))
    if s10 is not None and np.isfinite(s10).any():
        legend.append(("10-month SMA", SMA10M_COLOUR))
    lx = right
    for text, colour in reversed(legend):
        tw = 6.2 * len(text)
        out.append(f'<text x="{_f(lx)}" y="26" font-size="11" text-anchor="end" fill="{MUTED}">{_t(text)}</text>')
        out.append(f'<line x1="{_f(lx - tw - 24)}" y1="22" x2="{_f(lx - tw - 6)}" y2="22" stroke="{colour}" '
                   f'stroke-width="2.4"/>')
        lx -= tw + 36

    # ------------------------------------------------------------ x grid and panel frames
    ticks = _x_ticks(idx, step_x)
    for i, _, major in ticks:
        out.append(f'<line x1="{_f(xs[i])}" y1="{_f(p_top)}" x2="{_f(xs[i])}" y2="{_f(plot_bottom)}" '
                   f'stroke="{GRID_MAJOR if major else GRID}" stroke-width="1"/>')
    for name, (t, b) in geom.items():
        out.append(f'<rect class="frame frame-{name}" x="{_f(left)}" y="{_f(t)}" width="{_f(plot_w)}" '
                   f'height="{_f(b - t)}" fill="none" stroke="{FRAME}" stroke-width="1"/>')

    # ------------------------------------------------------------ price panel: y grid, candles
    for tv in _price_ticks(pax):
        ty = pax.y(tv)
        out.append(f'<line x1="{_f(left)}" y1="{_f(ty)}" x2="{_f(right)}" y2="{_f(ty)}" stroke="{GRID}" '
                   f'stroke-width="1"/>')
        out.append(f'<text class="ytick" x="{_f(left - 6)}" y="{_f(ty + 3.5)}" font-size="10" '
                   f'text-anchor="end" fill="{MUTED}">{_t(_fmt_price(tv))}</text>')
    out.append(f'<text x="{_f(left + 6)}" y="{_f(p_top + 13)}" font-size="10" fill="{MUTED}">'
               f'Price ({"log" if log_scale else "linear"} scale, weekly)</text>')

    close_y = np.full(n, np.nan)
    for i in np.flatnonzero(ok):
        close_y[i] = pax.y(c[i])
    if candles:
        body_w = max(1.0, min(step_x * 0.72, 10.0))
        stroke_w = 1.0 if body_w >= 3 else 0.7
        ups: list[str] = []
        downs: list[str] = []
        for i in np.flatnonzero(ok):
            x = xs[i]
            yt, yb = pax.y(max(o[i], c[i])), pax.y(min(o[i], c[i]))
            wick = f'<line x1="{_f(x)}" y1="{_f(pax.y(h[i]))}" x2="{_f(x)}" y2="{_f(pax.y(lo_[i]))}"/>'
            body = (f'<rect x="{_f(x - body_w / 2)}" y="{_f(yt)}" width="{_f(body_w)}" '
                    f'height="{_f(max(yb - yt, 0.8))}"/>')
            (ups if c[i] >= o[i] else downs).append(wick + body)
        up_fill = "#ffffff" if body_w >= 3 else UP_COLOUR
        out.append(f'<g class="candles up" stroke="{UP_COLOUR}" fill="{up_fill}" stroke-width="{stroke_w}">'
                   + "".join(ups) + "</g>")
        out.append(f'<g class="candles down" stroke="{DOWN_COLOUR}" fill="{DOWN_COLOUR}" '
                   f'stroke-width="{stroke_w}">' + "".join(downs) + "</g>")
    else:
        out.extend(_polylines("close", xs, close_y, INK, 1.4))
    for cls, s, colour in (("sma10m", s10, SMA10M_COLOUR), ("sma30", s30, SMA30_COLOUR)):
        if s is not None:
            sy = [pax.y(v) if math.isfinite(v) else math.nan for v in s]
            out.extend(_polylines(cls, xs, sy, colour, 1.6, 0.95))

    # last close tag on the right axis
    last = int(np.flatnonzero(ok)[-1])
    out.append(f'<g class="last-close"><rect x="{_f(right + 2)}" y="{_f(close_y[last] - 8)}" width="'
               f'{_f(MARGIN_RIGHT - 6)}" height="16" rx="2" fill="{INK}"/>'
               f'<text x="{_f(right + 2 + (MARGIN_RIGHT - 6) / 2)}" y="{_f(close_y[last] + 4)}" font-size="10.5" '
               f'text-anchor="middle" fill="#ffffff">{_t(_fmt_price(c[last]))}</text></g>')

    # ------------------------------------------------------------ volume panel
    if has_vol:
        v_top, v_bot = geom["volume"]
        vmax = float(np.nanmax(np.where(np.isfinite(vol), vol, np.nan)))
        vmax = vmax if vmax > 0 else 1.0
        vax = _Axis(0.0, vmax * 1.05, v_top, v_bot, False)
        bw = max(0.8, step_x * 0.8)
        vu: list[str] = []
        vd: list[str] = []
        for i in range(n):
            if not (math.isfinite(vol[i]) and vol[i] > 0 and ok[i]):
                continue
            yt = vax.y(vol[i])
            r = f'<rect x="{_f(xs[i] - bw / 2)}" y="{_f(yt)}" width="{_f(bw)}" height="{_f(v_bot - yt)}"/>'
            (vu if c[i] >= o[i] else vd).append(r)
        out.append(f'<g class="volume up" fill="{UP_COLOUR}" fill-opacity="0.55">' + "".join(vu) + "</g>")
        out.append(f'<g class="volume down" fill="{DOWN_COLOUR}" fill-opacity="0.55">' + "".join(vd) + "</g>")
        out.append(f'<text x="{_f(left + 6)}" y="{_f(v_top + 12)}" font-size="10" fill="{MUTED}">Volume (shares per week)</text>')
        out.append(f'<text class="ytick" x="{_f(left - 6)}" y="{_f(v_top + 10)}" font-size="10" text-anchor="end" '
                   f'fill="{MUTED}">{_t(_fmt_big(vmax))}</text>')

    # ------------------------------------------------------------ relative strength panel
    if has_rs:
        r_top, r_bot = geom["rs"]
        fin = rs_arr[np.isfinite(rs_arr)]
        rlo, rhi = _axis_range(float(fin.min()), float(fin.max()), log_scale)
        rax = _Axis(rlo, rhi, r_top, r_bot, log_scale)
        for tv in _price_ticks(rax, 13):
            ty = rax.y(tv)
            out.append(f'<line x1="{_f(left)}" y1="{_f(ty)}" x2="{_f(right)}" y2="{_f(ty)}" stroke="{GRID}" '
                       f'stroke-width="1"/>')
            out.append(f'<text class="ytick" x="{_f(left - 6)}" y="{_f(ty + 3.5)}" font-size="10" '
                       f'text-anchor="end" fill="{MUTED}">{_t(_fmt_price(tv))}</text>')
        ry = [rax.y(v) if math.isfinite(v) else math.nan for v in rs_arr]
        out.extend(_polylines("rs", xs, ry, RS_COLOUR, 1.5))
        out.append(f'<text x="{_f(left + 6)}" y="{_f(r_top + 12)}" font-size="10" fill="{MUTED}">'
                   f'Relative strength (stock / benchmark, {"log" if log_scale else "linear"})</text>')

    # ------------------------------------------------------------ x axis labels
    for i, text, major in ticks:
        out.append(f'<line x1="{_f(xs[i])}" y1="{_f(plot_bottom)}" x2="{_f(xs[i])}" y2="{_f(plot_bottom + 4)}" '
                   f'stroke="{FRAME}"/>')
        out.append(f'<text class="xtick" x="{_f(xs[i])}" y="{_f(plot_bottom + 17)}" font-size="11" '
                   f'text-anchor="middle" fill="{INK if major else MUTED}">{_t(text)}</text>')

    # ------------------------------------------------------------ markers
    # Drawn top lane first, so a line that has to cross a lower label (only when a long label is
    # pulled in from the right edge) is under that label's white halo and the text stays readable.
    strip_top = header_h
    draw_order = sorted(range(len(snapped)), key=lambda k: (lanes[k][1], snapped[k][1]))
    for k in draw_order:
        (m, i), (tx, lane) = snapped[k], lanes[k]
        kind = _clean(m.kind) if m.kind in MARKER_COLOURS else "event"
        colour = MARKER_COLOURS[kind]
        css_kind = _KIND_SAFE.sub("_", _clean(m.kind)) or "event"
        base_y = strip_top + 12 + lane * LANE_H
        d = pd.Timestamp(m.date)
        out.append(f'<g class="marker marker-{css_kind}" data-kind={_a(m.kind)} data-date="{d.date().isoformat()}" '
                   f'data-bar="{idx[i].date().isoformat()}">')
        out.append(f'<title>{_t(m.label)} {d.date().isoformat()}</title>')
        out.append(f'<line class="marker-line" x1="{_f(xs[i])}" y1="{_f(base_y + 3)}" x2="{_f(xs[i])}" '
                   f'y2="{_f(p_bot)}" stroke="{colour}" stroke-width="1.2" stroke-dasharray="4 3" '
                   f'stroke-opacity="0.8"/>')
        if math.isfinite(close_y[i]):
            out.append(f'<circle cx="{_f(xs[i])}" cy="{_f(close_y[i])}" r="4.2" fill="{colour}" stroke="#ffffff" '
                       f'stroke-width="1.5"/>')
        out.append(f'<rect class="label-bg" x="{_f(tx - 2)}" y="{_f(base_y - 11)}" '
                   f'width="{_f(LABEL_CHAR_W * len(_clean(m.label)) + 4)}" height="14" rx="2" fill="#ffffff" '
                   f'fill-opacity="0.92"/>')
        out.append(f'<text class="marker-label" x="{_f(tx)}" y="{_f(base_y)}" font-size="{LABEL_FONT_PX}" '
                   f'font-weight="600" fill="{colour}" stroke="#ffffff" stroke-width="3" '
                   f'paint-order="stroke">{_t(m.label)}</text>')
        out.append("</g>")

    comment = f"<!-- dropped {dropped} marker(s) dated before the first bar -->" if dropped else ""
    head = (f'<svg xmlns="{SVG_NS}" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
            f'font-family="{FONT}" role="img" aria-label={_a(title or "weekly chart")}>')
    return head + f"<title>{_t(title or 'weekly chart')}</title>" + comment + "".join(out) + "</svg>"


# ------------------------------------------------------------------------------------ files

def write_svg(path, svg: str) -> Path:
    """Write ``svg`` (UTF-8), creating parent directories. Returns the path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(svg, encoding="utf-8")
    return p


_INDEX_CSS = (
    "body{font-family:-apple-system,'Helvetica Neue',Helvetica,Arial,sans-serif;max-width:1180px;"
    "margin:24px auto;padding:0 16px;color:#1f2933;line-height:1.45}"
    "h1{font-size:24px;margin:0 0 8px}h2{font-size:19px;margin:0 0 6px}"
    "section{margin:28px 0;padding-top:12px;border-top:1px solid #d3d8df}"
    "nav a{display:inline-block;margin:2px 10px 2px 0}ul{margin:4px 0 10px;padding-left:20px}"
    "img{max-width:100%;height:auto;border:1px solid #e1e5ea}p.intro{margin:3px 0;color:#52606d}"
)


def write_html_index(path, entries, title: str, intro_lines=()) -> Path:
    """Write a static HTML5 page with one section per entry (no JavaScript).

    ``entries``: list of dict(symbol, svg_file (relative to the page), summary_lines list[str]).
    Each section has an <h2>, a <ul> of the summary lines and the chart as <img loading="lazy">.
    """
    entries = list(entries)
    parts = [
        "<!DOCTYPE html>", '<html lang="en">', "<head>", '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{_t(title)}</title>", f"<style>{_INDEX_CSS}</style>", "</head>", "<body>",
        f"<h1>{_t(title)}</h1>",
    ]
    parts.extend(f'<p class="intro">{_t(line)}</p>' for line in intro_lines)
    if entries:
        parts.append("<nav>" + "".join(
            f'<a href="#e{i}">{_t(e["symbol"])}</a>' for i, e in enumerate(entries)) + "</nav>")
    for i, e in enumerate(entries):
        parts.append(f'<section id="e{i}">')
        parts.append(f'<h2>{_t(e["symbol"])}</h2>')
        parts.append("<ul>" + "".join(f"<li>{_t(s)}</li>" for s in e.get("summary_lines", ())) + "</ul>")
        parts.append(f'<img src={_a(quote(str(e["svg_file"]), safe="/"))} loading="lazy" '
                     f'alt={_a(str(e["symbol"]) + " weekly chart")}>')
        parts.append("</section>")
    parts.extend(["</body>", "</html>", ""])
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(parts), encoding="utf-8")
    return p
