"""
A Finviz-style screener over daily bars: named filters with dropdown-style options.

``indicators(daily)`` turns one stock's daily bars into a fixed table of columns
(``COLUMNS``). ``Screen({"Price": "Over $5", "52-Week High": "0-5% below"})`` turns named
filters into boolean masks over that table. The filter names and option labels follow the
Finviz screener, so a screen can be written the way it would be clicked together.

THE NO-LOOK-AHEAD RULE
----------------------
Every column at row t is a function of bars 0..t only. Nothing is centred, nothing is shifted
backwards, and every rolling window is NaN until its own window is full (``min_periods`` equals
the window). ``tests/test_setup_engine/test_screen.py`` rebuilds the table from data cut at
several dates and requires every column, and every filter mask, to be unchanged.
Two choices keep the trigger bar from helping itself:
* ``avg_vol20`` and ``dollar_vol20`` use the PRIOR 20 bars (shifted by one), so a volume spike
  cannot lift itself over a liquidity filter, and ``relvol`` is today's volume over that prior
  average.
* "New High" / "New Low" compare the close with the ``prev_high_*`` / ``prev_low_*`` columns
  (the extreme of the window ending yesterday). ``high_*`` itself includes today, which is
  what ``pct_below_high_*`` wants (a stock at its high is 0% below it).

NaN NEVER MATCHES
-----------------
A filter is a conjunction of comparisons, and ``Screen.mask`` also requires every column the
rules use to be finite. No option is implemented as the negation of another ("Under" is not
"not Over"), because negation turns NaN into True and a warm-up bar would pass the screen.

THE ORDER MODEL
---------------
None here. A screen only says WHICH bars qualify; fills, stops and costs belong to
``setup_engine.trade_sim``. A screen evaluated at the close of bar t can be acted on from the
open of bar t+1 at the earliest.

KNOWN BIASES (not fixable here, reported alongside every result)
----------------------------------------------------------------
Survivorship (a universe of listings that still exist flatters every screen), Yahoo's
retroactive split adjustment (a reverse-split stock shows a price history it never traded at,
which distorts price and dollar-volume bands), and no dividend adjustment (closes are not total
return). Indicator quirks inherited for exact parity with ``IndicatorCalculator``: RSI is NaN
(not 100) while the average loss is exactly zero, and the first bar's NaN change counts as zero
gain and loss.

CALENDAR-FREE STAND-INS
-----------------------
"Monthly Low: Higher low" has no calendar: it compares the lowest low of the last 126 bars with
the lowest low of the 252 bars before those (bars t-377..t-126).

MULTI-TIMEFRAME FILTERS
-----------------------
``MTF_FILTERS`` is a SEPARATE vocabulary (weekly and monthly trend, weekly highs and volume,
relative strength against a benchmark, cross-sectional performance rank). It reads columns that
``indicators()`` does not produce; they come from ``setup_engine/mtf.py`` (``indicators_mtf``
for a daily table, ``mtf_features`` for a weekly one, ``add_xs_ranks`` for the ranks). The
default ``FILTERS`` / ``available_filters()`` are unchanged; ``available_filters(include_mtf=True)``
lists both. A ``Screen`` mixes the two freely, and raises a ValueError naming the missing columns
when the table has no MTF columns. THE COMPLETED-BAR RULE: a weekly (monthly) column describes the
last COMPLETED weekly (monthly) bar, so a bar still in progress is never read; NaN never matches,
exactly as above (no option is the negation of another).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

LOW_RECENT_BARS = 126
LOW_PRIOR_BARS = 252
EPS = 1e-9

# window key -> bars (high_<key>, low_<key>, prev_high_<key>, ...)
WINDOWS = {"20": 20, "50": 50, "52w": 252, "2y": 504}
# bars back for the performance columns
PERF_BARS = {"week": 5, "month": 21, "quarter": 63, "half": 126, "year": 252}

COLUMNS = (
    "open", "high", "low", "close", "volume",
    "prev_close", "change", "gap",
    "sma20", "sma50", "sma200", "prev_sma20", "prev_sma50", "prev_sma200",
    "rsi14", "atr14", "atr_pct",
    "avg_vol20", "relvol", "dollar_vol20",
    "high_20", "high_50", "high_52w", "high_2y",
    "low_20", "low_50", "low_52w", "low_2y",
    "prev_high_20", "prev_high_50", "prev_high_52w", "prev_high_2y",
    "prev_low_20", "prev_low_50", "prev_low_52w", "prev_low_2y",
    "pct_below_high_20", "pct_below_high_50", "pct_below_high_52w", "pct_below_high_2y",
    "pct_above_low_20", "pct_above_low_50", "pct_above_low_52w", "pct_above_low_2y",
    "perf_week", "perf_month", "perf_quarter", "perf_half", "perf_year",
    "low_recent", "low_prior", "higher_low",
)

_OHLCV = ("open", "high", "low", "close", "volume")


# ── Data cleaning and indicators ──────────────────────────────────────────────

def clean_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate dates (keep last), sort, drop impossible bars, zero-fill missing volume.

    A verbatim copy of ``scripts/build_setup_dataset.py::clean`` so the screener and the
    setup dataset treat bad bars identically.
    """
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[(df["close"] > 0) & (df["open"] > 0) & (df["high"] >= df["low"]) & (df["low"] > 0)]
    return df.dropna(subset=["open", "high", "low", "close"]).fillna({"volume": 0.0})


def _as_indexed(daily: pd.DataFrame) -> pd.DataFrame:
    df = daily
    if not isinstance(df.index, pd.DatetimeIndex) and "date" in df.columns:
        df = df.set_index(pd.to_datetime(df["date"])).drop(columns="date")
    missing = [c for c in _OHLCV if c not in df.columns]
    if missing:
        raise ValueError(f"daily bars are missing columns {missing}")
    if not (df.index.is_unique and df.index.is_monotonic_increasing):
        raise ValueError("daily bars must have a unique, increasing index (run clean_daily first)")
    return df


def indicators(daily: pd.DataFrame) -> pd.DataFrame:
    """One stock's daily bars -> the ``COLUMNS`` table (same index, float64, inf replaced by NaN).

    ``daily`` needs open/high/low/close/volume and a DatetimeIndex (or a ``date`` column).
    Row t uses bars 0..t only; every rolling window is NaN until it is full.
    """
    df = _as_indexed(daily)
    o, h, l, c, v = (df[k].astype("float64") for k in _OHLCV)
    out: dict[str, pd.Series] = {"open": o.copy(), "high": h.copy(), "low": l.copy(),
                                 "close": c.copy(), "volume": v.copy()}

    prev_close = c.shift(1)
    out["prev_close"] = prev_close
    out["change"] = c / prev_close - 1.0
    out["gap"] = o / prev_close - 1.0

    for n in (20, 50, 200):
        sma = c.rolling(n, min_periods=n).mean()
        out[f"sma{n}"] = sma
        out[f"prev_sma{n}"] = sma.shift(1)

    # RSI(14): identical to IndicatorCalculator.compute_rsi (Wilder smoothing)
    delta = c.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["rsi14"] = 100.0 - (100.0 / (1.0 + rs))

    # ATR(14): identical to IndicatorCalculator.compute_atr (row-wise max skips NaN)
    tr = pd.concat([h - l, (h - prev_close).abs(), (l - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / 14, min_periods=14, adjust=False).mean()
    out["atr14"] = atr
    out["atr_pct"] = atr / c

    # volume: PRIOR 20 bars, so today's volume cannot help it pass a liquidity filter
    avg_vol = v.rolling(20, min_periods=20).mean().shift(1)
    out["avg_vol20"] = avg_vol
    out["relvol"] = v / avg_vol
    out["dollar_vol20"] = (c * v).rolling(20, min_periods=20).mean().shift(1)

    for key, w in WINDOWS.items():
        hi = h.rolling(w, min_periods=w).max()
        lo = l.rolling(w, min_periods=w).min()
        out[f"high_{key}"] = hi
        out[f"low_{key}"] = lo
        out[f"prev_high_{key}"] = hi.shift(1)
        out[f"prev_low_{key}"] = lo.shift(1)
        out[f"pct_below_high_{key}"] = 1.0 - c / hi
        out[f"pct_above_low_{key}"] = c / lo - 1.0

    for name, k in PERF_BARS.items():
        out[f"perf_{name}"] = c / c.shift(k) - 1.0

    low_recent = l.rolling(LOW_RECENT_BARS, min_periods=LOW_RECENT_BARS).min()
    low_prior = l.rolling(LOW_PRIOR_BARS, min_periods=LOW_PRIOR_BARS).min().shift(LOW_RECENT_BARS)
    out["low_recent"] = low_recent
    out["low_prior"] = low_prior
    lr, lp = low_recent.to_numpy(), low_prior.to_numpy()
    out["higher_low"] = pd.Series(
        np.where(np.isnan(lr) | np.isnan(lp), np.nan, (lr > lp).astype("float64")), index=df.index)

    table = pd.DataFrame(out, index=df.index)[list(COLUMNS)].astype("float64")
    return table.replace([np.inf, -np.inf], np.nan)


# ── Option grammar ────────────────────────────────────────────────────────────

_NUM = r"\$?(-?\d+(?:\.\d+)?)\s*([KMB])?"      # 2 groups: value, K/M/B suffix
_PCT = r"(-?\d+(?:\.\d+)?)\s*%"                # 1 group; the % sign is required
_MULT = {"K": 1e3, "M": 1e6, "B": 1e9}
_TOKEN_RE = re.compile(r"(<N>|<P>)")


def _norm(text: str) -> str:
    return " ".join(str(text).split()).lower()


class _Form:
    """One option pattern, e.g. ``Over <N>`` -> ("gt", column, N). Matching is case-insensitive.

    ``<N>`` is a number with an optional ``$`` and K/M/B suffix; ``<P>`` is a percentage whose
    ``%`` is required (value N/100). ``nonneg`` rejects negative percentages (``Down -5%``).
    """

    def __init__(self, template, make, nonneg=False):
        self.template, self.make, self.nonneg = template, make, nonneg
        pieces, self.tokens = [], []
        for part in _TOKEN_RE.split(template):
            if part == "<N>":
                pieces.append(_NUM)
                self.tokens.append("N")
            elif part == "<P>":
                pieces.append(_PCT)
                self.tokens.append("P")
            else:
                pieces.append(re.escape(part.lower()))
        self.regex = re.compile("".join(pieces), re.IGNORECASE)

    def parse(self, text: str):
        m = self.regex.fullmatch(text)
        if m is None:
            return None
        vals, g = [], 1
        for t in self.tokens:
            if t == "N":
                val = float(m.group(g))
                if m.group(g + 1):
                    val *= _MULT[m.group(g + 1).upper()]
                g += 2
            else:
                val = float(m.group(g)) / 100.0
                g += 1
            vals.append(val)
        if self.nonneg and any(x < 0 for x in vals):
            return None
        spec = self.make(vals)
        if spec[0] == "between" and spec[2] > spec[3]:
            raise ValueError(f"option {text!r}: the lower bound is above the upper bound")
        return spec


def _levels(col: str, pct: bool = False) -> list:
    t = "<P>" if pct else "<N>"
    return [_Form(f"Over {t}", lambda v, c=col: ("gt", c, v[0])),
            _Form(f"Under {t}", lambda v, c=col: ("lt", c, v[0]))]


def _with_range(col: str) -> list:
    return _levels(col) + [_Form("<N> to <N>", lambda v, c=col: ("between", c, v[0], v[1]))]


def _moves(col: str) -> list:
    return [_Form("Up", lambda v, c=col: ("gt", c, 0.0)),
            _Form("Down", lambda v, c=col: ("lt", c, 0.0)),
            _Form("Up <P>", lambda v, c=col: ("ge", c, v[0]), nonneg=True),
            _Form("Down <P>", lambda v, c=col: ("le", c, -v[0]), nonneg=True)]


def _perf(col: str) -> list:
    return [_Form("Up", lambda v, c=col: ("gt", c, 0.0)),
            _Form("Down", lambda v, c=col: ("lt", c, 0.0)),
            _Form("Over <P>", lambda v, c=col: ("gt", c, v[0])),
            _Form("Under <P>", lambda v, c=col: ("lt", c, v[0]))]


def _sma(k: int) -> list:
    me, pme = f"sma{k}", f"prev_sma{k}"
    forms = [
        _Form(f"Price above SMA{k}", lambda v: ("gt_col", "close", me)),
        _Form(f"Price below SMA{k}", lambda v: ("lt_col", "close", me)),
        _Form(f"Price crossed above SMA{k}", lambda v: ("cross_above", "close", me, "prev_close", pme)),
        _Form(f"Price crossed below SMA{k}", lambda v: ("cross_below", "close", me, "prev_close", pme)),
        _Form(f"Price <P> above SMA{k}", lambda v: ("ge_ratio", "close", me, 1.0 + v[0]), nonneg=True),
        _Form(f"Price <P> below SMA{k}", lambda v: ("le_ratio", "close", me, 1.0 - v[0]), nonneg=True),
    ]
    for j in (20, 50, 200):
        if j == k:
            continue
        other, pother = f"sma{j}", f"prev_sma{j}"
        forms += [
            _Form(f"SMA{k} above SMA{j}", lambda v, o=other: ("gt_col", me, o)),
            _Form(f"SMA{k} below SMA{j}", lambda v, o=other: ("lt_col", me, o)),
            _Form(f"SMA{k} crossed above SMA{j}",
                  lambda v, o=other, po=pother: ("cross_above", me, o, pme, po)),
            _Form(f"SMA{k} crossed below SMA{j}",
                  lambda v, o=other, po=pother: ("cross_below", me, o, pme, po)),
        ]
    return forms


def _highs(key: str) -> list:
    return [_Form("New High", lambda v: ("gt_col", "close", f"prev_high_{key}")),
            _Form("<P> or more below", lambda v: ("ge", f"pct_below_high_{key}", v[0]), nonneg=True),
            _Form("0-<P> below", lambda v: ("between", f"pct_below_high_{key}", 0.0, v[0]), nonneg=True)]


def _lows(key: str) -> list:
    return [_Form("New Low", lambda v: ("lt_col", "close", f"prev_low_{key}")),
            _Form("<P> or more above", lambda v: ("ge", f"pct_above_low_{key}", v[0]), nonneg=True),
            _Form("0-<P> above", lambda v: ("between", f"pct_above_low_{key}", 0.0, v[0]), nonneg=True)]


def _under(*xs, pre="", post=""):
    return [f"Under {pre}{x}{post}" for x in xs]


def _over(*xs, pre="", post=""):
    return [f"Over {pre}{x}{post}" for x in xs]


def _sma_examples(k: int) -> list:
    out = [f"Price above SMA{k}", f"Price below SMA{k}", f"Price crossed above SMA{k}",
           f"Price crossed below SMA{k}", f"Price 10% above SMA{k}", f"Price 10% below SMA{k}"]
    for j in (20, 50, 200):
        if j != k:
            out += [f"SMA{k} above SMA{j}", f"SMA{k} below SMA{j}",
                    f"SMA{k} crossed above SMA{j}", f"SMA{k} crossed below SMA{j}"]
    return out


_PERF_EXAMPLES = ["Up", "Down"] + _over(10, 20, 50, 100, post="%") + _under(-10, -20, -50, -75, post="%")
_MOVE_EXAMPLES = ["Up", "Down", "Up 2%", "Up 5%", "Up 10%", "Up 20%", "Down 2%", "Down 5%", "Down 10%"]

# filter name -> (forms, example labels). FILTERS is built by PARSING the examples with the
# forms, so the dropdown and the parser cannot drift apart.
_DEFS: dict[str, tuple[list, list]] = {
    "Price": (_with_range("close"),
              _under(1, 2, 5, 10, 20, 50, pre="$") + _over(1, 2, 5, 10, 20, 50, 100, pre="$")
              + ["$1 to $5", "$1 to $10", "$5 to $20", "$10 to $50"]),
    "Average Volume": (_with_range("avg_vol20"),
                       _under("100K", "500K", "1M") + _over("100K", "200K", "500K", "1M", "2M")
                       + ["100K to 1M"]),
    "Current Volume": (_with_range("volume"),
                       _under("100K") + _over("100K", "500K", "1M", "5M")),
    "Dollar Volume 20d": (_levels("dollar_vol20"),
                          _under("1M") + _over("1M", "5M", "10M", "50M")),
    "Relative Volume": (_levels("relvol"),
                        _over(10, 5, 3, 2, 1.5, 1) + _under(1, 0.5)),
    "RSI (14)": (_levels("rsi14") + [
                     _Form("Overbought (<N>)", lambda v: ("gt", "rsi14", v[0])),
                     _Form("Oversold (<N>)", lambda v: ("lt", "rsi14", v[0]))],
                 [f"Overbought ({x})" for x in (80, 70, 60)]
                 + [f"Oversold ({x})" for x in (40, 30, 20)] + ["Over 50", "Under 50"]),
    "ATR %": (_levels("atr_pct", pct=True),
              _over(2, 3, 5, 8, post="%") + _under(2, 5, post="%")),
    "Average True Range": (_levels("atr14"),
                           _over(0.25, 0.5, 1, 2) + _under(0.5, 1)),
    "Change": (_moves("change"), list(_MOVE_EXAMPLES)),
    "Gap": (_moves("gap"), [x for x in _MOVE_EXAMPLES if x not in ("Up 20%",)]),
}
for _label, _name in (("Week", "week"), ("Month", "month"), ("Quarter", "quarter"),
                      ("Half Year", "half"), ("Year", "year")):
    _DEFS[f"Performance ({_label})"] = (_perf(f"perf_{_name}"), list(_PERF_EXAMPLES))
for _k in (20, 50, 200):
    _DEFS[f"SMA{_k}"] = (_sma(_k), _sma_examples(_k))
for _label, _key in (("20-Day", "20"), ("50-Day", "50"), ("52-Week", "52w"), ("2-Year", "2y")):
    _DEFS[f"{_label} High"] = (_highs(_key), ["New High", "0-3% below", "0-5% below", "0-10% below"]
                               + [f"{x}% or more below" for x in (10, 20, 30, 50, 60, 70, 80, 90)])
for _label, _key in (("20-Day", "20"), ("50-Day", "50"), ("52-Week", "52w"), ("2-Year", "2y")):
    _DEFS[f"{_label} Low"] = (_lows(_key), ["New Low", "0-3% above", "0-5% above", "0-10% above"]
                              + [f"{x}% or more above" for x in (10, 20, 50, 100, 200)])
_DEFS["Monthly Low"] = ([_Form("Higher low", lambda v: ("gt_col", "low_recent", "low_prior")),
                         _Form("Lower low", lambda v: ("lt_col", "low_recent", "low_prior"))],
                        ["Higher low", "Lower low"])



# ── Multi-timeframe vocabulary (columns come from setup_engine/mtf.py) ────────
# A SEPARATE registry: these filters read columns that ``indicators()`` does not produce, so
# merging them into ``_DEFS`` / ``FILTERS`` would make every consumer that walks the daily
# vocabulary (the dropdown, the "NaN never matches" sweep) fail on a table without them.

def _price_vs_line(close: str, line: str, prev_close: str, prev_line: str, pct_forms: bool = True) -> list:
    forms = [
        _Form("Price above", lambda v: ("gt_col", close, line)),
        _Form("Price below", lambda v: ("lt_col", close, line)),
        _Form("Price crossed above", lambda v: ("cross_above", close, line, prev_close, prev_line)),
        _Form("Price crossed below", lambda v: ("cross_below", close, line, prev_close, prev_line)),
    ]
    if pct_forms:
        forms += [
            _Form("Price <P> above", lambda v: ("ge_ratio", close, line, 1.0 + v[0]), nonneg=True),
            _Form("Price <P> below", lambda v: ("le_ratio", close, line, 1.0 - v[0]), nonneg=True),
        ]
    return forms


def _weekly_highs(k: int) -> list:
    """Weekly close vs the highest weekly close: ``w_dd<k>`` is <= 0, so "30% below" is <= -0.3."""
    return [_Form("New High", lambda v: ("gt_col", "w_close", f"w_prev_hi{k}")),
            _Form("<P> or more below", lambda v: ("le", f"w_dd{k}", -v[0]), nonneg=True),
            _Form("0-<P> below", lambda v: ("between", f"w_dd{k}", -v[0], 0.0), nonneg=True)]


def _volume_multiple(col: str) -> list:
    return [_Form("Over <N>x 10-week average", lambda v: ("gt", col, v[0])),
            _Form("Under <N>x 10-week average", lambda v: ("lt", col, v[0]))]


def _rank_forms(col: str) -> list:
    """``xs_rank_ret*`` is a percentile in (0, 1], 1.0 = best: "Top 20%" is >= 0.8."""
    return [_Form("Top <P>", lambda v: ("ge", col, 1.0 - v[0]), nonneg=True),
            _Form("Bottom <P>", lambda v: ("le", col, v[0]), nonneg=True)]


_PRICE_EXAMPLES = ["Price above", "Price below", "Price crossed above", "Price crossed below"]

_MTF_DEFS: dict[str, tuple[list[_Form], list[str]]] = {
    "Weekly SMA30": (_price_vs_line("w_close", "w_sma30", "w_prev_close", "w_prev_sma30"),
                     _PRICE_EXAMPLES + ["Price 10% above", "Price 10% below"]),
    "Weekly SMA30 Slope": ([_Form("Rising", lambda v: ("gt", "w_slope30", 0.0)),
                            _Form("Falling", lambda v: ("lt", "w_slope30", 0.0))],
                           ["Rising", "Falling"]),
    "Monthly SMA10": (_price_vs_line("m_close", "m_sma10", "m_prev_close", "m_prev_sma10", pct_forms=False),
                      list(_PRICE_EXAMPLES)),
}
for _k in (26, 52):
    _MTF_DEFS[f"Weekly High ({_k}w)"] = (
        _weekly_highs(_k),
        ["New High", "0-5% below", "0-10% below"] + [f"{x}% or more below" for x in (10, 20, 30, 40, 50, 60)])
_MTF_DEFS["Weekly Volume"] = (_volume_multiple("w_volx"),
                              _over(1.5, 2, 3, post="x 10-week average")
                              + _under(1, post="x 10-week average"))
_MTF_DEFS["Weekly RSI (14)"] = (_levels("w_rsi14"), _over(50, 60, 70) + _under(30, 40, 50))
_MTF_DEFS["Monthly RSI (14)"] = (_levels("m_rsi14"), _over(50, 60, 70) + _under(30, 40, 50))
_MTF_DEFS["Weekly Low"] = ([_Form("Higher low", lambda v: ("gt_col", "w_low13", "w_low_prior")),
                            _Form("Lower low", lambda v: ("lt_col", "w_low13", "w_low_prior"))],
                           ["Higher low", "Lower low"])
_MTF_DEFS["RS vs Benchmark (26w)"] = (_levels("rs_ret26", pct=True),
                                      _over(0, 10, 25, 50, post="%") + _under(0, -25, post="%"))
_MTF_DEFS["RS Line"] = ([_Form("New 26-week high", lambda v: ("gt_col", "rs_line", "rs_prev_hi26")),
                         _Form("New 52-week high", lambda v: ("gt_col", "rs_line", "rs_prev_hi52")),
                         _Form("New 26-week high before price", lambda v: ("gt", "rs_lead26", 0.5))],
                        ["New 26-week high", "New 52-week high", "New 26-week high before price"])
_MTF_DEFS["Benchmark Weekly SMA30"] = ([_Form("Price above", lambda v: ("gt", "bm_w_px_sma30", 0.0)),
                                        _Form("Price below", lambda v: ("lt", "bm_w_px_sma30", 0.0))],
                                       ["Price above", "Price below"])
_MTF_DEFS["Benchmark Monthly SMA10"] = ([_Form("Price above", lambda v: ("gt", "bm_m_px_sma10", 0.0)),
                                         _Form("Price below", lambda v: ("lt", "bm_m_px_sma10", 0.0))],
                                        ["Price above", "Price below"])
for _n in (13, 26):
    _MTF_DEFS[f"Performance Rank ({_n} Weeks)"] = (_rank_forms(f"xs_rank_ret{_n}"),
                                                   ["Top 10%", "Top 20%", "Top 50%", "Bottom 20%"])

_ALL_DEFS: dict[str, tuple[list, list]] = {**_DEFS, **_MTF_DEFS}

_FILTER_KEYS = {_norm(name): name for name in _ALL_DEFS}
_MONTHLY_LOW_NOTE = "(calendar-free stand-in: lowest low of last 126 bars vs the 252 before)"

_WEEKLY_NOTE = "(last COMPLETED weekly bar)"
_MONTHLY_NOTE = "(last COMPLETED monthly bar)"
_BENCH_NOTE = "(needs a benchmark; NaN never matches)"
_RANK_NOTE = "(cross-sectional rank among liquid stocks that week, added by the study)"
_MTF_NOTES = {
    **{name: _WEEKLY_NOTE for name in (
        "Weekly SMA30", "Weekly SMA30 Slope", "Weekly High (26w)", "Weekly High (52w)",
        "Weekly Volume", "Weekly RSI (14)", "Weekly Low")},
    **{name: _MONTHLY_NOTE for name in ("Monthly SMA10", "Monthly RSI (14)")},
    **{name: _BENCH_NOTE for name in (
        "RS vs Benchmark (26w)", "RS Line", "Benchmark Weekly SMA30", "Benchmark Monthly SMA10")},
    **{name: _RANK_NOTE for name in ("Performance Rank (13 Weeks)", "Performance Rank (26 Weeks)")},
}


def _parse_option(filter_name: str, option: str) -> tuple:
    """Parse ``option`` for the canonical ``filter_name`` into a spec tuple, or raise ValueError."""
    forms, examples = _ALL_DEFS[filter_name]
    text = _norm(option)
    for form in forms:
        spec = form.parse(text)
        if spec is not None:
            return spec
    raise ValueError(
        f"unknown option {option!r} for filter {filter_name!r}. "
        f"Example options: {', '.join(examples)}. "
        f"Patterns (N = number with optional $ and K/M/B, P% = percentage): "
        f"{'; '.join(f.template for f in forms)}")


FILTERS: dict[str, dict[str, tuple]] = {
    name: {label: _parse_option(name, label) for label in examples}
    for name, (_forms, examples) in _DEFS.items()
}


MTF_FILTERS: dict[str, dict[str, tuple]] = {
    name: {label: _parse_option(name, label) for label in examples}
    for name, (_forms, examples) in _MTF_DEFS.items()
}


def available_filters(include_mtf: bool = False) -> dict[str, list[str]]:
    """Filter name -> its example option labels (what a dropdown would offer).

    The default is the daily vocabulary only (``FILTERS``); ``include_mtf=True`` appends the
    multi-timeframe filters (``MTF_FILTERS``), which need the columns of ``setup_engine.mtf``.
    """
    out = {name: list(options) for name, options in FILTERS.items()}
    if include_mtf:
        out.update({name: list(options) for name, options in MTF_FILTERS.items()})
    return out


# ── Evaluating specs ──────────────────────────────────────────────────────────

def _col(ind: pd.DataFrame, name: str) -> np.ndarray:
    return ind[name].to_numpy(dtype="float64")


def _spec_columns(spec: tuple) -> list[str]:
    kind = spec[0]
    if kind in ("gt", "lt", "ge", "le", "between"):
        return [spec[1]]
    if kind in ("gt_col", "lt_col", "ge_ratio", "le_ratio"):
        return [spec[1], spec[2]]
    return [spec[1], spec[2], spec[3], spec[4]]          # cross_above / cross_below


# every column the multi-timeframe filters read (what ``indicators()`` does not produce)
MTF_FILTER_COLUMNS: frozenset = frozenset(
    c for options in MTF_FILTERS.values() for spec in options.values() for c in _spec_columns(spec))


def _evaluate(spec: tuple, ind: pd.DataFrame) -> np.ndarray:
    """One spec -> boolean array. Each kind is its own comparison: NaN compares False in all."""
    kind = spec[0]
    with np.errstate(invalid="ignore"):
        if kind == "gt":
            return _col(ind, spec[1]) > spec[2]
        if kind == "lt":
            return _col(ind, spec[1]) < spec[2]
        if kind == "ge":
            return _col(ind, spec[1]) >= spec[2] - EPS
        if kind == "le":
            return _col(ind, spec[1]) <= spec[2] + EPS
        if kind == "between":
            v = _col(ind, spec[1])
            return (v >= spec[2] - EPS) & (v <= spec[3] + EPS)
        if kind == "gt_col":
            return _col(ind, spec[1]) > _col(ind, spec[2])
        if kind == "lt_col":
            return _col(ind, spec[1]) < _col(ind, spec[2])
        if kind == "ge_ratio":
            return _col(ind, spec[1]) >= spec[3] * _col(ind, spec[2]) - EPS
        if kind == "le_ratio":
            return _col(ind, spec[1]) <= spec[3] * _col(ind, spec[2]) + EPS
        if kind == "cross_above":
            return (_col(ind, spec[1]) > _col(ind, spec[2])) & (_col(ind, spec[3]) <= _col(ind, spec[4]))
        if kind == "cross_below":
            return (_col(ind, spec[1]) < _col(ind, spec[2])) & (_col(ind, spec[3]) >= _col(ind, spec[4]))
    raise ValueError(f"unknown spec kind {kind!r}")


def _fmt(x: float) -> str:
    return f"{x:.10g}"


def _formula(spec: tuple) -> str:
    kind = spec[0]
    if kind == "gt":
        return f"{spec[1]} > {_fmt(spec[2])}"
    if kind == "lt":
        return f"{spec[1]} < {_fmt(spec[2])}"
    if kind == "ge":
        return f"{spec[1]} >= {_fmt(spec[2])}"
    if kind == "le":
        return f"{spec[1]} <= {_fmt(spec[2])}"
    if kind == "between":
        return f"{_fmt(spec[2])} <= {spec[1]} <= {_fmt(spec[3])}"
    if kind == "gt_col":
        return f"{spec[1]} > {spec[2]}"
    if kind == "lt_col":
        return f"{spec[1]} < {spec[2]}"
    if kind == "ge_ratio":
        return f"{spec[1]} >= {_fmt(spec[3])} * {spec[2]}"
    if kind == "le_ratio":
        return f"{spec[1]} <= {_fmt(spec[3])} * {spec[2]}"
    if kind == "cross_above":
        return f"{spec[1]} > {spec[2]} and {spec[3]} <= {spec[4]}"
    return f"{spec[1]} < {spec[2]} and {spec[3]} >= {spec[4]}"      # cross_below


# ── The screen ────────────────────────────────────────────────────────────────

class Screen:
    """A set of named filters, ANDed. ``Screen({"Price": "Over $5", "Relative Volume": "Over 2"})``.

    Filter names and option labels match case-insensitively, whitespace collapsed. Options are
    not limited to the dropdown examples: "Over $7.50" or "0-4% below" also parse.
    """

    def __init__(self, rules: dict[str, str]):
        self.rules: dict[str, str] = dict(rules)
        self._canon: list[str] = []
        self._specs: list[tuple] = []
        seen: set[str] = set()
        for name, option in self.rules.items():
            key = _norm(name)
            if key not in _FILTER_KEYS:
                raise ValueError(f"unknown filter {name!r}. Valid filters: {', '.join(_ALL_DEFS)}")
            if key in seen:
                raise ValueError(f"filter {name!r} given twice")
            seen.add(key)
            canon = _FILTER_KEYS[key]
            self._canon.append(canon)
            self._specs.append(_parse_option(canon, option))

    @classmethod
    def from_text(cls, text: str) -> "Screen":
        """``"Price: Over $5; Relative Volume: Over 2"`` (rules split on newlines and ``;``)."""
        rules: dict[str, str] = {}
        seen: set[str] = set()
        for part in re.split(r"[;\n]", text):
            part = part.strip()
            if not part:
                continue
            name, sep, option = part.partition(":")
            if not sep:
                raise ValueError(f"rule {part!r} has no ':' (expected 'Filter: Option')")
            name, option = name.strip(), option.strip()
            if _norm(name) in seen:
                raise ValueError(f"filter {name!r} given twice")
            seen.add(_norm(name))
            rules[name] = option
        return cls(rules)

    def __repr__(self) -> str:
        return f"Screen({self.rules!r})"

    def fields(self) -> list[str]:
        """The indicator columns the rules read, unique, in order of first use."""
        out: list[str] = []
        for spec in self._specs:
            for c in _spec_columns(spec):
                if c not in out:
                    out.append(c)
        return out

    def _require_mtf_columns(self, ind: pd.DataFrame) -> None:
        """A multi-timeframe rule on a table without its columns is a usage error, not a KeyError.

        Only MTF columns are checked here, so a missing daily column still fails exactly as before.
        """
        missing = [c for c in self.fields() if c in MTF_FILTER_COLUMNS and c not in ind.columns]
        if missing:
            raise ValueError(
                f"columns {missing} are not in the table: weekly/monthly/RS filters need "
                f"setup_engine.mtf.indicators_mtf or mtf_features; Performance Rank needs "
                f"mtf.add_xs_ranks")

    def valid(self, ind: pd.DataFrame) -> np.ndarray:
        """True where every column used by every rule is finite."""
        self._require_mtf_columns(ind)
        ok = np.ones(len(ind), dtype=bool)
        for c in self.fields():
            ok &= np.isfinite(_col(ind, c))
        return ok

    def mask(self, ind: pd.DataFrame) -> np.ndarray:
        """Boolean array over the rows of ``ind``: all rules hold and no input is NaN/inf."""
        self._require_mtf_columns(ind)
        out = np.ones(len(ind), dtype=bool)
        for spec in self._specs:
            out &= _evaluate(spec, ind)
        return out & self.valid(ind)

    def describe(self) -> str:
        """One line per rule: ``Filter: Option  [formula on columns]``."""
        if not self._specs:
            return "(no rules)"
        lines = []
        for (name, option), canon, spec in zip(self.rules.items(), self._canon, self._specs):
            line = f"{name}: {option}  [{_formula(spec)}]"
            if canon == "Monthly Low":
                line += f"  {_MONTHLY_LOW_NOTE}"
            elif canon in _MTF_NOTES:
                line += f"  {_MTF_NOTES[canon]}"
            lines.append(line)
        return "\n".join(lines)
