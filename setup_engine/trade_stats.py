"""
Statistics for simulated trades: summaries, cluster-robust confidence, matched control pools, and the
survivorship arithmetic. Nothing here simulates a trade or reads a price; it only describes R-multiples
(profit divided by initial risk) that ``setup_engine.trade_sim`` has already produced.

WHY IT EXISTS. A backtest that reports a mean R and a plain confidence interval overstates certainty:
trades opened in the same month share one market, so they are one observation, not forty. Every
interval here resamples whole CLUSTERS (calendar months or half-years), with the mean taken as a ratio
of sums exactly as ``setup_engine.learn.excess_return`` does for stocks. A result is also judged
against a control pool, the same-day, same-volatility-bucket trades that did NOT satisfy the setup, so
a good year or a volatile stock is never credited to the rule.

NO LOOK-AHEAD. Pools are keyed by the SIGNAL date and by ATR% as known at the signal close
(``vol_bucket``), never by anything that happened after entry. A treatment trade is compared with
controls that could have been picked on that same day with the same information.

ORDER MODEL. Not modelled here. The fills, stops, gaps and costs behind each R are ``trade_sim``'s, and
this module takes its numbers as given. The one cost-related rule is in the callers: treatment and
control must both be NET of the same cost model, so the excess is net against net.

KNOWN BIASES (not fixable here, reported alongside every result). Survivorship: the price universe is
today's listings, so stocks that were delisted are absent and any average is flattered;
``delisted_share_by_year`` measures how large that hole is. Yahoo adjusts history retroactively for
splits, which distorts old price bands for reverse-split stocks. No dividend adjustment. Clustering by
month removes within-month dependence but not dependence across months (overlapping holds, regimes),
so intervals are still somewhat too narrow.
"""
from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd

# ── shared layout (setup_engine step 4 imports these) ──────────────────────────────────────────────

EPOCH = np.datetime64("2000-01-01", "D")
N_DAYS = int((np.datetime64("2027-01-01", "D") - EPOCH).astype(int))      # 9862
ATR_PCT_EDGES = np.array([0.02, 0.03, 0.04, 0.06, 0.08, 0.12])
N_BUCKETS = 7                                                              # len(ATR_PCT_EDGES) + 1
POOL_STATS = ("n", "sum_r", "sum_r2", "n_win", "sum_win", "n_loss", "sum_loss", "n_ge3")
_IDX = {name: i for i, name in enumerate(POOL_STATS)}

_CHUNK_ELEMS = 2_000_000          # bootstrap draws are generated in blocks of at most this many indices


# ── dates and buckets ──────────────────────────────────────────────────────────────────────────────

def _as_dt(dates) -> pd.DatetimeIndex:
    if np.ndim(dates) == 0:                     # a single string / Timestamp / date
        dates = [dates]
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    return idx.tz_localize(None) if idx.tz is not None else idx


def day_index(dates) -> np.ndarray:
    """Days since EPOCH (2000-01-01) as an int64 ARRAY (length 1 for a scalar). ValueError outside
    [0, N_DAYS) or for a missing date: an index that silently wrapped would pool unrelated days."""
    d = _as_dt(dates)
    if d.hasnans:
        raise ValueError("day_index: missing date")
    days = (d.values.astype("datetime64[D]") - EPOCH).astype(np.int64)
    if len(days) and (days.min() < 0 or days.max() >= N_DAYS):
        raise ValueError(f"day_index: date outside 2000-01-01 .. 2026-12-31 "
                         f"(index range {int(days.min())}..{int(days.max())}, valid 0..{N_DAYS - 1})")
    return days


def vol_bucket(atr_pct) -> np.ndarray:
    """ATR% -> bucket 0..6 by ATR_PCT_EDGES. Missing/non-finite -> -1 (such trades join no pool).
    np.digitize(NaN) returns 6, the TOP bucket, so the explicit mapping is required."""
    a = np.asarray(atr_pct, dtype=float)
    b = np.digitize(a, ATR_PCT_EDGES).astype(np.int64)
    b[~np.isfinite(a)] = -1
    return b


def month_key(dates) -> np.ndarray:
    d = _as_dt(dates)
    return np.asarray(d.year, dtype=np.int64) * 100 + np.asarray(d.month, dtype=np.int64)


def half_year_key(dates) -> np.ndarray:
    d = _as_dt(dates)
    return np.asarray(d.year, dtype=np.int64) * 10 + np.where(np.asarray(d.month) <= 6, 1, 2)


# ── control pools ──────────────────────────────────────────────────────────────────────────────────

class PoolAccumulator:
    """Sufficient statistics of R by (signal day, volatility bucket), one array per pool key.

    ``arr[key]`` has shape (N_DAYS, N_BUCKETS, len(POOL_STATS)); the last axis follows POOL_STATS.
    win: r > 0, loss: r < 0, ge3: r >= 3. Only sums and counts are kept, so pools from separate
    chunks of the universe merge by addition.
    """

    def __init__(self) -> None:
        self.arr: dict[str, np.ndarray] = {}

    def add(self, key: str, day_idx, bucket, r) -> None:
        """Add trades. Rows with bucket < 0 or a non-finite r are dropped (unmatchable / unusable)."""
        day = np.asarray(day_idx, dtype=np.int64).ravel()
        b = np.asarray(bucket, dtype=np.int64).ravel()
        x = np.asarray(r, dtype=float).ravel()
        if not (len(day) == len(b) == len(x)):
            raise ValueError("add: day_idx, bucket and r must have the same length")
        if key not in self.arr:
            self.arr[key] = np.zeros((N_DAYS, N_BUCKETS, len(POOL_STATS)))
        keep = (b >= 0) & np.isfinite(x)
        day, b, x = day[keep], b[keep], x[keep]
        if len(day) == 0:
            return
        if day.min() < 0 or day.max() >= N_DAYS:
            raise ValueError("add: day_idx outside [0, N_DAYS)")
        if b.max() >= N_BUCKETS:
            raise ValueError("add: bucket outside [0, N_BUCKETS)")
        cell = day * N_BUCKETS + b
        win, loss = x > 0, x < 0
        weights = {
            "n": np.ones_like(x), "sum_r": x, "sum_r2": x * x,
            "n_win": win.astype(float), "sum_win": np.where(win, x, 0.0),
            "n_loss": loss.astype(float), "sum_loss": np.where(loss, x, 0.0),
            "n_ge3": (x >= 3.0).astype(float),
        }
        a = self.arr[key]
        for name, w in weights.items():
            a[:, :, _IDX[name]] += np.bincount(cell, weights=w, minlength=N_DAYS * N_BUCKETS).reshape(N_DAYS, N_BUCKETS)

    def merge(self, other: "PoolAccumulator") -> "PoolAccumulator":
        for key, a in other.arr.items():
            if key in self.arr:
                self.arr[key] += a
            else:
                self.arr[key] = a.copy()
        return self


def _cells(pool, day_idx, bucket) -> np.ndarray:
    """(m, len(POOL_STATS)) pooled statistics for each treatment trade: the trade's day, and its
    bucket (or all buckets when bucket is None). bucket may be a scalar or one value per trade; a
    bucket of -1 matches nothing (it must not wrap round to the top bucket)."""
    pool = np.asarray(pool)
    if pool.shape != (N_DAYS, N_BUCKETS, len(POOL_STATS)):
        raise ValueError(f"pool must have shape {(N_DAYS, N_BUCKETS, len(POOL_STATS))}, got {pool.shape}")
    d = np.asarray(day_idx, dtype=np.int64).ravel()
    if len(d) and (d.min() < 0 or d.max() >= N_DAYS):
        raise ValueError("day_idx outside [0, N_DAYS)")
    if bucket is None:
        return pool.sum(axis=1)[d]
    b = np.broadcast_to(np.asarray(bucket, dtype=np.int64), d.shape)
    ok = (b >= 0) & (b < N_BUCKETS)
    return np.where(ok[:, None], pool[d, np.where(ok, b, 0)], 0.0)


def pool_means(pool, day_idx, bucket=None):
    """(mean R, n) of the pool for each trade's day (and bucket, if given). Mean is NaN where n == 0."""
    cells = _cells(pool, day_idx, bucket)
    n = cells[:, _IDX["n"]]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, cells[:, _IDX["sum_r"]] / n, np.nan)
    return mean, n.astype(np.int64)


def excess_vs_pool(r, day_idx, pool, bucket=None) -> np.ndarray:
    """Each trade's R minus the pool's mean R on its day (and bucket). NaN where the pool is empty."""
    mean, _ = pool_means(pool, day_idx, bucket)
    return np.asarray(r, dtype=float) - mean


def matched_control_summary(pool, day_idx, bucket=None) -> dict:
    """What the control pool did, weighted so every treatment trade counts once.

    Each treatment trade with a non-empty pool (n_i > 0) gets weight 1, spread over its n_i controls,
    so a trade matched to 500 controls does not outweigh one matched to 5.
      expectancy    = mean(sum_r / n)
      win_rate      = mean(n_win / n)
      avg_win       = sum(sum_win / n) / sum(n_win / n)
      avg_loss      = sum(sum_loss / n) / sum(n_loss / n)         (negative)
      profit_factor = sum(sum_win / n) / -sum(sum_loss / n)
      share_ge_3r   = mean(n_ge3 / n)
    """
    cells = _cells(pool, day_idx, bucket)
    n_all = cells[:, _IDX["n"]]
    m = n_all > 0
    out = {"n_input": int(len(n_all)), "n_matched": int(m.sum()), "expectancy": None, "win_rate": None,
           "avg_win": None, "avg_loss": None, "profit_factor": None, "share_ge_3r": None,
           "median": None,        # a median cannot be recovered from sums and counts; keep it None
           "median_pool_size": None, "min_pool_size": None}
    if not m.any():
        return out
    c = cells[m]
    n = c[:, _IDX["n"]]

    def w(name):
        return c[:, _IDX[name]] / n

    wins, losses = w("sum_win").sum(), w("sum_loss").sum()
    n_win, n_loss = w("n_win").sum(), w("n_loss").sum()
    out.update(expectancy=float(w("sum_r").mean()), win_rate=float(w("n_win").mean()),
               avg_win=float(wins / n_win) if n_win > 0 else None,
               avg_loss=float(losses / n_loss) if n_loss > 0 else None,
               profit_factor=float(wins / -losses) if losses < 0 else None,
               share_ge_3r=float(w("n_ge3").mean()),
               median_pool_size=float(np.median(n)), min_pool_size=int(n.min()))
    return out


# ── summaries ──────────────────────────────────────────────────────────────────────────────────────

def summarize(r) -> dict:
    """Headline numbers for a set of R-multiples. Non-finite values are dropped."""
    x = np.asarray(r, dtype=float).ravel()
    x = x[np.isfinite(x)]
    out = {"n": int(len(x)), "win_rate": None, "avg_win": None, "avg_loss": None, "expectancy": None,
           "median": None, "profit_factor": None, "share_ge_3r": None, "winsorized_mean": None}
    if len(x) == 0:
        return out
    win, loss = x[x > 0], x[x < 0]
    lo, hi = np.quantile(x, [0.01, 0.99])
    out.update(win_rate=float((x > 0).mean()),
               avg_win=float(win.mean()) if len(win) else None,
               avg_loss=float(loss.mean()) if len(loss) else None,
               expectancy=float(x.mean()), median=float(np.median(x)),
               profit_factor=float(win.sum() / -loss.sum()) if len(loss) else None,
               share_ge_3r=float((x >= 3.0).mean()),
               winsorized_mean=float(np.clip(x, lo, hi).mean()))
    return out


# ── cluster bootstrap ──────────────────────────────────────────────────────────────────────────────

def _factorize(labels):
    """(codes, K): integer code per label; label missing -> -1."""
    codes, uniques = pd.factorize(pd.Series(labels), use_na_sentinel=True)
    return np.asarray(codes, dtype=np.int64), len(uniques)


def _clean(x, clusters):
    x = np.asarray(x, dtype=float).ravel()
    lab = np.asarray(clusters).ravel()
    if len(x) != len(lab):
        raise ValueError("x and clusters must have the same length")
    keep = np.isfinite(x)
    return x[keep], lab[keep]


def _draws(K: int, n_boot: int, rng):
    """Resampling index blocks of shape (<=n_boot, K), at most _CHUNK_ELEMS indices each."""
    step = max(1, _CHUNK_ELEMS // max(K, 1))
    done = 0
    while done < n_boot:
        m = min(step, n_boot - done)
        yield rng.integers(0, K, size=(m, K))
        done += m


def cluster_bootstrap(x, clusters, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Mean of x with a percentile interval from resampling whole clusters with replacement.

    The mean is a RATIO OF SUMS (total over total count), as in ``learn.excess_return``: a cluster
    holding 99 trades weighs 99, not 1. Non-finite x (and rows with no cluster label) are dropped.
    Fewer than two clusters cannot be resampled, so lo = hi = p_le_0 = None.
    """
    x, lab = _clean(x, clusters)
    codes, K = _factorize(lab)
    ok = codes >= 0
    x, codes = x[ok], codes[ok]
    out = {"mean": None, "lo": None, "hi": None, "p_le_0": None, "n": int(len(x)), "n_clusters": int(K)}
    if len(x) == 0:
        out["n_clusters"] = 0
        return out
    s = np.bincount(codes, weights=x, minlength=K)
    c = np.bincount(codes, minlength=K).astype(float)
    out["mean"] = float(s.sum() / c.sum())
    if K < 2:
        return out
    rng = np.random.default_rng(seed)
    boots = np.concatenate([s[i].sum(axis=1) / c[i].sum(axis=1) for i in _draws(K, n_boot, rng)])
    out.update(lo=float(np.quantile(boots, alpha / 2)), hi=float(np.quantile(boots, 1 - alpha / 2)),
               p_le_0=float((boots <= 0.0).mean()))
    return out


def cluster_bootstrap_diff(xa, ca, xb, cb, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict:
    """mean(xa) - mean(xb) with an interval from resampling the UNION of cluster labels jointly.

    Joint resampling keeps the two arms paired by period: if a bad month is drawn, it is drawn for
    both. Each arm's mean is a ratio of sums. A draw can contain no trade from one arm (its mean is
    then undefined); those draws are skipped with nanquantile rather than poisoning the interval.
    Returns the cluster_bootstrap keys plus n_a, n_b ("n" = n_a + n_b).
    """
    xa, la = _clean(xa, ca)
    xb, lb = _clean(xb, cb)
    na, nb = len(xa), len(xb)
    codes, K = _factorize(np.concatenate([la, lb]))
    out = {"mean": None, "lo": None, "hi": None, "p_le_0": None, "n": na + nb, "n_a": na, "n_b": nb,
           "n_clusters": int(K)}
    if na == 0 or nb == 0:
        return out
    ka, kb = codes[:na], codes[na:]
    if (ka < 0).any() or (kb < 0).any():
        raise ValueError("cluster labels must not be missing")
    sa = np.bincount(ka, weights=xa, minlength=K)
    cna = np.bincount(ka, minlength=K).astype(float)
    sb = np.bincount(kb, weights=xb, minlength=K)
    cnb = np.bincount(kb, minlength=K).astype(float)
    out["mean"] = float(xa.sum() / na - xb.sum() / nb)
    if K < 2:
        return out
    rng = np.random.default_rng(seed)
    parts = []
    with np.errstate(invalid="ignore", divide="ignore"):
        for i in _draws(K, n_boot, rng):
            parts.append(sa[i].sum(axis=1) / cna[i].sum(axis=1) - sb[i].sum(axis=1) / cnb[i].sum(axis=1))
    d = np.concatenate(parts)
    fin = np.isfinite(d)
    if not fin.any():
        return out
    out.update(lo=float(np.nanquantile(d, alpha / 2)), hi=float(np.nanquantile(d, 1 - alpha / 2)),
               p_le_0=float((d[fin] <= 0.0).mean()))
    return out


# ── sizing arithmetic ──────────────────────────────────────────────────────────────────────────────

def breakeven_f(E, L):
    """Share f of trades that hit a total loss L (R, negative) at which a book whose other trades earn
    E per trade breaks even: (1 - f) E + f L = 0  =>  f = E / (E - L). None unless E > 0 and L < 0."""
    if E is None or L is None:
        return None
    return float(E / (E - L)) if (E > 0 and L < 0) else None


def adjusted_expectancy(E, f, L):
    """Expectancy per trade when a share f of trades end at L instead of earning E: (1 - f) E + f L."""
    if E is None or f is None or L is None:
        return None
    return float((1.0 - f) * E + f * L)


# ── survivorship: how big is the hole in a today's-listings universe? ─────────────────────────────

NON_COMMON_RE = r"Acquisition|Merger|Warrant|\bUnits?\b|\bRights?\b|When Issued|Preferred|Pfd"   # case-insensitive


def delisted_share_by_year(delisted: pd.DataFrame, spans: pd.DataFrame, years, bottom_counts: Optional[dict] = None) -> pd.DataFrame:
    """Share of listed common stocks that are missing from a today's-listings price universe, by year.

    delisted: listing-status rows with columns name, assetType, ipoDate, delistingDate. Only
    assetType == "Stock" is kept and names matching NON_COMMON_RE are dropped: 2,376 of the 7,539
    "Stock" rows are SPAC shells, warrants and units; they never reach a BOTTOM state, and SPAC
    liquidations swamp 2021-2023. Rows with no ipoDate or delistingDate cannot be placed in time and
    are not counted.
    spans: the survivors' price history, columns symbol, first_date, last_date.

    Columns: survivors_alive (first_date <= Dec 31 y and last_date >= Jan 1 y), delisted_alive (same test
    on ipoDate / delistingDate), f_stock = d / (d + s). If bottom_counts (year -> number of survivor
    stocks in the BOTTOM state) is given: delisted_near (listed by Dec 31 y, delisted between Jan 1 y
    and Dec 31 y+2) and f_bottom_scenario = near / (near + bottom_counts[y]). Ratios with a zero
    denominator are NaN. list_incomplete is True for y < 2013 (the delisted list is thin before then).
    """
    for col in ("name", "assetType", "ipoDate", "delistingDate"):
        if col not in delisted.columns:
            raise ValueError(f"delisted is missing column {col!r}")
    for col in ("first_date", "last_date"):
        if col not in spans.columns:
            raise ValueError(f"spans is missing column {col!r}")
    dl = delisted[delisted["assetType"] == "Stock"]
    dl = dl[~dl["name"].astype(str).str.contains(NON_COMMON_RE, case=False, regex=True, na=False)]
    ipo = pd.to_datetime(dl["ipoDate"], errors="coerce")
    out_ = pd.to_datetime(dl["delistingDate"], errors="coerce")
    first = pd.to_datetime(spans["first_date"], errors="coerce")
    last = pd.to_datetime(spans["last_date"], errors="coerce")
    rows = []
    for y in years:
        y = int(y)
        jan1, dec31 = pd.Timestamp(y, 1, 1), pd.Timestamp(y, 12, 31)
        s = int(((first <= dec31) & (last >= jan1)).sum())
        d = int(((ipo <= dec31) & (out_ >= jan1)).sum())
        row = {"year": y, "survivors_alive": s, "delisted_alive": d,
               "f_stock": d / (d + s) if d + s > 0 else np.nan}
        if bottom_counts is not None:
            near = int(((ipo <= dec31) & (out_ >= jan1) & (out_ <= pd.Timestamp(y + 2, 12, 31))).sum())
            b = bottom_counts.get(y)
            row["delisted_near"] = near
            row["f_bottom_scenario"] = near / (near + b) if (b is not None and near + b > 0) else np.nan
        row["list_incomplete"] = y < 2013
        rows.append(row)
    return pd.DataFrame(rows).set_index("year")
