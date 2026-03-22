"""
signal_engine/level_engine.py
────────────────────────────────────────────────────────────────────────────────
Multi-horizon price level mapping: surface "gems" (high-confluence zones)
and signal when price is within pounce range.

The engine is self-contained — it fetches price history via yfinance so it
works for any ticker without requiring the symbol to be in the local DB.

Horizons:
  scalp  — sub-hour to 2-hour trades (tight, intraday levels)
  day    — intraday to end-of-day holds
  trend  — multi-day to multi-week swing positions

A "gem" is a level that:
  • has strength ≥ 5 (multiple independent sources clustered there), OR
  • spans 2+ time horizons (e.g., MA50 + Fib 50% = day AND trend)

A "pounce" level is a gem where price is currently within 0.30 ATR.
"""
from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Horizon map: source label fragment → horizons ─────────────────────────────
# Matched via `any(frag in source_label for frag in frags)`
_HORIZON_MAP: list[tuple[tuple[str, ...], list[str]]] = [
    # Scalp: tight intraday levels
    (("Round $",),                      ["scalp"]),
    (("BB Upper", "BB Lower"),          ["scalp"]),
    (("ATR -1×", "ATR +1×"),            ["scalp"]),
    (("Pivot P", "Pivot S1", "Pivot R1"),["scalp"]),
    # Day trade
    (("MA20",),                         ["day"]),
    (("ATR -1.5×", "ATR +1.5×",
      "ATR -2×",   "ATR +2×"),          ["day"]),
    (("Pivot S2", "Pivot R2"),          ["day"]),
    (("Call Wall", "Put Wall",
      "Max Pain"),                      ["day"]),
    # Day + Trend (multi-horizon → gem candidates)
    (("MA50",),                         ["day", "trend"]),
    (("Fib",),                          ["day", "trend"]),
    (("Gann",),                         ["day", "trend"]),
    (("ATR +3×",),                      ["day", "trend"]),
    (("Week High", "Week Low"),         ["trend"]),
    # Pure trend
    (("MA200",),                        ["trend"]),
    # GEX Flip spans all horizons — it's a magnetism level dealers defend
    (("GEX Flip",),                     ["scalp", "day", "trend"]),
]


def _tag_horizons(sources: list[str]) -> list[str]:
    """Return deduplicated sorted list of horizons for a cluster's source labels."""
    seen: set[str] = set()
    for source in sources:
        for frags, horizons in _HORIZON_MAP:
            if any(frag in source for frag in frags):
                seen.update(horizons)
    return sorted(seen) or ["day"]   # fallback: classify unknown as day


# ── Clustering helper (identical logic to PriceLevelEngine._cluster) ──────────

def _cluster_raw(
    sorted_levels: list[tuple[float, str, int]],
    threshold: float,
) -> list[dict]:
    """Merge sorted (price, label, weight) tuples within `threshold` price units."""
    if not sorted_levels:
        return []

    clusters: list[dict] = []
    i = 0
    while i < len(sorted_levels):
        prices   = [sorted_levels[i][0]]
        sources  = [sorted_levels[i][1]]
        weights  = [sorted_levels[i][2]]
        total_wt = sorted_levels[i][2]

        j = i + 1
        while j < len(sorted_levels) and sorted_levels[j][0] - prices[0] <= threshold:
            prices.append(sorted_levels[j][0])
            sources.append(sorted_levels[j][1])
            weights.append(sorted_levels[j][2])
            total_wt += sorted_levels[j][2]
            j += 1

        wp = sum(p * w for p, w in zip(prices, weights)) / total_wt
        clusters.append({
            "price":    round(wp, 2),
            "strength": min(total_wt, 10),
            "sources":  list(dict.fromkeys(sources)),
        })
        i = j

    return clusters


# ── ATR computation ───────────────────────────────────────────────────────────

def _compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    """14-period ATR from OHLC DataFrame with 0.5% floor."""
    c_prev = df["Close"].shift(1)
    tr     = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - c_prev).abs(),
        (df["Low"]  - c_prev).abs(),
    ], axis=1).max(axis=1)
    atr    = float(tr.rolling(period).mean().iloc[-1])
    price  = float(df["Close"].iloc[-1])
    return max(atr, price * 0.005)


# ── Trade idea generator ──────────────────────────────────────────────────────

def _make_trade_idea(level_price: float, level_type: str, atr: float) -> dict:
    """Generate entry/stop/target for a gem level.

    Entry uses 0.10 ATR offset (not a flat $0.10) so cheap stocks get
    proportional entries instead of nonsensical near-zero R:R.
    """
    entry_offset = round(atr * 0.10, 4)   # 10% of ATR — scales with price
    if level_type == "support":
        entry  = round(level_price + entry_offset, 2)
        stop   = round(level_price - atr * 0.5, 2)
        target = round(level_price + atr * 2.0, 2)
        risk   = max(entry - stop, 0.01)
        return {
            "direction": "long",
            "entry":     entry,
            "stop":      stop,
            "target":    target,
            "rr":        round((target - entry) / risk, 1),
        }
    else:
        entry  = round(level_price - entry_offset, 2)
        stop   = round(level_price + atr * 0.5, 2)
        target = round(level_price - atr * 2.0, 2)
        risk   = max(stop - entry, 0.01)
        return {
            "direction": "short",
            "entry":     entry,
            "stop":      stop,
            "target":    target,
            "rr":        round((entry - target) / risk, 1),
        }


# ── Round number step size ────────────────────────────────────────────────────

def _round_step(price: float) -> float:
    magnitude = 10 ** math.floor(math.log10(max(price, 0.01)))
    return magnitude / 10


# ── Main engine ───────────────────────────────────────────────────────────────

class LevelEngine:
    """
    Compute multi-horizon level map for any ticker.

    Uses yfinance for price history — works for any symbol without DB dependency.
    GEX levels are added opportunistically (options data not always available).
    """

    CLUSTER_PCT  = 0.006   # 0.6% proximity threshold (same as PriceLevelEngine)
    POUNCE_ATR   = 0.50    # within 0.50 ATR = pounce zone (wider net, still actionable)
    MAX_DISPLAY  = 30      # cap returned levels

    def compute(self, symbol: str) -> dict:
        """
        Main entry point. Returns full level map with gems and pounce signals.

        Returns:
            {
              symbol, current_price, atr, atr_pct, trend,
              levels: list[LevelDict],
              gems:   list[LevelDict],   # strength≥5 OR multi-horizon
              pounce: list[LevelDict],   # gems within 0.30 ATR
            }
        """
        import yfinance as yf

        sym = symbol.upper()

        # ── Fetch price history ────────────────────────────────────────────────
        try:
            ticker = yf.Ticker(sym)
            hist   = ticker.history(period="1y", interval="1d")
            if hist.empty or len(hist) < 20:
                return {"error": f"Insufficient price history for {sym}"}
        except Exception as e:
            logger.warning("yfinance history failed for %s: %s", sym, e)
            return {"error": f"Could not fetch price data for {sym}"}

        df      = hist.copy()
        current = float(df["Close"].iloc[-1])
        atr     = _compute_atr(df)
        threshold = current * self.CLUSTER_PCT

        # ── Build raw level list: (price, label, weight) ──────────────────────
        raw: list[tuple[float, str, int]] = []

        # 1. Moving averages
        close = df["Close"]
        if len(close) >= 200:
            raw.append((float(close.rolling(200).mean().iloc[-1]), "MA200", 4))
        if len(close) >= 50:
            raw.append((float(close.rolling(50).mean().iloc[-1]),  "MA50",  3))
        ma20_val = float(close.rolling(20).mean().iloc[-1])
        raw.append((ma20_val, "MA20", 2))

        # 2. Bollinger Bands
        std20    = float(close.rolling(20).std().iloc[-1])
        raw.append((ma20_val + 2 * std20, "BB Upper", 1))
        raw.append((ma20_val - 2 * std20, "BB Lower", 1))

        # 3. Fibonacci retracements (52-week range)
        hi52  = float(df["High"].max())
        lo52  = float(df["Low"].min())
        rng52 = hi52 - lo52
        if rng52 > 0:
            for ratio, label in [
                (0.236, "Fib 23.6%"),
                (0.382, "Fib 38.2%"),
                (0.500, "Fib 50%"),
                (0.618, "Fib 61.8%"),
                (0.786, "Fib 78.6%"),
            ]:
                raw.append((hi52 - ratio * rng52, label, 2))

        # 4. Classic Pivot Points (prior session)
        prev  = df.iloc[-2]
        ph, pl, pc = float(prev["High"]), float(prev["Low"]), float(prev["Close"])
        pivot = (ph + pl + pc) / 3.0
        for price, label in [
            (pivot,             "Pivot P"),
            (2 * pivot - pl,    "Pivot R1"),
            (pivot + ph - pl,   "Pivot R2"),
            (2 * pivot - ph,    "Pivot S1"),
            (pivot - (ph - pl), "Pivot S2"),
        ]:
            raw.append((price, label, 1))

        # 5. Weekly high/low (approximately prior 5 trading days)
        if len(df) >= 10:
            week_slice = df.iloc[-10:-5]
            raw.append((float(week_slice["High"].max()),  "Week High", 3))
            raw.append((float(week_slice["Low"].min()),   "Week Low",  3))

        # 6. Gann 1/8 divisions of 52-week range
        if rng52 > 0:
            for i, label in enumerate(
                ["Gann 1/8", "Gann 1/4", "Gann 3/8", "Gann 1/2",
                 "Gann 5/8", "Gann 3/4", "Gann 7/8"],
                start=1,
            ):
                raw.append((lo52 + i / 8.0 * rng52, label, 1))

        # 7. Psychological round numbers (±3 ATR window)
        step = _round_step(current)
        lo_w = current - 3 * atr
        hi_w = current + 3 * atr
        for mult in range(math.floor(lo_w / step), math.ceil(hi_w / step) + 1):
            rp = mult * step
            if lo_w <= rp <= hi_w and abs(rp - current) / current < 0.15:
                raw.append((rp, f"Round ${rp:,.2f}", 1))

        # 8. ATR-step zones from current price
        for mult, label in [
            (-2.0, "ATR -2×"), (-1.5, "ATR -1.5×"), (-1.0, "ATR -1×"),
            ( 1.0, "ATR +1×"), ( 1.5, "ATR +1.5×"), ( 2.0, "ATR +2×"),
            ( 3.0, "ATR +3×"),
        ]:
            raw.append((current + mult * atr, label, 1))

        # 9. GEX levels (optional — graceful failure)
        gex_levels = self._fetch_gex_levels(sym, current)
        raw.extend(gex_levels)

        # ── Cluster ────────────────────────────────────────────────────────────
        sorted_raw = sorted(raw, key=lambda x: x[0])
        clusters   = _cluster_raw(sorted_raw, threshold)

        # ── Annotate each cluster ──────────────────────────────────────────────
        ma50_val = next((p for p, l, _ in raw if l == "MA50"), None)
        trend = (
            "up"      if ma50_val and current > ma50_val * 1.005 else
            "down"    if ma50_val and current < ma50_val * 0.995 else
            "neutral"
        )

        levels: list[dict] = []
        for c in clusters:
            ltype        = "resistance" if c["price"] > current else "support"
            dist_pct     = round((c["price"] - current) / current * 100, 2)
            atr_distance = round(abs(c["price"] - current) / atr, 2)
            horizons     = _tag_horizons(c["sources"])
            is_gem       = c["strength"] >= 4 or len(set(horizons)) >= 2
            is_ready     = is_gem and atr_distance <= self.POUNCE_ATR

            level_dict: dict = {
                "price":        c["price"],
                "strength":     c["strength"],
                "sources":      c["sources"],
                "horizons":     horizons,
                "type":         ltype,
                "distance_pct": dist_pct,
                "atr_distance": atr_distance,
                "gem":          is_gem,
                "ready":        is_ready,
                "trade_idea":   _make_trade_idea(c["price"], ltype, atr) if is_gem else None,
            }
            levels.append(level_dict)

        # Sort: gems first → then by proximity
        levels.sort(key=lambda x: (not x["gem"], x["atr_distance"]))
        levels = levels[: self.MAX_DISPLAY]

        gems   = [l for l in levels if l["gem"]]
        pounce = [l for l in levels if l["ready"]]

        # Package last 60 daily bars for the sparkline hover chart.
        # We already have the full DataFrame here — no extra download needed.
        spark_bars: list[dict] = []
        for dt, row in df.tail(60).iterrows():
            try:
                spark_bars.append({
                    "date":   str(dt)[:10],
                    "open":   round(float(row["Open"]),  2),
                    "high":   round(float(row["High"]),  2),
                    "low":    round(float(row["Low"]),   2),
                    "close":  round(float(row["Close"]), 2),
                    "volume": int(row.get("Volume", 0) or 0),
                })
            except Exception:
                pass

        return {
            "symbol":        sym,
            "current_price": round(current, 2),
            "atr":           round(atr, 2),
            "atr_pct":       round(atr / current * 100, 2),
            "trend":         trend,
            "levels":        levels,
            "gems":          gems,
            "pounce":        pounce,
            "spark_bars":    spark_bars,   # free — already in memory
        }

    # ── GEX helper ────────────────────────────────────────────────────────────

    @staticmethod
    def _fetch_gex_levels(
        symbol: str,
        current: float,
    ) -> list[tuple[float, str, int]]:
        """
        Fetch GEX key levels from options chain.
        Returns empty list (silently) if options data is unavailable.
        """
        try:
            from data_engine.yahoo_options_fetcher import OptionsChainFetcher
            from options_engine.gamma_exposure import GammaExposureEngine

            chain = OptionsChainFetcher().fetch(symbol)
            if chain is None:
                return []

            gex = GammaExposureEngine().compute(chain)
            levels: list[tuple[float, str, int]] = []

            # Only include GEX levels that are within ±15% of current price
            # (wider strikes are irrelevant for near-term trading)
            for price, label, weight in [
                (gex.zero_gamma,  "GEX Flip",  3),
                (gex.call_wall,   "Call Wall", 3),
                (gex.put_wall,    "Put Wall",  3),
                (gex.max_pain,    "Max Pain",  2),
            ]:
                if price and price > 0 and abs(price - current) / current <= 0.15:
                    levels.append((float(price), label, weight))

            return levels
        except Exception as exc:
            logger.debug("GEX levels skipped for %s: %s", symbol, exc)
            return []
