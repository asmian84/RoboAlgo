"""
RoboAlgo — Price Level Clustering Engine  (Stage 5)

Aggregates significant price levels from multiple technical sources and
clusters nearby levels to surface high-probability support/resistance zones.

Sources and weights:
  MA200          4  – major trend anchor
  MA50           3  – intermediate trend
  MA20 (BB mid)  2  – short-term mean
  Fibonacci      2  – range retracement levels
  BB Bands       1  – volatility envelope extremes
  Pivot Points   1  – classic daily session pivots
  Gann 1/8 divs  1  – geometric equal-division levels
  Round Numbers  1  – psychological price magnets
  ATR Steps      1  – volatility-normalised zones

Clustering:
  Levels within 0.6% of each other are merged into one cluster.
  Cluster strength = sum of source weights (capped at 10).

Named zones (closest qualifying cluster per direction):
  buy_zone    – nearest strong support below current price
  accumulate  – secondary support (add on dip)
  stop        – deepest significant support (thesis invalidation)
  scale_in    – nearest resistance above (add on strength)
  target      – next resistance cluster (first take-profit)
  distribution– major overhead supply zone
"""

import math
import logging

import pandas as pd

from data_engine.loader import DataLoader

logger = logging.getLogger(__name__)


class PriceLevelEngine:
    """Clusters multi-source technical levels into high-probability price zones."""

    CLUSTER_PCT  = 0.006   # levels within 0.6% of each other are merged
    LOOKBACK     = 252     # 1-year daily lookback for swing high/low
    MAX_RETURNED = 24      # maximum clusters in the returned list

    # ── Public API ────────────────────────────────────────────────────────────

    def compute_levels(self, symbol: str) -> dict:
        """
        Compute clustered price levels for a symbol.

        Returns:
            {
              symbol, current_price, atr, atr_pct,
              levels: list[Cluster],   # sorted by strength desc
              zones:  dict[str, Cluster]  # named zones
            }

        Where Cluster = {price, strength, sources, distance_pct, type, label?}
        """
        loader = DataLoader()
        prices = loader.get_prices(symbol)
        if prices.empty:
            return {"error": f"No price data for {symbol}"}

        indicators = loader.get_indicators(symbol)

        # Keep last LOOKBACK daily bars
        df = prices.tail(self.LOOKBACK).copy()
        if not indicators.empty:
            df = df.join(indicators, how="left")

        current = float(df["close"].iloc[-1])
        prev    = df.iloc[-2] if len(df) >= 2 else df.iloc[-1]
        atr     = self._get_atr(df, current)

        # ── Collect raw levels: (price, label, weight) ────────────────────────
        raw: list[tuple[float, str, int]] = []

        # 1. Moving averages (highest weight — institutional anchors)
        self._add_if_valid(raw, df, "ma200",     "MA200",   4)
        self._add_if_valid(raw, df, "ma50",      "MA50",    3)
        self._add_if_valid(raw, df, "bb_middle", "MA20",    2)

        # 2. Bollinger Band extremes
        self._add_if_valid(raw, df, "bb_upper", "BB Upper", 1)
        self._add_if_valid(raw, df, "bb_lower", "BB Lower", 1)

        # 3. Fibonacci retracements of the full 52-week range
        hi52  = float(df["high"].max())
        lo52  = float(df["low"].min())
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

        # 4. Classic Pivot Points from the prior session
        ph    = float(prev["high"])
        pl    = float(prev["low"])
        pc    = float(prev["close"])
        pivot = (ph + pl + pc) / 3.0
        for price, label in [
            (pivot,             "Pivot P"),
            (2 * pivot - pl,    "Pivot R1"),
            (pivot + ph - pl,   "Pivot R2"),
            (2 * pivot - ph,    "Pivot S1"),
            (pivot - (ph - pl), "Pivot S2"),
        ]:
            raw.append((price, label, 1))

        # 5. Gann 1/8 divisions of the 52-week range
        if rng52 > 0:
            for i, label in enumerate(
                ["Gann 1/8", "Gann 1/4", "Gann 3/8", "Gann 1/2",
                 "Gann 5/8", "Gann 3/4", "Gann 7/8"],
                start=1,
            ):
                raw.append((lo52 + i / 8.0 * rng52, label, 1))

        # 6. Psychological round numbers in a ±3 ATR window
        step  = self._round_step(current)
        lo_w  = current - 3 * atr
        hi_w  = current + 3 * atr
        for mult in range(math.floor(lo_w / step), math.ceil(hi_w / step) + 1):
            rp = mult * step
            if lo_w <= rp <= hi_w and abs(rp - current) / current < 0.15:
                raw.append((rp, f"Round ${rp:,.2f}", 1))

        # 7. ATR-step zones from current price
        for mult, label in [
            (-2.0, "ATR -2×"), (-1.5, "ATR -1.5×"), (-1.0, "ATR -1×"),
            ( 1.0, "ATR +1×"), ( 1.5, "ATR +1.5×"), ( 2.0, "ATR +2×"),
            ( 3.0, "ATR +3×"),
        ]:
            raw.append((current + mult * atr, label, 1))

        # ── Cluster ───────────────────────────────────────────────────────────
        threshold = current * self.CLUSTER_PCT
        clusters  = self._cluster(sorted(raw, key=lambda x: x[0]), threshold)

        # Annotate each cluster
        for c in clusters:
            c["distance_pct"] = round((c["price"] - current) / current * 100, 2)
            c["type"]         = "resistance" if c["price"] > current else "support"

        # Sort by strength descending, then absolute distance ascending
        clusters.sort(key=lambda c: (-c["strength"], abs(c["distance_pct"])))

        # ── Build named zones ─────────────────────────────────────────────────
        below = sorted(
            [c for c in clusters if c["price"] < current],
            key=lambda c: -c["price"],   # closest first
        )
        above = sorted(
            [c for c in clusters if c["price"] > current],
            key=lambda c:  c["price"],   # closest first
        )

        zones: dict[str, dict] = {}
        for key, pool, idx, label in [
            ("buy_zone",     below,  0, "Buy Zone"),
            ("accumulate",   below,  1, "Accumulate"),
            ("stop",         below, -1, "Stop Zone"),
            ("scale_in",     above,  0, "Scale In"),
            ("target",       above,  1, "Target"),
            ("distribution", above, -1, "Distribution"),
        ]:
            if pool and abs(idx) < len(pool):
                zones[key] = {**pool[idx], "label": label}

        return {
            "symbol":        symbol,
            "current_price": round(current, 2),
            "atr":           round(atr,     2),
            "atr_pct":       round(atr / current * 100, 2),
            "levels":        clusters[: self.MAX_RETURNED],
            "zones":         zones,
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _get_atr(df: pd.DataFrame, current: float) -> float:
        """Return the 14-period ATR, with a 0.5% floor.

        Uses the pre-computed 'atr' column when available; otherwise falls
        back to the shared vectorised implementation in indicator_engine.technical.
        """
        if "atr" in df.columns:
            val = df["atr"].dropna()
            if not val.empty:
                return max(float(val.iloc[-1]), current * 0.005)
        # Compute from OHLC via the canonical shared implementation
        from indicator_engine.technical import atr_scalar
        return atr_scalar(
            df["high"].to_numpy(dtype=float),
            df["low"].to_numpy(dtype=float),
            df["close"].to_numpy(dtype=float),
            period=14,
            floor_pct=0.005,
        )

    @staticmethod
    def _add_if_valid(
        raw: list,
        df: pd.DataFrame,
        col: str,
        label: str,
        wt: int,
    ) -> None:
        """Append a (price, label, weight) tuple if the column has a valid value."""
        if col not in df.columns:
            return
        val = df[col].iloc[-1]
        if not pd.isna(val) and float(val) > 0:
            raw.append((float(val), label, wt))

    @staticmethod
    def _round_step(price: float) -> float:
        """
        Psychological level step size:
          $10 steps for $100–$999 stocks
          $1  steps for $10–$99  stocks
          etc.
        """
        magnitude = 10 ** math.floor(math.log10(max(price, 0.01)))
        return magnitude / 10

    @staticmethod
    def _cluster(
        sorted_levels: list[tuple[float, str, int]],
        threshold: float,
    ) -> list[dict]:
        """
        Merge nearby levels (within `threshold` price units) into clusters.

        Each cluster:
          price    – weighted-mean price of merged levels
          strength – sum of source weights (capped at 10)
          sources  – deduplicated list of contributing source labels
        """
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
                "sources":  list(dict.fromkeys(sources)),  # dedup, preserve order
            })
            i = j

        return clusters
