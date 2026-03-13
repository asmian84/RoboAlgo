"""Liquidity Heatmap Engine.

Identifies liquidity clusters above and below price using:
- previous highs/lows
- range boundaries
- swing highs/lows
- compression zones

Scoring (0-100):
- touch_count: 40%
- volume_cluster: 40%
- distance_from_price: 20%
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from sqlalchemy import text

from database.connection import get_engine

CLUSTER_TOLERANCE = 0.003  # 0.3%
APPROACH_PCT = 0.005       # 0.5%
MAX_TOUCHES = 8
VOL_RATIO_CAP = 3.0
PROXIMITY_CAP = 0.05


def _swing_points(high: np.ndarray, low: np.ndarray, window: int = 3) -> list[tuple[int, float, str]]:
    swings: list[tuple[int, float, str]] = []
    n = len(high)
    for i in range(window, n - window):
        if high[i] >= np.max(high[i - window : i + window + 1]):
            swings.append((i, float(high[i]), "H"))
        if low[i] <= np.min(low[i - window : i + window + 1]):
            swings.append((i, float(low[i]), "L"))
    swings.sort(key=lambda x: x[0])
    return swings


def _cluster_levels(levels: list[float], tolerance_pct: float) -> list[float]:
    if not levels:
        return []
    values = sorted(float(x) for x in levels)
    clusters: list[list[float]] = [[values[0]]]
    for lvl in values[1:]:
        center = float(np.mean(clusters[-1]))
        if abs(lvl - center) / max(center, 1e-9) <= tolerance_pct:
            clusters[-1].append(lvl)
        else:
            clusters.append([lvl])
    return [float(np.mean(c)) for c in clusters]


class LiquidityHeatmapEngine:
    def build_heatmap(self, symbol: str, as_of_date: date | None = None) -> dict:
        universe = self._load_universe(max_symbols=1, lookback_bars=260, symbol=symbol, as_of_date=as_of_date)
        if symbol.upper() not in universe:
            return {
                "symbol": symbol.upper(),
                "liquidity_levels": [],
                "nearest_above": None,
                "nearest_below": None,
            }
        return self._analyze_symbol(symbol.upper(), universe[symbol.upper()])

    def scan_universe(self, max_symbols: int = 5000, lookback_bars: int = 260) -> list[dict]:
        data = self._load_universe(max_symbols=max_symbols, lookback_bars=lookback_bars)
        return [self._analyze_symbol(symbol, df) for symbol, df in data.items()]

    def _load_universe(
        self,
        max_symbols: int,
        lookback_bars: int,
        symbol: str | None = None,
        as_of_date: date | None = None,
    ) -> dict[str, pd.DataFrame]:
        symbol_clause = "AND i.symbol = :symbol" if symbol else ""
        date_clause = "AND p.date <= :as_of_date" if as_of_date else ""
        q = text(
            f"""
            WITH selected AS (
              SELECT i.id, i.symbol
              FROM instruments i
              WHERE 1=1 {symbol_clause}
              ORDER BY i.symbol
              LIMIT :max_symbols
            ),
            ranked AS (
              SELECT
                p.instrument_id,
                p.date,
                p.high,
                p.low,
                p.close,
                p.volume,
                ROW_NUMBER() OVER (PARTITION BY p.instrument_id ORDER BY p.date DESC) AS rn
              FROM price_data p
              JOIN selected s ON s.id = p.instrument_id
              WHERE 1=1 {date_clause}
            )
            SELECT s.symbol, r.date, r.high, r.low, r.close, r.volume
            FROM ranked r
            JOIN selected s ON s.id = r.instrument_id
            WHERE r.rn <= :lookback_bars
            ORDER BY s.symbol, r.date
            """
        )
        params = {"max_symbols": max_symbols, "lookback_bars": lookback_bars}
        if symbol:
            params["symbol"] = symbol.upper()
        if as_of_date:
            params["as_of_date"] = as_of_date

        df = pd.read_sql_query(q, get_engine(), params=params)
        if df.empty:
            return {}

        out: dict[str, pd.DataFrame] = {}
        for sym, g in df.groupby("symbol", sort=False):
            g = g.dropna(subset=["high", "low", "close"]).reset_index(drop=True)
            if len(g) >= 30:
                out[sym] = g
        return out

    def _analyze_symbol(self, symbol: str, df: pd.DataFrame) -> dict:
        high = df["high"].to_numpy(dtype=float)
        low = df["low"].to_numpy(dtype=float)
        close = df["close"].to_numpy(dtype=float)
        volume = df["volume"].fillna(0.0).to_numpy(dtype=float)

        cur = float(close[-1])
        avg_vol = float(np.nanmean(volume)) if len(volume) else 1.0

        levels: list[float] = []

        # previous highs/lows
        levels.extend([float(np.max(high[-10:])), float(np.min(low[-10:]))])

        # range boundaries
        levels.extend([float(np.max(high[-40:])), float(np.min(low[-40:]))])

        # swing highs/lows
        swings = _swing_points(high, low, window=3)
        levels.extend([p for _, p, t in swings if t == "H"][-6:])
        levels.extend([p for _, p, t in swings if t == "L"][-6:])

        # compression zones: bars with small true-range percentile
        tr = np.maximum(high - low, 1e-9)
        tr_pct = np.argsort(np.argsort(tr)) / max(len(tr) - 1, 1)
        comp_idx = np.where(tr_pct < 0.2)[0]
        if len(comp_idx) > 0:
            levels.append(float(np.max(high[comp_idx])))
            levels.append(float(np.min(low[comp_idx])))

        clustered = _cluster_levels(levels, CLUSTER_TOLERANCE)

        scored = []
        for lvl in clustered:
            tol_abs = lvl * APPROACH_PCT
            near = (high >= lvl - tol_abs) & (low <= lvl + tol_abs)
            touches = int(np.sum(near))
            touch_score = min(touches / MAX_TOUCHES, 1.0) * 100.0

            near_vol = float(np.nanmean(volume[near])) if np.any(near) else 0.0
            vol_ratio = near_vol / max(avg_vol, 1.0)
            vol_score = min(vol_ratio / VOL_RATIO_CAP, 1.0) * 100.0

            dist_pct = abs(cur - lvl) / max(cur, 1e-9)
            dist_score = max(0.0, 100.0 - (dist_pct / PROXIMITY_CAP) * 100.0)

            liq_score = round(0.4 * touch_score + 0.4 * vol_score + 0.2 * dist_score, 1)
            scored.append({"price_level": round(float(lvl), 4), "liquidity_score": liq_score, "distance_pct": dist_pct})

        scored.sort(key=lambda x: x["liquidity_score"], reverse=True)
        levels_out = [{"price_level": x["price_level"], "liquidity_score": x["liquidity_score"]} for x in scored[:12]]

        above = [x for x in scored if x["price_level"] > cur]
        below = [x for x in scored if x["price_level"] < cur]
        nearest_above = min(above, key=lambda x: x["distance_pct"], default=None)
        nearest_below = min(below, key=lambda x: x["distance_pct"], default=None)

        return {
            "symbol": symbol,
            "liquidity_levels": levels_out,
            "nearest_above": nearest_above["price_level"] if nearest_above else None,
            "nearest_below": nearest_below["price_level"] if nearest_below else None,
        }
