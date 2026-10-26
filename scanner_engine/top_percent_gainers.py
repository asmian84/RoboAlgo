"""
scanner_engine/top_percent_gainers.py
────────────────────────────────────────────────────────────────────────────────
Ranks tickers by intraday percent change and returns top N movers.

Adds momentum quality metrics:
  - consistency_score: % of last N bars that moved in same direction
  - momentum_acceleration: is the move accelerating?
  - rsi_14: overbought/oversold context
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import yfinance as yf

from .scanner_base import BaseScanner, ScanCandidate

logger = logging.getLogger(__name__)


class TopPercentGainerScanner(BaseScanner):
    """Scan for top intraday percent movers with momentum quality scoring."""

    name = "top_percent_gainers"

    def __init__(
        self,
        top_n:       int   = 20,
        min_price:   float = 1.0,
        min_volume:  int   = 100_000,
        min_change:  float = 1.0,     # minimum % move (abs)
        max_workers: int   = 25,
    ):
        self.top_n       = top_n
        self.min_price   = min_price
        self.min_volume  = min_volume
        self.min_change  = min_change
        self.max_workers = max_workers

    @staticmethod
    def _rsi(closes: np.ndarray, period: int = 14) -> float:
        if len(closes) < period + 1:
            return 50.0
        deltas = np.diff(closes)
        gains  = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_g  = np.mean(gains[:period])
        avg_l  = np.mean(losses[:period])
        for i in range(period, len(gains)):
            avg_g = (avg_g * (period - 1) + gains[i]) / period
            avg_l = (avg_l * (period - 1) + losses[i]) / period
        rs = avg_g / avg_l if avg_l > 0 else 100
        return float(100 - 100 / (1 + rs))

    def _check_one(self, symbol: str) -> ScanCandidate | None:
        try:
            hist = yf.Ticker(symbol).history(period="30d", interval="1d")
            if hist is None or len(hist) < 3:
                return None

            closes    = hist["Close"].values.astype(float)
            volumes   = hist["Volume"].values.astype(int)
            price     = float(closes[-1])
            prev      = float(closes[-2])

            if price < self.min_price or prev <= 0:
                return None

            change_pct = ((price - prev) / prev) * 100
            if abs(change_pct) < self.min_change:
                return None

            vol_today = int(volumes[-1])
            avg_vol   = int(np.mean(volumes[:-1])) if len(volumes) > 1 else vol_today
            vol_ratio = vol_today / avg_vol if avg_vol > 0 else 1.0

            if vol_today < self.min_volume:
                return None

            # Momentum quality
            rsi = self._rsi(closes)

            # Consistency: how many of last 5 bars moved same direction?
            recent_chg = np.diff(closes[-6:]) if len(closes) >= 6 else np.array([change_pct])
            direction  = 1 if change_pct > 0 else -1
            consistent = int(np.sum(np.sign(recent_chg) == direction))
            consistency = consistent / len(recent_chg)

            # Acceleration: is today's move bigger than yesterday's?
            if len(closes) >= 3:
                prev_chg = abs((closes[-2] - closes[-3]) / closes[-3] * 100)
                accel    = abs(change_pct) > prev_chg
            else:
                accel = False

            return ScanCandidate(
                symbol         = symbol.upper(),
                price          = round(price, 4),
                change_percent = round(change_pct, 2),
                volume         = vol_today,
                avg_volume     = avg_vol,
                volume_ratio   = round(vol_ratio, 2),
                scanner_name   = self.name,
                metrics        = {
                    "rsi":                 round(rsi, 1),
                    "consistency_score":   round(consistency, 2),
                    "momentum_accel":      accel,
                    "direction":           "up" if change_pct > 0 else "down",
                },
            )
        except Exception as e:
            logger.debug("TopGainer skip %s: %s", symbol, e)
            return None

    def scan(self, tickers: list[str]) -> list[ScanCandidate]:
        results: list[ScanCandidate] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(self._check_one, sym): sym for sym in tickers}
            for future in as_completed(futures):
                c = future.result()
                if c:
                    results.append(c)
        # Sort by absolute change_pct descending, keep top_n
        results.sort(key=lambda c: abs(c.change_percent), reverse=True)
        return results[: self.top_n]

    def _sort_key(self, c: ScanCandidate) -> float:
        return abs(c.change_percent) * (1 + c.metrics.get("consistency_score", 0))


# ── Example ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    universe = ["AAPL", "NVDA", "TSLA", "AMD", "MSFT", "META", "AMZN", "GOOGL",
                "NFLX", "SPY", "QQQ", "SOXL", "TQQQ", "UPRO"]
    scanner  = TopPercentGainerScanner(top_n=5, min_change=0.5)
    hits     = scanner.run(universe)
    print(f"Top {len(hits)} gainers:")
    for h in hits:
        print(f"  {h['symbol']:6} {h['change_percent']:+.2f}%  RSI={h['rsi']}  consistent={h['consistency_score']:.0%}")
