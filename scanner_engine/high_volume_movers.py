"""
scanner_engine/high_volume_movers.py
────────────────────────────────────────────────────────────────────────────────
Detects stocks trading at unusually high volume relative to their 20-day average.

Criteria:
  - volume_ratio >= min_vol_ratio  (default 2.5×)
  - abs(change_percent) >= min_move (default 2%)
  - volume >= min_volume (default 500,000)
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import yfinance as yf

from .scanner_base import BaseScanner, ScanCandidate

logger = logging.getLogger(__name__)


class HighVolumeMoverScanner(BaseScanner):
    """Scan for stocks with exceptional volume and meaningful price move."""

    name = "high_volume_movers"

    def __init__(
        self,
        min_vol_ratio: float = 2.5,
        min_move_pct:  float = 2.0,
        min_volume:    int   = 500_000,
        lookback_days: int   = 20,
        max_workers:   int   = 20,
    ):
        self.min_vol_ratio = min_vol_ratio
        self.min_move_pct  = min_move_pct
        self.min_volume    = min_volume
        self.lookback_days = lookback_days
        self.max_workers   = max_workers

    def _check_one(self, symbol: str) -> ScanCandidate | None:
        try:
            hist = yf.Ticker(symbol).history(
                period=f"{self.lookback_days + 5}d", interval="1d"
            )
            if hist is None or len(hist) < self.lookback_days:
                return None

            today_vol    = int(hist["Volume"].iloc[-1])
            avg_vol      = int(hist["Volume"].iloc[-(self.lookback_days + 1):-1].mean())
            vol_ratio    = today_vol / avg_vol if avg_vol > 0 else 0.0

            if vol_ratio < self.min_vol_ratio or today_vol < self.min_volume:
                return None

            close_today  = float(hist["Close"].iloc[-1])
            close_prev   = float(hist["Close"].iloc[-2])
            change_pct   = ((close_today - close_prev) / close_prev) * 100

            if abs(change_pct) < self.min_move_pct:
                return None

            # Relative strength: compare to 5-day average return
            returns_5d = hist["Close"].pct_change().tail(5).dropna()
            avg_return = float(returns_5d.mean() * 100) if len(returns_5d) > 0 else 0.0

            # ATR for volatility context
            high = hist["High"].tail(14).values
            low  = hist["Low"].tail(14).values
            atr  = float(np.mean(high - low)) if len(high) > 0 else 0.0

            return ScanCandidate(
                symbol         = symbol.upper(),
                price          = round(close_today, 4),
                change_percent = round(change_pct, 2),
                volume         = today_vol,
                avg_volume     = avg_vol,
                volume_ratio   = round(vol_ratio, 2),
                scanner_name   = self.name,
                metrics        = {
                    "atr":          round(atr, 4),
                    "avg_return_5d": round(avg_return, 2),
                    "direction":    "up" if change_pct > 0 else "down",
                    "vol_spike":    round(vol_ratio, 1),
                },
            )
        except Exception as e:
            logger.debug("HighVolume skip %s: %s", symbol, e)
            return None

    def scan(self, tickers: list[str]) -> list[ScanCandidate]:
        results: list[ScanCandidate] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(self._check_one, sym): sym for sym in tickers}
            for future in as_completed(futures):
                c = future.result()
                if c:
                    results.append(c)
        return results

    def _sort_key(self, c: ScanCandidate) -> float:
        return c.volume_ratio * abs(c.change_percent)


# ── Example ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    universe = ["AAPL", "NVDA", "TSLA", "AMD", "MSFT", "META", "AMZN", "SPY", "QQQ"]
    scanner  = HighVolumeMoverScanner(min_vol_ratio=1.5, min_move_pct=1.0)
    hits     = scanner.run(universe)
    print(f"High volume movers: {len(hits)}")
    for h in hits:
        print(f"  {h['symbol']:6} chg={h['change_percent']:+.2f}%  vol={h['volume']:,}  ratio={h['volume_ratio']}x")
