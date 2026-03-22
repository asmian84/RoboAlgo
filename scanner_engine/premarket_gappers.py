"""
scanner_engine/premarket_gappers.py
────────────────────────────────────────────────────────────────────────────────
Detects stocks with significant premarket gaps vs prior close.

Gap = (premarket_price - prior_close) / prior_close × 100

Criteria:
  - |gap| >= min_gap_pct  (default 3%)
  - volume >= min_volume  (default 50,000)
  - Price in [min_price, max_price]
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import yfinance as yf

from .scanner_base import BaseScanner, ScanCandidate

logger = logging.getLogger(__name__)


class PremarketGapperScanner(BaseScanner):
    """Scan for premarket gap-up and gap-down candidates."""

    name = "premarket_gappers"

    def __init__(
        self,
        min_gap_pct:  float = 3.0,
        min_volume:   int   = 50_000,
        min_price:    float = 1.0,
        max_price:    float = 500.0,
        max_workers:  int   = 20,
    ):
        self.min_gap_pct  = min_gap_pct
        self.min_volume   = min_volume
        self.min_price    = min_price
        self.max_price    = max_price
        self.max_workers  = max_workers

    def _check_one(self, symbol: str) -> ScanCandidate | None:
        try:
            ticker = yf.Ticker(symbol)
            # Grab 2-day 1m bar to find premarket price and prior close
            hist2d = ticker.history(period="2d", interval="1d", prepost=True)
            hist1m = ticker.history(period="1d", interval="1m", prepost=True)

            if hist2d.empty or len(hist2d) < 2:
                return None

            prior_close   = float(hist2d["Close"].iloc[-2])
            current_price = float(hist2d["Close"].iloc[-1])

            if not (self.min_price <= current_price <= self.max_price):
                return None

            # Premarket price = first bar that trades before 09:30
            premarket_price = current_price
            if not hist1m.empty:
                try:
                    import pytz
                    eastern = pytz.timezone("America/New_York")
                    pm_bars = hist1m[
                        hist1m.index.tz_convert(eastern).hour < 9
                    ]
                    if not pm_bars.empty:
                        premarket_price = float(pm_bars["Close"].iloc[-1])
                except Exception:
                    pass

            if prior_close <= 0:
                return None

            gap_pct = ((premarket_price - prior_close) / prior_close) * 100
            if abs(gap_pct) < self.min_gap_pct:
                return None

            volume    = int(hist2d["Volume"].iloc[-1])
            avg_vol   = int(hist2d["Volume"].mean())
            vol_ratio = volume / avg_vol if avg_vol > 0 else 1.0

            if volume < self.min_volume:
                return None

            return ScanCandidate(
                symbol         = symbol.upper(),
                price          = round(current_price, 4),
                change_percent = round(gap_pct, 2),
                volume         = volume,
                avg_volume     = avg_vol,
                volume_ratio   = round(vol_ratio, 2),
                gap_percent    = round(gap_pct, 2),
                scanner_name   = self.name,
                metrics        = {
                    "prior_close":    round(prior_close, 4),
                    "premarket_price": round(premarket_price, 4),
                    "gap_direction":  "up" if gap_pct > 0 else "down",
                },
            )
        except Exception as e:
            logger.debug("PremarketGapper skip %s: %s", symbol, e)
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
        return abs(c.gap_percent) * c.volume_ratio


# ── Example ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    universe = ["AAPL", "NVDA", "TSLA", "AMD", "MSFT", "META", "AMZN", "GOOGL"]
    scanner  = PremarketGapperScanner(min_gap_pct=1.0)
    hits     = scanner.run(universe)
    print(f"Premarket gappers found: {len(hits)}")
    for h in hits:
        print(f"  {h['symbol']:6} gap={h['gap_percent']:+.2f}%  price=${h['price']}  vol_ratio={h['volume_ratio']}x")
