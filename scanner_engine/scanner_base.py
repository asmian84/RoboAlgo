"""
scanner_engine/scanner_base.py
────────────────────────────────────────────────────────────────────────────────
Abstract base class for all scanners.

All scanners:
  - Accept a universe of tickers
  - Apply filter criteria
  - Return normalized candidates with metrics
  - Are stateless and independently runnable
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScanCandidate:
    """Normalized output for every scanner hit."""
    symbol:         str
    price:          float
    change_percent: float
    volume:         int
    avg_volume:     int
    volume_ratio:   float          # volume / avg_volume
    gap_percent:    float = 0.0    # premarket gap
    market_cap:     float = 0.0
    scanner_name:   str  = ""
    metrics:        dict = field(default_factory=dict)  # scanner-specific extras

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol":         self.symbol,
            "price":          self.price,
            "change_percent": self.change_percent,
            "volume":         self.volume,
            "avg_volume":     self.avg_volume,
            "volume_ratio":   round(self.volume_ratio, 2),
            "gap_percent":    self.gap_percent,
            "scanner_name":   self.scanner_name,
            **self.metrics,
        }


class BaseScanner(ABC):
    """Abstract scanner — subclass and implement `scan()`."""

    name: str = "base"

    def run(self, tickers: list[str]) -> list[dict[str, Any]]:
        """Public entry point — returns list of candidate dicts sorted by score."""
        candidates = self.scan(tickers)
        candidates.sort(key=self._sort_key, reverse=True)
        return [c.to_dict() for c in candidates]

    @abstractmethod
    def scan(self, tickers: list[str]) -> list[ScanCandidate]:
        """Filter tickers and return matching ScanCandidate objects."""
        ...

    def _sort_key(self, c: ScanCandidate) -> float:
        """Default sort: volume ratio × abs(change_percent)."""
        return c.volume_ratio * abs(c.change_percent)
