"""
RoboAlgo — Expected Move Engine
Estimates the potential trade magnitude before signal generation.

Multiplier table by instrument category:
  Index leveraged ETFs          → 2.5–3.0×
  Single-stock leveraged ETFs   → 3.0–4.0×
  High-beta sectors (biotech/energy) → 4.0–5.0×

Trade filter:
  expected_move ≥ 10% of entry price → required to pass signal gate
  expected_move < 10% → signal rejected (insufficient edge)
"""

import logging
from typing import Optional

import numpy as np

from database.connection import get_session
from database.models import Instrument, Indicator, PriceData
from sqlalchemy import select, desc

logger = logging.getLogger(__name__)

# ── Multiplier Table by Instrument Category ────────────────────────────────────
# Maps instrument_type or symbol pattern → (min_mult, max_mult)
MULTIPLIER_TABLE = {
    # Index leveraged ETFs (3x)
    "index_leveraged_3x":  (2.5, 3.0),
    # Index leveraged ETFs (2x)
    "index_leveraged_2x":  (2.0, 2.5),
    # Single-stock leveraged ETFs (2x on individual stocks)
    "single_stock_2x":     (3.0, 4.0),
    # Biotech / healthcare sector ETFs
    "biotech":             (4.0, 5.0),
    # Energy / commodity sector ETFs
    "energy":              (4.0, 5.0),
    # Sector leveraged ETFs (general)
    "sector_leveraged_3x": (2.5, 3.0),
    # Crypto ETFs and miners
    "crypto":              (3.5, 5.0),
    # Underlying stocks (non-leveraged)
    "stock":               (1.0, 1.5),
    # Default
    "default":             (2.0, 2.5),
}

# Symbols that override default category lookup
SYMBOL_CATEGORY_OVERRIDES = {
    # Biotech
    "LABU": "biotech", "LABD": "biotech",
    # Energy
    "GUSH": "energy", "DRIP": "energy",
    "BOIL": "energy", "KOLD": "energy",
    "UCO":  "energy", "SCO":  "energy",
    # Crypto
    "BITU": "crypto", "BITI": "crypto",
    "MSTU": "crypto", "MSTZ": "crypto",
    "MRAL": "crypto", "RIOT": "crypto", "MARA": "crypto",
    # 3x index
    "TQQQ": "index_leveraged_3x", "SQQQ": "index_leveraged_3x",
    "UPRO": "index_leveraged_3x", "SPXU": "index_leveraged_3x",
    "TNA":  "index_leveraged_3x", "TZA":  "index_leveraged_3x",
    "UDOW": "index_leveraged_3x", "SDOW": "index_leveraged_3x",
    "SOXL": "index_leveraged_3x", "SOXS": "index_leveraged_3x",
    "FAS":  "index_leveraged_3x", "FAZ":  "index_leveraged_3x",
    "TECL": "index_leveraged_3x", "TECS": "index_leveraged_3x",
    # 2x single stock
    "NVDL": "single_stock_2x", "NVDS": "single_stock_2x",
    "TSLL": "single_stock_2x", "TSLZ": "single_stock_2x", "TSLQ": "single_stock_2x",
    "AAPU": "single_stock_2x", "AAPD": "single_stock_2x",
    "AMZU": "single_stock_2x", "AMZD": "single_stock_2x",
    "METU": "single_stock_2x", "METD": "single_stock_2x",
    "MSFU": "single_stock_2x", "MSFD": "single_stock_2x",
    "PLTU": "single_stock_2x", "PLTZ": "single_stock_2x",
}

# Minimum expected move to pass the trade filter (10%)
MIN_EXPECTED_MOVE_PCT = 0.10


class ExpectedMoveEngine:
    """
    Calculates expected move magnitude for a trade before it's taken.

    Usage:
        engine = ExpectedMoveEngine()
        result = engine.calculate(symbol="SOXL", atr=2.40, entry_price=52.30)
        result = engine.calculate_from_db(symbol="SOXL")   # uses latest DB ATR
    """

    def calculate(
        self,
        symbol: str,
        atr: float,
        entry_price: float,
        instrument_type: Optional[str] = None,
        compression_duration: int = 0,
    ) -> dict:
        """
        Calculate expected move for a trade.

        Args:
            symbol:              Ticker symbol
            atr:                 Average True Range (in price units)
            entry_price:         Current/entry price
            instrument_type:     Optional instrument_type from DB
            compression_duration: Compression duration (increases multiplier)

        Returns dict with expected_move_$, expected_move_%, passes_filter, multiplier, etc.
        """
        category   = self._get_category(symbol, instrument_type)
        mult_range = MULTIPLIER_TABLE.get(category, MULTIPLIER_TABLE["default"])

        # Duration bonus: longer compression → higher multiplier (up to +0.5×)
        duration_bonus = min(compression_duration / 20, 1.0) * 0.5

        # Use midpoint of range + duration bonus
        multiplier = (mult_range[0] + mult_range[1]) / 2 + duration_bonus
        multiplier = min(multiplier, mult_range[1] + 0.5)  # cap at range_max + 0.5

        # Expected move in dollars and percent
        expected_move_dollars = atr * multiplier
        expected_move_pct     = expected_move_dollars / max(entry_price, 1e-8)

        # Trade targets (optimistic/base/conservative)
        target_optimistic    = entry_price + atr * mult_range[1]
        target_base          = entry_price + atr * multiplier
        target_conservative  = entry_price + atr * mult_range[0]

        passes_filter = expected_move_pct >= MIN_EXPECTED_MOVE_PCT

        return {
            "symbol":               symbol.upper(),
            "category":             category,
            "atr":                  round(atr, 4),
            "entry_price":          round(entry_price, 4),
            "multiplier":           round(multiplier, 3),
            "multiplier_range":     [round(mult_range[0], 2), round(mult_range[1], 2)],
            "expected_move_dollars": round(expected_move_dollars, 4),
            "expected_move_pct":    round(expected_move_pct, 4),
            "expected_move_pct_display": f"{expected_move_pct:.1%}",
            "target_conservative":  round(target_conservative, 4),
            "target_base":          round(target_base, 4),
            "target_optimistic":    round(target_optimistic, 4),
            "passes_filter":        passes_filter,
            "filter_threshold_pct": MIN_EXPECTED_MOVE_PCT,
            "rejection_reason":     None if passes_filter else (
                f"Expected move {expected_move_pct:.1%} < minimum {MIN_EXPECTED_MOVE_PCT:.0%} threshold"
            ),
        }

    def calculate_from_db(self, symbol: str, compression_duration: int = 0) -> Optional[dict]:
        """
        Convenience: pull latest ATR and price from DB, then calculate.
        """
        with get_session() as session:
            instr = session.execute(
                select(Instrument).where(Instrument.symbol == symbol.upper())
            ).scalar_one_or_none()
            if not instr:
                return None

            ind = session.execute(
                select(Indicator)
                .where(Indicator.instrument_id == instr.id)
                .order_by(desc(Indicator.date))
                .limit(1)
            ).scalar_one_or_none()

            price = session.execute(
                select(PriceData)
                .where(PriceData.instrument_id == instr.id)
                .order_by(desc(PriceData.date))
                .limit(1)
            ).scalar_one_or_none()

            if not ind or not price:
                return None

            return self.calculate(
                symbol            = symbol,
                atr               = ind.atr or 0.0,
                entry_price       = price.close or 0.0,
                instrument_type   = instr.instrument_type,
                compression_duration = compression_duration,
            )

    def get_multiplier(self, symbol: str, instrument_type: Optional[str] = None) -> tuple[float, float]:
        """Return (min_mult, max_mult) for a symbol."""
        category = self._get_category(symbol, instrument_type)
        return MULTIPLIER_TABLE.get(category, MULTIPLIER_TABLE["default"])

    # ── Internal ───────────────────────────────────────────────────────────────

    def _get_category(self, symbol: str, instrument_type: Optional[str] = None) -> str:
        """Determine the multiplier category for a symbol."""
        sym = symbol.upper()

        # Explicit override
        if sym in SYMBOL_CATEGORY_OVERRIDES:
            return SYMBOL_CATEGORY_OVERRIDES[sym]

        # Use instrument_type from DB
        if instrument_type:
            t = instrument_type.lower()
            if "biotech" in t:
                return "biotech"
            if "energy" in t or "commodity" in t:
                return "energy"
            if "single" in t or "individual" in t:
                return "single_stock_2x"
            if "leveraged" in t and "sector" in t:
                return "sector_leveraged_3x"
            if "leveraged" in t:
                return "index_leveraged_3x"
            if "crypto" in t:
                return "crypto"
            if "stock" in t:
                return "stock"

        # Heuristics from symbol name
        if sym.endswith("L") or sym.endswith("U"):
            return "sector_leveraged_3x"
        if sym.endswith("S") or sym.endswith("D") or sym.endswith("Z"):
            return "sector_leveraged_3x"  # bear leveraged

        return "default"
