"""
RoboAlgo — Harmonic Pattern Engine
Detects XABCD harmonic structures using Fibonacci ratios.

Patterns:
  Gartley:    XAB=61.8%, ABC=38.2–88.6%, BCD=127.2–161.8%, XAD=78.6%
  Bat:        XAB=38.2–50%, ABC=38.2–88.6%, BCD=161.8–261.8%, XAD=88.6%
  Butterfly:  XAB=78.6%, ABC=38.2–88.6%, BCD=161.8–261.8%, XAD=127.2–161.8%
  Crab:       XAB=38.2–61.8%, ABC=38.2–88.6%, BCD=224–361.8%, XAD=161.8%

PRZ (Potential Reversal Zone): D-point confluence zone where reversal is expected.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from sqlalchemy import select, desc
from sqlalchemy.dialects.postgresql import insert as pg_insert
from tqdm import tqdm

from database.connection import get_session
from database.models import Instrument, PriceData, HarmonicPattern

logger = logging.getLogger(__name__)

# ── Fibonacci Ratio Definitions ────────────────────────────────────────────────
# Each pattern: {ratio_name: (min, max)} tolerance ±5%
PATTERN_RATIOS = {
    "Gartley": {
        "XAB": (0.588, 0.648),   # 61.8% ± 5%
        "ABC": (0.362, 0.886),
        "BCD": (1.227, 1.618),
        "XAD": (0.736, 0.836),   # 78.6% ± 5%
    },
    "Bat": {
        "XAB": (0.362, 0.500),
        "ABC": (0.362, 0.886),
        "BCD": (1.618, 2.618),
        "XAD": (0.836, 0.936),   # 88.6% ± 5%
    },
    "Butterfly": {
        "XAB": (0.736, 0.836),   # 78.6% ± 5%
        "ABC": (0.362, 0.886),
        "BCD": (1.618, 2.618),
        "XAD": (1.227, 1.618),
    },
    "Crab": {
        "XAB": (0.362, 0.618),
        "ABC": (0.362, 0.886),
        "BCD": (2.24, 3.618),
        "XAD": (1.568, 1.668),   # 161.8% ± 5%
    },
}

PIVOT_ORDER    = 5    # swing detection window
PRZ_MARGIN     = 0.03 # 3% PRZ width around D-point
MIN_CONFIDENCE = 0.45 # minimum to store pattern


class HarmonicEngine:
    """
    Detects XABCD harmonic patterns from swing points.

    Usage:
        engine = HarmonicEngine()
        engine.compute_and_store()
        patterns = engine.get_latest("SOXL")
    """

    # ── Public API ─────────────────────────────────────────────────────────────

    def compute_and_store(self, symbol: Optional[str] = None) -> int:
        """Detect and store harmonic patterns for one or all instruments."""
        with get_session() as session:
            if symbol:
                instruments = session.execute(
                    select(Instrument).where(Instrument.symbol == symbol.upper())
                ).scalars().all()
            else:
                instruments = session.execute(select(Instrument)).scalars().all()

            total = 0
            for inst in tqdm(instruments, desc="Harmonic patterns"):
                try:
                    rows = self._process_instrument(session, inst)
                    if rows:
                        self._upsert(session, rows)
                        total += len(rows)
                except Exception as e:
                    logger.warning(f"Harmonics failed for {inst.symbol}: {e}")
            logger.info(f"Harmonic engine: stored {total} patterns.")
            return total

    def get_latest(self, symbol: str, limit: int = 5) -> list[dict]:
        """Return the most recent harmonic patterns for a symbol."""
        with get_session() as session:
            instr = session.execute(
                select(Instrument).where(Instrument.symbol == symbol.upper())
            ).scalar_one_or_none()
            if not instr:
                return []
            rows = session.execute(
                select(HarmonicPattern)
                .where(HarmonicPattern.instrument_id == instr.id)
                .order_by(desc(HarmonicPattern.date))
                .limit(limit)
            ).scalars().all()
            return [self._row_to_dict(r) for r in rows]

    def get_pattern_score(self, symbol: str) -> float:
        """Return 0–100 harmonic pattern score for confluence."""
        patterns = self.get_latest(symbol, limit=3)
        if not patterns:
            return 50.0
        best = max(patterns, key=lambda p: p.get("pattern_score", 0))
        return float(best.get("pattern_score", 50.0))

    # ── Internal Processing ────────────────────────────────────────────────────

    def _process_instrument(self, session, instrument) -> list[dict]:
        """Detect harmonic patterns for one instrument."""
        price_rows = session.execute(
            select(PriceData)
            .where(PriceData.instrument_id == instrument.id)
            .order_by(PriceData.date)
        ).scalars().all()

        if len(price_rows) < 50:
            return []

        prices = pd.DataFrame([{
            "date": r.date, "high": r.high, "low": r.low, "close": r.close,
        } for r in price_rows]).set_index("date")

        return self._detect_patterns(instrument.id, prices)

    def _detect_patterns(self, instrument_id: int, df: pd.DataFrame) -> list[dict]:
        """Detect XABCD harmonic patterns using swing point pivots."""
        high  = df["high"].values
        low   = df["low"].values
        close = df["close"].values
        dates = df.index.tolist()

        # Find swing highs and lows
        swing_hi_idx = argrelextrema(high, np.greater, order=PIVOT_ORDER)[0]
        swing_lo_idx = argrelextrema(low,  np.less,   order=PIVOT_ORDER)[0]

        # Combine all swings sorted by index
        swings = []
        for idx in swing_hi_idx:
            swings.append({"idx": int(idx), "price": float(high[idx]), "type": "high"})
        for idx in swing_lo_idx:
            swings.append({"idx": int(idx), "price": float(low[idx]), "type": "low"})
        swings.sort(key=lambda s: s["idx"])

        if len(swings) < 5:
            return []

        records = []

        # Scan all 5-swing combinations for XABCD patterns
        for i in range(len(swings) - 4):
            X, A, B, C, D = swings[i], swings[i+1], swings[i+2], swings[i+3], swings[i+4]

            # Ensure alternating highs and lows
            if not self._alternating([X, A, B, C, D]):
                continue

            # Compute leg lengths
            xa = abs(A["price"] - X["price"])
            ab = abs(B["price"] - A["price"])
            bc = abs(C["price"] - B["price"])
            cd = abs(D["price"] - C["price"])
            xd = abs(D["price"] - X["price"])

            if xa == 0 or ab == 0 or bc == 0:
                continue

            # Fibonacci ratios
            xab = ab / xa
            abc = bc / ab
            bcd = cd / bc if bc > 0 else 0
            xad = xd / xa

            # Determine direction (bullish = X high, A low, etc.)
            direction = "bullish" if X["type"] == "high" else "bearish"

            # Test against each pattern definition
            for pattern_name, ratios in PATTERN_RATIOS.items():
                confidence = self._compute_confidence(
                    xab, abc, bcd, xad, ratios
                )
                if confidence < MIN_CONFIDENCE:
                    continue

                # PRZ around D-point
                d_price  = D["price"]
                prz_high = d_price * (1 + PRZ_MARGIN)
                prz_low  = d_price * (1 - PRZ_MARGIN)

                records.append({
                    "instrument_id": instrument_id,
                    "date":         dates[D["idx"]],
                    "pattern_type": pattern_name,
                    "direction":    direction,
                    "x_price":      round(X["price"], 4),
                    "a_price":      round(A["price"], 4),
                    "b_price":      round(B["price"], 4),
                    "c_price":      round(C["price"], 4),
                    "d_price":      round(D["price"], 4),
                    "xab_ratio":    round(xab, 4),
                    "abc_ratio":    round(abc, 4),
                    "bcd_ratio":    round(bcd, 4),
                    "xad_ratio":    round(xad, 4),
                    "prz_high":     round(prz_high, 4),
                    "prz_low":      round(prz_low, 4),
                    "confidence":   round(confidence, 4),
                    "pattern_score": round(confidence * 100, 2),
                })

        return records

    def _alternating(self, swings: list[dict]) -> bool:
        """Check that swing types alternate: high/low/high/low/high or reverse."""
        types = [s["type"] for s in swings]
        for i in range(len(types) - 1):
            if types[i] == types[i + 1]:
                return False
        return True

    def _compute_confidence(
        self,
        xab: float, abc: float, bcd: float, xad: float,
        ratios: dict
    ) -> float:
        """
        Compute confidence score based on how closely ratios match pattern definition.
        Each ratio contributes equally. Returns 0–1.
        """
        scores = []
        for ratio_name, (r_min, r_max) in ratios.items():
            val = {"XAB": xab, "ABC": abc, "BCD": bcd, "XAD": xad}.get(ratio_name, 0)
            if r_min <= val <= r_max:
                # Score by distance from midpoint
                mid   = (r_min + r_max) / 2
                width = (r_max - r_min) / 2
                dist  = abs(val - mid) / max(width, 1e-8)
                scores.append(1.0 - dist * 0.5)  # 0.5–1.0 range
            else:
                # How far outside the valid range?
                overshoot = min(abs(val - r_min), abs(val - r_max))
                tolerance = (r_max - r_min) * 0.15  # 15% tolerance
                if overshoot <= tolerance:
                    scores.append(0.3)  # partial credit
                else:
                    scores.append(0.0)

        return float(np.mean(scores)) if scores else 0.0

    def _upsert(self, session, records: list[dict]):
        if not records:
            return
        for record in records:
            stmt = pg_insert(HarmonicPattern).values([record])
            stmt = stmt.on_conflict_do_nothing()
            try:
                session.execute(stmt)
            except Exception:
                pass
        session.commit()

    def _row_to_dict(self, row: HarmonicPattern) -> dict:
        return {
            "date":          str(row.date),
            "pattern_type":  row.pattern_type,
            "direction":     row.direction,
            "x_price":       row.x_price,
            "a_price":       row.a_price,
            "b_price":       row.b_price,
            "c_price":       row.c_price,
            "d_price":       row.d_price,
            "xab_ratio":     row.xab_ratio,
            "abc_ratio":     row.abc_ratio,
            "bcd_ratio":     row.bcd_ratio,
            "xad_ratio":     row.xad_ratio,
            "prz_high":      row.prz_high,
            "prz_low":       row.prz_low,
            "confidence":    row.confidence,
            "pattern_score": row.pattern_score,
        }
