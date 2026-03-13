"""
RoboAlgo — Data Validation Engine
Detects and reports data quality issues in price history:
  - Missing trading day bars
  - Price outliers (> N × median absolute deviation from rolling median)
  - Timestamp/date gaps (weekdays with no data)
  - Zero-volume days
  - OHLC integrity violations (e.g. high < low, open outside high-low range)

Usage:
    validator = DataValidator()
    report = validator.validate(symbol="SOXL")
    full   = validator.validate_all()
"""

import logging
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import select

from database.connection import get_session
from database.models import Instrument, PriceData

logger = logging.getLogger(__name__)

# ── Thresholds ─────────────────────────────────────────────────────────────────
OUTLIER_MAD_MULT    = 6.0    # flag if |price - rolling_median| > N × MAD
OUTLIER_WINDOW      = 20     # rolling window for outlier detection
MAX_GAP_DAYS        = 5      # flag gaps longer than N calendar days (allows long weekends)
MIN_VOLUME          = 100    # flag bars with volume < N (suspiciously low)
OHLC_TOLERANCE      = 1e-4   # floating point tolerance for OHLC checks


class DataValidator:
    """
    Validates price data quality for one or all instruments.

    Returns structured reports with issue lists, severity levels,
    and an overall quality score (0–100).
    """

    def validate(self, symbol: str) -> dict:
        """Full data quality validation for a single symbol."""
        with get_session() as session:
            instr = session.execute(
                select(Instrument).where(Instrument.symbol == symbol.upper())
            ).scalar_one_or_none()
            if not instr:
                return {"symbol": symbol, "error": "Symbol not found"}

            rows = session.execute(
                select(PriceData)
                .where(PriceData.instrument_id == instr.id)
                .order_by(PriceData.date)
            ).scalars().all()

        if not rows:
            return {"symbol": symbol, "error": "No price data found"}

        df = pd.DataFrame([{
            "date":   r.date,
            "open":   r.open,
            "high":   r.high,
            "low":    r.low,
            "close":  r.close,
            "volume": r.volume,
        } for r in rows]).set_index("date")

        issues = []
        issues.extend(self._check_date_gaps(df, symbol))
        issues.extend(self._check_outliers(df, symbol))
        issues.extend(self._check_ohlc_integrity(df, symbol))
        issues.extend(self._check_zero_volume(df, symbol))
        issues.extend(self._check_duplicate_dates(df, symbol))

        quality_score = self._compute_quality_score(issues, len(df))

        return {
            "symbol":        symbol,
            "total_bars":    len(df),
            "date_range":    f"{df.index[0]} → {df.index[-1]}",
            "issue_count":   len(issues),
            "quality_score": round(quality_score, 1),
            "issues":        issues,
            "summary": {
                "critical": sum(1 for i in issues if i["severity"] == "CRITICAL"),
                "warning":  sum(1 for i in issues if i["severity"] == "WARNING"),
                "info":     sum(1 for i in issues if i["severity"] == "INFO"),
            },
        }

    def validate_all(self, min_quality: float = 0.0) -> list[dict]:
        """
        Run validation for all instruments.
        Returns list of reports sorted by quality score ascending (worst first).
        Optionally filter to only symbols with quality_score < min_quality.
        """
        with get_session() as session:
            instruments = session.execute(select(Instrument)).scalars().all()
            symbols = [i.symbol for i in instruments]

        results = []
        for sym in symbols:
            try:
                report = self.validate(sym)
                if "error" not in report:
                    results.append(report)
            except Exception as e:
                logger.warning(f"Validation failed for {sym}: {e}")

        results.sort(key=lambda r: r["quality_score"])

        if min_quality > 0:
            results = [r for r in results if r["quality_score"] < min_quality]

        return results

    def get_quality_summary(self) -> dict:
        """High-level quality summary across all instruments."""
        reports = self.validate_all()
        if not reports:
            return {"total": 0}

        scores = [r["quality_score"] for r in reports]
        critical_count = sum(r["summary"]["critical"] for r in reports)

        return {
            "total":           len(reports),
            "avg_quality":     round(np.mean(scores), 1),
            "min_quality":     round(min(scores), 1),
            "below_80":        sum(1 for s in scores if s < 80),
            "below_60":        sum(1 for s in scores if s < 60),
            "total_critical":  critical_count,
            "worst_symbols":   [r["symbol"] for r in reports[:5]],
        }

    # ── Check Methods ──────────────────────────────────────────────────────────

    def _check_date_gaps(self, df: pd.DataFrame, symbol: str) -> list[dict]:
        """Detect unexpectedly large gaps between consecutive trading dates."""
        issues = []
        dates = list(df.index)

        for i in range(1, len(dates)):
            d0 = dates[i - 1]
            d1 = dates[i]
            # Convert date objects if needed
            if isinstance(d0, date):
                gap = (d1 - d0).days
            else:
                gap = (pd.Timestamp(d1) - pd.Timestamp(d0)).days

            # Skip standard weekends (2–3 day gaps)
            if gap <= 3:
                continue
            # Gaps > MAX_GAP_DAYS are suspicious (not just long weekends)
            if gap > MAX_GAP_DAYS:
                issues.append({
                    "type":     "DATE_GAP",
                    "severity": "WARNING" if gap <= 10 else "CRITICAL",
                    "date":     str(d1),
                    "detail":   f"Gap of {gap} calendar days between {d0} and {d1}",
                })

        return issues

    def _check_outliers(self, df: pd.DataFrame, symbol: str) -> list[dict]:
        """Detect price outliers using rolling MAD (median absolute deviation)."""
        issues = []
        close = df["close"].copy()

        rolling_median = close.rolling(OUTLIER_WINDOW, center=True, min_periods=5).median()
        abs_dev = (close - rolling_median).abs()
        rolling_mad = abs_dev.rolling(OUTLIER_WINDOW, center=True, min_periods=5).median()

        # Normalize MAD (consistent with 1σ in Gaussian)
        normalized_mad = rolling_mad * 1.4826

        for date_idx in df.index:
            if pd.isna(rolling_median.get(date_idx)) or pd.isna(normalized_mad.get(date_idx)):
                continue
            dev  = abs(float(close.get(date_idx, 0)) - float(rolling_median.get(date_idx, 0)))
            mad  = float(normalized_mad.get(date_idx, 1)) or 1.0
            if mad > 0 and dev / mad > OUTLIER_MAD_MULT:
                px = float(close.get(date_idx, 0))
                med = float(rolling_median.get(date_idx, 0))
                issues.append({
                    "type":     "PRICE_OUTLIER",
                    "severity": "CRITICAL",
                    "date":     str(date_idx),
                    "detail":   (
                        f"Close {px:.4f} deviates {dev/mad:.1f}× MAD "
                        f"from rolling median {med:.4f}"
                    ),
                })

        return issues

    def _check_ohlc_integrity(self, df: pd.DataFrame, symbol: str) -> list[dict]:
        """Detect OHLC violations: high < low, open/close outside high-low range."""
        issues = []

        for date_idx, row in df.iterrows():
            o, h, l, c = (
                float(row.get("open", 0) or 0),
                float(row.get("high", 0) or 0),
                float(row.get("low", 0)  or 0),
                float(row.get("close", 0) or 0),
            )
            if h < l - OHLC_TOLERANCE:
                issues.append({
                    "type":     "OHLC_INTEGRITY",
                    "severity": "CRITICAL",
                    "date":     str(date_idx),
                    "detail":   f"High {h} < Low {l} — inverted OHLC",
                })
            if o > h + OHLC_TOLERANCE or o < l - OHLC_TOLERANCE:
                issues.append({
                    "type":     "OHLC_INTEGRITY",
                    "severity": "WARNING",
                    "date":     str(date_idx),
                    "detail":   f"Open {o} outside High-Low range [{l}, {h}]",
                })
            if c > h + OHLC_TOLERANCE or c < l - OHLC_TOLERANCE:
                issues.append({
                    "type":     "OHLC_INTEGRITY",
                    "severity": "WARNING",
                    "date":     str(date_idx),
                    "detail":   f"Close {c} outside High-Low range [{l}, {h}]",
                })
            if h <= 0 or l <= 0 or c <= 0:
                issues.append({
                    "type":     "OHLC_INTEGRITY",
                    "severity": "CRITICAL",
                    "date":     str(date_idx),
                    "detail":   f"Non-positive price values (H={h}, L={l}, C={c})",
                })

        return issues

    def _check_zero_volume(self, df: pd.DataFrame, symbol: str) -> list[dict]:
        """Flag bars with suspiciously low or zero volume."""
        issues = []
        if "volume" not in df.columns:
            return issues

        for date_idx, row in df.iterrows():
            vol = float(row.get("volume", 0) or 0)
            if vol < MIN_VOLUME:
                issues.append({
                    "type":     "LOW_VOLUME",
                    "severity": "INFO",
                    "date":     str(date_idx),
                    "detail":   f"Volume {int(vol):,} below minimum {MIN_VOLUME:,}",
                })

        return issues

    def _check_duplicate_dates(self, df: pd.DataFrame, symbol: str) -> list[dict]:
        """Flag duplicate index dates (data load error)."""
        issues = []
        seen = set()
        for d in df.index:
            key = str(d)
            if key in seen:
                issues.append({
                    "type":     "DUPLICATE_DATE",
                    "severity": "CRITICAL",
                    "date":     key,
                    "detail":   f"Duplicate date entry for {key}",
                })
            seen.add(key)
        return issues

    # ── Scoring ────────────────────────────────────────────────────────────────

    def _compute_quality_score(self, issues: list[dict], total_bars: int) -> float:
        """Compute 0–100 data quality score. More issues → lower score."""
        if total_bars == 0:
            return 0.0

        score = 100.0
        for issue in issues:
            severity = issue.get("severity", "INFO")
            if severity == "CRITICAL":
                score -= 10.0
            elif severity == "WARNING":
                score -= 3.0
            else:
                score -= 0.5

        return float(np.clip(score, 0, 100))
