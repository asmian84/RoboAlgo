"""Pattern orchestration service for backend-only detection and persistence."""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from typing import Any

import pandas as pd
from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from database.connection import get_session
from database.models import Instrument, PatternDetection, PatternSignal, PriceData
from pattern_engine.chart_patterns import detect_all as detect_all_chart_patterns
from pattern_engine.gann_levels import detect as detect_gann_levels
from pattern_engine.harmonic_patterns import detect as detect_harmonic_patterns
from pattern_engine.validation_engine import compute_confluence
from pattern_engine.wyckoff_structures import detect as detect_wyckoff_structures
from pattern_engine.candlestick_detector import detect as detect_candlestick_patterns
from pattern_engine.behavioral_detector import detect as detect_behavioral_patterns
from pattern_engine.rsi_divergence import detect as detect_rsi_divergence
from pattern_engine.macd_patterns import detect as detect_macd_patterns
from pattern_engine.volume_patterns import detect as detect_volume_patterns
from pattern_engine.measured_move import detect as detect_measured_move
from pattern_engine.strategy_patterns import detect as detect_strategy_patterns
from pattern_engine.market_analysis import detect as detect_market_analysis
from pattern_engine.astro_cycles import detect as detect_astro_cycles

logger = logging.getLogger("pattern_engine.service")

VALID_STATES = {"NOT_PRESENT", "FORMING", "READY", "BREAKOUT", "FAILED", "COMPLETED"}

# Ordered list of category detectors — each returns list[dict] or dict


def _detect_all_indicators(symbol: str, df) -> list:
    results = []
    results.extend(detect_rsi_divergence(symbol, df))
    results.extend(detect_macd_patterns(symbol, df))
    return results


CATEGORY_DETECTORS = (
    ("chart",           detect_all_chart_patterns),
    ("harmonic",        detect_harmonic_patterns),
    ("gann",            detect_gann_levels),
    ("wyckoff",         detect_wyckoff_structures),
    ("candlestick",     detect_candlestick_patterns),
    ("behavioral",      detect_behavioral_patterns),
    ("indicator",       _detect_all_indicators),
    ("volume",          detect_volume_patterns),
    ("measured_move",   detect_measured_move),
    ("strategy",        detect_strategy_patterns),
    ("market_analysis", detect_market_analysis),
    ("astro",           detect_astro_cycles),
)

# Minimum resolution_minutes for each category.  0 = daily is always allowed.
# resolution_minutes values: 1, 5, 15, 30, 60, 0(daily)
CATEGORY_MIN_RESOLUTION: dict[str, int] = {
    "chart":           0,    # all timeframes (chair skipped internally in chart_patterns)
    "harmonic":        0,    # all timeframes — Fibonacci ratios are fractal
    "gann":            0,    # all timeframes — geometric angles
    "wyckoff":         60,   # skip below 1h — needs structural development time
    "candlestick":     0,    # all timeframes
    "behavioral":      30,   # needs at least 30m bars for structure
    "indicator":       0,    # all timeframes
    "volume":          0,    # all timeframes
    "measured_move":   0,    # all timeframes
    "strategy":        0,    # all timeframes
    "market_analysis": 60,   # needs daily/1h bars for structure
    "astro":           0,    # always run — planetary positions don't depend on bar resolution
}


def _fetch_daily_via_yfinance(symbol: str, limit: int = 1500) -> pd.DataFrame:
    """Fetch daily OHLCV from yfinance for symbols not in the DB."""
    try:
        import yfinance as yf
        period = "max"  # Full history back to IPO / 1980s for long-term pattern detection
        df = yf.Ticker(symbol.upper()).history(period=period, interval="1d")
        if df.empty:
            return pd.DataFrame()
        df = df.rename(columns={"Open": "open", "High": "high", "Low": "low",
                                 "Close": "close", "Volume": "volume"})
        df.index.name = "date"
        df = df.reset_index()
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        return df[["date", "open", "high", "low", "close", "volume"]].tail(limit).reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def _fetch_intraday_via_yfinance(symbol: str, resolution_minutes: int = 30) -> pd.DataFrame:
    """Fetch intraday OHLCV from yfinance for a given resolution."""
    try:
        import yfinance as yf
        interval_map = {1: "1m", 5: "5m", 15: "15m", 30: "30m", 60: "1h"}
        interval = interval_map.get(resolution_minutes, "30m")
        max_days = 7 if resolution_minutes == 1 else 60
        df = yf.Ticker(symbol.upper()).history(period=f"{max_days}d", interval=interval)
        if df.empty:
            return pd.DataFrame()
        df = df.rename(columns={"Open": "open", "High": "high", "Low": "low",
                                 "Close": "close", "Volume": "volume"})
        df.index.name = "date"
        df = df.reset_index()
        # Normalise date column to string so detectors treat it the same as daily
        df["date"] = df["date"].astype(str)
        return df[["date", "open", "high", "low", "close", "volume"]].reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


class PatternService:
    def _ensure_schema(self, session) -> None:
        PatternSignal.__table__.create(bind=session.bind, checkfirst=True)
        PatternDetection.__table__.create(bind=session.bind, checkfirst=True)
        alters = [
            "ALTER TABLE pattern_detections ADD COLUMN IF NOT EXISTS pattern_category VARCHAR(20)",
            "ALTER TABLE pattern_detections ADD COLUMN IF NOT EXISTS status VARCHAR(20)",
            "ALTER TABLE pattern_detections ADD COLUMN IF NOT EXISTS breakout_level DOUBLE PRECISION",
            "ALTER TABLE pattern_detections ADD COLUMN IF NOT EXISTS target DOUBLE PRECISION",
            "ALTER TABLE pattern_detections ADD COLUMN IF NOT EXISTS confidence DOUBLE PRECISION",
            "ALTER TABLE pattern_detections ADD COLUMN IF NOT EXISTS points VARCHAR(4000)",
        ]
        for ddl in alters:
            try:
                session.execute(text(ddl))
            except Exception:
                pass

    def _load_price_data_from_db(self, session, symbol: str, limit: int = 1500) -> tuple[pd.DataFrame, Instrument | None]:
        instrument = session.execute(
            select(Instrument).where(Instrument.symbol == symbol.upper())
        ).scalar_one_or_none()
        if instrument is None:
            return pd.DataFrame(), None
        rows = session.execute(
            select(
                PriceData.date, PriceData.open, PriceData.high,
                PriceData.low, PriceData.close, PriceData.volume,
            )
            .where(PriceData.instrument_id == instrument.id)
            .order_by(PriceData.date.desc())
            .limit(limit)
        ).all()
        if not rows:
            return pd.DataFrame(), instrument
        df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
        df["date"] = df["date"].astype(str)
        return df.sort_values("date").reset_index(drop=True), instrument

    # Fields to pass through from detectors to the API response
    _PASSTHROUGH_KEYS = (
        "overlay_lines", "direction",
        # Point labels (e.g. harmonic XABCD)
        "point_labels",
        # Per-segment styling roles (chair/cup/H&S/etc.)
        "overlay_line_roles",
        # Wyckoff
        "phase", "phase_label", "events", "support_level", "resistance_level",
        # Gann
        "fan_lines", "retracement_levels", "time_cycles",
        "square_of_9", "bearish_fan_lines", "bearish_overlay_lines",
        # Cup & Handle named key prices
        "cup_left_price", "cup_right_price", "cup_bottom_price",
        # Harmonic-specific
        "ratios", "prz_low", "prz_high",
        # Shaded fill zone between two trendlines (channels, wedges, triangles)
        "fill_zone",
        # New detector enriched fields
        "obv_divergence_type",
        "squeeze_bars",
        "nr7_range",
        # Astrology & financial cycles
        "bradley_series", "bradley_turning_points", "upcoming_turning_points",
        "turn_date", "turn_type", "days_until",
        "retro_start", "retro_end", "retro_ongoing",
        "phase_date", "days_delta",
        "ingress_date", "from_sign", "to_sign",
        "aspects", "bullish_aspects", "bearish_aspects", "aspect_date",
        "sq9_planetary_levels",
        "raw_bradley_series", "planet_series",
    )

    @staticmethod
    def _idx_to_date(dates: list, idx: float) -> str:
        """Convert a bar index (possibly float) to a date string, clamped to valid range."""
        n = len(dates)
        i = max(0, min(n - 1, int(round(float(idx)))))
        return str(dates[i])

    def _normalize(self, raw: dict[str, Any], price_data: pd.DataFrame) -> dict[str, Any]:
        status = raw.get("status", "NOT_PRESENT")
        if status not in VALID_STATES:
            status = "NOT_PRESENT"
        base_conf = float(raw.get("confidence", raw.get("probability", 0.0)) or 0.0)

        # Pre-build date lookup for index → date conversion
        dates: list[str] = price_data["date"].astype(str).tolist()

        # Convert points [[idx, price], ...] → [[date, price], ...]
        raw_points = raw.get("points", [])
        date_points: list[list] = []
        for pt in raw_points:
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                date_points.append([self._idx_to_date(dates, pt[0]), float(pt[1])])

        normalized = {
            "pattern_name":     raw.get("pattern_name", "Unknown"),
            "pattern_category": raw.get("pattern_category", "chart"),
            "status":           status,
            "breakout_level":   raw.get("breakout_level"),
            "target":           raw.get("target", raw.get("projected_target")),
            "invalidation_level": raw.get("invalidation_level"),
            "confidence":       base_conf,
            "points":           date_points,
        }

        # Pass through enriched detector fields (overlay_lines, Wyckoff phases, Gann fans, etc.)
        for key in self._PASSTHROUGH_KEYS:
            if key in raw:
                normalized[key] = raw[key]

        # Convert overlay_lines [[idx,price],[idx,price]] → [[date,price],[date,price]]
        if "overlay_lines" in raw:
            date_lines: list[list] = []
            for seg in raw["overlay_lines"]:
                if not (isinstance(seg, (list, tuple)) and len(seg) >= 2):
                    continue
                a, b = seg[0], seg[1]
                if not (isinstance(a, (list, tuple)) and isinstance(b, (list, tuple))):
                    continue
                if len(a) < 2 or len(b) < 2:
                    continue
                da = self._idx_to_date(dates, a[0])
                db = self._idx_to_date(dates, b[0])
                # Skip zero-length segments (same date AND same price = degenerate)
                if da == db and abs(float(a[1]) - float(b[1])) < 1e-9:
                    continue
                date_lines.append([[da, float(a[1])], [db, float(b[1])]])
            normalized["overlay_lines"] = date_lines
        # Convert event_points [{label, index, price}] → [{label, date, price}]
        if "event_points" in raw:
            date_ep: list[dict] = []
            for ep in raw.get("event_points", []):
                if isinstance(ep, dict) and "index" in ep and "price" in ep:
                    date_ep.append({
                        "label": str(ep.get("label", "")),
                        "date": self._idx_to_date(dates, ep["index"]),
                        "price": float(ep["price"]),
                    })
            normalized["event_points"] = date_ep

        if status not in ("NOT_PRESENT", "FAILED"):
            confluence = compute_confluence(normalized, price_data)
            # Blend: 60% detector confidence + 40% confluence so strong patterns keep their score
            blended = 0.60 * base_conf + 0.40 * confluence["confluence_score"]
            import numpy as np
            normalized["confidence"] = float(np.clip(blended, 0, 100))
            normalized["confluence"] = confluence
        return normalized

    def _run_all_detectors(
        self, symbol: str, df: pd.DataFrame, resolution_minutes: int = 0,
    ) -> list[dict[str, Any]]:
        """Run every detector, collect ALL results (not just best-per-category).

        Skips detectors that don't apply at the given resolution_minutes.
        """
        results: list[dict[str, Any]] = []
        for category, detector_fn in CATEGORY_DETECTORS:
            # Timeframe routing — skip detectors that need longer bars
            min_res = CATEGORY_MIN_RESOLUTION.get(category, 0)
            if resolution_minutes > 0 and resolution_minutes < min_res:
                continue
            try:
                # chart_patterns.detect_all accepts resolution_minutes
                if category == "chart":
                    raw = detector_fn(symbol.upper(), df, resolution_minutes)
                else:
                    raw = detector_fn(symbol.upper(), df)
                # detector_fn returns either a list[dict] or a single dict
                if isinstance(raw, list):
                    for item in raw:
                        item.setdefault("pattern_category", category)
                        results.append(self._normalize(item, df))
                else:
                    raw.setdefault("pattern_category", category)
                    results.append(self._normalize(raw, df))
            except Exception as exc:
                logger.warning(
                    "Detector %s failed for %s (res=%dm): %s",
                    category, symbol, resolution_minutes, exc, exc_info=True,
                )
        results.sort(key=lambda p: p.get("confidence", 0.0), reverse=True)
        # Deduplicate by pattern_name — keep the highest-confidence instance
        seen: dict[str, dict] = {}
        for r in results:
            name = r.get("pattern_name", "")
            if name and name not in seen:
                seen[name] = r
        return list(seen.values())

    @staticmethod
    def _to_py(v: Any) -> Any:
        """Convert numpy scalars / NaN / Inf to plain Python types safe for psycopg2."""
        import math
        if v is None:
            return None
        try:
            import numpy as _np
            if isinstance(v, _np.generic):
                v = v.item()
        except ImportError:
            pass
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v

    def _persist_pattern_detections(self, session, instrument: Instrument, patterns: list[dict[str, Any]]) -> None:
        today = date.today()
        rows: list[dict] = []
        for p in patterns:
            if p["status"] in ("NOT_PRESENT",):
                continue
            breakout = self._to_py(p.get("breakout_level"))
            target   = self._to_py(p.get("target"))
            direction = p.get("direction", "neutral")
            if direction not in ("bullish", "bearish", "neutral"):
                direction = "neutral"
            if direction == "neutral" and breakout is not None and target is not None:
                direction = "bullish" if float(target) >= float(breakout) else "bearish"
            rows.append(dict(
                instrument_id=instrument.id,
                date=today,
                pattern_name=str(p["pattern_name"]),
                pattern_type="chart",
                pattern_category=str(p["pattern_category"]),
                status=str(p["status"]),
                direction=direction,
                strength=float(p.get("confidence", 0.0)) / 100.0,
                price_level=breakout,
                breakout_level=breakout,
                target=target,
                confidence=float(p.get("confidence", 0.0)),
                points=json.dumps(p.get("points", [])),
                created_at=datetime.utcnow(),
            ))
        if rows:
            # Deduplicate by (instrument_id, date, pattern_name) — keep highest confidence
            deduped: dict[tuple, dict] = {}
            for row in rows:
                key = (row["instrument_id"], row["date"], row["pattern_name"])
                if key not in deduped or row["confidence"] > deduped[key]["confidence"]:
                    deduped[key] = row
            rows = list(deduped.values())
            stmt = pg_insert(PatternDetection.__table__).values(rows)
            stmt = stmt.on_conflict_do_update(
                constraint="uq_pattern_instrument_date_name",
                set_={
                    "pattern_category": stmt.excluded.pattern_category,
                    "status":           stmt.excluded.status,
                    "direction":        stmt.excluded.direction,
                    "strength":         stmt.excluded.strength,
                    "price_level":      stmt.excluded.price_level,
                    "breakout_level":   stmt.excluded.breakout_level,
                    "target":           stmt.excluded.target,
                    "confidence":       stmt.excluded.confidence,
                    "points":           stmt.excluded.points,
                    "created_at":       stmt.excluded.created_at,
                },
            )
            session.execute(stmt)

    def _persist_pattern_signals(self, session, symbol: str, patterns: list[dict[str, Any]]) -> None:
        now = datetime.utcnow()
        rows: list[PatternSignal] = []
        seen_patterns: set[str] = set()
        for p in patterns:
            if p["status"] in ("NOT_PRESENT",):
                continue
            pname = str(p["pattern_name"])
            if pname in seen_patterns:
                continue
            seen_patterns.add(pname)
            rows.append(PatternSignal(
                symbol=symbol.upper(),
                pattern=pname,
                status=str(p["status"]),
                breakout_level=self._to_py(p.get("breakout_level")),
                target=self._to_py(p.get("target")),
                probability=float(p.get("confidence", 0.0)),
                timestamp=now,
            ))
        if rows:
            session.add_all(rows)

    def detect_for_symbol(self, symbol: str, resolution_minutes: int = 0) -> list[dict[str, Any]]:
        """
        Run pattern detection for a symbol.

        resolution_minutes = 0 → daily bars (from DB, yfinance fallback)
        resolution_minutes > 0 → intraday bars at that resolution via yfinance
        """
        # ── Intraday path: always use yfinance, no DB persist ──────────────
        if resolution_minutes > 0:
            df = _fetch_intraday_via_yfinance(symbol, resolution_minutes)
            if df.empty:
                return []
            return self._run_all_detectors(symbol, df, resolution_minutes)

        # ── Daily path ──────────────────────────────────────────────────────
        with get_session() as session:
            self._ensure_schema(session)
            df, instrument = self._load_price_data_from_db(session, symbol)

            # Fallback: symbol not in DB or no price rows → yfinance
            if df.empty:
                df = _fetch_daily_via_yfinance(symbol)
                if df.empty:
                    return []
                detected = self._run_all_detectors(symbol, df, resolution_minutes)
                # Can't persist without instrument row — return directly
                return detected

            detected = self._run_all_detectors(symbol, df, resolution_minutes)
            self._persist_pattern_detections(session, instrument, detected)
            self._persist_pattern_signals(session, symbol, detected)
            session.commit()
            return detected
