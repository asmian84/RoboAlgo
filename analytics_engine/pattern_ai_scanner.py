"""Pattern AI Scanner for RoboAlgo.

Runs the Pattern Library detectors in parallel and stores actionable results in
`pattern_scan_results`.
"""

from __future__ import annotations

import asyncio
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import insert, select, text

from database.connection import get_engine, get_session
from database.models import Base, Instrument, PatternDetection, PatternScanResult
from pattern_engine.chart_patterns import detect as detect_chart_patterns
from pattern_engine.gann_levels import detect as detect_gann_levels
from pattern_engine.harmonic_patterns import detect as detect_harmonic_patterns
from pattern_engine.validation_engine import compute_confluence
from pattern_engine.wyckoff_structures import detect as detect_wyckoff_structures


PATTERN_LIBRARY_DETECTORS = (
    detect_chart_patterns,
    detect_harmonic_patterns,
    detect_gann_levels,
    detect_wyckoff_structures,
)

VALID_STATES = {"NOT_PRESENT", "FORMING", "READY", "BREAKOUT", "FAILED", "COMPLETED"}
ACTIONABLE_STATES = {"READY", "BREAKOUT", "COMPLETED"}


def _scan_symbol_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    symbol = payload["symbol"]
    df = pd.DataFrame(
        {
            "date": payload["date"],
            "open": payload["open"],
            "high": payload["high"],
            "low": payload["low"],
            "close": payload["close"],
            "volume": payload["volume"],
        }
    ).dropna(subset=["high", "low", "close"])
    if len(df) < 60:
        return []

    detected = [det(symbol, df) for det in PATTERN_LIBRARY_DETECTORS]
    cleaned: list[dict[str, Any]] = []
    for p in detected:
        state = p.get("status", "NOT_PRESENT")
        if state not in VALID_STATES:
            state = "NOT_PRESENT"
        if state not in ACTIONABLE_STATES:
            continue
        breakout = p.get("breakout_level")
        target = p.get("target")
        if breakout is None or target is None:
            continue
        confluence = compute_confluence(p, df)
        cleaned.append(
            {
                "symbol": symbol,
                "pattern": str(p.get("pattern_name", "Unknown")),
                "status": state,
                "probability": float(round(float(confluence["confluence_score"]) / 100.0, 4)),
                "breakout_level": float(round(float(breakout), 6)),
                "target": float(round(float(target), 6)),
            }
        )

    cleaned.sort(key=lambda x: x["probability"], reverse=True)
    return cleaned[:2]


def _scan_chunk(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        rows.extend(_scan_symbol_payload(payload))
    return rows


class PatternAIScanner:
    """Universe scanner with process workers."""

    def __init__(
        self,
        max_symbols: int = 20000,
        lookback_bars: int = 260,
        max_workers: int | None = None,
        timeout_seconds: int = 60,
    ):
        self.max_symbols = max_symbols
        self.lookback_bars = lookback_bars
        self.max_workers = max_workers or max(1, (os.cpu_count() or 4) - 1)
        self.timeout_seconds = timeout_seconds

    async def run(self) -> list[dict[str, Any]]:
        Base.metadata.create_all(get_engine(), tables=[PatternScanResult.__table__])
        payloads = self._load_payloads(self.max_symbols, self.lookback_bars)
        if not payloads:
            return []

        worker_count = min(self.max_workers, len(payloads))
        chunk_count = max(worker_count * 2, 1)
        chunks = [list(c) for c in np.array_split(payloads, chunk_count) if len(c) > 0]

        loop = asyncio.get_running_loop()
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [loop.run_in_executor(executor, _scan_chunk, chunk) for chunk in chunks]
            results_nested = await asyncio.wait_for(asyncio.gather(*futures), timeout=self.timeout_seconds)

        rows = [r for chunk_rows in results_nested for r in chunk_rows]
        self._store_results(rows)
        return rows

    def _load_payloads(self, max_symbols: int, lookback_bars: int) -> list[dict[str, Any]]:
        q = text(
            """
            WITH selected AS (
                SELECT id, symbol
                FROM instruments
                ORDER BY symbol
                LIMIT :max_symbols
            ),
            ranked AS (
                SELECT
                    p.instrument_id,
                    p.date,
                    p.open,
                    p.high,
                    p.low,
                    p.close,
                    p.volume,
                    ROW_NUMBER() OVER (
                        PARTITION BY p.instrument_id
                        ORDER BY p.date DESC
                    ) AS rn
                FROM price_data p
                JOIN selected s ON s.id = p.instrument_id
            )
            SELECT s.symbol, r.date, r.open, r.high, r.low, r.close, r.volume
            FROM ranked r
            JOIN selected s ON s.id = r.instrument_id
            WHERE r.rn <= :lookback_bars
            ORDER BY s.symbol, r.date
            """
        )
        df = pd.read_sql_query(q, get_engine(), params={"max_symbols": max_symbols, "lookback_bars": lookback_bars})
        if df.empty:
            return []

        payloads: list[dict[str, Any]] = []
        for symbol, g in df.groupby("symbol", sort=False):
            if len(g) < 60:
                continue
            payloads.append(
                {
                    "symbol": symbol,
                    "date": g["date"].tolist(),
                    "open": g["open"].fillna(g["close"]).astype(float).tolist(),
                    "high": g["high"].astype(float).tolist(),
                    "low": g["low"].astype(float).tolist(),
                    "close": g["close"].astype(float).tolist(),
                    "volume": g["volume"].fillna(0.0).astype(float).tolist(),
                }
            )
        return payloads

    def _store_results(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        now = datetime.utcnow()
        today = now.date()
        payload = [
            {
                "symbol": r["symbol"],
                "pattern": r["pattern"],
                "probability": r["probability"],
                "breakout_level": r["breakout_level"],
                "target": r["target"],
                "timestamp": now,
            }
            for r in rows
        ]
        with get_session() as session:
            session.query(PatternScanResult).delete(synchronize_session=False)
            session.execute(insert(PatternScanResult), payload)
            symbols = sorted({r["symbol"] for r in rows})
            instruments = session.execute(
                select(Instrument).where(Instrument.symbol.in_(symbols))
            ).scalars().all()
            inst_map = {i.symbol: i.id for i in instruments}
            session.query(PatternDetection).where(PatternDetection.date == today).delete(synchronize_session=False)
            for r in rows:
                inst_id = inst_map.get(r["symbol"])
                if inst_id is None:
                    continue
                pname = str(r["pattern"])
                cat = "chart"
                if "gann" in pname.lower():
                    cat = "gann"
                elif "harmonic" in pname.lower() or pname in {"Gartley", "Bat", "Butterfly"}:
                    cat = "harmonic"
                elif "wyckoff" in pname.lower():
                    cat = "wyckoff"
                session.add(
                    PatternDetection(
                        instrument_id=inst_id,
                        date=today,
                        pattern_name=pname,
                        pattern_type="chart",
                        pattern_category=cat,
                        status=r.get("status", "READY"),
                        direction="bullish",
                        strength=float(r.get("probability", 0.0)),
                        price_level=float(r["breakout_level"]),
                        breakout_level=float(r["breakout_level"]),
                        target=float(r["target"]),
                        confidence=float(r.get("probability", 0.0)) * 100.0,
                        points="[]",
                        created_at=now,
                    )
                )
            session.commit()


async def run_pattern_ai_scan(
    max_symbols: int = 20000,
    lookback_bars: int = 260,
    max_workers: int | None = None,
    timeout_seconds: int = 60,
) -> list[dict[str, Any]]:
    scanner = PatternAIScanner(
        max_symbols=max_symbols,
        lookback_bars=lookback_bars,
        max_workers=max_workers,
        timeout_seconds=timeout_seconds,
    )
    return await scanner.run()


def run_pattern_ai_scan_sync(
    max_symbols: int = 20000,
    lookback_bars: int = 260,
    max_workers: int | None = None,
    timeout_seconds: int = 60,
) -> list[dict[str, Any]]:
    return asyncio.run(
        run_pattern_ai_scan(
            max_symbols=max_symbols,
            lookback_bars=lookback_bars,
            max_workers=max_workers,
            timeout_seconds=timeout_seconds,
        )
    )
