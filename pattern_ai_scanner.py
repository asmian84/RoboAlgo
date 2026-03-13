"""
Pattern AI Scanner

Asynchronous pattern scanner for large universes.

Detected patterns:
- Chair Pattern
- Harmonic (Gartley, Bat, Butterfly)
- Channels
- Cup and Handle
- Megaphone

Output rows:
{
  "symbol": str,
  "pattern": str,
  "probability": float,
  "breakout_level": float,
  "target": float,
}
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date
from time import perf_counter
from typing import Iterable

import numpy as np
import pandas as pd
from sqlalchemy import insert, text

from database.connection import get_engine, get_session
from database.models import Base, PatternScanResult


@dataclass
class PatternCandidate:
    pattern: str
    structure_quality: float
    breakout_level: float
    target: float
    breakout_index: int


class PatternAIScanner:
    """Async scanner for pattern candidates across large symbol universes."""

    def __init__(
        self,
        max_symbols: int = 5000,
        lookback_bars: int = 260,
        swing_window: int = 3,
        max_workers: int = 64,
        timeout_seconds: int = 60,
    ):
        self.max_symbols = max_symbols
        self.lookback_bars = lookback_bars
        self.swing_window = swing_window
        self.max_workers = max_workers
        self.timeout_seconds = timeout_seconds

    async def scan_universe(self) -> list[dict]:
        """Run async scan, persist, and return result payload rows."""
        t0 = perf_counter()

        # Ensure table exists even if schema migration has not run yet.
        Base.metadata.create_all(get_engine(), tables=[PatternScanResult.__table__])

        universe = self._load_universe_ohlcv(self.max_symbols, self.lookback_bars)
        semaphore = asyncio.Semaphore(self.max_workers)

        async def run_one(symbol: str, inst_id: int, df: pd.DataFrame):
            async with semaphore:
                return await asyncio.to_thread(self._scan_symbol, symbol, inst_id, df)

        tasks = [
            asyncio.create_task(run_one(symbol, meta["instrument_id"], meta["df"]))
            for symbol, meta in universe.items()
        ]

        timeout = max(5, self.timeout_seconds - 5)
        results: list[dict] = []
        try:
            scanned = await asyncio.wait_for(asyncio.gather(*tasks), timeout=timeout)
            for row in scanned:
                if row is not None:
                    results.append(row)
        except asyncio.TimeoutError:
            for task in tasks:
                if task.done() and not task.cancelled() and task.exception() is None:
                    row = task.result()
                    if row is not None:
                        results.append(row)
                else:
                    task.cancel()

        self._store_results(results)

        elapsed = perf_counter() - t0
        print(
            f"PatternAIScanner: scanned={len(universe)} matched={len(results)} "
            f"elapsed={elapsed:.2f}s"
        )
        return [
            {
                "symbol": r["symbol"],
                "pattern": r["pattern"],
                "probability": r["probability"],
                "breakout_level": r["breakout_level"],
                "target": r["target"],
            }
            for r in results
        ]

    def _load_universe_ohlcv(self, max_symbols: int, lookback_bars: int) -> dict[str, dict]:
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
            SELECT
                s.id   AS instrument_id,
                s.symbol,
                r.date,
                r.open,
                r.high,
                r.low,
                r.close,
                r.volume
            FROM ranked r
            JOIN selected s ON s.id = r.instrument_id
            WHERE r.rn <= :lookback_bars
            ORDER BY s.symbol, r.date
            """
        )

        df = pd.read_sql_query(q, get_engine(), params={"max_symbols": max_symbols, "lookback_bars": lookback_bars})
        if df.empty:
            return {}

        df["date"] = pd.to_datetime(df["date"])
        grouped: dict[str, dict] = {}
        for symbol, g in df.groupby("symbol", sort=False):
            grouped[symbol] = {
                "instrument_id": int(g["instrument_id"].iloc[0]),
                "df": g[["date", "open", "high", "low", "close", "volume"]].reset_index(drop=True),
            }
        return grouped

    def _scan_symbol(self, symbol: str, instrument_id: int, df: pd.DataFrame) -> dict | None:
        if len(df) < 80:
            return None

        highs = df["high"].to_numpy(dtype=float)
        lows = df["low"].to_numpy(dtype=float)
        closes = df["close"].to_numpy(dtype=float)
        volumes = df["volume"].fillna(0.0).to_numpy(dtype=float)

        swings = self._swing_points(highs, lows, self.swing_window)
        if len(swings) < 6:
            return None

        candidates: list[PatternCandidate] = []
        candidates.extend(self._detect_chair(swings, closes))
        candidates.extend(self._detect_harmonics(swings, closes))
        candidates.extend(self._detect_channel(swings, closes))
        candidates.extend(self._detect_cup_handle(closes))
        candidates.extend(self._detect_megaphone(swings, closes))

        if not candidates:
            return None

        scored: list[dict] = []
        for c in candidates:
            volume_confirmation = self._volume_confirmation(volumes, c.breakout_index)
            liquidity_alignment = self._liquidity_alignment(swings, c.breakout_level)
            market_regime = self._market_regime_score(closes)
            momentum = self._momentum_score(closes)
            probability = float(
                np.clip(
                    0.40 * c.structure_quality
                    + 0.15 * volume_confirmation
                    + 0.15 * liquidity_alignment
                    + 0.15 * market_regime
                    + 0.15 * momentum,
                    0.0,
                    1.0,
                )
            )

            scored.append(
                {
                    "instrument_id": instrument_id,
                    "symbol": symbol,
                    "pattern": c.pattern,
                    "probability": round(probability, 4),
                    "breakout_level": float(round(c.breakout_level, 6)),
                    "target": float(round(c.target, 6)),
                    "structure_quality": float(round(c.structure_quality, 4)),
                    "volume_confirmation": float(round(volume_confirmation, 4)),
                    "liquidity_alignment": float(round(liquidity_alignment, 4)),
                    "market_regime": float(round(market_regime, 4)),
                    "momentum": float(round(momentum, 4)),
                }
            )

        # Keep the strongest single actionable pattern per symbol for fast scanning.
        return max(scored, key=lambda x: x["probability"]) if scored else None

    @staticmethod
    def _swing_points(high: np.ndarray, low: np.ndarray, window: int) -> list[tuple[int, float, str]]:
        n = len(high)
        swings: list[tuple[int, float, str]] = []
        for i in range(window, n - window):
            h = high[i]
            l = low[i]
            if h >= np.max(high[i - window : i + window + 1]):
                swings.append((i, float(h), "H"))
            if l <= np.min(low[i - window : i + window + 1]):
                swings.append((i, float(l), "L"))
        swings.sort(key=lambda x: x[0])
        return swings

    @staticmethod
    def _alternating(points: Iterable[tuple[int, float, str]]) -> bool:
        pts = list(points)
        return all(pts[i][2] != pts[i - 1][2] for i in range(1, len(pts)))

    def _detect_chair(self, swings: list[tuple[int, float, str]], close: np.ndarray) -> list[PatternCandidate]:
        highs = [s for s in swings[-12:] if s[2] == "H"]
        lows = [s for s in swings[-12:] if s[2] == "L"]
        if len(highs) < 3 or len(lows) < 3:
            return []

        h3 = highs[-3:]
        l3 = lows[-3:]
        descending = h3[0][1] > h3[1][1] > h3[2][1] and l3[0][1] > l3[1][1] > l3[2][1]
        breakout_level = h3[-1][1]
        broke_out = close[-1] > breakout_level * 1.002
        if not (descending and broke_out):
            return []

        depth = breakout_level - min(l[1] for l in l3)
        target = breakout_level + depth
        quality = float(np.clip(0.65 + min(depth / max(breakout_level, 1e-9), 0.2), 0.0, 1.0))

        return [
            PatternCandidate(
                pattern="Chair Pattern",
                structure_quality=quality,
                breakout_level=breakout_level,
                target=target,
                breakout_index=len(close) - 1,
            )
        ]

    def _detect_harmonics(self, swings: list[tuple[int, float, str]], close: np.ndarray) -> list[PatternCandidate]:
        if len(swings) < 8:
            return []

        def fit_range(v: float, lo: float, hi: float) -> float:
            if lo <= v <= hi:
                mid = (lo + hi) / 2.0
                return 1.0 - abs(v - mid) / max(mid, 1e-9)
            d = min(abs(v - lo), abs(v - hi))
            return max(0.0, 1.0 - d / max(hi, 1e-9))

        best: list[PatternCandidate] = []
        templates = {
            "Harmonic Gartley": ((0.55, 0.70), (0.382, 0.886), (1.272, 1.618)),
            "Harmonic Bat": ((0.382, 0.50), (0.382, 0.886), (1.618, 2.618)),
            "Harmonic Butterfly": ((0.75, 0.85), (0.382, 0.886), (1.618, 2.24)),
        }

        scan_swings = swings[-20:]
        for i in range(len(scan_swings) - 4):
            seq = scan_swings[i : i + 5]
            if not self._alternating(seq):
                continue
            _, X, _ = seq[0]
            _, A, _ = seq[1]
            cidx, B, _ = seq[2]
            _, C, _ = seq[3]
            didx, D, dtype = seq[4]

            XA = abs(A - X)
            AB = abs(B - A)
            BC = abs(C - B)
            CD = abs(D - C)
            if min(XA, AB, BC) < 1e-9:
                continue

            r_ab = AB / XA
            r_bc = BC / AB
            r_cd = CD / BC

            for name, (ab_rng, bc_rng, cd_rng) in templates.items():
                score = (
                    fit_range(r_ab, *ab_rng)
                    + fit_range(r_bc, *bc_rng)
                    + fit_range(r_cd, *cd_rng)
                ) / 3.0

                # Prefer bullish completions (D swing low) with breakout over C.
                if dtype != "L" or close[-1] <= C * 1.001:
                    continue

                breakout = max(C, close[-1])
                target = breakout + abs(C - D)
                best.append(
                    PatternCandidate(
                        pattern=name,
                        structure_quality=float(np.clip(score, 0.0, 1.0)),
                        breakout_level=float(breakout),
                        target=float(target),
                        breakout_index=max(cidx, didx),
                    )
                )

        return sorted(best, key=lambda x: x.structure_quality, reverse=True)[:1]

    def _detect_channel(self, swings: list[tuple[int, float, str]], close: np.ndarray) -> list[PatternCandidate]:
        highs = np.array([(i, p) for i, p, t in swings[-60:] if t == "H"], dtype=float)
        lows = np.array([(i, p) for i, p, t in swings[-60:] if t == "L"], dtype=float)
        if len(highs) < 3 or len(lows) < 3:
            return []

        hi_slope, hi_intercept = np.polyfit(highs[:, 0], highs[:, 1], 1)
        lo_slope, lo_intercept = np.polyfit(lows[:, 0], lows[:, 1], 1)

        if np.sign(hi_slope) != np.sign(lo_slope):
            return []

        slope_gap = abs(hi_slope - lo_slope) / max(abs(hi_slope) + abs(lo_slope), 1e-9)
        if slope_gap > 0.35:
            return []

        n = len(close) - 1
        upper_now = hi_slope * n + hi_intercept
        lower_now = lo_slope * n + lo_intercept
        width = max(upper_now - lower_now, 1e-9)

        if close[-1] > upper_now * 1.001:
            quality = float(np.clip(0.7 * (1.0 - slope_gap) + 0.3 * min(width / max(close[-1], 1e-9), 0.2), 0, 1))
            return [
                PatternCandidate(
                    pattern="Channel Breakout",
                    structure_quality=quality,
                    breakout_level=float(upper_now),
                    target=float(close[-1] + width),
                    breakout_index=n,
                )
            ]
        return []

    def _detect_cup_handle(self, close: np.ndarray) -> list[PatternCandidate]:
        n = len(close)
        if n < 90:
            return []

        w = close[-140:] if n >= 140 else close
        m = len(w)
        left = int(np.argmax(w[: max(20, m // 3)]))
        right_base = int(2 * m / 3)
        right = right_base + int(np.argmax(w[right_base:]))
        if right <= left + 15:
            return []

        trough = left + int(np.argmin(w[left:right + 1]))
        left_rim = w[left]
        right_rim = w[right]
        bottom = w[trough]

        depth = (max(left_rim, right_rim) - bottom) / max(max(left_rim, right_rim), 1e-9)
        rim_similarity = 1.0 - abs(left_rim - right_rim) / max(left_rim, right_rim, 1e-9)

        if depth < 0.08 or depth > 0.45 or rim_similarity < 0.92:
            return []

        handle = w[right:]
        if len(handle) < 5:
            return []
        handle_pullback = (np.max(handle) - np.min(handle)) / max(np.max(handle), 1e-9)
        if handle_pullback > depth * 0.6:
            return []

        breakout = max(left_rim, right_rim)
        if w[-1] <= breakout * 1.002:
            return []

        quality = float(np.clip(0.45 + 0.35 * rim_similarity + 0.20 * min(depth / 0.25, 1.0), 0.0, 1.0))
        target = breakout + depth * breakout
        return [
            PatternCandidate(
                pattern="Cup and Handle",
                structure_quality=quality,
                breakout_level=float(breakout),
                target=float(target),
                breakout_index=n - 1,
            )
        ]

    def _detect_megaphone(self, swings: list[tuple[int, float, str]], close: np.ndarray) -> list[PatternCandidate]:
        highs = [s for s in swings[-20:] if s[2] == "H"]
        lows = [s for s in swings[-20:] if s[2] == "L"]
        if len(highs) < 3 or len(lows) < 3:
            return []

        h3 = highs[-3:]
        l3 = lows[-3:]
        expanding = h3[0][1] < h3[1][1] < h3[2][1] and l3[0][1] > l3[1][1] > l3[2][1]
        if not expanding:
            return []

        breakout = h3[-1][1]
        if close[-1] <= breakout * 1.001:
            return []

        width = h3[-1][1] - l3[-1][1]
        quality = float(np.clip(0.6 + min(width / max(close[-1], 1e-9), 0.25), 0.0, 1.0))
        return [
            PatternCandidate(
                pattern="Megaphone",
                structure_quality=quality,
                breakout_level=float(breakout),
                target=float(close[-1] + width),
                breakout_index=len(close) - 1,
            )
        ]

    @staticmethod
    def _volume_confirmation(volume: np.ndarray, idx: int) -> float:
        if len(volume) < 30:
            return 0.5
        idx = max(5, min(idx, len(volume) - 1))
        baseline = np.nanmean(volume[max(0, idx - 25) : idx - 5])
        trigger = np.nanmean(volume[max(0, idx - 2) : min(len(volume), idx + 3)])
        if not np.isfinite(baseline) or baseline <= 0:
            return 0.5
        return float(np.clip(trigger / baseline / 2.0, 0.0, 1.0))

    @staticmethod
    def _liquidity_alignment(swings: list[tuple[int, float, str]], breakout_level: float) -> float:
        if breakout_level <= 0:
            return 0.0
        recent = swings[-30:]
        tol = breakout_level * 0.006
        touches = sum(1 for _, p, _ in recent if abs(p - breakout_level) <= tol)
        return float(np.clip(touches / 4.0, 0.0, 1.0))

    @staticmethod
    def _market_regime_score(close: np.ndarray) -> float:
        if len(close) < 60:
            return 0.5
        s = pd.Series(close)
        sma20 = s.rolling(20).mean().iloc[-1]
        sma50 = s.rolling(50).mean().iloc[-1]
        trend = 1.0 if sma20 > sma50 and close[-1] > sma20 else 0.4

        rets = s.pct_change().dropna()
        vol = float(rets.tail(20).std()) if not rets.empty else 0.02
        # Prefer moderate vol (not dead, not chaotic).
        vol_score = float(np.clip(1.0 - abs(vol - 0.02) / 0.03, 0.0, 1.0))
        return float(np.clip(0.65 * trend + 0.35 * vol_score, 0.0, 1.0))

    @staticmethod
    def _momentum_score(close: np.ndarray) -> float:
        if len(close) < 30:
            return 0.5
        s = pd.Series(close)
        delta = s.diff()
        up = delta.clip(lower=0).rolling(14).mean().iloc[-1]
        down = (-delta.clip(upper=0)).rolling(14).mean().iloc[-1]
        rs = up / max(down, 1e-9)
        rsi = 100.0 - (100.0 / (1.0 + rs))

        ret20 = float(s.iloc[-1] / s.iloc[-21] - 1.0) if len(s) > 21 else 0.0
        rsi_score = np.clip((rsi - 45.0) / 30.0, 0.0, 1.0)
        ret_score = np.clip((ret20 + 0.02) / 0.08, 0.0, 1.0)
        return float(np.clip(0.6 * rsi_score + 0.4 * ret_score, 0.0, 1.0))

    def _store_results(self, rows: list[dict]) -> None:
        if not rows:
            return

        scan_day = date.today()
        with get_session() as session:
            inst_ids = sorted({r["instrument_id"] for r in rows})
            session.query(PatternScanResult).filter(
                PatternScanResult.scan_date == scan_day,
                PatternScanResult.instrument_id.in_(inst_ids),
            ).delete(synchronize_session=False)

            payload = [
                {
                    "instrument_id": r["instrument_id"],
                    "scan_date": scan_day,
                    "symbol": r["symbol"],
                    "pattern": r["pattern"],
                    "probability": r["probability"],
                    "breakout_level": r["breakout_level"],
                    "target": r["target"],
                    "structure_quality": r["structure_quality"],
                    "volume_confirmation": r["volume_confirmation"],
                    "liquidity_alignment": r["liquidity_alignment"],
                    "market_regime": r["market_regime"],
                    "momentum": r["momentum"],
                }
                for r in rows
            ]

            session.execute(insert(PatternScanResult), payload)
            session.commit()


async def run_pattern_ai_scan(
    max_symbols: int = 5000,
    lookback_bars: int = 260,
    max_workers: int = 64,
    timeout_seconds: int = 60,
) -> list[dict]:
    scanner = PatternAIScanner(
        max_symbols=max_symbols,
        lookback_bars=lookback_bars,
        max_workers=max_workers,
        timeout_seconds=timeout_seconds,
    )
    return await scanner.scan_universe()


def run_pattern_ai_scan_sync(
    max_symbols: int = 5000,
    lookback_bars: int = 260,
    max_workers: int = 64,
    timeout_seconds: int = 60,
) -> list[dict]:
    return asyncio.run(
        run_pattern_ai_scan(
            max_symbols=max_symbols,
            lookback_bars=lookback_bars,
            max_workers=max_workers,
            timeout_seconds=timeout_seconds,
        )
    )


if __name__ == "__main__":
    output = run_pattern_ai_scan_sync()
    print(f"Stored {len(output)} pattern scan rows")
