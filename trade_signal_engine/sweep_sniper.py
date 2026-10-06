"""
SweepSniper — Liquidity Sweep Entry Detection & Exit Target Mapping

Detects institutional liquidity sweeps (stop hunts) at support pools and maps
resistance-pool exit targets. Every entry is a reclaimed sweep; every exit
targets the next cluster of resting stops above.

LONG only. Works across all timeframes via the profile parameter:
  SCALP    — 5m/15m bars, swing order=3, lookback=60
  SWING    — 1h/4h bars, swing order=5, lookback=100
  POSITION — daily bars, swing order=5, lookback=120

Self-contained: does NOT import from smc_engine or structure_engine.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Profile configuration
# ---------------------------------------------------------------------------
PROFILE_CONFIG = {
    "SCALP":    {"swing_order": 3, "lookback": 60},
    "SWING":    {"swing_order": 5, "lookback": 100},
    "POSITION": {"swing_order": 5, "lookback": 120},
}

# Clustering tolerance — two swing points within 0.5% are the same pool
CLUSTER_PCT = 0.005

# Sweep detection window — check last N bars for a sweep event
SWEEP_WINDOW = 5

# ATR period
ATR_PERIOD = 14

# Minimum bars required to run analysis
MIN_BARS = 20


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class SweepResult:
    """Output of SweepSniper.analyze()."""

    # Sweep detection
    has_sweep: bool = False
    sweep_score: float = 0.0          # 0-100
    sweep_level: float = 0.0          # the support level that was swept
    sweep_bar_index: int = -1

    # Entry plan
    entry: float = 0.0
    stop: float = 0.0
    target: float = 0.0               # T1 — nearest liquidity pool above
    extended_target: float = 0.0      # T2 — second pool
    runner_target: float = 0.0        # T3 — third pool
    risk_reward: float = 0.0

    # Liquidity map
    support_pools: list[dict] = field(default_factory=list)
    resistance_pools: list[dict] = field(default_factory=list)

    # Context flags
    base_compressed: bool = False
    volume_confirmed: bool = False
    wick_rejection: bool = False
    stoch_oversold: bool = False       # Slow STO was in oversold zone (<20)
    stoch_cross: bool = False          # %K crossed above %D (bullish cross)
    stoch_k: float = 50.0
    stoch_d: float = 50.0
    notes: list[str] = field(default_factory=list)

    # Component breakdown for debugging
    components: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# SweepSniper
# ---------------------------------------------------------------------------
class SweepSniper:
    """Detect liquidity-sweep entries and map pool-based exit targets."""

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def analyze(
        self,
        df: pd.DataFrame,
        current_price: float,
        profile: str = "SWING",
    ) -> SweepResult:
        """Run full sweep analysis on an OHLCV DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain lowercase columns: open, high, low, close, volume.
        current_price : float
            Latest market price (may differ from last bar close in real-time).
        profile : str
            One of SCALP, SWING, POSITION.

        Returns
        -------
        SweepResult
        """
        result = SweepResult()
        profile = profile.upper()

        # -- Validate inputs ------------------------------------------------
        if df is None or df.empty:
            result.notes.append("Empty DataFrame — nothing to analyze.")
            return result

        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(set(df.columns)):
            missing = required - set(df.columns)
            result.notes.append(f"Missing columns: {missing}")
            return result

        if len(df) < MIN_BARS:
            result.notes.append(
                f"Insufficient bars ({len(df)}); need at least {MIN_BARS}."
            )
            return result

        cfg = PROFILE_CONFIG.get(profile)
        if cfg is None:
            result.notes.append(
                f"Unknown profile '{profile}'. Use SCALP, SWING, or POSITION."
            )
            return result

        order = cfg["swing_order"]
        lookback = cfg["lookback"]

        # Trim to lookback window
        work = df.tail(lookback).copy().reset_index(drop=True)
        highs = work["high"].values.astype(float)
        lows = work["low"].values.astype(float)
        closes = work["close"].values.astype(float)
        volumes = work["volume"].values.astype(float)

        # -- Step 1: Detect swing highs / lows -----------------------------
        swing_highs, swing_lows = self._detect_swings(highs, lows, order)

        if not swing_highs and not swing_lows:
            result.notes.append("No swing points detected.")
            return result

        # -- Step 2: Map liquidity pools ------------------------------------
        support_pools = self._cluster_levels(
            [s["price"] for s in swing_lows], CLUSTER_PCT
        )
        resistance_pools = self._cluster_levels(
            [s["price"] for s in swing_highs], CLUSTER_PCT
        )

        result.support_pools = [
            {"level": round(p["level"], 2), "strength": p["strength"]}
            for p in support_pools
        ]
        result.resistance_pools = [
            {"level": round(p["level"], 2), "strength": p["strength"]}
            for p in resistance_pools
        ]

        logger.info(
            "Liquidity map: %d support pools, %d resistance pools",
            len(support_pools),
            len(resistance_pools),
        )

        # -- ATR ------------------------------------------------------------
        atr = self._compute_atr(highs, lows)

        # -- Step 3: Detect active sweep ------------------------------------
        sweep = self._detect_sweep(
            work, support_pools, volumes, atr
        )

        if sweep is not None:
            result.has_sweep = True
            result.sweep_score = round(sweep["score"], 1)
            result.sweep_level = round(sweep["pool_level"], 2)
            result.sweep_bar_index = sweep["bar_index"]
            result.volume_confirmed = sweep["volume_confirmed"]
            result.wick_rejection = sweep["wick_rejection"]
            result.components = sweep["components"]

            logger.info(
                "Sweep detected at %.2f | score=%.1f | bar=%d",
                sweep["pool_level"],
                sweep["score"],
                sweep["bar_index"],
            )

        # -- Step 6: Base compression (run regardless — adds context) -------
        compressed = self._check_base_compression(closes)
        result.base_compressed = compressed

        if compressed and result.has_sweep:
            result.sweep_score = min(100.0, round(result.sweep_score + 15, 1))
            result.notes.append(
                "Base compression bonus (+15) — tight range before sweep."
            )
            logger.info("Base compression detected — score boosted to %.1f", result.sweep_score)

        # -- Stochastic confirmation ------------------------------------------
        stoch_k, stoch_d = self._compute_stochastic(highs, lows, closes)
        result.stoch_k = round(stoch_k, 1)
        result.stoch_d = round(stoch_d, 1)

        # Oversold: %K or %D below 20
        result.stoch_oversold = stoch_k < 25 or stoch_d < 25

        # Bullish cross: %K crosses above %D from oversold
        if len(closes) >= 16:
            prev_k, prev_d = self._compute_stochastic(
                highs[:-1], lows[:-1], closes[:-1]
            )
            result.stoch_cross = (prev_k <= prev_d) and (stoch_k > stoch_d)
        else:
            result.stoch_cross = False

        # Boost sweep score if stochastic confirms
        if result.has_sweep:
            if result.stoch_oversold:
                result.sweep_score = min(100.0, round(result.sweep_score + 12, 1))
                result.notes.append(
                    f"Stochastic oversold bonus (+12) — %K={stoch_k:.0f} %D={stoch_d:.0f}"
                )
            if result.stoch_cross:
                result.sweep_score = min(100.0, round(result.sweep_score + 10, 1))
                result.notes.append(
                    "Stochastic bullish cross bonus (+10) — %K crossed above %D"
                )

        # -- Gate: only build trade plan if score >= 50 ---------------------
        if result.has_sweep and result.sweep_score >= 50:
            # Step 4: Exit targets (resistance pools above current price)
            targets = sorted(
                [p for p in resistance_pools if p["level"] > current_price],
                key=lambda p: p["level"],
            )
            if targets:
                result.target = round(targets[0]["level"], 2)
            if len(targets) >= 2:
                result.extended_target = round(targets[1]["level"], 2)
            if len(targets) >= 3:
                result.runner_target = round(targets[2]["level"], 2)

            # Step 5: Entry / stop / risk-reward
            pool_level = sweep["pool_level"]
            sweep_idx = sweep["bar_index"]
            result.entry = round(pool_level + 0.02 * atr, 2)
            result.stop = round(float(lows[sweep_idx]) - 0.15 * atr, 2)

            risk = result.entry - result.stop
            if risk > 0 and result.target > 0:
                result.risk_reward = round(
                    (result.target - result.entry) / risk, 2
                )

            result.notes.append(
                f"Entry plan: entry={result.entry}, stop={result.stop}, "
                f"T1={result.target}, T2={result.extended_target}, "
                f"T3={result.runner_target}, R:R={result.risk_reward}"
            )
            logger.info(
                "Trade plan: entry=%.2f stop=%.2f T1=%.2f R:R=%.2f",
                result.entry,
                result.stop,
                result.target,
                result.risk_reward,
            )
        elif result.has_sweep:
            result.notes.append(
                f"Sweep detected but score ({result.sweep_score}) below 50 threshold."
            )

        return result

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _detect_swings(
        highs: np.ndarray,
        lows: np.ndarray,
        order: int,
    ) -> tuple[list[dict], list[dict]]:
        """Return lists of swing-high and swing-low dicts with index + price."""
        swing_highs: list[dict] = []
        swing_lows: list[dict] = []
        n = len(highs)

        for i in range(order, n - order):
            window_high = highs[i - order : i + order + 1]
            if highs[i] == np.max(window_high):
                swing_highs.append({"index": i, "price": float(highs[i])})

            window_low = lows[i - order : i + order + 1]
            if lows[i] == np.min(window_low):
                swing_lows.append({"index": i, "price": float(lows[i])})

        return swing_highs, swing_lows

    @staticmethod
    def _cluster_levels(
        prices: list[float],
        tolerance: float,
    ) -> list[dict]:
        """Cluster nearby price levels into liquidity pools.

        Parameters
        ----------
        prices : list[float]
            Raw swing-point prices.
        tolerance : float
            Max relative distance to merge (e.g. 0.005 = 0.5%).

        Returns
        -------
        list[dict]
            Each dict has ``level`` (average price) and ``strength`` (count).
        """
        if not prices:
            return []

        sorted_prices = sorted(prices)
        clusters: list[list[float]] = [[sorted_prices[0]]]

        for price in sorted_prices[1:]:
            cluster_avg = np.mean(clusters[-1])
            if abs(price - cluster_avg) / cluster_avg <= tolerance:
                clusters[-1].append(price)
            else:
                clusters.append([price])

        return [
            {"level": float(np.mean(c)), "strength": len(c)}
            for c in clusters
        ]

    @staticmethod
    def _compute_atr(highs: np.ndarray, lows: np.ndarray) -> float:
        """Simple ATR: mean of (high - low) over the last ATR_PERIOD bars."""
        ranges = highs[-ATR_PERIOD:] - lows[-ATR_PERIOD:]
        return float(np.mean(ranges))

    @staticmethod
    def _detect_sweep(
        df: pd.DataFrame,
        support_pools: list[dict],
        volumes: np.ndarray,
        atr: float,
    ) -> dict | None:
        """Scan last SWEEP_WINDOW bars for a support-pool sweep + reclaim.

        Returns the highest-scoring sweep dict or None.
        """
        if not support_pools:
            return None

        n = len(df)
        if n < SWEEP_WINDOW + 20:
            return None

        avg_vol = float(np.mean(volumes[-(20 + SWEEP_WINDOW) : -SWEEP_WINDOW]))
        if avg_vol == 0:
            avg_vol = 1.0  # prevent division by zero

        best: dict | None = None
        best_score = 0.0

        for bar_offset in range(1, SWEEP_WINDOW + 1):
            idx = n - bar_offset
            if idx < 1:
                continue

            bar_low = float(df["low"].iloc[idx])
            bar_close = float(df["close"].iloc[idx])
            bar_high = float(df["high"].iloc[idx])
            bar_open = float(df["open"].iloc[idx])
            bar_vol = float(volumes[idx])
            candle_range = bar_high - bar_low

            if candle_range == 0:
                continue

            lower_wick = min(bar_open, bar_close) - bar_low
            wick_ratio = lower_wick / candle_range

            for pool in support_pools:
                level = pool["level"]
                strength = pool["strength"]

                # Condition 1: low dipped below pool level (swept stops)
                if bar_low >= level:
                    continue

                # Condition 2: close reclaimed above pool level
                if bar_close <= level:
                    # Check if next bar reclaimed
                    reclaim_bars = 0
                    for fwd in range(1, 4):
                        if idx + fwd < n and float(df["close"].iloc[idx + fwd]) > level:
                            reclaim_bars = fwd
                            break
                    if reclaim_bars == 0:
                        continue
                else:
                    reclaim_bars = 0  # same bar reclaim

                # Condition 3: wick rejection (> 50% of range)
                wick_ok = wick_ratio > 0.50

                # Condition 4: volume elevated (> 1.5x average)
                vol_ratio = bar_vol / avg_vol
                vol_ok = vol_ratio > 1.5

                # ----- Score components -----
                # Wick ratio (30%) — floor 0.3, cap 0.7
                wick_raw = np.clip((wick_ratio - 0.3) / (0.7 - 0.3), 0.0, 1.0) * 100
                wick_component = round(float(wick_raw), 1)

                # Reclaim speed (30%)
                if reclaim_bars == 0:
                    reclaim_component = 100.0
                elif reclaim_bars == 1:
                    reclaim_component = 100.0
                elif reclaim_bars == 2:
                    reclaim_component = 70.0
                else:
                    reclaim_component = 40.0

                # Volume spike (20%) — floor 1.5x, cap 4.0x
                vol_raw = np.clip((vol_ratio - 1.5) / (4.0 - 1.5), 0.0, 1.0) * 100
                vol_component = round(float(vol_raw), 1)

                # Pool strength (20%)
                if strength >= 3:
                    pool_component = 100.0
                elif strength == 2:
                    pool_component = 60.0
                else:
                    pool_component = 30.0

                score = (
                    wick_component * 0.30
                    + reclaim_component * 0.30
                    + vol_component * 0.20
                    + pool_component * 0.20
                )
                score = round(score, 1)

                if score > best_score:
                    best_score = score
                    best = {
                        "score": score,
                        "pool_level": level,
                        "pool_strength": strength,
                        "bar_index": idx,
                        "wick_rejection": wick_ok,
                        "volume_confirmed": vol_ok,
                        "components": {
                            "wick_ratio": wick_component,
                            "reclaim_speed": reclaim_component,
                            "volume_spike": vol_component,
                            "pool_strength": pool_component,
                        },
                    }

        return best

    @staticmethod
    def _compute_stochastic(
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        k_period: int = 14,
        d_period: int = 3,
    ) -> tuple[float, float]:
        """Compute Slow Stochastic %K and %D.

        Slow STO: %K = SMA(raw_k, 3), %D = SMA(%K, 3)
        Returns the latest (%K, %D) values.
        """
        n = len(closes)
        if n < k_period + d_period:
            return 50.0, 50.0

        # Raw %K for each bar
        raw_k = np.full(n, 50.0)
        for i in range(k_period - 1, n):
            window_high = float(np.max(highs[i - k_period + 1 : i + 1]))
            window_low = float(np.min(lows[i - k_period + 1 : i + 1]))
            denom = window_high - window_low
            if denom > 0:
                raw_k[i] = ((closes[i] - window_low) / denom) * 100.0

        # Slow %K = SMA(raw_k, d_period)
        slow_k = np.convolve(raw_k, np.ones(d_period) / d_period, mode="full")
        slow_k = slow_k[d_period - 1 : n + d_period - 1][:n]

        # %D = SMA(slow_k, d_period)
        slow_d = np.convolve(slow_k, np.ones(d_period) / d_period, mode="full")
        slow_d = slow_d[d_period - 1 : n + d_period - 1][:n]

        return float(slow_k[-1]), float(slow_d[-1])

    @staticmethod
    def _check_base_compression(closes: np.ndarray) -> bool:
        """Check if last 20 bars form a compressed (tight) base.

        A compressed base has a close-to-close range < 15% of average close.
        """
        if len(closes) < 20:
            return False

        recent = closes[-20:]
        avg_close = float(np.mean(recent))
        if avg_close == 0:
            return False

        price_range = float(np.max(recent) - np.min(recent))
        compression = price_range / avg_close
        return compression < 0.15
