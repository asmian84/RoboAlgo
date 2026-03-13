"""
RoboAlgo — Rocket Scanner
Five-stage scanning pipeline that identifies breakout candidates.

Pipeline:
  Stage 1 — Universe Filter     (price > $3, avg_volume > 500 k)  → ~1 200 symbols
  Stage 2 — Volatility Squeeze  (BB + ATR + KC compression)        → ~250 symbols
  Stage 3 — Pattern Detection   (triangles, wedges, flags, …)      → ~120 symbols
  Stage 4 — Options/GEX Filter  (dealer short gamma or flip zone)  → ~60 symbols
  Stage 5 — Rocket Score        (weighted multi-factor rank)        → top 20

Performance design:
  • Stages 1-2 are fully vectorised (NumPy/pandas).
  • Stages 3-4 use a ProcessPoolExecutor (CPU-bound analysis).
  • Stage 5 is O(n log n) sort.
  • Target: ≤ 5 s end-to-end for 1 200 symbols on 400 bars each.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from signal_engine.volatility_squeeze import VolatilitySqueezeEngine, batch_squeeze, SqueezeResult
from pattern_engine.pivot_engine import PivotEngine, PatternResult

logger = logging.getLogger(__name__)

# ── Worker helpers (must be module-level for pickling) ────────────────────────

_squeeze_engine = None
_pivot_engine   = None


def _init_worker() -> None:
    global _squeeze_engine, _pivot_engine
    _squeeze_engine = VolatilitySqueezeEngine()
    _pivot_engine   = PivotEngine()


def _process_symbol(args: tuple) -> dict | None:
    """
    Worker function: runs pattern detection on a single symbol.
    Called inside a ProcessPoolExecutor.
    """
    global _squeeze_engine, _pivot_engine
    sym, high_arr, low_arr, close_arr, vol_arr = args
    try:
        if _pivot_engine is None:
            _pivot_engine = PivotEngine()

        patterns = _pivot_engine.detect(
            high   = high_arr,
            low    = low_arr,
            close  = close_arr,
            volume = vol_arr,
        )

        if not patterns:
            return None

        best = patterns[0]
        return {
            "symbol":          sym,
            "pattern_name":    best.pattern_name,
            "direction":       best.direction,
            "pattern_quality": best.confidence,
            "breakout_level":  best.breakout_level,
            "invalidation":    best.invalidation_level,
            "target":          best.target,
            "bars_forming":    best.bars_forming,
            "all_patterns":    [PivotEngine.result_to_dict(p) for p in patterns[:3]],
        }
    except Exception as exc:
        logger.debug("_process_symbol(%s): %s", sym, exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  RocketCandidate — result object per symbol
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RocketCandidate:
    symbol:              str
    rocket_score:        float          # 0-100
    pattern_name:        str
    direction:           str
    breakout_level:      float | None
    invalidation_level:  float | None
    target:              float | None
    gamma_levels:        dict           # call_wall, put_wall, zero_gamma
    gamma_regime:        str
    squeeze_active:      bool
    squeeze_intensity:   float
    # Component scores
    pattern_quality:     float
    gamma_score:         float
    vol_squeeze_score:   float
    volume_score:        float
    trend_score:         float
    # Metadata
    current_price:       float
    current_atr:         float
    stage_passed:        int           # 1-5, highest stage reached
    scan_time_ms:        float = 0.0
    notes:               list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
#  Stage computations
# ─────────────────────────────────────────────────────────────────────────────

def _stage1_filter(
    price_data: dict[str, pd.DataFrame],
    min_price:  float = 3.0,
    min_volume: float = 500_000,
) -> list[str]:
    """Keep symbols with price > min_price AND last 20-bar avg volume > min_volume."""
    passed: list[str] = []
    for sym, df in price_data.items():
        if df.empty or "close" not in df.columns:
            continue
        price = float(df["close"].iloc[-1])
        vol   = float(df["volume"].iloc[-20:].mean()) if "volume" in df.columns else 0.0
        if price >= min_price and vol >= min_volume:
            passed.append(sym)
    return passed


def _stage2_squeeze(
    symbols:    list[str],
    price_data: dict[str, pd.DataFrame],
    min_score:  float = 25.0,
) -> tuple[list[str], dict[str, SqueezeResult]]:
    """Vectorised squeeze scoring; keep symbols with vol_squeeze_score ≥ min_score."""
    engine   = VolatilitySqueezeEngine()
    sub_data = {s: price_data[s] for s in symbols if s in price_data}
    sq_map   = batch_squeeze(sub_data, engine)
    passed   = [s for s in symbols if sq_map.get(s, None) and sq_map[s].vol_squeeze_score >= min_score]
    return passed, sq_map


def _stage3_patterns(
    symbols:    list[str],
    price_data: dict[str, pd.DataFrame],
    workers:    int = 4,
) -> dict[str, dict]:
    """Parallel pattern detection using ProcessPoolExecutor."""
    args_list = []
    for sym in symbols:
        df = price_data.get(sym)
        if df is None or df.empty:
            continue
        h = df["high"].to_numpy(dtype=float)
        l = df["low"].to_numpy(dtype=float)
        c = df["close"].to_numpy(dtype=float)
        v = df["volume"].to_numpy(dtype=float) if "volume" in df.columns else np.ones(len(c))
        args_list.append((sym, h, l, c, v))

    results: dict[str, dict] = {}
    try:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
        ) as pool:
            futs = {pool.submit(_process_symbol, args): args[0] for args in args_list}
            for fut in as_completed(futs):
                r = fut.result()
                if r is not None:
                    results[r["symbol"]] = r
    except Exception:
        # Fall back to single-threaded if pickling fails (e.g., during uvicorn reload)
        _init_worker()
        for args in args_list:
            r = _process_symbol(args)
            if r is not None:
                results[r["symbol"]] = r

    return results


def _stage4_options(
    symbols: list[str],
    workers: int = 4,
) -> dict[str, dict]:
    """
    Fetch options chains + compute GEX + dealer positioning.
    Symbols without options data are kept but scored low.
    """
    from data_engine.yahoo_options_fetcher import fetch_bulk_chains, chain_to_dict
    from options_engine.gamma_exposure import GammaExposureEngine
    from options_engine.dealer_positioning import DealerPositioningEngine

    gex_engine  = GammaExposureEngine()
    dp_engine   = DealerPositioningEngine()

    chains = fetch_bulk_chains(symbols, max_workers=workers, max_expirations=3)
    results: dict[str, dict] = {}

    for sym in symbols:
        chain = chains.get(sym)
        if chain is None:
            results[sym] = {
                "gamma_regime":       "NEUTRAL",
                "gamma_score":        50.0,
                "gamma_levels":       {},
                "positioning_score":  50.0,
                "dealer_regime":      "UNKNOWN",
                "squeeze_risk":       "LOW",
                "flow_bias":          "NEUTRAL",
            }
            continue

        profile = gex_engine.compute(chain)
        dp      = dp_engine.compute(profile)

        results[sym] = {
            "gamma_regime":      profile.gamma_regime,
            "gamma_score":       profile.gex_score,
            "gamma_levels":      dp.gamma_levels,
            "positioning_score": dp.positioning_score,
            "dealer_regime":     dp.dealer_regime,
            "squeeze_risk":      dp.squeeze_risk,
            "flow_bias":         dp.flow_bias,
            "call_wall":         profile.call_wall,
            "put_wall":          profile.put_wall,
            "zero_gamma":        profile.zero_gamma,
            "gex_by_strike":     profile.gex_by_strike[:10],
        }

    return results


def _volume_score(df: pd.DataFrame) -> float:
    """Volume accumulation score 0-100: recent vol vs 50-bar avg."""
    if df.empty or "volume" not in df.columns:
        return 50.0
    vol    = df["volume"].to_numpy(dtype=float)
    avg50  = vol[-50:].mean() if len(vol) >= 50 else vol.mean()
    avg5   = vol[-5:].mean()
    ratio  = avg5 / max(avg50, 1.0)
    return float(min(ratio / 2.0, 1.0) * 100)


def _trend_score(df: pd.DataFrame) -> float:
    """Trend alignment: price above EMA50 above EMA200 → 100."""
    if df.empty or "close" not in df.columns:
        return 50.0
    c = df["close"].to_numpy(dtype=float)
    if len(c) < 50:
        return 50.0
    alpha50  = 2 / 51
    alpha200 = 2 / 201
    ema50  = c[0]
    ema200 = c[0]
    for price in c[1:]:
        ema50  = alpha50  * price + (1 - alpha50)  * ema50
        ema200 = alpha200 * price + (1 - alpha200) * ema200
    last = float(c[-1])
    score = 0.0
    if last > ema50:
        score += 50
    if ema50 > ema200:
        score += 30
    if last > ema200:
        score += 20
    return score


# ─────────────────────────────────────────────────────────────────────────────
#  Score weights
# ─────────────────────────────────────────────────────────────────────────────

ROCKET_WEIGHTS = {
    "pattern_quality":  0.30,
    "gamma_score":      0.25,
    "vol_squeeze":      0.20,
    "volume_accum":     0.15,
    "trend_alignment":  0.10,
}


def _compute_rocket_score(
    pattern_quality:  float,
    gamma_score:      float,
    vol_squeeze:      float,
    volume_accum:     float,
    trend_alignment:  float,
) -> float:
    w = ROCKET_WEIGHTS
    score = (
        w["pattern_quality"] * pattern_quality
        + w["gamma_score"]   * gamma_score
        + w["vol_squeeze"]   * vol_squeeze
        + w["volume_accum"]  * volume_accum
        + w["trend_alignment"] * trend_alignment
    )
    return round(min(score, 100), 1)


# ─────────────────────────────────────────────────────────────────────────────
#  Main scanner class
# ─────────────────────────────────────────────────────────────────────────────

class RocketScanner:
    """
    End-to-end 5-stage rocket-mover discovery engine.

    Usage:
        scanner = RocketScanner()
        # Provide pre-loaded OHLCV dict (from DB or AV fetcher):
        candidates = scanner.scan(price_data, top_n=20)
    """

    def __init__(
        self,
        min_price:          float = 3.0,
        min_volume:         float = 500_000,
        squeeze_min_score:  float = 25.0,
        top_n:              int   = 20,
        pattern_workers:    int   = 4,
        options_workers:    int   = 4,
        skip_options:       bool  = False,   # speed mode: skip options stage
    ):
        self.min_price         = min_price
        self.min_volume        = min_volume
        self.squeeze_min_score = squeeze_min_score
        self.top_n             = top_n
        self.pattern_workers   = pattern_workers
        self.options_workers   = options_workers
        self.skip_options      = skip_options

    def scan(
        self,
        price_data: dict[str, pd.DataFrame],
        extra_universe: list[str] | None = None,
    ) -> list[RocketCandidate]:
        """
        Run the full 5-stage scan.

        Args:
            price_data:       {symbol: OHLCV DataFrame}
            extra_universe:   optional additional symbols to include
        Returns:
            list of RocketCandidate sorted by rocket_score descending (top_n max)
        """
        t0 = time.perf_counter()
        all_symbols = list(price_data.keys())
        if extra_universe:
            all_symbols = list(set(all_symbols) | set(extra_universe))

        # ── Stage 1 ───────────────────────────────────────────────────────
        s1 = _stage1_filter(price_data, self.min_price, self.min_volume)
        logger.info("[S1] %d → %d symbols (price/volume filter)", len(all_symbols), len(s1))

        if not s1:
            return []

        # ── Stage 2 ───────────────────────────────────────────────────────
        s2, sq_map = _stage2_squeeze(s1, price_data, self.squeeze_min_score)
        logger.info("[S2] %d → %d symbols (squeeze filter)", len(s1), len(s2))

        if not s2:
            return []

        # ── Stage 3 ───────────────────────────────────────────────────────
        s3_patterns = _stage3_patterns(s2, price_data, self.pattern_workers)
        s3 = list(s3_patterns.keys())
        logger.info("[S3] %d → %d symbols (pattern detection)", len(s2), len(s3))

        if not s3:
            return []

        # ── Stage 4 ───────────────────────────────────────────────────────
        if self.skip_options:
            s4_options = {sym: {
                "gamma_regime": "NEUTRAL", "gamma_score": 50,
                "gamma_levels": {}, "positioning_score": 50,
                "dealer_regime": "UNKNOWN", "squeeze_risk": "LOW",
                "flow_bias": "NEUTRAL",
            } for sym in s3}
        else:
            s4_options = _stage4_options(s3, self.options_workers)

        logger.info("[S4] options/GEX data fetched for %d symbols", len(s4_options))

        # ── Stage 5: Score & Rank ─────────────────────────────────────────
        candidates: list[RocketCandidate] = []

        for sym in s3:
            df       = price_data.get(sym, pd.DataFrame())
            sq       = sq_map.get(sym)
            pat_data = s3_patterns.get(sym, {})
            opt_data = s4_options.get(sym, {})

            if df.empty:
                continue

            current_price = float(df["close"].iloc[-1]) if not df.empty else 0.0
            current_atr   = float(sq.current_atr) if sq else 0.0

            # Component scores
            pattern_quality  = float(pat_data.get("pattern_quality", 50))
            gamma_score      = float(opt_data.get("gamma_score", 50))
            vol_squeeze      = float(sq.vol_squeeze_score) if sq else 0.0
            volume_accum     = _volume_score(df)
            trend_align      = _trend_score(df)

            rocket_score = _compute_rocket_score(
                pattern_quality, gamma_score, vol_squeeze,
                volume_accum,    trend_align,
            )

            # Bonus modifiers
            notes: list[str] = []
            if sq and sq.kc_squeeze:
                rocket_score = min(rocket_score + 5, 100)
                notes.append("KC Squeeze active")
            if opt_data.get("dealer_regime") == "NET_SHORT_GAMMA":
                rocket_score = min(rocket_score + 5, 100)
                notes.append("Dealers short gamma — amplifying regime")
            if opt_data.get("squeeze_risk") == "HIGH":
                rocket_score = min(rocket_score + 3, 100)
                notes.append("Gamma squeeze risk HIGH")

            gamma_levels = {
                "call_wall":  opt_data.get("call_wall", 0),
                "put_wall":   opt_data.get("put_wall", 0),
                "zero_gamma": opt_data.get("zero_gamma", 0),
            }

            candidates.append(RocketCandidate(
                symbol             = sym,
                rocket_score       = rocket_score,
                pattern_name       = pat_data.get("pattern_name", "Unknown"),
                direction          = pat_data.get("direction", "neutral"),
                breakout_level     = pat_data.get("breakout_level"),
                invalidation_level = pat_data.get("invalidation"),
                target             = pat_data.get("target"),
                gamma_levels       = gamma_levels,
                gamma_regime       = opt_data.get("gamma_regime", "NEUTRAL"),
                squeeze_active     = bool(sq.squeeze_active) if sq else False,
                squeeze_intensity  = float(sq.squeeze_intensity) if sq else 0.0,
                pattern_quality    = pattern_quality,
                gamma_score        = gamma_score,
                vol_squeeze_score  = vol_squeeze,
                volume_score       = volume_accum,
                trend_score        = trend_align,
                current_price      = current_price,
                current_atr        = current_atr,
                stage_passed       = 5,
                notes              = notes,
            ))

        # Sort by rocket_score
        candidates.sort(key=lambda c: c.rocket_score, reverse=True)
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            "[S5] Top %d candidates | scan time: %.1f ms", self.top_n, elapsed
        )
        return candidates[: self.top_n]

    def scan_from_db(
        self,
        lookback_days: int = 400,
        symbols:       list[str] | None = None,
    ) -> list[RocketCandidate]:
        """
        Convenience wrapper: load price data from PostgreSQL then scan.
        """
        from data_engine.alphavantage_fetcher import load_ohlcv_from_db
        from config.settings import get_all_instruments

        syms = symbols or get_all_instruments()
        logger.info("Loading price data for %d symbols from DB …", len(syms))
        price_data = load_ohlcv_from_db(syms, lookback_days=lookback_days)
        return self.scan(price_data)

    @staticmethod
    def candidate_to_dict(c: RocketCandidate) -> dict:
        return {
            "symbol":             c.symbol,
            "rocket_score":       c.rocket_score,
            "pattern_name":       c.pattern_name,
            "direction":          c.direction,
            "breakout_level":     c.breakout_level,
            "invalidation_level": c.invalidation_level,
            "target":             c.target,
            "gamma_levels":       c.gamma_levels,
            "gamma_regime":       c.gamma_regime,
            "squeeze_active":     c.squeeze_active,
            "squeeze_intensity":  c.squeeze_intensity,
            "current_price":      c.current_price,
            "current_atr":        c.current_atr,
            "component_scores": {
                "pattern_quality": c.pattern_quality,
                "gamma_score":     c.gamma_score,
                "vol_squeeze":     c.vol_squeeze_score,
                "volume_accum":    c.volume_score,
                "trend_align":     c.trend_score,
            },
            "notes": c.notes,
        }
