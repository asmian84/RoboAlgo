"""
RoboAlgo — Volatility Squeeze Detector
Identifies multi-factor compression setups ready for explosive breakout.

Squeeze model (three independent checks, any 2/3 fires a squeeze):
  1. Bollinger Band Width < N-bar percentile threshold
  2. ATR < N-bar percentile threshold
  3. Keltner Channel squeeze: BB inside KC (John Carter / TTM Squeeze)

Outputs per symbol:
  • squeeze_active      — bool
  • squeeze_intensity   — 0-100 (higher = tighter compression)
  • squeeze_bars        — consecutive bars in compression
  • bb_pct_rank         — Bollinger width percentile rank
  • atr_pct_rank        — ATR percentile rank
  • kc_squeeze          — bool (BB inside KC)
  • momentum_histogram  — momentum oscillator for direction bias
  • vol_squeeze_score   — 0-100 for scanner integration
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from indicator_engine.technical import (
    atr          as _atr,
    bollinger    as _bollinger,
    keltner      as _keltner,
    percentile_rank as _percentile_rank,
    momentum_oscillator as _momentum_oscillator,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Squeeze result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SqueezeResult:
    symbol:              str
    squeeze_active:      bool
    squeeze_intensity:   float   # 0-100
    squeeze_bars:        int     # consecutive bars in squeeze
    bb_pct_rank:         float   # 0-1 (lower = tighter BB)
    atr_pct_rank:        float   # 0-1 (lower = lower ATR)
    kc_squeeze:          bool    # BB inside Keltner
    momentum_value:      float   # latest momentum histogram
    momentum_direction:  str     # "UP" | "DOWN" | "FLAT"
    vol_squeeze_score:   float   # 0-100 for scanner
    current_atr:         float
    current_bb_width:    float
    checks_fired:        int     # how many of 3 checks triggered


# ─────────────────────────────────────────────────────────────────────────────
#  Main engine
# ─────────────────────────────────────────────────────────────────────────────

class VolatilitySqueezeEngine:
    """
    Detects multi-factor volatility compression squeezes.

    Usage:
        engine = VolatilitySqueezeEngine()
        result = engine.compute("NVDA", high, low, close)
    """

    def __init__(
        self,
        bb_period:     int   = 20,
        bb_std:        float = 2.0,
        atr_period:    int   = 14,
        kc_period:     int   = 20,
        kc_mult:       float = 1.5,
        lookback:      int   = 252,
        bb_squeeze_pct: float = 0.20,   # BB width in bottom 20% = squeeze
        atr_squeeze_pct: float = 0.25,  # ATR in bottom 25% = squeeze
    ):
        self.bb_period      = bb_period
        self.bb_std         = bb_std
        self.atr_period     = atr_period
        self.kc_period      = kc_period
        self.kc_mult        = kc_mult
        self.lookback       = lookback
        self.bb_squeeze_pct = bb_squeeze_pct
        self.atr_squeeze_pct = atr_squeeze_pct

    def compute(
        self,
        symbol: str,
        high:   np.ndarray,
        low:    np.ndarray,
        close:  np.ndarray,
    ) -> SqueezeResult:
        """Run squeeze detection on OHLC arrays. Returns SqueezeResult."""
        n = len(close)
        if n < 50:
            return self._empty(symbol)

        # ── Indicators ────────────────────────────────────────────────────
        _, bb_upper, bb_lower, bb_bw = _bollinger(close, self.bb_period, self.bb_std)
        atr_arr                      = _atr(high, low, close, self.atr_period)
        kc_upper, kc_lower           = _keltner(high, low, close, self.kc_period, self.kc_mult)
        momentum                     = _momentum_oscillator(close, 12)

        # ── Percentile ranks (rolling lookback) ───────────────────────────
        bb_rank  = _percentile_rank(bb_bw,  self.lookback)
        atr_rank = _percentile_rank(atr_arr, self.lookback)

        # Latest valid values
        def _last(arr: np.ndarray) -> float:
            valid = arr[~np.isnan(arr)]
            return float(valid[-1]) if len(valid) else 0.0

        cur_bb_rank  = _last(bb_rank)
        cur_atr_rank = _last(atr_rank)
        cur_bb_bw    = _last(bb_bw)
        cur_atr      = _last(atr_arr)
        cur_momentum = _last(momentum)

        # ── Three squeeze checks ──────────────────────────────────────────
        check_bb = cur_bb_rank <= self.bb_squeeze_pct
        check_atr = cur_atr_rank <= self.atr_squeeze_pct

        # KC squeeze: BB upper < KC upper AND BB lower > KC lower
        last_bb_u = _last(bb_upper)
        last_bb_l = _last(bb_lower)
        last_kc_u = _last(kc_upper)
        last_kc_l = _last(kc_lower)
        check_kc  = (last_bb_u < last_kc_u) and (last_bb_l > last_kc_l)

        checks_fired = int(check_bb) + int(check_atr) + int(check_kc)
        squeeze_active = checks_fired >= 2

        # ── Consecutive squeeze bars ──────────────────────────────────────
        # Count bars where BB rank was in squeeze zone
        squeeze_bars = 0
        for i in range(len(bb_rank) - 1, -1, -1):
            if np.isnan(bb_rank[i]):
                break
            if bb_rank[i] <= self.bb_squeeze_pct:
                squeeze_bars += 1
            else:
                break

        # ── Intensity: how tight is the squeeze (0-100) ───────────────────
        # Lower bb_rank + lower atr_rank → higher intensity
        bb_intensity  = max(0, 1 - cur_bb_rank  / max(self.bb_squeeze_pct,  0.01)) * 100
        atr_intensity = max(0, 1 - cur_atr_rank / max(self.atr_squeeze_pct, 0.01)) * 100
        kc_intensity  = 100 if check_kc else 0
        squeeze_intensity = 0.40 * bb_intensity + 0.35 * atr_intensity + 0.25 * kc_intensity

        # ── Momentum direction ────────────────────────────────────────────
        if abs(cur_momentum) < 0.001 * float(close[-1]):
            momentum_dir = "FLAT"
        elif cur_momentum > 0:
            momentum_dir = "UP"
        else:
            momentum_dir = "DOWN"

        # ── Final vol_squeeze_score ───────────────────────────────────────
        # Squeeze active + many bars + strong intensity + bullish momentum
        duration_score = min(squeeze_bars / 20, 1) * 100
        momentum_score = 100 if momentum_dir == "UP" else (60 if momentum_dir == "FLAT" else 30)

        if squeeze_active:
            vol_squeeze_score = (
                0.40 * squeeze_intensity
                + 0.35 * duration_score
                + 0.25 * momentum_score
            )
        else:
            # Partial score even if not fully active
            vol_squeeze_score = squeeze_intensity * 0.3 * (checks_fired / 3)

        return SqueezeResult(
            symbol             = symbol,
            squeeze_active     = squeeze_active,
            squeeze_intensity  = round(squeeze_intensity, 1),
            squeeze_bars       = squeeze_bars,
            bb_pct_rank        = round(cur_bb_rank, 3),
            atr_pct_rank       = round(cur_atr_rank, 3),
            kc_squeeze         = check_kc,
            momentum_value     = round(cur_momentum, 4),
            momentum_direction = momentum_dir,
            vol_squeeze_score  = round(min(vol_squeeze_score, 100), 1),
            current_atr        = round(cur_atr, 4),
            current_bb_width   = round(cur_bb_bw, 4),
            checks_fired       = checks_fired,
        )

    @staticmethod
    def _empty(symbol: str) -> SqueezeResult:
        return SqueezeResult(
            symbol=symbol, squeeze_active=False, squeeze_intensity=0,
            squeeze_bars=0, bb_pct_rank=1.0, atr_pct_rank=1.0,
            kc_squeeze=False, momentum_value=0, momentum_direction="FLAT",
            vol_squeeze_score=0, current_atr=0, current_bb_width=0, checks_fired=0,
        )

    @staticmethod
    def result_to_dict(r: SqueezeResult) -> dict:
        return {
            "squeeze_active":    r.squeeze_active,
            "squeeze_intensity": r.squeeze_intensity,
            "squeeze_bars":      r.squeeze_bars,
            "bb_pct_rank":       r.bb_pct_rank,
            "atr_pct_rank":      r.atr_pct_rank,
            "kc_squeeze":        r.kc_squeeze,
            "momentum_value":    r.momentum_value,
            "momentum_direction": r.momentum_direction,
            "vol_squeeze_score": r.vol_squeeze_score,
            "current_atr":       r.current_atr,
            "current_bb_width":  r.current_bb_width,
            "checks_fired":      r.checks_fired,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Batch helper (used by rocket_scanner.py)
# ─────────────────────────────────────────────────────────────────────────────

def batch_squeeze(
    price_data: dict[str, "pd.DataFrame"],
    engine: VolatilitySqueezeEngine | None = None,
) -> dict[str, SqueezeResult]:
    """
    Compute squeeze for many symbols from a {symbol: DataFrame} dict.
    DataFrame must have columns: high, low, close.
    """
    if engine is None:
        engine = VolatilitySqueezeEngine()
    results: dict[str, SqueezeResult] = {}
    for sym, df in price_data.items():
        try:
            h = df["high"].to_numpy(dtype=float)
            l = df["low"].to_numpy(dtype=float)
            c = df["close"].to_numpy(dtype=float)
            results[sym] = engine.compute(sym, h, l, c)
        except Exception as exc:
            logger.debug("batch_squeeze(%s): %s", sym, exc)
    return results
