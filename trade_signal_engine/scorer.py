"""
UnifiedScorer — 18-component weighted scorer with hard gates and SMC integration.

Wires ALL existing engines + new SMC engine into one score per symbol.
Each component outputs a 0-1 value (0.5 = neutral, >0.5 = bullish, <0.5 = bearish).
Final score = weighted sum × 100 + bonuses, clamped to 0-100.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── MTF Stochastic Confluence ────────────────────────────────────────────────
# Computes Slow Stochastic (14,3,3) on any OHLC DataFrame and returns
# the full recent history needed to detect bull crosses.

def _slow_stochastic_series(
    df: pd.DataFrame, k_period: int = 14, d_period: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Return full slow_k and slow_d arrays for a DataFrame with high/low/close."""
    highs  = df["high"].values.astype(float)
    lows   = df["low"].values.astype(float)
    closes = df["close"].values.astype(float)
    n = len(closes)
    if n < k_period + d_period:
        return np.full(n, 50.0), np.full(n, 50.0)

    raw_k = np.full(n, 50.0)
    for i in range(k_period - 1, n):
        wh = float(np.max(highs[i - k_period + 1 : i + 1]))
        wl = float(np.min(lows[i - k_period + 1 : i + 1]))
        if wh - wl > 0:
            raw_k[i] = ((closes[i] - wl) / (wh - wl)) * 100.0

    slow_k = np.convolve(raw_k, np.ones(d_period) / d_period, mode="full")
    slow_k = slow_k[d_period - 1 : n + d_period - 1][:n]
    slow_d = np.convolve(slow_k, np.ones(d_period) / d_period, mode="full")
    slow_d = slow_d[d_period - 1 : n + d_period - 1][:n]
    return slow_k, slow_d


def _detect_stoch_setup(
    df: pd.DataFrame | None,
    lookback: int = 5,
) -> dict:
    """
    Detect the QUBT-style stochastic setup on a single timeframe.

    LONG setup (oversold bull cross):
      - %K and %D BOTH below 20 (deep oversold)
      - %K JUST crossed above %D (fresh cross, no run-up)
      - After cross: %K still below 30 (hasn't run yet)

    SHORT setup (overbought bear cross):
      - %K and %D BOTH above 80 (deep overbought)
      - %K JUST crossed below %D (fresh cross, no fade)
      - After cross: %K still above 70 (hasn't faded yet)

    Returns dict with setup info for both sides.
    """
    empty = {
        "available": False, "k": 50.0, "d": 50.0,
        "long_cross": False, "short_cross": False,
        "long_score": 0.0, "short_score": 0.0,
        "label": "no data",
    }
    if df is None or len(df) < 20:
        return empty

    sk, sd_arr = _slow_stochastic_series(df)
    k_now, d_now = float(sk[-1]), float(sd_arr[-1])

    result = {
        "available": True,
        "k": round(k_now, 1),
        "d": round(d_now, 1),
        "long_cross": False,
        "short_cross": False,
        "long_score": 0.0,
        "short_score": 0.0,
        "label": "",
    }

    # ── Scan last `lookback` bars for crosses ──────────────────────
    best_long_score = 0.0
    best_short_score = 0.0

    for i in range(-lookback, 0):
        if abs(i) >= len(sk) or abs(i - 1) >= len(sk):
            continue
        prev_k, prev_d = float(sk[i - 1]), float(sd_arr[i - 1])
        curr_k, curr_d = float(sk[i]),     float(sd_arr[i])

        # ── LONG: bull cross from oversold ─────────────────────────
        # prev_k <= prev_d AND curr_k > curr_d  (cross happened)
        # BOTH must be below 20 at time of cross  (deep oversold)
        # %K must be at least 5 (not scraping zero — that's freefall noise)
        # %K must separate from %D by at least 2 points (not rounding noise)
        if prev_k <= prev_d and curr_k > curr_d:
            spread = curr_k - curr_d
            if curr_k < 5 or spread < 2.0:
                pass  # noise cross — K too close to zero or spread too small
            elif curr_k < 20 and curr_d < 20:
                # PERFECT: both deep oversold, fresh cross with real separation
                best_long_score = max(best_long_score, 1.0)
            elif curr_k < 25 and curr_d < 25:
                # STRONG: both in oversold zone
                best_long_score = max(best_long_score, 0.85)
            elif curr_k < 30:
                # OK: cross near oversold
                best_long_score = max(best_long_score, 0.6)

        # ── SHORT: bear cross from overbought ──────────────────────
        # prev_k >= prev_d AND curr_k < curr_d  (cross happened)
        # BOTH must be above 80 at time of cross  (deep overbought)
        # %K must be below 95 (not pinned at 100 — that's melt-up noise)
        # %K must separate from %D by at least 2 points
        if prev_k >= prev_d and curr_k < curr_d:
            spread = curr_d - curr_k
            if curr_k > 95 or spread < 2.0:
                pass  # noise cross — K pinned at ceiling or spread too small
            elif curr_k > 80 and curr_d > 80:
                # PERFECT: both deep overbought, fresh bear cross
                best_short_score = max(best_short_score, 1.0)
            elif curr_k > 75 and curr_d > 75:
                # STRONG: both in overbought zone
                best_short_score = max(best_short_score, 0.85)
            elif curr_k > 70:
                # OK: cross near overbought
                best_short_score = max(best_short_score, 0.6)

    # ── ANTI-RUN-UP CHECK ──────────────────────────────────────────
    # Even if a cross happened recently, if %K has already run away from
    # the extreme zone, the setup is GONE — you missed it.
    # Thresholds are tight: we only want the absolute freshest crosses.
    if best_long_score > 0:
        if k_now > 30:
            # %K ran above 30 — the cross already played out
            best_long_score = 0.0
            result["label"] = f"Long cross happened but %K ran to {k_now:.0f} — missed"
        elif k_now > 22:
            # Slight run-up, reduce confidence significantly
            best_long_score *= 0.5

    if best_short_score > 0:
        if k_now < 70:
            # %K dropped below 70 — the bear cross already played out
            best_short_score = 0.0
            result["label"] = f"Short cross happened but %K dropped to {k_now:.0f} — missed"
        elif k_now < 78:
            best_short_score *= 0.5

    result["long_cross"] = best_long_score > 0
    result["short_cross"] = best_short_score > 0
    result["long_score"] = round(best_long_score, 2)
    result["short_score"] = round(best_short_score, 2)

    if best_long_score >= 0.85:
        result["label"] = f"OVERSOLD CROSS %K={k_now:.0f} %D={d_now:.0f}"
    elif best_long_score > 0:
        result["label"] = f"Bull cross near oversold %K={k_now:.0f}"
    elif best_short_score >= 0.85:
        result["label"] = f"OVERBOUGHT CROSS %K={k_now:.0f} %D={d_now:.0f}"
    elif best_short_score > 0:
        result["label"] = f"Bear cross near overbought %K={k_now:.0f}"
    elif k_now < 20 and d_now < 20:
        result["label"] = f"Oversold %K={k_now:.0f} %D={d_now:.0f} — cross forming"
    elif k_now > 80 and d_now > 80:
        result["label"] = f"Overbought %K={k_now:.0f} %D={d_now:.0f} — bear cross forming"
    else:
        result["label"] = f"No setup — %K={k_now:.0f} %D={d_now:.0f}"

    return result


def _detect_base_compression(df: pd.DataFrame, lookback: int = 20) -> dict:
    """
    Detect if price is forming a compressed base (tight range before breakout).
    This is the pink circle on price — base at support before the stochastic cross fires.
    """
    if df is None or len(df) < lookback:
        return {"compressed": False, "range_pct": 0.0, "at_support": False}

    closes = df["close"].values.astype(float)
    highs  = df["high"].values.astype(float)
    lows   = df["low"].values.astype(float)
    price  = float(closes[-1])

    # Range compression: recent 10-bar range vs 20-bar range
    recent_high = float(np.max(highs[-10:]))
    recent_low  = float(np.min(lows[-10:]))
    wider_high  = float(np.max(highs[-lookback:]))
    wider_low   = float(np.min(lows[-lookback:]))

    recent_range = (recent_high - recent_low) / price * 100 if price > 0 else 100
    wider_range  = (wider_high - wider_low) / price * 100 if price > 0 else 100
    range_ratio  = recent_range / wider_range if wider_range > 0 else 1.0

    # Base at support: price near the low of the range
    price_position = (price - wider_low) / (wider_high - wider_low) if (wider_high - wider_low) > 0 else 0.5
    at_support = price_position < 0.35  # in bottom 35% of range
    at_resistance = price_position > 0.65  # in top 35% of range

    # Compressed = recent range is <60% of wider range
    compressed = range_ratio < 0.60

    return {
        "compressed": compressed,
        "range_pct": round(recent_range, 2),
        "range_ratio": round(range_ratio, 2),
        "at_support": at_support,
        "at_resistance": at_resistance,
        "price_position": round(price_position, 2),
    }


def detect_golden_death_cross(df: pd.DataFrame, lookback: int = 5) -> dict:
    """
    Detect Golden Cross (50 SMA crosses above 200 SMA) and
    Death Cross (50 SMA crosses below 200 SMA) on daily bars.

    lookback: how many bars back to check for a recent cross.

    Returns:
      {
        "golden_cross": bool,   # 50 SMA just crossed above 200 SMA
        "death_cross": bool,    # 50 SMA just crossed below 200 SMA
        "sma50_above_200": bool, # current state: 50 > 200
        "sma50": float,
        "sma200": float,
        "cross_age": int,       # bars since last cross (0 = today)
      }
    """
    empty = {"golden_cross": False, "death_cross": False, "sma50_above_200": False,
             "sma50": 0.0, "sma200": 0.0, "cross_age": -1}

    if df is None or len(df) < 200:
        return empty

    closes = df["close"].values.astype(float)
    sma50 = np.convolve(closes, np.ones(50) / 50, mode="valid")
    sma200 = np.convolve(closes, np.ones(200) / 200, mode="valid")

    # Align: sma50 has len(closes)-49 elements, sma200 has len(closes)-199
    # We need the overlap region
    offset = 150  # sma50 starts 150 elements before sma200 ends
    sma50_aligned = sma50[offset:]  # trim to same length as sma200

    if len(sma50_aligned) < 2 or len(sma200) < 2:
        return empty

    current_50 = float(sma50_aligned[-1])
    current_200 = float(sma200[-1])
    above_now = current_50 > current_200

    result = {
        "golden_cross": False,
        "death_cross": False,
        "sma50_above_200": above_now,
        "sma50": round(current_50, 2),
        "sma200": round(current_200, 2),
        "cross_age": -1,
    }

    # Scan last `lookback` bars for a cross
    scan = min(lookback, len(sma50_aligned) - 1, len(sma200) - 1)
    for i in range(1, scan + 1):
        prev_above = sma50_aligned[-i - 1] > sma200[-i - 1]
        curr_above = sma50_aligned[-i] > sma200[-i]

        if not prev_above and curr_above:
            # Golden cross — 50 crossed above 200
            result["golden_cross"] = True
            result["cross_age"] = i - 1
            break
        elif prev_above and not curr_above:
            # Death cross — 50 crossed below 200
            result["death_cross"] = True
            result["cross_age"] = i - 1
            break

    return result


def _tf_direction(r: dict) -> str:
    """Get the directional bias of a single TF's stochastic state.

    Returns 'LONG', 'SHORT', 'NEUTRAL', or 'NONE' (no data).

    LONG  = %K > %D (bullish — %K leading %D upward)
    SHORT = %K < %D (bearish — %K leading %D downward)

    A bear cross (%K < %D) is bearish at ANY level — not just overbought.
    A bull cross (%K > %D) is bullish at ANY level — not just oversold.
    The *quality* of the cross depends on level, but *direction* is clear.
    """
    if not r.get("available", False):
        return "NONE"
    k, d = r.get("k", 50), r.get("d", 50)
    if k > d:
        return "LONG"
    if k < d:
        return "SHORT"
    return "NEUTRAL"


def _tf_has_fresh_cross(r: dict, direction: str) -> bool:
    """Does this TF have a FRESH cross in the given direction?"""
    if not r.get("available", False):
        return False
    if direction == "LONG":
        return r.get("long_score", 0) > 0
    elif direction == "SHORT":
        return r.get("short_score", 0) > 0
    return False


def compute_stoch_setup_gate(
    bars: dict[str, pd.DataFrame],
) -> dict:
    """
    CONFLUENCE-FIRST stochastic gate.

    We're looking to WIN — not just spit out scans. Only pass setups where
    multiple timeframes agree on direction with fresh crosses.

    SCORING TIERS (confluence_level):
      5 = "PERFECT"  — 1WK + 1D + 4H all have fresh crosses, same direction
      4 = "STRONG"   — 1WK + 1D fresh crosses aligned, 4H confirms direction
      3 = "GOOD"     — 1WK + 1D aligned direction (%K vs %D), at least one fresh cross
      2 = "WEAK"     — Only one higher TF has a cross, others neutral
      1 = "CONFLICT" — Higher TFs disagree (1WK bull, 1D bear) → REJECT
      0 = "NONE"     — No crosses anywhere → REJECT

    HARD RULES:
      - If 1WK is bullish and 1D is bearish (or vice versa) → REJECT
      - Every TF that agrees must be FRESH (not already ran)
      - The confluence_level directly scales the final score

    Returns per-TF breakdown + regime direction + confluence_level.
    """
    all_tfs = ["1wk", "1d", "4h", "1h", "15m"]
    tf_lookbacks = {"15m": 8, "1h": 5, "4h": 4, "1d": 5, "1wk": 4}

    tf_results = {}
    for tf in all_tfs:
        df = bars.get(tf)
        lb = tf_lookbacks.get(tf, 5)
        tf_results[tf] = _detect_stoch_setup(df, lookback=lb)

    # ── Step 1: Read direction + freshness of each key TF ────────
    wk_r = tf_results.get("1wk", {})
    dy_r = tf_results.get("1d", {})
    h4_r = tf_results.get("4h", {})
    h1_r = tf_results.get("1h", {})
    m15_r = tf_results.get("15m", {})

    wk_dir = _tf_direction(wk_r)
    dy_dir = _tf_direction(dy_r)
    h4_dir = _tf_direction(h4_r)

    wk_cross_long = _tf_has_fresh_cross(wk_r, "LONG")
    wk_cross_short = _tf_has_fresh_cross(wk_r, "SHORT")
    dy_cross_long = _tf_has_fresh_cross(dy_r, "LONG")
    dy_cross_short = _tf_has_fresh_cross(dy_r, "SHORT")
    h4_cross_long = _tf_has_fresh_cross(h4_r, "LONG")
    h4_cross_short = _tf_has_fresh_cross(h4_r, "SHORT")

    # ── Step 2: Determine confluence ─────────────────────────────
    regime_dir = "NO_TRADE"
    regime_tf = ""
    regime_score = 0.0
    confluence_level = 0
    confluence_label = "NONE"
    entry_confirms = 0
    entry_tfs_list = []
    label = ""

    # Check for CONFLICT first — weekly and daily disagree = no trade
    if wk_dir in ("LONG", "SHORT") and dy_dir in ("LONG", "SHORT") and wk_dir != dy_dir:
        confluence_level = 1
        confluence_label = "CONFLICT"
        label = (f"TF conflict — 1WK {wk_dir} (%K={wk_r.get('k',0):.0f}) "
                 f"vs 1D {dy_dir} (%K={dy_r.get('k',0):.0f}) — wait for alignment")
        # Still return tf_results but REJECT
    else:
        # Determine direction from the highest agreeing TFs
        # Priority: weekly > daily > 4h
        if wk_dir in ("LONG", "SHORT"):
            regime_dir = wk_dir
            regime_tf = "1wk"
        elif dy_dir in ("LONG", "SHORT"):
            regime_dir = dy_dir
            regime_tf = "1d"
        elif h4_dir in ("LONG", "SHORT"):
            regime_dir = h4_dir
            regime_tf = "4h"

        if regime_dir == "NO_TRADE":
            # Check for any cross at all on lower TFs
            for tf in all_tfs:
                r = tf_results[tf]
                if r.get("long_score", 0) > 0:
                    regime_dir = "LONG"
                    regime_tf = tf
                    regime_score = r["long_score"] * 0.3  # weak — single low TF
                    confluence_level = 2
                    confluence_label = "WEAK"
                    break
                if r.get("short_score", 0) > 0:
                    regime_dir = "SHORT"
                    regime_tf = tf
                    regime_score = r["short_score"] * 0.3
                    confluence_level = 2
                    confluence_label = "WEAK"
                    break

        if regime_dir != "NO_TRADE" and confluence_level == 0:
            # Count how many key TFs (1wk, 1d, 4h) agree
            is_long = regime_dir == "LONG"
            agree_tfs = []
            cross_tfs = []

            for tf_name, tf_dir, has_cross_l, has_cross_s in [
                ("1wk", wk_dir, wk_cross_long, wk_cross_short),
                ("1d",  dy_dir, dy_cross_long, dy_cross_short),
                ("4h",  h4_dir, h4_cross_long, h4_cross_short),
            ]:
                k_val = tf_results[tf_name].get("k", 50)
                # Zone-aware: LONG agreement requires K < 50 (still in bullish zone,
                # not an exhausted short at oversold). SHORT requires K > 50.
                in_zone = (k_val < 50) if is_long else (k_val > 50)
                if tf_dir == regime_dir and in_zone:
                    agree_tfs.append(tf_name)
                    if (is_long and has_cross_l) or (not is_long and has_cross_s):
                        cross_tfs.append(tf_name)

            # Also count 1h, 15m, and 4h as entry confirmations
            # Fresh cross = strongest, but direction agreement also counts
            for tf_name, tf_r in [("4h", h4_r), ("1h", h1_r), ("15m", m15_r)]:
                if tf_name in cross_tfs:
                    continue  # already counted as regime TF
                tf_d = _tf_direction(tf_r)
                has_cross = _tf_has_fresh_cross(tf_r, regime_dir)
                if has_cross:
                    entry_confirms += 1
                    entry_tfs_list.append(tf_name)
                elif tf_d == regime_dir:
                    entry_confirms += 1
                    entry_tfs_list.append(tf_name)

            n_agree = len(agree_tfs)
            n_cross = len(cross_tfs)

            # ── TIER 5: PERFECT — 1WK + 1D + 4H all have fresh crosses
            if n_cross >= 3:
                confluence_level = 5
                confluence_label = "PERFECT"
                regime_score = 1.0
                regime_tf = cross_tfs[0]  # highest TF with cross

            # ── TIER 4: STRONG — 1WK + 1D both have fresh crosses, 4H agrees
            elif n_cross >= 2 and "1wk" in cross_tfs and "1d" in cross_tfs:
                confluence_level = 5 if h4_dir == regime_dir else 4
                confluence_label = "PERFECT" if h4_dir == regime_dir else "STRONG"
                regime_score = 0.95 if confluence_level == 5 else 0.85
                regime_tf = "1wk"

            # ── TIER 4: STRONG — any 2 of (1wk,1d,4h) have fresh crosses
            elif n_cross >= 2:
                confluence_level = 4
                confluence_label = "STRONG"
                regime_score = 0.80
                regime_tf = cross_tfs[0]

            # ── TIER 3: GOOD — 1WK + 1D agree direction, at least one cross
            elif n_agree >= 2 and n_cross >= 1:
                confluence_level = 3
                confluence_label = "GOOD"
                regime_score = 0.65
                regime_tf = cross_tfs[0]

            # ── TIER 2: WEAK — only one TF has a cross, others neutral
            elif n_cross >= 1:
                confluence_level = 2
                confluence_label = "WEAK"
                regime_score = 0.35
                regime_tf = cross_tfs[0]

            else:
                # Direction agrees but no fresh crosses — stale
                confluence_level = 2
                confluence_label = "WEAK"
                regime_score = 0.20
                regime_tf = agree_tfs[0] if agree_tfs else ""

    # ── Base compression check (price structure) ──────────────────
    base_df = bars.get("1d") if bars.get("1d") is not None and len(bars.get("1d", [])) >= 20 else bars.get("1h")
    base = _detect_base_compression(base_df)

    # ── Gate decision ─────────────────────────────────────────────
    # Only pass setups with confluence level >= 4 (STRONG or better)
    # GOOD, WEAK, and CONFLICT are rejected — fresh cross on 2+ key TFs required.
    # This ensures the run literally hasn't started yet across multiple timeframes.
    setup_pass = confluence_level >= 4

    # ── Build label ───────────────────────────────────────────────
    if not label:  # conflict already set its own label
        if setup_pass:
            dir_emoji = "LONG" if regime_dir == "LONG" else "SHORT"
            cross_info = []
            for tf in ["1wk", "1d", "4h"]:
                r = tf_results[tf]
                if not r.get("available"):
                    continue
                d = _tf_direction(r)
                has_cross = _tf_has_fresh_cross(r, regime_dir)
                status = "CROSS" if has_cross else d
                cross_info.append(f"{tf.upper()}:{status}(%K={r['k']:.0f})")
            label = f"{dir_emoji} {confluence_label} — {' | '.join(cross_info)}"
            if entry_confirms > 0:
                label += f" | entry: {','.join(entry_tfs_list)}"
            if base.get("compressed"):
                label += " | base compressed"
        else:
            if confluence_level == 1:
                pass  # conflict label already set
            elif confluence_level == 2:
                parts = []
                for tf in ["1wk", "1d", "4h"]:
                    r = tf_results[tf]
                    if r.get("available"):
                        d = _tf_direction(r)
                        parts.append(f"{tf.upper()}:{d}(%K={r['k']:.0f})")
                label = f"Weak — only 1 TF cross, no confluence: {' | '.join(parts)}"
            else:
                parts = []
                for tf in all_tfs:
                    r = tf_results[tf]
                    if r.get("available"):
                        parts.append(f"{tf}:%K={r['k']:.0f}")
                label = f"No setup — {', '.join(parts)}"

    return {
        "pass": setup_pass,
        "direction": regime_dir,
        "regime_tf": regime_tf,
        "regime_score": round(regime_score, 2),
        "confluence_level": confluence_level,
        "confluence_label": confluence_label,
        "entry_confirms": entry_confirms,
        "entry_tfs": entry_tfs_list,
        "base": base,
        "label": label,
        "tf_results": tf_results,
    }


@dataclass
class ScoredSignal:
    symbol: str
    profile: str              # SCALP | SWING | POSITION
    score: float              # 0-100
    direction: str            # LONG | SHORT | NO_TRADE
    confidence: str           # HIGH | MEDIUM | LOW | NONE
    components: dict[str, float] = field(default_factory=dict)
    component_details: dict[str, Any] = field(default_factory=dict)
    rejection_reason: str = ""
    bonuses: list[str] = field(default_factory=list)


UNIFIED_WEIGHTS = {
    # Sweep Sniper (20%) — highest priority: liquidity sweep + stochastic
    "sweep_sniper":        0.12,
    "stoch_timing":        0.08,
    # Direction Agreement (25%)
    "mtf_alignment":       0.10,
    "trend_alignment":     0.08,
    "wave_phase":          0.04,
    "structure_shift":     0.03,
    # Smart Money (15%)
    "fvg_confluence":      0.05,
    "order_block":         0.05,
    "premium_discount":    0.03,
    "liquidity_sweep":     0.02,
    # Institutional Flow (12%)
    "gex_signal":          0.06,
    "dealer_positioning":  0.03,
    "catalyst":            0.03,
    # Structure Quality (13%)
    "pattern_score":       0.04,
    "level_confluence":    0.04,
    "breakout_strength":   0.03,
    "relative_strength":   0.02,
    # Timing (15%)
    "cycle_alignment":     0.04,
    "exhaustion_check":    0.04,
    "vol_compression":     0.07,
}

TRADE_PROFILES = {
    "SCALP": {
        "mtf_model": "DAY_TRADE",
        "bar_tf": "5m",
        "smc_tf": "5m",
        "structure_tf": "15m",
        "stop_atr_mult": 0.75,
        "target_rr": 2.5,
    },
    "SWING": {
        "mtf_model": "SWING_TRADE",
        "bar_tf": "1h",
        "smc_tf": "1h",
        "structure_tf": "4h",
        "stop_atr_mult": 1.5,
        "target_rr": 3.0,
    },
    "POSITION": {
        "mtf_model": "INVESTMENT",
        "bar_tf": "1d",
        "smc_tf": "4h",
        "structure_tf": "1d",
        "stop_atr_mult": 2.0,
        "target_rr": 4.0,
    },
}


class UnifiedScorer:
    """Score a symbol using 18 weighted components + hard gates + boosters."""

    def __init__(self):
        self._weights = UNIFIED_WEIGHTS

    def score(self, sd) -> ScoredSignal:
        """
        Score a SymbolData object.

        Args:
            sd: SymbolData from collector.py (has .symbol, .profile, .bars,
                .options_chain, .current_price)

        Returns:
            ScoredSignal with score 0-100, direction, confidence, and
            component breakdown
        """
        profile_cfg = TRADE_PROFILES.get(sd.profile, TRADE_PROFILES["SWING"])
        components: dict[str, float] = {}
        details: dict[str, Any] = {}
        bonuses: list[str] = []

        # Get the primary analysis DataFrame
        bar_tf = profile_cfg["bar_tf"]
        smc_tf = profile_cfg["smc_tf"]
        primary_df = sd.bars.get(bar_tf)
        if primary_df is None or (
            isinstance(primary_df, pd.DataFrame) and primary_df.empty
        ):
            primary_df = self._best_available_df(sd)

        if primary_df is None or len(primary_df) < 10:
            return ScoredSignal(
                symbol=sd.symbol,
                profile=sd.profile,
                score=0,
                direction="NO_TRADE",
                confidence="NONE",
                rejection_reason="Insufficient price data",
            )

        current_price = sd.current_price or float(primary_df["close"].iloc[-1])

        # Compute ATR
        atr = self._compute_atr(primary_df)

        # ── STOCHASTIC SETUP GATE — the ONLY hard gate ──────────────
        # Find the QUBT setup: Slow STO extreme-zone cross.
        # LONG:  %K & %D both < 20, %K just crossed above %D, no run-up
        # SHORT: %K & %D both > 80, %K just crossed below %D, no run-up
        # ANY timeframe showing this passes. Multiple TFs = higher confidence.
        stoch_gate = compute_stoch_setup_gate(sd.bars)
        if not stoch_gate["pass"]:
            return ScoredSignal(
                symbol=sd.symbol,
                profile=sd.profile,
                score=0,
                direction="NO_TRADE",
                confidence="NONE",
                rejection_reason=stoch_gate["label"],
                component_details={"stoch_gate": stoch_gate},
            )

        # Direction comes from the stochastic setup itself
        gate_direction = stoch_gate["direction"]  # "LONG" or "SHORT"

        # ── 1. MTF Alignment (15%) ──────────────────────────────────
        mtf_score = 0.5
        mtf_bias = "NEUTRAL"
        try:
            from mtf_alignment.mtf_alignment_engine import MTFAlignmentEngine

            mtf_engine = MTFAlignmentEngine(
                symbol=sd.symbol, model=profile_cfg["mtf_model"]
            )
            # Build timeframe dict from available bars
            tf_map = {}
            for tf_key, df in sd.bars.items():
                label = self._tf_to_mtf_label(tf_key)
                if label:
                    tf_map[label] = df
            if tf_map:
                mtf_result = mtf_engine.run(tf_map)
                mtf_score = mtf_result.get("alignment_score", 0.5)
                mtf_bias = mtf_result.get("bias_direction", "NEUTRAL")
                details["mtf"] = mtf_result
        except Exception as e:
            logger.debug("MTF engine unavailable: %s", e)
        components["mtf_alignment"] = mtf_score

        # ── MTF PENALTY (soft — no longer a hard kill) ─────────────
        # If MTF is 0 (RANGE/NEUTRAL), fall back to trend direction
        # and apply a penalty later instead of rejecting outright.
        mtf_penalty = 0.0
        if mtf_score < 0.20:
            mtf_penalty = 8.0   # moderate penalty, not a kill
        elif mtf_score < 0.40:
            mtf_penalty = 4.0   # slight penalty

        # Direction is dictated by the stochastic setup (not MTF alignment)
        target_dir = "long" if gate_direction == "LONG" else "short"

        # ── 2. Trend Alignment (10%) ────────────────────────────────
        components["trend_alignment"] = self._score_trend(primary_df)

        # ── 3. Wave Phase (5%) ──────────────────────────────────────
        wave_score = 0.5
        wave_phase = "TRANSITION"
        wave_direction = "NEUTRAL"
        wave_confidence = 0.0
        try:
            from wave_phase.wave_phase_engine import detect_wave_phase

            wave_result = detect_wave_phase(primary_df, symbol=sd.symbol)
            wave_phase = wave_result.get("wave_phase", "TRANSITION")
            wave_direction = wave_result.get("direction", "NEUTRAL")
            wave_confidence = wave_result.get("confidence", 0.0)
            if wave_phase == "IMPULSE":
                wave_score = 0.75 if wave_direction == "UP" else 0.25
            elif wave_phase == "CORRECTION":
                wave_score = 0.35
            details["wave"] = wave_result
        except Exception as e:
            logger.debug("Wave phase unavailable: %s", e)
        components["wave_phase"] = wave_score

        # ── PENALTY: Correction + Long + High Confidence ────────────
        correction_penalty = 0.0
        if wave_phase == "CORRECTION" and target_dir == "long" and wave_confidence > 0.6:
            correction_penalty = 12.0
            bonuses.append("penalty: correction phase active (-12)")

        # ── 4. SMC Components ──────────────────────────────────────
        smc_result = None
        smc_df = sd.bars.get(smc_tf)
        if smc_df is None or (
            isinstance(smc_df, pd.DataFrame) and smc_df.empty
        ):
            smc_df = primary_df
        try:
            from smc_engine.smc_scorer import score_smc

            if len(smc_df) >= 20:
                smc_result = score_smc(smc_df, current_price=current_price)

                # Structure shift (5%)
                if "CHoCH" in smc_result.latest_shift:
                    if "bullish" in smc_result.latest_shift:
                        components["structure_shift"] = 0.82
                    else:
                        components["structure_shift"] = 0.18
                elif "BOS" in smc_result.latest_shift:
                    if "bullish" in smc_result.latest_shift:
                        components["structure_shift"] = 0.70
                    else:
                        components["structure_shift"] = 0.30
                else:
                    components["structure_shift"] = 0.50

                # FVG confluence (6%)
                components["fvg_confluence"] = (
                    smc_result.smc_score if smc_result.fvg_count > 0 else 0.5
                )

                # Order block (6%)
                components["order_block"] = (
                    smc_result.smc_score if smc_result.ob_count > 0 else 0.5
                )

                # Premium/Discount (4%)
                if smc_result.zone == "DISCOUNT":
                    components["premium_discount"] = 0.75
                elif smc_result.zone == "PREMIUM":
                    components["premium_discount"] = 0.25
                else:
                    components["premium_discount"] = 0.50

                details["smc"] = {
                    "sniper_entry": smc_result.sniper_entry,
                    "entry_zone_top": smc_result.entry_zone_top,
                    "entry_zone_bottom": smc_result.entry_zone_bottom,
                    "bias": smc_result.smc_bias,
                    "in_ote": smc_result.in_ote,
                    "zone": smc_result.zone,
                    "fvg_count": smc_result.fvg_count,
                    "ob_count": smc_result.ob_count,
                    "latest_shift": smc_result.latest_shift,
                }
        except Exception as e:
            logger.debug("SMC engine unavailable: %s", e)

        # Defaults for SMC components if not set
        components.setdefault("structure_shift", 0.50)
        components.setdefault("fvg_confluence", 0.50)
        components.setdefault("order_block", 0.50)
        components.setdefault("premium_discount", 0.50)

        # ── PENALTY: Premium zone for longs / Discount for shorts ──
        zone_penalty = 0.0
        if smc_result:
            if smc_result.zone == "PREMIUM" and target_dir == "long":
                zone_penalty = 8.0
                bonuses.append("penalty: PREMIUM zone for longs (-8)")
            if smc_result.zone == "DISCOUNT" and target_dir == "short":
                zone_penalty = 8.0
                bonuses.append("penalty: DISCOUNT zone for shorts (-8)")

        # ── SWEEP SNIPER (20%) — highest priority component ────────
        sweep_result = None
        try:
            from trade_signal_engine.sweep_sniper import SweepSniper

            sniper = SweepSniper()
            sweep_result = sniper.analyze(primary_df, current_price, sd.profile)

            # sweep_sniper component (12%): based on sweep score
            if sweep_result.has_sweep:
                # Map 0-100 sweep score to 0.5-1.0 (sweep is always bullish)
                components["sweep_sniper"] = min(1.0, 0.5 + sweep_result.sweep_score / 200)
            else:
                components["sweep_sniper"] = 0.45  # slight bearish if no sweep

            # stoch_timing component (8%): stochastic oversold + cross
            stoch_score = 0.5
            if sweep_result.stoch_oversold and sweep_result.stoch_cross:
                stoch_score = 0.95  # perfect timing: oversold + bullish cross
            elif sweep_result.stoch_oversold:
                stoch_score = 0.80  # oversold but no cross yet
            elif sweep_result.stoch_cross:
                stoch_score = 0.70  # cross but not from deep oversold
            elif sweep_result.stoch_k < 30:
                stoch_score = 0.65  # approaching oversold
            elif sweep_result.stoch_k > 80:
                stoch_score = 0.30  # overbought — bad timing
            components["stoch_timing"] = stoch_score

            details["sweep"] = {
                "has_sweep": sweep_result.has_sweep,
                "sweep_score": sweep_result.sweep_score,
                "sweep_level": sweep_result.sweep_level,
                "entry": sweep_result.entry,
                "stop": sweep_result.stop,
                "target": sweep_result.target,
                "extended_target": sweep_result.extended_target,
                "runner_target": sweep_result.runner_target,
                "risk_reward": sweep_result.risk_reward,
                "stoch_k": sweep_result.stoch_k,
                "stoch_d": sweep_result.stoch_d,
                "stoch_oversold": sweep_result.stoch_oversold,
                "stoch_cross": sweep_result.stoch_cross,
                "base_compressed": sweep_result.base_compressed,
                "volume_confirmed": sweep_result.volume_confirmed,
                "support_pools": sweep_result.support_pools,
                "resistance_pools": sweep_result.resistance_pools,
            }
        except Exception as e:
            logger.debug("Sweep sniper unavailable: %s", e)

        components.setdefault("sweep_sniper", 0.50)
        components.setdefault("stoch_timing", 0.50)

        # ── 5. Liquidity Sweep (2%) — legacy component ────────────
        components["liquidity_sweep"] = self._score_liquidity(primary_df, atr)

        # ── 6. Time Exhaustion (5%) ────────────────────────────────
        exhaustion_detected = False
        exhaustion_dir = "NONE"
        try:
            from time_exhaustion.exhaustion_engine import detect_exhaustion

            exh_result = detect_exhaustion(
                primary_df, timeframe=bar_tf, symbol=sd.symbol
            )
            exhaustion_detected = exh_result.get("exhaustion_detected", False)
            exhaustion_dir = exh_result.get("direction", "NONE")
            exh_strength = exh_result.get("exhaustion_strength", 0.0)
            components["exhaustion_check"] = (
                1.0 - exh_strength if exhaustion_detected else 0.60
            )
            details["exhaustion"] = exh_result
        except Exception as e:
            logger.debug("Exhaustion engine unavailable: %s", e)
            components["exhaustion_check"] = 0.5

        # ── PENALTY: Exhaustion in trade direction ──────────────────
        exhaustion_penalty = 0.0
        if exhaustion_detected:
            if (exhaustion_dir == "UP" and target_dir == "long") or (
                exhaustion_dir == "DOWN" and target_dir == "short"
            ):
                exhaustion_penalty = 10.0
                bonuses.append(f"penalty: exhaustion in {exhaustion_dir} (-10)")

        # ── 7. GEX Signal (8%) ─────────────────────────────────────
        gex_profile = None
        if sd.options_chain:
            try:
                from options_engine.gamma_exposure import GammaExposureEngine

                gex_engine = GammaExposureEngine()
                gex_profile = gex_engine.compute(sd.options_chain)
                gex_score = gex_profile.gex_score / 100.0
                components["gex_signal"] = gex_score
                details["gex"] = {
                    "gamma_regime": gex_profile.gamma_regime,
                    "gex_score": gex_profile.gex_score,
                    "call_wall": gex_profile.call_wall,
                    "put_wall": gex_profile.put_wall,
                    "zero_gamma": gex_profile.zero_gamma,
                }
            except Exception as e:
                logger.debug("GEX engine unavailable: %s", e)
        components.setdefault("gex_signal", 0.50)

        # ── 8. Dealer Positioning (4%) ─────────────────────────────
        if gex_profile:
            try:
                from options_engine.dealer_positioning import (
                    DealerPositioningEngine,
                )

                dp_engine = DealerPositioningEngine()
                dp_result = dp_engine.compute(gex_profile, atr=atr)
                dp_score = dp_result.positioning_score / 100.0
                components["dealer_positioning"] = dp_score
                details["dealer"] = {
                    "regime": dp_result.dealer_regime,
                    "squeeze_risk": dp_result.squeeze_risk,
                    "flow_bias": dp_result.flow_bias,
                }
            except Exception as e:
                logger.debug("Dealer positioning unavailable: %s", e)
        components.setdefault("dealer_positioning", 0.50)

        # ── 9. Catalyst (3%) ───────────────────────────────────────
        try:
            from context_engine.catalyst_engine import CatalystEngine

            cat_engine = CatalystEngine()
            cat_result = cat_engine.score_symbol(sd.symbol)
            cat_score_raw = cat_result.get("catalyst_score", 0.0)
            cat_sentiment = cat_result.get("sentiment", "neutral")
            if cat_sentiment == "bullish":
                components["catalyst"] = 0.5 + cat_score_raw * 0.4
            elif cat_sentiment == "bearish":
                components["catalyst"] = 0.5 - cat_score_raw * 0.4
            else:
                components["catalyst"] = 0.50
            details["catalyst"] = cat_result
        except Exception as e:
            logger.debug("Catalyst engine unavailable: %s", e)
            components.setdefault("catalyst", 0.50)

        # ── 10. Pattern Score (5%) ─────────────────────────────────
        components.setdefault("pattern_score", 0.50)  # TODO: wire PatternService

        # ── 11. Level Confluence (5%) ──────────────────────────────
        try:
            from signal_engine.level_engine import LevelEngine

            level_engine = LevelEngine()
            level_result = level_engine.compute(sd.symbol)
            gems = level_result.get("gems", [])
            pounce = level_result.get("pounce", [])
            if pounce:
                components["level_confluence"] = 0.85
            elif gems:
                components["level_confluence"] = 0.70
            else:
                components["level_confluence"] = 0.50
            details["levels"] = {
                "gem_count": len(gems),
                "pounce_count": len(pounce),
                "pounce_ideas": [
                    g.get("trade_idea") for g in pounce if g.get("trade_idea")
                ],
            }
        except Exception as e:
            logger.debug("Level engine unavailable: %s", e)
            components.setdefault("level_confluence", 0.50)

        # ── 12. Breakout Strength (3%) ─────────────────────────────
        components["breakout_strength"] = self._score_breakout(primary_df, atr)

        # ── 13. Relative Strength vs SPY (2%) ──────────────────────
        components["relative_strength"] = self._score_relative_strength(
            sd.symbol, primary_df
        )

        # ── 14. Cycle Alignment (5%) ───────────────────────────────
        components.setdefault(
            "cycle_alignment", 0.50
        )  # TODO: wire FFT/wavelet

        # ── 15. Vol Compression (5%) ───────────────────────────────
        components["vol_compression"] = self._score_vol_compression(primary_df)

        # ── Compute weighted score ─────────────────────────────────
        weighted_sum = sum(
            components.get(k, 0.5) * w for k, w in self._weights.items()
        )
        raw_score = weighted_sum * 100  # 0-100 scale

        # ── PENALTY: No structure confirmation (very soft) ─────────
        structure_val = components.get("structure_shift", 0.50)
        if abs(structure_val - 0.50) < 0.05:  # neutral = no CHoCH/BOS
            raw_score *= 0.92  # reduce by 8% (was 30%)
            bonuses.append("penalty: no structure confirmation (-8%)")

        # ── SCORE BOOSTERS ─────────────────────────────────────────
        bonus = 0

        # ── CONFLUENCE BONUS ───────────────────────────────────────
        # The confluence level from the stochastic gate is the PRIMARY
        # score driver. More TFs agreeing = dramatically higher score.
        details["stoch_gate"] = stoch_gate
        regime_score = stoch_gate["regime_score"]
        regime_tf = stoch_gate["regime_tf"]
        entry_confirms = stoch_gate["entry_confirms"]
        gate_base = stoch_gate.get("base", {})
        confluence_level = stoch_gate.get("confluence_level", 2)
        confluence_label = stoch_gate.get("confluence_label", "WEAK")

        # Confluence-based scoring (replaces flat regime bonus)
        if confluence_level >= 5:
            bonus += 40
            bonuses.append(f"PERFECT CONFLUENCE — 1WK+1D+4H all crossing {stoch_gate['direction']} (+40)")
        elif confluence_level >= 4:
            bonus += 30
            bonuses.append(f"STRONG CONFLUENCE — multiple TF crosses aligned (+30)")
        elif confluence_level >= 3:
            bonus += 15
            bonuses.append(f"GOOD confluence — TFs agree with fresh cross (+15)")
        else:
            bonus += 5
            bonuses.append(f"Weak confluence — single TF cross (+5)")

        # Entry TF confirmations (1h, 15m confirming the regime)
        if entry_confirms >= 2:
            bonus += 15
            entry_label = ",".join(stoch_gate.get("entry_tfs", []))
            bonuses.append(f"{entry_confirms} entry TFs confirm: {entry_label} (+15)")
        elif entry_confirms >= 1:
            bonus += 8
            entry_label = ",".join(stoch_gate.get("entry_tfs", []))
            bonuses.append(f"Entry TF confirm: {entry_label} (+8)")

        # PENALTY for low confluence — scale down the entire raw score
        # Weak setups with only 1 TF agreeing get a significant haircut
        if confluence_level <= 3:
            raw_score *= 0.75
            bonuses.append(f"Confluence penalty: {confluence_label} (-25%)")

        # Base compression bonus — the pink circle on price
        if gate_base.get("compressed"):
            bonus += 12
            bonuses.append("Base compression — coiled spring (+12)")
        if target_dir == "long" and gate_base.get("at_support"):
            bonus += 8
            bonuses.append("Price at support level (+8)")
        elif target_dir == "short" and gate_base.get("at_resistance"):
            bonus += 8
            bonuses.append("Price at resistance level (+8)")

        # ── GOLDEN / DEATH CROSS ──────────────────────────────────
        # 50 SMA crossing 200 SMA on daily bars — strong trend signal
        daily_df = sd.bars.get("1d")
        gd_cross = detect_golden_death_cross(daily_df, lookback=10)
        details["golden_death_cross"] = gd_cross

        if target_dir == "long" and gd_cross["golden_cross"]:
            bonus += 20
            age = gd_cross["cross_age"]
            bonuses.append(f"GOLDEN CROSS — 50 SMA above 200 SMA ({age}d ago, +20)")
        elif target_dir == "long" and gd_cross["sma50_above_200"]:
            bonus += 5
            bonuses.append("50 SMA > 200 SMA (bullish structure, +5)")
        elif target_dir == "long" and not gd_cross["sma50_above_200"] and gd_cross["sma200"] > 0:
            bonus -= 10
            bonuses.append("50 SMA < 200 SMA (bearish structure, -10)")

        if target_dir == "short" and gd_cross["death_cross"]:
            bonus += 20
            age = gd_cross["cross_age"]
            bonuses.append(f"DEATH CROSS — 50 SMA below 200 SMA ({age}d ago, +20)")
        elif target_dir == "short" and not gd_cross["sma50_above_200"] and gd_cross["sma200"] > 0:
            bonus += 5
            bonuses.append("50 SMA < 200 SMA (bearish structure, +5)")
        elif target_dir == "short" and gd_cross["sma50_above_200"]:
            bonus -= 10
            bonuses.append("50 SMA > 200 SMA (fighting bullish trend, -10)")

        # ── SWEEP + STOCHASTIC CONFIRMATION (highest conviction) ──
        # The VFS pattern: liquidity sweep at support + stochastic oversold/cross
        # When all signals align, this is the highest-probability entry
        if sweep_result and sweep_result.has_sweep:
            if sweep_result.stoch_oversold and sweep_result.stoch_cross:
                bonus += 25
                bonuses.append("SNIPER: sweep + stoch oversold + bullish cross (+25)")
            elif sweep_result.stoch_oversold:
                bonus += 15
                bonuses.append("sweep + stoch oversold (+15)")
            elif sweep_result.stoch_cross:
                bonus += 12
                bonuses.append("sweep + stoch bullish cross (+12)")
            else:
                bonus += 8
                bonuses.append("sweep detected (+8)")

            if sweep_result.base_compressed:
                bonus += 10
                bonuses.append("base compression before sweep (+10)")

            if sweep_result.volume_confirmed:
                bonus += 5
                bonuses.append("volume confirmed sweep (+5)")

        # Stochastic-only signals (no sweep, but timing is right)
        elif sweep_result and sweep_result.stoch_oversold and sweep_result.stoch_cross:
            bonus += 10
            bonuses.append("stoch oversold + bullish cross (no sweep) (+10)")

        # ── PENALTY: Stochastic overbought — bad timing ───────────
        if sweep_result and sweep_result.stoch_k > 80 and sweep_result.stoch_d > 80:
            bonus -= 8
            bonuses.append("penalty: stoch overbought — late entry (-8)")

        if smc_result:
            if smc_result.fvg_count > 0 and smc_result.ob_count > 0:
                bonus += 15
                bonuses.append("FVG+OB overlap (+15)")
            if (
                "CHoCH" in smc_result.latest_shift
                and components.get("liquidity_sweep", 0.5) > 0.65
            ):
                bonus += 10
                bonuses.append("liquidity sweep + CHoCH (+10)")
            if smc_result.in_ote:
                bonus += 8
                bonuses.append("in OTE zone (+8)")

        # ── CHASE FILTER: reject stocks that already made their move ──
        # We want setups that are ABOUT TO run, not ones that already ran.
        # Checks: (1) extension from recent low, (2) stochastic position,
        #         (3) distance from MA20, (4) ATR expansion (volatility spike)
        chase_penalty = 0.0
        chase_reasons: list[str] = []
        if len(primary_df) >= 20 and atr > 0:
            closes = primary_df["close"].values
            lows   = primary_df["low"].values
            price  = float(closes[-1])
            ma20   = float(np.mean(closes[-20:]))

            # 1) Extension from 20-bar low — how far has it already run?
            recent_low_20  = float(np.min(lows[-20:]))
            extension_pct  = (price - recent_low_20) / recent_low_20 * 100 if recent_low_20 > 0 else 0

            # 2) Distance from MA20 in ATR units — is price stretched?
            ma20_distance_atr = (price - ma20) / atr if atr > 0 else 0

            # 3) ATR expansion — is recent volatility spiking vs average?
            if len(primary_df) >= 40:
                highs = primary_df["high"].values
                tr_recent = float(np.mean(highs[-5:] - lows[-5:]))
                tr_avg    = float(np.mean(highs[-40:] - lows[-40:]))
                atr_expansion = tr_recent / tr_avg if tr_avg > 0 else 1.0
            else:
                atr_expansion = 1.0

            # 4) Slow stochastic if available from sweep analysis
            stoch_k = sweep_result.stoch_k if sweep_result else 50

            # === Scoring the chase risk ===
            # Extended >25% from 20-bar low: the move already happened
            if extension_pct > 40:
                chase_penalty += 25
                chase_reasons.append(f"CHASE: +{extension_pct:.0f}% from 20-bar low — move complete (-25)")
            elif extension_pct > 25:
                chase_penalty += 15
                chase_reasons.append(f"CHASE: +{extension_pct:.0f}% from 20-bar low — extended (-15)")

            # Price >2 ATR above MA20: already far from mean
            if ma20_distance_atr > 3.0:
                chase_penalty += 15
                chase_reasons.append(f"CHASE: {ma20_distance_atr:.1f}x ATR above MA20 — overstretched (-15)")
            elif ma20_distance_atr > 2.0:
                chase_penalty += 8
                chase_reasons.append(f"CHASE: {ma20_distance_atr:.1f}x ATR above MA20 (-8)")

            # Stochastic deeply overbought (>85) while extended — the worst combo
            if stoch_k > 85 and extension_pct > 20:
                chase_penalty += 12
                chase_reasons.append(f"CHASE: stoch {stoch_k:.0f} + extended {extension_pct:.0f}% — chasing (-12)")

            # Volatility spike (recent TR > 2x average) — move is happening NOW, too late
            if atr_expansion > 2.5 and extension_pct > 15:
                chase_penalty += 10
                chase_reasons.append(f"CHASE: volatility spike {atr_expansion:.1f}x — parabolic move (-10)")

            bonuses.extend(chase_reasons)

        # ── Apply soft penalties (replaces hard gates) ─────────────
        total_penalty = mtf_penalty + correction_penalty + zone_penalty + exhaustion_penalty + chase_penalty
        final_score = min(100, max(0, raw_score + bonus - total_penalty))

        # ── Determine direction and confidence ─────────────────────
        # Surface every trade the market offers — both LONG and SHORT.
        # Inverse/hedge ETFs with bearish signal → treated as LONG (buy the inverse).
        _INVERSE_ETFS = {
            "SQQQ", "SPXS", "SDOW", "TZA", "SOXS", "LABD", "FAZ", "TECS", "DUST",
            "SH", "PSQ", "DOG", "RWM", "UVXY", "VXX", "VIXY",
        }
        is_inverse = sd.symbol in _INVERSE_ETFS

        if final_score >= 55:
            if target_dir == "long":
                direction = "LONG"
            elif is_inverse:
                direction = "LONG"   # buy the inverse ETF
            else:
                direction = "SHORT"  # bearish on regular stock → SHORT
            confidence = "HIGH" if final_score >= 80 else "MEDIUM" if final_score >= 65 else "LOW"
        elif final_score >= 35:
            if target_dir == "long":
                direction = "LONG"
            elif is_inverse:
                direction = "LONG"
            else:
                direction = "SHORT"
            confidence = "LOW"
        else:
            direction = "NO_TRADE"
            confidence = "NONE"

        # If chase penalty alone was >= 30, override to NO_TRADE regardless of score
        rejection = ""
        if chase_penalty >= 30 and direction != "NO_TRADE":
            direction = "NO_TRADE"
            confidence = "NONE"
            rejection = "Already ran — chasing extended move"

        return ScoredSignal(
            symbol=sd.symbol,
            profile=sd.profile,
            score=round(final_score, 1),
            direction=direction,
            confidence=confidence,
            components=components,
            component_details=details,
            rejection_reason=rejection,
            bonuses=bonuses,
        )

    # ── Helper methods ─────────────────────────────────────────────

    def _best_available_df(self, sd) -> Optional[pd.DataFrame]:
        """Get the best available DataFrame from SymbolData."""
        for tf in ["1h", "5m", "15m", "4h", "1d", "1wk"]:
            if tf in sd.bars and not sd.bars[tf].empty:
                return sd.bars[tf]
        return None

    def _compute_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Compute ATR-14."""
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        if len(df) < 2:
            return 1.0
        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1]),
            ),
        )
        atr = (
            float(np.mean(tr[-period:]))
            if len(tr) >= period
            else float(np.mean(tr))
        )
        return max(atr, 0.01)

    def _tf_to_mtf_label(self, tf: str) -> Optional[str]:
        """Map yfinance timeframe strings to MTF engine labels."""
        mapping = {
            "5m": "5m",
            "15m": "15m",
            "1h": "1h",
            "4h": "4h",
            "1d": "daily",
            "1wk": "weekly",
        }
        return mapping.get(tf)

    def _score_trend(self, df: pd.DataFrame) -> float:
        """Score trend alignment from MA20/50."""
        if len(df) < 50:
            return 0.5
        closes = df["close"].values
        ma20 = np.mean(closes[-20:])
        ma50 = np.mean(closes[-50:])
        price = closes[-1]

        score = 0.5
        if price > ma50:
            score += 0.15
        if price > ma20:
            score += 0.10
        if ma20 > ma50:
            score += 0.10
        if price < ma50:
            score -= 0.15
        if price < ma20:
            score -= 0.10
        if ma20 < ma50:
            score -= 0.10

        return max(0.0, min(1.0, score))

    def _score_liquidity(self, df: pd.DataFrame, atr: float) -> float:
        """Score liquidity sweep (stop hunt) detection."""
        if len(df) < 20:
            return 0.5
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values

        # Check if recent bar swept below prior low then closed above
        recent_low = min(lows[-20:-1])
        if lows[-1] < recent_low and closes[-1] > recent_low:
            sweep_size = (recent_low - lows[-1]) / atr
            return min(0.90, 0.65 + sweep_size * 0.15)

        # Check sweep above prior high then closed below
        recent_high = max(highs[-20:-1])
        if highs[-1] > recent_high and closes[-1] < recent_high:
            sweep_size = (highs[-1] - recent_high) / atr
            return min(0.90, 0.65 + sweep_size * 0.15)

        return 0.50

    def _score_breakout(self, df: pd.DataFrame, atr: float) -> float:
        """Score breakout strength."""
        if len(df) < 20:
            return 0.5
        closes = df["close"].values
        volumes = (
            df["volume"].values if "volume" in df.columns else np.ones(len(df))
        )

        high_20 = max(df["high"].values[-20:])
        low_20 = min(df["low"].values[-20:])
        price = closes[-1]

        # Breakout above 20-bar high
        if price >= high_20:
            vol_ratio = (
                volumes[-1] / np.mean(volumes[-20:])
                if np.mean(volumes[-20:]) > 0
                else 1
            )
            return min(1.0, 0.60 + vol_ratio * 0.10)

        # Breakdown below 20-bar low
        if price <= low_20:
            vol_ratio = (
                volumes[-1] / np.mean(volumes[-20:])
                if np.mean(volumes[-20:]) > 0
                else 1
            )
            return max(0.0, 0.40 - vol_ratio * 0.10)

        return 0.50

    def _score_relative_strength(
        self, symbol: str, df: pd.DataFrame
    ) -> float:
        """Score relative strength vs SPY."""
        if symbol in ("SPY", "SPX") or len(df) < 20:
            return 0.50
        # Simple: compare % change over 20 bars
        pct_change = (df["close"].iloc[-1] / df["close"].iloc[-20] - 1) * 100
        if pct_change > 5:
            return 0.75
        elif pct_change > 0:
            return 0.60
        elif pct_change > -5:
            return 0.40
        else:
            return 0.25

    def _score_vol_compression(self, df: pd.DataFrame) -> float:
        """Score volatility compression (BB squeeze)."""
        if len(df) < 20:
            return 0.5
        closes = df["close"].values
        ma20 = np.mean(closes[-20:])
        std20 = np.std(closes[-20:])
        bb_width = (std20 * 2) / ma20 if ma20 > 0 else 0

        # Also check ATR compression
        atr_now = self._compute_atr(df.iloc[-20:])
        atr_prior = (
            self._compute_atr(df.iloc[-40:-20]) if len(df) >= 40 else atr_now
        )
        atr_ratio = atr_now / atr_prior if atr_prior > 0 else 1.0

        # Low BB width + declining ATR = coiled spring
        if bb_width < 0.03 and atr_ratio < 0.8:
            return 0.85  # highly compressed
        elif bb_width < 0.05:
            return 0.70
        elif bb_width > 0.10:
            return 0.30  # already expanded
        return 0.50
