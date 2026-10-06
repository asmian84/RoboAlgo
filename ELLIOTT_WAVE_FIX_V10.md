# Elliott Wave V10 Fix — v3 Engine Swing Thresholds Are 3-5× Too High (Kills All Wave Detection)

## Problem

The v3 recursive wave engine (`analyze_wave_tree()` → `analyze_recursive()`) returns `None` for nearly every ticker. Only extreme-volatility penny stocks like EONR produce results. BTC-USD, SPY, PLTR, META, AAPL — all return no wave tree.

## Root Cause

**`DEGREE_SWING_MAP` values (line 187-195) are 3-5× too high**, especially when multiplied by `TIMEFRAME_SWING_SCALE` (line 214-218).

Example for equity on daily chart (PRIMARY degree):
- Current: `0.20 (swing map) × 1.5 (TF scale) = 0.30` → **30% min swing** → only 2 pivots survive → zero candidates → None
- Needed: `0.07 × 1.0 = 0.07` → **7% min swing** → 6-12 pivots → 15+ candidates → valid wave tree

The v2 engine works because it uses ATR-based adaptive thresholds (~3-8% effective swing) and binary search for optimal pivot counts. The v3 hardcoded values were set theoretically (based on EW textbook degree definitions) rather than empirically (based on what actually produces detectable pivots in real price data).

## Evidence

Testing on synthetic equity data (50→87 range, 250 bars daily):

| Swing % | Pivots | Monowaves | Candidates | Result |
|---------|--------|-----------|------------|--------|
| 30% (current PRIMARY×daily) | 2 | 1 | 0 | **None** |
| 20% | 2 | 1 | 0 | **None** |
| 15% | 2 | 1 | 0 | **None** |
| 10% | 4 | 3 | 1 | Marginal |
| **7%** | **11** | **10** | **15** | **Works** |
| 5% | 13 | 12 | 42 | Works |
| 3% | 22 | 21 | 206 | Works (noisy) |

After patching to 7% for PRIMARY: v3 produces a wave tree with 3 sub-waves, recursive decomposition, and 2 projections. Confidence 0.823.

---

## Fix A — Correct `DEGREE_SWING_MAP` values

**File: `.worktrees/ew-engine-v2/pattern_engine/elliott_wave.py`**
**Lines: 187-195**

Replace the entire `DEGREE_SWING_MAP` dictionary:

```python
# OLD (lines 187-195):
DEGREE_SWING_MAP: dict[WaveDegree, dict[str, float]] = {
    WaveDegree.SUPERCYCLE:   {'equity': 0.50, 'forex': 0.25, 'crypto': 0.60},
    WaveDegree.CYCLE:        {'equity': 0.35, 'forex': 0.15, 'crypto': 0.45},
    WaveDegree.PRIMARY:      {'equity': 0.20, 'forex': 0.08, 'crypto': 0.30},
    WaveDegree.INTERMEDIATE: {'equity': 0.12, 'forex': 0.05, 'crypto': 0.18},
    WaveDegree.MINOR:        {'equity': 0.06, 'forex': 0.025,'crypto': 0.10},
    WaveDegree.MINUTE:       {'equity': 0.03, 'forex': 0.012,'crypto': 0.05},
    WaveDegree.SUBMINUTE:    {'equity': 0.015,'forex': 0.006, 'crypto': 0.025},
}
```

```python
# NEW — calibrated empirically for 6-15 pivots per degree on typical charts:
DEGREE_SWING_MAP: dict[WaveDegree, dict[str, float]] = {
    WaveDegree.SUPERCYCLE:   {'equity': 0.25,  'forex': 0.10,  'crypto': 0.35},
    WaveDegree.CYCLE:        {'equity': 0.15,  'forex': 0.06,  'crypto': 0.22},
    WaveDegree.PRIMARY:      {'equity': 0.07,  'forex': 0.025, 'crypto': 0.12},
    WaveDegree.INTERMEDIATE: {'equity': 0.04,  'forex': 0.015, 'crypto': 0.07},
    WaveDegree.MINOR:        {'equity': 0.02,  'forex': 0.008, 'crypto': 0.035},
    WaveDegree.MINUTE:       {'equity': 0.01,  'forex': 0.004, 'crypto': 0.018},
    WaveDegree.SUBMINUTE:    {'equity': 0.005, 'forex': 0.002, 'crypto': 0.01},
}
```

Key changes:
- PRIMARY equity: 0.20 → **0.07** (7% swing — gives 6-12 pivots on typical daily equity)
- INTERMEDIATE equity: 0.12 → **0.04** (4% — gives 15-20 pivots for sub-wave detection)
- MINOR equity: 0.06 → **0.02** (2% — fine-grained sub-wave pivots)
- CYCLE equity: 0.35 → **0.15** (15% — for weekly charts, gives 6-10 pivots on multi-year data)
- All other classes scaled proportionally

## Fix B — Reduce `TIMEFRAME_SWING_SCALE` for daily and weekly

**File: `.worktrees/ew-engine-v2/pattern_engine/elliott_wave.py`**
**Lines: 214-218**

Replace:

```python
# OLD (lines 214-218):
TIMEFRAME_SWING_SCALE: dict[str, float] = {
    '1': 0.3, '5': 0.5, '15': 0.7, '60': 1.0,
    '240': 1.2, '4h': 1.2, '1h': 1.0,
    'D': 1.5, '1d': 1.5, 'W': 2.5, '1w': 2.5, 'M': 4.0, '1mo': 4.0,
}
```

```python
# NEW — daily = baseline (1.0); intraday scales down, weekly/monthly scales up gently:
TIMEFRAME_SWING_SCALE: dict[str, float] = {
    '1': 0.3, '5': 0.5, '15': 0.6, '60': 0.8,
    '240': 0.9, '4h': 0.9, '1h': 0.8,
    'D': 1.0, '1d': 1.0, 'W': 1.5, '1w': 1.5, 'M': 2.0, '1mo': 2.0,
}
```

Key changes:
- Daily: 1.5 → **1.0** (swing map values are now calibrated for daily)
- Weekly: 2.5 → **1.5** (gentle upscale, not 2.5×)
- Monthly: 4.0 → **2.0**
- Intraday: slightly reduced across the board

## Fix C — Add adaptive fallback in `analyze_recursive` when zero candidates

Even with corrected thresholds, some tickers may have unusual price action. Add a fallback that progressively relaxes the swing threshold if zero candidates are found.

**File: `.worktrees/ew-engine-v2/pattern_engine/elliott_wave.py`**
**Function: `analyze_recursive()` (line ~664)**

After the current candidate-gathering block (around line ~738-741 where it checks `if not all_cands`), replace:

```python
# OLD (lines ~738-741):
    if not all_cands:
        _memo[memo_key] = None
        return None
```

```python
# NEW — adaptive fallback: if no candidates at the expected swing, try 50% and 25% of the swing:
    if not all_cands:
        for fallback_scale in (0.5, 0.25):
            relaxed_swing = effective_swing * fallback_scale
            pivots    = detect_major_pivots(df, min_swing_pct=relaxed_swing, max_pivots=20)
            monowaves = build_monowaves(pivots) if len(pivots) >= 2 else []
            for mw in monowaves:
                structure, method, count = classify_internal_structure(df, mw.start.idx, mw.end.idx)
                mw.internal_structure    = structure
                mw.classification_method = method
                mw.sub_pivot_count       = count

            if len(monowaves) >= 5:
                all_cands.extend(find_impulse_candidates(monowaves))
            if len(monowaves) >= 3:
                all_cands.extend(find_zigzag_candidates(monowaves))
                all_cands.extend(find_flat_candidates(monowaves))
                all_cands.extend(find_triangle_candidates(monowaves))
            if len(monowaves) >= 5:
                all_cands.extend(find_diagonal_candidates(monowaves))
            if len(pivots) >= 6:
                all_cands.extend(find_pivot_based_impulse_candidates(pivots, df))
            
            if all_cands:
                break  # found candidates at relaxed threshold

        if not all_cands:
            _memo[memo_key] = None
            return None
```

This ensures that even if the primary threshold is slightly too aggressive for a particular chart, the engine tries 50% and then 25% of the threshold before giving up. This mimics the adaptive behavior of the v2 engine.

---

## Verification

After applying these fixes, test these 20 tickers to confirm the v3 engine produces wave trees:

**Equities:** AAPL, MSFT, PLTR, META, INTC, NVDA, TSLA, SPY, QQQ, AMZN
**Penny/volatile:** EONR, CRWV
**Crypto:** BTC-USD, ETH-USD, SOL-USD
**Forex:** EURUSD, GBPJPY
**Commodities:** GC=F, CL=F
**ETFs:** KWEB

For each ticker, verify:
1. `wave_tree` is NOT null in the API response
2. `wave_tree.primary_count.sub_waves` has 3 or 5 entries
3. `wave_tree.current_position` is a non-empty string
4. At least some sub-waves have their own `sub_waves` (recursive decomposition working)

Expected behavior: **all 20 tickers should produce a wave tree**. Previously only EONR worked.
