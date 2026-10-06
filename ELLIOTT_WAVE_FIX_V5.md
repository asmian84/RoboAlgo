# Elliott Wave Engine V5 Fix — Asset-Aware Pivots, Linked Corrections & Graceful Degradation

## Context

V4 added proportionality gates that fixed the "giant W5 with cramped W1-W4" problem.
Post-V4 verification shows:

- **GC=F** ✅ — Excellent proportional 5-wave impulse
- **CL=F** ✅ — Good extended-W3 pattern spanning chart  
- **EONR** ✅ — Correct bearish impulse spanning full decline
- **CAD=X** ❌ — Confused: partial labels, overlapping patterns, no clean dominant structure
- **BTC-USD** ❌ — Completely blank (no patterns returned at all, "INSUFFICIENT DATA")

This V5 fix addresses three root causes.

---

## Bug 13 — Asset-class-aware swing thresholds for pivot detection

### Problem

`analyze_multi_degree()` uses hardcoded swing thresholds:
- `min_swing_pct=0.15` (15%) for primary pivots (line 1979)
- `min_swing_pct=0.07` (7%) for sub-primary pivots (line 1995)

These thresholds are calibrated for US equities and commodities. They fail for:

- **Forex (CAD=X)**: USD/CAD's entire 2025 range is ~1.35–1.45 (a 7.4% total range). A 15% swing literally never occurs. Even 7% barely happens. Result: the pivot detector returns 0–2 pivots, making it impossible to form any 5-wave candidate. The engine falls back to small ATR-based pivots which capture noise, not structure.

- **Crypto (BTC-USD)**: BTC can swing 30-50% in a single impulse wave. With 15% threshold, the detector captures too many intermediate swings (every 15% move), producing 20+ pivots. The combinatorial explosion in `find_pivot_based_impulse_candidates()` may timeout or produce only degenerate candidates. With 7% threshold it's even worse.

### Fix

**File: `.worktrees/ew-engine-v2/pattern_engine/elliott_wave.py`**  
**Function: `analyze_multi_degree()` (line ~1921)**

Add asset-class detection and adaptive swing thresholds BEFORE the pivot detection calls.

Insert this block around line 1954 (after the `inter_atr` calculation, before the primary pivot detection):

```python
    # ── Asset-class-aware swing thresholds ─────────────────────────────
    # Different asset classes have vastly different volatility profiles.
    # Forex pairs rarely swing >5% while crypto can swing >30%.
    # Using the same thresholds across all asset classes produces garbage.
    _sym_upper = df.attrs.get("symbol", "") if hasattr(df, "attrs") else ""
    
    # Detect forex: symbols ending in =X, or known forex pairs
    _FOREX_SUFFIXES = ("=X",)
    _FOREX_PAIRS = {"EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD",
                    "NZDUSD", "USDCAD", "EURGBP", "EURJPY", "GBPJPY"}
    _is_forex = (
        any(_sym_upper.endswith(s) for s in _FOREX_SUFFIXES) or
        _sym_upper.replace("/", "").replace("-", "") in _FOREX_PAIRS
    )
    
    # Detect crypto: symbols containing -USD with known crypto tickers
    _CRYPTO_BASES = {"BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "AVAX",
                     "DOT", "MATIC", "LINK", "UNI", "AAVE", "LTC"}
    _is_crypto = any(_sym_upper.startswith(b + "-") for b in _CRYPTO_BASES)
    
    if _is_forex:
        _primary_swing = 0.035    # 3.5% — a large forex move
        _sub_primary_swing = 0.015  # 1.5%
    elif _is_crypto:
        _primary_swing = 0.22    # 22% — filters minor crypto noise
        _sub_primary_swing = 0.10   # 10%
    else:
        _primary_swing = 0.15    # 15% — equities/commodities default
        _sub_primary_swing = 0.07   # 7%
```

Then update the two `detect_major_pivots()` calls to use these variables:

**Line ~1979** — change:
```python
    primary_pivots    = detect_major_pivots(df, min_swing_pct=0.15, max_pivots=20)
```
to:
```python
    primary_pivots    = detect_major_pivots(df, min_swing_pct=_primary_swing, max_pivots=20)
```

**Line ~1995** — change:
```python
    sub_primary_pivots = detect_major_pivots(df, min_swing_pct=0.07, max_pivots=20)
```
to:
```python
    sub_primary_pivots = detect_major_pivots(df, min_swing_pct=_sub_primary_swing, max_pivots=20)
```

**CRITICAL**: The symbol string must be passed into `analyze_multi_degree()` so it can detect the asset class. Currently the function signature doesn't include `symbol`. Two options:

**Option A (recommended)**: Pass symbol via `df.attrs`:

In `patterns.py` around line 850, just BEFORE calling `analyze_multi_degree()`, add:
```python
    df.attrs["symbol"] = sym
```

This requires no signature change and pandas DataFrames support `.attrs` natively.

**Option B**: Add `symbol: str = ""` parameter to `analyze_multi_degree()` and pass it from patterns.py.

Use Option A — it's simpler and non-breaking.

---

## Bug 14 — Linked correction never populated

### Problem

The API endpoint at line 902 always returns `"linked_correction": None`. But the frontend (`KLineChartOverlay.tsx` lines 786-868) has rich rendering logic that:

1. Draws the primary impulse (1-2-3-4-5)
2. Draws the linked ABC correction chained from W5 end
3. Finds and draws a "next impulse" starting from C

Without `linked_correction`, the frontend only draws the impulse plus a synthetic projection — it never shows a real detected ABC correction. This makes charts like CAD=X confusing because the projected ABC is purely algorithmic (Fibonacci-based estimates) rather than anchored to actual detected pivot points.

### Fix

**File: `.worktrees/ew-engine-v2/api/routers/patterns.py`**  
**Location: after line ~865 (after `best_corrective` is computed)**

Add logic to find the best corrective count whose start coincides with the preferred impulse's end:

```python
    # ── Link the best corrective pattern that follows the primary impulse ──
    linked_correction_data = None
    if best_impulse and best_impulse.get("waves"):
        impulse_end_idx = best_impulse["waves"][-1]["end_idx"]
        # Find corrective counts starting at or just after the impulse end
        for wc in wave_counts:
            if wc.get("type") != "corrective":
                continue
            if not wc.get("waves"):
                continue
            corr_start_idx = wc["waves"][0]["start_idx"]
            # Allow ±2 bar tolerance for pivot alignment
            if abs(corr_start_idx - impulse_end_idx) <= 2:
                linked_correction_data = wc
                break
```

Then change line ~902:
```python
        "linked_correction": None,
```
to:
```python
        "linked_correction": linked_correction_data,
```

---

## Bug 15 — Graceful degradation when all candidates are rejected

### Problem

When ALL candidates fail the proportionality gate or `validate_and_score()` filtering, the engine returns `_empty_result()` ("INSUFFICIENT DATA"). BTC-USD currently hits this case — the engine finds pivot combinations that satisfy the three Elliott rules, but they all get filtered out by the 20% min_price_span filter in `validate_and_score()` or the proportionality gate.

An EW engine should never return nothing for a chart with 500+ bars of data. There is ALWAYS a most-likely wave count, even if confidence is low.

### Fix

**File: `.worktrees/ew-engine-v2/pattern_engine/elliott_wave.py`**  
**Function: `analyze_multi_degree()` — around line 2042-2047**

Replace the current scoring and early exit:

```python
    # ── Layers 4+5: Score and rank ────────────────────────────────────────
    scored = validate_and_score(all_candidates)
    top_counts = rank_counts(scored, top_n=3)

    if not top_counts:
        return _empty_result()
```

with a graceful degradation approach:

```python
    # ── Layers 4+5: Score and rank ────────────────────────────────────────
    scored = validate_and_score(all_candidates)
    top_counts = rank_counts(scored, top_n=3)

    # ── Graceful degradation: if strict scoring rejects everything,
    # retry with relaxed proportionality gate ─────────────────────────────
    if not top_counts and all_candidates:
        # Pass 2: relax the min_price_span filter from 20% to 5%
        scored = validate_and_score(all_candidates, min_span_pct=0.05)
        top_counts = rank_counts(scored, top_n=3)
        # Mark these as low-confidence since they needed relaxation
        for c in top_counts:
            c["confidence"] = round(c.get("confidence", 0) * 0.6, 4)

    if not top_counts:
        return _empty_result()
```

This requires adding a `min_span_pct` parameter to `validate_and_score()`.

**Function: `validate_and_score()` — line ~1588**

Change the signature from:
```python
def validate_and_score(candidates: list[dict]) -> list[dict]:
```
to:
```python
def validate_and_score(candidates: list[dict], min_span_pct: float = 0.20) -> list[dict]:
```

And change the hardcoded `0.20` on line ~1613:
```python
    min_price_span = max_price_span * 0.20
```
to:
```python
    min_price_span = max_price_span * min_span_pct
```

### Additionally: Relax proportionality gate in `find_pivot_based_impulse_candidates()`

The current gate uses `top_ratio > 5.0` and `smallest/largest < 0.05`. These can be slightly too strict for volatile assets. Change both bullish and bearish sections:

**Lines ~1205-1208 (bullish) and ~1296-1299 (bearish):**

Change:
```python
                            if impulse_sizes[0] > 0:
                                top_ratio = impulse_sizes[2] / impulse_sizes[1]
                                if top_ratio > 5.0:
                                    continue  # one wave is absurdly dominant
                            if impulse_sizes[2] > 0 and impulse_sizes[0] / impulse_sizes[2] < 0.05:
                                continue  # smallest wave is noise-level
```
to:
```python
                            if impulse_sizes[0] > 0:
                                top_ratio = impulse_sizes[2] / impulse_sizes[1]
                                if top_ratio > 6.0:
                                    continue  # one wave is absurdly dominant
                            if impulse_sizes[2] > 0 and impulse_sizes[0] / impulse_sizes[2] < 0.03:
                                continue  # smallest wave is noise-level
```

This widens the gate slightly: the top-ratio increases from 5.0→6.0 and the min-ratio drops from 0.05→0.03. This allows more candidates through while still blocking truly degenerate patterns (e.g., W1=$1 with W3=$100). The `_score_proportionality()` function still penalizes disproportionate patterns in scoring — they just won't be hard-rejected.

---

## Summary of all changes

| File | Location | Change |
|------|----------|--------|
| `elliott_wave.py` | `analyze_multi_degree()` ~L1954 | Add asset-class detection + adaptive swing thresholds |
| `elliott_wave.py` | `analyze_multi_degree()` ~L1979 | Use `_primary_swing` variable |
| `elliott_wave.py` | `analyze_multi_degree()` ~L1995 | Use `_sub_primary_swing` variable |
| `elliott_wave.py` | `analyze_multi_degree()` ~L2042 | Add graceful degradation retry |
| `elliott_wave.py` | `validate_and_score()` ~L1588 | Add `min_span_pct` parameter |
| `elliott_wave.py` | `validate_and_score()` ~L1613 | Use `min_span_pct` instead of hardcoded 0.20 |
| `elliott_wave.py` | `find_pivot_based_impulse_candidates()` ~L1205 | Relax top_ratio 5.0→6.0 |
| `elliott_wave.py` | `find_pivot_based_impulse_candidates()` ~L1207 | Relax min_ratio 0.05→0.03 |
| `elliott_wave.py` | `find_pivot_based_impulse_candidates()` ~L1296 | Same relaxation (bearish) |
| `elliott_wave.py` | `find_pivot_based_impulse_candidates()` ~L1298 | Same relaxation (bearish) |
| `patterns.py` | ~L850 | Add `df.attrs["symbol"] = sym` before calling engine |
| `patterns.py` | ~L865 | Add linked_correction detection logic |
| `patterns.py` | ~L902 | Replace `None` with `linked_correction_data` |

## Files to edit

1. `.worktrees/ew-engine-v2/pattern_engine/elliott_wave.py`
2. `.worktrees/ew-engine-v2/api/routers/patterns.py`

## Testing

After applying:
1. Restart backend (`start-ew-dev.sh`)
2. Clear browser cache or add `?nocache=1` to API calls
3. Check all 5 symbols: GC=F, CL=F, EONR, CAD=X, BTC-USD
4. CAD=X should now show a clean wave structure (forex thresholds will find proper pivots)
5. BTC-USD should now show a wave pattern (crypto thresholds + graceful degradation)
6. GC=F, CL=F, EONR should remain unchanged (they use default equity/commodity thresholds)
