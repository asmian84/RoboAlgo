# Elliott Wave V7 Fix — Shallow W4, Tighter Proportionality, Orphan Lines & ABC Alignment

## Context

Post-V5/V6 a 20-chart batch test + user screenshot review shows:

- **16/20 charts clean** (MSFT, GOOGL, AMZN, TSLA, NVDA, JPM, BAC, XOM, CVX, NFLX, DIS, AMD, GC=F, CL=F, BZ=F, BTC-USD)
- **PLTR** — catastrophic disproportion: W1 $14→$18, W3 $14→$22, W5 $18→$208 (25.7× ratio). The proportionality gate exists at `> 6.0` but the candidate still passes because `find_impulse_candidates()` computes `price_change` as absolute dollar change from MonoWave objects that may have different price ranges than the raw pivot prices. The gate needs tightening plus a scoring penalty.
- **META / INTC** — shallow W4: only 10% retracement of W3. In real Elliott waves, W4 must retrace at least ~23.6% of W3. These produce misleading patterns where a brief pause is labeled as W4.
- **CRWV** — pattern covers only 16% of bar span. The engine found a sub-scale pattern rather than the dominant structure.
- **GC=F** — orphan rendering line near W5 peak area (circular/loop artifact from bridge line drawing when linked correction start is clamped to impulse end).

This V7 fix addresses four bugs across engine + frontend.

---

## Bug 18 — Minimum W4 retracement depth (shallow W4 filter)

### Problem

The engine validates the three absolute EW rules (W2 < 100% of W1, W3 not shortest, W4 doesn't overlap W1) but has NO minimum depth check for W4. This allows candidates where W4 barely retraces W3 (e.g., 10%), which are structurally not real wave 4 pullbacks — they're micro-consolidations within a continuing W3.

In EW theory, the minimum expected W4 retracement of W3 is 23.6% (the smallest Fibonacci level). Anything less is virtually never labeled W4 by professional analysts.

### Fix

**File: `.worktrees/ew-engine-v2/pattern_engine/elliott_wave.py`**  
**Function: `find_impulse_candidates()` — after the proportionality gate at line ~1077, before `candidates.append()`**

Add this check after the existing proportionality gate block (after line 1077):

```python
        # ── Minimum W4 retracement gate ──────────────────────────────
        # W4 should retrace at least ~20% of W3. Anything shallower is
        # a micro-consolidation within W3, not a real corrective wave.
        if w3.price_change > 0:
            w4_retrace_ratio = w4.price_change / w3.price_change
            if w4_retrace_ratio < 0.20:
                continue  # W4 too shallow — not a real pullback
```

**Also add the same check in `find_pivot_based_impulse_candidates()`:**

**Bullish section — after the proportionality gate at line ~1208, before `candidates.append()` at line ~1210:**

```python
                            # ── Minimum W4 retracement gate ──────────────
                            if w3_move > 0:
                                w4_retrace = w4_move / w3_move
                                if w4_retrace < 0.20:
                                    continue  # W4 too shallow
```

**Bearish section — after the proportionality gate at line ~1299 (the bearish equivalent), before the bearish `candidates.append()`:**

```python
                            # ── Minimum W4 retracement gate ──────────────
                            if w3_move > 0:
                                w4_retrace = w4_move / w3_move
                                if w4_retrace < 0.20:
                                    continue  # W4 too shallow
```

### Finding `w4_move` in the pivot-based finder

In the bullish section of `find_pivot_based_impulse_candidates()`, the w4 move isn't currently computed as a named variable. You need to add it. Around line ~1170 (where `w1_move`, `w3_move`, `w5_move` are computed), also add:

```python
                            w4_move = h3.price - l4.price   # W4 = high of W3 to low of W4
```

In the bearish section around line ~1270:

```python
                            w4_move = l4.price - l3.price   # bearish W4 = W3 low to W4 high
```

Wait — let me be more precise. In the bullish section, the pivots are `L0-H1-L2-H3-L4-H5`. So:
- W3 move = `h3.price - l2.price` (already computed)
- W4 move = `h3.price - l4.price` (W4 retraces from H3 down to L4)

And in the bearish section (`H0-L1-H2-L3-H4-L5`):
- W3 move = `h2.price - l3.price` (already computed)
- W4 move = `h4.price - l3.price` (W4 retraces from L3 up to H4)

### Why this works

- A W4 that retraces < 20% of W3 is almost certainly noise or a brief consolidation
- The 0.20 threshold is intentionally just below the 0.236 Fibonacci level to allow some tolerance
- This filter catches META and INTC where W4 was only 10% of W3
- Real W4 corrections are at minimum 23.6% (zigzag minimum) up to 50%+ (flats/triangles)

---

## Bug 19 — Tighten proportionality gate + add soft scoring penalty

### Problem

The proportionality hard gate at `top_ratio > 6.0` is too loose. PLTR produced a 25.7× ratio candidate that may have bypassed the gate through a combination of:
1. The `price_change` values on MonoWaves being slightly different from the raw pivot prices the API test computed
2. The graceful degradation retry (which relaxes `min_span_pct` but does NOT relax the proportionality gate — this is correct)

Even with the gate, a 5.9× ratio still passes and produces questionable patterns. Additionally, `_score_proportionality()` assigns 0.3 score for ratios up to 5.0×, which still yields competitive confidence scores when combined with large span scores.

### Fix A — Tighten the hard gate

**File: `.worktrees/ew-engine-v2/pattern_engine/elliott_wave.py`**

Change the proportionality gate threshold in ALL THREE locations:

**1. `find_impulse_candidates()` — line ~1074:**
```python
            if _imp_sizes[2] / _imp_sizes[1] > 6.0:
```
→
```python
            if _imp_sizes[2] / _imp_sizes[1] > 5.0:
```

**2. `find_pivot_based_impulse_candidates()` bullish — line ~1205:**
```python
                                if top_ratio > 6.0:
```
→
```python
                                if top_ratio > 5.0:
```

**3. `find_pivot_based_impulse_candidates()` bearish — line ~1296 (find the corresponding line):**
```python
                                if top_ratio > 6.0:
```
→
```python
                                if top_ratio > 5.0:
```

### Fix B — Steeper scoring penalty for disproportionate patterns

**Function: `_score_proportionality()` — lines 1850-1860:**

Replace:
```python
    price_score = 0.1
    if sizes[1] > 0:
        price_ratio = sizes[2] / sizes[1]
        if price_ratio <= 2.0:
            price_score = 1.0    # ideal: ~golden ratio extension
        elif price_ratio <= 3.0:
            price_score = 0.7    # acceptable: strong extended wave
        elif price_ratio <= 5.0:
            price_score = 0.3    # poor: questionable structure
        else:
            price_score = 0.05   # near-zero: absurd ratio
```

with:

```python
    price_score = 0.05
    if sizes[1] > 0:
        price_ratio = sizes[2] / sizes[1]
        if price_ratio <= 1.618:
            price_score = 1.0    # ideal: golden ratio extension
        elif price_ratio <= 2.618:
            price_score = 0.8    # good: standard Fibonacci
        elif price_ratio <= 3.5:
            price_score = 0.4    # acceptable: strong extended wave
        elif price_ratio <= 5.0:
            price_score = 0.15   # poor: questionable structure
        else:
            price_score = 0.02   # near-zero: absurd ratio (should be gate-blocked)
```

### Why this works

- Tighter hard gate (5.0 vs 6.0) catches borderline cases
- Fibonacci-aligned scoring tiers reward textbook proportionality (≤ 1.618×) with full score
- Patterns with 3.5–5.0× ratio now score very low (0.15), making them lose to any better candidate
- Combined with the shallow W4 gate, this dramatically reduces degenerate patterns

---

## Bug 20 — Minimum bar span filter for small-scale patterns

### Problem

CRWV's pattern covers only 16% of the chart's bar span. The engine already filters by price span (`min_span_pct`) in `validate_and_score()`, but has NO bar span filter. A pattern that only covers 16% of bars is a sub-wave, not the dominant structure.

### Fix

**File: `.worktrees/ew-engine-v2/pattern_engine/elliott_wave.py`**  
**Function: `validate_and_score()` — after the price span filter at line ~1617**

After the existing `min_price_span` filter block:

```python
    min_price_span = max_price_span * min_span_pct
    candidates = [
        c for c in candidates
        if abs(c["waves"][-1].end.price - c["waves"][0].start.price) >= min_price_span
    ]
    if not candidates:
        return []
```

Add a bar span filter:

```python
    # Also filter by bar span — discard candidates spanning < 20% of the
    # largest bar span.  A pattern covering only 16% of bars is sub-scale.
    min_bar_span = max_bar_span * min_span_pct
    candidates = [
        c for c in candidates
        if (c["waves"][-1].end.idx - c["waves"][0].start.idx) >= min_bar_span
    ]
    if not candidates:
        return []
```

This reuses the same `min_span_pct` parameter so the graceful degradation retry (which relaxes to 0.05) also relaxes the bar span filter.

### Why this works

- `max_bar_span` is already computed at line 1604
- Using the same `min_span_pct` ensures consistency: if a pattern needs to span ≥ 20% of the largest price range, it also needs to span ≥ 20% of the largest bar range
- CRWV's 16% bar span pattern would be filtered out in the strict pass (< 20%), but might survive the relaxed pass at 5% — which is fine since it would get a low span_score and lose to any better candidate

---

## Bug 21 — Orphan bridge line when linked correction overlaps impulse end

### Problem

On GC=F the user reported a circular/loop line near the W5 peak. This happens when:

1. The backend's linked correction has `start_idx` slightly before the impulse's `end_idx` (pivot imprecision)
2. The frontend CLAMP code (lines 798-809) correctly forces `corrFirstWave.start_idx = impulseLastWave.end_idx`
3. But then the bridge line at lines 811-818 draws from `impulseLastWave.end_idx` → `corrFirstWave.start_idx`, which are now the SAME point (zero-length segment)
4. When both `start_price` and `end_price` also get clamped to the same value, klinecharts may render this as a dot or artifact

Additionally, if the clamped correction's first wave now has `start_idx == end_idx` (zero-length wave), the subsequent wave drawing creates degenerate geometry.

### Fix

**File: `frontend/src/components/chart/KLineChartOverlay.tsx`**  
**Location: lines 811-818 (the bridge line from impulse end to correction start)**

Replace:

```typescript
        add({
          name: 'patternTrendline',
          points: [
            { timestamp: snap(impulseLastWave.end_idx), value: impulseLastWave.end_price },
            { timestamp: snap(corrFirstWave.start_idx), value: corrFirstWave.start_price },
          ],
          extendData: { color: IMPULSE_CLR, lineWidth: MOTIVE_LINE_W, dash: [] },
        })
```

with:

```typescript
        // Only draw bridge line if there's an actual gap between impulse end
        // and correction start.  After clamping, they may be coincident — drawing
        // a zero-length segment creates rendering artifacts (orphan dots/loops).
        const bridgeGapBars = corrFirstWave.start_idx - impulseLastWave.end_idx
        if (bridgeGapBars >= 1) {
          add({
            name: 'patternTrendline',
            points: [
              { timestamp: snap(impulseLastWave.end_idx), value: impulseLastWave.end_price },
              { timestamp: snap(corrFirstWave.start_idx), value: corrFirstWave.start_price },
            ],
            extendData: { color: IMPULSE_CLR, lineWidth: MOTIVE_LINE_W, dash: [] },
          })
        }
```

**Also: skip drawing zero-length correction waves after clamping.**

After the CLAMP block (lines 798-809), add a filter to remove degenerate waves:

```typescript
        // Remove any zero-length waves created by clamping (start_idx == end_idx).
        // These produce degenerate line segments that look like orphan artifacts.
        linkedCorrection.waves = linkedCorrection.waves.filter(
          (w: any) => w.end_idx > w.start_idx
        )
        // If all correction waves were removed, skip drawing the correction entirely
        if (linkedCorrection.waves.length === 0) {
          // Fall through to the projection code which will draw a synthetic ABC
        } else {
```

Wait — this changes the control flow significantly. A simpler approach:

After the CLAMP block at line 809, add:

```typescript
        // Skip degenerate correction: if after clamping, the first wave has
        // zero or negative length, the correction is entirely within the
        // impulse and shouldn't be drawn.
        const corrFirstLen = corrFirstWave.end_idx - corrFirstWave.start_idx
        if (corrFirstLen < 1) {
          // Don't draw this correction — let the projection code handle it
        } else {
```

Then wrap the entire correction drawing block (lines 811-870 approximately) in this `else` clause and close it with `}` at the right place.

Actually, this is getting complex. Let me simplify.

### Simpler Fix

**Replace lines 798-818** (the CLAMP block + bridge line) with:

```typescript
        // ── CLAMP: correction waves cannot start before W5 ends ───────────
        if (corrFirstWave.start_idx < impulseLastWave.end_idx) {
          corrFirstWave.start_idx   = impulseLastWave.end_idx
          corrFirstWave.start_price = impulseLastWave.end_price
        }
        for (let wi = 1; wi < linkedCorrection.waves.length; wi++) {
          const prev = linkedCorrection.waves[wi - 1]
          const cur  = linkedCorrection.waves[wi]
          if (cur.start_idx < prev.end_idx) {
            cur.start_idx   = prev.end_idx
            cur.start_price = prev.end_price
          }
        }

        // Filter out zero-length waves created by clamping
        linkedCorrection.waves = linkedCorrection.waves.filter(
          (w: any) => w.end_idx > w.start_idx
        )

        // Only draw if correction still has valid waves after filtering
        if (linkedCorrection.waves.length > 0) {
          // Bridge line: only if there's an actual gap (≥1 bar)
          const corrStart = linkedCorrection.waves[0]
          const bridgeGap = corrStart.start_idx - impulseLastWave.end_idx
          if (bridgeGap >= 1) {
            add({
              name: 'patternTrendline',
              points: [
                { timestamp: snap(impulseLastWave.end_idx), value: impulseLastWave.end_price },
                { timestamp: snap(corrStart.start_idx), value: corrStart.start_price },
              ],
              extendData: { color: IMPULSE_CLR, lineWidth: MOTIVE_LINE_W, dash: [] },
            })
          }
```

Then the existing `drawWaveCount(linkedCorrection, ...)` at line 820 and the continuation logic continue as before. The closing `}` for the `if (linkedCorrection.waves.length > 0)` wraps around the drawWaveCount call through the end of the continuation logic (up to the closing `}` that matches the `if (linkedCorrection && linkedCorrection.waves ...` block).

---

## Bug 22 — Backend linked correction: enforce start ≥ impulse end

### Problem

The ±2 bar tolerance in the backend linked correction detection (patterns.py line 884) allows corrections that start BEFORE the impulse ends. This is structurally invalid — the correction must begin at or after W5 completes.

### Fix

**File: `.worktrees/ew-engine-v2/api/routers/patterns.py`**  
**Line ~884:**

Replace:
```python
                if abs(wc["waves"][0]["start_idx"] - impulse_end_idx) <= 2:
```

with:
```python
                corr_start = wc["waves"][0]["start_idx"]
                # Correction must start AT or AFTER the impulse end (not before).
                # Allow up to +3 bars after (for pivot detection misalignment).
                if 0 <= (corr_start - impulse_end_idx) <= 3:
```

### Why this works

- The original `abs(...) <= 2` allowed corrections starting 2 bars BEFORE the impulse ended, which is structurally invalid (the correction can't begin until the impulse finishes)
- The new check `0 <= (corr_start - impulse_end_idx) <= 3` only matches corrections that start at or just after the impulse end
- Increased tolerance to +3 bars (from +2) since we removed the negative direction

---

## Summary of all changes

| Bug | File | Location | Change |
|-----|------|----------|--------|
| 18 | `elliott_wave.py` | `find_impulse_candidates()` ~L1078 | Add W4 retracement ≥ 20% gate |
| 18 | `elliott_wave.py` | `find_pivot_based_impulse_candidates()` bullish ~L1209 | Add W4 retracement ≥ 20% gate + compute `w4_move` |
| 18 | `elliott_wave.py` | `find_pivot_based_impulse_candidates()` bearish ~L1300 | Same W4 gate for bearish |
| 19 | `elliott_wave.py` | `find_impulse_candidates()` ~L1074 | Tighten top_ratio 6.0 → 5.0 |
| 19 | `elliott_wave.py` | `find_pivot_based_impulse_candidates()` bullish ~L1205 | Tighten top_ratio 6.0 → 5.0 |
| 19 | `elliott_wave.py` | `find_pivot_based_impulse_candidates()` bearish ~L1296 | Tighten top_ratio 6.0 → 5.0 |
| 19 | `elliott_wave.py` | `_score_proportionality()` ~L1850 | Fibonacci-aligned scoring tiers |
| 20 | `elliott_wave.py` | `validate_and_score()` ~L1618 | Add minimum bar span filter |
| 21 | `KLineChartOverlay.tsx` | ~L798-818 | Filter zero-length waves, skip zero-gap bridge lines |
| 22 | `patterns.py` | ~L884 | Enforce correction starts ≥ impulse end (no negative tolerance) |

## Files to edit

1. `.worktrees/ew-engine-v2/pattern_engine/elliott_wave.py` (Bugs 18, 19, 20)
2. `.worktrees/ew-engine-v2/api/routers/patterns.py` (Bug 22)
3. `frontend/src/components/chart/KLineChartOverlay.tsx` (Bug 21)

## Testing

After applying:

1. Restart backend (`start-ew-dev.sh`)
2. Clear browser cache or hard-refresh
3. Test the previously-broken symbols:
   - **PLTR**: Should now show a properly proportional wave structure (not the degenerate $14→$208 W5)
   - **META**: Should either show a proper W4 with ≥ 20% retracement of W3, or shift to a different (better) candidate
   - **INTC**: Same as META — shallow W4 patterns should be rejected
   - **CRWV**: Should show a pattern covering more of the chart's bar span
   - **GC=F**: No orphan/loop line near W5 — the bridge line artifact should be gone
4. Verify no regressions on clean symbols: MSFT, GOOGL, AMZN, TSLA, NVDA (spot-check 3-4)
5. Verify BTC-USD and CAD=X still produce valid patterns (asset-class thresholds unchanged)
