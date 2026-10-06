# Elliott Wave V9 Fix — Pattern Must Always Reach the Current Candle

## The #1 Problem

After V8 fixed proportionality and adaptive pivots, a systematic 20-chart test reveals that **26% of charts (5/19) have stale patterns** where the wave count ends far before the current candle, leaving the right side of the chart completely empty:

| Ticker | Gap to current | Pattern span | Total bars |
|--------|---------------|-------------|------------|
| **JD** | 275 bars (55%) | idx 79-225 | 501 |
| **KWEB** | 130 bars (26%) | idx 96-370 | 501 |
| **RBLX** | 121 bars (24%) | idx 20-379 | 501 |
| **XPEV** | 102 bars (20%) | idx 84-398 | 501 |
| **COIN** | 158 bars (20%) | idx 69-621 | 780 |

JD is the worst: the impulse covers early 2024, and everything from mid-2024 to April 2026 (275 bars, 55% of the chart) has ZERO wave structure.

The WaveBasis competitor (shown in user screenshots) ALWAYS labels every bar on the chart, from the first candle to the current one. Our engine must do the same.

## Root Causes

### Root Cause 1 — Engine has no recency awareness
`validate_and_score()` ranks candidates by span, proportionality, fibonacci, etc. — but has **zero consideration for how close the pattern extends to the current bar.** A pattern from 2 years ago that spans 60% of bars can outscore a fresh pattern spanning 40% of bars.

### Root Cause 2 — No forward continuation analysis
When the best pattern (impulse) ends mid-chart, the engine tries to find a linked correction within the existing `scored` candidates. But for JD, ALL 3 candidates end at the same bar 225 — there's literally nothing in the candidate pool for bars 225-500. The engine doesn't run a SECOND analysis on the post-pattern bars.

### Root Cause 3 — Frontend suppresses projections for stale patterns
Line 1345-1346 of `KLineChartOverlay.tsx`:
```typescript
const maxAllowedBarsAfter = Math.max(15, Math.floor(serverBars.length * 0.15))
if (barsAfterImp > maxAllowedBarsAfter) return
```
For JD (275 bars after, max allowed = 75), projections are completely suppressed. The entire right side of the chart is blank.

---

## Fix A — Recency scoring in validate_and_score()

### Problem
The scoring function doesn't know or care how close the pattern gets to the current bar. A pattern ending at bar 225 of 501 scores the same as one ending at bar 490.

### Implementation

**File: `.worktrees/ew-engine-v2/pattern_engine/elliott_wave.py`**
**Function: `validate_and_score()` — change the signature to accept total bar count:**

Replace:
```python
def validate_and_score(candidates: list[dict], min_span_pct: float = 0.20) -> list[dict]:
```

with:
```python
def validate_and_score(candidates: list[dict], min_span_pct: float = 0.20, total_bars: int = 0) -> list[dict]:
```

**Then, inside the confidence calculation block (around line ~1700, after computing `confidence` but before `result["confidence"] = ...`):**

Add a recency factor:

```python
        # ── Recency factor ───────────────────────────────────────────
        # Patterns that extend closer to the current bar are more useful
        # for trading decisions.  Penalize patterns ending far from the
        # chart's right edge.
        if total_bars > 0:
            last_wave_idx = waves[-1].end.idx
            gap_ratio = (total_bars - 1 - last_wave_idx) / total_bars
            if gap_ratio > 0.15:
                # Linear penalty: 15% gap = no penalty, 65% gap = 50% penalty
                recency_factor = max(0.50, 1.0 - (gap_ratio - 0.15) * 1.0)
                confidence *= recency_factor
```

**Update all callers of `validate_and_score()` to pass `total_bars`:**

In `analyze_multi_degree()`, the two calls at approximately lines 2165 and 2171:

```python
    scored = validate_and_score(all_candidates, total_bars=len(df))
```

and:

```python
        scored = validate_and_score(all_candidates, min_span_pct=0.05, total_bars=len(df))
```

### Why this works

For JD (501 bars):
- Old pattern ending at bar 225: gap_ratio = 0.549. Recency factor = max(0.50, 1.0 - 0.399) = 0.601. Confidence drops by ~40%.
- If ANY fresher candidate exists (even lower raw quality), it now has a better shot at being selected.

For charts where all candidates are old (like JD where all 3 end at bar 225), this alone doesn't solve the problem — we also need Fix B.

---

## Fix B — Forward continuation analysis

This is the critical fix. When the best pattern ends far from the current bar, the engine runs a SECOND analysis on the remaining bars and returns it as a `continuation` field.

### Implementation

**File: `.worktrees/ew-engine-v2/pattern_engine/elliott_wave.py`**
**Function: `analyze_multi_degree()` — after finding `preferred` and `linked_correction` (around line 2200), before the return statement:**

Add this block:

```python
    # ── Forward continuation: analyze post-pattern bars ───────────────────
    # If the best pattern (+ any linked correction) ends more than 20% of
    # bars before the chart's right edge, run a second analysis on just the
    # remaining bars.  This ensures the wave count always extends to the
    # current candle — matching WaveBasis-style full-chart coverage.
    continuation = None
    effective_end_idx = preferred["waves"][-1].end.idx
    if linked_correction and linked_correction.get("waves"):
        effective_end_idx = max(
            effective_end_idx,
            linked_correction["waves"][-1].end.idx
        )

    gap_bars = len(df) - 1 - effective_end_idx
    gap_pct = gap_bars / len(df) if len(df) > 0 else 0

    if gap_pct > 0.15 and gap_bars > 20:
        # Slice dataframe from the pattern's end to the chart's end,
        # with a small overlap (10 bars) for pivot context.
        cont_start = max(0, effective_end_idx - 10)
        df_cont = df.iloc[cont_start:].copy().reset_index(drop=True)

        if len(df_cont) >= 30:  # need meaningful data
            # Preserve symbol for asset-class detection
            df_cont.attrs["symbol"] = df.attrs.get("symbol", "")

            cont_result = analyze_multi_degree(
                df_cont,
                primary_atr_mult=primary_atr_mult,
                minor_atr_mult=minor_atr_mult,
                is_commodity=is_commodity,
            )

            if cont_result.get("preferred"):
                # Re-map indices back to the original dataframe
                cont_pref = cont_result["preferred"]
                offset = cont_start
                for w in cont_pref.get("waves", []):
                    w.start = Pivot(
                        idx=w.start.idx + offset,
                        price=w.start.price,
                        is_high=w.start.is_high,
                        strength=w.start.strength,
                    )
                    w.end = Pivot(
                        idx=w.end.idx + offset,
                        price=w.end.price,
                        is_high=w.end.is_high,
                        strength=w.end.strength,
                    )

                continuation = cont_pref

                # Also remap linked_correction from continuation if present
                cont_lc = cont_result.get("linked_correction")
                if cont_lc and cont_lc.get("waves"):
                    for w in cont_lc["waves"]:
                        w.start = Pivot(
                            idx=w.start.idx + offset,
                            price=w.start.price,
                            is_high=w.start.is_high,
                            strength=w.start.strength,
                        )
                        w.end = Pivot(
                            idx=w.end.idx + offset,
                            price=w.end.price,
                            is_high=w.end.is_high,
                            strength=w.end.strength,
                        )
                    continuation["_linked_correction"] = cont_lc
```

**Then update the return dict** (around line 2202) to include continuation:

Replace:
```python
    return {
        # Backward-compatible fields
        "summary":    _format_summary(preferred),
        ...
        "linked_correction": linked_correction,
        ...
    }
```

with:
```python
    return {
        # Backward-compatible fields
        "summary":    _format_summary(preferred),
        "confidence": round(preferred["confidence"] * 100, 1),
        "phase":      position["phase"],
        "action":     position["action"],
        "waves":      waves_out,
        "projections": projections,
        # New v2 fields
        "counts":           top_counts,
        "preferred":        preferred,
        "linked_correction": linked_correction,
        "continuation":     continuation,      # NEW: wave structure for post-pattern bars
        "degree":           "intermediate",
        "trend":            "up" if preferred["direction"] == WaveDirection.UP else "down",
    }
```

### Recursion guard

The continuation analysis calls `analyze_multi_degree()` recursively. To prevent infinite recursion, **add a `_depth` parameter**:

Change the function signature:
```python
def analyze_multi_degree(
    df: pd.DataFrame,
    primary_atr_mult: float = 1.5,
    minor_atr_mult: float = 0.4,
    is_commodity: bool = False,
    _depth: int = 0,           # NEW: recursion guard
) -> dict[str, Any]:
```

At the top of the function, add:
```python
    if _depth > 1:
        # Max 2 levels: primary pattern + one continuation
        return _empty_result()
```

And in the continuation block, pass `_depth + 1`:
```python
            cont_result = analyze_multi_degree(
                df_cont,
                primary_atr_mult=primary_atr_mult,
                minor_atr_mult=minor_atr_mult,
                is_commodity=is_commodity,
                _depth=_depth + 1,
            )
```

### Why this works

For JD: the primary impulse ends at bar 225. After the impulse, bars 225-500 contain a corrective/consolidation phase. The continuation analysis runs on bars 215-500 (with 10-bar overlap) and finds the wave structure in that region — possibly a corrective A-B-C or a new impulse pattern. This gets returned as `continuation` and the frontend can draw it.

---

## Fix C — API endpoint: serialize continuation

**File: `.worktrees/ew-engine-v2/api/routers/patterns.py`**
**After serializing `linked_correction_data` (around line ~887), add serialization for continuation:**

```python
    # ── Serialize continuation (post-pattern wave structure) ──────────
    continuation_data = None
    raw_continuation = result.get("continuation")
    if raw_continuation is not None:
        continuation_data = _ser_v2_count(raw_continuation)
        # Also serialize continuation's linked correction if present
        cont_lc = raw_continuation.get("_linked_correction")
        if cont_lc:
            continuation_data["linked_correction"] = _ser_v2_count(cont_lc)
```

**Then add `continuation_data` to the response dict** (around line 914):

After `"linked_correction": linked_correction_data,` add:
```python
        "continuation":     continuation_data,
```

---

## Fix D — Frontend: draw continuation patterns

**File: `frontend/src/components/chart/KLineChartOverlay.tsx`**

### D1: Extract continuation data from API response

In the data parsing section (where `primaryCount`, `linkedCorrection`, etc. are extracted from the API response), add:

```typescript
const continuation = data?.continuation || null
```

### D2: Draw continuation after linked correction (or after impulse if no correction)

After the existing linked correction drawing block (around line ~870), and after the continuation/next-impulse code, add:

```typescript
      // ── CONTINUATION: draw the post-pattern wave structure ──────────────
      // When the primary pattern ends far from the current bar, the backend
      // runs a second analysis on the remaining bars and returns it here.
      if (continuation && continuation.waves && continuation.waves.length > 0) {
        // Bridge line from the last drawn wave to continuation start
        const lastDrawn = linkedCorrection?.waves?.length
          ? linkedCorrection.waves[linkedCorrection.waves.length - 1]
          : primaryCount.waves[primaryCount.waves.length - 1]
        const contFirst = continuation.waves[0]

        if (lastDrawn && contFirst && contFirst.start_idx > lastDrawn.end_idx) {
          add({
            name: 'patternTrendline',
            points: [
              { timestamp: snap(lastDrawn.end_idx), value: lastDrawn.end_price },
              { timestamp: snap(contFirst.start_idx), value: contFirst.start_price },
            ],
            extendData: { color: '#888888', lineWidth: 1, dash: [6, 4] },
          })
        }

        // Draw continuation waves with slightly muted styling
        drawWaveCount(continuation, {
          lineColor: continuation.type === 'impulse' ? IMPULSE_CLR : IMPULSE_CLR,
          circleColor: IMPULSE_CIRCLE,
          circleText: '#ffffff',
          correctiveLineColor: IMPULSE_CLR,
          annotColor: IMPULSE_CLR,
          lineWidth: MOTIVE_LINE_W,
          isImpulsePhase: continuation.type === 'impulse',
        })

        // If continuation has its own linked correction, draw that too
        if (continuation.linked_correction?.waves?.length > 0) {
          drawWaveCount(continuation.linked_correction, {
            lineColor: IMPULSE_CLR,
            circleColor: IMPULSE_CIRCLE,
            circleText: '#ffffff',
            correctiveLineColor: IMPULSE_CLR,
            annotColor: IMPULSE_CLR,
            lineWidth: CORR_LINE_W,
            isImpulsePhase: false,
          })
        }
      }
```

### D3: Relax projection suppression for stale patterns

**Replace lines 1344-1346:**

```typescript
        // Only project if impulse ended in the last 15% of visible bars — otherwise it's stale
        const maxAllowedBarsAfter = Math.max(15, Math.floor(serverBars.length * 0.15))
        if (barsAfterImp > maxAllowedBarsAfter) return
```

with:

```typescript
        // Skip projections only if a backend continuation pattern exists
        // (the continuation already covers the post-pattern bars).
        // If no continuation, allow projections even for stale patterns —
        // the user should always see wave structure to the current candle.
        if (continuation && continuation.waves && continuation.waves.length > 0) {
          // Continuation handles the post-pattern region — skip synthetic projection
        } else {
          // No continuation available — draw projection regardless of staleness
          // (previously suppressed stale projections, leaving the chart empty)
```

Then move the closing `}` to after the entire projection block (the `else` wraps the existing projection code).

**Alternative simpler approach**: Just remove the suppression entirely:

Replace:
```typescript
        const maxAllowedBarsAfter = Math.max(15, Math.floor(serverBars.length * 0.15))
        if (barsAfterImp > maxAllowedBarsAfter) return
```

with:

```typescript
        // Skip projections if a backend continuation pattern covers this region
        if (continuation && continuation.waves && continuation.waves.length > 0) return
```

This way: if the backend provides a continuation, use that (it's based on real data). If not, always project (never leave the chart empty).

---

## Fix E — Include continuation in auto-zoom calculation

**In the auto-zoom code (around line 1819-1950)**, ensure the continuation waves are included in the bounding box calculation so the chart zooms to show everything:

Where the code collects wave pivots for the bounding box, also include continuation pivots:

```typescript
// Also include continuation pattern in bounding box
if (continuation?.waves) {
  for (const w of continuation.waves) {
    allPivots.push({ ts: snap(w.start_idx), price: w.start_price })
    allPivots.push({ ts: snap(w.end_idx), price: w.end_price })
  }
}
```

---

## Summary of all changes

| Fix | File | Change |
|-----|------|--------|
| A | `elliott_wave.py` | Add `total_bars` param to `validate_and_score()`, apply recency penalty for gap > 15% |
| B | `elliott_wave.py` | Add forward continuation analysis in `analyze_multi_degree()` when gap > 15% + recursion guard |
| C | `patterns.py` | Serialize `continuation` field from engine result |
| D1 | `KLineChartOverlay.tsx` | Extract continuation from API response |
| D2 | `KLineChartOverlay.tsx` | Draw continuation waves with bridge line |
| D3 | `KLineChartOverlay.tsx` | Replace stale projection suppression: skip only when continuation exists |
| E | `KLineChartOverlay.tsx` | Include continuation in auto-zoom bounding box |

## Files to edit

1. `.worktrees/ew-engine-v2/pattern_engine/elliott_wave.py` (Fixes A, B)
2. `.worktrees/ew-engine-v2/api/routers/patterns.py` (Fix C)
3. `frontend/src/components/chart/KLineChartOverlay.tsx` (Fixes D, E)

## Expected results

**JD** — Primary impulse (bars 79-225) + continuation analysis on bars 215-500 finds the corrective/consolidation structure → wave count extends to current candle

**KWEB** — Primary impulse (bars 96-370) + continuation covers 370-500 → full coverage

**RBLX, XPEV, COIN** — Same pattern: primary + continuation = full chart coverage

**Charts that already reach current** (SNAP, LCID, NIO, etc.) — gap < 15%, continuation not triggered, no change

## Testing

1. Restart backend (`start-ew-dev.sh`) and rebuild frontend
2. Test JD first — should now show waves covering the full chart
3. Test BABA — should maintain existing pattern + potentially get continuation
4. Test KWEB — the impulse + continuation should cover up to current bar
5. Spot-check MSFT, AMZN — should be unchanged (no gap issue)
6. Verify no regressions: patterns that already reach current should be identical
