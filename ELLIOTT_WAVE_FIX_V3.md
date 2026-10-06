# Elliott Wave V2 Engine — Fix V3: Diagonal Mislabeling + Scoring Priority

**CRITICAL**: The backend runs from the WORKTREE:
```
File to fix (backend): .worktrees/ew-engine-v2/api/routers/patterns.py
File to fix (engine):  .worktrees/ew-engine-v2/pattern_engine/elliott_wave.py
```

## Previous Fixes Already Applied (DO NOT re-apply)

- V1 fixes (Bugs 1-5): ATH/ATL injection, linear span scoring, Rule 3 grace 2%, 20% min span, anchor bonus
- V2 fix: `find_pivot_based_impulse_candidates()` + sub-primary pivots at 7%

---

## Problem: Diagonals Rendered as "Impulse" — Hiding Rule 3 Violations

### What's happening

Charts like CL=F and CAD=X show what LOOKS like a Rule 3 violation (W4 overlapping W1) labeled "Impulse". Investigation reveals these are actually **diagonal** candidates (ENDING_DIAGONAL or LEADING_DIAGONAL) where W4/W1 overlap is the *defining* characteristic.

The root cause is in `api/routers/patterns.py` lines 581-583:

```python
_IMPULSE_TYPES = {
    "impulse", "impulse_ext_w1", "impulse_ext_w3", "impulse_ext_w5",
    "truncated_5th", "leading_diagonal", "ending_diagonal",  # ← BUG
}
```

This maps diagonals to `type: "impulse"` in the API response. The frontend then:
- Labels them "1, 2, 3, 4, 5" (should be "(1), (2), (3), (4), (5)")
- Shows "Impulse" text (should show "Leading Diagonal" or "Ending Diagonal")
- Treats them identically in the smart selection filter (`wc?.type === 'impulse'`)

### Evidence across charts

- **CL=F**: W1 end ≈ $72, W4 ≈ $55 — $17 overlap. This is a diagonal being shown as impulse.
- **CAD=X**: W1 end ≈ $1.40, W4 ≈ $1.38 — overlap. Same issue.
- **EONR**: Cramped pattern at top — probably a diagonal from the sliding window beating the correct bearish impulse.

---

## Bug 6: Separate diagonal from impulse in API type mapping

**File**: `.worktrees/ew-engine-v2/api/routers/patterns.py`

### Fix 6a: Split _IMPULSE_TYPES

**OLD** (line 581-584):
```python
_IMPULSE_TYPES = {
    "impulse", "impulse_ext_w1", "impulse_ext_w3", "impulse_ext_w5",
    "truncated_5th", "leading_diagonal", "ending_diagonal",
}
```

**NEW**:
```python
_IMPULSE_TYPES = {
    "impulse", "impulse_ext_w1", "impulse_ext_w3", "impulse_ext_w5",
    "truncated_5th",
}

_DIAGONAL_TYPES = {
    "leading_diagonal", "ending_diagonal",
}
```

### Fix 6b: Update _wt_str to return "diagonal" type

**OLD** (line 587-591):
```python
def _wt_str(wave_type) -> str:
    """WaveType → frontend type string."""
    from pattern_engine.elliott_wave import WaveType
    val = wave_type.value if hasattr(wave_type, "value") else str(wave_type)
    return "impulse" if val in _IMPULSE_TYPES else "corrective"
```

**NEW**:
```python
def _wt_str(wave_type) -> str:
    """WaveType → frontend type string."""
    from pattern_engine.elliott_wave import WaveType
    val = wave_type.value if hasattr(wave_type, "value") else str(wave_type)
    if val in _IMPULSE_TYPES:
        return "impulse"
    if val in _DIAGONAL_TYPES:
        return "diagonal"
    return "corrective"
```

### Fix 6c: Use parenthesized labels for diagonals

**OLD** (line 599-605):
```python
def _wave_labels(wave_type, n: int) -> list[str]:
    from pattern_engine.elliott_wave import WaveType
    wt = wave_type.value if hasattr(wave_type, "value") else str(wave_type)
    if wt in ("impulse", "impulse_ext_w1", "impulse_ext_w3",
              "impulse_ext_w5", "truncated_5th",
              "leading_diagonal", "ending_diagonal") and n == 5:
        return ["1", "2", "3", "4", "5"]
```

**NEW**:
```python
def _wave_labels(wave_type, n: int) -> list[str]:
    from pattern_engine.elliott_wave import WaveType
    wt = wave_type.value if hasattr(wave_type, "value") else str(wave_type)
    if wt in ("leading_diagonal", "ending_diagonal") and n == 5:
        return ["(1)", "(2)", "(3)", "(4)", "(5)"]
    if wt in ("impulse", "impulse_ext_w1", "impulse_ext_w3",
              "impulse_ext_w5", "truncated_5th") and n == 5:
        return ["1", "2", "3", "4", "5"]
```

---

## Bug 7: Diagonal confidence penalty in scoring

**File**: `.worktrees/ew-engine-v2/pattern_engine/elliott_wave.py`

Diagonals should score LOWER than impulses of similar span. Diagonals are less common structures in Elliott Wave theory and should only win when no valid impulse exists.

In `validate_and_score()`, after the confidence calculation (around line 1665), add a diagonal penalty:

```python
        # ── Diagonal penalty ─────────────────────────────────────────────
        # Diagonals are less common than impulses and should only be
        # preferred when no valid impulse candidate of similar span exists.
        # Apply a 25% penalty to prevent large diagonals from outranking
        # slightly smaller but structurally valid impulses.
        if wave_type in (WaveType.ENDING_DIAGONAL, WaveType.LEADING_DIAGONAL):
            confidence *= 0.75
```

Place this **before** the `result["confidence"] = round(...)` line.

---

## Bug 8: Frontend should distinguish diagonal from impulse rendering

**File**: `frontend/src/components/chart/KLineChartOverlay.tsx`

### Fix 8a: Include diagonals in the candidate pool

The smart selection filter at line 626 only considers `type === 'impulse'`. Change it to also consider diagonals (but rank them lower):

**OLD** (line 625-626):
```typescript
    const anchoredCandidates = (waveCounts || []).filter((wc: any) =>
      wc?.type === 'impulse' &&
```

**NEW**:
```typescript
    const anchoredCandidates = (waveCounts || []).filter((wc: any) =>
      (wc?.type === 'impulse' || wc?.type === 'diagonal') &&
```

### Fix 8b: Also update the _is5 check

**OLD** (line 641):
```typescript
    const _is5 = (wc: any) => wc?.type === 'impulse' && wc?.is_complete && wc?.waves?.length === 5
```

**NEW**:
```typescript
    const _is5 = (wc: any) => (wc?.type === 'impulse' || wc?.type === 'diagonal') && wc?.is_complete && wc?.waves?.length === 5
```

### Fix 8c: Show correct label text for diagonals

Find where the phase text "Impulse" is set (line 580):

**OLD**:
```typescript
            const phaseText = opts.isImpulsePhase ? `${arrow} Impulse` : `${arrow} Correction`
```

This needs to also check if the count is a diagonal. You'll need to pass the wave_type through to this rendering code. When wave_type contains "diagonal", display:
```
"Leading Diagonal" or "Ending Diagonal" 
```
instead of "Impulse".

### Fix 8d: best_impulse fallback

At line 853, the code picks `best_impulse` as the first count with `type === "impulse"`. Add diagonal fallback:

```typescript
    best_impulse = next((wc for wc in wave_counts if wc["type"] == "impulse"), None)
    if best_impulse is None:
        best_impulse = next((wc for wc in wave_counts if wc["type"] == "diagonal"), None)
```

---

## Bug 9: Diagonal candidate finder is too permissive

**File**: `.worktrees/ew-engine-v2/pattern_engine/elliott_wave.py`

The `find_diagonal_candidates()` function (line 1447) creates an ENDING_DIAGONAL for ANY 5-monowave sequence where W4 overlaps W1 — even with mixed/impulsive internal structures (line 1507-1511). This generates far too many diagonal candidates.

### Fix: Require at least partial corrective structure

**OLD** (line 1506-1511):
```python
        else:
            # Mixed internal structures — still generate an ENDING_DIAGONAL candidate.
            # The internal-structure score will be penalised, but W4/W1 overlap (the
            # primary diagnostic) is confirmed.  Span scoring will correctly elevate
            # this large-scale structure over smaller, structurally-perfect sub-patterns.
            wave_type = WaveType.ENDING_DIAGONAL
```

**NEW**:
```python
        else:
            # Mixed internal structures — only generate a diagonal candidate if
            # at least 3 of 5 waves have corrective/indeterminate structure.
            # This prevents impulse-like sequences from being reclassified as
            # diagonals just because W4 happens to overlap W1 slightly.
            corr_count = sum(
                1 for m in seq
                if m.internal_structure in ("corrective", "indeterminate")
            )
            if corr_count < 3:
                continue  # Too impulsive to be a diagonal
            wave_type = WaveType.ENDING_DIAGONAL
```

---

## VERIFICATION

After applying all fixes, restart backend and test:

1. **CL=F**: Should now show either a valid impulse (no W4/W1 overlap) or a diagonal with parenthesized labels "(1), (2), (3), (4), (5)" and "Ending Diagonal" text.
2. **CAD=X**: Same — either valid impulse or properly labeled diagonal.
3. **GC=F**: Should remain as-is (already a valid impulse with no overlap).
4. **EONR**: If the cramped pattern was a diagonal winning over a correct bearish impulse, the 25% diagonal penalty should let the impulse win.
5. **BTC-USD**: Check that intermediate structure appears (W2/W3/W4 should be visible).

**Key visual test**: NO chart should show "Impulse" text where W4 clearly overlaps W1. If it does overlap, the label MUST say "Diagonal".
