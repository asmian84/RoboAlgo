# Elliott Wave V8 Fix — Full-Chart Pivot Coverage, Corrective Proportionality, Wave Dominance Gate

## Context

After V7 killed the degenerate PLTR impulse (25.7× W5), the engine now falls back to equally degenerate **corrective** and **diagonal** patterns:

- **PLTR best count**: Zigzag A($14→$22), B($22→$18), C($18→$208) — wave C is 23.75× wave A. The entire $18→$208 rally is one straight line.
- **PLTR second count**: Leading diagonal with W5($18→$208) = 24× W1($14→$18). Same single-line problem.

The user's complaint: *"we cannot have such instances where 1 to 3 are stuck in small moves while you cover the entire chart with one wave"*

Also for CRWV: *"i dont like the wave C - that one straight line all the way, its not correct"*

## Root Cause Analysis

**Three distinct bugs** produce the degenerate single-wave patterns:

### Bug 23 — `max_pivots=20` truncates pivots before reaching the chart's major turns

`detect_major_pivots()` stops after finding 20 pivots chronologically. For volatile stocks like PLTR with many 15%-swing reversals in the early price history ($5-$45 IPO spike, $45→$14 crash, then choppy recovery), all 20 pivots get consumed by the first ~60% of the chart. **The pivots at $85 (H3), $58 (L4), and $208 (H5) are never included** because the cap is reached too early.

Result: the pivot-based finder literally cannot see the correct full-scale impulse pattern `L0=$14, H1=$22, L2=$18, H3=$85, L4=$58, H5=$208`. This pattern passes ALL rules (W4 retrace 40%, proportionality 2.24×, W3 not shortest), but the pivots aren't in the list.

### Bug 24 — Zigzag and flat finders have NO proportionality gate

`find_zigzag_candidates()` and `find_flat_candidates()` accept any A-B-C where directions alternate and internal structures match. There is NO check that wave C isn't absurdly larger than wave A. So PLTR's zigzag with C = 23.75× A passes unchallenged.

In real Elliott Wave theory, zigzag C typically equals A (1.0×) or extends to 1.618× A. A ratio of 23.75× is structurally impossible.

### Bug 25 — Diagonal finder has NO proportionality gate

`find_diagonal_candidates()` checks direction alternation, W4/W1 overlap, and internal structure — but has NO proportionality check. So the leading diagonal with W5 = 24× W1 passes.

---

## Fix A — Adaptive chart-wide pivot scale (Bug 23)

### Problem
`detect_major_pivots(df, min_swing_pct=0.15, max_pivots=20)` uses a FIXED swing threshold. For charts with many small reversals in early history, 20 pivots are consumed before the chart's major structural turns are seen.

### Solution
Add a **third pivot scale** in `analyze_multi_degree()` that adaptively increases the swing threshold until ≤ 20 pivots span the ENTIRE chart. This guarantees the pivot-based finder sees ALL major turns regardless of how volatile the early price history is.

### Implementation

**File: `.worktrees/ew-engine-v2/pattern_engine/elliott_wave.py`**
**Function: `analyze_multi_degree()` — after line ~2053 (after `sub_primary_pivots = detect_major_pivots(...)`) and before line ~2055 (`if len(monowaves) < 3`)**

Add this block:

```python
    # ── Adaptive chart-wide pivots ─────────────────────────────────────────
    # The primary and sub-primary pivot scales use a FIXED swing threshold,
    # which can exhaust the max_pivots budget before reaching the chart's
    # later (and often larger) structural turns.  This adaptive scale finds
    # the minimum swing% that gives ≤ 20 pivots over the FULL chart,
    # guaranteeing that the pivot-based finder can see ALL major turns.
    #
    # For PLTR: 15% swing yields 40+ pivots → first 20 all consumed by
    # early history → the $85/$58/$208 pivots are invisible.  Adaptive
    # scale raises swing to ~30-40% → sees $14, $22, $18, $85, $58, $208.

    # Check if primary pivots are truncated (hit the cap early)
    _chart_wide_pivots = primary_pivots  # default: use existing primary pivots
    if len(primary_pivots) >= 20:
        # Primary pivots hit the cap — need a wider swing threshold.
        # Binary search for the minimum swing% that gives ≤ 20 pivots.
        _cw_lo = _primary_swing
        _cw_hi = 0.60  # 60% swing = very coarse; any chart has ≤ 20 at this level
        for _ in range(10):  # 10 bisection steps → <0.1% precision
            _cw_mid = (_cw_lo + _cw_hi) / 2.0
            _cw_test = detect_major_pivots(df, min_swing_pct=_cw_mid, max_pivots=22)
            if len(_cw_test) > 20:
                _cw_lo = _cw_mid   # too many → increase swing threshold
            else:
                _cw_hi = _cw_mid   # good or too few → decrease threshold
        _chart_wide_pivots = detect_major_pivots(df, min_swing_pct=_cw_hi, max_pivots=20)
```

Then **modify the pivot-based finder loop** at line ~2074:

Replace:
```python
    for pivot_set in [primary_pivots, sub_primary_pivots]:
```

with:
```python
    for pivot_set in [primary_pivots, sub_primary_pivots, _chart_wide_pivots]:
```

**Note:** `_chart_wide_pivots` may be the same object as `primary_pivots` when the cap wasn't hit, in which case the deduplication at line ~2084 will eliminate duplicate candidates. No extra cost.

### Why this works

For PLTR with 780 daily bars:
- 15% swing → ~40+ pivots (cap hit at 20, misses $85/$58/$208)
- Adaptive raises to ~35% swing → ~12-15 pivots: $5(L), $45(H), $14(L), $22(H), $18(L), $85(H), $58(L), $208(H), plus a few more
- The pivot-based finder now tries L0=$14, H1=$22, L2=$18, H3=$85, L4=$58, H5=$208
- This forms: W1=$8, W3=$67, W5=$150. Proportionality: 150/67 = 2.24× ✓. W4 retrace 40% ✓. All rules pass.
- Scored with high confidence (large span + chart extremes + good proportionality)

For charts that DON'T have truncated pivots (16/20 in our test), `_chart_wide_pivots == primary_pivots` — zero overhead.

---

## Fix B — Proportionality gate for zigzag and flat finders (Bug 24)

### Problem
`find_zigzag_candidates()` and `find_flat_candidates()` have no proportionality check, allowing wave C to be 20×+ wave A.

### Implementation

**File: `.worktrees/ew-engine-v2/pattern_engine/elliott_wave.py`**

#### Zigzag — `find_zigzag_candidates()` at line ~1367 (just before `candidates.append()`):

Add after the "B must NOT retrace all of A" check:

```python
        # ── Proportionality gate: C must not dwarf A ──────────────────
        # In valid zigzags, C is typically 0.618× to 1.618× of A.
        # Ratios above 5.0× produce degenerate patterns where one wave
        # covers the entire chart as a single straight line.
        if a.price_change > 0:
            ca_ratio = c.price_change / a.price_change
            if ca_ratio > 5.0:
                continue  # C absurdly larger than A
            # Also reject if A is absurdly larger than C
            ac_ratio = a.price_change / c.price_change if c.price_change > 0 else 0
            if ac_ratio > 5.0:
                continue  # A absurdly larger than C
```

#### Flat — `find_flat_candidates()` at the equivalent location (just before `candidates.append()`, after the flat subtype classification):

Add the same gate:

```python
        # ── Proportionality gate: C must not dwarf A ──────────────────
        if a.price_change > 0:
            ca_ratio = c.price_change / a.price_change
            if ca_ratio > 5.0:
                continue  # C absurdly larger than A
            ac_ratio = a.price_change / c.price_change if c.price_change > 0 else 0
            if ac_ratio > 5.0:
                continue
```

### Why this works

PLTR's zigzag: C/A = $190/$8 = 23.75 > 5.0 → **REJECTED**

Normal zigzags where C ≈ 1.618× A → 1.618 < 5.0 → **KEPT**

The 5.0× threshold is intentionally generous to allow extended zigzags (C up to 3.236× A is documented in some EW texts) while rejecting the absurd 20×+ cases.

---

## Fix C — Proportionality gate for diagonal finder (Bug 25)

### Problem
`find_diagonal_candidates()` has no proportionality check. PLTR's leading diagonal with W5 = 24× W1 passes.

### Implementation

**File: `.worktrees/ew-engine-v2/pattern_engine/elliott_wave.py`**
**Function: `find_diagonal_candidates()` — at line ~1566, just before `candidates.append()`:**

Add:

```python
        # ── Proportionality gate (same as impulse finders) ────────────
        # One wave must not absurdly dominate the pattern.
        _diag_sizes = sorted([w1.price_change, w3.price_change, w5.price_change])
        if _diag_sizes[0] > 0 and _diag_sizes[1] > 0:
            if _diag_sizes[2] / _diag_sizes[1] > 5.0:
                continue  # one wave absurdly dominant
            if _diag_sizes[0] / _diag_sizes[2] < 0.03:
                continue  # smallest wave is noise-level
```

### Why this works

PLTR diagonal: sorted = [$4, $8, $190]. 190/8 = 23.75 > 5.0 → **REJECTED**

Normal diagonals where waves progressively contract or expand → ratios well within 5.0× → **KEPT**

---

## Fix D — Universal max-wave-dominance gate in validate_and_score() (Safety Net)

### Problem
Even with gates in individual finders, edge cases may slip through — especially for new pattern types added in the future. A universal check in the scoring layer catches any pattern where one wave overwhelms the others.

### Implementation

**File: `.worktrees/ew-engine-v2/pattern_engine/elliott_wave.py`**
**Function: `validate_and_score()` — at line ~1617, after the bar span filter added in V7, before the `chart_high`/`chart_low` computation:**

Add:

```python
    # ── Max wave dominance filter ────────────────────────────────────────
    # Reject candidates where a single wave's price_change exceeds 80% of
    # the SUM of all waves' price_changes. Such patterns have one wave
    # covering the entire chart as a single straight line.
    # Normal ext-W3 at 5.0× gate limit: 5/(1+5+1) = 71.4% — passes.
    # PLTR degenerate zigzag: 190/(8+4+190) = 94.1% — rejected.
    def _max_dominance(c: dict) -> bool:
        changes = [w.price_change for w in c["waves"]]
        total = sum(changes)
        if total <= 0:
            return True  # can't compute, keep the candidate
        return max(changes) / total <= 0.80

    candidates = [c for c in candidates if _max_dominance(c)]
    if not candidates:
        return []
```

### Why this works

- Normal impulse (equal waves): max = 33% ✓
- Extended W3 at 2.618×: 2.618/(1+0.5+2.618+0.8+1) = 2.618/5.938 = 44% ✓
- Extended W3 at 5.0× (gate limit): 5/(1+0.3+5+1.5+1) = 5/8.8 = 56.8% ✓
- Normal zigzag C=A: A=100, B=50, C=100 → 100/250 = 40% ✓
- Extended zigzag C=3×A: A=100, B=50, C=300 → 300/450 = 66.7% ✓
- PLTR zigzag: 190/(8+4+190) = 94.1% ✗ **REJECTED**
- PLTR diagonal: 190/(4+4+8+4+190) = 92.2% ✗ **REJECTED**

The 80% threshold is generous enough for all valid EW patterns while catching the absurd cases.

---

## Summary of all changes

| Bug | Fix | File | Location | Change |
|-----|-----|------|----------|--------|
| 23 | A | `elliott_wave.py` | `analyze_multi_degree()` ~L2053 | Add adaptive chart-wide pivot scale with binary search |
| 23 | A | `elliott_wave.py` | `analyze_multi_degree()` ~L2074 | Add `_chart_wide_pivots` to pivot-based finder loop |
| 24 | B | `elliott_wave.py` | `find_zigzag_candidates()` ~L1367 | Add C/A proportionality gate (> 5.0× → reject) |
| 24 | B | `elliott_wave.py` | `find_flat_candidates()` ~L1430 | Same C/A proportionality gate |
| 25 | C | `elliott_wave.py` | `find_diagonal_candidates()` ~L1566 | Add impulse-style proportionality gate |
| — | D | `elliott_wave.py` | `validate_and_score()` ~L1618 | Universal max-wave-dominance gate (> 80% → reject) |

## Files to edit

1. `.worktrees/ew-engine-v2/pattern_engine/elliott_wave.py` — ALL changes are in this single file

## Expected results after fix

**PLTR** — The degenerate zigzag (C = 23.75× A) and diagonal (W5 = 24× W1) are both rejected. The adaptive chart-wide pivots now include $85/$58/$208, so the pivot-based finder discovers the correct impulse: W1($14→$22), W2($22→$18), W3($18→$85), W4($85→$58), W5($58→$208). This pattern has good proportionality (2.24×), valid W4 retracement (40%), and scores high due to full-chart span + extreme anchoring.

**CRWV** — The linked correction's wave C ($148→$85 over 6 bars) is short and proportional to wave A ($187→$100). The 5.0× gate won't affect it. However, if the user wants sub-wave detail within correction legs, that's a future multi-degree rendering feature.

**Other charts** — The 16 clean charts from the 20-chart test are unaffected:
- The adaptive pivot scale only activates when primary pivots hit the 20-cap (most charts don't)
- The corrective/diagonal proportionality gates only reject ratios > 5.0× (normal patterns are 1-3×)
- The dominance gate at 80% only catches extreme outliers

## Testing

1. Restart backend (`start-ew-dev.sh`)
2. Test PLTR first — should now show a proper 5-wave impulse spanning the full $14→$208 rally
3. Spot-check MSFT, AMZN, NVDA, BTC-USD — should be unchanged
4. Test META, INTC — V7 shallow-W4 fix should still be active
5. Test GC=F — V7 orphan line fix should still be active
