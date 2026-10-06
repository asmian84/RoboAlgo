# Elliott Wave V2 Engine — Architecture Fix (Pivot-Based Candidate Finder)

**CRITICAL**: The backend runs from the WORKTREE, not the main repo:
```
Server directory: /Users/asmian/RoboAlgo/.worktrees/ew-engine-v2/
File to fix:      .worktrees/ew-engine-v2/pattern_engine/elliott_wave.py
```
(See `start-ew-dev.sh` — it does `cd "$WORKTREE" && uvicorn api.main:app`)

## Context: Previous Fixes Already Applied

Five fixes from the first prompt are already in the v2 worktree file and working:
- Bug 1: ATH/ATL injection in `detect_major_pivots()` (line 553)
- Bug 2: Linear span scoring (line 1452)
- Bug 3: Rule 3 grace reduced to 2% (line 2149)
- Bug 4: 20% minimum span filter (line 1386)
- Bug 5: Anchor bonus at 10% weight (line 1472)

**DO NOT re-apply these fixes. They are already in the file.**

---

## ROOT CAUSE: The Sliding Window of 5 Cannot Find Extended-W3 Patterns

The core remaining problem is architectural. `find_impulse_candidates()` (line 986) scans a **sliding window of exactly 5 consecutive monowaves** and treats each monowave as one Elliott Wave:

```python
for i in range(n - 4):
    seq = monowaves[i: i + 5]
    w1, w2, w3, w4, w5 = seq
```

This means **each wave = exactly 1 monowave**. In reality, the most common impulse pattern has an **extended Wave 3** that contains 5 sub-waves at a lower degree. At the intermediate pivot scale, W3 might span 8-15 monowaves. The sliding window can never capture this — it only sees tiny 5-monowave fragments.

### Why this causes cramped patterns on every chart:

**GC=F (Gold)**: Correct count is W1: $2,277→$2,800, W2: $2,800→$2,550, W3: $2,550→$5,586 (extended). At intermediate scale, W3 spans ~50+ monowaves. The sliding window finds tiny 5-monowave patterns near the top ($4,000-$5,586).

**AAPL**: Correct count is W1: $122→$190, W2: $190→$165, W3: $165→$288 (extended). At intermediate scale, W3 spans ~20 monowaves. Same cramped-pattern problem.

**Why the primary (15% swing) scale doesn't help**: GC=F's W2 pullback is only 8.9% ($2,800→$2,550), and AAPL's W2 pullback is only 13.2% ($190→$165). Both are BELOW the 15% threshold, so `detect_major_pivots(min_swing_pct=0.15)` merges W1+W2+W3 into a single monowave. The primary scale produces too few pivots to form any 5-wave window.

---

## FIX: Add a Pivot-Based Impulse Candidate Finder

### Step 1: Add `find_pivot_based_impulse_candidates()` — NEW FUNCTION

Add this function right after `find_impulse_candidates()` (after line 1080):

```python
def find_pivot_based_impulse_candidates(
    pivots: list[Pivot],
    df: pd.DataFrame,
    is_commodity: bool = False,
) -> list[dict]:
    """
    Find impulse candidates by enumerating valid 6-pivot combinations.

    Unlike find_impulse_candidates() which uses a sliding window of 5
    consecutive monowaves, this function selects 6 pivots (forming 5 waves)
    that may have any number of intermediate pivots between them.
    This captures extended-W3 patterns where W3 spans many intermediate
    pivot swings.

    The function works directly with pivot points (not monowaves), building
    MonoWave objects from the selected pivots and classifying their internal
    structure.

    Complexity: For n pivots, we enumerate combinations of 3 highs × 3 lows
    (bullish) or 3 lows × 3 highs (bearish). With n ≤ 20 (typical for
    primary/sub-primary scales), this is at most ~1000 iterations after
    pruning — negligible cost.
    """
    if len(pivots) < 6:
        return []

    candidates: list[dict] = []
    highs = [p for p in pivots if p.is_high]
    lows  = [p for p in pivots if not p.is_high]

    # ── Bullish impulse: L0-H1-L2-H3-L4-H5 ─────────────────────────────
    for i0, l0 in enumerate(lows):
        for i1, h1 in enumerate(highs):
            if h1.idx <= l0.idx:
                continue
            w1_move = h1.price - l0.price
            if w1_move <= 0:
                continue  # W1 must go up

            for i2, l2 in enumerate(lows):
                if l2.idx <= h1.idx:
                    continue
                if l2.price <= l0.price:
                    continue  # Rule 1: W2 retraces < 100%

                for i3, h3 in enumerate(highs):
                    if h3.idx <= l2.idx:
                        continue
                    w3_move = h3.price - l2.price
                    if w3_move <= 0:
                        continue  # W3 must go up

                    for i4, l4 in enumerate(lows):
                        if l4.idx <= h3.idx:
                            continue
                        # Rule 3: W4 must not overlap W1 territory
                        grace = w1_move * 0.02
                        if l4.price < h1.price - grace:
                            continue

                        for i5, h5 in enumerate(highs):
                            if h5.idx <= l4.idx:
                                continue
                            w5_move = h5.price - l4.price
                            if w5_move <= 0:
                                continue  # W5 must go up

                            # Rule 2: W3 must not be shortest
                            if w3_move < w1_move and w3_move < w5_move:
                                continue

                            # Build MonoWaves from pivot pairs
                            waves = [
                                MonoWave(start=l0, end=h1),  # W1
                                MonoWave(start=h1, end=l2),  # W2
                                MonoWave(start=l2, end=h3),  # W3
                                MonoWave(start=h3, end=l4),  # W4
                                MonoWave(start=l4, end=h5),  # W5
                            ]

                            # Classify internal structure of each wave
                            for mw in waves:
                                structure, method, count = classify_internal_structure(
                                    df, mw.start.idx, mw.end.idx
                                )
                                mw.internal_structure    = structure
                                mw.classification_method = method
                                mw.sub_pivot_count       = count

                            # Detect extension
                            sizes = [w1_move, w3_move, w5_move]
                            max_size = max(sizes)
                            ext_idx  = sizes.index(max_size)
                            ext_wave = [1, 3, 5][ext_idx]
                            is_extended = max_size >= 1.618 * min(sizes)

                            wave_type = _get_impulse_wave_type(
                                ext_wave if is_extended else None
                            )

                            # Truncation check: W5 fails to exceed W3
                            is_truncated = False
                            if h5.price < h3.price:
                                if w3_move > 1.618 * w1_move:
                                    is_truncated = True
                                    wave_type = WaveType.TRUNCATED_5TH
                                else:
                                    continue  # invalid

                            candidates.append({
                                "waves":          waves,
                                "wave_type":      wave_type,
                                "extension_wave": ext_wave if is_extended else None,
                                "is_truncated":   is_truncated,
                                "direction":      WaveDirection.UP,
                                "position_hint":  "",
                            })

    # ── Bearish impulse: H0-L1-H2-L3-H4-L5 ──────────────────────────────
    for i0, h0 in enumerate(highs):
        for i1, l1 in enumerate(lows):
            if l1.idx <= h0.idx:
                continue
            w1_move = h0.price - l1.price
            if w1_move <= 0:
                continue  # W1 must go down

            for i2, h2 in enumerate(highs):
                if h2.idx <= l1.idx:
                    continue
                if h2.price >= h0.price:
                    continue  # Rule 1: W2 retraces < 100%

                for i3, l3 in enumerate(lows):
                    if l3.idx <= h2.idx:
                        continue
                    w3_move = h2.price - l3.price
                    if w3_move <= 0:
                        continue  # W3 must go down

                    for i4, h4 in enumerate(highs):
                        if h4.idx <= l3.idx:
                            continue
                        # Rule 3: W4 must not overlap W1 territory
                        grace = w1_move * 0.02
                        if h4.price > l1.price + grace:
                            continue

                        for i5, l5 in enumerate(lows):
                            if l5.idx <= h4.idx:
                                continue
                            w5_move = h4.price - l5.price
                            if w5_move <= 0:
                                continue  # W5 must go down

                            # Rule 2: W3 must not be shortest
                            if w3_move < w1_move and w3_move < w5_move:
                                continue

                            waves = [
                                MonoWave(start=h0, end=l1),  # W1
                                MonoWave(start=l1, end=h2),  # W2
                                MonoWave(start=h2, end=l3),  # W3
                                MonoWave(start=l3, end=h4),  # W4
                                MonoWave(start=h4, end=l5),  # W5
                            ]

                            for mw in waves:
                                structure, method, count = classify_internal_structure(
                                    df, mw.start.idx, mw.end.idx
                                )
                                mw.internal_structure    = structure
                                mw.classification_method = method
                                mw.sub_pivot_count       = count

                            sizes = [w1_move, w3_move, w5_move]
                            max_size = max(sizes)
                            ext_idx  = sizes.index(max_size)
                            ext_wave = [1, 3, 5][ext_idx]
                            is_extended = max_size >= 1.618 * min(sizes)

                            wave_type = _get_impulse_wave_type(
                                ext_wave if is_extended else None
                            )

                            is_truncated = False
                            if l5.price > l3.price:
                                if w3_move > 1.618 * w1_move:
                                    is_truncated = True
                                    wave_type = WaveType.TRUNCATED_5TH
                                else:
                                    continue

                            candidates.append({
                                "waves":          waves,
                                "wave_type":      wave_type,
                                "extension_wave": ext_wave if is_extended else None,
                                "is_truncated":   is_truncated,
                                "direction":      WaveDirection.DOWN,
                                "position_hint":  "",
                            })

    return candidates
```

---

### Step 2: Add a sub-primary pivot scale with LOWER swing threshold

In `analyze_multi_degree()`, after the primary pivot detection (around line 1721-1730), add a sub-primary scale. The 15% primary threshold misses W2 corrections smaller than 15% (GC=F W2 is 8.9%, AAPL W2 is 13.2%).

Add this **right after** the primary_monowaves classification loop (after line 1730):

```python
    # Sub-primary scale: lower swing threshold to capture smaller W2 corrections.
    # The 15% primary threshold misses W2 pullbacks of 8-14% which are common
    # in strong trends (e.g. GC=F W2 = 8.9%, AAPL W2 = 13.2%).
    sub_primary_pivots = detect_major_pivots(df, min_swing_pct=0.07, max_pivots=20)
```

---

### Step 3: Integrate into the candidate gathering loop

In `analyze_multi_degree()`, replace the candidate-gathering section (approximately lines 1735-1744):

**OLD** (current code):
```python
    # ── Layer 3: Gather all candidates (all three scales) ─────────────────
    all_candidates: list[dict] = []
    for mws in [monowaves, macro_monowaves, primary_monowaves]:
        if len(mws) < 3:
            continue
        all_candidates.extend(find_impulse_candidates(mws, is_commodity=is_commodity))
        all_candidates.extend(find_zigzag_candidates(mws))
        all_candidates.extend(find_flat_candidates(mws))
        all_candidates.extend(find_triangle_candidates(mws))
        all_candidates.extend(find_diagonal_candidates(mws))
```

**NEW** (replace with):
```python
    # ── Layer 3: Gather all candidates (all scales) ──────────────────────
    all_candidates: list[dict] = []

    # Standard sliding-window candidates from all three monowave scales
    for mws in [monowaves, macro_monowaves, primary_monowaves]:
        if len(mws) < 3:
            continue
        all_candidates.extend(find_impulse_candidates(mws, is_commodity=is_commodity))
        all_candidates.extend(find_zigzag_candidates(mws))
        all_candidates.extend(find_flat_candidates(mws))
        all_candidates.extend(find_triangle_candidates(mws))
        all_candidates.extend(find_diagonal_candidates(mws))

    # Pivot-based candidates: enumerate valid 6-pivot combinations.
    # This captures extended-W3 patterns that the sliding window misses
    # because W3 may span many monowaves at finer scales.
    # Run on primary and sub-primary pivot sets (coarser = fewer pivots = faster).
    for pivot_set in [primary_pivots, sub_primary_pivots]:
        if len(pivot_set) >= 6:
            all_candidates.extend(
                find_pivot_based_impulse_candidates(pivot_set, df, is_commodity=is_commodity)
            )
```

---

### Step 4: Add deduplication before scoring

The pivot-based finder may produce candidates similar to the sliding-window finder. Add dedup logic right before scoring (between the candidate gathering and `validate_and_score()`):

```python
    # ── Deduplicate candidates by wave boundary signature ────────────────
    # Two candidates are "the same" if all 6 pivot points (W1 start through
    # W5 end) are within 3 bars and 1% price of each other.
    def _cand_sig(c):
        """Signature for deduplication — two candidates with same 6 pivots are duplicates."""
        ws = c["waves"]
        # Use all 6 pivot indices (W1 start, W1 end, W2 end, W3 end, W4 end, W5 end)
        return (
            ws[0].start.idx, ws[0].end.idx, ws[1].end.idx,
            ws[2].end.idx, ws[3].end.idx, ws[4].end.idx,
        )

    seen_sigs = set()
    deduped = []
    for c in all_candidates:
        sig = _cand_sig(c)
        if sig not in seen_sigs:
            seen_sigs.add(sig)
            deduped.append(c)
    all_candidates = deduped
```

---

## VERIFICATION

After all fixes, restart the backend (`start-ew-dev.sh`) and test:

**1. GC=F daily**: The pivot-based finder should discover L=$2,277, H=$2,800, L=$2,550, H=$5,586, L=~$5,100, H=~$5,500. This gives an extended-W3 impulse spanning the full chart. W3 ($2,550→$5,586) should be visually the longest wave.

**2. AAPL daily**: Should find L=$122, H=$190, L=$165, H=$288 and complete the pattern from there. W4 at ~$250 does NOT overlap W1 end at $190 (valid). Full chart span.

**3. EONR daily**: Bearish impulse from ~$4.00 ATH (if in data window) or ~$3.00. The ATH injection from Bug 1 should provide the $4.00 pivot if available. Look for H=$4.00, L=..., H=..., L=$0.27.

**4. BTC-USD daily**: Should find the full impulse from 2023 lows, not a cramped count near the top.

**Key visual test**: On every chart, the wave pattern should span most of the chart's price range, with W3 being the longest wave. If patterns are still cramped in a narrow zone, check:
- That `sub_primary_pivots` is being generated (add a log: `logger.info(f"Sub-primary pivots: {len(sub_primary_pivots)}")`)
- That `find_pivot_based_impulse_candidates()` is returning candidates (add a log before dedup)
- That the dedup isn't removing the good candidates (check signature collision)
