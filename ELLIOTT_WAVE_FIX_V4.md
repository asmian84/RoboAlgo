# Elliott Wave V2 Engine — Fix V4: Wave Proportionality

**File**: `.worktrees/ew-engine-v2/pattern_engine/elliott_wave.py`

## Previous fixes applied (DO NOT re-apply): V1 (Bugs 1-5), V2 (pivot-based finder), V3 (diagonal separation)

---

## Problem: Giant W5 With Cramped W1-W4

Every chart shows the same degenerate pattern:
- W1, W2, W3, W4 crammed into a tiny zone at one end of the chart
- W5 is a single giant line spanning 80-95% of the chart's price range

**BTC-USD**: W1-W4 all in $5,000-$10,000 zone, then W5 = single line to $126,198
**EONR**: W1-W4 in $2.70-$2.92 zone, then W5 = straight line down to $0.27
**CL=F**: W1-W4 in $55-$80 zone, then W5 = $82→$119 (better but still disproportionate)
**GC=F**: W1-W4 in $2,300-$3,500 zone, then W5 = $3,400→$5,586

These patterns pass all 3 Elliott rules, but they're structurally absurd. In real Elliott Wave analysis, the extended wave is typically W3 (for stocks) or occasionally W5 (for commodities), but even extended waves are at most 2.618× the next largest wave — not 20× or 80×.

## Root cause

1. The `find_pivot_based_impulse_candidates()` function has **no proportionality constraint** — any 6-pivot combo that passes the 3 rules becomes a candidate, regardless of wave ratios.

2. The `_score_proportionality()` function (line 1795) only checks **W4/W2 time ratio** — it says nothing about the relative price sizes of W1, W3, W5.

3. The span scoring (55% weight) + anchor bonus (10% weight) **rewards** these degenerate patterns because they cover the full chart.

---

## Bug 10: Add proportionality gate to pivot-based finder

In `find_pivot_based_impulse_candidates()`, add this check right **before** the `candidates.append(...)` call (both bullish and bearish branches):

### Bullish branch (add before `candidates.append` around line 1190):

```python
                            # ── Proportionality gate ──────────────────────────
                            # Reject grossly disproportionate patterns where one
                            # impulse wave dominates all others.  In valid Elliott
                            # Wave patterns, even an extended wave is typically
                            # ≤ 4× the next-largest impulse wave.  Patterns where
                            # the ratio exceeds 6 are structurally meaningless
                            # (e.g. tiny W1-W4 with a giant W5 spanning 90% of the
                            # chart).
                            impulse_sizes = sorted([w1_move, w3_move, w5_move])
                            if impulse_sizes[0] > 0:
                                # Ratio of largest to second-largest impulse wave
                                top_ratio = impulse_sizes[2] / impulse_sizes[1]
                                if top_ratio > 5.0:
                                    continue  # reject: one wave is absurdly dominant
                            # Also reject if smallest impulse wave is < 5% of largest
                            if impulse_sizes[2] > 0 and impulse_sizes[0] / impulse_sizes[2] < 0.05:
                                continue  # reject: smallest wave is noise-level
```

### Bearish branch (add the same block before `candidates.append` in the bearish section):

Same code — add identical proportionality gate before the bearish `candidates.append(...)`.

---

## Bug 11: Strengthen impulse-wave proportionality in scoring

Replace `_score_proportionality()` (line 1795) to also check price ratios:

**OLD**:
```python
def _score_proportionality(waves: list[MonoWave]) -> float:
    """
    Brown's box method: corrective waves should have similar time proportions.
    Score 1.0 if W4/W2 time ratio is within [0.382, 2.618].
    """
    if len(waves) < 5:
        return 0.5
    w2_bars = waves[1].bar_count
    w4_bars = waves[3].bar_count
    if w2_bars == 0:
        return 0.5
    ratio = w4_bars / w2_bars
    if 0.382 <= ratio <= 2.618:
        return 1.0
    return 0.2  # outside range — probably different degree
```

**NEW**:
```python
def _score_proportionality(waves: list[MonoWave]) -> float:
    """
    Score wave proportionality on two dimensions:
    1. Brown's box: W4/W2 time ratio within [0.382, 2.618]
    2. Impulse balance: largest/second-largest impulse wave price ratio

    A perfectly proportional impulse (all waves similar size) scores 1.0.
    A wildly disproportionate one (W5 = 20× W1) scores near 0.0.
    """
    if len(waves) < 5:
        return 0.5

    # ── Time proportionality (Brown's box) ────────────────────────────
    w2_bars = waves[1].bar_count
    w4_bars = waves[3].bar_count
    if w2_bars > 0:
        time_ratio = w4_bars / w2_bars
        time_score = 1.0 if 0.382 <= time_ratio <= 2.618 else 0.2
    else:
        time_score = 0.5

    # ── Price proportionality (impulse wave balance) ──────────────────
    w1_size = waves[0].price_change
    w3_size = waves[2].price_change
    w5_size = waves[4].price_change
    sizes = sorted([w1_size, w3_size, w5_size])

    if sizes[1] > 0:
        # Ratio of largest to second-largest impulse wave
        price_ratio = sizes[2] / sizes[1]
        # Perfect: ratio ~1.618 (golden ratio extension)
        # Acceptable: ratio up to 3.0 (strong extended wave)
        # Poor: ratio 3-5 (questionable)
        # Terrible: ratio > 5 (structurally invalid)
        if price_ratio <= 2.0:
            price_score = 1.0
        elif price_ratio <= 3.0:
            price_score = 0.7
        elif price_ratio <= 5.0:
            price_score = 0.3
        else:
            price_score = 0.05  # near-zero for absurd ratios
    else:
        price_score = 0.1

    # Weighted average: price balance matters more than time
    return price_score * 0.7 + time_score * 0.3
```

---

## Bug 12: Increase proportionality weight in scoring

The current proportionality weight is only **5%** (line 1539). This is far too low — a disproportionate pattern loses only 0.05 points, which the span score easily compensates.

**OLD** (line 1539):
```python
    "proportionality":    0.05,
```

**NEW**:
```python
    "proportionality":    0.15,
```

And reduce span weight to compensate:

**OLD**:
```python
    "span":               0.55,
```

**NEW**:
```python
    "span":               0.45,
```

This rebalances the scoring so that a large-span but disproportionate pattern can't simply overwhelm a well-proportioned one.

---

## VERIFICATION

After applying, restart backend and test:

**BTC-USD**: Should find W1=$3,500→$64,000, W2=$64,000→$16,000, W3=$16,000→$126,198 (extended W3). The cramped W1-W4 at $5K-$10K should be rejected by the proportionality gate (ratio >5×).

**EONR**: Should find bearish W1 from $2.92 or $4.00 down, with intermediate structure. Not a single giant W5 line.

**CL=F**: Should show W3 extended ($62→$119), not W5 extended with a cramped W3 at $80.

**GC=F**: May now prefer W3 extended ($2,560→$5,586) over the current W5 extended count, since the W5=$3,400→$5,586 to W1=$500 ratio is 4.4× which is borderline.

**Key visual test**: No chart should show 4 wave labels crammed in one corner with a single line spanning 80%+ of the chart. The waves should look proportional and spread across the chart.
