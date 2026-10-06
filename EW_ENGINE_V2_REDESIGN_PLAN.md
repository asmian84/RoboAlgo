# Elliott Wave Engine v2 → v3: Multi-Degree Recursive Wave Counting

## What We're Building

A true recursive multi-degree wave counter — like WaveBasis — that labels EVERY bar at EVERY degree, always identifies where we are in the count, and shows primary + alternate counts with confidence scores.

**Current engine (v2):** Finds ONE best pattern (impulse/zigzag/diagonal) at one degree. Draws connecting lines between pivot points. Leaves gaps.

**New engine (v3):** Recursively decomposes the entire chart into nested waves from Cycle degree down to Minute degree. Every bar belongs to a wave at every degree. No gaps.

---

## What We Keep (~70% of existing code)

| Component | File / Function | Reuse Status |
|-----------|----------------|--------------|
| Monowave decomposition | `detect_monowaves()` | **100% reused** — foundation of all wave identification |
| Major pivot detection | `detect_major_pivots()` | **100% reused** — adaptive chart-wide pivots from V8 |
| Impulse candidate finding | `find_impulse_candidates()` | **90% reused** — add recursive sub-count call |
| Zigzag/Flat/Diagonal finders | `find_zigzag_candidates()` etc. | **90% reused** — add recursive sub-count call |
| Validation rules | W2<100%, W3 not shortest, W4 no overlap | **100% reused** |
| Proportionality gates | V7+V8 gates (5.0× ratio, 80% dominance) | **100% reused** |
| Fibonacci scoring | `_score_proportionality()`, tiers | **100% reused** |
| Asset-class thresholds | Forex/crypto/equity swing % | **100% reused** |
| API serialization | `_ser_v2_count()` | **Rewritten** — new nested output format |
| Frontend rendering | `KLineChartOverlay.tsx` | **Major changes** — wave labels instead of connecting lines |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                 analyze_recursive()                   │
│  Input: price bars, degree=CYCLE                     │
│  Output: WaveTree (nested counts at all degrees)     │
├─────────────────────────────────────────────────────┤
│                                                       │
│  1. Detect monowaves at this degree's scale           │
│  2. Detect pivots at this degree's scale              │
│  3. Find ALL candidate patterns (impulse/zigzag/etc)  │
│  4. For each candidate:                               │
│     ├── Validate rules (existing gates)               │
│     ├── For EACH wave in the candidate:               │
│     │   └── RECURSE: analyze_recursive(sub-bars,      │
│     │                 degree=degree-1)                 │
│     ├── Score candidate (existing + sub-count quality) │
│     └── Store as alternate count                      │
│  5. Rank alternates, pick primary + top 3 alternates  │
│  6. Return WaveTree node                              │
│                                                       │
└─────────────────────────────────────────────────────┘
```

### Degree Hierarchy

```
SUPERCYCLE  →  60%+ swing pivots  (rarely used, multi-decade charts only)
CYCLE       →  35-50% swing pivots  (3-10 year charts)
PRIMARY     →  15-25% swing pivots  (1-3 year charts, this is our current "primary")
INTERMEDIATE→  7-12% swing pivots   (this is our current "sub_primary")
MINOR       →  3-6% swing pivots    (short-term waves)
MINUTE      →  1-3% swing pivots    (intraday on daily charts — terminal node)
```

Each degree uses a different swing threshold to detect monowaves/pivots. The swing thresholds scale by asset class (existing logic). Recursion stops at MINUTE degree or when the bar count is too small to decompose further (< 5 bars).

---

## New Data Structures

### WaveTree (replaces flat WaveCount)

```python
@dataclass
class WaveNode:
    """A single wave at a specific degree."""
    degree: WaveDegree           # CYCLE, PRIMARY, INTERMEDIATE, etc.
    wave_type: WaveType          # IMPULSE, ZIGZAG, FLAT, DIAGONAL, etc.
    label: str                   # "1", "2", "3", "A", "B", "C", "i", "ii" etc.
    start_idx: int               # bar index where this wave starts
    end_idx: int                 # bar index where this wave ends
    start_price: float
    end_price: float
    direction: WaveDirection     # UP or DOWN
    sub_waves: List['WaveNode']  # recursive children at next lower degree
    confidence: float            # 0.0-1.0 for this node
    
class WaveDegree(Enum):
    SUPERCYCLE = 6
    CYCLE = 5
    PRIMARY = 4
    INTERMEDIATE = 3
    MINOR = 2
    MINUTE = 1
    SUBMINUTE = 0  # terminal

@dataclass
class WaveTree:
    """Complete multi-degree wave count for a chart."""
    symbol: str
    timeframe: str
    root: WaveNode               # top-level degree
    alternates: List[WaveNode]   # top 3 alternate counts at the root degree
    current_position: str        # e.g., "Cycle III, Primary (4), Intermediate [c]"
    overall_confidence: float
```

### How Labels Work

The label format changes by degree (standard EW notation):

| Degree | Impulse Labels | Corrective Labels | Example |
|--------|---------------|-------------------|---------|
| Cycle | I II III IV V | A B C | Roman numerals |
| Primary | (1) (2) (3) (4) (5) | (A) (B) (C) | Parenthesized |
| Intermediate | [1] [2] [3] [4] [5] | [a] [b] [c] | Bracketed |
| Minor | 1 2 3 4 5 | a b c | Plain |
| Minute | i ii iii iv v | a b c | Lowercase roman |

---

## The Recursive Algorithm

### Step 1: Determine top degree from chart span

```python
def determine_top_degree(df, timeframe):
    """Pick the highest degree that makes sense for this chart."""
    bar_count = len(df)
    price_range_pct = (df['high'].max() - df['low'].min()) / df['low'].min()
    
    if timeframe == 'weekly' and bar_count > 500:
        return WaveDegree.SUPERCYCLE
    elif bar_count > 200 and price_range_pct > 2.0:
        return WaveDegree.CYCLE
    elif bar_count > 100:
        return WaveDegree.PRIMARY
    elif bar_count > 40:
        return WaveDegree.INTERMEDIATE
    else:
        return WaveDegree.MINOR
```

### Step 2: Recursive decomposition

```python
def analyze_recursive(df, degree, direction_hint=None):
    """
    Core recursive function. Decomposes df into waves at `degree`,
    then recurses into each wave at degree-1.
    """
    # Base case: too few bars or at terminal degree
    if len(df) < 5 or degree == WaveDegree.SUBMINUTE:
        return None  # terminal monowave
    
    # Get swing threshold for this degree + asset class
    swing_pct = DEGREE_SWING_MAP[degree][asset_class]
    
    # Reuse existing infrastructure
    monowaves = detect_monowaves(df, min_swing_pct=swing_pct)
    pivots = detect_major_pivots(df, min_swing_pct=swing_pct)
    
    # Find ALL candidate patterns at this degree (existing finders)
    candidates = []
    candidates += find_impulse_candidates(monowaves, pivots, df)
    candidates += find_pivot_based_impulse_candidates(pivots, df)
    candidates += find_zigzag_candidates(monowaves, df)
    candidates += find_flat_candidates(monowaves, df)
    candidates += find_diagonal_candidates(monowaves, df)
    
    # Validate and score (existing gates + scoring)
    candidates = validate_and_score(candidates, df)
    
    # KEY NEW STEP: For each candidate, recursively decompose sub-waves
    for candidate in candidates:
        for wave in candidate['waves']:
            sub_df = df.iloc[wave.start_idx : wave.end_idx + 1]
            wave.sub_count = analyze_recursive(
                sub_df, 
                degree=WaveDegree(degree.value - 1),
                direction_hint=wave.direction
            )
        # Boost score if sub-waves decompose cleanly
        candidate['score'] *= sub_wave_quality_multiplier(candidate)
    
    # Re-rank after sub-wave quality adjustment
    candidates.sort(key=lambda c: c['score'], reverse=True)
    
    # Build WaveNode for best candidate
    best = candidates[0]
    node = WaveNode(
        degree=degree,
        wave_type=best['wave_type'],
        sub_waves=[wave_to_node(w, degree) for w in best['waves']],
        confidence=best['score'],
        ...
    )
    
    return node
```

### Step 3: Full-chart coverage guarantee

The recursion naturally covers the entire chart because:

1. Top-level pattern spans the full chart (enforce: candidate must cover ≥85% of bars)
2. Each wave's sub-count covers that wave's full bar range
3. Terminal monowaves (at MINUTE degree) cover any remaining bars
4. If no valid pattern is found at a degree, the bars are labeled as an "unresolved" segment with just monowaves

### Step 4: Current position identification

```python
def identify_current_position(tree: WaveTree) -> str:
    """Walk the tree from root to the leaf that contains the last bar."""
    position_parts = []
    node = tree.root
    
    while node and node.sub_waves:
        # Find which sub-wave contains the last bar
        last_wave = node.sub_waves[-1]  # the most recent wave
        position_parts.append(f"{node.degree.name} {last_wave.label}")
        node = last_wave.sub_count  # recurse into it
    
    return ", ".join(position_parts)
    # e.g., "Cycle III, Primary (4), Intermediate [c] of (4)"
```

---

## Projection Engine (Fibonacci Forward Targets)

Once we have a complete wave tree, we can compute **forward projections** — price targets for the next expected wave based on Fibonacci extensions of prior waves. This is what WaveBasis shows as the lines extending to the right of the chart.

### How it works:

```python
def compute_projections(tree: WaveTree) -> List[Projection]:
    """
    Walk the tree to find the current incomplete wave,
    then compute Fibonacci extension targets.
    """
    current_wave = find_current_wave(tree)  # the wave we're "inside"
    parent = current_wave.parent
    
    projections = []
    
    if parent.wave_type in [IMPULSE, DIAGONAL]:
        if current_wave.label in ['3', '(3)', '[3]']:
            # W3 target = W1 length × fib ratio, projected from W2 end
            w1_length = parent.sub_waves[0].price_range
            w2_end = parent.sub_waves[1].end_price
            for fib in [1.0, 1.618, 2.618]:
                target = w2_end + (w1_length * fib * direction)
                projections.append(Projection(target, f'{fib*100}% ext'))
        
        elif current_wave.label in ['5', '(5)', '[5]']:
            # W5 target = W1-to-W3 length × fib ratio
            w1_to_w3 = parent.sub_waves[2].end_price - parent.sub_waves[0].start_price
            w4_end = parent.sub_waves[3].end_price
            for fib in [0.618, 1.0]:
                target = w4_end + (w1_to_w3 * fib * direction)
                projections.append(Projection(target, f'{fib*100}% ext'))
    
    # Invalidation level (where the count breaks)
    if current_wave.label in ['3', '(3)']:
        # W3 invalidates if price goes below W1 start
        projections.append(Projection(parent.sub_waves[0].start_price, 'invalidation', is_risk=True))
    
    return projections
```

### Projection display:
- Solid/dashed lines extending from current price to the right into empty space
- Color coded: green for targets, red for invalidation
- Labels on right axis: "988.2 (161.8%)" with R/R ratio
- Multiple fib levels shown: 61.8%, 100%, 161.8%, 261.8%

---

## Starting Point Detection

Instead of forcing the wave count to start at bar 0, the engine identifies the **most relevant structural extreme** as the starting point:

```python
def find_starting_point(df, timeframe):
    """
    Find the structural extreme that anchors the wave count.
    The count runs from this point to the current bar.
    """
    # Find the deepest low or highest high that forms a clear reversal
    # Use multiple criteria:
    # 1. Magnitude: largest % move from this point
    # 2. Recency: prefer more recent starting points
    # 3. Structure: must be a clear pivot (surrounded by opposite moves)
    
    candidates = detect_major_pivots(df, min_swing_pct=threshold)
    
    # Score candidates by how well they anchor a complete wave count
    for pivot in candidates:
        sub_df = df.iloc[pivot.idx:]
        tree = analyze_recursive(sub_df, top_degree)
        pivot.anchor_score = tree.overall_confidence
    
    return max(candidates, key=lambda p: p.anchor_score)
```

The starting point is rendered as a **circled "o" marker** with an info box showing the price, pattern type, and degree — exactly like WaveBasis.

---

## Timeframe-Agnostic Design

The engine works identically on any timeframe. The only thing that changes is the degree mapping:

```python
TIMEFRAME_DEGREE_MAP = {
    # Timeframe → (top_degree, degree_count)
    '1m':  (WaveDegree.MINUTE, 3),       # Minute → Subminute → Micro
    '5m':  (WaveDegree.MINOR, 3),        # Minor → Minute → Subminute
    '15m': (WaveDegree.MINOR, 4),        # Minor → Minute → Subminute → Micro
    '1h':  (WaveDegree.INTERMEDIATE, 4), # Intermediate → Minor → Minute → Subminute
    '4h':  (WaveDegree.INTERMEDIATE, 4),
    '1D':  (WaveDegree.PRIMARY, 5),      # Primary → Intermediate → Minor → Minute → Subminute
    '1W':  (WaveDegree.CYCLE, 5),        # Cycle → Primary → Intermediate → Minor → Minute
    '1M':  (WaveDegree.SUPERCYCLE, 5),   # Supercycle → Cycle → Primary → Intermediate → Minor
}
```

The swing thresholds also scale by timeframe:

```python
# Higher timeframes need larger swings to qualify as pivots
TIMEFRAME_SWING_SCALE = {
    '1m': 0.3, '5m': 0.5, '15m': 0.7, '1h': 1.0,
    '4h': 1.2, '1D': 1.5, '1W': 2.5, '1M': 4.0
}
```

All finders, validators, and scoring functions are timeframe-unaware — they operate on whatever bars are passed to them. The timeframe only affects the starting degree and swing thresholds.

---

## Performance Strategy

Naive recursion through 6 degrees × hundreds of candidates = combinatorial explosion. We control this with:

1. **Candidate pruning**: At each degree, only recurse into top 10 candidates (not all). The existing scoring already ranks well.

2. **Memoization**: Cache monowave and pivot results keyed by `(start_idx, end_idx, swing_pct)`. Many sub-waves overlap across alternate counts.

3. **Lazy sub-decomposition**: Only decompose the PRIMARY count fully (all degrees). For alternates, decompose 1 degree deep and estimate deeper quality from bar count + fibonacci alignment.

4. **Early termination**: If a candidate's parent score × best possible sub-wave multiplier can't beat the current best, skip its sub-decomposition.

5. **Target: < 3 seconds** for a 500-bar daily chart. Current engine runs in ~0.5s. Budget 6× for recursion across 4-5 effective degrees.

---

## API Output Format (New)

```json
{
  "symbol": "PLTR",
  "timeframe": "daily",
  "overall_confidence": 0.87,
  "current_position": "Cycle III, Primary (5), Intermediate [3]",
  "primary_count": {
    "degree": "CYCLE",
    "type": "impulse",
    "label": "I-II-III-IV-V",
    "confidence": 0.87,
    "waves": [
      {
        "label": "I",
        "start": {"idx": 0, "price": 9.50, "date": "2020-10-01"},
        "end": {"idx": 85, "price": 45.00, "date": "2021-01-27"},
        "direction": "up",
        "sub_count": {
          "degree": "PRIMARY",
          "type": "impulse",
          "label": "(1)-(2)-(3)-(4)-(5)",
          "waves": [
            {
              "label": "(1)",
              "start": {"idx": 0, "price": 9.50},
              "end": {"idx": 30, "price": 18.00},
              "sub_count": {
                "degree": "INTERMEDIATE",
                "type": "impulse",
                "label": "[1]-[2]-[3]-[4]-[5]",
                "waves": ["...further nesting..."]
              }
            }
          ]
        }
      },
      {
        "label": "II",
        "start": {"idx": 85, "price": 45.00},
        "end": {"idx": 200, "price": 14.00},
        "direction": "down",
        "sub_count": {
          "degree": "PRIMARY",
          "type": "zigzag",
          "label": "(A)-(B)-(C)",
          "waves": ["..."]
        }
      }
    ]
  },
  "alternate_counts": [
    {"degree": "CYCLE", "type": "leading_diagonal", "confidence": 0.72, "waves": ["..."]},
    {"degree": "CYCLE", "type": "impulse_ext_w3", "confidence": 0.68, "waves": ["..."]}
  ],
  "bar_labels": [
    {"idx": 0, "labels": {"CYCLE": "I", "PRIMARY": "(1)", "INTERMEDIATE": "[1]", "MINOR": "1"}},
    {"idx": 1, "labels": {"CYCLE": "I", "PRIMARY": "(1)", "INTERMEDIATE": "[1]", "MINOR": "1"}},
    {"idx": 30, "labels": {"CYCLE": "I", "PRIMARY": "(1)", "INTERMEDIATE": "[5]", "MINOR": "5"}}
  ]
}
```

The `bar_labels` array gives the frontend O(1) lookup for "what degree labels does bar X carry?" — essential for rendering labels at every pivot at every degree.

---

## Frontend Rendering Changes

### Current: Zigzag lines connecting pivots
The existing `drawWaveCount()` in `KLineChartOverlay.tsx` draws colored polylines from wave start to wave end, with numbered circles at pivot points.

### New: Wave labels at pivots, per degree

```
Strategy:
1. At each pivot point, stack labels vertically by degree
2. Each degree gets its own color (existing color scheme works)
3. Labels use standard notation: Roman, parenthesized, bracketed, plain, lowercase
4. User can toggle degree visibility (show/hide Minute, Minor, etc.)
5. On hover: highlight the entire wave at that degree
```

Visual comparison:

```
CURRENT ENGINE:
  Price ─────╱────╲────╱╲────╱──────
             1     2    3  4   5
             (one set of labels, one degree)

NEW ENGINE:
  Price ─────╱────╲────╱╲────╱──────
             III        IV    V        ← Cycle (gold)
             (3)  (4)  (5)            ← Primary (cyan)
             [3]  [4]  [5] [1][2][3]  ← Intermediate (gray)
```

### Active Pattern Zone (the WaveBasis "liquidity highlight")
When the engine identifies the current pattern being built (the most recent incomplete wave), the frontend renders a **semi-transparent shaded rectangle** over those candles — exactly like WaveBasis does:

- Orange/brown semi-transparent fill over the candle region
- Dashed border around the zone
- Pattern label above: "Building (1)-(2) of C" or "Wave [3] of (3) in progress"
- Sub-wave labels rendered inside the zone
- Zone updates in real-time as new bars arrive

This tells the trader instantly: "here's what's forming right now and what pattern it belongs to."

### Hover → Show Wave Lines
When the user hovers over any wave label (e.g., ((III))):
- Orange connecting lines appear showing that wave's internal sub-structure
- The wave label enlarges and highlights
- A tooltip shows: degree, pattern type, price change, fibonacci percentage
- All sub-wave labels within that wave brighten

This is the same behavior visible in the WaveBasis screenshots.

### Implementation approach:
- Keep the `drawWaveCount()` function but add a new `drawMultiDegreeLabels()` function
- Walk the WaveTree, collect all pivot points with their degree labels
- Render labels with vertical stacking and collision avoidance (existing collision logic from lines 516-534)
- Add `drawActiveZone()` — shaded rectangle over the currently-forming pattern
- Add hover detection: mouse proximity to labels triggers sub-structure line rendering
- Add a degree toggle panel in the UI sidebar

---

## Implementation Phases

### Phase 1 — Core Recursive Engine (Est: 2-3 days of Claude Code work)

**Goal:** `analyze_recursive()` produces correct WaveTree for any chart.

Changes to `elliott_wave.py`:
- Add `WaveNode`, `WaveDegree`, `WaveTree` dataclasses
- Add `DEGREE_SWING_MAP` mapping degree → asset-class → swing%
- Add `determine_top_degree()` function
- Add `analyze_recursive()` — the new main entry point
- Add memoization cache for monowaves/pivots
- Add `sub_wave_quality_multiplier()` — scores how cleanly sub-waves decompose
- Keep ALL existing finders, gates, and validation (called within the recursion)

The existing `analyze_multi_degree()` remains as a fallback / can be removed later.

### Phase 2 — API & Serialization (Est: 1 day)

Changes to `patterns.py`:
- New serializer `_ser_v3_tree()` that walks WaveTree → nested JSON
- Add `bar_labels` array generation
- New `/api/v3/patterns/{symbol}` endpoint (keep v2 endpoint working)
- `current_position` string generation from tree walk

### Phase 3 — Frontend Multi-Degree Labels (Est: 2-3 days)

Changes to `KLineChartOverlay.tsx`:
- New `drawMultiDegreeLabels()` function
- Degree toggle panel component
- Vertical label stacking at pivots
- Color coding per degree
- Hover highlighting of full wave spans
- Keep v2 rendering as fallback toggle

### Phase 4 — Alternate Counts & Polish (Est: 1-2 days)

- Side panel showing primary + 3 alternates with confidence bars
- Click an alternate to re-render chart with that count
- Current position display: "You are in Cycle III, Primary (4), Intermediate [c]"
- Performance optimization (lazy loading, web workers for heavy recursion)

---

## What This Solves

| Current Problem | How v3 Fixes It |
|----------------|-----------------|
| Single pattern, one degree | Recursive decomposition at every degree |
| Last 20-55% of chart uncovered | Full-chart coverage guaranteed (≥85% at top degree) |
| "One straight line" waves (PLTR, CRWV) | Sub-waves break every large wave into 5 or 3 sub-waves |
| No current position awareness | Tree walk tells you exactly where you are |
| Only 1-3 counts, same degree | Primary + 3 alternates, each with full sub-wave detail |
| Connecting lines (our look) | Wave labels at pivots (WaveBasis look) |

---

## Risk & Mitigation

**Risk: Combinatorial explosion in recursion**
→ Mitigated by candidate pruning (top 10), memoization, lazy alternates, early termination. Target < 3s.

**Risk: Sub-wave decomposition forces bad patterns at low degrees**
→ Mitigated by allowing "unresolved" segments. Not every 5-bar segment will form a valid pattern — that's fine, label it as a monowave at the terminal degree.

**Risk: Regressions in existing pattern quality**
→ Mitigated by keeping all V7+V8 gates intact. The finders and validation are unchanged — they're just called within the recursion instead of at the top level.

**Risk: Frontend rendering complexity**
→ Mitigated by phased rollout. Phase 1-2 work with existing v2 frontend (just render the top degree). Phase 3 adds the multi-degree labels.
