# Elliott Wave Engine v2 — Design Spec

## Problem

The current `pattern_engine/elliott_wave.py` (2,446 lines) produces nonsensical results because it treats price swings as waves. It brute-forces 5-swing combinations through O(n^5) nested loops, checks only 3 basic rules, and cannot distinguish corrective subtypes. It has no concept of internal wave structure, position-awareness, or multi-count ranking.

## Research Sources

- 16 Elliott Wave International PDFs (Frost & Prechter theory, lessons 1-34)
- Constance Brown, *Mastering Elliott Wave Principle* (2012), full book

---

## Unified Rule Set

### The 10 Basic Patterns

1. Simple five-wave impulse
2. Extending five-wave impulse (W3 most common in stocks, W5 in commodities)
3. Failure fifth wave (truncation) — only after W3 > 2.618× W1
4. Termination diagonal triangle (ending diagonal) — 3-3-3-3-3 internals, W5/C position only
5. Leading diagonal triangle (Type 2) — 5-3-5-3-5 internals, W1/A position only, very rare
6. Zigzag (5-3-5) — A=5 waves, B=3 waves (doesn't retrace all of A), C=5 waves
7. Flat (3-3-5) — A=3 waves, B≈100% of A, C=5 waves ending near A
8. Expanded flat (3-3-5) — B exceeds origin of A, C breaks beyond end of A
9. Contracting triangle (3-3-3-3-3) — converging trendlines, a-b-c-d-e
10. Expanding triangle (3-3-3-3-3) — diverging trendlines, very rare

### 3 Inviolable Rules (Impulse Waves)

| # | Rule | Detail |
|---|------|--------|
| 1 | W2 < 100% W1 | Wave 2 cannot retrace beyond the start of wave 1 |
| 2 | W3 not shortest | Compared to W1 and W5 (does NOT mean W3 must be longest) |
| 3 | W4 no W1 overlap | Wave 4 cannot enter the price territory of wave 1. Exception: leading diagonal type 2 (extremely rare, mostly Treasury markets) |

### Additional Impulse Rules (Brown)

4. Every impulse wave must contain an internal wave structure that subdivides into a separate and complete five-wave pattern
5. All corrective patterns ending with wave C develop an internal substructure of a five-wave pattern (triangles don't end at C, so they're exempt)

### Key Guidelines

| Guideline | Detail |
|-----------|--------|
| **Alternation** | If W2 is sharp (zigzag) → W4 sideways (flat/triangle), and vice versa. Applies to type AND depth |
| **Proportional time** | Corrective waves of same degree have proportional time. If W4 takes 2.618× the time of W2, it's likely a higher degree (Brown) |
| **Corrections target prior W4** | Corrections retrace to the prior fourth wave of one lesser degree |
| **Channeling** | Base channel: line from W1-start→W2-end + parallel through W1-end. Acceleration channel: W2→W4 + parallel through W3 |
| **Extensions** | When one wave extends, the other two actionary waves tend toward equality in price and time |
| **Fibonacci targets** | W3 common: 1.618×W1. W5 common: 1.0×W1 (or 0.618×W1 if W3 extended). C common: 1.0×A or 1.618×A |
| **W5 truncation condition** | Only valid after W3 > 2.618× W1 |
| **Strongest bar** | Single bar exceeding the longest bar in the prior trend signals trend damage (Brown/Gann) |

### Corrective Pattern Rules

| Pattern | Internal Structure | Key Identification |
|---------|-------------------|-------------------|
| Zigzag | 5-3-5 | ONLY pattern starting with 5 waves. B doesn't fully retrace A. C often 1.618×A |
| Flat | 3-3-5 | B retraces ~100% of A. C ≈ length of A. Sideways action |
| Expanded Flat | 3-3-5 | B exceeds origin of A. C breaks beyond end of A. Most common in financial markets |
| Running Flat | 3-3-5 | B exceeds origin of A. C fails to reach end of A (rare, implies very strong trend) |
| Contracting Triangle | 3-3-3-3-3 | Converging trendlines. NO leg may be a standalone 5-wave pattern. Labels d/e exclusive to triangles |
| Expanding Triangle | 3-3-3-3-3 | Diverging trendlines. Very rare |

### Triangle Position Rules (Critical)

- Triangles do NOT appear in wave 2 positions by themselves
- Triangles only in: W4, wave B (of larger correction), wave X (connector)
- No wave within a triangle may be a standalone five-wave pattern
- If first leg of apparent triangle is five waves → it's a zigzag, not a triangle
- Wave 'e' often creates its own small triangle
- **Thrust out of triangle**: resolution is strong and swift, usually retraces to origin

### Diagonal Triangle Rules

| Type | Position | Internals | Overlap | Slopes |
|------|----------|-----------|---------|--------|
| Termination (ending) | W5 or C only | 3-3-3-3-3 (all corrective) | W4 overlaps W1 | Each impulse wave flattens vs prior |
| Leading (Type 2) | W1 or A only | 5-3-5-3-5 (actionary = impulse) | W4 overlaps W1 | Very rare, mostly bonds |

### Extending Five-Wave Swing Counts (Brown)

Extensions must end with specific swing counts: 5, 9, 13, 17, 21... (each extension adds 4 more swings). Always adds pairs at front and back. 11 swings never end an impulse.

### Fibonacci Cluster Zones (Brown)

Multiple Fibonacci projections from different waves landing near the same price create high-probability target zones. When Fibonacci confluence zones are ignored by the market, look for a triangle developing.

### Balance/Proportion Pre-Filter (Brown)

Before counting waves, compare swing proportionality using "boxes" around swings. If a correction takes 2.618× the time of the comparable wave, it's likely a different degree. Visual proportion check helps filter invalid counts early.

### Oscillator Confirmation (Brown)

- RSI/Composite Index divergence confirms W5 endings
- RSI in corrective pullbacks falls to range of 40-45 in bull trends
- Momentum divergence between W3 and W5 validates completion
- Composite Index showing bullish divergence with price lows + RSI failing = corrective pattern, not new trend

### Do NOT Use Tick Data

Brown explicitly warns against using tick data for wave analysis — volume and sentiment are manipulated into artificial packages. Use time-based bars only.

---

## Architecture: Bottom-Up Hierarchical Detection

### Layer 1 — Pivot Detection (KEEP existing)

`detect_pivots()` and `detect_macro_pivots()` are structurally sound. Keep them. They produce the raw swing points.

### Layer 2 — MonoWave Identification

Convert pivot sequences into MonoWaves (single directional moves between adjacent pivots). Each MonoWave has: start_price, end_price, start_idx, end_idx, direction, bar_count, price_change, internal_bar_data.

### Layer 3 — Sub-Wave Structure Analysis (NEW — core innovation)

For each MonoWave, determine its internal structure:
- **Impulsive (5)**: Contains a recognizable 5-wave subdivision
- **Corrective (3)**: Contains overlapping, choppy, 3-wave structure
- **Indeterminate**: Too few bars or ambiguous

Method: Recursively apply pivot detection at smaller scale within the MonoWave's bar range. Check if internal pivots form valid 5-wave or 3-wave sequences. This is the step v1 completely lacks.

Implementation approach: Use a `classify_internal_structure(bars, start_idx, end_idx) -> "impulsive" | "corrective" | "indeterminate"` function that:
1. Detects sub-pivots within the bar range
2. If 4+ sub-pivots found: check if they satisfy 5-wave rules (W2<W1, W3 not shortest, W4 no W1 overlap)
3. If yes → "impulsive". If overlapping/choppy → "corrective"
4. If <4 sub-pivots → use proportional heuristics (Brown's box method)

### Layer 4 — Pattern Composition

Combine MonoWaves with known internal structures into candidate patterns:

**Impulse candidates**: Find 5 consecutive MonoWaves alternating direction where:
- Waves 1, 3, 5 are actionary (same direction as overall pattern)
- Waves 2, 4 are corrective (opposite direction)
- Internal structure check: W1=5, W2=3, W3=5, W4=3, W5=5
- Three rules pass
- Extension detection: if one actionary wave is >1.618× others, mark as extended

**Zigzag candidates**: Find 3 consecutive MonoWaves (A-B-C) where:
- A has impulsive internal structure (5 waves)
- B is corrective, doesn't retrace all of A
- C has impulsive internal structure (5 waves)
- C often ≈ 1.0× or 1.618× A

**Flat candidates**: Find 3 consecutive MonoWaves where:
- A has corrective internal structure (3 waves)
- B is corrective, retraces ~90-105% of A → regular flat
- B exceeds origin of A → expanded flat
- C has impulsive internal structure (5 waves)

**Triangle candidates**: Find 5 consecutive MonoWaves where:
- ALL have corrective internal structure (3 waves each)
- No leg is a standalone five-wave pattern
- Converging or diverging trendlines
- Position check: only valid in W4/B/X positions

**Diagonal candidates**: Find 5 consecutive MonoWaves with overlapping W4/W1 where:
- Termination: all internals corrective (3-3-3-3-3), only in W5/C
- Leading: actionary internals impulsive (5-3-5-3-5), only in W1/A

### Layer 5 — Validation & Scoring

For each candidate pattern, compute a confidence score:

| Factor | Weight | Detail |
|--------|--------|--------|
| Rule compliance | 30% | All inviolable rules pass (binary gate — fail = discard) |
| Internal structure match | 25% | How well sub-wave structures match expected pattern |
| Fibonacci alignment | 15% | Key ratios (W3/W1, W5/W1, C/A) near Fibonacci values |
| Alternation | 10% | W2/W4 alternate in type and depth |
| Proportionality | 10% | Time and price proportions follow Fibonacci relationships |
| Channel fit | 5% | How well price respects the base/acceleration channel |
| Position validity | 5% | Pattern appears in valid position (triangles in W4/B, diagonals in W5/C or W1/A) |

### Layer 6 — Multi-Count Ranking (NEW)

Never output a single interpretation. Always maintain multiple valid counts ranked by confidence. The top count is "preferred" but alternates are shown.

Output up to 3 counts:
1. **Preferred**: Highest confidence score
2. **Alternate 1**: Second-highest, structurally different interpretation
3. **Alternate 2**: Third option if significantly different from above

### Position Assessment

Determine where we are in the current pattern:
- If in W3: strong trend continuation expected
- If in W4: correction developing, look for W5 opportunity
- If in W5: trend exhaustion approaching, look for reversal
- If in correction (A/B/C): determine which leg, project completion

For each position, provide:
- **Direction**: bullish/bearish
- **Phase**: impulsive/corrective
- **Next expected move**: price range and direction
- **Key invalidation level**: price that would break the count

---

## File Changes

| File | Action | Lines (est.) |
|------|--------|-------------|
| `pattern_engine/elliott_wave.py` | REWRITE lines 392-1992 | ~1200 lines replaced |
| `pattern_engine/elliott_wave.py` | KEEP lines 1-391 (pivots, dataclasses, enums) | ~390 lines kept |
| `pattern_engine/elliott_wave.py` | KEEP `_validate_impulse_rules` (lines 1073-1118) | ~45 lines kept |
| `pattern_engine/elliott_wave.py` | KEEP channel computation (lines 2245-2447) | ~200 lines kept, used for scoring |

No new files needed. All changes contained within the existing module.

### Data Flow

```
OHLCV bars
  → detect_pivots() → Pivot[]
  → detect_macro_pivots() → Pivot[]
  → build_monowaves(pivots) → MonoWave[]
  → classify_internal_structure(monowave) → annotated MonoWave[]
  → find_impulse_candidates() → WaveCount[]
  → find_corrective_candidates() → WaveCount[]  (zigzag, flat, triangle, diagonal)
  → validate_and_score(candidates) → scored WaveCount[]
  → rank_counts(scored) → top 3 WaveCount[] with positions
  → assess_current_position(top_counts) → position dict with direction, targets, invalidation
```

### WaveCount Dataclass (Enhanced)

```python
@dataclass
class WaveCount:
    wave_type: WaveType          # IMPULSE, ZIGZAG, FLAT, EXPANDED_FLAT, TRIANGLE, etc.
    waves: list[MonoWave]        # The component waves
    degree: str                  # "Primary", "Intermediate", "Minor", etc.
    direction: WaveDirection     # UP or DOWN
    internal_structures: list[str]  # ["5","3","5","3","5"] for impulse
    confidence: float            # 0.0-1.0 composite score
    fibonacci_targets: dict      # projected price targets
    invalidation_level: float    # price that breaks this count
    position_in_parent: str      # "W1", "W3", "A", "B", etc. if known
    channel: dict | None         # base/acceleration channel lines
    alternation_quality: str     # "good", "fair", "poor"
    extension_wave: int | None   # which wave extends (1, 3, or 5)
```

### New WaveType Variants

Add to existing enum:
```python
class WaveType(str, Enum):
    IMPULSE = "impulse"
    IMPULSE_EXTENDED_3 = "impulse_ext3"
    IMPULSE_EXTENDED_5 = "impulse_ext5"
    ZIGZAG = "zigzag"
    FLAT = "flat"
    EXPANDED_FLAT = "expanded_flat"
    RUNNING_FLAT = "running_flat"
    CONTRACTING_TRIANGLE = "contracting_triangle"
    EXPANDING_TRIANGLE = "expanding_triangle"
    ENDING_DIAGONAL = "ending_diagonal"
    LEADING_DIAGONAL = "leading_diagonal"
    TRUNCATED_5TH = "truncated_5th"
    DOUBLE_ZIGZAG = "double_zigzag"  # WXY where W and Y are zigzags
```

---

## API Output Schema

The existing `analyze()` method signature stays the same. Internal return structure enhanced:

```python
{
    "symbol": "AAPL",
    "timeframe": "daily",
    "counts": [
        {
            "rank": 1,
            "label": "Preferred",
            "type": "impulse_ext3",
            "direction": "up",
            "confidence": 0.82,
            "waves": [
                {"label": "1", "start": 150.0, "end": 165.0, "internal": "5-wave", "bars": 12},
                {"label": "2", "start": 165.0, "end": 158.0, "internal": "zigzag", "bars": 8},
                {"label": "3", "start": 158.0, "end": 195.0, "internal": "5-wave (extended)", "bars": 25},
                {"label": "4", "start": 195.0, "end": 185.0, "internal": "flat", "bars": 10},
                {"label": "5", "start": 185.0, "end": null, "internal": "developing", "bars": 5}
            ],
            "position": {
                "current_wave": "5",
                "phase": "impulsive",
                "direction": "bullish",
                "message": "Wave 5 developing — trend exhaustion approaching",
                "targets": [198.5, 202.0],
                "invalidation": 185.0
            },
            "alternation": "good — W2 sharp (zigzag), W4 sideways (flat)",
            "extension": {"wave": 3, "ratio": 2.31},
            "channel": {"base_upper": [...], "base_lower": [...]}
        },
        {
            "rank": 2,
            "label": "Alternate",
            "type": "zigzag",
            "direction": "down",
            "confidence": 0.41,
            ...
        }
    ],
    "trend": "up",
    "degree": "Intermediate"
}
```

---

## What Gets Deleted vs Kept

### DELETE (complete rewrite):
- `find_impulse_waves()` — brute-force 5-loop approach
- `find_corrective_waves()` — direction-only check
- `find_triangle_waves()` — no position check, no internal structure
- `find_ending_diagonal()` — no 3-3-3-3-3 check
- `_validate_corrective()` — 20 lines, insufficient
- `_score_impulse_fibonacci()` — too many target ratios
- `_assess_current_position()` — first-match-wins, misleading
- `_build_wave_list()` — clever but can't fix bad candidates

### KEEP (refactor/integrate):
- `Pivot`, `MonoWave`, `WaveCount` dataclasses — enhance, don't replace
- `WaveType` enum — expand with new variants
- `detect_pivots()`, `detect_macro_pivots()` — working correctly
- `_validate_impulse_rules()` — correct 3-rule check, use as gate
- Channel computation — use for scoring, not just visualization
- `DEGREE_LABELS` — keep as-is

---

## Testing Strategy

1. **Unit tests per pattern type**: Feed known synthetic wave data, verify correct classification
2. **Regression on real data**: SPY, AAPL, QQQ — verify no "BULLISH IMPULSE REVERSE" nonsense
3. **Edge cases**: truncated 5ths, running flats, leading diagonals
4. **Multi-count verification**: Always produces ≥2 counts with different interpretations
5. **Position assessment**: Current wave identification matches visual chart inspection

---

## References

- Visual mocks: `docs/superpowers/references/ew-redesign-overview.html`
- Pattern interpretation cards: `docs/superpowers/references/ew-pattern-interpretations.html`
