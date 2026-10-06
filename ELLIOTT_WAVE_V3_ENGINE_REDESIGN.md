# Elliott Wave v3 — Multi-Degree Recursive Wave Counting Engine Redesign

## Overview

This is a **fundamental redesign** of the Elliott Wave engine from single-pattern detection (v2) to a true recursive multi-degree wave counter like WaveBasis.com. The engine must:

1. Recursively decompose the chart from the highest applicable degree down to Minute/Subminute
2. Label every relevant bar at every degree with proper EW notation
3. Identify a "starting point" (structural extreme) and count from there to the current candle
4. Show primary + 3 alternate counts with confidence scores
5. Compute forward Fibonacci projections (price targets + invalidation levels)
6. Display an "active pattern zone" highlighting the currently-forming pattern
7. Support hover-to-reveal sub-wave structure
8. Work identically on any timeframe (1m, 5m, 15m, 1h, 4h, 1D, 1W, 1M)
9. Provide a full theming/appearance system (chart background, candle colors, wave colors, toggleable overlays)

**~70% of existing code is reused.** All finders, gates, validators, scoring, monowaves, and pivots are unchanged — they're called within the recursive framework instead of at the top level.

Read the architecture plan at `EW_ENGINE_V2_REDESIGN_PLAN.md` in the project root for the full design. Open the visual mock at `EW_ENGINE_V2_MOCK.html` in a browser to see the target look & feel.

---

## PHASE 1 — Core Recursive Engine

### File: `.worktrees/ew-engine-v2/pattern_engine/elliott_wave.py`

### 1A. Add new data structures (after line ~160, after existing dataclasses)

Add these new dataclasses/enums alongside the existing `WaveType`, `WaveDirection`, `Pivot`, `MonoWave`, `WaveCount`:

```python
class WaveDegree(Enum):
    """Elliott Wave degree hierarchy — from largest to smallest."""
    SUPERCYCLE = 7
    CYCLE = 6
    PRIMARY = 5
    INTERMEDIATE = 4
    MINOR = 3
    MINUTE = 2
    SUBMINUTE = 1
    MICRO = 0  # terminal — no further decomposition

# Standard EW label notation by degree
DEGREE_LABELS = {
    WaveDegree.SUPERCYCLE: {
        'impulse': ['((I))', '((II))', '((III))', '((IV))', '((V))'],
        'corrective': ['((A))', '((B))', '((C))'],
    },
    WaveDegree.CYCLE: {
        'impulse': ['(I)', '(II)', '(III)', '(IV)', '(V)'],
        'corrective': ['(A)', '(B)', '(C)'],
    },
    WaveDegree.PRIMARY: {
        'impulse': ['I', 'II', 'III', 'IV', 'V'],
        'corrective': ['A', 'B', 'C'],
    },
    WaveDegree.INTERMEDIATE: {
        'impulse': ['(1)', '(2)', '(3)', '(4)', '(5)'],
        'corrective': ['(a)', '(b)', '(c)'],
    },
    WaveDegree.MINOR: {
        'impulse': ['[1]', '[2]', '[3]', '[4]', '[5]'],
        'corrective': ['[a]', '[b]', '[c]'],
    },
    WaveDegree.MINUTE: {
        'impulse': ['1', '2', '3', '4', '5'],
        'corrective': ['a', 'b', 'c'],
    },
    WaveDegree.SUBMINUTE: {
        'impulse': ['i', 'ii', 'iii', 'iv', 'v'],
        'corrective': ['a', 'b', 'c'],  # lowercase
    },
}

@dataclass
class WaveNode:
    """A single wave at a specific degree in the recursive tree."""
    degree: WaveDegree
    wave_type: WaveType  # IMPULSE, ZIGZAG, FLAT, DIAGONAL, etc.
    label: str           # e.g., "(1)", "[A]", "iii" — from DEGREE_LABELS
    start_idx: int       # bar index in the FULL dataframe
    end_idx: int
    start_price: float
    end_price: float
    direction: WaveDirection
    sub_waves: List['WaveNode']  # children at next lower degree
    confidence: float   # 0.0-1.0
    is_complete: bool = True  # False if this wave is still forming

@dataclass
class WaveTree:
    """Complete multi-degree wave count for a symbol/timeframe."""
    symbol: str
    timeframe: str
    root: WaveNode                   # top-level wave node
    alternates: List[WaveNode]       # top 3 alternate root counts
    starting_point_idx: int          # bar index of the structural extreme anchor
    starting_point_price: float
    current_position: str            # e.g., "Primary III, Intermediate (3), Minor [3]"
    overall_confidence: float
    projections: List['Projection']  # forward fibonacci targets

@dataclass
class Projection:
    """A forward price projection based on Fibonacci extensions."""
    target_price: float
    fib_ratio: float       # e.g., 1.618
    fib_label: str         # e.g., "161.8% ext of W1"
    from_wave_label: str   # which wave this projection is based on
    is_invalidation: bool = False  # True for risk/stop levels
    pct_from_current: float = 0.0  # percentage distance from current price
```

### 1B. Add degree-to-swing-threshold mapping (after the new dataclasses)

This maps each degree level to the swing percentage needed to detect meaningful pivots at that scale, per asset class:

```python
# Swing thresholds by degree and asset class
# Format: { WaveDegree: { 'equity': pct, 'forex': pct, 'crypto': pct } }
DEGREE_SWING_MAP = {
    WaveDegree.SUPERCYCLE: {'equity': 0.50, 'forex': 0.25, 'crypto': 0.60},
    WaveDegree.CYCLE:      {'equity': 0.35, 'forex': 0.15, 'crypto': 0.45},
    WaveDegree.PRIMARY:    {'equity': 0.20, 'forex': 0.08, 'crypto': 0.30},
    WaveDegree.INTERMEDIATE: {'equity': 0.12, 'forex': 0.05, 'crypto': 0.18},
    WaveDegree.MINOR:      {'equity': 0.06, 'forex': 0.025, 'crypto': 0.10},
    WaveDegree.MINUTE:     {'equity': 0.03, 'forex': 0.012, 'crypto': 0.05},
    WaveDegree.SUBMINUTE:  {'equity': 0.015, 'forex': 0.006, 'crypto': 0.025},
}

# Timeframe → starting degree and how many levels to recurse
TIMEFRAME_DEGREE_MAP = {
    '1':   (WaveDegree.MINUTE, 3),        # 1-min: Minute → Subminute → Micro
    '5':   (WaveDegree.MINOR, 3),         # 5-min: Minor → Minute → Subminute
    '15':  (WaveDegree.MINOR, 4),
    '60':  (WaveDegree.INTERMEDIATE, 4),  # 1-hour
    '240': (WaveDegree.INTERMEDIATE, 4),  # 4-hour
    'D':   (WaveDegree.PRIMARY, 5),       # Daily
    'W':   (WaveDegree.CYCLE, 5),         # Weekly
    'M':   (WaveDegree.SUPERCYCLE, 5),    # Monthly
}

# Timeframe swing scale multiplier (higher TF = need larger swings)
TIMEFRAME_SWING_SCALE = {
    '1': 0.3, '5': 0.5, '15': 0.7, '60': 1.0,
    '240': 1.2, 'D': 1.5, 'W': 2.5, 'M': 4.0
}
```

### 1C. Add the starting point detector (new function)

This finds the most relevant structural extreme to anchor the wave count. The count starts here and runs to the current candle — it does NOT have to start at bar 0.

```python
def find_starting_point(df: pd.DataFrame, asset_class: str, timeframe: str) -> Tuple[int, float, bool]:
    """
    Find the structural extreme (major high or low) that best anchors a wave count.
    
    Returns: (bar_index, price, is_low)
    - is_low=True means the count starts from a bottom (bullish count)
    - is_low=False means the count starts from a top (bearish count)
    
    Strategy:
    1. Find all major pivots using a medium swing threshold
    2. For each pivot, measure the magnitude of the move from that pivot to the chart end
    3. Penalize pivots that are too far back (less relevant)
    4. Pick the pivot that produces the best wave count when used as anchor
    """
    top_degree, _ = TIMEFRAME_DEGREE_MAP.get(timeframe, (WaveDegree.PRIMARY, 5))
    swing_pct = DEGREE_SWING_MAP[top_degree][asset_class]
    tf_scale = TIMEFRAME_SWING_SCALE.get(timeframe, 1.5)
    
    pivots = detect_major_pivots(df, min_swing_pct=swing_pct * tf_scale * 0.7, max_pivots=30)
    
    if not pivots:
        # Fallback: use the absolute min or max of the full chart
        min_idx = df['low'].idxmin()
        max_idx = df['high'].idxmax()
        last_price = df['close'].iloc[-1]
        if last_price > df['close'].iloc[0]:
            return (min_idx, df['low'].iloc[min_idx], True)
        else:
            return (max_idx, df['high'].iloc[max_idx], False)
    
    best_pivot = None
    best_score = -1
    last_price = df['close'].iloc[-1]
    total_bars = len(df)
    
    for piv in pivots:
        # Only consider pivots in the first 70% of the chart as starting points
        if piv.idx > total_bars * 0.7:
            continue
        
        # Magnitude: how big is the move from this pivot to current price?
        move_pct = abs(last_price - piv.price) / piv.price
        
        # Recency: prefer pivots that aren't at the very start (more structural)
        # but also not too late (need room for a full wave count)
        bars_from_start_pct = piv.idx / total_bars
        position_score = 1.0 - abs(bars_from_start_pct - 0.2)  # sweet spot around 20% into chart
        
        # Strength: prefer pivots with high strength (clear reversal)
        strength = piv.strength if hasattr(piv, 'strength') else 1.0
        
        # Direction check: if price rallied from here, it's a low; if dropped, it's a high
        bars_remaining = total_bars - piv.idx
        
        score = move_pct * 0.5 + position_score * 0.3 + strength * 0.2
        
        # Bonus: if this pivot is a clear extreme (lowest low or highest high in its region)
        region_start = max(0, piv.idx - 20)
        region_end = min(total_bars, piv.idx + 20)
        if piv.is_high and piv.price >= df['high'].iloc[region_start:region_end].max() * 0.99:
            score *= 1.2
        elif not piv.is_high and piv.price <= df['low'].iloc[region_start:region_end].min() * 1.01:
            score *= 1.2
        
        if score > best_score:
            best_score = score
            best_pivot = piv
    
    if best_pivot is None:
        # Fallback
        min_idx = df['low'].idxmin()
        return (min_idx, df['low'].iloc[min_idx], True)
    
    return (best_pivot.idx, best_pivot.price, not best_pivot.is_high)
```

### 1D. Add the core recursive analysis function (new main entry point)

This is the heart of v3. It replaces `analyze_multi_degree()` as the primary entry point.

```python
def analyze_recursive(
    df: pd.DataFrame,
    degree: WaveDegree,
    asset_class: str = 'equity',
    timeframe: str = 'D',
    direction_hint: Optional[WaveDirection] = None,
    max_depth: int = 5,
    _depth: int = 0,
    _memo: Optional[Dict] = None,
) -> Optional[WaveNode]:
    """
    Core recursive multi-degree wave analysis.
    
    Decomposes `df` into waves at `degree`, then recursively decomposes
    each wave's bar range at degree-1.
    
    Args:
        df: Price bars (must have 'open', 'high', 'low', 'close' columns)
        degree: Current wave degree to analyze at
        asset_class: 'equity', 'forex', or 'crypto'
        timeframe: '1', '5', '15', '60', '240', 'D', 'W', 'M'
        direction_hint: Expected direction (from parent wave context)
        max_depth: Maximum recursion depth
        _depth: Current recursion depth (internal)
        _memo: Memoization cache (internal)
    
    Returns:
        WaveNode with sub_waves populated recursively, or None if no valid pattern found
    """
    if _memo is None:
        _memo = {}
    
    # Base case: too few bars or at terminal degree or max depth
    if len(df) < 5 or degree.value <= WaveDegree.MICRO.value or _depth >= max_depth:
        return None
    
    # Memoization key
    memo_key = (df.index[0], df.index[-1], degree.value)
    if memo_key in _memo:
        return _memo[memo_key]
    
    # Get swing thresholds for this degree
    tf_scale = TIMEFRAME_SWING_SCALE.get(timeframe, 1.5)
    swing_pct = DEGREE_SWING_MAP.get(degree, DEGREE_SWING_MAP[WaveDegree.MINOR])[asset_class]
    swing_pct *= tf_scale
    
    # --- Use existing infrastructure to find candidates ---
    
    # Build monowaves at this degree's scale
    atr_mult = swing_pct * 10  # rough mapping of swing% to ATR multiplier
    monowaves = build_monowaves(df, min_swing_pct=swing_pct)
    
    # Detect pivots at this degree's scale
    pivots = detect_major_pivots(df, min_swing_pct=swing_pct, max_pivots=20)
    
    if len(monowaves) < 2 and len(pivots) < 3:
        _memo[memo_key] = None
        return None
    
    # Find ALL candidate patterns using existing finders (unchanged)
    candidates = []
    
    if len(monowaves) >= 5:
        candidates += find_impulse_candidates(monowaves, df)
    
    if len(pivots) >= 6:
        candidates += find_pivot_based_impulse_candidates(pivots, df)
    
    if len(monowaves) >= 3:
        candidates += find_zigzag_candidates(monowaves, df)
        candidates += find_flat_candidates(monowaves, df)
    
    if len(monowaves) >= 5:
        candidates += find_diagonal_candidates(monowaves, df)
    
    if not candidates:
        _memo[memo_key] = None
        return None
    
    # Validate and score using existing gates + scoring (unchanged)
    scored = []
    for c in candidates:
        result = validate_and_score(c, df, total_bars=len(df))
        if result and result.get('score', 0) > 0:
            scored.append(result)
    
    if not scored:
        _memo[memo_key] = None
        return None
    
    # Sort by score, take top 10 for recursive decomposition (pruning)
    scored.sort(key=lambda c: c.get('score', 0), reverse=True)
    top_candidates = scored[:10]
    
    # --- KEY NEW STEP: Recursively decompose sub-waves ---
    
    best_node = None
    best_total_score = -1
    alternates = []
    
    next_degree = WaveDegree(degree.value - 1)
    
    for cand in top_candidates:
        waves = cand.get('waves', [])
        if not waves:
            continue
        
        sub_nodes = []
        sub_quality_sum = 0
        sub_count = 0
        
        for i, wave in enumerate(waves):
            w_start = wave.get('start_idx', wave.get('start', {}).get('idx', 0))
            w_end = wave.get('end_idx', wave.get('end', {}).get('idx', 0))
            
            if w_end - w_start < 5:
                # Too few bars to decompose — create a terminal leaf node
                is_impulse = cand.get('wave_type', WaveType.IMPULSE) in [
                    WaveType.IMPULSE, WaveType.IMPULSE_EXT_W1, WaveType.IMPULSE_EXT_W3,
                    WaveType.IMPULSE_EXT_W5, WaveType.LEADING_DIAGONAL, WaveType.ENDING_DIAGONAL
                ]
                label_set = 'impulse' if is_impulse else 'corrective'
                labels = DEGREE_LABELS.get(degree, DEGREE_LABELS[WaveDegree.MINUTE])[label_set]
                label = labels[i] if i < len(labels) else str(i + 1)
                
                sub_nodes.append(WaveNode(
                    degree=degree,
                    wave_type=cand.get('wave_type', WaveType.IMPULSE),
                    label=label,
                    start_idx=w_start,
                    end_idx=w_end,
                    start_price=df['close'].iloc[w_start] if w_start < len(df) else 0,
                    end_price=df['close'].iloc[w_end] if w_end < len(df) else 0,
                    direction=WaveDirection.UP if df['close'].iloc[min(w_end, len(df)-1)] > df['close'].iloc[w_start] else WaveDirection.DOWN,
                    sub_waves=[],
                    confidence=cand.get('score', 0.5),
                ))
                continue
            
            # Recurse into this wave's bar range at the next lower degree
            sub_df = df.iloc[w_start:w_end + 1].copy()
            sub_df = sub_df.reset_index(drop=True)
            
            wave_dir = WaveDirection.UP if df['close'].iloc[min(w_end, len(df)-1)] > df['close'].iloc[w_start] else WaveDirection.DOWN
            
            sub_tree = analyze_recursive(
                sub_df, next_degree, asset_class, timeframe,
                direction_hint=wave_dir,
                max_depth=max_depth,
                _depth=_depth + 1,
                _memo=_memo,
            )
            
            # Build the wave node
            is_impulse = cand.get('wave_type', WaveType.IMPULSE) in [
                WaveType.IMPULSE, WaveType.IMPULSE_EXT_W1, WaveType.IMPULSE_EXT_W3,
                WaveType.IMPULSE_EXT_W5, WaveType.LEADING_DIAGONAL, WaveType.ENDING_DIAGONAL
            ]
            label_set = 'impulse' if is_impulse else 'corrective'
            labels = DEGREE_LABELS.get(degree, DEGREE_LABELS[WaveDegree.MINUTE])[label_set]
            label = labels[i] if i < len(labels) else str(i + 1)
            
            child_sub_waves = sub_tree.sub_waves if sub_tree else []
            child_confidence = sub_tree.confidence if sub_tree else 0.3
            
            sub_nodes.append(WaveNode(
                degree=degree,
                wave_type=cand.get('wave_type', WaveType.IMPULSE),
                label=label,
                start_idx=w_start,
                end_idx=w_end,
                start_price=df['close'].iloc[w_start] if w_start < len(df) else 0,
                end_price=df['close'].iloc[min(w_end, len(df)-1)] if w_end < len(df) else 0,
                direction=wave_dir,
                sub_waves=child_sub_waves,
                confidence=child_confidence,
            ))
            
            if sub_tree:
                sub_quality_sum += child_confidence
                sub_count += 1
        
        if not sub_nodes:
            continue
        
        # Compute total score: original score boosted by sub-wave decomposition quality
        sub_quality_avg = (sub_quality_sum / sub_count) if sub_count > 0 else 0.3
        sub_quality_mult = 0.7 + 0.3 * sub_quality_avg  # range 0.7-1.0
        total_score = cand.get('score', 0) * sub_quality_mult
        
        # Build the parent node for this candidate
        node = WaveNode(
            degree=degree,
            wave_type=cand.get('wave_type', WaveType.IMPULSE),
            label='root',  # gets overwritten by parent
            start_idx=sub_nodes[0].start_idx,
            end_idx=sub_nodes[-1].end_idx,
            start_price=sub_nodes[0].start_price,
            end_price=sub_nodes[-1].end_price,
            direction=WaveDirection.UP if sub_nodes[-1].end_price > sub_nodes[0].start_price else WaveDirection.DOWN,
            sub_waves=sub_nodes,
            confidence=total_score,
        )
        
        if total_score > best_total_score:
            if best_node:
                alternates.append(best_node)
            best_node = node
            best_total_score = total_score
        else:
            alternates.append(node)
    
    _memo[memo_key] = best_node
    return best_node
```

### 1E. Add the projection engine (new function)

```python
def compute_projections(tree: WaveTree, current_price: float) -> List[Projection]:
    """
    Compute Fibonacci forward projections based on the wave tree.
    Walks the tree to find the current incomplete wave and projects targets.
    """
    projections = []
    
    if not tree.root or not tree.root.sub_waves:
        return projections
    
    # Find the deepest incomplete wave (the one we're currently inside)
    def find_current_and_parent(node, parent=None):
        if not node.sub_waves:
            return node, parent
        # The last sub-wave is the most recent
        last = node.sub_waves[-1]
        return find_current_and_parent(last, node)
    
    current_wave, parent = find_current_and_parent(tree.root)
    
    if not parent or not parent.sub_waves:
        return projections
    
    subs = parent.sub_waves
    direction = 1 if parent.direction == WaveDirection.UP else -1
    
    # Determine which wave we're in and compute projections
    wave_idx = len(subs) - 1  # index of current wave within parent
    
    if parent.wave_type in [WaveType.IMPULSE, WaveType.IMPULSE_EXT_W1,
                            WaveType.IMPULSE_EXT_W3, WaveType.IMPULSE_EXT_W5,
                            WaveType.LEADING_DIAGONAL, WaveType.ENDING_DIAGONAL]:
        
        if wave_idx >= 2 and len(subs) >= 2:
            # We have at least W1 and W2 completed
            w1_range = abs(subs[0].end_price - subs[0].start_price)
            w2_end = subs[1].end_price
            
            if wave_idx == 2:
                # Currently in W3 — project W3 targets from W2 end
                for fib, label in [(1.0, '100%'), (1.618, '161.8%'), (2.618, '261.8%')]:
                    target = w2_end + (w1_range * fib * direction)
                    pct = ((target - current_price) / current_price) * 100
                    projections.append(Projection(
                        target_price=round(target, 2),
                        fib_ratio=fib,
                        fib_label=f'{label} ext of W1',
                        from_wave_label=subs[0].label,
                        pct_from_current=round(pct, 1),
                    ))
                
                # Invalidation: W3 cannot go below W1 start (for bullish)
                invalidation = subs[0].start_price
                inv_pct = ((invalidation - current_price) / current_price) * 100
                projections.append(Projection(
                    target_price=round(invalidation, 2),
                    fib_ratio=0,
                    fib_label='W1 start (invalidation)',
                    from_wave_label=subs[0].label,
                    is_invalidation=True,
                    pct_from_current=round(inv_pct, 1),
                ))
            
            elif wave_idx == 4 and len(subs) >= 4:
                # Currently in W5 — project from W1-W3 range
                w1_to_w3_range = abs(subs[2].end_price - subs[0].start_price)
                w4_end = subs[3].end_price
                for fib, label in [(0.618, '61.8%'), (1.0, '100%'), (1.618, '161.8%')]:
                    target = w4_end + (w1_to_w3_range * fib * direction)
                    pct = ((target - current_price) / current_price) * 100
                    projections.append(Projection(
                        target_price=round(target, 2),
                        fib_ratio=fib,
                        fib_label=f'{label} of W1-W3',
                        from_wave_label=f'{subs[0].label}-{subs[2].label}',
                        pct_from_current=round(pct, 1),
                    ))
                
                # W5 invalidation: shouldn't go below W4 start
                invalidation = subs[3].start_price if direction > 0 else subs[3].start_price
                inv_pct = ((invalidation - current_price) / current_price) * 100
                projections.append(Projection(
                    target_price=round(invalidation, 2),
                    fib_ratio=0,
                    fib_label='W4 start (invalidation)',
                    from_wave_label=subs[3].label,
                    is_invalidation=True,
                    pct_from_current=round(inv_pct, 1),
                ))
    
    elif parent.wave_type in [WaveType.ZIGZAG, WaveType.FLAT, WaveType.DOUBLE_ZIGZAG]:
        if wave_idx == 2 and len(subs) >= 2:
            # In wave C — project C targets
            a_range = abs(subs[0].end_price - subs[0].start_price)
            b_end = subs[1].end_price
            for fib, label in [(1.0, '100%'), (1.618, '161.8%')]:
                target = b_end + (a_range * fib * direction)
                pct = ((target - current_price) / current_price) * 100
                projections.append(Projection(
                    target_price=round(target, 2),
                    fib_ratio=fib,
                    fib_label=f'{label} of Wave A',
                    from_wave_label=subs[0].label,
                    pct_from_current=round(pct, 1),
                ))
    
    return projections
```

### 1F. Add the new main entry point (new function — top level)

This replaces `analyze_multi_degree()` as the entry point. **Do NOT delete `analyze_multi_degree()`** — keep it as a fallback. Add this new function:

```python
def analyze_wave_tree(
    df: pd.DataFrame,
    symbol: str = '',
    asset_class: str = 'equity',
    timeframe: str = 'D',
) -> Optional[WaveTree]:
    """
    Main entry point for v3 multi-degree recursive wave analysis.
    
    1. Finds the best starting point (structural extreme)
    2. Recursively decomposes from starting point to current bar
    3. Computes projections and current position
    4. Returns a complete WaveTree with alternates
    
    This replaces analyze_multi_degree() as the primary analysis function.
    """
    if len(df) < 10:
        return None
    
    # Determine asset class from symbol if not specified
    if not asset_class:
        asset_class = _detect_asset_class(symbol)
    
    # Determine top degree from timeframe
    top_degree, max_depth = TIMEFRAME_DEGREE_MAP.get(timeframe, (WaveDegree.PRIMARY, 5))
    
    # Dynamically adjust top degree based on chart data
    bar_count = len(df)
    price_range_pct = (df['high'].max() - df['low'].min()) / df['low'].min() if df['low'].min() > 0 else 0
    
    # If very few bars, step down the degree
    if bar_count < 50:
        top_degree = WaveDegree(max(top_degree.value - 2, WaveDegree.MINUTE.value))
        max_depth = min(max_depth, 3)
    elif bar_count < 100:
        top_degree = WaveDegree(max(top_degree.value - 1, WaveDegree.MINOR.value))
        max_depth = min(max_depth, 4)
    
    # Find starting point
    start_idx, start_price, is_bullish_start = find_starting_point(df, asset_class, timeframe)
    
    # Slice dataframe from starting point to end
    analysis_df = df.iloc[start_idx:].copy()
    analysis_df = analysis_df.reset_index(drop=True)
    
    if len(analysis_df) < 5:
        return None
    
    # Run recursive analysis
    memo = {}
    root = analyze_recursive(
        analysis_df, top_degree, asset_class, timeframe,
        direction_hint=WaveDirection.UP if is_bullish_start else WaveDirection.DOWN,
        max_depth=max_depth,
        _memo=memo,
    )
    
    if not root:
        # Fallback: try one degree lower
        root = analyze_recursive(
            analysis_df, WaveDegree(top_degree.value - 1), asset_class, timeframe,
            direction_hint=WaveDirection.UP if is_bullish_start else WaveDirection.DOWN,
            max_depth=max_depth - 1,
            _memo=memo,
        )
    
    if not root:
        return None
    
    # Adjust all indices back to full-dataframe coordinates
    def adjust_indices(node, offset):
        if not node:
            return
        node.start_idx += offset
        node.end_idx += offset
        for sw in node.sub_waves:
            adjust_indices(sw, offset)
    
    adjust_indices(root, start_idx)
    
    # Identify current position
    current_position = _identify_current_position(root)
    
    # Compute projections
    current_price = df['close'].iloc[-1]
    tree = WaveTree(
        symbol=symbol,
        timeframe=timeframe,
        root=root,
        alternates=[],  # populated from analyze_recursive's internal alternates
        starting_point_idx=start_idx,
        starting_point_price=start_price,
        current_position=current_position,
        overall_confidence=root.confidence,
        projections=[],
    )
    
    tree.projections = compute_projections(tree, current_price)
    
    return tree


def _identify_current_position(node: WaveNode) -> str:
    """Walk the tree to identify what wave we're currently in at every degree."""
    parts = []
    current = node
    
    while current and current.sub_waves:
        last_sub = current.sub_waves[-1]
        degree_name = current.degree.name.title()
        parts.append(f"{degree_name} {last_sub.label}")
        current = last_sub  # recurse into the last (most recent) sub-wave
    
    return ", ".join(parts) if parts else "Unknown"


def _detect_asset_class(symbol: str) -> str:
    """Detect asset class from symbol name."""
    symbol_upper = symbol.upper()
    forex_pairs = ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD']
    crypto_tokens = ['BTC', 'ETH', 'SOL', 'ADA', 'DOGE', 'XRP', 'BNB', 'AVAX', 'DOT', 'MATIC']
    
    for pair in forex_pairs:
        if symbol_upper.startswith(pair) or symbol_upper.endswith(pair):
            if len(symbol_upper) == 6:  # EURUSD, GBPJPY, etc.
                return 'forex'
    
    for token in crypto_tokens:
        if token in symbol_upper:
            return 'crypto'
    
    return 'equity'
```

---

## PHASE 2 — API & Serialization

### File: `.worktrees/ew-engine-v2/api/routers/patterns.py`

### 2A. Add WaveTree serializer (new function, add near `_ser_v2_count()` at line ~624)

```python
def _ser_v3_tree(tree) -> dict:
    """Serialize a WaveTree to JSON-friendly dict for the frontend."""
    if not tree:
        return None
    
    def ser_node(node):
        if not node:
            return None
        return {
            'degree': node.degree.name if hasattr(node.degree, 'name') else str(node.degree),
            'wave_type': node.wave_type.name if hasattr(node.wave_type, 'name') else str(node.wave_type),
            'label': node.label,
            'start_idx': node.start_idx,
            'end_idx': node.end_idx,
            'start_price': round(node.start_price, 4),
            'end_price': round(node.end_price, 4),
            'direction': node.direction.name if hasattr(node.direction, 'name') else str(node.direction),
            'confidence': round(node.confidence, 4),
            'is_complete': getattr(node, 'is_complete', True),
            'sub_waves': [ser_node(sw) for sw in (node.sub_waves or [])],
        }
    
    def ser_projection(proj):
        return {
            'target_price': proj.target_price,
            'fib_ratio': proj.fib_ratio,
            'fib_label': proj.fib_label,
            'from_wave_label': proj.from_wave_label,
            'is_invalidation': proj.is_invalidation,
            'pct_from_current': proj.pct_from_current,
        }
    
    return {
        'symbol': tree.symbol,
        'timeframe': tree.timeframe,
        'version': 'v3',
        'starting_point': {
            'idx': tree.starting_point_idx,
            'price': round(tree.starting_point_price, 4),
        },
        'current_position': tree.current_position,
        'overall_confidence': round(tree.overall_confidence, 4),
        'primary_count': ser_node(tree.root),
        'alternate_counts': [ser_node(alt) for alt in (tree.alternates or [])[:3]],
        'projections': [ser_projection(p) for p in (tree.projections or [])],
    }
```

### 2B. Update the `get_elliott_waves()` endpoint (at line ~717)

Add v3 analysis alongside existing v2. The endpoint should try v3 first, fall back to v2:

In the `get_elliott_waves()` function, **after** the existing v2 analysis code, add:

```python
    # --- v3: Multi-degree recursive wave tree ---
    from pattern_engine.elliott_wave import analyze_wave_tree, WaveTree
    
    v3_tree = None
    try:
        v3_tree = analyze_wave_tree(
            df=df,
            symbol=symbol,
            asset_class=asset_class,
            timeframe=timeframe,
        )
    except Exception as e:
        logger.warning(f"v3 wave tree analysis failed for {symbol}: {e}")
    
    # Add v3 data to response
    if v3_tree:
        result['wave_tree'] = _ser_v3_tree(v3_tree)
    else:
        result['wave_tree'] = None
```

This way the v2 data continues to be returned for backward compatibility, and the frontend can read `wave_tree` when available.

---

## PHASE 3 — Frontend: Multi-Degree Labels, Projections, Active Zone, Theming

### File: `frontend/src/components/chart/KLineChartOverlay.tsx`

The frontend already has a wave color theme system (`WaveColorTheme` interface at line 221, `WAVE_THEMES` at line 231, `waveThemeIdx` state at line 2119). We'll extend this into a comprehensive appearance system.

### 3A. Extend the theme/appearance system

Near the existing `WaveColorTheme` interface (line ~221), add a comprehensive settings interface:

```typescript
// Elliott Wave display settings — user-configurable
interface EWDisplaySettings {
  // Visibility toggles
  showProjections: boolean;        // toggle fibonacci projections on/off
  showActiveZone: boolean;         // toggle the active pattern zone highlight
  showStartingPoint: boolean;      // toggle the "o" starting point marker
  showWaveLines: boolean;          // toggle connecting wave lines
  showWaveLabels: boolean;         // toggle wave label text
  showAlternates: boolean;         // toggle alternate count display
  
  // Degree visibility (toggle each degree on/off)
  visibleDegrees: {
    supercycle: boolean;
    cycle: boolean;
    primary: boolean;
    intermediate: boolean;
    minor: boolean;
    minute: boolean;
    subminute: boolean;
  };
  
  // Appearance
  chartBackground: string;         // hex color for chart background
  candleBullish: string;           // hex color for bullish candles
  candleBearish: string;           // hex color for bearish candles
  
  // Wave degree colors (override theme colors per degree)
  degreeColors: {
    supercycle: string;
    cycle: string;
    primary: string;
    intermediate: string;
    minor: string;
    minute: string;
    subminute: string;
  };
  
  // Label sizing
  labelSize: 'small' | 'medium' | 'large';
  
  // Projection line style
  projectionStyle: 'solid' | 'dashed' | 'dotted';
  
  // Active zone opacity (0-1)
  activeZoneOpacity: number;
  
  // Wave line opacity (0-1)
  waveLineOpacity: number;
}

// Default settings
const DEFAULT_EW_SETTINGS: EWDisplaySettings = {
  showProjections: true,
  showActiveZone: true,
  showStartingPoint: true,
  showWaveLines: true,
  showWaveLabels: true,
  showAlternates: true,
  visibleDegrees: {
    supercycle: true, cycle: true, primary: true,
    intermediate: true, minor: true, minute: false, subminute: false,
  },
  chartBackground: '#1a1a2e',
  candleBullish: '#22c55e',
  candleBearish: '#ef4444',
  degreeColors: {
    supercycle: '#ff6b6b',
    cycle: '#f97316',
    primary: '#fbbf24',
    intermediate: '#22d3ee',
    minor: '#a78bfa',
    minute: '#6b7280',
    subminute: '#4b5563',
  },
  labelSize: 'medium',
  projectionStyle: 'dashed',
  activeZoneOpacity: 0.12,
  waveLineOpacity: 0.5,
};

// Preset themes
const EW_THEME_PRESETS: Record<string, Partial<EWDisplaySettings>> = {
  'Dark Pro': {
    chartBackground: '#1a1a2e',
    candleBullish: '#22c55e',
    candleBearish: '#ef4444',
    degreeColors: {
      supercycle: '#ff6b6b', cycle: '#f97316', primary: '#fbbf24',
      intermediate: '#22d3ee', minor: '#a78bfa', minute: '#6b7280', subminute: '#4b5563',
    },
  },
  'Light': {
    chartBackground: '#ffffff',
    candleBullish: '#16a34a',
    candleBearish: '#dc2626',
    degreeColors: {
      supercycle: '#dc2626', cycle: '#ea580c', primary: '#d97706',
      intermediate: '#0891b2', minor: '#7c3aed', minute: '#6b7280', subminute: '#9ca3af',
    },
  },
  'WaveBasis': {
    chartBackground: '#1a2332',
    candleBullish: '#26a69a',
    candleBearish: '#ef5350',
    degreeColors: {
      supercycle: '#ff5252', cycle: '#ff9800', primary: '#ffeb3b',
      intermediate: '#00bcd4', minor: '#ce93d8', minute: '#78909c', subminute: '#546e7a',
    },
  },
  'Classic': {
    chartBackground: '#000000',
    candleBullish: '#00ff00',
    candleBearish: '#ff0000',
    degreeColors: {
      supercycle: '#ff0000', cycle: '#ff8800', primary: '#ffff00',
      intermediate: '#00ffff', minor: '#ff00ff', minute: '#888888', subminute: '#555555',
    },
  },
  'Ocean': {
    chartBackground: '#0a192f',
    candleBullish: '#64ffda',
    candleBearish: '#ff6b6b',
    degreeColors: {
      supercycle: '#ff6b6b', cycle: '#ffd93d', primary: '#6bcb77',
      intermediate: '#4d96ff', minor: '#9b59b6', minute: '#7f8c8d', subminute: '#566573',
    },
  },
};
```

### 3B. Add settings state and persistence

In the component's state section (near line ~2119 where `waveThemeIdx` is defined), add:

```typescript
// EW Display Settings — persisted to localStorage
const [ewSettings, setEwSettings] = useState<EWDisplaySettings>(() => {
  const saved = localStorage.getItem(`ew_settings_${symbol}`);
  if (saved) {
    try { return { ...DEFAULT_EW_SETTINGS, ...JSON.parse(saved) }; }
    catch { return DEFAULT_EW_SETTINGS; }
  }
  return DEFAULT_EW_SETTINGS;
});

// Persist settings changes
useEffect(() => {
  localStorage.setItem(`ew_settings_${symbol}`, JSON.stringify(ewSettings));
}, [ewSettings, symbol]);
```

### 3C. Add the v3 wave tree rendering functions

Add these new drawing functions alongside the existing `drawWaveCount()` (line ~443). These render the recursive wave tree:

```typescript
/**
 * Draw multi-degree wave labels from the v3 WaveTree.
 * Walks the tree recursively, collecting all pivot labels, then renders them
 * with vertical stacking and collision avoidance per degree.
 */
const drawMultiDegreeLabels = (
  ctx: CanvasRenderingContext2D,
  waveTree: any,  // v3 WaveTree from API
  settings: EWDisplaySettings,
  xScale: (idx: number) => number,
  yScale: (price: number) => number,
) => {
  if (!waveTree?.primary_count) return;
  
  // Collect all labels from the tree
  interface LabelEntry {
    idx: number; price: number; label: string; degree: string;
    color: string; above: boolean; fontSize: number;
  }
  const labels: LabelEntry[] = [];
  
  const degreeOrder = ['supercycle','cycle','primary','intermediate','minor','minute','subminute'];
  const fontSizes: Record<string, number> = {
    supercycle: 14, cycle: 13, primary: 12,
    intermediate: 11, minor: 10, minute: 9, subminute: 8,
  };
  
  function collectLabels(node: any, depth: number = 0) {
    if (!node || !node.sub_waves) return;
    
    const degreeName = (node.degree || '').toLowerCase();
    if (!settings.visibleDegrees[degreeName as keyof typeof settings.visibleDegrees]) return;
    
    const color = settings.degreeColors[degreeName as keyof typeof settings.degreeColors] || '#888';
    const fontSize = fontSizes[degreeName] || 10;
    
    for (const sw of node.sub_waves) {
      const above = sw.direction === 'UP';
      const price = above ? sw.end_price : sw.end_price;
      labels.push({
        idx: sw.end_idx, price, label: sw.label,
        degree: degreeName, color, above, fontSize,
      });
      
      // Recurse into sub-waves
      if (sw.sub_waves && sw.sub_waves.length > 0) {
        collectLabels(sw, depth + 1);
      }
    }
  }
  
  collectLabels(waveTree.primary_count);
  
  // Render labels with collision avoidance
  // Group by x position and stack vertically
  // ... (use existing collision avoidance from lines 516-534)
  
  for (const lbl of labels) {
    if (!settings.showWaveLabels) continue;
    
    const px = xScale(lbl.idx);
    const py = yScale(lbl.price);
    const offsetMult = degreeOrder.indexOf(lbl.degree) <= 2 ? 2.5 : 1.5;
    const offset = lbl.above ? -10 * offsetMult : 10 * offsetMult;
    
    // Dot at pivot
    ctx.fillStyle = lbl.color;
    ctx.beginPath();
    ctx.arc(px, py, lbl.fontSize >= 12 ? 3 : 2, 0, Math.PI * 2);
    ctx.fill();
    
    // Label text
    const bold = lbl.fontSize >= 12 ? 'bold ' : '';
    ctx.font = `${bold}${lbl.fontSize}px -apple-system, sans-serif`;
    ctx.fillStyle = lbl.color;
    ctx.textAlign = 'center';
    ctx.textBaseline = lbl.above ? 'bottom' : 'top';
    ctx.fillText(lbl.label, px, py + offset);
  }
};

/**
 * Draw wave connecting lines for a specific degree (shown on hover or always).
 */
const drawWaveConnectingLines = (
  ctx: CanvasRenderingContext2D,
  node: any,
  settings: EWDisplaySettings,
  xScale: (idx: number) => number,
  yScale: (price: number) => number,
) => {
  if (!node?.sub_waves || !settings.showWaveLines) return;
  
  const degreeName = (node.degree || '').toLowerCase();
  const color = settings.degreeColors[degreeName as keyof typeof settings.degreeColors] || '#888';
  
  ctx.strokeStyle = color;
  ctx.globalAlpha = settings.waveLineOpacity;
  ctx.lineWidth = degreeName === 'cycle' || degreeName === 'primary' ? 2 : 1;
  ctx.beginPath();
  
  for (let i = 0; i < node.sub_waves.length; i++) {
    const sw = node.sub_waves[i];
    const startX = xScale(sw.start_idx);
    const startY = yScale(sw.start_price);
    const endX = xScale(sw.end_idx);
    const endY = yScale(sw.end_price);
    
    if (i === 0) ctx.moveTo(startX, startY);
    ctx.lineTo(endX, endY);
  }
  
  ctx.stroke();
  ctx.globalAlpha = 1;
};

/**
 * Draw the starting point marker ("o" circle with info box).
 */
const drawStartingPoint = (
  ctx: CanvasRenderingContext2D,
  waveTree: any,
  settings: EWDisplaySettings,
  xScale: (idx: number) => number,
  yScale: (price: number) => number,
) => {
  if (!settings.showStartingPoint || !waveTree?.starting_point) return;
  
  const sp = waveTree.starting_point;
  const px = xScale(sp.idx);
  const py = yScale(sp.price);
  
  // Circle
  ctx.strokeStyle = settings.degreeColors.intermediate;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(px, py + 15, 10, 0, Math.PI * 2);
  ctx.stroke();
  
  // "o" label
  ctx.fillStyle = settings.degreeColors.intermediate;
  ctx.font = 'bold 12px -apple-system, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText('o', px, py + 15);
};

/**
 * Draw projection lines extending to the right of current price.
 */
const drawProjections = (
  ctx: CanvasRenderingContext2D,
  projections: any[],
  currentBarIdx: number,
  currentPrice: number,
  settings: EWDisplaySettings,
  xScale: (idx: number) => number,
  yScale: (price: number) => number,
  chartRightEdge: number,
) => {
  if (!settings.showProjections || !projections?.length) return;
  
  const startX = xScale(currentBarIdx);
  const startY = yScale(currentPrice);
  
  for (const proj of projections) {
    const targetY = yScale(proj.target_price);
    const endX = chartRightEdge - 60; // leave room for label
    
    const color = proj.is_invalidation ? '#ef4444' : '#22d3ee';
    const alpha = proj.is_invalidation ? 0.5 : 0.4;
    
    ctx.strokeStyle = color;
    ctx.globalAlpha = alpha;
    ctx.lineWidth = 1;
    
    // Line style
    if (settings.projectionStyle === 'dashed') ctx.setLineDash([6, 4]);
    else if (settings.projectionStyle === 'dotted') ctx.setLineDash([2, 3]);
    else ctx.setLineDash([]);
    
    // Draw projection line
    ctx.beginPath();
    ctx.moveTo(startX, startY);
    ctx.lineTo(endX, targetY);
    ctx.stroke();
    
    // Horizontal target line
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(endX - 20, targetY);
    ctx.lineTo(chartRightEdge, targetY);
    ctx.stroke();
    ctx.setLineDash([]);
    
    // Label
    ctx.globalAlpha = Math.min(1, alpha + 0.3);
    ctx.fillStyle = color;
    ctx.font = 'bold 10px -apple-system, sans-serif';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillText(
      `${proj.target_price.toFixed(1)} (${proj.fib_label})`,
      chartRightEdge - 2, targetY
    );
    
    ctx.globalAlpha = 1;
  }
};

/**
 * Draw the active pattern zone (shaded rectangle over currently-forming pattern).
 */
const drawActiveZone = (
  ctx: CanvasRenderingContext2D,
  waveTree: any,
  settings: EWDisplaySettings,
  xScale: (idx: number) => number,
  yScale: (price: number) => number,
  bars: any[],
) => {
  if (!settings.showActiveZone || !waveTree?.primary_count) return;
  
  // Find the last incomplete wave — it defines the active zone
  function findLastWave(node: any): any {
    if (!node?.sub_waves?.length) return node;
    const last = node.sub_waves[node.sub_waves.length - 1];
    return findLastWave(last);
  }
  
  const lastWave = findLastWave(waveTree.primary_count);
  if (!lastWave) return;
  
  // Zone spans from the parent of the last wave to the current bar
  const startIdx = lastWave.start_idx;
  const endIdx = bars.length - 1;
  
  // Find price range in zone
  let zoneHigh = -Infinity, zoneLow = Infinity;
  for (let i = startIdx; i <= endIdx && i < bars.length; i++) {
    if (bars[i].high > zoneHigh) zoneHigh = bars[i].high;
    if (bars[i].low < zoneLow) zoneLow = bars[i].low;
  }
  
  const x1 = xScale(startIdx) - 3;
  const x2 = xScale(endIdx) + 10;
  const y1 = yScale(zoneHigh + (zoneHigh - zoneLow) * 0.05);
  const y2 = yScale(zoneLow - (zoneHigh - zoneLow) * 0.05);
  
  // Shaded fill
  ctx.fillStyle = `rgba(249, 115, 22, ${settings.activeZoneOpacity})`;
  ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
  
  // Dashed border
  ctx.strokeStyle = 'rgba(249, 115, 22, 0.35)';
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 3]);
  ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  ctx.setLineDash([]);
};
```

### 3D. Add a settings panel UI

Add a collapsible settings panel in the chart's sidebar/toolbar area. This should include:

1. **Toggle switches** for: Projections, Active Zone, Starting Point, Wave Lines, Wave Labels, Alternates
2. **Degree checkboxes** — toggle visibility per degree (Supercycle through Subminute)
3. **Theme presets** dropdown — Dark Pro, Light, WaveBasis, Classic, Ocean
4. **Color pickers** for: Chart background, Bullish candle, Bearish candle, each degree color
5. **Sliders** for: Label size, Active zone opacity, Wave line opacity
6. **Projection style** selector: solid / dashed / dotted

The settings panel should be accessible via a gear icon near the existing wave theme selector (line ~2756). All settings persist to localStorage per symbol.

### 3E. Integration: use v3 data when available

In the main rendering flow, check if the API returned `wave_tree` (v3 data):

```typescript
// In the main draw function, after existing v2 rendering:
if (apiData?.wave_tree && apiData.wave_tree.primary_count) {
  // Use v3 multi-degree rendering
  drawStartingPoint(ctx, apiData.wave_tree, ewSettings, xScale, yScale);
  drawWaveConnectingLines(ctx, apiData.wave_tree.primary_count, ewSettings, xScale, yScale);
  drawMultiDegreeLabels(ctx, apiData.wave_tree, ewSettings, xScale, yScale);
  drawActiveZone(ctx, apiData.wave_tree, ewSettings, xScale, yScale, bars);
  drawProjections(ctx, apiData.wave_tree.projections, bars.length - 1, currentPrice, ewSettings, xScale, yScale, chartRightEdge);
} else {
  // Fallback to v2 rendering
  drawWaveCount(/* existing v2 params */);
}
```

---

## PHASE 4 — Polish & Side Panel

### 4A. Side panel content (can be a new React component or extend existing)

The right side panel should display:
1. **Current Position** box — green border, shows degree path like "Primary III, Intermediate (3), Minor [3]"
2. **Next Subwaves** table — Wave | R/R | Target | Risk | Status columns
3. **Alternate Counts** — cards labeled P, 1, 2, 3 with confidence %
4. **Wave Guide** — shows the rules for the current pattern type (Diagonal 3-3-3-3-3, Impulse 5-3-5-3-5, etc.)
5. **Degree Legend** — colored dots with degree names and notation examples

### 4B. Hover interaction

When the user hovers over a wave label on the chart:
- Highlight that wave's connecting lines (increase opacity)
- Show a tooltip with: degree, pattern type, price range, fibonacci %, confidence
- Brighten all sub-wave labels within that wave
- Dim everything outside that wave

---

## Implementation Notes

### Files to modify:
1. **`.worktrees/ew-engine-v2/pattern_engine/elliott_wave.py`** — Add dataclasses, `find_starting_point()`, `analyze_recursive()`, `compute_projections()`, `analyze_wave_tree()`. Keep all existing functions unchanged.
2. **`.worktrees/ew-engine-v2/api/routers/patterns.py`** — Add `_ser_v3_tree()`, update `get_elliott_waves()` to call v3 and include `wave_tree` in response.
3. **`frontend/src/components/chart/KLineChartOverlay.tsx`** — Add `EWDisplaySettings`, theme presets, settings panel, and all `draw*` functions for v3 rendering. Extend existing theme system at line 221-231.

### What NOT to change:
- Do NOT modify any existing finder functions (`find_impulse_candidates`, `find_zigzag_candidates`, etc.)
- Do NOT modify `validate_and_score()` or `_score_proportionality()`
- Do NOT modify `detect_major_pivots()` or `build_monowaves()`
- Do NOT delete `analyze_multi_degree()` — keep it as fallback
- Do NOT break the existing v2 API contract — add v3 data alongside v2

### Testing:
After implementation, test with these symbols to verify:
- **SPY** (1D and 1W) — should produce multi-degree count similar to WaveBasis
- **PLTR** — previously had proportionality issues, should now decompose sub-waves properly
- **CRWV** — previously had "one straight line C wave", should now show sub-waves inside C
- **META, INTC** — verify no regressions from V7/V8 fixes
- **EURUSD** (forex), **BTCUSD** (crypto) — verify asset-class thresholds work
- **All timeframes** — verify 1D, 1W work with same engine

### Performance target:
- < 3 seconds for a 500-bar daily chart
- < 5 seconds for a 1000-bar weekly chart
- If exceeding targets, reduce `max_depth` or increase candidate pruning threshold
