# Elliott Wave V11 Fix — v3 Engine Performance (8.8s → target <1s)

## Problem

The v3 engine takes **8.8 seconds** for a 500-bar chart. For comparison, the v2 engine runs in ~0.5s. Users see slow page loads on every ticker.

## Root Cause: `classify_internal_structure` called 23,935 times with no cache

Profile results on 500-bar synthetic data:

```
Total: 8.77s
  classify_internal_structure: 23,935 calls, 8.5s  (97% of total time!)
    _classify_by_recursion:     9,854 calls, 8.2s
      _detect_sub_pivots:       9,854 calls, 6.2s
        detect_pivots:         14,110 calls, 4.9s  (pandas astype/iloc overhead)
    _classify_by_heuristic:    10,677 calls, 0.5s  (fast, not a problem)
  find_pivot_based_impulse:        55 calls, 8.6s  (triggers classify)
  analyze_recursive:              402 calls, 8.8s  (cascade)
```

The call chain is:
1. `analyze_recursive` runs 402 times (recursion across degrees)
2. Each call builds monowaves, then calls `classify_internal_structure` for **every monowave** (line 716)
3. `find_pivot_based_impulse_candidates` also calls `classify_internal_structure` for every 6-pivot combo it evaluates (lines 2023, 2123)
4. `classify_internal_structure` internally calls `_classify_by_recursion` → `_detect_sub_pivots` → `detect_pivots` which does heavy pandas operations (`.astype(float)`, `.iloc`, `.reset_index()`)
5. The same `(start_idx, end_idx)` pairs get classified **multiple times** across recursion levels, but there's NO memoization

**97% of compute time is in this one function chain.**

---

## Fix A — Memoize `classify_internal_structure` (biggest win)

**File: `.worktrees/ew-engine-v2/pattern_engine/elliott_wave.py`**

Add a module-level cache dict and modify `classify_internal_structure` (line 1479) to use it.

### Step 1: Add the cache (near line ~165, before the WaveDegree class, or right after the existing `_monowave_cache`)

Find where `_monowave_cache` is defined (or any module-level cache) and add:

```python
# Internal-structure classification cache.
# Keyed by (id(df), start_idx, end_idx) → (structure, method, count)
_classify_cache: dict[tuple, tuple[str, str, int]] = {}

def clear_classify_cache():
    """Clear the classification cache. Call when switching symbols."""
    _classify_cache.clear()
```

### Step 2: Modify `classify_internal_structure` (line 1479) to use the cache

Replace the function:

```python
# OLD (lines 1479-1506):
def classify_internal_structure(
    df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    min_bars_for_recursion: int = 8,
) -> tuple[str, str, int]:
    bar_count = end_idx - start_idx
    if bar_count < 4:
        return "indeterminate", "unknown", 0
    if bar_count < min_bars_for_recursion:
        return _classify_by_heuristic(df, start_idx, end_idx)
    return _classify_by_recursion(df, start_idx, end_idx)
```

```python
# NEW — with memoization:
def classify_internal_structure(
    df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    min_bars_for_recursion: int = 8,
) -> tuple[str, str, int]:
    bar_count = end_idx - start_idx
    if bar_count < 4:
        return "indeterminate", "unknown", 0

    cache_key = (id(df), start_idx, end_idx)
    cached = _classify_cache.get(cache_key)
    if cached is not None:
        return cached

    if bar_count < min_bars_for_recursion:
        result = _classify_by_heuristic(df, start_idx, end_idx)
    else:
        result = _classify_by_recursion(df, start_idx, end_idx)

    _classify_cache[cache_key] = result
    return result
```

### Step 3: Clear the cache when starting a new analysis

In `analyze_wave_tree()` (line ~1171), add `clear_classify_cache()` at the top:

```python
def analyze_wave_tree(
    df: pd.DataFrame,
    symbol:      str = '',
    asset_class: str = '',
    timeframe:   str = 'D',
) -> 'WaveTree | None':
    if len(df) < 10:
        return None

    clear_classify_cache()  # ← ADD THIS LINE
    # ... rest of the function
```

Also add it in `analyze_multi_degree()` (the v2 entry point, line ~2254 area) at the top of the function, so v2 also benefits:

```python
def analyze_multi_degree(df, ...):
    clear_classify_cache()  # ← ADD THIS LINE
    # ... rest
```

**Expected impact: ~60-80% reduction** — the 23,935 calls will collapse to ~2,000-3,000 unique (start_idx, end_idx) pairs, eliminating ~20,000 redundant expensive computations.

---

## Fix B — Convert `detect_pivots` inner loops to use pre-extracted numpy arrays

The `detect_pivots` function (line 265) already extracts numpy arrays on lines 280-282:
```python
high = df["high"].astype(float).values
low = df["low"].astype(float).values
close = df["close"].astype(float).values
```

**This is fine.** But the problem is `detect_pivots` is called 14,110 times, and each call does `.astype(float).values` on a fresh pandas DataFrame slice. The pandas overhead (type checking, memory allocation, finalization) adds up to 3+ seconds.

### Fix: Make `_detect_sub_pivots` pass numpy arrays directly instead of a DataFrame

**Replace `_detect_sub_pivots` (line 1644):**

```python
# OLD (lines 1644-1670):
def _detect_sub_pivots(sub_df: pd.DataFrame) -> list[Pivot]:
    n = len(sub_df)
    if n < 20:
        return _simple_pivot_detect(sub_df)
    for atr_mult in (0.2, 0.3, 0.5):
        try:
            pivots = detect_pivots(sub_df, atr_multiplier=atr_mult, min_lookback=2, max_lookback=4)
            if len(pivots) >= 4:
                return pivots
        except Exception:
            continue
    try:
        return detect_pivots(sub_df, atr_multiplier=0.15, min_lookback=1, max_lookback=3)
    except Exception:
        return _simple_pivot_detect(sub_df)
```

```python
# NEW — extract numpy arrays ONCE, pass to a fast inner function:
def _detect_sub_pivots(sub_df: pd.DataFrame) -> list[Pivot]:
    n = len(sub_df)
    if n < 20:
        return _simple_pivot_detect(sub_df)

    # Extract numpy arrays ONCE — avoid repeated pandas overhead
    high  = sub_df["high"].values.astype(float)
    low   = sub_df["low"].values.astype(float)
    close = sub_df["close"].values.astype(float)

    for atr_mult in (0.2, 0.3, 0.5):
        try:
            pivots = _detect_pivots_numpy(high, low, close, atr_multiplier=atr_mult, min_lookback=2, max_lookback=4)
            if len(pivots) >= 4:
                return pivots
        except Exception:
            continue
    try:
        return _detect_pivots_numpy(high, low, close, atr_multiplier=0.15, min_lookback=1, max_lookback=3)
    except Exception:
        return _simple_pivot_detect(sub_df)
```

### Add the numpy-based pivot detector (new function, add right after `_detect_sub_pivots`):

```python
def _detect_pivots_numpy(
    high: 'np.ndarray',
    low: 'np.ndarray',
    close: 'np.ndarray',
    atr_multiplier: float = 0.5,
    min_lookback: int = 3,
    max_lookback: int = 10,
) -> list[Pivot]:
    """
    Same logic as detect_pivots() but operates on pre-extracted numpy arrays
    instead of a pandas DataFrame. Avoids repeated .astype()/.values overhead.
    """
    n = len(high)
    if n < 20:
        return []

    atr = _compute_atr(high, low, close, period=14)
    threshold = atr * atr_multiplier

    raw_pivots: list[Pivot] = []
    for lookback in range(min_lookback, max_lookback + 1):
        for i in range(lookback, n - lookback):
            is_high_flag = True
            for j in range(1, lookback + 1):
                if high[i] <= high[i - j] or high[i] <= high[i + j]:
                    is_high_flag = False
                    break
            if is_high_flag:
                raw_pivots.append(Pivot(idx=i, price=float(high[i]), is_high=True, strength=lookback))

            is_low_flag = True
            for j in range(1, lookback + 1):
                if low[i] >= low[i - j] or low[i] >= low[i + j]:
                    is_low_flag = False
                    break
            if is_low_flag:
                raw_pivots.append(Pivot(idx=i, price=float(low[i]), is_high=False, strength=lookback))

    if not raw_pivots:
        return []

    raw_pivots.sort(key=lambda p: p.idx)

    # Deduplicate
    deduped: list[Pivot] = []
    i = 0
    while i < len(raw_pivots):
        cluster = [raw_pivots[i]]
        j = i + 1
        while j < len(raw_pivots) and raw_pivots[j].idx - raw_pivots[i].idx <= 3:
            if raw_pivots[j].is_high == raw_pivots[i].is_high:
                cluster.append(raw_pivots[j])
            j += 1
        best = max(cluster, key=lambda p: (p.strength, p.price if p.is_high else -p.price))
        deduped.append(best)
        i = j if j > i + 1 else i + 1

    alternating = _enforce_alternation(deduped, high, low, threshold)
    significant = _filter_insignificant(alternating, threshold)
    return significant
```

Also modify `_simple_pivot_detect` (line 1673) to accept numpy arrays too, to avoid the `.astype(float).values` call:

```python
# OLD (lines 1673-1710):
def _simple_pivot_detect(sub_df: pd.DataFrame) -> list[Pivot]:
    high = sub_df["high"].astype(float).values
    low = sub_df["low"].astype(float).values
    n = len(sub_df)
    # ... rest unchanged
```

```python
# NEW — accept arrays or DataFrame:
def _simple_pivot_detect(sub_df) -> list[Pivot]:
    if isinstance(sub_df, pd.DataFrame):
        high = sub_df["high"].values.astype(float)
        low = sub_df["low"].values.astype(float)
        n = len(sub_df)
    else:
        # Assume it's a tuple of (high_array, low_array)
        high, low = sub_df
        n = len(high)
    # ... rest unchanged
```

**Expected impact: ~50-60% reduction** on the remaining un-cached calls by eliminating pandas overhead.

---

## Fix C — Skip `classify_internal_structure` in `analyze_recursive` for the v3 monowave loop

In `analyze_recursive` (line 714-719), classify is called for EVERY monowave BEFORE any candidate finding. But the candidate finders (`find_impulse_candidates`, etc.) already use internal_structure from the monowaves — they don't need it pre-classified for the v3 recursion because the recursion itself provides the sub-wave structure.

**However** — changing the finders' behavior is risky. A safer fix: **only classify monowaves that the finders will actually use** (i.e., skip classification for the fallback-scale monowaves in the adaptive loop).

Actually, the simplest safe fix is to just ensure Fix A (memoization) is working — that alone should cut the 23,935 calls to ~2,000-3,000 unique pairs, which combined with Fix B's numpy optimization should bring the total under 1 second.

---

## Fix D — Reduce `top_candidates` pruning from 8 to 5

In `analyze_recursive` (line 750):

```python
# OLD:
    top_candidates = scored[:8]   # prune to top 8 for recursion performance
```

```python
# NEW:
    top_candidates = scored[:5]   # prune to top 5 — each recurses into sub-waves
```

Each candidate triggers recursive decomposition of all its sub-waves. With 8 candidates × 5 waves × recursion depth 5 = up to 8^5 = 32,768 recursive paths. Cutting to 5 candidates reduces this to 5^5 = 3,125 — a 10× reduction in worst-case recursion.

**Expected impact: ~30-50% reduction in recursion calls.**

---

## Combined Expected Impact

| Fix | What | Expected Speedup |
|-----|------|-----------------|
| A: Memoize classify | Cache (start_idx, end_idx) → result | 60-80% |
| B: Numpy arrays | Avoid pandas overhead in detect_pivots | 50-60% on remaining |
| C: (via A) | Covered by memoization | — |
| D: Prune to 5 | Fewer recursive paths | 30-50% |

**Combined: 8.8s → estimated 0.5-1.5s** (within the 3s target).

---

## Verification

After applying fixes, time the engine:

```python
import time
t0 = time.time()
tree = analyze_wave_tree(df, symbol='SPY', asset_class='equity', timeframe='D')
elapsed = time.time() - t0
print(f"v3 took {elapsed:.2f}s")  # Target: < 1.5s for 500 bars, < 3s for 780 bars
```

Test with SPY, PLTR, BTC-USD at 780 bars (the default). All should complete in under 3 seconds.

Also verify that the `clear_classify_cache()` call in `analyze_wave_tree` prevents stale results when switching between symbols — each new symbol gets a fresh cache.
