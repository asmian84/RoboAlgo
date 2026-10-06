# Elliott Wave V2 Engine — Fix Prompt

**CRITICAL**: The backend runs from the WORKTREE, not the main repo:
```
Server directory: /Users/asmian/RoboAlgo/.worktrees/ew-engine-v2/
File to fix:      .worktrees/ew-engine-v2/pattern_engine/elliott_wave.py
```
(See `start-ew-dev.sh` — it does `cd "$WORKTREE" && uvicorn api.main:app`)

The v2 engine has correct validation rules (no inverted logic), but produces bad wave counts because of pivot detection and scoring issues. Copy everything below and paste into Claude Code:

---

## Task: Fix 5 bugs in the v2 Elliott Wave engine

File: `.worktrees/ew-engine-v2/pattern_engine/elliott_wave.py`

The v2 engine validation is correct but produces bad counts on every chart tested (GC=F, AAPL, EONR, BTC-USD). The root causes are in pivot detection and scoring.

---

### BUG 1 — CRITICAL: `detect_major_pivots()` doesn't anchor to chart ATH/ATL (lines 445-553)

The ratchet zigzag algorithm starts at `idx=0` and uses the first bar's open/close to decide initial direction. It never scans for the chart's absolute high or low. This means:

- **EONR**: Starts at ~$2.80 instead of the $4.00 ATH (the spike happens BEFORE the ratchet settles)
- **GC=F**: Misses the $2,277 structural low as W1 start because the ratchet starts tracking from bar 0

**Fix**: After the ratchet zigzag loop completes, inject the chart's absolute ATH and ATL into the pivot list if they aren't already represented. Add this BEFORE the `return pivots` at the end of `detect_major_pivots()` (after line 551, before line 553):

```python
    # ── Force-include chart ATH and ATL as structural anchors ──────────
    # The ratchet zigzag can miss the absolute extreme if it occurs before
    # the algorithm establishes direction (e.g. a spike in the first few bars).
    if n > 0 and len(pivots) >= 2:
        ath_idx = int(high.argmax())
        ath_price = float(high[ath_idx])
        atl_idx = int(low.argmin())
        atl_price = float(low[atl_idx])

        # Check if ATH is already captured (within 1% tolerance)
        ath_captured = any(
            p.is_high and abs(p.price - ath_price) / max(ath_price, 1e-10) < 0.01
            for p in pivots
        )
        atl_captured = any(
            not p.is_high and abs(p.price - atl_price) / max(atl_price, 1e-10) < 0.01
            for p in pivots
        )

        if not ath_captured:
            # Insert ATH at the correct chronological position
            insert_pos = 0
            for i, p in enumerate(pivots):
                if p.idx > ath_idx:
                    break
                insert_pos = i + 1
            # Only insert if neighbors allow alternation (must be between two lows or at edge)
            if insert_pos == 0 or insert_pos >= len(pivots) or not pivots[insert_pos - 1].is_high:
                pivots.insert(insert_pos, Pivot(idx=ath_idx, price=ath_price, is_high=True, strength=10))

        if not atl_captured:
            insert_pos = 0
            for i, p in enumerate(pivots):
                if p.idx > atl_idx:
                    break
                insert_pos = i + 1
            if insert_pos == 0 or insert_pos >= len(pivots) or pivots[insert_pos - 1].is_high:
                pivots.insert(insert_pos, Pivot(idx=atl_idx, price=atl_price, is_high=False, strength=10))

    return pivots
```

---

### BUG 2: Span score uses sqrt damping — too weak for structural dominance (line 1393)

The span score formula is:
```python
span_score = (bar_span ** 0.5) * 0.5 + (price_span ** 0.5) * 0.5
```

With sqrt damping, a small-scale pattern (spanning 25% of the chart) scores `0.5^0.5 = 0.707` vs a full-chart pattern scoring `1.0^0.5 = 1.0`. That's only a 0.293 difference. After weighting at 55%, that's just 0.16 points — easily overridden by Fibonacci or internal structure scores. This is why small cramped counts near the chart top can outscore the correct large-scale structure.

**Fix**: Use **linear** span scoring instead of sqrt. Replace line 1393:
```python
        # Linear span scoring — large-scale structures get decisive advantage
        span_score = bar_span * 0.5 + price_span * 0.5
```

Now a 25% span pattern scores 0.25 vs 1.0 for full chart — a 0.75 difference × 0.55 weight = 0.41 points. This makes it nearly impossible for a cramped small pattern to outscore the correct structural reading.

---

### BUG 3: Rule 3 has a 15% grace margin that allows invalid overlaps (lines 2078-2080)

```python
w1_move     = abs(w1.end.price - w1.start.price)
rule3_grace = w1_move * 0.15
```

For AAPL, W1 move ≈ $60, so grace = $9. W4 low at $170 vs W1 end at $230 is a **$60 violation** — the grace doesn't save this particular case. But for smaller overlaps (like a $5 overlap on a $40 W1 move), the 15% grace ($6) would let it pass. This is too permissive.

**Fix**: Reduce grace to 2% (accounts for pivot noise without allowing real overlap):
```python
        rule3_grace = w1_move * 0.02
```

---

### BUG 4: Three-scale candidates compete with global span normalization (lines 1337-1343)

`max_bar_span` and `max_price_span` are computed across ALL candidates from ALL three ATR scales (intermediate, macro, primary). A macro-scale impulse spanning 400 bars gets `bar_span = 1.0`, but an intermediate-scale impulse spanning 200 bars gets `bar_span = 0.5`. This seems correct, but the intermediate candidate might win on Fibonacci/internal scoring, and the 0.5 span difference (after sqrt damping per Bug 2) isn't enough to overcome it.

**Fix**: After fixing Bug 2 (linear span), this should be largely resolved. But as additional insurance, add a **minimum span threshold** to filter out tiny patterns before scoring. Add at the beginning of `validate_and_score()`, after `max_bar_span` and `max_price_span` are computed (after line 1343):

```python
    # Filter: discard candidates spanning < 20% of the largest candidate's price range
    # These are sub-waves, not primary structure
    min_price_span = max_price_span * 0.20
    candidates = [
        c for c in candidates
        if abs(c["waves"][-1].end.price - c["waves"][0].start.price) >= min_price_span
    ]
    if not candidates:
        return []
```

---

### BUG 5: No preference for counts anchored at chart extremes in scoring

Unlike the v1 engine (which had `peak_bonus()` in `_build_wave_list`), the v2 scoring doesn't give any bonus for wave counts that start or end at the chart's ATH/ATL. A pattern from $2,277→$5,586 (the obvious GC=F structure) scores the same as one from $4,200→$5,586 if the other scoring factors are similar.

**Fix**: Add an extreme-anchoring bonus in `validate_and_score()`. After computing `span_score` (line 1393), add:

```python
        # ── Extreme anchoring bonus ─────────────────────────────────────
        # Bonus for candidates that start or end at/near the chart's price extremes.
        # Users expect the wave pattern to capture the obvious structural high/low.
        all_prices = [w.start.price for w in waves] + [w.end.price for w in waves]
        cand_high = max(all_prices)
        cand_low  = min(all_prices)
        # max_price_span is already computed above
        total_high = max(c["waves"][-1].end.price for c in candidates)  # approximate
        total_low  = min(c["waves"][0].start.price for c in candidates)

        high_match = 1.0 if abs(cand_high - total_high) / max(total_high, 1e-10) < 0.02 else 0.0
        low_match  = 1.0 if abs(cand_low - total_low) / max(total_low, 1e-10) < 0.02 else 0.0
        anchor_bonus = (high_match + low_match) * 0.5  # 0.0, 0.5, or 1.0
```

Actually, a cleaner approach: pre-compute `data_high` and `data_low` from the DataFrame and pass them into the function. But since `validate_and_score` doesn't have the DataFrame, use the candidates' own extremes as a proxy. Then add `anchor_bonus` to the confidence calculation:

```python
        confidence = (
            rule_score     * _SCORE_WEIGHTS["rule_compliance"]   +
            internal_score * _SCORE_WEIGHTS["internal_structure"] +
            fib_score      * _SCORE_WEIGHTS["fibonacci"]          +
            alt_score      * _SCORE_WEIGHTS["alternation"]        +
            prop_score     * _SCORE_WEIGHTS["proportionality"]    +
            channel_score  * _SCORE_WEIGHTS["channel_fit"]        +
            pos_score      * _SCORE_WEIGHTS["position"]           +
            span_score     * _SCORE_WEIGHTS["span"]               +
            anchor_bonus   * 0.10  # 10% weight for extreme anchoring
        )
```

---

## VERIFICATION

After ALL fixes, restart the backend (`start-ew-dev.sh`) and test:

**1. GC=F daily**: Should show W1 from ~$2,277 → ~$2,800, W2 ~$2,550, W3 (extended) → $5,586 ATH, W4 in progress. Wave 3 must be visually the longest wave.

**2. AAPL daily**: Should start from $122 structural low. W4 must NOT overlap W1 territory ($170 < $190 is invalid with 2% grace). Should show partial impulse W1-W2-W3 with W3 at ATH $288.

**3. EONR daily**: Bearish impulse should start from $4.00 ATH (not $3.00). The $4.00 spike must be captured as a structural pivot by the force-include logic.

**4. BTC-USD daily**: Should show the full impulse from 2023 lows, not a cramped count near the top.

**Key visual test**: On every chart, the wave pattern should span most of the chart's price range. If waves are still cramped in a narrow zone, the span scoring fix hasn't taken effect.
