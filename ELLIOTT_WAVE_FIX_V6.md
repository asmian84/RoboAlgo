# Elliott Wave V6 Fix — Magnifier DPR + Auto-Zoom Reliability

## Context

Post-V5 all 5 charts produce valid wave counts from the engine. Two frontend issues remain:

1. **Magnifying glass is offset from the cursor** — shows the wrong region, not centered on where you're actually pointing.
2. **Auto-zoom doesn't reliably frame the full wave pattern** — BTC-USD daily chart shows the pattern (confirmed via API) but the chart viewport stays zoomed into recent bars, leaving W1-W3 off-screen to the left.

Both fixes are in **one file**: `frontend/src/components/chart/KLineChartOverlay.tsx`

---

## Bug 16 — Magnifier not centered on cursor (devicePixelRatio mismatch)

### Problem

The magnifier reads pixel data from klinecharts' internal canvases using CSS-coordinate math:

```typescript
const sx = Math.max(0, Math.min(rect.width  - SRC_W, x - SRC_W / 2))
const sy = Math.max(0, Math.min(rect.height - SRC_H, y - SRC_H / 2))
// ...
ctx.drawImage(c, sx, sy, SRC_W, SRC_H, 0, 0, LENS_W, LENS_H)
```

But klinecharts (like most chart libraries) renders its internal canvases at `devicePixelRatio` scale for crisp Retina rendering. On a Mac with `devicePixelRatio = 2`, the canvas has `width = containerWidth * 2` and `height = containerHeight * 2`. So CSS coordinate `(100, 100)` maps to canvas pixel `(200, 200)`.

The `drawImage()` source coordinates (`sx, sy`) are in CSS space but are being used to index into a DPR-scaled canvas, causing the magnified region to be offset — typically showing content that's up and to the left of where the cursor actually is.

### Fix

**File: `frontend/src/components/chart/KLineChartOverlay.tsx`**  
**Function: `onMove` handler inside the magnifier `useEffect` (line ~2236)**

Replace lines ~2248-2259:

```typescript
      // Source region centered on cursor (clamped to chart bounds)
      const sx = Math.max(0, Math.min(rect.width  - SRC_W, x - SRC_W / 2))
      const sy = Math.max(0, Math.min(rect.height - SRC_H, y - SRC_H / 2))

      // Dark background
      ctx.fillStyle = '#0d0d12'
      ctx.fillRect(0, 0, LENS_W, LENS_H)

      // Composite all canvas layers (klinecharts uses 2-3 overlapping canvases)
      const canvases = container.querySelectorAll<HTMLCanvasElement>('canvas')
      for (const c of canvases) {
        try { ctx.drawImage(c, sx, sy, SRC_W, SRC_H, 0, 0, LENS_W, LENS_H) } catch {}
      }
```

with:

```typescript
      // Source region centered on cursor (clamped to chart bounds)
      // CRITICAL: klinecharts renders canvases at devicePixelRatio scale
      // for Retina crispness. CSS coordinates must be multiplied by dpr
      // to index into the actual canvas pixel buffer.
      const dpr = window.devicePixelRatio || 1
      const sx = Math.max(0, Math.min(rect.width  - SRC_W, x - SRC_W / 2))
      const sy = Math.max(0, Math.min(rect.height - SRC_H, y - SRC_H / 2))

      // Dark background
      ctx.fillStyle = '#0d0d12'
      ctx.fillRect(0, 0, LENS_W, LENS_H)

      // Composite all canvas layers (klinecharts uses 2-3 overlapping canvases)
      // Multiply source coordinates and dimensions by dpr to read the correct
      // region from the high-resolution canvas backing store.
      const canvases = container.querySelectorAll<HTMLCanvasElement>('canvas')
      for (const c of canvases) {
        try {
          ctx.drawImage(
            c,
            sx * dpr, sy * dpr,       // source x, y (in canvas pixels)
            SRC_W * dpr, SRC_H * dpr, // source width, height (in canvas pixels)
            0, 0,                      // destination x, y
            LENS_W, LENS_H            // destination width, height
          )
        } catch {}
      }
```

### Why this works

- `drawImage(canvas, sx, sy, sw, sh, dx, dy, dw, dh)` reads from the canvas's **pixel buffer**, not its CSS dimensions
- On a 2x Retina display, a 1000×600 CSS-sized canvas has a 2000×1200 pixel buffer
- Mouse position `x=500` maps to canvas pixel `1000`, not `500`
- Without the `dpr` multiplier, the magnifier reads pixels at half the correct offset — always showing content from the upper-left quadrant relative to where the cursor actually is

---

## Bug 17 — Auto-zoom not reliably framing the full wave pattern

### Problem

The existing auto-zoom code (lines ~1919-1945) correctly computes `iMin` (earliest wave pivot bar index) and sets bar space + scroll position. But it has a timing issue:

1. `chart.setBarSpace(targetBarSpace)` is called inside a `setTimeout(..., 100)`
2. Followed immediately by `chart.scrollToTimestamp(lastTs, 300)` which triggers a 300ms animated scroll
3. Then `chart.setOffsetRightDistance(rightPadBars * targetBarSpace)`

The problem is that `setBarSpace` triggers an internal relayout in klinecharts, and `scrollToTimestamp` fires before that relayout completes. On charts with 780 bars spanning from idx 6 to idx 780, the bar space shrinks to ~1.5px/bar. The scroll animation may overshoot or undershoot because it was computed against the old bar space.

Additionally, the y-axis override at lines 1876-1892 is released after 600ms, but the horizontal zoom happens at 100ms. If the y-axis hasn't finished adjusting, the initial paint may look wrong.

### Fix

**File: `frontend/src/components/chart/KLineChartOverlay.tsx`**  
**Location: lines ~1919-1945 (the `setTimeout(() => {` block for horizontal zoom)**

Replace the existing horizontal zoom `setTimeout` block:

```typescript
          setTimeout(() => {
            try {
              const totalBars = serverBars.length
              const lastBarIdx = Math.max(totalBars - 1, 0)
              // Coverage needed: from earliest pivot through last candle.
              // We can extend a bit LEFT of iMin too (user said "almost
              // even go backward") so wave 1 start has breathing room.
              const coverageBars = Math.max(lastBarIdx - iMin + 1, 20)
              // Left padding: 10% of coverage, min 5 bars, max 25 bars
              const leftPadBars = Math.min(Math.max(Math.floor(coverageBars * 0.10), 5), 25)
              // Right padding: small — the last candle should hug the right edge
              const rightPadBars = 6
              const visibleBars = Math.min(coverageBars + leftPadBars + rightPadBars, totalBars)

              const chartWidth = (chart as any).getSize?.()?.width
                || (typeof window !== 'undefined' ? window.innerWidth * 0.7 : 900)
              const targetBarSpace = Math.max(1.5, Math.min(40, chartWidth / visibleBars))
              chart.setBarSpace(targetBarSpace)

              // Scroll so the LAST bar is at the right edge of the viewport.
              const lastTs = snap(lastBarIdx)
              chart.scrollToTimestamp(lastTs, 300)
              chart.setOffsetRightDistance(rightPadBars * targetBarSpace)
            } catch (e) {
              console.warn('[EW auto-zoom] horizontal zoom failed:', e)
            }
          }, 100)
```

with:

```typescript
          // Two-phase zoom: first set bar space, then scroll after a short
          // delay to allow klinecharts to complete its internal relayout.
          // A single setTimeout with both operations races against the layout.
          setTimeout(() => {
            try {
              const totalBars = serverBars.length
              const lastBarIdx = Math.max(totalBars - 1, 0)
              const coverageBars = Math.max(lastBarIdx - iMin + 1, 20)
              const leftPadBars = Math.min(Math.max(Math.floor(coverageBars * 0.10), 5), 25)
              const rightPadBars = 6
              const visibleBars = Math.min(coverageBars + leftPadBars + rightPadBars, totalBars)

              const chartWidth = (chart as any).getSize?.()?.width
                || (typeof window !== 'undefined' ? window.innerWidth * 0.7 : 900)
              const targetBarSpace = Math.max(1.5, Math.min(40, chartWidth / visibleBars))

              // Phase 1: set bar space (triggers internal relayout)
              chart.setBarSpace(targetBarSpace)
              chart.setOffsetRightDistance(rightPadBars * targetBarSpace)

              // Phase 2: scroll AFTER relayout settles (use scrollToDataIndex
              // which is more reliable than scrollToTimestamp for exact positioning)
              setTimeout(() => {
                try {
                  // Scroll so the earliest wave pivot (iMin) is at the left edge.
                  // scrollToDataIndex places the given index in view. We want
                  // iMin - leftPadBars at the left edge.
                  const scrollTarget = Math.max(0, iMin - leftPadBars)
                  ;(chart as any).scrollToDataIndex?.(scrollTarget)

                  // Fallback: if scrollToDataIndex isn't available, use timestamp-based scroll
                  if (!(chart as any).scrollToDataIndex) {
                    const firstTs = snap(scrollTarget)
                    chart.scrollToTimestamp(firstTs)
                  }
                } catch (e2) {
                  // Last resort: scroll to last bar
                  try { chart.scrollToTimestamp(snap(lastBarIdx)) } catch {}
                  console.warn('[EW auto-zoom] scroll phase failed:', e2)
                }
              }, 50)
            } catch (e) {
              console.warn('[EW auto-zoom] horizontal zoom failed:', e)
            }
          }, 150)
```

### Key changes

1. **Two-phase approach**: `setBarSpace` first, then `scrollToDataIndex` after a 50ms micro-delay. This gives klinecharts time to complete the bar space relayout before we try to scroll.

2. **`scrollToDataIndex` instead of `scrollToTimestamp`**: More reliable for positioning by bar index. Falls back to `scrollToTimestamp` if the method isn't available.

3. **Scroll to leftmost pivot** instead of rightmost: We want the EARLIEST wave pivot visible at the left edge, not the last bar at the right edge. With bar space already set to show all bars, scrolling to `iMin - leftPadBars` puts W1 start in view with breathing room.

4. **Initial `setTimeout` increased from 100ms to 150ms**: Gives more time for the y-axis override to take effect first.

---

## Summary

| Bug | Location | Issue | Fix |
|-----|----------|-------|-----|
| 16 | Line ~2248 | Magnifier `drawImage` reads at CSS coords instead of DPR-scaled canvas coords | Multiply `sx, sy, SRC_W, SRC_H` by `devicePixelRatio` |
| 17 | Line ~1919 | Auto-zoom `setBarSpace` + `scrollToTimestamp` race condition | Two-phase zoom: setBarSpace first, scroll after 50ms delay; use `scrollToDataIndex` |

## File to edit

`frontend/src/components/chart/KLineChartOverlay.tsx`

## Testing

1. **Magnifier**: Enable magnifier tool (magnifying glass icon in left toolbar), hover over a specific candle. The magnified view should now show that exact candle at the crosshair center — not offset to the upper-left.

2. **Auto-zoom**: Navigate to BTC-USD daily chart. After the "Detecting Elliott Waves" loading completes, the chart should automatically zoom out to show the full 5-wave impulse starting from W1 (~$24K in mid-2023) through W5 (~$126K) and the ABC correction. Similarly test GC=F — the full impulse from ~$2,100 to ~$5,586 should be visible without manual scrolling.
