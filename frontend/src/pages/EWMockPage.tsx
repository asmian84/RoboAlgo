// EWMockPage — Elliott Wave pattern visual reference
// Covers all major EW corrective + motive structures from EW International spec:
//   contracting triangle, barrier triangle, expanding triangle,
//   double zigzag (W-X-Y), ending diagonal + throw-over, impulse W3 extension
import { useEffect, useRef } from 'react'
import { init, dispose, type Chart } from 'klinecharts'
import { registerPatternOverlays } from '../components/chart/patternOverlays'

// ── Synthetic OHLCV generator ────────────────────────────────────────────────
// Each scenario is a sequence of [barIndex, closePrice] waypoints.
// We linearly interpolate between waypoints and add slight OHLC noise.
function buildBars(waypoints: [number, number][], noiseAmt = 0.7): any[] {
  const baseTime = new Date('2025-01-02').getTime()
  const day = 86400000

  // Interpolate close prices across all waypoints
  const closes: number[] = []
  for (let seg = 0; seg < waypoints.length - 1; seg++) {
    const [i0, p0] = waypoints[seg]
    const [i1, p1] = waypoints[seg + 1]
    for (let i = i0; i < i1; i++) {
      const t = (i - i0) / (i1 - i0)
      closes.push(p0 + (p1 - p0) * t)
    }
  }
  closes.push(waypoints[waypoints.length - 1][1])

  return closes.map((c, i) => {
    const prev = i > 0 ? closes[i - 1] : c
    const noise = (Math.random() - 0.5) * noiseAmt
    const open  = prev + (Math.random() - 0.5) * noiseAmt * 0.5
    const isUp  = c >= open
    const h = Math.max(open, c) + Math.abs(noise) + 0.3
    const l = Math.min(open, c) - Math.abs(noise) - 0.2
    return {
      timestamp: baseTime + i * day,
      open: +open.toFixed(2),
      high: +h.toFixed(2),
      low:  +l.toFixed(2),
      close: +(c + noise * 0.2).toFixed(2),
      volume: 2_000_000 + Math.random() * 3_000_000,
    }
  })
}

function ts(data: any[], i: number) {
  return data[Math.min(i, data.length - 1)]?.timestamp ?? 0
}

// ── Chart styles ─────────────────────────────────────────────────────────────
const CHART_STYLES = {
  grid: {
    show: true,
    horizontal: { show: true, size: 1, color: 'rgba(255,255,255,0.04)', style: 'dashed' as const, dashedValue: [4, 4] },
    vertical:   { show: true, size: 1, color: 'rgba(255,255,255,0.04)', style: 'dashed' as const, dashedValue: [4, 4] },
  },
  candle: {
    type: 'candle_solid' as const,
    bar: {
      upColor: '#22c55e', downColor: '#ef4444', noChangeColor: '#6b7280',
      upBorderColor: '#22c55e', downBorderColor: '#ef4444', noChangeBorderColor: '#6b7280',
      upWickColor: '#22c55e', downWickColor: '#ef4444', noChangeWickColor: '#6b7280',
    },
    priceMark: {
      show: true,
      high: { show: true, color: '#22c55e', textOffset: 5, textSize: 10, textFamily: 'monospace', textWeight: 'normal' },
      low:  { show: true, color: '#ef4444', textOffset: 5, textSize: 10, textFamily: 'monospace', textWeight: 'normal' },
      last: { show: true, upColor: '#22c55e', downColor: '#ef4444', noChangeColor: '#6b7280',
        line: { show: true, style: 'dashed' as const, dashedValue: [4, 4], size: 1 },
        text: { show: true, style: 'fill' as const, color: '#fff', size: 10, family: 'monospace', weight: 'bold',
          paddingLeft: 4, paddingRight: 4, paddingTop: 2, paddingBottom: 2,
          borderStyle: 'solid' as const, borderSize: 0, borderColor: 'transparent',
          borderDashedValue: [] as number[], borderRadius: 2 },
      },
    },
    tooltip: { showRule: 'always' as const, showType: 'standard' as const },
  },
  xAxis: { show: true, tickText: { color: '#9ca3af', size: 10, family: 'monospace', weight: 'normal' } },
  yAxis: { show: true, tickText: { color: '#9ca3af', size: 10, family: 'monospace', weight: 'normal' } },
  crosshair: {
    show: true,
    horizontal: { show: true, line: { show: true, style: 'dashed' as const, dashedValue: [4,4], size: 1, color: 'rgba(255,255,255,0.2)' },
      text: { show: true, style: 'fill' as const, color: '#fff', size: 10, family: 'monospace', weight: 'normal',
        paddingLeft: 4, paddingRight: 4, paddingTop: 2, paddingBottom: 2,
        borderStyle: 'solid' as const, borderSize: 1, borderColor: '#374151', borderDashedValue: [] as number[],
        borderRadius: 2, backgroundColor: '#1f2937' } },
    vertical: { show: true, line: { show: true, style: 'dashed' as const, dashedValue: [4,4], size: 1, color: 'rgba(255,255,255,0.2)' },
      text: { show: true, style: 'fill' as const, color: '#fff', size: 10, family: 'monospace', weight: 'normal',
        paddingLeft: 4, paddingRight: 4, paddingTop: 2, paddingBottom: 2,
        borderStyle: 'solid' as const, borderSize: 1, borderColor: '#374151', borderDashedValue: [] as number[],
        borderRadius: 2, backgroundColor: '#1f2937' } },
  },
}

// ── Color palette ─────────────────────────────────────────────────────────────
const IMPULSE_CLR   = '#22c55e'    // emerald — motive waves
const CORRECT_CLR   = '#f59e0b'    // amber   — corrective legs
const DIAGONAL_CLR  = '#818cf8'    // indigo  — diagonal waves
const BOUNDARY_CLR  = 'rgba(148,163,184,0.5)'  // slate — triangle boundary lines
const APEX_CLR      = 'rgba(148,163,184,0.35)' // lighter — apex projection
const PREDICT_BULL  = '#4ade80'
const PREDICT_BEAR  = '#f87171'
const LABEL_CLR     = 'rgba(255,255,255,0.55)'

// ── Helper: draw a wave segment + waveCircle label at the END pivot ──────────
function drawSeg(
  chart: Chart,
  data: any[],
  i0: number, p0: number,
  i1: number, p1: number,
  color: string,
  lineWidth = 3,
  dash: number[] = [],
) {
  chart.createOverlay({
    name: 'patternTrendline',
    points: [
      { timestamp: ts(data, i0), value: p0 },
      { timestamp: ts(data, i1), value: p1 },
    ],
    extendData: { color, lineWidth, dash },
  })
}

function drawCircle(
  chart: Chart,
  data: any[],
  barIdx: number,
  price: number,
  label: string,
  color: string,
  offsetPct = 0.03,
  above = true,
) {
  const offset = price * offsetPct * (above ? 1 : -1)
  chart.createOverlay({
    name: 'waveCircle',
    points: [{ timestamp: ts(data, barIdx), value: price + offset }],
    extendData: { text: label, fillColor: color, textColor: '#fff', radius: 11 },
  })
}

function badge(chart: Chart, data: any[], barIdx: number, price: number, text: string, color: string) {
  chart.createOverlay({
    name: 'patternBadge',
    points: [{ timestamp: ts(data, barIdx), value: price }],
    extendData: { text, textColor: color },
  })
}

// ── Pattern definitions ───────────────────────────────────────────────────────

interface EWPattern {
  name: string
  subtitle: string
  direction: 'bullish' | 'bearish' | 'neutral'
  buildData: () => any[]
  draw: (chart: Chart, data: any[]) => void
}

const EW_PATTERNS: EWPattern[] = [

  // ── 1. CONTRACTING TRIANGLE (W4) ────────────────────────────────────────────
  {
    name: 'Contracting Triangle',
    subtitle: 'W4 · ABCDE · 3-3-3-3-3 · Bullish',
    direction: 'bullish',
    buildData: () => buildBars([
      [0, 82],   // start
      [8, 97],   // W1 high
      [13, 88],  // W2 low
      [23, 122], // W3 high
      [29, 108], // W4-A low     ← triangle starts
      [33, 116], // W4-B high
      [37, 110], // W4-C low
      [40, 114], // W4-D high
      [43, 111], // W4-E low     ← triangle ends (apex ~bar 48)
      [52, 132], // W5 high (thrust)
    ]),
    draw: (chart, data) => {
      // ── W1-W3 context ──
      drawSeg(chart, data, 0,  82,   8,  97,  IMPULSE_CLR, 3)
      drawSeg(chart, data, 8,  97,  13,  88,  CORRECT_CLR, 3)
      drawSeg(chart, data, 13, 88,  23, 122,  IMPULSE_CLR, 3)
      drawCircle(chart, data,  8,  97, '1', IMPULSE_CLR, 0.025, true)
      drawCircle(chart, data, 13,  88, '2', CORRECT_CLR, 0.025, false)
      drawCircle(chart, data, 23, 122, '3', IMPULSE_CLR, 0.025, true)

      // ── W4 Triangle ABCDE ──
      drawSeg(chart, data, 23, 122, 29, 108, CORRECT_CLR, 2.5)  // A
      drawSeg(chart, data, 29, 108, 33, 116, CORRECT_CLR, 2.5)  // B
      drawSeg(chart, data, 33, 116, 37, 110, CORRECT_CLR, 2.5)  // C
      drawSeg(chart, data, 37, 110, 40, 114, CORRECT_CLR, 2.5)  // D
      drawSeg(chart, data, 40, 114, 43, 111, CORRECT_CLR, 2.5)  // E

      // Wave labels
      drawCircle(chart, data, 29, 108, 'A', CORRECT_CLR, 0.025, false)
      drawCircle(chart, data, 33, 116, 'B', CORRECT_CLR, 0.025, true)
      drawCircle(chart, data, 37, 110, 'C', CORRECT_CLR, 0.025, false)
      drawCircle(chart, data, 40, 114, 'D', CORRECT_CLR, 0.025, true)
      drawCircle(chart, data, 43, 111, 'E', CORRECT_CLR, 0.025, false)
      drawCircle(chart, data, 23, 122, '4', CORRECT_CLR, 0.025, true)  // W4 label at start

      // ── Triangle boundary trendlines ──
      // Upper: B(116) → D(114) — declining → extended to apex ~bar 49
      drawSeg(chart, data, 33, 116, 49, 112,  BOUNDARY_CLR, 1.2, [5, 4])
      // Lower: A(108) → C(110) → rising → extended to apex
      drawSeg(chart, data, 29, 108, 49, 112,  BOUNDARY_CLR, 1.2, [5, 4])
      // Apex marker
      chart.createOverlay({
        name: 'patternLabel',
        points: [{ timestamp: ts(data, 49), value: 112 }],
        extendData: { text: 'apex', color: LABEL_CLR, offsetY: -12, fontSize: 9 },
      })

      // ── W5 thrust ──
      drawSeg(chart, data, 43, 111, 52, 132, IMPULSE_CLR, 3)
      drawCircle(chart, data, 52, 132, '5', IMPULSE_CLR, 0.025, true)

      // ── Thrust target level ──
      chart.createOverlay({ name: 'patternLevel',
        points: [{ timestamp: ts(data, 0), value: 132 }],
        extendData: { color: '#22c55e', label: 'THRUST TARGET $132 · E×1.618' },
      })

      badge(chart, data, 1, 136, 'CONTRACTING TRIANGLE  W4 · 91%', IMPULSE_CLR)

      // ── Prediction ──
      chart.createOverlay({ name: 'predictionPath',
        points: [
          { timestamp: ts(data, 43), value: 111 },
          { timestamp: ts(data, 46), value: 118 },
          { timestamp: ts(data, 49), value: 125 },
          { timestamp: ts(data, 52), value: 132 },
        ],
        extendData: { color: PREDICT_BULL, lineWidth: 2, numPoints: 4 },
      })
      chart.createOverlay({ name: 'predictionCone',
        points: [
          { timestamp: ts(data, 43), value: 111 },
          { timestamp: ts(data, 52), value: 138 },
          { timestamp: ts(data, 52), value: 126 },
        ],
        extendData: { color: 'rgba(74,222,128,0.06)', borderColor: 'rgba(74,222,128,0.2)' },
      })
      chart.createOverlay({ name: 'predictionZone',
        points: [
          { timestamp: ts(data, 48), value: 130 },
          { timestamp: ts(data, 54), value: 136 },
        ],
        extendData: { color: 'rgba(34,197,94,0.08)', borderColor: 'rgba(34,197,94,0.25)', label: 'THRUST ZONE', labelColor: '#4ade80' },
      })
      chart.createOverlay({ name: 'predictionZone',
        points: [
          { timestamp: ts(data, 43), value: 107 },
          { timestamp: ts(data, 54), value: 109 },
        ],
        extendData: { color: 'rgba(239,68,68,0.06)', borderColor: 'rgba(239,68,68,0.2)', label: 'STOP: E invalidated', labelColor: '#f87171' },
      })
    },
  },

  // ── 2. BARRIER TRIANGLE (W4) ─────────────────────────────────────────────────
  {
    name: 'Barrier Triangle',
    subtitle: 'W4 · ABCDE · Flat Upper Boundary · Bullish',
    direction: 'bullish',
    buildData: () => buildBars([
      [0, 82],
      [8, 97],
      [13, 88],
      [23, 122],
      [28, 107],  // W4-A low
      [32, 115],  // W4-B high  ← flat upper = 115
      [36, 109],  // W4-C low
      [40, 115],  // W4-D high  ← touches flat 115 again
      [44, 111],  // W4-E low
      [53, 132],  // W5 high
    ]),
    draw: (chart, data) => {
      // W1-W3
      drawSeg(chart, data, 0,  82,  8,  97,  IMPULSE_CLR, 3)
      drawSeg(chart, data, 8,  97, 13,  88,  CORRECT_CLR, 3)
      drawSeg(chart, data, 13, 88, 23, 122,  IMPULSE_CLR, 3)
      drawCircle(chart, data,  8,  97, '1', IMPULSE_CLR, 0.025, true)
      drawCircle(chart, data, 13,  88, '2', CORRECT_CLR, 0.025, false)
      drawCircle(chart, data, 23, 122, '3', IMPULSE_CLR, 0.025, true)

      // Triangle ABCDE
      drawSeg(chart, data, 23, 122, 28, 107, CORRECT_CLR, 2.5)
      drawSeg(chart, data, 28, 107, 32, 115, CORRECT_CLR, 2.5)
      drawSeg(chart, data, 32, 115, 36, 109, CORRECT_CLR, 2.5)
      drawSeg(chart, data, 36, 109, 40, 115, CORRECT_CLR, 2.5)
      drawSeg(chart, data, 40, 115, 44, 111, CORRECT_CLR, 2.5)

      drawCircle(chart, data, 23, 122, '4', CORRECT_CLR, 0.025, true)
      drawCircle(chart, data, 28, 107, 'A', CORRECT_CLR, 0.025, false)
      drawCircle(chart, data, 32, 115, 'B', CORRECT_CLR, 0.025, true)
      drawCircle(chart, data, 36, 109, 'C', CORRECT_CLR, 0.025, false)
      drawCircle(chart, data, 40, 115, 'D', CORRECT_CLR, 0.025, true)
      drawCircle(chart, data, 44, 111, 'E', CORRECT_CLR, 0.025, false)

      // ── BARRIER: upper line is FLAT at 115, lower line RISES ──
      // Flat upper: B(115) → D(115) → extended flat beyond E
      drawSeg(chart, data, 28, 115, 50, 115, BOUNDARY_CLR, 1.5, [5, 4])
      // Rising lower: A(107) → C(109) → E(111)
      drawSeg(chart, data, 28, 107, 50, 113, BOUNDARY_CLR, 1.2, [5, 4])

      // Annotation: flat boundary
      chart.createOverlay({
        name: 'patternLabel',
        points: [{ timestamp: ts(data, 42), value: 115 }],
        extendData: { text: 'flat barrier', color: LABEL_CLR, offsetY: -12, fontSize: 9 },
      })

      // W5
      drawSeg(chart, data, 44, 111, 53, 132, IMPULSE_CLR, 3)
      drawCircle(chart, data, 53, 132, '5', IMPULSE_CLR, 0.025, true)

      chart.createOverlay({ name: 'patternLevel',
        points: [{ timestamp: ts(data, 0), value: 115 }],
        extendData: { color: '#94a3b8', label: 'BARRIER / FLAT UPPER' },
      })
      chart.createOverlay({ name: 'patternLevel',
        points: [{ timestamp: ts(data, 0), value: 132 }],
        extendData: { color: '#22c55e', label: 'THRUST TARGET $132' },
      })
      badge(chart, data, 1, 137, 'BARRIER TRIANGLE  W4 · 88%', IMPULSE_CLR)

      chart.createOverlay({ name: 'predictionPath',
        points: [
          { timestamp: ts(data, 44), value: 111 },
          { timestamp: ts(data, 47), value: 119 },
          { timestamp: ts(data, 50), value: 126 },
          { timestamp: ts(data, 53), value: 132 },
        ],
        extendData: { color: PREDICT_BULL, lineWidth: 2, numPoints: 4 },
      })
      chart.createOverlay({ name: 'predictionCone',
        points: [
          { timestamp: ts(data, 44), value: 111 },
          { timestamp: ts(data, 53), value: 139 },
          { timestamp: ts(data, 53), value: 126 },
        ],
        extendData: { color: 'rgba(74,222,128,0.06)', borderColor: 'rgba(74,222,128,0.2)' },
      })
      chart.createOverlay({ name: 'predictionZone',
        points: [
          { timestamp: ts(data, 44), value: 106 },
          { timestamp: ts(data, 55), value: 108 },
        ],
        extendData: { color: 'rgba(239,68,68,0.06)', borderColor: 'rgba(239,68,68,0.2)', label: 'STOP', labelColor: '#f87171' },
      })
    },
  },

  // ── 3. EXPANDING TRIANGLE (B wave of ZigZag) ─────────────────────────────────
  {
    name: 'Expanding Triangle',
    subtitle: 'B wave · ABCDE · Diverging Boundaries · Bearish',
    direction: 'bearish',
    buildData: () => buildBars([
      [0, 118],  // context top
      [8, 100],  // A wave low
      [12, 108], // B-A high
      [16, 100], // B-B low    (= A-wave low, trendlines start diverging)
      [21, 114], // B-C high   (> B-A)
      [26, 96],  // B-D low    (< B-B, below A-wave low)
      [32, 121], // B-E high   (> B-C, expanding up)
      [45, 74],  // C wave low (strong sell-off)
    ]),
    draw: (chart, data) => {
      // A wave (prior downtrend context)
      drawSeg(chart, data, 0, 118, 8, 100, '#ef4444', 2.5)
      drawCircle(chart, data, 8, 100, 'A', '#ef4444', 0.025, false)

      // B wave = expanding triangle ABCDE
      drawSeg(chart, data,  8, 100, 12, 108, CORRECT_CLR, 2.5)  // B-A
      drawSeg(chart, data, 12, 108, 16, 100, CORRECT_CLR, 2.5)  // B-B
      drawSeg(chart, data, 16, 100, 21, 114, CORRECT_CLR, 2.5)  // B-C
      drawSeg(chart, data, 21, 114, 26,  96, CORRECT_CLR, 2.5)  // B-D
      drawSeg(chart, data, 26,  96, 32, 121, CORRECT_CLR, 2.5)  // B-E

      drawCircle(chart, data,  8, 100, 'B', CORRECT_CLR, 0.025, true)   // B-wave label at start
      drawCircle(chart, data, 12, 108, 'a', CORRECT_CLR, 0.025, true)
      drawCircle(chart, data, 16, 100, 'b', CORRECT_CLR, 0.025, false)
      drawCircle(chart, data, 21, 114, 'c', CORRECT_CLR, 0.025, true)
      drawCircle(chart, data, 26,  96, 'd', CORRECT_CLR, 0.025, false)
      drawCircle(chart, data, 32, 121, 'e', CORRECT_CLR, 0.025, true)

      // ── EXPANDING boundaries: upper rises, lower falls ──
      // Upper: B-A(108) → B-C(114) → B-E(121) — rising
      drawSeg(chart, data, 12, 108, 32, 121, BOUNDARY_CLR, 1.5, [5, 4])
      // Lower: B-B(100) → B-D(96) — declining
      drawSeg(chart, data, 16, 100, 30, 94,  BOUNDARY_CLR, 1.2, [5, 4])

      chart.createOverlay({
        name: 'patternLabel',
        points: [{ timestamp: ts(data, 22), value: 114 }],
        extendData: { text: 'diverging', color: LABEL_CLR, offsetY: -12, fontSize: 9 },
      })

      // C wave (bear impulse)
      drawSeg(chart, data, 32, 121, 45, 74, '#ef4444', 3)
      drawCircle(chart, data, 45, 74, 'C', '#ef4444', 0.025, false)

      chart.createOverlay({ name: 'patternLevel',
        points: [{ timestamp: ts(data, 0), value: 74 }],
        extendData: { color: '#22c55e', label: 'C TARGET $74 · A=C' },
      })
      chart.createOverlay({ name: 'patternLevel',
        points: [{ timestamp: ts(data, 0), value: 121 }],
        extendData: { color: '#ef4444', label: 'E HIGH / SHORT ENTRY' },
      })

      badge(chart, data, 1, 128, 'EXPANDING TRIANGLE  B wave · 84%', '#f87171')

      // Prediction: C wave down
      chart.createOverlay({ name: 'predictionPath',
        points: [
          { timestamp: ts(data, 32), value: 121 },
          { timestamp: ts(data, 36), value: 109 },
          { timestamp: ts(data, 40), value: 95 },
          { timestamp: ts(data, 45), value: 74 },
        ],
        extendData: { color: PREDICT_BEAR, lineWidth: 2, numPoints: 4 },
      })
      chart.createOverlay({ name: 'predictionCone',
        points: [
          { timestamp: ts(data, 32), value: 121 },
          { timestamp: ts(data, 45), value: 64 },
          { timestamp: ts(data, 45), value: 82 },
        ],
        extendData: { color: 'rgba(248,113,113,0.06)', borderColor: 'rgba(248,113,113,0.2)' },
      })
      chart.createOverlay({ name: 'predictionZone',
        points: [
          { timestamp: ts(data, 40), value: 68 },
          { timestamp: ts(data, 48), value: 80 },
        ],
        extendData: { color: 'rgba(34,197,94,0.08)', borderColor: 'rgba(34,197,94,0.25)', label: 'C TARGET ZONE', labelColor: '#4ade80' },
      })
      chart.createOverlay({ name: 'predictionZone',
        points: [
          { timestamp: ts(data, 32), value: 123 },
          { timestamp: ts(data, 48), value: 126 },
        ],
        extendData: { color: 'rgba(239,68,68,0.06)', borderColor: 'rgba(239,68,68,0.2)', label: 'STOP: above E', labelColor: '#f87171' },
      })
    },
  },

  // ── 4. DOUBLE ZIGZAG W-X-Y (W2) ─────────────────────────────────────────────
  {
    name: 'Double Zigzag  W-X-Y',
    subtitle: 'W2 · Two linked ABC corrections · Bearish',
    direction: 'bearish',
    buildData: () => buildBars([
      [0,  88],   // start
      [10, 122],  // W1 high
      // W zigzag (first ABC)
      [14, 112],  // W-A low
      [17, 118],  // W-B high (B retraces ~55% of A)
      [22, 104],  // W-C low  (C ≈ A)
      // X wave (small linking bounce)
      [26, 110],  // X high
      // Y zigzag (second ABC, deeper)
      [30, 102],  // Y-A low
      [33, 107],  // Y-B high
      [38,  95],  // Y-C low (Y ≈ W or larger)
      // W3 starts
      [48, 138],  // W3 partial (context)
    ]),
    draw: (chart, data) => {
      // W1 motive
      drawSeg(chart, data, 0, 88, 10, 122, IMPULSE_CLR, 3)
      drawCircle(chart, data, 10, 122, '1', IMPULSE_CLR, 0.025, true)

      // ── W (first zigzag) ──
      const W_CLR = '#f59e0b'  // amber for first zigzag
      drawSeg(chart, data, 10, 122, 14, 112, W_CLR, 2.5)  // W-A
      drawSeg(chart, data, 14, 112, 17, 118, W_CLR, 2.5)  // W-B
      drawSeg(chart, data, 17, 118, 22, 104, W_CLR, 2.5)  // W-C

      drawCircle(chart, data, 10, 122, 'W', W_CLR, 0.025, true)
      drawCircle(chart, data, 14, 112, 'A', W_CLR, 0.025, false)
      drawCircle(chart, data, 17, 118, 'B', W_CLR, 0.025, true)
      drawCircle(chart, data, 22, 104, 'C', W_CLR, 0.025, false)

      // ── X (linking wave — distinct color) ──
      const X_CLR = '#818cf8'  // indigo — stands out as the connector
      drawSeg(chart, data, 22, 104, 26, 110, X_CLR, 2.5, [4, 3])
      drawCircle(chart, data, 26, 110, 'X', X_CLR, 0.025, true)

      // ── Y (second zigzag — deeper, same color family but brighter) ──
      const Y_CLR = '#fbbf24'  // golden amber
      drawSeg(chart, data, 26, 110, 30, 102, Y_CLR, 2.5)  // Y-A
      drawSeg(chart, data, 30, 102, 33, 107, Y_CLR, 2.5)  // Y-B
      drawSeg(chart, data, 33, 107, 38,  95, Y_CLR, 2.5)  // Y-C

      drawCircle(chart, data, 26, 110, 'Y',   Y_CLR, 0.025, true)
      drawCircle(chart, data, 30, 102, 'A',   Y_CLR, 0.025, false)
      drawCircle(chart, data, 33, 107, 'B',   Y_CLR, 0.025, true)
      drawCircle(chart, data, 38,  95, 'C',   Y_CLR, 0.025, false)
      drawCircle(chart, data, 38,  95, '2',   '#ef4444', 0.055, false)  // W2 label

      // W3 context
      drawSeg(chart, data, 38, 95, 48, 138, IMPULSE_CLR, 2.5)
      drawCircle(chart, data, 48, 138, '3', IMPULSE_CLR, 0.025, true)

      // Levels
      chart.createOverlay({ name: 'patternLevel',
        points: [{ timestamp: ts(data, 0), value: 95 }],
        extendData: { color: '#22c55e', label: 'W2 END / W3 ENTRY ~$95' },
      })
      chart.createOverlay({ name: 'patternLevel',
        points: [{ timestamp: ts(data, 0), value: 88 }],
        extendData: { color: '#ef4444', label: 'W2 INVALIDATION: below W1 start' },
      })

      badge(chart, data, 1, 143, 'DOUBLE ZIGZAG  W2 · 89%', '#fbbf24')

      // Annotation: label the two zigzags
      chart.createOverlay({
        name: 'patternLabel',
        points: [{ timestamp: ts(data, 16), value: 118 }],
        extendData: { text: 'Zigzag W', color: W_CLR, offsetY: -14, fontSize: 9 },
      })
      chart.createOverlay({
        name: 'patternLabel',
        points: [{ timestamp: ts(data, 32), value: 107 }],
        extendData: { text: 'Zigzag Y', color: Y_CLR, offsetY: -14, fontSize: 9 },
      })

      // Prediction: W3 extension
      chart.createOverlay({ name: 'predictionPath',
        points: [
          { timestamp: ts(data, 38), value: 95 },
          { timestamp: ts(data, 42), value: 110 },
          { timestamp: ts(data, 46), value: 126 },
          { timestamp: ts(data, 50), value: 140 },
        ],
        extendData: { color: PREDICT_BULL, lineWidth: 2, numPoints: 4 },
      })
      chart.createOverlay({ name: 'predictionCone',
        points: [
          { timestamp: ts(data, 38), value: 95 },
          { timestamp: ts(data, 50), value: 150 },
          { timestamp: ts(data, 50), value: 132 },
        ],
        extendData: { color: 'rgba(74,222,128,0.06)', borderColor: 'rgba(74,222,128,0.2)' },
      })
      chart.createOverlay({ name: 'predictionZone',
        points: [
          { timestamp: ts(data, 44), value: 136 },
          { timestamp: ts(data, 52), value: 148 },
        ],
        extendData: { color: 'rgba(34,197,94,0.08)', borderColor: 'rgba(34,197,94,0.25)', label: 'W3 TARGET ZONE', labelColor: '#4ade80' },
      })
      chart.createOverlay({ name: 'predictionZone',
        points: [
          { timestamp: ts(data, 38), value: 87 },
          { timestamp: ts(data, 52), value: 89 },
        ],
        extendData: { color: 'rgba(239,68,68,0.06)', borderColor: 'rgba(239,68,68,0.2)', label: 'STOP: below W1 start', labelColor: '#f87171' },
      })
    },
  },

  // ── 5. ENDING DIAGONAL + THROW-OVER (W5) ────────────────────────────────────
  {
    name: 'Ending Diagonal + Throw-Over',
    subtitle: 'W5 · 3-3-3-3-3 · W4 overlaps W1 · Channel breach → Reversal',
    direction: 'bearish',
    buildData: () => buildBars([
      [0,  100],  // W1 start
      [7,  118],  // W1 high
      [12, 106],  // W2 low
      [22, 138],  // W3 high
      [28, 118],  // W4 low  ← overlaps W1 end price zone
      // Ending diagonal W5 sub-waves
      [32, 130],  // D1 high
      [35, 121],  // D2 low  (overlaps D1 zone = W4/W1 overlap confirmed)
      [39, 139],  // D3 high
      [43, 128],  // D4 low  (overlaps D1 high zone)
      [47, 148],  // D5 high — THROW-OVER: exceeds channel by ~3pts
      [54, 108],  // reversal low (sharp sell-off after throw-over)
    ]),
    draw: (chart, data) => {
      // W1-W4 context
      drawSeg(chart, data, 0, 100,  7, 118, IMPULSE_CLR, 3)
      drawSeg(chart, data, 7, 118, 12, 106, CORRECT_CLR, 3)
      drawSeg(chart, data, 12, 106, 22, 138, IMPULSE_CLR, 3)
      drawSeg(chart, data, 22, 138, 28, 118, CORRECT_CLR, 3)
      drawCircle(chart, data,  7, 118, '1', IMPULSE_CLR, 0.025, true)
      drawCircle(chart, data, 12, 106, '2', CORRECT_CLR, 0.025, false)
      drawCircle(chart, data, 22, 138, '3', IMPULSE_CLR, 0.025, true)
      drawCircle(chart, data, 28, 118, '4', CORRECT_CLR, 0.025, false)

      // W4 overlap annotation
      chart.createOverlay({
        name: 'patternLabel',
        points: [{ timestamp: ts(data, 28), value: 118 }],
        extendData: { text: 'W4 overlaps W1', color: '#94a3b8', offsetY: 18, fontSize: 9 },
      })

      // ── Ending diagonal sub-waves (all corrective = 3-3-3-3-3) ──
      drawSeg(chart, data, 28, 118, 32, 130, DIAGONAL_CLR, 2.5)  // D1
      drawSeg(chart, data, 32, 130, 35, 121, DIAGONAL_CLR, 2.5)  // D2
      drawSeg(chart, data, 35, 121, 39, 139, DIAGONAL_CLR, 2.5)  // D3
      drawSeg(chart, data, 39, 139, 43, 128, DIAGONAL_CLR, 2.5)  // D4
      drawSeg(chart, data, 43, 128, 47, 148, DIAGONAL_CLR, 3.5)  // D5 (throw-over = bolder)

      drawCircle(chart, data, 28, 118, '5',  DIAGONAL_CLR, 0.025, false)
      drawCircle(chart, data, 32, 130, 'i',  DIAGONAL_CLR, 0.025, true)
      drawCircle(chart, data, 35, 121, 'ii', DIAGONAL_CLR, 0.025, false)
      drawCircle(chart, data, 39, 139, 'iii',DIAGONAL_CLR, 0.025, true)
      drawCircle(chart, data, 43, 128, 'iv', DIAGONAL_CLR, 0.025, false)
      drawCircle(chart, data, 47, 148, 'v',  '#f87171',    0.025, true)  // red = overthrow

      // ── Channel lines (D1-D3 upper, D2-D4 lower) ──
      // Upper: D1(130) at bar32 → D3(139) at bar39. Slope = 9/7 per bar → bar47: 139 + 9*(47-39)/7 = 139+10.3 = 149.3... close enough
      // Actually: D1 peak(130) → D3 peak(139). At bar47 extrapolated = ~149. D5 = 148, so slightly below — let me use D1→D3 trendline
      drawSeg(chart, data, 32, 130, 49, 144, BOUNDARY_CLR, 1.2, [5, 4])  // upper channel extrapolated
      // Lower: D2(121) at bar35 → D4(128) at bar43. Slope = 7/8. At bar49: 128 + 7*(49-43)/8 = 128+5.25 = 133.25
      drawSeg(chart, data, 35, 121, 49, 133, BOUNDARY_CLR, 1.2, [5, 4])  // lower channel

      // ── Throw-over annotation ──
      chart.createOverlay({
        name: 'patternLabel',
        points: [{ timestamp: ts(data, 47), value: 148 }],
        extendData: { text: '⚡ THROW-OVER', color: '#f87171', offsetY: -14, fontSize: 10 },
      })

      // Channel boundary note
      chart.createOverlay({
        name: 'patternLabel',
        points: [{ timestamp: ts(data, 46), value: 144 }],
        extendData: { text: 'channel', color: LABEL_CLR, offsetY: -10, fontSize: 9 },
      })

      // ── Reversal ──
      drawSeg(chart, data, 47, 148, 54, 108, '#ef4444', 3.5)

      chart.createOverlay({ name: 'patternLevel',
        points: [{ timestamp: ts(data, 0), value: 118 }],
        extendData: { color: '#f59e0b', label: 'W2 END / W4 START = RETRACE TARGET' },
      })
      chart.createOverlay({ name: 'patternLevel',
        points: [{ timestamp: ts(data, 0), value: 100 }],
        extendData: { color: '#ef4444', label: 'W1 START (max retrace)' },
      })

      badge(chart, data, 1, 157, 'ENDING DIAGONAL  W5 + THROW-OVER · 87%', '#818cf8')

      // Prediction: reversal after throw-over
      chart.createOverlay({ name: 'predictionPath',
        points: [
          { timestamp: ts(data, 47), value: 148 },
          { timestamp: ts(data, 49), value: 138 },
          { timestamp: ts(data, 51), value: 124 },
          { timestamp: ts(data, 54), value: 108 },
        ],
        extendData: { color: PREDICT_BEAR, lineWidth: 2, numPoints: 4 },
      })
      chart.createOverlay({ name: 'predictionCone',
        points: [
          { timestamp: ts(data, 47), value: 148 },
          { timestamp: ts(data, 54), value: 96 },
          { timestamp: ts(data, 54), value: 118 },
        ],
        extendData: { color: 'rgba(248,113,113,0.06)', borderColor: 'rgba(248,113,113,0.2)' },
      })
      chart.createOverlay({ name: 'predictionZone',
        points: [
          { timestamp: ts(data, 51), value: 100 },
          { timestamp: ts(data, 57), value: 118 },
        ],
        extendData: { color: 'rgba(34,197,94,0.08)', borderColor: 'rgba(34,197,94,0.25)', label: 'POST-DIAGONAL TARGET', labelColor: '#4ade80' },
      })
      chart.createOverlay({ name: 'predictionZone',
        points: [
          { timestamp: ts(data, 47), value: 148 },
          { timestamp: ts(data, 57), value: 152 },
        ],
        extendData: { color: 'rgba(239,68,68,0.06)', borderColor: 'rgba(239,68,68,0.2)', label: 'STOP: above throw-over high', labelColor: '#f87171' },
      })
    },
  },

  // ── 6. IMPULSE W3 EXTENSION ──────────────────────────────────────────────────
  {
    name: 'Impulse · W3 Extension',
    subtitle: '5-wave motive · W3 = 2.618×W1 · 9-wave look · Bullish',
    direction: 'bullish',
    buildData: () => buildBars([
      [0,  80],   // W1 start
      [8,  96],   // W1 high
      [13, 85],   // W2 low (59% retrace)
      // W3 extended — subdivides into 5 visible sub-waves
      [17, 98],   // W3 w1 high
      [20, 91],   // W3 w2 low
      [27, 120],  // W3 w3 high (extended sub-wave — longest)
      [31, 112],  // W3 w4 low
      [37, 131],  // W3 w5 high (W3 complete)
      [43, 118],  // W4 low (30% retrace of W3)
      [52, 148],  // W5 high (W5 ≈ W1 equality = 68% of W3)
    ]),
    draw: (chart, data) => {
      const EXT_CLR = '#34d399'  // teal — extended wave highlight

      // ── W1 ──
      drawSeg(chart, data, 0, 80, 8, 96, IMPULSE_CLR, 3)
      drawCircle(chart, data, 0, 80, '0', '#6b7280', 0.025, false)
      drawCircle(chart, data, 8, 96, '1', IMPULSE_CLR, 0.025, true)

      // ── W2 ──
      drawSeg(chart, data, 8, 96, 13, 85, CORRECT_CLR, 3)
      drawCircle(chart, data, 13, 85, '2', CORRECT_CLR, 0.025, false)

      // ── W3 (extended — draw sub-waves in teal with smaller circles) ──
      // Main W3 outline
      drawSeg(chart, data, 13, 85, 17, 98,  EXT_CLR, 2.5)  // w3-1
      drawSeg(chart, data, 17, 98, 20, 91,  EXT_CLR, 2.5)  // w3-2
      drawSeg(chart, data, 20, 91, 27, 120, EXT_CLR, 3.5)  // w3-3 (sub-extended — bold)
      drawSeg(chart, data, 27, 120, 31, 112, EXT_CLR, 2.5) // w3-4
      drawSeg(chart, data, 31, 112, 37, 131, EXT_CLR, 2.5) // w3-5

      // Sub-wave circles (smaller, roman numerals)
      drawCircle(chart, data, 17, 98,  'i',   EXT_CLR, 0.020, true)
      drawCircle(chart, data, 20, 91,  'ii',  EXT_CLR, 0.020, false)
      drawCircle(chart, data, 27, 120, 'iii', EXT_CLR, 0.020, true)
      drawCircle(chart, data, 31, 112, 'iv',  EXT_CLR, 0.020, false)
      drawCircle(chart, data, 37, 131, 'v',   EXT_CLR, 0.020, true)

      // W3 boundary label
      drawCircle(chart, data, 13, 85, '3', EXT_CLR, 0.055, false)
      chart.createOverlay({
        name: 'patternLabel',
        points: [{ timestamp: ts(data, 25), value: 131 }],
        extendData: { text: '← W3 EXTENDED (2.62×W1)', color: EXT_CLR, offsetY: -12, fontSize: 9 },
      })

      // ── W4 ──
      drawSeg(chart, data, 37, 131, 43, 118, CORRECT_CLR, 3)
      drawCircle(chart, data, 43, 118, '4', CORRECT_CLR, 0.025, false)

      // ── W5 (= W1 equality) ──
      drawSeg(chart, data, 43, 118, 52, 148, IMPULSE_CLR, 3)
      drawCircle(chart, data, 52, 148, '5', IMPULSE_CLR, 0.025, true)

      // Fib relationship labels
      chart.createOverlay({ name: 'patternLevel',
        points: [{ timestamp: ts(data, 0), value: 148 }],
        extendData: { color: '#22c55e', label: 'W5 HIGH · W5 ≈ W1 equality' },
      })
      chart.createOverlay({ name: 'patternLevel',
        points: [{ timestamp: ts(data, 0), value: 131 }],
        extendData: { color: '#34d399', label: 'W3 TOP · 2.618×W1' },
      })
      chart.createOverlay({ name: 'patternLevel',
        points: [{ timestamp: ts(data, 0), value: 85 }],
        extendData: { color: '#f59e0b', label: 'W2 LOW · 0.618 retrace' },
      })

      badge(chart, data, 1, 158, 'IMPULSE  W3 EXTENSION · 9-wave look · 94%', EXT_CLR)

      // Channel: 0-2 base → parallel through W4 for W5 target
      // W1-start(80 at bar0) → W2-end(85 at bar13): slope = 5/13 per bar
      // Parallel through W4-end(118 at bar43): W5 channel target = 118 + slope * (52-43) = 118 + 5*9/13 = 121.5 ≈ nah
      // Better: draw 1-3 channel line → parallel through 4 = W5 target
      // W1-end(96 at bar8) → W3-end(131 at bar37): slope = 35/29. At bar52: 131 + 35*(52-37)/29 = 131+18 = 149 ≈ 148 ✓
      drawSeg(chart, data, 8, 96, 52, 150, BOUNDARY_CLR, 1.0, [5, 4])   // 1-3 channel upper
      drawSeg(chart, data, 0, 80, 52, 132, BOUNDARY_CLR, 1.0, [5, 4])   // 0-2 base lower

      chart.createOverlay({
        name: 'patternLabel',
        points: [{ timestamp: ts(data, 48), value: 149 }],
        extendData: { text: '1-3 channel', color: LABEL_CLR, offsetY: -10, fontSize: 9 },
      })

      // Prediction: continuation above W5 high
      chart.createOverlay({ name: 'predictionPath',
        points: [
          { timestamp: ts(data, 52), value: 148 },
          { timestamp: ts(data, 54), value: 140 },  // W-A of correction
          { timestamp: ts(data, 57), value: 144 },  // W-B bounce
          { timestamp: ts(data, 60), value: 133 },  // W-C low
        ],
        extendData: { color: '#f59e0b', lineWidth: 2, numPoints: 4 },  // amber = corrective
      })
      chart.createOverlay({ name: 'predictionCone',
        points: [
          { timestamp: ts(data, 52), value: 148 },
          { timestamp: ts(data, 60), value: 124 },
          { timestamp: ts(data, 60), value: 142 },
        ],
        extendData: { color: 'rgba(245,158,11,0.05)', borderColor: 'rgba(245,158,11,0.2)' },
      })
      chart.createOverlay({ name: 'predictionZone',
        points: [
          { timestamp: ts(data, 55), value: 128 },
          { timestamp: ts(data, 62), value: 138 },
        ],
        extendData: { color: 'rgba(34,197,94,0.07)', borderColor: 'rgba(34,197,94,0.2)', label: 'W4 CORRECTION TARGET', labelColor: '#4ade80' },
      })
      chart.createOverlay({ name: 'predictionZone',
        points: [
          { timestamp: ts(data, 52), value: 149 },
          { timestamp: ts(data, 62), value: 153 },
        ],
        extendData: { color: 'rgba(239,68,68,0.06)', borderColor: 'rgba(239,68,68,0.2)', label: 'STOP: above W5 high', labelColor: '#f87171' },
      })
    },
  },

]

// ── Chart Panel ───────────────────────────────────────────────────────────────
function EWPanel({ pattern }: { pattern: EWPattern }) {
  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<Chart | null>(null)

  useEffect(() => {
    if (!containerRef.current) return
    registerPatternOverlays()

    if (chartRef.current) {
      dispose(chartRef.current)
      chartRef.current = null
    }

    const chart = init(containerRef.current, { styles: CHART_STYLES as any })
    if (!chart) return
    chartRef.current = chart

    const data = pattern.buildData()

    chart.setDataLoader({
      getBars: async (params: any) => { params.callback(data, false) },
    } as any)
    ;(chart as any).setSymbol({ ticker: 'EW', pricePrecision: 2, volumePrecision: 0 })
    ;(chart as any).setPeriod({ type: 'day', span: 1 })

    setTimeout(() => {
      if (chartRef.current) pattern.draw(chartRef.current, data)
    }, 300)

    return () => {
      if (chartRef.current) { dispose(chartRef.current); chartRef.current = null }
    }
  }, [pattern])

  const dirColor = pattern.direction === 'bullish'
    ? 'text-emerald-400' : pattern.direction === 'bearish'
    ? 'text-red-400' : 'text-amber-400'
  const dirBg = pattern.direction === 'bullish'
    ? 'bg-emerald-500/15 border-emerald-500/30' : pattern.direction === 'bearish'
    ? 'bg-red-500/15 border-red-500/30' : 'bg-amber-500/15 border-amber-500/30'

  return (
    <div className="flex flex-col rounded-xl border border-gray-800 overflow-hidden bg-gray-950">
      <div className="flex items-center justify-between px-3 py-2 bg-gray-900/80 border-b border-gray-800">
        <div>
          <span className="text-white font-bold text-sm font-mono">{pattern.name}</span>
          <span className="ml-2 text-gray-500 text-[10px] font-mono">{pattern.subtitle}</span>
        </div>
        <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full border ${dirBg} ${dirColor} uppercase`}>
          {pattern.direction}
        </span>
      </div>
      <div ref={containerRef} className="w-full h-[340px]" style={{ background: '#0a0a0f' }} />
    </div>
  )
}

// ── Main Page ─────────────────────────────────────────────────────────────────
export default function EWMockPage() {
  return (
    <div className="min-h-screen bg-[#0a0a0f] p-6 space-y-6">
      <div className="max-w-7xl mx-auto">

        {/* Header */}
        <div className="mb-6">
          <div className="flex items-center gap-3 mb-1">
            <h1 className="text-2xl font-bold text-white font-mono">Elliott Wave Pattern Mock</h1>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded-full border border-indigo-500/40 bg-indigo-500/10 text-indigo-400">EW INTERNATIONAL SPEC</span>
          </div>
          <p className="text-gray-500 text-sm">Visual reference for all major EW corrective and motive structures. Based on EW International PDFs. Synthetic data with klinecharts overlay API.</p>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-x-6 gap-y-1 mt-4 text-[11px] font-mono">
            <span className="text-emerald-400">● Green = motive / impulse waves</span>
            <span className="text-amber-400">● Amber = corrective legs (ABC)</span>
            <span className="text-indigo-400">● Indigo = diagonal / X-wave</span>
            <span className="text-teal-400">● Teal = extended sub-wave</span>
            <span className="text-slate-400">- - Slate dashed = boundary trendlines</span>
            <span className="text-green-400/70">▒ Green zone = take profit / thrust</span>
            <span className="text-red-400/70">▒ Red zone = stop / invalidation</span>
            <span className="text-purple-300">△ Cone = probability range</span>
          </div>
        </div>

        {/* Pattern grid — 2 columns */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {EW_PATTERNS.map(p => (
            <EWPanel key={p.name} pattern={p} />
          ))}
        </div>

        {/* Rules reference */}
        <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="p-4 bg-gray-900/60 border border-gray-800 rounded-xl text-xs font-mono space-y-1">
            <p className="text-white font-bold mb-2">Triangle Rules (EWI)</p>
            <p className="text-gray-400">• All 5 legs corrective (3-3-3-3-3)</p>
            <p className="text-gray-400">• Never appears as W2 or wave A</p>
            <p className="text-gray-400">• Contracting: each leg smaller than prior same-set</p>
            <p className="text-gray-400">• Barrier: one trendline flat, other converges</p>
            <p className="text-gray-400">• Expanding: each leg larger, trendlines diverge</p>
            <p className="text-gray-400">• Thrust = E × 0.618–1.618</p>
          </div>
          <div className="p-4 bg-gray-900/60 border border-gray-800 rounded-xl text-xs font-mono space-y-1">
            <p className="text-white font-bold mb-2">Double Zigzag Rules (EWI)</p>
            <p className="text-gray-400">• W-X-Y: two zigzags connected by X</p>
            <p className="text-gray-400">• X wave rarely exceeds 70% of W</p>
            <p className="text-gray-400">• Y typically equals W in size</p>
            <p className="text-gray-400">• Common in W2 corrections of strong trends</p>
            <p className="text-gray-400">• X labeled distinctly — acts as separator</p>
            <p className="text-gray-400">• W2 total retrace: 50–78.6% of W1</p>
          </div>
          <div className="p-4 bg-gray-900/60 border border-gray-800 rounded-xl text-xs font-mono space-y-1">
            <p className="text-white font-bold mb-2">Diagonal + Throw-Over (EWI)</p>
            <p className="text-gray-400">• Ending diagonal: always in W5 or C position</p>
            <p className="text-gray-400">• W4 MUST overlap W1 price territory</p>
            <p className="text-gray-400">• All sub-waves corrective (3-3-3-3-3)</p>
            <p className="text-gray-400">• Throw-over: W5 briefly exceeds channel → snap reversal</p>
            <p className="text-gray-400">• Post-diagonal target: back to W2 or W4 start</p>
            <p className="text-gray-400">• W3 extension: Fib 2.618×W1 most common in stocks</p>
          </div>
        </div>

      </div>
    </div>
  )
}
