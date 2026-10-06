// PatternMockPage — Visual mock of how patterns would appear on klinecharts overlay
// Shows 6 pattern types drawn on synthetic data so user can approve the visual approach
import { useEffect, useRef, useState } from 'react'
import { init, dispose, type Chart } from 'klinecharts'
import { registerPatternOverlays } from '../components/chart/patternOverlays'

// ── Synthetic OHLCV generator ────────────────────────────────────────────
function generateData(scenario: string): any[] {
  const bars: any[] = []
  const baseTime = new Date('2025-06-01').getTime()
  const day = 86400000

  const push = (i: number, o: number, h: number, l: number, c: number, v = 2000000 + Math.random() * 3000000) => {
    bars.push({ timestamp: baseTime + i * day, open: o, high: h, low: l, close: c, volume: v })
  }

  if (scenario === 'head_shoulders') {
    const prices = [
      100,102,104,103,105,107,106,108,110,  // uptrend
      112,114,113,111,109,                    // left shoulder up
      108,107,109,108,110,                    // pullback
      112,115,117,119,120,118,116,113,111,   // head
      109,108,110,109,111,                    // pullback to neckline
      113,115,114,112,110,                    // right shoulder
      108,107,105,103,101,99,97,95,          // breakdown
      94,93,92,93,91,90,89,88,87,86,85,84   // future (continued decline)
    ]
    prices.forEach((p, i) => {
      const noise = (Math.random() - 0.5) * 1.5
      push(i, p + noise, p + Math.abs(noise) + 1.2, p - Math.abs(noise) - 0.8, p, 2000000 + (i > 30 ? 4000000 : 0))
    })
  }

  if (scenario === 'double_bottom') {
    const prices = [
      120,118,116,114,112,110,108,           // downtrend
      106,104,103,102,101,100,101,102,104,   // first bottom
      106,108,110,112,111,110,               // rally to neckline
      108,106,104,103,101,100,101,103,       // second bottom
      105,107,109,111,113,115,117,119,121,   // breakout
      122,124,123,125,126,128,127,129,130,131,132  // future (rally continues)
    ]
    prices.forEach((p, i) => {
      const noise = (Math.random() - 0.5) * 1.2
      push(i, p + noise, p + Math.abs(noise) + 0.8, p - Math.abs(noise) - 0.6, p)
    })
  }

  if (scenario === 'ascending_triangle') {
    const prices = [
      100,102,104,106,108,110,112,114,115,   // initial rise
      113,111,110,112,114,115,               // test 1
      114,112,111,113,114,115,               // test 2 (higher low)
      114,113,112.5,113.5,114.5,115,         // test 3 (higher low)
      114.5,113.5,113,114,115,               // test 4
      116,117,118,120,122,                    // breakout
      123,124,125,124,126,127,128,129,130    // future (rally)
    ]
    prices.forEach((p, i) => {
      const noise = (Math.random() - 0.5) * 0.8
      push(i, p + noise, p + Math.abs(noise) + 0.5, p - Math.abs(noise) - 0.4, p, i > 30 ? 5000000 : 2000000)
    })
  }

  if (scenario === 'bull_flag') {
    const prices = [
      80,82,85,88,92,96,100,105,110,115,118,120,  // flagpole
      119,118,117.5,117,116.5,116,115.5,115,114.5,114,113.5,  // flag
      114,115,117,119,121,123,126,130,             // breakout
      132,134,136,138,140,142,145,148,150,152      // future (measured move)
    ]
    prices.forEach((p, i) => {
      const noise = (Math.random() - 0.5) * 1.0
      push(i, p + noise, p + Math.abs(noise) + 0.8, p - Math.abs(noise) - 0.6, p, i < 12 ? 5000000 : i < 23 ? 1500000 : 4000000)
    })
  }

  if (scenario === 'rising_wedge') {
    const prices = [
      100,102,101,104,103,106,104.5,107,105.5,108,106.5,109,107.5,
      110,108,110.5,108.5,111,109,111.5,109.5,111.8,110,112,110.5,
      108,106,104,102,100,98,                 // breakdown
      96,94,93,91,90,88,87,86,85              // future (decline)
    ]
    prices.forEach((p, i) => {
      const noise = (Math.random() - 0.5) * 0.6
      push(i, p + noise, p + Math.abs(noise) + 0.5, p - Math.abs(noise) - 0.4, p)
    })
  }

  if (scenario === 'cup_handle') {
    const prices = [
      115,114,112,110,108,106,104,103,102,101,100,100,100,101,102,  // cup left + bottom
      103,104,106,108,110,112,114,115,                                // cup right
      114,113.5,113,112.5,112,112.5,113,                              // handle
      114,115,116,118,120,122,124,                                    // breakout
      125,127,128,130,131,132,133,134,135                             // future (rally)
    ]
    prices.forEach((p, i) => {
      const noise = (Math.random() - 0.5) * 0.8
      push(i, p + noise, p + Math.abs(noise) + 0.6, p - Math.abs(noise) - 0.5, p)
    })
  }

  return bars
}

// Pattern overlays are now registered via the shared module (imported above)

// ── Chart style (matches main overlay) ───────────────────────────────────
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
    priceMark: { show: true,
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

// ── Pattern drawing functions ────────────────────────────────────────────

interface PatternDef {
  name: string
  scenario: string
  direction: 'bullish' | 'bearish' | 'neutral'
  draw: (chart: Chart, data: any[]) => void
}

const BEARISH_LINE = '#ef4444'
const BULLISH_LINE = '#22c55e'
const NEUTRAL_LINE = '#f59e0b'
const BEARISH_FILL = 'rgba(239,68,68,0.10)'
const BULLISH_FILL = 'rgba(34,197,94,0.10)'
const NEUTRAL_FILL = 'rgba(245,158,11,0.10)'
const TARGET_COLOR = '#22c55e'
const STOP_COLOR = '#ef4444'
const NECKLINE_COLOR = '#60a5fa'
const PREDICT_COLOR = '#a78bfa'    // purple for prediction path
const PREDICT_BULL = '#4ade80'
const PREDICT_BEAR = '#f87171'

function ts(data: any[], i: number) { return data[Math.min(i, data.length - 1)]?.timestamp }

const PATTERNS: PatternDef[] = [
  {
    name: 'Head & Shoulders',
    scenario: 'head_shoulders',
    direction: 'bearish',
    draw: (chart, data) => {
      // Key points: left shoulder peak ~idx 13, head peak ~idx 23, right shoulder ~idx 34
      // Neckline: lows ~idx 18 and ~idx 29
      const lsIdx = 13, headIdx = 23, rsIdx = 34, nl1Idx = 18, nl2Idx = 29
      const lsPrice = 114, headPrice = 120, rsPrice = 115
      const nl1Price = 108, nl2Price = 109
      const breakdownTarget = 97

      // Left shoulder → head → right shoulder lines
      chart.createOverlay({ name: 'patternTrendline', points: [
        { timestamp: ts(data, 9), value: 110 }, { timestamp: ts(data, lsIdx), value: lsPrice }
      ], extendData: { color: BEARISH_LINE, lineWidth: 2 } })
      chart.createOverlay({ name: 'patternTrendline', points: [
        { timestamp: ts(data, lsIdx), value: lsPrice }, { timestamp: ts(data, nl1Idx), value: nl1Price }
      ], extendData: { color: BEARISH_LINE, lineWidth: 2 } })
      chart.createOverlay({ name: 'patternTrendline', points: [
        { timestamp: ts(data, nl1Idx), value: nl1Price }, { timestamp: ts(data, headIdx), value: headPrice }
      ], extendData: { color: BEARISH_LINE, lineWidth: 2 } })
      chart.createOverlay({ name: 'patternTrendline', points: [
        { timestamp: ts(data, headIdx), value: headPrice }, { timestamp: ts(data, nl2Idx), value: nl2Price }
      ], extendData: { color: BEARISH_LINE, lineWidth: 2 } })
      chart.createOverlay({ name: 'patternTrendline', points: [
        { timestamp: ts(data, nl2Idx), value: nl2Price }, { timestamp: ts(data, rsIdx), value: rsPrice }
      ], extendData: { color: BEARISH_LINE, lineWidth: 2 } })
      chart.createOverlay({ name: 'patternTrendline', points: [
        { timestamp: ts(data, rsIdx), value: rsPrice }, { timestamp: ts(data, 38), value: 105 }
      ], extendData: { color: BEARISH_LINE, lineWidth: 2 } })

      // Neckline (dashed blue, extended)
      chart.createOverlay({ name: 'patternTrendline', points: [
        { timestamp: ts(data, nl1Idx - 3), value: nl1Price - 0.3 },
        { timestamp: ts(data, 40), value: nl2Price + 1 }
      ], extendData: { color: NECKLINE_COLOR, lineWidth: 1.5, dash: [6, 4] } })

      // Labels
      chart.createOverlay({ name: 'patternLabel', points: [{ timestamp: ts(data, lsIdx), value: lsPrice }], extendData: { text: 'LS', color: '#f87171' } })
      chart.createOverlay({ name: 'patternLabel', points: [{ timestamp: ts(data, headIdx), value: headPrice }], extendData: { text: 'HEAD', color: '#f87171' } })
      chart.createOverlay({ name: 'patternLabel', points: [{ timestamp: ts(data, rsIdx), value: rsPrice }], extendData: { text: 'RS', color: '#f87171' } })

      // Target + Neckline labels
      chart.createOverlay({ name: 'patternLevel', points: [{ timestamp: ts(data, 0), value: breakdownTarget }], extendData: { color: TARGET_COLOR, label: `TARGET $${breakdownTarget}` } })
      chart.createOverlay({ name: 'patternLevel', points: [{ timestamp: ts(data, 0), value: nl1Price }], extendData: { color: NECKLINE_COLOR, label: 'NECKLINE' } })

      // Badge
      chart.createOverlay({ name: 'patternBadge', points: [{ timestamp: ts(data, 2), value: headPrice + 2 }], extendData: { text: 'HEAD & SHOULDERS  85%', bgColor: 'rgba(239,68,68,0.2)', textColor: '#f87171' } })

      // ── PREDICTION ──
      // Projected path (purple dotted → expected decline)
      chart.createOverlay({ name: 'predictionPath', points: [
        { timestamp: ts(data, 38), value: 105 },
        { timestamp: ts(data, 41), value: 100 },
        { timestamp: ts(data, 44), value: 96 },
        { timestamp: ts(data, 47), value: 92 },
        { timestamp: ts(data, 50), value: 88 },
      ], extendData: { color: PREDICT_BEAR, lineWidth: 2, numPoints: 5 } })

      // Probability cone (expanding uncertainty)
      chart.createOverlay({ name: 'predictionCone', points: [
        { timestamp: ts(data, 38), value: 105 },          // origin
        { timestamp: ts(data, 50), value: 82 },           // lower bound
        { timestamp: ts(data, 50), value: 96 },           // upper bound
      ], extendData: { color: 'rgba(248,113,113,0.06)', borderColor: 'rgba(248,113,113,0.2)' } })

      // Target zone (green shaded area)
      chart.createOverlay({ name: 'predictionZone', points: [
        { timestamp: ts(data, 42), value: 99 },
        { timestamp: ts(data, 52), value: 93 },
      ], extendData: { color: 'rgba(34,197,94,0.08)', borderColor: 'rgba(34,197,94,0.25)', label: 'TAKE PROFIT', labelColor: '#4ade80' } })

      // Stop zone (red shaded area — above neckline = invalidated)
      chart.createOverlay({ name: 'predictionZone', points: [
        { timestamp: ts(data, 38), value: nl2Price + 1 },
        { timestamp: ts(data, 52), value: rsPrice + 1 },
      ], extendData: { color: 'rgba(239,68,68,0.06)', borderColor: 'rgba(239,68,68,0.2)', label: 'STOP / INVALID', labelColor: '#f87171' } })
    },
  },
  {
    name: 'Double Bottom',
    scenario: 'double_bottom',
    direction: 'bullish',
    draw: (chart, data) => {
      const b1Idx = 13, rallyIdx = 20, b2Idx = 28
      const bottomPrice = 100, necklinePrice = 112, targetPrice = 124

      // V-shape lines connecting bottoms
      chart.createOverlay({ name: 'patternTrendline', points: [
        { timestamp: ts(data, 7), value: 108 }, { timestamp: ts(data, b1Idx), value: bottomPrice }
      ], extendData: { color: BULLISH_LINE, lineWidth: 2 } })
      chart.createOverlay({ name: 'patternTrendline', points: [
        { timestamp: ts(data, b1Idx), value: bottomPrice }, { timestamp: ts(data, rallyIdx), value: necklinePrice }
      ], extendData: { color: BULLISH_LINE, lineWidth: 2 } })
      chart.createOverlay({ name: 'patternTrendline', points: [
        { timestamp: ts(data, rallyIdx), value: necklinePrice }, { timestamp: ts(data, b2Idx), value: bottomPrice }
      ], extendData: { color: BULLISH_LINE, lineWidth: 2 } })
      chart.createOverlay({ name: 'patternTrendline', points: [
        { timestamp: ts(data, b2Idx), value: bottomPrice }, { timestamp: ts(data, 34), value: 115 }
      ], extendData: { color: BULLISH_LINE, lineWidth: 2 } })

      // Neckline
      chart.createOverlay({ name: 'patternTrendline', points: [
        { timestamp: ts(data, 5), value: necklinePrice },
        { timestamp: ts(data, 38), value: necklinePrice }
      ], extendData: { color: NECKLINE_COLOR, lineWidth: 1.5, dash: [6, 4] } })

      // Support line at bottom
      chart.createOverlay({ name: 'patternTrendline', points: [
        { timestamp: ts(data, b1Idx - 2), value: bottomPrice },
        { timestamp: ts(data, b2Idx + 2), value: bottomPrice }
      ], extendData: { color: STOP_COLOR, lineWidth: 1, dash: [4, 3] } })

      // Labels
      chart.createOverlay({ name: 'patternLabel', points: [{ timestamp: ts(data, b1Idx), value: bottomPrice }], extendData: { text: 'B1', color: '#4ade80', offsetY: 16 } })
      chart.createOverlay({ name: 'patternLabel', points: [{ timestamp: ts(data, b2Idx), value: bottomPrice }], extendData: { text: 'B2', color: '#4ade80', offsetY: 16 } })

      // Levels
      chart.createOverlay({ name: 'patternLevel', points: [{ timestamp: ts(data, 0), value: targetPrice }], extendData: { color: TARGET_COLOR, label: `TARGET $${targetPrice}` } })
      chart.createOverlay({ name: 'patternLevel', points: [{ timestamp: ts(data, 0), value: necklinePrice }], extendData: { color: NECKLINE_COLOR, label: 'NECKLINE' } })

      // Badge
      chart.createOverlay({ name: 'patternBadge', points: [{ timestamp: ts(data, 2), value: 122 }], extendData: { text: 'DOUBLE BOTTOM  82%', bgColor: 'rgba(34,197,94,0.2)', textColor: '#4ade80' } })

      // ── PREDICTION ──
      chart.createOverlay({ name: 'predictionPath', points: [
        { timestamp: ts(data, 34), value: 115 },
        { timestamp: ts(data, 37), value: 119 },
        { timestamp: ts(data, 40), value: 122 },
        { timestamp: ts(data, 44), value: 126 },
        { timestamp: ts(data, 48), value: 130 },
      ], extendData: { color: PREDICT_BULL, lineWidth: 2, numPoints: 5 } })
      chart.createOverlay({ name: 'predictionCone', points: [
        { timestamp: ts(data, 34), value: 115 },
        { timestamp: ts(data, 48), value: 135 },
        { timestamp: ts(data, 48), value: 122 },
      ], extendData: { color: 'rgba(74,222,128,0.06)', borderColor: 'rgba(74,222,128,0.2)' } })
      chart.createOverlay({ name: 'predictionZone', points: [
        { timestamp: ts(data, 40), value: targetPrice },
        { timestamp: ts(data, 50), value: targetPrice + 6 },
      ], extendData: { color: 'rgba(34,197,94,0.08)', borderColor: 'rgba(34,197,94,0.25)', label: 'TAKE PROFIT', labelColor: '#4ade80' } })
      chart.createOverlay({ name: 'predictionZone', points: [
        { timestamp: ts(data, 34), value: 98 },
        { timestamp: ts(data, 50), value: 100 },
      ], extendData: { color: 'rgba(239,68,68,0.06)', borderColor: 'rgba(239,68,68,0.2)', label: 'STOP / INVALID', labelColor: '#f87171' } })
    },
  },
  {
    name: 'Ascending Triangle',
    scenario: 'ascending_triangle',
    direction: 'bullish',
    draw: (chart, data) => {
      const resistPrice = 115, targetPrice = 122
      // Flat resistance line
      chart.createOverlay({ name: 'patternTrendline', points: [
        { timestamp: ts(data, 8), value: resistPrice },
        { timestamp: ts(data, 33), value: resistPrice }
      ], extendData: { color: BEARISH_LINE, lineWidth: 2 } })

      // Rising support trendline
      chart.createOverlay({ name: 'patternTrendline', points: [
        { timestamp: ts(data, 10), value: 110 },
        { timestamp: ts(data, 30), value: 113 }
      ], extendData: { color: BULLISH_LINE, lineWidth: 2 } })

      // Fill zone (triangle body)
      chart.createOverlay({ name: 'patternFillZone', points: [
        { timestamp: ts(data, 10), value: resistPrice },     // top-left
        { timestamp: ts(data, 30), value: resistPrice },     // top-right (apex)
        { timestamp: ts(data, 30), value: 113 },             // bottom-right
        { timestamp: ts(data, 10), value: 110 },             // bottom-left
      ], extendData: { color: BULLISH_FILL } })

      // Breakout arrow line
      chart.createOverlay({ name: 'patternTrendline', points: [
        { timestamp: ts(data, 33), value: resistPrice },
        { timestamp: ts(data, 37), value: targetPrice }
      ], extendData: { color: BULLISH_LINE, lineWidth: 2, dash: [4, 3] } })

      // Labels
      chart.createOverlay({ name: 'patternLabel', points: [{ timestamp: ts(data, 20), value: resistPrice }], extendData: { text: 'RESISTANCE', color: '#f87171', offsetY: -14 } })
      chart.createOverlay({ name: 'patternLabel', points: [{ timestamp: ts(data, 20), value: 111.5 }], extendData: { text: 'RISING SUPPORT', color: '#4ade80', offsetY: 16 } })

      // Target level
      chart.createOverlay({ name: 'patternLevel', points: [{ timestamp: ts(data, 0), value: targetPrice }], extendData: { color: TARGET_COLOR, label: `TARGET $${targetPrice}` } })

      // Badge
      chart.createOverlay({ name: 'patternBadge', points: [{ timestamp: ts(data, 2), value: 123 }], extendData: { text: 'ASC TRIANGLE  78%', bgColor: 'rgba(34,197,94,0.2)', textColor: '#4ade80' } })

      // ── PREDICTION ──
      chart.createOverlay({ name: 'predictionPath', points: [
        { timestamp: ts(data, 35), value: 116 },
        { timestamp: ts(data, 38), value: 119 },
        { timestamp: ts(data, 41), value: 122 },
        { timestamp: ts(data, 44), value: 126 },
        { timestamp: ts(data, 47), value: 129 },
      ], extendData: { color: PREDICT_BULL, lineWidth: 2, numPoints: 5 } })
      chart.createOverlay({ name: 'predictionCone', points: [
        { timestamp: ts(data, 35), value: 116 },
        { timestamp: ts(data, 47), value: 133 },
        { timestamp: ts(data, 47), value: 120 },
      ], extendData: { color: 'rgba(74,222,128,0.06)', borderColor: 'rgba(74,222,128,0.2)' } })
      chart.createOverlay({ name: 'predictionZone', points: [
        { timestamp: ts(data, 40), value: targetPrice },
        { timestamp: ts(data, 50), value: targetPrice + 5 },
      ], extendData: { color: 'rgba(34,197,94,0.08)', borderColor: 'rgba(34,197,94,0.25)', label: 'TAKE PROFIT', labelColor: '#4ade80' } })
      chart.createOverlay({ name: 'predictionZone', points: [
        { timestamp: ts(data, 35), value: 112 },
        { timestamp: ts(data, 50), value: 113.5 },
      ], extendData: { color: 'rgba(239,68,68,0.06)', borderColor: 'rgba(239,68,68,0.2)', label: 'STOP', labelColor: '#f87171' } })
    },
  },
  {
    name: 'Bull Flag',
    scenario: 'bull_flag',
    direction: 'bullish',
    draw: (chart, data) => {
      // Flagpole
      chart.createOverlay({ name: 'patternTrendline', points: [
        { timestamp: ts(data, 0), value: 80 },
        { timestamp: ts(data, 11), value: 120 }
      ], extendData: { color: BULLISH_LINE, lineWidth: 2.5 } })

      // Flag channel — upper and lower parallel lines (slight downslope)
      chart.createOverlay({ name: 'patternTrendline', points: [
        { timestamp: ts(data, 11), value: 120 },
        { timestamp: ts(data, 22), value: 114.5 }
      ], extendData: { color: NEUTRAL_LINE, lineWidth: 1.5 } })
      chart.createOverlay({ name: 'patternTrendline', points: [
        { timestamp: ts(data, 12), value: 118 },
        { timestamp: ts(data, 22), value: 113 }
      ], extendData: { color: NEUTRAL_LINE, lineWidth: 1.5 } })

      // Fill zone (flag body)
      chart.createOverlay({ name: 'patternFillZone', points: [
        { timestamp: ts(data, 11), value: 120 },
        { timestamp: ts(data, 22), value: 114.5 },
        { timestamp: ts(data, 22), value: 113 },
        { timestamp: ts(data, 12), value: 118 },
      ], extendData: { color: NEUTRAL_FILL } })

      // Breakout projection
      chart.createOverlay({ name: 'patternTrendline', points: [
        { timestamp: ts(data, 22), value: 114 },
        { timestamp: ts(data, 30), value: 130 }
      ], extendData: { color: BULLISH_LINE, lineWidth: 1.5, dash: [4, 3] } })

      // Labels
      chart.createOverlay({ name: 'patternLabel', points: [{ timestamp: ts(data, 5), value: 95 }], extendData: { text: 'POLE', color: '#4ade80', offsetY: 16 } })
      chart.createOverlay({ name: 'patternLabel', points: [{ timestamp: ts(data, 17), value: 116 }], extendData: { text: 'FLAG', color: '#f59e0b', offsetY: -14 } })

      // Target
      chart.createOverlay({ name: 'patternLevel', points: [{ timestamp: ts(data, 0), value: 130 }], extendData: { color: TARGET_COLOR, label: 'TARGET $130 (pole height)' } })

      // Badge
      chart.createOverlay({ name: 'patternBadge', points: [{ timestamp: ts(data, 2), value: 155 }], extendData: { text: 'BULL FLAG  88%', bgColor: 'rgba(34,197,94,0.2)', textColor: '#4ade80' } })

      // ── PREDICTION (measured move: pole height = 40, added to breakout ~114) ──
      chart.createOverlay({ name: 'predictionPath', points: [
        { timestamp: ts(data, 23), value: 115 },
        { timestamp: ts(data, 26), value: 123 },
        { timestamp: ts(data, 30), value: 134 },
        { timestamp: ts(data, 34), value: 142 },
        { timestamp: ts(data, 38), value: 152 },
      ], extendData: { color: PREDICT_BULL, lineWidth: 2, numPoints: 5 } })
      chart.createOverlay({ name: 'predictionCone', points: [
        { timestamp: ts(data, 23), value: 115 },
        { timestamp: ts(data, 38), value: 158 },
        { timestamp: ts(data, 38), value: 138 },
      ], extendData: { color: 'rgba(74,222,128,0.06)', borderColor: 'rgba(74,222,128,0.2)' } })
      chart.createOverlay({ name: 'predictionZone', points: [
        { timestamp: ts(data, 30), value: 148 },
        { timestamp: ts(data, 40), value: 156 },
      ], extendData: { color: 'rgba(34,197,94,0.08)', borderColor: 'rgba(34,197,94,0.25)', label: 'MEASURED MOVE TARGET', labelColor: '#4ade80' } })
      chart.createOverlay({ name: 'predictionZone', points: [
        { timestamp: ts(data, 23), value: 111 },
        { timestamp: ts(data, 40), value: 113 },
      ], extendData: { color: 'rgba(239,68,68,0.06)', borderColor: 'rgba(239,68,68,0.2)', label: 'STOP', labelColor: '#f87171' } })
    },
  },
  {
    name: 'Rising Wedge',
    scenario: 'rising_wedge',
    direction: 'bearish',
    draw: (chart, data) => {
      // Upper trendline (resistance, less steep)
      chart.createOverlay({ name: 'patternTrendline', points: [
        { timestamp: ts(data, 1), value: 102 },
        { timestamp: ts(data, 23), value: 112 }
      ], extendData: { color: BEARISH_LINE, lineWidth: 2 } })

      // Lower trendline (support, steeper)
      chart.createOverlay({ name: 'patternTrendline', points: [
        { timestamp: ts(data, 2), value: 101 },
        { timestamp: ts(data, 23), value: 110.5 }
      ], extendData: { color: BEARISH_LINE, lineWidth: 2 } })

      // Fill zone
      chart.createOverlay({ name: 'patternFillZone', points: [
        { timestamp: ts(data, 1), value: 102 },
        { timestamp: ts(data, 23), value: 112 },
        { timestamp: ts(data, 23), value: 110.5 },
        { timestamp: ts(data, 2), value: 101 },
      ], extendData: { color: BEARISH_FILL } })

      // Breakdown projection
      chart.createOverlay({ name: 'patternTrendline', points: [
        { timestamp: ts(data, 24), value: 108 },
        { timestamp: ts(data, 29), value: 98 }
      ], extendData: { color: BEARISH_LINE, lineWidth: 1.5, dash: [4, 3] } })

      // Labels
      chart.createOverlay({ name: 'patternLabel', points: [{ timestamp: ts(data, 12), value: 107.5 }], extendData: { text: 'CONVERGING', color: '#f87171', offsetY: -14 } })

      // Target
      chart.createOverlay({ name: 'patternLevel', points: [{ timestamp: ts(data, 0), value: 98 }], extendData: { color: TARGET_COLOR, label: 'TARGET $98' } })

      // Badge
      chart.createOverlay({ name: 'patternBadge', points: [{ timestamp: ts(data, 2), value: 114 }], extendData: { text: 'RISING WEDGE  76%', bgColor: 'rgba(239,68,68,0.2)', textColor: '#f87171' } })

      // ── PREDICTION ──
      chart.createOverlay({ name: 'predictionPath', points: [
        { timestamp: ts(data, 25), value: 108 },
        { timestamp: ts(data, 28), value: 102 },
        { timestamp: ts(data, 31), value: 96 },
        { timestamp: ts(data, 34), value: 91 },
        { timestamp: ts(data, 37), value: 86 },
      ], extendData: { color: PREDICT_BEAR, lineWidth: 2, numPoints: 5 } })
      chart.createOverlay({ name: 'predictionCone', points: [
        { timestamp: ts(data, 25), value: 108 },
        { timestamp: ts(data, 37), value: 80 },
        { timestamp: ts(data, 37), value: 95 },
      ], extendData: { color: 'rgba(248,113,113,0.06)', borderColor: 'rgba(248,113,113,0.2)' } })
      chart.createOverlay({ name: 'predictionZone', points: [
        { timestamp: ts(data, 30), value: 85 },
        { timestamp: ts(data, 39), value: 90 },
      ], extendData: { color: 'rgba(34,197,94,0.08)', borderColor: 'rgba(34,197,94,0.25)', label: 'TAKE PROFIT', labelColor: '#4ade80' } })
      chart.createOverlay({ name: 'predictionZone', points: [
        { timestamp: ts(data, 25), value: 111 },
        { timestamp: ts(data, 39), value: 113 },
      ], extendData: { color: 'rgba(239,68,68,0.06)', borderColor: 'rgba(239,68,68,0.2)', label: 'STOP / INVALID', labelColor: '#f87171' } })
    },
  },
  {
    name: 'Cup & Handle',
    scenario: 'cup_handle',
    direction: 'bullish',
    draw: (chart, data) => {
      // Cup left rim → bottom → right rim (curved via segments)
      const cupPoints = [
        [0, 115], [2, 112], [4, 108], [6, 104], [8, 102], [10, 100],
        [12, 100], [14, 101], [16, 103], [18, 106], [20, 110], [22, 114],
      ]
      for (let i = 0; i < cupPoints.length - 1; i++) {
        chart.createOverlay({ name: 'patternTrendline', points: [
          { timestamp: ts(data, cupPoints[i][0]), value: cupPoints[i][1] },
          { timestamp: ts(data, cupPoints[i+1][0]), value: cupPoints[i+1][1] },
        ], extendData: { color: BULLISH_LINE, lineWidth: 2 } })
      }

      // Handle
      chart.createOverlay({ name: 'patternTrendline', points: [
        { timestamp: ts(data, 22), value: 114 },
        { timestamp: ts(data, 26), value: 112 }
      ], extendData: { color: NEUTRAL_LINE, lineWidth: 1.5 } })
      chart.createOverlay({ name: 'patternTrendline', points: [
        { timestamp: ts(data, 26), value: 112 },
        { timestamp: ts(data, 29), value: 114 }
      ], extendData: { color: NEUTRAL_LINE, lineWidth: 1.5 } })

      // Rim line (resistance)
      chart.createOverlay({ name: 'patternTrendline', points: [
        { timestamp: ts(data, 0), value: 115 },
        { timestamp: ts(data, 32), value: 115 }
      ], extendData: { color: NECKLINE_COLOR, lineWidth: 1.5, dash: [6, 4] } })

      // Labels
      chart.createOverlay({ name: 'patternLabel', points: [{ timestamp: ts(data, 10), value: 100 }], extendData: { text: 'CUP', color: '#4ade80', offsetY: 16 } })
      chart.createOverlay({ name: 'patternLabel', points: [{ timestamp: ts(data, 26), value: 112 }], extendData: { text: 'HANDLE', color: '#f59e0b', offsetY: 16 } })

      // Target (cup depth projected above rim)
      chart.createOverlay({ name: 'patternLevel', points: [{ timestamp: ts(data, 0), value: 130 }], extendData: { color: TARGET_COLOR, label: 'TARGET $130' } })
      chart.createOverlay({ name: 'patternLevel', points: [{ timestamp: ts(data, 0), value: 115 }], extendData: { color: NECKLINE_COLOR, label: 'RIM / BREAKOUT' } })

      // Badge
      chart.createOverlay({ name: 'patternBadge', points: [{ timestamp: ts(data, 2), value: 138 }], extendData: { text: 'CUP & HANDLE  80%', bgColor: 'rgba(34,197,94,0.2)', textColor: '#4ade80' } })

      // ── PREDICTION (measured move: cup depth 15 added above rim 115 = target 130) ──
      chart.createOverlay({ name: 'predictionPath', points: [
        { timestamp: ts(data, 30), value: 115 },
        { timestamp: ts(data, 33), value: 120 },
        { timestamp: ts(data, 36), value: 125 },
        { timestamp: ts(data, 39), value: 129 },
        { timestamp: ts(data, 42), value: 133 },
      ], extendData: { color: PREDICT_BULL, lineWidth: 2, numPoints: 5 } })
      chart.createOverlay({ name: 'predictionCone', points: [
        { timestamp: ts(data, 30), value: 115 },
        { timestamp: ts(data, 42), value: 138 },
        { timestamp: ts(data, 42), value: 122 },
      ], extendData: { color: 'rgba(74,222,128,0.06)', borderColor: 'rgba(74,222,128,0.2)' } })
      chart.createOverlay({ name: 'predictionZone', points: [
        { timestamp: ts(data, 35), value: 128 },
        { timestamp: ts(data, 45), value: 135 },
      ], extendData: { color: 'rgba(34,197,94,0.08)', borderColor: 'rgba(34,197,94,0.25)', label: 'TAKE PROFIT', labelColor: '#4ade80' } })
      chart.createOverlay({ name: 'predictionZone', points: [
        { timestamp: ts(data, 30), value: 110 },
        { timestamp: ts(data, 45), value: 112 },
      ], extendData: { color: 'rgba(239,68,68,0.06)', borderColor: 'rgba(239,68,68,0.2)', label: 'STOP', labelColor: '#f87171' } })
    },
  },
]

// ── Chart Panel Component ────────────────────────────────────────────────

function PatternPanel({ pattern }: { pattern: PatternDef }) {
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

    // Generate data and load
    const data = generateData(pattern.scenario)

    chart.setDataLoader({
      getBars: async (params: any) => {
        params.callback(data, false)
      },
    } as any)
    ;(chart as any).setSymbol({ ticker: 'MOCK', pricePrecision: 2, volumePrecision: 0 })
    ;(chart as any).setPeriod({ type: 'day', span: 1 })

    // Draw pattern overlays after a short delay (let data render first)
    setTimeout(() => {
      if (chartRef.current) {
        pattern.draw(chartRef.current, data)
      }
    }, 300)

    return () => {
      if (chartRef.current) {
        dispose(chartRef.current)
        chartRef.current = null
      }
    }
  }, [pattern])

  const dirColor = pattern.direction === 'bullish' ? 'text-emerald-400' : pattern.direction === 'bearish' ? 'text-red-400' : 'text-amber-400'
  const dirBg = pattern.direction === 'bullish' ? 'bg-emerald-500/15 border-emerald-500/30' : pattern.direction === 'bearish' ? 'bg-red-500/15 border-red-500/30' : 'bg-amber-500/15 border-amber-500/30'

  return (
    <div className="flex flex-col rounded-xl border border-gray-800 overflow-hidden bg-gray-950">
      <div className="flex items-center justify-between px-3 py-2 bg-gray-900/80 border-b border-gray-800">
        <span className="text-white font-bold text-sm font-mono">{pattern.name}</span>
        <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full border ${dirBg} ${dirColor} uppercase`}>
          {pattern.direction}
        </span>
      </div>
      <div ref={containerRef} className="w-full h-[320px]" style={{ background: '#0a0a0f' }} />
    </div>
  )
}

// ── Main Page ────────────────────────────────────────────────────────────

export default function PatternMockPage() {
  return (
    <div className="min-h-screen bg-[#0a0a0f] p-6 space-y-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-6">
          <h1 className="text-2xl font-bold text-white font-mono mb-1">Pattern Drawing Mock</h1>
          <p className="text-gray-500 text-sm">Preview of how detected patterns would appear on the klinecharts overlay. Each chart shows synthetic data with pattern overlays drawn using the klinecharts v10 overlay API.</p>
          <div className="flex flex-wrap gap-x-4 gap-y-1 mt-3 text-[11px] font-mono">
            <span className="text-emerald-400">● Green lines = bullish structure</span>
            <span className="text-red-400">● Red lines = bearish structure</span>
            <span className="text-blue-400">- - Dashed blue = neckline / key level</span>
            <span className="text-amber-400">■ Shaded = pattern body (fill zone)</span>
            <span className="text-gray-400">◦ Dots + labels = pivot points</span>
          </div>
          <div className="flex flex-wrap gap-x-4 gap-y-1 mt-1 text-[11px] font-mono">
            <span className="text-purple-400">⤳ Dotted arrow = predicted path</span>
            <span className="text-purple-300">△ Expanding cone = probability range</span>
            <span className="text-emerald-400/70">▒ Green zone = take profit area</span>
            <span className="text-red-400/70">▒ Red zone = stop / invalidation</span>
          </div>
        </div>

        {/* Pattern grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {PATTERNS.map(p => (
            <PatternPanel key={p.name} pattern={p} />
          ))}
        </div>

        {/* Legend */}
        <div className="mt-6 p-4 bg-gray-900/60 border border-gray-800 rounded-xl text-xs text-gray-500 font-mono space-y-1">
          <p>This mock uses klinecharts v10 <code>registerOverlay</code> + <code>createOverlay</code> API.</p>
          <p>5 custom overlay types registered: <span className="text-gray-400">patternTrendline, patternFillZone, patternLabel, patternLevel, patternBadge</span></p>
          <p>In production: coordinates come from <code>/api/patterns/&#123;symbol&#125;</code> backend (overlay_lines, fill_zone, points, point_labels)</p>
          <p>Backend already detects 80+ patterns and returns all drawing coordinates — this mock proves the rendering works.</p>
        </div>
      </div>
    </div>
  )
}
