// KLineChartOverlay — full-screen KLineChart for deep analysis from scan results
// Uses klinecharts v10 beta DataLoader pattern + pattern overlays
import { useEffect, useRef, useState, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { init, dispose, type Chart, type KLineData } from 'klinecharts'
import api from '../../api/client'
import { registerPatternOverlays } from './patternOverlays'

// Register overlay types for wave rendering (reuses patternTrendline, patternBadge)
registerPatternOverlays()

// ── Types ──────────────────────────────────────────────────────────────────

interface Props {
  symbol: string
  onClose: () => void
}

type TF = '1m' | '5m' | '15m' | '1h' | '4h' | '1D' | '1W' | '1M'

const TF_CONFIG: Record<TF, { endpoint: string; params: Record<string, any>; label: string }> = {
  '1m':  { endpoint: 'intraday', params: { resolution: 1, days_back: 7 }, label: '1m' },
  '5m':  { endpoint: 'intraday', params: { resolution: 5, days_back: 60 }, label: '5m' },
  '15m': { endpoint: 'intraday', params: { resolution: 15, days_back: 60 }, label: '15m' },
  '1h':  { endpoint: 'intraday', params: { resolution: 60, days_back: 730 }, label: '1H' },
  '4h':  { endpoint: 'intraday', params: { resolution: 240, days_back: 730 }, label: '4H' },
  '1D':  { endpoint: 'chart', params: { interval: 'daily', bars: 780 }, label: '1D' },
  '1W':  { endpoint: 'chart', params: { interval: 'weekly' }, label: '1W' },
  '1M':  { endpoint: 'chart', params: { interval: 'monthly' }, label: '1M' },
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function parseDateToTimestamp(dateStr: string): number {
  return new Date(dateStr + 'T00:00:00').getTime()
}

async function fetchData(symbol: string, tfKey: TF): Promise<KLineData[]> {
  const cfg = TF_CONFIG[tfKey]
  if (cfg.endpoint === 'intraday') {
    const r = await api.get(`/candles/${symbol}`, { params: { resolution: cfg.params.resolution, days_back: cfg.params.days_back } })
    const bars = r.data || []
    return bars.map((b: any) => ({
      timestamp: b.time * 1000,
      open: b.open,
      high: b.high,
      low: b.low,
      close: b.close,
      volume: b.volume || 0,
    }))
  } else {
    // Daily / Weekly / Monthly: same /chart endpoint for consistent depth
    // (limit=0 → all rows). Daily was previously routed through a "fast"
    // minichart endpoint that silently fell back to a 3-month yfinance
    // window when its DB call errored — leaving the chart with only ~60
    // bars of history.
    const r = await api.get(`/chart/${symbol}`, { params: { limit: 0, interval: cfg.params.interval } })
    const prices = r.data?.prices || []
    return prices
      .filter((p: any) => p.open != null && p.close != null)
      .map((p: any) => ({
        timestamp: parseDateToTimestamp(p.date),
        open: p.open,
        high: p.high,
        low: p.low,
        close: p.close,
        volume: p.volume || 0,
      }))
  }
}

// ── Live quote hook ──────────────────────────────────────────────────────────

function useCompanyName(symbol: string) {
  const [name, setName] = useState<string>('')
  useEffect(() => {
    let cancelled = false
    api.get('/instruments').then(r => {
      if (cancelled) return
      const list = r.data || []
      const match = list.find((i: any) => i.symbol?.toUpperCase() === symbol.toUpperCase())
      if (match?.name) setName(match.name)
    }).catch(() => {})
    return () => { cancelled = true }
  }, [symbol])
  return name
}

function useLivePrice(symbol: string) {
  const [price, setPrice] = useState<{ last: number; change: number; changePct: number; high: number; low: number; open: number } | null>(null)

  useEffect(() => {
    let cancelled = false
    const poll = async () => {
      try {
        const r = await api.get(`/quote/${symbol}`)
        if (!cancelled && r.data) {
          setPrice({
            last: r.data.price ?? 0,
            change: r.data.change ?? 0,
            changePct: r.data.change_pct ?? 0,
            high: r.data.high ?? 0,
            low: r.data.low ?? 0,
            open: r.data.open ?? 0,
          })
        }
      } catch {}
    }
    poll()
    const iv = setInterval(poll, 15000)
    return () => { cancelled = true; clearInterval(iv) }
  }, [symbol])

  return price
}

// ── Elliott Wave overlay drawing ─────────────────────────────────────────────
// Renders multi-degree wave counts: zigzag lines connecting pivots + wave labels.
// Color coding by degree: primary (gold), intermediate (cyan), minor (gray).

const DEGREE_COLORS: Record<string, { line: string; label: string; bg: string }> = {
  primary:      { line: '#fbbf24', label: '#fbbf24', bg: 'rgba(251,191,36,0.15)' },
  intermediate: { line: '#22d3ee', label: '#22d3ee', bg: 'rgba(34,211,238,0.12)' },
  minor:        { line: '#9ca3af', label: '#9ca3af', bg: 'rgba(156,163,175,0.10)' },
}

interface WavePoint {
  label: string
  start_idx: number
  start_price: number
  end_idx: number
  end_price: number
  direction: string
}

interface WaveProjection {
  label: string
  price: number
  fib: number
  from_idx: number
  type: string     // "target" | "retrace"
  wave: number
}

interface WaveCountData {
  type: string       // "impulse" | "corrective"
  degree: string     // "primary" | "intermediate" | "minor"
  direction: string  // "up" | "down"
  confidence: number
  fib_score: number
  is_complete: boolean
  waves: WavePoint[]
  labels: string[]
  projections?: WaveProjection[]
}

interface ElliottResponse {
  symbol: string
  bars: { timestamp: number; open: number; high: number; low: number; close: number; volume: number }[]
  wave_counts: WaveCountData[]
  current_position: {
    degree: string
    phase: string            // "impulse" | "correction" | "unknown"
    wave_number: number
    wave_in_progress: string
    current_label: string
    direction: string
    confidence: number
    action: string
    bias: string
    strength: string         // "strong" | "moderate" | "cautious" | "weak" | "none"
    trend: string            // "BULLISH" | "BEARISH" | "NEUTRAL"
    pattern: string          // "bullish_impulse" | "bearish_correction" etc.
    sub_phase: string        // "motive" | "corrective" | "unknown"
    next_move: string
    type: string
    waves_complete: number
    summary: string
  }
  best_count: WaveCountData | null
  best_impulse: WaveCountData | null
  best_corrective: WaveCountData | null
  channels?: EWChannel[]
}

interface EWChannelLine {
  start_idx: number
  start_price: number
  end_idx: number
  end_price: number
}

interface EWChannelBreakout {
  idx: number
  price: number
  new_phase: string  // "corrective" | "impulse"
  label: string
}

interface EWChannel {
  type: string       // "base" | "acceleration" | "corrective"
  direction: string  // "up" | "down"
  lower: EWChannelLine
  upper: EWChannelLine
  breakout: EWChannelBreakout | null
}

interface EWLayerOptions {
  labels: boolean   // wave number labels (1,2,3,4,5,A,B,C)
  targets: boolean  // projected wave path + fib target zones
  status: boolean   // status badges (phase, summary, action)
}

// ── Wave color themes ────────────────────────────────────────────────────────
interface WaveColorTheme {
  id: string
  name: string
  impulse: string       // impulse line color
  impulseFill: string   // circle fill for impulse labels
  corrective: string    // corrective line color
  correctiveFill: string // circle fill for corrective labels
  swatch: [string, string]  // [primary, secondary] for UI swatch
}

const WAVE_THEMES: WaveColorTheme[] = [
  {
    id: 'gold', name: 'Gold',
    impulse: '#fde047', impulseFill: '#facc15',
    corrective: '#fb923c', correctiveFill: '#f97316',
    swatch: ['#fde047', '#fb923c'],
  },
  {
    id: 'classic', name: 'Classic',
    impulse: '#16a34a', impulseFill: '#15803d',
    corrective: '#dc2626', correctiveFill: '#b91c1c',
    swatch: ['#16a34a', '#dc2626'],
  },
  {
    id: 'neon', name: 'Neon',
    impulse: '#06b6d4', impulseFill: '#0891b2',
    corrective: '#e879f9', correctiveFill: '#c026d3',
    swatch: ['#06b6d4', '#e879f9'],
  },
  {
    id: 'ocean', name: 'Ocean',
    impulse: '#2dd4bf', impulseFill: '#14b8a6',
    corrective: '#60a5fa', correctiveFill: '#3b82f6',
    swatch: ['#2dd4bf', '#60a5fa'],
  },
  {
    id: 'mono', name: 'Mono',
    impulse: '#e2e8f0', impulseFill: '#94a3b8',
    corrective: '#64748b', correctiveFill: '#475569',
    swatch: ['#e2e8f0', '#64748b'],
  },
]

function hexToRgba(hex: string, alpha: number): string {
  const r = parseInt(hex.slice(1, 3), 16)
  const g = parseInt(hex.slice(3, 5), 16)
  const b = parseInt(hex.slice(5, 7), 16)
  return `rgba(${r},${g},${b},${alpha})`
}

// Map chart TF keys to API tf param values (empty = daily)
const TF_TO_API: Record<TF, string> = {
  '1m': '1m', '5m': '5m', '15m': '15m', '1h': '1h', '4h': '4h',
  '1D': '', '1W': '1w', '1M': '1mo',
}

// ── User EW adjustments ────────────────────────────────────────────────
// Module-level store for user-dragged wave pivot overrides. Keyed by
// `${symbol}:${tfKey}`. Each entry records the adjusted start/end of W1
// and the end of the last wave (typically W5). After the backend response
// is received, these overrides are applied to primaryCount.waves in place.
type EwAdjustment = {
  w1Start?: { idx: number; price: number }
  w1End?:   { idx: number; price: number }
  wLastEnd?:{ idx: number; price: number }
}
const ewAdjustmentsRef: Record<string, EwAdjustment> = {}
const adjustmentKey = (sym: string, tfKey: TF) => `${sym}:${tfKey}`
// Set by the React component so drag callbacks can trigger a redraw
let ewRedrawTrigger: (() => void) | null = null
function setEwRedrawTrigger(fn: (() => void) | null) { ewRedrawTrigger = fn }
// Loading state callback — lets drawElliottWaves report when it's fetching
// from the EW API so the component can show a progress indicator
let ewLoadingCallback: ((loading: boolean) => void) | null = null
function setEwLoadingCallback(fn: ((loading: boolean) => void) | null) { ewLoadingCallback = fn }

// Helper: find the bar index whose timestamp is closest to `ts`
function nearestBarIdx(barTs: number[], ts: number): number {
  if (!barTs.length) return 0
  if (ts <= barTs[0]) return 0
  if (ts >= barTs[barTs.length - 1]) return barTs.length - 1
  let lo = 0, hi = barTs.length - 1
  while (lo < hi - 1) {
    const mid = (lo + hi) >> 1
    if (barTs[mid] < ts) lo = mid
    else hi = mid
  }
  return Math.abs(barTs[lo] - ts) <= Math.abs(barTs[hi] - ts) ? lo : hi
}

async function drawElliottWaves(chart: Chart, symbol: string, bars: KLineData[], layers: EWLayerOptions = { labels: true, targets: true, status: true }, theme: WaveColorTheme = WAVE_THEMES[0], tfKey: TF = '1D', nocache = false) {
  if (!bars.length) return
  ewLoadingCallback?.(true)
  try {
    // Cap bars sent to EW detection — more bars = exponentially slower
    // detection (O(n²) pivot scanning) with no better wave quality.
    // 1200 daily bars (~4.7 years) covers most bull/bear cycles.
    // Weekly/monthly get more since each bar = 1 week/month.
    const isHigherTf = tfKey === '1W' || tfKey === '1M'
    const barsCount = isHigherTf
      ? Math.max(Math.min(bars.length, 1500), 400)
      : Math.min(bars.length, 1200)
    const apiTf = TF_TO_API[tfKey] || ''
    const params: any = { bars: barsCount }
    if (apiTf) params.tf = apiTf
    if (nocache) params.nocache = true
    const r = await api.get(`/patterns/${symbol}/elliott-waves`, { params })
    const resp: ElliottResponse = r.data || {}
    const serverBars = resp.bars || []
    const waveCounts = resp.wave_counts || []
    const currentPos = resp.current_position
    const bestImpulse = (resp as any).best_impulse as WaveCountData | null
    const linkedCorrection = (resp as any).linked_correction as WaveCountData | null
    const bestCount = resp.best_count

    if (!serverBars.length || !waveCounts.length) {
      console.warn(`[EW] No data for ${symbol}/${tfKey}: bars=${serverBars.length} counts=${waveCounts.length}`)
      return
    }

    const barTs = serverBars.map((b: any) => b.timestamp)
    const lastTs = barTs[barTs.length - 1]
    const barStep = barTs.length > 1 ? barTs[barTs.length - 1] - barTs[barTs.length - 2] : 86400000

    const snap = (idx: number): number => {
      if (idx < 0) idx = 0
      if (idx < barTs.length) return barTs[idx]
      return lastTs + (idx - barTs.length + 1) * barStep
    }

    const add = (config: any) => chart.createOverlay({ ...config, groupId: 'auto-overlays', lock: true, mode: 'normal', needDefaultPointFigure: false, needDefaultXAxisFigure: false, needDefaultYAxisFigure: false, onSelected: () => false, onMouseEnter: () => false, onClick: () => false, onRightClick: () => false, styles: { point: { activeColor: 'transparent', activeBorderColor: 'transparent', activeBorderSize: 0, activeRadius: 0, color: 'transparent', borderColor: 'transparent', borderSize: 0, radius: 0 } } })
    const round2 = (n: number) => Math.round(n * 100) / 100

    // ── Magnet: snap a projected pivot to a real candle body ──────────
    // isPeak=true → snap to candle body top (max(open,close))
    // isPeak=false → snap to candle body bottom (min(open,close))
    // Searches a window of ±radius bars around `idx` for the best match.
    // Returns { idx, price, ts } anchored to real data.
    // If idx is beyond bar data, returns the original projected values.
    const magnetSnap = (
      projIdx: number, projPrice: number, isPeak: boolean, radius = 5,
    ): { idx: number; price: number; ts: number } => {
      if (projIdx < 0 || projIdx >= serverBars.length) {
        // Beyond real data — can't magnet, return projection as-is
        return { idx: projIdx, price: projPrice, ts: snap(projIdx) }
      }
      const from = Math.max(0, projIdx - radius)
      const to = Math.min(serverBars.length - 1, projIdx + radius)
      let bestIdx = projIdx
      let bestPrice = isPeak
        ? Math.max(serverBars[projIdx].open, serverBars[projIdx].close)
        : Math.min(serverBars[projIdx].open, serverBars[projIdx].close)
      for (let i = from; i <= to; i++) {
        const bodyTop = Math.max(serverBars[i].open, serverBars[i].close)
        const bodyBot = Math.min(serverBars[i].open, serverBars[i].close)
        const candidate = isPeak ? bodyTop : bodyBot
        if (isPeak ? candidate > bestPrice : candidate < bestPrice) {
          bestPrice = candidate
          bestIdx = i
        }
      }
      return { idx: bestIdx, price: round2(bestPrice), ts: snap(bestIdx) }
    }

    // ── findSwing: find the best swing high/low in a bar range ────────
    // Uses candle body edges (not wicks) for more reliable pivots.
    // Clamps indices to valid range; if no valid bars, returns the last bar.
    const findSwing = (
      fromIdx: number, toIdx: number, isPeak: boolean,
    ): { idx: number; price: number; ts: number } => {
      const f = Math.max(0, Math.min(fromIdx, serverBars.length - 1))
      const t = Math.max(f, Math.min(toIdx, serverBars.length - 1))
      let bestIdx = f
      let bestPrice = isPeak
        ? Math.max(serverBars[f].open, serverBars[f].close)
        : Math.min(serverBars[f].open, serverBars[f].close)
      for (let i = f + 1; i <= t; i++) {
        const bodyTop = Math.max(serverBars[i].open, serverBars[i].close)
        const bodyBot = Math.min(serverBars[i].open, serverBars[i].close)
        const v = isPeak ? bodyTop : bodyBot
        if (isPeak ? v > bestPrice : v < bestPrice) {
          bestPrice = v
          bestIdx = i
        }
      }
      return { idx: bestIdx, price: round2(bestPrice), ts: snap(bestIdx) }
    }

    // ── Price floor: projected prices can never go below $0.01 ──────
    const priceFloor = (p: number) => Math.max(0.01, round2(p))

    // ── Scale projected range to fit available price space ────────────
    // When a projected move would push below $0, compress the range
    // proportionally so the pattern fits in the available price space.
    // Returns a scaled range that respects the price floor.
    const safeRange = (fromPrice: number, rawRange: number, dir: number): number => {
      if (dir >= 0) return rawRange // only constrain downward moves
      // Maximum downward range: fromPrice - $0.01
      const maxDown = Math.max(fromPrice - 0.01, 0)
      return Math.min(rawRange, maxDown * 0.9) // keep 10% buffer above $0
    }

    // ATR for label spacing
    const atr = bars.length > 14
      ? bars.slice(-14).reduce((s, b) => s + (b.high - b.low), 0) / 14
      : 1

    // ── Color palette from theme ─────────────────────────────────────
    const IMPULSE_CLR    = theme.impulse
    const IMPULSE_CIRCLE = theme.impulseFill
    const CORRECT_CLR    = theme.corrective
    const CORRECT_CIRCLE = theme.correctiveFill
    const MOTIVE_LINE_W  = 5.0   // uniform width for all waves (impulse + corrective)
    const CORR_LINE_W    = 5.0
    const LATE_MOTIVE_W  = 5.0
    const CIRCLE_R       = 12

    // ── Helper: draw a single wave count on chart ────────────────────
    const drawWaveCount = (
      wc: WaveCountData,
      opts: {
        lineColor: string; circleColor: string; circleText: string;
        correctiveLineColor: string; annotColor: string;
        lineWidth: number; isImpulsePhase: boolean;
      },
    ) => {
      if (!wc.waves || wc.waves.length === 0) return

      const isUp = wc.direction === 'up'

      // ── Build continuous pivot chain (no gaps) ──
      // Pivots: [start of wave 0, end of wave 0, end of wave 1, ...]
      // Each segment connects previous pivot to next pivot — always connected.
      const pivots: { ts: number; price: number }[] = []
      if (wc.waves.length > 0) {
        pivots.push({ ts: snap(wc.waves[0].start_idx), price: wc.waves[0].start_price })
        for (const w of wc.waves) {
          pivots.push({ ts: snap(w.end_idx), price: w.end_price })
        }
      }

      // ── Draw zigzag lines between consecutive pivots ──
      // Segment i connects pivot[i] → pivot[i+1], which is wave (i+1):
      //   i=0 → W1, i=1 → W2, i=2 → W3, i=3 → W4, i=4 → W5
      // For ABC corrections: i=0 → A, i=1 → B, i=2 → C (all drawn bold).
      for (let i = 0; i < pivots.length - 1; i++) {
        // Uniform: every wave uses the same base color and width (solid)
        const waveColor = opts.lineColor
        const waveWidth = opts.lineWidth
        const dash: number[] = []

        add({
          name: 'patternTrendline',
          points: [
            { timestamp: pivots[i].ts, value: pivots[i].price },
            { timestamp: pivots[i + 1].ts, value: pivots[i + 1].price },
          ],
          extendData: { color: waveColor, lineWidth: waveWidth, dash },
        })
      }

      // ── Live "in-progress" segment ──────────────────────────────────
      // If the last confirmed pivot is more than 2 bars before the rightmost
      // bar, draw a dashed line from the last pivot to the current price.
      // This ensures the overlay always extends to the right edge of the chart.
      if (pivots.length > 0) {
        const lastPivot = pivots[pivots.length - 1]
        const barsFromEdge = (lastTs - lastPivot.ts) / barStep
        if (barsFromEdge > 2) {
          // Use the last bar's close price as the live endpoint
          const lastBar = serverBars[serverBars.length - 1]
          const livePrice = lastBar?.close ?? lastPivot.price
          add({
            name: 'patternTrendline',
            points: [
              { timestamp: lastPivot.ts, value: lastPivot.price },
              { timestamp: lastTs, value: livePrice },
            ],
            extendData: {
              color: opts.lineColor + '88',  // semi-transparent = in-progress
              lineWidth: opts.lineWidth,
              dash: [6, 4],                  // dashed = not yet confirmed
            },
          })
        }
      }

      // ── Circled wave numbers at pivots (togglable) ──
      // Map roman numeral labels to clean numbers: i→1, ii→2, iii→3, iv→4, v→5, a→A, b→B, c→C
      const LABEL_MAP: Record<string, string> = {
        'i': '1', 'ii': '2', 'iii': '3', 'iv': '4', 'v': '5',
        'I': '1', 'II': '2', 'III': '3', 'IV': '4', 'V': '5',
        'a': 'A', 'b': 'B', 'c': 'C', 'd': 'D', 'e': 'E',
        '(i)': '1', '(ii)': '2', '(iii)': '3', '(iv)': '4', '(v)': '5',
        '(1)': '1', '(2)': '2', '(3)': '3', '(4)': '4', '(5)': '5',
        '(a)': 'A', '(b)': 'B', '(c)': 'C', '(d)': 'D', '(e)': 'E',
        '(A)': 'A', '(B)': 'B', '(C)': 'C', '(D)': 'D', '(E)': 'E',
      }

      if (layers.labels) {
        // Collect all circle positions first, then spread overlapping ones
        const circlePositions: { ts: number; price: number; label: string }[] = []
        for (let i = 0; i < wc.waves.length; i++) {
          const w = wc.waves[i]
          const rawLabel = wc.labels[i] || String(i + 1)
          const label = LABEL_MAP[rawLabel] || rawLabel.toUpperCase()
          const isHigh = w.direction === 'up'
          const offset = atr * (isHigh ? 0.35 : -0.35)
          circlePositions.push({ ts: snap(w.end_idx), price: w.end_price + offset, label })
        }

        // Collision avoidance: push overlapping circles apart vertically
        const minGap = atr * 1.4  // minimum price gap between circle centers
        for (let pass = 0; pass < 6; pass++) {
          for (let i = 0; i < circlePositions.length; i++) {
            for (let j = i + 1; j < circlePositions.length; j++) {
              const a = circlePositions[i], b = circlePositions[j]
              const timeDist = Math.abs(a.ts - b.ts) / barStep
              if (timeDist > 20) continue  // only worry about nearby circles
              const priceDist = Math.abs(a.price - b.price)
              if (priceDist < minGap) {
                const push = (minGap - priceDist) / 2 + atr * 0.1
                if (a.price >= b.price) {
                  a.price += push; b.price -= push
                } else {
                  a.price -= push; b.price += push
                }
              }
            }
          }
        }

        for (const cp of circlePositions) {
          add({
            name: 'waveCircle',
            points: [{ timestamp: cp.ts, value: cp.price }],
            extendData: {
              text: cp.label,
              fillColor: opts.circleColor,
              textColor: opts.circleText,
              radius: CIRCLE_R,
              borderColor: 'rgba(255,255,255,0.25)',
            },
          })
        }

        // Phase annotation — only show if the wave spans >20% of visible bars
        // (avoids cluttering tight sub-wave counts)
        if (wc.waves.length >= 3) {
          const first = wc.waves[0]
          const lastW = wc.waves[wc.waves.length - 1]
          const spanBars = lastW.end_idx - first.start_idx
          if (spanBars > barTs.length * 0.20) {
            const midIdx = Math.floor((first.start_idx + lastW.end_idx) / 2)
            const midTs = snap(midIdx)
            const allPrices = wc.waves.flatMap(w => [w.start_price, w.end_price])
            const arrow = isUp ? '↗' : '↘'
            const phaseText = opts.isImpulsePhase ? `${arrow} Impulse` : `${arrow} Correction`
            const annotPrice = opts.isImpulsePhase
              ? (isUp ? Math.min(...allPrices) - atr * 0.7 : Math.max(...allPrices) + atr * 0.7)
              : (isUp ? Math.max(...allPrices) + atr * 0.7 : Math.min(...allPrices) - atr * 0.7)
            add({
              name: 'patternLabel',
              points: [{ timestamp: midTs, value: annotPrice }],
              extendData: { text: phaseText, color: opts.annotColor + '99', fontSize: 10, offsetY: 0 },
            })
          }
        }
      }
    }

    // ── SINGLE WAVE LAYER: draw the current/recent wave structure (impulse + linked correction) ──
    // We previously had a separate "macro layer" that independently picked
    // another wave count to draw as background context. It caused
    // contradictory overlays (opposite directions, overlapping spans, wrong
    // labels). Removed entirely — one wave count = one drawing, period.
    // Prefer a wave count whose endpoints actually anchor to the chart's
    // true extremes (highest high / lowest low). The backend sometimes
    // picks `best_impulse` with a garbage starting pivot (e.g. 60.40 in
    // the middle of a down-leg) when a better candidate exists in
    // waveCounts that anchors at the real low and terminates at the real
    // high. A structure anchored at the actual chart extremes is always
    // more correct than one that floats off them.
    let trueLowIdx = 0, trueLowPrice = serverBars[0]?.low ?? 0
    let trueHighIdx = 0, trueHighPrice = serverBars[0]?.high ?? 0
    for (let i = 1; i < serverBars.length; i++) {
      if (serverBars[i].low  < trueLowPrice)  { trueLowPrice  = serverBars[i].low;  trueLowIdx  = i }
      if (serverBars[i].high > trueHighPrice) { trueHighPrice = serverBars[i].high; trueHighIdx = i }
    }
    const matchesExtremes = (wc: any): boolean => {
      if (!wc?.waves?.length) return false
      const first = wc.waves[0]
      const last  = wc.waves[wc.waves.length - 1]
      const priceTol = (trueHighPrice - trueLowPrice) * 0.005  // 0.5% of range
      const isUp = wc.direction === 'up'
      const startTarget = isUp ? trueLowPrice  : trueHighPrice
      const endTarget   = isUp ? trueHighPrice : trueLowPrice
      return Math.abs(first.start_price - startTarget) <= priceTol
          && Math.abs(last.end_price   - endTarget)   <= priceTol
    }
    // Scan all wave counts for a complete 5-wave impulse anchored at both
    // extremes; prefer primary degree when multiple match.
    const anchoredCandidates = (waveCounts || []).filter((wc: any) =>
      wc?.type === 'impulse' &&
      wc?.is_complete &&
      wc?.waves?.length === 5 &&
      matchesExtremes(wc)
    )
    const degRank: Record<string, number> = { primary: 0, intermediate: 1, minor: 2 }
    anchoredCandidates.sort((a: any, b: any) =>
      (degRank[a.degree] ?? 99) - (degRank[b.degree] ?? 99) ||
      (b.confidence || 0) - (a.confidence || 0)
    )
    // ── Smart primary selection: pick the structurally best 5-wave count ──
    // The backend ranks by raw confidence, but a high-confidence count
    // with a 12% W4 retracement is structurally invalid (that's noise
    // within W3, not a real pullback). We re-rank by combining confidence
    // with wave proportion quality so the best-looking count wins.
    const _is5 = (wc: any) => wc?.type === 'impulse' && wc?.is_complete && wc?.waves?.length === 5
    const _waveQuality = (wc: any): number => {
      if (!wc?.waves || wc.waves.length < 5) return 0
      const ws = wc.waves
      const w1 = Math.abs(ws[0].end_price - ws[0].start_price)
      const w3 = Math.abs(ws[2].end_price - ws[2].start_price)
      const w4 = Math.abs(ws[3].end_price - ws[3].start_price)
      const w5 = Math.abs(ws[4].end_price - ws[4].start_price)
      if (w3 <= 0) return 0
      const w4pct = w4 / w3  // W4 retracement of W3 (ideal: 0.236 - 0.50)
      // W4 quality: penalize < 15% (noise) or > 75% (violates impulse rules)
      let q = 1.0
      if (w4pct < 0.15) q *= 0.3      // too shallow — not a real pullback
      else if (w4pct < 0.20) q *= 0.7  // marginal
      else if (w4pct > 0.75) q *= 0.2  // too deep
      else if (w4pct > 0.618) q *= 0.6 // questionable
      // Bonus: W3 is longest (standard impulse rule)
      if (w3 >= w1 && w3 >= w5) q *= 1.1
      // Penalize if W5 is extremely short relative to W1 (< 20%)
      if (w1 > 0 && w5 / w1 < 0.2) q *= 0.5
      // Span bonus: prefer counts that cover more of the chart
      const span = ws[4].end_idx - ws[0].start_idx
      const spanRatio = span / Math.max(serverBars.length, 1)
      q *= (0.7 + spanRatio * 0.3)  // slight boost for wider coverage
      return q
    }
    // Score all complete 5-wave impulses and pick the best.
    // Boost counts whose endpoint reaches the chart's actual extreme —
    // users expect the wave to explain the peak/trough they can see.
    const qualifiedCounts = (waveCounts || [])
      .filter((wc: any) => _is5(wc))
      .map((wc: any) => {
        const ws = wc.waves
        const firstPrice = ws[0].start_price
        const lastPrice = ws[ws.length - 1].end_price
        const isUp = wc.direction === 'up'
        const startTarget = isUp ? trueLowPrice : trueHighPrice
        const endTarget   = isUp ? trueHighPrice : trueLowPrice
        const startErr = startTarget > 0 ? Math.abs(firstPrice - startTarget) / startTarget : 1
        const endErr   = endTarget   > 0 ? Math.abs(lastPrice  - endTarget)   / endTarget   : 1
        // Boost counts anchored at chart extremes — 1.5x each
        const startBonus = startErr <= 0.03 ? 1.5 : 1.0
        const peakBonus  = endErr   <= 0.03 ? 1.5 : 1.0
        return {
          wc,
          score: ((wc.confidence || 0) / 100) * _waveQuality(wc) * startBonus * peakBonus,
        }
      })
      .sort((a: any, b: any) => b.score - a.score)
    const bestQualified = qualifiedCounts.length > 0 ? qualifiedCounts[0].wc : null
    const primaryCount =
      anchoredCandidates[0] ||
      bestQualified ||
      bestImpulse || bestCount
    let drewSomething = false

    // ── Apply user adjustments to W1 start/end and the last wave's end ──
    // Stored overrides take precedence over backend data. The handles that
    // produced them are draggable diamonds added further down.
    const adjKey = adjustmentKey(symbol, tfKey)
    const savedAdj = ewAdjustmentsRef[adjKey]
    if (savedAdj && primaryCount && primaryCount.waves && primaryCount.waves.length > 0) {
      const waves = primaryCount.waves
      if (savedAdj.w1Start) {
        waves[0].start_idx = savedAdj.w1Start.idx
        waves[0].start_price = savedAdj.w1Start.price
      }
      if (savedAdj.w1End) {
        waves[0].end_idx = savedAdj.w1End.idx
        waves[0].end_price = savedAdj.w1End.price
        if (waves.length > 1) {
          waves[1].start_idx = savedAdj.w1End.idx
          waves[1].start_price = savedAdj.w1End.price
        }
      }
      if (savedAdj.wLastEnd) {
        const last = waves[waves.length - 1]
        last.end_idx = savedAdj.wLastEnd.idx
        last.end_price = savedAdj.wLastEnd.price
      }
    }

    // ── Snap last wave to chart extreme (cheat code) ──────────────────────
    // The backend pivot detector sometimes terminates the final wave before
    // the true chart high/low, producing an obviously wrong count (e.g.
    // wave 3 ending at 20,000 when the chart peak is at 24,019). Since
    // klinecharts' own `candle.priceMark.high/low` auto-draws the true
    // extremes, we use the same data (serverBars) to snap the last wave's
    // end to the real extreme — but only if the user hasn't manually
    // overridden it via the drag handle (savedAdj.wLastEnd wins).
    if (
      primaryCount &&
      primaryCount.waves &&
      primaryCount.waves.length > 0 &&
      !(savedAdj && savedAdj.wLastEnd) &&
      serverBars.length > 0
    ) {
      const isUpSnap = primaryCount.direction === 'up'
      let extremeIdx = 0
      let extremePrice = isUpSnap ? serverBars[0].high : serverBars[0].low
      for (let i = 1; i < serverBars.length; i++) {
        const v = isUpSnap ? serverBars[i].high : serverBars[i].low
        if (isUpSnap ? v > extremePrice : v < extremePrice) {
          extremePrice = v
          extremeIdx = i
        }
      }
      const waves = primaryCount.waves
      const firstStartIdx = waves[0].start_idx
      const last = waves[waves.length - 1]
      // Only snap if the extreme lies inside the wave count's span and
      // the current last-wave endpoint is materially off (>1.5%).
      if (extremeIdx >= firstStartIdx && extremePrice > 0) {
        const err = Math.abs(last.end_price - extremePrice) / extremePrice
        const directionOk = isUpSnap
          ? extremePrice >= last.end_price
          : extremePrice <= last.end_price
        if (err > 0.015 && directionOk) {
          last.end_idx = extremeIdx
          last.end_price = extremePrice
        }
      }
    }

    if (primaryCount && primaryCount.waves && primaryCount.waves.length > 0) {
      const isUp = primaryCount.direction === 'up'

      drawWaveCount(primaryCount, {
        lineColor: IMPULSE_CLR,
        circleColor: IMPULSE_CIRCLE,
        circleText: '#ffffff',
        correctiveLineColor: IMPULSE_CLR,
        annotColor: IMPULSE_CLR,
        lineWidth: MOTIVE_LINE_W,
        isImpulsePhase: true,
      })
      drewSomething = true

      // Draw linked correction (ABC) if available
      if (linkedCorrection && linkedCorrection.waves && linkedCorrection.waves.length > 0) {
        const corrIsDown = linkedCorrection.direction === 'down'

        // ── BRIDGE LINE: connect impulse end → correction start (no gap) ──
        const impulseLastWave = primaryCount.waves[primaryCount.waves.length - 1]
        const corrFirstWave = linkedCorrection.waves[0]
        add({
          name: 'patternTrendline',
          points: [
            { timestamp: snap(impulseLastWave.end_idx), value: impulseLastWave.end_price },
            { timestamp: snap(corrFirstWave.start_idx), value: corrFirstWave.start_price },
          ],
          extendData: { color: IMPULSE_CLR, lineWidth: MOTIVE_LINE_W, dash: [] },
        })

        drawWaveCount(linkedCorrection, {
          lineColor: IMPULSE_CLR,
          circleColor: IMPULSE_CIRCLE,
          circleText: '#ffffff',
          correctiveLineColor: IMPULSE_CLR,
          annotColor: IMPULSE_CLR,
          lineWidth: CORR_LINE_W,
          isImpulsePhase: false,
        })

        // ── CONTINUATION: look for a detected next-impulse after C ──────
        // The backend returns every candidate wave count; if it has detected a
        // new impulse whose start is at/after the C pivot, draw that too so
        // the structure visibly extends to the current candle. We do NOT
        // fabricate pivots — we only render waves the engine detected.
        const cWave = linkedCorrection.waves[linkedCorrection.waves.length - 1]
        const cIdx = cWave.end_idx
        const newDirExpected = corrIsDown ? 'up' : 'down'  // opposite of correction
        const nextImpulse = waveCounts.find((wc: any) =>
          wc !== primaryCount
          && wc !== linkedCorrection
          && wc.type === 'impulse'
          && wc.direction === newDirExpected
          && wc.waves?.length >= 2
          && wc.waves[0].start_idx >= cIdx - 2  // starts at or just after C
        )
        if (nextImpulse) {
          // Bridge C → start of next impulse (if not already coincident)
          const nextStart = nextImpulse.waves[0]
          if (nextStart.start_idx > cIdx) {
            add({
              name: 'patternTrendline',
              points: [
                { timestamp: snap(cIdx), value: cWave.end_price },
                { timestamp: snap(nextStart.start_idx), value: nextStart.start_price },
              ],
              extendData: { color: IMPULSE_CLR, lineWidth: MOTIVE_LINE_W, dash: [] },
            })
          }
          drawWaveCount(nextImpulse, {
            lineColor: IMPULSE_CLR,
            circleColor: IMPULSE_CIRCLE,
            circleText: '#ffffff',
            correctiveLineColor: IMPULSE_CLR,
            annotColor: IMPULSE_CLR,
            lineWidth: MOTIVE_LINE_W,
            isImpulsePhase: true,
          })
        }
      }
    }

    // ── PROJECTED WAVES — complete the picture ──────────────────────
    // 1) Partial impulse (2-4 waves): project remaining waves to complete 1-2-3-4-5
    // 2) Complete impulse (5 waves) with no linked correction: project ABC
    // 3) Complete correction with no linked impulse: project 1-2-3-4-5
    if (layers.targets && primaryCount && primaryCount.waves && primaryCount.waves.length >= 2) {
      const hasLinkedCorr = linkedCorrection && linkedCorrection.waves && linkedCorrection.waves.length > 0
      const isImpulse = primaryCount.type === 'impulse'
      const isUp = primaryCount.direction === 'up'
      const numWaves = primaryCount.waves.length
      const isComplete = primaryCount.is_complete === true || numWaves >= 5

      const anchorWc = hasLinkedCorr ? linkedCorrection! : primaryCount
      const anchorWave = anchorWc.waves[anchorWc.waves.length - 1]
      const anchorTs = snap(anchorWave.end_idx)
      const anchorPrice = anchorWave.end_price
      const stepBars = Math.max(12, Math.floor(barTs.length * 0.06))

      // Projection style constants — same color as realized waves. Impulse
      // completions (W3/W4/W5) draw solid; corrective projections (ABC) dotted.
      const projLineW = 5.0
      const projDashSolid: number[] = []
      const projDashDotted: number[] = [3, 6]
      const projCircleR = 12

      // Helper: draw projected zigzag + circled labels
      const visiblePriceMin = Math.min(...serverBars.map((b: any) => b.low  ?? b.close))
      const visiblePriceMax = Math.max(...serverBars.map((b: any) => b.high ?? b.close))
      const visiblePriceRange = Math.max(visiblePriceMax - visiblePriceMin, visiblePriceMin * 0.01)

      const drawProjection = (
        pivots: { ts: number; price: number }[],
        labels: { label: string; ts: number; price: number; above: boolean }[],
        phaseLabel: { ts: number; price: number; text: string; sub: string },
        color: string,
        cColor: string,
        dash: number[] = projDashDotted,
      ) => {
        // Sanity gate: reject if any projected price is >60% outside the visible price range
        const priceLo = visiblePriceMin - visiblePriceRange * 0.6
        const priceHi = visiblePriceMax + visiblePriceRange * 0.6
        if (pivots.some(p => p.price < priceLo || p.price > priceHi)) return

        // Apply price floor — no projected price can go below $0.01
        const fp = pivots.map(p => ({ ...p, price: priceFloor(p.price) }))
        const fl = labels.map(l => ({ ...l, price: priceFloor(l.price) }))
        const fph = { ...phaseLabel, price: priceFloor(phaseLabel.price) }

        for (let i = 0; i < fp.length - 1; i++) {
          add({
            name: 'patternTrendline',
            points: [
              { timestamp: fp[i].ts, value: fp[i].price },
              { timestamp: fp[i + 1].ts, value: fp[i + 1].price },
            ],
            extendData: { color, lineWidth: projLineW, dash },
          })
        }
        for (const pl of fl) {
          add({
            name: 'waveCircle',
            points: [{ timestamp: pl.ts, value: pl.price + atr * (pl.above ? 0.35 : -0.35) }],
            extendData: { text: pl.label, fillColor: cColor, textColor: '#ffffffcc', radius: projCircleR, borderColor: 'rgba(255,255,255,0.15)' },
          })
        }
        // Compact projected phase label — small text, not a big pill
        add({
          name: 'patternLabel',
          points: [{ timestamp: fph.ts, value: fph.price }],
          extendData: { text: `${fph.text}`, color: color + 'cc', fontSize: 9, offsetY: 0 },
        })
      }

      // ── Case 1: PARTIAL IMPULSE — project remaining waves (4-5, or 3-4-5, etc.) then ABC ──
      if (isImpulse && !isComplete && numWaves < 5) {
        const w1 = primaryCount.waves[0]
        const w1Range = Math.abs(w1.end_price - w1.start_price)
        const dir = isUp ? 1 : -1
        // Wave duration in bars — used for proportional projections
        const w1Bars = Math.max(3, w1.end_idx - w1.start_idx)
        const projStep = Math.max(5, Math.floor(w1Bars * 0.618))  // wave 5 ≈ 0.618 × wave 1 duration

        // Helper: find sequential wave pivots in actual bar data after the last detected wave.
        // Instead of finding global extremes, this finds them IN ORDER:
        //   Wave 4 = first significant counter-move, Wave 5 = extreme after that
        // This ensures waves are chronologically correct and reach the last candle.
        const lastDetectedIdx = primaryCount.waves[numWaves - 1].end_idx
        const afterBars = serverBars.slice(lastDetectedIdx + 1)
        const afterStart = lastDetectedIdx + 1

        // Find sequential pivots: alternating high/low swings in remaining bars
        const findSequentialPivots = (startAfterIdx: number, count: number, firstSeekLow: boolean) => {
          const pivots: { price: number; barIdx: number }[] = []
          let seekLow = firstSeekLow
          let searchFrom = startAfterIdx

          for (let p = 0; p < count; p++) {
            if (searchFrom >= serverBars.length) break
            // Find the extreme from searchFrom to end
            let bestPrice = seekLow ? Infinity : -Infinity
            let bestIdx = -1
            for (let bi = searchFrom; bi < serverBars.length; bi++) {
              const bar = serverBars[bi]
              const val = seekLow ? bar.low : bar.high
              if (seekLow ? val < bestPrice : val > bestPrice) {
                bestPrice = val; bestIdx = bi
              }
            }
            if (bestIdx < 0) break
            pivots.push({ price: round2(bestPrice), barIdx: bestIdx })
            // For next pivot, search AFTER this one (must be chronologically later)
            searchFrom = bestIdx + 1
            seekLow = !seekLow
          }
          return pivots
        }

        // Build remaining wave targets using actual bars when available
        const remaining: { label: string; price: number; ts: number; above: boolean }[] = []
        const projPivots: { ts: number; price: number }[] = [{ ts: anchorTs, price: anchorPrice }]

        let curPrice = anchorPrice
        let curTs = anchorTs

        if (numWaves === 4) {
          // Need only wave 5 — search chart bars after wave 4 (anchor) for the impulse extreme
          let w4BarIdx = bars.length - 1
          let bestDist = Infinity
          for (let i = 0; i < bars.length; i++) {
            const dist = Math.abs(bars[i].timestamp - curTs)
            if (dist < bestDist) { bestDist = dist; w4BarIdx = i }
          }
          let extremePrice = isUp ? -Infinity : Infinity
          let extremeIdx = -1
          for (let i = w4BarIdx + 1; i < bars.length; i++) {
            const b = bars[i]
            if (isUp && (b.high as number) > extremePrice) {
              extremePrice = b.high as number; extremeIdx = i
            } else if (!isUp && (b.low as number) < extremePrice) {
              extremePrice = b.low as number; extremeIdx = i
            }
          }
          const validSnap = extremeIdx >= 0 && (isUp ? extremePrice > curPrice : extremePrice < curPrice)
          if (validSnap) {
            const extremeTs = bars[extremeIdx].timestamp
            projPivots.push({ ts: extremeTs, price: round2(extremePrice) })
            remaining.push({ label: '5', ts: extremeTs, price: round2(extremePrice), above: isUp })
            curPrice = round2(extremePrice); curTs = extremeTs
          } else {
            const impStart = primaryCount.waves[0].start_price
            const fibW5 = round2(impStart + dir * w1Range * 2.618)
            const w5Ts = curTs + projStep * barStep
            projPivots.push({ ts: w5Ts, price: fibW5 })
            remaining.push({ label: '5', ts: w5Ts, price: fibW5, above: isUp })
            curPrice = fibW5; curTs = w5Ts
          }
        }

        if (numWaves === 3) {
          // Need waves 4 and 5
          // Wave 4 = counter-move (bullish: low, bearish: high)
          // Wave 5 = continuation (bullish: high, bearish: low)
          const seekLowFirst = isUp  // bullish wave 4 = pullback (low), bearish wave 4 = bounce (high→seekLow=false)
          const actuals = findSequentialPivots(afterStart, 2, seekLowFirst)

          if (actuals.length >= 1) {
            // Wave 4 from actual data
            projPivots.push({ ts: snap(actuals[0].barIdx), price: actuals[0].price })
            remaining.push({ label: '4', ts: snap(actuals[0].barIdx), price: actuals[0].price, above: !isUp })
            curPrice = actuals[0].price; curTs = snap(actuals[0].barIdx)
          } else {
            // Fibonacci fallback
            const w3Range = Math.abs(primaryCount.waves[2].end_price - primaryCount.waves[2].start_price)
            const w4Price = round2(anchorPrice - dir * w3Range * 0.382)
            const w4Ts = curTs + projStep * barStep
            projPivots.push({ ts: w4Ts, price: w4Price })
            remaining.push({ label: '4', ts: w4Ts, price: w4Price, above: !isUp })
            curPrice = w4Price; curTs = w4Ts
          }

          {
            // Wave 5: search chart bars AFTER wave 4 for the impulse extreme
            // Find the bar in chart data closest to wave 4's timestamp, then search after it
            const w4Ts = curTs
            let w4BarIdx = bars.length - 1
            let bestDist = Infinity
            for (let i = 0; i < bars.length; i++) {
              const dist = Math.abs(bars[i].timestamp - w4Ts)
              if (dist < bestDist) { bestDist = dist; w4BarIdx = i }
            }
            // Search bars after wave 4 for the impulse extreme
            let extremePrice = isUp ? -Infinity : Infinity
            let extremeIdx = -1
            for (let i = w4BarIdx + 1; i < bars.length; i++) {
              const b = bars[i]
              if (isUp && (b.high as number) > extremePrice) {
                extremePrice = b.high as number; extremeIdx = i
              } else if (!isUp && (b.low as number) < extremePrice) {
                extremePrice = b.low as number; extremeIdx = i
              }
            }
            // Only use candle snap if extreme is more extreme than wave 4 in the impulse direction
            const validSnap = extremeIdx >= 0 && (isUp ? extremePrice > curPrice : extremePrice < curPrice)
            if (validSnap) {
              const extremeTs = bars[extremeIdx].timestamp
              projPivots.push({ ts: extremeTs, price: round2(extremePrice) })
              remaining.push({ label: '5', ts: extremeTs, price: round2(extremePrice), above: isUp })
              curPrice = round2(extremePrice); curTs = extremeTs
            } else {
              // Fibonacci projection: wave 5 = impulse start + 2.618 * wave 1 range
              const impStart = primaryCount.waves[0].start_price
              const fibW5 = round2(impStart + dir * w1Range * 2.618)
              const w5Ts = curTs + projStep * barStep
              projPivots.push({ ts: w5Ts, price: fibW5 })
              remaining.push({ label: '5', ts: w5Ts, price: fibW5, above: isUp })
              curPrice = fibW5; curTs = w5Ts
            }
          }
        }
        if (numWaves === 2) {
          // Need waves 3, 4, 5 — find 3 sequential pivots
          // Wave 3 = continuation, Wave 4 = counter, Wave 5 = continuation
          const seekLowFirst = !isUp  // bullish wave 3 = high (seekLow=false), bearish wave 3 = low (seekLow=true)
          const actuals = findSequentialPivots(afterStart, 3, seekLowFirst)

          if (actuals.length >= 1) {
            projPivots.push({ ts: snap(actuals[0].barIdx), price: actuals[0].price })
            remaining.push({ label: '3', ts: snap(actuals[0].barIdx), price: actuals[0].price, above: isUp })
            curPrice = actuals[0].price; curTs = snap(actuals[0].barIdx)
          } else {
            const w3Price = round2(curPrice + dir * w1Range * 1.618)
            const w3Ts = curTs + projStep * barStep * 1.5
            projPivots.push({ ts: w3Ts, price: w3Price })
            remaining.push({ label: '3', ts: w3Ts, price: w3Price, above: isUp })
            curPrice = w3Price; curTs = w3Ts
          }

          if (actuals.length >= 2) {
            projPivots.push({ ts: snap(actuals[1].barIdx), price: actuals[1].price })
            remaining.push({ label: '4', ts: snap(actuals[1].barIdx), price: actuals[1].price, above: !isUp })
            curPrice = actuals[1].price; curTs = snap(actuals[1].barIdx)
          } else {
            const w3Range = Math.abs(curPrice - (actuals.length >= 1 ? actuals[0].price : anchorPrice))
            const w4Price = round2(curPrice - dir * Math.max(w3Range, w1Range) * 0.382)
            const w4Ts = curTs + projStep * barStep
            projPivots.push({ ts: w4Ts, price: w4Price })
            remaining.push({ label: '4', ts: w4Ts, price: w4Price, above: !isUp })
            curPrice = w4Price; curTs = w4Ts
          }

          {
            // Wave 5: search chart bars AFTER wave 4 for the impulse extreme
            const w4Ts2 = curTs
            let w4BarIdx2 = bars.length - 1
            let bestDist2 = Infinity
            for (let i = 0; i < bars.length; i++) {
              const dist = Math.abs(bars[i].timestamp - w4Ts2)
              if (dist < bestDist2) { bestDist2 = dist; w4BarIdx2 = i }
            }
            let extremePrice = isUp ? -Infinity : Infinity
            let extremeIdx = -1
            for (let i = w4BarIdx2 + 1; i < bars.length; i++) {
              const b = bars[i]
              if (isUp && (b.high as number) > extremePrice) {
                extremePrice = b.high as number; extremeIdx = i
              } else if (!isUp && (b.low as number) < extremePrice) {
                extremePrice = b.low as number; extremeIdx = i
              }
            }
            const validSnap2 = extremeIdx >= 0 && (isUp ? extremePrice > curPrice : extremePrice < curPrice)
            if (validSnap2) {
              const extremeTs = bars[extremeIdx].timestamp
              projPivots.push({ ts: extremeTs, price: round2(extremePrice) })
              remaining.push({ label: '5', ts: extremeTs, price: round2(extremePrice), above: isUp })
              curPrice = round2(extremePrice); curTs = extremeTs
            } else {
              const w5Price = round2(curPrice + dir * w1Range * 0.618)
              const w5Ts = curTs + projStep * barStep
              projPivots.push({ ts: w5Ts, price: w5Price })
              remaining.push({ label: '5', ts: w5Ts, price: w5Price, above: isUp })
              curPrice = w5Price; curTs = w5Ts
            }
          }
        }

        if (remaining.length > 0) {
          const projColor = IMPULSE_CLR
          const cColor = IMPULSE_CIRCLE
          const allProjPrices = projPivots.map(p => p.price)
          const midTs2 = projPivots[0].ts + (projPivots[projPivots.length - 1].ts - projPivots[0].ts) / 2
          const phaseY = isUp ? Math.min(...allProjPrices) - atr * 0.7 : Math.max(...allProjPrices) + atr * 0.7

          drawProjection(projPivots, remaining,
            { ts: midTs2, price: phaseY, text: 'Projected Waves', sub: isUp ? '↗ Completing impulse' : '↘ Completing impulse' },
            projColor, cColor, projDashSolid)

          // Also project ABC after the projected wave 5
          const lastProj = projPivots[projPivots.length - 1]
          const impStart = primaryCount.waves[0].start_price
          const impRange = Math.abs(lastProj.price - impStart)
          const corrDir2 = isUp ? -1 : 1
          // ABC step proportional to actual wave durations
          const avgWaveBarsC1 = primaryCount.waves.reduce((s: number, w: any) => s + (w.end_idx - w.start_idx), 0) / primaryCount.waves.length
          const abcStep = Math.max(5, Math.floor(avgWaveBarsC1 * 0.4))
          const aTs = lastProj.ts + abcStep * barStep
          const aPrice = round2(lastProj.price + corrDir2 * impRange * 0.382)
          const bTs = aTs + abcStep * barStep * 0.7
          const bPrice = round2(aPrice - corrDir2 * impRange * 0.382 * 0.50)
          const cTs = bTs + abcStep * barStep
          const cPrice = round2(lastProj.price + corrDir2 * impRange * 0.618)
          const abcColor = IMPULSE_CLR
          const abcCircle = IMPULSE_CIRCLE
          const abcPivots = [lastProj, { ts: aTs, price: aPrice }, { ts: bTs, price: bPrice }, { ts: cTs, price: cPrice }]
          const abcLabels = [
            { label: 'A', ts: aTs, price: aPrice, above: corrDir2 < 0 },
            { label: 'B', ts: bTs, price: bPrice, above: corrDir2 > 0 },
            { label: 'C', ts: cTs, price: cPrice, above: corrDir2 < 0 },
          ]
          const abcMid = aTs + (cTs - aTs) / 2
          const abcY = isUp ? Math.min(aPrice, cPrice) - atr * 0.7 : Math.max(aPrice, cPrice) + atr * 0.7
          drawProjection(abcPivots, abcLabels,
            { ts: abcMid, price: abcY, text: 'Corrective Wave', sub: isUp ? '↘ Projected ABC' : '↗ Projected ABC' },
            abcColor, abcCircle)
        }
      }

      // ── Case 2: COMPLETE IMPULSE, no linked correction → project ABC + next impulse ──
      // When enough post-impulse bars exist, anchor A and B to REAL candle
      // pivots so the correction follows what actually happened, not a
      // pure Fibonacci guess. Then project C and a new 1-2-3-4-5 impulse
      // continuation after C — the full EW predictive cycle.
      if (isImpulse && isComplete && !hasLinkedCorr) {
        const impStart = primaryCount.waves[0].start_price
        const impEnd = anchorPrice
        const impRange = Math.abs(impEnd - impStart)
        const corrDir = isUp ? -1 : 1
        const impEndIdx = primaryCount.waves[primaryCount.waves.length - 1].end_idx
        const barsAfterImp = serverBars.length - 1 - impEndIdx
        // Only project if impulse ended in the last 15% of visible bars — otherwise it's stale
        const maxAllowedBarsAfter = Math.max(15, Math.floor(serverBars.length * 0.15))
        if (barsAfterImp > maxAllowedBarsAfter) return

        // Proportional timing — base projection step on actual wave durations
        const avgWaveBars = primaryCount.waves.reduce((s: number, w: any) => s + (w.end_idx - w.start_idx), 0) / primaryCount.waves.length

        let aPrice: number, aTs: number, aIdx: number
        let bPrice: number, bTs: number, bIdx: number
        let cPrice: number, cTs: number

        if (barsAfterImp > 10) {
          // ── Use REAL candle data for A and B (magnet to candle body) ──
          // A = the first significant swing extreme after impulse end.
          // Search a limited window (half the impulse span or remaining bars).
          // Uses candle body (open/close) not wicks for reliable pivots.
          const impSpan = primaryCount.waves[primaryCount.waves.length - 1].end_idx - primaryCount.waves[0].start_idx
          const aSearchEnd = Math.min(impEndIdx + 1 + Math.max(Math.floor(impSpan * 0.5), 20), serverBars.length - 1)
          // For up impulse correction: A is a trough (isPeak=false)
          // For down impulse correction: A is a peak (isPeak=true)
          const aSwing = findSwing(impEndIdx + 1, aSearchEnd, !isUp)
          aIdx = aSwing.idx
          aPrice = aSwing.price
          aTs = aSwing.ts

          // Validate: A must retrace at least 5% of the impulse — otherwise skip ABC
          const aRetrace = Math.abs(impEnd - aPrice) / impRange
          if (aRetrace < 0.05) {
            // aRetrace too shallow — skip ABC
          } else {

          // B = the bounce after A (candle body magnet).
          // Cap B so it never exceeds the impulse end price.
          const aLegLen = aIdx - impEndIdx
          const bSearchEnd2 = Math.min(aIdx + 1 + Math.max(aLegLen, 10), serverBars.length - 1)
          const postACount = bSearchEnd2 - aIdx
          if (postACount >= 2) {
            // B is a peak for up-impulse correction, trough for down
            const bSwing = findSwing(aIdx + 1, bSearchEnd2, isUp)
            bIdx = bSwing.idx
            // Cap at impulse end price
            bPrice = isUp
              ? Math.min(bSwing.price, impEnd)
              : Math.max(bSwing.price, impEnd)
            bTs = bSwing.ts
          } else {
            // Not enough data after A for B — project B
            bIdx = aIdx + Math.max(5, Math.floor(stepBars * 0.2))
            const aMove = Math.abs(impEnd - aPrice)
            bPrice = round2(aPrice - corrDir * aMove * 0.50)
            bTs = snap(Math.min(bIdx, serverBars.length - 1))
          }

          // C = Fibonacci extension of the A leg (impEnd→A) projected from B.
          // In EW theory C typically equals 0.618× to 1.0× of A leg length.
          // Use 1.0× so C ≈ same depth as A (measured from B).
          // But cap C so it doesn't exceed A — C beyond A is rare (flat or
          // expanded flat territory, not a standard zigzag).
          const aLegMove = Math.abs(impEnd - aPrice) // full A leg
          const bRetrace = Math.abs(bPrice - aPrice) // B bounce
          const cFromB = Math.max(bRetrace * 1.0, aLegMove * 0.618) // C ≈ B retrace or 61.8% of A
          const cStepBars = Math.max(aIdx - impEndIdx, 8)
          const cRaw = bPrice + corrDir * cFromB
          // Cap: C shouldn't go deeper than A (standard zigzag)
          cPrice = isUp
            ? round2(Math.max(cRaw, aPrice))   // up impulse: correction goes down, C can't be below A
            : round2(Math.min(cRaw, aPrice))   // down impulse: correction goes up, C can't be above A
          cTs = bTs + cStepBars * barStep

          // Draw ABC + projected impulse (inside the aRetrace validation block)
          const projColor2 = IMPULSE_CLR
          const cColor2 = IMPULSE_CIRCLE

          drawProjection(
            [{ ts: anchorTs, price: impEnd }, { ts: aTs, price: aPrice }, { ts: bTs, price: bPrice }, { ts: cTs, price: cPrice }],
            [
              { label: 'A', ts: aTs, price: aPrice, above: corrDir < 0 },
              { label: 'B', ts: bTs, price: bPrice, above: corrDir > 0 },
              { label: 'C', ts: cTs, price: cPrice, above: corrDir < 0 },
            ],
            { ts: aTs + (cTs - aTs) / 2, price: isUp ? Math.min(aPrice, cPrice) - atr * 0.7 : Math.max(aPrice, cPrice) + atr * 0.7, text: 'Corrective Wave', sub: isUp ? '↘ Projected' : '↗ Projected' },
            projColor2, cColor2,
          )

          // ── CHANNEL-FIRST projection: define channel, waves fit ON it ──
          // In proper EW, waves travel within a parallel channel:
          //   Corrective line: through C, W2, W4 (corrective sub-wave troughs/peaks)
          //   Motive line:     through W1, W3, W5 (motive sub-wave peaks/troughs)
          // We define the channel first, then place waves on its boundaries.
          const newDir2 = isUp ? 1 : -1
          const corrRange2Raw = Math.abs(impEnd - cPrice)
          const corrRange2 = safeRange(cPrice, corrRange2Raw, newDir2)
          const newStep2 = Math.max(8, Math.floor(avgWaveBars * 0.6))

          // Channel parameters
          const totalProjMs = newStep2 * 4.4 * barStep  // total duration for ~5 waves
          const chEndTarget = cPrice + newDir2 * corrRange2 * 0.764  // corrective line end target
          const chSlope = (chEndTarget - cPrice) / totalProjMs       // price per ms
          const chWidth = corrRange2 * 0.35                           // channel width

          // Channel line helpers
          const corrLine = (ts: number) => cPrice + chSlope * (ts - cTs)
          const motiveLine = (ts: number) => corrLine(ts) + newDir2 * chWidth

          // Find the effective C index in bar data
          const cBarIdx = Math.min(bIdx + Math.max(aIdx - impEndIdx, 8), serverBars.length - 1)
          const barsAfterC = serverBars.length - 1 - bIdx

          // Wave timing
          const t1 = cTs + newStep2 * barStep
          const t2 = t1 + Math.floor(newStep2 * 0.5) * barStep
          const t3 = t2 + Math.floor(newStep2 * 1.5) * barStep
          const t4 = t3 + Math.floor(newStep2 * 0.4) * barStep
          const t5 = t4 + newStep2 * barStep

          let n1Price: number, n1Ts: number
          let n2Price: number, n2Ts: number
          let n3Price: number, n3Ts: number
          let n4Price: number, n4Ts: number
          let n5Price: number, n5Ts: number

          if (barsAfterC > 5) {
            // Use REAL candle body data for W1 and W2, then channel for W3-W5
            const w1Swing = findSwing(bIdx + 1, serverBars.length - 1, isUp)
            n1Price = w1Swing.price; n1Ts = w1Swing.ts

            if (serverBars.length - 1 - w1Swing.idx >= 3) {
              const w2Swing = findSwing(w1Swing.idx + 1, serverBars.length - 1, !isUp)
              n2Price = w2Swing.price; n2Ts = w2Swing.ts
            } else {
              n2Price = round2(corrLine(t2)); n2Ts = t2
            }
            // Derive channel from real W1/W2: corrective line through C and W2
            const realSlope = (n2Price - cPrice) / Math.max(1, n2Ts - cTs)
            const realCorrAt = (ts: number) => cPrice + realSlope * (ts - cTs)
            const realChW = Math.abs(n1Price - (cPrice + realSlope * (n1Ts - cTs)))
            const realMotiveAt = (ts: number) => realCorrAt(ts) + newDir2 * realChW

            // W3, W4, W5 placed ON the channel derived from real W1/W2
            n3Ts = n2Ts + Math.floor(newStep2 * 1.5) * barStep
            n3Price = round2(realMotiveAt(n3Ts))
            n4Ts = n3Ts + Math.floor(newStep2 * 0.4) * barStep
            n4Price = round2(realCorrAt(n4Ts))
            n5Ts = n4Ts + newStep2 * barStep
            n5Price = round2(realMotiveAt(n5Ts))
          } else {
            // Pure channel projection — all waves placed ON channel lines
            n1Ts = t1; n1Price = round2(motiveLine(t1))
            n2Ts = t2; n2Price = round2(corrLine(t2))
            n3Ts = t3; n3Price = round2(motiveLine(t3))
            n4Ts = t4; n4Price = round2(corrLine(t4))
            n5Ts = t5; n5Price = round2(motiveLine(t5))
          }

          drawProjection(
            [
              { ts: cTs, price: cPrice },
              { ts: n1Ts, price: n1Price },
              { ts: n2Ts, price: n2Price },
              { ts: n3Ts, price: n3Price },
              { ts: n4Ts, price: n4Price },
              { ts: n5Ts, price: n5Price },
            ],
            [
              { label: '1', ts: n1Ts, price: n1Price, above: newDir2 > 0 },
              { label: '2', ts: n2Ts, price: n2Price, above: newDir2 < 0 },
              { label: '3', ts: n3Ts, price: n3Price, above: newDir2 > 0 },
              { label: '4', ts: n4Ts, price: n4Price, above: newDir2 < 0 },
              { label: '5', ts: n5Ts, price: n5Price, above: newDir2 > 0 },
            ],
            { ts: n2Ts + (n5Ts - n2Ts) / 2, price: isUp ? cPrice - atr * 0.5 : cPrice + atr * 0.5, text: 'Projected Impulse', sub: '' },
            projColor2, cColor2, projDashDotted,
          )

          // ── Draw the projected channel (two parallel trendlines) ──
          const chColor = 'rgba(255,215,0,0.25)'
          const chEndTs = n5Ts + newStep2 * barStep  // extend slightly past W5
          add({
            name: 'patternTrendline',
            points: [
              { timestamp: cTs, value: cPrice },
              { timestamp: chEndTs, value: barsAfterC > 5 ? cPrice + ((n2Price - cPrice) / Math.max(1, n2Ts - cTs)) * (chEndTs - cTs) : corrLine(chEndTs) },
            ],
            extendData: { color: chColor, lineWidth: 1.5, dash: [8, 6] },
          })
          add({
            name: 'patternTrendline',
            points: [
              { timestamp: n1Ts, value: n1Price },
              { timestamp: chEndTs, value: barsAfterC > 5 ? n1Price + ((n2Price - cPrice) / Math.max(1, n2Ts - cTs)) * (chEndTs - n1Ts) : motiveLine(chEndTs) },
            ],
            extendData: { color: chColor, lineWidth: 1.5, dash: [8, 6] },
          })

          } // end aRetrace >= 0.05
        } else {
          // ── Not enough real data — use pure Fibonacci projection ──
          const abcStep2 = Math.max(5, Math.floor(avgWaveBars * 0.5))
          aPrice = round2(impEnd + corrDir * impRange * 0.382)
          aTs = anchorTs + abcStep2 * barStep
          aIdx = impEndIdx + abcStep2
          bPrice = round2(aPrice - corrDir * impRange * 0.382 * 0.50)
          bTs = aTs + abcStep2 * barStep * 0.7
          bIdx = aIdx + Math.floor(abcStep2 * 0.7)
          cPrice = round2(impEnd + corrDir * impRange * 0.618)
          cTs = bTs + abcStep2 * barStep

          const projColor = IMPULSE_CLR
          const cColor = IMPULSE_CIRCLE

          drawProjection(
            [{ ts: anchorTs, price: impEnd }, { ts: aTs, price: aPrice }, { ts: bTs, price: bPrice }, { ts: cTs, price: cPrice }],
            [
              { label: 'A', ts: aTs, price: aPrice, above: corrDir < 0 },
              { label: 'B', ts: bTs, price: bPrice, above: corrDir > 0 },
              { label: 'C', ts: cTs, price: cPrice, above: corrDir < 0 },
            ],
            { ts: aTs + (cTs - aTs) / 2, price: isUp ? Math.min(aPrice, cPrice) - atr * 0.7 : Math.max(aPrice, cPrice) + atr * 0.7, text: 'Corrective Wave', sub: isUp ? '↘ Projected' : '↗ Projected' },
            projColor, cColor,
          )
          // ── CHANNEL-FIRST: define channel, waves ON it (Fib branch) ──
          const newDir = isUp ? 1 : -1
          const corrRange = safeRange(cPrice, Math.abs(impEnd - cPrice), newDir)
          const newStepBars = Math.max(8, Math.floor(avgWaveBars * 0.6))

          // Channel: corrective line from C slopes toward impulse end, motive line parallel
          const totalMsF = newStepBars * 4.4 * barStep
          const chEndF = cPrice + newDir * corrRange * 0.764
          const chSlopeF = (chEndF - cPrice) / totalMsF
          const chWidthF = corrRange * 0.35
          const corrLineF = (ts: number) => cPrice + chSlopeF * (ts - cTs)
          const motiveLineF = (ts: number) => corrLineF(ts) + newDir * chWidthF

          const n1Ts = cTs + newStepBars * barStep
          const n1Price = round2(motiveLineF(n1Ts))
          const n2Ts = n1Ts + Math.floor(newStepBars * 0.5) * barStep
          const n2Price = round2(corrLineF(n2Ts))
          const n3Ts = n2Ts + Math.floor(newStepBars * 1.5) * barStep
          const n3Price = round2(motiveLineF(n3Ts))
          const n4Ts = n3Ts + Math.floor(newStepBars * 0.4) * barStep
          const n4Price = round2(corrLineF(n4Ts))
          const n5Ts = n4Ts + newStepBars * barStep
          const n5Price = round2(motiveLineF(n5Ts))

          drawProjection(
            [
              { ts: cTs, price: cPrice },
              { ts: n1Ts, price: n1Price },
              { ts: n2Ts, price: n2Price },
              { ts: n3Ts, price: n3Price },
              { ts: n4Ts, price: n4Price },
              { ts: n5Ts, price: n5Price },
            ],
            [
              { label: '1', ts: n1Ts, price: n1Price, above: newDir > 0 },
              { label: '2', ts: n2Ts, price: n2Price, above: newDir < 0 },
              { label: '3', ts: n3Ts, price: n3Price, above: newDir > 0 },
              { label: '4', ts: n4Ts, price: n4Price, above: newDir < 0 },
              { label: '5', ts: n5Ts, price: n5Price, above: newDir > 0 },
            ],
            { ts: n2Ts + (n5Ts - n2Ts) / 2, price: isUp ? cPrice - atr * 0.5 : cPrice + atr * 0.5, text: 'Projected Impulse', sub: '' },
            projColor, cColor, projDashDotted,
          )

          // ── Draw the projected channel ──
          const chColorF = 'rgba(255,215,0,0.25)'
          const chEndTsF = n5Ts + newStepBars * barStep
          add({
            name: 'patternTrendline',
            points: [
              { timestamp: cTs, value: cPrice },
              { timestamp: chEndTsF, value: corrLineF(chEndTsF) },
            ],
            extendData: { color: chColorF, lineWidth: 1.5, dash: [8, 6] },
          })
          add({
            name: 'patternTrendline',
            points: [
              { timestamp: n1Ts, value: n1Price },
              { timestamp: n5Ts, value: n5Price },
            ],
            extendData: { color: chColorF, lineWidth: 1.5, dash: [8, 6] },
          })
        }
      } // end Case 2

      // ── Case 2b: COMPLETE IMPULSE + LINKED CORRECTION → project 1-2-3-4-5 + ABC ──
      // When the backend provides both impulse and correction, project the next
      // impulse continuation from the correction end, then a second ABC.
      if (isImpulse && isComplete && hasLinkedCorr) {
        const corrEndPrice = anchorPrice
        const corrStartPrice = linkedCorrection!.waves[0].start_price
        const newDir = isUp ? 1 : -1
        const corrRange = safeRange(corrEndPrice, Math.abs(corrStartPrice - corrEndPrice), newDir)
        const avgWaveBars2b = primaryCount.waves.reduce((s: number, w: any) => s + (w.end_idx - w.start_idx), 0) / primaryCount.waves.length
        const newStep = Math.max(8, Math.floor(avgWaveBars2b * 0.6))
        const corrEndIdx = linkedCorrection!.waves[linkedCorrection!.waves.length - 1].end_idx
        const barsAfterCorr = serverBars.length - 1 - corrEndIdx
        const maxAllowedBarsAfterCorr = Math.max(15, Math.floor(serverBars.length * 0.15))
        if (barsAfterCorr > maxAllowedBarsAfterCorr) return

        let n1P: number, n1T: number, n2P: number, n2T: number
        let n3P: number, n3T: number, n4P: number, n4T: number
        let n5P: number, n5T: number

        // Channel-first: define channel from correction, waves on boundaries
        const totalMs2b = newStep * 4.4 * barStep
        const chEnd2b = corrEndPrice + newDir * corrRange * 0.764
        const chSlope2b = (chEnd2b - corrEndPrice) / totalMs2b
        const chWidth2b = corrRange * 0.35
        const corrLine2b = (ts: number) => corrEndPrice + chSlope2b * (ts - anchorTs)
        const motiveLine2b = (ts: number) => corrLine2b(ts) + newDir * chWidth2b

        if (barsAfterCorr > 5) {
          // W1/W2 from real candle body, then channel for W3-W5
          const w1s = findSwing(corrEndIdx + 1, serverBars.length - 1, isUp)
          n1P = w1s.price; n1T = w1s.ts
          if (serverBars.length - 1 - w1s.idx >= 3) {
            const w2s = findSwing(w1s.idx + 1, serverBars.length - 1, !isUp)
            n2P = w2s.price; n2T = w2s.ts
          } else {
            n2P = round2(corrLine2b(anchorTs + newStep * 1.5 * barStep))
            n2T = n1T + Math.floor(newStep * 0.5) * barStep
          }
          // Channel from real W1/W2
          const realSlope2b = (n2P - corrEndPrice) / Math.max(1, n2T - anchorTs)
          const realCorr2b = (ts: number) => corrEndPrice + realSlope2b * (ts - anchorTs)
          const realChW2b = Math.abs(n1P - realCorr2b(n1T))
          const realMotive2b = (ts: number) => realCorr2b(ts) + newDir * realChW2b
          n3T = n2T + Math.floor(newStep * 1.5) * barStep; n3P = round2(realMotive2b(n3T))
          n4T = n3T + Math.floor(newStep * 0.4) * barStep; n4P = round2(realCorr2b(n4T))
          n5T = n4T + newStep * barStep; n5P = round2(realMotive2b(n5T))
        } else {
          // Pure channel projection
          const t1 = anchorTs + newStep * barStep
          n1T = t1; n1P = round2(motiveLine2b(t1))
          const t2 = t1 + Math.floor(newStep * 0.5) * barStep
          n2T = t2; n2P = round2(corrLine2b(t2))
          const t3 = t2 + Math.floor(newStep * 1.5) * barStep
          n3T = t3; n3P = round2(motiveLine2b(t3))
          const t4 = t3 + Math.floor(newStep * 0.4) * barStep
          n4T = t4; n4P = round2(corrLine2b(t4))
          const t5 = t4 + newStep * barStep
          n5T = t5; n5P = round2(motiveLine2b(t5))
        }

        const projColor = IMPULSE_CLR
        const cColor = IMPULSE_CIRCLE

        drawProjection(
          [
            { ts: anchorTs, price: corrEndPrice },
            { ts: n1T, price: n1P },
            { ts: n2T, price: n2P },
            { ts: n3T, price: n3P },
            { ts: n4T, price: n4P },
            { ts: n5T, price: n5P },
          ],
          [
            { label: '1', ts: n1T, price: n1P, above: newDir > 0 },
            { label: '2', ts: n2T, price: n2P, above: newDir < 0 },
            { label: '3', ts: n3T, price: n3P, above: newDir > 0 },
            { label: '4', ts: n4T, price: n4P, above: newDir < 0 },
            { label: '5', ts: n5T, price: n5P, above: newDir > 0 },
          ],
          { ts: n2T + (n5T - n2T) / 2, price: newDir > 0 ? corrEndPrice - atr * 0.5 : corrEndPrice + atr * 0.5, text: 'Projected Impulse', sub: '' },
          projColor, cColor, projDashDotted,
        )

        // ── Projected channel (Case 2b) ──
        const chColor2bLine = 'rgba(255,215,0,0.25)'
        const chEndTs2b = n5T + newStep * barStep
        add({
          name: 'patternTrendline',
          points: [
            { timestamp: anchorTs, value: corrEndPrice },
            { timestamp: chEndTs2b, value: barsAfterCorr > 5 ? corrEndPrice + ((n2P - corrEndPrice) / Math.max(1, n2T - anchorTs)) * (chEndTs2b - anchorTs) : corrLine2b(chEndTs2b) },
          ],
          extendData: { color: chColor2bLine, lineWidth: 1.5, dash: [8, 6] },
        })
        add({
          name: 'patternTrendline',
          points: [
            { timestamp: n1T, value: n1P },
            { timestamp: chEndTs2b, value: barsAfterCorr > 5 ? n1P + ((n2P - corrEndPrice) / Math.max(1, n2T - anchorTs)) * (chEndTs2b - n1T) : motiveLine2b(chEndTs2b) },
          ],
          extendData: { color: chColor2bLine, lineWidth: 1.5, dash: [8, 6] },
        })
      } // end Case 2b

      // ── Case 3: COMPLETE CORRECTION, no linked impulse → project 12345 ──
      if (!isImpulse && !hasLinkedCorr) {
        const newDir = isUp ? -1 : 1
        const corrRange = safeRange(anchorPrice, Math.abs(primaryCount.waves[0].start_price - anchorPrice), newDir)

        // Channel-first (Case 3)
        const avgWaveBars3 = primaryCount.waves.reduce((s: number, w: any) => s + (w.end_idx - w.start_idx), 0) / primaryCount.waves.length
        const step3 = Math.max(8, Math.floor(avgWaveBars3 * 0.6))
        const totalMs3 = step3 * 4.4 * barStep
        const chEnd3 = anchorPrice + newDir * corrRange * 0.764
        const chSlope3 = (chEnd3 - anchorPrice) / totalMs3
        const chWidth3 = corrRange * 0.35
        const corrLine3 = (ts: number) => anchorPrice + chSlope3 * (ts - anchorTs)
        const motiveLine3 = (ts: number) => corrLine3(ts) + newDir * chWidth3

        const w1T = anchorTs + step3 * barStep
        const w1P = round2(motiveLine3(w1T))
        const w2T = w1T + Math.floor(step3 * 0.5) * barStep
        const w2P = round2(corrLine3(w2T))
        const w3T = w2T + Math.floor(step3 * 1.5) * barStep
        const w3P = round2(motiveLine3(w3T))
        const w4T = w3T + Math.floor(step3 * 0.4) * barStep
        const w4P = round2(corrLine3(w4T))
        const w5T = w4T + step3 * barStep
        const w5P = round2(motiveLine3(w5T))

        const projColor = IMPULSE_CLR
        const cColor = IMPULSE_CIRCLE

        drawProjection(
          [{ ts: anchorTs, price: anchorPrice }, { ts: w1T, price: w1P }, { ts: w2T, price: w2P }, { ts: w3T, price: w3P }, { ts: w4T, price: w4P }, { ts: w5T, price: w5P }],
          [
            { label: '1', ts: w1T, price: w1P, above: newDir > 0 },
            { label: '2', ts: w2T, price: w2P, above: newDir < 0 },
            { label: '3', ts: w3T, price: w3P, above: newDir > 0 },
            { label: '4', ts: w4T, price: w4P, above: newDir < 0 },
            { label: '5', ts: w5T, price: w5P, above: newDir > 0 },
          ],
          { ts: w1T + (w5T - w1T) / 2, price: newDir > 0 ? Math.min(anchorPrice, w2P) - atr * 0.7 : Math.max(anchorPrice, w2P) + atr * 0.7, text: 'Impulsive Wave', sub: newDir > 0 ? '↗ Projected 1-2-3-4-5' : '↘ Projected 1-2-3-4-5' },
          projColor, cColor, projDashSolid,
        )

        // ── Projected channel (Case 3) ──
        const chColor3 = 'rgba(255,215,0,0.25)'
        const chEndTs3 = w5T + step3 * barStep
        add({
          name: 'patternTrendline',
          points: [
            { timestamp: anchorTs, value: anchorPrice },
            { timestamp: chEndTs3, value: corrLine3(chEndTs3) },
          ],
          extendData: { color: chColor3, lineWidth: 1.5, dash: [8, 6] },
        })
        add({
          name: 'patternTrendline',
          points: [
            { timestamp: anchorTs, value: anchorPrice + newDir * chWidth3 },
            { timestamp: chEndTs3, value: motiveLine3(chEndTs3) },
          ],
          extendData: { color: chColor3, lineWidth: 1.5, dash: [8, 6] },
        })
      }
    }

    // ── EW CHANNELS — trend channels containing the wave structure ──────────
    const ewChannels = (resp as any).channels as EWChannel[] | undefined
    if (ewChannels && ewChannels.length > 0 && serverBars.length > 0) {
      const CHANNEL_COLORS: Record<string, { line: string; fill: string }> = {
        base:         { line: 'rgba(100,150,255,0.35)', fill: 'rgba(100,150,255,0.04)' },
        acceleration: { line: 'rgba(245,158,11,0.45)',  fill: 'rgba(245,158,11,0.05)' },
        corrective:   { line: 'rgba(239,68,68,0.40)',   fill: 'rgba(239,68,68,0.04)' },
      }

      for (const ch of ewChannels) {
        const colors = CHANNEL_COLORS[ch.type] || CHANNEL_COLORS.base

        // Convert bar indices to timestamps via serverBars
        const snapIdx = (idx: number) => {
          const clamped = Math.max(0, Math.min(idx, serverBars.length - 1))
          return serverBars[clamped]?.timestamp || lastTs
        }

        const lowerStartTs = snapIdx(ch.lower.start_idx)
        const lowerEndTs = ch.lower.end_idx >= serverBars.length
          ? lastTs + (ch.lower.end_idx - serverBars.length + 1) * barStep
          : snapIdx(ch.lower.end_idx)
        const upperStartTs = snapIdx(ch.upper.start_idx)
        const upperEndTs = ch.upper.end_idx >= serverBars.length
          ? lastTs + (ch.upper.end_idx - serverBars.length + 1) * barStep
          : snapIdx(ch.upper.end_idx)

        // Lower channel line
        add({
          name: 'patternTrendline',
          points: [
            { timestamp: lowerStartTs, value: ch.lower.start_price },
            { timestamp: lowerEndTs,   value: ch.lower.end_price },
          ],
          extendData: { color: colors.line, lineWidth: 1.5, dash: [6, 4] },
        })

        // Upper channel line
        add({
          name: 'patternTrendline',
          points: [
            { timestamp: upperStartTs, value: ch.upper.start_price },
            { timestamp: upperEndTs,   value: ch.upper.end_price },
          ],
          extendData: { color: colors.line, lineWidth: 1.5, dash: [6, 4] },
        })

        // Channel type label
        const labelMap: Record<string, string> = {
          base: 'Base Channel',
          acceleration: '2-4 Channel',
          corrective: 'Corrective Channel',
        }
        const labelTs = lowerStartTs + Math.floor((lowerEndTs - lowerStartTs) * 0.1)
        const labelPrice = ch.direction === 'up' ? ch.lower.start_price : ch.upper.start_price
        add({
          name: 'patternLabel',
          points: [{ timestamp: labelTs, value: labelPrice }],
          extendData: { text: labelMap[ch.type] || ch.type, color: colors.line, fontSize: 9, offsetY: ch.direction === 'up' ? 12 : -12 },
        })

        // Breakout marker
        if (ch.breakout) {
          const bkTs = ch.breakout.idx >= serverBars.length
            ? lastTs + (ch.breakout.idx - serverBars.length + 1) * barStep
            : snapIdx(ch.breakout.idx)

          add({
            name: 'patternBadge',
            points: [{ timestamp: bkTs, value: ch.breakout.price }],
            extendData: {
              text: ch.breakout.label,
              bgColor: ch.breakout.new_phase === 'corrective'
                ? 'rgba(239,68,68,0.85)'
                : 'rgba(34,197,94,0.85)',
              textColor: '#ffffff',
            },
          })
        }
      }
    }

    // ── STATUS BADGES — derived from the DRAWN wave count, not current_position ──
    if (layers.status && primaryCount && primaryCount.waves && primaryCount.waves.length > 0) {
      const badgeTs = lastTs + barStep * 8
      const allBarPrices = bars.map(b => b.high)
      const highPrice = Math.max(...allBarPrices.slice(-Math.min(80, allBarPrices.length)))
      const drawnIsUp = primaryCount.direction === 'up'
      const drawnTrend = drawnIsUp ? 'BULLISH' : 'BEARISH'
      const accentColor = drawnIsUp ? theme.impulse : theme.corrective

      const phaseIcon = drawnIsUp ? '↗' : '↘'
      const line1 = `${phaseIcon} ${drawnTrend} IMPULSE`
      add({
        name: 'patternBadge',
        points: [{ timestamp: badgeTs, value: highPrice + atr * 1.2 }],
        extendData: { text: line1, bgColor: 'rgba(10,15,30,0.88)', textColor: accentColor },
      })

      // Action badge from current_position if available, otherwise derive from wave count
      const numW = primaryCount.waves.length
      let actionLabel = 'WATCH'
      if (currentPos && currentPos.action) {
        actionLabel = currentPos.action.split(' — ')[0] || 'WATCH'
      } else if (numW >= 4) {
        actionLabel = 'SETUP'
      } else if (numW === 3) {
        actionLabel = 'WAIT'
      }
      const actionColor = actionLabel === 'SETUP' ? theme.impulse
        : actionLabel === 'WATCH' ? '#fbbf24'
        : actionLabel === 'WAIT' ? '#fb923c'
        : actionLabel === 'REVERSE' ? theme.corrective : 'rgba(148,163,184,0.7)'
      add({
        name: 'patternBadge',
        points: [{ timestamp: badgeTs, value: highPrice + atr * 0.4 }],
        extendData: { text: actionLabel, bgColor: 'rgba(10,15,30,0.85)', textColor: actionColor },
      })
    }

    // ── DRAGGABLE HANDLES: W1 start, W1 end, and last wave's end ──
    // Allow user to manually adjust the key pivots. On drag end, the new
    // point is saved to ewAdjustmentsRef and the overlays are re-rendered.
    if (primaryCount && primaryCount.waves && primaryCount.waves.length > 0) {
      const waves = primaryCount.waves
      const w1 = waves[0]
      const wLast = waves[waves.length - 1]
      const isUp = primaryCount.direction === 'up'

      type HandleKind = 'w1Start' | 'w1End' | 'wLastEnd'
      const makeHandle = (kind: HandleKind, idx: number, price: number, label: string) => {
        const ts = snap(idx)
        chart.createOverlay({
          name: 'waveHandle',
          groupId: 'auto-overlays',
          points: [{ timestamp: ts, value: price }],
          lock: false,
          mode: 'normal',
          needDefaultPointFigure: false,
          needDefaultXAxisFigure: false,
          needDefaultYAxisFigure: false,
          extendData: { color: '#fde047', label },
          onPressedMoveEnd: (e: any) => {
            try {
              const pt = e?.overlay?.points?.[0] || e?.points?.[0]
              if (!pt) return
              const newTs = pt.timestamp
              const newPrice = pt.value
              const newIdx = nearestBarIdx(barTs, newTs)
              const key = adjustmentKey(symbol, tfKey)
              const prev = ewAdjustmentsRef[key] || {}
              ewAdjustmentsRef[key] = {
                ...prev,
                [kind]: { idx: newIdx, price: newPrice },
              }
              if (ewRedrawTrigger) ewRedrawTrigger()
            } catch (err) {
              console.warn('[EW handle drag] failed:', err)
            }
            return false
          },
          // Right-click → revert this handle to its backend-detected position.
          // Deletes just this kind's entry from the adjustment ref so other
          // dragged handles keep their overrides.
          onRightClick: (_e: any) => {
            try {
              const key = adjustmentKey(symbol, tfKey)
              const prev = ewAdjustmentsRef[key]
              if (prev && prev[kind]) {
                const { [kind]: _removed, ...rest } = prev
                if (Object.keys(rest).length === 0) delete ewAdjustmentsRef[key]
                else ewAdjustmentsRef[key] = rest
                if (ewRedrawTrigger) ewRedrawTrigger()
              }
            } catch (err) {
              console.warn('[EW handle undo] failed:', err)
            }
            return true  // swallow the event (prevent default context menu)
          },
        } as any)
      }

      makeHandle('w1Start', w1.start_idx, w1.start_price, 'W1 start')
      makeHandle('w1End',   w1.end_idx,   w1.end_price,   'W1 end')
      makeHandle('wLastEnd', wLast.end_idx, wLast.end_price, `W${waves.length} end`)
      void isUp // silence unused-var lint in case isUp isn't referenced below
    }

    // ── AUTO-ZOOM: frame the EW structure vertically + horizontally ──
    // Collect all wave pivots (prices + bar indices) from the primary count,
    // the linked correction, and any projected continuation. Compute a bounding
    // box and force both the x-viewport and y-axis range to fit it. This fixes
    // cases like MNTS (MA200 at $4032 dominating the y-axis) where the EW
    // structure ends up crammed at the bottom of the pane.
    if (primaryCount && primaryCount.waves && primaryCount.waves.length > 0) {
      try {
        const collectPivots = (wc: WaveCountData | null | undefined): { idx: number; price: number }[] => {
          if (!wc || !wc.waves || !wc.waves.length) return []
          const out: { idx: number; price: number }[] = []
          out.push({ idx: wc.waves[0].start_idx, price: wc.waves[0].start_price })
          for (const w of wc.waves) out.push({ idx: w.end_idx, price: w.end_price })
          return out
        }

        const allPivots = [
          ...collectPivots(primaryCount),
          ...collectPivots(linkedCorrection),
        ]

        if (allPivots.length >= 2) {
          let pMin = Infinity, pMax = -Infinity
          let iMin = Infinity, iMax = -Infinity
          for (const p of allPivots) {
            if (p.price < pMin) pMin = p.price
            if (p.price > pMax) pMax = p.price
            if (p.idx < iMin) iMin = p.idx
            if (p.idx > iMax) iMax = p.idx
          }

          // Extend the y-range to also cover the candle prices that will
          // be visible horizontally (from iMin through the last bar).
          // Otherwise candles to the right of the last EW pivot can be
          // outside the framed y-range on initial render.
          const scanFrom = Math.max(iMin, 0)
          const scanTo = serverBars.length - 1
          for (let i = scanFrom; i <= scanTo; i++) {
            const b = serverBars[i]
            if (!b) continue
            if (b.low  < pMin) pMin = b.low
            if (b.high > pMax) pMax = b.high
          }

          // Vertical padding: 8% of price range on top/bottom
          const pRange = pMax - pMin
          const vPad = Math.max(pRange * 0.08, atr * 0.5)
          const yMin = pMin - vPad
          const yMax = pMax + vPad

          // Override the candle pane y-axis to frame the EW structure on
          // first render, then release the override ~600ms later so any
          // subsequent user scroll (zoom in OR zoom out) gets proper
          // klinecharts auto-rescale. Zooming in shrinks the y-axis to
          // visible candles; zooming out expands it, eventually showing
          // the whole chart with full vertical range.
          try {
            chart.setPaneOptions({
              id: 'candle_pane',
              axis: {
                name: 'normal',
                createRange: ({ defaultRange }: any) => {
                  const from = yMin
                  const to = yMax
                  const range = to - from
                  return {
                    ...defaultRange,
                    from, to, range,
                    realFrom: from, realTo: to, realRange: range,
                    displayFrom: from, displayTo: to, displayRange: range,
                  }
                },
              } as any,
            })
            // Release the y-axis override after the initial EW-framed
            // composition has been painted. From then on, klinecharts'
            // built-in visible-bars auto-scale takes over for both zoom
            // directions.
            setTimeout(() => {
              try {
                chart.setPaneOptions({
                  id: 'candle_pane',
                  axis: {
                    name: 'normal',
                    createRange: ({ defaultRange }: any) => defaultRange,
                  } as any,
                })
              } catch (e) {
                console.warn('[EW auto-zoom] release override failed:', e)
              }
            }, 600)
          } catch (e) {
            console.warn('[EW auto-zoom] setPaneOptions failed:', e)
          }

          // Horizontal: anchor the RIGHT edge to the last candle, then
          // size the viewport so EVERY EW pivot is on-screen — including
          // the earliest (iMin). Padding goes on the LEFT so pivots
          // aren't pressed against the left edge; the right edge hugs
          // the last bar (current price) with minimal padding.
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
        }
      } catch (e) {
        console.warn('[EW auto-zoom] failed:', e)
      }
    }
  } catch (e) {
    console.warn('Elliott Wave overlay failed:', e)
  } finally {
    ewLoadingCallback?.(false)
  }
}

// ── Chart Pattern overlay drawing ───────────────────────────────────────────
// Fetches detected chart patterns and draws trendlines + labels on chart

async function drawChartPatterns(chart: Chart, symbol: string, bars: KLineData[], tfKey: TF = '1D') {
  if (!bars.length) return
  try {
    const apiTf = TF_TO_API[tfKey] || ''
    const r = await api.get(`/patterns/${symbol}`, { params: { ...(apiTf ? { tf: apiTf } : {}) } })
    let patterns = r.data || []
    if (!patterns.length) return

    // Only show top 3 patterns by confidence — avoids label clutter
    patterns = patterns
      .filter((p: any) => (p.confidence ?? 0) >= 60)
      .sort((a: any, b: any) => (b.confidence ?? 0) - (a.confidence ?? 0))
      .slice(0, 3)

    const barTs = bars.map(b => b.timestamp)
    const lastTs = barTs[barTs.length - 1]
    const barStep = barTs.length > 1 ? barTs[barTs.length - 1] - barTs[barTs.length - 2] : 86400000

    const snap = (idx: number): number => {
      if (idx < 0) idx = 0
      if (idx < barTs.length) return barTs[idx]
      return lastTs + (idx - barTs.length + 1) * barStep
    }
    const add = (config: any) => chart.createOverlay({ ...config, groupId: 'auto-overlays', lock: true, mode: 'normal', needDefaultPointFigure: false, needDefaultXAxisFigure: false, needDefaultYAxisFigure: false, onSelected: () => false, onMouseEnter: () => false, onClick: () => false, onRightClick: () => false, styles: { point: { activeColor: 'transparent', activeBorderColor: 'transparent', activeBorderSize: 0, activeRadius: 0, color: 'transparent', borderColor: 'transparent', borderSize: 0, radius: 0 } } })

    for (const pat of patterns) {
      const pts: [number, number][] = pat.points || []
      if (pts.length < 2) continue

      const isBullish = pat.direction === 'bullish'
      const lineColor = 'rgba(148,163,184,0.30)' // neutral gray — doesn't compete with wave colors

      // Subtle trendlines only — thin, semi-transparent
      for (let i = 0; i < pts.length - 1; i++) {
        const [idx1, price1] = pts[i]
        const [idx2, price2] = pts[i + 1]
        add({
          name: 'patternTrendline',
          points: [
            { timestamp: snap(idx1), value: price1 },
            { timestamp: snap(idx2), value: price2 },
          ],
          extendData: { color: lineColor, lineWidth: 1, dash: [] },
        })
      }

      // Pattern name labels removed — user found them cluttery
    }
  } catch (e) {
    console.warn('Chart pattern overlay failed:', e)
  }
}

// ── Chart style config (dark theme, StockCharts.com-like) ────────────────────

const CHART_STYLES = {
  grid: {
    show: true,
    horizontal: { show: true, size: 1, color: 'rgba(255,255,255,0.08)', style: 'solid' as const, dashedValue: [0, 0] },
    vertical:   { show: true, size: 1, color: 'rgba(255,255,255,0.08)', style: 'solid' as const, dashedValue: [0, 0] },
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
      last: {
        show: true,
        upColor: '#22c55e', downColor: '#ef4444', noChangeColor: '#6b7280',
        line: { show: true, style: 'dashed' as const, dashedValue: [4, 4], size: 1 },
        text: { show: true, style: 'fill' as const, color: '#fff', size: 10, family: 'monospace', weight: 'bold',
                paddingLeft: 4, paddingRight: 4, paddingTop: 2, paddingBottom: 2,
                borderStyle: 'solid' as const, borderSize: 0, borderColor: 'transparent',
                borderDashedValue: [] as number[], borderRadius: 2 },
      },
    },
    tooltip: { showRule: 'always' as const, showType: 'standard' as const },
  },
  indicator: {
    tooltip: { showRule: 'always' as const, showType: 'standard' as const },
  },
  xAxis: {
    show: true,
    tickText: { color: '#9ca3af', size: 10, family: 'monospace', weight: 'normal' },
  },
  yAxis: {
    show: true,
    tickText: { color: '#9ca3af', size: 10, family: 'monospace', weight: 'normal' },
  },
  crosshair: {
    show: true,
    horizontal: {
      show: true,
      line: { show: true, style: 'dashed' as const, dashedValue: [4, 4], size: 1, color: 'rgba(255,255,255,0.2)' },
      text: { show: true, style: 'fill' as const, color: '#fff', size: 10, family: 'monospace', weight: 'normal',
              paddingLeft: 4, paddingRight: 4, paddingTop: 2, paddingBottom: 2,
              borderStyle: 'solid' as const, borderSize: 1, borderColor: '#374151', borderDashedValue: [] as number[],
              borderRadius: 2, backgroundColor: '#1f2937' },
    },
    vertical: {
      show: true,
      line: { show: true, style: 'dashed' as const, dashedValue: [4, 4], size: 1, color: 'rgba(255,255,255,0.2)' },
      text: { show: true, style: 'fill' as const, color: '#fff', size: 10, family: 'monospace', weight: 'normal',
              paddingLeft: 4, paddingRight: 4, paddingTop: 2, paddingBottom: 2,
              borderStyle: 'solid' as const, borderSize: 1, borderColor: '#374151', borderDashedValue: [] as number[],
              borderRadius: 2, backgroundColor: '#1f2937' },
    },
  },
  separator: { size: 2, color: '#374151', fill: true, activeBackgroundColor: 'rgba(59,130,246,0.3)' },
  overlay: {
    point: {
      color: 'transparent',
      borderColor: 'transparent',
      borderSize: 0,
      radius: 0,
      activeColor: 'transparent',
      activeBorderColor: 'transparent',
      activeBorderSize: 0,
      activeRadius: 0,
    },
  },
}

// ── Component ──────────────────────────────────────────────────────────────────

export default function KLineChartOverlay({ symbol, onClose }: Props) {
  const chartContainerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<Chart | null>(null)
  const [tf, setTf] = useState<TF>('1D')
  const [refreshKey, setRefreshKey] = useState(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const indicatorsVisibleRef = useRef(true)
  const [maVisible, setMaVisible] = useState(true)
  const maVisibleRef = useRef(true)
  const [ewVisible, setEwVisible] = useState(true)
  const [chartPatternsVisible, setChartPatternsVisible] = useState(true)
  const [waveThemeIdx, setWaveThemeIdx] = useState(0) // default: Gold
  // EW layer toggles
  const [ewShowLabels, setEwShowLabels] = useState(true)
  const [ewShowTargets, setEwShowTargets] = useState(true)
  const [ewShowStatus, setEwShowStatus] = useState(true)
  const ewLayersRef = useRef({ labels: true, targets: true, status: true })
  const ewVisibleRef = useRef(true)
  const chartPatternsVisibleRef = useRef(true)
  const waveThemeRef = useRef(WAVE_THEMES[0])
  const loadedBarsRef = useRef<KLineData[]>([])
  const [overlayPanelOpen, setOverlayPanelOpen] = useState(false)
  const overlayPanelRef = useRef<HTMLDivElement | null>(null)
  // Drawing tools — track which tool is active; null = cursor/pan mode.
  // `groupId` marks user-drawn overlays so the Eraser can remove just those
  // (leaving EW/pattern overlays intact).
  const [activeDrawTool, setActiveDrawTool] = useState<string | null>(null)
  const [drawToolbarOpen, setDrawToolbarOpen] = useState(true)
  // EW loading state — mirrors drawElliottWaves's fetch lifecycle
  const [ewLoading, setEwLoading] = useState(false)
  const ewLoadStartRef = useRef<number>(0)
  const [ewLoadElapsed, setEwLoadElapsed] = useState(0)
  const DRAW_GROUP = 'user-drawings'
  const startDrawing = useCallback((name: string) => {
    const chart = chartRef.current
    if (!chart) return
    setActiveDrawTool(name)
    try {
      chart.createOverlay({
        name,
        groupId: DRAW_GROUP,
        // Leave points empty — klinecharts enters interactive draw mode
        // and waits for clicks to place each pivot
      } as any)
    } catch (e) {
      console.warn(`[drawing] failed to start ${name}:`, e)
    }
  }, [])
  const clearAllDrawings = useCallback(() => {
    const chart = chartRef.current
    if (!chart) return
    try {
      ;(chart as any).removeOverlay({ groupId: DRAW_GROUP })
    } catch (e) {
      console.warn('[drawing] clear failed:', e)
    }
  }, [])

  // Close overlay panel on outside click / Escape
  useEffect(() => {
    if (!overlayPanelOpen) return
    const onDocClick = (e: MouseEvent) => {
      if (overlayPanelRef.current && !overlayPanelRef.current.contains(e.target as Node)) {
        setOverlayPanelOpen(false)
      }
    }
    const onEsc = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOverlayPanelOpen(false)
    }
    document.addEventListener('mousedown', onDocClick)
    document.addEventListener('keydown', onEsc)
    return () => {
      document.removeEventListener('mousedown', onDocClick)
      document.removeEventListener('keydown', onEsc)
    }
  }, [overlayPanelOpen])

  const navigate = useNavigate()
  const price = useLivePrice(symbol)
  const companyName = useCompanyName(symbol)
  const currentTfRef = useRef<TF>(tf)
  currentTfRef.current = tf

  // Redraw all overlays (called after any toggle change)
  const redrawOverlays = useCallback(async () => {
    const chart = chartRef.current
    if (!chart) return
    const bars = loadedBarsRef.current
    const curTf = currentTfRef.current
    // Clear auto-generated overlays only — preserve user drawings
    try { (chart as any).removeOverlay({ groupId: 'auto-overlays' }) } catch {}
    // Redraw waves if visible
    if (ewVisibleRef.current) {
      await drawElliottWaves(chart, symbol, bars, ewLayersRef.current, waveThemeRef.current, curTf)
    }
    // Redraw chart patterns if visible
    if (chartPatternsVisibleRef.current) {
      await drawChartPatterns(chart, symbol, bars, curTf)
    }
  }, [symbol])

  // Register drag-end redraw trigger so drawElliottWaves' handle callbacks
  // can re-render after a user drags W1 / W5 pivots
  useEffect(() => {
    setEwRedrawTrigger(() => { redrawOverlays() })
    return () => setEwRedrawTrigger(null)
  }, [redrawOverlays])

  // Register EW loading callback — drawElliottWaves reports fetch start/end
  useEffect(() => {
    setEwLoadingCallback((loading) => {
      setEwLoading(loading)
      if (loading) {
        ewLoadStartRef.current = Date.now()
        setEwLoadElapsed(0)
      }
    })
    return () => setEwLoadingCallback(null)
  }, [])

  // Tick elapsed time while EW is loading (for the progress bar animation)
  useEffect(() => {
    if (!ewLoading) return
    const id = window.setInterval(() => {
      setEwLoadElapsed(Date.now() - ewLoadStartRef.current)
    }, 100)
    return () => window.clearInterval(id)
  }, [ewLoading])

  // ── Create chart + load data (recreate on symbol/tf change) ────────────
  useEffect(() => {
    if (!chartContainerRef.current) return

    // Dispose previous chart
    if (chartRef.current) {
      dispose(chartRef.current)
      chartRef.current = null
    }

    setLoading(true)
    setError(null)

    const chart = init(chartContainerRef.current, { styles: CHART_STYLES as any })
    if (!chart) return
    chartRef.current = chart

    // MA overlays on main candle pane (same Y-axis as candles)
    chart.createIndicator({ name: 'MA', calcParams: [20, 50, 200], shortName: 'MA' }, false, { id: 'candle_pane' })
    // Volume — small dedicated sub-pane (own Y-axis so it doesn't squish candles)
    chart.createIndicator({
      name: 'VOL',
      shortName: 'Vol',
    }, false, { id: 'vol_pane', height: 60 })
    // Slow Stochastic — KDJ(14,3,3) with professional label
    chart.createIndicator({
      name: 'KDJ',
      calcParams: [14, 3, 3],
      shortName: 'Slow STO %K(14) %D(3)',
      styles: {
        lines: [
          { color: '#ffffff', size: 1.5 },  // %K — white
          { color: '#ef4444', size: 1 },     // %D — red
          { color: 'transparent', size: 0 }, // J — hide
        ],
      },
    } as any, false, { id: 'stoch_pane', height: 100 })

    // v10 beta1: must set symbol + period + dataLoader for getBars to fire
    const tfKey = currentTfRef.current
    const periodMap: Record<string, { type: string; span: number }> = {
      '1m':  { type: 'minute', span: 1 },
      '5m':  { type: 'minute', span: 5 },
      '15m': { type: 'minute', span: 15 },
      '1h':  { type: 'hour',   span: 1 },
      '4h':  { type: 'hour',   span: 4 },
      '1D':  { type: 'day',    span: 1 },
      '1W':  { type: 'week',   span: 1 },
      '1M':  { type: 'month',  span: 1 },
    }

    // Order matters: dataLoader first (no-op until symbol+period set),
    // then symbol (no-op until period set), then period (triggers getBars).
    // initialLoadDone is a closure-local flag so it resets when the effect
    // re-runs (new chart instance per symbol/TF change). Using the module
    // ref would leak state across TF switches and block subsequent loads.
    let loadedBars: KLineData[] = []
    let initialLoadDone = false
    chart.setDataLoader({
      getBars: async (params: any) => {
        try {
          // klinecharts v10 may call getBars multiple times — once for the
          // initial load, again for backward pagination when the user scrolls
          // left. We fetch the full history upfront, so pagination calls get
          // an empty response with more=false, preventing the chart from
          // scrolling the view to the newly-loaded "older" bars.
          if (initialLoadDone) {
            params.callback([], false)
            return
          }
          initialLoadDone = true
          const data = await fetchData(symbol, currentTfRef.current)
          loadedBars = data
          loadedBarsRef.current = data
          params.callback(data, false)
          setLoading(false)
          // Show the most recent bars by default — without this klinecharts
          // pans to the start of the dataset, which now spans years.
          // Per-timeframe bar spacing so each view frames the recent action.
          requestAnimationFrame(() => {
            try {
              const tfBarSpace: Record<string, number> = {
                '1m': 6, '5m': 6, '15m': 6, '1h': 6, '4h': 6,
                '1D': 14, '1W': 10, '1M': 12,
              }
              const space = tfBarSpace[currentTfRef.current] || 8
              ;(chart as any).setBarSpace?.(space)
              ;(chart as any).setOffsetRightDistance?.(60)
              ;(chart as any).scrollToRealTime?.(0)
            } catch {}
          })
          // Clear any stray built-in drawings before adding our overlays
          try { chart.removeOverlay() } catch {}
          // Fetch and draw overlays after chart data loads
          const curTf = currentTfRef.current
          if (ewVisibleRef.current) {
            drawElliottWaves(chart, symbol, data, ewLayersRef.current, waveThemeRef.current, curTf, refreshKey > 0)
          }
          if (chartPatternsVisibleRef.current) {
            drawChartPatterns(chart, symbol, data, curTf)
          }
        } catch (e: any) {
          params.callback([], false)
          setError(e?.message || 'Failed to load data')
          setLoading(false)
        }
      },
    } as any)
    ;(chart as any).setSymbol({ ticker: symbol, pricePrecision: 2, volumePrecision: 0 })
    // setPeriod triggers the initial load (all 3 conditions met)
    ;(chart as any).setPeriod(periodMap[tfKey] || { type: 'day', span: 1 })

    return () => {
      if (chartRef.current) {
        dispose(chartRef.current)
        chartRef.current = null
      }
    }
  }, [symbol, tf, refreshKey])

  // ── Resize on window resize ────────────────────────────────────────────────
  useEffect(() => {
    const handleResize = () => chartRef.current?.resize()
    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [])

  // ── Double-click candle area to toggle indicators ────────────────────────
  useEffect(() => {
    const el = chartContainerRef.current
    if (!el) return
    const handleDblClick = () => {
      const chart = chartRef.current
      if (!chart) return
      const visible = indicatorsVisibleRef.current
      if (visible) {
        // Hide all indicators
        ;(chart as any).removeIndicator('candle_pane', 'MA')
        ;(chart as any).removeIndicator('vol_pane', 'VOL')
        ;(chart as any).removeIndicator('stoch_pane', 'KDJ')
        indicatorsVisibleRef.current = false
      } else {
        // Show indicators
        chart.createIndicator({ name: 'MA', calcParams: [20, 50, 200], shortName: 'MA' }, false, { id: 'candle_pane' })
        chart.createIndicator({ name: 'VOL', shortName: 'Vol' }, false, { id: 'vol_pane', height: 60 })
        chart.createIndicator({
          name: 'KDJ', calcParams: [14, 3, 3], shortName: 'Slow STO %K(14) %D(3)',
          styles: { lines: [{ color: '#ffffff', size: 1.5 }, { color: '#ef4444', size: 1 }, { color: 'transparent', size: 0 }] },
        } as any, false, { id: 'stoch_pane', height: 100 })
        indicatorsVisibleRef.current = true
      }
    }
    el.addEventListener('dblclick', handleDblClick)
    return () => el.removeEventListener('dblclick', handleDblClick)
  }, [])

  // ── Keyboard: ESC to close ────────────────────────────────────────────────
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [onClose])

  const openFullChart = useCallback(() => {
    navigate({ pathname: '/chart', search: `symbol=${symbol}` })
  }, [navigate, symbol])

  const changeUp = price && price.change >= 0

  return (
    <div className="fixed inset-0 z-50 bg-[#0a0a0f] flex flex-col">
      {/* ── Header bar ──────────────────────────────────────────────── */}
      <div className="flex items-center gap-3 px-4 py-2 bg-gray-900/90 border-b border-gray-800 flex-shrink-0">
        {/* Back button */}
        <button
          onClick={onClose}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-gray-800 hover:bg-gray-700 text-gray-300 hover:text-white text-sm font-medium transition-colors"
          title="Back to scan results (Esc)"
        >
          <span className="text-lg leading-none">←</span>
          <span>Back</span>
        </button>

        {/* Symbol + company name + price */}
        <div className="flex items-center gap-3">
          <span className="text-xl font-bold font-mono text-white">{symbol}</span>
          {companyName && <span className="text-sm text-gray-400 hidden lg:inline">{companyName}</span>}
          {price && (
            <>
              <span className="text-lg font-mono font-bold text-white">${price.last.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</span>
              <span className={`text-sm font-mono font-bold ${changeUp ? 'text-emerald-400' : 'text-red-400'}`}>
                {changeUp ? '+' : ''}{price.change.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })} ({changeUp ? '+' : ''}{price.changePct.toFixed(2)}%)
              </span>
              <span className="text-xs text-gray-500 hidden md:inline">
                O: {price.open.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })} · H: <span className="text-emerald-400">{price.high.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</span> · L: <span className="text-red-400">{price.low.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</span>
              </span>
            </>
          )}
        </div>

        {/* Timeframe pills */}
        <div className="flex items-center gap-0.5 ml-auto bg-gray-800/60 rounded-lg p-0.5">
          {(Object.keys(TF_CONFIG) as TF[]).map(t => (
            <button
              key={t}
              onClick={() => setTf(t)}
              className={`px-2.5 py-1 rounded-md text-xs font-medium transition-all ${
                tf === t
                  ? 'bg-blue-600 text-white shadow-lg shadow-blue-600/20'
                  : 'text-gray-400 hover:text-white hover:bg-gray-700'
              }`}
            >
              {TF_CONFIG[t].label}
            </button>
          ))}
        </div>

        {/* Refresh button */}
        <button
          onClick={() => setRefreshKey(k => k + 1)}
          className={`px-2.5 py-1.5 rounded-lg bg-gray-800 hover:bg-gray-700 text-gray-400 hover:text-white text-xs font-medium transition-colors border border-gray-700 ${loading ? 'opacity-50 pointer-events-none' : ''}`}
          title="Refresh chart data (re-fetch + redraw)"
        >
          <span className={`inline-block ${loading ? 'animate-spin' : ''}`}>⟳</span>
        </button>

        {/* Full chart button */}
        <button
          onClick={openFullChart}
          className="px-3 py-1.5 rounded-lg bg-gray-800 hover:bg-gray-700 text-gray-400 hover:text-white text-xs font-medium transition-colors border border-gray-700"
          title="Open full chart page with all indicators"
        >
          Full Chart
        </button>
      </div>

      {/* ── Chart area ──────────────────────────────────────────────── */}
      <div className="flex-1 relative min-h-0 overflow-hidden">
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center z-10 bg-[#0a0a0f]/60 pointer-events-none">
            <div className="flex items-center gap-2 text-gray-400 text-sm">
              <div className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
              Loading {symbol} {TF_CONFIG[tf].label}...
            </div>
          </div>
        )}
        {error && !loading && (
          <div className="absolute inset-0 flex items-center justify-center z-10">
            <div className="text-red-400 text-sm">{error}</div>
          </div>
        )}

        {/* ── EW loading indicator ────────────────────────────────── */}
        {ewLoading && (() => {
          // Progress model: EW API typically responds in 3–30s. We simulate
          // an asymptotic curve that approaches (but never reaches) 95% so
          // the bar keeps moving while waiting. Snaps to 100% on completion.
          const EXPECTED_MS = 15000
          const pct = Math.min(95, (1 - Math.exp(-ewLoadElapsed / EXPECTED_MS)) * 100)
          const seconds = (ewLoadElapsed / 1000).toFixed(1)
          return (
            <div className="absolute top-3 left-1/2 -translate-x-1/2 z-30 flex items-center gap-2 px-3 py-1.5 rounded-full bg-gray-900/95 border border-amber-500/40 backdrop-blur-sm shadow-lg shadow-black/40 select-none">
              <div className="w-3 h-3 border-2 border-amber-400 border-t-transparent rounded-full animate-spin" />
              <span className="text-[11px] font-medium text-amber-300">Detecting Elliott Waves</span>
              <div className="w-24 h-1 bg-gray-700/80 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-amber-500 to-amber-300 transition-all duration-150 ease-out"
                  style={{ width: `${pct}%` }}
                />
              </div>
              <span className="text-[10px] font-mono text-gray-400 tabular-nums">{seconds}s</span>
            </div>
          )
        })()}

        {/* ── Drawing Tools Toolbar (left side) ───────────────────── */}
        <div className="absolute top-3 left-3 z-20 flex flex-col gap-1 bg-gray-900/90 border border-gray-700/60 backdrop-blur-sm rounded-lg p-1 shadow-lg shadow-black/30 select-none">
          {/* Collapse/expand toggle */}
          <button
            onClick={() => setDrawToolbarOpen(o => !o)}
            title={drawToolbarOpen ? 'Collapse tools' : 'Expand tools'}
            className="w-8 h-6 flex items-center justify-center rounded text-gray-400 hover:text-white hover:bg-gray-800 transition-colors"
          >
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" className={`transition-transform ${drawToolbarOpen ? '' : 'rotate-180'}`}>
              <polyline points="15 18 9 12 15 6"/>
            </svg>
          </button>
          {drawToolbarOpen && <><div className="h-px bg-gray-700/60" />
          {[
            { id: null as string | null, label: 'Cursor', icon: (
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M13 13l6 6"/><path d="M3 3l7.7 19 2.3-8.3L22 10.3z"/></svg>
            )},
            { id: 'segment', label: 'Trend Line', icon: (
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><line x1="4" y1="20" x2="20" y2="4"/></svg>
            )},
            { id: 'rayLine', label: 'Ray', icon: (
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><line x1="4" y1="20" x2="22" y2="2"/><circle cx="4" cy="20" r="1.6" fill="currentColor"/></svg>
            )},
            { id: 'horizontalStraightLine', label: 'Horizontal', icon: (
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><line x1="2" y1="12" x2="22" y2="12"/></svg>
            )},
            { id: 'verticalStraightLine', label: 'Vertical', icon: (
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><line x1="12" y1="2" x2="12" y2="22"/></svg>
            )},
            { id: 'rect', label: 'Rectangle', icon: (
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="4" y="6" width="16" height="12" rx="1"/></svg>
            )},
            { id: 'circle', label: 'Circle', icon: (
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="8"/></svg>
            )},
            { id: 'parallelStraightLine', label: 'Parallel Channel', icon: (
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><line x1="3" y1="18" x2="20" y2="5"/><line x1="6" y1="21" x2="23" y2="8"/></svg>
            )},
            { id: 'fibonacciLine', label: 'Fib Retracement', icon: (
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"><line x1="3" y1="4" x2="21" y2="4"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="3" y1="14" x2="21" y2="14"/><line x1="3" y1="20" x2="21" y2="20"/></svg>
            )},
            { id: 'priceLine', label: 'Price Label', icon: (
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><line x1="2" y1="12" x2="16" y2="12"/><rect x="16" y="8" width="6" height="8" rx="1"/></svg>
            )},
            { id: 'simpleAnnotation', label: 'Note', icon: (
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M4 4h12l4 4v12H4z"/><line x1="8" y1="10" x2="16" y2="10"/><line x1="8" y1="14" x2="14" y2="14"/></svg>
            )},
          ].map(t => (
            <button
              key={t.id || 'cursor'}
              onClick={() => {
                if (t.id === null) setActiveDrawTool(null)
                else startDrawing(t.id)
              }}
              title={t.label}
              className={`w-8 h-8 flex items-center justify-center rounded transition-colors ${
                (activeDrawTool === null && t.id === null) || activeDrawTool === t.id
                  ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40'
                  : 'text-gray-400 hover:text-white hover:bg-gray-800'
              }`}
            >
              {t.icon}
            </button>
          ))}
          {/* Divider */}
          <div className="h-px bg-gray-700/60 my-0.5" />
          {/* Clear all drawings */}
          <button
            onClick={clearAllDrawings}
            title="Clear all drawings"
            className="w-8 h-8 flex items-center justify-center rounded text-gray-400 hover:text-red-400 hover:bg-red-500/10 transition-colors"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/><path d="M10 11v6"/><path d="M14 11v6"/></svg>
          </button>
          </>}
        </div>

        {/* ── Floating Overlay Controls ────────────────────────────── */}
        <div ref={overlayPanelRef} className="absolute top-3 right-3 z-20 select-none">
          {/* Toggle pill */}
          <button
            onClick={() => setOverlayPanelOpen(p => !p)}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-gray-900/90 border border-gray-700/60 backdrop-blur-sm text-[11px] font-medium text-gray-300 hover:text-white hover:border-gray-600 transition-all shadow-lg shadow-black/30"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="opacity-70">
              <path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"/>
            </svg>
            Overlays
            <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" className={`transition-transform ${overlayPanelOpen ? 'rotate-180' : ''}`}>
              <polyline points="6 9 12 15 18 9"/>
            </svg>
          </button>

          {/* Expanded panel */}
          {overlayPanelOpen && (
            <div className="mt-1.5 w-[260px] rounded-xl bg-gray-900/95 border border-gray-700/60 backdrop-blur-md shadow-2xl shadow-black/50 overflow-hidden">
              {/* Elliott Waves section */}
              <div className="p-2.5 border-b border-gray-800/60">
                <button
                  onClick={() => {
                    const next = !ewVisible
                    setEwVisible(next)
                    ewVisibleRef.current = next
                    redrawOverlays()
                  }}
                  className="flex items-center justify-between w-full group"
                >
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 rounded-full transition-colors" style={{ background: ewVisible ? WAVE_THEMES[waveThemeIdx].impulse : '#4b5563', boxShadow: ewVisible ? `0 0 6px ${WAVE_THEMES[waveThemeIdx].impulse}60` : 'none' }} />
                    <span className={`text-xs font-semibold tracking-wide ${ewVisible ? 'text-gray-200' : 'text-gray-500'}`}>
                      Elliott Waves
                    </span>
                  </div>
                  <div className={`text-[10px] px-1.5 py-0.5 rounded font-mono ${ewVisible ? 'bg-white/10 text-gray-300' : 'bg-gray-800 text-gray-600'}`}>
                    {ewVisible ? 'ON' : 'OFF'}
                  </div>
                </button>

                {/* Sub-toggles + color themes */}
                {ewVisible && (
                  <>
                    <div className="flex items-center gap-1 mt-2 ml-4">
                      {([
                        { key: 'labels' as const, icon: '①', label: 'Labels', state: ewShowLabels, set: setEwShowLabels },
                        { key: 'targets' as const, icon: '↗', label: 'Projections', state: ewShowTargets, set: setEwShowTargets },
                        { key: 'status' as const, icon: '▤', label: 'Status', state: ewShowStatus, set: setEwShowStatus },
                      ]).map(({ key, icon, label, state, set }) => (
                        <button
                          key={key}
                          onClick={() => {
                            const next = !state
                            set(next)
                            ewLayersRef.current = { ...ewLayersRef.current, [key]: next }
                            redrawOverlays()
                          }}
                          className={`flex items-center gap-1 px-2 py-1 rounded-md text-[10px] font-medium transition-all ${
                            state
                              ? 'bg-white/5 text-gray-300 border border-white/10'
                              : 'bg-gray-800/40 text-gray-600 border border-transparent hover:text-gray-400'
                          }`}
                          title={label}
                        >
                          <span className="text-[11px]">{icon}</span>
                          {label}
                        </button>
                      ))}
                    </div>
                    {/* Color theme swatches */}
                    <div className="flex items-center gap-1.5 mt-2 ml-4">
                      <span className="text-[9px] text-gray-600 mr-0.5">Color</span>
                      {WAVE_THEMES.map((t, i) => (
                        <button
                          key={t.id}
                          onClick={() => {
                            setWaveThemeIdx(i)
                            waveThemeRef.current = WAVE_THEMES[i]
                            redrawOverlays()
                          }}
                          className={`flex items-center gap-px rounded-full p-px transition-all ${
                            waveThemeIdx === i
                              ? 'ring-1 ring-white/40 ring-offset-1 ring-offset-gray-900'
                              : 'opacity-60 hover:opacity-100'
                          }`}
                          title={t.name}
                        >
                          <div className="w-3 h-3 rounded-full" style={{ background: t.swatch[0] }} />
                          <div className="w-3 h-3 rounded-full -ml-1" style={{ background: t.swatch[1] }} />
                        </button>
                      ))}
                    </div>
                  </>
                )}
              </div>

              {/* Chart Patterns section */}
              <div className="p-2.5 border-b border-gray-800/60">
                <button
                  onClick={() => {
                    const next = !chartPatternsVisible
                    setChartPatternsVisible(next)
                    chartPatternsVisibleRef.current = next
                    redrawOverlays()
                  }}
                  className="flex items-center justify-between w-full group"
                >
                  <div className="flex items-center gap-2">
                    <div className={`w-2 h-2 rounded-full transition-colors ${chartPatternsVisible ? 'bg-amber-400 shadow-[0_0_6px_rgba(245,158,11,0.5)]' : 'bg-gray-600'}`} />
                    <span className={`text-xs font-semibold tracking-wide ${chartPatternsVisible ? 'text-amber-300' : 'text-gray-500'}`}>
                      Chart Patterns
                    </span>
                  </div>
                  <div className={`text-[10px] px-1.5 py-0.5 rounded font-mono ${chartPatternsVisible ? 'bg-amber-500/15 text-amber-400' : 'bg-gray-800 text-gray-600'}`}>
                    {chartPatternsVisible ? 'ON' : 'OFF'}
                  </div>
                </button>
                <p className="text-[9px] text-gray-600 ml-4 mt-1">
                  H&S · Triangles · Wedges · Flags · Cups
                </p>
              </div>

              {/* Moving Averages toggle */}
              <div className="p-2.5">
                <button
                  onClick={() => {
                    const chart = chartRef.current
                    if (!chart) return
                    const next = !maVisible
                    setMaVisible(next)
                    maVisibleRef.current = next
                    if (next) {
                      chart.createIndicator({ name: 'MA', calcParams: [20, 50, 200], shortName: 'MA' }, false, { id: 'candle_pane' })
                    } else {
                      ;(chart as any).removeIndicator('candle_pane', 'MA')
                    }
                  }}
                  className="flex items-center justify-between w-full group"
                >
                  <div className="flex items-center gap-2">
                    <div className={`w-2 h-2 rounded-full transition-colors ${maVisible ? 'bg-blue-400 shadow-[0_0_6px_rgba(96,165,250,0.5)]' : 'bg-gray-600'}`} />
                    <span className={`text-xs font-semibold tracking-wide ${maVisible ? 'text-blue-300' : 'text-gray-500'}`}>
                      Moving Averages
                    </span>
                  </div>
                  <div className={`text-[10px] px-1.5 py-0.5 rounded font-mono ${maVisible ? 'bg-blue-500/15 text-blue-400' : 'bg-gray-800 text-gray-600'}`}>
                    {maVisible ? 'ON' : 'OFF'}
                  </div>
                </button>
                <p className="text-[9px] text-gray-600 ml-4 mt-1">
                  MA20 · MA50 · MA200
                </p>
              </div>
            </div>
          )}
        </div>

        <div
          ref={chartContainerRef}
          className="w-full h-full"
          style={{ background: '#0a0a0f' }}
        />
      </div>

      {/* ── Footer hint ──────────────────────────────────────────────── */}
      <div className="flex items-center justify-between px-4 py-1 bg-gray-900/60 border-t border-gray-800/50 text-[10px] text-gray-600 flex-shrink-0">
        <span>Scroll to zoom · Drag to pan · Double-click to toggle indicators · ESC to close</span>
        <span>MA(20/50/200) · Vol · Slow STO %K(14) %D(3) · Elliott Wave multi-degree</span>
      </div>
    </div>
  )
}
