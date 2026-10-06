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
  // New backend format (v2 — from _compute_projections)
  sequence: string    // "correction" | "impulse" | "in_progress"
  wave: string        // "A" | "B" | "C" | "1" | "2" | "3" | "4" | "5"
  label: string       // human-readable description
  price: number       // Fibonacci target price
  bar_offset: number  // bars from last confirmed wave end
  bar_abs: number     // absolute bar index (use with snap())
  confidence: number  // 0.0–1.0
  ratio: number | string  // Fib ratio used
  color: string       // "corrective" | "impulse"
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
  projections?: WaveProjection[]  // backend Fibonacci chain (v2 format)
  wave_type?: string
  rank?: number
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

// ── v3: Multi-degree display settings ─────────────────────────────────────
interface EWDisplaySettings {
  showProjections:  boolean
  showStartingPoint: boolean
  showWaveLines:    boolean
  showWaveLabels:   boolean
  visibleDegrees: {
    PRIMARY: boolean; INTERMEDIATE: boolean; MINOR: boolean
    MINUTE: boolean;  SUBMINUTE: boolean
  }
  degreeColors: {
    PRIMARY: string; INTERMEDIATE: string; MINOR: string
    MINUTE: string;  SUBMINUTE: string;   CYCLE: string; SUPERCYCLE: string
  }
  projectionStyle:    'solid' | 'dashed'
  activeZoneOpacity:  number
  waveLineOpacity:    number
}

const DEFAULT_EW_SETTINGS: EWDisplaySettings = {
  showProjections:  true,
  showStartingPoint: true,
  showWaveLines:    true,
  showWaveLabels:   true,
  visibleDegrees: {
    PRIMARY: true, INTERMEDIATE: true, MINOR: true,
    MINUTE: false, SUBMINUTE: false,
  },
  degreeColors: {
    SUPERCYCLE: '#ff6b6b', CYCLE: '#f97316', PRIMARY: '#fbbf24',
    INTERMEDIATE: '#22d3ee', MINOR: '#a78bfa',
    MINUTE: '#6b7280',      SUBMINUTE: '#4b5563',
  },
  projectionStyle:   'dashed',
  activeZoneOpacity: 0.10,
  waveLineOpacity:   0.55,
}

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

// v3 settings ref — module-level so drawElliottWaves (standalone fn) can read it
let ewV3Settings: EWDisplaySettings = DEFAULT_EW_SETTINGS
function setEwV3Settings(s: EWDisplaySettings) { ewV3Settings = s }
// v3 wave tree cache — stores the last fetched wave_tree so UI can render it
let ewV3WaveTreeCache: any = null
let ewV3WaveTreeCallback: ((wt: any) => void) | null = null
function setEwV3WaveTreeCallback(fn: ((wt: any) => void) | null) { ewV3WaveTreeCallback = fn }

// v3 hover system — populated on each draw, read by mousemove handler
interface V3SegMeta {
  startTs: number; endTs: number
  startPrice: number; endPrice: number
  midTs: number; midPrice: number
  label: string; degree: string
  waveType: string; direction: string
  color: string
  priceDelta: number; pricePct: number
  confidence?: number
}
let _v3Segments: V3SegMeta[] = []

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
    // Projection color: clearly distinct from detected waves — muted gray-blue.
    // Solid (not dotted) so it reads as a real trend line, not noise/decoration.
    const PROJ_CLR       = 'rgba(160,180,220,0.55)'
    const PROJ_CIRCLE    = 'rgba(100,130,180,0.70)'

    // ── Helper: draw a single wave count on chart ────────────────────
    const drawWaveCount = (
      wc: WaveCountData,
      opts: {
        lineColor: string; circleColor: string; circleText: string;
        correctiveLineColor: string; annotColor: string;
        lineWidth: number; isImpulsePhase: boolean;
        suppressLiveSegment?: boolean;   // true when a linked wave continues from the last pivot
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

      // Live "in-progress" segment intentionally removed.
      // Projections (drawProjection) handle what comes after the last detected
      // pivot in a distinct color so there is only ever ONE line from any pivot.

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
            const phaseLabel = wc.type === 'diagonal'
              ? (wc.wave_type === 'ending_diagonal' ? 'Ending Diagonal' : 'Leading Diagonal')
              : 'Impulse'
            const phaseText = opts.isImpulsePhase ? `${arrow} ${phaseLabel}` : `${arrow} Correction`
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
      const priceTol = (trueHighPrice - trueLowPrice) * 0.015  // 1.5% of range
      const isUp = wc.direction === 'up'
      const startTarget = isUp ? trueLowPrice  : trueHighPrice
      const endTarget   = isUp ? trueHighPrice : trueLowPrice
      return Math.abs(first.start_price - startTarget) <= priceTol
          && Math.abs(last.end_price   - endTarget)   <= priceTol
    }
    // Scan all wave counts for a complete 5-wave impulse anchored at both
    // extremes; prefer primary degree when multiple match.
    const anchoredCandidates = (waveCounts || []).filter((wc: any) =>
      (wc?.type === 'impulse' || wc?.type === 'diagonal') &&
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
    const _is5 = (wc: any) => (wc?.type === 'impulse' || wc?.type === 'diagonal') && wc?.is_complete && wc?.waves?.length === 5
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

              // Phase 2: scroll to current (last) candle AFTER relayout settles.
              // Bar space is already set to show the full wave — scrollToTimestamp
              // positions the most recent candle at the right edge so the user
              // lands on the current price with the whole pattern in view.
              setTimeout(() => {
                try {
                  chart.scrollToTimestamp(snap(lastBarIdx))
                } catch (e2) {
                  console.warn('[EW auto-zoom] scroll phase failed:', e2)
                }
              }, 50)
            } catch (e) {
              console.warn('[EW auto-zoom] horizontal zoom failed:', e)
            }
          }, 150)
        }
      } catch (e) {
        console.warn('[EW auto-zoom] failed:', e)
      }
    }

    // ── v3: Multi-degree wave tree rendering ─────────────────────────────
    // Renders when: v3 toggle is ON and backend returned wave_tree data.
    // Falls back gracefully — v2 rendering is unaffected.
    const waveTree = (resp as any)?.wave_tree
    ewV3WaveTreeCache = waveTree || null
    if (ewV3WaveTreeCallback) ewV3WaveTreeCallback(ewV3WaveTreeCache)
    if (waveTree?.primary_count) {
      const primaryNode = waveTree.primary_count

      // Helper: walk the tree and collect all pivot labels to draw
      interface V3Label {
        idx: number; price: number; label: string
        degree: string; color: string; isHigh: boolean
      }
      const v3Labels: V3Label[] = []

      function collectV3Labels(node: any) {
        if (!node?.sub_waves?.length) return
        const deg = (node.degree || '').toUpperCase() as keyof typeof ewV3Settings.degreeColors
        if (!ewV3Settings.visibleDegrees[deg as keyof typeof ewV3Settings.visibleDegrees]) {
          // Still recurse — children may be at a visible degree
          for (const sw of node.sub_waves) collectV3Labels(sw)
          return
        }
        const color = ewV3Settings.degreeColors[deg] || '#888888'
        for (const sw of node.sub_waves) {
          v3Labels.push({
            idx:    sw.end_idx,
            price:  sw.end_price,
            label:  sw.label,
            degree: deg,
            color,
            isHigh: sw.direction === 'UP',
          })
          if (sw.sub_waves?.length) collectV3Labels(sw)
        }
      }
      collectV3Labels(primaryNode)
      _v3Segments = []

      // Draw starting point marker
      if (ewV3Settings.showStartingPoint && waveTree.starting_point) {
        const sp = waveTree.starting_point
        add({
          name: 'waveCircle',
          points: [{ timestamp: snap(sp.idx), value: sp.price }],
          extendData: {
            text: 'o',
            fillColor: ewV3Settings.degreeColors.INTERMEDIATE + 'aa',
            textColor: '#ffffff',
            radius: 10,
            borderColor: ewV3Settings.degreeColors.INTERMEDIATE,
          },
        })
      }

      // Draw connecting lines between sub-waves at each visible degree
      if (ewV3Settings.showWaveLines) {
        function drawV3Lines(node: any) {
          if (!node?.sub_waves?.length) return
          const deg = (node.degree || '').toUpperCase()
          const visible = ewV3Settings.visibleDegrees[deg as keyof typeof ewV3Settings.visibleDegrees]
          if (visible && node.sub_waves.length >= 2) {
            const color = ewV3Settings.degreeColors[deg as keyof typeof ewV3Settings.degreeColors] || '#888'
            const opacity = Math.round(ewV3Settings.waveLineOpacity * 255).toString(16).padStart(2, '0')
            // Draw zigzag lines connecting the sub-wave pivots
            const pivs = [
              { idx: node.sub_waves[0].start_idx, price: node.sub_waves[0].start_price },
              ...node.sub_waves.map((sw: any) => ({ idx: sw.end_idx, price: sw.end_price })),
            ]
            for (let i = 0; i < pivs.length - 1; i++) {
              add({
                name: 'patternTrendline',
                points: [
                  { timestamp: snap(pivs[i].idx),     value: pivs[i].price },
                  { timestamp: snap(pivs[i + 1].idx), value: pivs[i + 1].price },
                ],
                extendData: {
                  color:     color + opacity,
                  lineWidth: deg === 'PRIMARY' || deg === 'CYCLE' ? 2.5 : 1.5,
                  dash:      [],
                },
              })
              // Populate hit cache for hover tooltips
              const sw = node.sub_waves[i] as any
              const sTs = snap(pivs[i].idx), eTs = snap(pivs[i + 1].idx)
              const sp = pivs[i].price, ep = pivs[i + 1].price
              const delta = ep - sp
              const pct = sp !== 0 ? (delta / sp) * 100 : 0
              _v3Segments.push({
                startTs: sTs, endTs: eTs, startPrice: sp, endPrice: ep,
                midTs: Math.round((sTs + eTs) / 2), midPrice: (sp + ep) / 2,
                label:     sw?.label   ?? String(i + 1),
                degree:    deg,
                waveType:  sw?.wave_type ?? (Math.abs(delta) > 0 ? 'wave' : 'wave'),
                direction: delta >= 0 ? 'up' : 'down',
                color,
                priceDelta: delta,
                pricePct:   pct,
                confidence: node.confidence,
              })
            }
          }
          for (const sw of node.sub_waves) drawV3Lines(sw)
        }
        drawV3Lines(primaryNode)
      }

      // Draw wave labels at each visible degree
      if (ewV3Settings.showWaveLabels) {
        for (const lbl of v3Labels) {
          add({
            name: 'patternLabel',
            points: [{ timestamp: snap(lbl.idx), value: lbl.price }],
            extendData: {
              text:    lbl.label,
              color:   lbl.color,
              fontSize: lbl.degree === 'PRIMARY' || lbl.degree === 'CYCLE' ? 11 : 9,
              offsetY:  lbl.isHigh ? -18 : 18,
            },
          })
        }
      }

      // ── Draw linked correction (ABC after primary impulse) ─────────────────
      const linkedCorr = waveTree.linked_correction
      if (linkedCorr?.sub_waves?.length) {
        // Corrective waves use a separate color palette — orange-red tones
        // distinct from the impulse degree colors
        const CORR_COLORS: Record<string, string> = {
          A: '#f97316',  // orange
          B: '#fb923c',  // lighter orange
          C: '#ef4444',  // red
        }
        const CORR_DEFAULT = '#f97316'

        // Bridge line from impulse end → correction start (if gap)
        const corrFirstSub = linkedCorr.sub_waves[0]
        const impulseEndTs = snap(primaryNode.end_idx ?? linkedCorr.start_idx)
        const corrStartTs  = snap(corrFirstSub.start_idx)
        if (corrFirstSub.start_idx !== (primaryNode.end_idx ?? corrFirstSub.start_idx)) {
          add({
            name: 'patternTrendline',
            points: [
              { timestamp: impulseEndTs, value: primaryNode.end_price ?? corrFirstSub.start_price },
              { timestamp: corrStartTs,  value: corrFirstSub.start_price },
            ],
            extendData: { color: CORR_DEFAULT + '55', lineWidth: 1, dash: [4, 4] },
          })
        }

        // Draw each A-B-C segment
        const corrPivs = [
          { idx: corrFirstSub.start_idx, price: corrFirstSub.start_price },
          ...linkedCorr.sub_waves.map((sw: any) => ({ idx: sw.end_idx, price: sw.end_price })),
        ]
        for (let i = 0; i < corrPivs.length - 1; i++) {
          const sw = linkedCorr.sub_waves[i]
          const color = CORR_COLORS[sw.label] ?? CORR_DEFAULT
          const sTs = snap(corrPivs[i].idx), eTs = snap(corrPivs[i + 1].idx)
          const sp = corrPivs[i].price, ep = corrPivs[i + 1].price
          add({
            name: 'patternTrendline',
            points: [
              { timestamp: sTs, value: sp },
              { timestamp: eTs, value: ep },
            ],
            extendData: { color: color + 'dd', lineWidth: 2.5, dash: [] },
          })
          // Add to hover cache
          const delta = ep - sp
          const pct = sp !== 0 ? (delta / sp) * 100 : 0
          _v3Segments.push({
            startTs: sTs, endTs: eTs, startPrice: sp, endPrice: ep,
            midTs: Math.round((sTs + eTs) / 2), midPrice: (sp + ep) / 2,
            label: sw.label, degree: linkedCorr.degree ?? 'INTERMEDIATE',
            waveType: sw.wave_type ?? 'zigzag', direction: delta >= 0 ? 'up' : 'down',
            color, priceDelta: delta, pricePct: pct, confidence: linkedCorr.confidence,
          })
          // Label
          if (ewV3Settings.showWaveLabels) {
            add({
              name: 'patternLabel',
              points: [{ timestamp: eTs, value: ep }],
              extendData: {
                text: sw.label,
                color,
                fontSize: 11,
                offsetY: delta >= 0 ? -18 : 18,
              },
            })
          }
        }
      }

      // Draw v3 projections (Fibonacci targets)
      if (ewV3Settings.showProjections && waveTree.projections?.length) {
        const lastBar = serverBars[serverBars.length - 1]
        const lastTs  = snap(serverBars.length - 1)
        const projStep = Math.max(15, Math.floor(serverBars.length * 0.08)) * barStep

        for (const proj of waveTree.projections) {
          const projColor = proj.is_invalidation ? '#ef4444' : PROJ_CLR
          const projTs    = lastTs + projStep

          add({
            name: 'patternTrendline',
            points: [
              { timestamp: lastTs,    value: lastBar?.close ?? proj.target_price },
              { timestamp: projTs,    value: proj.target_price },
            ],
            extendData: {
              color:     projColor + '99',
              lineWidth: 1.5,
              dash:      ewV3Settings.projectionStyle === 'dashed' ? [5, 4] : [],
            },
          })
          add({
            name: 'patternLabel',
            points: [{ timestamp: projTs, value: proj.target_price }],
            extendData: {
              text:    `${proj.target_price.toFixed(2)} (${proj.fib_label})`,
              color:   projColor,
              fontSize: 9,
              offsetY:  0,
            },
          })
        }
      }
    }
    // ── end v3 ───────────────────────────────────────────────────────────
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

  // v3 display settings — persisted per symbol
  const [ewSettings, setEwSettings] = useState<EWDisplaySettings>(() => {
    try {
      const saved = localStorage.getItem(`ew_v3_settings_${symbol}`)
      if (saved) return { ...DEFAULT_EW_SETTINGS, ...JSON.parse(saved) }
    } catch {}
    return DEFAULT_EW_SETTINGS
  })
  const [showEWSettingsPanel, setShowEWSettingsPanel] = useState(false)
  const [ewV3WaveTree, setEwV3WaveTree] = useState<any>(null)

  // Persist settings
  useEffect(() => {
    try { localStorage.setItem(`ew_v3_settings_${symbol}`, JSON.stringify(ewSettings)) }
    catch {}
  }, [ewSettings, symbol])

  // Sync settings to module-level ref so drawElliottWaves can read them
  useEffect(() => {
    setEwV3Settings(ewSettings)
  }, [ewSettings])

  // Register wave_tree callback so the settings panel can display current position
  useEffect(() => {
    setEwV3WaveTreeCallback((wt) => setEwV3WaveTree(wt))
    return () => setEwV3WaveTreeCallback(null)
  }, [])

  // v3 wave hover — tooltip state
  const [v3HoverSeg, setV3HoverSeg] = useState<V3SegMeta | null>(null)
  const [v3HoverPos, setV3HoverPos] = useState({ x: 0, y: 0 })
  const v3HoverSegRef = useRef<V3SegMeta | null>(null)

  useEffect(() => {
    const container = chartContainerRef.current
    if (!container) return
    const HIT_R = 28  // px — hit radius around segment midpoint

    const onMove = (e: MouseEvent) => {
      const chart = chartRef.current
      if (!chart || !ewVisibleRef.current || !_v3Segments.length) {
        if (v3HoverSegRef.current) { v3HoverSegRef.current = null; setV3HoverSeg(null) }
        return
      }
      const rect = container.getBoundingClientRect()
      const mx = e.clientX - rect.left
      const my = e.clientY - rect.top

      let bestSeg: V3SegMeta | null = null
      let bestD2 = HIT_R * HIT_R

      for (const seg of _v3Segments) {
        const px = chart.convertToPixel(
          { timestamp: seg.midTs, value: seg.midPrice },
          { paneId: 'candle_pane' }
        ) as { x?: number; y?: number }
        if (px.x == null || px.y == null) continue
        const dx = mx - px.x, dy = my - px.y
        const d2 = dx * dx + dy * dy
        if (d2 < bestD2) { bestD2 = d2; bestSeg = seg }
      }

      if (bestSeg !== v3HoverSegRef.current) {
        v3HoverSegRef.current = bestSeg
        setV3HoverSeg(bestSeg)

        // Highlight overlay — remove old, draw new
        try { (chart as any).removeOverlay({ id: 'v3-hl' }) } catch {}
        if (bestSeg) {
          chart.createOverlay({
            id: 'v3-hl',
            name: 'patternTrendline',
            points: [
              { timestamp: bestSeg.startTs, value: bestSeg.startPrice },
              { timestamp: bestSeg.endTs,   value: bestSeg.endPrice   },
            ],
            extendData: { color: '#f97316', lineWidth: 4, dash: [] },
            groupId: 'ew-overlays',
            lock: true,
            mode: 'normal' as any,
            needDefaultPointFigure: false,
            needDefaultXAxisFigure: false,
            needDefaultYAxisFigure: false,
            styles: { point: { activeColor: 'transparent', activeBorderColor: 'transparent', activeBorderSize: 0, activeRadius: 0, color: 'transparent', borderColor: 'transparent', borderSize: 0, radius: 0 } },
          } as any)
        }
      }
      if (bestSeg) setV3HoverPos({ x: mx, y: my })
    }

    const onLeave = () => {
      v3HoverSegRef.current = null
      setV3HoverSeg(null)
      try { (chartRef.current as any)?.removeOverlay({ id: 'v3-hl' }) } catch {}
    }

    container.addEventListener('mousemove', onMove)
    container.addEventListener('mouseleave', onLeave)
    return () => {
      container.removeEventListener('mousemove', onMove)
      container.removeEventListener('mouseleave', onLeave)
    }
  }, [])  // no deps — reads refs at call time

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
  // ── Magnifying glass dissection tool ──────────────────────────────────────
  const [magnifierActive, setMagnifierActive] = useState(false)
  const [magnifierPos, setMagnifierPos] = useState<{ x: number; y: number } | null>(null)
  const magnifierCanvasRef = useRef<HTMLCanvasElement>(null)
  const LENS_W = 280
  const LENS_H = 190
  const ZOOM   = 2.8
  const SRC_W  = Math.round(LENS_W / ZOOM)
  const SRC_H  = Math.round(LENS_H / ZOOM)
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

  // ── Magnifying glass: composite all klinecharts canvas layers into lens ──
  useEffect(() => {
    const container = chartContainerRef.current
    if (!container) return

    const onMove = (e: MouseEvent) => {
      if (!magnifierActive) return
      const rect = container.getBoundingClientRect()
      const x = e.clientX - rect.left
      const y = e.clientY - rect.top
      setMagnifierPos({ x, y })

      const lensCanvas = magnifierCanvasRef.current
      if (!lensCanvas) return
      const ctx = lensCanvas.getContext('2d')
      if (!ctx) return

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

      // Crosshair at center of lens
      const cx = LENS_W / 2, cy = LENS_H / 2
      ctx.save()
      ctx.strokeStyle = 'rgba(255,255,255,0.35)'
      ctx.lineWidth = 0.75
      ctx.setLineDash([3, 3])
      ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, LENS_H); ctx.stroke()
      ctx.beginPath(); ctx.moveTo(0, cy); ctx.lineTo(LENS_W, cy); ctx.stroke()
      ctx.restore()
    }

    const onLeave = () => setMagnifierPos(null)

    container.addEventListener('mousemove', onMove)
    container.addEventListener('mouseleave', onLeave)
    return () => {
      container.removeEventListener('mousemove', onMove)
      container.removeEventListener('mouseleave', onLeave)
    }
  }, [magnifierActive, LENS_W, LENS_H, SRC_W, SRC_H])

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

        {/* Magnifying glass / dissection tool */}
        <button
          onClick={() => { setMagnifierActive(a => !a); setMagnifierPos(null) }}
          className={`px-2.5 py-1.5 rounded-lg text-xs font-medium transition-colors border ${
            magnifierActive
              ? 'bg-blue-600/30 border-blue-500/60 text-blue-300 hover:bg-blue-600/40'
              : 'bg-gray-800 hover:bg-gray-700 text-gray-400 hover:text-white border-gray-700'
          }`}
          title={magnifierActive ? 'Magnifier ON — click to disable' : 'Magnifier — hover to dissect candles'}
        >
          🔍
        </button>

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
      <div className="flex-1 relative min-h-0 overflow-hidden" style={{ cursor: magnifierActive ? 'crosshair' : undefined }}>
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

                    {/* Wave settings (v3 engine) */}
                    <div className="flex items-center gap-1.5 mt-2 ml-4 relative">
                      <button
                        onClick={() => setShowEWSettingsPanel(s => !s)}
                        className={`px-2 py-0.5 rounded text-xs border transition-colors ${
                          showEWSettingsPanel
                            ? 'bg-zinc-700/60 border-zinc-500/60 text-zinc-200'
                            : 'bg-zinc-800/60 border-zinc-600/40 text-zinc-400 hover:text-zinc-200'
                        }`}
                        title="Wave display settings"
                      >
                        ⚙ Settings
                      </button>

                      {/* Settings panel — opens UPWARD, scrollable so all options visible */}
                      {showEWSettingsPanel && (
                        <div className="absolute left-0 bottom-full mb-2 z-50 w-64 max-h-[70vh] overflow-y-auto bg-zinc-900 border border-zinc-700 rounded-lg p-3 shadow-xl text-xs">
                          <div className="flex justify-between items-center mb-2">
                            <span className="font-semibold text-zinc-200">Wave v3 Settings</span>
                            <button onClick={() => setShowEWSettingsPanel(false)} className="text-zinc-500 hover:text-zinc-200 text-base leading-none">✕</button>
                          </div>

                          {/* Degree visibility */}
                          <div className="mb-3">
                            <div className="text-zinc-400 mb-1 font-medium">Visible Degrees</div>
                            {(['PRIMARY','INTERMEDIATE','MINOR','MINUTE','SUBMINUTE'] as const).map(deg => (
                              <label key={deg} className="flex items-center gap-2 mb-1 cursor-pointer">
                                <input
                                  type="checkbox"
                                  checked={ewSettings.visibleDegrees[deg]}
                                  onChange={e => setEwSettings(s => ({
                                    ...s,
                                    visibleDegrees: { ...s.visibleDegrees, [deg]: e.target.checked }
                                  }))}
                                  className="accent-cyan-500"
                                />
                                <span style={{ color: ewSettings.degreeColors[deg] }} className="font-mono">{deg}</span>
                              </label>
                            ))}
                          </div>

                          {/* Projection toggle */}
                          <label className="flex items-center gap-2 mb-2 cursor-pointer">
                            <input
                              type="checkbox"
                              checked={ewSettings.showProjections}
                              onChange={e => setEwSettings(s => ({ ...s, showProjections: e.target.checked }))}
                              className="accent-cyan-500"
                            />
                            <span className="text-zinc-300">Show Projections</span>
                          </label>

                          {/* Starting point toggle */}
                          <label className="flex items-center gap-2 mb-2 cursor-pointer">
                            <input
                              type="checkbox"
                              checked={ewSettings.showStartingPoint}
                              onChange={e => setEwSettings(s => ({ ...s, showStartingPoint: e.target.checked }))}
                              className="accent-cyan-500"
                            />
                            <span className="text-zinc-300">Show Starting Point</span>
                          </label>

                          {/* Wave line opacity slider */}
                          <div className="mb-3">
                            <div className="flex justify-between text-zinc-400 mb-1">
                              <span>Line Opacity</span>
                              <span className="text-zinc-300 font-mono">{Math.round(ewSettings.waveLineOpacity * 100)}%</span>
                            </div>
                            <input
                              type="range" min="0.1" max="1.0" step="0.05"
                              value={ewSettings.waveLineOpacity}
                              onChange={e => setEwSettings(s => ({ ...s, waveLineOpacity: parseFloat(e.target.value) }))}
                              className="w-full accent-cyan-500"
                            />
                          </div>

                          {/* Projection style */}
                          <div className="mb-3">
                            <div className="text-zinc-400 mb-1">Projection Lines</div>
                            <div className="flex gap-2">
                              {(['solid','dashed'] as const).map(style => (
                                <button
                                  key={style}
                                  onClick={() => setEwSettings(s => ({ ...s, projectionStyle: style }))}
                                  className={`px-2.5 py-0.5 rounded border text-xs transition-colors ${
                                    ewSettings.projectionStyle === style
                                      ? 'border-cyan-500 bg-cyan-500/15 text-cyan-300'
                                      : 'border-zinc-600 text-zinc-400 hover:text-zinc-200'
                                  }`}
                                >{style}</button>
                              ))}
                            </div>
                          </div>

                          {/* Current position display */}
                          {ewV3WaveTree?.current_position && (
                            <div className="mt-1 pt-2 border-t border-zinc-700">
                              <div className="text-zinc-400 mb-1 font-medium">Current Position</div>
                              <div className="text-cyan-300 font-mono text-[10px] leading-relaxed break-words">
                                {ewV3WaveTree.current_position}
                              </div>
                              <div className="text-zinc-500 mt-1">
                                Confidence: {((ewV3WaveTree.overall_confidence ?? 0) * 100).toFixed(1)}%
                              </div>
                            </div>
                          )}
                        </div>
                      )}
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

        {/* ── Magnifier Lens ──────────────────────────────────────────── */}
        {/* Canvas always mounted when magnifier is active so the ref is available
            on the very first mousemove; visibility is controlled by the wrapper  */}
        {magnifierActive && (() => {
          const container = chartContainerRef.current
          const cW = container?.clientWidth  ?? 900
          const cH = container?.clientHeight ?? 500
          const PAD = 18
          let lx = 0, ly = 0
          if (magnifierPos) {
            lx = magnifierPos.x + PAD
            if (lx + LENS_W > cW - 4) lx = magnifierPos.x - LENS_W - PAD
            ly = magnifierPos.y - LENS_H / 2
            if (ly < 4) ly = 4
            if (ly + LENS_H > cH - 4) ly = cH - LENS_H - 4
          }

          return (
            <div
              className="absolute pointer-events-none z-50 transition-opacity duration-75"
              style={{
                left: lx, top: ly,
                width: LENS_W, height: LENS_H,
                opacity: magnifierPos ? 1 : 0,
              }}
            >
              {/* Lens border + glow */}
              <div className="absolute inset-0 rounded-xl border border-blue-400/50 shadow-[0_0_20px_rgba(59,130,246,0.25)]" />
              {/* Zoom badge */}
              <div className="absolute top-1.5 right-2 text-[9px] font-mono text-blue-300/60 select-none z-10">
                {ZOOM.toFixed(1)}×
              </div>
              {/* Zoomed canvas — updated on every mousemove */}
              <canvas
                ref={magnifierCanvasRef}
                width={LENS_W}
                height={LENS_H}
                className="rounded-xl block"
              />
            </div>
          )
        })()}

        {/* v3 wave hover tooltip */}
        {v3HoverSeg && ewVisible && (
          <div
            className="pointer-events-none absolute z-40 min-w-[180px] max-w-[240px] rounded-lg bg-zinc-900/95 border border-zinc-700/80 backdrop-blur-sm shadow-2xl shadow-black/60 text-xs overflow-hidden"
            style={{
              left: v3HoverPos.x + 16,
              top:  Math.max(8, v3HoverPos.y - 80),
            }}
          >
            {/* Header strip with degree color */}
            <div
              className="flex items-center justify-between px-3 py-1.5"
              style={{ borderBottom: `1px solid ${v3HoverSeg.color}40`, background: `${v3HoverSeg.color}18` }}
            >
              <span className="font-bold text-white text-sm">Wave {v3HoverSeg.label}</span>
              <span className="font-mono text-[10px] px-1.5 py-0.5 rounded" style={{ color: v3HoverSeg.color, background: `${v3HoverSeg.color}25` }}>
                {v3HoverSeg.degree}
              </span>
            </div>
            {/* Body */}
            <div className="px-3 py-2 space-y-1">
              <div className="flex justify-between items-center">
                <span className="text-zinc-400">Type</span>
                <span className="text-zinc-200 capitalize">{(v3HoverSeg.waveType || 'wave').replace(/_/g, ' ')}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-zinc-400">Direction</span>
                <span className={v3HoverSeg.direction === 'up' ? 'text-emerald-400' : 'text-red-400'}>
                  {v3HoverSeg.direction === 'up' ? '▲ Bullish' : '▼ Bearish'}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-zinc-400">From → To</span>
                <span className="text-zinc-200 font-mono text-[10px]">
                  {v3HoverSeg.startPrice.toFixed(2)} → {v3HoverSeg.endPrice.toFixed(2)}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-zinc-400">Move</span>
                <span className={`font-mono font-semibold ${v3HoverSeg.priceDelta >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                  {v3HoverSeg.priceDelta >= 0 ? '+' : ''}{v3HoverSeg.priceDelta.toFixed(2)}
                  {' '}({v3HoverSeg.pricePct >= 0 ? '+' : ''}{v3HoverSeg.pricePct.toFixed(1)}%)
                </span>
              </div>
              {v3HoverSeg.confidence != null && (
                <div className="flex justify-between items-center pt-0.5 border-t border-zinc-800">
                  <span className="text-zinc-500">Confidence</span>
                  <span className="text-zinc-300 font-mono">{(v3HoverSeg.confidence * 100).toFixed(0)}%</span>
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* ── Footer hint ──────────────────────────────────────────────── */}
      <div className="flex items-center justify-between px-4 py-1 bg-gray-900/60 border-t border-gray-800/50 text-[10px] text-gray-600 flex-shrink-0">
        <span>Scroll to zoom · Drag to pan · Double-click to toggle indicators · ESC to close</span>
        <span>MA(20/50/200) · Vol · Slow STO %K(14) %D(3) · Elliott Wave multi-degree</span>
      </div>
    </div>
  )
}
