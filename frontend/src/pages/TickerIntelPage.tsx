/**
 * Ticker Intelligence — Symbol deep-dive across all 5 stages.
 *
 * Pull any ticker → get the full picture: regime, signals, AI probability,
 * pattern confirmation, and high-probability trade levels.
 *
 * Three adaptive views:
 *   Novice       — plain-English verdict + entry/target/stop
 *   Intermediate — 4-stage checklist + trade plan
 *   Expert       — full component scores + all patterns + decision trace
 */

import { useState, useEffect, useRef } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { useConfluenceScore, usePatterns, usePriceLevels } from '../api/hooks'
import type { PatternEntry, PriceLevelCluster, PriceLevelZones } from '../types'

// ── Types ─────────────────────────────────────────────────────────────────────

type ViewMode = 'novice' | 'intermediate' | 'expert'

interface StageResult {
  pass: boolean
  partial: boolean
  label: string
  detail: string
  na?: boolean   // true when pipeline data unavailable for this stage
}

// ── Helpers ───────────────────────────────────────────────────────────────────

const TIER_COLOR: Record<string, string> = {
  HIGH:   '#22c55e',
  MEDIUM: '#f59e0b',
  WATCH:  '#60a5fa',
  NONE:   '#6b7280',
}

const TIER_LABEL: Record<string, string> = {
  HIGH:   'HIGH CONFIDENCE',
  MEDIUM: 'MEDIUM CONFIDENCE',
  WATCH:  'WATCHING',
  NONE:   'NO SIGNAL',
}

function fmt(n: number | undefined | null, decimals = 2) {
  if (n == null || isNaN(n)) return '—'
  return n.toFixed(decimals)
}

function fmtPct(current: number, reference: number) {
  if (!reference) return ''
  const diff = ((current - reference) / reference) * 100
  const sign = diff >= 0 ? '+' : ''
  return `${sign}${diff.toFixed(1)}%`
}

function activePatterns(patterns: PatternEntry[] | undefined) {
  return (patterns ?? []).filter(
    p => p.status && !['NOT_PRESENT', 'FAILED'].includes(p.status)
  )
}

// When direction is bearish, invert the bullish-framed confluence score to derive signal tier.
// A raw score of 46/100 (bullish: NONE) → bearish strength 54 → MEDIUM bearish.
function computeBearishTier(confluenceScore: number): string {
  const bearStr = 100 - confluenceScore
  if (bearStr >= 65) return 'HIGH'
  if (bearStr >= 50) return 'MEDIUM'
  return 'WATCH'    // there is always a trade — minimum WATCH for bearish
}

// There is always a trade — direction is always bullish or bearish.
// Trend score < 50 = bearish lean; ≥ 50 = bullish lean.
function inferDirection(
  compScores: Record<string, number> | undefined,
  patterns: PatternEntry[] | undefined,
  isBreakout: boolean
): 'bullish' | 'bearish' | 'neutral' {
  const active = activePatterns(patterns)
  const bullishPats = active.filter(p => p.direction === 'bullish').length
  const bearishPats = active.filter(p => p.direction === 'bearish').length

  // Pattern votes win by 2+ to be decisive
  if (bullishPats >= bearishPats + 2) return 'bullish'
  if (bearishPats >= bullishPats + 2) return 'bearish'

  // Breakout confirmed = bullish
  if (isBreakout) return 'bullish'

  // Trend score determines direction — anything below 50 is a bearish lean
  const trend = compScores?.trend ?? 50
  if (trend >= 50) return 'bullish'
  return 'bearish'
}

// Leveraged inverse ETF recommendation for bearish setups.
// Keys = underlying (non-leveraged) stocks or 1× ETFs that are valid analysis targets.
// Values = the leveraged short vehicle to trade when that underlying is bearish.
// Leveraged ETFs (SOXL, TQQQ, UPRO, etc.) are NOT keys here — they are not analysis targets.
const SHORT_ETF_MAP: Record<string, { etf: string; name: string; leverage: string }> = {
  // ── Semiconductors → SOXS (3× Short Philadelphia SOX) ───────────────────
  NVDA: { etf: 'SOXS', name: '3× Short SOX',          leverage: '3x' },
  AMD:  { etf: 'SOXS', name: '3× Short SOX',          leverage: '3x' },
  INTC: { etf: 'SOXS', name: '3× Short SOX',          leverage: '3x' },
  AVGO: { etf: 'SOXS', name: '3× Short SOX',          leverage: '3x' },
  TSM:  { etf: 'SOXS', name: '3× Short SOX',          leverage: '3x' },
  QCOM: { etf: 'SOXS', name: '3× Short SOX',          leverage: '3x' },
  MU:   { etf: 'SOXS', name: '3× Short SOX',          leverage: '3x' },
  SOXX: { etf: 'SOXS', name: '3× Short SOX',          leverage: '3x' },
  SMH:  { etf: 'SOXS', name: '3× Short SOX',          leverage: '3x' },

  // ── Nasdaq mega-cap & QQQ → SQQQ (3× Short QQQ) ─────────────────────────
  AAPL: { etf: 'SQQQ', name: '3× Short QQQ',          leverage: '3x' },
  MSFT: { etf: 'SQQQ', name: '3× Short QQQ',          leverage: '3x' },
  META: { etf: 'SQQQ', name: '3× Short QQQ',          leverage: '3x' },
  AMZN: { etf: 'SQQQ', name: '3× Short QQQ',          leverage: '3x' },
  GOOGL:{ etf: 'SQQQ', name: '3× Short QQQ',          leverage: '3x' },
  GOOG: { etf: 'SQQQ', name: '3× Short QQQ',          leverage: '3x' },
  TSLA: { etf: 'SQQQ', name: '3× Short QQQ',          leverage: '3x' },
  NFLX: { etf: 'SQQQ', name: '3× Short QQQ',          leverage: '3x' },
  ADBE: { etf: 'SQQQ', name: '3× Short QQQ',          leverage: '3x' },
  CRM:  { etf: 'SQQQ', name: '3× Short QQQ',          leverage: '3x' },
  PYPL: { etf: 'SQQQ', name: '3× Short QQQ',          leverage: '3x' },
  QQQ:  { etf: 'SQQQ', name: '3× Short QQQ',          leverage: '3x' },
  XLK:  { etf: 'SQQQ', name: '3× Short QQQ/Tech',     leverage: '3x' },

  // ── S&P 500 / broad market → SPXS (3× Short S&P 500) ────────────────────
  SPY:  { etf: 'SPXS', name: '3× Short S&P 500',      leverage: '3x' },
  IVV:  { etf: 'SPXS', name: '3× Short S&P 500',      leverage: '3x' },
  VOO:  { etf: 'SPXS', name: '3× Short S&P 500',      leverage: '3x' },
  XLP:  { etf: 'SPXS', name: '3× Short S&P (staples)', leverage: '3x' },
  XLV:  { etf: 'SPXS', name: '3× Short S&P (health)', leverage: '3x' },
  XLI:  { etf: 'SPXS', name: '3× Short S&P (indust.)', leverage: '3x' },
  XLB:  { etf: 'SPXS', name: '3× Short S&P (matls.)', leverage: '3x' },

  // ── Small-cap Russell → TZA (3× Short Russell 2000) ─────────────────────
  IWM:  { etf: 'TZA',  name: '3× Short Russell 2000', leverage: '3x' },
  IWO:  { etf: 'TZA',  name: '3× Short Russell 2000', leverage: '3x' },

  // ── Dow Jones → SDOW (3× Short DOW) ─────────────────────────────────────
  DIA:  { etf: 'SDOW', name: '3× Short DOW',           leverage: '3x' },
  BA:   { etf: 'SDOW', name: '3× Short DOW',           leverage: '3x' },
  CAT:  { etf: 'SDOW', name: '3× Short DOW',           leverage: '3x' },

  // ── Financials → FAZ (3× Short Financials) ───────────────────────────────
  XLF:  { etf: 'FAZ',  name: '3× Short Financials',   leverage: '3x' },
  JPM:  { etf: 'FAZ',  name: '3× Short Financials',   leverage: '3x' },
  BAC:  { etf: 'FAZ',  name: '3× Short Financials',   leverage: '3x' },
  GS:   { etf: 'FAZ',  name: '3× Short Financials',   leverage: '3x' },
  MS:   { etf: 'FAZ',  name: '3× Short Financials',   leverage: '3x' },
  C:    { etf: 'FAZ',  name: '3× Short Financials',   leverage: '3x' },

  // ── Energy → ERY (3× Short Energy) ──────────────────────────────────────
  XLE:  { etf: 'ERY',  name: '3× Short Energy',        leverage: '3x' },
  XOM:  { etf: 'ERY',  name: '3× Short Energy',        leverage: '3x' },
  CVX:  { etf: 'ERY',  name: '3× Short Energy',        leverage: '3x' },
  OXY:  { etf: 'ERY',  name: '3× Short Energy',        leverage: '3x' },
  USO:  { etf: 'ERY',  name: '3× Short Energy/Oil',    leverage: '3x' },

  // ── Gold & Gold Miners → DUST (2× Short Gold Miners) ────────────────────
  GLD:  { etf: 'DUST', name: '2× Short Gold Miners',   leverage: '2x' },
  GDX:  { etf: 'DUST', name: '2× Short Gold Miners',   leverage: '2x' },
  GDXJ: { etf: 'DUST', name: '2× Short Gold Miners',   leverage: '2x' },
  IAU:  { etf: 'DUST', name: '2× Short Gold',          leverage: '2x' },

  // ── Silver → ZSL (2× Short Silver) ──────────────────────────────────────
  SLV:  { etf: 'ZSL',  name: '2× Short Silver',        leverage: '2x' },
  SIVR: { etf: 'ZSL',  name: '2× Short Silver',        leverage: '2x' },

  // ── Utilities → SDP (2× Short Utilities) ─────────────────────────────────
  XLU:  { etf: 'SDP',  name: '2× Short Utilities',     leverage: '2x' },

  // ── Biotech → LABD (3× Short Biotech) ────────────────────────────────────
  IBB:  { etf: 'LABD', name: '3× Short Biotech',       leverage: '3x' },
  XBI:  { etf: 'LABD', name: '3× Short Biotech',       leverage: '3x' },

  // ── China → YANG (3× Short China) ───────────────────────────────────────
  FXI:  { etf: 'YANG', name: '3× Short China',         leverage: '3x' },
  KWEB: { etf: 'YANG', name: '3× Short China Tech',    leverage: '3x' },
}

function getShortETF(symbol: string) {
  // Leveraged ETFs are not analysis targets — return null so no ETF suggestion is made.
  const LEVERAGED_ETF = new Set([
    'SOXL','SOXS','TQQQ','SQQQ','UPRO','SPXS','SPXU','TNA','TZA',
    'LABU','LABD','DUST','NUGT','JNUG','JDST','SDOW','UDOW',
    'YANG','YINN','ERX','ERY','FAS','FAZ','ZSL','UGL','AGQ',
    'SDP','DIG','DUG','SDP','KOLD','BOIL',
  ])
  if (LEVERAGED_ETF.has(symbol.toUpperCase())) return null
  return SHORT_ETF_MAP[symbol.toUpperCase()] ?? { etf: 'SPXS', name: '3× Short S&P 500', leverage: '3x' }
}

// Returns up to 4 tiered ETF options for playing the bearish side of a symbol.
// Most direct first, then sector, then broad market fallback.
interface ETFOption { etf: string; name: string; leverage: string; note: string }

const SEMI_STOCKS   = new Set(['NVDA','AMD','INTC','AVGO','QCOM','MU','TSM','SOXX','SMH'])
const TECH_STOCKS   = new Set(['AAPL','MSFT','META','AMZN','GOOGL','GOOG','TSLA','NFLX','ADBE','CRM','PYPL'])
const ENERGY_STOCKS = new Set(['XOM','CVX','OXY','USO','XLE'])
const FIN_STOCKS    = new Set(['JPM','BAC','GS','MS','C','XLF'])

function getBearishETFList(symbol: string): ETFOption[] {
  const sym     = symbol.toUpperCase()
  const primary = SHORT_ETF_MAP[sym]
  const list: ETFOption[] = []

  if (primary) list.push({ ...primary, note: 'most direct' })

  // Add relevant secondary options
  if (SEMI_STOCKS.has(sym) && primary?.etf !== 'SQQQ') {
    list.push({ etf: 'SQQQ', name: '3× Short QQQ', leverage: '3x', note: 'tech broad' })
  }
  if (TECH_STOCKS.has(sym) && primary?.etf !== 'SOXS' && SEMI_STOCKS.has(sym) === false) {
    list.push({ etf: 'SQQQ', name: '3× Short QQQ', leverage: '3x', note: 'most direct' })
  }
  if (ENERGY_STOCKS.has(sym) && primary?.etf !== 'SPXS') {
    list.push({ etf: 'SPXS', name: '3× Short S&P 500', leverage: '3x', note: 'broad market' })
  }
  if (FIN_STOCKS.has(sym) && primary?.etf !== 'SPXS') {
    list.push({ etf: 'SPXS', name: '3× Short S&P 500', leverage: '3x', note: 'broad market' })
  }

  // Always add broad market unless it's already the primary
  if (!list.some(e => e.etf === 'SPXS')) {
    list.push({ etf: 'SPXS', name: '3× Short S&P 500', leverage: '3x', note: 'broad market' })
  }
  // Add SDS (2× Short S&P) as conservative option for most cases
  if (list.length < 3) {
    list.push({ etf: 'SDS', name: '2× Short S&P 500', leverage: '2x', note: 'moderate risk' })
  }

  return list.slice(0, 4)
}

function computeStages(
  scores: Record<string, number> | undefined,
  isComp: boolean,
  isBreakout: boolean,
  tier: string,
  patterns: PatternEntry[] | undefined,
  direction: 'bullish' | 'bearish' | 'neutral' = 'bullish',
  gated = false,
): StageResult[] {
  const s    = scores ?? {}
  const bear = direction === 'bearish'

  // When symbol is not in the pipeline DB, stages 1–3 have no data
  if (gated) {
    const active = activePatterns(patterns)
    const dirPatterns = bear
      ? active.filter(p => p.direction === 'bearish' || p.direction === 'neutral')
      : active.filter(p => p.direction === 'bullish' || p.direction === 'neutral')
    const s4Pass = dirPatterns.some(p => ['READY', 'BREAKOUT', 'COMPLETED'].includes(p.status ?? ''))
    const s4Part = dirPatterns.some(p => p.status === 'FORMING')
    return [
      { pass: false, partial: false, label: 'Regime', detail: 'Not in pipeline — no data', na: true },
      { pass: false, partial: false, label: 'Core Signals', detail: 'Not in pipeline — no data', na: true },
      { pass: false, partial: false, label: 'AI Probability', detail: 'Not in pipeline — no data', na: true },
      { pass: s4Pass, partial: !s4Pass && s4Part,
        label: 'Pattern Confirmation',
        detail: s4Pass ? `${dirPatterns.filter(p => ['READY','BREAKOUT','COMPLETED'].includes(p.status ?? '')).length} confirmed pattern(s)` : s4Part ? 'Patterns forming' : 'No active patterns' },
    ]
  }

  // Stage 1 — Regime (works for both; distribution = bearish regime confirmation)
  const volScore = s.vol_compression ?? 0
  const trend    = s.trend ?? 50
  const bo       = s.breakout ?? 0
  const liq      = s.liquidity ?? 0
  const wyckoff  = s.wyckoff ?? 0

  const s1Pass = bear
    ? trend < 42 || isComp || wyckoff > 60                // declining trend / distribution
    : volScore > 60 || isComp || isBreakout               // bullish regime
  const s1Part = bear
    ? trend < 50                                          // any below-mid trend
    : volScore > 40

  // Stage 2 — Core Signals (inverted thresholds for bear)
  const s2Pass = bear
    ? trend < 45 && (bo < 35 || liq < 45)                // weak trend + weak momentum
    : trend > 55 && (bo > 40 || liq > 55)
  const s2Part = bear
    ? trend < 52
    : trend > 45 || bo > 30

  // Stage 3 — AI Probability (same — HIGH/MEDIUM tier catches bearish too)
  const s3Pass = tier === 'HIGH' || tier === 'MEDIUM'
  const s3Part = tier === 'WATCH'

  // Stage 4 — Pattern Confirmation (filter by direction-relevant patterns)
  const active = activePatterns(patterns)
  const dirPatterns = bear
    ? active.filter(p => p.direction === 'bearish' || p.direction === 'neutral')
    : active.filter(p => p.direction === 'bullish' || p.direction === 'neutral')
  const s4Pass = dirPatterns.some(p => ['READY', 'BREAKOUT', 'COMPLETED'].includes(p.status ?? ''))
  const s4Part = dirPatterns.some(p => p.status === 'FORMING')

  // — Labels and detail strings ——————————————————————————————
  const s1Detail = bear
    ? isComp
      ? `Distribution compression — breakdown energy building (vol ${fmt(volScore, 0)})`
      : s1Pass
      ? `Bearish regime confirmed — Trend ${fmt(trend, 0)} · Wyckoff ${fmt(wyckoff, 0)}`
      : trend < 50
      ? `Trend weakening (${fmt(trend, 0)}) — conditions building`
      : 'Regime not yet bearish'
    : isBreakout
    ? 'Breakout in progress'
    : isComp
    ? `Volatility compressed — energy building (${fmt(volScore, 0)}/100)`
    : s1Pass
    ? `Market structure supports trade (${fmt(volScore, 0)}/100)`
    : 'Conditions not yet aligned'

  const s2Detail = bear
    ? s2Pass
      ? `Trend declining ${fmt(trend, 0)} · Breakout fading ${fmt(bo, 0)} · Liquidity ${fmt(liq, 0)}`
      : `Signals mixed — Trend ${fmt(trend, 0)} · Breakout ${fmt(bo, 0)}`
    : s2Pass
    ? `Trend ${fmt(trend, 0)} · Breakout ${fmt(bo, 0)} · Liquidity ${fmt(liq, 0)}`
    : `Signals mixed — Trend ${fmt(trend, 0)} · Breakout ${fmt(bo, 0)}`

  const s3Detail = tier === 'NONE' || !tier
    ? bear ? 'Low AI confidence — monitor for distribution signal' : 'Below signal threshold'
    : `${tier} tier — pattern ${fmt(s.pattern, 0)} · Wyckoff ${fmt(s.wyckoff, 0)}`

  const s4Detail = s4Pass
    ? dirPatterns
        .filter(p => ['READY', 'BREAKOUT', 'COMPLETED'].includes(p.status ?? ''))
        .map(p => `${p.pattern_name} ${p.status}`).slice(0, 2).join(' · ')
    : s4Part
    ? dirPatterns.filter(p => p.status === 'FORMING')
        .map(p => p.pattern_name).slice(0, 2).join(' · ') + ' — forming'
    : bear
    ? 'No bearish pattern confirmed yet'
    : 'No confirming patterns yet'

  return [
    { pass: s1Pass, partial: s1Part && !s1Pass, label: 'Regime',               detail: s1Detail },
    { pass: s2Pass, partial: s2Part && !s2Pass, label: 'Core Signals',          detail: s2Detail },
    { pass: s3Pass, partial: s3Part && !s3Pass, label: 'AI Probability',        detail: s3Detail },
    { pass: s4Pass, partial: s4Part && !s4Pass, label: 'Pattern Confirmation',  detail: s4Detail },
  ]
}

function buildNoviceDescription(
  direction: 'bullish' | 'bearish' | 'neutral',
  isComp: boolean,
  isBreakout: boolean,
  scores: Record<string, number> | undefined,
  bestPattern: PatternEntry | undefined,
  entry: number,
  target: number
): string {
  const s = scores ?? {}
  const parts: string[] = []

  if (direction === 'bullish') {
    if (isBreakout) parts.push('Price has broken out with strong volume — buyers are in control')
    else if (isComp) parts.push('Volatility has been coiling tight — energy is building for a move up')
    else parts.push('The trend is pointing upward')
    if ((s.trend ?? 0) > 60) parts.push('momentum and moving averages confirm the bullish direction')
    if (bestPattern) parts.push(`a ${bestPattern.pattern_name} pattern is ${(bestPattern.status ?? '').toLowerCase()} at $${fmt(bestPattern.breakout_level ?? entry)}`)
    if (target > entry) parts.push(`the measured move projects toward $${fmt(target)}`)
  } else if (direction === 'bearish') {
    if (isBreakout) parts.push('Price has broken down — sellers are pushing lower')
    else parts.push('The trend is turning downward')
    if ((s.trend ?? 0) < 40) parts.push('momentum is weak and moving averages are aligned bearishly')
    if (bestPattern) parts.push(`a ${bestPattern.pattern_name} pattern is ${(bestPattern.status ?? '').toLowerCase()}`)
  } else {
    parts.push('The market is not showing a clear directional bias yet')
    if (isComp) parts.push('volatility is compressed — watch for a breakout in either direction')
    else parts.push('wait for signals to align before taking action')
  }

  return parts.length
    ? parts[0].charAt(0).toUpperCase() + parts[0].slice(1) + '. ' + parts.slice(1).join(', ') + '.'
    : 'Analysis in progress — check back when more data is available.'
}

// ── Sub-components ─────────────────────────────────────────────────────────────

function ScoreBar({ value, color }: { value: number; color: string }) {
  return (
    <div className="w-full h-1.5 bg-gray-800 rounded-full overflow-hidden">
      <div
        className="h-full rounded-full transition-all duration-500"
        style={{ width: `${Math.min(100, Math.max(0, value))}%`, background: color }}
      />
    </div>
  )
}

function StageDot({ pass, partial, na }: { pass: boolean; partial: boolean; na?: boolean }) {
  if (na)      return <span className="text-gray-600 text-lg leading-none">⊘</span>
  if (pass)    return <span className="text-emerald-400 text-lg leading-none">●</span>
  if (partial) return <span className="text-amber-400 text-lg leading-none">◐</span>
  return              <span className="text-gray-700 text-lg leading-none">○</span>
}

function PatternBadge({ p }: { p: PatternEntry }) {
  const statusColors: Record<string, string> = {
    BREAKOUT:  'bg-emerald-900/60 text-emerald-300 border-emerald-800/40',
    READY:     'bg-amber-900/60 text-amber-300 border-amber-800/40',
    FORMING:   'bg-gray-800 text-gray-400 border-gray-700/40',
    COMPLETED: 'bg-emerald-900/30 text-emerald-400 border-emerald-800/30',
  }
  const cls = statusColors[p.status ?? ''] ?? 'bg-gray-800 text-gray-500 border-gray-700'
  const dirArrow = p.direction === 'bullish' ? '↑' : p.direction === 'bearish' ? '↓' : '→'
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded border text-xs font-mono ${cls}`}>
      {dirArrow} {p.pattern_name} · {p.status}
    </span>
  )
}

// ── Stage 5 — Price Level Zone Components ─────────────────────────────────────

const ZONE_ORDER = [
  { key: 'distribution', label: 'Distribution',  color: '#22c55e', above: true  },
  { key: 'target',       label: 'Target',         color: '#86efac', above: true  },
  { key: 'scale_in',     label: 'Scale In',       color: '#a78bfa', above: true  },
  { key: 'buy_zone',     label: 'Buy Zone',       color: '#60a5fa', above: false },
  { key: 'accumulate',   label: 'Accumulate',     color: '#93c5fd', above: false },
  { key: 'stop',         label: 'Stop Zone',      color: '#f87171', above: false },
]

function StrengthBar({ value }: { value: number }) {
  const pct = Math.round((value / 10) * 100)
  const color = value >= 7 ? '#22c55e' : value >= 4 ? '#f59e0b' : '#6b7280'
  return (
    <div className="flex items-center gap-1.5">
      <div className="h-1.5 w-16 bg-gray-800 rounded-full overflow-hidden">
        <div className="h-full rounded-full" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
      <span className="text-[10px] font-mono" style={{ color }}>{value}/10</span>
    </div>
  )
}

function PriceLevelZonePanel({
  zones, currentPrice, atrPct, compact = false,
}: {
  zones: PriceLevelZones
  currentPrice: number
  atrPct: number
  compact?: boolean
}) {
  const zoneEntries = ZONE_ORDER.map(z => ({
    ...z,
    cluster: zones[z.key as keyof PriceLevelZones] as PriceLevelCluster | undefined,
  })).filter(z => z.cluster != null)

  if (zoneEntries.length === 0) return null

  if (compact) {
    // Novice / compact: just show colour-coded prices in a row
    return (
      <div className="grid grid-cols-3 gap-2 text-center">
        {zoneEntries.slice(0, 6).map(z => (
          <div key={z.key} className="bg-gray-900 border border-gray-800 rounded-xl p-3">
            <p className="text-xs text-gray-500 mb-1">{z.label}</p>
            <p className="text-base font-mono font-bold" style={{ color: z.color }}>
              ${z.cluster!.price.toFixed(2)}
            </p>
            <p className="text-[10px] mt-0.5" style={{ color: z.color }}>
              {z.cluster!.distance_pct > 0 ? '+' : ''}{z.cluster!.distance_pct.toFixed(1)}%
            </p>
          </div>
        ))}
      </div>
    )
  }

  // Full zone table with strength + sources
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
      <div className="px-4 py-2.5 border-b border-gray-800 flex items-center justify-between">
        <span className="text-xs text-gray-500 uppercase tracking-widest font-semibold">Stage 5 — Price Levels</span>
        <span className="text-[10px] text-gray-600 font-mono">ATR {atrPct.toFixed(1)}%</span>
      </div>
      <div className="divide-y divide-gray-800/40">
        {/* Current price marker */}
        <div className="px-4 py-2 flex items-center justify-between bg-gray-800/30">
          <span className="text-xs text-gray-400 font-semibold">Current Price</span>
          <span className="text-sm font-mono font-bold text-white">${currentPrice.toFixed(2)}</span>
        </div>
        {zoneEntries.map(z => (
          <div key={z.key} className="px-4 py-2.5 flex items-center gap-3">
            {/* Colour dot + label */}
            <div className="w-2 h-2 rounded-full flex-shrink-0" style={{ backgroundColor: z.color }} />
            <span className="text-xs font-semibold w-24 flex-shrink-0" style={{ color: z.color }}>
              {z.label}
            </span>
            {/* Price */}
            <span className="text-sm font-mono font-bold text-gray-100 w-20 flex-shrink-0">
              ${z.cluster!.price.toFixed(2)}
            </span>
            {/* Distance */}
            <span className="text-[10px] font-mono text-gray-500 w-12 flex-shrink-0">
              {z.cluster!.distance_pct > 0 ? '+' : ''}{z.cluster!.distance_pct.toFixed(1)}%
            </span>
            {/* Strength bar */}
            <StrengthBar value={z.cluster!.strength} />
            {/* Sources */}
            <span className="text-[10px] text-gray-600 truncate hidden sm:block">
              {z.cluster!.sources.slice(0, 3).join(' · ')}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}

// ── View: Novice ───────────────────────────────────────────────────────────────

function ShortETFBadge({ etf, name, leverage, onClick }: { etf: string; name: string; leverage: string; onClick: () => void }) {
  return (
    <button onClick={onClick}
      className="flex items-center gap-2 px-3 py-2 rounded-xl border border-red-800/50 bg-red-950/30 hover:bg-red-900/30 transition-colors text-left w-full"
    >
      <span className="text-red-400 text-lg">↓</span>
      <div>
        <p className="text-xs font-bold text-red-300">Trade via {etf} <span className="font-mono text-[10px] text-red-500/80">{leverage}</span></p>
        <p className="text-[10px] text-red-500/70">{name} — analyse this ETF →</p>
      </div>
    </button>
  )
}

// Full ETF list card — shows all tiered bearish trade vehicles for the right-hand card.
function BearishETFListCard({ etfList, onSelect }: {
  etfList: ETFOption[]
  onSelect: (etf: string) => void
}) {
  if (etfList.length === 0) return null
  return (
    <div className="rounded-xl border border-red-900/40 bg-red-950/10 overflow-hidden">
      <div className="px-3 py-2 border-b border-red-900/20 bg-red-950/20 flex items-center gap-2">
        <span className="text-red-400 text-sm">↓</span>
        <p className="text-[10px] text-red-300 uppercase tracking-widest font-semibold flex-1">Bearish ETF Vehicles</p>
        <span className="text-[10px] text-gray-600">click to analyse</span>
      </div>
      <div className="divide-y divide-gray-800/30">
        {etfList.map(({ etf, name, leverage, note }) => (
          <button
            key={etf}
            onClick={() => onSelect(etf)}
            className="w-full px-3 py-2.5 flex items-center gap-3 hover:bg-red-950/25 transition-colors text-left group"
          >
            <span className="font-mono font-bold text-sm text-red-300 w-14 flex-shrink-0 group-hover:text-red-200 transition-colors">{etf}</span>
            <span className="text-xs text-gray-400 flex-1 truncate">{name}</span>
            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-red-950/60 text-red-400 border border-red-900/30 flex-shrink-0">{leverage}</span>
            <span className="text-[10px] text-gray-600 flex-shrink-0 hidden sm:block w-20 text-right">{note}</span>
          </button>
        ))}
      </div>
    </div>
  )
}

function NoviceView({
  symbol, direction, tier, entry, target, stop, description, bestPattern, shortETF, etfList, onUpgrade, onPaperTrade, onShortETF,
}: {
  symbol: string
  direction: 'bullish' | 'bearish' | 'neutral'
  tier: string
  entry: number
  target: number
  stop: number
  description: string
  bestPattern: PatternEntry | undefined
  shortETF: { etf: string; name: string; leverage: string } | null
  etfList: ETFOption[]
  onUpgrade: () => void
  onPaperTrade: () => void
  onShortETF: (etf?: string) => void
}) {
  const navigate   = useNavigate()
  const dirColor   = direction === 'bullish' ? '#22c55e' : direction === 'bearish' ? '#ef4444' : '#9ca3af'
  const dirLabel   = direction === 'bullish' ? '⬆ LOOKS BULLISH' : direction === 'bearish' ? '⬇ LOOKS BEARISH' : '→ NO CLEAR DIRECTION'
  const tierColor  = TIER_COLOR[tier] ?? TIER_COLOR.NONE
  const tierLabel  = TIER_LABEL[tier] ?? TIER_LABEL.NONE

  return (
    <div className="space-y-6 max-w-xl">
      {/* Verdict */}
      <div className="bg-gray-900 border border-gray-800 rounded-2xl p-6 space-y-3">
        <div className="flex items-baseline gap-3">
          <span className="text-3xl font-black" style={{ color: dirColor }}>{dirLabel}</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs px-2 py-0.5 rounded font-semibold" style={{ background: tierColor + '22', color: tierColor, border: `1px solid ${tierColor}44` }}>
            {tierLabel}
          </span>
          {bestPattern && <PatternBadge p={bestPattern} />}
        </div>
        <p className="text-sm text-gray-300 leading-relaxed">{description}</p>
      </div>

      {/* Trade levels */}
      {entry > 0 && (
        <div className="grid grid-cols-3 gap-3">
          {[
            {
              label: direction === 'bearish' ? 'Short Entry' : 'Entry',
              value: entry,
              color: '#f3f4f6',
              note: '',
            },
            {
              label: direction === 'bearish' ? 'Cover Target' : 'Target',
              value: target,
              color: direction === 'bearish' ? '#60a5fa' : '#22c55e',
              note: fmtPct(target, entry),
            },
            {
              label: 'Stop Loss',
              value: stop,
              color: '#ef4444',
              note: fmtPct(stop, entry),
            },
          ].map(({ label, value, color, note }) => (
            <div key={label} className="bg-gray-900 border border-gray-800 rounded-xl p-4 text-center">
              <p className="text-xs text-gray-500 mb-1">{label}</p>
              <p className="text-xl font-mono font-bold" style={{ color }}>${fmt(value)}</p>
              {note && <p className="text-xs mt-0.5" style={{ color }}>{note}</p>}
            </div>
          ))}
        </div>
      )}

      {/* Bearish ETF list — full options for bearish setups */}
      {direction === 'bearish' && etfList.length > 0 && (
        <BearishETFListCard etfList={etfList} onSelect={onShortETF} />
      )}

      {/* Actions */}
      <div className="flex gap-3">
        <button
          onClick={onPaperTrade}
          className="flex-1 py-2.5 rounded-xl text-sm font-bold transition-colors"
          style={{ background: direction === 'bearish' ? '#7f1d1d' : '#14532d', color: direction === 'bearish' ? '#fca5a5' : '#86efac' }}
        >
          {direction === 'bearish' ? '▼ Paper Trade Short' : '▲ Paper Trade Long'}
        </button>
        <button
          onClick={() => navigate(`/chart?symbol=${symbol}`)}
          className="px-4 py-2.5 rounded-xl text-sm font-medium bg-gray-800 text-gray-300 hover:bg-gray-700 transition-colors"
        >
          View Chart →
        </button>
      </div>

      <button onClick={onUpgrade} className="text-xs text-gray-600 hover:text-gray-400 transition-colors">
        I want to understand why ↓
      </button>
    </div>
  )
}

// ── View: Intermediate ────────────────────────────────────────────────────────

function IntermediateView({
  symbol, direction, tier, confluence, stages, entry, addPrice, scale, target, stop,
  patterns, priceLevels, shortETF, etfList, onUpgrade, onPaperTrade, onShortETF,
}: {
  symbol: string
  direction: 'bullish' | 'bearish' | 'neutral'
  tier: string
  confluence: number
  stages: StageResult[]
  entry: number
  addPrice: number
  scale: number
  target: number
  stop: number
  patterns: PatternEntry[]
  priceLevels?: { zones: PriceLevelZones; current_price: number; atr_pct: number }
  shortETF: { etf: string; name: string; leverage: string } | null
  etfList: ETFOption[]
  onUpgrade: () => void
  onPaperTrade: () => void
  onShortETF: (etf?: string) => void
}) {
  const navigate  = useNavigate()
  const tierColor = TIER_COLOR[tier] ?? TIER_COLOR.NONE
  const dirColor  = direction === 'bullish' ? '#22c55e' : direction === 'bearish' ? '#ef4444' : '#9ca3af'
  const activeP   = activePatterns(patterns)

  return (
    <div className="space-y-4 max-w-2xl">
      {/* Header row */}
      <div className="flex items-center gap-4 flex-wrap">
        <span className="text-2xl font-black" style={{ color: dirColor }}>
          {direction === 'bullish' ? '⬆ BULLISH' : direction === 'bearish' ? '⬇ BEARISH' : '→ NEUTRAL'}
        </span>
        <div className="flex items-center gap-2">
          <span className="text-xs px-2 py-0.5 rounded font-semibold"
            style={{ background: tierColor + '22', color: tierColor, border: `1px solid ${tierColor}44` }}>
            {TIER_LABEL[tier] ?? tier}
          </span>
          <span className="text-sm font-mono text-gray-400">{fmt(confluence, 0)}/100</span>
        </div>
        <div className="flex-1 min-w-32">
          <ScoreBar value={confluence} color={tierColor} />
        </div>
      </div>

      {/* 4-Stage checklist */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 space-y-3">
        <p className="text-xs text-gray-500 uppercase tracking-widest font-semibold">Confluence Stages</p>
        {stages.map((st, i) => (
          <div key={i} className="flex items-start gap-3">
            <div className="mt-0.5"><StageDot pass={st.pass} partial={st.partial} na={st.na} /></div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <span className={`text-xs font-semibold ${st.na ? 'text-gray-600' : 'text-gray-200'}`}>Stage {i + 1} — {st.label}</span>
                {st.pass && <span className="text-xs text-emerald-500">✓ confirmed</span>}
                {st.partial && !st.pass && <span className="text-xs text-amber-500">◐ partial</span>}
              </div>
              <p className="text-xs text-gray-500 mt-0.5">{st.detail}</p>
            </div>
          </div>
        ))}
      </div>

      {/* Trade levels */}
      {entry > 0 && (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
          <p className="text-xs text-gray-500 uppercase tracking-widest font-semibold mb-3">
            {direction === 'bearish' ? 'Short Trade Plan' : 'Trade Plan'}
          </p>
          <div className="grid grid-cols-5 gap-2 text-center">
            {[
              {
                label: direction === 'bearish' ? 'Short Entry'   : 'Entry',
                value: entry,
                color: '#f3f4f6',
              },
              {
                label: direction === 'bearish' ? 'Add Short'     : 'Accumulate',
                value: addPrice,
                color: '#60a5fa',
              },
              {
                label: direction === 'bearish' ? 'Partial Cover' : 'Scale',
                value: scale,
                color: '#a78bfa',
              },
              {
                label: direction === 'bearish' ? 'Cover Target'  : 'Distribution',
                value: target,
                color: direction === 'bearish' ? '#60a5fa' : '#22c55e',
              },
              {
                label: 'Stop',
                value: stop,
                color: '#ef4444',
              },
            ].map(({ label, value, color }) => (
              <div key={label}>
                <p className="text-xs text-gray-600 mb-1">{label}</p>
                <p className="text-sm font-mono font-bold" style={{ color }}>${fmt(value)}</p>
                <p className="text-xs mt-0.5" style={{ color: '#6b7280' }}>{fmtPct(value, entry)}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Bearish ETF vehicles list */}
      {direction === 'bearish' && etfList.length > 0 && (
        <BearishETFListCard etfList={etfList} onSelect={onShortETF} />
      )}

      {/* Active patterns */}
      {activeP.length > 0 && (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
          <p className="text-xs text-gray-500 uppercase tracking-widest font-semibold mb-2">Active Patterns</p>
          <div className="flex flex-wrap gap-2">
            {activeP.slice(0, 6).map((p, i) => <PatternBadge key={i} p={p} />)}
          </div>
        </div>
      )}

      {/* Stage 5 — Price Levels */}
      {priceLevels && Object.keys(priceLevels.zones).length > 0 && (
        <PriceLevelZonePanel
          zones={priceLevels.zones}
          currentPrice={priceLevels.current_price}
          atrPct={priceLevels.atr_pct}
          compact={false}
        />
      )}

      {/* Actions */}
      <div className="flex gap-3">
        <button onClick={onPaperTrade}
          className="flex-1 py-2.5 rounded-xl text-sm font-bold transition-colors"
          style={{ background: direction === 'bearish' ? '#7f1d1d' : '#14532d', color: direction === 'bearish' ? '#fca5a5' : '#86efac' }}>
          {direction === 'bearish' ? '▼ Paper Trade Short' : '▲ Paper Trade Long'}
        </button>
        <button onClick={() => navigate(`/chart?symbol=${symbol}`)}
          className="px-4 py-2.5 rounded-xl text-sm font-medium bg-gray-800 text-gray-300 hover:bg-gray-700 transition-colors">
          View Chart →
        </button>
      </div>
      <button onClick={onUpgrade} className="text-xs text-gray-600 hover:text-gray-400 transition-colors">
        Show full breakdown ↓
      </button>
    </div>
  )
}

// ── View: Expert ───────────────────────────────────────────────────────────────

function ExpertView({
  symbol, direction, tier, confluence, stages, scores, entry, addPrice, scale, target, stop,
  patterns, isComp, isBreakout, volRegime, expectedMove, decisionTrace, priceLevels,
  shortETF, etfList, onPaperTrade, onShortETF, isGated,
}: {
  symbol: string
  direction: 'bullish' | 'bearish' | 'neutral'
  tier: string
  confluence: number
  stages: StageResult[]
  scores: Record<string, number>
  entry: number
  addPrice: number
  scale: number
  target: number
  stop: number
  patterns: PatternEntry[]
  isComp: boolean
  isBreakout: boolean
  volRegime: string
  expectedMove: number
  decisionTrace: string
  priceLevels?: { zones: PriceLevelZones; current_price: number; atr_pct: number; levels: PriceLevelCluster[] }
  shortETF: { etf: string; name: string; leverage: string } | null
  etfList: ETFOption[]
  onPaperTrade: () => void
  onShortETF: (etf?: string) => void
  isGated?: boolean
}) {
  const navigate   = useNavigate()
  const tierColor  = TIER_COLOR[tier] ?? TIER_COLOR.NONE
  const dirColor   = direction === 'bullish' ? '#22c55e' : direction === 'bearish' ? '#ef4444' : '#9ca3af'
  const activeP    = activePatterns(patterns)

  const SCORE_ROWS = [
    { key: 'vol_compression', label: 'Vol Compression', weight: '25%' },
    { key: 'breakout',        label: 'Breakout Strength', weight: '20%' },
    { key: 'trend',           label: 'Trend Alignment',   weight: '15%' },
    { key: 'liquidity',       label: 'Liquidity',          weight: '15%' },
    { key: 'pattern',         label: 'Pattern Score',      weight: '10%' },
    { key: 'wyckoff',         label: 'Wyckoff Phase',      weight: '10%' },
    { key: 'gann',            label: 'Gann Projection',    weight:  '5%' },
  ]

  let traceText = ''
  try { traceText = JSON.parse(decisionTrace ?? '{}').text ?? '' } catch { traceText = decisionTrace ?? '' }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center gap-4 flex-wrap">
        <span className="text-2xl font-black" style={{ color: dirColor }}>
          {direction === 'bullish' ? '⬆ BULLISH' : direction === 'bearish' ? '⬇ BEARISH' : '→ NEUTRAL'}
          <span className="text-sm font-mono text-gray-400 ml-3">{fmt(confluence, 1)}/100 · {tier}</span>
        </span>
        <span className="text-xs text-gray-600">{volRegime} · {isComp ? 'COMPRESSED' : isBreakout ? 'BREAKOUT' : 'TRENDING'}</span>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Left col */}
        <div className="space-y-4">
          {/* Stage gates */}
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 space-y-2">
            <div className="flex items-center justify-between">
              <p className="text-xs text-gray-500 uppercase tracking-widest font-semibold">Stage Gates</p>
              {isGated && <span className="text-[10px] text-amber-600 bg-amber-950/40 border border-amber-800/40 rounded px-1.5 py-0.5">Not in pipeline</span>}
            </div>
            {stages.map((st, i) => (
              <div key={i} className="flex items-center gap-2 py-1 border-b border-gray-800/40 last:border-0">
                <StageDot pass={st.pass} partial={st.partial} na={st.na} />
                <span className="text-xs font-mono text-gray-400 w-20 shrink-0">Stage {i + 1}</span>
                <span className={`text-xs flex-1 truncate ${st.na ? 'text-gray-600' : 'text-gray-200'}`}>{st.label}</span>
                <span className={`text-xs ${st.na ? 'text-gray-700' : st.pass ? 'text-emerald-400' : st.partial ? 'text-amber-400' : 'text-gray-600'}`}>
                  {st.na ? 'N/A' : st.pass ? 'PASS' : st.partial ? 'PART' : 'FAIL'}
                </span>
              </div>
            ))}
          </div>

          {/* Component scores */}
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 space-y-2">
            <p className="text-xs text-gray-500 uppercase tracking-widest font-semibold">Component Scores</p>
            {isGated && (
              <p className="text-[10px] text-gray-600 italic">Run pipeline to get AI scores for this symbol</p>
            )}
            {SCORE_ROWS.map(row => {
              const v   = scores[row.key] ?? 0
              const col = isGated ? '#374151' : v >= 70 ? '#22c55e' : v >= 50 ? '#f59e0b' : '#ef4444'
              return (
                <div key={row.key} className="flex items-center gap-2">
                  <span className="text-xs text-gray-500 w-36 shrink-0">{row.label}</span>
                  <div className="flex-1">
                    {isGated
                      ? <div className="h-1.5 bg-gray-800 rounded-full w-full" />
                      : <ScoreBar value={v} color={col} />
                    }
                  </div>
                  <span className="text-xs font-mono w-10 text-right" style={{ color: col }}>
                    {isGated ? '—' : fmt(v, 0)}
                  </span>
                  <span className="text-xs text-gray-700 w-8 text-right">{row.weight}</span>
                </div>
              )
            })}
            <div className="pt-2 border-t border-gray-800 flex items-center justify-between">
              <span className="text-xs font-semibold text-gray-300">CONFLUENCE</span>
              <span className="text-sm font-mono font-bold" style={{ color: (isGated && !confluence) ? '#374151' : tierColor }}>
                {(isGated && !confluence) ? '— / 100' : `${fmt(confluence, 1)} / 100`}
              </span>
            </div>
          </div>
        </div>

        {/* Right col */}
        <div className="space-y-4">
          {/* Trade levels */}
          {entry > 0 && (
            <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
              <div className="flex items-center justify-between mb-3">
                <p className="text-xs text-gray-500 uppercase tracking-widest font-semibold">
                  {direction === 'bearish' ? 'Short Trade Plan' : 'Trade Plan'}
                </p>
                <span className="text-xs text-gray-600">Expected move: {expectedMove > 0 ? fmt(expectedMove, 1) + '%' : '—'}</span>
              </div>
              <div className="space-y-1.5">
                {[
                  {
                    label: direction === 'bearish' ? 'Short Entry'   : 'Entry',
                    value: entry,
                    color: '#f3f4f6',
                    tag:   direction === 'bearish' ? 'enter short'       : 'initial position',
                  },
                  {
                    label: direction === 'bearish' ? 'Add Short'     : 'Accumulate',
                    value: addPrice,
                    color: '#60a5fa',
                    tag:   direction === 'bearish' ? 'add on failed bounce' : 'add on strength',
                  },
                  {
                    label: direction === 'bearish' ? 'Partial Cover' : 'Scale',
                    value: scale,
                    color: '#a78bfa',
                    tag:   direction === 'bearish' ? 'first support'     : 'scale out zone',
                  },
                  {
                    label: direction === 'bearish' ? 'Cover Target'  : 'Distribution',
                    value: target,
                    color: direction === 'bearish' ? '#60a5fa' : '#22c55e',
                    tag:   direction === 'bearish' ? 'full cover'        : 'full target',
                  },
                  {
                    label: 'Stop',
                    value: stop,
                    color: '#ef4444',
                    tag:   'invalidation',
                  },
                ].map(({ label, value, color, tag }) => (
                  <div key={label} className="flex items-center gap-3">
                    <span className="text-xs text-gray-500 w-24 shrink-0">{label}</span>
                    <span className="text-sm font-mono font-bold flex-1" style={{ color }}>${fmt(value)}</span>
                    <span className="text-xs text-gray-700">{fmtPct(value, entry)}</span>
                    <span className="text-xs text-gray-600 hidden lg:block">{tag}</span>
                  </div>
                ))}
              </div>
              <div className="mt-2 pt-2 border-t border-gray-800 text-xs text-gray-600">
                R:R = {entry && stop && target ? fmt(Math.abs(target - entry) / Math.abs(entry - stop), 1) + ':1' : '—'}
              </div>
            </div>
          )}

          {/* Bearish ETF vehicles — right-hand card */}
          {direction === 'bearish' && etfList.length > 0 && (
            <BearishETFListCard etfList={etfList} onSelect={onShortETF} />
          )}

          {/* All patterns */}
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
            <p className="text-xs text-gray-500 uppercase tracking-widest font-semibold mb-3">
              All Patterns ({patterns.length})
            </p>
            {patterns.length === 0 ? (
              <p className="text-xs text-gray-600">None detected</p>
            ) : (
              <div className="space-y-1.5">
                {patterns.map((p, i) => (
                  <div key={i} className="flex items-center gap-2 text-xs">
                    <span className={`w-16 shrink-0 ${
                      p.status === 'BREAKOUT' ? 'text-emerald-400' :
                      p.status === 'READY'    ? 'text-amber-400'   :
                      p.status === 'FORMING'  ? 'text-gray-400'    : 'text-gray-600'
                    }`}>{p.status}</span>
                    <span className="text-gray-300 flex-1 truncate">{p.pattern_name}</span>
                    <span className={`${p.direction === 'bullish' ? 'text-emerald-500' : p.direction === 'bearish' ? 'text-red-400' : 'text-gray-500'}`}>
                      {p.direction === 'bullish' ? '↑' : p.direction === 'bearish' ? '↓' : '→'}
                    </span>
                    <span className="text-gray-600 font-mono">{fmt(p.confidence ?? p.probability, 0)}%</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Stage 5 — Clustered Price Levels (expert: full table + top clusters) */}
      {priceLevels && (
        <div className="space-y-3">
          <PriceLevelZonePanel
            zones={priceLevels.zones}
            currentPrice={priceLevels.current_price}
            atrPct={priceLevels.atr_pct}
            compact={false}
          />
          {priceLevels.levels.length > 0 && (
            <details className="bg-gray-950 border border-gray-800 rounded-xl overflow-hidden">
              <summary className="px-4 py-3 text-xs text-gray-500 cursor-pointer hover:text-gray-300 uppercase tracking-widest font-semibold">
                All {priceLevels.levels.length} Clusters ↓
              </summary>
              <div className="px-4 pb-4 space-y-1">
                {priceLevels.levels.map((cl, i) => (
                  <div key={i} className="flex items-center gap-3 text-xs py-1 border-b border-gray-800/30 last:border-0">
                    <span className="font-mono font-bold text-gray-100 w-20 shrink-0">${cl.price.toFixed(2)}</span>
                    <StrengthBar value={cl.strength} />
                    <span className={`text-[10px] w-12 shrink-0 font-mono ${cl.type === 'support' ? 'text-emerald-600' : 'text-red-500'}`}>
                      {cl.distance_pct > 0 ? '+' : ''}{cl.distance_pct.toFixed(1)}%
                    </span>
                    <span className="text-gray-600 truncate">{cl.sources.join(' · ')}</span>
                  </div>
                ))}
              </div>
            </details>
          )}
        </div>
      )}

      {/* Decision trace */}
      {traceText && (
        <details className="bg-gray-950 border border-gray-800 rounded-xl overflow-hidden">
          <summary className="px-4 py-3 text-xs text-gray-500 cursor-pointer hover:text-gray-300 uppercase tracking-widest font-semibold">
            Decision Trace ↓
          </summary>
          <pre className="px-4 pb-4 text-xs text-gray-500 font-mono overflow-x-auto whitespace-pre-wrap leading-relaxed">
            {traceText}
          </pre>
        </details>
      )}

      {/* Actions */}
      <div className="flex gap-3">
        <button onClick={onPaperTrade}
          className="flex-1 py-2.5 rounded-xl text-sm font-bold transition-colors"
          style={{ background: direction === 'bearish' ? '#7f1d1d' : '#14532d', color: direction === 'bearish' ? '#fca5a5' : '#86efac' }}>
          {direction === 'bearish' ? '▼ Paper Trade Short' : '▲ Paper Trade Long'}
        </button>
        <button onClick={() => navigate(`/chart?symbol=${symbol}`)}
          className="px-4 py-2.5 rounded-xl text-sm font-medium bg-gray-800 text-gray-300 hover:bg-gray-700 transition-colors">
          View Chart →
        </button>
      </div>
    </div>
  )
}

// ── Page ──────────────────────────────────────────────────────────────────────

const RECENT_KEY = 'roboalgo_recent_intel'
const MODE_KEY   = 'roboalgo_intel_mode'

// Quick-pick default list — underlying stocks AND non-leveraged 1× ETFs.
// Valid: stocks, sector ETFs (XLF, XLE, XLK, XLP), commodity ETFs (GLD, SLV),
//        index ETFs (SPY, QQQ, IWM), country/theme ETFs (FXI, IBB, GDX).
// NOT valid: leveraged bull/bear ETFs (SOXL, TQQQ, UPRO, SOXS, SQQQ, SPXS, etc.)
// Those are suggested as TRADE VEHICLES when the analysis warrants it.
const RECENT_TICKERS = [
  'NVDA','AAPL','TSLA','AMD','MSFT','SPY','QQQ','GLD','SLV','XLF','XLE','IWM',
]

export default function TickerIntelPage() {
  const navigate         = useNavigate()
  const [searchParams]   = useSearchParams()
  const initSymbol       = (searchParams.get('symbol') || 'SPY').toUpperCase()

  const [inputVal, setInputVal] = useState(initSymbol)
  const [symbol,   setSymbol]   = useState(initSymbol)
  const [mode,     setMode]     = useState<ViewMode>(
    () => (localStorage.getItem(MODE_KEY) as ViewMode) ?? 'intermediate'
  )
  const [recentSymbols, setRecentSymbols] = useState<string[]>(
    () => JSON.parse(localStorage.getItem(RECENT_KEY) ?? '[]')
  )
  const inputRef = useRef<HTMLInputElement>(null)

  const { data: confluence, isLoading: confLoading, isError: confError } = useConfluenceScore(symbol)
  const { data: patterns,   isLoading: patLoading                       } = usePatterns(symbol)
  const { data: priceLevels }                                             = usePriceLevels(symbol)

  const isLoading = (confLoading || patLoading) && !!symbol

  function submitSymbol(sym: string) {
    const s = sym.trim().toUpperCase()
    if (!s) return
    setSymbol(s)
    setInputVal(s)
    const updated = [s, ...recentSymbols.filter(r => r !== s)].slice(0, 8)
    setRecentSymbols(updated)
    localStorage.setItem(RECENT_KEY, JSON.stringify(updated))
  }

  function changeMode(m: ViewMode) {
    setMode(m)
    localStorage.setItem(MODE_KEY, m)
  }

  // Derived data
  const scores       = confluence?.component_scores ?? {}
  const rawTier      = confError ? 'NONE' : (confluence?.signal_tier ?? 'NONE')
  const rawConf      = confluence?.confluence_score ?? 0
  const direction    = inferDirection(scores, patterns, confluence?.is_breakout ?? false)
  const isShort      = direction === 'bearish'
  const activeP      = activePatterns(patterns)

  // Detect symbols not in pipeline: API returns conf=0 with all-zero component scores
  const hasNoPipelineData = !confError && rawConf === 0
    && Object.values(scores).every(v => (v as number ?? 0) === 0)

  // Pattern-based fallback score for out-of-pipeline symbols that have active patterns.
  // Confirmed (BREAKOUT/READY/COMPLETED) patterns each contribute to a base score;
  // average pattern confidence contributes 22% of the final value; capped at 68.
  const patternFallbackScore: number | null = hasNoPipelineData && activeP.length > 0
    ? (() => {
        const confirmed = activeP.filter(
          p => ['BREAKOUT', 'COMPLETED', 'READY'].includes(p.status ?? '')
        )
        const avgConf = activeP.reduce(
          (s, p) => s + (p.confidence ?? p.probability ?? 0), 0
        ) / activeP.length
        const base = confirmed.length >= 2 ? 50 : confirmed.length === 1 ? 40 : 25
        return Math.min(68, Math.round(base + avgConf * 0.22))
      })()
    : null

  // Use fallback score when available; otherwise use the real pipeline score
  const conf = patternFallbackScore ?? rawConf

  // For bearish setups the API tier is bullish-framed (low score = NO SIGNAL on API).
  // Invert: bearish strength = 100 - conf. 46/100 bullish → 54/100 bearish = MEDIUM.
  // There is always a trade — bearish minimum tier is WATCH.
  // Guard: if confError we have no real score, don't compute a false tier.
  const tier = patternFallbackScore != null
    ? (patternFallbackScore >= 60 ? 'MEDIUM' : 'WATCH')
    : confError ? 'NONE' : isShort ? computeBearishTier(rawConf) : rawTier

  // isGated covers: explicitly gated by API, fetch error, or no pipeline data.
  // When gated, component-score rows show "—" and stages 1–3 show N/A.
  const isGated = !!(confluence?.gated) || !!(confError) || hasNoPipelineData

  // Pass direction to computeStages so bearish criteria are applied
  const stages    = computeStages(
    scores, confluence?.is_compression ?? false,
    confluence?.is_breakout ?? false, tier, patterns, direction, isGated
  )
  const bestPat   = isShort
    ? (activeP.find(p => p.direction === 'bearish' && ['BREAKOUT','READY'].includes(p.status ?? '')) ?? activeP[0])
    : (activeP.find(p => ['BREAKOUT','READY'].includes(p.status ?? '')) ?? activeP[0])

  // Raw ATR levels from confluence engine (bullish framing)
  const rawEntry  = confluence?.entry_price    ?? 0
  const rawAdd    = confluence?.add_price      ?? 0
  const rawScale  = confluence?.scale_price    ?? 0
  const rawTarget = confluence?.target_price   ?? 0
  const rawStop   = confluence?.stop_price     ?? 0

  // For SHORT trades: invert the plan
  //   Short Entry  = same entry level (resistance area)
  //   Short Add    = scale_price (add more short on failed bounce to higher resistance)
  //   Short Scale  = add_price (partial cover at first support below)
  //   Short Target = stop_price (bull stop = bear cover target — key support)
  //   Short Stop   = target_price (bull target = bear stop — invalidation above)
  const entry    = rawEntry
  const addPrice = isShort ? rawScale   : rawAdd
  const scale    = isShort ? rawAdd     : rawScale
  const target   = isShort ? rawStop    : rawTarget
  const stop     = isShort ? rawTarget  : rawStop

  const shortETF  = isShort ? getShortETF(symbol) : null
  const etfList   = isShort ? getBearishETFList(symbol) : []

  // 10%+ minimum move check — both long AND short setups must project ≥10% to qualify.
  // For leveraged ETF shorts: the underlying move × leverage = effective ETF move.
  //   e.g. NVDA 5% move × SOXS 3x = 15% effective → qualifies
  const movePct        = entry > 0 && target > 0 ? Math.abs((target - entry) / entry * 100) : 0
  const leverageMult   = shortETF
    ? parseFloat(shortETF.leverage.replace('x', '')) || 1
    : 1
  const effectiveMovePct   = movePct * leverageMult
  const meetsMoveMinimum   = effectiveMovePct >= 10

  const description = buildNoviceDescription(direction, confluence?.is_compression ?? false,
    confluence?.is_breakout ?? false, scores, bestPat, entry, target)

  function handlePaperTrade() {
    const dir = isShort ? 'sell' : 'buy'
    navigate(`/paper?symbol=${symbol}&direction=${dir}&entry=${fmt(entry)}&stop=${fmt(stop)}&target=${fmt(target)}`)
  }

  function handleShortETFTrade(etfSymbol?: string) {
    const sym = etfSymbol ?? shortETF?.etf
    if (!sym) return
    navigate(`/intel?symbol=${sym}`)
  }

  return (
    <div className="space-y-5">
      {/* ── Page header ── */}
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div>
          <h2 className="text-2xl font-black text-white tracking-tight">TICKER INTELLIGENCE</h2>
          <p className="text-xs text-gray-500 mt-0.5">Pull any ticker — all 5 stages, every timeframe</p>
        </div>

        {/* View mode selector */}
        <div className="flex gap-1 bg-gray-900 border border-gray-800 rounded-xl p-1">
          {(['novice', 'intermediate', 'expert'] as ViewMode[]).map(m => (
            <button key={m}
              onClick={() => changeMode(m)}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all capitalize ${
                mode === m ? 'bg-emerald-700 text-white' : 'text-gray-500 hover:text-gray-300'
              }`}
            >
              {m === 'novice' ? '🟢 Simple' : m === 'intermediate' ? '🟡 Standard' : '🔵 Full'}
            </button>
          ))}
        </div>
      </div>

      {/* ── Search bar ── */}
      <div className="flex gap-2 max-w-lg">
        <input
          ref={inputRef}
          value={inputVal}
          onChange={e => setInputVal(e.target.value.toUpperCase())}
          onKeyDown={e => { if (e.key === 'Enter') submitSymbol(inputVal) }}
          placeholder="Type a ticker — AAPL, NVDA, SPY…"
          className="flex-1 bg-gray-900 border border-gray-700 rounded-xl px-4 py-2.5 text-sm font-mono text-gray-200 focus:outline-none focus:border-emerald-500 placeholder-gray-600"
        />
        <button
          onClick={() => submitSymbol(inputVal)}
          className="px-5 py-2.5 bg-emerald-700 hover:bg-emerald-600 text-white text-sm font-bold rounded-xl transition-colors"
        >
          Analyse
        </button>
      </div>

      {/* Recent / quick picks */}
      <div className="flex flex-wrap gap-1.5">
        {(recentSymbols.length > 0 ? recentSymbols : RECENT_TICKERS.slice(0, 8)).map(s => (
          <button key={s}
            onClick={() => submitSymbol(s)}
            className={`px-2.5 py-1 rounded-lg text-xs font-mono transition-colors ${
              s === symbol
                ? 'bg-emerald-800/60 text-emerald-300 border border-emerald-700/40'
                : 'bg-gray-800/80 text-gray-400 hover:text-gray-200 border border-gray-700/30'
            }`}
          >
            {s}
          </button>
        ))}
      </div>

      {/* ── Content area ── */}
      {!symbol && (
        <div className="flex flex-col items-center justify-center py-24 text-center">
          <div className="text-5xl mb-4">🔍</div>
          <p className="text-gray-400 text-lg font-semibold">Enter a ticker above</p>
          <p className="text-gray-600 text-sm mt-1">Get the full 5-stage analysis for any stock or ETF</p>
        </div>
      )}

      {isLoading && symbol && (
        <div className="flex items-center gap-3 py-12">
          <div className="w-5 h-5 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin" />
          <span className="text-gray-400 text-sm">Analysing {symbol}…</span>
        </div>
      )}

      {!isLoading && symbol && (
        <div>
          <div className="mb-4 flex items-baseline gap-3 flex-wrap">
            <span className="text-3xl font-black text-white tracking-tight">{symbol}</span>
            {confluence?.date && (
              <span className="text-xs text-gray-600">as of {confluence.date}</span>
            )}
            {confError && (
              <span className="text-xs text-gray-600 border border-gray-700 rounded px-2 py-0.5">
                ◦ AI scoring unavailable — showing patterns &amp; price levels
              </span>
            )}
          </div>

          {/* ── Move quality gate ── */}
          {entry > 0 && shortETF && movePct > 0 && movePct < 10 && effectiveMovePct >= 10 && (
            /* Direct stock trade is < 10%, but leveraged ETF amplifies to ≥ 10% → suggest ETF */
            <div className="mb-4 flex items-start gap-3 bg-red-950/30 border border-red-700/50 rounded-xl px-4 py-3 max-w-2xl">
              <span className="text-red-400 text-xl shrink-0">⬇</span>
              <div className="flex-1">
                <p className="text-xs font-bold text-red-300">
                  Direct move {movePct.toFixed(1)}% × {shortETF.leverage} {shortETF.etf} = <span className="text-red-200">{effectiveMovePct.toFixed(1)}%</span> — trade the ETF
                </p>
                <p className="text-xs text-red-500/70 mt-0.5">
                  Stock move is small, but {shortETF.etf} ({shortETF.name}) amplifies it to a qualifying {effectiveMovePct.toFixed(1)}% trade.
                  Direct short on {symbol} is marginal — use {shortETF.etf} instead.
                </p>
              </div>
            </div>
          )}
          {entry > 0 && !meetsMoveMinimum && (movePct >= 10 || !shortETF || effectiveMovePct < 10) && (
            /* Even with leverage, move is too small — compressed range */
            <div className="mb-4 flex items-start gap-3 bg-amber-950/30 border border-amber-800/40 rounded-xl px-4 py-3 max-w-2xl">
              <span className="text-amber-400 text-lg shrink-0">⚠</span>
              <div>
                <p className="text-xs font-bold text-amber-300">
                  {isShort ? '↓ Bearish:' : '↑ Bullish:'} Projected move {leverageMult > 1 ? `${movePct.toFixed(1)}% × ${leverageMult}x = ${effectiveMovePct.toFixed(1)}%` : `${movePct.toFixed(1)}%`} — range is compressed
                </p>
                <p className="text-xs text-amber-600 mt-0.5">
                  {isShort
                    ? `Minimum 10% needed. Range is tight — price needs to break down further before a full ${isShort ? 'short' : 'long'} entry qualifies.`
                    : 'Minimum 10% required. Wait for volatility to expand before entering.'}
                </p>
              </div>
            </div>
          )}

          {mode === 'novice' && (
            <NoviceView
              symbol={symbol}
              direction={direction}
              tier={tier}
              entry={entry}
              target={target}
              stop={stop}
              description={description}
              bestPattern={bestPat}
              shortETF={shortETF}
              etfList={etfList}
              onUpgrade={() => changeMode('intermediate')}
              onPaperTrade={handlePaperTrade}
              onShortETF={handleShortETFTrade}
            />
          )}

          {mode === 'intermediate' && (
            <IntermediateView
              symbol={symbol}
              direction={direction}
              tier={tier}
              confluence={conf}
              stages={stages}
              entry={entry}
              addPrice={addPrice}
              scale={scale}
              target={target}
              stop={stop}
              patterns={patterns ?? []}
              priceLevels={priceLevels}
              shortETF={shortETF}
              etfList={etfList}
              onUpgrade={() => changeMode('expert')}
              onPaperTrade={handlePaperTrade}
              onShortETF={handleShortETFTrade}
            />
          )}

          {mode === 'expert' && (
            <ExpertView
              symbol={symbol}
              direction={direction}
              tier={tier}
              confluence={conf}
              stages={stages}
              scores={scores as Record<string, number>}
              entry={entry}
              addPrice={addPrice}
              scale={scale}
              target={target}
              stop={stop}
              patterns={patterns ?? []}
              isComp={confluence?.is_compression ?? false}
              isBreakout={confluence?.is_breakout ?? false}
              volRegime={confluence?.volatility_regime ?? ''}
              expectedMove={confluence?.expected_move_pct ?? 0}
              decisionTrace={confluence?.decision_trace ?? ''}
              priceLevels={priceLevels}
              shortETF={shortETF}
              etfList={etfList}
              onPaperTrade={handlePaperTrade}
              onShortETF={handleShortETFTrade}
              isGated={isGated}
            />
          )}
        </div>
      )}
    </div>
  )
}
