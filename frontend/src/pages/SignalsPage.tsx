/**
 * Signals — unified hub for all signal types.
 *
 * Tabs:
 *   Confluence  — staged funnel: Full → Near → Developing → Watching
 *   Rocket      — 5-stage scanner: patterns + gamma + volatility coupled
 *   Gamma       — GEX / VEX / DEX explorer per symbol
 *   Patterns    — batch pattern scanner with TF selector
 *   AI Signals  — existing recommendation engine
 *   Probability — XGBoost gauges
 */

import { useState, useMemo, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import RecommendationPage from './RecommendationPage'
import ProbabilityPage    from './ProbabilityPage'
import PatternScanPage    from './PatternScanPage'
import WatchlistPage      from './WatchlistPage'
import { useConfluenceScore } from '../api/hooks'
import api from '../api/client'
import { useQuery } from '@tanstack/react-query'
import type { ConfluenceResult } from '../api/hooks'
import {
  BarChart, Bar, Cell, XAxis, YAxis, CartesianGrid,
  Tooltip, ReferenceLine, ResponsiveContainer, Legend,
} from 'recharts'

// ── Types ─────────────────────────────────────────────────────────────────────

type Tab = 'confluence' | 'sniper-neg-gamma' | 'sniper-technical' | 'gamma' | 'patterns' | 'recommendation' | 'probability' | 'watchlist'

type Profile = 'scalper' | 'daytrader' | 'swing' | 'investor' | 'all'

const PROFILE_LABEL: Record<Profile, string> = {
  scalper:   '🎯 Scalper',
  daytrader: '⚡ Day Trader',
  swing:     '📈 Swing',
  investor:  '💼 Investor',
  all:       '🌐 All',
}

// Funnel stage: ALL 4 stages passed = FULL CONFLUENCE
type FunnelBucket = 'full' | 'near' | 'developing' | 'watching'

interface FunnelEntry {
  symbol:     string
  tier:       string
  score:      number
  isComp:     boolean
  isBreakout: boolean
  stages:     boolean[]  // [s1, s2, s3, s4] pass/fail
  stagesPass: number
  bucket:     FunnelBucket
  volRegime:  string
  entry:      number
  target:     number
  stop:       number
  compScores: Record<string, number>
}

function computeBucket(stages: boolean[]): FunnelBucket {
  const pass = stages.filter(Boolean).length
  if (pass === 4) return 'full'
  if (pass === 3) return 'near'
  if (pass >= 2)  return 'developing'
  return 'watching'
}

function tierToScore(tier: string): number {
  return tier === 'HIGH' ? 90 : tier === 'MEDIUM' ? 70 : tier === 'WATCH' ? 55 : 30
}

const TIER_COLOR: Record<string, string> = {
  HIGH:   '#22c55e',
  MEDIUM: '#f59e0b',
  WATCH:  '#60a5fa',
  NONE:   '#6b7280',
}

// ── Top-signals hook (uses existing /confluence/top) ─────────────────────────

function useTopSignals(tier = 'WATCH', limit = 50) {
  return useQuery<ConfluenceResult[]>({
    queryKey: ['confluence', 'top', tier, limit],
    queryFn:  () => api.get('/confluence/top', { params: { tier, limit } }).then(r => r.data?.signals ?? []),
    staleTime: 2 * 60_000,
  })
}

// ── Profile filter definitions ────────────────────────────────────────────────
// Each profile maps to a set of component-score thresholds that reflect the
// trader's time-horizon and risk appetite.
const PROFILE_FILTERS: Record<Profile, (e: FunnelEntry) => boolean> = {
  // Scalper — needs explosive setups: active breakout OR very tight compression
  scalper:   e => e.isBreakout || (e.compScores.vol_compression ?? 0) >= 72,

  // Day Trader — breakout imminent or compression + momentum building
  daytrader: e => e.isBreakout
                  || ((e.compScores.vol_compression ?? 0) >= 62
                      && (e.compScores.breakout ?? 0) >= 45),

  // Swing — balanced setup: trend aligned + pattern/wyckoff developing
  swing:     e => (e.compScores.trend ?? 0) >= 55
                  && ((e.compScores.pattern ?? 0) >= 50
                      || (e.compScores.wyckoff ?? 0) >= 50),

  // Investor — strong long-term trend + structural phase confirmation
  investor:  e => (e.compScores.trend ?? 0) >= 62
                  && (e.compScores.wyckoff ?? 0) >= 55,

  // All — no filter
  all:       _e => true,
}

// ── Confluence Funnel ─────────────────────────────────────────────────────────

function ConfluenceFunnel() {
  const navigate = useNavigate()
  const [profile, setProfile] = useState<Profile>('all')

  const { data: rawSignals = [], isLoading } = useTopSignals('WATCH', 60)

  // Map raw confluence results → funnel entries with stage gate logic
  const entries = useMemo<FunnelEntry[]>(() => {
    return rawSignals.map((r: any) => {
      const s = r.component_scores ?? {}

      // Stage 1 — Regime: vol_compression > 55 OR compressed OR breakout
      const s1 = (s.vol_compression ?? 0) > 55 || r.is_compression || r.is_breakout
      // Stage 2 — Core signals: trend > 55 AND (breakout > 40 OR liquidity > 55)
      const s2 = (s.trend ?? 0) > 55 && ((s.breakout ?? 0) > 40 || (s.liquidity ?? 0) > 55)
      // Stage 3 — AI probability: MEDIUM or HIGH
      const s3 = r.signal_tier === 'HIGH' || r.signal_tier === 'MEDIUM'
      // Stage 4 — Pattern: pattern score or wyckoff > 60
      const s4 = (s.pattern ?? 0) > 60 || (s.wyckoff ?? 0) > 60

      const stages = [s1, s2, s3, s4]
      const bucket = computeBucket(stages)

      return {
        symbol:     r.symbol,
        tier:       r.signal_tier ?? 'NONE',
        score:      r.confluence_score ?? 0,
        isComp:     r.is_compression ?? false,
        isBreakout: r.is_breakout ?? false,
        stages,
        stagesPass: stages.filter(Boolean).length,
        bucket,
        volRegime:  r.volatility_regime ?? '',
        entry:      r.entry_price ?? 0,
        target:     r.target_price ?? 0,
        stop:       r.stop_price ?? 0,
        compScores: s,
      }
    }).sort((a, b) => {
      // Sort: full first, then by stages count, then score
      if (a.stagesPass !== b.stagesPass) return b.stagesPass - a.stagesPass
      return b.score - a.score
    })
  }, [rawSignals])

  // Apply profile filter
  const filtered = useMemo<FunnelEntry[]>(
    () => entries.filter(PROFILE_FILTERS[profile]),
    [entries, profile]
  )

  const BUCKETS: { key: FunnelBucket; label: string; color: string; desc: string }[] = [
    { key: 'full',       label: 'Full Confluence',     color: '#22c55e', desc: 'All 4 stages confirmed — trade-ready' },
    { key: 'near',       label: 'Near Confluence',     color: '#f59e0b', desc: 'Stage 1–3 confirmed — pattern emerging' },
    { key: 'developing', label: 'Developing',          color: '#60a5fa', desc: 'Stage 1–2 — signals building' },
    { key: 'watching',   label: 'Watching',            color: '#6b7280', desc: 'Early stage — monitor only' },
  ]

  if (isLoading) return (
    <div className="flex items-center gap-3 py-16">
      <div className="w-5 h-5 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin" />
      <span className="text-gray-400 text-sm">Loading confluence data…</span>
    </div>
  )

  return (
    <div className="space-y-4">
      {/* Profile selector */}
      <div className="flex items-center gap-2 flex-wrap">
        <span className="text-xs text-gray-600 uppercase tracking-widest">Trader profile:</span>
        <div className="flex gap-1 bg-gray-900 border border-gray-800 rounded-xl p-0.5 flex-wrap">
          {(Object.keys(PROFILE_LABEL) as Profile[]).map(p => (
            <button key={p}
              onClick={() => setProfile(p)}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                profile === p ? 'bg-emerald-700 text-white' : 'text-gray-500 hover:text-gray-300'
              }`}>
              {PROFILE_LABEL[p]}
            </button>
          ))}
        </div>
        <span className="text-xs text-gray-600 ml-2">
          {filtered.filter(e => e.bucket === 'full').length} full · {filtered.filter(e => e.bucket === 'near').length} near · {filtered.length} total
        </span>
      </div>

      {/* Funnel sections */}
      {BUCKETS.map(bucket => {
        const rows = filtered.filter(e => e.bucket === bucket.key)
        if (rows.length === 0) return null
        return (
          <div key={bucket.key} className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
            {/* Bucket header */}
            <div className="px-4 py-2.5 border-b border-gray-800 flex items-center justify-between"
              style={{ borderLeft: `3px solid ${bucket.color}` }}>
              <div className="flex items-center gap-3">
                <span className="text-sm font-bold" style={{ color: bucket.color }}>{bucket.label}</span>
                <span className="text-xs text-gray-600">{bucket.desc}</span>
              </div>
              <span className="text-xs font-mono text-gray-600">{rows.length} symbols</span>
            </div>

            {/* Rows */}
            <div className="divide-y divide-gray-800/40">
              {rows.map(e => {
                const tierCol = TIER_COLOR[e.tier] ?? TIER_COLOR.NONE
                return (
                  <div key={e.symbol}
                    className="grid items-center gap-3 px-4 py-3 hover:bg-gray-800/30 transition-colors cursor-pointer"
                    style={{ gridTemplateColumns: '80px 120px 1fr 100px 100px' }}
                    onClick={() => navigate(`/intel?symbol=${e.symbol}`)}
                  >
                    {/* Symbol */}
                    <span className="font-mono font-bold text-sm text-gray-100">{e.symbol}</span>

                    {/* Score + tier */}
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-mono font-bold" style={{ color: tierCol }}>
                        {e.score.toFixed(0)}
                      </span>
                      <span className="text-[10px] px-1.5 py-0.5 rounded"
                        style={{ background: tierCol + '20', color: tierCol, border: `1px solid ${tierCol}44` }}>
                        {e.tier}
                      </span>
                    </div>

                    {/* Stage dots */}
                    <div className="flex items-center gap-2">
                      {e.stages.map((pass, i) => (
                        <div key={i} className="flex items-center gap-1">
                          <span className={`text-[10px] ${pass ? 'text-emerald-400' : 'text-gray-700'}`}>
                            {pass ? '●' : '○'}
                          </span>
                          <span className="text-[9px] text-gray-700">S{i + 1}</span>
                        </div>
                      ))}
                      {e.isBreakout && (
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-900/40 text-emerald-400 border border-emerald-800/30 ml-1">BREAKOUT</span>
                      )}
                      {e.isComp && !e.isBreakout && (
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-blue-900/30 text-blue-400 border border-blue-800/20 ml-1">COMPRESSED</span>
                      )}
                    </div>

                    {/* Entry */}
                    <div className="text-xs font-mono text-right">
                      {e.entry > 0 ? (
                        <>
                          <span className="text-gray-400">${e.entry.toFixed(2)}</span>
                          {e.target > 0 && <span className="text-emerald-600"> → ${e.target.toFixed(2)}</span>}
                        </>
                      ) : <span className="text-gray-700">—</span>}
                    </div>

                    {/* Arrow */}
                    <div className="text-right">
                      <span className="text-xs text-gray-600 hover:text-emerald-400 transition-colors">Intel →</span>
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        )
      })}

      {filtered.length === 0 && (
        <div className="py-16 text-center text-gray-600">
          <p>No signals above threshold right now.</p>
          <p className="text-xs mt-1">Run pipeline to refresh confluence scores.</p>
        </div>
      )}
    </div>
  )
}

// ── Rocket Scanner Tab ────────────────────────────────────────────────────────

function RocketScannerPanel() {
  const navigate = useNavigate()
  const [expandedSym, setExpandedSym] = useState<string | null>(null)
  const [trackedSymbols, setTrackedSymbols] = useState<Set<string>>(new Set())
  const [filterTracked, setFilterTracked] = useState(false)
  const [priceFilter, setPriceFilter] = useState<'penny' | 'all'>('all') // penny = <$5, all = no limit
  const [sortBy, setSortBy] = useState<'score' | 'price' | 'rr'>('score')

  const { data: statusData } = useQuery({
    queryKey: ['rocket', 'status'],
    queryFn: () => api.get('/rocket/status').then(r => r.data),
    refetchInterval: 30_000,
  })

  // Auto-fetch on mount — Gamma tracker returns hundreds of negative gamma candidates
  const { data: scanData, isLoading } = useQuery<any[]>({
    queryKey: ['gamma', 'tracker'],
    queryFn: async () => {
      const r = await api.get('/gamma-tracker/scan', { params: { top_n: 5000, min_volume: 0 } })
      const results = r.data?.results ?? []
      console.log('Gamma tracker scan:', { count: results.length, firstResult: results[0]?.symbol })
      return results
    },
    staleTime: 5 * 60_000,
  })

  const { data: gexData } = useQuery({
    queryKey: ['rocket', 'gex', expandedSym],
    queryFn: () => expandedSym
      ? api.get(`/rocket/gex/${expandedSym}`, { params: { include_vex: true } }).then(r => r.data)
      : null,
    enabled: !!expandedSym,
    staleTime: 15 * 60_000,
  })

  const SCORE_COLOR = (s: number) =>
    s >= 80 ? '#22c55e' : s >= 65 ? '#f59e0b' : s >= 50 ? '#60a5fa' : '#6b7280'

  const REGIME_STYLE: Record<string, { bg: string; text: string }> = {
    EXPLOSIVE:     { bg: 'bg-red-900/40',    text: 'text-red-400' },
    PINNED:        { bg: 'bg-emerald-900/30', text: 'text-emerald-400' },
    TRENDING:      { bg: 'bg-amber-900/30',   text: 'text-amber-400' },
    TRANSITIONING: { bg: 'bg-blue-900/30',    text: 'text-blue-400' },
    HEDGING:       { bg: 'bg-purple-900/30',  text: 'text-purple-400' },
    NEGATIVE:      { bg: 'bg-red-900/30',     text: 'text-red-400' },
    POSITIVE:      { bg: 'bg-emerald-900/30', text: 'text-emerald-400' },
  }
  const regimeStyle = (r?: string) => {
    const s = REGIME_STYLE[r ?? '']
    return s ? `${s.bg} ${s.text}` : 'bg-gray-800/50 text-gray-500'
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <p className="text-xs text-gray-500">5-stage pipeline: Universe → Squeeze → Patterns → Options → Rank</p>
          {statusData?.last_scan_age_s != null && (
            <p className="text-[11px] text-gray-600 mt-0.5">
              Last scan {Math.round(statusData.last_scan_age_s / 60)}m ago · {statusData.cached_count} cached
            </p>
          )}
        </div>
        {isLoading && (
          <div className="flex items-center gap-2 text-xs text-gray-500">
            <span className="w-3.5 h-3.5 border-2 border-emerald-600 border-t-transparent rounded-full animate-spin" />
            Scanning…
          </div>
        )}
      </div>

      {/* Loading skeleton */}
      {isLoading && (
        <div className="space-y-2">
          {[...Array(5)].map((_, i) => (
            <div key={i} className="h-12 bg-gray-900 border border-gray-800 rounded-xl animate-pulse" />
          ))}
        </div>
      )}

      {/* Filters & Sort */}
      {scanData && scanData.length > 0 && (
        <div className="flex items-center gap-4 mb-3 flex-wrap">
          {/* Tracked filter */}
          <label className="flex items-center gap-2 text-sm cursor-pointer">
            <input
              type="checkbox"
              checked={filterTracked}
              onChange={(e) => setFilterTracked(e.target.checked)}
              className="w-4 h-4"
            />
            <span className="text-gray-400">Tracked only ({trackedSymbols.size})</span>
          </label>

          {/* Price filter */}
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-600">Price:</span>
            <select
              value={priceFilter}
              onChange={(e) => setPriceFilter(e.target.value as 'penny' | 'all')}
              className="px-2 py-1 text-xs bg-gray-800 border border-gray-700 rounded text-gray-300"
            >
              <option value="penny">Penny (&lt;$5)</option>
              <option value="all">All Prices</option>
            </select>
          </div>

          {/* Sort by */}
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-600">Sort:</span>
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as 'score' | 'price' | 'rr')}
              className="px-2 py-1 text-xs bg-gray-800 border border-gray-700 rounded text-gray-300"
            >
              <option value="score">Rocket Score ↓</option>
              <option value="rr">Risk:Reward ↓</option>
              <option value="price">Price ↑</option>
            </select>
          </div>
        </div>
      )}

      {/* Results — all gamma squeeze candidates, sorted & filtered */}
      {scanData && scanData.length > 0 && (() => {
        let filtered = scanData
          // Price filter
          .filter(c => {
            const price = c.current_price ?? 0
            if (priceFilter === 'penny') return price > 0 && price < 5
            return true
          })
          // Tracked filter
          .filter(c => !filterTracked || trackedSymbols.has(c.symbol))

        // Sort
        if (sortBy === 'score') {
          filtered.sort((a, b) => (b.rocket_score ?? 0) - (a.rocket_score ?? 0))
        } else if (sortBy === 'rr') {
          filtered.sort((a, b) => (b.risk_reward ?? 0) - (a.risk_reward ?? 0))
        } else if (sortBy === 'price') {
          filtered.sort((a, b) => (a.current_price ?? 0) - (b.current_price ?? 0))
        }

        return filtered.length > 0 ? (
        <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
          {/* Column headers */}
          <div className="grid px-4 py-2 border-b border-gray-800 text-[10px] text-gray-600 uppercase tracking-widest"
            style={{ gridTemplateColumns: '30px 80px 90px 1fr 80px 100px 120px 60px' }}>
            <span>Track</span><span>Symbol</span><span>Rocket Score</span><span>Pattern · GEX · Squeeze · Vol</span>
            <span className="text-right">Confluence</span><span className="text-right">Price / R:R</span><span className="text-right">Gamma</span><span />
          </div>
          <div className="divide-y divide-gray-800/40">
            {filtered.map((c: any) => {
              const col = SCORE_COLOR(c.rocket_score ?? 0)
              const isExp = expandedSym === c.symbol
              return (
                <div key={c.symbol}>
                  {/* Row */}
                  <div className="grid items-center gap-3 px-4 py-3 hover:bg-gray-800/30 transition-colors"
                    style={{ gridTemplateColumns: '30px 80px 90px 1fr 80px 100px 120px 60px' }}>

                    {/* Tracking checkbox */}
                    <div onClick={(e) => e.stopPropagation()} className="flex justify-center">
                      <input
                        type="checkbox"
                        checked={trackedSymbols.has(c.symbol)}
                        onChange={(e) => {
                          const newSet = new Set(trackedSymbols)
                          if (e.target.checked) newSet.add(c.symbol)
                          else newSet.delete(c.symbol)
                          setTrackedSymbols(newSet)
                        }}
                        className="w-4 h-4 cursor-pointer"
                      />
                    </div>

                    {/* Symbol */}
                    <span className="font-mono font-bold text-sm text-gray-100 cursor-pointer"
                      onClick={() => setExpandedSym(isExp ? null : c.symbol)}>
                      {c.symbol}
                    </span>

                    {/* Score bar */}
                    <div className="flex items-center gap-2">
                      <div className="w-10 h-1.5 rounded-full bg-gray-800 overflow-hidden">
                        <div className="h-full rounded-full" style={{ width: `${c.rocket_score ?? 0}%`, background: col }} />
                      </div>
                      <span className="text-xs font-mono font-bold" style={{ color: col }}>
                        {(c.rocket_score ?? 0).toFixed(0)}
                      </span>
                    </div>

                    {/* Sub-scores + pattern tag */}
                    <div className="flex items-center gap-3 flex-wrap">
                      {[
                        { label: 'PAT', val: c.component_scores?.pattern_quality, color: '#a78bfa' },
                        { label: 'GEX', val: c.component_scores?.gamma_score,     color: '#f59e0b' },
                        { label: 'SQZ', val: c.component_scores?.vol_squeeze,     color: '#60a5fa' },
                        { label: 'VOL', val: c.component_scores?.volume_accum,    color: '#34d399' },
                        { label: 'TRD', val: c.component_scores?.trend_align,     color: '#fb7185' },
                      ].map(({ label, val, color }) => val != null ? (
                        <div key={label} className="flex flex-col items-center gap-0.5">
                          <div className="w-8 h-1 rounded-full bg-gray-800 overflow-hidden">
                            <div className="h-full rounded-full" style={{ width: `${val}%`, background: color }} />
                          </div>
                          <span className="text-[9px] text-gray-600">{label}</span>
                        </div>
                      ) : null)}
                      {c.pattern_name && (
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-purple-900/30 text-purple-400 border border-purple-800/20">
                          {c.pattern_name}
                        </span>
                      )}
                      {c.squeeze_active && (
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-blue-900/30 text-blue-400 border border-blue-800/20">SQZ</span>
                      )}
                    </div>

                    {/* Confluence score */}
                    <div className="text-right">
                      <span className="text-xs font-mono font-bold text-emerald-400">
                        {c.confluence_score != null ? `${Math.round(c.confluence_score)}%` : '—'}
                      </span>
                    </div>

                    {/* Price + R:R */}
                    <div className="text-right">
                      <span className="text-xs font-mono text-gray-300">
                        {c.current_price > 0 ? `$${Number(c.current_price).toFixed(2)}` : '—'}
                      </span>
                      {c.risk_reward > 0 && (
                        <p className="text-[10px] text-emerald-600">{c.risk_reward.toFixed(1)}:1</p>
                      )}
                    </div>

                    {/* Gamma regime */}
                    <div className="text-right">
                      {c.gamma_regime && (
                        <span className={`text-[10px] px-2 py-0.5 rounded font-medium ${regimeStyle(c.gamma_regime)}`}>
                          {c.gamma_regime}
                        </span>
                      )}
                    </div>

                    {/* Expand + nav */}
                    <div className="flex items-center justify-end gap-2">
                      <span className="text-xs text-gray-700 hover:text-emerald-400"
                        onClick={e => { e.stopPropagation(); navigate(`/intel?symbol=${c.symbol}`) }}>→</span>
                      <span className="text-xs text-gray-700">{isExp ? '▲' : '▼'}</span>
                    </div>
                  </div>

                  {/* Expanded GEX detail */}
                  {isExp && (
                    <div className="px-4 pb-4 bg-gray-950/60 border-t border-gray-800/40">
                      {!gexData ? (
                        <div className="flex items-center gap-2 py-4 text-gray-600 text-xs">
                          <span className="w-3 h-3 border border-gray-600 border-t-transparent rounded-full animate-spin" />
                          Loading GEX…
                        </div>
                      ) : (
                        <GexDetail data={gexData} compact />
                      )}
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        </div>
      ) : (
        <div className="py-10 text-center text-gray-600 text-sm">No candidates match filters ({priceFilter === 'penny' ? 'Penny stocks' : 'All prices'}{filterTracked ? ' · Tracked only' : ''}).</div>
      )
      })()}

      {!isLoading && scanData?.length === 0 && (
        <div className="py-10 text-center text-gray-600 text-sm">No candidates surfaced — pipeline may need a fresh run or market conditions are low-volatility.</div>
      )}
    </div>
  )
}

// ── GEX / VEX / DEX detail component (shared by Rocket + Gamma tab) ───────────

function GexDetail({ data, compact = false }: { data: any; compact?: boolean }) {
  if (!data || data.error) return (
    <div className="py-6 text-center text-gray-600 text-sm">{data?.error ?? 'No data'}</div>
  )

  const spot = data.spot ?? 0
  const strikeRows: any[] = (data.gex_by_strike ?? [])
    .filter((r: any) => Math.abs(r.strike - spot) / spot < 0.20)  // ±20% of spot
    .sort((a: any, b: any) => a.strike - b.strike)

  const maxAbs = Math.max(...strikeRows.map((r: any) => Math.abs(r.net_gex)), 1)

  const REGIME_PILL: Record<string, string> = {
    EXPLOSIVE: 'bg-red-900/50 text-red-300 border-red-800/40',
    TRENDING:  'bg-amber-900/40 text-amber-300 border-amber-800/30',
    PINNED:    'bg-emerald-900/40 text-emerald-300 border-emerald-800/30',
    TRANSITIONING: 'bg-blue-900/40 text-blue-300 border-blue-800/30',
    HEDGING:   'bg-purple-900/40 text-purple-300 border-purple-800/30',
    NEGATIVE:  'bg-red-900/40 text-red-300 border-red-800/30',
    POSITIVE:  'bg-emerald-900/40 text-emerald-300 border-emerald-800/30',
    NET_SHORT_GAMMA: 'bg-red-900/40 text-red-300 border-red-800/30',
    NET_LONG_GAMMA:  'bg-emerald-900/40 text-emerald-300 border-emerald-800/30',
    DAMPENING: 'bg-blue-900/40 text-blue-300 border-blue-800/30',
    AMPLIFYING:'bg-red-900/40 text-red-300 border-red-800/30',
  }

  const pill = (v?: string) => v
    ? <span className={`text-[10px] px-2 py-0.5 rounded border font-medium ${REGIME_PILL[v] ?? 'bg-gray-800/40 text-gray-400 border-gray-700/30'}`}>{v}</span>
    : null

  const dealer = data.dealer_positioning ?? {}
  const vex    = data.vex ?? {}
  const sens   = data.gex_sensitivity ?? {}
  const pins   = data.gex_0dte_summary?.top_pins ?? []

  // Bar chart data
  const chartData = strikeRows.map((r: any) => ({
    strike: r.strike,
    call_gex: r.call_gex,
    put_gex: r.put_gex,
    net: r.net_gex,
  }))

  return (
    <div className={`space-y-4 ${compact ? 'pt-3' : ''}`}>
      {/* Top stats row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
        {[
          { label: 'Spot', val: `$${spot.toFixed(2)}`, color: 'text-white' },
          { label: 'Call Wall', val: data.call_wall ? `$${data.call_wall}` : '—', color: 'text-emerald-400' },
          { label: 'Put Wall',  val: data.put_wall  ? `$${data.put_wall}`  : '—', color: 'text-red-400' },
          { label: 'Zero Gamma',val: data.zero_gamma ? `$${data.zero_gamma.toFixed(0)}` : '—', color: 'text-amber-400' },
          { label: 'Max Pain',  val: data.max_pain  ? `$${data.max_pain}`  : '—', color: 'text-purple-400' },
          { label: 'GEX Total', val: `${(data.abs_total_gex_bn ?? 0).toFixed(1)}B`, color: 'text-gray-300' },
          { label: 'PC Ratio',  val: (data.pc_ratio ?? 0).toFixed(2), color: 'text-gray-300' },
          { label: 'GEX Score', val: `${data.gex_score ?? 0}`, color: 'text-blue-400' },
        ].map(({ label, val, color }) => (
          <div key={label} className="bg-gray-900 border border-gray-800/60 rounded-lg px-3 py-2">
            <p className="text-[10px] text-gray-600 uppercase tracking-wider">{label}</p>
            <p className={`text-sm font-mono font-bold mt-0.5 ${color}`}>{val}</p>
          </div>
        ))}
      </div>

      {/* Regime pills */}
      <div className="flex items-center gap-2 flex-wrap">
        <span className="text-[10px] text-gray-600 uppercase tracking-widest">Regimes:</span>
        {pill(data.options_regime)}
        {pill(data.gamma_regime)}
        {pill(dealer.dealer_regime)}
        {pill(vex.vex_regime)}
        {data.positioning_bias && (
          <span className={`text-[10px] px-2 py-0.5 rounded border font-medium ${
            data.positioning_bias === 'BULLISH' ? 'bg-emerald-900/30 text-emerald-400 border-emerald-800/30' : 'bg-red-900/30 text-red-400 border-red-800/30'
          }`}>{data.positioning_bias}</span>
        )}
        {dealer.squeeze_risk && (
          <span className={`text-[10px] px-2 py-0.5 rounded border font-medium ${
            dealer.squeeze_risk === 'HIGH' ? 'bg-red-900/40 text-red-300 border-red-800' : 'bg-gray-800/40 text-gray-500 border-gray-700/30'
          }`}>Squeeze: {dealer.squeeze_risk}</span>
        )}
      </div>

      {/* GEX Bar Chart */}
      {chartData.length > 0 && (
        <div className="bg-gray-900 border border-gray-800/60 rounded-xl p-3">
          <p className="text-[10px] text-gray-500 uppercase tracking-widest mb-2">GEX by Strike (±20%)</p>
          <ResponsiveContainer width="100%" height={compact ? 140 : 200}>
            <BarChart data={chartData} margin={{ top: 4, right: 4, bottom: 20, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
              <XAxis dataKey="strike" tick={{ fill: '#6b7280', fontSize: 9 }}
                tickFormatter={v => `$${v}`} interval="preserveStartEnd" />
              <YAxis tick={{ fill: '#6b7280', fontSize: 9 }}
                tickFormatter={v => {
                  const abs = Math.abs(v)
                  if (abs >= 1000) return `${(v/1000).toFixed(0)}kB`
                  if (abs >= 1) return `${v.toFixed(1)}B`
                  return `${(v*1000).toFixed(0)}M`
                }} width={42} />
              <Tooltip
                contentStyle={{ background: '#111827', border: '1px solid #374151', borderRadius: 8 }}
                labelStyle={{ color: '#9ca3af', fontSize: 11 }}
                itemStyle={{ fontSize: 11 }}
                formatter={(val: any) => [`${Number(val).toFixed(2)}B`]}
              />
              <ReferenceLine x={spot} stroke="#f59e0b" strokeDasharray="4 2" label={{ value: 'SPOT', fill: '#f59e0b', fontSize: 9 }} />
              {data.call_wall && <ReferenceLine x={data.call_wall} stroke="#22c55e" strokeDasharray="3 3" label={{ value: 'CW', fill: '#22c55e', fontSize: 9 }} />}
              {data.put_wall  && <ReferenceLine x={data.put_wall}  stroke="#ef4444" strokeDasharray="3 3" label={{ value: 'PW', fill: '#ef4444', fontSize: 9 }} />}
              {data.dex_zero_level && <ReferenceLine x={data.dex_zero_level} stroke="#a78bfa" strokeDasharray="2 4" label={{ value: 'DEX0', fill: '#a78bfa', fontSize: 9 }} />}
              <Bar dataKey="call_gex" name="Call GEX" stackId="a" fill="#22c55e" opacity={0.7} />
              <Bar dataKey="put_gex"  name="Put GEX"  stackId="a" fill="#ef4444" opacity={0.7} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* 3-column detail: Sensitivity | 0DTE Pins | Dealer Notes */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">

        {/* GEX Sensitivity */}
        {Object.keys(sens).length > 0 && (
          <div className="bg-gray-900 border border-gray-800/60 rounded-xl p-3">
            <p className="text-[10px] text-gray-500 uppercase tracking-widest mb-2">GEX Sensitivity</p>
            <div className="space-y-1.5">
              {[
                { key: 'dn5', label: '−5%', color: 'text-red-400' },
                { key: 'dn2', label: '−2%', color: 'text-red-300' },
                { key: 'flat', label: 'Flat', color: 'text-gray-300' },
                { key: 'up2', label: '+2%', color: 'text-emerald-300' },
                { key: 'up5', label: '+5%', color: 'text-emerald-400' },
              ].map(({ key, label, color }) => {
                const val = sens[key]
                if (val == null) return null
                const absMax = Math.max(...Object.values(sens as Record<string, number>).map(Math.abs), 1)
                const pct = Math.abs(val) / absMax * 100
                return (
                  <div key={key} className="flex items-center gap-2">
                    <span className={`text-[10px] w-8 font-mono ${color}`}>{label}</span>
                    <div className="flex-1 h-1.5 bg-gray-800 rounded-full overflow-hidden">
                      <div className="h-full rounded-full" style={{
                        width: `${pct}%`,
                        background: val < 0 ? '#ef4444' : '#22c55e'
                      }} />
                    </div>
                    <span className="text-[10px] font-mono text-gray-500 w-16 text-right">{val.toFixed(0)}B</span>
                  </div>
                )
              })}
            </div>
          </div>
        )}

        {/* 0DTE Pins */}
        {pins.length > 0 && (
          <div className="bg-gray-900 border border-gray-800/60 rounded-xl p-3">
            <p className="text-[10px] text-gray-500 uppercase tracking-widest mb-2">0DTE Pins</p>
            <div className="space-y-2">
              {pins.slice(0, 5).map((pin: any, i: number) => (
                <div key={i} className="flex items-center justify-between">
                  <span className={`text-xs font-mono font-bold ${i === 0 ? 'text-amber-400' : 'text-gray-400'}`}>
                    ${pin.strike}
                  </span>
                  <div className="flex items-center gap-2">
                    <span className="text-[10px] text-gray-600">{pin.dist_pct?.toFixed(2)}% away</span>
                    <div className="w-12 h-1 bg-gray-800 rounded-full overflow-hidden">
                      <div className="h-full rounded-full bg-amber-500" style={{
                        width: `${Math.min(pin.pin_score / (pins[0]?.pin_score ?? 1) * 100, 100)}%`
                      }} />
                    </div>
                  </div>
                </div>
              ))}
            </div>
            {data.gex_0dte_summary?.call_wall && (
              <div className="mt-2 pt-2 border-t border-gray-800/40 flex justify-between text-[10px] text-gray-600">
                <span>0DTE CW: <span className="text-emerald-500">${data.gex_0dte_summary.call_wall}</span></span>
                <span>0DTE PW: <span className="text-red-500">${data.gex_0dte_summary.put_wall}</span></span>
              </div>
            )}
          </div>
        )}

        {/* VEX + Dealer */}
        <div className="bg-gray-900 border border-gray-800/60 rounded-xl p-3 space-y-3">
          {/* VEX */}
          {vex.net_vex_total_mn != null && (
            <div>
              <p className="text-[10px] text-gray-500 uppercase tracking-widest mb-1.5">Vanna Exposure (VEX)</p>
              <div className="space-y-1">
                {[
                  { label: 'Net VEX', val: `${(vex.net_vex_total_mn ?? 0).toFixed(0)}M`, color: vex.net_vex_total_mn > 0 ? 'text-emerald-400' : 'text-red-400' },
                  { label: 'VEX Zero', val: vex.vex_zero_level ? `$${vex.vex_zero_level.toFixed(0)}` : '—', color: 'text-purple-400' },
                  { label: 'Charm 24h', val: vex.charm_decay_24h ? `${(vex.charm_decay_24h / 1000).toFixed(0)}K Δ` : '—', color: 'text-amber-400' },
                  { label: 'DEX Zero', val: data.dex_zero_level ? `$${data.dex_zero_level.toFixed(0)}` : '—', color: 'text-blue-400' },
                ].map(({ label, val, color }) => (
                  <div key={label} className="flex justify-between">
                    <span className="text-[10px] text-gray-600">{label}</span>
                    <span className={`text-[10px] font-mono font-bold ${color}`}>{val}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Dealer notes */}
          {dealer.notes?.length > 0 && (
            <div>
              <p className="text-[10px] text-gray-500 uppercase tracking-widest mb-1.5">Dealer Notes</p>
              <ul className="space-y-1">
                {dealer.notes.map((n: string, i: number) => (
                  <li key={i} className="text-[10px] text-gray-400 flex items-start gap-1.5">
                    <span className="text-amber-600 mt-0.5">▸</span>{n}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

// ── Gamma Explorer Tab ────────────────────────────────────────────────────────

function GammaExplorerPanel() {
  const [input, setInput] = useState('')
  const [symbol, setSymbol] = useState('')
  const inputRef = useRef<HTMLInputElement>(null)

  const { data, isLoading, error } = useQuery({
    queryKey: ['gex', 'detail', symbol],
    queryFn: () => symbol ? api.get(`/rocket/gex/${symbol}`, { params: { include_vex: true } }).then(r => r.data) : null,
    enabled: !!symbol,
    staleTime: 15 * 60_000,
  })

  const handleSearch = () => {
    const sym = input.trim().toUpperCase()
    if (sym) setSymbol(sym)
  }

  const QUICK = ['SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA', 'META', 'MSFT', 'AMZN']

  return (
    <div className="space-y-4">
      {/* Search bar */}
      <div className="flex items-center gap-2">
        <div className="flex-1 max-w-xs flex items-center gap-2 bg-gray-900 border border-gray-800 rounded-xl px-3 py-2">
          <span className="text-gray-600 text-sm">⌕</span>
          <input ref={inputRef} value={input}
            onChange={e => setInput(e.target.value.toUpperCase())}
            onKeyDown={e => e.key === 'Enter' && handleSearch()}
            placeholder="Symbol…"
            className="bg-transparent text-sm text-white placeholder-gray-700 outline-none w-full font-mono" />
        </div>
        <button onClick={handleSearch}
          className="px-4 py-2 rounded-xl bg-emerald-700 hover:bg-emerald-600 text-white text-sm font-medium transition-colors">
          Load GEX
        </button>
      </div>

      {/* Quick symbols */}
      <div className="flex items-center gap-1.5 flex-wrap">
        <span className="text-[10px] text-gray-700 uppercase tracking-widest">Quick:</span>
        {QUICK.map(s => (
          <button key={s} onClick={() => { setInput(s); setSymbol(s) }}
            className={`px-2.5 py-1 rounded-lg text-xs font-mono transition-colors ${
              symbol === s ? 'bg-emerald-800/50 text-emerald-300 border border-emerald-700/50' : 'bg-gray-900 border border-gray-800 text-gray-500 hover:text-gray-300 hover:border-gray-700'
            }`}>{s}</button>
        ))}
      </div>

      {/* Loading */}
      {isLoading && (
        <div className="flex items-center gap-3 py-12 justify-center">
          <div className="w-5 h-5 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin" />
          <span className="text-gray-400 text-sm">Fetching options chain for {symbol}…</span>
        </div>
      )}

      {/* Error */}
      {error && !isLoading && (
        <div className="py-8 text-center text-red-400 text-sm">Failed to load GEX for {symbol}</div>
      )}

      {/* Data */}
      {data && !isLoading && (
        <>
          <div className="flex items-center gap-2">
            <h3 className="text-lg font-black text-white">{data.symbol}</h3>
            <span className="text-sm text-gray-500 font-mono">${(data.spot ?? 0).toFixed(2)}</span>
          </div>
          <GexDetail data={data} />
        </>
      )}

      {/* Placeholder */}
      {!symbol && (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-12 text-center">
          <div className="text-3xl mb-3">⚡</div>
          <p className="text-gray-400 text-sm font-medium">GEX · VEX · DEX Explorer</p>
          <p className="text-gray-600 text-xs mt-1">Enter a symbol to view gamma exposure, dealer positioning, vanna/charm exposure, and 0DTE pins</p>
        </div>
      )}
    </div>
  )
}

// ── Page ─────────────────────────────────────────────────────────────────────

const TABS: { id: Tab; label: string; icon: string }[] = [
  { id: 'sniper-neg-gamma',    label: 'Gamma Sniper',           icon: '🎯' },
  { id: 'sniper-technical',    label: 'Technical Sniper',        icon: '📍' },
  { id: 'confluence',          label: 'All Confluence',          icon: '◉' },
]

// ── Sniper Panel ──────────────────────────────────────────────────────────────
interface SniperPanelProps {
  type: 'negative_gamma' | 'technical'
  data: any[]
}

function SniperPanel({ type, data }: SniperPanelProps) {
  const navigate = useNavigate()
  const title = type === 'negative_gamma' ? 'Negative Gamma Sniper Entries' : 'Technical Sniper Entries'
  const subtitle = type === 'negative_gamma'
    ? 'Stocks at absolute bottoms with options-confirmed gamma traps'
    : 'Stocks at absolute bottoms with RSI/MACD capitulation (includes no-options stocks like EONR)'

  if (!data || data.length === 0) {
    return (
      <div className="space-y-4">
        <div>
          <h2 className="text-xl font-bold">{title}</h2>
          <p className="text-sm text-gray-500">{subtitle}</p>
        </div>
        <div className="p-8 text-center text-gray-600">No {type === 'negative_gamma' ? 'negative gamma' : 'technical'} entry points found yet</div>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-xl font-bold">{title}</h2>
        <p className="text-sm text-gray-500">{subtitle}</p>
        <p className="text-sm text-emerald-400 mt-1">{data.length} candidates</p>
      </div>

      {/* Table */}
      <div className="grid gap-2 grid-cols-1" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(400px, 1fr))' }}>
        {data.map((entry: any) => (
          <div
            key={entry.symbol}
            onClick={() => navigate(`/chart/${entry.symbol}`)}
            className="p-3 bg-gray-900 border border-gray-800 rounded hover:border-emerald-700 cursor-pointer transition"
          >
            <div className="flex justify-between items-start mb-2">
              <div>
                <div className="font-bold text-lg">{entry.symbol}</div>
                <div className="text-sm text-gray-500">${entry.current_price.toFixed(2)}</div>
              </div>
              <div className="text-right">
                <div className="text-emerald-400 font-bold">{entry.entry_quality?.toFixed(0) || entry.readiness_score?.toFixed(0) || 'N/A'}</div>
                <div className="text-xs text-gray-500">Entry Quality</div>
              </div>
            </div>

            {/* Score breakdown */}
            <div className="grid grid-cols-2 gap-2 text-xs mb-2">
              {entry.support_score !== undefined && (
                <div className="flex justify-between">
                  <span className="text-gray-500">Support:</span>
                  <span className="text-gray-300">{entry.support_score.toFixed(0)}</span>
                </div>
              )}
              {entry.capitulation_score !== undefined && (
                <div className="flex justify-between">
                  <span className="text-gray-500">Capitulation:</span>
                  <span className="text-gray-300">{entry.capitulation_score.toFixed(0)}</span>
                </div>
              )}
              {entry.rsi !== undefined && (
                <div className="flex justify-between">
                  <span className="text-gray-500">RSI:</span>
                  <span className={entry.rsi < 25 ? 'text-red-400' : 'text-gray-300'}>{entry.rsi.toFixed(0)}</span>
                </div>
              )}
              {entry.distance_from_60_low_pct !== undefined && (
                <div className="flex justify-between">
                  <span className="text-gray-500">From Low:</span>
                  <span className="text-gray-300">{entry.distance_from_60_low_pct.toFixed(1)}%</span>
                </div>
              )}
            </div>

            {/* Notes */}
            <div className="text-xs text-gray-400 pt-2 border-t border-gray-800">
              {entry.notes?.map((note: string, i: number) => (
                <div key={i}>{note}</div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

export default function SignalsPage() {
  const [tab, setTab] = useState<Tab>('sniper-neg-gamma')

  // Sniper entry scanners - moved to main component for accessibility
  const { data: sniperData } = useQuery<{ negative_gamma: any[]; technical: any[] }>({
    queryKey: ['sniper', 'entry'],
    queryFn: async () => {
      const r = await api.get('/sniper-entry/scan', { params: { top_n: 500 } })
      const results = r.data?.results ?? { negative_gamma: [], technical: [] }
      return results
    },
    staleTime: 5 * 60_000,
  })

  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-2xl font-black text-white tracking-tight">SIGNALS</h2>
        <p className="text-xs text-gray-500 mt-0.5">Staged confluence · patterns · AI · probability — all in one place</p>
      </div>

      <div className="flex gap-1 bg-gray-900 border border-gray-800 rounded-xl p-1 w-fit flex-wrap">
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              tab === t.id ? 'bg-emerald-700 text-white' : 'text-gray-500 hover:text-gray-300 hover:bg-gray-800/50'
            }`}>
            <span>{t.icon}</span>
            <span>{t.label}</span>
          </button>
        ))}
      </div>

      <div>
        {tab === 'sniper-neg-gamma'    && <SniperPanel type="negative_gamma" data={sniperData?.negative_gamma ?? []} />}
        {tab === 'sniper-technical'    && <SniperPanel type="technical" data={sniperData?.technical ?? []} />}
        {tab === 'confluence'          && <ConfluenceFunnel />}
      </div>
    </div>
  )
}
