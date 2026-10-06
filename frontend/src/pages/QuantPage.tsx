/**
 * frontend/src/pages/QuantPage.tsx
 * ────────────────────────────────────────────────────────────────────────────
 * Quant Scanner — unified UI for the full scan → score → allocate pipeline.
 *
 * Tabs:
 *   Scanner  — Run full pipeline, view scored candidates + trade allocations
 *   Score    — Score a single symbol with explanation
 *   GEX      — Dealer positioning: GEX / DEX / VEX bar charts
 *   Tracker  — Trade journal performance stats
 */

import { useState, useCallback, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import axios from 'axios'
import { ConfluenceFunnel } from './SignalsPage'
import PatternScanPage from './PatternScanPage'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ReferenceLine, Cell, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
} from 'recharts'
import { createChart, ColorType, LineStyle } from 'lightweight-charts'

const API = '/api/quant'

// ── Types ─────────────────────────────────────────────────────────────────────

interface Candidate {
  symbol:         string
  score:          number
  signal_type:    string
  confidence:     string
  price:          number
  change_percent: number
  volume_ratio:   number
}

interface Allocation {
  symbol:       string
  signal_type:  string
  confidence:   string
  score:        number
  entry_price:  number
  shares:       number
  stop_price:   number
  target_price: number
  dollar_risk:  number
  position_value: number
  risk_reward:  number
}

interface Explanation {
  symbol:            string
  summary:           string
  detailed_reasoning?: string
  trade_idea?:       string
  key_levels?:       { entry: number; stop: number; target: number }
  warnings?:         string[]
}

interface ScanResult {
  cached:       boolean
  timestamp:    number
  candidates:   Candidate[]
  allocations:  Allocation[]
  explanations: Explanation[]
  summary:      Record<string, number>
}

interface ScoreResult {
  symbol:      string
  quote:       Record<string, number>
  signal:      { symbol: string; score: number; signal_type: string; confidence: string; factors: Record<string, number>; gex_context?: Record<string, unknown>; explanation: string }
  explanation: Explanation
}

interface GexResult {
  symbol:      string
  spot_price:  number
  gex:         Record<string, unknown>
  dex:         Record<string, unknown>
  gamma_flip:  number | null
  gex_regime:  string | null
  dealer_bias: string | null
  positioning: Record<string, unknown>
}

interface TrackerStats {
  overall:       Record<string, number | number[]>
  by_regime:     Record<string, Record<string, number>>
  by_confidence: Record<string, Record<string, number>>
  by_symbol:     Record<string, Record<string, number>>
  open_trades:   unknown[]
}

// ── Helper components ─────────────────────────────────────────────────────────

function Badge({ val, label }: { val: string; label?: string }) {
  const color =
    val === 'high'   || val === 'long'     || val === 'won'   ? 'text-emerald-400 bg-emerald-950 border-emerald-800' :
    val === 'medium' || val === 'NEUTRAL'                     ? 'text-amber-400 bg-amber-950 border-amber-800' :
    val === 'low'    || val === 'short'    || val === 'lost'  ? 'text-red-400 bg-red-950 border-red-800' :
    val === 'open'   || val === 'scratch'                     ? 'text-blue-400 bg-blue-950 border-blue-800' :
    'text-gray-400 bg-gray-800 border-gray-700'

  return (
    <span className={`inline-flex items-center px-1.5 py-0.5 rounded text-[9px] font-bold uppercase border ${color}`}>
      {label ?? val}
    </span>
  )
}

function ScoreBar({ score }: { score: number }) {
  const pct = Math.round(score * 100)
  const color = pct >= 72 ? '#22c55e' : pct >= 52 ? '#eab308' : '#ef4444'
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 bg-gray-800 rounded-full overflow-hidden">
        <div className="h-full rounded-full transition-all" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
      <span className="text-[10px] font-mono font-bold" style={{ color }}>{pct}</span>
    </div>
  )
}

function Stat({ label, value, sub }: { label: string; value: string | number; sub?: string }) {
  return (
    <div className="bg-gray-800/60 rounded-lg p-3 border border-gray-700/50">
      <div className="text-[9px] text-gray-500 uppercase tracking-wide mb-1">{label}</div>
      <div className="text-sm font-bold text-gray-100">{value}</div>
      {sub && <div className="text-[9px] text-gray-500 mt-0.5">{sub}</div>}
    </div>
  )
}

function LoadSpinner() {
  return (
    <div className="flex items-center justify-center py-12">
      <div className="w-5 h-5 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin" />
      <span className="ml-3 text-xs text-gray-500">Loading…</span>
    </div>
  )
}

function ErrorBox({ msg }: { msg: string }) {
  return (
    <div className="bg-red-950/40 border border-red-800/50 rounded-lg p-3 text-xs text-red-400">{msg}</div>
  )
}

// ── Thin refresh bar — 2px gradient strip shown during background refetch ────────
const _POUNCE_LS_KEY = 'roboalgo:pounce-v1'

function RefreshBar({ active }: { active: boolean }) {
  const [width,  setWidth]  = useState(0)
  const [fading, setFading] = useState(false)

  useEffect(() => {
    if (active) {
      setFading(false); setWidth(0)
      const t = setTimeout(() => setWidth(80), 30)
      return () => clearTimeout(t)
    } else {
      setWidth(100)
      const t = setTimeout(() => setFading(true), 300)
      return () => clearTimeout(t)
    }
  }, [active])

  if (!active && fading) return null
  return (
    <div className="h-[2px] w-full rounded-full overflow-hidden bg-transparent"
         style={{ opacity: fading ? 0 : 1, transition: 'opacity 0.4s ease' }}>
      <div className="h-full rounded-full bg-gradient-to-r from-violet-500 via-blue-400 to-emerald-400"
           style={{ width: `${width}%`, transition: active ? 'width 6s ease-out' : 'width 0.25s ease-in' }} />
    </div>
  )
}

// ── Scan progress bar — replaces the generic spinner during long scans ─────────
const SCAN_STAGES = [
  { at:  0, label: 'Initialising scanner…' },
  { at: 12, label: 'Fetching universe prices…' },
  { at: 28, label: 'Applying momentum filters…' },
  { at: 45, label: 'Running chart pattern detection…' },
  { at: 62, label: 'Checking options flow & GEX…' },
  { at: 78, label: 'Ranking by confluence score…' },
  { at: 88, label: 'Finalising signals…' },
]

function ScanProgressBar({ loading }: { loading: boolean }) {
  const [pct, setPct] = useState(0)
  const startRef = useRef<number>(0)

  useEffect(() => {
    if (!loading) { setPct(0); return }
    startRef.current = Date.now()
    const DURATION = 38_000  // ~38 s typical full scan
    const id = setInterval(() => {
      const t = (Date.now() - startRef.current) / DURATION
      // Ease-out: quick start, decelerates toward 90% ceiling so it never "finishes" early
      const next = Math.min(90, 90 * (1 - Math.pow(Math.max(0, 1 - t), 2.4)))
      setPct(next)
    }, 120)
    return () => clearInterval(id)
  }, [loading])

  const stage = [...SCAN_STAGES].reverse().find(s => pct >= s.at)?.label ?? ''

  if (!loading && pct === 0) return null

  return (
    <div className="py-8 px-2 space-y-3">
      {/* Stage label + percentage */}
      <div className="flex items-center justify-between">
        <span className="text-[11px] text-gray-400 font-medium">{stage}</span>
        <span className="text-[10px] text-gray-600 font-mono tabular-nums">{Math.round(pct)}%</span>
      </div>

      {/* Progress track */}
      <div className="relative h-1.5 bg-gray-800 rounded-full overflow-hidden">
        <div
          className="absolute inset-y-0 left-0 rounded-full transition-all duration-150"
          style={{
            width: `${pct}%`,
            background: 'linear-gradient(90deg, #7c3aed 0%, #059669 100%)',
            boxShadow: '0 0 8px rgba(124,58,237,0.5)',
          }}
        />
      </div>

      {/* Stage milestone dots */}
      <div className="flex items-center justify-between px-0.5">
        {SCAN_STAGES.map(s => (
          <div
            key={s.at}
            className="w-1.5 h-1.5 rounded-full transition-colors duration-300"
            style={{ backgroundColor: pct >= s.at ? '#7c3aed' : '#1f2937' }}
          />
        ))}
      </div>
    </div>
  )
}

// ── Tab: Scanner ──────────────────────────────────────────────────────────────

function ScannerTab() {
  const [symbols,     setSymbols]     = useState('AAPL,NVDA,TSLA,SPY,QQQ,AMD,META,MSFT,AMZN,GOOGL')
  const [accountSize, setAccountSize] = useState('10000')
  const [minScore,    setMinScore]    = useState('0.40')
  const [maxTrades,   setMaxTrades]   = useState('3')
  const [mode,        setMode]        = useState<'beginner' | 'advanced'>('beginner')
  const [loading,     setLoading]     = useState(false)
  const [error,       setError]       = useState('')
  const [result,      setResult]      = useState<ScanResult | null>(null)

  const runScan = useCallback(async () => {
    setLoading(true); setError('')
    try {
      const { data } = await axios.get<ScanResult>(`${API}/scan`, {
        params: {
          symbols:      symbols,
          account_size: Number(accountSize),
          min_score:    Number(minScore),
          max_trades:   Number(maxTrades),
          mode,
          force:        true,
        },
      })
      setResult(data)
    } catch (e: unknown) {
      const msg = (e as { response?: { data?: { detail?: string } }; message?: string })
      setError(msg?.response?.data?.detail ?? msg?.message ?? 'Scan failed')
    } finally {
      setLoading(false)
    }
  }, [symbols, accountSize, minScore, maxTrades, mode])

  return (
    <div className="space-y-4">
      {/* Config bar */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 space-y-3">
        <div className="text-[10px] text-gray-500 font-bold uppercase tracking-widest">Pipeline Parameters</div>
        <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
          <div>
            <label className="text-[9px] text-gray-600 block mb-1">Account Size ($)</label>
            <input
              value={accountSize}
              onChange={e => setAccountSize(e.target.value)}
              className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-xs font-mono text-gray-200 focus:outline-none focus:border-emerald-600"
            />
          </div>
          <div>
            <label className="text-[9px] text-gray-600 block mb-1">Min Score</label>
            <input
              value={minScore}
              onChange={e => setMinScore(e.target.value)}
              className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-xs font-mono text-gray-200 focus:outline-none focus:border-emerald-600"
            />
          </div>
          <div>
            <label className="text-[9px] text-gray-600 block mb-1">Max Trades</label>
            <input
              value={maxTrades}
              onChange={e => setMaxTrades(e.target.value)}
              className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-xs font-mono text-gray-200 focus:outline-none focus:border-emerald-600"
            />
          </div>
          <div>
            <label className="text-[9px] text-gray-600 block mb-1">Mode</label>
            <select
              value={mode}
              onChange={e => setMode(e.target.value as 'beginner' | 'advanced')}
              className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-xs text-gray-200 focus:outline-none focus:border-emerald-600"
            >
              <option value="beginner">Beginner</option>
              <option value="advanced">Advanced</option>
            </select>
          </div>
        </div>
        <div>
          <label className="text-[9px] text-gray-600 block mb-1">Symbols (comma-separated)</label>
          <input
            value={symbols}
            onChange={e => setSymbols(e.target.value.toUpperCase())}
            className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-xs font-mono text-gray-200 focus:outline-none focus:border-emerald-600"
          />
        </div>
        <button
          onClick={runScan}
          disabled={loading}
          className="bg-emerald-700 hover:bg-emerald-600 disabled:opacity-50 text-white text-xs font-bold px-4 py-2 rounded-lg transition-colors"
        >
          {loading ? 'Scanning…' : '▶ Run Full Pipeline'}
        </button>
      </div>

      {loading && <LoadSpinner />}
      {error   && <ErrorBox msg={error} />}

      {result && (
        <div className="space-y-4">
          {/* Meta */}
          <div className="flex items-center gap-3 text-[10px] text-gray-500">
            <span className="text-emerald-400 font-bold">{result.candidates.length} candidates</span>
            <span>·</span>
            <span className="text-violet-400 font-bold">{result.allocations.length} allocations</span>
            {result.cached && <span className="text-amber-400">⚡ cached</span>}
            <span className="ml-auto">{new Date(result.timestamp * 1000).toLocaleTimeString()}</span>
          </div>

          {/* Portfolio summary */}
          {Object.keys(result.summary).length > 0 && (
            <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
              {['total_allocated', 'total_risk', 'positions', 'avg_score'].map(k => {
                const v = result.summary[k]
                if (v == null) return null
                const fmt = k === 'total_allocated' || k === 'total_risk'
                  ? `$${Number(v).toLocaleString()}`
                  : k === 'avg_score' ? `${(Number(v) * 100).toFixed(0)}%` : String(v)
                return (
                  <Stat
                    key={k}
                    label={k.replace(/_/g, ' ')}
                    value={fmt}
                  />
                )
              })}
            </div>
          )}

          {/* Allocations — primary */}
          {result.allocations.length > 0 && (
            <div className="space-y-2">
              <div className="text-[10px] text-gray-500 font-bold uppercase tracking-widest">Trade Allocations</div>
              <div className="space-y-2">
                {result.allocations.map((a, i) => {
                  const exp = result.explanations[i]
                  return (
                    <div key={a.symbol} className="bg-gray-900 border border-gray-800 rounded-xl p-4">
                      <div className="flex items-start justify-between gap-4 mb-3">
                        <div className="flex items-center gap-3">
                          <span className="text-base font-black text-white">{a.symbol}</span>
                          <Badge val={a.signal_type} />
                          <Badge val={a.confidence} />
                        </div>
                        <div className="text-right">
                          <div className="text-xs font-bold text-gray-200">${a.entry_price != null ? a.entry_price.toFixed(2) : '—'}</div>
                          <div className="text-[9px] text-gray-500">{a.shares ?? '—'} shares</div>
                        </div>
                      </div>

                      <ScoreBar score={a.score} />

                      <div className="grid grid-cols-3 gap-2 mt-3 text-center">
                        <div className="bg-gray-800/50 rounded p-2">
                          <div className="text-[9px] text-gray-600">Stop</div>
                          <div className="text-xs font-bold text-red-400">${a.stop_price != null ? a.stop_price.toFixed(2) : '—'}</div>
                        </div>
                        <div className="bg-gray-800/50 rounded p-2">
                          <div className="text-[9px] text-gray-600">Risk</div>
                          <div className="text-xs font-bold text-amber-400">${a.dollar_risk != null ? a.dollar_risk.toFixed(0) : '—'}</div>
                        </div>
                        <div className="bg-gray-800/50 rounded p-2">
                          <div className="text-[9px] text-gray-600">Target</div>
                          <div className="text-xs font-bold text-emerald-400">${a.target_price != null ? a.target_price.toFixed(2) : '—'}</div>
                        </div>
                      </div>

                      {exp && (
                        <div className="mt-3 space-y-2">
                          <p className="text-[11px] text-gray-300">{exp.summary}</p>
                          {exp.trade_idea && (
                            <p className="text-[10px] text-violet-300 italic">{exp.trade_idea}</p>
                          )}
                          {exp.warnings && exp.warnings.length > 0 && (
                            <div className="flex flex-wrap gap-1 mt-1">
                              {exp.warnings.map((w, wi) => (
                                <span key={wi} className="text-[9px] text-amber-400 bg-amber-950/40 border border-amber-800/40 rounded px-1.5 py-0.5">⚠ {w}</span>
                              ))}
                            </div>
                          )}
                        </div>
                      )}

                      <div className="mt-2 text-[9px] text-gray-600">
                        Position ${a.position_value.toFixed(0)} · RR {a.risk_reward.toFixed(1)}:1
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          )}

          {/* All candidates — compact */}
          {result.candidates.length > 0 && (
            <div className="space-y-2">
              <div className="text-[10px] text-gray-500 font-bold uppercase tracking-widest">All Scored Candidates</div>
              <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-gray-800 text-[9px] text-gray-600 uppercase">
                      <th className="px-3 py-2 text-left">Symbol</th>
                      <th className="px-3 py-2 text-left">Score</th>
                      <th className="px-3 py-2 text-left">Signal</th>
                      <th className="px-3 py-2 text-right">Price</th>
                      <th className="px-3 py-2 text-right">Chg%</th>
                      <th className="px-3 py-2 text-right">Vol×</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.candidates.map(c => (
                      <tr key={c.symbol} className="border-b border-gray-800/50 hover:bg-gray-800/30 transition-colors">
                        <td className="px-3 py-2 font-bold text-white">{c.symbol}</td>
                        <td className="px-3 py-2 w-28">
                          <ScoreBar score={c.score} />
                        </td>
                        <td className="px-3 py-2">
                          <div className="flex items-center gap-1">
                            <Badge val={c.signal_type} />
                            <Badge val={c.confidence} />
                          </div>
                        </td>
                        <td className="px-3 py-2 text-right font-mono text-gray-200">${c.price.toFixed(2)}</td>
                        <td className={`px-3 py-2 text-right font-mono font-bold ${c.change_percent >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                          {c.change_percent >= 0 ? '+' : ''}{c.change_percent.toFixed(2)}%
                        </td>
                        <td className="px-3 py-2 text-right font-mono text-gray-400">{c.volume_ratio.toFixed(1)}×</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// ── Tab: Score ────────────────────────────────────────────────────────────────

const FACTOR_LABELS: Record<string, string> = {
  cycle_alignment:   'Cycle Alignment',
  gex_signal:        'GEX Signal',
  momentum_strength: 'Momentum',
  volatility_state:  'Volatility',
  structure_quality: 'Structure',
  catalyst_score:    'Catalyst',
}

function ScoreTab() {
  const [sym,     setSym]     = useState('')
  const [mode,    setMode]    = useState<'beginner' | 'advanced'>('beginner')
  const [loading, setLoading] = useState(false)
  const [error,   setError]   = useState('')
  const [result,  setResult]  = useState<ScoreResult | null>(null)

  const runScore = useCallback(async () => {
    const s = sym.trim().toUpperCase()
    if (!s) return
    setLoading(true); setError('')
    try {
      const { data } = await axios.get<ScoreResult>(`${API}/score/${s}`, { params: { mode } })
      setResult(data)
    } catch (e: unknown) {
      const msg = (e as { response?: { data?: { detail?: string } }; message?: string })
      setError(msg?.response?.data?.detail ?? msg?.message ?? 'Score failed')
    } finally {
      setLoading(false)
    }
  }, [sym, mode])

  const factors = result?.signal?.factors
  const radarData = factors
    ? Object.entries(factors).map(([k, v]) => ({
        factor: FACTOR_LABELS[k] ?? k,
        value:  Math.round(Number(v) * 100),
        fullMark: 100,
      }))
    : []

  return (
    <div className="space-y-4">
      {/* Search bar */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 flex gap-3 items-end">
        <div className="flex-1">
          <label className="text-[9px] text-gray-600 block mb-1">Symbol</label>
          <input
            value={sym}
            onChange={e => setSym(e.target.value.toUpperCase())}
            onKeyDown={e => e.key === 'Enter' && runScore()}
            placeholder="AAPL"
            className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-xs font-mono text-gray-200 focus:outline-none focus:border-emerald-600"
          />
        </div>
        <div>
          <label className="text-[9px] text-gray-600 block mb-1">Mode</label>
          <select
            value={mode}
            onChange={e => setMode(e.target.value as 'beginner' | 'advanced')}
            className="bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-xs text-gray-200 focus:outline-none focus:border-emerald-600"
          >
            <option value="beginner">Beginner</option>
            <option value="advanced">Advanced</option>
          </select>
        </div>
        <button
          onClick={runScore}
          disabled={loading || !sym.trim()}
          className="bg-emerald-700 hover:bg-emerald-600 disabled:opacity-50 text-white text-xs font-bold px-4 py-2 rounded-lg transition-colors"
        >
          {loading ? '…' : 'Score'}
        </button>
      </div>

      {loading && <LoadSpinner />}
      {error   && <ErrorBox msg={error} />}

      {result && (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          {/* Left: score summary + factors */}
          <div className="space-y-4">
            {/* Header */}
            <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
              <div className="flex items-center justify-between mb-3">
                <span className="text-lg font-black text-white">{result.symbol}</span>
                <div className="flex items-center gap-2">
                  <Badge val={result.signal.signal_type} />
                  <Badge val={result.signal.confidence} />
                </div>
              </div>
              <div className="mb-3">
                <ScoreBar score={result.signal.score} />
                <div className="text-[9px] text-gray-500 mt-1">Composite score</div>
              </div>
              {result.quote && (
                <div className="grid grid-cols-3 gap-2 text-center">
                  {(['price', 'change_percent', 'volume'] as const).map(k => {
                    const v = result.quote[k]
                    if (v == null) return null
                    const fmt = k === 'price' ? `$${Number(v).toFixed(2)}`
                      : k === 'change_percent' ? `${Number(v) >= 0 ? '+' : ''}${Number(v).toFixed(2)}%`
                      : Number(v).toLocaleString()
                    const color = k === 'change_percent'
                      ? Number(v) >= 0 ? 'text-emerald-400' : 'text-red-400'
                      : 'text-gray-200'
                    return (
                      <div key={k} className="bg-gray-800/50 rounded p-2">
                        <div className="text-[9px] text-gray-600">{k.replace(/_/g, ' ')}</div>
                        <div className={`text-xs font-bold font-mono ${color}`}>{fmt}</div>
                      </div>
                    )
                  })}
                </div>
              )}
            </div>

            {/* Factor bars */}
            {factors && (
              <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
                <div className="text-[10px] text-gray-500 font-bold uppercase tracking-widest mb-3">Factor Breakdown</div>
                <div className="space-y-2">
                  {Object.entries(factors).map(([k, v]) => (
                    <div key={k}>
                      <div className="flex justify-between text-[9px] text-gray-500 mb-0.5">
                        <span>{FACTOR_LABELS[k] ?? k}</span>
                        <span className="font-mono">{(Number(v) * 100).toFixed(0)}</span>
                      </div>
                      <div className="h-1 bg-gray-800 rounded-full overflow-hidden">
                        <div
                          className="h-full rounded-full"
                          style={{
                            width: `${Math.min(Number(v) * 100, 100)}%`,
                            backgroundColor: Number(v) >= 0.7 ? '#22c55e' : Number(v) >= 0.5 ? '#eab308' : '#ef4444',
                          }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Right: radar + explanation */}
          <div className="space-y-4">
            {radarData.length > 0 && (
              <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
                <div className="text-[10px] text-gray-500 font-bold uppercase tracking-widest mb-2">Signal Radar</div>
                <ResponsiveContainer width="100%" height={220}>
                  <RadarChart data={radarData}>
                    <PolarGrid stroke="#374151" />
                    <PolarAngleAxis dataKey="factor" tick={{ fontSize: 9, fill: '#9ca3af' }} />
                    <PolarRadiusAxis domain={[0, 100]} tick={false} axisLine={false} />
                    <Radar dataKey="value" stroke="#22c55e" fill="#22c55e" fillOpacity={0.2} />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
            )}

            {/* Explanation */}
            {result.explanation && (
              <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 space-y-3">
                <div className="text-[10px] text-gray-500 font-bold uppercase tracking-widest">Explanation</div>
                <p className="text-xs text-gray-300 leading-relaxed">{result.explanation.summary}</p>
                {result.explanation.detailed_reasoning && (
                  <p className="text-[11px] text-gray-400 leading-relaxed">{result.explanation.detailed_reasoning}</p>
                )}
                {result.explanation.trade_idea && (
                  <p className="text-[11px] text-violet-300 italic">{result.explanation.trade_idea}</p>
                )}
                {result.explanation.key_levels && (
                  <div className="grid grid-cols-3 gap-2 text-center">
                    <div className="bg-gray-800/50 rounded p-2">
                      <div className="text-[9px] text-gray-600">Entry</div>
                      <div className="text-xs font-bold text-gray-200">${result.explanation.key_levels.entry.toFixed(2)}</div>
                    </div>
                    <div className="bg-gray-800/50 rounded p-2">
                      <div className="text-[9px] text-gray-600">Stop</div>
                      <div className="text-xs font-bold text-red-400">${result.explanation.key_levels.stop.toFixed(2)}</div>
                    </div>
                    <div className="bg-gray-800/50 rounded p-2">
                      <div className="text-[9px] text-gray-600">Target</div>
                      <div className="text-xs font-bold text-emerald-400">${result.explanation.key_levels.target.toFixed(2)}</div>
                    </div>
                  </div>
                )}
                {result.explanation.warnings && result.explanation.warnings.length > 0 && (
                  <div className="space-y-1">
                    {result.explanation.warnings.map((w, i) => (
                      <div key={i} className="text-[10px] text-amber-400 bg-amber-950/30 border border-amber-800/30 rounded px-2 py-1">⚠ {w}</div>
                    ))}
                  </div>
                )}

                {/* GEX context */}
                {result.signal.gex_context && Object.keys(result.signal.gex_context).length > 0 && (
                  <div className="border-t border-gray-800 pt-3">
                    <div className="text-[9px] text-gray-600 uppercase tracking-wide mb-2">GEX Context</div>
                    <div className="grid grid-cols-2 gap-2 text-[10px]">
                      {Object.entries(result.signal.gex_context).map(([k, v]) => (
                        <div key={k} className="flex justify-between">
                          <span className="text-gray-600">{k.replace(/_/g, ' ')}</span>
                          <span className="text-gray-300 font-mono">{String(v)}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

// ── Tab: GEX ──────────────────────────────────────────────────────────────────

function GexTab() {
  const [sym,        setSym]        = useState('')
  const [nearExpiry, setNearExpiry] = useState(false)
  const [loading,    setLoading]    = useState(false)
  const [error,      setError]      = useState('')
  const [result,     setResult]     = useState<GexResult | null>(null)

  const runGex = useCallback(async () => {
    const s = sym.trim().toUpperCase()
    if (!s) return
    setLoading(true); setError('')
    try {
      const { data } = await axios.get<GexResult>(`${API}/gex/${s}`, { params: { near_expiry: nearExpiry } })
      setResult(data)
    } catch (e: unknown) {
      const msg = (e as { response?: { data?: { detail?: string } }; message?: string })
      setError(msg?.response?.data?.detail ?? msg?.message ?? 'GEX fetch failed')
    } finally {
      setLoading(false)
    }
  }, [sym, nearExpiry])

  // Build strike-level bar data from gex/dex strike maps
  const gex = result?.gex as Record<string, unknown> | undefined
  const dex = result?.dex as Record<string, unknown> | undefined

  // gex_by_strike: { [strike]: number }
  const strikeGexData: { strike: number; gex: number; color: string }[] = []
  const rawGexByStrike = gex?.gex_by_strike as Record<string, number> | undefined
  if (rawGexByStrike) {
    const entries = Object.entries(rawGexByStrike)
      .map(([k, v]) => ({ strike: Number(k), gex: Number(v) }))
      .sort((a, b) => a.strike - b.strike)

    const spot = result?.spot_price ?? 0
    const nearby = entries.filter(e => Math.abs(e.strike - spot) / spot < 0.15)
    nearby.forEach(e => strikeGexData.push({ ...e, color: e.gex >= 0 ? '#22c55e' : '#ef4444' }))
  }

  // dex_by_strike for DEX chart
  const strikeDexData: { strike: number; dex: number; color: string }[] = []
  const rawDexByStrike = dex?.dex_by_strike as Record<string, number> | undefined
  if (rawDexByStrike) {
    const entries = Object.entries(rawDexByStrike)
      .map(([k, v]) => ({ strike: Number(k), dex: Number(v) }))
      .sort((a, b) => a.strike - b.strike)
    const spot = result?.spot_price ?? 0
    const nearby = entries.filter(e => Math.abs(e.strike - spot) / spot < 0.15)
    nearby.forEach(e => strikeDexData.push({ ...e, color: e.dex >= 0 ? '#3b82f6' : '#f97316' }))
  }

  const regimeColor = (regime: string | null) => {
    if (!regime) return '#9ca3af'
    if (regime.includes('POSITIVE')) return '#22c55e'
    if (regime.includes('NEGATIVE')) return '#ef4444'
    if (regime.includes('EXPLOSIVE')) return '#f97316'
    if (regime.includes('PINNED'))    return '#3b82f6'
    return '#9ca3af'
  }

  return (
    <div className="space-y-4">
      {/* Search */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 flex gap-3 items-end">
        <div className="flex-1">
          <label className="text-[9px] text-gray-600 block mb-1">Symbol</label>
          <input
            value={sym}
            onChange={e => setSym(e.target.value.toUpperCase())}
            onKeyDown={e => e.key === 'Enter' && runGex()}
            placeholder="SPY"
            className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-xs font-mono text-gray-200 focus:outline-none focus:border-emerald-600"
          />
        </div>
        <label className="flex items-center gap-1.5 text-[10px] text-gray-400 cursor-pointer mb-2">
          <input
            type="checkbox"
            checked={nearExpiry}
            onChange={e => setNearExpiry(e.target.checked)}
            className="rounded"
          />
          Near expiry only
        </label>
        <button
          onClick={runGex}
          disabled={loading || !sym.trim()}
          className="bg-violet-700 hover:bg-violet-600 disabled:opacity-50 text-white text-xs font-bold px-4 py-2 rounded-lg transition-colors"
        >
          {loading ? '…' : 'Load GEX'}
        </button>
      </div>

      {loading && <LoadSpinner />}
      {error   && <ErrorBox msg={error} />}

      {result && (
        <div className="space-y-4">
          {/* Summary row */}
          <div className="flex flex-wrap items-center gap-3">
            <span className="text-lg font-black text-white">{result.symbol}</span>
            <span className="text-sm font-mono text-gray-300">${result.spot_price?.toFixed(2)}</span>
            {result.gex_regime && (
              <span
                className="text-[10px] font-bold px-2 py-1 rounded-full border"
                style={{ color: regimeColor(result.gex_regime), borderColor: regimeColor(result.gex_regime) + '40', backgroundColor: regimeColor(result.gex_regime) + '15' }}
              >
                {result.gex_regime}
              </span>
            )}
            {result.dealer_bias && (
              <span className="text-[10px] text-gray-400">Dealer: <strong className="text-gray-200">{result.dealer_bias}</strong></span>
            )}
            {result.gamma_flip != null && (
              <span className="text-[10px] text-gray-400">Gamma Flip: <strong className="text-amber-400">${result.gamma_flip.toFixed(2)}</strong></span>
            )}
          </div>

          {/* Metric cards */}
          <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
            {gex && (
              <>
                {['total_gex', 'call_gex', 'put_gex', 'zero_gamma'].map(k => {
                  const v = (gex as Record<string, unknown>)[k]
                  if (v == null) return null
                  const fmt = typeof v === 'number'
                    ? Math.abs(Number(v)) > 1e6 ? `${(Number(v) / 1e6).toFixed(1)}M` : Number(v).toFixed(2)
                    : String(v)
                  return <Stat key={k} label={k.replace(/_/g, ' ')} value={fmt} />
                })}
              </>
            )}
          </div>

          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            {/* GEX by strike */}
            {strikeGexData.length > 0 && (
              <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
                <div className="text-[10px] text-gray-500 font-bold uppercase tracking-widest mb-3">
                  GEX by Strike <span className="text-gray-600 normal-case font-normal">(±15% from spot)</span>
                </div>
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={strikeGexData} margin={{ top: 4, right: 4, bottom: 4, left: 4 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                    <XAxis dataKey="strike" tick={{ fontSize: 8, fill: '#6b7280' }} />
                    <YAxis tick={{ fontSize: 8, fill: '#6b7280' }} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#111827', border: '1px solid #374151', borderRadius: 6, fontSize: 10 }}
                      formatter={(v: number) => [v.toFixed(2), 'GEX']}
                    />
                    <ReferenceLine y={0} stroke="#374151" />
                    {result.gamma_flip != null && (
                      <ReferenceLine x={result.gamma_flip} stroke="#f59e0b" strokeDasharray="4 2" label={{ value: 'γ flip', fontSize: 8, fill: '#f59e0b' }} />
                    )}
                    <Bar dataKey="gex" radius={[2, 2, 0, 0]}>
                      {strikeGexData.map((entry, i) => (
                        <Cell key={i} fill={entry.color} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}

            {/* DEX by strike */}
            {strikeDexData.length > 0 && (
              <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
                <div className="text-[10px] text-gray-500 font-bold uppercase tracking-widest mb-3">
                  DEX by Strike <span className="text-gray-600 normal-case font-normal">(dealer delta exposure)</span>
                </div>
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={strikeDexData} margin={{ top: 4, right: 4, bottom: 4, left: 4 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                    <XAxis dataKey="strike" tick={{ fontSize: 8, fill: '#6b7280' }} />
                    <YAxis tick={{ fontSize: 8, fill: '#6b7280' }} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#111827', border: '1px solid #374151', borderRadius: 6, fontSize: 10 }}
                      formatter={(v: number) => [v.toFixed(2), 'DEX']}
                    />
                    <ReferenceLine y={0} stroke="#374151" />
                    <Bar dataKey="dex" radius={[2, 2, 0, 0]}>
                      {strikeDexData.map((entry, i) => (
                        <Cell key={i} fill={entry.color} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>

          {/* DEX / VEX summary metrics */}
          {dex && Object.keys(dex).length > 0 && (
            <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
              <div className="text-[10px] text-gray-500 font-bold uppercase tracking-widest mb-3">DEX / Dealer Summary</div>
              <div className="grid grid-cols-2 gap-2 md:grid-cols-4 text-[10px]">
                {(['total_dex', 'call_dex', 'put_dex', 'net_dex', 'zero_delta', 'dealer_bias'] as const).map(k => {
                  const v = (dex as Record<string, unknown>)[k]
                  if (v == null) return null
                  const fmt = typeof v === 'number'
                    ? Math.abs(Number(v)) > 1e6 ? `${(Number(v) / 1e6).toFixed(1)}M` : Number(v).toFixed(2)
                    : String(v)
                  return <Stat key={k} label={k.replace(/_/g, ' ')} value={fmt} />
                })}
              </div>
            </div>
          )}

          {/* Positioning */}
          {result.positioning && Object.keys(result.positioning).length > 0 && (
            <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
              <div className="text-[10px] text-gray-500 font-bold uppercase tracking-widest mb-3">Dealer Positioning</div>
              <div className="grid grid-cols-2 gap-2 md:grid-cols-4 text-[10px]">
                {Object.entries(result.positioning).map(([k, v]) => (
                  <Stat key={k} label={k.replace(/_/g, ' ')} value={typeof v === 'number' ? v.toFixed(2) : String(v)} />
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// ── Tab: Tracker ──────────────────────────────────────────────────────────────

function TrackerTab() {
  const [loading, setLoading] = useState(false)
  const [error,   setError]   = useState('')
  const [stats,   setStats]   = useState<TrackerStats | null>(null)

  const loadStats = useCallback(async () => {
    setLoading(true); setError('')
    try {
      const { data } = await axios.get<TrackerStats>(`${API}/tracker/stats`)
      setStats(data)
    } catch (e: unknown) {
      const msg = (e as { response?: { data?: { detail?: string } }; message?: string })
      setError(msg?.response?.data?.detail ?? msg?.message ?? 'Load failed')
    } finally {
      setLoading(false)
    }
  }, [])

  const overall = stats?.overall as Record<string, number | number[]> | undefined
  const equity  = overall?.equity_curve as number[] | undefined

  // Equity curve data
  const equityData = equity?.map((v, i) => ({ index: i + 1, pnl: v })) ?? []

  const pnlColor = (v: number) => v >= 0 ? '#22c55e' : '#ef4444'

  return (
    <div className="space-y-4">
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 flex items-center justify-between">
        <div>
          <div className="text-sm font-bold text-gray-100">Trade Tracker</div>
          <div className="text-[10px] text-gray-500">Performance journal & analytics</div>
        </div>
        <button
          onClick={loadStats}
          disabled={loading}
          className="bg-blue-700 hover:bg-blue-600 disabled:opacity-50 text-white text-xs font-bold px-4 py-2 rounded-lg transition-colors"
        >
          {loading ? '…' : '↻ Refresh'}
        </button>
      </div>

      {loading && <LoadSpinner />}
      {error   && <ErrorBox msg={error} />}

      {stats && overall && (
        <div className="space-y-4">
          {/* Overall stats grid */}
          <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
            {[
              { k: 'total_trades', label: 'Total Trades' },
              { k: 'win_rate',     label: 'Win Rate',    fmt: (v: number) => `${(v * 100).toFixed(1)}%` },
              { k: 'expectancy',   label: 'Expectancy',  fmt: (v: number) => `$${v.toFixed(2)}` },
              { k: 'profit_factor',label: 'Prof. Factor' },
              { k: 'total_pnl',    label: 'Total P&L',   fmt: (v: number) => `$${v.toFixed(2)}` },
              { k: 'max_drawdown', label: 'Max DD',      fmt: (v: number) => `$${v.toFixed(2)}` },
              { k: 'sharpe_ratio', label: 'Sharpe' },
              { k: 'avg_win',      label: 'Avg Win',     fmt: (v: number) => `$${v.toFixed(2)}` },
            ].map(({ k, label, fmt }) => {
              const v = overall[k]
              if (v == null || Array.isArray(v)) return null
              const displayed = fmt ? fmt(v as number) : String(v)
              const color =
                k === 'total_pnl'    ? pnlColor(v as number) :
                k === 'win_rate'     ? (v as number) >= 0.5 ? '#22c55e' : '#ef4444' :
                k === 'profit_factor'? (v as number) >= 1.5 ? '#22c55e' : '#9ca3af' :
                undefined
              return (
                <div key={k} className="bg-gray-900 border border-gray-800 rounded-lg p-3">
                  <div className="text-[9px] text-gray-500 uppercase tracking-wide mb-1">{label}</div>
                  <div className="text-sm font-bold" style={color ? { color } : { color: '#e5e7eb' }}>{displayed}</div>
                </div>
              )
            })}
          </div>

          {/* Equity curve */}
          {equityData.length > 1 && (
            <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
              <div className="text-[10px] text-gray-500 font-bold uppercase tracking-widest mb-3">Equity Curve</div>
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={equityData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                  <XAxis dataKey="index" tick={{ fontSize: 8, fill: '#6b7280' }} />
                  <YAxis tick={{ fontSize: 8, fill: '#6b7280' }} tickFormatter={v => `$${v}`} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#111827', border: '1px solid #374151', borderRadius: 6, fontSize: 10 }}
                    formatter={(v: number) => [`$${v.toFixed(2)}`, 'Cumulative P&L']}
                  />
                  <ReferenceLine y={0} stroke="#374151" />
                  <Bar dataKey="pnl" radius={[2, 2, 0, 0]}>
                    {equityData.map((entry, i) => (
                      <Cell key={i} fill={entry.pnl >= 0 ? '#22c55e' : '#ef4444'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* By confidence */}
          {stats.by_confidence && (
            <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
              <div className="text-[10px] text-gray-500 font-bold uppercase tracking-widest mb-3">By Confidence Level</div>
              <div className="space-y-2">
                {Object.entries(stats.by_confidence).map(([level, s]) => (
                  s.total_trades > 0 && (
                    <div key={level} className="flex items-center gap-3">
                      <Badge val={level} />
                      <div className="flex-1">
                        <ScoreBar score={s.win_rate} />
                      </div>
                      <span className="text-[10px] font-mono text-gray-400 w-12 text-right">{s.total_trades}T</span>
                      <span className={`text-[10px] font-mono font-bold w-16 text-right ${s.total_pnl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                        ${s.total_pnl.toFixed(0)}
                      </span>
                    </div>
                  )
                ))}
              </div>
            </div>
          )}

          {/* By regime */}
          {stats.by_regime && Object.keys(stats.by_regime).some(k => stats.by_regime[k]?.total_trades > 0) && (
            <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
              <div className="text-[10px] text-gray-500 font-bold uppercase tracking-widest mb-3">By GEX Regime</div>
              <div className="space-y-2">
                {Object.entries(stats.by_regime).map(([regime, s]) => (
                  s.total_trades > 0 && (
                    <div key={regime} className="flex items-center gap-3">
                      <span className="text-[9px] font-bold text-gray-400 w-20 shrink-0">{regime}</span>
                      <div className="flex-1">
                        <ScoreBar score={s.win_rate} />
                      </div>
                      <span className="text-[10px] font-mono text-gray-400 w-10 text-right">{s.total_trades}T</span>
                      <span className={`text-[10px] font-mono font-bold w-16 text-right ${s.total_pnl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                        ${s.total_pnl.toFixed(0)}
                      </span>
                    </div>
                  )
                ))}
              </div>
            </div>
          )}

          {/* Open trades */}
          {stats.open_trades && stats.open_trades.length > 0 && (
            <div className="space-y-2">
              <div className="text-[10px] text-gray-500 font-bold uppercase tracking-widest">Open Trades ({stats.open_trades.length})</div>
              <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-gray-800 text-[9px] text-gray-600 uppercase">
                      <th className="px-3 py-2 text-left">ID</th>
                      <th className="px-3 py-2 text-left">Symbol</th>
                      <th className="px-3 py-2 text-left">Type</th>
                      <th className="px-3 py-2 text-right">Entry</th>
                      <th className="px-3 py-2 text-right">Shares</th>
                      <th className="px-3 py-2 text-right">Stop</th>
                      <th className="px-3 py-2 text-right">Target</th>
                    </tr>
                  </thead>
                  <tbody>
                    {stats.open_trades.map((t: unknown) => {
                      const trade = t as Record<string, unknown>
                      return (
                        <tr key={String(trade.id)} className="border-b border-gray-800/50">
                          <td className="px-3 py-2 font-mono text-gray-500">{String(trade.id)}</td>
                          <td className="px-3 py-2 font-bold text-white">{String(trade.symbol)}</td>
                          <td className="px-3 py-2"><Badge val={String(trade.signal_type)} /></td>
                          <td className="px-3 py-2 text-right font-mono text-gray-200">${Number(trade.entry_price).toFixed(2)}</td>
                          <td className="px-3 py-2 text-right font-mono text-gray-400">{String(trade.shares)}</td>
                          <td className="px-3 py-2 text-right font-mono text-red-400">${Number(trade.stop_price).toFixed(2)}</td>
                          <td className="px-3 py-2 text-right font-mono text-emerald-400">${Number(trade.target_price).toFixed(2)}</td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}

      {!stats && !loading && !error && (
        <div className="text-center py-12 text-xs text-gray-600">
          Click Refresh to load performance statistics
        </div>
      )}
    </div>
  )
}

// ── Tab: Pounce Scanner ───────────────────────────────────────────────────────

type PounceStatus = 'READY' | 'NEAR' | 'WATCH'

interface PounceGem {
  price:        number
  strength:     number
  sources:      string[]
  horizons:     string[]
  type:         string
  distance_pct: number
  atr_distance: number
  gem:          boolean
  ready:        boolean
  trade_idea:   { direction: string; entry: number; stop: number; target: number; rr: number } | null
}

interface OptionsRec {
  strike:             number
  expiration:         string
  dte:                number
  delta:              number
  gamma:              number
  iv_pct:             number
  contract_price:     number
  cost_per_contract:  number
  oi:                 number
  target_stock:       number
  move_needed:        number
  move_pct:           number
}

interface GexData {
  zero_gamma:    number | null
  call_wall:     number | null
  put_wall:      number | null
  max_pain:      number | null
  regime:        string
  net_gex:       number
  near_call_wall: boolean
  near_put_wall:  boolean
  near_flip:      boolean
}

interface ScanSetup {
  symbol:           string
  price:            number
  atr:              number
  atr_pct:          number
  trend:            string
  is_inverse_etf:   boolean
  play_type:        string
  status:           PounceStatus
  confluence_score: number
  gem_count:        number
  ready_count:      number
  top_gem:          PounceGem | null
  ready_gems:       PounceGem[]
  gex:              GexData | null
  options_rec:      OptionsRec | null
  strategies_matched: string[]
  strategy_quality?: Record<string, { win_rate: number; profit_factor: number; total_trades: number }>
  mtf?: {
    '15m': string; '1H': string; '4H': string; '1D': string
    aligned_bull: boolean; aligned_bear: boolean; score: number
  }
  spark_bars?: SparkBar[]   // embedded in scan result — no extra fetch needed
  patterns?:   ScanPattern[]
}

interface ScanPattern {
  name:       string
  confidence: number  // 0–1
  desc:       string
}

// Keep alias for compat
type PounceSetup = ScanSetup

interface PounceScanResult {
  cached:   boolean
  timestamp: number
  scanned:  number
  surfaced: number
  ready:    number
  near:     number
  watch:    number
  results:  ScanSetup[]
}

interface SparkBar {
  date:   string
  open:   number
  high:   number
  low:    number
  close:  number
  volume: number
}

const PRICE_BANDS = [
  { label: 'All',     min: 0,    max: Infinity },
  { label: '$0–3',    min: 0,    max: 3 },
  { label: '$4–10',   min: 4,    max: 10 },
  { label: '$10–20',  min: 10,   max: 20 },
  { label: '$20–50',  min: 20,   max: 50 },
  { label: '$50–500', min: 50,   max: 500 },
]

// ── GEX on-demand loader ──────────────────────────────────────────────────────
// Fetches GEX data for a single symbol only when the user clicks "Load GEX".
// This avoids hammering Yahoo options during the mass scan.

function GexInlineLoader({ symbol }: { symbol: string }) {
  const [gex,     setGex]     = useState<GexData | null>(null)
  const [loading, setLoading] = useState(false)
  const [loaded,  setLoaded]  = useState(false)

  const load = async (e: React.MouseEvent) => {
    e.stopPropagation()
    if (loaded || loading) return
    setLoading(true)
    try {
      const res = await axios.get(`${API}/gex/${symbol}`)
      const d = res.data
      if (d?.gex) {
        const g = d.gex
        const price = res.data.spot ?? 0
        setGex({
          zero_gamma:    g.zero_gamma    ?? null,
          call_wall:     g.call_wall     ?? null,
          put_wall:      g.put_wall      ?? null,
          max_pain:      g.max_pain      ?? null,
          regime:        g.regime        ?? 'neutral',
          net_gex:       g.net_gex       ?? 0,
          near_call_wall: g.call_wall != null && Math.abs(price - g.call_wall) / Math.max(price, 1) < 0.015,
          near_put_wall:  g.put_wall  != null && Math.abs(price - g.put_wall)  / Math.max(price, 1) < 0.015,
          near_flip:      g.zero_gamma != null && Math.abs(price - g.zero_gamma) / Math.max(price, 1) < 0.01,
        })
      }
    } catch { /* no options data — silent */ }
    setLoading(false)
    setLoaded(true)
  }

  if (!loaded && !loading) {
    return (
      <button
        onClick={load}
        className="w-full text-left px-2 py-1 rounded border border-purple-900/30 text-[8px] text-purple-500 hover:text-purple-300 hover:border-purple-700/50 transition-colors"
      >
        ⚡ Load GEX / Dealer Positioning
      </button>
    )
  }

  if (loading) {
    return <div className="text-[8px] text-purple-500 animate-pulse px-1">⚡ Loading GEX…</div>
  }

  if (!gex) {
    return <div className="text-[8px] text-gray-700 px-1">No options data</div>
  }

  const regimeColor = gex.regime === 'above_flip' ? '#22c55e' : gex.regime === 'below_flip' ? '#ef4444' : '#f59e0b'
  const regimeLabel = gex.regime === 'above_flip' ? '↑ Above γFlip' : gex.regime === 'below_flip' ? '↓ Below γFlip' : '≈ At γFlip'

  return (
    <div className="rounded p-2 border border-purple-900/40 bg-purple-950/10 space-y-1.5" onClick={e => e.stopPropagation()}>
      <div className="flex items-center justify-between">
        <span className="text-[8px] font-black text-purple-400 uppercase tracking-wide">⚡ GEX / Dealer</span>
        <span className="text-[8px] font-bold px-1.5 py-0.5 rounded" style={{ color: regimeColor, backgroundColor: regimeColor + '18' }}>
          {regimeLabel}
        </span>
      </div>
      <div className="grid grid-cols-4 gap-1 text-center">
        {gex.put_wall != null && (
          <div className={gex.near_put_wall ? 'rounded px-1 py-0.5 border border-emerald-700/40 bg-emerald-950/20' : ''}>
            <div className="text-[7px] text-gray-600">Put Wall</div>
            <div className="text-[9px] font-mono font-bold text-emerald-400">${gex.put_wall.toFixed(0)}</div>
            {gex.near_put_wall && <div className="text-[6px] text-emerald-600">◉ NEAR</div>}
          </div>
        )}
        {gex.zero_gamma != null && (
          <div className={gex.near_flip ? 'rounded px-1 py-0.5 border border-yellow-700/40 bg-yellow-950/20' : ''}>
            <div className="text-[7px] text-gray-600">γ Flip</div>
            <div className="text-[9px] font-mono font-bold text-yellow-400">${gex.zero_gamma.toFixed(0)}</div>
            {gex.near_flip && <div className="text-[6px] text-yellow-600">◉ NEAR</div>}
          </div>
        )}
        {gex.call_wall != null && (
          <div className={gex.near_call_wall ? 'rounded px-1 py-0.5 border border-red-700/40 bg-red-950/20' : ''}>
            <div className="text-[7px] text-gray-600">Call Wall</div>
            <div className="text-[9px] font-mono font-bold text-red-400">${gex.call_wall.toFixed(0)}</div>
            {gex.near_call_wall && <div className="text-[6px] text-red-600">◉ NEAR</div>}
          </div>
        )}
        {gex.max_pain != null && (
          <div>
            <div className="text-[7px] text-gray-600">Max Pain</div>
            <div className="text-[9px] font-mono font-bold text-gray-400">${gex.max_pain.toFixed(0)}</div>
          </div>
        )}
      </div>
    </div>
  )
}

// ── Enrichment on-demand loader (MTF + strategies) ───────────────────────────
// Fetches /api/quant/enrich/{symbol} per card.  Avoids running _mtf_alignment
// and _detect_strategy during the mass scan (each makes 2-3 network calls).

interface EnrichData {
  mtf: { '15m': string; '1H': string; '4H': string; '1D': string; aligned_bull: boolean; aligned_bear: boolean; score: number }
  strategies_matched: string[]
  strategy_quality: Record<string, { win_rate: number; profit_factor: number; total_trades: number }>
  gex: GexData | null
}

function EnrichInlineLoader({ symbol }: { symbol: string }) {
  const [data,    setData]    = useState<EnrichData | null>(null)
  const [loading, setLoading] = useState(false)
  const [loaded,  setLoaded]  = useState(false)

  const load = async (e: React.MouseEvent) => {
    e.stopPropagation()
    if (loaded || loading) return
    setLoading(true)
    try {
      const res = await axios.get(`${API}/enrich/${symbol}`)
      setData(res.data)
    } catch { /* silent */ }
    setLoading(false)
    setLoaded(true)
  }

  if (!loaded && !loading) {
    return (
      <button
        onClick={load}
        className="w-full text-left px-2 py-1 rounded border border-gray-800 text-[8px] text-gray-500 hover:text-violet-300 hover:border-violet-800/50 transition-colors"
      >
        ◎ Load MTF + Strategy Analysis
      </button>
    )
  }

  if (loading) {
    return <div className="text-[8px] text-violet-500 animate-pulse px-1">◎ Analysing…</div>
  }

  if (!data) {
    return <div className="text-[8px] text-gray-700 px-1">Enrichment unavailable</div>
  }

  const mtf = data.mtf
  return (
    <div className="space-y-1.5" onClick={e => e.stopPropagation()}>
      {/* MTF Alignment */}
      <div className="flex items-center gap-1.5 flex-wrap">
        {(['15m', '1H', '4H', '1D'] as const).map(tf => {
          const d = mtf[tf]
          const c = d === 'bull' ? '#22c55e' : d === 'bear' ? '#ef4444' : '#6b7280'
          const icon = d === 'bull' ? '▲' : d === 'bear' ? '▼' : '—'
          return (
            <div key={tf} className="flex items-center gap-0.5 bg-gray-800/60 rounded px-1.5 py-0.5">
              <span className="text-[8px] text-gray-500">{tf}</span>
              <span className="text-[8px] font-bold" style={{ color: c }}>{icon}</span>
            </div>
          )
        })}
        {mtf.aligned_bull && (
          <span className="text-[8px] font-black text-emerald-400 bg-emerald-950 border border-emerald-700 px-1.5 py-0.5 rounded">
            🔥 ALL BULL
          </span>
        )}
        {mtf.aligned_bear && (
          <span className="text-[8px] font-black text-red-400 bg-red-950 border border-red-700 px-1.5 py-0.5 rounded">
            ❄ ALL BEAR
          </span>
        )}
        <span className="text-[9px] text-gray-600 ml-auto">MTF {mtf.score > 0 ? '+' : ''}{mtf.score}</span>
      </div>

      {/* Strategy badges */}
      {data.strategies_matched.length > 0 && (
        <div className="flex gap-1 flex-wrap items-center">
          {data.strategies_matched.map(s => {
            const q = data.strategy_quality?.[s]
            const pfColor = q
              ? q.profit_factor >= 1.5 ? '#22c55e' : q.profit_factor >= 1.0 ? '#eab308' : '#ef4444'
              : '#a78bfa'
            return (
              <div key={s} className="flex items-center gap-1 bg-violet-950 border border-violet-800/50 rounded px-1.5 py-0.5">
                <span className="text-[8px] font-bold text-violet-300">{s}</span>
                {q && q.total_trades > 0 ? (
                  <span className="text-[8px] font-bold" style={{ color: pfColor }}>
                    {q.win_rate}%WR · PF{q.profit_factor}
                  </span>
                ) : (
                  <span className="text-[8px] text-emerald-500">✓</span>
                )}
              </div>
            )
          })}
        </div>
      )}
      {data.strategies_matched.length === 0 && (
        <div className="text-[8px] text-gray-700 px-1">No strategy signal active</div>
      )}
    </div>
  )
}

function StatusBadge({ status }: { status: PounceStatus }) {
  const cfg = {
    READY: { cls: 'text-emerald-400 bg-emerald-950 border-emerald-700 animate-pulse', icon: '✦' },
    NEAR:  { cls: 'text-amber-400 bg-amber-950 border-amber-800', icon: '⚡' },
    WATCH: { cls: 'text-gray-400 bg-gray-800 border-gray-700', icon: '◎' },
  }[status]
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[9px] font-black border ${cfg.cls}`}>
      {cfg.icon} {status}
    </span>
  )
}

function TrendIcon({ trend }: { trend: string }) {
  if (trend === 'up')   return <span className="text-emerald-400 font-bold text-xs">↑</span>
  if (trend === 'down') return <span className="text-red-400 font-bold text-xs">↓</span>
  return <span className="text-gray-500 text-xs">→</span>
}

// ── Candlestick chart (lightweight-charts) ────────────────────────────────────
// Reusable component used by both the hover card and the symbol drawer.
// `compact` mode hides axes and disables scroll/zoom (for the tiny tooltip).
// `levels` draws horizontal price lines (support, entry, stop, target).

interface ChartLevel {
  price: number
  color: string
  title: string
}

function CandleChart({
  bars, height, levels, compact = false, bgColor = '#111827',
}: {
  bars:     SparkBar[]
  height:   number
  levels?:  ChartLevel[]
  compact?: boolean
  bgColor?: string
}) {
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const el = containerRef.current
    if (!el || !bars.length) return

    const chart = createChart(el, {
      width:  el.clientWidth || 240,
      height,
      layout: {
        background: { type: ColorType.Solid, color: bgColor },
        textColor:  '#6b7280',
        fontSize:   10,
      },
      grid: {
        vertLines: { color: '#1f2937' },
        horzLines: { color: '#1f2937' },
      },
      crosshair:  { mode: 1 },
      rightPriceScale: { borderColor: '#374151', visible: !compact, textColor: '#6b7280' },
      leftPriceScale:  { visible: false },
      timeScale: {
        borderColor:  '#374151',
        visible:      !compact,
        timeVisible:  false,
        fixLeftEdge:  true,
        fixRightEdge: true,
      },
      handleScroll: !compact,
      handleScale:  !compact,
    })

    const series = chart.addCandlestickSeries({
      upColor:         '#22c55e',
      downColor:       '#ef4444',
      borderUpColor:   '#22c55e',
      borderDownColor: '#ef4444',
      wickUpColor:     '#22c55e',
      wickDownColor:   '#ef4444',
    })

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    series.setData(bars.map(b => ({ time: b.date as any, open: b.open, high: b.high, low: b.low, close: b.close })))

    levels?.forEach(lvl => {
      series.createPriceLine({
        price:            lvl.price,
        color:            lvl.color,
        lineWidth:        1,
        lineStyle:        LineStyle.Dashed,
        axisLabelVisible: !compact,
        title:            lvl.title,
      })
    })

    chart.timeScale().fitContent()

    const ro = new ResizeObserver(() => {
      if (el.clientWidth > 0) chart.applyOptions({ width: el.clientWidth })
    })
    ro.observe(el)

    return () => { ro.disconnect(); chart.remove() }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bars, height, compact, bgColor, JSON.stringify(levels)])

  if (!bars.length) {
    return (
      <div style={{ height }} className="flex items-center justify-center">
        <div className="w-4 h-4 border border-emerald-500 border-t-transparent rounded-full animate-spin" />
      </div>
    )
  }

  return <div ref={containerRef} style={{ width: '100%', height }} />
}

// ── Hover sparkline tooltip ────────────────────────────────────────────────────
// Floats to the right of the card (or left if near edge), shows 60-day
// interactive candlestick. pointer-events:none so mouse-leave fires correctly.

// Confidence → colour pill
const CONF_COLOR: Record<string, string> = {
  high:   'bg-emerald-900/60 text-emerald-300 border-emerald-700/50',
  medium: 'bg-amber-900/50   text-amber-300   border-amber-700/40',
  low:    'bg-gray-800       text-gray-400    border-gray-700/40',
}
function confTier(c: number) {
  return c >= 0.85 ? 'high' : c >= 0.75 ? 'medium' : 'low'
}

function SparkHoverCard({ symbol, bars, patterns, rect }: {
  symbol:   string
  bars:     SparkBar[] | null
  patterns: ScanPattern[]
  rect:     DOMRect
}) {
  const top2      = patterns.slice(0, 2)
  const CHART_H   = 140
  const PILLS_H   = top2.length ? 28 : 0
  const TOOLTIP_W = 268
  const TOOLTIP_H = CHART_H + 36 + PILLS_H
  const margin    = 8
  const spaceRight = window.innerWidth - rect.right
  const left = spaceRight >= TOOLTIP_W + margin
    ? rect.right + margin
    : rect.left - TOOLTIP_W - margin
  const top = Math.min(rect.top, window.innerHeight - TOOLTIP_H - margin)

  return (
    <div
      style={{ position: 'fixed', top, left, width: TOOLTIP_W, zIndex: 9999, pointerEvents: 'none' }}
      className="bg-gray-900 border border-gray-700/80 rounded-xl overflow-hidden shadow-2xl"
    >
      <div className="text-[9px] font-bold text-gray-500 px-3 pt-2.5 pb-1">{symbol} · 60-day candlestick</div>
      <CandleChart bars={bars ?? []} height={CHART_H} compact bgColor="#111827" />
      {top2.length > 0 && (
        <div className="flex gap-1.5 px-3 py-2">
          {top2.map(p => (
            <span
              key={p.name}
              className={`text-[9px] font-bold px-2 py-0.5 rounded-full border ${CONF_COLOR[confTier(p.confidence)]}`}
              title={p.desc}
            >
              {p.name}
            </span>
          ))}
        </div>
      )}
    </div>
  )
}

// ── Symbol drawer — 2nd-level drill-down ──────────────────────────────────────
// Slides in from the right when a scan card is clicked.
// Shows: price chart with level overlays, trade idea, MTF, GEX, strategies.
// "→ Full Chart" button navigates to the full chart page.

interface DrawerEnrich {
  mtf: { '15m': string; '1H': string; '4H': string; '1D': string; aligned_bull: boolean; aligned_bear: boolean; score: number } | null
  strategies_matched: string[]
  strategy_quality:   Record<string, { win_rate: number; profit_factor: number; total_trades: number }>
  gex: {
    zero_gamma:      number | null
    call_wall:       number | null
    put_wall:        number | null
    max_pain:        number | null
    net_gex:         number | null
    dealer_position: string | null
  } | null
}

function SymbolDrawer({ setup, onClose }: { setup: ScanSetup; onClose: () => void }) {
  const navigate = useNavigate()
  const [sparkBars,     setSparkBars]     = useState<SparkBar[] | null>(null)
  const [enrich,        setEnrich]        = useState<DrawerEnrich | null>(null)
  const [enrichLoading, setEnrichLoading] = useState(false)

  useEffect(() => {
    // spark_bars are already in the scan result — no extra fetch needed
    setSparkBars(setup.spark_bars?.length ? setup.spark_bars : null)
    setEnrich(null)
    // Only fetch enrichment (MTF + strategies) — this is the deferred heavy call
    setEnrichLoading(true)
    axios.get(`${API}/enrich/${setup.symbol}`)
      .then(r => setEnrich(r.data))
      .catch(() => {})
      .finally(() => setEnrichLoading(false))
  }, [setup.symbol])

  const gem  = setup.top_gem
  const idea = gem?.trade_idea ?? null

  // GEX: prefer scan-level data (already fetched), supplement with enrich fallback
  const rawGex = setup.gex ?? (enrich?.gex as typeof setup.gex | null)

  const strats = enrich?.strategies_matched ?? setup.strategies_matched
  const qual   = enrich?.strategy_quality   ?? setup.strategy_quality ?? {}

  return (
    <>
      {/* Backdrop */}
      <div className="fixed inset-0 bg-black/65 z-40" onClick={onClose} />

      {/* Drawer panel */}
      <div className="fixed right-0 top-0 h-full w-full max-w-[520px] bg-gray-950 border-l border-gray-800/70 z-50 flex flex-col shadow-2xl">

        {/* ── Sticky header ── */}
        <div className="flex items-center justify-between px-5 py-3.5 border-b border-gray-800 bg-gray-950 flex-shrink-0">
          <div className="flex items-center gap-2 min-w-0">
            <span className="text-xl font-black text-white tracking-tight">{setup.symbol}</span>
            {setup.is_inverse_etf
              ? <span className="text-[9px] font-bold text-red-400 bg-red-950 border border-red-800 px-1.5 py-0.5 rounded">BEAR</span>
              : <TrendIcon trend={setup.trend} />
            }
            <span className="font-mono font-bold text-gray-200">${setup.price.toFixed(2)}</span>
            <span className="text-[10px] text-gray-600">{setup.atr_pct.toFixed(1)}% ATR</span>
          </div>
          <div className="flex items-center gap-2 flex-shrink-0">
            <button
              onClick={() => navigate(`/chart?symbol=${setup.symbol}`)}
              className="px-3 py-1.5 text-[11px] font-bold bg-emerald-700 hover:bg-emerald-600 text-white rounded-lg transition-colors flex items-center gap-1"
            >
              → Full Chart
            </button>
            <button
              onClick={onClose}
              className="w-7 h-7 flex items-center justify-center rounded-lg text-gray-500 hover:text-white hover:bg-gray-800 transition-colors text-base"
            >
              ✕
            </button>
          </div>
        </div>

        {/* ── Scrollable content ── */}
        <div className="flex-1 overflow-y-auto p-5 space-y-5">

          {/* Interactive candlestick chart with key level overlays */}
          <div>
            <div className="text-[9px] text-gray-500 font-bold uppercase tracking-widest mb-2">60-Day Candlestick</div>
            <div className="rounded-xl overflow-hidden border border-gray-800/50">
              <CandleChart
                bars={sparkBars ?? []}
                height={280}
                bgColor="#0f172a"
                levels={[
                  ...(gem  ? [{ price: gem.price,   color: '#fbbf24', title: '⬟ Support' }] : []),
                  ...(idea ? [
                    { price: idea.entry,  color: '#94a3b8', title: 'Entry'  },
                    { price: idea.stop,   color: '#ef4444', title: 'Stop'   },
                    { price: idea.target, color: '#22c55e', title: 'Target' },
                  ] : []),
                ]}
              />
            </div>
          </div>

          {/* Trade setup — the most important section */}
          {idea && (
            <div>
              <div className="text-[9px] text-gray-500 font-bold uppercase tracking-widest mb-2">Trade Setup</div>
              <div className="grid grid-cols-4 gap-2">
                {[
                  { label: 'Entry',  value: `$${idea.entry.toFixed(2)}`,  color: 'text-gray-100' },
                  { label: 'Stop',   value: `$${idea.stop.toFixed(2)}`,   color: 'text-red-400'  },
                  { label: 'Target', value: `$${idea.target.toFixed(2)}`, color: 'text-emerald-400' },
                  { label: 'R:R',    value: `${idea.rr}:1`,               color: 'text-amber-400' },
                ].map(({ label, value, color }) => (
                  <div key={label} className="bg-gray-900 rounded-xl p-3 border border-gray-800 text-center">
                    <div className="text-[8px] text-gray-600 uppercase tracking-wide mb-1">{label}</div>
                    <div className={`text-sm font-mono font-bold ${color}`}>{value}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Key level (gem info) */}
          {gem && (
            <div>
              <div className="text-[9px] text-gray-500 font-bold uppercase tracking-widest mb-2">⬟ Key Level</div>
              <div className="bg-yellow-950/15 border border-yellow-700/30 rounded-xl p-4 space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="text-yellow-400 font-mono font-black">${gem.price.toFixed(2)}</span>
                    <span className="text-gray-600 text-xs">support</span>
                    <HorizonPill horizons={gem.horizons} />
                  </div>
                  <div className="text-right">
                    <div className="text-[8px] text-gray-600">ATR Distance</div>
                    <div className="text-xs font-mono font-bold text-gray-300">{gem.atr_distance.toFixed(2)} ATR</div>
                  </div>
                </div>
                <div className="flex gap-1 flex-wrap">
                  {gem.sources.map(src => (
                    <span key={src} className="text-[8px] bg-gray-800/80 text-gray-400 px-1.5 py-0.5 rounded border border-gray-700/40">{src}</span>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* ── Detected patterns ── */}
          {(setup.patterns?.length ?? 0) > 0 && (
            <div>
              <div className="text-[9px] text-gray-500 font-bold uppercase tracking-widest mb-2">Detected Patterns</div>
              <div className="space-y-2">
                {setup.patterns!.map(p => (
                  <div
                    key={p.name}
                    className="flex items-start gap-3 bg-gray-900/60 border border-gray-800/50 rounded-xl px-3 py-2.5"
                  >
                    {/* confidence bar */}
                    <div className="flex flex-col items-center gap-1 flex-shrink-0 mt-0.5">
                      <div className="w-1 rounded-full" style={{ height: 36, background: '#1f2937' }}>
                        <div
                          className="w-1 rounded-full"
                          style={{
                            height: `${p.confidence * 36}px`,
                            background: p.confidence >= 0.85 ? '#22c55e' : p.confidence >= 0.75 ? '#f59e0b' : '#6b7280',
                          }}
                        />
                      </div>
                      <span className="text-[8px] text-gray-600">{Math.round(p.confidence * 100)}%</span>
                    </div>
                    <div className="min-w-0">
                      <div className={`text-[11px] font-bold mb-0.5 ${
                        p.confidence >= 0.85 ? 'text-emerald-400'
                        : p.confidence >= 0.75 ? 'text-amber-400'
                        : 'text-gray-400'
                      }`}>{p.name}</div>
                      <div className="text-[9px] text-gray-500 leading-relaxed">{p.desc}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Multi-timeframe alignment */}
          <div>
            <div className="text-[9px] text-gray-500 font-bold uppercase tracking-widest mb-2">Multi-Timeframe</div>
            {enrichLoading && !enrich ? (
              <div className="text-[9px] text-violet-400 animate-pulse">Analysing timeframes…</div>
            ) : enrich?.mtf ? (
              <div className="bg-gray-900/50 border border-gray-800/50 rounded-xl p-3">
                <div className="flex items-center gap-2 flex-wrap">
                  {(['15m', '1H', '4H', '1D'] as const).map(tf => {
                    const d = enrich.mtf![tf]
                    const c = d === 'bull' ? '#22c55e' : d === 'bear' ? '#ef4444' : '#6b7280'
                    const icon = d === 'bull' ? '▲' : d === 'bear' ? '▼' : '—'
                    return (
                      <div key={tf} className="flex items-center gap-1 bg-gray-800 rounded px-2 py-1.5">
                        <span className="text-[9px] text-gray-500 font-bold">{tf}</span>
                        <span className="text-[11px] font-black" style={{ color: c }}>{icon}</span>
                      </div>
                    )
                  })}
                  {enrich.mtf.aligned_bull && (
                    <span className="text-[9px] font-black text-emerald-400 bg-emerald-950 border border-emerald-700 px-2 py-1 rounded">🔥 ALL BULL</span>
                  )}
                  {enrich.mtf.aligned_bear && (
                    <span className="text-[9px] font-black text-red-400 bg-red-950 border border-red-700 px-2 py-1 rounded">❄ ALL BEAR</span>
                  )}
                  <span className="text-[9px] text-gray-600 ml-auto">Score {enrich.mtf.score > 0 ? '+' : ''}{enrich.mtf.score}</span>
                </div>
              </div>
            ) : !enrichLoading && (
              <div className="text-[9px] text-gray-700">MTF data unavailable</div>
            )}
          </div>

          {/* GEX / Dealer positioning */}
          {rawGex && (rawGex.put_wall ?? rawGex.zero_gamma ?? rawGex.call_wall) != null && (
            <div>
              <div className="text-[9px] text-gray-500 font-bold uppercase tracking-widest mb-2">⚡ Dealer Positioning</div>
              <div className="bg-purple-950/10 border border-purple-800/25 rounded-xl p-3">
                <div className="grid grid-cols-4 gap-2 text-center">
                  {rawGex.put_wall != null && (
                    <div>
                      <div className="text-[7px] text-gray-600 mb-1">Put Wall</div>
                      <div className="text-xs font-mono font-bold text-emerald-400">${rawGex.put_wall.toFixed(0)}</div>
                    </div>
                  )}
                  {rawGex.zero_gamma != null && (
                    <div>
                      <div className="text-[7px] text-gray-600 mb-1">γ Flip</div>
                      <div className="text-xs font-mono font-bold text-yellow-400">${rawGex.zero_gamma.toFixed(0)}</div>
                    </div>
                  )}
                  {rawGex.call_wall != null && (
                    <div>
                      <div className="text-[7px] text-gray-600 mb-1">Call Wall</div>
                      <div className="text-xs font-mono font-bold text-red-400">${rawGex.call_wall.toFixed(0)}</div>
                    </div>
                  )}
                  {rawGex.max_pain != null && (
                    <div>
                      <div className="text-[7px] text-gray-600 mb-1">Max Pain</div>
                      <div className="text-xs font-mono font-bold text-gray-400">${rawGex.max_pain.toFixed(0)}</div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Strategies firing */}
          {strats.length > 0 && (
            <div>
              <div className="text-[9px] text-gray-500 font-bold uppercase tracking-widest mb-2">Strategies Firing</div>
              <div className="flex gap-1.5 flex-wrap">
                {strats.map(s => {
                  const q = qual[s]
                  const pfColor = q
                    ? q.profit_factor >= 1.5 ? '#22c55e' : q.profit_factor >= 1.0 ? '#eab308' : '#ef4444'
                    : '#a78bfa'
                  return (
                    <div key={s} className="flex items-center gap-1.5 bg-violet-950/50 border border-violet-800/35 rounded-lg px-2 py-1.5">
                      <span className="text-[9px] font-bold text-violet-300">{s}</span>
                      {q && q.total_trades > 0 && (
                        <span className="text-[8px] font-bold" style={{ color: pfColor }}>
                          {q.win_rate}%WR · PF{q.profit_factor}
                        </span>
                      )}
                    </div>
                  )
                })}
              </div>
            </div>
          )}

          {/* Confluence summary */}
          <div>
            <div className="text-[9px] text-gray-500 font-bold uppercase tracking-widest mb-2">Confluence</div>
            <div className="grid grid-cols-3 gap-2">

              {/* Score card */}
              <div className="group relative bg-gray-900 rounded-xl p-3 border border-gray-800 text-center cursor-help">
                <div className="text-[8px] text-gray-600 uppercase tracking-wide mb-1">Score</div>
                <div className="text-lg font-bold text-amber-400">{setup.confluence_score}</div>
                <div className="pointer-events-none absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-56 bg-gray-900 border border-gray-700 rounded-lg p-2.5 text-left opacity-0 group-hover:opacity-100 transition-opacity z-50 shadow-2xl">
                  <p className="text-[9px] font-bold text-amber-400 mb-1">Confluence Score</p>
                  <p className="text-[9px] text-gray-300 leading-relaxed">How many independent signals agree on this setup. Combines: momentum squeeze, volume absorption, support shelf, higher-lows pattern, and multi-timeframe trend alignment. Max 10 — higher = stronger conviction.</p>
                </div>
              </div>

              {/* Gems card */}
              <div className="group relative bg-gray-900 rounded-xl p-3 border border-gray-800 text-center cursor-help">
                <div className="text-[8px] text-gray-600 uppercase tracking-wide mb-1">Gems</div>
                <div className="text-lg font-bold text-yellow-400">{setup.gem_count}</div>
                <div className="pointer-events-none absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-56 bg-gray-900 border border-gray-700 rounded-lg p-2.5 text-left opacity-0 group-hover:opacity-100 transition-opacity z-50 shadow-2xl">
                  <p className="text-[9px] font-bold text-yellow-400 mb-1">Support Gems ⬟</p>
                  <p className="text-[9px] text-gray-300 leading-relaxed">Price levels where 2+ independent indicators overlap — e.g. weekly pivot + high-volume node + VWAP band. Each gem is a high-probability bounce zone. More gems = deeper, denser support structure below price.</p>
                </div>
              </div>

              {/* Ready card */}
              <div className="group relative bg-gray-900 rounded-xl p-3 border border-gray-800 text-center cursor-help">
                <div className="text-[8px] text-gray-600 uppercase tracking-wide mb-1">Ready</div>
                <div className="text-lg font-bold text-emerald-400">{setup.ready_count}</div>
                <div className="pointer-events-none absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-56 bg-gray-900 border border-gray-700 rounded-lg p-2.5 text-left opacity-0 group-hover:opacity-100 transition-opacity z-50 shadow-2xl">
                  <p className="text-[9px] font-bold text-emerald-400 mb-1">Entry-Ready Gems ✦</p>
                  <p className="text-[9px] text-gray-300 leading-relaxed">Subset of gems where price is currently within 1 ATR of the level, the gem is confirmed, and the last candle closed above it. These are <span className="text-emerald-400 font-semibold">actionable entries right now</span> — not just levels to watch.</p>
                </div>
              </div>

            </div>
          </div>

          {/* Bottom spacer so last item isn't flush at edge */}
          <div className="h-6" />
        </div>
      </div>
    </>
  )
}

function PounceTab() {
  const [data,         setData]         = useState<PounceScanResult | null>(() => {
    // Seed from localStorage on first render — instant display, no flicker
    try {
      const raw = localStorage.getItem(_POUNCE_LS_KEY)
      return raw ? JSON.parse(raw) as PounceScanResult : null
    } catch { return null }
  })
  const [loading,      setLoading]      = useState(false)
  const [error,        setError]        = useState('')
  const [bandIdx,      setBandIdx]      = useState(0)
  const [optTab,       setOptTab]       = useState<'all' | 'optionable'>('all')

  // Progressive disclosure
  const [drawerSetup, setDrawerSetup] = useState<ScanSetup | null>(null)
  const [hoverInfo,   setHoverInfo]   = useState<{ symbol: string; bars: SparkBar[] | null; patterns: ScanPattern[]; rect: DOMRect } | null>(null)
  const sparkCache = useRef<Record<string, SparkBar[]>>({})

  // Seed sparkline cache from scan result — spark_bars come embedded in each
  // ScanSetup so no extra API call is ever needed.  Runs whenever scan data lands.
  useEffect(() => {
    if (!data?.results) return
    data.results.forEach(r => {
      if (r.spark_bars?.length && !sparkCache.current[r.symbol]) {
        sparkCache.current[r.symbol] = r.spark_bars
      }
    })
  }, [data])

  // Instant on hover — data is already in cache from the scan result above.
  const onCardEnter = (setup: ScanSetup, e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect()
    setHoverInfo({ symbol: setup.symbol, bars: sparkCache.current[setup.symbol] ?? null, patterns: setup.patterns ?? [], rect })
  }

  const onCardLeave = () => setHoverInfo(null)

  const runScan = async (force = false) => {
    setLoading(true); setError('')
    try {
      const res = await axios.get(`${API}/pounce`, {
        params: { include_bear: true, force },  // bear ETFs always included
      })
      setData(res.data)
      // Persist fresh result for next visit
      try { localStorage.setItem(_POUNCE_LS_KEY, JSON.stringify(res.data)) } catch { /* quota */ }
    } catch (err: unknown) {
      const msg = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail
      setError(msg ?? 'Scan failed')
    } finally {
      setLoading(false)
    }
  }

  // Auto-load on mount — background refresh (cached data already shown if available)
  useEffect(() => { runScan() }, [])

  const band    = PRICE_BANDS[bandIdx]
  const visible = (data?.results ?? []).filter(r => {
    if (r.status !== 'READY') return false                                   // only READY
    const inBand       = r.price >= band.min && r.price <= band.max
    const isOptionable = r.price >= 50 && !r.is_inverse_etf
    const inOptTab     = optTab === 'all' || isOptionable
    return inBand && inOptTab
  })

  return (
    <div className="space-y-3">
      {/* ── Controls — refresh + two filters ── */}
      <div className="flex items-center gap-2 flex-wrap">
        <button
          onClick={() => runScan(true)}
          disabled={loading}
          className="px-3 py-1.5 text-xs font-bold bg-violet-700 hover:bg-violet-600 disabled:bg-gray-700 text-white rounded transition-colors flex items-center gap-2"
        >
          {loading ? '⏳ Scanning…' : '⟳ Scan'}
        </button>

        {/* Optionable filter */}
        {(['all', 'optionable'] as const).map(o => (
          <button
            key={o}
            onClick={() => setOptTab(o)}
            className={`px-2.5 py-1 rounded text-[10px] font-bold transition-all border ${
              optTab === o
                ? 'bg-blue-900/40 border-blue-600/60 text-blue-300'
                : 'border-gray-700 text-gray-500 hover:text-gray-200 hover:border-gray-600'
            }`}
          >
            {o === 'all' ? 'All' : '⚡ Optionable'}
          </button>
        ))}

        {data && (
          <span className="text-[9px] text-gray-600 ml-auto">
            {data.scanned} scanned · {data.surfaced} surfaced
            {data.cached && ' · cached'}
          </span>
        )}
      </div>

      {/* ── Price band filter ── */}
      <div className="flex gap-1 flex-wrap">
        {PRICE_BANDS.map((b, i) => (
          <button
            key={b.label}
            onClick={() => setBandIdx(i)}
            className={`px-3 py-1 rounded text-[10px] font-bold border transition-all ${
              bandIdx === i
                ? 'bg-emerald-900/30 border-emerald-700/50 text-emerald-300'
                : 'border-gray-700 text-gray-500 hover:text-gray-200'
            }`}
          >
            {b.label}
            <span className="ml-1 opacity-50 text-[9px]">
              ({(data?.results ?? []).filter(r => r.status === 'READY' && r.price >= b.min && r.price <= b.max).length})
            </span>
          </button>
        ))}
      </div>

      {/* Thin refresh bar — shown during background fetch when cached data is visible */}
      <RefreshBar active={loading && data != null} />

      {error && <ErrorBox msg={error} />}

      {loading && !data && <ScanProgressBar loading={loading} />}

      {/* ── Signal cards ── */}
      {visible.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4 gap-2">
          {visible.map(setup => {
            const gem  = setup.top_gem
            const idea = gem?.trade_idea ?? null

            return (
              <div
                key={setup.symbol}
                onClick={() => setDrawerSetup(setup)}
                onMouseEnter={e => onCardEnter(setup, e)}
                onMouseLeave={onCardLeave}
                className="group rounded-xl border p-3 cursor-pointer transition-all duration-150 hover:border-emerald-500/50 hover:bg-emerald-950/10"
                style={{ borderColor: '#22c55e28', backgroundColor: '#0f1a14' }}
              >
                {/* ── Row 1: Symbol · Price · Trend ── */}
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-1.5">
                    <span className="text-base font-black tracking-tight text-white">{setup.symbol}</span>
                    {setup.is_inverse_etf
                      ? <span className="text-[8px] font-bold text-red-400">BEAR</span>
                      : <TrendIcon trend={setup.trend} />
                    }
                  </div>
                  <div className="text-right">
                    <span className="text-sm font-mono font-bold text-gray-200">${setup.price.toFixed(2)}</span>
                    <span className="text-[9px] text-gray-600 ml-1.5">{setup.atr_pct.toFixed(1)}% ATR</span>
                  </div>
                </div>

                {/* ── Row 2: Support level ── */}
                {gem && (
                  <div className="flex items-center gap-1.5 mb-2.5">
                    <span className="text-yellow-500 text-[10px]">⬟</span>
                    <span className="text-[10px] font-mono text-yellow-400/90 font-bold">${gem.price.toFixed(2)}</span>
                    <span className="text-[9px] text-gray-600">support</span>
                    <HorizonPill horizons={gem.horizons} />
                    <span className="ml-auto text-[9px] text-gray-700 font-mono">{gem.atr_distance.toFixed(2)} ATR</span>
                  </div>
                )}

                {/* ── Row 3: Entry / Stop / Target ── */}
                {idea ? (
                  <div className="grid grid-cols-4 gap-1 pt-2 border-t border-gray-800/50">
                    <div>
                      <div className="text-[8px] text-gray-600 uppercase tracking-wide mb-0.5">Entry</div>
                      <div className="text-[11px] font-mono font-bold text-gray-200">${idea.entry.toFixed(2)}</div>
                    </div>
                    <div>
                      <div className="text-[8px] text-red-800 uppercase tracking-wide mb-0.5">Stop</div>
                      <div className="text-[11px] font-mono font-bold text-red-400">${idea.stop.toFixed(2)}</div>
                    </div>
                    <div>
                      <div className="text-[8px] text-emerald-800 uppercase tracking-wide mb-0.5">Target</div>
                      <div className="text-[11px] font-mono font-bold text-emerald-400">${idea.target.toFixed(2)}</div>
                    </div>
                    <div className="text-right">
                      <div className="text-[8px] text-gray-600 uppercase tracking-wide mb-0.5">R:R</div>
                      <div className="text-[11px] font-mono font-bold text-amber-400">{idea.rr}:1</div>
                    </div>
                  </div>
                ) : (
                  <div className="pt-2 border-t border-gray-800/50 text-[9px] text-gray-700">No trade idea</div>
                )}
              </div>
            )
          })}
        </div>
      ) : (
        !loading && (
          <div className="text-center py-12 text-xs text-gray-600">
            {data ? 'No setups match filters — try a different price band' : 'Loading scan…'}
          </div>
        )
      )}

      {/* ── 1st drill-down: hover sparkline tooltip ── */}
      {hoverInfo && (
        <SparkHoverCard symbol={hoverInfo.symbol} bars={hoverInfo.bars} patterns={hoverInfo.patterns} rect={hoverInfo.rect} />
      )}

      {/* ── 2nd drill-down: symbol analysis drawer ── */}
      {drawerSetup && (
        <SymbolDrawer setup={drawerSetup} onClose={() => setDrawerSetup(null)} />
      )}
    </div>
  )
}

// ── Tab: Levels ───────────────────────────────────────────────────────────────

interface LevelItem {
  price:        number
  strength:     number
  sources:      string[]
  horizons:     string[]
  type:         'support' | 'resistance'
  distance_pct: number
  atr_distance: number
  gem:          boolean
  ready:        boolean
  trade_idea:   { direction: string; entry: number; stop: number; target: number; rr: number } | null
}

interface LevelsResult {
  symbol:        string
  current_price: number
  atr:           number
  atr_pct:       number
  trend:         'up' | 'down' | 'neutral'
  levels:        LevelItem[]
  gems:          LevelItem[]
  pounce:        LevelItem[]
}

function horizonColor(horizons: string[]): string {
  const h = new Set(horizons)
  if (h.size >= 2) return '#fbbf24'          // gold — multi-horizon gem
  if (h.has('trend'))  return '#a855f7'       // purple
  if (h.has('day'))    return '#f59e0b'        // amber
  return '#3b82f6'                             // blue — scalp
}

function HorizonPill({ horizons }: { horizons: string[] }) {
  const color = horizonColor(horizons)
  const label = horizons.join('+').toUpperCase()
  return (
    <span
      className="inline-flex items-center px-1.5 py-0.5 rounded text-[8px] font-bold tracking-wide border"
      style={{ color, borderColor: color + '50', backgroundColor: color + '15' }}
    >
      {label}
    </span>
  )
}

function StrengthBar({ strength }: { strength: number }) {
  const pct  = Math.round((strength / 10) * 100)
  const color = strength >= 7 ? '#22c55e' : strength >= 5 ? '#fbbf24' : '#6b7280'
  return (
    <div className="w-10 h-1.5 bg-gray-800 rounded-full overflow-hidden">
      <div className="h-full rounded-full" style={{ width: `${pct}%`, backgroundColor: color }} />
    </div>
  )
}

function LevelsTab() {
  const [inputSym, setInputSym] = useState('SPY')
  const [loading,  setLoading]  = useState(false)
  const [error,    setError]    = useState('')
  const [data,     setData]     = useState<LevelsResult | null>(null)

  async function fetchLevels(e?: React.FormEvent) {
    e?.preventDefault()
    const sym = inputSym.trim().toUpperCase()
    if (!sym) return
    setLoading(true); setError(''); setData(null)
    try {
      const res = await axios.get(`${API}/levels/${sym}`)
      setData(res.data)
    } catch (err: unknown) {
      const msg = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail
      setError(msg ?? 'Failed to load levels')
    } finally {
      setLoading(false)
    }
  }

  const trendIcon  = data?.trend === 'up' ? '↑' : data?.trend === 'down' ? '↓' : '→'
  const trendColor = data?.trend === 'up' ? '#22c55e' : data?.trend === 'down' ? '#ef4444' : '#9ca3af'

  // Split levels into above/below current price for the ladder
  const above = data ? [...data.levels].filter(l => l.type === 'resistance').sort((a, b) => a.price - b.price) : []
  const below = data ? [...data.levels].filter(l => l.type === 'support').sort((a, b) => b.price - a.price) : []

  return (
    <div className="space-y-4">
      {/* ── Symbol input ── */}
      <form onSubmit={fetchLevels} className="flex gap-2 items-center">
        <input
          value={inputSym}
          onChange={e => setInputSym(e.target.value.toUpperCase())}
          placeholder="AAPL"
          className="w-28 bg-gray-800 border border-gray-700 rounded px-2.5 py-1.5 text-xs font-mono text-gray-200 focus:outline-none focus:border-violet-500"
        />
        <button
          type="submit"
          disabled={loading}
          className="px-4 py-1.5 text-xs font-bold bg-violet-700 hover:bg-violet-600 disabled:bg-gray-700 text-white rounded transition-colors"
        >
          {loading ? 'Loading…' : 'Map Levels'}
        </button>
        {data && (
          <span className="text-[10px] text-gray-500 font-mono">
            {data.levels.length} levels · {data.gems.length} gems
          </span>
        )}
      </form>

      {error && <ErrorBox msg={error} />}
      {loading && <LoadSpinner />}

      {data && (
        <div className="space-y-5">
          {/* ── Header strip ── */}
          <div className="flex items-center gap-4 flex-wrap">
            <div>
              <div className="text-[9px] text-gray-500 uppercase tracking-wide">Symbol</div>
              <div className="text-base font-black text-gray-100">{data.symbol}</div>
            </div>
            <div>
              <div className="text-[9px] text-gray-500 uppercase tracking-wide">Price</div>
              <div className="text-base font-mono font-bold text-gray-100">${data.current_price.toFixed(2)}</div>
            </div>
            <div>
              <div className="text-[9px] text-gray-500 uppercase tracking-wide">ATR</div>
              <div className="text-sm font-mono text-gray-400">${data.atr.toFixed(2)} ({data.atr_pct.toFixed(1)}%)</div>
            </div>
            <div>
              <div className="text-[9px] text-gray-500 uppercase tracking-wide">Trend</div>
              <div className="text-sm font-bold" style={{ color: trendColor }}>{trendIcon} {data.trend.toUpperCase()}</div>
            </div>

            {data.pounce.length > 0 && (
              <div className="ml-auto flex items-center gap-2 px-3 py-1.5 rounded-lg bg-emerald-950 border border-emerald-700 animate-pulse">
                <span className="text-base">✦</span>
                <div>
                  <div className="text-[10px] font-black text-emerald-400 tracking-wide">ENTRY READY</div>
                  <div className="text-[9px] text-emerald-600">{data.pounce.length} gem{data.pounce.length > 1 ? 's' : ''} at key level</div>
                </div>
              </div>
            )}
          </div>

          {/* ── Price Ladder ── */}
          <div>
            <div className="text-[9px] text-gray-500 uppercase tracking-wide mb-2">Price Ladder</div>
            <div className="border border-gray-800 rounded-lg overflow-hidden">
              {/* Resistance levels (above price) — show top 8, closest first from bottom */}
              <div className="flex flex-col-reverse">
                {above.slice(0, 8).map((lvl, i) => {
                  const color = lvl.gem ? '#fbbf24' : horizonColor(lvl.horizons)
                  return (
                    <div
                      key={i}
                      className="flex items-center gap-3 px-3 py-1.5 border-b border-gray-800/60"
                      style={{
                        backgroundColor: lvl.ready ? '#052e1680' : lvl.gem ? color + '08' : 'transparent',
                        borderLeft: lvl.gem ? `2px solid ${color}` : '2px solid transparent',
                        boxShadow: lvl.ready ? `0 0 8px ${color}30` : 'none',
                      }}
                    >
                      <HorizonPill horizons={lvl.horizons} />
                      <span className="text-xs font-mono font-bold text-red-400 w-20 text-right">
                        ${lvl.price.toFixed(2)}
                      </span>
                      <StrengthBar strength={lvl.strength} />
                      <span className="text-[9px] text-gray-600 w-10 text-right">+{lvl.distance_pct.toFixed(1)}%</span>
                      <span className="text-[9px] text-gray-600 w-12 text-right">{lvl.atr_distance.toFixed(2)} ATR</span>
                      {lvl.gem && <span className="text-xs text-yellow-400 ml-auto">⬟</span>}
                      {lvl.ready && <span className="text-[9px] font-bold text-emerald-400">READY</span>}
                    </div>
                  )
                })}
              </div>

              {/* Current price marker */}
              <div className="flex items-center gap-3 px-3 py-2 bg-gray-900 border-y-2 border-emerald-700/60">
                <span className="text-[9px] text-emerald-500 uppercase tracking-wide font-bold">Current</span>
                <span className="text-sm font-mono font-black text-emerald-400">${data.current_price.toFixed(2)}</span>
                <div className="flex-1 h-px bg-emerald-700/40" />
                <span className="text-[9px] text-gray-600">ATR ${data.atr.toFixed(2)}</span>
              </div>

              {/* Support levels (below price) — closest first from top */}
              {below.slice(0, 8).map((lvl, i) => {
                const color = lvl.gem ? '#fbbf24' : horizonColor(lvl.horizons)
                return (
                  <div
                    key={i}
                    className="flex items-center gap-3 px-3 py-1.5 border-b border-gray-800/60"
                    style={{
                      backgroundColor: lvl.ready ? '#052e1680' : lvl.gem ? color + '08' : 'transparent',
                      borderLeft: lvl.gem ? `2px solid ${color}` : '2px solid transparent',
                      boxShadow: lvl.ready ? `0 0 8px ${color}30` : 'none',
                    }}
                  >
                    <HorizonPill horizons={lvl.horizons} />
                    <span className="text-xs font-mono font-bold text-emerald-400 w-20 text-right">
                      ${lvl.price.toFixed(2)}
                    </span>
                    <StrengthBar strength={lvl.strength} />
                    <span className="text-[9px] text-gray-600 w-10 text-right">{lvl.distance_pct.toFixed(1)}%</span>
                    <span className="text-[9px] text-gray-600 w-12 text-right">{lvl.atr_distance.toFixed(2)} ATR</span>
                    {lvl.gem && <span className="text-xs text-yellow-400 ml-auto">⬟</span>}
                    {lvl.ready && <span className="text-[9px] font-bold text-emerald-400">READY</span>}
                  </div>
                )
              })}
            </div>

            {/* Legend */}
            <div className="flex gap-4 mt-2 flex-wrap">
              {[
                { color: '#3b82f6', label: 'Scalp' },
                { color: '#f59e0b', label: 'Day' },
                { color: '#a855f7', label: 'Trend' },
                { color: '#fbbf24', label: 'Gem (multi-horizon)' },
              ].map(({ color, label }) => (
                <div key={label} className="flex items-center gap-1.5">
                  <div className="w-2 h-2 rounded-sm" style={{ backgroundColor: color }} />
                  <span className="text-[9px] text-gray-500">{label}</span>
                </div>
              ))}
            </div>
          </div>

          {/* ── Gem Cards ── */}
          {data.gems.length > 0 && (
            <div>
              <div className="text-[9px] text-gray-500 uppercase tracking-wide mb-2">
                ⬟ Confluence Gems ({data.gems.length})
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {data.gems.map((gem, i) => {
                  const color = gem.ready ? '#22c55e' : '#fbbf24'
                  const idea  = gem.trade_idea
                  return (
                    <div
                      key={i}
                      className="rounded-lg p-3 border space-y-2"
                      style={{
                        borderColor: color + '40',
                        backgroundColor: color + '08',
                        boxShadow: gem.ready ? `0 0 12px ${color}20` : 'none',
                      }}
                    >
                      {/* Gem header */}
                      <div className="flex items-start justify-between gap-2">
                        <div className="flex items-center gap-2">
                          <span className="text-yellow-400 text-sm">⬟</span>
                          <span className="text-sm font-mono font-black text-gray-100">
                            ${gem.price.toFixed(2)}
                          </span>
                          <span className={`text-[9px] font-bold ${gem.type === 'support' ? 'text-emerald-400' : 'text-red-400'}`}>
                            {gem.type.toUpperCase()}
                          </span>
                        </div>
                        <div className="flex items-center gap-1.5">
                          <HorizonPill horizons={gem.horizons} />
                          {gem.ready && (
                            <span className="text-[9px] font-black text-emerald-400 bg-emerald-950 border border-emerald-700 px-1.5 py-0.5 rounded animate-pulse">
                              ✦ ENTRY ZONE
                            </span>
                          )}
                        </div>
                      </div>

                      {/* Sources + distance */}
                      <div className="flex items-center gap-2">
                        <span className="text-[9px] text-gray-500 flex-1 truncate">
                          {gem.sources.slice(0, 4).join(' · ')}
                          {gem.sources.length > 4 && ' · …'}
                        </span>
                        <span className="text-[9px] font-mono text-gray-600 shrink-0">
                          {gem.atr_distance.toFixed(2)} ATR away
                        </span>
                      </div>

                      {/* Trade idea */}
                      {idea && (
                        <div className="grid grid-cols-3 gap-1.5 pt-1 border-t border-gray-800/60">
                          <div className="text-center">
                            <div className="text-[8px] text-gray-600 uppercase">Entry</div>
                            <div className="text-[10px] font-mono font-bold text-gray-200">${idea.entry.toFixed(2)}</div>
                          </div>
                          <div className="text-center">
                            <div className="text-[8px] text-red-700 uppercase">Stop</div>
                            <div className="text-[10px] font-mono font-bold text-red-400">${idea.stop.toFixed(2)}</div>
                          </div>
                          <div className="text-center">
                            <div className="text-[8px] text-emerald-700 uppercase">Target</div>
                            <div className="text-[10px] font-mono font-bold text-emerald-400">${idea.target.toFixed(2)}</div>
                          </div>
                          <div className="col-span-3 text-center">
                            <span className="text-[9px] text-gray-500">
                              {idea.direction === 'long' ? '▲ Long' : '▼ Short'} · {idea.rr}:1 R:R
                            </span>
                          </div>
                        </div>
                      )}
                    </div>
                  )
                })}
              </div>
            </div>
          )}
        </div>
      )}

      {!data && !loading && !error && (
        <div className="text-center py-12 text-xs text-gray-600">
          Enter a ticker symbol and click Map Levels to see all key price zones
        </div>
      )}
    </div>
  )
}

// ── Tab: Backtest ─────────────────────────────────────────────────────────────

interface BtStats {
  total: number; wins: number; losses: number; win_rate: number
  avg_pnl_pct: number; avg_win_pct: number; avg_loss_pct: number
  profit_factor: number; total_pnl: number; avg_rr: number
  avg_hold_days: number; best_trade: number; worst_trade: number
  max_drawdown: number
}

interface BtTrade {
  strategy: string; symbol: string; entry_date: string; entry_price: number
  stop: number; target: number; exit_date: string; exit_price: number
  exit_reason: string; pnl_pct: number; pnl_dollar: number; rr_actual: number
  hold_days: number; won: boolean
}

interface BtEquityPoint { date: string; pnl: number; equity: number; symbol: string }

interface BtStratResult {
  name: string; description: string
  stats: BtStats; trades: BtTrade[]; equity_curve: BtEquityPoint[]
}

interface BtResult {
  symbol: string; period: string; bars: number
  strategies: Record<string, BtStratResult>
  combined: { stats: BtStats; equity_curve: BtEquityPoint[]; trade_count: number }
}

interface BtUniverseResult {
  cached: boolean; symbols_tested: number; total_trades: number
  per_strategy: Record<string, { name: string; stats: BtStats }>
  combined: { stats: BtStats; equity_curve: BtEquityPoint[] }
}

function StatCard({ label, value, color }: { label: string; value: string | number; color?: string }) {
  return (
    <div className="bg-gray-800/60 rounded p-2 border border-gray-700/50 text-center">
      <div className="text-[8px] text-gray-500 uppercase tracking-wide">{label}</div>
      <div className="text-sm font-mono font-bold" style={{ color: color ?? '#e5e7eb' }}>{value}</div>
    </div>
  )
}

function MiniEquityCurve({ curve }: { curve: BtEquityPoint[] }) {
  if (curve.length < 2) return null
  const maxEq  = Math.max(...curve.map(p => p.equity))
  const minEq  = Math.min(...curve.map(p => p.equity))
  const range  = maxEq - minEq || 1
  const height = 80
  const width  = Math.min(curve.length * 4, 600)

  const points = curve.map((p, i) => {
    const x = (i / (curve.length - 1)) * width
    const y = height - ((p.equity - minEq) / range) * height
    return `${x},${y}`
  }).join(' ')

  const finalColor = curve[curve.length - 1].equity >= 0 ? '#22c55e' : '#ef4444'

  return (
    <svg width={width} height={height} className="overflow-visible">
      <line x1={0} y1={height - ((0 - minEq) / range) * height}
            x2={width} y2={height - ((0 - minEq) / range) * height}
            stroke="#374151" strokeWidth={1} strokeDasharray="4" />
      <polyline fill="none" stroke={finalColor} strokeWidth={2} points={points} />
    </svg>
  )
}

function BacktestTab() {
  const [mode, setMode]       = useState<'single' | 'universe'>('universe')
  const [symbol, setSymbol]   = useState('AAPL')
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState('')
  const [single, setSingle]   = useState<BtResult | null>(null)
  const [universe, setUniverse] = useState<BtUniverseResult | null>(null)

  async function runSingle() {
    setLoading(true); setError(''); setSingle(null)
    try {
      const res = await axios.get(`${API}/backtest/${symbol.trim().toUpperCase()}`)
      setSingle(res.data)
    } catch (err: unknown) {
      setError((err as any)?.response?.data?.detail ?? 'Backtest failed')
    } finally { setLoading(false) }
  }

  async function runUniverse() {
    setLoading(true); setError(''); setUniverse(null)
    try {
      const res = await axios.get(`${API}/backtest-universe`)
      setUniverse(res.data)
    } catch (err: unknown) {
      setError((err as any)?.response?.data?.detail ?? 'Universe backtest failed')
    } finally { setLoading(false) }
  }

  // Auto-load cached universe backtest on mount (cached 1h — fast if already ran)
  useEffect(() => { runUniverse() }, [])

  function renderStats(s: BtStats) {
    const wrColor = s.win_rate >= 55 ? '#22c55e' : s.win_rate >= 45 ? '#eab308' : '#ef4444'
    const pfColor = s.profit_factor >= 1.5 ? '#22c55e' : s.profit_factor >= 1.0 ? '#eab308' : '#ef4444'
    return (
      <div className="grid grid-cols-4 md:grid-cols-7 gap-2">
        <StatCard label="Trades" value={s.total} />
        <StatCard label="Win Rate" value={`${s.win_rate}%`} color={wrColor} />
        <StatCard label="Profit Factor" value={s.profit_factor} color={pfColor} />
        <StatCard label="Total P&L" value={`$${s.total_pnl.toLocaleString()}`}
                  color={s.total_pnl >= 0 ? '#22c55e' : '#ef4444'} />
        <StatCard label="Avg Win" value={`${s.avg_win_pct}%`} color="#22c55e" />
        <StatCard label="Avg Loss" value={`${s.avg_loss_pct}%`} color="#ef4444" />
        <StatCard label="Avg R:R" value={s.avg_rr} />
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {/* Mode toggle */}
      <div className="flex items-center gap-3">
        <div className="flex gap-1">
          {(['universe', 'single'] as const).map(m => (
            <button key={m} onClick={() => setMode(m)}
              className={`px-3 py-1.5 rounded text-xs font-bold border transition-all ${
                mode === m ? 'bg-violet-700/50 border-violet-600 text-violet-200'
                           : 'border-gray-700 text-gray-500 hover:text-gray-200'
              }`}>
              {m === 'universe' ? '🌐 Full Universe' : '🔬 Single Symbol'}
            </button>
          ))}
        </div>

        {mode === 'single' && (
          <div className="flex gap-2 items-center">
            <input value={symbol} onChange={e => setSymbol(e.target.value.toUpperCase())}
              onKeyDown={e => e.key === 'Enter' && runSingle()}
              className="w-24 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs font-mono text-gray-200 focus:outline-none focus:border-violet-500" />
            <button onClick={runSingle} disabled={loading}
              className="px-3 py-1.5 text-xs font-bold bg-violet-700 hover:bg-violet-600 disabled:bg-gray-700 text-white rounded">
              {loading ? 'Running…' : 'Backtest'}
            </button>
          </div>
        )}

        {mode === 'universe' && (
          <button onClick={runUniverse} disabled={loading}
            className="px-4 py-1.5 text-xs font-bold bg-violet-700 hover:bg-violet-600 disabled:bg-gray-700 text-white rounded flex items-center gap-2">
            {loading ? (
              <><div className="w-3 h-3 border border-white border-t-transparent rounded-full animate-spin" /> Running backtest…</>
            ) : '⟳ Run Universe Backtest'}
          </button>
        )}
      </div>

      {error && <ErrorBox msg={error} />}
      {loading && <LoadSpinner />}

      {/* ── Universe results ── */}
      {mode === 'universe' && universe && (
        <div className="space-y-4">
          <div className="text-[10px] text-gray-500">
            {universe.symbols_tested} symbols · {universe.total_trades} trades · 2 years
            {universe.cached && ' · cached'}
          </div>

          {/* Combined stats */}
          <div>
            <div className="text-[9px] text-gray-500 uppercase tracking-wide mb-2">Combined Performance (all strategies)</div>
            {renderStats(universe.combined.stats)}
          </div>

          {/* Equity curve */}
          {universe.combined.equity_curve.length > 0 && (
            <div>
              <div className="text-[9px] text-gray-500 uppercase tracking-wide mb-2">Equity Curve ($1K risk/trade)</div>
              <div className="bg-gray-900 rounded border border-gray-800 p-3 overflow-x-auto">
                <MiniEquityCurve curve={universe.combined.equity_curve} />
              </div>
            </div>
          )}

          {/* Per-strategy breakdown */}
          <div>
            <div className="text-[9px] text-gray-500 uppercase tracking-wide mb-2">Per Strategy</div>
            <div className="space-y-3">
              {Object.entries(universe.per_strategy).map(([key, strat]) => (
                <div key={key} className="bg-gray-900 rounded-lg border border-gray-800 p-3">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs font-bold text-gray-200">{strat.name}</span>
                    <span className={`text-[10px] font-bold ${strat.stats.profit_factor >= 1.0 ? 'text-emerald-400' : 'text-red-400'}`}>
                      PF {strat.stats.profit_factor} · {strat.stats.win_rate}% WR · {strat.stats.total} trades
                    </span>
                  </div>
                  {renderStats(strat.stats)}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* ── Single symbol results ── */}
      {mode === 'single' && single && (
        <div className="space-y-4">
          <div className="text-xs text-gray-400">
            {single.symbol} · {single.period} · {single.bars} bars
          </div>

          {/* Combined */}
          <div>
            <div className="text-[9px] text-gray-500 uppercase tracking-wide mb-2">Combined</div>
            {renderStats(single.combined.stats)}
          </div>

          {single.combined.equity_curve.length > 0 && (
            <div className="bg-gray-900 rounded border border-gray-800 p-3 overflow-x-auto">
              <MiniEquityCurve curve={single.combined.equity_curve} />
            </div>
          )}

          {/* Per strategy */}
          {Object.entries(single.strategies).map(([key, strat]) => (
            <div key={key} className="bg-gray-900 rounded-lg border border-gray-800 p-3 space-y-2">
              <div className="flex items-center justify-between">
                <div>
                  <span className="text-xs font-bold text-gray-200">{strat.name}</span>
                  <span className="text-[9px] text-gray-500 ml-2">{strat.description}</span>
                </div>
                <span className={`text-[10px] font-bold ${strat.stats.profit_factor >= 1.0 ? 'text-emerald-400' : 'text-red-400'}`}>
                  PF {strat.stats.profit_factor} · {strat.stats.win_rate}% WR
                </span>
              </div>
              {renderStats(strat.stats)}

              {/* Trade log (last 10) */}
              {strat.trades.length > 0 && (
                <div className="mt-2">
                  <div className="text-[8px] text-gray-600 uppercase mb-1">Recent Trades</div>
                  <div className="overflow-x-auto">
                    <table className="w-full text-[9px]">
                      <thead>
                        <tr className="text-gray-600 border-b border-gray-800">
                          <th className="text-left py-1 pr-2">Date</th>
                          <th className="text-right pr-2">Entry</th>
                          <th className="text-right pr-2">Exit</th>
                          <th className="text-right pr-2">P&L%</th>
                          <th className="text-right pr-2">R:R</th>
                          <th className="text-left">Reason</th>
                        </tr>
                      </thead>
                      <tbody>
                        {strat.trades.slice(-15).reverse().map((t, i) => (
                          <tr key={i} className="border-b border-gray-800/40">
                            <td className="py-0.5 pr-2 text-gray-400">{t.entry_date}</td>
                            <td className="text-right pr-2 font-mono text-gray-300">${t.entry_price}</td>
                            <td className="text-right pr-2 font-mono text-gray-300">${t.exit_price}</td>
                            <td className={`text-right pr-2 font-mono font-bold ${t.won ? 'text-emerald-400' : 'text-red-400'}`}>
                              {t.pnl_pct > 0 ? '+' : ''}{t.pnl_pct}%
                            </td>
                            <td className="text-right pr-2 font-mono text-gray-400">{t.rr_actual}</td>
                            <td className="text-gray-500">{t.exit_reason}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {!loading && !single && !universe && !error && (
        <div className="text-center py-12 text-xs text-gray-600">
          Run a backtest to see which strategies make money and which don't
        </div>
      )}
    </div>
  )
}

// ── Combined Analysis Tab (Score + Levels) ────────────────────────────────────

function AnalysisTab() {
  const [view, setView] = useState<'score' | 'levels'>('score')

  return (
    <div className="space-y-3">
      <div className="flex gap-1.5 border-b border-gray-800 pb-2">
        <button
          onClick={() => setView('score')}
          className={`px-3 py-1.5 text-[11px] font-bold rounded transition-all ${
            view === 'score'
              ? 'bg-violet-900/30 text-violet-300 border-b-2 border-violet-500'
              : 'text-gray-500 hover:text-gray-300'
          }`}
        >
          ⚡ Score
        </button>
        <button
          onClick={() => setView('levels')}
          className={`px-3 py-1.5 text-[11px] font-bold rounded transition-all ${
            view === 'levels'
              ? 'bg-amber-900/30 text-amber-300 border-b-2 border-amber-500'
              : 'text-gray-500 hover:text-gray-300'
          }`}
        >
          ◈ Levels
        </button>
      </div>
      {view === 'score' && <ScoreTab />}
      {view === 'levels' && <LevelsTab />}
    </div>
  )
}

// ── Panel (embedded inside SignalsPage) ───────────────────────────────────────
// GEX tab dropped — covered by existing Gamma Sniper in SignalsPage.

type QuantTab = 'pounce' | 'analysis' | 'confluence' | 'patterns' | 'backtest' | 'tracker'

const QUANT_TABS: { id: QuantTab; label: string; icon: string }[] = [
  { id: 'pounce',     label: 'Signals',    icon: '◎' },
  { id: 'analysis',   label: 'Analysis',   icon: '🔍' },
  { id: 'confluence', label: 'Confluence', icon: '◉' },
  { id: 'patterns',   label: 'Patterns',   icon: '📊' },
  { id: 'backtest',   label: 'Backtest',   icon: '📈' },
  { id: 'tracker',    label: 'Tracker',    icon: '□' },
]

export default function QuantPanel() {
  const [tab, setTab] = useState<QuantTab>('pounce')

  return (
    <div className="space-y-3">
      {/* ── Compact tool strip ── */}
      <div className="flex items-center gap-1 flex-wrap">
        {QUANT_TABS.map(t => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[11px] font-bold transition-all border ${
              tab === t.id
                ? 'bg-gray-800 border-gray-600 text-gray-100'
                : 'border-transparent text-gray-600 hover:text-gray-300 hover:bg-gray-800/40'
            }`}
          >
            <span>{t.icon}</span>
            {t.label}
          </button>
        ))}
      </div>

      {tab === 'pounce'     && <PounceTab />}
      {tab === 'analysis'   && <AnalysisTab />}
      {tab === 'confluence' && <ConfluenceFunnel />}
      {tab === 'patterns'   && <PatternScanPage />}
      {tab === 'backtest'   && <BacktestTab />}
      {tab === 'tracker'    && <TrackerTab />}
    </div>
  )
}
