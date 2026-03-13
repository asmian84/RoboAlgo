/**
 * Pattern Scanner — batch scan of all tracked instruments.
 *
 * Shows every active pattern (FORMING / READY / BREAKOUT / COMPLETED)
 * across the full instrument universe, grouped and filterable by category,
 * status, and direction.
 *
 * Results come from GET /api/patterns/scan (5-min backend TTL cache).
 */

import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { usePatternScan } from '../api/hooks'
import { useQueryClient } from '@tanstack/react-query'
import type { PatternScanEntry } from '../types'

// ── Helpers ───────────────────────────────────────────────────────────────────

type Category = 'all' | 'chart' | 'harmonic' | 'gann' | 'wyckoff'
type StatusFilter = 'all' | 'FORMING' | 'READY' | 'BREAKOUT' | 'COMPLETED'
type DirFilter = 'all' | 'bullish' | 'bearish'
type SortKey = 'confidence' | 'symbol' | 'status'

function classifyCategory(entry: PatternScanEntry): 'chart' | 'harmonic' | 'gann' | 'wyckoff' | 'other' {
  const c = (entry.pattern_category || '').toLowerCase()
  if (c === 'chart' || c === 'harmonic' || c === 'gann' || c === 'wyckoff') return c as any
  const n = entry.pattern_name.toLowerCase()
  if (n.includes('gann')) return 'gann'
  if (n.includes('wyckoff') || n.includes('accumulation') || n.includes('distribution')) return 'wyckoff'
  if (n.includes('gartley') || n.includes('bat') || n.includes('butterfly') || n.includes('crab') || n.includes('cypher') || n.includes('harmonic')) return 'harmonic'
  if (n.includes('chair') || n.includes('cup') || n.includes('flag') || n.includes('channel') || n.includes('triangle') || n.includes('compression')) return 'chart'
  return 'other'
}

const CAT_COLOR: Record<string, string> = {
  chart:    '#3b82f6',
  harmonic: '#14b8a6',
  gann:     '#f59e0b',
  wyckoff:  '#a855f7',
  other:    '#9ca3af',
}

const CAT_LABEL: Record<string, string> = {
  chart: 'Chart', harmonic: 'Harmonic', gann: 'Gann', wyckoff: 'Wyckoff', other: '?',
}

const STATUS_STYLE: Record<string, string> = {
  BREAKOUT:  'bg-emerald-900/50 text-emerald-300 border border-emerald-800/40',
  COMPLETED: 'bg-emerald-900/30 text-emerald-400 border border-emerald-800/30',
  READY:     'bg-amber-900/50 text-amber-300 border border-amber-800/40',
  FORMING:   'bg-gray-800/80 text-gray-400 border border-gray-700/40',
}

const STATUS_ORDER: Record<string, number> = {
  BREAKOUT: 0, READY: 1, COMPLETED: 2, FORMING: 3,
}

// ── Component ─────────────────────────────────────────────────────────────────

const SCAN_TFS = ['1d', '1h', '4h', '30m', '15m', '5m', '1m'] as const
type ScanTF = typeof SCAN_TFS[number]

const TF_LABEL: Record<ScanTF, string> = {
  '1d': 'Daily', '1h': '1H', '4h': '4H', '30m': '30m', '15m': '15m', '5m': '5m', '1m': '1m',
}

// Patterns not available on short timeframes
const TF_UNAVAILABLE: Record<string, string[]> = {
  '1m':  ['Wyckoff Structure', 'Chair Pattern'],
  '5m':  ['Wyckoff Structure', 'Chair Pattern'],
  '15m': ['Wyckoff Structure', 'Chair Pattern'],
  '30m': ['Wyckoff Structure'],
}

export default function PatternScanPage() {
  const navigate = useNavigate()
  const qc = useQueryClient()

  const [scanTF,           setScanTF]           = useState<ScanTF>('1d')
  const [categoryFilter,   setCategoryFilter]   = useState<Category>('all')
  const [patternNameFilter, setPatternNameFilter] = useState<string | null>(null)
  const [statusFilter,     setStatusFilter]     = useState<StatusFilter>('all')
  const [dirFilter,        setDirFilter]         = useState<DirFilter>('all')
  const [symbolSearch,     setSymbolSearch]      = useState('')
  const [sortKey,          setSortKey]           = useState<SortKey>('confidence')
  const [sortAsc,          setSortAsc]           = useState(false)

  const tfParam = scanTF === '1d' ? undefined : scanTF
  const { data, isLoading, isFetching, dataUpdatedAt } = usePatternScan(true, tfParam)

  const entries = data ?? []

  // Reset pattern name filter when category changes
  const handleCategoryChange = (cat: Category) => {
    setCategoryFilter(cat)
    setPatternNameFilter(null)
  }

  // Category counts for tab badges
  const catCounts = useMemo(() => {
    const counts: Record<string, number> = { all: 0, chart: 0, harmonic: 0, gann: 0, wyckoff: 0 }
    for (const e of entries) {
      counts.all++
      const c = classifyCategory(e)
      if (c in counts) counts[c]++
    }
    return counts
  }, [entries])

  // Unique pattern names within the active category, with counts
  const patternNames = useMemo(() => {
    const catRows = categoryFilter === 'all'
      ? entries
      : entries.filter(e => classifyCategory(e) === categoryFilter)
    const counts = new Map<string, number>()
    for (const e of catRows) counts.set(e.pattern_name, (counts.get(e.pattern_name) ?? 0) + 1)
    return [...counts.entries()].sort((a, b) => b[1] - a[1])
  }, [entries, categoryFilter])

  // Filter + sort
  const filtered = useMemo(() => {
    let rows = entries.filter(e => {
      if (categoryFilter !== 'all' && classifyCategory(e) !== categoryFilter) return false
      if (patternNameFilter && e.pattern_name !== patternNameFilter) return false
      if (statusFilter !== 'all' && e.status !== statusFilter) return false
      if (dirFilter !== 'all' && e.direction !== dirFilter) return false
      if (symbolSearch && !e.symbol.toLowerCase().includes(symbolSearch.toLowerCase())) return false
      return true
    })

    rows = [...rows].sort((a, b) => {
      let diff = 0
      if (sortKey === 'confidence') diff = (b.confidence ?? 0) - (a.confidence ?? 0)
      else if (sortKey === 'symbol') diff = a.symbol.localeCompare(b.symbol)
      else if (sortKey === 'status') diff = (STATUS_ORDER[a.status] ?? 9) - (STATUS_ORDER[b.status] ?? 9)
      return sortAsc ? -diff : diff
    })

    return rows
  }, [entries, categoryFilter, patternNameFilter, statusFilter, dirFilter, symbolSearch, sortKey, sortAsc])

  const handleRefresh = () => {
    qc.invalidateQueries({ queryKey: ['patterns', 'scan', tfParam ?? 'daily'] })
  }

  const unavailable = TF_UNAVAILABLE[scanTF] ?? []

  const updatedLabel = dataUpdatedAt
    ? new Date(dataUpdatedAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    : null

  const toggleSort = (key: SortKey) => {
    if (sortKey === key) setSortAsc(v => !v)
    else { setSortKey(key); setSortAsc(false) }
  }

  // ── Loading skeleton ────────────────────────────────────────────────────────
  if (isLoading) {
    return (
      <div className="flex flex-col items-center justify-center py-24 gap-4 text-gray-500">
        <div className="relative">
          <div className="w-10 h-10 rounded-full border-2 border-gray-700 border-t-blue-500 animate-spin" />
        </div>
        <div className="text-center">
          <p className="text-sm font-medium text-gray-400">Scanning all instruments…</p>
          <p className="text-xs text-gray-600 mt-1">Running pattern detection in parallel · this takes ~10–20 s on first load</p>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-3">

      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <div className="flex items-center justify-between gap-3 flex-wrap">
        <div>
          <h2 className="text-sm font-semibold text-gray-200">Pattern Scanner</h2>
          <p className="text-[11px] text-gray-600 mt-0.5">
            {entries.length} active patterns across {new Set(entries.map(e => e.symbol)).size} instruments
            {updatedLabel && <span className="ml-2">· updated {updatedLabel}</span>}
          </p>
        </div>
        <button
          onClick={handleRefresh}
          disabled={isFetching}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border border-gray-700 bg-gray-800 text-gray-400 hover:text-gray-200 hover:border-gray-600 transition-colors disabled:opacity-40"
        >
          <span className={isFetching ? 'animate-spin' : ''}>↻</span>
          {isFetching ? 'Scanning…' : 'Refresh'}
        </button>
      </div>

      {/* ── Timeframe selector ─────────────────────────────────────────────── */}
      <div className="flex items-center gap-2 flex-wrap">
        <span className="text-[10px] text-gray-600 uppercase tracking-widest">Timeframe:</span>
        <div className="flex gap-0.5 bg-gray-900 border border-gray-800 rounded-lg p-0.5">
          {SCAN_TFS.map(tf => (
            <button
              key={tf}
              onClick={() => { setScanTF(tf); setPatternNameFilter(null) }}
              className={`px-2.5 py-1 rounded text-[11px] font-mono font-medium transition-colors ${
                scanTF === tf
                  ? tf === '1d' ? 'bg-emerald-800/60 text-emerald-300' : 'bg-blue-900/60 text-blue-300'
                  : 'text-gray-500 hover:text-gray-300'
              }`}
            >
              {TF_LABEL[tf]}
            </button>
          ))}
        </div>
        {unavailable.length > 0 && (
          <span className="text-[10px] text-gray-600">
            ⚠ {unavailable.join(', ')} skipped at {scanTF}
          </span>
        )}
        {scanTF !== '1d' && (
          <span className="text-[10px] text-amber-700 bg-amber-900/20 border border-amber-800/30 px-2 py-0.5 rounded">
            Intraday — yfinance · first load slower
          </span>
        )}
      </div>

      {/* ── Filters ────────────────────────────────────────────────────────── */}
      <div className="space-y-2">

        {/* Row 1: Category · Status · Direction · Symbol search */}
        <div className="flex flex-wrap gap-2 items-center">

          {/* Category dropdown */}
          <select
            value={categoryFilter}
            onChange={e => handleCategoryChange(e.target.value as Category)}
            className="bg-gray-900 border border-gray-800 rounded-lg px-2.5 py-1.5 text-[11px] font-medium text-gray-300 focus:outline-none focus:border-gray-600 cursor-pointer"
            style={{ color: categoryFilter === 'all' ? '#9ca3af' : CAT_COLOR[categoryFilter] }}
          >
            <option value="all">All patterns ({catCounts.all})</option>
            {(['chart', 'harmonic', 'gann', 'wyckoff'] as Category[]).map(cat => {
              const count = catCounts[cat] ?? 0
              if (count === 0) return null
              return (
                <option key={cat} value={cat}>
                  {CAT_LABEL[cat]} ({count})
                </option>
              )
            })}
          </select>

          {/* Status filter */}
          <div className="flex gap-0.5 bg-gray-900 border border-gray-800 rounded-lg p-0.5">
            {(['all', 'BREAKOUT', 'READY', 'FORMING'] as StatusFilter[]).map(s => (
              <button
                key={s}
                onClick={() => setStatusFilter(s)}
                className={`px-2 py-1 rounded text-[11px] font-medium transition-colors ${
                  statusFilter === s ? 'bg-gray-700 text-gray-200' : 'text-gray-500 hover:text-gray-300'
                }`}
              >
                {s === 'all' ? 'Any status' : s}
              </button>
            ))}
          </div>

          {/* Direction filter */}
          <div className="flex gap-0.5 bg-gray-900 border border-gray-800 rounded-lg p-0.5">
            {(['all', 'bullish', 'bearish'] as DirFilter[]).map(d => (
              <button
                key={d}
                onClick={() => setDirFilter(d)}
                className={`px-2 py-1 rounded text-[11px] font-medium transition-colors ${
                  dirFilter === d
                    ? d === 'bullish' ? 'bg-emerald-900/40 text-emerald-300'
                    : d === 'bearish' ? 'bg-red-900/40 text-red-300'
                    : 'bg-gray-700 text-gray-200'
                    : 'text-gray-500 hover:text-gray-300'
                }`}
              >
                {d === 'all' ? 'Any dir' : d === 'bullish' ? '▲ Bullish' : '▼ Bearish'}
              </button>
            ))}
          </div>

          {/* Symbol search */}
          <div className="relative flex-1 min-w-[140px] max-w-[200px]">
            <span className="absolute left-2 top-1/2 -translate-y-1/2 text-gray-600 text-[11px]">🔍</span>
            <input
              value={symbolSearch}
              onChange={e => setSymbolSearch(e.target.value)}
              placeholder="Filter symbol…"
              className="w-full pl-6 pr-3 py-1.5 bg-gray-900 border border-gray-800 rounded-lg text-[11px] text-gray-300 placeholder-gray-600 focus:outline-none focus:border-gray-600"
            />
            {symbolSearch && (
              <button onClick={() => setSymbolSearch('')} className="absolute right-2 top-1/2 -translate-y-1/2 text-gray-600 hover:text-gray-400 text-[11px]">✕</button>
            )}
          </div>

          <span className="ml-auto text-[11px] text-gray-600">{filtered.length} rows</span>
        </div>

        {/* Row 2: Pattern name chips — quick drill-in by specific pattern */}
        {patternNames.length > 1 && (
          <div className="flex flex-wrap gap-1 items-center">
            <span className="text-[10px] text-gray-600 mr-1">Pattern:</span>
            <button
              onClick={() => setPatternNameFilter(null)}
              className={`px-2 py-0.5 rounded text-[10px] border transition-colors ${
                patternNameFilter === null
                  ? 'bg-gray-700/80 text-gray-200 border-gray-600'
                  : 'text-gray-500 border-gray-800 hover:text-gray-300 hover:border-gray-700'
              }`}
            >
              All
            </button>
            {patternNames.map(([name, count]) => {
              // Infer category color from first matching entry
              const sample = entries.find(e => e.pattern_name === name)
              const color = sample ? (CAT_COLOR[classifyCategory(sample)] ?? '#9ca3af') : '#9ca3af'
              const isActive = patternNameFilter === name
              return (
                <button
                  key={name}
                  onClick={() => setPatternNameFilter(prev => prev === name ? null : name)}
                  className="px-2 py-0.5 rounded text-[10px] border transition-all"
                  style={{
                    borderColor:     isActive ? color + '70' : '#374151',
                    color:           isActive ? color : '#6b7280',
                    backgroundColor: isActive ? color + '15' : 'transparent',
                  }}
                >
                  {name}
                  <span className="ml-1 opacity-40 font-mono">{count}</span>
                </button>
              )
            })}
          </div>
        )}
      </div>

      {/* ── Table ──────────────────────────────────────────────────────────── */}
      {filtered.length === 0 ? (
        <div className="py-16 flex flex-col items-center justify-center gap-2 text-gray-600 border border-gray-800 rounded-lg">
          <span className="text-3xl">🔍</span>
          <p className="text-sm">No patterns match the current filters.</p>
          <button onClick={() => { setCategoryFilter('all'); setStatusFilter('all'); setDirFilter('all'); setSymbolSearch('') }}
            className="text-xs text-blue-500 hover:text-blue-400 mt-1">Clear filters</button>
        </div>
      ) : (
        <div className="rounded-lg border border-gray-800 overflow-hidden">
          {/* Table header */}
          <div className="grid grid-cols-[110px_1fr_130px_76px_90px_140px] gap-3 px-3 py-2 bg-gray-900/80 border-b border-gray-800 text-[10px] font-semibold uppercase tracking-wider text-gray-600">
            <button className="text-left flex items-center gap-1 hover:text-gray-400" onClick={() => toggleSort('symbol')}>
              Symbol {sortKey === 'symbol' ? (sortAsc ? '↑' : '↓') : ''}
            </button>
            <span>Pattern</span>
            <button className="text-left flex items-center gap-1 hover:text-gray-400" onClick={() => toggleSort('status')}>
              Status {sortKey === 'status' ? (sortAsc ? '↑' : '↓') : ''}
            </button>
            <span>Dir</span>
            <button className="text-left flex items-center gap-1 hover:text-gray-400" onClick={() => toggleSort('confidence')}>
              Conf {sortKey === 'confidence' ? (sortAsc ? '↑' : '↓') : ''}
            </button>
            <span>Levels</span>
          </div>

          {/* Rows */}
          <div className="divide-y divide-gray-800/60 max-h-[calc(100vh-320px)] overflow-y-auto">
            {filtered.map((entry, idx) => {
              const cat   = classifyCategory(entry)
              const color = CAT_COLOR[cat] ?? '#9ca3af'
              const conf  = Math.round(entry.confidence ?? 0)

              return (
                <button
                  key={`${entry.symbol}-${entry.pattern_name}-${idx}`}
                  onClick={() => navigate(`/chart?symbol=${entry.symbol}`)}
                  className="w-full grid grid-cols-[110px_1fr_130px_76px_90px_140px] gap-3 px-3 py-2.5 hover:bg-gray-800/30 transition-colors text-left items-center"
                >
                  {/* Symbol + category badge */}
                  <div className="flex items-center gap-1.5 min-w-0">
                    <span
                      className="flex-shrink-0 text-[9px] font-bold px-1 py-px rounded"
                      style={{ color, backgroundColor: color + '20' }}
                    >
                      {CAT_LABEL[cat]}
                    </span>
                    <span className="font-mono font-bold text-[13px] text-gray-100 truncate">{entry.symbol}</span>
                    {entry.tf && entry.tf !== 'daily' && (
                      <span className="flex-shrink-0 text-[9px] font-mono px-1 py-px rounded bg-blue-900/30 text-blue-400 border border-blue-800/30">
                        {entry.tf.toUpperCase()}
                      </span>
                    )}
                  </div>

                  {/* Pattern name + phase badge */}
                  <div className="min-w-0 flex items-center gap-1.5 flex-wrap">
                    <span className="text-[11px] text-gray-300 truncate">{entry.pattern_name}</span>
                    {entry.phase && (
                      <span className="flex-shrink-0 text-[9px] px-1 py-px rounded bg-purple-950/60 text-purple-400 border border-purple-900/30">
                        Ph.{entry.phase}
                      </span>
                    )}
                    {entry.events && entry.events.length > 0 && (
                      <span className="text-[9px] text-gray-600" title={entry.events.join(', ')}>
                        {entry.events.slice(0, 2).join(' · ')}
                      </span>
                    )}
                  </div>

                  {/* Status */}
                  <span className={`flex-shrink-0 text-[10px] px-2 py-0.5 rounded w-fit font-medium ${STATUS_STYLE[entry.status] ?? 'text-gray-500'}`}>
                    {entry.status}
                  </span>

                  {/* Direction */}
                  <span className={`flex-shrink-0 text-[11px] font-bold ${
                    entry.direction === 'bullish' ? 'text-emerald-400'
                    : entry.direction === 'bearish' ? 'text-red-400'
                    : 'text-gray-600'
                  }`}>
                    {entry.direction === 'bullish' ? '▲' : entry.direction === 'bearish' ? '▼' : '◆'}
                    <span className="ml-0.5 text-[9px] font-normal opacity-60">
                      {entry.direction === 'neutral' ? 'N' : ''}
                    </span>
                  </span>

                  {/* Confidence bar */}
                  <div className="flex items-center gap-1.5">
                    <div className="flex-1 h-1 rounded-full bg-gray-800 overflow-hidden">
                      <div
                        className="h-full rounded-full transition-all"
                        style={{ width: `${conf}%`, backgroundColor: color + 'b0' }}
                      />
                    </div>
                    <span className="flex-shrink-0 font-mono text-[10px] font-bold w-7 text-right" style={{ color }}>
                      {conf}%
                    </span>
                  </div>

                  {/* Breakout → Target */}
                  <div className="text-[10px] font-mono min-w-0 truncate">
                    {entry.breakout_level != null ? (
                      <span>
                        <span className="text-gray-400">${entry.breakout_level.toFixed(2)}</span>
                        {entry.target != null && (
                          <span className="text-emerald-600"> → ${entry.target.toFixed(2)}</span>
                        )}
                      </span>
                    ) : (
                      <span className="text-gray-700">—</span>
                    )}
                  </div>
                </button>
              )
            })}
          </div>
        </div>
      )}

      {/* Footer note */}
      <p className="text-[10px] text-gray-700 pt-1">
        Backend caches scan results for 5 minutes · click Refresh to force re-scan · click any row to open chart
      </p>
    </div>
  )
}
