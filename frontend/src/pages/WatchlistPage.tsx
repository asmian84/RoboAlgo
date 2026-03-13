import { useState, useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import { useInstruments } from '../api/hooks'
import type { Instrument } from '../types'

// ── Types ─────────────────────────────────────────────────────────────────────

interface WatchlistItem {
  symbol:  string
  note?:   string
  addedAt: string
}

// ── Instrument type color ─────────────────────────────────────────────────────

const TYPE_COLORS: Record<string, string> = {
  leveraged_etf_bull: '#22c55e',
  leveraged_etf_bear: '#ef4444',
  stock:              '#60a5fa',
  index:              '#f97316',
  commodity:          '#eab308',
}

// ── Instrument search bar ─────────────────────────────────────────────────────

function SymbolSearchBar({
  instruments,
  watchlist,
  onAdd,
}: {
  instruments: Instrument[]
  watchlist:   WatchlistItem[]
  onAdd:       (symbol: string) => void
}) {
  const [query, setQuery]   = useState('')
  const [focused, setFocused] = useState(false)

  const results = useMemo(() => {
    if (!query.trim()) return []
    const q = query.toLowerCase()
    return instruments
      .filter(i =>
        (i.symbol.toLowerCase().includes(q) || (i.name ?? '').toLowerCase().includes(q)) &&
        !watchlist.some(w => w.symbol === i.symbol)
      )
      .slice(0, 8)
  }, [query, instruments, watchlist])

  return (
    <div className="relative">
      <input
        value={query}
        onChange={e => setQuery(e.target.value.toUpperCase())}
        onFocus={() => setFocused(true)}
        onBlur={() => setTimeout(() => setFocused(false), 150)}
        placeholder="Search symbol or name…"
        className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm font-mono text-gray-200 placeholder-gray-600 focus:outline-none focus:border-emerald-600"
      />
      {focused && results.length > 0 && (
        <div className="absolute top-full left-0 right-0 mt-1 bg-gray-800 border border-gray-700 rounded-lg overflow-hidden shadow-xl z-50">
          {results.map(inst => {
            const typeColor = TYPE_COLORS[inst.instrument_type ?? ''] ?? '#6b7280'
            return (
              <button
                key={inst.symbol}
                className="w-full flex items-center gap-3 px-3 py-2 hover:bg-gray-700 transition-colors text-left"
                onClick={() => {
                  onAdd(inst.symbol)
                  setQuery('')
                }}
              >
                <span className="font-mono font-bold text-gray-100 w-16">{inst.symbol}</span>
                <span className="text-xs text-gray-400 flex-1 truncate">{inst.name ?? '—'}</span>
                {inst.instrument_type && (
                  <span className="text-[10px] px-1.5 py-0.5 rounded font-medium" style={{ color: typeColor, backgroundColor: typeColor + '20' }}>
                    {inst.instrument_type.replace(/_/g, ' ')}
                  </span>
                )}
                {inst.leverage_factor && inst.leverage_factor !== 1 && (
                  <span className="text-[10px] font-mono text-gray-500">{inst.leverage_factor}×</span>
                )}
              </button>
            )
          })}
        </div>
      )}
    </div>
  )
}

// ── Watchlist table ───────────────────────────────────────────────────────────

function WatchlistTable({
  items,
  instruments,
  onRemove,
  onNote,
}: {
  items:       WatchlistItem[]
  instruments: Instrument[]
  onRemove:    (symbol: string) => void
  onNote:      (symbol: string, note: string) => void
}) {
  const navigate = useNavigate()
  const instMap  = useMemo(() =>
    new Map(instruments.map(i => [i.symbol, i])),
    [instruments]
  )

  if (items.length === 0) {
    return (
      <div className="bg-gray-900 border border-gray-800 rounded-xl">
        <div className="flex flex-col items-center justify-center h-40 gap-3">
          <span className="text-3xl">👁</span>
          <p className="text-xs text-gray-600 text-center">
            Your watchlist is empty.<br />Search for a symbol above to add it.
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
      <div className="px-4 py-3 border-b border-gray-800">
        <h3 className="text-sm font-semibold text-gray-100">My Watchlist ({items.length})</h3>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-gray-800 text-gray-500 uppercase">
              <th className="text-left px-4 py-2 font-medium">Symbol</th>
              <th className="text-left px-4 py-2 font-medium">Name</th>
              <th className="text-left px-4 py-2 font-medium">Type</th>
              <th className="text-left px-4 py-2 font-medium">Note</th>
              <th className="text-right px-4 py-2 font-medium">Added</th>
              <th className="px-4 py-2" />
            </tr>
          </thead>
          <tbody>
            {items.map(item => {
              const inst      = instMap.get(item.symbol)
              const typeColor = TYPE_COLORS[inst?.instrument_type ?? ''] ?? '#6b7280'

              return (
                <tr key={item.symbol} className="border-b border-gray-800/40 hover:bg-gray-800/20 group">
                  <td className="px-4 py-2">
                    <button
                      onClick={() => navigate(`/chart?symbol=${item.symbol}`)}
                      className="font-mono font-bold text-gray-100 hover:text-emerald-400 transition-colors"
                    >
                      {item.symbol}
                    </button>
                  </td>
                  <td className="px-4 py-2 text-gray-400 truncate max-w-[200px]">
                    {inst?.name ?? '—'}
                  </td>
                  <td className="px-4 py-2">
                    {inst?.instrument_type && (
                      <span
                        className="px-1.5 py-0.5 rounded text-[10px] font-medium"
                        style={{ color: typeColor, backgroundColor: typeColor + '20' }}
                      >
                        {inst.instrument_type.replace(/_/g, ' ')}
                      </span>
                    )}
                  </td>
                  <td className="px-4 py-2">
                    <input
                      value={item.note ?? ''}
                      onChange={e => onNote(item.symbol, e.target.value)}
                      placeholder="Add note…"
                      className="w-full bg-transparent border-b border-gray-700 text-xs text-gray-400 focus:outline-none focus:border-emerald-600 placeholder-gray-700"
                    />
                  </td>
                  <td className="px-4 py-2 text-right text-gray-600">
                    {new Date(item.addedAt).toLocaleDateString([], { month: 'short', day: 'numeric' })}
                  </td>
                  <td className="px-4 py-2 text-right">
                    <button
                      onClick={() => onRemove(item.symbol)}
                      className="opacity-0 group-hover:opacity-100 text-gray-600 hover:text-red-400 transition-all text-xs"
                    >
                      ✕
                    </button>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}

// ── All Instruments Grid ──────────────────────────────────────────────────────

function AllInstrumentsGrid({
  instruments,
  watchlist,
  onAdd,
}: {
  instruments: Instrument[]
  watchlist:   WatchlistItem[]
  onAdd:       (symbol: string) => void
}) {
  const navigate = useNavigate()
  const [filter, setFilter] = useState<string>('all')

  const types = useMemo(() => {
    const set = new Set(instruments.map(i => i.instrument_type ?? 'unknown'))
    return ['all', ...Array.from(set)]
  }, [instruments])

  const filtered = useMemo(() => {
    if (filter === 'all') return instruments
    return instruments.filter(i => (i.instrument_type ?? 'unknown') === filter)
  }, [instruments, filter])

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800 flex-wrap gap-2">
        <h3 className="text-sm font-semibold text-gray-100">
          All Instruments ({instruments.length})
        </h3>
        {/* Type filter */}
        <div className="flex items-center gap-1 flex-wrap">
          {types.map(t => (
            <button
              key={t}
              onClick={() => setFilter(t)}
              className={`px-2 py-0.5 rounded text-[10px] font-medium transition-colors ${
                filter === t ? 'bg-emerald-700 text-white' : 'bg-gray-800 text-gray-400 hover:text-gray-200'
              }`}
            >
              {t === 'all' ? 'All' : t.replace(/_/g, ' ')}
            </button>
          ))}
        </div>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-gray-800 text-gray-500 uppercase">
              <th className="text-left px-4 py-2 font-medium">Symbol</th>
              <th className="text-left px-4 py-2 font-medium">Name</th>
              <th className="text-left px-4 py-2 font-medium">Type</th>
              <th className="text-right px-4 py-2 font-medium">Leverage</th>
              <th className="px-4 py-2" />
            </tr>
          </thead>
          <tbody>
            {filtered.map(inst => {
              const inWatchlist = watchlist.some(w => w.symbol === inst.symbol)
              const typeColor   = TYPE_COLORS[inst.instrument_type ?? ''] ?? '#6b7280'
              return (
                <tr key={inst.symbol} className="border-b border-gray-800/40 hover:bg-gray-800/20 group">
                  <td className="px-4 py-2">
                    <button
                      onClick={() => navigate(`/chart?symbol=${inst.symbol}`)}
                      className="font-mono font-bold text-gray-100 hover:text-emerald-400 transition-colors"
                    >
                      {inst.symbol}
                    </button>
                  </td>
                  <td className="px-4 py-2 text-gray-400 truncate max-w-[250px]">{inst.name ?? '—'}</td>
                  <td className="px-4 py-2">
                    {inst.instrument_type && (
                      <span
                        className="px-1.5 py-0.5 rounded text-[10px] font-medium"
                        style={{ color: typeColor, backgroundColor: typeColor + '20' }}
                      >
                        {inst.instrument_type.replace(/_/g, ' ')}
                      </span>
                    )}
                  </td>
                  <td className="px-4 py-2 text-right font-mono text-gray-400">
                    {inst.leverage_factor && inst.leverage_factor !== 1 ? `${inst.leverage_factor}×` : '—'}
                  </td>
                  <td className="px-4 py-2 text-right">
                    <button
                      onClick={() => !inWatchlist && onAdd(inst.symbol)}
                      disabled={inWatchlist}
                      className={`text-[10px] px-2 py-0.5 rounded transition-colors ${
                        inWatchlist
                          ? 'text-emerald-500 bg-emerald-900/20 cursor-default'
                          : 'text-gray-600 hover:text-emerald-400 hover:bg-emerald-900/20'
                      }`}
                    >
                      {inWatchlist ? '✓ Watching' : '+ Watch'}
                    </button>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}

// ── Page ─────────────────────────────────────────────────────────────────────

export default function WatchlistPage() {
  const { data: instruments = [] } = useInstruments()
  const [watchlist, setWatchlist]  = useState<WatchlistItem[]>([])

  function addSymbol(symbol: string) {
    if (watchlist.some(w => w.symbol === symbol)) return
    setWatchlist(prev => [...prev, { symbol, addedAt: new Date().toISOString() }])
  }

  function removeSymbol(symbol: string) {
    setWatchlist(prev => prev.filter(w => w.symbol !== symbol))
  }

  function updateNote(symbol: string, note: string) {
    setWatchlist(prev => prev.map(w => w.symbol === symbol ? { ...w, note } : w))
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div>
        <h1 className="text-xl font-bold text-gray-100 flex items-center gap-2">
          👁 Watchlist Manager
        </h1>
        <p className="text-xs text-gray-500 mt-0.5">
          Track instruments of interest · {instruments.length} instruments available
        </p>
      </div>

      {/* Search */}
      <SymbolSearchBar instruments={instruments} watchlist={watchlist} onAdd={addSymbol} />

      {/* My Watchlist */}
      <WatchlistTable
        items={watchlist}
        instruments={instruments}
        onRemove={removeSymbol}
        onNote={updateNote}
      />

      {/* All instruments */}
      <AllInstrumentsGrid instruments={instruments} watchlist={watchlist} onAdd={addSymbol} />
    </div>
  )
}
