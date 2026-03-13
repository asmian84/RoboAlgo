import { useEffect, useRef, useState, useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import { useInstruments } from '../../api/hooks'
import type { Instrument } from '../../types'

// ── Color maps ────────────────────────────────────────────────────────────────

const TYPE_COLORS: Record<string, string> = {
  leveraged_etf_bull: '#22c55e',
  leveraged_etf_bear: '#ef4444',
  stock:              '#60a5fa',
  index:              '#f97316',
  commodity:          '#eab308',
}

// ── Result row ────────────────────────────────────────────────────────────────

function ResultRow({
  inst,
  selected,
  onClick,
}: {
  inst:     Instrument
  selected: boolean
  onClick:  () => void
}) {
  const typeColor = TYPE_COLORS[inst.instrument_type ?? ''] ?? '#6b7280'
  return (
    <button
      className={`w-full flex items-center gap-3 px-4 py-3 text-left transition-colors ${
        selected ? 'bg-emerald-900/30 border-l-2 border-emerald-500' : 'hover:bg-gray-800/60 border-l-2 border-transparent'
      }`}
      onClick={onClick}
    >
      <div className="flex flex-col w-16 flex-shrink-0">
        <span className="text-sm font-mono font-bold text-gray-100">{inst.symbol}</span>
        {inst.leverage_factor && inst.leverage_factor !== 1 && (
          <span className="text-[10px] font-mono text-amber-400">{inst.leverage_factor}×</span>
        )}
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-xs text-gray-300 truncate">{inst.name ?? '—'}</p>
        {inst.underlying && (
          <p className="text-[10px] text-gray-600 truncate">↳ {inst.underlying}</p>
        )}
      </div>
      {inst.instrument_type && (
        <span
          className="text-[10px] px-1.5 py-0.5 rounded flex-shrink-0"
          style={{ color: typeColor, backgroundColor: typeColor + '20' }}
        >
          {inst.instrument_type.replace(/_/g, ' ')}
        </span>
      )}
    </button>
  )
}

// ── Palette ───────────────────────────────────────────────────────────────────

interface Props {
  isOpen:    boolean
  onClose:   () => void
}

export default function SymbolCommandPalette({ isOpen, onClose }: Props) {
  const navigate  = useNavigate()
  const inputRef  = useRef<HTMLInputElement>(null)
  const listRef   = useRef<HTMLDivElement>(null)

  const { data: instruments = [] } = useInstruments()
  const [query,    setQuery]   = useState('')
  const [selected, setSelected] = useState(0)

  // Focus on open
  useEffect(() => {
    if (isOpen) {
      setQuery('')
      setSelected(0)
      setTimeout(() => inputRef.current?.focus(), 50)
    }
  }, [isOpen])

  // Keyboard: Escape to close, ↑↓ to navigate, Enter to select
  useEffect(() => {
    if (!isOpen) return
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape')   { e.preventDefault(); onClose() }
      if (e.key === 'ArrowDown')  { e.preventDefault(); setSelected(s => Math.min(s + 1, results.length - 1)) }
      if (e.key === 'ArrowUp')    { e.preventDefault(); setSelected(s => Math.max(s - 1, 0)) }
      if (e.key === 'Enter')      { e.preventDefault(); if (results[selected]) openSymbol(results[selected]) }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  })  // intentional: re-register each render so `results` and `selected` are fresh

  const results: Instrument[] = useMemo(() => {
    if (!query.trim()) return instruments.slice(0, 12)
    const q = query.toLowerCase()
    return instruments
      .filter(i =>
        i.symbol.toLowerCase().includes(q) ||
        (i.name ?? '').toLowerCase().includes(q) ||
        (i.underlying ?? '').toLowerCase().includes(q)
      )
      .slice(0, 12)
  }, [query, instruments])

  function openSymbol(inst: Instrument) {
    navigate(`/chart?symbol=${inst.symbol}`)
    onClose()
    setQuery('')
  }

  if (!isOpen) return null

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/60 z-[100] backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="fixed top-[15vh] left-1/2 -translate-x-1/2 z-[101] w-full max-w-xl">
        <div className="bg-gray-900 border border-gray-700 rounded-xl shadow-2xl overflow-hidden">
          {/* Search input */}
          <div className="flex items-center gap-3 px-4 py-3 border-b border-gray-800">
            <span className="text-gray-600">⌕</span>
            <input
              ref={inputRef}
              value={query}
              onChange={e => { setQuery(e.target.value.toUpperCase()); setSelected(0) }}
              placeholder="Search symbol or name…"
              className="flex-1 bg-transparent text-sm font-mono text-gray-200 placeholder-gray-600 focus:outline-none"
            />
            {query && (
              <button onClick={() => setQuery('')} className="text-gray-600 hover:text-gray-400 text-xs">✕</button>
            )}
            <kbd className="text-[10px] text-gray-600 bg-gray-800 px-1.5 py-0.5 rounded font-mono">ESC</kbd>
          </div>

          {/* Results */}
          <div ref={listRef} className="max-h-80 overflow-y-auto">
            {results.length === 0 ? (
              <div className="px-4 py-8 text-center text-xs text-gray-600">
                No instruments found for "{query}"
              </div>
            ) : (
              results.map((inst, i) => (
                <ResultRow
                  key={inst.symbol}
                  inst={inst}
                  selected={i === selected}
                  onClick={() => openSymbol(inst)}
                />
              ))
            )}
          </div>

          {/* Footer */}
          <div className="px-4 py-2 border-t border-gray-800 bg-gray-900/50 flex items-center gap-4 text-[10px] text-gray-600">
            <span><kbd className="bg-gray-800 px-1 rounded">↑↓</kbd> navigate</span>
            <span><kbd className="bg-gray-800 px-1 rounded">Enter</kbd> open chart</span>
            <span><kbd className="bg-gray-800 px-1 rounded">/</kbd> to open</span>
          </div>
        </div>
      </div>
    </>
  )
}
