import { useState, useEffect, useRef, useMemo } from 'react'
import { useSearchParams } from 'react-router-dom'
import KLineChartOverlay from '../components/chart/KLineChartOverlay'
import { useInstruments } from '../api/hooks'

// ── ChartPage ─────────────────────────────────────────────────────────────────
// Clean chart page: symbol search bar -> full-screen KLineChartOverlay.
// Supports URL query param: /chart?symbol=AAPL

export default function ChartPage() {
  const [searchParams, setSearchParams] = useSearchParams()
  // URL is the single source of truth for the selected symbol. This avoids
  // the two-way state/URL sync races that previously broke the Back button
  // and the ticker-bar click (where the old URL would re-populate state).
  const symbol = searchParams.get('symbol')?.toUpperCase() || null
  const [input, setInput] = useState(symbol || '')
  const [showSuggestions, setShowSuggestions] = useState(false)
  const [highlightIdx, setHighlightIdx] = useState(-1)
  const inputRef = useRef<HTMLInputElement>(null)
  const suggestionsRef = useRef<HTMLDivElement>(null)

  // Instrument list for autocomplete
  const { data: instruments } = useInstruments()

  // Keep the input textbox in sync with whatever symbol the URL points at
  // (e.g. ticker-bar navigation arrives at /chart?symbol=XYZ).
  useEffect(() => {
    if (symbol && symbol !== input.toUpperCase()) setInput(symbol)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [symbol])

  // Filtered suggestions based on input
  const suggestions = useMemo(() => {
    if (!input.trim() || !instruments?.length) return []
    const q = input.toUpperCase()
    return instruments
      .filter(
        (i: any) =>
          i.symbol.toUpperCase().startsWith(q) ||
          (i.name && i.name.toUpperCase().includes(q))
      )
      .slice(0, 12)
  }, [input, instruments])

  // Close suggestions on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (
        suggestionsRef.current &&
        !suggestionsRef.current.contains(e.target as Node) &&
        inputRef.current &&
        !inputRef.current.contains(e.target as Node)
      ) {
        setShowSuggestions(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  // Focus input on mount when no symbol
  useEffect(() => {
    if (!symbol) inputRef.current?.focus()
  }, [symbol])

  const loadSymbol = (sym: string) => {
    const s = sym.trim().toUpperCase()
    if (!s) return
    setInput(s)
    setShowSuggestions(false)
    setSearchParams({ symbol: s }, { replace: true })
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      if (highlightIdx >= 0 && highlightIdx < suggestions.length) {
        loadSymbol(suggestions[highlightIdx].symbol)
      } else {
        loadSymbol(input)
      }
      setHighlightIdx(-1)
    } else if (e.key === 'ArrowDown') {
      e.preventDefault()
      setHighlightIdx((prev) => Math.min(prev + 1, suggestions.length - 1))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setHighlightIdx((prev) => Math.max(prev - 1, -1))
    } else if (e.key === 'Escape') {
      setShowSuggestions(false)
      setHighlightIdx(-1)
    }
  }

  // ── When a symbol is loaded, render the chart overlay ──────────────
  if (symbol) {
    return (
      <KLineChartOverlay
        symbol={symbol}
        onClose={() => {
          // Return to the chart search screen so the user can enter a new symbol.
          setInput('')
          setSearchParams({}, { replace: true })
        }}
      />
    )
  }

  // ── No symbol selected: show centered search UI ─────────────────────
  return (
    <div className="h-full flex flex-col">
      {/* Centered search */}
      <div className="flex-1 flex items-center justify-center">
        <div className="flex flex-col items-center gap-6 w-full max-w-md px-4">
          {/* Title */}
          <div className="text-center">
            <h1 className="text-2xl font-bold text-white font-mono mb-1">
              Chart
            </h1>
            <p className="text-sm text-gray-500">
              Enter a ticker to load the chart
            </p>
          </div>

          {/* Input + Go */}
          <div className="relative w-full">
            <div className="flex items-center gap-2">
              <div className="relative flex-1">
                <input
                  ref={inputRef}
                  type="text"
                  value={input}
                  onChange={(e) => {
                    setInput(e.target.value.toUpperCase())
                    setShowSuggestions(true)
                    setHighlightIdx(-1)
                  }}
                  onFocus={() => setShowSuggestions(true)}
                  onKeyDown={handleKeyDown}
                  placeholder="AAPL, SPY, TSLA ..."
                  className="w-full px-4 py-3 rounded-lg bg-gray-800 border border-gray-700 text-white font-mono text-lg placeholder-gray-600 focus:outline-none focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500/30 transition-colors"
                  autoComplete="off"
                  spellCheck={false}
                />
                {/* Autocomplete dropdown */}
                {showSuggestions && suggestions.length > 0 && (
                  <div
                    ref={suggestionsRef}
                    className="absolute left-0 right-0 top-full mt-1 bg-gray-800 border border-gray-700 rounded-lg overflow-hidden shadow-xl shadow-black/40 z-20 max-h-72 overflow-y-auto"
                  >
                    {suggestions.map((inst: any, idx: number) => (
                      <button
                        key={inst.symbol}
                        onClick={() => loadSymbol(inst.symbol)}
                        onMouseEnter={() => setHighlightIdx(idx)}
                        className={`w-full flex items-center gap-3 px-4 py-2.5 text-left transition-colors ${
                          idx === highlightIdx
                            ? 'bg-emerald-600/20 text-white'
                            : 'text-gray-300 hover:bg-gray-700/60'
                        }`}
                      >
                        <span className="font-mono font-bold text-sm w-16 flex-shrink-0 text-emerald-400">
                          {inst.symbol}
                        </span>
                        <span className="text-xs text-gray-500 truncate">
                          {inst.name || ''}
                        </span>
                        {inst.instrument_type && (
                          <span className="ml-auto text-[10px] text-gray-600 uppercase tracking-wider flex-shrink-0">
                            {inst.instrument_type}
                          </span>
                        )}
                      </button>
                    ))}
                  </div>
                )}
              </div>
              <button
                onClick={() => loadSymbol(input)}
                disabled={!input.trim()}
                className="px-5 py-3 rounded-lg bg-emerald-600 hover:bg-emerald-500 disabled:bg-gray-700 disabled:text-gray-500 text-white font-bold text-sm transition-colors"
              >
                Go
              </button>
            </div>
          </div>

          {/* Keyboard hint */}
          <p className="text-[11px] text-gray-600 font-mono">
            Press Enter to load chart · Arrow keys to navigate suggestions
          </p>
        </div>
      </div>
    </div>
  )
}
