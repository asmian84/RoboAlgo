/**
 * MultiChartGrid — compare 1, 2, or 4 symbols side-by-side.
 *
 * Hotkeys (when no input focused):
 *   1 → single chart
 *   2 → dual  chart
 *   4 → quad  chart
 */
import { useEffect, useRef, useState } from 'react'
import { createChart, ColorType, type IChartApi } from 'lightweight-charts'
import { useChartData, useInstruments } from '../../api/hooks'

// ── Single mini chart cell ────────────────────────────────────────────────────

function MiniChart({ symbol }: { symbol: string }) {
  const containerRef = useRef<HTMLDivElement>(null)
  const chartApi     = useRef<IChartApi | null>(null)
  const { data, isLoading } = useChartData(symbol, 250)

  useEffect(() => {
    if (!containerRef.current || !data?.prices.length) return
    try { chartApi.current?.remove() } catch { /* disposed */ }

    const chart = createChart(containerRef.current, {
      autoSize: true,
      layout: {
        background: { type: ColorType.Solid as const, color: '#0a0a0a' },
        textColor: '#6b7280',
      },
      grid: { vertLines: { color: '#111827' }, horzLines: { color: '#111827' } },
      crosshair: { mode: 0 as const },
      timeScale: { borderColor: '#1f2937', timeVisible: false },
      rightPriceScale: { borderColor: '#1f2937', scaleMargins: { top: 0.1, bottom: 0.1 } },
      handleScroll: false,
      handleScale: false,
    })
    chartApi.current = chart

    const prices = data.prices.filter(p => p.open && p.high && p.low && p.close)
    const candle = chart.addCandlestickSeries({
      upColor:   '#22c55e', downColor:   '#ef4444',
      borderUpColor: '#22c55e', borderDownColor: '#ef4444',
      wickUpColor:   '#22c55e', wickDownColor:   '#ef4444',
    })
    candle.setData(prices.map(p => ({
      time:  p.date as import('lightweight-charts').Time,
      open:  p.open!,
      high:  p.high!,
      low:   p.low!,
      close: p.close!,
    })))

    // MA50
    const inds = data.indicators.filter(i => i.ma50 != null)
    if (inds.length > 0) {
      const ma50 = chart.addLineSeries({ color: '#f97316', lineWidth: 1, priceLineVisible: false })
      ma50.setData(inds.map(i => ({ time: i.date as import('lightweight-charts').Time, value: i.ma50! })))
    }

    chart.timeScale().fitContent()

    return () => { try { chart.remove() } catch { /* disposed */ } }
  }, [data])

  // Cleanup on unmount
  useEffect(() => () => { try { chartApi.current?.remove() } catch { /* disposed */ } }, [])

  if (isLoading) {
    return (
      <div className="h-full flex items-center justify-center text-gray-600 text-xs">
        Loading {symbol}…
      </div>
    )
  }
  if (!data?.prices.length) {
    return (
      <div className="h-full flex items-center justify-center text-gray-600 text-xs">
        No data for {symbol}
      </div>
    )
  }

  return <div ref={containerRef} className="h-full" />
}

// ── Symbol selector ────────────────────────────────────────────────────────────

function SymbolInput({
  value,
  onChange,
  idx,
}: {
  value: string
  onChange: (s: string) => void
  idx: number
}) {
  const [draft, setDraft] = useState(value)
  const { data: instruments } = useInstruments()

  const commit = () => {
    const s = draft.trim().toUpperCase()
    if (s) onChange(s)
    else setDraft(value)
  }

  return (
    <div className="flex items-center gap-2 px-3 py-2 border-b border-gray-800 bg-gray-900/60">
      <span className="text-[10px] text-gray-600 font-mono">{idx + 1}</span>
      <input
        value={draft}
        onChange={e => setDraft(e.target.value.toUpperCase())}
        onBlur={commit}
        onKeyDown={e => { if (e.key === 'Enter') { commit(); (e.target as HTMLInputElement).blur() } }}
        className="flex-1 bg-transparent text-xs font-mono font-bold text-gray-100 outline-none"
        placeholder="SYMBOL"
        list={`mc-suggestions-${idx}`}
      />
      {instruments && (
        <datalist id={`mc-suggestions-${idx}`}>
          {instruments.map(i => <option key={i.symbol} value={i.symbol}>{i.name ?? ''}</option>)}
        </datalist>
      )}
    </div>
  )
}

// ── Layout selector ────────────────────────────────────────────────────────────

type Layout = 1 | 2 | 4

const LAYOUT_ICONS: Record<Layout, string> = {
  1: '□',
  2: '⊟',
  4: '⊞',
}

// ── Main component ─────────────────────────────────────────────────────────────

const DEFAULT_SYMBOLS: Record<Layout, string[]> = {
  1: ['TQQQ'],
  2: ['TQQQ', 'SOXL'],
  4: ['TQQQ', 'SOXL', 'UPRO', 'LABU'],
}

export default function MultiChartGrid({ className = '' }: { className?: string }) {
  const [layout, setLayout] = useState<Layout>(2)
  const [symbols, setSymbols] = useState<Record<Layout, string[]>>(DEFAULT_SYMBOLS)

  const active = symbols[layout]

  const updateSymbol = (idx: number, sym: string) => {
    setSymbols(prev => ({
      ...prev,
      [layout]: prev[layout].map((s, i) => i === idx ? sym : s),
    }))
  }

  // Global hotkeys 1/2/4
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement
      if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA') return
      if (e.key === '1') setLayout(1)
      if (e.key === '2') setLayout(2)
      if (e.key === '4') setLayout(4)
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [])

  // CSS grid template based on layout
  const gridStyle: React.CSSProperties =
    layout === 1 ? { display: 'grid', gridTemplateColumns: '1fr' } :
    layout === 2 ? { display: 'grid', gridTemplateColumns: '1fr 1fr' } :
                   { display: 'grid', gridTemplateColumns: '1fr 1fr', gridTemplateRows: '1fr 1fr' }

  return (
    <div className={`bg-gray-900 border border-gray-800 rounded-xl flex flex-col overflow-hidden ${className}`}>
      {/* Toolbar */}
      <div className="flex items-center gap-3 px-4 py-2.5 border-b border-gray-800 flex-shrink-0">
        <h2 className="text-sm font-semibold text-gray-100">Multi-Chart Grid</h2>
        <div className="ml-auto flex items-center gap-1">
          {([1, 2, 4] as Layout[]).map(l => (
            <button
              key={l}
              onClick={() => setLayout(l)}
              title={`${l} chart${l > 1 ? 's' : ''} (hotkey: ${l})`}
              className={`w-7 h-7 rounded flex items-center justify-center text-sm transition-colors border ${
                layout === l
                  ? 'bg-blue-600 text-white border-blue-500'
                  : 'bg-gray-800 text-gray-400 border-gray-700 hover:border-gray-500'
              }`}
            >
              {LAYOUT_ICONS[l]}
            </button>
          ))}
        </div>
        <span className="text-[10px] text-gray-600">Press 1/2/4 to switch layout</span>
      </div>

      {/* Chart grid */}
      <div className="flex-1 min-h-0" style={gridStyle}>
        {active.map((sym, idx) => (
          <div key={idx} className="flex flex-col border border-gray-800/50 min-h-0 overflow-hidden">
            <SymbolInput value={sym} onChange={s => updateSymbol(idx, s)} idx={idx} />
            <div className="flex-1 min-h-0">
              <MiniChart symbol={sym} />
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
