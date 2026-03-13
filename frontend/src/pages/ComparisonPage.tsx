import { useState } from 'react'
import { useInstruments, useChartData } from '../api/hooks'

function MiniChart({ symbol }: { symbol: string }) {
  const { data: chartData, isLoading } = useChartData(symbol, 60)

  if (isLoading) return <div className="h-32 bg-gray-900 rounded animate-pulse" />

  const prices = chartData?.prices || []
  if (prices.length < 2) return <div className="h-32 bg-gray-900 rounded flex items-center justify-center text-gray-600 text-xs">No data</div>

  const closes = prices.map(p => p.close!).filter(Boolean)
  const min = Math.min(...closes)
  const max = Math.max(...closes)
  const range = max - min || 1
  const w = 280
  const h = 100

  const points = closes.map((c, i) => {
    const x = (i / (closes.length - 1)) * w
    const y = h - ((c - min) / range) * h
    return `${x},${y}`
  }).join(' ')

  const lastClose = closes[closes.length - 1]
  const firstClose = closes[0]
  const change = ((lastClose - firstClose) / firstClose) * 100
  const color = change >= 0 ? '#22c55e' : '#ef4444'

  return (
    <div className="bg-gray-900 rounded-lg p-3 border border-gray-800">
      <div className="flex justify-between items-center mb-2">
        <span className="font-bold text-white">{symbol}</span>
        <span className="text-sm font-mono" style={{ color }}>
          ${lastClose.toFixed(2)} ({change >= 0 ? '+' : ''}{change.toFixed(1)}%)
        </span>
      </div>
      <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-24">
        <polyline fill="none" stroke={color} strokeWidth="2" points={points} />
      </svg>
    </div>
  )
}

export default function ComparisonPage() {
  const { data: instruments } = useInstruments()
  const [selected, setSelected] = useState<string[]>(['TQQQ', 'SQQQ'])

  const toggleSymbol = (sym: string) => {
    setSelected(prev =>
      prev.includes(sym)
        ? prev.filter(s => s !== sym)
        : prev.length < 4 ? [...prev, sym] : prev
    )
  }

  return (
    <div>
      <h2 className="text-2xl font-bold mb-2">Instrument Comparison</h2>
      <p className="text-sm text-gray-500 mb-4">Select up to 4 instruments to compare side by side.</p>

      <div className="flex flex-wrap gap-1.5 mb-6">
        {(instruments || []).map(i => (
          <button
            key={i.symbol}
            onClick={() => toggleSymbol(i.symbol)}
            className={`px-2 py-1 rounded text-xs transition-colors ${
              selected.includes(i.symbol)
                ? 'bg-emerald-600 text-white'
                : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
            }`}
          >
            {i.symbol}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {selected.map(sym => (
          <MiniChart key={sym} symbol={sym} />
        ))}
      </div>

      {selected.length === 0 && (
        <p className="text-gray-500 text-center py-10">Select instruments to compare.</p>
      )}
    </div>
  )
}
