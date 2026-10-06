// PatternChartGrid — Fast SVG chart grid with time/price axes + pattern overlays
// Uses the lightweight minichart endpoint (DB-first, instant) instead of heavy klinecharts
import { useEffect, useState, memo } from 'react'
import api from '../../api/client'

// ── Types ──────────────────────────────────────────────────────────────────

interface Candle {
  o: number; h: number; l: number; c: number
  k: number; d: number
  m20?: number; m50?: number
  t?: string; ts?: number
}

interface PatternInfo {
  pattern_name: string
  direction: string
  status: string
  confidence: number
  breakout_level: number | null
  target: number | null
  invalidation_level: number | null
}

// ── Hooks ──────────────────────────────────────────────────────────────────

function useChartData(symbol: string) {
  const [candles, setCandles] = useState<Candle[] | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(false)

  useEffect(() => {
    setLoading(true)
    setError(false)
    const baseUrl = (api.defaults.baseURL || '/api').replace(/\/$/, '')
    fetch(`${baseUrl}/trade-signals/minichart/${symbol}?tf=1d&bars=60`)
      .then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      })
      .then(d => { setCandles(d.candles || []); setLoading(false) })
      .catch(() => { setError(true); setLoading(false) })
  }, [symbol])

  return { candles, loading, error }
}

function usePatterns(symbol: string) {
  const [patterns, setPatterns] = useState<PatternInfo[]>([])

  useEffect(() => {
    const baseUrl = (api.defaults.baseURL || '/api').replace(/\/$/, '')
    fetch(`${baseUrl}/patterns/${symbol}`)
      .then(r => r.json())
      .then(data => {
        const active = (data || []).filter((p: any) =>
          p.status === 'FORMING' || p.status === 'READY' || p.status === 'BREAKOUT' || p.status === 'COMPLETED'
        )
        setPatterns(active.slice(0, 3)) // top 3 patterns
      })
      .catch(() => {}) // non-fatal
  }, [symbol])

  return patterns
}

// ── Chart Grid Item ────────────────────────────────────────────────────────

const PatternChartGridItem = memo(function PatternChartGridItem({
  symbol,
  onOpenChart,
}: {
  symbol: string
  onOpenChart: () => void
}) {
  const { candles, loading, error } = useChartData(symbol)
  const patterns = usePatterns(symbol)

  if (loading) {
    return (
      <div className="bg-gray-900/60 border border-gray-800 rounded p-2 h-52 flex items-center justify-center text-gray-600 text-xs">
        Loading {symbol}...
      </div>
    )
  }
  if (error || !candles || candles.length === 0) {
    return (
      <div className="bg-gray-900/60 border border-gray-800/50 rounded p-2 h-52 flex flex-col items-center justify-center text-gray-600 text-xs gap-1">
        <span className="font-bold text-gray-500">{symbol}</span>
        <span className="text-[10px]">No chart data</span>
      </div>
    )
  }

  // ── Layout constants ────────────────────────────────────────────────
  const priceAxisW = 44
  const timeAxisH = 16
  const W = 320
  const chartW = W - priceAxisW
  const stochH = 32
  const priceH = 140
  const totalH = priceH + 6 + stochH + timeAxisH
  const pad = 2
  const barW = Math.max(1.5, (chartW - pad * 2) / candles.length)

  // ── Price scaling ───────────────────────────────────────────────────
  const allPrices = candles.flatMap(c => [c.h, c.l, ...(c.m20 != null ? [c.m20] : []), ...(c.m50 != null ? [c.m50] : [])])
  // Include pattern levels in price range
  for (const p of patterns) {
    if (p.breakout_level) allPrices.push(p.breakout_level)
    if (p.target) allPrices.push(p.target)
    if (p.invalidation_level) allPrices.push(p.invalidation_level)
  }
  const pHigh = Math.max(...allPrices)
  const pLow = Math.min(...allPrices)
  const pRange = pHigh - pLow || 1
  const py = (p: number) => pad + (1 - (p - pLow) / pRange) * (priceH - pad * 2)
  const sy = (v: number) => priceH + 6 + pad + (1 - v / 100) * (stochH - pad * 2)

  const lastCandle = candles[candles.length - 1]
  const prevCandle = candles[candles.length - 2]
  const change = prevCandle ? ((lastCandle.c - prevCandle.c) / prevCandle.c * 100) : 0

  // Price axis ticks — 4 evenly spaced
  const priceTicks = [0, 1, 2, 3].map(i => {
    const p = pLow + (pRange * i) / 3
    return { price: p, y: py(p) }
  })

  // Time axis labels — show every Nth label
  const timeStep = Math.max(1, Math.floor(candles.length / 5))

  // Top pattern info
  const topPattern = patterns.length > 0
    ? patterns.reduce((a, b) => (b.confidence || 0) > (a.confidence || 0) ? b : a)
    : null
  const topDir = topPattern?.direction || 'neutral'

  // Border color from pattern direction
  const dirBorder = topDir === 'bullish'
    ? 'border-emerald-500/30'
    : topDir === 'bearish'
    ? 'border-red-500/30'
    : 'border-gray-800'

  return (
    <div
      className={`bg-gray-900/60 border ${dirBorder} rounded p-2 cursor-pointer hover:border-gray-600 transition-colors`}
      onClick={onOpenChart}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-1">
        <span className="text-white font-bold text-xs font-mono">{symbol}</span>
        <div className="flex items-center gap-1.5 text-[10px] font-mono">
          <span className="text-gray-400">${lastCandle.c.toFixed(2)}</span>
          <span className={change >= 0 ? 'text-emerald-400' : 'text-red-400'}>
            {change >= 0 ? '+' : ''}{change.toFixed(1)}%
          </span>
        </div>
      </div>

      {/* Stoch badge + pattern badge */}
      <div className="flex items-center gap-1 mb-1 flex-wrap">
        <span className={`text-[9px] font-mono px-1 py-0.5 rounded ${
          lastCandle.k < 20 ? 'bg-emerald-500/15 text-emerald-400'
          : lastCandle.k > 80 ? 'bg-red-500/15 text-red-400'
          : 'bg-gray-800 text-gray-500'
        }`}>
          %K {lastCandle.k.toFixed(0)} / %D {lastCandle.d.toFixed(0)}
        </span>
        {topPattern && (
          <span className={`text-[8px] font-mono px-1 py-0.5 rounded truncate max-w-[140px] ${
            topDir === 'bullish' ? 'bg-emerald-500/10 text-emerald-400'
            : topDir === 'bearish' ? 'bg-red-500/10 text-red-400'
            : 'bg-amber-500/10 text-amber-400'
          }`} title={`${topPattern.pattern_name} (${Math.round(topPattern.confidence || 0)}%)`}>
            {topPattern.pattern_name} {Math.round(topPattern.confidence || 0)}%
          </span>
        )}
      </div>

      {/* Chart SVG */}
      <svg width={W} height={totalH} className="w-full" viewBox={`0 0 ${W} ${totalH}`}>
        {/* ── Price chart area ── */}

        {/* MA20 line (orange) */}
        {candles.some(c => c.m20 != null) && (
          <polyline
            points={candles.map((c, i) => c.m20 != null ? `${pad + i * barW + barW / 2},${py(c.m20)}` : '').filter(Boolean).join(' ')}
            fill="none" stroke="#f97316" strokeWidth="1" opacity="0.8"
          />
        )}
        {/* MA50 line (blue) */}
        {candles.some(c => c.m50 != null) && (
          <polyline
            points={candles.map((c, i) => c.m50 != null ? `${pad + i * barW + barW / 2},${py(c.m50)}` : '').filter(Boolean).join(' ')}
            fill="none" stroke="#60a5fa" strokeWidth="1" opacity="0.8"
          />
        )}

        {/* Candlesticks */}
        {candles.map((c, i) => {
          const x = pad + i * barW
          const green = c.c >= c.o
          const color = green ? '#10b981' : '#ef4444'
          const bodyTop = Math.min(py(c.o), py(c.c))
          const bodyH = Math.max(1, Math.abs(py(c.o) - py(c.c)))
          return (
            <g key={i}>
              <line x1={x + barW / 2} y1={py(c.h)} x2={x + barW / 2} y2={py(c.l)} stroke={color} strokeWidth="0.5" />
              <rect x={x + 0.5} y={bodyTop} width={Math.max(1, barW - 1)} height={bodyH} fill={color} />
            </g>
          )
        })}

        {/* ── Pattern horizontal levels ── */}
        {patterns.map((p, pi) => {
          const levels: { price: number; color: string; label: string }[] = []
          if (p.breakout_level && p.breakout_level >= pLow && p.breakout_level <= pHigh) {
            levels.push({ price: p.breakout_level, color: '#3b82f6', label: 'BRK' })
          }
          if (p.target && p.target >= pLow * 0.95 && p.target <= pHigh * 1.05) {
            levels.push({ price: p.target, color: '#22c55e', label: 'TGT' })
          }
          if (p.invalidation_level && p.invalidation_level >= pLow * 0.95 && p.invalidation_level <= pHigh * 1.05) {
            levels.push({ price: p.invalidation_level, color: '#ef4444', label: 'INV' })
          }
          return levels.map((lv, li) => {
            const yy = py(lv.price)
            if (yy < 0 || yy > priceH) return null
            return (
              <g key={`p${pi}-l${li}`}>
                <line x1={0} y1={yy} x2={chartW} y2={yy} stroke={lv.color} strokeWidth="0.7" strokeDasharray="4,3" opacity="0.7" />
                <text x={chartW - 2} y={yy - 2} fill={lv.color} fontSize="7" fontFamily="monospace" textAnchor="end" opacity="0.8">
                  {lv.label}
                </text>
              </g>
            )
          })
        })}

        {/* ── Price axis (right side) ── */}
        <line x1={chartW} y1={pad} x2={chartW} y2={priceH} stroke="#374151" strokeWidth="0.5" />
        {priceTicks.map((t, i) => (
          <g key={`pt${i}`}>
            <line x1={chartW} y1={t.y} x2={chartW + 3} y2={t.y} stroke="#4b5563" strokeWidth="0.5" />
            <text x={chartW + 5} y={t.y + 3} fill="#6b7280" fontSize="7" fontFamily="monospace">
              {t.price >= 100 ? t.price.toFixed(0) : t.price.toFixed(2)}
            </text>
          </g>
        ))}
        {/* Current price line + label */}
        <line x1={pad} y1={py(lastCandle.c)} x2={chartW} y2={py(lastCandle.c)}
              stroke={change >= 0 ? '#10b981' : '#ef4444'} strokeWidth="0.5" strokeDasharray="2,2" opacity="0.5" />
        <text x={chartW + 5} y={py(lastCandle.c) + 3} fill={change >= 0 ? '#10b981' : '#ef4444'} fontSize="7" fontWeight="bold" fontFamily="monospace">
          {lastCandle.c >= 100 ? lastCandle.c.toFixed(0) : lastCandle.c.toFixed(2)}
        </text>

        {/* ── Stochastic separator ── */}
        <line x1={0} y1={priceH + 3} x2={chartW} y2={priceH + 3} stroke="#374151" strokeWidth="0.5" />
        {/* 20/80 lines */}
        <line x1={0} y1={sy(80)} x2={chartW} y2={sy(80)} stroke="#374151" strokeWidth="0.3" strokeDasharray="2,2" />
        <line x1={0} y1={sy(20)} x2={chartW} y2={sy(20)} stroke="#374151" strokeWidth="0.3" strokeDasharray="2,2" />
        {/* %K line */}
        <polyline
          points={candles.map((c, i) => `${pad + i * barW + barW / 2},${sy(c.k)}`).join(' ')}
          fill="none" stroke="#3b82f6" strokeWidth="1"
        />
        {/* %D line */}
        <polyline
          points={candles.map((c, i) => `${pad + i * barW + barW / 2},${sy(c.d)}`).join(' ')}
          fill="none" stroke="#f97316" strokeWidth="1"
        />
        {/* Oversold/overbought zones */}
        <rect x={0} y={sy(20)} width={chartW} height={sy(0) - sy(20)} fill="#10b981" opacity="0.04" />
        <rect x={0} y={sy(100)} width={chartW} height={sy(80) - sy(100)} fill="#ef4444" opacity="0.04" />
        {/* Stoch axis labels */}
        <text x={chartW + 5} y={sy(80) + 3} fill="#6b7280" fontSize="6" fontFamily="monospace">80</text>
        <text x={chartW + 5} y={sy(20) + 3} fill="#6b7280" fontSize="6" fontFamily="monospace">20</text>

        {/* ── Time axis (bottom) ── */}
        <line x1={pad} y1={priceH + 6 + stochH + 2} x2={chartW - pad} y2={priceH + 6 + stochH + 2} stroke="#374151" strokeWidth="0.5" />
        {candles.map((c, i) => {
          if (i % timeStep !== 0 || !c.t) return null
          const x = pad + i * barW + barW / 2
          return (
            <g key={`t${i}`}>
              <line x1={x} y1={priceH + 6 + stochH + 2} x2={x} y2={priceH + 6 + stochH + 5} stroke="#4b5563" strokeWidth="0.5" />
              <text x={x} y={priceH + 6 + stochH + timeAxisH} fill="#6b7280" fontSize="6.5" textAnchor="middle" fontFamily="monospace">
                {c.t}
              </text>
            </g>
          )
        })}
      </svg>
    </div>
  )
})

// ── PatternChartGrid ──────────────────────────────────────────────────────

interface TradeRec {
  symbol: string
  [key: string]: any
}

export default function PatternChartGrid({
  trades,
  onOpenChart,
}: {
  trades: TradeRec[]
  onOpenChart: (symbol: string) => void
}) {
  if (trades.length === 0) return null
  return (
    <div className="grid grid-cols-3 gap-2">
      {trades.map(rec => (
        <PatternChartGridItem
          key={rec.symbol}
          symbol={rec.symbol}
          onOpenChart={() => onOpenChart(rec.symbol)}
        />
      ))}
    </div>
  )
}
