/**
 * TradeReplayPlayer — replay closed trades bar-by-bar on a mini chart.
 *
 * Features:
 *   - Select a closed trade from a dropdown
 *   - Play / Pause / Step forward / Speed control (1×/2×/4×)
 *   - Bar-by-bar price replay with entry/exit markers
 *   - Shows cumulative P&L as the replay progresses
 *   - Reset rewinds to the trade entry bar
 */
import { useEffect, useRef, useState, useCallback } from 'react'
import { createChart, ColorType, type IChartApi, type ISeriesApi } from 'lightweight-charts'
import { useQuery } from '@tanstack/react-query'
import api from '../../api/client'

// ── Types ─────────────────────────────────────────────────────────────────────

interface ClosedTrade {
  id:           number
  symbol:       string
  state:        string
  setup_type:   string | null
  entry_price:  number | null
  exit_price:   number | null
  stop_price:   number | null
  tier1_sell:   number | null
  tier2_sell:   number | null
  pnl:          number | null
  return_pct:   number | null
  entry_at:     string | null
  exit_at:      string | null
  holding_days: number | null
}

interface PriceBar {
  date:  string
  open:  number | null
  high:  number | null
  low:   number | null
  close: number | null
}

// ── Fetch helpers ─────────────────────────────────────────────────────────────

function useClosedTrades() {
  return useQuery<ClosedTrade[]>({
    queryKey: ['trades', 'closed'],
    // No state param → defaults to EXIT (completed) trades on the backend
    queryFn:  () => api.get('/analytics/trade-history', { params: { limit: 50 } }).then(r => r.data.trades ?? r.data),
    staleTime: 60_000,
  })
}

function useReplayBars(symbol: string, entryDate: string | null) {
  return useQuery<PriceBar[]>({
    queryKey: ['replay-bars', symbol, entryDate],
    queryFn:  () => api.get(`/chart/${symbol}`, { params: { limit: 500 } }).then(r => r.data.prices as PriceBar[]),
    enabled:  !!symbol && !!entryDate,
    staleTime: 5 * 60_000,
  })
}

// ── Speed options ─────────────────────────────────────────────────────────────

const SPEED_OPTIONS = [1, 2, 4, 8] as const
type Speed = typeof SPEED_OPTIONS[number]

// ── P&L color helper ─────────────────────────────────────────────────────────

function pnlColor(v: number | null): string {
  if (v == null) return '#6b7280'
  return v >= 0 ? '#22c55e' : '#ef4444'
}

// ── Main Component ────────────────────────────────────────────────────────────

export default function TradeReplayPlayer({ className = '' }: { className?: string }) {
  const { data: trades, isLoading: tradesLoading } = useClosedTrades()

  const [selectedId, setSelectedId] = useState<number | null>(null)
  const [isPlaying,  setIsPlaying]  = useState(false)
  const [speed,      setSpeed]      = useState<Speed>(1)
  const [barIdx,     setBarIdx]     = useState(0)    // current bar index in window

  const trade = trades?.find(t => t.id === selectedId) ?? null

  const { data: allBars } = useReplayBars(trade?.symbol ?? '', trade?.entry_at ?? null)

  // Slice bars starting from entry (up to 100 bars after)
  const replayBars = (() => {
    if (!allBars || !trade?.entry_at) return []
    const entryDate = trade.entry_at.slice(0, 10)
    const startIdx = allBars.findIndex(b => b.date >= entryDate)
    if (startIdx < 0) return []
    // Include 30 bars before entry for context
    const contextStart = Math.max(0, startIdx - 30)
    return allBars.slice(contextStart, startIdx + 120).filter(b => b.close != null)
  })()

  const entryBarIdx = (() => {
    if (!trade?.entry_at || !replayBars.length) return 0
    const entryDate = trade.entry_at.slice(0, 10)
    return Math.max(0, replayBars.findIndex(b => b.date >= entryDate))
  })()

  const exitBarIdx = (() => {
    if (!trade?.exit_at || !replayBars.length) return replayBars.length - 1
    const exitDate = trade.exit_at.slice(0, 10)
    const idx = replayBars.findIndex(b => b.date >= exitDate)
    return idx < 0 ? replayBars.length - 1 : idx
  })()

  const atEnd = barIdx >= replayBars.length - 1

  // ── Chart setup ─────────────────────────────────────────────────────────────

  const containerRef = useRef<HTMLDivElement>(null)
  const chartApi     = useRef<IChartApi | null>(null)
  const candleRef    = useRef<ISeriesApi<'Candlestick'> | null>(null)
  const markersDirty = useRef(false)

  useEffect(() => {
    if (!containerRef.current) return
    try { chartApi.current?.remove() } catch { /* disposed */ }

    const chart = createChart(containerRef.current, {
      autoSize: true,
      layout: {
        background: { type: ColorType.Solid as const, color: '#0a0a0a' },
        textColor: '#9ca3af',
      },
      grid: { vertLines: { color: '#1f2937' }, horzLines: { color: '#1f2937' } },
      crosshair: { mode: 0 as const },
      timeScale: { borderColor: '#374151', timeVisible: false },
      rightPriceScale: { borderColor: '#374151' },
      handleScroll: true,
      handleScale: true,
    })
    chartApi.current = chart
    candleRef.current = chart.addCandlestickSeries({
      upColor:   '#22c55e', downColor:   '#ef4444',
      borderUpColor: '#22c55e', borderDownColor: '#ef4444',
      wickUpColor:   '#22c55e', wickDownColor:   '#ef4444',
    })

    return () => { try { chart.remove() } catch { /* disposed */ } }
  }, [])

  // Reset when trade changes
  useEffect(() => {
    setBarIdx(0)
    setIsPlaying(false)
  }, [selectedId])

  // Update chart data based on current bar index
  useEffect(() => {
    if (!candleRef.current || !replayBars.length || barIdx === 0) return

    const visible = replayBars.slice(0, barIdx + 1)
    candleRef.current.setData(visible.map(b => ({
      time:  b.date as import('lightweight-charts').Time,
      open:  b.open!,
      high:  b.high!,
      low:   b.low!,
      close: b.close!,
    })))

    // Markers: entry + exit (if reached)
    const markers: import('lightweight-charts').SeriesMarker<import('lightweight-charts').Time>[] = []
    if (barIdx >= entryBarIdx && replayBars[entryBarIdx]) {
      markers.push({
        time:     replayBars[entryBarIdx].date as import('lightweight-charts').Time,
        position: 'belowBar',
        color:    '#22c55e',
        shape:    'arrowUp',
        text:     `ENTRY $${trade?.entry_price?.toFixed(2) ?? ''}`,
      })
    }
    if (barIdx >= exitBarIdx && replayBars[exitBarIdx]) {
      const isWin = (trade?.pnl ?? 0) >= 0
      markers.push({
        time:     replayBars[exitBarIdx].date as import('lightweight-charts').Time,
        position: 'aboveBar',
        color:    isWin ? '#22c55e' : '#ef4444',
        shape:    'arrowDown',
        text:     `EXIT $${trade?.exit_price?.toFixed(2) ?? ''}`,
      })
    }
    candleRef.current.setMarkers(markers)
    markersDirty.current = true

    if (barIdx <= entryBarIdx + 5) {
      chartApi.current?.timeScale().fitContent()
    }
  }, [barIdx, replayBars, entryBarIdx, exitBarIdx, trade])

  // Playback interval
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const stopInterval = useCallback(() => {
    if (intervalRef.current) { clearInterval(intervalRef.current); intervalRef.current = null }
  }, [])

  useEffect(() => {
    stopInterval()
    if (!isPlaying || atEnd) { setIsPlaying(false); return }

    const delay = Math.max(60, 400 / speed)
    intervalRef.current = setInterval(() => {
      setBarIdx(prev => {
        if (prev >= replayBars.length - 1) { setIsPlaying(false); return prev }
        return prev + 1
      })
    }, delay)

    return stopInterval
  }, [isPlaying, speed, atEnd, replayBars.length, stopInterval])

  // ── Computed P&L at current bar ──────────────────────────────────────────────

  const currentClose  = replayBars[barIdx]?.close ?? null
  const entryPrice    = trade?.entry_price ?? null
  const unrealizedPct = currentClose && entryPrice ? ((currentClose - entryPrice) / entryPrice) * 100 : null
  const inTrade       = barIdx >= entryBarIdx && barIdx <= exitBarIdx

  // ── Render ─────────────────────────────────────────────────────────────────

  if (tradesLoading) {
    return (
      <div className={`bg-gray-900 border border-gray-800 rounded-xl flex items-center justify-center h-48 ${className}`}>
        <span className="text-gray-600 text-sm">Loading trades…</span>
      </div>
    )
  }

  if (!trades?.length) {
    return (
      <div className={`bg-gray-900 border border-gray-800 rounded-xl flex flex-col items-center justify-center h-48 gap-2 ${className}`}>
        <span className="text-3xl">📺</span>
        <p className="text-gray-600 text-sm">No closed trades to replay</p>
        <p className="text-gray-700 text-xs">Complete some trades first</p>
      </div>
    )
  }

  const progress = replayBars.length > 0 ? (barIdx / (replayBars.length - 1)) * 100 : 0

  return (
    <div className={`bg-gray-900 border border-gray-800 rounded-xl flex flex-col overflow-hidden ${className}`}>

      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800 flex-shrink-0">
        <div>
          <h2 className="text-sm font-semibold text-gray-100">📺 Trade Replay</h2>
          <p className="text-xs text-gray-500 mt-0.5">Bar-by-bar playback of closed trades</p>
        </div>
        {trade && (
          <div className="flex items-center gap-3 text-xs">
            {inTrade && unrealizedPct != null && (
              <span className="font-mono font-bold" style={{ color: pnlColor(unrealizedPct) }}>
                {unrealizedPct >= 0 ? '+' : ''}{unrealizedPct.toFixed(2)}%
              </span>
            )}
            {barIdx > exitBarIdx && trade.return_pct != null && (
              <span className="font-mono font-bold" style={{ color: pnlColor(trade.return_pct) }}>
                Final: {trade.return_pct >= 0 ? '+' : ''}{(trade.return_pct * 100).toFixed(2)}%
              </span>
            )}
          </div>
        )}
      </div>

      {/* Trade selector */}
      <div className="px-4 py-2 border-b border-gray-800 flex-shrink-0">
        <select
          value={selectedId ?? ''}
          onChange={e => setSelectedId(e.target.value ? parseInt(e.target.value) : null)}
          className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-xs text-gray-200 font-mono"
        >
          <option value="">-- Select a closed trade --</option>
          {trades.map(t => (
            <option key={t.id} value={t.id}>
              #{t.id} {t.symbol} · {t.setup_type ?? 'unknown'} ·{' '}
              {t.return_pct != null ? `${(t.return_pct * 100).toFixed(1)}%` : 'N/A'} ·{' '}
              {t.entry_at?.slice(0, 10) ?? '?'}
            </option>
          ))}
        </select>
      </div>

      {/* Chart area */}
      <div className="flex-1 min-h-0 relative">
        {!selectedId && (
          <div className="absolute inset-0 flex items-center justify-center text-gray-600 text-sm">
            Select a trade to replay
          </div>
        )}
        {selectedId && !replayBars.length && (
          <div className="absolute inset-0 flex items-center justify-center text-gray-600 text-sm">
            Loading price data…
          </div>
        )}
        <div ref={containerRef} className="h-full" />
      </div>

      {/* Progress bar */}
      {replayBars.length > 0 && (
        <div className="h-1 bg-gray-800 flex-shrink-0">
          <div
            className="h-full bg-blue-500 transition-all duration-100"
            style={{ width: `${progress}%` }}
          />
        </div>
      )}

      {/* Controls */}
      <div className="flex items-center gap-3 px-4 py-3 border-t border-gray-800 flex-shrink-0">
        {/* Play / Pause */}
        <button
          onClick={() => {
            if (atEnd) { setBarIdx(0); setIsPlaying(true) }
            else setIsPlaying(p => !p)
          }}
          disabled={!replayBars.length}
          className="w-8 h-8 rounded-lg bg-blue-600 hover:bg-blue-500 disabled:opacity-40 flex items-center justify-center text-white text-sm transition-colors"
        >
          {atEnd ? '↺' : isPlaying ? '⏸' : '▶'}
        </button>

        {/* Step forward */}
        <button
          onClick={() => setBarIdx(i => Math.min(i + 1, replayBars.length - 1))}
          disabled={!replayBars.length || atEnd}
          className="w-8 h-8 rounded-lg bg-gray-700 hover:bg-gray-600 disabled:opacity-40 flex items-center justify-center text-gray-300 text-sm transition-colors"
        >
          ▸
        </button>

        {/* Reset */}
        <button
          onClick={() => { setBarIdx(0); setIsPlaying(false) }}
          disabled={!replayBars.length}
          className="w-8 h-8 rounded-lg bg-gray-700 hover:bg-gray-600 disabled:opacity-40 flex items-center justify-center text-gray-300 text-sm transition-colors"
        >
          ⏮
        </button>

        {/* Speed */}
        <div className="flex items-center gap-1 ml-auto">
          {SPEED_OPTIONS.map(s => (
            <button
              key={s}
              onClick={() => setSpeed(s)}
              className={`px-2 py-0.5 rounded text-[10px] font-mono transition-colors ${speed === s ? 'bg-blue-600 text-white' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'}`}
            >
              {s}×
            </button>
          ))}
        </div>

        {/* Bar info */}
        {replayBars.length > 0 && (
          <span className="text-[10px] font-mono text-gray-600">
            {replayBars[barIdx]?.date ?? ''} · bar {barIdx + 1}/{replayBars.length}
          </span>
        )}
      </div>

      {/* Trade metadata */}
      {trade && (
        <div className="px-4 py-2 border-t border-gray-800 grid grid-cols-4 gap-2 text-[10px] flex-shrink-0">
          <div>
            <p className="text-gray-600">Entry</p>
            <p className="font-mono text-emerald-400">${trade.entry_price?.toFixed(2) ?? '—'}</p>
          </div>
          <div>
            <p className="text-gray-600">Stop</p>
            <p className="font-mono text-red-400">${trade.stop_price?.toFixed(2) ?? '—'}</p>
          </div>
          <div>
            <p className="text-gray-600">T1 Target</p>
            <p className="font-mono text-blue-400">${trade.tier1_sell?.toFixed(2) ?? '—'}</p>
          </div>
          <div>
            <p className="text-gray-600">Final P&L</p>
            <p className="font-mono font-bold" style={{ color: pnlColor(trade.return_pct) }}>
              {trade.return_pct != null ? `${(trade.return_pct * 100).toFixed(2)}%` : '—'}
            </p>
          </div>
        </div>
      )}
    </div>
  )
}
