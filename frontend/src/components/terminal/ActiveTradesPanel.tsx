import { useNavigate } from 'react-router-dom'
import type { CommandCenterTrade } from '../../types'

const LIFECYCLE_COLORS: Record<string, string> = {
  SETUP:   '#6b7280',
  TRIGGER: '#eab308',
  ENTRY:   '#60a5fa',
  ACTIVE:  '#22c55e',
}

function fmt(v: number | null | undefined, dp = 2, prefix = '') {
  if (v == null) return '—'
  return `${prefix}${v.toFixed(dp)}`
}

// ── Single trade row (vertical layout) ────────────────────────────────────────

function TradeRow({ trade }: { trade: CommandCenterTrade }) {
  const navigate   = useNavigate()
  const stateColor = LIFECYCLE_COLORS[trade.state] ?? '#6b7280'

  return (
    <button
      className="w-full flex flex-col gap-1 px-3 py-2.5 border-b border-gray-800/50 hover:bg-gray-800/30 transition-colors text-left"
      onClick={() => navigate(`/chart?symbol=${trade.symbol}`)}
    >
      {/* Symbol + state */}
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-1.5 min-w-0">
          <span
            className="w-1.5 h-1.5 rounded-full flex-shrink-0"
            style={{ backgroundColor: stateColor }}
          />
          <span className="text-xs font-mono font-bold text-gray-100 truncate">{trade.symbol}</span>
        </div>
        <span
          className="text-[9px] font-bold px-1.5 py-0.5 rounded flex-shrink-0"
          style={{ color: stateColor, backgroundColor: stateColor + '20' }}
        >
          {trade.state}
        </span>
      </div>

      {/* Price levels */}
      <div className="flex items-center gap-2 text-[10px] font-mono pl-3">
        <span className="text-gray-600">E</span>
        <span className="text-gray-300">{fmt(trade.entry_price, 2, '$')}</span>
        <span className="text-gray-600">S</span>
        <span className="text-red-400">{fmt(trade.stop_price, 2, '$')}</span>
        {trade.tier1_sell != null && (
          <>
            <span className="text-gray-600">T1</span>
            <span className="text-emerald-400">{fmt(trade.tier1_sell, 2, '$')}</span>
          </>
        )}
      </div>

      {/* Setup type if available */}
      {trade.setup_type && (
        <p className="text-[9px] text-gray-600 pl-3 truncate">
          {trade.setup_type.replace(/_/g, ' ')}
        </p>
      )}
    </button>
  )
}

// ── Panel ─────────────────────────────────────────────────────────────────────

interface Props {
  trades: CommandCenterTrade[]
  count:  number
}

export default function ActiveTradesPanel({ trades, count }: Props) {
  return (
    <div className="flex flex-col h-full bg-gray-900 border-t border-gray-800 overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-gray-800 flex-shrink-0">
        <span className="text-[10px] font-bold text-gray-400 uppercase tracking-wider">
          Active Trades
        </span>
        <span
          className="text-xs font-bold font-mono px-1.5 py-0.5 rounded-full"
          style={{
            color: count > 0 ? '#22c55e' : '#6b7280',
            backgroundColor: (count > 0 ? '#22c55e' : '#6b7280') + '20',
          }}
        >
          {count}
        </span>
      </div>

      {/* Scrollable trade list */}
      <div className="flex-1 overflow-y-auto overflow-x-hidden">
        {trades.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full gap-2 text-xs text-gray-600 py-4">
            <span className="text-2xl">💤</span>
            <span>No open trades</span>
          </div>
        ) : (
          trades.map(trade => <TradeRow key={trade.id} trade={trade} />)
        )}
      </div>
    </div>
  )
}
