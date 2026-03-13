/**
 * MarketOverviewBanner — Barchart-style major market snapshot.
 *
 * Shown at the top of the Dashboard page as a quick market-context strip.
 * Each cell fetches its own live quote via the shared useLiveQuote hook so
 * that data refreshes every 30 s without a dedicated endpoint.
 */

import { useLiveQuote } from '../../api/hooks'
import { useNavigate } from 'react-router-dom'

// ── Market cells definition ───────────────────────────────────────────────────

interface MarketItem {
  symbol:  string
  label:   string
  sublabel: string
}

const MARKET_ITEMS: MarketItem[] = [
  { symbol: 'SPY',   label: 'S&P 500',   sublabel: 'SPY' },
  { symbol: 'QQQ',   label: 'Nasdaq',    sublabel: 'QQQ' },
  { symbol: 'IWM',   label: 'Russell 2k',sublabel: 'IWM' },
  { symbol: '^VIX',  label: 'VIX',       sublabel: 'Fear Index' },
  { symbol: 'GLD',   label: 'Gold',      sublabel: 'GLD' },
  { symbol: 'USO',   label: 'Crude Oil', sublabel: 'USO' },
  { symbol: 'TLT',   label: '20Y Bond',  sublabel: 'TLT' },
  { symbol: 'UUP',   label: 'US Dollar', sublabel: 'UUP' },
  { symbol: 'IBIT',  label: 'Bitcoin',   sublabel: 'IBIT' },
  { symbol: 'SOXL',  label: 'Semi',      sublabel: 'SOXL 3×' },
]

// ── Individual quote cell ─────────────────────────────────────────────────────

function MarketCell({ item }: { item: MarketItem }) {
  const { data: q, isLoading } = useLiveQuote(item.symbol)
  const navigate = useNavigate()

  const pct    = q?.change_pct
  const change = q?.change
  const isUp   = pct != null && pct >= 0
  const isDown = pct != null && pct < 0
  const color  = isUp ? '#22c55e' : isDown ? '#ef4444' : '#9ca3af'
  const arrow  = isUp ? '▲' : isDown ? '▼' : '◆'

  // VIX behaves inverted — rising VIX = fear = bad
  const vixInverted = item.symbol === '^VIX'
  const displayColor = vixInverted
    ? (isUp ? '#ef4444' : isDown ? '#22c55e' : '#9ca3af')
    : color

  return (
    <button
      onClick={() => navigate(`/chart?symbol=${item.symbol}`)}
      className="flex-shrink-0 flex flex-col items-start gap-0.5 px-3 py-2.5 rounded-xl bg-gray-900 border border-gray-800 hover:border-gray-700 hover:bg-gray-800/60 transition-all min-w-[100px]"
      title={`Open ${item.symbol} chart`}
    >
      {/* Label row */}
      <div className="flex items-center gap-1 w-full">
        <span className="text-[10px] font-bold text-gray-300 truncate">{item.label}</span>
        <span className="ml-auto text-[8px] text-gray-600">{item.sublabel}</span>
      </div>

      {/* Price */}
      <span className="text-sm font-bold font-mono text-white">
        {isLoading
          ? <span className="text-gray-600">—</span>
          : q?.price != null
            ? `$${q.price.toFixed(item.symbol === '^VIX' ? 2 : 2)}`
            : <span className="text-gray-600">—</span>
        }
      </span>

      {/* Change */}
      <div className="flex items-center gap-1 text-[10px] font-mono">
        <span style={{ color: displayColor }}>{arrow}</span>
        <span style={{ color: displayColor }}>
          {pct != null
            ? `${pct >= 0 ? '+' : ''}${pct.toFixed(2)}%`
            : <span className="text-gray-600">—</span>
          }
        </span>
        {change != null && (
          <span className="text-gray-600 text-[9px]">
            ({change >= 0 ? '+' : ''}{change.toFixed(2)})
          </span>
        )}
      </div>
    </button>
  )
}

// ── Panel ─────────────────────────────────────────────────────────────────────

export default function MarketOverviewBanner() {
  return (
    <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-3">
      <div className="flex items-center justify-between mb-2">
        <span className="text-[10px] font-semibold text-gray-500 uppercase tracking-widest">
          Markets
        </span>
        <span className="text-[9px] text-gray-700">Live · 30s refresh · click to open chart</span>
      </div>
      <div className="flex gap-2 overflow-x-auto pb-1 scrollbar-none">
        {MARKET_ITEMS.map(item => (
          <MarketCell key={item.symbol} item={item} />
        ))}
      </div>
    </div>
  )
}
