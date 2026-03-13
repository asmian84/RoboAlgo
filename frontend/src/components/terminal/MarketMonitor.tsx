import { useNavigate } from 'react-router-dom'
import type { CommandCenterData } from '../../types'

// ── Color maps ────────────────────────────────────────────────────────────────

const STATE_COLORS: Record<string, string> = {
  EXPANSION:   '#f97316',
  TREND:       '#22c55e',
  COMPRESSION: '#60a5fa',
  CHAOS:       '#ef4444',
}

// ── Sub-components ────────────────────────────────────────────────────────────

function SectionHeader({ label }: { label: string }) {
  return (
    <div className="px-3 py-1.5 bg-gray-800/50 border-b border-gray-800">
      <span className="text-[10px] font-bold text-gray-500 uppercase tracking-wider">{label}</span>
    </div>
  )
}

function MarketRow({
  symbol,
  state,
  volPct,
  trendStr,
  onClick,
}: {
  symbol: string
  state: string
  volPct: number | null
  trendStr: number | null
  onClick: () => void
}) {
  const stateColor = STATE_COLORS[state] ?? '#6b7280'
  return (
    <button
      className="w-full flex items-center gap-2 px-3 py-1.5 hover:bg-gray-800/50 transition-colors border-b border-gray-800/30 text-left"
      onClick={onClick}
    >
      {/* State dot */}
      <span
        className="w-2 h-2 rounded-full flex-shrink-0"
        style={{ backgroundColor: stateColor }}
        title={state}
      />
      {/* Symbol */}
      <span className="text-xs font-mono font-bold text-gray-200 w-14 flex-shrink-0">{symbol}</span>
      {/* State badge */}
      <span
        className="text-[9px] font-bold px-1 rounded uppercase flex-shrink-0"
        style={{ color: stateColor, backgroundColor: stateColor + '20' }}
      >
        {state === 'COMPRESSION' ? 'COMP' : state.slice(0, 4)}
      </span>
      {/* Mini vol bar */}
      {volPct != null && (
        <div className="flex-1 h-1 bg-gray-800 rounded-full overflow-hidden ml-auto">
          <div
            className="h-full rounded-full"
            style={{ width: `${Math.min(volPct * 100, 100)}%`, backgroundColor: stateColor + '80' }}
          />
        </div>
      )}
      {/* Trend strength */}
      {trendStr != null && (
        <span className="text-[9px] font-mono text-gray-500 w-6 text-right flex-shrink-0">
          {trendStr.toFixed(0)}
        </span>
      )}
    </button>
  )
}

function RadarRow({
  symbol,
  opportunityScore,
  isEarly,
  onClick,
}: {
  symbol: string
  opportunityScore: number
  isEarly: boolean
  onClick: () => void
}) {
  const color = opportunityScore >= 70 ? '#22c55e' : opportunityScore >= 50 ? '#eab308' : '#6b7280'
  return (
    <button
      className="w-full flex items-center gap-2 px-3 py-1.5 hover:bg-gray-800/50 transition-colors border-b border-gray-800/30 text-left"
      onClick={onClick}
    >
      <span className="text-xs font-mono font-bold text-gray-200 w-14 flex-shrink-0">{symbol}</span>
      <div className="flex-1 h-1 bg-gray-800 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full"
          style={{ width: `${opportunityScore}%`, backgroundColor: color }}
        />
      </div>
      <span className="text-[10px] font-mono w-6 text-right flex-shrink-0" style={{ color }}>
        {opportunityScore.toFixed(0)}
      </span>
      {isEarly && (
        <span className="text-[9px] font-bold px-1 rounded bg-emerald-900/40 text-emerald-400 flex-shrink-0">
          E
        </span>
      )}
    </button>
  )
}

// ── Main component ─────────────────────────────────────────────────────────────

interface Props {
  data: CommandCenterData | undefined
  isLoading?: boolean
}

export default function MarketMonitor({ data, isLoading }: Props) {
  const navigate = useNavigate()

  const instruments   = data?.market_state_summary?.instruments ?? []
  const radarEntries  = data?.opportunity_radar?.instruments ?? []
  const counts        = data?.market_state_summary?.counts ?? {}

  // Group instruments by state for the mini count row
  const STATE_ORDER = ['EXPANSION', 'TREND', 'COMPRESSION', 'CHAOS']

  return (
    <div className="flex flex-col h-full overflow-hidden bg-gray-900 border-r border-gray-800">
      {/* Panel title */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-gray-800">
        <span className="text-[11px] font-bold text-gray-400 uppercase tracking-wider">Market Monitor</span>
        {isLoading && (
          <span className="text-[10px] text-gray-600 animate-pulse">updating…</span>
        )}
      </div>

      {/* State summary pills */}
      <div className="flex items-center gap-1 px-3 py-2 border-b border-gray-800 flex-wrap">
        {STATE_ORDER.map(s => {
          const n = counts[s] ?? 0
          const c = STATE_COLORS[s]
          return (
            <span
              key={s}
              className="text-[10px] px-1.5 py-0.5 rounded font-bold"
              style={{ color: c, backgroundColor: c + '20' }}
            >
              {s.slice(0, 3)} {n}
            </span>
          )
        })}
      </div>

      {/* Market State — scrollable, takes remaining space */}
      <div className="flex-1 min-h-0 overflow-y-auto">
        {instruments.length > 0 ? (
          <>
            <SectionHeader label={`Market State · ${instruments.length}`} />
            {instruments.map(inst => (
              <MarketRow
                key={inst.symbol}
                symbol={inst.symbol}
                state={inst.state}
                volPct={inst.volatility_percentile}
                trendStr={inst.trend_strength}
                onClick={() => navigate(`/chart?symbol=${inst.symbol}`)}
              />
            ))}
          </>
        ) : (
          <div className="flex flex-col items-center justify-center h-32 gap-2">
            <span className="text-2xl">📡</span>
            <span className="text-xs text-gray-600 text-center px-4">
              No market data yet.
            </span>
          </div>
        )}
      </div>

      {/* Opportunity Radar — fixed, max 5 items, independently scrollable */}
      {radarEntries.length > 0 && (
        <div className="flex-shrink-0 border-t border-gray-800 flex flex-col" style={{ maxHeight: '180px' }}>
          <SectionHeader label={`Radar · Top ${Math.min(radarEntries.length, 5)}`} />
          <div className="overflow-y-auto">
            {radarEntries.slice(0, 5).map(entry => (
              <RadarRow
                key={entry.symbol}
                symbol={entry.symbol}
                opportunityScore={entry.opportunity_score}
                isEarly={entry.is_early_stage}
                onClick={() => navigate(`/chart?symbol=${entry.symbol}`)}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
