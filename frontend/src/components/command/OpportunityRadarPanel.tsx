import { useNavigate } from 'react-router-dom'
import type { OpportunityRadarEntry } from '../../types'

// ── Color helpers ────────────────────────────────────────────────────────────

const STATE_COLORS: Record<string, string> = {
  EXPANSION:   '#f97316',
  TREND:       '#22c55e',
  COMPRESSION: '#60a5fa',
  CHAOS:       '#ef4444',
  UNKNOWN:     '#6b7280',
}

function scoreColor(s: number): string {
  if (s >= 70) return '#22c55e'
  if (s >= 50) return '#eab308'
  if (s >= 30) return '#f97316'
  return '#ef4444'
}

// ── Sub-components ────────────────────────────────────────────────────────────

function MiniBar({ score }: { score: number }) {
  const pct  = Math.max(0, Math.min(100, score))
  const col  = scoreColor(score)
  return (
    <div className="flex items-center gap-1.5">
      <div className="w-16 h-1.5 bg-gray-800 rounded-full overflow-hidden">
        <div className="h-full rounded-full" style={{ width: `${pct}%`, backgroundColor: col }} />
      </div>
      <span className="text-[10px] font-mono w-6 text-right" style={{ color: col }}>
        {score.toFixed(0)}
      </span>
    </div>
  )
}

function EarlyBadge() {
  return (
    <span className="px-1.5 py-0.5 rounded text-[10px] font-bold bg-emerald-900/40 text-emerald-400 border border-emerald-700/40">
      EARLY
    </span>
  )
}

// ── Panel ─────────────────────────────────────────────────────────────────────

interface Props {
  instruments:      OpportunityRadarEntry[]
  earlyStageCount:  number
  error?:           string
}

export default function OpportunityRadarPanel({ instruments, earlyStageCount, error }: Props) {
  const navigate = useNavigate()

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800">
        <div>
          <h2 className="text-sm font-semibold text-gray-100">Opportunity Radar</h2>
          <p className="text-xs text-gray-500 mt-0.5">
            Early-stage compression setups · {instruments.length} scanned
          </p>
        </div>
        <span className="px-2 py-0.5 rounded-full text-xs font-bold bg-emerald-900/30 text-emerald-400">
          {earlyStageCount} Early
        </span>
      </div>

      {/* Error banner */}
      {error && (
        <div className="mx-4 mt-3 px-3 py-2 rounded-lg bg-red-900/20 border border-red-800/40 text-xs text-red-400">
          {error}
        </div>
      )}

      {/* Table */}
      {instruments.length === 0 ? (
        <div className="flex items-center justify-center h-24 text-gray-600 text-sm">
          No radar data — run opportunity scan first
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-gray-800 text-gray-500 uppercase">
                <th className="text-left px-4 py-2 font-medium">Symbol</th>
                <th className="px-4 py-2 font-medium">Opp Score</th>
                <th className="px-4 py-2 font-medium">Compress</th>
                <th className="px-4 py-2 font-medium">Shelf</th>
                <th className="px-4 py-2 font-medium">Proximity</th>
                <th className="text-left px-4 py-2 font-medium">State</th>
              </tr>
            </thead>
            <tbody>
              {instruments.map(inst => {
                const stateColor = STATE_COLORS[inst.market_state] ?? '#6b7280'
                return (
                  <tr
                    key={inst.symbol}
                    className="border-b border-gray-800/40 hover:bg-gray-800/30 cursor-pointer transition-colors"
                    onClick={() => navigate(`/chart?symbol=${inst.symbol}`)}
                  >
                    <td className="px-4 py-2">
                      <div className="flex items-center gap-2">
                        <span className="font-mono font-bold text-gray-100">{inst.symbol}</span>
                        {inst.is_early_stage && <EarlyBadge />}
                      </div>
                    </td>
                    <td className="px-4 py-2">
                      <MiniBar score={inst.opportunity_score} />
                    </td>
                    <td className="px-4 py-2">
                      <MiniBar score={inst.compression_score} />
                    </td>
                    <td className="px-4 py-2">
                      <MiniBar score={inst.shelf_score} />
                    </td>
                    <td className="px-4 py-2">
                      <MiniBar score={inst.proximity_score} />
                    </td>
                    <td className="px-4 py-2">
                      <span
                        className="px-1.5 py-0.5 rounded text-[10px] font-bold uppercase"
                        style={{ color: stateColor, backgroundColor: stateColor + '20' }}
                      >
                        {inst.market_state}
                      </span>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
