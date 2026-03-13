import type { SignalReliabilityData, SignalReliabilityEntry, ReliabilityStatus } from '../../types'

// ── Color helpers ────────────────────────────────────────────────────────────

const STATUS_COLORS: Record<ReliabilityStatus, string> = {
  healthy:  '#22c55e',
  warning:  '#eab308',
  disabled: '#ef4444',
  no_data:  '#6b7280',
}

const STATUS_LABELS: Record<ReliabilityStatus, string> = {
  healthy:  'Healthy',
  warning:  'Warning',
  disabled: 'Disabled',
  no_data:  'No Data',
}

function scoreColor(s: number | null): string {
  if (s == null) return '#6b7280'
  if (s >= 70)   return '#22c55e'
  if (s >= 50)   return '#eab308'
  return '#ef4444'
}

// ── Circular progress ring ────────────────────────────────────────────────────

function ReliabilityRing({ score, status }: { score: number | null; status: ReliabilityStatus }) {
  const color  = status === 'healthy' ? scoreColor(score) : STATUS_COLORS[status]
  const pct    = score != null ? Math.max(0, Math.min(100, score)) : 0
  const r      = 14
  const circ   = 2 * Math.PI * r
  const dash   = (pct / 100) * circ

  return (
    <svg width="36" height="36" viewBox="0 0 36 36" className="flex-shrink-0">
      {/* Track */}
      <circle cx="18" cy="18" r={r} fill="none" stroke="#374151" strokeWidth="3" />
      {/* Progress */}
      <circle
        cx="18" cy="18" r={r}
        fill="none"
        stroke={color}
        strokeWidth="3"
        strokeDasharray={`${dash} ${circ}`}
        strokeLinecap="round"
        transform="rotate(-90 18 18)"
        style={{ transition: 'stroke-dasharray 0.4s ease' }}
      />
      {/* Label */}
      <text x="18" y="22" textAnchor="middle" fontSize="9" fill={color} fontFamily="monospace" fontWeight="bold">
        {score != null ? score.toFixed(0) : '—'}
      </text>
    </svg>
  )
}

// ── Strategy row ─────────────────────────────────────────────────────────────

function StrategyRow({ entry }: { entry: SignalReliabilityEntry }) {
  const statusColor = STATUS_COLORS[entry.status]
  const multColor   = entry.position_multiplier >= 1.0 ? '#22c55e'
                    : entry.position_multiplier > 0     ? '#eab308' : '#ef4444'

  return (
    <tr className="border-b border-gray-800/40 hover:bg-gray-800/20 transition-colors">
      <td className="px-4 py-2">
        <ReliabilityRing score={entry.reliability_score} status={entry.status} />
      </td>
      <td className="px-4 py-2">
        <div>
          <p className="text-xs font-semibold text-gray-200 truncate max-w-[140px]">
            {entry.strategy_label}
          </p>
          <span
            className="text-[10px] font-medium"
            style={{ color: statusColor }}
          >
            {STATUS_LABELS[entry.status]}
          </span>
        </div>
      </td>
      <td className="px-4 py-2 text-right font-mono">
        {entry.win_rate != null
          ? <span className={entry.win_rate >= 0.5 ? 'text-emerald-400' : 'text-red-400'}>
              {(entry.win_rate * 100).toFixed(0)}%
            </span>
          : <span className="text-gray-600">—</span>}
      </td>
      <td className="px-4 py-2 text-right font-mono">
        {entry.expectancy != null
          ? <span className={entry.expectancy >= 0 ? 'text-emerald-400' : 'text-red-400'}>
              {entry.expectancy >= 0 ? '+' : ''}{(entry.expectancy * 100).toFixed(1)}%
            </span>
          : <span className="text-gray-600">—</span>}
      </td>
      <td className="px-4 py-2 text-right font-mono text-gray-400">
        {entry.trade_count > 0
          ? <span title={entry.proxy ? 'Pattern signals (estimated)' : undefined}>
              {entry.trade_count}{entry.proxy && <span className="text-gray-600 text-[9px] ml-0.5">~</span>}
            </span>
          : <span className="text-gray-600">0</span>}
      </td>
      <td className="px-4 py-2 text-right font-mono">
        <span style={{ color: multColor }}>{entry.position_multiplier.toFixed(1)}×</span>
      </td>
    </tr>
  )
}

// ── Panel ─────────────────────────────────────────────────────────────────────

interface Props {
  data: SignalReliabilityData
}

export default function SignalReliabilityPanel({ data }: Props) {
  const { strategies = [], system_reliability, disabled_count, warning_count, error } = data
  const hasProxy = strategies.some(s => s.proxy)

  const sysColor   = scoreColor(system_reliability)
  const sysLabel   = system_reliability == null ? '—'
                   : system_reliability >= 70 ? 'Reliable'
                   : system_reliability >= 50 ? 'Degraded'
                   : 'At Risk'

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800">
        <div>
          <h2 className="text-sm font-semibold text-gray-100">Signal Reliability</h2>
          <p className="text-xs text-gray-500 mt-0.5">
            Per-strategy win rates · {strategies.length} strategies
          </p>
        </div>
        <div className="flex items-center gap-2">
          {disabled_count > 0 && (
            <span className="px-1.5 py-0.5 rounded text-[10px] font-bold bg-red-900/30 text-red-400">
              {disabled_count} Disabled
            </span>
          )}
          {warning_count > 0 && (
            <span className="px-1.5 py-0.5 rounded text-[10px] font-bold bg-amber-900/30 text-amber-400">
              {warning_count} Warning
            </span>
          )}
        </div>
      </div>

      {/* Error banner */}
      {error && (
        <div className="mx-4 mt-3 px-3 py-2 rounded-lg bg-red-900/20 border border-red-800/40 text-xs text-red-400">
          {error}
        </div>
      )}

      {/* System reliability summary */}
      {system_reliability != null && (
        <div className="flex items-center gap-3 px-4 py-3 border-b border-gray-800/50 bg-gray-800/20">
          <div className="flex flex-col items-center">
            <span className="text-xl font-bold font-mono" style={{ color: sysColor }}>
              {system_reliability.toFixed(0)}
            </span>
            <span className="text-[10px] text-gray-500">System</span>
          </div>
          <div className="flex-1 h-2 bg-gray-800 rounded-full overflow-hidden">
            <div
              className="h-full rounded-full transition-all"
              style={{ width: `${system_reliability}%`, backgroundColor: sysColor }}
            />
          </div>
          <span className="text-xs font-semibold" style={{ color: sysColor }}>{sysLabel}</span>
        </div>
      )}

      {/* Strategy table */}
      {strategies.length === 0 ? (
        <div className="flex items-center justify-center h-24 text-gray-600 text-sm">
          No reliability data — run trades first
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-gray-800 text-gray-500 uppercase">
                <th className="px-4 py-2 w-10" />
                <th className="text-left px-4 py-2 font-medium">Strategy</th>
                <th className="text-right px-4 py-2 font-medium">Win%</th>
                <th className="text-right px-4 py-2 font-medium">Expect</th>
                <th className="text-right px-4 py-2 font-medium">Signals~</th>
                <th className="text-right px-4 py-2 font-medium">Mult</th>
              </tr>
            </thead>
            <tbody>
              {strategies.map(entry => (
                <StrategyRow key={entry.setup_type} entry={entry} />
              ))}
            </tbody>
          </table>
          {hasProxy && (
            <p className="px-4 py-2 text-[10px] text-gray-600 border-t border-gray-800/50">
              ~ Estimated from pattern signal outcomes · Live trade data will replace once available
            </p>
          )}
        </div>
      )}
    </div>
  )
}
