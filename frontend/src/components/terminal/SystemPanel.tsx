import { useState } from 'react'
import type { CommandCenterData } from '../../types'

// ── Tab definitions ───────────────────────────────────────────────────────────

type Tab = 'portfolio' | 'reliability' | 'health'

const TABS: { id: Tab; label: string }[] = [
  { id: 'portfolio',   label: 'Portfolio' },
  { id: 'reliability', label: 'Reliability' },
  { id: 'health',      label: 'System Health' },
]

// ── Metric pill ───────────────────────────────────────────────────────────────

function Metric({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div className="flex flex-col items-center px-3 py-1">
      <span className="text-[9px] text-gray-600 uppercase">{label}</span>
      <span className="text-xs font-mono font-bold" style={{ color: color ?? '#f3f4f6' }}>{value}</span>
    </div>
  )
}

// ── Portfolio tab ─────────────────────────────────────────────────────────────

function PortfolioTab({ data }: { data: CommandCenterData['portfolio_risk'] }) {
  const pnlColor  = (data.daily_pnl_pct ?? 0) >= 0 ? '#22c55e' : '#ef4444'
  const riskColor = (data.risk_budget_remaining ?? 0) > 0.02 ? '#22c55e' : '#ef4444'
  const slotsAvail = data.slots_available ?? 0

  return (
    <div className="flex items-center gap-2 h-full overflow-x-auto px-2">
      <Metric
        label="Equity"
        value={data.account_equity != null ? `$${(data.account_equity / 1000).toFixed(0)}K` : '—'}
      />
      <div className="w-px h-6 bg-gray-800" />
      <Metric
        label="Positions"
        value={`${data.open_positions ?? 0}/${data.max_positions ?? 5}`}
        color={slotsAvail > 0 ? '#22c55e' : '#ef4444'}
      />
      <Metric
        label="Slots"
        value={String(slotsAvail)}
        color={slotsAvail > 0 ? '#22c55e' : '#ef4444'}
      />
      <div className="w-px h-6 bg-gray-800" />
      <Metric
        label="Daily P&L"
        value={data.daily_pnl_pct != null ? `${(data.daily_pnl_pct * 100).toFixed(2)}%` : '—'}
        color={pnlColor}
      />
      <Metric
        label="Risk Budget"
        value={data.risk_budget_remaining != null ? `${(data.risk_budget_remaining * 100).toFixed(1)}%` : '—'}
        color={riskColor}
      />
      <Metric
        label="Daily Limit"
        value={data.daily_loss_limit != null ? `${(data.daily_loss_limit * 100).toFixed(0)}%` : '—'}
        color="#f97316"
      />
      {/* Sector mini bars */}
      {data.sector_exposure && Object.keys(data.sector_exposure).length > 0 && (
        <>
          <div className="w-px h-6 bg-gray-800" />
          <div className="flex items-center gap-1.5 overflow-x-auto">
            {Object.entries(data.sector_exposure)
              .sort(([,a],[,b]) => b - a)
              .slice(0, 6)
              .map(([sector, pct]) => {
                const over  = pct > 0.40
                const color = over ? '#ef4444' : pct > 0.25 ? '#f97316' : '#22c55e'
                const name  = sector.replace('Technology', 'Tech').replace('Healthcare', 'Health').replace('Financial', 'Fin').slice(0, 6)
                return (
                  <div key={sector} className="flex flex-col items-center gap-0.5" title={`${sector}: ${(pct*100).toFixed(0)}%`}>
                    <span className="text-[9px] text-gray-600">{name}</span>
                    <div className="w-6 h-1 bg-gray-800 rounded-full overflow-hidden">
                      <div className="h-full rounded-full" style={{ width: `${Math.min(pct,1)*100}%`, backgroundColor: color }} />
                    </div>
                  </div>
                )
              })}
          </div>
        </>
      )}
    </div>
  )
}

// ── Reliability tab ───────────────────────────────────────────────────────────

function ReliabilityTab({ data }: { data: CommandCenterData['signal_reliability'] }) {
  const { strategies = [], system_reliability, disabled_count, warning_count } = data
  const sysColor = system_reliability == null ? '#6b7280'
                 : system_reliability >= 70 ? '#22c55e'
                 : system_reliability >= 50 ? '#eab308' : '#ef4444'

  return (
    <div className="flex items-center gap-3 h-full overflow-x-auto px-2">
      {/* System score */}
      <div className="flex flex-col items-center px-2">
        <span className="text-[9px] text-gray-600 uppercase">System</span>
        <span className="text-sm font-mono font-bold" style={{ color: sysColor }}>
          {system_reliability?.toFixed(0) ?? '—'}
        </span>
      </div>
      {disabled_count > 0 && (
        <Metric label="Disabled" value={String(disabled_count)} color="#ef4444" />
      )}
      {warning_count > 0 && (
        <Metric label="Warning" value={String(warning_count)} color="#eab308" />
      )}
      <div className="w-px h-6 bg-gray-800" />
      {/* Strategy tiles */}
      {strategies.slice(0, 8).map(s => {
        const statusColor = s.status === 'healthy' ? '#22c55e'
                          : s.status === 'warning'  ? '#eab308'
                          : s.status === 'disabled' ? '#ef4444' : '#6b7280'
        return (
          <div key={s.setup_type} className="flex flex-col items-center px-1.5" title={s.strategy_label}>
            <span className="text-[9px] text-gray-600 truncate max-w-[48px]">
              {s.strategy_label.split(' ')[0]}
            </span>
            <span className="text-[10px] font-mono font-bold" style={{ color: statusColor }}>
              {s.reliability_score?.toFixed(0) ?? '—'}
            </span>
          </div>
        )
      })}
    </div>
  )
}

// ── Health tab ────────────────────────────────────────────────────────────────

function HealthTab({ data }: { data: CommandCenterData['system_health'] }) {
  const score       = data.data_quality_score ?? 0
  const scoreColor  = score >= 90 ? '#22c55e' : score >= 75 ? '#4ade80' : score >= 50 ? '#eab308' : '#ef4444'
  const scoreLabel  = score >= 90 ? 'Excellent' : score >= 75 ? 'Good' : score >= 50 ? 'Fair' : 'Poor'
  const pipelineColors: Record<string, string> = { OK: '#22c55e', STALE: '#eab308', ERROR: '#ef4444' }
  const pipelineColor = pipelineColors[data.pipeline_status] ?? '#6b7280'

  function fmtDate(iso: string | null) {
    if (!iso) return '—'
    try { return new Date(iso).toLocaleDateString([], { month: 'short', day: 'numeric' }) } catch { return iso }
  }

  return (
    <div className="flex items-center gap-3 h-full overflow-x-auto px-2">
      <Metric label="Quality" value={`${score.toFixed(0)} · ${scoreLabel}`} color={scoreColor} />
      <div className="w-px h-6 bg-gray-800" />
      <Metric label="Pipeline" value={data.pipeline_status ?? '—'} color={pipelineColor} />
      <Metric label="Last Update" value={fmtDate(data.last_data_update)} color={pipelineColor} />
      <Metric label="Instruments" value={String(data.total_instruments ?? 0)} />
      {data.data_issues && (
        <>
          <div className="w-px h-6 bg-gray-800" />
          <Metric
            label="Critical"
            value={String(data.data_issues.total_critical ?? 0)}
            color={(data.data_issues.total_critical ?? 0) > 0 ? '#ef4444' : '#22c55e'}
          />
          <Metric
            label="Below 80"
            value={String(data.data_issues.below_80 ?? 0)}
            color={(data.data_issues.below_80 ?? 0) > 0 ? '#f97316' : '#22c55e'}
          />
        </>
      )}
    </div>
  )
}

// ── Main panel ────────────────────────────────────────────────────────────────

interface Props {
  data: CommandCenterData | undefined
}

export default function SystemPanel({ data }: Props) {
  const [tab, setTab] = useState<Tab>('portfolio')

  return (
    <div className="flex items-stretch h-full bg-gray-900 border-t border-gray-800 overflow-hidden">
      {/* Tabs */}
      <div className="flex flex-col justify-center border-r border-gray-800 px-1 flex-shrink-0 gap-0.5">
        {TABS.map(t => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`text-[10px] font-bold px-2 py-1 rounded transition-colors text-left whitespace-nowrap ${
              tab === t.id
                ? 'bg-gray-700 text-gray-200'
                : 'text-gray-600 hover:text-gray-400 hover:bg-gray-800/50'
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-hidden">
        {!data ? (
          <div className="flex items-center justify-center h-full text-xs text-gray-600">
            Loading…
          </div>
        ) : (
          <>
            {tab === 'portfolio'   && data.portfolio_risk    && <PortfolioTab   data={data.portfolio_risk}    />}
            {tab === 'reliability' && data.signal_reliability && <ReliabilityTab data={data.signal_reliability} />}
            {tab === 'health'      && data.system_health      && <HealthTab      data={data.system_health}      />}
          </>
        )}
      </div>
    </div>
  )
}
