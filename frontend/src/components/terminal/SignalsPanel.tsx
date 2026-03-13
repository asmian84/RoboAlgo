import { useNavigate } from 'react-router-dom'
import type { CommandCenterSignal } from '../../types'

// ── Color helpers ─────────────────────────────────────────────────────────────

const STATE_COLORS: Record<string, string> = {
  EXPANSION:   '#f97316',
  TREND:       '#22c55e',
  COMPRESSION: '#60a5fa',
  CHAOS:       '#ef4444',
}

const TIER_COLORS: Record<string, string> = {
  HIGH:   '#ef4444',
  MEDIUM: '#f97316',
  WATCH:  '#eab308',
  NONE:   '#6b7280',
}

function qualityColor(s: number | null): string {
  if (s == null)  return '#6b7280'
  if (s >= 70)    return '#22c55e'
  if (s >= 50)    return '#eab308'
  return '#ef4444'
}

function MiniScoreBar({ score, color }: { score: number; color: string }) {
  return (
    <div className="flex items-center gap-1">
      <div className="w-10 h-1 bg-gray-800 rounded-full overflow-hidden">
        <div className="h-full rounded-full" style={{ width: `${Math.min(score, 100)}%`, backgroundColor: color }} />
      </div>
      <span className="text-[10px] font-mono w-5 text-right" style={{ color }}>{score.toFixed(0)}</span>
    </div>
  )
}

// ── Signal row ────────────────────────────────────────────────────────────────

function SignalRow({ sig }: { sig: CommandCenterSignal }) {
  const navigate    = useNavigate()
  const stateColor  = STATE_COLORS[sig.market_state] ?? '#6b7280'
  const tierColor   = TIER_COLORS[sig.signal_tier]   ?? '#6b7280'
  const qColor      = qualityColor(sig.setup_quality_score)
  const confColor   = qualityColor(sig.confluence_score)
  const sizeColor   = sig.size_approved ? '#22c55e' : '#eab308'

  return (
    <button
      className="w-full flex flex-col gap-1 px-3 py-2 border-b border-gray-800/40 hover:bg-gray-800/40 transition-colors text-left"
      onClick={() => navigate(`/chart?symbol=${sig.symbol}`)}
    >
      {/* Row 1: Symbol + tier + state */}
      <div className="flex items-center gap-1.5">
        <span className="text-xs font-mono font-bold text-gray-100 w-16 flex-shrink-0">{sig.symbol}</span>
        <span
          className="text-[9px] font-bold px-1 py-0.5 rounded uppercase"
          style={{ color: tierColor, backgroundColor: tierColor + '20' }}
        >
          {sig.signal_tier}
        </span>
        <span
          className="text-[9px] font-bold px-1 py-0.5 rounded uppercase ml-auto"
          style={{ color: stateColor, backgroundColor: stateColor + '20' }}
        >
          {sig.market_state === 'COMPRESSION' ? 'COMP' : (sig.market_state ?? '?').slice(0, 4)}
        </span>
      </div>

      {/* Row 2: Score bars */}
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-1">
          <span className="text-[9px] text-gray-600">CF</span>
          <MiniScoreBar score={sig.confluence_score} color={confColor} />
        </div>
        {sig.setup_quality_score != null && (
          <div className="flex items-center gap-1">
            <span className="text-[9px] text-gray-600">Q</span>
            <MiniScoreBar score={sig.setup_quality_score} color={qColor} />
          </div>
        )}
        {sig.position_size_multiplier != null && (
          <div className="flex items-center gap-1 ml-auto">
            <span className="text-[9px] text-gray-600">Sz</span>
            <span className="text-[10px] font-mono font-bold" style={{ color: sizeColor }}>
              {sig.position_size_multiplier.toFixed(1)}×
            </span>
          </div>
        )}
      </div>

      {/* Row 3: Move + flags */}
      <div className="flex items-center gap-1.5">
        {sig.expected_move_pct != null && (
          <span className="text-[10px] font-mono text-emerald-400">
            +{(sig.expected_move_pct * 100).toFixed(1)}%
          </span>
        )}
        {sig.is_compression && (
          <span className="text-[9px] px-1 rounded bg-blue-900/30 text-blue-400 font-bold">COMP</span>
        )}
        {sig.is_breakout && (
          <span className="text-[9px] px-1 rounded bg-orange-900/30 text-orange-400 font-bold">BKT</span>
        )}
        {sig.entry_price != null && (
          <span className="text-[9px] font-mono text-gray-600 ml-auto">
            E ${sig.entry_price.toFixed(2)}
          </span>
        )}
      </div>
    </button>
  )
}

// ── Panel ─────────────────────────────────────────────────────────────────────

interface Props {
  signals:   CommandCenterSignal[]
  isLoading?: boolean
}

export default function SignalsPanel({ signals, isLoading }: Props) {
  return (
    <div className="flex flex-col h-full overflow-hidden bg-gray-900 border-t border-gray-800">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-gray-800 flex-shrink-0">
        <span className="text-[11px] font-bold text-gray-400 uppercase tracking-wider">Signals</span>
        <div className="flex items-center gap-2">
          <span className="text-[10px] text-gray-600">{signals.length} setups</span>
          {isLoading && <span className="text-[10px] text-gray-600 animate-pulse">…</span>}
        </div>
      </div>

      {/* Signal list */}
      <div className="flex-1 overflow-y-auto">
        {signals.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-32 gap-2">
            <span className="text-xl">📊</span>
            <span className="text-[11px] text-gray-600 text-center px-3">
              No signals. Run confluence engine.
            </span>
          </div>
        ) : (
          signals.map(sig => <SignalRow key={sig.symbol} sig={sig} />)
        )}
      </div>
    </div>
  )
}
