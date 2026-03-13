import { useNavigate } from 'react-router-dom'
import type { CommandCenterSignal } from '../../types'

const STATE_COLORS: Record<string, string> = {
  EXPANSION:   '#f97316',
  TREND:       '#22c55e',
  COMPRESSION: '#60a5fa',
  CHAOS:       '#ef4444',
}

function scoreColor(s: number | null): string {
  if (s == null)  return '#6b7280'
  if (s >= 70)    return '#22c55e'
  if (s >= 50)    return '#eab308'
  return '#ef4444'
}

function ScoreBar({ score }: { score: number | null }) {
  if (score == null) return <span className="text-gray-600 text-xs font-mono">—</span>
  const color = scoreColor(score)
  return (
    <div className="flex items-center gap-1.5">
      <div className="flex-1 h-1.5 bg-gray-800 rounded-full overflow-hidden">
        <div className="h-full rounded-full" style={{ width: `${Math.min(score, 100)}%`, backgroundColor: color }} />
      </div>
      <span className="text-[10px] font-mono w-6 text-right" style={{ color }}>{score.toFixed(0)}</span>
    </div>
  )
}

interface Props {
  signals: CommandCenterSignal[]
}

export default function TopSignalsPanel({ signals }: Props) {
  const navigate = useNavigate()
  // Sort by setup_quality_score desc
  const top = [...signals]
    .sort((a, b) => (b.setup_quality_score ?? 0) - (a.setup_quality_score ?? 0))
    .slice(0, 10)

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl flex flex-col h-full">
      <div className="px-4 py-3 border-b border-gray-800">
        <h2 className="text-sm font-semibold text-gray-100">Top Signals Radar</h2>
        <p className="text-xs text-gray-500 mt-0.5">Ranked by setup quality score</p>
      </div>
      <div className="flex-1 overflow-auto">
        {top.length === 0 ? (
          <div className="flex items-center justify-center h-32 text-gray-600 text-xs">
            No signals — run confluence engine
          </div>
        ) : (
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-gray-800 text-gray-500 uppercase text-[10px]">
                <th className="text-left px-4 py-2">#</th>
                <th className="text-left px-4 py-2">Symbol</th>
                <th className="px-4 py-2">Setup Q</th>
                <th className="px-4 py-2">Breakout</th>
                <th className="px-4 py-2">Liq Align</th>
                <th className="text-left px-4 py-2">Regime</th>
              </tr>
            </thead>
            <tbody>
              {top.map((sig, i) => {
                const stateColor = STATE_COLORS[sig.market_state] ?? '#6b7280'
                return (
                  <tr
                    key={sig.symbol}
                    className="border-b border-gray-800/40 hover:bg-gray-800/30 cursor-pointer transition-colors"
                    onClick={() => navigate(`/chart?symbol=${sig.symbol}`)}
                  >
                    <td className="px-4 py-2 text-gray-600 font-mono">{i + 1}</td>
                    <td className="px-4 py-2 font-mono font-bold text-gray-100">{sig.symbol}</td>
                    <td className="px-4 py-2 w-24">
                      <ScoreBar score={sig.setup_quality_score} />
                    </td>
                    <td className="px-4 py-2 w-24">
                      <ScoreBar score={sig.breakout_quality_score} />
                    </td>
                    <td className="px-4 py-2 w-24">
                      <ScoreBar score={sig.liquidity_alignment} />
                    </td>
                    <td className="px-4 py-2">
                      <span
                        className="text-[10px] font-bold px-1.5 py-0.5 rounded uppercase"
                        style={{ color: stateColor, backgroundColor: stateColor + '20' }}
                      >
                        {sig.market_state === 'COMPRESSION' ? 'COMP' : (sig.market_state ?? '?').slice(0, 4)}
                      </span>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}
