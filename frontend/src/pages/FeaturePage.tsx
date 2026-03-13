import { useFeatureMatrix } from '../api/hooks'

const FEATURE_KEYS = [
  'trend_strength', 'momentum', 'volatility_percentile', 'volume_ratio',
  'cycle_phase', 'macd_norm', 'bb_position', 'price_to_ma50',
  'return_5d', 'return_20d',
] as const

const FEATURE_LABELS: Record<string, string> = {
  trend_strength: 'Trend',
  momentum: 'Mom',
  volatility_percentile: 'Vol%',
  volume_ratio: 'VolR',
  cycle_phase: 'Cycle',
  macd_norm: 'MACD',
  bb_position: 'BB%',
  price_to_ma50: 'MA50%',
  return_5d: 'R5d',
  return_20d: 'R20d',
}

function cellColor(val: number | null): string {
  if (val == null) return 'bg-gray-800'
  const clamped = Math.max(-2, Math.min(2, val))
  if (clamped > 0.5) return 'bg-emerald-700/60'
  if (clamped > 0.1) return 'bg-emerald-900/40'
  if (clamped < -0.5) return 'bg-red-700/60'
  if (clamped < -0.1) return 'bg-red-900/40'
  return 'bg-gray-800/50'
}

export default function FeaturePage() {
  const { data: matrix, isLoading } = useFeatureMatrix()

  if (isLoading) return <div className="text-gray-500 py-20 text-center">Loading features...</div>

  return (
    <div>
      <h2 className="text-2xl font-bold mb-2">Feature Matrix</h2>
      <p className="text-sm text-gray-500 mb-6">
        Normalized feature values across all instruments.
        <span className="ml-2 text-emerald-400">Green = positive/strong</span>
        <span className="ml-2 text-red-400">Red = negative/weak</span>
      </p>

      <div className="overflow-x-auto">
        <table className="text-xs w-full">
          <thead>
            <tr className="border-b border-gray-800">
              <th className="text-left py-2 px-2 text-gray-400 sticky left-0 bg-gray-950">Symbol</th>
              {FEATURE_KEYS.map(k => (
                <th key={k} className="text-center py-2 px-2 text-gray-400">{FEATURE_LABELS[k]}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {(matrix || []).map(row => (
              <tr key={row.symbol} className="border-b border-gray-800/30 hover:bg-gray-800/20">
                <td className="py-1.5 px-2 font-bold text-white sticky left-0 bg-gray-950">{row.symbol}</td>
                {FEATURE_KEYS.map(k => {
                  const val = row[k as keyof typeof row] as number | null
                  return (
                    <td key={k} className={`py-1.5 px-2 text-center font-mono ${cellColor(val)}`}>
                      {val != null ? val.toFixed(2) : '—'}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {(!matrix || matrix.length === 0) && (
        <p className="text-gray-500 text-center py-10">No feature data. Run: python scripts/run_roboalgo.py --step features</p>
      )}
    </div>
  )
}
