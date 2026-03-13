import { useFeatureMatrix } from '../api/hooks'
import type { FeatureRow } from '../types'

function deriveProb(row: FeatureRow): number {
  return row.momentum ?? 0.5
}

function gaugeColor(pct: number): string {
  if (pct >= 80) return '#22c55e'
  if (pct >= 60) return '#eab308'
  if (pct >= 40) return '#f97316'
  return '#ef4444'
}

function ProbabilityRow({ row, rank }: { row: FeatureRow & { prob: number }; rank: number }) {
  const pct = row.prob * 100
  const color = gaugeColor(pct)

  return (
    <div className="flex items-center gap-3 py-1.5 border-b border-gray-800/40 hover:bg-gray-800/20 rounded px-2">
      <span className="text-gray-600 text-xs w-5 text-right flex-shrink-0">{rank}</span>

      <svg width="36" height="22" viewBox="0 0 36 22" className="flex-shrink-0">
        <path d="M 3 20 A 15 15 0 0 1 33 20" fill="none" stroke="#374151" strokeWidth="3" strokeLinecap="round" />
        <path
          d="M 3 20 A 15 15 0 0 1 33 20"
          fill="none"
          stroke={color}
          strokeWidth="3"
          strokeLinecap="round"
          strokeDasharray={`${Math.PI * 15}`}
          strokeDashoffset={Math.PI * 15 * (1 - row.prob)}
        />
      </svg>

      <span className="font-bold text-white text-xs w-12 flex-shrink-0">{row.symbol}</span>

      <div className="flex-1 h-1.5 bg-gray-800 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full"
          style={{ width: `${Math.round(pct)}%`, backgroundColor: color }}
        />
      </div>

      <span className="font-mono text-xs w-10 text-right flex-shrink-0" style={{ color }}>
        {pct.toFixed(0)}%
      </span>

      <div className="flex gap-1 flex-shrink-0 w-14 justify-end">
        {row.return_5d != null && (
          <span className={`text-[9px] ${row.return_5d > 0 ? 'text-emerald-500' : 'text-red-500'}`}>
            {row.return_5d > 0 ? '▲' : '▼'}5d
          </span>
        )}
        {row.return_20d != null && (
          <span className={`text-[9px] ${row.return_20d > 0 ? 'text-emerald-500' : 'text-red-500'}`}>
            {row.return_20d > 0 ? '▲' : '▼'}20d
          </span>
        )}
      </div>
    </div>
  )
}

export default function ProbabilityPage() {
  const { data: matrix, isLoading } = useFeatureMatrix()

  if (isLoading) return <div className="text-gray-500 py-20 text-center">Loading probabilities...</div>

  const rows = (matrix || [])
    .map(row => ({ ...row, prob: deriveProb(row) }))
    .sort((a, b) => b.prob - a.prob)

  if (rows.length === 0) return (
    <p className="text-gray-500 text-center py-10">No data. Run: python scripts/run_roboalgo.py --step features</p>
  )

  const mid = Math.ceil(rows.length / 2)
  const leftCol  = rows.slice(0, mid)
  const rightCol = rows.slice(mid)

  const strong  = rows.filter(r => r.prob >= 0.70).length
  const neutral = rows.filter(r => r.prob >= 0.45 && r.prob < 0.70).length
  const weak    = rows.filter(r => r.prob < 0.45).length

  return (
    <div>
      <h2 className="text-2xl font-bold mb-1">Probability Rankings</h2>
      <p className="text-sm text-gray-500 mb-4">
        All {rows.length} instruments ranked highest → lowest by momentum probability.
      </p>

      {/* Summary strip */}
      <div className="flex items-center gap-6 mb-5 p-3 bg-gray-900 rounded-lg border border-gray-800 text-sm">
        <div className="flex items-center gap-2">
          <div className="w-2.5 h-2.5 rounded-full bg-emerald-500" />
          <span className="text-gray-300">{strong} Strong <span className="text-gray-500 text-xs">(≥70%)</span></span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-2.5 h-2.5 rounded-full bg-yellow-500" />
          <span className="text-gray-300">{neutral} Neutral <span className="text-gray-500 text-xs">(45-70%)</span></span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-2.5 h-2.5 rounded-full bg-red-500" />
          <span className="text-gray-300">{weak} Weak <span className="text-gray-500 text-xs">(&lt;45%)</span></span>
        </div>
        <div className="ml-auto text-xs text-gray-500">{rows.length} instruments · sorted by probability ↓</div>
      </div>

      {/* Two vertical columns */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8">
        <div>
          <div className="text-xs text-gray-500 mb-1.5 px-2 flex justify-between">
            <span className="text-emerald-500 font-medium">Top {leftCol.length}</span>
            <span>Rank 1 – {mid}</span>
          </div>
          {leftCol.map((row, i) => (
            <ProbabilityRow key={row.symbol} row={row} rank={i + 1} />
          ))}
        </div>
        <div>
          <div className="text-xs text-gray-500 mb-1.5 px-2 flex justify-between">
            <span className="text-gray-400 font-medium">Bottom {rightCol.length}</span>
            <span>Rank {mid + 1} – {rows.length}</span>
          </div>
          {rightCol.map((row, i) => (
            <ProbabilityRow key={row.symbol} row={row} rank={mid + i + 1} />
          ))}
        </div>
      </div>
    </div>
  )
}
