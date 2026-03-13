import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useBullBearAnalysis } from '../api/hooks'
import type { BullBearGroup, BullBearInstrument } from '../types'

function ScoreGauge({ instrument, label, onClick }: { instrument: BullBearInstrument | null; label: string; onClick?: () => void }) {
  if (!instrument) return null

  const score = instrument.score
  const radius = 30
  const circumference = Math.PI * radius
  const color = score == null ? '#4b5563' : score >= 60 ? '#22c55e' : score >= 45 ? '#eab308' : '#ef4444'
  const offset = score != null ? circumference - (score / 100) * circumference : circumference

  return (
    <button
      onClick={onClick}
      className="flex flex-col items-center gap-1 min-w-[90px] rounded-lg p-1 transition-colors hover:bg-gray-800/60 cursor-pointer"
      title={instrument.symbol ? `View chart for ${instrument.symbol}` : undefined}
    >
      <svg width="72" height="44" viewBox="0 0 72 44">
        <path d="M 6 40 A 30 30 0 0 1 66 40" fill="none" stroke="#374151" strokeWidth="5" strokeLinecap="round" />
        {score != null && (
          <path
            d="M 6 40 A 30 30 0 0 1 66 40"
            fill="none"
            stroke={color}
            strokeWidth="5"
            strokeLinecap="round"
            strokeDasharray={`${circumference}`}
            strokeDashoffset={offset}
          />
        )}
      </svg>
      <p className="text-base font-bold font-mono -mt-3" style={{ color }}>
        {score != null ? score.toFixed(0) : '—'}
      </p>
      <p className="text-[10px] text-gray-500 -mt-1">{label}</p>
      <p className="text-xs font-bold text-white">{instrument.symbol ?? '—'}</p>
      <p className="text-[10px] text-gray-500">{instrument.phase ?? ''}</p>
    </button>
  )
}

function VerdictBadge({ verdict, color }: { verdict: string; color: string }) {
  return (
    <span
      className="px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider"
      style={{ backgroundColor: color + '22', color, border: `1px solid ${color}44` }}
    >
      {verdict}
    </span>
  )
}

function FeatureBar({ label, value }: { label: string; value: number | null }) {
  if (value == null) return null
  const pct = Math.max(0, Math.min(100, ((value + 0.5) / 1) * 100))
  const color = value > 0.1 ? '#22c55e' : value < -0.1 ? '#ef4444' : '#6b7280'
  return (
    <div className="flex items-center gap-2 text-[10px]">
      <span className="text-gray-500 w-10 text-right">{label}</span>
      <div className="flex-1 h-1.5 bg-gray-800 rounded-full overflow-hidden">
        <div className="h-full rounded-full" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
      <span className="font-mono w-10" style={{ color }}>{value.toFixed(2)}</span>
    </div>
  )
}

function GroupCard({ group }: { group: BullBearGroup }) {
  const navigate = useNavigate()
  const bullFeats = group.bull.features
  const bearFeats = group.bear?.features ?? null
  const underFeats = group.underlying?.features

  const colCount = group.underlying ? 3 : group.bear ? 3 : 1
  const nav = (sym: string | null | undefined) => sym ? navigate(`/chart?symbol=${sym}`) : undefined

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-800 overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 border-b border-gray-800 flex items-center justify-between">
        <div>
          <h3 className="font-bold text-white text-sm">{group.description}</h3>
          <p className="text-[11px] text-gray-500 mt-0.5">
            {group.bull.symbol}
            {group.bear ? ` / ${group.bear.symbol}` : ' (bull-only)'}
            {group.underlying ? ` · ${group.underlying.symbol}` : ''}
          </p>
        </div>
        <VerdictBadge verdict={group.verdict} color={group.verdict_color} />
      </div>

      {/* Gauges row — always show underlying slot */}
      <div className="flex items-start justify-center gap-6 py-4 px-4">
        <ScoreGauge instrument={group.underlying ?? { symbol: group.bull.symbol ? group.description.split(' ')[0] : null, score: null, phase: '', features: null }} label="Underlying" onClick={() => nav(group.underlying?.symbol)} />
        <ScoreGauge instrument={group.bull} label="Bull" onClick={() => nav(group.bull.symbol)} />
        {group.bear && <ScoreGauge instrument={group.bear} label="Bear" onClick={() => nav(group.bear?.symbol)} />}
      </div>

      {/* Reasoning */}
      <div className="px-4 pb-3">
        <p className="text-[11px] text-gray-400 leading-relaxed">{group.reasoning}</p>
      </div>

      {/* Feature comparison */}
      <div className={`px-4 pb-4 grid gap-3`} style={{ gridTemplateColumns: `repeat(${colCount}, 1fr)` }}>
        {group.underlying && (
          <div>
            <p className="text-[10px] text-gray-500 mb-1.5 font-medium">{group.underlying.symbol} Features</p>
            <FeatureBar label="Trend" value={underFeats?.trend_strength ?? null} />
            <FeatureBar label="Mom" value={underFeats?.momentum != null ? underFeats.momentum - 0.5 : null} />
            <FeatureBar label="MACD" value={underFeats?.macd_norm ?? null} />
            <FeatureBar label="R5d" value={underFeats?.return_5d ?? null} />
            <FeatureBar label="R20d" value={underFeats?.return_20d ?? null} />
          </div>
        )}
        <div>
          <p className="text-[10px] text-emerald-500 mb-1.5 font-medium">{group.bull.symbol} Features</p>
          <FeatureBar label="Trend" value={bullFeats?.trend_strength ?? null} />
          <FeatureBar label="Mom" value={bullFeats?.momentum != null ? bullFeats.momentum - 0.5 : null} />
          <FeatureBar label="MACD" value={bullFeats?.macd_norm ?? null} />
          <FeatureBar label="R5d" value={bullFeats?.return_5d ?? null} />
          <FeatureBar label="R20d" value={bullFeats?.return_20d ?? null} />
        </div>
        {group.bear && (
          <div>
            <p className="text-[10px] text-red-500 mb-1.5 font-medium">{group.bear.symbol} Features</p>
            <FeatureBar label="Trend" value={bearFeats?.trend_strength ?? null} />
            <FeatureBar label="Mom" value={bearFeats?.momentum != null ? bearFeats.momentum - 0.5 : null} />
            <FeatureBar label="MACD" value={bearFeats?.macd_norm ?? null} />
            <FeatureBar label="R5d" value={bearFeats?.return_5d ?? null} />
            <FeatureBar label="R20d" value={bearFeats?.return_20d ?? null} />
          </div>
        )}
      </div>
    </div>
  )
}

type VerdictFilter = 'bullish' | 'bearish' | 'neutral' | null

export default function BullBearPage() {
  const { data: groups, isLoading } = useBullBearAnalysis()
  const [filter, setFilter] = useState<VerdictFilter>(null)

  if (isLoading) return <div className="text-gray-500 py-20 text-center">Loading analysis...</div>

  // Summary counts
  const verdicts = (groups || []).map(g => g.verdict)
  const bullish = verdicts.filter(v => v === 'BULLISH' || v === 'LEAN BULL').length
  const bearish = verdicts.filter(v => v === 'BEARISH' || v === 'LEAN BEAR').length
  const neutral = verdicts.filter(v => v === 'NEUTRAL' || v === 'VOLATILE').length

  const filtered = (groups || []).filter(g => {
    if (!filter) return true
    if (filter === 'bullish') return g.verdict === 'BULLISH' || g.verdict === 'LEAN BULL'
    if (filter === 'bearish') return g.verdict === 'BEARISH' || g.verdict === 'LEAN BEAR'
    return g.verdict === 'NEUTRAL' || g.verdict === 'VOLATILE'
  })

  const toggle = (v: VerdictFilter) => setFilter(f => f === v ? null : v)

  return (
    <div>
      <h2 className="text-2xl font-bold mb-2">Bull / Bear Analysis</h2>
      <p className="text-sm text-gray-500 mb-4">
        Leveraged pairs analyzed with their underlying. Scores derived from trend, momentum, MACD, and returns.
      </p>

      {/* Summary bar — clickable filters */}
      <div className="flex flex-wrap items-center gap-2 mb-6 bg-gray-900 rounded-lg px-4 py-3 border border-gray-800">
        {[
          { key: 'bullish' as const, count: bullish, label: 'Bullish', dot: 'bg-emerald-500', text: 'text-emerald-400', active: 'bg-emerald-900/60 ring-1 ring-emerald-500/50' },
          { key: 'bearish' as const, count: bearish, label: 'Bearish', dot: 'bg-red-500', text: 'text-red-400', active: 'bg-red-900/60 ring-1 ring-red-500/50' },
          { key: 'neutral' as const, count: neutral, label: 'Neutral', dot: 'bg-gray-500', text: 'text-gray-300', active: 'bg-gray-700/60 ring-1 ring-gray-500/50' },
        ].map(({ key, count, label, dot, text, active }) => (
          <button
            key={key}
            onClick={() => toggle(key)}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-lg transition-all ${
              filter === key ? active : 'hover:bg-gray-800'
            }`}
          >
            <div className={`w-2.5 h-2.5 rounded-full flex-shrink-0 ${dot}`} />
            <span className="text-sm text-gray-300 whitespace-nowrap">
              <span className={`font-bold ${text}`}>{count}</span> {label}
            </span>
            {filter === key && <span className="text-[10px] text-gray-500 ml-0.5">✕</span>}
          </button>
        ))}
        <div className="ml-auto text-xs text-gray-500 whitespace-nowrap">
          {filter ? `${filtered.length} of ` : ''}{(groups || []).length} pairs
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {filtered.map(g => (
          <GroupCard key={g.description} group={g} />
        ))}
      </div>

      {filtered.length === 0 && (
        <p className="text-gray-500 text-center py-10">No pairs match the selected filter.</p>
      )}

      {(!groups || groups.length === 0) && (
        <p className="text-gray-500 text-center py-10">No analysis data. Run the pipeline first.</p>
      )}
    </div>
  )
}
