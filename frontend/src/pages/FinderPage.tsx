import { useEffect, useState } from 'react'

// ── Types ─────────────────────────────────────────────────────────────────

interface FinderCriteria {
  min_price?: number
  max_price?: number
  min_volume?: number
  require_golden_cross?: boolean
  require_consolidation?: boolean
  max_gap_pct?: number
  gap_lookback?: number
  min_confluence?: number
  require_breakout_ready?: boolean
  consolidation_lookback?: number
  min_compression_score?: number
}

interface PresetInfo {
  id: string
  label: string
  description: string
  criteria: FinderCriteria
}

interface Checks {
  price_band: boolean
  liquidity: boolean
  golden_cross: boolean
  consolidating: boolean
  no_gaps: boolean
  breakout_ready: boolean | null
  confluence: boolean | null
}

interface ScanResult {
  symbol: string
  price: number
  volume: number
  checks: Checks
  passed: number
  failed: number
  ma50: number
  ma200: number
  compression_score: number
  compression_duration: number
  gap_max_pct: number
  breakout_strength: number | null
  confluence: number | null
}

interface ScanResponse {
  results: ScanResult[]
  count: number
  criteria: FinderCriteria
  elapsed: number
}

interface HealthTable {
  table: string
  latest_date: string | null
  current_count: number
  total_count: number
  stale?: boolean
}

interface HealthResponse {
  tables: HealthTable[]
}

const API = '/api/finder'

const PRESET_LABELS: Record<string, string> = {
  signal:    'SIGNAL FINDER',
  pennyking: 'PENNYKING',
  maker:     'MILLIONAIRE MAKER',
}

// ── Small presentational helpers ────────────────────────────────────────────

function priceBand(c: FinderCriteria): string {
  const lo = c.min_price
  const hi = c.max_price
  if (lo == null && hi == null) return 'any price'
  if (lo == null) return `under $${hi}`
  if (hi == null) return `$${lo}+`
  return `$${lo}–${hi}`
}

function fmtNum(n: number | null | undefined, digits = 2): string {
  if (n == null || Number.isNaN(n)) return '—'
  return n.toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits })
}

function fmtVolume(n: number | null | undefined): string {
  if (n == null) return '—'
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`
  return String(n)
}

function CheckPill({ label, v }: { label: string; v: boolean | null }) {
  const style = v === null
    ? 'bg-gray-500/10 text-gray-600 border-gray-700/30'
    : v
      ? 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30'
      : 'bg-red-500/15 text-red-400 border-red-500/30'
  const mark = v === null ? '○' : v ? '✓' : '✕'
  return (
    <span className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded border text-[9px] font-bold tracking-wide ${style}`}>
      {mark} {label}
    </span>
  )
}

const CHECK_LABELS: { key: keyof Checks; label: string }[] = [
  { key: 'price_band',     label: 'PRICE' },
  { key: 'liquidity',      label: 'LIQ' },
  { key: 'golden_cross',   label: 'GLDN' },
  { key: 'consolidating',  label: 'CONSOL' },
  { key: 'no_gaps',        label: 'GAPS' },
  { key: 'breakout_ready', label: 'BRKOUT' },
  { key: 'confluence',     label: 'CONFL' },
]

function NumberField({ label, value, onChange }: { label: string; value: number | undefined; onChange: (v: number | undefined) => void }) {
  return (
    <label className="flex flex-col gap-1">
      <span className="text-[9px] font-bold text-gray-500 tracking-wider uppercase">{label}</span>
      <input
        type="number"
        value={value ?? ''}
        onChange={e => onChange(e.target.value === '' ? undefined : Number(e.target.value))}
        className="px-2.5 py-1.5 bg-gray-800 border border-gray-700 rounded text-gray-200 text-xs font-mono focus:outline-none focus:border-emerald-600"
      />
    </label>
  )
}

function BoolField({ label, value, onChange }: { label: string; value: boolean | undefined; onChange: (v: boolean) => void }) {
  return (
    <label className="flex items-center gap-2 text-[11px] text-gray-400 select-none cursor-pointer mt-4">
      <input
        type="checkbox"
        checked={!!value}
        onChange={e => onChange(e.target.checked)}
        className="accent-emerald-600"
      />
      {label}
    </label>
  )
}

function Th({ label, onClick, align = 'left' }: { label: string; onClick?: () => void; align?: 'left' | 'right' }) {
  return (
    <th
      className={`px-3 py-2.5 font-medium ${align === 'right' ? 'text-right' : 'text-left'} ${onClick ? 'cursor-pointer hover:text-gray-300' : ''}`}
      onClick={onClick}
    >
      {label}
    </th>
  )
}

// ── Page ──────────────────────────────────────────────────────────────────

export default function FinderPage() {
  const [presets, setPresets] = useState<PresetInfo[]>([])
  const [activePreset, setActivePreset] = useState<string>('signal')
  const [criteria, setCriteria] = useState<FinderCriteria>({})
  const [panelOpen, setPanelOpen] = useState(false)

  const [health, setHealth] = useState<HealthTable[]>([])
  const [healthError, setHealthError] = useState<string | null>(null)

  const [results, setResults] = useState<ScanResult[] | null>(null)
  const [scanMeta, setScanMeta] = useState<{ count: number; elapsed: number } | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [sortKey, setSortKey] = useState<keyof ScanResult | null>(null)
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc')

  useEffect(() => {
    fetch(`${API}/presets`)
      .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json() })
      .then(d => {
        const p: PresetInfo[] = d.presets ?? []
        setPresets(p)
        if (p.length) {
          setActivePreset(p[0].id)
          setCriteria(p[0].criteria ?? {})
        }
      })
      .catch(err => setError(err instanceof Error ? err.message : String(err)))
  }, [])

  useEffect(() => {
    fetch(`${API}/health`)
      .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json() })
      .then((d: HealthResponse) => setHealth(d.tables ?? []))
      .catch(err => setHealthError(err instanceof Error ? err.message : String(err)))
  }, [])

  function selectPreset(p: PresetInfo) {
    setActivePreset(p.id)
    setCriteria(p.criteria ?? {})
    setResults(null)
    setScanMeta(null)
  }

  async function runScan() {
    setLoading(true); setError(null)
    try {
      const params = new URLSearchParams()
      params.set('preset', activePreset)
      params.set('limit', '100')
      Object.entries(criteria).forEach(([k, v]) => {
        if (v !== undefined && v !== null && String(v) !== '') params.set(k, String(v))
      })
      const res = await fetch(`${API}/scan?${params.toString()}`)
      if (!res.ok) {
        const body = await res.json().catch(() => null)
        throw new Error(body?.detail ?? `HTTP ${res.status}`)
      }
      const d: ScanResponse = await res.json()
      setResults(d.results ?? [])
      setScanMeta({ count: d.count ?? (d.results?.length ?? 0), elapsed: d.elapsed ?? 0 })
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      setResults(null)
      setScanMeta(null)
    } finally {
      setLoading(false)
    }
  }

  function updateCriterion<K extends keyof FinderCriteria>(key: K, value: FinderCriteria[K]) {
    setCriteria(c => ({ ...c, [key]: value }))
  }

  function toggleSort(key: keyof ScanResult) {
    if (sortKey === key) {
      setSortDir(d => d === 'asc' ? 'desc' : 'asc')
    } else {
      setSortKey(key); setSortDir('desc')
    }
  }

  const sortedResults = (() => {
    if (!results) return []
    if (!sortKey) return results
    const arr = [...results]
    arr.sort((a, b) => {
      const av = a[sortKey]; const bv = b[sortKey]
      if (av == null && bv == null) return 0
      if (av == null) return 1
      if (bv == null) return -1
      if (typeof av === 'number' && typeof bv === 'number') return sortDir === 'asc' ? av - bv : bv - av
      return sortDir === 'asc' ? String(av).localeCompare(String(bv)) : String(bv).localeCompare(String(av))
    })
    return arr
  })()

  const staleTables = health.filter(t => {
    if (t.stale) return true
    if (t.total_count > 0) return t.current_count / t.total_count < 0.9
    return false
  })
  const hasAnyHealthInfo = health.length > 0
  const activePresetInfo = presets.find(p => p.id === activePreset)

  return (
    <div className="p-6 max-w-6xl mx-auto">
      {/* Header */}
      <div className="mb-1">
        <h1 className="text-2xl font-bold text-white tracking-tight">FINDER</h1>
        <p className="text-gray-500 text-xs mt-0.5">
          Scan the universe against preset setups — pass/fail is shown per-check, not just a final score.
        </p>
      </div>

      {/* Freshness banner */}
      {staleTables.length > 0 && (
        <div className="mt-4 border border-amber-600/40 bg-amber-500/10 rounded-lg p-3">
          <div className="flex items-start gap-2">
            <span className="text-amber-500 text-sm">⚠</span>
            <div className="flex-1">
              <div className="text-amber-400 text-xs font-bold tracking-wider">DATA MAY BE STALE</div>
              <div className="mt-1 space-y-0.5">
                {staleTables.map(t => (
                  <div key={t.table} className="text-[11px] text-amber-500/80 font-mono">
                    {t.table}: latest {t.latest_date ?? 'unknown'} · {t.current_count}/{t.total_count} instruments current
                  </div>
                ))}
              </div>
              <p className="mt-1.5 text-[10px] text-amber-500/60">
                An empty result below may mean the underlying table has no current data — not that nothing matched.
              </p>
            </div>
          </div>
        </div>
      )}
      {healthError && (
        <div className="mt-4 text-[11px] text-gray-600">
          Freshness check unavailable ({healthError}) — results below cannot be cross-checked against data currency.
        </div>
      )}

      {/* Preset tabs */}
      <div className="flex flex-wrap gap-2 mt-4">
        {presets.map(p => (
          <button
            key={p.id}
            onClick={() => selectPreset(p)}
            className={`px-4 py-2 rounded-lg text-left transition-colors ${
              activePreset === p.id ? 'bg-emerald-600 text-white' : 'bg-gray-900 text-gray-400 hover:text-gray-200 border border-gray-800'
            }`}
          >
            <div className="text-xs font-bold tracking-wide">{PRESET_LABELS[p.id] ?? p.label}</div>
            <div className={`text-[10px] mt-0.5 ${activePreset === p.id ? 'text-emerald-100' : 'text-gray-600'}`}>
              {priceBand(p.criteria)}
            </div>
          </button>
        ))}
        {presets.length === 0 && !error && (
          <div className="text-[11px] text-gray-600">Loading presets…</div>
        )}
      </div>
      {activePresetInfo?.description && (
        <p className="mt-2 text-[11px] text-gray-500">{activePresetInfo.description}</p>
      )}

      {/* Criteria panel */}
      <div className="mt-4 border border-gray-800 rounded-lg bg-gray-900/60">
        <button
          onClick={() => setPanelOpen(o => !o)}
          className="w-full flex items-center justify-between px-4 py-2.5 text-left"
        >
          <span className="text-[11px] font-bold text-gray-300 tracking-wider">CRITERIA</span>
          <span className="text-gray-500 text-[10px]">{panelOpen ? '▾ hide' : '▸ show'}</span>
        </button>

        {panelOpen && (
          <div className="px-4 pb-4 grid grid-cols-2 md:grid-cols-3 gap-3 border-t border-gray-800/60 pt-3">
            <NumberField label="Min price" value={criteria.min_price} onChange={v => updateCriterion('min_price', v)} />
            <NumberField label="Max price" value={criteria.max_price} onChange={v => updateCriterion('max_price', v)} />
            <NumberField label="Min volume" value={criteria.min_volume} onChange={v => updateCriterion('min_volume', v)} />
            <NumberField label="Max gap %" value={criteria.max_gap_pct} onChange={v => updateCriterion('max_gap_pct', v)} />
            <NumberField label="Gap lookback" value={criteria.gap_lookback} onChange={v => updateCriterion('gap_lookback', v)} />
            <NumberField label="Min confluence" value={criteria.min_confluence} onChange={v => updateCriterion('min_confluence', v)} />
            <NumberField label="Consolidation lookback" value={criteria.consolidation_lookback} onChange={v => updateCriterion('consolidation_lookback', v)} />
            <NumberField label="Min compression score" value={criteria.min_compression_score} onChange={v => updateCriterion('min_compression_score', v)} />
            <BoolField label="Require golden cross" value={criteria.require_golden_cross} onChange={v => updateCriterion('require_golden_cross', v)} />
            <BoolField label="Require consolidation" value={criteria.require_consolidation} onChange={v => updateCriterion('require_consolidation', v)} />
            <BoolField label="Require breakout ready" value={criteria.require_breakout_ready} onChange={v => updateCriterion('require_breakout_ready', v)} />
          </div>
        )}

        <div className="px-4 pb-4">
          <button
            onClick={runScan}
            disabled={loading}
            className="px-6 py-2 bg-emerald-600 hover:bg-emerald-500 disabled:bg-gray-700 text-white rounded-lg text-xs font-bold tracking-wide transition-colors"
          >
            {loading ? 'SCANNING…' : 'RUN SCAN'}
          </button>
          {scanMeta && (
            <span className="ml-3 text-[11px] text-gray-500">
              {scanMeta.count} matches · {scanMeta.elapsed.toFixed(2)}s
            </span>
          )}
        </div>
      </div>

      {error && (
        <div className="mt-4 text-red-400 text-sm p-3 bg-red-500/10 border border-red-500/20 rounded-lg">
          {error}
        </div>
      )}

      {/* Results */}
      {results !== null && !error && (
        results.length === 0 ? (
          <div className="mt-6 border border-gray-800 rounded-xl bg-gray-900/40 p-8 text-center">
            {staleTables.length > 0 ? (
              <>
                <p className="text-amber-500 text-sm font-bold">No current data, not &ldquo;no matches&rdquo;</p>
                <p className="text-gray-500 text-xs mt-1">
                  {staleTables.map(t => t.table).join(', ')} {staleTables.length === 1 ? 'is' : 'are'} stale — the scan
                  ran against whatever data exists, but coverage is incomplete. Check the freshness banner above.
                </p>
              </>
            ) : hasAnyHealthInfo ? (
              <>
                <p className="text-gray-400 text-sm">Scan ran, nothing matched your criteria.</p>
                <p className="text-gray-600 text-xs mt-1">Data is current — try loosening the criteria panel above.</p>
              </>
            ) : (
              <>
                <p className="text-gray-400 text-sm">Scan ran, nothing matched your criteria.</p>
                <p className="text-gray-600 text-xs mt-1">Freshness could not be confirmed — an empty result may also mean stale data.</p>
              </>
            )}
          </div>
        ) : (
          <div className="mt-6 border border-gray-800 rounded-xl bg-gray-900/60 overflow-hidden overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-gray-800 text-[10px] text-gray-500 uppercase tracking-wider">
                  <Th label="Symbol" onClick={() => toggleSort('symbol')} />
                  <Th label="Price" onClick={() => toggleSort('price')} align="right" />
                  <Th label="Volume" onClick={() => toggleSort('volume')} align="right" />
                  <Th label="Compression" onClick={() => toggleSort('compression_score')} align="right" />
                  <Th label="MA50 / MA200" align="right" />
                  <Th label="Max gap %" onClick={() => toggleSort('gap_max_pct')} align="right" />
                  <Th label="Breakout" onClick={() => toggleSort('breakout_strength')} align="right" />
                  <Th label="Checks" />
                </tr>
              </thead>
              <tbody>
                {sortedResults.map(r => {
                  const golden = r.ma50 >= r.ma200
                  return (
                    <tr key={r.symbol} className="border-b border-gray-800/40 last:border-0 hover:bg-gray-800/30">
                      <td className="px-4 py-2.5 font-mono font-bold text-white">{r.symbol}</td>
                      <td className="px-3 py-2.5 text-right font-mono text-gray-300">${fmtNum(r.price)}</td>
                      <td className="px-3 py-2.5 text-right font-mono text-gray-400">{fmtVolume(r.volume)}</td>
                      <td className="px-3 py-2.5 text-right font-mono text-gray-300">
                        {fmtNum(r.compression_score, 1)}
                        <span className="text-gray-600"> / {r.compression_duration}d</span>
                      </td>
                      <td className="px-3 py-2.5 text-right font-mono">
                        <span className="text-gray-400">{fmtNum(r.ma50)}</span>
                        <span className={golden ? 'text-emerald-400' : 'text-red-400'}> {golden ? '▲' : '▼'} </span>
                        <span className="text-gray-500">{fmtNum(r.ma200)}</span>
                      </td>
                      <td className="px-3 py-2.5 text-right font-mono text-gray-400">{fmtNum(r.gap_max_pct, 1)}%</td>
                      <td className="px-3 py-2.5 text-right font-mono">
                        {r.breakout_strength === null
                          ? <span className="text-gray-700">—</span>
                          : <span className="text-gray-300">{fmtNum(r.breakout_strength, 1)}</span>}
                      </td>
                      <td className="px-4 py-2.5">
                        <div className="flex flex-wrap gap-1 max-w-[280px]">
                          {CHECK_LABELS.map(c => (
                            <CheckPill key={c.key} label={c.label} v={r.checks[c.key]} />
                          ))}
                        </div>
                        <div className="text-[9px] text-gray-600 mt-1">{r.passed} passed / {r.failed} failed</div>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )
      )}
    </div>
  )
}
