import { useEffect, useState } from 'react'

// ── Types ─────────────────────────────────────────────────────────────────

type Verdict = 'BULLISH' | 'BEARISH' | 'NEUTRAL' | 'INSUFFICIENT_DATA'

interface Criterion { label: string; passed: boolean | null; detail: string; weight: number }
interface PersonaVerdict {
  id: string; name: string; style: string
  verdict: Verdict; score: number; confidence: number
  criteria: Criterion[]; reasoning: string
}
interface Incumbent { verdict: string | null; score?: number | null; confidence_label?: string | null; error?: string }
interface PanelResult {
  run_id: string; symbol: string; provider: string
  verdicts: PersonaVerdict[]
  consensus: Verdict; avg_score: number
  tally: { bullish: number; bearish: number; neutral: number; insufficient: number }
  disagreement: number
  incumbent: Incumbent | null
  fundamentals: Record<string, unknown>
  data_quality: { fields_present: number; error: string | null }
  logged_rows: number
}
interface Provider { id: string; label: string; available: boolean; note: string }
interface SourceScore {
  source: string; n: number; scored: number; hits: number
  hit_rate: number | null; avg_return: number | null
  bullish: number; bearish: number; neutral: number
}

const API = '/api/lab'

// ── Small presentational helpers ──────────────────────────────────────────

const VERDICT_STYLE: Record<string, string> = {
  BULLISH:           'bg-emerald-500/15 text-emerald-400 border-emerald-500/30',
  BEARISH:           'bg-red-500/15 text-red-400 border-red-500/30',
  NEUTRAL:           'bg-gray-500/15 text-gray-400 border-gray-600/30',
  INSUFFICIENT_DATA: 'bg-amber-500/10 text-amber-500/70 border-amber-600/20',
}

function VerdictPill({ v }: { v: string }) {
  return (
    <span className={`px-2.5 py-0.5 rounded-full border text-[10px] font-bold tracking-wider ${VERDICT_STYLE[v] ?? VERDICT_STYLE.NEUTRAL}`}>
      {v === 'INSUFFICIENT_DATA' ? 'NO DATA' : v}
    </span>
  )
}

function ScoreBar({ score }: { score: number }) {
  const pct = Math.abs(score) / 2 // -100..100 → 0..50 each side
  const pos = score >= 0
  return (
    <div className="relative w-full h-1.5 bg-gray-800 rounded-full overflow-hidden">
      <div className="absolute left-1/2 top-0 h-full w-px bg-gray-600" />
      <div
        className={`absolute top-0 h-full ${pos ? 'bg-emerald-500' : 'bg-red-500'}`}
        style={{ width: `${pct}%`, left: pos ? '50%' : `${50 - pct}%` }}
      />
    </div>
  )
}

// ── Persona card ──────────────────────────────────────────────────────────

function PersonaCard({ v }: { v: PersonaVerdict }) {
  const [open, setOpen] = useState(false)
  return (
    <div className="border border-gray-800 rounded-lg bg-gray-900/60 p-3">
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="text-white font-bold text-sm truncate">{v.name}</div>
          <div className="text-gray-500 text-[10px] truncate">{v.style}</div>
        </div>
        <VerdictPill v={v.verdict} />
      </div>

      <div className="mt-2.5">
        <ScoreBar score={v.score} />
        <div className="flex justify-between mt-1 text-[10px] font-mono text-gray-500">
          <span className={v.score >= 0 ? 'text-emerald-400' : 'text-red-400'}>
            {v.score >= 0 ? '+' : ''}{v.score}
          </span>
          <span>{v.confidence}% data</span>
        </div>
      </div>

      <button
        onClick={() => setOpen(o => !o)}
        className="mt-2 text-[10px] text-gray-500 hover:text-gray-300 transition-colors"
      >
        {open ? '▾ hide criteria' : '▸ show criteria'}
      </button>

      {open && (
        <div className="mt-2 space-y-1 border-t border-gray-800/60 pt-2">
          {v.criteria.map((c, i) => (
            <div key={i} className="flex items-center gap-2 text-[10px]">
              <span className={
                c.passed === null ? 'text-gray-700'
                  : c.passed ? 'text-emerald-500' : 'text-red-500'
              }>
                {c.passed === null ? '○' : c.passed ? '✓' : '✕'}
              </span>
              <span className={c.passed === null ? 'text-gray-600' : 'text-gray-400'}>{c.label}</span>
              <span className="ml-auto font-mono text-gray-600">{c.detail}</span>
            </div>
          ))}
          <p className="text-[10px] text-gray-500 pt-1.5 leading-relaxed">{v.reasoning}</p>
        </div>
      )}
    </div>
  )
}

// ── Page ──────────────────────────────────────────────────────────────────

export default function LabPage() {
  const [tab, setTab] = useState<'panel' | 'scorecard'>('panel')
  const [symbol, setSymbol] = useState('')
  const [provider, setProvider] = useState('yfinance')
  const [providers, setProviders] = useState<Provider[]>([])
  const [compareScanner, setCompareScanner] = useState(true)
  const [data, setData] = useState<PanelResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [scores, setScores] = useState<SourceScore[]>([])
  const [scoreMeta, setScoreMeta] = useState<{ total: number; note?: string }>({ total: 0 })
  const [scoreLoading, setScoreLoading] = useState(false)

  useEffect(() => {
    fetch(`${API}/providers`)
      .then(r => r.json())
      .then(d => { setProviders(d.providers ?? []); setProvider(d.default ?? 'yfinance') })
      .catch(() => {})
  }, [])

  async function runPanel(e: React.FormEvent) {
    e.preventDefault()
    const sym = symbol.trim().toUpperCase()
    if (!sym) return
    setLoading(true); setError(null); setData(null)
    try {
      const res = await fetch(
        `${API}/panel/${sym}?provider=${provider}&compare_scanner=${compareScanner}`
      )
      if (!res.ok) throw new Error((await res.json()).detail ?? `HTTP ${res.status}`)
      setData(await res.json())
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setLoading(false)
    }
  }

  async function loadScorecard() {
    setScoreLoading(true)
    try {
      const res = await fetch(`${API}/scorecard`)
      const d = await res.json()
      setScores(d.sources ?? [])
      setScoreMeta({ total: d.total_verdicts ?? 0, note: d.note })
    } catch { /* silent */ } finally { setScoreLoading(false) }
  }

  useEffect(() => { if (tab === 'scorecard') loadScorecard() }, [tab])

  return (
    <div className="p-6 max-w-6xl mx-auto">
      {/* Header */}
      <div className="flex items-start justify-between mb-1 flex-wrap gap-3">
        <div>
          <h1 className="text-2xl font-bold text-white tracking-tight">LAB</h1>
          <p className="text-gray-500 text-xs mt-0.5">
            Isolated experiment bench — nothing here feeds the live scanner. Hire what earns it, retire what doesn't.
          </p>
        </div>
        <span className="px-2 py-1 rounded border border-amber-600/30 bg-amber-500/10 text-amber-500 text-[10px] font-bold tracking-wider">
          EXPERIMENTAL
        </span>
      </div>

      {/* Tabs */}
      <div className="flex gap-2 mt-4 mb-5">
        {(['panel', 'scorecard'] as const).map(t => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-4 py-1.5 rounded-lg text-xs font-bold tracking-wide transition-colors ${
              tab === t ? 'bg-emerald-600 text-white' : 'bg-gray-900 text-gray-500 hover:text-gray-300'
            }`}
          >
            {t === 'panel' ? 'ANALYST PANEL' : 'SCORECARD'}
          </button>
        ))}
      </div>

      {tab === 'panel' && (
        <>
          <form onSubmit={runPanel} className="flex flex-wrap gap-2 items-center">
            <input
              value={symbol}
              onChange={e => setSymbol(e.target.value.toUpperCase())}
              placeholder="Enter ticker (e.g., AAPL, NVDA)"
              className="flex-1 min-w-[200px] px-4 py-2 bg-gray-900 border border-gray-700 rounded-lg text-white font-mono text-sm placeholder-gray-600 focus:outline-none focus:border-emerald-600"
            />
            <select
              value={provider}
              onChange={e => setProvider(e.target.value)}
              className="px-3 py-2 bg-gray-900 border border-gray-700 rounded-lg text-gray-300 text-xs font-mono focus:outline-none focus:border-emerald-600"
            >
              {providers.map(p => (
                <option key={p.id} value={p.id} disabled={!p.available}>
                  {p.label}{p.available ? '' : ' (no key)'}
                </option>
              ))}
            </select>
            <label className="flex items-center gap-1.5 text-[11px] text-gray-400 select-none cursor-pointer">
              <input
                type="checkbox"
                checked={compareScanner}
                onChange={e => setCompareScanner(e.target.checked)}
                className="accent-emerald-600"
              />
              vs. scanner
            </label>
            <button
              type="submit"
              disabled={loading}
              className="px-6 py-2 bg-emerald-600 hover:bg-emerald-500 disabled:bg-gray-700 text-white rounded-lg text-xs font-bold tracking-wide transition-colors"
            >
              {loading ? 'RUNNING…' : 'RUN PANEL'}
            </button>
          </form>

          {providers.find(p => p.id === provider && !p.available) && (
            <p className="mt-2 text-[11px] text-amber-500/80">
              {providers.find(p => p.id === provider)?.note}
            </p>
          )}

          {error && (
            <div className="mt-4 text-red-400 text-sm p-3 bg-red-500/10 border border-red-500/20 rounded-lg">
              {error}
            </div>
          )}

          {data && (
            <div className="mt-6 space-y-5">
              {/* Summary strip */}
              <div className="border border-gray-800 rounded-xl bg-gray-900/60 p-4">
                <div className="flex flex-wrap items-center gap-x-8 gap-y-3">
                  <div>
                    <div className="text-[10px] text-gray-500 uppercase tracking-wider">Symbol</div>
                    <div className="text-white font-mono font-bold text-lg">{data.symbol}</div>
                  </div>
                  <div>
                    <div className="text-[10px] text-gray-500 uppercase tracking-wider">Panel consensus</div>
                    <div className="mt-1"><VerdictPill v={data.consensus} /></div>
                  </div>
                  <div>
                    <div className="text-[10px] text-gray-500 uppercase tracking-wider">Incumbent scanner</div>
                    <div className="mt-1">
                      {data.incumbent?.verdict
                        ? <VerdictPill v={data.incumbent.verdict} />
                        : <span className="text-[10px] text-gray-600">unavailable</span>}
                    </div>
                  </div>
                  <div>
                    <div className="text-[10px] text-gray-500 uppercase tracking-wider">Tally</div>
                    <div className="text-xs font-mono mt-1">
                      <span className="text-emerald-400">{data.tally.bullish}B</span>
                      <span className="text-gray-600"> / </span>
                      <span className="text-red-400">{data.tally.bearish}Br</span>
                      <span className="text-gray-600"> / </span>
                      <span className="text-gray-400">{data.tally.neutral}N</span>
                    </div>
                  </div>
                  <div>
                    <div className="text-[10px] text-gray-500 uppercase tracking-wider">Disagreement</div>
                    <div className="text-xs font-mono text-amber-400 mt-1">{data.disagreement}</div>
                  </div>
                  <div className="ml-auto text-right">
                    <div className="text-[10px] text-gray-500 uppercase tracking-wider">Data</div>
                    <div className="text-[11px] font-mono text-gray-400 mt-1">
                      {data.data_quality.fields_present} fields · {data.provider}
                    </div>
                  </div>
                </div>
                {data.disagreement > 100 && (
                  <p className="mt-3 pt-3 border-t border-gray-800/60 text-[11px] text-amber-500/80">
                    Wide split — the styles genuinely disagree here. That's the interesting case, not a bug.
                  </p>
                )}
              </div>

              {/* Persona grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                {data.verdicts.map(v => <PersonaCard key={v.id} v={v} />)}
              </div>

              <p className="text-[10px] text-gray-600">
                Logged {data.logged_rows} verdicts under run {data.run_id} — check the Scorecard tab to mark them later.
              </p>
            </div>
          )}
        </>
      )}

      {tab === 'scorecard' && (
        <div className="space-y-4">
          <div className="flex items-center gap-3">
            <button
              onClick={loadScorecard}
              className="px-4 py-1.5 bg-gray-800 hover:bg-gray-700 text-gray-300 rounded-lg text-xs font-bold"
            >
              {scoreLoading ? 'LOADING…' : 'REFRESH'}
            </button>
            <span className="text-[11px] text-gray-500">{scoreMeta.total} verdicts logged</span>
          </div>

          {scores.length === 0 ? (
            <div className="border border-gray-800 rounded-xl bg-gray-900/40 p-8 text-center">
              <p className="text-gray-500 text-sm">No verdicts logged yet.</p>
              <p className="text-gray-600 text-xs mt-1">Run a panel to start building the record.</p>
            </div>
          ) : (
            <div className="border border-gray-800 rounded-xl bg-gray-900/60 overflow-hidden">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-gray-800 text-[10px] text-gray-500 uppercase tracking-wider">
                    <th className="text-left  px-4 py-2.5 font-medium">Source</th>
                    <th className="text-right px-3 py-2.5 font-medium">Logged</th>
                    <th className="text-right px-3 py-2.5 font-medium">Scored</th>
                    <th className="text-right px-3 py-2.5 font-medium">Hit rate</th>
                    <th className="text-right px-3 py-2.5 font-medium">Avg return</th>
                    <th className="text-right px-4 py-2.5 font-medium">B / Br / N</th>
                  </tr>
                </thead>
                <tbody>
                  {scores.map(s => {
                    const incumbent = s.source.startsWith('scanner')
                    return (
                      <tr
                        key={s.source}
                        className={`border-b border-gray-800/40 last:border-0 ${incumbent ? 'bg-blue-500/5' : ''}`}
                      >
                        <td className="px-4 py-2.5 font-mono text-gray-300">
                          {s.source}
                          {incumbent && <span className="ml-2 text-[9px] text-blue-400">INCUMBENT</span>}
                        </td>
                        <td className="px-3 py-2.5 text-right font-mono text-gray-500">{s.n}</td>
                        <td className="px-3 py-2.5 text-right font-mono text-gray-400">{s.scored}</td>
                        <td className="px-3 py-2.5 text-right font-mono">
                          {s.hit_rate === null
                            ? <span className="text-gray-700">—</span>
                            : <span className={s.hit_rate >= 50 ? 'text-emerald-400' : 'text-red-400'}>
                                {s.hit_rate}%
                              </span>}
                        </td>
                        <td className="px-3 py-2.5 text-right font-mono">
                          {s.avg_return === null
                            ? <span className="text-gray-700">—</span>
                            : <span className={s.avg_return >= 0 ? 'text-emerald-400' : 'text-red-400'}>
                                {s.avg_return >= 0 ? '+' : ''}{s.avg_return}%
                              </span>}
                        </td>
                        <td className="px-4 py-2.5 text-right font-mono text-gray-500">
                          {s.bullish}/{s.bearish}/{s.neutral}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          )}

          {scoreMeta.note && (
            <p className="text-[11px] text-gray-500 leading-relaxed">{scoreMeta.note}</p>
          )}
        </div>
      )}
    </div>
  )
}
