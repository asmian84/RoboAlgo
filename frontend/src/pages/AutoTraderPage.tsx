import { useEffect, useState, useCallback } from 'react'

// ── Types ─────────────────────────────────────────────────────────────────

interface KillMeta {
  reason?: string | null
  actor?: string | null
  at?: string | null
}

interface StatusData {
  enabled: boolean
  running: boolean
  kill_switch: boolean
  kill_meta: KillMeta | null
  mode: string
  broker: string | null
  is_paper: boolean | null
  broker_error: string | null
  redis_available: boolean
  lock: boolean | string | null
  last_cycle: string | null
}

interface AutoTraderConfig {
  risk: Record<string, unknown>
  [key: string]: unknown
}

interface Cycle {
  cycle_id: string | number
  time?: string | null
  started_at?: string | null
  status?: string
  mode?: string
  scanned?: number
  candidates?: number
  placed?: number
  rejected?: number
  equity_delta?: number | null
  halt_reason?: string | null
}

interface Decision {
  id?: string | number
  cycle_id?: string | number
  symbol?: string
  action?: string
  qty?: number | null
  entry?: number | null
  stop?: number | null
  target?: number | null
  governor_rule?: string | null
  reason?: string | null
}

interface Discrepancy {
  kind?: string
  symbol?: string
  expected?: string | number
  actual?: string | number
  resolution?: string
}

const API = '/api/auto-trader'
const TOKEN_KEY = 'roboalgo_control_token'
const ACTIONS = ['', 'PLACE', 'REJECT', 'SKIP', 'DRY_RUN', 'HALT', 'NO_ACTION'] as const

// ── localStorage helpers (private-window safe) ──────────────────────────────

function loadToken(): string {
  try { return localStorage.getItem(TOKEN_KEY) ?? '' } catch { return '' }
}
function persistToken(v: string) {
  try { localStorage.setItem(TOKEN_KEY, v) } catch { /* private window etc — non-fatal */ }
}

// ── Guarded fetch — distinguishes "not configured" (503) from "wrong token" (403) ──

async function guarded(path: string, token: string, init: RequestInit = {}) {
  try {
    const res = await fetch(`${API}${path}`, {
      ...init,
      headers: { 'Content-Type': 'application/json', ...(init.headers ?? {}), 'X-RoboAlgo-Control-Token': token },
    })
    if (res.status === 503) {
      return { ok: false as const, error: 'Control token not configured on the server (missing env var) — guarded actions are disabled until an operator sets it.' }
    }
    if (res.status === 403) {
      return { ok: false as const, error: 'Wrong control token — check the value and try again.' }
    }
    if (!res.ok) {
      const body = await res.json().catch(() => null)
      return { ok: false as const, error: body?.detail ?? `HTTP ${res.status}` }
    }
    const data = await res.json().catch(() => null)
    return { ok: true as const, data }
  } catch (err) {
    return { ok: false as const, error: err instanceof Error ? err.message : String(err) }
  }
}

// ── Small presentational helpers ─────────────────────────────────────────────

function timeAgo(iso: string | null | undefined): string {
  if (!iso) return '—'
  const t = new Date(iso).getTime()
  if (Number.isNaN(t)) return '—'
  const secs = Math.max(0, Math.floor((Date.now() - t) / 1000))
  if (secs < 60) return `${secs}s ago`
  if (secs < 3600) return `${Math.floor(secs / 60)}m ago`
  if (secs < 86400) return `${Math.floor(secs / 3600)}h ago`
  return `${Math.floor(secs / 86400)}d ago`
}

function fmtNum(n: number | null | undefined, digits = 2): string {
  if (n === null || n === undefined) return '—'
  return n.toFixed(digits)
}

const ACTION_STYLE: Record<string, string> = {
  PLACE:     'bg-emerald-500/15 text-emerald-400 border-emerald-500/30',
  REJECT:    'bg-amber-500/15 text-amber-400 border-amber-500/30',
  SKIP:      'bg-gray-500/15 text-gray-400 border-gray-600/30',
  DRY_RUN:   'bg-blue-500/15 text-blue-400 border-blue-500/30',
  HALT:      'bg-red-500/15 text-red-400 border-red-500/30',
  NO_ACTION: 'bg-gray-800/40 text-gray-600 border-gray-700/40',
}

function ActionPill({ a }: { a: string }) {
  return (
    <span className={`px-2 py-0.5 rounded-full border text-[10px] font-bold tracking-wider ${ACTION_STYLE[a] ?? ACTION_STYLE.NO_ACTION}`}>
      {a}
    </span>
  )
}

// ── Page ──────────────────────────────────────────────────────────────────

export default function AutoTraderPage() {
  const [token, setToken] = useState<string>(loadToken)

  const [status, setStatus] = useState<StatusData | null>(null)
  const [statusError, setStatusError] = useState<string | null>(null)

  const [config, setConfig] = useState<AutoTraderConfig | null>(null)
  const [riskEdits, setRiskEdits] = useState<Record<string, unknown>>({})
  const [configMsg, setConfigMsg] = useState<{ ok: boolean; text: string } | null>(null)
  const [savingConfig, setSavingConfig] = useState(false)

  const [killReason, setKillReason] = useState('')
  const [actionMsg, setActionMsg] = useState<{ ok: boolean; text: string } | null>(null)
  const [busy, setBusy] = useState<string | null>(null)

  const [cycles, setCycles] = useState<Cycle[]>([])
  const [cyclesLoading, setCyclesLoading] = useState(false)
  const [selectedCycle, setSelectedCycle] = useState<string | number | null>(null)

  const [decisions, setDecisions] = useState<Decision[]>([])
  const [decisionsLoading, setDecisionsLoading] = useState(false)
  const [actionFilter, setActionFilter] = useState<string>('')

  const [discrepancies, setDiscrepancies] = useState<Discrepancy[]>([])
  const [discLoading, setDiscLoading] = useState(false)

  // Persist control token
  useEffect(() => { persistToken(token) }, [token])

  // ── Status polling (every 5s, cleaned up on unmount) ──
  const fetchStatus = useCallback(() => {
    fetch(`${API}/status`)
      .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json() })
      .then(d => { setStatus(d); setStatusError(null) })
      .catch(err => setStatusError(err instanceof Error ? err.message : String(err)))
  }, [])

  useEffect(() => {
    fetchStatus()
    const id = setInterval(fetchStatus, 5000)
    return () => clearInterval(id)
  }, [fetchStatus])

  // ── Config / risk limits ──
  const loadConfig = useCallback(() => {
    fetch(`${API}/config`)
      .then(r => r.json())
      .then(d => { setConfig(d); setRiskEdits(d?.risk ?? {}) })
      .catch(() => {})
  }, [])

  useEffect(() => { loadConfig() }, [loadConfig])

  async function saveRisk() {
    setSavingConfig(true); setConfigMsg(null)
    const res = await guarded('/config', token, { method: 'POST', body: JSON.stringify({ risk: riskEdits }) })
    if (res.ok) { setConfigMsg({ ok: true, text: 'Saved.' }); loadConfig() }
    else setConfigMsg({ ok: false, text: res.error })
    setSavingConfig(false)
  }

  // ── Kill / start / stop ──
  async function doKillOn() {
    setBusy('kill'); setActionMsg(null)
    try {
      const res = await fetch(`${API}/kill`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ on: true, reason: killReason.trim() || 'Manual kill from Auto Trader UI' }),
      })
      if (!res.ok) { const b = await res.json().catch(() => null); throw new Error(b?.detail ?? `HTTP ${res.status}`) }
      setActionMsg({ ok: true, text: 'Kill switch engaged.' })
      fetchStatus()
    } catch (err) {
      setActionMsg({ ok: false, text: err instanceof Error ? err.message : String(err) })
    } finally {
      setBusy(null)
    }
  }

  async function doKillOff() {
    setBusy('unkill'); setActionMsg(null)
    const res = await guarded('/kill', token, { method: 'POST', body: JSON.stringify({ on: false, reason: 'Cleared from Auto Trader UI' }) })
    setActionMsg(res.ok ? { ok: true, text: 'Kill switch cleared.' } : { ok: false, text: res.error })
    if (res.ok) fetchStatus()
    setBusy(null)
  }

  async function doStart() {
    setBusy('start'); setActionMsg(null)
    const res = await guarded('/start', token, { method: 'POST' })
    setActionMsg(res.ok ? { ok: true, text: 'Started.' } : { ok: false, text: res.error })
    if (res.ok) fetchStatus()
    setBusy(null)
  }

  async function doStop() {
    setBusy('stop'); setActionMsg(null)
    try {
      const res = await fetch(`${API}/stop`, { method: 'POST' })
      if (!res.ok) { const b = await res.json().catch(() => null); throw new Error(b?.detail ?? `HTTP ${res.status}`) }
      setActionMsg({ ok: true, text: 'Stopped.' })
      fetchStatus()
    } catch (err) {
      setActionMsg({ ok: false, text: err instanceof Error ? err.message : String(err) })
    } finally {
      setBusy(null)
    }
  }

  // ── Cycles ──
  const loadCycles = useCallback(() => {
    setCyclesLoading(true)
    fetch(`${API}/cycles?limit=50`)
      .then(r => r.json())
      .then(d => setCycles(d?.cycles ?? (Array.isArray(d) ? d : [])))
      .catch(() => {})
      .finally(() => setCyclesLoading(false))
  }, [])

  useEffect(() => { loadCycles() }, [loadCycles])

  // ── Decisions — global feed (filterable) or a single cycle's ──
  const loadGlobalDecisions = useCallback(() => {
    setDecisionsLoading(true); setSelectedCycle(null)
    const qs = new URLSearchParams({ limit: '200' })
    if (actionFilter) qs.set('action', actionFilter)
    fetch(`${API}/decisions?${qs.toString()}`)
      .then(r => r.json())
      .then(d => setDecisions(d?.decisions ?? (Array.isArray(d) ? d : [])))
      .catch(() => {})
      .finally(() => setDecisionsLoading(false))
  }, [actionFilter])

  useEffect(() => { loadGlobalDecisions() }, [loadGlobalDecisions])

  function loadCycleDecisions(id: string | number) {
    setDecisionsLoading(true); setSelectedCycle(id)
    fetch(`${API}/cycles/${id}/decisions`)
      .then(r => r.json())
      .then(d => setDecisions(d?.decisions ?? (Array.isArray(d) ? d : [])))
      .catch(() => {})
      .finally(() => setDecisionsLoading(false))
  }

  // ── Discrepancies ──
  const loadDiscrepancies = useCallback(() => {
    setDiscLoading(true)
    fetch(`${API}/discrepancies?limit=100`)
      .then(r => r.json())
      .then(d => setDiscrepancies(d?.discrepancies ?? (Array.isArray(d) ? d : [])))
      .catch(() => {})
      .finally(() => setDiscLoading(false))
  }, [])

  useEffect(() => { loadDiscrepancies() }, [loadDiscrepancies])

  // When viewing a single cycle's decisions (loaded unfiltered from the API), apply the
  // action filter client-side; the global feed is already filtered server-side.
  const filteredDecisions = selectedCycle !== null && actionFilter
    ? decisions.filter(d => d.action === actionFilter)
    : decisions

  const isLive = status ? (status.mode?.toLowerCase().includes('live') || status.is_paper === false) : false

  return (
    <div className="p-6 max-w-7xl mx-auto pb-10">
      {/* Header */}
      <div className="flex items-start justify-between mb-1 flex-wrap gap-3">
        <div>
          <h1 className="text-2xl font-bold text-white tracking-tight">AUTO TRADER</h1>
          <p className="text-gray-500 text-xs mt-0.5">
            Autonomous execution loop — status, kill switch, risk limits, and the full decision trail.
          </p>
        </div>
        {status && (
          <span className={`px-3 py-1.5 rounded-lg border text-xs font-black tracking-widest ${
            isLive
              ? 'bg-red-600 text-white border-red-500 animate-pulse'
              : 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30'
          }`}>
            {isLive ? 'LIVE' : 'PAPER'}
          </span>
        )}
      </div>

      {statusError && (
        <div className="mt-3 text-red-400 text-xs p-2.5 bg-red-500/10 border border-red-500/20 rounded-lg">
          Status poll failing: {statusError}
        </div>
      )}

      {/* Redis degraded banner */}
      {status && !status.redis_available && (
        <div className="mt-3 px-4 py-2.5 bg-amber-500/10 border border-amber-500/30 rounded-lg text-amber-400 text-xs font-bold tracking-wide">
          ⚠ DEGRADED — Redis unavailable. Running single-host; the loop force-downgrades to dry-run until Redis is back.
        </div>
      )}

      {/* Kill switch banner */}
      {status?.kill_switch && (
        <div className="mt-3 px-4 py-2.5 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400 text-xs">
          <span className="font-black tracking-wide">KILL SWITCH ENGAGED</span>
          {' — '}
          <span className="text-red-300">{status.kill_meta?.reason ?? 'no reason given'}</span>
          {status.kill_meta?.actor && <span className="text-red-400/70"> · by {status.kill_meta.actor}</span>}
          {status.kill_meta?.at && <span className="text-red-400/50"> · {timeAgo(status.kill_meta.at)}</span>}
        </div>
      )}

      {/* ── Status bar ── */}
      <div className="mt-4 border border-gray-800 rounded-xl bg-gray-900/60 p-4">
        <div className="flex flex-wrap items-center gap-x-8 gap-y-3">
          <div>
            <div className="text-[10px] text-gray-500 uppercase tracking-wider">Loop</div>
            <div className="mt-1">
              <span className={`px-2.5 py-0.5 rounded-full border text-[10px] font-bold tracking-wider ${
                status?.running ? 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30' : 'bg-gray-500/15 text-gray-400 border-gray-600/30'
              }`}>
                {status?.running ? 'RUNNING' : 'STOPPED'}
              </span>
            </div>
          </div>
          <div>
            <div className="text-[10px] text-gray-500 uppercase tracking-wider">Mode</div>
            <div className="mt-1 font-mono text-xs text-gray-300">{status?.mode ?? '—'}</div>
          </div>
          <div>
            <div className="text-[10px] text-gray-500 uppercase tracking-wider">Broker</div>
            <div className="mt-1 font-mono text-xs text-gray-300">
              {status?.broker ?? '—'}
              {status?.is_paper === true && <span className="ml-1.5 text-[9px] text-emerald-400">PAPER ACCT</span>}
              {status?.is_paper === false && <span className="ml-1.5 text-[9px] text-red-400 font-bold">LIVE ACCT</span>}
            </div>
            {status?.is_paper === null && status?.broker_error && (
              <div className="text-[10px] text-amber-500/70 mt-0.5 max-w-[220px]">account type unknown — {status.broker_error}</div>
            )}
          </div>
          <div>
            <div className="text-[10px] text-gray-500 uppercase tracking-wider">Redis</div>
            <div className="mt-1">
              <span className={`px-2.5 py-0.5 rounded-full border text-[10px] font-bold tracking-wider ${
                status?.redis_available ? 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30' : 'bg-red-500/15 text-red-400 border-red-500/30'
              }`}>
                {status?.redis_available ? 'UP' : 'DOWN'}
              </span>
            </div>
          </div>
          <div>
            <div className="text-[10px] text-gray-500 uppercase tracking-wider">Lock</div>
            <div className="mt-1 font-mono text-xs text-gray-400">{status?.lock ? String(status.lock) : 'free'}</div>
          </div>
          <div className="ml-auto text-right">
            <div className="text-[10px] text-gray-500 uppercase tracking-wider">Last cycle</div>
            <div className="text-xs font-mono text-gray-400 mt-1">{timeAgo(status?.last_cycle)}</div>
          </div>
        </div>
      </div>

      {/* ── Kill / Start / Stop ── */}
      <div className="mt-4 border border-gray-800 rounded-xl bg-gray-900/60 p-4">
        <div className="flex flex-wrap items-end gap-3">
          <div className="flex-1 min-w-[220px]">
            <label className="text-[10px] text-gray-500 uppercase tracking-wider">Kill reason (optional)</label>
            <input
              value={killReason}
              onChange={e => setKillReason(e.target.value)}
              placeholder="e.g. broker outage, manual halt"
              className="mt-1 w-full px-3 py-2 bg-gray-950 border border-gray-700 rounded-lg text-white text-xs font-mono placeholder-gray-600 focus:outline-none focus:border-red-600"
            />
          </div>
          <button
            onClick={doKillOn}
            disabled={busy === 'kill'}
            className="px-6 py-2.5 bg-red-600 hover:bg-red-500 disabled:bg-red-900 text-white rounded-lg text-sm font-black tracking-widest shadow-lg shadow-red-900/40 transition-colors"
          >
            {busy === 'kill' ? 'KILLING…' : '⛔ KILL'}
          </button>
          <button
            onClick={doKillOff}
            disabled={busy === 'unkill' || !status?.kill_switch}
            title="Requires control token"
            className="px-4 py-2 bg-gray-800 hover:bg-gray-700 disabled:bg-gray-900 disabled:text-gray-700 text-gray-300 rounded-lg text-[11px] font-bold tracking-wide border border-gray-700 transition-colors"
          >
            {busy === 'unkill' ? 'CLEARING…' : 'CLEAR KILL'}
          </button>
          <div className="flex-1" />
          <button
            onClick={doStart}
            disabled={busy === 'start' || !!status?.running}
            title="Requires control token"
            className="px-5 py-2 bg-emerald-600 hover:bg-emerald-500 disabled:bg-gray-700 text-white rounded-lg text-xs font-bold tracking-wide transition-colors"
          >
            {busy === 'start' ? 'STARTING…' : 'START'}
          </button>
          <button
            onClick={doStop}
            disabled={busy === 'stop' || !status?.running}
            className="px-5 py-2 bg-gray-800 hover:bg-gray-700 disabled:bg-gray-900 disabled:text-gray-700 text-gray-300 rounded-lg text-xs font-bold tracking-wide border border-gray-700 transition-colors"
          >
            {busy === 'stop' ? 'STOPPING…' : 'STOP'}
          </button>
        </div>

        {actionMsg && (
          <p className={`mt-3 text-[11px] ${actionMsg.ok ? 'text-emerald-400' : 'text-red-400'}`}>{actionMsg.text}</p>
        )}

        {/* Control token */}
        <div className="mt-4 pt-3 border-t border-gray-800/60">
          <label className="text-[10px] text-gray-500 uppercase tracking-wider">Control token (for guarded actions)</label>
          <input
            type="password"
            value={token}
            onChange={e => setToken(e.target.value)}
            placeholder="X-RoboAlgo-Control-Token"
            className="mt-1 w-full max-w-sm px-3 py-2 bg-gray-950 border border-gray-700 rounded-lg text-white text-xs font-mono placeholder-gray-600 focus:outline-none focus:border-emerald-600"
          />
          <p className="mt-1 text-[10px] text-gray-600">
            Stored locally in this browser only. Guarded endpoints return 503 if the server has no token configured, 403 if this one is wrong.
          </p>
        </div>
      </div>

      {/* ── Risk limits ── */}
      <div className="mt-4 border border-gray-800 rounded-xl bg-gray-900/60 p-4">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-xs font-bold tracking-wider text-gray-300 uppercase">Risk limits</h2>
          <button
            onClick={saveRisk}
            disabled={savingConfig}
            className="px-4 py-1.5 bg-emerald-600 hover:bg-emerald-500 disabled:bg-gray-700 text-white rounded-lg text-[11px] font-bold tracking-wide transition-colors"
          >
            {savingConfig ? 'SAVING…' : 'SAVE'}
          </button>
        </div>

        {!config ? (
          <p className="text-gray-600 text-xs">Loading config…</p>
        ) : Object.keys(riskEdits).length === 0 ? (
          <p className="text-gray-600 text-xs">No risk config found.</p>
        ) : (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {Object.entries(riskEdits).map(([key, value]) => {
              const isBool = typeof value === 'boolean'
              const isNum  = typeof value === 'number'
              return (
                <div key={key}>
                  <label className="text-[9px] text-gray-500 uppercase tracking-wider truncate block">{key}</label>
                  {isBool ? (
                    <label className="mt-1 flex items-center gap-1.5 text-[11px] text-gray-400 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={value as boolean}
                        onChange={e => setRiskEdits(r => ({ ...r, [key]: e.target.checked }))}
                        className="accent-emerald-600"
                      />
                      {(value as boolean) ? 'true' : 'false'}
                    </label>
                  ) : (
                    <input
                      type={isNum ? 'number' : 'text'}
                      value={value == null ? '' : String(value)}
                      onChange={e => setRiskEdits(r => ({ ...r, [key]: isNum ? Number(e.target.value) : e.target.value }))}
                      className="mt-1 w-full px-2.5 py-1.5 bg-gray-950 border border-gray-700 rounded-lg text-gray-200 text-xs font-mono focus:outline-none focus:border-emerald-600"
                    />
                  )}
                </div>
              )
            })}
          </div>
        )}

        {configMsg && (
          <p className={`mt-3 text-[11px] ${configMsg.ok ? 'text-emerald-400' : 'text-red-400'}`}>{configMsg.text}</p>
        )}
      </div>

      {/* ── Cycles ── */}
      <div className="mt-4 border border-gray-800 rounded-xl bg-gray-900/60 overflow-hidden">
        <div className="flex items-center justify-between px-4 py-2.5 border-b border-gray-800">
          <h2 className="text-xs font-bold tracking-wider text-gray-300 uppercase">Cycles</h2>
          <button onClick={loadCycles} className="text-[10px] text-gray-500 hover:text-gray-300 transition-colors">
            {cyclesLoading ? 'LOADING…' : '↻ REFRESH'}
          </button>
        </div>
        {cycles.length === 0 ? (
          <p className="text-gray-600 text-xs p-4">No cycles recorded yet.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-gray-800 text-[10px] text-gray-500 uppercase tracking-wider">
                  <th className="text-left  px-4 py-2 font-medium">Time</th>
                  <th className="text-left  px-3 py-2 font-medium">Status</th>
                  <th className="text-left  px-3 py-2 font-medium">Mode</th>
                  <th className="text-right px-3 py-2 font-medium">Scanned</th>
                  <th className="text-right px-3 py-2 font-medium">Candidates</th>
                  <th className="text-right px-3 py-2 font-medium">Placed</th>
                  <th className="text-right px-3 py-2 font-medium">Rejected</th>
                  <th className="text-right px-3 py-2 font-medium">Equity Δ</th>
                  <th className="text-left  px-4 py-2 font-medium">Halt reason</th>
                </tr>
              </thead>
              <tbody>
                {cycles.map(c => {
                  const halted = (c.status ?? '').toLowerCase() === 'halted'
                  const selected = selectedCycle === c.cycle_id
                  return (
                    <tr
                      key={c.cycle_id}
                      onClick={() => loadCycleDecisions(c.cycle_id)}
                      className={`border-b border-gray-800/40 last:border-0 cursor-pointer hover:bg-gray-800/40 transition-colors ${
                        halted ? 'bg-red-500/5' : ''
                      } ${selected ? 'ring-1 ring-inset ring-emerald-600/50' : ''}`}
                    >
                      <td className="px-4 py-2 font-mono text-gray-400">{timeAgo(c.time ?? c.started_at)}</td>
                      <td className="px-3 py-2">
                        <span className={`px-2 py-0.5 rounded-full border text-[9px] font-bold tracking-wider ${
                          halted ? 'bg-red-500/15 text-red-400 border-red-500/30' : 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30'
                        }`}>
                          {(c.status ?? '—').toUpperCase()}
                        </span>
                      </td>
                      <td className="px-3 py-2 font-mono text-gray-400">{c.mode ?? '—'}</td>
                      <td className="px-3 py-2 text-right font-mono text-gray-400">{c.scanned ?? '—'}</td>
                      <td className="px-3 py-2 text-right font-mono text-gray-400">{c.candidates ?? '—'}</td>
                      <td className="px-3 py-2 text-right font-mono text-emerald-400">{c.placed ?? '—'}</td>
                      <td className="px-3 py-2 text-right font-mono text-amber-400">{c.rejected ?? '—'}</td>
                      <td className="px-3 py-2 text-right font-mono">
                        {c.equity_delta == null
                          ? <span className="text-gray-700">—</span>
                          : <span className={c.equity_delta >= 0 ? 'text-emerald-400' : 'text-red-400'}>
                              {c.equity_delta >= 0 ? '+' : ''}{fmtNum(c.equity_delta)}
                            </span>}
                      </td>
                      <td className="px-4 py-2 text-gray-500 truncate max-w-[220px]">{c.halt_reason ?? '—'}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* ── Decisions ── */}
      <div className="mt-4 border border-gray-800 rounded-xl bg-gray-900/60 overflow-hidden">
        <div className="flex items-center justify-between px-4 py-2.5 border-b border-gray-800 flex-wrap gap-2">
          <h2 className="text-xs font-bold tracking-wider text-gray-300 uppercase">
            Decisions{selectedCycle !== null && <span className="text-gray-600 normal-case font-normal"> — cycle {selectedCycle}</span>}
          </h2>
          <div className="flex items-center gap-1.5 flex-wrap">
            {ACTIONS.map(a => (
              <button
                key={a || 'ALL'}
                onClick={() => setActionFilter(a)}
                className={`px-2.5 py-1 rounded text-[10px] font-bold tracking-wide transition-colors ${
                  actionFilter === a ? 'bg-emerald-600 text-white' : 'bg-gray-800 text-gray-500 hover:text-gray-300'
                }`}
              >
                {a || 'ALL'}
              </button>
            ))}
            {selectedCycle !== null && (
              <button onClick={loadGlobalDecisions} className="ml-2 text-[10px] text-gray-500 hover:text-gray-300 underline">
                back to feed
              </button>
            )}
          </div>
        </div>

        {decisionsLoading ? (
          <p className="text-gray-600 text-xs p-4">Loading…</p>
        ) : filteredDecisions.length === 0 ? (
          <p className="text-gray-600 text-xs p-4">No decisions.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-gray-800 text-[10px] text-gray-500 uppercase tracking-wider">
                  <th className="text-left  px-4 py-2 font-medium">Action</th>
                  <th className="text-left  px-3 py-2 font-medium">Symbol</th>
                  <th className="text-right px-3 py-2 font-medium">Qty</th>
                  <th className="text-right px-3 py-2 font-medium">Entry</th>
                  <th className="text-right px-3 py-2 font-medium">Stop</th>
                  <th className="text-right px-3 py-2 font-medium">Target</th>
                  <th className="text-left  px-3 py-2 font-medium">Governor rule</th>
                  <th className="text-left  px-4 py-2 font-medium">Reason</th>
                </tr>
              </thead>
              <tbody>
                {filteredDecisions.map((d, i) => (
                  <tr key={d.id ?? i} className="border-b border-gray-800/40 last:border-0">
                    <td className="px-4 py-2"><ActionPill a={d.action ?? 'NO_ACTION'} /></td>
                    <td className="px-3 py-2 font-mono text-gray-200">{d.symbol ?? '—'}</td>
                    <td className="px-3 py-2 text-right font-mono text-gray-400">{d.qty ?? '—'}</td>
                    <td className="px-3 py-2 text-right font-mono text-gray-400">{fmtNum(d.entry)}</td>
                    <td className="px-3 py-2 text-right font-mono text-red-400/80">{fmtNum(d.stop)}</td>
                    <td className="px-3 py-2 text-right font-mono text-emerald-400/80">{fmtNum(d.target)}</td>
                    <td className="px-3 py-2 font-mono text-gray-500">{d.governor_rule ?? '—'}</td>
                    <td className="px-4 py-2 text-gray-500 truncate max-w-[260px]">{d.reason ?? '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* ── Discrepancies ── */}
      <div className="mt-4 border border-gray-800 rounded-xl bg-gray-900/60 overflow-hidden">
        <div className="flex items-center justify-between px-4 py-2.5 border-b border-gray-800">
          <h2 className="text-xs font-bold tracking-wider text-gray-300 uppercase">Discrepancies</h2>
          <button onClick={loadDiscrepancies} className="text-[10px] text-gray-500 hover:text-gray-300 transition-colors">
            {discLoading ? 'LOADING…' : '↻ REFRESH'}
          </button>
        </div>
        {discrepancies.length === 0 ? (
          <p className="text-gray-600 text-xs p-4">No discrepancies logged.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-gray-800 text-[10px] text-gray-500 uppercase tracking-wider">
                  <th className="text-left  px-4 py-2 font-medium">Kind</th>
                  <th className="text-left  px-3 py-2 font-medium">Symbol</th>
                  <th className="text-left  px-3 py-2 font-medium">Expected</th>
                  <th className="text-left  px-3 py-2 font-medium">Actual</th>
                  <th className="text-left  px-4 py-2 font-medium">Resolution</th>
                </tr>
              </thead>
              <tbody>
                {discrepancies.map((d, i) => (
                  <tr key={i} className="border-b border-gray-800/40 last:border-0">
                    <td className="px-4 py-2 font-mono text-amber-400">{d.kind ?? '—'}</td>
                    <td className="px-3 py-2 font-mono text-gray-300">{d.symbol ?? '—'}</td>
                    <td className="px-3 py-2 font-mono text-gray-500">{d.expected ?? '—'}</td>
                    <td className="px-3 py-2 font-mono text-gray-500">{d.actual ?? '—'}</td>
                    <td className="px-4 py-2 text-gray-500">{d.resolution ?? '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}
