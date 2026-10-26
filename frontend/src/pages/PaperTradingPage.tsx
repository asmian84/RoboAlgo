// frontend/src/pages/PaperTradingPage.tsx
//
// PAPER — the mock account, and the auto-trader engine that trades into it,
// as two tabs on one page. The standalone AUTO page is retired; its control
// surface (status, kill switch, risk limits, cycles, decisions,
// discrepancies) moves here under the AUTO-TRADER tab.
//
// ACCOUNT reads/administers the local simulated broker directly —
// `/api/paper-account/*` (api/routers/paper_account.py) → PaperBroker
// (broker_engine/paper.py), backed by the `paper_account` / `paper_positions`
// / `paper_trades` tables. This is a DIFFERENT account than `/api/paper/*`
// (paper_trading.py → paper_engine/trader.py), which is a historical
// backtest simulator with different accounting semantics (it treats
// `ending_balance` as equity; PaperBroker treats it as a cash ledger and
// derives equity as cash + open position market value on every read). The
// two are not interchangeable — this page reads PaperBroker exclusively.
//
// This account is monitored, not hand-traded from here: the auto-trader
// places the orders, this page shows what it did and lets an operator mark
// positions to market or wipe the simulation. There is no manual order
// entry and no token/credit economy — those described capabilities the
// backend never had.
import { useCallback, useEffect, useState, type ReactNode } from 'react'

// ── Shared formatting ───────────────────────────────────────────────────────

function fmtMoney(n: number | null | undefined, digits = 2): string {
  if (n == null || Number.isNaN(n)) return '—'
  return n.toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits })
}

function fmtSigned(n: number | null | undefined, digits = 2): string {
  if (n == null || Number.isNaN(n)) return '—'
  return `${n >= 0 ? '+' : '-'}$${fmtMoney(Math.abs(n), digits)}`
}

function fmtNum(n: number | null | undefined, digits = 2): string {
  if (n == null || Number.isNaN(n)) return '—'
  return n.toFixed(digits)
}

// `iso` — an ISO datetime string, as returned by auto_trader.audit (`_iso()`).
function timeAgo(iso: string | null | undefined): string {
  if (!iso) return '—'
  const t = new Date(iso).getTime()
  if (Number.isNaN(t)) return '—'
  return timeAgoMs(t)
}

// `sec` — unix epoch SECONDS, as returned by auto_trader.state (`time.time()`
// in kill_meta.ts) — not an ISO string. Verified against the real response:
// {"on":false,"reason":"","actor":"api","ts":1790707852.07}
function timeAgoEpoch(sec: number | null | undefined): string {
  if (sec == null) return '—'
  return timeAgoMs(sec * 1000)
}

function timeAgoMs(t: number): string {
  const secs = Math.max(0, Math.floor((Date.now() - t) / 1000))
  if (secs < 60) return `${secs}s ago`
  if (secs < 3600) return `${Math.floor(secs / 60)}m ago`
  if (secs < 86400) return `${Math.floor(secs / 3600)}h ago`
  return `${Math.floor(secs / 86400)}d ago`
}

// ── ACCOUNT tab — types (verified with curl against a live backend) ────────
//
//   GET  /api/paper-account/account
//     {"account_id":"paper-1","cash":95000.0,"equity":98611.0,
//      "buying_power":95000.0,"broker":"paper","currency":"USD","is_paper":true}
//   GET  /api/paper-account/positions
//     [{"symbol":"TEST","qty":100,"avg_cost":50.0,"market_value":3611.0,
//       "unrealized_pnl":-1389.0,"unrealized_pct":-27.78,"side":"long",
//       "stop":45.0,"target":60.0,"entry_date":"2026-09-29"}]
//   GET  /api/paper-account/trades
//     [] (empty in this environment; shape per api/routers/paper_account.py:
//        id, symbol, entry_date, exit_date, entry_price, exit_price,
//        position_size, direction, pnl, return_percent, exit_reason)
//   POST /api/paper-account/mark-to-market
//     {"checked":1,"closed":[{"symbol":"TEST","exit_reason":"stop",
//      "exit_price":45.0,"pnl":-500.0}],"skipped_no_quote":0,"still_open":0}
//   POST /api/paper-account/reset  (optional body: {starting_balance})
//     {"status":"reset","account_id":"paper-1","cash":100000.0,"equity":100000.0}

interface Account {
  account_id: string
  cash: number
  equity: number
  buying_power: number
  broker: string
  currency: string
  is_paper: boolean
}

interface Position {
  symbol: string
  qty: number
  avg_cost: number
  market_value: number
  unrealized_pnl: number
  unrealized_pct: number
  side: string
  stop: number | null
  target: number | null
  entry_date: string | null
}

interface Trade {
  id: number
  symbol: string
  entry_date: string
  exit_date: string
  entry_price: number
  exit_price: number
  position_size: number
  direction: string
  pnl: number
  return_percent: number
  exit_reason: string | null
}

interface MarkToMarketResult {
  checked: number
  closed: { symbol: string; exit_reason: string; exit_price: number; pnl: number }[]
  skipped_no_quote: number
  still_open?: number
  error?: string
}

const PAPER_API = '/api/paper-account'
const BALANCE_OPTS = [10_000, 25_000, 100_000, 250_000, 1_000_000]

// ── AUTO-TRADER tab — types (verified with curl) ────────────────────────────
//
//   GET /api/auto-trader/status
//     {"enabled":false,"running":true,"kill_switch":false,
//      "kill_meta":{"on":false,"reason":"","actor":"api","ts":1790707852.07},
//      "mode":"paper","broker":"alpaca","is_paper":null,
//      "broker_error":"ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env",
//      "redis_available":true,"lock":null,"last_cycle":null}
//   GET /api/auto-trader/config
//     {"enabled":false,"mode":"paper","live_confirmed":false,"broker":"alpaca",
//      "asset_class":"equity","profile":"SWING","symbols":[],"max_symbols":30,
//      "max_workers":2,"max_new_positions_per_cycle":2,
//      "ignore_market_hours":false,"dry_run":true,
//      "risk":{"max_open_positions":5,"max_total_exposure_pct":60.0,
//               "per_trade_risk_pct":1.0,"daily_loss_limit_pct":3.0,
//               "consecutive_loss_halt":3,"max_position_size_pct":20.0}}
//   GET /api/auto-trader/cycles → {"cycles":[...]}  — fields per
//     auto_trader/audit.py `_cycle_to_dict`: id, cycle_id, started_at,
//     finished_at, mode, broker, status, symbols_scanned, candidates,
//     orders_placed, orders_rejected, equity_start, equity_end,
//     day_start_equity, regime, positions, halt_reason, error, created_at.
//     (NOT `time`/`scanned`/`placed`/`rejected`/`equity_delta` — those field
//     names don't exist on this row; equity_delta is derived here.)
//   GET /api/auto-trader/decisions → {"decisions":[...]} — fields per
//     `_decision_to_dict`: id, cycle_id, ts, symbol, action, verdict,
//     direction, score, confidence, entry, stop, target, qty, risk_amount,
//     realized_pnl, reason, governor_rule, broker_order_id, client_order_id.
//   GET /api/auto-trader/discrepancies → {"discrepancies":[...]}

interface KillMeta {
  on?: boolean
  reason?: string | null
  actor?: string | null
  ts?: number | null   // unix epoch SECONDS, not ISO — see timeAgoEpoch
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
  last_cycle: unknown
}

interface AutoTraderConfig {
  risk: Record<string, unknown>
  [key: string]: unknown
}

interface Cycle {
  id?: number
  cycle_id: string | number
  started_at?: string | null
  finished_at?: string | null
  mode?: string | null
  broker?: string | null
  status?: string | null
  symbols_scanned?: number | null
  candidates?: number | null
  orders_placed?: number | null
  orders_rejected?: number | null
  equity_start?: number | null
  equity_end?: number | null
  halt_reason?: string | null
  error?: string | null
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

const AUTO_API = '/api/auto-trader'
const TOKEN_KEY = 'roboalgo_control_token'
const ACTIONS = ['', 'PLACE', 'REJECT', 'SKIP', 'DRY_RUN', 'HALT', 'NO_ACTION'] as const

function loadToken(): string {
  try { return localStorage.getItem(TOKEN_KEY) ?? '' } catch { return '' }
}
function persistToken(v: string) {
  try { localStorage.setItem(TOKEN_KEY, v) } catch { /* private window etc — non-fatal */ }
}

// Distinguishes "not configured" (503) from "wrong token" (403) per the
// auth-asymmetry contract documented in api/routers/trading_control.py.
async function guarded(path: string, token: string, init: RequestInit = {}) {
  try {
    const res = await fetch(`${AUTO_API}${path}`, {
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

// ── Small presentational pieces (shared) ────────────────────────────────────

// Checks/pills: passed and failed are both subtle — colour never encodes
// pass/fail, only the glyph does. Not-evaluated is a dimmed dash, never red.
function CheckMark({ v }: { v: boolean | undefined | null }) {
  if (v === undefined || v === null) {
    return <span className="text-[color:var(--text-muted)] opacity-50">–</span>
  }
  return <span className="text-[color:var(--text-secondary)]">{v ? '✓' : '✕'}</span>
}

function Stat({
  label, value, tone, money = true,
}: {
  label: string
  value: number | string | null
  tone?: 'up' | 'down'
  money?: boolean
}) {
  const color = tone === 'up' ? 'text-[color:var(--up)]' : tone === 'down' ? 'text-[color:var(--down)]' : 'text-[color:var(--text-primary)]'
  const display = value == null
    ? '—'
    : typeof value === 'string'
      ? value
      : money
        ? (tone ? fmtSigned(value) : `$${fmtMoney(value)}`)
        : String(value)
  return (
    <div>
      <div className="text-[length:var(--text-xs)] text-[color:var(--text-muted)] uppercase tracking-wide">{label}</div>
      <div className={`num mt-1 text-[length:var(--text-xl)] font-[var(--weight-medium)] ${color}`}>{display}</div>
    </div>
  )
}

function Section({ title, action, children }: { title: string; action?: ReactNode; children: ReactNode }) {
  return (
    <div className="rounded-[var(--radius-lg)] bg-[color:var(--bg-panel)] overflow-hidden">
      <div className="flex items-center justify-between px-4 py-2.5 border-b border-[color:var(--line-subtle)]">
        <h2 className="text-[length:var(--text-xs)] font-[var(--weight-bold)] text-[color:var(--text-secondary)] uppercase tracking-wider">{title}</h2>
        {action}
      </div>
      {children}
    </div>
  )
}

// ── Reset modal (shared by ACCOUNT tab) ─────────────────────────────────────

function ResetModal({
  onConfirm, onCancel, busy,
}: {
  onConfirm: (startingBalance: number | null) => void
  onCancel: () => void
  busy: boolean
}) {
  const [choice, setChoice] = useState<number | 'default'>('default')

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70">
      <div className="rounded-[var(--radius-lg)] bg-[color:var(--bg-overlay)] p-6 w-full max-w-sm space-y-4">
        <h3 className="text-[length:var(--text-sm)] font-[var(--weight-bold)] text-[color:var(--text-primary)]">Reset paper account</h3>
        <p className="text-[length:var(--text-xs)] text-[color:var(--text-muted)]">
          This wipes every open position and every closed trade and restores the opening cash balance. It cannot be undone.
        </p>
        <div className="space-y-1.5">
          <div className="text-[length:var(--text-xs)] text-[color:var(--text-muted)] uppercase tracking-wide">Starting balance</div>
          <div className="grid grid-cols-3 gap-1.5">
            <button
              onClick={() => setChoice('default')}
              className={`py-2 rounded-[var(--radius-md)] text-[length:var(--text-xs)] font-[var(--weight-bold)] transition-colors ${
                choice === 'default' ? 'bg-[color:var(--accent)] text-[color:var(--accent-fg)]' : 'bg-[color:var(--bg-raised)] text-[color:var(--text-secondary)] hover:text-[color:var(--text-primary)]'
              }`}
            >
              Keep current
            </button>
            {BALANCE_OPTS.map(b => (
              <button
                key={b}
                onClick={() => setChoice(b)}
                className={`py-2 rounded-[var(--radius-md)] text-[length:var(--text-xs)] font-[var(--weight-bold)] num transition-colors ${
                  choice === b ? 'bg-[color:var(--accent)] text-[color:var(--accent-fg)]' : 'bg-[color:var(--bg-raised)] text-[color:var(--text-secondary)] hover:text-[color:var(--text-primary)]'
                }`}
              >
                ${b >= 1_000_000 ? `${b / 1_000_000}M` : `${b / 1000}K`}
              </button>
            ))}
          </div>
        </div>
        <div className="flex gap-2 pt-1">
          <button
            onClick={onCancel}
            className="flex-1 py-2 rounded-[var(--radius-md)] bg-[color:var(--bg-raised)] text-[color:var(--text-secondary)] text-[length:var(--text-sm)] hover:text-[color:var(--text-primary)] transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={() => onConfirm(choice === 'default' ? null : choice)}
            disabled={busy}
            className="flex-1 py-2 rounded-[var(--radius-md)] bg-[color:var(--accent)] text-[color:var(--accent-fg)] font-[var(--weight-bold)] text-[length:var(--text-sm)] disabled:opacity-50 transition-opacity"
          >
            {busy ? 'Resetting…' : 'Reset'}
          </button>
        </div>
      </div>
    </div>
  )
}

// ── ACCOUNT tab ──────────────────────────────────────────────────────────────

function AccountTab() {
  const [account, setAccount] = useState<Account | null>(null)
  const [positions, setPositions] = useState<Position[]>([])
  const [trades, setTrades] = useState<Trade[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const [mtmBusy, setMtmBusy] = useState(false)
  const [mtmResult, setMtmResult] = useState<MarkToMarketResult | null>(null)

  const [showReset, setShowReset] = useState(false)
  const [resetBusy, setResetBusy] = useState(false)

  const loadAll = useCallback(async () => {
    setLoading(true)
    try {
      const [acctRes, posRes, tradesRes] = await Promise.all([
        fetch(`${PAPER_API}/account`),
        fetch(`${PAPER_API}/positions`),
        fetch(`${PAPER_API}/trades?limit=100`),
      ])
      if (!acctRes.ok) throw new Error(`account: HTTP ${acctRes.status}`)
      if (!posRes.ok) throw new Error(`positions: HTTP ${posRes.status}`)
      if (!tradesRes.ok) throw new Error(`trades: HTTP ${tradesRes.status}`)
      setAccount(await acctRes.json())
      setPositions(await posRes.json())
      setTrades(await tradesRes.json())
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { loadAll() }, [loadAll])

  async function doMarkToMarket() {
    setMtmBusy(true)
    try {
      const res = await fetch(`${PAPER_API}/mark-to-market`, { method: 'POST' })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data: MarkToMarketResult = await res.json()
      setMtmResult(data)
      await loadAll()
    } catch (err) {
      setMtmResult({ checked: 0, closed: [], skipped_no_quote: 0, error: err instanceof Error ? err.message : String(err) })
    } finally {
      setMtmBusy(false)
    }
  }

  async function doReset(startingBalance: number | null) {
    setResetBusy(true)
    try {
      const res = await fetch(`${PAPER_API}/reset`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(startingBalance != null ? { starting_balance: startingBalance } : {}),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      setShowReset(false)
      setMtmResult(null)
      await loadAll()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setResetBusy(false)
    }
  }

  const openPnl = positions.reduce((a, p) => a + (p.unrealized_pnl ?? 0), 0)
  const closedPnl = trades.reduce((a, t) => a + (t.pnl ?? 0), 0)
  const wins = trades.filter(t => (t.pnl ?? 0) > 0).length

  if (loading && !account) {
    return <div className="rounded-[var(--radius-lg)] bg-[color:var(--bg-panel)] p-8 text-center text-[length:var(--text-sm)] text-[color:var(--text-secondary)]">Loading account…</div>
  }

  if (error && !account) {
    return (
      <div className="rounded-[var(--radius-lg)] bg-[color:var(--bg-panel)] border border-[color:var(--line-strong)] p-4 text-[length:var(--text-sm)] text-[color:var(--text-secondary)]">
        Could not load the paper account: {error}
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {showReset && <ResetModal onConfirm={doReset} onCancel={() => setShowReset(false)} busy={resetBusy} />}

      {/* Hero */}
      <div className="rounded-[var(--radius-lg)] bg-[color:var(--bg-panel)] p-6">
        <div className="flex flex-wrap items-end justify-between gap-6">
          <div>
            <div className="text-[length:var(--text-xs)] text-[color:var(--text-muted)] uppercase tracking-wide">Equity</div>
            <div className="num mt-1 text-[length:var(--text-2xl)] font-[var(--weight-bold)] text-[color:var(--text-primary)]">
              ${fmtMoney(account?.equity)}
            </div>
            <div className="mt-1 text-[length:var(--text-xs)] text-[color:var(--text-muted)]">
              {account ? `${account.broker} · ${account.is_paper ? 'simulated' : 'live'} · ${account.currency}` : '—'}
            </div>
          </div>
          <div className="flex items-end gap-8 flex-wrap">
            <Stat label="Cash" value={account?.cash ?? null} />
            <Stat label="Open P&L" value={positions.length ? openPnl : null} tone={openPnl >= 0 ? 'up' : 'down'} />
            <Stat label="Positions" value={positions.length} money={false} />
          </div>
        </div>

        <div className="mt-5 flex flex-wrap items-center gap-2">
          <button
            onClick={doMarkToMarket}
            disabled={mtmBusy}
            className="px-4 py-2 rounded-[var(--radius-md)] bg-[color:var(--accent)] text-[color:var(--accent-fg)] font-[var(--weight-bold)] text-[length:var(--text-xs)] tracking-wide uppercase disabled:opacity-50 transition-opacity"
          >
            {mtmBusy ? 'Marking…' : 'Mark to market'}
          </button>
          <button
            onClick={() => setShowReset(true)}
            className="px-4 py-2 rounded-[var(--radius-md)] bg-[color:var(--bg-raised)] text-[color:var(--text-secondary)] hover:text-[color:var(--text-primary)] font-[var(--weight-bold)] text-[length:var(--text-xs)] tracking-wide uppercase transition-colors"
          >
            Reset
          </button>
          {mtmResult && (
            <span className="text-[length:var(--text-xs)] text-[color:var(--text-muted)]">
              {mtmResult.error
                ? `mark-to-market failed: ${mtmResult.error}`
                : `checked ${mtmResult.checked} · closed ${mtmResult.closed.length}${mtmResult.skipped_no_quote ? ` · ${mtmResult.skipped_no_quote} skipped (no quote)` : ''}`}
            </span>
          )}
        </div>
        {mtmResult && mtmResult.closed.length > 0 && (
          <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[length:var(--text-xs)] text-[color:var(--text-secondary)]">
            {mtmResult.closed.map((c, i) => (
              <span key={i} className="num">
                {c.symbol} {c.exit_reason} @ {fmtNum(c.exit_price)}{' '}
                <span className={c.pnl >= 0 ? 'text-[color:var(--up)]' : 'text-[color:var(--down)]'}>{fmtSigned(c.pnl)}</span>
              </span>
            ))}
          </div>
        )}
      </div>

      {/* Open positions */}
      <Section title="Open positions" action={<span className="text-[length:var(--text-xs)] text-[color:var(--text-muted)]">{positions.length} open</span>}>
        {positions.length === 0 ? (
          <p className="p-4 text-[length:var(--text-sm)] text-[color:var(--text-muted)]">No open positions.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-[length:var(--text-sm)]">
              <thead>
                <tr className="border-b border-[color:var(--line-subtle)] text-[length:var(--text-xs)] text-[color:var(--text-muted)] uppercase tracking-wide">
                  <th className="text-left px-4 py-2 font-[var(--weight-medium)]">Symbol</th>
                  <th className="text-left px-3 py-2 font-[var(--weight-medium)]">Side</th>
                  <th className="text-right px-3 py-2 font-[var(--weight-medium)]">Qty</th>
                  <th className="text-right px-3 py-2 font-[var(--weight-medium)]">Avg cost</th>
                  <th className="text-right px-3 py-2 font-[var(--weight-medium)]">Current</th>
                  <th className="text-right px-3 py-2 font-[var(--weight-medium)]">Mkt value</th>
                  <th className="text-right px-3 py-2 font-[var(--weight-medium)]">Unrealized P&L</th>
                  <th className="text-right px-3 py-2 font-[var(--weight-medium)]">Stop</th>
                  <th className="text-right px-4 py-2 font-[var(--weight-medium)]">Target</th>
                </tr>
              </thead>
              <tbody>
                {positions.map(p => {
                  const current = p.qty ? p.market_value / p.qty : null
                  const up = p.side === 'long'
                  return (
                    <tr key={p.symbol} className="border-b border-[color:var(--line-subtle)] last:border-0">
                      <td className="px-4 py-2 font-[var(--weight-bold)] text-[color:var(--text-primary)]">{p.symbol}</td>
                      <td className="px-3 py-2">
                        <span className={up ? 'text-[color:var(--up)]' : 'text-[color:var(--down)]'}>{up ? '▲ LONG' : '▼ SHORT'}</span>
                      </td>
                      <td className="px-3 py-2 text-right num text-[color:var(--text-secondary)]">{p.qty}</td>
                      <td className="px-3 py-2 text-right num text-[color:var(--text-secondary)]">${fmtMoney(p.avg_cost)}</td>
                      <td className="px-3 py-2 text-right num text-[color:var(--text-secondary)]">{current == null ? '—' : `$${fmtMoney(current)}`}</td>
                      <td className="px-3 py-2 text-right num text-[color:var(--text-secondary)]">${fmtMoney(p.market_value)}</td>
                      <td className={`px-3 py-2 text-right num font-[var(--weight-medium)] ${p.unrealized_pnl >= 0 ? 'text-[color:var(--up)]' : 'text-[color:var(--down)]'}`}>
                        {fmtSigned(p.unrealized_pnl)} ({fmtNum(p.unrealized_pct, 1)}%)
                      </td>
                      <td className="px-3 py-2 text-right num text-[color:var(--down)]">{p.stop == null ? '—' : `$${fmtMoney(p.stop)}`}</td>
                      <td className="px-4 py-2 text-right num text-[color:var(--up)]">{p.target == null ? '—' : `$${fmtMoney(p.target)}`}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </Section>

      {/* Closed trades */}
      <Section
        title="Closed trades"
        action={
          trades.length > 0 ? (
            <span className="text-[length:var(--text-xs)] text-[color:var(--text-muted)]">
              {wins}/{trades.length} won ·{' '}
              <span className={closedPnl >= 0 ? 'text-[color:var(--up)]' : 'text-[color:var(--down)]'}>{fmtSigned(closedPnl)}</span> realized
            </span>
          ) : undefined
        }
      >
        {trades.length === 0 ? (
          <p className="p-4 text-[length:var(--text-sm)] text-[color:var(--text-muted)]">No closed trades yet.</p>
        ) : (
          <div className="overflow-x-auto max-h-96">
            <table className="w-full text-[length:var(--text-sm)]">
              <thead>
                <tr className="border-b border-[color:var(--line-subtle)] text-[length:var(--text-xs)] text-[color:var(--text-muted)] uppercase tracking-wide sticky top-0 bg-[color:var(--bg-panel)]">
                  <th className="text-left px-4 py-2 font-[var(--weight-medium)]">Symbol</th>
                  <th className="text-left px-3 py-2 font-[var(--weight-medium)]">Side</th>
                  <th className="text-right px-3 py-2 font-[var(--weight-medium)]">Entry</th>
                  <th className="text-right px-3 py-2 font-[var(--weight-medium)]">Exit</th>
                  <th className="text-right px-3 py-2 font-[var(--weight-medium)]">Size</th>
                  <th className="text-right px-3 py-2 font-[var(--weight-medium)]">P&L</th>
                  <th className="text-left px-3 py-2 font-[var(--weight-medium)]">Reason</th>
                  <th className="text-right px-4 py-2 font-[var(--weight-medium)]">Exit date</th>
                </tr>
              </thead>
              <tbody>
                {trades.map(t => {
                  const up = t.direction === 'long'
                  return (
                    <tr key={t.id} className="border-b border-[color:var(--line-subtle)] last:border-0">
                      <td className="px-4 py-2 font-[var(--weight-bold)] text-[color:var(--text-primary)]">{t.symbol}</td>
                      <td className="px-3 py-2">
                        <span className={up ? 'text-[color:var(--up)]' : 'text-[color:var(--down)]'}>{up ? '▲' : '▼'}</span>
                      </td>
                      <td className="px-3 py-2 text-right num text-[color:var(--text-secondary)]">${fmtMoney(t.entry_price)}</td>
                      <td className="px-3 py-2 text-right num text-[color:var(--text-secondary)]">${fmtMoney(t.exit_price)}</td>
                      <td className="px-3 py-2 text-right num text-[color:var(--text-secondary)]">{t.position_size}</td>
                      <td className={`px-3 py-2 text-right num font-[var(--weight-medium)] ${t.pnl >= 0 ? 'text-[color:var(--up)]' : 'text-[color:var(--down)]'}`}>
                        {fmtSigned(t.pnl)} ({fmtNum(t.return_percent, 1)}%)
                      </td>
                      <td className="px-3 py-2 text-[length:var(--text-xs)] text-[color:var(--text-muted)]">{t.exit_reason ?? '—'}</td>
                      <td className="px-4 py-2 text-right text-[length:var(--text-xs)] text-[color:var(--text-muted)]">{t.exit_date}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </Section>
    </div>
  )
}

// ── AUTO-TRADER tab ──────────────────────────────────────────────────────────

function AutoTraderTab() {
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

  useEffect(() => { persistToken(token) }, [token])

  const fetchStatus = useCallback(() => {
    fetch(`${AUTO_API}/status`)
      .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json() })
      .then(d => { setStatus(d); setStatusError(null) })
      .catch(err => setStatusError(err instanceof Error ? err.message : String(err)))
  }, [])

  useEffect(() => {
    fetchStatus()
    const id = setInterval(fetchStatus, 5000)
    return () => clearInterval(id)
  }, [fetchStatus])

  const loadConfig = useCallback(() => {
    fetch(`${AUTO_API}/config`)
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

  async function doKillOn() {
    setBusy('kill'); setActionMsg(null)
    try {
      const res = await fetch(`${AUTO_API}/kill`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ on: true, reason: killReason.trim() || 'Manual kill from PAPER / AUTO-TRADER' }),
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
    const res = await guarded('/kill', token, { method: 'POST', body: JSON.stringify({ on: false, reason: 'Cleared from PAPER / AUTO-TRADER' }) })
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
      const res = await fetch(`${AUTO_API}/stop`, { method: 'POST' })
      if (!res.ok) { const b = await res.json().catch(() => null); throw new Error(b?.detail ?? `HTTP ${res.status}`) }
      setActionMsg({ ok: true, text: 'Stopped.' })
      fetchStatus()
    } catch (err) {
      setActionMsg({ ok: false, text: err instanceof Error ? err.message : String(err) })
    } finally {
      setBusy(null)
    }
  }

  const loadCycles = useCallback(() => {
    setCyclesLoading(true)
    fetch(`${AUTO_API}/cycles?limit=50`)
      .then(r => r.json())
      .then(d => setCycles(d?.cycles ?? (Array.isArray(d) ? d : [])))
      .catch(() => {})
      .finally(() => setCyclesLoading(false))
  }, [])

  useEffect(() => { loadCycles() }, [loadCycles])

  const loadGlobalDecisions = useCallback(() => {
    setDecisionsLoading(true); setSelectedCycle(null)
    const qs = new URLSearchParams({ limit: '200' })
    if (actionFilter) qs.set('action', actionFilter)
    fetch(`${AUTO_API}/decisions?${qs.toString()}`)
      .then(r => r.json())
      .then(d => setDecisions(d?.decisions ?? (Array.isArray(d) ? d : [])))
      .catch(() => {})
      .finally(() => setDecisionsLoading(false))
  }, [actionFilter])

  useEffect(() => { loadGlobalDecisions() }, [loadGlobalDecisions])

  function loadCycleDecisions(id: string | number) {
    setDecisionsLoading(true); setSelectedCycle(id)
    fetch(`${AUTO_API}/cycles/${id}/decisions`)
      .then(r => r.json())
      .then(d => setDecisions(d?.decisions ?? (Array.isArray(d) ? d : [])))
      .catch(() => {})
      .finally(() => setDecisionsLoading(false))
  }

  const loadDiscrepancies = useCallback(() => {
    setDiscLoading(true)
    fetch(`${AUTO_API}/discrepancies?limit=100`)
      .then(r => r.json())
      .then(d => setDiscrepancies(d?.discrepancies ?? (Array.isArray(d) ? d : [])))
      .catch(() => {})
      .finally(() => setDiscLoading(false))
  }, [])

  useEffect(() => { loadDiscrepancies() }, [loadDiscrepancies])

  const filteredDecisions = selectedCycle !== null && actionFilter
    ? decisions.filter(d => d.action === actionFilter)
    : decisions

  const isLive = status ? (status.mode?.toLowerCase().includes('live') || status.is_paper === false) : false

  return (
    <div className="space-y-4">
      {/* Kill switch banner — follows FinderPage's warning-banner precedent: the
          single accent colour, not a bespoke red, since --down is reserved
          for P&L direction only. */}
      {status?.kill_switch && (
        <div className="rounded-[var(--radius-lg)] bg-[color:var(--bg-raised)] border border-[color:var(--line-strong)] p-3">
          <div className="flex items-start gap-2">
            <span className="text-[color:var(--accent)] text-[length:var(--text-sm)]">⛔</span>
            <div className="flex-1 min-w-0">
              <div className="text-[length:var(--text-xs)] font-[var(--weight-bold)] text-[color:var(--accent)] tracking-wide">KILL SWITCH ENGAGED</div>
              <div className="mt-1 text-[length:var(--text-xs)] text-[color:var(--text-secondary)]">
                {status.kill_meta?.reason || 'no reason given'}
                {status.kill_meta?.actor && <span className="text-[color:var(--text-muted)]"> · by {status.kill_meta.actor}</span>}
                {status.kill_meta?.ts != null && <span className="text-[color:var(--text-muted)]"> · {timeAgoEpoch(status.kill_meta.ts)}</span>}
              </div>
            </div>
          </div>
        </div>
      )}

      {status && !status.redis_available && (
        <div className="rounded-[var(--radius-lg)] bg-[color:var(--bg-raised)] border border-[color:var(--line-strong)] p-3">
          <div className="flex items-start gap-2">
            <span className="text-[color:var(--accent)] text-[length:var(--text-sm)]">⚠</span>
            <div className="flex-1 min-w-0">
              <div className="text-[length:var(--text-xs)] font-[var(--weight-bold)] text-[color:var(--accent)] tracking-wide">DEGRADED — REDIS UNAVAILABLE</div>
              <p className="mt-1 text-[length:var(--text-xs)] text-[color:var(--text-muted)]">
                Running single-host; the loop force-downgrades to dry-run until Redis is back.
              </p>
            </div>
          </div>
        </div>
      )}

      {statusError && (
        <div className="text-[length:var(--text-xs)] text-[color:var(--text-muted)]">Status poll failing: {statusError}</div>
      )}

      {/* Status bar */}
      <div className="rounded-[var(--radius-lg)] bg-[color:var(--bg-panel)] p-4">
        <div className="flex flex-wrap items-center gap-x-8 gap-y-3">
          <div>
            <div className="text-[length:var(--text-xs)] text-[color:var(--text-muted)] uppercase tracking-wide">Loop</div>
            <div className={`mt-1 text-[length:var(--text-sm)] font-[var(--weight-bold)] ${status?.running ? 'text-[color:var(--accent)]' : 'text-[color:var(--text-muted)]'}`}>
              {status?.running ? 'RUNNING' : 'STOPPED'}
            </div>
          </div>
          <div>
            <div className="text-[length:var(--text-xs)] text-[color:var(--text-muted)] uppercase tracking-wide">Mode</div>
            <div className="mt-1 num text-[length:var(--text-sm)] text-[color:var(--text-secondary)]">{status?.mode ?? '—'}</div>
          </div>
          <div>
            <div className="text-[length:var(--text-xs)] text-[color:var(--text-muted)] uppercase tracking-wide">Broker</div>
            <div className="mt-1 num text-[length:var(--text-sm)] text-[color:var(--text-secondary)]">
              {status?.broker ?? '—'}
              {status?.is_paper === true && <span className="ml-1.5 text-[length:var(--text-xs)] text-[color:var(--text-muted)]">simulated</span>}
              {status?.is_paper === false && <span className="ml-1.5 text-[length:var(--text-xs)] font-[var(--weight-bold)] text-[color:var(--accent)]">LIVE</span>}
            </div>
            {status?.is_paper === null && status?.broker_error && (
              <div className="mt-0.5 text-[length:var(--text-xs)] text-[color:var(--text-muted)] max-w-[240px]">account type unknown — {status.broker_error}</div>
            )}
          </div>
          <div>
            <div className="text-[length:var(--text-xs)] text-[color:var(--text-muted)] uppercase tracking-wide">Redis</div>
            <div className="mt-1 text-[length:var(--text-sm)] text-[color:var(--text-secondary)]">
              <CheckMark v={status?.redis_available} /> {status?.redis_available ? 'up' : 'down'}
            </div>
          </div>
          <div>
            <div className="text-[length:var(--text-xs)] text-[color:var(--text-muted)] uppercase tracking-wide">Lock</div>
            <div className="mt-1 num text-[length:var(--text-sm)] text-[color:var(--text-muted)]">{status?.lock ? String(status.lock) : 'free'}</div>
          </div>
        </div>
      </div>

      {/* Kill / start / stop */}
      <div className="rounded-[var(--radius-lg)] bg-[color:var(--bg-panel)] p-4">
        <div className="flex flex-wrap items-end gap-3">
          <div className="flex-1 min-w-[220px]">
            <label className="text-[length:var(--text-xs)] text-[color:var(--text-muted)] uppercase tracking-wide">Kill reason (optional)</label>
            <input
              value={killReason}
              onChange={e => setKillReason(e.target.value)}
              placeholder="e.g. broker outage, manual halt"
              className="mt-1 w-full px-3 py-2 rounded-[var(--radius-md)] bg-[color:var(--bg-raised)] text-[length:var(--text-sm)] text-[color:var(--text-primary)] placeholder:text-[color:var(--text-muted)] focus:outline-none"
            />
          </div>
          <button
            onClick={doKillOn}
            disabled={busy === 'kill'}
            className="px-6 py-2.5 rounded-[var(--radius-md)] bg-[color:var(--accent)] text-[color:var(--accent-fg)] text-[length:var(--text-sm)] font-[var(--weight-bold)] tracking-widest disabled:opacity-50 transition-opacity"
          >
            {busy === 'kill' ? 'KILLING…' : '⛔ KILL'}
          </button>
          <button
            onClick={doKillOff}
            disabled={busy === 'unkill' || !status?.kill_switch}
            title="Requires control token"
            className="px-4 py-2 rounded-[var(--radius-md)] bg-[color:var(--bg-raised)] text-[color:var(--text-secondary)] hover:text-[color:var(--text-primary)] disabled:opacity-40 text-[length:var(--text-xs)] font-[var(--weight-bold)] tracking-wide transition-colors"
          >
            {busy === 'unkill' ? 'CLEARING…' : 'CLEAR KILL'}
          </button>
          <div className="flex-1" />
          <button
            onClick={doStart}
            disabled={busy === 'start' || !!status?.running}
            title="Requires control token"
            className="px-5 py-2 rounded-[var(--radius-md)] bg-[color:var(--bg-raised)] text-[color:var(--accent)] disabled:text-[color:var(--text-muted)] disabled:opacity-60 text-[length:var(--text-xs)] font-[var(--weight-bold)] tracking-wide transition-colors"
          >
            {busy === 'start' ? 'STARTING…' : 'START'}
          </button>
          <button
            onClick={doStop}
            disabled={busy === 'stop' || !status?.running}
            className="px-5 py-2 rounded-[var(--radius-md)] bg-[color:var(--bg-raised)] text-[color:var(--text-secondary)] disabled:opacity-40 text-[length:var(--text-xs)] font-[var(--weight-bold)] tracking-wide transition-colors"
          >
            {busy === 'stop' ? 'STOPPING…' : 'STOP'}
          </button>
        </div>

        {actionMsg && (
          <p className={`mt-3 text-[length:var(--text-xs)] ${actionMsg.ok ? 'text-[color:var(--up)]' : 'text-[color:var(--down)]'}`}>{actionMsg.text}</p>
        )}

        <div className="mt-4 pt-3 border-t border-[color:var(--line-subtle)]">
          <label className="text-[length:var(--text-xs)] text-[color:var(--text-muted)] uppercase tracking-wide">Control token (for guarded actions)</label>
          <input
            type="password"
            value={token}
            onChange={e => setToken(e.target.value)}
            placeholder="X-RoboAlgo-Control-Token"
            className="mt-1 w-full max-w-sm px-3 py-2 rounded-[var(--radius-md)] bg-[color:var(--bg-raised)] text-[length:var(--text-sm)] text-[color:var(--text-primary)] placeholder:text-[color:var(--text-muted)] focus:outline-none"
          />
          <p className="mt-1 text-[length:var(--text-xs)] text-[color:var(--text-muted)]">
            Stored locally in this browser only. Guarded endpoints return 503 if the server has no token configured, 403 if this one is wrong.
          </p>
        </div>
      </div>

      {/* Risk limits */}
      <Section
        title="Risk limits"
        action={
          <button
            onClick={saveRisk}
            disabled={savingConfig}
            className="px-4 py-1.5 rounded-[var(--radius-md)] bg-[color:var(--accent)] text-[color:var(--accent-fg)] text-[length:var(--text-xs)] font-[var(--weight-bold)] tracking-wide disabled:opacity-50 transition-opacity"
          >
            {savingConfig ? 'SAVING…' : 'SAVE'}
          </button>
        }
      >
        <div className="p-4">
          {!config ? (
            <p className="text-[length:var(--text-sm)] text-[color:var(--text-muted)]">Loading config…</p>
          ) : Object.keys(riskEdits).length === 0 ? (
            <p className="text-[length:var(--text-sm)] text-[color:var(--text-muted)]">No risk config found.</p>
          ) : (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {Object.entries(riskEdits).map(([key, value]) => {
                const isBool = typeof value === 'boolean'
                const isNum = typeof value === 'number'
                return (
                  <div key={key}>
                    <label className="text-[length:var(--text-xs)] text-[color:var(--text-muted)] uppercase tracking-wide truncate block">{key}</label>
                    {isBool ? (
                      <label className="mt-1 flex items-center gap-1.5 text-[length:var(--text-sm)] text-[color:var(--text-secondary)] cursor-pointer">
                        <input
                          type="checkbox"
                          checked={value as boolean}
                          onChange={e => setRiskEdits(r => ({ ...r, [key]: e.target.checked }))}
                        />
                        {(value as boolean) ? 'true' : 'false'}
                      </label>
                    ) : (
                      <input
                        type={isNum ? 'number' : 'text'}
                        value={value == null ? '' : String(value)}
                        onChange={e => setRiskEdits(r => ({ ...r, [key]: isNum ? Number(e.target.value) : e.target.value }))}
                        className="mt-1 w-full px-2.5 py-1.5 rounded-[var(--radius-md)] bg-[color:var(--bg-raised)] num text-[length:var(--text-sm)] text-[color:var(--text-primary)] focus:outline-none"
                      />
                    )}
                  </div>
                )
              })}
            </div>
          )}
          {configMsg && (
            <p className={`mt-3 text-[length:var(--text-xs)] ${configMsg.ok ? 'text-[color:var(--up)]' : 'text-[color:var(--down)]'}`}>{configMsg.text}</p>
          )}
        </div>
      </Section>

      {/* Cycles */}
      <Section
        title="Cycles"
        action={
          <button onClick={loadCycles} className="text-[length:var(--text-xs)] text-[color:var(--text-muted)] hover:text-[color:var(--text-secondary)] transition-colors">
            {cyclesLoading ? 'LOADING…' : '↻ REFRESH'}
          </button>
        }
      >
        {cycles.length === 0 ? (
          <p className="p-4 text-[length:var(--text-sm)] text-[color:var(--text-muted)]">No cycles recorded yet.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-[length:var(--text-sm)]">
              <thead>
                <tr className="border-b border-[color:var(--line-subtle)] text-[length:var(--text-xs)] text-[color:var(--text-muted)] uppercase tracking-wide">
                  <th className="text-left px-4 py-2 font-[var(--weight-medium)]">Started</th>
                  <th className="text-left px-3 py-2 font-[var(--weight-medium)]">Status</th>
                  <th className="text-left px-3 py-2 font-[var(--weight-medium)]">Mode</th>
                  <th className="text-right px-3 py-2 font-[var(--weight-medium)]">Scanned</th>
                  <th className="text-right px-3 py-2 font-[var(--weight-medium)]">Candidates</th>
                  <th className="text-right px-3 py-2 font-[var(--weight-medium)]">Placed</th>
                  <th className="text-right px-3 py-2 font-[var(--weight-medium)]">Rejected</th>
                  <th className="text-right px-3 py-2 font-[var(--weight-medium)]">Equity Δ</th>
                  <th className="text-left px-4 py-2 font-[var(--weight-medium)]">Halt reason</th>
                </tr>
              </thead>
              <tbody>
                {cycles.map(c => {
                  const halted = (c.status ?? '').toLowerCase() === 'halted'
                  const selected = selectedCycle === c.cycle_id
                  const equityDelta = c.equity_end != null && c.equity_start != null ? c.equity_end - c.equity_start : null
                  return (
                    <tr
                      key={c.cycle_id}
                      onClick={() => loadCycleDecisions(c.cycle_id)}
                      className={`border-b border-[color:var(--line-subtle)] last:border-0 cursor-pointer hover:bg-[color:var(--bg-raised)] transition-colors ${
                        selected ? 'bg-[color:var(--bg-raised)]' : ''
                      }`}
                    >
                      <td className="px-4 py-2 num text-[color:var(--text-muted)]">{timeAgo(c.started_at)}</td>
                      <td className="px-3 py-2 text-[color:var(--text-secondary)]">{(c.status ?? '—').toUpperCase()}{halted ? ' ⚠' : ''}</td>
                      <td className="px-3 py-2 num text-[color:var(--text-muted)]">{c.mode ?? '—'}</td>
                      <td className="px-3 py-2 text-right num text-[color:var(--text-secondary)]">{c.symbols_scanned ?? '—'}</td>
                      <td className="px-3 py-2 text-right num text-[color:var(--text-secondary)]">{c.candidates ?? '—'}</td>
                      <td className="px-3 py-2 text-right num text-[color:var(--text-secondary)]">{c.orders_placed ?? '—'}</td>
                      <td className="px-3 py-2 text-right num text-[color:var(--text-secondary)]">{c.orders_rejected ?? '—'}</td>
                      <td className="px-3 py-2 text-right num">
                        {equityDelta == null
                          ? <span className="text-[color:var(--text-muted)]">—</span>
                          : <span className={equityDelta >= 0 ? 'text-[color:var(--up)]' : 'text-[color:var(--down)]'}>{fmtSigned(equityDelta)}</span>}
                      </td>
                      <td className="px-4 py-2 text-[color:var(--text-muted)] truncate max-w-[220px]">{c.halt_reason ?? c.error ?? '—'}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </Section>

      {/* Decisions */}
      <Section
        title={`Decisions${selectedCycle !== null ? ` — cycle ${selectedCycle}` : ''}`}
        action={
          <div className="flex items-center gap-1.5 flex-wrap">
            {ACTIONS.map(a => (
              <button
                key={a || 'ALL'}
                onClick={() => setActionFilter(a)}
                className={`px-2.5 py-1 rounded-[var(--radius-sm)] text-[length:var(--text-xs)] font-[var(--weight-bold)] tracking-wide transition-colors ${
                  actionFilter === a ? 'bg-[color:var(--accent)] text-[color:var(--accent-fg)]' : 'bg-[color:var(--bg-raised)] text-[color:var(--text-muted)] hover:text-[color:var(--text-secondary)]'
                }`}
              >
                {a || 'ALL'}
              </button>
            ))}
            {selectedCycle !== null && (
              <button onClick={loadGlobalDecisions} className="ml-2 text-[length:var(--text-xs)] text-[color:var(--text-muted)] hover:text-[color:var(--text-secondary)] underline">
                back to feed
              </button>
            )}
          </div>
        }
      >
        {decisionsLoading ? (
          <p className="p-4 text-[length:var(--text-sm)] text-[color:var(--text-muted)]">Loading…</p>
        ) : filteredDecisions.length === 0 ? (
          <p className="p-4 text-[length:var(--text-sm)] text-[color:var(--text-muted)]">No decisions.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-[length:var(--text-sm)]">
              <thead>
                <tr className="border-b border-[color:var(--line-subtle)] text-[length:var(--text-xs)] text-[color:var(--text-muted)] uppercase tracking-wide">
                  <th className="text-left px-4 py-2 font-[var(--weight-medium)]">Action</th>
                  <th className="text-left px-3 py-2 font-[var(--weight-medium)]">Symbol</th>
                  <th className="text-right px-3 py-2 font-[var(--weight-medium)]">Qty</th>
                  <th className="text-right px-3 py-2 font-[var(--weight-medium)]">Entry</th>
                  <th className="text-right px-3 py-2 font-[var(--weight-medium)]">Stop</th>
                  <th className="text-right px-3 py-2 font-[var(--weight-medium)]">Target</th>
                  <th className="text-left px-3 py-2 font-[var(--weight-medium)]">Governor rule</th>
                  <th className="text-left px-4 py-2 font-[var(--weight-medium)]">Reason</th>
                </tr>
              </thead>
              <tbody>
                {filteredDecisions.map((d, i) => (
                  <tr key={d.id ?? i} className="border-b border-[color:var(--line-subtle)] last:border-0">
                    <td className="px-4 py-2 text-[length:var(--text-xs)] font-[var(--weight-bold)] text-[color:var(--text-secondary)] tracking-wide">{d.action ?? 'NO_ACTION'}</td>
                    <td className="px-3 py-2 num text-[color:var(--text-primary)]">{d.symbol ?? '—'}</td>
                    <td className="px-3 py-2 text-right num text-[color:var(--text-muted)]">{d.qty ?? '—'}</td>
                    <td className="px-3 py-2 text-right num text-[color:var(--text-secondary)]">{fmtNum(d.entry)}</td>
                    <td className="px-3 py-2 text-right num text-[color:var(--down)]">{fmtNum(d.stop)}</td>
                    <td className="px-3 py-2 text-right num text-[color:var(--up)]">{fmtNum(d.target)}</td>
                    <td className="px-3 py-2 num text-[color:var(--text-muted)]">{d.governor_rule ?? '—'}</td>
                    <td className="px-4 py-2 text-[color:var(--text-muted)] truncate max-w-[260px]">{d.reason ?? '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Section>

      {/* Discrepancies */}
      <Section
        title="Discrepancies"
        action={
          <button onClick={loadDiscrepancies} className="text-[length:var(--text-xs)] text-[color:var(--text-muted)] hover:text-[color:var(--text-secondary)] transition-colors">
            {discLoading ? 'LOADING…' : '↻ REFRESH'}
          </button>
        }
      >
        {discrepancies.length === 0 ? (
          <p className="p-4 text-[length:var(--text-sm)] text-[color:var(--text-muted)]">No discrepancies logged.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-[length:var(--text-sm)]">
              <thead>
                <tr className="border-b border-[color:var(--line-subtle)] text-[length:var(--text-xs)] text-[color:var(--text-muted)] uppercase tracking-wide">
                  <th className="text-left px-4 py-2 font-[var(--weight-medium)]">Kind</th>
                  <th className="text-left px-3 py-2 font-[var(--weight-medium)]">Symbol</th>
                  <th className="text-left px-3 py-2 font-[var(--weight-medium)]">Expected</th>
                  <th className="text-left px-3 py-2 font-[var(--weight-medium)]">Actual</th>
                  <th className="text-left px-4 py-2 font-[var(--weight-medium)]">Resolution</th>
                </tr>
              </thead>
              <tbody>
                {discrepancies.map((d, i) => (
                  <tr key={i} className="border-b border-[color:var(--line-subtle)] last:border-0">
                    <td className="px-4 py-2 num text-[color:var(--text-secondary)]">{d.kind ?? '—'}</td>
                    <td className="px-3 py-2 num text-[color:var(--text-primary)]">{d.symbol ?? '—'}</td>
                    <td className="px-3 py-2 num text-[color:var(--text-muted)]">{d.expected ?? '—'}</td>
                    <td className="px-3 py-2 num text-[color:var(--text-muted)]">{d.actual ?? '—'}</td>
                    <td className="px-4 py-2 text-[color:var(--text-muted)]">{d.resolution ?? '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Section>
    </div>
  )
}

// ── Page ─────────────────────────────────────────────────────────────────────

type Tab = 'account' | 'auto-trader'

export default function PaperTradingPage() {
  const [tab, setTab] = useState<Tab>('account')

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-[length:var(--text-xl)] font-[var(--weight-bold)] text-[color:var(--text-primary)] tracking-tight">PAPER</h1>
        <p className="mt-0.5 text-[length:var(--text-xs)] text-[color:var(--text-muted)]">
          The mock account, and the auto-trader engine that trades into it.
        </p>
      </div>

      <div className="flex gap-2">
        {([
          { id: 'account' as const, label: 'Account' },
          { id: 'auto-trader' as const, label: 'Auto-trader' },
        ]).map(t => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`px-4 py-2 rounded-[var(--radius-lg)] text-[length:var(--text-sm)] font-[var(--weight-bold)] transition-colors ${
              tab === t.id ? 'bg-[color:var(--bg-raised)] text-[color:var(--accent)]' : 'bg-[color:var(--bg-panel)] text-[color:var(--text-muted)] hover:text-[color:var(--text-secondary)] hover:bg-[color:var(--bg-raised)]'
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {tab === 'account' ? <AccountTab /> : <AutoTraderTab />}
    </div>
  )
}
