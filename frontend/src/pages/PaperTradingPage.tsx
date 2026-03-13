/**
 * Paper Trading — simulated trading with full persistence, token-gated resets,
 * configurable starting balance, and pre-fill from the Intelligence page.
 *
 * State is persisted in localStorage so positions survive refresh.
 * Resets cost tokens (doubling cost per reset within a 30-day window).
 * Starting balance is configurable: $10K · $25K · $100K · $250K · $1M · custom.
 */

import { useState, useEffect, useCallback } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'

// ── Types ─────────────────────────────────────────────────────────────────────

interface PaperOrder {
  id:          number
  symbol:      string
  direction:   'buy' | 'sell'
  size:        number
  entry:       number
  stop:        number
  target:      number
  timestamp:   string
  status:      'open' | 'closed'
  exitPrice?:  number
  exitTime?:   string
  pnl?:        number
  stageScore?: number   // confluence score at time of entry
}

interface ResetRecord {
  timestamp:        string
  balanceAtReset:   number
  tokensSpent:      number
  winRate:          string
}

interface PaperAccount {
  startingBalance:  number
  currentBalance:   number
  tokens:           number
  resetCount:       number           // lifetime
  resetsThisWindow: number           // within rolling 30-day window
  windowStart:      string           // ISO date of first reset in current window
  resetHistory:     ResetRecord[]
  orders:           PaperOrder[]
  nextId:           number
  createdAt:        string
}

// ── Constants ─────────────────────────────────────────────────────────────────

const ACCOUNT_KEY   = 'roboalgo_paper_account'
const BALANCE_OPTS  = [10_000, 25_000, 100_000, 250_000, 1_000_000]
const INITIAL_TOKENS = 10
const MONTHLY_TOKENS =  5
const WINDOW_DAYS    = 30

function resetCost(resetsThisWindow: number): number {
  // 1st reset = 3 tokens, 2nd = 6, 3rd = 12, 4th = 24 …
  return 3 * Math.pow(2, resetsThisWindow)
}

function defaultAccount(balance: number): PaperAccount {
  return {
    startingBalance:  balance,
    currentBalance:   balance,
    tokens:           INITIAL_TOKENS,
    resetCount:       0,
    resetsThisWindow: 0,
    windowStart:      new Date().toISOString(),
    resetHistory:     [],
    orders:           [],
    nextId:           1,
    createdAt:        new Date().toISOString(),
  }
}

function loadAccount(): PaperAccount | null {
  try {
    const raw = localStorage.getItem(ACCOUNT_KEY)
    if (!raw) return null
    return JSON.parse(raw) as PaperAccount
  } catch { return null }
}

function saveAccount(acct: PaperAccount) {
  localStorage.setItem(ACCOUNT_KEY, JSON.stringify(acct))
}

function isWindowExpired(windowStart: string): boolean {
  const start = new Date(windowStart).getTime()
  const now   = Date.now()
  return (now - start) > WINDOW_DAYS * 24 * 60 * 60 * 1000
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function fmt(n: number | undefined | null, d = 2) {
  if (n == null || isNaN(n)) return '—'
  return n.toFixed(d)
}

function fmtDollar(n: number) {
  if (n >= 1_000_000) return `$${(n / 1_000_000).toFixed(2)}M`
  if (n >= 1_000)     return `$${(n / 1_000).toFixed(1)}K`
  return `$${n.toFixed(2)}`
}

// ── Account Setup ─────────────────────────────────────────────────────────────

function AccountSetup({ onCreate }: { onCreate: (balance: number) => void }) {
  const [selected, setSelected] = useState<number>(100_000)
  const [custom,   setCustom]   = useState('')
  const [useCustom, setUseCustom] = useState(false)

  function handleCreate() {
    const bal = useCustom ? parseFloat(custom.replace(/,/g, '')) : selected
    if (!bal || isNaN(bal) || bal < 1000) return
    onCreate(bal)
  }

  return (
    <div className="flex flex-col items-center justify-center min-h-[60vh] space-y-8">
      <div className="text-center">
        <h2 className="text-2xl font-black text-white">Set Your Paper Account</h2>
        <p className="text-sm text-gray-500 mt-1">Choose a starting balance — all trades tracked from here</p>
      </div>

      <div className="bg-gray-900 border border-gray-800 rounded-2xl p-8 max-w-sm w-full space-y-6">
        <div className="space-y-2">
          <p className="text-xs text-gray-500 uppercase tracking-widest">Starting Balance</p>
          <div className="grid grid-cols-3 gap-2">
            {BALANCE_OPTS.map(b => (
              <button key={b}
                onClick={() => { setSelected(b); setUseCustom(false) }}
                className={`py-2.5 rounded-xl text-sm font-bold transition-all ${
                  selected === b && !useCustom
                    ? 'bg-emerald-700 text-white ring-1 ring-emerald-500'
                    : 'bg-gray-800 text-gray-400 hover:text-gray-200'
                }`}>
                {fmtDollar(b)}
              </button>
            ))}
            <button
              onClick={() => setUseCustom(true)}
              className={`py-2.5 rounded-xl text-sm font-bold transition-all col-span-3 ${
                useCustom ? 'bg-gray-700 text-gray-200 ring-1 ring-gray-500' : 'bg-gray-800 text-gray-500 hover:text-gray-300'
              }`}>
              Custom
            </button>
          </div>
          {useCustom && (
            <input
              autoFocus
              value={custom}
              onChange={e => setCustom(e.target.value)}
              placeholder="Enter amount, e.g. 50000"
              className="w-full mt-2 bg-gray-800 border border-gray-700 rounded-xl px-3 py-2 text-sm font-mono text-gray-200 focus:outline-none focus:border-emerald-500"
            />
          )}
        </div>

        <div className="bg-gray-800/50 rounded-xl p-3 text-xs text-gray-500 space-y-1">
          <p>● {INITIAL_TOKENS} tokens included — used for account resets</p>
          <p>● Reset cost doubles each time within 30 days</p>
          <p>● All trades tracked — history preserved across resets</p>
        </div>

        <button
          onClick={handleCreate}
          className="w-full py-3 bg-emerald-700 hover:bg-emerald-600 text-white font-bold rounded-xl transition-colors text-sm"
        >
          Create Paper Account →
        </button>
      </div>
    </div>
  )
}

// ── Close Position Modal ───────────────────────────────────────────────────────

function CloseModal({
  order, onClose, onCancel,
}: {
  order: PaperOrder
  onClose: (exitPrice: number) => void
  onCancel: () => void
}) {
  const [exitPrice, setExitPrice] = useState(
    order.direction === 'buy' ? order.target : order.stop
  )
  const pnl = (exitPrice - order.entry) * order.size * (order.direction === 'buy' ? 1 : -1)

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70">
      <div className="bg-gray-900 border border-gray-700 rounded-2xl p-6 w-full max-w-sm space-y-4">
        <h3 className="text-sm font-bold text-gray-200">Close {order.symbol} Position</h3>
        <div className="space-y-1 text-xs text-gray-500">
          <div className="flex justify-between">
            <span>Entry</span><span className="font-mono text-gray-300">${fmt(order.entry)}</span>
          </div>
          <div className="flex justify-between">
            <span>Size</span><span className="font-mono text-gray-300">{order.size} shares</span>
          </div>
        </div>
        <div>
          <label className="text-xs text-gray-500 block mb-1">Exit Price</label>
          <input
            type="number" step="0.01"
            value={exitPrice || ''}
            onChange={e => setExitPrice(parseFloat(e.target.value) || 0)}
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm font-mono text-gray-200 focus:outline-none focus:border-emerald-500"
          />
        </div>
        <div className={`text-center text-xl font-bold font-mono ${pnl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
          {pnl >= 0 ? '+' : ''}${fmt(pnl)}
        </div>
        <div className="flex gap-2">
          <button onClick={onCancel} className="flex-1 py-2 bg-gray-800 text-gray-400 rounded-xl text-sm hover:bg-gray-700 transition-colors">Cancel</button>
          <button
            onClick={() => onClose(exitPrice)}
            className="flex-1 py-2 bg-emerald-700 hover:bg-emerald-600 text-white font-bold rounded-xl text-sm transition-colors"
          >
            Confirm Close
          </button>
        </div>
      </div>
    </div>
  )
}

// ── Reset Modal ───────────────────────────────────────────────────────────────

function ResetModal({
  account, onReset, onCancel,
}: {
  account: PaperAccount
  onReset: () => void
  onCancel: () => void
}) {
  const cost     = resetCost(account.resetsThisWindow)
  const canAfford = account.tokens >= cost
  const closed   = account.orders.filter(o => o.status === 'closed')
  const wins     = closed.filter(o => (o.pnl ?? 0) > 0).length
  const winRate  = closed.length > 0 ? `${Math.round(wins / closed.length * 100)}%` : 'N/A'

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70">
      <div className="bg-gray-900 border border-gray-700 rounded-2xl p-6 w-full max-w-sm space-y-4">
        <h3 className="text-sm font-bold text-red-400">Reset Paper Account</h3>
        <div className="bg-gray-800 rounded-xl p-4 space-y-2 text-xs">
          <div className="flex justify-between text-gray-400">
            <span>Current Balance</span>
            <span className={`font-mono font-bold ${account.currentBalance >= account.startingBalance ? 'text-emerald-400' : 'text-red-400'}`}>
              {fmtDollar(account.currentBalance)}
            </span>
          </div>
          <div className="flex justify-between text-gray-400">
            <span>Win Rate (this session)</span>
            <span className="font-mono text-gray-200">{winRate}</span>
          </div>
          <div className="flex justify-between text-gray-400">
            <span>Lifetime Resets</span>
            <span className="font-mono text-gray-200">{account.resetCount}</span>
          </div>
          <div className="flex justify-between text-gray-400">
            <span>Resets this 30-day window</span>
            <span className="font-mono text-amber-400">{account.resetsThisWindow}</span>
          </div>
          <div className="border-t border-gray-700 pt-2 flex justify-between">
            <span className="text-red-400 font-semibold">Token cost</span>
            <span className="font-mono font-bold text-red-400">{cost} tokens</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Your tokens</span>
            <span className={`font-mono font-bold ${canAfford ? 'text-gray-200' : 'text-red-500'}`}>
              {account.tokens} tokens
            </span>
          </div>
        </div>
        {!canAfford && (
          <p className="text-xs text-red-400 text-center">Not enough tokens. Trade more to earn tokens.</p>
        )}
        <p className="text-xs text-gray-600 text-center">Your trade history will be preserved. Balance and open positions reset.</p>
        <div className="flex gap-2">
          <button onClick={onCancel} className="flex-1 py-2 bg-gray-800 text-gray-400 rounded-xl text-sm hover:bg-gray-700 transition-colors">Cancel</button>
          <button
            onClick={onReset}
            disabled={!canAfford}
            className="flex-1 py-2 bg-red-800 hover:bg-red-700 text-white font-bold rounded-xl text-sm transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
          >
            Reset ({cost} tokens)
          </button>
        </div>
      </div>
    </div>
  )
}

// ── Quick Trade Panel ─────────────────────────────────────────────────────────

function QuickTradePanel({
  account,
  onSubmit,
  prefill,
}: {
  account: PaperAccount
  onSubmit: (order: Omit<PaperOrder, 'id' | 'timestamp' | 'status'>) => void
  prefill: { symbol?: string; direction?: 'buy' | 'sell'; entry?: number; stop?: number; target?: number }
}) {
  const [symbol,    setSymbol]    = useState(prefill.symbol    ?? 'AAPL')
  const [direction, setDirection] = useState<'buy' | 'sell'>(prefill.direction ?? 'buy')
  const [entry,     setEntry]     = useState(prefill.entry     ?? 0)
  const [stop,      setStop]      = useState(prefill.stop      ?? 0)
  const [target,    setTarget]    = useState(prefill.target    ?? 0)
  const [riskPct,   setRiskPct]   = useState(1.0)  // % of account to risk

  // Update when prefill changes (e.g., navigating from Intel page)
  useEffect(() => {
    if (prefill.symbol)    setSymbol(prefill.symbol)
    if (prefill.direction) setDirection(prefill.direction)
    if (prefill.entry)     setEntry(prefill.entry)
    if (prefill.stop)      setStop(prefill.stop)
    if (prefill.target)    setTarget(prefill.target)
  }, [prefill.symbol, prefill.direction, prefill.entry, prefill.stop, prefill.target])

  // Auto-size based on risk %
  const riskPerShare  = entry > 0 && stop > 0 ? Math.abs(entry - stop) : 0
  const accountRisk   = account.currentBalance * (riskPct / 100)
  const autoSize      = riskPerShare > 0 ? Math.floor(accountRisk / riskPerShare) : 0
  const [size, setSize] = useState(0)
  const effectiveSize = size > 0 ? size : autoSize

  const positionValue = effectiveSize * entry
  const riskAmount    = effectiveSize * riskPerShare
  const rewardAmount  = entry > 0 && target > 0 ? Math.abs(target - entry) * effectiveSize : 0
  const rrRatio       = riskAmount > 0 ? (rewardAmount / riskAmount).toFixed(1) : null

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (!symbol || !entry || !stop || !target || effectiveSize <= 0) return
    onSubmit({ symbol: symbol.toUpperCase(), direction, size: effectiveSize, entry, stop, target, pnl: undefined })
    setEntry(0); setStop(0); setTarget(0); setSize(0)
  }

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 space-y-3">
      <h3 className="text-sm font-semibold text-gray-100">New Paper Trade</h3>

      {prefill.symbol && (
        <div className="flex items-center gap-2 px-3 py-2 bg-emerald-900/20 border border-emerald-800/30 rounded-lg text-xs text-emerald-300">
          ◉ Pre-filled from Intelligence — {prefill.symbol}
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-3">
        {/* Symbol + Direction */}
        <div className="grid grid-cols-2 gap-2">
          <div>
            <label className="text-xs text-gray-500 block mb-1">Symbol</label>
            <input value={symbol} onChange={e => setSymbol(e.target.value.toUpperCase())}
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-2.5 py-1.5 text-sm font-mono text-gray-200 focus:outline-none focus:border-emerald-600"
              placeholder="AAPL" />
          </div>
          <div>
            <label className="text-xs text-gray-500 block mb-1">Direction</label>
            <div className="flex gap-1">
              {(['buy', 'sell'] as const).map(d => (
                <button key={d} type="button" onClick={() => setDirection(d)}
                  className={`flex-1 py-1.5 rounded-lg text-sm font-bold uppercase transition-colors ${
                    direction === d
                      ? d === 'buy' ? 'bg-emerald-700 text-white' : 'bg-red-700 text-white'
                      : 'bg-gray-800 text-gray-500 hover:text-gray-300'
                  }`}>
                  {d === 'buy' ? '▲' : '▼'}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Prices */}
        <div className="grid grid-cols-3 gap-2">
          {[
            { label: 'Entry',  value: entry,  setter: setEntry,  color: '#f3f4f6' },
            { label: 'Stop',   value: stop,   setter: setStop,   color: '#ef4444' },
            { label: 'Target', value: target, setter: setTarget, color: '#22c55e' },
          ].map(({ label, value, setter, color }) => (
            <div key={label}>
              <label className="text-xs block mb-1" style={{ color }}>{label}</label>
              <input type="number" step="0.01" value={value || ''}
                onChange={e => setter(parseFloat(e.target.value) || 0)}
                className="w-full bg-gray-800 border border-gray-700 rounded-lg px-2 py-1.5 text-xs font-mono text-gray-200 focus:outline-none focus:border-emerald-600" />
            </div>
          ))}
        </div>

        {/* Risk % + Size */}
        <div className="grid grid-cols-2 gap-2">
          <div>
            <label className="text-xs text-gray-500 block mb-1">Risk % of account</label>
            <div className="flex gap-1">
              {[0.5, 1, 2, 3].map(r => (
                <button key={r} type="button" onClick={() => setRiskPct(r)}
                  className={`flex-1 py-1 rounded text-xs font-mono transition-colors ${riskPct === r ? 'bg-gray-700 text-gray-200' : 'bg-gray-800 text-gray-500 hover:text-gray-300'}`}>
                  {r}%
                </button>
              ))}
            </div>
          </div>
          <div>
            <label className="text-xs text-gray-500 block mb-1">
              Shares {autoSize > 0 ? <span className="text-gray-600">(auto: {autoSize})</span> : ''}
            </label>
            <input type="number" value={size || ''}
              onChange={e => setSize(parseInt(e.target.value) || 0)}
              placeholder={autoSize > 0 ? String(autoSize) : '0'}
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-2.5 py-1.5 text-xs font-mono text-gray-200 focus:outline-none focus:border-emerald-600" />
          </div>
        </div>

        {/* Metrics */}
        {entry > 0 && stop > 0 && effectiveSize > 0 && (
          <div className="grid grid-cols-3 gap-2 text-xs bg-gray-800/50 rounded-lg px-3 py-2">
            <div>
              <p className="text-gray-600">Position</p>
              <p className="font-mono text-gray-300">${(positionValue / 1000).toFixed(1)}K</p>
            </div>
            <div>
              <p className="text-gray-600">Risk</p>
              <p className="font-mono text-red-400">${fmt(riskAmount, 0)}</p>
            </div>
            <div>
              <p className="text-gray-600">R:R</p>
              <p className="font-mono text-emerald-400">{rrRatio ? `1:${rrRatio}` : '—'}</p>
            </div>
          </div>
        )}

        <button type="submit"
          className={`w-full py-2.5 rounded-xl text-sm font-bold uppercase transition-colors ${
            direction === 'buy' ? 'bg-emerald-700 hover:bg-emerald-600 text-white' : 'bg-red-700 hover:bg-red-600 text-white'
          }`}>
          {direction === 'buy' ? '▲ Place Long' : '▼ Place Short'}
        </button>
      </form>
    </div>
  )
}

// ── Page ─────────────────────────────────────────────────────────────────────

export default function PaperTradingPage() {
  const navigate          = useNavigate()
  const [searchParams]    = useSearchParams()
  const [account, setAccountRaw] = useState<PaperAccount | null>(null)
  const [showSetup,  setShowSetup]  = useState(false)
  const [closeOrder, setCloseOrder] = useState<PaperOrder | null>(null)
  const [showReset,  setShowReset]  = useState(false)

  // Read prefill from URL params (set by Intel page "Paper Trade" button)
  const prefill = {
    symbol:    searchParams.get('symbol')    ?? undefined,
    direction: (searchParams.get('direction') as 'buy' | 'sell') ?? undefined,
    entry:     parseFloat(searchParams.get('entry')  ?? '') || undefined,
    stop:      parseFloat(searchParams.get('stop')   ?? '') || undefined,
    target:    parseFloat(searchParams.get('target') ?? '') || undefined,
  }

  // Load account from localStorage on mount
  useEffect(() => {
    const saved = loadAccount()
    if (saved) {
      // Award monthly tokens if it's been a month since account was created
      // Simple approach: add 5 tokens if tokens < 5 (floor refill)
      const updated = { ...saved }
      if (updated.tokens < MONTHLY_TOKENS) {
        updated.tokens = Math.max(updated.tokens, MONTHLY_TOKENS)
      }
      setAccountRaw(updated)
      saveAccount(updated)
    } else {
      setShowSetup(true)
    }
  }, [])

  const setAccount = useCallback((acct: PaperAccount) => {
    setAccountRaw(acct)
    saveAccount(acct)
  }, [])

  function handleCreate(balance: number) {
    const acct = defaultAccount(balance)
    setAccount(acct)
    setShowSetup(false)
  }

  function handleOrder(order: Omit<PaperOrder, 'id' | 'timestamp' | 'status'>) {
    if (!account) return
    const cost = order.size * order.entry
    if (cost > account.currentBalance) {
      alert(`Insufficient balance. Position costs $${cost.toFixed(0)}, balance is $${account.currentBalance.toFixed(0)}.`)
      return
    }
    const newOrder: PaperOrder = {
      ...order,
      id:        account.nextId,
      timestamp: new Date().toISOString(),
      status:    'open',
    }
    setAccount({
      ...account,
      currentBalance: account.currentBalance - cost,   // reserve capital
      orders:   [...account.orders, newOrder],
      nextId:   account.nextId + 1,
    })
  }

  function handleCloseConfirm(exitPrice: number) {
    if (!account || !closeOrder) return
    const { size, entry, direction } = closeOrder
    const pnl      = (exitPrice - entry) * size * (direction === 'buy' ? 1 : -1)
    const returned = size * entry   // return reserved capital
    const updatedOrders = account.orders.map(o =>
      o.id === closeOrder.id
        ? { ...o, status: 'closed' as const, exitPrice, exitTime: new Date().toISOString(), pnl: parseFloat(pnl.toFixed(2)) }
        : o
    )
    // Earn 1 token for every 5 winning trades (as a reward mechanic)
    const wins    = updatedOrders.filter(o => o.status === 'closed' && (o.pnl ?? 0) > 0).length
    const bonusToken = wins > 0 && wins % 5 === 0 ? 1 : 0
    setAccount({
      ...account,
      currentBalance: account.currentBalance + returned + pnl,
      tokens:         account.tokens + bonusToken,
      orders:         updatedOrders,
    })
    setCloseOrder(null)
  }

  function handleReset() {
    if (!account) return
    const windowExpired = isWindowExpired(account.windowStart)
    const resetsThisWindow = windowExpired ? 0 : account.resetsThisWindow
    const cost  = resetCost(resetsThisWindow)
    if (account.tokens < cost) return

    const closed  = account.orders.filter(o => o.status === 'closed')
    const wins    = closed.filter(o => (o.pnl ?? 0) > 0).length
    const winRate = closed.length > 0 ? `${Math.round(wins / closed.length * 100)}%` : 'N/A'

    const resetRecord: ResetRecord = {
      timestamp:      new Date().toISOString(),
      balanceAtReset: account.currentBalance,
      tokensSpent:    cost,
      winRate,
    }

    setAccount({
      ...account,
      currentBalance:   account.startingBalance,
      tokens:           account.tokens - cost,
      resetCount:       account.resetCount + 1,
      resetsThisWindow: windowExpired ? 1 : resetsThisWindow + 1,
      windowStart:      windowExpired ? new Date().toISOString() : account.windowStart,
      resetHistory:     [...account.resetHistory, resetRecord],
      orders:           account.orders.filter(o => o.status === 'closed'), // keep history, clear open
    })
    setShowReset(false)
  }

  if (showSetup || !account) return <AccountSetup onCreate={handleCreate} />

  const open   = account.orders.filter(o => o.status === 'open')
  const closed = account.orders.filter(o => o.status === 'closed')
  const wins   = closed.filter(o => (o.pnl ?? 0) > 0).length
  const totalPnl = closed.reduce((a, o) => a + (o.pnl ?? 0), 0)
  const pnlPct   = ((account.currentBalance - account.startingBalance) / account.startingBalance) * 100
  const winRate  = closed.length > 0 ? Math.round(wins / closed.length * 100) : 0

  return (
    <div className="space-y-4">
      {/* ── Modals ── */}
      {closeOrder && (
        <CloseModal order={closeOrder} onClose={handleCloseConfirm} onCancel={() => setCloseOrder(null)} />
      )}
      {showReset && (
        <ResetModal account={account} onReset={handleReset} onCancel={() => setShowReset(false)} />
      )}

      {/* ── Header ── */}
      <div className="flex items-center justify-between flex-wrap gap-4">
        <div>
          <h1 className="text-xl font-bold text-gray-100">📋 Paper Trading</h1>
          <p className="text-xs text-gray-500 mt-0.5">
            Simulated — no real money · started {new Date(account.createdAt).toLocaleDateString()}
          </p>
        </div>

        {/* Account stats */}
        <div className="flex items-center gap-4 flex-wrap">
          <div className="text-center">
            <p className="text-xs text-gray-500">Balance</p>
            <p className={`font-mono font-bold text-sm ${account.currentBalance >= account.startingBalance ? 'text-emerald-400' : 'text-red-400'}`}>
              {fmtDollar(account.currentBalance)}
            </p>
          </div>
          <div className="text-center">
            <p className="text-xs text-gray-500">P&L</p>
            <p className={`font-mono font-bold text-sm ${pnlPct >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
              {pnlPct >= 0 ? '+' : ''}{pnlPct.toFixed(1)}%
            </p>
          </div>
          <div className="text-center">
            <p className="text-xs text-gray-500">Win Rate</p>
            <p className="font-mono font-bold text-sm text-gray-200">
              {closed.length > 0 ? `${winRate}% (${wins}/${closed.length})` : '—'}
            </p>
          </div>
          <div className="text-center">
            <p className="text-xs text-gray-500">Tokens</p>
            <p className="font-mono font-bold text-sm text-amber-400">{account.tokens} 🪙</p>
          </div>
          <div className="flex gap-2">
            <button onClick={() => navigate('/intel')}
              className="px-3 py-1.5 bg-gray-800 text-gray-400 hover:text-gray-200 rounded-lg text-xs transition-colors">
              ◉ Find Trade
            </button>
            <button onClick={() => setShowReset(true)}
              className="px-3 py-1.5 bg-gray-800 text-red-500 hover:text-red-400 rounded-lg text-xs transition-colors">
              ⟲ Reset
            </button>
          </div>
        </div>
      </div>

      {/* ── Main layout ── */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">

        {/* Left: Order entry */}
        <div>
          <QuickTradePanel account={account} onSubmit={handleOrder} prefill={prefill} />

          {/* Reset history */}
          {account.resetHistory.length > 0 && (
            <div className="mt-4 bg-gray-900 border border-gray-800 rounded-xl p-4">
              <p className="text-xs text-gray-500 uppercase tracking-widest font-semibold mb-2">Reset History</p>
              <div className="space-y-1.5">
                {account.resetHistory.slice().reverse().slice(0, 5).map((r, i) => (
                  <div key={i} className="flex items-center justify-between text-xs">
                    <span className="text-gray-600">{new Date(r.timestamp).toLocaleDateString()}</span>
                    <span className={`font-mono ${r.balanceAtReset >= account.startingBalance ? 'text-emerald-500' : 'text-red-500'}`}>
                      {fmtDollar(r.balanceAtReset)}
                    </span>
                    <span className="text-gray-600">{r.winRate} WR</span>
                    <span className="text-amber-600">-{r.tokensSpent}🪙</span>
                  </div>
                ))}
              </div>
              <p className="text-xs text-gray-700 mt-2">Lifetime resets: {account.resetCount}</p>
            </div>
          )}
        </div>

        {/* Right: Positions + Journal */}
        <div className="lg:col-span-2 space-y-4">

          {/* Open positions */}
          <div className="bg-gray-900 border border-gray-800 rounded-xl">
            <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800">
              <h3 className="text-sm font-semibold text-gray-100">Open Positions</h3>
              <span className="text-xs text-gray-500">{open.length} open</span>
            </div>
            {open.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-24 text-xs text-gray-600 gap-1">
                <span>No open positions</span>
                <button onClick={() => navigate('/intel')} className="text-emerald-600 hover:text-emerald-400 transition-colors">
                  Find a trade →
                </button>
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-gray-800 text-gray-500 uppercase">
                      <th className="text-left px-4 py-2">Symbol</th>
                      <th className="text-left px-4 py-2">Dir</th>
                      <th className="text-right px-4 py-2">Size</th>
                      <th className="text-right px-4 py-2">Entry</th>
                      <th className="text-right px-4 py-2">Stop</th>
                      <th className="text-right px-4 py-2">Target</th>
                      <th className="text-right px-4 py-2">Date</th>
                      <th className="px-4 py-2" />
                    </tr>
                  </thead>
                  <tbody>
                    {open.map(o => (
                      <tr key={o.id} className="border-b border-gray-800/40 hover:bg-gray-800/20">
                        <td className="px-4 py-2 font-mono font-bold text-gray-100">
                          <button onClick={() => navigate(`/intel?symbol=${o.symbol}`)} className="hover:text-emerald-400">
                            {o.symbol}
                          </button>
                        </td>
                        <td className="px-4 py-2">
                          <span className={`font-bold ${o.direction === 'buy' ? 'text-emerald-400' : 'text-red-400'}`}>
                            {o.direction === 'buy' ? '▲ LONG' : '▼ SHORT'}
                          </span>
                        </td>
                        <td className="px-4 py-2 text-right font-mono text-gray-300">{o.size}</td>
                        <td className="px-4 py-2 text-right font-mono text-gray-200">${fmt(o.entry)}</td>
                        <td className="px-4 py-2 text-right font-mono text-red-400">${fmt(o.stop)}</td>
                        <td className="px-4 py-2 text-right font-mono text-emerald-400">${fmt(o.target)}</td>
                        <td className="px-4 py-2 text-right text-gray-500">
                          {new Date(o.timestamp).toLocaleDateString([], { month: 'short', day: 'numeric' })}
                        </td>
                        <td className="px-4 py-2 flex gap-1">
                          <button onClick={() => navigate(`/chart?symbol=${o.symbol}`)}
                            className="px-2 py-0.5 rounded text-xs bg-gray-800 text-gray-400 hover:text-gray-200 transition-colors">
                            Chart
                          </button>
                          <button onClick={() => setCloseOrder(o)}
                            className="px-2 py-0.5 rounded text-xs bg-gray-800 text-gray-400 hover:bg-red-900/30 hover:text-red-400 transition-colors">
                            Close
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          {/* Trade journal */}
          <div className="bg-gray-900 border border-gray-800 rounded-xl">
            <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800">
              <h3 className="text-sm font-semibold text-gray-100">Trade Journal</h3>
              {closed.length > 0 && (
                <div className="flex items-center gap-3 text-xs">
                  <span className="text-gray-500">
                    W/L: <span className="text-emerald-400 font-mono">{wins}/{closed.length - wins}</span>
                  </span>
                  <span className="text-gray-500">
                    Total: <span className={`font-mono font-bold ${totalPnl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                      {totalPnl >= 0 ? '+' : ''}${totalPnl.toFixed(2)}
                    </span>
                  </span>
                </div>
              )}
            </div>
            {closed.length === 0 ? (
              <div className="flex items-center justify-center h-20 text-xs text-gray-600">
                No closed trades yet
              </div>
            ) : (
              <div className="overflow-x-auto max-h-72">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-gray-800 text-gray-500 uppercase sticky top-0 bg-gray-900">
                      <th className="text-left px-4 py-2">Symbol</th>
                      <th className="text-left px-4 py-2">Dir</th>
                      <th className="text-right px-4 py-2">Entry</th>
                      <th className="text-right px-4 py-2">Exit</th>
                      <th className="text-right px-4 py-2">Size</th>
                      <th className="text-right px-4 py-2">P&L</th>
                      <th className="text-right px-4 py-2">Date</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[...closed].reverse().map(o => {
                      const pnl = o.pnl ?? 0
                      return (
                        <tr key={o.id} className="border-b border-gray-800/40 hover:bg-gray-800/10">
                          <td className="px-4 py-2 font-mono font-bold text-gray-200">
                            <button onClick={() => navigate(`/intel?symbol=${o.symbol}`)} className="hover:text-emerald-400">
                              {o.symbol}
                            </button>
                          </td>
                          <td className="px-4 py-2">
                            <span className={o.direction === 'buy' ? 'text-emerald-400' : 'text-red-400'}>
                              {o.direction === 'buy' ? '▲' : '▼'}
                            </span>
                          </td>
                          <td className="px-4 py-2 text-right font-mono text-gray-400">${fmt(o.entry)}</td>
                          <td className="px-4 py-2 text-right font-mono text-gray-400">${fmt(o.exitPrice)}</td>
                          <td className="px-4 py-2 text-right font-mono text-gray-500">{o.size}</td>
                          <td className="px-4 py-2 text-right font-mono font-bold"
                            style={{ color: pnl >= 0 ? '#22c55e' : '#ef4444' }}>
                            {pnl >= 0 ? '+' : ''}${fmt(pnl)}
                          </td>
                          <td className="px-4 py-2 text-right text-gray-600">
                            {new Date(o.timestamp).toLocaleDateString([], { month: 'short', day: 'numeric' })}
                          </td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
