// frontend/src/pages/TradesPage.tsx
import { useState, useCallback, useRef, useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import api from '../api/client'

// ── Types ────────────────────────────────────────────────────────────────────

interface TradeRec {
  symbol: string
  profile: string
  verdict: string       // BUY | SELL | HOLD | NO_TRADE
  direction: string     // LONG | SHORT | NO_TRADE
  score: number
  confidence: string
  entry: number
  stop: number
  target: number
  risk_reward: number
  level_source: string
  position_size: number
  position_value: number
  risk_amount: number
  projected_profit: number
  projected_profit_pct: number
  entry_notes: string[]
  rejection_reason: string
  components: Record<string, number>
  bonuses: string[]
}

interface ScanResult {
  scan_time: number
  symbols_scanned: number
  profile: string
  account_size: number
  top_trades: TradeRec[]
  watchlist: TradeRec[]
  no_trade: TradeRec[]
  errors: { symbol: string; error: string }[]
}

interface LookupResult extends TradeRec {
  current_price: number
  component_details: Record<string, any>
}

// ── Progress types ──────────────────────────────────────────────────────────

interface ScanProgress {
  completed: number
  total: number
  symbol: string
  verdict: string
  score: number
}

// ── Hooks ────────────────────────────────────────────────────────────────────

function useStreamingScan() {
  const [progress, setProgress] = useState<ScanProgress | null>(null)
  const [results, setResults] = useState<ScanResult | null>(null)
  const [isScanning, setIsScanning] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [recentSymbols, setRecentSymbols] = useState<ScanProgress[]>([])
  const abortRef = useRef<AbortController | null>(null)

  const startScan = useCallback((profile: string, account: number, symbols?: string) => {
    // Abort previous scan if running
    if (abortRef.current) abortRef.current.abort()
    const controller = new AbortController()
    abortRef.current = controller

    setIsScanning(true)
    setError(null)
    setProgress(null)
    setResults(null)
    setRecentSymbols([])

    const params = new URLSearchParams({
      profile, account: String(account), max_symbols: '30',
    })
    if (symbols) params.set('symbols', symbols)

    const baseUrl = (api.defaults.baseURL || '/api').replace(/\/$/, '')
    const url = `${baseUrl}/trade-signals/scan/stream?${params}`

    fetch(url, { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`)
        const reader = response.body?.getReader()
        if (!reader) throw new Error('No response body')
        const decoder = new TextDecoder()
        let buffer = ''

        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          buffer += decoder.decode(value, { stream: true })

          // Parse SSE lines
          const lines = buffer.split('\n')
          buffer = lines.pop() || ''
          for (const line of lines) {
            if (!line.startsWith('data: ')) continue
            try {
              const event = JSON.parse(line.slice(6))
              if (event.type === 'progress') {
                setProgress(event)
                setRecentSymbols(prev => [...prev.slice(-29), event])
              } else if (event.type === 'result') {
                setResults(event)
              }
            } catch {}
          }
        }
        setIsScanning(false)
      })
      .catch((err) => {
        if (err.name !== 'AbortError') {
          setError(err.message || 'Scan failed')
          setIsScanning(false)
        }
      })
  }, [])

  const abort = useCallback(() => {
    abortRef.current?.abort()
    setIsScanning(false)
  }, [])

  return { progress, results, isScanning, error, recentSymbols, startScan, abort }
}

function useTradeLookup(symbol: string, profile: string, account: number) {
  return useQuery<LookupResult>({
    queryKey: ['trade-signals', 'lookup', symbol, profile, account],
    queryFn: () => api.get(`/trade-signals/lookup/${symbol}?profile=${profile}&account=${account}`).then(r => r.data),
    enabled: !!symbol && symbol.length >= 1,
    staleTime: 60_000,
    retry: false,
  })
}

// ── Verdict Badge ────────────────────────────────────────────────────────────

function VerdictBadge({ verdict, score }: { verdict: string; score: number }) {
  const colors: Record<string, string> = {
    BUY:      'bg-emerald-500/20 text-emerald-400 border-emerald-500/40',
    SELL:     'bg-red-500/20 text-red-400 border-red-500/40',
    HOLD:     'bg-amber-500/20 text-amber-400 border-amber-500/40',
    NO_TRADE: 'bg-gray-700/30 text-gray-500 border-gray-600/40',
  }
  return (
    <span className={`px-3 py-1 rounded-full border text-xs font-bold tracking-wider ${colors[verdict] || colors.NO_TRADE}`}>
      {verdict} {score.toFixed(0)}
    </span>
  )
}

// ── Score Bar ────────────────────────────────────────────────────────────────

function ScoreBar({ score }: { score: number }) {
  const color = score >= 85 ? 'bg-emerald-400' : score >= 70 ? 'bg-emerald-600' : score >= 55 ? 'bg-amber-500' : 'bg-gray-600'
  return (
    <div className="w-24 h-1.5 bg-gray-800 rounded-full overflow-hidden">
      <div className={`h-full rounded-full ${color}`} style={{ width: `${Math.min(100, score)}%` }} />
    </div>
  )
}

// ── Trade Card (scanner results) ─────────────────────────────────────────────

function TradeCard({ rec }: { rec: TradeRec }) {
  const [expanded, setExpanded] = useState(false)
  const isActionable = rec.verdict === 'BUY' || rec.verdict === 'SELL'

  return (
    <div
      className={`border rounded-lg p-4 cursor-pointer transition-all ${
        isActionable
          ? 'border-emerald-500/30 bg-gray-900/80 hover:border-emerald-400/50'
          : 'border-gray-800/60 bg-gray-950/50 hover:border-gray-700'
      }`}
      onClick={() => setExpanded(!expanded)}
    >
      {/* Header row */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="text-white font-bold text-lg font-mono">{rec.symbol}</span>
          <VerdictBadge verdict={rec.verdict} score={rec.score} />
          <span className="text-gray-500 text-xs uppercase">{rec.profile}</span>
        </div>
        <div className="flex items-center gap-4">
          {isActionable && (
            <span className="text-emerald-400 font-mono text-sm">
              +${rec.projected_profit.toFixed(0)} ({rec.projected_profit_pct.toFixed(1)}%)
            </span>
          )}
          <ScoreBar score={rec.score} />
        </div>
      </div>

      {/* Entry/Stop/Target row */}
      {isActionable && (
        <div className="mt-3 grid grid-cols-5 gap-2 text-xs font-mono">
          <div>
            <span className="text-gray-500 block">Entry</span>
            <span className="text-white">${rec.entry.toFixed(2)}</span>
          </div>
          <div>
            <span className="text-gray-500 block">Stop</span>
            <span className="text-red-400">${rec.stop.toFixed(2)}</span>
          </div>
          <div>
            <span className="text-gray-500 block">Target</span>
            <span className="text-emerald-400">${rec.target.toFixed(2)}</span>
          </div>
          <div>
            <span className="text-gray-500 block">R:R</span>
            <span className="text-white">{rec.risk_reward.toFixed(1)}:1</span>
          </div>
          <div>
            <span className="text-gray-500 block">Size</span>
            <span className="text-white">{rec.position_size} shr</span>
          </div>
        </div>
      )}

      {/* Rejection reason for NO_TRADE */}
      {rec.verdict === 'NO_TRADE' && rec.rejection_reason && (
        <p className="mt-2 text-gray-500 text-xs">{rec.rejection_reason}</p>
      )}

      {/* Expanded details */}
      {expanded && (
        <div className="mt-4 pt-3 border-t border-gray-800/60 space-y-2">
          {rec.entry_notes.length > 0 && (
            <div className="text-xs text-gray-400">
              {rec.entry_notes.map((note, i) => <p key={i}>• {note}</p>)}
            </div>
          )}
          {rec.bonuses.length > 0 && (
            <div className="text-xs text-amber-400">
              {rec.bonuses.map((b, i) => <p key={i}>★ {b}</p>)}
            </div>
          )}
          <div className="text-xs text-gray-600 font-mono">
            Risk: ${rec.risk_amount.toFixed(0)} | Value: ${rec.position_value.toFixed(0)} | Source: {rec.level_source}
          </div>
        </div>
      )}
    </div>
  )
}

// ── Lookup Panel ─────────────────────────────────────────────────────────────

function LookupPanel({ profile, account }: { profile: string; account: number }) {
  const [input, setInput] = useState('')
  const [symbol, setSymbol] = useState('')

  const { data, isLoading, error } = useTradeLookup(symbol, profile, account)

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    setSymbol(input.toUpperCase().trim())
  }

  return (
    <div className="space-y-4">
      <form onSubmit={handleSubmit} className="flex gap-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Enter ticker (e.g., AAPL)"
          className="flex-1 px-4 py-2 bg-gray-900 border border-gray-700 rounded-lg text-white font-mono
                     placeholder-gray-600 focus:border-emerald-500 focus:outline-none"
        />
        <button
          type="submit"
          disabled={!input.trim() || isLoading}
          className="px-6 py-2 bg-emerald-600 hover:bg-emerald-500 disabled:bg-gray-700 text-white
                     rounded-lg font-bold text-sm transition-colors"
        >
          {isLoading ? 'Analyzing...' : 'ANALYZE'}
        </button>
      </form>

      {error && (
        <div className="text-red-400 text-sm p-3 bg-red-500/10 border border-red-500/20 rounded-lg">
          Failed to analyze {symbol}: {(error as any)?.response?.data?.detail || 'Unknown error'}
        </div>
      )}

      {data && (
        <div className="border border-gray-800 rounded-xl p-6 bg-gray-900/80 space-y-6">
          {/* Header */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <h2 className="text-2xl font-bold text-white font-mono">{data.symbol}</h2>
              <span className="text-gray-400 font-mono">${data.current_price?.toFixed(2)}</span>
              <VerdictBadge verdict={data.verdict} score={data.score} />
            </div>
            <span className="text-gray-500 text-xs uppercase">{data.profile}</span>
          </div>

          {/* Main verdict */}
          {(data.verdict === 'BUY' || data.verdict === 'SELL') ? (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-gray-950 rounded-lg p-4 text-center">
                <div className="text-gray-500 text-xs mb-1">Entry</div>
                <div className="text-white font-mono text-xl">${data.entry.toFixed(2)}</div>
              </div>
              <div className="bg-gray-950 rounded-lg p-4 text-center">
                <div className="text-gray-500 text-xs mb-1">Stop Loss</div>
                <div className="text-red-400 font-mono text-xl">${data.stop.toFixed(2)}</div>
              </div>
              <div className="bg-gray-950 rounded-lg p-4 text-center">
                <div className="text-gray-500 text-xs mb-1">Target</div>
                <div className="text-emerald-400 font-mono text-xl">${data.target.toFixed(2)}</div>
              </div>
              <div className="bg-gray-950 rounded-lg p-4 text-center">
                <div className="text-gray-500 text-xs mb-1">Profit on ${account.toLocaleString()}</div>
                <div className="text-emerald-400 font-mono text-xl">
                  +${data.projected_profit.toFixed(0)}
                  <span className="text-sm text-gray-500 ml-1">({data.projected_profit_pct.toFixed(1)}%)</span>
                </div>
              </div>
            </div>
          ) : (
            <div className="bg-gray-950 rounded-lg p-6 text-center">
              <div className="text-gray-500 text-lg">{data.rejection_reason || 'Not tradeable right now'}</div>
            </div>
          )}

          {/* Position details */}
          {(data.verdict === 'BUY' || data.verdict === 'SELL') && (
            <div className="flex gap-6 text-xs font-mono text-gray-400">
              <span>R:R {data.risk_reward.toFixed(1)}:1</span>
              <span>Size: {data.position_size} shares</span>
              <span>Value: ${data.position_value.toFixed(0)}</span>
              <span>Risk: ${data.risk_amount.toFixed(0)}</span>
              <span>Source: {data.level_source}</span>
            </div>
          )}

          {/* Entry notes */}
          {data.entry_notes?.length > 0 && (
            <div className="text-sm text-gray-400 space-y-1">
              {data.entry_notes.map((note, i) => <p key={i}>• {note}</p>)}
            </div>
          )}

          {/* Bonuses */}
          {data.bonuses?.length > 0 && (
            <div className="text-sm text-amber-400 space-y-1">
              {data.bonuses.map((b, i) => <p key={i}>★ {b}</p>)}
            </div>
          )}

          {/* Component breakdown (collapsible) */}
          <details className="text-xs">
            <summary className="text-gray-600 cursor-pointer hover:text-gray-400">Component Scores</summary>
            <div className="mt-2 grid grid-cols-3 gap-2">
              {Object.entries(data.components || {}).map(([key, val]) => (
                <div key={key} className="flex items-center justify-between bg-gray-950 rounded px-2 py-1">
                  <span className="text-gray-500">{key.replace(/_/g, ' ')}</span>
                  <span className={`font-mono ${(val as number) > 0.65 ? 'text-emerald-400' : (val as number) < 0.35 ? 'text-red-400' : 'text-gray-400'}`}>
                    {(val as number).toFixed(2)}
                  </span>
                </div>
              ))}
            </div>
          </details>
        </div>
      )}
    </div>
  )
}

// ── Main Page ────────────────────────────────────────────────────────────────

type Mode = 'scanner' | 'lookup'
type Profile = 'SCALP' | 'SWING' | 'POSITION'

// ── Scan Progress Bar ───────────────────────────────────────────────────────

function ScanProgressBar({ progress, recentSymbols }: { progress: ScanProgress; recentSymbols: ScanProgress[] }) {
  const pct = Math.round((progress.completed / progress.total) * 100)
  const elapsed = recentSymbols.length > 0 ? recentSymbols.length : 1
  const avgPerSymbol = elapsed > 1 ? 1 : 1.5 // rough estimate in seconds
  const remaining = Math.max(0, progress.total - progress.completed)
  const eta = Math.ceil(remaining * avgPerSymbol)

  // Count results so far
  const buys = recentSymbols.filter(s => s.verdict === 'BUY' || s.verdict === 'SELL').length
  const holds = recentSymbols.filter(s => s.verdict === 'HOLD').length

  return (
    <div className="space-y-3">
      {/* Main progress bar */}
      <div className="relative">
        <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
          <div
            className="h-full bg-emerald-500 rounded-full transition-all duration-300 ease-out"
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>

      {/* Stats row */}
      <div className="flex items-center justify-between text-xs font-mono">
        <div className="flex items-center gap-3">
          <span className="text-white">{progress.completed}/{progress.total}</span>
          <span className="text-gray-600">|</span>
          <span className="text-emerald-400">{buys} trades</span>
          {holds > 0 && <span className="text-amber-400">{holds} watchlist</span>}
        </div>
        <div className="flex items-center gap-3">
          <span className="text-gray-400">Scanning <span className="text-white">{progress.symbol}</span></span>
          {remaining > 0 && <span className="text-gray-600">~{eta}s left</span>}
        </div>
      </div>

      {/* Recent symbol ticker */}
      <div className="flex flex-wrap gap-1.5">
        {recentSymbols.slice(-15).map((s, i) => {
          const color = s.verdict === 'BUY' || s.verdict === 'SELL'
            ? 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30'
            : s.verdict === 'HOLD'
            ? 'bg-amber-500/20 text-amber-400 border-amber-500/30'
            : s.verdict === 'ERROR'
            ? 'bg-red-500/20 text-red-400 border-red-500/30'
            : 'bg-gray-800/50 text-gray-500 border-gray-700/30'
          return (
            <span key={`${s.symbol}-${i}`} className={`px-2 py-0.5 rounded border text-[10px] font-mono ${color}`}>
              {s.symbol} {s.score > 0 ? s.score.toFixed(0) : ''}
            </span>
          )
        })}
      </div>
    </div>
  )
}

export default function TradesPage() {
  const [mode, setMode] = useState<Mode>('lookup')
  const [profile, setProfile] = useState<Profile>('SWING')
  const [account, setAccount] = useState(10000)

  const { progress, results: scanData, isScanning, error: scanError, recentSymbols, startScan, abort } = useStreamingScan()

  const handleScanStart = () => {
    setMode('scanner')
    startScan(profile, account)
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-white">TRADES</h1>
          <p className="text-gray-500 text-xs mt-0.5">Precision entries powered by 18-component SMC scoring</p>
        </div>
        <div className="flex items-center gap-3">
          {/* Profile selector */}
          <select
            value={profile}
            onChange={(e) => setProfile(e.target.value as Profile)}
            className="px-3 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm text-white font-mono"
          >
            <option value="SCALP">SCALP (5-15m)</option>
            <option value="SWING">SWING (1H-4H)</option>
            <option value="POSITION">POSITION (Daily)</option>
          </select>

          {/* Account size */}
          <input
            type="number"
            value={account}
            onChange={(e) => setAccount(Number(e.target.value))}
            className="w-28 px-3 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm text-white font-mono"
          />
        </div>
      </div>

      {/* Mode tabs */}
      <div className="flex gap-1 bg-gray-900/50 p-1 rounded-lg w-fit">
        {(['lookup', 'scanner'] as Mode[]).map((m) => (
          <button
            key={m}
            onClick={() => { if (m === 'scanner') handleScanStart(); else setMode(m) }}
            className={`px-4 py-1.5 rounded text-xs font-bold uppercase transition-colors ${
              mode === m ? 'bg-emerald-600 text-white' : 'text-gray-500 hover:text-gray-300'
            }`}
          >
            {m === 'lookup' ? 'Symbol Lookup' : 'Market Scan'}
          </button>
        ))}
      </div>

      {/* Content */}
      {mode === 'lookup' ? (
        <LookupPanel profile={profile} account={account} />
      ) : (
        <div className="space-y-4">
          {/* Scan controls */}
          <div className="flex items-center gap-3">
            <button
              onClick={() => startScan(profile, account)}
              disabled={isScanning}
              className="px-4 py-1.5 bg-emerald-600 hover:bg-emerald-500 disabled:bg-gray-700 text-white
                         rounded text-xs font-bold transition-colors"
            >
              {isScanning ? 'SCANNING...' : 'RUN SCAN'}
            </button>
            {isScanning && (
              <button
                onClick={abort}
                className="px-4 py-1.5 bg-red-600/20 hover:bg-red-600/40 text-red-400
                           border border-red-500/30 rounded text-xs font-bold transition-colors"
              >
                ABORT
              </button>
            )}
          </div>

          {/* Progress bar */}
          {isScanning && progress && (
            <ScanProgressBar progress={progress} recentSymbols={recentSymbols} />
          )}

          {scanError && (
            <div className="text-red-400 text-sm p-3 bg-red-500/10 border border-red-500/20 rounded-lg">
              Scan failed: {scanError}
            </div>
          )}

          {scanData && (
            <>
              <div className="flex items-center gap-4 text-xs text-gray-500 font-mono">
                <span>{scanData.symbols_scanned} symbols scanned</span>
                <span>{scanData.scan_time}s</span>
                <span>{scanData.top_trades.length} actionable trades</span>
                <span>{scanData.watchlist.length} watchlist</span>
              </div>

              {scanData.top_trades.length > 0 && (
                <div>
                  <h3 className="text-emerald-400 text-sm font-bold mb-2 uppercase tracking-wider">
                    Actionable Trades ({scanData.top_trades.length})
                  </h3>
                  <div className="space-y-2">
                    {scanData.top_trades.map((rec) => <TradeCard key={rec.symbol} rec={rec} />)}
                  </div>
                </div>
              )}

              {scanData.watchlist.length > 0 && (
                <div>
                  <h3 className="text-amber-400 text-sm font-bold mb-2 uppercase tracking-wider">
                    Watchlist ({scanData.watchlist.length})
                  </h3>
                  <div className="space-y-2">
                    {scanData.watchlist.map((rec) => <TradeCard key={rec.symbol} rec={rec} />)}
                  </div>
                </div>
              )}

              {scanData.top_trades.length === 0 && scanData.watchlist.length === 0 && (
                <div className="text-center py-8">
                  <div className="text-gray-500 text-sm">No actionable trades found in this scan</div>
                  <div className="text-gray-600 text-xs mt-1">
                    {scanData.no_trade.length} symbols rejected by hard gates (MTF alignment, exhaustion, etc.)
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  )
}
