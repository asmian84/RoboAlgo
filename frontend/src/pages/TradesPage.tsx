// frontend/src/pages/TradesPage.tsx
import { useState, useCallback, useRef, useEffect, useMemo, Fragment } from 'react'
import { createPortal } from 'react-dom'
import { useNavigate } from 'react-router-dom'
import KLineChartOverlay from '../components/chart/KLineChartOverlay'
import PatternChartGrid from '../components/chart/PatternChartGrid'
import { useQuery } from '@tanstack/react-query'
import api from '../api/client'

// ── Types ────────────────────────────────────────────────────────────────────

interface TradeRec {
  symbol: string
  profile: string
  verdict: string       // BUY | HOLD | NO_TRADE
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
  component_details?: Record<string, any>
}

interface ScanResult {
  scan_time: number
  symbols_scanned: number
  profile: string
  account_size: number
  recommendations: TradeRec[]
  top_trades: TradeRec[]
  watchlist: TradeRec[]
  no_trade: TradeRec[]
  errors: { symbol: string; error: string }[]
  regime?: { regime: string; bias: string }
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

interface RegimeInfo {
  regime: string
  bias: string
  hedges: string[]
  spy_trend: string
  spy_price: number
  ma20: number
  ma50: number
}

interface DiscoveryInfo {
  movers_found: number
  hedges_added: number
  total_symbols: number
  regime: string
  bias: string
}

// ── Scan Type definitions (single-select radio-style) ──────────────────────

const SCAN_TYPES = [
  { id: 'all',      label: 'All'         },
  { id: 'bluechip', label: 'Blue Chips'  },
  { id: 'growth',   label: 'Growth'      },
  { id: 'etfs',     label: 'ETFs'        },
  { id: 'penny',    label: 'Penny (<$5)' },
] as const

type ScanTypeId = typeof SCAN_TYPES[number]['id']

// Client-side filtering is now minimal:
// - bluechip/growth/etfs: backend sends the right pool, no client filtering needed
// - penny: client-side price gate (< $5) in tradePassesFilters()


/** Format number with commas: 1000000 → "1,000,000" */
function formatWithCommas(n: number): string {
  return n.toLocaleString('en-US', { maximumFractionDigits: 0 })
}

/** Parse comma-formatted string back to number: "1,000,000" → 1000000 */
function parseCommaNumber(s: string): number {
  const cleaned = s.replace(/[^0-9]/g, '')
  return cleaned ? parseInt(cleaned, 10) : 0
}

/**
 * Recalculate position size + risk for a trade based on current account size.
 * Uses the same logic as the server: 3% max risk, risk-based sizing.
 */
function recalcSize(rec: TradeRec, accountSize: number): { size: number; risk: number; value: number; profit: number } {
  if (!rec.entry || !rec.stop || rec.entry <= 0 || rec.stop <= 0) {
    return { size: 0, risk: 0, value: 0, profit: 0 }
  }
  const riskPerShare = Math.abs(rec.entry - rec.stop)
  if (riskPerShare <= 0) return { size: 0, risk: 0, value: 0, profit: 0 }

  const maxRiskPct = 3.0
  const maxRiskDollars = accountSize * (maxRiskPct / 100)
  const riskBasedSize = Math.floor(maxRiskDollars / riskPerShare)
  const capitalCap = Math.floor(accountSize / rec.entry)
  const size = Math.max(1, Math.min(riskBasedSize, capitalCap))
  const risk = Math.round(size * riskPerShare)
  const value = Math.round(size * rec.entry)
  let profit = 0
  if (rec.target > 0) {
    if (rec.direction === 'LONG') {
      profit = Math.round(size * (rec.target - rec.entry))
    } else {
      profit = Math.round(size * (rec.entry - rec.target))
    }
  }
  return { size, risk, value, profit }
}

/** Client-side filter: does this trade pass the current scan type selection? */
function tradePassesFilters(trade: TradeRec, selectedCats: string[]): boolean {
  // The backend already filters the universe by category (bluechip/growth/etfs/penny).
  // Client-side only needs to double-check penny (price-based, since prices drift).
  // For all other categories the backend pool IS the filter — pass everything through.
  const scanTypeId = selectedCats.find(id =>
    SCAN_TYPES.some(t => t.id === id && t.id !== 'all')
  )

  if (!scanTypeId) return true  // 'all' or no type selected → pass everything

  if (scanTypeId === 'penny') {
    // Penny: hard price gate — only show stocks actually under $5
    const price = trade.entry || 0
    return price > 0 && price < 5
  }

  // bluechip, growth, etfs — backend already used the right pool, pass all results
  return true
}

const SCAN_DEPTHS = [
  { id: 100,   label: 'Top 100',        desc: 'Blue chips + popular' },
  { id: 500,   label: 'Top 500',        desc: 'Broad coverage' },
  { id: 1000,  label: 'Top 1,000',      desc: 'Extended market' },
  { id: 8000,  label: 'Full Universe',  desc: '~8K all exchanges' },
] as const

interface LogLine { ts: string; color: string; text: string }

// ── Module-level scan cache (survives navigation + page refresh) ──────────
// Uses sessionStorage so scan results survive F5/refresh (cleared on tab close).
const _SCAN_STORAGE_KEY = 'roboalgo_scan_cache'

function _loadScanCache(): {
  results: ScanResult | null; liveTrades: TradeRec[]; regime: RegimeInfo | null
  discovery: DiscoveryInfo | null; scanLog: LogLine[]; recentSymbols: ScanProgress[]
  progress: ScanProgress | null
} {
  try {
    const raw = sessionStorage.getItem(_SCAN_STORAGE_KEY)
    if (raw) {
      const parsed = JSON.parse(raw)
      return {
        results: parsed.results ?? null,
        liveTrades: parsed.liveTrades ?? [],
        regime: parsed.regime ?? null,
        discovery: parsed.discovery ?? null,
        scanLog: parsed.scanLog ?? [],
        recentSymbols: parsed.recentSymbols ?? [],
        progress: parsed.progress ?? null,
      }
    }
  } catch {}
  return { results: null, liveTrades: [], regime: null, discovery: null, scanLog: [], recentSymbols: [], progress: null }
}

function _saveScanCache(cache: typeof _scanCache) {
  try {
    sessionStorage.setItem(_SCAN_STORAGE_KEY, JSON.stringify(cache))
  } catch {} // ignore quota errors
}

const _scanCache = _loadScanCache()

function useStreamingScan() {
  const [progress, setProgress] = useState<ScanProgress | null>(_scanCache.progress)
  const [results, setResults] = useState<ScanResult | null>(_scanCache.results)
  const [isScanning, setIsScanning] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [recentSymbols, setRecentSymbols] = useState<ScanProgress[]>(_scanCache.recentSymbols)
  const [liveTrades, setLiveTrades] = useState<TradeRec[]>(_scanCache.liveTrades)
  const [regime, setRegime] = useState<RegimeInfo | null>(_scanCache.regime)
  const [discovery, setDiscovery] = useState<DiscoveryInfo | null>(_scanCache.discovery)
  const [statusMsg, setStatusMsg] = useState<string>('')
  const [scanLog, setScanLog] = useState<LogLine[]>(_scanCache.scanLog)
  const abortRef = useRef<AbortController | null>(null)

  const addLog = useCallback((text: string, color = '#6b7280') => {
    const ts = new Date().toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' })
    setScanLog(prev => [...prev.slice(-199), { ts, color, text }])
  }, [])

  const startScan = useCallback((profile: string, account: number, symbols?: string, categories?: string[], maxSymbols?: number) => {
    // Abort previous scan if running
    if (abortRef.current) abortRef.current.abort()
    const controller = new AbortController()
    abortRef.current = controller

    setIsScanning(true)
    setError(null)
    setProgress(null)
    setResults(null)
    setRecentSymbols([])
    setLiveTrades([])
    setRegime(null)
    setDiscovery(null)
    setStatusMsg('Detecting market regime...')
    setScanLog([{ ts: new Date().toLocaleTimeString('en-US', { hour12: false }), color: '#10b981', text: '▶ Scan started' }])

    const isFullUniverse = maxSymbols != null && maxSymbols >= 8000
    const params = new URLSearchParams({ profile, account: String(account) })
    if (symbols) params.set('symbols', symbols)
    // When max_symbols >= 8000 (Full Universe), skip categories so the backend
    // hits the "All Scan" path that fetches ALL exchange tickers (~8K).
    // Otherwise category filters limit us to the curated ~500 symbol list.
    if (categories && categories.length > 0 && !isFullUniverse) {
      params.set('categories', categories.join(','))
    }
    if (maxSymbols) params.set('max_symbols', String(maxSymbols))
    // Only run the 6,400-stock pre-screening when user explicitly selects Full Universe
    if (!isFullUniverse) params.set('discover', 'false')

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
              if (event.type === 'regime') {
                setRegime(event)
                setStatusMsg(`Market: ${event.regime} (${event.bias})`)
                addLog(`REGIME  ${event.regime} · Bias: ${event.bias}${event.spy_price != null ? ` · SPY $${event.spy_price}` : ''}`, event.bias === 'LONG' ? '#10b981' : event.bias === 'SHORT' ? '#ef4444' : '#eab308')
              } else if (event.type === 'status') {
                setStatusMsg(event.message)
                addLog(`STATUS  ${event.message}`, '#60a5fa')
              } else if (event.type === 'discovery') {
                setDiscovery(event)
                setStatusMsg(`Found ${event.movers_found} unusual movers + ${event.hedges_added} hedges → scanning ${event.total_symbols} total`)
                addLog(`DISCOVER  ${event.movers_found} movers found · ${event.hedges_added} hedges added · ${event.total_symbols} symbols queued`, '#a78bfa')
              } else if (event.type === 'progress') {
                setProgress(event)
                setRecentSymbols(prev => [...prev.slice(-49), event])
                setStatusMsg('')
                const verdict = event.verdict
                const col = verdict === 'BUY' ? '#10b981' : verdict === 'HOLD' ? '#f59e0b' : verdict === 'ERROR' ? '#ef4444' : '#4b5563'
                const tag = verdict === 'BUY' ? '✦ BUY ' : verdict === 'HOLD' ? '◈ HOLD' : verdict === 'ERROR' ? '✗ ERR ' : '· skip'
                const phaseTag = event.phase === 2 ? '★' : ' '
                addLog(`[${String(event.completed).padStart(3)}/${event.total}]${phaseTag} ${tag}  ${event.symbol.padEnd(6)}  score ${event.score > 0 ? event.score.toFixed(0) : '—'}`, col)
              } else if (event.type === 'trade') {
                setLiveTrades(prev => {
                  const updated = [...prev, event.trade]
                  return updated.sort((a: TradeRec, b: TradeRec) => b.score - a.score)
                })
                addLog(`▲ TRADE  ${event.trade.symbol}  ${event.trade.verdict}  +${event.trade.projected_profit_pct?.toFixed(1)}%  entry $${event.trade.entry?.toFixed(2)}`, '#10b981')
              } else if (event.type === 'result') {
                setResults(event)
                addLog(`■ DONE  ${event.symbols_scanned} scanned · ${event.top_trades?.length ?? 0} trades · ${event.watchlist?.length ?? 0} watchlist · ${event.scan_time}s`, '#10b981')
              } else if (event.type === 'error') {
                setError(`Scan error: ${event.message}`)
                addLog(`✗ ERROR  ${event.message}`, '#ef4444')
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

  // Sync state back to module-level cache + sessionStorage (survives navigation + refresh)
  useEffect(() => { _scanCache.results = results; _saveScanCache(_scanCache) }, [results])
  useEffect(() => { _scanCache.liveTrades = liveTrades; _saveScanCache(_scanCache) }, [liveTrades])
  useEffect(() => { _scanCache.regime = regime; _saveScanCache(_scanCache) }, [regime])
  useEffect(() => { _scanCache.discovery = discovery; _saveScanCache(_scanCache) }, [discovery])
  useEffect(() => { _scanCache.scanLog = scanLog; _saveScanCache(_scanCache) }, [scanLog])
  useEffect(() => { _scanCache.recentSymbols = recentSymbols; _saveScanCache(_scanCache) }, [recentSymbols])
  useEffect(() => { _scanCache.progress = progress; _saveScanCache(_scanCache) }, [progress])

  // ── Live quote polling ────────────────────────────────────────────────
  // After a scan completes, poll /api/quote/{symbol} for each trade's symbol
  // every 10s to keep `change_pct` and `current_price` fresh without re-running
  // the 60+ second scan. Also flashes rows whose price ticked since last poll.
  const [quoteFlash, setQuoteFlash] = useState<Record<string, 'up' | 'down'>>({})
  const prevPricesRef = useRef<Record<string, number>>({})
  useEffect(() => {
    // Don't poll while scan is running — SSE is already streaming fresh data
    if (isScanning) return
    // Gather all symbols we want live quotes for (liveTrades + results lists)
    const gather = (): string[] => {
      const set = new Set<string>()
      liveTrades.forEach(t => t.symbol && set.add(t.symbol))
      if (results) {
        ;(results.top_trades || []).forEach(t => t.symbol && set.add(t.symbol))
        ;(results.watchlist || []).forEach(t => t.symbol && set.add(t.symbol))
      }
      return Array.from(set)
    }
    const syms = gather()
    if (syms.length === 0) return

    let cancelled = false
    const fetchQuotes = async () => {
      try {
        const results = await Promise.allSettled(
          syms.map(s => api.get(`/quote/${encodeURIComponent(s)}`).then(r => r.data).catch(() => null))
        )
        if (cancelled) return
        // Build symbol → quote lookup
        const quotes: Record<string, { price: number; change_pct: number; change: number }> = {}
        results.forEach((res, idx) => {
          if (res.status === 'fulfilled' && res.value && typeof res.value.price === 'number') {
            quotes[syms[idx]] = {
              price: res.value.price,
              change_pct: res.value.change_pct ?? 0,
              change: res.value.change ?? 0,
            }
          }
        })
        if (Object.keys(quotes).length === 0) return

        // Detect price ticks for flash animation
        const newFlash: Record<string, 'up' | 'down'> = {}
        for (const [sym, q] of Object.entries(quotes)) {
          const prev = prevPricesRef.current[sym]
          if (prev !== undefined && prev !== q.price) {
            newFlash[sym] = q.price > prev ? 'up' : 'down'
          }
          prevPricesRef.current[sym] = q.price
        }

        // Merge new quote fields into trade recs (immutable update)
        const merge = (trade: TradeRec): TradeRec => {
          const q = quotes[trade.symbol]
          if (!q) return trade
          return { ...trade, change_pct: q.change_pct, current_price: q.price } as any
        }
        setLiveTrades(prev => prev.map(merge))
        setResults(prev => {
          if (!prev) return prev
          return {
            ...prev,
            top_trades: (prev.top_trades || []).map(merge),
            watchlist: (prev.watchlist || []).map(merge),
            recommendations: (prev.recommendations || []).map(merge),
          }
        })

        if (Object.keys(newFlash).length) {
          setQuoteFlash(newFlash)
          setTimeout(() => !cancelled && setQuoteFlash({}), 800)
        }
      } catch { /* silent */ }
    }

    // Fire once immediately, then every 10s
    fetchQuotes()
    const id = window.setInterval(fetchQuotes, 10_000)
    return () => { cancelled = true; window.clearInterval(id) }
  }, [isScanning, liveTrades.length, results])

  return { progress, results, isScanning, error, recentSymbols, liveTrades, regime, discovery, statusMsg, scanLog, startScan, abort, quoteFlash }
}

interface MultiLookupResult {
  symbol: string
  current_price: number
  overall_trend: string
  best_profile: string | null
  profiles: Record<string, LookupResult & { warnings?: string[] }>
}

function useMultiLookup(symbol: string, account: number) {
  return useQuery<MultiLookupResult>({
    queryKey: ['trade-signals', 'multi-lookup', symbol, account],
    queryFn: () => api.get(`/trade-signals/multi-lookup/${symbol}?account=${account}`).then(r => r.data),
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

// ── Chart data hook ──────────────────────────────────────────────────────────

// Hover mini-chart and useChartData removed — chart grid uses PatternChartGrid component

// ── Trade Card (scanner results) ─────────────────────────────────────────────

function TradeCard({ rec, onOpenChart, account, onOpenDetail }: { rec: TradeRec; onOpenChart?: (symbol: string) => void; account: number; onOpenDetail?: (rec: TradeRec) => void }) {
  const [expanded, setExpanded] = useState(false)
  const navigate = useNavigate()
  const isActionable = rec.verdict === 'BUY' || rec.verdict === 'SELL'
  const isShort = rec.direction === 'SHORT' || rec.verdict === 'SELL'

  return (
    <div className="relative">
      <div
        className={`border rounded-lg p-4 cursor-pointer transition-all ${
          isActionable && isShort
            ? 'border-red-500/30 bg-gray-900/80 hover:border-red-400/50'
            : isActionable
            ? 'border-emerald-500/30 bg-gray-900/80 hover:border-emerald-400/50'
            : 'border-gray-800/60 bg-gray-950/50 hover:border-gray-700'
        }`}
        onClick={() => setExpanded(!expanded)}
      >
      {/* Header row */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <button
            onClick={(e) => { e.stopPropagation(); onOpenChart ? onOpenChart(rec.symbol) : navigate({ pathname: '/chart', search: `symbol=${rec.symbol}` }) }}
            className={`text-white font-bold text-lg font-mono transition-colors ${isShort ? 'hover:text-red-400' : 'hover:text-emerald-400'}`}
            title="Open chart"
          >{rec.symbol} ↗</button>
          <VerdictBadge verdict={rec.verdict} score={rec.score} />
          {isShort && <span className="text-red-400 text-[10px] font-bold border border-red-500/30 px-1.5 py-0.5 rounded">SHORT</span>}
          {/* Pattern badges */}
          {(rec as any).patterns?.length > 0 && (
            <div className="flex items-center gap-1 flex-wrap">
              {(rec as any).patterns.slice(0, 2).map((p: string, i: number) => (
                <span key={i} className="text-purple-400 text-[9px] font-mono bg-purple-500/10 border border-purple-500/20 px-1.5 py-0.5 rounded">
                  {p}
                </span>
              ))}
              {(rec as any).pattern_score > 0 && (
                <span className="text-purple-300 text-[9px] font-mono font-bold">
                  +{(rec as any).pattern_score.toFixed(0)}
                </span>
              )}
            </div>
          )}
        </div>
        <div className="flex items-center gap-3">
          {/* Regime TF badge */}
          {(() => {
            const gate = rec.component_details?.stoch_gate
            const regimeTf = gate?.regime_tf
            const entryTfs = gate?.entry_tfs || []
            if (!regimeTf) return null
            return (
              <div className="flex items-center gap-1">
                <span className="text-blue-400 text-[10px] font-mono font-bold bg-blue-500/10 border border-blue-500/30 px-1.5 py-0.5 rounded">
                  {regimeTf.toUpperCase()}
                </span>
                {entryTfs.length > 0 && (
                  <span className="text-gray-500 text-[9px] font-mono">
                    +{entryTfs.join(',')}
                  </span>
                )}
              </div>
            )
          })()}
          {isActionable && (() => {
            const s = recalcSize(rec, account)
            return (
              <span className={`${isShort ? 'text-red-400' : 'text-emerald-400'} font-mono text-sm`}>
                +${formatWithCommas(s.profit)} ({rec.projected_profit_pct.toFixed(1)}%)
              </span>
            )
          })()}
          <ScoreBar score={rec.score} />
          {onOpenDetail && (
            <button
              onClick={(e) => { e.stopPropagation(); onOpenDetail(rec) }}
              className="text-[10px] font-mono text-gray-500 hover:text-blue-400 border border-gray-700 hover:border-blue-500/40 px-1.5 py-0.5 rounded transition-colors"
              title="Show full why / why-not reasoning"
            >Why?</button>
          )}
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
          {(() => { const s = recalcSize(rec, account); return (
          <div>
            <span className="text-gray-500 block">Size</span>
            <span className="text-white">{formatWithCommas(s.size)} shr</span>
          </div>
          )})()}
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
          {(() => { const s = recalcSize(rec, account); return (
          <div className="text-xs text-gray-600 font-mono">
            Risk: ${formatWithCommas(s.risk)} | Value: ${formatWithCommas(s.value)} | Source: {rec.level_source}
          </div>
          )})()}
        </div>
      )}
      </div>
    </div>
  )
}

// ── Trade Detail Drawer (why / why-not reasoning) ───────────────────────────

const COMPONENT_META: Record<string, { label: string; group: string }> = {
  mtf_alignment:      { label: 'MTF Alignment',       group: 'Structure' },
  trend_alignment:    { label: 'Trend Alignment',     group: 'Structure' },
  wave_phase:         { label: 'Wave Phase (EW)',     group: 'Structure' },
  structure_shift:    { label: 'CHoCH / BOS',         group: 'Structure' },
  cycle_alignment:    { label: 'Cycle Alignment',     group: 'Structure' },
  fvg_confluence:     { label: 'FVG Confluence',      group: 'SMC' },
  order_block:        { label: 'Order Block',         group: 'SMC' },
  premium_discount:   { label: 'Premium / Discount',  group: 'SMC' },
  liquidity_sweep:    { label: 'Liquidity Sweep',     group: 'SMC' },
  level_confluence:   { label: 'Level Confluence',    group: 'Levels' },
  breakout_strength:  { label: 'Breakout Strength',   group: 'Momentum' },
  relative_strength:  { label: 'Relative Strength',   group: 'Momentum' },
  vol_compression:    { label: 'Vol Compression',     group: 'Momentum' },
  exhaustion_check:   { label: 'Exhaustion',          group: 'Momentum' },
  gex_signal:         { label: 'GEX Signal',          group: 'Options' },
  dealer_positioning: { label: 'Dealer Positioning',  group: 'Options' },
  catalyst:           { label: 'Catalyst',            group: 'Context' },
  pattern_score:      { label: 'Pattern Score',       group: 'Context' },
}

function ComponentBar({ name, value }: { name: string; value: number }) {
  const meta = COMPONENT_META[name] ?? { label: name, group: '—' }
  // For premium_discount, lower = discount = bullish. Normalize for color.
  const adj = name === 'premium_discount' ? 1 - value : value
  const pct = Math.max(0, Math.min(100, adj * 100))
  const color = adj >= 0.7 ? 'bg-emerald-500' : adj >= 0.5 ? 'bg-amber-500' : adj >= 0.3 ? 'bg-gray-600' : 'bg-red-500'
  return (
    <div className="flex items-center gap-2 text-[11px] font-mono">
      <span className="w-32 text-gray-400 truncate" title={meta.label}>{meta.label}</span>
      <div className="flex-1 h-1.5 bg-gray-900 rounded-full overflow-hidden">
        <div className={`h-full ${color} transition-all`} style={{ width: `${pct}%` }} />
      </div>
      <span className="w-10 text-right text-gray-300">{value.toFixed(2)}</span>
    </div>
  )
}

function TradeDetailDrawer({ trade, onClose, onOpenChart, account }: {
  trade: TradeRec
  onClose: () => void
  onOpenChart?: (symbol: string) => void
  account: number
}) {
  const isShort = trade.direction === 'SHORT' || trade.verdict === 'SELL'
  const isActionable = trade.verdict === 'BUY' || trade.verdict === 'SELL'
  const accent = isShort ? 'red' : isActionable ? 'emerald' : 'amber'
  const comps = trade.components || {}
  const details = trade.component_details || {}
  const summary = (trade as any).component_summary || {}
  const topBullish: string[] = summary.top_bullish || []
  const topBearish: string[] = summary.top_bearish || []
  const gatesPassed = summary.hard_gates_passed !== false
  const bonuses: string[] = trade.bonuses || []
  const notes: string[] = trade.entry_notes || []
  const rejection = trade.rejection_reason || ''
  const gate = details.stoch_gate || {}
  const mtfDetail = details.mtf || {}

  // Group components
  const grouped: Record<string, Array<[string, number]>> = {}
  Object.entries(comps).forEach(([k, v]) => {
    const group = COMPONENT_META[k]?.group || 'Other'
    if (!grouped[group]) grouped[group] = []
    grouped[group].push([k, v as number])
  })
  const groupOrder = ['Structure', 'SMC', 'Levels', 'Momentum', 'Options', 'Context', 'Other']

  // Flags
  const flagList: Array<[string, string]> = []
  if ((trade as any).coiling_3tf) flagList.push(['◆ GENESIS', 'amber'])
  else if ((trade as any).coiling) flagList.push(['◈ COMPRESSION', 'orange'])
  if ((trade as any).bear_fade) flagList.push(['▽ DISTRIBUTION', 'red'])
  if ((trade as any).hourly_starting_up) flagList.push(['▲ Hourly starting', 'blue'])

  const s = recalcSize(trade, account)

  return (
    <div className="fixed inset-0 z-50 flex justify-end" role="dialog" aria-modal="true">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/70 backdrop-blur-sm" onClick={onClose} />
      {/* Drawer */}
      <div className={`relative w-full max-w-xl bg-gray-950 border-l border-${accent}-500/30 overflow-y-auto shadow-2xl`}>
        {/* Header */}
        <div className={`sticky top-0 z-10 bg-gray-950 border-b border-${accent}-500/20 px-5 py-3 flex items-center justify-between`}>
          <div className="flex items-center gap-3">
            <button
              className="text-white font-bold text-xl font-mono hover:text-blue-400"
              onClick={() => onOpenChart?.(trade.symbol)}
              title="Open chart"
            >{trade.symbol} ↗</button>
            <VerdictBadge verdict={trade.verdict} score={trade.score} />
            {isShort && <span className="text-red-400 text-[10px] font-bold border border-red-500/30 px-1.5 py-0.5 rounded">SHORT</span>}
            {flagList.map(([txt, c], i) => (
              <span key={i} className={`text-${c}-300 text-[10px] font-bold bg-${c}-500/10 border border-${c}-500/30 px-2 py-0.5 rounded`}>{txt}</span>
            ))}
          </div>
          <button onClick={onClose} className="text-gray-400 hover:text-white text-2xl leading-none px-2" title="Close">×</button>
        </div>

        <div className="p-5 space-y-5">
          {/* Plan row */}
          {isActionable && (
            <div className="grid grid-cols-5 gap-2 text-xs font-mono bg-gray-900/60 border border-gray-800/60 rounded p-3">
              <div><span className="text-gray-500 block">Entry</span><span className="text-white">${trade.entry.toFixed(2)}</span></div>
              <div><span className="text-gray-500 block">Stop</span><span className="text-red-400">${trade.stop.toFixed(2)}</span></div>
              <div><span className="text-gray-500 block">Target</span><span className="text-emerald-400">${trade.target.toFixed(2)}</span></div>
              <div><span className="text-gray-500 block">R:R</span><span className="text-white">{trade.risk_reward.toFixed(1)}:1</span></div>
              <div><span className="text-gray-500 block">Size</span><span className="text-white">{formatWithCommas(s.size)} shr</span></div>
            </div>
          )}

          {/* Why (bullish) */}
          <section>
            <h4 className="text-[11px] uppercase tracking-wider text-emerald-400 font-bold mb-2">Why — strongest signals</h4>
            {topBullish.length > 0 ? (
              <ul className="space-y-1 text-[12px] font-mono">
                {topBullish.map((b, i) => (
                  <li key={i} className="flex items-start gap-2 text-emerald-300">
                    <span className="text-emerald-500 mt-0.5">✓</span><span>{b}</span>
                  </li>
                ))}
              </ul>
            ) : <p className="text-gray-600 text-xs italic">No dominant bullish components.</p>}
            {bonuses.filter(b => !b.startsWith('penalty')).length > 0 && (
              <div className="mt-2 space-y-0.5">
                {bonuses.filter(b => !b.startsWith('penalty')).map((b, i) => (
                  <p key={i} className="text-amber-300 text-[11px] font-mono">★ {b}</p>
                ))}
              </div>
            )}
            {notes.length > 0 && (
              <div className="mt-2 space-y-0.5">
                {notes.map((n, i) => (
                  <p key={i} className="text-gray-400 text-[11px] font-mono">• {n}</p>
                ))}
              </div>
            )}
          </section>

          {/* Why not (bearish / penalties / gates) */}
          <section>
            <h4 className="text-[11px] uppercase tracking-wider text-red-400 font-bold mb-2">Why not — headwinds</h4>
            {!gatesPassed && (
              <p className="text-red-400 text-[11px] font-mono mb-2">✗ Hard gate failed — trade not actionable</p>
            )}
            {topBearish.length > 0 ? (
              <ul className="space-y-1 text-[12px] font-mono">
                {topBearish.map((b, i) => (
                  <li key={i} className="flex items-start gap-2 text-red-300">
                    <span className="text-red-500 mt-0.5">✗</span><span>{b}</span>
                  </li>
                ))}
              </ul>
            ) : <p className="text-gray-600 text-xs italic">No significant bearish components.</p>}
            {bonuses.filter(b => b.startsWith('penalty')).length > 0 && (
              <div className="mt-2 space-y-0.5">
                {bonuses.filter(b => b.startsWith('penalty')).map((b, i) => (
                  <p key={i} className="text-red-300 text-[11px] font-mono">− {b}</p>
                ))}
              </div>
            )}
            {rejection && (
              <p className="mt-2 text-red-400 text-[11px] font-mono bg-red-500/5 border-l-2 border-red-500/50 pl-2 py-1">
                Rejection: {rejection}
              </p>
            )}
          </section>

          {/* MTF alignment */}
          {mtfDetail.alignment_score != null && (
            <section>
              <h4 className="text-[11px] uppercase tracking-wider text-blue-400 font-bold mb-2">Multi-Timeframe</h4>
              <div className="text-[11px] font-mono space-y-1">
                <div className="flex justify-between"><span className="text-gray-500">Bias</span><span className={mtfDetail.bias_direction === 'LONG' ? 'text-emerald-400' : mtfDetail.bias_direction === 'SHORT' ? 'text-red-400' : 'text-gray-400'}>{mtfDetail.bias_direction || '—'}</span></div>
                <div className="flex justify-between"><span className="text-gray-500">Score</span><span className="text-white">{(mtfDetail.alignment_score * 100).toFixed(0)}%</span></div>
                {mtfDetail.aligned_timeframes?.length > 0 && (
                  <div className="flex justify-between"><span className="text-gray-500">Aligned</span><span className="text-emerald-400">{mtfDetail.aligned_timeframes.join(', ')}</span></div>
                )}
                {mtfDetail.conflicting_timeframes?.length > 0 && (
                  <div className="flex justify-between"><span className="text-gray-500">Conflict</span><span className="text-red-400">{mtfDetail.conflicting_timeframes.join(', ')}</span></div>
                )}
                {gate.regime_tf && (
                  <div className="flex justify-between"><span className="text-gray-500">Regime TF</span><span className="text-blue-400">{String(gate.regime_tf).toUpperCase()}{gate.entry_tfs?.length ? ` + ${gate.entry_tfs.join(',')}` : ''}</span></div>
                )}
              </div>
            </section>
          )}

          {/* Component bars */}
          <section>
            <h4 className="text-[11px] uppercase tracking-wider text-gray-400 font-bold mb-2">All 18 components</h4>
            <div className="space-y-3">
              {groupOrder.filter(g => grouped[g]?.length).map(g => (
                <div key={g}>
                  <div className="text-[9px] uppercase tracking-wider text-gray-600 mb-1">{g}</div>
                  <div className="space-y-1">
                    {grouped[g].map(([k, v]) => <ComponentBar key={k} name={k} value={v} />)}
                  </div>
                </div>
              ))}
            </div>
          </section>

          {/* Footer */}
          <div className="text-[10px] font-mono text-gray-600 pt-3 border-t border-gray-800/60">
            Risk: ${formatWithCommas(s.risk)} · Value: ${formatWithCommas(s.value)} · Level: {trade.level_source}
          </div>
        </div>
      </div>
    </div>
  )
}

// ── List View ───────────────────────────────────────────────────────────────

function TradeListView({ trades, onOpenChart, sortKey, sortAsc, onSort, account, quoteFlash, onOpenDetail }: {
  trades: TradeRec[]
  onOpenChart: (s: string) => void
  sortKey: SortKey
  sortAsc: boolean
  onSort: (key: SortKey) => void
  account: number
  quoteFlash?: Record<string, 'up' | 'down'>
  onOpenDetail?: (rec: TradeRec) => void
}) {
  if (trades.length === 0) return null

  const SortTh = ({ label, col, className = '' }: { label: string; col: SortKey; className?: string }) => (
    <th
      className={`px-2 py-1.5 cursor-pointer hover:text-white select-none transition-colors ${className}`}
      onClick={() => onSort(col)}
    >
      {label} {sortKey === col ? (sortAsc ? '↑' : '↓') : ''}
    </th>
  )

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs font-mono">
        <thead>
          <tr className="text-gray-500 border-b border-gray-800 text-left">
            <SortTh label="Symbol" col="symbol" />
            <th className="px-2 py-1.5 cursor-pointer hover:text-white select-none" onClick={() => onSort('change_pct')}>%Chg</th>
            <SortTh label="Score" col="score" />
            <SortTh label="Entry" col="entry" />
            <SortTh label="Stop" col="stop" />
            <SortTh label="Target" col="target" />
            <SortTh label="R:R" col="rr" />
            <SortTh label="Gain%" col="gain_pct" />
            <th className="px-2 py-1.5">Confluence</th>
            <th className="px-2 py-1.5">Regime TF</th>
            <th className="px-2 py-1.5 cursor-pointer hover:text-white select-none" onClick={() => onSort('score')}>Pattern</th>
            <th className="px-2 py-1.5">Stoch %K/%D</th>
            <th className="px-2 py-1.5">Entry TFs</th>
            <SortTh label="Size" col="size" />
            <SortTh label="Risk $" col="risk" />
          </tr>
        </thead>
        <tbody>
          {trades.map(rec => {
            const gate = rec.component_details?.stoch_gate || {}
            const regimeTf = gate.regime_tf || ''
            const tfResults = gate.tf_results || {}
            const regimeData = tfResults[regimeTf] || {}
            const entryTfs = (gate.entry_tfs || []) as string[]
            const isShort = rec.direction === 'SHORT'
            const dirColor = isShort ? 'text-red-400' : 'text-emerald-400'
            const verdictBg = rec.verdict === 'BUY'
              ? 'bg-emerald-500/10' : rec.verdict === 'SELL'
              ? 'bg-red-500/10' : rec.verdict === 'HOLD'
              ? 'bg-amber-500/10' : ''
            const fl = quoteFlash?.[rec.symbol]
            const flashBg = fl === 'up'
              ? 'bg-emerald-500/20'
              : fl === 'down'
              ? 'bg-red-500/20'
              : ''
            return (
              <tr
                key={rec.symbol}
                className={`border-b border-gray-800/40 hover:bg-gray-800/30 cursor-pointer transition-colors duration-500 ${flashBg || verdictBg}`}
                onClick={() => onOpenDetail ? onOpenDetail(rec) : onOpenChart(rec.symbol)}
                title="Click row for why/why-not detail · click symbol for chart"
              >
                <td className="px-2 py-1.5">
                  <button
                    onClick={(e) => { e.stopPropagation(); onOpenChart(rec.symbol) }}
                    className={`font-bold ${dirColor} hover:underline`}
                    title="Open chart"
                  >{rec.symbol}</button>
                </td>
                <td className="px-2 py-1.5">
                  {(() => {
                    const chg = (rec as any).change_pct ?? 0
                    return (
                      <span className={chg > 0 ? 'text-emerald-400' : chg < 0 ? 'text-red-400' : 'text-gray-500'}>
                        {chg > 0 ? '+' : ''}{chg.toFixed(2)}%
                      </span>
                    )
                  })()}
                </td>
                <td className="px-2 py-1.5">
                  <span className={rec.score >= 85 ? 'text-emerald-400 font-bold' : rec.score >= 70 ? 'text-emerald-500' : 'text-amber-400'}>
                    {rec.score.toFixed(0)}
                  </span>
                </td>
                <td className="px-2 py-1.5 text-white">${rec.entry > 0 ? rec.entry.toFixed(2) : '—'}</td>
                <td className="px-2 py-1.5 text-red-400">${rec.stop > 0 ? rec.stop.toFixed(2) : '—'}</td>
                <td className="px-2 py-1.5 text-emerald-400">${rec.target > 0 ? rec.target.toFixed(2) : '—'}</td>
                <td className="px-2 py-1.5 text-white">{rec.risk_reward > 0 ? `${rec.risk_reward.toFixed(1)}:1` : '—'}</td>
                <td className={`px-2 py-1.5 ${rec.projected_profit_pct > 0 ? dirColor : 'text-gray-500'}`}>
                  {rec.projected_profit_pct > 0 ? `${rec.projected_profit_pct.toFixed(1)}%` : '—'}
                </td>
                <td className="px-2 py-1.5">
                  {(() => {
                    const cl        = (rec as any).confluence_level || gate?.confluence_level || 0
                    const lb        = (rec as any).confluence_label || gate?.confluence_label || gate?.confluence_tier || ''
                    const isCoiling = (rec as any).coiling === true || gate?.coiling === true
                    const is3tf     = (rec as any).coiling_3tf === true || gate?.coiling_3tf === true
                    const isBearFade = (rec as any).bear_fade === true || gate?.bear_fade === true
                    const invEtf    = (rec as any).inverse_etf as string | undefined
                    const spread    = (rec as any).daily_kd_spread ?? gate?.daily_kd_spread
                    if (!lb && !isBearFade && cl === 0) return <span className="text-gray-600 text-[9px]">—</span>
                    const cls = isBearFade ? 'text-red-300 bg-red-500/15 border-red-500/40'
                              : is3tf     ? 'text-amber-300 bg-amber-400/20 border-amber-400/50'
                              : cl >= 5   ? 'text-yellow-300 bg-yellow-500/15 border-yellow-500/40'
                              : cl >= 4   ? 'text-emerald-400 bg-emerald-500/10 border-emerald-500/30'
                              : isCoiling ? 'text-orange-300 bg-orange-500/15 border-orange-500/40'
                              : cl >= 3   ? 'text-amber-400 bg-amber-500/10 border-amber-500/30'
                              : cl >= 2   ? 'text-gray-400 bg-gray-500/10 border-gray-500/20'
                              : 'text-red-400 bg-red-500/10 border-red-500/20'
                    const icon = isBearFade ? '▽ ' : is3tf ? '◆ ' : isCoiling ? '◈ ' : cl >= 5 ? '⬟ ' : cl >= 4 ? '★ ' : ''
                    const displayLabel = isBearFade ? 'DISTRIBUTION'
                                       : is3tf      ? 'GENESIS'
                                       : isCoiling  ? 'COMPRESSION'
                                       : lb
                    return (
                      <div className="flex flex-col gap-0.5">
                        <span className={`text-[9px] font-bold border px-1.5 py-0.5 rounded ${cls}`}>
                          {icon}{displayLabel}
                        </span>
                        {isBearFade && invEtf && (
                          <span className="text-[8px] font-mono text-red-400">→ {invEtf}</span>
                        )}
                        {!isBearFade && spread != null && (
                          <span className={`text-[8px] font-mono ${Math.abs(spread) <= 1.5 ? 'text-orange-400' : 'text-gray-600'}`}>
                            Δ{spread > 0 ? '+' : ''}{spread.toFixed(1)}
                          </span>
                        )}
                      </div>
                    )
                  })()}
                </td>
                <td className="px-2 py-1.5">
                  {regimeTf && (
                    <span className="text-blue-400 text-[10px] font-bold bg-blue-500/10 border border-blue-500/30 px-1.5 py-0.5 rounded">
                      {regimeTf.toUpperCase()}
                    </span>
                  )}
                </td>
                <td className="px-2 py-1.5">
                  {((rec as any).patterns || []).length > 0 ? (
                    <div className="flex flex-wrap gap-0.5">
                      {(rec as any).patterns.slice(0, 2).map((p: string, i: number) => (
                        <span key={i} className="text-purple-400 text-[9px] bg-purple-500/10 border border-purple-500/20 px-1 py-0.5 rounded whitespace-nowrap">
                          {p}
                        </span>
                      ))}
                    </div>
                  ) : <span className="text-gray-600">—</span>}
                </td>
                <td className="px-2 py-1.5">
                  {regimeData.k != null ? (
                    <div className="flex flex-col gap-0.5">
                      <span className={regimeData.k < 20 ? 'text-emerald-400' : regimeData.k > 80 ? 'text-red-400' : 'text-gray-400'}>
                        {Number(regimeData.k).toFixed(1)} / {Number(regimeData.d).toFixed(1)}
                      </span>
                      {(() => {
                        const dy = gate.tf_results?.['1d']
                        if (!dy || !dy.available || regimeTf === '1d') return null
                        const spread = Number(dy.k) - Number(dy.d)
                        const color = Math.abs(spread) <= 3 ? 'text-orange-400 font-bold' : 'text-gray-600'
                        return (
                          <span className={`text-[8px] font-mono ${color}`} title="Daily K-D spread — orange = within 3pts of cross">
                            1d Δ{spread > 0 ? '+' : ''}{spread.toFixed(1)}
                          </span>
                        )
                      })()}
                    </div>
                  ) : '—'}
                </td>
                <td className="px-2 py-1.5 text-gray-500 text-[10px]">
                  {entryTfs.length > 0 ? entryTfs.join(', ') : '—'}
                </td>
                {(() => { const s = recalcSize(rec, account); return (<>
                <td className="px-2 py-1.5 text-white">{s.size > 0 ? formatWithCommas(s.size) : '—'}</td>
                <td className="px-2 py-1.5 text-gray-400">{s.risk > 0 ? `$${formatWithCommas(s.risk)}` : '—'}</td>
                </>)})()}
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

// ChartGridItem / TradeChartGrid removed — replaced by PatternChartGrid component

// ── Lookup Panel ─────────────────────────────────────────────────────────────

const PROFILE_META: Record<string, { label: string; icon: string; color: string; border: string; glow: string; desc: string }> = {
  SCALP:    { label: 'SCALP',    icon: '⚡', color: 'text-cyan-400',    border: 'border-cyan-500/40',    glow: 'shadow-cyan-500/10',    desc: '5m–15m · Quick in-and-out' },
  SWING:    { label: 'SWING',    icon: '↗',  color: 'text-blue-400',    border: 'border-blue-500/40',    glow: 'shadow-blue-500/10',    desc: '1H–4H · Days to weeks' },
  POSITION: { label: 'POSITION', icon: '◆',  color: 'text-purple-400',  border: 'border-purple-500/40',  glow: 'shadow-purple-500/10',  desc: '1D–1W · Weeks to months' },
}

function ProfileCard({ data, account, isBest }: { data: any; account: number; isBest: boolean }) {
  const prof = data?.profile || 'SWING'
  const meta = PROFILE_META[prof] || PROFILE_META.SWING
  const tradeable = data?.verdict === 'BUY' || data?.verdict === 'SELL'
  const hasError = !!data?.error
  const warnings = data?.warnings || []

  // Recalc with current account
  const sizing = tradeable && data ? recalcSize(data, account) : null

  const dirColor = data?.direction === 'LONG' ? 'text-emerald-400' : data?.direction === 'SHORT' ? 'text-red-400' : 'text-gray-500'
  const dirLabel = data?.direction === 'LONG' ? 'LONG ↑' : data?.direction === 'SHORT' ? 'SHORT ↓' : '—'

  return (
    <div className={`relative rounded-xl border ${isBest ? meta.border + ' shadow-lg ' + meta.glow : 'border-gray-800'} bg-gray-900/80 overflow-hidden transition-all`}>
      {/* Profile header */}
      <div className={`flex items-center justify-between px-4 py-2.5 ${isBest ? 'bg-gray-800/60' : 'bg-gray-900/40'} border-b border-gray-800/60`}>
        <div className="flex items-center gap-2">
          <span className={`text-lg ${meta.color}`}>{meta.icon}</span>
          <span className={`font-bold text-sm ${meta.color}`}>{meta.label}</span>
          <span className="text-[10px] text-gray-600">{meta.desc}</span>
        </div>
        <div className="flex items-center gap-2">
          {isBest && <span className="px-2 py-0.5 bg-amber-500/20 text-amber-400 text-[10px] font-bold rounded-full border border-amber-500/30 animate-pulse">BEST</span>}
          {tradeable && <VerdictBadge verdict={data.verdict} score={data.score} />}
        </div>
      </div>

      <div className="p-4 space-y-3">
        {hasError ? (
          <div className="text-gray-600 text-sm text-center py-2">Data unavailable</div>
        ) : tradeable ? (
          <>
            {/* Direction + Entry/Stop/Target */}
            <div className="flex items-center gap-3 mb-2">
              <span className={`font-bold font-mono text-sm ${dirColor}`}>{dirLabel}</span>
              <span className="text-gray-600 text-xs">Score: {data.score.toFixed(0)}</span>
            </div>

            <div className="grid grid-cols-4 gap-2">
              <div className="bg-gray-950 rounded-lg p-2.5 text-center">
                <div className="text-gray-600 text-[10px] mb-0.5">Entry</div>
                <div className="text-white font-mono text-sm">${data.entry?.toFixed(2)}</div>
              </div>
              <div className="bg-gray-950 rounded-lg p-2.5 text-center">
                <div className="text-gray-600 text-[10px] mb-0.5">Stop</div>
                <div className="text-red-400 font-mono text-sm">${data.stop?.toFixed(2)}</div>
              </div>
              <div className="bg-gray-950 rounded-lg p-2.5 text-center">
                <div className="text-gray-600 text-[10px] mb-0.5">Target</div>
                <div className="text-emerald-400 font-mono text-sm">${data.target?.toFixed(2)}</div>
              </div>
              <div className="bg-gray-950 rounded-lg p-2.5 text-center">
                <div className="text-gray-600 text-[10px] mb-0.5">R:R</div>
                <div className="text-white font-mono text-sm">{data.risk_reward?.toFixed(1)}:1</div>
              </div>
            </div>

            {/* Position sizing */}
            {sizing && (
              <div className="flex items-center gap-4 text-[11px] font-mono text-gray-500">
                <span>Size: <span className="text-gray-300">{sizing.size}</span> shares</span>
                <span>Risk: <span className="text-red-400">${sizing.risk.toLocaleString()}</span></span>
                <span>Profit: <span className="text-emerald-400">+${sizing.profit.toLocaleString()}</span></span>
              </div>
            )}

            {/* Stoch gate info */}
            {data.component_details?.stoch_gate && (
              <div className="text-[10px] font-mono text-gray-500">
                {data.component_details.stoch_gate.confluence_tier && (
                  <span className={`mr-2 ${data.component_details.stoch_gate.confluence_level >= 4 ? 'text-emerald-400' : data.component_details.stoch_gate.confluence_level >= 3 ? 'text-amber-400' : 'text-gray-600'}`}>
                    Confluence: {data.component_details.stoch_gate.confluence_tier}
                  </span>
                )}
                {data.component_details.stoch_gate.entry_tfs && (
                  <span>Entry TFs: {data.component_details.stoch_gate.entry_tfs}</span>
                )}
              </div>
            )}

            {/* Entry notes */}
            {data.entry_notes?.length > 0 && (
              <div className="space-y-0.5">
                {data.entry_notes.slice(0, 3).map((note: string, i: number) => (
                  <div key={i} className="text-[11px] text-gray-400">• {note}</div>
                ))}
              </div>
            )}

            {/* Bonuses */}
            {data.bonuses?.length > 0 && (
              <div className="space-y-0.5">
                {data.bonuses.slice(0, 2).map((b: string, i: number) => (
                  <div key={i} className="text-[11px] text-amber-400">★ {b}</div>
                ))}
              </div>
            )}
          </>
        ) : (
          /* Not tradeable */
          <div className="text-center py-3">
            <div className="text-gray-600 text-sm mb-1">
              {data?.rejection_reason || 'No setup detected'}
            </div>
            {data?.direction && data.direction !== 'NO_TRADE' && (
              <div className="text-[11px] text-gray-600">Direction bias: <span className={dirColor}>{data.direction}</span></div>
            )}
          </div>
        )}

        {/* Cross-horizon warnings */}
        {warnings.length > 0 && (
          <div className="border-t border-gray-800/50 pt-2 space-y-0.5">
            {warnings.map((w: string, i: number) => (
              <div key={i} className={`text-[11px] font-mono ${w.startsWith('✦') ? 'text-emerald-400' : 'text-amber-400'}`}>
                {w}
              </div>
            ))}
          </div>
        )}

        {/* Component scores (collapsible) */}
        {data?.components && Object.keys(data.components).length > 0 && (
          <details className="text-[10px]">
            <summary className="text-gray-700 cursor-pointer hover:text-gray-500">Components</summary>
            <div className="mt-1.5 grid grid-cols-3 gap-1">
              {Object.entries(data.components).map(([key, val]) => (
                <div key={key} className="flex items-center justify-between bg-gray-950 rounded px-1.5 py-0.5">
                  <span className="text-gray-600 truncate">{key.replace(/_/g, ' ')}</span>
                  <span className={`font-mono ml-1 ${(val as number) > 0.65 ? 'text-emerald-400' : (val as number) < 0.35 ? 'text-red-400' : 'text-gray-500'}`}>
                    {(val as number).toFixed(2)}
                  </span>
                </div>
              ))}
            </div>
          </details>
        )}
      </div>
    </div>
  )
}

function LookupPanel({ profile, account }: { profile: string; account: number }) {
  const [input, setInput] = useState('')
  const [symbol, setSymbol] = useState('')

  const { data, isLoading, error } = useMultiLookup(symbol, account)

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    setSymbol(input.toUpperCase().trim())
  }

  const trendColor = data?.overall_trend === 'LONG' ? 'text-emerald-400' : data?.overall_trend === 'SHORT' ? 'text-red-400' : 'text-gray-500'
  const trendArrow = data?.overall_trend === 'LONG' ? '↑' : data?.overall_trend === 'SHORT' ? '↓' : '→'

  return (
    <div className="space-y-4">
      <form onSubmit={handleSubmit} className="flex gap-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="AAPL"
          className="w-28 px-3 py-2 bg-gray-900 border border-gray-700 rounded-lg text-white font-mono text-sm
                     placeholder-gray-600 focus:border-emerald-500 focus:outline-none"
        />
        <button
          type="submit"
          disabled={!input.trim() || isLoading}
          className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 disabled:bg-gray-700 text-white
                     rounded-lg font-bold text-xs transition-colors"
        >
          {isLoading ? '...' : 'GO'}
        </button>
      </form>

      {isLoading && (
        <div className="flex items-center gap-2 text-gray-500 text-sm py-6">
          <div className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
          Analyzing {symbol} across all horizons...
        </div>
      )}

      {error && (
        <div className="text-red-400 text-sm p-3 bg-red-500/10 border border-red-500/20 rounded-lg">
          Failed to analyze {symbol}: {(error as any)?.response?.data?.detail || 'Unknown error'}
        </div>
      )}

      {data && (
        <div className="space-y-4">
          {/* Symbol header + overall trend */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <h2 className="text-2xl font-bold text-white font-mono">{data.symbol}</h2>
              <span className="text-gray-400 font-mono text-lg">${data.current_price?.toFixed(2)}</span>
            </div>
            <div className="flex items-center gap-3">
              <span className="text-gray-600 text-xs uppercase">Overall Trend</span>
              <span className={`font-bold font-mono text-lg ${trendColor}`}>
                {trendArrow} {data.overall_trend || 'NEUTRAL'}
              </span>
              {data.best_profile && (
                <span className={`text-xs px-2 py-0.5 rounded-full border ${PROFILE_META[data.best_profile]?.border || ''} ${PROFILE_META[data.best_profile]?.color || ''}`}>
                  Best: {data.best_profile}
                </span>
              )}
            </div>
          </div>

          {/* 3 profile cards */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">
            {['SCALP', 'SWING', 'POSITION'].map(prof => (
              <ProfileCard
                key={prof}
                data={data.profiles?.[prof]}
                account={account}
                isBest={data.best_profile === prof}
              />
            ))}
          </div>

          {/* Cross-horizon summary */}
          {(() => {
            const tradeableProfiles = ['SCALP', 'SWING', 'POSITION'].filter(p => {
              const v = data.profiles?.[p]?.verdict
              return v === 'BUY' || v === 'SELL'
            })
            const noTrade = tradeableProfiles.length === 0
            const allFlat = ['SCALP', 'SWING', 'POSITION'].every(p => {
              const d = data.profiles?.[p]?.direction
              return !d || d === 'NO_TRADE' || d === 'NEUTRAL'
            })

            if (allFlat) return (
              <div className="text-center py-3 text-gray-600 text-sm border border-gray-800/50 rounded-lg bg-gray-950/50">
                No directional bias detected — flat / choppy. Sit this one out.
              </div>
            )
            if (noTrade) return (
              <div className="text-center py-3 text-gray-600 text-sm border border-gray-800/50 rounded-lg bg-gray-950/50">
                Directional bias present but no clean entry across any horizon. Wait for setup.
              </div>
            )
            return null
          })()}
        </div>
      )}
    </div>
  )
}

// ── Main Page ────────────────────────────────────────────────────────────────

type Mode = 'scanner' | 'lookup'
type Profile = 'SCALP' | 'SWING' | 'POSITION'
type SortKey = 'score' | 'gain_pct' | 'change_pct' | 'price' | 'symbol' | 'entry' | 'stop' | 'target' | 'rr' | 'size' | 'risk'
type DirFilter = 'long' | 'short' | 'watchlist'
type ViewMode = 'cards' | 'list' | 'charts'

// ── Scan Progress Bar ───────────────────────────────────────────────────────

// ── Scan Terminal ────────────────────────────────────────────────────────────

function ScanTerminal({ lines, isScanning }: { lines: LogLine[]; isScanning: boolean }) {
  const bodyRef = useRef<HTMLDivElement>(null)
  const [open, setOpen] = useState(true)

  // Auto-open on scan start, minimize 2s after completion
  useEffect(() => {
    if (isScanning) setOpen(true)
  }, [isScanning])
  useEffect(() => {
    if (!isScanning && lines.length > 0) {
      const t = setTimeout(() => setOpen(false), 2000)
      return () => clearTimeout(t)
    }
  }, [isScanning, lines.length])

  // Auto-scroll log to bottom
  useEffect(() => {
    if (open && bodyRef.current) bodyRef.current.scrollTop = bodyRef.current.scrollHeight
  }, [lines.length, open])

  // Derive current status from last few lines
  const lastProgress = [...lines].reverse().find(l => l.text.match(/^\[/))
  const lastStatus   = [...lines].reverse().find(l => l.text.startsWith('STATUS') || l.text.startsWith('REGIME') || l.text.startsWith('DISCOVER'))
  const trades       = lines.filter(l => l.text.startsWith('▲ TRADE')).length
  const doneL        = [...lines].reverse().find(l => l.text.startsWith('■ DONE'))

  const terminal = (
    <div className="fixed bottom-4 right-4 z-[9999] font-mono text-[11px] w-[460px] shadow-2xl">
      {open ? (
        <div className="rounded-xl border border-gray-700/60 overflow-hidden bg-[#0d0f12]">
          {/* Title bar */}
          <div className="flex items-center justify-between px-3 py-2 bg-gray-900 border-b border-gray-800 select-none">
            <div className="flex items-center gap-2">
              <div className="flex gap-1.5">
                <div className="w-2.5 h-2.5 rounded-full bg-[#ff5f57]" />
                <div className="w-2.5 h-2.5 rounded-full bg-[#febc2e]" />
                <div className="w-2.5 h-2.5 rounded-full bg-[#28c840]" />
              </div>
              <span className="text-gray-400 text-[11px] ml-1">scan terminal</span>
              {isScanning && <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />}
            </div>
            <div className="flex items-center gap-3 text-[10px]">
              {trades > 0 && <span className="text-emerald-400">✦ {trades} trade{trades !== 1 ? 's' : ''}</span>}
              <span className="text-gray-600">{lines.length} lines</span>
              <button onClick={() => setOpen(false)} className="text-gray-600 hover:text-gray-300 transition-colors">─</button>
            </div>
          </div>

          {/* Status ticker — current activity */}
          <div className="px-3 py-2 bg-gray-950 border-b border-gray-800/50 min-h-[32px]">
            {lastProgress && (
              <span className="text-gray-300">{lastProgress.text}</span>
            )}
            {!lastProgress && lastStatus && (
              <span style={{ color: lastStatus.color }}>{lastStatus.text}</span>
            )}
            {!lastProgress && !lastStatus && (
              <span className="text-gray-700 animate-pulse">initializing...</span>
            )}
          </div>

          {/* Scrolling log */}
          <div ref={bodyRef} className="h-56 overflow-y-auto p-2 space-y-px">
            {lines.map((l, i) => (
              <div key={i} className="flex gap-2 leading-5 hover:bg-white/[0.02] px-1 rounded">
                <span className="text-gray-700 shrink-0 tabular-nums">{l.ts}</span>
                <span style={{ color: l.color }} className="break-all">{l.text}</span>
              </div>
            ))}
          </div>
        </div>
      ) : (
        /* Minimized pill */
        <button
          onClick={() => setOpen(true)}
          className="flex items-center gap-2 px-3 py-1.5 rounded-full border border-gray-700/60
                     bg-gray-950/95 text-gray-500 hover:text-gray-300 hover:border-gray-600
                     transition-colors shadow-xl backdrop-blur-sm"
        >
          <span className={`w-1.5 h-1.5 rounded-full ${isScanning ? 'bg-emerald-400 animate-pulse' : 'bg-gray-600'}`} />
          <span>scan log</span>
          {doneL
            ? <span className="text-emerald-500 text-[10px]">{doneL.text.replace('■ DONE  ', '')}</span>
            : trades > 0 && <span className="text-emerald-500 text-[10px]">{trades} trades found</span>
          }
          <span className="text-gray-700 text-[10px]">▲</span>
        </button>
      )}
    </div>
  )

  return createPortal(terminal, document.body)
}

// ── Scan Progress Bar ───────────────────────────────────────────────────────

/** Rich real-time scan loading panel — replaces the old thin progress bar. */
function ScanLoadingPanel({
  progress, recentSymbols, statusMsg, discovery, liveTrades,
}: {
  progress: ScanProgress | null
  recentSymbols: ScanProgress[]
  statusMsg: string
  discovery: DiscoveryInfo | null
  liveTrades: TradeRec[]
}) {
  // ── Derive phase ─────────────────────────────────────────────────────
  // 0 = pre-screen/regime   1 = daily fetch + fast score   2 = deep score
  const phase = progress
    ? ((progress as any).phase === 2 ? 2 : 1)
    : (discovery ? 1 : 0)

  const pct = progress
    ? Math.round((progress.completed / Math.max(progress.total, 1)) * 100)
    : (phase === 0 ? 0 : null)   // null = indeterminate

  // ── ETA (rolling 3-second window) ───────────────────────────────────
  const startRef = useRef<number>(Date.now())
  const etaSec = useMemo(() => {
    if (!progress || progress.completed < 5) return null
    const elapsed = (Date.now() - startRef.current) / 1000
    const rate = progress.completed / elapsed          // symbols/s
    const remaining = Math.max(0, progress.total - progress.completed)
    return rate > 0 ? Math.ceil(remaining / rate) : null
  }, [progress])

  // ── Live stats ───────────────────────────────────────────────────────
  const buys   = liveTrades.filter(t => t.verdict === 'BUY').length
  const sells  = liveTrades.filter(t => t.verdict === 'SELL').length
  const holds  = liveTrades.filter(t => t.verdict === 'HOLD').length
  const mtfHits = liveTrades.filter(t =>
    ((t as any).confluence_level ?? t.component_details?.stoch_gate?.confluence_level ?? 0) >= 3
  ).length
  const coilingHits = liveTrades.filter(t =>
    (t as any).coiling === true || t.component_details?.stoch_gate?.coiling === true
  ).length

  // ── Phase pipeline steps ─────────────────────────────────────────────
  const steps = [
    { label: 'Pre-screen', done: phase >= 1 },
    { label: 'Daily bars', done: !!progress },
    { label: 'Fast score', done: phase >= 2, active: phase === 1 },
    { label: 'Deep score', done: false, active: phase === 2 },
  ]

  // ── Bar colour: emerald phase-1, yellow phase-2 ──────────────────────
  const barClass = phase === 2
    ? 'bg-gradient-to-r from-yellow-500 to-amber-400'
    : 'bg-gradient-to-r from-emerald-500 to-teal-400'

  return (
    <div className="rounded-xl border border-gray-700/50 bg-gray-900/60 backdrop-blur p-4 space-y-4">

      {/* ── Header row ─────────────────────────────────────────────── */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
          <span className="text-[11px] font-bold font-mono text-white tracking-wider">
            SCANNING {discovery ? `${discovery.total_symbols.toLocaleString()}` : '8,000+'} STOCKS
          </span>
        </div>
        <div className="flex items-center gap-1.5">
          {steps.map((s, i) => (
            <Fragment key={s.label}>
              <span className={`text-[9px] font-mono px-1.5 py-0.5 rounded transition-all ${
                s.done    ? 'text-emerald-400 bg-emerald-500/15 border border-emerald-500/30' :
                s.active  ? 'text-yellow-300 bg-yellow-500/15 border border-yellow-500/30 animate-pulse' :
                            'text-gray-600 bg-transparent'
              }`}>
                {s.done ? '✓ ' : s.active ? '▶ ' : ''}{s.label}
              </span>
              {i < steps.length - 1 && <span className="text-gray-700 text-[9px]">──</span>}
            </Fragment>
          ))}
        </div>
      </div>

      {/* ── Main progress bar ───────────────────────────────────────── */}
      <div className="relative">
        <div className="h-3 bg-gray-800 rounded-full overflow-hidden">
          {pct !== null ? (
            <div
              className={`h-full ${barClass} rounded-full transition-all duration-500 ease-out`}
              style={{ width: `${pct}%` }}
            />
          ) : (
            /* Indeterminate slider while pre-screening */
            <div className={`h-full ${barClass} rounded-full w-1/4`}
                 style={{ animation: 'scanSlide 1.6s ease-in-out infinite' }} />
          )}
        </div>
        {pct !== null && (
          <div className="absolute right-0 -top-5 flex items-center gap-3 text-[10px] font-mono">
            {etaSec !== null && (
              <span className="text-gray-500">~{etaSec < 60 ? `${etaSec}s` : `${Math.ceil(etaSec/60)}m`} left</span>
            )}
            <span className={`font-bold ${phase === 2 ? 'text-yellow-300' : 'text-emerald-400'}`}>{pct}%</span>
          </div>
        )}
      </div>

      {/* ── Status line ────────────────────────────────────────────── */}
      <div className="flex items-center justify-between text-[11px] font-mono">
        <span className={`${phase === 2 ? 'text-yellow-300' : 'text-blue-300'} truncate max-w-[65%]`}>
          {progress
            ? phase === 2
              ? `★ Deep scoring ${progress.symbol}…`
              : `Fast scoring ${progress.symbol}…`
            : statusMsg || 'Initializing…'
          }
        </span>
        {progress && (
          <span className="text-gray-500 shrink-0">
            {progress.completed.toLocaleString()} / {progress.total.toLocaleString()}
          </span>
        )}
      </div>

      {/* ── Live counters ──────────────────────────────────────────── */}
      <div className="flex items-center gap-4 text-[11px] font-mono">
        {(buys + sells) > 0 && (
          <span className="flex items-center gap-1 text-emerald-400">
            <span className="text-[9px]">✦</span>
            {buys + sells} trade{buys + sells !== 1 ? 's' : ''} live
          </span>
        )}
        {holds > 0 && (
          <span className="text-amber-400">{holds} watchlist</span>
        )}
        {coilingHits > 0 && (
          <span className="flex items-center gap-1 text-orange-300">
            <span>◈</span>{coilingHits} compression
          </span>
        )}
        {mtfHits > 0 && (
          <span className="flex items-center gap-1 text-yellow-300">
            <span>⬟</span>{mtfHits} MTF cross
          </span>
        )}
        {discovery && (
          <span className="text-gray-500 ml-auto">
            {discovery.movers_found.toLocaleString()} screened · {discovery.total_symbols.toLocaleString()} queued
          </span>
        )}
      </div>

      {/* ── Symbol ticker ──────────────────────────────────────────── */}
      {recentSymbols.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {recentSymbols.slice(-24).map((s, i) => {
            const isDeep = (s as any).phase === 2
            const bg =
              s.verdict === 'BUY'   ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30' :
              s.verdict === 'SELL'  ? 'bg-red-500/20 text-red-300 border-red-500/30' :
              s.verdict === 'HOLD'  ? 'bg-amber-500/20 text-amber-300 border-amber-500/30' :
              s.verdict === 'ERROR' ? 'bg-red-900/30 text-red-600 border-red-900/30' :
                                      'bg-gray-800/40 text-gray-600 border-gray-800/30'
            return (
              <span
                key={`${s.symbol}-${i}`}
                className={`px-1.5 py-0.5 rounded border text-[9px] font-mono leading-none ${bg} ${
                  isDeep ? 'ring-1 ring-yellow-500/40' : ''
                }`}
                title={`${s.symbol} — ${s.verdict} (score ${s.score?.toFixed(0) ?? '—'})`}
              >
                {isDeep && <span className="text-yellow-400">★</span>}
                {s.symbol}
                {s.verdict !== 'NO_TRADE' && s.score > 0 && (
                  <span className="ml-0.5 opacity-60">{s.score.toFixed(0)}</span>
                )}
              </span>
            )
          })}
        </div>
      )}

      <style>{`
        @keyframes scanSlide {
          0%   { margin-left: -25%; }
          100% { margin-left: 100%; }
        }
      `}</style>
    </div>
  )
}

/** @deprecated use ScanLoadingPanel */
function ScanProgressBar({ progress, recentSymbols }: { progress: ScanProgress; recentSymbols: ScanProgress[] }) {
  return null  // replaced by ScanLoadingPanel
}

// ── Provider types ────────────────────────────────────────────────────────────

interface ProviderInfo {
  id:                string
  name:              string
  supports_screener: boolean
  supports_intraday: boolean
  active:            boolean
}

// ── ProviderModal ─────────────────────────────────────────────────────────────

function ProviderModal({ onClose }: { onClose: () => void }) {
  const [providers, setProviders]       = useState<ProviderInfo[]>([])
  const [loading, setLoading]           = useState(true)
  const [switching, setSwitching]       = useState(false)
  const [testing, setTesting]           = useState<string | null>(null)
  const [testResult, setTestResult]     = useState<Record<string, any>>({})
  const [fmpKey, setFmpKey]             = useState('')
  const [savingKey, setSavingKey]       = useState(false)
  const [activeId, setActiveId]         = useState<string>('')

  const baseUrl = (api.defaults.baseURL || '/api').replace(/\/$/, '')

  useEffect(() => {
    fetch(`${baseUrl}/settings/providers`)
      .then(r => r.json())
      .then(d => {
        setProviders(d.providers || [])
        const act = (d.providers || []).find((p: ProviderInfo) => p.active)
        if (act) setActiveId(act.id)
      })
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [])

  const handleSwitch = async (id: string) => {
    setSwitching(true)
    try {
      await fetch(`${baseUrl}/settings/providers/switch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ provider_id: id }),
      })
      setActiveId(id)
      setProviders(prev => prev.map(p => ({ ...p, active: p.id === id })))
    } catch {}
    setSwitching(false)
  }

  const handleTest = async (id: string) => {
    setTesting(id)
    setTestResult(prev => ({ ...prev, [id]: null }))
    try {
      const body = id === 'fmp' && fmpKey ? JSON.stringify({ api_key: fmpKey }) : undefined
      const r = await fetch(`${baseUrl}/settings/test/${id}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body,
      })
      const d = await r.json()
      setTestResult(prev => ({ ...prev, [id]: d }))
    } catch (e: any) {
      setTestResult(prev => ({ ...prev, [id]: { ok: false, error: e.message } }))
    }
    setTesting(null)
  }

  const handleSaveFmpKey = async () => {
    if (!fmpKey.trim()) return
    setSavingKey(true)
    try {
      await fetch(`${baseUrl}/settings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fmp: { api_key: fmpKey, enabled: true } }),
      })
    } catch {}
    setSavingKey(false)
  }

  const PROVIDER_ICONS: Record<string, string> = {
    fmp:      '📊',
    yfinance: '🐍',
  }

  const PROVIDER_DESC: Record<string, string> = {
    fmp:      'Full-market screener · Fast batch bars · Intraday on Starter+',
    yfinance: 'Free · No key needed · Good for testing · Slower bulk scans',
  }

  return createPortal(
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm"
      onClick={(e) => e.target === e.currentTarget && onClose()}
    >
      <div className="bg-gray-950 border border-gray-700 rounded-2xl w-full max-w-lg mx-4 shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-800">
          <div>
            <h2 className="text-white font-bold text-base">Data Provider</h2>
            <p className="text-gray-500 text-xs mt-0.5">Choose where scan data comes from</p>
          </div>
          <button onClick={onClose} className="text-gray-500 hover:text-white text-xl leading-none">×</button>
        </div>

        {/* Body */}
        <div className="px-6 py-4 space-y-3">
          {loading && (
            <div className="text-center py-8 text-gray-500 text-sm">Loading providers…</div>
          )}

          {!loading && providers.map(prov => {
            const res = testResult[prov.id]
            const isActive = prov.id === activeId
            return (
              <div
                key={prov.id}
                className={`rounded-xl border p-4 transition-all ${
                  isActive
                    ? 'border-emerald-600/60 bg-emerald-950/20'
                    : 'border-gray-800 bg-gray-900/40 hover:border-gray-600'
                }`}
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="text-lg">{PROVIDER_ICONS[prov.id] || '🔌'}</span>
                      <span className="text-white font-semibold text-sm">{prov.name}</span>
                      {isActive && (
                        <span className="px-1.5 py-0.5 bg-emerald-600/30 text-emerald-400 text-[10px] font-bold rounded border border-emerald-600/40">
                          ACTIVE
                        </span>
                      )}
                    </div>
                    <p className="text-gray-500 text-[11px] mt-1 leading-relaxed">
                      {PROVIDER_DESC[prov.id] || ''}
                    </p>
                    <div className="flex gap-3 mt-2 text-[10px]">
                      <span className={`flex items-center gap-1 ${prov.supports_screener ? 'text-emerald-400' : 'text-gray-600'}`}>
                        {prov.supports_screener ? '✓' : '✗'} Market Screener
                      </span>
                      <span className={`flex items-center gap-1 ${prov.supports_intraday ? 'text-emerald-400' : 'text-gray-600'}`}>
                        {prov.supports_intraday ? '✓' : '✗'} Intraday Bars
                      </span>
                    </div>
                  </div>
                  <div className="flex flex-col gap-2 shrink-0">
                    <button
                      onClick={() => handleTest(prov.id)}
                      disabled={testing === prov.id}
                      className="px-3 py-1 text-[10px] font-bold bg-gray-800 hover:bg-gray-700 text-gray-300 rounded border border-gray-700 transition-colors disabled:opacity-40"
                    >
                      {testing === prov.id ? '…' : 'Test'}
                    </button>
                    {!isActive && (
                      <button
                        onClick={() => handleSwitch(prov.id)}
                        disabled={switching}
                        className="px-3 py-1 text-[10px] font-bold bg-emerald-700 hover:bg-emerald-600 text-white rounded transition-colors disabled:opacity-40"
                      >
                        Use This
                      </button>
                    )}
                  </div>
                </div>

                {/* Test result */}
                {res !== undefined && res !== null && (
                  <div className={`mt-2 px-3 py-2 rounded-lg text-[11px] font-mono ${
                    res.ok ? 'bg-emerald-950/40 text-emerald-300 border border-emerald-800/40' : 'bg-red-950/40 text-red-300 border border-red-800/40'
                  }`}>
                    {res.ok
                      ? `✓ ${res.sample}  ·  ${res.tier || 'free'}  ·  ${res.elapsed_ms}ms`
                      : `✗ ${res.error}`
                    }
                  </div>
                )}

                {/* FMP key input */}
                {prov.id === 'fmp' && (
                  <div className="mt-3 flex gap-2">
                    <input
                      type="password"
                      value={fmpKey}
                      onChange={e => setFmpKey(e.target.value)}
                      placeholder="Paste FMP API key…"
                      className="flex-1 px-3 py-1.5 bg-gray-800 border border-gray-700 rounded text-xs text-white font-mono placeholder-gray-600 focus:outline-none focus:border-emerald-600"
                    />
                    <button
                      onClick={handleSaveFmpKey}
                      disabled={!fmpKey.trim() || savingKey}
                      className="px-3 py-1.5 text-[10px] font-bold bg-blue-700 hover:bg-blue-600 text-white rounded transition-colors disabled:opacity-40"
                    >
                      {savingKey ? '…' : 'Save'}
                    </button>
                  </div>
                )}
              </div>
            )
          })}
        </div>

        {/* Footer note */}
        <div className="px-6 py-3 border-t border-gray-800">
          <p className="text-gray-600 text-[10px]">
            Switch takes effect on next scan. FMP Starter plan recommended for full intraday + screener.
          </p>
        </div>
      </div>
    </div>,
    document.body
  )
}

// ── TradesPage ────────────────────────────────────────────────────────────────

export default function TradesPage() {
  const [mode, setMode] = useState<Mode>('scanner')
  const [profile, setProfile] = useState<Profile>('SWING')
  const [account, setAccount] = useState(10000)
  const [showProviderModal, setShowProviderModal] = useState(false)
  const [detailTrade, setDetailTrade] = useState<TradeRec | null>(null)

  // Scan type: single-select ('all' | 'bluechip' | 'growth' | 'etfs' | 'penny')
  const [scanType, setScanType] = useState<ScanTypeId>('all')

  // Setup filters: multi-select chips (client-side result filters)
  const [activeSetups, setActiveSetups] = useState<string[]>([])

  // selectedCategories sent to backend: [scanType (if not 'all')] + activeSetups
  const selectedCategories = [
    ...(scanType !== 'all' ? [scanType] : []),
    ...activeSetups,
  ]

  const [scanDepth, setScanDepth] = useState(500)
  const [sortKey, setSortKey]   = useState<SortKey>('score')
  const [sortAsc, setSortAsc]   = useState(false)   // default: descending (best first)
  const [dirFilters, setDirFilters] = useState<DirFilter[]>([])  // empty = all
  const [viewMode, setViewMode] = useState<ViewMode>('cards')
  const [chartOverlaySymbol, setChartOverlaySymbol] = useState<string | null>(null)
  const [filtersCollapsed, setFiltersCollapsed] = useState(true)

  // TF filter — show only trades where regime cross is on selected TF(s)
  const [tfFilters, setTfFilters] = useState<string[]>([])  // empty = all

  // Pattern filter — filter scan results by detected pattern name(s)
  const [patternFilters, setPatternFilters] = useState<string[]>([])

  const togglePatternFilter = useCallback((name: string) => {
    setPatternFilters(prev => prev.includes(name) ? prev.filter(p => p !== name) : [...prev, name])
  }, [])

  const toggleTfFilter = useCallback((tf: string) => {
    setTfFilters(prev => prev.includes(tf) ? prev.filter(t => t !== tf) : [...prev, tf])
  }, [])

  const toggleDirFilter = useCallback((d: DirFilter) => {
    setDirFilters(prev => prev.includes(d) ? prev.filter(x => x !== d) : [...prev, d])
  }, [])

  const toggleSetup = useCallback((id: string) => {
    setActiveSetups(prev => prev.includes(id) ? prev.filter(s => s !== id) : [...prev, id])
  }, [])

  // Derived booleans from activeSetups (used in result filtering below)
  const goldenCrossFilter    = activeSetups.includes('goldencross')
  const deathCrossFilter     = activeSetups.includes('deathcross')
  const maBreakoutFilter     = activeSetups.includes('maBreakout')
  const rsLeaderFilter       = activeSetups.includes('rsleader')
  const squeezeFilter        = activeSetups.includes('squeeze')
  const volumeClimaxFilter   = activeSetups.includes('volclimaxfilter')
  const mtfConfluenceFilter  = activeSetups.includes('mtfconfluence')
  const coilingFilter        = activeSetups.includes('coiling') || activeSetups.includes('coiling3tf')
  const coiling3tfFilter     = activeSetups.includes('coiling3tf')
  const bearFadeFilter       = activeSetups.includes('bearfade')

  const getSortVal = useCallback((r: TradeRec): number | string => {
    switch (sortKey) {
      case 'score':    return r.score
      case 'gain_pct': return r.projected_profit_pct
      case 'change_pct': return (r as any).change_pct ?? 0
      case 'price':    return r.entry
      case 'symbol':   return r.symbol
      case 'entry':    return r.entry
      case 'stop':     return r.stop
      case 'target':   return r.target
      case 'rr':       return r.risk_reward
      case 'size':     return r.position_size
      case 'risk':     return r.risk_amount
      default:         return r.score
    }
  }, [sortKey])

  const sortTrades = useCallback((arr: TradeRec[]) =>
    [...arr].sort((a, b) => {
      const av = getSortVal(a)
      const bv = getSortVal(b)
      if (typeof av === 'string' && typeof bv === 'string') {
        return sortAsc ? av.localeCompare(bv) : bv.localeCompare(av)
      }
      return sortAsc ? (av as number) - (bv as number) : (bv as number) - (av as number)
    }), [getSortVal, sortAsc])

  const handleSort = useCallback((key: SortKey) => {
    if (sortKey === key) setSortAsc(a => !a)
    else { setSortKey(key); setSortAsc(false) }
  }, [sortKey])

  const { progress, results: scanData, isScanning, error: scanError, recentSymbols, liveTrades, regime, discovery, statusMsg, scanLog, startScan, abort, quoteFlash } = useStreamingScan()

  // Collect all unique pattern names from current scan data for filter pills
  const allPatternNames = (() => {
    const trades: any[] = scanData ? scanData.recommendations : liveTrades
    const seen = new Set<string>()
    const names: string[] = []
    for (const r of trades) {
      for (const p of (r.patterns || [])) {
        if (p && !seen.has(p)) { seen.add(p); names.push(p) }
      }
    }
    return names.sort()
  })()

  // Persist regime to localStorage so sidebar can display it globally
  useEffect(() => {
    if (regime) {
      localStorage.setItem('robo_market_regime', JSON.stringify(regime))
      window.dispatchEvent(new Event('storage'))
    }
  }, [regime])

  // Auto re-scan when profile changes (only if we already have results)
  const prevProfileRef = useRef(profile)
  useEffect(() => {
    if (prevProfileRef.current !== profile && mode === 'scanner' && (scanData || liveTrades.length > 0) && !isScanning) {
      prevProfileRef.current = profile
      startScan(profile, account, undefined, selectedCategories, scanDepth)
    } else {
      prevProfileRef.current = profile
    }
  }, [profile])

  const handleScanStart = () => {
    setMode('scanner')
    setFiltersCollapsed(true)
    startScan(profile, account, undefined, selectedCategories, scanDepth)
  }

  return (
    <div className="space-y-4">
      {/* Header + mode tabs in one row */}
      <div className="flex items-center gap-3 flex-wrap">
        <h1 className="text-xl font-bold text-white mr-2">TRADES</h1>
        {(['scanner', 'lookup'] as Mode[]).map((m) => (
          <button
            key={m}
            onClick={() => setMode(m)}
            className={`px-4 py-1.5 rounded text-xs font-bold uppercase transition-colors ${
              mode === m ? 'bg-emerald-600 text-white' : 'bg-gray-900/50 text-gray-500 hover:text-gray-300'
            }`}
          >
            {m === 'lookup' ? 'Symbol Lookup' : 'Market Scan'}
          </button>
        ))}
        <div className="w-px h-5 bg-gray-700 mx-1" />
        {/* Profile toggle buttons — click to cycle */}
        <div className="flex rounded overflow-hidden border border-gray-700/60">
          {([
            { id: 'SCALP',    label: 'SCALP',    sub: '5-15m'  },
            { id: 'SWING',    label: 'SWING',    sub: '1H-4H'  },
            { id: 'POSITION', label: 'POSITION', sub: 'Daily'  },
          ] as { id: Profile; label: string; sub: string }[]).map((p, i) => (
            <button
              key={p.id}
              onClick={() => setProfile(p.id)}
              className={`px-3 py-1.5 text-[11px] font-bold font-mono transition-colors flex flex-col items-center leading-tight
                ${i > 0 ? 'border-l border-gray-700/60' : ''}
                ${profile === p.id
                  ? 'bg-emerald-600 text-white'
                  : 'bg-gray-900 text-gray-500 hover:text-gray-300 hover:bg-gray-800'
                }`}
            >
              <span>{p.label}</span>
              <span className={`text-[8px] font-normal ${profile === p.id ? 'text-emerald-200' : 'text-gray-600'}`}>{p.sub}</span>
            </button>
          ))}
        </div>
        <div className="relative">
          <span className="absolute left-2 top-1/2 -translate-y-1/2 text-gray-500 text-xs font-mono">$</span>
          <input
            type="text"
            value={formatWithCommas(account)}
            onChange={(e) => setAccount(parseCommaNumber(e.target.value))}
            className="w-32 pl-5 pr-2.5 py-1.5 bg-gray-900 border border-gray-700 rounded text-xs text-white font-mono text-right"
            placeholder="Account"
          />
        </div>
        {/* Data provider gear */}
        <button
          onClick={() => setShowProviderModal(true)}
          title="Data provider settings"
          className="ml-auto p-1.5 text-gray-500 hover:text-gray-300 hover:bg-gray-800 rounded-lg transition-colors text-base leading-none"
        >
          ⚙
        </button>
      </div>

      {showProviderModal && <ProviderModal onClose={() => setShowProviderModal(false)} />}

      {/* Content */}
      {mode === 'lookup' ? (
        <LookupPanel profile={profile} account={account} />
      ) : (
        <div className="space-y-4">
          {/* Scan controls — collapsible */}
          {filtersCollapsed ? (
            <div
              onClick={() => setFiltersCollapsed(false)}
              className="flex items-center gap-3 px-3 py-2 bg-gray-900/60 border border-gray-800 rounded-lg cursor-pointer hover:border-gray-600 transition-colors group"
            >
              <span className="text-gray-500 group-hover:text-gray-300 text-xs">▸ Filters</span>
              <div className="flex items-center gap-2 flex-wrap text-[10px] font-mono">
                <span className="px-1.5 py-0.5 bg-emerald-500/15 text-emerald-400 rounded">
                  {SCAN_TYPES.find(t => t.id === scanType)?.label ?? 'All'}
                </span>
                <span className="px-1.5 py-0.5 bg-gray-800 text-gray-400 rounded">
                  {SCAN_DEPTHS.find(d => d.id === scanDepth)?.label || scanDepth}
                </span>
                {dirFilters.length > 0 && (
                  <span className="px-1.5 py-0.5 bg-amber-500/15 text-amber-400 rounded">
                    {dirFilters.join(', ')}
                  </span>
                )}
                {activeSetups.length > 0 && (
                  <span className="px-1.5 py-0.5 bg-purple-500/15 text-purple-400 rounded">
                    {activeSetups.length} setup{activeSetups.length > 1 ? 's' : ''}
                  </span>
                )}
                {patternFilters.length > 0 && (
                  <span className="px-1.5 py-0.5 bg-purple-500/15 text-purple-400 rounded">
                    Pattern: {patternFilters.join(', ')}
                  </span>
                )}
                <span className="px-1.5 py-0.5 bg-gray-800 text-gray-400 rounded">
                  Sort: {sortKey === 'score' ? 'Score' : sortKey === 'gain_pct' ? '% Gain' : 'Price'} {sortAsc ? '↑' : '↓'}
                </span>
              </div>
              <span className="ml-auto text-gray-600 group-hover:text-gray-400 text-[10px]">click to expand</span>
            </div>
          ) : (
          <div className="bg-gray-900/40 border border-gray-800 rounded-xl p-4 space-y-4">

            {/* Row 1: SCAN TYPE (single-select radio pills) + Scan Action */}
            <div className="flex items-center gap-4">
              <div className="space-y-1.5 flex-1">
                <div className="text-[10px] text-gray-500 uppercase tracking-wider font-semibold">Scan Type</div>
                <div className="flex items-center gap-1.5 flex-wrap">
                  {SCAN_TYPES.map(t => {
                    const active = scanType === t.id
                    return (
                      <button
                        key={t.id}
                        onClick={() => setScanType(t.id)}
                        className={`px-2.5 py-1 rounded-md text-[11px] font-mono font-semibold transition-all ${
                          active
                            ? 'bg-emerald-500/20 text-emerald-400 ring-1 ring-emerald-500/40'
                            : 'bg-gray-800/40 text-gray-500 hover:text-gray-300 hover:bg-gray-800/60'
                        }`}
                      >{t.label}</button>
                    )
                  })}
                </div>
              </div>
              {/* Scan Action — same row, pushed right */}
              <div className="flex items-center gap-2 self-end pb-0.5">
                {isScanning && (
                  <button
                    onClick={abort}
                    className="px-4 py-1.5 bg-red-500/10 hover:bg-red-500/20 text-red-400
                               ring-1 ring-red-500/30 rounded-lg text-xs font-bold transition-all"
                  >ABORT</button>
                )}
                <button
                  onClick={handleScanStart}
                  disabled={isScanning}
                  className="px-6 py-2 bg-emerald-600 hover:bg-emerald-500 disabled:bg-gray-700 disabled:text-gray-500
                             text-white rounded-lg text-xs font-bold tracking-wide transition-all shadow-lg shadow-emerald-900/30"
                >{isScanning ? 'SCANNING...' : 'RUN SCAN'}</button>
              </div>
            </div>

            {/* Divider */}
            <div className="border-t border-gray-800/60" />

            {/* Row 2: DIRECTION | DEPTH | SORT (inline) */}
            <div className="flex items-start gap-4 flex-wrap">
              {/* Direction */}
              <div className="space-y-1.5">
                <div className="text-[10px] text-gray-500 uppercase tracking-wider font-semibold">Direction</div>
                <div className="flex items-center gap-1">
                  {([
                    { id: 'long' as DirFilter, label: 'Long', activeClass: 'bg-emerald-500/15 text-emerald-400 ring-1 ring-emerald-500/30' },
                    { id: 'short' as DirFilter, label: 'Short', activeClass: 'bg-red-500/15 text-red-400 ring-1 ring-red-500/30' },
                    { id: 'watchlist' as DirFilter, label: 'Both', activeClass: 'bg-amber-500/15 text-amber-400 ring-1 ring-amber-500/30' },
                  ]).map(d => {
                    const active = dirFilters.includes(d.id)
                    return (
                      <button
                        key={d.id}
                        onClick={() => toggleDirFilter(d.id)}
                        className={`px-2.5 py-1 rounded-md text-[11px] font-mono transition-all ${
                          active ? d.activeClass : 'bg-gray-800/40 text-gray-500 hover:text-gray-300 hover:bg-gray-800/60'
                        }`}
                      >{d.label}</button>
                    )
                  })}
                </div>
              </div>

              <div className="w-px h-10 bg-gray-800/60 self-center" />

              {/* Depth */}
              <div className="space-y-1.5">
                <div className="text-[10px] text-gray-500 uppercase tracking-wider font-semibold">Depth</div>
                <div className="flex items-center gap-1 flex-wrap">
                  {SCAN_DEPTHS.map(d => (
                    <button
                      key={d.id}
                      onClick={() => !isScanning && setScanDepth(d.id)}
                      className={`px-2.5 py-1 rounded-md text-[11px] font-mono transition-all ${
                        isScanning ? 'opacity-50 cursor-not-allowed' : ''
                      } ${
                        scanDepth === d.id
                          ? 'bg-emerald-500/15 text-emerald-400 ring-1 ring-emerald-500/30'
                          : 'bg-gray-800/40 text-gray-500 hover:text-gray-300 hover:bg-gray-800/60'
                      }`}
                      title={d.desc}
                    >{d.label}</button>
                  ))}
                </div>
              </div>

              <div className="w-px h-10 bg-gray-800/60 self-center" />

              {/* Sort */}
              <div className="space-y-1.5">
                <div className="text-[10px] text-gray-500 uppercase tracking-wider font-semibold">Sort</div>
                <div className="flex items-center gap-1">
                  {([
                    { k: 'score' as SortKey, label: 'Score' },
                    { k: 'gain_pct' as SortKey, label: '%Gain' },
                    { k: 'price' as SortKey, label: 'Price' },
                    { k: 'rr' as SortKey, label: 'R:R' },
                  ]).map(({ k, label }) => (
                    <button
                      key={k}
                      onClick={() => handleSort(k)}
                      className={`px-2 py-1 rounded-md text-[11px] font-mono transition-all ${
                        sortKey === k
                          ? 'bg-emerald-500/15 text-emerald-400 ring-1 ring-emerald-500/30'
                          : 'bg-gray-800/40 text-gray-500 hover:text-gray-300 hover:bg-gray-800/60'
                      }`}
                    >{label}{sortKey === k && <span className="ml-0.5 text-[10px]">{sortAsc ? '↑' : '↓'}</span>}</button>
                  ))}
                </div>
              </div>
            </div>

            {/* Divider */}
            <div className="border-t border-gray-800/60" />

            {/* Row 3: SETUPS (multi-select chips — combines former Setups + MA Cross) */}
            <div className="space-y-1.5">
              <div className="flex items-center gap-2">
                <div className="text-[10px] text-gray-500 uppercase tracking-wider font-semibold">Setups</div>
                {activeSetups.length > 0 && (
                  <button
                    onClick={() => setActiveSetups([])}
                    className="text-[10px] text-gray-500 hover:text-gray-300 transition-colors"
                  >clear</button>
                )}
              </div>
              <div className="flex items-center gap-1.5 flex-wrap">
                {([
                  { id: 'coiling3tf',     label: '◆ GENESIS',      title: 'Origin of cross — Weekly + Daily + Hourly all at the first bar of confirmed bull cross, K<25 with K-D spread ≤3 on each TF. Rarest, highest precision entry.', activeClass: 'bg-amber-400/25 text-amber-200 ring-1 ring-amber-400/60' },
                  { id: 'coiling',        label: '◈ COMPRESSION',  title: 'Pre-launch coil — Daily + Weekly at the first bar of confirmed bull cross, K-D spread tight (≤3 daily, ≤5 weekly). Two-TF alignment before expansion.', activeClass: 'bg-orange-500/20 text-orange-200 ring-1 ring-orange-500/40' },
                  { id: 'bearfade',       label: '▽ DISTRIBUTION', title: 'Overbought rotation — Weekly in bull (K>55, already ran) while Daily + Hourly print fresh bear crosses from >75. Short via paired inverse ETF.', activeClass: 'bg-red-500/20 text-red-200 ring-1 ring-red-500/40' },
                  { id: 'mtfconfluence',  label: '⬟ CONFLUENCE',   title: 'Multi-TF agreement ≥ GOOD — three or more timeframes carrying fresh crosses in the same direction. Broader than GENESIS, catches setups already in motion.', activeClass: 'bg-yellow-500/20 text-yellow-200 ring-1 ring-yellow-500/40' },
                ] as const).map(s => {
                  const active = activeSetups.includes(s.id)
                  return (
                    <button
                      key={s.id}
                      onClick={() => toggleSetup(s.id)}
                      title={s.title}
                      className={`px-2.5 py-1 rounded-md text-[11px] font-mono transition-all ${
                        active ? s.activeClass : 'bg-gray-800/40 text-gray-500 hover:text-gray-300 hover:bg-gray-800/60'
                      }`}
                    >{s.label}</button>
                  )
                })}
              </div>
            </div>

            {/* Row 4: Patterns (only if patterns exist from current scan) */}
            {allPatternNames.length > 0 && (
              <>
                <div className="border-t border-gray-800/60" />
                <div className="space-y-1.5">
                  <div className="flex items-center gap-2">
                    <div className="text-[10px] text-gray-500 uppercase tracking-wider font-semibold">
                      Patterns
                      <span className="text-purple-400 ml-1">({allPatternNames.length})</span>
                    </div>
                    {patternFilters.length > 0 && (
                      <button
                        onClick={() => setPatternFilters([])}
                        className="text-[10px] text-gray-500 hover:text-gray-300 transition-colors"
                      >clear</button>
                    )}
                  </div>
                  <div className="flex flex-wrap gap-1.5 max-h-24 overflow-y-auto">
                    {allPatternNames.map(name => {
                      const active = patternFilters.includes(name)
                      const trades: any[] = scanData ? scanData.recommendations : liveTrades
                      const count = trades.filter((r: any) => (r.patterns || []).includes(name)).length
                      return (
                        <button
                          key={name}
                          onClick={() => togglePatternFilter(name)}
                          className={`px-2 py-0.5 rounded-md text-[10px] font-mono transition-all ${
                            active
                              ? 'bg-purple-500/20 text-purple-300 ring-1 ring-purple-500/40'
                              : 'bg-gray-800/40 text-gray-500 hover:text-gray-300 hover:bg-gray-800/60'
                          }`}
                        >{name} <span className="opacity-50">{count}</span></button>
                      )
                    })}
                  </div>
                </div>
              </>
            )}

            {/* Collapse bar */}
            <div className="flex justify-center pt-1">
              <button
                onClick={() => setFiltersCollapsed(true)}
                className="px-6 py-1 text-gray-600 hover:text-gray-400 text-[10px] font-mono transition-colors flex items-center gap-1.5 rounded-full hover:bg-gray-800/40"
              >
                <span className="text-[8px]">▲</span> Collapse Filters
              </button>
            </div>
          </div>
          )}

          {/* Regime is shown in the left sidebar */}

          {/* Progress bar — indeterminate slide during pre-screen, real bar once scoring starts */}
          {isScanning && (
            <ScanLoadingPanel
              progress={progress}
              recentSymbols={recentSymbols}
              statusMsg={statusMsg}
              discovery={discovery}
              liveTrades={liveTrades}
            />
          )}

          {/* Regime is shown in the left sidebar (terminal is a fixed overlay — see below) */}

          {scanError && (
            <div className="text-red-400 text-sm p-3 bg-red-500/10 border border-red-500/20 rounded-lg">
              Scan failed: {scanError}
            </div>
          )}

          {/* Results — filtered by direction + TF regime cross + price/asset type */}
          {(() => {
            const trades = scanData ? scanData.recommendations : liveTrades
            const noTrade = scanData ? scanData.no_trade : []

            // TF filter: only show trades where regime cross matches any selected TF
            const tfFiltered = tfFilters.length > 0
              ? trades.filter((r: TradeRec) => {
                  const gate = r.component_details?.stoch_gate
                  return gate?.regime_tf && tfFilters.includes(gate.regime_tf)
                })
              : trades

            // Client-side price range + asset type filter (always runs)
            const catFiltered = tfFiltered.filter((r: TradeRec) => tradePassesFilters(r, selectedCategories))

            // Golden / Death cross filter (client-side, instant)
            const crossFiltered = (goldenCrossFilter || deathCrossFilter)
              ? catFiltered.filter((r: TradeRec) => {
                  const gc = r.component_details?.golden_death_cross
                  if (!gc) return false
                  if (goldenCrossFilter && deathCrossFilter) return gc.golden_cross || gc.death_cross
                  if (goldenCrossFilter) return gc.golden_cross
                  return gc.death_cross
                })
              : catFiltered

            // MA Cluster Breakout filter (client-side, instant)
            const maFiltered = maBreakoutFilter
              ? crossFiltered.filter((r: any) => r.ma_cluster_breakout === true)
              : crossFiltered

            // RS Leaders filter
            const rsFiltered = rsLeaderFilter
              ? maFiltered.filter((r: any) => r.rs_leader === true)
              : maFiltered

            // Squeeze filter
            const squeezeFiltered = squeezeFilter
              ? rsFiltered.filter((r: any) => r.squeeze_firing === true)
              : rsFiltered

            // Volume Climax filter
            const volFiltered = volumeClimaxFilter
              ? squeezeFiltered.filter((r: any) => r.volume_climax === true)
              : squeezeFiltered

            // Pattern filter (client-side, instant)
            const patternFiltered = patternFilters.length > 0
              ? volFiltered.filter((r: any) => {
                  const pats: string[] = r.patterns || []
                  return patternFilters.some(f => pats.includes(f))
                })
              : volFiltered

            // COILING filter — daily K<35 + weekly K<40, cross confirmed on both TFs
            // coiling3tfFilter = stricter: all 3 TFs (weekly+daily+hourly) at the start
            const coilingFiltered = coilingFilter
              ? patternFiltered.filter((r: TradeRec) => {
                  const gate = r.component_details?.stoch_gate ?? (r as any)
                  const has3tf = gate.coiling_3tf === true || (r as any).coiling_3tf === true
                  const has2tf = gate.coiling === true || (r as any).coiling === true
                  if (coiling3tfFilter) return has3tf
                  return has2tf || has3tf
                })
              : patternFiltered

            // BEAR FADE filter — weekly already ran, daily+hourly overbought bear cross → inverse ETF
            const bearFadeFiltered = bearFadeFilter
              ? coilingFiltered.filter((r: TradeRec) =>
                  (r as any).bear_fade === true ||
                  r.component_details?.stoch_gate?.bear_fade === true
                )
              : coilingFiltered

            // MTF Stoch Confluence filter — only show GOOD/STRONG/PERFECT (confluence_level >= 3)
            const mtfFiltered = mtfConfluenceFilter
              ? bearFadeFiltered.filter((r: TradeRec) => {
                  const cl = (r as any).confluence_level
                    ?? (r.component_details?.stoch_gate?.confluence_level ?? 0)
                  return cl >= 3
                })
              : bearFadeFiltered

            const allLongs  = sortTrades(mtfFiltered.filter((r: TradeRec) => r.verdict === 'BUY'))
            const allShorts = sortTrades(mtfFiltered.filter((r: TradeRec) => r.verdict === 'SELL'))
            const holds     = sortTrades(mtfFiltered.filter((r: TradeRec) => r.verdict === 'HOLD'))

            // Apply direction filter
            // Default (nothing checked): show Long + Short, but NOT watchlist
            // Watchlist only shows when explicitly checked
            const showLongs  = dirFilters.length === 0 || dirFilters.includes('long')  ? allLongs  : []
            const showShorts = dirFilters.length === 0 || dirFilters.includes('short') ? allShorts : []
            const showHolds  = dirFilters.includes('watchlist') ? holds : []

            return (
              <>
                {/* Stats bar + view toggle — visible during scan AND after */}
                {(scanData || liveTrades.length > 0) && (
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4 text-xs text-gray-500 font-mono">
                      {scanData ? (
                        <>
                          <span>{scanData.symbols_scanned} scanned</span>
                          <span>{scanData.scan_time}s</span>
                        </>
                      ) : progress ? (
                        <span>{progress.completed}/{progress.total} scanned</span>
                      ) : null}
                      <span className="text-emerald-400">{allLongs.length} long</span>
                      <span className="text-red-400">{allShorts.length} short</span>
                      <span className="text-amber-400">{holds.length} watch</span>
                      {scanData && <span>{noTrade.length} rejected</span>}
                      {tfFilters.length > 0 && <span className="text-blue-400">regime: {tfFilters.map(t => t.toUpperCase()).join(', ')}</span>}
                      {patternFilters.length > 0 && <span className="text-purple-400">pattern: {patternFiltered.length} match</span>}
                      {coiling3tfFilter && <span className="text-amber-300">◆ Genesis: {coilingFiltered.length}</span>}
                      {coilingFilter && !coiling3tfFilter && <span className="text-orange-300">◈ Compression: {coilingFiltered.length}</span>}
                      {bearFadeFilter && <span className="text-red-300">▽ Distribution: {bearFadeFiltered.length}</span>}
                      {mtfConfluenceFilter && <span className="text-yellow-300">⬟ Confluence: {mtfFiltered.length}</span>}
                      {(scanType !== 'all' || goldenCrossFilter || deathCrossFilter) && <span className="text-purple-400">filtered: {crossFiltered.length}/{tfFiltered.length}</span>}
                    </div>
                    <div className="flex items-center border border-gray-700 rounded overflow-hidden">
                      {([
                        { id: 'cards' as ViewMode, label: '▦', title: 'Card view' },
                        { id: 'list'  as ViewMode, label: '☰', title: 'List view' },
                        { id: 'charts' as ViewMode, label: '▣', title: 'Chart grid' },
                      ]).map(v => (
                        <button
                          key={v.id}
                          onClick={() => setViewMode(v.id)}
                          title={v.title}
                          className={`px-2.5 py-1 text-xs font-mono transition-colors ${
                            viewMode === v.id
                              ? 'bg-gray-700 text-white'
                              : 'bg-gray-900 text-gray-500 hover:text-gray-300'
                          }`}
                        >{v.label}</button>
                      ))}
                    </div>
                  </div>
                )}

                {/* ── Card View (default) ── */}
                {viewMode === 'cards' && (
                  <>
                    {showLongs.length > 0 && (
                      <div>
                        <h3 className="text-emerald-400 text-sm font-bold uppercase tracking-wider mb-2">
                          {isScanning ? 'LIVE — ' : ''}Long ({showLongs.length})
                        </h3>
                        <div className="grid grid-cols-2 gap-2">
                          {showLongs.map((rec: TradeRec) => <TradeCard key={rec.symbol} rec={rec} onOpenChart={setChartOverlaySymbol} account={account} onOpenDetail={setDetailTrade} />)}
                        </div>
                      </div>
                    )}
                    {showShorts.length > 0 && (
                      <div>
                        <h3 className="text-red-400 text-sm font-bold uppercase tracking-wider mb-2">
                          {isScanning ? 'LIVE — ' : ''}Short ({showShorts.length})
                        </h3>
                        <div className="grid grid-cols-2 gap-2">
                          {showShorts.map((rec: TradeRec) => <TradeCard key={rec.symbol} rec={rec} onOpenChart={setChartOverlaySymbol} account={account} onOpenDetail={setDetailTrade} />)}
                        </div>
                      </div>
                    )}
                    {showHolds.length > 0 && (
                      <div>
                        <h3 className="text-amber-400 text-sm font-bold uppercase tracking-wider mb-2">
                          Watchlist ({showHolds.length})
                        </h3>
                        <div className="grid grid-cols-2 gap-2">
                          {showHolds.map((rec: TradeRec) => <TradeCard key={rec.symbol} rec={rec} onOpenChart={setChartOverlaySymbol} account={account} onOpenDetail={setDetailTrade} />)}
                        </div>
                      </div>
                    )}
                  </>
                )}

                {/* ── List View ── */}
                {viewMode === 'list' && (
                  <>
                    {showLongs.length > 0 && (
                      <div>
                        <h3 className="text-emerald-400 text-sm font-bold uppercase tracking-wider mb-2">Long ({showLongs.length})</h3>
                        <TradeListView trades={showLongs} onOpenChart={setChartOverlaySymbol} sortKey={sortKey} sortAsc={sortAsc} onSort={handleSort} account={account} quoteFlash={quoteFlash} onOpenDetail={setDetailTrade} />
                      </div>
                    )}
                    {showShorts.length > 0 && (
                      <div>
                        <h3 className="text-red-400 text-sm font-bold uppercase tracking-wider mb-2">Short ({showShorts.length})</h3>
                        <TradeListView trades={showShorts} onOpenChart={setChartOverlaySymbol} sortKey={sortKey} sortAsc={sortAsc} onSort={handleSort} account={account} quoteFlash={quoteFlash} onOpenDetail={setDetailTrade} />
                      </div>
                    )}
                    {showHolds.length > 0 && (
                      <div>
                        <h3 className="text-amber-400 text-sm font-bold uppercase tracking-wider mb-2">Watchlist ({showHolds.length})</h3>
                        <TradeListView trades={showHolds} onOpenChart={setChartOverlaySymbol} sortKey={sortKey} sortAsc={sortAsc} onSort={handleSort} account={account} quoteFlash={quoteFlash} onOpenDetail={setDetailTrade} />
                      </div>
                    )}
                  </>
                )}

                {/* ── Chart Grid View (klinecharts + pattern overlays) ── */}
                {viewMode === 'charts' && (
                  <>
                    {showLongs.length > 0 && (
                      <div>
                        <h3 className="text-emerald-400 text-sm font-bold uppercase tracking-wider mb-2">Long ({showLongs.length})</h3>
                        <PatternChartGrid trades={showLongs} onOpenChart={setChartOverlaySymbol} />
                      </div>
                    )}
                    {showShorts.length > 0 && (
                      <div>
                        <h3 className="text-red-400 text-sm font-bold uppercase tracking-wider mb-2">Short ({showShorts.length})</h3>
                        <PatternChartGrid trades={showShorts} onOpenChart={setChartOverlaySymbol} />
                      </div>
                    )}
                    {showHolds.length > 0 && (
                      <div>
                        <h3 className="text-amber-400 text-sm font-bold uppercase tracking-wider mb-2">Watchlist ({showHolds.length})</h3>
                        <PatternChartGrid trades={showHolds} onOpenChart={setChartOverlaySymbol} />
                      </div>
                    )}
                  </>
                )}

                {!isScanning && trades.length === 0 && !scanData && (
                  <div className="text-center py-8">
                    <div className="text-gray-500 text-sm">Select categories and click RUN SCAN</div>
                    <div className="text-gray-600 text-xs mt-1">Finds stocks with fresh stochastic extreme-zone crosses</div>
                  </div>
                )}

                {scanData && showLongs.length === 0 && showShorts.length === 0 && showHolds.length === 0 && (
                  <div className="text-center py-8">
                    <div className="text-gray-500 text-sm">
                      {tfFilters.length > 0 ? `No setups with ${tfFilters.map(t => t.toUpperCase()).join('/')} regime cross` : 'No setups found'}
                    </div>
                    <div className="text-gray-600 text-xs mt-1">
                      {noTrade.length} symbols had no stochastic extreme-zone cross
                    </div>
                  </div>
                )}
              </>
            )
          })()}
        </div>
      )}

      {/* ── Fixed terminal overlay — visible anywhere, bottom-right corner ── */}
      {(isScanning || scanLog.length > 0) && (
        <div className="fixed bottom-4 right-4 z-[200] w-[480px]">
          <ScanTerminal lines={scanLog} isScanning={isScanning} />
        </div>
      )}

      {/* ── KLineChart full-screen overlay — scan results retained underneath ── */}
      {chartOverlaySymbol && (
        <KLineChartOverlay
          symbol={chartOverlaySymbol}
          onClose={() => setChartOverlaySymbol(null)}
        />
      )}

      {/* ── Trade Detail Drawer (why / why-not reasoning) ── */}
      {detailTrade && (
        <TradeDetailDrawer
          trade={detailTrade}
          onClose={() => setDetailTrade(null)}
          onOpenChart={(sym) => { setDetailTrade(null); setChartOverlaySymbol(sym) }}
          account={account}
        />
      )}
    </div>
  )
}
