import { useState, useCallback } from 'react'
import axios from 'axios'

const API = '/api/tv'

// ── Types ────────────────────────────────────────────────────────────────────
interface Summary { recommendation: string; buy: number; sell: number; neutral: number }
interface Analysis {
  symbol: string; exchange: string; interval: string
  summary: Summary
  oscillators: Summary & { details: Record<string, string> }
  moving_averages: Summary & { details: Record<string, string> }
  indicators: Record<string, number | null>
}
interface MTFResult {
  symbol: string
  timeframes: Record<string, Summary>
  alignment: string
  buy_count: number
  sell_count: number
}
interface NewsItem { source: string; title: string; link: string; published: string; summary: string }

type Tab = 'analysis' | 'mtf' | 'screener' | 'news'

const TABS: { id: Tab; label: string; icon: string }[] = [
  { id: 'analysis', label: 'Analysis', icon: 'T' },
  { id: 'mtf',      label: 'MTF',      icon: 'M' },
  { id: 'screener', label: 'Screener',  icon: 'S' },
  { id: 'news',     label: 'News',      icon: 'N' },
]

const INTERVALS = ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '1d', '1w', '1M']

// ── Color helpers ────────────────────────────────────────────────────────────
function recColor(r: string) {
  if (!r) return '#9ca3af'
  const u = r.toUpperCase()
  if (u.includes('STRONG_BUY') || u.includes('STRONG BUY')) return '#10b981'
  if (u.includes('BUY')) return '#22c55e'
  if (u.includes('STRONG_SELL') || u.includes('STRONG SELL')) return '#dc2626'
  if (u.includes('SELL')) return '#ef4444'
  return '#eab308'
}

function recBg(r: string) {
  const c = recColor(r)
  return c + '18'
}

// ── Gauge component ──────────────────────────────────────────────────────────
function Gauge({ label, data }: { label: string; data: Summary }) {
  const total = data.buy + data.sell + data.neutral || 1
  const buyPct = (data.buy / total) * 100
  const sellPct = (data.sell / total) * 100
  return (
    <div className="rounded-lg border border-gray-800 bg-gray-900/60 p-3">
      <div className="flex items-center justify-between mb-2">
        <span className="text-[10px] text-gray-500 font-bold uppercase tracking-wider">{label}</span>
        <span className="text-xs font-bold px-2 py-0.5 rounded" style={{ color: recColor(data.recommendation), backgroundColor: recBg(data.recommendation) }}>
          {data.recommendation}
        </span>
      </div>
      <div className="flex h-2 rounded-full overflow-hidden bg-gray-800">
        <div className="bg-emerald-500 transition-all" style={{ width: `${buyPct}%` }} />
        <div className="bg-gray-600 transition-all" style={{ width: `${100 - buyPct - sellPct}%` }} />
        <div className="bg-red-500 transition-all" style={{ width: `${sellPct}%` }} />
      </div>
      <div className="flex justify-between mt-1.5 text-[10px] font-mono">
        <span className="text-emerald-400">{data.buy} Buy</span>
        <span className="text-gray-500">{data.neutral} Neutral</span>
        <span className="text-red-400">{data.sell} Sell</span>
      </div>
    </div>
  )
}

// ── Analysis Tab ─────────────────────────────────────────────────────────────
function AnalysisTab() {
  const [symbol, setSymbol]     = useState('SPY')
  const [exchange, setExchange] = useState('NASDAQ')
  const [interval, setInterval] = useState('1d')
  const [data, setData]         = useState<Analysis | null>(null)
  const [loading, setLoading]   = useState(false)
  const [error, setError]       = useState('')

  const fetch_ = useCallback(async () => {
    setLoading(true); setError('')
    try {
      const r = await axios.get<Analysis>(`${API}/analysis/${symbol}`, { params: { exchange, interval } })
      setData(r.data)
    } catch (e: any) { setError(e?.response?.data?.detail || e.message) }
    setLoading(false)
  }, [symbol, exchange, interval])

  const ind = data?.indicators

  return (
    <div className="space-y-4">
      {/* Controls */}
      <div className="flex gap-2 items-end flex-wrap">
        <div>
          <label className="text-[9px] text-gray-600 uppercase block mb-1">Symbol</label>
          <input value={symbol} onChange={e => setSymbol(e.target.value.toUpperCase())}
            className="bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-xs font-mono text-gray-200 w-24 focus:outline-none focus:border-emerald-600" />
        </div>
        <div>
          <label className="text-[9px] text-gray-600 uppercase block mb-1">Exchange</label>
          <select value={exchange} onChange={e => setExchange(e.target.value)}
            className="bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-xs text-gray-200 focus:outline-none focus:border-emerald-600">
            {['NASDAQ', 'NYSE', 'AMEX', 'OTC'].map(x => <option key={x}>{x}</option>)}
          </select>
        </div>
        <div>
          <label className="text-[9px] text-gray-600 uppercase block mb-1">Interval</label>
          <select value={interval} onChange={e => setInterval(e.target.value)}
            className="bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-xs text-gray-200 focus:outline-none focus:border-emerald-600">
            {INTERVALS.map(x => <option key={x}>{x}</option>)}
          </select>
        </div>
        <button onClick={fetch_} disabled={loading}
          className="px-4 py-1.5 rounded bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-bold disabled:opacity-50 transition-colors">
          {loading ? 'Loading...' : 'Analyze'}
        </button>
      </div>

      {error && <div className="text-red-400 text-xs bg-red-900/20 border border-red-800 rounded p-2">{error}</div>}

      {data && (
        <div className="space-y-4">
          {/* Summary gauges */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <Gauge label="Overall" data={data.summary} />
            <Gauge label="Oscillators" data={data.oscillators} />
            <Gauge label="Moving Averages" data={data.moving_averages} />
          </div>

          {/* Key indicators */}
          {ind && (
            <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-6 gap-2">
              {[
                { label: 'RSI', value: ind.rsi, warn: ind.rsi ? ind.rsi > 70 || ind.rsi < 30 : false },
                { label: 'CCI', value: ind.cci },
                { label: 'ADX', value: ind.adx },
                { label: 'MACD', value: ind.macd },
                { label: 'ATR', value: ind.atr },
                { label: 'Stoch K', value: ind.stoch_k },
                { label: 'EMA 20', value: ind.ema20 },
                { label: 'EMA 50', value: ind.ema50 },
                { label: 'EMA 200', value: ind.ema200 },
                { label: 'SMA 20', value: ind.sma20 },
                { label: 'SMA 50', value: ind.sma50 },
                { label: 'SMA 200', value: ind.sma200 },
                { label: 'BB Upper', value: ind.bbands_upper },
                { label: 'BB Lower', value: ind.bbands_lower },
                { label: 'Pivot', value: ind.pivot_p },
                { label: 'Close', value: ind.close },
                { label: 'Volume', value: ind.volume },
              ].filter(x => x.value != null).map(({ label, value, warn }) => (
                <div key={label} className={`rounded border p-2 ${warn ? 'border-amber-700/50 bg-amber-900/10' : 'border-gray-800 bg-gray-900/40'}`}>
                  <div className="text-[9px] text-gray-500 uppercase">{label}</div>
                  <div className="text-xs font-mono font-bold text-gray-200">
                    {typeof value === 'number' ? (Math.abs(value) > 1000 ? value.toLocaleString('en-US', { maximumFractionDigits: 0 }) : value.toFixed(2)) : value}
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Oscillator details */}
          {data.oscillators.details && Object.keys(data.oscillators.details).length > 0 && (
            <div className="rounded-lg border border-gray-800 bg-gray-900/40 p-3">
              <div className="text-[10px] text-gray-500 font-bold uppercase mb-2">Oscillator Signals</div>
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-1">
                {Object.entries(data.oscillators.details).map(([name, signal]) => (
                  <div key={name} className="flex items-center justify-between px-2 py-1 rounded bg-gray-800/50">
                    <span className="text-[10px] text-gray-400 truncate mr-2">{name}</span>
                    <span className="text-[10px] font-bold" style={{ color: recColor(signal) }}>{signal}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* MA details */}
          {data.moving_averages.details && Object.keys(data.moving_averages.details).length > 0 && (
            <div className="rounded-lg border border-gray-800 bg-gray-900/40 p-3">
              <div className="text-[10px] text-gray-500 font-bold uppercase mb-2">Moving Average Signals</div>
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-1">
                {Object.entries(data.moving_averages.details).map(([name, signal]) => (
                  <div key={name} className="flex items-center justify-between px-2 py-1 rounded bg-gray-800/50">
                    <span className="text-[10px] text-gray-400 truncate mr-2">{name}</span>
                    <span className="text-[10px] font-bold" style={{ color: recColor(signal) }}>{signal}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// ── MTF Tab ──────────────────────────────────────────────────────────────────
function MTFTab() {
  const [symbol, setSymbol]     = useState('SPY')
  const [exchange, setExchange] = useState('NASDAQ')
  const [data, setData]         = useState<MTFResult | null>(null)
  const [loading, setLoading]   = useState(false)
  const [error, setError]       = useState('')

  const fetch_ = useCallback(async () => {
    setLoading(true); setError('')
    try {
      const r = await axios.get<MTFResult>(`${API}/mtf/${symbol}`, { params: { exchange } })
      setData(r.data)
    } catch (e: any) { setError(e?.response?.data?.detail || e.message) }
    setLoading(false)
  }, [symbol, exchange])

  const alignColor = data?.alignment === 'BULLISH' ? '#22c55e' : data?.alignment === 'BEARISH' ? '#ef4444' : '#eab308'

  return (
    <div className="space-y-4">
      <div className="flex gap-2 items-end">
        <div>
          <label className="text-[9px] text-gray-600 uppercase block mb-1">Symbol</label>
          <input value={symbol} onChange={e => setSymbol(e.target.value.toUpperCase())}
            className="bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-xs font-mono text-gray-200 w-24 focus:outline-none focus:border-emerald-600" />
        </div>
        <div>
          <label className="text-[9px] text-gray-600 uppercase block mb-1">Exchange</label>
          <select value={exchange} onChange={e => setExchange(e.target.value)}
            className="bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-xs text-gray-200 focus:outline-none focus:border-emerald-600">
            {['NASDAQ', 'NYSE', 'AMEX'].map(x => <option key={x}>{x}</option>)}
          </select>
        </div>
        <button onClick={fetch_} disabled={loading}
          className="px-4 py-1.5 rounded bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-bold disabled:opacity-50 transition-colors">
          {loading ? 'Scanning...' : 'Scan MTF'}
        </button>
      </div>

      {error && <div className="text-red-400 text-xs bg-red-900/20 border border-red-800 rounded p-2">{error}</div>}

      {data && (
        <div className="space-y-4">
          {/* Alignment badge */}
          <div className="flex items-center gap-3 p-3 rounded-lg border" style={{ borderColor: alignColor + '40', backgroundColor: alignColor + '10' }}>
            <span className="text-lg font-black" style={{ color: alignColor }}>{data.alignment}</span>
            <span className="text-xs text-gray-400">
              {data.buy_count}/5 bullish timeframes | {data.sell_count}/5 bearish
            </span>
          </div>

          {/* Timeframe grid */}
          <div className="grid grid-cols-5 gap-2">
            {Object.entries(data.timeframes).map(([tf, s]) => (
              <div key={tf} className="rounded-lg border border-gray-800 bg-gray-900/60 p-3 text-center">
                <div className="text-[10px] text-gray-500 font-bold uppercase mb-2">{tf}</div>
                <div className="text-sm font-bold px-2 py-1 rounded" style={{ color: recColor(s.recommendation), backgroundColor: recBg(s.recommendation) }}>
                  {s.recommendation}
                </div>
                <div className="flex justify-between mt-2 text-[9px] font-mono">
                  <span className="text-emerald-500">{s.buy}</span>
                  <span className="text-gray-600">{s.neutral}</span>
                  <span className="text-red-500">{s.sell}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

// ── Screener Tab ─────────────────────────────────────────────────────────────
function ScreenerTab() {
  const [data, setData]       = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState('')
  const [mode, setMode]       = useState<'movers' | 'squeeze'>('movers')

  const fetch_ = useCallback(async () => {
    setLoading(true); setError('')
    try {
      const endpoint = mode === 'movers' ? '/screener' : '/squeeze'
      const r = await axios.get(`${API}${endpoint}`)
      setData(r.data)
    } catch (e: any) { setError(e?.response?.data?.detail || e.message) }
    setLoading(false)
  }, [mode])

  const items = mode === 'movers' ? data?.movers : data?.squeeze

  return (
    <div className="space-y-4">
      <div className="flex gap-2 items-center">
        <button onClick={() => setMode('movers')}
          className={`px-3 py-1.5 rounded text-xs font-bold transition-colors ${mode === 'movers' ? 'bg-emerald-600 text-white' : 'bg-gray-800 text-gray-400 hover:text-gray-200'}`}>
          Top Movers
        </button>
        <button onClick={() => setMode('squeeze')}
          className={`px-3 py-1.5 rounded text-xs font-bold transition-colors ${mode === 'squeeze' ? 'bg-emerald-600 text-white' : 'bg-gray-800 text-gray-400 hover:text-gray-200'}`}>
          BB Squeeze
        </button>
        <button onClick={fetch_} disabled={loading}
          className="px-4 py-1.5 rounded bg-blue-600 hover:bg-blue-500 text-white text-xs font-bold disabled:opacity-50 transition-colors ml-auto">
          {loading ? 'Scanning...' : 'Scan'}
        </button>
      </div>

      {error && <div className="text-red-400 text-xs bg-red-900/20 border border-red-800 rounded p-2">{error}</div>}

      {items && items.length > 0 && (
        <div className="rounded-lg border border-gray-800 overflow-hidden">
          <table className="w-full text-xs">
            <thead>
              <tr className="bg-gray-900 text-gray-500 text-[10px] uppercase">
                <th className="text-left px-3 py-2">Symbol</th>
                <th className="text-right px-3 py-2">Close</th>
                <th className="text-right px-3 py-2">{mode === 'squeeze' ? 'BB Width%' : 'Change%'}</th>
                <th className="text-right px-3 py-2">RSI</th>
                <th className="text-right px-3 py-2">Volume</th>
              </tr>
            </thead>
            <tbody>
              {items.map((r: any, i: number) => {
                const chg = mode === 'squeeze' ? r.bb_width_pct : r.change_pct
                const chgUp = chg >= 0
                return (
                  <tr key={i} className="border-t border-gray-800/50 hover:bg-gray-800/30">
                    <td className="px-3 py-1.5 font-mono font-bold text-gray-200">{r.symbol}</td>
                    <td className="px-3 py-1.5 text-right font-mono text-gray-300">{r.close?.toFixed(2)}</td>
                    <td className={`px-3 py-1.5 text-right font-mono font-bold ${chgUp ? 'text-emerald-400' : 'text-red-400'}`}>
                      {chgUp ? '+' : ''}{chg?.toFixed(2)}%
                    </td>
                    <td className="px-3 py-1.5 text-right font-mono text-gray-400">{r.rsi?.toFixed(1)}</td>
                    <td className="px-3 py-1.5 text-right font-mono text-gray-500">
                      {r.volume ? (r.volume > 1e6 ? (r.volume / 1e6).toFixed(1) + 'M' : (r.volume / 1e3).toFixed(0) + 'K') : '-'}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

// ── News Tab ─────────────────────────────────────────────────────────────────
function NewsTab() {
  const [articles, setArticles] = useState<NewsItem[]>([])
  const [loading, setLoading]   = useState(false)
  const [error, setError]       = useState('')

  const fetch_ = useCallback(async () => {
    setLoading(true); setError('')
    try {
      const r = await axios.get<{ articles: NewsItem[] }>(`${API}/news`)
      setArticles(r.data.articles)
    } catch (e: any) { setError(e?.response?.data?.detail || e.message) }
    setLoading(false)
  }, [])

  return (
    <div className="space-y-4">
      <button onClick={fetch_} disabled={loading}
        className="px-4 py-1.5 rounded bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-bold disabled:opacity-50 transition-colors">
        {loading ? 'Loading...' : 'Fetch News'}
      </button>

      {error && <div className="text-red-400 text-xs bg-red-900/20 border border-red-800 rounded p-2">{error}</div>}

      <div className="space-y-2">
        {articles.map((a, i) => (
          <a key={i} href={a.link} target="_blank" rel="noopener noreferrer"
            className="block rounded-lg border border-gray-800 bg-gray-900/40 p-3 hover:border-gray-700 hover:bg-gray-800/40 transition-colors">
            <div className="flex items-center gap-2 mb-1">
              <span className="text-[9px] font-bold text-emerald-500 uppercase">{a.source}</span>
              {a.published && <span className="text-[9px] text-gray-600">{new Date(a.published).toLocaleString()}</span>}
            </div>
            <div className="text-xs font-bold text-gray-200 leading-snug">{a.title}</div>
            {a.summary && <div className="text-[10px] text-gray-500 mt-1 line-clamp-2">{a.summary}</div>}
          </a>
        ))}
      </div>
    </div>
  )
}

// ── Main Page ────────────────────────────────────────────────────────────────
export default function TradingViewPage() {
  const [tab, setTab] = useState<Tab>('analysis')

  return (
    <div className="h-full flex flex-col overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-gray-800 shrink-0">
        <div className="flex items-center gap-3">
          <span className="text-emerald-400 font-black text-sm">TV</span>
          <span className="text-gray-400 text-xs">TradingView Analysis</span>
          <span className="text-[9px] text-gray-600 bg-gray-800 px-2 py-0.5 rounded">EVALUATION</span>
        </div>
        <div className="flex gap-1">
          {TABS.map(t => (
            <button key={t.id} onClick={() => setTab(t.id)}
              className={`px-3 py-1 rounded text-[11px] font-bold transition-colors ${
                tab === t.id ? 'bg-emerald-700/30 text-emerald-300 border border-emerald-700/40' : 'text-gray-500 hover:text-gray-300 border border-transparent'
              }`}>
              <span className="mr-1 opacity-60">{t.icon}</span>{t.label}
            </button>
          ))}
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4">
        {tab === 'analysis' && <AnalysisTab />}
        {tab === 'mtf'      && <MTFTab />}
        {tab === 'screener' && <ScreenerTab />}
        {tab === 'news'     && <NewsTab />}
      </div>
    </div>
  )
}
