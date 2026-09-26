import { useEffect, useRef, useState, useCallback } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import axios from 'axios'
import type { MarketSafetyData } from '../../types'
import { useAlerts, AlertModal } from '../alerts/AlertManager'
import { useLiveQuote, useMarketBreadth } from '../../api/hooks'
import { useAuth } from '../../context/AuthContext'

const API = '/api'

// ── Macro ticker panel (cycling single-instrument view) ───────────────────────

interface MacroItem { symbol: string; name: string; price: number; change: number; pct: number }

function MacroTickerPanel() {
  const [items,   setItems]   = useState<MacroItem[]>([])
  const [idx,     setIdx]     = useState(0)
  const [visible, setVisible] = useState(true)
  const [flash,   setFlash]   = useState<'up' | 'down' | null>(null)
  const prevRef = useRef<Record<string, number>>({})
  const idxRef  = useRef(0)

  const fetchMacro = useCallback(async () => {
    try {
      const res  = await axios.get<{ items: MacroItem[] }>(`${API}/macro`)
      const next = res.data.items
      const cur  = next[idxRef.current]
      if (cur) {
        const prev = prevRef.current[cur.symbol]
        if (prev !== undefined && prev !== cur.price) {
          setFlash(cur.price > prev ? 'up' : 'down')
          setTimeout(() => setFlash(null), 800)
        }
        prevRef.current[cur.symbol] = cur.price
      }
      setItems(next)
    } catch { /* silent */ }
  }, [])

  useEffect(() => {
    fetchMacro()
    const id = setInterval(fetchMacro, 5000)
    return () => clearInterval(id)
  }, [fetchMacro])

  // Cycle every 5 s with fade
  useEffect(() => {
    const id = setInterval(() => {
      setVisible(false)
      setTimeout(() => {
        setIdx(i => {
          const next = (i + 1) % (items.length || 1)
          idxRef.current = next
          return next
        })
        setVisible(true)
      }, 250)
    }, 5000)
    return () => clearInterval(id)
  }, [items.length])

  if (!items.length) return (
    <div className="flex-1 flex items-center justify-center">
      <div className="w-3 h-3 border border-gray-700 border-t-gray-500 rounded-full animate-spin" />
    </div>
  )

  const it = items[idx % items.length]
  const up = it.pct >= 0
  const priceStr = it.price >= 10000
    ? it.price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
    : it.price >= 100 ? it.price.toFixed(2) : it.price.toFixed(4)

  return (
    <div className="flex-1 px-2 py-1 flex flex-col justify-center overflow-hidden">
      <div
        className="rounded-lg px-2.5 py-2.5"
        style={{
          opacity: visible ? 1 : 0,
          backgroundColor: flash === 'up'   ? 'rgba(34,197,94,0.14)'
                         : flash === 'down' ? 'rgba(239,68,68,0.12)'
                         : 'rgba(255,255,255,0.03)',
          border: `1px solid ${up ? 'rgba(34,197,94,0.15)' : 'rgba(239,68,68,0.15)'}`,
          transition: 'opacity 0.25s ease, background-color 0.5s ease',
        }}
      >
        <div className="text-[9px] text-gray-500 font-medium mb-1 truncate">{it.name}</div>
        <div className={`text-[15px] font-mono font-black leading-none ${up ? 'text-gray-100' : 'text-gray-200'}`}>
          {priceStr}
        </div>
        <div className="flex items-center justify-between mt-1">
          <span className={`text-[10px] font-bold ${up ? 'text-emerald-400' : 'text-red-400'}`}>
            {up ? '▲' : '▼'} {Math.abs(it.pct).toFixed(2)}%
          </span>
          <div className="flex gap-1">
            {items.map((_, i) => (
              <div key={i} className="rounded-full transition-all duration-300" style={{
                width: i === idx ? 10 : 4, height: 4,
                backgroundColor: i === idx ? (up ? '#22c55e' : '#ef4444') : 'rgba(255,255,255,0.12)',
              }} />
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

// ── Nav items ─────────────────────────────────────────────────────────────────

const NAV_ITEMS = [
  { path: '/command', label: 'CMD',     icon: '⌘' },
  { path: '/edge',    label: 'EDGE',    icon: '◆' },
  { path: '/chart',   label: 'CHART',   icon: '↗' },
  { path: '/signals', label: 'SIGNALS', icon: '◎' },
  { path: '/paper',   label: 'PAPER',   icon: '□' },
  { path: '/trades',  label: 'TRADES',  icon: '⚡' },
  { path: '/crypto',  label: 'CRYPTO',  icon: '₿' },
  { path: '/lab',     label: 'LAB',     icon: '⚗' },
  { path: '/auto',    label: 'AUTO',    icon: '◉' },
] as const

// ── Props ─────────────────────────────────────────────────────────────────────

interface Props {
  safety?:        MarketSafetyData
  currentSymbol?: string
}

// ── Component ─────────────────────────────────────────────────────────────────

export default function SideNav({ safety, currentSymbol: _currentSymbol }: Props) {
  const navigate   = useNavigate()
  const location   = useLocation()
  const searchRef  = useRef<HTMLInputElement>(null)
  const [query,     setQuery]     = useState('')
  const [time,      setTime]      = useState(() => new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }))
  const [alertOpen, setAlertOpen] = useState(false)
  const [dataSrc,   setDataSrc]   = useState<{ mtf: string; daily: string; tier: string } | null>(null)
  const [mobileOpen, setMobileOpen] = useState(false)
  const { fired }         = useAlerts()
  const unreadCount       = fired.filter(f => !f.dismissed).length
  const { data: breadth } = useMarketBreadth()
  const { data: vixQuote} = useLiveQuote('^VIX')
  const { user, signOut } = useAuth()

  // Clock
  useEffect(() => {
    const id = setInterval(() => setTime(new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })), 1000)
    return () => clearInterval(id)
  }, [])

  // Data source indicator
  useEffect(() => {
    const load = () =>
      fetch('/api/settings')
        .then(r => r.ok ? r.json() : null)
        .then(d => {
          if (!d) return
          setDataSrc({
            mtf:   d.data_sources?.intraday_mtf ?? 'yfinance',
            daily: d.data_sources?.daily_ohlcv  ?? 'alphavantage',
            tier:  d.alphavantage?.tier          ?? 'free',
          })
        })
        .catch(() => {})
    load()
    const id = setInterval(load, 60_000)
    return () => clearInterval(id)
  }, [])

  // / hotkey → focus search
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement
      if (e.key === '/' && t.tagName !== 'INPUT' && t.tagName !== 'TEXTAREA') {
        e.preventDefault(); searchRef.current?.focus(); searchRef.current?.select()
      }
      if (e.key === 'Escape') { searchRef.current?.blur(); setQuery('') }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [])

  function handleSearch(e: React.FormEvent) {
    e.preventDefault()
    if (!query.trim()) return
    navigate(`/chart?symbol=${query.trim().toUpperCase()}`)
    setQuery('')
    searchRef.current?.blur()
    setMobileOpen(false)
  }

  function isActive(path: string) {
    if (path === '/edge') return location.pathname === '/edge' || location.pathname === '/'
    return location.pathname.startsWith(path)
  }

  // Derived market data
  const vix    = breadth?.vix    ?? vixQuote?.price
  const vixChg = breadth?.vix_change ?? vixQuote?.change
  const vixDir = breadth?.vix_direction ?? (vix != null ? (vix >= 30 ? 'extreme_fear' : vix >= 20 ? 'fear' : 'normal') : 'normal')
  const vixColor = vixDir === 'extreme_fear' ? '#ef4444' : vixDir === 'fear' ? '#f97316' : vixDir === 'complacency' ? '#22c55e' : '#9ca3af'

  const bScore = safety?.safety_score
  const breadthInfo = bScore == null ? null :
    bScore >= 65 ? { label: 'BULL', color: '#22c55e', icon: '↑' } :
    bScore >= 40 ? { label: 'MIX',  color: '#eab308', icon: '→' } :
                   { label: 'BEAR', color: '#ef4444', icon: '↓' }

  const fg = breadth?.fear_greed
  const fgColor = fg == null ? '#9ca3af' : fg < 20 ? '#ef4444' : fg < 40 ? '#f97316' : fg < 60 ? '#9ca3af' : fg < 80 ? '#22c55e' : '#10b981'

  return (
    <>
    {/* Mobile hamburger button */}
    <button
      onClick={() => setMobileOpen(!mobileOpen)}
      className="fixed top-4 left-4 z-50 md:hidden p-2 rounded-lg bg-gray-800 border border-gray-700 text-gray-300 hover:bg-gray-700"
    >
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d={mobileOpen ? "M6 18L18 6M6 6l12 12" : "M4 6h16M4 12h16M4 18h16"} />
      </svg>
    </button>

    {/* Mobile overlay */}
    {mobileOpen && (
      <div
        className="fixed inset-0 z-30 bg-black/50 md:hidden"
        onClick={() => setMobileOpen(false)}
      />
    )}

    {/* Sidebar nav */}
    <nav className={`
      fixed md:relative
      top-0 left-0 h-full
      flex flex-col
      w-[168px] bg-gray-950 border-r border-gray-800 shrink-0 select-none overflow-hidden
      z-40 md:z-auto
      transition-transform duration-300 ease-out
      ${mobileOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'}
    `}>

      {/* ── Brand ── */}
      <div className="flex items-center border-b border-gray-800/60 h-10 shrink-0 px-3 gap-2">
        <div className="w-2 h-2 rounded-full bg-emerald-500 shrink-0" />
        <span className="text-[11px] font-black text-emerald-400 tracking-widest flex-1">ROBOALGO</span>
        <span className="text-[9px] text-gray-700">v3</span>
      </div>

      {/* ── Search ── */}
      <div className="px-2 pt-2 border-b border-gray-800/60">
        <form onSubmit={handleSearch}>
          <input
            ref={searchRef}
            value={query}
            onChange={e => setQuery(e.target.value.toUpperCase())}
            placeholder="/ Symbol…"
            className="w-full bg-gray-800 border border-gray-700 rounded px-2.5 py-1.5 text-xs font-mono text-gray-200 placeholder-gray-600 focus:outline-none focus:border-emerald-600 transition-colors"
          />
        </form>
        <div className="mt-2 pb-2 space-y-1">
          {breadthInfo && (
            <div className="flex items-center justify-between px-2 py-1 rounded" style={{ backgroundColor: breadthInfo.color + '12', border: `1px solid ${breadthInfo.color}30` }}>
              <span className="text-[9px] text-gray-600 uppercase tracking-wide">Breadth</span>
              <span className="text-[10px] font-bold" style={{ color: breadthInfo.color }}>
                {breadthInfo.icon} {breadthInfo.label} {bScore?.toFixed(0)}
              </span>
            </div>
          )}
          {vix != null && (
            <div className="flex items-center justify-between px-2 py-1 rounded" style={{ backgroundColor: vixColor + '12', border: `1px solid ${vixColor}25` }}>
              <span className="text-[9px] text-gray-600">VIX</span>
              <span className="text-[10px] font-mono font-bold" style={{ color: vixColor }}>
                {vix.toFixed(1)}{vixChg != null && <span className="opacity-60 ml-1">{vixChg >= 0 ? '+' : ''}{vixChg.toFixed(1)}</span>}
              </span>
            </div>
          )}
          {fg != null && (
            <div className="flex items-center justify-between px-2 py-1 rounded" style={{ backgroundColor: fgColor + '12', border: `1px solid ${fgColor}25` }}>
              <span className="text-[9px] text-gray-600">Fear/Greed</span>
              <span className="text-[10px] font-mono font-bold" style={{ color: fgColor }}>{fg}</span>
            </div>
          )}
        </div>
      </div>

      {/* ── Nav items ── */}
      <div className="flex flex-col gap-0.5 py-2 px-2">
        {NAV_ITEMS.map(({ path, label, icon }) => {
          const active = isActive(path)
          return (
            <button
              key={path}
              onClick={() => {
                navigate(path)
                setMobileOpen(false)
              }}
              className={`flex items-center w-full gap-2.5 px-2.5 py-2 rounded-lg text-left transition-all ${
                active ? 'bg-emerald-700/25 text-emerald-300' : 'text-gray-500 hover:text-gray-200 hover:bg-gray-800/70'
              }`}
            >
              <span className="text-sm shrink-0">{icon}</span>
              <span className="text-[11px] font-bold tracking-wide">{label}</span>
              {active && <div className="ml-auto w-1 h-1 rounded-full bg-emerald-400" />}
            </button>
          )
        })}
      </div>

      {/* ── Macro ticker panel ── */}
      <MacroTickerPanel />

      {/* ── Data source indicator ── */}
      {dataSrc && (
        <div
          className="mx-2 mb-1 px-2 py-1.5 rounded-lg border cursor-pointer"
          style={{ backgroundColor: '#0f172a', borderColor: '#1e293b' }}
          onClick={() => {
            navigate('/settings')
            setMobileOpen(false)
          }}
          title="Data sources — click to configure"
        >
          <div className="text-[8px] text-gray-600 uppercase tracking-widest mb-1">Data Providers</div>
          <div className="flex items-center justify-between">
            <span className="text-[9px] text-gray-500">MTF</span>
            <span className="text-[9px] font-bold" style={{ color: dataSrc.mtf === 'alphavantage' ? '#34d399' : '#60a5fa' }}>
              {dataSrc.mtf === 'alphavantage'
                ? `AlphaVantage · ${dataSrc.tier === 'premium' ? '75/m' : dataSrc.tier === 'professional' ? '600/m' : '5/m'}`
                : 'Yahoo Finance'}
            </span>
          </div>
          <div className="flex items-center justify-between mt-0.5">
            <span className="text-[9px] text-gray-500">Daily</span>
            <span className="text-[9px] font-bold" style={{ color: dataSrc.daily === 'alphavantage' ? '#34d399' : '#60a5fa' }}>
              {dataSrc.daily === 'alphavantage' ? 'AlphaVantage' : 'Yahoo Finance'}
            </span>
          </div>
          <div className="flex items-center justify-between mt-0.5">
            <span className="text-[9px] text-gray-500">Universe</span>
            <span className="text-[9px] font-bold" style={{ color: '#818cf8' }}>FinnHub</span>
          </div>
        </div>
      )}

      {/* ── Bottom: clock · alerts · settings · user ── */}
      <div className="py-2 px-2 border-t border-gray-800/60 shrink-0 space-y-2">
        <div className="text-center text-[11px] font-mono text-gray-500">{time}</div>

        <div className="flex items-center gap-2">
          {/* Alerts */}
          <button
            onClick={() => setAlertOpen(true)}
            className="relative flex items-center justify-center w-7 h-7 rounded-lg bg-gray-800 hover:bg-gray-700 border border-gray-700 transition-colors shrink-0"
            title="Alerts"
          >
            <span className="text-sm">🔔</span>
            {unreadCount > 0 && (
              <span className="absolute -top-1 -right-1 w-4 h-4 rounded-full bg-red-500 text-white text-[9px] font-bold flex items-center justify-center">
                {unreadCount > 9 ? '9+' : unreadCount}
              </span>
            )}
          </button>

          {/* Settings */}
          <button
            onClick={() => navigate('/settings')}
            className={`flex items-center justify-center w-7 h-7 rounded-lg border transition-colors shrink-0 ${
              isActive('/settings')
                ? 'bg-emerald-700/25 border-emerald-700/40 text-emerald-300'
                : 'bg-gray-800 hover:bg-gray-700 border-gray-700 text-gray-400 hover:text-gray-200'
            }`}
            title="Settings"
          >
            <span className="text-sm">⚙</span>
          </button>

          {/* User */}
          {user && (
            <div className="flex items-center gap-1.5 flex-1 min-w-0">
              <div className="w-6 h-6 rounded-full bg-emerald-700 flex items-center justify-center text-[10px] font-bold text-white shrink-0">
                {(user.email ?? 'U')[0].toUpperCase()}
              </div>
              <span className="text-[9px] text-gray-500 truncate flex-1 min-w-0">{user.email}</span>
              <button onClick={signOut} title="Sign out" className="text-gray-600 hover:text-red-400 transition-colors shrink-0">⏏</button>
            </div>
          )}
        </div>
      </div>
    </nav>

    <AlertModal isOpen={alertOpen} onClose={() => setAlertOpen(false)} />
    </>
  )
}
