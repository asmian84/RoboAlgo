import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import type { MarketSafetyData } from '../../types'
import { useAlerts, AlertModal } from '../alerts/AlertManager'
import { useLiveQuote, useMarketBreadth } from '../../api/hooks'
import { useAuth } from '../../context/AuthContext'

// ── Nav links ─────────────────────────────────────────────────────────────────

const NAV_ITEMS = [
  { path: '/command',   label: 'CMD',       title: 'Command Center'                         },
  { path: '/',          label: 'DASH',      title: 'Dashboard'                              },
  { path: '/chart',     label: 'CHART',     title: 'Charts'                                 },
  { path: '/scan',      label: 'SCAN',      title: 'Scan — Matrix & Watchlist'              },
  { path: '/signals',   label: 'SIG',       title: 'Signals — AI Recommendations'          },
  { path: '/analytics', label: 'ANALYTICS', title: 'Analytics — Backtest & Evolution'       },
  { path: '/paper',     label: 'PAPER',     title: 'Paper Trading'                          },
] as const

// ── Market Breadth pill — replaces the old "SAFE_MODE" safety pill ────────────

// Maps internal safety state (bullish-framed engine score) to market breadth direction.
// High score → broad market is bullish. Low score → broad market is bearish.
function MarketBreadthPill({ safety }: { safety?: MarketSafetyData }) {
  if (!safety) return null

  // score ≥ 65 = broad bullish, 40-64 = mixed/neutral, < 40 = bearish
  const score = safety.safety_score
  const breadth =
    score >= 65 ? { label: 'BULLISH',  color: '#22c55e', icon: '↑', bg: '18' } :
    score >= 40 ? { label: 'MIXED',    color: '#eab308', icon: '→', bg: '18' } :
                  { label: 'BEARISH',  color: '#ef4444', icon: '↓', bg: '18' }

  return (
    <div
      className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs"
      style={{
        color:           breadth.color,
        backgroundColor: breadth.color + breadth.bg,
        border:          `1px solid ${breadth.color}35`,
      }}
    >
      <span className="text-[9px] text-gray-500 font-normal tracking-wide uppercase">Mkt Breadth</span>
      <span className="font-bold">{breadth.icon} {breadth.label}</span>
      <span className="font-mono text-[10px] opacity-60">{score.toFixed(0)}</span>
    </div>
  )
}

// ── TopBar ────────────────────────────────────────────────────────────────────

interface Props {
  safety?: MarketSafetyData
  currentSymbol?: string
}

export default function TopBar({ safety, currentSymbol }: Props) {
  const navigate  = useNavigate()
  const searchRef = useRef<HTMLInputElement>(null)
  const [query, setQuery] = useState('')
  const [time, setTime]   = useState(() => new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }))
  const [alertOpen, setAlertOpen] = useState(false)
  const { fired } = useAlerts()
  const unreadCount = fired.filter(f => !f.dismissed).length
  const { data: breadth } = useMarketBreadth()
  const { data: vixQuote } = useLiveQuote('^VIX')
  const { user, signOut } = useAuth()

  // Clock tick
  useEffect(() => {
    const id = setInterval(() => {
      setTime(new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }))
    }, 1000)
    return () => clearInterval(id)
  }, [])

  // Global hotkey: / to focus search
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement
      if (e.key === '/' && target.tagName !== 'INPUT' && target.tagName !== 'TEXTAREA' && target.tagName !== 'SELECT') {
        e.preventDefault()
        searchRef.current?.focus()
        searchRef.current?.select()
      }
      if (e.key === 'Escape') {
        searchRef.current?.blur()
        setQuery('')
      }
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
  }

  return (
    <>
    <div className="flex items-center gap-2 px-3 h-12 bg-gray-950 border-b border-gray-800 select-none overflow-hidden">
      {/* Brand */}
      <div className="flex items-center gap-2 mr-2 shrink-0">
        <span className="text-xs font-bold text-emerald-400 tracking-widest">ROBOALGO</span>
        <span className="text-[10px] text-gray-600">v3</span>
      </div>

      {/* Nav items */}
      <nav className="flex items-center gap-0.5">
        {NAV_ITEMS.map(({ path, label, title }) => (
          <button
            key={path}
            title={title}
            onClick={() => navigate(path)}
            className="px-2.5 py-1 rounded text-[11px] font-bold text-gray-500 hover:text-gray-200 hover:bg-gray-800 transition-colors"
          >
            {label}
          </button>
        ))}
      </nav>

      {/* Divider */}
      <div className="w-px h-4 bg-gray-800 mx-1 shrink-0" />

      {/* Symbol search */}
      <form onSubmit={handleSearch} className="flex items-center">
        <div className="relative">
          <input
            ref={searchRef}
            value={query}
            onChange={e => setQuery(e.target.value.toUpperCase())}
            placeholder="/ Symbol…"
            className="w-32 bg-gray-800 border border-gray-700 rounded px-2.5 py-1 text-xs font-mono text-gray-200 placeholder-gray-600 focus:outline-none focus:border-emerald-600 focus:w-40 transition-all"
          />
        </div>
      </form>

      {/* Current symbol */}
      {currentSymbol && (
        <span className="text-xs font-mono font-bold text-emerald-400 shrink-0">{currentSymbol}</span>
      )}

      {/* Spacer */}
      <div className="flex-1" />

      {/* Alert bell */}
      <button
        onClick={() => setAlertOpen(true)}
        className="relative flex items-center justify-center w-7 h-7 rounded-lg bg-gray-800 hover:bg-gray-700 border border-gray-700 transition-colors"
        title="Alert Manager"
      >
        <span className="text-sm">🔔</span>
        {unreadCount > 0 && (
          <span className="absolute -top-1 -right-1 w-4 h-4 rounded-full bg-red-500 text-white text-[9px] font-bold flex items-center justify-center">
            {unreadCount > 9 ? '9+' : unreadCount}
          </span>
        )}
      </button>

      {/* Market Breadth indicator */}
      <MarketBreadthPill safety={safety} />

      {/* VIX badge */}
      {(breadth?.vix ?? vixQuote?.price) != null && (() => {
        const vix = breadth?.vix ?? vixQuote?.price ?? 0
        const chg = breadth?.vix_change ?? vixQuote?.change
        const dir = breadth?.vix_direction ?? (vix >= 30 ? 'extreme_fear' : vix >= 20 ? 'fear' : 'normal')
        const color = dir === 'extreme_fear' ? '#ef4444' : dir === 'fear' ? '#f97316' : dir === 'complacency' ? '#22c55e' : '#9ca3af'
        return (
          <div className="flex items-center gap-1 px-2 py-1 rounded text-xs"
            style={{ color, backgroundColor: color + '15', border: `1px solid ${color}30` }}>
            <span className="text-[9px] text-gray-600">VIX</span>
            <span className="font-mono font-bold">{vix.toFixed(1)}</span>
            {chg != null && <span className="text-[10px] opacity-60">{chg >= 0 ? '+' : ''}{chg.toFixed(1)}</span>}
          </div>
        )
      })()}

      {/* Fear / Greed badge */}
      {breadth?.fear_greed != null && (() => {
        const fg = breadth.fear_greed
        const label = breadth.fear_greed_label
        const color = fg < 20 ? '#ef4444' : fg < 40 ? '#f97316' : fg < 60 ? '#9ca3af' : fg < 80 ? '#22c55e' : '#10b981'
        return (
          <div className="items-center gap-1 px-2 py-1 rounded text-xs hidden 2xl:flex"
            style={{ color, backgroundColor: color + '15', border: `1px solid ${color}30` }}>
            <span className="text-[9px] text-gray-600">F/G</span>
            <span className="font-mono font-bold">{fg}</span>
            <span className="text-[9px] opacity-70">{label}</span>
          </div>
        )
      })()}

      {/* Keyboard hint */}
      <span className="text-[10px] text-gray-700 hidden 2xl:block shrink-0">
        / search · Esc clear
      </span>

      {/* Clock */}
      <span className="text-xs font-mono text-gray-500 shrink-0">{time}</span>

      {/* User avatar + sign out */}
      {user && (
        <div className="flex items-center gap-2 ml-1 pl-2 border-l border-gray-800 shrink-0">
          {/* Avatar initials */}
          <div
            className="w-7 h-7 rounded-full bg-emerald-700 flex items-center justify-center text-[11px] font-bold text-white cursor-default select-none"
            title={user.email ?? 'Signed in'}
          >
            {(user.email ?? 'U')[0].toUpperCase()}
          </div>
          {/* Email (hidden on small screens) */}
          <span className="text-[10px] text-gray-500 hidden 2xl:block max-w-[120px] truncate" title={user.email}>
            {user.email}
          </span>
          {/* Sign out */}
          <button
            onClick={signOut}
            title="Sign out"
            className="text-[10px] text-gray-600 hover:text-red-400 transition-colors px-1.5 py-1 rounded hover:bg-gray-800"
          >
            ⏏
          </button>
        </div>
      )}
    </div>

    <AlertModal isOpen={alertOpen} onClose={() => setAlertOpen(false)} />
    </>
  )
}
