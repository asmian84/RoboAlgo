import { useState } from 'react'
import { NavLink } from 'react-router-dom'
import clsx from 'clsx'

const links = [
  { to: '/intel',    label: 'Intelligence',  icon: '◉' },
  { to: '/chart',    label: 'Charts',        icon: '⊟' },
  { to: '/signals',  label: 'Signals',       icon: '★' },
  { to: '/paper',    label: 'Paper Trade',   icon: '◈' },
  { to: '/analytics',label: 'Analytics',     icon: '⊛' },
  { to: '/command',  label: 'Command',       icon: '⌘' },
]

function NavItems({ onNav }: { onNav?: () => void }) {
  return (
    <nav className="flex-1 py-4 space-y-1">
      {links.map(({ to, label, icon }) => (
        <NavLink
          key={to}
          to={to}
          end={to === '/'}
          onClick={onNav}
          className={({ isActive }) =>
            clsx(
              'flex items-center gap-3 px-4 py-2.5 text-sm transition-colors',
              isActive
                ? 'bg-gray-800 text-emerald-400 border-r-2 border-emerald-400'
                : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800/50'
            )
          }
        >
          <span className="text-base">{icon}</span>
          {label}
        </NavLink>
      ))}
    </nav>
  )
}

export default function Sidebar() {
  const [mobileOpen, setMobileOpen] = useState(false)

  return (
    <>
      {/* ── Desktop sidebar ── */}
      <aside className="hidden md:flex w-56 bg-gray-900 border-r border-gray-800 flex-col shrink-0">
        <div className="p-4 border-b border-gray-800">
          <h1 className="text-lg font-bold text-emerald-400 tracking-wide">RoboAlgo</h1>
          <p className="text-xs text-gray-500 mt-0.5">Cycle Trading Intelligence</p>
        </div>
        <NavItems />
        <div className="p-4 border-t border-gray-800">
          <p className="text-xs text-gray-600">3-Tier Trading System</p>
        </div>
      </aside>

      {/* ── Mobile top bar ── */}
      <div className="md:hidden fixed top-0 left-0 right-0 z-50 bg-gray-900 border-b border-gray-800 flex items-center justify-between px-4 py-3">
        <div>
          <span className="text-base font-bold text-emerald-400 tracking-wide">RoboAlgo</span>
          <span className="text-xs text-gray-600 ml-2">Cycle Trading</span>
        </div>
        <button
          onClick={() => setMobileOpen(!mobileOpen)}
          className="text-gray-400 hover:text-white p-1"
          aria-label="Toggle menu"
        >
          {mobileOpen ? (
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          ) : (
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          )}
        </button>
      </div>

      {/* ── Mobile drawer ── */}
      {mobileOpen && (
        <div className="md:hidden fixed inset-0 z-40">
          {/* Backdrop */}
          <div
            className="absolute inset-0 bg-black/60"
            onClick={() => setMobileOpen(false)}
          />
          {/* Drawer */}
          <aside className="absolute top-0 left-0 h-full w-56 bg-gray-900 border-r border-gray-800 flex flex-col pt-14">
            <NavItems onNav={() => setMobileOpen(false)} />
            <div className="p-4 border-t border-gray-800">
              <p className="text-xs text-gray-600">3-Tier Trading System</p>
            </div>
          </aside>
        </div>
      )}
    </>
  )
}
