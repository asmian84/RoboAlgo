import { useNavigate, useLocation } from 'react-router-dom'

const NAV_ITEMS = [
  { path: '/command',   label: 'CMD',       icon: '⌘' },
  { path: '/',          label: 'DASH',      icon: '▦' },
  { path: '/chart',     label: 'CHART',     icon: '↗' },
  { path: '/signals',   label: 'SIGNALS',   icon: '◎' },
  { path: '/analytics', label: 'ANALYTICS', icon: '⚡' },
  { path: '/paper',     label: 'PAPER',     icon: '□' },
] as const

export default function SideNav() {
  const navigate  = useNavigate()
  const location  = useLocation()

  function isActive(path: string) {
    if (path === '/') return location.pathname === '/' || location.pathname === '/intel'
    return location.pathname.startsWith(path)
  }

  return (
    <nav className="flex flex-col items-center gap-0.5 py-2 bg-gray-950 border-r border-gray-800 w-14 shrink-0">
      {/* Brand dot */}
      <div className="w-2 h-2 rounded-full bg-emerald-500 mb-3 shrink-0" />

      {NAV_ITEMS.map(({ path, label, icon }) => {
        const active = isActive(path)
        return (
          <button
            key={path}
            onClick={() => navigate(path)}
            title={label}
            className={`flex flex-col items-center gap-0.5 w-12 py-2.5 rounded-lg transition-all group ${
              active
                ? 'bg-emerald-700/30 text-emerald-400'
                : 'text-gray-600 hover:text-gray-300 hover:bg-gray-800/60'
            }`}
          >
            <span className="text-base leading-none">{icon}</span>
            <span className="text-[8px] font-bold tracking-wider leading-none">
              {label.length > 5 ? label.slice(0, 5) : label}
            </span>
          </button>
        )
      })}
    </nav>
  )
}
