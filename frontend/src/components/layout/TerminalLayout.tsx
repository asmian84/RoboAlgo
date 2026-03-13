import { useEffect, useMemo, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { useCommandCenter } from '../../api/hooks'
import TopBar from './TopBar'
import SideNav from './SideNav'
import CommandPalette from '../terminal/CommandPalette'

interface Props {
  children: React.ReactNode
}

export default function TerminalLayout({ children }: Props) {
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const currentSymbol = searchParams.get('symbol') ?? undefined
  const [paletteOpen, setPaletteOpen] = useState(false)

  const { data } = useCommandCenter()
  const signals = data?.opportunity_map?.signals ?? []

  const fallbackSymbol = useMemo(
    () => [...signals].sort((a, b) => (b.confluence_score ?? 0) - (a.confluence_score ?? 0))[0]?.symbol,
    [signals],
  )

  const activeSymbol = currentSymbol ?? fallbackSymbol

  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement
      const isInput = target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.tagName === 'SELECT'
      if (isInput) return

      if (event.key === '/') {
        event.preventDefault()
        setPaletteOpen(true)
      }

      if (event.key.toLowerCase() === 'c') {
        event.preventDefault()
        navigate(`/chart${activeSymbol ? `?symbol=${activeSymbol}` : ''}`)
      }
    }

    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [activeSymbol, navigate])

  return (
    <>
      <CommandPalette isOpen={paletteOpen} onClose={() => setPaletteOpen(false)} />

      <div className="flex h-screen overflow-hidden bg-gray-950 text-gray-100">
        {/* Left nav */}
        <SideNav />

        {/* Right: topbar + content stacked */}
        <div className="flex flex-col flex-1 min-w-0">
          <div className="shrink-0 border-b border-gray-800">
            <TopBar safety={data?.market_safety} currentSymbol={activeSymbol} />
          </div>
          <main className="flex-1 min-h-0 overflow-auto bg-gray-950 p-2">{children}</main>
        </div>
      </div>
    </>
  )
}
