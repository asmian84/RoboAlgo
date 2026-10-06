import { useEffect, useMemo, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { useCommandCenter } from '../../api/hooks'
import SideNav from './SideNav'
import MacroTickerBar from './MacroTickerBar'
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
        <SideNav safety={data?.market_safety} currentSymbol={activeSymbol} />

        {/* ── Right column: ticker bar + page content ── */}
        <div className="flex-1 flex flex-col min-h-0 overflow-hidden">
          <MacroTickerBar />
          <main className="flex-1 min-h-0 overflow-y-auto overflow-x-hidden bg-gray-950 p-2">{children}</main>
        </div>
      </div>
    </>
  )
}
