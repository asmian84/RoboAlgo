import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useInstruments } from '../../api/hooks'

interface Props {
  isOpen: boolean
  onClose: () => void
}

type CommandItem = {
  id: string
  label: string
  kind: 'symbol' | 'page' | 'feature'
  path: string
}

const PAGES: CommandItem[] = [
  { id: 'page-chart', label: 'Chart', kind: 'page', path: '/chart' },
  { id: 'page-scan', label: 'Scan', kind: 'page', path: '/scan' },
  { id: 'page-matrix', label: 'Opportunity Matrix', kind: 'page', path: '/matrix' },
  { id: 'page-watchlist', label: 'Watchlist', kind: 'page', path: '/watchlist' },
  { id: 'page-signals', label: 'Signals', kind: 'page', path: '/signals' },
  { id: 'page-analytics', label: 'Analytics', kind: 'page', path: '/analytics' },
]

const FEATURES: CommandItem[] = [
  { id: 'feature-trade-coach', label: 'Trade Coach Panel', kind: 'feature', path: '/chart' },
  { id: 'feature-signal-panel', label: 'Signal Panel', kind: 'feature', path: '/chart' },
  { id: 'feature-watchlist-panel', label: 'Watchlist Panel', kind: 'feature', path: '/chart' },
]

export default function CommandPalette({ isOpen, onClose }: Props) {
  const navigate = useNavigate()
  const { data: instruments = [] } = useInstruments()
  const [query, setQuery] = useState('')
  const [selected, setSelected] = useState(0)

  const symbolItems: CommandItem[] = useMemo(
    () => instruments.slice(0, 200).map(inst => ({
      id: `symbol-${inst.symbol}`,
      label: inst.symbol,
      kind: 'symbol',
      path: `/chart?symbol=${inst.symbol}`,
    })),
    [instruments],
  )

  const allItems = useMemo(() => [...symbolItems, ...PAGES, ...FEATURES], [symbolItems])

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase()
    if (!q) return allItems.slice(0, 14)
    return allItems.filter(item => item.label.toLowerCase().includes(q)).slice(0, 14)
  }, [allItems, query])

  useEffect(() => {
    if (!isOpen) return
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault()
        onClose()
        return
      }
      if (event.key === 'ArrowDown') {
        event.preventDefault()
        setSelected(current => Math.min(current + 1, Math.max(filtered.length - 1, 0)))
      }
      if (event.key === 'ArrowUp') {
        event.preventDefault()
        setSelected(current => Math.max(current - 1, 0))
      }
      if (event.key === 'Enter' && filtered[selected]) {
        event.preventDefault()
        navigate(filtered[selected].path)
        onClose()
      }
    }

    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [filtered, isOpen, navigate, onClose, selected])

  useEffect(() => {
    if (isOpen) {
      setQuery('')
      setSelected(0)
    }
  }, [isOpen])

  if (!isOpen) return null

  return (
    <>
      <div className="fixed inset-0 z-40 bg-black/70" onClick={onClose} />
      <div className="fixed left-1/2 top-20 z-50 w-[min(720px,92vw)] -translate-x-1/2 rounded border border-gray-700 bg-gray-900 shadow-2xl">
        <div className="border-b border-gray-800 px-3 py-2">
          <input
            autoFocus
            value={query}
            onChange={event => {
              setQuery(event.target.value)
              setSelected(0)
            }}
            placeholder="Search symbols, pages, features..."
            className="w-full bg-transparent text-sm text-gray-100 placeholder-gray-500 focus:outline-none"
          />
        </div>

        <div className="max-h-96 overflow-y-auto p-1">
          {filtered.map((item, index) => (
            <button
              key={item.id}
              onClick={() => {
                navigate(item.path)
                onClose()
              }}
              className={`flex w-full items-center justify-between rounded px-2 py-2 text-left text-sm transition-colors ${
                selected === index ? 'bg-gray-800 text-white' : 'text-gray-300 hover:bg-gray-800/70'
              }`}
            >
              <span>{item.label}</span>
              <span className="text-[10px] uppercase tracking-wide text-gray-500">{item.kind}</span>
            </button>
          ))}

          {filtered.length === 0 && <p className="px-2 py-6 text-center text-sm text-gray-500">No results.</p>}
        </div>
      </div>
    </>
  )
}
