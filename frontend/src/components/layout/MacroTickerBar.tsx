import { useCallback, useEffect, useRef, useState } from 'react'
import axios from 'axios'

interface MacroItem { symbol: string; name: string; price: number; change: number; pct: number }

const API = '/api'

export default function MacroTickerBar() {
  const [items, setItems] = useState<MacroItem[]>([])
  const [flash, setFlash] = useState<Record<string, 'up' | 'down'>>({})
  const prevRef = useRef<Record<string, number>>({})

  const fetchMacro = useCallback(async () => {
    try {
      const res  = await axios.get<{ items: MacroItem[] }>(`${API}/macro`)
      const next = res.data.items
      const newFlash: Record<string, 'up' | 'down'> = {}
      next.forEach(it => {
        const prev = prevRef.current[it.symbol]
        if (prev !== undefined && prev !== it.price)
          newFlash[it.symbol] = it.price > prev ? 'up' : 'down'
        prevRef.current[it.symbol] = it.price
      })
      setItems(next)
      if (Object.keys(newFlash).length) {
        setFlash(newFlash)
        setTimeout(() => setFlash({}), 700)
      }
    } catch { /* silent */ }
  }, [])

  useEffect(() => {
    fetchMacro()
    const id = setInterval(fetchMacro, 5000)
    return () => clearInterval(id)
  }, [fetchMacro])

  if (!items.length) return (
    <div className="h-12 border-b border-gray-800 bg-gray-950 flex items-center justify-center">
      <div className="w-3 h-3 border border-gray-700 border-t-gray-500 rounded-full animate-spin" />
    </div>
  )

  return (
    <div className="shrink-0 border-b border-gray-800 bg-gray-950 overflow-x-auto">
      <div className="flex h-12 min-w-max">
        {items.map((it, i) => {
          const up     = it.pct >= 0
          const fl     = flash[it.symbol]
          const color  = up ? '#22c55e' : '#ef4444'
          const priceStr = it.price >= 10000
            ? it.price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
            : it.price >= 100
            ? it.price.toFixed(2)
            : it.price.toFixed(4)

          return (
            <div
              key={it.symbol}
              className="flex items-center gap-3 px-4 transition-colors duration-500"
              style={{
                borderLeft: `3px solid ${color}`,
                borderRight: i < items.length - 1 ? '1px solid rgba(255,255,255,0.06)' : 'none',
                backgroundColor: fl === 'up'   ? 'rgba(34,197,94,0.07)'
                               : fl === 'down' ? 'rgba(239,68,68,0.07)'
                               : 'transparent',
                minWidth: 130,
              }}
            >
              {/* Text block */}
              <div>
                <div className="text-[9px] text-gray-500 font-semibold tracking-wide leading-none mb-0.5">
                  {it.name}
                </div>
                <div className="flex items-baseline gap-2">
                  <span
                    className="text-[13px] font-mono font-black leading-none"
                    style={{ color: up ? '#f3f4f6' : '#d1d5db' }}
                  >
                    {priceStr}
                  </span>
                  <span
                    className="text-[10px] font-bold leading-none"
                    style={{ color }}
                  >
                    {up ? '▲' : '▼'}{Math.abs(it.change) >= 1
                      ? it.change.toFixed(2)
                      : it.change.toFixed(4)}
                  </span>
                </div>
                <div className="text-[9px] font-bold mt-0.5" style={{ color }}>
                  {up ? '+' : ''}{it.pct.toFixed(2)}%
                </div>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
