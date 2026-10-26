import { createContext, useContext, useEffect, useState, type ReactNode } from 'react'
import axios from 'axios'

export interface MacroItem { symbol: string; name: string; price: number; change: number; pct: number }

interface MacroContextValue {
  items:   MacroItem[]
  loading: boolean
  error:   string | null
}

const MacroContext = createContext<MacroContextValue>({
  items:   [],
  loading: true,
  error:   null,
})

// ONE poll for the whole app. The macro endpoint is yfinance-backed and was
// being hit by two independent 5 s intervals (ticker bar + side nav), which
// tripped Yahoo's rate limiter. Every consumer reads from here instead.
const MACRO_URL        = '/api/macro'
const MACRO_POLL_MS    = 15_000

export function MacroProvider({ children }: { children: ReactNode }) {
  const [items,   setItems]   = useState<MacroItem[]>([])
  const [loading, setLoading] = useState(true)
  const [error,   setError]   = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false

    const fetchMacro = async () => {
      try {
        const res  = await axios.get<{ items?: MacroItem[] }>(MACRO_URL)
        if (cancelled) return
        const next = res.data?.items
        // An empty payload means the upstream failed; keep the last good data
        // rather than blanking the bar.
        if (Array.isArray(next) && next.length) {
          setItems(next)
          setError(null)
        } else {
          setError('Macro feed returned no data')
        }
      } catch (e) {
        if (cancelled) return
        setError(e instanceof Error ? e.message : 'Macro fetch failed')
      } finally {
        if (!cancelled) setLoading(false)
      }
    }

    fetchMacro()
    const id = setInterval(fetchMacro, MACRO_POLL_MS)
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [])

  return (
    <MacroContext.Provider value={{ items, loading, error }}>
      {children}
    </MacroContext.Provider>
  )
}

export function useMacro(): MacroContextValue {
  return useContext(MacroContext)
}
