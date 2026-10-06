import { useCallback, useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import axios from 'axios'

interface MacroItem { symbol: string; name: string; price: number; change: number; pct: number }

const API = '/api'
const AUTO_RESUME_MS = 3000 // resume auto-scroll 3s after last user interaction

export default function MacroTickerBar() {
  const navigate = useNavigate()
  const [items, setItems] = useState<MacroItem[]>([])
  const [flash, setFlash] = useState<Record<string, 'up' | 'down'>>({})
  const prevRef = useRef<Record<string, number>>({})

  // Scroll refs
  const trackRef = useRef<HTMLDivElement>(null)
  const scrollRef = useRef<HTMLDivElement>(null)
  const animRef = useRef<number>(0)
  const offsetRef = useRef(0)           // current CSS translateX offset (negative)
  const userScrolling = useRef(false)
  const resumeTimer = useRef<ReturnType<typeof setTimeout>>(undefined)
  const lastTime = useRef(0)
  const isDragging = useRef(false)
  const dragStartX = useRef(0)
  const dragStartScroll = useRef(0)

  const openChart = useCallback((symbol: string) => {
    navigate(`/chart?symbol=${encodeURIComponent(symbol)}`)
  }, [navigate])

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

  // ── Auto-scroll animation loop ──
  const speed = 40 // px per second

  const animate = useCallback((time: number) => {
    if (!trackRef.current) { animRef.current = requestAnimationFrame(animate); return }
    if (lastTime.current === 0) lastTime.current = time

    const dt = (time - lastTime.current) / 1000
    lastTime.current = time

    if (!userScrolling.current) {
      offsetRef.current -= speed * dt
      // Seamless loop: track is duplicated, so when we scroll past half, reset
      const halfWidth = trackRef.current.scrollWidth / 2
      if (halfWidth > 0 && Math.abs(offsetRef.current) >= halfWidth) {
        offsetRef.current += halfWidth
      }
      trackRef.current.style.transform = `translateX(${offsetRef.current}px)`
    }

    animRef.current = requestAnimationFrame(animate)
  }, [speed])

  useEffect(() => {
    animRef.current = requestAnimationFrame(animate)
    return () => cancelAnimationFrame(animRef.current)
  }, [animate])

  // ── Pause / resume helpers ──
  const pauseAutoScroll = useCallback(() => {
    userScrolling.current = true
    if (resumeTimer.current) clearTimeout(resumeTimer.current)
  }, [])

  const scheduleResume = useCallback(() => {
    if (resumeTimer.current) clearTimeout(resumeTimer.current)
    resumeTimer.current = setTimeout(() => {
      userScrolling.current = false
      lastTime.current = 0 // reset dt so we don't get a jump
    }, AUTO_RESUME_MS)
  }, [])

  // ── Mouse wheel horizontal scroll ──
  const onWheel = useCallback((e: React.WheelEvent) => {
    // Use deltaX if available (trackpad), otherwise convert deltaY to horizontal
    const dx = Math.abs(e.deltaX) > Math.abs(e.deltaY) ? e.deltaX : e.deltaY
    if (dx === 0) return
    e.preventDefault()
    pauseAutoScroll()
    offsetRef.current -= dx
    // Keep in bounds
    if (trackRef.current) {
      const halfWidth = trackRef.current.scrollWidth / 2
      if (halfWidth > 0) {
        while (Math.abs(offsetRef.current) >= halfWidth) offsetRef.current += halfWidth
        while (offsetRef.current > 0) offsetRef.current -= halfWidth
      }
      trackRef.current.style.transform = `translateX(${offsetRef.current}px)`
    }
    scheduleResume()
  }, [pauseAutoScroll, scheduleResume])

  // ── Mouse drag to scroll ──
  const onMouseDown = useCallback((e: React.MouseEvent) => {
    // Don't start drag on ticker item click (handled by item's own handler)
    isDragging.current = true
    dragStartX.current = e.clientX
    dragStartScroll.current = offsetRef.current
    pauseAutoScroll()
  }, [pauseAutoScroll])

  const onMouseMove = useCallback((e: React.MouseEvent) => {
    if (!isDragging.current) return
    const dx = e.clientX - dragStartX.current
    offsetRef.current = dragStartScroll.current + dx
    if (trackRef.current) {
      const halfWidth = trackRef.current.scrollWidth / 2
      if (halfWidth > 0) {
        while (Math.abs(offsetRef.current) >= halfWidth) offsetRef.current += halfWidth
        while (offsetRef.current > 0) offsetRef.current -= halfWidth
      }
      trackRef.current.style.transform = `translateX(${offsetRef.current}px)`
    }
  }, [])

  const onMouseUp = useCallback(() => {
    if (!isDragging.current) return
    isDragging.current = false
    scheduleResume()
  }, [scheduleResume])

  // ── Touch drag to scroll ──
  const onTouchStart = useCallback((e: React.TouchEvent) => {
    isDragging.current = true
    dragStartX.current = e.touches[0].clientX
    dragStartScroll.current = offsetRef.current
    pauseAutoScroll()
  }, [pauseAutoScroll])

  const onTouchMove = useCallback((e: React.TouchEvent) => {
    if (!isDragging.current) return
    const dx = e.touches[0].clientX - dragStartX.current
    offsetRef.current = dragStartScroll.current + dx
    if (trackRef.current) {
      const halfWidth = trackRef.current.scrollWidth / 2
      if (halfWidth > 0) {
        while (Math.abs(offsetRef.current) >= halfWidth) offsetRef.current += halfWidth
        while (offsetRef.current > 0) offsetRef.current -= halfWidth
      }
      trackRef.current.style.transform = `translateX(${offsetRef.current}px)`
    }
  }, [])

  const onTouchEnd = useCallback(() => {
    isDragging.current = false
    scheduleResume()
  }, [scheduleResume])

  // ── Render a single ticker item ──
  const dragMoved = useRef(false)

  const renderItem = (it: MacroItem, i: number, keyPrefix: string) => {
    const up       = it.pct >= 0
    const fl       = flash[it.symbol]
    const color    = up ? '#22c55e' : '#ef4444'
    const priceStr = it.price >= 10000
      ? it.price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
      : it.price >= 100
      ? it.price.toFixed(2)
      : it.price.toFixed(4)

    return (
      <div
        key={`${keyPrefix}-${i}`}
        role="button"
        tabIndex={0}
        onMouseDown={(e) => {
          dragMoved.current = false
          dragStartX.current = e.clientX
        }}
        onMouseUp={(e) => {
          // Only navigate if user didn't drag (< 5px movement)
          const dist = Math.abs(e.clientX - dragStartX.current)
          if (dist < 5) openChart(it.symbol)
        }}
        onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); openChart(it.symbol) } }}
        title={`Open ${it.name} chart`}
        className="flex items-center gap-3 px-4 transition-colors duration-500 cursor-pointer hover:brightness-125 focus:outline-none focus:ring-1 focus:ring-blue-500/50 select-none"
        style={{
          borderLeft: `3px solid ${color}`,
          borderRight: '1px solid rgba(255,255,255,0.06)',
          backgroundColor: fl === 'up'   ? 'rgba(34,197,94,0.07)'
                         : fl === 'down' ? 'rgba(239,68,68,0.07)'
                         : 'transparent',
          minWidth: 130,
        }}
      >
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
  }

  if (!items.length) return (
    <div className="h-12 border-b border-gray-800 bg-gray-950 flex items-center justify-center">
      <div className="w-3 h-3 border border-gray-700 border-t-gray-500 rounded-full animate-spin" />
    </div>
  )

  return (
    <div
      ref={scrollRef}
      className="shrink-0 border-b border-gray-800 bg-gray-950 overflow-hidden"
      style={{ cursor: isDragging.current ? 'grabbing' : 'grab' }}
      onWheel={onWheel}
      onMouseDown={onMouseDown}
      onMouseMove={onMouseMove}
      onMouseUp={onMouseUp}
      onMouseLeave={onMouseUp}
      onTouchStart={onTouchStart}
      onTouchMove={onTouchMove}
      onTouchEnd={onTouchEnd}
    >
      {/* Duplicate items so the loop is seamless */}
      <div
        ref={trackRef}
        className="flex h-12 w-max"
        style={{ willChange: 'transform' }}
      >
        {items.map((it, i) => renderItem(it, i, 'a'))}
        {items.map((it, i) => renderItem(it, i, 'b'))}
      </div>
    </div>
  )
}
