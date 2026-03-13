/**
 * Hook managing chart drawing tools: state, interaction, persistence, and
 * primitive lifecycle (attach/detach on chart recreation).
 */
import { useState, useCallback, useRef, useEffect } from 'react'
import type { IChartApi, Time, MouseEventParams } from 'lightweight-charts'
import type { DrawingData, DrawingToolType, DrawingPoint, DrawingInteractionState } from '../types/drawings'
import { DRAWING_COLORS } from '../types/drawings'
import {
  createDrawingPrimitive,
  PreviewPrimitive,
  type DrawingPrimitiveBase,
} from '../components/chart/drawingPrimitives'
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type CandlestickRef = React.RefObject<ReturnType<IChartApi['addCandlestickSeries']> | null>

const STORAGE_KEY = 'roboalgo-chart-drawings'

function loadAllDrawings(): Record<string, DrawingData[]> {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}') }
  catch { return {} }
}

function persistAll(store: Record<string, DrawingData[]>) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(store))
}

// ── Hook ──────────────────────────────────────────────────────────────────────

export function useChartDrawings(
  symbol: string,
  tf: string,
  chartApi: React.RefObject<IChartApi | null>,
  candlestickRef: CandlestickRef,
) {
  const storeKey = `${symbol}::${tf}`

  // ── Drawing state ──────────────────────────────────────────────────────
  const [drawings, setDrawings] = useState<DrawingData[]>(() => loadAllDrawings()[storeKey] ?? [])
  const [interaction, setInteraction] = useState<DrawingInteractionState>({
    activeTool: null, pendingPoint: null, selectedDrawingId: null, previewPoint: null,
  })
  const [activeColor, setActiveColor] = useState<string>(DRAWING_COLORS[0])

  // ── Refs for primitive lifecycle ───────────────────────────────────────
  const primitivesRef = useRef<Map<string, DrawingPrimitiveBase>>(new Map())
  const previewRef = useRef<PreviewPrimitive | null>(null)

  // Latest-value refs so stable handlers can read current state
  const drawingsRef = useRef(drawings)
  drawingsRef.current = drawings
  const interactionRef = useRef(interaction)
  interactionRef.current = interaction
  const activeColorRef = useRef(activeColor)
  activeColorRef.current = activeColor

  // ── Persist on change ──────────────────────────────────────────────────
  useEffect(() => {
    const store = loadAllDrawings()
    store[storeKey] = drawings
    persistAll(store)
  }, [drawings, storeKey])

  // ── Reload when symbol/tf changes ──────────────────────────────────────
  useEffect(() => {
    setDrawings(loadAllDrawings()[storeKey] ?? [])
    setInteraction({ activeTool: null, pendingPoint: null, selectedDrawingId: null, previewPoint: null })
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [storeKey])

  // ── Tool selection ─────────────────────────────────────────────────────
  const selectTool = useCallback((tool: DrawingToolType | null) => {
    // Deselect any currently selected drawing when switching tools
    const prevId = interactionRef.current.selectedDrawingId
    if (prevId) primitivesRef.current.get(prevId)?.setSelected(false)
    setInteraction({ activeTool: tool, pendingPoint: null, selectedDrawingId: null, previewPoint: null })
    const chart = chartApi.current
    if (chart) {
      chart.applyOptions({ handleScroll: tool === null, handleScale: tool === null })
    }
  }, [chartApi])

  // ── Add / remove drawings ──────────────────────────────────────────────
  const addDrawing = useCallback((drawing: DrawingData) => {
    setDrawings(prev => [...prev, drawing])
    const series = candlestickRef.current
    if (series) {
      const prim = createDrawingPrimitive(drawing)
      if (prim) {
        series.attachPrimitive(prim)
        primitivesRef.current.set(drawing.id, prim)
      }
    }
  }, [candlestickRef])

  const deleteDrawing = useCallback((id: string) => {
    const series = candlestickRef.current
    const prim = primitivesRef.current.get(id)
    if (series && prim) {
      try { series.detachPrimitive(prim) } catch { /* ok */ }
      primitivesRef.current.delete(id)
    }
    setDrawings(prev => prev.filter(d => d.id !== id))
    setInteraction(prev => ({ ...prev, selectedDrawingId: null }))
  }, [candlestickRef])

  const clearAllDrawings = useCallback(() => {
    const series = candlestickRef.current
    if (series) {
      primitivesRef.current.forEach(p => { try { series.detachPrimitive(p) } catch { /* ok */ } })
      primitivesRef.current.clear()
    }
    setDrawings([])
  }, [candlestickRef])

  /**
   * Update one drawing's visual parameters (color, lineWidth, lineStyle,
   * extended, fillOpacity). Destroys and recreates the primitive so the
   * canvas reflects changes immediately; keeps the drawing selected.
   */
  const updateDrawing = useCallback((id: string, updates: Partial<DrawingData>) => {
    const series = candlestickRef.current
    const current = drawingsRef.current.find(d => d.id === id)
    if (!current) return
    const updated: DrawingData = { ...current, ...updates }

    // Replace primitive
    if (series) {
      const oldPrim = primitivesRef.current.get(id)
      if (oldPrim) {
        try { series.detachPrimitive(oldPrim) } catch { /* ok */ }
        primitivesRef.current.delete(id)
      }
      const prim = createDrawingPrimitive(updated)
      if (prim) {
        series.attachPrimitive(prim)
        prim.setSelected(true)
        primitivesRef.current.set(id, prim)
      }
    }

    // Update React state → triggers persistence useEffect
    setDrawings(prev => prev.map(d => d.id === id ? updated : d))
  }, [candlestickRef])

  // ── Attach all primitives (called after chart recreation) ──────────────
  const attachAllPrimitives = useCallback(() => {
    const series = candlestickRef.current
    if (!series) return
    primitivesRef.current.forEach(p => { try { series.detachPrimitive(p) } catch { /* ok */ } })
    primitivesRef.current.clear()
    for (const d of drawingsRef.current) {
      const prim = createDrawingPrimitive(d)
      if (prim) {
        series.attachPrimitive(prim)
        primitivesRef.current.set(d.id, prim)
      }
    }
  }, [candlestickRef])

  // ── Stable click handler ───────────────────────────────────────────────
  const handleClick = useCallback((param: MouseEventParams<Time>) => {
    const { activeTool, pendingPoint } = interactionRef.current

    if (!activeTool) {
      // ── Selection mode: hit-test every primitive ─────────────────────
      if (!param.point) return
      const px = param.point.x as number
      const py = param.point.y as number

      let hitId: string | null = null
      for (const [id, prim] of primitivesRef.current) {
        if (prim.hitTestPoint(px, py)) { hitId = id; break }
      }

      const prevId = interactionRef.current.selectedDrawingId
      if (prevId !== hitId) {
        if (prevId) primitivesRef.current.get(prevId)?.setSelected(false)
        if (hitId)  primitivesRef.current.get(hitId)?.setSelected(true)
        setInteraction(prev => ({ ...prev, selectedDrawingId: hitId }))
      }
      return
    }

    if (!param.time || !param.point) return
    const series = candlestickRef.current
    if (!series) return
    const price = series.coordinateToPrice(param.point.y)
    if (price === null) return

    const clickPt: DrawingPoint = { time: param.time, price: price as number }

    if (activeTool === 'horizontal') {
      addDrawing({ id: crypto.randomUUID(), type: 'horizontal', points: [clickPt], color: activeColorRef.current })
      selectTool(null)
      return
    }

    if (!pendingPoint) {
      setInteraction(prev => ({ ...prev, pendingPoint: clickPt }))
    } else {
      addDrawing({ id: crypto.randomUUID(), type: activeTool, points: [pendingPoint, clickPt], color: activeColorRef.current })
      const ser = candlestickRef.current
      if (ser && previewRef.current) {
        try { ser.detachPrimitive(previewRef.current) } catch { /* ok */ }
        previewRef.current = null
      }
      selectTool(null)
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []) // stable: reads from refs

  // ── Stable crosshair-move handler (rubber-band preview) ────────────────
  const handleCrosshairMove = useCallback((param: MouseEventParams<Time>) => {
    const { activeTool, pendingPoint } = interactionRef.current
    if (!activeTool || !pendingPoint) return
    if (!param.time || !param.point) return
    const series = candlestickRef.current
    if (!series) return
    const price = series.coordinateToPrice(param.point.y)
    if (price === null) return

    const previewPt: DrawingPoint = { time: param.time, price: price as number }
    if (previewRef.current) {
      previewRef.current.updateTo(previewPt)
    } else {
      const preview = new PreviewPrimitive(activeTool, pendingPoint, previewPt, activeColorRef.current)
      series.attachPrimitive(preview)
      previewRef.current = preview
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []) // stable: reads from refs

  // ── Keyboard shortcuts ─────────────────────────────────────────────────
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if ((e.key === 'Delete' || e.key === 'Backspace') && interactionRef.current.selectedDrawingId) {
        if ((e.target as HTMLElement)?.tagName === 'INPUT' || (e.target as HTMLElement)?.tagName === 'TEXTAREA') return
        e.preventDefault()
        deleteDrawing(interactionRef.current.selectedDrawingId)
      }
      if (e.key === 'Escape') {
        const ser = candlestickRef.current
        if (ser && previewRef.current) {
          try { ser.detachPrimitive(previewRef.current) } catch { /* ok */ }
          previewRef.current = null
        }
        selectTool(null)
      }
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [deleteDrawing, selectTool, candlestickRef])

  return {
    drawings,
    interaction,
    activeColor,
    setActiveColor,
    selectTool,
    handleClick,
    handleCrosshairMove,
    attachAllPrimitives,
    addDrawing,
    deleteDrawing,
    clearAllDrawings,
    updateDrawing,
  }
}
