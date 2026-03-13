import type { Time } from 'lightweight-charts'

// ── Tool types ────────────────────────────────────────────────────────────────
export type DrawingToolType = 'trendline' | 'horizontal' | 'fibonacci' | 'rectangle'
export type LineStyle = 'solid' | 'dashed' | 'dotted'

// ── Data structures ───────────────────────────────────────────────────────────
export interface DrawingPoint {
  time: Time    // string ("2024-06-15") for daily, number (unix) for intraday
  price: number
}

export interface DrawingData {
  id: string
  type: DrawingToolType
  points: DrawingPoint[]  // 1 for horizontal, 2 for others
  color: string
  extended?: boolean      // ray mode for trendlines
  lineWidth?: number      // 1–4 px (default 2)
  lineStyle?: LineStyle   // default 'solid'; horizontal is always dashed
  fillOpacity?: number    // 0.05–0.4 for rectangle / fib zone fill (default 0.1)
}

// ── Interaction state machine ─────────────────────────────────────────────────
export interface DrawingInteractionState {
  activeTool: DrawingToolType | null
  pendingPoint: DrawingPoint | null
  selectedDrawingId: string | null
  previewPoint: DrawingPoint | null
}

// ── Constants ─────────────────────────────────────────────────────────────────
export const DRAWING_COLORS = [
  '#3b82f6', // blue
  '#ef4444', // red
  '#22c55e', // green
  '#f59e0b', // amber
  '#a855f7', // purple
  '#ec4899', // pink
  '#14b8a6', // teal
  '#f97316', // orange
] as const

export const DEFAULT_FIB_LEVELS = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0] as const
