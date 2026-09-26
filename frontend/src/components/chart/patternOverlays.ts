// patternOverlays — custom klinecharts overlay types used by KLineChartOverlay.
//
// KLineChartOverlay.tsx creates overlays by name (`chart.createOverlay({ name: … })`)
// and passes all per-overlay styling through `extendData`. klinecharts only knows
// the built-in overlay types, so each custom name has to be registered once before
// any chart is created — hence the module-scope `registerPatternOverlays()` call
// in KLineChartOverlay.tsx.
//
// Five types, matching exactly what the call sites pass:
//
//   patternTrendline  2 points  { color, lineWidth, dash }                       → line segment
//   patternBadge      1 point   { text, bgColor, textColor }                     → filled pill + text
//   patternLabel      1 point   { text, color, fontSize, offsetY }               → bare text, no background
//   waveCircle        1 point   { text, fillColor, textColor, radius, borderColor } → circle + centred text
//   waveHandle        1 point   { color, label }                                 → draggable grab handle
//
// Registration is idempotent: `registerPatternOverlays()` is safe to call more
// than once (React fast-refresh re-executes module scope), because re-registering
// the same name would otherwise replace the template on every reload.

import {
  registerOverlay,
  type OverlayFigure,
  type OverlayTemplate,
} from 'klinecharts'

// ── extendData shapes ───────────────────────────────────────────────────────

interface TrendlineData {
  color?: string
  lineWidth?: number
  dash?: number[]
}

interface BadgeData {
  text?: string
  bgColor?: string
  textColor?: string
}

interface LabelData {
  text?: string
  color?: string
  fontSize?: number
  offsetY?: number
}

interface CircleData {
  text?: string
  fillColor?: string
  textColor?: string
  radius?: number
  borderColor?: string
}

interface HandleData {
  color?: string
  label?: string
}

// `extendData` is typed `unknown` on the overlay; these read it defensively so a
// malformed or absent payload renders a sane default instead of throwing inside
// the chart's render loop (which would blank the whole canvas, not just one overlay).
function data<T>(raw: unknown): Partial<T> {
  return (raw && typeof raw === 'object' ? raw : {}) as Partial<T>
}

const FONT_FAMILY =
  'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace'

// ── Templates ───────────────────────────────────────────────────────────────

const patternTrendline: OverlayTemplate = {
  name: 'patternTrendline',
  totalStep: 2,
  needDefaultPointFigure: false,
  needDefaultXAxisFigure: false,
  needDefaultYAxisFigure: false,
  createPointFigures: ({ coordinates, overlay }) => {
    if (coordinates.length < 2) return []
    const d = data<TrendlineData>(overlay.extendData)
    const dash = d.dash ?? []
    return [
      {
        type: 'line',
        attrs: { coordinates: [coordinates[0], coordinates[1]] },
        styles: {
          color: d.color ?? '#60a5fa',
          size: d.lineWidth ?? 1,
          // klinecharts only honours dashedValue when style is 'dashed'
          style: dash.length > 0 ? 'dashed' : 'solid',
          dashedValue: dash.length > 0 ? dash : [2, 2],
        },
      },
    ]
  },
}

const patternBadge: OverlayTemplate = {
  name: 'patternBadge',
  totalStep: 1,
  needDefaultPointFigure: false,
  needDefaultXAxisFigure: false,
  needDefaultYAxisFigure: false,
  createPointFigures: ({ coordinates, overlay }) => {
    if (coordinates.length < 1) return []
    const d = data<BadgeData>(overlay.extendData)
    const text = d.text ?? ''
    if (!text) return []
    return [
      {
        type: 'text',
        attrs: {
          x: coordinates[0].x,
          y: coordinates[0].y,
          text,
          align: 'center',
          baseline: 'middle',
        },
        styles: {
          color: d.textColor ?? '#ffffff',
          size: 10,
          family: FONT_FAMILY,
          weight: 'bold',
          backgroundColor: d.bgColor ?? 'rgba(10,15,30,0.85)',
          borderRadius: 3,
          borderSize: 0,
          borderColor: 'transparent',
          paddingLeft: 6,
          paddingRight: 6,
          paddingTop: 3,
          paddingBottom: 3,
        },
      },
    ]
  },
}

const patternLabel: OverlayTemplate = {
  name: 'patternLabel',
  totalStep: 1,
  needDefaultPointFigure: false,
  needDefaultXAxisFigure: false,
  needDefaultYAxisFigure: false,
  createPointFigures: ({ coordinates, overlay }) => {
    if (coordinates.length < 1) return []
    const d = data<LabelData>(overlay.extendData)
    const text = d.text ?? ''
    if (!text) return []
    return [
      {
        type: 'text',
        attrs: {
          x: coordinates[0].x,
          // offsetY is in screen px and may be negative (label below the point)
          y: coordinates[0].y + (d.offsetY ?? 0),
          text,
          align: 'center',
          baseline: 'middle',
        },
        styles: {
          color: d.color ?? 'rgba(148,163,184,0.8)',
          size: d.fontSize ?? 10,
          family: FONT_FAMILY,
          weight: 'normal',
          // No background — patternBadge is the pill variant.
          backgroundColor: 'transparent',
          borderSize: 0,
          borderColor: 'transparent',
          paddingLeft: 0,
          paddingRight: 0,
          paddingTop: 0,
          paddingBottom: 0,
        },
      },
    ]
  },
}

const waveCircle: OverlayTemplate = {
  name: 'waveCircle',
  totalStep: 1,
  needDefaultPointFigure: false,
  needDefaultXAxisFigure: false,
  needDefaultYAxisFigure: false,
  createPointFigures: ({ coordinates, overlay }) => {
    if (coordinates.length < 1) return []
    const d = data<CircleData>(overlay.extendData)
    const r = d.radius ?? 9
    const { x, y } = coordinates[0]
    const figures: OverlayFigure[] = [
      {
        type: 'circle',
        attrs: { x, y, r },
        styles: {
          style: 'stroke_fill',
          color: d.fillColor ?? '#1e293b',
          borderColor: d.borderColor ?? 'rgba(255,255,255,0.25)',
          borderSize: 1,
        },
      },
    ]
    const text = d.text ?? ''
    if (text) {
      figures.push({
        type: 'text',
        attrs: { x, y, text, align: 'center', baseline: 'middle' },
        // Scale the label down for multi-character wave names so it stays
        // inside the circle instead of spilling past the border.
        styles: {
          color: d.textColor ?? '#ffffff',
          size: text.length > 2 ? Math.max(7, r - 2) : r,
          family: FONT_FAMILY,
          weight: 'bold',
          backgroundColor: 'transparent',
          borderSize: 0,
          borderColor: 'transparent',
          paddingLeft: 0,
          paddingRight: 0,
          paddingTop: 0,
          paddingBottom: 0,
        },
      })
    }
    return figures
  },
}

const waveHandle: OverlayTemplate = {
  name: 'waveHandle',
  totalStep: 1,
  needDefaultPointFigure: false,
  needDefaultXAxisFigure: false,
  needDefaultYAxisFigure: false,
  createPointFigures: ({ coordinates, overlay }) => {
    if (coordinates.length < 1) return []
    const d = data<HandleData>(overlay.extendData)
    const color = d.color ?? '#fde047'
    const { x, y } = coordinates[0]
    const figures: OverlayFigure[] = [
      // Outer ring gives a bigger grab target than the visible dot; this
      // overlay is created with lock:false and is meant to be dragged.
      {
        type: 'circle',
        attrs: { x, y, r: 7 },
        styles: {
          style: 'stroke_fill',
          color: 'rgba(253,224,71,0.18)',
          borderColor: color,
          borderSize: 1.5,
        },
      },
      {
        type: 'circle',
        attrs: { x, y, r: 3 },
        styles: { style: 'fill', color, borderColor: color, borderSize: 0 },
      },
    ]
    const label = d.label ?? ''
    if (label) {
      figures.push({
        type: 'text',
        attrs: { x, y: y - 14, text: label, align: 'center', baseline: 'middle' },
        styles: {
          color,
          size: 9,
          family: FONT_FAMILY,
          weight: 'bold',
          backgroundColor: 'rgba(10,15,30,0.8)',
          borderRadius: 2,
          borderSize: 0,
          borderColor: 'transparent',
          paddingLeft: 3,
          paddingRight: 3,
          paddingTop: 1,
          paddingBottom: 1,
        },
      })
    }
    return figures
  },
}

// ── Registration ────────────────────────────────────────────────────────────

const TEMPLATES: OverlayTemplate[] = [
  patternTrendline,
  patternBadge,
  patternLabel,
  waveCircle,
  waveHandle,
]

let registered = false

/**
 * Register every custom overlay type used by KLineChartOverlay.
 *
 * Must run before the first `init()` — KLineChartOverlay calls it at module
 * scope. Idempotent, so React fast-refresh re-running module scope is harmless.
 */
export function registerPatternOverlays(): void {
  if (registered) return
  for (const template of TEMPLATES) {
    registerOverlay(template)
  }
  registered = true
}

export const PATTERN_OVERLAY_NAMES = TEMPLATES.map(t => t.name)
