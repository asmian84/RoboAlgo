/**
 * ISeriesPrimitive implementations for manual chart drawing tools.
 *
 * Each primitive stores chart/series refs from attached(), renders via
 * paneViews() → renderer's draw(target), and supports hit-testing for selection.
 * All visual parameters (lineWidth, lineStyle, fillOpacity) come from DrawingData.
 */
import type {
  ISeriesPrimitive,
  ISeriesPrimitivePaneView,
  ISeriesPrimitivePaneRenderer,
  ISeriesPrimitiveAxisView,
  SeriesAttachedParameter,
  SeriesType,
  Time,
  IChartApiBase,
  ISeriesApi,
  Coordinate,
} from 'lightweight-charts'
import type { CanvasRenderingTarget2D } from 'fancy-canvas'
import type { DrawingData, DrawingPoint, DrawingToolType, LineStyle } from '../../types/drawings'
import { DEFAULT_FIB_LEVELS } from '../../types/drawings'

// ── Shared utilities ──────────────────────────────────────────────────────────

function toPixel(
  point: DrawingPoint,
  series: ISeriesApi<SeriesType, Time>,
  chart: IChartApiBase<Time>,
): { x: number; y: number } | null {
  const y = series.priceToCoordinate(point.price) as number | null
  const x = chart.timeScale().timeToCoordinate(point.time) as number | null
  if (y === null || x === null) return null
  return { x, y }
}

/** Perpendicular distance from point P to line segment AB */
function distToSegment(
  px: number, py: number,
  ax: number, ay: number,
  bx: number, by: number,
): number {
  const dx = bx - ax
  const dy = by - ay
  const lenSq = dx * dx + dy * dy
  if (lenSq === 0) return Math.hypot(px - ax, py - ay)
  let t = ((px - ax) * dx + (py - ay) * dy) / lenSq
  t = Math.max(0, Math.min(1, t))
  return Math.hypot(px - (ax + t * dx), py - (ay + t * dy))
}

/** Hex color string to rgba with given alpha */
function hexAlpha(hex: string, alpha: number): string {
  const r = parseInt(hex.slice(1, 3), 16)
  const g = parseInt(hex.slice(3, 5), 16)
  const b = parseInt(hex.slice(5, 7), 16)
  return `rgba(${r},${g},${b},${alpha})`
}

/** Apply line style dash pattern to a canvas context */
function applyLineStyle(ctx: CanvasRenderingContext2D, style: LineStyle | undefined, lw: number) {
  if (style === 'dashed')      ctx.setLineDash([lw * 5, lw * 2.5])
  else if (style === 'dotted') ctx.setLineDash([lw, lw * 2.5])
  else                         ctx.setLineDash([])
}

const HIT_THRESHOLD = 8 // px — slightly generous for comfortable selection

// ── Base class ────────────────────────────────────────────────────────────────

export abstract class DrawingPrimitiveBase implements ISeriesPrimitive<Time> {
  protected _chart: IChartApiBase<Time> | null = null
  protected _series: ISeriesApi<SeriesType, Time> | null = null
  protected _requestUpdate: (() => void) | null = null
  protected _selected = false

  constructor(public readonly data: DrawingData) {}

  attached(param: SeriesAttachedParameter<Time, SeriesType>) {
    this._chart = param.chart
    this._series = param.series
    this._requestUpdate = param.requestUpdate
  }

  detached() {
    this._chart = null
    this._series = null
    this._requestUpdate = null
  }

  updateAllViews() { /* coordinate conversion happens in renderer */ }

  setSelected(selected: boolean) {
    if (this._selected !== selected) {
      this._selected = selected
      this._requestUpdate?.()
    }
  }

  get chart() { return this._chart }
  get series() { return this._series }
  get selected() { return this._selected }

  abstract paneViews(): readonly ISeriesPrimitivePaneView[]

  /** Return true if the canvas point (px, py) is close enough to trigger selection. */
  abstract hitTestPoint(px: number, py: number): boolean
}

// ═════════════════════════════════════════════════════════════════════════════
// TRENDLINE
// ═════════════════════════════════════════════════════════════════════════════

class TrendlineRenderer implements ISeriesPrimitivePaneRenderer {
  constructor(private _src: TrendlinePrimitive) {}

  draw(target: CanvasRenderingTarget2D) {
    const { chart, series, data, selected } = this._src
    if (!chart || !series || data.points.length < 2) return
    const p1 = toPixel(data.points[0], series, chart)
    const p2 = toPixel(data.points[1], series, chart)
    if (!p1 || !p2) return

    const lw = (data.lineWidth ?? 2) + (selected ? 1 : 0)

    target.useMediaCoordinateSpace(({ context, mediaSize }) => {
      context.strokeStyle = data.color
      context.lineWidth = lw
      applyLineStyle(context, data.lineStyle, lw)
      context.beginPath()

      if (data.extended) {
        const dx = p2.x - p1.x
        const dy = p2.y - p1.y
        if (Math.abs(dx) > 0.001) {
          const slope = dy / dx
          const y0 = p1.y - slope * p1.x
          const yEnd = y0 + slope * mediaSize.width
          context.moveTo(0, y0)
          context.lineTo(mediaSize.width, yEnd)
        } else {
          context.moveTo(p1.x, 0)
          context.lineTo(p1.x, mediaSize.height)
        }
      } else {
        context.moveTo(p1.x, p1.y)
        context.lineTo(p2.x, p2.y)
      }
      context.stroke()
      context.setLineDash([])

      // Selection handles
      if (selected) {
        context.fillStyle = data.color
        for (const p of [p1, p2]) {
          context.fillRect(p.x - 4, p.y - 4, 8, 8)
        }
      }
    })
  }
}

class TrendlinePaneView implements ISeriesPrimitivePaneView {
  constructor(private _src: TrendlinePrimitive) {}
  zOrder(): 'normal' { return 'normal' }
  renderer() { return new TrendlineRenderer(this._src) }
}

export class TrendlinePrimitive extends DrawingPrimitiveBase {
  private _views: ISeriesPrimitivePaneView[]
  constructor(data: DrawingData) {
    super(data)
    this._views = [new TrendlinePaneView(this)]
  }
  paneViews() { return this._views }

  hitTestPoint(px: number, py: number): boolean {
    if (!this._chart || !this._series || this.data.points.length < 2) return false
    const p1 = toPixel(this.data.points[0], this._series, this._chart)
    const p2 = toPixel(this.data.points[1], this._series, this._chart)
    if (!p1 || !p2) return false
    if (this.data.extended) {
      const dx = p2.x - p1.x
      const dy = p2.y - p1.y
      if (Math.abs(dx) < 0.001) return Math.abs(px - p1.x) < HIT_THRESHOLD
      const slope = dy / dx
      const expectedY = p1.y + slope * (px - p1.x)
      return Math.abs(py - expectedY) < HIT_THRESHOLD
    }
    return distToSegment(px, py, p1.x, p1.y, p2.x, p2.y) < HIT_THRESHOLD
  }
}

// ═════════════════════════════════════════════════════════════════════════════
// HORIZONTAL LINE
// ═════════════════════════════════════════════════════════════════════════════

class HorizontalRenderer implements ISeriesPrimitivePaneRenderer {
  constructor(private _src: HorizontalLinePrimitive) {}

  draw(target: CanvasRenderingTarget2D) {
    const { chart, series, data, selected } = this._src
    if (!chart || !series || data.points.length < 1) return
    const y = series.priceToCoordinate(data.points[0].price) as number | null
    if (y === null) return

    const lw = (data.lineWidth ?? 1.5) + (selected ? 0.5 : 0)
    // Horizontal lines are dashed by default; lineStyle override applies
    const style: LineStyle = data.lineStyle ?? 'dashed'

    target.useMediaCoordinateSpace(({ context, mediaSize }) => {
      context.strokeStyle = data.color
      context.lineWidth = lw
      applyLineStyle(context, style, lw)
      context.beginPath()
      context.moveTo(0, y)
      context.lineTo(mediaSize.width, y)
      context.stroke()
      context.setLineDash([])

      // Price label on right edge
      const label = data.points[0].price.toFixed(2)
      context.font = '10px monospace'
      const tw = context.measureText(label).width
      const lx = mediaSize.width - tw - 8
      context.fillStyle = hexAlpha(data.color, 0.85)
      context.fillRect(lx - 3, y - 8, tw + 6, 16)
      context.fillStyle = '#fff'
      context.fillText(label, lx, y + 3)
    })
  }
}

class HorizontalPaneView implements ISeriesPrimitivePaneView {
  constructor(private _src: HorizontalLinePrimitive) {}
  zOrder(): 'normal' { return 'normal' }
  renderer() { return new HorizontalRenderer(this._src) }
}

export class HorizontalLinePrimitive extends DrawingPrimitiveBase {
  private _views: ISeriesPrimitivePaneView[]
  constructor(data: DrawingData) {
    super(data)
    this._views = [new HorizontalPaneView(this)]
  }
  paneViews() { return this._views }

  priceAxisViews(): readonly ISeriesPrimitiveAxisView[] {
    const self = this
    return [{
      coordinate(): Coordinate { return (self.series?.priceToCoordinate(self.data.points[0]?.price ?? 0) ?? 0) as Coordinate },
      text(): string { return self.data.points[0]?.price.toFixed(2) ?? '' },
      textColor(): string { return '#fff' },
      backColor(): string { return self.data.color },
      visible(): boolean { return true },
    }]
  }

  hitTestPoint(_px: number, py: number): boolean {
    if (!this._series || this.data.points.length < 1) return false
    const y = this._series.priceToCoordinate(this.data.points[0].price) as number | null
    if (y === null) return false
    return Math.abs(py - y) < HIT_THRESHOLD
  }
}

// ═════════════════════════════════════════════════════════════════════════════
// FIBONACCI RETRACEMENT
// ═════════════════════════════════════════════════════════════════════════════

class FibRenderer implements ISeriesPrimitivePaneRenderer {
  constructor(private _src: FibonacciRetracementPrimitive) {}

  draw(target: CanvasRenderingTarget2D) {
    const { chart, series, data, selected } = this._src
    if (!chart || !series || data.points.length < 2) return

    const p1 = toPixel(data.points[0], series, chart)
    const p2 = toPixel(data.points[1], series, chart)
    if (!p1 || !p2) return

    const priceHigh = Math.max(data.points[0].price, data.points[1].price)
    const priceLow = Math.min(data.points[0].price, data.points[1].price)
    const range = priceHigh - priceLow
    if (range <= 0) return

    const fillAlpha = data.fillOpacity ?? 0.07
    const lw = (data.lineWidth ?? 1) + (selected ? 0.5 : 0)

    target.useMediaCoordinateSpace(({ context, mediaSize }) => {
      const levels: { level: number; y: number; price: number }[] = []
      for (const lev of DEFAULT_FIB_LEVELS) {
        const price = priceHigh - range * lev
        const y = series.priceToCoordinate(price) as number | null
        if (y !== null) levels.push({ level: lev, y, price })
      }

      // Zone fills between adjacent levels
      for (let i = 0; i < levels.length - 1; i++) {
        const top = levels[i]
        const bot = levels[i + 1]
        const alpha = i % 2 === 0 ? fillAlpha : fillAlpha * 0.5
        context.fillStyle = hexAlpha(data.color, alpha)
        context.fillRect(0, top.y, mediaSize.width, bot.y - top.y)
      }

      // Horizontal lines at each level
      context.strokeStyle = data.color
      context.lineWidth = lw
      context.setLineDash([4, 2])
      for (const lev of levels) {
        context.beginPath()
        context.moveTo(0, lev.y)
        context.lineTo(mediaSize.width, lev.y)
        context.stroke()
      }
      context.setLineDash([])

      // Level labels on left side
      context.font = '10px monospace'
      context.fillStyle = hexAlpha(data.color, 0.9)
      for (const lev of levels) {
        const pct = (lev.level * 100).toFixed(1) + '%'
        const txt = `${pct}  ${lev.price.toFixed(2)}`
        context.fillText(txt, 6, lev.y - 3)
      }

      // Selection handles at anchor points
      if (selected) {
        context.fillStyle = data.color
        for (const p of [p1, p2]) {
          context.fillRect(p.x - 4, p.y - 4, 8, 8)
        }
      }
    })
  }
}

class FibPaneView implements ISeriesPrimitivePaneView {
  constructor(private _src: FibonacciRetracementPrimitive) {}
  zOrder(): 'normal' { return 'normal' }
  renderer() { return new FibRenderer(this._src) }
}

export class FibonacciRetracementPrimitive extends DrawingPrimitiveBase {
  private _views: ISeriesPrimitivePaneView[]
  constructor(data: DrawingData) {
    super(data)
    this._views = [new FibPaneView(this)]
  }
  paneViews() { return this._views }

  hitTestPoint(_px: number, py: number): boolean {
    if (!this._series || this.data.points.length < 2) return false
    const priceHigh = Math.max(this.data.points[0].price, this.data.points[1].price)
    const priceLow = Math.min(this.data.points[0].price, this.data.points[1].price)
    const range = priceHigh - priceLow
    if (range <= 0) return false
    for (const lev of DEFAULT_FIB_LEVELS) {
      const price = priceHigh - range * lev
      const y = this._series.priceToCoordinate(price) as number | null
      if (y !== null && Math.abs(py - y) < HIT_THRESHOLD) return true
    }
    return false
  }
}

// ═════════════════════════════════════════════════════════════════════════════
// RECTANGLE
// ═════════════════════════════════════════════════════════════════════════════

class RectRenderer implements ISeriesPrimitivePaneRenderer {
  constructor(private _src: RectanglePrimitive) {}

  draw(target: CanvasRenderingTarget2D) {
    const { chart, series, data, selected } = this._src
    if (!chart || !series || data.points.length < 2) return
    const p1 = toPixel(data.points[0], series, chart)
    const p2 = toPixel(data.points[1], series, chart)
    if (!p1 || !p2) return

    const lw = (data.lineWidth ?? 1) + (selected ? 1 : 0)
    const fillAlpha = data.fillOpacity ?? 0.1

    target.useMediaCoordinateSpace(({ context }) => {
      const x = Math.min(p1.x, p2.x)
      const y = Math.min(p1.y, p2.y)
      const w = Math.abs(p2.x - p1.x)
      const h = Math.abs(p2.y - p1.y)

      // Fill
      context.fillStyle = hexAlpha(data.color, fillAlpha)
      context.fillRect(x, y, w, h)

      // Border
      context.strokeStyle = data.color
      context.lineWidth = lw
      applyLineStyle(context, data.lineStyle, lw)
      context.strokeRect(x, y, w, h)
      context.setLineDash([])

      // Selection handles at corners
      if (selected) {
        context.fillStyle = data.color
        for (const p of [p1, p2, { x: p1.x, y: p2.y }, { x: p2.x, y: p1.y }]) {
          context.fillRect(p.x - 3, p.y - 3, 6, 6)
        }
      }
    })
  }
}

class RectPaneView implements ISeriesPrimitivePaneView {
  constructor(private _src: RectanglePrimitive) {}
  zOrder(): 'normal' { return 'normal' }
  renderer() { return new RectRenderer(this._src) }
}

export class RectanglePrimitive extends DrawingPrimitiveBase {
  private _views: ISeriesPrimitivePaneView[]
  constructor(data: DrawingData) {
    super(data)
    this._views = [new RectPaneView(this)]
  }
  paneViews() { return this._views }

  hitTestPoint(px: number, py: number): boolean {
    if (!this._chart || !this._series || this.data.points.length < 2) return false
    const p1 = toPixel(this.data.points[0], this._series, this._chart)
    const p2 = toPixel(this.data.points[1], this._series, this._chart)
    if (!p1 || !p2) return false
    const x1 = Math.min(p1.x, p2.x)
    const x2 = Math.max(p1.x, p2.x)
    const y1 = Math.min(p1.y, p2.y)
    const y2 = Math.max(p1.y, p2.y)
    // Near any of the 4 border edges
    if (px < x1 - HIT_THRESHOLD || px > x2 + HIT_THRESHOLD) return false
    if (py < y1 - HIT_THRESHOLD || py > y2 + HIT_THRESHOLD) return false
    return (
      Math.abs(px - x1) < HIT_THRESHOLD ||
      Math.abs(px - x2) < HIT_THRESHOLD ||
      Math.abs(py - y1) < HIT_THRESHOLD ||
      Math.abs(py - y2) < HIT_THRESHOLD
    )
  }
}

// ═════════════════════════════════════════════════════════════════════════════
// PREVIEW (rubber-band while drawing in progress)
// ═════════════════════════════════════════════════════════════════════════════

class PreviewRenderer implements ISeriesPrimitivePaneRenderer {
  constructor(private _src: PreviewPrimitive) {}

  draw(target: CanvasRenderingTarget2D) {
    const { chart, series, toolType, from, to, color } = this._src
    if (!chart || !series) return
    const p1 = toPixel(from, series, chart)
    const p2 = toPixel(to, series, chart)
    if (!p1 || !p2) return

    target.useMediaCoordinateSpace(({ context, mediaSize }) => {
      context.globalAlpha = 0.6
      context.setLineDash([5, 3])
      context.strokeStyle = color
      context.lineWidth = 1.5

      switch (toolType) {
        case 'trendline': {
          context.beginPath()
          context.moveTo(p1.x, p1.y)
          context.lineTo(p2.x, p2.y)
          context.stroke()
          break
        }
        case 'horizontal': {
          const y = p2.y
          context.beginPath()
          context.moveTo(0, y)
          context.lineTo(mediaSize.width, y)
          context.stroke()
          break
        }
        case 'fibonacci': {
          const priceHigh = Math.max(from.price, to.price)
          const priceLow = Math.min(from.price, to.price)
          const range = priceHigh - priceLow
          if (range <= 0) break
          for (const lev of DEFAULT_FIB_LEVELS) {
            const price = priceHigh - range * lev
            const y = series.priceToCoordinate(price) as number | null
            if (y === null) continue
            context.beginPath()
            context.moveTo(0, y)
            context.lineTo(mediaSize.width, y)
            context.stroke()
          }
          break
        }
        case 'rectangle': {
          const x = Math.min(p1.x, p2.x)
          const y = Math.min(p1.y, p2.y)
          const w = Math.abs(p2.x - p1.x)
          const h = Math.abs(p2.y - p1.y)
          context.fillStyle = hexAlpha(color, 0.08)
          context.fillRect(x, y, w, h)
          context.strokeRect(x, y, w, h)
          break
        }
      }

      context.globalAlpha = 1
      context.setLineDash([])
    })
  }
}

class PreviewPaneView implements ISeriesPrimitivePaneView {
  constructor(private _src: PreviewPrimitive) {}
  zOrder(): 'top' { return 'top' }
  renderer() { return new PreviewRenderer(this._src) }
}

export class PreviewPrimitive implements ISeriesPrimitive<Time> {
  private _chart: IChartApiBase<Time> | null = null
  private _series: ISeriesApi<SeriesType, Time> | null = null
  private _requestUpdate: (() => void) | null = null
  private _views: ISeriesPrimitivePaneView[]

  constructor(
    public readonly toolType: DrawingToolType,
    public from: DrawingPoint,
    public to: DrawingPoint,
    public color: string,
  ) {
    this._views = [new PreviewPaneView(this)]
  }

  attached(param: SeriesAttachedParameter<Time, SeriesType>) {
    this._chart = param.chart
    this._series = param.series
    this._requestUpdate = param.requestUpdate
  }
  detached() { this._chart = null; this._series = null; this._requestUpdate = null }
  updateAllViews() {}
  paneViews() { return this._views }

  get chart() { return this._chart }
  get series() { return this._series }

  /** Update preview endpoint and trigger redraw */
  updateTo(point: DrawingPoint) {
    this.to = point
    this._requestUpdate?.()
  }
}

// ── Factory ───────────────────────────────────────────────────────────────────

export function createDrawingPrimitive(drawing: DrawingData): DrawingPrimitiveBase | null {
  switch (drawing.type) {
    case 'trendline':  return new TrendlinePrimitive(drawing)
    case 'horizontal': return new HorizontalLinePrimitive(drawing)
    case 'fibonacci':  return new FibonacciRetracementPrimitive(drawing)
    case 'rectangle':  return new RectanglePrimitive(drawing)
    default:           return null
  }
}

// ── Channel / Wedge / Triangle fill primitive ─────────────────────────────────
// Renders a semi-transparent filled quadrilateral between two trendlines.
// Used for automatic pattern overlays (channels, wedges, triangles).
// Not interactive — no hit testing, no selection.

export class ChannelFillPrimitive implements ISeriesPrimitive<Time> {
  private _chart: IChartApiBase<Time> | null = null
  private _series: ISeriesApi<SeriesType, Time> | null = null
  private readonly _upper: readonly [string, number][]   // upper trendline [date, price] endpoints
  private readonly _lower: readonly [string, number][]   // lower trendline [date, price] endpoints
  private readonly _color: string
  private readonly _opacity: number

  constructor(
    upper: readonly [string, number][],
    lower: readonly [string, number][],
    color: string,
    opacity: number,
  ) {
    this._upper   = upper
    this._lower   = lower
    this._color   = color
    this._opacity = opacity
  }

  attached(param: SeriesAttachedParameter<Time, SeriesType>): void {
    this._chart  = param.chart
    this._series = param.series
  }

  detached(): void {
    this._chart  = null
    this._series = null
  }

  updateAllViews(): void {}

  paneViews(): readonly ISeriesPrimitivePaneView[] {
    const chart   = this._chart
    const series  = this._series
    const upper   = this._upper
    const lower   = this._lower
    const color   = this._color
    const opacity = this._opacity

    const renderer: ISeriesPrimitivePaneRenderer = {
      draw(target: CanvasRenderingTarget2D): void {
        target.useBitmapCoordinateSpace(({ context: ctx, horizontalPixelRatio: hpr, verticalPixelRatio: vpr }) => {
          if (!chart || !series) return

          const ts = chart.timeScale()
          const toXY = (date: string, price: number): [number, number] | null => {
            const x = ts.timeToCoordinate(date as Time) as number | null
            const y = series.priceToCoordinate(price) as number | null
            if (x == null || y == null) return null
            return [Math.round(x * hpr), Math.round(y * vpr)]
          }

          // Build polygon: upper-left → upper-right → lower-right → lower-left
          // upper[] and lower[] each have 2+ points (start, end of trendline)
          const ul = toXY(upper[0][0], upper[0][1])
          const ur = toXY(upper[upper.length - 1][0], upper[upper.length - 1][1])
          const lr = toXY(lower[lower.length - 1][0], lower[lower.length - 1][1])
          const ll = toXY(lower[0][0], lower[0][1])

          if (!ul || !ur || !lr || !ll) return

          ctx.save()
          ctx.beginPath()
          ctx.moveTo(ul[0], ul[1])
          ctx.lineTo(ur[0], ur[1])
          ctx.lineTo(lr[0], lr[1])
          ctx.lineTo(ll[0], ll[1])
          ctx.closePath()
          ctx.globalAlpha = opacity
          ctx.fillStyle   = color
          ctx.fill()
          ctx.restore()
        })
      },
    }

    return [{ zOrder(): 'bottom' { return 'bottom' }, renderer() { return renderer } }]
  }
}
