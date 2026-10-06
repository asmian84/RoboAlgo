// Shared pattern overlay registration for klinecharts v10
// Used by both PatternMockPage and the chart grid pattern view
import { registerOverlay } from 'klinecharts'

let overlaysRegistered = false

// In development, Vite HMR re-executes this module but klinecharts keeps its
// global overlay registry alive across hot reloads.  Re-registering the same
// overlay name is a no-op in klinecharts, so changes to createPointFigures
// never take effect until a hard reload.  We work around this by embedding
// all dynamic values (numPoints, color, etc.) in extendData rather than
// relying on closure-captured state that may be stale.

export function registerPatternOverlays() {
  if (overlaysRegistered) return
  overlaysRegistered = true

  // 1. Pattern trendline — solid/dashed line between two price points
  registerOverlay({
    name: 'patternTrendline',
    totalStep: 2,  // exactly 2 points — no ghost cursor point
    needDefaultPointFigure: false,
    needDefaultXAxisFigure: false,
    needDefaultYAxisFigure: false,
    createPointFigures: ({ coordinates, overlay }: any) => {
      if (coordinates.length < 2) return []
      const color = overlay.extendData?.color || '#f59e0b'
      const lineWidth = overlay.extendData?.lineWidth || 2
      const dash = overlay.extendData?.dash || []
      return [{
        type: 'line',
        attrs: { coordinates: [coordinates[0], coordinates[1]] },
        styles: { style: dash.length ? 'dashed' : 'solid', color, size: lineWidth, dashedValue: dash },
        ignoreEvent: true,
      }]
    },
  })

  // 2. Pattern fill zone — semi-transparent polygon between 4 corners
  registerOverlay({
    name: 'patternFillZone',
    totalStep: 5,
    needDefaultPointFigure: false,
    needDefaultXAxisFigure: false,
    needDefaultYAxisFigure: false,
    createPointFigures: ({ coordinates, overlay }: any) => {
      if (coordinates.length < 4) return []
      const color = overlay.extendData?.color || 'rgba(245,158,11,0.12)'
      return [{
        type: 'polygon',
        attrs: { coordinates: [coordinates[0], coordinates[1], coordinates[2], coordinates[3]] },
        styles: { style: 'fill', color },
        ignoreEvent: true,
      }]
    },
  })

  // 3. Pattern label — text annotation at a point with dot marker
  registerOverlay({
    name: 'patternLabel',
    totalStep: 2,
    needDefaultPointFigure: false,
    needDefaultXAxisFigure: false,
    needDefaultYAxisFigure: false,
    createPointFigures: ({ coordinates, overlay }: any) => {
      if (coordinates.length < 1) return []
      const text = overlay.extendData?.text || ''
      const color = overlay.extendData?.color || '#ffffff'
      const fontSize = overlay.extendData?.fontSize || 11
      const offsetY = overlay.extendData?.offsetY || -12
      return [
        {
          type: 'circle',
          attrs: { x: coordinates[0].x, y: coordinates[0].y, r: 3 },
          styles: { style: 'fill', color },
          ignoreEvent: true,
        },
        {
          type: 'text',
          attrs: { x: coordinates[0].x, y: coordinates[0].y + offsetY, text, align: 'center', baseline: 'bottom' },
          styles: { style: 'fill', color, size: fontSize, family: 'monospace', weight: 'bold', backgroundColor: 'transparent', borderSize: 0 },
          ignoreEvent: true,
        },
      ]
    },
  })

  // 4. Horizontal level — REMOVED (causes noisy full-width dashed lines)
  // 4. Horizontal level — dashed line across full width with label
  // Used by PatternMockPage; NOT used by real chart overlay (too noisy)
  registerOverlay({
    name: 'patternLevel',
    totalStep: 2,
    needDefaultPointFigure: false,
    needDefaultXAxisFigure: false,
    needDefaultYAxisFigure: false,
    createPointFigures: ({ coordinates, bounding, overlay }: any) => {
      if (coordinates.length < 1) return []
      const color = overlay.extendData?.color || '#22c55e'
      const label = overlay.extendData?.label || ''
      const y = coordinates[0].y
      return [
        {
          type: 'line',
          attrs: { coordinates: [{ x: 0, y }, { x: bounding.width, y }] },
          styles: { style: 'dashed', color, size: 1, dashedValue: [6, 4] },
          ignoreEvent: true,
        },
        {
          type: 'text',
          attrs: { x: bounding.width - 8, y: y - 4, text: label, align: 'right', baseline: 'bottom' },
          styles: { style: 'fill', color, size: 10, family: 'monospace', backgroundColor: 'transparent', borderSize: 0 },
          ignoreEvent: true,
        },
      ]
    },
  })

  // 5. Pattern name badge — bold text only, no background box
  registerOverlay({
    name: 'patternBadge',
    totalStep: 2,
    needDefaultPointFigure: false,
    needDefaultXAxisFigure: false,
    needDefaultYAxisFigure: false,
    createPointFigures: ({ coordinates, overlay }: any) => {
      if (coordinates.length < 1) return []
      const text = overlay.extendData?.text || ''
      const textColor = overlay.extendData?.textColor || '#f59e0b'
      const x = coordinates[0].x
      const y = coordinates[0].y
      return [
        {
          type: 'text',
          attrs: { x, y, text, align: 'left', baseline: 'middle' },
          styles: { style: 'fill', color: textColor, size: 11, family: 'system-ui, -apple-system, sans-serif', weight: '700', backgroundColor: 'transparent', borderSize: 0 },
          ignoreEvent: true,
        },
      ]
    },
  })

  // 6. Prediction projected path — dotted line with arrowhead
  //    Uses overlay.points.length to cap rendered coords so the klinecharts
  //    ghost cursor-tracking point (added when totalStep > placed points) is
  //    never drawn as part of the path.  totalStep stays at 6 so the API
  //    accepts up to 5 pre-specified waypoints without complaining.
  registerOverlay({
    name: 'predictionPath',
    totalStep: 6,
    needDefaultPointFigure: false,
    needDefaultXAxisFigure: false,
    needDefaultYAxisFigure: false,
    createPointFigures: ({ coordinates, overlay }: any) => {
      // numPoints is embedded in extendData by the caller — this is the ONLY
      // reliable count because klinecharts may also append the ghost cursor
      // position to overlay.points[] while in drawing mode, making that array
      // the same length as coordinates[].  extendData is never modified by
      // klinecharts so it always reflects what the caller actually provided.
      const nPts = typeof overlay.extendData?.numPoints === 'number'
        ? Math.min(overlay.extendData.numPoints, coordinates.length)
        : coordinates.length  // fallback for overlays that omit numPoints
      if (nPts < 2) return []
      const color = overlay.extendData?.color || '#a78bfa'
      const lineWidth = overlay.extendData?.lineWidth || 2
      const figures: any[] = []
      for (let i = 0; i < nPts - 1; i++) {
        figures.push({
          type: 'line',
          attrs: { coordinates: [coordinates[i], coordinates[i + 1]] },
          styles: { style: 'dashed', color, size: lineWidth, dashedValue: [3, 3] },
          ignoreEvent: true,
        })
      }
      // arrowhead at the last PLACED point — never at the ghost cursor coord
      const last = coordinates[nPts - 1]
      const prev = coordinates[nPts - 2]
      const angle = Math.atan2(last.y - prev.y, last.x - prev.x)
      const arrowSize = 8
      figures.push({
        type: 'polygon',
        attrs: {
          coordinates: [
            { x: last.x, y: last.y },
            { x: last.x - arrowSize * Math.cos(angle - 0.4), y: last.y - arrowSize * Math.sin(angle - 0.4) },
            { x: last.x - arrowSize * Math.cos(angle + 0.4), y: last.y - arrowSize * Math.sin(angle + 0.4) },
          ],
        },
        styles: { style: 'fill', color },
        ignoreEvent: true,
      })
      return figures
    },
  })

  // 7. Prediction zone — shaded rectangle for target/stop zone
  registerOverlay({
    name: 'predictionZone',
    totalStep: 2,  // exactly 2 points — no ghost cursor point
    needDefaultPointFigure: false,
    needDefaultXAxisFigure: false,
    needDefaultYAxisFigure: false,
    createPointFigures: ({ coordinates, overlay }: any) => {
      if (coordinates.length < 2) return []
      const color = overlay.extendData?.color || 'rgba(34,197,94,0.08)'
      const borderColor = overlay.extendData?.borderColor || 'rgba(34,197,94,0.3)'
      const label = overlay.extendData?.label || ''
      const labelColor = overlay.extendData?.labelColor || '#4ade80'
      const x1 = Math.min(coordinates[0].x, coordinates[1].x)
      const x2 = Math.max(coordinates[0].x, coordinates[1].x)
      const y1 = Math.min(coordinates[0].y, coordinates[1].y)
      const y2 = Math.max(coordinates[0].y, coordinates[1].y)
      return [
        {
          type: 'rect',
          attrs: { x: x1, y: y1, width: x2 - x1, height: y2 - y1 },
          styles: { style: 'fill', color },
          ignoreEvent: true,
        },
        {
          type: 'rect',
          attrs: { x: x1, y: y1, width: x2 - x1, height: y2 - y1 },
          styles: { style: 'stroke', color: borderColor, size: 1, dashedValue: [4, 3] },
          ignoreEvent: true,
        },
        ...(label ? [{
          type: 'text' as const,
          attrs: { x: x2 - 4, y: (y1 + y2) / 2, text: label, align: 'right' as const, baseline: 'middle' as const },
          styles: { style: 'fill' as const, color: labelColor, size: 10, family: 'monospace', weight: 'bold', backgroundColor: 'transparent', borderSize: 0 },
          ignoreEvent: true,
        }] : []),
      ]
    },
  })

  // 8. Wave marker — clean outlined circle with number, no heavy fill
  registerOverlay({
    name: 'waveCircle',
    totalStep: 2,
    needDefaultPointFigure: false,
    needDefaultXAxisFigure: false,
    needDefaultYAxisFigure: false,
    createPointFigures: ({ coordinates, overlay }: any) => {
      if (coordinates.length < 1) return []
      const text = overlay.extendData?.text || '1'
      const fillColor = overlay.extendData?.fillColor || '#22c55e'
      const textColor = overlay.extendData?.textColor || '#ffffff'
      const radius = overlay.extendData?.radius || 12
      const x = coordinates[0].x
      const y = coordinates[0].y
      return [
        // Subtle backdrop — dark translucent fill for contrast
        {
          type: 'circle',
          attrs: { x, y, r: radius },
          styles: { style: 'fill', color: 'rgba(8,12,24,0.70)' },
          ignoreEvent: true,
        },
        // Accent ring — the wave color as a clean outline
        {
          type: 'circle',
          attrs: { x, y, r: radius },
          styles: { style: 'stroke', color: fillColor, size: 1.8 },
          ignoreEvent: true,
        },
        // Number text — bold, readable
        {
          type: 'text',
          attrs: { x, y: y + 1, text, align: 'center', baseline: 'middle' },
          styles: { style: 'fill', color: textColor, size: radius > 10 ? 12 : 10, family: 'system-ui, -apple-system, sans-serif', weight: '700', backgroundColor: 'transparent', borderSize: 0 },
          ignoreEvent: true,
        },
      ]
    },
  })

  // 8b. Wave handle — draggable pivot marker for manual W1/W5 adjustments.
  // Rendered as a hollow diamond with a small grip dot. Unlike waveCircle,
  // its figures respond to events so klinecharts' built-in drag kicks in.
  registerOverlay({
    name: 'waveHandle',
    totalStep: 2,
    needDefaultPointFigure: false,
    needDefaultXAxisFigure: false,
    needDefaultYAxisFigure: false,
    createPointFigures: ({ coordinates, overlay }: any) => {
      if (coordinates.length < 1) return []
      const color = overlay.extendData?.color || '#fde047'
      const label = overlay.extendData?.label || ''
      const x = coordinates[0].x
      const y = coordinates[0].y
      const r = 14
      return [
        // Hit area — transparent but catches drag events
        {
          type: 'circle',
          attrs: { x, y, r: r + 4 },
          styles: { style: 'fill', color: 'rgba(255,255,255,0.001)' },
        },
        // Outer diamond ring (visual indicator)
        {
          type: 'polygon',
          attrs: { coordinates: [
            { x, y: y - r },
            { x: x + r, y },
            { x, y: y + r },
            { x: x - r, y },
          ] },
          styles: { style: 'stroke', color, size: 2.5 },
          ignoreEvent: true,
        },
        // Inner filled dot — grip
        {
          type: 'circle',
          attrs: { x, y, r: 4 },
          styles: { style: 'fill', color },
          ignoreEvent: true,
        },
        // Small label ("W1" / "W5" etc)
        ...(label ? [{
          type: 'text',
          attrs: { x: x + r + 4, y: y + 1, text: label, align: 'left', baseline: 'middle' },
          styles: { style: 'fill', color, size: 10, family: 'system-ui, -apple-system, sans-serif', weight: '700', backgroundColor: 'transparent', borderSize: 0 },
          ignoreEvent: true,
        }] : []),
      ]
    },
  })

  // 9. Wave phase label — "Impulsive Wave" / "Corrective Wave" with dark bg
  registerOverlay({
    name: 'wavePhaseLabel',
    totalStep: 2,
    needDefaultPointFigure: false,
    needDefaultXAxisFigure: false,
    needDefaultYAxisFigure: false,
    createPointFigures: ({ coordinates, overlay }: any) => {
      if (coordinates.length < 1) return []
      const text = overlay.extendData?.text || ''
      const subText = overlay.extendData?.subText || ''
      const color = overlay.extendData?.color || '#22c55e'
      const x = coordinates[0].x
      const y = coordinates[0].y
      // Background pill dimensions
      const mainW = text.length * 9 + 24
      const totalH = subText ? 42 : 24
      const figures: any[] = [
        // Dark background pill
        {
          type: 'rect',
          attrs: { x: x - mainW / 2, y: y - totalH / 2, width: mainW, height: totalH },
          styles: { style: 'fill', color: 'rgba(10,15,30,0.80)', borderRadius: 6 },
          ignoreEvent: true,
        },
        // Accent border
        {
          type: 'rect',
          attrs: { x: x - mainW / 2, y: y - totalH / 2, width: mainW, height: totalH },
          styles: { style: 'stroke', color: color + '50', size: 1, borderRadius: 6 },
          ignoreEvent: true,
        },
        // Main label
        {
          type: 'text',
          attrs: { x, y: subText ? y - 8 : y, text, align: 'center', baseline: 'middle' },
          styles: { style: 'fill', color, size: 13, family: 'system-ui, -apple-system, sans-serif', weight: '700', backgroundColor: 'transparent', borderSize: 0 },
          ignoreEvent: true,
        },
      ]
      if (subText) {
        figures.push({
          type: 'text',
          attrs: { x, y: y + 10, text: subText, align: 'center', baseline: 'middle' },
          styles: { style: 'fill', color: 'rgba(200,210,230,0.7)', size: 10, family: 'system-ui, -apple-system, sans-serif', weight: '500', backgroundColor: 'transparent', borderSize: 0 },
          ignoreEvent: true,
        })
      }
      return figures
    },
  })

  // 10. Probability cone — expanding triangle
  registerOverlay({
    name: 'predictionCone',
    totalStep: 3,  // exactly 3 points (origin, hi-bound, lo-bound) — no ghost
    needDefaultPointFigure: false,
    needDefaultXAxisFigure: false,
    needDefaultYAxisFigure: false,
    createPointFigures: ({ coordinates, overlay }: any) => {
      if (coordinates.length < 3) return []
      const color = overlay.extendData?.color || 'rgba(167,139,250,0.06)'
      const borderColor = overlay.extendData?.borderColor || 'rgba(167,139,250,0.25)'
      return [
        {
          type: 'polygon',
          attrs: { coordinates: [coordinates[0], coordinates[1], coordinates[2]] },
          styles: { style: 'fill', color },
          ignoreEvent: true,
        },
        {
          type: 'line',
          attrs: { coordinates: [coordinates[0], coordinates[1]] },
          styles: { style: 'dashed', color: borderColor, size: 1, dashedValue: [2, 2] },
          ignoreEvent: true,
        },
        {
          type: 'line',
          attrs: { coordinates: [coordinates[0], coordinates[2]] },
          styles: { style: 'dashed', color: borderColor, size: 1, dashedValue: [2, 2] },
          ignoreEvent: true,
        },
      ]
    },
  })
}

// ── Direction-based color scheme ──────────────────────────────────────────
export const PATTERN_COLORS = {
  bullish: { line: '#22c55e', fill: 'rgba(34,197,94,0.12)', badge: 'rgba(34,197,94,0.25)', text: '#4ade80', prediction: '#a78bfa' },
  bearish: { line: '#ef4444', fill: 'rgba(239,68,68,0.12)', badge: 'rgba(239,68,68,0.25)', text: '#f87171', prediction: '#f472b6' },
  neutral: { line: '#f59e0b', fill: 'rgba(245,158,11,0.12)', badge: 'rgba(245,158,11,0.25)', text: '#fbbf24', prediction: '#818cf8' },
}
