/**
 * PatternThumbnail — tiny educational SVG diagrams shown in the pattern overlay selector.
 * Each diagram shows the canonical idealised shape of a chart / harmonic / Wyckoff pattern.
 * ViewBox is 60×38; rendered at `width × height` pixels (default 48×30).
 */
import React from 'react'

interface Props {
  patternName: string
  color: string
  width?: number
  height?: number
}

const PatternThumbnail: React.FC<Props> = ({
  patternName,
  color,
  width = 48,
  height = 30,
}) => (
  <svg
    viewBox="0 0 60 38"
    width={width}
    height={height}
    style={{ display: 'block', flexShrink: 0 }}
    xmlns="http://www.w3.org/2000/svg"
  >
    {renderShape(patternName.toLowerCase(), color)}
  </svg>
)

// ── Shared style helpers ──────────────────────────────────────────────────────

const priceLine = (d: string, extra?: React.SVGProps<SVGPathElement>) => (
  <path d={d} stroke="rgba(220,220,220,0.75)" strokeWidth={1.5} fill="none" {...extra} />
)
const levelLine = (x1: number, y1: number, x2: number, y2: number, c: string, op = 1) => (
  <line x1={x1} y1={y1} x2={x2} y2={y2}
    stroke={c} strokeWidth={0.9} strokeDasharray="2.5,1.5" strokeOpacity={op} />
)
const thinLine = (x1: number, y1: number, x2: number, y2: number, c: string, op = 0.5) => (
  <line x1={x1} y1={y1} x2={x2} y2={y2}
    stroke={c} strokeWidth={0.75} strokeDasharray="2,1.5" strokeOpacity={op} />
)
const solidLine = (x1: number, y1: number, x2: number, y2: number, c: string, w = 1.5) => (
  <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={c} strokeWidth={w} />
)
const fillPoly = (pts: string, c: string, op = 0.08) => (
  <polygon points={pts} fill={c} fillOpacity={op} stroke="none" />
)
const label = (x: number, y: number, t: string, c: string, size = 5) => (
  <text x={x} y={y} fontSize={size} fill={c} fillOpacity={0.85}
    fontFamily="monospace" textAnchor="middle">{t}</text>
)

// ── Shape renderer ────────────────────────────────────────────────────────────

function renderShape(n: string, c: string): React.ReactNode {

  // ── Head & Shoulders (bearish) ─────────────────────────────────────────────
  if (n.includes('head') && n.includes('shoulder') && !n.includes('inv')) {
    return <>
      {priceLine('M4,34 L13,21 L19,27 L30,7 L41,25 L47,19 L54,34')}
      {levelLine(19, 25, 41, 25, c)}
      {levelLine(41, 25, 54, 25, c, 0.4)}
      {label(13, 19, 'LS', c, 4)}{label(30, 5, 'H', c, 4)}{label(47, 17, 'RS', c, 4)}
    </>
  }

  // ── Inv. Head & Shoulders (bullish) ───────────────────────────────────────
  if (n.includes('inv') && n.includes('shoulder')) {
    return <>
      {priceLine('M4,4 L13,17 L19,11 L30,31 L41,13 L47,19 L54,4')}
      {levelLine(19, 13, 41, 13, c)}
      {levelLine(41, 13, 54, 13, c, 0.4)}
      {label(13, 27, 'LS', c, 4)}{label(30, 37, 'H', c, 4)}{label(47, 27, 'RS', c, 4)}
    </>
  }

  // ── Triple Top ─────────────────────────────────────────────────────────────
  if (n.includes('triple top')) {
    return <>
      {priceLine('M4,34 L12,8 L18,24 L26,8 L32,24 L40,8 L54,34')}
      {levelLine(4, 24, 54, 24, c)}
      {label(12, 6, 'P1', c, 4)}{label(26, 6, 'P2', c, 4)}{label(40, 6, 'P3', c, 4)}
    </>
  }

  // ── Triple Bottom ──────────────────────────────────────────────────────────
  if (n.includes('triple bottom')) {
    return <>
      {priceLine('M4,4 L12,30 L18,14 L26,30 L32,14 L40,30 L54,4')}
      {levelLine(4, 14, 54, 14, c)}
      {label(12, 37, 'B1', c, 4)}{label(26, 37, 'B2', c, 4)}{label(40, 37, 'B3', c, 4)}
    </>
  }

  // ── Double Top ─────────────────────────────────────────────────────────────
  if (n.includes('double top')) {
    return <>
      {priceLine('M4,34 L15,8 L26,24 L38,8 L52,34')}
      {levelLine(4, 24, 52, 24, c)}
      {label(15, 6, 'P1', c, 4)}{label(38, 6, 'P2', c, 4)}
    </>
  }

  // ── Double Bottom ──────────────────────────────────────────────────────────
  if (n.includes('double bottom')) {
    return <>
      {priceLine('M4,4 L15,30 L28,14 L41,30 L54,4')}
      {levelLine(4, 14, 54, 14, c)}
      {label(15, 37, 'B1', c, 4)}{label(41, 37, 'B2', c, 4)}
    </>
  }

  // ── Cup & Handle ───────────────────────────────────────────────────────────
  if (n.includes('cup')) {
    return <>
      {priceLine('M4,10 C14,36 44,36 50,10 L52,16 L56,8')}
      {levelLine(4, 10, 50, 10, c)}
      {label(8, 8, 'Rim', c, 4)}{label(28, 37, 'Base', c, 4)}{label(52, 6, 'BO', c, 4)}
    </>
  }

  // ── Rising Wedge ───────────────────────────────────────────────────────────
  if (n.includes('rising wedge')) {
    return <>
      {fillPoly('4,21 54,6 54,10 4,31', c, 0.07)}
      {solidLine(4, 21, 54, 6, 'rgba(248,113,113,0.8)', 1.5)}
      {solidLine(4, 31, 54, 10, 'rgba(74,222,128,0.7)', 1.2)}
      {priceLine('M4,31 L14,27 L22,25 L32,20 L40,17 L50,13', { strokeWidth: 1, stroke: 'rgba(220,220,220,0.5)' })}
      {label(30, 2, '↗ WEDGE', c, 4)}
    </>
  }

  // ── Falling Wedge ─────────────────────────────────────────────────────────
  if (n.includes('falling wedge')) {
    return <>
      {fillPoly('4,7 54,22 54,30 4,15', c, 0.07)}
      {solidLine(4, 7, 54, 22, 'rgba(248,113,113,0.8)', 1.5)}
      {solidLine(4, 15, 54, 30, 'rgba(74,222,128,0.7)', 1.2)}
      {priceLine('M4,15 L14,19 L22,21 L32,24 L40,27 L50,29', { strokeWidth: 1, stroke: 'rgba(220,220,220,0.5)' })}
      {label(30, 37, '↘ WEDGE', c, 4)}
    </>
  }

  // ── Rectangle / Trading Range ─────────────────────────────────────────────
  if (n.includes('rectangle')) {
    return <>
      {fillPoly('4,10 54,10 54,28 4,28', c, 0.06)}
      {levelLine(4, 10, 54, 10, c)}
      {levelLine(4, 28, 54, 28, c)}
      {priceLine('M4,20 L14,10 L24,28 L34,10 L44,28 L54,18', { strokeWidth: 1.2 })}
    </>
  }

  // ── Ascending Triangle ────────────────────────────────────────────────────
  if (n.includes('ascend') && n.includes('triangle')) {
    return <>
      {fillPoly('4,8 54,8 4,32', c, 0.06)}
      {solidLine(4, 8, 54, 8, c, 1.2)}
      {solidLine(4, 32, 54, 8, 'rgba(220,220,220,0.6)', 1.2)}
      {priceLine('M4,26 L14,8 L22,22 L30,8 L38,16 L46,8 L54,10', { strokeWidth: 1 })}
    </>
  }

  // ── Descending Triangle ───────────────────────────────────────────────────
  if (n.includes('descend') && n.includes('triangle')) {
    return <>
      {fillPoly('4,10 4,30 54,30', c, 0.06)}
      {solidLine(4, 30, 54, 30, c, 1.2)}
      {solidLine(4, 10, 54, 30, 'rgba(220,220,220,0.6)', 1.2)}
      {priceLine('M4,14 L14,30 L22,18 L30,30 L38,24 L46,30 L54,28', { strokeWidth: 1 })}
    </>
  }

  // ── Symmetrical Triangle ──────────────────────────────────────────────────
  if (n.includes('sym') && n.includes('triangle')) {
    return <>
      {fillPoly('4,6 54,22 4,32', c, 0.06)}
      {solidLine(4, 6, 54, 22, 'rgba(248,113,113,0.7)', 1.2)}
      {solidLine(4, 32, 54, 22, 'rgba(74,222,128,0.7)', 1.2)}
      {priceLine('M4,10 L16,26 L26,14 L36,22 L44,17 L52,20', { strokeWidth: 1 })}
    </>
  }

  // ── Bull Flag ─────────────────────────────────────────────────────────────
  if ((n.includes('bull') || n.includes('bull')) && n.includes('flag') && !n.includes('bear')) {
    return <>
      {solidLine(4, 30, 20, 8, 'rgba(220,220,220,0.8)', 2)}
      {solidLine(20, 8, 42, 14, c, 0.9)}{solidLine(20, 14, 42, 20, c, 0.9)}
      {solidLine(42, 11, 54, 3, 'rgba(220,220,220,0.8)', 2)}
      {label(12, 36, 'Pole', 'rgba(220,220,220,0.6)', 4)}
      {label(31, 25, 'Flag', c, 4)}
    </>
  }

  // ── Bear Flag ─────────────────────────────────────────────────────────────
  if (n.includes('bear') && n.includes('flag')) {
    return <>
      {solidLine(4, 8, 20, 30, 'rgba(220,220,220,0.8)', 2)}
      {solidLine(20, 30, 42, 24, c, 0.9)}{solidLine(20, 34, 42, 28, c, 0.9)}
      {solidLine(42, 27, 54, 35, 'rgba(220,220,220,0.8)', 2)}
      {label(12, 6, 'Pole', 'rgba(220,220,220,0.6)', 4)}
      {label(31, 20, 'Flag', c, 4)}
    </>
  }

  // ── Pennant ────────────────────────────────────────────────────────────────
  if (n.includes('pennant')) {
    return <>
      {solidLine(4, 30, 18, 8, 'rgba(220,220,220,0.8)', 2)}
      {solidLine(18, 8, 40, 19, c, 0.9)}{solidLine(18, 14, 40, 19, c, 0.9)}
      {solidLine(40, 17, 54, 4, 'rgba(220,220,220,0.8)', 2)}
    </>
  }

  // ── Wyckoff Accumulation ───────────────────────────────────────────────────
  if ((n.includes('wyckoff') || n.includes('accum')) && !n.includes('dist')) {
    return <>
      {thinLine(4, 10, 38, 10, c, 0.7)}
      {thinLine(4, 27, 38, 27, c, 0.7)}
      {priceLine('M4,20 L8,27 L10,31 L12,20 L16,10 L20,23 L22,31 L24,19 L28,13 L30,19 L34,10 L40,4 L52,2')}
      {label(10, 37, 'SC', c, 4)}{label(16, 8, 'AR', c, 4)}
      {label(22, 37, 'SP', c, 4)}{label(28, 11, 'SOS', c, 4)}
    </>
  }

  // ── Wyckoff Distribution ──────────────────────────────────────────────────
  if (n.includes('dist') || (n.includes('wyckoff') && n.includes('dist'))) {
    return <>
      {thinLine(4, 12, 38, 12, c, 0.7)}
      {thinLine(4, 28, 38, 28, c, 0.7)}
      {priceLine('M4,18 L8,12 L10,8 L12,18 L16,28 L20,16 L22,10 L24,20 L28,28 L30,22 L34,28 L40,32 L52,36')}
      {label(10, 6, 'BC', c, 4)}{label(16, 36, 'AR', c, 4)}
      {label(22, 8, 'UT', c, 4)}{label(28, 36, 'SOW', c, 4)}
    </>
  }

  // ── Harmonic: Crab ────────────────────────────────────────────────────────
  if (n.includes('crab')) {
    // Crab: D extends furthest beyond X (XD_XA=1.618)
    const bull = !n.includes('bear')
    return bull
      ? <>
          {priceLine('M4,20 L16,6 L26,16 L38,10 L54,32')}
          {thinLine(4, 20, 26, 16, c, 0.6)}{thinLine(16, 6, 38, 10, c, 0.5)}
          {label(4, 18, 'X', c, 4)}{label(16, 4, 'A', c, 4)}{label(26, 14, 'B', c, 4)}
          {label(38, 8, 'C', c, 4)}{label(54, 30, 'D', c, 4)}
          {label(30, 36, '1.618', c, 4)}
        </>
      : <>
          {priceLine('M4,18 L16,32 L26,22 L38,28 L54,6')}
          {thinLine(4, 18, 26, 22, c, 0.6)}{thinLine(16, 32, 38, 28, c, 0.5)}
          {label(4, 16, 'X', c, 4)}{label(16, 37, 'A', c, 4)}{label(26, 20, 'B', c, 4)}
          {label(38, 26, 'C', c, 4)}{label(54, 4, 'D', c, 4)}
          {label(30, 4, '1.618', c, 4)}
        </>
  }

  // ── Harmonic: Bat ─────────────────────────────────────────────────────────
  if (n.includes('bat')) {
    // Bat: D retraces 0.886 of XA (deep but within X)
    const bull = !n.includes('bear')
    return bull
      ? <>
          {priceLine('M4,30 L16,6 L26,20 L38,10 L52,27')}
          {thinLine(4, 30, 26, 20, c, 0.6)}{thinLine(16, 6, 38, 10, c, 0.5)}
          {label(4, 37, 'X', c, 4)}{label(16, 4, 'A', c, 4)}{label(26, 26, 'B', c, 4)}
          {label(38, 8, 'C', c, 4)}{label(52, 35, 'D', c, 4)}
          {label(30, 36, '0.886', c, 4)}
        </>
      : <>
          {priceLine('M4,8 L16,32 L26,18 L38,28 L52,11')}
          {thinLine(4, 8, 26, 18, c, 0.6)}{thinLine(16, 32, 38, 28, c, 0.5)}
          {label(4, 6, 'X', c, 4)}{label(16, 37, 'A', c, 4)}{label(26, 16, 'B', c, 4)}
          {label(38, 26, 'C', c, 4)}{label(52, 9, 'D', c, 4)}
          {label(30, 4, '0.886', c, 4)}
        </>
  }

  // ── Harmonic: Butterfly ───────────────────────────────────────────────────
  if (n.includes('butterfly')) {
    // Butterfly: D extends beyond X (XD_XA=1.27-1.618)
    const bull = !n.includes('bear')
    return bull
      ? <>
          {priceLine('M4,16 L16,6 L26,16 L38,10 L54,28')}
          {thinLine(4, 16, 26, 16, c, 0.6)}{thinLine(16, 6, 38, 10, c, 0.5)}
          {label(4, 22, 'X', c, 4)}{label(16, 4, 'A', c, 4)}{label(26, 22, 'B', c, 4)}
          {label(38, 8, 'C', c, 4)}{label(54, 34, 'D', c, 4)}
          {label(30, 36, '1.27', c, 4)}
        </>
      : <>
          {priceLine('M4,22 L16,32 L26,22 L38,28 L54,10')}
          {thinLine(4, 22, 26, 22, c, 0.6)}{thinLine(16, 32, 38, 28, c, 0.5)}
          {label(4, 20, 'X', c, 4)}{label(16, 37, 'A', c, 4)}{label(26, 20, 'B', c, 4)}
          {label(38, 26, 'C', c, 4)}{label(54, 8, 'D', c, 4)}
          {label(30, 6, '1.27', c, 4)}
        </>
  }

  // ── Harmonic: Gartley ─────────────────────────────────────────────────────
  if (n.includes('gartley')) {
    // Gartley: D at 0.786 of XA (within range, near X)
    const bull = !n.includes('bear')
    return bull
      ? <>
          {priceLine('M4,28 L16,6 L26,18 L38,10 L52,24')}
          {thinLine(4, 28, 26, 18, c, 0.6)}{thinLine(16, 6, 38, 10, c, 0.5)}
          {label(4, 36, 'X', c, 4)}{label(16, 4, 'A', c, 4)}{label(26, 24, 'B', c, 4)}
          {label(38, 8, 'C', c, 4)}{label(52, 32, 'D', c, 4)}
          {label(30, 36, '0.786', c, 4)}
        </>
      : <>
          {priceLine('M4,10 L16,32 L26,20 L38,28 L52,14')}
          {thinLine(4, 10, 26, 20, c, 0.6)}{thinLine(16, 32, 38, 28, c, 0.5)}
          {label(4, 8, 'X', c, 4)}{label(16, 37, 'A', c, 4)}{label(26, 18, 'B', c, 4)}
          {label(38, 26, 'C', c, 4)}{label(52, 12, 'D', c, 4)}
          {label(30, 4, '0.786', c, 4)}
        </>
  }

  // ── Harmonic: Cypher ──────────────────────────────────────────────────────
  if (n.includes('cypher')) {
    const bull = !n.includes('bear')
    return bull
      ? <>
          {priceLine('M4,28 L18,8 L30,22 L46,4 L54,22')}
          {thinLine(4, 28, 30, 22, c, 0.6)}{thinLine(18, 8, 46, 4, c, 0.5)}
          {label(4, 36, 'X', c, 4)}{label(18, 6, 'A', c, 4)}{label(30, 28, 'B', c, 4)}
          {label(46, 2, 'C', c, 4)}{label(54, 28, 'D', c, 4)}
        </>
      : <>
          {priceLine('M4,10 L18,30 L30,16 L46,34 L54,16')}
          {thinLine(4, 10, 30, 16, c, 0.6)}{thinLine(18, 30, 46, 34, c, 0.5)}
          {label(4, 8, 'X', c, 4)}{label(18, 37, 'A', c, 4)}{label(30, 14, 'B', c, 4)}
          {label(46, 37, 'C', c, 4)}{label(54, 14, 'D', c, 4)}
        </>
  }

  // ── Harmonic: generic XABCD fallback ─────────────────────────────────────
  if (['gartley','bat','butterfly','crab','shark','ab=cd','harmonic'].some(h => n.includes(h))) {
    return <>
      {priceLine('M4,30 L16,8 L28,22 L40,8 L52,26')}
      {thinLine(4, 30, 28, 22, c, 0.55)}{thinLine(16, 8, 40, 8, c, 0.4)}
      {label(4, 37, 'X', c, 4)}{label(16, 6, 'A', c, 4)}{label(28, 28, 'B', c, 4)}
      {label(40, 6, 'C', c, 4)}{label(52, 33, 'D', c, 4)}
    </>
  }

  // ── Gann (fan lines from a pivot) ─────────────────────────────────────────
  if (n.includes('gann')) {
    return <>
      <circle cx="8" cy="32" r="2" fill={c} fillOpacity={0.8} />
      {solidLine(8, 32, 54, 4, c, 1.2)}
      {solidLine(8, 32, 54, 10, c, 0.85)}
      {solidLine(8, 32, 54, 18, c, 0.6)}
      {solidLine(8, 32, 54, 26, c, 0.35)}
      {label(30, 2, '1×1', c, 4)}{label(30, 20, '2×1', c, 4)}
    </>
  }

  // ── Megaphone / Broadening ─────────────────────────────────────────────────
  if (n.includes('megaphone') || n.includes('broaden')) {
    return <>
      {fillPoly('4,16 54,4 54,34 4,22', c, 0.06)}
      {solidLine(4, 16, 54, 4, 'rgba(248,113,113,0.7)', 1.2)}
      {solidLine(4, 22, 54, 34, 'rgba(74,222,128,0.7)', 1.2)}
      {priceLine('M4,19 L16,6 L26,32 L36,4 L46,34 L54,16', { strokeWidth: 1.2 })}
    </>
  }

  // ── Channel (ascending) ───────────────────────────────────────────────────
  if (n.includes('channel') && (n.includes('ascend') || n.includes('bull'))) {
    return <>
      {fillPoly('4,28 54,8 54,14 4,34', c, 0.06)}
      {solidLine(4, 28, 54, 8, c, 1.2)}
      {solidLine(4, 34, 54, 14, c, 0.7)}
      {priceLine('M4,34 L18,28 L22,22 L36,16 L40,10 L54,8', { strokeWidth: 1 })}
    </>
  }

  // ── Channel (descending) ──────────────────────────────────────────────────
  if (n.includes('channel') && (n.includes('descend') || n.includes('bear'))) {
    return <>
      {fillPoly('4,10 54,28 54,34 4,16', c, 0.06)}
      {solidLine(4, 10, 54, 28, c, 1.2)}
      {solidLine(4, 16, 54, 34, c, 0.7)}
      {priceLine('M4,16 L18,22 L22,28 L36,32 L40,28 L54,34', { strokeWidth: 1 })}
    </>
  }

  // ── Compression / Squeeze ─────────────────────────────────────────────────
  if (n.includes('compress') || n.includes('squeeze')) {
    return <>
      {solidLine(4, 10, 54, 18, 'rgba(248,113,113,0.7)', 1.2)}
      {solidLine(4, 28, 54, 20, 'rgba(74,222,128,0.7)', 1.2)}
      {priceLine('M4,20 L12,14 L20,24 L28,16 L36,22 L44,19 L52,21', { strokeWidth: 1 })}
      {label(30, 36, '⚡ squeeze', c, 4)}
    </>
  }

  // ── Chair Pattern ─────────────────────────────────────────────────────────
  if (n.includes('chair')) {
    return <>
      {priceLine('M4,28 L14,14 L22,22 L32,10 L42,16 L54,6')}
      {levelLine(22, 22, 54, 22, c, 0.5)}
      {label(14, 12, 'Impulse', c, 4)}{label(32, 8, 'Recovery', c, 4)}
    </>
  }

  // ── Rounding Bottom / Saucer ──────────────────────────────────────────────
  if (n.includes('round') || n.includes('saucer')) {
    return <>
      {priceLine('M4,8 C18,32 42,32 54,8')}
      {levelLine(4, 8, 54, 8, c)}
    </>
  }

  // ── Default: generic breakout ──────────────────────────────────────────────
  return <>
    {priceLine('M4,28 L18,24 L26,18 L34,20 L42,12 L54,6')}
    {levelLine(4, 20, 54, 20, c)}
  </>
}

export default PatternThumbnail
