// AllPatternsMockPage — Complete pattern library visual reference
// 31 patterns across 5 categories:
//   EW Motive · EW Corrective · EW Triangle · EW Complex · Classical
import { useEffect, useRef, useState } from 'react'
import { init, dispose, type Chart } from 'klinecharts'
import { registerPatternOverlays } from '../components/chart/patternOverlays'

// ── Data builder ──────────────────────────────────────────────────────────────
function buildBars(waypoints: [number, number][], noise = 0.7): any[] {
  const base = new Date('2025-01-02').getTime()
  const day  = 86400000
  const closes: number[] = []
  for (let s = 0; s < waypoints.length - 1; s++) {
    const [i0, p0] = waypoints[s], [i1, p1] = waypoints[s + 1]
    for (let i = i0; i < i1; i++) {
      const t = (i - i0) / (i1 - i0)
      closes.push(p0 + (p1 - p0) * t)
    }
  }
  closes.push(waypoints[waypoints.length - 1][1])
  return closes.map((c, i) => {
    const prev = i > 0 ? closes[i - 1] : c
    const n = (Math.random() - 0.5) * noise
    const o = prev + (Math.random() - 0.5) * noise * 0.4
    return {
      timestamp: base + i * day,
      open:  +o.toFixed(2),
      high:  +(Math.max(o, c) + Math.abs(n) + 0.25).toFixed(2),
      low:   +(Math.min(o, c) - Math.abs(n) - 0.2).toFixed(2),
      close: +(c + n * 0.15).toFixed(2),
      volume: 2_000_000 + Math.random() * 3_000_000,
    }
  })
}

function ts(d: any[], i: number) { return d[Math.min(i, d.length - 1)]?.timestamp ?? 0 }

// ── Chart styles ──────────────────────────────────────────────────────────────
const STYLES = {
  grid: {
    show: true,
    horizontal: { show: true, size: 1, color: 'rgba(255,255,255,0.04)', style: 'dashed' as const, dashedValue: [4,4] },
    vertical:   { show: true, size: 1, color: 'rgba(255,255,255,0.04)', style: 'dashed' as const, dashedValue: [4,4] },
  },
  candle: {
    type: 'candle_solid' as const,
    bar: {
      upColor: '#22c55e', downColor: '#ef4444', noChangeColor: '#6b7280',
      upBorderColor: '#22c55e', downBorderColor: '#ef4444', noChangeBorderColor: '#6b7280',
      upWickColor:   '#22c55e', downWickColor:   '#ef4444', noChangeWickColor:   '#6b7280',
    },
    priceMark: {
      show: true,
      high: { show: true, color: '#22c55e', textOffset: 5, textSize: 10, textFamily: 'monospace', textWeight: 'normal' },
      low:  { show: true, color: '#ef4444', textOffset: 5, textSize: 10, textFamily: 'monospace', textWeight: 'normal' },
      last: { show: true, upColor: '#22c55e', downColor: '#ef4444', noChangeColor: '#6b7280',
        line: { show: true, style: 'dashed' as const, dashedValue: [4,4], size: 1 },
        text: { show: true, style: 'fill' as const, color: '#fff', size: 10, family: 'monospace', weight: 'bold',
          paddingLeft: 4, paddingRight: 4, paddingTop: 2, paddingBottom: 2,
          borderStyle: 'solid' as const, borderSize: 0, borderColor: 'transparent',
          borderDashedValue: [] as number[], borderRadius: 2 },
      },
    },
    tooltip: { showRule: 'always' as const, showType: 'standard' as const },
  },
  xAxis: { show: true, tickText: { color: '#9ca3af', size: 10, family: 'monospace', weight: 'normal' } },
  yAxis: { show: true, tickText: { color: '#9ca3af', size: 10, family: 'monospace', weight: 'normal' } },
  crosshair: {
    show: true,
    horizontal: { show: true, line: { show: true, style: 'dashed' as const, dashedValue: [4,4], size: 1, color: 'rgba(255,255,255,0.2)' },
      text: { show: true, style: 'fill' as const, color: '#fff', size: 10, family: 'monospace', weight: 'normal',
        paddingLeft: 4, paddingRight: 4, paddingTop: 2, paddingBottom: 2,
        borderStyle: 'solid' as const, borderSize: 1, borderColor: '#374151',
        borderDashedValue: [] as number[], borderRadius: 2, backgroundColor: '#1f2937' } },
    vertical: { show: true, line: { show: true, style: 'dashed' as const, dashedValue: [4,4], size: 1, color: 'rgba(255,255,255,0.2)' },
      text: { show: true, style: 'fill' as const, color: '#fff', size: 10, family: 'monospace', weight: 'normal',
        paddingLeft: 4, paddingRight: 4, paddingTop: 2, paddingBottom: 2,
        borderStyle: 'solid' as const, borderSize: 1, borderColor: '#374151',
        borderDashedValue: [] as number[], borderRadius: 2, backgroundColor: '#1f2937' } },
  },
}

// ── Colors ────────────────────────────────────────────────────────────────────
const IMP  = '#22c55e'   // impulse (motive)
const COR  = '#f59e0b'   // corrective
const DIAG = '#818cf8'   // diagonal / X-wave
const EXT  = '#34d399'   // extended sub-wave
const BDY  = 'rgba(148,163,184,0.45)'  // boundary trendline
const NECK = '#60a5fa'   // neckline (classical)
const BULL = '#4ade80'   // prediction bullish
const BEAR = '#f87171'   // prediction bearish

// ── Drawing helpers ───────────────────────────────────────────────────────────
const seg = (c: Chart, d: any[], i0: number, p0: number, i1: number, p1: number,
             color = IMP, lw = 2.5, dash: number[] = []) =>
  c.createOverlay({ name: 'patternTrendline',
    points: [{ timestamp: ts(d,i0), value: p0 }, { timestamp: ts(d,i1), value: p1 }],
    extendData: { color, lineWidth: lw, dash } })

const circle = (c: Chart, d: any[], i: number, p: number, label: string, color = IMP,
                pct = 0.025, above = true) =>
  c.createOverlay({ name: 'waveCircle',
    points: [{ timestamp: ts(d,i), value: p + p * pct * (above ? 1 : -1) }],
    extendData: { text: label, fillColor: color, textColor: '#fff', radius: 11 } })

const lbl = (c: Chart, d: any[], i: number, p: number, text: string, color = '#94a3b8',
             offsetY = -12, fontSize = 9) =>
  c.createOverlay({ name: 'patternLabel',
    points: [{ timestamp: ts(d,i), value: p }],
    extendData: { text, color, offsetY, fontSize } })

const lvl = (c: Chart, d: any[], p: number, label: string, color = '#22c55e') =>
  c.createOverlay({ name: 'patternLevel',
    points: [{ timestamp: ts(d,0), value: p }],
    extendData: { color, label } })

const bdg = (c: Chart, d: any[], i: number, p: number, text: string, color = IMP) =>
  c.createOverlay({ name: 'patternBadge',
    points: [{ timestamp: ts(d,i), value: p }],
    extendData: { text, textColor: color } })

const path = (c: Chart, d: any[], pts: [number,number][], color = BULL, lw = 2) =>
  c.createOverlay({ name: 'predictionPath',
    points: pts.map(([i,p]) => ({ timestamp: ts(d,i), value: p })),
    extendData: { color, lineWidth: lw, numPoints: pts.length } })

const cone = (c: Chart, d: any[], orig: [number,number], hi: [number,number], lo: [number,number],
              color = 'rgba(74,222,128,0.06)', border = 'rgba(74,222,128,0.2)') =>
  c.createOverlay({ name: 'predictionCone',
    points: [orig, hi, lo].map(([i,p]) => ({ timestamp: ts(d,i), value: p })),
    extendData: { color, borderColor: border } })

const zone = (c: Chart, d: any[], i0: number, p0: number, i1: number, p1: number,
              color: string, border: string, label: string, labelColor: string) =>
  c.createOverlay({ name: 'predictionZone',
    points: [{ timestamp: ts(d,i0), value: p0 }, { timestamp: ts(d,i1), value: p1 }],
    extendData: { color, borderColor: border, label, labelColor } })

// ── Pattern type ──────────────────────────────────────────────────────────────
type Category = 'EW Motive' | 'EW Corrective' | 'EW Triangle' | 'EW Complex' | 'Classical'

interface Pattern {
  name: string
  sub: string
  category: Category
  direction: 'bullish' | 'bearish' | 'neutral'
  buildData: () => any[]
  draw: (c: Chart, d: any[]) => void
}

// ─────────────────────────────────────────────────────────────────────────────
// ██  PATTERN LIBRARY  ████████████████████████████████████████████████████████
// ─────────────────────────────────────────────────────────────────────────────
const PATTERNS: Pattern[] = [

  // ══════════════════════════════════════════════════════════════════════════
  //  EW MOTIVE
  // ══════════════════════════════════════════════════════════════════════════

  {
    name: 'Impulse', sub: 'Basic 5-wave · W3 longest · Bullish',
    category: 'EW Motive', direction: 'bullish',
    buildData: () => buildBars([[0,80],[8,96],[13,85],[22,118],[28,108],[38,132]]),
    draw: (c,d) => {
      seg(c,d,0,80,8,96,IMP); seg(c,d,8,96,13,85,COR); seg(c,d,13,85,22,118,IMP)
      seg(c,d,22,118,28,108,COR); seg(c,d,28,108,38,132,IMP)
      circle(c,d,8,96,'1',IMP); circle(c,d,13,85,'2',COR,0.025,false)
      circle(c,d,22,118,'3',IMP); circle(c,d,28,108,'4',COR,0.025,false)
      circle(c,d,38,132,'5',IMP)
      lvl(c,d,132,'W5 TARGET'); lvl(c,d,85,'W2 LOW · 0.59 retrace',COR)
      bdg(c,d,1,139,'IMPULSE  5-wave · 92%',IMP)
      path(c,d,[[38,132],[42,126],[46,120],[50,114]],COR)
      cone(c,d,[38,132],[50,120],[50,108],'rgba(245,158,11,0.05)','rgba(245,158,11,0.2)')
      zone(c,d,44,108,52,118,'rgba(34,197,94,0.07)','rgba(34,197,94,0.2)','W4 RETRACE',BULL)
      zone(c,d,38,133,52,137,'rgba(239,68,68,0.06)','rgba(239,68,68,0.2)','STOP',BEAR)
    },
  },

  {
    name: 'Impulse · W3 Extension', sub: '9-wave look · W3 = 2.62×W1 · Bullish',
    category: 'EW Motive', direction: 'bullish',
    buildData: () => buildBars([[0,80],[8,96],[13,85],[17,98],[20,91],[27,120],[31,112],[37,131],[43,118],[52,148]]),
    draw: (c,d) => {
      seg(c,d,0,80,8,96,IMP); seg(c,d,8,96,13,85,COR)
      circle(c,d,8,96,'1',IMP); circle(c,d,13,85,'2',COR,0.025,false)
      seg(c,d,13,85,17,98,EXT); seg(c,d,17,98,20,91,EXT)
      seg(c,d,20,91,27,120,EXT,3.5); seg(c,d,27,120,31,112,EXT)
      seg(c,d,31,112,37,131,EXT)
      circle(c,d,13,85,'3',EXT,0.055,false)
      circle(c,d,17,98,'i',EXT,0.02); circle(c,d,20,91,'ii',EXT,0.02,false)
      circle(c,d,27,120,'iii',EXT,0.02); circle(c,d,31,112,'iv',EXT,0.02,false)
      circle(c,d,37,131,'v',EXT,0.02)
      seg(c,d,37,131,43,118,COR); seg(c,d,43,118,52,148,IMP)
      circle(c,d,43,118,'4',COR,0.025,false); circle(c,d,52,148,'5',IMP)
      seg(c,d,8,96,52,150,BDY,1,[5,4]); seg(c,d,0,80,52,132,BDY,1,[5,4])
      lbl(c,d,48,149,'1-3 channel')
      lvl(c,d,148,'W5 = W1 equality'); lvl(c,d,131,'W3 TOP · 2.62×W1',EXT)
      bdg(c,d,1,158,'IMPULSE  W3 EXT · 94%',EXT)
      path(c,d,[[52,148],[55,140],[58,134],[61,128]],COR)
      cone(c,d,[52,148],[61,122],[61,138],'rgba(245,158,11,0.05)','rgba(245,158,11,0.2)')
    },
  },

  {
    name: 'Impulse · W1 Extension', sub: 'W1 longest (commodity pattern) · Bullish',
    category: 'EW Motive', direction: 'bullish',
    buildData: () => buildBars([[0,80],[14,114],[18,104],[22,118],[25,112],[30,122]]),
    draw: (c,d) => {
      seg(c,d,0,80,14,114,EXT,3.5)   // W1 extended
      seg(c,d,14,114,18,104,COR)
      seg(c,d,18,104,22,118,IMP)
      seg(c,d,22,118,25,112,COR)
      seg(c,d,25,112,30,122,IMP)
      circle(c,d,14,114,'1',EXT); circle(c,d,18,104,'2',COR,0.025,false)
      circle(c,d,22,118,'3',IMP); circle(c,d,25,112,'4',COR,0.025,false)
      circle(c,d,30,122,'5',IMP)
      lbl(c,d,7,115,'W1 EXTENDED',EXT,-14,9)
      lvl(c,d,122,'W5 TARGET')
      bdg(c,d,1,129,'IMPULSE  W1 EXT · 88%',EXT)
      path(c,d,[[30,122],[34,116],[38,110],[42,106]],COR)
      cone(c,d,[30,122],[42,100],[42,112],'rgba(245,158,11,0.05)','rgba(245,158,11,0.2)')
      zone(c,d,30,123,44,127,'rgba(239,68,68,0.06)','rgba(239,68,68,0.2)','STOP',BEAR)
    },
  },

  {
    name: 'Impulse · W5 Extension', sub: 'W5 longest (crypto pattern) · Bullish',
    category: 'EW Motive', direction: 'bullish',
    buildData: () => buildBars([[0,80],[8,96],[13,86],[20,112],[25,103],[36,138]]),
    draw: (c,d) => {
      seg(c,d,0,80,8,96,IMP); seg(c,d,8,96,13,86,COR)
      seg(c,d,13,86,20,112,IMP); seg(c,d,20,112,25,103,COR)
      seg(c,d,25,103,36,138,EXT,3.5)  // W5 extended
      circle(c,d,8,96,'1',IMP); circle(c,d,13,86,'2',COR,0.025,false)
      circle(c,d,20,112,'3',IMP); circle(c,d,25,103,'4',COR,0.025,false)
      circle(c,d,36,138,'5',EXT)
      lbl(c,d,30,138,'W5 EXTENDED (crypto)',EXT,-14,9)
      lvl(c,d,138,'W5 TARGET · common in crypto',EXT)
      bdg(c,d,1,146,'IMPULSE  W5 EXT · 85%',EXT)
      path(c,d,[[36,138],[40,128],[44,118],[48,110]],COR)
      cone(c,d,[36,138],[48,100],[48,122],'rgba(245,158,11,0.05)','rgba(245,158,11,0.2)')
      zone(c,d,36,139,50,145,'rgba(239,68,68,0.06)','rgba(239,68,68,0.2)','STOP',BEAR)
    },
  },

  {
    name: 'Truncated 5th', sub: 'W5 fails to exceed W3 · Bearish reversal signal',
    category: 'EW Motive', direction: 'bearish',
    buildData: () => buildBars([[0,80],[8,96],[13,85],[22,120],[28,108],[36,117]]),
    draw: (c,d) => {
      seg(c,d,0,80,8,96,IMP); seg(c,d,8,96,13,85,COR)
      seg(c,d,13,85,22,120,IMP); seg(c,d,22,120,28,108,COR)
      seg(c,d,28,108,36,117,'#f87171',2.5)  // W5 truncated — below W3 peak, red = warning
      circle(c,d,8,96,'1',IMP); circle(c,d,13,85,'2',COR,0.025,false)
      circle(c,d,22,120,'3',IMP); circle(c,d,28,108,'4',COR,0.025,false)
      circle(c,d,36,117,'5','#f87171')
      seg(c,d,36,120,0,120,'#f87171',1,[4,3])  // W3 high reference line
      lbl(c,d,30,120,'W3 HIGH',IMP,-14,9)
      lbl(c,d,36,117,'TRUNCATED (below W3)',BEAR,14,9)
      lvl(c,d,120,'W3 HIGH — W5 failed to exceed','#ef4444')
      bdg(c,d,1,128,'TRUNCATED 5TH · Reversal signal · 83%',BEAR)
      path(c,d,[[36,117],[40,108],[44,98],[48,88]],BEAR)
      cone(c,d,[36,117],[48,78],[48,100],'rgba(248,113,113,0.06)','rgba(248,113,113,0.2)')
      zone(c,d,40,80,52,92,'rgba(34,197,94,0.07)','rgba(34,197,94,0.2)','TARGET',BULL)
      zone(c,d,36,121,52,126,'rgba(239,68,68,0.06)','rgba(239,68,68,0.2)','STOP: new high',BEAR)
    },
  },

  {
    name: 'Leading Diagonal', sub: 'W1 position · 5-3-5-3-5 · W4 overlaps W1',
    category: 'EW Motive', direction: 'bullish',
    buildData: () => buildBars([[0,90],[6,106],[9,98],[15,116],[18,108],[24,124],[32,108],[42,138]]),
    draw: (c,d) => {
      seg(c,d,0,90,6,106,DIAG); seg(c,d,6,106,9,98,DIAG)
      seg(c,d,9,98,15,116,DIAG); seg(c,d,15,116,18,108,DIAG)
      seg(c,d,18,108,24,124,DIAG)
      circle(c,d,0,90,'W1',DIAG,0.02,false)
      circle(c,d,6,106,'i',DIAG,0.02); circle(c,d,9,98,'ii',DIAG,0.02,false)
      circle(c,d,15,116,'iii',DIAG,0.02); circle(c,d,18,108,'iv',DIAG,0.02,false)
      circle(c,d,24,124,'v',DIAG)
      // D4 overlaps D1 — show overlap zone
      lbl(c,d,18,108,'W4 overlaps W1',DIAG,18,9)
      seg(c,d,6,106,24,122,BDY,1.2,[5,4]); seg(c,d,9,98,24,110,BDY,1.2,[5,4])
      seg(c,d,24,124,32,108,COR); seg(c,d,32,108,42,138,IMP)
      circle(c,d,32,108,'2',COR,0.025,false); circle(c,d,42,138,'3',IMP)
      bdg(c,d,1,146,'LEADING DIAGONAL  W1 · 86%',DIAG)
      lvl(c,d,138,'W3 TARGET')
      path(c,d,[[42,138],[47,128],[52,120]],COR)
      cone(c,d,[42,138],[52,112],[52,126],'rgba(245,158,11,0.05)','rgba(245,158,11,0.2)')
    },
  },

  {
    name: 'Ending Diagonal + Throw-Over', sub: 'W5 · 3-3-3-3-3 · Channel breach → Reversal',
    category: 'EW Motive', direction: 'bearish',
    buildData: () => buildBars([[0,100],[7,118],[12,106],[22,138],[28,118],[32,130],[35,121],[39,139],[43,128],[47,148],[54,108]]),
    draw: (c,d) => {
      seg(c,d,0,100,7,118,IMP); seg(c,d,7,118,12,106,COR)
      seg(c,d,12,106,22,138,IMP); seg(c,d,22,138,28,118,COR)
      circle(c,d,7,118,'1',IMP); circle(c,d,12,106,'2',COR,0.025,false)
      circle(c,d,22,138,'3',IMP); circle(c,d,28,118,'4',COR,0.025,false)
      lbl(c,d,28,118,'W4 overlaps W1',DIAG,18,9)
      seg(c,d,28,118,32,130,DIAG); seg(c,d,32,130,35,121,DIAG)
      seg(c,d,35,121,39,139,DIAG); seg(c,d,39,139,43,128,DIAG)
      seg(c,d,43,128,47,148,DIAG,3.5)
      circle(c,d,28,118,'5',DIAG,0.025,false)
      circle(c,d,32,130,'i',DIAG,0.02); circle(c,d,35,121,'ii',DIAG,0.02,false)
      circle(c,d,39,139,'iii',DIAG,0.02); circle(c,d,43,128,'iv',DIAG,0.02,false)
      circle(c,d,47,148,'v',BEAR)
      seg(c,d,32,130,49,144,BDY,1.2,[5,4]); seg(c,d,35,121,49,133,BDY,1.2,[5,4])
      lbl(c,d,47,148,'⚡ THROW-OVER',BEAR,-14,10)
      seg(c,d,47,148,54,108,'#ef4444',3.5)
      bdg(c,d,1,157,'ENDING DIAGONAL  W5 + THROW-OVER · 87%',DIAG)
      lvl(c,d,118,'W2/W4 ZONE — post-diagonal target',COR)
      lvl(c,d,100,'W1 START — full retrace target','#ef4444')
      path(c,d,[[47,148],[49,138],[51,124],[54,108]],BEAR)
      cone(c,d,[47,148],[54,96],[54,118],'rgba(248,113,113,0.06)','rgba(248,113,113,0.2)')
      zone(c,d,51,100,58,118,'rgba(34,197,94,0.07)','rgba(34,197,94,0.2)','POST-DIAG TARGET',BULL)
      zone(c,d,47,149,58,154,'rgba(239,68,68,0.06)','rgba(239,68,68,0.2)','STOP: above throwover',BEAR)
    },
  },

  // ══════════════════════════════════════════════════════════════════════════
  //  EW CORRECTIVE
  // ══════════════════════════════════════════════════════════════════════════

  {
    name: 'Zigzag  (Bearish)', sub: 'A-B-C · 5-3-5 · B retraces < 61.8% of A',
    category: 'EW Corrective', direction: 'bearish',
    buildData: () => buildBars([[0,120],[10,104],[15,112],[25,96]]),
    draw: (c,d) => {
      seg(c,d,0,120,10,104,COR,3); seg(c,d,10,104,15,112,COR,3); seg(c,d,15,112,25,96,COR,3)
      circle(c,d,0,120,'A',COR); circle(c,d,10,104,'B',COR,0.025,false)
      circle(c,d,15,112,'B',COR); circle(c,d,25,96,'C',COR,0.025,false)
      lbl(c,d,12,113,'B < 61.8% retrace of A',COR,-14,9)
      lvl(c,d,96,'C TARGET · C = A')
      bdg(c,d,1,128,'ZIGZAG  5-3-5 Bearish · 90%',COR)
      path(c,d,[[25,96],[29,102],[33,108]],BULL)
      cone(c,d,[25,96],[33,114],[33,103],'rgba(74,222,128,0.06)','rgba(74,222,128,0.2)')
      zone(c,d,27,93,35,97,'rgba(34,197,94,0.07)','rgba(34,197,94,0.2)','C COMPLETION ZONE',BULL)
      zone(c,d,25,119,35,122,'rgba(239,68,68,0.06)','rgba(239,68,68,0.2)','STOP: above A start',BEAR)
    },
  },

  {
    name: 'Zigzag  (Bullish)', sub: 'A-B-C up · Prior bear trend · Corrective rally',
    category: 'EW Corrective', direction: 'bullish',
    buildData: () => buildBars([[0,85],[10,101],[15,94],[25,112]]),
    draw: (c,d) => {
      seg(c,d,0,85,10,101,COR,3); seg(c,d,10,101,15,94,COR,3); seg(c,d,15,94,25,112,COR,3)
      circle(c,d,0,85,'A',COR,0.025,false); circle(c,d,10,101,'A',COR)
      circle(c,d,15,94,'B',COR,0.025,false); circle(c,d,25,112,'C',COR)
      lbl(c,d,12,100,'B retraces A',COR,14,9)
      lvl(c,d,112,'C TARGET · C = A','#22c55e')
      bdg(c,d,1,118,'ZIGZAG  5-3-5 Bullish · 90%','#4ade80')
      path(c,d,[[25,112],[29,106],[33,100]],BEAR)
      cone(c,d,[25,112],[33,96],[33,106],'rgba(248,113,113,0.06)','rgba(248,113,113,0.2)')
      zone(c,d,25,113,35,117,'rgba(239,68,68,0.06)','rgba(239,68,68,0.2)','STOP: above C',BEAR)
    },
  },

  {
    name: 'Flat  Regular', sub: 'A-B-C · B ≈ A · C ≈ A',
    category: 'EW Corrective', direction: 'bullish',
    buildData: () => buildBars([[0,110],[8,100],[15,110],[23,98]]),
    draw: (c,d) => {
      seg(c,d,0,110,8,100,COR,3); seg(c,d,8,100,15,110,COR,3); seg(c,d,15,110,23,98,COR,3)
      circle(c,d,0,110,'A',COR); circle(c,d,8,100,'A',COR,0.025,false)
      circle(c,d,15,110,'B',COR); circle(c,d,23,98,'C',COR,0.025,false)
      lbl(c,d,15,110,'B ≈ A start (100%)',COR,-14,9)
      lbl(c,d,23,98,'C ≈ A',COR,14,9)
      lvl(c,d,98,'C TARGET (= A end)')
      bdg(c,d,1,117,'FLAT REGULAR · 86%',COR)
      path(c,d,[[23,98],[27,104],[31,110],[35,115]],BULL)
      cone(c,d,[23,98],[35,120],[35,108],'rgba(74,222,128,0.06)','rgba(74,222,128,0.2)')
      zone(c,d,25,95,37,98,'rgba(34,197,94,0.07)','rgba(34,197,94,0.2)','BUY ZONE',BULL)
      zone(c,d,23,97,37,99,'rgba(239,68,68,0.06)','rgba(239,68,68,0.2)','STOP: below C',BEAR)
    },
  },

  {
    name: 'Flat  Expanded', sub: 'A-B-C · B > A · C > A · Most common flat',
    category: 'EW Corrective', direction: 'bullish',
    buildData: () => buildBars([[0,110],[8,100],[16,113],[26,94]]),
    draw: (c,d) => {
      seg(c,d,0,110,8,100,COR,3); seg(c,d,8,100,16,113,COR,3); seg(c,d,16,113,26,94,COR,3)
      circle(c,d,0,110,'A',COR); circle(c,d,8,100,'A',COR,0.025,false)
      circle(c,d,16,113,'B',COR); circle(c,d,26,94,'C',COR,0.025,false)
      seg(c,d,0,110,26,110,BDY,1,[4,3])  // A start reference
      lbl(c,d,16,113,'B > A start (103%)',COR,-14,9)
      lbl(c,d,26,94,'C > A (1.618×A)',COR,14,9)
      lvl(c,d,94,'C TARGET (1.618×A)'); lvl(c,d,100,'A END reference',COR)
      bdg(c,d,1,120,'FLAT EXPANDED · 88%',COR)
      path(c,d,[[26,94],[30,100],[34,108],[38,114]],BULL)
      cone(c,d,[26,94],[38,120],[38,108],'rgba(74,222,128,0.06)','rgba(74,222,128,0.2)')
      zone(c,d,24,90,38,95,'rgba(34,197,94,0.07)','rgba(34,197,94,0.2)','BUY ZONE',BULL)
      zone(c,d,26,93,38,95,'rgba(239,68,68,0.06)','rgba(239,68,68,0.2)','STOP: below C',BEAR)
    },
  },

  {
    name: 'Flat  Running', sub: 'A-B-C · B > A · C ends above A · Rare, strong trend',
    category: 'EW Corrective', direction: 'bullish',
    buildData: () => buildBars([[0,108],[8,100],[16,115],[24,104]]),
    draw: (c,d) => {
      seg(c,d,0,108,8,100,COR,3); seg(c,d,8,100,16,115,COR,3); seg(c,d,16,115,24,104,COR,3)
      circle(c,d,0,108,'A',COR); circle(c,d,8,100,'A',COR,0.025,false)
      circle(c,d,16,115,'B',COR); circle(c,d,24,104,'C',COR,0.025,false)
      seg(c,d,0,108,24,108,BDY,1,[4,3])
      lbl(c,d,16,115,'B > A start',COR,-14,9)
      lbl(c,d,24,104,'C > A end (running!)',IMP,14,9)
      lvl(c,d,100,'A END'); lvl(c,d,104,'C (above A — running)','#22c55e')
      bdg(c,d,1,122,'FLAT RUNNING · Rare, strong trend · 80%',COR)
      path(c,d,[[24,104],[28,112],[32,122],[36,130]],BULL)
      cone(c,d,[24,104],[36,136],[36,122],'rgba(74,222,128,0.06)','rgba(74,222,128,0.2)')
    },
  },

  // ══════════════════════════════════════════════════════════════════════════
  //  EW TRIANGLE
  // ══════════════════════════════════════════════════════════════════════════

  {
    name: 'Contracting Triangle', sub: 'W4 · ABCDE · Trendlines converge → Thrust',
    category: 'EW Triangle', direction: 'bullish',
    buildData: () => buildBars([[0,82],[8,97],[13,88],[23,122],[29,108],[33,116],[37,110],[40,114],[43,111],[52,132]]),
    draw: (c,d) => {
      seg(c,d,0,82,8,97,IMP); seg(c,d,8,97,13,88,COR); seg(c,d,13,88,23,122,IMP)
      circle(c,d,8,97,'1',IMP); circle(c,d,13,88,'2',COR,0.025,false); circle(c,d,23,122,'3',IMP)
      seg(c,d,23,122,29,108,COR,2.5); seg(c,d,29,108,33,116,COR,2.5)
      seg(c,d,33,116,37,110,COR,2.5); seg(c,d,37,110,40,114,COR,2.5)
      seg(c,d,40,114,43,111,COR,2.5)
      circle(c,d,23,122,'4',COR); circle(c,d,29,108,'A',COR,0.025,false)
      circle(c,d,33,116,'B',COR); circle(c,d,37,110,'C',COR,0.025,false)
      circle(c,d,40,114,'D',COR); circle(c,d,43,111,'E',COR,0.025,false)
      seg(c,d,33,116,49,112,BDY,1.2,[5,4]); seg(c,d,29,108,49,112,BDY,1.2,[5,4])
      lbl(c,d,49,112,'apex')
      seg(c,d,43,111,52,132,IMP); circle(c,d,52,132,'5',IMP)
      lvl(c,d,132,'THRUST TARGET'); bdg(c,d,1,139,'CONTRACTING TRIANGLE  W4 · 91%',IMP)
      path(c,d,[[43,111],[46,118],[49,125],[52,132]],BULL)
      cone(c,d,[43,111],[52,138],[52,126],'rgba(74,222,128,0.06)','rgba(74,222,128,0.2)')
      zone(c,d,43,107,54,109,'rgba(239,68,68,0.06)','rgba(239,68,68,0.2)','STOP: below E',BEAR)
    },
  },

  {
    name: 'Barrier Triangle', sub: 'W4 · ABCDE · One flat trendline · Thrust',
    category: 'EW Triangle', direction: 'bullish',
    buildData: () => buildBars([[0,82],[8,97],[13,88],[23,122],[28,107],[32,115],[36,109],[40,115],[44,111],[53,132]]),
    draw: (c,d) => {
      seg(c,d,0,82,8,97,IMP); seg(c,d,8,97,13,88,COR); seg(c,d,13,88,23,122,IMP)
      circle(c,d,8,97,'1',IMP); circle(c,d,13,88,'2',COR,0.025,false); circle(c,d,23,122,'3',IMP)
      seg(c,d,23,122,28,107,COR,2.5); seg(c,d,28,107,32,115,COR,2.5)
      seg(c,d,32,115,36,109,COR,2.5); seg(c,d,36,109,40,115,COR,2.5)
      seg(c,d,40,115,44,111,COR,2.5)
      circle(c,d,23,122,'4',COR); circle(c,d,28,107,'A',COR,0.025,false)
      circle(c,d,32,115,'B',COR); circle(c,d,36,109,'C',COR,0.025,false)
      circle(c,d,40,115,'D',COR); circle(c,d,44,111,'E',COR,0.025,false)
      seg(c,d,28,115,50,115,BDY,1.5,[5,4])   // flat upper
      seg(c,d,28,107,50,112,BDY,1.2,[5,4])   // rising lower
      lbl(c,d,42,115,'flat barrier',BDY,-12,9)
      seg(c,d,44,111,53,132,IMP); circle(c,d,53,132,'5',IMP)
      lvl(c,d,115,'BARRIER / FLAT UPPER','#94a3b8'); lvl(c,d,132,'THRUST TARGET')
      bdg(c,d,1,139,'BARRIER TRIANGLE  W4 · 88%',IMP)
      path(c,d,[[44,111],[47,119],[50,126],[53,132]],BULL)
      cone(c,d,[44,111],[53,139],[53,126],'rgba(74,222,128,0.06)','rgba(74,222,128,0.2)')
    },
  },

  {
    name: 'Expanding Triangle', sub: 'B wave · ABCDE · Trendlines diverge',
    category: 'EW Triangle', direction: 'bearish',
    buildData: () => buildBars([[0,118],[8,100],[12,108],[16,100],[21,114],[26,96],[32,121],[45,74]]),
    draw: (c,d) => {
      seg(c,d,0,118,8,100,'#ef4444',2.5); circle(c,d,8,100,'A','#ef4444',0.025,false)
      seg(c,d,8,100,12,108,COR,2.5); seg(c,d,12,108,16,100,COR,2.5)
      seg(c,d,16,100,21,114,COR,2.5); seg(c,d,21,114,26,96,COR,2.5)
      seg(c,d,26,96,32,121,COR,2.5)
      circle(c,d,8,100,'B',COR); circle(c,d,12,108,'a',COR)
      circle(c,d,16,100,'b',COR,0.025,false); circle(c,d,21,114,'c',COR)
      circle(c,d,26,96,'d',COR,0.025,false); circle(c,d,32,121,'e',COR)
      seg(c,d,12,108,32,121,BDY,1.5,[5,4])   // upper rising
      seg(c,d,16,100,30,94,BDY,1.2,[5,4])    // lower declining
      lbl(c,d,22,114,'diverging',BDY,-14,9)
      seg(c,d,32,121,45,74,'#ef4444',3); circle(c,d,45,74,'C','#ef4444',0.025,false)
      lvl(c,d,74,'C TARGET  A=C','#22c55e'); lvl(c,d,121,'E HIGH / SHORT','#ef4444')
      bdg(c,d,1,128,'EXPANDING TRIANGLE  B wave · 84%','#f87171')
      path(c,d,[[32,121],[36,109],[40,95],[45,74]],BEAR)
      cone(c,d,[32,121],[45,64],[45,82],'rgba(248,113,113,0.06)','rgba(248,113,113,0.2)')
      zone(c,d,40,68,48,80,'rgba(34,197,94,0.07)','rgba(34,197,94,0.2)','C TARGET',BULL)
      zone(c,d,32,123,48,126,'rgba(239,68,68,0.06)','rgba(239,68,68,0.2)','STOP: above E',BEAR)
    },
  },

  {
    name: 'Running Triangle', sub: 'W4 · B exceeds A start · Net trend progress',
    category: 'EW Triangle', direction: 'bullish',
    buildData: () => buildBars([[0,82],[8,97],[13,88],[23,122],[28,106],[34,125],[38,111],[42,119],[45,112],[54,138]]),
    draw: (c,d) => {
      seg(c,d,0,82,8,97,IMP); seg(c,d,8,97,13,88,COR); seg(c,d,13,88,23,122,IMP)
      circle(c,d,8,97,'1',IMP); circle(c,d,13,88,'2',COR,0.025,false); circle(c,d,23,122,'3',IMP)
      seg(c,d,23,122,28,106,COR,2.5); seg(c,d,28,106,34,125,COR,2.5)  // B exceeds A start!
      seg(c,d,34,125,38,111,COR,2.5); seg(c,d,38,111,42,119,COR,2.5)
      seg(c,d,42,119,45,112,COR,2.5)
      circle(c,d,23,122,'4',COR); circle(c,d,28,106,'A',COR,0.025,false)
      circle(c,d,34,125,'B',COR); circle(c,d,38,111,'C',COR,0.025,false)
      circle(c,d,42,119,'D',COR); circle(c,d,45,112,'E',COR,0.025,false)
      lbl(c,d,34,125,'B > prior W3 high!',COR,-14,9)
      seg(c,d,34,125,52,122,BDY,1.2,[5,4]); seg(c,d,28,106,52,112,BDY,1.2,[5,4])
      lbl(c,d,48,123,'running',BDY,-12,9)
      seg(c,d,45,112,54,138,IMP); circle(c,d,54,138,'5',IMP)
      lvl(c,d,138,'THRUST TARGET'); lvl(c,d,125,'B HIGH — above W3 (running!)',COR)
      bdg(c,d,1,146,'RUNNING TRIANGLE  W4 · 82%',IMP)
      path(c,d,[[45,112],[48,120],[51,129],[54,138]],BULL)
      cone(c,d,[45,112],[54,144],[54,132],'rgba(74,222,128,0.06)','rgba(74,222,128,0.2)')
    },
  },

  // ══════════════════════════════════════════════════════════════════════════
  //  EW COMPLEX
  // ══════════════════════════════════════════════════════════════════════════

  {
    name: 'Double Zigzag  W-X-Y', sub: 'W2 · Two ABC corrections linked by X',
    category: 'EW Complex', direction: 'bearish',
    buildData: () => buildBars([[0,88],[10,122],[14,112],[17,118],[22,104],[26,110],[30,102],[33,107],[38,95],[48,138]]),
    draw: (c,d) => {
      seg(c,d,0,88,10,122,IMP); circle(c,d,10,122,'1',IMP)
      const W='#f59e0b',X='#818cf8',Y='#fbbf24'
      seg(c,d,10,122,14,112,W,2.5); seg(c,d,14,112,17,118,W,2.5); seg(c,d,17,118,22,104,W,2.5)
      circle(c,d,10,122,'W',W); circle(c,d,14,112,'A',W,0.025,false)
      circle(c,d,17,118,'B',W); circle(c,d,22,104,'C',W,0.025,false)
      seg(c,d,22,104,26,110,X,2.5,[4,3]); circle(c,d,26,110,'X',X)
      seg(c,d,26,110,30,102,Y,2.5); seg(c,d,30,102,33,107,Y,2.5); seg(c,d,33,107,38,95,Y,2.5)
      circle(c,d,26,110,'Y',Y); circle(c,d,30,102,'A',Y,0.025,false)
      circle(c,d,33,107,'B',Y); circle(c,d,38,95,'C',Y,0.025,false)
      circle(c,d,38,95,'2','#ef4444',0.055,false)
      seg(c,d,38,95,48,138,IMP); circle(c,d,48,138,'3',IMP)
      lbl(c,d,16,118,'Zigzag W',W,-14,9); lbl(c,d,32,107,'Zigzag Y',Y,-14,9)
      lvl(c,d,95,'W2 END · W3 ENTRY'); lvl(c,d,88,'W2 INVALIDATION','#ef4444')
      bdg(c,d,1,143,'DOUBLE ZIGZAG  W-X-Y · 89%','#fbbf24')
      path(c,d,[[38,95],[42,110],[46,126],[50,140]],BULL)
      cone(c,d,[38,95],[50,150],[50,132],'rgba(74,222,128,0.06)','rgba(74,222,128,0.2)')
      zone(c,d,44,136,52,148,'rgba(34,197,94,0.07)','rgba(34,197,94,0.2)','W3 TARGET',BULL)
      zone(c,d,38,87,52,89,'rgba(239,68,68,0.06)','rgba(239,68,68,0.2)','STOP: below W1 start',BEAR)
    },
  },

  {
    name: 'Triple Zigzag  W-X-Y-XX-Z', sub: 'Three ABC corrections · Deepest correction type',
    category: 'EW Complex', direction: 'bearish',
    buildData: () => buildBars([[0,88],[8,118],[11,110],[14,115],[18,103],[21,108],[25,100],[27,105],[30,98],[33,103],[37,90],[46,130]]),
    draw: (c,d) => {
      seg(c,d,0,88,8,118,IMP); circle(c,d,8,118,'1',IMP)
      const W='#f59e0b',X='#818cf8',Y='#fbbf24',XX='#6366f1',Z='#fb923c'
      seg(c,d,8,118,11,110,W,2); seg(c,d,11,110,14,115,W,2); seg(c,d,14,115,18,103,W,2)
      circle(c,d,8,118,'W',W); circle(c,d,18,103,'C',W,0.025,false)
      seg(c,d,18,103,21,108,X,2,[4,3]); circle(c,d,21,108,'X',X)
      seg(c,d,21,108,25,100,Y,2); seg(c,d,25,100,27,105,Y,2); seg(c,d,27,105,30,98,Y,2)
      circle(c,d,21,108,'Y',Y); circle(c,d,30,98,'C',Y,0.025,false)
      seg(c,d,30,98,33,103,XX,2,[4,3]); circle(c,d,33,103,'XX',XX)
      seg(c,d,33,103,37,90,Z,2.5); circle(c,d,33,103,'Z',Z)
      circle(c,d,37,90,'C',Z,0.025,false); circle(c,d,37,90,'2','#ef4444',0.055,false)
      seg(c,d,37,90,46,130,IMP); circle(c,d,46,130,'3',IMP)
      lbl(c,d,19,108,'X',X,-14,9); lbl(c,d,31,103,'XX',XX,-14,9)
      lvl(c,d,90,'W2 COMPLETION · W3 ENTRY')
      bdg(c,d,1,137,'TRIPLE ZIGZAG  W-X-Y-XX-Z · 78%',Z)
      path(c,d,[[37,90],[41,106],[44,118],[47,130]],BULL)
      cone(c,d,[37,90],[47,140],[47,122],'rgba(74,222,128,0.06)','rgba(74,222,128,0.2)')
    },
  },

  {
    name: 'Double Combination  W-X-Y', sub: 'Flat + Triangle · Two different correction types',
    category: 'EW Complex', direction: 'neutral',
    buildData: () => buildBars([[0,88],[8,118],[12,108],[18,118],[24,104],[27,109],[30,105],[33,108],[36,106],[38,107],[46,130]]),
    draw: (c,d) => {
      seg(c,d,0,88,8,118,IMP); circle(c,d,8,118,'1',IMP)
      const F='#f59e0b',X='#818cf8',T='#fbbf24'
      // W = flat (A-B-C)
      seg(c,d,8,118,12,108,F,2.5); seg(c,d,12,108,18,118,F,2.5); seg(c,d,18,118,24,104,F,2.5)
      circle(c,d,8,118,'W',F); circle(c,d,12,108,'A',F,0.025,false)
      circle(c,d,18,118,'B',F); circle(c,d,24,104,'C',F,0.025,false)
      lbl(c,d,14,118,'FLAT (W)',F,-14,9)
      // X wave
      seg(c,d,24,104,27,109,X,2.5,[4,3]); circle(c,d,27,109,'X',X)
      // Y = triangle (A-B-C-D-E)
      seg(c,d,27,109,30,105,T,2); seg(c,d,30,105,33,108,T,2)
      seg(c,d,33,108,36,106,T,2); seg(c,d,36,106,38,107,T,2)
      seg(c,d,30,105,38,107,BDY,1,[5,4]); seg(c,d,27,109,38,107,BDY,1,[5,4])
      circle(c,d,27,109,'Y',T); circle(c,d,30,105,'a',T,0.025,false)
      circle(c,d,33,108,'b',T); circle(c,d,36,106,'c',T,0.025,false)
      circle(c,d,38,107,'e',T)
      lbl(c,d,34,108,'TRIANGLE (Y)',T,-14,9)
      circle(c,d,38,107,'2','#ef4444',0.055)
      seg(c,d,38,107,46,130,IMP); circle(c,d,46,130,'3',IMP)
      lvl(c,d,104,'FLAT C · Deepest point')
      bdg(c,d,1,137,'DOUBLE COMBINATION  Flat + Triangle · 81%',T)
      path(c,d,[[38,107],[41,115],[44,122],[46,130]],BULL)
      cone(c,d,[38,107],[46,138],[46,124],'rgba(74,222,128,0.06)','rgba(74,222,128,0.2)')
    },
  },

  // ══════════════════════════════════════════════════════════════════════════
  //  CLASSICAL
  // ══════════════════════════════════════════════════════════════════════════

  {
    name: 'Head & Shoulders', sub: 'Left shoulder · Head · Right shoulder · Neckline',
    category: 'Classical', direction: 'bearish',
    buildData: () => buildBars([[0,100],[4,110],[7,108],[9,114],[13,108],[16,120],[20,108],[23,115],[27,108],[30,106],[33,100],[40,88]]),
    draw: (c,d) => {
      seg(c,d,0,100,4,110,'#ef4444',2); seg(c,d,4,110,7,108,'#ef4444',2)
      seg(c,d,7,108,9,114,'#ef4444',2); seg(c,d,9,114,13,108,'#ef4444',2)
      seg(c,d,13,108,16,120,'#ef4444',2); seg(c,d,16,120,20,108,'#ef4444',2)
      seg(c,d,20,108,23,115,'#ef4444',2); seg(c,d,23,115,27,108,'#ef4444',2)
      seg(c,d,27,108,30,106,'#ef4444',2)
      lbl(c,d,9,114,'LS',BEAR,-14,10); lbl(c,d,16,120,'HEAD',BEAR,-14,10); lbl(c,d,23,115,'RS',BEAR,-14,10)
      seg(c,d,0,108,38,108,NECK,1.2,[6,4])   // neckline
      seg(c,d,30,106,40,88,'#ef4444',3)
      lvl(c,d,96,'TARGET  (neckline - head depth)','#22c55e'); lvl(c,d,108,'NECKLINE',NECK)
      bdg(c,d,1,128,'HEAD & SHOULDERS · 85%','#f87171')
      path(c,d,[[30,106],[33,100],[37,94],[40,88]],BEAR)
      cone(c,d,[30,106],[40,78],[40,96],'rgba(248,113,113,0.06)','rgba(248,113,113,0.2)')
      zone(c,d,35,84,43,94,'rgba(34,197,94,0.07)','rgba(34,197,94,0.2)','TARGET',BULL)
      zone(c,d,30,109,43,113,'rgba(239,68,68,0.06)','rgba(239,68,68,0.2)','STOP: above neckline',BEAR)
    },
  },

  {
    name: 'Inverse Head & Shoulders', sub: 'Bottoming reversal · Neckline breakout',
    category: 'Classical', direction: 'bullish',
    buildData: () => buildBars([[0,120],[4,110],[7,112],[9,106],[13,112],[16,100],[20,112],[23,105],[27,112],[30,114],[33,120],[40,132]]),
    draw: (c,d) => {
      seg(c,d,0,120,4,110,IMP,2); seg(c,d,4,110,7,112,IMP,2)
      seg(c,d,7,112,9,106,IMP,2); seg(c,d,9,106,13,112,IMP,2)
      seg(c,d,13,112,16,100,IMP,2); seg(c,d,16,100,20,112,IMP,2)
      seg(c,d,20,112,23,105,IMP,2); seg(c,d,23,105,27,112,IMP,2)
      seg(c,d,27,112,30,114,IMP,2)
      lbl(c,d,9,106,'LS',BULL,14,10); lbl(c,d,16,100,'HEAD',BULL,14,10); lbl(c,d,23,105,'RS',BULL,14,10)
      seg(c,d,0,112,38,112,NECK,1.2,[6,4])
      seg(c,d,30,114,40,132,IMP,3)
      lvl(c,d,124,'TARGET  (depth above neckline)'); lvl(c,d,112,'NECKLINE',NECK)
      bdg(c,d,1,138,'INV HEAD & SHOULDERS · 86%','#4ade80')
      path(c,d,[[30,114],[33,118],[36,124],[40,132]],BULL)
      cone(c,d,[30,114],[40,140],[40,126],'rgba(74,222,128,0.06)','rgba(74,222,128,0.2)')
      zone(c,d,35,122,44,132,'rgba(34,197,94,0.07)','rgba(34,197,94,0.2)','TARGET',BULL)
      zone(c,d,30,110,44,112,'rgba(239,68,68,0.06)','rgba(239,68,68,0.2)','STOP: below neckline',BEAR)
    },
  },

  {
    name: 'Double Top', sub: 'M-pattern · Two equal highs · Bearish reversal',
    category: 'Classical', direction: 'bearish',
    buildData: () => buildBars([[0,100],[8,120],[13,112],[20,120],[26,112],[30,106],[37,90]]),
    draw: (c,d) => {
      seg(c,d,0,100,8,120,'#ef4444',2.5); seg(c,d,8,120,13,112,'#ef4444',2.5)
      seg(c,d,13,112,20,120,'#ef4444',2.5); seg(c,d,20,120,26,112,'#ef4444',2.5)
      seg(c,d,26,112,30,106,'#ef4444',2.5)
      seg(c,d,0,120,35,120,'#ef4444',1,[4,3])  // resistance at tops
      lbl(c,d,8,120,'TOP 1',BEAR,-14,10); lbl(c,d,20,120,'TOP 2',BEAR,-14,10)
      seg(c,d,0,112,38,112,NECK,1.2,[6,4])   // neckline
      seg(c,d,30,106,37,90,'#ef4444',3)
      lvl(c,d,104,'TARGET  (top - neckline depth)','#22c55e'); lvl(c,d,112,'NECKLINE',NECK)
      bdg(c,d,1,128,'DOUBLE TOP · 84%','#f87171')
      path(c,d,[[30,106],[33,100],[36,95],[37,90]],BEAR)
      cone(c,d,[30,106],[40,82],[40,96],'rgba(248,113,113,0.06)','rgba(248,113,113,0.2)')
      zone(c,d,34,86,42,96,'rgba(34,197,94,0.07)','rgba(34,197,94,0.2)','TARGET',BULL)
      zone(c,d,30,121,42,124,'rgba(239,68,68,0.06)','rgba(239,68,68,0.2)','STOP: above tops',BEAR)
    },
  },

  {
    name: 'Double Bottom', sub: 'W-pattern · Two equal lows · Bullish reversal',
    category: 'Classical', direction: 'bullish',
    buildData: () => buildBars([[0,120],[8,100],[14,112],[20,100],[26,112],[30,116],[37,132]]),
    draw: (c,d) => {
      seg(c,d,0,120,8,100,IMP,2.5); seg(c,d,8,100,14,112,IMP,2.5)
      seg(c,d,14,112,20,100,IMP,2.5); seg(c,d,20,100,26,112,IMP,2.5)
      seg(c,d,26,112,30,116,IMP,2.5)
      seg(c,d,0,100,28,100,'#ef4444',1,[4,3])
      lbl(c,d,8,100,'B1',BULL,14,10); lbl(c,d,20,100,'B2',BULL,14,10)
      seg(c,d,0,112,38,112,NECK,1.2,[6,4])
      seg(c,d,30,116,37,132,IMP,3)
      lvl(c,d,124,'TARGET  (depth above neckline)'); lvl(c,d,112,'NECKLINE',NECK)
      bdg(c,d,1,139,'DOUBLE BOTTOM · 82%','#4ade80')
      path(c,d,[[30,116],[33,120],[36,126],[37,132]],BULL)
      cone(c,d,[30,116],[40,140],[40,126],'rgba(74,222,128,0.06)','rgba(74,222,128,0.2)')
      zone(c,d,34,128,42,136,'rgba(34,197,94,0.07)','rgba(34,197,94,0.2)','TARGET',BULL)
      zone(c,d,30,98,42,100,'rgba(239,68,68,0.06)','rgba(239,68,68,0.2)','STOP: below lows',BEAR)
    },
  },

  {
    name: 'Ascending Triangle', sub: 'Flat resistance · Rising support · Bullish breakout',
    category: 'Classical', direction: 'bullish',
    buildData: () => buildBars([[0,100],[8,115],[11,110],[15,115],[17,111],[21,115],[23,112],[27,115],[30,118],[37,128]]),
    draw: (c,d) => {
      seg(c,d,0,100,8,115,IMP,1.5)  // initial
      seg(c,d,8,115,38,115,'#ef4444',2)    // flat resistance
      seg(c,d,8,110,28,113,IMP,2)          // rising support
      c.createOverlay({ name: 'patternFillZone',
        points: [{ timestamp:ts(d,8),value:115 },{ timestamp:ts(d,28),value:115 },
                 { timestamp:ts(d,28),value:113 },{ timestamp:ts(d,8),value:110 }],
        extendData: { color: 'rgba(34,197,94,0.06)' } })
      seg(c,d,28,115,37,128,IMP,3,[4,3])   // breakout
      lbl(c,d,18,115,'RESISTANCE',BEAR,-14,9); lbl(c,d,18,111,'RISING SUPPORT',BULL,14,9)
      lvl(c,d,128,'TARGET  (triangle height)'); bdg(c,d,1,134,'ASCENDING TRIANGLE · 78%','#4ade80')
      path(c,d,[[28,115],[31,120],[34,124],[37,128]],BULL)
      cone(c,d,[28,115],[40,134],[40,122],'rgba(74,222,128,0.06)','rgba(74,222,128,0.2)')
      zone(c,d,34,126,42,132,'rgba(34,197,94,0.07)','rgba(34,197,94,0.2)','TARGET',BULL)
      zone(c,d,28,112,42,113,'rgba(239,68,68,0.06)','rgba(239,68,68,0.2)','STOP',BEAR)
    },
  },

  {
    name: 'Descending Triangle', sub: 'Flat support · Declining resistance · Bearish breakdown',
    category: 'Classical', direction: 'bearish',
    buildData: () => buildBars([[0,130],[8,115],[11,120],[15,115],[17,119],[21,115],[23,118],[27,115],[30,112],[37,100]]),
    draw: (c,d) => {
      seg(c,d,0,130,8,115,IMP,1.5)
      seg(c,d,8,115,38,115,'#22c55e',2)    // flat support
      seg(c,d,8,120,28,117,'#ef4444',2)    // declining resistance
      c.createOverlay({ name: 'patternFillZone',
        points: [{ timestamp:ts(d,8),value:120 },{ timestamp:ts(d,28),value:117 },
                 { timestamp:ts(d,28),value:115 },{ timestamp:ts(d,8),value:115 }],
        extendData: { color: 'rgba(239,68,68,0.06)' } })
      seg(c,d,28,115,37,100,'#ef4444',3,[4,3])   // breakdown
      lbl(c,d,18,115,'SUPPORT',BULL,14,9); lbl(c,d,18,119,'DECLINING RESISTANCE',BEAR,-14,9)
      lvl(c,d,100,'TARGET  (triangle height)'); bdg(c,d,1,126,'DESCENDING TRIANGLE · 79%','#f87171')
      path(c,d,[[28,115],[31,110],[34,105],[37,100]],BEAR)
      cone(c,d,[28,115],[40,90],[40,104],'rgba(248,113,113,0.06)','rgba(248,113,113,0.2)')
      zone(c,d,34,96,42,104,'rgba(34,197,94,0.07)','rgba(34,197,94,0.2)','TARGET',BULL)
      zone(c,d,28,116,42,118,'rgba(239,68,68,0.06)','rgba(239,68,68,0.2)','STOP',BEAR)
    },
  },

  {
    name: 'Symmetrical Triangle', sub: 'Coiling price action · Direction breakout',
    category: 'Classical', direction: 'neutral',
    buildData: () => buildBars([[0,100],[5,115],[8,108],[12,113],[15,109],[19,111.5],[22,110],[26,111],[30,117],[37,106]]),
    draw: (c,d) => {
      seg(c,d,5,115,26,111,'#ef4444',1.5)   // descending resistance
      seg(c,d,8,108,26,110,'#22c55e',1.5)   // ascending support
      c.createOverlay({ name: 'patternFillZone',
        points: [{ timestamp:ts(d,5),value:115 },{ timestamp:ts(d,26),value:111 },
                 { timestamp:ts(d,26),value:110 },{ timestamp:ts(d,8),value:108 }],
        extendData: { color: 'rgba(245,158,11,0.07)' } })
      lbl(c,d,15,113,'DESCENDING RESIST',BEAR,-14,9)
      lbl(c,d,15,109,'ASCENDING SUPPORT',BULL,14,9)
      // Bull breakout (shown)
      seg(c,d,26,111,30,117,IMP,2.5,[4,3])
      // Bear breakout (dashed other side)
      seg(c,d,26,110,30,104,'#ef4444',1.5,[4,3])
      lvl(c,d,117,'BULL BREAKOUT TARGET'); lvl(c,d,104,'BEAR BREAKDOWN TARGET','#ef4444')
      bdg(c,d,1,124,'SYMMETRICAL TRIANGLE · Neutral · 76%','#fbbf24')
      path(c,d,[[26,111],[29,114],[32,117],[35,120]],BULL)
      cone(c,d,[26,111],[35,124],[35,114],'rgba(74,222,128,0.06)','rgba(74,222,128,0.2)')
    },
  },

  {
    name: 'Bull Flag', sub: 'Strong pole · Tight consolidation · Measured move',
    category: 'Classical', direction: 'bullish',
    buildData: () => buildBars([[0,80],[11,120],[12,119],[15,117],[18,115],[21,116.5],[35,150]]),
    draw: (c,d) => {
      seg(c,d,0,80,11,120,IMP,2.5)           // flagpole
      seg(c,d,11,120,21,115,COR,1.5)          // upper flag line
      seg(c,d,12,118,21,113,COR,1.5)          // lower flag line
      c.createOverlay({ name: 'patternFillZone',
        points: [{ timestamp:ts(d,11),value:120 },{ timestamp:ts(d,21),value:115 },
                 { timestamp:ts(d,21),value:113 },{ timestamp:ts(d,12),value:118 }],
        extendData: { color: 'rgba(245,158,11,0.10)' } })
      seg(c,d,21,115,35,150,IMP,3,[4,3])      // breakout measured move
      lbl(c,d,5,95,'POLE',BULL,14,10); lbl(c,d,16,117,'FLAG',COR,-14,10)
      lvl(c,d,150,'TARGET  (pole height added)'); bdg(c,d,1,158,'BULL FLAG · 88%','#4ade80')
      path(c,d,[[21,115],[26,126],[30,137],[35,150]],BULL)
      cone(c,d,[21,115],[35,158],[35,142],'rgba(74,222,128,0.06)','rgba(74,222,128,0.2)')
      zone(c,d,30,146,40,154,'rgba(34,197,94,0.07)','rgba(34,197,94,0.2)','MEASURED MOVE',BULL)
      zone(c,d,21,111,40,113,'rgba(239,68,68,0.06)','rgba(239,68,68,0.2)','STOP: below flag',BEAR)
    },
  },

  {
    name: 'Bear Flag', sub: 'Strong down pole · Brief bounce · Continuation down',
    category: 'Classical', direction: 'bearish',
    buildData: () => buildBars([[0,150],[11,110],[12,111],[15,113],[18,115],[21,113.5],[35,80]]),
    draw: (c,d) => {
      seg(c,d,0,150,11,110,'#ef4444',2.5)
      seg(c,d,11,110,21,115,'#ef4444',1.5)  // upper flag
      seg(c,d,12,111,21,113,'#ef4444',1.5)  // lower flag
      c.createOverlay({ name: 'patternFillZone',
        points: [{ timestamp:ts(d,11),value:110 },{ timestamp:ts(d,21),value:115 },
                 { timestamp:ts(d,21),value:113 },{ timestamp:ts(d,12),value:111 }],
        extendData: { color: 'rgba(239,68,68,0.08)' } })
      seg(c,d,21,113,35,80,'#ef4444',3,[4,3])
      lbl(c,d,5,135,'POLE',BEAR,-14,10); lbl(c,d,16,113,'FLAG (bounce)',COR,-14,10)
      lvl(c,d,80,'TARGET  (pole height)','#22c55e'); bdg(c,d,1,158,'BEAR FLAG · 87%','#f87171')
      path(c,d,[[21,113],[26,102],[30,92],[35,80]],BEAR)
      cone(c,d,[21,113],[35,72],[35,88],'rgba(248,113,113,0.06)','rgba(248,113,113,0.2)')
      zone(c,d,30,76,42,84,'rgba(34,197,94,0.07)','rgba(34,197,94,0.2)','TARGET',BULL)
      zone(c,d,21,116,42,120,'rgba(239,68,68,0.06)','rgba(239,68,68,0.2)','STOP: above flag',BEAR)
    },
  },

  {
    name: 'Rising Wedge', sub: 'Converging upward lines · Bearish reversal',
    category: 'Classical', direction: 'bearish',
    buildData: () => buildBars([[0,100],[3,102],[5,104],[7,103],[9,105.5],[11,104.5],[13,106.5],[15,105.5],[17,107.5],[19,106.5],[21,108],[24,104],[28,98],[34,90]]),
    draw: (c,d) => {
      seg(c,d,1,102,21,108,'#ef4444',2)   // upper resistance (less steep)
      seg(c,d,3,100,21,106,'#ef4444',2)   // lower support (steeper)
      c.createOverlay({ name: 'patternFillZone',
        points: [{ timestamp:ts(d,1),value:102 },{ timestamp:ts(d,21),value:108 },
                 { timestamp:ts(d,21),value:106 },{ timestamp:ts(d,3),value:100 }],
        extendData: { color: 'rgba(239,68,68,0.08)' } })
      seg(c,d,22,107,28,98,'#ef4444',3,[4,3])
      seg(c,d,28,98,34,90,'#ef4444',3)
      lbl(c,d,11,107,'CONVERGING',BEAR,-14,9)
      lvl(c,d,90,'TARGET'); bdg(c,d,1,115,'RISING WEDGE · 76%','#f87171')
      path(c,d,[[22,107],[25,101],[29,95],[34,90]],BEAR)
      cone(c,d,[22,107],[34,82],[34,96],'rgba(248,113,113,0.06)','rgba(248,113,113,0.2)')
      zone(c,d,30,84,40,92,'rgba(34,197,94,0.07)','rgba(34,197,94,0.2)','TARGET',BULL)
      zone(c,d,22,109,40,112,'rgba(239,68,68,0.06)','rgba(239,68,68,0.2)','STOP',BEAR)
    },
  },

  {
    name: 'Falling Wedge', sub: 'Converging downward lines · Bullish reversal',
    category: 'Classical', direction: 'bullish',
    buildData: () => buildBars([[0,130],[3,128],[5,126],[7,127],[9,124.5],[11,125.5],[13,123.5],[15,124.5],[17,122.5],[19,123.5],[21,122],[24,124],[28,130],[34,138]]),
    draw: (c,d) => {
      seg(c,d,1,128,21,122,'#22c55e',2)   // upper resistance (steeper)
      seg(c,d,3,126,21,120,'#22c55e',2)   // lower support (less steep)
      c.createOverlay({ name: 'patternFillZone',
        points: [{ timestamp:ts(d,1),value:128 },{ timestamp:ts(d,21),value:122 },
                 { timestamp:ts(d,21),value:120 },{ timestamp:ts(d,3),value:126 }],
        extendData: { color: 'rgba(34,197,94,0.07)' } })
      seg(c,d,22,122,28,130,'#22c55e',3,[4,3])
      seg(c,d,28,130,34,138,'#22c55e',3)
      lbl(c,d,11,123,'CONVERGING',BULL,14,9)
      lvl(c,d,138,'TARGET'); bdg(c,d,1,144,'FALLING WEDGE · 77%','#4ade80')
      path(c,d,[[22,122],[25,127],[29,132],[34,138]],BULL)
      cone(c,d,[22,122],[34,146],[34,132],'rgba(74,222,128,0.06)','rgba(74,222,128,0.2)')
      zone(c,d,30,135,40,141,'rgba(34,197,94,0.07)','rgba(34,197,94,0.2)','TARGET',BULL)
      zone(c,d,22,119,40,121,'rgba(239,68,68,0.06)','rgba(239,68,68,0.2)','STOP',BEAR)
    },
  },

  {
    name: 'Cup & Handle', sub: 'U-shaped base · Handle pullback · Breakout',
    category: 'Classical', direction: 'bullish',
    buildData: () => buildBars([[0,115],[2,112],[5,108],[8,104],[11,101],[14,100],[17,101],[20,104],[23,108],[26,112],[28,115],[30,113.5],[33,113],[35,114],[38,115],[42,122],[48,130]]),
    draw: (c,d) => {
      const cup = [[0,115],[2,112],[5,108],[8,104],[11,101],[14,100],[17,101],[20,104],[23,108],[26,112],[28,115]] as [number,number][]
      for (let i=0;i<cup.length-1;i++) seg(c,d,cup[i][0],cup[i][1],cup[i+1][0],cup[i+1][1],IMP,2)
      seg(c,d,28,115,33,113,COR,1.5); seg(c,d,33,113,35,114,COR,1.5); seg(c,d,35,114,38,115,COR,1.5)
      seg(c,d,0,115,42,115,NECK,1.2,[6,4])   // rim line
      seg(c,d,38,115,48,130,IMP,3,[4,3])
      lbl(c,d,14,100,'CUP',BULL,14,10); lbl(c,d,33,113,'HANDLE',COR,14,10)
      lvl(c,d,130,'TARGET  (cup depth + rim)'); lvl(c,d,115,'RIM / BREAKOUT',NECK)
      bdg(c,d,1,137,'CUP & HANDLE · 80%','#4ade80')
      path(c,d,[[38,115],[41,120],[44,125],[48,130]],BULL)
      cone(c,d,[38,115],[48,138],[48,124],'rgba(74,222,128,0.06)','rgba(74,222,128,0.2)')
      zone(c,d,43,127,52,133,'rgba(34,197,94,0.07)','rgba(34,197,94,0.2)','TARGET',BULL)
      zone(c,d,38,110,52,112,'rgba(239,68,68,0.06)','rgba(239,68,68,0.2)','STOP: below handle',BEAR)
    },
  },

]

// ── Category config ───────────────────────────────────────────────────────────
const CATEGORIES: { id: Category | 'All'; label: string; color: string }[] = [
  { id: 'All',            label: 'All Patterns',   color: '#94a3b8' },
  { id: 'EW Motive',      label: 'EW Motive',      color: '#22c55e' },
  { id: 'EW Corrective',  label: 'EW Corrective',  color: '#f59e0b' },
  { id: 'EW Triangle',    label: 'EW Triangle',    color: '#818cf8' },
  { id: 'EW Complex',     label: 'EW Complex',     color: '#fbbf24' },
  { id: 'Classical',      label: 'Classical',      color: '#60a5fa' },
]

// ── Panel ─────────────────────────────────────────────────────────────────────
function PatternPanel({ pattern }: { pattern: Pattern }) {
  const ref  = useRef<HTMLDivElement>(null)
  const cRef = useRef<Chart | null>(null)

  useEffect(() => {
    if (!ref.current) return
    registerPatternOverlays()
    if (cRef.current) { dispose(cRef.current); cRef.current = null }
    const chart = init(ref.current, { styles: STYLES as any })
    if (!chart) return
    cRef.current = chart
    const data = pattern.buildData()
    chart.setDataLoader({ getBars: async (p: any) => p.callback(data, false) } as any)
    ;(chart as any).setSymbol({ ticker: 'MOCK', pricePrecision: 2, volumePrecision: 0 })
    ;(chart as any).setPeriod({ type: 'day', span: 1 })
    setTimeout(() => { if (cRef.current) pattern.draw(cRef.current, data) }, 300)
    return () => { if (cRef.current) { dispose(cRef.current); cRef.current = null } }
  }, [pattern])

  const catCfg = CATEGORIES.find(c => c.id === pattern.category)!
  const dirColor = pattern.direction === 'bullish' ? '#22c55e' : pattern.direction === 'bearish' ? '#ef4444' : '#f59e0b'
  const dirBg    = pattern.direction === 'bullish' ? 'bg-emerald-500/15 border-emerald-500/30'
                 : pattern.direction === 'bearish' ? 'bg-red-500/15 border-red-500/30'
                 : 'bg-amber-500/15 border-amber-500/30'

  return (
    <div className="flex flex-col rounded-xl border border-gray-800 overflow-hidden bg-gray-950">
      <div className="flex items-center justify-between px-3 py-2 bg-gray-900/80 border-b border-gray-800">
        <div className="flex items-center gap-2 min-w-0">
          <span className="text-[9px] font-bold px-1.5 py-0.5 rounded shrink-0"
                style={{ color: catCfg.color, background: catCfg.color + '20' }}>
            {pattern.category}
          </span>
          <span className="text-white font-bold text-sm font-mono truncate">{pattern.name}</span>
          <span className="text-gray-500 text-[10px] font-mono hidden xl:block truncate">{pattern.sub}</span>
        </div>
        <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full border shrink-0 ml-2 ${dirBg} uppercase`}
              style={{ color: dirColor }}>
          {pattern.direction}
        </span>
      </div>
      <div ref={ref} className="w-full h-[300px]" style={{ background: '#0a0a0f' }} />
    </div>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function AllPatternsMockPage() {
  const [active, setActive] = useState<Category | 'All'>('All')

  const visible = active === 'All' ? PATTERNS : PATTERNS.filter(p => p.category === active)
  const counts = Object.fromEntries(
    CATEGORIES.map(c => [c.id, c.id === 'All' ? PATTERNS.length : PATTERNS.filter(p => p.category === c.id).length])
  )

  return (
    <div className="min-h-screen bg-[#0a0a0f] p-6">
      <div className="max-w-[1600px] mx-auto space-y-6">

        {/* Header */}
        <div>
          <div className="flex items-baseline gap-3 mb-1">
            <h1 className="text-2xl font-bold text-white font-mono">Complete Pattern Library</h1>
            <span className="text-gray-500 text-sm font-mono">{PATTERNS.length} patterns</span>
          </div>
          <p className="text-gray-500 text-sm">Every EW and classical pattern in the engine — synthetic OHLCV data, klinecharts v10 overlay API. Based on EW International + classical TA specs.</p>

          {/* Legend */}
          <div className="flex flex-wrap gap-x-5 gap-y-1 mt-3 text-[11px] font-mono">
            <span style={{color: IMP}}>● Emerald = impulse/motive</span>
            <span style={{color: COR}}>● Amber = corrective/ABC</span>
            <span style={{color: DIAG}}>● Indigo = diagonal/X-wave</span>
            <span style={{color: EXT}}>● Teal = extended sub-wave</span>
            <span style={{color: BDY}}>— Slate dashed = boundary lines</span>
            <span style={{color: NECK}}>— Blue dashed = neckline</span>
            <span style={{color: BULL}}>▒ Green = target/take-profit</span>
            <span style={{color: BEAR}}>▒ Red = stop/invalidation</span>
          </div>
        </div>

        {/* Category filter */}
        <div className="flex flex-wrap gap-2">
          {CATEGORIES.map(cat => (
            <button
              key={cat.id}
              onClick={() => setActive(cat.id)}
              className={`px-3 py-1.5 rounded-lg text-xs font-bold font-mono border transition-all ${
                active === cat.id
                  ? 'border-opacity-60 bg-opacity-20'
                  : 'border-gray-700 bg-gray-900/40 text-gray-500 hover:text-gray-300'
              }`}
              style={active === cat.id ? {
                color: cat.color,
                borderColor: cat.color + '60',
                background: cat.color + '15',
              } : {}}
            >
              {cat.label}
              <span className="ml-1.5 opacity-60">({counts[cat.id]})</span>
            </button>
          ))}
        </div>

        {/* Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-4">
          {visible.map(p => (
            <PatternPanel key={p.name} pattern={p} />
          ))}
        </div>

        {/* Rules footer */}
        <div className="grid grid-cols-1 md:grid-cols-5 gap-3 pt-2">
          {[
            { title: 'EW Motive', color: IMP, rules: ['W2 never > 100% of W1','W3 never shortest (1,3,5)','W4 never enters W1 territory','Extensions make 9- or 13-wave look','Diagonal: W4 MUST overlap W1'] },
            { title: 'EW Corrective', color: COR, rules: ['Zigzag: 5-3-5, B < 61.8% of A','Flat: B ≥ 90% of A','Expanded flat: B > A start, C > A','Running flat: C ends above A end','W2 retrace: 50–78.6% of W1'] },
            { title: 'EW Triangle', color: DIAG, rules: ['All 5 legs corrective (3-3-3-3-3)','Never W2 or wave A position','Contracting: each leg smaller','Barrier: one boundary flat','Expanding: each leg larger, diverge','Running: B exceeds prior high'] },
            { title: 'EW Complex', color: '#fbbf24', rules: ['W-X-Y: two corrections + linker','X < 70% of W (rarely exceeds)','Y ≈ W in size','XX = second linking wave (triple)','Common in strong-trend corrections'] },
            { title: 'Classical', color: NECK, rules: ['H&S: measure head→neckline depth','Double top/bottom: equal extremes','Triangle breakout = pattern height','Flag = measured move from pole','Wedge: converging, counter-trend break'] },
          ].map(s => (
            <div key={s.title} className="p-3 bg-gray-900/60 border border-gray-800 rounded-xl text-[11px] font-mono space-y-1">
              <p className="font-bold mb-1.5" style={{ color: s.color }}>{s.title}</p>
              {s.rules.map(r => <p key={r} className="text-gray-500">• {r}</p>)}
            </div>
          ))}
        </div>

      </div>
    </div>
  )
}
