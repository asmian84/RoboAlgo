import { useEffect, useMemo, useRef, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { createChart, type IChartApi, ColorType, LineStyle } from 'lightweight-charts'
import { useChartData, useInstruments, usePatterns, useLiveQuote, useNews, useTrendlines, useIntradayCandles, useRecommendation, useSignals, usePriceLevels, usePatternMTF, useConfluenceScore, useWatchlist, useWatchlistToggle, useFundamentals, useCycleAdvanced, useMarketForce, useConfluenceNodes, useSignalConfidence, usePriceDistribution, useStrategyHealth, useOptionsData } from '../api/hooks'
import type { PatternEntry } from '../types'
import DrawingToolbar from '../components/chart/DrawingToolbar'
import PatternThumbnail from '../components/chart/PatternThumbnail'
import { useChartDrawings } from '../hooks/useChartDrawings'
import { ChannelFillPrimitive } from '../components/chart/drawingPrimitives'
import { STOCK_UNIVERSE_DEDUP } from '../data/stockUniverse'

// Bar-resolution timeframes (fetch new data from yfinance at different intervals)
const BAR_TFS    = ['W', 'M'] as const
// Daily zoom-range timeframes (same daily data, different visible window)
const DAILY_TFS  = ['1M', '3M', '6M', '1Y', '2Y', 'All'] as const
const INTRADAY_TFS = ['1m', '5m', '15m', '30m', '1h', '2h', '4h'] as const
type TF = typeof BAR_TFS[number] | typeof DAILY_TFS[number] | typeof INTRADAY_TFS[number]
const TF_DAYS: Record<string, number> = { '1M': 30, '3M': 90, '6M': 180, '1Y': 365, '2Y': 730 }
const INTRADAY_RESOLUTION: Record<string, number> = { '1m': 1, '5m': 5, '15m': 15, '30m': 30, '1h': 60, '2h': 120, '4h': 240 }
const MAX_PATTERN_OVERLAYS = 4
const MAX_LIQUIDITY_ZONES = 3

export default function ChartPage() {
  const [searchParams, setSearchParams] = useSearchParams()
  const symbol = searchParams.get('symbol') || 'TQQQ'

  // tf must be declared before usePatterns so we can pass it below
  const [tf, setTf] = useState<TF>('1Y')
  const isIntraday  = (INTRADAY_TFS as readonly string[]).includes(tf)
  const isBarTF     = (BAR_TFS     as readonly string[]).includes(tf)
  // Map tf → API interval param (daily is default, W/M fetch different bar resolutions)
  const chartInterval = tf === 'W' ? 'weekly' : tf === 'M' ? 'monthly' : 'daily'

  const { data: instruments } = useInstruments()
  const { data: watchlistItems = [] } = useWatchlist()
  const watchlistToggle = useWatchlistToggle()
  const isWatching = watchlistItems.some(w => w.symbol === symbol)
  const { data: chartData, isLoading } = useChartData(symbol, 0, chartInterval)
  // Pass tf to backend so intraday charts run intraday pattern detection
  const { data: patterns } = usePatterns(symbol, isIntraday ? (tf as string) : undefined)
  const { data: quote } = useLiveQuote(symbol)
  const { data: news } = useNews(symbol)
  const { data: trendlineData } = useTrendlines(symbol)
  const { data: rec } = useRecommendation(symbol)
  const { data: latestSignals = [] } = useSignals(0)
  const { data: priceLevels } = usePriceLevels(symbol)
  const { data: mtfData }     = usePatternMTF(symbol)
  const { data: confluenceData } = useConfluenceScore(symbol)
  const { data: fundamentals } = useFundamentals(symbol)
  const { data: cycleAdvanced } = useCycleAdvanced(symbol)
  const { data: marketForce }   = useMarketForce(symbol)
  const { data: confluenceNodes } = useConfluenceNodes(symbol)
  const { data: signalConfidence } = useSignalConfidence(symbol)
  const { data: priceDistribution } = usePriceDistribution(symbol)
  const { data: strategyHealth } = useStrategyHealth()
  const { data: optionsData } = useOptionsData(symbol)

  const chartRef = useRef<HTMLDivElement>(null)
  const rsiRef = useRef<HTMLDivElement>(null)
  const macdRef = useRef<HTMLDivElement>(null)
  const stoRef = useRef<HTMLDivElement>(null)
  const chartApi = useRef<IChartApi | null>(null)
  const rsiApi = useRef<IChartApi | null>(null)
  const macdApi = useRef<IChartApi | null>(null)
  const stoApi = useRef<IChartApi | null>(null)
  const tsiRef   = useRef<HTMLDivElement>(null)
  const tsiApi   = useRef<IChartApi | null>(null)
  const willrRef    = useRef<HTMLDivElement>(null)
  const willrApi    = useRef<IChartApi | null>(null)
  const bradleyRef  = useRef<HTMLDivElement>(null)
  const bradleyApi  = useRef<IChartApi | null>(null)
  const planetsRef  = useRef<HTMLDivElement>(null)
  const planetsApi  = useRef<IChartApi | null>(null)
  const cciRef   = useRef<HTMLDivElement>(null)
  const cciApi   = useRef<IChartApi | null>(null)
  const obvRef   = useRef<HTMLDivElement>(null)
  const obvApi   = useRef<IChartApi | null>(null)
  const atrPanelRef = useRef<HTMLDivElement>(null)
  const atrPanelApi = useRef<IChartApi | null>(null)
  const candlestickRef = useRef<ReturnType<IChartApi['addCandlestickSeries']> | null>(null)

  // ── Drawing tools ─────────────────────────────────────────────────────
  const {
    drawings, interaction, activeColor, setActiveColor,
    selectTool: selectDrawingTool, handleClick: drawingHandleClick,
    handleCrosshairMove: drawingHandleCrosshairMove, attachAllPrimitives,
    deleteDrawing, clearAllDrawings, updateDrawing,
  } = useChartDrawings(symbol, tf, chartApi, candlestickRef)

  // Derive the full DrawingData object for the currently selected drawing
  const selectedDrawing = interaction.selectedDrawingId
    ? (drawings.find(d => d.id === interaction.selectedDrawingId) ?? null)
    : null

  const [showBB,                setShowBB]                = useState(false)
  const [showMA,                setShowMA]                = useState(false)
  const [showTrendlines,        setShowTrendlines]        = useState(false)
  const [trendlineTF] = useState<'short' | 'medium' | 'long'>('medium')
  const [showGaps,              setShowGaps]              = useState(false)
  const [showPriceZones,        setShowPriceZones]        = useState(false)
  const [activePatternName,    setActivePatternName]    = useState<string | null>(null)
  const [showPatternDropdown,  setShowPatternDropdown]  = useState(false)
  const patternDropdownRef = useRef<HTMLDivElement>(null)
  const [showOverlayDropdown,  setShowOverlayDropdown]  = useState(false)
  const overlayDropdownRef  = useRef<HTMLDivElement>(null)
  const [showLiquidityMap,      setShowLiquidityMap]      = useState(false)
  const [showMonteCarloForecast,setShowMonteCarloForecast]= useState(false)
  const [showStochastics,       setShowStochastics]       = useState(true)
  const [showAddZones,          setShowAddZones]          = useState(false)
  const [noChartData,    setNoChartData]    = useState(false)
  const [fullChart,      setFullChart]      = useState(false)
  const [traceExpanded,  setTraceExpanded]  = useState(false)
  const [collapsedCards, setCollapsedCards] = useState<Record<string, boolean>>({})
  const cc = collapsedCards
  const toggleCC = (id: string) => setCollapsedCards(p => ({ ...p, [id]: !p[id] }))
  const collapseBtn = (id: string) => (
    <button
      onClick={() => toggleCC(id)}
      className="text-[10px] text-gray-700 hover:text-gray-400 transition-colors leading-none ml-1 flex-shrink-0"
      title={cc[id] ? 'Expand' : 'Collapse'}
    >{cc[id] ? '▶' : '▼'}</button>
  )
  const [showCycleWindow,       setShowCycleWindow]       = useState(false)
  const [showForceGauge,        setShowForceGauge]        = useState(false)
  const [showDecisionNodes,     setShowDecisionNodes]     = useState(false)
  const [showSignalConfidence,  setShowSignalConfidence]  = useState(false)
  const [showProbBands,         setShowProbBands]         = useState(false)
  const [showModelHealth,       setShowModelHealth]       = useState(false)
  const [showOptionsFlow,       setShowOptionsFlow]       = useState(false)
  const [showAstroCycles,       setShowAstroCycles]       = useState(false)
  const [showBradleyPane,       setShowBradleyPane]       = useState(false)
  const [showPlanetsPane,       setShowPlanetsPane]       = useState(false)
  const [patternCategoryFilter, setPatternCategoryFilter] = useState<string | null>(null)
  // cleanMode is derived — true when every overlay is off (pure candlestick view)
  const cleanMode = !showBB && !showMA && !showTrendlines && !showGaps && !showPriceZones &&
                    !showLiquidityMap && !showMonteCarloForecast && !showAddZones

  // ── Symbol autocomplete ────────────────────────────────────────────────────
  const [acInput, setAcInput]   = useState(symbol)
  const [acOpen,  setAcOpen]    = useState(false)
  const acRef = useRef<HTMLDivElement>(null)
  const acInputRef = useRef<HTMLInputElement>(null)
  // Only render pattern overlay when the user has explicitly selected one
  const showPatterns = !!activePatternName
  const intradayResolution = isIntraday ? (INTRADAY_RESOLUTION[tf] ?? 5) : 5
  const { data: intradayData, isLoading: intradayLoading } = useIntradayCandles(symbol, intradayResolution, isIntraday)
  const signalForSymbol = useMemo(
    () => latestSignals.filter(s => s.symbol === symbol).sort((a, b) => b.date.localeCompare(a.date))[0],
    [latestSignals, symbol],
  )
  const predictabilityScore = useMemo(() => {
    const probScore = Math.round((signalForSymbol?.probability ?? rec?.components?.xgb_probability ?? 0.5) * 100)
    const alignRaw = rec?.components?.tf_alignment_pct ?? 50
    const alignScore = alignRaw > 1 ? Math.round(alignRaw) : Math.round(alignRaw * 100)
    return Math.round(probScore * 0.7 + alignScore * 0.3)
  }, [rec?.components?.tf_alignment_pct, rec?.components?.xgb_probability, signalForSymbol?.probability])

  const classifyPattern = (p: PatternEntry): 'chart' | 'harmonic' | 'gann' | 'wyckoff' | 'candlestick' | 'behavioral' | 'indicator' | 'volume' | 'strategy' | 'market_analysis' | 'measured_move' | 'astro' | 'other' => {
    const cat = p.pattern_category?.toLowerCase() ?? ''
    if (cat === 'harmonic') return 'harmonic'
    if (cat === 'gann') return 'gann'
    if (cat === 'wyckoff') return 'wyckoff'
    if (cat === 'candlestick') return 'candlestick'
    if (cat === 'behavioral') return 'behavioral'
    if (cat === 'indicator' || cat === 'oscillator') return 'indicator'
    if (cat === 'volume') return 'volume'
    if (cat === 'strategy') return 'strategy'
    if (cat === 'market_analysis') return 'market_analysis'
    if (cat === 'measured_move') return 'measured_move'
    if (cat === 'chart') return 'chart'
    const n = p.pattern_name.toLowerCase()
    if (n.includes('harmonic') || n.includes('gartley') || n.includes('bat') || n.includes('butterfly') || n.includes('crab') || n.includes('cypher')) return 'harmonic'
    if (n.includes('gann')) return 'gann'
    if (n.includes('wyckoff') || n.includes('accumulation') || n.includes('distribution')) return 'wyckoff'
    // All chart sub-patterns (including new ones)
    if (
      n.includes('chart') || n.includes('chair') || n.includes('cup') ||
      n.includes('flag') || n.includes('pennant') || n.includes('channel') ||
      n.includes('triangle') || n.includes('wedge') || n.includes('compression') ||
      n.includes('head') || n.includes('shoulder') || n.includes('double') ||
      n.includes('triple') || n.includes('rounding') || n.includes('rectangle') ||
      n.includes('megaphone') || n.includes('ascending') || n.includes('descending') ||
      n.includes('rising') || n.includes('falling') || n.includes('symmetrical') ||
      n.includes('bear flag') || n.includes('bull flag')
    ) return 'chart'
    return 'other'
  }

  // Direction-aware color for chart patterns (bearish = red, bullish = blue/green)
  const patternColor = (kind: ReturnType<typeof classifyPattern>, direction?: string): string => {
    if (kind === 'harmonic') return '#14b8a6'
    if (kind === 'gann')     return '#f59e0b'
    if (kind === 'wyckoff')  return '#a855f7'
    if (kind === 'candlestick') {
      if (direction === 'bearish') return '#fb923c'
      if (direction === 'bullish') return '#facc15'
      return '#94a3b8'
    }
    if (kind === 'behavioral') {
      if (direction === 'bearish') return '#f472b6'
      return '#c084fc'
    }
    if (kind === 'indicator') return '#38bdf8'
    if (kind === 'volume') {
      if (direction === 'bearish') return '#f87171'
      if (direction === 'bullish') return '#4ade80'
      return '#94a3b8'
    }
    if (kind === 'strategy') {
      if (direction === 'bearish') return '#f43f5e'
      return '#fb923c'
    }
    if (kind === 'market_analysis') return '#22d3ee'
    if (kind === 'measured_move') {
      if (direction === 'bearish') return '#f87171'
      return '#4ade80'
    }
    if (kind === 'astro')  return '#fbbf24'          // amber for astro/Gann signals
    if (kind === 'chart') {
      if (direction === 'bearish') return '#f472b6'   // vivid pink for bearish chart patterns
      if (direction === 'bullish') return '#34d399'   // emerald for bullish
      return '#94a3b8'                                  // slate for neutral
    }
    return '#9ca3af'
  }

  // Check if active pattern is harmonic — if so, hide other overlays for clarity
  const activePattern = patterns?.find(p => p.pattern_name === activePatternName)
  const isHarmonicActive = activePattern ? classifyPattern(activePattern) === 'harmonic' : false

  // Auto-hide overlays when viewing harmonic pattern
  const effectiveShowBB = isHarmonicActive ? false : showBB
  const effectiveShowMA = isHarmonicActive ? false : showMA
  const effectiveShowTrendlines = isHarmonicActive ? false : showTrendlines
  const effectiveShowGaps = isHarmonicActive ? false : showGaps
  const effectiveShowPriceZones = isHarmonicActive ? false : showPriceZones
  const effectiveShowLiquidityMap = isHarmonicActive ? false : showLiquidityMap
  const effectiveShowMonteCarloForecast = isHarmonicActive ? false : showMonteCarloForecast
  const effectiveShowAddZones = isHarmonicActive ? false : showAddZones
  const effectiveShowStochastics = isHarmonicActive ? false : showStochastics

  // Reset state when symbol changes
  useEffect(() => {
    setActivePatternName(null)
    setNoChartData(false)   // clear stale "no data" while new data loads
  }, [symbol])

  // Sync autocomplete input when URL symbol changes
  useEffect(() => { setAcInput(symbol) }, [symbol])

  // Close autocomplete on outside click
  useEffect(() => {
    function handler(e: MouseEvent) {
      if (acRef.current && !acRef.current.contains(e.target as Node)) setAcOpen(false)
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  // Close pattern dropdown on outside click
  useEffect(() => {
    function handler(e: MouseEvent) {
      if (patternDropdownRef.current && !patternDropdownRef.current.contains(e.target as Node))
        setShowPatternDropdown(false)
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  // Close overlay dropdown on outside click
  useEffect(() => {
    function handler(e: MouseEvent) {
      if (overlayDropdownRef.current && !overlayDropdownRef.current.contains(e.target as Node))
        setShowOverlayDropdown(false)
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  // Merged symbol list: DB instruments first (they have price data), then universe
  const allSymbols = useMemo(() => {
    const dbSet = new Set((instruments || []).map(i => i.symbol))
    const universe = STOCK_UNIVERSE_DEDUP.filter(s => !dbSet.has(s.symbol))
    return [
      ...(instruments || []).map(i => ({ symbol: i.symbol, name: i.name || '' })),
      ...universe,
    ]
  }, [instruments])

  // Autocomplete suggestions — symbol prefix match first, then name substring
  const acSuggestions = useMemo(() => {
    const q = acInput.trim().toUpperCase()
    if (!q) return []
    const prefixMatches = allSymbols.filter(s => s.symbol.startsWith(q))
    const nameMatches   = allSymbols.filter(s =>
      !s.symbol.startsWith(q) && s.name.toUpperCase().includes(q)
    )
    return [...prefixMatches, ...nameMatches].slice(0, 10)
  }, [acInput, allSymbols])

  function navigateToSymbol(sym: string) {
    const upper = sym.trim().toUpperCase()
    if (!upper) return
    setSearchParams({ symbol: upper })
    setAcInput(upper)
    setAcOpen(false)
    acInputRef.current?.blur()
  }

  // Toggle all overlays on or off (Clean = all off, Full = all on)
  function setAllOverlays(val: boolean) {
    setShowMA(val); setShowBB(val); setShowTrendlines(val)
    setShowGaps(val); setShowPriceZones(val)
    setShowLiquidityMap(val); setShowMonteCarloForecast(val); setShowAddZones(val)
    // Note: showStochastics controls its own sub-panel visibility — toggled separately
  }

  // ── Stochastics map (shared between chart + position builder) ──────────────
  const stochMap = useMemo(() => {
    if (!chartData) return new Map<string, number>()
    const prices = chartData.prices.filter(p => p.close != null && p.high != null && p.low != null)
    const K = 14
    const result = new Map<string, number>()
    for (let i = K - 1; i < prices.length; i++) {
      const w = prices.slice(i - K + 1, i + 1)
      const lo = Math.min(...w.map(p => p.low!))
      const hi = Math.max(...w.map(p => p.high!))
      result.set(prices[i].date, hi === lo ? 50 : (prices[i].close! - lo) / (hi - lo) * 100)
    }
    return result
  }, [chartData])

  // ── Current add signal (most recent bar) ───────────────────────────────────
  const currentAddSignal = useMemo(() => {
    if (!chartData) return null
    const prices = chartData.prices.filter(p => p.close != null)
    const inds   = chartData.indicators.filter(i => i.rsi != null || i.bb_lower != null)
    if (!prices.length || !inds.length) return null
    const lastP  = prices[prices.length - 1]
    const lastI  = inds[inds.length - 1]
    const stochK = stochMap.get(lastP.date) ?? 50
    const rsiOk   = (lastI.rsi ?? 50) < 40
    const stochOk = stochK < 30
    const bbOk    = lastI.bb_lower != null && lastP.close! <= lastI.bb_lower * 1.03
    const redDay  = lastP.close! < (lastP.open ?? lastP.close!)
    const score   = [rsiOk, stochOk, bbOk, redDay].filter(Boolean).length
    return { rsi: lastI.rsi, stochK, bbLower: lastI.bb_lower, close: lastP.close, rsiOk, stochOk, bbOk, redDay, score }
  }, [chartData, stochMap])

  // ── Live price update ───────────────────────────────────────────────────────
  useEffect(() => {
    if (!quote?.price || !chartData?.prices.length || !candlestickRef.current) return
    if (isIntraday) return
    const lastBar = chartData.prices[chartData.prices.length - 1]
    if (!lastBar.open) return
    const liveBar = {
      time: lastBar.date as import('lightweight-charts').Time,
      open:  lastBar.open!,
      high:  Math.max(lastBar.high ?? quote.price, quote.high ?? quote.price),
      low:   Math.min(lastBar.low  ?? quote.price, quote.low  ?? quote.price),
      close: quote.price,
    }
    try { candlestickRef.current.update(liveBar) } catch { /* chart disposed */ }
  }, [quote?.price])

  // Reset no-data flag when symbol changes
  useEffect(() => { setNoChartData(false) }, [symbol])

  // ── Build chart ────────────────────────────────────────────────────────────
  useEffect(() => {
    try { chartApi.current?.remove() } catch { /* already disposed */ }
    try { rsiApi.current?.remove()  } catch { /* already disposed */ }
    try { macdApi.current?.remove() } catch { /* already disposed */ }
    try { stoApi.current?.remove()  } catch { /* already disposed */ }
    try { tsiApi.current?.remove()  } catch { /* already disposed */ }
    chartApi.current = null
    rsiApi.current = null
    macdApi.current = null
    stoApi.current = null
    tsiApi.current = null
    willrApi.current = null
    cciApi.current = null
    obvApi.current = null
    atrPanelApi.current = null
    candlestickRef.current = null

    const chartOpts = {
      autoSize: true,
      layout: { background: { type: ColorType.Solid as const, color: '#0a0a0a' }, textColor: '#9ca3af' },
      grid: { vertLines: { color: '#1f2937' }, horzLines: { color: '#1f2937' } },
      crosshair: { mode: 0 as const },
      timeScale: { borderColor: '#374151', timeVisible: true, secondsVisible: false },
      rightPriceScale: { borderColor: '#374151' },
    }

    // ── Intraday chart ─────────────────────────────────────────────────────
    if (isIntraday) {
      if (!chartRef.current || !intradayData) return
      if (intradayData.length === 0) { setNoChartData(true); return }
      setNoChartData(false)

      const chart = createChart(chartRef.current, { ...chartOpts, height: 400 })
      chartApi.current = chart

      const candlestick = chart.addCandlestickSeries({
        upColor: '#22c55e', downColor: '#ef4444',
        borderUpColor: '#22c55e', borderDownColor: '#ef4444',
        wickUpColor: '#22c55e', wickDownColor: '#ef4444',
      })
      candlestickRef.current = candlestick
      candlestick.setData(intradayData.map(b => ({
        time: b.time as import('lightweight-charts').Time,
        open: b.open, high: b.high, low: b.low, close: b.close,
      })))

      const volumeSeries = chart.addHistogramSeries({ priceFormat: { type: 'volume' }, priceScaleId: 'vol' })
      chart.priceScale('vol').applyOptions({ scaleMargins: { top: 0.85, bottom: 0 } })
      volumeSeries.setData(intradayData.map(b => ({
        time: b.time as import('lightweight-charts').Time,
        value: b.volume,
        color: b.close >= b.open ? 'rgba(34,197,94,0.3)' : 'rgba(239,68,68,0.3)',
      })))

      // ── Intraday indicator helpers ──────────────────────────────────────
      type ITime = import('lightweight-charts').Time
      const iCloses = intradayData.map(b => b.close)
      const iN      = intradayData.length

      // EMA seeded with SMA
      const iEMA = (data: number[], period: number): number[] => {
        const k = 2 / (period + 1)
        const out: number[] = new Array(data.length).fill(NaN)
        if (period > data.length) return out
        out[period - 1] = data.slice(0, period).reduce((a, c) => a + c, 0) / period
        for (let i = period; i < data.length; i++) out[i] = data[i] * k + out[i - 1] * (1 - k)
        return out
      }
      // Simple SMA
      const iSMA = (data: number[], period: number): number[] => {
        const out: number[] = new Array(data.length).fill(NaN)
        for (let i = period - 1; i < data.length; i++)
          out[i] = data.slice(i - period + 1, i + 1).reduce((a, c) => a + c, 0) / period
        return out
      }

      // ── MA overlays ────────────────────────────────────────────────────
      if (effectiveShowMA && iN >= 20) {
        const ma20 = iSMA(iCloses, 20)
        const ma20Series = chart.addLineSeries({ color: '#3b82f6', lineWidth: 1, priceLineVisible: false })
        ma20Series.setData(intradayData.map((b, i) => ({ time: b.time as ITime, value: ma20[i] })).filter(d => !isNaN(d.value)))
        if (iN >= 50) {
          const ma50 = iSMA(iCloses, 50)
          const ma50Series = chart.addLineSeries({ color: '#f59e0b', lineWidth: 1, priceLineVisible: false })
          ma50Series.setData(intradayData.map((b, i) => ({ time: b.time as ITime, value: ma50[i] })).filter(d => !isNaN(d.value)))
        }
      }

      // ── Bollinger Bands ────────────────────────────────────────────────
      if (effectiveShowBB && iN >= 20) {
        const BB_P = 20; const BB_M = 2
        const bbSma    = iSMA(iCloses, BB_P)
        const bbUpper  = bbSma.map((mean, i) => {
          if (isNaN(mean)) return NaN
          const slice = iCloses.slice(i - BB_P + 1, i + 1)
          const std   = Math.sqrt(slice.reduce((a, v) => a + (v - mean) ** 2, 0) / BB_P)
          return mean + BB_M * std
        })
        const bbLower  = bbSma.map((mean, i) => {
          if (isNaN(mean)) return NaN
          const slice = iCloses.slice(i - BB_P + 1, i + 1)
          const std   = Math.sqrt(slice.reduce((a, v) => a + (v - mean) ** 2, 0) / BB_P)
          return mean - BB_M * std
        })
        chart.addLineSeries({ color: 'rgba(139,92,246,0.5)', lineWidth: 1, priceLineVisible: false })
          .setData(intradayData.map((b, i) => ({ time: b.time as ITime, value: bbUpper[i] })).filter(d => !isNaN(d.value)))
        chart.addLineSeries({ color: 'rgba(139,92,246,0.5)', lineWidth: 1, priceLineVisible: false })
          .setData(intradayData.map((b, i) => ({ time: b.time as ITime, value: bbLower[i] })).filter(d => !isNaN(d.value)))
      }

      // ── RSI sub-panel ──────────────────────────────────────────────────
      if (rsiRef.current && iN >= 15) {
        const RSI_P = 14
        const rsiVals: number[] = new Array(iN).fill(NaN)
        let avgG = 0, avgL = 0
        for (let i = 1; i <= RSI_P; i++) {
          const d = iCloses[i] - iCloses[i - 1]
          if (d > 0) avgG += d; else avgL -= d
        }
        avgG /= RSI_P; avgL /= RSI_P
        rsiVals[RSI_P] = avgL === 0 ? 100 : 100 - 100 / (1 + avgG / avgL)
        for (let i = RSI_P + 1; i < iN; i++) {
          const d = iCloses[i] - iCloses[i - 1]
          avgG = (avgG * (RSI_P - 1) + Math.max(d, 0)) / RSI_P
          avgL = (avgL * (RSI_P - 1) + Math.max(-d, 0)) / RSI_P
          rsiVals[i] = avgL === 0 ? 100 : 100 - 100 / (1 + avgG / avgL)
        }
        const rsiPts = intradayData.map((b, i) => ({ time: b.time as ITime, value: rsiVals[i] })).filter(d => !isNaN(d.value))
        const rsiChart = createChart(rsiRef.current, { ...chartOpts, height: 120 })
        rsiApi.current = rsiChart
        rsiChart.addLineSeries({ color: '#a78bfa',               lineWidth: 1, priceLineVisible: false }).setData(rsiPts)
        rsiChart.addLineSeries({ color: 'rgba(239,68,68,0.4)',   lineWidth: 1, priceLineVisible: false }).setData(rsiPts.map(d => ({ ...d, value: 70 })))
        rsiChart.addLineSeries({ color: 'rgba(34,197,94,0.4)',   lineWidth: 1, priceLineVisible: false }).setData(rsiPts.map(d => ({ ...d, value: 30 })))
        rsiChart.addLineSeries({ color: 'rgba(16,185,129,0.25)', lineWidth: 1, lineStyle: LineStyle.Dashed, priceLineVisible: false }).setData(rsiPts.map(d => ({ ...d, value: 40 })))
        rsiChart.timeScale().fitContent()
        chart.timeScale().subscribeVisibleLogicalRangeChange(range => { if (range) rsiChart.timeScale().setVisibleLogicalRange(range) })
      }

      // ── MACD sub-panel ─────────────────────────────────────────────────
      if (macdRef.current && iN >= 35) {
        const ema12    = iEMA(iCloses, 12)
        const ema26    = iEMA(iCloses, 26)
        const macdLine = ema12.map((v, i) => isNaN(v) || isNaN(ema26[i]) ? NaN : v - ema26[i])
        const macdStart = macdLine.findIndex(v => !isNaN(v))
        const validMacd = macdLine.filter(v => !isNaN(v))
        const sigRaw    = iEMA(validMacd, 9)
        const sigLine   = new Array(iN).fill(NaN)
        for (let i = 0; i < sigRaw.length; i++) sigLine[macdStart + i] = sigRaw[i]

        const macdPts = intradayData
          .map((b, i) => ({ time: b.time as ITime, value: macdLine[i], sig: sigLine[i] }))
          .filter(d => !isNaN(d.value))
        const macdChart = createChart(macdRef.current, { ...chartOpts, height: 120 })
        macdApi.current = macdChart
        macdChart.addHistogramSeries({ priceLineVisible: false })
          .setData(macdPts.map(d => ({ time: d.time, value: d.value, color: d.value >= 0 ? 'rgba(34,197,94,0.6)' : 'rgba(239,68,68,0.6)' })))
        macdChart.addLineSeries({ color: '#3b82f6', lineWidth: 1, priceLineVisible: false })
          .setData(macdPts.map(d => ({ time: d.time, value: d.value })))
        macdChart.addLineSeries({ color: '#f97316', lineWidth: 1, priceLineVisible: false })
          .setData(macdPts.filter(d => !isNaN(d.sig)).map(d => ({ time: d.time, value: d.sig })))
        macdChart.timeScale().fitContent()
        chart.timeScale().subscribeVisibleLogicalRangeChange(range => { if (range) macdChart.timeScale().setVisibleLogicalRange(range) })
      }

      // ── Stochastics sub-panel ──────────────────────────────────────────
      if (effectiveShowStochastics && stoRef.current && iN >= 14) {
        const K_P = 14; const D_P = 3
        const kVals: { t: ITime; k: number }[] = []
        for (let i = K_P - 1; i < iN; i++) {
          const w  = intradayData.slice(i - K_P + 1, i + 1)
          const lo = Math.min(...w.map(b => b.low))
          const hi = Math.max(...w.map(b => b.high))
          kVals.push({ t: intradayData[i].time as ITime, k: hi === lo ? 50 : (iCloses[i] - lo) / (hi - lo) * 100 })
        }
        const stoData = kVals.map((v, i) => ({
          ...v, d: i >= D_P - 1 ? (kVals[i].k + kVals[i - 1].k + kVals[i - 2].k) / 3 : NaN,
        }))
        const stoChart = createChart(stoRef.current, { ...chartOpts, height: 100 })
        stoApi.current = stoChart
        stoChart.addLineSeries({ color: '#e2e8f0',               lineWidth: 1, priceLineVisible: false }).setData(stoData.map(d => ({ time: d.t, value: d.k })))
        stoChart.addLineSeries({ color: '#f97316',               lineWidth: 1, priceLineVisible: false }).setData(stoData.filter(d => !isNaN(d.d)).map(d => ({ time: d.t, value: d.d })))
        stoChart.addLineSeries({ color: 'rgba(239,68,68,0.4)',   lineWidth: 1, priceLineVisible: false }).setData(stoData.map(d => ({ time: d.t, value: 80 })))
        stoChart.addLineSeries({ color: 'rgba(34,197,94,0.4)',   lineWidth: 1, priceLineVisible: false }).setData(stoData.map(d => ({ time: d.t, value: 20 })))
        stoChart.addLineSeries({ color: 'rgba(16,185,129,0.25)', lineWidth: 1, lineStyle: LineStyle.Dashed, priceLineVisible: false }).setData(stoData.map(d => ({ time: d.t, value: 30 })))
        stoChart.timeScale().fitContent()
        chart.timeScale().subscribeVisibleLogicalRangeChange(range => { if (range) stoChart.timeScale().setVisibleLogicalRange(range) })
      }

      // ── Williams %R sub-panel (intraday) ────────────────────────────────
      if (willrRef.current && iN >= 14) {
        const W_P = 14
        const willrData: { time: ITime; value: number }[] = []
        for (let i = W_P - 1; i < iN; i++) {
          const w  = intradayData.slice(i - W_P + 1, i + 1)
          const hi = Math.max(...w.map(b => b.high))
          const lo = Math.min(...w.map(b => b.low))
          willrData.push({ time: intradayData[i].time as ITime, value: hi === lo ? -50 : ((hi - iCloses[i]) / (hi - lo)) * -100 })
        }
        const willrChart = createChart(willrRef.current, { ...chartOpts, height: 100 })
        willrApi.current = willrChart
        willrChart.addLineSeries({ color: '#34d399', lineWidth: 1, priceLineVisible: false }).setData(willrData)
        willrChart.addLineSeries({ color: 'rgba(239,68,68,0.35)',  lineWidth: 1, priceLineVisible: false }).setData(willrData.map(d => ({ time: d.time, value: -20 })))
        willrChart.addLineSeries({ color: 'rgba(34,197,94,0.35)',  lineWidth: 1, priceLineVisible: false }).setData(willrData.map(d => ({ time: d.time, value: -80 })))
        willrChart.addLineSeries({ color: 'rgba(156,163,175,0.1)', lineWidth: 1, priceLineVisible: false }).setData(willrData.map(d => ({ time: d.time, value: -50 })))
        willrChart.timeScale().fitContent()
        chart.timeScale().subscribeVisibleLogicalRangeChange(range => { if (range) willrChart.timeScale().setVisibleLogicalRange(range) })
      }

      // ── CCI sub-panel (intraday) ─────────────────────────────────────────
      if (cciRef.current && iN >= 20) {
        const CCI_P = 20
        const cciData: { time: ITime; value: number }[] = []
        for (let i = CCI_P - 1; i < iN; i++) {
          const w   = intradayData.slice(i - CCI_P + 1, i + 1)
          const tps = w.map(b => (b.high + b.low + b.close) / 3)
          const sma = tps.reduce((a, b) => a + b, 0) / CCI_P
          const dev = tps.reduce((a, v) => a + Math.abs(v - sma), 0) / CCI_P
          cciData.push({ time: intradayData[i].time as ITime, value: dev === 0 ? 0 : (tps[CCI_P - 1] - sma) / (0.015 * dev) })
        }
        const cciChart = createChart(cciRef.current, { ...chartOpts, height: 100 })
        cciApi.current = cciChart
        cciChart.addLineSeries({ color: '#38bdf8', lineWidth: 1, priceLineVisible: false }).setData(cciData)
        cciChart.addLineSeries({ color: 'rgba(239,68,68,0.3)',   lineWidth: 1, priceLineVisible: false }).setData(cciData.map(d => ({ time: d.time, value:  100 })))
        cciChart.addLineSeries({ color: 'rgba(34,197,94,0.3)',   lineWidth: 1, priceLineVisible: false }).setData(cciData.map(d => ({ time: d.time, value: -100 })))
        cciChart.addLineSeries({ color: 'rgba(156,163,175,0.1)', lineWidth: 1, priceLineVisible: false }).setData(cciData.map(d => ({ time: d.time, value:    0 })))
        cciChart.timeScale().fitContent()
        chart.timeScale().subscribeVisibleLogicalRangeChange(range => { if (range) cciChart.timeScale().setVisibleLogicalRange(range) })
      }

      // ── OBV sub-panel (intraday) ─────────────────────────────────────────
      if (obvRef.current && iN >= 2) {
        let obv = 0
        const obvData: { time: ITime; value: number }[] = [{ time: intradayData[0].time as ITime, value: 0 }]
        for (let i = 1; i < iN; i++) {
          if (iCloses[i] > iCloses[i - 1])      obv += intradayData[i].volume
          else if (iCloses[i] < iCloses[i - 1]) obv -= intradayData[i].volume
          obvData.push({ time: intradayData[i].time as ITime, value: obv })
        }
        const obvChart = createChart(obvRef.current, { ...chartOpts, height: 100 })
        obvApi.current = obvChart
        obvChart.addLineSeries({ color: '#fb923c', lineWidth: 1, priceLineVisible: false }).setData(obvData)
        obvChart.timeScale().fitContent()
        chart.timeScale().subscribeVisibleLogicalRangeChange(range => { if (range) obvChart.timeScale().setVisibleLogicalRange(range) })
      }

      // ── ATR sub-panel (intraday) ─────────────────────────────────────────
      if (atrPanelRef.current && iN >= 15) {
        const ATR_P = 14
        const atrData: { time: ITime; value: number }[] = []
        let atrVal = 0
        const iTRs: number[] = []
        for (let i = 1; i < iN; i++) {
          const tr = Math.max(
            intradayData[i].high - intradayData[i].low,
            Math.abs(intradayData[i].high - iCloses[i - 1]),
            Math.abs(intradayData[i].low  - iCloses[i - 1])
          )
          iTRs.push(tr)
          if (i === ATR_P) {
            atrVal = iTRs.reduce((a, b) => a + b, 0) / ATR_P
          } else if (i > ATR_P) {
            atrVal = (atrVal * (ATR_P - 1) + tr) / ATR_P
          }
          if (i >= ATR_P) atrData.push({ time: intradayData[i].time as ITime, value: atrVal })
        }
        const atrChart = createChart(atrPanelRef.current, { ...chartOpts, height: 100 })
        atrPanelApi.current = atrChart
        atrChart.addLineSeries({ color: '#e879f9', lineWidth: 1, priceLineVisible: false }).setData(atrData)
        atrChart.timeScale().fitContent()
        chart.timeScale().subscribeVisibleLogicalRangeChange(range => { if (range) atrChart.timeScale().setVisibleLogicalRange(range) })
      }

      chart.timeScale().fitContent()

      // Drawing tools integration
      chart.subscribeClick(drawingHandleClick)
      chart.subscribeCrosshairMove(drawingHandleCrosshairMove)
      attachAllPrimitives()

      return () => {
        try { chart.unsubscribeClick(drawingHandleClick) } catch {}
        try { chart.unsubscribeCrosshairMove(drawingHandleCrosshairMove) } catch {}
        try { chart.remove() } catch {}
        chartApi.current = null
        candlestickRef.current = null
      }
    }

    // ── Daily chart ────────────────────────────────────────────────────────
    if (!chartRef.current) return
    // chartData undefined means the symbol isn't in the DB yet — show "no data" UI
    if (!chartData) { setNoChartData(true); return }

    const candles = chartData.prices
      .filter(p => p.open != null)
      .map(p => ({ time: p.date, open: p.open!, high: p.high!, low: p.low!, close: p.close! }))

    if (candles.length === 0) { setNoChartData(true); return }
    setNoChartData(false)

    // autoSize:true makes lightweight-charts use the container's CSS height, not the
    // height option — the container divs have explicit heights set in JSX, so this works.
    const chart = createChart(chartRef.current, { ...chartOpts, height: 400 })
    chartApi.current = chart

    const candlestick = chart.addCandlestickSeries({
      upColor: '#22c55e', downColor: '#ef4444',
      borderUpColor: '#22c55e', borderDownColor: '#ef4444',
      wickUpColor: '#22c55e', wickDownColor: '#ef4444',
    })
    candlestickRef.current = candlestick
    candlestick.setData(candles)

    const volumeSeries = chart.addHistogramSeries({ priceFormat: { type: 'volume' }, priceScaleId: 'vol' })
    chart.priceScale('vol').applyOptions({ scaleMargins: { top: 0.85, bottom: 0 } })
    volumeSeries.setData(
      chartData.prices
        .filter(p => p.volume != null)
        .map(p => ({
          time: p.date,
          value: p.volume!,
          color: (p.close ?? 0) >= (p.open ?? 0) ? 'rgba(34,197,94,0.3)' : 'rgba(239,68,68,0.3)',
        }))
    )

    // ── Gap detection ──────────────────────────────────────────────────────
    if (effectiveShowGaps) {
      const MIN_GAP_PCT = 0.002  // 0.2% minimum gap size
      const SCAN_BARS = Math.min(candles.length, 252)
      const gaps: Array<{ top: number; bottom: number; type: 'up' | 'down'; filled: boolean }> = []
      const recent = candles.slice(-SCAN_BARS)

      for (let i = 1; i < recent.length; i++) {
        const prev = recent[i - 1]
        const curr = recent[i]
        // Gap up: prev high < curr low
        if (curr.low > prev.high && (curr.low - prev.high) / prev.high >= MIN_GAP_PCT) {
          const filled = recent.slice(i + 1).some(c => c.low <= prev.high)
          gaps.push({ top: curr.low, bottom: prev.high, type: 'up', filled })
        }
        // Gap down: prev low > curr high
        if (curr.high < prev.low && (prev.low - curr.high) / prev.low >= MIN_GAP_PCT) {
          const filled = recent.slice(i + 1).some(c => c.high >= prev.low)
          gaps.push({ top: prev.low, bottom: curr.high, type: 'down', filled })
        }
      }

      // Show last 8 unfilled + last 4 filled (dimmer)
      const openGaps  = gaps.filter(g => !g.filled).slice(-8)
      const closedGaps = gaps.filter(g => g.filled).slice(-4)

      for (const gap of [...closedGaps, ...openGaps]) {
        const rgb    = gap.type === 'up' ? '34,197,94' : '239,68,68'
        const alpha  = gap.filled ? 0.18 : 0.45
        const label  = gap.filled ? '' : `${gap.type === 'up' ? '↑' : '↓'} Gap ${((gap.top - gap.bottom) / gap.bottom * 100).toFixed(1)}%`
        candlestick.createPriceLine({
          price: gap.top, color: `rgba(${rgb},${alpha})`,
          lineStyle: LineStyle.Dashed, lineWidth: 1,
          axisLabelVisible: !gap.filled, title: label,
        })
        candlestick.createPriceLine({
          price: gap.bottom, color: `rgba(${rgb},${alpha})`,
          lineStyle: LineStyle.Dashed, lineWidth: 1,
          axisLabelVisible: false, title: '',
        })
      }
    } // end gap detection

    if (effectiveShowMA && chartData.indicators.length) {
      const ma50 = chart.addLineSeries({ color: '#3b82f6', lineWidth: 1, priceLineVisible: false })
      ma50.setData(chartData.indicators.filter(i => i.ma50 != null).map(i => ({ time: i.date, value: i.ma50! })))
      const ma200 = chart.addLineSeries({ color: '#f59e0b', lineWidth: 1, priceLineVisible: false })
      ma200.setData(chartData.indicators.filter(i => i.ma200 != null).map(i => ({ time: i.date, value: i.ma200! })))
    }

    if (effectiveShowBB && chartData.indicators.length) {
      const bbUpper = chart.addLineSeries({ color: 'rgba(139,92,246,0.5)', lineWidth: 1, priceLineVisible: false })
      bbUpper.setData(chartData.indicators.filter(i => i.bb_upper != null).map(i => ({ time: i.date, value: i.bb_upper! })))
      const bbLower = chart.addLineSeries({ color: 'rgba(139,92,246,0.5)', lineWidth: 1, priceLineVisible: false })
      bbLower.setData(chartData.indicators.filter(i => i.bb_lower != null).map(i => ({ time: i.date, value: i.bb_lower! })))
    }

    // ── Trendlines ─────────────────────────────────────────────────────────
    if (effectiveShowTrendlines && trendlineData && trendlineData[trendlineTF]) {
      const tfData = trendlineData[trendlineTF]
      for (const line of tfData.resistance.slice(-3)) {
        const pts = [...line.points]; if (line.projected) pts.push(line.projected)
        if (pts.length >= 2) {
          chart.addLineSeries({ color: 'rgba(239,68,68,0.7)', lineWidth: 1, lineStyle: LineStyle.Dashed, priceLineVisible: false, crosshairMarkerVisible: false, lastValueVisible: false })
            .setData(pts.map(p => ({ time: p.date as import('lightweight-charts').Time, value: p.value })))
        }
      }
      for (const line of tfData.support.slice(-3)) {
        const pts = [...line.points]; if (line.projected) pts.push(line.projected)
        if (pts.length >= 2) {
          chart.addLineSeries({ color: 'rgba(34,197,94,0.7)', lineWidth: 1, lineStyle: LineStyle.Dashed, priceLineVisible: false, crosshairMarkerVisible: false, lastValueVisible: false })
            .setData(pts.map(p => ({ time: p.date as import('lightweight-charts').Time, value: p.value })))
        }
      }
    }


    // ── Pattern layer — rich visual rendering for selected pattern ────────
    if (showPatterns && activePatternName && patterns && patterns.length > 0) {
      const eligible = patterns.filter(p =>
        p.status != null && p.status !== 'NOT_PRESENT' && p.status !== 'FAILED'
      )
      const selected: PatternEntry[] = []
      const active = eligible.find(p => p.pattern_name === activePatternName)
      if (active) selected.push(active)

      // Shared time converter: date string (primary) or legacy numeric index
      const toTime = (coord: unknown): import('lightweight-charts').Time | null => {
        if (typeof coord === 'string') return coord as import('lightweight-charts').Time
        const idx = Math.max(0, Math.min(candles.length - 1, Math.round(Number(coord))))
        return (candles[idx]?.time ?? null) as import('lightweight-charts').Time | null
      }

      // Unified role → style map (covers all patterns)
      type SegStyle = { color: string; dashed?: boolean; width?: number }
      const ROLE_STYLES: Record<string, SegStyle> = {
        // Chair Pattern — local structure
        impulse:         { color: '#22c55e', width: 2 },
        pullback:        { color: '#f97316', width: 2 },
        recovery:        { color: '#86efac', dashed: true, width: 1 },
        box_edge:        { color: 'rgba(107,114,128,0.2)', width: 1 },
        target:          { color: '#a855f7', dashed: true, width: 2 },
        // Chair Pattern — full-chart macro channel trendlines
        channel_upper:   { color: '#f59e0b', width: 2 },
        channel_lower:   { color: '#f59e0b', width: 2 },
        channel_lower_2: { color: 'rgba(245,158,11,0.45)', width: 1 },
        channel_lower_3: { color: 'rgba(245,158,11,0.25)', width: 1 },
        // Cup & Handle
        cup_arc:         { color: '#14b8a6', width: 3 },
        handle_pullback: { color: '#f97316', width: 2 },
        handle_recovery: { color: '#22c55e', dashed: true, width: 2 },
        // Rounding Bottom/Top
        arc:             { color: '#14b8a6', width: 2 },
        // Shared: necklines
        neckline:        { color: 'rgba(251,191,36,0.85)', width: 2 },
        neckline_ext:    { color: 'rgba(251,191,36,0.6)',  width: 2 },
        // Flag / Pennant pole
        pole:            { color: '#94a3b8', width: 2 },
        // Shared: trendlines (also used by wedges)
        resistance:      { color: '#f87171', width: 2 },
        support:         { color: '#4ade80', width: 2 },
        // H&S / IH&S segments
        ls_down:         { color: '#94a3b8', width: 1 },
        ls_up:           { color: '#94a3b8', width: 1 },
        head_up:         { color: '#ef4444', width: 2 },
        head_down:       { color: '#ef4444', width: 2 },
        rs_up:           { color: '#94a3b8', width: 1 },
        rs_down:         { color: '#94a3b8', width: 1 },
        // Double/Triple Top/Bottom
        peak_up:         { color: '#ef4444', width: 2 },
        peak_down:       { color: '#ef4444', width: 2 },
        bottom_up:       { color: '#22c55e', width: 2 },
        bottom_down:     { color: '#22c55e', width: 2 },
        // Wedge-specific (also covered by resistance/support but explicit aliases)
        wedge_upper:     { color: '#f87171', width: 2 },
        wedge_lower:     { color: '#4ade80', width: 2 },
        // Harmonic XABCD legs (colour is overridden per-direction in rendering logic)
        xa:              { color: '#14b8a6', width: 2 },
        ab:              { color: '#14b8a6', width: 2 },
        bc:              { color: '#14b8a6', width: 2 },
        cd:              { color: '#14b8a6', width: 2 },
        // Internal reference diagonals (A→C and X→D) — thin dashed teal
        harm_ref_ac:     { color: 'rgba(20,184,166,0.45)', dashed: true, width: 1 },
        harm_ref_xd:     { color: 'rgba(20,184,166,0.25)', dashed: true, width: 1 },
      }

      for (const p of selected) {
        const kind = classifyPattern(p)
        const color = patternColor(kind, p.direction)

        // ── Overlay line segments ─────────────────────────────────────────
        const rawLines = (p.overlay_lines ?? []) as [unknown, number][][]

        const renderSeg = (a: unknown[], b: unknown[], style: SegStyle) => {
          const t1 = toTime(a[0]), t2 = toTime(b[0])
          if (!t1 || !t2) return
          const [lo, hi] = String(t1) <= String(t2)
            ? [{ time: t1, value: Number(a[1]) }, { time: t2, value: Number(b[1]) }]
            : [{ time: t2, value: Number(b[1]) }, { time: t1, value: Number(a[1]) }]
          if (String(lo.time) === String(hi.time)) return
          chart.addLineSeries({
            color: style.color,
            lineWidth: (style.width ?? 2) as import('lightweight-charts').LineWidth,
            lineStyle: style.dashed ? LineStyle.Dashed : LineStyle.Solid,
            priceLineVisible: false, crosshairMarkerVisible: false, lastValueVisible: false,
          }).setData([lo, hi])
        }

        // ── Harmonic patterns: XABCD structure in StockCharts style ──────────
        // 4 solid zigzag legs + X→B and B→D diagonals form the M/W skeleton.
        // PRZ shaded band + Target/Stop lines extend from D to the current bar
        // so the trader can see where price sits relative to the reversal zone.
        if (kind === 'harmonic' && p.points && p.points.length >= 2) {
          const hPts     = p.points as unknown[][]
          const isBullish = p.direction === 'bullish'
          const pR       = p as unknown as Record<string, unknown>

          const HARM_PALETTE: Record<string, string> = {
            'Gartley':   '#14b8a6',
            'Bat':       '#3b82f6',
            'Butterfly': '#a855f7',
            'Crab':      '#f97316',
            'Cypher':    '#eab308',
          }
          const harmColor  = HARM_PALETTE[p.pattern_name] ?? '#14b8a6'
          const transparent = 'rgba(0,0,0,0)'

          // Helper — draw a 2-point line in ascending time order
          const hLine = (
            t1: import('lightweight-charts').Time | null,
            v1: number,
            t2: import('lightweight-charts').Time | null,
            v2: number,
            color: string,
            width: 1 | 2 | 3 | 4,
            style: LineStyle,
          ) => {
            if (!t1 || !t2 || String(t1) === String(t2)) return
            const [lo, hi] = String(t1) <= String(t2)
              ? [{ time: t1, value: v1 }, { time: t2, value: v2 }]
              : [{ time: t2, value: v2 }, { time: t1, value: v1 }]
            chart.addLineSeries({
              color, lineWidth: width, lineStyle: style,
              priceLineVisible: false, crosshairMarkerVisible: false, lastValueVisible: false,
            }).setData([lo, hi])
          }

          if (hPts.length >= 5) {
            const prices  = hPts.slice(0, 5).map(pt => Number(pt[1]))
            const times   = hPts.slice(0, 5).map(pt => toTime(pt[0]))
            const [pX, pA, pB, pC, pD] = prices
            const [tX, tA, tB, tC, tD] = times
            const todayStr = candles[candles.length - 1]?.time as import('lightweight-charts').Time | undefined

            // ── 4 solid XABCD zigzag legs ──────────────────────────────────
            hLine(tX, pX, tA, pA, harmColor, 2, LineStyle.Solid)
            hLine(tA, pA, tB, pB, harmColor, 2, LineStyle.Solid)
            hLine(tB, pB, tC, pC, harmColor, 2, LineStyle.Solid)
            hLine(tC, pC, tD, pD, harmColor, 2, LineStyle.Solid)

            // ── X→B dashed diagonal — left "wing" of the M/W ───────────────
            hLine(tX, pX, tB, pB, harmColor + '66', 1, LineStyle.Dashed)

            // ── B→D dashed diagonal — right "wing" of the M/W ──────────────
            hLine(tB, pB, tD, pD, harmColor + '66', 1, LineStyle.Dashed)

            // ── PRZ shaded band from D to current bar ──────────────────────
            // Renders a coloured rectangle so the trader can see if price is
            // inside, above, or below the Potential Reversal Zone right now.
            const przLow   = pR.prz_low            as number | undefined
            const przHigh  = pR.prz_high           as number | undefined
            const tgtPrice = pR.target             as number | undefined
            const stopPrice= pR.invalidation_level as number | undefined

            const dPastToday = tD && todayStr && String(tD) < String(todayStr)

            if (przLow && przHigh && przLow > 0 && przHigh > 0) {
              if (dPastToday && tD && todayStr) {
                // Filled band: baseline at przLow, data at przHigh → fills the zone
                try {
                  chart.addBaselineSeries({
                    baseValue: { type: 'price', price: przLow },
                    topLineColor: harmColor + 'cc',
                    topFillColor1: harmColor + '40',
                    topFillColor2: harmColor + '18',
                    bottomLineColor: harmColor + '66',
                    bottomFillColor1: transparent,
                    bottomFillColor2: transparent,
                    priceLineVisible: false, crosshairMarkerVisible: false, lastValueVisible: false,
                  }).setData([
                    { time: tD,       value: przHigh },
                    { time: todayStr, value: przHigh },
                  ])
                } catch (_) { /* degenerate range */ }
              }
              // Axis labels regardless of extension
              candlestick.createPriceLine({ price: przHigh, color: harmColor,   lineWidth: 1, lineStyle: LineStyle.Dashed, axisLabelVisible: true, title: 'PRZ ▲' })
              candlestick.createPriceLine({ price: przLow,  color: harmColor,   lineWidth: 1, lineStyle: LineStyle.Dashed, axisLabelVisible: true, title: 'PRZ ▼' })
            }

            // ── Target line from D to right edge ───────────────────────────
            if (tgtPrice && tgtPrice > 0) {
              if (dPastToday && tD && todayStr) {
                hLine(tD, tgtPrice, todayStr, tgtPrice, '#22c55e99', 1, LineStyle.Dotted)
              }
              candlestick.createPriceLine({ price: tgtPrice, color: '#22c55e', lineWidth: 1, lineStyle: LineStyle.Dotted, axisLabelVisible: true, title: 'Target' })
            }

            // ── Stop line from D to right edge ─────────────────────────────
            if (stopPrice && stopPrice > 0) {
              if (dPastToday && tD && todayStr) {
                hLine(tD, stopPrice, todayStr, stopPrice, '#ef444499', 1, LineStyle.Dotted)
              }
              candlestick.createPriceLine({ price: stopPrice, color: '#ef4444', lineWidth: 1, lineStyle: LineStyle.Dotted, axisLabelVisible: true, title: 'Stop' })
            }

            // ── D-level dotted line to right edge (shows D completion price) ─
            if (dPastToday && tD && todayStr) {
              hLine(tD, pD, todayStr, pD, harmColor + '55', 1, LineStyle.Dotted)
            }

            // ── Fit chart to show complete pattern from X to today ──────────
            // Ensures all 5 pivots + current price are visible simultaneously
            if (tX && todayStr) {
              const xStr    = String(tX)
              const todStr  = String(todayStr)
              const padding = Math.max(
                Math.round((candles.length * 0.03)),  // 3% right margin
                2
              )
              const xIdx = candles.findIndex(c => String(c.time) >= xStr)
              if (xIdx >= 0) {
                const fromTime = candles[Math.max(0, xIdx - 2)]?.time ?? tX
                const toIdx    = candles.length - 1 + padding
                const toTime_  = candles[Math.min(toIdx, candles.length - 1)]?.time ?? todayStr
                // Only adjust if current visible range would hide X
                chart.timeScale().setVisibleRange({
                  from: fromTime as import('lightweight-charts').Time,
                  to:   toTime_  as import('lightweight-charts').Time,
                })
              }
            }
          }

        } else if (rawLines.length > 0) {
          // ── Generic overlay lines (chart patterns, Wyckoff, Gann) ───────
          // Gann fan: 9 lines in steepest→shallowest order (8x1, 4x1, 3x1, 2x1,
          // 1x1, 1x2, 1x3, 1x4, 1x8) — each gets a distinct colour.
          const GANN_FAN_PALETTE = [
            '#22c55e',  // 8x1 — bright green  (very steep bullish)
            '#4ade80',  // 4x1 — light green
            '#84cc16',  // 3x1 — lime
            '#bef264',  // 2x1 — yellow-green
            '#fbbf24',  // 1x1 — amber / gold  (45° master line)
            '#f97316',  // 1x2 — orange
            '#ef4444',  // 1x3 — red
            '#dc2626',  // 1x4 — darker red
            '#991b1b',  // 1x8 — dark red      (very shallow / bearish support)
          ]
          for (let i = 0; i < rawLines.length; i++) {
            const seg = rawLines[i]
            if (!Array.isArray(seg) || seg.length < 2) continue
            const a = seg[0] as unknown[], b = seg[1] as unknown[]
            if (!Array.isArray(a) || !Array.isArray(b) || a.length < 2 || b.length < 2) continue
            let style: SegStyle = { color, width: 2 }
            const role = p.overlay_line_roles?.[i]
            if (kind === 'wyckoff') {
              style = i === 0 ? { color: 'rgba(168,85,247,0.7)', width: 2 }
                     : i === 1 ? { color: 'rgba(168,85,247,0.5)', width: 2 }
                     : { color: 'rgba(168,85,247,0.2)', dashed: true, width: 1 }
            } else if (kind === 'gann') {
              // 1x1 (index 4) is the master line — draw thicker + fully opaque
              const is1x1 = i === 4
              const fanColor = GANN_FAN_PALETTE[i] ?? '#fbbf24'
              style = {
                color: is1x1 ? fanColor : fanColor + 'cc',   // 80% opacity for non-master lines
                width: is1x1 ? 2 : 1,
                dashed: i > 4,   // shallow angles (1x2 … 1x8) shown dashed
              }
            } else if (kind === 'chart') {
              // Chart patterns: unified bold color, all lines solid (no dashes).
              const isNeck = role === 'neckline' || role === 'neckline_ext'
              const isPole = role === 'pole'
              style = {
                color: isNeck ? 'rgba(251,191,36,0.85)' : color,
                width: isNeck ? 2 : isPole ? 2 : 3,
              }
            } else if (role && ROLE_STYLES[role]) {
              style = ROLE_STYLES[role]
            }

            // Flat horizontal support/resistance → full-width price line so
            // the level extends all the way across the visible chart.
            // Detect "flat" as price delta < 0.3 % of mean price.
            const aPrice = Number(a[1]), bPrice = Number(b[1])
            const meanPx  = (Math.abs(aPrice) + Math.abs(bPrice)) / 2
            const isFlat  = meanPx > 0 && Math.abs(bPrice - aPrice) / meanPx < 0.003
            if (isFlat && (role === 'resistance' || role === 'support')) {
              continue  // skip flat S/R price lines — show pattern shape only
            }
            renderSeg(a, b, style)
          }

          // ── Gann bearish fan from swing high (red gradient descending lines) ─
          if (kind === 'gann') {
            const bearLines = (p as unknown as Record<string, unknown>).bearish_overlay_lines as
              [[string, number], [string, number]][] | undefined
            if (bearLines?.length) {
              const BEAR_FAN_PALETTE = [
                '#991b1b',  // 8x1 bear — dark red (steepest descent)
                '#dc2626',  // 4x1 bear
                '#ef4444',  // 3x1 bear
                '#f87171',  // 2x1 bear
                '#fca5a5',  // 1x1 bear — master descending line (light red/pink)
                '#f97316',  // 1x2 bear — orange (shallow descent)
                '#fb923c',  // 1x3 bear
                '#fcd34d',  // 1x4 bear — gold
                '#fef08a',  // 1x8 bear — near-flat, very shallow
              ]
              bearLines.forEach((seg, i) => {
                if (!Array.isArray(seg) || seg.length < 2) return
                const a = seg[0] as unknown[], b = seg[1] as unknown[]
                if (!Array.isArray(a) || !Array.isArray(b)) return
                const is1x1 = i === 4
                const bearColor = BEAR_FAN_PALETTE[i] ?? '#ef4444'
                renderSeg(a, b, {
                  color: is1x1 ? bearColor : bearColor + '99',
                  width: is1x1 ? 2 : 1,
                  dashed: i < 4,   // steeper-than-master bear lines are dashed
                })
              })
            }
          }

          // ── Shaded fill between two trendlines (channels, wedges, triangles)
          const fillZone = (p as unknown as Record<string, unknown>).fill_zone as
            { upper: number; lower: number; color: string; opacity: number } | undefined
          if (fillZone != null && candlestick) {
            const upperSeg = rawLines[fillZone.upper]
            const lowerSeg = rawLines[fillZone.lower]
            if (
              Array.isArray(upperSeg) && upperSeg.length >= 2 &&
              Array.isArray(lowerSeg) && lowerSeg.length >= 2
            ) {
              const ua = upperSeg[0] as unknown[], ub = upperSeg[1] as unknown[]
              const la = lowerSeg[0] as unknown[], lb = lowerSeg[1] as unknown[]
              if (
                Array.isArray(ua) && ua.length >= 2 && Array.isArray(ub) && ub.length >= 2 &&
                Array.isArray(la) && la.length >= 2 && Array.isArray(lb) && lb.length >= 2
              ) {
                const upperEndpts: [string, number][] = [
                  [String(ua[0]), Number(ua[1])],
                  [String(ub[0]), Number(ub[1])],
                ]
                const lowerEndpts: [string, number][] = [
                  [String(la[0]), Number(la[1])],
                  [String(lb[0]), Number(lb[1])],
                ]
                try {
                  const fillPrim = new ChannelFillPrimitive(
                    upperEndpts, lowerEndpts,
                    fillZone.color, fillZone.opacity,
                  )
                  candlestick.attachPrimitive(fillPrim)
                } catch { /* skip if attach fails */ }
              }
            }
          }

        } else if (p.points && p.points.length >= 2) {
          // Fallback: join consecutive points as connected line
          const pts: { time: import('lightweight-charts').Time; value: number }[] = []
          const seenDates = new Set<string>()
          for (const pt of (p.points as unknown[][])) {
            const t = toTime(pt[0])
            if (!t) continue
            const ds = String(t)
            if (seenDates.has(ds)) continue
            seenDates.add(ds)
            pts.push({ time: t, value: Number(pt[1]) })
          }
          pts.sort((a, b) => String(a.time).localeCompare(String(b.time)))
          if (pts.length >= 2) {
            chart.addLineSeries({
              color, lineWidth: 2, lineStyle: LineStyle.Solid,
              priceLineVisible: false, crosshairMarkerVisible: false, lastValueVisible: false,
            }).setData(pts)
          }
        }

        // ── Target + Stop projection for chart patterns ───────────────────
        if (kind === 'chart') {
          const pAny = p as unknown as Record<string, unknown>
          const tgt = pAny.projected_target as number | undefined
          const inv = pAny.invalidation_level as number | undefined
          if (tgt && tgt > 0) {
            candlestick.createPriceLine({
              price: tgt,
              color: p.direction === 'bullish' ? '#22c55e' : '#ef4444',
              lineWidth: 1,
              lineStyle: LineStyle.Dotted,
              axisLabelVisible: true,
              title: 'Target',
            })
          }
          if (inv && inv > 0) {
            candlestick.createPriceLine({
              price: inv,
              color: 'rgba(239,68,68,0.45)',
              lineWidth: 1,
              lineStyle: LineStyle.Dotted,
              axisLabelVisible: true,
              title: 'Stop',
            })
          }
        }

        // Gann retracement / SQ9 price lines removed — show fan lines only
      }

      // ── Measured move lines + Bradley Siderograph ─────────────────────────
      if (!isHarmonicActive && patterns) {
        const toTimeSimple = (coord: unknown): import('lightweight-charts').Time | null => {
          if (typeof coord === 'string') return coord as import('lightweight-charts').Time
          const idx = Math.max(0, Math.min(candles.length - 1, Math.round(Number(coord))))
          return (candles[idx]?.time ?? null) as import('lightweight-charts').Time | null
        }

        for (const p of patterns) {
          if (p.status === 'NOT_PRESENT' || p.status === 'FAILED') continue
          const kind = classifyPattern(p)
          const dir = p.direction

          if (kind === 'measured_move' && p.overlay_lines?.length) {
            for (const seg of p.overlay_lines) {
              if (!Array.isArray(seg) || seg.length < 2) continue
              const a = seg[0] as unknown[], b = seg[1] as unknown[]
              if (!Array.isArray(a) || !Array.isArray(b) || a.length < 2 || b.length < 2) continue
              const t1 = toTimeSimple(a[0]), t2 = toTimeSimple(b[0])
              if (!t1 || !t2 || String(t1) === String(t2)) continue
              const [lo, hi] = String(t1) <= String(t2)
                ? [{ time: t1, value: Number(a[1]) }, { time: t2, value: Number(b[1]) }]
                : [{ time: t2, value: Number(b[1]) }, { time: t1, value: Number(a[1]) }]
              chart.addLineSeries({
                color: dir === 'bearish' ? '#f87171' : '#4ade80',
                lineWidth: 2,
                lineStyle: LineStyle.Solid,
                priceLineVisible: false, crosshairMarkerVisible: false, lastValueVisible: false,
              }).setData([lo, hi])
            }
          } else if (kind === 'astro' && p.pattern_name === 'Bradley Siderograph' && (p.points?.length ?? 0) >= 2) {
            const bradleyPts = (p.points as unknown[][])
              .map(pt => {
                const t = toTimeSimple(pt[0])
                return t ? { time: t, value: Number(pt[1]) } : null
              })
              .filter(Boolean) as { time: import('lightweight-charts').Time; value: number }[]
            if (bradleyPts.length >= 2) {
              bradleyPts.sort((a, b) => String(a.time).localeCompare(String(b.time)))
              chart.addLineSeries({
                color: 'rgba(251,191,36,0.45)',
                lineWidth: 1,
                lineStyle: LineStyle.Dashed,
                priceLineVisible: false,
                crosshairMarkerVisible: false,
                lastValueVisible: false,
              }).setData(bradleyPts)
            }
          }
        }
      }

    }

    // ── Stage 5 — Clustered price-level zones ──────────────────────────────
    if (effectiveShowPriceZones && priceLevels && priceLevels.zones) {
      const ZONE_STYLES: Record<string, { color: string; title: string }> = {
        buy_zone:     { color: 'rgba(34,197,94,0.65)',   title: 'Buy Zone'     },
        accumulate:   { color: 'rgba(96,165,250,0.55)',  title: 'Accumulate'   },
        stop:         { color: 'rgba(248,113,113,0.55)', title: 'Stop'         },
        scale_in:     { color: 'rgba(167,139,250,0.60)', title: 'Scale In'     },
        target:       { color: 'rgba(134,239,172,0.60)', title: 'Target'       },
        distribution: { color: 'rgba(52,211,153,0.55)',  title: 'Distribution' },
      }
      for (const [key, style] of Object.entries(ZONE_STYLES)) {
        const zone = priceLevels.zones[key as keyof typeof priceLevels.zones]
        if (zone && zone.price > 0) {
          candlestick.createPriceLine({
            price: zone.price,
            color: style.color,
            lineWidth: 1,
            lineStyle: LineStyle.Dashed,
            axisLabelVisible: true,
            title: `${style.title} ·${zone.strength}/10`,
          })
        }
      }
    }

    // ── Liquidity layer (max 3 zones) ──────────────────────────────────────
    let liquidityTargets: number[] = []
    if (effectiveShowLiquidityMap && candles.length > 0) {
      const lastClose = candles[candles.length - 1].close
      const levelCandidates = [
        rec?.trade_plan?.buy_price,
        rec?.trade_plan?.accumulate_price,
        rec?.trade_plan?.scale_price,
        rec?.trade_plan?.sell_price,
        ...((patterns || []).filter(p => p.price_level != null).map(p => p.price_level as number)),
      ].filter((v): v is number => typeof v === 'number' && Number.isFinite(v))

      const merged: number[] = []
      const tolerance = lastClose * 0.003
      for (const lv of levelCandidates.sort((a, b) => Math.abs(a - lastClose) - Math.abs(b - lastClose))) {
        const tooClose = merged.some(m => Math.abs(m - lv) <= tolerance)
        if (!tooClose) merged.push(lv)
        if (merged.length >= MAX_LIQUIDITY_ZONES) break
      }
      liquidityTargets = merged

      merged.forEach((level, idx) => {
        candlestick.createPriceLine({
          price: level,
          color: 'rgba(168,85,247,0.75)',
          lineStyle: LineStyle.Dashed,
          lineWidth: 1,
          axisLabelVisible: true,
          title: `LQ-${idx + 1}`,
        })
      })
    }

    // ── Add Zones (position building signals) ──────────────────────────────
    if (effectiveShowAddZones && chartData.indicators.length) {
      const priceByDate = new Map(chartData.prices.map(p => [p.date, p]))
      const addData = chartData.indicators
        .map(ind => {
          const p = priceByDate.get(ind.date)
          if (!p?.close) return null
          const k = stochMap.get(ind.date) ?? 50
          const score =
            ((ind.rsi ?? 50) < 40 ? 1 : 0) +
            (k < 30 ? 1 : 0) +
            (ind.bb_lower != null && p.close! <= ind.bb_lower * 1.03 ? 1 : 0) +
            (p.close! < (p.open ?? p.close!) ? 1 : 0)
          if (score < 1) return null
          return {
            time: ind.date,
            value: score,
            color: score >= 3 ? 'rgba(16,185,129,0.7)' : score === 2 ? 'rgba(245,158,11,0.5)' : 'rgba(107,114,128,0.2)',
          }
        })
        .filter((d): d is NonNullable<typeof d> => d !== null)
      if (addData.length > 0) {
        const addSeries = chart.addHistogramSeries({ priceFormat: { type: 'volume' }, priceScaleId: 'add' })
        chart.priceScale('add').applyOptions({ scaleMargins: { top: 0.93, bottom: 0 } })
        addSeries.setData(addData)
      }
    }

    // ── Forecast layer (Monte Carlo style path + confidence band) ─────────
    if (effectiveShowMonteCarloForecast && chartData.indicators.length > 0 && candles.length > 0) {
      const lastClose = candles[candles.length - 1].close
      const lastAtr = chartData.indicators.filter(i => i.atr != null).slice(-1)[0]?.atr
      const baseAtrPct = lastAtr ? (lastAtr / lastClose) * 100 : 2
      const prob = signalForSymbol?.probability ?? rec?.components?.xgb_probability ?? 0.5
      const expectedPct = Math.max(1, Math.min(15, baseAtrPct * (0.8 + prob)))
      const upper = lastClose * (1 + expectedPct / 100)
      const lower = lastClose * (1 - expectedPct / 100)
      const vol = (expectedPct / 100) / 6
      const drift = (prob - 0.5) * 0.04
      const steps = 30
      const lastDate = new Date(`${candles[candles.length - 1].time as string}T00:00:00Z`)
      const medianSeries = chart.addLineSeries({ color: 'rgba(59,130,246,0.95)', lineWidth: 2, priceLineVisible: false, lastValueVisible: false })
      const upperSeries = chart.addLineSeries({ color: 'rgba(59,130,246,0.45)', lineWidth: 1, lineStyle: LineStyle.Dashed, priceLineVisible: false, lastValueVisible: false })
      const lowerSeries = chart.addLineSeries({ color: 'rgba(59,130,246,0.45)', lineWidth: 1, lineStyle: LineStyle.Dashed, priceLineVisible: false, lastValueVisible: false })

      const medPath: { time: string; value: number }[] = []
      const upPath: { time: string; value: number }[] = []
      const lowPath: { time: string; value: number }[] = []
      for (let step = 1; step <= steps; step++) {
        const t = step / 252
        const d = new Date(lastDate)
        d.setUTCDate(lastDate.getUTCDate() + step)
        const time = d.toISOString().slice(0, 10)
        const median = lastClose * Math.exp((drift - 0.5 * vol * vol) * t)
        const sigma = vol * Math.sqrt(t) * 1.28
        medPath.push({ time, value: median })
        upPath.push({ time, value: median * Math.exp(sigma) })
        lowPath.push({ time, value: median * Math.exp(-sigma) })
      }
      medianSeries.setData(medPath)
      upperSeries.setData(upPath)
      lowerSeries.setData(lowPath)

      candlestick.createPriceLine({
        price: upper,
        color: 'rgba(59,130,246,0.7)',
        lineStyle: LineStyle.Dashed,
        lineWidth: 1,
        axisLabelVisible: true,
        title: 'MC Upper',
      })
      candlestick.createPriceLine({
        price: lower,
        color: 'rgba(59,130,246,0.7)',
        lineStyle: LineStyle.Dashed,
        lineWidth: 1,
        axisLabelVisible: true,
        title: 'MC Lower',
      })
      liquidityTargets.forEach((target, idx) => {
        candlestick.createPriceLine({
          price: target,
          color: 'rgba(245,158,11,0.8)',
          lineStyle: LineStyle.Solid,
          lineWidth: 1,
          axisLabelVisible: true,
          title: `Target ${idx + 1}`,
        })
      })
    }

    chart.timeScale().fitContent()

    // ── RSI sub-panel ───────────────────────────────────────────────────────
    if (rsiRef.current) {
      const rsiChart = createChart(rsiRef.current, { ...chartOpts, height: 120 })
      rsiApi.current = rsiChart
      rsiChart.addLineSeries({ color: '#a78bfa', lineWidth: 1, priceLineVisible: false })
        .setData(chartData.indicators.filter(i => i.rsi != null).map(i => ({ time: i.date, value: i.rsi! })))
      rsiChart.addLineSeries({ color: 'rgba(239,68,68,0.4)', lineWidth: 1, priceLineVisible: false })
        .setData(chartData.indicators.filter(i => i.rsi != null).map(i => ({ time: i.date, value: 70 })))
      rsiChart.addLineSeries({ color: 'rgba(34,197,94,0.4)', lineWidth: 1, priceLineVisible: false })
        .setData(chartData.indicators.filter(i => i.rsi != null).map(i => ({ time: i.date, value: 30 })))
      // Add zone at RSI 40 (buy threshold)
      rsiChart.addLineSeries({ color: 'rgba(16,185,129,0.25)', lineWidth: 1, lineStyle: LineStyle.Dashed, priceLineVisible: false })
        .setData(chartData.indicators.filter(i => i.rsi != null).map(i => ({ time: i.date, value: 40 })))
      rsiChart.timeScale().fitContent()
      chart.timeScale().subscribeVisibleLogicalRangeChange(range => {
        if (range) rsiChart.timeScale().setVisibleLogicalRange(range)
      })
    }

    // ── MACD sub-panel ──────────────────────────────────────────────────────
    if (macdRef.current) {
      const macdChart = createChart(macdRef.current, { ...chartOpts, height: 120 })
      macdApi.current = macdChart
      const histSeries = macdChart.addHistogramSeries({ priceLineVisible: false })
      histSeries.setData(
        chartData.indicators.filter(i => i.macd_histogram != null).map(i => ({
          time: i.date, value: i.macd_histogram!,
          color: i.macd_histogram! >= 0 ? 'rgba(34,197,94,0.6)' : 'rgba(239,68,68,0.6)',
        }))
      )
      macdChart.addLineSeries({ color: '#3b82f6', lineWidth: 1, priceLineVisible: false })
        .setData(chartData.indicators.filter(i => i.macd_line != null).map(i => ({ time: i.date, value: i.macd_line! })))
      macdChart.addLineSeries({ color: '#f97316', lineWidth: 1, priceLineVisible: false })
        .setData(chartData.indicators.filter(i => i.macd_signal != null).map(i => ({ time: i.date, value: i.macd_signal! })))
      macdChart.timeScale().fitContent()
      chart.timeScale().subscribeVisibleLogicalRangeChange(range => {
        if (range) macdChart.timeScale().setVisibleLogicalRange(range)
      })
    }

    // ── Stochastics (%K/%D) ─────────────────────────────────────────────────
    if (effectiveShowStochastics && stoRef.current && chartData.prices.length >= 14) {
      const prices = chartData.prices.filter(p => p.close != null && p.high != null && p.low != null)
      const K_PERIOD = 14; const D_PERIOD = 3

      const kValues: { date: string; k: number }[] = []
      for (let i = K_PERIOD - 1; i < prices.length; i++) {
        const w = prices.slice(i - K_PERIOD + 1, i + 1)
        const lo = Math.min(...w.map(p => p.low!))
        const hi = Math.max(...w.map(p => p.high!))
        kValues.push({ date: prices[i].date, k: hi === lo ? 50 : (prices[i].close! - lo) / (hi - lo) * 100 })
      }
      const stoData = kValues.map((v, i) => ({
        ...v,
        d: i >= D_PERIOD - 1 ? (kValues[i].k + kValues[i - 1].k + kValues[i - 2].k) / 3 : null,
      }))

      const stoChart = createChart(stoRef.current, { ...chartOpts, height: 100 })
      stoApi.current = stoChart
      stoChart.addLineSeries({ color: '#e2e8f0', lineWidth: 1, priceLineVisible: false })
        .setData(stoData.map(d => ({ time: d.date, value: d.k })))
      stoChart.addLineSeries({ color: '#f97316', lineWidth: 1, priceLineVisible: false })
        .setData(stoData.filter(d => d.d != null).map(d => ({ time: d.date, value: d.d! })))
      stoChart.addLineSeries({ color: 'rgba(239,68,68,0.4)', lineWidth: 1, priceLineVisible: false })
        .setData(stoData.map(d => ({ time: d.date, value: 80 })))
      stoChart.addLineSeries({ color: 'rgba(34,197,94,0.4)', lineWidth: 1, priceLineVisible: false })
        .setData(stoData.map(d => ({ time: d.date, value: 20 })))
      // Add zone line at stoch 30
      stoChart.addLineSeries({ color: 'rgba(16,185,129,0.25)', lineWidth: 1, lineStyle: LineStyle.Dashed, priceLineVisible: false })
        .setData(stoData.map(d => ({ time: d.date, value: 30 })))
      stoChart.timeScale().fitContent()
      chart.timeScale().subscribeVisibleLogicalRangeChange(range => {
        if (range) stoChart.timeScale().setVisibleLogicalRange(range)
      })
    }

    // ── TSI (True Strength Index) sub-panel ─────────────────────────────────
    if (tsiRef.current && chartData.prices.length >= 40) {
      // Helper: single-pass EMA
      const ema = (data: number[], period: number): number[] => {
        const k = 2 / (period + 1)
        const out: number[] = [data[0]]
        for (let i = 1; i < data.length; i++) out.push(data[i] * k + out[i - 1] * (1 - k))
        return out
      }
      const pricesFiltered = chartData.prices.filter(p => p.close != null)
      const closes   = pricesFiltered.map(p => p.close!)
      const momentum = closes.slice(1).map((c, i) => c - closes[i])
      const absMom   = momentum.map(Math.abs)
      // Double-smooth momentum and |momentum| (r=25, s=13 — standard TSI periods)
      const smoothNum = ema(ema(momentum, 25), 13)
      const smoothDen = ema(ema(absMom,    25), 13)
      const tsiVals   = smoothNum.map((n, i) => smoothDen[i] === 0 ? 0 : (100 * n) / smoothDen[i])
      const sigVals   = ema(tsiVals, 13)
      // Align with pricesFiltered.slice(1)
      const tsiData = tsiVals.map((v, i) => ({ time: pricesFiltered[i + 1].date, value: v }))
      const sigData = sigVals.map((v, i) => ({ time: pricesFiltered[i + 1].date, value: v }))

      const tsiChart = createChart(tsiRef.current, { ...chartOpts, height: 100 })
      tsiApi.current = tsiChart
      // TSI line (blue)
      tsiChart.addLineSeries({ color: '#60a5fa', lineWidth: 1, priceLineVisible: false }).setData(tsiData)
      // Signal line (orange)
      tsiChart.addLineSeries({ color: '#f97316', lineWidth: 1, priceLineVisible: false }).setData(sigData)
      // Zero line (faint)
      tsiChart.addLineSeries({ color: 'rgba(156,163,175,0.2)', lineWidth: 1, priceLineVisible: false })
        .setData(tsiData.map(d => ({ time: d.time, value: 0 })))
      // +25 / -25 reference bands
      tsiChart.addLineSeries({ color: 'rgba(239,68,68,0.25)', lineWidth: 1, priceLineVisible: false })
        .setData(tsiData.map(d => ({ time: d.time, value: 25 })))
      tsiChart.addLineSeries({ color: 'rgba(34,197,94,0.25)', lineWidth: 1, priceLineVisible: false })
        .setData(tsiData.map(d => ({ time: d.time, value: -25 })))
      tsiChart.timeScale().fitContent()
      chart.timeScale().subscribeVisibleLogicalRangeChange(range => {
        if (range) tsiChart.timeScale().setVisibleLogicalRange(range)
      })
    }

    // ── Williams %R sub-panel (daily) ────────────────────────────────────────
    const dPrices = chartData.prices.filter(p => p.close != null && p.high != null && p.low != null)
    if (willrRef.current && dPrices.length >= 14) {
      const W_P = 14
      const willrData: { time: string; value: number }[] = []
      for (let i = W_P - 1; i < dPrices.length; i++) {
        const w  = dPrices.slice(i - W_P + 1, i + 1)
        const hi = Math.max(...w.map(p => p.high!))
        const lo = Math.min(...w.map(p => p.low!))
        willrData.push({ time: dPrices[i].date, value: hi === lo ? -50 : ((hi - dPrices[i].close!) / (hi - lo)) * -100 })
      }
      const willrChart = createChart(willrRef.current, { ...chartOpts, height: 100 })
      willrApi.current = willrChart
      willrChart.addLineSeries({ color: '#34d399', lineWidth: 1, priceLineVisible: false }).setData(willrData)
      willrChart.addLineSeries({ color: 'rgba(239,68,68,0.35)',  lineWidth: 1, priceLineVisible: false }).setData(willrData.map(d => ({ time: d.time, value: -20 })))
      willrChart.addLineSeries({ color: 'rgba(34,197,94,0.35)',  lineWidth: 1, priceLineVisible: false }).setData(willrData.map(d => ({ time: d.time, value: -80 })))
      willrChart.addLineSeries({ color: 'rgba(156,163,175,0.1)', lineWidth: 1, priceLineVisible: false }).setData(willrData.map(d => ({ time: d.time, value: -50 })))
      willrChart.timeScale().fitContent()
      chart.timeScale().subscribeVisibleLogicalRangeChange(range => { if (range) willrChart.timeScale().setVisibleLogicalRange(range) })
    }

    // ── CCI sub-panel (daily) ─────────────────────────────────────────────────
    if (cciRef.current && dPrices.length >= 20) {
      const CCI_P = 20
      const cciData: { time: string; value: number }[] = []
      for (let i = CCI_P - 1; i < dPrices.length; i++) {
        const w   = dPrices.slice(i - CCI_P + 1, i + 1)
        const tps = w.map(p => (p.high! + p.low! + p.close!) / 3)
        const sma = tps.reduce((a, b) => a + b, 0) / CCI_P
        const dev = tps.reduce((a, v) => a + Math.abs(v - sma), 0) / CCI_P
        cciData.push({ time: dPrices[i].date, value: dev === 0 ? 0 : (tps[CCI_P - 1] - sma) / (0.015 * dev) })
      }
      const cciChart = createChart(cciRef.current, { ...chartOpts, height: 100 })
      cciApi.current = cciChart
      cciChart.addLineSeries({ color: '#38bdf8', lineWidth: 1, priceLineVisible: false }).setData(cciData)
      cciChart.addLineSeries({ color: 'rgba(239,68,68,0.3)',   lineWidth: 1, priceLineVisible: false }).setData(cciData.map(d => ({ time: d.time, value:  100 })))
      cciChart.addLineSeries({ color: 'rgba(34,197,94,0.3)',   lineWidth: 1, priceLineVisible: false }).setData(cciData.map(d => ({ time: d.time, value: -100 })))
      cciChart.addLineSeries({ color: 'rgba(156,163,175,0.1)', lineWidth: 1, priceLineVisible: false }).setData(cciData.map(d => ({ time: d.time, value:    0 })))
      cciChart.timeScale().fitContent()
      chart.timeScale().subscribeVisibleLogicalRangeChange(range => { if (range) cciChart.timeScale().setVisibleLogicalRange(range) })
    }

    // ── OBV sub-panel (daily) ─────────────────────────────────────────────────
    const dPricesVol = chartData.prices.filter(p => p.close != null && p.volume != null)
    if (obvRef.current && dPricesVol.length >= 2) {
      let obv = 0
      const obvData: { time: string; value: number }[] = [{ time: dPricesVol[0].date, value: 0 }]
      for (let i = 1; i < dPricesVol.length; i++) {
        if (dPricesVol[i].close! > dPricesVol[i - 1].close!)      obv += dPricesVol[i].volume!
        else if (dPricesVol[i].close! < dPricesVol[i - 1].close!) obv -= dPricesVol[i].volume!
        obvData.push({ time: dPricesVol[i].date, value: obv })
      }
      const obvChart = createChart(obvRef.current, { ...chartOpts, height: 100 })
      obvApi.current = obvChart
      obvChart.addLineSeries({ color: '#fb923c', lineWidth: 1, priceLineVisible: false }).setData(obvData)
      obvChart.timeScale().fitContent()
      chart.timeScale().subscribeVisibleLogicalRangeChange(range => { if (range) obvChart.timeScale().setVisibleLogicalRange(range) })
    }

    // ── ATR sub-panel (daily) ─────────────────────────────────────────────────
    if (atrPanelRef.current && dPrices.length >= 15) {
      const ATR_P = 14
      const atrData: { time: string; value: number }[] = []
      let atrVal = 0
      const dTRs: number[] = []
      for (let i = 1; i < dPrices.length; i++) {
        const tr = Math.max(
          dPrices[i].high! - dPrices[i].low!,
          Math.abs(dPrices[i].high! - dPrices[i - 1].close!),
          Math.abs(dPrices[i].low!  - dPrices[i - 1].close!)
        )
        dTRs.push(tr)
        if (i === ATR_P) {
          atrVal = dTRs.reduce((a, b) => a + b, 0) / ATR_P
        } else if (i > ATR_P) {
          atrVal = (atrVal * (ATR_P - 1) + tr) / ATR_P
        }
        if (i >= ATR_P) atrData.push({ time: dPrices[i].date, value: atrVal })
      }
      const atrChart = createChart(atrPanelRef.current, { ...chartOpts, height: 100 })
      atrPanelApi.current = atrChart
      atrChart.addLineSeries({ color: '#e879f9', lineWidth: 1, priceLineVisible: false }).setData(atrData)
      atrChart.timeScale().fitContent()
      chart.timeScale().subscribeVisibleLogicalRangeChange(range => { if (range) atrChart.timeScale().setVisibleLogicalRange(range) })
    }

    // ── Bradley Siderograph sub-panel ───────────────────────────────────────
    if (showBradleyPane && bradleyRef.current && patterns) {
      const bradleyPattern = patterns.find(p => p.pattern_name === 'Bradley Siderograph')
      const rawSeries = (bradleyPattern as unknown as Record<string, unknown>)?.raw_bradley_series as
        Array<[string, number]> | undefined
      if (rawSeries?.length) {
        const bChart = createChart(bradleyRef.current, {
          ...chartOpts, height: 140,
          layout: { ...chartOpts.layout, background: { type: ColorType.Solid, color: '#0d0f14' } },
        })
        bradleyApi.current = bChart

        // Score line (amber/gold — Bradley's "planetary energy")
        const scorePts = rawSeries.map(([d, v]) => ({ time: d as import('lightweight-charts').Time, value: v }))
          .sort((a, b) => String(a.time).localeCompare(String(b.time)))
        const scoreSeries = bChart.addLineSeries({
          color: '#fbbf24', lineWidth: 2, priceLineVisible: false,
          crosshairMarkerVisible: true, lastValueVisible: true, title: 'Bradley',
        })
        scoreSeries.setData(scorePts)

        // Zero reference line
        bChart.addLineSeries({ color: 'rgba(156,163,175,0.25)', lineWidth: 1, priceLineVisible: false, crosshairMarkerVisible: false, lastValueVisible: false })
          .setData(scorePts.map(d => ({ ...d, value: 0 })))

        // Turning point markers (red ▼ at highs, green ▲ at lows)
        const turningPts = (bradleyPattern as unknown as Record<string, unknown>)?.bradley_turning_points as
          Array<{ date: string; type: string }> | undefined
        if (turningPts?.length) {
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          scoreSeries.setMarkers(turningPts.map(tp => ({
            time: tp.date as import('lightweight-charts').Time,
            position: tp.type === 'high' ? 'aboveBar' as const : 'belowBar' as const,
            color:    tp.type === 'high' ? '#ef4444' : '#22c55e',
            shape:    tp.type === 'high' ? 'arrowDown' as const : 'arrowUp' as const,
            text:     tp.type === 'high' ? 'H' : 'L',
            size: 1,
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          })) as any)
        }

        bChart.timeScale().fitContent()
        chart.timeScale().subscribeVisibleLogicalRangeChange(range => { if (range) bChart.timeScale().setVisibleLogicalRange(range) })
      }
    }

    // ── Planetary Degrees sub-panel (Path of Planets) ───────────────────────
    if (showPlanetsPane && planetsRef.current && patterns) {
      const planetsPattern = patterns.find(p => p.pattern_name === 'Planetary Degrees')
      const planetSeries = (planetsPattern as unknown as Record<string, unknown>)?.planet_series as
        Record<string, [string, number][]> | undefined
      if (planetSeries && Object.keys(planetSeries).length) {
        const pChart = createChart(planetsRef.current, {
          ...chartOpts, height: 160,
          layout: { ...chartOpts.layout, background: { type: ColorType.Solid, color: '#0d0f14' } },
        })
        planetsApi.current = pChart

        const PLANET_COLORS: Record<string, string> = {
          Sun:     '#fbbf24',  // gold
          Moon:    '#e2e8f0',  // silver
          Mercury: '#a78bfa',  // violet
          Venus:   '#f9a8d4',  // pink
          Mars:    '#ef4444',  // red
          Jupiter: '#fb923c',  // orange
          Saturn:  '#94a3b8',  // slate
        }
        // Draw each planet as a colored line; 0–360° y-axis (ecliptic longitude)
        for (const [planet, pts] of Object.entries(planetSeries)) {
          const color = PLANET_COLORS[planet] ?? '#9ca3af'
          const data = (pts as [string, number][])
            .map(([d, lon]) => ({ time: d as import('lightweight-charts').Time, value: lon }))
            .sort((a, b) => String(a.time).localeCompare(String(b.time)))
          if (data.length < 2) continue
          pChart.addLineSeries({
            color, lineWidth: 1, priceLineVisible: false,
            crosshairMarkerVisible: false, lastValueVisible: true, title: planet.slice(0, 3),
          }).setData(data)
        }

        pChart.timeScale().fitContent()
        chart.timeScale().subscribeVisibleLogicalRangeChange(range => { if (range) pChart.timeScale().setVisibleLogicalRange(range) })
      }
    }

    // Drawing tools integration (daily chart)
    chart.subscribeClick(drawingHandleClick)
    chart.subscribeCrosshairMove(drawingHandleCrosshairMove)
    attachAllPrimitives()

    return () => {
      try { chart.unsubscribeClick(drawingHandleClick) } catch { /* already disposed */ }
      try { chart.unsubscribeCrosshairMove(drawingHandleCrosshairMove) } catch { /* already disposed */ }
      try { chart.remove() } catch { /* already disposed */ }
      try { rsiApi.current?.remove() } catch { /* already disposed */ }
      try { macdApi.current?.remove() } catch { /* already disposed */ }
      try { stoApi.current?.remove()      } catch { /* already disposed */ }
      try { tsiApi.current?.remove()      } catch { /* already disposed */ }
      try { willrApi.current?.remove()    } catch { /* already disposed */ }
      try { cciApi.current?.remove()      } catch { /* already disposed */ }
      try { obvApi.current?.remove()      } catch { /* already disposed */ }
      try { atrPanelApi.current?.remove()  } catch { /* already disposed */ }
      try { bradleyApi.current?.remove()  } catch { /* already disposed */ }
      try { planetsApi.current?.remove()  } catch { /* already disposed */ }
      chartApi.current = null; rsiApi.current = null; macdApi.current = null
      stoApi.current = null; tsiApi.current = null
      willrApi.current = null; cciApi.current = null
      obvApi.current = null; atrPanelApi.current = null
      bradleyApi.current = null; planetsApi.current = null
      candlestickRef.current = null
    }
  }, [chartData, intradayData, isIntraday, showBB, showMA, showTrendlines, showGaps, showPriceZones, trendlineTF, trendlineData, showPatterns, activePatternName, showLiquidityMap, showStochastics, showAddZones, showMonteCarloForecast, patterns, signalForSymbol?.probability, rec?.components?.xgb_probability, rec?.trade_plan?.accumulate_price, rec?.trade_plan?.buy_price, rec?.trade_plan?.scale_price, rec?.trade_plan?.sell_price, stochMap, priceLevels, showBradleyPane, showPlanetsPane])

  // ── Timeframe zoom ─────────────────────────────────────────────────────────
  useEffect(() => {
    if (!chartApi.current) return
    const applyZoom = (api: IChartApi) => {
      // When a pattern is selected, auto-zoom to its time extent so the pattern
      // fills the chart regardless of which TF the user selected.
      if (showPatterns && activePatternName && patterns?.length) {
        const sel = patterns.find(p =>
          p.pattern_name === activePatternName &&
          p.status != null && p.status !== 'NOT_PRESENT' && p.status !== 'FAILED'
        )
        if (sel) {
          const rawLines = (sel.overlay_lines ?? []) as [[string, number], [string, number]][]
          const dates: string[] = []
          for (const seg of rawLines) {
            if (!Array.isArray(seg) || seg.length < 2) continue
            const [a, b] = seg
            if (Array.isArray(a) && a.length >= 2) dates.push(String(a[0]))
            if (Array.isArray(b) && b.length >= 2) dates.push(String(b[0]))
          }
          // Also include pattern key points
          for (const pt of ((sel.points ?? []) as [string, number][])) {
            if (Array.isArray(pt) && pt.length >= 2) dates.push(String(pt[0]))
          }
          const valid = dates.filter(d => /^\d{4}-\d{2}-\d{2}/.test(d)).sort()
          if (valid.length >= 2) {
            const newestDate = new Date(valid[valid.length - 1] + 'T00:00:00Z')
            const oldestDate = new Date(valid[0] + 'T00:00:00Z')
            const spanDays   = (newestDate.getTime() - oldestDate.getTime()) / 86_400_000

            // Cap lookback to 550 days (≈18 months) so long-span patterns
            // (Gann, harmonics anchored years ago) don't zoom out too far.
            const lookbackDays = Math.min(spanDays, 550)
            const fromDate = new Date(newestDate.getTime() - lookbackDays * 86_400_000)
            const toDate   = new Date(newestDate.getTime())

            // Pad left (show 12% of lookback before first visible bar)
            fromDate.setUTCDate(fromDate.getUTCDate() - Math.max(30, Math.round(lookbackDays * 0.12)))
            // Pad right (show 90 extra days for target projection)
            toDate.setUTCDate(toDate.getUTCDate() + 90)

            const fmt = (d: Date) => d.toISOString().split('T')[0]
            try {
              api.timeScale().setVisibleRange({ from: fmt(fromDate) as never, to: fmt(toDate) as never })
              return   // zoom applied — skip default TF zoom
            } catch { /* chart disposed — fall through */ }
          }
        }
      }

      // Default TF zoom — W/M bar-resolution charts and All/intraday fit content
      if (isIntraday || isBarTF || tf === 'All') {
        api.timeScale().fitContent()
      } else {
        const days = TF_DAYS[tf]
        const to = new Date(); const from = new Date(to.getTime() - days * 86_400_000)
        const fmt = (d: Date) => d.toISOString().split('T')[0]
        api.timeScale().setVisibleRange({ from: fmt(from) as never, to: fmt(to) as never })
      }
    }
    applyZoom(chartApi.current)
    if (rsiApi.current) applyZoom(rsiApi.current)
    if (macdApi.current) applyZoom(macdApi.current)
    if (stoApi.current) applyZoom(stoApi.current)
    if (tsiApi.current) applyZoom(tsiApi.current)
  }, [tf, chartData, intradayData, activePatternName, patterns, showPatterns])

  const detectedPatterns = (() => {
    const sorted = (patterns || [])
      .filter((p) => {
        const status = p.status ?? 'FORMING'
        return status !== 'NOT_PRESENT' && status !== 'FAILED' && status !== 'FORMING'
      })
      .sort((a, b) => (b.confidence ?? b.probability ?? b.strength * 100) - (a.confidence ?? a.probability ?? a.strength * 100))
    // Deduplicate by pattern_name — keep highest-confidence entry
    const seen = new Set<string>()
    return sorted.filter(p => {
      if (seen.has(p.pattern_name)) return false
      seen.add(p.pattern_name)
      return true
    }).slice(0, 10)
  })()

  const filteredPatterns = patternCategoryFilter
    ? detectedPatterns.filter(p => classifyPattern(p) === patternCategoryFilter)
    : detectedPatterns

  const loading = isIntraday ? intradayLoading : isLoading

  return (
    <div className={`flex gap-4 ${fullChart ? 'relative' : ''}`}>
      {/* Main chart area */}
      <div className="flex-1 min-w-0">
        {/* Live quote bar */}
        {quote && quote.price != null && (
          <div className="flex items-center gap-4 mb-3 p-2.5 bg-gray-900 rounded-lg border border-gray-800 text-xs flex-wrap">
            <span className="font-bold text-white text-base">${quote.price.toFixed(2)}</span>
            {quote.change != null && (
              <span className={`font-mono font-bold ${quote.change >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                {quote.change >= 0 ? '+' : ''}{quote.change.toFixed(2)} ({quote.change_pct?.toFixed(2)}%)
              </span>
            )}
            <span className="text-gray-500">O: <span className="text-gray-300">{quote.open?.toFixed(2)}</span></span>
            <span className="text-gray-500">H: <span className="text-emerald-400">{quote.high?.toFixed(2)}</span></span>
            <span className="text-gray-500">L: <span className="text-red-400">{quote.low?.toFixed(2)}</span></span>
            <span className="text-gray-500">Prev: <span className="text-gray-300">{quote.prev_close?.toFixed(2)}</span></span>
            {quote.pre_market_price != null && (
              <span className="text-gray-500">Pre: <span className="text-blue-300 font-mono">${quote.pre_market_price.toFixed(2)}</span>
                {quote.pre_market_change_pct != null && (
                  <span className={`ml-1 font-mono ${quote.pre_market_change_pct >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                    {quote.pre_market_change_pct >= 0 ? '+' : ''}{quote.pre_market_change_pct.toFixed(2)}%
                  </span>
                )}
              </span>
            )}
            {quote.post_market_price != null && (
              <span className="text-gray-500">AH: <span className="text-violet-300 font-mono">${quote.post_market_price.toFixed(2)}</span>
                {quote.post_market_change_pct != null && (
                  <span className={`ml-1 font-mono ${quote.post_market_change_pct >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                    {quote.post_market_change_pct >= 0 ? '+' : ''}{quote.post_market_change_pct.toFixed(2)}%
                  </span>
                )}
              </span>
            )}
            <span className="ml-auto text-gray-600 text-[10px]">Live · Finnhub · refreshes 30s</span>
          </div>
        )}

        {/* Controls — two explicit rows so nothing overflows */}
        <div className="flex flex-col gap-1.5 mb-3">

          {/* ── Row 1: Symbol info + autocomplete + MTF dots ── */}
          <div className="flex items-center gap-2 flex-wrap">
            {/* Symbol header with full name + watchlist toggle */}
            <h2 className="text-2xl font-bold font-mono">{symbol}</h2>
            <button
              onClick={() => watchlistToggle.mutate({ symbol, watching: isWatching })}
              title={isWatching ? 'Remove from watchlist' : 'Add to watchlist'}
              className={`flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium border transition-all ${
                isWatching
                  ? 'border-amber-600/60 text-amber-400 bg-amber-900/20 hover:bg-amber-900/40'
                  : 'border-gray-700 text-gray-500 hover:border-amber-600/50 hover:text-amber-400 hover:bg-amber-900/10'
              }`}
            >
              {isWatching ? '★ Watching' : '☆ Watch'}
            </button>
            {(() => {
              const inst = (instruments || []).find(i => i.symbol === symbol)
              return inst ? (
                <span className="text-sm text-gray-400">
                  {inst.name ?? ''}
                  {inst.leverage_factor && inst.leverage_factor !== 1 && (
                    <span className="ml-1 text-xs text-amber-400">{inst.leverage_factor}× ETF</span>
                  )}
                </span>
              ) : null
            })()}

            {/* Symbol autocomplete */}
            <div ref={acRef} className="relative ml-1">
              <input
                ref={acInputRef}
                value={acInput}
                onChange={e => { setAcInput(e.target.value.toUpperCase()); setAcOpen(true) }}
                onFocus={() => setAcOpen(true)}
                onKeyDown={e => {
                  if (e.key === 'Enter') {
                    e.preventDefault()
                    navigateToSymbol(acSuggestions[0]?.symbol || acInput)
                  } else if (e.key === 'Escape') {
                    setAcOpen(false)
                    setAcInput(symbol) // revert to current
                  }
                }}
                placeholder="Symbol…"
                spellCheck={false}
                autoComplete="off"
                className="bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm font-mono w-28 focus:outline-none focus:border-emerald-600 focus:w-36 transition-all"
              />
              {acOpen && (acSuggestions.length > 0 || acInput.trim().length > 0) && (
                <div className="absolute top-full left-0 mt-1 w-72 bg-gray-900 border border-gray-700 rounded-lg shadow-2xl z-[200] overflow-hidden">
                  {acInput.trim() && !acSuggestions.some(s => s.symbol === acInput.trim().toUpperCase()) && (
                    <button
                      onMouseDown={e => { e.preventDefault(); navigateToSymbol(acInput) }}
                      className="w-full px-3 py-2 text-left flex items-center gap-2 border-b border-gray-800 hover:bg-emerald-900/20 group"
                    >
                      <span className="font-mono font-bold text-emerald-400 text-sm w-20 flex-shrink-0">
                        {acInput.trim().toUpperCase()}
                      </span>
                      <span className="text-[11px] text-gray-500 group-hover:text-gray-300">→ go to this symbol</span>
                    </button>
                  )}
                  {acSuggestions.map((s, idx) => (
                    <button
                      key={s.symbol + idx}
                      onMouseDown={e => { e.preventDefault(); navigateToSymbol(s.symbol) }}
                      className="w-full px-3 py-2 text-left flex items-center gap-2 hover:bg-gray-800 transition-colors"
                    >
                      <span className="font-mono font-bold text-sm text-gray-100 w-20 flex-shrink-0">{s.symbol}</span>
                      <span className="text-[11px] text-gray-500 truncate flex-1">{s.name}</span>
                    </button>
                  ))}
                </div>
              )}
            </div>

            {/* MTF Alignment badges */}
            {mtfData && (
              <div className="flex items-center gap-1 px-2 py-1 bg-gray-900 border border-gray-800 rounded">
                <span className="text-[9px] text-gray-600 mr-0.5">MTF:</span>
                {Object.entries(mtfData.tfs).map(([key, tf_]) => {
                  const hasPatterns = tf_.count > 0
                  const isCached   = tf_.cached
                  const dotColor   = !isCached ? '#374151' : hasPatterns && tf_.top_confidence >= 60 ? '#22c55e' : hasPatterns ? '#f59e0b' : '#374151'
                  return (
                    <span key={key}
                      title={isCached ? `${tf_.label}: ${tf_.count} pattern(s) · top ${tf_.top_confidence.toFixed(0)}%` : `${tf_.label}: not scanned yet`}
                      className="flex items-center gap-0.5 text-[9px] font-mono cursor-default"
                      style={{ color: dotColor }}
                    >
                      <span style={{ fontSize: 8 }}>●</span>
                      <span className="text-gray-600">{tf_.label}</span>
                    </span>
                  )
                })}
              </div>
            )}
          </div>

          {/* ── Row 2: Timeframe buttons (scrollable left) | action buttons (fixed right) ── */}
          <div className="flex items-center gap-2 min-w-0 overflow-hidden">

            {/* Left: TF buttons + badge — scrolls horizontally if too wide */}
            <div className="flex items-center gap-1.5 min-w-0 overflow-x-auto flex-1 scrollbar-none">
              <div className="flex items-center gap-0.5 bg-gray-900 border border-gray-800 rounded p-0.5 flex-shrink-0">
                {INTRADAY_TFS.map(t => (
                  <button key={t} onClick={() => setTf(t)}
                    className={`px-2.5 py-1 rounded text-xs font-medium transition-colors ${tf === t ? 'bg-blue-600 text-white' : 'text-gray-400 hover:text-gray-200'}`}>
                    {t}
                  </button>
                ))}
                <div className="w-px h-4 bg-gray-700 mx-0.5" />
                {DAILY_TFS.map(t => (
                  <button key={t} onClick={() => setTf(t)}
                    className={`px-2.5 py-1 rounded text-xs font-medium transition-colors ${tf === t ? 'bg-emerald-600 text-white' : 'text-gray-400 hover:text-gray-200'}`}>
                    {t}
                  </button>
                ))}
                <div className="w-px h-4 bg-gray-700 mx-0.5" />
                {BAR_TFS.map(t => (
                  <button key={t} onClick={() => setTf(t)}
                    title={t === 'W' ? 'Weekly bars' : 'Monthly bars'}
                    className={`px-2.5 py-1 rounded text-xs font-medium transition-colors ${tf === t ? 'bg-amber-600 text-white' : 'text-gray-400 hover:text-amber-300'}`}>
                    {t}
                  </button>
                ))}
              </div>

              {/* Active timeframe badge */}
              <span className={`flex-shrink-0 text-[10px] px-2 py-0.5 rounded border font-mono ${
                isIntraday
                  ? 'text-blue-300/80 bg-blue-900/20 border-blue-800/40'
                  : isBarTF
                  ? 'text-amber-400/80 bg-amber-900/20 border-amber-800/40'
                  : 'text-emerald-400/70 bg-emerald-900/20 border-emerald-800/30'
              }`}>
                {isIntraday ? `Intraday · ${tf}` : isBarTF ? (tf === 'W' ? 'Weekly bars' : 'Monthly bars') : `Daily · ${tf}`}
              </span>
            </div>

            {/* Right: action buttons — always fully visible, never pushed off */}
            <div className="flex items-center gap-1.5 flex-shrink-0">
<button
                onClick={() => setAllOverlays(false)}
                title="Hide all overlays — candlesticks + volume only"
                className={`px-2.5 py-1 rounded-lg text-xs font-medium border transition-colors ${
                  cleanMode
                    ? 'bg-emerald-900/40 text-emerald-300 border-emerald-700/50'
                    : 'bg-gray-800 text-gray-400 border-gray-700 hover:border-gray-500'
                }`}
              >
                {cleanMode ? '◉ Clean' : '○ Clean'}
              </button>

              <button
                onClick={() => setFullChart(f => !f)}
                title="Toggle full chart width (hides sidebar)"
                className={`px-2.5 py-1 rounded-lg text-xs font-medium border transition-colors ${
                  fullChart
                    ? 'bg-blue-900/40 text-blue-300 border-blue-700/50'
                    : 'bg-gray-800 text-gray-400 border-gray-700 hover:border-gray-500'
                }`}
              >
                {fullChart ? '⊟' : '⊞ Full'}
              </button>
            </div>
          </div>
        </div>

        {/* Overlay controls — both dropdowns on one row to save space */}
        <div className="flex gap-2 items-start mb-3">
        {/* Overlays dropdown — compact trigger + multi-select popover */}
        {(() => {
          const overlays = [
            { key: 'ma',    label: 'MA 50/200',  show: showMA,                 toggle: () => setShowMA(v => !v),                 color: '#3b82f6' },
            { key: 'bb',    label: 'Bollinger',  show: showBB,                 toggle: () => setShowBB(v => !v),                 color: '#8b5cf6' },
            { key: 'tl',    label: 'Trendlines', show: showTrendlines,         toggle: () => setShowTrendlines(v => !v),         color: '#22c55e' },
            { key: 'gaps',  label: 'Gaps',       show: showGaps,               toggle: () => setShowGaps(v => !v),               color: '#f59e0b' },
            { key: 'zones', label: 'Zones',      show: showPriceZones,         toggle: () => setShowPriceZones(v => !v),         color: '#60a5fa' },
            { key: 'lq',    label: 'Liquidity',  show: showLiquidityMap,       toggle: () => setShowLiquidityMap(v => !v),       color: '#a855f7' },
            { key: 'mc',    label: 'Forecast',   show: showMonteCarloForecast, toggle: () => setShowMonteCarloForecast(v => !v), color: '#3b82f6' },
            { key: 'add',   label: 'Add Zones',  show: showAddZones,           toggle: () => setShowAddZones(v => !v),           color: '#10b981' },
            { key: 'cyc',   label: 'Cycles',     show: showCycleWindow,        toggle: () => setShowCycleWindow(v => !v),        color: '#06b6d4' },
            { key: 'force', label: 'Force',      show: showForceGauge,         toggle: () => setShowForceGauge(v => !v),         color: '#ec4899' },
            { key: 'nodes', label: 'Dec. Nodes', show: showDecisionNodes,      toggle: () => setShowDecisionNodes(v => !v),      color: '#f97316' },
            { key: 'stoch', label: 'Stoch',      show: showStochastics,        toggle: () => setShowStochastics(v => !v),        color: '#e2e8f0' },
            { key: 'conf',  label: 'Confidence', show: showSignalConfidence,   toggle: () => setShowSignalConfidence(v => !v),   color: '#34d399' },
            { key: 'prob',  label: 'Prob Bands', show: showProbBands,          toggle: () => setShowProbBands(v => !v),          color: '#818cf8' },
            { key: 'health',label: 'Health',     show: showModelHealth,        toggle: () => setShowModelHealth(v => !v),        color: '#fb923c' },
            { key: 'opts',  label: 'Options',    show: showOptionsFlow,        toggle: () => setShowOptionsFlow(v => !v),        color: '#38bdf8' },
            { key: 'astro', label: 'Astro',      show: showAstroCycles,        toggle: () => setShowAstroCycles(v => !v),        color: '#fbbf24' },
          ]
          const activeOverlays = overlays.filter(o => o.show)
          const nActive = activeOverlays.length
          return (
            <div ref={overlayDropdownRef} className="relative">
              {/* Trigger */}
              <button
                onClick={() => setShowOverlayDropdown(v => !v)}
                className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg border text-[11px] font-medium transition-all"
                style={{
                  borderColor:     nActive > 0 ? '#374151' : '#1f2937',
                  color:           nActive > 0 ? '#d1d5db' : '#4b5563',
                  backgroundColor: nActive > 0 ? 'rgba(255,255,255,0.04)' : 'transparent',
                }}
              >
                {/* Active color dots */}
                {nActive > 0 && (
                  <span className="flex items-center gap-0.5">
                    {activeOverlays.slice(0, 5).map(o => (
                      <span key={o.key} className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: o.color }} />
                    ))}
                    {nActive > 5 && <span className="text-[9px] text-gray-500">+{nActive - 5}</span>}
                  </span>
                )}
                <span>Overlays{nActive > 0 ? ` · ${nActive} on` : ''}</span>
                <span className="text-[9px] opacity-50">{showOverlayDropdown ? '▲' : '▼'}</span>
              </button>

              {/* Dropdown panel */}
              {showOverlayDropdown && (
                <div className="absolute top-full left-0 mt-1 w-56 bg-gray-900 border border-gray-700 rounded-xl shadow-2xl z-[150] overflow-hidden">
                  <div className="px-2.5 py-1.5 border-b border-gray-800 flex items-center justify-between">
                    <span className="text-[10px] text-gray-500">Toggle chart overlays</span>
                    <div className="flex items-center gap-2">
                      <button
                        onClick={() => { setAllOverlays(true); setShowStochastics(true) }}
                        className="text-[10px] text-emerald-600 hover:text-emerald-400 transition-colors"
                      >all on</button>
                      <button
                        onClick={() => { setAllOverlays(false); setShowStochastics(false) }}
                        className="text-[10px] text-gray-600 hover:text-gray-300 transition-colors"
                      >all off</button>
                    </div>
                  </div>
                  <div className="py-1">
                    {overlays.map(({ key, label, show, toggle, color }) => (
                      <button
                        key={key}
                        onClick={toggle}
                        className="w-full flex items-center gap-2.5 px-3 py-1.5 text-left text-[11px] transition-colors hover:bg-gray-800"
                        style={{ backgroundColor: show ? color + '12' : undefined }}
                      >
                        <span
                          className="w-2 h-2 rounded-sm flex-shrink-0 border"
                          style={{
                            backgroundColor: show ? color : 'transparent',
                            borderColor:     show ? color : '#374151',
                          }}
                        />
                        <span className="flex-1" style={{ color: show ? '#f3f4f6' : '#6b7280' }}>{label}</span>
                        {show && <span className="text-[9px] font-mono" style={{ color }}>ON</span>}
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )
        })()}

        {/* Pattern overlay dropdown — compact trigger + popover list */}
        {detectedPatterns.length > 0 && (() => {
          const activeP   = detectedPatterns.find(p => p.pattern_name === activePatternName)
          const activeKind  = activeP ? classifyPattern(activeP) : null
          const activeColor = activeP ? patternColor(activeKind!, activeP.direction) : null
          const activeConf  = activeP ? Math.round((activeP.confidence ?? activeP.probability ?? 0) * 100) / 100 : 0
          const triggerLabel = activeP
            ? `${activeP.pattern_name.split(' ').slice(0, 3).join(' ')} · ${Math.round(activeConf)}%`
            : `Overlay pattern (${detectedPatterns.length})`
          return (
            <div ref={patternDropdownRef} className="relative">
              {/* Trigger button */}
              <button
                onClick={() => setShowPatternDropdown(v => !v)}
                className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg border text-[11px] font-medium transition-all"
                style={{
                  borderColor:     activeColor ?? '#374151',
                  color:           activeColor ?? '#6b7280',
                  backgroundColor: activeColor ? activeColor + '15' : 'transparent',
                }}
              >
                {activeColor && (
                  <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ backgroundColor: activeColor }} />
                )}
                <span>{triggerLabel}</span>
                <span className="ml-1 text-[9px] opacity-60">{showPatternDropdown ? '▲' : '▼'}</span>
                {activeP && (
                  <span
                    role="button"
                    onClick={e => { e.stopPropagation(); setActivePatternName(null); setShowPatternDropdown(false) }}
                    className="ml-1 text-[10px] opacity-50 hover:opacity-100 cursor-pointer"
                    title="Clear overlay"
                  >✕</span>
                )}
              </button>

              {/* Dropdown panel — wider to accommodate educational thumbnails */}
              {showPatternDropdown && (
                <div className="absolute top-full left-0 mt-1 w-[320px] bg-gray-900 border border-gray-700 rounded-xl shadow-2xl z-[150] overflow-hidden">
                  <div className="px-2.5 py-1.5 border-b border-gray-800 flex items-center justify-between">
                    <span className="text-[10px] text-gray-500">Pattern overlay — hover for educational diagram</span>
                    {activePatternName && (
                      <button
                        onClick={() => { setActivePatternName(null); setShowPatternDropdown(false) }}
                        className="text-[10px] text-gray-600 hover:text-gray-300 transition-colors"
                      >✕ clear</button>
                    )}
                  </div>
                  <div className="max-h-80 overflow-y-auto py-1">
                    {detectedPatterns.map((p) => {
                      const kind    = classifyPattern(p)
                      const color   = patternColor(kind, p.direction)
                      const isAct   = activePatternName === p.pattern_name
                      const conf    = Math.round((p.confidence ?? p.probability ?? 0))
                      const dirIcon = p.direction === 'bullish' ? '▲' : p.direction === 'bearish' ? '▼' : '—'
                      return (
                        <button
                          key={`dd-${p.pattern_name}`}
                          onClick={() => {
                            setActivePatternName(prev => prev === p.pattern_name ? null : p.pattern_name)
                            setShowPatternDropdown(false)
                          }}
                          className="w-full flex items-center gap-2 px-2.5 py-2 text-left text-[11px] transition-colors hover:bg-gray-800"
                          style={{ backgroundColor: isAct ? color + '18' : undefined }}
                        >
                          {/* Educational pattern thumbnail */}
                          <div
                            className="flex-shrink-0 rounded overflow-hidden"
                            style={{
                              background: isAct ? color + '10' : 'rgba(255,255,255,0.03)',
                              border: `1px solid ${isAct ? color + '40' : 'rgba(255,255,255,0.06)'}`,
                            }}
                          >
                            <PatternThumbnail
                              patternName={p.pattern_name}
                              color={color}
                              width={54}
                              height={34}
                            />
                          </div>
                          {/* Pattern name + direction + confidence */}
                          <div className="flex-1 min-w-0">
                            <div className="font-medium truncate" style={{ color: isAct ? color : '#d1d5db' }}>
                              {p.pattern_name}
                            </div>
                            <div className="text-[10px] mt-0.5" style={{ color: isAct ? color : '#6b7280' }}>
                              {dirIcon} {conf}% confidence
                            </div>
                          </div>
                        </button>
                      )
                    })}
                  </div>
                </div>
              )}
            </div>
          )
        })()}
        </div>{/* end overlay controls flex row */}

        {/* Regime status bar — shows active market regime above the chart */}
        {confluenceData && !isIntraday && (() => {
          const isBreakout = confluenceData.is_breakout
          const isComp     = confluenceData.is_compression
          const regime     = confluenceData.volatility_regime ?? ''
          const tier       = confluenceData.signal_tier ?? 'NONE'
          if (!isBreakout && !isComp && tier === 'NONE') return null
          const bgColor  = isBreakout ? 'rgba(34,197,94,0.08)'  : isComp ? 'rgba(245,158,11,0.08)' : 'transparent'
          const bdColor  = isBreakout ? 'rgba(34,197,94,0.30)'  : isComp ? 'rgba(245,158,11,0.30)' : 'transparent'
          const txtColor = isBreakout ? '#22c55e' : isComp ? '#f59e0b' : '#6b7280'
          const label    = isBreakout ? '⚡ BREAKOUT REGIME' : isComp ? '◉ COMPRESSION — building energy' : ''
          return (
            <div className="mb-1.5 px-3 py-1.5 rounded-lg flex items-center justify-between text-[10px]"
              style={{ background: bgColor, border: `1px solid ${bdColor}` }}>
              <span className="font-semibold" style={{ color: txtColor }}>{label}</span>
              <div className="flex items-center gap-3">
                {regime && <span className="text-gray-600">{regime}</span>}
                {tier !== 'NONE' && (
                  <span className="font-mono px-1.5 py-0.5 rounded text-[9px]"
                    style={{ background: txtColor + '20', color: txtColor }}>
                    {tier} · {confluenceData.confluence_score.toFixed(0)}/100
                  </span>
                )}
                {confluenceData.expected_move_pct > 0 && (
                  <span className="text-gray-500">Expected: {confluenceData.expected_move_pct.toFixed(1)}%</span>
                )}
              </div>
            </div>
          )
        })()}

        {loading ? (
          <div className="h-96 flex items-center justify-center text-gray-500">Loading chart...</div>
        ) : noChartData ? (
          <div className="h-96 flex flex-col items-center justify-center gap-4 text-gray-500 border border-gray-800 rounded-lg">
            <span className="text-4xl">🔍</span>
            <div className="text-center">
              <p className="text-sm font-semibold text-gray-300 mb-1">
                No data found for <span className="text-white font-bold font-mono">{symbol}</span>
              </p>
              <p className="text-xs text-gray-600 max-w-xs">
                {isIntraday
                  ? 'Intraday data unavailable for this symbol. Try switching to a Daily timeframe.'
                  : 'This symbol was not found on any data source — it may be delisted, OTC-only, or misspelled.'}
              </p>
            </div>
            {!isIntraday && (
              <div className="flex gap-2">
                <button
                  onClick={() => setTf('1Y')}
                  className="px-3 py-1.5 rounded text-xs bg-gray-800 hover:bg-gray-700 border border-gray-700 text-gray-300 transition-colors"
                >
                  Try Daily Chart
                </button>
              </div>
            )}
          </div>
        ) : (
          <div className="space-y-1">
            {/* Main candlestick chart — explicit h-96 so autoSize picks up the height */}
            <div className="relative">
              <DrawingToolbar
                activeTool={interaction.activeTool}
                activeColor={activeColor}
                selectedDrawing={selectedDrawing}
                drawingCount={drawings.length}
                onSelectTool={selectDrawingTool}
                onSetColor={setActiveColor}
                onDeleteSelected={() => selectedDrawing && deleteDrawing(selectedDrawing.id)}
                onClearAll={clearAllDrawings}
                onUpdateDrawing={updateDrawing}
              />
              <div ref={chartRef} className="rounded border border-gray-800 h-96"
                   style={{ cursor: interaction.activeTool ? 'crosshair' : undefined }} />
            </div>
            {effectiveShowStochastics && (
              <div>
                <p className="text-[10px] text-gray-600 mb-0.5 px-1">Slow Stoch %K (white) / %D (orange) · 14/3</p>
                <div ref={stoRef} className="rounded border border-gray-800 h-24" />
              </div>
            )}
            <div>
              <p className="text-[10px] text-gray-600 mb-0.5 px-1">RSI 14 · <span className="text-emerald-700/70">≤40 add zone</span></p>
              <div ref={rsiRef} className="rounded border border-gray-800 h-24" />
            </div>
            <div>
              <p className="text-[10px] text-gray-600 mb-0.5 px-1">MACD 12/26/9 · <span className="text-blue-600/60">line</span> / <span className="text-orange-500/60">signal</span></p>
              <div ref={macdRef} className="rounded border border-gray-800 h-24" />
            </div>
            {!isIntraday && (
              <div>
                <p className="text-[10px] text-gray-600 mb-0.5 px-1">TSI 25/13 · <span className="text-blue-400/60">TSI</span> / <span className="text-orange-400/60">signal</span> · ±25 bands</p>
                <div ref={tsiRef} className="rounded border border-gray-800 h-24" />
              </div>
            )}
            <div>
              <p className="text-[10px] text-gray-600 mb-0.5 px-1">Williams %R 14 · <span className="text-red-400/60">-20 overbought</span> / <span className="text-emerald-400/60">-80 oversold</span></p>
              <div ref={willrRef} className="rounded border border-gray-800 h-24" />
            </div>
            <div>
              <p className="text-[10px] text-gray-600 mb-0.5 px-1">CCI 20 · <span className="text-sky-400/60">cyan line</span> · <span className="text-red-400/60">±100 bands</span></p>
              <div ref={cciRef} className="rounded border border-gray-800 h-24" />
            </div>
            <div>
              <p className="text-[10px] text-gray-600 mb-0.5 px-1">OBV · <span className="text-orange-400/60">on-balance volume</span></p>
              <div ref={obvRef} className="rounded border border-gray-800 h-24" />
            </div>
            <div>
              <p className="text-[10px] text-gray-600 mb-0.5 px-1">ATR 14 · <span className="text-fuchsia-400/60">average true range</span></p>
              <div ref={atrPanelRef} className="rounded border border-gray-800 h-24" />
            </div>

            {/* ── Gann / Astro sub-charts ── */}
            {showBradleyPane && (
              <div>
                <div className="flex items-center gap-2 px-1 mb-0.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-amber-400/70 flex-shrink-0" />
                  <p className="text-[10px] text-amber-400/70">Bradley Siderograph · <span className="text-gray-500">planetary aspect energy · H/L = turning points</span></p>
                  <button onClick={() => setShowBradleyPane(false)} className="ml-auto text-[9px] text-gray-600 hover:text-gray-400">✕</button>
                </div>
                <div ref={bradleyRef} className="rounded border border-amber-900/30 h-36" />
              </div>
            )}

            {showPlanetsPane && (
              <div>
                <div className="flex items-center gap-2 px-1 mb-0.5 flex-wrap">
                  <span className="w-1.5 h-1.5 rounded-full bg-fuchsia-400/70 flex-shrink-0" />
                  <p className="text-[10px] text-fuchsia-400/70">Path of Planets · <span className="text-gray-500">geocentric ecliptic longitude (0–360°)</span></p>
                  <button onClick={() => setShowPlanetsPane(false)} className="ml-auto text-[9px] text-gray-600 hover:text-gray-400">✕</button>
                </div>
                {/* Planet colour legend */}
                <div className="flex flex-wrap gap-x-3 gap-y-0.5 px-1 mb-1">
                  {[['Sun','#fbbf24'],['Moon','#e2e8f0'],['Mercury','#a78bfa'],['Venus','#f9a8d4'],['Mars','#ef4444'],['Jupiter','#fb923c'],['Saturn','#94a3b8']].map(([name,color]) => (
                    <span key={name} className="text-[9px] flex items-center gap-1">
                      <span style={{ background: color }} className="inline-block w-2 h-0.5 rounded-full" />
                      <span style={{ color }}>{name}</span>
                    </span>
                  ))}
                </div>
                <div ref={planetsRef} className="rounded border border-fuchsia-900/30 h-40" />
              </div>
            )}

            {/* Toggle row for Gann/Astro sub-charts */}
            {!isIntraday && (
              <div className="flex gap-2 px-1 pt-1">
                <button
                  onClick={() => setShowBradleyPane(v => !v)}
                  className={`text-[10px] px-2 py-0.5 rounded border transition-colors ${showBradleyPane ? 'border-amber-500/60 text-amber-400 bg-amber-950/30' : 'border-gray-700 text-gray-500 hover:text-amber-400 hover:border-amber-700'}`}
                >
                  ⊙ Bradley
                </button>
                <button
                  onClick={() => setShowPlanetsPane(v => !v)}
                  className={`text-[10px] px-2 py-0.5 rounded border transition-colors ${showPlanetsPane ? 'border-fuchsia-500/60 text-fuchsia-400 bg-fuchsia-950/30' : 'border-gray-700 text-gray-500 hover:text-fuchsia-400 hover:border-fuchsia-700'}`}
                >
                  ♃ Planets
                </button>
              </div>
            )}

            {/* Clean mode badge — still relevant for overlay indicators */}
            {cleanMode && !isIntraday && (
              <div className="flex items-center gap-2 px-2 py-1 mt-0.5">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 flex-shrink-0" />
                <span className="text-[10px] text-emerald-500/70">
                  {activePatternName
                    ? `Showing: ${activePatternName} · click chip above or card to change`
                    : 'Clean · select a pattern from the list to overlay it'}
                </span>
              </div>
            )}
          </div>
        )}

      </div>

      {/* Sidebar — hidden when fullChart is active */}
      {!fullChart && <div className="w-64 flex-shrink-0">

        {/* ── Decision Trace ─────────────────────────────────────────────── */}
        {(() => {
          // Extract raw trace text
          let rawTrace = ''
          if (confluenceData?.decision_trace) {
            try { rawTrace = JSON.parse(confluenceData.decision_trace).text ?? '' }
            catch { rawTrace = confluenceData.decision_trace }
          }

          // Parse into structured sections
          type TRow     = { key: string; val: string }
          type TSection = { heading: string | null; rows: TRow[] }
          const sections: TSection[] = []
          if (rawTrace) {
            const lines = rawTrace.split('\n').map((l: string) => l.trim())
            let cur: TSection = { heading: null, rows: [] }
            for (const line of lines) {
              if (!line || /^[═─]+$/.test(line)) continue
              if (/^TRADE SIGNAL\s*[—–-]/i.test(line)) continue
              const div = line.match(/^──\s+(.+?)\s+──/)
              if (div) {
                if (cur.rows.length || cur.heading) sections.push(cur)
                cur = { heading: div[1], rows: [] }
                continue
              }
              const kv = line.match(/^([^:]+?):\s{2,}(.+)$/)
              if (kv) { cur.rows.push({ key: kv[1].trim(), val: kv[2].trim() }); continue }
              if (/^[A-Z][A-Z\s]+:/.test(line)) {
                const sv = line.match(/^([^:]+?):\s+(.+)$/)
                if (sv) cur.rows.push({ key: sv[1].trim(), val: sv[2].trim() })
              }
            }
            if (cur.rows.length || cur.heading) sections.push(cur)
          }

          // In collapsed mode: show section 0 (general info) + score row from last section
          const scoreRow = sections.flatMap(s => s.rows).find(r => r.key === 'CONFLUENCE SCORE')
          const visibleSections: TSection[] = traceExpanded
            ? sections
            : sections.length > 0 ? [sections[0]] : []

          const renderRow = (row: TRow, ri: number) => {
            const isScore = row.key === 'CONFLUENCE SCORE'
            const scoreColor = isScore
              ? row.val.includes('HIGH')   ? '#22c55e'
              : row.val.includes('MEDIUM') ? '#eab308'
              : row.val.includes('WATCH')  ? '#a78bfa'
              : '#9ca3af'
              : undefined
            return (
              <div key={ri} className="flex justify-between items-baseline gap-3">
                <span className={`text-[10px] flex-shrink-0 leading-snug ${isScore ? 'font-semibold text-gray-300' : 'text-gray-500'}`}>
                  {row.key}
                </span>
                <span
                  className={`text-[10px] font-mono text-right leading-snug ${isScore ? 'font-bold' : 'text-gray-200'}`}
                  style={scoreColor ? { color: scoreColor } : undefined}
                >
                  {row.val}
                </span>
              </div>
            )
          }

          const hasContent = sections.length > 0
          const canExpand  = sections.length > 1 || (sections.length === 1 && sections[0].rows.length > 3)

          return (
            <div className="bg-gray-900 rounded-lg border border-gray-800 p-3 mb-3">
              {/* Header row */}
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Decision Trace</h3>
                <div className="flex items-center gap-2">
                  <span className="text-[9px] text-gray-600 font-mono">{symbol}</span>
                  {!cc.trace && hasContent && canExpand && (
                    <button
                      onClick={() => setTraceExpanded(p => !p)}
                      className="text-[9px] text-gray-600 hover:text-gray-400 transition-colors font-mono"
                    >
                      {traceExpanded ? '▲ less' : '▼ more'}
                    </button>
                  )}
                  {collapseBtn('trace')}
                </div>
              </div>

              {!cc.trace && (!hasContent ? (
                <div className="space-y-1.5 py-0.5">
                  <p className="text-[10px] text-gray-600 italic">No confluence data for {symbol}</p>
                </div>
              ) : (
                <div className="space-y-3">
                  {/* Visible sections */}
                  {visibleSections.map((sec, si) => (
                    <div key={si}>
                      {sec.heading && (
                        <p className="text-[9px] font-semibold text-gray-500 uppercase tracking-widest mb-1.5 pb-1 border-b border-gray-800">
                          {sec.heading}
                        </p>
                      )}
                      <div className="space-y-1.5">
                        {sec.rows.map((row, ri) => renderRow(row, ri))}
                      </div>
                    </div>
                  ))}

                  {/* When collapsed, always pin the CONFLUENCE SCORE summary row */}
                  {!traceExpanded && scoreRow && sections.length > 1 && (
                    <div className="pt-2 border-t border-gray-800">
                      {renderRow(scoreRow, 99)}
                    </div>
                  )}

                  {/* Expand hint when there's more content */}
                  {!traceExpanded && canExpand && (
                    <button
                      onClick={() => setTraceExpanded(true)}
                      className="w-full text-[9px] text-gray-700 hover:text-gray-500 transition-colors text-center pt-1"
                    >
                      ▼ Show trade plan &amp; component scores
                    </button>
                  )}
                </div>
              ))}
            </div>
          )
        })()}

        {/* Harmonic Pattern Details Panel */}
        {(() => {
          const activeHarmonic = patterns?.find(p => {
            const c = p.pattern_category ?? ''
            const n = (p.pattern_name ?? '').toLowerCase()
            return c === 'harmonic' || ['gartley', 'bat', 'butterfly', 'crab', 'cypher'].some(h => n.includes(h))
          })
          if (!activeHarmonic) return null

          const target = (activeHarmonic as unknown as Record<string, unknown>).target as number | undefined
          const invalidation = (activeHarmonic as unknown as Record<string, unknown>).invalidation_level as number | undefined
          const ratios = (activeHarmonic as unknown as Record<string, unknown>).ratios as Record<string, number> | undefined
          const confidence = activeHarmonic.confidence ?? activeHarmonic.probability ?? 0
          const status = activeHarmonic.status ?? 'FORMING'

          const riskReward = target && invalidation && invalidation > 0
            ? Math.abs((target - (quote?.price ?? 0)) / (Math.abs(quote?.price ?? 0 - invalidation)))
            : null

          return (
            <div className="bg-cyan-950/30 rounded-lg border border-cyan-800/50 p-3 mb-3">
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-xs font-semibold text-cyan-400 uppercase tracking-wider">
                  {activeHarmonic.pattern_name}
                </h3>
                <div className="flex items-center gap-1">
                  <span className={`text-[9px] px-1.5 py-0.5 rounded font-mono ${
                    status === 'BREAKOUT' ? 'bg-emerald-900/50 text-emerald-300' :
                    status === 'READY' ? 'bg-yellow-900/50 text-yellow-300' :
                    status === 'COMPLETED' ? 'bg-blue-900/50 text-blue-300' :
                    status === 'FAILED' ? 'bg-red-900/50 text-red-300' :
                    'bg-gray-800 text-gray-400'
                  }`}>
                    {status}
                  </span>
                  {collapseBtn('harmonic')}
                </div>
              </div>

              {!cc.harmonic && <>
              {/* Ratios */}
              {ratios && (
                <div className="mb-2 pt-2 border-t border-cyan-800/30">
                  <p className="text-[9px] text-cyan-500 uppercase tracking-wide mb-1 font-semibold">Fibonacci Ratios</p>
                  <div className="grid grid-cols-2 gap-1 text-[9px]">
                    {Object.entries(ratios).map(([key, val]) => (
                      <div key={key} className="flex justify-between">
                        <span className="text-gray-500">{key}:</span>
                        <span className="text-cyan-300 font-mono">{(val * 100).toFixed(1)}%</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Levels */}
              <div className="mb-2 pt-2 border-t border-cyan-800/30 space-y-1">
                <div className="flex items-center justify-between text-[9px]">
                  <span className="text-gray-500">Current</span>
                  <span className="text-gray-200 font-mono">${quote?.price?.toFixed(2)}</span>
                </div>
                {target != null && (
                  <div className="flex items-center justify-between text-[9px]">
                    <span className="text-emerald-500">🎯 Target</span>
                    <span className="text-emerald-300 font-mono">${target.toFixed(2)}</span>
                  </div>
                )}
                {invalidation != null && (
                  <div className="flex items-center justify-between text-[9px]">
                    <span className="text-red-500">❌ Stop</span>
                    <span className="text-red-300 font-mono">${invalidation.toFixed(2)}</span>
                  </div>
                )}
                {riskReward != null && (
                  <div className="flex items-center justify-between text-[9px]">
                    <span className="text-amber-500">R:R</span>
                    <span className="text-amber-300 font-mono">{riskReward.toFixed(2)}:1</span>
                  </div>
                )}
              </div>

              {/* Confidence */}
              <div className="pt-2 border-t border-cyan-800/30">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-[9px] text-gray-500">Confidence</span>
                  <span className="text-[9px] text-cyan-300 font-mono">{Math.round(confidence)}</span>
                </div>
                <div className="h-1 bg-gray-800 rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full bg-cyan-400/70"
                    style={{ width: `${Math.min(confidence, 100)}%` }}
                  />
                </div>
              </div>
              </>}
            </div>
          )
        })()}

        {/* Pattern Detection Panel */}
        <div className="bg-gray-900 rounded-lg border border-gray-800 p-3 mb-3">
          {/* Header */}
          <div className="flex items-center justify-between mb-1.5">
            <div className="flex items-center gap-1.5">
              <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Pattern Detection</h3>
              <span className="text-[8px] text-gray-600 font-mono bg-gray-800 px-1 py-0.5 rounded">STAGE 4 / 5</span>
            </div>
            <div className="flex items-center gap-1">
            <span className="text-[10px] text-gray-600 font-mono">
              {detectedPatterns.length > 0 ? `${detectedPatterns.length} active` : 'none'}
            </span>
            {collapseBtn('patterns')}
            </div>
          </div>
          {!cc.patterns && <>{/* Stage 4 Signal Summary */}
          {detectedPatterns.length > 0 && (() => {
            const bullish  = detectedPatterns.filter(p => p.direction === 'bullish' && p.status !== 'NOT_PRESENT' && p.status !== 'FAILED')
            const bearish  = detectedPatterns.filter(p => p.direction === 'bearish' && p.status !== 'NOT_PRESENT' && p.status !== 'FAILED')
            const bullScore = bullish.reduce((s, p) => s + (p.confidence ?? 0), 0)
            const bearScore = bearish.reduce((s, p) => s + (p.confidence ?? 0), 0)
            const total = bullScore + bearScore
            const net = total > 0 ? ((bullScore - bearScore) / total) * 100 : 0
            const signal = net > 15 ? 'bullish' : net < -15 ? 'bearish' : 'neutral'
            const color  = signal === 'bullish' ? '#22c55e' : signal === 'bearish' ? '#f87171' : '#94a3b8'
            const arrow  = signal === 'bullish' ? '▲' : signal === 'bearish' ? '▼' : '—'
            return (
              <div className="flex items-center justify-between mb-2 px-1.5 py-1 rounded bg-gray-800/60 border border-gray-700/50">
                <div className="flex items-center gap-1">
                  <span style={{ color }} className="text-[10px] font-bold">{arrow} {signal.toUpperCase()}</span>
                  <span className="text-[9px] text-gray-500">signal</span>
                </div>
                <div className="flex items-center gap-2 text-[9px] text-gray-500">
                  <span style={{ color: '#22c55e' }}>▲ {bullish.length}</span>
                  <span style={{ color: '#f87171' }}>▼ {bearish.length}</span>
                  <span className="text-gray-600">score {Math.abs(Math.round(net))}</span>
                </div>
              </div>
            )
          })()}

          {/* Category filter tabs */}
          {detectedPatterns.length > 0 && (() => {
            const cats = (['chart', 'harmonic', 'gann', 'wyckoff'] as const).filter(
              cat => detectedPatterns.some(p => classifyPattern(p) === cat),
            )
            if (cats.length < 2) return null
            return (
              <div className="flex gap-1 mb-2 overflow-x-auto pb-0.5 scrollbar-none">
                <button
                  onClick={() => setPatternCategoryFilter(null)}
                  className="flex-shrink-0 px-2 py-0.5 rounded text-[10px] font-medium border transition-colors"
                  style={{
                    borderColor: patternCategoryFilter === null ? '#6b728080' : '#374151',
                    color: patternCategoryFilter === null ? '#e5e7eb' : '#6b7280',
                    backgroundColor: patternCategoryFilter === null ? '#374151' : 'transparent',
                  }}
                >
                  All <span className="opacity-50">{detectedPatterns.length}</span>
                </button>
                {cats.map(cat => {
                  const count = detectedPatterns.filter(p => classifyPattern(p) === cat).length
                  const color = patternColor(cat, undefined)
                  const isActive = patternCategoryFilter === cat
                  return (
                    <button
                      key={cat}
                      onClick={() => setPatternCategoryFilter(prev => prev === cat ? null : cat)}
                      className="flex-shrink-0 px-2 py-0.5 rounded text-[10px] font-medium border transition-colors"
                      style={{
                        borderColor: isActive ? color + '80' : '#374151',
                        color: isActive ? color : '#6b7280',
                        backgroundColor: isActive ? color + '18' : 'transparent',
                      }}
                    >
                      {cat.charAt(0).toUpperCase() + cat.slice(1)}{' '}
                      <span className="opacity-50">{count}</span>
                    </button>
                  )
                })}
              </div>
            )
          })()}

          {/* Pattern cards */}
          <div className="space-y-1.5 max-h-[360px] overflow-y-auto pr-0.5">
            {filteredPatterns.length === 0 ? (
              <p className="text-[11px] text-gray-600 text-center py-4">
                {detectedPatterns.length === 0 ? 'No patterns detected.' : `No ${patternCategoryFilter} patterns active.`}
              </p>
            ) : (
              filteredPatterns.map((p, i) => {
                const kind = classifyPattern(p)
                const color = patternColor(kind, p.direction)
                const selected = activePatternName === p.pattern_name
                const confidence = Math.round((p.confidence ?? p.probability ?? p.strength * 100) ?? 0)
                const status = p.status ?? 'FORMING'
                const hasBreakout = p.breakout_level != null
                const hasGann = p.fan_lines && p.fan_lines.length > 0

                return (
                  <div
                    key={`${p.date}-${p.pattern_name}-${i}`}
                    className={`rounded border transition-all overflow-hidden h-[68px] flex flex-col ${
                      selected ? 'border-gray-600 shadow-sm' : 'border-gray-800/80'
                    }`}
                    style={{
                      borderLeftWidth: '3px',
                      borderLeftColor: selected ? color : color + '40',
                      backgroundColor: selected ? '#1a1a1a' : '#0d0d0d',
                    }}
                  >
                    {/* Card header — click to select / deselect overlay */}
                    <button
                      className="w-full text-left px-2 pt-2 pb-1.5 hover:bg-white/[0.02] transition-colors flex-1"
                      onClick={() => setActivePatternName(prev => prev === p.pattern_name ? null : p.pattern_name)}
                    >
                      <div className="flex items-center gap-1.5 mb-1">
                        {/* Selection dot */}
                        <span className="w-1.5 h-1.5 rounded-full flex-shrink-0 transition-colors"
                          style={{ backgroundColor: selected ? color : '#374151' }} />
                        <span className="text-[11px] font-semibold flex-1 leading-tight text-gray-200">
                          {p.pattern_name}
                        </span>
                        <span className="text-[10px] font-mono font-bold flex-shrink-0" style={{ color }}>
                          {confidence}%
                        </span>
                      </div>

                      {/* Confidence bar */}
                      <div className="h-0.5 rounded-full bg-gray-800 mb-1.5">
                        <div
                          className="h-full rounded-full transition-all"
                          style={{ width: `${confidence}%`, backgroundColor: color + 'a0' }}
                        />
                      </div>

                      {/* Status + direction row */}
                      <div className="flex items-center gap-1.5 flex-wrap">
                        <span className={`text-[9px] px-1.5 py-px rounded font-medium ${
                          status === 'BREAKOUT' || status === 'COMPLETED'
                            ? 'bg-emerald-900/50 text-emerald-300'
                            : status === 'READY'
                            ? 'bg-amber-900/50 text-amber-300'
                            : 'bg-gray-800/80 text-gray-500'
                        }`}>{status}</span>
                        {p.direction && p.direction !== 'neutral' && (
                          <span className={`text-[9px] font-bold ${p.direction === 'bullish' ? 'text-emerald-400' : 'text-red-400'}`}>
                            {p.direction === 'bullish' ? '▲ Bullish' : '▼ Bearish'}
                          </span>
                        )}
                        {p.phase && (
                          <span className="text-[9px] px-1.5 py-px rounded bg-purple-950/60 text-purple-300 ml-auto">
                            Phase {p.phase}
                          </span>
                        )}
                      </div>
                    </button>

                    {/* Analysis section — only visible when card is selected */}
                    {selected && (
                    <div className="px-2 pb-2 pt-0.5 border-t border-gray-800/30 space-y-1.5 flex-1 overflow-y-auto">

                      {/* Key levels row — always shown */}
                      {hasBreakout && (
                        <div className="grid grid-cols-3 gap-1 pt-0.5">
                          <div className="text-center">
                            <div className="text-[8px] text-gray-600 uppercase tracking-wide mb-0.5">Breakout</div>
                            <div className="text-[10px] font-mono font-bold" style={{ color }}>
                              ${p.breakout_level!.toFixed(2)}
                            </div>
                          </div>
                          {(p.target ?? p.projected_target) != null && (
                            <div className="text-center">
                              <div className="text-[8px] text-gray-600 uppercase tracking-wide mb-0.5">Target</div>
                              <div className="text-[10px] font-mono font-bold text-emerald-400">
                                ${(p.target ?? p.projected_target)!.toFixed(2)}
                              </div>
                            </div>
                          )}
                          {p.invalidation_level != null && (
                            <div className="text-center">
                              <div className="text-[8px] text-gray-600 uppercase tracking-wide mb-0.5">Stop</div>
                              <div className="text-[10px] font-mono font-bold text-red-400">
                                ${p.invalidation_level.toFixed(2)}
                              </div>
                            </div>
                          )}
                        </div>
                      )}

                      {/* Wyckoff: phase description + event tags */}
                      {p.phase_label && (
                        <div className="flex items-center gap-1.5">
                          <span className="text-[8px] px-1.5 py-px rounded-md bg-purple-900/40 text-purple-300 border border-purple-700/40 font-bold">
                            Phase {p.phase}
                          </span>
                          <span className="text-[9px] text-purple-300/80">{p.phase_label}</span>
                        </div>
                      )}
                      {p.events && p.events.length > 0 && (
                        <div className="flex flex-wrap gap-1">
                          {p.events.map((ev: string) => {
                            const pinned = p.event_points?.some(ep => ep.label === ev)
                            return (
                              <span key={ev} className={`text-[9px] px-1 py-px rounded border flex items-center gap-0.5 ${
                                pinned
                                  ? 'bg-purple-900/40 text-purple-300 border-purple-700/50'
                                  : 'bg-purple-950/30 text-purple-400/50 border-purple-900/20'
                              }`}>
                                {pinned && <span className="text-[7px]">📍</span>}
                                {ev}
                              </span>
                            )
                          })}
                        </div>
                      )}
                      {/* Wyckoff: support/resistance levels */}
                      {(p.support_level != null || p.resistance_level != null) && (
                        <div className="flex gap-2 text-[9px]">
                          {p.resistance_level != null && (
                            <span className="text-red-400/70">R: ${p.resistance_level.toFixed(2)}</span>
                          )}
                          {p.support_level != null && (
                            <span className="text-emerald-400/70">S: ${p.support_level.toFixed(2)}</span>
                          )}
                        </div>
                      )}

                      {/* Gann: all fan levels table */}
                      {hasGann && p.fan_lines && (() => {
                        const lastClose = quote?.price ?? 0
                        const sorted = [...p.fan_lines]
                          .filter(f => f.current_price > 0)
                          .sort((a, b) => b.current_price - a.current_price)
                        return (
                          <div>
                            <p className="text-[8px] text-gray-600 uppercase tracking-wide mb-1">Fan Levels</p>
                            <div className="space-y-0.5">
                              {sorted.map(f => {
                                const dist = lastClose > 0 ? ((f.current_price / lastClose - 1) * 100) : 0
                                const isAbove = dist >= 0
                                const isNearest = Math.abs(dist) < 3
                                return (
                                  <div key={f.angle} className={`flex items-center gap-1.5 text-[9px] px-1 py-px rounded ${isNearest ? 'bg-amber-950/30' : ''}`}>
                                    <span className="w-7 font-mono font-bold text-amber-500/80">{f.angle}</span>
                                    <span className="font-mono text-gray-300 flex-1">${f.current_price.toFixed(2)}</span>
                                    <span className={`font-mono text-[8px] ${isAbove ? 'text-emerald-600' : 'text-red-600'}`}>
                                      {isAbove ? '+' : ''}{dist.toFixed(1)}%
                                    </span>
                                  </div>
                                )
                              })}
                            </div>
                            {p.time_cycles && (() => {
                              const next = p.time_cycles.find(c => c.status === 'future')
                              return next ? (
                                <p className="text-[8px] text-amber-500/50 mt-1">Next cycle: {next.cycle_bars}-bar</p>
                              ) : null
                            })()}
                          </div>
                        )
                      })()}

                      {/* Harmonic: ratio info */}
                      {classifyPattern(p) === 'harmonic' && p.point_labels && (
                        <div className="flex items-center gap-1 text-[9px] text-gray-500">
                          <span className="font-mono tracking-wider text-gray-400">{p.point_labels.join(' → ')}</span>
                        </div>
                      )}
                    </div>
                    )}
                  </div>
                )
              })
            )}
          </div>

          {/* Active pattern footer */}
          {activePatternName && (
            <div className="mt-2 pt-2 border-t border-gray-800/60 flex items-center justify-between">
              <p className="text-[10px] text-blue-400 truncate flex-1 mr-2">⬤ {activePatternName}</p>
              <button
                onClick={() => setActivePatternName(null)}
                className="flex-shrink-0 text-[10px] text-gray-600 hover:text-gray-400 transition-colors"
              >
                ✕ Clear
              </button>
            </div>
          )}
          </>}
        </div>


        {/* ── Position Builder card ─────────────────────────────────────────── */}
        {currentAddSignal && (
          <div className={`rounded-lg border p-3 mb-3 ${
            currentAddSignal.score >= 3 ? 'bg-emerald-950/40 border-emerald-800/50' :
            currentAddSignal.score === 2 ? 'bg-yellow-950/30 border-yellow-800/40' :
            'bg-gray-900 border-gray-800'
          }`}>
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Position Builder</h3>
              <div className="flex items-center gap-1">
                <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${
                  currentAddSignal.score >= 3 ? 'bg-emerald-800/60 text-emerald-300' :
                  currentAddSignal.score === 2 ? 'bg-yellow-800/60 text-yellow-300' :
                  'bg-gray-800 text-gray-500'
                }`}>
                  {currentAddSignal.score >= 3 ? '▲ ADD' : currentAddSignal.score === 2 ? '◐ WATCH' : '○ WAIT'}
                </span>
                {collapseBtn('position')}
              </div>
            </div>
            {!cc.position && <>
            <div className="space-y-1.5">
              {[
                { label: 'RSI',       value: currentAddSignal.rsi?.toFixed(1),           ok: currentAddSignal.rsiOk,   hint: '< 40' },
                { label: 'Stoch %K', value: currentAddSignal.stochK.toFixed(1),          ok: currentAddSignal.stochOk, hint: '< 30' },
                { label: 'BB Lower', value: currentAddSignal.bbOk ? 'At/Below' : 'Above', ok: currentAddSignal.bbOk,   hint: '≤ BB−' },
                { label: 'Red Day',  value: currentAddSignal.redDay ? 'Yes' : 'No',      ok: currentAddSignal.redDay,  hint: 'Close<Open' },
              ].map(({ label, value, ok, hint }) => (
                <div key={label} className="flex items-center gap-1 text-[10px]">
                  <span className={`w-2 h-2 rounded-full flex-shrink-0 ${ok ? 'bg-emerald-500' : 'bg-gray-700'}`} />
                  <span className="text-gray-500 w-14 flex-shrink-0">{label}</span>
                  <span className={`font-mono font-bold flex-1 ${ok ? 'text-emerald-400' : 'text-gray-600'}`}>{value ?? '—'}</span>
                  {!ok && <span className="text-gray-700 text-[9px]">{hint}</span>}
                  {ok && <span className="text-emerald-600 text-[9px]">✓</span>}
                </div>
              ))}
            </div>
            <div className="mt-2.5 pt-2 border-t border-gray-800/60">
              <div className="flex items-center gap-2 mb-1">
                <div className="flex-1 h-1.5 bg-gray-800 rounded-full overflow-hidden">
                  <div className="h-full rounded-full transition-all" style={{
                    width: `${currentAddSignal.score * 25}%`,
                    backgroundColor: currentAddSignal.score >= 3 ? '#10b981' : currentAddSignal.score === 2 ? '#f59e0b' : '#6b7280',
                  }} />
                </div>
                <span className="text-[10px] text-gray-500 font-mono flex-shrink-0">{currentAddSignal.score}/4</span>
              </div>
              <p className="text-[10px] text-gray-600 leading-tight">
                {currentAddSignal.score >= 3 ? 'Strong setup — consider adding +10% to position' :
                 currentAddSignal.score === 2 ? 'Partial signal — wait for red day or deeper pullback' :
                 'No signal — hold cash, wait for pullback conditions'}
              </p>
            </div>
            </>}
          </div>
        )}

        {/* Fundamentals panel */}
        {fundamentals && !fundamentals.is_etf && (
          <div className="bg-gray-900 rounded-lg border border-gray-800 p-3 mt-3">
            <div className="flex items-center justify-between mb-2">
              <p className="text-[10px] text-gray-500 uppercase font-semibold tracking-wider">Fundamentals</p>
              <div className="flex items-center gap-1">
                {fundamentals.sector && (
                  <span className="text-[9px] text-gray-600 bg-gray-800 px-1.5 py-0.5 rounded truncate max-w-[110px]" title={fundamentals.sector}>
                    {fundamentals.sector}
                  </span>
                )}
                {collapseBtn('fundamentals')}
              </div>
            </div>

            {!cc.fundamentals && <>
            {/* Company name */}
            {fundamentals.company_name && (
              <p className="text-[11px] text-gray-300 font-medium mb-2 leading-tight">{fundamentals.company_name}</p>
            )}

            {/* Price range: 52w */}
            {(fundamentals.year_high != null || fundamentals.year_low != null) && (() => {
              const lo  = fundamentals.year_low  ?? 0
              const hi  = fundamentals.year_high ?? 0
              const cur = fundamentals.price ?? 0
              const pct = hi > lo ? ((cur - lo) / (hi - lo)) * 100 : 0
              return (
                <div className="mb-2">
                  <div className="flex justify-between text-[9px] text-gray-600 mb-0.5">
                    <span>${lo.toFixed(0)}</span>
                    <span className="text-gray-500">52w range</span>
                    <span>${hi.toFixed(0)}</span>
                  </div>
                  <div className="relative h-1.5 bg-gray-800 rounded-full overflow-visible">
                    <div className="h-full bg-gradient-to-r from-red-700 via-amber-600 to-emerald-600 rounded-full" />
                    <div
                      className="absolute top-1/2 -translate-y-1/2 w-2 h-2 rounded-full bg-white border border-gray-900 shadow"
                      style={{ left: `calc(${Math.min(Math.max(pct, 2), 98)}% - 4px)` }}
                    />
                  </div>
                </div>
              )
            })()}

            {/* Key metrics grid */}
            <div className="grid grid-cols-2 gap-x-3 gap-y-1 mb-2">
              {[
                { label: 'Mkt Cap',   value: fundamentals.market_cap != null ? `$${(fundamentals.market_cap / 1e9).toFixed(1)}B` : null },
                { label: 'P/E',       value: fundamentals.pe_ratio   != null ? fundamentals.pe_ratio.toFixed(1)   : null },
                { label: 'P/S',       value: fundamentals.ps_ratio   != null ? fundamentals.ps_ratio.toFixed(1)   : null },
                { label: 'P/B',       value: fundamentals.pb_ratio   != null ? fundamentals.pb_ratio.toFixed(1)   : null },
                { label: 'EV/EBITDA', value: fundamentals.ev_ebitda  != null ? fundamentals.ev_ebitda.toFixed(1)  : null },
                { label: 'Beta',      value: fundamentals.beta       != null ? fundamentals.beta.toFixed(2)       : null },
              ].filter(m => m.value != null).map(m => (
                <div key={m.label} className="flex items-center justify-between">
                  <span className="text-[9px] text-gray-600">{m.label}</span>
                  <span className="text-[10px] text-gray-300 font-mono">{m.value}</span>
                </div>
              ))}
            </div>

            {/* Margins */}
            {(fundamentals.net_margin != null || fundamentals.operating_margin != null || fundamentals.gross_margin != null) && (
              <div className="mb-2 pt-2 border-t border-gray-800/60">
                <p className="text-[9px] text-gray-600 uppercase tracking-wide mb-1">Margins</p>
                {[
                  { label: 'Gross',   value: fundamentals.gross_margin,     color: '#22c55e' },
                  { label: 'Oper',    value: fundamentals.operating_margin,  color: '#3b82f6' },
                  { label: 'Net',     value: fundamentals.net_margin,        color: '#a78bfa' },
                ].filter(m => m.value != null).map(m => (
                  <div key={m.label} className="flex items-center gap-1.5 mb-0.5">
                    <span className="w-7 text-[9px] text-gray-600 flex-shrink-0">{m.label}</span>
                    <div className="flex-1 h-1 bg-gray-800 rounded-full overflow-hidden">
                      <div
                        className="h-full rounded-full transition-all"
                        style={{ width: `${Math.max(0, Math.min(100, m.value!))}%`, backgroundColor: m.color }}
                      />
                    </div>
                    <span className="text-[9px] font-mono flex-shrink-0" style={{ color: (m.value ?? 0) < 0 ? '#f87171' : m.color }}>
                      {(m.value ?? 0).toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>
            )}

            {/* Analyst target */}
            {fundamentals.analyst_target_1q != null && (
              <div className="mb-2 pt-2 border-t border-gray-800/60 flex items-center justify-between">
                <span className="text-[9px] text-gray-600">Analyst target</span>
                <div className="flex items-center gap-1.5">
                  <span className="text-[10px] font-mono text-amber-400">${fundamentals.analyst_target_1q.toFixed(2)}</span>
                  {fundamentals.price != null && fundamentals.analyst_target_1q != null && (() => {
                    const upside = ((fundamentals.analyst_target_1q - fundamentals.price) / fundamentals.price) * 100
                    return (
                      <span className={`text-[9px] font-mono ${upside >= 0 ? 'text-emerald-500' : 'text-red-400'}`}>
                        {upside >= 0 ? '+' : ''}{upside.toFixed(1)}%
                      </span>
                    )
                  })()}
                  {fundamentals.analyst_count_1q != null && (
                    <span className="text-[9px] text-gray-700">({fundamentals.analyst_count_1q})</span>
                  )}
                </div>
              </div>
            )}

            {/* Next earnings */}
            {fundamentals.next_earnings_date && (() => {
              const days = Math.round((new Date(fundamentals.next_earnings_date).getTime() - Date.now()) / 86_400_000)
              const warn = days <= 7
              return (
                <div className="pt-2 border-t border-gray-800/60 flex items-center justify-between">
                  <span className="text-[9px] text-gray-600">Next earnings</span>
                  <div className="flex items-center gap-1.5">
                    <span className={`text-[10px] font-mono ${warn ? 'text-amber-400' : 'text-gray-400'}`}>
                      {fundamentals.next_earnings_date}
                    </span>
                    {days >= 0 && (
                      <span className={`text-[9px] ${warn ? 'text-amber-500' : 'text-gray-600'}`}>
                        {days === 0 ? 'today' : `${days}d`}
                      </span>
                    )}
                    {warn && <span className="text-[9px] text-amber-500">⚠</span>}
                  </div>
                </div>
              )
            })()}

            {/* EPS estimate */}
            {fundamentals.next_eps_estimate != null && (
              <div className="flex items-center justify-between mt-1">
                <span className="text-[9px] text-gray-600">EPS est.</span>
                <span className="text-[10px] font-mono text-gray-400">${fundamentals.next_eps_estimate.toFixed(2)}</span>
              </div>
            )}
            </>}
          </div>
        )}

        {/* ── Cycle Analysis Panel ─────────────────────────────────── */}
        {showCycleWindow && cycleAdvanced && !cycleAdvanced.error && (
          <div className="bg-gray-900 rounded-lg border border-cyan-900/40 p-3 mt-3">
            <div className="flex items-center gap-1.5 mb-2">
              <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
              <h3 className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider">Cycle Analysis</h3>
              <span className="text-[8px] text-gray-600 font-mono bg-gray-800 px-1 py-0.5 rounded">FFT+Wavelet+Hilbert</span>
              <span className="ml-auto">{collapseBtn('cycles')}</span>
            </div>
            {!cc.cycles && <div className="space-y-1.5 text-[10px]">
              <div className="flex justify-between">
                <span className="text-gray-500">Dominant Cycle</span>
                <span className="text-cyan-300 font-mono">{cycleAdvanced.dominant_cycle_length} bars</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">Phase</span>
                <span className="font-mono" style={{ color: cycleAdvanced.cycle_phase < 0.3 || cycleAdvanced.cycle_phase > 0.7 ? '#34d399' : '#fbbf24' }}>
                  {(cycleAdvanced.cycle_phase * 100).toFixed(0)}%
                  {cycleAdvanced.cycle_phase < 0.15 ? ' ◉ Trough' : cycleAdvanced.cycle_phase > 0.85 ? ' ◉ Trough' : cycleAdvanced.cycle_phase > 0.4 && cycleAdvanced.cycle_phase < 0.6 ? ' ◉ Peak' : ''}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">Alignment Score</span>
                <span className={`font-mono ${cycleAdvanced.cycle_alignment_score > 0.7 ? 'text-emerald-400' : cycleAdvanced.cycle_alignment_score > 0.4 ? 'text-yellow-400' : 'text-gray-500'}`}>
                  {(cycleAdvanced.cycle_alignment_score * 100).toFixed(0)}%
                </span>
              </div>
              {cycleAdvanced.projected_peak_date && (
                <div className="flex justify-between">
                  <span className="text-gray-500">Next Peak</span>
                  <span className="text-orange-300 font-mono text-[9px]">{cycleAdvanced.projected_peak_date}</span>
                </div>
              )}
              {cycleAdvanced.projected_trough_date && (
                <div className="flex justify-between">
                  <span className="text-gray-500">Next Trough</span>
                  <span className="text-emerald-300 font-mono text-[9px]">{cycleAdvanced.projected_trough_date}</span>
                </div>
              )}
              {/* Hilbert instantaneous phase mini gauge */}
              <div className="mt-1 pt-1 border-t border-gray-800">
                <div className="flex justify-between mb-0.5">
                  <span className="text-[9px] text-gray-600">Hilbert Phase Velocity</span>
                  <span className="text-[9px] font-mono text-gray-400">{cycleAdvanced.hilbert?.phase_velocity?.toFixed(3) ?? '—'}</span>
                </div>
              </div>
            </div>}
          </div>
        )}

        {/* ── Market Force Gauge ──────────────────────────────────── */}
        {showForceGauge && marketForce && !marketForce.error && (
          <div className="bg-gray-900 rounded-lg border border-pink-900/40 p-3 mt-3">
            <div className="flex items-center gap-1.5 mb-2">
              <span className="w-1.5 h-1.5 rounded-full bg-pink-400" />
              <h3 className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider">Market Force</h3>
              <span className={`text-[9px] px-1.5 py-0.5 rounded font-semibold ${
                marketForce.bias === 'bullish' ? 'bg-emerald-900/40 text-emerald-300' :
                marketForce.bias === 'bearish' ? 'bg-red-900/40 text-red-300' :
                'bg-gray-800 text-gray-400'
              }`}>{marketForce.bias?.toUpperCase()}</span>
              <span className="ml-auto">{collapseBtn('force')}</span>
            </div>
            {!cc.force && <>{/* Net Force bar */}
            <div className="mb-2">
              <div className="flex justify-between text-[9px] mb-0.5">
                <span className="text-gray-600">Net Force</span>
                <span className="font-mono" style={{ color: marketForce.net_force > 0 ? '#34d399' : marketForce.net_force < 0 ? '#f87171' : '#9ca3af' }}>
                  {marketForce.net_force > 0 ? '+' : ''}{marketForce.net_force.toFixed(3)}
                </span>
              </div>
              <div className="h-1.5 bg-gray-800 rounded-full relative overflow-hidden">
                <div
                  className="absolute top-0 h-full rounded-full transition-all"
                  style={{
                    left:  marketForce.net_force >= 0 ? '50%' : `${50 + marketForce.net_force * 50}%`,
                    width: `${Math.abs(marketForce.net_force) * 50}%`,
                    backgroundColor: marketForce.net_force >= 0 ? '#34d399' : '#f87171',
                  }}
                />
                <div className="absolute top-0 left-1/2 w-px h-full bg-gray-600" />
              </div>
            </div>
            {/* Individual forces */}
            <div className="space-y-1 text-[9px]">
              {[
                { label: 'Trend',      value: marketForce.trend_force,      color: '#3b82f6' },
                { label: 'Liquidity',   value: marketForce.liquidity_force,  color: '#a855f7' },
                { label: 'Volatility',  value: marketForce.volatility_force, color: '#f59e0b' },
                { label: 'Cycle',       value: marketForce.cycle_force,      color: '#06b6d4' },
                { label: 'Pattern',     value: marketForce.pattern_force,    color: '#22c55e' },
              ].map(f => (
                <div key={f.label} className="flex items-center gap-2">
                  <span className="w-12 text-gray-500">{f.label}</span>
                  <div className="flex-1 h-1 bg-gray-800 rounded-full relative overflow-hidden">
                    <div
                      className="absolute top-0 h-full rounded-full"
                      style={{
                        left:  f.value >= 0 ? '50%' : `${50 + f.value * 50}%`,
                        width: `${Math.abs(f.value) * 50}%`,
                        backgroundColor: f.color,
                      }}
                    />
                    <div className="absolute top-0 left-1/2 w-px h-full bg-gray-700" />
                  </div>
                  <span className="w-10 text-right font-mono text-gray-400">
                    {f.value > 0 ? '+' : ''}{f.value.toFixed(2)}
                  </span>
                </div>
              ))}
            </div></>}
          </div>
        )}

        {/* ── Decision Nodes Panel ────────────────────────────────── */}
        {showDecisionNodes && confluenceNodes && !confluenceNodes.error && confluenceNodes.nodes?.length > 0 && (
          <div className="bg-gray-900 rounded-lg border border-orange-900/40 p-3 mt-3">
            <div className="flex items-center gap-1.5 mb-2">
              <span className="w-1.5 h-1.5 rounded-full bg-orange-400" />
              <h3 className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider">Decision Nodes</h3>
              <span className="text-[9px] text-gray-600 font-mono">{confluenceNodes.nodes.length}</span>
              <span className="ml-auto">{collapseBtn('nodes')}</span>
            </div>
            {!cc.nodes && <div className="space-y-1.5 max-h-48 overflow-y-auto">
              {confluenceNodes.nodes
                .filter(n => n.status === 'active')
                .sort((a, b) => b.confluence_score - a.confluence_score)
                .slice(0, 6)
                .map((node, i) => (
                <div key={i} className="flex items-center gap-2 py-1 border-b border-gray-800/50 last:border-0">
                  {/* Type badge */}
                  <span className={`text-[8px] px-1 py-0.5 rounded font-semibold ${
                    node.node_type === 'support' ? 'bg-emerald-900/40 text-emerald-400' : 'bg-red-900/40 text-red-400'
                  }`}>
                    {node.node_type === 'support' ? 'S' : 'R'}
                  </span>
                  {/* Price range */}
                  <div className="flex-1 min-w-0">
                    <div className="text-[10px] font-mono text-gray-200">
                      ${node.price_low.toFixed(2)} – ${node.price_high.toFixed(2)}
                    </div>
                    <div className="text-[8px] text-gray-600 truncate">
                      {node.supporting_signals?.join(' · ')}
                    </div>
                  </div>
                  {/* Score */}
                  <span className={`text-[10px] font-mono font-semibold ${
                    node.confluence_score > 0.7 ? 'text-orange-300' : node.confluence_score > 0.4 ? 'text-yellow-400' : 'text-gray-500'
                  }`}>
                    {(node.confluence_score * 100).toFixed(0)}
                  </span>
                </div>
              ))}
            </div>}
          </div>
        )}

        {/* ── Signal Confidence Panel ───────────────────────────────────────── */}
        {showSignalConfidence && signalConfidence && !signalConfidence.error && (
          <div className="bg-gray-900 rounded-lg border border-emerald-900/40 p-3 mt-3">
            <div className="flex items-center gap-1.5 mb-2">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
              <h3 className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider">Signal Confidence</h3>
              <span className="ml-auto">{collapseBtn('confidence')}</span>
            </div>
            {!cc.confidence && <>
            {/* Tier badge + score */}
            <div className="flex items-center gap-2 mb-2">
              <span className={`text-xs font-bold font-mono ${
                signalConfidence.confidence_tier === 'Very Reliable' ? 'text-emerald-300' :
                signalConfidence.confidence_tier === 'Reliable' ? 'text-blue-300' :
                signalConfidence.confidence_tier === 'Moderate' ? 'text-yellow-300' : 'text-gray-500'
              }`}>
                {signalConfidence.confidence_score.toFixed(0)}
              </span>
              <span className={`text-[9px] px-1.5 py-0.5 rounded font-semibold ${
                signalConfidence.confidence_tier === 'Very Reliable' ? 'bg-emerald-900/50 text-emerald-300' :
                signalConfidence.confidence_tier === 'Reliable' ? 'bg-blue-900/50 text-blue-300' :
                signalConfidence.confidence_tier === 'Moderate' ? 'bg-yellow-900/50 text-yellow-300' : 'bg-gray-800 text-gray-500'
              }`}>
                {signalConfidence.confidence_tier}
              </span>
            </div>
            {/* Component bars */}
            <div className="space-y-1">
              {(Object.entries(signalConfidence.components) as [string, number][]).map(([key, val]) => (
                <div key={key} className="flex items-center gap-1.5">
                  <span className="text-[8px] text-gray-500 w-16 truncate">{key.replace(/_/g, ' ')}</span>
                  <div className="flex-1 h-1 bg-gray-800 rounded-full overflow-hidden">
                    <div className="h-full rounded-full bg-emerald-500/70" style={{ width: `${Math.min(val, 100)}%` }} />
                  </div>
                  <span className="text-[8px] text-gray-500 font-mono w-6 text-right">{val.toFixed(0)}</span>
                </div>
              ))}
            </div>
            </>}
          </div>
        )}

        {/* ── Probability Bands Panel ─────────────────────────────────────────── */}
        {showProbBands && priceDistribution && !priceDistribution.error && (
          <div className="bg-gray-900 rounded-lg border border-indigo-900/40 p-3 mt-3">
            <div className="flex items-center gap-1.5 mb-2">
              <span className="w-1.5 h-1.5 rounded-full bg-indigo-400" />
              <h3 className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider">Price Distribution ({priceDistribution.horizon_days}d)</h3>
              <span className="ml-auto">{collapseBtn('probands')}</span>
            </div>
            {!cc.probands && <>
            {/* Up/Down probabilities */}
            <div className="flex gap-2 mb-2">
              <span className="text-[10px] font-mono text-emerald-400">▲ {(priceDistribution.probability_up * 100).toFixed(0)}%</span>
              <span className="text-[10px] font-mono text-red-400">▼ {(priceDistribution.probability_down * 100).toFixed(0)}%</span>
              <span className="text-[9px] text-gray-600">E[r] = {(priceDistribution.expected_return * 100).toFixed(2)}%</span>
            </div>
            {/* Quantile levels */}
            <div className="space-y-0.5">
              {priceDistribution.quantiles && ([
                { label: 'P90', value: (priceDistribution.quantiles as Record<string,number>).p90, color: 'text-emerald-400' },
                { label: 'P75', value: (priceDistribution.quantiles as Record<string,number>).p75, color: 'text-emerald-300' },
                { label: 'P50', value: (priceDistribution.quantiles as Record<string,number>).p50, color: 'text-gray-200' },
                { label: 'P25', value: (priceDistribution.quantiles as Record<string,number>).p25, color: 'text-red-300' },
                { label: 'P10', value: (priceDistribution.quantiles as Record<string,number>).p10, color: 'text-red-400' },
              ]).filter(q => q.value != null).map(q => {
                const pctFromCurrent = ((q.value - priceDistribution.current_price) / priceDistribution.current_price * 100)
                return (
                  <div key={q.label} className="flex items-center gap-2">
                    <span className="text-[9px] text-gray-500 w-6">{q.label}</span>
                    <span className={`text-[10px] font-mono ${q.color}`}>${q.value.toFixed(2)}</span>
                    <span className={`text-[8px] font-mono ${pctFromCurrent >= 0 ? 'text-emerald-600' : 'text-red-600'}`}>
                      {pctFromCurrent >= 0 ? '+' : ''}{pctFromCurrent.toFixed(1)}%
                    </span>
                  </div>
                )
              })}
            </div>
            {/* Targets */}
            {priceDistribution.targets && priceDistribution.targets.length > 0 && (
              <div className="mt-2 pt-2 border-t border-gray-800">
                <p className="text-[8px] text-gray-600 uppercase mb-1">Target Probabilities</p>
                {priceDistribution.targets.map((t, i) => (
                  <div key={i} className="flex items-center gap-1.5">
                    <span className={`text-[9px] ${t.direction === 'up' ? 'text-emerald-400' : 'text-red-400'}`}>
                      {t.direction === 'up' ? '▲' : '▼'}
                    </span>
                    <span className="text-[9px] font-mono text-gray-300">${t.price.toFixed(2)}</span>
                    <span className="text-[8px] font-mono text-gray-500">{(t.probability * 100).toFixed(0)}%</span>
                  </div>
                ))}
              </div>
            )}
            </>}
          </div>
        )}

        {/* ── Model Health Panel ──────────────────────────────────────────────── */}
        {showModelHealth && strategyHealth && !strategyHealth.error && (
          <div className="bg-gray-900 rounded-lg border border-orange-900/40 p-3 mt-3">
            <div className="flex items-center gap-1.5 mb-2">
              <span className={`w-1.5 h-1.5 rounded-full ${
                strategyHealth.overall_health === 'HEALTHY' ? 'bg-emerald-400' :
                strategyHealth.overall_health === 'WARNING' ? 'bg-yellow-400' : 'bg-red-400'
              }`} />
              <h3 className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider">Model Health</h3>
              <div className="flex items-center gap-1 ml-auto">
                <span className={`text-[9px] px-1.5 py-0.5 rounded font-semibold ${
                  strategyHealth.overall_health === 'HEALTHY' ? 'bg-emerald-900/50 text-emerald-300' :
                  strategyHealth.overall_health === 'WARNING' ? 'bg-yellow-900/50 text-yellow-300' : 'bg-red-900/50 text-red-300'
                }`}>
                  {strategyHealth.overall_health}
                </span>
                {collapseBtn('health')}
              </div>
            </div>
            {!cc.health && <>
            {/* Summary counts */}
            <div className="flex gap-3 mb-2 text-[9px] font-mono">
              <span className="text-emerald-400">✓ {strategyHealth.healthy_count}</span>
              <span className="text-yellow-400">⚠ {strategyHealth.warning_count}</span>
              <span className="text-red-400">✕ {strategyHealth.critical_count}</span>
            </div>
            {/* Strategy list */}
            <div className="space-y-1 max-h-40 overflow-y-auto">
              {strategyHealth.strategies.map((s, i) => (
                <div key={i} className="flex items-center gap-1.5 py-0.5 border-b border-gray-800/50 last:border-0">
                  <span className={`w-1 h-1 rounded-full ${
                    s.health_state === 'HEALTHY' ? 'bg-emerald-400' :
                    s.health_state === 'WARNING' ? 'bg-yellow-400' : 'bg-red-400'
                  }`} />
                  <span className="text-[9px] text-gray-300 flex-1 truncate">{s.setup_type}</span>
                  <span className={`text-[8px] px-1 py-0.5 rounded font-mono ${
                    s.action === 'maintain' ? 'text-emerald-500' :
                    s.action === 'reduce_size' ? 'text-yellow-500' : 'text-red-500'
                  }`}>
                    {s.action}
                  </span>
                  {s.win_rate_drift !== null && (
                    <span className={`text-[8px] font-mono ${s.win_rate_drift >= 0 ? 'text-emerald-600' : 'text-red-500'}`}>
                      {s.win_rate_drift >= 0 ? '+' : ''}{(s.win_rate_drift * 100).toFixed(1)}%
                    </span>
                  )}
                </div>
              ))}
            </div>
            </>}
          </div>
        )}

        {/* ── Options Flow Analysis ────────────────────────────────── */}
        {showOptionsFlow && optionsData && !optionsData.error && (
          <div className="bg-gray-900 rounded-lg border border-sky-900/40 p-3 mt-3">
            <div className="flex items-center gap-1.5 mb-2">
              <span className="w-1.5 h-1.5 rounded-full bg-sky-400" />
              <h3 className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider">Options Flow</h3>
              <span className={`text-[9px] px-1.5 py-0.5 rounded font-semibold ${
                optionsData.pc_signal === 'bullish' ? 'bg-emerald-900/40 text-emerald-300' :
                optionsData.pc_signal === 'bearish' ? 'bg-red-900/40 text-red-300' :
                'bg-yellow-900/40 text-yellow-300'
              }`}>{optionsData.pc_signal?.toUpperCase()}</span>
              <span className="ml-auto">{collapseBtn('options')}</span>
            </div>
            {!cc.options && <>
              {/* Put/Call Ratio */}
              <div className="grid grid-cols-2 gap-1.5 mb-2">
                <div className="bg-gray-800/60 rounded p-1.5">
                  <div className="text-[9px] text-gray-600 mb-0.5">P/C Ratio (Vol)</div>
                  <div className={`text-[12px] font-mono font-semibold ${
                    optionsData.pc_signal === 'bullish' ? 'text-emerald-400' :
                    optionsData.pc_signal === 'bearish' ? 'text-red-400' :
                    'text-yellow-400'
                  }`}>{optionsData.put_call_ratio.toFixed(2)}</div>
                </div>
                <div className="bg-gray-800/60 rounded p-1.5">
                  <div className="text-[9px] text-gray-600 mb-0.5">Max Pain</div>
                  <div className="text-[12px] font-mono font-semibold text-sky-300">
                    {optionsData.max_pain != null ? `$${optionsData.max_pain.toFixed(2)}` : '—'}
                  </div>
                </div>
              </div>
              {/* IV Skew */}
              <div className="bg-gray-800/60 rounded p-1.5 mb-2">
                <div className="flex justify-between items-center">
                  <span className="text-[9px] text-gray-600">IV Skew (Put − Call)</span>
                  <span className={`text-[10px] font-mono font-semibold ${
                    optionsData.iv_skew > 0.02 ? 'text-red-400' :
                    optionsData.iv_skew < -0.02 ? 'text-emerald-400' :
                    'text-yellow-400'
                  }`}>
                    {optionsData.iv_skew > 0 ? '+' : ''}{(optionsData.iv_skew * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="flex justify-between text-[9px] text-gray-600 mt-0.5">
                  <span>Call IV: {(optionsData.avg_call_iv * 100).toFixed(1)}%</span>
                  <span>Put IV: {(optionsData.avg_put_iv * 100).toFixed(1)}%</span>
                </div>
              </div>
              {/* Volume bars */}
              <div className="mb-2">
                <div className="text-[9px] text-gray-600 mb-1">Volume Distribution</div>
                <div className="flex items-center gap-1 text-[9px]">
                  <span className="text-emerald-400 w-6">C</span>
                  <div className="flex-1 h-1.5 bg-gray-800 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-emerald-500/60 rounded-full"
                      style={{ width: `${Math.min(100, (optionsData.total_call_volume / Math.max(optionsData.total_call_volume + optionsData.total_put_volume, 1)) * 100)}%` }}
                    />
                  </div>
                  <span className="text-gray-500 w-12 text-right font-mono">{optionsData.total_call_volume.toLocaleString()}</span>
                </div>
                <div className="flex items-center gap-1 text-[9px] mt-0.5">
                  <span className="text-red-400 w-6">P</span>
                  <div className="flex-1 h-1.5 bg-gray-800 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-red-500/60 rounded-full"
                      style={{ width: `${Math.min(100, (optionsData.total_put_volume / Math.max(optionsData.total_call_volume + optionsData.total_put_volume, 1)) * 100)}%` }}
                    />
                  </div>
                  <span className="text-gray-500 w-12 text-right font-mono">{optionsData.total_put_volume.toLocaleString()}</span>
                </div>
              </div>
              {/* Unusual Activity */}
              {optionsData.unusual_activity.length > 0 && (
                <div className="mb-2">
                  <div className="text-[9px] text-gray-600 mb-1">Unusual Activity</div>
                  <div className="space-y-0.5">
                    {optionsData.unusual_activity.slice(0, 5).map((u, i) => (
                      <div key={i} className="flex items-center gap-1 text-[9px]">
                        <span className={`w-6 font-semibold ${u.type === 'call' ? 'text-emerald-400' : 'text-red-400'}`}>
                          {u.type === 'call' ? 'C' : 'P'}
                        </span>
                        <span className="text-gray-400 font-mono w-12">${u.strike}</span>
                        <span className="text-gray-600 flex-1 truncate">{u.expiration}</span>
                        <span className="text-gray-400 font-mono">{u.vol_oi_ratio.toFixed(1)}x</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              {/* Key Strike Levels */}
              {optionsData.key_strikes.length > 0 && (
                <div>
                  <div className="text-[9px] text-gray-600 mb-1">Key Strikes (OI)</div>
                  <div className="flex flex-wrap gap-1">
                    {optionsData.key_strikes.slice(0, 10).map((s, i) => (
                      <span key={i} className="text-[9px] font-mono bg-gray-800 text-sky-300 px-1 py-0.5 rounded">
                        ${s}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </>}
          </div>
        )}

        {/* ── Astro Cycles Panel ───────────────────────────────────────────── */}
        {showAstroCycles && (() => {
          const astroPatterns = patterns?.filter(p => {
            const k = classifyPattern(p)
            return k === 'astro' && p.status !== 'NOT_PRESENT' && p.status !== 'FAILED'
          }) ?? []
          const bradleyTPs = astroPatterns.find(p => p.pattern_name === 'Bradley Siderograph')
          const retrogrades = astroPatterns.filter(p => p.pattern_name.includes('Retrograde'))
          const moonPhases  = astroPatterns.filter(p =>
            p.pattern_name.includes('New Moon') || p.pattern_name.includes('Full Moon') || p.pattern_name.includes('Quarter')
          )
          const ingresses   = astroPatterns.filter(p => p.pattern_name.includes('Ingress'))
          const aspects     = astroPatterns.filter(p => p.pattern_name.includes('Aspect'))
          const sq9         = astroPatterns.filter(p => p.pattern_name.includes('Square of Nine'))
          type AstroRec = Record<string, unknown>
          const bRec = bradleyTPs as unknown as AstroRec | undefined
          const upcomingTPs = bRec?.upcoming_turning_points as Array<{turn_date:string;turn_type:string;days_until:number}> | undefined
          if (astroPatterns.length === 0) return null
          return (
            <div className="bg-gray-900 rounded-lg border border-yellow-900/40 p-3 mt-3">
              <div className="flex items-center gap-1.5 mb-2">
                <span className="w-1.5 h-1.5 rounded-full bg-yellow-400" />
                <h3 className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider">Astro Cycles</h3>
                <span className="text-[9px] text-yellow-400/70 ml-1">⊙ Gann / Bradley</span>
                <span className="ml-auto">{collapseBtn('astro')}</span>
              </div>
              {!cc.astro && <>
                {/* Bradley Siderograph turning points */}
                {upcomingTPs && upcomingTPs.length > 0 && (
                  <div className="mb-2">
                    <div className="text-[9px] text-gray-600 mb-1">Bradley Turning Points</div>
                    <div className="space-y-0.5">
                      {upcomingTPs.slice(0, 4).map((tp, i) => (
                        <div key={i} className="flex items-center gap-1.5 text-[9px]">
                          <span className={`w-1 h-1 rounded-full flex-shrink-0 ${tp.turn_type === 'peak' ? 'bg-red-400' : 'bg-emerald-400'}`} />
                          <span className="text-gray-300 font-mono">{tp.turn_date}</span>
                          <span className={`ml-auto text-[8px] px-1 rounded ${tp.turn_type === 'peak' ? 'bg-red-900/40 text-red-300' : 'bg-emerald-900/40 text-emerald-300'}`}>
                            {tp.turn_type === 'peak' ? '▲ Peak' : '▼ Trough'}
                          </span>
                          <span className="text-gray-600 text-[8px]">{tp.days_until}d</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                {/* Mercury / Planet Retrogrades */}
                {retrogrades.length > 0 && (
                  <div className="mb-2">
                    <div className="text-[9px] text-gray-600 mb-1">Retrogrades</div>
                    <div className="space-y-0.5">
                      {retrogrades.slice(0, 4).map((r, i) => {
                        const rr = r as unknown as AstroRec
                        return (
                          <div key={i} className="flex items-center gap-1.5 text-[9px]">
                            <span className="text-yellow-400">☿</span>
                            <span className="text-gray-300 flex-1 truncate">{r.pattern_name}</span>
                            {!!rr.retro_ongoing && (
                              <span className="text-[8px] bg-yellow-900/40 text-yellow-300 px-1 rounded">ACTIVE</span>
                            )}
                            {!!rr.retro_start && (
                              <span className="text-gray-600 text-[8px] font-mono">{String(rr.retro_start).slice(0,10)}</span>
                            )}
                          </div>
                        )
                      })}
                    </div>
                  </div>
                )}
                {/* Moon Phases */}
                {moonPhases.length > 0 && (
                  <div className="mb-2">
                    <div className="text-[9px] text-gray-600 mb-1">Moon Phases</div>
                    <div className="flex flex-wrap gap-1">
                      {moonPhases.slice(0, 6).map((m, i) => {
                        const mm = m as unknown as AstroRec
                        const isNew  = m.pattern_name.includes('New Moon')
                        const isFull = m.pattern_name.includes('Full Moon')
                        return (
                          <div key={i} className="bg-gray-800/60 rounded px-1.5 py-0.5 text-[8px]">
                            <span className="mr-0.5">{isNew ? '🌑' : isFull ? '🌕' : '🌓'}</span>
                            <span className="text-gray-400 font-mono">{String(mm.phase_date ?? m.date ?? '').slice(0,10)}</span>
                          </div>
                        )
                      })}
                    </div>
                  </div>
                )}
                {/* Planetary Ingresses */}
                {ingresses.length > 0 && (
                  <div className="mb-2">
                    <div className="text-[9px] text-gray-600 mb-1">Planetary Ingress</div>
                    <div className="space-y-0.5">
                      {ingresses.slice(0, 3).map((ing, i) => {
                        const ir = ing as unknown as AstroRec
                        return (
                          <div key={i} className="flex items-center gap-1 text-[9px]">
                            <span className="text-yellow-400/70">♃</span>
                            <span className="text-gray-400 flex-1 truncate">{ing.pattern_name}</span>
                            {!!ir.ingress_date && (
                              <span className="text-gray-600 text-[8px] font-mono">{String(ir.ingress_date).slice(0,10)}</span>
                            )}
                          </div>
                        )
                      })}
                    </div>
                  </div>
                )}
                {/* Aspect clusters */}
                {aspects.length > 0 && (
                  <div className="mb-2">
                    <div className="text-[9px] text-gray-600 mb-1">Active Aspects</div>
                    <div className="space-y-0.5">
                      {aspects.slice(0, 3).map((asp, i) => {
                        const ar = asp as unknown as AstroRec
                        return (
                          <div key={i} className="flex items-center gap-1.5 text-[9px]">
                            <span className={`w-1 h-1 rounded-full flex-shrink-0 ${asp.direction === 'bullish' ? 'bg-emerald-400' : 'bg-red-400'}`} />
                            <span className="text-gray-400 flex-1 truncate">{asp.pattern_name}</span>
                            {!!ar.aspect_date && (
                              <span className="text-gray-600 text-[8px] font-mono">{String(ar.aspect_date).slice(0,10)}</span>
                            )}
                          </div>
                        )
                      })}
                    </div>
                  </div>
                )}
                {/* Square of Nine key levels */}
                {sq9.length > 0 && (() => {
                  const sq9Rec = sq9[0] as unknown as AstroRec
                  const levels = sq9Rec.sq9_planetary_levels as Array<{planet:string;harmonic:number;price:number}> | undefined
                  return levels?.length ? (
                    <div>
                      <div className="text-[9px] text-gray-600 mb-1">Square of Nine Levels</div>
                      <div className="flex flex-wrap gap-1">
                        {levels.slice(0, 8).map((lvl, i) => (
                          <span key={i} className="text-[8px] font-mono bg-gray-800 text-yellow-300/70 px-1 py-0.5 rounded" title={`${lvl.planet} ×${lvl.harmonic}`}>
                            ${lvl.price.toFixed(2)}
                          </span>
                        ))}
                      </div>
                    </div>
                  ) : null
                })()}
              </>}
            </div>
          )
        })()}

        {/* News panel */}
        {news && news.length > 0 && (
          <div className="bg-gray-900 rounded-lg border border-gray-800 p-3 mt-3">
            <div className="flex items-center justify-between mb-2">
              <p className="text-[10px] text-gray-500 uppercase font-semibold tracking-wider">Recent News</p>
              {collapseBtn('news')}
            </div>
            {!cc.news && <div className="space-y-2 max-h-64 overflow-y-auto">
              {news.map((item, i) => (
                <a key={i} href={item.url} target="_blank" rel="noopener noreferrer"
                   className="block hover:bg-gray-800 rounded p-1.5 transition-colors">
                  <p className="text-[11px] text-gray-200 leading-tight line-clamp-2">{item.headline}</p>
                  <div className="flex items-center gap-2 mt-1">
                    <span className="text-[9px] text-gray-600">{item.source}</span>
                    <span className="text-[9px] text-gray-700">{new Date(item.datetime * 1000).toLocaleDateString()}</span>
                  </div>
                </a>
              ))}
            </div>}
          </div>
        )}
      </div>}
    </div>
  )
}
