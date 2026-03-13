import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import api from './client'
import type { Instrument, ChartData, Signal, CycleEntry, FeatureRow, Categories, BullBearGroup, PatternEntry, PatternCatalogueItem, PatternScanEntry, LiveQuote, NewsItem, MTFResult, Recommendation, BacktestStats, BacktestSimilar, BacktestTicker, BacktestDrill, TrendlineData, CommandCenterData, RegimeTimelineData, PriceLevels, PatternMTF, Fundamentals, CycleAdvanced, MarketForceData, ConfluenceNodeData, ConfluenceHeatmapData, StrategiesData, SignalConfidenceData, StrategyHealthData, PriceDistributionData } from '../types'

// ── Watchlist ─────────────────────────────────────────────────────────────────

export interface WatchlistItem { symbol: string; added_at: string }

export function useWatchlist() {
  return useQuery<WatchlistItem[]>({
    queryKey: ['watchlist'],
    queryFn:  () => api.get('/watchlist').then(r => r.data),
    staleTime: 30_000,
  })
}

export function useWatchlistToggle() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ symbol, watching }: { symbol: string; watching: boolean }) =>
      watching
        ? api.delete(`/watchlist/${symbol}`).then(r => r.data)
        : api.post(`/watchlist/${symbol}`).then(r => r.data),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['watchlist'] }),
  })
}

export function useInstruments() {
  return useQuery<Instrument[]>({
    queryKey: ['instruments'],
    queryFn: () => api.get('/instruments').then(r => r.data),
    staleTime: 5 * 60_000,
  })
}

export function useCategories() {
  return useQuery<Categories>({
    queryKey: ['categories'],
    queryFn: () => api.get('/instruments/categories').then(r => r.data),
    staleTime: 10 * 60_000,
  })
}

export function useChartData(symbol: string, limit = 0, interval = 'daily') {
  return useQuery<ChartData>({
    queryKey: ['chart', symbol, limit, interval],
    queryFn: () => api.get(`/chart/${symbol}`, { params: { limit, interval } }).then(r => r.data),
    enabled: !!symbol,
    staleTime: 60_000,
  })
}

export interface IntradayBar { time: number; open: number; high: number; low: number; close: number; volume: number }

export function useIntradayCandles(symbol: string, resolution: number, enabled: boolean) {
  // yfinance hard caps: 1m→7d, 2-30m→60d, 60m+(1h/2h/4h)→730d — send maximum
  const daysBack = resolution <= 1 ? 7 : resolution <= 30 ? 60 : 730
  return useQuery<IntradayBar[]>({
    queryKey: ['candles', symbol, resolution],
    queryFn: () => api.get(`/candles/${symbol}`, { params: { resolution, days_back: daysBack } }).then(r => r.data),
    enabled: enabled && !!symbol,
    staleTime: 60_000,
    refetchInterval: 60_000,
  })
}

export function useSignals(minProbability = 0.0, tier?: string) {
  return useQuery<Signal[]>({
    queryKey: ['signals', minProbability, tier],
    queryFn: () => api.get('/signals/latest', {
      params: { min_probability: minProbability, ...(tier ? { tier } : {}) }
    }).then(r => r.data),
    staleTime: 60_000,
  })
}

export function useCycleHeatmap() {
  return useQuery<CycleEntry[]>({
    queryKey: ['cycles', 'heatmap'],
    queryFn: () => api.get('/cycles/heatmap/latest').then(r => r.data),
    staleTime: 5 * 60_000,
  })
}

export function useFeatureMatrix() {
  return useQuery<FeatureRow[]>({
    queryKey: ['features', 'matrix'],
    queryFn: () => api.get('/features/matrix/latest').then(r => r.data),
    staleTime: 5 * 60_000,
  })
}

export function usePatterns(symbol: string, tf?: string, limit = 500) {
  return useQuery<PatternEntry[]>({
    queryKey: ['patterns', symbol, tf ?? 'daily', limit],
    queryFn: () => api.get(`/patterns/${symbol}`, {
      params: { limit, ...(tf ? { tf } : {}) },
    }).then(r => r.data),
    enabled: !!symbol,
    staleTime: 60_000,
  })
}

export function usePatternCatalogue() {
  return useQuery<PatternCatalogueItem[]>({
    queryKey: ['patterns', 'catalogue'],
    queryFn: () => api.get('/patterns/catalogue').then(r => r.data),
    staleTime: 60 * 60_000,
  })
}

export function usePatternScan(enabled = true, tf?: string) {
  return useQuery<PatternScanEntry[]>({
    queryKey: ['patterns', 'scan', tf ?? 'daily'],
    queryFn: () => api.get('/patterns/scan', { params: tf ? { tf } : {} }).then(r => r.data),
    enabled,
    staleTime: 5 * 60_000,   // matches backend 5-min TTL cache
    gcTime:    10 * 60_000,
  })
}

export interface ConfluenceResult {
  symbol: string
  date: string
  confluence_score: number
  signal_tier: 'HIGH' | 'MEDIUM' | 'WATCH' | 'NONE'
  volatility_regime: string
  is_compression: boolean
  is_breakout: boolean
  expected_move_pct: number
  expected_move_display: string
  passes_move_filter: boolean
  entry_price: number
  add_price: number
  scale_price: number
  target_price: number
  stop_price: number
  component_scores: {
    vol_compression: number
    breakout: number
    trend: number
    liquidity: number
    pattern: number
    wyckoff: number
    gann: number
  }
  decision_trace: string
  error?: string
  gated?: boolean
}

export function useConfluenceScore(symbol: string) {
  return useQuery<ConfluenceResult>({
    queryKey: ['confluence', 'score', symbol],
    queryFn: () => api.get(`/confluence/score/${symbol}`).then(r => r.data),
    enabled: !!symbol,
    staleTime: 2 * 60_000,
    retry: false,
  })
}

export function useLiveQuote(symbol: string) {
  return useQuery<LiveQuote>({
    queryKey: ['quote', symbol],
    queryFn: () => api.get(`/quote/${symbol}`).then(r => r.data),
    enabled: !!symbol,
    staleTime: 30_000,       // refresh every 30s
    refetchInterval: 30_000,
  })
}

export function useNews(symbol: string, days = 7) {
  return useQuery<NewsItem[]>({
    queryKey: ['news', symbol, days],
    queryFn: () => api.get(`/news/${symbol}`, { params: { days } }).then(r => r.data),
    enabled: !!symbol,
    staleTime: 5 * 60_000,
  })
}

export function useMTF(symbol: string) {
  return useQuery<MTFResult>({
    queryKey: ['mtf', symbol],
    queryFn: () => api.get(`/mtf/${symbol}`).then(r => r.data),
    enabled: !!symbol,
    staleTime: 60_000,
  })
}

export function useRecommendation(symbol: string) {
  return useQuery<Recommendation>({
    queryKey: ['recommendation', symbol],
    queryFn: () => api.get(`/recommendation/${symbol}`).then(r => r.data),
    enabled: !!symbol,
    staleTime: 60_000,
  })
}

export function useTrendlines(symbol: string) {
  return useQuery<TrendlineData>({
    queryKey: ['trendlines', symbol],
    queryFn: () => api.get(`/trendlines/${symbol}`).then(r => r.data),
    enabled: !!symbol,
    staleTime: 5 * 60_000,
  })
}

export function useBacktestStats() {
  return useQuery<BacktestStats>({
    queryKey: ['backtest', 'stats'],
    queryFn: () => api.get('/backtest/stats').then(r => r.data),
    staleTime: 60 * 60_000,  // 1 hour (matches server cache)
  })
}

export function useBacktestSimilar(symbol: string) {
  return useQuery<BacktestSimilar>({
    queryKey: ['backtest', 'similar', symbol],
    queryFn: () => api.get(`/backtest/similar/${symbol}`).then(r => r.data),
    enabled: !!symbol,
    staleTime: 60 * 60_000,
  })
}

export function useBacktestDrill(bucket: string, phase: string = '', enabled = false) {
  return useQuery<BacktestDrill>({
    queryKey: ['backtest', 'drill', bucket, phase],
    queryFn: () => api.get('/backtest/drill', { params: { bucket, ...(phase ? { phase } : {}) } }).then(r => r.data),
    enabled: enabled && !!bucket,
    staleTime: 60 * 60_000,
  })
}

export function useBacktestTicker(symbol: string) {
  return useQuery<BacktestTicker>({
    queryKey: ['backtest', 'ticker', symbol],
    queryFn: () => api.get(`/backtest/ticker/${symbol}`).then(r => r.data),
    enabled: !!symbol,
    staleTime: 60 * 60_000,
  })
}

export function useBullBearAnalysis() {
  return useQuery<BullBearGroup[]>({
    queryKey: ['analysis', 'bull-bear'],
    queryFn: () => api.get('/analysis/bull-bear').then(r => r.data),
    staleTime: 5 * 60_000,
  })
}

export function useCommandCenter() {
  return useQuery<CommandCenterData>({
    queryKey: ['command-center'],
    queryFn: () => api.get('/command-center').then(r => r.data),
    staleTime: 30_000,
    refetchInterval: 60_000,
  })
}

// ── Pipeline ──────────────────────────────────────────────────────────────────

export interface PipelineEngineStatus {
  status:    'idle' | 'running' | 'ok' | 'error'
  started_at: string | null
  finished_at: string | null
  error:     string | null
}

export interface PipelineState {
  is_running:    boolean
  started_at:    string | null
  finished_at:   string | null
  current_step:  string | null
  engines:       Record<string, PipelineEngineStatus>
}

export function usePipelineStatus() {
  return useQuery<PipelineState>({
    queryKey: ['pipeline', 'status'],
    queryFn:  () => api.get('/pipeline/status').then(r => r.data),
    staleTime: 10_000,
    refetchInterval: 15_000,
  })
}

export function usePipelineRun() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: () => api.post('/pipeline/run').then(r => r.data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['pipeline'] })
    },
  })
}

// ── Trade Coach ───────────────────────────────────────────────────────────────

export interface TradeCoachScenario {
  label:       string
  probability: number
  price_target: number | null
  description: string
  color:       string
}

export interface TradeCoachExplanation {
  symbol:        string
  quality_score: number | null
  setup_type:    string | null
  evidence:      string[]
  risk_factors:  string[]
  scenario_map:  TradeCoachScenario[]
  entry_price:   number | null
  stop_price:    number | null
  generated_at:  string
  error?:        string
}

export interface SimilarSetup {
  symbol:       string
  setup_date:   string | null
  setup_type:   string | null
  outcome:      string | null
  return_pct:   number | null
  holding_days: number | null
}

export interface SimilarSetupsData {
  symbol:          string
  setup_type:      string | null
  sample_size:     number
  win_rate:        number | null
  avg_return:      number | null
  max_drawdown:    number | null
  profit_factor:   number | null
  similar_setups:  SimilarSetup[]
  error?:          string
}

export interface TradeReview {
  trade_id:          number
  symbol:            string
  entry_quality:     number
  exit_quality:      number
  verdict:           string
  entry_notes:       string[]
  exit_notes:        string[]
  missed_profit_pct: number | null
  error?:            string
}

export function useTradeCoach(symbol: string) {
  return useQuery<TradeCoachExplanation>({
    queryKey: ['trade-coach', 'signal', symbol],
    queryFn:  () => api.get(`/trade-coach/signal/${symbol}`).then(r => r.data),
    enabled:  !!symbol,
    staleTime: 60_000,
  })
}

export function useSimilarSetups(symbol: string) {
  return useQuery<SimilarSetupsData>({
    queryKey: ['trade-coach', 'similar', symbol],
    queryFn:  () => api.get(`/trade-coach/similar/${symbol}`).then(r => r.data),
    enabled:  !!symbol,
    staleTime: 5 * 60_000,
  })
}

export function useTradeReview(tradeId: number | null) {
  return useQuery<TradeReview>({
    queryKey: ['trade-coach', 'review', tradeId],
    queryFn:  () => api.get(`/trade-coach/review/${tradeId}`).then(r => r.data),
    enabled:  tradeId != null,
    staleTime: 5 * 60_000,
  })
}

// ── Strategy Evolution ────────────────────────────────────────────────────────

export interface EvolutionStrategyStats {
  setup_type:    string
  fitness_score: number
  status:        'strong' | 'acceptable' | 'weak' | 'disabled'
  trade_count:   number
  return_count:  number
  win_rate:      number | null
  avg_return:    number | null
  profit_factor: number | null
  max_drawdown:  number | null
  sharpe_ratio:  number | null
}

export interface EvolutionSuggestion {
  setup_type:    string
  fitness_score: number
  status:        string
  suggestions:   string[]
}

export interface EvolutionReport {
  system_fitness:        number | null
  total_strategies:      number
  underperforming_count: number
  strategies:            EvolutionStrategyStats[]
  suggestions:           EvolutionSuggestion[]
  safety_note:           string
  generated_at:          string
}

export function useEvolutionReport() {
  return useQuery<EvolutionReport>({
    queryKey: ['evolution', 'report'],
    queryFn:  () => api.get('/evolution/report').then(r => r.data),
    staleTime: 5 * 60_000,
  })
}

export function useStrategyFitness(setupType: string) {
  return useQuery({
    queryKey: ['evolution', 'strategy', setupType],
    queryFn:  () => api.get(`/evolution/strategy/${setupType}`).then(r => r.data),
    enabled:  !!setupType,
    staleTime: 5 * 60_000,
  })
}

// ── Regime Timeline ───────────────────────────────────────────────────────────

export function useRegimeTimeline(
  symbol: string,
  startDate?: string,
  endDate?: string,
) {
  return useQuery<RegimeTimelineData>({
    queryKey: ['regime-timeline', symbol, startDate, endDate],
    queryFn: () =>
      api
        .get('/analytics/regime-timeline', {
          params: {
            symbol,
            ...(startDate ? { start_date: startDate } : {}),
            ...(endDate   ? { end_date:   endDate   } : {}),
          },
        })
        .then(r => r.data),
    enabled:  !!symbol,
    staleTime: 5 * 60_000,  // 5 minutes (matches backend cache)
  })
}

// ── Price Levels (Stage 5) ────────────────────────────────────────────────────

export function usePriceLevels(symbol: string) {
  return useQuery<PriceLevels>({
    queryKey: ['price-levels', symbol],
    queryFn:  () => api.get(`/price-levels/${symbol}`).then(r => r.data),
    enabled:  !!symbol,
    staleTime: 2 * 60_000,
    retry:    false,
  })
}

export function usePatternMTF(symbol: string) {
  return useQuery<PatternMTF>({
    queryKey: ['patterns', 'mtf', symbol],
    queryFn:  () => api.get(`/patterns/${symbol}/mtf`).then(r => r.data),
    enabled:  !!symbol,
    staleTime: 5 * 60_000,
    retry:    false,
  })
}

// ── Market Breadth (McClellan, VIX, Fear/Greed) ──────────────────────────────

export interface MarketBreadthData {
  // VIX
  vix:                  number | null
  vix_change:           number | null
  vix_direction:        'extreme_fear' | 'fear' | 'normal' | 'complacency' | 'unknown'
  // McClellan Oscillator
  mco:                  number | null
  mco_sum:              number | null
  mco_direction:        'bullish' | 'bearish' | 'unknown'
  mco_series:           number[]
  sum_series:           number[]
  mco_dates:            string[]
  adv_latest:           number | null
  dec_latest:           number | null
  // SPY momentum
  spy_momentum:         number | null
  spy_above_ma:         boolean | null
  // Fear/Greed composite
  fear_greed:           number | null
  fear_greed_label:     string
  fear_greed_direction: 'bullish' | 'bearish' | 'neutral'
  error?:               string
}

export function useMarketBreadth() {
  return useQuery<MarketBreadthData>({
    queryKey: ['market', 'breadth'],
    queryFn:  () => api.get('/market/breadth').then(r => r.data),
    staleTime: 10 * 60_000,   // 10 minutes (backend cache is 15 min)
    refetchInterval: 15 * 60_000,
    retry: false,
  })
}

// ── FMP Fundamentals ──────────────────────────────────────────────────────────

export function useFundamentals(symbol: string) {
  return useQuery<Fundamentals>({
    queryKey: ['fundamentals', symbol],
    queryFn:  () => api.get(`/fmp/${symbol}/fundamentals`).then(r => r.data),
    enabled:  !!symbol,
    staleTime: 15 * 60_000,   // 15 minutes — ratios/profile rarely change intraday
    retry: false,
  })
}

// ── Advanced Cycle Analysis ──────────────────────────────────────────────────

export function useCycleAdvanced(symbol: string) {
  return useQuery<CycleAdvanced>({
    queryKey: ['cycles', 'advanced', symbol],
    queryFn:  () => api.get(`/cycles/${symbol}/advanced`).then(r => r.data),
    enabled:  !!symbol,
    staleTime: 5 * 60_000,    // 5 min cache matches backend TTL
    retry: false,
  })
}

// ── Market Force Vectors ─────────────────────────────────────────────────────

export function useMarketForce(symbol: string) {
  return useQuery<MarketForceData>({
    queryKey: ['force', symbol],
    queryFn:  () => api.get(`/force/${symbol}`).then(r => r.data),
    enabled:  !!symbol,
    staleTime: 5 * 60_000,
    retry: false,
  })
}

// ── Confluence Nodes & Heatmap ───────────────────────────────────────────────

export function useConfluenceNodes(symbol: string) {
  return useQuery<ConfluenceNodeData>({
    queryKey: ['confluence-nodes', symbol],
    queryFn:  () => api.get(`/confluence-nodes/nodes/${symbol}`).then(r => r.data),
    enabled:  !!symbol,
    staleTime: 5 * 60_000,
    retry: false,
  })
}

export function useConfluenceHeatmap(symbol: string, bins = 30) {
  return useQuery<ConfluenceHeatmapData>({
    queryKey: ['confluence-heatmap', symbol, bins],
    queryFn:  () => api.get(`/confluence-nodes/heatmap/${symbol}?bins=${bins}`).then(r => r.data),
    enabled:  !!symbol,
    staleTime: 5 * 60_000,
    retry: false,
  })
}

// ── Strategy Evolution ───────────────────────────────────────────────────────

export function useTopStrategies(limit = 10) {
  return useQuery<StrategiesData>({
    queryKey: ['strategies', 'top', limit],
    queryFn:  () => api.get(`/strategies/top?limit=${limit}`).then(r => r.data),
    staleTime: 10 * 60_000,
    retry: false,
  })
}

// ── Signal Confidence ──────────────────────────────────────────────────────

export function useSignalConfidence(symbol: string) {
  return useQuery<SignalConfidenceData>({
    queryKey: ['signal-confidence', symbol],
    queryFn:  () => api.get(`/signal-confidence/${symbol}`).then(r => r.data),
    enabled:  !!symbol,
    staleTime: 5 * 60_000,
    retry: false,
  })
}

// ── Strategy Health (Model Monitor) ────────────────────────────────────────

export function useStrategyHealth() {
  return useQuery<StrategyHealthData>({
    queryKey: ['strategy-health'],
    queryFn:  () => api.get('/strategy-health').then(r => r.data),
    staleTime: 5 * 60_000,
    retry: false,
  })
}

// ── Price Distribution Forecast ────────────────────────────────────────────

export function usePriceDistribution(symbol: string, horizon = 20) {
  return useQuery<PriceDistributionData>({
    queryKey: ['distribution', symbol, horizon],
    queryFn:  () => api.get(`/distribution/${symbol}`, { params: { horizon } }).then(r => r.data),
    enabled:  !!symbol,
    staleTime: 5 * 60_000,
    retry: false,
  })
}

// ── Options Flow Analysis ──────────────────────────────────────────────────

export interface OptionsUnusualActivity {
  type: 'call' | 'put'
  strike: number
  expiration: string
  volume: number
  open_interest: number
  vol_oi_ratio: number
  implied_volatility: number
  delta: number
}

export interface OptionsData {
  symbol: string
  put_call_ratio: number
  put_call_ratio_oi: number
  pc_signal: 'bullish' | 'bearish' | 'neutral'
  total_call_volume: number
  total_put_volume: number
  total_call_oi: number
  total_put_oi: number
  max_pain: number | null
  nearest_expiration: string | null
  iv_skew: number
  avg_call_iv: number
  avg_put_iv: number
  key_strikes: number[]
  unusual_activity: OptionsUnusualActivity[]
  top_calls: OptionsUnusualActivity[]
  top_puts: OptionsUnusualActivity[]
  expirations: string[]
  total_contracts: number
  error?: string
}

export function useOptionsData(symbol: string) {
  return useQuery<OptionsData>({
    queryKey: ['options', symbol],
    queryFn: () => api.get(`/options/${symbol}`).then(r => r.data),
    enabled: !!symbol,
    staleTime: 5 * 60 * 1000, // 5 min
    retry: false,
  })
}

