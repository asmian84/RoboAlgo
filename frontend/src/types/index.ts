export interface Instrument {
  symbol: string
  name: string | null
  instrument_type: string | null
  leverage_factor: number | null
  underlying: string | null
  pair_symbol: string | null
}

export interface PriceBar {
  date: string
  open: number | null
  high: number | null
  low: number | null
  close: number | null
  volume: number | null
}

export interface IndicatorRow {
  date: string
  rsi: number | null
  atr: number | null
  macd_line: number | null
  macd_signal: number | null
  macd_histogram: number | null
  bb_upper: number | null
  bb_middle: number | null
  bb_lower: number | null
  bb_width: number | null
  ma50: number | null
  ma200: number | null
}

export interface ChartData {
  prices: PriceBar[]
  indicators: IndicatorRow[]
}

export interface FeatureRow {
  date: string
  symbol?: string
  trend_strength: number | null
  momentum: number | null
  volatility_percentile: number | null
  volume_ratio: number | null
  cycle_phase: number | null
  macd_norm: number | null
  bb_position: number | null
  price_to_ma50: number | null
  return_5d: number | null
  return_20d: number | null
}

export interface CycleEntry {
  symbol: string
  date: string
  cycle_length: number | null
  cycle_phase: number | null
  cycle_strength: number | null
}

export interface Signal {
  symbol: string
  date: string
  probability: number
  confidence_tier: 'HIGH' | 'MEDIUM' | 'LOW'
  market_phase: string
  buy_price: number
  accumulate_price: number
  scale_price: number
  sell_price: number
}

export interface BullBearInstrument {
  symbol: string | null
  score: number | null
  phase: string
  features: FeatureRow | null
}

export interface BullBearGroup {
  description: string
  underlying: BullBearInstrument | null
  bull: BullBearInstrument
  bear: BullBearInstrument
  verdict: string
  verdict_color: string
  reasoning: string
}

export interface PatternEntry {
  date: string
  pattern_name: string
  pattern_category?: 'chart' | 'harmonic' | 'gann' | 'wyckoff' | 'candlestick' | 'behavioral' | 'indicator' | 'volume' | 'measured_move' | 'strategy' | 'market_analysis' | string
  pattern_type: 'candlestick' | 'chart' | 'behavioral' | 'indicator' | 'volume' | 'measured_move' | 'strategy' | 'market_analysis' | 'harmonic' | 'gann' | 'wyckoff'
  direction: 'bullish' | 'bearish' | 'neutral'
  strength: number
  price_level: number | null
  status?: 'NOT_PRESENT' | 'FORMING' | 'READY' | 'BREAKOUT' | 'FAILED' | 'COMPLETED'
  breakout_level?: number | null
  invalidation_level?: number | null
  projected_target?: number | null
  target?: number | null
  confidence?: number
  probability?: number
  points?: Array<[string | number, number]>
  point_labels?: string[]
  overlay_lines?: Array<[[string | number, number], [string | number, number]]>
  /** Per-segment styling roles — parallel array to overlay_lines */
  overlay_line_roles?: string[]
  message?: string | null
  // Wyckoff enriched fields
  phase?: string | null
  phase_label?: string | null
  events?: string[]
  support_level?: number | null
  resistance_level?: number | null
  /** Wyckoff/pattern event coordinates for chart annotation markers */
  event_points?: Array<{ label: string; date: string; price: number }>
  // Gann enriched fields
  fan_lines?: Array<{ angle: string; multiplier: number; current_price: number }>
  retracement_levels?: Array<{ fraction: number; label: string; price: number }>
  time_cycles?: Array<{ cycle_bars: number; anchor_index: number; projected_index: number; status: string }>
  // Cup & Handle extra
  cup_left_price?: number
  cup_right_price?: number
  cup_bottom_price?: number
  // Measured move / multi-pivot patterns
  ratios?: Record<string, number> | null
  prz_low?: number | null
  prz_high?: number | null
  fill_zone?: Array<[[string|number, number], [string|number, number]]> | null
  // Strategy-specific
  squeeze_bars?: number | null
  nr7_range?: number | null
}

export interface PatternCatalogueItem {
  pattern_name: string
  type: 'candlestick' | 'chart' | 'behavioral' | 'indicator' | 'volume' | 'measured_move' | 'strategy' | 'market_analysis' | 'harmonic' | 'gann' | 'wyckoff'
  direction: 'bullish' | 'bearish' | 'neutral'
  description: string
}

export interface LiveQuote {
  symbol: string
  price: number | null
  open: number | null
  high: number | null
  low: number | null
  prev_close: number | null
  change: number | null
  change_pct: number | null
  timestamp: number | null
  pre_market_price?: number | null
  pre_market_change_pct?: number | null
  post_market_price?: number | null
  post_market_change_pct?: number | null
}

export interface NewsItem {
  headline: string
  source: string
  url: string
  datetime: number
  summary: string
}

export interface MTFEntry {
  timeframe: string
  confidence: number | null
  signal: 'bullish' | 'bearish' | 'neutral' | 'no_data'
  details: {
    rsi: number | null
    macd_hist: number | null
    bb_position: number | null
    above_ma: number | null
    momentum: number | null
  }
}

export interface MTFResult {
  symbol: string
  underlying: string
  timeframes: MTFEntry[]
}

export interface TradePlan {
  date: string
  confidence_tier: 'HIGH' | 'MEDIUM' | 'LOW'
  market_phase: string
  buy_price: number | null
  accumulate_price: number | null
  scale_price: number | null
  sell_price: number | null
}

export interface HorizonPlan {
  buy_price: number
  accumulate_price: number
  scale_price: number
  sell_price: number
}

export interface Recommendation {
  symbol: string
  underlying: string
  recommendation: string
  recommendation_color: string
  overall_score: number
  conviction: 'HIGH' | 'MEDIUM' | 'LOW'
  conviction_color: string
  conviction_score: number
  components: {
    mtf_weighted_avg: number
    xgb_probability: number
    tf_alignment_pct: number
    pattern_score: number
    phase_quality: number
    news_sentiment_score: number | null
  }
  mtf_timeframes: MTFEntry[]
  pattern_context: {
    bullish: number
    bearish: number
    total: number
    recent: Array<{ date: string; name: string; direction: string; strength: number }>
  }
  behavioral: BehavioralSignal
  trade_plan: TradePlan | null
  top_down: {
    higher_bias: 'bullish' | 'bearish' | 'neutral'
    higher_score: number
    higher_aligned: boolean
    pullback_score: number
    signal: string
    signal_color: string
  }
  horizon_plans: { short: HorizonPlan | null; medium: HorizonPlan | null; long: HorizonPlan | null }
  news_sentiment: {
    score: number | null
    article_count: number
    label: string
    color: string
  }
  earnings_risk: {
    has_risk: boolean
    earnings_date: string | null
    days_until: number | null
  }
  index_correlation: {
    correlation: number | null
    benchmark: string | null
    label: string
    color: string
    description?: string
  }
}

export interface TrendlinePoint { date: string; value: number }
export interface TrendlineSegment {
  type: 'resistance' | 'support'
  points: TrendlinePoint[]
  projected?: TrendlinePoint
}
export interface TrendlineWindow {
  window: number
  lookback_bars: number
  swing_highs: TrendlinePoint[]
  swing_lows: TrendlinePoint[]
  resistance: TrendlineSegment[]
  support: TrendlineSegment[]
}
export interface TrendlineData {
  symbol: string
  short: TrendlineWindow
  medium: TrendlineWindow
  long: TrendlineWindow
}

export interface BehavioralSignal {
  signal: 'FEAR_CAPITULATION' | 'MILD_FEAR' | 'NEUTRAL' | 'MILD_GREED' | 'GREED_FOMO'
  label: string
  description: string
  strength: number
  action: string
  color: string
}

export interface BacktestBucket {
  total: number
  closed: number
  wins_t1: number
  wins_target: number
  stops: number
  open: number
  win_rate: number
  target_rate: number
  stop_rate: number
  avg_t1_return: number | null
  avg_tgt_return: number | null
  avg_stop_return: number | null
}

export interface BacktestStats {
  forward_days: number
  overall: BacktestBucket
  by_probability: Record<string, BacktestBucket>
  by_phase: Record<string, BacktestBucket>
  by_tier: Record<string, BacktestBucket>
}

export interface BacktestSignalRow {
  date: string
  probability: number
  confidence_tier: string
  market_phase: string
  buy_price: number
  accumulate_price: number | null
  scale_price: number | null
  sell_price: number | null
  outcome: 'open' | 'stop' | 't1' | 'target'
  t1_return: number | null
  tgt_return: number | null
  stop_return: number | null
}

export interface BacktestDrillRow {
  symbol: string
  date: string
  probability: number
  confidence_tier: string
  market_phase: string
  buy_price: number
  outcome: 'open' | 'stop' | 't1' | 'target'
  t1_return: number | null
  tgt_return: number | null
  stop_return: number | null
}

export interface BacktestDrill {
  bucket: string
  phase: string
  signals: BacktestDrillRow[]
}

export interface BacktestTicker {
  symbol: string
  forward_days: number
  stats: BacktestBucket | null
  signals: BacktestSignalRow[]
}

export interface BacktestSimilar {
  symbol: string
  found: boolean
  phase?: string
  probability?: number
  prob_bucket?: string
  phase_stats?: BacktestBucket
  prob_stats?: BacktestBucket
}

export interface Categories {
  index_drivers: string[]
  index_leveraged: string[]
  sector_leveraged: string[]
  commodity_leveraged: string[]
  single_stock_leveraged: string[]
  underlying_leaders: string[]
}

// ── Regime Timeline ─────────────────────────────────────────────────────────────

export interface RegimeTimelineBar {
  date:           string
  market_state:   string
  trend_strength: number | null
  volatility_pct: number | null
  close_price:    number | null
  trade_event:    'ENTRY' | 'EXIT' | 'SETUP' | 'TRIGGER' | null
  trade_id:       number | null
  trade_pnl:      number | null
  cumulative_pnl: number
  daily_pnl:      number
}

export interface RegimeStatePeriod {
  state:      string
  start_date: string
  end_date:   string
  days:       number
}

export interface RegimeStateStat {
  trades:    number
  wins:      number
  total_pnl: number
  win_rate:  number
  avg_pnl:   number
}

export interface RegimeTradeEntry {
  id:           number
  symbol:       string
  state:        string
  setup_type:   string | null
  market_state: string | null
  entry_price:  number | null
  exit_price:   number | null
  pnl:          number | null
  return_pct:   number | null
  entry_date:   string | null
  exit_date:    string | null
  holding_days: number | null
  exit_reason:  string | null
}

export interface RegimeTimelineData {
  symbol:        string
  start_date:    string
  end_date:      string
  timeline:      RegimeTimelineBar[]
  state_periods: RegimeStatePeriod[]
  regime_stats:  Record<string, RegimeStateStat>
  trades:        RegimeTradeEntry[]
  generated_at:  string
  error?:        string
}

// ── Opportunity Radar ────────────────────────────────────────────────────────────

export interface OpportunityRadarEntry {
  symbol:            string
  date:              string
  opportunity_score: number
  compression_score: number
  shelf_score:       number
  proximity_score:   number
  market_state:      string
  is_early_stage:    boolean
  computed_at:       string
}

// ── Command Center ──────────────────────────────────────────────────────────────

export interface CommandCenterInstrument {
  symbol: string
  state: string
  volatility_percentile: number | null
  trend_strength: number | null
  expansion_strength: number | null
  ma_alignment: string | null
  adx: number | null
  volume_ratio: number | null
  size_multiplier: number | null
}

export interface LiquidityLevel {
  price_level: number
  liquidity_score: number
  distance_pct: number
  side: 'above' | 'below'
  touch_count: number
}

export interface CommandCenterSignal {
  symbol: string
  confluence_score: number
  signal_tier: string
  setup_quality_score:    number | null
  quality_grade:          string | null
  breakout_quality_score: number | null
  breakout_gate_passed:   boolean | null
  liquidity_shelf_score:  number | null
  liquidity_sweep_score:  number | null
  sweep_type:             string | null   // "high_sweep" | "low_sweep" | "none"
  sweep_gate_passed:      boolean | null
  liquidity_alignment:    number | null   // 0-100, directional alignment score
  nearest_above:          LiquidityLevel | null
  nearest_below:          LiquidityLevel | null
  expected_move_pct: number | null
  expected_move_display: string
  is_compression: boolean
  is_breakout: boolean
  entry_price: number | null
  stop_price: number | null
  target_price: number | null
  market_state: string
  volatility_regime: string
  component_scores: Record<string, number>
  decision_trace: string | null
  position_size_multiplier: number | null
  risk_per_trade:           number | null
  size_approved:            boolean | null
}

export interface CommandCenterTrade {
  id: number
  symbol: string
  state: string
  setup_type: string | null
  market_state: string | null
  entry_price: number | null
  stop_price: number | null
  tier1_sell: number | null
  tier2_sell: number | null
  tier3_hold: number | null
  position_size: number | null
  setup_at: string | null
  trigger_at: string | null
  entry_at: string | null
  notes: string | null
}

export interface PlaybookEntry {
  regime:              string
  strategy_type:       string
  strategy_key:        string
  position_multiplier: number
  risk_per_trade:      number
  max_positions:       number
  risk_mode:           string
  confluence_min:      number
  quality_score_min:   number
  volume_requirement:  number
  allowed_setups:      string[]
  entry_description:   string
  stop_description:    string
  size_description:    string
  color:               string
}

export interface ActiveStrategyData {
  dominant_regime:     string
  regime_counts:       Record<string, number>
  strategy_type:       string
  strategy_key:        string
  position_multiplier: number
  risk_per_trade:      number
  max_positions:       number
  risk_mode:           string
  color:               string
  entry_description:   string
  size_description:    string
  playbook:            PlaybookEntry[]
  error?:              string
}

// ── Signal Reliability ───────────────────────────────────────────────────────

export type ReliabilityStatus = 'healthy' | 'warning' | 'disabled' | 'no_data'

export interface SignalReliabilityEntry {
  setup_type:          string
  strategy_label:      string
  reliability_score:   number | null
  status:              ReliabilityStatus
  position_multiplier: number          // 1.0 | 0.5 | 0.0
  win_rate:            number | null
  expectancy:          number | null
  max_drawdown:        number | null
  trade_count:         number
  proxy?:              boolean         // true = estimated from pattern signals, not real trades
}

export interface SignalReliabilityData {
  strategies:         SignalReliabilityEntry[]
  system_reliability: number | null     // lowest individual score (weakest link)
  disabled_count:     number
  warning_count:      number
  error?:             string
}

export interface HedgeSuggestion {
  symbol:      string
  label:       string
  category:    'volatility' | 'inverse_broad' | 'inverse_sector' | 'safe_haven'
  description: string
}

export interface MarketSafetyData {
  safety_score:      number
  safety_state:      'NORMAL' | 'CAUTION' | 'SAFE_MODE'
  trading_allowed:   boolean
  size_multiplier:   number
  system_action:     string
  components: {
    volatility:   number
    gap:          number
    portfolio:    number
    data_quality: number
  }
  triggers:          string[]
  suggested_hedges?: HedgeSuggestion[]
  computed_at:       string
  error?:            string
}

export interface CommandCenterData {
  active_strategy: ActiveStrategyData
  market_state_summary: {
    counts: Record<string, number>
    instruments: CommandCenterInstrument[]
    error?: string
  }
  opportunity_map: {
    signals: CommandCenterSignal[]
    error?: string
  }
  active_trades: {
    trades: CommandCenterTrade[]
    count: number
    error?: string
  }
  portfolio_risk: {
    account_equity: number
    open_positions: number
    max_positions: number
    daily_pnl_pct: number
    daily_loss_limit: number
    sector_exposure: Record<string, number>
    slots_available: number
    risk_budget_remaining: number
    error?: string
  }
  system_health: {
    data_quality_score: number
    last_data_update: string | null
    pipeline_status: string
    total_instruments: number
    data_issues: { total_critical: number; below_80: number }
    worst_symbols: string[]
    error?: string
  }
  opportunity_radar: {
    instruments:       OpportunityRadarEntry[]
    early_stage_count: number
    error?:            string
  }
  signal_reliability: SignalReliabilityData
  market_safety:      MarketSafetyData
  generated_at:       string
}

// ── Pattern Scan (batch scan across all instruments) ─────────────────────────

export interface PatternScanEntry {
  symbol:             string
  date:               string
  tf?:                string              // timeframe key, e.g. "1h", "4h", "daily"
  pattern_name:       string
  pattern_category:   'chart' | 'harmonic' | 'gann' | 'wyckoff' | string
  direction:          'bullish' | 'bearish' | 'neutral'
  status:             'FORMING' | 'READY' | 'BREAKOUT' | 'COMPLETED'
  confidence:         number
  breakout_level:     number | null
  target:             number | null
  invalidation_level: number | null
  phase:              string | null
  phase_label:        string | null
  events:             string[]
}

// ── Price Level Clustering (Stage 5) ──────────────────────────────────────────

export interface PriceLevelCluster {
  price:        number
  strength:     number      // 1–10 (sum of source weights, capped)
  sources:      string[]    // e.g. ["MA50", "Fib 61.8%", "Pivot S1"]
  distance_pct: number      // % from current price (negative = below)
  type:         'support' | 'resistance'
  label?:       string      // only on named zones
}

export interface PriceLevelZones {
  buy_zone?:     PriceLevelCluster
  accumulate?:   PriceLevelCluster
  stop?:         PriceLevelCluster
  scale_in?:     PriceLevelCluster
  target?:       PriceLevelCluster
  distribution?: PriceLevelCluster
}

export interface PriceLevels {
  symbol:        string
  current_price: number
  atr:           number
  atr_pct:       number
  levels:        PriceLevelCluster[]
  zones:         PriceLevelZones
}

export interface PatternMTFTF {
  label:          string
  count:          number
  top_confidence: number
  cached:         boolean
}

export interface PatternMTF {
  symbol: string
  tfs:    Record<string, PatternMTFTF>
}

// ── FMP Fundamentals ──────────────────────────────────────────────────────────

export interface Fundamentals {
  symbol: string
  // Company
  company_name:   string | null
  sector:         string | null
  industry:       string | null
  description:    string | null
  ceo:            string | null
  employees:      number | null
  website:        string | null
  ipo_date:       string | null
  is_etf:         boolean
  beta:           number | null
  // Market data
  market_cap:     number | null
  price:          number | null
  year_high:      number | null
  year_low:       number | null
  price_avg_50:   number | null
  price_avg_200:  number | null
  // Valuation
  pe_ratio:       number | null
  peg_ratio:      number | null
  ps_ratio:       number | null
  pb_ratio:       number | null
  ev_ebitda:      number | null
  // Profitability (%)
  gross_margin:       number | null
  operating_margin:   number | null
  net_margin:         number | null
  free_cashflow_margin: number | null
  // Per-share
  eps:            number | null
  fcf_per_share:  number | null
  book_per_share: number | null
  dividend_yield: number | null
  // Analyst targets
  analyst_target_1m:  number | null
  analyst_target_1q:  number | null
  analyst_count_1q:   number | null
  // Earnings
  next_earnings_date: string | null
  next_eps_estimate:  number | null
  next_rev_estimate:  number | null
}

// ── Stage 4 — Pattern Signal (part of 5-stage decision pipeline) ─────────────

export interface Stage4Signal {
  symbol:        string
  signal:        'bullish' | 'bearish' | 'neutral'
  score:         number     // 0-100 magnitude
  net_score:     number     // -100 to +100 directional
  active:        PatternScanEntry[]
  top_pattern:   string | null
  entry:         number | null    // best breakout_level
  target:        number | null    // best target
  stop:          number | null    // best invalidation_level
  pattern_count: number
  bull_count:    number
  bear_count:    number
}

// ── Advanced Cycle Engine ────────────────────────────────────────────────────

export interface CycleMethod {
  cycle_length: number
  strength:     number
}

export interface CycleAdvanced {
  symbol:                string
  dominant_cycle_length: number
  cycle_phase:           number   // 0–1
  cycle_amplitude:       number
  cycle_strength:        number   // 0–1
  cycle_alignment_score: number   // 0–1
  fft_cycles:           CycleMethod[]
  wavelet_cycles:       CycleMethod[]
  hilbert: {
    phase:          number
    amplitude:      number
    phase_velocity: number
  }
  projected_peak_date:   string | null
  projected_peak_price:  number | null
  projected_trough_date:  string | null
  projected_trough_price: number | null
  error?:                string
}

// ── Market Force Vectors ─────────────────────────────────────────────────────

export interface MarketForceData {
  symbol:           string
  trend_force:      number   // -1 to +1
  liquidity_force:  number
  volatility_force: number
  cycle_force:      number
  pattern_force:    number
  net_force:        number
  bias:             'bullish' | 'bearish' | 'neutral'
  force_magnitude:  number   // 0–1
  error?:           string
}

// ── Confluence Nodes & Heatmap ───────────────────────────────────────────────

export interface ConfluenceNode {
  price_low:         number
  price_high:        number
  time_start:        string
  time_end:          string
  confluence_score:  number   // 0–1
  node_type:         'support' | 'resistance'
  direction:         'bullish' | 'bearish' | 'neutral'
  status:            'active' | 'triggered' | 'expired'
  component_scores:  Record<string, number>
  supporting_signals: string[]
}

export interface ConfluenceNodeData {
  symbol: string
  nodes:  ConfluenceNode[]
  error?: string
}

export interface ConfluenceHeatmapData {
  symbol:       string
  price_bins:   number[]
  time_bins:    string[]
  intensity:    number[][]   // price × time grid, 0–1
  max_intensity: number
  error?:       string
}

// ── Strategy Evolution ───────────────────────────────────────────────────────

export interface EvolvedStrategy {
  genome_id:            string
  generation:           number
  entry_confluence_min: number | null
  pattern_type:         string | null
  regime_filter:        string | null
  stop_atr_mult:        number | null
  target_atr_mult:      number | null
  hold_days_max:        number | null
  fitness:              number
  sharpe_ratio:         number | null
  win_rate:             number | null
  profit_factor:        number | null
  max_drawdown:         number | null
  trade_count:          number | null
  is_active:            boolean
}

export interface StrategiesData {
  strategies: EvolvedStrategy[]
  count:      number
}

// ── Signal Confidence ──────────────────────────────────────────────────────

export interface SignalConfidenceComponents {
  historical_reliability: number
  model_agreement:        number
  regime_match:           number
  feature_stability:      number
  confluence_density:     number
}

export interface SignalConfidenceData {
  symbol:           string
  confidence_score: number        // 0–100
  confidence_tier:  'Very Reliable' | 'Reliable' | 'Moderate' | 'Weak'
  components:       SignalConfidenceComponents
  error?:           string
}

// ── Strategy Health (Model Monitor) ────────────────────────────────────────

export interface StrategyHealthEntry {
  setup_type:     string
  health_state:   'HEALTHY' | 'WARNING' | 'CRITICAL'
  action:         'maintain' | 'reduce_size' | 'disable'
  win_rate_drift: number | null
  pf_drift:       number | null
  drawdown_drift: number | null
  recent_win_rate:   number | null
  baseline_win_rate: number | null
  recent_profit_factor:   number | null
  baseline_profit_factor: number | null
  recent_max_drawdown:    number | null
  baseline_max_drawdown:  number | null
}

export interface StrategyHealthData {
  overall_health:  'HEALTHY' | 'WARNING' | 'CRITICAL'
  strategies:      StrategyHealthEntry[]
  healthy_count:   number
  warning_count:   number
  critical_count:  number
  error?:          string
}

// ── Price Distribution Forecast ────────────────────────────────────────────

export interface PriceDistributionData {
  symbol:          string
  current_price:   number
  horizon_days:    number
  quantiles: {
    p10: number
    p25: number
    p50: number
    p75: number
    p90: number
  }
  probability_up:   number
  probability_down: number
  expected_return:  number
  targets?: Array<{
    price:       number
    probability: number
    direction:   'up' | 'down'
  }>
  error?: string
}
