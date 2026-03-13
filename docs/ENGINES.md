# RoboAlgo — Engine Architecture Reference

## Pipeline Stages (1–11)

| Stage | Module | Schedule | Description |
|-------|--------|----------|-------------|
| 1 | `stage1_prices` | Daily 18:30 | Price data ingestion (FMP/yfinance) |
| 2 | `stage2_indicators` | Daily 18:35 | RSI, ATR, MACD, BB, MA50/200 |
| 3 | `stage3_volatility` | Daily 18:40 | Volatility regimes, range compression, breakout detection |
| 4 | `stage4_patterns` | Daily 18:45 | Pattern detection (chart, harmonic, gann, wyckoff) |
| 5 | `stage5_confluence` | Daily 18:50 | 10-component confluence scoring |
| 6 | `stage6_cycles` | Daily 18:55 | FFT/wavelet/Hilbert cycle projection |
| 7 | `stage7_confluence_nodes` | Daily 19:00 | Price-time decision node detection |
| 8 | `stage8_market_physics` | Daily 19:05 | 5-vector net force computation |
| 9 | `stage9_strategy_evolution` | Weekly Sun 02:00 | Genetic algorithm strategy optimization |
| 10 | `stage10_signal_confidence` | Daily 19:15 | 5-component signal trustworthiness scoring |
| 11 | `stage11_model_monitor` | Daily 19:30 | Strategy health drift detection |

---

## Advanced Cycle Engine (`cycle_engine/`)

### Purpose
Detect dominant market cycles using spectral analysis, compute instantaneous phase, and project future turning points.

### Detection Logic

**FFT Cycles (`fft_cycles.py`)**
- Uses Welch's method (scipy.signal.welch) for noise-reduced power spectral density
- Detrends close prices, applies Hann window
- Identifies top N peaks in the power spectrum as dominant cycles
- Returns cycle_length (in bars) and relative strength

**Wavelet Cycles (`wavelet_cycles.py`)**
- Continuous Wavelet Transform (CWT) with Morlet wavelet
- Provides time-frequency decomposition (cycles can change over time)
- Scans periods from 5 to 120 bars
- Returns cycle_length and strength at current time

**Hilbert Phase (`hilbert_phase.py`)**
- Bandpass-filters price around the dominant cycle length
- Applies Hilbert Transform to get analytic signal
- Computes instantaneous phase (0-2pi), amplitude, and phase velocity
- Phase velocity indicates cycle acceleration/deceleration

**Cycle Projection (`cycle_projection.py`)**
- Orchestrator: runs FFT + wavelet, takes consensus dominant cycle
- Applies Hilbert phase to the dominant cycle
- Projects next peak and trough dates/prices using:
  - `bars_to_peak = (0.5 - phase_normalized) * cycle_length` (if phase < 0.5)
  - `bars_to_trough = (1.0 - phase_normalized) * cycle_length`
- Returns `cycle_alignment_score` (0-1): how well FFT and wavelet agree

### Schema
```
CycleProjection: instrument_id, date, dominant_cycle_length, cycle_strength,
                 cycle_phase, cycle_amplitude, cycle_alignment_score,
                 projected_peak_date, projected_peak_price,
                 projected_trough_date, projected_trough_price
```

---

## Geometry Engine (`geometry_engine/`)

### Purpose
Generate Gann geometric price levels, fan projections, and detect price-time symmetry.

### Algorithms

**Gann Angles (`gann_angles.py`)**
- 9 canonical angles: 8x1, 4x1, 3x1, 2x1, 1x1, 1x2, 1x3, 1x4, 1x8
- The 1x1 angle (45 degrees) represents equilibrium: 1 unit of price per 1 unit of time
- Normalization: uses ATR to convert price units to time-comparable units
- `slope_1x1 = ATR_14 / 1 bar` — the "natural" price-per-bar rate

**Gann Fans (`gann_fans.py`)**
- Projects fan lines from significant pivot points (swing highs/lows)
- Selects pivots by amplitude (largest price swings)
- Each fan has 9 lines radiating from the anchor at different angles
- Fan from HIGH: lines project downward
- Fan from LOW: lines project upward
- `fan_adherence_score()`: measures what % of bars touch a fan line (within 2%)

**Square-of-9 (`square_of_9.py`)**
- Gann's geometric price spiral: `level = (sqrt(price) +/- n * increment)^2`
- Default increment = 2.0 (cardinal cross of the spiral)
- Generates n_levels above and below the current price
- Returns nearest support and resistance from the spiral

**Square-of-144 (`square_of_144.py`)**
- Divides a 144-unit price range into Fibonacci fractions: 1/8, 2/8, ... 7/8
- Anchors to the nearest round number below price
- Returns support/resistance levels at each division
- Based on Gann's 144-bar cycle theory

**Price-Time Symmetry (`price_time_symmetry.py`)**
- ATR-normalizes price moves and counts time in bars
- Computes `ratio = price_move_atr / time_move_bars` for each swing
- Ratio near 1.0 = perfect symmetry (price and time are balanced)
- Markets tend to reverse at symmetric points (Gann's "squaring price and time")
- Symmetric swings (0.7-1.4 ratio) scored higher
- Returns symmetry_score (0-100) and list of symmetric swing zones

### Schema
```
GeometryLevel: instrument_id, date, gann_anchor_price, gann_anchor_date,
               fan_adherence_score, sq9_levels (JSON), sq144_levels (JSON),
               price_time_symmetry_score, geometry_score
```

---

## Price-Time Confluence Engine (`confluence_engine/`)

### Purpose
Detect high-probability Decision Nodes where multiple independent analysis engines converge on the same price-time zone.

### Node Detection Logic (`node_detector.py`)

1. **Collect levels from 5 sources:**
   - Cycle projections: peak/trough prices from FFT/wavelet/Hilbert
   - Geometry: Square-of-9 and Square-of-144 support/resistance
   - Swing structure: recent swing highs/lows as S/R
   - Patterns: breakout levels and targets from active patterns
   - Price-time symmetry: zones where ratio approaches 1.0

2. **Cluster nearby levels:**
   - Groups levels within `tolerance_pct` (default 1.5%) of each other
   - More sources agreeing on a level = higher confluence

3. **Score each cluster:**
   - `swing_structure`: 0-60 points based on touch count
   - `cycle`: 0-100 based on alignment score
   - `geometry`: 0-20 based on how many Sq-9/Sq-144 levels cluster
   - `symmetry`: 0-100 from symmetry score
   - Final score = weighted combination, normalized to 0-1

4. **Classify nodes:**
   - `support`: cluster below current price
   - `resistance`: cluster above current price
   - Direction based on majority of contributing signal directions

### Heatmap Generation (`heatmap.py`)

- Creates a 2D grid: price (Y) x time (X)
- Price axis: current_price +/- 5 ATR, divided into `n_price_bins` bins
- Time axis: 0 to `n_time_bins` bars forward
- Each level contributor adds Gaussian-spread intensity to nearby grid cells
- Normalization: all cells divided by max intensity to produce 0-1 range
- Higher intensity = more engines agree on that price-time zone

### Schema
```
ConfluenceNode: instrument_id, date, price_low, price_high,
                time_window_start, time_window_end, confluence_score,
                component_scores (JSON), supporting_signals (JSON),
                node_type, direction, status
```

---

## Market Physics Engine (`physics_engine/`)

### Purpose
Model market forces as directional vectors (-1 to +1), then combine into a net force indicating overall market pressure.

### Force Vectors

| Force | Weight | Source | Logic |
|-------|--------|--------|-------|
| Trend | 30% | MA50 slope + price vs MA + MACD | Positive when price trending up |
| Liquidity | 20% | Distance to recent highs/lows/VWAP | Positive when near support |
| Volatility | 15% | ATR expansion rate + BB width change | Positive on expansion |
| Cycle | 20% | `-sin(2pi * phase) * strength` | Positive approaching trough |
| Pattern | 15% | Confidence * direction per pattern | Weighted by pattern status |

### Net Force
```
net_force = sum(weight_i * force_i)
bias = "bullish" if net > 0.1 else "bearish" if net < -0.1 else "neutral"
force_magnitude = abs(net_force)
```

---

## Signal Confidence Engine (`signal_confidence_engine/`)

### Purpose
Measure the trustworthiness of each trading signal using 5 independent metrics.

### Components

| Component | Weight | Score Range | Logic |
|-----------|--------|-------------|-------|
| Historical Reliability | 30% | 0-100 | Win rate + profit factor from past 50 signals |
| Model Agreement | 25% | 0-100 | Directional consensus across 4 analysis methods |
| Regime Match | 20% | 0-100 | Current vol regime suitability for trading |
| Feature Stability | 15% | 0-100 | Low coefficient-of-variation in RSI/ATR/MACD |
| Confluence Density | 10% | 0-100 | Latest confluence score from master scorer |

### Tiers
- 80-100: **Very Reliable** — high conviction signal
- 60-80: **Reliable** — standard execution
- 40-60: **Moderate** — reduced size
- <40: **Weak** — monitor only

---

## Model Monitoring Engine (`model_monitor/`)

### Purpose
Detect when trading signal models degrade in live markets by comparing recent vs historical performance.

### Detection Logic

1. **Live Metrics (`live_metrics.py`)**: Computes win_rate, profit_factor, average_return, max_drawdown from PaperTrade table for a given lookback period.

2. **Drift Detection (`drift_detector.py`)**: Compares 30-day (recent) vs 180-day (baseline) metrics:
   - `win_rate_drift = recent - baseline` (negative = degrading)
   - HEALTHY: all drifts within tolerance
   - WARNING: moderate degradation (-5% to -15% win rate)
   - CRITICAL: severe degradation (>15% win rate drop or >25% drawdown increase)

3. **Signal Guard (`signal_guard.py`)**: Evaluates all strategies, produces:
   - Overall health state (worst across all strategies)
   - Per-strategy action: maintain / reduce_size / disable

---

## Price Distribution Engine (`distribution_engine/`)

### Purpose
Forecast future price distributions using historical returns and Monte Carlo simulation.

### Methods

**Quantile Model (`quantile_model.py`)**
- Computes log returns from close prices
- Scales return distribution by `sqrt(horizon_days)` for time projection
- Produces p10/p25/p50/p75/p90 price quantiles

**Monte Carlo (`monte_carlo.py`)**
- Geometric Brownian Motion: `S(t+dt) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)`
- 1000 simulation paths over the forecast horizon
- Returns quantile prices + probability_up/down + representative path trajectories

**Range Probability (`range_probability.py`)**
- Blends quantile and MC results (equal weight)
- Computes probability of reaching specific price targets
- Auto-generates targets at +/-5% and +/-10% if not specified

---

## Celery/Redis Infrastructure (`infrastructure/`)

### Purpose
Distributed task execution for parallel symbol processing across pipeline stages.

### Architecture
- **Redis**: Message broker + result backend
- **Queues**: `pipeline` (sequential stages), `symbols` (parallel per-symbol)
- **Beat scheduler**: Cron-based recurring pipeline execution

### Worker Profiles
- `pipeline`: Single-process worker for sequential stage execution
- `symbols`: Multi-process worker (4 concurrent) for parallel symbol processing
- `all`: Development worker handling all queues

### Commands
```bash
# Start pipeline worker
celery -A infrastructure.celery_app worker -Q pipeline -c 1 --loglevel=info

# Start symbol workers (parallel)
celery -A infrastructure.celery_app worker -Q symbols -c 4 --loglevel=info

# Start scheduler
celery -A infrastructure.celery_app beat --loglevel=info
```
