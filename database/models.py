"""
RoboAlgo - Database Models
SQLAlchemy ORM models for all core tables.
"""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, Date, DateTime, ForeignKey,
    UniqueConstraint, Index, Boolean,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Instrument(Base):
    """Tradeable instrument metadata."""
    __tablename__ = "instruments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), unique=True, nullable=False, index=True)
    name = Column(String(200))
    instrument_type = Column(String(50))  # leveraged_etf_bull, leveraged_etf_bear, stock, index
    leverage_factor = Column(Float)
    underlying = Column(String(200))
    pair_symbol = Column(String(20))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    prices = relationship("PriceData", back_populates="instrument", cascade="all, delete-orphan")
    indicators = relationship("Indicator", back_populates="instrument", cascade="all, delete-orphan")
    features = relationship("Feature", back_populates="instrument", cascade="all, delete-orphan")
    cycle_metrics = relationship("CycleMetric", back_populates="instrument", cascade="all, delete-orphan")
    signals = relationship("Signal", back_populates="instrument", cascade="all, delete-orphan")
    patterns = relationship("PatternDetection", back_populates="instrument", cascade="all, delete-orphan")
    volatility_regimes = relationship("VolatilityRegime", back_populates="instrument", cascade="all, delete-orphan")
    # New confluence-layer tables
    range_compressions = relationship("RangeCompression", back_populates="instrument", cascade="all, delete-orphan")
    breakout_signals = relationship("BreakoutSignal", back_populates="instrument", cascade="all, delete-orphan")
    liquidity_levels = relationship("LiquidityLevel", back_populates="instrument", cascade="all, delete-orphan")
    wyckoff_phases = relationship("WyckoffPhase", back_populates="instrument", cascade="all, delete-orphan")
    harmonic_patterns = relationship("HarmonicPattern", back_populates="instrument", cascade="all, delete-orphan")
    chart_patterns = relationship("ChartPattern", back_populates="instrument", cascade="all, delete-orphan")
    confluence_scores = relationship("ConfluenceScore", back_populates="instrument", cascade="all, delete-orphan")
    cycle_projections = relationship("CycleProjection", back_populates="instrument", cascade="all, delete-orphan")
    geometry_levels = relationship("GeometryLevel", back_populates="instrument", cascade="all, delete-orphan")
    confluence_nodes = relationship("ConfluenceNode", back_populates="instrument", cascade="all, delete-orphan")
    market_forces = relationship("MarketForce", back_populates="instrument", cascade="all, delete-orphan")
    price_distributions = relationship("PriceDistribution", back_populates="instrument", cascade="all, delete-orphan")
    signal_confidences = relationship("SignalConfidence", back_populates="instrument", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Instrument(symbol='{self.symbol}')>"


class PriceData(Base):
    """Daily OHLCV price data."""
    __tablename__ = "price_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    instrument_id = Column(Integer, ForeignKey("instruments.id"), nullable=False)
    date = Column(Date, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

    instrument = relationship("Instrument", back_populates="prices")

    __table_args__ = (
        UniqueConstraint("instrument_id", "date", name="uq_price_instrument_date"),
        Index("ix_price_data_instrument_date", "instrument_id", "date"),
    )


class Indicator(Base):
    """Technical indicator values."""
    __tablename__ = "indicators"

    id = Column(Integer, primary_key=True, autoincrement=True)
    instrument_id = Column(Integer, ForeignKey("instruments.id"), nullable=False)
    date = Column(Date, nullable=False)

    rsi = Column(Float)
    atr = Column(Float)
    macd_line = Column(Float)
    macd_signal = Column(Float)
    macd_histogram = Column(Float)
    bb_upper = Column(Float)
    bb_middle = Column(Float)
    bb_lower = Column(Float)
    bb_width = Column(Float)
    ma50 = Column(Float)
    ma200 = Column(Float)

    created_at = Column(DateTime, default=datetime.utcnow)

    instrument = relationship("Instrument", back_populates="indicators")

    __table_args__ = (
        UniqueConstraint("instrument_id", "date", name="uq_indicator_instrument_date"),
        Index("ix_indicators_instrument_date", "instrument_id", "date"),
    )


class Feature(Base):
    """Normalized feature vectors for ML models."""
    __tablename__ = "features"

    id = Column(Integer, primary_key=True, autoincrement=True)
    instrument_id = Column(Integer, ForeignKey("instruments.id"), nullable=False)
    date = Column(Date, nullable=False)

    trend_strength = Column(Float)
    momentum = Column(Float)
    volatility_percentile = Column(Float)
    volume_ratio = Column(Float)
    cycle_phase = Column(Float)
    macd_norm = Column(Float)
    bb_position = Column(Float)
    price_to_ma50 = Column(Float)
    return_5d = Column(Float)
    return_20d = Column(Float)

    # ── v2 factors (from system audit) ─────────────────────────────────────
    momentum_acceleration = Column(Float)   # RSI 3-day acceleration, normalised -1→+1
    volume_participation  = Column(Float)   # directional volume quality, -1→+1
    correlation_exposure  = Column(Float)   # 20-day rolling corr to equal-weight portfolio, -1→+1

    created_at = Column(DateTime, default=datetime.utcnow)

    instrument = relationship("Instrument", back_populates="features")

    __table_args__ = (
        UniqueConstraint("instrument_id", "date", name="uq_feature_instrument_date"),
        Index("ix_features_instrument_date", "instrument_id", "date"),
    )


class CycleMetric(Base):
    """Cycle detection metrics from spectral analysis."""
    __tablename__ = "cycle_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    instrument_id = Column(Integer, ForeignKey("instruments.id"), nullable=False)
    date = Column(Date, nullable=False)

    cycle_length = Column(Float)      # dominant cycle period in trading days
    cycle_phase = Column(Float)       # 0.0-1.0 normalized position in cycle
    cycle_strength = Column(Float)    # spectral power of dominant cycle

    created_at = Column(DateTime, default=datetime.utcnow)

    instrument = relationship("Instrument", back_populates="cycle_metrics")

    __table_args__ = (
        UniqueConstraint("instrument_id", "date", name="uq_cycle_instrument_date"),
        Index("ix_cycle_metrics_instrument_date", "instrument_id", "date"),
    )


class PatternDetection(Base):
    """Detected backend pattern engine outputs (chart/harmonic/gann/wyckoff)."""
    __tablename__ = "pattern_detections"

    id = Column(Integer, primary_key=True, autoincrement=True)
    instrument_id = Column(Integer, ForeignKey("instruments.id"), nullable=False)
    date = Column(Date, nullable=False)

    pattern_name = Column(String(50), nullable=False)   # e.g. "Hammer", "Double Bottom"
    pattern_type = Column(String(20))                   # candlestick | chart
    pattern_category = Column(String(20))               # chart | harmonic | gann | wyckoff
    status = Column(String(20))                         # NOT_PRESENT|FORMING|READY|BREAKOUT|FAILED|COMPLETED
    direction = Column(String(10))                      # bullish | bearish | neutral
    strength = Column(Float)                            # 0.0 – 1.0
    price_level = Column(Float)                         # relevant price (low, high, close, S/R)
    breakout_level = Column(Float)
    target = Column(Float)
    confidence = Column(Float)                          # confluence-weighted score 0–100
    points = Column(String(4000))                       # JSON-encoded anchor points

    created_at = Column(DateTime, default=datetime.utcnow)

    instrument = relationship("Instrument", back_populates="patterns")

    __table_args__ = (
        UniqueConstraint("instrument_id", "date", "pattern_name",
                         name="uq_pattern_instrument_date_name"),
        Index("ix_pattern_instrument_date", "instrument_id", "date"),
        Index("ix_pattern_name", "pattern_name"),
        # Confluence engine filters heavily on status and confidence
        Index("ix_pattern_status", "status"),
        Index("ix_pattern_confidence", "confidence"),
        # Composite for confluence query: WHERE instrument_id=? ORDER BY date DESC, confidence DESC
        Index("ix_pattern_inst_date_conf", "instrument_id", "date", "confidence"),
    )


class Signal(Base):
    """Trading signals with confidence tiers and 6-phase market classification."""
    __tablename__ = "signals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    instrument_id = Column(Integer, ForeignKey("instruments.id"), nullable=False)
    date = Column(Date, nullable=False)

    # Probability & classification
    probability = Column(Float)           # XGBoost probability 0.0–1.0
    confidence_tier = Column(String(10))  # HIGH / MEDIUM / LOW
    market_phase = Column(String(30))     # one of 6 cycle phases

    # ATR-based trade plan
    buy_price = Column(Float)             # entry — current close
    accumulate_price = Column(Float)      # add zone — close − 1 ATR
    scale_price = Column(Float)           # scale-out — close + 2 ATR
    sell_price = Column(Float)            # full target — close + 4 ATR

    # ── Regime-aware fields (playbook v2) ──────────────────────────────────
    market_state        = Column(String(15))    # COMPRESSION/TREND/EXPANSION/CHAOS
    strategy_mode       = Column(String(50))    # "Breakout Momentum" / "Trend Pullback" / …
    setup_quality_score = Column(Float)         # SetupQualityScore 0–100
    decision_trace      = Column(String(4000))  # full human-readable trace

    created_at = Column(DateTime, default=datetime.utcnow)

    instrument = relationship("Instrument", back_populates="signals")

    __table_args__ = (
        Index("ix_signals_instrument_date", "instrument_id", "date"),
        Index("ix_signals_tier", "confidence_tier"),
        Index("ix_signals_phase", "market_phase"),
        Index("ix_signals_market_state", "market_state"),
    )


class VolatilityRegime(Base):
    """
    Daily volatility regime classification per instrument.
    Gates signal generation: LOW_VOL → no signals, NORMAL_VOL → limited, HIGH_VOL → active.
    """
    __tablename__ = "volatility_regimes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    instrument_id = Column(Integer, ForeignKey("instruments.id"), nullable=False)
    date = Column(Date, nullable=False)

    # Raw volatility metrics
    atr_pct           = Column(Float)   # ATR / close (normalized ATR % of price)
    bb_width          = Column(Float)   # (BB_upper - BB_lower) / BB_middle
    realized_vol_20d  = Column(Float)   # 20-day annualized realized volatility

    # Rolling percentile ranks (0.0–1.0) over 252-day lookback
    atr_percentile      = Column(Float)
    bb_width_percentile = Column(Float)
    vol_percentile      = Column(Float)

    # Regime classification
    regime = Column(String(15), nullable=False, default="NORMAL_VOL")
    # LOW_VOL | NORMAL_VOL | HIGH_VOL

    # Structural event flags
    is_compression = Column(Boolean, default=False)  # BB_pct<15% AND ATR_pct<20%
    is_expansion   = Column(Boolean, default=False)  # breakout + volume surge + momentum shift

    # Expansion detail
    compression_range_high = Column(Float)   # top of prior compression range
    compression_range_low  = Column(Float)   # bottom of prior compression range

    created_at = Column(DateTime, default=datetime.utcnow)

    instrument = relationship("Instrument", back_populates="volatility_regimes")

    __table_args__ = (
        UniqueConstraint("instrument_id", "date", name="uq_vol_regime_instrument_date"),
        Index("ix_vol_regime_instrument_date", "instrument_id", "date"),
        Index("ix_vol_regime_regime", "regime"),
        Index("ix_vol_regime_compression", "is_compression"),
    )


# ── Confluence-Layer Tables ────────────────────────────────────────────────────

class RangeCompression(Base):
    """
    Multi-timeframe range compression detection.
    Detects energy buildup before volatility expansion.
    """
    __tablename__ = "range_compressions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    instrument_id = Column(Integer, ForeignKey("instruments.id"), nullable=False)
    date = Column(Date, nullable=False)

    # Raw compression metrics
    bb_width_pct       = Column(Float)  # BB width percentile rank (0–1)
    atr_pct            = Column(Float)  # ATR percentile rank (0–1)
    range_10bar        = Column(Float)  # 10-bar high-low range
    range_30bar        = Column(Float)  # 30-bar high-low range
    range_ratio        = Column(Float)  # range_10bar / range_30bar (<1.0 = contracting)

    # Compression state
    is_compressed      = Column(Boolean, default=False)
    compression_duration = Column(Integer, default=0)  # consecutive bars in compression

    # Range geometry
    range_high  = Column(Float)  # highest high in compression window
    range_low   = Column(Float)  # lowest low in compression window
    range_mid   = Column(Float)  # midpoint of compression range

    # Composite scores
    daily_compression = Column(Float)  # daily compression sub-score (0–100)
    h4_compression    = Column(Float)  # 4h compression sub-score (0–100)
    h1_compression    = Column(Float)  # 1h compression sub-score (0–100)
    compression_score = Column(Float)  # weighted MTF: 0.5×daily + 0.3×4h + 0.2×1h (0–100)

    created_at = Column(DateTime, default=datetime.utcnow)
    instrument = relationship("Instrument", back_populates="range_compressions")

    __table_args__ = (
        UniqueConstraint("instrument_id", "date", name="uq_range_comp_inst_date"),
        Index("ix_range_comp_inst_date", "instrument_id", "date"),
        Index("ix_range_comp_compressed", "is_compressed"),
    )


class BreakoutSignal(Base):
    """
    Breakout detection from range compression.
    Requires ≥2 of 3 triggers: price, volume, momentum.
    """
    __tablename__ = "breakout_signals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    instrument_id = Column(Integer, ForeignKey("instruments.id"), nullable=False)
    date = Column(Date, nullable=False)

    # Direction
    breakout_direction = Column(String(10))  # up / down
    breakout_price     = Column(Float)
    compression_range_high = Column(Float)
    compression_range_low  = Column(Float)

    # Trigger flags (need ≥2 of 3)
    price_trigger    = Column(Boolean, default=False)  # close > range_high (or < range_low)
    volume_trigger   = Column(Boolean, default=False)  # volume_ratio > 1.5
    momentum_trigger = Column(Boolean, default=False)  # RSI rising + MACD positive
    triggers_met     = Column(Integer, default=0)      # count of triggers met (0–3)

    # Breakout strength formula components
    breakout_distance          = Column(Float)  # |close - range edge| / ATR
    volume_ratio               = Column(Float)  # current volume / 20-day avg volume
    momentum_score             = Column(Float)  # composite momentum (RSI + MACD + velocity)
    volatility_expansion_speed = Column(Float)  # rate of ATR expansion
    compression_duration       = Column(Integer)  # bars in prior compression
    compression_score          = Column(Float)    # compression score at breakout

    # Final strength score: 0.35×comp_dur + 0.30×vol_ratio + 0.20×dist/ATR + 0.15×exp_speed
    breakout_strength = Column(Float)  # 0–100

    created_at = Column(DateTime, default=datetime.utcnow)
    instrument = relationship("Instrument", back_populates="breakout_signals")

    __table_args__ = (
        UniqueConstraint("instrument_id", "date", name="uq_breakout_inst_date"),
        Index("ix_breakout_inst_date", "instrument_id", "date"),
        Index("ix_breakout_direction", "breakout_direction"),
        Index("ix_breakout_strength", "breakout_strength"),
    )


class LiquidityLevel(Base):
    """Key liquidity price levels and sweep events for confluence scoring."""
    __tablename__ = "liquidity_levels"

    id = Column(Integer, primary_key=True, autoincrement=True)
    instrument_id = Column(Integer, ForeignKey("instruments.id"), nullable=False)
    date = Column(Date, nullable=False)

    # Key reference levels
    prev_day_high = Column(Float)
    prev_day_low  = Column(Float)
    high_3d       = Column(Float)
    low_3d        = Column(Float)
    high_week     = Column(Float)  # 5-bar high
    low_week      = Column(Float)  # 5-bar low
    vwap          = Column(Float)  # approximate VWAP (cumulative / volume-weighted avg)

    # Volume-node levels (highest-volume price zones)
    vol_node_high = Column(Float)  # upper high-volume node
    vol_node_low  = Column(Float)  # lower high-volume node

    # Sweep events (liquidity grabs above/below key levels)
    swept_prev_high  = Column(Boolean, default=False)  # price swept prev-day high
    swept_prev_low   = Column(Boolean, default=False)  # price swept prev-day low
    swept_week_high  = Column(Boolean, default=False)
    swept_week_low   = Column(Boolean, default=False)

    # Price position relative to liquidity
    above_vwap       = Column(Boolean, default=False)
    near_vol_node    = Column(Boolean, default=False)  # within 1% of major volume node

    # Confluence contribution
    liquidity_score  = Column(Float)  # 0–100

    created_at = Column(DateTime, default=datetime.utcnow)
    instrument = relationship("Instrument", back_populates="liquidity_levels")

    __table_args__ = (
        UniqueConstraint("instrument_id", "date", name="uq_liq_inst_date"),
        Index("ix_liq_inst_date", "instrument_id", "date"),
    )


class WyckoffPhase(Base):
    """Wyckoff market phase identification with structural event detection."""
    __tablename__ = "wyckoff_phases"

    id = Column(Integer, primary_key=True, autoincrement=True)
    instrument_id = Column(Integer, ForeignKey("instruments.id"), nullable=False)
    date = Column(Date, nullable=False)

    # Phase classification
    phase     = Column(String(20))   # Accumulation / Markup / Distribution / Markdown / Unknown
    sub_phase = Column(String(30))   # Phase A, B, C, D, E (Wyckoff sub-stages)

    # Structural event flags
    spring_detected          = Column(Boolean, default=False)  # false breakdown below support
    upthrust_detected        = Column(Boolean, default=False)  # false breakout above resistance
    secondary_test_detected  = Column(Boolean, default=False)  # retest of S/R after event

    # Key levels
    support_level    = Column(Float)
    resistance_level = Column(Float)

    # Volume context
    volume_trend = Column(String(15))  # rising / falling / neutral

    # Confidence and contribution
    confidence   = Column(Float)   # 0–1
    phase_score  = Column(Float)   # 0–100 for confluence

    created_at = Column(DateTime, default=datetime.utcnow)
    instrument = relationship("Instrument", back_populates="wyckoff_phases")

    __table_args__ = (
        UniqueConstraint("instrument_id", "date", name="uq_wyckoff_inst_date"),
        Index("ix_wyckoff_inst_date", "instrument_id", "date"),
        Index("ix_wyckoff_phase", "phase"),
    )


class HarmonicPattern(Base):
    """Detected XABCD harmonic patterns (Gartley, Bat, Butterfly, Crab)."""
    __tablename__ = "harmonic_patterns"

    id = Column(Integer, primary_key=True, autoincrement=True)
    instrument_id = Column(Integer, ForeignKey("instruments.id"), nullable=False)
    date = Column(Date, nullable=False)  # D-point detection date

    # Pattern classification
    pattern_type = Column(String(20))  # Gartley / Bat / Butterfly / Crab
    direction    = Column(String(10))  # bullish / bearish

    # XABCD pivot prices
    x_price = Column(Float)
    a_price = Column(Float)
    b_price = Column(Float)
    c_price = Column(Float)
    d_price = Column(Float)  # completion / reversal zone

    # Fibonacci ratios
    xab_ratio = Column(Float)  # B retracement of XA
    abc_ratio = Column(Float)  # C retracement of AB
    bcd_ratio = Column(Float)  # D extension of BC
    xad_ratio = Column(Float)  # D retracement of XA

    # Potential Reversal Zone
    prz_high = Column(Float)
    prz_low  = Column(Float)

    # Quality metrics
    confidence    = Column(Float)   # 0–1 (ratio accuracy)
    pattern_score = Column(Float)   # 0–100 for confluence

    created_at = Column(DateTime, default=datetime.utcnow)
    instrument = relationship("Instrument", back_populates="harmonic_patterns")

    __table_args__ = (
        Index("ix_harmonic_inst_date", "instrument_id", "date"),
        Index("ix_harmonic_type", "pattern_type"),
    )


class ChartPattern(Base):
    """Detected chart patterns: flags, triangles, wedges, channels, head & shoulders."""
    __tablename__ = "chart_patterns"

    id = Column(Integer, primary_key=True, autoincrement=True)
    instrument_id = Column(Integer, ForeignKey("instruments.id"), nullable=False)
    date = Column(Date, nullable=False)  # detection date (latest bar)

    # Pattern classification
    pattern_type = Column(String(30))   # flag / triangle / wedge / channel / head_and_shoulders
    direction    = Column(String(10))   # bullish / bearish / neutral
    sub_type     = Column(String(30))   # ascending_triangle / descending_wedge / etc.

    # Pattern geometry
    start_date       = Column(Date)
    bars_in_pattern  = Column(Integer)
    breakout_price   = Column(Float)  # expected breakout level
    target_price     = Column(Float)  # measured move target (pattern height projected)
    stop_price       = Column(Float)  # invalidation level

    # Quality
    pattern_confidence = Column(Float)  # 0–1
    pattern_score      = Column(Float)  # 0–100 for confluence

    created_at = Column(DateTime, default=datetime.utcnow)
    instrument = relationship("Instrument", back_populates="chart_patterns")

    __table_args__ = (
        Index("ix_chart_pattern_inst_date", "instrument_id", "date"),
        Index("ix_chart_pattern_type", "pattern_type"),
    )


class ConfluenceScore(Base):
    """
    Weighted multi-engine confluence score with full decision trace.
    Gates final signal generation — only HIGH/MEDIUM tiers generate executable signals.
    """
    __tablename__ = "confluence_scores"

    id = Column(Integer, primary_key=True, autoincrement=True)
    instrument_id = Column(Integer, ForeignKey("instruments.id"), nullable=False)
    date = Column(Date, nullable=False)

    # Component scores 0–100 (pre-weight) — 10 components
    vol_compression_score     = Column(Float)   # weight: 20%
    breakout_score            = Column(Float)   # weight: 15%
    trend_score               = Column(Float)   # weight: 12%
    liquidity_score           = Column(Float)   # weight: 10%
    pattern_score             = Column(Float)   # weight:  8%
    wyckoff_score             = Column(Float)   # weight:  8%
    gann_score                = Column(Float)   # weight:  4%
    cycle_alignment_score     = Column(Float)   # weight: 10% (FFT/wavelet/Hilbert)
    price_time_symmetry_score = Column(Float)   # weight:  8% (Gann geometry)
    harmonic_confluence_score = Column(Float)   # weight:  5% (PRZ proximity)

    # Weighted total
    confluence_score = Column(Float)  # 0–100
    signal_tier      = Column(String(10))  # HIGH (≥80) / MEDIUM (60-79) / WATCH (50-59) / NONE

    # Context at scoring time
    volatility_regime   = Column(String(15))
    is_compression      = Column(Boolean)
    is_breakout         = Column(Boolean)
    expected_move_pct   = Column(Float)   # expected move as % of price

    # Trade plan
    entry_price  = Column(Float)
    add_price    = Column(Float)   # accumulate zone
    scale_price  = Column(Float)  # first target / scale-out
    target_price = Column(Float)  # full target
    stop_price   = Column(Float)  # invalidation stop

    # Full human-readable decision trace (JSON)
    decision_trace = Column(String(8000))

    created_at = Column(DateTime, default=datetime.utcnow)
    instrument = relationship("Instrument", back_populates="confluence_scores")

    __table_args__ = (
        UniqueConstraint("instrument_id", "date", name="uq_confluence_inst_date"),
        Index("ix_confluence_inst_date", "instrument_id", "date"),
        Index("ix_confluence_tier", "signal_tier"),
        Index("ix_confluence_score", "confluence_score"),
        # Composite for get_top_signals: WHERE score >= X ORDER BY date DESC, score DESC
        Index("ix_confluence_date_score", "date", "confluence_score"),
        # Composite for signal tier + score lookup (dashboard panel)
        Index("ix_confluence_tier_score", "signal_tier", "confluence_score"),
    )


# ── Paper Trading Tables ───────────────────────────────────────────────────────

class PaperAccount(Base):
    """Daily paper account snapshot tracking balance and P&L."""
    __tablename__ = "paper_account"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, unique=True)
    starting_balance = Column(Float, nullable=False)
    ending_balance = Column(Float, nullable=False)
    daily_pnl = Column(Float, default=0.0)
    open_positions = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)


class PaperPosition(Base):
    """Open paper trading positions."""
    __tablename__ = "paper_positions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    entry_date = Column(Date, nullable=False)
    entry_price = Column(Float, nullable=False)
    position_size = Column(Integer, nullable=False)   # number of shares
    position_value = Column(Float, nullable=False)    # entry_price * position_size
    direction = Column(String(10), default="long")    # long / short
    stop_price = Column(Float)
    target_price = Column(Float)
    signal_probability = Column(Float)
    confidence_tier = Column(String(10))
    market_phase = Column(String(30))
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_paper_positions_symbol", "symbol"),
        Index("ix_paper_positions_entry_date", "entry_date"),
    )


class PaperTrade(Base):
    """Completed paper trades with P&L."""
    __tablename__ = "paper_trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    entry_date = Column(Date, nullable=False)
    exit_date = Column(Date, nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=False)
    position_size = Column(Integer, nullable=False)   # shares
    direction = Column(String(10), default="long")
    pnl = Column(Float, nullable=False)               # dollar P&L
    return_percent = Column(Float, nullable=False)    # % return
    exit_reason = Column(String(20))                  # target / stop / signal_reversal
    signal_probability = Column(Float)
    confidence_tier = Column(String(10))
    market_phase = Column(String(30))
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_paper_trades_symbol", "symbol"),
        Index("ix_paper_trades_exit_date", "exit_date"),
    )


# ── Market State Table ─────────────────────────────────────────────────────────

class MarketState(Base):
    """Daily market state classification per instrument (COMPRESSION/TREND/EXPANSION/CHAOS)."""
    __tablename__ = "market_states"

    id = Column(Integer, primary_key=True, autoincrement=True)
    instrument_id = Column(Integer, ForeignKey("instruments.id"), nullable=False)
    date = Column(Date, nullable=False)

    state = Column(String(15), nullable=False, default="COMPRESSION")
    volatility_percentile  = Column(Float)
    trend_strength         = Column(Float)
    expansion_strength     = Column(Float)
    adx                    = Column(Float)
    ma_alignment           = Column(String(15))
    volume_ratio           = Column(Float)
    atr_change_pct         = Column(Float)
    size_multiplier        = Column(Float, default=1.0)

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("instrument_id", "date", name="uq_market_state_inst_date"),
        Index("ix_market_state_inst_date", "instrument_id", "date"),
        Index("ix_market_state_state", "state"),
    )


# ── Expectancy + Trade Lifecycle Tables ────────────────────────────────────────

class SetupPerformance(Base):
    """
    Per-setup-type expectancy tracking.
    Updated after every trade exit to track what setups actually work.
    """
    __tablename__ = "setup_performance"

    id = Column(Integer, primary_key=True, autoincrement=True)
    setup_type    = Column(String(40), nullable=False)  # compression_breakout, trend_pullback, etc.
    market_state  = Column(String(15))                  # COMPRESSION / TREND / EXPANSION / CHAOS

    # Performance metrics
    win_rate      = Column(Float, default=0.0)   # wins / total
    avg_win       = Column(Float, default=0.0)   # avg return on wins
    avg_loss      = Column(Float, default=0.0)   # avg return on losses (negative)
    profit_factor = Column(Float, default=0.0)   # gross_profit / gross_loss
    expected_value = Column(Float, default=0.0)  # win_rate×avg_win + loss_rate×avg_loss
    trade_count   = Column(Integer, default=0)
    updated_at    = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("setup_type", "market_state", name="uq_setup_perf_type_state"),
        Index("ix_setup_perf_type", "setup_type"),
        Index("ix_setup_perf_ev", "expected_value"),
    )


class TradeLifecycle(Base):
    """
    Full trade lifecycle: SETUP → TRIGGER → ENTRY → ACTIVE → EXIT
    Every trade moves through these states for full auditability.
    """
    __tablename__ = "trade_lifecycle"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol        = Column(String(20), nullable=False, index=True)
    setup_type    = Column(String(40))  # compression_breakout / trend_pullback / etc.
    market_state  = Column(String(15))  # state when trade was set up

    # Lifecycle state
    state = Column(String(10), default="SETUP")  # SETUP/TRIGGER/ENTRY/ACTIVE/EXIT

    # Prices — entry + 3-Tier Trade Plan
    entry_price   = Column(Float)
    exit_price    = Column(Float)
    stop_price    = Column(Float)
    target_price  = Column(Float)    # legacy / general target
    tier1_sell    = Column(Float)    # Tier 1 — first pop (sell 1/3)
    tier2_sell    = Column(Float)    # Tier 2 — swing target (sell 1/3)
    tier3_hold    = Column(Float)    # Tier 3 — long-term (hold as house money)

    # Size
    position_size = Column(Integer)   # shares
    direction     = Column(String(10), default="long")

    # Timestamps
    setup_timestamp   = Column(DateTime)
    trigger_timestamp = Column(DateTime)
    entry_timestamp   = Column(DateTime)
    exit_timestamp    = Column(DateTime)

    # Outcomes
    pnl            = Column(Float)
    return_percent = Column(Float)
    holding_period = Column(Integer)  # days held
    exit_reason    = Column(String(30))
    notes          = Column(String(500))

    # Confluence context
    confluence_score = Column(Float)
    setup_type_ev    = Column(Float)   # expected value of this setup type at trade time

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_lifecycle_symbol", "symbol"),
        Index("ix_lifecycle_state", "state"),
        Index("ix_lifecycle_setup_type", "setup_type"),
    )


class RegimeStrategyPerformance(Base):
    """
    Performance tracking by market regime + setup type combination.
    Feeds the regime-adaptive strategy engine.
    """
    __tablename__ = "regime_strategy_performance"

    id = Column(Integer, primary_key=True, autoincrement=True)
    market_state  = Column(String(15), nullable=False)
    setup_type    = Column(String(40), nullable=False)

    win_rate       = Column(Float, default=0.0)
    avg_return     = Column(Float, default=0.0)
    expected_value = Column(Float, default=0.0)
    trade_count    = Column(Integer, default=0)
    updated_at     = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("market_state", "setup_type", name="uq_regime_strategy_perf"),
        Index("ix_regime_strat_state", "market_state"),
        Index("ix_regime_strat_ev", "expected_value"),
    )


# ── Setup Quality Score (v2 — from system audit) ───────────────────────────────

class SetupQualityScore(Base):
    """
    Composite setup-quality score (0–100) aggregating confluence, breakout,
    momentum_acceleration, volume_participation, and correlation_exposure.

    Grades:  A ≥ 78  |  B 62–77  |  C 48–61  |  D < 48

    Weights
    -------
    confluence_score      30 %   primary gate score
    breakout_score        20 %   breakout power (0 when no breakout)
    momentum_acceleration 15 %   RSI 3-day acceleration
    volume_participation  15 %   directional volume quality
    correlation_exposure  10 %   lower correlation → higher score
    market_state          05 %   EXPANSION=1.0, TREND=0.85, COMP=0.65, CHAOS=0.20
    volatility_regime     05 %   LOW=0.90, NORMAL=0.75, HIGH=0.40
    """
    __tablename__ = "setup_quality_scores"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    instrument_id = Column(Integer, ForeignKey("instruments.id"), nullable=False)
    date          = Column(Date, nullable=False)
    symbol        = Column(String(20), nullable=False)  # denormalised for fast lookup

    # ── Input factors ──────────────────────────────────────────────────────
    confluence_score      = Column(Float)  # 0–100
    breakout_score        = Column(Float)  # 0–100
    momentum_acceleration = Column(Float)  # -1→+1
    volume_participation  = Column(Float)  # -1→+1
    correlation_exposure  = Column(Float)  # -1→+1

    # ── Context ────────────────────────────────────────────────────────────
    market_state      = Column(String(15))
    volatility_regime = Column(String(15))

    # ── Output ─────────────────────────────────────────────────────────────
    quality_score  = Column(Float)    # composite 0–100
    quality_grade  = Column(String(2))  # A / B / C / D
    score_breakdown = Column(String(2000))  # JSON

    created_at = Column(DateTime, default=datetime.utcnow)

    instrument = relationship("Instrument")

    __table_args__ = (
        UniqueConstraint("instrument_id", "date", name="uq_setup_quality_inst_date"),
        Index("ix_setup_quality_score", "quality_score"),
        Index("ix_setup_quality_date",  "date"),
        Index("ix_setup_quality_symbol", "symbol"),
    )


# ── Breakout Quality Scores ───────────────────────────────────────────────────

class BreakoutQualityScore(Base):
    """
    Per-day breakout quality score (0–100) for each instrument.
    Gate: score < 60 → reject breakout signal.
    """
    __tablename__ = "breakout_quality_scores"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    instrument_id = Column(Integer, ForeignKey("instruments.id"), nullable=False)
    symbol        = Column(String(20), nullable=False)
    date          = Column(Date, nullable=False)

    breakout_quality_score  = Column(Float)
    volume_confirmation     = Column(Float)   # component 0-100
    momentum_continuation   = Column(Float)   # component 0-100
    candle_quality          = Column(Float)   # component 0-100
    retest_stability        = Column(Float)   # component 0-100

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("instrument_id", "date", name="uq_breakout_quality_inst_date"),
        Index("ix_bq_score",  "breakout_quality_score"),
        Index("ix_bq_symbol", "symbol"),
        Index("ix_bq_date",   "date"),
    )


# ── Liquidity Shelf Scores ────────────────────────────────────────────────────

class LiquidityShelfScore(Base):
    """
    Per-day liquidity shelf (absorption zone) score (0–100) per instrument.
    Gate: if breakout detected but score < 40 → penalise setup score.
    """
    __tablename__ = "liquidity_shelf_scores"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    instrument_id = Column(Integer, ForeignKey("instruments.id"), nullable=False)
    symbol        = Column(String(20), nullable=False)
    date          = Column(Date, nullable=False)

    liquidity_shelf_score    = Column(Float)
    shelf_level              = Column(Float)    # key price level detected
    shelf_type               = Column(String(15))  # support | resistance | unknown
    touch_count              = Column(Integer)
    touch_count_score        = Column(Float)    # component 0-100
    range_compression_score  = Column(Float)    # component 0-100
    volume_absorption_score  = Column(Float)    # component 0-100

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("instrument_id", "date", name="uq_liq_shelf_inst_date"),
        Index("ix_liq_shelf_score",  "liquidity_shelf_score"),
        Index("ix_liq_shelf_symbol", "symbol"),
        Index("ix_liq_shelf_date",   "date"),
    )


# ── Liquidity Sweep Scores ────────────────────────────────────────────────────

class LiquiditySweepScore(Base):
    """
    Per-day liquidity sweep (stop-hunt) score (0–100) per instrument.
    Gate: score ≥ 70 → flag as reversal / trap-trade candidate.

    A sweep is detected when price briefly breaks a key swing level,
    then reverses with a dominant wick within RETEST_BARS bars.
    """
    __tablename__ = "liquidity_sweep_scores"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    instrument_id = Column(Integer, ForeignKey("instruments.id"), nullable=False)
    symbol        = Column(String(20), nullable=False)
    date          = Column(Date, nullable=False)

    liquidity_sweep_score = Column(Float)
    sweep_type            = Column(String(15))   # high_sweep | low_sweep | none
    sweep_level           = Column(Float)        # key price level that was swept
    break_pct             = Column(Float)        # % price broke the level
    level_break_score     = Column(Float)        # component 0–100
    reversal_speed_score  = Column(Float)        # component 0–100
    wick_dominance_score  = Column(Float)        # component 0–100

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("instrument_id", "date", name="uq_liq_sweep_inst_date"),
        Index("ix_liq_sweep_score",  "liquidity_sweep_score"),
        Index("ix_liq_sweep_symbol", "symbol"),
        Index("ix_liq_sweep_date",   "date"),
    )


# ── Signal Reliability Scores ─────────────────────────────────────────────────

class SignalReliabilityScore(Base):
    """
    Per-setup-type signal reliability score computed over a rolling window of
    completed trades.

    Reliability score 0–100:
      ≥ 70  → healthy   — full position size
      50–69 → warning   — 50% position size
      < 50  → disabled  — strategy suspended

    One row per setup_type per day (upserted nightly by the pipeline).
    """
    __tablename__ = "signal_reliability_scores"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    setup_type  = Column(String(40), nullable=False)
    date        = Column(Date, nullable=False)

    reliability_score = Column(Float)
    status            = Column(String(15))   # healthy | warning | disabled | no_data

    # Core metrics
    win_rate    = Column(Float)
    avg_win     = Column(Float)
    avg_loss    = Column(Float)
    expectancy  = Column(Float)
    max_drawdown = Column(Float)
    trade_count  = Column(Integer, default=0)

    # Component scores (0–100 each)
    expectancy_score = Column(Float)
    win_rate_score   = Column(Float)
    stability_score  = Column(Float)
    drawdown_score   = Column(Float)

    computed_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("setup_type", "date", name="uq_signal_reliability_type_date"),
        Index("ix_sig_rel_score",      "reliability_score"),
        Index("ix_sig_rel_setup_type", "setup_type"),
        Index("ix_sig_rel_date",       "date"),
        Index("ix_sig_rel_status",     "status"),
    )


class PatternSignal(Base):
    """Pattern signal output used by chart overlays and signal pipeline."""
    __tablename__ = "pattern_signals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    pattern = Column(String(60), nullable=False)
    status = Column(String(20), nullable=False, default="FORMING")
    breakout_level = Column(Float)
    target = Column(Float)
    probability = Column(Float, nullable=False, default=0.0)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("ix_pattern_signals_symbol_timestamp", "symbol", "timestamp"),
        Index("ix_pattern_signals_status", "status"),
        Index("ix_pattern_signals_probability", "probability"),
    )


class PatternScanResult(Base):
    """
    Async pattern-ai scanner output for the latest universe scan.

    probability is a composite of:
      structure_quality, volume_confirmation, liquidity_alignment,
      market_regime, momentum.
    """
    __tablename__ = "pattern_scan_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    pattern = Column(String(50), nullable=False)
    probability = Column(Float, nullable=False)
    breakout_level = Column(Float, nullable=False)
    target = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("ix_pattern_scan_symbol_timestamp", "symbol", "timestamp"),
        Index("ix_pattern_scan_probability", "probability"),
    )


# ── Watchlist ──────────────────────────────────────────────────────────────────

class Watchlist(Base):
    """User watchlist — saved ticker symbols."""
    __tablename__ = "watchlist"

    id       = Column(Integer, primary_key=True, autoincrement=True)
    symbol   = Column(String(20), unique=True, nullable=False, index=True)
    added_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Watchlist(symbol='{self.symbol}')>"


# ── Advanced Cycle Projections ──────────────────────────────────────────────

class CycleProjection(Base):
    """Advanced cycle analysis: FFT, wavelet, Hilbert phase, and projected peaks/troughs."""
    __tablename__ = "cycle_projections"

    id = Column(Integer, primary_key=True, autoincrement=True)
    instrument_id = Column(Integer, ForeignKey("instruments.id"), nullable=False)
    date = Column(Date, nullable=False)

    # Dominant cycle parameters
    dominant_cycle_length = Column(Float)     # trading days
    cycle_strength = Column(Float)            # 0-1 normalized spectral power
    cycle_phase = Column(Float)               # 0-1 normalized phase position

    # Multi-method cycle detection
    fft_cycle_length = Column(Float)
    fft_strength = Column(Float)
    wavelet_cycle_length = Column(Float)
    wavelet_strength = Column(Float)
    hilbert_phase = Column(Float)             # instantaneous phase from Hilbert transform
    hilbert_amplitude = Column(Float)         # instantaneous amplitude

    # Projections
    next_peak_date = Column(Date)
    next_trough_date = Column(Date)
    next_peak_price = Column(Float)
    next_trough_price = Column(Float)

    # Confluence contribution
    cycle_alignment_score = Column(Float)     # 0-100

    created_at = Column(DateTime, default=datetime.utcnow)
    instrument = relationship("Instrument", back_populates="cycle_projections")

    __table_args__ = (
        UniqueConstraint("instrument_id", "date", name="uq_cycle_proj_inst_date"),
        Index("ix_cycle_proj_inst_date", "instrument_id", "date"),
    )


# ── Geometry Levels (Gann Geometry) ─────────────────────────────────────────

class GeometryLevel(Base):
    """Gann geometry: angles, fans, boxes, Square-of-9, Square-of-144, price-time symmetry."""
    __tablename__ = "geometry_levels"

    id = Column(Integer, primary_key=True, autoincrement=True)
    instrument_id = Column(Integer, ForeignKey("instruments.id"), nullable=False)
    date = Column(Date, nullable=False)

    # Gann angle analysis
    anchor_price = Column(Float)
    anchor_date = Column(Date)
    slope_1x1 = Column(Float)                # price-per-bar at 45°

    # Fan adherence (how well price follows fan lines)
    fan_adherence_score = Column(Float)       # 0-100
    trend_consistency = Column(Float)         # 0-1

    # Square-of-9 levels
    sq9_next_support = Column(Float)
    sq9_next_resistance = Column(Float)

    # Square-of-144 levels
    sq144_next_support = Column(Float)
    sq144_next_resistance = Column(Float)

    # Price-time symmetry
    price_time_symmetry_score = Column(Float)  # 0-100 (price_move ≈ time_move)
    symmetry_ratio = Column(Float)             # price_move / time_move (1.0 = perfect)

    # Composite
    geometry_score = Column(Float)             # 0-100

    created_at = Column(DateTime, default=datetime.utcnow)
    instrument = relationship("Instrument", back_populates="geometry_levels")

    __table_args__ = (
        UniqueConstraint("instrument_id", "date", name="uq_geometry_inst_date"),
        Index("ix_geometry_inst_date", "instrument_id", "date"),
    )


# ── Price-Time Confluence Nodes ─────────────────────────────────────────────

class ConfluenceNode(Base):
    """Price-time confluence decision nodes where multiple analysis converge."""
    __tablename__ = "confluence_nodes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    instrument_id = Column(Integer, ForeignKey("instruments.id"), nullable=False)
    date = Column(Date, nullable=False)       # date computed

    # Node location
    price_low = Column(Float)                 # lower bound of price range
    price_high = Column(Float)                # upper bound of price range
    time_window_start = Column(Date)
    time_window_end = Column(Date)

    # Confluence components
    confluence_score = Column(Float)           # 0-100
    component_scores = Column(String(2000))    # JSON: {cycle: 80, gann: 70, ...}

    # Supporting signals
    supporting_signals = Column(String(4000))  # JSON: list of signal descriptions
    node_type = Column(String(20))             # "support" | "resistance" | "reversal" | "breakout"
    direction = Column(String(10))             # "bullish" | "bearish" | "neutral"

    # Status
    status = Column(String(15))                # "upcoming" | "active" | "past" | "hit" | "missed"

    created_at = Column(DateTime, default=datetime.utcnow)
    instrument = relationship("Instrument", back_populates="confluence_nodes")

    __table_args__ = (
        Index("ix_conf_node_inst_date", "instrument_id", "date"),
        Index("ix_conf_node_status", "status"),
        Index("ix_conf_node_score", "confluence_score"),
    )


# ── Market Force Vectors ────────────────────────────────────────────────────

class MarketForce(Base):
    """Market physics: directional force vectors from multiple engines."""
    __tablename__ = "market_forces"

    id = Column(Integer, primary_key=True, autoincrement=True)
    instrument_id = Column(Integer, ForeignKey("instruments.id"), nullable=False)
    date = Column(Date, nullable=False)

    # Individual force vectors (-1 to +1, negative = bearish, positive = bullish)
    trend_force = Column(Float)
    liquidity_force = Column(Float)
    volatility_force = Column(Float)
    cycle_force = Column(Float)
    pattern_force = Column(Float)

    # Net force (weighted combination)
    net_force = Column(Float)                  # -1 to +1
    bias = Column(String(10))                  # "bullish" | "bearish" | "neutral"
    force_magnitude = Column(Float)            # abs(net_force), 0-1

    created_at = Column(DateTime, default=datetime.utcnow)
    instrument = relationship("Instrument", back_populates="market_forces")

    __table_args__ = (
        UniqueConstraint("instrument_id", "date", name="uq_market_force_inst_date"),
        Index("ix_market_force_inst_date", "instrument_id", "date"),
        Index("ix_market_force_bias", "bias"),
    )


# ── Strategy Genomes (Evolutionary) ────────────────────────────────────────

class StrategyGenome(Base):
    """Evolved strategy parameter sets from genetic algorithm optimization."""
    __tablename__ = "strategy_genomes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    generation = Column(Integer, nullable=False)
    genome_id = Column(String(40), nullable=False, unique=True)  # UUID

    # Strategy parameters (the genome)
    entry_confluence_min = Column(Float)        # min confluence to enter
    pattern_type = Column(String(40))           # pattern filter
    regime_filter = Column(String(15))          # COMPRESSION | TREND | EXPANSION | ALL
    stop_atr_mult = Column(Float)              # stop distance in ATR multiples
    target_atr_mult = Column(Float)            # target distance in ATR multiples
    hold_days_max = Column(Integer)            # max holding period

    # Fitness metrics (from backtesting)
    fitness = Column(Float)                    # composite fitness score
    sharpe_ratio = Column(Float)
    win_rate = Column(Float)
    profit_factor = Column(Float)
    max_drawdown = Column(Float)
    trade_count = Column(Integer)

    # Status
    is_active = Column(Boolean, default=False)  # currently deployed
    genome_data = Column(String(4000))          # full JSON genome

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_genome_generation", "generation"),
        Index("ix_genome_fitness", "fitness"),
        Index("ix_genome_active", "is_active"),
    )


# ── Strategy Health Monitoring ─────────────────────────────────────────────

class StrategyHealth(Base):
    """Strategy health monitoring — tracks performance drift."""
    __tablename__ = "strategy_health"

    id = Column(Integer, primary_key=True, autoincrement=True)
    setup_type = Column(String(50), nullable=False)
    date = Column(Date, nullable=False)

    state = Column(String(10))   # HEALTHY / WARNING / CRITICAL
    action = Column(String(20))  # maintain / reduce_size / disable

    win_rate = Column(Float)
    profit_factor = Column(Float)
    average_return = Column(Float)
    max_drawdown = Column(Float)
    trade_count = Column(Integer)

    win_rate_drift = Column(Float)
    pf_drift = Column(Float)
    return_drift = Column(Float)
    dd_drift = Column(Float)

    recent_metrics = Column(String(2000))   # JSON
    baseline_metrics = Column(String(2000)) # JSON

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("setup_type", "date", name="uq_strathlth_setup_date"),
        Index("ix_strathlth_setup_date", "setup_type", "date"),
        Index("ix_strathlth_state", "state"),
    )


# ── Price Distribution Forecasts ──────────────────────────────────────────

class PriceDistribution(Base):
    """Forward price distribution forecasts."""
    __tablename__ = "price_distributions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    instrument_id = Column(Integer, ForeignKey("instruments.id"), nullable=False)
    date = Column(Date, nullable=False)
    horizon_days = Column(Integer, default=20)

    expected_price = Column(Float)
    p10 = Column(Float)
    p25 = Column(Float)
    p50 = Column(Float)
    p75 = Column(Float)
    p90 = Column(Float)

    probability_up = Column(Float)
    probability_down = Column(Float)
    annualized_vol = Column(Float)
    daily_vol = Column(Float)

    created_at = Column(DateTime, default=datetime.utcnow)
    instrument = relationship("Instrument", back_populates="price_distributions")

    __table_args__ = (
        UniqueConstraint("instrument_id", "date", "horizon_days", name="uq_pricedist_inst_date_hz"),
        Index("ix_pricedist_inst_date", "instrument_id", "date"),
    )


# ── Signal Confidence Scores ─────────────────────────────────────────────────

class SignalConfidence(Base):
    """Signal confidence scores measuring trustworthiness."""
    __tablename__ = "signal_confidence"

    id = Column(Integer, primary_key=True, autoincrement=True)
    instrument_id = Column(Integer, ForeignKey("instruments.id"), nullable=False)
    date = Column(Date, nullable=False)

    confidence_score = Column(Float)  # 0-100
    confidence_tier = Column(String(20))  # Very Reliable / Reliable / Moderate / Weak

    historical_reliability = Column(Float)
    model_agreement = Column(Float)
    regime_match = Column(Float)
    feature_stability = Column(Float)
    confluence_density = Column(Float)

    created_at = Column(DateTime, default=datetime.utcnow)
    instrument = relationship("Instrument", back_populates="signal_confidences")

    __table_args__ = (
        UniqueConstraint("instrument_id", "date", name="uq_sigconf_inst_date"),
        Index("ix_sigconf_inst_date", "instrument_id", "date"),
    )
