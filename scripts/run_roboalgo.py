#!/usr/bin/env python3
"""
RoboAlgo - Main CLI Runner
Orchestrates the full pipeline:

  Phase 1 — Core data
    1. data         Download OHLCV price history
    2. indicators   Calculate technical indicators (MA, ATR, RSI, MACD, BB)
    3. volatility   Classify volatility regimes (ATR/BB percentiles)
    4. features     Generate ML feature vectors
    5. cycles       Run FFT + Hilbert cycle detection
    6. train        Train XGBoost probability model

  Phase 2 — Confluence stack
    7.  compression Detect range compression (BB_width_pct, ATR_pct)
    8.  breakout    Detect breakouts from compression ranges
    9.  liquidity   Map liquidity levels (prev H/L, VWAP, vol nodes)
    10. wyckoff     Classify Wyckoff phases (Accumulation/Markup/Distribution/Markdown)
    11. chart_patt  Detect chart patterns (flags, triangles, wedges, H&S)
    12. harmonics   Detect harmonic patterns (Gartley, Bat, Butterfly, Crab)
    13. market_st   Classify market state (COMPRESSION/TREND/EXPANSION/CHAOS)
    14. confluence  Compute weighted confluence scores and signal tiers

  Phase 3 — Signals & output
    15. signals     Generate 3-tier trading signals
    16. print       Print signal report to console

Usage:
    python scripts/run_roboalgo.py               # Run full pipeline
    python scripts/run_roboalgo.py --step data    # Run only data download
    python scripts/run_roboalgo.py --step confluence  # Run confluence scoring only
    python scripts/run_roboalgo.py --symbol TQQQ  # Run for a single symbol
    python scripts/run_roboalgo.py --phase2       # Run confluence stack only (steps 7–14)
"""

import argparse
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.connection import init_db, check_connection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("roboalgo")


def print_banner():
    print("\n" + "=" * 68)
    print("  RoboAlgo — Volatility Confluence Trading Intelligence System")
    print("=" * 68 + "\n")


# ── Phase 1: Core Data Pipeline ───────────────────────────────────────────────

def step_data(symbol: str | None = None):
    """Step 1: Download market data."""
    logger.info("STEP 1: Downloading market data...")
    from data_engine.downloader import MarketDataDownloader
    downloader = MarketDataDownloader()
    symbols = [symbol] if symbol else None
    result = downloader.download_all(symbols)
    logger.info(f"Data download: {result['success']} succeeded, {result['failed']} failed")


def step_indicators(symbol: str | None = None):
    """Step 2: Calculate technical indicators."""
    logger.info("STEP 2: Calculating indicators...")
    from indicator_engine.calculator import IndicatorCalculator
    calculator = IndicatorCalculator()
    calculator.compute_and_store(symbol)
    logger.info("Indicator calculation complete.")


def step_volatility(symbol: str | None = None):
    """Step 3: Classify volatility regimes (compression + expansion detection)."""
    logger.info("STEP 3: Classifying volatility regimes...")
    from volatility_engine.regime import VolatilityRegimeEngine
    engine = VolatilityRegimeEngine()
    count = engine.compute_and_store(symbol)
    logger.info(f"Volatility regime classification complete: {count} rows stored.")


def step_features(symbol: str | None = None):
    """Step 4: Generate ML feature vectors."""
    logger.info("STEP 4: Generating features...")
    from feature_engine.generator import FeatureGenerator
    generator = FeatureGenerator()
    generator.compute_and_store(symbol)
    logger.info("Feature generation complete.")


def step_cycles(symbol: str | None = None):
    """Step 5: Run cycle detection."""
    logger.info("STEP 5: Detecting market cycles...")
    from cycle_engine.detector import CycleDetector
    detector = CycleDetector()
    detector.compute_and_store(symbol)
    logger.info("Cycle detection complete.")


def step_train():
    """Step 6: Train the XGBoost probability model."""
    logger.info("STEP 6: Training probability model...")
    from probability_engine.classifier import ProbabilityClassifier
    classifier = ProbabilityClassifier()
    metrics = classifier.train()
    if "error" not in metrics:
        logger.info(
            f"Model trained: {metrics['samples']} samples, "
            f"AUC={metrics['cv_auc_mean']:.4f}, "
            f"positive_rate={metrics['positive_rate']:.3f}"
        )
    else:
        logger.error(f"Training failed: {metrics['error']}")


# ── Phase 2: Confluence Stack ─────────────────────────────────────────────────

def step_compression(symbol: str | None = None):
    """Step 7: Detect range compression."""
    logger.info("STEP 7: Computing range compression...")
    from range_engine.compression import RangeCompressionEngine
    engine = RangeCompressionEngine()
    count = engine.compute_and_store(symbol)
    logger.info(f"Range compression: {count} rows stored.")


def step_breakout(symbol: str | None = None):
    """Step 8: Detect breakouts from compression ranges."""
    logger.info("STEP 8: Detecting breakouts...")
    from range_engine.breakout import BreakoutEngine
    engine = BreakoutEngine()
    count = engine.compute_and_store(symbol)
    logger.info(f"Breakout detection: {count} rows stored.")


def step_liquidity(symbol: str | None = None):
    """Step 9: Map liquidity levels."""
    logger.info("STEP 9: Mapping liquidity levels...")
    from structure_engine.liquidity import LiquidityEngine
    engine = LiquidityEngine()
    count = engine.compute_and_store(symbol)
    logger.info(f"Liquidity mapping: {count} rows stored.")


def step_wyckoff(symbol: str | None = None):
    """Step 10: Classify Wyckoff phases."""
    logger.info("STEP 10: Classifying Wyckoff phases...")
    from structure_engine.wyckoff import WyckoffEngine
    engine = WyckoffEngine()
    count = engine.compute_and_store(symbol)
    logger.info(f"Wyckoff analysis: {count} rows stored.")


def step_chart_patterns(symbol: str | None = None):
    """Step 11: Detect chart patterns (flags, triangles, wedges, H&S)."""
    logger.info("STEP 11: Detecting chart patterns...")
    from pattern_engine.detection import ChartPatternEngine
    engine = ChartPatternEngine()
    count = engine.compute_and_store(symbol)
    logger.info(f"Chart pattern detection: {count} patterns stored.")


def step_harmonics(symbol: str | None = None):
    """Step 12: Detect harmonic patterns (Gartley, Bat, Butterfly, Crab)."""
    logger.info("STEP 12: Detecting harmonic patterns...")
    from pattern_engine.harmonics import HarmonicEngine
    engine = HarmonicEngine()
    count = engine.compute_and_store(symbol)
    logger.info(f"Harmonic pattern detection: {count} patterns stored.")


def step_market_state(symbol: str | None = None):
    """Step 13: Classify market state (COMPRESSION/TREND/EXPANSION/CHAOS)."""
    logger.info("STEP 13: Classifying market states...")
    from market_state_engine.state import MarketStateEngine
    engine = MarketStateEngine()
    count = engine.compute_and_store(symbol)
    logger.info(f"Market state classification: {count} rows stored.")


def step_confluence(symbol: str | None = None):
    """Step 14: Compute weighted confluence scores and signal tiers."""
    logger.info("STEP 14: Computing confluence scores...")
    from confluence_engine.score import ConfluenceEngine
    engine = ConfluenceEngine()
    count = engine.compute_and_store(symbol)
    logger.info(f"Confluence scoring: {count} symbols scored.")


# ── Phase 3: Signals ──────────────────────────────────────────────────────────

def step_candlestick_patterns(symbol: str | None = None):
    """Legacy candlestick pattern step (existing PatternDetector)."""
    logger.info("Detecting candlestick patterns (legacy)...")
    from pattern_engine.detector import PatternDetector
    detector = PatternDetector()
    count = detector.compute_and_store(symbol)
    logger.info(f"Candlestick pattern detection: {count} patterns stored.")


def step_signals(symbol: str | None = None):
    """Step 15: Generate 3-tier trading signals."""
    logger.info("STEP 15: Generating signals...")
    from signal_engine.generator import SignalGenerator
    generator = SignalGenerator()
    count = generator.generate_and_store(symbol)
    logger.info(f"Generated {count} signals.")


def print_signals():
    """Print latest confluence-scored signals grouped by tier."""
    from confluence_engine.score import ConfluenceEngine
    from signal_engine.generator import SignalGenerator

    engine  = ConfluenceEngine()
    results = engine.get_top_signals(limit=50)

    if not results:
        print("\nNo confluence signals found. Run the full pipeline first.\n")
        return

    def _print_tier(tier_name: str, tier_signals: list):
        if not tier_signals:
            return
        print(f"\n{'─'*90}")
        print(f"  {tier_name}  ({len(tier_signals)})")
        print(f"{'─'*90}")
        hdr = (
            f"  {'SYMBOL':<8} {'SCORE':>6}  {'TIER':<8}  {'STATE':<12}"
            f"  {'ENTRY':>7}  {'STOP':>7}  {'T1':>7}  {'T2':>7}  {'T3':>7}"
        )
        print(hdr)
        print(f"  {'─'*85}")
        for s in tier_signals:
            print(
                f"  {s.get('symbol',''):<8} {s.get('confluence_score',0):>5.1f}  "
                f"{s.get('signal_tier',''):<8}  {s.get('market_state',''):<12}  "
                f"${s.get('entry_price',0):>6.2f}  "
                f"${s.get('stop_price',0):>6.2f}  "
                f"${s.get('tier1_sell',0):>6.2f}  "
                f"${s.get('tier2_sell',0):>6.2f}  "
                f"${s.get('tier3_hold',0):>6.2f}"
            )

    by_tier = {"HIGH": [], "MEDIUM": [], "WATCH": []}
    for s in results:
        t = s.get("signal_tier", "WATCH")
        if t in by_tier:
            by_tier[t].append(s)

    print("\n" + "=" * 90)
    print("  RoboAlgo Confluence Signal Report — 3-Tier Trade Plans")
    print("=" * 90)
    _print_tier("🔥 HIGH CONFIDENCE  (≥80 confluence)", by_tier["HIGH"])
    _print_tier("⚡ MEDIUM CONFIDENCE  (60–79 confluence)", by_tier["MEDIUM"])
    _print_tier("👀 WATCH LIST  (50–59 confluence)", by_tier["WATCH"])
    total = sum(len(v) for v in by_tier.values())
    print(f"\n{'─'*90}")
    print(
        f"  Total: {total}  |  HIGH: {len(by_tier['HIGH'])}  "
        f"MEDIUM: {len(by_tier['MEDIUM'])}  WATCH: {len(by_tier['WATCH'])}"
    )
    print()


# ── Pipeline Orchestration ────────────────────────────────────────────────────

def run_phase1(symbol: str | None = None):
    """Phase 1: Core data pipeline (steps 1–6)."""
    step_data(symbol)
    step_indicators(symbol)
    step_volatility(symbol)
    step_features(symbol)
    step_cycles(symbol)
    step_train()


def run_phase2(symbol: str | None = None):
    """Phase 2: Confluence stack (steps 7–14)."""
    step_compression(symbol)
    step_breakout(symbol)
    step_liquidity(symbol)
    step_wyckoff(symbol)
    step_chart_patterns(symbol)
    step_harmonics(symbol)
    step_market_state(symbol)
    step_confluence(symbol)


def run_full_pipeline(symbol: str | None = None):
    """Execute the complete RoboAlgo pipeline (all 16 steps)."""
    run_phase1(symbol)
    run_phase2(symbol)
    step_signals(symbol)
    print_signals()


# ── CLI Entry Point ───────────────────────────────────────────────────────────

STEP_CHOICES = [
    "data", "indicators", "volatility", "features", "cycles", "train",
    "compression", "breakout", "liquidity", "wyckoff",
    "chart_patterns", "harmonics", "market_state", "confluence",
    "signals", "print",
]


def main():
    parser = argparse.ArgumentParser(
        description="RoboAlgo — Volatility Confluence Trading Intelligence System"
    )
    parser.add_argument(
        "--step",
        choices=STEP_CHOICES,
        help="Run a specific pipeline step",
    )
    parser.add_argument("--symbol", type=str, help="Process a single symbol")
    parser.add_argument("--phase1", action="store_true", help="Run Phase 1 only (data + indicators)")
    parser.add_argument("--phase2", action="store_true", help="Run Phase 2 only (confluence stack)")
    parser.add_argument("--init-db", action="store_true", help="Initialize database tables")
    args = parser.parse_args()

    print_banner()

    # Check database connection
    if not check_connection():
        logger.error("Cannot connect to PostgreSQL. Check DATABASE_URL in .env")
        sys.exit(1)
    logger.info("Database connection OK.")

    # Initialize database if requested or running full pipeline
    if args.init_db or (args.step is None and not args.phase1 and not args.phase2):
        init_db()

    sym = args.symbol

    # Dispatch
    if args.phase1:
        run_phase1(sym)
    elif args.phase2:
        run_phase2(sym)
    elif args.step is None:
        run_full_pipeline(sym)
    elif args.step == "data":
        step_data(sym)
    elif args.step == "indicators":
        step_indicators(sym)
    elif args.step == "volatility":
        step_volatility(sym)
    elif args.step == "features":
        step_features(sym)
    elif args.step == "cycles":
        step_cycles(sym)
    elif args.step == "train":
        step_train()
    elif args.step == "compression":
        step_compression(sym)
    elif args.step == "breakout":
        step_breakout(sym)
    elif args.step == "liquidity":
        step_liquidity(sym)
    elif args.step == "wyckoff":
        step_wyckoff(sym)
    elif args.step == "chart_patterns":
        step_chart_patterns(sym)
    elif args.step == "harmonics":
        step_harmonics(sym)
    elif args.step == "market_state":
        step_market_state(sym)
    elif args.step == "confluence":
        step_confluence(sym)
    elif args.step == "signals":
        step_signals(sym)
    elif args.step == "print":
        print_signals()


if __name__ == "__main__":
    main()
