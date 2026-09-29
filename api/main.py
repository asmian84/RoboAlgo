"""
RoboAlgo - FastAPI Application
REST API serving market data, indicators, features, cycles, and 3TT signals.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import (
    instruments, prices, indicators, features, cycles, signals,
    analysis, patterns, quotes, mtf, recommendation, backtest,
    trendlines, alphavantage, paper_trading, trades, volatility,
    confluence, compression, breakout, market_state,
    analytics, strategy, command_center,
    pipeline, trade_coach, evolution,
    price_levels, market_breadth, watchlist, fmp,
    market_force, confluence_nodes, strategies,
    geometry, price_distribution,
    signal_confidence, strategy_health,
    options_data,
    rocket_scanner, gamma_tracker, sniper_entry, sniper_gamma, penny_sniper,
    quant_scanner,
    trade_signals,
    crypto_signals,
    lab,
    trading_control,
    settings,
    finder,
    paper_account,
)

app = FastAPI(title="RoboAlgo API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://roboalgo.app",
        "https://www.roboalgo.app",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(instruments.router, prefix="/api/instruments", tags=["instruments"])
app.include_router(prices.router, prefix="/api", tags=["prices"])
app.include_router(indicators.router, prefix="/api/indicators", tags=["indicators"])
app.include_router(features.router, prefix="/api/features", tags=["features"])
app.include_router(cycles.router, prefix="/api/cycles", tags=["cycles"])
app.include_router(signals.router, prefix="/api/signals", tags=["signals"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["analysis"])
app.include_router(patterns.router, prefix="/api/patterns", tags=["patterns"])
app.include_router(quotes.router, prefix="/api", tags=["quotes"])
app.include_router(mtf.router, prefix="/api/mtf", tags=["mtf"])
app.include_router(recommendation.router, prefix="/api/recommendation", tags=["recommendation"])
app.include_router(backtest.router, prefix="/api/backtest", tags=["backtest"])
app.include_router(trendlines.router, prefix="/api/trendlines", tags=["trendlines"])
app.include_router(alphavantage.router, prefix="/api/av", tags=["alphavantage"])
app.include_router(paper_trading.router, prefix="/api/paper", tags=["paper_trading"])
app.include_router(trades.router, prefix="/api/trades", tags=["trades"])
app.include_router(volatility.router,    prefix="/api/volatility",    tags=["volatility"])
app.include_router(confluence.router,   prefix="/api/confluence",    tags=["confluence"])
app.include_router(compression.router,  prefix="/api/compression",   tags=["compression"])
app.include_router(breakout.router,     prefix="/api/breakout",      tags=["breakout"])
app.include_router(market_state.router, prefix="/api/market-state",  tags=["market_state"])
app.include_router(analytics.router,       prefix="/api/analytics",      tags=["analytics"])
app.include_router(strategy.router,        prefix="/api/strategy",       tags=["strategy"])
app.include_router(command_center.router,  prefix="/api/command-center", tags=["command_center"])
app.include_router(pipeline.router,    prefix="/api/pipeline",    tags=["pipeline"])
app.include_router(trade_coach.router, prefix="/api/trade-coach", tags=["trade_coach"])
app.include_router(evolution.router,    prefix="/api/evolution",     tags=["evolution"])
app.include_router(price_levels.router,    prefix="/api/price-levels", tags=["price_levels"])
app.include_router(market_breadth.router,  prefix="/api/market",       tags=["market_breadth"])
app.include_router(watchlist.router,       prefix="/api/watchlist",    tags=["watchlist"])
app.include_router(fmp.router,             prefix="/api/fmp",          tags=["fmp"])
app.include_router(market_force.router,    prefix="/api/force",        tags=["market_force"])
app.include_router(confluence_nodes.router, prefix="/api/confluence-nodes", tags=["confluence_nodes"])
app.include_router(strategies.router,      prefix="/api/strategies",   tags=["strategies"])
app.include_router(geometry.router,        prefix="/api/geometry",     tags=["geometry"])
app.include_router(price_distribution.router, prefix="/api/distribution", tags=["price_distribution"])
app.include_router(signal_confidence.router, prefix="/api/signal-confidence", tags=["signal_confidence"])
app.include_router(strategy_health.router,   prefix="/api/strategy-health",   tags=["strategy_health"])
app.include_router(options_data.router,      prefix="/api/options",           tags=["options_data"])
app.include_router(rocket_scanner.router,    prefix="/api/rocket",            tags=["rocket_scanner"])
app.include_router(gamma_tracker.router,     prefix="/api/gamma-tracker",     tags=["gamma_tracker"])
app.include_router(sniper_entry.router,      prefix="/api/sniper-entry",      tags=["sniper_entry"])
app.include_router(sniper_gamma.router,      prefix="/api/sniper-gamma",      tags=["sniper_gamma"])
app.include_router(penny_sniper.router,      prefix="/api/penny-sniper",      tags=["penny_sniper"])
app.include_router(quant_scanner.router,     prefix="/api/quant",             tags=["quant_scanner"])
app.include_router(trade_signals.router,     prefix="/api/trade-signals",     tags=["trade_signals"])
app.include_router(crypto_signals.router,    prefix="/api/crypto-signals",    tags=["crypto_signals"])
app.include_router(lab.router,               prefix="/api/lab",               tags=["lab"])
app.include_router(finder.router,            prefix="/api/finder",            tags=["finder"])
app.include_router(trading_control.router, prefix="/api/auto-trader", tags=["auto_trader"])
app.include_router(settings.router,          prefix="/api/settings",          tags=["settings"])
app.include_router(paper_account.router,     prefix="/api/paper-account",     tags=["paper_account"])


@app.get("/api/health")
def health_check():
    from database.connection import check_connection
    return {"status": "ok", "db_connected": check_connection()}


# ── Automatic Pipeline Scheduler ─────────────────────────────────────────────
# Run the data pipeline every 5 minutes in the background to keep market_state
# and other derived data consistently fresh, without blocking the API.

import logging
import os
import threading
import time
from datetime import datetime

logger = logging.getLogger(__name__)
_scheduler_thread = None
_scheduler_running = False


def _pipeline_scheduler():
    """Background thread that runs the full pipeline every 5 minutes."""
    global _scheduler_running
    import requests

    # Wait 30 seconds before first run to let the server fully start
    time.sleep(30)

    while _scheduler_running:
        try:
            logger.info("Pipeline: automatic scheduled run started")
            # Hit the pipeline endpoint to trigger the run
            response = requests.post("http://localhost:8000/api/pipeline/run", timeout=10)
            if response.status_code == 200:
                logger.info("Pipeline: scheduled run accepted")
            else:
                logger.warning(f"Pipeline: scheduled run failed with status {response.status_code}")
        except Exception as e:
            logger.error(f"Pipeline: scheduled run error: {e}")

        # Sleep for 5 minutes before next run
        time.sleep(300)


@app.on_event("startup")
async def startup_scheduler():
    """Optionally start the in-process pipeline scheduler.

    OFF by default, and that default is deliberate. This thread POSTs to
    /api/pipeline/run every five minutes, which regenerates features for ~239
    instruments inside this process. Uvicorn runs one worker here, so for
    roughly two minutes out of every five the API answered nothing at all —
    every page in the app intermittently failed with a network error, which
    looked like a frontend bug and was not.

    Scheduling belongs in Celery beat (infrastructure/scheduler.py), which
    already owns the pipeline stages and runs them in a worker process where a
    two-minute job blocks nothing. Set ROBOALGO_INPROCESS_SCHEDULER=1 only if
    you are deliberately running without a Celery worker and accept that the
    API will stall periodically.
    """
    global _scheduler_thread, _scheduler_running

    if os.getenv("ROBOALGO_INPROCESS_SCHEDULER") != "1":
        logger.info(
            "In-process pipeline scheduler disabled (default). Celery beat owns "
            "scheduling; set ROBOALGO_INPROCESS_SCHEDULER=1 to re-enable."
        )
        return

    _scheduler_running = True
    _scheduler_thread = threading.Thread(target=_pipeline_scheduler, daemon=True)
    _scheduler_thread.start()
    logger.warning(
        "In-process pipeline scheduler ENABLED - this API will stall for ~2 "
        "minutes out of every 5 while the pipeline runs."
    )


@app.on_event("shutdown")
async def shutdown_scheduler():
    """Stop the background pipeline scheduler on server shutdown."""
    global _scheduler_running
    _scheduler_running = False
    logger.info("Pipeline scheduler stopped")
