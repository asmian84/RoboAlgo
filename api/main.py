"""
RoboAlgo - FastAPI Application
REST API serving market data, indicators, features, cycles, and 3TT signals.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import (
    instruments, prices, indicators, features, cycles, signals,
    analysis, patterns, quotes, mtf, recommendation, backtest,
    trendlines, alphavantage, paper_trading, volatility,
    confluence, compression, breakout, market_state,
    analytics, strategy, command_center,
    pipeline, trade_coach, evolution,
    price_levels, market_breadth, watchlist, fmp,
    market_force, confluence_nodes, strategies,
    geometry, price_distribution,
    signal_confidence, strategy_health,
    options_data,
    rocket_scanner, gamma_tracker, sniper_entry,
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


@app.get("/api/health")
def health_check():
    from database.connection import check_connection
    return {"status": "ok", "db_connected": check_connection()}
