"""
RoboAlgo — Command Center API Router
Single aggregated endpoint providing the full operational dashboard state.

GET /api/command-center
  Returns: market_state_summary, opportunity_map, active_trades,
           portfolio_risk, system_health

30-second in-memory cache to avoid hammering all engines on every poll.
"""

import logging
from datetime import datetime, timedelta
from typing import Any

from fastapi import APIRouter

logger = logging.getLogger(__name__)
router = APIRouter()

# ── In-memory cache ────────────────────────────────────────────────────────────
_CACHE_TTL_SECONDS = 30
_cache: dict[str, Any] = {"data": None, "expires_at": datetime.min}


def _is_cache_valid() -> bool:
    return _cache["data"] is not None and datetime.utcnow() < _cache["expires_at"]


def _set_cache(data: dict) -> None:
    _cache["data"] = data
    _cache["expires_at"] = datetime.utcnow() + timedelta(seconds=_CACHE_TTL_SECONDS)


# ── Main endpoint ──────────────────────────────────────────────────────────────

@router.get("")
def get_command_center():
    """
    Aggregate and return the full Command Center state.
    Cached for 30 seconds to reduce engine load on frequent polling.
    """
    if _is_cache_valid():
        return _cache["data"]

    result = {
        "active_strategy":      _get_active_strategy(),
        "market_state_summary": _get_market_state_summary(),
        "opportunity_radar":    _get_opportunity_radar(),
        "opportunity_map":      _get_opportunity_map(),
        "active_trades":        _get_active_trades(),
        "portfolio_risk":       _get_portfolio_risk(),
        "system_health":        _get_system_health(),
        "signal_reliability":   _get_signal_reliability(),
        "market_safety":        _get_market_safety(),
        "generated_at":         datetime.utcnow().isoformat() + "Z",
    }

    _set_cache(result)
    return result


@router.get("/refresh")
def force_refresh():
    """Force-clear the cache and return fresh data immediately."""
    _cache["data"] = None
    _cache["expires_at"] = datetime.min
    return get_command_center()


# ── Panel Data Collectors ──────────────────────────────────────────────────────

def _get_active_strategy() -> dict:
    """
    Determine the dominant market regime across all instruments and return
    the active strategy mode from the Regime Playbook.

    Dominant regime = the state with the most instruments (EXPANSION beats TREND
    on tie, per the priority ordering used across the platform).
    """
    STATE_PRIORITY = {"EXPANSION": 0, "TREND": 1, "COMPRESSION": 2, "CHAOS": 3}

    try:
        from market_state_engine.state import MarketStateEngine
        from strategy_engine.regime_playbook import get_rule, get_active_strategy, get_playbook_summary

        engine = MarketStateEngine()
        counts: dict = engine.get_state_summary()  # {state: count}

        if not counts:
            dominant = "COMPRESSION"
        else:
            # Sort by count descending, break ties by STATE_PRIORITY
            dominant = max(
                counts.keys(),
                key=lambda s: (counts[s], -STATE_PRIORITY.get(s, 99)),
            )

        rule   = get_rule(dominant)
        strat  = get_active_strategy(dominant)

        return {
            "dominant_regime":     dominant,
            "regime_counts":       counts,
            "strategy_type":       rule.strategy_type,
            "strategy_key":        strat["strategy"],
            "position_multiplier": rule.position_multiplier,
            "risk_per_trade":      rule.risk_per_trade,
            "max_positions":       rule.max_positions,
            "risk_mode":           strat["risk_mode"],
            "color":               rule.color,
            "entry_description":   rule.entry_description,
            "size_description":    rule.size_description,
            "playbook":            get_playbook_summary(),
        }

    except Exception as e:
        logger.warning(f"Command center: active_strategy failed: {e}")
        return {
            "dominant_regime":     "UNKNOWN",
            "regime_counts":       {},
            "strategy_type":       "Unknown",
            "strategy_key":        "watchlist",
            "position_multiplier": 0.0,
            "risk_per_trade":      0.01,
            "max_positions":       3,
            "risk_mode":           "reduced",
            "color":               "#6b7280",
            "entry_description":   "Unable to determine active strategy.",
            "size_description":    "Reduced size until regime is resolved.",
            "playbook":            [],
            "error":               str(e),
        }


def _get_market_state_summary() -> dict:
    """
    Aggregate market state data for all instruments.
    Returns state counts + per-symbol detail sorted by state priority.
    """
    STATE_PRIORITY = {"EXPANSION": 0, "TREND": 1, "COMPRESSION": 2, "CHAOS": 3}

    try:
        from market_state_engine.state import MarketStateEngine
        from database.connection import get_session
        from database.models import Instrument, MarketState
        from sqlalchemy import select, desc

        engine = MarketStateEngine()
        counts = engine.get_state_summary()

        # Get latest state per instrument
        instruments_data = []
        with get_session() as session:
            instruments = session.execute(select(Instrument)).scalars().all()
            for inst in instruments:
                row = session.execute(
                    select(MarketState)
                    .where(MarketState.instrument_id == inst.id)
                    .order_by(desc(MarketState.date))
                    .limit(1)
                ).scalar_one_or_none()

                if row:
                    instruments_data.append({
                        "symbol":                inst.symbol,
                        "state":                 row.state,
                        "volatility_percentile": row.volatility_percentile,
                        "trend_strength":        row.trend_strength,
                        "expansion_strength":    row.expansion_strength,
                        "ma_alignment":          row.ma_alignment,
                        "adx":                   row.adx,
                        "volume_ratio":          row.volume_ratio,
                        "size_multiplier":       row.size_multiplier,
                    })

        # Sort: EXPANSION first, then TREND, then COMPRESSION, then CHAOS
        instruments_data.sort(
            key=lambda x: (STATE_PRIORITY.get(x["state"], 99), -(x["trend_strength"] or 0))
        )

        return {
            "counts":      counts,
            "instruments": instruments_data,
        }

    except Exception as e:
        logger.warning(f"Command center: market_state_summary failed: {e}")
        return {"counts": {}, "instruments": [], "error": str(e)}


def _get_opportunity_map() -> dict:
    """
    Return top confluence-scored signals as the opportunity map.
    Includes expected move, compression/breakout state, and trade entry levels.
    """
    try:
        from confluence_engine.score import ConfluenceEngine

        engine  = ConfluenceEngine()
        signals = engine.get_top_signals(min_tier="WATCH", limit=25)

        # Pre-load quality scores for enrichment
        from analytics_engine.setup_quality import SetupQualityScorer
        from analytics_engine.breakout_quality import BreakoutQualityEngine
        from structure_engine.liquidity_shelf import LiquidityShelfEngine
        from structure_engine.liquidity_sweep import LiquiditySweepEngine
        from structure_engine.liquidity_map import LiquidityMapEngine

        quality_scorer  = SetupQualityScorer()
        breakout_engine = BreakoutQualityEngine()
        shelf_engine    = LiquidityShelfEngine()
        sweep_engine    = LiquiditySweepEngine()
        lmap_engine     = LiquidityMapEngine()

        from portfolio_engine.position_scaler import PositionScalingEngine
        scaler = PositionScalingEngine()

        # Normalize signal fields for frontend
        normalized = []
        for s in signals:
            sym = s.get("symbol", "")

            # Live quality scores (with graceful fallback)
            try:
                q_result = quality_scorer.score_symbol(sym)
                setup_quality_score = q_result.get("quality_score")
                quality_grade       = q_result.get("quality_grade")
            except Exception:
                setup_quality_score = None
                quality_grade       = None

            try:
                bq_result = breakout_engine.calculate_breakout_quality(sym)
                breakout_quality_score = bq_result.get("breakout_quality_score")
                breakout_gate_passed   = bq_result.get("gate_passed", False)
            except Exception:
                breakout_quality_score = None
                breakout_gate_passed   = None

            try:
                shelf_result = shelf_engine.detect_liquidity_shelf(sym)
                shelf_score  = shelf_result.get("liquidity_shelf_score")
            except Exception:
                shelf_score = None

            try:
                sweep_result = sweep_engine.detect_liquidity_sweep(sym)
                sweep_score  = sweep_result.get("liquidity_sweep_score")
                sweep_type   = sweep_result.get("sweep_type", "none")
                sweep_gate   = sweep_result.get("gate_passed", False)
            except Exception:
                sweep_score = None
                sweep_type  = None
                sweep_gate  = None

            try:
                lmap_result      = lmap_engine.build_liquidity_map(sym)
                nearest_above    = lmap_result.get("nearest_above")
                nearest_below    = lmap_result.get("nearest_below")
                # Alignment: default to long (breakout direction)
                lmap_alignment   = lmap_engine.get_alignment_score(sym, direction="long")
            except Exception:
                nearest_above  = None
                nearest_below  = None
                lmap_alignment = None

            try:
                size_result        = scaler.calculate_position_size(
                    symbol              = sym,
                    setup_quality_score = float(setup_quality_score or 50.0),
                    reliability_score   = None,   # neutral; real gate via regime_playbook
                    regime              = s.get("market_state", "TREND"),
                )
                position_size_mult = size_result.get("position_size_multiplier")
                risk_per_trade_val = size_result.get("risk_per_trade")
                size_approved      = size_result.get("approved", True)
            except Exception:
                position_size_mult = None
                risk_per_trade_val = None
                size_approved      = None

            normalized.append({
                "symbol":                  sym,
                "confluence_score":        s.get("confluence_score", 0),
                "signal_tier":             s.get("signal_tier", "NONE"),
                "setup_quality_score":     setup_quality_score,
                "quality_grade":           quality_grade,
                "breakout_quality_score":  breakout_quality_score,
                "breakout_gate_passed":    breakout_gate_passed,
                "liquidity_shelf_score":   shelf_score,
                "liquidity_sweep_score":   sweep_score,
                "sweep_type":              sweep_type,
                "sweep_gate_passed":       sweep_gate,
                "liquidity_alignment":     lmap_alignment,
                "nearest_above":           nearest_above,
                "nearest_below":           nearest_below,
                "expected_move_pct":       s.get("expected_move_pct"),
                "expected_move_display":   s.get("expected_move_display", ""),
                "is_compression":          s.get("is_compression", False),
                "is_breakout":             s.get("is_breakout", False),
                "entry_price":             s.get("entry_price"),
                "stop_price":              s.get("stop_price"),
                "target_price":            s.get("target_price"),
                "market_state":            s.get("market_state", ""),
                "volatility_regime":       s.get("volatility_regime", ""),
                "component_scores":        s.get("component_scores", {}),
                "decision_trace":           s.get("decision_trace"),
                "position_size_multiplier": position_size_mult,
                "risk_per_trade":           risk_per_trade_val,
                "size_approved":            size_approved,
            })

        # Sort by setup quality score (composite) descending; fall back to confluence
        normalized.sort(
            key=lambda x: (x["setup_quality_score"] or 0, x["confluence_score"]),
            reverse=True,
        )

        return {"signals": normalized}

    except Exception as e:
        logger.warning(f"Command center: opportunity_map failed: {e}")
        return {"signals": [], "error": str(e)}


def _get_active_trades() -> dict:
    """
    Return all open lifecycle trades (SETUP/TRIGGER/ENTRY/ACTIVE).
    """
    try:
        from trade_engine.lifecycle import TradeLifecycleEngine

        engine = TradeLifecycleEngine()
        trades = engine.get_active_trades()

        return {
            "trades": trades,
            "count":  len(trades),
        }

    except Exception as e:
        logger.warning(f"Command center: active_trades failed: {e}")
        return {"trades": [], "count": 0, "error": str(e)}


def _get_portfolio_risk() -> dict:
    """
    Return portfolio summary: equity, positions, exposure, daily P&L.
    """
    try:
        from portfolio_engine.manager import PortfolioManager

        manager = PortfolioManager()
        summary = manager.get_portfolio_summary()
        return summary

    except Exception as e:
        logger.warning(f"Command center: portfolio_risk failed: {e}")
        return {
            "account_equity":       100_000.0,
            "open_positions":       0,
            "max_positions":        5,
            "daily_pnl_pct":        0.0,
            "daily_loss_limit":     0.05,
            "sector_exposure":      {},
            "slots_available":      5,
            "risk_budget_remaining": 0.05,
            "error":                str(e),
        }


def _get_system_health() -> dict:
    """
    Return system health: data quality score, last update time, pipeline status.
    Pipeline status:
      OK    — last data update within 26 hours
      STALE — 26–72 hours ago
      ERROR — older than 72 hours or no data
    """
    try:
        from data_engine.validation import DataValidator
        from database.connection import get_session
        from database.models import PriceData, Instrument
        from sqlalchemy import select, func

        # Data quality summary (fast — from cache if possible)
        validator = DataValidator()
        quality   = validator.get_quality_summary()

        # Last data update
        last_update = None
        pipeline_status = "ERROR"

        with get_session() as session:
            instr_count = session.execute(
                select(func.count()).select_from(Instrument)
            ).scalar() or 0

            max_date = session.execute(
                select(func.max(PriceData.date))
            ).scalar()

        if max_date:
            last_update_str = str(max_date)
            # Parse date to compare with now
            if hasattr(max_date, 'strftime'):
                last_dt = datetime.combine(max_date, datetime.min.time())
            else:
                last_dt = datetime.fromisoformat(str(max_date))

            age_hours = (datetime.utcnow() - last_dt).total_seconds() / 3600

            if age_hours <= 26:
                pipeline_status = "OK"
            elif age_hours <= 72:
                pipeline_status = "STALE"
            else:
                pipeline_status = "ERROR"
        else:
            last_update_str = None

        return {
            "data_quality_score": quality.get("avg_quality", 0.0),
            "last_data_update":   last_update_str,
            "pipeline_status":    pipeline_status,
            "total_instruments":  instr_count,
            "data_issues": {
                "total_critical": quality.get("total_critical", 0),
                "below_80":       quality.get("below_80", 0),
            },
            "worst_symbols": quality.get("worst_symbols", []),
        }

    except Exception as e:
        logger.warning(f"Command center: system_health failed: {e}")
        return {
            "data_quality_score": 0.0,
            "last_data_update":   None,
            "pipeline_status":    "ERROR",
            "total_instruments":  0,
            "data_issues":        {"total_critical": 0, "below_80": 0},
            "worst_symbols":      [],
            "error":              str(e),
        }


# ── Pattern → setup-type mapping for proxy win-rate computation ────────────────
_PATTERN_SETUP_MAP: dict[str, list[str]] = {
    "compression_breakout": [
        "Symmetrical Triangle", "Descending Triangle", "Rectangle",
        "Cup & Handle", "Falling Wedge", "Rising Wedge", "Megaphone",
        "Bull Flag", "Bear Flag", "Bullish Pennant",
    ],
    "trend_pullback": [
        "Ascending Channel", "Descending Channel",
        "Double Bottom", "Triple Bottom",
    ],
    "liquidity_sweep": [
        "Gartley", "Bat", "Butterfly", "Crab", "Cypher",
    ],
    "pattern_reversal": [
        "Head & Shoulders", "Inv. Head & Shoulders",
        "Double Top", "Triple Top",
        "Rounding Bottom", "Rounding Top",
    ],
    "wyckoff_spring": [
        "Wyckoff Accumulation", "Wyckoff Distribution",
    ],
}

_WIN_STATUSES  = ("BREAKOUT", "COMPLETED")
_LOSS_STATUSES = ("FAILED",)


def _compute_proxy_reliability(setup_type: str) -> dict | None:
    """
    Compute a proxy reliability score from *pattern_signals* outcomes.

    Used when trade_lifecycle has no completed trades yet.  Returns a dict
    with the same schema as SignalReliabilityEngine.compute_reliability(),
    or None if fewer than 5 resolved signals exist for the setup type.
    """
    import math
    from database.connection import get_session
    from sqlalchemy import text
    from analytics_engine.signal_reliability import (
        _linear_score, _reliability_status, _position_multiplier,
        WEIGHTS, SETUP_TYPE_LABELS,
        EXPECTANCY_FLOOR, EXPECTANCY_CAP, STD_DEV_CAP, DRAWDOWN_CAP,
    )

    pattern_names = _PATTERN_SETUP_MAP.get(setup_type, [])
    if not pattern_names:
        return None

    placeholders = ", ".join(f":p{i}" for i in range(len(pattern_names)))
    bind = {f"p{i}": n for i, n in enumerate(pattern_names)}

    with get_session() as session:
        rows = session.execute(
            text(f"""
                SELECT status, COUNT(*) AS cnt
                FROM pattern_signals
                WHERE pattern IN ({placeholders})
                GROUP BY status
            """),
            bind,
        ).fetchall()

    status_counts: dict[str, int] = {row.status: row.cnt for row in rows}

    wins   = sum(status_counts.get(s, 0) for s in _WIN_STATUSES)
    losses = sum(status_counts.get(s, 0) for s in _LOSS_STATUSES)
    total  = wins + losses

    if total < 5:
        return None

    # Bayesian prior: 6 virtual wins + 4 virtual losses keeps the prior at 60%
    adj_win_rate = (wins + 6) / (total + 10)

    # Proxy metrics with a fixed 2:1 reward-risk ratio (8% avg win / 4% avg loss)
    avg_win_pct  = 0.08
    avg_loss_pct = 0.04
    expectancy   = adj_win_rate * avg_win_pct - (1 - adj_win_rate) * avg_loss_pct

    # Stability proxy: variance of a mixed Bernoulli outcome distribution
    variance = (
        adj_win_rate * avg_win_pct ** 2
        + (1 - adj_win_rate) * avg_loss_pct ** 2
        - expectancy ** 2
    )
    std_dev = math.sqrt(max(variance, 0.0))

    # Drawdown proxy: scales with failure rate
    max_dd = max(0.0, (1 - adj_win_rate) * 0.25)

    # Score components (same formula as SignalReliabilityEngine)
    expectancy_score = _linear_score(expectancy, EXPECTANCY_FLOOR, EXPECTANCY_CAP)
    win_rate_score   = adj_win_rate * 100.0
    stability_score  = max(0.0, (1.0 - std_dev / max(STD_DEV_CAP, 1e-9))) * 100.0
    stability_score  = min(stability_score, 100.0)
    drawdown_score   = max(0.0, (1.0 - max_dd / max(DRAWDOWN_CAP, 1e-9))) * 100.0
    drawdown_score   = min(drawdown_score, 100.0)

    component_scores = {
        "expectancy": round(expectancy_score, 1),
        "win_rate":   round(win_rate_score,   1),
        "stability":  round(stability_score,  1),
        "drawdown":   round(drawdown_score,   1),
    }
    reliability_score = round(
        sum(component_scores[k] * WEIGHTS[k] for k in WEIGHTS), 1
    )

    return {
        "setup_type":          setup_type,
        "strategy_label":      SETUP_TYPE_LABELS.get(setup_type, setup_type),
        "reliability_score":   reliability_score,
        "status":              _reliability_status(reliability_score),
        "position_multiplier": _position_multiplier(reliability_score),
        "metrics": {
            "win_rate":    round(adj_win_rate, 4),
            "avg_win":     round(avg_win_pct,  4),
            "avg_loss":    round(avg_loss_pct, 4),
            "expectancy":  round(expectancy,   4),
            "stability":   round(1.0 - std_dev / max(STD_DEV_CAP, 1e-9), 4),
            "max_drawdown":round(max_dd, 4),
            "trade_count": total,
        },
        "proxy": True,   # flag so UI can show "estimated from pattern signals"
    }


def _get_signal_reliability() -> dict:
    """
    Return signal reliability scores for all known setup types.

    Status hierarchy:
      healthy  (≥ 70) — normal trading, full position size
      warning  (50–69) — reduced position size (50%)
      disabled (< 50)  — strategy suspended until edge recovers
      no_data          — insufficient trade history, allow at full size
    """
    # All canonical setup types from the regime playbook
    KNOWN_SETUPS = [
        "compression_breakout",
        "trend_pullback",
        "liquidity_sweep",
        "pattern_reversal",
        "wyckoff_spring",
    ]

    try:
        from analytics_engine.signal_reliability import SignalReliabilityEngine, SETUP_TYPE_LABELS

        engine = SignalReliabilityEngine()

        # Try to get live data from completed trades; fall back to pattern proxy
        strategies = []
        for setup_type in KNOWN_SETUPS:
            try:
                result = engine.get_reliability_status(setup_type)
                # If trade_lifecycle has no data, try the pattern-signal proxy
                if result.get("status") == "no_data":
                    proxy = _compute_proxy_reliability(setup_type)
                    if proxy is not None:
                        result = proxy
                strategies.append({
                    "setup_type":          setup_type,
                    "strategy_label":      SETUP_TYPE_LABELS.get(setup_type, setup_type),
                    "reliability_score":   result.get("reliability_score"),
                    "status":              result.get("status", "no_data"),
                    "position_multiplier": result.get("position_multiplier", 1.0),
                    "win_rate":            result.get("metrics", {}).get("win_rate"),
                    "expectancy":          result.get("metrics", {}).get("expectancy"),
                    "max_drawdown":        result.get("metrics", {}).get("max_drawdown"),
                    "trade_count":         result.get("metrics", {}).get("trade_count", 0),
                    "proxy":               result.get("proxy", False),
                })
            except Exception:
                strategies.append({
                    "setup_type":          setup_type,
                    "strategy_label":      SETUP_TYPE_LABELS.get(setup_type, setup_type),
                    "reliability_score":   None,
                    "status":              "no_data",
                    "position_multiplier": 1.0,
                    "win_rate":            None,
                    "expectancy":          None,
                    "max_drawdown":        None,
                    "trade_count":         0,
                    "proxy":               False,
                })

        # Sort: disabled first (most urgent), then by score ascending within status
        STATUS_PRIORITY = {"disabled": 0, "warning": 1, "healthy": 2, "no_data": 3}
        strategies.sort(key=lambda x: (
            STATUS_PRIORITY.get(x["status"], 3),
            x["reliability_score"] or 999,
        ))

        # Overall system reliability: lowest non-null score, or None
        scored = [s for s in strategies if s["reliability_score"] is not None]
        system_reliability = round(min(s["reliability_score"] for s in scored), 1) if scored else None
        disabled_count = sum(1 for s in strategies if s["status"] == "disabled")
        warning_count  = sum(1 for s in strategies if s["status"] == "warning")

        return {
            "strategies":         strategies,
            "system_reliability": system_reliability,
            "disabled_count":     disabled_count,
            "warning_count":      warning_count,
        }

    except Exception as e:
        logger.warning(f"Command center: signal_reliability failed: {e}")
        return {
            "strategies":         [],
            "system_reliability": None,
            "disabled_count":     0,
            "warning_count":      0,
            "error":              str(e),
        }


def _get_opportunity_radar() -> dict:
    """
    Scan all instruments for early-stage opportunity scores.

    Identifies symbols with active compression + institutional shelf absorption
    — the optimal watch window before a breakout occurs.

    Returns instruments ranked by opportunity_score descending, plus a count
    of symbols that meet the early-stage criteria (compression ≥ 50, shelf ≥ 35,
    state in COMPRESSION or TREND).
    """
    try:
        from analytics_engine.opportunity_radar import OpportunityRadarEngine

        engine  = OpportunityRadarEngine()
        results = engine.scan_all(limit=25, early_only=False)

        early_count = sum(1 for r in results if r.get("is_early_stage"))

        return {
            "instruments":       results,
            "early_stage_count": early_count,
        }

    except Exception as e:
        logger.warning(f"Command center: opportunity_radar failed: {e}")
        return {
            "instruments":       [],
            "early_stage_count": 0,
            "error":             str(e),
        }


# ── Hedge instrument definitions ──────────────────────────────────────────────
# Suggested when a volatility spike trigger is detected.
# Ordered: most direct hedge first within each category.
_VOLATILITY_SPIKE_HEDGES = [
    # Volatility surge instruments — profit when VIX spikes
    {
        "symbol":      "UVXY",
        "label":       "Ultra VIX Short-Term Futures ETF",
        "category":    "volatility",
        "description": "1.5× leveraged long VIX futures. Gains sharply on volatility spikes.",
    },
    {
        "symbol":      "VIXY",
        "label":       "ProShares VIX Short-Term Futures ETF",
        "category":    "volatility",
        "description": "1× long VIX futures. Less volatile than UVXY; cleaner hedge.",
    },
    # Broad-market inverse ETFs — profit when S&P/QQQ falls
    {
        "symbol":      "SQQQ",
        "label":       "ProShares UltraPro Short QQQ (3×)",
        "category":    "inverse_broad",
        "description": "3× inverse Nasdaq-100. Aggressive short on tech sell-off.",
    },
    {
        "symbol":      "SPXS",
        "label":       "Direxion Daily S&P 500 Bear 3× ETF",
        "category":    "inverse_broad",
        "description": "3× inverse S&P 500. Broad market protection.",
    },
    # Sector inverse — semiconductors typically lead vol-driven sell-offs
    {
        "symbol":      "SOXS",
        "label":       "Direxion Daily Semiconductor Bear 3×",
        "category":    "inverse_sector",
        "description": "3× inverse semiconductor index. High-beta sector hedge.",
    },
    # Safe havens — store of value / flight-to-quality during risk-off
    {
        "symbol":      "TLT",
        "label":       "iShares 20+ Year Treasury Bond ETF",
        "category":    "safe_haven",
        "description": "Long-duration treasuries. Classic risk-off flight to safety.",
    },
    {
        "symbol":      "GLD",
        "label":       "SPDR Gold Shares ETF",
        "category":    "safe_haven",
        "description": "Gold — inflation + uncertainty hedge. Holds well during crashes.",
    },
]


def _get_market_safety() -> dict:
    """
    Evaluate current market safety state and derive the system action.

    Safety state → system action mapping:
      NORMAL    — trading allowed, full position size
      CAUTION   — trading allowed, position size reduced to 50%
      SAFE_MODE — all new entries BLOCKED; existing positions monitored only

    When a volatility spike trigger is present, `suggested_hedges` is populated
    so the frontend can surface actionable hedge instruments immediately.
    """
    try:
        from analytics_engine.market_safety import MarketSafetyEngine

        engine = MarketSafetyEngine()
        result = engine.evaluate()

        # Derive human-readable system action from state
        if not result["trading_allowed"]:
            action = "All new entries BLOCKED. Monitor open positions only."
        elif result["size_multiplier"] < 1.0:
            action = (
                f"Trading allowed. Position size reduced to "
                f"{result['size_multiplier']:.0%}."
            )
        else:
            action = "Trading allowed at normal position size."

        result["system_action"] = action

        # Inject hedge suggestions whenever a volatility spike is flagged
        vol_spike = any(
            "volatility spike" in t.lower()
            for t in result.get("triggers", [])
        )
        if vol_spike:
            result["suggested_hedges"] = _VOLATILITY_SPIKE_HEDGES

        return result

    except Exception as e:
        logger.warning(f"Command center: market_safety failed: {e}")
        return {
            "safety_score":    50.0,
            "safety_state":    "CAUTION",
            "trading_allowed": True,
            "size_multiplier": 0.5,
            "system_action":   f"Safety engine unavailable — defaulting to CAUTION.",
            "components":      {
                "volatility":   50.0,
                "gap":          100.0,
                "portfolio":    100.0,
                "data_quality": 50.0,
            },
            "triggers":    [f"Safety engine error: {e}"],
            "computed_at": datetime.utcnow().isoformat() + "Z",
            "error":       str(e),
        }
