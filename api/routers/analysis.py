"""Bull/Bear analysis endpoint — groups leveraged pairs with their underlying."""

from fastapi import APIRouter
from sqlalchemy import select

from database.connection import get_session
from database.models import Instrument, Feature, CycleMetric
from config.settings import LEVERAGED_ETF_PAIRS

router = APIRouter()


def _get_latest_features(session, symbol: str) -> dict | None:
    inst = session.execute(
        select(Instrument).where(Instrument.symbol == symbol)
    ).scalar()
    if not inst:
        return None
    feat = session.execute(
        select(Feature)
        .where(Feature.instrument_id == inst.id)
        .order_by(Feature.date.desc())
        .limit(1)
    ).scalar()
    if not feat:
        return None
    return {
        "symbol": symbol,
        "trend_strength": feat.trend_strength,
        "momentum": feat.momentum,
        "volatility_percentile": feat.volatility_percentile,
        "volume_ratio": feat.volume_ratio,
        "cycle_phase": feat.cycle_phase,
        "macd_norm": feat.macd_norm,
        "bb_position": feat.bb_position,
        "price_to_ma50": feat.price_to_ma50,
        "return_5d": feat.return_5d,
        "return_20d": feat.return_20d,
    }


def _get_latest_cycle(session, symbol: str) -> dict | None:
    inst = session.execute(
        select(Instrument).where(Instrument.symbol == symbol)
    ).scalar()
    if not inst:
        return None
    cyc = session.execute(
        select(CycleMetric)
        .where(CycleMetric.instrument_id == inst.id)
        .order_by(CycleMetric.date.desc())
        .limit(1)
    ).scalar()
    if not cyc:
        return None
    return {
        "cycle_length": cyc.cycle_length,
        "cycle_phase": cyc.cycle_phase,
        "cycle_strength": cyc.cycle_strength,
    }


def _compute_score(features: dict | None) -> float | None:
    """Compute a composite bullish score 0-100 from features."""
    if not features:
        return None
    weights = {
        "trend_strength": 20,
        "momentum": 25,
        "macd_norm": 15,
        "bb_position": 10,
        "return_5d": 15,
        "return_20d": 15,
    }
    score = 0.0
    total_weight = 0.0
    for key, w in weights.items():
        val = features.get(key)
        if val is None:
            continue
        # Normalize each feature to 0-1 range
        if key == "momentum":
            # Already 0-1 (RSI/100)
            norm = val
        elif key == "bb_position":
            # 0-1 range already
            norm = max(0, min(1, val))
        elif key in ("trend_strength", "price_to_ma50"):
            # Typically -0.5 to 0.5, map to 0-1
            norm = max(0, min(1, (val + 0.5)))
        elif key in ("return_5d", "return_20d"):
            # Typically -0.2 to 0.2, map to 0-1
            norm = max(0, min(1, (val + 0.2) / 0.4))
        elif key == "macd_norm":
            # Typically -1 to 1, map to 0-1
            norm = max(0, min(1, (val + 1) / 2))
        else:
            norm = max(0, min(1, val))
        score += norm * w
        total_weight += w

    if total_weight == 0:
        return None
    return round((score / total_weight) * 100, 1)


PHASE_LABELS = {0: "Accumulation", 1: "Markup", 2: "Distribution", 3: "Markdown"}


def _phase_label(cycle_data: dict | None) -> str:
    if not cycle_data or cycle_data.get("cycle_phase") is None:
        return "Unknown"
    phase = cycle_data["cycle_phase"]
    if phase < 0.25:
        return "Accumulation"
    elif phase < 0.50:
        return "Markup"
    elif phase < 0.75:
        return "Distribution"
    else:
        return "Markdown"


def _verdict(underlying_score, bull_score, bear_score) -> dict:
    """Determine bull/bear verdict from scores."""
    scores = {"underlying": underlying_score, "bull": bull_score, "bear": bear_score}
    valid = {k: v for k, v in scores.items() if v is not None}

    if not valid:
        return {"verdict": "NO DATA", "color": "gray", "reasoning": "Insufficient data for analysis."}

    avg = sum(valid.values()) / len(valid)

    # Check bear ETF — if bear is strong, market is bearish
    bear_strong = bear_score is not None and bear_score > 60
    bull_strong = bull_score is not None and bull_score > 60
    underlying_bullish = underlying_score is not None and underlying_score > 55

    if bull_strong and underlying_bullish and not bear_strong:
        return {
            "verdict": "BULLISH",
            "color": "#22c55e",
            "reasoning": f"Bull ETF score {bull_score:.0f}, underlying {underlying_score:.0f} — momentum and trend favor longs.",
        }
    elif bear_strong and not bull_strong:
        return {
            "verdict": "BEARISH",
            "color": "#ef4444",
            "reasoning": f"Bear ETF score {bear_score:.0f} outperforming — defensive positioning recommended.",
        }
    elif bull_strong and bear_strong:
        return {
            "verdict": "VOLATILE",
            "color": "#f59e0b",
            "reasoning": f"Both bull ({bull_score:.0f}) and bear ({bear_score:.0f}) showing strength — high volatility regime.",
        }
    elif avg > 55:
        return {
            "verdict": "LEAN BULL",
            "color": "#86efac",
            "reasoning": f"Avg score {avg:.0f} — slight bullish tilt, build positions cautiously (10/10 strategy).",
        }
    elif avg < 45:
        return {
            "verdict": "LEAN BEAR",
            "color": "#fca5a5",
            "reasoning": f"Avg score {avg:.0f} — slight bearish tilt, reduce exposure or hedge.",
        }
    else:
        return {
            "verdict": "NEUTRAL",
            "color": "#9ca3af",
            "reasoning": f"Avg score {avg:.0f} — no clear directional edge. Wait for setup.",
        }


@router.get("/bull-bear")
def get_bull_bear_analysis():
    """Bull/Bear analysis for all leveraged pairs with underlying."""
    session = get_session()
    try:
        result = []
        seen_desc = set()
        for bull, bear, desc, underlying in LEVERAGED_ETF_PAIRS:
            # Deduplicate entries with identical descriptions (same underlying/pair)
            if desc in seen_desc:
                continue
            seen_desc.add(desc)

            bull_feat = _get_latest_features(session, bull)
            bear_feat = _get_latest_features(session, bear) if bear else None
            underlying_feat = _get_latest_features(session, underlying) if underlying else None

            bull_cycle = _get_latest_cycle(session, bull)
            bear_cycle = _get_latest_cycle(session, bear) if bear else None
            underlying_cycle = _get_latest_cycle(session, underlying) if underlying else None

            bull_score = _compute_score(bull_feat)
            bear_score = _compute_score(bear_feat)
            underlying_score = _compute_score(underlying_feat)

            v = _verdict(underlying_score, bull_score, bear_score)

            result.append({
                "description": desc,
                "underlying": {
                    "symbol": underlying,
                    "score": underlying_score,
                    "phase": _phase_label(underlying_cycle),
                    "features": underlying_feat,
                } if underlying else None,
                "bull": {
                    "symbol": bull,
                    "score": bull_score,
                    "phase": _phase_label(bull_cycle),
                    "features": bull_feat,
                },
                "bear": {
                    "symbol": bear,
                    "score": bear_score,
                    "phase": _phase_label(bear_cycle),
                    "features": bear_feat,
                } if bear else None,
                "verdict": v["verdict"],
                "verdict_color": v["color"],
                "reasoning": v["reasoning"],
            })

        return result
    finally:
        session.close()
