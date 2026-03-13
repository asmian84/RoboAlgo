"""
Trade Coach Engine
Builds concise, deterministic natural-language explanations for trade signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class TradeSignalContext:
    symbol: str
    pattern: str
    breakout: float
    target: float
    risk: float
    confidence: float
    liquidity_target_above: Optional[float] = None
    momentum_delta: Optional[float] = None
    compression_detected: Optional[bool] = None


class TradeCoachEngine:
    """
    Converts structured signal fields into trader-friendly plain language.
    Confidence is treated as a 0-100 score.
    """

    def explain_signal(self, ctx: TradeSignalContext) -> dict:
        confidence = max(0.0, min(100.0, float(ctx.confidence)))
        risk = max(0.0, float(ctx.risk))

        bullets: list[str] = []
        if ctx.compression_detected:
            bullets.append("Breakout compression detected.")
        if ctx.liquidity_target_above is not None and ctx.liquidity_target_above > ctx.breakout:
            bullets.append("Liquidity target above breakout level.")
        if ctx.momentum_delta is not None:
            bullets.append("Momentum increasing." if ctx.momentum_delta > 0 else "Momentum weakening.")
        if not bullets:
            bullets.append("Signal generated from structure and momentum alignment.")

        return {
            "symbol": ctx.symbol.upper(),
            "pattern": ctx.pattern,
            "breakout": round(float(ctx.breakout), 4),
            "target": round(float(ctx.target), 4),
            "risk": round(risk, 4),
            "confidence": round(confidence, 2),
            "explanation": " ".join(bullets),
            "evidence": bullets,
        }

