"""
education_engine/signal_explainer.py
────────────────────────────────────────────────────────────────────────────────
Signal Explanation Engine.

Converts raw signal data into human-readable explanations at two levels:
  BEGINNER  → simple language, emojis, clear action steps
  ADVANCED  → full metrics, raw values, technical terminology

Output per signal:
  {
    "summary":            str,           # 1-sentence summary
    "detailed_reasoning": list[str],     # bullet points explaining why
    "trade_idea":         str,           # specific action to take
    "confidence":         str,           # high | medium | low
    "mode":               str,           # beginner | advanced
    "metrics":            dict,          # raw values (advanced only)
    "key_levels":         dict,          # stop, target, entry, gamma flip
    "warnings":           list[str],     # risk factors to watch
  }
"""
from __future__ import annotations

from typing import Any


class SignalExplainer:
    """Generate natural language explanations for trading signals."""

    def __init__(self, mode: str = "beginner"):
        """
        Args:
            mode: "beginner" (simplified) or "advanced" (full technical detail)
        """
        assert mode in ("beginner", "advanced"), "mode must be 'beginner' or 'advanced'"
        self.mode = mode

    def explain(self, signal: dict[str, Any]) -> dict[str, Any]:
        """
        Generate explanation for a single signal.

        Args:
            signal: Output dict from ScoringEngine.score()

        Returns:
            Explanation dict with summary, reasoning, and trade idea.
        """
        symbol      = signal.get("symbol", "")
        score       = float(signal.get("score", 0))
        sig_type    = signal.get("signal_type", "long")
        confidence  = signal.get("confidence", "low")
        factors     = signal.get("factors", {})
        gex_ctx     = signal.get("gex_context", {})
        raw_exp     = signal.get("explanation", [])

        reasoning   = []
        warnings    = []
        key_levels  = {}

        # ── GEX context explanation ───────────────────────────────────────
        scenario = gex_ctx.get("scenario", "")
        regime   = gex_ctx.get("regime", "NEUTRAL")
        flip     = gex_ctx.get("gamma_flip", 0)
        call_w   = gex_ctx.get("call_wall", 0)
        put_w    = gex_ctx.get("put_wall", 0)
        price    = gex_ctx.get("price", 0)

        if self.mode == "beginner":
            reasoning.extend(self._beginner_gex(scenario, regime, flip, price, sig_type))
        else:
            reasoning.extend(self._advanced_gex(gex_ctx, scenario))

        # ── Factor explanations ───────────────────────────────────────────
        cycle_score  = factors.get("cycle_alignment", 0.5)
        mom_score    = factors.get("momentum_strength", 0.5)
        vol_score    = factors.get("volatility_state", 0.5)
        struct_score = factors.get("structure_quality", 0.5)
        cat_score    = factors.get("catalyst_score", 0.05)

        if self.mode == "beginner":
            reasoning.extend(self._beginner_factors(cycle_score, mom_score, vol_score, cat_score))
        else:
            reasoning.extend(self._advanced_factors(factors))

        # ── Key levels ────────────────────────────────────────────────────
        if flip:
            key_levels["gamma_flip"] = flip
        if call_w:
            key_levels["call_wall"]  = call_w
        if put_w:
            key_levels["put_wall"]   = put_w

        # ── Warnings ──────────────────────────────────────────────────────
        if regime == "NEGATIVE" and scenario not in ("gamma_squeeze", "breakdown"):
            warnings.append("Negative GEX regime — moves can reverse sharply. Use stop loss.")
        if vol_score < 0.4:
            warnings.append("Low volatility — expect smaller moves. Reduce position size.")
        if cycle_score < 0.4:
            warnings.append("Cycle timing is unfavorable. Consider waiting for better alignment.")
        if confidence == "low":
            warnings.append("Low confidence signal — consider paper trading this setup first.")

        # ── Summary ───────────────────────────────────────────────────────
        summary = self._build_summary(symbol, sig_type, score, confidence, scenario)

        # ── Trade idea ────────────────────────────────────────────────────
        trade_idea = self._build_trade_idea(sig_type, scenario, confidence, self.mode)

        result = {
            "symbol":             symbol,
            "mode":               self.mode,
            "summary":            summary,
            "detailed_reasoning": reasoning,
            "trade_idea":         trade_idea,
            "confidence":         confidence,
            "key_levels":         key_levels,
            "warnings":           warnings,
        }

        # Advanced mode appends raw metrics
        if self.mode == "advanced":
            result["metrics"] = {
                "score":            round(score, 4),
                "cycle_alignment":  round(cycle_score, 3),
                "gex_signal":       round(factors.get("gex_signal", 0), 3),
                "momentum_strength": round(mom_score, 3),
                "volatility_state": round(vol_score, 3),
                "structure_quality": round(struct_score, 3),
                "catalyst_score":   round(cat_score, 3),
                "gex_regime":       regime,
                "gex_scenario":     scenario,
                "flip_dist_pct":    gex_ctx.get("flip_dist_pct", 0),
            }

        return result

    # ── Explanation builders ──────────────────────────────────────────────────

    def _beginner_gex(self, scenario, regime, flip, price, sig_type) -> list[str]:
        lines = []
        if scenario == "gamma_squeeze":
            lines.append("🚀 Market makers are being forced to BUY as price rises — this amplifies the move upward")
            lines.append(f"   Price crossed the gamma flip level (${flip:.2f}) — this is a key structural shift")
        elif scenario == "breakdown":
            lines.append("💥 Market makers are being forced to SELL as price falls — this amplifies the drop")
            lines.append(f"   Price fell below the put wall (support level) — selling pressure accelerating")
        elif scenario == "mean_reversion_call_wall":
            lines.append(f"📌 Price is near the call wall (${flip:.2f}) — a strong resistance zone")
            lines.append("   Market makers will sell here to hedge, pushing price back down")
        elif scenario == "near_gamma_flip":
            lines.append(f"⚡ Price is very close to the gamma flip level (${flip:.2f})")
            lines.append("   A break through this level could trigger a fast directional move")
        elif regime == "NEGATIVE":
            lines.append("💡 The market is in an unstable regime — moves tend to be faster than normal")
        elif regime == "POSITIVE":
            lines.append("📌 The market is in a stable regime — price tends to stay in a tight range")
        return lines

    def _advanced_gex(self, gex_ctx: dict, scenario: str) -> list[str]:
        lines = [
            f"GEX Regime: {gex_ctx.get('regime', 'N/A')}  |  Scenario: {scenario}",
            f"Gamma Flip: ${gex_ctx.get('gamma_flip', 0):.2f}  |  "
            f"Call Wall: ${gex_ctx.get('call_wall', 0):.2f}  |  "
            f"Put Wall: ${gex_ctx.get('put_wall', 0):.2f}",
            f"Distance to flip: {gex_ctx.get('flip_dist_pct', 0):.2f}%  |  "
            f"Dealer bias: {gex_ctx.get('dealer_bias', 0):+d}",
        ]
        return lines

    def _beginner_factors(self, cycle, momentum, volatility, catalyst) -> list[str]:
        lines = []
        if cycle > 0.70:
            lines.append("✅ Timing is good — the market cycle aligns with this trade direction")
        elif cycle < 0.40:
            lines.append("⏰ The timing cycle is not ideal — consider waiting for better alignment")

        if momentum > 0.70:
            lines.append("💪 Strong price momentum with high volume — the move has conviction")
        elif momentum > 0.40:
            lines.append("📊 Moderate momentum — move is developing but not yet explosive")
        else:
            lines.append("😴 Weak momentum — this could be a slow or choppy trade")

        if catalyst > 0.50:
            lines.append("📰 A significant news event is driving this move (earnings / upgrade / announcement)")

        return lines

    def _advanced_factors(self, factors: dict) -> list[str]:
        return [
            f"Cycle Alignment:   {factors.get('cycle_alignment', 0):.3f} "
            f"(weight 25%) — {'✓ favorable' if factors.get('cycle_alignment',0) > 0.6 else '✗ weak'}",
            f"GEX Signal:        {factors.get('gex_signal', 0):.3f} "
            f"(weight 25%) — dealer positioning signal",
            f"Momentum Strength: {factors.get('momentum_strength', 0):.3f} "
            f"(weight 20%) — price × volume composite",
            f"Volatility State:  {factors.get('volatility_state', 0):.3f} "
            f"(weight 15%) — VIX regime suitability",
            f"Structure Quality: {factors.get('structure_quality', 0):.3f} "
            f"(weight 10%) — pattern confirmation",
            f"Catalyst Score:    {factors.get('catalyst_score', 0):.3f} "
            f"(weight 5%)  — news/fundamental catalyst",
        ]

    def _build_summary(self, symbol, sig_type, score, confidence, scenario) -> str:
        conf_map = {"high": "HIGH conviction", "medium": "Moderate", "low": "Weak"}
        c = conf_map.get(confidence, "")
        pct = int(score * 100)
        if scenario == "gamma_squeeze":
            return f"{c} LONG signal on {symbol} — gamma squeeze setup (score {pct}/100). Explosive upside potential."
        elif scenario == "breakdown":
            return f"{c} SHORT signal on {symbol} — breakdown below put wall (score {pct}/100). Downside acceleration expected."
        elif scenario == "mean_reversion_call_wall":
            return f"{c} SHORT signal on {symbol} — pinned at call wall, expect fade (score {pct}/100)."
        elif sig_type == "long":
            return f"{c} LONG signal on {symbol} (score {pct}/100). Multiple factors align bullishly."
        else:
            return f"{c} SHORT signal on {symbol} (score {pct}/100). Multiple factors align bearishly."

    def _build_trade_idea(self, sig_type, scenario, confidence, mode) -> str:
        if mode == "beginner":
            if scenario == "gamma_squeeze":
                return ("BUY the stock or a call option. "
                        "Set stop loss 1.5–2 ATR below entry. "
                        "Hold for a fast, explosive move. Take partial profits quickly.")
            elif scenario == "breakdown":
                return ("Consider buying SQQQ/SPXU (inverse ETFs) or put options. "
                        "Set stop above the put wall level. Target is 1.5–2× your risk.")
            elif confidence == "high" and sig_type == "long":
                return ("BUY with full position size. "
                        "Stop below nearest support or put wall. Target at resistance/call wall.")
            elif confidence == "medium" and sig_type == "long":
                return ("BUY with half position size. "
                        "Wait for a confirming green candle before entry. "
                        "Stop below support.")
            else:
                return ("Paper trade this setup first. Confidence is low — observe the move before committing capital.")
        else:
            direction = "long" if sig_type == "long" else "short"
            return (f"Enter {direction} on next candle close. "
                    f"Stop = 1.5 × ATR {'below' if sig_type == 'long' else 'above'} entry. "
                    f"Target = 2× risk. Reduce to 50% at 1× risk. "
                    f"{'Max 25% of account.' if confidence == 'high' else 'Cap at 12.5% of account.'}")

    def explain_batch(self, signals: list[dict]) -> list[dict]:
        """Explain multiple signals."""
        return [self.explain(s) for s in signals]

    def set_mode(self, mode: str) -> None:
        """Switch between beginner and advanced mode."""
        assert mode in ("beginner", "advanced")
        self.mode = mode


# ── Example ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mock_signal = {
        "symbol": "SPY",
        "score":  0.87,
        "signal_type": "long",
        "confidence": "high",
        "factors": {
            "cycle_alignment": 0.78,
            "gex_signal": 0.95,
            "momentum_strength": 0.82,
            "volatility_state": 0.75,
            "structure_quality": 0.65,
            "catalyst_score": 0.20,
        },
        "gex_context": {
            "regime": "NEGATIVE",
            "scenario": "gamma_squeeze",
            "gamma_flip": 498,
            "call_wall": 510,
            "put_wall": 490,
            "price": 505,
            "flip_dist_pct": 1.4,
            "dealer_bias": 1,
        },
    }

    for mode in ("beginner", "advanced"):
        print(f"\n{'='*60}")
        print(f"MODE: {mode.upper()}")
        print(f"{'='*60}")
        explainer = SignalExplainer(mode=mode)
        result    = explainer.explain(mock_signal)
        print(f"Summary: {result['summary']}")
        print("\nReasoning:")
        for r in result["detailed_reasoning"]:
            print(f"  {r}")
        print(f"\nTrade Idea: {result['trade_idea']}")
        if result["warnings"]:
            print("\nWarnings:")
            for w in result["warnings"]:
                print(f"  ⚠️  {w}")
        if "metrics" in result:
            print("\nRaw Metrics:")
            for k, v in result["metrics"].items():
                print(f"  {k}: {v}")
